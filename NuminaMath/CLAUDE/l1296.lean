import Mathlib

namespace sum_at_13th_position_l1296_129654

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℕ
  is_permutation : Function.Bijective vertices

/-- The sum of numbers in a specific position across all rotations of a regular polygon -/
def sum_at_position (p : RegularPolygon 100) (pos : ℕ) : ℕ :=
  (Finset.range 100).sum (λ i => p.vertices ((i + pos - 1) % 100 : Fin 100))

/-- The main theorem -/
theorem sum_at_13th_position (p : RegularPolygon 100) 
  (h_vertices : ∀ i : Fin 100, p.vertices i = i.val + 1) : 
  sum_at_position p 13 = 10100 := by
  sorry

end sum_at_13th_position_l1296_129654


namespace shaded_triangle_probability_l1296_129653

theorem shaded_triangle_probability 
  (total_triangles : ℕ) 
  (shaded_triangles : ℕ) 
  (h1 : total_triangles > 4) 
  (h2 : total_triangles = 10) 
  (h3 : shaded_triangles = 4) : 
  (shaded_triangles : ℚ) / total_triangles = 2 / 5 := by
sorry

end shaded_triangle_probability_l1296_129653


namespace division_problem_l1296_129618

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3)
  (h2 : b / c = 1 / 2)
  : c / a = 2 / 3 := by
  sorry

end division_problem_l1296_129618


namespace triangular_pyramid_no_circular_cross_section_l1296_129604

-- Define the types of solids
inductive Solid
  | Cone
  | Cylinder
  | Sphere
  | TriangularPyramid

-- Define a predicate for having a circular cross-section
def has_circular_cross_section (s : Solid) : Prop :=
  match s with
  | Solid.Cone => True
  | Solid.Cylinder => True
  | Solid.Sphere => True
  | Solid.TriangularPyramid => False

-- Theorem statement
theorem triangular_pyramid_no_circular_cross_section :
  ∀ s : Solid, ¬(has_circular_cross_section s) ↔ s = Solid.TriangularPyramid :=
by sorry


end triangular_pyramid_no_circular_cross_section_l1296_129604


namespace algebraic_identities_l1296_129641

theorem algebraic_identities (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hab : a ≠ b) :
  (a / (a - b) + b / (b - a) = 1) ∧
  (a^2 / (b^2 * c) * (-b * c^2 / (2 * a)) / (a / b) = -c) :=
sorry

end algebraic_identities_l1296_129641


namespace problem_1_problem_2_l1296_129658

-- Problem 1
theorem problem_1 : Real.sqrt 12 + (-1/3)⁻¹ + (-2)^2 = 2 * Real.sqrt 3 + 1 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (2*a / (a^2 - 4)) / (1 + (a - 2) / (a + 2)) = 1 / (a - 2) := by sorry

end problem_1_problem_2_l1296_129658


namespace quadratic_increasing_condition_l1296_129692

/-- The function f(x) = ax^2 - 2x + 1 is increasing on [1, 2] iff a > 0 and 1/a < 1 -/
theorem quadratic_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, Monotone (fun x => a * x^2 - 2 * x + 1)) ↔ (a > 0 ∧ 1 / a < 1) := by
  sorry


end quadratic_increasing_condition_l1296_129692


namespace third_row_sum_l1296_129667

def is_valid_grid (grid : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  ∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 9 ∧
  ∀ i' j', (i ≠ i' ∨ j ≠ j') → grid i j ≠ grid i' j'

theorem third_row_sum (grid : Matrix (Fin 3) (Fin 3) ℕ) 
  (h_valid : is_valid_grid grid)
  (h_row1 : (grid 0 0) * (grid 0 1) * (grid 0 2) = 60)
  (h_row2 : (grid 1 0) * (grid 1 1) * (grid 1 2) = 96) :
  (grid 2 0) + (grid 2 1) + (grid 2 2) = 17 := by
  sorry

end third_row_sum_l1296_129667


namespace hot_dog_cost_l1296_129679

/-- The cost of a hot dog given the conditions of the concession stand problem -/
theorem hot_dog_cost (soda_cost : ℝ) (total_revenue : ℝ) (total_items : ℕ) (hot_dogs_sold : ℕ) :
  soda_cost = 0.50 →
  total_revenue = 78.50 →
  total_items = 87 →
  hot_dogs_sold = 35 →
  ∃ (hot_dog_cost : ℝ), 
    hot_dog_cost * hot_dogs_sold + soda_cost * (total_items - hot_dogs_sold) = total_revenue ∧
    hot_dog_cost = 1.50 := by
  sorry


end hot_dog_cost_l1296_129679


namespace rectangle_dimensions_l1296_129638

theorem rectangle_dimensions (x : ℝ) : 
  (x - 3) * (3 * x + 4) = 12 * x - 9 → x = (17 + 5 * Real.sqrt 13) / 6 := by
sorry

end rectangle_dimensions_l1296_129638


namespace distance_between_cities_l1296_129621

/-- The distance between City A and City B -/
def distance : ℝ := 427.5

/-- The time for the first trip in hours -/
def time_first_trip : ℝ := 6

/-- The time for the return trip in hours -/
def time_return_trip : ℝ := 4.5

/-- The time saved on each trip in hours -/
def time_saved_per_trip : ℝ := 0.5

/-- The speed of the round trip if time was saved, in miles per hour -/
def speed_with_time_saved : ℝ := 90

theorem distance_between_cities :
  distance = 427.5 ∧
  (2 * distance) / (time_first_trip + time_return_trip - 2 * time_saved_per_trip) = speed_with_time_saved :=
by sorry

end distance_between_cities_l1296_129621


namespace irrational_sqrt_three_rational_others_l1296_129619

theorem irrational_sqrt_three_rational_others : 
  (Irrational (Real.sqrt 3)) ∧ 
  (¬ Irrational (-8 : ℝ)) ∧ 
  (¬ Irrational (0.3070809 : ℝ)) ∧ 
  (¬ Irrational (22 / 7 : ℝ)) := by
  sorry

end irrational_sqrt_three_rational_others_l1296_129619


namespace unique_solution_l1296_129698

/-- A function satisfying the given conditions -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = -f (-x)) ∧
  (∀ x, f (x + 1) = f x + 1) ∧
  (∀ x ≠ 0, f (1 / x) = (1 / x^2) * f x)

/-- Theorem stating that the only function satisfying the conditions is f(x) = x -/
theorem unique_solution (f : ℝ → ℝ) (h : SatisfiesConditions f) :
  ∀ x, f x = x := by
  sorry

end unique_solution_l1296_129698


namespace min_value_not_five_max_value_half_x_gt_y_iff_x_over_c_gt_y_over_c_min_value_eight_l1296_129655

-- Statement 1
theorem min_value_not_five : 
  ¬ (∀ x : ℝ, x + 4 / (x - 1) ≥ 5) :=
sorry

-- Statement 2
theorem max_value_half : 
  (∀ x : ℝ, x * Real.sqrt (1 - x^2) ≤ 1/2) ∧ 
  (∃ x : ℝ, x * Real.sqrt (1 - x^2) = 1/2) :=
sorry

-- Statement 3
theorem x_gt_y_iff_x_over_c_gt_y_over_c :
  ∀ x y c : ℝ, c ≠ 0 → (x > y ↔ x / c^2 > y / c^2) :=
sorry

-- Statement 4
theorem min_value_eight :
  ∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 1 →
  (∀ a b : ℝ, a > 0 → b > 0 → a + 2*b = 1 → 2/a + 1/b ≥ 2/x + 1/y) →
  2/x + 1/y = 8 :=
sorry

end min_value_not_five_max_value_half_x_gt_y_iff_x_over_c_gt_y_over_c_min_value_eight_l1296_129655


namespace sugar_solution_concentration_increases_l1296_129632

theorem sugar_solution_concentration_increases 
  (a b m : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : m > 0) : 
  (b + m) / (a + m) > b / a := by
  sorry

end sugar_solution_concentration_increases_l1296_129632


namespace most_probable_hits_l1296_129676

-- Define the parameters
def n : ℕ := 5
def p : ℝ := 0.6

-- Define the binomial probability mass function
def binomialPMF (k : ℕ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

-- Theorem statement
theorem most_probable_hits :
  ∃ (k : ℕ), k ≤ n ∧ 
  (∀ (j : ℕ), j ≤ n → binomialPMF j ≤ binomialPMF k) ∧
  k = 3 := by
  sorry

end most_probable_hits_l1296_129676


namespace alberts_expression_l1296_129639

theorem alberts_expression (p q r s t u : ℚ) : 
  p = 2 ∧ q = 3 ∧ r = 4 ∧ s = 5 ∧ t = 6 →
  p - (q - (r - (s * (t + u)))) = p - q - r - s * t + u →
  u = 4/3 := by sorry

end alberts_expression_l1296_129639


namespace product_divisible_by_twelve_l1296_129617

theorem product_divisible_by_twelve (a b c d : ℤ) 
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) : 
  12 ∣ ((a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d)) := by
sorry

end product_divisible_by_twelve_l1296_129617


namespace max_k_value_l1296_129691

theorem max_k_value (x y k : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_k : k > 0)
  (h_eq : 3 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 7) / 2 :=
sorry

end max_k_value_l1296_129691


namespace gcd_lcm_problem_l1296_129649

theorem gcd_lcm_problem (a b : ℕ+) 
  (h1 : Nat.gcd a b = 24)
  (h2 : Nat.lcm a b = 432)
  (h3 : a = 144) :
  b = 72 := by
sorry

end gcd_lcm_problem_l1296_129649


namespace charity_ticket_sales_l1296_129668

theorem charity_ticket_sales (total_tickets : ℕ) (total_revenue : ℕ) 
  (donation : ℕ) (h_total_tickets : total_tickets = 200) 
  (h_total_revenue : total_revenue = 3200) (h_donation : donation = 200) :
  ∃ (full_price : ℕ) (half_price : ℕ) (price : ℕ),
    full_price + half_price = total_tickets ∧
    full_price * price + half_price * (price / 2) + donation = total_revenue ∧
    full_price * price = 1000 := by
  sorry

end charity_ticket_sales_l1296_129668


namespace necessary_but_not_sufficient_l1296_129685

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number (1-m²) + (1+m)i where m is a real number -/
def complexNumber (m : ℝ) : ℂ :=
  ⟨1 - m^2, 1 + m⟩

theorem necessary_but_not_sufficient :
  (∀ m : ℝ, IsPurelyImaginary (complexNumber m) → m = 1 ∨ m = -1) ∧
  (∃ m : ℝ, (m = 1 ∨ m = -1) ∧ ¬IsPurelyImaginary (complexNumber m)) :=
by sorry

end necessary_but_not_sufficient_l1296_129685


namespace sum_of_fractions_l1296_129622

theorem sum_of_fractions : (3 : ℚ) / 5 + 5 / 11 + 1 / 3 = 229 / 165 := by sorry

end sum_of_fractions_l1296_129622


namespace union_of_A_and_complement_of_B_l1296_129666

def U : Finset ℕ := {1,2,3,4,5,6,7}
def A : Finset ℕ := {1,3,5}
def B : Finset ℕ := {2,3,6}

theorem union_of_A_and_complement_of_B :
  A ∪ (U \ B) = {1,3,4,5,7} := by sorry

end union_of_A_and_complement_of_B_l1296_129666


namespace area_relationship_l1296_129642

/-- Triangle with sides 13, 14, and 15 inscribed in a circle -/
structure InscribedTriangle where
  -- Define the sides of the triangle
  a : ℝ := 13
  b : ℝ := 14
  c : ℝ := 15
  -- Define the areas of non-triangular regions
  A : ℝ
  B : ℝ
  C : ℝ
  -- C is the largest area
  hC_largest : C ≥ A ∧ C ≥ B

/-- The relationship between areas A, B, C, and the triangle area -/
theorem area_relationship (t : InscribedTriangle) : t.A + t.B + 84 = t.C := by
  sorry

end area_relationship_l1296_129642


namespace dice_arithmetic_progression_probability_l1296_129669

def num_dice : ℕ := 4
def faces_per_die : ℕ := 6

def total_outcomes : ℕ := faces_per_die ^ num_dice

def valid_progressions : List (List ℕ) := [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]

def favorable_outcomes : ℕ := valid_progressions.length * (num_dice.factorial)

theorem dice_arithmetic_progression_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 18 := by sorry

end dice_arithmetic_progression_probability_l1296_129669


namespace base12_addition_correct_l1296_129636

/-- Converts a base 12 number represented as a list of digits to its decimal (base 10) equivalent -/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 12 + d) 0

/-- Converts a decimal (base 10) number to its base 12 representation as a list of digits -/
def decimalToBase12 (n : Nat) : List Nat :=
  if n < 12 then [n]
  else (n % 12) :: decimalToBase12 (n / 12)

/-- Represents a number in base 12 -/
structure Base12 where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 12

/-- Addition of two Base12 numbers -/
def add (a b : Base12) : Base12 :=
  let sum := base12ToDecimal a.digits + base12ToDecimal b.digits
  ⟨decimalToBase12 sum, sorry⟩

theorem base12_addition_correct :
  let a : Base12 := ⟨[3, 12, 5, 10], sorry⟩  -- 3C5A₁₂
  let b : Base12 := ⟨[4, 10, 3, 11], sorry⟩  -- 4A3B₁₂
  let result : Base12 := ⟨[8, 10, 9, 8], sorry⟩  -- 8A98₁₂
  add a b = result :=
sorry

end base12_addition_correct_l1296_129636


namespace max_amount_is_7550_l1296_129602

-- Define the total value of chips bought
def total_value : ℕ := 10000

-- Define the chip denominations
def chip_50_value : ℕ := 50
def chip_200_value : ℕ := 200

-- Define the total number of chips lost
def total_chips_lost : ℕ := 30

-- Define the relationship between lost chips
axiom lost_chips_relation : ∃ (x y : ℕ), x = 3 * y ∧ x + y = total_chips_lost

-- Define the function to calculate the maximum amount received back
def max_amount_received : ℕ := 
  total_value - (7 * chip_200_value + 21 * chip_50_value)

-- Theorem to prove
theorem max_amount_is_7550 : max_amount_received = 7550 := by
  sorry

end max_amount_is_7550_l1296_129602


namespace city_of_pythagoras_schools_l1296_129614

/-- Represents a student in the math contest -/
structure Student where
  school : Nat
  rank : Nat

/-- The math contest setup -/
structure MathContest where
  numSchools : Nat
  students : Finset Student

theorem city_of_pythagoras_schools (contest : MathContest) : contest.numSchools = 40 :=
  by
  have h1 : ∀ s : Student, s ∈ contest.students → s.rank ≤ 4 * contest.numSchools :=
    sorry
  have h2 : ∀ s1 s2 : Student, s1 ∈ contest.students → s2 ∈ contest.students → s1 ≠ s2 → s1.rank ≠ s2.rank :=
    sorry
  have h3 : ∃ andrea : Student, andrea ∈ contest.students ∧
    andrea.rank = (2 * contest.numSchools) ∨ andrea.rank = (2 * contest.numSchools + 1) :=
    sorry
  have h4 : ∃ beth : Student, beth ∈ contest.students ∧ beth.rank = 41 :=
    sorry
  have h5 : ∃ carla : Student, carla ∈ contest.students ∧ carla.rank = 82 :=
    sorry
  have h6 : ∃ andrea beth carla : Student, 
    andrea ∈ contest.students ∧ beth ∈ contest.students ∧ carla ∈ contest.students ∧
    andrea.school = beth.school ∧ andrea.school = carla.school ∧
    andrea.rank < beth.rank ∧ andrea.rank < carla.rank :=
    sorry
  sorry


end city_of_pythagoras_schools_l1296_129614


namespace triangle_perimeter_l1296_129645

/-- Given a right-angled triangle PQR with the right angle at R, PR = 8, PQ = 15, and QR = 6,
    prove that the perimeter of the triangle is 24. -/
theorem triangle_perimeter (P Q R : ℝ × ℝ) : 
  (R.2 - P.2) * (Q.1 - P.1) = (Q.2 - P.2) * (R.1 - P.1) →  -- Right angle at R
  dist P R = 8 →
  dist P Q = 15 →
  dist Q R = 6 →
  dist P R + dist P Q + dist Q R = 24 := by
  sorry

end triangle_perimeter_l1296_129645


namespace green_disks_count_l1296_129694

theorem green_disks_count (total : ℕ) (red green blue : ℕ) : 
  total = 14 →
  red = 2 * green →
  blue = green / 2 →
  total = red + green + blue →
  green = 4 := by
  sorry

end green_disks_count_l1296_129694


namespace cattle_truck_capacity_l1296_129611

/-- Calculates the capacity of a cattle transport truck given the total number of cattle,
    distance to safety, truck speed, and total transport time. -/
theorem cattle_truck_capacity
  (total_cattle : ℕ)
  (distance : ℝ)
  (speed : ℝ)
  (total_time : ℝ)
  (h_total_cattle : total_cattle = 400)
  (h_distance : distance = 60)
  (h_speed : speed = 60)
  (h_total_time : total_time = 40)
  : ℕ :=
by
  sorry

#check cattle_truck_capacity

end cattle_truck_capacity_l1296_129611


namespace sams_new_books_l1296_129678

theorem sams_new_books (adventure_books : ℕ) (mystery_books : ℕ) (used_books : ℕ) 
  (h1 : adventure_books = 13)
  (h2 : mystery_books = 17)
  (h3 : used_books = 15) : 
  adventure_books + mystery_books - used_books = 15 := by
  sorry

end sams_new_books_l1296_129678


namespace star_expression_l1296_129601

/-- The star operation on real numbers -/
def star (a b : ℝ) : ℝ := a^2 + b^2 - a*b

/-- Theorem stating the result of (x+2y) ⋆ (y+3x) -/
theorem star_expression (x y : ℝ) : star (x + 2*y) (y + 3*x) = 7*x^2 + 3*y^2 + 3*x*y := by
  sorry

end star_expression_l1296_129601


namespace smallest_double_square_triple_cube_l1296_129610

theorem smallest_double_square_triple_cube : ∃! k : ℕ, 
  (∃ m : ℕ, k = 2 * m^2) ∧ 
  (∃ n : ℕ, k = 3 * n^3) ∧ 
  (∀ j : ℕ, j < k → ¬(∃ x : ℕ, j = 2 * x^2) ∨ ¬(∃ y : ℕ, j = 3 * y^3)) ∧
  k = 648 := by
sorry

end smallest_double_square_triple_cube_l1296_129610


namespace river_speed_is_two_l1296_129670

/-- The speed of the river that satisfies the given conditions -/
def river_speed (mans_speed : ℝ) (distance : ℝ) (total_time : ℝ) : ℝ :=
  2

/-- Theorem stating that the river speed is 2 kmph given the conditions -/
theorem river_speed_is_two :
  let mans_speed : ℝ := 4
  let distance : ℝ := 2.25
  let total_time : ℝ := 1.5
  river_speed mans_speed distance total_time = 2 := by
  sorry

#check river_speed_is_two

end river_speed_is_two_l1296_129670


namespace usual_time_to_catch_bus_l1296_129626

/-- Given a person who misses the bus by 4 minutes when walking at 4/5 of their usual speed,
    their usual time to catch the bus is 16 minutes. -/
theorem usual_time_to_catch_bus (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (4/5 * usual_speed) * (usual_time + 4) = usual_speed * usual_time → usual_time = 16 := by
  sorry

#check usual_time_to_catch_bus

end usual_time_to_catch_bus_l1296_129626


namespace worker_selection_probability_l1296_129671

theorem worker_selection_probability 
  (total_workers : ℕ) 
  (eliminated_workers : ℕ) 
  (remaining_workers : ℕ) 
  (representatives : ℕ) 
  (h1 : total_workers = 2009)
  (h2 : eliminated_workers = 9)
  (h3 : remaining_workers = 2000)
  (h4 : representatives = 100)
  (h5 : remaining_workers = total_workers - eliminated_workers) :
  (representatives : ℚ) / (total_workers : ℚ) = 100 / 2009 :=
by sorry

end worker_selection_probability_l1296_129671


namespace transformed_area_l1296_129600

-- Define the transformation matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 8, -2]

-- Define the original region T and its area
def area_T : ℝ := 12

-- Define the transformed region T' and its area
def area_T' : ℝ := |A.det| * area_T

-- Theorem statement
theorem transformed_area :
  area_T' = 456 :=
sorry

end transformed_area_l1296_129600


namespace arithmetic_calculation_l1296_129612

theorem arithmetic_calculation : 4 * 6 * 8 - 24 / 6 = 188 := by
  sorry

end arithmetic_calculation_l1296_129612


namespace amp_fifteen_amp_l1296_129690

-- Define the ampersand operations
def amp_right (x : ℝ) : ℝ := 8 - x
def amp_left (x : ℝ) : ℝ := x - 9

-- State the theorem
theorem amp_fifteen_amp : amp_left (amp_right 15) = -16 := by
  sorry

end amp_fifteen_amp_l1296_129690


namespace sum_of_roots_is_one_l1296_129620

-- Define a quadratic polynomial Q(x) = ax^2 + bx + c
def Q (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem sum_of_roots_is_one 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, Q a b c (x^4 + x^2) ≥ Q a b c (x^3 + 1)) : 
  (- b) / a = 1 := by
  sorry

end sum_of_roots_is_one_l1296_129620


namespace equal_areas_in_circle_configuration_l1296_129608

/-- Given a circle with radius R and four smaller circles with radius r = R/2 drawn through its center and touching it, 
    the area of the region not covered by the smaller circles (black region) 
    is equal to the sum of the areas of the overlapping regions of the smaller circles (gray regions). -/
theorem equal_areas_in_circle_configuration (R : ℝ) (h : R > 0) : 
  ∃ (black_area gray_area : ℝ),
    black_area = R^2 * π - 4 * (R/2)^2 * π ∧
    gray_area = 4 * ((R/2)^2 * π - (R/2)^2 * π / 3) ∧
    black_area = gray_area :=
by sorry

end equal_areas_in_circle_configuration_l1296_129608


namespace permutations_theorem_l1296_129682

-- Define the number of books
def n : ℕ := 30

-- Define the function to calculate the number of permutations where two specific objects are not adjacent
def permutations_not_adjacent (n : ℕ) : ℕ := 28 * Nat.factorial (n - 1)

-- Theorem statement
theorem permutations_theorem :
  permutations_not_adjacent n = (n - 2) * Nat.factorial (n - 1) :=
by sorry

end permutations_theorem_l1296_129682


namespace store_B_cheapest_l1296_129643

/-- Represents a store with its pricing strategy -/
structure Store :=
  (name : String)
  (basePrice : ℕ)
  (discountStrategy : ℕ → ℕ)

/-- Calculates the cost of buying balls from a store -/
def cost (s : Store) (balls : ℕ) : ℕ :=
  s.discountStrategy balls

/-- Store A's discount strategy -/
def storeAStrategy (balls : ℕ) : ℕ :=
  let freeBalls := (balls / 10) * 2
  (balls - freeBalls) * 25

/-- Store B's discount strategy -/
def storeBStrategy (balls : ℕ) : ℕ :=
  balls * (25 - 5)

/-- Store C's discount strategy -/
def storeCStrategy (balls : ℕ) : ℕ :=
  let totalSpent := balls * 25
  let cashback := (totalSpent / 200) * 30
  totalSpent - cashback

/-- The three stores -/
def storeA : Store := ⟨"A", 25, storeAStrategy⟩
def storeB : Store := ⟨"B", 25, storeBStrategy⟩
def storeC : Store := ⟨"C", 25, storeCStrategy⟩

/-- The theorem to prove -/
theorem store_B_cheapest : 
  cost storeB 60 < cost storeA 60 ∧ cost storeB 60 < cost storeC 60 := by
  sorry

end store_B_cheapest_l1296_129643


namespace repeating_decimal_equals_fraction_l1296_129615

/-- The repeating decimal 5.8̄ -/
def repeating_decimal : ℚ := 5 + 8/9

/-- The fraction 53/9 -/
def target_fraction : ℚ := 53/9

/-- Theorem stating that the repeating decimal 5.8̄ is equal to the fraction 53/9 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end repeating_decimal_equals_fraction_l1296_129615


namespace original_price_after_discounts_l1296_129606

/-- Given an article sold at $144 after two successive discounts of 10% and 20%, 
    prove that its original price was $200. -/
theorem original_price_after_discounts (final_price : ℝ) 
  (h1 : final_price = 144)
  (discount1 : ℝ) (h2 : discount1 = 0.1)
  (discount2 : ℝ) (h3 : discount2 = 0.2) :
  ∃ (original_price : ℝ), 
    original_price = 200 ∧
    final_price = original_price * (1 - discount1) * (1 - discount2) :=
by
  sorry


end original_price_after_discounts_l1296_129606


namespace arrange_five_books_two_identical_l1296_129624

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (identical : ℕ) : ℕ :=
  (Nat.factorial total) / (Nat.factorial identical)

/-- Theorem: The number of ways to arrange 5 books, where 2 are identical, is 60 -/
theorem arrange_five_books_two_identical :
  arrange_books 5 2 = 60 := by
  sorry

end arrange_five_books_two_identical_l1296_129624


namespace sin_integral_minus_two_to_two_l1296_129683

theorem sin_integral_minus_two_to_two : ∫ x in (-2)..2, Real.sin x = 0 := by sorry

end sin_integral_minus_two_to_two_l1296_129683


namespace largest_of_seven_consecutive_integers_l1296_129664

theorem largest_of_seven_consecutive_integers (a : ℕ) : 
  (∃ (x : ℕ), x > 0 ∧ 
    (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) = 3020) ∧
    (∀ (y : ℕ), y > 0 → 
      (y + (y + 1) + (y + 2) + (y + 3) + (y + 4) + (y + 5) + (y + 6) = 3020) → 
      y = x)) →
  a = 434 ∧
  (∃ (x : ℕ), x > 0 ∧ 
    (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + a = 3020) ∧
    (∀ (y : ℕ), y > 0 → 
      (y + (y + 1) + (y + 2) + (y + 3) + (y + 4) + (y + 5) + a = 3020) → 
      y = x)) :=
by sorry

end largest_of_seven_consecutive_integers_l1296_129664


namespace complex_product_real_l1296_129650

theorem complex_product_real (x : ℝ) : 
  let z₁ : ℂ := 1 + I
  let z₂ : ℂ := x - I
  (z₁ * z₂).im = 0 → x = 1 := by
sorry

end complex_product_real_l1296_129650


namespace pen_discount_problem_l1296_129686

/-- Proves that given a 12.5% discount on pens and the ability to buy 13 more pens
    after the discount, the original number of pens that could be bought before
    the discount is 91. -/
theorem pen_discount_problem (money : ℝ) (original_price : ℝ) 
  (original_price_positive : original_price > 0) :
  let discount_rate : ℝ := 0.125
  let discounted_price : ℝ := original_price * (1 - discount_rate)
  let original_quantity : ℝ := money / original_price
  let discounted_quantity : ℝ := money / discounted_price
  discounted_quantity - original_quantity = 13 →
  original_quantity = 91 := by
sorry


end pen_discount_problem_l1296_129686


namespace board_length_proof_l1296_129684

theorem board_length_proof :
  ∀ (short_piece long_piece total_length : ℝ),
  short_piece > 0 →
  long_piece = 2 * short_piece →
  long_piece = 46 →
  total_length = short_piece + long_piece →
  total_length = 69 :=
by
  sorry

end board_length_proof_l1296_129684


namespace surface_area_unchanged_after_cube_removal_l1296_129688

theorem surface_area_unchanged_after_cube_removal 
  (l w h : ℝ) (cube_side : ℝ) 
  (hl : l = 10) (hw : w = 5) (hh : h = 3) (hc : cube_side = 2) : 
  2 * (l * w + l * h + w * h) = 
  2 * (l * w + l * h + w * h) - 3 * cube_side^2 + 3 * cube_side^2 := by
  sorry

end surface_area_unchanged_after_cube_removal_l1296_129688


namespace roots_sum_magnitude_l1296_129633

theorem roots_sum_magnitude (p : ℝ) (r₁ r₂ : ℝ) : 
  (∃ x : ℝ, x^2 + p*x + 12 = 0) →
  r₁^2 + p*r₁ + 12 = 0 →
  r₂^2 + p*r₂ + 12 = 0 →
  r₁ ≠ r₂ →
  |r₁ + r₂| > 6 := by
sorry

end roots_sum_magnitude_l1296_129633


namespace pucks_not_in_original_position_l1296_129693

/-- Represents the arrangement of three objects -/
inductive Arrangement
  | Clockwise
  | Counterclockwise

/-- Represents a single hit that changes the arrangement -/
def hit (a : Arrangement) : Arrangement :=
  match a with
  | Arrangement.Clockwise => Arrangement.Counterclockwise
  | Arrangement.Counterclockwise => Arrangement.Clockwise

/-- Applies n hits to the initial arrangement -/
def applyHits (initial : Arrangement) (n : Nat) : Arrangement :=
  match n with
  | 0 => initial
  | n + 1 => hit (applyHits initial n)

theorem pucks_not_in_original_position (initial : Arrangement) :
  applyHits initial 25 ≠ initial := by
  sorry


end pucks_not_in_original_position_l1296_129693


namespace equal_sum_sequence_sixth_term_l1296_129651

/-- An Equal Sum Sequence is a sequence where the sum of each term and its next term is always the same constant. -/
def EqualSumSequence (a : ℕ → ℝ) (c : ℝ) :=
  ∀ n, a n + a (n + 1) = c

theorem equal_sum_sequence_sixth_term
  (a : ℕ → ℝ)
  (h1 : EqualSumSequence a 5)
  (h2 : a 1 = 2) :
  a 6 = 3 := by
sorry

end equal_sum_sequence_sixth_term_l1296_129651


namespace sqrt_sum_equals_sqrt_192_l1296_129677

theorem sqrt_sum_equals_sqrt_192 (N : ℕ+) :
  Real.sqrt 12 + Real.sqrt 108 = Real.sqrt N.1 → N.1 = 192 := by
  sorry

end sqrt_sum_equals_sqrt_192_l1296_129677


namespace eight_power_twelve_sum_equals_two_power_y_l1296_129628

theorem eight_power_twelve_sum_equals_two_power_y (y : ℕ) : 
  (8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 = 2^y) → y = 39 := by
sorry

end eight_power_twelve_sum_equals_two_power_y_l1296_129628


namespace num_non_congruent_triangles_l1296_129623

/-- Represents a point on a 2D grid --/
structure GridPoint where
  x : ℚ
  y : ℚ

/-- The set of points on the grid --/
def gridPoints : Finset GridPoint := sorry

/-- Predicate to check if three points form a triangle --/
def isTriangle (p q r : GridPoint) : Prop := sorry

/-- Predicate to check if two triangles are congruent --/
def areCongruent (t1 t2 : GridPoint × GridPoint × GridPoint) : Prop := sorry

/-- The set of all possible triangles formed by the grid points --/
def allTriangles : Finset (GridPoint × GridPoint × GridPoint) := sorry

/-- The set of non-congruent triangles --/
def nonCongruentTriangles : Finset (GridPoint × GridPoint × GridPoint) := sorry

theorem num_non_congruent_triangles :
  Finset.card nonCongruentTriangles = 4 := by sorry

end num_non_congruent_triangles_l1296_129623


namespace x_over_y_value_l1296_129674

theorem x_over_y_value (x y : ℝ) 
  (h1 : 3 < (x - y) / (x + y)) 
  (h2 : (x - y) / (x + y) < 6)
  (h3 : ∃ (n : ℤ), x / y = n) : 
  x / y = -2 := by
sorry

end x_over_y_value_l1296_129674


namespace set_A_elements_l1296_129629

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | x^2 + 2*x + a = 0}

-- State the theorem
theorem set_A_elements (a : ℝ) (h : 1 ∈ A a) : A a = {-3, 1} := by
  sorry

end set_A_elements_l1296_129629


namespace symmetric_line_l1296_129663

/-- Given a line L1 with equation x - 2y + 1 = 0 and a line of symmetry x = 1,
    the symmetric line L2 has the equation x + 2y - 3 = 0 -/
theorem symmetric_line (x y : ℝ) :
  (x - 2*y + 1 = 0) →  -- equation of L1
  (x = 1) →            -- line of symmetry
  (x + 2*y - 3 = 0)    -- equation of L2
:= by sorry

end symmetric_line_l1296_129663


namespace highest_power_of_two_dividing_difference_of_sixth_powers_l1296_129635

theorem highest_power_of_two_dividing_difference_of_sixth_powers :
  ∃ k : ℕ, 2^k = (Nat.gcd (15^6 - 9^6) (2^64)) ∧ k = 4 := by
  sorry

end highest_power_of_two_dividing_difference_of_sixth_powers_l1296_129635


namespace diamond_value_l1296_129607

/-- Represents a digit (0-9) -/
def Digit := Fin 10

theorem diamond_value (diamond : Digit) :
  (9 * diamond.val + 6 = 10 * diamond.val + 3) → diamond.val = 3 := by
  sorry

end diamond_value_l1296_129607


namespace chord_length_l1296_129697

/-- The parabola M: y^2 = 2px where p > 0 -/
def parabola_M (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

/-- The circle C: x^2 + (y-4)^2 = a^2 -/
def circle_C (a : ℝ) (x y : ℝ) : Prop := x^2 + (y-4)^2 = a^2

/-- Point A is in the first quadrant -/
def point_A_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- Distance from A to focus of parabola M is a -/
def distance_A_to_focus (p a x y : ℝ) : Prop := (x - p/2)^2 + y^2 = a^2

/-- Sum of distances from a point on M to its directrix and to point C has a maximum of 2 -/
def max_distance_sum (p : ℝ) : Prop := ∃ (x y : ℝ), parabola_M p x y ∧ (x + p/2) + ((x - 0)^2 + (y - 4)^2).sqrt ≤ 2

/-- The theorem: Length of chord intercepted by line OA on circle C is 7√2/3 -/
theorem chord_length (p a x y : ℝ) : 
  parabola_M p x y → 
  circle_C a x y → 
  point_A_first_quadrant x y → 
  distance_A_to_focus p a x y → 
  max_distance_sum p → 
  ((2 * a)^2 - (8/3)^2).sqrt = 7 * Real.sqrt 2 / 3 := by sorry

end chord_length_l1296_129697


namespace trig_expression_equality_l1296_129648

theorem trig_expression_equality : 
  (Real.sin (24 * π / 180) * Real.cos (16 * π / 180) + Real.cos (156 * π / 180) * Real.sin (66 * π / 180)) / 
  (Real.sin (28 * π / 180) * Real.cos (12 * π / 180) + Real.cos (152 * π / 180) * Real.sin (72 * π / 180)) = 
  1 / Real.sin (80 * π / 180) := by sorry

end trig_expression_equality_l1296_129648


namespace complement_of_angle_alpha_l1296_129662

/-- Represents an angle in degrees, minutes, and seconds -/
structure AngleDMS where
  degrees : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Calculates the complement of an angle in DMS format -/
def angleComplement (α : AngleDMS) : AngleDMS :=
  sorry

/-- The given angle α -/
def α : AngleDMS := ⟨36, 14, 25⟩

/-- Theorem: The complement of angle α is 53°45'35" -/
theorem complement_of_angle_alpha :
  angleComplement α = ⟨53, 45, 35⟩ := by sorry

end complement_of_angle_alpha_l1296_129662


namespace apple_boxes_l1296_129659

theorem apple_boxes (class_5A class_5B class_5C : ℕ) 
  (h1 : class_5A = 560)
  (h2 : class_5B = 595)
  (h3 : class_5C = 735) :
  let box_weight := Nat.gcd class_5A (Nat.gcd class_5B class_5C)
  (class_5A / box_weight, class_5B / box_weight, class_5C / box_weight) = (16, 17, 21) := by
  sorry

#check apple_boxes

end apple_boxes_l1296_129659


namespace circle_center_distance_to_line_l1296_129603

/-- A circle passing through (1, 2) and tangent to both coordinate axes has its center
    at distance 2√5/5 from the line 2x - y - 3 = 0 -/
theorem circle_center_distance_to_line :
  ∀ (a : ℝ), 
    (∃ (x y : ℝ), (x - a)^2 + (y - a)^2 = a^2 ∧ x = 1 ∧ y = 2) →  -- Circle passes through (1, 2)
    (∃ (x : ℝ), (x - a)^2 + a^2 = a^2) →                          -- Circle is tangent to x-axis
    (∃ (y : ℝ), a^2 + (y - a)^2 = a^2) →                          -- Circle is tangent to y-axis
    (|a - 3| / Real.sqrt 5 : ℝ) = 2 * Real.sqrt 5 / 5 :=
by sorry


end circle_center_distance_to_line_l1296_129603


namespace boat_upstream_distance_l1296_129673

/-- Represents the distance traveled by a boat in one hour -/
def boat_distance (still_speed : ℝ) (stream_speed : ℝ) : ℝ :=
  still_speed - stream_speed

/-- The boat's speed in still water (km/hr) -/
def still_speed : ℝ := 7

/-- The distance the boat travels along the stream in one hour (km) -/
def downstream_distance : ℝ := 9

/-- The stream speed (km/hr) -/
def stream_speed : ℝ := downstream_distance - still_speed

theorem boat_upstream_distance :
  boat_distance still_speed stream_speed = 5 := by
  sorry


end boat_upstream_distance_l1296_129673


namespace terminal_side_first_quadrant_l1296_129665

-- Define the angle in degrees
def angle : ℝ := -330

-- Define the quadrants
inductive Quadrant
  | first
  | second
  | third
  | fourth

-- Define a function to determine the quadrant of an angle
def angle_quadrant (θ : ℝ) : Quadrant :=
  sorry

-- Theorem statement
theorem terminal_side_first_quadrant :
  angle_quadrant angle = Quadrant.first :=
sorry

end terminal_side_first_quadrant_l1296_129665


namespace negative_one_squared_equals_negative_one_l1296_129625

theorem negative_one_squared_equals_negative_one : -1^2 = -1 := by
  sorry

end negative_one_squared_equals_negative_one_l1296_129625


namespace john_pill_schedule_l1296_129695

/-- The number of pills John takes per week -/
def pills_per_week : ℕ := 28

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of hours between each pill John takes -/
def hours_between_pills : ℚ :=
  (days_per_week * hours_per_day) / pills_per_week

theorem john_pill_schedule :
  hours_between_pills = 6 := by sorry

end john_pill_schedule_l1296_129695


namespace min_value_of_f_zero_l1296_129687

/-- A quadratic function from reals to reals -/
def QuadraticFunction := ℝ → ℝ

/-- Predicate to check if a function is quadratic -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- Predicate to check if a function is ever more than another function -/
def EverMore (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x ≥ g x

/-- The theorem statement -/
theorem min_value_of_f_zero
  (f : QuadraticFunction)
  (hquad : IsQuadratic f)
  (hf1 : f 1 = 16)
  (hg : EverMore f (fun x ↦ (x + 3)^2))
  (hh : EverMore f (fun x ↦ x^2 + 9)) :
  ∃ (min_f0 : ℝ), min_f0 = 21/2 ∧ f 0 ≥ min_f0 :=
by sorry

end min_value_of_f_zero_l1296_129687


namespace count_negative_numbers_l1296_129616

def number_list : List ℚ := [-2 - 2/3, 9/14, -3, 5/2, 0, -48/10, 5, -1]

theorem count_negative_numbers : 
  (number_list.filter (λ x => x < 0)).length = 4 := by
  sorry

end count_negative_numbers_l1296_129616


namespace sum_range_for_cube_sum_two_l1296_129689

theorem sum_range_for_cube_sum_two (x y : ℝ) (h : x^3 + y^3 = 2) :
  0 < x + y ∧ x + y ≤ 2 := by
  sorry

end sum_range_for_cube_sum_two_l1296_129689


namespace pant_cost_l1296_129646

theorem pant_cost (num_shirts : ℕ) (shirt_cost : ℕ) (total_cost : ℕ) : 
  num_shirts = 10 →
  shirt_cost = 6 →
  total_cost = 100 →
  ∃ (pant_cost : ℕ), 
    pant_cost = 8 ∧ 
    num_shirts * shirt_cost + (num_shirts / 2) * pant_cost = total_cost :=
by sorry

end pant_cost_l1296_129646


namespace equation_solution_l1296_129640

theorem equation_solution : 
  ∃ x : ℝ, 3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * x)) = 2800.0000000000005 ∧ x = 1.25 := by
  sorry

end equation_solution_l1296_129640


namespace lily_initial_money_l1296_129647

def celery_cost : ℝ := 5
def cereal_original_cost : ℝ := 12
def cereal_discount : ℝ := 0.5
def bread_cost : ℝ := 8
def milk_original_cost : ℝ := 10
def milk_discount : ℝ := 0.1
def potato_cost : ℝ := 1
def potato_quantity : ℕ := 6
def coffee_budget : ℝ := 26

def total_cost : ℝ := 
  celery_cost + 
  cereal_original_cost * (1 - cereal_discount) + 
  bread_cost + 
  milk_original_cost * (1 - milk_discount) + 
  potato_cost * potato_quantity +
  coffee_budget

theorem lily_initial_money : total_cost = 60 := by
  sorry

end lily_initial_money_l1296_129647


namespace candy_distribution_l1296_129660

def is_valid_student_count (n : ℕ) : Prop :=
  n > 1 ∧ 129 % n = 0

theorem candy_distribution (total_candies : ℕ) (h_total : total_candies = 130) :
  ∀ n : ℕ, is_valid_student_count n ↔ (n = 3 ∨ n = 43 ∨ n = 129) :=
sorry

end candy_distribution_l1296_129660


namespace zlatoust_miass_distance_l1296_129637

theorem zlatoust_miass_distance :
  ∀ (g m k : ℝ), g > 0 → m > 0 → k > 0 →
  ∃ (x : ℝ), x > 0 ∧
  (x + 18) / k = (x - 18) / m ∧
  (x + 25) / k = (x - 25) / g ∧
  (x + 8) / m = (x - 8) / g ∧
  x = 60 :=
by sorry

end zlatoust_miass_distance_l1296_129637


namespace balls_without_holes_count_l1296_129699

/-- The number of soccer balls Matthias has -/
def total_soccer_balls : ℕ := 40

/-- The number of basketballs Matthias has -/
def total_basketballs : ℕ := 15

/-- The number of soccer balls with holes -/
def soccer_balls_with_holes : ℕ := 30

/-- The number of basketballs with holes -/
def basketballs_with_holes : ℕ := 7

/-- The total number of balls without holes -/
def total_balls_without_holes : ℕ := 
  (total_soccer_balls - soccer_balls_with_holes) + (total_basketballs - basketballs_with_holes)

theorem balls_without_holes_count : total_balls_without_holes = 18 := by
  sorry

end balls_without_holes_count_l1296_129699


namespace incorrect_inequality_l1296_129661

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ¬(-a + 2 > -b + 2) := by
  sorry

end incorrect_inequality_l1296_129661


namespace number_of_possible_lists_l1296_129630

def number_of_balls : ℕ := 15
def list_length : ℕ := 4

theorem number_of_possible_lists :
  (number_of_balls ^ list_length : ℕ) = 50625 := by
sorry

end number_of_possible_lists_l1296_129630


namespace library_book_combinations_l1296_129675

theorem library_book_combinations : Nat.choose 5 2 = 10 := by
  sorry

end library_book_combinations_l1296_129675


namespace a_range_l1296_129605

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3-2*a)^x < (3-2*a)^y

def range_a (a : ℝ) : Prop := a ≤ -2 ∨ (1 ≤ a ∧ a < 2)

theorem a_range (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_a a := by sorry

end a_range_l1296_129605


namespace waiter_customers_l1296_129657

/-- The initial number of customers -/
def initial_customers : ℕ := 47

/-- The number of customers who left -/
def customers_left : ℕ := 41

/-- The number of new customers who arrived -/
def new_customers : ℕ := 20

/-- The final number of customers -/
def final_customers : ℕ := 26

theorem waiter_customers : 
  initial_customers - customers_left + new_customers = final_customers :=
by sorry

end waiter_customers_l1296_129657


namespace polygon_sides_l1296_129696

theorem polygon_sides (n : ℕ) : n > 2 →
  (n - 2) * 180 = 3 * 360 - 180 →
  n = 7 :=
by sorry

end polygon_sides_l1296_129696


namespace cookie_difference_l1296_129634

theorem cookie_difference (initial_sweet initial_salty sweet_eaten salty_eaten : ℕ) :
  initial_sweet = 39 →
  initial_salty = 6 →
  sweet_eaten = 32 →
  salty_eaten = 23 →
  sweet_eaten - salty_eaten = 9 :=
by
  sorry

end cookie_difference_l1296_129634


namespace total_carrots_is_twenty_l1296_129656

/-- The number of carrots grown by Sally -/
def sally_carrots : ℕ := 6

/-- The number of carrots grown by Fred -/
def fred_carrots : ℕ := 4

/-- The number of carrots grown by Mary -/
def mary_carrots : ℕ := 10

/-- The total number of carrots grown by Sally, Fred, and Mary -/
def total_carrots : ℕ := sally_carrots + fred_carrots + mary_carrots

theorem total_carrots_is_twenty : total_carrots = 20 := by sorry

end total_carrots_is_twenty_l1296_129656


namespace count_special_numbers_eq_56_l1296_129681

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def count_special_numbers : ℕ :=
  let thousands_digit := 8
  let units_digit := 2
  let hundreds_choices := 8
  let tens_choices := 7
  thousands_digit * units_digit * hundreds_choices * tens_choices

theorem count_special_numbers_eq_56 :
  count_special_numbers = 56 :=
sorry

end count_special_numbers_eq_56_l1296_129681


namespace jen_final_distance_l1296_129652

/-- Calculates the final distance from the starting point for a person walking
    at a constant rate, forward for a certain time, then back for another time. -/
def final_distance (rate : ℝ) (forward_time : ℝ) (back_time : ℝ) : ℝ :=
  rate * forward_time - rate * back_time

/-- Theorem stating that given the specific conditions of Jen's walk,
    her final distance from the starting point is 4 miles. -/
theorem jen_final_distance :
  let rate : ℝ := 4
  let forward_time : ℝ := 2
  let back_time : ℝ := 1
  final_distance rate forward_time back_time = 4 := by
  sorry

end jen_final_distance_l1296_129652


namespace anns_age_l1296_129627

theorem anns_age (ann barbara : ℕ) : 
  ann + barbara = 62 →  -- Sum of their ages is 62
  ann = 2 * (barbara - (ann - barbara)) →  -- Ann's age relation
  ann = 50 :=
by sorry

end anns_age_l1296_129627


namespace diamond_two_seven_l1296_129672

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 3 * x + 5 * y

-- State the theorem
theorem diamond_two_seven : diamond 2 7 = 41 := by
  sorry

end diamond_two_seven_l1296_129672


namespace team_selection_probability_l1296_129644

/-- The probability of randomly selecting a team that includes three specific players -/
theorem team_selection_probability 
  (total_players : ℕ) 
  (team_size : ℕ) 
  (specific_players : ℕ) 
  (h1 : total_players = 12) 
  (h2 : team_size = 6) 
  (h3 : specific_players = 3) :
  (Nat.choose (total_players - specific_players) (team_size - specific_players)) / 
  (Nat.choose total_players team_size) = 1 / 11 := by
  sorry

end team_selection_probability_l1296_129644


namespace option1_better_than_option2_l1296_129631

def initial_amount : ℝ := 12000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_discounts : List ℝ := [0.15, 0.25, 0.10]
def option2_discounts : List ℝ := [0.25, 0.10, 0.10]

def apply_successive_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

theorem option1_better_than_option2 :
  apply_successive_discounts initial_amount option1_discounts <
  apply_successive_discounts initial_amount option2_discounts :=
sorry

end option1_better_than_option2_l1296_129631


namespace roots_pure_imaginary_l1296_129609

theorem roots_pure_imaginary (k : ℝ) (hk : k > 0) :
  ∃ (b c : ℝ), ∀ (z : ℂ), 8 * z^2 - 5 * I * z - k = 0 → z = b * I ∨ z = c * I :=
by sorry

end roots_pure_imaginary_l1296_129609


namespace thirty_percent_less_than_eighty_l1296_129613

theorem thirty_percent_less_than_eighty (x : ℝ) : x + (1/4) * x = 80 - (30/100) * 80 → x = 45 := by
  sorry

end thirty_percent_less_than_eighty_l1296_129613


namespace quadratic_real_solutions_l1296_129680

theorem quadratic_real_solutions (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2 * x + 1 = 0) ↔ (m ≤ 1 ∧ m ≠ 0) :=
by sorry

end quadratic_real_solutions_l1296_129680
