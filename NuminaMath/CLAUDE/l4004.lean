import Mathlib

namespace divisibility_by_three_l4004_400406

theorem divisibility_by_three (a b : ℕ) : 
  (3 ∣ (a * b)) → ¬(¬(3 ∣ a) ∧ ¬(3 ∣ b)) := by
  sorry

end divisibility_by_three_l4004_400406


namespace group_value_l4004_400442

theorem group_value (a : ℝ) (h : 21 ≤ a ∧ a < 41) : (21 + 41) / 2 = 31 := by
  sorry

#check group_value

end group_value_l4004_400442


namespace sequence_property_l4004_400441

theorem sequence_property (x y z : ℝ) 
  (h1 : (4 * y) ^ 2 = (3 * x) * (5 * z))  -- Geometric sequence condition
  (h2 : 2 / y = 1 / x + 1 / z)            -- Arithmetic sequence condition
  : x / z + z / x = 34 / 15 := by
  sorry

end sequence_property_l4004_400441


namespace average_diff_100_400_50_250_l4004_400473

def average_difference : ℤ → ℤ → ℤ → ℤ → ℤ :=
  fun a b c d => ((b + a) / 2) - ((d + c) / 2)

theorem average_diff_100_400_50_250 :
  average_difference 100 400 50 250 = 100 := by
  sorry

end average_diff_100_400_50_250_l4004_400473


namespace water_fraction_in_mixture_l4004_400434

/-- Given a cement mixture with total weight, sand fraction, and gravel weight,
    calculate the fraction of water in the mixture. -/
theorem water_fraction_in_mixture
  (total_weight : ℝ)
  (sand_fraction : ℝ)
  (gravel_weight : ℝ)
  (h1 : total_weight = 48)
  (h2 : sand_fraction = 1/3)
  (h3 : gravel_weight = 8) :
  (total_weight - (sand_fraction * total_weight + gravel_weight)) / total_weight = 1/2 := by
  sorry

#check water_fraction_in_mixture

end water_fraction_in_mixture_l4004_400434


namespace rocket_arrangements_l4004_400418

def word : String := "ROCKET"

theorem rocket_arrangements : 
  (∃ (c : Char), c ∈ word.data ∧ 
    (word.data.count c = 2) ∧ 
    (∀ (d : Char), d ∈ word.data ∧ d ≠ c → word.data.count d = 1)) →
  (Nat.factorial (word.length + 1) / 2 = 2520) :=
by sorry

end rocket_arrangements_l4004_400418


namespace quadratic_function_minimum_l4004_400495

theorem quadratic_function_minimum (a b c : ℝ) (h_a : a ≠ 0) :
  let f := fun x => a * x^2 + b * x + c
  let f' := fun x => 2 * a * x + b
  (f' 0 > 0) →
  (∀ x, f x ≥ 0) →
  ∀ ε > 0, ∃ x, f x / f' 0 < 2 + ε :=
by sorry

end quadratic_function_minimum_l4004_400495


namespace roommate_difference_l4004_400465

theorem roommate_difference (bob_roommates john_roommates : ℕ) 
  (h1 : bob_roommates = 10) 
  (h2 : john_roommates = 25) : 
  john_roommates - 2 * bob_roommates = 5 := by
  sorry

end roommate_difference_l4004_400465


namespace modulus_of_complex_number_l4004_400462

theorem modulus_of_complex_number : 
  Complex.abs (Complex.mk 1 (-2)) = Real.sqrt 5 := by sorry

end modulus_of_complex_number_l4004_400462


namespace decreasing_interval_of_sine_function_l4004_400437

/-- Given a function f(x) = 2sin(2x + φ) where 0 < φ < π/2 and f(0) = √3,
    prove that the decreasing interval of f(x) on [0, π] is [π/12, 7π/12]. -/
theorem decreasing_interval_of_sine_function (φ : Real) 
    (h1 : 0 < φ) (h2 : φ < π/2) 
    (f : Real → Real) 
    (hf : ∀ x, f x = 2 * Real.sin (2 * x + φ)) 
    (h3 : f 0 = Real.sqrt 3) :
    (Set.Icc (π/12 : Real) (7*π/12) : Set Real) = 
    {x ∈ Set.Icc (0 : Real) π | ∀ y ∈ Set.Icc (0 : Real) π, x < y → f y < f x} :=
  sorry

end decreasing_interval_of_sine_function_l4004_400437


namespace geometric_progression_solution_l4004_400438

theorem geometric_progression_solution : 
  ∃! x : ℝ, (30 + x)^2 = (15 + x) * (60 + x) :=
by
  sorry

end geometric_progression_solution_l4004_400438


namespace selling_price_calculation_l4004_400458

theorem selling_price_calculation (cost_price : ℝ) (profit_percentage : ℝ) : 
  cost_price = 240 → profit_percentage = 20 → 
  cost_price * (1 + profit_percentage / 100) = 288 := by
  sorry

end selling_price_calculation_l4004_400458


namespace exists_interior_rectangle_l4004_400429

/-- A rectangle in a square partition -/
structure Rectangle where
  left : ℝ
  right : ℝ
  bottom : ℝ
  top : ℝ
  left_lt_right : left < right
  bottom_lt_top : bottom < top

/-- A partition of a square into rectangles -/
structure SquarePartition where
  rectangles : List Rectangle
  n_gt_one : rectangles.length > 1
  covers_square : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 →
    ∃ r ∈ rectangles, r.left ≤ x ∧ x ≤ r.right ∧ r.bottom ≤ y ∧ y ≤ r.top
  intersects_line : ∀ l : ℝ, 0 < l ∧ l < 1 →
    (∃ r ∈ rectangles, r.left < l ∧ l < r.right) ∧
    (∃ r ∈ rectangles, r.bottom < l ∧ l < r.top)

/-- A rectangle touches the side of the square if any of its sides coincide with the square's sides -/
def touches_side (r : Rectangle) : Prop :=
  r.left = 0 ∨ r.right = 1 ∨ r.bottom = 0 ∨ r.top = 1

/-- Main theorem: There exists a rectangle that doesn't touch the sides of the square -/
theorem exists_interior_rectangle (p : SquarePartition) :
  ∃ r ∈ p.rectangles, ¬touches_side r := by
  sorry

end exists_interior_rectangle_l4004_400429


namespace bracelets_made_l4004_400488

/-- The number of beads in each bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of metal beads Nancy has -/
def nancy_metal_beads : ℕ := 40

/-- The number of pearl beads Nancy has -/
def nancy_pearl_beads : ℕ := nancy_metal_beads + 20

/-- The number of crystal beads Rose has -/
def rose_crystal_beads : ℕ := 20

/-- The number of stone beads Rose has -/
def rose_stone_beads : ℕ := 2 * rose_crystal_beads

/-- The total number of beads Nancy and Rose have -/
def total_beads : ℕ := nancy_metal_beads + nancy_pearl_beads + rose_crystal_beads + rose_stone_beads

/-- The theorem stating the number of bracelets Nancy and Rose can make -/
theorem bracelets_made : total_beads / beads_per_bracelet = 20 := by
  sorry

end bracelets_made_l4004_400488


namespace cube_sum_problem_l4004_400421

theorem cube_sum_problem (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a * b + a * c + b * c = 1) 
  (h3 : a * b * c = -2) : 
  a^3 + b^3 + c^3 = -6 := by
sorry

end cube_sum_problem_l4004_400421


namespace polynomial_factor_implies_coefficients_l4004_400460

theorem polynomial_factor_implies_coefficients 
  (a b : ℚ) 
  (h : ∃ (c d k : ℚ), ax^4 + bx^3 + 38*x^2 - 12*x + 15 = (3*x^2 - 2*x + 2)*(c*x^2 + d*x + k)) :
  a = -75/2 ∧ b = 59/2 := by
  sorry

end polynomial_factor_implies_coefficients_l4004_400460


namespace discount_percentage_is_ten_percent_l4004_400412

/-- Calculates the discount percentage on a retail price given the wholesale price, retail price, and profit percentage. -/
def discount_percentage (wholesale_price retail_price profit_percentage : ℚ) : ℚ :=
  let profit := wholesale_price * profit_percentage / 100
  let selling_price := wholesale_price + profit
  let discount_amount := retail_price - selling_price
  (discount_amount / retail_price) * 100

/-- Proves that the discount percentage is 10% given the problem conditions. -/
theorem discount_percentage_is_ten_percent :
  discount_percentage 90 120 20 = 10 := by
  sorry

#eval discount_percentage 90 120 20

end discount_percentage_is_ten_percent_l4004_400412


namespace symmetry_of_shifted_even_function_l4004_400453

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define axis of symmetry for a function
def AxisOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- Theorem statement
theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) :
  EvenFunction f → AxisOfSymmetry (fun x ↦ f (x + 1)) (-1) := by
  sorry

end symmetry_of_shifted_even_function_l4004_400453


namespace intersection_of_planes_and_line_l4004_400419

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relation for intersection between planes
variable (intersect : Plane → Plane → Prop)

-- Define the relation for perpendicularity between planes
variable (perpendicular : Plane → Plane → Prop)

-- Define the relation for a line lying in a plane
variable (lies_in : Line → Plane → Prop)

-- Define the relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Define the relation for perpendicular lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the theorem
theorem intersection_of_planes_and_line 
  (α β : Plane) (m : Line)
  (h1 : intersect α β)
  (h2 : ¬ perpendicular α β)
  (h3 : lies_in m α) :
  (∃ (n : Line), lies_in n β ∧ ¬ (∀ (n : Line), lies_in n β → parallel m n)) ∧
  (∃ (p : Line), lies_in p β ∧ perpendicular_lines m p) :=
sorry

end intersection_of_planes_and_line_l4004_400419


namespace f_decreasing_on_interval_l4004_400490

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo 0 2, 
    ∀ y ∈ Set.Ioo 0 2, 
      x < y → f x > f y :=
by sorry

end f_decreasing_on_interval_l4004_400490


namespace isosceles_triangle_perimeter_l4004_400435

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12. -/
theorem isosceles_triangle_perimeter : 
  ∀ (a b c : ℝ), 
  a = 5 → b = 5 → c = 2 →  -- Two sides are 5, one side is 2
  a = b →                  -- The triangle is isosceles
  a + b + c = 12 :=        -- The perimeter is 12
by
  sorry


end isosceles_triangle_perimeter_l4004_400435


namespace coords_of_A_wrt_origin_l4004_400424

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin of the Cartesian coordinate system -/
def origin : Point := ⟨0, 0⟩

/-- The coordinates of a point with respect to the origin -/
def coordsWrtOrigin (p : Point) : ℝ × ℝ := (p.x, p.y)

/-- Theorem: The coordinates of point A(-1,3) with respect to the origin are (-1,3) -/
theorem coords_of_A_wrt_origin :
  let A : Point := ⟨-1, 3⟩
  coordsWrtOrigin A = (-1, 3) := by sorry

end coords_of_A_wrt_origin_l4004_400424


namespace rock_volume_l4004_400447

/-- Calculates the volume of a rock based on the water level rise in a rectangular tank. -/
theorem rock_volume (tank_length tank_width water_rise : ℝ) 
  (h1 : tank_length = 30)
  (h2 : tank_width = 20)
  (h3 : water_rise = 4) :
  tank_length * tank_width * water_rise = 2400 := by
  sorry

#check rock_volume

end rock_volume_l4004_400447


namespace jennifer_remaining_money_l4004_400400

def initial_amount : ℚ := 120

def sandwich_fraction : ℚ := 1/5
def museum_fraction : ℚ := 1/6
def book_fraction : ℚ := 1/2

def remaining_amount : ℚ := 
  initial_amount - (initial_amount * sandwich_fraction + 
                    initial_amount * museum_fraction + 
                    initial_amount * book_fraction)

theorem jennifer_remaining_money : remaining_amount = 16 := by
  sorry

end jennifer_remaining_money_l4004_400400


namespace quadratic_inequality_solution_sets_l4004_400431

theorem quadratic_inequality_solution_sets 
  (c b a : ℝ) 
  (h : Set.Ioo (-3 : ℝ) (1/2) = {x : ℝ | c * x^2 + b * x + a < 0}) : 
  {x : ℝ | a * x^2 + b * x + c ≥ 0} = Set.Icc (-1/3 : ℝ) 2 := by
  sorry

end quadratic_inequality_solution_sets_l4004_400431


namespace exam_candidates_count_l4004_400454

theorem exam_candidates_count :
  ∀ (N : ℕ) (total_avg marks_11th : ℝ) (avg_first_10 avg_last_11 : ℝ),
    total_avg = 48 →
    avg_first_10 = 55 →
    avg_last_11 = 40 →
    marks_11th = 66 →
    N * total_avg = 10 * avg_first_10 + 11 * avg_last_11 - marks_11th →
    N = 21 := by
  sorry

end exam_candidates_count_l4004_400454


namespace kellys_vacation_duration_l4004_400467

/-- Kelly's vacation duration calculation -/
theorem kellys_vacation_duration :
  let travel_days : ℕ := 1 + 1 + 2 + 2  -- Sum of all travel days
  let stay_days : ℕ := 5 + 5 + 5        -- Sum of all stay days
  let total_days : ℕ := travel_days + stay_days
  let days_per_week : ℕ := 7
  (total_days / days_per_week : ℚ) = 3 := by
  sorry

end kellys_vacation_duration_l4004_400467


namespace quadratic_equation_solution_l4004_400472

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 - 16 = 0 ↔ x = 4 ∨ x = -4 := by sorry

end quadratic_equation_solution_l4004_400472


namespace sufficient_not_necessary_condition_l4004_400459

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  ¬(∀ a b, a^2 > b^2 → a > b ∧ b > 0) :=
by sorry

end sufficient_not_necessary_condition_l4004_400459


namespace no_matrix_satisfies_condition_l4004_400456

theorem no_matrix_satisfies_condition : 
  ¬∃ (N : Matrix (Fin 2) (Fin 2) ℝ), 
    ∀ (x y z w : ℝ), 
      N * !![x, y; z, w] = !![2*x, 3*y; 4*z, 5*w] := by
sorry

end no_matrix_satisfies_condition_l4004_400456


namespace x_range_l4004_400436

/-- The function f(x) = x^2 + ax -/
def f (x a : ℝ) : ℝ := x^2 + a*x

/-- The theorem stating the range of x given the conditions -/
theorem x_range (x : ℝ) :
  (∀ a ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f x a ≥ 3 - a) →
  (x ≤ -1 - Real.sqrt 2 ∨ x ≥ 1 + Real.sqrt 6) :=
by sorry

end x_range_l4004_400436


namespace triangle_properties_l4004_400471

theorem triangle_properties (a b c : ℝ) (A : ℝ) (area : ℝ) :
  (a + b + c) * (b + c - a) = 3 * b * c →
  a = 2 →
  area = Real.sqrt 3 →
  A = π / 3 ∧ b = 2 ∧ c = 2 := by
  sorry

end triangle_properties_l4004_400471


namespace multiplication_division_equivalence_l4004_400491

theorem multiplication_division_equivalence (x : ℝ) : 
  (x * (4/5)) / (2/7) = x * (14/5) := by
sorry

end multiplication_division_equivalence_l4004_400491


namespace simplify_fraction_l4004_400433

theorem simplify_fraction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (10 * a^2 * b) / (5 * a * b) = 2 * a :=
by sorry

end simplify_fraction_l4004_400433


namespace abs_neg_two_l4004_400497

theorem abs_neg_two : |(-2 : ℤ)| = 2 := by
  sorry

end abs_neg_two_l4004_400497


namespace movie_of_the_year_fraction_l4004_400478

theorem movie_of_the_year_fraction (total_members : ℕ) (min_appearances : ℚ) : 
  total_members = 795 → min_appearances = 198.75 → min_appearances / total_members = 1 / 4 := by
  sorry

end movie_of_the_year_fraction_l4004_400478


namespace negation_of_or_implies_both_false_l4004_400496

theorem negation_of_or_implies_both_false (p q : Prop) :
  (¬(p ∨ q)) → (¬p ∧ ¬q) := by
  sorry

end negation_of_or_implies_both_false_l4004_400496


namespace intersection_x_coordinate_l4004_400410

def f (x : ℝ) : ℝ := x^2

theorem intersection_x_coordinate 
  (A B C E : ℝ × ℝ) 
  (hA : A = (2, f 2)) 
  (hB : B = (8, f 8)) 
  (hC : C.1 = (A.1 + B.1) / 2 ∧ C.2 = (A.2 + B.2) / 2) 
  (hE : E.1^2 = E.2 ∧ E.2 = C.2) : 
  E.1 = Real.sqrt 34 := by
  sorry

end intersection_x_coordinate_l4004_400410


namespace range_of_k_l4004_400401

theorem range_of_k (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 + c^2 = 16) (h2 : b^2 + c^2 = 25) :
  let k := a^2 + b^2
  9 < k ∧ k < 41 := by sorry

end range_of_k_l4004_400401


namespace max_value_of_expression_l4004_400440

theorem max_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) :
  (∃ (x y z : ℝ), x^2 + y^2 + z^2 = 9 ∧ 
    (x - y)^2 + (y - z)^2 + (z - x)^2 ≥ (a - b)^2 + (b - c)^2 + (c - a)^2) ∧
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 9 → 
    (x - y)^2 + (y - z)^2 + (z - x)^2 ≤ 27) :=
by sorry

end max_value_of_expression_l4004_400440


namespace sum_of_digits_3_plus_4_pow_17_l4004_400425

/-- The sum of the tens digit and the ones digit of (3+4)^17 in integer form is 7 -/
theorem sum_of_digits_3_plus_4_pow_17 : 
  let n : ℕ := (3 + 4)^17
  let tens_digit : ℕ := (n / 10) % 10
  let ones_digit : ℕ := n % 10
  tens_digit + ones_digit = 7 := by
sorry

end sum_of_digits_3_plus_4_pow_17_l4004_400425


namespace twenty_is_eighty_percent_of_twentyfive_l4004_400408

theorem twenty_is_eighty_percent_of_twentyfive (x : ℝ) : 20 = 0.8 * x → x = 25 := by
  sorry

end twenty_is_eighty_percent_of_twentyfive_l4004_400408


namespace sum_and_average_of_squares_of_multiples_of_7_l4004_400414

def multiples_of_7 (n : ℕ) : List ℕ :=
  List.range n |>.map (· * 7 + 7)

def sum_of_squares (lst : List ℕ) : ℕ :=
  lst.map (· ^ 2) |>.sum

theorem sum_and_average_of_squares_of_multiples_of_7 :
  let lst := multiples_of_7 10
  let sum := sum_of_squares lst
  let avg := (sum : ℚ) / 10
  sum = 16865 ∧ avg = 1686.5 := by sorry

end sum_and_average_of_squares_of_multiples_of_7_l4004_400414


namespace systematic_sampling_20_4_l4004_400411

def is_systematic_sample (n : ℕ) (k : ℕ) (sample : List ℕ) : Prop :=
  sample.length = k ∧
  ∀ i, i ∈ sample → i ≤ n ∧
  ∀ i j, i < j → i ∈ sample → j ∈ sample → (j - i) = n / k

theorem systematic_sampling_20_4 :
  is_systematic_sample 20 4 [5, 10, 15, 20] := by
sorry

end systematic_sampling_20_4_l4004_400411


namespace squirrel_acorn_division_l4004_400452

theorem squirrel_acorn_division (total_acorns : ℕ) (acorns_per_month : ℕ) (spring_acorns : ℕ) : 
  total_acorns = 210 → acorns_per_month = 60 → spring_acorns = 30 →
  (total_acorns - 3 * acorns_per_month) / (total_acorns / 3 - acorns_per_month) = 3 :=
by
  sorry

end squirrel_acorn_division_l4004_400452


namespace unique_solution_gcd_system_l4004_400420

theorem unique_solution_gcd_system (a b c : ℕ+) :
  a + b = (Nat.gcd a b)^2 ∧
  b + c = (Nat.gcd b c)^2 ∧
  c + a = (Nat.gcd c a)^2 →
  a = 2 ∧ b = 2 ∧ c = 2 := by
  sorry

end unique_solution_gcd_system_l4004_400420


namespace collinear_points_sum_l4004_400407

/-- Three points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Collinearity of three points -/
def collinear (p q r : Point3D) : Prop :=
  ∃ (t s : ℝ), q.x - p.x = t * (r.x - p.x) ∧ 
                q.y - p.y = t * (r.y - p.y) ∧
                q.z - p.z = t * (r.z - p.z) ∧
                q.x - p.x = s * (r.x - q.x) ∧
                q.y - p.y = s * (r.y - q.y) ∧
                q.z - p.z = s * (r.z - q.z)

theorem collinear_points_sum (a b : ℝ) :
  collinear (Point3D.mk 2 a b) (Point3D.mk a 3 b) (Point3D.mk a b 4) →
  a + b = 6 := by
  sorry

end collinear_points_sum_l4004_400407


namespace unique_k_for_prime_roots_l4004_400446

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 0 → m < n → n % m ≠ 0

/-- The roots of a quadratic equation ax² + bx + c = 0 are given by the quadratic formula:
    x = (-b ± √(b² - 4ac)) / (2a) -/
def isRootOfQuadratic (x k : ℝ) : Prop := x^2 - 72*x + k = 0

theorem unique_k_for_prime_roots : 
  ∃! k : ℝ, ∃ p q : ℕ, 
    isPrime p ∧ 
    isPrime q ∧ 
    isRootOfQuadratic p k ∧ 
    isRootOfQuadratic q k ∧
    k = 335 := by sorry

end unique_k_for_prime_roots_l4004_400446


namespace probability_at_least_one_woman_l4004_400409

-- Define the total number of people
def total_people : ℕ := 12

-- Define the number of men
def num_men : ℕ := 8

-- Define the number of women
def num_women : ℕ := 4

-- Define the number of people to be selected
def num_selected : ℕ := 4

-- Define the probability of selecting at least one woman
def prob_at_least_one_woman : ℚ := 85 / 99

-- Theorem statement
theorem probability_at_least_one_woman :
  (1 : ℚ) - (num_men.choose num_selected : ℚ) / (total_people.choose num_selected : ℚ) = prob_at_least_one_woman :=
by sorry

end probability_at_least_one_woman_l4004_400409


namespace largest_three_digit_sum_l4004_400492

/-- A function that computes the sum ABC + CA + B -/
def digit_sum (A B C : ℕ) : ℕ := 101 * A + 11 * B + 11 * C

/-- A predicate that checks if three natural numbers are different digits -/
def are_different_digits (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A < 10 ∧ B < 10 ∧ C < 10

theorem largest_three_digit_sum :
  ∃ A B C : ℕ, are_different_digits A B C ∧ 
  digit_sum A B C = 986 ∧
  ∀ X Y Z : ℕ, are_different_digits X Y Z → 
  digit_sum X Y Z ≤ 986 ∧ digit_sum X Y Z < 1000 :=
sorry

end largest_three_digit_sum_l4004_400492


namespace variance_of_sick_cows_l4004_400416

/-- The variance of a binomial distribution with n trials and probability p --/
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

/-- The number of cows in the pasture --/
def num_cows : ℕ := 10

/-- The incidence rate of the disease --/
def incidence_rate : ℝ := 0.02

/-- Theorem stating that the variance of the number of sick cows is 0.196 --/
theorem variance_of_sick_cows :
  binomial_variance num_cows incidence_rate = 0.196 := by
  sorry

end variance_of_sick_cows_l4004_400416


namespace range_of_p_l4004_400470

def h (x : ℝ) : ℝ := 2 * x + 3

def p (x : ℝ) : ℝ := h (h (h (h x)))

theorem range_of_p :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3, 29 ≤ p x ∧ p x ≤ 93 :=
by
  sorry

end range_of_p_l4004_400470


namespace mars_other_elements_weight_l4004_400481

/-- The weight of the moon in tons -/
def moon_weight : ℝ := 250

/-- The ratio of iron in the moon's composition -/
def iron_ratio : ℝ := 0.5

/-- The ratio of carbon in the moon's composition -/
def carbon_ratio : ℝ := 0.2

/-- The ratio of Mars' weight to the moon's weight -/
def mars_moon_weight_ratio : ℝ := 2

theorem mars_other_elements_weight :
  let other_ratio : ℝ := 1 - iron_ratio - carbon_ratio
  let moon_other_weight : ℝ := other_ratio * moon_weight
  let mars_other_weight : ℝ := mars_moon_weight_ratio * moon_other_weight
  mars_other_weight = 150 := by sorry

end mars_other_elements_weight_l4004_400481


namespace line_segment_param_sum_squares_l4004_400413

/-- 
Given a line segment connecting (-2,7) and (3,11), parameterized by x = at + b and y = ct + d 
where 0 ≤ t ≤ 1 and t = 0 corresponds to (-2,7), the sum a^2 + b^2 + c^2 + d^2 equals 94.
-/
theorem line_segment_param_sum_squares : 
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    (a * t + b = -2 * (1 - t) + 3 * t) ∧ 
    (c * t + d = 7 * (1 - t) + 11 * t)) →
  b = -2 →
  d = 7 →
  a^2 + b^2 + c^2 + d^2 = 94 :=
by sorry

end line_segment_param_sum_squares_l4004_400413


namespace mile_to_yard_l4004_400466

-- Define the units
def mile : ℝ := 1
def furlong : ℝ := 1
def yard : ℝ := 1

-- Define the conversion factors
axiom mile_to_furlong : mile = 8 * furlong
axiom furlong_to_yard : furlong = 220 * yard

-- Theorem to prove
theorem mile_to_yard : mile = 1760 * yard := by
  sorry

end mile_to_yard_l4004_400466


namespace min_value_expression_l4004_400468

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x * y) ≥ 3 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ (((x₀^2 + y₀^2) * (4*x₀^2 + y₀^2)).sqrt) / (x₀ * y₀) = 3 :=
sorry

end min_value_expression_l4004_400468


namespace certain_number_proof_l4004_400432

theorem certain_number_proof : ∃! x : ℝ, x + (1/4 * 48) = 27 ∧ x = 15 := by
  sorry

end certain_number_proof_l4004_400432


namespace ellipse_tangent_min_length_l4004_400443

/-
  Define the ellipse C₁: x²/a² + y² = 1 (a > 1)
  where |F₁F₂|² is the arithmetic mean of |A₁A₂|² and |B₁B₂|²
-/
def C₁ (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 = 1 ∧ a > 1 ∧ 
  ∃ (b c : ℝ), 2 * (2*c)^2 = (2*a)^2 + (2*b)^2 ∧ b = 1

-- Define the curve C₂
def C₂ (t x y : ℝ) : Prop :=
  (x - t)^2 + y^2 = (t^2 + Real.sqrt 3 * t)^2 ∧ 0 < t ∧ t ≤ Real.sqrt 2 / 2

-- Define the tangent line l passing through the left vertex of C₁
def tangent_line (a t k : ℝ) : Prop :=
  ∃ (x y : ℝ), C₂ t x y ∧ y = k * (x + Real.sqrt 3)

-- Theorem statement
theorem ellipse_tangent_min_length :
  ∃ (a : ℝ), C₁ a (-Real.sqrt 3) 0 ∧
  (∀ x y, C₁ a x y ↔ x^2 / 3 + y^2 = 1) ∧
  (∀ t k, tangent_line a t k →
    ∃ (x y : ℝ), C₁ a x y ∧ y = k * (x + Real.sqrt 3) ∧
    ∀ (x' y' : ℝ), C₁ a x' y' ∧ y' = k * (x' + Real.sqrt 3) →
      (x - (-Real.sqrt 3))^2 + y^2 ≥ 3/2) :=
sorry

end ellipse_tangent_min_length_l4004_400443


namespace geometric_series_ratio_l4004_400482

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (hr : abs r < 1) :
  (a / (1 - r)) = 8 * (a * r^2 / (1 - r)) →
  r = 1 / (2 * Real.sqrt 2) ∨ r = -1 / (2 * Real.sqrt 2) := by
sorry

end geometric_series_ratio_l4004_400482


namespace fair_coin_tosses_l4004_400455

/-- 
Given a fair coin with probability 1/2 for each side, 
if the probability of landing on the same side n times is 1/16, 
then n must be 4.
-/
theorem fair_coin_tosses (n : ℕ) : 
  (1 / 2 : ℝ) ^ n = 1 / 16 → n = 4 := by sorry

end fair_coin_tosses_l4004_400455


namespace radio_operator_distribution_probability_radio_operator_distribution_probability_proof_l4004_400461

/-- The probability of each group having exactly one radio operator when 12 soldiers 
    (including 3 radio operators) are randomly divided into groups of 3, 4, and 5 soldiers. -/
theorem radio_operator_distribution_probability : ℝ :=
  let total_soldiers : ℕ := 12
  let radio_operators : ℕ := 3
  let group_sizes : List ℕ := [3, 4, 5]
  3 / 11

/-- Proof of the radio operator distribution probability theorem -/
theorem radio_operator_distribution_probability_proof :
  radio_operator_distribution_probability = 3 / 11 := by
  sorry

end radio_operator_distribution_probability_radio_operator_distribution_probability_proof_l4004_400461


namespace birdseed_supply_l4004_400402

/-- Represents a box of birdseed -/
structure BirdseedBox where
  totalAmount : ℕ
  typeAAmount : ℕ
  typeBAmount : ℕ

/-- Represents a bird's weekly seed consumption -/
structure BirdConsumption where
  totalAmount : ℕ
  typeAPercentage : ℚ
  typeBPercentage : ℚ

/-- The problem statement -/
theorem birdseed_supply (pantryBoxes : List BirdseedBox)
  (parrot cockatiel canary : BirdConsumption) :
  pantryBoxes.length = 5 →
  (pantryBoxes.map (·.typeAAmount)).sum ≥ 650 →
  (pantryBoxes.map (·.typeBAmount)).sum ≥ 675 →
  parrot.totalAmount = 100 ∧ parrot.typeAPercentage = 3/5 ∧ parrot.typeBPercentage = 2/5 →
  cockatiel.totalAmount = 50 ∧ cockatiel.typeAPercentage = 1/2 ∧ cockatiel.typeBPercentage = 1/2 →
  canary.totalAmount = 25 ∧ canary.typeAPercentage = 2/5 ∧ canary.typeBPercentage = 3/5 →
  ∃ (weeks : ℕ), weeks ≥ 6 ∧
    (pantryBoxes.map (·.typeAAmount)).sum ≥ weeks * (parrot.totalAmount * parrot.typeAPercentage +
      cockatiel.totalAmount * cockatiel.typeAPercentage +
      canary.totalAmount * canary.typeAPercentage) ∧
    (pantryBoxes.map (·.typeBAmount)).sum ≥ weeks * (parrot.totalAmount * parrot.typeBPercentage +
      cockatiel.totalAmount * cockatiel.typeBPercentage +
      canary.totalAmount * canary.typeBPercentage) := by
  sorry


end birdseed_supply_l4004_400402


namespace non_negative_iff_geq_zero_l4004_400405

theorem non_negative_iff_geq_zero (a b : ℝ) :
  (a ≥ 0 ∧ b ≥ 0) ↔ (a ≥ 0 ∧ b ≥ 0) := by
  sorry

end non_negative_iff_geq_zero_l4004_400405


namespace geometric_sequence_eighth_term_l4004_400475

theorem geometric_sequence_eighth_term 
  (a : ℝ) (r : ℝ) 
  (positive_sequence : ∀ n : ℕ, a * r^n > 0)
  (fourth_term : a * r^3 = 12)
  (twelfth_term : a * r^11 = 3) :
  a * r^7 = 6 * Real.sqrt 2 := by
sorry

end geometric_sequence_eighth_term_l4004_400475


namespace sequence_sum_l4004_400417

theorem sequence_sum (P Q R S T U V : ℝ) : 
  S = 7 ∧ 
  P + Q + R = 27 ∧ 
  Q + R + S = 27 ∧ 
  R + S + T = 27 ∧ 
  S + T + U = 27 ∧ 
  T + U + V = 27 → 
  P + V = 0 := by sorry

end sequence_sum_l4004_400417


namespace conference_handshakes_l4004_400483

def number_of_attendees : ℕ := 10

def handshake (a b : ℕ) : Prop := a ≠ b

def total_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

theorem conference_handshakes :
  total_handshakes number_of_attendees = 45 :=
sorry

end conference_handshakes_l4004_400483


namespace expression_factorization_l4004_400493

theorem expression_factorization (b : ℝ) :
  (4 * b^3 - 84 * b^2 - 12 * b) - (-3 * b^3 - 9 * b^2 + 3 * b) = b * (7 * b + 3) * (b - 5) := by
  sorry

end expression_factorization_l4004_400493


namespace bobbys_paycheck_l4004_400457

/-- Calculates Bobby's final paycheck amount --/
def calculate_paycheck (salary : ℝ) (performance_rate : ℝ) (federal_tax_rate : ℝ) 
  (state_tax_rate : ℝ) (local_tax_rate : ℝ) (health_insurance : ℝ) (life_insurance : ℝ) 
  (parking_fee : ℝ) (retirement_rate : ℝ) : ℝ :=
  let bonus := salary * performance_rate
  let total_income := salary + bonus
  let federal_tax := total_income * federal_tax_rate
  let state_tax := total_income * state_tax_rate
  let local_tax := total_income * local_tax_rate
  let total_taxes := federal_tax + state_tax + local_tax
  let other_deductions := health_insurance + life_insurance + parking_fee
  let retirement_contribution := salary * retirement_rate
  total_income - total_taxes - other_deductions - retirement_contribution

/-- Theorem stating that Bobby's final paycheck amount is $176.98 --/
theorem bobbys_paycheck : 
  calculate_paycheck 450 0.12 (1/3) 0.08 0.05 50 20 10 0.03 = 176.98 := by
  sorry

end bobbys_paycheck_l4004_400457


namespace x_range_for_quadratic_inequality_l4004_400422

theorem x_range_for_quadratic_inequality :
  (∀ m : ℝ, |m| ≤ 2 → ∀ x : ℝ, m * x^2 - 2 * x - m + 1 < 0) →
  ∃ a b : ℝ, a = (-1 + Real.sqrt 7) / 2 ∧ b = (1 + Real.sqrt 3) / 2 ∧
    ∀ x : ℝ, (a < x ∧ x < b) ↔ (∀ m : ℝ, |m| ≤ 2 → m * x^2 - 2 * x - m + 1 < 0) :=
by sorry

end x_range_for_quadratic_inequality_l4004_400422


namespace pool_wall_area_ratio_l4004_400415

theorem pool_wall_area_ratio :
  let pool_radius : ℝ := 20
  let wall_width : ℝ := 4
  let pool_area := π * pool_radius^2
  let total_area := π * (pool_radius + wall_width)^2
  let wall_area := total_area - pool_area
  wall_area / pool_area = 11 / 25 := by sorry

end pool_wall_area_ratio_l4004_400415


namespace cone_lateral_surface_area_l4004_400487

/-- Given a cone with base radius 4 cm and unfolded lateral surface radius 5 cm,
    prove that its lateral surface area is 20π cm². -/
theorem cone_lateral_surface_area :
  ∀ (base_radius unfolded_radius : ℝ),
    base_radius = 4 →
    unfolded_radius = 5 →
    let lateral_area := (1/2) * unfolded_radius^2 * (2 * Real.pi * base_radius / unfolded_radius)
    lateral_area = 20 * Real.pi :=
by sorry

end cone_lateral_surface_area_l4004_400487


namespace min_sum_abcd_l4004_400426

theorem min_sum_abcd (a b c d : ℕ) (h : a * b + b * c + c * d + d * a = 707) :
  a + b + c + d ≥ 108 := by
  sorry

end min_sum_abcd_l4004_400426


namespace hannah_dog_food_theorem_l4004_400484

/-- The amount of dog food Hannah needs to prepare daily for her three dogs -/
def total_dog_food (first_dog_food : ℝ) (second_dog_multiplier : ℝ) (third_dog_additional : ℝ) : ℝ :=
  first_dog_food + 
  (first_dog_food * second_dog_multiplier) + 
  (first_dog_food * second_dog_multiplier + third_dog_additional)

/-- Theorem stating that Hannah needs to prepare 10 cups of dog food daily -/
theorem hannah_dog_food_theorem : 
  total_dog_food 1.5 2 2.5 = 10 := by
  sorry

#eval total_dog_food 1.5 2 2.5

end hannah_dog_food_theorem_l4004_400484


namespace sunflower_height_l4004_400498

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Converts inches to feet, rounding down -/
def inches_to_feet (inches : ℕ) : ℕ :=
  inches / 12

theorem sunflower_height (sister_height_feet : ℕ) (sister_height_inches : ℕ) 
  (height_difference : ℕ) :
  sister_height_feet = 4 →
  sister_height_inches = 3 →
  height_difference = 21 →
  inches_to_feet (feet_inches_to_inches sister_height_feet sister_height_inches + height_difference) = 6 :=
by sorry

end sunflower_height_l4004_400498


namespace sandwich_cost_l4004_400450

theorem sandwich_cost (num_sandwiches num_drinks drink_cost total_cost : ℕ) 
  (h1 : num_sandwiches = 3)
  (h2 : num_drinks = 2)
  (h3 : drink_cost = 4)
  (h4 : total_cost = 26) :
  ∃ (sandwich_cost : ℕ), 
    sandwich_cost = 6 ∧ 
    num_sandwiches * sandwich_cost + num_drinks * drink_cost = total_cost := by
  sorry

end sandwich_cost_l4004_400450


namespace regular_nonagon_side_equals_diagonal_difference_l4004_400485

/-- A regular nonagon -/
structure RegularNonagon where
  -- Define the necessary properties of a regular nonagon
  side_length : ℝ
  longest_diagonal : ℝ
  shortest_diagonal : ℝ
  side_length_pos : 0 < side_length
  longest_diagonal_pos : 0 < longest_diagonal
  shortest_diagonal_pos : 0 < shortest_diagonal
  longest_ge_shortest : shortest_diagonal ≤ longest_diagonal

/-- 
The side length of a regular nonagon is equal to the difference 
between its longest diagonal and shortest diagonal 
-/
theorem regular_nonagon_side_equals_diagonal_difference 
  (n : RegularNonagon) : 
  n.side_length = n.longest_diagonal - n.shortest_diagonal :=
sorry

end regular_nonagon_side_equals_diagonal_difference_l4004_400485


namespace function_and_value_proof_l4004_400474

noncomputable section

-- Define the function f
def f (A : ℝ) (φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (x + φ)

-- State the theorem
theorem function_and_value_proof 
  (A : ℝ) (φ : ℝ) (α β : ℝ) 
  (h1 : A > 0) 
  (h2 : 0 < φ) (h3 : φ < π) 
  (h4 : ∀ x, f A φ x ≤ 1) 
  (h5 : f A φ (π/3) = 1/2) 
  (h6 : 0 < α) (h7 : α < π/2) 
  (h8 : 0 < β) (h9 : β < π/2) 
  (h10 : f A φ α = 3/5) 
  (h11 : f A φ β = 12/13) :
  (∀ x, f A φ x = Real.cos x) ∧ (f A φ (α - β) = 56/65) := by
  sorry

end

end function_and_value_proof_l4004_400474


namespace blueberries_per_box_l4004_400499

/-- The number of blueberries in each blue box -/
def B : ℕ := sorry

/-- The number of strawberries in each red box -/
def S : ℕ := sorry

/-- The difference between strawberries in a red box and blueberries in a blue box is 12 -/
axiom diff_strawberries_blueberries : S - B = 12

/-- Replacing one blue box with one red box increases the difference between total strawberries and total blueberries by 76 -/
axiom replacement_difference : 2 * S = 76

/-- The number of blueberries in each blue box is 26 -/
theorem blueberries_per_box : B = 26 := by sorry

end blueberries_per_box_l4004_400499


namespace equation_solution_l4004_400486

theorem equation_solution : ∃ x : ℝ, 7 * (4 * x + 3) - 9 = -3 * (2 - 9 * x) + 5 * x ∧ x = 4.5 := by
  sorry

end equation_solution_l4004_400486


namespace expression_evaluation_l4004_400480

theorem expression_evaluation :
  let x : ℚ := -3
  let numerator := 5 + x * (2 + x) - 2^2
  let denominator := x - 2 + x^2
  numerator / denominator = 1 := by sorry

end expression_evaluation_l4004_400480


namespace current_average_score_l4004_400479

/-- Represents the bonus calculation and test scores for Karen's class -/
structure TestScores where
  baseBonus : ℕ := 500
  bonusPerPoint : ℕ := 10
  baseScore : ℕ := 75
  maxScore : ℕ := 150
  gradedTests : ℕ := 8
  totalTests : ℕ := 10
  targetBonus : ℕ := 600
  lastTwoTestsScore : ℕ := 290

/-- The theorem states that given the conditions, the current average score of the graded tests is 70 -/
theorem current_average_score (ts : TestScores) : 
  (ts.targetBonus - ts.baseBonus) / ts.bonusPerPoint + ts.baseScore = 85 →
  ts.gradedTests * (((ts.targetBonus - ts.baseBonus) / ts.bonusPerPoint + ts.baseScore) * ts.totalTests - ts.lastTwoTestsScore) / ts.totalTests = 70 := by
  sorry

end current_average_score_l4004_400479


namespace imaginary_part_of_complex_fraction_l4004_400464

theorem imaginary_part_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (4 - 3*i) / i
  Complex.im z = -4 := by sorry

end imaginary_part_of_complex_fraction_l4004_400464


namespace min_sum_intercepts_l4004_400404

/-- The minimum sum of intercepts for a line passing through (1, 2) -/
theorem min_sum_intercepts : 
  ∀ a b : ℝ, a > 0 → b > 0 → 
  (1 : ℝ) / a + (2 : ℝ) / b = 1 → 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → (1 : ℝ) / a' + (2 : ℝ) / b' = 1 → a + b ≤ a' + b') → 
  a + b = 3 + 2 * Real.sqrt 2 := by
sorry

end min_sum_intercepts_l4004_400404


namespace range_of_a_l4004_400428

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ∈ Set.union (Set.Iic (-2)) {1} :=
sorry

end range_of_a_l4004_400428


namespace sweet_potatoes_remaining_l4004_400403

def sweet_potatoes_problem (initial : ℕ) (sold_adams : ℕ) (sold_lenon : ℕ) (traded : ℕ) (donated : ℕ) : ℕ :=
  initial - (sold_adams + sold_lenon + traded + donated)

theorem sweet_potatoes_remaining :
  sweet_potatoes_problem 80 20 15 10 5 = 30 := by
  sorry

end sweet_potatoes_remaining_l4004_400403


namespace percent_calculation_l4004_400476

theorem percent_calculation (x : ℝ) (h : 0.4 * x = 160) : 0.2 * x = 80 := by
  sorry

end percent_calculation_l4004_400476


namespace sunrise_is_certain_event_l4004_400494

-- Define the type for events
inductive Event
| TV : Event
| Dice : Event
| Sunrise : Event
| SeedGermination : Event

-- Define the property of being a certain event
def isCertainEvent (e : Event) : Prop :=
  match e with
  | Event.TV => False
  | Event.Dice => False
  | Event.Sunrise => True
  | Event.SeedGermination => False

-- Theorem statement
theorem sunrise_is_certain_event : isCertainEvent Event.Sunrise := by
  sorry

end sunrise_is_certain_event_l4004_400494


namespace factorization_equality_l4004_400451

theorem factorization_equality (x y : ℝ) : x^2 * y + 2 * x * y + y = y * (x + 1)^2 := by
  sorry

end factorization_equality_l4004_400451


namespace tom_helicopter_rental_cost_l4004_400423

/-- The total cost for renting a helicopter -/
def helicopter_rental_cost (hours_per_day : ℕ) (days : ℕ) (hourly_rate : ℕ) : ℕ :=
  hours_per_day * days * hourly_rate

/-- Theorem stating the total cost for Tom's helicopter rental -/
theorem tom_helicopter_rental_cost :
  helicopter_rental_cost 2 3 75 = 450 := by
  sorry

end tom_helicopter_rental_cost_l4004_400423


namespace total_lemons_picked_l4004_400448

theorem total_lemons_picked (sally_lemons mary_lemons : ℕ) 
  (h1 : sally_lemons = 7)
  (h2 : mary_lemons = 9) :
  sally_lemons + mary_lemons = 16 := by
sorry

end total_lemons_picked_l4004_400448


namespace perpendicular_line_equation_l4004_400463

/-- A line passing through (1, -1) and perpendicular to 3x - 2y = 0 has the equation 2x + 3y + 1 = 0 -/
theorem perpendicular_line_equation :
  ∀ (x y : ℝ),
  (2 * x + 3 * y + 1 = 0) ↔
  (∃ (m : ℝ), (y - (-1) = m * (x - 1)) ∧ 
              (m * 3 = -1/2) ∧
              (2 * 1 + 3 * (-1) + 1 = 0)) :=
by sorry

end perpendicular_line_equation_l4004_400463


namespace sandwich_count_l4004_400427

def num_bread_types : ℕ := 12
def num_spread_types : ℕ := 10

def sandwich_combinations : ℕ := num_bread_types * (num_spread_types.choose 2)

theorem sandwich_count : sandwich_combinations = 540 := by
  sorry

end sandwich_count_l4004_400427


namespace negative_three_squared_opposite_l4004_400489

/-- Two real numbers are opposite if their sum is zero -/
def are_opposite (a b : ℝ) : Prop := a + b = 0

/-- Theorem stating that (-3)² and -3² are opposite numbers -/
theorem negative_three_squared_opposite : are_opposite ((-3)^2) (-3^2) := by
  sorry

end negative_three_squared_opposite_l4004_400489


namespace race_heartbeats_l4004_400439

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Proves that the total number of heartbeats during the specified race is 28800 -/
theorem race_heartbeats :
  total_heartbeats 160 30 6 = 28800 := by
  sorry

#eval total_heartbeats 160 30 6

end race_heartbeats_l4004_400439


namespace pump_x_portion_l4004_400444

/-- Represents the pumping scenario with two pumps -/
structure PumpingScenario where
  total_water : ℝ
  pump_x_rate : ℝ
  pump_y_rate : ℝ

/-- The conditions of the pumping scenario -/
def pumping_conditions (s : PumpingScenario) : Prop :=
  s.pump_x_rate > 0 ∧
  s.pump_y_rate > 0 ∧
  3 * s.pump_x_rate + 3 * (s.pump_x_rate + s.pump_y_rate) = s.total_water ∧
  20 * s.pump_y_rate = s.total_water

/-- The theorem stating that Pump X pumps out 17/40 of the total water in the first 3 hours -/
theorem pump_x_portion (s : PumpingScenario) 
  (h : pumping_conditions s) : 
  3 * s.pump_x_rate = (17 / 40) * s.total_water := by
  sorry

end pump_x_portion_l4004_400444


namespace equation_equivalence_l4004_400445

theorem equation_equivalence (x : ℝ) : x * (2 * x - 1) = 5 * (x + 3) ↔ 2 * x^2 - 6 * x - 15 = 0 := by
  sorry

end equation_equivalence_l4004_400445


namespace quadratic_coefficients_l4004_400477

/-- A quadratic function with a vertex at (-1, -3) -/
def f (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

/-- The vertex of the quadratic function is at (-1, -3) -/
def has_vertex (b c : ℝ) : Prop :=
  (∀ x, f b c x ≤ f b c (-1)) ∧ (f b c (-1) = -3)

/-- Theorem stating that b = -2 and c = -4 for the given quadratic function -/
theorem quadratic_coefficients :
  ∃ b c : ℝ, has_vertex b c ∧ b = -2 ∧ c = -4 := by sorry

end quadratic_coefficients_l4004_400477


namespace smallest_common_multiple_l4004_400469

theorem smallest_common_multiple : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 6 = 0 ∧ n % 5 = 0 ∧ n % 8 = 0 ∧ n % 9 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 6 = 0 ∧ m % 5 = 0 ∧ m % 8 = 0 ∧ m % 9 = 0 → m ≥ n) ∧
  n = 360 :=
by sorry

end smallest_common_multiple_l4004_400469


namespace max_teams_tied_for_most_wins_l4004_400430

/-- Represents a round-robin tournament --/
structure Tournament :=
  (num_teams : ℕ)
  (wins : Fin num_teams → ℕ)

/-- The total number of games in a round-robin tournament --/
def total_games (t : Tournament) : ℕ :=
  t.num_teams * (t.num_teams - 1) / 2

/-- The maximum number of wins for any team in the tournament --/
def max_wins (t : Tournament) : ℕ :=
  Finset.sup Finset.univ t.wins

/-- The number of teams tied for the maximum number of wins --/
def num_teams_with_max_wins (t : Tournament) : ℕ :=
  Finset.card (Finset.filter (λ i => t.wins i = max_wins t) Finset.univ)

/-- The main theorem --/
theorem max_teams_tied_for_most_wins :
  ∃ (t : Tournament), t.num_teams = 8 ∧
  (∀ (t' : Tournament), t'.num_teams = 8 →
    num_teams_with_max_wins t' ≤ num_teams_with_max_wins t) ∧
  num_teams_with_max_wins t = 7 :=
sorry

end max_teams_tied_for_most_wins_l4004_400430


namespace arithmetic_sequence_common_difference_l4004_400449

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h : ∀ n, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence with a₂ = 3 and a₅ = 6 is 1. -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h₂ : seq.a 2 = 3)
  (h₅ : seq.a 5 = 6) :
  seq.d = 1 := by
  sorry

end arithmetic_sequence_common_difference_l4004_400449
