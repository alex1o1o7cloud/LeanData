import Mathlib

namespace smaller_two_digit_factor_l1825_182510

theorem smaller_two_digit_factor (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 3774 → 
  min a b = 51 := by
sorry

end smaller_two_digit_factor_l1825_182510


namespace sum_of_a_values_l1825_182550

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : ℝ := 4 * x^2 + a * x + 8 * x + 9

-- Define the condition for the equation to have only one solution
def has_one_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, quadratic_equation a x = 0

-- Define the set of 'a' values that satisfy the condition
def a_values : Set ℝ := {a | has_one_solution a}

-- State the theorem
theorem sum_of_a_values :
  ∃ a₁ a₂ : ℝ, a₁ ∈ a_values ∧ a₂ ∈ a_values ∧ a₁ ≠ a₂ ∧ a₁ + a₂ = -16 :=
sorry

end sum_of_a_values_l1825_182550


namespace students_travel_speed_l1825_182538

/-- Proves that given the conditions of the problem, student B's bicycle speed is 14.4 km/h -/
theorem students_travel_speed (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ)
  (h_distance : distance = 2.4)
  (h_speed_ratio : speed_ratio = 4)
  (h_time_difference : time_difference = 0.5) :
  let walking_speed := distance / (distance / (speed_ratio * walking_speed) + time_difference)
  speed_ratio * walking_speed = 14.4 := by
  sorry

end students_travel_speed_l1825_182538


namespace romans_remaining_coins_l1825_182562

/-- Represents the problem of calculating Roman's remaining gold coins --/
theorem romans_remaining_coins 
  (initial_worth : ℕ) 
  (coins_sold : ℕ) 
  (money_after_sale : ℕ) 
  (h1 : initial_worth = 20)
  (h2 : coins_sold = 3)
  (h3 : money_after_sale = 12) :
  initial_worth / (money_after_sale / coins_sold) - coins_sold = 2 :=
sorry

end romans_remaining_coins_l1825_182562


namespace greatest_integer_with_gcd_six_greatest_integer_with_gcd_six_exists_l1825_182586

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 150 ∧ Nat.gcd n 18 = 6 → n ≤ 132 :=
by sorry

theorem greatest_integer_with_gcd_six_exists : ∃ n : ℕ, n = 132 ∧ n < 150 ∧ Nat.gcd n 18 = 6 :=
by sorry

end greatest_integer_with_gcd_six_greatest_integer_with_gcd_six_exists_l1825_182586


namespace pencil_difference_l1825_182558

/-- The price of a single pencil in dollars -/
def pencil_price : ℚ := 0.04

/-- The number of pencils Jamar bought -/
def jamar_pencils : ℕ := 81

/-- The number of pencils Michael bought -/
def michael_pencils : ℕ := 104

/-- The amount Jamar paid in dollars -/
def jamar_paid : ℚ := 2.32

/-- The amount Michael paid in dollars -/
def michael_paid : ℚ := 3.24

theorem pencil_difference : 
  (pencil_price > 0.01) ∧ 
  (jamar_paid = pencil_price * jamar_pencils) ∧
  (michael_paid = pencil_price * michael_pencils) ∧
  (∃ n : ℕ, n^2 = jamar_pencils) →
  michael_pencils - jamar_pencils = 23 := by
sorry

end pencil_difference_l1825_182558


namespace vertex_to_center_equals_side_length_l1825_182567

/-- A regular hexagon with side length 16 units -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : Bool)
  (side_length_eq_16 : side_length = 16)

/-- The length of a segment from a vertex to the center of a regular hexagon -/
def vertex_to_center_length (h : RegularHexagon) : ℝ := sorry

/-- Theorem: The length of a segment from a vertex to the center of a regular hexagon
    with side length 16 units is equal to 16 units -/
theorem vertex_to_center_equals_side_length (h : RegularHexagon) :
  vertex_to_center_length h = h.side_length :=
sorry

end vertex_to_center_equals_side_length_l1825_182567


namespace S_equals_seven_l1825_182504

noncomputable def S : ℝ :=
  1 / (4 - Real.sqrt 15) -
  1 / (Real.sqrt 15 - Real.sqrt 14) +
  1 / (Real.sqrt 14 - Real.sqrt 13) -
  1 / (Real.sqrt 13 - Real.sqrt 12) +
  1 / (Real.sqrt 12 - 3)

theorem S_equals_seven : S = 7 := by
  sorry

end S_equals_seven_l1825_182504


namespace paper_I_maximum_mark_l1825_182593

/-- The maximum mark for paper I -/
def maximum_mark : ℕ := 186

/-- The passing percentage as a rational number -/
def passing_percentage : ℚ := 35 / 100

/-- The marks scored by the candidate -/
def scored_marks : ℕ := 42

/-- The marks by which the candidate failed -/
def failing_margin : ℕ := 23

/-- Theorem stating the maximum mark for paper I -/
theorem paper_I_maximum_mark :
  (↑maximum_mark * passing_percentage).floor = scored_marks + failing_margin :=
sorry

end paper_I_maximum_mark_l1825_182593


namespace power_equation_solution_l1825_182539

theorem power_equation_solution (n : ℕ) : 5^29 * 4^15 = 2 * 10^n → n = 29 := by
  sorry

end power_equation_solution_l1825_182539


namespace milk_division_l1825_182546

theorem milk_division (total_milk : ℚ) (num_kids : ℕ) (milk_per_kid : ℚ) : 
  total_milk = 3 → 
  num_kids = 5 → 
  milk_per_kid = total_milk / num_kids → 
  milk_per_kid = 3 / 5 := by
  sorry

end milk_division_l1825_182546


namespace negation_of_universal_quantifier_l1825_182591

theorem negation_of_universal_quantifier :
  (¬ (∀ x : ℝ, x^2 - x + 1/4 ≥ 0)) ↔ (∃ x : ℝ, x^2 - x + 1/4 < 0) :=
by sorry

end negation_of_universal_quantifier_l1825_182591


namespace cube_after_carving_l1825_182560

def cube_side_length : ℝ := 9

-- Volume of the cube after carving the cross-shaped groove
def remaining_volume : ℝ := 639

-- Surface area of the cube after carving the cross-shaped groove
def new_surface_area : ℝ := 510

-- Theorem statement
theorem cube_after_carving (groove_volume : ℝ) (groove_surface_area : ℝ) :
  cube_side_length ^ 3 - groove_volume = remaining_volume ∧
  6 * cube_side_length ^ 2 + groove_surface_area = new_surface_area :=
by sorry

end cube_after_carving_l1825_182560


namespace subset_intersection_condition_l1825_182585

theorem subset_intersection_condition (n : ℕ) (h : n ≥ 4) :
  (∀ (S : Finset (Finset (Fin n))) (h_card : S.card = n) 
    (h_subsets : ∀ s ∈ S, s.card = 3),
    ∃ (s1 s2 : Finset (Fin n)), s1 ∈ S ∧ s2 ∈ S ∧ s1 ≠ s2 ∧ (s1 ∩ s2).card = 1) ↔
  n % 4 ≠ 0 :=
sorry

end subset_intersection_condition_l1825_182585


namespace perfect_square_power_of_two_plus_33_l1825_182508

theorem perfect_square_power_of_two_plus_33 :
  ∀ n : ℕ, (∃ m : ℕ, 2^n + 33 = m^2) ↔ n = 4 ∨ n = 8 := by
  sorry

end perfect_square_power_of_two_plus_33_l1825_182508


namespace gracie_number_l1825_182555

/-- Represents the counting pattern for a student --/
def student_count (n : ℕ) : Set ℕ :=
  {m | m ≤ 2000 ∧ m ≠ 0 ∧ ∃ k, m = 5*k + 1 ∨ m = 5*k + 2 ∨ m = 5*k + 4 ∨ m = 5*k + 5}

/-- Represents the numbers skipped by a student --/
def student_skip (n : ℕ) : Set ℕ :=
  {m | m ≤ 2000 ∧ m ≠ 0 ∧ ∃ k, m = 5^n * (5*k - 2)}

/-- The set of numbers said by the first n students --/
def numbers_said (n : ℕ) : Set ℕ :=
  if n = 0 then ∅ else (student_count n) ∪ (numbers_said (n-1)) \ (student_skip n)

theorem gracie_number :
  ∃! x, x ∈ {m | 1 ≤ m ∧ m ≤ 2000} \ (numbers_said 7) ∧ x = 1623 :=
sorry

end gracie_number_l1825_182555


namespace triangle_angle_cosine_l1825_182577

theorem triangle_angle_cosine (A B C : ℝ) (a b c : ℝ) (h1 : a = 4) (h2 : b = 2) (h3 : c = 3) 
  (h4 : B = 2 * C) (h5 : A + B + C = π) : Real.cos C = 11/16 := by
  sorry

end triangle_angle_cosine_l1825_182577


namespace cylinder_volume_increase_l1825_182595

theorem cylinder_volume_increase (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  let new_radius := 4 * r
  let new_height := 3 * h
  (π * new_radius^2 * new_height) / (π * r^2 * h) = 48 := by sorry

end cylinder_volume_increase_l1825_182595


namespace inequality_solution_l1825_182565

theorem inequality_solution (x : ℝ) :
  x ≤ 4 ∧ |2*x - 3| + |x + 1| < 7 → -5/3 < x ∧ x < 3 := by
  sorry

end inequality_solution_l1825_182565


namespace product_cde_eq_1000_l1825_182584

theorem product_cde_eq_1000 
  (a b c d e f : ℝ) 
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 0.5) :
  c * d * e = 1000 := by
sorry

end product_cde_eq_1000_l1825_182584


namespace sum_of_roots_l1825_182582

theorem sum_of_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x / Real.sqrt y - y / Real.sqrt x = 7 / 12)
  (h2 : x - y = 7) : x + y = 25 := by
  sorry

end sum_of_roots_l1825_182582


namespace f_e_plus_f_prime_e_l1825_182520

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem f_e_plus_f_prime_e : f (Real.exp 1) + (deriv f) (Real.exp 1) = 2 * Real.exp (Real.exp 1) := by
  sorry

end f_e_plus_f_prime_e_l1825_182520


namespace geometric_sum_5_quarters_l1825_182523

def geometric_sum (a₀ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₀ * (1 - r^n) / (1 - r)

theorem geometric_sum_5_quarters : 
  geometric_sum (1/4) (1/4) 5 = 341/1024 := by
  sorry

end geometric_sum_5_quarters_l1825_182523


namespace complement_intersection_problem_l1825_182544

def I : Set ℤ := {x | -3 < x ∧ x < 3}
def A : Set ℤ := {-2, 0, 1}
def B : Set ℤ := {-1, 0, 1, 2}

theorem complement_intersection_problem : (I \ A) ∩ B = {-1, 2} := by sorry

end complement_intersection_problem_l1825_182544


namespace fraction_sum_l1825_182563

theorem fraction_sum : (3 : ℚ) / 8 + 9 / 12 + 5 / 6 = 47 / 24 := by
  sorry

end fraction_sum_l1825_182563


namespace negation_equivalence_l1825_182569

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ (∀ x : ℝ, -1 < x ∧ x < 2) := by sorry

end negation_equivalence_l1825_182569


namespace quadratic_with_zero_root_l1825_182556

/-- Given a quadratic equation (k-2)x^2 + x + k^2 - 4 = 0 where 0 is one of its roots,
    prove that k = -2 -/
theorem quadratic_with_zero_root (k : ℝ) : 
  (∀ x : ℝ, (k - 2) * x^2 + x + k^2 - 4 = 0 ↔ x = 0 ∨ x = (k^2 - 4) / (2 - k)) →
  ((k - 2) * 0^2 + 0 + k^2 - 4 = 0) →
  k = -2 := by
  sorry

end quadratic_with_zero_root_l1825_182556


namespace subset_implies_a_range_l1825_182589

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 2)^2 ≤ 5/4}
def B (a : ℝ) : Set (ℝ × ℝ) := {p | |p.1 - 1| + 2*|p.2 - 2| ≤ a}

-- State the theorem
theorem subset_implies_a_range (a : ℝ) : A ⊆ B a → a ≥ 5/2 := by
  sorry

end subset_implies_a_range_l1825_182589


namespace quadratic_equation_solutions_l1825_182503

theorem quadratic_equation_solutions (x : ℝ) :
  x^2 = 8*x - 15 →
  (∃ s p : ℝ, s = 8 ∧ p = 15 ∧
    (∀ x₁ x₂ : ℝ, x₁^2 = 8*x₁ - 15 ∧ x₂^2 = 8*x₂ - 15 → x₁ + x₂ = s ∧ x₁ * x₂ = p)) :=
by sorry

end quadratic_equation_solutions_l1825_182503


namespace dog_accessible_area_l1825_182506

/-- Represents the shed's dimensions and rope configuration --/
structure DogTieSetup where
  shedSideLength : ℝ
  ropeLength : ℝ
  attachmentDistance : ℝ

/-- Calculates the area accessible to the dog --/
def accessibleArea (setup : DogTieSetup) : ℝ :=
  sorry

/-- Theorem stating the area accessible to the dog --/
theorem dog_accessible_area (setup : DogTieSetup) 
  (h1 : setup.shedSideLength = 30)
  (h2 : setup.ropeLength = 10)
  (h3 : setup.attachmentDistance = 5) :
  accessibleArea setup = 37.5 * Real.pi := by
  sorry

end dog_accessible_area_l1825_182506


namespace angle_complement_when_supplement_is_110_l1825_182517

/-- If the supplement of an angle is 110°, then its complement is 20°. -/
theorem angle_complement_when_supplement_is_110 (x : ℝ) : 
  x + 110 = 180 → 90 - (180 - 110) = 20 := by
  sorry

end angle_complement_when_supplement_is_110_l1825_182517


namespace hannah_stocking_stuffers_l1825_182512

/-- The number of candy canes per stocking -/
def candy_canes_per_stocking : ℕ := 4

/-- The number of beanie babies per stocking -/
def beanie_babies_per_stocking : ℕ := 2

/-- The number of books per stocking -/
def books_per_stocking : ℕ := 1

/-- The number of kids Hannah has -/
def number_of_kids : ℕ := 3

/-- The total number of stocking stuffers Hannah buys -/
def total_stocking_stuffers : ℕ := 
  (candy_canes_per_stocking + beanie_babies_per_stocking + books_per_stocking) * number_of_kids

theorem hannah_stocking_stuffers : total_stocking_stuffers = 21 := by
  sorry

end hannah_stocking_stuffers_l1825_182512


namespace complex_magnitude_l1825_182519

theorem complex_magnitude (a : ℝ) (z : ℂ) : 
  z = a + Complex.I ∧ z^2 + z = 1 - 3*Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l1825_182519


namespace gcf_40_56_l1825_182554

theorem gcf_40_56 : Nat.gcd 40 56 = 8 := by
  sorry

end gcf_40_56_l1825_182554


namespace squares_below_line_l1825_182597

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of integer points strictly below a line in the first quadrant -/
def countPointsBelowLine (l : Line) : ℕ :=
  sorry

/-- The specific line from the problem -/
def problemLine : Line :=
  { a := 12, b := 180, c := 2160 }

/-- The theorem statement -/
theorem squares_below_line :
  countPointsBelowLine problemLine = 1969 := by
  sorry

end squares_below_line_l1825_182597


namespace smallest_x_value_l1825_182545

theorem smallest_x_value (x : ℝ) : 
  (5 * x^2 + 7 * x + 3 = 6) → x ≥ -3 :=
by sorry

end smallest_x_value_l1825_182545


namespace area_of_union_equals_20_5_l1825_182501

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point about the line y = x -/
def reflect (p : Point) : Point :=
  { x := p.y, y := p.x }

/-- Calculates the area of a triangle given its three vertices using the shoelace formula -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- The main theorem stating the area of the union of the original and reflected triangles -/
theorem area_of_union_equals_20_5 :
  let A : Point := { x := 3, y := 4 }
  let B : Point := { x := 5, y := -2 }
  let C : Point := { x := 7, y := 3 }
  let A' := reflect A
  let B' := reflect B
  let C' := reflect C
  triangleArea A B C + triangleArea A' B' C' = 20.5 := by
  sorry

end area_of_union_equals_20_5_l1825_182501


namespace motel_weekly_charge_l1825_182548

/-- The weekly charge for Casey's motel stay --/
def weekly_charge : ℕ → Prop :=
  fun w => 
    let months : ℕ := 3
    let weeks_per_month : ℕ := 4
    let monthly_rate : ℕ := 1000
    let savings : ℕ := 360
    let total_weeks : ℕ := months * weeks_per_month
    let total_monthly_cost : ℕ := months * monthly_rate
    (total_weeks * w = total_monthly_cost + savings) ∧ (w = 280)

/-- Proof that the weekly charge is $280 --/
theorem motel_weekly_charge : weekly_charge 280 := by
  sorry

end motel_weekly_charge_l1825_182548


namespace xyz_equals_five_l1825_182525

theorem xyz_equals_five (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) : 
  x * y * z = 5 := by
sorry

end xyz_equals_five_l1825_182525


namespace pen_purchase_theorem_l1825_182592

def budget : ℕ := 31
def price1 : ℕ := 2
def price2 : ℕ := 3
def price3 : ℕ := 4

def max_pens (b p1 p2 p3 : ℕ) : ℕ :=
  (b - p2 - p3) / p1 + 2

def min_pens (b p1 p2 p3 : ℕ) : ℕ :=
  (b - p1 - p2) / p3 + 3

theorem pen_purchase_theorem :
  max_pens budget price1 price2 price3 = 14 ∧
  min_pens budget price1 price2 price3 = 9 :=
by sorry

end pen_purchase_theorem_l1825_182592


namespace distribution_schemes_l1825_182541

/-- The number of ways to distribute students among projects -/
def distribute_students (n_students : ℕ) (n_projects : ℕ) : ℕ :=
  -- Number of ways to choose 2 students from n_students
  (n_students.choose 2) * 
  -- Number of ways to permute n_projects
  (n_projects.factorial)

/-- Theorem stating the number of distribution schemes -/
theorem distribution_schemes :
  distribute_students 5 4 = 240 :=
sorry

end distribution_schemes_l1825_182541


namespace min_value_a_l1825_182587

theorem min_value_a (a : ℕ) (h : 17 ∣ (50^2023 + a)) : 
  ∀ b : ℕ, (17 ∣ (50^2023 + b)) → a ≤ b → 18 ≤ a := by
sorry

end min_value_a_l1825_182587


namespace locus_of_midpoint_is_circle_l1825_182583

/-- Given a circle with center O and radius R, and a point P inside the circle,
    we rotate a right angle around P. The legs of the right angle intersect
    the circle at points A and B. This theorem proves that the locus of the
    midpoint of chord AB is a circle. -/
theorem locus_of_midpoint_is_circle
  (O : ℝ × ℝ)  -- Center of the circle
  (R : ℝ)      -- Radius of the circle
  (P : ℝ × ℝ)  -- Point inside the circle
  (h_R_pos : R > 0)  -- R is positive
  (h_P_inside : dist P O < R)  -- P is inside the circle
  (A B : ℝ × ℝ)  -- Points on the circle
  (h_A_on_circle : dist A O = R)  -- A is on the circle
  (h_B_on_circle : dist B O = R)  -- B is on the circle
  (h_right_angle : (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = 0)  -- ∠APB is a right angle
  : ∃ (C : ℝ × ℝ) (r : ℝ),
    let a := dist P O
    let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)  -- Midpoint of AB
    C = (a / 2, 0) ∧ r = (1 / 2) * Real.sqrt (2 * R^2 - a^2) ∧
    dist M C = r :=
by sorry

end locus_of_midpoint_is_circle_l1825_182583


namespace geometric_sequence_ratio_l1825_182598

theorem geometric_sequence_ratio (a : ℝ) (r : ℝ) (h1 : a > 0) (h2 : r > 0) :
  a + a * r + a * r^2 + a * r^3 = 5 * (a + a * r) →
  r = 2 := by
sorry

end geometric_sequence_ratio_l1825_182598


namespace samantha_birth_year_l1825_182575

-- Define the year of the first AMC 8
def first_amc8_year : ℕ := 1985

-- Define the frequency of AMC 8 (every 2 years)
def amc8_frequency : ℕ := 2

-- Define Samantha's age when she took the fourth AMC 8
def samantha_age_fourth_amc8 : ℕ := 12

-- Function to calculate the year of the nth AMC 8
def nth_amc8_year (n : ℕ) : ℕ :=
  first_amc8_year + (n - 1) * amc8_frequency

-- Theorem to prove Samantha's birth year
theorem samantha_birth_year :
  nth_amc8_year 4 - samantha_age_fourth_amc8 = 1981 :=
by sorry

end samantha_birth_year_l1825_182575


namespace fraction_multiplication_l1825_182596

theorem fraction_multiplication : (2 : ℚ) / 3 * 4 / 7 * 9 / 11 * 5 / 8 = 15 / 77 := by
  sorry

end fraction_multiplication_l1825_182596


namespace percentage_problem_l1825_182534

theorem percentage_problem (y : ℝ) (h1 : y > 0) (h2 : y * (y / 100) = 9) : y = 30 := by
  sorry

end percentage_problem_l1825_182534


namespace compare_expressions_compare_square_roots_l1825_182574

-- Problem 1
theorem compare_expressions (x y : ℝ) : x^2 + y^2 + 1 > 2*(x + y - 1) := by
  sorry

-- Problem 2
theorem compare_square_roots (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) :
  Real.sqrt (a / d) > Real.sqrt (b / c) := by
  sorry

end compare_expressions_compare_square_roots_l1825_182574


namespace housing_boom_construction_l1825_182524

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before : ℕ := 1426

/-- The number of houses in Lawrence County after the housing boom -/
def houses_after : ℕ := 2000

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := houses_after - houses_before

theorem housing_boom_construction :
  houses_built = 574 :=
by sorry

end housing_boom_construction_l1825_182524


namespace cody_final_tickets_l1825_182522

/-- Calculates the final number of tickets Cody has after various transactions at the arcade. -/
def final_tickets (initial_tickets : ℕ) (won_tickets : ℕ) (beanie_cost : ℕ) (traded_tickets : ℕ) (games_played : ℕ) (tickets_per_game : ℕ) : ℕ :=
  initial_tickets + won_tickets - beanie_cost - traded_tickets + (games_played * tickets_per_game)

/-- Theorem stating that Cody ends up with 82 tickets given the specific conditions of the problem. -/
theorem cody_final_tickets :
  final_tickets 50 49 25 10 3 6 = 82 := by sorry

end cody_final_tickets_l1825_182522


namespace unique_solution_condition_l1825_182507

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 4) ↔ d ≠ 4 := by
sorry

end unique_solution_condition_l1825_182507


namespace segment_ratio_l1825_182553

/-- Given a line segment AD with points B and C on it, where AB = 3BD and AC = 5CD,
    the length of BC is 1/12 of the length of AD. -/
theorem segment_ratio (A B C D : ℝ) (h1 : A ≤ B) (h2 : B ≤ C) (h3 : C ≤ D)
  (h4 : B - A = 3 * (D - B)) (h5 : C - A = 5 * (D - C)) :
  (C - B) = (1 / 12) * (D - A) := by
  sorry

end segment_ratio_l1825_182553


namespace ratio_sum_problem_l1825_182530

theorem ratio_sum_problem (x y z a : ℚ) : 
  x / y = 3 / 4 →
  y / z = 4 / 6 →
  y = 15 * a + 5 →
  x + y + z = 52 →
  a = 11 / 15 := by
sorry

end ratio_sum_problem_l1825_182530


namespace eggplant_basket_weight_l1825_182535

def cucumber_baskets : ℕ := 25
def eggplant_baskets : ℕ := 32
def total_weight : ℕ := 1870
def cucumber_basket_weight : ℕ := 30

theorem eggplant_basket_weight :
  (total_weight - cucumber_baskets * cucumber_basket_weight) / eggplant_baskets =
  (1870 - 25 * 30) / 32 := by
  sorry

end eggplant_basket_weight_l1825_182535


namespace max_value_f_range_of_a_l1825_182573

-- Define the function f(x) = x^2 - 2ax + 2
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Theorem 1: Maximum value of f(x) when a = 1 and x ∈ [-1, 2]
theorem max_value_f (x : ℝ) (h : x ∈ Set.Icc (-1) 2) : 
  f 1 x ≤ 5 :=
sorry

-- Theorem 2: Range of a when f(x) ≥ a for x ∈ [-1, +∞)
theorem range_of_a (a : ℝ) :
  (∀ x ≥ -1, f a x ≥ a) ↔ a ∈ Set.Icc (-3) 1 :=
sorry

end max_value_f_range_of_a_l1825_182573


namespace newlandia_density_l1825_182566

/-- Represents the population and area data for a country -/
structure CountryData where
  population : ℕ
  area_sq_miles : ℕ

/-- Calculates the average square feet per person in a country -/
def avg_sq_feet_per_person (country : CountryData) : ℚ :=
  (country.area_sq_miles * (5280 * 5280) : ℚ) / country.population

/-- Theorem stating the properties of Newlandia's population density -/
theorem newlandia_density (newlandia : CountryData) 
  (h1 : newlandia.population = 350000000)
  (h2 : newlandia.area_sq_miles = 4500000) :
  let density := avg_sq_feet_per_person newlandia
  (358000 : ℚ) < density ∧ density < (359000 : ℚ) ∧ density > 700 := by
  sorry

#eval avg_sq_feet_per_person ⟨350000000, 4500000⟩

end newlandia_density_l1825_182566


namespace prime_divisor_of_binomial_coefficients_l1825_182564

theorem prime_divisor_of_binomial_coefficients (p : ℕ) (n : ℕ) (h_p : Prime p) (h_n : n > 1) :
  (∀ x : ℕ, 1 ≤ x ∧ x < n → p ∣ Nat.choose n x) ↔ ∃ a : ℕ, a > 0 ∧ n = p^a :=
sorry

end prime_divisor_of_binomial_coefficients_l1825_182564


namespace C_satisfies_equation_C_specific_value_l1825_182547

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2
def B (x y : ℝ) : ℝ := x^2 + 2*x*y + y^2

-- Define C as a function of x and y
def C (x y : ℝ) : ℝ := -x^2 + 10*x*y - y^2

-- Theorem 1: C satisfies the given equation
theorem C_satisfies_equation (x y : ℝ) : 3 * A x y - 2 * B x y + C x y = 0 := by
  sorry

-- Theorem 2: C equals -57/4 when x = 1/2 and y = -2
theorem C_specific_value : C (1/2) (-2) = -57/4 := by
  sorry

end C_satisfies_equation_C_specific_value_l1825_182547


namespace hedge_trimming_charge_equals_685_l1825_182516

/-- Calculates the total charge for trimming a hedge with various shapes -/
def hedge_trimming_charge (basic_trim_price : ℚ) (sphere_price : ℚ) (pyramid_price : ℚ) 
  (cube_price : ℚ) (combined_shape_extra : ℚ) (total_boxwoods : ℕ) (sphere_count : ℕ) 
  (pyramid_count : ℕ) (cube_count : ℕ) (sphere_pyramid_count : ℕ) (sphere_cube_count : ℕ) : ℚ :=
  let basic_trim_total := basic_trim_price * total_boxwoods
  let sphere_total := sphere_price * sphere_count
  let pyramid_total := pyramid_price * pyramid_count
  let cube_total := cube_price * cube_count
  let sphere_pyramid_total := (sphere_price + pyramid_price + combined_shape_extra) * sphere_pyramid_count
  let sphere_cube_total := (sphere_price + cube_price + combined_shape_extra) * sphere_cube_count
  basic_trim_total + sphere_total + pyramid_total + cube_total + sphere_pyramid_total + sphere_cube_total

/-- The total charge for trimming the hedge is $685.00 -/
theorem hedge_trimming_charge_equals_685 : 
  hedge_trimming_charge 5 15 20 25 10 40 2 5 3 4 2 = 685 := by
  sorry

end hedge_trimming_charge_equals_685_l1825_182516


namespace bill_john_score_difference_l1825_182581

-- Define the scores as natural numbers
def bill_score : ℕ := 45
def sue_score : ℕ := bill_score * 2
def john_score : ℕ := 160 - bill_score - sue_score

-- Theorem statement
theorem bill_john_score_difference :
  bill_score > john_score ∧
  bill_score = sue_score / 2 ∧
  bill_score + john_score + sue_score = 160 →
  bill_score - john_score = 20 := by
sorry

end bill_john_score_difference_l1825_182581


namespace corner_rectangles_area_sum_l1825_182502

/-- Given a 2019x2019 square divided into 9 rectangles, with the central rectangle
    having dimensions 1511x1115, the sum of the areas of the four corner rectangles
    is 1832128. -/
theorem corner_rectangles_area_sum (square_side : ℕ) (central_length central_width : ℕ) :
  square_side = 2019 →
  central_length = 1511 →
  central_width = 1115 →
  4 * ((square_side - central_length) * (square_side - central_width)) = 1832128 :=
by sorry

end corner_rectangles_area_sum_l1825_182502


namespace monochromatic_subgrid_exists_l1825_182551

/-- Represents a cell color -/
inductive Color
| Black
| White

/-- Represents the grid -/
def Grid := Fin 3 → Fin 7 → Color

/-- Checks if a 2x2 subgrid has all cells of the same color -/
def has_monochromatic_2x2_subgrid (g : Grid) : Prop :=
  ∃ (i : Fin 2) (j : Fin 6),
    g i j = g i (j + 1) ∧
    g i j = g (i + 1) j ∧
    g i j = g (i + 1) (j + 1)

/-- Main theorem: Any 3x7 grid with black and white cells contains a monochromatic 2x2 subgrid -/
theorem monochromatic_subgrid_exists (g : Grid) : 
  has_monochromatic_2x2_subgrid g :=
sorry

end monochromatic_subgrid_exists_l1825_182551


namespace necklace_profit_is_1500_l1825_182552

/-- Calculates the profit from selling necklaces --/
def calculate_profit (charms_per_necklace : ℕ) (cost_per_charm : ℕ) (selling_price : ℕ) (necklaces_sold : ℕ) : ℕ :=
  let cost_per_necklace := charms_per_necklace * cost_per_charm
  let profit_per_necklace := selling_price - cost_per_necklace
  profit_per_necklace * necklaces_sold

/-- Proves that the profit from selling 30 necklaces is $1500 --/
theorem necklace_profit_is_1500 :
  calculate_profit 10 15 200 30 = 1500 := by
  sorry

end necklace_profit_is_1500_l1825_182552


namespace total_spent_is_2094_l1825_182594

def apple_price : ℚ := 40
def pear_price : ℚ := 50
def orange_price : ℚ := 30
def grape_price : ℚ := 60

def apple_quantity : ℚ := 14
def pear_quantity : ℚ := 18
def orange_quantity : ℚ := 10
def grape_quantity : ℚ := 8

def apple_discount : ℚ := 0.1
def pear_discount : ℚ := 0.05
def orange_discount : ℚ := 0.15
def grape_discount : ℚ := 0

def total_spent : ℚ := 
  apple_quantity * apple_price * (1 - apple_discount) +
  pear_quantity * pear_price * (1 - pear_discount) +
  orange_quantity * orange_price * (1 - orange_discount) +
  grape_quantity * grape_price * (1 - grape_discount)

theorem total_spent_is_2094 : total_spent = 2094 := by
  sorry

end total_spent_is_2094_l1825_182594


namespace no_integer_roots_l1825_182528

theorem no_integer_roots : ∀ x : ℤ, x^3 - 4*x^2 - 4*x + 24 ≠ 0 := by
  sorry

end no_integer_roots_l1825_182528


namespace function_comparison_l1825_182588

theorem function_comparison (a b : ℝ) (f g : ℝ → ℝ) 
  (h_diff_f : Differentiable ℝ f)
  (h_diff_g : Differentiable ℝ g)
  (h_deriv : ∀ x ∈ Set.Icc a b, deriv f x > deriv g x)
  (h_eq : f a = g a)
  (h_le : a ≤ b) :
  ∀ x ∈ Set.Icc a b, f x ≥ g x :=
sorry

end function_comparison_l1825_182588


namespace fraction_simplification_l1825_182549

theorem fraction_simplification : 
  (2+4+6+8+10+12+14+16+18+20) / (1+2+3+4+5+6+7+8+9+10) = 2 := by
  sorry

end fraction_simplification_l1825_182549


namespace max_sum_property_terms_l1825_182576

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluation of a quadratic polynomial at a given point -/
def evaluate (P : QuadraticPolynomial) (x : ℝ) : ℝ :=
  P.a * x^2 + P.b * x + P.c

/-- The property that P(n+1) = P(n) + P(n-1) -/
def hasSumProperty (P : QuadraticPolynomial) (n : ℕ) : Prop :=
  evaluate P (n + 1 : ℝ) = evaluate P n + evaluate P (n - 1 : ℝ)

/-- The main theorem: maximum number of terms with sum property is 2 -/
theorem max_sum_property_terms (P : QuadraticPolynomial) :
  (∃ (S : Finset ℕ), (∀ n ∈ S, n ≥ 2 ∧ hasSumProperty P n) ∧ S.card > 2) → False :=
sorry

end max_sum_property_terms_l1825_182576


namespace statement_a_statement_b_statement_c_statement_d_all_statements_correct_l1825_182526

-- Statement A
theorem statement_a (a b : ℝ) : a^2 = b^2 → |a| = |b| := by
  sorry

-- Statement B
theorem statement_b (a b : ℝ) : a + b = 0 → a^3 + b^3 = 0 := by
  sorry

-- Statement C
theorem statement_c (a b : ℝ) : a < b ∧ a ≠ 0 ∧ b ≠ 0 → 1/a > 1/b := by
  sorry

-- Statement D
theorem statement_d (a : ℝ) : -1 < a ∧ a < 0 → a^3 < a^5 := by
  sorry

-- All statements are correct
theorem all_statements_correct : 
  (∀ a b : ℝ, a^2 = b^2 → |a| = |b|) ∧
  (∀ a b : ℝ, a + b = 0 → a^3 + b^3 = 0) ∧
  (∀ a b : ℝ, a < b ∧ a ≠ 0 ∧ b ≠ 0 → 1/a > 1/b) ∧
  (∀ a : ℝ, -1 < a ∧ a < 0 → a^3 < a^5) := by
  sorry

end statement_a_statement_b_statement_c_statement_d_all_statements_correct_l1825_182526


namespace tetrahedron_circumsphere_area_l1825_182509

/-- The surface area of a circumscribed sphere of a regular tetrahedron with side length 2 -/
theorem tetrahedron_circumsphere_area : 
  let side_length : ℝ := 2
  let circumradius : ℝ := side_length * Real.sqrt 3 / 3
  let sphere_area : ℝ := 4 * Real.pi * circumradius^2
  sphere_area = 16 * Real.pi / 3 := by
  sorry

end tetrahedron_circumsphere_area_l1825_182509


namespace arithmetic_sequence_middle_term_l1825_182578

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_3 : a 3 = 16)
  (h_9 : a 9 = 80) :
  a 6 = 48 := by
sorry

end arithmetic_sequence_middle_term_l1825_182578


namespace sum_of_zeros_is_16_l1825_182572

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := (x - 3)^2 + 4

-- Define the vertex of the original parabola
def original_vertex : ℝ × ℝ := (3, 4)

-- Define the transformed parabola after all operations
def transformed_parabola (x : ℝ) : ℝ := (x - 8)^2

-- Define the new vertex after transformations
def new_vertex : ℝ × ℝ := (8, 8)

-- Define the zeros of the transformed parabola
def zeros : Set ℝ := {x : ℝ | transformed_parabola x = 0}

-- Theorem statement
theorem sum_of_zeros_is_16 : ∀ p q : ℝ, p ∈ zeros ∧ q ∈ zeros → p + q = 16 := by
  sorry

end sum_of_zeros_is_16_l1825_182572


namespace tournament_schools_l1825_182536

theorem tournament_schools (n : ℕ) : 
  (∀ (school : ℕ), school ≤ n → ∃ (team : Fin 4 → ℕ), 
    (∀ i j, i ≠ j → team i ≠ team j) ∧ 
    (∃ (theo leah mark nora : ℕ), 
      theo = (4 * n + 1) / 2 ∧
      leah = 48 ∧ 
      mark = 75 ∧ 
      nora = 97 ∧
      theo < leah ∧ theo < mark ∧ theo < nora ∧
      (∀ k, k ∈ [theo, leah, mark, nora] → k ≤ 4 * n) ∧
      (∀ k, k ∉ [theo, leah, mark, nora] → 
        (k < theo ∧ k ≤ 4 * n - 3) ∨ (k > theo ∧ k ≤ 4 * n)))) → 
  n = 36 := by
sorry

end tournament_schools_l1825_182536


namespace no_real_roots_l1825_182533

theorem no_real_roots (a b c : ℕ) (ha : a < 1000000) (hb : b < 1000000) (hc : c < 1000000) :
  ¬∃ x : ℝ, (a * x^2)^(1/21) + (b * x)^(1/21) + c^(1/21) = 0 := by
  sorry

end no_real_roots_l1825_182533


namespace parabola_y_relationship_l1825_182570

-- Define the parabola function
def f (x : ℝ) (m : ℝ) : ℝ := -3 * x^2 - 12 * x + m

-- Define the theorem
theorem parabola_y_relationship (m : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : f (-3) m = y₁)
  (h₂ : f (-2) m = y₂)
  (h₃ : f 1 m = y₃) :
  y₂ > y₁ ∧ y₁ > y₃ :=
sorry

end parabola_y_relationship_l1825_182570


namespace power_of_integer_for_3150_l1825_182599

theorem power_of_integer_for_3150 (a : ℕ) (h1 : a = 14) 
  (h2 : ∀ k < a, ∃ n : ℕ, 3150 * k = n ^ 2 → False) : 
  ∃ n : ℕ, 3150 * a = n ^ 2 :=
sorry

end power_of_integer_for_3150_l1825_182599


namespace pie_eating_contest_l1825_182532

theorem pie_eating_contest (first_student second_student : ℚ) : 
  first_student = 8/9 → second_student = 5/6 → first_student - second_student = 1/18 := by
  sorry

end pie_eating_contest_l1825_182532


namespace doris_monthly_expenses_l1825_182529

/-- Calculates Doris's monthly expenses based on her work schedule and hourly rate -/
def monthly_expenses (hourly_rate : ℕ) (weekday_hours : ℕ) (saturday_hours : ℕ) (weeks : ℕ) : ℕ :=
  let weekly_hours := weekday_hours * 5 + saturday_hours
  let weekly_earnings := hourly_rate * weekly_hours
  weekly_earnings * weeks

/-- Proves that Doris's monthly expenses are $1200 given her work schedule and hourly rate -/
theorem doris_monthly_expenses :
  monthly_expenses 20 3 5 3 = 1200 := by
  sorry

end doris_monthly_expenses_l1825_182529


namespace product_of_sum_and_difference_l1825_182543

theorem product_of_sum_and_difference (x y : ℝ) : 
  x + y = 27 ∧ x - y = 9 → x * y = 162 := by
sorry

end product_of_sum_and_difference_l1825_182543


namespace window_treatment_cost_l1825_182527

/-- The number of windows that need treatment -/
def num_windows : ℕ := 3

/-- The cost of a pair of sheers in dollars -/
def sheer_cost : ℚ := 40

/-- The cost of a pair of drapes in dollars -/
def drape_cost : ℚ := 60

/-- The total cost of window treatments for all windows -/
def total_cost : ℚ := num_windows * (sheer_cost + drape_cost)

theorem window_treatment_cost : total_cost = 300 := by
  sorry

end window_treatment_cost_l1825_182527


namespace lindas_bakery_profit_l1825_182537

/-- Calculate Linda's total profit for the day given her bread sales strategy -/
theorem lindas_bakery_profit :
  let total_loaves : ℕ := 60
  let morning_price : ℚ := 3
  let afternoon_price : ℚ := 3/2
  let evening_price : ℚ := 1
  let production_cost : ℚ := 1
  let morning_sales : ℕ := total_loaves / 3
  let afternoon_sales : ℕ := (total_loaves - morning_sales) / 2
  let evening_sales : ℕ := total_loaves - morning_sales - afternoon_sales
  let total_revenue : ℚ := morning_sales * morning_price + 
                           afternoon_sales * afternoon_price + 
                           evening_sales * evening_price
  let total_cost : ℚ := total_loaves * production_cost
  let profit : ℚ := total_revenue - total_cost
  profit = 50 := by
sorry

end lindas_bakery_profit_l1825_182537


namespace dreamy_vacation_probability_l1825_182531

/-- The probability of drawing a dreamy vacation note -/
def p : ℝ := 0.4

/-- The total number of people drawing notes -/
def n : ℕ := 5

/-- The number of people drawing a dreamy vacation note -/
def k : ℕ := 3

/-- The target probability -/
def target_prob : ℝ := 0.2304

/-- Theorem stating that the probability of exactly k people out of n drawing a dreamy vacation note is equal to the target probability -/
theorem dreamy_vacation_probability :
  Nat.choose n k * p^k * (1 - p)^(n - k) = target_prob := by
  sorry

end dreamy_vacation_probability_l1825_182531


namespace green_marble_probability_l1825_182571

/-- The probability of selecting a green marble from a basket -/
theorem green_marble_probability :
  let total_marbles : ℕ := 4 + 9 + 5 + 10
  let green_marbles : ℕ := 9
  (green_marbles : ℚ) / total_marbles = 9 / 28 :=
by
  sorry

end green_marble_probability_l1825_182571


namespace infinite_sum_equals_three_fortieths_l1825_182540

/-- The sum of the infinite series n / (n^4 + 16) from n = 1 to infinity equals 3/40 -/
theorem infinite_sum_equals_three_fortieths :
  (∑' n : ℕ+, (n : ℝ) / ((n : ℝ)^4 + 16)) = 3/40 := by sorry

end infinite_sum_equals_three_fortieths_l1825_182540


namespace systematic_sample_max_l1825_182559

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  totalProducts : Nat
  sampleSize : Nat
  interval : Nat

/-- Creates a systematic sample given total products and sample size -/
def createSystematicSample (totalProducts sampleSize : Nat) : SystematicSample :=
  { totalProducts := totalProducts
  , sampleSize := sampleSize
  , interval := totalProducts / sampleSize }

/-- Checks if a number is in the sample given the first element -/
def isInSample (sample : SystematicSample) (first last : Nat) : Prop :=
  ∃ k, k < sample.sampleSize ∧ first + k * sample.interval = last

/-- Theorem: In a systematic sample of size 5 from 80 products,
    if product 28 is in the sample, then the maximum number in the sample is 76 -/
theorem systematic_sample_max (sample : SystematicSample) 
  (h1 : sample.totalProducts = 80)
  (h2 : sample.sampleSize = 5)
  (h3 : isInSample sample 28 28) :
  isInSample sample 28 76 ∧ ∀ n, isInSample sample 28 n → n ≤ 76 := by
  sorry

end systematic_sample_max_l1825_182559


namespace parallelepiped_volume_example_l1825_182515

/-- The volume of a parallelepiped with given dimensions -/
def parallelepipedVolume (base height depth : ℝ) : ℝ :=
  base * depth * height

/-- Theorem: The volume of a parallelepiped with base 28 cm, height 32 cm, and depth 15 cm is 13440 cubic centimeters -/
theorem parallelepiped_volume_example : parallelepipedVolume 28 32 15 = 13440 := by
  sorry

end parallelepiped_volume_example_l1825_182515


namespace angle_C_measure_side_c_length_l1825_182557

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesCondition1 (t : Triangle) : Prop :=
  t.b * Real.cos t.A + t.a * Real.cos t.B = -2 * t.c * Real.cos t.C

def satisfiesCondition2 (t : Triangle) : Prop :=
  t.b = 2 * t.a ∧ 
  (1/2) * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3

-- Theorem 1
theorem angle_C_measure (t : Triangle) (h : satisfiesCondition1 t) : t.C = 2 * π / 3 := by
  sorry

-- Theorem 2
theorem side_c_length (t : Triangle) (h1 : satisfiesCondition1 t) (h2 : satisfiesCondition2 t) : 
  t.c = 2 * Real.sqrt 7 := by
  sorry

end angle_C_measure_side_c_length_l1825_182557


namespace freyja_age_l1825_182513

/-- Represents the ages of the people in the problem -/
structure Ages where
  kaylin : ℕ
  sarah : ℕ
  eli : ℕ
  freyja : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.kaylin = ages.sarah - 5 ∧
  ages.sarah = 2 * ages.eli ∧
  ages.eli = ages.freyja + 9 ∧
  ages.kaylin = 33

/-- The theorem stating Freyja's age given the problem conditions -/
theorem freyja_age (ages : Ages) (h : problem_conditions ages) : ages.freyja = 10 := by
  sorry


end freyja_age_l1825_182513


namespace inequality_system_solution_set_l1825_182590

theorem inequality_system_solution_set :
  let S := {x : ℝ | 3 * x + 2 ≥ 1 ∧ (5 - x) / 2 < 0}
  S = {x : ℝ | -1/3 ≤ x ∧ x < 5} := by
  sorry

end inequality_system_solution_set_l1825_182590


namespace second_quadrant_m_range_l1825_182542

theorem second_quadrant_m_range (m : ℝ) : 
  let z : ℂ := m^2 * (1 + Complex.I) - m * (4 + Complex.I) - 6 * Complex.I
  (z.re < 0 ∧ z.im > 0) → (3 < m ∧ m < 4) :=
by sorry

end second_quadrant_m_range_l1825_182542


namespace triangle_area_l1825_182514

/-- Given a triangle with perimeter 42 cm and inradius 5.0 cm, its area is 105 cm². -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 42 → inradius = 5 → area = perimeter / 2 * inradius → area = 105 := by
  sorry

end triangle_area_l1825_182514


namespace first_player_always_wins_l1825_182511

/-- Represents a point on a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a color of a dot --/
inductive Color
  | Red
  | Blue

/-- Represents a dot on the plane --/
structure Dot where
  point : Point
  color : Color

/-- Represents the game state --/
structure GameState where
  dots : List Dot

/-- Represents a player's strategy --/
def Strategy := GameState → Point

/-- Checks if three points form an equilateral triangle --/
def isEquilateralTriangle (p1 p2 p3 : Point) : Prop := sorry

/-- The main theorem stating that the first player can always win --/
theorem first_player_always_wins :
  ∀ (second_player_strategy : Strategy),
  ∃ (first_player_strategy : Strategy) (n : ℕ),
  ∀ (game : GameState),
  ∃ (p1 p2 p3 : Point),
  (p1 ∈ game.dots.map Dot.point) ∧
  (p2 ∈ game.dots.map Dot.point) ∧
  (p3 ∈ game.dots.map Dot.point) ∧
  isEquilateralTriangle p1 p2 p3 :=
sorry

end first_player_always_wins_l1825_182511


namespace angle_C_is_60_degrees_a_and_b_values_l1825_182561

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a + t.b = 5 ∧
  t.c = Real.sqrt 7 ∧
  4 * (Real.sin (t.A + t.B) / 2)^2 - Real.cos (2 * t.C) = 7/2 ∧
  t.A + t.B + t.C = Real.pi

-- Theorem 1: Prove that C = 60°
theorem angle_C_is_60_degrees (t : Triangle) 
  (h : triangle_conditions t) : t.C = Real.pi / 3 := by
  sorry

-- Theorem 2: Prove that if a > b, then a = 3 and b = 2
theorem a_and_b_values (t : Triangle) 
  (h1 : triangle_conditions t) (h2 : t.a > t.b) : t.a = 3 ∧ t.b = 2 := by
  sorry

end angle_C_is_60_degrees_a_and_b_values_l1825_182561


namespace sine_is_periodic_l1825_182521

-- Define the property of being a trigonometric function
def IsTrigonometric (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being a periodic function
def IsPeriodic (f : ℝ → ℝ) : Prop := sorry

-- Define the sine function
def sin : ℝ → ℝ := sorry

-- Theorem statement
theorem sine_is_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →
  IsTrigonometric sin →
  IsPeriodic sin := by sorry

end sine_is_periodic_l1825_182521


namespace camp_attendance_l1825_182580

theorem camp_attendance (total_lawrence : ℕ) (stayed_home : ℕ) (went_to_camp : ℕ)
  (h1 : total_lawrence = 1538832)
  (h2 : stayed_home = 644997)
  (h3 : went_to_camp = 893835)
  (h4 : total_lawrence = stayed_home + went_to_camp) :
  0 = went_to_camp - (total_lawrence - stayed_home) :=
by sorry

end camp_attendance_l1825_182580


namespace unique_solution_prime_cube_equation_l1825_182505

theorem unique_solution_prime_cube_equation :
  ∀ (p m n : ℕ), 
    Prime p → 
    1 + p^n = m^3 → 
    p = 7 ∧ n = 1 ∧ m = 2 :=
by sorry

end unique_solution_prime_cube_equation_l1825_182505


namespace yellow_face_probability_l1825_182568

/-- The probability of rolling a yellow face on a 12-sided die with 4 yellow faces is 1/3 -/
theorem yellow_face_probability (total_faces : ℕ) (yellow_faces : ℕ) 
  (h1 : total_faces = 12) (h2 : yellow_faces = 4) : 
  (yellow_faces : ℚ) / total_faces = 1 / 3 := by
  sorry

end yellow_face_probability_l1825_182568


namespace specific_trapezoid_area_l1825_182518

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  longerBase : ℝ
  baseAngle : ℝ
  height : ℝ

/-- Calculate the area of the isosceles trapezoid -/
def trapezoidArea (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific trapezoid is 100 -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    longerBase := 20,
    baseAngle := Real.arcsin 0.6,
    height := 9
  }
  trapezoidArea t = 100 := by sorry

end specific_trapezoid_area_l1825_182518


namespace symmetric_graph_phi_l1825_182579

/-- Given a function f and a real number φ, proves that if the graph of y = f(x + φ) 
    is symmetric about x = 0 and |φ| ≤ π/2, then φ = π/6 -/
theorem symmetric_graph_phi (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x : ℝ, f x = 2 * Real.sin (x + π/3)) →
  (∀ x : ℝ, f (x + φ) = f (-x + φ)) →
  |φ| ≤ π/2 →
  φ = π/6 := by
  sorry

#check symmetric_graph_phi

end symmetric_graph_phi_l1825_182579


namespace sequence_inequality_l1825_182500

theorem sequence_inequality (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, n ≥ 2 → 2 * a n = a (n - 1) + a (n + 1)) : 
  a 2 * a 4 ≤ a 3 ^ 2 := by
sorry

end sequence_inequality_l1825_182500
