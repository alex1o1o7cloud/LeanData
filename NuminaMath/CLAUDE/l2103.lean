import Mathlib

namespace decimal_division_subtraction_l2103_210347

theorem decimal_division_subtraction : (0.24 / 0.004) - 0.1 = 59.9 := by
  sorry

end decimal_division_subtraction_l2103_210347


namespace unique_solution_system_l2103_210304

theorem unique_solution_system (x y : ℝ) :
  (y = (x + 2)^2 ∧ x * y + y = 2) ↔ (x = 2^(1/3) - 2 ∧ y = 2^(2/3)) :=
by sorry

end unique_solution_system_l2103_210304


namespace smallest_number_satisfying_conditions_l2103_210326

theorem smallest_number_satisfying_conditions : ∃ n : ℕ, 
  n > 0 ∧
  n % 6 = 2 ∧
  n % 7 = 3 ∧
  n % 8 = 4 ∧
  (∀ m : ℕ, m > 0 → m % 6 = 2 → m % 7 = 3 → m % 8 = 4 → n ≤ m) ∧
  n = 164 :=
by sorry

end smallest_number_satisfying_conditions_l2103_210326


namespace evaluate_expression_l2103_210327

theorem evaluate_expression : (2 : ℕ)^(3^2) + 1^(3^3) = 513 := by
  sorry

end evaluate_expression_l2103_210327


namespace inequality_proof_l2103_210345

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b > 1) : a * b < a + b := by
  sorry

end inequality_proof_l2103_210345


namespace problem_solution_l2103_210378

def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + a^2 - 12 = 0}
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 8 = 0}
def C (m : ℝ) : Set ℝ := {x : ℝ | m*x + 1 = 0}

theorem problem_solution :
  (∀ a : ℝ, A a = B → a = 2) ∧
  (∀ m : ℝ, B ∪ C m = B → m = -1/4 ∨ m = 0 ∨ m = 1/2) :=
sorry

end problem_solution_l2103_210378


namespace simplify_expression_l2103_210343

theorem simplify_expression (x : ℝ) : (3 * x + 20) - (7 * x - 5) = -4 * x + 25 := by
  sorry

end simplify_expression_l2103_210343


namespace prob_coprime_42_l2103_210377

/-- The number of positive integers less than or equal to n that are relatively prime to n -/
def phi (n : ℕ) : ℕ := sorry

theorem prob_coprime_42 : (phi 42 : ℚ) / 42 = 2 / 7 := by sorry

end prob_coprime_42_l2103_210377


namespace two_greater_than_negative_three_l2103_210380

theorem two_greater_than_negative_three : 2 > -3 := by
  sorry

end two_greater_than_negative_three_l2103_210380


namespace sports_club_size_l2103_210387

/-- The number of members in a sports club -/
def sports_club_members (badminton tennis both neither : ℕ) : ℕ :=
  badminton + tennis - both + neither

/-- Theorem: The sports club has 28 members -/
theorem sports_club_size :
  ∃ (badminton tennis both neither : ℕ),
    badminton = 17 ∧
    tennis = 19 ∧
    both = 10 ∧
    neither = 2 ∧
    sports_club_members badminton tennis both neither = 28 := by
  sorry

end sports_club_size_l2103_210387


namespace zero_in_M_l2103_210321

def M : Set Int := {-1, 0, 1}

theorem zero_in_M : 0 ∈ M := by sorry

end zero_in_M_l2103_210321


namespace queen_middle_teachers_l2103_210366

/-- Represents a school with students, teachers, and classes. -/
structure School where
  num_students : ℕ
  classes_per_student : ℕ
  classes_per_teacher : ℕ
  students_per_class : ℕ

/-- Calculates the number of teachers in a school. -/
def num_teachers (s : School) : ℕ :=
  (s.num_students * s.classes_per_student) / (s.students_per_class * s.classes_per_teacher)

/-- Queen Middle School -/
def queen_middle : School :=
  { num_students := 1500
  , classes_per_student := 6
  , classes_per_teacher := 5
  , students_per_class := 25
  }

/-- Theorem stating that Queen Middle School has 72 teachers -/
theorem queen_middle_teachers : num_teachers queen_middle = 72 := by
  sorry

end queen_middle_teachers_l2103_210366


namespace ones_digit_of_largest_power_of_three_dividing_27_factorial_l2103_210376

/-- The largest power of 3 that divides n! -/
def largest_power_of_three_dividing_factorial (n : ℕ) : ℕ :=
  (n / 3) + (n / 9) + (n / 27)

/-- The ones digit of 3^n -/
def ones_digit_of_power_of_three (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0 -- This case is unreachable, but needed for exhaustiveness

theorem ones_digit_of_largest_power_of_three_dividing_27_factorial :
  ones_digit_of_power_of_three (largest_power_of_three_dividing_factorial 27) = 3 :=
by sorry

end ones_digit_of_largest_power_of_three_dividing_27_factorial_l2103_210376


namespace binomial_coefficient_22_5_l2103_210309

theorem binomial_coefficient_22_5 (h1 : Nat.choose 20 3 = 1140)
                                  (h2 : Nat.choose 20 4 = 4845)
                                  (h3 : Nat.choose 20 5 = 15504) :
  Nat.choose 22 5 = 26334 := by
  sorry

end binomial_coefficient_22_5_l2103_210309


namespace power_of_two_pairs_l2103_210334

theorem power_of_two_pairs (a b : ℕ) (h1 : a ≠ b) (h2 : ∃ k : ℕ, a + b = 2^k) (h3 : ∃ m : ℕ, a * b + 1 = 2^m) :
  (∃ k : ℕ, k ≥ 1 ∧ ((a = 1 ∧ b = 2^k - 1) ∨ (a = 2^k - 1 ∧ b = 1))) ∨
  (∃ k : ℕ, k ≥ 2 ∧ ((a = 2^k - 1 ∧ b = 2^k + 1) ∨ (a = 2^k + 1 ∧ b = 2^k - 1))) :=
by sorry

end power_of_two_pairs_l2103_210334


namespace largest_quantity_l2103_210313

theorem largest_quantity (x y z w : ℝ) 
  (h : x + 5 = y - 3 ∧ x + 5 = z + 2 ∧ x + 5 = w - 4) : 
  w ≥ x ∧ w ≥ y ∧ w ≥ z := by
  sorry

end largest_quantity_l2103_210313


namespace jude_change_l2103_210370

def chair_price : ℕ := 13
def table_price : ℕ := 50
def plate_set_price : ℕ := 20
def num_chairs : ℕ := 3
def num_plate_sets : ℕ := 2
def total_paid : ℕ := 130

def total_cost : ℕ := chair_price * num_chairs + table_price + plate_set_price * num_plate_sets

theorem jude_change : 
  total_paid - total_cost = 1 :=
sorry

end jude_change_l2103_210370


namespace woo_jun_age_l2103_210305

theorem woo_jun_age :
  ∀ (w m : ℕ),
  w = m / 4 - 1 →
  m = 5 * w - 5 →
  w = 9 := by
sorry

end woo_jun_age_l2103_210305


namespace red_cube_latin_square_bijection_l2103_210336

/-- A Latin square of order n is an n × n array filled with n different symbols, 
    each occurring exactly once in each row and exactly once in each column. -/
def is_latin_square (s : Fin 4 → Fin 4 → Fin 4) : Prop :=
  ∀ i j k : Fin 4, 
    (∀ j' : Fin 4, j ≠ j' → s i j ≠ s i j') ∧ 
    (∀ i' : Fin 4, i ≠ i' → s i j ≠ s i' j)

/-- The number of 4 × 4 Latin squares -/
def num_latin_squares : ℕ := sorry

/-- A configuration of red cubes in a 4 × 4 × 4 cube -/
def red_cube_config : Type := Fin 4 → Fin 4 → Fin 4

/-- A valid configuration of red cubes satisfies the constraint that
    in every 1 × 1 × 4 rectangular prism, exactly 1 unit cube is red -/
def is_valid_config (c : red_cube_config) : Prop :=
  ∀ i j : Fin 4, ∃! k : Fin 4, c i j = k

/-- The number of valid red cube configurations -/
def num_valid_configs : ℕ := sorry

theorem red_cube_latin_square_bijection :
  num_valid_configs = num_latin_squares :=
sorry

end red_cube_latin_square_bijection_l2103_210336


namespace three_digit_number_problem_l2103_210310

/-- Given a three-digit number abc where a, b, and c are non-zero digits,
    prove that abc = 425 if the sum of the other five three-digit numbers
    formed by rearranging a, b, c is 2017. -/
theorem three_digit_number_problem (a b c : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  a < 10 → b < 10 → c < 10 →
  (100 * a + 10 * b + c) +
  (100 * a + 10 * c + b) +
  (100 * b + 10 * a + c) +
  (100 * b + 10 * c + a) +
  (100 * c + 10 * a + b) = 2017 →
  100 * a + 10 * b + c = 425 := by
sorry

end three_digit_number_problem_l2103_210310


namespace opposite_of_abs_negative_2023_l2103_210340

theorem opposite_of_abs_negative_2023 : -(|-2023|) = -2023 := by
  sorry

end opposite_of_abs_negative_2023_l2103_210340


namespace line_slope_range_l2103_210322

/-- A line passing through (1,1) with y-intercept in (0,2) has slope in (-1,1) -/
theorem line_slope_range (l : Set (ℝ × ℝ)) (y_intercept : ℝ) :
  (∀ p ∈ l, ∃ k : ℝ, p.2 - 1 = k * (p.1 - 1)) →  -- l is a line
  (1, 1) ∈ l →  -- l passes through (1,1)
  0 < y_intercept ∧ y_intercept < 2 →  -- y-intercept is in (0,2)
  (∃ b : ℝ, ∀ x y : ℝ, (x, y) ∈ l ↔ y = y_intercept + (y_intercept - 1) * (x - 1)) →
  ∃ k : ℝ, -1 < k ∧ k < 1 ∧ ∀ x y : ℝ, (x, y) ∈ l ↔ y - 1 = k * (x - 1) := by
  sorry

end line_slope_range_l2103_210322


namespace inverse_variation_cube_l2103_210301

/-- Given positive real numbers x and y that vary inversely with respect to x^3,
    prove that if y = 8 when x = 2, then x = 0.4 when y = 1000. -/
theorem inverse_variation_cube (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h_inverse : ∃ k : ℝ, ∀ x y, x^3 * y = k) 
    (h_initial : 2^3 * 8 = (x^3 * y)) :
  y = 1000 → x = 0.4 := by
sorry

end inverse_variation_cube_l2103_210301


namespace algebraic_equality_l2103_210399

theorem algebraic_equality (a b c : ℝ) : a - b + c = a - (b - c) := by
  sorry

end algebraic_equality_l2103_210399


namespace lewis_age_l2103_210356

theorem lewis_age (ages : List Nat) 
  (h1 : ages = [4, 6, 8, 10, 12])
  (h2 : ∃ (a b : Nat), a ∈ ages ∧ b ∈ ages ∧ a + b = 18 ∧ a ≠ b)
  (h3 : ∃ (c d : Nat), c ∈ ages ∧ d ∈ ages ∧ c > 5 ∧ c < 11 ∧ d > 5 ∧ d < 11 ∧ c ≠ d)
  (h4 : 6 ∈ ages)
  (h5 : ∀ (x : Nat), x ∈ ages → x = 4 ∨ x = 6 ∨ x = 8 ∨ x = 10 ∨ x = 12) :
  4 ∈ ages := by
  sorry

end lewis_age_l2103_210356


namespace gcd_193116_127413_properties_l2103_210335

theorem gcd_193116_127413_properties :
  let g := Nat.gcd 193116 127413
  ∃ (g : ℕ),
    g = 3 ∧
    g ∣ 3 ∧
    ¬(2 ∣ g) ∧
    ¬(9 ∣ g) ∧
    ¬(11 ∣ g) ∧
    ¬(33 ∣ g) ∧
    ¬(99 ∣ g) := by
  sorry

end gcd_193116_127413_properties_l2103_210335


namespace distinct_outfits_l2103_210383

theorem distinct_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) (hats : ℕ) 
  (h_shirts : shirts = 7)
  (h_pants : pants = 5)
  (h_ties : ties = 6)
  (h_hats : hats = 4) :
  shirts * pants * (ties + 1) * (hats + 1) = 1225 := by
  sorry

end distinct_outfits_l2103_210383


namespace m_range_theorem_l2103_210385

-- Define propositions p and q as functions of x and m
def p (x m : ℝ) : Prop := (x - m)^2 > 3*(x - m)
def q (x : ℝ) : Prop := x^2 + 3*x - 4 < 0

-- Define the range of m
def m_range (m : ℝ) : Prop := m ≤ -7 ∨ m ≥ 1

-- Theorem statement
theorem m_range_theorem :
  (∀ x m : ℝ, q x → p x m) ∧ 
  (∃ x m : ℝ, p x m ∧ ¬(q x)) →
  ∀ m : ℝ, m_range m ↔ ∃ x : ℝ, p x m :=
sorry

end m_range_theorem_l2103_210385


namespace chord_length_in_circle_l2103_210319

theorem chord_length_in_circle (r : ℝ) (h : r = 15) : 
  ∃ (c : ℝ), c = 26 ∧ 
  c^2 = 4 * (r^2 - (r/2)^2) ∧ 
  c > 0 := by
sorry

end chord_length_in_circle_l2103_210319


namespace complex_fraction_simplification_l2103_210362

theorem complex_fraction_simplification :
  (Complex.mk 3 (-5)) / (Complex.mk 2 (-7)) = Complex.mk (-41/45) (-11/45) := by
  sorry

end complex_fraction_simplification_l2103_210362


namespace max_min_product_l2103_210395

theorem max_min_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 12) (h_prod_sum : x*y + y*z + z*x = 30) :
  ∃ (n : ℝ), n = min (x*y) (min (y*z) (z*x)) ∧ n ≤ 2 ∧ 
  ∀ (m : ℝ), (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b + c = 12 ∧ a*b + b*c + c*a = 30 ∧ 
    m = min (a*b) (min (b*c) (c*a))) → m ≤ n :=
sorry

end max_min_product_l2103_210395


namespace unequal_gender_probability_l2103_210354

theorem unequal_gender_probability (n : ℕ) (p : ℚ) : 
  n = 12 → p = 1/2 → 
  (1 - (Nat.choose n (n/2) : ℚ) * p^(n/2) * (1-p)^(n/2)) = 793/1024 := by
  sorry

end unequal_gender_probability_l2103_210354


namespace solution_set_of_even_decreasing_quadratic_l2103_210311

def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2*a) * x - 2*b

theorem solution_set_of_even_decreasing_quadratic 
  (a b : ℝ) 
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_decreasing : ∀ x y, 0 < x → x < y → f a b y < f a b x) :
  {x : ℝ | f a b x > 0} = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end solution_set_of_even_decreasing_quadratic_l2103_210311


namespace quadratic_no_real_roots_l2103_210331

theorem quadratic_no_real_roots (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + a*x - 4*a = 0) ↔ (-16 < a ∧ a < 0) :=
sorry

end quadratic_no_real_roots_l2103_210331


namespace square_area_l2103_210368

/-- The area of a square with side length 11 cm is 121 cm². -/
theorem square_area (side_length : ℝ) (h : side_length = 11) : side_length ^ 2 = 121 := by
  sorry

end square_area_l2103_210368


namespace conference_center_distance_l2103_210374

/-- Represents the problem of calculating the distance to the conference center --/
theorem conference_center_distance :
  -- Initial speed
  ∀ (initial_speed : ℝ),
  -- Speed increase
  ∀ (speed_increase : ℝ),
  -- Distance covered in first hour
  ∀ (first_hour_distance : ℝ),
  -- Late arrival time if continued at initial speed
  ∀ (late_arrival_time : ℝ),
  -- Early arrival time with increased speed
  ∀ (early_arrival_time : ℝ),
  -- Conditions from the problem
  initial_speed = 40 →
  speed_increase = 20 →
  first_hour_distance = 40 →
  late_arrival_time = 1.5 →
  early_arrival_time = 1 →
  -- Conclusion: The distance to the conference center is 100 miles
  ∃ (distance : ℝ), distance = 100 := by
  sorry


end conference_center_distance_l2103_210374


namespace tangent_line_passes_through_point_l2103_210339

/-- The function f(x) = ax³ + x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem tangent_line_passes_through_point (a : ℝ) :
  (f_derivative a 1) * (2 - 1) + (f a 1) = 7 → a = 1 := by
  sorry

end tangent_line_passes_through_point_l2103_210339


namespace stating_italian_regular_clock_coincidences_l2103_210360

/-- Represents a clock with specified rotations for hour and minute hands per day. -/
structure Clock :=
  (hour_rotations : ℕ)
  (minute_rotations : ℕ)

/-- The Italian clock with 1 hour hand rotation and 24 minute hand rotations per day. -/
def italian_clock : Clock :=
  { hour_rotations := 1, minute_rotations := 24 }

/-- The regular clock with 2 hour hand rotations and 24 minute hand rotations per day. -/
def regular_clock : Clock :=
  { hour_rotations := 2, minute_rotations := 24 }

/-- 
  The number of times the hands of an Italian clock coincide with 
  the hands of a regular clock in a 24-hour period.
-/
def coincidence_count (ic : Clock) (rc : Clock) : ℕ := sorry

/-- 
  Theorem stating that the number of hand coincidences between 
  the Italian clock and regular clock in a 24-hour period is 12.
-/
theorem italian_regular_clock_coincidences : 
  coincidence_count italian_clock regular_clock = 12 := by sorry

end stating_italian_regular_clock_coincidences_l2103_210360


namespace difference_of_squares_l2103_210341

theorem difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) := by
  sorry

end difference_of_squares_l2103_210341


namespace sum_of_reciprocals_positive_l2103_210398

theorem sum_of_reciprocals_positive (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (product_neg_hundred : a * b * c = -100) : 
  1/a + 1/b + 1/c > 0 := by
sorry

end sum_of_reciprocals_positive_l2103_210398


namespace train_length_l2103_210303

/-- Given a train traveling at 45 kmph that passes a 140 m long bridge in 40 seconds,
    prove that the length of the train is 360 m. -/
theorem train_length (speed : ℝ) (bridge_length : ℝ) (time : ℝ) (train_length : ℝ) : 
  speed = 45 → bridge_length = 140 → time = 40 → 
  train_length = (speed * 1000 / 3600 * time) - bridge_length → 
  train_length = 360 := by
sorry

end train_length_l2103_210303


namespace percentage_difference_l2103_210355

theorem percentage_difference (x y : ℝ) (h : y = x + 0.6 * x) :
  (y - x) / y = 0.375 := by
  sorry

end percentage_difference_l2103_210355


namespace rain_probability_jan20_l2103_210397

-- Define the initial probability and the number of days
def initial_prob : ℚ := 1/2
def days : ℕ := 5

-- Define the daily probability adjustment factors
def factor1 : ℚ := 2017/2016
def factor2 : ℚ := 1007/2016

-- Define the function to calculate the probability after n days
def prob_after_n_days (n : ℕ) : ℚ :=
  initial_prob * (((factor1 + factor2) / 2) ^ n)

-- The theorem to prove
theorem rain_probability_jan20 :
  prob_after_n_days days = 243/2048 := by
  sorry


end rain_probability_jan20_l2103_210397


namespace tan_alpha_value_l2103_210312

theorem tan_alpha_value (α : Real) (h1 : π < α ∧ α < 3*π/2) (h2 : Real.sin (α/2) = Real.sqrt 5 / 3) :
  Real.tan α = -4 * Real.sqrt 5 := by
  sorry

end tan_alpha_value_l2103_210312


namespace symmetric_absolute_value_function_l2103_210323

/-- A function f is symmetric about a point c if f(c + x) = f(c - x) for all x -/
def IsSymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

/-- The main theorem -/
theorem symmetric_absolute_value_function (a : ℝ) :
  IsSymmetricAbout (fun x ↦ |x + 2*a| - 1) 1 → a = -1/2 := by
  sorry

end symmetric_absolute_value_function_l2103_210323


namespace inequality_solution_set_l2103_210390

theorem inequality_solution_set (x : ℝ) : (3*x - 1) / (2 - x) ≥ 1 ↔ 3/4 ≤ x ∧ x < 2 := by
  sorry

end inequality_solution_set_l2103_210390


namespace square_elements_iff_odd_order_l2103_210365

theorem square_elements_iff_odd_order (G : Type*) [Group G] [Fintype G] :
  (∀ g : G, ∃ h : G, h ^ 2 = g) ↔ Odd (Fintype.card G) :=
sorry

end square_elements_iff_odd_order_l2103_210365


namespace integer_solution_system_l2103_210344

theorem integer_solution_system :
  ∀ x y z : ℤ,
  (x * y + y * z + z * x = -4) →
  (x^2 + y^2 + z^2 = 24) →
  (x^3 + y^3 + z^3 + 3*x*y*z = 16) →
  ((x = 2 ∧ y = -2 ∧ z = 4) ∨
   (x = 2 ∧ y = 4 ∧ z = -2) ∨
   (x = -2 ∧ y = 2 ∧ z = 4) ∨
   (x = -2 ∧ y = 4 ∧ z = 2) ∨
   (x = 4 ∧ y = 2 ∧ z = -2) ∨
   (x = 4 ∧ y = -2 ∧ z = 2)) :=
by sorry


end integer_solution_system_l2103_210344


namespace perpendicular_planes_not_transitive_l2103_210381

-- Define the type for planes
variable (Plane : Type)

-- Define the perpendicular relation between planes
variable (perp : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_not_transitive :
  ∃ (α β γ : Plane),
    α ≠ β ∧ β ≠ γ ∧ α ≠ γ ∧
    perp α β ∧ perp β γ ∧
    ¬(∀ (α β γ : Plane), perp α β → perp β γ → perp α γ) :=
sorry

end perpendicular_planes_not_transitive_l2103_210381


namespace least_three_digit_multiple_of_2_5_7_l2103_210384

theorem least_three_digit_multiple_of_2_5_7 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 140 → ¬(2 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) :=
by sorry

end least_three_digit_multiple_of_2_5_7_l2103_210384


namespace inequality_problem_l2103_210371

theorem inequality_problem :
  (∀ (x : ℝ), (∀ (m : ℝ), -2 ≤ m ∧ m ≤ 2 → 2 * x - 1 > m * (x^2 - 1)) ↔ 
    ((Real.sqrt 7 - 1) / 2 < x ∧ x < (Real.sqrt 3 + 1) / 2)) ∧
  (¬ ∃ (m : ℝ), ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → 2 * x - 1 > m * (x^2 - 1)) :=
by sorry

end inequality_problem_l2103_210371


namespace aaron_cards_proof_l2103_210308

def aaron_final_cards (initial_aaron : ℕ) (found : ℕ) (lost : ℕ) (given : ℕ) : ℕ :=
  initial_aaron + found - lost - given

theorem aaron_cards_proof (initial_arthur : ℕ) (initial_aaron : ℕ) (found : ℕ) (lost : ℕ) (given : ℕ)
  (h1 : initial_arthur = 6)
  (h2 : initial_aaron = 5)
  (h3 : found = 62)
  (h4 : lost = 15)
  (h5 : given = 28) :
  aaron_final_cards initial_aaron found lost given = 24 :=
by
  sorry

end aaron_cards_proof_l2103_210308


namespace determine_M_value_l2103_210358

theorem determine_M_value (a b c d : ℤ) (M : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
  a + b + c + d = 0 →
  M = (b * c - a * d) * (a * c - b * d) * (a * b - c * d) →
  96100 < M ∧ M < 98000 →
  M = 97344 := by
  sorry

end determine_M_value_l2103_210358


namespace meetings_percentage_of_workday_l2103_210350

-- Define the work day in minutes
def work_day_minutes : ℕ := 8 * 60

-- Define the durations of the meetings
def meeting1_duration : ℕ := 30
def meeting2_duration : ℕ := 60
def meeting3_duration : ℕ := meeting1_duration + meeting2_duration

-- Define the total meeting time
def total_meeting_time : ℕ := meeting1_duration + meeting2_duration + meeting3_duration

-- Theorem to prove
theorem meetings_percentage_of_workday :
  (total_meeting_time : ℚ) / work_day_minutes * 100 = 37.5 := by
  sorry


end meetings_percentage_of_workday_l2103_210350


namespace solve_for_time_l2103_210306

-- Define the exponential growth formula
def exponential_growth (P₀ A r t : ℝ) : Prop :=
  A = P₀ * Real.exp (r * t)

-- Theorem statement
theorem solve_for_time (P₀ A r t : ℝ) (h_pos : P₀ > 0) (h_r_nonzero : r ≠ 0) :
  exponential_growth P₀ A r t ↔ t = Real.log (A / P₀) / r :=
sorry

end solve_for_time_l2103_210306


namespace expression_equality_l2103_210329

theorem expression_equality : 
  (2^3 ≠ 3^2) ∧ 
  (-2^3 = (-2)^3) ∧ 
  ((-2 * 3)^2 ≠ -2 * 3^2) ∧ 
  ((-5)^2 ≠ -5^2) :=
by sorry

end expression_equality_l2103_210329


namespace parity_relation_l2103_210328

theorem parity_relation (a b : ℤ) : 
  (Even (5*b + a) → Even (a - 3*b)) ∧ 
  (Odd (5*b + a) → Odd (a - 3*b)) := by sorry

end parity_relation_l2103_210328


namespace fraction_equality_l2103_210393

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) (h1 : a / b = 3 / 4) : a / (a + b) = 3 / 7 := by
  sorry

end fraction_equality_l2103_210393


namespace w_squared_value_l2103_210391

theorem w_squared_value (w : ℚ) (h : 13 = 13 * w / (1 - w)) : w^2 = 1/4 := by
  sorry

end w_squared_value_l2103_210391


namespace polynomial_always_positive_l2103_210349

theorem polynomial_always_positive (x : ℝ) : x^12 - x^9 + x^4 - x + 1 > 0 := by
  sorry

end polynomial_always_positive_l2103_210349


namespace total_leaves_l2103_210361

/-- The number of ferns Karen hangs around her house. -/
def num_ferns : ℕ := 12

/-- The number of fronds each fern has. -/
def fronds_per_fern : ℕ := 15

/-- The number of leaves each frond has. -/
def leaves_per_frond : ℕ := 45

/-- Theorem stating the total number of leaves on all ferns. -/
theorem total_leaves : num_ferns * fronds_per_fern * leaves_per_frond = 8100 := by
  sorry

end total_leaves_l2103_210361


namespace sum_product_inequality_l2103_210320

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 1) : a * b + b * c + c * a ≤ 1/3 := by
  sorry

end sum_product_inequality_l2103_210320


namespace line_through_interior_point_no_intersection_l2103_210342

/-- Theorem: A line through a point inside a parabola has no intersection with the parabola --/
theorem line_through_interior_point_no_intersection 
  (x y_o : ℝ) (h : y_o^2 < 4*x) : 
  ∀ y : ℝ, (y^2 = 4*((y*y_o)/(2) - x)) → False :=
by sorry

end line_through_interior_point_no_intersection_l2103_210342


namespace product_of_decimals_l2103_210364

theorem product_of_decimals : 3.6 * 0.04 = 0.144 := by
  sorry

end product_of_decimals_l2103_210364


namespace circular_pool_volume_l2103_210382

/-- The volume of a circular pool with given dimensions -/
theorem circular_pool_volume (diameter : ℝ) (depth1 : ℝ) (depth2 : ℝ) :
  diameter = 20 →
  depth1 = 3 →
  depth2 = 5 →
  (π * (diameter / 2)^2 * depth1 + π * (diameter / 2)^2 * depth2) = 800 * π := by
  sorry

end circular_pool_volume_l2103_210382


namespace orange_harvest_total_l2103_210375

/-- The number of days the orange harvest lasts -/
def harvest_days : ℕ := 4

/-- The number of sacks harvested per day -/
def sacks_per_day : ℕ := 14

/-- The total number of sacks harvested -/
def total_sacks : ℕ := harvest_days * sacks_per_day

theorem orange_harvest_total :
  total_sacks = 56 := by
  sorry

end orange_harvest_total_l2103_210375


namespace probability_b_greater_than_a_l2103_210332

def A : Finset ℕ := {2, 3, 4, 5, 6}
def B : Finset ℕ := {1, 2, 3, 5}

theorem probability_b_greater_than_a : 
  (Finset.filter (λ (p : ℕ × ℕ) => p.2 > p.1) (A.product B)).card / (A.card * B.card : ℚ) = 1/5 :=
sorry

end probability_b_greater_than_a_l2103_210332


namespace base7_to_base10_65432_l2103_210314

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number --/
def base7Number : List Nat := [2, 3, 4, 5, 6]

/-- Theorem stating that the base 10 equivalent of 65432 in base 7 is 16340 --/
theorem base7_to_base10_65432 :
  base7ToBase10 base7Number = 16340 := by
  sorry

end base7_to_base10_65432_l2103_210314


namespace hyperbola_foci_distance_l2103_210363

/-- The distance between the foci of the hyperbola xy = 2 is 2√2 -/
theorem hyperbola_foci_distance :
  ∃ (f₁ f₂ : ℝ × ℝ),
    (∀ (x y : ℝ), x * y = 2 → (x - f₁.1) * (y - f₁.2) = (x - f₂.1) * (y - f₂.2)) ∧
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 2 * Real.sqrt 2 :=
by sorry

end hyperbola_foci_distance_l2103_210363


namespace image_of_two_zero_l2103_210333

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

-- Theorem statement
theorem image_of_two_zero :
  f (2, 0) = (2, 2) := by
  sorry

end image_of_two_zero_l2103_210333


namespace bar_chart_ratio_difference_l2103_210353

theorem bar_chart_ratio_difference 
  (a b c d : ℝ) 
  (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) :
  a / (a + b) - c / (c + d) = (a * d - b * c) / ((a + b) * (c + d)) :=
by sorry

end bar_chart_ratio_difference_l2103_210353


namespace fraction_decomposition_l2103_210324

theorem fraction_decomposition (x A B : ℝ) : 
  (8 * x - 17) / (3 * x^2 + 4 * x - 15) = A / (3 * x + 5) + B / (x - 3) →
  A = 6.5 ∧ B = 0.5 := by
sorry

end fraction_decomposition_l2103_210324


namespace min_value_of_expression_l2103_210389

theorem min_value_of_expression :
  (∀ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2023 ≥ 2022) ∧
  (∃ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2023 = 2022) := by
  sorry

end min_value_of_expression_l2103_210389


namespace ellipse_line_intersection_area_l2103_210346

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line in the form y = mx + c -/
structure Line where
  m : ℝ
  c : ℝ

/-- Calculate the area of triangle AOB formed by the intersection of an ellipse and a line -/
def area_triangle_AOB (e : Ellipse) (l : Line) : ℝ :=
  sorry

theorem ellipse_line_intersection_area :
  ∀ (e : Ellipse) (l : Line),
    e.b = 1 →
    e.a^2 = 2 →
    l.m = 1 ∧ l.c = Real.sqrt 2 →
    area_triangle_AOB e l = 2/3 :=
  sorry

end ellipse_line_intersection_area_l2103_210346


namespace target_destruction_probability_l2103_210317

def prob_at_least_two (p1 p2 p3 : ℝ) : ℝ :=
  p1 * p2 * p3 +
  p1 * p2 * (1 - p3) +
  p1 * (1 - p2) * p3 +
  (1 - p1) * p2 * p3

theorem target_destruction_probability :
  prob_at_least_two 0.9 0.9 0.8 = 0.954 := by
  sorry

end target_destruction_probability_l2103_210317


namespace laundromat_cost_l2103_210307

def service_fee : ℝ := 3
def first_hour_cost : ℝ := 10
def additional_hour_cost : ℝ := 15
def usage_time : ℝ := 2.75
def discount_rate : ℝ := 0.1

def calculate_cost : ℝ :=
  let base_cost := first_hour_cost + (usage_time - 1) * additional_hour_cost
  let total_cost := base_cost + service_fee
  let discount := total_cost * discount_rate
  total_cost - discount

theorem laundromat_cost :
  calculate_cost = 35.32 := by sorry

end laundromat_cost_l2103_210307


namespace factorization_equality_l2103_210338

theorem factorization_equality (b : ℝ) : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) := by
  sorry

end factorization_equality_l2103_210338


namespace not_right_triangle_l2103_210337

theorem not_right_triangle (a b c : ℕ) (h : a = 3 ∧ b = 4 ∧ c = 6) : 
  ¬(a^2 + b^2 = c^2) := by
  sorry

#check not_right_triangle

end not_right_triangle_l2103_210337


namespace inequality_condition_l2103_210351

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, -2 < x ∧ x < -1 → (a*x + 1)*(1 + x) < 0) ∧
  (∃ x : ℝ, (a*x + 1)*(1 + x) < 0 ∧ (x ≤ -2 ∨ x ≥ -1)) →
  0 ≤ a ∧ a < 1/2 :=
by sorry

end inequality_condition_l2103_210351


namespace binomial_square_constant_l2103_210315

theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + c = (a * x + b)^2) → c = 25 := by
  sorry

end binomial_square_constant_l2103_210315


namespace sum_of_fractions_equals_one_l2103_210388

theorem sum_of_fractions_equals_one
  (a b c p q r : ℝ)
  (eq1 : 19 * p + b * q + c * r = 0)
  (eq2 : a * p + 29 * q + c * r = 0)
  (eq3 : a * p + b * q + 56 * r = 0)
  (ha : a ≠ 19)
  (hp : p ≠ 0) :
  a / (a - 19) + b / (b - 29) + c / (c - 56) = 1 := by
sorry

end sum_of_fractions_equals_one_l2103_210388


namespace sum_in_base7_l2103_210325

/-- Converts a base 7 number to base 10 --/
def toBase10 (x : List Nat) : Nat :=
  x.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

/-- Converts a base 10 number to base 7 --/
def toBase7 (x : Nat) : List Nat :=
  if x = 0 then [0] else
  let rec aux (n : Nat) (acc : List Nat) : List Nat :=
    if n = 0 then acc else aux (n / 7) ((n % 7) :: acc)
  aux x []

theorem sum_in_base7 :
  let a := [2, 3, 4]  -- 432 in base 7
  let b := [4, 5]     -- 54 in base 7
  let c := [6]        -- 6 in base 7
  let sum := toBase10 a + toBase10 b + toBase10 c
  toBase7 sum = [5, 2, 5] := by sorry

end sum_in_base7_l2103_210325


namespace robert_gets_two_more_than_kate_l2103_210316

/-- The number of candy pieces each child receives. -/
structure CandyDistribution where
  robert : ℕ
  kate : ℕ
  bill : ℕ
  mary : ℕ

/-- The conditions of the candy distribution problem. -/
def ValidDistribution (d : CandyDistribution) : Prop :=
  d.robert + d.kate + d.bill + d.mary = 20 ∧
  d.robert > d.kate ∧
  d.bill = d.mary - 6 ∧
  d.mary = d.robert + 2 ∧
  d.kate = d.bill + 2 ∧
  d.kate = 4

theorem robert_gets_two_more_than_kate (d : CandyDistribution) 
  (h : ValidDistribution d) : d.robert - d.kate = 2 := by
  sorry

end robert_gets_two_more_than_kate_l2103_210316


namespace cos_2alpha_value_l2103_210348

theorem cos_2alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin α - Real.cos α = Real.sqrt 5 / 5) : 
  Real.cos (2 * α) = -3 / 5 := by
  sorry

end cos_2alpha_value_l2103_210348


namespace two_rolls_give_target_prob_l2103_210330

-- Define a type for a six-sided die
def Die := Fin 6

-- Define the number of sides on the die
def numSides : ℕ := 6

-- Define the target sum
def targetSum : ℕ := 9

-- Define the target probability
def targetProb : ℚ := 1 / 9

-- Function to calculate the number of ways to get a sum of 9 with n rolls
def waysToGetSum (n : ℕ) : ℕ := sorry

-- Function to calculate the total number of possible outcomes with n rolls
def totalOutcomes (n : ℕ) : ℕ := numSides ^ n

-- Theorem stating that rolling the die twice gives the target probability
theorem two_rolls_give_target_prob :
  ∃ (n : ℕ), (waysToGetSum n : ℚ) / (totalOutcomes n) = targetProb ∧ n = 2 := by sorry

end two_rolls_give_target_prob_l2103_210330


namespace rectangles_form_square_l2103_210300

/-- A rectangle represented by its width and height --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculate the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Check if a list of rectangles can form a square of side length 16 --/
def canFormSquare (rectangles : List Rectangle) : Prop :=
  ∃ (arrangement : List Rectangle), 
    arrangement.length = rectangles.length ∧
    (∀ r ∈ arrangement, r ∈ rectangles) ∧
    (∀ r ∈ rectangles, r ∈ arrangement) ∧
    arrangement.foldr (λ r acc => acc + r.width * r.height) 0 = 16 * 16

/-- The main theorem to prove --/
theorem rectangles_form_square :
  ∃ (rectangles : List Rectangle),
    rectangles.foldr (λ r acc => acc + perimeter r) 0 = 100 ∧
    canFormSquare rectangles := by sorry

end rectangles_form_square_l2103_210300


namespace school_student_count_l2103_210396

theorem school_student_count (total : ℕ) (junior_increase senior_increase total_increase : ℚ) 
  (h1 : total = 4200)
  (h2 : junior_increase = 8 / 100)
  (h3 : senior_increase = 11 / 100)
  (h4 : total_increase = 10 / 100) :
  ∃ (junior senior : ℕ), 
    junior + senior = total ∧
    (1 + junior_increase) * junior + (1 + senior_increase) * senior = (1 + total_increase) * total ∧
    junior = 1400 ∧
    senior = 2800 := by
  sorry

end school_student_count_l2103_210396


namespace eight_positions_l2103_210373

def number : ℚ := 38.82

theorem eight_positions (n : ℚ) (h : n = number) : 
  (n - 38 = 0.82) ∧ 
  (n - 38.8 = 0.02) :=
by sorry

end eight_positions_l2103_210373


namespace largest_package_size_l2103_210369

theorem largest_package_size (a b c : ℕ) (ha : a = 60) (hb : b = 36) (hc : c = 48) :
  Nat.gcd a (Nat.gcd b c) = 12 := by
  sorry

end largest_package_size_l2103_210369


namespace fraction_equation_solution_l2103_210352

theorem fraction_equation_solution (P Q : ℤ) :
  (∀ x : ℝ, x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 5 →
    (P / (x + 7 : ℝ)) + (Q / (x^2 - 6*x : ℝ)) = ((x^2 - 6*x + 14) / (x^3 + x^2 - 30*x) : ℝ)) →
  (Q : ℚ) / (P : ℚ) = 12 := by
sorry

end fraction_equation_solution_l2103_210352


namespace right_angled_figure_l2103_210357

def top_side (X : ℝ) : ℝ := 2 + 1 + 3 + X
def bottom_side : ℝ := 3 + 4 + 5

theorem right_angled_figure (X : ℝ) : 
  top_side X = bottom_side → X = 6 := by
  sorry

end right_angled_figure_l2103_210357


namespace cousins_money_correct_l2103_210379

/-- The amount of money Jim's cousin brought to the restaurant --/
def cousins_money : ℝ := 10

/-- The total cost of the meal --/
def meal_cost : ℝ := 24

/-- The percentage of their combined money spent on the meal --/
def spent_percentage : ℝ := 0.8

/-- The amount of money Jim brought --/
def jims_money : ℝ := 20

/-- Theorem stating that the calculated amount Jim's cousin brought is correct --/
theorem cousins_money_correct : 
  cousins_money = (meal_cost / spent_percentage) - jims_money := by
  sorry

end cousins_money_correct_l2103_210379


namespace money_redistribution_theorem_l2103_210386

/-- Represents the money redistribution problem with four friends --/
def MoneyRedistribution (a b j t : ℕ) : Prop :=
  -- Initial conditions
  a ≠ b ∧ a ≠ j ∧ a ≠ t ∧ b ≠ j ∧ b ≠ t ∧ j ≠ t ∧
  -- Toy starts and ends with the same amount
  t = 48 ∧
  -- After four rounds of redistribution
  ∃ (a₁ b₁ j₁ t₁ : ℕ),
    -- First round (Amy redistributes)
    a₁ + b₁ + j₁ + t₁ = a + b + j + t ∧
    b₁ = 2 * b ∧ j₁ = 2 * j ∧ t₁ = 2 * t ∧
    ∃ (a₂ b₂ j₂ t₂ : ℕ),
      -- Second round (Beth redistributes)
      a₂ + b₂ + j₂ + t₂ = a₁ + b₁ + j₁ + t₁ ∧
      a₂ = 2 * a₁ ∧ j₂ = 2 * j₁ ∧ t₂ = 2 * t₁ ∧
      ∃ (a₃ b₃ j₃ t₃ : ℕ),
        -- Third round (Jan redistributes)
        a₃ + b₃ + j₃ + t₃ = a₂ + b₂ + j₂ + t₂ ∧
        a₃ = 2 * a₂ ∧ b₃ = 2 * b₂ ∧ t₃ = 2 * t₂ ∧
        ∃ (a₄ b₄ j₄ t₄ : ℕ),
          -- Fourth round (Toy redistributes)
          a₄ + b₄ + j₄ + t₄ = a₃ + b₃ + j₃ + t₃ ∧
          a₄ = 2 * a₃ ∧ b₄ = 2 * b₃ ∧ j₄ = 2 * j₃ ∧
          -- Toy ends with the same amount
          t₄ = t

/-- The theorem stating that the total money is 15 times Toy's amount --/
theorem money_redistribution_theorem {a b j t : ℕ} (h : MoneyRedistribution a b j t) :
  a + b + j + t = 15 * t :=
sorry

end money_redistribution_theorem_l2103_210386


namespace polynomial_division_remainder_l2103_210359

theorem polynomial_division_remainder 
  (dividend : Polynomial ℚ) 
  (divisor : Polynomial ℚ) 
  (quotient : Polynomial ℚ) 
  (remainder : Polynomial ℚ) :
  dividend = 3 * X^4 + 7 * X^3 - 28 * X^2 - 32 * X + 53 →
  divisor = X^2 + 5 * X + 3 →
  dividend = divisor * quotient + remainder →
  Polynomial.degree remainder < Polynomial.degree divisor →
  remainder = 97 * X + 116 := by
    sorry

end polynomial_division_remainder_l2103_210359


namespace percentage_of_cat_owners_l2103_210394

theorem percentage_of_cat_owners (total_students : ℕ) (cat_owners : ℕ) 
  (h1 : total_students = 300) (h2 : cat_owners = 30) : 
  (cat_owners : ℚ) / total_students * 100 = 10 := by
  sorry

end percentage_of_cat_owners_l2103_210394


namespace extreme_values_and_range_l2103_210318

-- Define the function f
def f (a b c x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8 * c

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_and_range (a b c : ℝ) :
  (∀ x : ℝ, f' a b x = 0 ↔ x = 1 ∨ x = 2) →
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f a b c x < c^2) →
  (a = -3 ∧ b = 4) ∧ (c < -1 ∨ c > 9) := by
  sorry

#check extreme_values_and_range

end extreme_values_and_range_l2103_210318


namespace problem_statement_l2103_210372

theorem problem_statement (a b c d : ℕ) 
  (h1 : d ∣ a^(2*b) + c) 
  (h2 : d ≥ a + c) : 
  d ≥ a + a^(1/(2*b)) := by
  sorry

end problem_statement_l2103_210372


namespace sum_of_digits_power_of_six_l2103_210302

def last_two_digits (n : ℕ) : ℕ := n % 100

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_power_of_six :
  tens_digit (last_two_digits ((4 + 2)^21)) + ones_digit (last_two_digits ((4 + 2)^21)) = 6 := by
  sorry

end sum_of_digits_power_of_six_l2103_210302


namespace original_cost_l2103_210392

theorem original_cost (final_cost : ℝ) : 
  final_cost = 72 → 
  ∃ (original_cost : ℝ), 
    original_cost * (1 + 0.2) * (1 - 0.2) = final_cost ∧ 
    original_cost = 75 :=
by
  sorry

end original_cost_l2103_210392


namespace number_of_hiding_snakes_l2103_210367

/-- Given a cage with snakes, some of which are hiding, this theorem proves
    the number of hiding snakes. -/
theorem number_of_hiding_snakes
  (total_snakes : ℕ)
  (visible_snakes : ℕ)
  (h1 : total_snakes = 95)
  (h2 : visible_snakes = 31) :
  total_snakes - visible_snakes = 64 := by
  sorry

end number_of_hiding_snakes_l2103_210367
