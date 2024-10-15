import Mathlib

namespace NUMINAMATH_CALUDE_cylinder_height_equals_half_cm_l3636_363663

/-- Given a cylinder and a sphere with specific dimensions, proves that the height of the cylinder is 0.5 cm when their volumes are equal. -/
theorem cylinder_height_equals_half_cm 
  (d_cylinder : ℝ) 
  (d_sphere : ℝ) 
  (h_cylinder : ℝ) :
  d_cylinder = 6 →
  d_sphere = 3 →
  π * (d_cylinder / 2)^2 * h_cylinder = (4/3) * π * (d_sphere / 2)^3 →
  h_cylinder = 0.5 := by
  sorry

#check cylinder_height_equals_half_cm

end NUMINAMATH_CALUDE_cylinder_height_equals_half_cm_l3636_363663


namespace NUMINAMATH_CALUDE_special_sequence_bound_l3636_363690

def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n m, n < m → a n < a m) ∧ 
  (∀ k : ℕ, k ∈ Set.range a ∨ ∃ i j, k = a i + a j)

theorem special_sequence_bound (a : ℕ → ℕ) (h : SpecialSequence a) : 
  ∀ n : ℕ, n > 0 → a n ≤ n^2 := by
sorry

end NUMINAMATH_CALUDE_special_sequence_bound_l3636_363690


namespace NUMINAMATH_CALUDE_x_value_l3636_363626

theorem x_value (w y z x : ℤ) 
  (hw : w = 95)
  (hz : z = w + 25)
  (hy : y = z + 15)
  (hx : x = y + 7) : 
  x = 142 := by sorry

end NUMINAMATH_CALUDE_x_value_l3636_363626


namespace NUMINAMATH_CALUDE_value_difference_is_50p_minus_250_l3636_363636

/-- The value of a fifty-cent coin in pennies -/
def fifty_cent_value : ℕ := 50

/-- The number of fifty-cent coins Liam has -/
def liam_coins (p : ℕ) : ℕ := 3 * p + 2

/-- The number of fifty-cent coins Mia has -/
def mia_coins (p : ℕ) : ℕ := 2 * p + 7

/-- The difference in total value (in pennies) between Liam's and Mia's fifty-cent coins -/
def value_difference (p : ℕ) : ℤ := fifty_cent_value * (liam_coins p - mia_coins p)

theorem value_difference_is_50p_minus_250 (p : ℕ) :
  value_difference p = 50 * p - 250 := by sorry

end NUMINAMATH_CALUDE_value_difference_is_50p_minus_250_l3636_363636


namespace NUMINAMATH_CALUDE_matrix_is_own_inverse_l3636_363611

def A (c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; c, d]

theorem matrix_is_own_inverse (c d : ℝ) :
  A c d * A c d = 1 ↔ c = 7.5 ∧ d = -4 := by
  sorry

end NUMINAMATH_CALUDE_matrix_is_own_inverse_l3636_363611


namespace NUMINAMATH_CALUDE_g_prime_zero_f_symmetry_f_prime_symmetry_l3636_363681

-- Define the functions and their derivatives
variable (f g : ℝ → ℝ)
variable (f' g' : ℝ → ℝ)

-- Define the conditions
axiom func_relation : ∀ x, f (x + 3) = g (-x) + 4
axiom deriv_relation : ∀ x, f' x + g' (1 + x) = 0
axiom g_even : ∀ x, g (2*x + 1) = g (-2*x + 1)

-- Define the derivative relationship
axiom f_deriv : ∀ x, (deriv f) x = f' x
axiom g_deriv : ∀ x, (deriv g) x = g' x

-- State the theorems to be proved
theorem g_prime_zero : g' 1 = 0 := by sorry

theorem f_symmetry : ∀ x, f (x + 4) = f x := by sorry

theorem f_prime_symmetry : ∀ x, f' (x + 2) = f' x := by sorry

end NUMINAMATH_CALUDE_g_prime_zero_f_symmetry_f_prime_symmetry_l3636_363681


namespace NUMINAMATH_CALUDE_inequality_proof_l3636_363658

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * (a + b)) + Real.sqrt (b * c * (b + c)) + Real.sqrt (c * a * (c + a)) >
  Real.sqrt ((a + b) * (b + c) * (c + a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3636_363658


namespace NUMINAMATH_CALUDE_system_solution_l3636_363630

theorem system_solution :
  let S := {(x, y) : ℝ × ℝ | x^2 - 9*y^2 = 0 ∧ x^2 + y^2 = 9}
  S = {(9/Real.sqrt 10, 3/Real.sqrt 10), (-9/Real.sqrt 10, 3/Real.sqrt 10),
       (9/Real.sqrt 10, -3/Real.sqrt 10), (-9/Real.sqrt 10, -3/Real.sqrt 10)} :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3636_363630


namespace NUMINAMATH_CALUDE_parabola_translation_l3636_363610

/-- Represents a parabola in the form y = a(x - h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a
    h := p.h - dx
    k := p.k + dy }

theorem parabola_translation (p : Parabola) :
  p.a = -1/3 ∧ p.h = 5 ∧ p.k = 3 →
  let p' := translate p 5 3
  p'.a = -1/3 ∧ p'.h = 0 ∧ p'.k = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l3636_363610


namespace NUMINAMATH_CALUDE_pizza_slices_per_person_l3636_363637

theorem pizza_slices_per_person (num_people : ℕ) (num_pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_people = 6)
  (h2 : num_pizzas = 3)
  (h3 : slices_per_pizza = 8) :
  (num_pizzas * slices_per_pizza) / num_people = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_per_person_l3636_363637


namespace NUMINAMATH_CALUDE_cyclists_meeting_time_l3636_363668

/-- Two cyclists meet on a course -/
theorem cyclists_meeting_time
  (course_length : ℝ)
  (speed1 : ℝ)
  (speed2 : ℝ)
  (h1 : course_length = 45)
  (h2 : speed1 = 14)
  (h3 : speed2 = 16) :
  ∃ t : ℝ, t * (speed1 + speed2) = course_length ∧ t = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_cyclists_meeting_time_l3636_363668


namespace NUMINAMATH_CALUDE_math_club_attendance_l3636_363680

theorem math_club_attendance (total_sessions : Nat) (students_per_session : Nat)
  (three_session_attendees : Nat) (two_session_attendees : Nat) (one_session_attendees : Nat)
  (h1 : total_sessions = 4)
  (h2 : students_per_session = 20)
  (h3 : three_session_attendees = 9)
  (h4 : two_session_attendees = 5)
  (h5 : one_session_attendees = 3) :
  ∃ (all_session_attendees : Nat),
    all_session_attendees * total_sessions +
    three_session_attendees * 3 +
    two_session_attendees * 2 +
    one_session_attendees * 1 =
    total_sessions * students_per_session ∧
    all_session_attendees = 10 := by
  sorry

end NUMINAMATH_CALUDE_math_club_attendance_l3636_363680


namespace NUMINAMATH_CALUDE_tan_22_5_deg_over_one_minus_tan_squared_l3636_363671

theorem tan_22_5_deg_over_one_minus_tan_squared (
  angle_22_5 : ℝ)
  (h1 : 45 * Real.pi / 180 = 2 * angle_22_5)
  (h2 : Real.tan (45 * Real.pi / 180) = 1)
  (h3 : ∀ θ : ℝ, Real.tan (2 * θ) = (2 * Real.tan θ) / (1 - Real.tan θ ^ 2)) :
  Real.tan angle_22_5 / (1 - Real.tan angle_22_5 ^ 2) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_tan_22_5_deg_over_one_minus_tan_squared_l3636_363671


namespace NUMINAMATH_CALUDE_grey_pairs_coincide_l3636_363652

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  green : Nat
  yellow : Nat
  grey : Nat

/-- Represents the number of coinciding pairs of each type when folded -/
structure CoincidingPairs where
  green_green : Nat
  yellow_yellow : Nat
  green_grey : Nat
  grey_grey : Nat

/-- The main theorem statement -/
theorem grey_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) : 
  counts.green = 4 ∧ 
  counts.yellow = 6 ∧ 
  counts.grey = 10 ∧
  pairs.green_green = 3 ∧
  pairs.yellow_yellow = 4 ∧
  pairs.green_grey = 3 →
  pairs.grey_grey = 5 := by
  sorry

end NUMINAMATH_CALUDE_grey_pairs_coincide_l3636_363652


namespace NUMINAMATH_CALUDE_nested_square_root_equality_l3636_363601

theorem nested_square_root_equality : Real.sqrt (13 + Real.sqrt (7 + Real.sqrt 4)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_equality_l3636_363601


namespace NUMINAMATH_CALUDE_largest_integer_l3636_363602

theorem largest_integer (x y z w : ℤ) 
  (sum1 : x + y + z = 234)
  (sum2 : x + y + w = 255)
  (sum3 : x + z + w = 271)
  (sum4 : y + z + w = 198) :
  max x (max y (max z w)) = 121 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_l3636_363602


namespace NUMINAMATH_CALUDE_puzzle_solutions_l3636_363604

def is_valid_solution (a b : Nat) : Prop :=
  a ≠ b ∧
  a ≥ 1 ∧ a ≤ 9 ∧
  b ≥ 1 ∧ b ≤ 9 ∧
  a^b = 10*b + a ∧
  10*b + a ≠ b*a

theorem puzzle_solutions :
  {(a, b) : Nat × Nat | is_valid_solution a b} =
  {(2, 5), (6, 2), (4, 3)} :=
sorry

end NUMINAMATH_CALUDE_puzzle_solutions_l3636_363604


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l3636_363691

theorem bobby_candy_problem (initial_candy : ℕ) : 
  initial_candy + 4 + 14 = 51 → initial_candy = 33 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l3636_363691


namespace NUMINAMATH_CALUDE_mountain_climb_speed_l3636_363628

theorem mountain_climb_speed 
  (total_time : ℝ) 
  (speed_difference : ℝ) 
  (time_difference : ℝ) 
  (total_distance : ℝ) 
  (h1 : total_time = 14) 
  (h2 : speed_difference = 0.5) 
  (h3 : time_difference = 2) 
  (h4 : total_distance = 52) : 
  ∃ (v : ℝ), v > 0 ∧ 
    v * (total_time / 2 + time_difference) + 
    (v + speed_difference) * (total_time / 2 - time_difference) = total_distance ∧
    v + speed_difference = 4 := by
  sorry

#check mountain_climb_speed

end NUMINAMATH_CALUDE_mountain_climb_speed_l3636_363628


namespace NUMINAMATH_CALUDE_income_ratio_is_5_to_4_l3636_363650

-- Define the incomes and expenditures
def income_A : ℕ := 4000
def income_B : ℕ := 3200
def expenditure_A : ℕ := 2400
def expenditure_B : ℕ := 1600

-- Define the savings
def savings : ℕ := 1600

-- Theorem to prove
theorem income_ratio_is_5_to_4 :
  -- Conditions
  (expenditure_A / expenditure_B = 3 / 2) ∧
  (income_A - expenditure_A = savings) ∧
  (income_B - expenditure_B = savings) ∧
  (income_A = 4000) →
  -- Conclusion
  (income_A : ℚ) / (income_B : ℚ) = 5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_income_ratio_is_5_to_4_l3636_363650


namespace NUMINAMATH_CALUDE_max_sum_is_42_l3636_363640

/-- Represents the configuration of numbers in the squares -/
structure SquareConfig where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  numbers : Finset ℕ
  sum_equality : a + b + e = b + d + e
  valid_numbers : numbers = {2, 5, 8, 11, 14, 17}
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

/-- The maximum sum of either horizontal or vertical line is 42 -/
theorem max_sum_is_42 (config : SquareConfig) : 
  (max (config.a + config.b + config.e) (config.b + config.d + config.e)) ≤ 42 ∧ 
  ∃ (config : SquareConfig), (config.a + config.b + config.e) = 42 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_is_42_l3636_363640


namespace NUMINAMATH_CALUDE_right_triangle_and_inverse_l3636_363676

theorem right_triangle_and_inverse (a b c : Nat) (m : Nat) : 
  a = 48 → b = 55 → c = 73 → m = 4273 →
  a * a + b * b = c * c →
  (∃ (x : Nat), x * 480 ≡ 1 [MOD m]) →
  (∃ (y : Nat), y * 480 ≡ 1643 [MOD m]) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_and_inverse_l3636_363676


namespace NUMINAMATH_CALUDE_elise_cab_ride_cost_l3636_363685

/-- Calculates the total cost of a cab ride given the base price, cost per mile, and distance traveled. -/
def cab_ride_cost (base_price : ℝ) (cost_per_mile : ℝ) (distance : ℝ) : ℝ :=
  base_price + cost_per_mile * distance

/-- Proves that Elise's cab ride cost $23 -/
theorem elise_cab_ride_cost : cab_ride_cost 3 4 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_elise_cab_ride_cost_l3636_363685


namespace NUMINAMATH_CALUDE_no_solution_implies_m_leq_two_l3636_363649

theorem no_solution_implies_m_leq_two (m : ℝ) : 
  (∀ x : ℝ, ¬(x - 1 > 1 ∧ x < m)) → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_leq_two_l3636_363649


namespace NUMINAMATH_CALUDE_mac_total_loss_l3636_363651

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | "quarter" => 25
  | "half-dollar" => 50
  | _ => 0

/-- Calculates the expected loss for a single trade -/
def expected_loss (given_coins : List String) (probability : ℚ) : ℚ :=
  let given_value : ℚ := (given_coins.map coin_value).sum
  let quarter_value : ℚ := coin_value "quarter"
  (given_value - quarter_value) * probability

/-- Represents Mac's trading scenario -/
def mac_trades : List (List String × ℚ × ℕ) := [
  (["dime", "dime", "dime", "dime", "penny", "penny"], 1/20, 20),
  (["nickel", "nickel", "nickel", "nickel", "nickel", "nickel", "nickel", "nickel", "nickel", "penny"], 1/10, 20),
  (["half-dollar", "penny", "penny", "penny"], 17/20, 20)
]

/-- Theorem stating the total expected loss for Mac's trades -/
theorem mac_total_loss :
  (mac_trades.map (λ (coins, prob, repeats) => expected_loss coins prob * repeats)).sum = 535/100 := by
  sorry


end NUMINAMATH_CALUDE_mac_total_loss_l3636_363651


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3636_363629

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x^4 + y^2 = 3*y + 3 := by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3636_363629


namespace NUMINAMATH_CALUDE_square_sum_formula_l3636_363631

theorem square_sum_formula (x y a b : ℝ) 
  (h1 : x * y = 2 * b) 
  (h2 : 1 / x^2 + 1 / y^2 = a) : 
  (x + y)^2 = 4 * a * b^2 + 4 * b := by
  sorry

end NUMINAMATH_CALUDE_square_sum_formula_l3636_363631


namespace NUMINAMATH_CALUDE_movie_theater_tickets_l3636_363633

theorem movie_theater_tickets (matinee_price evening_price threeD_price : ℕ)
  (evening_sold threeD_sold total_revenue : ℕ) :
  matinee_price = 5 →
  evening_price = 12 →
  threeD_price = 20 →
  evening_sold = 300 →
  threeD_sold = 100 →
  total_revenue = 6600 →
  ∃ (matinee_sold : ℕ), 
    matinee_sold * matinee_price + 
    evening_sold * evening_price + 
    threeD_sold * threeD_price = total_revenue ∧
    matinee_sold = 200 :=
by sorry

end NUMINAMATH_CALUDE_movie_theater_tickets_l3636_363633


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l3636_363620

/-- Time for a train to pass a jogger given their speeds, train length, and initial distance -/
theorem train_passing_jogger_time 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (initial_distance : ℝ) 
  (h1 : jogger_speed = 9) 
  (h2 : train_speed = 45) 
  (h3 : train_length = 210) 
  (h4 : initial_distance = 240) : 
  (initial_distance + train_length) / (train_speed - jogger_speed) * (3600 / 1000) = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_l3636_363620


namespace NUMINAMATH_CALUDE_not_coprime_sum_equal_l3636_363670

/-- For any two natural numbers a and b, if a+n and b+n are not coprime for all natural numbers n, then a = b. -/
theorem not_coprime_sum_equal (a b : ℕ) 
  (h : ∀ n : ℕ, ¬ Nat.Coprime (a + n) (b + n)) : 
  a = b := by
  sorry

end NUMINAMATH_CALUDE_not_coprime_sum_equal_l3636_363670


namespace NUMINAMATH_CALUDE_female_democrats_count_l3636_363689

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 810 →
  female + male = total →
  (female / 2 + male / 4 : ℚ) = total / 3 →
  female / 2 = 135 :=
by sorry

end NUMINAMATH_CALUDE_female_democrats_count_l3636_363689


namespace NUMINAMATH_CALUDE_probability_two_yellow_balls_probability_two_yellow_balls_is_one_third_l3636_363643

/-- The probability of drawing two yellow balls from a bag containing 1 white and 2 yellow balls --/
theorem probability_two_yellow_balls : ℚ :=
  let total_balls : ℕ := 3
  let yellow_balls : ℕ := 2
  let first_draw : ℚ := yellow_balls / total_balls
  let second_draw : ℚ := (yellow_balls - 1) / (total_balls - 1)
  first_draw * second_draw

/-- Proof that the probability of drawing two yellow balls is 1/3 --/
theorem probability_two_yellow_balls_is_one_third :
  probability_two_yellow_balls = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_yellow_balls_probability_two_yellow_balls_is_one_third_l3636_363643


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l3636_363679

theorem sqrt_fraction_simplification :
  Real.sqrt (8^2 + 15^2) / Real.sqrt (25 + 16) = 17 * Real.sqrt 41 / 41 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l3636_363679


namespace NUMINAMATH_CALUDE_binomial_60_3_l3636_363684

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l3636_363684


namespace NUMINAMATH_CALUDE_average_side_length_of_squares_l3636_363617

theorem average_side_length_of_squares (a b c : ℝ) 
  (ha : a = 25) (hb : b = 64) (hc : c = 144) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_side_length_of_squares_l3636_363617


namespace NUMINAMATH_CALUDE_sum_of_roots_is_51_l3636_363653

-- Define the function f
def f (x : ℝ) : ℝ := 16 * x + 3

-- Define the inverse function of f
noncomputable def f_inv (x : ℝ) : ℝ := (x - 3) / 16

-- Theorem statement
theorem sum_of_roots_is_51 :
  ∃ (x₁ x₂ : ℝ), 
    (f_inv x₁ = f ((2 * x₁)⁻¹)) ∧
    (f_inv x₂ = f ((2 * x₂)⁻¹)) ∧
    (∀ x : ℝ, f_inv x = f ((2 * x)⁻¹) → x = x₁ ∨ x = x₂) ∧
    x₁ + x₂ = 51 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_51_l3636_363653


namespace NUMINAMATH_CALUDE_cone_from_sector_l3636_363659

theorem cone_from_sector (θ : Real) (r : Real) (base_radius : Real) (slant : Real) : 
  θ = 270 ∧ r = 12 ∧ base_radius = 9 ∧ slant = 12 →
  (θ / 360) * (2 * Real.pi * r) = 2 * Real.pi * base_radius ∧
  r = slant :=
by sorry

end NUMINAMATH_CALUDE_cone_from_sector_l3636_363659


namespace NUMINAMATH_CALUDE_fans_with_all_items_count_l3636_363606

/-- The capacity of the stadium --/
def stadium_capacity : ℕ := 5000

/-- The interval for hot dog coupons --/
def hot_dog_interval : ℕ := 60

/-- The interval for soda coupons --/
def soda_interval : ℕ := 40

/-- The interval for ice cream coupons --/
def ice_cream_interval : ℕ := 90

/-- The number of fans who received all three types of free items --/
def fans_with_all_items : ℕ := stadium_capacity / (Nat.lcm hot_dog_interval (Nat.lcm soda_interval ice_cream_interval))

theorem fans_with_all_items_count : fans_with_all_items = 13 := by
  sorry

end NUMINAMATH_CALUDE_fans_with_all_items_count_l3636_363606


namespace NUMINAMATH_CALUDE_box_makers_solution_l3636_363600

/-- Represents the possible makers of the boxes -/
inductive Maker
| Bellini
| BelliniSon
| Cellini

/-- Represents the two boxes -/
inductive Box
| Gold
| Silver

/-- The inscription on the gold box -/
def gold_inscription (gold_maker silver_maker : Maker) : Prop :=
  (gold_maker = Maker.Bellini ∨ gold_maker = Maker.BelliniSon) → silver_maker = Maker.Cellini

/-- The inscription on the silver box -/
def silver_inscription (gold_maker : Maker) : Prop :=
  gold_maker = Maker.BelliniSon

/-- The theorem stating the solution to the problem -/
theorem box_makers_solution :
  ∃ (gold_maker silver_maker : Maker),
    gold_inscription gold_maker silver_maker ∧
    silver_inscription gold_maker ∧
    gold_maker = Maker.Bellini ∧
    silver_maker = Maker.Cellini :=
sorry

end NUMINAMATH_CALUDE_box_makers_solution_l3636_363600


namespace NUMINAMATH_CALUDE_car_distance_problem_l3636_363656

/-- Calculates the distance between two cars given their speeds and overtake time -/
def distance_between_cars (red_speed black_speed overtake_time : ℝ) : ℝ :=
  (black_speed - red_speed) * overtake_time

/-- Theorem stating that the distance between the cars is 30 miles -/
theorem car_distance_problem :
  let red_speed : ℝ := 40
  let black_speed : ℝ := 50
  let overtake_time : ℝ := 3
  distance_between_cars red_speed black_speed overtake_time = 30 := by
sorry


end NUMINAMATH_CALUDE_car_distance_problem_l3636_363656


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3636_363673

-- Define sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3636_363673


namespace NUMINAMATH_CALUDE_max_average_profit_l3636_363693

def profit (t : ℕ+) : ℚ := -2 * (t : ℚ)^2 + 30 * (t : ℚ) - 98

def average_profit (t : ℕ+) : ℚ := (profit t) / (t : ℚ)

theorem max_average_profit :
  ∃ (t : ℕ+), ∀ (k : ℕ+), average_profit t ≥ average_profit k ∧ t = 7 :=
sorry

end NUMINAMATH_CALUDE_max_average_profit_l3636_363693


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3636_363660

/-- Represents a tetrahedron with vertices A, B, C, and D -/
structure Tetrahedron where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  BD : ℝ
  AD : ℝ
  CD : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating that the volume of the specific tetrahedron is 14√13.75 / 9 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    AB := 6,
    AC := 4,
    BC := 5,
    BD := 5,
    AD := 4,
    CD := 3
  }
  tetrahedronVolume t = 14 * Real.sqrt 13.75 / 9 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3636_363660


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3636_363634

theorem binomial_coefficient_equality (n : ℕ) : 
  (∃ k : ℕ, k ∈ Finset.range (n - 1) ∧ 
    2 * Nat.choose n k = Nat.choose n (k - 1) + Nat.choose n (k + 1)) ↔ 
  (∃ m : ℕ, m ≥ 3 ∧ n = m^2 - 2) :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3636_363634


namespace NUMINAMATH_CALUDE_mod_17_graph_intercepts_sum_l3636_363657

theorem mod_17_graph_intercepts_sum :
  ∀ x_0 y_0 : ℕ,
  x_0 < 17 →
  y_0 < 17 →
  (5 * x_0) % 17 = 2 →
  (3 * y_0) % 17 = 15 →
  x_0 + y_0 = 19 := by
sorry

end NUMINAMATH_CALUDE_mod_17_graph_intercepts_sum_l3636_363657


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_equation_l3636_363614

theorem arithmetic_geometric_mean_equation (α β : ℝ) :
  (α + β) / 2 = 8 →
  Real.sqrt (α * β) = 15 →
  (∀ x : ℝ, x^2 - 16*x + 225 = 0 ↔ x = α ∨ x = β) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_equation_l3636_363614


namespace NUMINAMATH_CALUDE_distinct_z_values_l3636_363623

def is_two_digit (n : ℤ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℤ) : ℤ :=
  10 * (n % 10) + (n / 10)

def z (x : ℤ) : ℤ := |x - reverse_digits x|

theorem distinct_z_values (x : ℤ) (hx : is_two_digit x) :
  ∃ (S : Finset ℤ), (∀ y, is_two_digit y → z y ∈ S) ∧ Finset.card S = 8 := by
  sorry

#check distinct_z_values

end NUMINAMATH_CALUDE_distinct_z_values_l3636_363623


namespace NUMINAMATH_CALUDE_parabola_expression_l3636_363675

theorem parabola_expression (f : ℝ → ℝ) (h1 : f (-3) = 0) (h2 : f 1 = 0) (h3 : f 0 = 2) :
  ∀ x, f x = -2/3 * x^2 - 4/3 * x + 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_expression_l3636_363675


namespace NUMINAMATH_CALUDE_flour_bag_weight_l3636_363647

/-- Calculates the weight of each bag of flour given the problem conditions --/
theorem flour_bag_weight 
  (flour_needed : ℕ) 
  (bag_cost : ℚ) 
  (salt_needed : ℕ) 
  (salt_cost_per_pound : ℚ) 
  (promotion_cost : ℕ) 
  (ticket_price : ℕ) 
  (tickets_sold : ℕ) 
  (total_profit : ℚ) 
  (h1 : flour_needed = 500) 
  (h2 : bag_cost = 20) 
  (h3 : salt_needed = 10) 
  (h4 : salt_cost_per_pound = 0.2) 
  (h5 : promotion_cost = 1000) 
  (h6 : ticket_price = 20) 
  (h7 : tickets_sold = 500) 
  (h8 : total_profit = 8798) : 
  ℕ := by
  sorry

#check flour_bag_weight

end NUMINAMATH_CALUDE_flour_bag_weight_l3636_363647


namespace NUMINAMATH_CALUDE_expression_simplification_l3636_363632

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) :
  (x^2 - y^2) / (x * y) - (x * y - 2 * y^2) / (x * y - x^2) = (x^2 - 2 * y^2) / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3636_363632


namespace NUMINAMATH_CALUDE_fishbowl_water_volume_l3636_363695

/-- Calculates the volume of water in a cuboid-shaped container. -/
def water_volume (length width water_height : ℝ) : ℝ :=
  length * width * water_height

/-- Proves that the volume of water in the given cuboid-shaped container is 600 cm³. -/
theorem fishbowl_water_volume :
  water_volume 12 10 5 = 600 := by
  sorry

end NUMINAMATH_CALUDE_fishbowl_water_volume_l3636_363695


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3636_363642

/-- Given a line segment with midpoint (3, -1) and one endpoint at (7, 2),
    prove that the other endpoint is at (-1, -4). -/
theorem line_segment_endpoint (midpoint endpoint1 endpoint2 : ℝ × ℝ) :
  midpoint = (3, -1) →
  endpoint1 = (7, 2) →
  midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) →
  endpoint2 = (-1, -4) := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3636_363642


namespace NUMINAMATH_CALUDE_equation_solution_l3636_363605

theorem equation_solution : ∃! x : ℝ, (1 / (x + 12) + 1 / (x + 10) = 1 / (x + 13) + 1 / (x + 9)) ∧ x = -11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3636_363605


namespace NUMINAMATH_CALUDE_paint_per_statue_l3636_363661

-- Define the total amount of paint
def total_paint : ℚ := 1/2

-- Define the number of statues that can be painted
def num_statues : ℕ := 2

-- Theorem: Each statue requires 1/4 gallon of paint
theorem paint_per_statue : total_paint / num_statues = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_paint_per_statue_l3636_363661


namespace NUMINAMATH_CALUDE_triangles_in_200_sided_polygon_l3636_363646

/-- The number of sides in the regular polygon -/
def n : ℕ := 200

/-- The number of vertices to select for each triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed from a regular n-sided polygon -/
def num_triangles (n : ℕ) : ℕ := Nat.choose n k

theorem triangles_in_200_sided_polygon :
  num_triangles n = 1313400 := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_200_sided_polygon_l3636_363646


namespace NUMINAMATH_CALUDE_calcium_oxide_molecular_weight_l3636_363608

/-- Represents an element with its atomic weight -/
structure Element where
  name : String
  atomic_weight : ℝ

/-- Represents a compound made of elements -/
structure Compound where
  name : String
  elements : List Element

/-- Calculates the molecular weight of a compound -/
def molecular_weight (c : Compound) : ℝ :=
  c.elements.map (λ e => e.atomic_weight) |>.sum

/-- Calcium element -/
def calcium : Element := ⟨"Calcium", 40⟩

/-- Oxygen element -/
def oxygen : Element := ⟨"Oxygen", 16⟩

/-- Calcium oxide compound -/
def calcium_oxide : Compound := ⟨"Calcium oxide", [calcium, oxygen]⟩

/-- Theorem: The molecular weight of Calcium oxide is 56 -/
theorem calcium_oxide_molecular_weight :
  molecular_weight calcium_oxide = 56 := by sorry

end NUMINAMATH_CALUDE_calcium_oxide_molecular_weight_l3636_363608


namespace NUMINAMATH_CALUDE_optimal_selling_price_l3636_363665

/-- Represents the annual profit function for a clothing distributor -/
def annual_profit (x : ℝ) : ℝ := -x^2 + 1000*x - 200000

/-- Represents the annual sales volume function -/
def sales_volume (x : ℝ) : ℝ := 800 - x

theorem optimal_selling_price :
  ∃ (x : ℝ),
    x = 400 ∧
    annual_profit x = 40000 ∧
    ∀ y, y ≠ x → annual_profit y = 40000 → sales_volume x > sales_volume y :=
sorry

end NUMINAMATH_CALUDE_optimal_selling_price_l3636_363665


namespace NUMINAMATH_CALUDE_central_angle_of_specific_sector_l3636_363699

/-- A circular sector with given circumference and area -/
structure CircularSector where
  circumference : ℝ
  area : ℝ

/-- The possible central angles of a circular sector -/
def central_angles (s : CircularSector) : Set ℝ :=
  {θ : ℝ | ∃ r : ℝ, r > 0 ∧ 2 * r + r * θ = s.circumference ∧ 1/2 * r^2 * θ = s.area}

/-- Theorem: The central angle of a sector with circumference 6 and area 2 is either 1 or 4 -/
theorem central_angle_of_specific_sector :
  let s : CircularSector := ⟨6, 2⟩
  central_angles s = {1, 4} := by sorry

end NUMINAMATH_CALUDE_central_angle_of_specific_sector_l3636_363699


namespace NUMINAMATH_CALUDE_james_baked_1380_muffins_l3636_363648

/-- The number of muffins Arthur baked -/
def arthur_muffins : ℕ := 115

/-- The multiplier for James's muffins compared to Arthur's -/
def james_multiplier : ℕ := 12

/-- The number of muffins James baked -/
def james_muffins : ℕ := arthur_muffins * james_multiplier

/-- Proof that James baked 1380 muffins -/
theorem james_baked_1380_muffins : james_muffins = 1380 := by
  sorry

end NUMINAMATH_CALUDE_james_baked_1380_muffins_l3636_363648


namespace NUMINAMATH_CALUDE_optimal_weight_combination_l3636_363654

/-- Represents a combination of weights -/
structure WeightCombination where
  weight3 : ℕ
  weight5 : ℕ
  weight7 : ℕ

/-- Calculates the total weight of a combination -/
def totalWeight (c : WeightCombination) : ℕ :=
  3 * c.weight3 + 5 * c.weight5 + 7 * c.weight7

/-- Calculates the total number of weights in a combination -/
def totalWeights (c : WeightCombination) : ℕ :=
  c.weight3 + c.weight5 + c.weight7

/-- Checks if a combination is valid (totals 130 grams) -/
def isValid (c : WeightCombination) : Prop :=
  totalWeight c = 130

/-- The optimal combination of weights -/
def optimalCombination : WeightCombination :=
  { weight3 := 2, weight5 := 1, weight7 := 17 }

theorem optimal_weight_combination :
  isValid optimalCombination ∧
  (∀ c : WeightCombination, isValid c → totalWeights optimalCombination ≤ totalWeights c) :=
by sorry

end NUMINAMATH_CALUDE_optimal_weight_combination_l3636_363654


namespace NUMINAMATH_CALUDE_largest_number_l3636_363612

def a : ℚ := 24680 + 1 / 13579
def b : ℚ := 24680 - 1 / 13579
def c : ℚ := 24680 * (1 / 13579)
def d : ℚ := 24680 / (1 / 13579)
def e : ℚ := 24680.13579

theorem largest_number : d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3636_363612


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l3636_363688

theorem rope_cutting_problem (a b c : ℕ) (ha : a = 39) (hb : b = 52) (hc : c = 65) :
  Nat.gcd a (Nat.gcd b c) = 13 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l3636_363688


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3636_363667

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : a 1 + a 2 + a 3 = 21
  product_property : a 1 * a 2 * a 3 = 231

/-- Theorem about the second term and general formula of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (seq.a 2 = 7) ∧
  ((∀ n, seq.a n = -4 * n + 15) ∨ (∀ n, seq.a n = 4 * n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3636_363667


namespace NUMINAMATH_CALUDE_m_intersect_n_equals_n_l3636_363697

open Set

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 - 2*x > 0}
def N : Set ℝ := {x : ℝ | x > 3}

-- Theorem statement
theorem m_intersect_n_equals_n : M ∩ N = N := by
  sorry

end NUMINAMATH_CALUDE_m_intersect_n_equals_n_l3636_363697


namespace NUMINAMATH_CALUDE_school_girls_count_l3636_363683

theorem school_girls_count (boys : ℕ) (girls_boys_diff : ℕ) : 
  boys = 469 → girls_boys_diff = 228 → boys + girls_boys_diff = 697 := by
  sorry

end NUMINAMATH_CALUDE_school_girls_count_l3636_363683


namespace NUMINAMATH_CALUDE_unique_solution_l3636_363645

def A (p : ℝ) : Set ℝ := {x | x^2 - p*x - 2 = 0}
def B (q r : ℝ) : Set ℝ := {x | x^2 + q*x + r = 0}

theorem unique_solution :
  ∃! (p q r : ℝ),
    (A p ∪ B q r = {-2, 1, 7}) ∧
    (A p ∩ B q r = {-2}) ∧
    p = -1 ∧ q = -5 ∧ r = -14 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3636_363645


namespace NUMINAMATH_CALUDE_girls_fraction_is_half_l3636_363615

/-- Represents a school with a given number of students and boy-to-girl ratio -/
structure School where
  total_students : ℕ
  boy_ratio : ℕ
  girl_ratio : ℕ

/-- Calculates the number of girls in a school -/
def girls_count (s : School) : ℕ :=
  s.total_students * s.girl_ratio /(s.boy_ratio + s.girl_ratio)

/-- The fraction of girls at a dance attended by students from two schools -/
def girls_fraction (school_a : School) (school_b : School) : ℚ :=
  (girls_count school_a + girls_count school_b : ℚ) /
  (school_a.total_students + school_b.total_students)

theorem girls_fraction_is_half :
  let school_a : School := ⟨300, 3, 2⟩
  let school_b : School := ⟨240, 3, 5⟩
  girls_fraction school_a school_b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_girls_fraction_is_half_l3636_363615


namespace NUMINAMATH_CALUDE_circle_equation_l3636_363677

/-- Given a circle with center (2, -1) and a chord of length 2√2 intercepted by the line x - y - 1 = 0,
    prove that the equation of the circle is (x-2)² + (y+1)² = 4 -/
theorem circle_equation (x y : ℝ) : 
  let center := (2, -1)
  let line := {(x, y) : ℝ × ℝ | x - y - 1 = 0}
  let chord_length := 2 * Real.sqrt 2
  true → (x - 2)^2 + (y + 1)^2 = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_equation_l3636_363677


namespace NUMINAMATH_CALUDE_jayden_half_ernesto_age_l3636_363616

/-- 
Given:
- Ernesto is currently 11 years old
- Jayden is currently 4 years old

Prove that in 3 years, Jayden will be half of Ernesto's age
-/
theorem jayden_half_ernesto_age :
  let ernesto_age : ℕ := 11
  let jayden_age : ℕ := 4
  let years_until_half : ℕ := 3
  (jayden_age + years_until_half : ℚ) = (1/2 : ℚ) * (ernesto_age + years_until_half : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_jayden_half_ernesto_age_l3636_363616


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_l3636_363686

theorem regular_triangular_pyramid (a : ℝ) : 
  (∃ h : ℝ, h = (a * Real.sqrt 3) / 3) → -- height in terms of base side
  (∃ V : ℝ, V = (1 / 3) * ((a^2 * Real.sqrt 3) / 4) * ((a * Real.sqrt 3) / 3)) → -- volume formula
  V = 18 → -- given volume
  a = 6 := by sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_l3636_363686


namespace NUMINAMATH_CALUDE_binomial_max_at_one_l3636_363603

/-- The number of trials in the binomial distribution -/
def n : ℕ := 6

/-- The probability of success in a single trial -/
def p : ℚ := 1/6

/-- The binomial probability mass function -/
def binomial_pmf (k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

/-- Theorem stating that the binomial probability is maximized when k = 1 -/
theorem binomial_max_at_one :
  ∀ k : ℕ, k ≤ n → binomial_pmf 1 ≥ binomial_pmf k :=
sorry

end NUMINAMATH_CALUDE_binomial_max_at_one_l3636_363603


namespace NUMINAMATH_CALUDE_side_c_length_l3636_363622

-- Define the triangle ABC
def triangle_ABC (A B C a b c : Real) : Prop :=
  -- Angles sum to 180°
  A + B + C = Real.pi ∧
  -- Positive side lengths
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  -- Sine law
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  a / Real.sin A = c / Real.sin C

-- Theorem statement
theorem side_c_length :
  ∀ (A B C a b c : Real),
    triangle_ABC A B C a b c →
    A = Real.pi / 6 →  -- 30°
    B = 7 * Real.pi / 12 →  -- 105°
    a = 2 →
    c = 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_side_c_length_l3636_363622


namespace NUMINAMATH_CALUDE_circle_radius_sqrt_61_l3636_363641

/-- Given a circle with center on the x-axis passing through points (2,5) and (3,2),
    prove that its radius is √61. -/
theorem circle_radius_sqrt_61 (x : ℝ) :
  (∀ (y : ℝ), y = 0 →  -- Center is on x-axis
    (x - 2)^2 + (y - 5)^2 = (x - 3)^2 + (y - 2)^2) →  -- Points (2,5) and (3,2) are equidistant from center
  (x - 2)^2 + 5^2 = 61 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_sqrt_61_l3636_363641


namespace NUMINAMATH_CALUDE_no_base_with_final_digit_four_l3636_363655

theorem no_base_with_final_digit_four : 
  ∀ b : ℕ, 3 ≤ b ∧ b ≤ 10 → ¬(981 % b = 4) :=
by sorry

end NUMINAMATH_CALUDE_no_base_with_final_digit_four_l3636_363655


namespace NUMINAMATH_CALUDE_rice_on_eighth_day_l3636_363678

/-- Represents the number of laborers on a given day -/
def laborers (day : ℕ) : ℕ := 64 + 7 * (day - 1)

/-- The amount of rice given to each laborer per day -/
def ricePerLaborer : ℕ := 3

/-- The amount of rice given out on a specific day -/
def riceOnDay (day : ℕ) : ℕ := laborers day * ricePerLaborer

theorem rice_on_eighth_day : riceOnDay 8 = 339 := by
  sorry

end NUMINAMATH_CALUDE_rice_on_eighth_day_l3636_363678


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3636_363666

theorem fraction_evaluation : 
  (11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2) / 
  (0 - 1 + 2 - 3 + 4 - 5 + 6 - 7 + 8) = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3636_363666


namespace NUMINAMATH_CALUDE_sphere_volume_diameter_6_l3636_363619

/-- The volume of a sphere with diameter 6 is 36π. -/
theorem sphere_volume_diameter_6 : 
  let d : ℝ := 6
  let r : ℝ := d / 2
  let V : ℝ := (4 / 3) * Real.pi * r ^ 3
  V = 36 * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_volume_diameter_6_l3636_363619


namespace NUMINAMATH_CALUDE_polyhedron_exists_l3636_363674

-- Define a custom type for vertices
inductive Vertex : Type
  | A | B | C | D | E | F | G | H

-- Define an edge as a pair of vertices
def Edge : Type := Vertex × Vertex

-- Define the list of edges
def edgeList : List Edge :=
  [(Vertex.A, Vertex.B), (Vertex.A, Vertex.C), (Vertex.B, Vertex.C),
   (Vertex.B, Vertex.D), (Vertex.C, Vertex.D), (Vertex.D, Vertex.E),
   (Vertex.E, Vertex.F), (Vertex.E, Vertex.G), (Vertex.F, Vertex.G),
   (Vertex.F, Vertex.H), (Vertex.G, Vertex.H), (Vertex.A, Vertex.H)]

-- Define a polyhedron as a list of edges
def Polyhedron : Type := List Edge

-- Theorem: There exists a polyhedron with the given list of edges
theorem polyhedron_exists : ∃ (p : Polyhedron), p = edgeList := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_exists_l3636_363674


namespace NUMINAMATH_CALUDE_range_of_a_l3636_363638

theorem range_of_a (a : ℝ) : (∀ x ∈ Set.Icc (-1) 1, a * x + 1 > 0) → a ∈ Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3636_363638


namespace NUMINAMATH_CALUDE_no_real_roots_for_distinct_abc_l3636_363609

theorem no_real_roots_for_distinct_abc (a b c : ℝ) 
  (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) : 
  let discriminant := 4 * (a + b + c)^2 - 12 * (a^2 + b^2 + c^2)
  discriminant < 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_for_distinct_abc_l3636_363609


namespace NUMINAMATH_CALUDE_thor_jump_count_l3636_363662

def jump_distance (n : ℕ) : ℝ := 3 * (3 ^ (n - 1))

theorem thor_jump_count :
  (∀ k < 10, jump_distance k ≤ 29000) ∧
  jump_distance 10 > 29000 :=
by sorry

end NUMINAMATH_CALUDE_thor_jump_count_l3636_363662


namespace NUMINAMATH_CALUDE_marble_count_l3636_363613

/-- The number of marbles each person has --/
structure Marbles where
  ed : ℕ
  doug : ℕ
  charlie : ℕ

/-- The initial state of marbles before Ed lost some --/
def initial_marbles : Marbles → Marbles
| ⟨ed, doug, charlie⟩ => ⟨ed + 20, doug, charlie⟩

theorem marble_count (m : Marbles) :
  (initial_marbles m).ed = (initial_marbles m).doug + 12 →
  m.ed = 17 →
  m.charlie = 4 * m.doug →
  m.doug = 25 ∧ m.charlie = 100 := by
  sorry

end NUMINAMATH_CALUDE_marble_count_l3636_363613


namespace NUMINAMATH_CALUDE_inverse_mod_53_l3636_363698

theorem inverse_mod_53 (h : (17⁻¹ : ZMod 53) = 23) : (36⁻¹ : ZMod 53) = 30 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_53_l3636_363698


namespace NUMINAMATH_CALUDE_compute_expression_l3636_363687

theorem compute_expression : 75 * 1313 - 25 * 1313 = 65650 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3636_363687


namespace NUMINAMATH_CALUDE_smallest_consecutive_even_sum_l3636_363672

theorem smallest_consecutive_even_sum (a : ℤ) : 
  (∃ (b c d e : ℤ), 
    (a + 2 = b) ∧ (b + 2 = c) ∧ (c + 2 = d) ∧ (d + 2 = e) ∧  -- Consecutive even integers
    (a % 2 = 0) ∧                                            -- First number is even
    (a + b + c + d + e = 380)) →                             -- Sum is 380
  a = 72 := by
sorry

end NUMINAMATH_CALUDE_smallest_consecutive_even_sum_l3636_363672


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3636_363618

theorem absolute_value_inequality (x : ℝ) :
  3 ≤ |x + 2| ∧ |x + 2| ≤ 6 ↔ (1 ≤ x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x ≤ -5) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3636_363618


namespace NUMINAMATH_CALUDE_range_of_a_l3636_363627

theorem range_of_a (x a : ℝ) : 
  (3 * x + 2 * (3 * a + 1) = 6 * x + a) → 
  (x ≥ 0) → 
  (a ≥ -2/5) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3636_363627


namespace NUMINAMATH_CALUDE_cows_equivalent_to_buffaloes_or_oxen_l3636_363664

-- Define the variables
variable (B : ℕ) -- Daily fodder consumption of a buffalo
variable (C : ℕ) -- Daily fodder consumption of a cow
variable (O : ℕ) -- Daily fodder consumption of an ox
variable (F : ℕ) -- Total available fodder

-- Define the conditions
axiom buffalo_ox_equiv : 3 * B = 2 * O
axiom initial_fodder : F = (15 * B + 8 * O + 24 * C) * 48
axiom additional_cattle : F = (30 * B + 64 * C) * 24

-- The theorem to prove
theorem cows_equivalent_to_buffaloes_or_oxen : ∃ x : ℕ, x = 2 ∧ 3 * B = x * C := by
  sorry

end NUMINAMATH_CALUDE_cows_equivalent_to_buffaloes_or_oxen_l3636_363664


namespace NUMINAMATH_CALUDE_amount_distribution_l3636_363669

theorem amount_distribution (A : ℕ) : 
  (A / 14 = A / 18 + 80) → A = 5040 :=
by
  sorry

end NUMINAMATH_CALUDE_amount_distribution_l3636_363669


namespace NUMINAMATH_CALUDE_grandfathers_age_l3636_363607

theorem grandfathers_age (x : ℕ) (y z : ℕ) : 
  (6 * x = 6 * x) →  -- Current year
  (6 * x + y = 5 * (x + y)) →  -- In y years
  (6 * x + y + z = 4 * (x + y + z)) →  -- In y + z years
  (x > 0) →  -- Ming's age is positive
  (y > 0) →  -- First time gap is positive
  (z > 0) →  -- Second time gap is positive
  6 * x = 72 := by
  sorry

end NUMINAMATH_CALUDE_grandfathers_age_l3636_363607


namespace NUMINAMATH_CALUDE_sum_of_squares_inequality_l3636_363625

theorem sum_of_squares_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_inequality_l3636_363625


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3636_363692

def is_valid_number (x : ℕ) : Prop :=
  x > 0 ∧ 
  ∃ (multiples : Finset ℕ), 
    multiples.card = 10 ∧ 
    ∀ m ∈ multiples, 
      m < 100 ∧ 
      m % 2 = 1 ∧ 
      ∃ k : ℕ, k % 2 = 1 ∧ m = k * x

theorem smallest_valid_number : 
  ∀ y : ℕ, y < 3 → ¬(is_valid_number y) ∧ is_valid_number 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3636_363692


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3636_363694

theorem complex_fraction_equality : (4 - 2*I) / (1 + I)^2 = -1 - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3636_363694


namespace NUMINAMATH_CALUDE_probability_three_ones_or_twos_in_five_rolls_l3636_363644

-- Define the probability of rolling a 1 or 2 on a fair six-sided die
def prob_one_or_two : ℚ := 1 / 3

-- Define the probability of not rolling a 1 or 2 on a fair six-sided die
def prob_not_one_or_two : ℚ := 2 / 3

-- Define the number of rolls
def num_rolls : ℕ := 5

-- Define the number of times we want to roll a 1 or 2
def target_rolls : ℕ := 3

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem probability_three_ones_or_twos_in_five_rolls :
  (binomial num_rolls target_rolls : ℚ) * prob_one_or_two ^ target_rolls * prob_not_one_or_two ^ (num_rolls - target_rolls) = 40 / 243 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_ones_or_twos_in_five_rolls_l3636_363644


namespace NUMINAMATH_CALUDE_apple_pie_pieces_l3636_363696

/-- Calculates the number of pieces each pie is cut into -/
def piecesPer (totalApples : ℕ) (numPies : ℕ) (applesPerSlice : ℕ) : ℕ :=
  (totalApples / numPies) / applesPerSlice

/-- Proves that each pie is cut into 6 pieces given the problem conditions -/
theorem apple_pie_pieces : 
  piecesPer (4 * 12) 4 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_pieces_l3636_363696


namespace NUMINAMATH_CALUDE_power_inequality_l3636_363624

theorem power_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (0.2 : ℝ) ^ x < (1/2 : ℝ) ^ x ∧ (1/2 : ℝ) ^ x < 2 ^ x := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3636_363624


namespace NUMINAMATH_CALUDE_smallest_m_plus_n_l3636_363635

/-- Given that m and n are natural numbers satisfying 3n^3 = 5m^2, 
    the smallest possible value of m + n is 60. -/
theorem smallest_m_plus_n : ∃ (m n : ℕ), 
  (3 * n^3 = 5 * m^2) ∧ 
  (m + n = 60) ∧ 
  (∀ (m' n' : ℕ), (3 * n'^3 = 5 * m'^2) → (m' + n' ≥ 60)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_plus_n_l3636_363635


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_digit_difference_l3636_363639

/-- Given two different digits C and D where C > D, prove that the smallest prime factor
    of the difference between the two-digit number CD and its reverse DC is 3. -/
theorem smallest_prime_factor_of_digit_difference (C D : ℕ) : 
  C ≠ D → C > D → C < 10 → D < 10 → Nat.minFac (10 * C + D - (10 * D + C)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_digit_difference_l3636_363639


namespace NUMINAMATH_CALUDE_remainder_of_sum_of_primes_l3636_363621

def first_eight_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

theorem remainder_of_sum_of_primes :
  (3 * (List.sum (List.take 7 first_eight_primes))) % (List.get! first_eight_primes 7) = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_of_primes_l3636_363621


namespace NUMINAMATH_CALUDE_decimal_multiplication_addition_l3636_363682

theorem decimal_multiplication_addition : 0.45 * 0.65 + 0.1 * 0.2 = 0.3125 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_addition_l3636_363682
