import Mathlib

namespace NUMINAMATH_CALUDE_circle_properties_l2206_220688

theorem circle_properties :
  let center : ℝ × ℝ := (1, -1)
  let radius : ℝ := Real.sqrt 2
  let origin : ℝ × ℝ := (0, 0)
  let tangent_point : ℝ × ℝ := (2, 0)
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let on_circle (p : ℝ × ℝ) := distance p center = radius
  let tangent_line (x y : ℝ) := x + y - 2 = 0
  
  (on_circle origin) ∧ 
  (on_circle tangent_point) ∧
  (tangent_line tangent_point.1 tangent_point.2) ∧
  (∀ (p : ℝ × ℝ), tangent_line p.1 p.2 → distance p center ≥ radius) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l2206_220688


namespace NUMINAMATH_CALUDE_division_into_proportional_parts_l2206_220616

theorem division_into_proportional_parts (total : ℚ) (a b c : ℚ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 104 →
  a = 1 →
  b = 1/2 →
  c = 1/3 →
  let x := total * b / (a + b + c)
  x = 28 + 4/11 := by
  sorry

end NUMINAMATH_CALUDE_division_into_proportional_parts_l2206_220616


namespace NUMINAMATH_CALUDE_summer_sun_salutations_l2206_220686

/-- The number of sun salutations Summer performs in a year -/
def sun_salutations_per_year (poses_per_day : ℕ) (weekdays_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  poses_per_day * weekdays_per_week * weeks_per_year

/-- Theorem: Summer performs 1300 sun salutations in a year -/
theorem summer_sun_salutations :
  sun_salutations_per_year 5 5 52 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_summer_sun_salutations_l2206_220686


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l2206_220647

def Rectangle (O A B : ℝ × ℝ) : Prop :=
  ∃ C : ℝ × ℝ, (O.1 - A.1) * (A.1 - C.1) + (O.2 - A.2) * (A.2 - C.2) = 0 ∧
              (O.1 - B.1) * (B.1 - C.1) + (O.2 - B.2) * (B.2 - C.2) = 0

theorem rectangle_diagonal (O A B : ℝ × ℝ) (h : Rectangle O A B) :
  let OA : ℝ × ℝ := (-3, 1)
  let OB : ℝ × ℝ := (-2, k)
  k = 4 :=
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l2206_220647


namespace NUMINAMATH_CALUDE_condition_equivalence_l2206_220634

theorem condition_equivalence (a : ℝ) (h : a > 0) : (a > 1) ↔ (a > Real.sqrt a) := by
  sorry

end NUMINAMATH_CALUDE_condition_equivalence_l2206_220634


namespace NUMINAMATH_CALUDE_count_with_3_or_6_in_base_7_eq_1776_l2206_220680

/-- The count of integers among the first 2401 positive integers in base 7 that use 3 or 6 as a digit -/
def count_with_3_or_6_in_base_7 : ℕ :=
  2401 - 5^4

theorem count_with_3_or_6_in_base_7_eq_1776 :
  count_with_3_or_6_in_base_7 = 1776 := by sorry

end NUMINAMATH_CALUDE_count_with_3_or_6_in_base_7_eq_1776_l2206_220680


namespace NUMINAMATH_CALUDE_savings_equality_l2206_220696

def total_salary : ℝ := 4000
def a_salary : ℝ := 3000
def a_spend_rate : ℝ := 0.95
def b_spend_rate : ℝ := 0.85

def b_salary : ℝ := total_salary - a_salary

def a_savings : ℝ := a_salary * (1 - a_spend_rate)
def b_savings : ℝ := b_salary * (1 - b_spend_rate)

theorem savings_equality : a_savings = b_savings := by
  sorry

end NUMINAMATH_CALUDE_savings_equality_l2206_220696


namespace NUMINAMATH_CALUDE_expression_evaluation_l2206_220685

theorem expression_evaluation (a b c d e : ℚ) 
  (h1 : a * b = 1) 
  (h2 : c + d = 0) 
  (h3 : |e| = 2) : 
  (c + d) / 5 - (1 / 2) * a * b + e = 3 / 2 ∨ 
  (c + d) / 5 - (1 / 2) * a * b + e = -(5 / 2) :=
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2206_220685


namespace NUMINAMATH_CALUDE_scott_running_distance_l2206_220668

/-- Scott's running schedule and total distance for a month --/
theorem scott_running_distance :
  let miles_mon_to_wed : ℕ := 3 * 3
  let miles_thu_fri : ℕ := 2 * (2 * 3)
  let miles_per_week : ℕ := miles_mon_to_wed + miles_thu_fri
  let weeks_in_month : ℕ := 4
  miles_per_week * weeks_in_month = 84 := by
  sorry

end NUMINAMATH_CALUDE_scott_running_distance_l2206_220668


namespace NUMINAMATH_CALUDE_expression_simplification_l2206_220639

theorem expression_simplification (x y : ℝ) (hx : x = -3) (hy : y = -1) :
  (-3 * x^2 - 4*y) - (2 * x^2 - 5*y + 6) + (x^2 - 5*y - 1) = -39 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2206_220639


namespace NUMINAMATH_CALUDE_remaining_ribbon_length_l2206_220612

/-- Calculates the remaining ribbon length after wrapping gifts -/
theorem remaining_ribbon_length
  (num_gifts : ℕ)
  (ribbon_per_gift : ℝ)
  (initial_ribbon_length : ℝ)
  (h1 : num_gifts = 8)
  (h2 : ribbon_per_gift = 1.5)
  (h3 : initial_ribbon_length = 15) :
  initial_ribbon_length - (↑num_gifts * ribbon_per_gift) = 3 :=
by sorry

end NUMINAMATH_CALUDE_remaining_ribbon_length_l2206_220612


namespace NUMINAMATH_CALUDE_stating_same_white_wins_exist_l2206_220674

/-- Represents a chess tournament with participants and their scores. -/
structure ChessTournament where
  /-- The number of participants in the tournament. -/
  participants : Nat
  /-- The number of games won with white pieces by each participant. -/
  white_wins : Fin participants → Nat
  /-- Assumption that all participants have the same total score. -/
  same_total_score : ∀ i j : Fin participants, 
    white_wins i + (participants - 1 - white_wins j) = participants - 1

/-- 
Theorem stating that in a chess tournament where all participants have the same total score,
there must be at least two participants who won the same number of games with white pieces.
-/
theorem same_white_wins_exist (t : ChessTournament) : 
  ∃ i j : Fin t.participants, i ≠ j ∧ t.white_wins i = t.white_wins j := by
  sorry


end NUMINAMATH_CALUDE_stating_same_white_wins_exist_l2206_220674


namespace NUMINAMATH_CALUDE_tiles_for_taylors_room_l2206_220690

/-- Calculates the total number of tiles needed for a rectangular room with a border of smaller tiles --/
def total_tiles (room_length room_width border_tile_size interior_tile_size : ℕ) : ℕ :=
  let border_tiles := 2 * (room_length + room_width) - 4
  let interior_length := room_length - 2 * border_tile_size
  let interior_width := room_width - 2 * border_tile_size
  let interior_area := interior_length * interior_width
  let interior_tiles := interior_area / (interior_tile_size * interior_tile_size)
  border_tiles + interior_tiles

/-- Theorem stating that for a 12x16 room with 1x1 border tiles and 2x2 interior tiles, 87 tiles are needed --/
theorem tiles_for_taylors_room : total_tiles 12 16 1 2 = 87 := by
  sorry

end NUMINAMATH_CALUDE_tiles_for_taylors_room_l2206_220690


namespace NUMINAMATH_CALUDE_rock_paper_scissors_probabilities_l2206_220645

-- Define the game structure
structure RockPaperScissors where
  players : Finset Char := {'A', 'B', 'C'}

-- Define the probability of winning in a single throw
def win_prob : ℚ := 1 / 3

-- Define the probability of a tie in a single throw
def tie_prob : ℚ := 1 / 3

-- Define the probability that A wins against B with no more than two throws
def prob_A_wins_B (game : RockPaperScissors) : ℚ := sorry

-- Define the probability that C will treat after two throws
def prob_C_treats (game : RockPaperScissors) : ℚ := sorry

-- Define the probability that exactly two days out of three C will treat after two throws
def prob_C_treats_two_days (game : RockPaperScissors) : ℚ := sorry

theorem rock_paper_scissors_probabilities (game : RockPaperScissors) :
  prob_A_wins_B game = 4 / 9 ∧
  prob_C_treats game = 2 / 9 ∧
  prob_C_treats_two_days game = 28 / 243 := by sorry

end NUMINAMATH_CALUDE_rock_paper_scissors_probabilities_l2206_220645


namespace NUMINAMATH_CALUDE_jake_peaches_l2206_220602

/-- The number of peaches Jill has -/
def jill_peaches : ℕ := 5

/-- The number of peaches Steven has more than Jill -/
def steven_more_than_jill : ℕ := 18

/-- The number of peaches Jake has fewer than Steven -/
def jake_fewer_than_steven : ℕ := 6

/-- Theorem: Jake has 17 peaches -/
theorem jake_peaches : 
  jill_peaches + steven_more_than_jill - jake_fewer_than_steven = 17 := by
  sorry

end NUMINAMATH_CALUDE_jake_peaches_l2206_220602


namespace NUMINAMATH_CALUDE_projectile_height_l2206_220651

theorem projectile_height (t : ℝ) : 
  t > 0 ∧ -16 * t^2 + 60 * t = 56 ∧ 
  ∀ s, s > 0 ∧ -16 * s^2 + 60 * s = 56 → t ≤ s → 
  t = 1.75 := by
sorry

end NUMINAMATH_CALUDE_projectile_height_l2206_220651


namespace NUMINAMATH_CALUDE_evaluate_f_l2206_220650

/-- The function f(x) = x^3 + 3∛x -/
def f (x : ℝ) : ℝ := x^3 + 3 * (x^(1/3))

/-- Theorem stating that 3f(3) + f(27) = 19818 -/
theorem evaluate_f : 3 * f 3 + f 27 = 19818 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_f_l2206_220650


namespace NUMINAMATH_CALUDE_difference_largest_third_smallest_l2206_220640

def digits : List Nat := [1, 6, 8]

def largest_number : Nat := 861

def third_smallest_number : Nat := 618

theorem difference_largest_third_smallest :
  largest_number - third_smallest_number = 243 := by
  sorry

end NUMINAMATH_CALUDE_difference_largest_third_smallest_l2206_220640


namespace NUMINAMATH_CALUDE_logarithm_difference_l2206_220630

theorem logarithm_difference (a b c d : ℕ+) 
  (h1 : (Real.log b) / (Real.log a) = 3/2)
  (h2 : (Real.log d) / (Real.log c) = 5/4)
  (h3 : a - c = 9) : 
  b - d = 93 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_difference_l2206_220630


namespace NUMINAMATH_CALUDE_fraction_sum_equals_62_l2206_220617

theorem fraction_sum_equals_62 (a b : ℝ) : 
  a = (Real.sqrt 5 + Real.sqrt 3) / (Real.sqrt 5 - Real.sqrt 3) →
  b = (Real.sqrt 5 - Real.sqrt 3) / (Real.sqrt 5 + Real.sqrt 3) →
  b / a + a / b = 62 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_62_l2206_220617


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l2206_220666

theorem largest_n_for_factorization : ∃ (n : ℤ),
  (∀ m : ℤ, (∃ (a b : ℤ), 3 * X^2 + m * X + 54 = (3 * X + a) * (X + b)) → m ≤ n) ∧
  (∃ (a b : ℤ), 3 * X^2 + n * X + 54 = (3 * X + a) * (X + b)) ∧
  n = 163 :=
by sorry


end NUMINAMATH_CALUDE_largest_n_for_factorization_l2206_220666


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2206_220636

theorem smallest_n_congruence : 
  ∃ (n : ℕ), n > 0 ∧ (5 * n) % 26 = 789 % 26 ∧ 
  ∀ (m : ℕ), m > 0 ∧ (5 * m) % 26 = 789 % 26 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2206_220636


namespace NUMINAMATH_CALUDE_largest_c_for_negative_two_in_range_l2206_220669

/-- The function f(x) defined as x^2 + 4x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

/-- The theorem stating that the largest value of c for which -2 is in the range of f is 2 -/
theorem largest_c_for_negative_two_in_range :
  ∀ c : ℝ, (∃ x : ℝ, f c x = -2) ↔ c ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_two_in_range_l2206_220669


namespace NUMINAMATH_CALUDE_race_track_outer_radius_l2206_220665

/-- Given a circular race track with an inner circumference of 880 m and a width of 25 m,
    the radius of the outer circle is 165 m. -/
theorem race_track_outer_radius :
  ∀ (inner_radius outer_radius : ℝ),
    inner_radius * 2 * Real.pi = 880 →
    outer_radius = inner_radius + 25 →
    outer_radius = 165 := by
  sorry

end NUMINAMATH_CALUDE_race_track_outer_radius_l2206_220665


namespace NUMINAMATH_CALUDE_solve_equations_l2206_220619

theorem solve_equations (t u s : ℝ) : 
  t = 15 * s^2 → 
  u = 5 * s + 3 → 
  t = 3.75 → 
  s = 0.5 ∧ u = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equations_l2206_220619


namespace NUMINAMATH_CALUDE_remainder_of_binary_div_4_l2206_220683

def binary_number : List Bool := [true, true, false, true, false, true, false, false, true, false, true, true]

def last_two_digits (n : List Bool) : (Bool × Bool) :=
  match n.reverse with
  | b0 :: b1 :: _ => (b1, b0)
  | _ => (false, false)  -- Default case, should not occur for valid input

def remainder_mod_4 (digits : Bool × Bool) : Nat :=
  let (b1, b0) := digits
  2 * (if b1 then 1 else 0) + (if b0 then 1 else 0)

theorem remainder_of_binary_div_4 :
  remainder_mod_4 (last_two_digits binary_number) = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_binary_div_4_l2206_220683


namespace NUMINAMATH_CALUDE_right_triangle_rotation_creates_cone_l2206_220621

/-- A right triangle is a triangle with one right angle -/
structure RightTriangle where
  -- We don't need to define the specifics of a right triangle for this statement
  mk :: 

/-- A cone is a three-dimensional geometric shape with a circular base that tapers to a point -/
structure Cone where
  -- We don't need to define the specifics of a cone for this statement
  mk ::

/-- Rotation of a right triangle around one of its legs -/
def rotateAroundLeg (t : RightTriangle) : Cone :=
  sorry

theorem right_triangle_rotation_creates_cone (t : RightTriangle) :
  ∃ (c : Cone), rotateAroundLeg t = c :=
sorry

end NUMINAMATH_CALUDE_right_triangle_rotation_creates_cone_l2206_220621


namespace NUMINAMATH_CALUDE_house_transaction_loss_l2206_220679

theorem house_transaction_loss (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : 
  initial_value = 12000 →
  loss_percent = 0.15 →
  gain_percent = 0.20 →
  let first_sale := initial_value * (1 - loss_percent)
  let second_sale := first_sale * (1 + gain_percent)
  second_sale - initial_value = 2040 :=
by sorry

end NUMINAMATH_CALUDE_house_transaction_loss_l2206_220679


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l2206_220622

theorem bowling_ball_weight : 
  ∀ (b c : ℝ),
  5 * b = 2 * c →
  3 * c = 72 →
  b = 9.6 := by
sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l2206_220622


namespace NUMINAMATH_CALUDE_tom_age_l2206_220631

theorem tom_age (adam_age : ℕ) (future_years : ℕ) (future_combined_age : ℕ) :
  adam_age = 8 →
  future_years = 12 →
  future_combined_age = 44 →
  ∃ tom_age : ℕ, tom_age + adam_age + 2 * future_years = future_combined_age ∧ tom_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_tom_age_l2206_220631


namespace NUMINAMATH_CALUDE_number_satisfies_equation_l2206_220657

theorem number_satisfies_equation : ∃ x : ℝ, (0.8 * 90 : ℝ) = 0.7 * x + 30 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfies_equation_l2206_220657


namespace NUMINAMATH_CALUDE_dans_helmet_craters_l2206_220615

theorem dans_helmet_craters (dans_craters daniel_craters rins_craters : ℕ) : 
  dans_craters = daniel_craters + 10 →
  rins_craters = dans_craters + daniel_craters + 15 →
  rins_craters = 75 →
  dans_craters = 35 := by
  sorry

end NUMINAMATH_CALUDE_dans_helmet_craters_l2206_220615


namespace NUMINAMATH_CALUDE_complex_roots_unity_l2206_220644

theorem complex_roots_unity (z₁ z₂ z₃ : ℂ) 
  (h1 : Complex.abs z₁ = 1) 
  (h2 : Complex.abs z₂ = 1) 
  (h3 : Complex.abs z₃ = 1)
  (h4 : z₁ + z₂ + z₃ = 1) 
  (h5 : z₁ * z₂ * z₃ = 1) :
  ({z₁, z₂, z₃} : Finset ℂ) = {1, Complex.I, -Complex.I} := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_unity_l2206_220644


namespace NUMINAMATH_CALUDE_estimate_products_and_quotients_l2206_220699

theorem estimate_products_and_quotients 
  (ε₁ ε₂ ε₃ ε₄ : ℝ) 
  (h₁ : ε₁ > 0) 
  (h₂ : ε₂ > 0) 
  (h₃ : ε₃ > 0) 
  (h₄ : ε₄ > 0) : 
  (|99 * 71 - 7000| ≤ ε₁) ∧ 
  (|25 * 39 - 1000| ≤ ε₂) ∧ 
  (|124 / 3 - 40| ≤ ε₃) ∧ 
  (|398 / 5 - 80| ≤ ε₄) := by
  sorry

end NUMINAMATH_CALUDE_estimate_products_and_quotients_l2206_220699


namespace NUMINAMATH_CALUDE_cat_collar_nylon_l2206_220618

/-- The number of inches of nylon needed for one dog collar -/
def dog_collar_nylon : ℝ := 18

/-- The total number of inches of nylon needed for all collars -/
def total_nylon : ℝ := 192

/-- The number of dog collars -/
def num_dog_collars : ℕ := 9

/-- The number of cat collars -/
def num_cat_collars : ℕ := 3

/-- Theorem stating that the number of inches of nylon needed for one cat collar is 10 -/
theorem cat_collar_nylon : 
  (total_nylon - dog_collar_nylon * num_dog_collars) / num_cat_collars = 10 := by
sorry

end NUMINAMATH_CALUDE_cat_collar_nylon_l2206_220618


namespace NUMINAMATH_CALUDE_diameter_endpoint_l2206_220625

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a diameter of a circle --/
structure Diameter where
  circle : Circle
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- Given a circle with center (3, 4) and one endpoint of a diameter at (1, -2),
    the other endpoint of the diameter is at (5, 10) --/
theorem diameter_endpoint (P : Circle) (d : Diameter) :
  P.center = (3, 4) →
  d.circle = P →
  d.endpoint1 = (1, -2) →
  d.endpoint2 = (5, 10) := by
sorry

end NUMINAMATH_CALUDE_diameter_endpoint_l2206_220625


namespace NUMINAMATH_CALUDE_max_value_inequality_l2206_220610

theorem max_value_inequality (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 2*a*b*Real.sqrt 2 + 2*b*c ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2206_220610


namespace NUMINAMATH_CALUDE_simplify_fraction_l2206_220659

theorem simplify_fraction : (2^6 + 2^4) / (2^5 - 2^2) = 20 / 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2206_220659


namespace NUMINAMATH_CALUDE_houses_per_block_l2206_220678

theorem houses_per_block (mail_per_block : ℕ) (mail_per_house : ℕ) 
  (h1 : mail_per_block = 32) (h2 : mail_per_house = 8) :
  mail_per_block / mail_per_house = 4 := by
  sorry

end NUMINAMATH_CALUDE_houses_per_block_l2206_220678


namespace NUMINAMATH_CALUDE_area_of_triangle_l2206_220676

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

-- Define the foci of the hyperbola
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  let c := Real.sqrt 7
  F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2

-- Define the angle between PF₁ and PF₂
def angle_F₁PF₂ (P F₁ F₂ : ℝ × ℝ) : Prop :=
  let v₁ := (F₁.1 - P.1, F₁.2 - P.2)
  let v₂ := (F₂.1 - P.1, F₂.2 - P.2)
  let cos_angle := (v₁.1 * v₂.1 + v₁.2 * v₂.2) / 
    (Real.sqrt (v₁.1^2 + v₁.2^2) * Real.sqrt (v₂.1^2 + v₂.2^2))
  cos_angle = 1/2  -- cos 60° = 1/2

-- Theorem statement
theorem area_of_triangle (P F₁ F₂ : ℝ × ℝ) :
  point_on_hyperbola P →
  foci F₁ F₂ →
  angle_F₁PF₂ P F₁ F₂ →
  let a := Real.sqrt ((F₁.1 - P.1)^2 + (F₁.2 - P.2)^2)
  let b := Real.sqrt ((F₂.1 - P.1)^2 + (F₂.2 - P.2)^2)
  let s := (a + b + 2 * Real.sqrt 7) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - 2 * Real.sqrt 7)) = 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_l2206_220676


namespace NUMINAMATH_CALUDE_not_closed_set_1_not_closed_set_2_closed_set_3_exist_closed_sets_union_not_closed_l2206_220626

-- Definition of a closed set
def is_closed_set (M : Set Int) : Prop :=
  ∀ a b : Int, a ∈ M → b ∈ M → (a + b) ∈ M ∧ (a - b) ∈ M

-- Statement 1
theorem not_closed_set_1 : ¬ is_closed_set {-4, -2, 0, 2, 4} := by sorry

-- Statement 2
def positive_integers : Set Int := {n | n > 0}

theorem not_closed_set_2 : ¬ is_closed_set positive_integers := by sorry

-- Statement 3
def multiples_of_three : Set Int := {n | ∃ k : Int, n = 3 * k}

theorem closed_set_3 : is_closed_set multiples_of_three := by sorry

-- Statement 4
theorem exist_closed_sets_union_not_closed :
  ∃ A₁ A₂ : Set Int, is_closed_set A₁ ∧ is_closed_set A₂ ∧ ¬ is_closed_set (A₁ ∪ A₂) := by sorry

end NUMINAMATH_CALUDE_not_closed_set_1_not_closed_set_2_closed_set_3_exist_closed_sets_union_not_closed_l2206_220626


namespace NUMINAMATH_CALUDE_committee_size_is_24_l2206_220662

/-- The number of sandwiches per person -/
def sandwiches_per_person : ℕ := 2

/-- The number of croissants per pack -/
def croissants_per_pack : ℕ := 12

/-- The cost of one pack of croissants in cents -/
def cost_per_pack : ℕ := 800

/-- The total amount spent on croissants in cents -/
def total_spent : ℕ := 3200

/-- The number of people on the committee -/
def committee_size : ℕ := total_spent / cost_per_pack * croissants_per_pack / sandwiches_per_person

theorem committee_size_is_24 : committee_size = 24 := by
  sorry

end NUMINAMATH_CALUDE_committee_size_is_24_l2206_220662


namespace NUMINAMATH_CALUDE_digit_sum_theorem_l2206_220633

-- Define the conditions
def is_valid_digits (A B C : ℕ) : Prop :=
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A < 10 ∧ B < 10 ∧ C < 10

def BC (B C : ℕ) : ℕ := 10 * B + C

def ABC (A B C : ℕ) : ℕ := 100 * A + 10 * B + C

-- State the theorem
theorem digit_sum_theorem (A B C : ℕ) :
  is_valid_digits A B C →
  BC B C + ABC A B C + ABC A B C = 876 →
  A + B + C = 14 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_theorem_l2206_220633


namespace NUMINAMATH_CALUDE_paper_towel_savings_l2206_220672

theorem paper_towel_savings : 
  let case_price : ℚ := 9
  let individual_price : ℚ := 1
  let rolls_per_case : ℕ := 12
  let case_price_per_roll : ℚ := case_price / rolls_per_case
  let savings_per_roll : ℚ := individual_price - case_price_per_roll
  let percent_savings : ℚ := (savings_per_roll / individual_price) * 100
  percent_savings = 25 := by sorry

end NUMINAMATH_CALUDE_paper_towel_savings_l2206_220672


namespace NUMINAMATH_CALUDE_tom_teaching_years_l2206_220646

/-- Represents the number of years Tom has been teaching. -/
def tom_years : ℕ := sorry

/-- Represents the number of years Devin has been teaching. -/
def devin_years : ℕ := sorry

/-- The total number of years Tom and Devin have been teaching. -/
def total_years : ℕ := 70

/-- Theorem stating that Tom has been teaching for 50 years, given the conditions. -/
theorem tom_teaching_years :
  (tom_years + devin_years = total_years) ∧
  (devin_years = tom_years / 2 - 5) →
  tom_years = 50 :=
by sorry

end NUMINAMATH_CALUDE_tom_teaching_years_l2206_220646


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l2206_220671

/-- A coloring of the edges of a complete graph on 6 vertices -/
def Coloring := Fin 6 → Fin 6 → Fin 5

/-- A valid coloring ensures that each vertex has exactly one edge of each color -/
def is_valid_coloring (c : Coloring) : Prop :=
  ∀ v : Fin 6, ∀ color : Fin 5,
    ∃! w : Fin 6, w ≠ v ∧ c v w = color

/-- There exists a valid coloring of the complete graph K₆ using 5 colors -/
theorem exists_valid_coloring : ∃ c : Coloring, is_valid_coloring c := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l2206_220671


namespace NUMINAMATH_CALUDE_min_product_sum_l2206_220620

theorem min_product_sum (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) : 
  ({a₁, a₂, a₃, b₁, b₂, b₃} : Finset ℕ) = {1, 2, 3, 4, 5, 6} →
  a₁ * a₂ * a₃ + b₁ * b₂ * b₃ ≥ 56 ∧ 
  ∃ (x₁ x₂ x₃ y₁ y₂ y₃ : ℕ), 
    ({x₁, x₂, x₃, y₁, y₂, y₃} : Finset ℕ) = {1, 2, 3, 4, 5, 6} ∧
    x₁ * x₂ * x₃ + y₁ * y₂ * y₃ = 56 :=
by sorry

end NUMINAMATH_CALUDE_min_product_sum_l2206_220620


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2206_220600

/-- The interest rate at which B lent to C, given the conditions of the problem -/
def interest_rate_B_to_C (principal : ℚ) (rate_A_to_B : ℚ) (years : ℚ) (gain_B : ℚ) : ℚ :=
  let interest_A_to_B := principal * rate_A_to_B * years
  let total_interest_B_from_C := interest_A_to_B + gain_B
  (total_interest_B_from_C * 100) / (principal * years)

theorem interest_rate_calculation (principal : ℚ) (rate_A_to_B : ℚ) (years : ℚ) (gain_B : ℚ) :
  principal = 2000 →
  rate_A_to_B = 15 / 100 →
  years = 4 →
  gain_B = 160 →
  interest_rate_B_to_C principal rate_A_to_B years gain_B = 17 / 100 := by
  sorry

#eval interest_rate_B_to_C 2000 (15/100) 4 160

end NUMINAMATH_CALUDE_interest_rate_calculation_l2206_220600


namespace NUMINAMATH_CALUDE_block_placement_probability_l2206_220675

/-- Represents a person in the block placement problem -/
inductive Person
  | Louis
  | Maria
  | Neil

/-- Represents the colors of the blocks -/
inductive Color
  | Red
  | Blue
  | Yellow
  | White
  | Green
  | Purple

/-- The number of boxes -/
def num_boxes : ℕ := 6

/-- The number of blocks each person has -/
def num_blocks_per_person : ℕ := 6

/-- A function representing a random block placement for a person -/
def block_placement := Person → Fin num_boxes → Color

/-- The probability of a specific color being chosen for a specific box by all three people -/
def prob_color_match : ℚ := 1 / 216

/-- The probability that at least one box receives exactly 3 blocks of the same color,
    placed in alphabetical order by the people's names -/
def prob_at_least_one_box_match : ℚ := 235 / 1296

theorem block_placement_probability :
  prob_at_least_one_box_match = 1 - (1 - prob_color_match) ^ num_boxes :=
sorry

end NUMINAMATH_CALUDE_block_placement_probability_l2206_220675


namespace NUMINAMATH_CALUDE_xiao_jun_age_problem_l2206_220664

/-- Represents the current age of Xiao Jun -/
def xiao_jun_age : ℕ := 6

/-- Represents the current age ratio between Xiao Jun's mother and Xiao Jun -/
def current_age_ratio : ℕ := 5

/-- Represents the future age ratio between Xiao Jun's mother and Xiao Jun -/
def future_age_ratio : ℕ := 3

/-- Calculates the number of years that need to pass for Xiao Jun's mother's age 
    to be 3 times Xiao Jun's age -/
def years_passed : ℕ := 6

theorem xiao_jun_age_problem : 
  xiao_jun_age * current_age_ratio + years_passed = 
  (xiao_jun_age + years_passed) * future_age_ratio :=
sorry

end NUMINAMATH_CALUDE_xiao_jun_age_problem_l2206_220664


namespace NUMINAMATH_CALUDE_gcd_problem_l2206_220670

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, k % 2 = 1 ∧ b = k * 7769) :
  Int.gcd (4 * b^2 + 81 * b + 144) (2 * b + 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2206_220670


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l2206_220689

theorem complement_intersection_problem (U M N : Set Nat) : 
  U = {1, 2, 3, 4, 5} →
  M = {1, 2, 3} →
  N = {2, 3, 5} →
  (U \ M) ∩ N = {5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l2206_220689


namespace NUMINAMATH_CALUDE_tablets_consumed_l2206_220655

/-- Proves that given a person who takes one tablet every 15 minutes and consumes all tablets in 60 minutes, the total number of tablets taken is 4. -/
theorem tablets_consumed (interval : ℕ) (total_time : ℕ) (h1 : interval = 15) (h2 : total_time = 60) :
  total_time / interval = 4 := by
  sorry

end NUMINAMATH_CALUDE_tablets_consumed_l2206_220655


namespace NUMINAMATH_CALUDE_min_balls_needed_l2206_220693

/-- Represents the number of balls of each color -/
structure BallCounts where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Conditions for drawing balls -/
def satisfiesConditions (counts : BallCounts) : Prop :=
  counts.red ≥ 4 ∧
  counts.white ≥ 1 ∧
  counts.blue ≥ 1 ∧
  counts.green ≥ 1 ∧
  (counts.red.choose 4 : ℚ) = 
    (counts.red.choose 3 * counts.white : ℚ) ∧
  (counts.red.choose 3 * counts.white : ℚ) = 
    (counts.red.choose 2 * counts.white * counts.blue : ℚ) ∧
  (counts.red.choose 2 * counts.white * counts.blue : ℚ) = 
    (counts.red * counts.white * counts.blue * counts.green : ℚ)

/-- The theorem to be proved -/
theorem min_balls_needed : 
  ∃ (counts : BallCounts), 
    satisfiesConditions counts ∧ 
    (∀ (other : BallCounts), satisfiesConditions other → 
      counts.red + counts.white + counts.blue + counts.green ≤ 
      other.red + other.white + other.blue + other.green) ∧
    counts.red + counts.white + counts.blue + counts.green = 21 :=
sorry

end NUMINAMATH_CALUDE_min_balls_needed_l2206_220693


namespace NUMINAMATH_CALUDE_max_abs_sum_under_condition_l2206_220613

theorem max_abs_sum_under_condition (x y : ℝ) (h : x^2 + y^2 + x*y = 1) :
  |x| + |y| ≤ Real.sqrt (4/3) := by
  sorry

end NUMINAMATH_CALUDE_max_abs_sum_under_condition_l2206_220613


namespace NUMINAMATH_CALUDE_power_sum_fifth_l2206_220611

/-- Given real numbers a, b, x, y satisfying certain conditions, 
    prove that ax^5 + by^5 = 180.36 -/
theorem power_sum_fifth (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 24)
  (h4 : a * x^4 + b * y^4 = 56) :
  a * x^5 + b * y^5 = 180.36 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_fifth_l2206_220611


namespace NUMINAMATH_CALUDE_area_triangle_BFE_l2206_220623

/-- Given a rectangle ABCD with area 48 square units and points E and F dividing sides AD and BC
    in a 2:1 ratio, the area of triangle BFE is 24 square units. -/
theorem area_triangle_BFE (A B C D E F : ℝ × ℝ) : 
  let rectangle_area := 48
  let ratio := (2 : ℝ) / 3
  (∃ u v : ℝ, 
    A = (0, 0) ∧ 
    B = (3*u, 0) ∧ 
    C = (3*u, 3*v) ∧ 
    D = (0, 3*v) ∧
    E = (0, 2*v) ∧ 
    F = (2*u, 0) ∧
    3*u*3*v = rectangle_area ∧
    (D.2 - E.2) / D.2 = ratio ∧
    (C.1 - F.1) / C.1 = ratio) →
  (1/2 * |B.1*(E.2 - F.2) + E.1*(F.2 - B.2) + F.1*(B.2 - E.2)| = 24) :=
by sorry

end NUMINAMATH_CALUDE_area_triangle_BFE_l2206_220623


namespace NUMINAMATH_CALUDE_fraction_problem_l2206_220601

theorem fraction_problem (f : ℚ) : 3 + (1/2) * f * (1/5) * 90 = (1/15) * 90 ↔ f = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2206_220601


namespace NUMINAMATH_CALUDE_eq2_most_suitable_for_factorization_l2206_220609

/-- Represents a quadratic equation --/
inductive QuadraticEquation
  | Eq1 : QuadraticEquation  -- (x+1)(x-3)=2
  | Eq2 : QuadraticEquation  -- 2(x-2)^2=x^2-4
  | Eq3 : QuadraticEquation  -- x^2+3x-1=0
  | Eq4 : QuadraticEquation  -- 5(2-x)^2=3

/-- Predicate to determine if an equation is suitable for factorization --/
def isSuitableForFactorization : QuadraticEquation → Prop :=
  fun eq => match eq with
    | QuadraticEquation.Eq1 => False
    | QuadraticEquation.Eq2 => True
    | QuadraticEquation.Eq3 => False
    | QuadraticEquation.Eq4 => False

/-- Theorem stating that Eq2 is the most suitable for factorization --/
theorem eq2_most_suitable_for_factorization :
  ∀ eq : QuadraticEquation, 
    isSuitableForFactorization eq → eq = QuadraticEquation.Eq2 :=
by
  sorry

end NUMINAMATH_CALUDE_eq2_most_suitable_for_factorization_l2206_220609


namespace NUMINAMATH_CALUDE_tank_volume_ratio_l2206_220691

theorem tank_volume_ratio :
  ∀ (tank1_volume tank2_volume : ℚ),
  tank1_volume > 0 →
  tank2_volume > 0 →
  (3 / 4 : ℚ) * tank1_volume = (5 / 8 : ℚ) * tank2_volume →
  tank1_volume / tank2_volume = (5 / 6 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_tank_volume_ratio_l2206_220691


namespace NUMINAMATH_CALUDE_furniture_cost_price_l2206_220649

theorem furniture_cost_price (final_price : ℝ) : 
  final_price = 9522.84 →
  ∃ (cost_price : ℝ),
    cost_price = 7695 ∧
    final_price = (1.12 * (0.85 * (1.3 * cost_price))) :=
by sorry

end NUMINAMATH_CALUDE_furniture_cost_price_l2206_220649


namespace NUMINAMATH_CALUDE_inequality_relations_l2206_220673

theorem inequality_relations :
  (∃ a b : ℝ, a > b ∧ a^2 ≤ b^2) ∧
  (∀ a b : ℝ, a^2 > b^2 → |a| > |b|) ∧
  (∀ a b c : ℝ, a > b ↔ a + c > b + c) :=
sorry

end NUMINAMATH_CALUDE_inequality_relations_l2206_220673


namespace NUMINAMATH_CALUDE_ratio_of_areas_ratio_of_perimeters_l2206_220604

-- Define the side lengths of squares A and B
def side_length_A : ℝ := 48
def side_length_B : ℝ := 60

-- Define the areas of squares A and B
def area_A : ℝ := side_length_A ^ 2
def area_B : ℝ := side_length_B ^ 2

-- Define the perimeters of squares A and B
def perimeter_A : ℝ := 4 * side_length_A
def perimeter_B : ℝ := 4 * side_length_B

-- Theorem stating the ratio of areas
theorem ratio_of_areas :
  area_A / area_B = 16 / 25 := by sorry

-- Theorem stating the ratio of perimeters
theorem ratio_of_perimeters :
  perimeter_A / perimeter_B = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_ratio_of_areas_ratio_of_perimeters_l2206_220604


namespace NUMINAMATH_CALUDE_martin_ticket_count_l2206_220627

/-- The number of tickets Martin bought at full price -/
def full_price_tickets : ℕ := sorry

/-- The price of a full-price ticket in cents -/
def full_price : ℕ := 200

/-- The number of discounted tickets Martin bought -/
def discounted_tickets : ℕ := 4

/-- The price of a discounted ticket in cents -/
def discounted_price : ℕ := 160

/-- The total amount Martin spent in cents -/
def total_spent : ℕ := 1840

theorem martin_ticket_count :
  full_price_tickets * full_price + discounted_tickets * discounted_price = total_spent ∧
  full_price_tickets + discounted_tickets = 10 :=
sorry

end NUMINAMATH_CALUDE_martin_ticket_count_l2206_220627


namespace NUMINAMATH_CALUDE_number_of_divisors_of_60_l2206_220652

theorem number_of_divisors_of_60 : Nat.card (Nat.divisors 60) = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_60_l2206_220652


namespace NUMINAMATH_CALUDE_trapezoid_shaded_fraction_l2206_220684

/-- Represents a trapezoid divided into strips -/
structure StripedTrapezoid where
  num_strips : ℕ
  shaded_strips : ℕ

/-- The fraction of the trapezoid's area that is shaded -/
def shaded_fraction (t : StripedTrapezoid) : ℚ :=
  t.shaded_strips / t.num_strips

theorem trapezoid_shaded_fraction :
  ∀ t : StripedTrapezoid,
    t.num_strips = 7 →
    shaded_fraction t = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_shaded_fraction_l2206_220684


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2206_220632

theorem simple_interest_problem (P R : ℝ) (h1 : P > 0) (h2 : R > 0) :
  (P * (R + 1) * 3 / 100 = P * R * 3 / 100 + 75) → P = 2500 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2206_220632


namespace NUMINAMATH_CALUDE_intersection_perpendicular_tangents_l2206_220605

theorem intersection_perpendicular_tangents (a : ℝ) (h : a > 0) : 
  ∃ x : ℝ, x ∈ Set.Ioo 0 (Real.pi / 2) ∧ 
  (2 * Real.sin x = a * Real.cos x) ∧
  (2 * Real.cos x) * (-a * Real.sin x) = -1 
  → a = 2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_tangents_l2206_220605


namespace NUMINAMATH_CALUDE_derivative_neg_cos_l2206_220677

theorem derivative_neg_cos (x : ℝ) : deriv (fun x => -Real.cos x) x = Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_derivative_neg_cos_l2206_220677


namespace NUMINAMATH_CALUDE_sacks_filled_l2206_220694

theorem sacks_filled (wood_per_sack : ℕ) (total_wood : ℕ) (h1 : wood_per_sack = 20) (h2 : total_wood = 80) :
  total_wood / wood_per_sack = 4 := by
  sorry

end NUMINAMATH_CALUDE_sacks_filled_l2206_220694


namespace NUMINAMATH_CALUDE_unique_triplet_sum_l2206_220638

theorem unique_triplet_sum (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b ≤ c)
  (heq : (25 : ℚ) / 84 = 1 / a + 1 / (a * b) + 1 / (a * b * c)) :
  a + b + c = 17 := by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_sum_l2206_220638


namespace NUMINAMATH_CALUDE_eldorado_license_plates_l2206_220695

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of letter positions in a license plate -/
def letter_positions : ℕ := 3

/-- The number of digit positions in a license plate -/
def digit_positions : ℕ := 4

/-- The total number of possible valid license plates in Eldorado -/
def total_license_plates : ℕ := num_letters ^ letter_positions * num_digits ^ digit_positions

theorem eldorado_license_plates :
  total_license_plates = 175760000 := by
  sorry

end NUMINAMATH_CALUDE_eldorado_license_plates_l2206_220695


namespace NUMINAMATH_CALUDE_man_speed_against_current_l2206_220682

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem stating that given the specific speeds in the problem,
    the man's speed against the current is 10 km/hr. -/
theorem man_speed_against_current :
  speed_against_current 15 2.5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_against_current_l2206_220682


namespace NUMINAMATH_CALUDE_janet_action_figures_l2206_220654

/-- Calculates the final number of action figures Janet has -/
def final_action_figure_count (initial_count : ℕ) (sold_count : ℕ) (bought_count : ℕ) : ℕ :=
  let remaining_count := initial_count - sold_count
  let after_purchase_count := remaining_count + bought_count
  after_purchase_count + 2 * after_purchase_count

theorem janet_action_figures :
  final_action_figure_count 10 6 4 = 24 := by
  sorry

#eval final_action_figure_count 10 6 4

end NUMINAMATH_CALUDE_janet_action_figures_l2206_220654


namespace NUMINAMATH_CALUDE_unique_distribution_l2206_220692

/-- Represents the number of ways to distribute n identical balls into boxes with given capacities -/
def distribution_count (n : ℕ) (capacities : List ℕ) : ℕ :=
  sorry

/-- The capacities of the four boxes -/
def box_capacities : List ℕ := [3, 5, 7, 8]

/-- The total number of balls to distribute -/
def total_balls : ℕ := 19

/-- Theorem stating that there's only one way to distribute the balls -/
theorem unique_distribution : distribution_count total_balls box_capacities = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_distribution_l2206_220692


namespace NUMINAMATH_CALUDE_polygon_150_diagonals_l2206_220641

/-- Number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem polygon_150_diagonals :
  (num_diagonals 150 = 11025) ∧
  (9900 ≠ num_diagonals 150 / 2) :=
by sorry

end NUMINAMATH_CALUDE_polygon_150_diagonals_l2206_220641


namespace NUMINAMATH_CALUDE_largest_arithmetic_mean_of_special_pairs_l2206_220635

theorem largest_arithmetic_mean_of_special_pairs : ∃ (a b : ℕ),
  10 ≤ a ∧ a < 100 ∧
  10 ≤ b ∧ b < 100 ∧
  (a + b) / 2 = (25 / 24) * Real.sqrt (a * b) ∧
  ∀ (c d : ℕ),
    10 ≤ c ∧ c < 100 ∧
    10 ≤ d ∧ d < 100 ∧
    (c + d) / 2 = (25 / 24) * Real.sqrt (c * d) →
    (a + b) / 2 ≥ (c + d) / 2 ∧
  (a + b) / 2 = 75 :=
sorry

end NUMINAMATH_CALUDE_largest_arithmetic_mean_of_special_pairs_l2206_220635


namespace NUMINAMATH_CALUDE_georges_initial_socks_l2206_220687

theorem georges_initial_socks (bought new_from_dad total_now : ℕ) 
  (h1 : bought = 36)
  (h2 : new_from_dad = 4)
  (h3 : total_now = 68)
  : total_now - bought - new_from_dad = 28 := by
  sorry

end NUMINAMATH_CALUDE_georges_initial_socks_l2206_220687


namespace NUMINAMATH_CALUDE_expression_undefined_iff_x_eq_11_l2206_220607

theorem expression_undefined_iff_x_eq_11 (x : ℝ) :
  ¬ (∃ y : ℝ, y = (3 * x^3 + 4) / (x^2 - 22*x + 121)) ↔ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_undefined_iff_x_eq_11_l2206_220607


namespace NUMINAMATH_CALUDE_two_consecutive_late_charges_l2206_220606

theorem two_consecutive_late_charges (original_bill : ℝ) (late_charge_rate : ℝ) : 
  original_bill = 400 → 
  late_charge_rate = 0.01 → 
  original_bill * (1 + late_charge_rate)^2 = 408.04 := by
sorry


end NUMINAMATH_CALUDE_two_consecutive_late_charges_l2206_220606


namespace NUMINAMATH_CALUDE_range_of_f_l2206_220608

-- Define the function f
def f (x : ℝ) : ℝ := 2*x - x^2

-- Define the domain
def domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3

-- Theorem statement
theorem range_of_f :
  ∃ (y : ℝ), (∃ (x : ℝ), domain x ∧ f x = y) ↔ -3 ≤ y ∧ y ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2206_220608


namespace NUMINAMATH_CALUDE_exam_students_count_l2206_220628

theorem exam_students_count :
  ∀ (N : ℕ) (T : ℝ),
    N > 5 →
    T = N * 80 →
    (T - 250) / (N - 5 : ℝ) = 90 →
    N = 20 := by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l2206_220628


namespace NUMINAMATH_CALUDE_unique_root_monotonic_continuous_l2206_220697

theorem unique_root_monotonic_continuous {f : ℝ → ℝ} {a b : ℝ} (h_mono : Monotone f) (h_cont : Continuous f) (h_sign : f a * f b < 0) (h_le : a ≤ b) :
  ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_root_monotonic_continuous_l2206_220697


namespace NUMINAMATH_CALUDE_new_person_weight_l2206_220681

theorem new_person_weight (initial_count : ℕ) (initial_average : ℝ) (weight_decrease : ℝ) :
  initial_count = 20 →
  initial_average = 60 →
  weight_decrease = 5 →
  let total_weight := initial_count * initial_average
  let new_count := initial_count + 1
  let new_average := initial_average - weight_decrease
  let new_person_weight := new_count * new_average - total_weight
  new_person_weight = 55 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l2206_220681


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l2206_220663

/-- The trajectory of point P given the conditions in the problem -/
def trajectory (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

theorem trajectory_is_ellipse (x y : ℝ) :
  let P : ℝ × ℝ := (x, y)
  let M : ℝ × ℝ := (1, 0)
  let d : ℝ := |x - 2|
  (‖P - M‖ : ℝ) / d = Real.sqrt 2 / 2 →
  trajectory x y :=
by sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l2206_220663


namespace NUMINAMATH_CALUDE_keith_pears_count_l2206_220658

def total_pears : Nat := 5
def jason_pears : Nat := 2

theorem keith_pears_count : total_pears - jason_pears = 3 := by
  sorry

end NUMINAMATH_CALUDE_keith_pears_count_l2206_220658


namespace NUMINAMATH_CALUDE_power_two_greater_than_square_plus_one_l2206_220614

theorem power_two_greater_than_square_plus_one (n : ℕ) (h : n ≥ 5) :
  2^n > n^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_two_greater_than_square_plus_one_l2206_220614


namespace NUMINAMATH_CALUDE_power_function_through_point_l2206_220656

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the theorem
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f → f 3 = Real.sqrt 3 → f 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2206_220656


namespace NUMINAMATH_CALUDE_f_properties_l2206_220642

noncomputable def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

theorem f_properties :
  (∃ (f' : ℝ → ℝ), DifferentiableAt ℝ f 2 ∧ deriv f 2 = f' 2 ∧ f' 2 < 0) ∧
  (∃ (x_max : ℝ), x_max = 1 ∧ ∀ x, f x ≤ f x_max ∧ f x_max = Real.exp 1) ∧
  (∀ x > 1, ∃ (f' : ℝ → ℝ), DifferentiableAt ℝ f x ∧ deriv f x = f' x ∧ f' x < 0) ∧
  (∀ a : ℝ, (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = a ∧ f x₂ = a) → 0 < a ∧ a < Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2206_220642


namespace NUMINAMATH_CALUDE_probability_sum_less_than_product_l2206_220653

def valid_pairs : Finset (ℕ × ℕ) :=
  (Finset.range 6).product (Finset.range 6)

def satisfying_pairs : Finset (ℕ × ℕ) :=
  valid_pairs.filter (fun p => p.1 + p.2 < p.1 * p.2)

theorem probability_sum_less_than_product :
  (satisfying_pairs.card : ℚ) / valid_pairs.card = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_less_than_product_l2206_220653


namespace NUMINAMATH_CALUDE_equation_solution_in_interval_l2206_220648

theorem equation_solution_in_interval :
  ∃ x₀ ∈ Set.Ioo 2 3, Real.log x₀ + x₀ - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_in_interval_l2206_220648


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l2206_220637

theorem sqrt_equality_implies_t_value :
  ∀ t : ℝ, (Real.sqrt (3 * Real.sqrt (t - 3)) = (9 - t) ^ (1/4)) → t = 3.6 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l2206_220637


namespace NUMINAMATH_CALUDE_fraction_value_unchanged_keep_fraction_unchanged_l2206_220698

theorem fraction_value_unchanged (a b c : ℤ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (a : ℚ) / b = (a + c) / (b + ((a + c) * b / a - b)) :=
by sorry

theorem keep_fraction_unchanged :
  let original_numerator := 3
  let original_denominator := 4
  let numerator_increase := 9
  let new_numerator := original_numerator + numerator_increase
  let denominator_increase := new_numerator * original_denominator / original_numerator - original_denominator
  denominator_increase = 12 :=
by sorry

end NUMINAMATH_CALUDE_fraction_value_unchanged_keep_fraction_unchanged_l2206_220698


namespace NUMINAMATH_CALUDE_henrys_game_purchase_l2206_220643

/-- Henry's money problem -/
theorem henrys_game_purchase (initial : ℕ) (birthday_gift : ℕ) (final : ℕ) 
  (h1 : initial = 11)
  (h2 : birthday_gift = 18)
  (h3 : final = 19) :
  initial + birthday_gift - final = 10 := by
  sorry

end NUMINAMATH_CALUDE_henrys_game_purchase_l2206_220643


namespace NUMINAMATH_CALUDE_swimming_lane_length_l2206_220629

/-- Represents the length of a swimming lane in meters -/
def lane_length : ℝ := 100

/-- Represents the number of round trips swam -/
def round_trips : ℕ := 4

/-- Represents the total distance swam in meters -/
def total_distance : ℝ := 800

/-- Represents the number of lane lengths in a round trip -/
def lengths_per_round_trip : ℕ := 2

theorem swimming_lane_length :
  lane_length * (round_trips * lengths_per_round_trip) = total_distance :=
sorry

end NUMINAMATH_CALUDE_swimming_lane_length_l2206_220629


namespace NUMINAMATH_CALUDE_triangle_properties_l2206_220667

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (4 * a = Real.sqrt 5 * c) →
  (Real.cos C = 3 / 5) →
  (b = 11) →
  (Real.sin A = Real.sqrt 5 / 5) ∧
  (1 / 2 * a * b * Real.sin C = 22) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2206_220667


namespace NUMINAMATH_CALUDE_polynomial_equality_l2206_220660

theorem polynomial_equality (x t s : ℝ) : 
  (3 * x^2 - 4 * x + 9) * (5 * x^2 + t * x + s) = 
  15 * x^4 - 22 * x^3 + (41 + s) * x^2 - 34 * x + 9 * s ↔ 
  t = -2 ∧ s = s := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2206_220660


namespace NUMINAMATH_CALUDE_function_solution_l2206_220624

-- Define the function type
def FunctionType := ℝ → ℝ

-- Define the functional equation
def SatisfiesEquation (g : FunctionType) : Prop :=
  ∀ x y : ℝ, (g x * g y - g (x * y)) / 4 = x + y + 1

-- State the theorem
theorem function_solution (g : FunctionType) (h : SatisfiesEquation g) :
  (∀ x : ℝ, g x = 2 * x + 3) ∨ (∀ x : ℝ, g x = -2 * x - 1) :=
sorry

end NUMINAMATH_CALUDE_function_solution_l2206_220624


namespace NUMINAMATH_CALUDE_f_inequality_l2206_220661

-- Define a function f on the real numbers
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x < y → f x < f y

-- State the theorem
theorem f_inequality (h1 : is_even f) (h2 : is_increasing_on_nonneg f) : f (-2) > f 1 ∧ f 1 > f 0 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l2206_220661


namespace NUMINAMATH_CALUDE_divisibility_by_24_l2206_220603

theorem divisibility_by_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := by sorry

end NUMINAMATH_CALUDE_divisibility_by_24_l2206_220603
