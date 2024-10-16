import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l2053_205361

theorem equation_solution : ∃! y : ℚ, (1 / 6 : ℚ) + 6 / y = 14 / y + (1 / 14 : ℚ) ∧ y = 84 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2053_205361


namespace NUMINAMATH_CALUDE_stirling_second_kind_recurrence_stirling_second_kind_5_3_l2053_205379

def S (n k : ℕ) : ℕ := sorry

theorem stirling_second_kind_recurrence (n k : ℕ) (h : 1 ≤ k ∧ k ≤ n) :
  S (n + 1) k = S n (k - 1) + k * S n k := by sorry

theorem stirling_second_kind_5_3 :
  S 5 3 = 25 := by sorry

end NUMINAMATH_CALUDE_stirling_second_kind_recurrence_stirling_second_kind_5_3_l2053_205379


namespace NUMINAMATH_CALUDE_expression_equals_m_times_ten_to_1006_l2053_205334

theorem expression_equals_m_times_ten_to_1006 : 
  (3^1005 + 7^1006)^2 - (3^1005 - 7^1006)^2 = 114337548 * 10^1006 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_m_times_ten_to_1006_l2053_205334


namespace NUMINAMATH_CALUDE_min_xy_value_l2053_205335

theorem min_xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2*x + y + 6 = x*y) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2*a + b + 6 = a*b → x*y ≤ a*b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y + 6 = x*y ∧ x*y = 18 :=
sorry

end NUMINAMATH_CALUDE_min_xy_value_l2053_205335


namespace NUMINAMATH_CALUDE_race_distance_l2053_205331

theorem race_distance (race_length : ℝ) (gap : ℝ) : 
  race_length > 0 → 
  gap > 0 → 
  gap < race_length → 
  let v1 := race_length
  let v2 := race_length - gap
  let v3 := (race_length - gap) * ((race_length - gap) / race_length)
  (race_length - v3) = 19 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l2053_205331


namespace NUMINAMATH_CALUDE_speaker_arrangement_count_l2053_205327

def number_of_speakers : ℕ := 5

theorem speaker_arrangement_count :
  (number_of_speakers.factorial / 2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_speaker_arrangement_count_l2053_205327


namespace NUMINAMATH_CALUDE_no_rational_roots_odd_coefficients_l2053_205360

theorem no_rational_roots_odd_coefficients (a b c : ℤ) 
  (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_odd_coefficients_l2053_205360


namespace NUMINAMATH_CALUDE_function_equivalence_l2053_205347

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_equivalence : 
  (∀ x : ℝ, f (2 * x) = 6 * x - 1) → 
  (∀ x : ℝ, f x = 3 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_function_equivalence_l2053_205347


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_l2053_205346

theorem arithmetic_geometric_progression (a b : ℝ) : 
  (1 - a = b - 1) ∧ (1 = a^2 * b^2) → 
  ((a = 1 ∧ b = 1) ∨ 
   (a = 1 + Real.sqrt 2 ∧ b = 1 - Real.sqrt 2) ∨ 
   (a = 1 - Real.sqrt 2 ∧ b = 1 + Real.sqrt 2)) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_l2053_205346


namespace NUMINAMATH_CALUDE_mary_final_cards_l2053_205350

/-- Calculates the final number of baseball cards Mary has after a series of transactions -/
def final_card_count (initial_cards torn_cards fred_cards bought_cards lost_cards lisa_trade_in lisa_trade_out alex_trade_in alex_trade_out : ℕ) : ℕ :=
  initial_cards - torn_cards + fred_cards + bought_cards - lost_cards - lisa_trade_in + lisa_trade_out - alex_trade_in + alex_trade_out

/-- Theorem stating that Mary ends up with 70 baseball cards -/
theorem mary_final_cards : 
  final_card_count 18 8 26 40 5 3 4 7 5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_mary_final_cards_l2053_205350


namespace NUMINAMATH_CALUDE_neil_initial_games_l2053_205337

theorem neil_initial_games (henry_initial : ℕ) (games_given : ℕ) (neil_initial : ℕ) :
  henry_initial = 58 →
  games_given = 6 →
  henry_initial - games_given = 4 * (neil_initial + games_given) →
  neil_initial = 7 := by
sorry

end NUMINAMATH_CALUDE_neil_initial_games_l2053_205337


namespace NUMINAMATH_CALUDE_max_profit_at_180_l2053_205358

/-- The total cost function for a certain product -/
def total_cost (x : ℝ) : ℝ := 0.1 * x^2 - 11 * x + 3000

/-- The selling price per unit in ten thousand yuan -/
def selling_price : ℝ := 25

/-- The profit function -/
def profit (x : ℝ) : ℝ := selling_price * x - total_cost x

/-- Theorem: The production volume that maximizes profit is 180 units -/
theorem max_profit_at_180 : 
  ∃ (max_x : ℝ), (∀ x : ℝ, profit x ≤ profit max_x) ∧ max_x = 180 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_180_l2053_205358


namespace NUMINAMATH_CALUDE_min_product_of_three_numbers_l2053_205367

theorem min_product_of_three_numbers (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1)
  (x_leq_2y : x ≤ 2*y)
  (y_leq_2z : y ≤ 2*z) :
  x * y * z ≥ 6 / 343 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_three_numbers_l2053_205367


namespace NUMINAMATH_CALUDE_max_garden_area_l2053_205372

/-- The maximum area of a rectangular garden with one side along a wall and 400 feet of fencing for the other three sides. -/
theorem max_garden_area : 
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ l + 2*w = 400 ∧
  (∀ (l' w' : ℝ), l' > 0 → w' > 0 → l' + 2*w' = 400 → l'*w' ≤ l*w) ∧
  l*w = 20000 :=
by sorry

end NUMINAMATH_CALUDE_max_garden_area_l2053_205372


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2053_205382

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 2*x*y - 3 = 0) :
  ∀ z, z = 2*x + y → z ≥ 3 ∧ ∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ x₀^2 + 2*x₀*y₀ - 3 = 0 ∧ 2*x₀ + y₀ = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2053_205382


namespace NUMINAMATH_CALUDE_opposite_numbers_equation_l2053_205385

theorem opposite_numbers_equation (x : ℝ) : 2 * (x - 3) = -(4 * (1 - x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_equation_l2053_205385


namespace NUMINAMATH_CALUDE_square_sum_bound_l2053_205390

theorem square_sum_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a^2 + b^2 ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_bound_l2053_205390


namespace NUMINAMATH_CALUDE_triangle_area_l2053_205329

/-- The area of a triangle with base 9 cm and height 12 cm is 54 cm² -/
theorem triangle_area : 
  let base : ℝ := 9
  let height : ℝ := 12
  (1/2 : ℝ) * base * height = 54
  := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2053_205329


namespace NUMINAMATH_CALUDE_james_college_cost_l2053_205355

/-- The cost of James's community college units over 2 semesters -/
theorem james_college_cost (units_per_semester : ℕ) (cost_per_unit : ℕ) (num_semesters : ℕ) : 
  units_per_semester = 20 → cost_per_unit = 50 → num_semesters = 2 →
  units_per_semester * cost_per_unit * num_semesters = 2000 := by
  sorry

#check james_college_cost

end NUMINAMATH_CALUDE_james_college_cost_l2053_205355


namespace NUMINAMATH_CALUDE_circle_and_line_equations_l2053_205396

-- Define the circles and points
def circle_O (x y : ℝ) := x^2 + y^2 = 16
def circle_C (x y : ℝ) := (x + 1)^2 + (y + 1)^2 = 2
def point_P : ℝ × ℝ := (-4, 0)

-- Define the line l
def line_l (x y : ℝ) := (3 * x - y = 0) ∨ (3 * x - y + 4 = 0)

-- Define the conditions
def condition_P_on_O := circle_O point_P.1 point_P.2

-- Theorem statement
theorem circle_and_line_equations :
  ∃ (A B M N : ℝ × ℝ),
    -- Line l intersects circle O at A and B
    circle_O A.1 A.2 ∧ circle_O B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    -- Line l intersects circle C at M and N
    circle_C M.1 M.2 ∧ circle_C N.1 N.2 ∧
    line_l M.1 M.2 ∧ line_l N.1 N.2 ∧
    -- M is the midpoint of AB
    M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2 ∧
    -- |PM| = |PN|
    (M.1 - point_P.1)^2 + (M.2 - point_P.2)^2 =
    (N.1 - point_P.1)^2 + (N.2 - point_P.2)^2 ∧
    -- Point P is on circle O
    condition_P_on_O :=
  by sorry

end NUMINAMATH_CALUDE_circle_and_line_equations_l2053_205396


namespace NUMINAMATH_CALUDE_function_range_l2053_205344

theorem function_range (θ : ℝ) : 
  ∀ x : ℝ, 2 - Real.sqrt 3 ≤ (x^2 + 2*x*Real.sin θ + 2) / (x^2 + 2*x*Real.cos θ + 2) 
         ∧ (x^2 + 2*x*Real.sin θ + 2) / (x^2 + 2*x*Real.cos θ + 2) ≤ 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l2053_205344


namespace NUMINAMATH_CALUDE_product_simplification_l2053_205394

theorem product_simplification (x y : ℝ) :
  (3 * x^4 - 7 * y^3) * (9 * x^8 + 21 * x^4 * y^3 + 49 * y^6) = 27 * x^12 - 343 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l2053_205394


namespace NUMINAMATH_CALUDE_train_length_calculation_l2053_205387

-- Define the given values
def train_speed : ℝ := 60  -- km/hr
def man_speed : ℝ := 6     -- km/hr
def time_to_pass : ℝ := 17.998560115190788  -- seconds

-- Define the theorem
theorem train_length_calculation :
  let relative_speed : ℝ := (train_speed + man_speed) * (5 / 18)  -- Convert to m/s
  let train_length : ℝ := relative_speed * time_to_pass
  train_length = 330 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2053_205387


namespace NUMINAMATH_CALUDE_jane_tickets_l2053_205374

/-- The maximum number of tickets Jane can buy after purchasing a scarf -/
def max_tickets (initial_money : ℕ) (scarf_cost : ℕ) (ticket_cost : ℕ) : ℕ :=
  (initial_money - scarf_cost) / ticket_cost

/-- Proof that Jane can buy 9 tickets after purchasing the scarf -/
theorem jane_tickets :
  max_tickets 160 25 15 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jane_tickets_l2053_205374


namespace NUMINAMATH_CALUDE_cube_root_of_5488000_l2053_205302

theorem cube_root_of_5488000 : (5488000 : ℝ) ^ (1/3 : ℝ) = 40 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_5488000_l2053_205302


namespace NUMINAMATH_CALUDE_unique_solution_l2053_205310

/-- The price of Lunasa's violin -/
def violin_price : ℝ := sorry

/-- The price of Merlin's trumpet -/
def trumpet_price : ℝ := sorry

/-- The price of Lyrica's piano -/
def piano_price : ℝ := sorry

/-- Condition (a): If violin price is raised by 50% and trumpet price is decreased by 50%,
    violin is $50 more expensive than trumpet -/
axiom condition_a : 1.5 * violin_price = 0.5 * trumpet_price + 50

/-- Condition (b): If trumpet price is raised by 50% and piano price is decreased by 50%,
    trumpet is $50 more expensive than piano -/
axiom condition_b : 1.5 * trumpet_price = 0.5 * piano_price + 50

/-- The percentage m by which violin price is raised and piano price is decreased -/
def m : ℤ := sorry

/-- The price difference n between the adjusted violin and piano prices -/
def n : ℤ := sorry

/-- The relationship between adjusted violin and piano prices -/
axiom price_relationship : (100 + m) * violin_price / 100 = n + (100 - m) * piano_price / 100

theorem unique_solution : m = 80 ∧ n = 80 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l2053_205310


namespace NUMINAMATH_CALUDE_sum_of_first_45_terms_l2053_205336

def a (n : ℕ) : ℕ := 2^(n-1)

def b (n : ℕ) : ℕ := 3*n - 1

def c (n : ℕ) : ℕ := a n + b n

def S (n : ℕ) : ℕ := (2^n - 1) + n * (3*n + 1) / 2 - (2 + 8 + 32)

theorem sum_of_first_45_terms : S 45 = 2^45 - 3017 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_45_terms_l2053_205336


namespace NUMINAMATH_CALUDE_red_pigment_in_brown_l2053_205306

/-- Represents the composition of a paint mixture -/
structure PaintMixture where
  blue : Real
  red : Real
  yellow : Real
  weight : Real

/-- The sky blue paint composition -/
def skyBlue : PaintMixture := {
  blue := 0.1
  red := 0.9
  yellow := 0
  weight := 1
}

/-- The green paint composition -/
def green : PaintMixture := {
  blue := 0.7
  red := 0
  yellow := 0.3
  weight := 1
}

/-- The resulting brown paint composition -/
def brown : PaintMixture := {
  blue := 0.4
  red := 0
  yellow := 0
  weight := 10
}

/-- Theorem stating the amount of red pigment in the brown paint -/
theorem red_pigment_in_brown :
  ∃ (x y : Real),
    x + y = brown.weight ∧
    x * skyBlue.blue + y * green.blue = brown.blue * brown.weight ∧
    x * skyBlue.red = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_red_pigment_in_brown_l2053_205306


namespace NUMINAMATH_CALUDE_rectangle_diagonal_pi_irrational_l2053_205365

theorem rectangle_diagonal_pi_irrational 
  (m n p q : ℤ) 
  (hn : n ≠ 0) 
  (hq : q ≠ 0) :
  let l : ℚ := m / n
  let w : ℚ := p / q
  let d : ℝ := Real.sqrt ((l * l + w * w : ℚ) : ℝ)
  Irrational (π * d) := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_pi_irrational_l2053_205365


namespace NUMINAMATH_CALUDE_total_watermelon_seeds_l2053_205399

/-- The number of watermelon seeds each person has -/
structure WatermelonSeeds where
  bom : ℕ
  gwi : ℕ
  yeon : ℕ
  eun : ℕ

/-- Given conditions about watermelon seeds -/
def watermelon_seed_conditions (w : WatermelonSeeds) : Prop :=
  w.yeon = 3 * w.gwi ∧
  w.gwi = w.bom + 40 ∧
  w.eun = 2 * w.gwi ∧
  w.bom = 300

/-- Theorem stating the total number of watermelon seeds -/
theorem total_watermelon_seeds (w : WatermelonSeeds) 
  (h : watermelon_seed_conditions w) : 
  w.bom + w.gwi + w.yeon + w.eun = 2340 := by
  sorry

end NUMINAMATH_CALUDE_total_watermelon_seeds_l2053_205399


namespace NUMINAMATH_CALUDE_homework_time_difference_l2053_205395

/-- Given the conditions of the homework problem, prove that Greg has 6 hours less than Jacob. -/
theorem homework_time_difference :
  ∀ (greg_hours patrick_hours jacob_hours : ℕ),
  patrick_hours = 2 * greg_hours - 4 →
  jacob_hours = 18 →
  patrick_hours + greg_hours + jacob_hours = 50 →
  jacob_hours - greg_hours = 6 := by
sorry

end NUMINAMATH_CALUDE_homework_time_difference_l2053_205395


namespace NUMINAMATH_CALUDE_crayons_difference_l2053_205311

/-- Given the initial number of crayons, the number of crayons given away, and the number of crayons lost,
    prove that the difference between crayons given away and crayons lost is 410. -/
theorem crayons_difference (initial : ℕ) (given_away : ℕ) (lost : ℕ)
  (h1 : initial = 589)
  (h2 : given_away = 571)
  (h3 : lost = 161) :
  given_away - lost = 410 := by
  sorry

end NUMINAMATH_CALUDE_crayons_difference_l2053_205311


namespace NUMINAMATH_CALUDE_alice_win_condition_l2053_205342

/-- The game state represents the positions of the red and blue pieces -/
structure GameState where
  red : ℚ
  blue : ℚ

/-- Alice's move function -/
def move (r : ℚ) (state : GameState) (k : ℤ) : GameState :=
  { red := state.red,
    blue := state.red + r^k * (state.blue - state.red) }

/-- Alice can win the game -/
def can_win (r : ℚ) : Prop :=
  ∃ (moves : List ℤ), moves.length ≤ 2021 ∧
    (moves.foldl (move r) { red := 0, blue := 1 }).red = 1

/-- The main theorem stating the condition for Alice to win -/
theorem alice_win_condition (r : ℚ) : 
  (r > 1 ∧ can_win r) ↔ (∃ d : ℕ, d ≥ 1 ∧ d ≤ 1010 ∧ r = 1 + 1 / d) := by
  sorry


end NUMINAMATH_CALUDE_alice_win_condition_l2053_205342


namespace NUMINAMATH_CALUDE_house_rent_fraction_l2053_205322

theorem house_rent_fraction (salary : ℕ) (food_fraction : ℚ) (clothes_fraction : ℚ) (remaining : ℕ) 
  (h1 : salary = 160000)
  (h2 : food_fraction = 1/5)
  (h3 : clothes_fraction = 3/5)
  (h4 : remaining = 16000)
  (h5 : ∃ (house_rent_fraction : ℚ), salary * (1 - food_fraction - clothes_fraction - house_rent_fraction) = remaining) :
  ∃ (house_rent_fraction : ℚ), house_rent_fraction = 1/10 := by
sorry

end NUMINAMATH_CALUDE_house_rent_fraction_l2053_205322


namespace NUMINAMATH_CALUDE_combined_weight_in_pounds_l2053_205352

-- Define the weight of the elephant in tons
def elephant_weight_tons : ℝ := 3

-- Define the conversion factor from tons to pounds
def tons_to_pounds : ℝ := 2000

-- Define the weight ratio of the donkey compared to the elephant
def donkey_weight_ratio : ℝ := 0.1

-- Theorem statement
theorem combined_weight_in_pounds :
  let elephant_weight_pounds := elephant_weight_tons * tons_to_pounds
  let donkey_weight_pounds := elephant_weight_pounds * donkey_weight_ratio
  elephant_weight_pounds + donkey_weight_pounds = 6600 := by
sorry

end NUMINAMATH_CALUDE_combined_weight_in_pounds_l2053_205352


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l2053_205313

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- First asymptote function -/
  asymptote1 : ℝ → ℝ
  /-- Second asymptote function -/
  asymptote2 : ℝ → ℝ
  /-- A point that the hyperbola passes through -/
  point : ℝ × ℝ

/-- The distance between the foci of a hyperbola -/
def foci_distance (h : Hyperbola) : ℝ :=
  sorry

/-- Theorem stating that for a hyperbola with given asymptotes and passing through (4,4),
    the distance between its foci is 8 -/
theorem hyperbola_foci_distance :
  ∀ (h : Hyperbola),
    h.asymptote1 = (λ x => x + 2) ∧
    h.asymptote2 = (λ x => 4 - x) ∧
    h.point = (4, 4) →
    foci_distance h = 8 :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l2053_205313


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2053_205353

theorem solution_set_inequality (f : ℝ → ℝ) (hf : ∀ x, f x + (deriv f) x > 1) (hf0 : f 0 = 4) :
  {x : ℝ | f x > 3 / Real.exp x + 1} = {x : ℝ | x > 0} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2053_205353


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2053_205364

theorem fraction_subtraction : (18 : ℚ) / 42 - 2 / 9 = 13 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2053_205364


namespace NUMINAMATH_CALUDE_max_distance_circle_ellipse_l2053_205328

/-- The maximum distance between any point on the circle x^2 + (y-6)^2 = 2 
    and any point on the ellipse x^2/10 + y^2 = 1 is 6√2 -/
theorem max_distance_circle_ellipse : 
  ∃ (max_dist : ℝ), max_dist = 6 * Real.sqrt 2 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 + (y₁ - 6)^2 = 2) →  -- Point on the circle
    (x₂^2 / 10 + y₂^2 = 1) →   -- Point on the ellipse
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_circle_ellipse_l2053_205328


namespace NUMINAMATH_CALUDE_ariel_fencing_start_year_l2053_205398

def birth_year : ℕ := 1992
def current_age : ℕ := 30
def fencing_years : ℕ := 16

theorem ariel_fencing_start_year :
  birth_year + current_age - fencing_years = 2006 :=
by sorry

end NUMINAMATH_CALUDE_ariel_fencing_start_year_l2053_205398


namespace NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l2053_205333

theorem sqrt_sum_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l2053_205333


namespace NUMINAMATH_CALUDE_jenny_house_worth_l2053_205393

/-- The current worth of Jenny's house -/
def house_worth : ℝ := 500000

/-- Jenny's property tax rate -/
def tax_rate : ℝ := 0.02

/-- The increase in house value due to the high-speed rail project -/
def value_increase : ℝ := 0.25

/-- The maximum amount Jenny can spend on property tax per year -/
def max_tax : ℝ := 15000

/-- The value of improvements Jenny can make to her house -/
def improvements : ℝ := 250000

theorem jenny_house_worth :
  tax_rate * (house_worth * (1 + value_increase) + improvements) = max_tax := by
  sorry

#check jenny_house_worth

end NUMINAMATH_CALUDE_jenny_house_worth_l2053_205393


namespace NUMINAMATH_CALUDE_expression_evaluation_l2053_205389

/-- Convert a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The result of the given expression in base 10 -/
def result : ℚ :=
  (toBase10 [4, 5, 2] 8 : ℚ) / (toBase10 [3, 1] 3) +
  (toBase10 [3, 0, 2] 5 : ℚ) / (toBase10 [2, 2] 4)

theorem expression_evaluation :
  result = 33.966666666666665 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2053_205389


namespace NUMINAMATH_CALUDE_range_of_f_l2053_205378

-- Define the function
def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ∈ Set.Iic 8 :=
by sorry

-- Note: Set.Iic 8 represents the set (-∞, 8]

end NUMINAMATH_CALUDE_range_of_f_l2053_205378


namespace NUMINAMATH_CALUDE_max_sum_of_a_and_b_l2053_205359

theorem max_sum_of_a_and_b : ∀ a b : ℕ+,
  b > 2 →
  a^(b:ℕ) < 600 →
  a + b ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_a_and_b_l2053_205359


namespace NUMINAMATH_CALUDE_sara_wins_731_l2053_205301

/-- Represents the state of a wall in the brick removal game -/
def Wall := List Nat

/-- Calculates the nim-value of a single wall -/
def nimValue (wall : Nat) : Nat :=
  sorry

/-- Calculates the nim-sum (XOR) of a list of natural numbers -/
def nimSum (values : List Nat) : Nat :=
  sorry

/-- Determines if a given game state is a winning position for the second player -/
def isWinningForSecondPlayer (state : Wall) : Prop :=
  nimSum (state.map nimValue) = 0

/-- The main theorem stating that (7, 3, 1) is a winning position for the second player -/
theorem sara_wins_731 : isWinningForSecondPlayer [7, 3, 1] := by
  sorry

end NUMINAMATH_CALUDE_sara_wins_731_l2053_205301


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2053_205341

theorem complex_expression_equality : (8 * 5.4 - 0.6 * 10 / 1.2) ^ 2 = 1459.24 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2053_205341


namespace NUMINAMATH_CALUDE_complex_product_simplification_l2053_205348

theorem complex_product_simplification :
  let i : ℂ := Complex.I
  ((4 - 3*i) - (2 + 5*i)) * (2*i) = 16 + 4*i := by sorry

end NUMINAMATH_CALUDE_complex_product_simplification_l2053_205348


namespace NUMINAMATH_CALUDE_elvis_recording_time_l2053_205300

/-- Calculates the time to record each song given the total number of songs,
    total studio time in hours, editing time for all songs in minutes,
    and writing time for each song in minutes. -/
def time_to_record_each_song (total_songs : ℕ) (studio_time_hours : ℕ) 
    (editing_time_mins : ℕ) (writing_time_per_song_mins : ℕ) : ℕ :=
  let total_studio_time_mins := studio_time_hours * 60
  let total_writing_time_mins := total_songs * writing_time_per_song_mins
  let total_recording_time_mins := total_studio_time_mins - total_writing_time_mins - editing_time_mins
  total_recording_time_mins / total_songs

theorem elvis_recording_time :
  time_to_record_each_song 10 5 30 15 = 12 := by
  sorry

end NUMINAMATH_CALUDE_elvis_recording_time_l2053_205300


namespace NUMINAMATH_CALUDE_remaining_payment_l2053_205380

/-- Given a product with a 5% deposit of $50, prove that the remaining amount to be paid is $950 -/
theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (total_price : ℝ) : 
  deposit = 50 ∧ 
  deposit_percentage = 0.05 ∧ 
  deposit = deposit_percentage * total_price → 
  total_price - deposit = 950 := by
  sorry

end NUMINAMATH_CALUDE_remaining_payment_l2053_205380


namespace NUMINAMATH_CALUDE_race_win_probability_l2053_205326

theorem race_win_probability (pA pB pC pD pE : ℚ) 
  (hA : pA = 1/8) (hB : pB = 1/12) (hC : pC = 1/15) (hD : pD = 1/18) (hE : pE = 1/20)
  (h_mutually_exclusive : ∀ (x y : Fin 5), x ≠ y → pA + pB + pC + pD + pE ≤ 1) :
  pA + pB + pC + pD + pE = 137/360 := by
sorry

end NUMINAMATH_CALUDE_race_win_probability_l2053_205326


namespace NUMINAMATH_CALUDE_rainwater_farm_l2053_205381

theorem rainwater_farm (cows goats chickens : ℕ) : 
  cows = 9 →
  goats = 4 * cows →
  goats = 2 * chickens →
  chickens = 18 := by
sorry

end NUMINAMATH_CALUDE_rainwater_farm_l2053_205381


namespace NUMINAMATH_CALUDE_part_one_part_two_l2053_205386

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem part_one (a : ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 4 → f a x ≤ 2) : a = 2 := by
  sorry

-- Part 2
theorem part_two (a : ℝ) (h : 0 ≤ a ∧ a ≤ 3) :
  ∀ x : ℝ, f a (x + a) + f a (x - a) ≥ f a (a * x) - a * f a x := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2053_205386


namespace NUMINAMATH_CALUDE_mrs_sheridan_cats_l2053_205392

theorem mrs_sheridan_cats (initial_cats : ℕ) : 
  initial_cats + 14 = 31 → initial_cats = 17 := by
  sorry

end NUMINAMATH_CALUDE_mrs_sheridan_cats_l2053_205392


namespace NUMINAMATH_CALUDE_wicket_keeper_age_difference_l2053_205320

theorem wicket_keeper_age_difference (team_size : ℕ) (captain_age : ℕ) (team_avg_age : ℕ) 
  (h1 : team_size = 11)
  (h2 : captain_age = 28)
  (h3 : team_avg_age = 25)
  (h4 : (team_size * team_avg_age - captain_age - wicket_keeper_age) / (team_size - 2) = team_avg_age - 1) :
  wicket_keeper_age - captain_age = 3 :=
by sorry

end NUMINAMATH_CALUDE_wicket_keeper_age_difference_l2053_205320


namespace NUMINAMATH_CALUDE_product_divisible_by_504_l2053_205314

theorem product_divisible_by_504 (a : ℤ) : 504 ∣ ((a^3 - 1) * a^3 * (a^3 + 1)) := by
  sorry

#check product_divisible_by_504

end NUMINAMATH_CALUDE_product_divisible_by_504_l2053_205314


namespace NUMINAMATH_CALUDE_smallest_class_size_l2053_205373

theorem smallest_class_size (n : ℕ) : 
  (∃ (m : ℕ), 4 * n + (n + 1) = m ∧ m > 40) → 
  (∀ (k : ℕ), k < n → ¬(∃ (m : ℕ), 4 * k + (k + 1) = m ∧ m > 40)) → 
  4 * n + (n + 1) = 41 :=
sorry

end NUMINAMATH_CALUDE_smallest_class_size_l2053_205373


namespace NUMINAMATH_CALUDE_binary_addition_theorem_l2053_205371

/-- Represents a binary number as a list of bits (0 or 1) in little-endian order -/
def BinaryNumber := List Bool

/-- Converts a decimal number to its binary representation -/
def decimalToBinary (n : Int) : BinaryNumber :=
  sorry

/-- Converts a binary number to its decimal representation -/
def binaryToDecimal (b : BinaryNumber) : Int :=
  sorry

/-- Adds two binary numbers -/
def addBinary (a b : BinaryNumber) : BinaryNumber :=
  sorry

/-- Negates a binary number (two's complement) -/
def negateBinary (b : BinaryNumber) : BinaryNumber :=
  sorry

theorem binary_addition_theorem :
  let b1 := decimalToBinary 13  -- 1101₂
  let b2 := decimalToBinary 10  -- 1010₂
  let b3 := decimalToBinary 7   -- 111₂
  let b4 := negateBinary (decimalToBinary 11)  -- -1011₂
  let sum := addBinary b1 (addBinary b2 (addBinary b3 b4))
  binaryToDecimal sum = 35  -- 100011₂
  := by sorry

end NUMINAMATH_CALUDE_binary_addition_theorem_l2053_205371


namespace NUMINAMATH_CALUDE_mary_anne_sparkling_water_l2053_205375

-- Define the cost per bottle
def cost_per_bottle : ℚ := 2

-- Define the total spent per year
def total_spent_per_year : ℚ := 146

-- Define the number of days in a year
def days_per_year : ℕ := 365

-- Define the fraction of a bottle drunk each night
def fraction_per_night : ℚ := 1 / 5

-- Theorem statement
theorem mary_anne_sparkling_water :
  fraction_per_night * (days_per_year : ℚ) = total_spent_per_year / cost_per_bottle :=
sorry

end NUMINAMATH_CALUDE_mary_anne_sparkling_water_l2053_205375


namespace NUMINAMATH_CALUDE_range_of_positive_f_l2053_205388

/-- A function is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem range_of_positive_f 
  (f : ℝ → ℝ) 
  (f' : ℝ → ℝ) 
  (hf_odd : OddFunction f)
  (hf_deriv : ∀ x, HasDerivAt f (f' x) x)
  (hf_neg_one : f (-1) = 0)
  (hf_pos : ∀ x > 0, x * f' x - f x > 0) :
  {x | f x > 0} = Set.Ioo (-1) 0 ∪ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_positive_f_l2053_205388


namespace NUMINAMATH_CALUDE_expected_percentage_proof_l2053_205376

/-- The probability of rain for a county on Monday -/
def prob_rain_monday : ℝ := 0.70

/-- The probability of rain for a county on Tuesday -/
def prob_rain_tuesday : ℝ := 0.80

/-- The probability of rain for a county on Wednesday -/
def prob_rain_wednesday : ℝ := 0.60

/-- The proportion of counties with chance of rain on Monday -/
def prop_counties_monday : ℝ := 0.60

/-- The proportion of counties with chance of rain on Tuesday -/
def prop_counties_tuesday : ℝ := 0.55

/-- The proportion of counties with chance of rain on Wednesday -/
def prop_counties_wednesday : ℝ := 0.40

/-- The proportion of counties that received rain on at least one day -/
def prop_counties_with_rain : ℝ := 0.80

/-- The expected percentage of counties that will receive rain on all three days -/
def expected_percentage : ℝ :=
  prop_counties_monday * prob_rain_monday *
  prop_counties_tuesday * prob_rain_tuesday *
  prop_counties_wednesday * prob_rain_wednesday *
  prop_counties_with_rain

theorem expected_percentage_proof :
  expected_percentage = 0.60 * 0.70 * 0.55 * 0.80 * 0.40 * 0.60 * 0.80 :=
by sorry

end NUMINAMATH_CALUDE_expected_percentage_proof_l2053_205376


namespace NUMINAMATH_CALUDE_attendance_ratio_l2053_205368

/-- Proves that given the charges on three days and the average charge, 
    the ratio of attendance on these days is 4:1:5. -/
theorem attendance_ratio 
  (charge1 charge2 charge3 avg_charge : ℚ)
  (h1 : charge1 = 15)
  (h2 : charge2 = 15/2)
  (h3 : charge3 = 5/2)
  (h4 : avg_charge = 5)
  (x y z : ℚ) -- attendance on day 1, 2, and 3 respectively
  (h5 : (charge1 * x + charge2 * y + charge3 * z) / (x + y + z) = avg_charge) :
  ∃ (k : ℚ), k > 0 ∧ x = 4*k ∧ y = k ∧ z = 5*k := by
sorry


end NUMINAMATH_CALUDE_attendance_ratio_l2053_205368


namespace NUMINAMATH_CALUDE_other_items_percentage_correct_l2053_205345

/-- The percentage of money spent on other items in Jill's shopping trip -/
def other_items_percentage : ℝ := 
  let total := 100
  let clothing_percentage := 45
  let food_percentage := 45
  let clothing_tax_rate := 5
  let other_items_tax_rate := 10
  let total_tax_percentage := 3.25
  10

/-- Theorem stating that the percentage spent on other items is correct -/
theorem other_items_percentage_correct : 
  let total := 100
  let clothing_percentage := 45
  let food_percentage := 45
  let clothing_tax_rate := 5
  let other_items_tax_rate := 10
  let total_tax_percentage := 3.25
  (clothing_percentage + food_percentage + other_items_percentage = total) ∧
  (clothing_tax_rate * clothing_percentage / 100 + 
   other_items_tax_rate * other_items_percentage / 100 = total_tax_percentage) := by
  sorry

#check other_items_percentage_correct

end NUMINAMATH_CALUDE_other_items_percentage_correct_l2053_205345


namespace NUMINAMATH_CALUDE_partition_of_positive_integers_l2053_205312

def nth_prime (n : ℕ) : ℕ := sorry

def count_primes (n : ℕ) : ℕ := sorry

def set_A : Set ℕ := {m | ∃ n : ℕ, n > 0 ∧ m = n + nth_prime n - 1}

def set_B : Set ℕ := {m | ∃ n : ℕ, n > 0 ∧ m = n + count_primes n}

theorem partition_of_positive_integers : 
  ∀ m : ℕ, m > 0 → (m ∈ set_A ∧ m ∉ set_B) ∨ (m ∉ set_A ∧ m ∈ set_B) :=
sorry

end NUMINAMATH_CALUDE_partition_of_positive_integers_l2053_205312


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l2053_205384

theorem polynomial_multiplication :
  ∀ x : ℝ, (5 * x + 3) * (2 * x - 4 + x^2) = 5 * x^3 + 13 * x^2 - 14 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l2053_205384


namespace NUMINAMATH_CALUDE_max_value_problem_l2053_205303

theorem max_value_problem (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_sum : x + y + z = 2) :
  x^3 * y^2 * z^4 ≤ 13824 / 40353607 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), 0 < x₀ ∧ 0 < y₀ ∧ 0 < z₀ ∧ x₀ + y₀ + z₀ = 2 ∧ x₀^3 * y₀^2 * z₀^4 = 13824 / 40353607 :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_l2053_205303


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2053_205354

theorem sum_of_roots_quadratic (α β : ℝ) : 
  (∀ x : ℝ, x^2 + x - 2 = 0 ↔ x = α ∨ x = β) →
  α + β = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2053_205354


namespace NUMINAMATH_CALUDE_red_candles_count_l2053_205349

/-- Given the ratio of red candles to blue candles and the number of blue candles,
    calculate the number of red candles. -/
theorem red_candles_count (blue_candles : ℕ) (ratio_red : ℕ) (ratio_blue : ℕ) 
    (h1 : blue_candles = 27) 
    (h2 : ratio_red = 5) 
    (h3 : ratio_blue = 3) : ℕ :=
  45

#check red_candles_count

end NUMINAMATH_CALUDE_red_candles_count_l2053_205349


namespace NUMINAMATH_CALUDE_badminton_championship_probability_l2053_205321

theorem badminton_championship_probability 
  (p : ℝ) 
  (h1 : p = 1 / 3) 
  (h2 : 0 ≤ p) 
  (h3 : p ≤ 1) : 
  p * p * p = 1 / 27 := by
sorry

end NUMINAMATH_CALUDE_badminton_championship_probability_l2053_205321


namespace NUMINAMATH_CALUDE_solar_system_median_moons_l2053_205309

/-- Represents the number of moons for each planet in the solar system -/
def moon_counts : List Nat := [0, 1, 1, 3, 3, 6, 8, 14, 18, 21]

/-- Calculates the median of a sorted list with an even number of elements -/
def median (l : List Nat) : Rat :=
  let n := l.length
  if n % 2 = 0 then
    let mid := n / 2
    (l.get! (mid - 1) + l.get! mid) / 2
  else
    l.get! (n / 2)

theorem solar_system_median_moons :
  median moon_counts = 4.5 := by sorry

end NUMINAMATH_CALUDE_solar_system_median_moons_l2053_205309


namespace NUMINAMATH_CALUDE_razorback_tshirt_sales_l2053_205340

theorem razorback_tshirt_sales 
  (revenue_per_tshirt : ℕ) 
  (total_tshirts : ℕ) 
  (revenue_one_game : ℕ) 
  (h1 : revenue_per_tshirt = 98)
  (h2 : total_tshirts = 163)
  (h3 : revenue_one_game = 8722) :
  ∃ (arkansas_tshirts : ℕ), 
    arkansas_tshirts * revenue_per_tshirt = revenue_one_game ∧
    arkansas_tshirts ≤ total_tshirts ∧
    arkansas_tshirts = 89 :=
by sorry

end NUMINAMATH_CALUDE_razorback_tshirt_sales_l2053_205340


namespace NUMINAMATH_CALUDE_surface_area_specific_parallelepiped_l2053_205307

/-- The surface area of a rectangular parallelepiped with given face areas -/
def surface_area_parallelepiped (a b c : ℝ) : ℝ :=
  2 * (a + b + c)

/-- Theorem: The surface area of a rectangular parallelepiped with face areas 4, 3, and 6 is 26 -/
theorem surface_area_specific_parallelepiped :
  surface_area_parallelepiped 4 3 6 = 26 := by
  sorry

#check surface_area_specific_parallelepiped

end NUMINAMATH_CALUDE_surface_area_specific_parallelepiped_l2053_205307


namespace NUMINAMATH_CALUDE_card_problem_l2053_205319

theorem card_problem (x y : ℕ) : 
  x - 1 = y + 1 → 
  x + 1 = 2 * (y - 1) → 
  x + y = 12 := by
sorry

end NUMINAMATH_CALUDE_card_problem_l2053_205319


namespace NUMINAMATH_CALUDE_four_number_sequence_l2053_205357

theorem four_number_sequence (x y z t : ℝ) : 
  (y - x = z - y) →  -- arithmetic sequence condition
  (z^2 = y * t) →    -- geometric sequence condition
  (x + t = 37) → 
  (y + z = 36) → 
  (x = 12 ∧ y = 16 ∧ z = 20 ∧ t = 25) := by
sorry

end NUMINAMATH_CALUDE_four_number_sequence_l2053_205357


namespace NUMINAMATH_CALUDE_philippe_can_win_l2053_205338

/-- Represents a game state with cards remaining and sums for each player -/
structure GameState :=
  (remaining : Finset Nat)
  (philippe_sum : Nat)
  (emmanuel_sum : Nat)

/-- The initial game state -/
def initial_state : GameState :=
  { remaining := Finset.range 2018,
    philippe_sum := 0,
    emmanuel_sum := 0 }

/-- A strategy is a function that selects a card from the remaining set -/
def Strategy := (GameState → Nat)

/-- Applies a strategy to a game state, returning the new state -/
def apply_strategy (s : Strategy) (g : GameState) : GameState :=
  let card := s g
  { remaining := g.remaining.erase card,
    philippe_sum := g.philippe_sum + card,
    emmanuel_sum := g.emmanuel_sum }

/-- Plays the game to completion using the given strategies -/
def play_game (philippe_strategy : Strategy) (emmanuel_strategy : Strategy) : GameState :=
  sorry

/-- Theorem stating that Philippe can always win -/
theorem philippe_can_win :
  ∃ (philippe_strategy : Strategy),
    ∀ (emmanuel_strategy : Strategy),
      let final_state := play_game philippe_strategy emmanuel_strategy
      Even final_state.philippe_sum ∧ Odd final_state.emmanuel_sum :=
sorry

end NUMINAMATH_CALUDE_philippe_can_win_l2053_205338


namespace NUMINAMATH_CALUDE_yolanda_bike_speed_yolanda_speed_equals_husband_speed_l2053_205315

/-- Yolanda's bike ride problem -/
theorem yolanda_bike_speed (husband_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  husband_speed > 0 ∧ head_start > 0 ∧ catch_up_time > 0 →
  ∃ (bike_speed : ℝ),
    bike_speed > 0 ∧
    bike_speed * (head_start + catch_up_time) = husband_speed * catch_up_time :=
by
  sorry

/-- Yolanda's bike speed is equal to her husband's car speed -/
theorem yolanda_speed_equals_husband_speed :
  ∃ (bike_speed : ℝ),
    bike_speed > 0 ∧
    bike_speed = 40 ∧
    bike_speed * (15/60 + 15/60) = 40 * (15/60) :=
by
  sorry

end NUMINAMATH_CALUDE_yolanda_bike_speed_yolanda_speed_equals_husband_speed_l2053_205315


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2053_205330

theorem arithmetic_mean_of_fractions : 
  (1 / 3 : ℚ) * ((3 / 7 : ℚ) + (5 / 9 : ℚ) + (2 / 3 : ℚ)) = 104 / 189 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2053_205330


namespace NUMINAMATH_CALUDE_redWhiteJellyBeansCount_l2053_205308

/-- Represents the number of jelly beans of each color in one bag -/
structure JellyBeanBag where
  red : ℕ
  black : ℕ
  green : ℕ
  purple : ℕ
  yellow : ℕ
  white : ℕ

/-- Calculates the total number of red and white jelly beans in the fishbowl -/
def totalRedWhiteInFishbowl (bag : JellyBeanBag) (bagsToFill : ℕ) : ℕ :=
  (bag.red + bag.white) * bagsToFill

/-- Theorem: The total number of red and white jelly beans in the fishbowl is 126 -/
theorem redWhiteJellyBeansCount : 
  let bag : JellyBeanBag := {
    red := 24,
    black := 13,
    green := 36,
    purple := 28,
    yellow := 32,
    white := 18
  }
  let bagsToFill : ℕ := 3
  totalRedWhiteInFishbowl bag bagsToFill = 126 := by
  sorry


end NUMINAMATH_CALUDE_redWhiteJellyBeansCount_l2053_205308


namespace NUMINAMATH_CALUDE_ironman_age_l2053_205377

/-- Represents the ages of the characters in the problem -/
structure Ages where
  thor : ℕ
  captainAmerica : ℕ
  peterParker : ℕ
  ironman : ℕ

/-- The conditions of the problem -/
def problemConditions (ages : Ages) : Prop :=
  ages.thor = 13 * ages.captainAmerica ∧
  ages.captainAmerica = 7 * ages.peterParker ∧
  ages.ironman = ages.peterParker + 32 ∧
  ages.thor = 1456

/-- The theorem to be proved -/
theorem ironman_age (ages : Ages) :
  problemConditions ages → ages.ironman = 48 := by
  sorry

end NUMINAMATH_CALUDE_ironman_age_l2053_205377


namespace NUMINAMATH_CALUDE_range_of_a_l2053_205323

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (x - a + Real.log (x / a)) * (-2 * x^2 + a * x + 10) ≤ 0) → 
  a = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2053_205323


namespace NUMINAMATH_CALUDE_compound_interest_problem_l2053_205391

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Total amount calculation -/
def total_amount (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

/-- Theorem statement -/
theorem compound_interest_problem (P : ℝ) :
  compound_interest P 0.04 2 = 326.40 →
  total_amount P 326.40 = 4326.40 := by
sorry

#eval compound_interest 4000 0.04 2
#eval total_amount 4000 326.40

end NUMINAMATH_CALUDE_compound_interest_problem_l2053_205391


namespace NUMINAMATH_CALUDE_lucky_sum_equality_l2053_205317

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to select k distinct numbers from 1 to n with sum s -/
def sumCombinations (n k s : ℕ) : ℕ := sorry

/-- The probability of selecting k balls from n balls with sum s -/
def probability (n k s : ℕ) : ℚ :=
  (sumCombinations n k s : ℚ) / (choose n k : ℚ)

theorem lucky_sum_equality (N : ℕ) :
  probability N 10 63 = probability N 8 44 ↔ N = 18 := by
  sorry

end NUMINAMATH_CALUDE_lucky_sum_equality_l2053_205317


namespace NUMINAMATH_CALUDE_chord_ratio_constant_l2053_205325

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define a chord of the ellipse
def chord (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2

-- Define parallel lines
def parallel (A B M N : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (N.1 - M.1) = (B.1 - A.1) * (N.2 - M.2)

-- Define a point on a line passing through the origin
def through_origin (M N : ℝ × ℝ) : Prop :=
  M.2 * N.1 = M.1 * N.2

-- Main theorem
theorem chord_ratio_constant
  (A B M N : ℝ × ℝ)
  (h_AB : chord A B)
  (h_MN : chord M N)
  (h_parallel : parallel A B M N)
  (h_origin : through_origin M N) :
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 1/4 * ((N.1 - M.1)^2 + (N.2 - M.2)^2)^2 :=
sorry

end NUMINAMATH_CALUDE_chord_ratio_constant_l2053_205325


namespace NUMINAMATH_CALUDE_work_completion_time_l2053_205397

/-- The time taken for A, B, and C to complete the work together -/
def time_together (time_A time_B time_C : ℚ) : ℚ :=
  1 / (1 / time_A + 1 / time_B + 1 / time_C)

/-- Theorem stating that A, B, and C can complete the work together in 2 days -/
theorem work_completion_time :
  time_together 4 6 12 = 2 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2053_205397


namespace NUMINAMATH_CALUDE_tan_alpha_eq_two_implies_fraction_eq_negative_two_l2053_205316

theorem tan_alpha_eq_two_implies_fraction_eq_negative_two (α : Real) 
  (h : Real.tan α = 2) : 
  (2 * Real.sin α - 2 * Real.cos α) / (4 * Real.sin α - 9 * Real.cos α) = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_eq_two_implies_fraction_eq_negative_two_l2053_205316


namespace NUMINAMATH_CALUDE_two_digit_powers_of_three_l2053_205305

theorem two_digit_powers_of_three : 
  ∃! (count : ℕ), ∃ (S : Finset ℕ), 
    (∀ n ∈ S, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ 
    (∀ n ∉ S, 3^n < 10 ∨ 99 < 3^n) ∧ 
    Finset.card S = count ∧
    count = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_powers_of_three_l2053_205305


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l2053_205351

theorem consecutive_even_integers_sum (y : ℤ) : 
  y % 2 = 0 ∧ 
  (y + 2) % 2 = 0 ∧ 
  y = 2 * (y + 2) → 
  y + (y + 2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l2053_205351


namespace NUMINAMATH_CALUDE_total_faces_painted_l2053_205366

/-- The number of cuboids painted by Ezekiel -/
def num_cuboids : ℕ := 5

/-- The number of faces on each cuboid -/
def faces_per_cuboid : ℕ := 6

/-- Theorem stating the total number of faces painted by Ezekiel -/
theorem total_faces_painted :
  num_cuboids * faces_per_cuboid = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_faces_painted_l2053_205366


namespace NUMINAMATH_CALUDE_money_conditions_l2053_205356

theorem money_conditions (a b : ℝ) 
  (h1 : b - 4*a < 78)
  (h2 : 6*a - b = 36)
  : a < 57 ∧ b > -36 := by
  sorry

end NUMINAMATH_CALUDE_money_conditions_l2053_205356


namespace NUMINAMATH_CALUDE_product_set_sum_l2053_205383

theorem product_set_sum (a₁ a₂ a₃ a₄ : ℚ) :
  ({a₁ * a₂, a₁ * a₃, a₁ * a₄, a₂ * a₃, a₂ * a₄, a₃ * a₄} : Finset ℚ) =
  {-24, -2, -3/2, -1/8, 1, 3} →
  (a₁ + a₂ + a₃ + a₄ = 9/4) ∨ (a₁ + a₂ + a₃ + a₄ = -9/4) := by
  sorry

end NUMINAMATH_CALUDE_product_set_sum_l2053_205383


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l2053_205363

theorem min_value_squared_sum (a b t s : ℝ) (h1 : a + b = t) (h2 : a - b = s) :
  a^2 + b^2 = (t^2 + s^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l2053_205363


namespace NUMINAMATH_CALUDE_jenny_chocolate_squares_count_l2053_205332

/-- The number of chocolate squares Jenny ate -/
def jenny_chocolate_squares (mike_chocolate_squares : ℕ) : ℕ :=
  3 * mike_chocolate_squares + 5

/-- The number of candies Mike's friend ate -/
def mikes_friend_candies (mike_candies : ℕ) : ℕ :=
  mike_candies - 10

/-- The number of candies Jenny ate -/
def jenny_candies (mikes_friend_candies : ℕ) : ℕ :=
  2 * mikes_friend_candies

theorem jenny_chocolate_squares_count 
  (mike_chocolate_squares : ℕ) 
  (mike_candies : ℕ) 
  (h1 : mike_chocolate_squares = 20) 
  (h2 : mike_candies = 20) :
  jenny_chocolate_squares mike_chocolate_squares = 65 := by
  sorry

#check jenny_chocolate_squares_count

end NUMINAMATH_CALUDE_jenny_chocolate_squares_count_l2053_205332


namespace NUMINAMATH_CALUDE_marble_probability_l2053_205318

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) 
  (h_total : total = 90)
  (h_white : p_white = 1 / 6)
  (h_green : p_green = 1 / 5) :
  1 - (p_white + p_green) = 19 / 30 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l2053_205318


namespace NUMINAMATH_CALUDE_equation_solution_l2053_205362

theorem equation_solution (x y : ℚ) : 
  (0.009 / x = 0.01 / y) → (x + y = 50) → (x = 450 / 19 ∧ y = 500 / 19) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2053_205362


namespace NUMINAMATH_CALUDE_min_soldiers_to_add_l2053_205343

theorem min_soldiers_to_add (N : ℕ) : 
  N % 7 = 2 → N % 12 = 2 → (84 - N % 84) = 82 := by
  sorry

end NUMINAMATH_CALUDE_min_soldiers_to_add_l2053_205343


namespace NUMINAMATH_CALUDE_expression_equals_x_power_44_l2053_205339

def numerator_sequence (n : ℕ) : ℕ := 2 * n + 1

def denominator_sequence (n : ℕ) : ℕ := 4 * n

def numerator_sum (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ i => numerator_sequence (i + 1))

def denominator_sum (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ i => denominator_sequence (i + 1))

theorem expression_equals_x_power_44 (x : ℝ) (hx : x = 3) :
  (x ^ numerator_sum 14) / (x ^ denominator_sum 9) = x ^ 44 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_x_power_44_l2053_205339


namespace NUMINAMATH_CALUDE_second_month_sale_l2053_205369

def sale_first : ℕ := 6435
def sale_third : ℕ := 6855
def sale_fourth : ℕ := 7230
def sale_fifth : ℕ := 6562
def sale_sixth : ℕ := 7391
def average_sale : ℕ := 6900
def num_months : ℕ := 6

theorem second_month_sale :
  sale_first + sale_third + sale_fourth + sale_fifth + sale_sixth +
  (average_sale * num_months - (sale_first + sale_third + sale_fourth + sale_fifth + sale_sixth)) = 
  average_sale * num_months :=
by sorry

end NUMINAMATH_CALUDE_second_month_sale_l2053_205369


namespace NUMINAMATH_CALUDE_girls_without_notebooks_l2053_205304

theorem girls_without_notebooks (total_girls : Nat) (students_with_notebooks : Nat) (boys_with_notebooks : Nat) 
  (h1 : total_girls = 20)
  (h2 : students_with_notebooks = 25)
  (h3 : boys_with_notebooks = 16) :
  total_girls - (students_with_notebooks - boys_with_notebooks) = 11 := by
  sorry

end NUMINAMATH_CALUDE_girls_without_notebooks_l2053_205304


namespace NUMINAMATH_CALUDE_valid_domains_for_range_l2053_205324

def f (x : ℝ) := x^2 - 2*x + 2

theorem valid_domains_for_range (a b : ℝ) (h : a < b) :
  (∀ x ∈ Set.Icc a b, 1 ≤ f x ∧ f x ≤ 2) →
  (∀ y ∈ Set.Icc 1 2, ∃ x ∈ Set.Icc a b, f x = y) →
  (a = 0 ∧ b = 1) ∨ (a = 1/4 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_valid_domains_for_range_l2053_205324


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l2053_205370

theorem trigonometric_expression_equality : 
  4 * Real.sin (80 * π / 180) - Real.cos (10 * π / 180) / Real.sin (10 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l2053_205370
