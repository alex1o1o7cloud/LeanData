import Mathlib

namespace NUMINAMATH_GPT_min_value_of_expression_ge_9_l2169_216996

theorem min_value_of_expression_ge_9 
    (x : ℝ)
    (h1 : -2 < x ∧ x < -1)
    (m n : ℝ)
    (a b : ℝ)
    (ha : a = -2)
    (hb : b = -1)
    (h2 : mn > 0)
    (h3 : m * a + n * b + 1 = 0) :
    (2 / m) + (1 / n) ≥ 9 := by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_ge_9_l2169_216996


namespace NUMINAMATH_GPT_ratio_A_to_B_l2169_216967

theorem ratio_A_to_B (total_weight_X : ℕ) (weight_B : ℕ) (weight_A : ℕ) (h₁ : total_weight_X = 324) (h₂ : weight_B = 270) (h₃ : weight_A = total_weight_X - weight_B):
  weight_A / gcd weight_A weight_B = 1 ∧ weight_B / gcd weight_A weight_B = 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_A_to_B_l2169_216967


namespace NUMINAMATH_GPT_mark_spends_47_l2169_216961

def apple_price : ℕ := 2
def apple_quantity : ℕ := 4
def bread_price : ℕ := 3
def bread_quantity : ℕ := 5
def cheese_price : ℕ := 6
def cheese_quantity : ℕ := 3
def cereal_price : ℕ := 5
def cereal_quantity : ℕ := 4
def coupon : ℕ := 10

def calculate_total_cost (apple_price apple_quantity bread_price bread_quantity cheese_price cheese_quantity cereal_price cereal_quantity coupon : ℕ) : ℕ :=
  let apples_cost := apple_price * (apple_quantity / 2)  -- Apply buy-one-get-one-free
  let bread_cost := bread_price * bread_quantity
  let cheese_cost := cheese_price * cheese_quantity
  let cereal_cost := cereal_price * cereal_quantity
  let subtotal := apples_cost + bread_cost + cheese_cost + cereal_cost
  let total_cost := if subtotal > 50 then subtotal - coupon else subtotal
  total_cost

theorem mark_spends_47 : calculate_total_cost apple_price apple_quantity bread_price bread_quantity cheese_price cheese_quantity cereal_price cereal_quantity coupon = 47 :=
  sorry

end NUMINAMATH_GPT_mark_spends_47_l2169_216961


namespace NUMINAMATH_GPT_train_length_and_speed_l2169_216992

theorem train_length_and_speed (L_bridge : ℕ) (t_cross : ℕ) (t_on_bridge : ℕ) (L_train : ℕ) (v_train : ℕ)
  (h_bridge : L_bridge = 1000)
  (h_t_cross : t_cross = 60)
  (h_t_on_bridge : t_on_bridge = 40)
  (h_crossing_eq : (L_bridge + L_train) / t_cross = v_train)
  (h_on_bridge_eq : L_bridge / t_on_bridge = v_train) : 
  L_train = 200 ∧ v_train = 20 := 
  by
  sorry

end NUMINAMATH_GPT_train_length_and_speed_l2169_216992


namespace NUMINAMATH_GPT_find_threedigit_number_l2169_216947

-- Define the three-digit number and its reverse
def original_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def reversed_number (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

-- Define the condition of adding the number and its reverse to get 1777
def number_sum_condition (a b c : ℕ) : Prop :=
  original_number a b c + reversed_number a b c = 1777

-- Prove the existence of digits a, b, and c that satisfy the conditions
theorem find_threedigit_number :
  ∃ a b c : ℕ, a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  original_number a b c = 859 ∧ 
  reversed_number a b c = 958 ∧ 
  number_sum_condition a b c :=
sorry

end NUMINAMATH_GPT_find_threedigit_number_l2169_216947


namespace NUMINAMATH_GPT_filling_rate_in_cubic_meters_per_hour_l2169_216927

def barrels_per_minute_filling_rate : ℝ := 3
def liters_per_barrel : ℝ := 159
def liters_per_cubic_meter : ℝ := 1000
def minutes_per_hour : ℝ := 60

theorem filling_rate_in_cubic_meters_per_hour :
  (barrels_per_minute_filling_rate * liters_per_barrel / liters_per_cubic_meter * minutes_per_hour) = 28.62 :=
sorry

end NUMINAMATH_GPT_filling_rate_in_cubic_meters_per_hour_l2169_216927


namespace NUMINAMATH_GPT_Maria_selling_price_l2169_216940

-- Define the constants based on the given conditions
def brush_cost : ℕ := 20
def canvas_cost : ℕ := 3 * brush_cost
def paint_cost_per_liter : ℕ := 8
def paint_needed : ℕ := 5
def earnings : ℕ := 80

-- Calculate the total cost and the selling price
def total_cost : ℕ := brush_cost + canvas_cost + (paint_cost_per_liter * paint_needed)
def selling_price : ℕ := total_cost + earnings

-- Proof statement
theorem Maria_selling_price : selling_price = 200 := by
  sorry

end NUMINAMATH_GPT_Maria_selling_price_l2169_216940


namespace NUMINAMATH_GPT_bad_carrots_eq_13_l2169_216936

-- Define the number of carrots picked by Haley
def haley_picked : ℕ := 39

-- Define the number of carrots picked by her mom
def mom_picked : ℕ := 38

-- Define the number of good carrots
def good_carrots : ℕ := 64

-- Define the total number of carrots picked
def total_carrots : ℕ := haley_picked + mom_picked

-- State the theorem to prove the number of bad carrots
theorem bad_carrots_eq_13 : total_carrots - good_carrots = 13 := by
  sorry

end NUMINAMATH_GPT_bad_carrots_eq_13_l2169_216936


namespace NUMINAMATH_GPT_lecture_room_configuration_l2169_216989

theorem lecture_room_configuration (m n : ℕ) (boys_per_row girls_per_column unoccupied_chairs : ℕ) :
    boys_per_row = 6 →
    girls_per_column = 8 →
    unoccupied_chairs = 15 →
    (m * n = boys_per_row * m + girls_per_column * n + unoccupied_chairs) →
    (m = 71 ∧ n = 7) ∨
    (m = 29 ∧ n = 9) ∨
    (m = 17 ∧ n = 13) ∨
    (m = 15 ∧ n = 15) ∨
    (m = 11 ∧ n = 27) ∨
    (m = 9 ∧ n = 69) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_lecture_room_configuration_l2169_216989


namespace NUMINAMATH_GPT_find_constant_b_l2169_216922

variable (x : ℝ)
variable (b d e : ℝ)

theorem find_constant_b   
  (h1 : (7 * x ^ 2 - 2 * x + 4 / 3) * (d * x ^ 2 + b * x + e) = 28 * x ^ 4 - 10 * x ^ 3 + 18 * x ^ 2 - 8 * x + 5 / 3)
  (h2 : d = 4) : 
  b = -2 / 7 := 
sorry

end NUMINAMATH_GPT_find_constant_b_l2169_216922


namespace NUMINAMATH_GPT_work_completion_time_l2169_216903

-- Define the work rates of A, B, and C
def work_rate_A : ℚ := 1 / 6
def work_rate_B : ℚ := 1 / 6
def work_rate_C : ℚ := 1 / 6

-- Define the combined work rate
def combined_work_rate : ℚ := work_rate_A + work_rate_B + work_rate_C

-- Define the total work to be done (1 represents the whole job)
def total_work : ℚ := 1

-- Calculate the number of days to complete the work together
def days_to_complete_work : ℚ := total_work / combined_work_rate

theorem work_completion_time :
  work_rate_A = 1 / 6 ∧
  work_rate_B = 1 / 6 ∧
  work_rate_C = 1 / 6 →
  combined_work_rate = (work_rate_A + work_rate_B + work_rate_C) →
  days_to_complete_work = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_work_completion_time_l2169_216903


namespace NUMINAMATH_GPT_percent_of_pizza_not_crust_l2169_216938

theorem percent_of_pizza_not_crust (total_weight crust_weight : ℝ) (h_total : total_weight = 800) (h_crust : crust_weight = 200) :
  (total_weight - crust_weight) / total_weight * 100 = 75 :=
by
  sorry

end NUMINAMATH_GPT_percent_of_pizza_not_crust_l2169_216938


namespace NUMINAMATH_GPT_exponents_subtraction_l2169_216928

theorem exponents_subtraction (m n : ℕ) (hm : 3 ^ m = 8) (hn : 3 ^ n = 2) : 3 ^ (m - n) = 4 := 
by
  sorry

end NUMINAMATH_GPT_exponents_subtraction_l2169_216928


namespace NUMINAMATH_GPT_tan2α_sin_β_l2169_216942

open Real

variables {α β : ℝ}

axiom α_acute : 0 < α ∧ α < π / 2
axiom β_acute : 0 < β ∧ β < π / 2
axiom sin_α : sin α = 4 / 5
axiom cos_alpha_beta : cos (α + β) = 5 / 13

theorem tan2α : tan 2 * α = -24 / 7 :=
by sorry

theorem sin_β : sin β = 16 / 65 :=
by sorry

end NUMINAMATH_GPT_tan2α_sin_β_l2169_216942


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2169_216978

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) -- a_n is a sequence of real numbers
  (q : ℝ) -- q is the common ratio
  (h1 : a 1 + a 2 = 20) -- first condition
  (h2 : a 3 + a 4 = 80) -- second condition
  (h_geom : ∀ n, a (n + 1) = a n * q) -- property of geometric sequence
  : a 5 + a 6 = 320 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l2169_216978


namespace NUMINAMATH_GPT_inequality_solution_l2169_216995

theorem inequality_solution (x : ℝ) (h_pos : 0 < x) :
  (3 / 8 + |x - 14 / 24| < 8 / 12) ↔ x ∈ Set.Ioo (7 / 24) (7 / 8) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2169_216995


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_l2169_216987

-- Problem 1
theorem problem_1 (x y : ℝ) : 
  -4 * x^2 * y * (x * y - 5 * y^2 - 1) = -4 * x^3 * y^2 + 20 * x^2 * y^3 + 4 * x^2 * y :=
by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) :
  (-3 * a)^2 - (2 * a + 1) * (a - 2) = 7 * a^2 + 3 * a + 2 :=
by
  sorry

-- Problem 3
theorem problem_3 (x y : ℝ) :
  (-2 * x - 3 * y) * (3 * y - 2 * x) - (2 * x - 3 * y)^2 = 12 * x * y - 18 * y^2 :=
by
  sorry

-- Problem 4
theorem problem_4 : 2010^2 - 2011 * 2009 = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_l2169_216987


namespace NUMINAMATH_GPT_hot_dogs_served_today_l2169_216920

theorem hot_dogs_served_today : 9 + 2 = 11 :=
by
  sorry

end NUMINAMATH_GPT_hot_dogs_served_today_l2169_216920


namespace NUMINAMATH_GPT_Cody_reads_books_in_7_weeks_l2169_216909

noncomputable def CodyReadsBooks : ℕ :=
  let total_books := 54
  let first_week_books := 6
  let second_week_books := 3
  let book_per_week := 9
  let remaining_books := total_books - first_week_books - second_week_books
  let remaining_weeks := remaining_books / book_per_week
  let total_weeks := 1 + 1 + remaining_weeks
  total_weeks

theorem Cody_reads_books_in_7_weeks : CodyReadsBooks = 7 := by
  sorry

end NUMINAMATH_GPT_Cody_reads_books_in_7_weeks_l2169_216909


namespace NUMINAMATH_GPT_students_same_group_in_all_lessons_l2169_216965

theorem students_same_group_in_all_lessons (students : Fin 28 → Fin 3 × Fin 3 × Fin 3) :
  ∃ (i j : Fin 28), i ≠ j ∧ students i = students j :=
by
  sorry

end NUMINAMATH_GPT_students_same_group_in_all_lessons_l2169_216965


namespace NUMINAMATH_GPT_range_of_a_l2169_216988

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, abs (2 * x - 1) + abs (x + 2) ≥ a^2 + (1 / 2) * a + 2) →
  -1 ≤ a ∧ a ≤ (1 / 2) := by
sorry

end NUMINAMATH_GPT_range_of_a_l2169_216988


namespace NUMINAMATH_GPT_negation_of_proposition_l2169_216914

theorem negation_of_proposition :
  (¬ ∃ x_0 : ℝ, x_0^3 - x_0^2 + 1 ≥ 0) ↔ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l2169_216914


namespace NUMINAMATH_GPT_find_x2_times_x1_plus_x3_l2169_216991

noncomputable def a := Real.sqrt 2023
noncomputable def x1 := -Real.sqrt 7
noncomputable def x2 := 1 / a
noncomputable def x3 := Real.sqrt 7

theorem find_x2_times_x1_plus_x3 :
  let x1 := -Real.sqrt 7
  let x2 := 1 / Real.sqrt 2023
  let x3 := Real.sqrt 7
  x2 * (x1 + x3) = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_x2_times_x1_plus_x3_l2169_216991


namespace NUMINAMATH_GPT_sequence_limit_l2169_216932

noncomputable def sequence_converges (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n > 1 ∧ a (n + 1) ^ 2 ≥ a n * a (n + 2)

theorem sequence_limit (a : ℕ → ℝ) (h : sequence_converges a) : 
  ∃ l : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (Real.log (a (n + 1)) / Real.log (a n) - l) < ε := 
sorry

end NUMINAMATH_GPT_sequence_limit_l2169_216932


namespace NUMINAMATH_GPT_plot_area_is_nine_hectares_l2169_216923

-- Definition of the dimensions of the plot
def length := 450
def width := 200

-- Definition of conversion factor from square meters to hectares
def sqMetersPerHectare := 10000

-- Calculated area in hectares
def area_hectares := (length * width) / sqMetersPerHectare

-- Theorem statement: prove that the area in hectares is 9
theorem plot_area_is_nine_hectares : area_hectares = 9 := 
by
  sorry

end NUMINAMATH_GPT_plot_area_is_nine_hectares_l2169_216923


namespace NUMINAMATH_GPT_max_discriminant_l2169_216953

noncomputable def f (a b c x : ℤ) := a * x^2 + b * x + c

theorem max_discriminant (a b c u v w : ℤ)
  (h1 : u ≠ v) (h2 : v ≠ w) (h3 : u ≠ w)
  (hu : f a b c u = 0)
  (hv : f a b c v = 0)
  (hw : f a b c w = 2) :
  ∃ (a b c : ℤ), b^2 - 4 * a * c = 16 :=
sorry

end NUMINAMATH_GPT_max_discriminant_l2169_216953


namespace NUMINAMATH_GPT_Haley_has_25_necklaces_l2169_216946

theorem Haley_has_25_necklaces (J H Q : ℕ) 
  (h1 : H = J + 5) 
  (h2 : Q = J / 2) 
  (h3 : H = Q + 15) : 
  H = 25 := 
sorry

end NUMINAMATH_GPT_Haley_has_25_necklaces_l2169_216946


namespace NUMINAMATH_GPT_quadratic_function_properties_l2169_216910

-- Definitions based on given conditions
def quadraticFunction (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def pointCondition (a b c : ℝ) : Prop := quadraticFunction a b c (-2) = 0
def inequalityCondition (a b c : ℝ) : Prop := ∀ x : ℝ, 2 * x ≤ quadraticFunction a b c x ∧ quadraticFunction a b c x ≤ (1 / 2) * x^2 + 2
def strengthenCondition (f : ℝ → ℝ) (t : ℝ) : Prop := ∀ x, -1 ≤ x ∧ x ≤ 1 → f (x + t) < f (x / 3)

-- Our primary statement to prove
theorem quadratic_function_properties :
  ∃ a b c, pointCondition a b c ∧ inequalityCondition a b c ∧
           (a = 1 / 4 ∧ b = 1 ∧ c = 1) ∧
           (∀ t, (-8 / 3 < t ∧ t < -2 / 3) ↔ strengthenCondition (quadraticFunction (1 / 4) 1 1) t) :=
by sorry 

end NUMINAMATH_GPT_quadratic_function_properties_l2169_216910


namespace NUMINAMATH_GPT_abs_inequality_l2169_216907

theorem abs_inequality (x : ℝ) (h : |x - 2| < 1) : 1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_GPT_abs_inequality_l2169_216907


namespace NUMINAMATH_GPT_bella_earrings_l2169_216962

theorem bella_earrings (B M R : ℝ) 
  (h1 : B = 0.25 * M) 
  (h2 : M = 2 * R) 
  (h3 : B + M + R = 70) : 
  B = 10 := by 
  sorry

end NUMINAMATH_GPT_bella_earrings_l2169_216962


namespace NUMINAMATH_GPT_jackson_miles_l2169_216916

theorem jackson_miles (beka_miles jackson_miles : ℕ) (h1 : beka_miles = 873) (h2 : beka_miles = jackson_miles + 310) : jackson_miles = 563 := by
  sorry

end NUMINAMATH_GPT_jackson_miles_l2169_216916


namespace NUMINAMATH_GPT_johns_horses_l2169_216908

theorem johns_horses 
  (feeding_per_day : ℕ := 2) 
  (food_per_feeding : ℝ := 20) 
  (bag_weight : ℝ := 1000) 
  (num_bags : ℕ := 60) 
  (days : ℕ := 60)
  (total_food : ℝ := num_bags * bag_weight) 
  (daily_food_consumption : ℝ := total_food / days) 
  (food_per_horse_per_day : ℝ := food_per_feeding * feeding_per_day) :
  ∀ H : ℝ, (daily_food_consumption / food_per_horse_per_day = H) → H = 25 := 
by
  intros H hH
  sorry

end NUMINAMATH_GPT_johns_horses_l2169_216908


namespace NUMINAMATH_GPT_james_and_david_probability_l2169_216945

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem james_and_david_probability :
  let total_workers := 22
  let chosen_workers := 4
  let j_and_d_chosen := 2
  (choose 20 2) / (choose 22 4) = (2 / 231) :=
by
  sorry

end NUMINAMATH_GPT_james_and_david_probability_l2169_216945


namespace NUMINAMATH_GPT_polynomial_factorization_l2169_216911

noncomputable def polyExpression (a b c : ℕ) : ℕ := a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4

theorem polynomial_factorization (a b c : ℕ) :
  ∃ q : ℕ → ℕ → ℕ → ℕ, q a b c = (a + b + c)^3 - 3 * a * b * c ∧
  polyExpression a b c = (a - b) * (b - c) * (c - a) * q a b c := by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_polynomial_factorization_l2169_216911


namespace NUMINAMATH_GPT_constant_ratio_arithmetic_progressions_l2169_216950

theorem constant_ratio_arithmetic_progressions
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d p a1 b1 : ℝ)
  (h_a : ∀ k : ℕ, a (k + 1) = a1 + k * d)
  (h_b : ∀ k : ℕ, b (k + 1) = b1 + k * p)
  (h_pos : ∀ k : ℕ, a (k + 1) > 0 ∧ b (k + 1) > 0)
  (h_int : ∀ k : ℕ, ∃ n : ℤ, (a (k + 1) / b (k + 1)) = n) :
  ∃ r : ℝ, ∀ k : ℕ, (a (k + 1) / b (k + 1)) = r :=
by
  sorry

end NUMINAMATH_GPT_constant_ratio_arithmetic_progressions_l2169_216950


namespace NUMINAMATH_GPT_Mike_found_seashells_l2169_216930

/-!
# Problem:
Mike found some seashells on the beach, he gave Tom 49 of his seashells.
He has thirteen seashells left. How many seashells did Mike find on the beach?

# Conditions:
1. Mike gave Tom 49 seashells.
2. Mike has 13 seashells left.

# Proof statement:
Prove that Mike found 62 seashells on the beach.
-/

/-- Define the variables and conditions -/
def seashells_given_to_Tom : ℕ := 49
def seashells_left_with_Mike : ℕ := 13

/-- Prove that Mike found 62 seashells on the beach -/
theorem Mike_found_seashells : 
  seashells_given_to_Tom + seashells_left_with_Mike = 62 := 
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_Mike_found_seashells_l2169_216930


namespace NUMINAMATH_GPT_basketball_team_wins_l2169_216969

-- Define the known quantities
def games_won_initial : ℕ := 60
def games_total_initial : ℕ := 80
def games_left : ℕ := 50
def total_games : ℕ := games_total_initial + games_left
def desired_win_fraction : ℚ := 3 / 4

-- The main goal: Prove that the team must win 38 of the remaining 50 games to reach the desired win fraction
theorem basketball_team_wins :
  ∃ x : ℕ, x = 38 ∧ (games_won_initial + x : ℚ) / total_games = desired_win_fraction :=
by
  sorry

end NUMINAMATH_GPT_basketball_team_wins_l2169_216969


namespace NUMINAMATH_GPT_impossible_to_create_3_piles_l2169_216976

-- Defining similar piles
def similar (x y : ℝ) : Prop :=
  x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

-- Main theorem statement
theorem impossible_to_create_3_piles (initial_pile : ℝ) (h_initial : initial_pile > 0) :
  ∀ (x y z : ℝ), 
  x + y + z = initial_pile → 
  similar x y ∧ similar y z ∧ similar z x → 
  false := 
by 
  sorry

end NUMINAMATH_GPT_impossible_to_create_3_piles_l2169_216976


namespace NUMINAMATH_GPT_sum_of_all_different_possible_areas_of_cool_rectangles_l2169_216900

-- Define the concept of a cool rectangle
def is_cool_rectangle (a b : ℕ) : Prop :=
  a * b = 2 * (2 * a + 2 * b)

-- Define the function to calculate the area of a rectangle
def area (a b : ℕ) : ℕ := a * b

-- Define the set of pairs (a, b) that satisfy the cool rectangle condition
def cool_rectangle_pairs : List (ℕ × ℕ) :=
  [(5, 20), (6, 12), (8, 8)]

-- Calculate the sum of all different possible areas of cool rectangles
def sum_of_cool_rectangle_areas : ℕ :=
  List.sum (cool_rectangle_pairs.map (λ p => area p.fst p.snd))

-- Theorem statement
theorem sum_of_all_different_possible_areas_of_cool_rectangles :
  sum_of_cool_rectangle_areas = 236 :=
by
  -- This is where the proof would go based on the given solution.
  sorry

end NUMINAMATH_GPT_sum_of_all_different_possible_areas_of_cool_rectangles_l2169_216900


namespace NUMINAMATH_GPT_find_m_plus_n_l2169_216999

theorem find_m_plus_n (x : ℝ) (m n : ℕ) (h₁ : (1 + Real.sin x) / (Real.cos x) = 22 / 7) 
                      (h₂ : (1 + Real.cos x) / (Real.sin x) = m / n) :
                      m + n = 44 := by
  sorry

end NUMINAMATH_GPT_find_m_plus_n_l2169_216999


namespace NUMINAMATH_GPT_greatest_of_5_consec_even_numbers_l2169_216925

-- Definitions based on the conditions
def avg_of_5_consec_even_numbers (N : ℤ) : ℤ := (N - 4 + N - 2 + N + N + 2 + N + 4) / 5

-- Proof statement
theorem greatest_of_5_consec_even_numbers (N : ℤ) (h : avg_of_5_consec_even_numbers N = 35) : N + 4 = 39 :=
by
  sorry -- proof is omitted

end NUMINAMATH_GPT_greatest_of_5_consec_even_numbers_l2169_216925


namespace NUMINAMATH_GPT_billy_videos_within_limit_l2169_216944

def total_videos_watched_within_time_limit (time_limit : ℕ) (video_time : ℕ) (search_time : ℕ) (break_time : ℕ) (num_trials : ℕ) (videos_per_trial : ℕ) (categories : ℕ) (videos_per_category : ℕ) : ℕ :=
  let total_trial_time := videos_per_trial * video_time + search_time + break_time
  let total_category_time := videos_per_category * video_time
  let full_trial_time := num_trials * total_trial_time
  let full_category_time := categories * total_category_time
  let total_time := full_trial_time + full_category_time
  let non_watching_time := search_time * num_trials + break_time * (num_trials - 1)
  let available_time := time_limit - non_watching_time
  let max_videos := available_time / video_time
  max_videos

theorem billy_videos_within_limit : total_videos_watched_within_time_limit 90 4 3 5 5 15 2 10 = 13 := by
  sorry

end NUMINAMATH_GPT_billy_videos_within_limit_l2169_216944


namespace NUMINAMATH_GPT_probability_of_two_red_balls_l2169_216983

theorem probability_of_two_red_balls :
  let red_balls := 4
  let blue_balls := 4
  let green_balls := 2
  let total_balls := red_balls + blue_balls + green_balls
  let prob_red1 := (red_balls : ℚ) / total_balls
  let prob_red2 := ((red_balls - 1 : ℚ) / (total_balls - 1))
  (prob_red1 * prob_red2 = (2 : ℚ) / 15) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_two_red_balls_l2169_216983


namespace NUMINAMATH_GPT_marilyn_total_caps_l2169_216960

def marilyn_initial_caps : ℝ := 51.0
def nancy_gives_caps : ℝ := 36.0
def total_caps (initial: ℝ) (given: ℝ) : ℝ := initial + given

theorem marilyn_total_caps : total_caps marilyn_initial_caps nancy_gives_caps = 87.0 :=
by
  sorry

end NUMINAMATH_GPT_marilyn_total_caps_l2169_216960


namespace NUMINAMATH_GPT_general_term_formula_l2169_216915

theorem general_term_formula (a S : ℕ → ℝ) (h : ∀ n, S n = (2 / 3) * a n + (1 / 3)) :
  (a 1 = 1) ∧ (∀ n, n ≥ 2 → a n = -2 * a (n - 1)) →
  ∀ n, a n = (-2)^(n - 1) :=
by
  sorry

end NUMINAMATH_GPT_general_term_formula_l2169_216915


namespace NUMINAMATH_GPT_find_k_of_quadratic_eq_ratio_3_to_1_l2169_216901

theorem find_k_of_quadratic_eq_ratio_3_to_1 (k : ℝ) :
  (∃ (x : ℝ), x ≠ 0 ∧ (x^2 + 8 * x + k = 0) ∧
              (∃ (r : ℝ), x = 3 * r ∧ 3 * r + r = -8)) → k = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_k_of_quadratic_eq_ratio_3_to_1_l2169_216901


namespace NUMINAMATH_GPT_mulch_cost_l2169_216958

-- Definitions based on conditions
def cost_per_cubic_foot : ℕ := 8
def cubic_yard_to_cubic_feet : ℕ := 27
def volume_in_cubic_yards : ℕ := 7

-- Target statement to prove
theorem mulch_cost :
    (volume_in_cubic_yards * cubic_yard_to_cubic_feet) * cost_per_cubic_foot = 1512 := by
  sorry

end NUMINAMATH_GPT_mulch_cost_l2169_216958


namespace NUMINAMATH_GPT_prove_fractions_sum_equal_11_l2169_216931

variable (a b c : ℝ)

-- Given conditions
axiom h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -9
axiom h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 10

-- The proof problem statement
theorem prove_fractions_sum_equal_11 : (b / (a + b) + c / (b + c) + a / (c + a)) = 11 :=
by
  sorry

end NUMINAMATH_GPT_prove_fractions_sum_equal_11_l2169_216931


namespace NUMINAMATH_GPT_maximize_profit_l2169_216949

def total_orders := 100
def max_days := 160
def time_per_A := 5 / 4 -- days
def time_per_B := 5 / 3 -- days
def profit_per_A := 0.5 -- (10,000 RMB)
def profit_per_B := 0.8 -- (10,000 RMB)

theorem maximize_profit : 
  ∃ (x : ℝ) (y : ℝ), 
    (time_per_A * x + time_per_B * (total_orders - x) ≤ max_days) ∧ 
    (y = -0.3 * x + 80) ∧ 
    (x = 16) ∧ 
    (y = 75.2) :=
by 
  sorry

end NUMINAMATH_GPT_maximize_profit_l2169_216949


namespace NUMINAMATH_GPT_jason_initial_cards_l2169_216990

/-- Jason initially had some Pokemon cards, Alyssa bought him 224 more, 
and now Jason has 900 Pokemon cards in total.
Prove that initially Jason had 676 Pokemon cards. -/
theorem jason_initial_cards (a b c : ℕ) (h_a : a = 224) (h_b : b = 900) (h_cond : b = a + 676) : 676 = c :=
by 
  sorry

end NUMINAMATH_GPT_jason_initial_cards_l2169_216990


namespace NUMINAMATH_GPT_evaluate_statements_l2169_216975

-- Defining what it means for angles to be vertical
def vertical_angles (α β : ℝ) : Prop := α = β

-- Defining what complementary angles are
def complementary (α β : ℝ) : Prop := α + β = 90

-- Defining what supplementary angles are
def supplementary (α β : ℝ) : Prop := α + β = 180

-- Define the geometric properties for perpendicular and parallel lines
def unique_perpendicular_through_point (l : ℝ → ℝ) (p : ℝ × ℝ): Prop :=
  ∃! m, ∀ x, m * x + p.2 = l x

def unique_parallel_through_point (l : ℝ → ℝ) (p : ℝ × ℝ): Prop :=
  ∃! m, ∀ x, (l x ≠ m * x + p.2) ∧ (∀ y, y ≠ p.2 → l y ≠ m * y)

theorem evaluate_statements :
  (¬ ∃ α β, α = β ∧ vertical_angles α β) ∧
  (¬ ∃ α β, supplementary α β ∧ complementary α β) ∧
  ∃ l p, unique_perpendicular_through_point l p ∧
  ∃ l p, unique_parallel_through_point l p →
  2 = 2
  :=
by
  sorry  -- Proof is omitted

end NUMINAMATH_GPT_evaluate_statements_l2169_216975


namespace NUMINAMATH_GPT_knight_count_l2169_216919

theorem knight_count (K L : ℕ) (h1 : K + L = 15) 
  (h2 : ∀ k, k < K → (∃ l, l < L ∧ l = 6)) 
  (h3 : ∀ l, l < L → (K > 7)) : K = 9 :=
by 
  sorry

end NUMINAMATH_GPT_knight_count_l2169_216919


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2169_216956

def hyperbola : Prop :=
  ∀ (x y : ℝ), (x^2 / 9) - (y^2 / 16) = 1

noncomputable def eccentricity : ℝ :=
  let a := 3
  let b := 4
  let c := Real.sqrt (a^2 + b^2)
  c / a

theorem hyperbola_eccentricity : 
  (∀ (x y : ℝ), (x^2 / 9) - (y^2 / 16) = 1) → eccentricity = 5 / 3 :=
by
  intros h
  funext
  exact sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l2169_216956


namespace NUMINAMATH_GPT_doughnut_machine_completion_l2169_216971

noncomputable def completion_time (start_time : ℕ) (partial_duration : ℕ) : ℕ :=
  start_time + 4 * partial_duration

theorem doughnut_machine_completion :
  let start_time := 8 * 60  -- 8:00 AM in minutes
  let partial_completion_time := 11 * 60 + 40  -- 11:40 AM in minutes
  let one_fourth_duration := partial_completion_time - start_time
  completion_time start_time one_fourth_duration = (22 * 60 + 40) := -- 10:40 PM in minutes
by
  sorry

end NUMINAMATH_GPT_doughnut_machine_completion_l2169_216971


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l2169_216954

theorem system1_solution (x y : ℝ) (h1 : 3 * x + y = 4) (h2 : 3 * x + 2 * y = 6) : x = 2 / 3 ∧ y = 2 :=
by
  sorry

theorem system2_solution (x y : ℝ) (h1 : 2 * x + y = 3) (h2 : 3 * x - 5 * y = 11) : x = 2 ∧ y = -1 :=
by
  sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l2169_216954


namespace NUMINAMATH_GPT_problem_f_of_f_neg1_eq_neg1_l2169_216951

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(1 - x) else 1 - Real.logb 2 x

-- State the proposition to be proved
theorem problem_f_of_f_neg1_eq_neg1 : f (f (-1)) = -1 := by
  sorry

end NUMINAMATH_GPT_problem_f_of_f_neg1_eq_neg1_l2169_216951


namespace NUMINAMATH_GPT_polynomial_identity_l2169_216974

theorem polynomial_identity :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_identity_l2169_216974


namespace NUMINAMATH_GPT_triangular_pyramid_volume_l2169_216980

theorem triangular_pyramid_volume (a b c : ℝ) 
  (h1 : 1 / 2 * a * b = 6) 
  (h2 : 1 / 2 * a * c = 4) 
  (h3 : 1 / 2 * b * c = 3) : 
  (1 / 3) * (1 / 2) * a * b * c = 4 := by 
  sorry

end NUMINAMATH_GPT_triangular_pyramid_volume_l2169_216980


namespace NUMINAMATH_GPT_trig_identity_proof_l2169_216997

variable (α : ℝ)

theorem trig_identity_proof : 
  16 * (Real.sin α)^5 - 20 * (Real.sin α)^3 + 5 * Real.sin α = Real.sin (5 * α) :=
  sorry

end NUMINAMATH_GPT_trig_identity_proof_l2169_216997


namespace NUMINAMATH_GPT_students_can_be_helped_on_fourth_day_l2169_216970

theorem students_can_be_helped_on_fourth_day : 
  ∀ (total_books first_day_students second_day_students third_day_students books_per_student : ℕ),
  total_books = 120 →
  first_day_students = 4 →
  second_day_students = 5 →
  third_day_students = 6 →
  books_per_student = 5 →
  (total_books - (first_day_students * books_per_student + second_day_students * books_per_student + third_day_students * books_per_student)) / books_per_student = 9 :=
by
  intros total_books first_day_students second_day_students third_day_students books_per_student h_total h_first h_second h_third h_books_per_student
  sorry

end NUMINAMATH_GPT_students_can_be_helped_on_fourth_day_l2169_216970


namespace NUMINAMATH_GPT_neznaika_is_wrong_l2169_216906

theorem neznaika_is_wrong (avg_december avg_january : ℝ)
  (h_avg_dec : avg_december = 10)
  (h_avg_jan : avg_january = 5) : 
  ∃ (dec_days jan_days : ℕ), 
    (avg_december = (dec_days * 10 + (31 - dec_days) * 0) / 31) ∧
    (avg_january = (jan_days * 10 + (31 - jan_days) * 0) / 31) ∧
    jan_days > dec_days :=
by 
  sorry

end NUMINAMATH_GPT_neznaika_is_wrong_l2169_216906


namespace NUMINAMATH_GPT_operation_is_commutative_and_associative_l2169_216905

variables {S : Type} (op : S → S → S)

-- defining the properties given in the conditions
def idempotent (op : S → S → S) : Prop :=
  ∀ (a : S), op a a = a

def medial (op : S → S → S) : Prop :=
  ∀ (a b c : S), op (op a b) c = op (op b c) a

-- defining commutative and associative properties
def commutative (op : S → S → S) : Prop :=
  ∀ (a b : S), op a b = op b a

def associative (op : S → S → S) : Prop :=
  ∀ (a b c : S), op (op a b) c = op a (op b c)

-- statement of the theorem to prove
theorem operation_is_commutative_and_associative 
  (idemp : idempotent op) 
  (med : medial op) : commutative op ∧ associative op :=
sorry

end NUMINAMATH_GPT_operation_is_commutative_and_associative_l2169_216905


namespace NUMINAMATH_GPT_find_focus_with_larger_x_l2169_216984

def hyperbola_foci_coordinates : Prop :=
  let center := (5, 10)
  let a := 7
  let b := 3
  let c := Real.sqrt (a^2 + b^2)
  let focus1 := (5 + c, 10)
  let focus2 := (5 - c, 10)
  focus1 = (5 + Real.sqrt 58, 10)
  
theorem find_focus_with_larger_x : hyperbola_foci_coordinates := 
  by
    sorry

end NUMINAMATH_GPT_find_focus_with_larger_x_l2169_216984


namespace NUMINAMATH_GPT_smallest_k_exists_l2169_216994

theorem smallest_k_exists : ∃ (k : ℕ), k > 0 ∧ (∃ (n m : ℕ), n > 0 ∧ m > 0 ∧ k = 19^n - 5^m) ∧ k = 14 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_k_exists_l2169_216994


namespace NUMINAMATH_GPT_reservoir_water_level_l2169_216985

theorem reservoir_water_level (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : 
  ∃ y : ℝ, y = 6 + 0.3 * x :=
by sorry

end NUMINAMATH_GPT_reservoir_water_level_l2169_216985


namespace NUMINAMATH_GPT_students_like_neither_l2169_216952

theorem students_like_neither (N_Total N_Chinese N_Math N_Both N_Neither : ℕ)
  (h_total: N_Total = 62)
  (h_chinese: N_Chinese = 37)
  (h_math: N_Math = 49)
  (h_both: N_Both = 30)
  (h_neither: N_Neither = N_Total - (N_Chinese - N_Both) - (N_Math - N_Both) - N_Both) : 
  N_Neither = 6 :=
by 
  rw [h_total, h_chinese, h_math, h_both] at h_neither
  exact h_neither.trans (by norm_num)


end NUMINAMATH_GPT_students_like_neither_l2169_216952


namespace NUMINAMATH_GPT_grasshopper_jump_distance_l2169_216904

-- Definitions based on conditions
def frog_jump : ℤ := 39
def higher_jump_distance : ℤ := 22
def grasshopper_jump : ℤ := frog_jump - higher_jump_distance

-- The statement we need to prove
theorem grasshopper_jump_distance :
  grasshopper_jump = 17 :=
by
  -- Here, proof would be provided but we skip with sorry
  sorry

end NUMINAMATH_GPT_grasshopper_jump_distance_l2169_216904


namespace NUMINAMATH_GPT_total_potatoes_l2169_216943

open Nat

theorem total_potatoes (P T R : ℕ) (h1 : P = 5) (h2 : T = 6) (h3 : R = 48) : P + (R / T) = 13 := by
  sorry

end NUMINAMATH_GPT_total_potatoes_l2169_216943


namespace NUMINAMATH_GPT_simplify_expression_l2169_216917

theorem simplify_expression (x : ℝ) (h : x ≠ 0) : 
  (x-2) ^ 2 - x * (x-1) + (x^3 - 4 * x^2) / x^2 = -2 * x := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l2169_216917


namespace NUMINAMATH_GPT_ellipse_properties_l2169_216912

theorem ellipse_properties :
  (∃ a e : ℝ, (∃ b c : ℝ, a^2 = 25 ∧ b^2 = 9 ∧ c^2 = a^2 - b^2 ∧ c = 4 ∧ e = c / a) ∧ a = 5 ∧ e = 4 / 5) :=
sorry

end NUMINAMATH_GPT_ellipse_properties_l2169_216912


namespace NUMINAMATH_GPT_f_prime_at_zero_l2169_216972

-- Lean definition of the conditions.
def a (n : ℕ) : ℝ := 2 * (2 ^ (1/7)) ^ (n - 1)

-- The function f(x) based on the given conditions.
noncomputable def f (x : ℝ) : ℝ := 
  x * (x - a 1) * (x - a 2) * (x - a 3) * (x - a 4) * 
  (x - a 5) * (x - a 6) * (x - a 7) * (x - a 8)

-- The main goal to prove: f'(0) = 2^12
theorem f_prime_at_zero : deriv f 0 = 2^12 := by
  sorry

end NUMINAMATH_GPT_f_prime_at_zero_l2169_216972


namespace NUMINAMATH_GPT_coins_ratio_l2169_216933

-- Conditions
def initial_coins : Nat := 125
def gift_coins : Nat := 35
def sold_coins : Nat := 80

-- Total coins after receiving the gift
def total_coins := initial_coins + gift_coins

-- Statement to prove the ratio simplifies to 1:2
theorem coins_ratio : (sold_coins : ℚ) / total_coins = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_coins_ratio_l2169_216933


namespace NUMINAMATH_GPT_pair_cannot_appear_l2169_216981

theorem pair_cannot_appear :
  ¬ ∃ (sequence_of_pairs : List (ℤ × ℤ)), 
    (1, 2) ∈ sequence_of_pairs ∧ 
    (2022, 2023) ∈ sequence_of_pairs ∧ 
    ∀ (a b : ℤ) (seq : List (ℤ × ℤ)), 
      (a, b) ∈ seq → 
      ((-a, -b) ∈ seq ∨ (-b, a+b) ∈ seq ∨ 
      ∃ (c d : ℤ), ((a+c, b+d) ∈ seq ∧ (c, d) ∈ seq)) := 
sorry

end NUMINAMATH_GPT_pair_cannot_appear_l2169_216981


namespace NUMINAMATH_GPT_savings_increase_l2169_216964

variable (I : ℝ) -- Initial income
variable (E : ℝ) -- Initial expenditure
variable (S : ℝ) -- Initial savings
variable (I_new : ℝ) -- New income
variable (E_new : ℝ) -- New expenditure
variable (S_new : ℝ) -- New savings

theorem savings_increase (h1 : E = 0.75 * I) 
                         (h2 : I_new = 1.20 * I) 
                         (h3 : E_new = 1.10 * E) : 
                         (S_new - S) / S * 100 = 50 :=
by 
  have h4 : S = 0.25 * I := by sorry
  have h5 : E_new = 0.825 * I := by sorry
  have h6 : S_new = 0.375 * I := by sorry
  have increase : (S_new - S) / S * 100 = 50 := by sorry
  exact increase

end NUMINAMATH_GPT_savings_increase_l2169_216964


namespace NUMINAMATH_GPT_distance_ratio_l2169_216937

-- Define the distances as given in the conditions
def distance_from_city_sky_falls := 8 -- Distance in miles
def distance_from_city_rocky_mist := 400 -- Distance in miles

theorem distance_ratio : distance_from_city_rocky_mist / distance_from_city_sky_falls = 50 := 
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_distance_ratio_l2169_216937


namespace NUMINAMATH_GPT_find_slope_of_q_l2169_216966

theorem find_slope_of_q (j : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + 3 → y = j * x + 1 → x = 1 → y = 5) → j = 4 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_find_slope_of_q_l2169_216966


namespace NUMINAMATH_GPT_smallest_a_satisfies_sin_condition_l2169_216982

open Real

theorem smallest_a_satisfies_sin_condition :
  ∃ (a : ℝ), (∀ x : ℤ, sin (a * x + 0) = sin (45 * x)) ∧ 0 ≤ a ∧ ∀ b : ℝ, (∀ x : ℤ, sin (b * x + 0) = sin (45 * x)) ∧ 0 ≤ b → 45 ≤ b :=
by
  -- To be proved.
  sorry

end NUMINAMATH_GPT_smallest_a_satisfies_sin_condition_l2169_216982


namespace NUMINAMATH_GPT_geom_seq_product_l2169_216939

theorem geom_seq_product {a : ℕ → ℝ} (h_geom : ∀ n, a (n + 1) = a n * r)
 (h_a1 : a 1 = 1 / 2) (h_a5 : a 5 = 8) : a 2 * a 3 * a 4 = 8 := 
sorry

end NUMINAMATH_GPT_geom_seq_product_l2169_216939


namespace NUMINAMATH_GPT_orchestra_admission_l2169_216979

theorem orchestra_admission (x v c t: ℝ) 
  -- Conditions
  (h1 : v = 1.25 * 1.6 * x)
  (h2 : c = 0.8 * x)
  (h3 : t = 0.4 * x)
  (h4 : v + c + t = 32) :
  -- Conclusion
  v = 20 ∧ c = 8 ∧ t = 4 :=
sorry

end NUMINAMATH_GPT_orchestra_admission_l2169_216979


namespace NUMINAMATH_GPT_mutually_exclusive_events_not_complementary_l2169_216948

def event_a (ball: ℕ) (box: ℕ): Prop := ball = 1 ∧ box = 1
def event_b (ball: ℕ) (box: ℕ): Prop := ball = 1 ∧ box = 2

theorem mutually_exclusive_events_not_complementary :
  (∀ ball box, event_a ball box → ¬ event_b ball box) ∧ 
  (∃ box, ¬((event_a 1 box) ∨ (event_b 1 box))) :=
by
  sorry

end NUMINAMATH_GPT_mutually_exclusive_events_not_complementary_l2169_216948


namespace NUMINAMATH_GPT_days_wages_l2169_216986

theorem days_wages (S W_a W_b : ℝ) 
    (h1 : S = 28 * W_b) 
    (h2 : S = 12 * (W_a + W_b)) 
    (h3 : S = 21 * W_a) : 
    true := 
by sorry

end NUMINAMATH_GPT_days_wages_l2169_216986


namespace NUMINAMATH_GPT_functional_equation_solution_l2169_216902

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  1 / (1 + a * x)

theorem functional_equation_solution (a : ℝ) (x y : ℝ)
  (ha : 0 < a) (hx : 0 < x) (hy : 0 < y) :
  f a x * f a (y * f a x) = f a (x + y) :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l2169_216902


namespace NUMINAMATH_GPT_binary_arith_proof_l2169_216921

theorem binary_arith_proof :
  let a := 0b1101110  -- binary representation of 1101110_2
  let b := 0b101010   -- binary representation of 101010_2
  let c := 0b100      -- binary representation of 100_2
  (a * b / c) = 0b11001000010 :=  -- binary representation of the final result
by
  sorry

end NUMINAMATH_GPT_binary_arith_proof_l2169_216921


namespace NUMINAMATH_GPT_find_m_l2169_216935

open Real

namespace VectorPerpendicular

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (-1, m)

def perpendicular (v₁ v₂ : ℝ × ℝ) : Prop := (v₁.1 * v₂.1 + v₁.2 * v₂.2) = 0

theorem find_m (m : ℝ) (h : perpendicular a (b m)) : m = 1 / 2 :=
by
  sorry -- Proof is omitted

end VectorPerpendicular

end NUMINAMATH_GPT_find_m_l2169_216935


namespace NUMINAMATH_GPT_evaluate_fraction_l2169_216926

theorem evaluate_fraction :
  1 + 1 / (2 + 1 / (3 + 1 / (3 + 3))) = 63 / 44 := 
by
  -- Skipping the proof part with 'sorry'
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l2169_216926


namespace NUMINAMATH_GPT_sun_salutations_per_year_l2169_216941

-- Definitions 
def sun_salutations_per_weekday : ℕ := 5
def weekdays_per_week : ℕ := 5
def weeks_per_year : ℕ := 52

-- Problem statement to prove
theorem sun_salutations_per_year :
  sun_salutations_per_weekday * weekdays_per_week * weeks_per_year = 1300 :=
by
  sorry

end NUMINAMATH_GPT_sun_salutations_per_year_l2169_216941


namespace NUMINAMATH_GPT_distance_between_andrey_and_valentin_l2169_216934

-- Definitions based on conditions
def speeds_relation_andrey_boris (a b : ℝ) := b = 0.94 * a
def speeds_relation_boris_valentin (b c : ℝ) := c = 0.95 * b

theorem distance_between_andrey_and_valentin
  (a b c : ℝ)
  (h1 : speeds_relation_andrey_boris a b)
  (h2 : speeds_relation_boris_valentin b c)
  : 1000 - 1000 * c / a = 107 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_andrey_and_valentin_l2169_216934


namespace NUMINAMATH_GPT_min_p_plus_q_l2169_216959

-- Define the conditions
variables {p q : ℕ}

-- Problem statement in Lean 4
theorem min_p_plus_q (h₁ : p > 0) (h₂ : q > 0) (h₃ : 108 * p = q^3) : p + q = 8 :=
sorry

end NUMINAMATH_GPT_min_p_plus_q_l2169_216959


namespace NUMINAMATH_GPT_num_solution_pairs_l2169_216913

theorem num_solution_pairs : 
  ∃! (n : ℕ), 
    n = 2 ∧ 
    ∃ x y : ℕ, 
      x > 0 ∧ y >0 ∧ 
      4^x = y^2 + 15 := 
by 
  sorry

end NUMINAMATH_GPT_num_solution_pairs_l2169_216913


namespace NUMINAMATH_GPT_evaluate_expression_at_3_l2169_216963

theorem evaluate_expression_at_3 :
  ((3^(3^2))^(3^3)) = 3^(243) := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_3_l2169_216963


namespace NUMINAMATH_GPT_find_original_denominator_l2169_216993

theorem find_original_denominator (d : ℕ) (h : (3 + 7) / (d + 7) = 1 / 3) : d = 23 :=
sorry

end NUMINAMATH_GPT_find_original_denominator_l2169_216993


namespace NUMINAMATH_GPT_calculateSurfaceArea_l2169_216924

noncomputable def totalSurfaceArea (r : ℝ) : ℝ :=
  let hemisphereCurvedArea := 2 * Real.pi * r^2
  let cylinderLateralArea := 2 * Real.pi * r * r
  hemisphereCurvedArea + cylinderLateralArea

theorem calculateSurfaceArea :
  ∃ r : ℝ, (Real.pi * r^2 = 144 * Real.pi) ∧ totalSurfaceArea r = 576 * Real.pi :=
by
  exists 12
  constructor
  . sorry -- Proof that 144π = π*12^2 can be shown
  . sorry -- Proof that 576π = 288π + 288π can be shown

end NUMINAMATH_GPT_calculateSurfaceArea_l2169_216924


namespace NUMINAMATH_GPT_revenue_comparison_l2169_216955

theorem revenue_comparison 
  (D N J F : ℚ) 
  (hN : N = (2 / 5) * D) 
  (hJ : J = (2 / 25) * D) 
  (hF : F = (3 / 4) * D) : 
  D / ((N + J + F) / 3) = 100 / 41 := 
by 
  sorry

end NUMINAMATH_GPT_revenue_comparison_l2169_216955


namespace NUMINAMATH_GPT_total_animals_in_jacobs_flock_l2169_216918

-- Define the conditions of the problem
def one_third_of_animals_are_goats (total goats : ℕ) : Prop := 
  3 * goats = total

def twelve_more_sheep_than_goats (goats sheep : ℕ) : Prop :=
  sheep = goats + 12

-- Define the main theorem to prove
theorem total_animals_in_jacobs_flock : 
  ∃ total goats sheep : ℕ, one_third_of_animals_are_goats total goats ∧ 
                           twelve_more_sheep_than_goats goats sheep ∧ 
                           total = 36 := 
by
  sorry

end NUMINAMATH_GPT_total_animals_in_jacobs_flock_l2169_216918


namespace NUMINAMATH_GPT_intersection_of_sets_l2169_216998

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {1, 2, 3}) (hB : B = {3, 4}) :
  A ∩ B = {3} :=
by
  rw [hA, hB]
  exact sorry

end NUMINAMATH_GPT_intersection_of_sets_l2169_216998


namespace NUMINAMATH_GPT_part_a_l2169_216968

def system_of_equations (x y z a : ℝ) := 
  (x - a * y = y * z) ∧ (y - a * z = z * x) ∧ (z - a * x = x * y)

theorem part_a (x y z : ℝ) : 
  system_of_equations x y z 0 ↔ (x = 0 ∧ y = 0 ∧ z = 0) 
  ∨ (∃ x, y = x ∧ z = 1) 
  ∨ (∃ x, y = -x ∧ z = -1) := 
  sorry

end NUMINAMATH_GPT_part_a_l2169_216968


namespace NUMINAMATH_GPT_blue_ball_higher_probability_l2169_216973

noncomputable def probability_blue_ball_higher : ℝ :=
  let p (k : ℕ) : ℝ := 1 / (2^k : ℝ)
  let same_bin_prob := ∑' k : ℕ, (p (k + 1))^2
  let higher_prob := (1 - same_bin_prob) / 2
  higher_prob

theorem blue_ball_higher_probability :
  probability_blue_ball_higher = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_blue_ball_higher_probability_l2169_216973


namespace NUMINAMATH_GPT_asian_countries_visited_l2169_216977

theorem asian_countries_visited (total_countries europe_countries south_america_countries remaining_asian_countries : ℕ)
  (h1 : total_countries = 42)
  (h2 : europe_countries = 20)
  (h3 : south_america_countries = 10)
  (h4 : remaining_asian_countries = (total_countries - (europe_countries + south_america_countries)) / 2) :
  remaining_asian_countries = 6 :=
by sorry

end NUMINAMATH_GPT_asian_countries_visited_l2169_216977


namespace NUMINAMATH_GPT_units_digit_of_p_is_6_l2169_216957

theorem units_digit_of_p_is_6 (p : ℕ) (h_even : Even p) (h_units_p_plus_1 : (p + 1) % 10 = 7) (h_units_p3_minus_p2 : ((p^3) % 10 - (p^2) % 10) % 10 = 0) : p % 10 = 6 := 
by 
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_units_digit_of_p_is_6_l2169_216957


namespace NUMINAMATH_GPT_simplify_sqrt_square_l2169_216929

theorem simplify_sqrt_square (h : Real.sqrt 7 < 3) : Real.sqrt ((Real.sqrt 7 - 3)^2) = 3 - Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_square_l2169_216929
