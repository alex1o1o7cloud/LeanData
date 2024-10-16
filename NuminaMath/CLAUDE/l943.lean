import Mathlib

namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l943_94342

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The slope of the tangent line to f(x) at x = 1 is 2 -/
theorem tangent_slope_at_one : 
  (deriv f) 1 = 2 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l943_94342


namespace NUMINAMATH_CALUDE_inequality_equivalence_l943_94335

theorem inequality_equivalence (x : ℝ) : 5 * x - 12 ≤ 2 * (4 * x - 3) ↔ x ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l943_94335


namespace NUMINAMATH_CALUDE_theater_ticket_price_l943_94386

/-- Proves that the price of a balcony seat is $8 given the theater ticket sales conditions --/
theorem theater_ticket_price (total_tickets : ℕ) (total_revenue : ℕ) 
  (orchestra_price : ℕ) (balcony_orchestra_diff : ℕ) :
  total_tickets = 360 →
  total_revenue = 3320 →
  orchestra_price = 12 →
  balcony_orchestra_diff = 140 →
  ∃ (balcony_price : ℕ), 
    balcony_price = 8 ∧
    balcony_price * (total_tickets / 2 + balcony_orchestra_diff / 2) + 
    orchestra_price * (total_tickets / 2 - balcony_orchestra_diff / 2) = total_revenue :=
by
  sorry

#check theater_ticket_price

end NUMINAMATH_CALUDE_theater_ticket_price_l943_94386


namespace NUMINAMATH_CALUDE_quadratic_sum_zero_l943_94330

theorem quadratic_sum_zero (a b c x₁ x₂ : ℝ) (ha : a ≠ 0)
  (hx₁ : a * x₁^2 + b * x₁ + c = 0)
  (hx₂ : a * x₂^2 + b * x₂ + c = 0)
  (s₁ : ℝ := x₁^2005 + x₂^2005)
  (s₂ : ℝ := x₁^2004 + x₂^2004)
  (s₃ : ℝ := x₁^2003 + x₂^2003) :
  a * s₁ + b * s₂ + c * s₃ = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_zero_l943_94330


namespace NUMINAMATH_CALUDE_road_cleaning_problem_l943_94315

/-- The distance between East City and West City in kilometers -/
def total_distance : ℝ := 60

/-- The time it takes Vehicle A to clean the entire road alone in hours -/
def time_A : ℝ := 10

/-- The time it takes Vehicle B to clean the entire road alone in hours -/
def time_B : ℝ := 15

/-- The additional distance cleaned by Vehicle A compared to Vehicle B when they meet, in kilometers -/
def extra_distance_A : ℝ := 12

theorem road_cleaning_problem :
  let speed_A := total_distance / time_A
  let speed_B := total_distance / time_B
  let combined_speed := speed_A + speed_B
  let meeting_time := total_distance / combined_speed
  speed_A * meeting_time - speed_B * meeting_time = extra_distance_A :=
by sorry

#check road_cleaning_problem

end NUMINAMATH_CALUDE_road_cleaning_problem_l943_94315


namespace NUMINAMATH_CALUDE_lanas_final_page_count_l943_94320

theorem lanas_final_page_count (lana_initial : ℕ) (duane_total : ℕ) : 
  lana_initial = 8 → duane_total = 42 → lana_initial + duane_total / 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_lanas_final_page_count_l943_94320


namespace NUMINAMATH_CALUDE_simplify_fraction_l943_94300

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l943_94300


namespace NUMINAMATH_CALUDE_min_oranges_in_new_box_l943_94376

theorem min_oranges_in_new_box (m n x : ℕ) : 
  m + n ≤ 60 →
  59 * m = 60 * n + x →
  x > 0 →
  (∀ y : ℕ, y < x → ¬(∃ m' n' : ℕ, m' + n' ≤ 60 ∧ 59 * m' = 60 * n' + y)) →
  x = 30 :=
by sorry

end NUMINAMATH_CALUDE_min_oranges_in_new_box_l943_94376


namespace NUMINAMATH_CALUDE_quadratic_root_value_l943_94357

theorem quadratic_root_value (k : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 2 * x + k = 0 ↔ x = (1 + Complex.I * Real.sqrt 39) / 10 ∨ x = (1 - Complex.I * Real.sqrt 39) / 10) →
  k = 2.15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l943_94357


namespace NUMINAMATH_CALUDE_furniture_fraction_l943_94338

theorem furniture_fraction (original_savings tv_cost : ℚ) 
  (h1 : original_savings = 500)
  (h2 : tv_cost = 100) : 
  (original_savings - tv_cost) / original_savings = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_furniture_fraction_l943_94338


namespace NUMINAMATH_CALUDE_pond_length_l943_94328

theorem pond_length (field_length field_width pond_area : ℝ) : 
  field_length = 28 ∧ 
  field_width = 14 ∧ 
  field_length = 2 * field_width ∧ 
  pond_area = (field_length * field_width) / 8 → 
  Real.sqrt pond_area = 7 := by
  sorry

end NUMINAMATH_CALUDE_pond_length_l943_94328


namespace NUMINAMATH_CALUDE_find_numbers_with_difference_and_quotient_equal_l943_94360

theorem find_numbers_with_difference_and_quotient_equal (x y : ℚ) :
  x - y = 5 ∧ x / y = 5 → x = 25 / 4 ∧ y = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_find_numbers_with_difference_and_quotient_equal_l943_94360


namespace NUMINAMATH_CALUDE_september_solution_l943_94397

/-- A function that maps a month number to its corresponding solution in the equations -/
def month_solution : ℕ → ℝ
| 2 => 2  -- February
| 4 => 4  -- April
| 9 => 9  -- September
| _ => 0  -- Other months (not relevant for this problem)

/-- The theorem stating that the solution of 48 = 5x + 3 corresponds to the 9th month -/
theorem september_solution :
  (month_solution 2 - 1 = 1) ∧
  (18 - 2 * month_solution 4 = 10) ∧
  (48 = 5 * month_solution 9 + 3) := by
  sorry

#check september_solution

end NUMINAMATH_CALUDE_september_solution_l943_94397


namespace NUMINAMATH_CALUDE_equal_volume_equal_capacity_container2_capacity_l943_94378

/-- Represents a rectangular container -/
structure Container where
  height : ℝ
  width : ℝ
  length : ℝ
  capacity : ℝ

/-- Calculates the volume of a container -/
def volume (c : Container) : ℝ := c.height * c.width * c.length

/-- Theorem: Two containers with the same volume have the same capacity -/
theorem equal_volume_equal_capacity (c1 c2 : Container) 
  (h_volume : volume c1 = volume c2) 
  (h_capacity : c1.capacity = 80) : 
  c2.capacity = 80 := by
  sorry

/-- The first container -/
def container1 : Container := {
  height := 2,
  width := 3,
  length := 10,
  capacity := 80
}

/-- The second container -/
def container2 : Container := {
  height := 1,
  width := 3,
  length := 20,
  capacity := 80  -- We'll prove this
}

/-- Proof that container2 can hold 80 grams -/
theorem container2_capacity : container2.capacity = 80 := by
  apply equal_volume_equal_capacity container1 container2
  · -- Prove that volumes are equal
    simp [volume, container1, container2]
    -- 2 * 3 * 10 = 1 * 3 * 20
    ring
  · -- Show that container1's capacity is 80
    rfl

#check container2_capacity

end NUMINAMATH_CALUDE_equal_volume_equal_capacity_container2_capacity_l943_94378


namespace NUMINAMATH_CALUDE_shoe_cost_l943_94321

/-- The cost of shoes given an initial budget and remaining amount --/
theorem shoe_cost (initial_budget remaining : ℚ) (h1 : initial_budget = 999) (h2 : remaining = 834) :
  initial_budget - remaining = 165 := by
  sorry

end NUMINAMATH_CALUDE_shoe_cost_l943_94321


namespace NUMINAMATH_CALUDE_catch_up_time_l943_94350

/-- Two people walk in opposite directions at the same speed for 10 minutes,
    then one increases speed by 5 times and chases the other. -/
theorem catch_up_time (s : ℝ) (h : s > 0) : 
  let initial_distance := 2 * 10 * s
  let relative_speed := 5 * s - s
  initial_distance / relative_speed = 5 :=
by sorry

end NUMINAMATH_CALUDE_catch_up_time_l943_94350


namespace NUMINAMATH_CALUDE_painting_sale_difference_l943_94383

def previous_painting_sale : ℕ := 9000
def recent_painting_sale : ℕ := 44000

theorem painting_sale_difference : 
  (5 * previous_painting_sale + previous_painting_sale) - recent_painting_sale = 10000 := by
  sorry

end NUMINAMATH_CALUDE_painting_sale_difference_l943_94383


namespace NUMINAMATH_CALUDE_dance_team_members_l943_94355

theorem dance_team_members :
  ∀ (track_members choir_members dance_members : ℕ),
    track_members + choir_members + dance_members = 100 →
    choir_members = 2 * track_members →
    dance_members = choir_members + 10 →
    dance_members = 46 := by
  sorry

end NUMINAMATH_CALUDE_dance_team_members_l943_94355


namespace NUMINAMATH_CALUDE_cardinals_second_inning_home_runs_l943_94340

theorem cardinals_second_inning_home_runs :
  let cubs_third_inning : ℕ := 2
  let cubs_fifth_inning : ℕ := 1
  let cubs_eighth_inning : ℕ := 2
  let cardinals_fifth_inning : ℕ := 1
  let cubs_total : ℕ := cubs_third_inning + cubs_fifth_inning + cubs_eighth_inning
  let cardinals_total : ℕ := cubs_total - 3
  let cardinals_second_inning : ℕ := cardinals_total - cardinals_fifth_inning
  cardinals_second_inning = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_cardinals_second_inning_home_runs_l943_94340


namespace NUMINAMATH_CALUDE_same_speed_is_two_l943_94354

-- Define Jack's speed function
def jack_speed (x : ℝ) : ℝ := x^2 - 7*x - 18

-- Define Jill's distance function
def jill_distance (x : ℝ) : ℝ := x^2 + x - 72

-- Define Jill's time function
def jill_time (x : ℝ) : ℝ := x + 8

-- Theorem statement
theorem same_speed_is_two :
  ∀ x : ℝ, 
  x ≠ -8 →  -- Ensure division by zero is avoided
  (jill_distance x) / (jill_time x) = jack_speed x →
  jack_speed x = 2 :=
by sorry

end NUMINAMATH_CALUDE_same_speed_is_two_l943_94354


namespace NUMINAMATH_CALUDE_derivative_positive_implies_increasing_l943_94394

open Set

theorem derivative_positive_implies_increasing
  {f : ℝ → ℝ} {I : Set ℝ} (hI : IsOpen I) (hf : DifferentiableOn ℝ f I)
  (h : ∀ x ∈ I, deriv f x > 0) :
  StrictMonoOn f I :=
sorry

end NUMINAMATH_CALUDE_derivative_positive_implies_increasing_l943_94394


namespace NUMINAMATH_CALUDE_min_value_a_plus_4b_l943_94375

theorem min_value_a_plus_4b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = x + y → a + 4 * b ≤ x + 4 * y ∧ 
  (a + 4 * b = 9 ↔ a = 3 ∧ b = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_4b_l943_94375


namespace NUMINAMATH_CALUDE_tournament_probability_l943_94347

/-- The number of teams in the tournament -/
def num_teams : ℕ := 30

/-- The total number of games played in the tournament -/
def total_games : ℕ := num_teams.choose 2

/-- The probability of a team winning any given game -/
def win_probability : ℚ := 1/2

/-- The probability that no two teams win the same number of games -/
noncomputable def unique_wins_probability : ℚ := (num_teams.factorial : ℚ) / 2^total_games

theorem tournament_probability :
  ∃ (m : ℕ), m % 2 = 1 ∧ unique_wins_probability = (m : ℚ) / 2^409 :=
sorry

end NUMINAMATH_CALUDE_tournament_probability_l943_94347


namespace NUMINAMATH_CALUDE_red_permutations_l943_94384

theorem red_permutations : 
  let n : ℕ := 1
  let total_letters : ℕ := 3 * n
  let permutations : ℕ := Nat.factorial total_letters / (Nat.factorial n)^3
  permutations = 6 := by sorry

end NUMINAMATH_CALUDE_red_permutations_l943_94384


namespace NUMINAMATH_CALUDE_bills_initial_money_l943_94339

theorem bills_initial_money (ann_initial : ℕ) (transfer : ℕ) (bill_initial : ℕ) : 
  ann_initial = 777 →
  transfer = 167 →
  ann_initial + transfer = bill_initial - transfer →
  bill_initial = 1111 := by
sorry

end NUMINAMATH_CALUDE_bills_initial_money_l943_94339


namespace NUMINAMATH_CALUDE_inequality_solution_l943_94353

theorem inequality_solution (x : ℝ) : (x^2 - 9) / (x^3 - 1) > 0 ↔ x < -3 ∨ (-3 < x ∧ x < 1) ∨ x > 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l943_94353


namespace NUMINAMATH_CALUDE_perpendicular_solution_parallel_solution_l943_94396

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 3)
def b (x : ℝ) : ℝ × ℝ := (x^2 - 1, x + 1)

-- Define perpendicularity condition
def perpendicular (x : ℝ) : Prop := a.1 * (b x).1 + a.2 * (b x).2 = 0

-- Define parallelism condition
def parallel (x : ℝ) : Prop := a.1 * (b x).2 = a.2 * (b x).1

-- Theorem for perpendicular case
theorem perpendicular_solution :
  ∀ x : ℝ, perpendicular x → x = -1 ∨ x = -2 := by sorry

-- Theorem for parallel case
theorem parallel_solution :
  ∀ x : ℝ, parallel x → 
    ‖(a.1 - (b x).1, a.2 - (b x).2)‖ = Real.sqrt 10 ∨
    ‖(a.1 - (b x).1, a.2 - (b x).2)‖ = 2 * Real.sqrt 10 / 9 := by sorry

end NUMINAMATH_CALUDE_perpendicular_solution_parallel_solution_l943_94396


namespace NUMINAMATH_CALUDE_anna_initial_stamps_l943_94365

theorem anna_initial_stamps (x : ℕ) (alison_stamps : ℕ) : 
  alison_stamps = 28 → 
  x + alison_stamps / 2 = 50 → 
  x = 36 := by
sorry

end NUMINAMATH_CALUDE_anna_initial_stamps_l943_94365


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l943_94301

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hc_def : c = (a + b) / 2) :
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l943_94301


namespace NUMINAMATH_CALUDE_train_length_calculation_train_problem_solution_l943_94314

/-- The length of each train in kilometers -/
def train_length : ℝ := 0.475

theorem train_length_calculation (v1 v2 t : ℝ) (h1 : v1 = 55) (h2 : v2 = 40) (h3 : t = 36) :
  2 * train_length = (v1 + v2) * t / 3600 :=
by sorry

/-- The main theorem stating the length of each train -/
theorem train_problem_solution :
  train_length = 0.475 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_train_problem_solution_l943_94314


namespace NUMINAMATH_CALUDE_last_twelve_average_l943_94319

theorem last_twelve_average (total_count : Nat) (total_average : ℝ) 
  (first_twelve_average : ℝ) (thirteenth_result : ℝ) :
  total_count = 25 →
  total_average = 18 →
  first_twelve_average = 10 →
  thirteenth_result = 90 →
  (total_count * total_average - 12 * first_twelve_average - thirteenth_result) / 12 = 20 := by
sorry

end NUMINAMATH_CALUDE_last_twelve_average_l943_94319


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l943_94317

/-- Given a right triangle ABC with angle C = 90°, BC = 6, and tan B = 0.75, prove that AC = 4.5 -/
theorem right_triangle_side_length (A B C : ℝ × ℝ) : 
  let triangle := (A, B, C)
  (∃ (AC BC : ℝ), 
    -- ABC is a right triangle with angle C = 90°
    (C.2 - A.2) * (B.1 - A.1) = (C.1 - A.1) * (B.2 - A.2) ∧
    -- BC = 6
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 6 ∧
    -- tan B = 0.75
    (C.2 - B.2) / (C.1 - B.1) = 0.75 ∧
    -- AC is the length we're solving for
    AC = Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)) →
  AC = 4.5 := by
sorry


end NUMINAMATH_CALUDE_right_triangle_side_length_l943_94317


namespace NUMINAMATH_CALUDE_mistaken_calculation_correction_l943_94385

theorem mistaken_calculation_correction (x : ℝ) :
  5.46 - x = 3.97 → 5.46 + x = 6.95 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_calculation_correction_l943_94385


namespace NUMINAMATH_CALUDE_arcsin_cos_4pi_over_7_l943_94349

theorem arcsin_cos_4pi_over_7 : 
  Real.arcsin (Real.cos (4 * π / 7)) = -π / 14 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_cos_4pi_over_7_l943_94349


namespace NUMINAMATH_CALUDE_fraction_simplification_l943_94341

theorem fraction_simplification (x : ℝ) (h : x ≠ -3) : 
  (x^2 - 9) / (x^2 + 6*x + 9) - (2*x + 1) / (2*x + 6) = -7 / (2*x + 6) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l943_94341


namespace NUMINAMATH_CALUDE_fraction_problem_l943_94387

theorem fraction_problem (x : ℚ) : 
  x^35 * (1/4)^18 = 1/(2*(10)^35) → x = 1/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l943_94387


namespace NUMINAMATH_CALUDE_min_value_and_reciprocal_sum_l943_94352

-- Define the function f
def f (a b c x : ℝ) : ℝ := |x + a| + |x - b| + c

-- State the theorem
theorem min_value_and_reciprocal_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hmin : ∀ x, f a b c x ≥ 5) 
  (hf_attains_min : ∃ x, f a b c x = 5) : 
  a + b + c = 5 ∧ (1/a + 1/b + 1/c ≥ 9/5 ∧ ∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 1/a' + 1/b' + 1/c' = 9/5) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_reciprocal_sum_l943_94352


namespace NUMINAMATH_CALUDE_seungho_original_marble_difference_l943_94392

/-- Proves that Seungho originally had 1023 more marbles than Hyukjin -/
theorem seungho_original_marble_difference (s h : ℕ) : 
  s - 273 = (h + 273) + 477 → s = h + 1023 := by
  sorry

end NUMINAMATH_CALUDE_seungho_original_marble_difference_l943_94392


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_two_l943_94327

/-- The curve function -/
def f (x : ℝ) : ℝ := x^2 + 3*x

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 2*x + 3

theorem tangent_slope_at_point_two :
  f' 2 = 7 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_two_l943_94327


namespace NUMINAMATH_CALUDE_annie_hamburgers_l943_94358

/-- Proves that Annie bought 8 hamburgers given the problem conditions -/
theorem annie_hamburgers :
  ∀ (initial_money : ℕ) (hamburger_cost : ℕ) (milkshake_cost : ℕ) 
    (milkshakes_bought : ℕ) (money_left : ℕ),
  initial_money = 132 →
  hamburger_cost = 4 →
  milkshake_cost = 5 →
  milkshakes_bought = 6 →
  money_left = 70 →
  ∃ (hamburgers_bought : ℕ),
    hamburgers_bought * hamburger_cost + milkshakes_bought * milkshake_cost = initial_money - money_left ∧
    hamburgers_bought = 8 :=
by sorry

end NUMINAMATH_CALUDE_annie_hamburgers_l943_94358


namespace NUMINAMATH_CALUDE_min_a_for_inequality_l943_94303

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^3 - 3*x + 3) - a * Real.exp x - x

theorem min_a_for_inequality :
  ∃ (a : ℝ), a = 1 - 1 / Real.exp 1 ∧
  (∀ (x : ℝ), x ≥ -2 → f a x ≤ 0) ∧
  (∀ (b : ℝ), (∀ (x : ℝ), x ≥ -2 → f b x ≤ 0) → b ≥ a) :=
sorry

end NUMINAMATH_CALUDE_min_a_for_inequality_l943_94303


namespace NUMINAMATH_CALUDE_allison_bought_28_items_l943_94322

/-- The number of glue sticks Marie bought -/
def marie_glue_sticks : ℕ := 15

/-- The number of construction paper packs Marie bought -/
def marie_paper_packs : ℕ := 30

/-- The difference in glue sticks between Allison and Marie -/
def glue_stick_difference : ℕ := 8

/-- The ratio of construction paper packs between Marie and Allison -/
def paper_pack_ratio : ℕ := 6

/-- The total number of craft supply items Allison bought -/
def allison_total_items : ℕ := marie_glue_sticks + glue_stick_difference + marie_paper_packs / paper_pack_ratio

theorem allison_bought_28_items : allison_total_items = 28 := by
  sorry

end NUMINAMATH_CALUDE_allison_bought_28_items_l943_94322


namespace NUMINAMATH_CALUDE_sin_monotone_interval_l943_94366

theorem sin_monotone_interval (t : ℝ) : 
  (∀ x ∈ Set.Icc (-t) t, StrictMono (fun x ↦ Real.sin (2 * x + π / 6))) ↔ 
  t ∈ Set.Ioo 0 (π / 6) := by
  sorry

end NUMINAMATH_CALUDE_sin_monotone_interval_l943_94366


namespace NUMINAMATH_CALUDE_right_triangle_from_number_and_reciprocal_l943_94343

theorem right_triangle_from_number_and_reciprocal (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let s := (a + 1/a) / 2
  let d := (a - 1/a) / 2
  let p := a * (1/a)
  s^2 = d^2 + p^2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_from_number_and_reciprocal_l943_94343


namespace NUMINAMATH_CALUDE_garden_land_ratio_l943_94380

/-- Represents a rectangle with width 3/5 of its length -/
structure Rectangle where
  length : ℝ
  width : ℝ
  width_prop : width = 3/5 * length

theorem garden_land_ratio (land garden : Rectangle) 
  (h : garden.length = 3/5 * land.length) :
  (garden.length * garden.width) / (land.length * land.width) = 36/100 := by
  sorry

end NUMINAMATH_CALUDE_garden_land_ratio_l943_94380


namespace NUMINAMATH_CALUDE_warehouse_cleaning_time_l943_94367

def lara_rate : ℚ := 1/4
def chris_rate : ℚ := 1/6
def break_time : ℚ := 2

theorem warehouse_cleaning_time (t : ℚ) : 
  (lara_rate + chris_rate) * (t - break_time) = 1 ↔ 
  t = (1 / (lara_rate + chris_rate)) + break_time :=
by sorry

end NUMINAMATH_CALUDE_warehouse_cleaning_time_l943_94367


namespace NUMINAMATH_CALUDE_range_of_m_l943_94372

def p (m : ℝ) : Prop := m < -1

def q (m : ℝ) : Prop := -2 < m ∧ m < 3

theorem range_of_m : 
  {m : ℝ | (p m ∨ q m) ∧ ¬(p m ∧ q m)} = 
  {m : ℝ | m ≤ -2 ∨ (-1 ≤ m ∧ m < 3)} := by sorry

end NUMINAMATH_CALUDE_range_of_m_l943_94372


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l943_94312

/-- Given two squares with perimeters 20 and 28, prove that a third square with side length
    equal to the positive difference of the side lengths of the first two squares has a perimeter of 8. -/
theorem square_perimeter_problem (square_I square_II square_III : ℝ → ℝ) :
  (∀ s, square_I s = 4 * s) →
  (∀ s, square_II s = 4 * s) →
  (∀ s, square_III s = 4 * s) →
  (∃ s_I, square_I s_I = 20) →
  (∃ s_II, square_II s_II = 28) →
  (∃ s_III, s_III = |s_I - s_II| ∧ square_III s_III = 8) :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l943_94312


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l943_94305

/-- Given a geometric sequence {aₙ}, prove that if a₅ - a₁ = 15 and a₄ - a₂ = 6,
    then (a₃ = 4 and q = 2) or (a₃ = -4 and q = 1/2) -/
theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 5 - a 1 = 15 →              -- First given condition
  a 4 - a 2 = 6 →               -- Second given condition
  ((a 3 = 4 ∧ q = 2) ∨ (a 3 = -4 ∧ q = 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l943_94305


namespace NUMINAMATH_CALUDE_first_subject_grade_l943_94390

/-- 
Given a student's grades in three subjects, prove that if the second subject is 60%,
the third subject is 70%, and the overall average is 60%, then the first subject's grade must be 50%.
-/
theorem first_subject_grade (grade1 : ℝ) (grade2 grade3 overall : ℝ) : 
  grade2 = 60 → grade3 = 70 → overall = 60 → (grade1 + grade2 + grade3) / 3 = overall → grade1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_first_subject_grade_l943_94390


namespace NUMINAMATH_CALUDE_particles_tend_to_unit_circle_l943_94370

/-- Velocity field of the fluid -/
def velocity_field (x y : ℝ) : ℝ × ℝ :=
  (y + 2*x - 2*x^3 - 2*x*y^2, -x)

/-- The rate of change of r^2 with respect to t -/
def r_squared_derivative (x y : ℝ) : ℝ :=
  2*x*(y + 2*x - 2*x^3 - 2*x*y^2) + 2*y*(-x)

/-- Theorem stating that particles tend towards the unit circle as t → ∞ -/
theorem particles_tend_to_unit_circle :
  ∀ (x y : ℝ), x ≠ 0 →
  (r_squared_derivative x y > 0 ↔ x^2 + y^2 < 1) ∧
  (r_squared_derivative x y < 0 ↔ x^2 + y^2 > 1) :=
sorry

end NUMINAMATH_CALUDE_particles_tend_to_unit_circle_l943_94370


namespace NUMINAMATH_CALUDE_taxi_fare_formula_l943_94351

/-- Represents the taxi fare function for distances greater than 3 km -/
def taxiFare (x : ℝ) : ℝ :=
  10 + 2 * (x - 3)

/-- Theorem stating that the taxi fare function is equivalent to 2x + 4 for x > 3 -/
theorem taxi_fare_formula (x : ℝ) (h : x > 3) :
  taxiFare x = 2 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_formula_l943_94351


namespace NUMINAMATH_CALUDE_band_members_formation_l943_94361

theorem band_members_formation :
  ∃! n : ℕ, 200 < n ∧ n < 300 ∧
  (∃ k : ℕ, n = 10 * k + 4) ∧
  (∃ m : ℕ, n = 12 * m + 6) := by
  sorry

end NUMINAMATH_CALUDE_band_members_formation_l943_94361


namespace NUMINAMATH_CALUDE_inequality_proof_l943_94309

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0) (non_neg_d : d ≥ 0)
  (sum_eq_four : a + b + c + d = 4) : 
  (a*b + c*d) * (a*c + b*d) * (a*d + b*c) ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l943_94309


namespace NUMINAMATH_CALUDE_children_on_bus_after_stop_l943_94346

/-- The number of children on a bus after a stop, given the initial number,
    the number who got on, and the relationship between those who got on and off. -/
theorem children_on_bus_after_stop
  (initial : ℕ)
  (got_on : ℕ)
  (h1 : initial = 28)
  (h2 : got_on = 82)
  (h3 : ∃ (got_off : ℕ), got_on = got_off + 2) :
  initial + got_on - (got_on - 2) = 28 := by
  sorry


end NUMINAMATH_CALUDE_children_on_bus_after_stop_l943_94346


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l943_94334

/-- Given a geometric sequence {a_n} where 3*a_5 = a_6 and a_2 = 1, prove a_4 = 9 -/
theorem geometric_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) -- geometric sequence condition
  (h2 : 3 * a 5 = a 6) -- given condition
  (h3 : a 2 = 1) -- given condition
  : a 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l943_94334


namespace NUMINAMATH_CALUDE_conference_handshakes_l943_94304

theorem conference_handshakes (n : ℕ) (h : n = 30) : (n * (n - 1)) / 2 = 435 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l943_94304


namespace NUMINAMATH_CALUDE_max_distinct_squares_sum_l943_94388

/-- The sum of squares of the first n positive integers -/
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- A function that checks if there exists a set of n distinct positive integers
    whose squares sum to 2531 -/
def exists_distinct_squares_sum (n : ℕ) : Prop :=
  ∃ (s : Finset ℕ), s.card = n ∧ (∀ x ∈ s, x > 0) ∧ (s.sum (λ x => x^2) = 2531)

theorem max_distinct_squares_sum :
  (∃ n : ℕ, exists_distinct_squares_sum n ∧
    ∀ m : ℕ, m > n → ¬exists_distinct_squares_sum m) ∧
  (∃ n : ℕ, exists_distinct_squares_sum n ∧ n = 18) := by
  sorry

end NUMINAMATH_CALUDE_max_distinct_squares_sum_l943_94388


namespace NUMINAMATH_CALUDE_age_ratio_proof_l943_94393

/-- Proves that given the conditions about A's and B's ages, the ratio between A's age 4 years hence and B's age 4 years ago is 3:1 -/
theorem age_ratio_proof (x : ℕ) (h1 : 5 * x > 4) (h2 : 3 * x > 4) : 
  (5 * x + 4) / (3 * x - 4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l943_94393


namespace NUMINAMATH_CALUDE_dislike_tv_and_books_l943_94374

theorem dislike_tv_and_books (total_population : ℕ) 
  (tv_dislike_percentage : ℚ) (book_dislike_percentage : ℚ) :
  total_population = 800 →
  tv_dislike_percentage = 25 / 100 →
  book_dislike_percentage = 15 / 100 →
  (tv_dislike_percentage * total_population : ℚ) * book_dislike_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_dislike_tv_and_books_l943_94374


namespace NUMINAMATH_CALUDE_max_abc_value_l943_94382

theorem max_abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b + c + a * b = (a + c) * (b + c))
  (h2 : a + b + c = 2) :
  a * b * c ≤ 8 / 27 :=
sorry

end NUMINAMATH_CALUDE_max_abc_value_l943_94382


namespace NUMINAMATH_CALUDE_hare_run_distance_l943_94332

/-- The distance between trees in meters -/
def tree_distance : ℕ := 5

/-- The number of the first tree -/
def first_tree : ℕ := 1

/-- The number of the last tree -/
def last_tree : ℕ := 10

/-- The total distance between the first and last tree -/
def total_distance : ℕ := tree_distance * (last_tree - first_tree)

theorem hare_run_distance :
  total_distance = 45 := by sorry

end NUMINAMATH_CALUDE_hare_run_distance_l943_94332


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l943_94313

def M : Set ℤ := {0, 1}
def N : Set ℤ := {-1, 0}

theorem intersection_of_M_and_N : M ∩ N = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l943_94313


namespace NUMINAMATH_CALUDE_multiplication_difference_l943_94318

theorem multiplication_difference (number : ℕ) (correct_multiplier : ℕ) (mistaken_multiplier : ℕ) :
  number = 135 →
  correct_multiplier = 43 →
  mistaken_multiplier = 34 →
  (number * correct_multiplier) - (number * mistaken_multiplier) = 1215 := by
sorry

end NUMINAMATH_CALUDE_multiplication_difference_l943_94318


namespace NUMINAMATH_CALUDE_janabel_widget_sales_l943_94395

theorem janabel_widget_sales (n : ℕ) (h : n = 15) : 
  let a₁ := 2
  let d := 3
  let aₙ := a₁ + (n - 1) * d
  n / 2 * (a₁ + aₙ) = 345 := by
  sorry

end NUMINAMATH_CALUDE_janabel_widget_sales_l943_94395


namespace NUMINAMATH_CALUDE_expand_product_l943_94306

theorem expand_product (x : ℝ) : 
  (2 * x^3 - 3 * x + 1) * (x^2 + 4 * x + 3) = 
  2 * x^5 + 8 * x^4 + 3 * x^3 - 11 * x^2 - 5 * x + 3 := by
sorry

end NUMINAMATH_CALUDE_expand_product_l943_94306


namespace NUMINAMATH_CALUDE_japanese_turtle_crane_problem_l943_94302

/-- Represents the number of cranes in the cage. -/
def num_cranes : ℕ := sorry

/-- Represents the number of turtles in the cage. -/
def num_turtles : ℕ := sorry

/-- The total number of heads in the cage. -/
def total_heads : ℕ := 35

/-- The total number of feet in the cage. -/
def total_feet : ℕ := 94

/-- The number of feet a crane has. -/
def crane_feet : ℕ := 2

/-- The number of feet a turtle has. -/
def turtle_feet : ℕ := 4

/-- Theorem stating that the system of equations correctly represents the Japanese turtle and crane problem. -/
theorem japanese_turtle_crane_problem :
  (num_cranes + num_turtles = total_heads) ∧
  (crane_feet * num_cranes + turtle_feet * num_turtles = total_feet) :=
sorry

end NUMINAMATH_CALUDE_japanese_turtle_crane_problem_l943_94302


namespace NUMINAMATH_CALUDE_sum_product_inequality_l943_94381

theorem sum_product_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) : 
  a * b + b * c + c * d + d * a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l943_94381


namespace NUMINAMATH_CALUDE_coat_price_calculations_l943_94391

def original_price : ℝ := 500
def initial_reduction : ℝ := 300
def discount1 : ℝ := 0.1
def discount2 : ℝ := 0.15

theorem coat_price_calculations :
  let percent_reduction := (initial_reduction / original_price) * 100
  let reduced_price := original_price - initial_reduction
  let percent_increase := ((original_price - reduced_price) / reduced_price) * 100
  let price_after_initial_reduction := reduced_price
  let price_after_discount1 := price_after_initial_reduction * (1 - discount1)
  let final_price := price_after_discount1 * (1 - discount2)
  (percent_reduction = 60 ∧
   percent_increase = 150 ∧
   final_price = 153) := by sorry

end NUMINAMATH_CALUDE_coat_price_calculations_l943_94391


namespace NUMINAMATH_CALUDE_total_distributions_l943_94324

def number_of_balls : ℕ := 8
def number_of_boxes : ℕ := 3

def valid_distribution (d : List ℕ) : Prop :=
  d.length = number_of_boxes ∧
  d.sum = number_of_balls ∧
  d.all (· > 0) ∧
  d.Pairwise (· ≠ ·)

def count_distributions : ℕ := sorry

theorem total_distributions :
  count_distributions = 2688 := by sorry

end NUMINAMATH_CALUDE_total_distributions_l943_94324


namespace NUMINAMATH_CALUDE_weekend_getaway_cost_sharing_l943_94311

/-- A weekend getaway cost-sharing problem -/
theorem weekend_getaway_cost_sharing 
  (henry_paid linda_paid jack_paid : ℝ)
  (h l : ℝ)
  (henry_paid_amount : henry_paid = 120)
  (linda_paid_amount : linda_paid = 150)
  (jack_paid_amount : jack_paid = 210)
  (total_cost : henry_paid + linda_paid + jack_paid = henry_paid + linda_paid + jack_paid)
  (even_split : (henry_paid + linda_paid + jack_paid) / 3 = henry_paid + h)
  (even_split' : (henry_paid + linda_paid + jack_paid) / 3 = linda_paid + l)
  : h - l = 30 := by sorry

end NUMINAMATH_CALUDE_weekend_getaway_cost_sharing_l943_94311


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_l943_94326

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 2| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x > 2/3 ∨ x < -6} := by sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (7/2)*t} = {t : ℝ | 3/2 ≤ t ∧ t ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_l943_94326


namespace NUMINAMATH_CALUDE_roadwork_truckloads_per_mile_l943_94379

theorem roadwork_truckloads_per_mile :
  let road_length : ℝ := 16
  let gravel_bags_per_truck : ℕ := 2
  let gravel_to_pitch_ratio : ℕ := 5
  let day1_miles : ℝ := 4
  let day2_miles : ℝ := 7
  let day3_pitch_barrels : ℕ := 6
  
  let total_paved_miles : ℝ := day1_miles + day2_miles
  let remaining_miles : ℝ := road_length - total_paved_miles
  let truckloads_per_mile : ℝ := day3_pitch_barrels / remaining_miles
  
  truckloads_per_mile = 1.2 := by sorry

end NUMINAMATH_CALUDE_roadwork_truckloads_per_mile_l943_94379


namespace NUMINAMATH_CALUDE_sequence_gcd_property_l943_94359

theorem sequence_gcd_property (a : ℕ → ℕ) :
  (∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) →
  ∀ i : ℕ, a i = i :=
by sorry

end NUMINAMATH_CALUDE_sequence_gcd_property_l943_94359


namespace NUMINAMATH_CALUDE_bodhi_cow_count_l943_94371

/-- Proves that the number of cows is 20 given the conditions of Mr. Bodhi's animal transportation problem -/
theorem bodhi_cow_count :
  let foxes : ℕ := 15
  let zebras : ℕ := 3 * foxes
  let sheep : ℕ := 20
  let total_animals : ℕ := 100
  ∃ cows : ℕ, cows + foxes + zebras + sheep = total_animals ∧ cows = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_bodhi_cow_count_l943_94371


namespace NUMINAMATH_CALUDE_carlson_problem_max_candies_l943_94369

/-- The maximum number of candies that can be eaten in the Carlson problem -/
def max_candies (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem stating the maximum number of candies for 32 initial ones -/
theorem carlson_problem_max_candies :
  max_candies 32 = 496 := by
  sorry

#eval max_candies 32  -- Should output 496

end NUMINAMATH_CALUDE_carlson_problem_max_candies_l943_94369


namespace NUMINAMATH_CALUDE_pascal_triangle_30_rows_sum_l943_94331

/-- The number of elements in the n-th row of Pascal's Triangle -/
def pascal_row_elements (n : ℕ) : ℕ := n + 1

/-- The sum of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_sum (n : ℕ) : ℕ :=
  (n * (pascal_row_elements 0 + pascal_row_elements (n - 1))) / 2

theorem pascal_triangle_30_rows_sum :
  pascal_triangle_sum 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_30_rows_sum_l943_94331


namespace NUMINAMATH_CALUDE_unique_x_exists_l943_94344

theorem unique_x_exists : ∃! x : ℝ, x > 0 ∧ x * ↑(⌊x⌋) = 50 ∧ |x - 7.142857| < 0.000001 := by sorry

end NUMINAMATH_CALUDE_unique_x_exists_l943_94344


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l943_94356

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 80 → 
  b = 150 → 
  c^2 = a^2 + b^2 → 
  c = 170 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l943_94356


namespace NUMINAMATH_CALUDE_max_polygon_area_l943_94345

/-- A point with integer coordinates satisfying the given conditions -/
structure ValidPoint where
  x : ℕ+
  y : ℕ+
  cond1 : x ∣ (2 * y + 1)
  cond2 : y ∣ (2 * x + 1)

/-- The set of all valid points -/
def ValidPoints : Set ValidPoint := {p : ValidPoint | True}

/-- The area of a polygon formed by a set of points -/
noncomputable def polygonArea (points : Set ValidPoint) : ℝ := sorry

/-- The maximum area of a polygon formed by valid points -/
theorem max_polygon_area :
  ∃ (points : Set ValidPoint), points ⊆ ValidPoints ∧ polygonArea points = 20 ∧
    ∀ (otherPoints : Set ValidPoint), otherPoints ⊆ ValidPoints →
      polygonArea otherPoints ≤ 20 := by sorry

end NUMINAMATH_CALUDE_max_polygon_area_l943_94345


namespace NUMINAMATH_CALUDE_equation_solution_l943_94310

theorem equation_solution : ∃! x : ℚ, (5 * x / (x + 3) - 3 / (x + 3) = 1 / (x + 3)) ∧ x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l943_94310


namespace NUMINAMATH_CALUDE_quadratic_fixed_point_l943_94364

/-- The quadratic function y = -x² + (m-1)x + m has a fixed point at (-1, 0) for all m -/
theorem quadratic_fixed_point :
  ∀ (m : ℝ), -(-1)^2 + (m - 1)*(-1) + m = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_fixed_point_l943_94364


namespace NUMINAMATH_CALUDE_power_inequality_l943_94316

theorem power_inequality (x : ℝ) (h : x < 27) : 27^9 > x^24 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l943_94316


namespace NUMINAMATH_CALUDE_saras_house_is_1000_l943_94377

def nadas_house_size : ℕ := 450

def saras_house_size (nadas_size : ℕ) : ℕ :=
  2 * nadas_size + 100

theorem saras_house_is_1000 : saras_house_size nadas_house_size = 1000 := by
  sorry

end NUMINAMATH_CALUDE_saras_house_is_1000_l943_94377


namespace NUMINAMATH_CALUDE_unique_positive_integer_l943_94323

theorem unique_positive_integer : ∃! (x : ℕ), x > 0 ∧ (4 * x)^2 - 2 * x = 2652 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_l943_94323


namespace NUMINAMATH_CALUDE_sum_squares_possible_values_l943_94337

/-- Given a positive real number A, prove that for any y in the open interval (0, A^2),
    there exists a sequence of positive real numbers {x_j} such that the sum of x_j equals A
    and the sum of x_j^2 equals y. -/
theorem sum_squares_possible_values (A : ℝ) (hA : A > 0) (y : ℝ) (hy1 : y > 0) (hy2 : y < A^2) :
  ∃ (x : ℕ → ℝ), (∀ j, x j > 0) ∧
    (∑' j, x j) = A ∧
    (∑' j, (x j)^2) = y :=
sorry

end NUMINAMATH_CALUDE_sum_squares_possible_values_l943_94337


namespace NUMINAMATH_CALUDE_fifth_rollercoaster_speed_l943_94333

/-- Theorem: Given 5 rollercoasters with specific speeds and average, prove the speed of the fifth rollercoaster -/
theorem fifth_rollercoaster_speed 
  (v₁ v₂ v₃ v₄ v₅ : ℝ) 
  (h1 : v₁ = 50)
  (h2 : v₂ = 62)
  (h3 : v₃ = 73)
  (h4 : v₄ = 70)
  (h_avg : (v₁ + v₂ + v₃ + v₄ + v₅) / 5 = 59) :
  v₅ = 40 := by
  sorry

end NUMINAMATH_CALUDE_fifth_rollercoaster_speed_l943_94333


namespace NUMINAMATH_CALUDE_max_value_2x_plus_y_l943_94362

theorem max_value_2x_plus_y (x y : ℝ) (h1 : x + 2*y ≤ 3) (h2 : y ≥ 0) :
  (∀ x' y', x' + 2*y' ≤ 3 → y' ≥ 0 → 2*x' + y' ≤ 6) ∧ 
  (∃ x₀ y₀, x₀ + 2*y₀ ≤ 3 ∧ y₀ ≥ 0 ∧ 2*x₀ + y₀ = 6) :=
by sorry

end NUMINAMATH_CALUDE_max_value_2x_plus_y_l943_94362


namespace NUMINAMATH_CALUDE_right_angle_and_trig_relation_l943_94363

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)
  (sum_angles : A + B + C = 180)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- Define the condition for right angle
def is_right_angled (t : Triangle) : Prop :=
  t.C = 90

-- Define the condition for equal sum of sine and cosine
def equal_sin_cos_sum (t : Triangle) : Prop :=
  Real.cos t.A + Real.sin t.A = Real.cos t.B + Real.sin t.B

-- Theorem statement
theorem right_angle_and_trig_relation (t : Triangle) :
  (is_right_angled t → equal_sin_cos_sum t) ∧
  ∃ t', equal_sin_cos_sum t' ∧ ¬is_right_angled t' :=
sorry

end NUMINAMATH_CALUDE_right_angle_and_trig_relation_l943_94363


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l943_94368

/-- Given a line with equation 3x - 4y + 5 = 0, its symmetric line with respect to the y-axis
    has the equation 3x + 4y - 5 = 0. -/
theorem symmetric_line_wrt_y_axis :
  ∀ (x y : ℝ), (3 * (-x) - 4 * y + 5 = 0) ↔ (3 * x + 4 * y - 5 = 0) := by
sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l943_94368


namespace NUMINAMATH_CALUDE_painting_job_completion_time_l943_94329

/-- Represents the painting job with given conditions -/
structure PaintingJob where
  original_men : ℕ
  original_days : ℕ
  additional_men : ℕ
  efficiency_increase : ℚ

/-- Calculates the number of days required to complete the job with additional skilled workers -/
def days_with_skilled_workers (job : PaintingJob) : ℚ :=
  let total_man_days := job.original_men * job.original_days
  let original_daily_output := job.original_men
  let skilled_daily_output := job.additional_men * (1 + job.efficiency_increase)
  let total_daily_output := original_daily_output + skilled_daily_output
  total_man_days / total_daily_output

/-- The main theorem stating that the job will be completed in 4 days -/
theorem painting_job_completion_time :
  let job := PaintingJob.mk 10 6 4 (1/4)
  days_with_skilled_workers job = 4 := by
  sorry

#eval days_with_skilled_workers (PaintingJob.mk 10 6 4 (1/4))

end NUMINAMATH_CALUDE_painting_job_completion_time_l943_94329


namespace NUMINAMATH_CALUDE_rotation_result_l943_94398

-- Define the shapes
inductive Shape
| Triangle
| SmallCircle
| Pentagon

-- Define the positions
inductive Position
| Top
| LowerLeft
| LowerRight

-- Define the configuration as a function from Shape to Position
def Configuration := Shape → Position

-- Define the initial configuration
def initial_config : Configuration
| Shape.Triangle => Position.Top
| Shape.SmallCircle => Position.LowerLeft
| Shape.Pentagon => Position.LowerRight

-- Define the rotation function
def rotate_150_clockwise (config : Configuration) : Configuration :=
  fun shape =>
    match config shape with
    | Position.Top => Position.LowerRight
    | Position.LowerLeft => Position.Top
    | Position.LowerRight => Position.LowerLeft

-- Theorem statement
theorem rotation_result :
  let final_config := rotate_150_clockwise initial_config
  final_config Shape.Triangle = Position.LowerRight ∧
  final_config Shape.SmallCircle = Position.Top ∧
  final_config Shape.Pentagon = Position.LowerLeft :=
sorry

end NUMINAMATH_CALUDE_rotation_result_l943_94398


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l943_94389

/-- A conic section type -/
inductive ConicType
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola
  | None

/-- Determines the type of conic section from its equation -/
def determineConicType (f : ℝ → ℝ → ℝ) : ConicType :=
  sorry

/-- The equation of the conic section -/
def conicEquation (x y : ℝ) : ℝ :=
  (x - 3)^2 - 2*(y + 1)^2 - 50

theorem conic_is_hyperbola :
  determineConicType conicEquation = ConicType.Hyperbola :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l943_94389


namespace NUMINAMATH_CALUDE_faye_candy_problem_l943_94373

/-- Calculates the number of candy pieces Faye's sister gave her -/
def candy_from_sister (initial : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial - eaten)

/-- Theorem stating that given the problem conditions, Faye's sister gave her 40 pieces of candy -/
theorem faye_candy_problem (initial eaten final : ℕ) 
  (h_initial : initial = 47)
  (h_eaten : eaten = 25)
  (h_final : final = 62) :
  candy_from_sister initial eaten final = 40 := by
  sorry

end NUMINAMATH_CALUDE_faye_candy_problem_l943_94373


namespace NUMINAMATH_CALUDE_regular_hexagonal_prism_sum_l943_94348

/-- A regular hexagonal prism -/
structure RegularHexagonalPrism where
  /-- The number of faces of the prism -/
  faces : ℕ
  /-- The number of edges of the prism -/
  edges : ℕ
  /-- The number of vertices of the prism -/
  vertices : ℕ

/-- The sum of faces, edges, and vertices of a regular hexagonal prism is 38 -/
theorem regular_hexagonal_prism_sum (prism : RegularHexagonalPrism) :
  prism.faces + prism.edges + prism.vertices = 38 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagonal_prism_sum_l943_94348


namespace NUMINAMATH_CALUDE_basketball_tournament_l943_94325

theorem basketball_tournament (x : ℕ) : 
  (3 * x / 4 : ℚ) = x - x / 4 ∧ 
  (2 * (x + 4) / 3 : ℚ) = (x + 4) - (x + 4) / 3 ∧ 
  (2 * (x + 4) / 3 : ℚ) = 3 * x / 4 + 9 ∧ 
  ((x + 4) / 3 : ℚ) = x / 4 + 5 → 
  x = 76 := by
sorry

end NUMINAMATH_CALUDE_basketball_tournament_l943_94325


namespace NUMINAMATH_CALUDE_happy_family_cows_count_cow_ratio_l943_94399

/-- The number of cows We the People has -/
def we_the_people_cows : ℕ := 17

/-- The total number of cows when both groups are together -/
def total_cows : ℕ := 70

/-- The number of cows Happy Good Healthy Family has -/
def happy_family_cows : ℕ := total_cows - we_the_people_cows

theorem happy_family_cows_count : happy_family_cows = 53 := by
  sorry

theorem cow_ratio : 
  (happy_family_cows : ℚ) / (we_the_people_cows : ℚ) = 53 / 17 := by
  sorry

end NUMINAMATH_CALUDE_happy_family_cows_count_cow_ratio_l943_94399


namespace NUMINAMATH_CALUDE_complex_equation_solution_l943_94308

theorem complex_equation_solution (a b : ℝ) :
  (Complex.mk 1 2) / (Complex.mk a b) = Complex.mk 1 1 →
  a = (3 : ℝ) / 2 ∧ b = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l943_94308


namespace NUMINAMATH_CALUDE_parallel_and_perpendicular_relations_l943_94307

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)
variable (line_perpendicular_to_plane : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- State that m and n are different lines
variable (m_ne_n : m ≠ n)

-- State that α, β, and γ are different planes
variable (α_ne_β : α ≠ β)
variable (α_ne_γ : α ≠ γ)
variable (β_ne_γ : β ≠ γ)

-- Define the theorem
theorem parallel_and_perpendicular_relations :
  (∀ (a b c : Plane), parallel_planes a c → parallel_planes b c → parallel_planes a b) ∧
  (∀ (l1 l2 : Line) (p1 p2 : Plane), 
    line_perpendicular_to_plane l1 p1 → 
    line_perpendicular_to_plane l2 p2 → 
    parallel_planes p1 p2 → 
    parallel_lines l1 l2) :=
sorry

end NUMINAMATH_CALUDE_parallel_and_perpendicular_relations_l943_94307


namespace NUMINAMATH_CALUDE_andrews_to_jeffreys_steps_ratio_l943_94336

theorem andrews_to_jeffreys_steps_ratio : 
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ 150 * b = 200 * a ∧ a = 3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_andrews_to_jeffreys_steps_ratio_l943_94336
