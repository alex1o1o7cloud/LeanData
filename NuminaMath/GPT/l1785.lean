import Mathlib

namespace NUMINAMATH_GPT_determine_event_C_l1785_178568

variable (A B C : Prop)
variable (Tallest Shortest : Prop)
variable (Running LongJump ShotPut : Prop)

variables (part_A_Running part_A_LongJump part_A_ShotPut
           part_B_Running part_B_LongJump part_B_ShotPut
           part_C_Running part_C_LongJump part_C_ShotPut : Prop)

variable (not_tallest_A : ¬Tallest → A)
variable (not_tallest_ShotPut : Tallest → ¬ShotPut)
variable (shortest_LongJump : Shortest → LongJump)
variable (not_shortest_B : ¬Shortest → B)
variable (not_running_B : ¬Running → B)

theorem determine_event_C :
  (¬Tallest → A) →
  (Tallest → ¬ShotPut) →
  (Shortest → LongJump) →
  (¬Shortest → B) →
  (¬Running → B) →
  part_C_Running :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_determine_event_C_l1785_178568


namespace NUMINAMATH_GPT_number_of_students_like_photography_l1785_178578

variable (n_dislike n_like n_neutral : ℕ)

theorem number_of_students_like_photography :
  (3 * n_dislike = n_dislike + 12) →
  (5 * n_dislike = n_like) →
  n_like = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_like_photography_l1785_178578


namespace NUMINAMATH_GPT_find_area_of_triangle_ABQ_l1785_178525

noncomputable def area_triangle_ABQ {A B C P Q R : Type*}
  (AP PB : ℝ) (area_ABC area_ABQ : ℝ) (h_areas_equal : area_ABQ = 15 / 2)
  (h_triangle_area : area_ABC = 15) (h_AP : AP = 3) (h_PB : PB = 2)
  (h_AB : AB = AP + PB) : Prop := area_ABQ = 15

theorem find_area_of_triangle_ABQ
  (A B C P Q R : Type*) (AP PB : ℝ)
  (h_triangle_area : area_ABC = 15)
  (h_AP : AP = 3) (h_PB : PB = 2)
  (h_AB : AB = AP + PB) (h_areas_equal : area_ABQ = 15 / 2) :
  area_ABQ = 15 := sorry

end NUMINAMATH_GPT_find_area_of_triangle_ABQ_l1785_178525


namespace NUMINAMATH_GPT_solution_set_inequality_l1785_178531

theorem solution_set_inequality (x : ℝ) (h : x ≠ 0) : 
  (x - 1) / x > 1 → x < 0 := 
by 
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1785_178531


namespace NUMINAMATH_GPT_find_value_divide_subtract_l1785_178572

theorem find_value_divide_subtract :
  (Number = 8 * 156 + 2) → 
  (CorrectQuotient = Number / 5) → 
  (Value = CorrectQuotient - 3) → 
  Value = 247 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_find_value_divide_subtract_l1785_178572


namespace NUMINAMATH_GPT_range_of_a_l1785_178584

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - 4 * x + 3 < 0 ∧ 2^(1 - x) + a ≤ 0 ∧ x^2 - 2 * (a + 7) * x + 5 ≤ 0 ) ↔ (-4 ≤ a ∧ a ≤ -1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1785_178584


namespace NUMINAMATH_GPT_bill_original_selling_price_l1785_178547

variable (P : ℝ) (S : ℝ) (S_new : ℝ)

theorem bill_original_selling_price :
  (S = P + 0.10 * P) ∧ (S_new = 0.90 * P + 0.27 * P) ∧ (S_new = S + 28) →
  S = 440 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_bill_original_selling_price_l1785_178547


namespace NUMINAMATH_GPT_percentage_of_men_l1785_178528

theorem percentage_of_men (M W : ℝ) (h1 : M + W = 1) (h2 : 0.60 * M + 0.2364 * W = 0.40) : M = 0.45 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_men_l1785_178528


namespace NUMINAMATH_GPT_verify_option_a_l1785_178544

-- Define Option A's condition
def option_a_condition (a : ℝ) : Prop :=
  2 * a^2 - 4 * a + 2 = 2 * (a - 1)^2

-- State the theorem that Option A's factorization is correct
theorem verify_option_a (a : ℝ) : option_a_condition a := by sorry

end NUMINAMATH_GPT_verify_option_a_l1785_178544


namespace NUMINAMATH_GPT_fraction_sum_l1785_178522

theorem fraction_sum (y : ℝ) (a b : ℤ) (h : y = 3.834834834) (h_frac : y = (a : ℝ) / b) (h_coprime : Int.gcd a b = 1) : a + b = 4830 :=
sorry

end NUMINAMATH_GPT_fraction_sum_l1785_178522


namespace NUMINAMATH_GPT_range_of_m_l1785_178557

variable (x m : ℝ)

def alpha (x : ℝ) : Prop := x ≤ -5
def beta (x m : ℝ) : Prop := 2 * m - 3 ≤ x ∧ x ≤ 2 * m + 1

theorem range_of_m (x : ℝ) : (∀ x, beta x m → alpha x) → m ≤ -3 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l1785_178557


namespace NUMINAMATH_GPT_reinforcement_size_l1785_178587

theorem reinforcement_size (initial_men : ℕ) (initial_days : ℕ) (days_before_reinforcement : ℕ) (days_remaining : ℕ) (reinforcement : ℕ) : 
  initial_men = 150 → initial_days = 31 → days_before_reinforcement = 16 → days_remaining = 5 → (150 * 15) = (150 + reinforcement) * 5 → reinforcement = 300 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_reinforcement_size_l1785_178587


namespace NUMINAMATH_GPT_meat_cost_per_pound_l1785_178553

def total_cost_box : ℝ := 5
def cost_per_bell_pepper : ℝ := 1.5
def num_bell_peppers : ℝ := 4
def num_pounds_meat : ℝ := 2
def total_spent : ℝ := 17

theorem meat_cost_per_pound : total_spent - (total_cost_box + num_bell_peppers * cost_per_bell_pepper) = 6 -> 
                             6 / num_pounds_meat = 3 := by
  sorry

end NUMINAMATH_GPT_meat_cost_per_pound_l1785_178553


namespace NUMINAMATH_GPT_problem_equivalent_l1785_178548

noncomputable def h (y : ℝ) : ℝ := y^5 - y^3 + 2
noncomputable def k (y : ℝ) : ℝ := y^2 - 3

theorem problem_equivalent (y₁ y₂ y₃ y₄ y₅ : ℝ) (h_roots : ∀ y, h y = 0 ↔ y = y₁ ∨ y = y₂ ∨ y = y₃ ∨ y = y₄ ∨ y = y₅) :
  (k y₁) * (k y₂) * (k y₃) * (k y₄) * (k y₅) = 104 :=
sorry

end NUMINAMATH_GPT_problem_equivalent_l1785_178548


namespace NUMINAMATH_GPT_kevin_prizes_l1785_178595

theorem kevin_prizes (total_prizes stuffed_animals yo_yos frisbees : ℕ)
  (h1 : total_prizes = 50) (h2 : stuffed_animals = 14) (h3 : yo_yos = 18) :
  frisbees = total_prizes - (stuffed_animals + yo_yos) → frisbees = 18 :=
by
  intro h4
  sorry

end NUMINAMATH_GPT_kevin_prizes_l1785_178595


namespace NUMINAMATH_GPT_parallel_case_perpendicular_case_l1785_178591

variables (m : ℝ)
def a := (2, -1)
def b := (-1, m)
def c := (-1, 2)
def sum_ab := (1, m - 1)

-- Parallel case (dot product is zero)
theorem parallel_case : (sum_ab m).fst * c.fst + (sum_ab m).snd * c.snd = 0 ↔ m = -1 :=
by
  sorry

-- Perpendicular case (dot product is zero)
theorem perpendicular_case : (sum_ab m).fst * c.fst + (sum_ab m).snd * c.snd = 0 ↔ m = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_case_perpendicular_case_l1785_178591


namespace NUMINAMATH_GPT_total_time_to_row_l1785_178592

theorem total_time_to_row (boat_speed_in_still_water : ℝ) (stream_speed : ℝ) (distance : ℝ) :
  boat_speed_in_still_water = 9 → stream_speed = 1.5 → distance = 105 → 
  (distance / (boat_speed_in_still_water + stream_speed)) + (distance / (boat_speed_in_still_water - stream_speed)) = 24 :=
by
  intro h_boat_speed h_stream_speed h_distance
  rw [h_boat_speed, h_stream_speed, h_distance]
  sorry

end NUMINAMATH_GPT_total_time_to_row_l1785_178592


namespace NUMINAMATH_GPT_decimal_to_base7_conversion_l1785_178571

theorem decimal_to_base7_conversion :
  (2023 : ℕ) = 5 * (7^3) + 6 * (7^2) + 2 * (7^1) + 0 * (7^0) :=
by
  sorry

end NUMINAMATH_GPT_decimal_to_base7_conversion_l1785_178571


namespace NUMINAMATH_GPT_value_of_b_plus_a_l1785_178582

theorem value_of_b_plus_a (a b : ℝ) (h1 : |a| = 8) (h2 : |b| = 2) (h3 : |a - b| = |b - a|) : b + a = -6 ∨ b + a = -10 :=
by
  sorry

end NUMINAMATH_GPT_value_of_b_plus_a_l1785_178582


namespace NUMINAMATH_GPT_percentage_of_singles_l1785_178596

/-- In a baseball season, Lisa had 50 hits. Among her hits were 2 home runs, 
2 triples, 8 doubles, and 1 quadruple. The rest of her hits were singles. 
What percent of her hits were singles? --/
theorem percentage_of_singles
  (total_hits : ℕ := 50)
  (home_runs : ℕ := 2)
  (triples : ℕ := 2)
  (doubles : ℕ := 8)
  (quadruples : ℕ := 1)
  (non_singles := home_runs + triples + doubles + quadruples)
  (singles := total_hits - non_singles) :
  (singles : ℚ) / (total_hits : ℚ) * 100 = 74 := by
  sorry

end NUMINAMATH_GPT_percentage_of_singles_l1785_178596


namespace NUMINAMATH_GPT_range_of_s_l1785_178554

noncomputable def s (x : ℝ) := 1 / (2 + x)^3

theorem range_of_s :
  Set.range s = {y : ℝ | y < 0} ∪ {y : ℝ | y > 0} :=
by
  sorry

end NUMINAMATH_GPT_range_of_s_l1785_178554


namespace NUMINAMATH_GPT_billy_restaurant_bill_l1785_178534

def adults : ℕ := 2
def children : ℕ := 5
def meal_cost : ℕ := 3

def total_people : ℕ := adults + children
def total_bill : ℕ := total_people * meal_cost

theorem billy_restaurant_bill : total_bill = 21 := 
by
  -- This is the placeholder for the proof.
  sorry

end NUMINAMATH_GPT_billy_restaurant_bill_l1785_178534


namespace NUMINAMATH_GPT_total_number_of_orders_l1785_178598

-- Define the conditions
def num_original_programs : Nat := 6
def num_added_programs : Nat := 3

-- State the theorem
theorem total_number_of_orders : ∃ n : ℕ, n = 210 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_total_number_of_orders_l1785_178598


namespace NUMINAMATH_GPT_number_of_correct_propositions_is_zero_l1785_178539

-- Defining the propositions as functions
def proposition1 (f : ℝ → ℝ) (increasing_pos : ∀ x > 0, f x ≤ f (x + 1))
  (increasing_neg : ∀ x < 0, f x ≤ f (x + 1)) : Prop :=
  ∀ x1 x2, x1 ≤ x2 → f x1 ≤ f x2

def proposition2 (a b : ℝ) (no_intersection : ∀ x, a * x^2 + b * x + 2 ≠ 0) : Prop :=
  b^2 < 8 * a ∧ (a > 0 ∨ (a = 0 ∧ b = 0))

def proposition3 : Prop :=
  ∀ x, (x ≥ 1 → (x^2 - 2 * x - 3) ≥ (x^2 - 2 * (x + 1) - 3))

-- The main theorem to prove
theorem number_of_correct_propositions_is_zero :
  ∀ (f : ℝ → ℝ)
    (increasing_pos : ∀ x > 0, f x ≤ f (x + 1))
    (increasing_neg : ∀ x < 0, f x ≤ f (x + 1))
    (a b : ℝ)
    (no_intersection : ∀ x, a * x^2 + b * x + 2 ≠ 0),
    (¬ proposition1 f increasing_pos increasing_neg ∧
     ¬ proposition2 a b no_intersection ∧
     ¬ proposition3) :=
by
  sorry

end NUMINAMATH_GPT_number_of_correct_propositions_is_zero_l1785_178539


namespace NUMINAMATH_GPT_sum_of_distinct_digits_l1785_178507

theorem sum_of_distinct_digits
  (w x y z : ℕ)
  (h1 : y + w = 10)
  (h2 : x + y = 9)
  (h3 : w + z = 10)
  (h4 : w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z)
  (hw : w < 10) (hx : x < 10) (hy : y < 10) (hz : z < 10) :
  w + x + y + z = 20 := sorry

end NUMINAMATH_GPT_sum_of_distinct_digits_l1785_178507


namespace NUMINAMATH_GPT_expression_evaluation_l1785_178508

theorem expression_evaluation (a b c d : ℤ) : 
  a / b - c * d^2 = a / (b - c * d^2) :=
sorry

end NUMINAMATH_GPT_expression_evaluation_l1785_178508


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1785_178551

theorem sufficient_but_not_necessary (m : ℕ) :
  m = 9 → m > 8 ∧ ∃ k : ℕ, k > 8 ∧ k ≠ 9 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1785_178551


namespace NUMINAMATH_GPT_calories_in_dressing_l1785_178515

noncomputable def lettuce_calories : ℝ := 50
noncomputable def carrot_calories : ℝ := 2 * lettuce_calories
noncomputable def crust_calories : ℝ := 600
noncomputable def pepperoni_calories : ℝ := crust_calories / 3
noncomputable def cheese_calories : ℝ := 400

noncomputable def salad_calories : ℝ := lettuce_calories + carrot_calories
noncomputable def pizza_calories : ℝ := crust_calories + pepperoni_calories + cheese_calories

noncomputable def salad_eaten : ℝ := salad_calories / 4
noncomputable def pizza_eaten : ℝ := pizza_calories / 5

noncomputable def total_eaten : ℝ := salad_eaten + pizza_eaten

theorem calories_in_dressing : ((330 : ℝ) - total_eaten) = 52.5 := by
  sorry

end NUMINAMATH_GPT_calories_in_dressing_l1785_178515


namespace NUMINAMATH_GPT_percent_of_y_l1785_178574

theorem percent_of_y (y : ℝ) (hy : y > 0) : (8 * y) / 20 + (3 * y) / 10 = 0.7 * y :=
by
  sorry

end NUMINAMATH_GPT_percent_of_y_l1785_178574


namespace NUMINAMATH_GPT_treehouse_total_planks_l1785_178593

theorem treehouse_total_planks (T : ℕ) 
    (h1 : T / 4 + T / 2 + 20 + 30 = T) : T = 200 :=
sorry

end NUMINAMATH_GPT_treehouse_total_planks_l1785_178593


namespace NUMINAMATH_GPT_trapezoid_perimeter_l1785_178541

theorem trapezoid_perimeter (height : ℝ) (radius : ℝ) (LM KN : ℝ) (LM_eq : LM = 16.5) (KN_eq : KN = 37.5)
  (LK MN : ℝ) (LK_eq : LK = 37.5) (MN_eq : MN = 37.5) (H : height = 36) (R : radius = 11) : 
  (LM + KN + LK + MN) = 129 :=
by
  -- The proof is omitted; only the statement is provided as specified.
  sorry

end NUMINAMATH_GPT_trapezoid_perimeter_l1785_178541


namespace NUMINAMATH_GPT_probability_of_fourth_roll_l1785_178545

-- Define the conditions 
structure Die :=
(fair : Bool) 
(biased_six : Bool)
(biased_one : Bool)

-- Define the probability function
def roll_prob (d : Die) (f : Bool) : ℚ :=
  if d.fair then 1/6
  else if d.biased_six then if f then 1/2 else 1/10
  else if d.biased_one then if f then 1/10 else 1/5
  else 0

def probability_of_fourth_six (p q : ℕ) (r1 r2 r3 : Bool) (d : Die) : ℚ :=
  (if r1 && r2 && r3 then roll_prob d true else 0) 

noncomputable def final_probability (d1 d2 d3 : Die) (prob_fair distorted_rolls : Bool) : ℚ :=
  let fair_prob := if distorted_rolls then roll_prob d1 true else roll_prob d1 false
  let biased_six_prob := if distorted_rolls then roll_prob d2 true else roll_prob d2 false
  let total_prob := fair_prob + biased_six_prob
  let fair := fair_prob / total_prob
  let biased_six := biased_six_prob / total_prob
  fair * roll_prob d1 true + biased_six * roll_prob d2 true

theorem probability_of_fourth_roll
  (d1 : Die) (d2 : Die) (d3 : Die)
  (h1 : d1.fair = true)
  (h2 : d2.biased_six = true)
  (h3 : d3.biased_one = true)
  (h4 : ∀ d, d1 = d ∨ d2 = d ∨ d3 = d)
  (r1 r2 r3 : Bool)
  : ∃ p q : ℕ, p + q = 11 ∧ final_probability d1 d2 d3 true = 5/6 := 
sorry

end NUMINAMATH_GPT_probability_of_fourth_roll_l1785_178545


namespace NUMINAMATH_GPT_volume_of_rectangular_solid_l1785_178555

theorem volume_of_rectangular_solid
  (a b c : ℝ)
  (h1 : a * b = 3)
  (h2 : a * c = 5)
  (h3 : b * c = 15) :
  a * b * c = 15 :=
sorry

end NUMINAMATH_GPT_volume_of_rectangular_solid_l1785_178555


namespace NUMINAMATH_GPT_xy_difference_l1785_178537

theorem xy_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end NUMINAMATH_GPT_xy_difference_l1785_178537


namespace NUMINAMATH_GPT_total_sum_of_rupees_l1785_178594

theorem total_sum_of_rupees :
  ∃ (total_coins : ℕ) (paise20_coins : ℕ) (paise25_coins : ℕ),
    total_coins = 344 ∧ paise20_coins = 300 ∧ paise25_coins = total_coins - paise20_coins ∧
    (60 + (44 * 0.25)) = 71 :=
by
  sorry

end NUMINAMATH_GPT_total_sum_of_rupees_l1785_178594


namespace NUMINAMATH_GPT_tan_7pi_over_6_l1785_178558

noncomputable def tan_periodic (θ : ℝ) : Prop :=
  ∀ k : ℤ, Real.tan (θ + k * Real.pi) = Real.tan θ

theorem tan_7pi_over_6 : Real.tan (7 * Real.pi / 6) = Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_7pi_over_6_l1785_178558


namespace NUMINAMATH_GPT_algebraic_expression_evaluation_l1785_178523

theorem algebraic_expression_evaluation (a : ℝ) (h : a^2 + 2 * a - 1 = 0) :
  ( ( (a^2 - 1) / (a^2 - 2 * a + 1) - 1 / (1 - a)) / (1 / (a^2 - a)) ) = 1 :=
by sorry

end NUMINAMATH_GPT_algebraic_expression_evaluation_l1785_178523


namespace NUMINAMATH_GPT_find_triplets_l1785_178524

noncomputable def triplets_solution (x y z : ℝ) : Prop := 
  (x^2 + y^2 = -x + 3*y + z) ∧ 
  (y^2 + z^2 = x + 3*y - z) ∧ 
  (x^2 + z^2 = 2*x + 2*y - z) ∧ 
  (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z)

theorem find_triplets : 
  { (x, y, z) : ℝ × ℝ × ℝ | triplets_solution x y z } = 
  { (0, 1, -2), (-3/2, 5/2, -1/2) } :=
sorry

end NUMINAMATH_GPT_find_triplets_l1785_178524


namespace NUMINAMATH_GPT_calculate_discount_percentage_l1785_178590

theorem calculate_discount_percentage :
  ∃ (x : ℝ), (∀ (P S : ℝ),
    (S = 439.99999999999966) →
    (S = 1.10 * P) →
    (1.30 * (1 - x / 100) * P = S + 28) →
    x = 10) :=
sorry

end NUMINAMATH_GPT_calculate_discount_percentage_l1785_178590


namespace NUMINAMATH_GPT_find_a3_a4_a5_l1785_178560

variable (a : ℕ → ℝ)

-- Recurrence relation for the sequence (condition for n ≥ 2)
axiom rec_relation (n : ℕ) (h : n ≥ 2) : 2 * a n = a (n - 1) + a (n + 1)

-- Additional conditions
axiom cond1 : a 1 + a 3 + a 5 = 9
axiom cond2 : a 3 + a 5 + a 7 = 15

-- Statement to prove
theorem find_a3_a4_a5 : a 3 + a 4 + a 5 = 12 :=
  sorry

end NUMINAMATH_GPT_find_a3_a4_a5_l1785_178560


namespace NUMINAMATH_GPT_range_of_abscissa_l1785_178576

/--
Given three points A, F1, F2 in the Cartesian plane and a point P satisfying the given conditions,
prove that the range of the abscissa of point P is [0, 3].

Conditions:
- A = (1, 0)
- F1 = (-2, 0)
- F2 = (2, 0)
- \| overrightarrow{PF1} \| + \| overrightarrow{PF2} \| = 6
- \| overrightarrow{PA} \| ≤ sqrt(6)
-/
theorem range_of_abscissa :
  ∀ (P : ℝ × ℝ),
    (|P.1 + 2| + |P.1 - 2| = 6) →
    ((P.1 - 1)^2 + P.2^2 ≤ 6) →
    (0 ≤ P.1 ∧ P.1 ≤ 3) :=
by
  intros P H1 H2
  sorry

end NUMINAMATH_GPT_range_of_abscissa_l1785_178576


namespace NUMINAMATH_GPT_manfred_average_paycheck_l1785_178577

def average_paycheck : ℕ → ℕ → ℕ → ℕ := fun total_paychecks first_paychecks_value num_first_paychecks =>
  let remaining_paychecks_value := first_paychecks_value + 20
  let total_payment := (num_first_paychecks * first_paychecks_value) + ((total_paychecks - num_first_paychecks) * remaining_paychecks_value)
  let average_payment := total_payment / total_paychecks
  average_payment

theorem manfred_average_paycheck :
  average_paycheck 26 750 6 = 765 := by
  sorry

end NUMINAMATH_GPT_manfred_average_paycheck_l1785_178577


namespace NUMINAMATH_GPT_jake_reaches_ground_later_by_2_seconds_l1785_178501

noncomputable def start_floor : ℕ := 12
noncomputable def steps_per_floor : ℕ := 25
noncomputable def jake_steps_per_second : ℕ := 3
noncomputable def elevator_B_time : ℕ := 90

noncomputable def total_steps_jake := (start_floor - 1) * steps_per_floor
noncomputable def time_jake := (total_steps_jake + jake_steps_per_second - 1) / jake_steps_per_second
noncomputable def time_difference := time_jake - elevator_B_time

theorem jake_reaches_ground_later_by_2_seconds :
  time_difference = 2 := by
  sorry

end NUMINAMATH_GPT_jake_reaches_ground_later_by_2_seconds_l1785_178501


namespace NUMINAMATH_GPT_prove_identical_numbers_l1785_178532

variable {x y : ℝ}

theorem prove_identical_numbers (hx : x ≠ 0) (hy : y ≠ 0)
    (h1 : x + (1 / y^2) = y + (1 / x^2))
    (h2 : y^2 + (1 / x) = x^2 + (1 / y)) : x = y :=
by 
  sorry

end NUMINAMATH_GPT_prove_identical_numbers_l1785_178532


namespace NUMINAMATH_GPT_find_f_property_l1785_178556

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_property :
  (f 0 = 3) ∧ (∀ x y : ℝ, f (xy) = f ((x^2 + y^2) / 2) + (x - y)^2) →
  (∀ x : ℝ, 0 ≤ x → f x = 3 - 2 * x) :=
by
  intros hypothesis
  -- Proof would be placed here
  sorry

end NUMINAMATH_GPT_find_f_property_l1785_178556


namespace NUMINAMATH_GPT_no_such_integers_exists_l1785_178502

theorem no_such_integers_exists 
  (a b c d : ℤ) 
  (h1 : a * 19^3 + b * 19^2 + c * 19 + d = 1) 
  (h2 : a * 62^3 + b * 62^2 + c * 62 + d = 2) : 
  false :=
by
  sorry

end NUMINAMATH_GPT_no_such_integers_exists_l1785_178502


namespace NUMINAMATH_GPT_union_complement_eq_l1785_178561

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {-1, 0, 3}

theorem union_complement_eq :
  A ∪ (U \ B) = {-2, -1, 0, 1, 2} := by
  sorry

end NUMINAMATH_GPT_union_complement_eq_l1785_178561


namespace NUMINAMATH_GPT_count_sums_of_three_cubes_l1785_178535

theorem count_sums_of_three_cubes :
  let possible_sums := {n | ∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ n = a^3 + b^3 + c^3}
  ∃ unique_sums : Finset ℕ, (∀ x ∈ possible_sums, x < 1000) ∧ unique_sums.card = 153 :=
by sorry

end NUMINAMATH_GPT_count_sums_of_three_cubes_l1785_178535


namespace NUMINAMATH_GPT_net_change_over_week_l1785_178505

-- Definitions of initial quantities on Day 1
def baking_powder_day1 : ℝ := 4
def flour_day1 : ℝ := 12
def sugar_day1 : ℝ := 10
def chocolate_chips_day1 : ℝ := 6

-- Definitions of final quantities on Day 7
def baking_powder_day7 : ℝ := 2.5
def flour_day7 : ℝ := 7
def sugar_day7 : ℝ := 6.5
def chocolate_chips_day7 : ℝ := 3.7

-- Definitions of changes in quantities
def change_baking_powder : ℝ := baking_powder_day1 - baking_powder_day7
def change_flour : ℝ := flour_day1 - flour_day7
def change_sugar : ℝ := sugar_day1 - sugar_day7
def change_chocolate_chips : ℝ := chocolate_chips_day1 - chocolate_chips_day7

-- Statement to prove
theorem net_change_over_week : change_baking_powder + change_flour + change_sugar + change_chocolate_chips = 12.3 :=
by
  -- (Proof omitted)
  sorry

end NUMINAMATH_GPT_net_change_over_week_l1785_178505


namespace NUMINAMATH_GPT_geometric_progression_common_ratio_l1785_178509

theorem geometric_progression_common_ratio (r : ℝ) :
  (r > 0) ∧ (r^3 + r^2 + r - 1 = 0) ↔
  r = ( -1 + ((19 + 3 * Real.sqrt 33)^(1/3)) + ((19 - 3 * Real.sqrt 33)^(1/3)) ) / 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_common_ratio_l1785_178509


namespace NUMINAMATH_GPT_max_value_fraction_l1785_178510

theorem max_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x / (2 * x + y) + y / (x + 2 * y)) <= (2 / 3) := 
sorry

end NUMINAMATH_GPT_max_value_fraction_l1785_178510


namespace NUMINAMATH_GPT_pharmacy_incurs_loss_l1785_178564

variable (a b : ℝ)
variable (h : a < b)

theorem pharmacy_incurs_loss 
  (H : (41 * a + 59 * b) > 100 * (a + b) / 2) : true :=
by
  sorry

end NUMINAMATH_GPT_pharmacy_incurs_loss_l1785_178564


namespace NUMINAMATH_GPT_merchant_marked_price_l1785_178506

theorem merchant_marked_price (L P x S : ℝ)
  (h1 : L = 100)
  (h2 : P = 70)
  (h3 : S = 0.8 * x)
  (h4 : 0.8 * x - 70 = 0.3 * (0.8 * x)) :
  x = 125 :=
by
  sorry

end NUMINAMATH_GPT_merchant_marked_price_l1785_178506


namespace NUMINAMATH_GPT_train_speed_180_kmph_l1785_178500

def train_speed_in_kmph (length_meters : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_m_per_s := length_meters / time_seconds
  let speed_km_per_h := speed_m_per_s * 36 / 10
  speed_km_per_h

theorem train_speed_180_kmph:
  train_speed_in_kmph 400 8 = 180 := by
  sorry

end NUMINAMATH_GPT_train_speed_180_kmph_l1785_178500


namespace NUMINAMATH_GPT_range_of_a_l1785_178538

/--
Let f be a function defined on the interval [-1, 1] that is increasing and odd.
If f(-a+1) + f(4a-5) > 0, then the range of the real number a is (4/3, 3/2].
-/
theorem range_of_a
  (f : ℝ → ℝ)
  (h_dom : ∀ x, -1 ≤ x ∧ x ≤ 1 → f x = f x)  -- domain condition
  (h_incr : ∀ x y, x < y → f x < f y)          -- increasing condition
  (h_odd : ∀ x, f (-x) = - f x)                -- odd function condition
  (a : ℝ)
  (h_ineq : f (-a + 1) + f (4 * a - 5) > 0) :
  4 / 3 < a ∧ a ≤ 3 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1785_178538


namespace NUMINAMATH_GPT_value_of_a_l1785_178588

theorem value_of_a (a : ℝ) (h : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 2 → (a * x + 6 ≤ 10)) :
  a = 2 ∨ a = -4 ∨ a = 0 :=
sorry

end NUMINAMATH_GPT_value_of_a_l1785_178588


namespace NUMINAMATH_GPT_find_fourth_student_in_sample_l1785_178570

theorem find_fourth_student_in_sample :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 48 ∧ 
           (∀ (k : ℕ), k = 29 → 1 ≤ k ∧ k ≤ 48 ∧ ((k = 5 + 2 * 12) ∨ (k = 41 - 12)) ∧ n = 17) :=
sorry

end NUMINAMATH_GPT_find_fourth_student_in_sample_l1785_178570


namespace NUMINAMATH_GPT_circles_intersect_probability_l1785_178536

noncomputable def probability_circles_intersect : ℝ :=
  sorry

theorem circles_intersect_probability :
  probability_circles_intersect = (5 * Real.sqrt 2 - 7) / 4 :=
  sorry

end NUMINAMATH_GPT_circles_intersect_probability_l1785_178536


namespace NUMINAMATH_GPT_trapezoid_area_l1785_178543

noncomputable def area_of_trapezoid : ℝ :=
  let y1 := 12
  let y2 := 5
  let x1 := 12 / 2
  let x2 := 5 / 2
  ((x1 + x2) / 2) * (y1 - y2)

theorem trapezoid_area : area_of_trapezoid = 29.75 := by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1785_178543


namespace NUMINAMATH_GPT_problem_31_36_l1785_178516

noncomputable def is_prime (n : ℕ) : Prop := sorry

theorem problem_31_36 (p k : ℕ) (hp : is_prime (4 * k + 1)) :
  (∃ x y m : ℕ, x^2 + y^2 = m * p) ∧ (∀ m > 1, ∃ x y m1 : ℕ, x^2 + y^2 = m * p ∧ 0 < m1 ∧ m1 < m) :=
by sorry

end NUMINAMATH_GPT_problem_31_36_l1785_178516


namespace NUMINAMATH_GPT_hexadecagon_area_l1785_178530

theorem hexadecagon_area (r : ℝ) : 
  (∃ A : ℝ, A = 4 * r^2 * Real.sqrt (2 - Real.sqrt 2)) :=
sorry

end NUMINAMATH_GPT_hexadecagon_area_l1785_178530


namespace NUMINAMATH_GPT_bisection_method_root_interval_l1785_178581

def f (x : ℝ) : ℝ := x^3 + x - 8

theorem bisection_method_root_interval :
  f 1 < 0 → f 1.5 < 0 → f 1.75 < 0 → f 2 > 0 → ∃ x, (1.75 < x ∧ x < 2 ∧ f x = 0) :=
by
  intros h1 h15 h175 h2
  sorry

end NUMINAMATH_GPT_bisection_method_root_interval_l1785_178581


namespace NUMINAMATH_GPT_treasure_coins_l1785_178569

theorem treasure_coins (m : ℕ) 
  (h : (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m) : 
  m = 120 := 
sorry

end NUMINAMATH_GPT_treasure_coins_l1785_178569


namespace NUMINAMATH_GPT_combined_age_of_staff_l1785_178503

/--
In a school, the average age of a class of 50 students is 25 years. 
The average age increased by 2 years when the ages of 5 additional 
staff members, including the teacher, are also taken into account. 
Prove that the combined age of these 5 staff members is 235 years.
-/
theorem combined_age_of_staff 
    (n_students : ℕ) (avg_age_students : ℕ) (n_staff : ℕ) (avg_age_total : ℕ)
    (h1 : n_students = 50) 
    (h2 : avg_age_students = 25) 
    (h3 : n_staff = 5) 
    (h4 : avg_age_total = 27) :
  n_students * avg_age_students + (n_students + n_staff) * avg_age_total - 
  n_students * avg_age_students = 235 :=
by
  sorry

end NUMINAMATH_GPT_combined_age_of_staff_l1785_178503


namespace NUMINAMATH_GPT_Ethan_uses_8_ounces_each_l1785_178597

def Ethan (b: ℕ): Prop :=
  let number_of_candles := 10 - 3
  let total_coconut_oil := number_of_candles * 1
  let total_beeswax := 63 - total_coconut_oil
  let beeswax_per_candle := total_beeswax / number_of_candles
  beeswax_per_candle = b

theorem Ethan_uses_8_ounces_each (b: ℕ) (hb: Ethan b): b = 8 :=
  sorry

end NUMINAMATH_GPT_Ethan_uses_8_ounces_each_l1785_178597


namespace NUMINAMATH_GPT_expression_evaluation_l1785_178579

theorem expression_evaluation : 
  2000 * 1995 * 0.1995 - 10 = 0.2 * 1995^2 - 10 := 
by 
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1785_178579


namespace NUMINAMATH_GPT_number_of_boys_at_reunion_l1785_178559

theorem number_of_boys_at_reunion (n : ℕ) (h : (n * (n - 1)) / 2 = 21) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_at_reunion_l1785_178559


namespace NUMINAMATH_GPT_find_cos_sum_l1785_178512

-- Defining the conditions based on the problem
variable (P A B C D : Type) (α β : ℝ)

-- Assumptions stating the given conditions
def regular_quadrilateral_pyramid (P A B C D : Type) : Prop :=
  -- Placeholder for the exact definition of a regular quadrilateral pyramid
  sorry

def dihedral_angle_lateral_base (P A B C D : Type) (α : ℝ) : Prop :=
  -- Placeholder for the exact property stating the dihedral angle between lateral face and base is α
  sorry

def dihedral_angle_adjacent_lateral (P A B C D : Type) (β : ℝ) : Prop :=
  -- Placeholder for the exact property stating the dihedral angle between two adjacent lateral faces is β
  sorry

-- The final theorem that we want to prove
theorem find_cos_sum (P A B C D : Type) (α β : ℝ)
  (H1 : regular_quadrilateral_pyramid P A B C D)
  (H2 : dihedral_angle_lateral_base P A B C D α)
  (H3 : dihedral_angle_adjacent_lateral P A B C D β) :
  2 * Real.cos β + Real.cos (2 * α) = -1 :=
sorry

end NUMINAMATH_GPT_find_cos_sum_l1785_178512


namespace NUMINAMATH_GPT_A_scores_2_points_B_scores_at_least_2_points_l1785_178589

-- Define the probabilities of outcomes.
def prob_A_win := 0.5
def prob_A_lose := 0.3
def prob_A_draw := 0.2

-- Calculate the probability of A scoring 2 points.
theorem A_scores_2_points : 
    (prob_A_win * prob_A_lose + prob_A_lose * prob_A_win + prob_A_draw * prob_A_draw) = 0.34 :=
by
  sorry

-- Calculate the probability of B scoring at least 2 points.
theorem B_scores_at_least_2_points : 
    (1 - (prob_A_win * prob_A_win + (prob_A_win * prob_A_draw + prob_A_draw * prob_A_win))) = 0.55 :=
by
  sorry

end NUMINAMATH_GPT_A_scores_2_points_B_scores_at_least_2_points_l1785_178589


namespace NUMINAMATH_GPT_arithmetic_sequence_term_number_l1785_178517

theorem arithmetic_sequence_term_number
  (a : ℕ → ℤ)
  (ha1 : a 1 = 1)
  (ha2 : a 2 = 3)
  (n : ℕ)
  (hn : a n = 217) :
  n = 109 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_number_l1785_178517


namespace NUMINAMATH_GPT_original_selling_price_is_1100_l1785_178580

-- Let P be the original purchase price.
variable (P : ℝ)

-- Condition 1: Bill made a profit of 10% on the original purchase price.
def original_selling_price := 1.10 * P

-- Condition 2: If he had purchased that product for 10% less 
-- and sold it at a profit of 30%, he would have received $70 more.
def new_purchase_price := 0.90 * P
def new_selling_price := 1.17 * P
def price_difference := new_selling_price - original_selling_price

-- Theorem: The original selling price was $1100.
theorem original_selling_price_is_1100 (h : price_difference P = 70) : 
  original_selling_price P = 1100 :=
sorry

end NUMINAMATH_GPT_original_selling_price_is_1100_l1785_178580


namespace NUMINAMATH_GPT_max_valid_committees_l1785_178519

-- Define the conditions
def community_size : ℕ := 20
def english_speakers : ℕ := 10
def german_speakers : ℕ := 10
def french_speakers : ℕ := 10
def total_subsets : ℕ := Nat.choose community_size 3
def invalid_subsets_per_language : ℕ := Nat.choose 10 3

-- Lean statement to verify the number of valid committees
theorem max_valid_committees :
  total_subsets - 3 * invalid_subsets_per_language = 1020 :=
by
  simp [community_size, total_subsets, invalid_subsets_per_language]
  sorry

end NUMINAMATH_GPT_max_valid_committees_l1785_178519


namespace NUMINAMATH_GPT_visited_both_countries_l1785_178550

theorem visited_both_countries {Total Iceland Norway Neither Both : ℕ} 
  (h1 : Total = 50) 
  (h2 : Iceland = 25)
  (h3 : Norway = 23)
  (h4 : Neither = 23) 
  (h5 : Total - Neither = 27) 
  (h6 : Iceland + Norway - Both = 27) : 
  Both = 21 := 
by
  sorry

end NUMINAMATH_GPT_visited_both_countries_l1785_178550


namespace NUMINAMATH_GPT_area_to_paint_correct_l1785_178533

-- Define the measurements used in the problem
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window1_height : ℕ := 3
def window1_length : ℕ := 5
def window2_height : ℕ := 2
def window2_length : ℕ := 2

-- Definition of areas
def wall_area : ℕ := wall_height * wall_length
def window1_area : ℕ := window1_height * window1_length
def window2_area : ℕ := window2_height * window2_length

-- Definition of total area to paint
def total_area_to_paint : ℕ := wall_area - (window1_area + window2_area)

-- Theorem statement to prove the total area to paint is 131 square feet
theorem area_to_paint_correct : total_area_to_paint = 131 := by
  sorry

end NUMINAMATH_GPT_area_to_paint_correct_l1785_178533


namespace NUMINAMATH_GPT_people_left_line_l1785_178575

theorem people_left_line (initial new final L : ℕ) 
  (h1 : initial = 30) 
  (h2 : new = 5) 
  (h3 : final = 25) 
  (h4 : initial - L + new = final) : L = 10 := by
  sorry

end NUMINAMATH_GPT_people_left_line_l1785_178575


namespace NUMINAMATH_GPT_ends_with_two_zeros_l1785_178529

theorem ends_with_two_zeros (x y : ℕ) (h : (x^2 + x * y + y^2) % 10 = 0) : (x^2 + x * y + y^2) % 100 = 0 :=
sorry

end NUMINAMATH_GPT_ends_with_two_zeros_l1785_178529


namespace NUMINAMATH_GPT_sum_div_by_24_l1785_178583

theorem sum_div_by_24 (m n : ℕ) (h : ∃ k : ℤ, mn + 1 = 24 * k): (m + n) % 24 = 0 := 
by
  sorry

end NUMINAMATH_GPT_sum_div_by_24_l1785_178583


namespace NUMINAMATH_GPT_equivalent_single_discount_l1785_178504

theorem equivalent_single_discount (P : ℝ) (hP : 0 < P) : 
    let first_discount : ℝ := 0.15
    let second_discount : ℝ := 0.25
    let single_discount : ℝ := 0.3625
    (1 - first_discount) * (1 - second_discount) * P = (1 - single_discount) * P := by
    sorry

end NUMINAMATH_GPT_equivalent_single_discount_l1785_178504


namespace NUMINAMATH_GPT_range_of_a_l1785_178599

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x
noncomputable def g (x a : ℝ) : ℝ := x + 1 / (x - a)

theorem range_of_a (a : ℝ) :
  (∀ x1 : ℝ, x1 ∈ Set.Icc 0 2 → ∃ x2 : ℝ, x2 ∈ Set.Ioi a ∧ f x1 ≥ g x2 a) →
  a ≤ -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1785_178599


namespace NUMINAMATH_GPT_number_of_geese_is_correct_l1785_178514

noncomputable def number_of_ducks := 37
noncomputable def total_number_of_birds := 95
noncomputable def number_of_geese := total_number_of_birds - number_of_ducks

theorem number_of_geese_is_correct : number_of_geese = 58 := by
  sorry

end NUMINAMATH_GPT_number_of_geese_is_correct_l1785_178514


namespace NUMINAMATH_GPT_sasha_remainder_20_l1785_178573

theorem sasha_remainder_20
  (n a b c d : ℕ)
  (h1 : n = 102 * a + b)
  (h2 : 0 ≤ b ∧ b ≤ 101)
  (h3 : n = 103 * c + d)
  (h4 : d = 20 - a) :
  b = 20 :=
by
  sorry

end NUMINAMATH_GPT_sasha_remainder_20_l1785_178573


namespace NUMINAMATH_GPT_value_of_w_div_x_l1785_178552

theorem value_of_w_div_x (w x y : ℝ) 
  (h1 : w / x = a) 
  (h2 : w / y = 1 / 5) 
  (h3 : (x + y) / y = 2.2) : 
  w / x = 6 / 25 := by
  sorry

end NUMINAMATH_GPT_value_of_w_div_x_l1785_178552


namespace NUMINAMATH_GPT_candy_bar_calories_l1785_178562

theorem candy_bar_calories:
  ∀ (calories_per_candy_bar : ℕ) (num_candy_bars : ℕ), 
  calories_per_candy_bar = 3 → 
  num_candy_bars = 5 → 
  calories_per_candy_bar * num_candy_bars = 15 :=
by
  sorry

end NUMINAMATH_GPT_candy_bar_calories_l1785_178562


namespace NUMINAMATH_GPT_arithmetic_sequence_x_value_l1785_178566

theorem arithmetic_sequence_x_value
  (x : ℝ)
  (h₁ : 2 * x - (1 / 3) = (x + 4) - 2 * x) :
  x = 13 / 3 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_x_value_l1785_178566


namespace NUMINAMATH_GPT_value_of_x_l1785_178567

theorem value_of_x (x : ℤ) (h : x + 3 = 4 ∨ x + 3 = -4) : x = 1 ∨ x = -7 := sorry

end NUMINAMATH_GPT_value_of_x_l1785_178567


namespace NUMINAMATH_GPT_right_triangle_tangent_length_l1785_178546

theorem right_triangle_tangent_length (DE DF : ℝ) (h1 : DE = 7) (h2 : DF = Real.sqrt 85)
  (h3 : ∀ (EF : ℝ), DE^2 + EF^2 = DF^2 → EF = 6): FQ = 6 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_tangent_length_l1785_178546


namespace NUMINAMATH_GPT_max_regions_two_convex_polygons_l1785_178527

theorem max_regions_two_convex_polygons (M N : ℕ) (hM : M > N) :
    ∃ R, R = 2 * N + 2 := 
sorry

end NUMINAMATH_GPT_max_regions_two_convex_polygons_l1785_178527


namespace NUMINAMATH_GPT_number_of_players_in_tournament_l1785_178563

theorem number_of_players_in_tournament (G : ℕ) (h1 : G = 42) (h2 : ∀ n : ℕ, G = n * (n - 1)) : ∃ n : ℕ, G = 42 ∧ n = 7 :=
by
  -- Let's suppose n is the number of players, then we need to prove
  -- ∃ n : ℕ, 42 = n * (n - 1) ∧ n = 7
  sorry

end NUMINAMATH_GPT_number_of_players_in_tournament_l1785_178563


namespace NUMINAMATH_GPT_compute_expression_l1785_178565

theorem compute_expression :
  (5 + 7)^2 + (5^2 + 7^2) * 2 = 292 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l1785_178565


namespace NUMINAMATH_GPT_find_remainder_l1785_178521

-- Definitions
variable (x y : ℕ)
variable (h1 : x > 0)
variable (h2 : y > 0)
variable (h3 : (x : ℝ) / y = 96.15)
variable (h4 : approximately_equal (y : ℝ) 60)

-- Target statement
theorem find_remainder : x % y = 9 :=
sorry

end NUMINAMATH_GPT_find_remainder_l1785_178521


namespace NUMINAMATH_GPT_mary_total_nickels_l1785_178513

theorem mary_total_nickels (n1 n2 : ℕ) (h1 : n1 = 7) (h2 : n2 = 5) : n1 + n2 = 12 := by
  sorry

end NUMINAMATH_GPT_mary_total_nickels_l1785_178513


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1785_178586

theorem isosceles_triangle_perimeter 
  (a b c : ℝ)  (h_iso : a = b ∨ b = c ∨ c = a)
  (h_len1 : a = 4 ∨ b = 4 ∨ c = 4)
  (h_len2 : a = 9 ∨ b = 9 ∨ c = 9)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a + b + c = 22 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1785_178586


namespace NUMINAMATH_GPT_smallest_k_for_square_l1785_178542

theorem smallest_k_for_square : ∃ k : ℕ, (2016 * 2017 * 2018 * 2019 + k) = n^2 ∧ k = 1 :=
by
  use 1
  sorry

end NUMINAMATH_GPT_smallest_k_for_square_l1785_178542


namespace NUMINAMATH_GPT_arithmetic_sequence_k_is_10_l1785_178540

noncomputable def a_n (n : ℕ) (d : ℝ) : ℝ := (n - 1) * d

theorem arithmetic_sequence_k_is_10 (d : ℝ) (h : d ≠ 0) : 
  (∃ k : ℕ, a_n k d = (a_n 1 d) + (a_n 2 d) + (a_n 3 d) + (a_n 4 d) + (a_n 5 d) + (a_n 6 d) + (a_n 7 d) ∧ k = 10) := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_k_is_10_l1785_178540


namespace NUMINAMATH_GPT_beavers_working_on_home_l1785_178518

noncomputable def initial_beavers : ℝ := 2.0
noncomputable def additional_beavers : ℝ := 1.0

theorem beavers_working_on_home : initial_beavers + additional_beavers = 3.0 :=
by
  sorry

end NUMINAMATH_GPT_beavers_working_on_home_l1785_178518


namespace NUMINAMATH_GPT_calculate_new_average_weight_l1785_178520

noncomputable def new_average_weight (original_team_weight : ℕ) (num_original_players : ℕ) 
 (new_player1_weight : ℕ) (new_player2_weight : ℕ) (num_new_players : ℕ) : ℕ :=
 (original_team_weight + new_player1_weight + new_player2_weight) / (num_original_players + num_new_players)

theorem calculate_new_average_weight : 
  new_average_weight 847 7 110 60 2 = 113 := 
by 
sorry

end NUMINAMATH_GPT_calculate_new_average_weight_l1785_178520


namespace NUMINAMATH_GPT_jen_ducks_l1785_178585

theorem jen_ducks (c d : ℕ) (h1 : d = 4 * c + 10) (h2 : c + d = 185) : d = 150 := by
  sorry

end NUMINAMATH_GPT_jen_ducks_l1785_178585


namespace NUMINAMATH_GPT_find_incorrect_statement_l1785_178549

theorem find_incorrect_statement :
  ¬ (∀ a b c : ℝ, c ≠ 0 → (a < b → a * c^2 < b * c^2)) :=
by
  sorry

end NUMINAMATH_GPT_find_incorrect_statement_l1785_178549


namespace NUMINAMATH_GPT_series_sum_eq_l1785_178526

theorem series_sum_eq :
  (1^25 + 2^24 + 3^23 + 4^22 + 5^21 + 6^20 + 7^19 + 8^18 + 9^17 + 10^16 + 
  11^15 + 12^14 + 13^13 + 14^12 + 15^11 + 16^10 + 17^9 + 18^8 + 19^7 + 20^6 + 
  21^5 + 22^4 + 23^3 + 24^2 + 25^1) = 66071772829247409 := 
by
  sorry

end NUMINAMATH_GPT_series_sum_eq_l1785_178526


namespace NUMINAMATH_GPT_cylinder_height_proof_l1785_178511

noncomputable def cone_base_radius : ℝ := 15
noncomputable def cone_height : ℝ := 25
noncomputable def cylinder_base_radius : ℝ := 10
noncomputable def cylinder_water_height : ℝ := 18.75

theorem cylinder_height_proof :
  (1 / 3 * π * cone_base_radius^2 * cone_height) = π * cylinder_base_radius^2 * cylinder_water_height :=
by sorry

end NUMINAMATH_GPT_cylinder_height_proof_l1785_178511
