import Mathlib

namespace NUMINAMATH_GPT_probability_three_red_before_two_green_l762_76248

noncomputable def probability_red_chips_drawn_before_green (red_chips green_chips : ℕ) (total_chips : ℕ) : ℚ := sorry

theorem probability_three_red_before_two_green 
    (red_chips green_chips : ℕ) (total_chips : ℕ)
    (h_red : red_chips = 3) (h_green : green_chips = 2) 
    (h_total: total_chips = red_chips + green_chips) :
  probability_red_chips_drawn_before_green red_chips green_chips total_chips = 3 / 10 :=
  sorry

end NUMINAMATH_GPT_probability_three_red_before_two_green_l762_76248


namespace NUMINAMATH_GPT_min_packs_120_cans_l762_76238

theorem min_packs_120_cans (p8 p16 p32 : ℕ) (total_cans packs_needed : ℕ) :
  total_cans = 120 →
  p8 * 8 + p16 * 16 + p32 * 32 = total_cans →
  packs_needed = p8 + p16 + p32 →
  (∀ (q8 q16 q32 : ℕ), q8 * 8 + q16 * 16 + q32 * 32 = total_cans → q8 + q16 + q32 ≥ packs_needed) →
  packs_needed = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_packs_120_cans_l762_76238


namespace NUMINAMATH_GPT_basic_astrophysics_degrees_l762_76216

def budget_allocation : Nat := 100
def microphotonics_perc : Nat := 14
def home_electronics_perc : Nat := 19
def food_additives_perc : Nat := 10
def genetically_modified_perc : Nat := 24
def industrial_lubricants_perc : Nat := 8

def arc_of_sector (percentage : Nat) : Nat := percentage * 360 / budget_allocation

theorem basic_astrophysics_degrees :
  arc_of_sector (budget_allocation - (microphotonics_perc + home_electronics_perc + food_additives_perc + genetically_modified_perc + industrial_lubricants_perc)) = 90 :=
  by
  sorry

end NUMINAMATH_GPT_basic_astrophysics_degrees_l762_76216


namespace NUMINAMATH_GPT_intersection_point_for_m_l762_76242

variable (n : ℕ) (x_0 y_0 : ℕ)
variable (h₁ : n ≥ 2)
variable (h₂ : y_0 ^ 2 = n * x_0 - 1)
variable (h₃ : y_0 = x_0)

theorem intersection_point_for_m (m : ℕ) (hm : 0 < m) : ∃ k : ℕ, k ≥ 2 ∧ (y_0 ^ m = x_0 ^ m) ∧ (y_0 ^ m) ^ 2 = k * (x_0 ^ m) - 1 :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_for_m_l762_76242


namespace NUMINAMATH_GPT_min_passengers_to_fill_bench_l762_76232

theorem min_passengers_to_fill_bench (width_per_passenger : ℚ) (total_seat_width : ℚ) (num_seats : ℕ):
  width_per_passenger = 1/6 → total_seat_width = num_seats → num_seats = 6 → 3 ≥ (total_seat_width / width_per_passenger) :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_min_passengers_to_fill_bench_l762_76232


namespace NUMINAMATH_GPT_train_times_valid_l762_76274

-- Define the parameters and conditions
def trainA_usual_time : ℝ := 180 -- minutes
def trainB_travel_time : ℝ := 810 -- minutes

theorem train_times_valid (t : ℝ) (T_B : ℝ) 
  (cond1 : (7 / 6) * t = t + 30)
  (cond2 : T_B = 4.5 * t) : 
  t = trainA_usual_time ∧ T_B = trainB_travel_time :=
by
  sorry

end NUMINAMATH_GPT_train_times_valid_l762_76274


namespace NUMINAMATH_GPT_sharks_problem_l762_76275

variable (F : ℝ)
variable (S : ℝ := 0.25 * (F + 3 * F))
variable (total_sharks : ℝ := 15)

theorem sharks_problem : 
  (0.25 * (F + 3 * F) = 15) ↔ (F = 15) :=
by 
  sorry

end NUMINAMATH_GPT_sharks_problem_l762_76275


namespace NUMINAMATH_GPT_coffee_price_l762_76247

theorem coffee_price (qd : ℝ) (d : ℝ) (rp : ℝ) :
  qd = 4.5 ∧ d = 0.25 → rp = 12 :=
by 
  sorry

end NUMINAMATH_GPT_coffee_price_l762_76247


namespace NUMINAMATH_GPT_tom_paid_amount_correct_l762_76290

def kg (n : Nat) : Nat := n -- Just a type alias clarification

theorem tom_paid_amount_correct :
  ∀ (quantity_apples : Nat) (rate_apples : Nat) (quantity_mangoes : Nat) (rate_mangoes : Nat),
  quantity_apples = kg 8 →
  rate_apples = 70 →
  quantity_mangoes = kg 9 →
  rate_mangoes = 55 →
  (quantity_apples * rate_apples) + (quantity_mangoes * rate_mangoes) = 1055 :=
by
  intros quantity_apples rate_apples quantity_mangoes rate_mangoes
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end NUMINAMATH_GPT_tom_paid_amount_correct_l762_76290


namespace NUMINAMATH_GPT_cost_of_paint_per_kg_l762_76237

/-- The cost of painting one square foot is Rs. 50. -/
theorem cost_of_paint_per_kg (side_length : ℝ) (cost_total : ℝ) (coverage_per_kg : ℝ) (total_surface_area : ℝ) (total_paint_needed : ℝ) (cost_per_kg : ℝ) 
  (h1 : side_length = 20)
  (h2 : cost_total = 6000)
  (h3 : coverage_per_kg = 20)
  (h4 : total_surface_area = 6 * side_length^2)
  (h5 : total_paint_needed = total_surface_area / coverage_per_kg)
  (h6 : cost_per_kg = cost_total / total_paint_needed) :
  cost_per_kg = 50 :=
sorry

end NUMINAMATH_GPT_cost_of_paint_per_kg_l762_76237


namespace NUMINAMATH_GPT_marbles_left_l762_76270

def initial_marbles : ℕ := 64
def marbles_given : ℕ := 14

theorem marbles_left : (initial_marbles - marbles_given) = 50 := by
  sorry

end NUMINAMATH_GPT_marbles_left_l762_76270


namespace NUMINAMATH_GPT_rectangle_area_is_432_l762_76210

-- Definition of conditions and problem in Lean 4
noncomputable def circle_radius : ℝ := 6
noncomputable def rectangle_ratio_length_width : ℝ := 3 / 1
noncomputable def calculate_rectangle_area (radius : ℝ) (ratio : ℝ) : ℝ :=
  let diameter := 2 * radius
  let width := diameter
  let length := ratio * width
  length * width

-- Lean statement to prove the area
theorem rectangle_area_is_432 : calculate_rectangle_area circle_radius rectangle_ratio_length_width = 432 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_is_432_l762_76210


namespace NUMINAMATH_GPT_no_three_positive_reals_l762_76292

noncomputable def S (a : ℝ) : Set ℕ := { n | ∃ (k : ℕ), n = ⌊(k : ℝ) * a⌋ }

theorem no_three_positive_reals (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  (S a ∩ S b = ∅) ∧ (S b ∩ S c = ∅) ∧ (S c ∩ S a = ∅) ∧ (S a ∪ S b ∪ S c = Set.univ) → false :=
sorry

end NUMINAMATH_GPT_no_three_positive_reals_l762_76292


namespace NUMINAMATH_GPT_least_n_divisibility_l762_76246

theorem least_n_divisibility :
  ∃ n : ℕ, 0 < n ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ n → k ∣ (n - 1)^2) ∧ (∃ k : ℕ, 2 ≤ k ∧ k ≤ n ∧ ¬ k ∣ (n - 1)^2) ∧ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_least_n_divisibility_l762_76246


namespace NUMINAMATH_GPT_find_digits_of_abc_l762_76277

theorem find_digits_of_abc (a b c : ℕ) (h1 : a ≠ c) (h2 : c - a = 3) (h3 : (100 * a + 10 * b + c) - (100 * c + 10 * a + b) = 100 * (a - (c - 1)) + 0 + (b - b)) : 
  100 * a + 10 * b + c = 619 :=
by
  sorry

end NUMINAMATH_GPT_find_digits_of_abc_l762_76277


namespace NUMINAMATH_GPT_fish_in_aquarium_l762_76255

theorem fish_in_aquarium (initial_fish : ℕ) (added_fish : ℕ) (h1 : initial_fish = 10) (h2 : added_fish = 3) : initial_fish + added_fish = 13 := by
  sorry

end NUMINAMATH_GPT_fish_in_aquarium_l762_76255


namespace NUMINAMATH_GPT_find_rate_percent_l762_76258

theorem find_rate_percent (SI P T : ℝ) (h1 : SI = 160) (h2 : P = 800) (h3 : T = 5) : P * (4:ℝ) * T / 100 = SI :=
by
  sorry

end NUMINAMATH_GPT_find_rate_percent_l762_76258


namespace NUMINAMATH_GPT_find_general_students_l762_76295

-- Define the conditions and the question
structure Halls :=
  (general : ℕ)
  (biology : ℕ)
  (math : ℕ)
  (total : ℕ)

def conditions_met (h : Halls) : Prop :=
  h.biology = 2 * h.general ∧
  h.math = (3 / 5 : ℚ) * (h.general + h.biology) ∧
  h.total = h.general + h.biology + h.math ∧
  h.total = 144

-- The proof problem statement
theorem find_general_students (h : Halls) (h_cond : conditions_met h) : h.general = 30 :=
sorry

end NUMINAMATH_GPT_find_general_students_l762_76295


namespace NUMINAMATH_GPT_distribute_books_l762_76236

-- Definition of books and people
def num_books : Nat := 2
def num_people : Nat := 10

-- The main theorem statement that we need to prove.
theorem distribute_books : (num_people ^ num_books) = 100 :=
by
  -- Proof body
  sorry

end NUMINAMATH_GPT_distribute_books_l762_76236


namespace NUMINAMATH_GPT_evenness_oddness_of_f_min_value_of_f_l762_76281

noncomputable def f (a x : ℝ) : ℝ :=
  x^2 + |x - a| + 1

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

theorem evenness_oddness_of_f (a : ℝ) :
  (is_even (f a) ↔ a = 0) ∧ (a ≠ 0 → ¬ is_even (f a) ∧ ¬ is_odd (f a)) :=
by
  sorry

theorem min_value_of_f (a x : ℝ) (h : x ≥ a) :
  (a ≤ -1 / 2 → f a x = 3 / 4 - a) ∧ (a > -1 / 2 → f a x = a^2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_evenness_oddness_of_f_min_value_of_f_l762_76281


namespace NUMINAMATH_GPT_max_value_of_b_minus_a_l762_76252

theorem max_value_of_b_minus_a (a b : ℝ) (h₀ : a < 0)
  (h₁ : ∀ x : ℝ, a < x ∧ x < b → (x^2 + 2017 * a) * (x + 2016 * b) ≥ 0) :
  b - a ≤ 2017 :=
sorry

end NUMINAMATH_GPT_max_value_of_b_minus_a_l762_76252


namespace NUMINAMATH_GPT_smallest_possible_value_l762_76297

-- Definitions of the digits
def P := 1
def A := 9
def B := 2
def H := 8
def O := 3

-- Expression for continued fraction T
noncomputable def T : ℚ :=
  P + 1 / (A + 1 / (B + 1 / (H + 1 / O)))

-- The goal is to prove that T is the smallest possible value given the conditions
theorem smallest_possible_value : T = 555 / 502 :=
by
  -- The detailed proof would be done here, but for now we use sorry because we only need the statement
  sorry

end NUMINAMATH_GPT_smallest_possible_value_l762_76297


namespace NUMINAMATH_GPT_carlos_goal_l762_76251

def july_books : ℕ := 28
def august_books : ℕ := 30
def june_books : ℕ := 42

theorem carlos_goal (goal : ℕ) :
  goal = june_books + july_books + august_books := by
  sorry

end NUMINAMATH_GPT_carlos_goal_l762_76251


namespace NUMINAMATH_GPT_compare_abc_l762_76289

theorem compare_abc (a b c : ℝ)
  (h1 : a = Real.log 0.9 / Real.log 2)
  (h2 : b = 3 ^ (-1 / 3 : ℝ))
  (h3 : c = (1 / 3 : ℝ) ^ (1 / 2 : ℝ)) :
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_GPT_compare_abc_l762_76289


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l762_76225

theorem geometric_sequence_first_term (a b c : ℕ) (r : ℕ) (h1 : r = 2) (h2 : b = a * r)
  (h3 : c = b * r) (h4 : 32 = c * r) (h5 : 64 = 32 * r) :
  a = 4 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l762_76225


namespace NUMINAMATH_GPT_min_value_expression_l762_76201

theorem min_value_expression (x y : ℝ) : (∃ z : ℝ, (forall x y : ℝ, z ≤ 5*x^2 + 4*y^2 - 8*x*y + 2*x + 4) ∧ z = 3) :=
sorry

end NUMINAMATH_GPT_min_value_expression_l762_76201


namespace NUMINAMATH_GPT_speed_difference_l762_76204

theorem speed_difference (h_cyclist : 88 / 8 = 11) (h_car : 48 / 8 = 6) :
  (11 - 6 = 5) :=
by
  sorry

end NUMINAMATH_GPT_speed_difference_l762_76204


namespace NUMINAMATH_GPT_quadratic_solution_l762_76208

theorem quadratic_solution (x : ℝ) : 
  x^2 - 2 * x - 3 = 0 → (x = 3 ∨ x = -1) :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_solution_l762_76208


namespace NUMINAMATH_GPT_box_height_l762_76276

theorem box_height (x h : ℕ) 
  (h1 : h = x + 5) 
  (h2 : 6 * x^2 + 20 * x ≥ 150) 
  (h3 : 5 * x + 5 ≥ 25) 
  : h = 9 :=
by 
  sorry

end NUMINAMATH_GPT_box_height_l762_76276


namespace NUMINAMATH_GPT_problem_solution_l762_76214

theorem problem_solution (a0 a1 a2 a3 a4 a5 : ℝ) :
  (1 + 2*x)^5 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5 →
  a0 + a2 + a4 = 121 := 
sorry

end NUMINAMATH_GPT_problem_solution_l762_76214


namespace NUMINAMATH_GPT_mass_percentage_of_Cl_in_compound_l762_76213

theorem mass_percentage_of_Cl_in_compound (mass_percentage_Cl : ℝ) (h : mass_percentage_Cl = 92.11) : mass_percentage_Cl = 92.11 :=
sorry

end NUMINAMATH_GPT_mass_percentage_of_Cl_in_compound_l762_76213


namespace NUMINAMATH_GPT_time_A_reaches_destination_l762_76231

theorem time_A_reaches_destination (x t : ℝ) (h_ratio : (4 * t) = 3 * (t + 0.5)) : (t + 0.5) = 2 :=
by {
  -- derived by algebraic manipulation
  sorry
}

end NUMINAMATH_GPT_time_A_reaches_destination_l762_76231


namespace NUMINAMATH_GPT_solve_x_l762_76245

theorem solve_x :
  ∀ (x y z w : ℤ),
    x = y + 7 →
    y = z + 15 →
    z = w + 25 →
    w = 65 →
    x = 112 :=
by
  intros x y z w
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_solve_x_l762_76245


namespace NUMINAMATH_GPT_max_intersection_distance_l762_76261

theorem max_intersection_distance :
  let C1_x (α : ℝ) := 2 + 2 * Real.cos α
  let C1_y (α : ℝ) := 2 * Real.sin α
  let C2_x (β : ℝ) := 2 * Real.cos β
  let C2_y (β : ℝ) := 2 + 2 * Real.sin β
  let l1 (α : ℝ) := α
  let l2 (α : ℝ) := α - Real.pi / 6
  (0 < Real.pi / 2) →
  let OP (α : ℝ) := 4 * Real.cos α
  let OQ (α : ℝ) := 4 * Real.sin (α - Real.pi / 6)
  let pq_prod (α : ℝ) := OP α * OQ α
  ∀α, 0 < α ∧ α < Real.pi / 2 → pq_prod α ≤ 4 := by
  sorry

end NUMINAMATH_GPT_max_intersection_distance_l762_76261


namespace NUMINAMATH_GPT_min_value_of_expression_l762_76219

theorem min_value_of_expression 
  (a b : ℝ) 
  (h : a > 0) 
  (h₀ : b > 0) 
  (h₁ : 2*a + b = 2) : 
  ∃ c : ℝ, c = (8*a + b) / (a*b) ∧ c = 9 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l762_76219


namespace NUMINAMATH_GPT_females_count_l762_76234

-- Defining variables and constants
variables (P M F : ℕ)
-- The condition given the total population
def town_population := P = 600
-- The condition given the proportion of males
def proportion_of_males := M = P / 3
-- The condition determining the number of females
def number_of_females := F = P - M

-- The theorem stating the number of females is 400
theorem females_count (P M F : ℕ) (h1 : town_population P)
  (h2 : proportion_of_males P M) 
  (h3 : number_of_females P M F) : 
  F = 400 := 
sorry

end NUMINAMATH_GPT_females_count_l762_76234


namespace NUMINAMATH_GPT_ex1_simplified_ex2_simplified_l762_76279

-- Definitions and problem setup
def ex1 (a : ℝ) : ℝ := ((-a^3)^2 * a^3 - 4 * a^2 * a^7)
def ex2 (a : ℝ) : ℝ := (2 * a + 1) * (-2 * a + 1)

-- Proof goals
theorem ex1_simplified (a : ℝ) : ex1 a = -3 * a^9 :=
by sorry

theorem ex2_simplified (a : ℝ) : ex2 a = 4 * a^2 - 1 :=
by sorry

end NUMINAMATH_GPT_ex1_simplified_ex2_simplified_l762_76279


namespace NUMINAMATH_GPT_xy_equals_18_l762_76241

theorem xy_equals_18 (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 :=
by
  sorry

end NUMINAMATH_GPT_xy_equals_18_l762_76241


namespace NUMINAMATH_GPT_retailer_profit_percentage_l762_76233

theorem retailer_profit_percentage (items_sold : ℕ) (profit_per_item : ℝ) (discount_rate : ℝ)
  (discounted_items_needed : ℝ) (total_profit : ℝ) (item_cost : ℝ) :
  items_sold = 100 → 
  profit_per_item = 30 →
  discount_rate = 0.05 →
  discounted_items_needed = 156.86274509803923 →
  total_profit = 3000 →
  (discounted_items_needed * ((item_cost + profit_per_item) * (1 - discount_rate) - item_cost) = total_profit) →
  ((profit_per_item / item_cost) * 100 = 16) :=
by {
  sorry 
}

end NUMINAMATH_GPT_retailer_profit_percentage_l762_76233


namespace NUMINAMATH_GPT_melted_mixture_weight_l762_76227

/-- 
If the ratio of zinc to copper is 9:11 and 27 kg of zinc has been consumed, then the total weight of the melted mixture is 60 kg.
-/
theorem melted_mixture_weight (zinc_weight : ℕ) (ratio_zinc_to_copper : ℕ → ℕ → Prop)
  (h_ratio : ratio_zinc_to_copper 9 11) (h_zinc : zinc_weight = 27) :
  ∃ (total_weight : ℕ), total_weight = 60 :=
by
  sorry

end NUMINAMATH_GPT_melted_mixture_weight_l762_76227


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l762_76280

-- Expression simplification proof statement 1
theorem simplify_expr1 (m n : ℤ) : 
  (5 * m + 3 * n - 7 * m - n) = (-2 * m + 2 * n) :=
sorry

-- Expression simplification proof statement 2
theorem simplify_expr2 (x : ℤ) : 
  (2 * x^2 - (3 * x - 2 * (x^2 - x + 3) + 2 * x^2)) = (2 * x^2 - 5 * x + 6) :=
sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l762_76280


namespace NUMINAMATH_GPT_complement_of_M_in_U_l762_76202

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}

theorem complement_of_M_in_U : (U \ M) = {2, 4, 6} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_in_U_l762_76202


namespace NUMINAMATH_GPT_max_min_f_in_rectangle_l762_76235

def f (x y : ℝ) : ℝ := x^3 + y^3 + 6 * x * y

def in_rectangle (x y : ℝ) : Prop := 
  (-3 ≤ x ∧ x ≤ 1) ∧ (-3 ≤ y ∧ y ≤ 2)

theorem max_min_f_in_rectangle :
  ∃ (x_max y_max x_min y_min : ℝ),
    in_rectangle x_max y_max ∧ in_rectangle x_min y_min ∧
    (∀ x y, in_rectangle x y → f x y ≤ f x_max y_max) ∧
    (∀ x y, in_rectangle x y → f x_min y_min ≤ f x y) ∧
    f x_max y_max = 21 ∧ f x_min y_min = -55 :=
by
  sorry

end NUMINAMATH_GPT_max_min_f_in_rectangle_l762_76235


namespace NUMINAMATH_GPT_smallest_prime_with_digit_sum_23_l762_76223

noncomputable def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Prime p ∧ digit_sum p = 23 ∧ ∀ q : ℕ, Prime q ∧ digit_sum q = 23 → p ≤ q :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_with_digit_sum_23_l762_76223


namespace NUMINAMATH_GPT_find_a_max_min_f_l762_76264

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.exp x

theorem find_a (a : ℝ) (h : (deriv (f a) 0 = 1)) : a = 1 :=
by sorry

noncomputable def f_one (x : ℝ) : ℝ := f 1 x

theorem max_min_f (h : ∀ x, 0 ≤ x → x ≤ 2 → deriv (f_one) x > 0) :
  (f_one 0 = 0) ∧ (f_one 2 = 2 * Real.exp 2) :=
by sorry

end NUMINAMATH_GPT_find_a_max_min_f_l762_76264


namespace NUMINAMATH_GPT_find_subtracted_number_l762_76265

theorem find_subtracted_number (t k x : ℝ) (h1 : t = 20) (h2 : k = 68) (h3 : t = 5/9 * (k - x)) :
  x = 32 :=
by
  sorry

end NUMINAMATH_GPT_find_subtracted_number_l762_76265


namespace NUMINAMATH_GPT_A_alone_finishes_in_27_days_l762_76285

noncomputable def work (B : ℝ) : ℝ := 54 * B  -- amount of work W
noncomputable def days_to_finish_alone (B : ℝ) : ℝ := (work B) / (2 * B)

theorem A_alone_finishes_in_27_days (B : ℝ) (h : (work B) / (2 * B + B) = 18) : 
  days_to_finish_alone B = 27 :=
by
  sorry

end NUMINAMATH_GPT_A_alone_finishes_in_27_days_l762_76285


namespace NUMINAMATH_GPT_behavior_on_1_2_l762_76262

/-- Definition of an odd function -/
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

/-- Definition of being decreasing on an interval -/
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≥ f y

/-- Definition of having a minimum value on an interval -/
def has_minimum_on (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  ∀ x, a ≤ x → x ≤ b → f x ≥ m

theorem behavior_on_1_2 
  {f : ℝ → ℝ} 
  (h_odd : is_odd_function f) 
  (h_dec : is_decreasing_on f (-2) (-1)) 
  (h_min : has_minimum_on f (-2) (-1) 3) :
  is_decreasing_on f 1 2 ∧ ∀ x, 1 ≤ x → x ≤ 2 → f x ≤ -3 := 
by 
  sorry

end NUMINAMATH_GPT_behavior_on_1_2_l762_76262


namespace NUMINAMATH_GPT_common_chord_eq_l762_76266

theorem common_chord_eq : 
  (∀ x y : ℝ, x^2 + y^2 + 2*x + 8*y - 8 = 0) ∧ 
  (∀ x y : ℝ, x^2 + y^2 - 4*x - 4*y - 2 = 0) → 
  (∀ x y : ℝ, x + 2*y - 1 = 0) :=
by 
  sorry

end NUMINAMATH_GPT_common_chord_eq_l762_76266


namespace NUMINAMATH_GPT_number_of_sodas_l762_76272

theorem number_of_sodas (cost_sandwich : ℝ) (num_sandwiches : ℕ) (cost_soda : ℝ) (total_cost : ℝ):
  cost_sandwich = 2.45 → 
  num_sandwiches = 2 → 
  cost_soda = 0.87 → 
  total_cost = 8.38 → 
  (total_cost - num_sandwiches * cost_sandwich) / cost_soda = 4 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_number_of_sodas_l762_76272


namespace NUMINAMATH_GPT_square_perimeter_l762_76268

theorem square_perimeter (s : ℝ)
  (h1 : ∃ (s : ℝ), 4 * s = s * 1 + s / 4 * 1 + s * 1 + s / 4 * 1)
  (h2 : ∃ (P : ℝ), P = 4 * s)
  : (5/2) * s = 40 → 4 * s = 64 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_square_perimeter_l762_76268


namespace NUMINAMATH_GPT_speed_of_water_l762_76284

-- Definitions based on conditions
def swim_speed_in_still_water : ℝ := 4
def distance_against_current : ℝ := 6
def time_against_current : ℝ := 3
def effective_speed (v : ℝ) : ℝ := swim_speed_in_still_water - v

-- Theorem to prove the speed of the water
theorem speed_of_water (v : ℝ) : 
  effective_speed v * time_against_current = distance_against_current → 
  v = 2 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_water_l762_76284


namespace NUMINAMATH_GPT_mod_pow_solution_l762_76296

def m (x : ℕ) := x

theorem mod_pow_solution :
  ∃ (m : ℕ), 0 ≤ m ∧ m < 8 ∧ 13^6 % 8 = m ∧ m = 1 :=
by
  use 1
  sorry

end NUMINAMATH_GPT_mod_pow_solution_l762_76296


namespace NUMINAMATH_GPT_correct_result_l762_76257

-- Given condition
def mistaken_calculation (x : ℤ) : Prop :=
  x / 3 = 45

-- Proposition to prove the correct result
theorem correct_result (x : ℤ) (h : mistaken_calculation x) : 3 * x = 405 := by
  -- Here we can solve the proof later
  sorry

end NUMINAMATH_GPT_correct_result_l762_76257


namespace NUMINAMATH_GPT_value_of_a2_sub_b2_l762_76203

theorem value_of_a2_sub_b2 (a b : ℝ) (h1 : a + b = 6) (h2 : a - b = 2) : a^2 - b^2 = 12 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a2_sub_b2_l762_76203


namespace NUMINAMATH_GPT_range_of_a_l762_76254

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → 2 * x * Real.log x ≥ -x^2 + a * x - 3) → a ≤ 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l762_76254


namespace NUMINAMATH_GPT_find_a_plus_b_plus_c_l762_76273

-- Definitions of conditions
def is_vertex (a b c : ℝ) (vertex_x vertex_y : ℝ) := 
  ∀ x : ℝ, vertex_y = (a * (vertex_x ^ 2)) + (b * vertex_x) + c

def contains_point (a b c : ℝ) (x y : ℝ) := 
  y = (a * (x ^ 2)) + (b * x) + c

theorem find_a_plus_b_plus_c
  (a b c : ℝ)
  (h_vertex : is_vertex a b c 3 4)
  (h_symmetry : ∃ h : ℝ, ∀ x : ℝ, a * (x - h) ^ 2 = a * (h - x) ^ 2)
  (h_contains : contains_point a b c 1 0)
  : a + b + c = 0 := 
sorry

end NUMINAMATH_GPT_find_a_plus_b_plus_c_l762_76273


namespace NUMINAMATH_GPT_identify_set_A_l762_76294

open Set

def A : Set ℕ := {x | 0 ≤ x ∧ x < 3}

theorem identify_set_A : A = {0, 1, 2} := 
by
  sorry

end NUMINAMATH_GPT_identify_set_A_l762_76294


namespace NUMINAMATH_GPT_find_percentage_l762_76228

theorem find_percentage (P : ℝ) (h : P / 100 * 3200 = 0.20 * 650 + 190) : P = 10 :=
by 
  sorry

end NUMINAMATH_GPT_find_percentage_l762_76228


namespace NUMINAMATH_GPT_gain_percentage_l762_76229

theorem gain_percentage (selling_price gain : ℝ) (h_selling : selling_price = 90) (h_gain : gain = 15) : 
  (gain / (selling_price - gain)) * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_gain_percentage_l762_76229


namespace NUMINAMATH_GPT_f_2015_2016_l762_76287

theorem f_2015_2016 (f : ℤ → ℤ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_periodic : ∀ x, f (x + 2) = -f x)
  (h_f1 : f 1 = 2) :
  f 2015 + f 2016 = -2 :=
sorry

end NUMINAMATH_GPT_f_2015_2016_l762_76287


namespace NUMINAMATH_GPT_total_rainfall_2010_to_2012_l762_76288

noncomputable def average_rainfall (year : ℕ) : ℕ :=
  if year = 2010 then 35
  else if year = 2011 then 38
  else if year = 2012 then 41
  else 0

theorem total_rainfall_2010_to_2012 :
  (12 * average_rainfall 2010) + 
  (12 * average_rainfall 2011) + 
  (12 * average_rainfall 2012) = 1368 :=
by
  sorry

end NUMINAMATH_GPT_total_rainfall_2010_to_2012_l762_76288


namespace NUMINAMATH_GPT_find_value_of_x_squared_plus_one_over_x_squared_l762_76240

theorem find_value_of_x_squared_plus_one_over_x_squared (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 :=
by 
  sorry

end NUMINAMATH_GPT_find_value_of_x_squared_plus_one_over_x_squared_l762_76240


namespace NUMINAMATH_GPT_johns_total_weekly_gas_consumption_l762_76211

-- Definitions of conditions
def highway_mpg : ℝ := 30
def city_mpg : ℝ := 25
def work_miles_each_way : ℝ := 20
def work_days_per_week : ℝ := 5
def highway_miles_each_way : ℝ := 15
def city_miles_each_way : ℝ := 5
def leisure_highway_miles_per_week : ℝ := 30
def leisure_city_miles_per_week : ℝ := 10
def idling_gas_consumption_per_week : ℝ := 0.3

-- Proof problem
theorem johns_total_weekly_gas_consumption :
  let work_commute_miles_per_week := work_miles_each_way * 2 * work_days_per_week
  let highway_miles_work := highway_miles_each_way * 2 * work_days_per_week
  let city_miles_work := city_miles_each_way * 2 * work_days_per_week
  let total_highway_miles := highway_miles_work + leisure_highway_miles_per_week
  let total_city_miles := city_miles_work + leisure_city_miles_per_week
  let highway_gas_consumption := total_highway_miles / highway_mpg
  let city_gas_consumption := total_city_miles / city_mpg
  (highway_gas_consumption + city_gas_consumption + idling_gas_consumption_per_week) = 8.7 := by
  sorry

end NUMINAMATH_GPT_johns_total_weekly_gas_consumption_l762_76211


namespace NUMINAMATH_GPT_polygon_sides_16_l762_76298

def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

noncomputable def arithmetic_sequence_sum (a1 an : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (a1 + an) / 2

theorem polygon_sides_16 (n : ℕ) (a1 an : ℝ) (d : ℝ) 
  (h1 : d = 5) (h2 : an = 160) (h3 : a1 = 160 - 5 * (n - 1))
  (h4 : arithmetic_sequence_sum a1 an d n = sum_of_interior_angles n)
  : n = 16 :=
sorry

end NUMINAMATH_GPT_polygon_sides_16_l762_76298


namespace NUMINAMATH_GPT_ages_sum_l762_76230

theorem ages_sum (Beckett_age Olaf_age Shannen_age Jack_age : ℕ) 
  (h1 : Beckett_age = 12) 
  (h2 : Olaf_age = Beckett_age + 3) 
  (h3 : Shannen_age = Olaf_age - 2) 
  (h4 : Jack_age = 2 * Shannen_age + 5) : 
  Beckett_age + Olaf_age + Shannen_age + Jack_age = 71 := 
by
  sorry

end NUMINAMATH_GPT_ages_sum_l762_76230


namespace NUMINAMATH_GPT_water_added_l762_76282

theorem water_added (W X : ℝ) 
  (h1 : 45 / W = 2 / 1)
  (h2 : 45 / (W + X) = 6 / 5) : 
  X = 15 := 
by
  sorry

end NUMINAMATH_GPT_water_added_l762_76282


namespace NUMINAMATH_GPT_find_second_number_l762_76286

theorem find_second_number (x y z : ℝ) 
  (h1 : x + y + z = 120) 
  (h2 : x = (3/4) * y) 
  (h3 : z = (9/7) * y) 
  : y = 40 :=
sorry

end NUMINAMATH_GPT_find_second_number_l762_76286


namespace NUMINAMATH_GPT_rate_of_interest_l762_76200

theorem rate_of_interest (SI P T R : ℝ) 
  (hSI : SI = 4016.25) 
  (hP : P = 6693.75) 
  (hT : T = 5) 
  (h : SI = (P * R * T) / 100) : 
  R = 12 :=
by 
  sorry

end NUMINAMATH_GPT_rate_of_interest_l762_76200


namespace NUMINAMATH_GPT_min_x_plus_y_l762_76226

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) :
  x + y ≥ 16 :=
sorry

end NUMINAMATH_GPT_min_x_plus_y_l762_76226


namespace NUMINAMATH_GPT_power_function_decreasing_n_value_l762_76222

theorem power_function_decreasing_n_value (n : ℤ) (f : ℝ → ℝ) :
  (∀ x : ℝ, 0 < x → f x = (n^2 + 2 * n - 2) * x^(n^2 - 3 * n)) →
  (∀ x y : ℝ, 0 < x ∧ 0 < y → x < y → f y < f x) →
  n = 1 := 
by
  sorry

end NUMINAMATH_GPT_power_function_decreasing_n_value_l762_76222


namespace NUMINAMATH_GPT_polynomial_remainder_l762_76269

theorem polynomial_remainder (y : ℂ) (h1 : y^5 + y^4 + y^3 + y^2 + y + 1 = 0) (h2 : y^6 = 1) :
  (y^55 + y^40 + y^25 + y^10 + 1) % (y^5 + y^4 + y^3 + y^2 + y + 1) = 2 * y + 3 :=
sorry

end NUMINAMATH_GPT_polynomial_remainder_l762_76269


namespace NUMINAMATH_GPT_original_proposition_converse_inverse_contrapositive_l762_76215

def is_integer (x : ℝ) : Prop := ∃ (n : ℤ), x = n
def is_real (x : ℝ) : Prop := true

theorem original_proposition (x : ℝ) : is_integer x → is_real x := 
by sorry

theorem converse (x : ℝ) : ¬(is_real x → is_integer x) := 
by sorry

theorem inverse (x : ℝ) : ¬((¬ is_integer x) → (¬ is_real x)) := 
by sorry

theorem contrapositive (x : ℝ) : (¬ is_real x) → (¬ is_integer x) := 
by sorry

end NUMINAMATH_GPT_original_proposition_converse_inverse_contrapositive_l762_76215


namespace NUMINAMATH_GPT_salary_increase_after_three_years_l762_76218

-- Define the initial salary S and the raise percentage 12%
def initial_salary (S : ℝ) : ℝ := S
def raise_percentage : ℝ := 0.12

-- Define the salary after n raises
def salary_after_raises (S : ℝ) (n : ℕ) : ℝ :=
  S * (1 + raise_percentage)^n

-- Prove that the percentage increase after 3 years is 40.49%
theorem salary_increase_after_three_years (S : ℝ) :
  ((salary_after_raises S 3 - S) / S) * 100 = 40.49 :=
by sorry

end NUMINAMATH_GPT_salary_increase_after_three_years_l762_76218


namespace NUMINAMATH_GPT_gcd_abcd_dcba_l762_76209

-- Definitions based on the conditions
def abcd (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d
def dcba (a b c d : ℕ) : ℕ := 1000 * d + 100 * c + 10 * b + a
def consecutive_digits (a b c d : ℕ) : Prop := (b = a + 1) ∧ (c = a + 2) ∧ (d = a + 3)

-- Theorem statement
theorem gcd_abcd_dcba (a b c d : ℕ) (h : consecutive_digits a b c d) : 
  Nat.gcd (abcd a b c d + dcba a b c d) 1111 = 1111 :=
sorry

end NUMINAMATH_GPT_gcd_abcd_dcba_l762_76209


namespace NUMINAMATH_GPT_millions_place_correct_l762_76256

def number := 345000000
def hundred_millions_place := number / 100000000 % 10  -- 3
def ten_millions_place := number / 10000000 % 10  -- 4
def millions_place := number / 1000000 % 10  -- 5

theorem millions_place_correct : millions_place = 5 := 
by 
  -- Mathematical proof goes here
  sorry

end NUMINAMATH_GPT_millions_place_correct_l762_76256


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l762_76260

variable (α : ℝ)

theorem trigonometric_identity_proof
  (h : Real.tan (α + Real.pi / 4) = -3) :
  Real.cos (2 * α) + 2 * Real.sin (2 * α) = 1 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l762_76260


namespace NUMINAMATH_GPT_total_number_of_guests_l762_76220

theorem total_number_of_guests (A C S : ℕ) (hA : A = 58) (hC : C = A - 35) (hS : S = 2 * C) : 
  A + C + S = 127 := 
by
  sorry

end NUMINAMATH_GPT_total_number_of_guests_l762_76220


namespace NUMINAMATH_GPT_total_votes_is_120_l762_76267

-- Define the conditions
def Fiona_votes : ℕ := 48
def fraction_of_votes : ℚ := 2 / 5

-- The proof goal
theorem total_votes_is_120 (V : ℕ) (h : Fiona_votes = fraction_of_votes * V) : V = 120 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_is_120_l762_76267


namespace NUMINAMATH_GPT_coefficient_of_x_in_expansion_l762_76217

noncomputable def binomial_expansion_term (r : ℕ) : ℤ :=
  (-1)^r * (2^(5-r)) * Nat.choose 5 r

theorem coefficient_of_x_in_expansion :
  binomial_expansion_term 3 = -40 := by
  sorry

end NUMINAMATH_GPT_coefficient_of_x_in_expansion_l762_76217


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l762_76250

theorem hyperbola_eccentricity 
  (a b e : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : e = Real.sqrt (1 + (b^2 / a^2))) 
  (h4 : e ≤ Real.sqrt 5) : 
  e = 2 := 
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l762_76250


namespace NUMINAMATH_GPT_Maddie_bought_two_white_packs_l762_76224

theorem Maddie_bought_two_white_packs 
  (W : ℕ)
  (total_cost : ℕ)
  (cost_per_shirt : ℕ)
  (white_pack_size : ℕ)
  (blue_pack_size : ℕ)
  (blue_packs : ℕ)
  (cost_per_white_pack : ℕ)
  (cost_per_blue_pack : ℕ) :
  total_cost = 66 ∧ cost_per_shirt = 3 ∧ white_pack_size = 5 ∧ blue_pack_size = 3 ∧ blue_packs = 4 ∧ cost_per_white_pack = white_pack_size * cost_per_shirt ∧ cost_per_blue_pack = blue_pack_size * cost_per_shirt ∧ 3 * (white_pack_size * W + blue_pack_size * blue_packs) = total_cost → W = 2 :=
by
  sorry

end NUMINAMATH_GPT_Maddie_bought_two_white_packs_l762_76224


namespace NUMINAMATH_GPT_smallest_b_l762_76293

theorem smallest_b (a b : ℝ) (h1 : 2 < a) (h2 : a < b) 
(h3 : 2 + a ≤ b) (h4 : 1 / a + 1 / b ≤ 2) : b = 2 :=
sorry

end NUMINAMATH_GPT_smallest_b_l762_76293


namespace NUMINAMATH_GPT_minimum_score_4th_quarter_l762_76271

theorem minimum_score_4th_quarter (q1 q2 q3 : ℕ) (q4 : ℕ) :
  q1 = 85 → q2 = 80 → q3 = 90 →
  (q1 + q2 + q3 + q4) / 4 ≥ 85 →
  q4 ≥ 85 :=
by intros hq1 hq2 hq3 h_avg
   sorry

end NUMINAMATH_GPT_minimum_score_4th_quarter_l762_76271


namespace NUMINAMATH_GPT_xiao_ming_water_usage_ge_8_l762_76205

def min_monthly_water_usage (x : ℝ) : Prop :=
  ∀ (c : ℝ), c ≥ 15 →
    (c = if x ≤ 5 then x * 1.8 else (5 * 1.8 + (x - 5) * 2)) →
      x ≥ 8

theorem xiao_ming_water_usage_ge_8 : ∃ x : ℝ, min_monthly_water_usage x :=
  sorry

end NUMINAMATH_GPT_xiao_ming_water_usage_ge_8_l762_76205


namespace NUMINAMATH_GPT_combination_10_5_l762_76253

theorem combination_10_5 :
  (Nat.choose 10 5) = 2520 :=
by
  sorry

end NUMINAMATH_GPT_combination_10_5_l762_76253


namespace NUMINAMATH_GPT_algebra_expression_value_l762_76221

theorem algebra_expression_value (a b : ℝ) (h : a - 2 * b = -1) : 1 - 2 * a + 4 * b = 3 :=
by
  sorry

end NUMINAMATH_GPT_algebra_expression_value_l762_76221


namespace NUMINAMATH_GPT_gage_skating_time_l762_76291

theorem gage_skating_time :
  let minutes_skated_first_5_days := 75 * 5
  let minutes_skated_next_3_days := 90 * 3
  let total_minutes_skated_8_days := minutes_skated_first_5_days + minutes_skated_next_3_days
  let total_minutes_required := 85 * 9
  let minutes_needed_ninth_day := total_minutes_required - total_minutes_skated_8_days
  minutes_needed_ninth_day = 120 :=
by
  let minutes_skated_first_5_days := 75 * 5
  let minutes_skated_next_3_days := 90 * 3
  let total_minutes_skated_8_days := minutes_skated_first_5_days + minutes_skated_next_3_days
  let total_minutes_required := 85 * 9
  let minutes_needed_ninth_day := total_minutes_required - total_minutes_skated_8_days
  sorry

end NUMINAMATH_GPT_gage_skating_time_l762_76291


namespace NUMINAMATH_GPT_inequality_holds_for_all_x_l762_76206

theorem inequality_holds_for_all_x (m : ℝ) : 
  (∀ x : ℝ, (x^2 + m * x - 1) / (2 * x^2 - 2 * x + 3) < 1) ↔ -6 < m ∧ m < 2 := 
sorry -- Proof to be provided

end NUMINAMATH_GPT_inequality_holds_for_all_x_l762_76206


namespace NUMINAMATH_GPT_students_suggested_tomatoes_l762_76259

theorem students_suggested_tomatoes (students_total mashed_potatoes bacon tomatoes : ℕ) 
  (h_total : students_total = 826)
  (h_mashed_potatoes : mashed_potatoes = 324)
  (h_bacon : bacon = 374)
  (h_tomatoes : students_total = mashed_potatoes + bacon + tomatoes) :
  tomatoes = 128 :=
by {
  sorry
}

end NUMINAMATH_GPT_students_suggested_tomatoes_l762_76259


namespace NUMINAMATH_GPT_rectangle_area_l762_76299

/-- 
In the rectangle \(ABCD\), \(AD - AB = 9\) cm. The area of trapezoid \(ABCE\) is 5 times 
the area of triangle \(ADE\). The perimeter of triangle \(ADE\) is 68 cm less than the 
perimeter of trapezoid \(ABCE\). Prove that the area of the rectangle \(ABCD\) 
is 3060 square centimeters.
-/
theorem rectangle_area (AB AD : ℝ) (S_ABC : ℝ) (S_ADE : ℝ) (P_ADE : ℝ) (P_ABC : ℝ) :
  AD - AB = 9 →
  S_ABC = 5 * S_ADE →
  P_ADE = P_ABC - 68 →
  (AB * AD = 3060) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l762_76299


namespace NUMINAMATH_GPT_rectangle_area_l762_76207

-- Define the width and length of the rectangle
def w : ℚ := 20 / 3
def l : ℚ := 2 * w

-- Define the perimeter constraint
def perimeter_condition : Prop := 2 * (l + w) = 40

-- Define the area of the rectangle
def area : ℚ := l * w

-- The theorem to prove
theorem rectangle_area : perimeter_condition → area = 800 / 9 :=
by
  intro h
  have hw : w = 20 / 3 := rfl
  have hl : l = 2 * w := rfl
  have hp : 2 * (l + w) = 40 := h
  sorry

end NUMINAMATH_GPT_rectangle_area_l762_76207


namespace NUMINAMATH_GPT_total_profit_l762_76263

-- Definitions
def investment_a : ℝ := 45000
def investment_b : ℝ := 63000
def investment_c : ℝ := 72000
def c_share : ℝ := 24000

-- Theorem statement
theorem total_profit : (investment_a + investment_b + investment_c) * (c_share / investment_c) = 60000 := by
  sorry

end NUMINAMATH_GPT_total_profit_l762_76263


namespace NUMINAMATH_GPT_flowers_in_each_row_l762_76283

theorem flowers_in_each_row (rows : ℕ) (total_remaining_flowers : ℕ) 
  (percentage_remaining : ℚ) (correct_rows : rows = 50) 
  (correct_remaining : total_remaining_flowers = 8000) 
  (correct_percentage : percentage_remaining = 0.40) :
  (total_remaining_flowers : ℚ) / percentage_remaining / (rows : ℚ) = 400 := 
by {
 sorry
}

end NUMINAMATH_GPT_flowers_in_each_row_l762_76283


namespace NUMINAMATH_GPT_f_value_at_3_l762_76239

def f (x : ℝ) := 2 * (x + 1) + 1

theorem f_value_at_3 : f 3 = 9 :=
by sorry

end NUMINAMATH_GPT_f_value_at_3_l762_76239


namespace NUMINAMATH_GPT_original_weight_of_potatoes_l762_76212

theorem original_weight_of_potatoes (W : ℝ) (h : W / (W / 2) = 36) : W = 648 := by
  sorry

end NUMINAMATH_GPT_original_weight_of_potatoes_l762_76212


namespace NUMINAMATH_GPT_fraction_product_l762_76278

theorem fraction_product (a b : ℕ) 
  (h1 : 1/5 < a / b)
  (h2 : a / b < 1/4)
  (h3 : b ≤ 19) :
  ∃ a1 a2 b1 b2, 4 * a2 < b1 ∧ b1 < 5 * a2 ∧ b2 ≤ 19 ∧ 4 * a2 < b2 ∧ b2 < 20 ∧ a = 4 ∧ b = 19 ∧ a1 = 2 ∧ b1 = 9 ∧ 
  (a + b = 23 ∨ a + b = 11) ∧ (23 * 11 = 253) := by
  sorry

end NUMINAMATH_GPT_fraction_product_l762_76278


namespace NUMINAMATH_GPT_sum_of_roots_is_k_over_5_l762_76249

noncomputable def sum_of_roots 
  (x1 x2 k d : ℝ) 
  (hx : x1 ≠ x2) 
  (h1 : 5 * x1^2 - k * x1 = d) 
  (h2 : 5 * x2^2 - k * x2 = d) : ℝ :=
x1 + x2

theorem sum_of_roots_is_k_over_5 
  {x1 x2 k d : ℝ} 
  (hx : x1 ≠ x2) 
  (h1 : 5 * x1^2 - k * x1 = d) 
  (h2 : 5 * x2^2 - k * x2 = d) : 
  sum_of_roots x1 x2 k d hx h1 h2 = k / 5 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_is_k_over_5_l762_76249


namespace NUMINAMATH_GPT_log_equation_l762_76244

theorem log_equation (x : ℝ) (h0 : x < 1) (h1 : (Real.log x / Real.log 10)^3 - 3 * (Real.log x / Real.log 10) = 243) :
  (Real.log x / Real.log 10)^4 - 4 * (Real.log x / Real.log 10) = 6597 :=
by
  sorry

end NUMINAMATH_GPT_log_equation_l762_76244


namespace NUMINAMATH_GPT_infinite_primes_p_solutions_eq_p2_l762_76243

theorem infinite_primes_p_solutions_eq_p2 :
  ∃ᶠ p in Filter.atTop, Prime p ∧ 
  (∃ (S : Finset (ZMod p × ZMod p × ZMod p)),
    S.card = p^2 ∧ ∀ (x y z : ZMod p), (3 * x^3 + 4 * y^4 + 5 * z^3 - y^4 * z = 0) ↔ (x, y, z) ∈ S) :=
sorry

end NUMINAMATH_GPT_infinite_primes_p_solutions_eq_p2_l762_76243
