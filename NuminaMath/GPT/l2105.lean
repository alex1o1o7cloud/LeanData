import Mathlib

namespace simplify_expression_l2105_210539

theorem simplify_expression : 8 * (15 / 4) * (-56 / 45) = -112 / 3 :=
by sorry

end simplify_expression_l2105_210539


namespace inequality_am_gm_l2105_210538

theorem inequality_am_gm (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end inequality_am_gm_l2105_210538


namespace cheaper_store_difference_in_cents_l2105_210540

/-- Given the following conditions:
1. Best Deals offers \$12 off the list price of \$52.99.
2. Market Value offers 20% off the list price of \$52.99.
 -/
theorem cheaper_store_difference_in_cents :
  let list_price : ℝ := 52.99
  let best_deals_price := list_price - 12
  let market_value_price := list_price * 0.80
  best_deals_price < market_value_price →
  let difference_in_dollars := market_value_price - best_deals_price
  let difference_in_cents := difference_in_dollars * 100
  difference_in_cents = 140 := by
  intro h
  let list_price : ℝ := 52.99
  let best_deals_price := list_price - 12
  let market_value_price := list_price * 0.80
  let difference_in_dollars := market_value_price - best_deals_price
  let difference_in_cents := difference_in_dollars * 100
  sorry

end cheaper_store_difference_in_cents_l2105_210540


namespace parabola_shifting_produces_k_l2105_210543

theorem parabola_shifting_produces_k
  (k : ℝ)
  (h1 : -k/2 > 0)
  (h2 : (0 : ℝ) = (((0 : ℝ) - 3) + k/2)^2 - (5*k^2)/4 + 1)
  :
  k = -5 :=
sorry

end parabola_shifting_produces_k_l2105_210543


namespace total_cost_of_mangoes_l2105_210597

-- Definition of prices per dozen in one box
def prices_per_dozen : List ℕ := [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

-- Number of dozens per box (constant for all boxes)
def dozens_per_box : ℕ := 10

-- Number of boxes
def number_of_boxes : ℕ := 36

-- Calculate the total cost of mangoes in all boxes
theorem total_cost_of_mangoes :
  (prices_per_dozen.sum * number_of_boxes = 3060) := by
  -- Proof goes here
  sorry

end total_cost_of_mangoes_l2105_210597


namespace minimum_value_y_range_of_a_l2105_210524

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*a*x - 1 + a

theorem minimum_value_y (x : ℝ) 
  (hx_pos : x > 0) : (f x 2 / x) = -2 :=
by sorry

theorem range_of_a : 
  ∀ a : ℝ, ∀ x ∈ (Set.Icc 0 2), (f x a) ≤ a ↔ a ≥ 3 / 4 :=
by sorry

end minimum_value_y_range_of_a_l2105_210524


namespace volume_first_cube_l2105_210569

theorem volume_first_cube (a b : ℝ) (h_ratio : a = 3 * b) (h_volume : b^3 = 8) : a^3 = 216 :=
by
  sorry

end volume_first_cube_l2105_210569


namespace molecular_weight_of_one_mole_l2105_210555

-- Conditions
def molecular_weight_6_moles : ℤ := 1404
def num_moles : ℤ := 6

-- Theorem
theorem molecular_weight_of_one_mole : (molecular_weight_6_moles / num_moles) = 234 := by
  sorry

end molecular_weight_of_one_mole_l2105_210555


namespace present_age_of_son_l2105_210561

variable (S M : ℕ)

-- Conditions
def age_difference : Prop := M = S + 40
def age_relation_in_seven_years : Prop := M + 7 = 3 * (S + 7)

-- Theorem to prove
theorem present_age_of_son : age_difference S M → age_relation_in_seven_years S M → S = 13 := by
  sorry

end present_age_of_son_l2105_210561


namespace right_triangle_ratio_l2105_210584

theorem right_triangle_ratio (a b c r s : ℝ) (h_right_angle : (a:ℝ)^2 + (b:ℝ)^2 = c^2)
  (h_perpendicular : ∀ h : ℝ, c = r + s)
  (h_ratio_ab : a / b = 2 / 5)
  (h_geometry_r : r = a^2 / c)
  (h_geometry_s : s = b^2 / c) :
  r / s = 4 / 25 :=
sorry

end right_triangle_ratio_l2105_210584


namespace harriet_trip_time_l2105_210579

theorem harriet_trip_time
  (speed_AB : ℕ := 100)
  (speed_BA : ℕ := 150)
  (total_trip_time : ℕ := 5)
  (time_threshold : ℕ := 180) :
  let D := (speed_AB * speed_BA * total_trip_time) / (speed_AB + speed_BA)
  let time_AB := D / speed_AB
  let time_AB_min := time_AB * 60
  time_AB_min = time_threshold :=
by
  sorry

end harriet_trip_time_l2105_210579


namespace total_amount_collected_l2105_210535

-- Define ticket prices and quantities
def adult_ticket_price : ℕ := 12
def child_ticket_price : ℕ := 4
def total_tickets_sold : ℕ := 130
def adult_tickets_sold : ℕ := 40

-- Calculate the number of child tickets sold
def child_tickets_sold : ℕ := total_tickets_sold - adult_tickets_sold

-- Calculate the total amount collected from adult tickets
def total_adult_amount_collected : ℕ := adult_tickets_sold * adult_ticket_price

-- Calculate the total amount collected from child tickets
def total_child_amount_collected : ℕ := child_tickets_sold * child_ticket_price

-- Prove the total amount collected from ticket sales
theorem total_amount_collected : total_adult_amount_collected + total_child_amount_collected = 840 := by
  sorry

end total_amount_collected_l2105_210535


namespace melanie_picked_plums_l2105_210572

variable (picked_plums : ℕ)
variable (given_plums : ℕ := 3)
variable (total_plums : ℕ := 10)

theorem melanie_picked_plums :
  picked_plums + given_plums = total_plums → picked_plums = 7 := by
  sorry

end melanie_picked_plums_l2105_210572


namespace sum_series_eq_one_l2105_210595

noncomputable def sum_series : ℝ :=
  ∑' n : ℕ, (2^n + 1) / (3^(2^n) + 1)

theorem sum_series_eq_one : sum_series = 1 := 
by 
  sorry

end sum_series_eq_one_l2105_210595


namespace turnip_heavier_than_zhuchka_l2105_210550

theorem turnip_heavier_than_zhuchka {C B M T : ℝ} 
  (h1 : B = 3 * C)
  (h2 : M = C / 10)
  (h3 : T = 60 * M) : 
  T / B = 2 :=
by
  sorry

end turnip_heavier_than_zhuchka_l2105_210550


namespace robin_total_spending_l2105_210552

def jelly_bracelets_total_cost : ℕ :=
  let names := ["Jessica", "Tori", "Lily", "Patrice"]
  let total_letters := names.foldl (λ acc name => acc + name.length) 0
  total_letters * 2

theorem robin_total_spending : jelly_bracelets_total_cost = 44 := by
  sorry

end robin_total_spending_l2105_210552


namespace math_problem_l2105_210575

theorem math_problem (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : x + y + x * y = 1) :
  x * y + 1 / (x * y) - y / x - x / y = 4 :=
sorry

end math_problem_l2105_210575


namespace shoe_store_sale_l2105_210556

theorem shoe_store_sale (total_sneakers : ℕ) (total_sandals : ℕ) (total_shoes : ℕ) (total_boots : ℕ) 
  (h1 : total_sneakers = 2) 
  (h2 : total_sandals = 4) 
  (h3 : total_shoes = 17) 
  (h4 : total_boots = total_shoes - (total_sneakers + total_sandals)) : 
  total_boots = 11 :=
by
  rw [h1, h2, h3] at h4
  exact h4
-- sorry

end shoe_store_sale_l2105_210556


namespace g_of_3_l2105_210502

def g (x : ℝ) : ℝ := 5 * x ^ 4 + 4 * x ^ 3 - 7 * x ^ 2 + 3 * x - 2

theorem g_of_3 : g 3 = 401 :=
by
    -- proof will go here
    sorry

end g_of_3_l2105_210502


namespace area_of_square_field_l2105_210581

-- Define the side length of the square
def side_length : ℝ := 15

-- Define the area of the square based on the side length
def square_area (side : ℝ) : ℝ := side * side

-- The theorem stating the area of a square with side length 15 meters
theorem area_of_square_field : square_area side_length = 225 := 
by 
  sorry

end area_of_square_field_l2105_210581


namespace percentage_discount_on_pencils_l2105_210536

-- Establish the given conditions
variable (cucumbers pencils price_per_cucumber price_per_pencil total_spent : ℕ)
variable (h1 : cucumbers = 100)
variable (h2 : price_per_cucumber = 20)
variable (h3 : price_per_pencil = 20)
variable (h4 : total_spent = 2800)
variable (h5 : cucumbers = 2 * pencils)

-- Propose the statement to be proved
theorem percentage_discount_on_pencils : 20 * pencils * price_per_pencil = 20 * (total_spent - cucumbers * price_per_cucumber) ∧ pencils = 50 ∧ ((total_spent - cucumbers * price_per_cucumber) * 100 = 80 * pencils * price_per_pencil) :=
by
  sorry

end percentage_discount_on_pencils_l2105_210536


namespace nikita_productivity_l2105_210547

theorem nikita_productivity 
  (x y : ℕ) 
  (h1 : 3 * x + 2 * y = 7) 
  (h2 : 5 * x + 3 * y = 11) : 
  y = 2 := 
sorry

end nikita_productivity_l2105_210547


namespace evaluate_g_at_neg3_l2105_210510

def g (x : ℤ) : ℤ := x^2 - x + 2 * x^3

theorem evaluate_g_at_neg3 : g (-3) = -42 := by
  sorry

end evaluate_g_at_neg3_l2105_210510


namespace value_of_k_l2105_210594

noncomputable def roots_in_ratio_equation {k : ℝ} (h : k ≠ 0) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ 
  (r₁ / r₂ = 3) ∧ 
  (r₁ + r₂ = -8) ∧ 
  (r₁ * r₂ = k)

theorem value_of_k (k : ℝ) (h : k ≠ 0) (hr : roots_in_ratio_equation h) : k = 12 :=
sorry

end value_of_k_l2105_210594


namespace point_in_fourth_quadrant_l2105_210570

def point (x y : ℝ) := (x, y)
def x_positive (p : ℝ × ℝ) : Prop := p.1 > 0
def y_negative (p : ℝ × ℝ) : Prop := p.2 < 0
def in_fourth_quadrant (p : ℝ × ℝ) : Prop := x_positive p ∧ y_negative p

theorem point_in_fourth_quadrant : in_fourth_quadrant (2, -4) :=
by
  -- The proof states that (2, -4) is in the fourth quadrant.
  sorry

end point_in_fourth_quadrant_l2105_210570


namespace converse_false_inverse_false_l2105_210585

-- Definitions of the conditions
def is_rhombus (Q : Type) : Prop := -- definition of a rhombus
  sorry

def is_parallelogram (Q : Type) : Prop := -- definition of a parallelogram
  sorry

variable {Q : Type}

-- Initial statement: If a quadrilateral is a rhombus, then it is a parallelogram.
axiom initial_statement : is_rhombus Q → is_parallelogram Q

-- Goals: Prove both the converse and inverse are false
theorem converse_false : ¬ ((is_parallelogram Q) → (is_rhombus Q)) :=
sorry

theorem inverse_false : ¬ (¬ (is_rhombus Q) → ¬ (is_parallelogram Q)) :=
    sorry

end converse_false_inverse_false_l2105_210585


namespace masha_dolls_l2105_210551

theorem masha_dolls (n : ℕ) (h : (n / 2) * 1 + (n / 4) * 2 + (n / 4) * 4 = 24) : n = 12 :=
sorry

end masha_dolls_l2105_210551


namespace sum_squares_not_perfect_square_l2105_210589

theorem sum_squares_not_perfect_square (x y z : ℤ) (h : x^2 + y^2 + z^2 = 1993) : ¬ ∃ a : ℤ, x + y + z = a^2 :=
sorry

end sum_squares_not_perfect_square_l2105_210589


namespace cost_of_purchasing_sandwiches_and_sodas_l2105_210508

def sandwich_price : ℕ := 4
def soda_price : ℕ := 1
def num_sandwiches : ℕ := 6
def num_sodas : ℕ := 5
def total_cost : ℕ := 29

theorem cost_of_purchasing_sandwiches_and_sodas :
  (num_sandwiches * sandwich_price + num_sodas * soda_price) = total_cost :=
by
  sorry

end cost_of_purchasing_sandwiches_and_sodas_l2105_210508


namespace maximum_value_l2105_210533

theorem maximum_value (R P K : ℝ) (h₁ : 3 * Real.sqrt 3 * R ≥ P) (h₂ : K = P * R / 4) : 
  (K * P) / (R^3) ≤ 27 / 4 :=
by
  sorry

end maximum_value_l2105_210533


namespace base_2_base_3_product_is_144_l2105_210531

def convert_base_2_to_10 (n : ℕ) : ℕ :=
  match n with
  | 1001 => 9
  | _ => 0 -- For simplicity, only handle 1001_2

def convert_base_3_to_10 (n : ℕ) : ℕ :=
  match n with
  | 121 => 16
  | _ => 0 -- For simplicity, only handle 121_3

theorem base_2_base_3_product_is_144 :
  convert_base_2_to_10 1001 * convert_base_3_to_10 121 = 144 :=
by
  sorry

end base_2_base_3_product_is_144_l2105_210531


namespace samatha_routes_l2105_210534

-- Definitions based on the given conditions
def blocks_from_house_to_southwest_corner := 4
def blocks_through_park := 1
def blocks_from_northeast_corner_to_school := 4
def blocks_from_school_to_library := 3

-- Number of ways to arrange movements
def number_of_routes_house_to_southwest : ℕ :=
  Nat.choose blocks_from_house_to_southwest_corner 1

def number_of_routes_through_park : ℕ := blocks_through_park

def number_of_routes_northeast_to_school : ℕ :=
  Nat.choose blocks_from_northeast_corner_to_school 1

def number_of_routes_school_to_library : ℕ :=
  Nat.choose blocks_from_school_to_library 1

-- Total number of different routes
def total_number_of_routes : ℕ :=
  number_of_routes_house_to_southwest *
  number_of_routes_through_park *
  number_of_routes_northeast_to_school *
  number_of_routes_school_to_library

theorem samatha_routes (n : ℕ) (h : n = 48) :
  total_number_of_routes = n :=
  by
    -- Proof is skipped
    sorry

end samatha_routes_l2105_210534


namespace triangular_weight_60_l2105_210583

def round_weight := ℝ
def triangular_weight := ℝ
def rectangular_weight := 90

variables (c t : ℝ)

-- Conditions
axiom condition1 : c + t = 3 * c
axiom condition2 : 4 * c + t = t + c + rectangular_weight

theorem triangular_weight_60 : t = 60 :=
  sorry

end triangular_weight_60_l2105_210583


namespace perimeter_ratio_l2105_210560

variables (K T k R r : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (r = 2 * T / K)
def condition2 : Prop := (2 * T = R * k)

-- The statement we want to prove
theorem perimeter_ratio :
  condition1 K T r →
  condition2 T R k →
  K / k = R / r :=
by
  intros h1 h2
  sorry

end perimeter_ratio_l2105_210560


namespace tylenol_intake_proof_l2105_210519

noncomputable def calculate_tylenol_intake_grams
  (tablet_mg : ℕ) (tablets_per_dose : ℕ) (hours_per_dose : ℕ) (total_hours : ℕ) : ℕ :=
  let doses := total_hours / hours_per_dose
  let total_mg := doses * tablets_per_dose * tablet_mg
  total_mg / 1000

theorem tylenol_intake_proof : calculate_tylenol_intake_grams 500 2 4 12 = 3 :=
  by sorry

end tylenol_intake_proof_l2105_210519


namespace problem_statement_l2105_210518

def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 2 * x - 3

theorem problem_statement : (f (g (f 1))) / (g (f (g 1))) = (-23 : ℝ) / 5 :=
by 
  sorry

end problem_statement_l2105_210518


namespace domain_g_l2105_210546

def domain_f (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 2
def g (x : ℝ) (f : ℝ → ℝ) : Prop := 
  ((1 < x) ∧ (x ≤ Real.sqrt 3)) ∧ domain_f (x^2 - 1) ∧ (0 < x - 1 ∧ x - 1 < 1)

theorem domain_g (x : ℝ) (f : ℝ → ℝ) (hf : ∀ a, domain_f a → True) : 
  g x f ↔ 1 < x ∧ x ≤ Real.sqrt 3 :=
by 
  sorry

end domain_g_l2105_210546


namespace original_selling_price_l2105_210562

-- Definitions and conditions
def cost_price (CP : ℝ) := CP
def profit (CP : ℝ) := 1.25 * CP
def loss (CP : ℝ) := 0.75 * CP
def loss_price (CP : ℝ) := 600

-- Main theorem statement
theorem original_selling_price (CP : ℝ) (h1 : loss CP = loss_price CP) : profit CP = 1000 :=
by
  -- Note: adding the proof that CP = 800 and then profit CP = 1000 would be here.
  sorry

end original_selling_price_l2105_210562


namespace mutually_exclusive_not_necessarily_complementary_l2105_210576

-- Define what it means for events to be mutually exclusive
def mutually_exclusive (E1 E2 : Prop) : Prop :=
  ¬ (E1 ∧ E2)

-- Define what it means for events to be complementary
def complementary (E1 E2 : Prop) : Prop :=
  (E1 ∨ E2) ∧ ¬ (E1 ∧ E2) ∧ (¬ E1 ∨ ¬ E2)

theorem mutually_exclusive_not_necessarily_complementary :
  ∀ E1 E2 : Prop, mutually_exclusive E1 E2 → ¬ complementary E1 E2 :=
sorry

end mutually_exclusive_not_necessarily_complementary_l2105_210576


namespace find_m_of_inverse_proportion_l2105_210500

theorem find_m_of_inverse_proportion (k : ℝ) (m : ℝ) 
(A_cond : (-1) * 3 = k) 
(B_cond : 2 * m = k) : 
m = -3 / 2 := 
by 
  sorry

end find_m_of_inverse_proportion_l2105_210500


namespace water_cost_is_1_l2105_210565

-- Define the conditions
def cost_cola : ℝ := 3
def cost_juice : ℝ := 1.5
def bottles_sold_cola : ℝ := 15
def bottles_sold_juice : ℝ := 12
def bottles_sold_water : ℝ := 25
def total_revenue : ℝ := 88

-- Compute the revenue from cola and juice
def revenue_cola : ℝ := bottles_sold_cola * cost_cola
def revenue_juice : ℝ := bottles_sold_juice * cost_juice

-- Define a proof that the cost of a bottle of water is $1
theorem water_cost_is_1 : (total_revenue - revenue_cola - revenue_juice) / bottles_sold_water = 1 :=
by
  -- Proof is omitted
  sorry

end water_cost_is_1_l2105_210565


namespace train_stoppage_time_l2105_210506

theorem train_stoppage_time
    (speed_without_stoppages : ℕ)
    (speed_with_stoppages : ℕ)
    (time_unit : ℕ)
    (h1 : speed_without_stoppages = 50)
    (h2 : speed_with_stoppages = 30)
    (h3 : time_unit = 60) :
    (time_unit * (speed_without_stoppages - speed_with_stoppages) / speed_without_stoppages) = 24 :=
by
  sorry

end train_stoppage_time_l2105_210506


namespace orchard_trees_l2105_210578

theorem orchard_trees (n : ℕ) (hn : n^2 + 146 = 7890) : 
    n^2 + 146 + 31 = 89^2 := by
  sorry

end orchard_trees_l2105_210578


namespace three_buses_interval_l2105_210591

theorem three_buses_interval (interval_two_buses : ℕ) (loop_time : ℕ) :
  interval_two_buses = 21 →
  loop_time = interval_two_buses * 2 →
  (loop_time / 3) = 14 :=
by
  intros h1 h2
  rw [h1] at h2
  simp at h2
  sorry

end three_buses_interval_l2105_210591


namespace digit_B_in_4B52B_divisible_by_9_l2105_210509

theorem digit_B_in_4B52B_divisible_by_9 (B : ℕ) (h : (2 * B + 11) % 9 = 0) : B = 8 :=
by {
  sorry
}

end digit_B_in_4B52B_divisible_by_9_l2105_210509


namespace complete_square_solution_l2105_210512

theorem complete_square_solution (x : ℝ) :
  x^2 - 2*x - 3 = 0 → (x - 1)^2 = 4 :=
by
  sorry

end complete_square_solution_l2105_210512


namespace triangle_properties_l2105_210549

-- Definitions of sides of the triangle
def a : ℕ := 15
def b : ℕ := 11
def c : ℕ := 18

-- Definition of the triangle inequality theorem in the context
def triangle_inequality (x y z : ℕ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

-- Perimeter calculation
def perimeter (x y z : ℕ) : ℕ :=
  x + y + z

-- Stating the proof problem
theorem triangle_properties : triangle_inequality a b c ∧ perimeter a b c = 44 :=
by
  -- Start the process for the actual proof that will be filled out
  sorry

end triangle_properties_l2105_210549


namespace quadratic_expression_rewrite_l2105_210593

theorem quadratic_expression_rewrite (a b c : ℝ) :
  (∀ x : ℝ, 6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) → a + b + c = 171 :=
sorry

end quadratic_expression_rewrite_l2105_210593


namespace goods_train_length_l2105_210580

noncomputable def length_of_goods_train (speed_first_train_kmph speed_goods_train_kmph time_seconds : ℕ) : ℝ :=
  let relative_speed_kmph := speed_first_train_kmph + speed_goods_train_kmph
  let relative_speed_mps := (relative_speed_kmph : ℝ) * (5.0 / 18.0)
  relative_speed_mps * (time_seconds : ℝ)

theorem goods_train_length
  (speed_first_train_kmph : ℕ) (speed_goods_train_kmph : ℕ) (time_seconds : ℕ) 
  (h1 : speed_first_train_kmph = 50)
  (h2 : speed_goods_train_kmph = 62)
  (h3 : time_seconds = 9) :
  length_of_goods_train speed_first_train_kmph speed_goods_train_kmph time_seconds = 280 :=
  sorry

end goods_train_length_l2105_210580


namespace find_k_l2105_210528

theorem find_k 
  (t k r : ℝ)
  (h1 : t = 5 / 9 * (k - 32))
  (h2 : r = 3 * t)
  (h3 : r = 150) : 
  k = 122 := 
sorry

end find_k_l2105_210528


namespace solve_for_y_l2105_210522

theorem solve_for_y (y : ℝ) : 7 - y = 4 → y = 3 :=
by
  sorry

end solve_for_y_l2105_210522


namespace problem1_problem2_l2105_210563

namespace MathProofs

theorem problem1 : (-3 - (-8) + (-6) + 10) = 9 :=
by
  sorry

theorem problem2 : (-12 * ((1 : ℚ) / 6 - (1 : ℚ) / 3 - 3 / 4)) = 11 :=
by
  sorry

end MathProofs

end problem1_problem2_l2105_210563


namespace operation_on_b_l2105_210568

theorem operation_on_b (t b0 b1 : ℝ) (h : t * b1^4 = 16 * t * b0^4) : b1 = 2 * b0 :=
by
  sorry

end operation_on_b_l2105_210568


namespace minimum_value_of_x_minus_y_l2105_210592

variable (x y : ℝ)
open Real

theorem minimum_value_of_x_minus_y (hx : x > 0) (hy : y < 0) 
  (h : (1 / (x + 2)) + (1 / (1 - y)) = 1 / 6) : 
  x - y = 21 :=
sorry

end minimum_value_of_x_minus_y_l2105_210592


namespace rhombus_diagonal_l2105_210545

theorem rhombus_diagonal
  (d1 : ℝ) (d2 : ℝ) (area : ℝ) 
  (h1 : d1 = 17) (h2 : area = 170) 
  (h3 : area = (d1 * d2) / 2) : d2 = 20 :=
by
  sorry

end rhombus_diagonal_l2105_210545


namespace domain_condition_l2105_210598

variable (k : ℝ)
def quadratic_expression (x : ℝ) : ℝ := k * x^2 - 4 * k * x + k + 8

theorem domain_condition (k : ℝ) : (∀ x : ℝ, quadratic_expression k x > 0) ↔ (0 ≤ k ∧ k < 8/3) :=
sorry

end domain_condition_l2105_210598


namespace angle_at_intersection_l2105_210527

theorem angle_at_intersection (n : ℕ) (h₁ : n = 8)
  (h₂ : ∀ i j : ℕ, (i + 1) % n ≠ j ∧ i < j)
  (h₃ : ∀ i : ℕ, i < n)
  (h₄ : ∀ i j : ℕ, (i + 1) % n = j ∨ (i + n - 1) % n = j)
  : (2 * (180 / n - (180 * (n - 2) / n) / 2)) = 90 :=
by
  sorry

end angle_at_intersection_l2105_210527


namespace rachel_weight_l2105_210529

theorem rachel_weight :
  ∃ R : ℝ, (R + (R + 6) + (R - 15)) / 3 = 72 ∧ R = 75 :=
by
  sorry

end rachel_weight_l2105_210529


namespace shaded_area_correct_l2105_210596

-- Define the side lengths of the squares
def side_length_large_square : ℕ := 14
def side_length_small_square : ℕ := 10

-- Define the areas of the squares
def area_large_square : ℕ := side_length_large_square * side_length_large_square
def area_small_square : ℕ := side_length_small_square * side_length_small_square

-- Define the area of the shaded regions
def area_shaded_regions : ℕ := area_large_square - area_small_square

-- State the theorem
theorem shaded_area_correct : area_shaded_regions = 49 := by
  sorry

end shaded_area_correct_l2105_210596


namespace product_repeating_decimal_l2105_210526

theorem product_repeating_decimal (p : ℚ) (h₁ : p = 152 / 333) : 
  p * 7 = 1064 / 333 :=
  by
    sorry

end product_repeating_decimal_l2105_210526


namespace rectangle_perimeter_l2105_210559

theorem rectangle_perimeter (z w : ℝ) (h : z > w) :
  (2 * ((z - w) + w)) = 2 * z := by
  sorry

end rectangle_perimeter_l2105_210559


namespace find_word_l2105_210505

theorem find_word (antonym : Nat) (cond : antonym = 26) : String :=
  "seldom"

end find_word_l2105_210505


namespace a5_gt_b5_l2105_210573

variables {a_n b_n : ℕ → ℝ}
variables {a1 b1 a3 b3 : ℝ}
variables {q : ℝ} {d : ℝ}

/-- Given conditions -/
axiom h1 : a1 = b1
axiom h2 : a1 > 0
axiom h3 : a3 = b3
axiom h4 : a3 = a1 * q^2
axiom h5 : b3 = a1 + 2 * d
axiom h6 : a1 ≠ a3

/-- Prove that a_5 is greater than b_5 -/
theorem a5_gt_b5 : a1 * q^4 > a1 + 4 * d :=
by sorry

end a5_gt_b5_l2105_210573


namespace mass_of_man_proof_l2105_210599

def volume_displaced (L B h : ℝ) : ℝ :=
  L * B * h

def mass_of_man (V ρ : ℝ) : ℝ :=
  ρ * V

theorem mass_of_man_proof :
  ∀ (L B h ρ : ℝ), L = 9 → B = 3 → h = 0.01 → ρ = 1000 →
  mass_of_man (volume_displaced L B h) ρ = 270 :=
by
  intros L B h ρ L_eq B_eq h_eq ρ_eq
  rw [L_eq, B_eq, h_eq, ρ_eq]
  unfold volume_displaced
  unfold mass_of_man
  simp
  sorry

end mass_of_man_proof_l2105_210599


namespace find_c_l2105_210582

-- Define the problem conditions and statement

variables (a b c : ℝ) (A B C : ℝ)
variable (cos_C : ℝ)
variable (sin_A sin_B : ℝ)

-- Given conditions
axiom h1 : a = 2
axiom h2 : cos_C = -1/4
axiom h3 : 3 * sin_A = 2 * sin_B
axiom sine_rule : sin_A / a = sin_B / b

-- Using sine rule to derive relation between a and b
axiom h4 : 3 * a = 2 * b

-- Cosine rule axiom
axiom cosine_rule : c^2 = a^2 + b^2 - 2 * a * b * cos_C

-- Prove c = 4
theorem find_c : c = 4 :=
by
  sorry

end find_c_l2105_210582


namespace angle_in_third_quadrant_l2105_210513

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.cos α < 0) (h2 : Real.tan α > 0) : 
  (π < α ∧ α < 3 * π / 2) :=
sorry

end angle_in_third_quadrant_l2105_210513


namespace tanya_number_75_less_l2105_210553

def rotate180 (d : ℕ) : ℕ :=
  match d with
  | 0 => 0
  | 1 => 1
  | 6 => 9
  | 8 => 8
  | 9 => 6
  | _ => 0 -- invalid assumption for digits outside the defined scope

def two_digit_upside_down (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  10 * rotate180 units + rotate180 tens

theorem tanya_number_75_less (n : ℕ) : 
  ∀ n, (∃ a b, n = 10 * a + b ∧ (a = 0 ∨ a = 1 ∨ a = 6 ∨ a = 8 ∨ a = 9) ∧ 
      (b = 0 ∨ b = 1 ∨ b = 6 ∨ b = 8 ∨ b = 9) ∧  
      n - two_digit_upside_down n = 75) :=
by {
  sorry
}

end tanya_number_75_less_l2105_210553


namespace count_1320_factors_l2105_210525

-- Prime factorization function
def primeFactors (n : ℕ) : List ℕ :=
  sorry

-- Count factors function based on prime factorization
def countFactors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem count_1320_factors : countFactors [(2, 3), (3, 1), (5, 1), (11, 1)] = 32 :=
by
  sorry

end count_1320_factors_l2105_210525


namespace emily_stickers_l2105_210574

theorem emily_stickers:
  ∃ S : ℕ, (S % 4 = 2) ∧
           (S % 6 = 2) ∧
           (S % 9 = 2) ∧
           (S % 10 = 2) ∧
           (S > 2) ∧
           (S = 182) :=
  sorry

end emily_stickers_l2105_210574


namespace square_circle_area_ratio_l2105_210501

theorem square_circle_area_ratio {r : ℝ} (h : ∀ s : ℝ, 2 * r = s * Real.sqrt 2) :
  (2 * r ^ 2) / (Real.pi * r ^ 2) = 2 / Real.pi :=
by
  sorry

end square_circle_area_ratio_l2105_210501


namespace sum_of_cubes_eq_five_l2105_210544

noncomputable def root_polynomial (a b c : ℂ) : Prop :=
  (a + b + c = 2) ∧ (a*b + b*c + c*a = 3) ∧ (a*b*c = 5)

theorem sum_of_cubes_eq_five (a b c : ℂ) (h : root_polynomial a b c) :
  a^3 + b^3 + c^3 = 5 :=
sorry

end sum_of_cubes_eq_five_l2105_210544


namespace factorization_of_polynomial_solve_quadratic_equation_l2105_210516

-- Problem 1: Factorization
theorem factorization_of_polynomial : ∀ y : ℝ, 2 * y^2 - 8 = 2 * (y + 2) * (y - 2) :=
by
  intro y
  sorry

-- Problem 2: Solving the quadratic equation
theorem solve_quadratic_equation : ∀ x : ℝ, x^2 + 4 * x + 3 = 0 ↔ x = -1 ∨ x = -3 :=
by
  intro x
  sorry

end factorization_of_polynomial_solve_quadratic_equation_l2105_210516


namespace angle_relation_in_triangle_l2105_210548

theorem angle_relation_in_triangle
    (a b c : ℝ)
    (A B C : ℝ)
    (h1 : b * (a + b) * (b + c) = a^3 + b * (a^2 + c^2) + c^3)
    (h2 : A + B + C = π) 
    (h3 : A > 0) 
    (h4 : B > 0) 
    (h5 : C > 0) :
    (1 / (Real.sqrt A + Real.sqrt B)) + (1 / (Real.sqrt B + Real.sqrt C)) = (2 / (Real.sqrt C + Real.sqrt A)) :=
sorry

end angle_relation_in_triangle_l2105_210548


namespace min_value_quadratic_l2105_210504

theorem min_value_quadratic (x : ℝ) : -2 * x^2 + 8 * x + 5 ≥ -2 * (2 - x)^2 + 13 :=
by
  sorry

end min_value_quadratic_l2105_210504


namespace corridor_perimeter_l2105_210557

theorem corridor_perimeter
  (P1 P2 : ℕ)
  (h₁ : P1 = 16)
  (h₂ : P2 = 24) : 
  2 * ((P2 / 4 + (P1 + P2) / 4) + (P2 / 4) - (P1 / 4)) = 40 :=
by {
  -- The proof can be filled here
  sorry
}

end corridor_perimeter_l2105_210557


namespace line_through_intersection_of_circles_l2105_210503

theorem line_through_intersection_of_circles :
  ∀ (x y : ℝ),
    (x^2 + y^2 + 4 * x - 4 * y - 12 = 0) ∧
    (x^2 + y^2 + 2 * x + 4 * y - 4 = 0) →
    (x - 4 * y - 4 = 0) :=
by sorry

end line_through_intersection_of_circles_l2105_210503


namespace students_only_english_l2105_210515

variable (total_students both_english_german enrolled_german: ℕ)

theorem students_only_english :
  total_students = 45 ∧ both_english_german = 12 ∧ enrolled_german = 22 ∧
  (∀ S E G B : ℕ, S = total_students ∧ B = both_english_german ∧ G = enrolled_german - B ∧
   (S = E + G + B) → E = 23) :=
by
  sorry

end students_only_english_l2105_210515


namespace range_of_a_satisfies_l2105_210523

noncomputable def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-(x + 1)) = -f (x + 1)) ∧
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ 0 ≤ x2 → (f x1 - f x2) / (x1 - x2) > -1) ∧
  (∀ a : ℝ, f (a^2 - 1) + f (a - 1) + a^2 + a > 2)

theorem range_of_a_satisfies (f : ℝ → ℝ) (hf_conditions : satisfies_conditions f) :
  {a : ℝ | f (a^2 - 1) + f (a - 1) + a^2 + a > 2} = {a | a < -2 ∨ a > 1} :=
by
  sorry

end range_of_a_satisfies_l2105_210523


namespace correct_conclusions_l2105_210520

noncomputable def quadratic_solution_set (a b c : ℝ) : Prop :=
  ∀ x : ℝ, (-1 / 2 < x ∧ x < 3) ↔ (a * x^2 + b * x + c > 0)

theorem correct_conclusions (a b c : ℝ) (h : quadratic_solution_set a b c) : c > 0 ∧ 4 * a + 2 * b + c > 0 :=
  sorry

end correct_conclusions_l2105_210520


namespace opposite_of_neg_one_third_l2105_210541

theorem opposite_of_neg_one_third : -(- (1 / 3)) = (1 / 3) :=
by sorry

end opposite_of_neg_one_third_l2105_210541


namespace penny_identified_whales_l2105_210558

theorem penny_identified_whales (sharks eels total : ℕ)
  (h_sharks : sharks = 35)
  (h_eels   : eels = 15)
  (h_total  : total = 55) :
  total - (sharks + eels) = 5 :=
by
  sorry

end penny_identified_whales_l2105_210558


namespace jessica_games_attended_l2105_210590

def total_games : ℕ := 6
def games_missed_by_jessica : ℕ := 4

theorem jessica_games_attended : total_games - games_missed_by_jessica = 2 := by
  sorry

end jessica_games_attended_l2105_210590


namespace arithmetic_sequence_a5_l2105_210532

variable (a : ℕ → ℝ)

theorem arithmetic_sequence_a5 (h : a 2 + a 8 = 15 - a 5) : a 5 = 5 :=
by
  sorry

end arithmetic_sequence_a5_l2105_210532


namespace pencils_distribution_count_l2105_210530

def count_pencils_distribution : ℕ :=
  let total_pencils := 10
  let friends := 4
  let adjusted_pencils := total_pencils - friends
  Nat.choose (adjusted_pencils + friends - 1) (friends - 1)

theorem pencils_distribution_count :
  count_pencils_distribution = 84 := 
  by sorry

end pencils_distribution_count_l2105_210530


namespace largest_d_l2105_210537

variable (a b c d : ℤ)

def condition : Prop := a + 2 = b - 1 ∧ a + 2 = c + 3 ∧ a + 2 = d - 4

theorem largest_d (h : condition a b c d) : d > a ∧ d > b ∧ d > c :=
by
  -- Assuming the condition holds, we need to prove d > a, d > b, and d > c
  sorry

end largest_d_l2105_210537


namespace jackson_difference_l2105_210507

theorem jackson_difference :
  let Jackson_initial := 500
  let Brandon_initial := 500
  let Meagan_initial := 700
  let Jackson_final := Jackson_initial * 4
  let Brandon_final := Brandon_initial * 0.20
  let Meagan_final := Meagan_initial + (Meagan_initial * 0.50)
  Jackson_final - (Brandon_final + Meagan_final) = 850 :=
by
  sorry

end jackson_difference_l2105_210507


namespace sector_area_l2105_210564

theorem sector_area (r α l S : ℝ) (h1 : l + 2 * r = 8) (h2 : α = 2) (h3 : l = α * r) :
  S = 4 :=
by
  -- Let the radius be 2 as a condition derived from h1 and h2
  have r := 2
  -- Substitute and compute to find S
  have S_calculated := (1 / 2 * α * r * r)
  sorry

end sector_area_l2105_210564


namespace black_white_area_ratio_l2105_210521

theorem black_white_area_ratio :
  let r1 := 2
  let r2 := 6
  let r3 := 10
  let r4 := 14
  let r5 := 18
  let area (r : ℝ) := π * r^2
  let black_area := area r1 + (area r3 - area r2) + (area r5 - area r4)
  let white_area := (area r2 - area r1) + (area r4 - area r3)
  black_area / white_area = (49 : ℝ) / 32 :=
by
  sorry

end black_white_area_ratio_l2105_210521


namespace probability_of_selecting_A_and_B_l2105_210554

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l2105_210554


namespace fraction_decomposition_l2105_210542

theorem fraction_decomposition :
  (1 : ℚ) / 4 = (1 : ℚ) / 8 + (1 : ℚ) / 8 := 
by
  -- proof goes here
  sorry

end fraction_decomposition_l2105_210542


namespace max_value_of_expression_l2105_210577

theorem max_value_of_expression (a b c : ℝ) (ha : 0 ≤ a) (ha2 : a ≤ 2) (hb : 0 ≤ b) (hb2 : b ≤ 2) (hc : 0 ≤ c) (hc2 : c ≤ 2) :
  2 * Real.sqrt (abc / 8) + Real.sqrt ((2 - a) * (2 - b) * (2 - c)) ≤ 2 :=
by
  sorry

end max_value_of_expression_l2105_210577


namespace geometric_sequence_q_cubed_l2105_210517

theorem geometric_sequence_q_cubed (q a_1 : ℝ) (h1 : q ≠ 0) (h2 : q ≠ 1) 
(h3 : 2 * (a_1 * (1 - q^9) / (1 - q)) = (a_1 * (1 - q^3) / (1 - q)) + (a_1 * (1 - q^6) / (1 - q))) : 
  q^3 = -1/2 := by
  sorry

end geometric_sequence_q_cubed_l2105_210517


namespace num_integer_terms_sequence_l2105_210514

noncomputable def sequence_starting_at_8820 : Nat := 8820

def divide_by_5 (n : Nat) : Nat := n / 5

theorem num_integer_terms_sequence :
  let seq := [sequence_starting_at_8820, divide_by_5 sequence_starting_at_8820]
  seq = [8820, 1764] →
  seq.length = 2 := by
  sorry

end num_integer_terms_sequence_l2105_210514


namespace fangfang_travel_time_l2105_210571

theorem fangfang_travel_time (time_1_to_5 : ℕ) (start_floor end_floor : ℕ) (floors_1_to_5 : ℕ) (floors_2_to_7 : ℕ) :
  time_1_to_5 = 40 →
  floors_1_to_5 = 5 - 1 →
  floors_2_to_7 = 7 - 2 →
  end_floor = 7 →
  start_floor = 2 →
  (end_floor - start_floor) * (time_1_to_5 / floors_1_to_5) = 50 :=
by 
  sorry

end fangfang_travel_time_l2105_210571


namespace determinant_of_matrixA_l2105_210566

variable (x : ℝ)

def matrixA : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x + 2, x, x], ![x, x + 2, x], ![x, x, x + 2]]

theorem determinant_of_matrixA : Matrix.det (matrixA x) = 8 * x + 8 := by
  sorry

end determinant_of_matrixA_l2105_210566


namespace monotonic_sufficient_not_necessary_maximum_l2105_210588

noncomputable def f (x : ℝ) : ℝ := sorry  -- Placeholder for the function f
def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop := (∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y) ∨ (∀ x y, a ≤ x → x ≤ y → y ≤ b → f y ≤ f x)
def has_max_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∃ M, ∀ x, a ≤ x → x ≤ b → f x ≤ M

theorem monotonic_sufficient_not_necessary_maximum : 
  ∀ f : ℝ → ℝ,
  ∀ a b : ℝ,
  a ≤ b →
  monotonic_on f a b → 
  has_max_on f a b :=
sorry  -- Proof is omitted

end monotonic_sufficient_not_necessary_maximum_l2105_210588


namespace find_y_l2105_210567

theorem find_y (x y : ℝ) (h1 : 2 * x - 3 * y = 24) (h2 : x + 2 * y = 15) : y = 6 / 7 :=
by sorry

end find_y_l2105_210567


namespace correct_divisor_l2105_210586

noncomputable def dividend := 12 * 35

theorem correct_divisor (x : ℕ) : (x * 20 = dividend) → x = 21 :=
sorry

end correct_divisor_l2105_210586


namespace diff_of_two_numbers_l2105_210587

theorem diff_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : |x - y| = 22 :=
sorry

end diff_of_two_numbers_l2105_210587


namespace solve_for_x_l2105_210511

open Real

theorem solve_for_x (x : ℝ) (h1 : x > 0) (h2 : 6 * sqrt (4 + x) + 6 * sqrt (4 - x) = 9 * sqrt 2) : 
  x = sqrt 255 / 4 :=
sorry

end solve_for_x_l2105_210511
