import Mathlib

namespace AB_not_together_correct_l1164_116457

-- Definitions based on conditions
def total_people : ℕ := 5

-- The result from the complementary counting principle
def total_arrangements : ℕ := 120
def AB_together_arrangements : ℕ := 48

-- The arrangement count of A and B not next to each other
def AB_not_together_arrangements : ℕ := total_arrangements - AB_together_arrangements

theorem AB_not_together_correct : 
  AB_not_together_arrangements = 72 :=
sorry

end AB_not_together_correct_l1164_116457


namespace candy_bar_calories_l1164_116468

theorem candy_bar_calories (calories : ℕ) (bars : ℕ) (dozen : ℕ) (total_calories : ℕ) 
  (H1 : total_calories = 2016) (H2 : bars = 42) (H3 : dozen = 12) 
  (H4 : total_calories = bars * calories) : 
  calories / dozen = 4 := 
by 
  sorry

end candy_bar_calories_l1164_116468


namespace suff_not_nec_condition_l1164_116410

/-- f is an even function --/
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- Condition x1 + x2 = 0 --/
def sum_eq_zero (x1 x2 : ℝ) : Prop := x1 + x2 = 0

/-- Prove: sufficient but not necessary condition --/
theorem suff_not_nec_condition (f : ℝ → ℝ) (h_even : is_even f) (x1 x2 : ℝ) :
  sum_eq_zero x1 x2 → f x1 - f x2 = 0 ∧ (f x1 - f x2 = 0 → ¬ sum_eq_zero x1 x2) :=
by
  sorry

end suff_not_nec_condition_l1164_116410


namespace expand_polynomial_l1164_116488

theorem expand_polynomial : 
  (∀ (x : ℝ), (5 * x^3 + 7) * (3 * x + 4) = 15 * x^4 + 20 * x^3 + 21 * x + 28) :=
by
  intro x
  sorry

end expand_polynomial_l1164_116488


namespace plane_through_points_l1164_116471

-- Define the vectors as tuples of three integers
def point := (ℤ × ℤ × ℤ)

-- The given points
def p : point := (2, -1, 3)
def q : point := (4, -1, 5)
def r : point := (5, -3, 4)

-- A function to find the equation of the plane given three points
def plane_equation (p q r : point) : ℤ × ℤ × ℤ × ℤ :=
  let (px, py, pz) := p
  let (qx, qy, qz) := q
  let (rx, ry, rz) := r
  let a := (qy - py) * (rz - pz) - (qy - py) * (rz - pz)
  let b := (qx - px) * (rz - pz) - (qx - px) * (rz - pz)
  let c := (qx - px) * (ry - py) - (qx - px) * (ry - py)
  let d := -(a * px + b * py + c * pz)
  (a, b, c, d)

-- The proof statement
theorem plane_through_points : plane_equation (2, -1, 3) (4, -1, 5) (5, -3, 4) = (1, 2, -2, 6) :=
  by sorry

end plane_through_points_l1164_116471


namespace work_completion_alternate_days_l1164_116455

theorem work_completion_alternate_days (h₁ : ∀ (work : ℝ), ∃ a_days : ℝ, a_days = 12 → (∀ t : ℕ, t / a_days <= work / 12))
                                      (h₂ : ∀ (work : ℝ), ∃ b_days : ℝ, b_days = 36 → (∀ t : ℕ, t / b_days <= work / 36)) :
  ∃ days : ℝ, days = 18 := by
  sorry

end work_completion_alternate_days_l1164_116455


namespace angle_rotation_l1164_116465

theorem angle_rotation (α : ℝ) (β : ℝ) (k : ℤ) :
  (∃ k' : ℤ, α + 30 = 120 + 360 * k') →
  (β = 360 * k + 90) ↔ (∃ k'' : ℤ, β = 360 * k'' + α) :=
by
  sorry

end angle_rotation_l1164_116465


namespace combined_salaries_of_B_C_D_E_l1164_116437

theorem combined_salaries_of_B_C_D_E
    (A_salary : ℕ)
    (average_salary_all : ℕ)
    (total_individuals : ℕ)
    (combined_salaries_B_C_D_E : ℕ) :
    A_salary = 8000 →
    average_salary_all = 8800 →
    total_individuals = 5 →
    combined_salaries_B_C_D_E = (average_salary_all * total_individuals) - A_salary →
    combined_salaries_B_C_D_E = 36000 :=
by
  sorry

end combined_salaries_of_B_C_D_E_l1164_116437


namespace margie_change_l1164_116456

def cost_of_banana_cents : ℕ := 30
def cost_of_orange_cents : ℕ := 60
def num_bananas : ℕ := 4
def num_oranges : ℕ := 2
def amount_paid_dollars : ℝ := 10.0

noncomputable def cost_of_banana_dollars := (cost_of_banana_cents : ℝ) / 100
noncomputable def cost_of_orange_dollars := (cost_of_orange_cents : ℝ) / 100

noncomputable def total_cost := 
  (num_bananas * cost_of_banana_dollars) + (num_oranges * cost_of_orange_dollars)

noncomputable def change_received := amount_paid_dollars - total_cost

theorem margie_change : change_received = 7.60 := 
by sorry

end margie_change_l1164_116456


namespace chocolate_bars_l1164_116416

theorem chocolate_bars (num_small_boxes : ℕ) (num_bars_per_box : ℕ) (total_bars : ℕ) (h1 : num_small_boxes = 20) (h2 : num_bars_per_box = 32) (h3 : total_bars = num_small_boxes * num_bars_per_box) :
  total_bars = 640 :=
by
  sorry

end chocolate_bars_l1164_116416


namespace sum_of_all_possible_values_is_correct_l1164_116440

noncomputable def M_sum_of_all_possible_values (a b c M : ℝ) : Prop :=
  M = a * b * c ∧ M = 8 * (a + b + c) ∧ c = a + b ∧ b = 2 * a

theorem sum_of_all_possible_values_is_correct :
  ∃ M, (∃ a b c, M_sum_of_all_possible_values a b c M) ∧ M = 96 * Real.sqrt 2 := by
  sorry

end sum_of_all_possible_values_is_correct_l1164_116440


namespace inscribed_squares_ratio_l1164_116492

theorem inscribed_squares_ratio (x y : ℝ) (h1 : ∃ (x : ℝ), x * (13 * 12 + 13 * 5 - 5 * 12) = 60) 
  (h2 : ∃ (y : ℝ), 30 * y = 13 ^ 2) :
  x / y = 1800 / 2863 := 
sorry

end inscribed_squares_ratio_l1164_116492


namespace volume_of_rice_pile_l1164_116460

theorem volume_of_rice_pile
  (arc_length_bottom : ℝ)
  (height : ℝ)
  (one_fourth_cone : ℝ)
  (approx_pi : ℝ)
  (h_arc : arc_length_bottom = 8)
  (h_height : height = 5)
  (h_one_fourth_cone : one_fourth_cone = 1/4)
  (h_approx_pi : approx_pi = 3) :
  ∃ V : ℝ, V = one_fourth_cone * (1 / 3) * π * (16^2 / π^2) * height :=
by
  sorry

end volume_of_rice_pile_l1164_116460


namespace b_horses_pasture_l1164_116413

theorem b_horses_pasture (H : ℕ) : (9 * H / (96 + 9 * H + 108)) * 870 = 360 → H = 6 :=
by
  -- Here we state the problem and skip the proof
  sorry

end b_horses_pasture_l1164_116413


namespace new_average_l1164_116424

open Nat

-- The Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

-- Sum of the first 35 Fibonacci numbers
def sum_fibonacci_first_35 : ℕ :=
  (List.range 35).map fibonacci |>.sum -- or critical to use: List.foldr (λ x acc, fibonacci x + acc) 0 (List.range 35) 

theorem new_average (n : ℕ) (avg : ℕ) (Fib_Sum : ℕ) 
  (h₁ : n = 35) 
  (h₂ : avg = 25) 
  (h₃ : Fib_Sum = sum_fibonacci_first_35) : 
  (25 * Fib_Sum / 35) = avg * (sum_fibonacci_first_35) / n := 
by 
  sorry

end new_average_l1164_116424


namespace Jake_has_62_balls_l1164_116400

theorem Jake_has_62_balls 
  (C A J : ℕ)
  (h1 : C = 41 + 7)
  (h2 : A = 2 * C)
  (h3 : J = A - 34) : 
  J = 62 :=
by 
  sorry

end Jake_has_62_balls_l1164_116400


namespace money_distribution_l1164_116435

variable (A B C : ℝ)

theorem money_distribution
  (h₁ : A + B + C = 500)
  (h₂ : A + C = 200)
  (h₃ : C = 60) :
  B + C = 360 :=
by
  sorry

end money_distribution_l1164_116435


namespace measure_exactly_10_liters_l1164_116402

theorem measure_exactly_10_liters (A B : ℕ) (A_cap B_cap : ℕ) (hA : A_cap = 11) (hB : B_cap = 9) :
  ∃ (A B : ℕ), A + B = 10 ∧ A ≤ A_cap ∧ B ≤ B_cap := 
sorry

end measure_exactly_10_liters_l1164_116402


namespace find_M_l1164_116433

theorem find_M (p q r s M : ℚ)
  (h1 : p + q + r + s = 100)
  (h2 : p + 10 = M)
  (h3 : q - 5 = M)
  (h4 : 10 * r = M)
  (h5 : s / 2 = M) :
  M = 1050 / 41 :=
by
  sorry

end find_M_l1164_116433


namespace distance_to_grocery_store_l1164_116458

-- Definitions of given conditions
def miles_to_mall := 6
def miles_to_pet_store := 5
def miles_back_home := 9
def miles_per_gallon := 15
def cost_per_gallon := 3.5
def total_cost := 7

-- The Lean statement to prove the distance driven to the grocery store.
theorem distance_to_grocery_store (miles_to_mall miles_to_pet_store miles_back_home miles_per_gallon cost_per_gallon total_cost : ℝ) :
(total_cost / cost_per_gallon) * miles_per_gallon - (miles_to_mall + miles_to_pet_store + miles_back_home) = 10 := by
  sorry

end distance_to_grocery_store_l1164_116458


namespace minimum_value_quadratic_function_l1164_116469

-- Defining the quadratic function y
def quadratic_function (x : ℝ) : ℝ := 4 * x^2 + 8 * x + 16

-- Statement asserting the minimum value of the quadratic function
theorem minimum_value_quadratic_function : ∃ (y_min : ℝ), (∀ x : ℝ, quadratic_function x ≥ y_min) ∧ y_min = 12 :=
by
  -- Here we would normally insert the proof, but we skip it with sorry
  sorry

end minimum_value_quadratic_function_l1164_116469


namespace functional_eq_solution_l1164_116436

variable (f g : ℝ → ℝ)

theorem functional_eq_solution (h : ∀ x y : ℝ, f (x + y * g x) = g x + x * f y) : f = id := 
sorry

end functional_eq_solution_l1164_116436


namespace louisa_average_speed_l1164_116459

theorem louisa_average_speed :
  ∃ v : ℝ, 
  (100 / v = 175 / v - 3) ∧ 
  v = 25 :=
by
  sorry

end louisa_average_speed_l1164_116459


namespace wand_cost_l1164_116418

theorem wand_cost (c : ℕ) (h1 : 3 * c = 3 * c) (h2 : 2 * (c + 5) = 130) : c = 60 :=
by
  sorry

end wand_cost_l1164_116418


namespace range_of_k_l1164_116407

noncomputable def quadratic_has_real_roots (k : ℝ) :=
  ∃ (x : ℝ), (k - 3) * x^2 - 4 * x + 2 = 0

theorem range_of_k (k : ℝ) : quadratic_has_real_roots k ↔ k ≤ 5 := 
  sorry

end range_of_k_l1164_116407


namespace total_profit_correct_l1164_116447

-- We define the conditions
variables (a m : ℝ)

-- The item's cost per piece
def cost_per_piece : ℝ := a
-- The markup percentage
def markup_percentage : ℝ := 0.20
-- The discount percentage
def discount_percentage : ℝ := 0.10
-- The number of pieces sold
def pieces_sold : ℝ := m

-- Definitions derived from conditions
def selling_price_markup : ℝ := cost_per_piece a * (1 + markup_percentage)
def selling_price_discount : ℝ := selling_price_markup a * (1 - discount_percentage)
def profit_per_piece : ℝ := selling_price_discount a - cost_per_piece a
def total_profit : ℝ := profit_per_piece a * pieces_sold m

theorem total_profit_correct (a m : ℝ) : total_profit a m = 0.08 * a * m :=
by sorry

end total_profit_correct_l1164_116447


namespace mans_speed_against_current_l1164_116415

theorem mans_speed_against_current (V_with_current V_current V_against : ℝ) (h1 : V_with_current = 21) (h2 : V_current = 4.3) : 
  V_against = V_with_current - 2 * V_current := 
sorry

end mans_speed_against_current_l1164_116415


namespace lawn_length_is_70_l1164_116431

-- Definitions for conditions
def width_of_lawn : ℕ := 50
def road_width : ℕ := 10
def cost_of_roads : ℕ := 3600
def cost_per_sqm : ℕ := 3

-- Proof problem
theorem lawn_length_is_70 :
  ∃ L : ℕ, 10 * L + 10 * width_of_lawn = cost_of_roads / cost_per_sqm ∧ L = 70 := by
  sorry

end lawn_length_is_70_l1164_116431


namespace stock_initial_value_l1164_116462

theorem stock_initial_value (V : ℕ) (h : ∀ n ≤ 99, V + n = 200 - (99 - n)) : V = 101 :=
sorry

end stock_initial_value_l1164_116462


namespace ratio_a3_a6_l1164_116427

variable (a : ℕ → ℝ) (d : ℝ)
-- aₙ is an arithmetic sequence
variable (h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d)
-- d ≠ 0
variable (h_d_nonzero : d ≠ 0)
-- a₃² = a₁a₉
variable (h_condition : (a 2)^2 = (a 0) * (a 8))

theorem ratio_a3_a6 : (a 2) / (a 5) = 1 / 2 :=
by
  -- Proof omitted
  sorry

end ratio_a3_a6_l1164_116427


namespace integer_values_of_b_for_polynomial_root_l1164_116496

theorem integer_values_of_b_for_polynomial_root
    (b : ℤ) :
    (∃ x : ℤ, x^3 + 6 * x^2 + b * x + 12 = 0) ↔
    b = -217 ∨ b = -74 ∨ b = -43 ∨ b = -31 ∨ b = -22 ∨ b = -19 ∨
    b = 19 ∨ b = 22 ∨ b = 31 ∨ b = 43 ∨ b = 74 ∨ b = 217 :=
    sorry

end integer_values_of_b_for_polynomial_root_l1164_116496


namespace no_domovoi_exists_l1164_116417

variables {Domovoi Creature : Type}

def likes_pranks (c : Creature) : Prop := sorry
def likes_cleanliness_order (c : Creature) : Prop := sorry
def is_domovoi (c : Creature) : Prop := sorry

axiom all_domovoi_like_pranks : ∀ (c : Creature), is_domovoi c → likes_pranks c
axiom all_domovoi_like_cleanliness : ∀ (c : Creature), is_domovoi c → likes_cleanliness_order c
axiom cleanliness_implies_no_pranks : ∀ (c : Creature), likes_cleanliness_order c → ¬ likes_pranks c

theorem no_domovoi_exists : ¬ ∃ (c : Creature), is_domovoi c := 
sorry

end no_domovoi_exists_l1164_116417


namespace no_integer_solutions_l1164_116484

theorem no_integer_solutions (x y z : ℤ) (h1 : x > y) (h2 : y > z) : 
  x * (x - y) + y * (y - z) + z * (z - x) ≠ 3 := 
by
  sorry

end no_integer_solutions_l1164_116484


namespace jacket_final_price_l1164_116476

theorem jacket_final_price 
  (original_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) (final_discount : ℝ)
  (price_after_first : ℝ := original_price * (1 - first_discount))
  (price_after_second : ℝ := price_after_first * (1 - second_discount))
  (final_price : ℝ := price_after_second * (1 - final_discount)) :
  original_price = 250 ∧ first_discount = 0.4 ∧ second_discount = 0.3 ∧ final_discount = 0.1 →
  final_price = 94.5 := 
by 
  sorry

end jacket_final_price_l1164_116476


namespace sum_of_fractions_eq_sum_of_cubes_l1164_116438

theorem sum_of_fractions_eq_sum_of_cubes (x : ℝ) (h : x^2 - x + 1 ≠ 0) :
  ( (x-1)*(x+1) / (x*(x-1) + 1) + (2*(0.5-x)) / (x*(1-x) -1) ) = 
  ( ((x-1)*(x+1) / (x*(x-1) + 1))^3 + ((2*(0.5-x)) / (x*(1-x) -1))^3 ) :=
sorry

end sum_of_fractions_eq_sum_of_cubes_l1164_116438


namespace largest_y_coordinate_l1164_116454

theorem largest_y_coordinate (x y : ℝ) (h : (x^2 / 25) + ((y - 3)^2 / 25) = 0) : y = 3 := by
  sorry

end largest_y_coordinate_l1164_116454


namespace find_y_l1164_116486

theorem find_y (y : ℝ) (h : 3 * y / 4 = 15) : y = 20 :=
sorry

end find_y_l1164_116486


namespace inequality_proof_l1164_116446

-- Define the context of non-negative real numbers and sum to 1
variable {x y z : ℝ}
variable (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z)
variable (h_sum : x + y + z = 1)

-- State the theorem to be proved
theorem inequality_proof (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : x + y + z = 1) :
    0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
    sorry

end inequality_proof_l1164_116446


namespace range_of_a_l1164_116485

theorem range_of_a (a : ℝ) :
  (∀ p : ℝ × ℝ, (p.1 - 2 * a) ^ 2 + (p.2 - (a + 3)) ^ 2 = 4 → p.1 ^ 2 + p.2 ^ 2 = 1) →
  -1 < a ∧ a < 0 := 
sorry

end range_of_a_l1164_116485


namespace relationship_P_Q_l1164_116461

variable (a : ℝ)
variable (P : ℝ := Real.sqrt a + Real.sqrt (a + 5))
variable (Q : ℝ := Real.sqrt (a + 2) + Real.sqrt (a + 3))

theorem relationship_P_Q (h : 0 ≤ a) : P < Q :=
by
  sorry

end relationship_P_Q_l1164_116461


namespace no_solution_exists_l1164_116470

theorem no_solution_exists : ¬ ∃ n : ℕ, 0 < n ∧ (2^n % 60 = 29 ∨ 2^n % 60 = 31) := 
by
  sorry

end no_solution_exists_l1164_116470


namespace mary_added_peanuts_l1164_116449

theorem mary_added_peanuts (initial final added : Nat) 
  (h1 : initial = 4)
  (h2 : final = 16)
  (h3 : final = initial + added) : 
  added = 12 := 
by {
  sorry
}

end mary_added_peanuts_l1164_116449


namespace solve_system_l1164_116444

theorem solve_system (x y : ℝ) (h1 : x + 3 * y = 20) (h2 : x + y = 10) : x = 5 ∧ y = 5 := 
by 
  sorry

end solve_system_l1164_116444


namespace symmetric_coordinates_l1164_116426

structure Point :=
  (x : Int)
  (y : Int)

def symmetric_about_origin (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

theorem symmetric_coordinates (P : Point) (h : P = Point.mk (-1) 2) :
  symmetric_about_origin P = Point.mk 1 (-2) :=
by
  sorry

end symmetric_coordinates_l1164_116426


namespace solve_quadratic_l1164_116432

theorem solve_quadratic (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : ∀ x : ℝ, x^2 + c*x + d = 0 ↔ x = c ∨ x = d) : (c, d) = (1, -2) :=
sorry

end solve_quadratic_l1164_116432


namespace required_line_equation_l1164_116425

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ

-- Line structure with general form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- A point P on a line
def on_line (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

-- Perpendicular condition between two lines
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- The known line
def known_line : Line := {a := 1, b := -2, c := 3}

-- The given point
def P : Point := {x := -1, y := 3}

noncomputable def required_line : Line := {a := 2, b := 1, c := -1}

-- The theorem to be proved
theorem required_line_equation (l : Line) (P : Point) :
  (on_line P l) ∧ (perpendicular l known_line) ↔ l = required_line :=
  by
    sorry

end required_line_equation_l1164_116425


namespace rate_of_interest_l1164_116480

theorem rate_of_interest (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) (h : P > 0 ∧ T = 7 ∧ SI = P / 5 ∧ SI = (P * R * T) / 100) : 
  R = 20 / 7 := 
by
  sorry

end rate_of_interest_l1164_116480


namespace number_of_pencils_l1164_116408

theorem number_of_pencils (E P : ℕ) (h1 : E + P = 8) (h2 : 300 * E + 500 * P = 3000) (hE : E ≥ 1) (hP : P ≥ 1) : P = 3 :=
by
  sorry

end number_of_pencils_l1164_116408


namespace max_k_no_real_roots_l1164_116401

theorem max_k_no_real_roots : ∀ k : ℤ, (∀ x : ℝ, x^2 - 2 * x - (k : ℝ) ≠ 0) → k ≤ -2 :=
by
  sorry

end max_k_no_real_roots_l1164_116401


namespace polynomial_characterization_l1164_116452

noncomputable def homogeneous_polynomial (P : ℝ → ℝ → ℝ) (n : ℕ) :=
  ∀ t x y : ℝ, P (t * x) (t * y) = t^n * P x y

def polynomial_condition (P : ℝ → ℝ → ℝ) :=
  ∀ a b c : ℝ, P (a + b) c + P (b + c) a + P (c + a) b = 0

def P_value (P : ℝ → ℝ → ℝ) :=
  P 1 0 = 1

theorem polynomial_characterization (P : ℝ → ℝ → ℝ) (n : ℕ) :
  homogeneous_polynomial P n →
  polynomial_condition P →
  P_value P →
  ∃ A : ℝ → ℝ → ℝ, ∀ x y : ℝ, P x y = (x + y)^(n - 1) * (x - 2 * y) :=
by
  sorry

end polynomial_characterization_l1164_116452


namespace circle_area_is_162_pi_l1164_116498

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def circle_area (radius : ℝ) : ℝ :=
  Real.pi * radius ^ 2

def R : ℝ × ℝ := (5, -2)
def S : ℝ × ℝ := (-4, 7)

theorem circle_area_is_162_pi :
  circle_area (distance R S) = 162 * Real.pi :=
by
  sorry

end circle_area_is_162_pi_l1164_116498


namespace reflection_equation_l1164_116428

theorem reflection_equation
  (incident_line : ∀ x y : ℝ, 2 * x - y + 2 = 0)
  (reflection_axis : ∀ x y : ℝ, x + y - 5 = 0) :
  ∃ x y : ℝ, x - 2 * y + 7 = 0 :=
by
  sorry

end reflection_equation_l1164_116428


namespace billiard_ball_weight_l1164_116497

theorem billiard_ball_weight (w_box w_box_with_balls : ℝ) (h_w_box : w_box = 0.5) 
(h_w_box_with_balls : w_box_with_balls = 1.82) : 
    let total_weight_balls := w_box_with_balls - w_box;
    let weight_one_ball := total_weight_balls / 6;
    weight_one_ball = 0.22 :=
by
  sorry

end billiard_ball_weight_l1164_116497


namespace right_triangle_min_hypotenuse_right_triangle_min_hypotenuse_achieved_l1164_116490

theorem right_triangle_min_hypotenuse (a b c : ℝ) (h_right : (a^2 + b^2 = c^2)) (h_perimeter : (a + b + c = 8)) : c ≥ 4 * Real.sqrt 2 := by
  sorry

theorem right_triangle_min_hypotenuse_achieved (a b c : ℝ) (h_right : (a^2 + b^2 = c^2)) (h_perimeter : (a + b + c = 8)) (h_isosceles : a = b) : c = 4 * Real.sqrt 2 := by
  sorry

end right_triangle_min_hypotenuse_right_triangle_min_hypotenuse_achieved_l1164_116490


namespace seven_power_expression_l1164_116441

theorem seven_power_expression (x y z : ℝ) (h₀ : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) (h₁ : x + y + z = 0) (h₂ : xy + xz + yz ≠ 0) :
  (x^7 + y^7 + z^7) / (xyz * (x^2 + y^2 + z^2)) = 14 :=
by
  sorry

end seven_power_expression_l1164_116441


namespace time_upstream_is_correct_l1164_116434

-- Define the conditions
def speed_of_stream : ℝ := 3
def speed_in_still_water : ℝ := 15
def downstream_time : ℝ := 1
def downstream_speed : ℝ := speed_in_still_water + speed_of_stream
def distance_downstream : ℝ := downstream_speed * downstream_time
def upstream_speed : ℝ := speed_in_still_water - speed_of_stream

-- Theorem statement
theorem time_upstream_is_correct :
  (distance_downstream / upstream_speed) = 1.5 := by
  sorry

end time_upstream_is_correct_l1164_116434


namespace percent_difference_l1164_116474

theorem percent_difference:
  let percent_value1 := (55 / 100) * 40
  let fraction_value2 := (4 / 5) * 25
  percent_value1 - fraction_value2 = 2 :=
by
  sorry

end percent_difference_l1164_116474


namespace outlined_square_digit_l1164_116466

theorem outlined_square_digit :
  ∀ (digit : ℕ), (digit ∈ {n | ∃ (m : ℕ), 10 ≤ 3^m ∧ 3^m < 1000 ∧ digit = (3^m / 10) % 10 }) →
  (digit ∈ {n | ∃ (n : ℕ), 10 ≤ 7^n ∧ 7^n < 1000 ∧ digit = (7^n / 10) % 10 }) →
  digit = 4 :=
by sorry

end outlined_square_digit_l1164_116466


namespace larger_number_of_product_and_sum_l1164_116420

theorem larger_number_of_product_and_sum (x y : ℕ) (h_prod : x * y = 35) (h_sum : x + y = 12) : max x y = 7 :=
by {
  sorry
}

end larger_number_of_product_and_sum_l1164_116420


namespace volume_of_sphere_in_cone_l1164_116494

/-- The volume of a sphere inscribed in a right circular cone with
a base diameter of 16 inches and a cross-section with a vertex angle of 45 degrees
is 4096 * sqrt 2 * π / 3 cubic inches. -/
theorem volume_of_sphere_in_cone :
  let d := 16 -- the diameter of the base of the cone in inches
  let angle := 45 -- the vertex angle of the cross-section triangle in degrees
  let r := 8 * Real.sqrt 2 -- the radius of the sphere in inches
  let V := 4 / 3 * Real.pi * r^3 -- the volume of the sphere in cubic inches
  V = 4096 * Real.sqrt 2 * Real.pi / 3 :=
by
  simp only [Real.sqrt]
  sorry -- proof goes here

end volume_of_sphere_in_cone_l1164_116494


namespace cylinder_area_ratio_l1164_116412

theorem cylinder_area_ratio (r h : ℝ) (h_eq : h = 2 * r * Real.sqrt π) :
  let S_lateral := 2 * π * r * h
  let S_total := S_lateral + 2 * π * r^2
  S_total / S_lateral = 1 + (1 / (2 * Real.sqrt π)) := by
sorry

end cylinder_area_ratio_l1164_116412


namespace product_of_possible_values_l1164_116487

theorem product_of_possible_values : 
  (∀ x : ℝ, |x - 5| - 4 = -1 → (x = 2 ∨ x = 8)) → (2 * 8) = 16 :=
by 
  sorry

end product_of_possible_values_l1164_116487


namespace consecutive_odd_sum_l1164_116472

theorem consecutive_odd_sum (n : ℤ) (h : n + 2 = 9) : 
  let a := n
  let b := n + 2
  let c := n + 4
  (a + b + c) = a + 20 := by
  sorry

end consecutive_odd_sum_l1164_116472


namespace matrix_operation_correct_l1164_116411

open Matrix

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ := ![![7, -3], ![2, 5]]
def matrix2 : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 4], ![0, -3]]
def matrix3 : Matrix (Fin 2) (Fin 2) ℤ := ![![6, 0], ![-1, 8]]
def result : Matrix (Fin 2) (Fin 2) ℤ := ![![12, -7], ![1, 16]]

theorem matrix_operation_correct:
  matrix1 - matrix2 + matrix3 = result :=
by
  sorry

end matrix_operation_correct_l1164_116411


namespace number_of_integer_values_of_x_l1164_116409

theorem number_of_integer_values_of_x (x : ℕ) (hx1 : x + 15 > 40) (hx2 : x + 40 > 15) (hx3 : 15 + 40 > x) :
  ∃ n : ℕ, n = 29 ∧ ∀ y : ℕ, (26 ≤ y ∧ y ≤ 54) ↔ true :=
by
  sorry

end number_of_integer_values_of_x_l1164_116409


namespace father_age_38_l1164_116473

variable (F S : ℕ)
variable (h1 : S = 14)
variable (h2 : F - 10 = 7 * (S - 10))

theorem father_age_38 : F = 38 :=
by
  sorry

end father_age_38_l1164_116473


namespace chord_intersection_l1164_116439

theorem chord_intersection {AP BP CP DP : ℝ} (hAP : AP = 2) (hBP : BP = 6) (hCP_DP : ∃ k : ℝ, CP = k ∧ DP = 3 * k) :
  DP = 6 :=
by sorry

end chord_intersection_l1164_116439


namespace remainder_of_2n_div_10_l1164_116477

theorem remainder_of_2n_div_10 (n : ℤ) (h : n % 20 = 11) : (2 * n) % 10 = 2 := by
  sorry

end remainder_of_2n_div_10_l1164_116477


namespace number_of_boxes_l1164_116443

theorem number_of_boxes (total_eggs : ℕ) (eggs_per_box : ℕ) (boxes : ℕ) : 
  total_eggs = 21 → eggs_per_box = 7 → boxes = total_eggs / eggs_per_box → boxes = 3 :=
by
  intros h_total_eggs h_eggs_per_box h_boxes
  rw [h_total_eggs, h_eggs_per_box] at h_boxes
  exact h_boxes

end number_of_boxes_l1164_116443


namespace union_of_A_and_B_l1164_116404

open Set -- to use set notation and operations

def A : Set ℝ := { x | -1/2 < x ∧ x < 2 }

def B : Set ℝ := { x | x^2 ≤ 1 }

theorem union_of_A_and_B :
  A ∪ B = Ico (-1:ℝ) 2 := 
by
  -- proof steps would go here, but we skip these with sorry.
  sorry

end union_of_A_and_B_l1164_116404


namespace personal_income_tax_correct_l1164_116422

-- Defining the conditions
def monthly_income : ℕ := 30000
def vacation_bonus : ℕ := 20000
def car_sale_income : ℕ := 250000
def land_purchase_cost : ℕ := 300000

def standard_deduction_car_sale : ℕ := 250000
def property_deduction_land_purchase : ℕ := 300000

-- Define total income
def total_income : ℕ := (monthly_income * 12) + vacation_bonus + car_sale_income

-- Define total deductions
def total_deductions : ℕ := standard_deduction_car_sale + property_deduction_land_purchase

-- Define taxable income (total income - total deductions)
def taxable_income : ℕ := total_income - total_deductions

-- Define tax rate
def tax_rate : ℚ := 0.13

-- Define the correct answer for the tax payable
def tax_payable : ℚ := taxable_income * tax_rate

-- Prove the tax payable is 10400 rubles
theorem personal_income_tax_correct : tax_payable = 10400 := by
  sorry

end personal_income_tax_correct_l1164_116422


namespace tank_emptying_time_l1164_116482

theorem tank_emptying_time (fill_without_leak fill_with_leak : ℝ) (h1 : fill_without_leak = 7) (h2 : fill_with_leak = 8) : 
  let R := 1 / fill_without_leak
  let L := R - 1 / fill_with_leak
  let emptying_time := 1 / L
  emptying_time = 56 :=
by
  sorry

end tank_emptying_time_l1164_116482


namespace first_dog_walks_two_miles_per_day_l1164_116499

variable (x : ℝ)

theorem first_dog_walks_two_miles_per_day  
  (h1 : 7 * x + 56 = 70) : 
  x = 2 := 
by 
  sorry

end first_dog_walks_two_miles_per_day_l1164_116499


namespace minimum_time_for_tomato_egg_soup_l1164_116414

noncomputable def cracking_egg_time : ℕ := 1
noncomputable def washing_chopping_tomatoes_time : ℕ := 2
noncomputable def boiling_tomatoes_time : ℕ := 3
noncomputable def adding_eggs_heating_time : ℕ := 1
noncomputable def stirring_egg_time : ℕ := 1

theorem minimum_time_for_tomato_egg_soup :
  washing_chopping_tomatoes_time + boiling_tomatoes_time + adding_eggs_heating_time = 6 :=
by
  -- proof to be filled
  sorry

end minimum_time_for_tomato_egg_soup_l1164_116414


namespace analysis_method_inequality_l1164_116451

def analysis_method_seeks (inequality : Prop) : Prop :=
  ∃ (sufficient_condition : Prop), (inequality → sufficient_condition)

theorem analysis_method_inequality (inequality : Prop) :
  (∃ sufficient_condition, (inequality → sufficient_condition)) :=
sorry

end analysis_method_inequality_l1164_116451


namespace gcd_of_7854_and_15246_is_6_six_is_not_prime_l1164_116419

theorem gcd_of_7854_and_15246_is_6 : gcd 7854 15246 = 6 := sorry

theorem six_is_not_prime : ¬ Prime 6 := sorry

end gcd_of_7854_and_15246_is_6_six_is_not_prime_l1164_116419


namespace find_x_l1164_116481

theorem find_x (x : ℝ) (h : 0.75 * x + 2 = 8) : x = 8 :=
sorry

end find_x_l1164_116481


namespace gracie_height_is_56_l1164_116450

noncomputable def Gracie_height : Nat := 56

theorem gracie_height_is_56 : Gracie_height = 56 := by
  sorry

end gracie_height_is_56_l1164_116450


namespace solve_for_x_l1164_116405

variables {x y : ℝ}

theorem solve_for_x (h : x / (x - 3) = (y^2 + 3 * y + 1) / (y^2 + 3 * y - 4)) : 
  x = (3 * y^2 + 9 * y + 3) / 5 :=
sorry

end solve_for_x_l1164_116405


namespace probability_not_miss_is_correct_l1164_116464

-- Define the probability that Peter will miss his morning train
def p_miss : ℚ := 5 / 12

-- Define the probability that Peter does not miss his morning train
def p_not_miss : ℚ := 1 - p_miss

-- The theorem to prove
theorem probability_not_miss_is_correct : p_not_miss = 7 / 12 :=
by
  -- Proof omitted
  sorry

end probability_not_miss_is_correct_l1164_116464


namespace positive_integers_m_divisors_l1164_116423

theorem positive_integers_m_divisors :
  ∃ n, n = 3 ∧ ∀ m : ℕ, (0 < m ∧ ∃ k, 2310 = k * (m^2 + 2)) ↔ m = 1 ∨ m = 2 ∨ m = 3 :=
by
  sorry

end positive_integers_m_divisors_l1164_116423


namespace find_a_f_odd_f_increasing_l1164_116475

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x - a / x

theorem find_a : (f 1 a = 3) → (a = -1) :=
by
  sorry

noncomputable def f_1 (x : ℝ) : ℝ := 2 * x + 1 / x

theorem f_odd : ∀ x : ℝ, f_1 (-x) = -f_1 x :=
by
  sorry

theorem f_increasing : ∀ x1 x2 : ℝ, (x1 > 1) → (x2 > 1) → (x1 > x2) → (f_1 x1 > f_1 x2) :=
by
  sorry

end find_a_f_odd_f_increasing_l1164_116475


namespace smallest_n_l1164_116445

theorem smallest_n (n : ℕ) (h1 : n > 1) (h2 : 2016 ∣ (3 * n^3 + 2013)) : n = 193 := 
sorry

end smallest_n_l1164_116445


namespace disprove_prime_statement_l1164_116491

theorem disprove_prime_statement : ∃ n : ℕ, ((¬ Nat.Prime n) ∧ Nat.Prime (n + 2)) ∨ (Nat.Prime n ∧ ¬ Nat.Prime (n + 2)) :=
sorry

end disprove_prime_statement_l1164_116491


namespace solve_inequality_and_find_positive_int_solutions_l1164_116406

theorem solve_inequality_and_find_positive_int_solutions :
  ∀ (x : ℝ), (2 * x + 1) / 3 - 1 ≤ (2 / 5) * x → x ≤ 2.5 ∧ ∃ (n : ℕ), n = 1 ∨ n = 2 :=
by
  intro x
  intro h
  sorry

end solve_inequality_and_find_positive_int_solutions_l1164_116406


namespace students_between_min_and_hos_l1164_116463

theorem students_between_min_and_hos
  (total_students : ℕ)
  (minyoung_left_position : ℕ)
  (hoseok_right_position : ℕ)
  (total_students_eq : total_students = 13)
  (minyoung_left_position_eq : minyoung_left_position = 8)
  (hoseok_right_position_eq : hoseok_right_position = 9) :
  (minyoung_left_position - (total_students - hoseok_right_position + 1) - 1) = 2 := 
by
  sorry

end students_between_min_and_hos_l1164_116463


namespace votes_difference_l1164_116453

theorem votes_difference (T : ℕ) (V_a : ℕ) (V_f : ℕ) 
  (h1 : T = 330) (h2 : V_a = 40 * T / 100) (h3 : V_f = T - V_a) : V_f - V_a = 66 :=
by
  sorry

end votes_difference_l1164_116453


namespace jogged_time_l1164_116493

theorem jogged_time (J : ℕ) (W : ℕ) (r : ℚ) (h1 : r = 5 / 3) (h2 : W = 9) (h3 : r = J / W) : J = 15 := 
by
  sorry

end jogged_time_l1164_116493


namespace value_of_mn_squared_l1164_116478

theorem value_of_mn_squared (m n : ℤ) (h1 : |m| = 4) (h2 : |n| = 3) (h3 : m - n < 0) : (m + n)^2 = 1 ∨ (m + n)^2 = 49 :=
by sorry

end value_of_mn_squared_l1164_116478


namespace max_parrots_l1164_116483

theorem max_parrots (x y z : ℕ) (h1 : y + z ≤ 9) (h2 : x + z ≤ 11) : x + y + z ≤ 19 :=
sorry

end max_parrots_l1164_116483


namespace remaining_money_l1164_116495

def initial_amount : Float := 499.9999999999999

def spent_on_clothes (initial : Float) : Float :=
  (1/3) * initial

def remaining_after_clothes (initial : Float) : Float :=
  initial - spent_on_clothes initial

def spent_on_food (remaining_clothes : Float) : Float :=
  (1/5) * remaining_clothes

def remaining_after_food (remaining_clothes : Float) : Float :=
  remaining_clothes - spent_on_food remaining_clothes

def spent_on_travel (remaining_food : Float) : Float :=
  (1/4) * remaining_food

def remaining_after_travel (remaining_food : Float) : Float :=
  remaining_food - spent_on_travel remaining_food

theorem remaining_money :
  remaining_after_travel (remaining_after_food (remaining_after_clothes initial_amount)) = 199.99 :=
by
  sorry

end remaining_money_l1164_116495


namespace tangent_line_x_squared_at_one_one_l1164_116479

open Real

theorem tangent_line_x_squared_at_one_one :
  ∀ (x y : ℝ), y = x^2 → (x, y) = (1, 1) → (2 * x - y - 1 = 0) :=
by
  intros x y h_curve h_point
  sorry

end tangent_line_x_squared_at_one_one_l1164_116479


namespace total_seeds_in_garden_l1164_116429

-- Definitions based on conditions
def large_bed_rows : Nat := 4
def large_bed_seeds_per_row : Nat := 25
def medium_bed_rows : Nat := 3
def medium_bed_seeds_per_row : Nat := 20
def num_large_beds : Nat := 2
def num_medium_beds : Nat := 2

-- Theorem statement to show total seeds
theorem total_seeds_in_garden : 
  num_large_beds * (large_bed_rows * large_bed_seeds_per_row) + 
  num_medium_beds * (medium_bed_rows * medium_bed_seeds_per_row) = 320 := 
by
  sorry

end total_seeds_in_garden_l1164_116429


namespace car_gasoline_tank_capacity_l1164_116430

theorem car_gasoline_tank_capacity
    (speed : ℝ)
    (usage_rate : ℝ)
    (travel_time : ℝ)
    (fraction_used : ℝ)
    (tank_capacity : ℝ)
    (gallons_used : ℝ)
    (distance_traveled : ℝ) :
  speed = 50 →
  usage_rate = 1 / 30 →
  travel_time = 5 →
  fraction_used = 0.5555555555555556 →
  distance_traveled = speed * travel_time →
  gallons_used = distance_traveled * usage_rate →
  gallon_used = tank_capacity * fraction_used →
  tank_capacity = 15 :=
by
  intros hs hr ht hf hd hu hf
  sorry

end car_gasoline_tank_capacity_l1164_116430


namespace value_of_b_l1164_116442

theorem value_of_b (a b : ℕ) (h1 : a * b = 2 * (a + b) + 10) (h2 : b - a = 5) : b = 9 := 
by {
  -- Proof is not required, so we use sorry to complete the statement
  sorry
}

end value_of_b_l1164_116442


namespace garden_length_l1164_116403

theorem garden_length (columns : ℕ) (distance_between_trees : ℕ) (boundary_distance : ℕ) (h_columns : columns = 12) (h_distance_between_trees : distance_between_trees = 2) (h_boundary_distance : boundary_distance = 5) : 
  ((columns - 1) * distance_between_trees + 2 * boundary_distance) = 32 :=
by 
  sorry

end garden_length_l1164_116403


namespace earning_hours_per_week_l1164_116421

theorem earning_hours_per_week (totalEarnings : ℝ) (originalWeeks : ℝ) (missedWeeks : ℝ) 
  (originalHoursPerWeek : ℝ) : 
  missedWeeks = 3 → originalWeeks = 15 → originalHoursPerWeek = 25 → totalEarnings = 3750 → 
  (totalEarnings / ((totalEarnings / (originalWeeks * originalHoursPerWeek)) * (originalWeeks - missedWeeks))) = 31.25 :=
by
  intros
  sorry

end earning_hours_per_week_l1164_116421


namespace min_max_of_f_l1164_116467

def f (x : ℝ) : ℝ := -2 * x + 1

-- defining the minimum and maximum values
def min_val : ℝ := -3
def max_val : ℝ := 5

theorem min_max_of_f :
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f x ≥ min_val) ∧ 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f x ≤ max_val) :=
by 
  sorry

end min_max_of_f_l1164_116467


namespace total_wage_l1164_116489

theorem total_wage (work_days_A work_days_B : ℕ) (wage_A : ℕ) (total_wage : ℕ) 
  (h1 : work_days_A = 10) 
  (h2 : work_days_B = 15) 
  (h3 : wage_A = 1980)
  (h4 : (wage_A / (wage_A / (total_wage * 3 / 5))) = 3)
  : total_wage = 3300 :=
sorry

end total_wage_l1164_116489


namespace red_ball_second_given_red_ball_first_l1164_116448

noncomputable def probability_of_red_second_given_first : ℚ :=
  let totalBalls := 6
  let redBallsOnFirst := 4
  let whiteBalls := 2
  let redBallsOnSecond := 3
  let remainingBalls := 5

  let P_A := redBallsOnFirst / totalBalls
  let P_AB := (redBallsOnFirst / totalBalls) * (redBallsOnSecond / remainingBalls)
  P_AB / P_A

theorem red_ball_second_given_red_ball_first :
  probability_of_red_second_given_first = 3 / 5 :=
sorry

end red_ball_second_given_red_ball_first_l1164_116448
