import Mathlib

namespace three_legged_reptiles_count_l225_225468

noncomputable def total_heads : ℕ := 300
noncomputable def total_legs : ℕ := 798

def number_of_three_legged_reptiles (b r m : ℕ) : Prop :=
  b + r + m = total_heads ∧
  2 * b + 3 * r + 4 * m = total_legs

theorem three_legged_reptiles_count (b r m : ℕ) (h : number_of_three_legged_reptiles b r m) :
  r = 102 :=
sorry

end three_legged_reptiles_count_l225_225468


namespace directrix_of_parabola_l225_225969

theorem directrix_of_parabola (p : ℝ) (y x : ℝ) :
  y = x^2 → x^2 = 4 * p * y → 4 * y + 1 = 0 :=
by
  intros hyp1 hyp2
  sorry

end directrix_of_parabola_l225_225969


namespace tan_product_l225_225240

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end tan_product_l225_225240


namespace collinear_probability_l225_225749

-- Define the rectangular array
def rows : ℕ := 4
def cols : ℕ := 5
def total_dots : ℕ := rows * cols
def chosen_dots : ℕ := 4

-- Define the collinear sets
def horizontal_lines : ℕ := rows
def vertical_lines : ℕ := cols
def collinear_sets : ℕ := horizontal_lines + vertical_lines

-- Define the total combinations of choosing 4 dots out of 20
def total_combinations : ℕ := Nat.choose total_dots chosen_dots

-- Define the probability
def probability : ℚ := collinear_sets / total_combinations

theorem collinear_probability : probability = 9 / 4845 := by
  sorry

end collinear_probability_l225_225749


namespace product_of_remaining_numbers_l225_225640

theorem product_of_remaining_numbers {a b c d : ℕ} (h1 : a = 11) (h2 : b = 22) (h3 : c = 33) (h4 : d = 44) :
  ∃ (x y z : ℕ), 
  (∃ n: ℕ, (a + b + c + d) - n * 3 = 3 ∧ -- We removed n groups of 3 different numbers
             x + y + z = 2 * n + (a + b + c + d)) ∧ -- We added 2 * n numbers back
  x * y * z = 12 := 
sorry

end product_of_remaining_numbers_l225_225640


namespace perfect_square_trinomial_l225_225565

theorem perfect_square_trinomial (m : ℝ) : (∃ (a b : ℝ), (a * x + b) ^ 2 = x^2 + m * x + 16) -> (m = 8 ∨ m = -8) :=
sorry

end perfect_square_trinomial_l225_225565


namespace distance_from_T_to_ABC_l225_225912

noncomputable def distance_to_plane (A B C T : EuclideanSpace ℝ (Fin 3))
  (h1 : T.dist A = 15) (h2 : T.dist B = 15) (h3 : T.dist C = 9)
  (h4 : A ≠ B) (h5 : B ≠ C) (h6 : A ≠ C)
  (h7 : InnerProductSpace.isOrthogonal ℝ (A - T) (B - T))
  (h8 : InnerProductSpace.isOrthogonal ℝ (A - T) (C - T))
  (h9 : InnerProductSpace.isOrthogonal ℝ (B - T) (C - T)) : ℝ :=
  
  let AB : EuclideanSpace ℝ (Fin 3) := B - A in
  let AC : EuclideanSpace ℝ (Fin 3) := C - A in
  let TA : ℝ := T.dist A in
  let TB : ℝ := T.dist B in
  let TC : ℝ := T.dist C in
  let AB_length : ℝ := AB.norm in
  let AC_length : ℝ := AC.norm in
  let area_ABC : ℝ := 0.5 * AB_length * sqrt (AC_length^2 - (AB_length/2)^2) in
  (3 * (TA * TB / 2 * TC) / area_ABC)

theorem distance_from_T_to_ABC
  (A B C T : EuclideanSpace ℝ (Fin 3))
  (h1 : T.dist A = 15) (h2 : T.dist B = 15) (h3 : T.dist C = 9)
  (h4 : A ≠ B) (h5 : B ≠ C) (h6 : A ≠ C)
  (h7 : InnerProductSpace.isOrthogonal ℝ (A - T) (B - T))
  (h8 : InnerProductSpace.isOrthogonal ℝ (A - T) (C - T))
  (h9 : InnerProductSpace.isOrthogonal ℝ (B - T) (C - T)) :
  distance_to_plane A B C T h1 h2 h3 h4 h5 h6 h7 h8 h9 = 6 * sqrt 6 :=
by sorry

end distance_from_T_to_ABC_l225_225912


namespace find_annual_compound_interest_rate_l225_225645

noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) (r : ℝ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

theorem find_annual_compound_interest_rate :
  compound_interest_rate 10000 24882.50 1 7 0.125 :=
by sorry

end find_annual_compound_interest_rate_l225_225645


namespace male_salmon_count_l225_225530

theorem male_salmon_count (total_count : ℕ) (female_count : ℕ) (male_count : ℕ) :
  total_count = 971639 →
  female_count = 259378 →
  male_count = (total_count - female_count) →
  male_count = 712261 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end male_salmon_count_l225_225530


namespace evaluate_expression_l225_225532

theorem evaluate_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2 * x + 2) / x) * ((y^2 + 2 * y + 2) / y) + ((x^2 - 3 * x + 2) / y) * ((y^2 - 3 * y + 2) / x) 
  = 2 * x * y - (x / y) - (y / x) + 13 + 10 / x + 4 / y + 8 / (x * y) :=
by
  sorry

end evaluate_expression_l225_225532


namespace compute_expression_l225_225229

theorem compute_expression : 42 * 52 + 48 * 42 = 4200 :=
by sorry

end compute_expression_l225_225229


namespace number_of_rational_coefficient_terms_l225_225646

open Nat

def is_rational_coefficient (k : ℕ) : Prop :=
  k % 4 = 0 ∧ (988 - k) % 2 = 0

def rational_coefficient_terms_count : ℕ :=
  (Finset.range 989).filter is_rational_coefficient |>.card

theorem number_of_rational_coefficient_terms :
  rational_coefficient_terms_count = 248 :=
by
  sorry

end number_of_rational_coefficient_terms_l225_225646


namespace last_two_digits_of_squared_expression_l225_225926

theorem last_two_digits_of_squared_expression (n : ℕ) :
  (n * 2 * 3 * 4 * 46 * 47 * 48 * 49) ^ 2 % 100 = 76 :=
by
  sorry

end last_two_digits_of_squared_expression_l225_225926


namespace midpoint_on_nine_point_circle_l225_225491

def triangle (A B C : Point) : Prop :=
  ∃ (a b c : Real), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

def diameter_of_circle (B C : Point) (k : Circle) : Prop :=
  diameter B C k

def intersect_line_at (circle : Circle) (line : Line) (point : Point) : Prop :=
  intersects circle line point

def circumcircle_of_triangle (A E F : Point) (k' : Circle) : Prop :=
  circumcircle A E F k'

theorem midpoint_on_nine_point_circle
  {A B C E F P Q M : Point}
  {k k' : Circle} :
  triangle A B C →
  diameter_of_circle B C k →
  intersect_line_at k (Line.of_points C A) E →
  intersect_line_at k (Line.of_points B A) F →
  circumcircle_of_triangle A E F k' →
  Line.exists_midpoint P Q M →
  M ∈ nine_point_circle (triangle A B C) :=
begin
  intros h_triangle h_diameter h_intersect_CE h_intersect_BA h_circumcircle heqx_M,
  sorry -- Proof goes here.
end

end midpoint_on_nine_point_circle_l225_225491


namespace max_min_of_f_on_interval_l225_225171

-- Conditions
def f (x : ℝ) : ℝ := x^3 - 3 * x + 1
def interval : Set ℝ := Set.Icc (-3) 0

-- Problem statement
theorem max_min_of_f_on_interval : 
  ∃ (max min : ℝ), max = 1 ∧ min = -17 ∧ 
  (∀ x ∈ interval, f x ≤ max) ∧ 
  (∀ x ∈ interval, f x ≥ min) := 
sorry

end max_min_of_f_on_interval_l225_225171


namespace decreasing_interval_l225_225282

noncomputable def f (x : ℝ) := Real.exp (abs (x - 1))

theorem decreasing_interval : ∀ x y : ℝ, x ≤ y → y ≤ 1 → f y ≤ f x :=
by
  sorry

end decreasing_interval_l225_225282


namespace find_solution_to_inequality_l225_225428

open Set

noncomputable def inequality_solution : Set ℝ := {x : ℝ | 0.5 ≤ x ∧ x < 2 ∨ 3 ≤ x}

theorem find_solution_to_inequality :
  {x : ℝ | (x^2 + 1) / (x - 2) + (2 * x + 3) / (2 * x - 1) ≥ 4} = inequality_solution := 
sorry

end find_solution_to_inequality_l225_225428


namespace initial_investment_proof_l225_225425

noncomputable def initial_investment (A : ℝ) (r t : ℕ) : ℝ := 
  A / (1 + r / 100) ^ t

theorem initial_investment_proof : 
  initial_investment 1000 8 8 = 630.17 := sorry

end initial_investment_proof_l225_225425


namespace negation_of_p_l225_225489

variable (x : ℝ)

def p : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem negation_of_p : (¬p) ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 :=
by
  sorry

end negation_of_p_l225_225489


namespace smallest_number_from_digits_l225_225667

theorem smallest_number_from_digits : 
  ∀ (d1 d2 d3 d4 : ℕ), (d1 = 2) → (d2 = 0) → (d3 = 1) → (d4 = 6) →
  ∃ n : ℕ, (n = 1026) ∧ 
  ((n = d1 * 1000 + d2 * 100 + d3 * 10 + d4) ∨ 
   (n = d1 * 1000 + d2 * 100 + d4 * 10 + d3) ∨ 
   (n = d1 * 1000 + d3 * 100 + d2 * 10 + d4) ∨ 
   (n = d1 * 1000 + d3 * 100 + d4 * 10 + d2) ∨ 
   (n = d1 * 1000 + d4 * 100 + d2 * 10 + d3) ∨ 
   (n = d1 * 1000 + d4 * 100 + d3 * 10 + d2) ∨ 
   (n = d2 * 1000 + d1 * 100 + d3 * 10 + d4) ∨ 
   (n = d2 * 1000 + d1 * 100 + d4 * 10 + d3) ∨ 
   (n = d2 * 1000 + d3 * 100 + d1 * 10 + d4) ∨ 
   (n = d2 * 1000 + d3 * 100 + d4 * 10 + d1) ∨ 
   (n = d2 * 1000 + d4 * 100 + d1 * 10 + d3) ∨ 
   (n = d2 * 1000 + d4 * 100 + d3 * 10 + d1) ∨ 
   (n = d3 * 1000 + d1 * 100 + d2 * 10 + d4) ∨ 
   (n = d3 * 1000 + d1 * 100 + d4 * 10 + d2) ∨ 
   (n = d3 * 1000 + d2 * 100 + d1 * 10 + d4) ∨ 
   (n = d3 * 1000 + d2 * 100 + d4 * 10 + d1) ∨ 
   (n = d3 * 1000 + d4 * 100 + d1 * 10 + d2) ∨ 
   (n = d3 * 1000 + d4 * 100 + d2 * 10 + d1) ∨ 
   (n = d4 * 1000 + d1 * 100 + d2 * 10 + d3) ∨ 
   (n = d4 * 1000 + d1 * 100 + d3 * 10 + d2) ∨ 
   (n = d4 * 1000 + d2 * 100 + d1 * 10 + d3) ∨ 
   (n = d4 * 1000 + d2 * 100 + d3 * 10 + d1) ∨ 
   (n = d4 * 1000 + d3 * 100 + d1 * 10 + d2) ∨ 
   (n = d4 * 1000 + d3 * 100 + d2 * 10 + d1)) := sorry

end smallest_number_from_digits_l225_225667


namespace total_order_cost_l225_225402

theorem total_order_cost (n : ℕ) (cost_geo cost_eng : ℝ)
  (h1 : n = 35)
  (h2 : cost_geo = 10.50)
  (h3 : cost_eng = 7.50) :
  n * cost_geo + n * cost_eng = 630 := by
  -- proof steps should go here
  sorry

end total_order_cost_l225_225402


namespace demand_decrease_l225_225659

theorem demand_decrease (original_price_increase effective_price_increase demand_decrease : ℝ)
  (h1 : original_price_increase = 0.2)
  (h2 : effective_price_increase = original_price_increase / 2)
  (h3 : new_price = original_price * (1 + effective_price_increase))
  (h4 : 1 / new_price = original_demand)
  : demand_decrease = 0.0909 := sorry

end demand_decrease_l225_225659


namespace binomial_10_2_l225_225680

noncomputable def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binomial_10_2 : binom 10 2 = 45 := by
  sorry

end binomial_10_2_l225_225680


namespace circle_equation_center_line_l225_225078

theorem circle_equation_center_line (x y : ℝ) :
  -- Conditions
  (∀ (x1 y1 : ℝ), x1 + y1 - 2 = 0 → (x = 1 ∧ y = 1)) ∧
  ((x - 1)^2 + (y - 1)^2 = 4) ∧
  -- Points A and B
  (∀ (xA yA : ℝ), xA = 1 ∧ yA = -1 ∨ xA = -1 ∧ yA = 1 →
    ((xA - x)^2 + (yA - y)^2 = 4)) :=
by
  sorry

end circle_equation_center_line_l225_225078


namespace sum_of_digits_0_to_2012_l225_225330

-- Define the sum of digits function
def sum_of_digits (n : Nat) : Nat := 
  n.digits 10 |>.sum

-- Define the problem to calculate the sum of all digits from 0 to 2012
def sum_digits_up_to (n : Nat) : Nat := 
  (List.range (n + 1)).map sum_of_digits |>.sum

-- Lean theorem statement to prove the sum of digits from 0 to 2012 is 28077
theorem sum_of_digits_0_to_2012 : sum_digits_up_to 2012 = 28077 := 
  sorry

end sum_of_digits_0_to_2012_l225_225330


namespace ellipse_ratio_squared_l225_225687

theorem ellipse_ratio_squared (a b c : ℝ) 
    (h1 : b / a = a / c) 
    (h2 : c^2 = a^2 - b^2) : (b / a)^2 = 1 / 2 :=
by
  sorry

end ellipse_ratio_squared_l225_225687


namespace probability_at_least_one_boy_one_girl_l225_225175

def boys := 12
def girls := 18
def total_members := 30
def committee_size := 6

def total_ways := Nat.choose total_members committee_size
def all_boys_ways := Nat.choose boys committee_size
def all_girls_ways := Nat.choose girls committee_size
def all_boys_or_girls_ways := all_boys_ways + all_girls_ways
def complementary_probability := all_boys_or_girls_ways / total_ways
def desired_probability := 1 - complementary_probability

theorem probability_at_least_one_boy_one_girl :
  desired_probability = (574287 : ℚ) / 593775 :=
  sorry

end probability_at_least_one_boy_one_girl_l225_225175


namespace Barry_reach_l225_225335

noncomputable def Larry_full_height : ℝ := 5
noncomputable def Larry_shoulder_height : ℝ := Larry_full_height - 0.2 * Larry_full_height
noncomputable def combined_reach : ℝ := 9

theorem Barry_reach :
  combined_reach - Larry_shoulder_height = 5 := 
by
  -- Correct answer verification comparing combined reach minus Larry's shoulder height equals 5
  sorry

end Barry_reach_l225_225335


namespace ratio_of_m_l225_225462

theorem ratio_of_m (a b m m1 m2 : ℝ)
  (h1 : a * m^2 + b * m + c = 0)
  (h2 : (a / b + b / a) = 3 / 7)
  (h3 : a + b = (3 * m - 2) / m)
  (h4 : a * b = 7 / m)
  (h5 : (a + b)^2 = ab / (m * (7/ m)) - 2) :
  (m1 + m2 = 21) ∧ (m1 * m2 = 4) → 
  (m1/m2 + m2/m1 = 108.25) := sorry

end ratio_of_m_l225_225462


namespace custom_operator_example_l225_225729

def custom_operator (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

theorem custom_operator_example : custom_operator 5 3 = 4 := by
  sorry

end custom_operator_example_l225_225729


namespace archer_prob_6_or_less_l225_225712

noncomputable def prob_event_D (P_A P_B P_C : ℝ) : ℝ :=
  1 - (P_A + P_B + P_C)

theorem archer_prob_6_or_less :
  let P_A := 0.5
  let P_B := 0.2
  let P_C := 0.1
  prob_event_D P_A P_B P_C = 0.2 :=
by
  sorry

end archer_prob_6_or_less_l225_225712


namespace luigi_pizza_cost_l225_225002

theorem luigi_pizza_cost (num_pizzas pieces_per_pizza cost_per_piece : ℕ) 
  (h1 : num_pizzas = 4) 
  (h2 : pieces_per_pizza = 5) 
  (h3 : cost_per_piece = 4) :
  num_pizzas * pieces_per_pizza * cost_per_piece / pieces_per_pizza = 80 := by
  sorry

end luigi_pizza_cost_l225_225002


namespace solution_correct_l225_225692

noncomputable def satisfies_conditions (f : ℤ → ℝ) : Prop :=
  (f 1 = 5 / 2) ∧ (f 0 ≠ 0) ∧ (∀ m n : ℤ, f m * f n = f (m + n) + f (m - n))

theorem solution_correct (f : ℤ → ℝ) :
  satisfies_conditions f → ∀ n : ℤ, f n = 2^n + (1/2)^n :=
by sorry

end solution_correct_l225_225692


namespace correct_system_of_equations_l225_225600

theorem correct_system_of_equations (x y : ℕ) :
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) ↔ 
  (y = 7 * x + 7) ∧ (y = 9 * (x - 1)) :=
by
  sorry

end correct_system_of_equations_l225_225600


namespace num_terms_divisible_by_b_eq_gcd_l225_225472

theorem num_terms_divisible_by_b_eq_gcd (a b d : ℕ) (h_gcd : Nat.gcd a b = d) :
  (∃ count : ℕ, count = d ∧ ∀ k, (1 ≤ k ∧ k ≤ b) → (a * k) % b = 0 → k = (b / d) * i for some i : ℕ) :=
sorry

end num_terms_divisible_by_b_eq_gcd_l225_225472


namespace period_of_sin_minus_cos_l225_225196

def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem period_of_sin_minus_cos : ∀ x : ℝ, f (x + 2 * Real.pi) = f x :=
by
  intros x
  sorry

end period_of_sin_minus_cos_l225_225196


namespace fabian_total_cost_l225_225533

def cost_of_apples (kg : ℕ) (price_per_kg : ℕ) : ℕ :=
  kg * price_per_kg

def cost_of_sugar (packs : ℕ) (price_per_pack : ℕ) : ℕ :=
  packs * price_per_pack

def cost_of_walnuts (kg : ℕ) (price_per_kg : ℕ) : ℕ :=
  kg * price_per_kg

theorem fabian_total_cost : 
  let apples_cost := cost_of_apples 5 2 in
  let sugar_cost := cost_of_sugar 3 1 in
  let walnuts_cost := cost_of_walnuts (1/2) 6 in
  apples_cost + sugar_cost + walnuts_cost = 16 :=
by
  sorry

end fabian_total_cost_l225_225533


namespace probability_ball_sports_l225_225354

theorem probability_ball_sports (clubs : Finset String)
  (ball_clubs : Finset String)
  (count_clubs : clubs.card = 5)
  (count_ball_clubs : ball_clubs.card = 3)
  (h1 : "basketball" ∈ clubs)
  (h2 : "soccer" ∈ clubs)
  (h3 : "volleyball" ∈ clubs)
  (h4 : "swimming" ∈ clubs)
  (h5 : "gymnastics" ∈ clubs)
  (h6 : "basketball" ∈ ball_clubs)
  (h7 : "soccer" ∈ ball_clubs)
  (h8 : "volleyball" ∈ ball_clubs) :
  (2 / ((5 : ℝ) * (4 : ℝ)) * ((3 : ℝ) * (2 : ℝ)) = (3 / 10)) :=
by
  sorry

end probability_ball_sports_l225_225354


namespace shop_conditions_l225_225603

theorem shop_conditions (x y : ℕ) :
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) ↔ 
  ∃ x y : ℕ, 7 * x + 7 = y ∧ 9 * (x - 1) = y :=
sorry

end shop_conditions_l225_225603


namespace perimeter_of_triangle_l225_225782

theorem perimeter_of_triangle (r a : ℝ) (p : ℝ) (h1 : r = 3.5) (h2 : a = 56) :
  p = 32 :=
by
  sorry

end perimeter_of_triangle_l225_225782


namespace find_other_root_l225_225490

theorem find_other_root (a b c x : ℝ) (h₁ : a ≠ 0) 
  (h₂ : b ≠ 0) (h₃ : c ≠ 0)
  (h₄ : a * (b + 2 * c) * x^2 + b * (2 * c - a) * x + c * (2 * a - b) = 0)
  (h₅ : a * (b + 2 * c) - b * (2 * c - a) + c * (2 * a - b) = 0) :
  ∃ y : ℝ, y = - (c * (2 * a - b)) / (a * (b + 2 * c)) :=
sorry

end find_other_root_l225_225490


namespace vertex_on_xaxis_l225_225512

-- Definition of the parabola equation with vertex on the x-axis
def parabola (x m : ℝ) := x^2 - 8 * x + m

-- The problem statement: show that m = 16 given that the vertex of the parabola is on the x-axis
theorem vertex_on_xaxis (m : ℝ) : ∃ x : ℝ, parabola x m = 0 → m = 16 :=
by
  sorry

end vertex_on_xaxis_l225_225512


namespace total_cost_fencing_l225_225483

-- Define the conditions
def length : ℝ := 75
def breadth : ℝ := 25
def cost_per_meter : ℝ := 26.50

-- Define the perimeter of the rectangular plot
def perimeter : ℝ := 2 * length + 2 * breadth

-- Define the total cost of fencing
def total_cost : ℝ := perimeter * cost_per_meter

-- The theorem statement
theorem total_cost_fencing : total_cost = 5300 := 
by 
  -- This is the statement we want to prove
  sorry

end total_cost_fencing_l225_225483


namespace proof_complement_union_l225_225047

open Set

variable (U A B: Set Nat)

def complement_equiv_union (U A B: Set Nat) : Prop :=
  (U \ A) ∪ B = {0, 2, 3, 6}

theorem proof_complement_union: 
  U = {0, 1, 3, 5, 6, 8} → 
  A = {1, 5, 8} → 
  B = {2} → 
  complement_equiv_union U A B :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  -- Proof omitted
  sorry

end proof_complement_union_l225_225047


namespace min_L_pieces_correct_l225_225194

noncomputable def min_L_pieces : ℕ :=
  have pieces : Nat := 11
  pieces

theorem min_L_pieces_correct :
  min_L_pieces = 11 := 
by
  sorry

end min_L_pieces_correct_l225_225194


namespace cos_sum_equals_fraction_sqrt_13_minus_1_div_4_l225_225772

noncomputable def cos_sum : ℝ :=
  (Real.cos (2 * Real.pi / 17) +
   Real.cos (6 * Real.pi / 17) +
   Real.cos (8 * Real.pi / 17))

theorem cos_sum_equals_fraction_sqrt_13_minus_1_div_4 :
  cos_sum = (Real.sqrt 13 - 1) / 4 := 
sorry

end cos_sum_equals_fraction_sqrt_13_minus_1_div_4_l225_225772


namespace find_common_divisor_same_remainder_l225_225290

theorem find_common_divisor_same_remainder :
  let a := 480608
  let b := 508811
  let c := 723217
  let d1 := b - a -- 28203
  let d2 := c - b -- 214406
  let d3 := c - a -- 242609
  Int.gcd (Int.gcd d1 d2) d3 = 79 :=
by
  sorry

end find_common_divisor_same_remainder_l225_225290


namespace f_at_1_l225_225551

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x^2 + (5 : ℝ) * x
  else if x = 2 then 6
  else  - (x^2 + (5 : ℝ) * x)

theorem f_at_1 : f 1 = 4 :=
by {
  sorry
}

end f_at_1_l225_225551


namespace length_of_PQ_is_8_l225_225324

-- Define the lengths of the sides and conditions
variables (PQ QR PS SR : ℕ) (perimeter : ℕ)

-- State the conditions
def conditions : Prop :=
  SR = 16 ∧
  perimeter = 40 ∧
  PQ = QR ∧ QR = PS

-- State the goal
theorem length_of_PQ_is_8 (h : conditions PQ QR PS SR perimeter) : PQ = 8 :=
sorry

end length_of_PQ_is_8_l225_225324


namespace unique_pair_not_opposite_l225_225834

def QuantumPair (a b : String): Prop := ∃ oppositeMeanings : Bool, a ≠ b ∧ oppositeMeanings

theorem unique_pair_not_opposite :
  ∃ (a b : String), 
    (a = "increase of 2 years" ∧ b = "decrease of 2 liters") ∧ 
    (¬ QuantumPair a b) :=
by 
  sorry

end unique_pair_not_opposite_l225_225834


namespace x_intercept_is_3_l225_225218

-- Define the given points
def point1 : ℝ × ℝ := (2, -2)
def point2 : ℝ × ℝ := (6, 6)

-- Prove the x-intercept is 3
theorem x_intercept_is_3 (x : ℝ) :
  (∃ m b : ℝ, (∀ x1 y1 x2 y2 : ℝ, (y1 = m * x1 + b) ∧ (x1, y1) = point1 ∧ (x2, y2) = point2) ∧ y = 0 ∧ x = -b / m) → x = 3 :=
sorry

end x_intercept_is_3_l225_225218


namespace minimum_value_quot_l225_225441

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem minimum_value_quot (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : f a = f b) :
  (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2 :=
by
  sorry

end minimum_value_quot_l225_225441


namespace find_f_2011_l225_225867

def f: ℝ → ℝ :=
sorry

axiom f_periodicity (x : ℝ) : f (x + 3) = -f x
axiom f_initial_value : f 4 = -2

theorem find_f_2011 : f 2011 = 2 :=
by
  sorry

end find_f_2011_l225_225867


namespace sum_of_ages_l225_225059

theorem sum_of_ages (A B C : ℕ)
  (h1 : A = C + 8)
  (h2 : A + 10 = 3 * (C - 6))
  (h3 : B = 2 * C) :
  A + B + C = 80 := 
by 
  sorry

end sum_of_ages_l225_225059


namespace P_sufficient_but_not_necessary_for_Q_l225_225103

def P (x : ℝ) : Prop := (x - 3) * (x + 1) > 0
def Q (x : ℝ) : Prop := x^2 - 2 * x + 1 > 0

theorem P_sufficient_but_not_necessary_for_Q : 
  (∀ x : ℝ, P x → Q x) ∧ ¬ (∀ x : ℝ, Q x → P x) :=
by 
  sorry

end P_sufficient_but_not_necessary_for_Q_l225_225103


namespace player1_coins_l225_225177

theorem player1_coins (coin_distribution : Fin 9 → ℕ) :
  let rotations := 11
  let player_4_coins := 90
  let player_8_coins := 35
  ∀ player : Fin 9, player = 0 → 
    let player_1_coins := coin_distribution player
    (coin_distribution 3 = player_4_coins) →
    (coin_distribution 7 = player_8_coins) →
    player_1_coins = 57 := 
sorry

end player1_coins_l225_225177


namespace intersection_points_l225_225694

noncomputable def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 10)^2 = 50
noncomputable def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2 * (x - y) - 18 = 0

theorem intersection_points : 
  (circle1 3 3 ∧ circle2 3 3) ∧ (circle1 (-3) 5 ∧ circle2 (-3) 5) :=
by sorry

end intersection_points_l225_225694


namespace least_number_added_to_divide_l225_225803

-- Definitions of conditions
def lcm_three_five_seven_eight : ℕ := Nat.lcm (Nat.lcm (Nat.lcm 3 5) 7) 8
def remainder_28523_lcm := 28523 % lcm_three_five_seven_eight

-- Lean statement to prove the correct answer
theorem least_number_added_to_divide (n : ℕ) :
  n = lcm_three_five_seven_eight - remainder_28523_lcm :=
sorry

end least_number_added_to_divide_l225_225803


namespace cos_squared_pi_over_4_minus_alpha_l225_225435

theorem cos_squared_pi_over_4_minus_alpha (α : ℝ) (h : Real.tan (α + Real.pi / 4) = 3 / 4) :
  Real.cos (Real.pi / 4 - α) ^ 2 = 9 / 25 :=
by
  sorry

end cos_squared_pi_over_4_minus_alpha_l225_225435


namespace margie_change_is_6_25_l225_225616

-- The conditions are given as definitions in Lean
def numberOfApples : Nat := 5
def costPerApple : ℝ := 0.75
def amountPaid : ℝ := 10.00

-- The statement to be proved
theorem margie_change_is_6_25 :
  (amountPaid - (numberOfApples * costPerApple)) = 6.25 := 
  sorry

end margie_change_is_6_25_l225_225616


namespace billy_used_54_tickets_l225_225959

-- Definitions
def ferris_wheel_rides := 7
def bumper_car_rides := 3
def ferris_wheel_cost := 6
def bumper_car_cost := 4

-- Theorem Statement
theorem billy_used_54_tickets : 
  ferris_wheel_rides * ferris_wheel_cost + bumper_car_rides * bumper_car_cost = 54 := 
by
  sorry

end billy_used_54_tickets_l225_225959


namespace bisection_interval_l225_225189

def f(x : ℝ) := x^3 - 2 * x - 5

theorem bisection_interval :
  f 2 < 0 ∧ f 3 > 0 ∧ f 2.5 > 0 →
  ∃ a b : ℝ, a = 2 ∧ b = 2.5 ∧ f a * f b ≤ 0 :=
by
  sorry

end bisection_interval_l225_225189


namespace hunter_saw_32_frogs_l225_225882

noncomputable def total_frogs (g1 : ℕ) (g2 : ℕ) (d : ℕ) : ℕ :=
g1 + g2 + d

theorem hunter_saw_32_frogs :
  total_frogs 5 3 (2 * 12) = 32 := by
  sorry

end hunter_saw_32_frogs_l225_225882


namespace min_value_fraction_l225_225985

theorem min_value_fraction (x : ℝ) (h : x > 4) : 
  ∃ y, y = x - 4 ∧ (x + 11) / Real.sqrt (x - 4) = 2 * Real.sqrt 15 := by
  sorry

end min_value_fraction_l225_225985


namespace difference_of_two_smallest_integers_l225_225077

/--
The difference between the two smallest integers greater than 1 which, when divided by any integer 
\( k \) in the range from \( 3 \leq k \leq 13 \), leave a remainder of \( 2 \), is \( 360360 \).
-/
theorem difference_of_two_smallest_integers (n m : ℕ) (h_n : ∀ k : ℕ, 3 ≤ k ∧ k ≤ 13 → n % k = 2) (h_m : ∀ k : ℕ, 3 ≤ k ∧ k ≤ 13 → m % k = 2) (h_smallest : m > n) :
  m - n = 360360 :=
sorry

end difference_of_two_smallest_integers_l225_225077


namespace technology_elective_courses_l225_225323

theorem technology_elective_courses (m : ℕ) :
  let subject_elective := m,
      arts_elective := m + 9,
      technology_elective := 1 / 3 * arts_elective + 5
  in technology_elective = 1 / 3 * m + 8 :=
by
  sorry

end technology_elective_courses_l225_225323


namespace solution_set_l225_225016

-- Given conditions
variable (x : ℝ)

def inequality1 := 2 * x + 1 > 0
def inequality2 := (x + 1) / 3 > x - 1

-- The proof statement
theorem solution_set (h1 : inequality1 x) (h2 : inequality2 x) :
  -1 / 2 < x ∧ x < 2 :=
sorry

end solution_set_l225_225016


namespace total_cost_full_units_l225_225211

def total_units : Nat := 12
def cost_1_bedroom : Nat := 360
def cost_2_bedroom : Nat := 450
def num_2_bedroom : Nat := 7
def num_1_bedroom : Nat := total_units - num_2_bedroom

def total_cost : Nat := (num_1_bedroom * cost_1_bedroom) + (num_2_bedroom * cost_2_bedroom)

theorem total_cost_full_units : total_cost = 4950 := by
  -- proof would go here
  sorry

end total_cost_full_units_l225_225211


namespace gcd_digit_bound_l225_225579

theorem gcd_digit_bound
  (a b : ℕ)
  (h1 : 10^6 ≤ a)
  (h2 : a < 10^7)
  (h3 : 10^6 ≤ b)
  (h4 : b < 10^7)
  (h_lcm : 10^{11} ≤ Nat.lcm a b)
  (h_lcm2 : Nat.lcm a b < 10^{12}) :
  Nat.gcd a b < 10^3 :=
sorry

end gcd_digit_bound_l225_225579


namespace area_of_rectangle_is_32_proof_l225_225399

noncomputable def triangle_sides : ℝ := 7.3 + 5.4 + 11.3
def equality_of_perimeters (rectangle_length rectangle_width : ℝ) : Prop := 
  2 * (rectangle_length + rectangle_width) = triangle_sides

def rectangle_length (rectangle_width : ℝ) : ℝ := 2 * rectangle_width

def area_of_rectangle_is_32 (rectangle_width : ℝ) : Prop :=
  rectangle_length rectangle_width * rectangle_width = 32

theorem area_of_rectangle_is_32_proof : 
  ∃ (rectangle_width : ℝ), 
  equality_of_perimeters (rectangle_length rectangle_width) rectangle_width ∧ area_of_rectangle_is_32 rectangle_width :=
by
  sorry

end area_of_rectangle_is_32_proof_l225_225399


namespace find_n_l225_225386

theorem find_n (n : ℕ) (h : (1 + n) / (2 ^ n) = 3 / 16) : n = 5 :=
by sorry

end find_n_l225_225386


namespace inverse_function_properties_l225_225866

theorem inverse_function_properties {f : ℝ → ℝ} 
  (h_monotonic_decreasing : ∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 3 → f x2 < f x1)
  (h_range : ∀ y : ℝ, 4 ≤ y ∧ y ≤ 7 ↔ ∃ x : ℝ, 1 ≤ x ∧ x ≤ 3 ∧ y = f x)
  (h_inverse_exists : ∃ g : ℝ → ℝ, ∀ x : ℝ, f (g x) = x ∧ g (f x) = x) :
  ∃ g : ℝ → ℝ, (∀ y1 y2 : ℝ, 4 ≤ y1 ∧ y1 < y2 ∧ y2 ≤ 7 → g y2 < g y1) ∧ (∀ y : ℝ, 4 ≤ y ∧ y ≤ 7 → g y ≤ 3) :=
sorry

end inverse_function_properties_l225_225866


namespace oaks_not_adjacent_probability_l225_225808

theorem oaks_not_adjacent_probability :
  let total_trees := 13
  let oaks := 5
  let other_trees := total_trees - oaks
  let possible_slots := other_trees + 1
  let combinations := Nat.choose possible_slots oaks
  let total_arrangements := Nat.factorial total_trees / (Nat.factorial oaks * Nat.factorial (total_trees - oaks))
  let probability := combinations / total_arrangements
  probability = 1 / 220 :=
by
  sorry

end oaks_not_adjacent_probability_l225_225808


namespace stratified_sampling_third_year_l225_225217

-- The total number of students in the school
def total_students : ℕ := 2000

-- The probability of selecting a female student from the second year
def prob_female_second_year : ℚ := 0.19

-- The number of students to be selected through stratified sampling
def sample_size : ℕ := 100

-- The total number of third-year students
def third_year_students : ℕ := 500

-- The number of students to be selected from the third year in stratified sampling
def third_year_sample (total : ℕ) (third_year : ℕ) (sample : ℕ) : ℕ :=
  sample * third_year / total

-- Lean statement expressing the goal
theorem stratified_sampling_third_year :
  third_year_sample total_students third_year_students sample_size = 25 :=
by
  sorry

end stratified_sampling_third_year_l225_225217


namespace maximize_area_partition_l225_225374

noncomputable def optimLengthPartition (material: ℝ) (partitions: ℕ) : ℝ :=
  (material / (4 + partitions))

theorem maximize_area_partition :
  optimLengthPartition 24 (2 * 1) = 3 / 100 :=
by
  sorry

end maximize_area_partition_l225_225374


namespace quadrilateral_inequality_l225_225989

theorem quadrilateral_inequality 
  (A B C D : Type)
  (AB AC AD BC BD CD : ℝ)
  (hAB_pos : 0 < AB)
  (hBC_pos : 0 < BC)
  (hCD_pos : 0 < CD)
  (hDA_pos : 0 < DA)
  (hAC_pos : 0 < AC)
  (hBD_pos : 0 < BD): 
  AC * BD ≤ AB * CD + BC * AD := 
sorry

end quadrilateral_inequality_l225_225989


namespace line_BC_eq_l225_225304

def altitude1 (x y : ℝ) : Prop := x + y = 0
def altitude2 (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0
def point_A : ℝ × ℝ := (1, 2)

def line_eq (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

theorem line_BC_eq (x y : ℝ) :
  (∃ b c : ℝ × ℝ, altitude1 b.1 b.2 ∧ altitude2 c.1 c.2 ∧
                   line_eq 2 3 7 b.1 b.2 ∧ line_eq 2 3 7 c.1 c.2 ∧
                   b ≠ c) → 
    line_eq 2 3 7 x y :=
by sorry

end line_BC_eq_l225_225304


namespace no_real_solution_ineq_l225_225015

theorem no_real_solution_ineq (x : ℝ) (h : x ≠ 5) : ¬ (x^3 - 125) / (x - 5) < 0 := 
by
  sorry

end no_real_solution_ineq_l225_225015


namespace find_Y_l225_225697

theorem find_Y (Y : ℝ) (h : (100 + Y / 90) * 90 = 9020) : Y = 20 := 
by 
  sorry

end find_Y_l225_225697


namespace probability_three_heads_in_a_row_l225_225795

theorem probability_three_heads_in_a_row (h : ℝ) (p_head : h = 1/2) (ind_flips : ∀ (n : ℕ), true) : 
  (1/2 * 1/2 * 1/2 = 1/8) :=
by
  sorry

end probability_three_heads_in_a_row_l225_225795


namespace necessary_but_not_sufficient_l225_225069

theorem necessary_but_not_sufficient (a c : ℝ) : 
  (c ≠ 0) → (∀ (x y : ℝ), ax^2 + y^2 = c → (c = 0 → false) ∧ (c ≠ 0 → (∃ x y : ℝ, ax^2 + y^2 = c))) :=
by
  sorry

end necessary_but_not_sufficient_l225_225069


namespace sum_of_first_ten_terms_l225_225545

theorem sum_of_first_ten_terms (a : ℕ → ℝ)
  (h1 : a 3 ^ 2 + a 8 ^ 2 + 2 * a 3 * a 8 = 9)
  (h2 : ∀ n, a n < 0) :
  (5 * (a 3 + a 8) = -15) :=
sorry

end sum_of_first_ten_terms_l225_225545


namespace a5_a6_value_l225_225704

def S (n : ℕ) : ℕ := n^3

theorem a5_a6_value : S 6 - S 4 = 152 :=
by
  sorry

end a5_a6_value_l225_225704


namespace impossible_return_l225_225061

def Point := (ℝ × ℝ)

-- Conditions
def is_valid_point (p: Point) : Prop :=
  let (a, b) := p
  ∃ a_int b_int : ℤ, (a = a_int + b_int * Real.sqrt 2 ∧ b = a_int + b_int * Real.sqrt 2)

def valid_movement (p q: Point) : Prop :=
  let (x1, y1) := p
  let (x2, y2) := q
  abs x2 > abs x1 ∧ abs y2 > abs y1 

-- Theorem statement
theorem impossible_return (start: Point) (h: start = (1, Real.sqrt 2)) 
  (valid_start: is_valid_point start) :
  ∀ (p: Point), (is_valid_point p ∧ valid_movement start p) → p ≠ start :=
sorry

end impossible_return_l225_225061


namespace JulioHasMoreSoda_l225_225756

-- Define the number of bottles each person has
def JulioOrangeBottles : ℕ := 4
def JulioGrapeBottles : ℕ := 7
def MateoOrangeBottles : ℕ := 1
def MateoGrapeBottles : ℕ := 3

-- Define the volume of each bottle in liters
def BottleVolume : ℕ := 2

-- Define the total liters of soda each person has
def JulioTotalLiters : ℕ := JulioOrangeBottles * BottleVolume + JulioGrapeBottles * BottleVolume
def MateoTotalLiters : ℕ := MateoOrangeBottles * BottleVolume + MateoGrapeBottles * BottleVolume

-- Prove the difference in total liters of soda between Julio and Mateo
theorem JulioHasMoreSoda : JulioTotalLiters - MateoTotalLiters = 14 := by
  sorry

end JulioHasMoreSoda_l225_225756


namespace probability_at_least_one_red_l225_225897

def total_balls : ℕ := 6
def red_balls : ℕ := 4
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_at_least_one_red :
  (choose_two red_balls + red_balls * (total_balls - red_balls - 1) / 2) / choose_two total_balls = 14 / 15 :=
sorry

end probability_at_least_one_red_l225_225897


namespace tan_product_l225_225238

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end tan_product_l225_225238


namespace building_height_l225_225376

theorem building_height (h : ℕ) 
  (shadow_building : ℕ) 
  (shadow_pole : ℕ) 
  (height_pole : ℕ) 
  (ratio_proportional : shadow_building * height_pole = shadow_pole * h) 
  (shadow_building_val : shadow_building = 63) 
  (shadow_pole_val : shadow_pole = 32) 
  (height_pole_val : height_pole = 28) : 
  h = 55 := 
by 
  sorry

end building_height_l225_225376


namespace factor_polynomial_l225_225699

theorem factor_polynomial {x : ℝ} : 4 * x^3 - 16 * x = 4 * x * (x + 2) * (x - 2) := 
sorry

end factor_polynomial_l225_225699


namespace leggings_needed_l225_225443

theorem leggings_needed (dogs : ℕ) (cats : ℕ) (dogs_legs : ℕ) (cats_legs : ℕ) (pair_of_leggings : ℕ) 
                        (hd : dogs = 4) (hc : cats = 3) (hl1 : dogs_legs = 4) (hl2 : cats_legs = 4) (lp : pair_of_leggings = 2)
                        : (dogs * dogs_legs + cats * cats_legs) / pair_of_leggings = 14 :=
by
  sorry

end leggings_needed_l225_225443


namespace total_kids_in_lawrence_county_l225_225461

theorem total_kids_in_lawrence_county :
  ∀ (h c T : ℕ), h = 274865 → c = 38608 → T = h + c → T = 313473 :=
by
  intros h c T h_eq c_eq T_eq
  rw [h_eq, c_eq] at T_eq
  exact T_eq

end total_kids_in_lawrence_county_l225_225461


namespace choir_members_l225_225017

theorem choir_members (n k c : ℕ) (h1 : n = k^2 + 11) (h2 : n = c * (c + 5)) : n = 300 :=
sorry

end choir_members_l225_225017


namespace tan_identity_l225_225247

theorem tan_identity :
  tan (π / 9) * tan (2 * π / 9) * tan (4 * π / 9) = 3 :=
sorry

end tan_identity_l225_225247


namespace construct_all_naturals_starting_from_4_l225_225185

-- Define the operations f, g, h
def f (n : ℕ) : ℕ := 10 * n
def g (n : ℕ) : ℕ := 10 * n + 4
def h (n : ℕ) : ℕ := if n % 2 = 0 then n / 2 else n  -- h is only meaningful if n is even

-- Main theorem: prove that starting from 4, every natural number can be constructed
theorem construct_all_naturals_starting_from_4 :
  ∀ (n : ℕ), ∃ (k : ℕ), (f^[k] 4 = n ∨ g^[k] 4 = n ∨ h^[k] 4 = n) :=
by sorry


end construct_all_naturals_starting_from_4_l225_225185


namespace max_sum_of_arithmetic_sequence_l225_225341

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) 
(h1 : 3 * a 8 = 5 * a 13) 
(h2 : a 1 > 0)
(hS : ∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * d)) :
S 20 > S 21 ∧ S 20 > S 10 ∧ S 20 > S 11 :=
sorry

end max_sum_of_arithmetic_sequence_l225_225341


namespace constant_function_l225_225285

theorem constant_function (f : ℕ → ℤ)
  (h1 : ∀ a b : ℕ, a > 0 → b > 0 → a ∣ b → f(a) ≥ f(b))
  (h2 : ∀ a b : ℕ, a > 0 → b > 0 → f(a * b) + f(a^2 + b^2) = f(a) + f(b)) :
  ∃ C : ℤ, ∀ n : ℕ, f(n) = C :=
by
  sorry

end constant_function_l225_225285


namespace units_digit_of_largest_power_of_two_dividing_2_pow_5_factorial_l225_225088

/-- Find the units digit of the largest power of 2 that divides into (2^5)! -/
theorem units_digit_of_largest_power_of_two_dividing_2_pow_5_factorial : ∃ d : ℕ, d = 8 := by
  sorry

end units_digit_of_largest_power_of_two_dividing_2_pow_5_factorial_l225_225088


namespace gcd_digit_bound_l225_225590

theorem gcd_digit_bound {a b : ℕ} (ha : 10^6 ≤ a ∧ a < 10^7) (hb : 10^6 ≤ b ∧ b < 10^7) (hlcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 10^3 :=
by
  sorry

end gcd_digit_bound_l225_225590


namespace gravitational_equal_forces_point_l225_225410

variable (d M m : ℝ) (hM : 0 < M) (hm : 0 < m) (hd : 0 < d)

theorem gravitational_equal_forces_point :
  ∃ x : ℝ, (0 < x ∧ x < d) ∧ x = d / (1 + Real.sqrt (m / M)) :=
by
  sorry

end gravitational_equal_forces_point_l225_225410


namespace total_blue_points_l225_225619

variables (a b c d : ℕ)

theorem total_blue_points (h1 : a * b = 56) (h2 : c * d = 50) (h3 : a + b = c + d) :
  a + b = 15 :=
sorry

end total_blue_points_l225_225619


namespace general_formula_sum_first_n_terms_l225_225296

-- Definitions for arithmetic sequence, geometric aspects and sum conditions 
variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {d : ℕ}
variable {b_n : ℕ → ℕ}
variable {T_n : ℕ → ℕ}

-- Given conditions
axiom sum_condition (S3 S5 : ℕ) : S3 + S5 = 50
axiom common_difference : d ≠ 0
axiom first_term (a1 : ℕ) : a_n 1 = a1
axiom geometric_conditions (a1 a4 a13 : ℕ)
  (h1 : a_n 1 = a1) (h4 : a_n 4 = a4) (h13 : a_n 13 = a13) :
  a4 = a1 + 3 * d ∧ a13 = a1 + 12 * d ∧ (a4 ^ 2 = a1 * a13)

-- Proving the general formula for a_n
theorem general_formula (a_n : ℕ → ℕ)
  (h : ∀ (n : ℕ), a_n n = 2 * n + 1) : 
  a_n n = 2 * n + 1 := 
sorry

-- Proving the sum of the first n terms of sequence {b_n}
theorem sum_first_n_terms (a_n b_n : ℕ → ℕ) (T_n : ℕ → ℕ)
  (h_bn : ∀ (n : ℕ), b_n n = (2 * n + 1) * 2 ^ (n - 1))
  (h_Tn: ∀ (n : ℕ), T_n n = 1 + (2 * n - 1) * 2^n) :
  T_n n = 1 + (2 * n - 1) * 2^n :=
sorry

end general_formula_sum_first_n_terms_l225_225296


namespace minimum_phi_l225_225450

noncomputable def initial_function (x : ℝ) (ϕ : ℝ) : ℝ :=
  2 * Real.sin (4 * x + ϕ)

noncomputable def translated_function (x : ℝ) (ϕ : ℝ) : ℝ :=
  2 * Real.sin (4 * (x - (Real.pi / 6)) + ϕ)

theorem minimum_phi (ϕ : ℝ) :
  (∃ k : ℤ, ϕ = k * Real.pi + 7 * Real.pi / 6) →
  (∃ ϕ_min : ℝ, (ϕ_min = ϕ ∧ ϕ_min = Real.pi / 6)) :=
by
  sorry

end minimum_phi_l225_225450


namespace find_x_plus_2y_squared_l225_225564

theorem find_x_plus_2y_squared (x y : ℝ) (h1 : x * (x + 2 * y) = 48) (h2 : y * (x + 2 * y) = 72) :
  (x + 2 * y) ^ 2 = 96 := 
sorry

end find_x_plus_2y_squared_l225_225564


namespace elena_pens_l225_225531

theorem elena_pens (X Y : ℕ) (h1 : X + Y = 12) (h2 : 4*X + 22*Y = 420) : X = 9 := by
  sorry

end elena_pens_l225_225531


namespace tan_product_l225_225274

theorem tan_product : 
  tan (Real.pi / 9) * tan (2 * Real.pi / 9) * tan (4 * Real.pi / 9) = 3 :=
by
  sorry

end tan_product_l225_225274


namespace part1_part2_l225_225988

variables (α β : Real)

theorem part1 (h1 : Real.cos (α + β) = 1 / 3) (h2 : Real.sin α * Real.sin β = 1 / 4) :
  Real.cos α * Real.cos β = 7 / 12 := 
sorry

theorem part2 (h1 : Real.cos (α + β) = 1 / 3) (h2 : Real.sin α * Real.sin β = 1 / 4) :
  Real.cos (2 * α - 2 * β) = 7 / 18 := 
sorry

end part1_part2_l225_225988


namespace Donny_change_l225_225419

theorem Donny_change (tank_capacity : ℕ) (initial_fuel : ℕ) (money_available : ℕ) (fuel_cost_per_liter : ℕ) 
  (h1 : tank_capacity = 150) 
  (h2 : initial_fuel = 38) 
  (h3 : money_available = 350) 
  (h4 : fuel_cost_per_liter = 3) : 
  money_available - (tank_capacity - initial_fuel) * fuel_cost_per_liter = 14 := 
by 
  sorry

end Donny_change_l225_225419


namespace cone_lateral_area_l225_225553

theorem cone_lateral_area (cos_ASB : ℝ)
  (angle_SA_base : ℝ)
  (triangle_SAB_area : ℝ) :
  cos_ASB = 7 / 8 →
  angle_SA_base = 45 →
  triangle_SAB_area = 5 * Real.sqrt 15 →
  (lateral_area : ℝ) = 40 * Real.sqrt 2 * Real.pi :=
by
  intros h1 h2 h3
  sorry

end cone_lateral_area_l225_225553


namespace inequality_proof_l225_225449

theorem inequality_proof {x y z : ℝ} (n : ℕ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x + y + z = 1)
  : (x^4 / (y * (1 - y^n))) + (y^4 / (z * (1 - z^n))) + (z^4 / (x * (1 - x^n))) 
    ≥ (3^n) / (3^(n - 2) - 9) :=
by
  sorry

end inequality_proof_l225_225449


namespace cyclists_cannot_reach_point_B_l225_225925

def v1 := 35 -- Speed of the first cyclist in km/h
def v2 := 25 -- Speed of the second cyclist in km/h
def t := 2   -- Total time in hours
def d  := 30 -- Distance from A to B in km

-- Each cyclist does not rest simultaneously
-- Time equations based on their speed proportions

theorem cyclists_cannot_reach_point_B 
  (v1 := 35) (v2 := 25) (t := 2) (d := 30) 
  (h1 : t * (v1 * (5 / (5 + 7)) / 60) + t * (v2 * (7 / (5 + 7)) / 60) < d) : 
  False := 
sorry

end cyclists_cannot_reach_point_B_l225_225925


namespace subway_ways_l225_225930

theorem subway_ways (total_ways : ℕ) (bus_ways : ℕ) (h1 : total_ways = 7) (h2 : bus_ways = 4) :
  total_ways - bus_ways = 3 :=
by
  sorry

end subway_ways_l225_225930


namespace product_of_two_integers_l225_225634

def gcd_lcm_prod (x y : ℕ) :=
  Nat.gcd x y = 8 ∧ Nat.lcm x y = 48

theorem product_of_two_integers (x y : ℕ) (h : gcd_lcm_prod x y) : x * y = 384 :=
by
  sorry

end product_of_two_integers_l225_225634


namespace cars_sold_proof_l225_225805

noncomputable def total_cars_sold : Nat := 300
noncomputable def perc_audi : ℝ := 0.10
noncomputable def perc_toyota : ℝ := 0.15
noncomputable def perc_acura : ℝ := 0.20
noncomputable def perc_honda : ℝ := 0.18

theorem cars_sold_proof : total_cars_sold * (1 - (perc_audi + perc_toyota + perc_acura + perc_honda)) = 111 := by
  sorry

end cars_sold_proof_l225_225805


namespace tan_identity_proof_l225_225270

noncomputable def tan_identity : ℝ := Real.tan (π / 9) * Real.tan (2 * (π / 9)) * Real.tan (4 * (π / 9))

theorem tan_identity_proof : tan_identity = 1 := 
by 
  -- Use the polynomial relation: 9t - 84t^3 + 126t^5 - 36t^7 + t^9 = 0
  -- derived from \tan(9θ)
  -- And factor it to get t(t^8 - 84t^6 + 126t^4 - 36t^2 + 1) = 0
  -- For theta = π/9, 2π/9, 4π/9
  -- Use Vieta’s formulas to find the product
  sorry

end tan_identity_proof_l225_225270


namespace day_crew_fraction_correct_l225_225223

-- Given conditions
variables (D W : ℕ)
def night_boxes_per_worker := (5 : ℚ) / 8 * D
def night_workers := (3 : ℚ) / 5 * W

-- Total boxes loaded
def total_day_boxes := D * W
def total_night_boxes := night_boxes_per_worker D * night_workers W

-- Fraction of boxes loaded by day crew
def fraction_loaded_by_day_crew := total_day_boxes D W / (total_day_boxes D W + total_night_boxes D W)

-- Theorem to prove
theorem day_crew_fraction_correct (D W : ℕ) : fraction_loaded_by_day_crew D W = (8 : ℚ) / 11 :=
by
  sorry

end day_crew_fraction_correct_l225_225223


namespace smallest_integer_ending_in_6_divisible_by_13_l225_225928

theorem smallest_integer_ending_in_6_divisible_by_13 (n : ℤ) (h1 : ∃ n : ℤ, 10 * n + 6 = x) (h2 : x % 13 = 0) : x = 26 :=
  sorry

end smallest_integer_ending_in_6_divisible_by_13_l225_225928


namespace part1_part2_l225_225716

-- Definitions for the problem
def f (x a : ℝ) : ℝ := |x - a| + 3 * x

-- Part (1)
theorem part1 (x : ℝ) (h : f x 1 ≥ 3 * x + 2) : x ≥ 3 ∨ x ≤ -1 :=
sorry

-- Part (2)
theorem part2 (h : ∀ x, f x a ≤ 0 → x ≤ -1) : a = 2 :=
sorry

end part1_part2_l225_225716


namespace find_m_l225_225711

variable {a b c m : ℝ}

theorem find_m (h1 : a + b = 4)
               (h2 : a * b = m)
               (h3 : b + c = 8)
               (h4 : b * c = 5 * m) : m = 0 ∨ m = 3 :=
by {
  sorry
}

end find_m_l225_225711


namespace minimum_a_l225_225563

theorem minimum_a (x : ℝ) (h : ∀ x ≥ 0, x * Real.exp x + a * Real.exp x * Real.log (x + 1) + 1 ≥ Real.exp x * (x + 1) ^ a) : 
    a ≥ -1 := by
  sorry

end minimum_a_l225_225563


namespace daves_initial_apps_l225_225843

theorem daves_initial_apps : ∃ (X : ℕ), X + 11 - 17 = 4 ∧ X = 10 :=
by {
  sorry
}

end daves_initial_apps_l225_225843


namespace avg_age_10_students_l225_225777

-- Defining the given conditions
def avg_age_15_students : ℕ := 15
def total_students : ℕ := 15
def avg_age_4_students : ℕ := 14
def num_4_students : ℕ := 4
def age_15th_student : ℕ := 9

-- Calculating the total age based on given conditions
def total_age_15_students : ℕ := avg_age_15_students * total_students
def total_age_4_students : ℕ := avg_age_4_students * num_4_students
def total_age_10_students : ℕ := total_age_15_students - total_age_4_students - age_15th_student

-- Problem to be proved
theorem avg_age_10_students : total_age_10_students / 10 = 16 := 
by sorry

end avg_age_10_students_l225_225777


namespace gcd_has_at_most_3_digits_l225_225576

noncomputable def lcm (a b : ℕ) : ℕ := sorry -- definition for lcm is already in Mathlib
noncomputable def gcd (a b : ℕ) : ℕ := sorry -- definition for gcd is already in Mathlib

theorem gcd_has_at_most_3_digits
  (a b : ℕ)
  (ha : 10^6 ≤ a ∧ a < 10^7)  -- a is a 7-digit integer
  (hb : 10^6 ≤ b ∧ b < 10^7)  -- b is a 7-digit integer
  (hlcm_digits : 10^11 ≤ lcm a b ∧ lcm a b < 10^12)  -- lcm of a and b has 12 digits
  : gcd a b < 10^3 := by
  sorry

end gcd_has_at_most_3_digits_l225_225576


namespace mod_inverse_13_997_l225_225195

-- The theorem statement
theorem mod_inverse_13_997 : ∃ x : ℕ, 0 ≤ x ∧ x < 997 ∧ (13 * x) % 997 = 1 ∧ x = 767 := 
by
  sorry

end mod_inverse_13_997_l225_225195


namespace min_alpha_beta_l225_225864

theorem min_alpha_beta (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1)
  (alpha : ℝ := a + 1 / a) (beta : ℝ := b + 1 / b) :
  alpha + beta ≥ 10 := by
  sorry

end min_alpha_beta_l225_225864


namespace greatest_possible_gcd_3Hn_n_plus_1_l225_225091

theorem greatest_possible_gcd_3Hn_n_plus_1 (n : ℕ) (h : n > 0) :
  let H_n := 2 * n^2 - n in gcd (3 * H_n) (n + 1) ≤ 12 :=
by
  let H_n := 2 * n^2 - n
  have h₁ : H_n = 2 * n^2 - n := rfl
  have h₂ : 3 * H_n = 6 * n^2 - 3 * n := by simp [h₁]
  have h₃ : gcd (6 * n^2 - 3 * n) (n + 1) = gcd (12, n + 1) :=
    by sorry  -- arithmetic and gcd properties
  have h₄ : gcd (12, n + 1) ≤ 12 := by simp [gcd_le_right, gcd_le_left]
  exact h₄

end greatest_possible_gcd_3Hn_n_plus_1_l225_225091


namespace homework_checked_on_friday_l225_225375

-- Define the events
def event_no_homework_checked : Prop := ∀ (d : ℕ), d ∈ {0, 1, 2, 3, 4} → ¬ checked d
def event_homework_checked_friday : Prop := checked 4
def event_homework_not_checked_until_thursday : Prop := ∀ (d : ℕ), d ∈ {0, 1, 2, 3} → ¬ checked d

-- Define the probabilities
def prob_teacher_checks_homework : ℙ := 1 / 2
def prob_teacher_does_not_check_homework : ℙ := 1 / 2
def prob_day_check (d : ℕ) : ℙ := 1 / 5

-- Main statement
theorem homework_checked_on_friday : 
  ℙ (event_homework_checked_friday | event_homework_not_checked_until_thursday) = 1 / 6 := 
begin
  sorry -- Proof to be provided
end

end homework_checked_on_friday_l225_225375


namespace tan_product_equals_three_l225_225245

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end tan_product_equals_three_l225_225245


namespace time_needed_to_gather_remaining_flowers_l225_225147

-- conditions
def classmates : ℕ := 30
def time_per_flower : ℕ := 10
def gathering_time : ℕ := 2 * 60
def lost_flowers : ℕ := 3

-- question and proof goal
theorem time_needed_to_gather_remaining_flowers : 
  let flowers_needed := classmates - ((gathering_time / time_per_flower) - lost_flowers)
  flowers_needed * time_per_flower = 210 :=
by
  sorry

end time_needed_to_gather_remaining_flowers_l225_225147


namespace tan_product_equals_three_l225_225243

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end tan_product_equals_three_l225_225243


namespace jennifer_initial_oranges_l225_225138

theorem jennifer_initial_oranges (O : ℕ) : 
  ∀ (pears apples remaining_fruits : ℕ),
    pears = 10 →
    apples = 2 * pears →
    remaining_fruits = pears - 2 + apples - 2 + O - 2 →
    remaining_fruits = 44 →
    O = 20 :=
by
  intros pears apples remaining_fruits h1 h2 h3 h4
  sorry

end jennifer_initial_oranges_l225_225138


namespace base_h_addition_eq_l225_225432

theorem base_h_addition_eq (h : ℕ) (h_eq : h = 9) : 
  (8 * h^3 + 3 * h^2 + 7 * h + 4) + (6 * h^3 + 9 * h^2 + 2 * h + 5) = 1 * h^4 + 5 * h^3 + 3 * h^2 + 0 * h + 9 :=
by
  rw [h_eq]
  sorry

end base_h_addition_eq_l225_225432


namespace initial_volume_of_solution_l225_225391

theorem initial_volume_of_solution (V : ℝ) (h0 : 0.10 * V = 0.08 * (V + 20)) : V = 80 :=
by
  sorry

end initial_volume_of_solution_l225_225391


namespace radius_of_circle_l225_225937

noncomputable def circle_radius {k : ℝ} (hk : k > -6) : ℝ := 6 * Real.sqrt 2 + 6

theorem radius_of_circle (k : ℝ) (hk : k > -6)
  (tangent_y_eq_x : ∀ (P : ℝ × ℝ), P.1 = 0 → P.2 = k → dist P (0, P.2) = 6 * Real.sqrt 2 + 6)
  (tangent_y_eq_negx : ∀ (P : ℝ × ℝ), P.1 = 0 → P.2 = k → dist P (0, -P.2) = 6 * Real.sqrt 2 + 6)
  (tangent_y_eq_neg6 : ∀ (P : ℝ × ℝ), P.1 = 0 → P.2 = k → dist P (0, -6) = 6 * Real.sqrt 2 + 6) :
  circle_radius hk = 6 * Real.sqrt 2 + 6 :=
by
  sorry

end radius_of_circle_l225_225937


namespace impossible_distance_l225_225874

noncomputable def radius_O1 : ℝ := 2
noncomputable def radius_O2 : ℝ := 5

theorem impossible_distance :
  ∀ (d : ℝ), ¬ (radius_O1 ≠ radius_O2 → ¬ (d < abs (radius_O2 - radius_O1) ∨ d > radius_O2 + radius_O1) → d = 5) :=
by
  sorry

end impossible_distance_l225_225874


namespace tan_product_l225_225237

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end tan_product_l225_225237


namespace final_ratio_of_milk_to_water_l225_225740

-- Initial conditions definitions
def initial_milk_ratio : ℚ := 5 / 8
def initial_water_ratio : ℚ := 3 / 8
def additional_milk : ℚ := 8
def total_capacity : ℚ := 72

-- Final ratio statement
theorem final_ratio_of_milk_to_water :
  (initial_milk_ratio * (total_capacity - additional_milk) + additional_milk) / (initial_water_ratio * (total_capacity - additional_milk)) = 2 := by
  sorry

end final_ratio_of_milk_to_water_l225_225740


namespace alex_piles_of_jelly_beans_l225_225832

theorem alex_piles_of_jelly_beans : 
  ∀ (initial_weight eaten weight_per_pile remaining_weight piles : ℕ),
    initial_weight = 36 →
    eaten = 6 →
    weight_per_pile = 10 →
    remaining_weight = initial_weight - eaten →
    piles = remaining_weight / weight_per_pile →
    piles = 3 :=
by
  intros initial_weight eaten weight_per_pile remaining_weight piles h_init h_eat h_wpile h_remaining h_piles
  sorry

end alex_piles_of_jelly_beans_l225_225832


namespace two_abs_inequality_l225_225430

theorem two_abs_inequality (x y : ℝ) :
  2 * abs (x + y) ≤ abs x + abs y ↔ 
  (x ≥ 0 ∧ -3 * x ≤ y ∧ y ≤ -x / 3) ∨ 
  (x < 0 ∧ -x / 3 ≤ y ∧ y ≤ -3 * x) :=
by
  sorry

end two_abs_inequality_l225_225430


namespace min_ω_value_l225_225170

def min_ω (ω : Real) : Prop :=
  ω > 0 ∧ (∃ k : Int, ω = 2 * k + 2 / 3)

theorem min_ω_value : ∃ ω : Real, min_ω ω ∧ ω = 2 / 3 := by
  sorry

end min_ω_value_l225_225170


namespace amount_each_girl_gets_l225_225950

theorem amount_each_girl_gets
  (B G : ℕ) 
  (total_sum : ℝ)
  (amount_each_boy : ℝ)
  (sum_boys_girls : B + G = 100)
  (total_sum_distributed : total_sum = 312)
  (amount_boy : amount_each_boy = 3.60)
  (B_approx : B = 60) :
  (total_sum - amount_each_boy * B) / G = 2.40 := 
by 
  sorry

end amount_each_girl_gets_l225_225950


namespace tan_product_l225_225258

noncomputable def tan_pi_nine := Real.tan (Real.pi / 9)
noncomputable def tan_two_pi_nine := Real.tan (2 * Real.pi / 9)
noncomputable def tan_four_pi_nine := Real.tan (4 * Real.pi / 9)

-- Key polynomial condition
def poly := λ t : ℝ, t^8 - 36 * t^6 + 126 * t^4 - 84 * t^2 + 9

-- Condition: the polynomial has roots tan(π/9), tan(2π/9), tan(4π/9)
axiom poly_roots: 
  (poly tan_pi_nine = 0 ∧
   poly tan_two_pi_nine = 0 ∧
   poly tan_four_pi_nine = 0)

-- The proof statement
theorem tan_product : 
  tan_pi_nine * 
  tan_two_pi_nine * 
  tan_four_pi_nine = 3 :=
sorry

end tan_product_l225_225258


namespace exists_root_satisfying_inequality_l225_225075

theorem exists_root_satisfying_inequality (n : ℕ) (x : ℝ) :
  n^2 * x^2 - (2 * n^2 + n) * x + (n^2 + n - 6) ≤ 0 → x = 1 :=
sorry

end exists_root_satisfying_inequality_l225_225075


namespace shaded_triangle_area_l225_225604

-- Definitions and conditions
def grid_width : ℕ := 15
def grid_height : ℕ := 5

def larger_triangle_base : ℕ := grid_width
def larger_triangle_height : ℕ := grid_height - 1

def smaller_triangle_base : ℕ := 12
def smaller_triangle_height : ℕ := 3

-- The proof problem stating that the area of the smaller shaded triangle is 18 units
theorem shaded_triangle_area :
  (smaller_triangle_base * smaller_triangle_height) / 2 = 18 :=
by
  sorry

end shaded_triangle_area_l225_225604


namespace remainder_when_divided_by_6_l225_225380

theorem remainder_when_divided_by_6 (a : ℕ) (h1 : a % 2 = 1) (h2 : a % 3 = 2) : a % 6 = 5 :=
sorry

end remainder_when_divided_by_6_l225_225380


namespace arithmetic_sequence_condition_l225_225745

theorem arithmetic_sequence_condition (a : ℕ → ℝ) (h : 2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36) : 
a 6 = 3 := 
by 
  sorry

end arithmetic_sequence_condition_l225_225745


namespace bob_cleaning_time_is_correct_l225_225220

-- Definitions for conditions
def timeAliceTakes : ℕ := 32
def bobTimeFactor : ℚ := 3 / 4

-- Theorem to prove
theorem bob_cleaning_time_is_correct : (bobTimeFactor * timeAliceTakes : ℚ) = 24 := 
by
  sorry

end bob_cleaning_time_is_correct_l225_225220


namespace tan_80_l225_225447

theorem tan_80 (m : ℝ) (h : Real.cos (100 * Real.pi / 180) = m) :
    Real.tan (80 * Real.pi / 180) = Real.sqrt (1 - m^2) / -m :=
by
  sorry

end tan_80_l225_225447


namespace tan_product_eq_three_l225_225234

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end tan_product_eq_three_l225_225234


namespace average_is_700_l225_225570

-- Define the list of known numbers
def numbers_without_x : List ℕ := [744, 745, 747, 748, 749, 752, 752, 753, 755]

-- Define the value of x
def x : ℕ := 755

-- Define the list of all numbers including x
def all_numbers : List ℕ := numbers_without_x.append [x]

-- Define the total length of the list containing x
def n : ℕ := all_numbers.length

-- Define the sum of the numbers in the list including x
noncomputable def sum_all_numbers : ℕ := all_numbers.sum

-- Define the average formula
noncomputable def average : ℕ := sum_all_numbers / n

-- State the theorem
theorem average_is_700 : average = 700 := by
  sorry

end average_is_700_l225_225570


namespace ratio_of_shaded_area_to_non_shaded_area_l225_225186

noncomputable def equilateral_triangle_shaded_area_ratio (s : ℝ) : ℝ :=
  let area_triangle : ℝ := (Math.sqrt 3) / 4 * s^2
  let area_inner_triangle : ℝ := (Math.sqrt 3) / 16 * s^2
  let trapezoid_area (b1 b2 h : ℝ) : ℝ := (1/2) * (b1 + b2) * h
  let area_trapezoids : ℝ := 3 * trapezoid_area (s/2) (s/4) ((Math.sqrt 3)/4 * s)
  let shaded_area : ℝ := area_inner_triangle + area_trapezoids
  let non_shaded_area : ℝ := area_triangle - shaded_area
  shaded_area / non_shaded_area

theorem ratio_of_shaded_area_to_non_shaded_area (s : ℝ) (s_pos : 0 < s) :
  equilateral_triangle_shaded_area_ratio s = 11 / 21 :=
by
  sorry

end ratio_of_shaded_area_to_non_shaded_area_l225_225186


namespace sum_computation_l225_225067

noncomputable def ceil_minus_floor (x : ℝ) : ℝ :=
  if x ≠ ⌊x⌋ then 1 else 0

def is_power_of_three (n : ℕ) : Prop :=
  ∃ (j : ℕ), 3^j = n

theorem sum_computation :
  (∑ k in Finset.range 501, k * (ceil_minus_floor (Real.log k / Real.log 3))) = 124886 :=
by
  sorry

end sum_computation_l225_225067


namespace identify_wrong_operator_l225_225497

def original_expr (x y z w u v p q : Int) : Int := x + y - z + w - u + v - p + q
def wrong_expr (x y z w u v p q : Int) : Int := x + y - z - w - u + v - p + q

theorem identify_wrong_operator :
  original_expr 3 5 7 9 11 13 15 17 ≠ -4 →
  wrong_expr 3 5 7 9 11 13 15 17 = -4 :=
by
  sorry

end identify_wrong_operator_l225_225497


namespace percentage_of_200_l225_225191

theorem percentage_of_200 : ((1/4) / 100) * 200 = 0.5 := 
by
  sorry

end percentage_of_200_l225_225191


namespace simplify_fraction_l225_225013

theorem simplify_fraction :
  (1 / (1 / (Real.sqrt 2 + 1) + 1 / (Real.sqrt 5 - 2))) =
  ((Real.sqrt 2 + Real.sqrt 5 - 1) / (6 + 2 * Real.sqrt 10)) :=
by
  sorry

end simplify_fraction_l225_225013


namespace moles_of_KHSO4_formed_l225_225106

-- Chemical reaction definition
def reaction (n_KOH n_H2SO4 : ℕ) : ℕ :=
  if n_KOH = n_H2SO4 then n_KOH else 0

-- Given conditions
def moles_KOH : ℕ := 2
def moles_H2SO4 : ℕ := 2

-- Proof statement to be proved
theorem moles_of_KHSO4_formed : reaction moles_KOH moles_H2SO4 = 2 :=
by sorry

end moles_of_KHSO4_formed_l225_225106


namespace tan_identity_proof_l225_225266

noncomputable def tan_identity : ℝ := Real.tan (π / 9) * Real.tan (2 * (π / 9)) * Real.tan (4 * (π / 9))

theorem tan_identity_proof : tan_identity = 1 := 
by 
  -- Use the polynomial relation: 9t - 84t^3 + 126t^5 - 36t^7 + t^9 = 0
  -- derived from \tan(9θ)
  -- And factor it to get t(t^8 - 84t^6 + 126t^4 - 36t^2 + 1) = 0
  -- For theta = π/9, 2π/9, 4π/9
  -- Use Vieta’s formulas to find the product
  sorry

end tan_identity_proof_l225_225266


namespace average_rainfall_in_normal_year_l225_225924

def first_day_rainfall : ℕ := 26
def second_day_rainfall : ℕ := 34
def third_day_rainfall : ℕ := second_day_rainfall - 12
def total_rainfall_this_year : ℕ := first_day_rainfall + second_day_rainfall + third_day_rainfall
def rainfall_difference : ℕ := 58

theorem average_rainfall_in_normal_year :
  (total_rainfall_this_year + rainfall_difference) = 140 :=
by
  sorry

end average_rainfall_in_normal_year_l225_225924


namespace units_digit_p_l225_225097

theorem units_digit_p (p : ℕ) (h1 : p % 2 = 0) (h2 : ((p ^ 3 % 10) - (p ^ 2 % 10)) % 10 = 0) 
(h3 : (p + 4) % 10 = 0) : p % 10 = 6 :=
sorry

end units_digit_p_l225_225097


namespace ex3_solutions_abs_eq_l225_225303

theorem ex3_solutions_abs_eq (a : ℝ) : (∃ x1 x2 x3 x4 : ℝ, 
        2 * abs (abs (x1 - 1) - 3) = a ∧ 
        2 * abs (abs (x2 - 1) - 3) = a ∧ 
        2 * abs (abs (x3 - 1) - 3) = a ∧ 
        2 * abs (abs (x4 - 1) - 3) = a ∧ 
        x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ (x1 = x4 ∨ x2 = x4 ∨ x3 = x4)) ↔ a = 6 :=
by
    sorry

end ex3_solutions_abs_eq_l225_225303


namespace tan_identity_l225_225250

theorem tan_identity :
  tan (π / 9) * tan (2 * π / 9) * tan (4 * π / 9) = 3 :=
sorry

end tan_identity_l225_225250


namespace number_of_zero_points_l225_225629

theorem number_of_zero_points (f : ℝ → ℝ) (h_odd : ∀ x, f x = -f (-x)) (h_period : ∀ x, f (x - π) = f (x + π)) :
  ∃ (points : Finset ℝ), (∀ x ∈ points, 0 ≤ x ∧ x ≤ 8 ∧ f x = 0) ∧ points.card = 7 :=
by
  sorry

end number_of_zero_points_l225_225629


namespace locus_is_circle_l225_225167

open Complex

noncomputable def circle_center (a b : ℝ) : ℂ := Complex.ofReal (-a / (a^2 + b^2)) + Complex.I * (b / (a^2 + b^2))
noncomputable def circle_radius (a b : ℝ) : ℝ := 1 / Real.sqrt (a^2 + b^2)

theorem locus_is_circle (z0 z1 z : ℂ) (h1 : abs (z1 - z0) = abs z1) (h2 : z0 ≠ 0) (h3 : z1 * z = -1) :
  ∃ (a b : ℝ), z0 = Complex.ofReal a + Complex.I * b ∧
    (∃ c : ℂ, z = c ∧ 
      (c.re + a / (a^2 + b^2))^2 + (c.im - b / (a^2 + b^2))^2 = 1 / (a^2 + b^2)) := by
  sorry

end locus_is_circle_l225_225167


namespace find_angle_A_triangle_is_right_l225_225610

theorem find_angle_A (A : ℝ) (h : 2 * Real.cos (Real.pi + A) + Real.sin (Real.pi / 2 + 2 * A) + 3 / 2 = 0) :
  A = Real.pi / 3 := 
sorry

theorem triangle_is_right (a b c : ℝ) (A : ℝ) (ha : c - b = (Real.sqrt 3) / 3 * a) (hA : A = Real.pi / 3) :
  c^2 = a^2 + b^2 :=
sorry

end find_angle_A_triangle_is_right_l225_225610


namespace angle_E_measure_l225_225599

-- Definition of degrees for each angle in the quadrilateral
def angle_measure (E F G H : ℝ) : Prop :=
  E = 3 * F ∧ E = 4 * G ∧ E = 6 * H ∧ E + F + G + H = 360

-- Prove the measure of angle E
theorem angle_E_measure (E F G H : ℝ) (h : angle_measure E F G H) : E = 360 * (4 / 7) :=
by {
  sorry
}

end angle_E_measure_l225_225599


namespace isosceles_triangle_base_length_l225_225021

theorem isosceles_triangle_base_length (s a b : ℕ) (h1 : 3 * s = 45)
  (h2 : 2 * a + b = 40) (h3 : a = s) : b = 10 :=
by
  sorry

end isosceles_triangle_base_length_l225_225021


namespace power_function_not_pass_origin_l225_225451

noncomputable def does_not_pass_through_origin (m : ℝ) : Prop :=
  ∀ x:ℝ, (m^2 - 3 * m + 3) * x^(m^2 - m - 2) ≠ 0

theorem power_function_not_pass_origin (m : ℝ) :
  does_not_pass_through_origin m ↔ (m = 1 ∨ m = 2) :=
sorry

end power_function_not_pass_origin_l225_225451


namespace find_largest_negative_root_of_equation_l225_225978

theorem find_largest_negative_root_of_equation :
  ∃ x ∈ {x : ℝ | (sin (real.pi * x) - cos (2 * real.pi * x)) / ((sin (real.pi * x) - 1)^2 + cos (real.pi * x)^2 - 1) = 0}, 
  ∀ y ∈ {y : ℝ | (sin (real.pi * y) - cos (2 * real.pi * y)) / ((sin (real.pi * y) - 1)^2 + cos (real.pi * y)^2 - 1) = 0 },
  y < 0 → y ≤ x :=
begin
  use -0.5,
  split,
  { -- proof that -0.5 is a root
    sorry
  },
  { -- proof that -0.5 is the largest negative root
    sorry
  }
end

end find_largest_negative_root_of_equation_l225_225978


namespace Q_mul_P_plus_Q_eq_one_l225_225781

noncomputable def sqrt5_plus_2_pow (n : ℕ) :=
  (Real.sqrt 5 + 2)^(2 * n + 1)

noncomputable def P (n : ℕ) :=
  Int.floor (sqrt5_plus_2_pow n)

noncomputable def Q (n : ℕ) :=
  sqrt5_plus_2_pow n - P n

theorem Q_mul_P_plus_Q_eq_one (n : ℕ) : Q n * (P n + Q n) = 1 := by
  sorry

end Q_mul_P_plus_Q_eq_one_l225_225781


namespace tan_identity_l225_225248

theorem tan_identity :
  tan (π / 9) * tan (2 * π / 9) * tan (4 * π / 9) = 3 :=
sorry

end tan_identity_l225_225248


namespace percentage_increase_of_toad_bugs_eaten_l225_225605

theorem percentage_increase_of_toad_bugs_eaten (total_bugs gecko_bugs lizard_bugs frog_bugs toad_bugs : ℕ) :
  gecko_bugs = 12 →
  lizard_bugs = gecko_bugs / 2 →
  frog_bugs = 3 * lizard_bugs →
  total_bugs = 63 →
  toad_bugs = total_bugs - (gecko_bugs + lizard_bugs + frog_bugs) →
  ((toad_bugs - frog_bugs : ℤ) * 100) / frog_bugs = 50 :=
by
  sorry

end percentage_increase_of_toad_bugs_eaten_l225_225605


namespace sum_of_digits_is_2640_l225_225539

theorem sum_of_digits_is_2640 (x : ℕ) (h_cond : (1 + 3 + 4 + 6 + x) * (Nat.factorial 5) = 2640) : x = 8 := by
  sorry

end sum_of_digits_is_2640_l225_225539


namespace determine_C_cards_l225_225370

-- Define the card numbers
def card_numbers : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12]

-- Define the card sum each person should have
def card_sum := 26

-- Define person's cards
def A_cards : List ℕ := [10, 12]
def B_cards : List ℕ := [6, 11]

-- Define sum constraints for A and B
def sum_A := A_cards.sum
def sum_B := B_cards.sum

-- Define C's complete set of numbers based on remaining cards and sum constraints
def remaining_cards := card_numbers.diff (A_cards ++ B_cards)
def sum_remaining := remaining_cards.sum

theorem determine_C_cards :
  (sum_A + (26 - sum_A)) = card_sum ∧
  (sum_B + (26 - sum_B)) = card_sum ∧
  (sum_remaining = card_sum) → 
  (remaining_cards = [8, 9]) :=
by
  sorry

end determine_C_cards_l225_225370


namespace binom_10_2_eq_45_l225_225682

theorem binom_10_2_eq_45 :
  binom 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l225_225682


namespace least_integer_value_x_l225_225980

theorem least_integer_value_x (x : ℤ) : (3 * |2 * (x : ℤ) - 1| + 6 < 24) → x = -2 :=
by
  sorry

end least_integer_value_x_l225_225980


namespace most_likely_outcome_is_D_l225_225916

-- Define the basic probability of rolling any specific number with a fair die
def probability_of_specific_roll : ℚ := 1/6

-- Define the probability of each option
def P_A : ℚ := probability_of_specific_roll
def P_B : ℚ := 2 * probability_of_specific_roll
def P_C : ℚ := 3 * probability_of_specific_roll
def P_D : ℚ := 4 * probability_of_specific_roll

-- Define the proof problem statement
theorem most_likely_outcome_is_D : P_D = max P_A (max P_B (max P_C P_D)) :=
sorry

end most_likely_outcome_is_D_l225_225916


namespace sequence_bound_l225_225150

-- Definitions and assumptions based on the conditions
def valid_sequence (a : ℕ → ℕ) (N : ℕ) (m : ℕ) :=
  (1 ≤ a 1) ∧ (a m ≤ N) ∧ (∀ i j, 1 ≤ i → i < j → j ≤ m → a i < a j) ∧ 
  (∀ i j, 1 ≤ i → i < j → j ≤ m → Nat.lcm (a i) (a j) ≤ N)

-- The main theorem to prove
theorem sequence_bound (a : ℕ → ℕ) (N : ℕ) (m : ℕ) 
  (h : valid_sequence a N m) : m ≤ 2 * Nat.floor (Real.sqrt N) :=
sorry

end sequence_bound_l225_225150


namespace marble_counts_l225_225786

theorem marble_counts (A B C : ℕ) : 
  (∃ x : ℕ, 
    A = 165 ∧ 
    B = 57 ∧ 
    C = 21 ∧ 
    (A = 55 * x / 27) ∧ 
    (B = 19 * x / 27) ∧ 
    (C = 7 * x / 27) ∧ 
    (7 * x / 9 = x / 9 + 54) ∧ 
    (A + B + C) = 3 * x
  ) :=
sorry

end marble_counts_l225_225786


namespace hunter_saw_32_frogs_l225_225880

noncomputable def total_frogs (g1 : ℕ) (g2 : ℕ) (d : ℕ) : ℕ :=
g1 + g2 + d

theorem hunter_saw_32_frogs :
  total_frogs 5 3 (2 * 12) = 32 := by
  sorry

end hunter_saw_32_frogs_l225_225880


namespace exists_two_same_remainder_l225_225358

theorem exists_two_same_remainder (n : ℤ) (a : ℕ → ℤ) :
  ∃ i j : ℕ, i ≠ j ∧ 0 ≤ i ∧ i ≤ n ∧ 0 ≤ j ∧ j ≤ n ∧ (a i % n = a j % n) := sorry

end exists_two_same_remainder_l225_225358


namespace number_of_pictures_deleted_l225_225381

-- Definitions based on the conditions
def total_files_deleted : ℕ := 17
def songs_deleted : ℕ := 8
def text_files_deleted : ℕ := 7

-- The question rewritten as a Lean theorem statement
theorem number_of_pictures_deleted : 
  (total_files_deleted - songs_deleted - text_files_deleted) = 2 := 
by
  sorry

end number_of_pictures_deleted_l225_225381


namespace part1_part2_l225_225135

theorem part1 (A B C a b c : ℝ) (h1 : 3 * a * Real.cos A = Real.sqrt 6 * (c * Real.cos B + b * Real.cos C)) :
    Real.tan (2 * A) = 2 * Real.sqrt 2 := sorry

theorem part2 (A B C a b c S : ℝ) 
  (h_sin_B : Real.sin (Real.pi / 2 + B) = 2 * Real.sqrt 2 / 3)
  (hc : c = 2 * Real.sqrt 2) :
    S = 2 * Real.sqrt 2 / 3 := sorry

end part1_part2_l225_225135


namespace midpoint_C_l225_225355

variables (A B C : ℝ × ℝ)
variables (x1 y1 x2 y2 : ℝ)
variables (AC CB : ℝ)

def segment_division (A B C : ℝ × ℝ) (m n : ℝ) : Prop :=
  C = ((m * B.1 + n * A.1) / (m + n), (m * B.2 + n * A.2) / (m + n))

theorem midpoint_C :
  A = (-2, 1) →
  B = (4, 9) →
  AC = 2 * CB →
  segment_division A B C 2 1 →
  C = (2, 19 / 3) :=
by
  sorry

end midpoint_C_l225_225355


namespace tom_tickets_left_l225_225836

-- Define the conditions
def tickets_whack_a_mole : ℕ := 32
def tickets_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

-- Define what we need to prove
theorem tom_tickets_left : tickets_whack_a_mole + tickets_skee_ball - tickets_spent_on_hat = 50 :=
by sorry

end tom_tickets_left_l225_225836


namespace cost_price_of_article_l225_225404

theorem cost_price_of_article (C MP : ℝ) (h1 : 0.90 * MP = 1.25 * C) (h2 : 1.25 * C = 65.97) : C = 52.776 :=
by
  sorry

end cost_price_of_article_l225_225404


namespace base_8_to_decimal_77_eq_63_l225_225192

-- Define the problem in Lean 4
theorem base_8_to_decimal_77_eq_63 (k a1 a2 : ℕ) (h_k : k = 8) (h_a1 : a1 = 7) (h_a2 : a2 = 7) :
    a2 * k^1 + a1 * k^0 = 63 := 
by
  -- Placeholder for proof
  sorry

end base_8_to_decimal_77_eq_63_l225_225192


namespace max_value_y_l225_225981

open Real

noncomputable def y (x : ℝ) : ℝ :=
  tan (x + π / 4) - tan (x + π / 3) + sin (x + π / 3)

theorem max_value_y :
  ∃ x, -π / 2 ≤ x ∧ x ≤ -π / 4 ∧ y x = -sqrt 3 + 1 - sqrt 2 / 2 :=
by
  sorry

end max_value_y_l225_225981


namespace derivative_at_2_l225_225306

noncomputable def f (x : ℝ) : ℝ := (1 - x) / x + Real.log x

theorem derivative_at_2 : (deriv f 2) = 1 / 4 :=
by 
  sorry

end derivative_at_2_l225_225306


namespace neither_sufficient_nor_necessary_l225_225340

theorem neither_sufficient_nor_necessary (a b : ℝ) : ¬ ((a + b > 0) ↔ (ab > 0)) := 
sorry

end neither_sufficient_nor_necessary_l225_225340


namespace quadratic_equation_solution_unique_l225_225525

noncomputable def b_solution := (-3 + 3 * Real.sqrt 21) / 2
noncomputable def c_solution := (33 - 3 * Real.sqrt 21) / 2

theorem quadratic_equation_solution_unique :
  (∃ (b c : ℝ), 
     (∀ (x : ℝ), 3 * x^2 + b * x + c = 0 → x = b_solution) ∧ 
     b + c = 15 ∧ 3 * c = b^2 ∧
     b = b_solution ∧ c = c_solution) :=
by { sorry }

end quadratic_equation_solution_unique_l225_225525


namespace sum_of_x_coords_Q3_l225_225502

-- Definitions
def Q1_vertices_sum_x (S : ℝ) := S = 1050

def Q2_vertices_sum_x (S' : ℝ) (S : ℝ) := S' = S

def Q3_vertices_sum_x (S'' : ℝ) (S' : ℝ) := S'' = S'

-- Lean 4 statement
theorem sum_of_x_coords_Q3 (S : ℝ) (S' : ℝ) (S'' : ℝ) :
  Q1_vertices_sum_x S →
  Q2_vertices_sum_x S' S →
  Q3_vertices_sum_x S'' S' →
  S'' = 1050 :=
by
  sorry

end sum_of_x_coords_Q3_l225_225502


namespace tangent_product_eq_three_l225_225263

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end tangent_product_eq_three_l225_225263


namespace egg_rolls_total_l225_225767

def total_egg_rolls (omar_rolls : ℕ) (karen_rolls : ℕ) : ℕ :=
  omar_rolls + karen_rolls

theorem egg_rolls_total :
  total_egg_rolls 219 229 = 448 :=
by
  sorry

end egg_rolls_total_l225_225767


namespace total_chickens_after_purchase_l225_225839

def initial_chickens : ℕ := 400
def percentage_died : ℕ := 40
def times_to_buy : ℕ := 10

noncomputable def chickens_died : ℕ := (percentage_died * initial_chickens) / 100
noncomputable def chickens_remaining : ℕ := initial_chickens - chickens_died
noncomputable def chickens_bought : ℕ := times_to_buy * chickens_died
noncomputable def total_chickens : ℕ := chickens_remaining + chickens_bought

theorem total_chickens_after_purchase : total_chickens = 1840 :=
by
  sorry

end total_chickens_after_purchase_l225_225839


namespace seventh_graders_trip_count_l225_225775

theorem seventh_graders_trip_count (fifth_graders sixth_graders teachers_per_grade parents_per_grade grades buses seats_per_bus : ℕ) 
  (hf : fifth_graders = 109) 
  (hs : sixth_graders = 115)
  (ht : teachers_per_grade = 4) 
  (hp : parents_per_grade = 2) 
  (hg : grades = 3) 
  (hb : buses = 5)
  (hsb : seats_per_bus = 72) : 
  ∃ seventh_graders : ℕ, seventh_graders = 118 := 
by
  sorry

end seventh_graders_trip_count_l225_225775


namespace laptop_sticker_price_l225_225139

theorem laptop_sticker_price (x : ℝ) (h₁ : 0.70 * x = 0.80 * x - 50 - 30) : x = 800 := 
  sorry

end laptop_sticker_price_l225_225139


namespace balance_squares_circles_l225_225776

theorem balance_squares_circles (x y z : ℕ) (h1 : 5 * x + 2 * y = 21 * z) (h2 : 2 * x = y + 3 * z) : 
  3 * y = 9 * z :=
by 
  sorry

end balance_squares_circles_l225_225776


namespace lily_milk_quantity_l225_225345

theorem lily_milk_quantity :
  let init_gallons := (5 : ℝ)
  let given_away := (18 / 4 : ℝ)
  let received_back := (7 / 4 : ℝ)
  init_gallons - given_away + received_back = 2 + 1 / 4 :=
by
  sorry

end lily_milk_quantity_l225_225345


namespace alex_ride_time_l225_225518

theorem alex_ride_time
  (T : ℝ) -- time on flat ground
  (flat_speed : ℝ := 20) -- flat ground speed
  (uphill_speed : ℝ := 12) -- uphill speed
  (uphill_time : ℝ := 2.5) -- uphill time
  (downhill_speed : ℝ := 24) -- downhill speed
  (downhill_time : ℝ := 1.5) -- downhill time
  (walk_distance : ℝ := 8) -- distance walked
  (total_distance : ℝ := 164) -- total distance to the town
  (hup : uphill_speed * uphill_time = 30)
  (hdown : downhill_speed * downhill_time = 36)
  (hwalk : walk_distance = 8) :
  flat_speed * T + 30 + 36 + 8 = total_distance → T = 4.5 :=
by
  intros h
  sorry

end alex_ride_time_l225_225518


namespace pascal_elements_sum_l225_225377

theorem pascal_elements_sum :
  (Nat.choose 20 4 + Nat.choose 20 5) = 20349 :=
by
  sorry

end pascal_elements_sum_l225_225377


namespace number_of_ways_pairs_l225_225294

theorem number_of_ways_pairs (n : ℕ) (hc1 : n = 4) :
  let total_ways := (nat.choose 8 4),
      non_paired_ways := 16,
      paired_ways := total_ways - non_paired_ways in
  paired_ways = 54 :=
by
  sorry

end number_of_ways_pairs_l225_225294


namespace sum_of_first_eight_terms_l225_225086

-- Define the first term, common ratio, and the number of terms
def a : ℚ := 1 / 3
def r : ℚ := 1 / 3
def n : ℕ := 8

-- Sum of the first n terms of a geometric sequence
def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- Proof statement
theorem sum_of_first_eight_terms : geometric_sum a r n = 3280 / 6561 :=
by
  sorry

end sum_of_first_eight_terms_l225_225086


namespace compute_fraction_l225_225448

def x : ℚ := 2 / 3
def y : ℚ := 3 / 2
def z : ℚ := 1 / 3

theorem compute_fraction :
  (1 / 3) * x^7 * y^5 * z^4 = 11 / 600 :=
by
  sorry

end compute_fraction_l225_225448


namespace tan_product_l225_225256

noncomputable def tan_pi_nine := Real.tan (Real.pi / 9)
noncomputable def tan_two_pi_nine := Real.tan (2 * Real.pi / 9)
noncomputable def tan_four_pi_nine := Real.tan (4 * Real.pi / 9)

-- Key polynomial condition
def poly := λ t : ℝ, t^8 - 36 * t^6 + 126 * t^4 - 84 * t^2 + 9

-- Condition: the polynomial has roots tan(π/9), tan(2π/9), tan(4π/9)
axiom poly_roots: 
  (poly tan_pi_nine = 0 ∧
   poly tan_two_pi_nine = 0 ∧
   poly tan_four_pi_nine = 0)

-- The proof statement
theorem tan_product : 
  tan_pi_nine * 
  tan_two_pi_nine * 
  tan_four_pi_nine = 3 :=
sorry

end tan_product_l225_225256


namespace number_of_men_in_second_group_l225_225656

variable (n m : ℕ)

theorem number_of_men_in_second_group 
  (h1 : 42 * 18 = n)
  (h2 : n = m * 28) : 
  m = 27 := by
  sorry

end number_of_men_in_second_group_l225_225656


namespace work_days_B_l225_225499

theorem work_days_B (A B: ℕ) (work_per_day_B: ℕ) (total_days : ℕ) (total_units : ℕ) :
  (A = 2 * B) → (work_per_day_B = 1) → (total_days = 36) → (B = 1) → (total_units = total_days * (A + B)) → 
  total_units / work_per_day_B = 108 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end work_days_B_l225_225499


namespace evaluate_expression_l225_225861

theorem evaluate_expression (a : ℝ) (h : 2 * a^2 - 3 * a - 5 = 0) : 4 * a^4 - 12 * a^3 + 9 * a^2 - 10 = 15 :=
by
  sorry

end evaluate_expression_l225_225861


namespace quadratic_has_two_real_roots_find_m_for_roots_difference_4_l225_225307

-- Define the function representing the quadratic equation
def quadratic_eq (m x : ℝ) := x^2 + (2 - m) * x + 1 - m

-- Part 1
theorem quadratic_has_two_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 :=
sorry

-- Part 2
theorem find_m_for_roots_difference_4 (m : ℝ) (H : m < 0) :
  (∃ (x1 x2 : ℝ), quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 ∧ x1 - x2 = 4) → m = -4 :=
sorry

end quadratic_has_two_real_roots_find_m_for_roots_difference_4_l225_225307


namespace find_matrix_M_l225_225040
    
open Matrix

-- Given definitions and assumptions
variables {M : Matrix (Fin 2) (Fin 2) ℝ}
variables {e : Fin 2 → ℝ} {v : Fin 2 → ℝ} {w : Fin 2 → ℝ}

def eigenvector_and_value_of_M (eigenval : ℝ) : Prop :=
  M.mul_vec e = eigenval • e

def transformed_point (point : Fin 2 → ℝ) (result : Fin 2 → ℝ) : Prop :=
  M.mul_vec point = result

-- Vectors given in the conditions
def e : Fin 2 → ℝ := ![1, 1]
def point : Fin 2 → ℝ := ![-1, 2]
def result : Fin 2 → ℝ := ![9, 15]

-- Proof that M satisfies the given conditions
theorem find_matrix_M (M : Matrix (Fin 2) (Fin 2) ℝ) :
  eigenvector_and_value_of_M 3 ∧ transformed_point point result →
  M = !![-1, 4, -3, 6] :=
begin
  sorry
end

end find_matrix_M_l225_225040


namespace wire_problem_l225_225058

theorem wire_problem (a b : ℝ) (h_perimeter : a = b) : a / b = 1 := by
  sorry

end wire_problem_l225_225058


namespace max_value_on_interval_l225_225092

noncomputable def f (x : ℝ) := 2 * x ^ 3 - 6 * x ^ 2 + 10

theorem max_value_on_interval :
  (∀ x ∈ Set.Icc (1 : ℝ) 3, f 2 <= f x) → 
  ∃ y ∈ Set.Icc (1 : ℝ) 3, ∀ z ∈ Set.Icc (1 : ℝ) 3, f y >= f z :=
by
  sorry

end max_value_on_interval_l225_225092


namespace kerosene_cost_l225_225741

theorem kerosene_cost (A B C : ℝ)
  (h1 : A = B)
  (h2 : C = A / 2)
  (h3 : C * 2 = 24 / 100) :
  24 = 24 := 
sorry

end kerosene_cost_l225_225741


namespace simplify_fractions_sum_l225_225166

theorem simplify_fractions_sum :
  (48 / 72) + (30 / 45) = 4 / 3 := 
by
  sorry

end simplify_fractions_sum_l225_225166


namespace time_needed_to_gather_remaining_flowers_l225_225148

-- conditions
def classmates : ℕ := 30
def time_per_flower : ℕ := 10
def gathering_time : ℕ := 2 * 60
def lost_flowers : ℕ := 3

-- question and proof goal
theorem time_needed_to_gather_remaining_flowers : 
  let flowers_needed := classmates - ((gathering_time / time_per_flower) - lost_flowers)
  flowers_needed * time_per_flower = 210 :=
by
  sorry

end time_needed_to_gather_remaining_flowers_l225_225148


namespace largest_divisor_consecutive_odd_squares_l225_225463

theorem largest_divisor_consecutive_odd_squares (m n : ℤ) 
  (hmn : m = n + 2) 
  (hodd_m : m % 2 = 1) 
  (hodd_n : n % 2 = 1) 
  (horder : n < m) : ∃ k : ℤ, m^2 - n^2 = 8 * k :=
by 
  sorry

end largest_divisor_consecutive_odd_squares_l225_225463


namespace area_of_farm_l225_225216

theorem area_of_farm (W L : ℝ) (hW : W = 30) 
  (hL_fence_cost : 14 * (L + W + Real.sqrt (L^2 + W^2)) = 1680) : 
  W * L = 1200 :=
by
  sorry -- Proof not required

end area_of_farm_l225_225216


namespace quadratic_real_roots_condition_l225_225312

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 + x + m = 0) → m ≤ 1/4 :=
by
  sorry

end quadratic_real_roots_condition_l225_225312


namespace measuring_cup_size_l225_225003

-- Defining the conditions
def total_flour := 8
def flour_needed := 6
def scoops_removed := 8 

-- Defining the size of the cup
def cup_size (x : ℚ) := 8 - scoops_removed * x = flour_needed

-- Stating the theorem
theorem measuring_cup_size : ∃ x : ℚ, cup_size x ∧ x = 1 / 4 :=
by {
    sorry
}

end measuring_cup_size_l225_225003


namespace order_of_x_y_z_l225_225891

theorem order_of_x_y_z (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  let x : ℝ := (a + b) * (c + d)
  let y : ℝ := (a + c) * (b + d)
  let z : ℝ := (a + d) * (b + c)
  x < y ∧ y < z :=
by
  let x : ℝ := (a + b) * (c + d)
  let y : ℝ := (a + c) * (b + d)
  let z : ℝ := (a + d) * (b + c)
  sorry

end order_of_x_y_z_l225_225891


namespace marys_total_cards_l225_225158

def initial_cards : ℕ := 18
def torn_cards : ℕ := 8
def cards_from_fred : ℕ := 26
def cards_bought_by_mary : ℕ := 40

theorem marys_total_cards :
  initial_cards - torn_cards + cards_from_fred + cards_bought_by_mary = 76 :=
by
  sorry

end marys_total_cards_l225_225158


namespace power_inequality_l225_225865

variable (a b c : ℝ)

theorem power_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a^3 + b^3 + c^3) ≥ a * b^2 + a^2 * b + b * c^2 + b^2 * c + a * c^2 + a^2 * c :=
by sorry

end power_inequality_l225_225865


namespace find_root_interval_l225_225031

noncomputable def f : ℝ → ℝ := sorry

theorem find_root_interval :
  f 2 < 0 ∧ f 3 > 0 ∧ f 2.5 < 0 ∧ f 2.75 > 0 ∧ f 2.625 > 0 ∧ f 2.5625 > 0 →
  ∃ x, 2.5 < x ∧ x < 2.5625 ∧ f x = 0 := sorry

end find_root_interval_l225_225031


namespace area_of_triangle_bounded_by_line_and_axes_l225_225831

theorem area_of_triangle_bounded_by_line_and_axes (x y : ℝ) (hx : 3 * x + 2 * y = 12) :
  ∃ (area : ℝ), area = 12 := by
sorry

end area_of_triangle_bounded_by_line_and_axes_l225_225831


namespace total_money_found_l225_225847

-- Define the conditions
def donna_share := 0.40
def friendA_share := 0.35
def friendB_share := 0.25
def donna_amount := 39.0

-- Define the problem statement/proof
theorem total_money_found (donna_share friendA_share friendB_share donna_amount : ℝ) 
  (h1 : donna_share = 0.40) 
  (h2 : friendA_share = 0.35) 
  (h3 : friendB_share = 0.25) 
  (h4 : donna_amount = 39.0) :
  ∃ total_money : ℝ, total_money = 97.50 := 
by
  -- The calculations and actual proof will go here
  sorry

end total_money_found_l225_225847


namespace find_sin_cos_sum_find_beta_value_l225_225862

open Real

-- Define α in the interval (0, π/2) and with cos(2α) = 4/5
variable (α β : ℝ)
variable (h1 : α ∈ set.Ioo 0 (π / 2))
variable (h2 : cos (2 * α) = 4 / 5)

-- Define β in the interval (π/2, π) and with 5sin(2α + β) = sin(β)
variable (h3 : β ∈ set.Ioo (π / 2) π)
variable (h4 : 5 * sin (2 * α + β) = sin β)

-- Statement proving the required results
theorem find_sin_cos_sum : sin α + cos α = 2 * sqrt 10 / 5 := by sorry

theorem find_beta_value : β = 3 * π / 4 := by sorry

end find_sin_cos_sum_find_beta_value_l225_225862


namespace range_of_n_l225_225437

theorem range_of_n (n : ℝ) (x : ℝ) (h1 : 180 - n > 0) (h2 : ∀ x, 180 - n != x ∧ 180 - n != x + 24 → 180 - n + x + x + 24 = 180 → 44 ≤ x ∧ x ≤ 52 → 112 ≤ n ∧ n ≤ 128)
  (h3 : ∀ n, 180 - n = max (180 - n) (180 - n) - 24 ∧ min (180 - n) (180 - n) = n - 24 → 104 ≤ n ∧ n ≤ 112)
  (h4 : ∀ n, 180 - n = min (180 - n) (180 - n) ∧ max (180 - n) (180 - n) = 180 - n + 24 → 128 ≤ n ∧ n ≤ 136) :
  104 ≤ n ∧ n ≤ 136 :=
by sorry

end range_of_n_l225_225437


namespace rocky_miles_total_l225_225744

-- Defining the conditions
def m1 : ℕ := 4
def m2 : ℕ := 2 * m1
def m3 : ℕ := 3 * m2

-- The statement to be proven
theorem rocky_miles_total : m1 + m2 + m3 = 36 := by
  sorry

end rocky_miles_total_l225_225744


namespace percentage_increase_is_20_l225_225383

noncomputable def total_stocks : ℕ := 1980
noncomputable def stocks_higher : ℕ := 1080
noncomputable def stocks_lower : ℕ := total_stocks - stocks_higher

/--
Given that the total number of stocks is 1,980, and 1,080 stocks closed at a higher price today than yesterday.
Furthermore, the number of stocks that closed higher today is greater than the number that closed lower.

Prove that the percentage increase in the number of stocks that closed at a higher price today compared to the number that closed at a lower price is 20%.
-/
theorem percentage_increase_is_20 :
  (stocks_higher - stocks_lower) / stocks_lower * 100 = 20 := by
  sorry

end percentage_increase_is_20_l225_225383


namespace min_value_expr_l225_225854

noncomputable def expr (x : ℝ) : ℝ := (Real.sin x)^8 + (Real.cos x)^8 + 3 / (Real.sin x)^6 + (Real.cos x)^6 + 3

theorem min_value_expr : ∃ x : ℝ, expr x = 14 / 31 := 
by
  sorry

end min_value_expr_l225_225854


namespace fraction_problem_l225_225887

theorem fraction_problem
    (q r s u : ℚ)
    (h1 : q / r = 8)
    (h2 : s / r = 4)
    (h3 : s / u = 1 / 3) :
    u / q = 3 / 2 :=
  sorry

end fraction_problem_l225_225887


namespace percentage_markup_on_cost_price_l225_225172

theorem percentage_markup_on_cost_price 
  (SP : ℝ) (CP : ℝ) (hSP : SP = 6400) (hCP : CP = 5565.217391304348) : 
  ((SP - CP) / CP) * 100 = 15 :=
by
  -- proof would go here
  sorry

end percentage_markup_on_cost_price_l225_225172


namespace sum_of_digits_0_to_2012_l225_225331

def sum_of_digits (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

def sum_of_digits_in_range (a b : Nat) : Nat :=
  ((List.range (b + 1)).drop a).map sum_of_digits |>.sum

theorem sum_of_digits_0_to_2012 : 
  sum_of_digits_in_range 0 2012 = 28077 := 
by
  sorry

end sum_of_digits_0_to_2012_l225_225331


namespace range_of_a_l225_225302

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2)*x^2 + (2 * a - 1) * x + 6 > 0 ↔ x = 3 → true) ∧
  (∀ x : ℝ, (a - 2)*x^2 + (2 * a - 1) * x + 6 > 0 ↔ x = 5 → false) →
  1 < a ∧ a ≤ 7 / 5 :=
by
  sorry

end range_of_a_l225_225302


namespace find_x_l225_225228

-- Define the conditions
def cherryGum := 25
def grapeGum := 35
def packs (x : ℚ) := x -- Each pack contains exactly x pieces of gum

-- Define the ratios after losing one pack of cherry gum and finding 6 packs of grape gum
def ratioAfterLosingCherryPack (x : ℚ) := (cherryGum - packs x) / grapeGum
def ratioAfterFindingGrapePacks (x : ℚ) := cherryGum / (grapeGum + 6 * packs x)

-- State the theorem to be proved
theorem find_x (x : ℚ) (h : ratioAfterLosingCherryPack x = ratioAfterFindingGrapePacks x) : x = 115 / 6 :=
by
  sorry

end find_x_l225_225228


namespace find_B_and_area_l225_225607

variable (A B C a b c d : ℝ)

-- Given conditions
axiom cond1 : (a + c) * Real.sin A = Real.sin A + Real.sin C
axiom cond2 : c^2 + c = b^2 - 1
axiom midpoint_D : D = (A + C) / 2
axiom BD_val : BD = sqrt 3 / 2

-- To Prove
theorem find_B_and_area : 
  ∃ B : ℝ, 
    (B = 2 * π / 3) ∧ 
    let area := (1 / 2) * a * c * Real.sin B in 
    area = sqrt 3 / 2 :=
by
  sorry

end find_B_and_area_l225_225607


namespace triangle_area_l225_225825

theorem triangle_area : 
  ∃ (A : ℝ), A = 12 ∧ (∃ (x_intercept y_intercept : ℝ), 3 * x_intercept + 2 * y_intercept = 12 ∧ x_intercept * y_intercept / 2 = A) :=
by
  sorry

end triangle_area_l225_225825


namespace tan_product_eq_three_l225_225233

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end tan_product_eq_three_l225_225233


namespace total_practice_hours_l225_225348

-- Definitions based on conditions
def weekday_practice_hours : ℕ := 3
def saturday_practice_hours : ℕ := 5
def weekdays_per_week : ℕ := 5
def weeks_until_game : ℕ := 3

-- Theorem statement
theorem total_practice_hours : (weekday_practice_hours * weekdays_per_week + saturday_practice_hours) * weeks_until_game = 60 := 
by sorry

end total_practice_hours_l225_225348


namespace mutually_exclusive_not_complementary_l225_225206

namespace BallDrawingProblem

def bag : set ℕ := { red := 2, black := 2 }

def draw (bag : set ℕ) : set ℕ := by
  sorry -- Define the set of possible outcomes when drawing two balls.

def event_one_black (outcome : set ℕ) : Prop :=
  outcome.count (λ b, b = black) = 1

def event_two_black (outcome : set ℕ) : Prop :=
  outcome.count (λ b, b = black) = 2

theorem mutually_exclusive_not_complementary :
  ∀ outcome : set ℕ, event_one_black outcome → ¬ event_two_black outcome :=
by
  sorry -- The actual proof is omitted
end BallDrawingProblem

end mutually_exclusive_not_complementary_l225_225206


namespace solution_per_beaker_l225_225642

theorem solution_per_beaker (solution_per_tube : ℕ) (num_tubes : ℕ) (num_beakers : ℕ)
    (h1 : solution_per_tube = 7) (h2 : num_tubes = 6) (h3 : num_beakers = 3) :
    (solution_per_tube * num_tubes) / num_beakers = 14 :=
by
  sorry

end solution_per_beaker_l225_225642


namespace valid_subsets_12_even_subsets_305_l225_225875

def valid_subsets_count(n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 4
  else
    valid_subsets_count (n - 1) +
    valid_subsets_count (n - 2) +
    valid_subsets_count (n - 3)
    -- Recurrence relation for valid subsets which satisfy the conditions

theorem valid_subsets_12 : valid_subsets_count 12 = 610 :=
  by sorry
  -- We need to verify recurrence and compute for n = 12 (optional step if just computing, not proving the sequence.)

theorem even_subsets_305 :
  (valid_subsets_count 12) / 2 = 305 :=
  by sorry
  -- Concludes that half the valid subsets for n = 12 are even-sized sets.

end valid_subsets_12_even_subsets_305_l225_225875


namespace div_by_6_for_all_k_l225_225339

def b_n_sum_of_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem div_by_6_for_all_k : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 50 → (b_n_sum_of_squares k) % 6 = 0 :=
by
  intros k hk
  sorry

end div_by_6_for_all_k_l225_225339


namespace binom_10_2_eq_45_l225_225677

theorem binom_10_2_eq_45 : Nat.binomial 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l225_225677


namespace merchant_gross_profit_l225_225944

noncomputable def purchase_price : ℝ := 48
noncomputable def markup_rate : ℝ := 0.40
noncomputable def discount_rate : ℝ := 0.20

theorem merchant_gross_profit :
  ∃ S : ℝ, S = purchase_price + markup_rate * S ∧ 
  ((S - discount_rate * S) - purchase_price = 16) :=
by
  sorry

end merchant_gross_profit_l225_225944


namespace similar_right_triangle_hypotenuse_length_l225_225056

theorem similar_right_triangle_hypotenuse_length :
  ∀ (a b c d : ℝ), a = 15 → c = 39 → d = 45 → 
  (b^2 = c^2 - a^2) → 
  ∃ e : ℝ, e = (c * (d / b)) ∧ e = 48.75 :=
by
  intros a b c d ha hc hd hb
  sorry

end similar_right_triangle_hypotenuse_length_l225_225056


namespace tan_product_identity_l225_225278

noncomputable def tan_pi_over_n (n : ℕ) : ℝ := Real.tan (Real.pi / n)

theorem tan_product_identity :
  tan_pi_over_n 9 * tan_pi_over_n 4.5 * tan_pi_over_n 2.25 = Real.sqrt 3 := 
sorry

end tan_product_identity_l225_225278


namespace last_two_digits_sum_eq_13_l225_225801

def is_contributing (n : ℕ) : Prop :=
  ¬ ((n % 3 = 0) ∧ (n % 5 = 0))

def last_two_digits (n : ℕ) := n % 100

def sum_of_contributing_factorials : ℕ :=
  (Finset.range 101).filter is_contributing
    .sum (λ n => last_two_digits (Nat.factorial n))

theorem last_two_digits_sum_eq_13 : last_two_digits sum_of_contributing_factorials = 13 :=
  sorry

end last_two_digits_sum_eq_13_l225_225801


namespace Donny_change_l225_225422

/-- The change Donny will receive after filling up his truck. -/
theorem Donny_change
  (capacity : ℝ)
  (initial_fuel : ℝ)
  (cost_per_liter : ℝ)
  (money_available : ℝ)
  (change : ℝ) :
  capacity = 150 →
  initial_fuel = 38 →
  cost_per_liter = 3 →
  money_available = 350 →
  change = money_available - cost_per_liter * (capacity - initial_fuel) →
  change = 14 :=
by
  intros h_capacity h_initial_fuel h_cost_per_liter h_money_available h_change
  rw [h_capacity, h_initial_fuel, h_cost_per_liter, h_money_available] at h_change
  sorry

end Donny_change_l225_225422


namespace a_eq_b_pow_n_l225_225707

theorem a_eq_b_pow_n (a b n : ℕ) (h : ∀ k : ℕ, k ≠ b → (a - k^n) % (b - k) = 0) : a = b^n :=
sorry

end a_eq_b_pow_n_l225_225707


namespace cody_steps_away_from_goal_l225_225963

def steps_in_week (daily_steps : ℕ) : ℕ :=
  daily_steps * 7

def total_steps_in_4_weeks (initial_steps : ℕ) : ℕ :=
  steps_in_week initial_steps +
  steps_in_week (initial_steps + 1000) +
  steps_in_week (initial_steps + 2000) +
  steps_in_week (initial_steps + 3000)

theorem cody_steps_away_from_goal :
  let goal := 100000
  let initial_daily_steps := 1000
  let total_steps := total_steps_in_4_weeks initial_daily_steps
  goal - total_steps = 30000 :=
by
  sorry

end cody_steps_away_from_goal_l225_225963


namespace average_increase_l225_225052

theorem average_increase (A A' : ℕ) (runs_in_17th : ℕ) (total_innings : ℕ) (new_avg : ℕ) 
(h1 : total_innings = 17)
(h2 : runs_in_17th = 87)
(h3 : new_avg = 39)
(h4 : A' = new_avg)
(h5 : 16 * A + runs_in_17th = total_innings * new_avg) 
: A' - A = 3 := by
  sorry

end average_increase_l225_225052


namespace same_leading_digit_l225_225537

theorem same_leading_digit (n : ℕ) (hn : 0 < n) : 
  (∀ a k l : ℕ, (a * 10^k < 2^n ∧ 2^n < (a+1) * 10^k) ∧ (a * 10^l < 5^n ∧ 5^n < (a+1) * 10^l) → a = 3) := 
sorry

end same_leading_digit_l225_225537


namespace calculation_proof_l225_225674

theorem calculation_proof :
  5^(Real.log 9 / Real.log 5) + (1 / 2) * (Real.log 32 / Real.log 2) - Real.log (Real.log 8 / Real.log 2) / Real.log 3 = 21 / 2 := 
  sorry

end calculation_proof_l225_225674


namespace smallest_abs_value_l225_225955

theorem smallest_abs_value : 
    ∀ (a b c d : ℝ), 
    a = -1/2 → b = -2/3 → c = 4 → d = -5 → 
    abs a < abs b ∧ abs a < abs c ∧ abs a < abs d := 
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  simp
  -- Proof omitted for brevity
  sorry

end smallest_abs_value_l225_225955


namespace students_in_class_l225_225895

theorem students_in_class (x : ℕ) (S : ℕ)
  (h1 : S = 3 * (S / x) + 24)
  (h2 : S = 4 * (S / x) - 26) : 3 * x + 24 = 4 * x - 26 :=
by
  sorry

end students_in_class_l225_225895


namespace each_child_gets_one_slice_l225_225943

-- Define the conditions
def couple_slices_per_person : ℕ := 3
def number_of_people : ℕ := 2
def number_of_children : ℕ := 6
def pizzas_ordered : ℕ := 3
def slices_per_pizza : ℕ := 4

-- Calculate slices required by the couple
def total_slices_for_couple : ℕ := couple_slices_per_person * number_of_people

-- Calculate total slices available
def total_slices : ℕ := pizzas_ordered * slices_per_pizza

-- Calculate slices for children
def slices_for_children : ℕ := total_slices - total_slices_for_couple

-- Calculate slices each child gets
def slices_per_child : ℕ := slices_for_children / number_of_children

-- The proof statement
theorem each_child_gets_one_slice : slices_per_child = 1 := by
  sorry

end each_child_gets_one_slice_l225_225943


namespace prove_divisibility_by_polynomial_l225_225858

theorem prove_divisibility_by_polynomial {a b c : ℤ} :
  ∀ n : ℕ, (n = 4) ↔ (a^n * (b - c) + b^n * (c - a) + c^n * (a - b)) ∣ (a^2 + b^2 + c^2 + a*b + b*c + c*a) :=
by sorry

end prove_divisibility_by_polynomial_l225_225858


namespace dan_stationery_spent_l225_225417

def total_spent : ℕ := 32
def backpack_cost : ℕ := 15
def notebook_cost : ℕ := 3
def number_of_notebooks : ℕ := 5
def stationery_cost_each : ℕ := 1

theorem dan_stationery_spent : 
  (total_spent - (backpack_cost + notebook_cost * number_of_notebooks)) = 2 :=
by
  sorry

end dan_stationery_spent_l225_225417


namespace area_and_cost_of_path_l225_225384

-- Define the dimensions of the grass field
def length_field : ℝ := 85
def width_field : ℝ := 55

-- Define the width of the path around the field
def width_path : ℝ := 2.5

-- Define the cost per square meter of constructing the path
def cost_per_sqm : ℝ := 2

-- Define new dimensions including the path
def new_length : ℝ := length_field + 2 * width_path
def new_width : ℝ := width_field + 2 * width_path

-- Define the area of the entire field including the path
def area_with_path : ℝ := new_length * new_width

-- Define the area of the grass field without the path
def area_field : ℝ := length_field * width_field

-- Define the area of the path alone
def area_path : ℝ := area_with_path - area_field

-- Define the cost of constructing the path
def cost_constructing_path : ℝ := area_path * cost_per_sqm

-- Theorem to prove the area of the path and cost of constructing it
theorem area_and_cost_of_path :
  area_path = 725 ∧ cost_constructing_path = 1450 :=
by
  -- Skipping the proof as instructed
  sorry

end area_and_cost_of_path_l225_225384


namespace number_of_boys_l225_225389

-- Definitions reflecting the conditions
def total_students := 1200
def sample_size := 200
def extra_boys := 10

-- Main problem statement
theorem number_of_boys (B G b g : ℕ) 
  (h_total_students : B + G = total_students)
  (h_sample_size : b + g = sample_size)
  (h_extra_boys : b = g + extra_boys)
  (h_stratified : b * G = g * B) :
  B = 660 :=
by sorry

end number_of_boys_l225_225389


namespace find_A_l225_225055

theorem find_A (A : ℕ) (B : ℕ) (h₁ : 0 ≤ B ∧ B ≤ 999) (h₂ : 1000 * A + B = A * (A + 1) / 2) : A = 1999 :=
  sorry

end find_A_l225_225055


namespace max_distance_travel_l225_225187

-- Each car can carry at most 24 barrels of gasoline
def max_gasoline_barrels : ℕ := 24

-- Each barrel allows a car to travel 60 kilometers
def distance_per_barrel : ℕ := 60

-- The maximum distance one car can travel one way on a full tank
def max_one_way_distance := max_gasoline_barrels * distance_per_barrel

-- Total trip distance for the furthest traveling car
def total_trip_distance := 2160

-- Distance the other car turns back
def turn_back_distance := 360

-- Formalize in Lean
theorem max_distance_travel :
  (∃ x : ℕ, x = turn_back_distance ∧ max_gasoline_barrels * distance_per_barrel = 360) ∧
  (∃ y : ℕ, y = max_one_way_distance * 3 - turn_back_distance * 6 ∧ y = total_trip_distance) :=
by
  sorry

end max_distance_travel_l225_225187


namespace find_y_given_conditions_l225_225478

theorem find_y_given_conditions (a x y : ℝ) (h1 : y = a * x + (1 - a)) 
  (x_val : x = 3) (y_val : y = 7) (x_new : x = 8) :
  y = 22 := 
  sorry

end find_y_given_conditions_l225_225478


namespace perfect_square_trinomial_l225_225728

theorem perfect_square_trinomial (m : ℤ) : 
  (∃ x y : ℝ, 16 * x^2 + m * x * y + 25 * y^2 = (4 * x + 5 * y)^2 ∨ 16 * x^2 + m * x * y + 25 * y^2 = (4 * x - 5 * y)^2) ↔ (m = 40 ∨ m = -40) :=
by
  sorry

end perfect_square_trinomial_l225_225728


namespace problem_a_problem_b_l225_225846

-- Problem a conditions and statement
def digit1a : Nat := 1
def digit2a : Nat := 4
def digit3a : Nat := 2
def digit4a : Nat := 8
def digit5a : Nat := 5

theorem problem_a : (digit1a * 100000 + digit2a * 10000 + digit3a * 1000 + digit4a * 100 + digit5a * 10 + 7) * 5 = 
                    7 * (digit1a * 100000 + digit2a * 10000 + digit3a * 1000 + digit4a * 100 + digit5a * 10 + 285) := by
  sorry

-- Problem b conditions and statement
def digit1b : Nat := 4
def digit2b : Nat := 2
def digit3b : Nat := 8
def digit4b : Nat := 5
def digit5b : Nat := 7

theorem problem_b : (1 * 100000 + digit1b * 10000 + digit2b * 1000 + digit3b * 100 + digit4b * 10 + digit5b) * 3 = 
                    (digit1b * 100000 + digit2b * 10000 + digit3b * 1000 + digit4b * 100 + digit5b * 10 + 1) := by
  sorry

end problem_a_problem_b_l225_225846


namespace mabel_initial_daisies_l225_225907

theorem mabel_initial_daisies (D: ℕ) (h1: 8 * (D - 2) = 24) : D = 5 :=
by
  sorry

end mabel_initial_daisies_l225_225907


namespace second_player_wins_when_2003_candies_l225_225369

def game_winning_strategy (n : ℕ) : ℕ :=
  if n % 2 = 0 then 1 else 2

theorem second_player_wins_when_2003_candies :
  game_winning_strategy 2003 = 2 :=
by 
  sorry

end second_player_wins_when_2003_candies_l225_225369


namespace Tony_science_degree_years_l225_225028

theorem Tony_science_degree_years (X : ℕ) (Total : ℕ)
  (h1 : Total = 14)
  (h2 : Total = X + 2 * X + 2) :
  X = 4 :=
by
  sorry

end Tony_science_degree_years_l225_225028


namespace prime_cannot_be_sum_of_three_squares_l225_225094

theorem prime_cannot_be_sum_of_three_squares (p : ℕ) (hp : Nat.Prime p) (hmod : p % 8 = 7) :
  ¬∃ a b c : ℤ, p = a^2 + b^2 + c^2 :=
by
  sorry

end prime_cannot_be_sum_of_three_squares_l225_225094


namespace gcd_max_two_digits_l225_225587

theorem gcd_max_two_digits (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (h3 : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 100 :=
by
  -- Definitions
  sorry

end gcd_max_two_digits_l225_225587


namespace gift_box_spinning_tops_l225_225475

theorem gift_box_spinning_tops
  (red_box_cost : ℕ) (red_box_tops : ℕ)
  (yellow_box_cost : ℕ) (yellow_box_tops : ℕ)
  (total_spent : ℕ) (total_boxes : ℕ)
  (h_red_box_cost : red_box_cost = 5)
  (h_red_box_tops : red_box_tops = 3)
  (h_yellow_box_cost : yellow_box_cost = 9)
  (h_yellow_box_tops : yellow_box_tops = 5)
  (h_total_spent : total_spent = 600)
  (h_total_boxes : total_boxes = 72) :
  ∃ (red_boxes : ℕ) (yellow_boxes : ℕ), (red_boxes + yellow_boxes = total_boxes) ∧
  (red_box_cost * red_boxes + yellow_box_cost * yellow_boxes = total_spent) ∧
  (red_box_tops * red_boxes + yellow_box_tops * yellow_boxes = 336) :=
by
  sorry

end gift_box_spinning_tops_l225_225475


namespace weight_of_raisins_proof_l225_225900

-- Define the conditions
def weight_of_peanuts : ℝ := 0.1
def total_weight_of_snacks : ℝ := 0.5

-- Theorem to prove that the weight of raisins equals 0.4 pounds
theorem weight_of_raisins_proof : total_weight_of_snacks - weight_of_peanuts = 0.4 := by
  sorry

end weight_of_raisins_proof_l225_225900


namespace sum_of_consecutive_multiples_of_4_l225_225793

theorem sum_of_consecutive_multiples_of_4 (n : ℝ) (h : 4 * n + (4 * n + 8) = 140) :
  4 * n + (4 * n + 4) + (4 * n + 8) = 210 :=
sorry

end sum_of_consecutive_multiples_of_4_l225_225793


namespace closest_integer_to_sqrt_11_l225_225363

theorem closest_integer_to_sqrt_11 : 
  ∀ (x : ℝ), (3 : ℝ) ≤ x → x ≤ 3.5 → x = 3 :=
by
  intro x hx h3_5
  sorry

end closest_integer_to_sqrt_11_l225_225363


namespace kids_on_Monday_l225_225142

-- Defining the conditions
def kidsOnTuesday : ℕ := 10
def difference : ℕ := 8

-- Formulating the theorem to prove the number of kids Julia played with on Monday
theorem kids_on_Monday : kidsOnTuesday + difference = 18 := by
  sorry

end kids_on_Monday_l225_225142


namespace find_g_60_l225_225018

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_func_eq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y^2
axiom g_45 : g 45 = 15

theorem find_g_60 : g 60 = 8.4375 := sorry

end find_g_60_l225_225018


namespace fewer_columns_after_rearrangement_l225_225812

theorem fewer_columns_after_rearrangement : 
  ∀ (T R R' C C' fewer_columns : ℕ),
    T = 30 → 
    R = 5 → 
    R' = R + 4 →
    C * R = T →
    C' * R' = T →
    fewer_columns = C - C' →
    fewer_columns = 3 :=
by
  intros T R R' C C' fewer_columns hT hR hR' hCR hC'R' hfewer_columns
  -- sorry to skip the proof part
  sorry

end fewer_columns_after_rearrangement_l225_225812


namespace matrix_det_l225_225966

def matrix := ![
  ![2, -4, 2],
  ![0, 6, -1],
  ![5, -3, 1]
]

theorem matrix_det : Matrix.det matrix = -34 := by
  sorry

end matrix_det_l225_225966


namespace find_workers_l225_225182

def total_workers := 20
def male_work_days := 2
def female_work_days := 3

theorem find_workers (X Y : ℕ) 
  (h1 : X + Y = total_workers)
  (h2 : X / male_work_days + Y / female_work_days = 1) : 
  X = 12 ∧ Y = 8 :=
sorry

end find_workers_l225_225182


namespace parabola_vertex_and_point_l225_225780

theorem parabola_vertex_and_point (a b c : ℝ) (h_vertex : (1, -2) = (1, a * 1^2 + b * 1 + c))
  (h_point : (3, 7) = (3, a * 3^2 + b * 3 + c)) : a = 3 := 
by {
  sorry
}

end parabola_vertex_and_point_l225_225780


namespace nitin_borrowed_amount_l225_225909

theorem nitin_borrowed_amount (P : ℝ) (I1 I2 I3 : ℝ) :
  (I1 = P * 0.06 * 3) ∧
  (I2 = P * 0.09 * 5) ∧
  (I3 = P * 0.13 * 3) ∧
  (I1 + I2 + I3 = 8160) →
  P = 8000 :=
by
  sorry

end nitin_borrowed_amount_l225_225909


namespace brick_width_l225_225504

theorem brick_width (l_brick : ℕ) (w_courtyard l_courtyard : ℕ) (num_bricks : ℕ) (w_brick : ℕ)
  (H1 : l_courtyard = 24) 
  (H2 : w_courtyard = 14) 
  (H3 : num_bricks = 8960) 
  (H4 : l_brick = 25) 
  (H5 : (w_courtyard * 100 * l_courtyard * 100 = (num_bricks * (l_brick * w_brick)))) :
  w_brick = 15 :=
by
  sorry

end brick_width_l225_225504


namespace problem_solution_l225_225068

noncomputable def solve_problem : Prop :=
  ∃ (d : ℝ), 
    (∃ int_part : ℤ, 
        (3 * int_part^2 - 12 * int_part + 9 = 0 ∧ ⌊d⌋ = int_part) ∧
        ∀ frac_part : ℝ,
            (4 * frac_part^3 - 8 * frac_part^2 + 3 * frac_part - 0.5 = 0 ∧ frac_part = d - ⌊d⌋) )
    ∧ (d = 1.375 ∨ d = 3.375)

theorem problem_solution : solve_problem :=
by sorry

end problem_solution_l225_225068


namespace find_weight_of_a_l225_225204

-- Define the weights
variables (a b c d e : ℝ)

-- Given conditions
def condition1 := (a + b + c) / 3 = 50
def condition2 := (a + b + c + d) / 4 = 53
def condition3 := (b + c + d + e) / 4 = 51
def condition4 := e = d + 3

-- Proof goal
theorem find_weight_of_a : condition1 a b c → condition2 a b c d → condition3 b c d e → condition4 d e → a = 73 :=
by
  intros h1 h2 h3 h4
  sorry

end find_weight_of_a_l225_225204


namespace base10_to_base7_conversion_l225_225967

theorem base10_to_base7_conversion :
  ∃ b1 b2 b3 b4 b5 : ℕ, 3 * 7^3 + 1 * 7^2 + 6 * 7^1 + 6 * 7^0 = 3527 ∧ 
  b1 = 1 ∧ b2 = 3 ∧ b3 = 1 ∧ b4 = 6 ∧ b5 = 6 ∧ (3527:ℕ) = (1*7^4 + b1*7^3 + b2*7^2 + b3*7^1 + b4*7^0) := by
sorry

end base10_to_base7_conversion_l225_225967


namespace my_age_now_l225_225048

theorem my_age_now (Y S : ℕ) (h1 : Y - 9 = 5 * (S - 9)) (h2 : Y = 3 * S) : Y = 54 := by
  sorry

end my_age_now_l225_225048


namespace twelfth_term_is_three_l225_225929

-- Define the first term and the common difference of the arithmetic sequence
def first_term : ℚ := 1 / 4
def common_difference : ℚ := 1 / 4

-- Define the nth term of an arithmetic sequence
def nth_term (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1) * d

-- Prove that the twelfth term is equal to 3
theorem twelfth_term_is_three : nth_term first_term common_difference 12 = 3 := 
  by 
    sorry

end twelfth_term_is_three_l225_225929


namespace broker_investment_increase_l225_225210

noncomputable def final_value_stock_A := 
  let initial := 100.0
  let year1 := initial * (1 + 0.80)
  let year2 := year1 * (1 - 0.30)
  year2 * (1 + 0.10)

noncomputable def final_value_stock_B := 
  let initial := 100.0
  let year1 := initial * (1 + 0.50)
  let year2 := year1 * (1 - 0.10)
  year2 * (1 - 0.25)

noncomputable def final_value_stock_C := 
  let initial := 100.0
  let year1 := initial * (1 - 0.30)
  let year2 := year1 * (1 - 0.40)
  year2 * (1 + 0.80)

noncomputable def final_value_stock_D := 
  let initial := 100.0
  let year1 := initial * (1 + 0.40)
  let year2 := year1 * (1 + 0.20)
  year2 * (1 - 0.15)

noncomputable def total_final_value := 
  final_value_stock_A + final_value_stock_B + final_value_stock_C + final_value_stock_D

noncomputable def initial_total_value := 4 * 100.0

noncomputable def net_increase := total_final_value - initial_total_value

noncomputable def net_increase_percentage := (net_increase / initial_total_value) * 100

theorem broker_investment_increase : net_increase_percentage = 14.5625 := 
by
  sorry

end broker_investment_increase_l225_225210


namespace gabi_final_prices_l225_225434

theorem gabi_final_prices (x y : ℝ) (hx : 0.8 * x = 1.2 * y) (hl : (x - 0.8 * x) + (y - 1.2 * y) = 10) :
  x = 30 ∧ y = 20 := sorry

end gabi_final_prices_l225_225434


namespace total_frogs_seen_by_hunter_l225_225879

/-- Hunter saw 5 frogs sitting on lily pads in the pond. -/
def initial_frogs : ℕ := 5

/-- Three more frogs climbed out of the water onto logs floating in the pond. -/
def frogs_on_logs : ℕ := 3

/-- Two dozen baby frogs (24 frogs) hopped onto a big rock jutting out from the pond. -/
def baby_frogs : ℕ := 24

/--
The total number of frogs Hunter saw in the pond.
-/
theorem total_frogs_seen_by_hunter : initial_frogs + frogs_on_logs + baby_frogs = 32 := by
sorry

end total_frogs_seen_by_hunter_l225_225879


namespace volume_increase_by_eight_l225_225174

noncomputable def sphere_volume (r : ℝ) : ℝ :=
  (4 / 3) * π * (r^3)

theorem volume_increase_by_eight (r : ℝ) :
  sphere_volume (2 * r) = 8 * sphere_volume r :=
by
  sorry

end volume_increase_by_eight_l225_225174


namespace distance_range_l225_225626

variable (x : ℝ)
variable (starting_fare : ℝ := 6) -- fare in yuan for up to 2 kilometers
variable (surcharge : ℝ := 1) -- yuan surcharge per ride
variable (additional_fare : ℝ := 1) -- fare for every additional 0.5 kilometers
variable (additional_distance : ℝ := 0.5) -- distance in kilometers for every additional fare

theorem distance_range (h_total_fare : 9 = starting_fare + (x - 2) / additional_distance * additional_fare + surcharge) :
  2.5 < x ∧ x ≤ 3 :=
by
  -- Proof goes here
  sorry

end distance_range_l225_225626


namespace parabola_vertex_l225_225482

theorem parabola_vertex :
  (∃ x y : ℝ, y^2 + 6 * y + 4 * x - 7 = 0 ∧ (x, y) = (4, -3)) :=
sorry

end parabola_vertex_l225_225482


namespace triangle_properties_l225_225606

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (BD : ℝ) (D : ℝ) : 
  (a + c) * Real.sin A = Real.sin A + Real.sin C →
  c^2 + c = b^2 - 1 →
  D = (a + c) / 2 →
  BD = Real.sqrt 3 / 2 →
  B = 2 * Real.pi / 3 ∧ (1 / 2) * a * c * Real.sin B = Real.sqrt 3 / 2 :=
by
  sorry

end triangle_properties_l225_225606


namespace isosceles_triangle_perimeter_l225_225835

theorem isosceles_triangle_perimeter (a b : ℕ) (h_isosceles : a = 3 ∨ a = 7 ∨ b = 3 ∨ b = 7) (h_ineq1 : 3 + 3 ≤ b ∨ b + b ≤ 3) (h_ineq2 : 7 + 7 ≥ a ∨ a + a ≥ 7) :
  (a = 3 ∧ b = 7) → 3 + 7 + 7 = 17 :=
by
  -- To be completed
  sorry

end isosceles_triangle_perimeter_l225_225835


namespace seventh_numbers_sum_l225_225469

def first_row_seq (n : ℕ) : ℕ := n^2 + n - 1

def second_row_seq (n : ℕ) : ℕ := n * (n + 1) / 2

theorem seventh_numbers_sum :
  first_row_seq 7 + second_row_seq 7 = 83 :=
by
  -- Skipping the proof
  sorry

end seventh_numbers_sum_l225_225469


namespace count_two_digit_perfect_squares_divisible_by_four_l225_225114

theorem count_two_digit_perfect_squares_divisible_by_four : ∃ n, n = 3 ∧
  (∀ k, (10 ≤ k ∧ k < 100) → (∃ m, k = m^2) → k % 4 = 0 → ∃ p, (p = 16 ∨ p = 36 ∨ p = 64) ∧ p = k) := 
by 
  use 3
  intro k h1 h2 h3
  cases h2 with m hm
  sorry

end count_two_digit_perfect_squares_divisible_by_four_l225_225114


namespace apple_multiple_l225_225759

theorem apple_multiple (K Ka : ℕ) (M : ℕ) 
  (h1 : K + Ka = 340)
  (h2 : Ka = M * K + 10)
  (h3 : Ka = 274) : 
  M = 4 := 
by
  sorry

end apple_multiple_l225_225759


namespace max_area_dog_roam_l225_225009

theorem max_area_dog_roam (r : ℝ) (s : ℝ) (half_s : ℝ) (midpoint : Prop) :
  r = 10 → s = 20 → half_s = s / 2 → midpoint → 
  r > half_s → 
  π * r^2 = 100 * π :=
by 
  intros hr hs h_half_s h_midpoint h_rope_length
  sorry

end max_area_dog_roam_l225_225009


namespace bicycle_speed_l225_225353

theorem bicycle_speed (d1 d2 v1 v_avg : ℝ)
  (h1 : d1 = 300) 
  (h2 : d1 + d2 = 450) 
  (h3 : v1 = 20) 
  (h4 : v_avg = 18) : 
  (d2 / ((d1 / v1) + d2 / (d2 * v_avg / 450)) = 15) :=
by 
  sorry

end bicycle_speed_l225_225353


namespace soda_difference_l225_225755

theorem soda_difference :
  let Julio_orange_bottles := 4
  let Julio_grape_bottles := 7
  let Mateo_orange_bottles := 1
  let Mateo_grape_bottles := 3
  let liters_per_bottle := 2
  let Julio_total_liters := Julio_orange_bottles * liters_per_bottle + Julio_grape_bottles * liters_per_bottle
  let Mateo_total_liters := Mateo_orange_bottles * liters_per_bottle + Mateo_grape_bottles * liters_per_bottle
  Julio_total_liters - Mateo_total_liters = 14 := by
    sorry

end soda_difference_l225_225755


namespace gcd_digit_bound_l225_225582

theorem gcd_digit_bound (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (lcm_ab : ℕ) (h_lcm : 10^11 ≤ lcm_ab ∧ lcm_ab < 10^12) :
  int.log10 (Nat.gcd a b) < 3 := 
sorry

end gcd_digit_bound_l225_225582


namespace find_other_outlet_rate_l225_225219

open Real

-- Definitions based on conditions
def V : ℝ := 20 * 1728   -- volume of the tank in cubic inches
def r1 : ℝ := 5          -- rate of inlet pipe in cubic inches/min
def r2 : ℝ := 8          -- rate of one outlet pipe in cubic inches/min
def t : ℝ := 2880        -- time in minutes required to empty the tank
 
-- Mathematically equivalent proof statement
theorem find_other_outlet_rate (x : ℝ) : 
  -- Given conditions
  V = 34560 →
  r1 = 5 →
  r2 = 8 →
  t = 2880 →
  -- Statement to prove
  V = (r2 + x - r1) * t → x = 9 :=
by
  intro hV hr1 hr2 ht hEq
  sorry

end find_other_outlet_rate_l225_225219


namespace percentage_of_students_liking_chess_l225_225454

theorem percentage_of_students_liking_chess (total_students : ℕ) (basketball_percentage : ℝ) (soccer_percentage : ℝ) 
(identified_chess_or_basketball : ℕ) (students_liking_basketball : ℕ) : 
total_students = 250 ∧ basketball_percentage = 0.40 ∧ soccer_percentage = 0.28 ∧ identified_chess_or_basketball = 125 ∧ 
students_liking_basketball = 100 → ∃ C : ℝ, C = 0.10 :=
by
  sorry

end percentage_of_students_liking_chess_l225_225454


namespace range_of_t_l225_225311

theorem range_of_t (a b c t: ℝ) 
  (h1 : 6 * a = 2 * b - 6)
  (h2 : 6 * a = 3 * c)
  (h3 : b ≥ 0)
  (h4 : c ≤ 2)
  (h5 : t = 2 * a + b - c) : 
  0 ≤ t ∧ t ≤ 6 :=
sorry

end range_of_t_l225_225311


namespace arrangement_problem_l225_225484
noncomputable def num_arrangements : ℕ := 144

theorem arrangement_problem (A B C D E F : ℕ) 
  (adjacent_easy : A = B) 
  (not_adjacent_difficult : E ≠ F) : num_arrangements = 144 :=
by sorry

end arrangement_problem_l225_225484


namespace percentage_refund_l225_225202

theorem percentage_refund
  (initial_amount : ℕ)
  (sweater_cost : ℕ)
  (tshirt_cost : ℕ)
  (shoes_cost : ℕ)
  (amount_left_after_refund : ℕ)
  (refund_percentage : ℕ) :
  initial_amount = 74 →
  sweater_cost = 9 →
  tshirt_cost = 11 →
  shoes_cost = 30 →
  amount_left_after_refund = 51 →
  refund_percentage = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end percentage_refund_l225_225202


namespace scientific_notation_of_8450_l225_225747

theorem scientific_notation_of_8450 :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ (8450 : ℝ) = a * 10^n ∧ (a = 8.45) ∧ (n = 3) :=
sorry

end scientific_notation_of_8450_l225_225747


namespace student_solved_correctly_l225_225949

theorem student_solved_correctly (x : ℕ) :
  (x + 2 * x = 36) → x = 12 :=
by
  intro h
  sorry

end student_solved_correctly_l225_225949


namespace math_problem_l225_225869

theorem math_problem (a b c m n : ℝ)
  (h1 : a = -b)
  (h2 : c = -1)
  (h3 : m * n = 1) : 
  (a + b) / 3 + c^2 - 4 * m * n = -3 := 
by 
  -- Proof steps would be here
  sorry

end math_problem_l225_225869


namespace problem1_problem2_l225_225045

open Real

/-- Problem 1: Simplify trigonometric expression. -/
theorem problem1 : 
  (sqrt (1 - 2 * sin (10 * pi / 180) * cos (10 * pi / 180)) /
  (sin (170 * pi / 180) - sqrt (1 - sin (170 * pi / 180)^2))) = -1 :=
sorry

/-- Problem 2: Given tan(θ) = 2, find the value.
  Required to prove: 2 + sin(θ) * cos(θ) - cos(θ)^2 equals 11/5 -/
theorem problem2 (θ : ℝ) (h : tan θ = 2) :
  2 + sin θ * cos θ - cos θ^2 = 11 / 5 :=
sorry

end problem1_problem2_l225_225045


namespace determine_phi_l225_225871

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem determine_phi 
  (φ : ℝ)
  (H1 : ∀ x : ℝ, f x φ ≤ |f (π / 6) φ|)
  (H2 : f (π / 3) φ > f (π / 2) φ) :
  φ = π / 6 :=
sorry

end determine_phi_l225_225871


namespace circle_common_chord_l225_225921

theorem circle_common_chord (x y : ℝ) :
  (x^2 + y^2 - 4 * x + 6 * y = 0) ∧
  (x^2 + y^2 - 6 * x = 0) →
  (x + 3 * y = 0) :=
by
  sorry

end circle_common_chord_l225_225921


namespace orvin_max_balloons_l225_225768

variable (C : ℕ) (P : ℕ)

noncomputable def max_balloons (C P : ℕ) : ℕ :=
  let pair_cost := P + P / 2  -- Cost for two balloons
  let pairs := C / pair_cost  -- Maximum number of pairs
  pairs * 2 + (if C % pair_cost >= P then 1 else 0) -- Total balloons considering the leftover money

theorem orvin_max_balloons (hC : C = 120) (hP : P = 3) : max_balloons C P = 53 :=
by
  sorry

end orvin_max_balloons_l225_225768


namespace leopards_count_l225_225961

theorem leopards_count (L : ℕ) (h1 : 100 + 80 + L + 10 * L + 50 + 2 * (80 + L) = 670) : L = 20 :=
by
  sorry

end leopards_count_l225_225961


namespace relationship_among_abc_l225_225297

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := (1 / 3) ^ 2
noncomputable def c : ℝ := Real.log (1 / 30) / Real.log (1 / 3)

theorem relationship_among_abc : c > a ∧ a > b :=
by
  sorry

end relationship_among_abc_l225_225297


namespace geometric_sequence_sum_l225_225080

theorem geometric_sequence_sum :
  let a := (1:ℚ)/3
  let r := (1:ℚ)/3
  let n := 8
  (a * (1 - r^n) / (1 - r)) = 3280 / 6561 :=
by {
  let a := (1:ℚ)/3
  let r := (1:ℚ)/3
  let n := 8
  calc
  (a * (1 - r^n) / (1 - r)) = (1/3 * (1 - (1/3)^8) / (1 - 1/3)) : by rw a; rw r
  ... = 3280 / 6561 : sorry
}

end geometric_sequence_sum_l225_225080


namespace area_of_triangle_bounded_by_coordinate_axes_and_line_l225_225816

def area_of_triangle (b h : ℕ) : ℕ := (b * h) / 2

theorem area_of_triangle_bounded_by_coordinate_axes_and_line :
  area_of_triangle 4 6 = 12 :=
by
  sorry

end area_of_triangle_bounded_by_coordinate_axes_and_line_l225_225816


namespace book_total_pages_l225_225648

theorem book_total_pages (n : ℕ) (h1 : 5 * n / 8 - 3 * n / 7 = 33) : n = n :=
by 
  -- We skip the proof as instructed
  sorry

end book_total_pages_l225_225648


namespace problem_statement_l225_225709

open Real

theorem problem_statement (α : ℝ) 
  (h1 : cos (α + π / 4) = (7 * sqrt 2) / 10)
  (h2 : cos (2 * α) = 7 / 25) :
  sin α + cos α = 1 / 5 :=
sorry

end problem_statement_l225_225709


namespace largest_base4_is_largest_l225_225403

theorem largest_base4_is_largest 
  (n1 : ℕ) (n2 : ℕ) (n3 : ℕ) (n4 : ℕ)
  (h1 : n1 = 31) (h2 : n2 = 52) (h3 : n3 = 54) (h4 : n4 = 46) :
  n3 = Nat.max (Nat.max n1 n2) (Nat.max n3 n4) :=
by
  sorry

end largest_base4_is_largest_l225_225403


namespace tangent_product_eq_three_l225_225265

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end tangent_product_eq_three_l225_225265


namespace evaluate_at_neg_one_l225_225440

def f (x : ℝ) : ℝ := -2 * x ^ 2 + 1

theorem evaluate_at_neg_one : f (-1) = -1 := 
by
  -- Proof goes here
  sorry

end evaluate_at_neg_one_l225_225440


namespace tan_product_eq_three_l225_225235

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end tan_product_eq_three_l225_225235


namespace maximize_S_n_decreasing_arithmetic_sequence_l225_225783

theorem maximize_S_n_decreasing_arithmetic_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d < 0)
  (h3 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2)
  (h4 : S 5 = S 10) :
  S 7 = S 8 :=
sorry

end maximize_S_n_decreasing_arithmetic_sequence_l225_225783


namespace maximum_area_of_rectangle_with_given_perimeter_l225_225173

noncomputable def perimeter : ℝ := 30
noncomputable def area (length width : ℝ) : ℝ := length * width
noncomputable def max_area : ℝ := 56.25

theorem maximum_area_of_rectangle_with_given_perimeter :
  ∃ length width : ℝ, 2 * length + 2 * width = perimeter ∧ area length width = max_area :=
sorry

end maximum_area_of_rectangle_with_given_perimeter_l225_225173


namespace inequality_solution_empty_set_l225_225923

theorem inequality_solution_empty_set : ∀ x : ℝ, ¬ (x * (2 - x) > 3) :=
by
  -- Translate the condition and show that there are no x satisfying the inequality
  sorry

end inequality_solution_empty_set_l225_225923


namespace count_two_digit_perfect_squares_divisible_by_four_l225_225113

theorem count_two_digit_perfect_squares_divisible_by_four : ∃ n, n = 3 ∧
  (∀ k, (10 ≤ k ∧ k < 100) → (∃ m, k = m^2) → k % 4 = 0 → ∃ p, (p = 16 ∨ p = 36 ∨ p = 64) ∧ p = k) := 
by 
  use 3
  intro k h1 h2 h3
  cases h2 with m hm
  sorry

end count_two_digit_perfect_squares_divisible_by_four_l225_225113


namespace prob1_prob2_prob3_l225_225284

def star (a b : ℤ) : ℤ :=
  if a = 0 then b^2
  else if b = 0 then a^2
  else if a > 0 ∧ b > 0 then a^2 + b^2
  else if a < 0 ∧ b < 0 then a^2 + b^2
  else -(a^2 + b^2)

theorem prob1 :
  star (-1) (-1) = 2 :=
sorry

theorem prob2 :
  star (-1) (star 0 (-2)) = -17 :=
sorry

theorem prob3 (m n : ℤ) :
  star (m-1) (n+2) = -2 → (m - n = 1 ∨ m - n = 5) :=
sorry

end prob1_prob2_prob3_l225_225284


namespace number_of_zeros_l225_225485

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then |x| - 2 else 2 * x - 6 + Real.log x

theorem number_of_zeros :
  (∃ x : ℝ, f x = 0) ∧ (∃ y : ℝ, f y = 0) ∧ (∀ z : ℝ, f z = 0 → z = x ∨ z = y) :=
by
  sorry

end number_of_zeros_l225_225485


namespace right_triangles_with_specific_area_and_perimeter_l225_225726

theorem right_triangles_with_specific_area_and_perimeter :
  ∃ (count : ℕ),
    count = 7 ∧
    ∀ (a b : ℕ), 
      (a > 0 ∧ b > 0 ∧ (a ≠ b) ∧ (a^2 + b^2 = c^2) ∧ (a * b / 2 = 5 * (a + b + c))) → 
      count = 7 :=
by
  sorry

end right_triangles_with_specific_area_and_perimeter_l225_225726


namespace system_unique_solution_l225_225289

theorem system_unique_solution 
  (x y z : ℝ) 
  (h1 : x + y + z = 3 * x * y) 
  (h2 : x^2 + y^2 + z^2 = 3 * x * z) 
  (h3 : x^3 + y^3 + z^3 = 3 * y * z) 
  (hx : 0 ≤ x) 
  (hy : 0 ≤ y) 
  (hz : 0 ≤ z) : 
  (x = 1 ∧ y = 1 ∧ z = 1) := 
sorry

end system_unique_solution_l225_225289


namespace q_compound_l225_225763

def q (x y : ℤ) : ℤ :=
  if x ≥ 1 ∧ y ≥ 1 then 2 * x + 3 * y
  else if x < 0 ∧ y < 0 then x + y^2
  else 4 * x - 2 * y

theorem q_compound : q (q 2 (-2)) (q 0 0) = 48 := 
by 
  sorry

end q_compound_l225_225763


namespace negation_example_l225_225019

theorem negation_example (h : ∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 + 3 * x - 1 > 0) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, x^2 + 3 * x - 1 ≤ 0 :=
sorry

end negation_example_l225_225019


namespace parabola_equation_l225_225853

def equation_of_parabola (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, y = a * x^2 + b * x + c ↔ 
              (∃ a : ℝ, y = a * (x - 3)^2 + 5) ∧
              y = (if x = 0 then 2 else y)

theorem parabola_equation :
  equation_of_parabola (-1 / 3) 2 2 :=
by
  -- First, show that the vertex form (x-3)^2 + 5 meets the conditions
  sorry

end parabola_equation_l225_225853


namespace towel_bleaching_l225_225952

theorem towel_bleaching
  (original_length original_breadth : ℝ)
  (percentage_decrease_area : ℝ)
  (percentage_decrease_breadth : ℝ)
  (final_length final_breadth : ℝ)
  (h1 : percentage_decrease_area = 28)
  (h2 : percentage_decrease_breadth = 10)
  (h3 : final_breadth = original_breadth * (1 - percentage_decrease_breadth / 100))
  (h4 : final_area = original_area (1 - percentage_decrease_area / 100))
  (original_area final_area : ℝ) :
  final_length = original_length * 0.8 :=
begin
  -- Here, the goal would be to prove that final_length is 80% of the original_length
  sorry
end

end towel_bleaching_l225_225952


namespace solve_eq_l225_225850

noncomputable def fx (x : ℝ) : ℝ :=
  ((x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 3) * (x - 2) * (x - 1) * (x - 5)) /
  ((x - 2) * (x - 4) * (x - 2) * (x - 5))

theorem solve_eq (x : ℝ) (h : x ≠ 2 ∧ x ≠ 4 ∧ x ≠ 5) :
  fx x = 1 ↔ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 :=
by
  sorry

end solve_eq_l225_225850


namespace remainder_is_x_plus_2_l225_225431

noncomputable def problem_division := 
  ∀ x : ℤ, ∃ q r : ℤ, (x^3 + 2 * x^2) = q * (x^2 + 3 * x + 2) + r ∧ r < x^2 + 3 * x + 2 ∧ r = x + 2

theorem remainder_is_x_plus_2 : problem_division := sorry

end remainder_is_x_plus_2_l225_225431


namespace natasha_can_achieve_plan_l225_225617

noncomputable def count_ways : Nat :=
  let num_1x1 := 4
  let num_1x2 := 24
  let target := 2021
  6517

theorem natasha_can_achieve_plan (num_1x1 num_1x2 target : Nat) (h1 : num_1x1 = 4) (h2 : num_1x2 = 24) (h3 : target = 2021) :
  count_ways = 6517 :=
by
  sorry

end natasha_can_achieve_plan_l225_225617


namespace decrease_in_length_l225_225951

theorem decrease_in_length (L B : ℝ) (h₀ : L ≠ 0) (h₁ : B ≠ 0)
  (h₂ : ∃ (A' : ℝ), A' = 0.72 * L * B)
  (h₃ : ∃ B' : ℝ, B' = B * 0.9) :
  ∃ (x : ℝ), x = 20 :=
by
  sorry

end decrease_in_length_l225_225951


namespace triangle_inequality_l225_225914
open Real

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_abc : a + b > c) (h_acb : a + c > b) (h_bca : b + c > a) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end triangle_inequality_l225_225914


namespace sqrt_inequality_l225_225129

theorem sqrt_inequality (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
  sorry

end sqrt_inequality_l225_225129


namespace major_axis_length_of_intersecting_ellipse_l225_225395

theorem major_axis_length_of_intersecting_ellipse (radius : ℝ) (h_radius : radius = 2) 
  (minor_axis_length : ℝ) (h_minor_axis : minor_axis_length = 2 * radius) (major_axis_length : ℝ) 
  (h_major_axis : major_axis_length = minor_axis_length * 1.6) :
  major_axis_length = 6.4 :=
by 
  -- The proof will follow here, but currently it's not required.
  sorry

end major_axis_length_of_intersecting_ellipse_l225_225395


namespace log_one_plus_x_sq_lt_x_sq_l225_225890

theorem log_one_plus_x_sq_lt_x_sq {x : ℝ} (hx : 0 < x) : 
  Real.log (1 + x^2) < x^2 := 
sorry

end log_one_plus_x_sq_lt_x_sq_l225_225890


namespace total_practice_hours_l225_225347

-- Definitions based on conditions
def weekday_practice_hours : ℕ := 3
def saturday_practice_hours : ℕ := 5
def weekdays_per_week : ℕ := 5
def weeks_until_game : ℕ := 3

-- Theorem statement
theorem total_practice_hours : (weekday_practice_hours * weekdays_per_week + saturday_practice_hours) * weeks_until_game = 60 := 
by sorry

end total_practice_hours_l225_225347


namespace frustum_lateral_area_l225_225024

def frustum_upper_base_radius : ℝ := 3
def frustum_lower_base_radius : ℝ := 4
def frustum_slant_height : ℝ := 6

theorem frustum_lateral_area : 
  (1 / 2) * (frustum_upper_base_radius + frustum_lower_base_radius) * 2 * Real.pi * frustum_slant_height = 42 * Real.pi :=
by
  sorry

end frustum_lateral_area_l225_225024


namespace bread_count_at_end_of_day_l225_225814

def initial_loaves : ℕ := 2355
def sold_loaves : ℕ := 629
def delivered_loaves : ℕ := 489

theorem bread_count_at_end_of_day : 
  initial_loaves - sold_loaves + delivered_loaves = 2215 := by
  sorry

end bread_count_at_end_of_day_l225_225814


namespace total_frogs_in_pond_l225_225883

def frogsOnLilyPads : ℕ := 5
def frogsOnLogs : ℕ := 3
def babyFrogsOnRock : ℕ := 2 * 12 -- Two dozen

theorem total_frogs_in_pond : frogsOnLilyPads + frogsOnLogs + babyFrogsOnRock = 32 :=
by
  sorry

end total_frogs_in_pond_l225_225883


namespace find_c_l225_225547

theorem find_c :
  ∃ c : ℝ, 0 < c ∧ ∀ line : ℝ, (∃ x y : ℝ, (x = 1 ∧ y = c) ∧ (x*x + y*y - 2*x - 2*y - 7 = 0)) ∧ (line = 1*x + 0 + y*c - 0) :=
sorry

end find_c_l225_225547


namespace steps_away_from_goal_l225_225964

-- Given conditions
def goal : ℕ := 100000
def initial_steps_per_day : ℕ := 1000
def increase_steps_per_week : ℕ := 1000
def days_per_week : ℕ := 7
def weeks : ℕ := 4

-- Computation of total steps in 4 weeks
def total_steps : ℕ :=
  (initial_steps_per_day * days_per_week) +
  ((initial_steps_per_day + increase_steps_per_week) * days_per_week) +
  ((initial_steps_per_day + 2 * increase_steps_per_week) * days_per_week) +
  ((initial_steps_per_day + 3 * increase_steps_per_week) * days_per_week)

-- Desired proof statement
theorem steps_away_from_goal : goal - total_steps = 30000 :=
by
  have h1: 7 * 1000 = 7000 := by norm_num
  have h2: 7 * 2000 = 14000 := by norm_num
  have h3: 7 * 3000 = 21000 := by norm_num
  have h4: 7 * 4000 = 28000 := by norm_num
  have h5: total_steps = 7000 + 14000 + 21000 + 28000 := by
    simp [total_steps, initial_steps_per_day, days_per_week, increase_steps_per_week]
    ring
  have h6: 7000 + 14000 + 21000 + 28000 = 70000 := by norm_num
  rw [h5, h6]
  show goal - 70000 = 30000
  norm_num

end steps_away_from_goal_l225_225964


namespace player1_coins_l225_225178

theorem player1_coins (coin_distribution : Fin 9 → ℕ) :
  let rotations := 11
  let player_4_coins := 90
  let player_8_coins := 35
  ∀ player : Fin 9, player = 0 → 
    let player_1_coins := coin_distribution player
    (coin_distribution 3 = player_4_coins) →
    (coin_distribution 7 = player_8_coins) →
    player_1_coins = 57 := 
sorry

end player1_coins_l225_225178


namespace part1_part2_l225_225342

def f (x a : ℝ) := |x - a| + x

theorem part1 (a : ℝ) (h_a : a = 1) : 
  {x : ℝ | f x a ≥ x + 2} = {x | x ≥ 3} ∪ {x | x ≤ -1} := 
by
  sorry

theorem part2 (a : ℝ) (h : {x : ℝ | f x a ≤ 3 * x} = {x | x ≥ 2}) : 
  a = 6 := 
by
  sorry

end part1_part2_l225_225342


namespace f_at_one_f_decreasing_f_min_on_interval_l225_225099

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_defined : ∀ x, 0 < x → ∃ y, f y = y
axiom f_eq : ∀ x1 x2, 0 < x1 → 0 < x2 → f (x1 / x2) = f x1 - f x2
axiom f_neg : ∀ x, 1 < x → f x < 0

-- Proof statements
theorem f_at_one : f 1 = 0 := sorry

theorem f_decreasing : ∀ x1 x2, 0 < x1 → 0 < x2 → x1 < x2 → f x1 > f x2 := sorry

axiom f_at_three : f 3 = -1

theorem f_min_on_interval : ∀ x, 2 ≤ x ∧ x ≤ 9 → f x ≥ -2 := sorry

end f_at_one_f_decreasing_f_min_on_interval_l225_225099


namespace mass_percent_O_CaOH2_is_correct_mass_percent_O_Na2CO3_is_correct_mass_percent_O_K2SO4_is_correct_l225_225079

-- Definitions for molar masses used in calculations
def molar_mass_Ca := 40.08
def molar_mass_O := 16.00
def molar_mass_H := 1.01
def molar_mass_Na := 22.99
def molar_mass_C := 12.01
def molar_mass_K := 39.10
def molar_mass_S := 32.07

-- Molar masses of the compounds
def molar_mass_CaOH2 := molar_mass_Ca + 2 * molar_mass_O + 2 * molar_mass_H
def molar_mass_Na2CO3 := 2 * molar_mass_Na + molar_mass_C + 3 * molar_mass_O
def molar_mass_K2SO4 := 2 * molar_mass_K + molar_mass_S + 4 * molar_mass_O

-- Mass of O in each compound
def mass_O_CaOH2 := 2 * molar_mass_O
def mass_O_Na2CO3 := 3 * molar_mass_O
def mass_O_K2SO4 := 4 * molar_mass_O

-- Mass percentages of O in each compound
def mass_percent_O_CaOH2 := (mass_O_CaOH2 / molar_mass_CaOH2) * 100
def mass_percent_O_Na2CO3 := (mass_O_Na2CO3 / molar_mass_Na2CO3) * 100
def mass_percent_O_K2SO4 := (mass_O_K2SO4 / molar_mass_K2SO4) * 100

theorem mass_percent_O_CaOH2_is_correct :
  mass_percent_O_CaOH2 = 43.19 := by sorry

theorem mass_percent_O_Na2CO3_is_correct :
  mass_percent_O_Na2CO3 = 45.29 := by sorry

theorem mass_percent_O_K2SO4_is_correct :
  mass_percent_O_K2SO4 = 36.73 := by sorry

end mass_percent_O_CaOH2_is_correct_mass_percent_O_Na2CO3_is_correct_mass_percent_O_K2SO4_is_correct_l225_225079


namespace nonnegative_integer_solutions_l225_225281

theorem nonnegative_integer_solutions (x y : ℕ) :
  3 * x^2 + 2 * 9^y = x * (4^(y+1) - 1) ↔ (x, y) ∈ [(2, 1), (3, 1), (3, 2), (18, 2)] :=
by sorry

end nonnegative_integer_solutions_l225_225281


namespace range_of_a_l225_225715

variable {R : Type} [LinearOrderedField R]

def f (x a : R) : R := |x - 1| + |x - 2| - a

theorem range_of_a (h : ∀ x : R, f x a > 0) : a < 1 :=
by
  sorry

end range_of_a_l225_225715


namespace amount_received_by_a_l225_225500

namespace ProofProblem

/-- Total amount of money divided -/
def total_amount : ℕ := 600

/-- Ratio part for 'a' -/
def part_a : ℕ := 1

/-- Ratio part for 'b' -/
def part_b : ℕ := 2

/-- Total parts in the ratio -/
def total_parts : ℕ := part_a + part_b

/-- Amount per part when total is divided evenly by the total number of parts -/
def amount_per_part : ℕ := total_amount / total_parts

/-- Amount received by 'a' when total amount is divided according to the given ratio -/
def amount_a : ℕ := part_a * amount_per_part

theorem amount_received_by_a : amount_a = 200 := by
  -- Proof will be filled in here
  sorry

end ProofProblem

end amount_received_by_a_l225_225500


namespace unique_integer_sequence_exists_l225_225913

open Nat

def a (n : ℕ) : ℤ := sorry

theorem unique_integer_sequence_exists :
  ∃ (a : ℕ → ℤ), a 1 = 1 ∧ a 2 > 1 ∧ (∀ n ≥ 1, (a (n+1))^3 + 1 = a n * a (n+2)) ∧
  (∀ b, (b 1 = 1) → (b 2 > 1) → (∀ n ≥ 1, (b (n+1))^3 + 1 = b n * b (n+2)) → b = a) :=
by
  sorry

end unique_integer_sequence_exists_l225_225913


namespace find_number_of_books_l225_225766

-- Define the constants and equation based on the conditions
def price_paid_per_book : ℕ := 11
def price_sold_per_book : ℕ := 25
def total_difference : ℕ := 210

def books_equation (x : ℕ) : Prop :=
  (price_sold_per_book * x) - (price_paid_per_book * x) = total_difference

-- The theorem statement that needs to be proved
theorem find_number_of_books (x : ℕ) (h : books_equation x) : 
  x = 15 :=
sorry

end find_number_of_books_l225_225766


namespace find_divisor_l225_225945

theorem find_divisor (x : ℝ) (h : 740 / x - 175 = 10) : x = 4 := by
  sorry

end find_divisor_l225_225945


namespace count_two_digit_perfect_squares_divisible_by_4_l225_225117

-- Define what it means to be a two-digit number perfect square divisible by 4
def two_digit_perfect_squares_divisible_by_4 : List ℕ :=
  [16, 36, 64] -- Manually identified two-digit perfect squares which are divisible by 4

-- 6^2 = 36 and 8^2 = 64 both fit, hypothesis checks are already done manually in solution steps
def valid_two_digit_perfect_squares : List ℕ :=
  [16, 25, 36, 49, 64, 81] -- all two-digit perfect squares

-- Define the theorem statement
theorem count_two_digit_perfect_squares_divisible_by_4 :
  (two_digit_perfect_squares_divisible_by_4.count 16 + 
   two_digit_perfect_squares_divisible_by_4.count 36 +
   two_digit_perfect_squares_divisible_by_4.count 64) = 3 :=
by
  -- Proof would go here, omitted by "sorry"
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l225_225117


namespace smallest_k_exists_l225_225527

theorem smallest_k_exists : ∃ (k : ℕ) (n : ℕ), k = 53 ∧ k^2 + 49 = 180 * n :=
sorry

end smallest_k_exists_l225_225527


namespace slope_of_line_l225_225291

theorem slope_of_line : ∀ (x y : ℝ), (6 * x + 10 * y = 30) → (y = -((3 / 5) * x) + 3) :=
by
  -- Proof needs to be filled out
  sorry

end slope_of_line_l225_225291


namespace triangle_area_l225_225821

theorem triangle_area : 
  let line_eq (x y : ℝ) := 3 * x + 2 * y = 12
  let x_intercept := (4 : ℝ)
  let y_intercept := (6 : ℝ)
  ∃ (x y : ℝ), line_eq x y ∧ x = x_intercept ∧ y = y_intercept ∧
  ∃ (area : ℝ), area = 1 / 2 * x * y ∧ area = 12 :=
by
  sorry

end triangle_area_l225_225821


namespace initial_birds_count_l225_225493

variable (init_birds landed_birds total_birds : ℕ)

theorem initial_birds_count :
  (landed_birds = 8) →
  (total_birds = 20) →
  (init_birds + landed_birds = total_birds) →
  (init_birds = 12) :=
by
  intros h1 h2 h3
  sorry

end initial_birds_count_l225_225493


namespace total_eggs_today_l225_225908

def eggs_morning : ℕ := 816
def eggs_afternoon : ℕ := 523

theorem total_eggs_today : eggs_morning + eggs_afternoon = 1339 :=
by {
  sorry
}

end total_eggs_today_l225_225908


namespace find_parameters_l225_225556

noncomputable def cubic_function (a b : ℝ) (x : ℝ) : ℝ :=
  x^3 + a * x^2 + b * x + 27

def deriv_cubic_function (a b : ℝ) (x : ℝ) : ℝ :=
  3 * x^2 + 2 * a * x + b

theorem find_parameters
  (a b : ℝ)
  (h1 : deriv_cubic_function a b (-1) = 0)
  (h2 : deriv_cubic_function a b 3 = 0) :
  a = -3 ∧ b = -9 :=
by
  -- leaving proof as sorry since the task doesn't require proving
  sorry

end find_parameters_l225_225556


namespace hexagon_area_within_rectangle_of_5x4_l225_225899

-- Define the given conditions
def is_rectangle (length width : ℝ) := length > 0 ∧ width > 0

def vertices_touch_midpoints (length width : ℝ) (hexagon_area : ℝ) : Prop :=
  let rectangle_area := length * width
  let triangle_area := (1 / 2) * (length / 2) * (width / 2)
  let total_triangle_area := 4 * triangle_area
  rectangle_area - total_triangle_area = hexagon_area

-- Formulate the main statement to be proved
theorem hexagon_area_within_rectangle_of_5x4 : 
  vertices_touch_midpoints 5 4 10 := 
by
  -- Proof is omitted for this theorem
  sorry

end hexagon_area_within_rectangle_of_5x4_l225_225899


namespace gcd_max_digits_l225_225593

theorem gcd_max_digits {a b : ℕ} (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) 
  (h3 : ∃ k, 10^11 ≤ k ∧ k < 10^{12} ∧ k = lcm a b) : 
  (gcd a b) < 10^3 :=
sorry

end gcd_max_digits_l225_225593


namespace proof_eq1_proof_eq2_l225_225773

variable (x : ℝ)

-- Proof problem for Equation (1)
theorem proof_eq1 (h : (1 - x) / 3 - 2 = x / 6) : x = -10 / 3 := sorry

-- Proof problem for Equation (2)
theorem proof_eq2 (h : (x + 1) / 0.25 - (x - 2) / 0.5 = 5) : x = -3 / 2 := sorry

end proof_eq1_proof_eq2_l225_225773


namespace solve_f_log2_20_l225_225300

noncomputable def f (x : ℝ) : ℝ :=
if -1 ≤ x ∧ x < 0 then 2^x else 0 -- Placeholder for other values

theorem solve_f_log2_20 :
  (∀ x, f (-x) = -f x) →
  (∀ x, f (x + 4) = f x) →
  (∀ x, -1 ≤ x ∧ x < 0 → f x = 2^x) →
  f (Real.log 20 / Real.log 2) = -4 / 5 :=
by
  sorry

end solve_f_log2_20_l225_225300


namespace find_m_l225_225310

-- Definitions for the given vectors
def vector_a : ℝ × ℝ := (-2, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (4, m)

-- The condition that (vector_a + 2 * vector_b) is parallel to (vector_a - vector_b)
def parallel_condition (m : ℝ) : Prop :=
  let left_vec := (vector_a.1 + 2 * 4, vector_a.2 + 2 * m)
  let right_vec := (vector_a.1 - 4, vector_a.2 - m)
  left_vec.1 * right_vec.2 - right_vec.1 * left_vec.2 = 0

-- The main theorem to prove
theorem find_m : ∃ m : ℝ, parallel_condition m ∧ m = -6 := 
sorry

end find_m_l225_225310


namespace long_side_length_l225_225686

variable {a b d : ℝ}

theorem long_side_length (h1 : a / b = 2 * (b / d)) (h2 : a = 4) (hd : d = Real.sqrt (a^2 + b^2)) :
  b = Real.sqrt (2 + 4 * Real.sqrt 17) :=
sorry

end long_side_length_l225_225686


namespace nominal_rate_of_interest_l225_225481

theorem nominal_rate_of_interest
  (EAR : ℝ)
  (n : ℕ)
  (h_EAR : EAR = 0.0609)
  (h_n : n = 2) :
  ∃ i : ℝ, (1 + i / n)^n - 1 = EAR ∧ i = 0.059 := 
by 
  sorry

end nominal_rate_of_interest_l225_225481


namespace color_property_l225_225844

theorem color_property (k : ℕ) (h : k ≥ 1) : k = 1 ∨ k = 2 :=
by
  sorry

end color_property_l225_225844


namespace triangle_area_proof_l225_225720

noncomputable def area_of_triangle (a b c : ℝ) (A B C : ℝ) : ℝ := 
  1 / 2 * a * c * Real.sin B

theorem triangle_area_proof (a b c : ℝ) (A B C : ℝ) (h1 : b = 3) 
  (h2 : Real.cos B = 1 / 4) 
  (h3 : Real.sin C = 2 * Real.sin A) 
  (h4 : c = 2 * a) 
  (h5 : 9 = 5 * a ^ 2 - 4 * a ^ 2 * Real.cos B): 
  area_of_triangle a b c A B C = 9 * Real.sqrt 15 / 16 :=
by 
  sorry

end triangle_area_proof_l225_225720


namespace sales_volume_conditions_l225_225549

noncomputable def sales_volume (x : ℝ) (a k : ℝ) : ℝ :=
if 1 < x ∧ x ≤ 3 then a * (x - 4)^2 + 6 / (x - 1)
else if 3 < x ∧ x ≤ 5 then k * x + 7
else 0

theorem sales_volume_conditions (a k : ℝ) :
  (sales_volume 3 a k = 4) ∧ (sales_volume 5 a k = 2) ∧
  ((∃ x, 1 < x ∧ x ≤ 3 ∧ sales_volume x a k = 10) ∨ 
   (∃ x, 3 < x ∧ x ≤ 5 ∧ sales_volume x a k = 9)) :=
sorry

end sales_volume_conditions_l225_225549


namespace exists_unique_i_l225_225613

-- Let p be an odd prime number.
variable {p : ℕ} [Fact (Nat.Prime p)] (odd_prime : p % 2 = 1)

-- Let a be an integer in the sequence {2, 3, 4, ..., p-3, p-2}
variable (a : ℕ) (a_range : 2 ≤ a ∧ a ≤ p - 2)

-- Prove that there exists a unique i such that i * a ≡ 1 (mod p) and i ≠ a
theorem exists_unique_i (h1 : ∀ k, 1 ≤ k ∧ k ≤ p - 1 → Nat.gcd k p = 1) :
  ∃! (i : ℕ), 1 ≤ i ∧ i ≤ p - 1 ∧ i * a % p = 1 ∧ i ≠ a :=
by 
  sorry

end exists_unique_i_l225_225613


namespace shop_conditions_l225_225602

theorem shop_conditions (x y : ℕ) :
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) ↔ 
  ∃ x y : ℕ, 7 * x + 7 = y ∧ 9 * (x - 1) = y :=
sorry

end shop_conditions_l225_225602


namespace right_triangle_345_l225_225797

theorem right_triangle_345 : 
  (∃ a b c : ℕ, a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2) ∧ 
  ¬(∃ a b c : ℕ, a = 2 ∧ b = 3 ∧ c = 4 ∧ a^2 + b^2 = c^2) ∧ 
  ¬(∃ a b c : ℕ, a = 4 ∧ b = 5 ∧ c = 6 ∧ a^2 + b^2 = c^2) ∧ 
  ¬(∃ a b c : ℕ, a = 6 ∧ b = 8 ∧ c = 9 ∧ a^2 + b^2 = c^2) :=
by {
  sorry
}

end right_triangle_345_l225_225797


namespace tan_product_pi_nine_l225_225252

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end tan_product_pi_nine_l225_225252


namespace tan_product_pi_nine_l225_225251

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end tan_product_pi_nine_l225_225251


namespace sum_geometric_sequence_l225_225083

theorem sum_geometric_sequence (a r : ℝ) (n : ℕ) (h_a : a = 1/3) (h_r : r = 1/3) (h_n : n = 8) :
  let S_n := a * (1 - r^n) / (1 - r) in S_n = 3280/6561 :=
by
  sorry

end sum_geometric_sequence_l225_225083


namespace smallest_four_digit_multiple_of_17_l225_225855

theorem smallest_four_digit_multiple_of_17 : ∃ n, n ≥ 1000 ∧ n < 10000 ∧ 17 ∣ n ∧ ∀ m, m ≥ 1000 ∧ m < 10000 ∧ 17 ∣ m → n ≤ m := 
by
  use 1003
  sorry

end smallest_four_digit_multiple_of_17_l225_225855


namespace third_studio_students_l225_225643

theorem third_studio_students 
  (total_students : ℕ)
  (first_studio : ℕ)
  (second_studio : ℕ) 
  (third_studio : ℕ) 
  (h1 : total_students = 376) 
  (h2 : first_studio = 110) 
  (h3 : second_studio = 135) 
  (h4 : total_students = first_studio + second_studio + third_studio) :
  third_studio = 131 := 
sorry

end third_studio_students_l225_225643


namespace rectangle_area_eq_2a_squared_l225_225894

variable {α : Type} [Semiring α] (a : α)

-- Conditions
def width (a : α) : α := a
def length (a : α) : α := 2 * a

-- Proof statement
theorem rectangle_area_eq_2a_squared (a : α) : (length a) * (width a) = 2 * a^2 := 
sorry

end rectangle_area_eq_2a_squared_l225_225894


namespace minimize_theta_l225_225426

theorem minimize_theta (K : ℤ) : ∃ θ : ℝ, -495 = K * 360 + θ ∧ |θ| ≤ 180 ∧ θ = -135 :=
by
  sorry

end minimize_theta_l225_225426


namespace infinite_solutions_iff_m_eq_2_l225_225990

theorem infinite_solutions_iff_m_eq_2 (m x y : ℝ) :
  (m*x + 4*y = m + 2 ∧ x + m*y = m) ↔ (m = 2) ∧ (m > 1) :=
by
  sorry

end infinite_solutions_iff_m_eq_2_l225_225990


namespace solve_inequality_l225_225359

-- Defining the inequality
def inequality (x : ℝ) : Prop := 1 / (x - 1) ≤ 1

-- Stating the theorem
theorem solve_inequality :
  { x : ℝ | inequality x } = { x : ℝ | x < 1 } ∪ { x : ℝ | 2 ≤ x } :=
by
  sorry

end solve_inequality_l225_225359


namespace solve_for_t_l225_225886

theorem solve_for_t (s t : ℝ) (h1 : 12 * s + 8 * t = 160) (h2 : s = t^2 + 2) :
  t = (Real.sqrt 103 - 1) / 3 :=
sorry

end solve_for_t_l225_225886


namespace JulioHasMoreSoda_l225_225757

-- Define the number of bottles each person has
def JulioOrangeBottles : ℕ := 4
def JulioGrapeBottles : ℕ := 7
def MateoOrangeBottles : ℕ := 1
def MateoGrapeBottles : ℕ := 3

-- Define the volume of each bottle in liters
def BottleVolume : ℕ := 2

-- Define the total liters of soda each person has
def JulioTotalLiters : ℕ := JulioOrangeBottles * BottleVolume + JulioGrapeBottles * BottleVolume
def MateoTotalLiters : ℕ := MateoOrangeBottles * BottleVolume + MateoGrapeBottles * BottleVolume

-- Prove the difference in total liters of soda between Julio and Mateo
theorem JulioHasMoreSoda : JulioTotalLiters - MateoTotalLiters = 14 := by
  sorry

end JulioHasMoreSoda_l225_225757


namespace other_root_l225_225734

theorem other_root (m n : ℝ) (h : (3 : ℂ) + (1 : ℂ) * Complex.I ∈ {x : ℂ | x^2 + ↑m * x + ↑n = 0}) : 
    (3 : ℂ) - (1 : ℂ) * Complex.I ∈ {x : ℂ | x^2 + ↑m * x + ↑n = 0} :=
sorry

end other_root_l225_225734


namespace meal_total_cost_l225_225292

theorem meal_total_cost (x : ℝ) (h_initial: x/5 - 15 = x/8) : x = 200 :=
by sorry

end meal_total_cost_l225_225292


namespace tan_product_l225_225260

noncomputable def tan_pi_nine := Real.tan (Real.pi / 9)
noncomputable def tan_two_pi_nine := Real.tan (2 * Real.pi / 9)
noncomputable def tan_four_pi_nine := Real.tan (4 * Real.pi / 9)

-- Key polynomial condition
def poly := λ t : ℝ, t^8 - 36 * t^6 + 126 * t^4 - 84 * t^2 + 9

-- Condition: the polynomial has roots tan(π/9), tan(2π/9), tan(4π/9)
axiom poly_roots: 
  (poly tan_pi_nine = 0 ∧
   poly tan_two_pi_nine = 0 ∧
   poly tan_four_pi_nine = 0)

-- The proof statement
theorem tan_product : 
  tan_pi_nine * 
  tan_two_pi_nine * 
  tan_four_pi_nine = 3 :=
sorry

end tan_product_l225_225260


namespace sequence1_sixth_seventh_terms_sequence2_sixth_term_sequence3_ninth_tenth_terms_l225_225800

def seq1 (n : ℕ) : ℕ := 2 * (n + 1)
def seq2 (n : ℕ) : ℕ := 3 * 2 ^ n
def seq3 (n : ℕ) : ℕ :=
  if n % 2 = 0 then 36 + n
  else 10 + n
  
theorem sequence1_sixth_seventh_terms :
  seq1 5 = 12 ∧ seq1 6 = 14 :=
by
  sorry

theorem sequence2_sixth_term :
  seq2 5 = 96 :=
by
  sorry

theorem sequence3_ninth_tenth_terms :
  seq3 8 = 44 ∧ seq3 9 = 19 :=
by
  sorry

end sequence1_sixth_seventh_terms_sequence2_sixth_term_sequence3_ninth_tenth_terms_l225_225800


namespace gcd_digit_bound_l225_225591

theorem gcd_digit_bound {a b : ℕ} (ha : 10^6 ≤ a ∧ a < 10^7) (hb : 10^6 ≤ b ∧ b < 10^7) (hlcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 10^3 :=
by
  sorry

end gcd_digit_bound_l225_225591


namespace brownie_leftover_is_zero_l225_225327

-- Define the dimensions of the pan
def pan_length : ℕ := 24
def pan_width : ℕ := 15

-- Define the dimensions of one piece of brownie
def piece_length : ℕ := 3
def piece_width : ℕ := 4

-- The total area of the pan
def pan_area : ℕ := pan_length * pan_width

-- The total area of one piece
def piece_area : ℕ := piece_length * piece_width

-- The number of full pieces that can be cut
def number_of_pieces : ℕ := pan_area / piece_area

-- The total used area when pieces are cut
def used_area : ℕ := number_of_pieces * piece_area

-- The leftover area
def leftover_area : ℕ := pan_area - used_area

theorem brownie_leftover_is_zero (pan_length pan_width piece_length piece_width : ℕ)
  (h1 : pan_length = 24) (h2 : pan_width = 15) 
  (h3 : piece_length = 3) (h4 : piece_width = 4) :
  pan_width * pan_length - (pan_width * pan_length / (piece_width * piece_length)) * (piece_width * piece_length) = 0 := 
by sorry

end brownie_leftover_is_zero_l225_225327


namespace arithmetic_seq_a7_constant_l225_225436

variable {α : Type*} [AddCommGroup α] [Module ℤ α]

-- Definition of an arithmetic sequence
def is_arithmetic_seq (a : ℕ → α) : Prop :=
∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

-- Given arithmetic sequence {a_n}
variable (a : ℕ → α)
-- Given the property that a_2 + a_4 + a_{15} is a constant
variable (C : α)
variable (h : is_arithmetic_seq a)
variable (h_constant : a 2 + a 4 + a 15 = C)

-- Prove that a_7 is a constant
theorem arithmetic_seq_a7_constant (h : is_arithmetic_seq a) (h_constant : a 2 + a 4 + a 15 = C) : ∃ k : α, a 7 = k :=
by
  sorry

end arithmetic_seq_a7_constant_l225_225436


namespace cos_75_degree_l225_225416

theorem cos_75_degree (cos : ℝ → ℝ) (sin : ℝ → ℝ) :
    cos 75 = (Real.sqrt 6 - Real.sqrt 2) / 4 :=
by
  sorry

end cos_75_degree_l225_225416


namespace identify_quadratic_equation_l225_225038

theorem identify_quadratic_equation :
  (∀ b c d : Prop, ∀ (f : ℕ → Prop), f 0 → ¬ f 1 → ¬ f 2 → ¬ f 3 → b ∧ ¬ c ∧ ¬ d) →
  (∀ x y : ℝ,  (x^2 + 2 = 0) = (b ∧ ¬ b → c ∧ ¬ c → d ∧ ¬ d)) :=
by
  intros;
  sorry

end identify_quadratic_equation_l225_225038


namespace max_successful_free_throws_l225_225956

theorem max_successful_free_throws (a b : ℕ) 
  (h1 : a + b = 105) 
  (h2 : a > 0)
  (h3 : b > 0)
  (ha : a % 3 = 0)
  (hb : b % 5 = 0)
  : (a / 3 + 3 * (b / 5)) ≤ 59 := sorry

end max_successful_free_throws_l225_225956


namespace gasoline_distribution_impossible_l225_225608

theorem gasoline_distribution_impossible
  (x1 x2 x3 : ℝ)
  (h1 : x1 + x2 + x3 = 50)
  (h2 : x1 = x2 + 10)
  (h3 : x3 + 26 = x2) : false :=
by {
  sorry
}

end gasoline_distribution_impossible_l225_225608


namespace part_a_part_b_part_c_l225_225205

-- Part (a)
theorem part_a : ∃ a b, a * b = 80 ∧ (a = 8 ∨ a = 4) ∧ (b = 10 ∨ b = 5) :=
by sorry

-- Part (b)
theorem part_b : ∃ a b c, (a * b) / c = 50 ∧ (a = 10 ∨ a = 5) ∧ (b = 10 ∨ b = 5) ∧ (c = 2 ∨ c = 1) :=
by sorry

-- Part (c)
theorem part_c : ∃ n, n = 4 ∧ ∀ a b c, (a + b) / c = 23 :=
by sorry

end part_a_part_b_part_c_l225_225205


namespace each_nap_duration_l225_225960

-- Definitions based on the problem conditions
def BillProjectDurationInDays : ℕ := 4
def HoursPerDay : ℕ := 24
def TotalProjectHours : ℕ := BillProjectDurationInDays * HoursPerDay
def WorkHours : ℕ := 54
def NapsTaken : ℕ := 6

-- Calculate the time spent on naps and the duration of each nap
def NapHoursTotal : ℕ := TotalProjectHours - WorkHours
def DurationEachNap : ℕ := NapHoursTotal / NapsTaken

-- The theorem stating the expected answer
theorem each_nap_duration :
  DurationEachNap = 7 := by
  sorry

end each_nap_duration_l225_225960


namespace coins_player_1_received_l225_225180

def round_table := List Nat
def players := List Nat
def coins_received (table: round_table) (player_idx: Nat) : Nat :=
sorry -- the function to calculate coins received by player's index

-- Define the given conditions
def sectors : round_table := [1, 2, 3, 4, 5, 6, 7, 8, 9]
def num_players := 9
def num_rotations := 11
def player_4 := 4
def player_8 := 8
def player_1 := 1
def coins_player_4 := 90
def coins_player_8 := 35

theorem coins_player_1_received : coins_received sectors player_1 = 57 :=
by
  -- Setup the conditions
  have h1 : coins_received sectors player_4 = 90 := sorry
  have h2 : coins_received sectors player_8 = 35 := sorry
  -- Prove the target statement
  show coins_received sectors player_1 = 57
  sorry

end coins_player_1_received_l225_225180


namespace quadratic_roots_relation_l225_225796

variable (a b c X1 X2 : ℝ)

theorem quadratic_roots_relation (h : a ≠ 0) : 
  (X1 + X2 = -b / a) ∧ (X1 * X2 = c / a) :=
sorry

end quadratic_roots_relation_l225_225796


namespace tan_product_pi_nine_l225_225253

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end tan_product_pi_nine_l225_225253


namespace find_S6_l225_225614

noncomputable def geometric_series_nth_term (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * q^(n - 1)

noncomputable def geometric_series_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a1 * n else a1 * (1 - q^n) / (1 - q)

variables (a2 q : ℝ)

-- Conditions
axiom a_n_pos : ∀ n, n > 0 → geometric_series_nth_term a2 q n > 0
axiom q_gt_one : q > 1
axiom condition1 : geometric_series_nth_term a2 q 3 + geometric_series_nth_term a2 q 5 = 20
axiom condition2 : geometric_series_nth_term a2 q 2 * geometric_series_nth_term a2 q 6 = 64

-- Question/statement of the theorem
theorem find_S6 : geometric_series_sum 1 q 6 = 63 :=
  sorry

end find_S6_l225_225614


namespace chimps_moved_l225_225361

theorem chimps_moved (total_chimps : ℕ) (chimps_staying : ℕ) (chimps_moved : ℕ) 
  (h_total : total_chimps = 45)
  (h_staying : chimps_staying = 27) :
  chimps_moved = 18 :=
by
  sorry

end chimps_moved_l225_225361


namespace buttons_on_first_type_of_shirt_l225_225919

/--
The GooGoo brand of clothing manufactures two types of shirts.
- The first type of shirt has \( x \) buttons.
- The second type of shirt has 5 buttons.
- The department store ordered 200 shirts of each type.
- A total of 1600 buttons are used for the entire order.

Prove that the first type of shirt has exactly 3 buttons.
-/
theorem buttons_on_first_type_of_shirt (x : ℕ) 
  (h1 : 200 * x + 200 * 5 = 1600) : 
  x = 3 :=
  sorry

end buttons_on_first_type_of_shirt_l225_225919


namespace slope_tangent_line_at_pi_over_3_l225_225637

-- Define the function y = sin(3x)
def f (x : ℝ) := Real.sin (3 * x)

-- Define the derivative of the function
def f' (x : ℝ) := 3 * Real.cos (3 * x)

theorem slope_tangent_line_at_pi_over_3 :
  f' (Real.pi / 3) = -3 :=
by
  -- proof steps will be filled here
  sorry

end slope_tangent_line_at_pi_over_3_l225_225637


namespace cars_sold_first_day_l225_225053

theorem cars_sold_first_day (c_2 c_3 : ℕ) (total : ℕ) (h1 : c_2 = 16) (h2 : c_3 = 27) (h3 : total = 57) :
  ∃ c_1 : ℕ, c_1 + c_2 + c_3 = total ∧ c_1 = 14 :=
by
  sorry

end cars_sold_first_day_l225_225053


namespace gcd_max_two_digits_l225_225586

theorem gcd_max_two_digits (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (h3 : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 100 :=
by
  -- Definitions
  sorry

end gcd_max_two_digits_l225_225586


namespace fg_value_correct_l225_225360

def f_table (x : ℕ) : ℕ :=
  if x = 1 then 3
  else if x = 3 then 7
  else if x = 5 then 9
  else if x = 7 then 13
  else if x = 9 then 17
  else 0  -- Default value to handle unexpected inputs

def g_table (x : ℕ) : ℕ :=
  if x = 1 then 54
  else if x = 3 then 9
  else if x = 5 then 25
  else if x = 7 then 19
  else if x = 9 then 44
  else 0  -- Default value to handle unexpected inputs

theorem fg_value_correct : f_table (g_table 3) = 17 := 
by sorry

end fg_value_correct_l225_225360


namespace distance_house_to_market_l225_225406

-- Define each of the given conditions
def distance_to_school := 50
def distance_to_park_from_school := 25
def return_distance := 60
def total_distance_walked := 220

-- Proven distance to the market
def distance_to_market := 85

-- Statement to prove
theorem distance_house_to_market (d1 d2 d3 d4 : ℕ) 
  (h1 : d1 = distance_to_school) 
  (h2 : d2 = distance_to_park_from_school) 
  (h3 : d3 = return_distance) 
  (h4 : d4 = total_distance_walked) :
  d4 - (d1 + d2 + d3) = distance_to_market := 
by
  sorry

end distance_house_to_market_l225_225406


namespace simplify_expression_l225_225838

theorem simplify_expression (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -3) :
    (x + 2 - 5 / (x - 2)) / ((x + 3) / (x - 2)) = x - 3 :=
sorry

end simplify_expression_l225_225838


namespace find_S10_l225_225761

noncomputable def S (n : ℕ) : ℤ := 2 * (-2 ^ (n - 1)) + 1

theorem find_S10 : S 10 = -1023 :=
by
  sorry

end find_S10_l225_225761


namespace binomial_10_2_equals_45_l225_225685

open Nat

theorem binomial_10_2_equals_45 : Nat.choose 10 2 = 45 := 
by
  sorry

end binomial_10_2_equals_45_l225_225685


namespace remainder_of_expression_l225_225200

theorem remainder_of_expression (n : ℤ) (h : n % 8 = 3) : (4 * n - 10) % 8 = 2 :=
sorry

end remainder_of_expression_l225_225200


namespace binomial_10_2_l225_225681

noncomputable def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binomial_10_2 : binom 10 2 = 45 := by
  sorry

end binomial_10_2_l225_225681


namespace custom_operator_example_l225_225730

def custom_operator (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

theorem custom_operator_example : custom_operator 5 3 = 4 := by
  sorry

end custom_operator_example_l225_225730


namespace vertex_on_x_axis_l225_225514

theorem vertex_on_x_axis (m : ℝ) : 
  (∃ x : ℝ, x^2 - 8 * x + m = 0) ↔ m = 16 :=
by
  sorry

end vertex_on_x_axis_l225_225514


namespace supermarket_sold_54_pints_l225_225662

theorem supermarket_sold_54_pints (x s : ℝ) 
  (h1 : x * s = 216)
  (h2 : x * (s + 2) = 324) : 
  x = 54 := 
by 
  sorry

end supermarket_sold_54_pints_l225_225662


namespace average_minutes_per_day_l225_225408

theorem average_minutes_per_day
  (g : ℕ) -- number of fifth graders
  (h1 : ℕ) -- number of fourth graders
  (h2 : ℕ) -- number of sixth graders
  (h1_eq : h1 = 3 * g) -- Fourth graders are three times fifth graders
  (h2_eq : h2 = g) -- Sixth graders are equal to fifth graders
  (average_fourth : ℚ := 18) -- Average minutes ran by fourth graders
  (average_fifth : ℚ := 12) -- Average minutes ran by fifth graders
  (average_sixth : ℚ := 9) -- Average minutes ran by sixth graders
:
  ((h1 * average_fourth + g * average_fifth + h2 * average_sixth) / (h1 + g + h2)) = 15
:=
  sorry

end average_minutes_per_day_l225_225408


namespace factorize_1_factorize_2_factorize_3_l225_225073

theorem factorize_1 (x : ℝ) : x^4 - 9*x^2 = x^2 * (x + 3) * (x - 3) :=
sorry

theorem factorize_2 (x y : ℝ) : 25*x^2*y + 20*x*y^2 + 4*y^3 = y * (5*x + 2*y)^2 :=
sorry

theorem factorize_3 (x y a : ℝ) : x^2 * (a - 1) + y^2 * (1 - a) = (a - 1) * (x + y) * (x - y) :=
sorry

end factorize_1_factorize_2_factorize_3_l225_225073


namespace justin_additional_time_l225_225144

theorem justin_additional_time (classmates : ℕ) (gathering_hours : ℕ) (minutes_per_flower : ℕ) 
  (flowers_lost : ℕ) : gathering_hours = 2 →
  minutes_per_flower = 10 →
  flowers_lost = 3 →
  classmates = 30 →
  let flowers_gathered := (gathering_hours * 60) / minutes_per_flower in
  let flowers_remaining := flowers_gathered - flowers_lost in
  let flowers_needed := classmates - flowers_remaining in
  let additional_time := flowers_needed * minutes_per_flower in
  additional_time = 210 :=
begin
  intros,
  unfold flowers_gathered flowers_remaining flowers_needed additional_time,
  rw [gathering_hours_eq, minutes_per_flower_eq, flowers_lost_eq, classmates_eq],
  norm_num,
end

end justin_additional_time_l225_225144


namespace hours_felt_good_l225_225958

variable (x : ℝ)

theorem hours_felt_good (h1 : 15 * x + 10 * (8 - x) = 100) : x == 4 := 
by
  sorry

end hours_felt_good_l225_225958


namespace min_beans_l225_225560

theorem min_beans (r b : ℕ) (H1 : r ≥ 3 + 2 * b) (H2 : r ≤ 3 * b) : b ≥ 3 := 
sorry

end min_beans_l225_225560


namespace probability_of_making_pro_shot_l225_225804

-- Define the probabilities given in the problem
def P_free_throw : ℚ := 4 / 5
def P_high_school_3 : ℚ := 1 / 2
def P_at_least_one : ℚ := 0.9333333333333333

-- Define the unknown probability for professional 3-pointer
def P_pro := 1 / 3

-- Calculate the probability of missing each shot
def P_miss_free_throw : ℚ := 1 - P_free_throw
def P_miss_high_school_3 : ℚ := 1 - P_high_school_3
def P_miss_pro : ℚ := 1 - P_pro

-- Define the probability of missing all shots
def P_miss_all := P_miss_free_throw * P_miss_high_school_3 * P_miss_pro

-- Now state what needs to be proved
theorem probability_of_making_pro_shot :
  (1 - P_miss_all = P_at_least_one) → P_pro = 1 / 3 :=
by
  sorry

end probability_of_making_pro_shot_l225_225804


namespace check_line_properties_l225_225714

-- Define the conditions
def line_equation (x y : ℝ) : Prop := y + 7 = -x - 3

-- Define the point and slope
def point_and_slope (x y : ℝ) (m : ℝ) : Prop := (x, y) = (-3, -7) ∧ m = -1

-- State the theorem to prove
theorem check_line_properties :
  ∃ x y m, line_equation x y ∧ point_and_slope x y m :=
sorry

end check_line_properties_l225_225714


namespace geometric_series_sixth_term_l225_225569

theorem geometric_series_sixth_term :
  ∃ r : ℝ, r > 0 ∧ (16 * r^7 = 11664) ∧ (16 * r^5 = 3888) :=
by 
  sorry

end geometric_series_sixth_term_l225_225569


namespace count_two_digit_perfect_squares_divisible_by_4_l225_225115

-- Define the range of integers we are interested in
def two_digit_perfect_squares_divisible_by_4 : List Nat :=
  [4, 5, 6, 7, 8, 9].filter (λ n => (n * n >= 10) ∧ (n * n < 100) ∧ ((n * n) % 4 = 0))

-- Statement of the math proof problem
theorem count_two_digit_perfect_squares_divisible_by_4 :
  two_digit_perfect_squares_divisible_by_4.length = 3 :=
sorry

end count_two_digit_perfect_squares_divisible_by_4_l225_225115


namespace proposition_p_proposition_q_l225_225621

theorem proposition_p : ∅ ≠ ({∅} : Set (Set Empty)) := by
  sorry

theorem proposition_q (A : Set ℕ) (B : Set (Set ℕ)) (hA : A = {1, 2})
    (hB : B = {x | x ⊆ A}) : A ∈ B := by
  sorry

end proposition_p_proposition_q_l225_225621


namespace greatest_b_l225_225193

theorem greatest_b (b : ℤ) (h : ∀ x : ℝ, x^2 + b * x + 20 ≠ -6) : b = 10 := sorry

end greatest_b_l225_225193


namespace max_rational_sums_is_1250_l225_225352

/-- We define a structure to represent the problem's conditions. -/
structure GridConfiguration where
  grid_rows : Nat
  grid_cols : Nat
  total_numbers : Nat
  rational_count : Nat
  irrational_count : Nat
  (h_grid : grid_rows = 50)
  (h_grid_col : grid_cols = 50)
  (h_total_numbers : total_numbers = 100)
  (h_rational_count : rational_count = 50)
  (h_irrational_count : irrational_count = 50)

/-- We define a function to calculate the number of rational sums in the grid. -/
def max_rational_sums (config : GridConfiguration) : Nat :=
  let x := config.rational_count / 2 -- rational numbers to the left
  let ni := 2 * x * x - 100 * x + 2500
  let rational_sums := 2500 - ni
  rational_sums

/-- The theorem stating the maximum number of rational sums is 1250. -/
theorem max_rational_sums_is_1250 (config : GridConfiguration) : max_rational_sums config = 1250 :=
  sorry

end max_rational_sums_is_1250_l225_225352


namespace ellipse_major_axis_length_l225_225394

-- Conditions
def cylinder_radius : ℝ := 2
def minor_axis (r : ℝ) := 2 * r
def major_axis (minor: ℝ) := minor + 0.6 * minor

-- Problem
theorem ellipse_major_axis_length :
  major_axis (minor_axis cylinder_radius) = 6.4 :=
by
  sorry

end ellipse_major_axis_length_l225_225394


namespace minimum_additional_marbles_l225_225615

theorem minimum_additional_marbles (friends marbles : ℕ) (h_friends : friends = 12) (h_marbles : marbles = 34) : 
  ∃ additional_marbles : ℕ, additional_marbles = 44 :=
by
  -- The formal proof would go here.
  sorry

end minimum_additional_marbles_l225_225615


namespace total_frogs_in_pond_l225_225885

def frogsOnLilyPads : ℕ := 5
def frogsOnLogs : ℕ := 3
def babyFrogsOnRock : ℕ := 2 * 12 -- Two dozen

theorem total_frogs_in_pond : frogsOnLilyPads + frogsOnLogs + babyFrogsOnRock = 32 :=
by
  sorry

end total_frogs_in_pond_l225_225885


namespace smallest_n_for_multiple_of_11_l225_225477

theorem smallest_n_for_multiple_of_11 
  (x y : ℤ) 
  (hx : x ≡ -2 [ZMOD 11]) 
  (hy : y ≡ 2 [ZMOD 11]) : 
  ∃ n : ℕ, n > 0 ∧ (x^2 + x * y + y^2 + n ≡ 0 [ZMOD 11]) ∧ n = 7 :=
sorry

end smallest_n_for_multiple_of_11_l225_225477


namespace probability_sum_is_multiple_of_4_l225_225188

open ProbabilityTheory

noncomputable def SpinnerA := [1, 2, 3]
noncomputable def SpinnerB := [2, 3, 3, 4]

def sumIsMultipleOf4 (a b : ℕ) := (a + b) % 4 = 0

def eventProbability (A B : List ℕ) : ℚ :=
  let outcomes := for a in A, b in B, a + b 
  let favorable := List.countp (λ s, s % 4 = 0) outcomes
  favorable / (A.length * B.length)

theorem probability_sum_is_multiple_of_4 :
  eventProbability SpinnerA SpinnerB = 1 / 4 :=
by
  sorry

end probability_sum_is_multiple_of_4_l225_225188


namespace total_members_in_sports_club_l225_225320

-- Definitions as per the conditions
def B : ℕ := 20 -- number of members who play badminton
def T : ℕ := 23 -- number of members who play tennis
def Both : ℕ := 7 -- number of members who play both badminton and tennis
def Neither : ℕ := 6 -- number of members who do not play either sport

-- Theorem statement to prove the correct answer
theorem total_members_in_sports_club : B + T - Both + Neither = 42 :=
by
  sorry

end total_members_in_sports_club_l225_225320


namespace courier_speeds_correctness_l225_225214

noncomputable def courier_speeds : Prop :=
  ∃ (s1 s2 : ℕ), (s1 * 8 + s2 * 8 = 176) ∧ (s1 = 60 / 5) ∧ (s2 = 60 / 6)

theorem courier_speeds_correctness : courier_speeds :=
by
  sorry

end courier_speeds_correctness_l225_225214


namespace tan_product_l225_225272

theorem tan_product : 
  tan (Real.pi / 9) * tan (2 * Real.pi / 9) * tan (4 * Real.pi / 9) = 3 :=
by
  sorry

end tan_product_l225_225272


namespace range_of_a_l225_225991

-- Definitions from conditions 
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := x > a

-- The Lean statement for the problem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 1 → x ≤ a) → a ≥ 1 :=
by sorry

end range_of_a_l225_225991


namespace dice_sum_less_than_16_l225_225184

open Probability

def dice_sum_probability : ℚ :=
  let outcomes := (finset.range 6).product (finset.range 6).product (finset.range 6)
  let valid_outcomes := outcomes.filter (λ (t : ℕ × ℕ × ℕ), t.1 + t.2.1 + t.2.2 < 16)
  valid_outcomes.card / outcomes.card

theorem dice_sum_less_than_16 : dice_sum_probability = 103 / 108 :=
by
  sorry

end dice_sum_less_than_16_l225_225184


namespace minimum_beta_value_l225_225892

variable (α β : Real)

-- Defining the conditions given in the problem
def sin_alpha_condition : Prop := Real.sin α = -Real.sqrt 2 / 2
def cos_alpha_minus_beta_condition : Prop := Real.cos (α - β) = 1 / 2
def beta_greater_than_zero : Prop := β > 0

-- The theorem to be proven
theorem minimum_beta_value (h1 : sin_alpha_condition α) (h2 : cos_alpha_minus_beta_condition α β) (h3 : beta_greater_than_zero β) : β = Real.pi / 12 := 
sorry

end minimum_beta_value_l225_225892


namespace distance_focus_directrix_parabola_l225_225779

theorem distance_focus_directrix_parabola (p : ℝ) (h : y^2 = 20 * x) : 
  2 * p = 10 :=
by
  -- h represents the given condition y^2 = 20x.
  sorry

end distance_focus_directrix_parabola_l225_225779


namespace measure_of_α_l225_225548

variables (α β : ℝ)
-- Condition 1: α and β are complementary angles
def complementary := α + β = 180

-- Condition 2: Half of angle β is 30° less than α
def half_less_30 := α - (1 / 2) * β = 30

-- Theorem: Measure of angle α
theorem measure_of_α (α β : ℝ) (h1 : complementary α β) (h2 : half_less_30 α β) :
  α = 80 :=
by
  sorry

end measure_of_α_l225_225548


namespace range_of_function_l225_225034

theorem range_of_function : ∀ y : ℝ, ∃ x : ℝ, y = (x^2 + 3*x + 2)/(x^2 + x + 1) :=
by
  sorry

end range_of_function_l225_225034


namespace power_of_2_not_sum_of_consecutive_not_power_of_2_is_sum_of_consecutive_l225_225474

-- Definitions and conditions
def is_power_of_2 (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2^k

def is_sum_of_two_or_more_consecutive_naturals (n : ℕ) : Prop :=
  ∃ (a k : ℕ), k ≥ 2 ∧ n = (k * a) + (k * (k - 1)) / 2

-- Proofs to be stated
theorem power_of_2_not_sum_of_consecutive (n : ℕ) (h : is_power_of_2 n) : ¬ is_sum_of_two_or_more_consecutive_naturals n :=
by
    sorry

theorem not_power_of_2_is_sum_of_consecutive (M : ℕ) (h : ¬ is_power_of_2 M) : is_sum_of_two_or_more_consecutive_naturals M :=
by
    sorry

end power_of_2_not_sum_of_consecutive_not_power_of_2_is_sum_of_consecutive_l225_225474


namespace abs_sum_ge_sqrt_three_over_two_l225_225693

open Real

theorem abs_sum_ge_sqrt_three_over_two
  (a b : ℝ) : (|a| + |b| ≥ 2 / sqrt 3) ∧ (∀ x, |a * sin x + b * sin (2 * x)| ≤ 1) ↔
  (a, b) = (4 / (3 * sqrt 3), 2 / (3 * sqrt 3)) ∨ 
  (a, b) = (-4 / (3 * sqrt 3), -2 / (3 * sqrt 3)) ∨
  (a, b) = (4 / (3 * sqrt 3), -2 / (3 * sqrt 3)) ∨
  (a, b) = (-4 / (3 * sqrt 3), 2 / (3 * sqrt 3)) := 
sorry

end abs_sum_ge_sqrt_three_over_two_l225_225693


namespace greatest_b_value_ineq_l225_225976

theorem greatest_b_value_ineq (b : ℝ) (h : -b^2 + 8 * b - 15 ≥ 0) : b ≤ 5 := 
sorry

end greatest_b_value_ineq_l225_225976


namespace hyperbola_eccentricity_l225_225098

theorem hyperbola_eccentricity (C : Type) (a b c e : ℝ)
  (h_asymptotes : ∀ x : ℝ, (∃ y : ℝ, y = x ∨ y = -x)) :
  a = b ∧ c = Real.sqrt (a^2 + b^2) ∧ e = c / a → e = Real.sqrt 2 := 
by
  sorry

end hyperbola_eccentricity_l225_225098


namespace tire_circumference_l225_225650

/-- 
Given:
1. The tire rotates at 400 revolutions per minute.
2. The car is traveling at a speed of 168 km/h.

Prove that the circumference of the tire is 7 meters.
-/
theorem tire_circumference (rpm : ℕ) (speed_km_h : ℕ) (C : ℕ) 
  (h1 : rpm = 400) 
  (h2 : speed_km_h = 168)
  (h3 : C = 7) : 
  C = (speed_km_h * 1000 / 60) / rpm :=
by
  rw [h1, h2]
  exact h3

end tire_circumference_l225_225650


namespace number_of_valid_codes_l225_225517

theorem number_of_valid_codes : 
  let digits := {1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let valid_codes := { (a, b, c) | a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ a < b ∧ b < c } in
  valid_codes.card = 84 :=
by
  sorry

end number_of_valid_codes_l225_225517


namespace sum_of_digits_of_A15B94_multiple_of_99_l225_225738

theorem sum_of_digits_of_A15B94_multiple_of_99 (A B : ℕ) 
  (hA : A < 10) (hB : B < 10)
  (h_mult_99 : ∃ n : ℕ, (100000 * A + 10000 + 5000 + 100 * B + 90 + 4) = 99 * n) :
  A + B = 8 := 
by
  sorry

end sum_of_digits_of_A15B94_multiple_of_99_l225_225738


namespace a_3_eq_5_l225_225706

variable (a : ℕ → ℕ) -- Defines the arithmetic sequence
variable (S : ℕ → ℕ) -- The sum of the first n terms of the sequence

-- Condition: S_5 = 25
axiom S_5_eq_25 : S 5 = 25

-- Define what it means for S to be the sum of the first n terms of the arithmetic sequence
axiom sum_arith_seq : ∀ n, S n = n * (a 1 + a n) / 2

theorem a_3_eq_5 : a 3 = 5 :=
by
  -- Proof is skipped using sorry
  sorry

end a_3_eq_5_l225_225706


namespace polygon_interior_sum_polygon_angle_ratio_l225_225655

-- Part 1: Number of sides based on the sum of interior angles
theorem polygon_interior_sum (n: ℕ) (h: (n - 2) * 180 = 2340) : n = 15 :=
  sorry

-- Part 2: Number of sides based on the ratio of interior to exterior angles
theorem polygon_angle_ratio (n: ℕ) (exterior_angle: ℕ) (ratio: 13 * exterior_angle + 2 * exterior_angle = 180) : n = 15 :=
  sorry

end polygon_interior_sum_polygon_angle_ratio_l225_225655


namespace borel_sets_cardinality_lebesgue_measurable_sets_cardinality_combined_result_l225_225152

open Cardinal

noncomputable def c := Cardinal.mk ℝ

theorem borel_sets_cardinality : ∀ (c : Cardinal), (c = Cardinal.mk ℝ) → (Cardinal.mk (set.borel ℝ) = c) :=
begin
  intros c hc,
  rw hc, -- Using the fact that \(\mathfrak{c}\) is the cardinality of the real numbers
  sorry
end

theorem lebesgue_measurable_sets_cardinality : ∀ (c : Cardinal), (c = Cardinal.mk ℝ) → (Cardinal.mk (measure_theory.measurable_set ℝ) = 2^c) :=
begin
  intros c hc,
  rw hc, -- Using the fact that \(\mathfrak{c}\) is the cardinality of the real numbers
  sorry
end

-- Combine both results in one conclusive form if necessary
theorem combined_result : ∀ (c : Cardinal), (c = Cardinal.mk ℝ) → ((Cardinal.mk (set.borel ℝ) = c) ∧ (Cardinal.mk (measure_theory.measurable_set ℝ) = 2^c)) :=
begin
  intros c hc,
  split,
  { apply borel_sets_cardinality, exact hc },
  { apply lebesgue_measurable_sets_cardinality, exact hc }
end

end borel_sets_cardinality_lebesgue_measurable_sets_cardinality_combined_result_l225_225152


namespace passed_both_tests_l225_225938

theorem passed_both_tests :
  ∀ (total_students passed_long_jump passed_shot_put failed_both passed_both: ℕ),
  total_students = 50 →
  passed_long_jump = 40 →
  passed_shot_put = 31 →
  failed_both = 4 →
  passed_both + (passed_long_jump - passed_both) + (passed_shot_put - passed_both) + failed_both = total_students →
  passed_both = 25 :=
by
  intros total_students passed_long_jump passed_shot_put failed_both passed_both h1 h2 h3 h4 h5
  -- proof can be skipped using sorry
  sorry

end passed_both_tests_l225_225938


namespace methane_required_l225_225725

def mole_of_methane (moles_of_oxygen : ℕ) : ℕ := 
  if moles_of_oxygen = 2 then 1 else 0

theorem methane_required (moles_of_oxygen : ℕ) : 
  moles_of_oxygen = 2 → mole_of_methane moles_of_oxygen = 1 := 
by 
  intros h
  simp [mole_of_methane, h]

end methane_required_l225_225725


namespace common_difference_l225_225466

theorem common_difference (a : ℕ → ℝ) (d : ℝ) (h_seq : ∀ n, a n = 1 + (n - 1) * d) 
  (h_geom : (a 3) ^ 2 = (a 1) * (a 13)) (h_ne_zero: d ≠ 0) : d = 2 :=
by
  sorry

end common_difference_l225_225466


namespace annual_subscription_cost_l225_225507

theorem annual_subscription_cost :
  (10 * 12) * (1 - 0.2) = 96 :=
by
  sorry

end annual_subscription_cost_l225_225507


namespace odd_primes_pq_division_l225_225000

theorem odd_primes_pq_division (p q : ℕ) (m : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
(hp_odd : ¬Even p) (hq_odd : ¬Even q) (hp_gt_hq : p > q) (hm_pos : 0 < m) : ¬(p * q ∣ m ^ (p - q) + 1) :=
by 
  sorry

end odd_primes_pq_division_l225_225000


namespace triangle_area_l225_225827

theorem triangle_area : 
  ∃ (A : ℝ), A = 12 ∧ (∃ (x_intercept y_intercept : ℝ), 3 * x_intercept + 2 * y_intercept = 12 ∧ x_intercept * y_intercept / 2 = A) :=
by
  sorry

end triangle_area_l225_225827


namespace Larry_sessions_per_day_eq_2_l225_225149

variable (x : ℝ)
variable (sessions_per_day_time : ℝ)
variable (feeding_time_per_day : ℝ)
variable (total_time_per_day : ℝ)

theorem Larry_sessions_per_day_eq_2
  (h1: sessions_per_day_time = 30 * x)
  (h2: feeding_time_per_day = 12)
  (h3: total_time_per_day = 72) :
  x = 2 := by
  sorry

end Larry_sessions_per_day_eq_2_l225_225149


namespace tan_product_identity_l225_225276

noncomputable def tan_pi_over_n (n : ℕ) : ℝ := Real.tan (Real.pi / n)

theorem tan_product_identity :
  tan_pi_over_n 9 * tan_pi_over_n 4.5 * tan_pi_over_n 2.25 = Real.sqrt 3 := 
sorry

end tan_product_identity_l225_225276


namespace arrangement_of_70616_l225_225321

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangement_count (digits : List ℕ) : ℕ :=
  let count := digits.length
  let duplicates := List.length (List.filter (fun x => x = 6) digits)
  factorial count / factorial duplicates

theorem arrangement_of_70616 : arrangement_count [7, 0, 6, 6, 1] = 4 * 12 := by
  -- We need to prove that the number of ways to arrange the digits 7, 0, 6, 6, 1 without starting with 0 is 48
  sorry

end arrangement_of_70616_l225_225321


namespace find_p_l225_225721

theorem find_p (x y : ℝ) (h : y = 1.15 * x * (1 - p / 100)) : p = 15 :=
sorry

end find_p_l225_225721


namespace factorize_polynomial_l225_225534

theorem factorize_polynomial (a b c : ℚ) : 
  b^2 - c^2 + a * (a + 2 * b) = (a + b + c) * (a + b - c) :=
by
  sorry

end factorize_polynomial_l225_225534


namespace limit_sum_perimeters_l225_225841

theorem limit_sum_perimeters (a : ℝ) : ∑' n : ℕ, (4 * a) * (1 / 2) ^ n = 8 * a :=
by sorry

end limit_sum_perimeters_l225_225841


namespace hyperbola_standard_eq_line_eq_AB_l225_225998

noncomputable def fixed_points : (Real × Real) × (Real × Real) := ((-Real.sqrt 2, 0.0), (Real.sqrt 2, 0.0))

def locus_condition (P : Real × Real) (F1 F2 : Real × Real) : Prop :=
  abs (dist P F2 - dist P F1) = 2

def curve_E (P : Real × Real) : Prop :=
  (P.1 < 0) ∧ (P.1 * P.1 - P.2 * P.2 = 1)

theorem hyperbola_standard_eq :
  ∃ P : Real × Real, locus_condition P (fixed_points.1) (fixed_points.2) ↔ curve_E P :=
sorry

def line_intersects_hyperbola (P : Real × Real) (k : Real) : Prop :=
  P.2 = k * P.1 - 1 ∧ curve_E P

def dist_A_B (A B : Real × Real) : Real :=
  dist A B

theorem line_eq_AB :
  ∃ k : Real, k = -Real.sqrt 5 / 2 ∧
              ∃ A B : Real × Real, line_intersects_hyperbola A k ∧ 
              line_intersects_hyperbola B k ∧ 
              dist_A_B A B = 6 * Real.sqrt 3 ∧
              ∀ x y : Real, y = k * x - 1 ↔ x * (Real.sqrt 5/2) + y + 1 = 0 :=
sorry

end hyperbola_standard_eq_line_eq_AB_l225_225998


namespace volume_in_cubic_yards_l225_225811

-- Adding the conditions as definitions
def feet_to_yards : ℝ := 3 -- 3 feet in a yard
def cubic_feet_to_cubic_yards : ℝ := feet_to_yards^3 -- convert to cubic yards
def volume_in_cubic_feet : ℝ := 108 -- volume in cubic feet

-- The theorem to prove the equivalence
theorem volume_in_cubic_yards
  (h1 : feet_to_yards = 3)
  (h2 : volume_in_cubic_feet = 108)
  : (volume_in_cubic_feet / cubic_feet_to_cubic_yards) = 4 := 
sorry

end volume_in_cubic_yards_l225_225811


namespace andrew_bought_6_kg_of_grapes_l225_225666

def rate_grapes := 74
def rate_mangoes := 59
def kg_mangoes := 9
def total_paid := 975

noncomputable def number_of_kg_grapes := 6

theorem andrew_bought_6_kg_of_grapes :
  ∃ G : ℕ, (rate_grapes * G + rate_mangoes * kg_mangoes = total_paid) ∧ G = number_of_kg_grapes := 
by
  sorry

end andrew_bought_6_kg_of_grapes_l225_225666


namespace quadratic_root_2020_l225_225736

theorem quadratic_root_2020 (a b : ℝ) (h₀ : a ≠ 0) (h₁ : a * 2019^2 + b * 2019 - 1 = 0) :
    ∃ x : ℝ, (a * (x - 1)^2 + b * (x - 1) = 1) ∧ x = 2020 :=
by
  sorry

end quadratic_root_2020_l225_225736


namespace red_and_purple_probability_l225_225212

def total_balls : ℕ := 120
def white_balls : ℕ := 30
def green_balls : ℕ := 25
def yellow_balls : ℕ := 24
def red_balls : ℕ := 20
def blue_balls : ℕ := 10
def purple_balls : ℕ := 5
def orange_balls : ℕ := 4
def gray_balls : ℕ := 2

def probability_red_purple : ℚ := 5 / 357

theorem red_and_purple_probability :
  ((red_balls / total_balls) * (purple_balls / (total_balls - 1)) +
  (purple_balls / total_balls) * (red_balls / (total_balls - 1))) = probability_red_purple :=
by
  sorry

end red_and_purple_probability_l225_225212


namespace total_frogs_seen_by_hunter_l225_225877

/-- Hunter saw 5 frogs sitting on lily pads in the pond. -/
def initial_frogs : ℕ := 5

/-- Three more frogs climbed out of the water onto logs floating in the pond. -/
def frogs_on_logs : ℕ := 3

/-- Two dozen baby frogs (24 frogs) hopped onto a big rock jutting out from the pond. -/
def baby_frogs : ℕ := 24

/--
The total number of frogs Hunter saw in the pond.
-/
theorem total_frogs_seen_by_hunter : initial_frogs + frogs_on_logs + baby_frogs = 32 := by
sorry

end total_frogs_seen_by_hunter_l225_225877


namespace evaluate_expression_l225_225689

theorem evaluate_expression : 
  -3 * 5 - (-4 * -2) + (-15 * -3) / 3 = -8 :=
by
  sorry

end evaluate_expression_l225_225689


namespace find_value_l225_225433

def star (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y + 2010

variable (a b : ℝ)

axiom h1 : 3 * a + 5 * b = 1
axiom h2 : 4 * a + 9 * b = -1

theorem find_value : star a b 1 2 = 2010 := 
by 
  sorry

end find_value_l225_225433


namespace zoo_visitors_l225_225162

theorem zoo_visitors (visitors_friday : ℕ) 
  (h1 : 3 * visitors_friday = 3750) :
  visitors_friday = 1250 := 
sorry

end zoo_visitors_l225_225162


namespace statement_D_incorrect_l225_225987

theorem statement_D_incorrect (a b c : ℝ) : a^2 > b^2 ∧ a * b > 0 → ¬(1 / a < 1 / b) :=
by sorry

end statement_D_incorrect_l225_225987


namespace B_completes_remaining_work_in_23_days_l225_225050

noncomputable def A_work_rate : ℝ := 1 / 45
noncomputable def B_work_rate : ℝ := 1 / 40
noncomputable def combined_work_rate : ℝ := A_work_rate + B_work_rate
noncomputable def work_done_together_in_9_days : ℝ := combined_work_rate * 9
noncomputable def remaining_work : ℝ := 1 - work_done_together_in_9_days
noncomputable def days_B_completes_remaining_work : ℝ := remaining_work / B_work_rate

theorem B_completes_remaining_work_in_23_days :
  days_B_completes_remaining_work = 23 :=
by 
  -- Proof omitted - please fill in the proof steps
  sorry

end B_completes_remaining_work_in_23_days_l225_225050


namespace Haleigh_needs_leggings_l225_225445

/-- Haleigh's pet animals -/
def dogs : Nat := 4
def cats : Nat := 3
def legs_per_dog : Nat := 4
def legs_per_cat : Nat := 4
def leggings_per_pair : Nat := 2

/-- The proof statement -/
theorem Haleigh_needs_leggings : (dogs * legs_per_dog + cats * legs_per_cat) / leggings_per_pair = 14 := by
  sorry

end Haleigh_needs_leggings_l225_225445


namespace prove_additional_minutes_needed_l225_225145

-- Assume the given conditions as definitions in Lean 4
def number_of_classmates := 30
def initial_gathering_time := 120   -- in minutes (2 hours)
def time_per_flower := 10           -- in minutes
def flowers_lost := 3

-- Calculate the flowers gathered initially
def initial_flowers_gathered := initial_gathering_time / time_per_flower

-- Calculate flowers remaining after loss
def flowers_remaining := initial_flowers_gathered - flowers_lost

-- Calculate additional flowers needed
def additional_flowers_needed := number_of_classmates - flowers_remaining

-- Therefore, calculate the additional minutes required to gather the remaining flowers
def additional_minutes_needed := additional_flowers_needed * time_per_flower

theorem prove_additional_minutes_needed :
  additional_minutes_needed = 210 :=
by 
  unfold additional_minutes_needed additional_flowers_needed flowers_remaining initial_flowers_gathered
  sorry

end prove_additional_minutes_needed_l225_225145


namespace measure_of_angle_B_l225_225326

theorem measure_of_angle_B 
  (A B C: ℝ)
  (a b c: ℝ)
  (h1: A + B + C = π)
  (h2: B / A = C / B)
  (h3: b^2 - a^2 = a * c) : B = 2 * π / 7 :=
  sorry

end measure_of_angle_B_l225_225326


namespace solution_set_inequality_l225_225176

theorem solution_set_inequality (x : ℝ) : (1 - x) * (2 + x) < 0 ↔ x < -2 ∨ x > 1 :=
by
  -- Proof omitted
  sorry

end solution_set_inequality_l225_225176


namespace ellipse_major_axis_length_l225_225393

-- Conditions
def cylinder_radius : ℝ := 2
def minor_axis (r : ℝ) := 2 * r
def major_axis (minor: ℝ) := minor + 0.6 * minor

-- Problem
theorem ellipse_major_axis_length :
  major_axis (minor_axis cylinder_radius) = 6.4 :=
by
  sorry

end ellipse_major_axis_length_l225_225393


namespace tan_product_l225_225257

noncomputable def tan_pi_nine := Real.tan (Real.pi / 9)
noncomputable def tan_two_pi_nine := Real.tan (2 * Real.pi / 9)
noncomputable def tan_four_pi_nine := Real.tan (4 * Real.pi / 9)

-- Key polynomial condition
def poly := λ t : ℝ, t^8 - 36 * t^6 + 126 * t^4 - 84 * t^2 + 9

-- Condition: the polynomial has roots tan(π/9), tan(2π/9), tan(4π/9)
axiom poly_roots: 
  (poly tan_pi_nine = 0 ∧
   poly tan_two_pi_nine = 0 ∧
   poly tan_four_pi_nine = 0)

-- The proof statement
theorem tan_product : 
  tan_pi_nine * 
  tan_two_pi_nine * 
  tan_four_pi_nine = 3 :=
sorry

end tan_product_l225_225257


namespace last_two_digits_of_7_pow_10_l225_225495

theorem last_two_digits_of_7_pow_10 :
  (7 ^ 10) % 100 = 49 := by
  sorry

end last_two_digits_of_7_pow_10_l225_225495


namespace ceil_floor_sum_l225_225070

theorem ceil_floor_sum :
  (Int.ceil (7 / 3 : ℚ)) + (Int.floor (-7 / 3 : ℚ)) = 0 := 
sorry

end ceil_floor_sum_l225_225070


namespace smallest_denominator_fraction_l225_225695

theorem smallest_denominator_fraction 
  (p q : ℕ) (hp : 0 < p) (hq : 0 < q) 
  (h1 : 99 / 100 < p / q) 
  (h2 : p / q < 100 / 101) :
  p = 199 ∧ q = 201 := 
by 
  sorry

end smallest_denominator_fraction_l225_225695


namespace number_of_valid_b_l225_225698

theorem number_of_valid_b : ∃ (bs : Finset ℂ), bs.card = 2 ∧ ∀ b ∈ bs, ∃ (x : ℂ), (x + b = b^2) :=
by
  sorry

end number_of_valid_b_l225_225698


namespace Jules_height_l225_225910

theorem Jules_height (Ben_initial_height Jules_initial_height Ben_current_height Jules_current_height : ℝ) 
  (h_initial : Ben_initial_height = Jules_initial_height)
  (h_Ben_growth : Ben_current_height = 1.25 * Ben_initial_height)
  (h_Jules_growth : Jules_current_height = Jules_initial_height + (Ben_current_height - Ben_initial_height) / 3)
  (h_Ben_current : Ben_current_height = 75) 
  : Jules_current_height = 65 := 
by
  -- Use the conditions to prove that Jules is now 65 inches tall
  sorry

end Jules_height_l225_225910


namespace transformation_correct_l225_225701

theorem transformation_correct (a b : ℝ) (h : a > b) : 2 * a + 1 > 2 * b + 1 :=
by
  sorry

end transformation_correct_l225_225701


namespace trains_cross_time_l225_225385

def length_train1 := 140 -- in meters
def length_train2 := 160 -- in meters

def speed_train1_kmph := 60 -- in km/h
def speed_train2_kmph := 48 -- in km/h

def kmph_to_mps (speed : ℕ) := speed * 1000 / 3600

def speed_train1_mps := kmph_to_mps speed_train1_kmph
def speed_train2_mps := kmph_to_mps speed_train2_kmph

def relative_speed_mps := speed_train1_mps + speed_train2_mps

def total_length := length_train1 + length_train2

def time_to_cross := total_length / relative_speed_mps

theorem trains_cross_time : time_to_cross = 10 :=
  by sorry

end trains_cross_time_l225_225385


namespace people_dislike_both_radio_and_music_l225_225620

theorem people_dislike_both_radio_and_music :
  let total_people := 1500
  let dislike_radio_percent := 0.35
  let dislike_both_percent := 0.20
  let dislike_radio := dislike_radio_percent * total_people
  let dislike_both := dislike_both_percent * dislike_radio
  dislike_both = 105 :=
by
  sorry

end people_dislike_both_radio_and_music_l225_225620


namespace count_two_digit_perfect_squares_divisible_by_4_l225_225118

-- Define what it means to be a two-digit number perfect square divisible by 4
def two_digit_perfect_squares_divisible_by_4 : List ℕ :=
  [16, 36, 64] -- Manually identified two-digit perfect squares which are divisible by 4

-- 6^2 = 36 and 8^2 = 64 both fit, hypothesis checks are already done manually in solution steps
def valid_two_digit_perfect_squares : List ℕ :=
  [16, 25, 36, 49, 64, 81] -- all two-digit perfect squares

-- Define the theorem statement
theorem count_two_digit_perfect_squares_divisible_by_4 :
  (two_digit_perfect_squares_divisible_by_4.count 16 + 
   two_digit_perfect_squares_divisible_by_4.count 36 +
   two_digit_perfect_squares_divisible_by_4.count 64) = 3 :=
by
  -- Proof would go here, omitted by "sorry"
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l225_225118


namespace john_total_payment_l225_225752

-- Definitions of the conditions
def yearly_cost_first_8_years : ℕ := 10000
def yearly_cost_9_to_18_years : ℕ := 2 * yearly_cost_first_8_years
def university_tuition : ℕ := 250000
def total_cost := (8 * yearly_cost_first_8_years) + (10 * yearly_cost_9_to_18_years) + university_tuition

-- John pays half of the total cost
def johns_total_cost := total_cost / 2

-- Theorem stating the total cost John pays
theorem john_total_payment : johns_total_cost = 265000 := by
  sorry

end john_total_payment_l225_225752


namespace polynomial_factorization_l225_225382

theorem polynomial_factorization : (∀ x : ℤ, x^9 + x^6 + x^3 + 1 = (x^3 + 1) * (x^6 - x^3 + 1)) := by
  intro x
  sorry

end polynomial_factorization_l225_225382


namespace rachel_weight_l225_225915

theorem rachel_weight :
  ∃ R : ℝ, (R + (R + 6) + (R - 15)) / 3 = 72 ∧ R = 75 :=
by
  sorry

end rachel_weight_l225_225915


namespace minimum_value_inequality_l225_225338

variable {x y z : ℝ}
variable (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)

theorem minimum_value_inequality : (x + y + z) * (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ 9 / 2 :=
sorry

end minimum_value_inequality_l225_225338


namespace area_of_triangle_bounded_by_line_and_axes_l225_225829

theorem area_of_triangle_bounded_by_line_and_axes (x y : ℝ) (hx : 3 * x + 2 * y = 12) :
  ∃ (area : ℝ), area = 12 := by
sorry

end area_of_triangle_bounded_by_line_and_axes_l225_225829


namespace min_value_frac_inverse_l225_225994

theorem min_value_frac_inverse (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : 
  (1 / a + 1 / b) >= 2 :=
by
  sorry

end min_value_frac_inverse_l225_225994


namespace primes_dividing_expression_l225_225076

theorem primes_dividing_expression (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) : 
  6 * p * q ∣ p^3 + q^2 + 38 ↔ (p = 3 ∧ (q = 5 ∨ q = 13)) := 
sorry

end primes_dividing_expression_l225_225076


namespace hunter_saw_32_frogs_l225_225881

noncomputable def total_frogs (g1 : ℕ) (g2 : ℕ) (d : ℕ) : ℕ :=
g1 + g2 + d

theorem hunter_saw_32_frogs :
  total_frogs 5 3 (2 * 12) = 32 := by
  sorry

end hunter_saw_32_frogs_l225_225881


namespace baseball_card_value_decrease_l225_225203

theorem baseball_card_value_decrease (V0 : ℝ) (V1 V2 : ℝ) :
  V1 = V0 * 0.5 → V2 = V1 * 0.9 → (V0 - V2) / V0 * 100 = 55 :=
by 
  intros hV1 hV2
  sorry

end baseball_card_value_decrease_l225_225203


namespace geometric_sequence_sum_l225_225081

theorem geometric_sequence_sum :
  let a := (1:ℚ)/3
  let r := (1:ℚ)/3
  let n := 8
  (a * (1 - r^n) / (1 - r)) = 3280 / 6561 :=
by {
  let a := (1:ℚ)/3
  let r := (1:ℚ)/3
  let n := 8
  calc
  (a * (1 - r^n) / (1 - r)) = (1/3 * (1 - (1/3)^8) / (1 - 1/3)) : by rw a; rw r
  ... = 3280 / 6561 : sorry
}

end geometric_sequence_sum_l225_225081


namespace count_two_digit_perfect_squares_divisible_by_4_l225_225112

theorem count_two_digit_perfect_squares_divisible_by_4 :
  {n : ℕ | n ∈ (set.range (λ m, m ^ 2)) ∧ 10 ≤ n ∧ n < 100 ∧ n % 4 = 0}.to_finset.card = 3 :=
by
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l225_225112


namespace sequence_contains_infinitely_many_powers_of_two_l225_225544

theorem sequence_contains_infinitely_many_powers_of_two (a : ℕ → ℕ) (b : ℕ → ℕ) : 
  (∃ a1, a1 % 5 ≠ 0 ∧ a 0 = a1) →
  (∀ n : ℕ, a (n + 1) = a n + b n) →
  (∀ n : ℕ, b n = a n % 10) →
  (∃ n : ℕ, ∃ k : ℕ, 2^k = a n) :=
by
  sorry

end sequence_contains_infinitely_many_powers_of_two_l225_225544


namespace zero_is_multiple_of_every_integer_l225_225379

theorem zero_is_multiple_of_every_integer (x : ℤ) : ∃ n : ℤ, 0 = n * x := by
  use 0
  exact (zero_mul x).symm

end zero_is_multiple_of_every_integer_l225_225379


namespace solution_set_empty_iff_a_in_range_l225_225893

theorem solution_set_empty_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, ¬ (2 * x^2 + a * x + 2 < 0)) ↔ (-4 ≤ a ∧ a ≤ 4) :=
by
  sorry

end solution_set_empty_iff_a_in_range_l225_225893


namespace num_paths_7x6_avoid_3_3_l225_225727

theorem num_paths_7x6_avoid_3_3 : 
  let total_paths := Nat.choose 11 5,
      paths_to_33 := Nat.choose 6 3 * Nat.choose 5 2,
      valid_paths := total_paths - paths_to_33
  valid_paths = 262 :=
by
  let total_paths := Nat.choose 11 5
  let paths_to_33 := Nat.choose 6 3 * Nat.choose 5 2
  let valid_paths := total_paths - paths_to_33
  have h_total : total_paths = 462 := by simp
  have h_invalid : paths_to_33 = 200 := by simp
  rw [h_total, h_invalid]
  norm_num

end num_paths_7x6_avoid_3_3_l225_225727


namespace expression_value_l225_225298

theorem expression_value (a b c d : ℝ) (h1 : a * b = 1) (h2 : c + d = 0) :
  -((a * b) ^ (1/3)) + (c + d).sqrt + 1 = 0 :=
by sorry

end expression_value_l225_225298


namespace sally_bought_48_eggs_l225_225164

-- Define the number of eggs in a dozen
def eggs_in_a_dozen : ℕ := 12

-- Define the number of dozens Sally bought
def dozens_sally_bought : ℕ := 4

-- Define the total number of eggs Sally bought
def total_eggs_sally_bought : ℕ := dozens_sally_bought * eggs_in_a_dozen

-- Theorem stating the number of eggs Sally bought
theorem sally_bought_48_eggs : total_eggs_sally_bought = 48 :=
sorry

end sally_bought_48_eggs_l225_225164


namespace play_area_l225_225060

theorem play_area (posts : ℕ) (space : ℝ) (extra_posts : ℕ) (short_posts long_posts : ℕ) (short_spaces long_spaces : ℕ) 
  (short_length long_length area : ℝ)
  (h1 : posts = 24) 
  (h2 : space = 5)
  (h3 : extra_posts = 6)
  (h4 : long_posts = short_posts + extra_posts)
  (h5 : 2 * short_posts + 2 * long_posts - 4 = posts)
  (h6 : short_spaces = short_posts - 1)
  (h7 : long_spaces = long_posts - 1)
  (h8 : short_length = short_spaces * space)
  (h9 : long_length = long_spaces * space)
  (h10 : area = short_length * long_length) :
  area = 675 := 
sorry

end play_area_l225_225060


namespace tangent_product_eq_three_l225_225264

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end tangent_product_eq_three_l225_225264


namespace quadratic_roots_transformation_l225_225155

theorem quadratic_roots_transformation {a b c r s : ℝ}
  (h1 : r + s = -b / a)
  (h2 : r * s = c / a) :
  (∃ p q : ℝ, p = a * r + 2 * b ∧ q = a * s + 2 * b ∧ 
     (∀ x, x^2 - 3 * b * x + 2 * b^2 + a * c = (x - p) * (x - q))) :=
by
  sorry

end quadratic_roots_transformation_l225_225155


namespace angle_same_terminal_side_l225_225975

theorem angle_same_terminal_side (θ : ℝ) (α : ℝ) 
  (hθ : θ = -950) 
  (hα_range : 0 ≤ α ∧ α ≤ 180) 
  (h_terminal_side : ∃ k : ℤ, θ = α + k * 360) : 
  α = 130 := by
  sorry

end angle_same_terminal_side_l225_225975


namespace perpendicular_line_through_intersection_l225_225859

theorem perpendicular_line_through_intersection :
  ∃ (x y : ℝ), (x + y - 2 = 0) ∧ (3 * x + 2 * y - 5 = 0) ∧ (4 * x - 3 * y - 1 = 0) :=
sorry

end perpendicular_line_through_intersection_l225_225859


namespace vasya_petya_distance_l225_225190

theorem vasya_petya_distance :
  ∀ (D : ℝ), 
    (3 : ℝ) ≠ 0 → (6 : ℝ) ≠ 0 →
    ((D / 3) + (D / 6) = 2.5) →
    ((D / 6) + (D / 3) = 3.5) →
    D = 12 := 
by
  intros D h3 h6 h1 h2
  sorry

end vasya_petya_distance_l225_225190


namespace range_of_independent_variable_l225_225325

theorem range_of_independent_variable (x : ℝ) : 
  (y = 3 / (x + 2)) → (x ≠ -2) :=
by
  -- suppose the function y = 3 / (x + 2) is given
  -- we need to prove x ≠ -2 for the function to be defined
  sorry

end range_of_independent_variable_l225_225325


namespace isosceles_trapezoid_base_ratio_correct_l225_225133

def isosceles_trapezoid_ratio (x y a b : ℝ) : Prop :=
  b = 2 * x ∧ a = 2 * y ∧ a + b = 10 ∧ (y * (Real.sqrt 2 + 1) = 5) →

  (a / b = (2 * (Real.sqrt 2) - 1) / 2)

theorem isosceles_trapezoid_base_ratio_correct: ∃ (x y a b : ℝ), 
  isosceles_trapezoid_ratio x y a b := sorry

end isosceles_trapezoid_base_ratio_correct_l225_225133


namespace cooking_dishes_time_l225_225954

def total_awake_time : ℝ := 16
def work_time : ℝ := 8
def gym_time : ℝ := 2
def bath_time : ℝ := 0.5
def homework_bedtime_time : ℝ := 1
def packing_lunches_time : ℝ := 0.5
def cleaning_time : ℝ := 0.5
def shower_leisure_time : ℝ := 2
def total_allocated_time : ℝ := work_time + gym_time + bath_time + homework_bedtime_time + packing_lunches_time + cleaning_time + shower_leisure_time

theorem cooking_dishes_time : total_awake_time - total_allocated_time = 1.5 := by
  sorry

end cooking_dishes_time_l225_225954


namespace division_decimal_l225_225790

theorem division_decimal (x : ℝ) (h : x = 0.3333): 12 / x = 36 :=
  by
    sorry

end division_decimal_l225_225790


namespace divisibility_by_5_l225_225769

theorem divisibility_by_5 (n : ℕ) (h : 0 < n) : (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
sorry

end divisibility_by_5_l225_225769


namespace two_digit_integers_satisfy_R_n_eq_R_n_plus_2_l225_225293

def R (n : ℕ) : ℕ := 
  let remainders := List.range' 2 11 |>.map (λ k => n % k)
  remainders.sum

theorem two_digit_integers_satisfy_R_n_eq_R_n_plus_2 :
  let two_digit_numbers := List.range' 10 89
  (two_digit_numbers.filter (λ n => R n = R (n + 2))).length = 2 := 
by
  sorry

end two_digit_integers_satisfy_R_n_eq_R_n_plus_2_l225_225293


namespace find_c_l225_225762

variable (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)

theorem find_c (h1 : x = 2.5 * y) (h2 : 2 * y = (c / 100) * x) : c = 80 :=
sorry

end find_c_l225_225762


namespace solve_fraction_x_l225_225794

theorem solve_fraction_x (a b c d : ℤ) (hb : b ≠ 0) (hdc : d + c ≠ 0) 
: (2 * a + (bc - 2 * a * d) / (d + c)) / (b - (bc - 2 * a * d) / (d + c)) = c / d := 
sorry

end solve_fraction_x_l225_225794


namespace fraction_product_l225_225226

theorem fraction_product :
  (3 / 7) * (5 / 8) * (9 / 13) * (11 / 17) = 1485 / 12376 := 
by
  sorry

end fraction_product_l225_225226


namespace regular_14_gon_inequality_l225_225660

noncomputable def side_length_of_regular_14_gon : ℝ := 2 * Real.sin (Real.pi / 14)

theorem regular_14_gon_inequality (a : ℝ) (h : a = side_length_of_regular_14_gon) :
  (2 - a) / (2 * a) > Real.sqrt (3 * Real.cos (Real.pi / 7)) :=
by
  sorry

end regular_14_gon_inequality_l225_225660


namespace find_x_l225_225568

theorem find_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x = 2 + 1/y) (h2 : y = 2 + 1/x) (h3 : x + y = 5) : 
  x = (7 + Real.sqrt 5) / 2 :=
by 
  sorry

end find_x_l225_225568


namespace change_amount_l225_225420

theorem change_amount 
    (tank_capacity : ℕ) 
    (current_fuel : ℕ) 
    (price_per_liter : ℕ) 
    (total_money : ℕ) 
    (full_tank : tank_capacity = 150) 
    (fuel_in_truck : current_fuel = 38) 
    (cost_per_liter : price_per_liter = 3) 
    (money_with_donny : total_money = 350) : 
    total_money - ((tank_capacity - current_fuel) * price_per_liter) = 14 :=
by
sorr

end change_amount_l225_225420


namespace sqrt_inequality_l225_225128

theorem sqrt_inequality (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
  sorry

end sqrt_inequality_l225_225128


namespace degree_of_d_l225_225397

noncomputable def f : Polynomial ℝ := sorry
noncomputable def d : Polynomial ℝ := sorry
noncomputable def q : Polynomial ℝ := sorry
noncomputable def r : Polynomial ℝ := 5 * Polynomial.X^2 + 3 * Polynomial.X - 8

axiom deg_f : f.degree = 15
axiom deg_q : q.degree = 7
axiom deg_r : r.degree = 2
axiom poly_div : f = d * q + r

theorem degree_of_d : d.degree = 8 :=
by
  sorry

end degree_of_d_l225_225397


namespace triangle_area_l225_225824

theorem triangle_area : 
  ∃ (A : ℝ), A = 12 ∧ (∃ (x_intercept y_intercept : ℝ), 3 * x_intercept + 2 * y_intercept = 12 ∧ x_intercept * y_intercept / 2 = A) :=
by
  sorry

end triangle_area_l225_225824


namespace gcd_max_digits_l225_225592

theorem gcd_max_digits {a b : ℕ} (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) 
  (h3 : ∃ k, 10^11 ≤ k ∧ k < 10^{12} ∧ k = lcm a b) : 
  (gcd a b) < 10^3 :=
sorry

end gcd_max_digits_l225_225592


namespace sum_of_first_eight_terms_l225_225087

-- Define the first term, common ratio, and the number of terms
def a : ℚ := 1 / 3
def r : ℚ := 1 / 3
def n : ℕ := 8

-- Sum of the first n terms of a geometric sequence
def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- Proof statement
theorem sum_of_first_eight_terms : geometric_sum a r n = 3280 / 6561 :=
by
  sorry

end sum_of_first_eight_terms_l225_225087


namespace probability_no_shaded_rectangle_l225_225208

-- Definitions
def total_rectangles_per_row : ℕ := (2005 * 2004) / 2
def shaded_rectangles_per_row : ℕ := 1002 * 1002

-- Proposition to prove
theorem probability_no_shaded_rectangle : 
  (1 - (shaded_rectangles_per_row : ℝ) / (total_rectangles_per_row : ℝ)) = (0.25 / 1002.25) := 
sorry

end probability_no_shaded_rectangle_l225_225208


namespace problem_statement_l225_225802

def reading_method (n : ℕ) : String := sorry
-- Assume reading_method correctly implements the reading method for integers

def is_read_with_only_one_zero (n : ℕ) : Prop :=
  (reading_method n).count '0' = 1

theorem problem_statement : is_read_with_only_one_zero 83721000 = false := sorry

end problem_statement_l225_225802


namespace isosceles_triangle_angle_sum_l225_225134

theorem isosceles_triangle_angle_sum (x : ℝ) (h1 : x = 50 ∨ x = 65 ∨ x = 80) : (50 + 65 + 80 = 195) :=
by sorry

end isosceles_triangle_angle_sum_l225_225134


namespace speed_of_each_train_l225_225041

theorem speed_of_each_train (v : ℝ) (train_length time_cross : ℝ) (km_pr_s : ℝ) 
  (h_train_length : train_length = 120)
  (h_time_cross : time_cross = 8)
  (h_km_pr_s : km_pr_s = 3.6)
  (h_relative_speed : 2 * v = (2 * train_length) / time_cross) :
  v * km_pr_s = 54 := 
by sorry

end speed_of_each_train_l225_225041


namespace matrix_power_100_l225_225230

def matrix_100_pow : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![200, 1]]

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![2, 1]]

theorem matrix_power_100 (A : Matrix (Fin 2) (Fin 2) ℤ) :
  A^100 = matrix_100_pow :=
by
  sorry

end matrix_power_100_l225_225230


namespace cube_largest_ne_sum_others_l225_225356

theorem cube_largest_ne_sum_others (n : ℕ) : (n + 1)^3 ≠ n^3 + (n - 1)^3 :=
by
  sorry

end cube_largest_ne_sum_others_l225_225356


namespace triangle_inequality_positive_difference_l225_225748

theorem triangle_inequality_positive_difference (x : ℕ) (h1 : 3 < x) (h2 : x < 17) : 
  (Nat.greatest (4 <= x)) - (Nat.least (x <= 16)) = 12 :=
by
  sorry

end triangle_inequality_positive_difference_l225_225748


namespace number_of_yellow_balls_l225_225657

-- Definitions based on conditions
def number_of_red_balls : ℕ := 10
def probability_red_ball := (1 : ℚ) / 3

-- Theorem stating the number of yellow balls
theorem number_of_yellow_balls :
  ∃ (y : ℕ), (number_of_red_balls : ℚ) / (number_of_red_balls + y) = probability_red_ball ∧ y = 20 :=
by
  sorry

end number_of_yellow_balls_l225_225657


namespace solution_proof_l225_225724

def count_multiples (n : ℕ) (m : ℕ) (limit : ℕ) : ℕ :=
  (limit - 1) / m + 1

def problem_statement : Prop :=
  let multiples_of_10 := count_multiples 1 10 300
  let multiples_of_10_and_6 := count_multiples 1 30 300
  let multiples_of_10_and_11 := count_multiples 1 110 300
  let unwanted_multiples := multiples_of_10_and_6 + multiples_of_10_and_11
  multiples_of_10 - unwanted_multiples = 20

theorem solution_proof : problem_statement :=
  by {
    sorry
  }

end solution_proof_l225_225724


namespace ratio_square_l225_225889

theorem ratio_square (x y : ℕ) (h1 : x * (x + y) = 40) (h2 : y * (x + y) = 90) (h3 : 2 * y = 3 * x) : (x + y) ^ 2 = 100 := 
by 
  sorry

end ratio_square_l225_225889


namespace required_number_of_shirts_l225_225319

/-
In a shop, there is a sale of clothes. Every shirt costs $5, every hat $4, and a pair of jeans $10.
You need to pay $51 for a certain number of shirts, two pairs of jeans, and four hats.
Prove that the number of shirts you need to buy is 3.
-/

def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def jeans_cost : ℕ := 10
def total_payment : ℕ := 51
def number_of_jeans : ℕ := 2
def number_of_hats : ℕ := 4

theorem required_number_of_shirts (S : ℕ) (h : 5 * S + 2 * jeans_cost + 4 * hat_cost = total_payment) : S = 3 :=
by
  -- This statement asserts that given the defined conditions, the number of shirts that satisfies the equation is 3.
  sorry

end required_number_of_shirts_l225_225319


namespace problem_statement_l225_225731

   def f (a : ℤ) : ℤ := a - 2
   def F (a b : ℤ) : ℤ := b^2 + a

   theorem problem_statement : F 3 (f 4) = 7 := by
     sorry
   
end problem_statement_l225_225731


namespace mike_practice_hours_l225_225350

def weekday_practice_hours_per_day : ℕ := 3
def days_per_weekday_practice : ℕ := 5
def saturday_practice_hours : ℕ := 5
def weeks_until_game : ℕ := 3

def total_weekday_practice_hours : ℕ := weekday_practice_hours_per_day * days_per_weekday_practice
def total_weekly_practice_hours : ℕ := total_weekday_practice_hours + saturday_practice_hours
def total_practice_hours : ℕ := total_weekly_practice_hours * weeks_until_game

theorem mike_practice_hours :
  total_practice_hours = 60 := by
  sorry

end mike_practice_hours_l225_225350


namespace area_of_hexagon_l225_225789

theorem area_of_hexagon (c d : ℝ) (a b : ℝ)
  (h₁ : a^2 + b^2 = c^2)
  (h₂ : a + b = d) : 
  (c^2 + d^2 = c^2 + a^2 + b^2 + 2*a*b) :=
by
  sorry

end area_of_hexagon_l225_225789


namespace stanleyRanMore_l225_225627

def distanceStanleyRan : ℝ := 0.4
def distanceStanleyWalked : ℝ := 0.2

theorem stanleyRanMore : distanceStanleyRan - distanceStanleyWalked = 0.2 := by
  sorry

end stanleyRanMore_l225_225627


namespace intersection_at_one_point_l225_225837

theorem intersection_at_one_point (b : ℝ) :
  (∃ x₀ : ℝ, bx^2 + 7*x₀ + 4 = 0 ∧ (7)^2 - 4*b*4 = 0) →
  b = 49 / 16 :=
by
  sorry

end intersection_at_one_point_l225_225837


namespace infinite_rational_points_l225_225020

theorem infinite_rational_points (x y : ℚ) (h_pos : 0 < x ∧ 0 < y) (h_ineq : x + y ≤ 5) : 
  set.infinite { p : ℚ × ℚ | 0 < p.1 ∧ 0 < p.2 ∧ p.1 + p.2 ≤ 5 } :=
sorry

end infinite_rational_points_l225_225020


namespace justin_additional_time_l225_225143

theorem justin_additional_time (classmates : ℕ) (gathering_hours : ℕ) (minutes_per_flower : ℕ) 
  (flowers_lost : ℕ) : gathering_hours = 2 →
  minutes_per_flower = 10 →
  flowers_lost = 3 →
  classmates = 30 →
  let flowers_gathered := (gathering_hours * 60) / minutes_per_flower in
  let flowers_remaining := flowers_gathered - flowers_lost in
  let flowers_needed := classmates - flowers_remaining in
  let additional_time := flowers_needed * minutes_per_flower in
  additional_time = 210 :=
begin
  intros,
  unfold flowers_gathered flowers_remaining flowers_needed additional_time,
  rw [gathering_hours_eq, minutes_per_flower_eq, flowers_lost_eq, classmates_eq],
  norm_num,
end

end justin_additional_time_l225_225143


namespace count_multiples_of_13_three_digit_l225_225561

-- Definitions based on the conditions in the problem
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_multiple_of_13 (n : ℕ) : Prop := ∃ k : ℕ, n = 13 * k

-- Statement of the proof problem
theorem count_multiples_of_13_three_digit :
  ∃ (count : ℕ), count = (76 - 8 + 1) :=
sorry

end count_multiples_of_13_three_digit_l225_225561


namespace child_height_at_last_visit_l225_225946

-- Definitions for the problem
def h_current : ℝ := 41.5 -- current height in inches
def Δh : ℝ := 3 -- height growth in inches

-- The proof statement
theorem child_height_at_last_visit : h_current - Δh = 38.5 := by
  sorry

end child_height_at_last_visit_l225_225946


namespace percentage_failed_both_l225_225132

theorem percentage_failed_both 
    (p_h p_e p_p p_pe : ℝ)
    (h_p_h : p_h = 32)
    (h_p_e : p_e = 56)
    (h_p_p : p_p = 24)
    : p_pe = 12 := by 
    sorry

end percentage_failed_both_l225_225132


namespace bugs_eat_total_flowers_l225_225351

def num_bugs : ℝ := 2.0
def flowers_per_bug : ℝ := 1.5
def total_flowers_eaten : ℝ := 3.0

theorem bugs_eat_total_flowers : 
  (num_bugs * flowers_per_bug) = total_flowers_eaten := 
  by 
    sorry

end bugs_eat_total_flowers_l225_225351


namespace length_of_DE_l225_225750

theorem length_of_DE 
  (area_ABC : ℝ) 
  (area_trapezoid : ℝ) 
  (altitude_ABC : ℝ) 
  (h1 : area_ABC = 144) 
  (h2 : area_trapezoid = 96)
  (h3 : altitude_ABC = 24) :
  ∃ (DE_length : ℝ), DE_length = 2 * Real.sqrt 3 := 
sorry

end length_of_DE_l225_225750


namespace set_diff_N_M_l225_225688

universe u

def set_difference {α : Type u} (A B : Set α) : Set α :=
  { x | x ∈ A ∧ x ∉ B }

def M : Set ℕ := { 1, 2, 3, 4, 5 }
def N : Set ℕ := { 1, 2, 3, 7 }

theorem set_diff_N_M : set_difference N M = { 7 } :=
  by
    sorry

end set_diff_N_M_l225_225688


namespace triangle_ratio_and_angle_l225_225316

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (sinA sinB sinC : ℝ)

theorem triangle_ratio_and_angle
  (h_triangle : a / sinA = b / sinB ∧ b / sinB = c / sinC)
  (h_sin_ratio : sinA / sinB = 5 / 7 ∧ sinB / sinC = 7 / 8) :
  (a / b = 5 / 7 ∧ b / c = 7 / 8) ∧ B = 60 :=
by
  sorry

end triangle_ratio_and_angle_l225_225316


namespace annual_subscription_cost_l225_225510

-- Definitions based on the conditions

def monthly_cost : ℝ := 10
def months_per_year : ℕ := 12
def discount_rate : ℝ := 0.20

-- The statement based on the correct answer
theorem annual_subscription_cost : 
  (monthly_cost * months_per_year) * (1 - discount_rate) = 96 := 
by
  sorry

end annual_subscription_cost_l225_225510


namespace find_modulus_l225_225439

open Complex -- Open the Complex namespace for convenience

noncomputable def modulus_of_z (a : ℝ) (h : (1 + 2 * Complex.I) * (a + Complex.I : ℂ) = Complex.re ((1 + 2 * Complex.I) * (a + Complex.I)) + Complex.im ((1 + 2 * Complex.I) * (a + Complex.I)) * Complex.I) : ℝ :=
  Complex.abs ((1 + 2 * Complex.I) * (a + Complex.I))

theorem find_modulus : modulus_of_z (-3) (by {
  -- Provide the condition that real part equals imaginary part
  admit -- This 'admit' serves as a placeholder for the proof of the condition 
}) = 5 * Real.sqrt 2 := sorry

end find_modulus_l225_225439


namespace primer_cost_before_discount_l225_225669

theorem primer_cost_before_discount (primer_cost_after_discount : ℝ) (paint_cost : ℝ) (total_cost : ℝ) 
  (rooms : ℕ) (primer_discount : ℝ) (paint_cost_per_gallon : ℝ) :
  (primer_cost_after_discount = total_cost - (rooms * paint_cost_per_gallon)) →
  (rooms * (primer_cost - primer_discount * primer_cost) = primer_cost_after_discount) →
  primer_cost = 30 := by
  sorry

end primer_cost_before_discount_l225_225669


namespace integral_value_l225_225746

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the conditions of the problem
def a : ℝ := 2 -- This is derived from the problem condition

-- The main theorem statement
theorem integral_value :
  (∫ x in (0 : ℝ)..a, (Real.exp x + 2 * x)) = Real.exp 2 + 3 := by
  sorry

end integral_value_l225_225746


namespace tan_product_l225_225271

theorem tan_product : 
  tan (Real.pi / 9) * tan (2 * Real.pi / 9) * tan (4 * Real.pi / 9) = 3 :=
by
  sorry

end tan_product_l225_225271


namespace tan_product_pi_nine_l225_225255

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end tan_product_pi_nine_l225_225255


namespace fraction_unchanged_when_increased_by_ten_l225_225733

variable {x y : ℝ}

theorem fraction_unchanged_when_increased_by_ten (x y : ℝ) :
  (5 * (10 * x)) / (10 * x + 10 * y) = 5 * x / (x + y) :=
by
  sorry

end fraction_unchanged_when_increased_by_ten_l225_225733


namespace total_cost_of_shoes_before_discount_l225_225751

theorem total_cost_of_shoes_before_discount (S J H : ℝ) (D : ℝ) (shoes jerseys hats : ℝ) :
  jerseys = 1/4 * shoes ∧
  hats = 2 * jerseys ∧
  D = 0.9 * (6 * shoes + 4 * jerseys + 3 * hats) ∧
  D = 620 →
  6 * shoes = 486.30 := by
  sorry

end total_cost_of_shoes_before_discount_l225_225751


namespace count_two_digit_perfect_squares_divisible_by_4_l225_225116

-- Define the range of integers we are interested in
def two_digit_perfect_squares_divisible_by_4 : List Nat :=
  [4, 5, 6, 7, 8, 9].filter (λ n => (n * n >= 10) ∧ (n * n < 100) ∧ ((n * n) % 4 = 0))

-- Statement of the math proof problem
theorem count_two_digit_perfect_squares_divisible_by_4 :
  two_digit_perfect_squares_divisible_by_4.length = 3 :=
sorry

end count_two_digit_perfect_squares_divisible_by_4_l225_225116


namespace major_axis_length_of_intersecting_ellipse_l225_225396

theorem major_axis_length_of_intersecting_ellipse (radius : ℝ) (h_radius : radius = 2) 
  (minor_axis_length : ℝ) (h_minor_axis : minor_axis_length = 2 * radius) (major_axis_length : ℝ) 
  (h_major_axis : major_axis_length = minor_axis_length * 1.6) :
  major_axis_length = 6.4 :=
by 
  -- The proof will follow here, but currently it's not required.
  sorry

end major_axis_length_of_intersecting_ellipse_l225_225396


namespace interest_rate_b_to_c_l225_225392

open Real

noncomputable def calculate_rate_b_to_c (P : ℝ) (r1 : ℝ) (t : ℝ) (G : ℝ) : ℝ :=
  let I_a_b := P * (r1 / 100) * t
  let I_b_c := I_a_b + G
  (100 * I_b_c) / (P * t)

theorem interest_rate_b_to_c :
  calculate_rate_b_to_c 3200 12 5 400 = 14.5 := by
  sorry

end interest_rate_b_to_c_l225_225392


namespace joel_garden_size_l225_225141

-- Definitions based on the conditions
variable (G : ℕ) -- G is the size of Joel's garden.

-- Condition 1: Half of the garden is for fruits.
def half_garden_fruits (G : ℕ) := G / 2

-- Condition 2: Half of the garden is for vegetables.
def half_garden_vegetables (G : ℕ) := G / 2

-- Condition 3: A quarter of the fruit section is used for strawberries.
def quarter_fruit_section (G : ℕ) := (half_garden_fruits G) / 4

-- Condition 4: The quarter for strawberries takes up 8 square feet.
axiom strawberry_section : quarter_fruit_section G = 8

-- Hypothesis: The size of Joel's garden is 64 square feet.
theorem joel_garden_size : G = 64 :=
by
  -- Insert the logical progression of the proof here.
  sorry

end joel_garden_size_l225_225141


namespace either_x_or_y_is_even_l225_225771

theorem either_x_or_y_is_even (x y z : ℤ) (h : x^2 + y^2 = z^2) : (2 ∣ x) ∨ (2 ∣ y) :=
by
  sorry

end either_x_or_y_is_even_l225_225771


namespace length_AD_l225_225999

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) (hab_perp : (a + b) ⬝ a = 0)
variables (A B C D : EuclideanSpace ℝ (Fin 2))
variables (AB_eq_a : B - A = a) (AC_eq_b : C - A = b)
variables (D_mid_BC : D = (B + C) / 2)

theorem length_AD :
  ‖D - A‖ = (Real.sqrt 3) / 2 :=
sorry

end length_AD_l225_225999


namespace tangent_product_eq_three_l225_225262

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end tangent_product_eq_three_l225_225262


namespace marble_probability_l225_225051

noncomputable def total_marbles : ℕ := 15 + 10 + 5

noncomputable def total_draws : ℕ := 4

noncomputable def total_outcomes : ℕ := (Nat.choose total_marbles total_draws)

noncomputable def red_selections : ℕ := Nat.choose 15 2

noncomputable def blue_selection : ℕ := Nat.choose 10 1

noncomputable def green_selection : ℕ := Nat.choose 5 1

noncomputable def favorable_outcomes : ℕ := red_selections * blue_selection * green_selection

noncomputable def probability : ℚ := favorable_outcomes / total_outcomes

theorem marble_probability :
  probability = 350 / 1827 := 
by 
  sorry

end marble_probability_l225_225051


namespace equation_one_solution_equation_two_solution_l225_225696

theorem equation_one_solution (x : ℕ) : 8 * (x + 1)^3 = 64 ↔ x = 1 := by 
  sorry

theorem equation_two_solution (x : ℤ) : (x + 1)^2 = 100 ↔ x = 9 ∨ x = -11 := by 
  sorry

end equation_one_solution_equation_two_solution_l225_225696


namespace sum_digits_0_to_2012_l225_225334

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

theorem sum_digits_0_to_2012 : ∑ n in Finset.range 2013, sum_of_digits n = 28077 := 
by
  sorry

end sum_digits_0_to_2012_l225_225334


namespace probability_A_greater_B_l225_225411

theorem probability_A_greater_B :
  let A := [10, 10, 1, 1, 1]
  let B := [5, 5, 5, 5, 1, 1, 1]
  let remaining_value (bag : List ℕ) (drawn : List ℕ) :=
    (bag.sum - drawn.sum)
  let valid_pairs := do
    a_drawn ← A.combinations 2
    b_drawn ← B.combinations 2
    guard $ remaining_value A a_drawn > remaining_value B b_drawn
    pure (a_drawn, b_drawn)
  let total_pairs := A.combinations 2.product B.combinations 2
  (valid_pairs.length / total_pairs.length : ℚ) = 9 / 35 :=
by
  sorry

end probability_A_greater_B_l225_225411


namespace carla_chickens_l225_225840

theorem carla_chickens (initial_chickens : ℕ) (percent_died : ℕ) (bought_factor : ℕ) :
  initial_chickens = 400 →
  percent_died = 40 →
  bought_factor = 10 →
  let died := (percent_died * initial_chickens) / 100 in
  let bought := bought_factor * died in
  let total := initial_chickens - died + bought in
  total = 1840 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  let died := (40 * 400) / 100
  have hdied : died = 160 := rfl
  let bought := 10 * died
  have hbought : bought = 1600 := rfl
  let total := 400 - 160 + 1600
  have htotal : total = 1840 := rfl
  exact htotal

end carla_chickens_l225_225840


namespace find_a_l225_225997

open Set

variable (a : ℝ)

def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2 * a^2 - a + 7}
def B (a : ℝ) : Set ℝ := {-4, a + 3, a^2 - 2 * a + 2, a^3 + a^2 + 3 * a + 7}

theorem find_a (h : (A a ∩ B a) = {2, 5}) : a = 2 :=
sorry

end find_a_l225_225997


namespace porter_monthly_earnings_l225_225006

/--
Porter earns $8 per day and works 5 times a week. He is promised an extra
50% on top of his daily rate for an extra day each week. There are 4 weeks in a month.
Prove that Porter will earn $208 in a month if he works the extra day every week.
-/
theorem porter_monthly_earnings :
  let daily_rate := 8
  let days_per_week := 5
  let weeks_per_month := 4
  let overtime_extra_rate := 0.5
  let weekly_earnings := daily_rate * days_per_week
  let monthly_earnings_without_overtime := weekly_earnings * weeks_per_month
  let overtime_earnings_per_day := daily_rate * (1 + overtime_extra_rate)
  let total_overtime_earnings_per_month := overtime_earnings_per_day * weeks_per_month
  in monthly_earnings_without_overtime + total_overtime_earnings_per_month = 208 :=
by
  let daily_rate := 8
  let days_per_week := 5
  let weeks_per_month := 4
  let overtime_extra_rate := 0.5
  let weekly_earnings := daily_rate * days_per_week
  let monthly_earnings_without_overtime := weekly_earnings * weeks_per_month
  let overtime_earnings_per_day := daily_rate * (1 + overtime_extra_rate)
  let total_overtime_earnings_per_month := overtime_earnings_per_day * weeks_per_month
  show Prop, from monthly_earnings_without_overtime + total_overtime_earnings_per_month = 208

end porter_monthly_earnings_l225_225006


namespace evaluate_g_at_neg3_l225_225611

def g (x : ℝ) : ℝ := 3 * x ^ 5 - 5 * x ^ 4 + 7 * x ^ 3 - 10 * x ^ 2 - 12 * x + 36

theorem evaluate_g_at_neg3 : g (-3) = -1341 := by
  sorry

end evaluate_g_at_neg3_l225_225611


namespace hadley_total_distance_l225_225559

def distance_to_grocery := 2
def distance_to_pet_store := 2 - 1
def distance_back_home := 4 - 1

theorem hadley_total_distance : distance_to_grocery + distance_to_pet_store + distance_back_home = 6 :=
by
  -- Proof is omitted.
  sorry

end hadley_total_distance_l225_225559


namespace rope_cut_ratio_l225_225049

theorem rope_cut_ratio (L : ℕ) (a b : ℕ) (hL : L = 40) (ha : a = 2) (hb : b = 3) :
  L / (a + b) * a = 16 :=
by
  sorry

end rope_cut_ratio_l225_225049


namespace smallest_possible_other_integer_l225_225037

theorem smallest_possible_other_integer (n : ℕ) (h1 : Nat.lcm 60 n / Nat.gcd 60 n = 84) : n = 35 :=
sorry

end smallest_possible_other_integer_l225_225037


namespace larger_integer_of_two_integers_diff_8_prod_120_l225_225787

noncomputable def larger_integer (a b : ℕ) : ℕ :=
if a > b then a else b

theorem larger_integer_of_two_integers_diff_8_prod_120 (a b : ℕ) 
  (h_diff : a - b = 8) 
  (h_product : a * b = 120) 
  (h_positive_a : 0 < a) 
  (h_positive_b : 0 < b) : larger_integer a b = 20 := by
  sorry

end larger_integer_of_two_integers_diff_8_prod_120_l225_225787


namespace solve_system_of_equations_l225_225774

theorem solve_system_of_equations (x y : ℝ) (h1 : x + y = 5) (h2 : 2 * x - y = 1) : x = 2 ∧ y = 3 := 
sorry

end solve_system_of_equations_l225_225774


namespace negation_of_implication_l225_225652

theorem negation_of_implication (x : ℝ) :
  (¬ (x = 0 ∨ x = 1) → x^2 - x ≠ 0) ↔ (x ≠ 0 ∧ x ≠ 1 → x^2 - x ≠ 0) :=
by sorry

end negation_of_implication_l225_225652


namespace vertex_on_xaxis_l225_225511

-- Definition of the parabola equation with vertex on the x-axis
def parabola (x m : ℝ) := x^2 - 8 * x + m

-- The problem statement: show that m = 16 given that the vertex of the parabola is on the x-axis
theorem vertex_on_xaxis (m : ℝ) : ∃ x : ℝ, parabola x m = 0 → m = 16 :=
by
  sorry

end vertex_on_xaxis_l225_225511


namespace complement_union_eq_l225_225308

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x ≥ 2}

theorem complement_union_eq :
  U \ (A ∪ B) = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by
  sorry

end complement_union_eq_l225_225308


namespace smallest_m_l225_225927

theorem smallest_m (m : ℕ) (h1 : m > 0) (h2 : 3 ^ ((m + m ^ 2) / 4) > 500) : m = 5 := 
by sorry

end smallest_m_l225_225927


namespace integral_value_l225_225529

noncomputable def integral_sin_pi_over_2_to_pi : ℝ := ∫ x in (Real.pi / 2)..Real.pi, Real.sin x

theorem integral_value : integral_sin_pi_over_2_to_pi = 1 := by
  sorry

end integral_value_l225_225529


namespace fraction_undefined_at_one_l225_225199

theorem fraction_undefined_at_one (x : ℤ) (h : x = 1) : (x / (x - 1) = 1) := by
  have h : 1 / (1 - 1) = 1 := sorry
  sorry

end fraction_undefined_at_one_l225_225199


namespace x1_sufficient_not_necessary_l225_225044

theorem x1_sufficient_not_necessary : (x : ℝ) → (x = 1 ↔ (x - 1) * (x + 2) = 0) ∧ ∀ x, (x = 1 ∨ x = -2) → (x - 1) * (x + 2) = 0 ∧ (∀ y, (y - 1) * (y + 2) = 0 → (y = 1 ∨ y = -2)) :=
by
  sorry

end x1_sufficient_not_necessary_l225_225044


namespace tan_identity_l225_225249

theorem tan_identity :
  tan (π / 9) * tan (2 * π / 9) * tan (4 * π / 9) = 3 :=
sorry

end tan_identity_l225_225249


namespace minimum_perimeter_triangle_MAF_is_11_l225_225922

-- Define point, parabola, and focus
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the specific points in the problem
def A : Point := ⟨5, 3⟩

-- Parabola with the form y^2 = 4x has the focus at (1, 0)
def F : Point := ⟨1, 0⟩

-- Minimum perimeter problem for ΔMAF
noncomputable def minimum_perimeter_triangle_MAF (M : Point) : ℝ :=
  (dist (M.x, M.y) (A.x, A.y)) + (dist (M.x, M.y) (F.x, F.y))

-- The goal is to show the minimum value of the perimeter is 11
theorem minimum_perimeter_triangle_MAF_is_11 (M : Point) 
  (hM_parabola : M.y^2 = 4 * M.x) 
  (hM_not_AF : M.x ≠ (5 + (3 * ((M.y - 0) / (M.x - 1))) )) : 
  ∃ M, minimum_perimeter_triangle_MAF M = 11 :=
sorry

end minimum_perimeter_triangle_MAF_is_11_l225_225922


namespace mul_pos_neg_eq_neg_l225_225412

theorem mul_pos_neg_eq_neg (a : Int) : 3 * (-2) = -6 := by
  sorry

end mul_pos_neg_eq_neg_l225_225412


namespace simplify_expr_l225_225165

noncomputable def expr : ℝ := (18 * 10^10) / (6 * 10^4) * 2

theorem simplify_expr : expr = 6 * 10^6 := sorry

end simplify_expr_l225_225165


namespace find_b_minus_a_l225_225492

/-- Proof to find the value of b - a given the inequality conditions on x.
    The conditions are:
    1. x - a < 1
    2. x + b > 2
    3. 0 < x < 4
    We need to show that b - a = -1.
-/
theorem find_b_minus_a (a b x : ℝ) 
  (h1 : x - a < 1) 
  (h2 : x + b > 2) 
  (h3 : 0 < x) 
  (h4 : x < 4) 
  : b - a = -1 := 
sorry

end find_b_minus_a_l225_225492


namespace part_I_part_II_l225_225555

open Real

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 4)

theorem part_I (x : ℝ) : f x > 0 ↔ (x > 1 ∨ x < -5) := 
sorry

theorem part_II (m : ℝ) : (∀ x : ℝ, f x + 3 * abs (x - 4) > m) ↔ (m < 9) :=
sorry

end part_I_part_II_l225_225555


namespace select_10_teams_l225_225501

def football_problem (teams : Finset ℕ) (played_on_day1 : Finset (ℕ × ℕ)) (played_on_day2 : Finset (ℕ × ℕ)) : Prop :=
  ∀ (v : ℕ), v ∈ teams → (∃ u w : ℕ, (u, v) ∈ played_on_day1 ∧ (v, w) ∈ played_on_day2)

theorem select_10_teams {teams : Finset ℕ}
  (h : teams.card = 20)
  {played_on_day1 played_on_day2 : Finset (ℕ × ℕ)}
  (h1 : ∀ ⦃u v : ℕ⦄, (u, v) ∈ played_on_day1 → u ∈ teams ∧ v ∈ teams)
  (h2 : ∀ ⦃u v : ℕ⦄, (u, v) ∈ played_on_day2 → u ∈ teams ∧ v ∈ teams)
  (h3 : ∀ x ∈ teams, ∃ u w, (u, x) ∈ played_on_day1 ∧ (x, w) ∈ played_on_day2) :
  ∃ S : Finset ℕ, S.card = 10 ∧ (∀ ⦃x y⦄, x ∈ S → y ∈ S → x ≠ y → (¬((x, y) ∈ played_on_day1) ∧ ¬((x, y) ∈ played_on_day2))) :=
by
  sorry

end select_10_teams_l225_225501


namespace liquid_flow_problem_l225_225598

variables (x y z : ℝ)

theorem liquid_flow_problem 
    (h1 : 1/x + 1/y + 1/z = 1/6) 
    (h2 : y = 0.75 * x) 
    (h3 : z = y + 10) : 
    x = 56/3 ∧ y = 14 ∧ z = 24 :=
sorry

end liquid_flow_problem_l225_225598


namespace sequence_terms_divisible_by_b_l225_225473

theorem sequence_terms_divisible_by_b (a b : ℕ) :
  let d := Nat.gcd a b in
  (d = (List.range (b + 1)).filter (λ n, (a * n) % b = 0).length) :=
by
  sorry

end sequence_terms_divisible_by_b_l225_225473


namespace triple_layers_area_l225_225488

-- Defining the conditions
def hall : Type := {x // x = 10 * 10}
def carpet1 : hall := ⟨60, sorry⟩ -- First carpet size: 6 * 8
def carpet2 : hall := ⟨36, sorry⟩ -- Second carpet size: 6 * 6
def carpet3 : hall := ⟨35, sorry⟩ -- Third carpet size: 5 * 7

-- The final theorem statement
theorem triple_layers_area : ∃ area : ℕ, area = 6 :=
by
  have intersection_area : ℕ := 2 * 3
  use intersection_area
  sorry

end triple_layers_area_l225_225488


namespace correct_relation_l225_225516

def satisfies_relation : Prop :=
  (∀ x y, (x = 0 ∧ y = 200) ∨ (x = 1 ∧ y = 170) ∨ (x = 2 ∧ y = 120) ∨ (x = 3 ∧ y = 50) ∨ (x = 4 ∧ y = 0) →
  y = 200 - 10 * x - 10 * x^2) 

theorem correct_relation : satisfies_relation :=
sorry

end correct_relation_l225_225516


namespace gcd_digit_bound_l225_225581

theorem gcd_digit_bound (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (lcm_ab : ℕ) (h_lcm : 10^11 ≤ lcm_ab ∧ lcm_ab < 10^12) :
  int.log10 (Nat.gcd a b) < 3 := 
sorry

end gcd_digit_bound_l225_225581


namespace benjamin_speed_l225_225732

-- Define the problem conditions
def distance : ℕ := 800 -- Distance in kilometers
def time : ℕ := 10 -- Time in hours

-- Define the main statement
theorem benjamin_speed : distance / time = 80 := by
  sorry

end benjamin_speed_l225_225732


namespace catherine_initial_pens_l225_225675

-- Defining the conditions
def equal_initial_pencils_and_pens (P : ℕ) : Prop := true
def pens_given_away_per_friend : ℕ := 8
def pencils_given_away_per_friend : ℕ := 6
def number_of_friends : ℕ := 7
def remaining_pens_and_pencils : ℕ := 22

-- The total number of items given away
def total_pens_given_away : ℕ := pens_given_away_per_friend * number_of_friends
def total_pencils_given_away : ℕ := pencils_given_away_per_friend * number_of_friends

-- The problem statement in Lean 4
theorem catherine_initial_pens (P : ℕ) 
  (h1 : equal_initial_pencils_and_pens P)
  (h2 : P - total_pens_given_away + P - total_pencils_given_away = remaining_pens_and_pencils) : 
  P = 60 :=
sorry

end catherine_initial_pens_l225_225675


namespace four_consecutive_even_impossible_l225_225201

def is_four_consecutive_even_sum (S : ℕ) : Prop :=
  ∃ n : ℤ, S = 4 * n + 12

theorem four_consecutive_even_impossible :
  ¬ is_four_consecutive_even_sum 34 :=
by
  sorry

end four_consecutive_even_impossible_l225_225201


namespace parabola_equation_l225_225852

open Real

noncomputable def parabola_vertex_form (x : ℝ) (a : ℝ) : ℝ := a * (x - 3)^2 + 5

theorem parabola_equation :
  ∃ a b c : ℝ,
  (∀ x : ℝ, parabola_vertex_form x a = a * (x - 3)^2 + 5) ∧
  -- Point (0,2) lies on the parabola
  (∀ x : ℝ, x = 0 → (parabola_vertex_form x a) = 2) ∧
  -- Given point x=3 is the vertex
  (∀ y : ℝ, ∃ x : ℝ, x = 3 → y = 5) →
  -- General equation in the form ax^2 + bx + c
  ∀ x : ℝ, (-⅓) * x^2 + 2 * x + 2 = -⅓ * x^2 + 2 * x + 2 :=
begin
  use [-⅓, 2, 2],
  split,
  { intros x,
    exact calc
    (-⅓) * (x - 3)^2 + 5 = (-⅓) * (3 - x)^2 + 5 : by ring
    ... = (-⅓) * (x^2 - 6 * x + 9) + 5 : by ring
    ... = (-⅓) * x^2 + 2 * x + 2 : by ring },
  split,
  { intros x h,
    rw h,
    refl, },
  intro y,
  use [3],
  intro hx,
  rw hx,
  refl,
end

end parabola_equation_l225_225852


namespace ralph_has_18_fewer_pictures_l225_225624

/-- Ralph has 58 pictures of wild animals. Derrick has 76 pictures of wild animals.
    Prove that Ralph has 18 fewer pictures of wild animals compared to Derrick. -/
theorem ralph_has_18_fewer_pictures :
  let Ralph_pictures := 58
  let Derrick_pictures := 76
  76 - 58 = 18 :=
by
  let Ralph_pictures := 58
  let Derrick_pictures := 76
  show 76 - 58 = 18
  sorry

end ralph_has_18_fewer_pictures_l225_225624


namespace arithmetic_sequence_general_term_l225_225337

noncomputable def an (a_1 d : ℤ) (n : ℕ) : ℤ := a_1 + (n - 1) * d
def bn (a_n : ℤ) : ℚ := (1 / 2)^a_n

theorem arithmetic_sequence_general_term
  (a_n : ℕ → ℤ)
  (b_1 b_2 b_3 : ℚ)
  (a_1 d : ℤ)
  (h_seq : ∀ n, a_n n = a_1 + (n - 1) * d)
  (h_b1 : b_1 = (1 / 2)^(a_n 1))
  (h_b2 : b_2 = (1 / 2)^(a_n 2))
  (h_b3 : b_3 = (1 / 2)^(a_n 3))
  (h_sum : b_1 + b_2 + b_3 = 21 / 8)
  (h_prod : b_1 * b_2 * b_3 = 1 / 8)
  : (∀ n, a_n n = 2 * n - 3) ∨ (∀ n, a_n n = 5 - 2 * n) :=
sorry

end arithmetic_sequence_general_term_l225_225337


namespace mario_pizza_area_l225_225346

theorem mario_pizza_area
  (pizza_area : ℝ)
  (cut_distance : ℝ)
  (largest_piece : ℝ)
  (smallest_piece : ℝ)
  (total_pieces : ℕ)
  (pieces_mario_gets_area : ℝ) :
  pizza_area = 4 →
  cut_distance = 0.5 →
  total_pieces = 4 →
  pieces_mario_gets_area = (pizza_area - (largest_piece + smallest_piece)) / 2 →
  pieces_mario_gets_area = 1.5 :=
sorry

end mario_pizza_area_l225_225346


namespace candies_on_second_day_l225_225765

noncomputable def total_candies := 45
noncomputable def days := 5
noncomputable def difference := 3

def arithmetic_sum (n : ℕ) (a₁ d : ℕ) :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

theorem candies_on_second_day (a : ℕ) (h : arithmetic_sum days a difference = total_candies) :
  a + difference = 6 := by
  sorry

end candies_on_second_day_l225_225765


namespace simplified_expression_evaluation_l225_225014

def expression (x y : ℝ) : ℝ :=
  3 * (x^2 - 2 * x^2 * y) - 3 * x^2 + 2 * y - 2 * (x^2 * y + y)

def x := 1/2
def y := -3

theorem simplified_expression_evaluation : expression x y = 6 :=
  sorry

end simplified_expression_evaluation_l225_225014


namespace sqrt_condition_l225_225126

theorem sqrt_condition (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
sorry

end sqrt_condition_l225_225126


namespace vector_dot_product_l225_225105

-- Definitions
def vec_a : ℝ × ℝ := (1, 3)
def vec_b : ℝ × ℝ := (-2, -1)

-- Theorem to prove
theorem vector_dot_product : 
  ((vec_a.1 + vec_b.1, vec_a.2 + vec_b.2) : ℝ × ℝ) • (2 * vec_a.1 + vec_b.1, 2 * vec_a.2 + vec_b.2) = 10 :=
by
  sorry

end vector_dot_product_l225_225105


namespace gcd_digits_le_3_l225_225583

theorem gcd_digits_le_3 (a b : ℕ) (h_a : 10^6 ≤ a < 10^7) (h_b : 10^6 ≤ b < 10^7) (h_lcm : 10^11 ≤ Nat.lcm a b):
  Nat.gcd a b < 1000 := 
sorry

end gcd_digits_le_3_l225_225583


namespace pine_tree_taller_than_birch_l225_225156

def height_birch : ℚ := 49 / 4
def height_pine : ℚ := 74 / 4

def height_difference : ℚ :=
  height_pine - height_birch

theorem pine_tree_taller_than_birch :
  height_difference = 25 / 4 :=
by
  sorry

end pine_tree_taller_than_birch_l225_225156


namespace count_two_digit_perfect_squares_divisible_by_4_l225_225110

theorem count_two_digit_perfect_squares_divisible_by_4 : 
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = k^2 ∧ k^2 % 4 = 0}.to_finset.card = 3 :=
by
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l225_225110


namespace KA_eq_KT_l225_225151

namespace ProofProblem

open EuclideanGeometry

variables (A B C O M N T X Y K : Point)
variables (ω : Circle)
variables (circumcircle_ABC : IsCircumcircle ω A B C)
variables (M_mid_AB : Midpoint M A B)
variables (N_mid_AC : Midpoint N A C)
variables (T_mid_arc_BC_no_A : IsArcMidpoint T ω B C)
variables (circumcircle_AMT : IsCircumcircle (circumcircle A M T) A M T)
variables (circumcircle_ANT : IsCircumcircle (circumcircle A N T) A N T)
variables (X_on_perp_bisector_AC : OnPerpendicularBisector X A C)
variables (Y_on_perp_bisector_AB : OnPerpendicularBisector Y A B)
variables (X_inside_ABC : InsideTriangle X A B C)
variables (Y_inside_ABC : InsideTriangle Y A B C)
variables (K_intersection_MN_XY : Intersection K (Line M N) (Line X Y))

theorem KA_eq_KT :
  KA = KT :=
sorry

end KA_eq_KT_l225_225151


namespace flight_duration_sum_l225_225807

theorem flight_duration_sum 
  (departure_time : ℕ×ℕ) (arrival_time : ℕ×ℕ) (delay : ℕ)
  (h m : ℕ)
  (h0 : 0 < m ∧ m < 60)
  (h1 : departure_time = (9, 20))
  (h2 : arrival_time = (13, 45)) -- using 13 for 1 PM, 24-hour format
  (h3 : delay = 25)
  (h4 : ((arrival_time.1 * 60 + arrival_time.2) - (departure_time.1 * 60 + departure_time.2) + delay) = h * 60 + m) :
  h + m = 29 :=
by {
  -- Proof is skipped
  sorry
}

end flight_duration_sum_l225_225807


namespace chairs_removal_correct_chairs_removal_l225_225939

theorem chairs_removal (initial_chairs : ℕ) (chairs_per_row : ℕ) (participants : ℕ) : ℕ :=
  let total_chairs := 169
  let per_row := 13
  let attendees := 95
  let needed_chairs := (attendees + per_row - 1) / per_row * per_row
  let chairs_to_remove := total_chairs - needed_chairs
  chairs_to_remove

theorem correct_chairs_removal : chairs_removal 169 13 95 = 65 :=
by
  sorry

end chairs_removal_correct_chairs_removal_l225_225939


namespace sqrt_condition_l225_225127

theorem sqrt_condition (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
sorry

end sqrt_condition_l225_225127


namespace tan_product_eq_three_l225_225231

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end tan_product_eq_three_l225_225231


namespace tan_identity_proof_l225_225268

noncomputable def tan_identity : ℝ := Real.tan (π / 9) * Real.tan (2 * (π / 9)) * Real.tan (4 * (π / 9))

theorem tan_identity_proof : tan_identity = 1 := 
by 
  -- Use the polynomial relation: 9t - 84t^3 + 126t^5 - 36t^7 + t^9 = 0
  -- derived from \tan(9θ)
  -- And factor it to get t(t^8 - 84t^6 + 126t^4 - 36t^2 + 1) = 0
  -- For theta = π/9, 2π/9, 4π/9
  -- Use Vieta’s formulas to find the product
  sorry

end tan_identity_proof_l225_225268


namespace gcd_digit_bound_l225_225580

theorem gcd_digit_bound (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (lcm_ab : ℕ) (h_lcm : 10^11 ≤ lcm_ab ∧ lcm_ab < 10^12) :
  int.log10 (Nat.gcd a b) < 3 := 
sorry

end gcd_digit_bound_l225_225580


namespace integer_coordinates_for_all_vertices_l225_225505

-- Define a three-dimensional vector with integer coordinates
structure Vec3 :=
  (x : ℤ)
  (y : ℤ)
  (z : ℤ)

-- Define a cube with 8 vertices in 3D space
structure Cube :=
  (A1 A2 A3 A4 A1' A2' A3' A4' : Vec3)

-- Assumption: four vertices with integer coordinates that do not lie on the same plane
def has_four_integer_vertices (cube : Cube) : Prop :=
  ∃ (A B C D : Vec3),
    A ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'] ∧
    B ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'] ∧
    C ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'] ∧
    D ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'] ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (C.x - A.x) * (D.y - B.y) ≠ (D.x - B.x) * (C.y - A.y) ∧  -- Ensure not co-planar
    (C.y - A.y) * (D.z - B.z) ≠ (D.y - B.y) * (C.z - A.z)

-- The proof problem: prove all vertices have integer coordinates given the condition
theorem integer_coordinates_for_all_vertices (cube : Cube) (h : has_four_integer_vertices cube) : 
  ∀ v ∈ [cube.A1, cube.A2, cube.A3, cube.A4, cube.A1', cube.A2', cube.A3', cube.A4'], 
    ∃ (v' : Vec3), v = v' := 
  by
  sorry

end integer_coordinates_for_all_vertices_l225_225505


namespace hexagons_cover_65_percent_l225_225635

noncomputable def hexagon_percent_coverage
    (a : ℝ)
    (square_area : ℝ := a^2) 
    (hexagon_area : ℝ := (3 * Real.sqrt 3 / 8 * a^2))
    (tile_pattern : ℝ := 3): Prop :=
    hexagon_area / square_area * tile_pattern = (65 / 100)

theorem hexagons_cover_65_percent (a : ℝ) : hexagon_percent_coverage a :=
by
    sorry

end hexagons_cover_65_percent_l225_225635


namespace quadratic_has_one_solution_l225_225089

theorem quadratic_has_one_solution (k : ℝ) : (4 : ℝ) * (4 : ℝ) - k ^ 2 = 0 → k = 8 ∨ k = -8 := by
  sorry

end quadratic_has_one_solution_l225_225089


namespace part1_part2_l225_225868

noncomputable def f (x : ℝ) (a : ℝ) := x^2 + 2*a*x + 2

theorem part1 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 5 → f x a > 3*a*x) → a < 2*Real.sqrt 2 :=
sorry

theorem part2 (a : ℝ) :
  ∀ x : ℝ,
    ((a = 0) → x > 2) ∧
    ((a > 0) → (x < -1/a ∨ x > 2)) ∧
    ((-1/2 < a ∧ a < 0) → (2 < x ∧ x < -1/a)) ∧
    ((a = -1/2) → false) ∧
    ((a < -1/2) → (-1/a < x ∧ x < 2)) :=
sorry

end part1_part2_l225_225868


namespace proof_problem_l225_225717

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 1 / x

noncomputable def f'' (x : ℝ) : ℝ := Real.exp x + 2 / x^3

theorem proof_problem {x0 m n : ℝ} (hx0_pos : 0 < x0)
  (H : f'' x0 = 0) (hm : 0 < m) (hmx0 : m < x0) (hn : x0 < n) :
  f'' m < 0 ∧ f'' n > 0 := sorry

end proof_problem_l225_225717


namespace reciprocal_eq_self_l225_225737

theorem reciprocal_eq_self {x : ℝ} (h : x ≠ 0) : (1 / x = x) → (x = 1 ∨ x = -1) :=
by
  intro h1
  sorry

end reciprocal_eq_self_l225_225737


namespace obtuse_triangle_acute_angles_l225_225723

theorem obtuse_triangle_acute_angles (A B C : ℝ) (h : A + B + C = 180)
  (hA : A > 90) : (B < 90) ∧ (C < 90) :=
sorry

end obtuse_triangle_acute_angles_l225_225723


namespace bipin_chandan_age_ratio_l225_225523

-- Define the condition statements
def AlokCurrentAge : Nat := 5
def BipinCurrentAge : Nat := 6 * AlokCurrentAge
def ChandanCurrentAge : Nat := 7 + 3

-- Define the ages after 10 years
def BipinAgeAfter10Years : Nat := BipinCurrentAge + 10
def ChandanAgeAfter10Years : Nat := ChandanCurrentAge + 10

-- Define the ratio and the statement to prove
def AgeRatio := BipinAgeAfter10Years / ChandanAgeAfter10Years

-- The theorem to prove the ratio is 2
theorem bipin_chandan_age_ratio : AgeRatio = 2 := by
  sorry

end bipin_chandan_age_ratio_l225_225523


namespace exercise_books_quantity_l225_225911

theorem exercise_books_quantity (ratio_pencil : ℕ) (ratio_exercise_book : ℕ) (pencils : ℕ) (h_ratio : ratio_pencil = 10) (h_ratio_exercise : ratio_exercise_book = 3) (h_pencils : pencils = 120) : (3 * (pencils / 10) = 36) :=
by
  have h1 : pencils / ratio_pencil = pencils / 10 := by rw [h_ratio]
  have h2 : (3 * (pencils / 10) = 3 * 12) := by rw [h1, h_pencils, Nat.div_eq_of_lt]
  have h3 : 3 * 12 = 36 := by norm_num
  rw [h2, h3]
  norm_num

end exercise_books_quantity_l225_225911


namespace value_of_f_at_neg1_l225_225030

def f (x : ℤ) : ℤ := 1 + 2 * x + x^2 - 3 * x^3 + 2 * x^4

theorem value_of_f_at_neg1 : f (-1) = 6 :=
by
  sorry

end value_of_f_at_neg1_l225_225030


namespace function_is_odd_and_monotonically_increasing_on_pos_l225_225665

-- Define odd function
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

-- Define monotonically increasing on (0, +∞)
def monotonically_increasing_on_pos (f : ℝ → ℝ) := ∀ x y : ℝ, (0 < x ∧ x < y) → f (x) < f (y)

-- Define the function in question
def f (x : ℝ) := x * |x|

-- Prove the function is odd and monotonically increasing on (0, +∞)
theorem function_is_odd_and_monotonically_increasing_on_pos :
  odd_function f ∧ monotonically_increasing_on_pos f :=
by
  sorry

end function_is_odd_and_monotonically_increasing_on_pos_l225_225665


namespace gcd_relatively_prime_l225_225901

theorem gcd_relatively_prime (a : ℤ) (m n : ℕ) (h_odd : a % 2 = 1) (h_pos_m : m > 0) (h_pos_n : n > 0) (h_diff : n ≠ m) :
  Int.gcd (a ^ 2^m + 2 ^ 2^m) (a ^ 2^n + 2 ^ 2^n) = 1 :=
by
  sorry

end gcd_relatively_prime_l225_225901


namespace rectangle_width_l225_225367

theorem rectangle_width (P l: ℕ) (hP : P = 50) (hl : l = 13) : 
  ∃ w : ℕ, 2 * l + 2 * w = P ∧ w = 12 := 
by
  sorry

end rectangle_width_l225_225367


namespace d_is_distance_function_l225_225543

noncomputable def d (x y : ℝ) : ℝ := |x - y| / (Real.sqrt (1 + x^2) * Real.sqrt (1 + y^2))

theorem d_is_distance_function : 
  (∀ x, d x x = 0) ∧ 
  (∀ x y, d x y = d y x) ∧ 
  (∀ x y z, d x y + d y z ≥ d x z) :=
by
  sorry

end d_is_distance_function_l225_225543


namespace shaded_area_z_shape_l225_225074

theorem shaded_area_z_shape (L W s1 s2 : ℕ) (hL : L = 6) (hW : W = 4) (hs1 : s1 = 2) (hs2 : s2 = 1) :
  (L * W - (s1 * s1 + s2 * s2)) = 19 := by
  sorry

end shaded_area_z_shape_l225_225074


namespace Alyssa_missed_games_l225_225833

theorem Alyssa_missed_games (total_games attended_games : ℕ) (h1 : total_games = 31) (h2 : attended_games = 13) : total_games - attended_games = 18 :=
by sorry

end Alyssa_missed_games_l225_225833


namespace tan_product_equals_three_l225_225241

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end tan_product_equals_three_l225_225241


namespace area_of_quadrilateral_ABCD_l225_225286

theorem area_of_quadrilateral_ABCD
  (BD : ℝ) (hA : ℝ) (hC : ℝ) (angle_ABD : ℝ) :
  BD = 28 ∧ hA = 8 ∧ hC = 2 ∧ angle_ABD = 60 →
  ∃ (area_ABCD : ℝ), area_ABCD = 140 :=
by
  sorry

end area_of_quadrilateral_ABCD_l225_225286


namespace minimum_balls_to_draw_l225_225936

theorem minimum_balls_to_draw
  (red green yellow blue white : ℕ)
  (h_red : red = 30)
  (h_green : green = 25)
  (h_yellow : yellow = 20)
  (h_blue : blue = 15)
  (h_white : white = 10) :
  ∃ (n : ℕ), n = 81 ∧
    (∀ (r g y b w : ℕ), 
       (r + g + y + b + w >= n) →
       ((r ≥ 20 ∨ g ≥ 20 ∨ y ≥ 20 ∨ b ≥ 20 ∨ w ≥ 20) ∧ 
        (r ≥ 10 ∨ g ≥ 10 ∨ y ≥ 10 ∨ b ≥ 10 ∨ w ≥ 10))
    ) := sorry

end minimum_balls_to_draw_l225_225936


namespace infinite_series_sum_l225_225524

theorem infinite_series_sum :
  ∑' (n : ℕ), (n + 1) * (1 / 1000)^n = 3000000 / 998001 :=
by sorry

end infinite_series_sum_l225_225524


namespace minimum_uninteresting_vertices_correct_maximum_unusual_vertices_correct_l225_225130

-- Definition for the minimum number of uninteresting vertices
def minimum_uninteresting_vertices (n : ℕ) (h : n > 3) : ℕ := 2

-- Theorem for the minimum number of uninteresting vertices
theorem minimum_uninteresting_vertices_correct (n : ℕ) (h : n > 3) :
  minimum_uninteresting_vertices n h = 2 := 
sorry

-- Definition for the maximum number of unusual vertices
def maximum_unusual_vertices (n : ℕ) (h : n > 3) : ℕ := 3

-- Theorem for the maximum number of unusual vertices
theorem maximum_unusual_vertices_correct (n : ℕ) (h : n > 3) :
  maximum_unusual_vertices n h = 3 :=
sorry

end minimum_uninteresting_vertices_correct_maximum_unusual_vertices_correct_l225_225130


namespace unique_solution_ffx_eq_27_l225_225996

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 27

-- Prove that there is exactly one solution for f(f(x)) = 27 in the domain -3 ≤ x ≤ 5
theorem unique_solution_ffx_eq_27 :
  (∃! x : ℝ, -3 ≤ x ∧ x ≤ 5 ∧ f (f x) = 27) :=
by
  sorry

end unique_solution_ffx_eq_27_l225_225996


namespace perfect_squares_two_digit_divisible_by_4_count_l225_225107

-- Define two-digit
def is_two_digit (n : ℤ) : Prop :=
  10 ≤ n ∧ n < 100

-- Define perfect square
def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k^2

-- Define divisible by 4
def divisible_by_4 (n : ℤ) : Prop :=
  n % 4 = 0

-- Define the main statement: number of two-digit perfect squares that are divisible by 4 is 3
theorem perfect_squares_two_digit_divisible_by_4_count :
  { n : ℤ | is_two_digit n ∧ is_perfect_square n ∧ divisible_by_4 n }.size = 3 :=
by sorry

end perfect_squares_two_digit_divisible_by_4_count_l225_225107


namespace expansion_a0_alternating_sum_l225_225446

open Finset

section
variables {A : Type*} [CommRing A]

theorem expansion_a0 (a : ℕ → A) (x : A) :
  ((1 - (2 : A) * x)^2023 = ∑ i in range (2023 + 1), a i * x^i) →
  a 0 = 1 :=
begin
  intro h,
  have := congr_arg (λ p, p.eval 0) h,
  simp only [eval_pow, eval_sub, eval_one, eval_mul, eval_C, eval_X, zero_pow (by norm_num : 0 < 2023 + 1), zero_mul, mul_zero, sub_zero, one_pow] at this,
  exact this,
end

theorem alternating_sum (a : ℕ → A) (x : A):
  ((1 - (2 : A) * x)^2023 = ∑ i in range (2023 + 1), a i * x^i) →
  a 1 - a 2 + a 3 - a 4 + ∑ i in range (2023).succ.succ \ {0, 1, 2, 3},
    if i.even then -a i else a i = 1 - (3 : A)^2023 :=
begin
  intro h,
  have := congr_arg (λ p, p.eval (-1)) h,
  simp only [eval_pow, eval_sub, eval_one, eval_neg, eval_mul, eval_bit0, eval_bit1, eval_X, neg_one_pow_eq_one_iff_even, add_eq_zero_iff_eq_neg, one_pow, neg_one_pow_eq_zero_iff_odd, eval_C, neg_mul, eval_X, sub_eq_add_neg, zero_pow, mul_one, one_mul, neg_neg, pow_eq_pow] at this,
  rw [finset.sum_sub_distrib, sum_singleton, sum_odd_succ] at this,
  convert this,
end

end

end expansion_a0_alternating_sum_l225_225446


namespace simple_interest_amount_l225_225313

noncomputable def simple_interest (P r t : ℝ) : ℝ := (P * r * t) / 100
noncomputable def compound_interest (P r t : ℝ) : ℝ := P * (1 + r / 100)^t - P

theorem simple_interest_amount:
  ∀ (P : ℝ), compound_interest P 5 2 = 51.25 → simple_interest P 5 2 = 50 :=
by
  intros P h
  -- this is where the proof would go
  sorry

end simple_interest_amount_l225_225313


namespace find_pairs_l225_225691

def is_solution_pair (m n : ℕ) : Prop :=
  Nat.lcm m n = 3 * m + 2 * n + 1

theorem find_pairs :
  { pairs : List (ℕ × ℕ) // ∀ (m n : ℕ), (m, n) ∈ pairs ↔ is_solution_pair m n } :=
by
  let pairs := [(3,10), (4,9)]
  have key : ∀ (m n : ℕ), (m, n) ∈ pairs ↔ is_solution_pair m n := sorry
  exact ⟨pairs, key⟩

end find_pairs_l225_225691


namespace paint_area_is_correct_l225_225010

-- Define the dimensions of the wall, window, and door
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window_height : ℕ := 3
def window_length : ℕ := 5
def door_height : ℕ := 1
def door_length : ℕ := 7

-- Calculate area
def wall_area : ℕ := wall_height * wall_length
def window_area : ℕ := window_height * window_length
def door_area : ℕ := door_height * door_length

-- Calculate area to be painted
def area_to_be_painted : ℕ := wall_area - window_area - door_area

-- The theorem statement
theorem paint_area_is_correct : area_to_be_painted = 128 := 
by
  -- The proof would go here (omitted)
  sorry

end paint_area_is_correct_l225_225010


namespace products_not_all_greater_than_one_quarter_l225_225623

theorem products_not_all_greater_than_one_quarter
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 1)
  (hb : 0 < b ∧ b < 1)
  (hc : 0 < c ∧ c < 1) :
  ¬ ((1 - a) * b > 1 / 4 ∧ (1 - b) * c > 1 / 4 ∧ (1 - c) * a > 1 / 4) :=
by
  sorry

end products_not_all_greater_than_one_quarter_l225_225623


namespace local_minimum_of_reflected_function_l225_225046

noncomputable def f : ℝ → ℝ := sorry

theorem local_minimum_of_reflected_function (f : ℝ → ℝ) (x_0 : ℝ) (h1 : x_0 ≠ 0) (h2 : ∃ ε > 0, ∀ x, abs (x - x_0) < ε → f x ≤ f x_0) :
  ∃ δ > 0, ∀ x, abs (x - (-x_0)) < δ → -f (-x) ≥ -f (-x_0) :=
sorry

end local_minimum_of_reflected_function_l225_225046


namespace triangle_area_l225_225826

theorem triangle_area : 
  ∃ (A : ℝ), A = 12 ∧ (∃ (x_intercept y_intercept : ℝ), 3 * x_intercept + 2 * y_intercept = 12 ∧ x_intercept * y_intercept / 2 = A) :=
by
  sorry

end triangle_area_l225_225826


namespace simplify_power_of_power_l225_225012

variable (x : ℝ)

theorem simplify_power_of_power : (3 * x^4)^4 = 81 * x^16 := 
by 
sorry

end simplify_power_of_power_l225_225012


namespace tan_identity_proof_l225_225267

noncomputable def tan_identity : ℝ := Real.tan (π / 9) * Real.tan (2 * (π / 9)) * Real.tan (4 * (π / 9))

theorem tan_identity_proof : tan_identity = 1 := 
by 
  -- Use the polynomial relation: 9t - 84t^3 + 126t^5 - 36t^7 + t^9 = 0
  -- derived from \tan(9θ)
  -- And factor it to get t(t^8 - 84t^6 + 126t^4 - 36t^2 + 1) = 0
  -- For theta = π/9, 2π/9, 4π/9
  -- Use Vieta’s formulas to find the product
  sorry

end tan_identity_proof_l225_225267


namespace distinct_real_roots_find_other_root_and_k_l225_225558

-- Definition of the quadratic equation
def quadratic_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Part (1): Proving the discriminant condition
theorem distinct_real_roots (k : ℝ) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_eq 2 k (-1) x1 = 0 ∧ quadratic_eq 2 k (-1) x2 = 0 := by
  sorry

-- Part (2): Finding the other root and the value of k
theorem find_other_root_and_k : 
  ∃ k : ℝ, ∃ x2 : ℝ,
    quadratic_eq 2 1 (-1) (-1) = 0 ∧ quadratic_eq 2 1 (-1) x2 = 0 ∧ k = 1 ∧ x2 = 1/2 := by
  sorry

end distinct_real_roots_find_other_root_and_k_l225_225558


namespace Donny_change_l225_225423

/-- The change Donny will receive after filling up his truck. -/
theorem Donny_change
  (capacity : ℝ)
  (initial_fuel : ℝ)
  (cost_per_liter : ℝ)
  (money_available : ℝ)
  (change : ℝ) :
  capacity = 150 →
  initial_fuel = 38 →
  cost_per_liter = 3 →
  money_available = 350 →
  change = money_available - cost_per_liter * (capacity - initial_fuel) →
  change = 14 :=
by
  intros h_capacity h_initial_fuel h_cost_per_liter h_money_available h_change
  rw [h_capacity, h_initial_fuel, h_cost_per_liter, h_money_available] at h_change
  sorry

end Donny_change_l225_225423


namespace annual_subscription_cost_l225_225508

theorem annual_subscription_cost :
  (10 * 12) * (1 - 0.2) = 96 :=
by
  sorry

end annual_subscription_cost_l225_225508


namespace length_of_other_parallel_side_l225_225429

theorem length_of_other_parallel_side 
  (a : ℝ) (h : ℝ) (A : ℝ) (x : ℝ) 
  (h_a : a = 16) (h_h : h = 15) (h_A : A = 270) 
  (h_area_formula : A = 1 / 2 * (a + x) * h) : 
  x = 20 :=
sorry

end length_of_other_parallel_side_l225_225429


namespace initial_candies_l225_225283

theorem initial_candies (L R : ℕ) (h1 : L + R = 27) (h2 : R - L = 2 * L + 3) : L = 6 ∧ R = 21 :=
by
  sorry

end initial_candies_l225_225283


namespace find_b_l225_225760

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b * x ^ 3 + b * x - 3

theorem find_b (b : ℝ) (h : g b (g b 1) = 1) : b = 1 / 2 :=
by
  sorry

end find_b_l225_225760


namespace min_gb_for_plan_y_to_be_cheaper_l225_225664

theorem min_gb_for_plan_y_to_be_cheaper (g : ℕ) : 20 * g > 3000 + 10 * g → g ≥ 301 := by
  sorry

end min_gb_for_plan_y_to_be_cheaper_l225_225664


namespace probability_sum_even_for_three_cubes_l225_225026

-- Define the probability function
def probability_even_sum (n: ℕ) : ℚ :=
  if n > 0 then 1 / 2 else 0

theorem probability_sum_even_for_three_cubes : probability_even_sum 3 = 1 / 2 :=
by
  sorry

end probability_sum_even_for_three_cubes_l225_225026


namespace solve_equation_l225_225918

theorem solve_equation (x : ℝ) (h : x ≠ 3) : 
  -x^2 = (3*x - 3) / (x - 3) → x = 1 :=
by
  intro h1
  sorry

end solve_equation_l225_225918


namespace area_of_triangle_bounded_by_coordinate_axes_and_line_l225_225819

def area_of_triangle (b h : ℕ) : ℕ := (b * h) / 2

theorem area_of_triangle_bounded_by_coordinate_axes_and_line :
  area_of_triangle 4 6 = 12 :=
by
  sorry

end area_of_triangle_bounded_by_coordinate_axes_and_line_l225_225819


namespace count_two_digit_perfect_squares_divisible_by_4_l225_225111

theorem count_two_digit_perfect_squares_divisible_by_4 :
  {n : ℕ | n ∈ (set.range (λ m, m ^ 2)) ∧ 10 ≤ n ∧ n < 100 ∧ n % 4 = 0}.to_finset.card = 3 :=
by
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l225_225111


namespace students_play_both_l225_225470

-- Definitions of problem conditions
def total_students : ℕ := 1200
def play_football : ℕ := 875
def play_cricket : ℕ := 450
def play_neither : ℕ := 100
def play_either := total_students - play_neither

-- Lean statement to prove that the number of students playing both football and cricket
theorem students_play_both : play_football + play_cricket - 225 = play_either :=
by
  -- The proof is omitted
  sorry

end students_play_both_l225_225470


namespace cos_product_identity_l225_225654

theorem cos_product_identity :
  3.422 * (Real.cos (π / 15)) * (Real.cos (2 * π / 15)) * (Real.cos (3 * π / 15)) *
  (Real.cos (4 * π / 15)) * (Real.cos (5 * π / 15)) * (Real.cos (6 * π / 15)) * (Real.cos (7 * π / 15)) =
  (1 / 2^7) :=
sorry

end cos_product_identity_l225_225654


namespace find_m_for_even_function_l225_225995

def f (x : ℝ) (m : ℝ) := x^2 + (m - 1) * x + 3

theorem find_m_for_even_function : ∃ m : ℝ, (∀ x : ℝ, f (-x) m = f x m) ∧ m = 1 :=
sorry

end find_m_for_even_function_l225_225995


namespace alex_score_l225_225161

theorem alex_score (n : ℕ) (avg19 avg20 alex : ℚ)
  (h1 : n = 20)
  (h2 : avg19 = 72)
  (h3 : avg20 = 74)
  (h_totalscore19 : 19 * avg19 = 1368)
  (h_totalscore20 : 20 * avg20 = 1480)
  (h_alexscore : alex = 112) :
  alex = (1480 - 1368 : ℚ) := 
sorry

end alex_score_l225_225161


namespace solve_for_a_b_c_d_l225_225628

theorem solve_for_a_b_c_d :
  ∃ a b c d : ℕ, (a + b + c + d) * (a^2 + b^2 + c^2 + d^2)^2 = 2023 ∧ a^3 + b^3 + c^3 + d^3 = 43 := 
by
  sorry

end solve_for_a_b_c_d_l225_225628


namespace parabola_x0_range_l225_225904

variables {x₀ y₀ : ℝ}
def parabola (x₀ y₀ : ℝ) : Prop := y₀^2 = 8 * x₀

def focus (x : ℝ) : ℝ := 2

def directrix (x : ℝ) : Prop := x = -2

/-- Prove that for any point (x₀, y₀) on the parabola y² = 8x and 
if a circle centered at the focus intersects the directrix, then x₀ > 2. -/
theorem parabola_x0_range (x₀ y₀ : ℝ) (h1 : parabola x₀ y₀)
  (h2 : ((x₀ - 2)^2 + y₀^2)^(1/2) > (2 : ℝ)) : x₀ > 2 := 
sorry

end parabola_x0_range_l225_225904


namespace last_passenger_probability_l225_225639

noncomputable def probability_last_passenger_gets_seat {n : ℕ} (h : n > 0) : ℚ :=
  if n = 1 then 1 else 1/2

theorem last_passenger_probability
  (n : ℕ) (h : n > 0) :
  probability_last_passenger_gets_seat h = 1/2 :=
  sorry

end last_passenger_probability_l225_225639


namespace sum_of_digits_0_to_2012_l225_225332

def sum_of_digits (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

def sum_of_digits_in_range (a b : Nat) : Nat :=
  ((List.range (b + 1)).drop a).map sum_of_digits |>.sum

theorem sum_of_digits_0_to_2012 : 
  sum_of_digits_in_range 0 2012 = 28077 := 
by
  sorry

end sum_of_digits_0_to_2012_l225_225332


namespace probability_sum_less_than_16_l225_225183

-- The number of possible outcomes when three six-sided dice are rolled
def total_outcomes : ℕ := 6 * 6 * 6

-- The number of favorable outcomes where the sum of the dice is less than 16
def favorable_outcomes : ℕ := (6 * 6 * 6) - (3 + 3 + 3 + 1)

-- The probability that the sum of the dice is less than 16
def probability_less_than_16 : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_less_than_16 : probability_less_than_16 = 103 / 108 := 
by sorry

end probability_sum_less_than_16_l225_225183


namespace value_of_a_l225_225096

/--
Given that x = 3 is a solution to the equation 3x - 2a = 5,
prove that a = 2.
-/
theorem value_of_a (x a : ℤ) (h : 3 * x - 2 * a = 5) (hx : x = 3) : a = 2 :=
by
  sorry

end value_of_a_l225_225096


namespace probability_sum_eq_k_l225_225295

open Polynomial

theorem probability_sum_eq_k (n m k : ℕ) :
  ∃ p : ℚ, p = coeff (m * (X + X^2 + ⋯ + X^n)) k / n^m :=
sorry

end probability_sum_eq_k_l225_225295


namespace smallest_positive_integer_l225_225647

theorem smallest_positive_integer (n : ℕ) (h1 : 0 < n) (h2 : ∃ k1 : ℕ, 3 * n = k1^2) (h3 : ∃ k2 : ℕ, 4 * n = k2^3) : 
  n = 54 := 
sorry

end smallest_positive_integer_l225_225647


namespace find_page_added_twice_l225_225487

theorem find_page_added_twice (m p : ℕ) (h1 : 1 ≤ p) (h2 : p ≤ m) (h3 : (m * (m + 1)) / 2 + p = 2550) : p = 6 :=
sorry

end find_page_added_twice_l225_225487


namespace total_cookies_in_box_l225_225136

-- Definitions from the conditions
def oldest_son_cookies : ℕ := 4
def youngest_son_cookies : ℕ := 2
def days_box_lasts : ℕ := 9

-- Total cookies consumed per day
def daily_cookies_consumption : ℕ := oldest_son_cookies + youngest_son_cookies

-- Theorem statement: total number of cookies in the box
theorem total_cookies_in_box : (daily_cookies_consumption * days_box_lasts) = 54 := by
  sorry

end total_cookies_in_box_l225_225136


namespace sqrt_121_eq_pm_11_l225_225932

theorem sqrt_121_eq_pm_11 : (∀ x : ℝ, x^2 = 121 → x = 11 ∨ x = -11) :=
by {
  intro x,
  intro h,
  have hx : x * x = 121 := by assumption,
  have pos_x : x = real.sqrt 121 ∨ x = - real.sqrt 121 := by
    have sqrt_121 := real.sqrt_eq_iff_sqr_eq (by norm_num) (by norm_num),
    rw sqrt_121 at hx,
    exact hx,
  rw real.sqrt_eq_iff_sqr_eq (by norm_num) (by norm_num) at pos_x,
  exact pos_x
}

end sqrt_121_eq_pm_11_l225_225932


namespace escalator_rate_is_15_l225_225405

noncomputable def rate_escalator_moves (escalator_length : ℝ) (person_speed : ℝ) (time : ℝ) : ℝ :=
  (escalator_length / time) - person_speed

theorem escalator_rate_is_15 :
  rate_escalator_moves 200 5 10 = 15 := by
  sorry

end escalator_rate_is_15_l225_225405


namespace green_valley_ratio_l225_225407

variable (j s : ℕ)

theorem green_valley_ratio (h : (3 / 4 : ℚ) * j = (1 / 2 : ℚ) * s) : s = 3 / 2 * j :=
by
  sorry

end green_valley_ratio_l225_225407


namespace regular_tetrahedron_subdivision_l225_225413

theorem regular_tetrahedron_subdivision :
  ∃ (n : ℕ), n ≤ 7 ∧ (∀ (i : ℕ) (h : i ≥ n), (1 / 2^i) < (1 / 100)) :=
by
  sorry

end regular_tetrahedron_subdivision_l225_225413


namespace christmas_gift_count_l225_225618

theorem christmas_gift_count (initial_gifts : ℕ) (additional_gifts : ℕ) (gifts_to_orphanage : ℕ)
  (h1 : initial_gifts = 77)
  (h2 : additional_gifts = 33)
  (h3 : gifts_to_orphanage = 66) :
  (initial_gifts + additional_gifts - gifts_to_orphanage = 44) :=
by
  sorry

end christmas_gift_count_l225_225618


namespace calculate_expression_l225_225671

theorem calculate_expression : 
  (-7 : ℤ)^7 / (7 : ℤ)^4 + 2^6 - 8^2 = -343 :=
by
  sorry

end calculate_expression_l225_225671


namespace housewife_spend_money_l225_225813

theorem housewife_spend_money (P M: ℝ) (h1: 0.75 * P = 30) (h2: M / (0.75 * P) - M / P = 5) : 
  M = 600 :=
by
  sorry

end housewife_spend_money_l225_225813


namespace sequence_general_term_l225_225535

theorem sequence_general_term (a : ℕ → ℤ) (n : ℕ) 
  (h₀ : a 0 = 1) 
  (h_rec : ∀ n, a (n + 1) = 2 * a n + n) :
  a n = 2^(n + 1) - n - 1 :=
by sorry

end sequence_general_term_l225_225535


namespace suitable_storage_temp_l225_225025

theorem suitable_storage_temp : -5 ≤ -1 ∧ -1 ≤ 1 := by {
  sorry
}

end suitable_storage_temp_l225_225025


namespace line_BC_l225_225705

noncomputable def Point := (ℝ × ℝ)
def A : Point := (-1, -4)
def l₁ := { p : Point | p.2 + 1 = 0 }
def l₂ := { p : Point | p.1 + p.2 + 1 = 0 }
def A' : Point := (-1, 2)
def A'' : Point := (3, 0)

theorem line_BC :
  ∃ (c₁ c₂ c₃ : ℝ), c₁ ≠ 0 ∨ c₂ ≠ 0 ∧
  ∀ (p : Point), (c₁ * p.1 + c₂ * p.2 + c₃ = 0) ↔ p ∈ { x | x = A ∨ x = A'' } :=
by sorry

end line_BC_l225_225705


namespace diagonals_of_hexadecagon_l225_225940

-- Define the function to calculate number of diagonals in a convex polygon
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- State the theorem for the number of diagonals in a convex hexadecagon
theorem diagonals_of_hexadecagon : num_diagonals 16 = 104 := by
  -- sorry is used to indicate the proof is skipped
  sorry

end diagonals_of_hexadecagon_l225_225940


namespace gum_cost_700_eq_660_cents_l225_225632

-- defining the cost function
def gum_cost (n : ℕ) : ℝ :=
  if n ≤ 500 then n * 0.01
  else 5 + (n - 500) * 0.008

-- proving the specific case for 700 pieces of gum
theorem gum_cost_700_eq_660_cents : gum_cost 700 = 6.60 := by
  sorry

end gum_cost_700_eq_660_cents_l225_225632


namespace zero_in_interval_l225_225101

theorem zero_in_interval {b : ℝ} (f : ℝ → ℝ)
  (h₁ : ∀ x, f x = 2 * b * x - 3 * b + 1)
  (h₂ : b > 1/5)
  (h₃ : b < 1) :
  ∃ x, -1 < x ∧ x < 1 ∧ f x = 0 :=
by
  sorry

end zero_in_interval_l225_225101


namespace annual_subscription_cost_l225_225509

-- Definitions based on the conditions

def monthly_cost : ℝ := 10
def months_per_year : ℕ := 12
def discount_rate : ℝ := 0.20

-- The statement based on the correct answer
theorem annual_subscription_cost : 
  (monthly_cost * months_per_year) * (1 - discount_rate) = 96 := 
by
  sorry

end annual_subscription_cost_l225_225509


namespace sqrt_121_pm_11_l225_225933

theorem sqrt_121_pm_11 :
  (∃ y : ℤ, y * y = 121) ∧ (∃ x : ℤ, x = 11 ∨ x = -11) → (∃ x : ℤ, x * x = 121 ∧ (x = 11 ∨ x = -11)) :=
by
  sorry

end sqrt_121_pm_11_l225_225933


namespace derivative_of_function_l225_225169

open Real

theorem derivative_of_function : ∀ x : ℝ, deriv (λ x : ℝ, 2 * x + cos x) x = 2 - sin x :=
by
  intro x
  sorry

end derivative_of_function_l225_225169


namespace profit_days_l225_225364

theorem profit_days (total_days : ℕ) (mean_profit_month first_half_days second_half_days : ℕ)
  (mean_profit_first_half mean_profit_second_half : ℕ)
  (h1 : mean_profit_month * total_days = (mean_profit_first_half * first_half_days + mean_profit_second_half * second_half_days))
  (h2 : first_half_days + second_half_days = total_days)
  (h3 : mean_profit_month = 350)
  (h4 : mean_profit_first_half = 225)
  (h5 : mean_profit_second_half = 475)
  (h6 : total_days = 30) : 
  first_half_days = 15 ∧ second_half_days = 15 := 
by 
  sorry

end profit_days_l225_225364


namespace four_distinct_real_roots_l225_225906

noncomputable def f (x d : ℝ) : ℝ := x^2 + 10*x + d

theorem four_distinct_real_roots (d : ℝ) :
  (∀ r, f r d = 0 → (∃! x, f x d = r)) → d < 25 :=
by
  sorry

end four_distinct_real_roots_l225_225906


namespace range_of_a_l225_225104

-- Define the propositions p and q
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

-- Define the main theorem which combines both propositions and infers the range of a
theorem range_of_a (a : ℝ) : prop_p a ∧ prop_q a → a ≤ -2 := sorry

end range_of_a_l225_225104


namespace sum_digits_0_to_2012_l225_225333

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

theorem sum_digits_0_to_2012 : ∑ n in Finset.range 2013, sum_of_digits n = 28077 := 
by
  sorry

end sum_digits_0_to_2012_l225_225333


namespace sum_underlined_numbers_non_negative_l225_225207

def sum_underlined_numbers (seq : Fin 100 → Int) : Bool :=
  let underlined_indices : List (Fin 100) :=
    List.range 100 |>.filter (λ i =>
      seq i > 0 ∨ (i < 99 ∧ seq i + seq (i + 1) > 0) ∨ (i < 98 ∧ seq i + seq (i + 1) + seq (i + 2) > 0))
  let underlined_sum : Int := underlined_indices.map (λ i => seq i) |>.sum
  underlined_sum ≤ 0

theorem sum_underlined_numbers_non_negative {seq : Fin 100 → Int} :
  ¬ sum_underlined_numbers seq :=
sorry

end sum_underlined_numbers_non_negative_l225_225207


namespace gcd_has_at_most_3_digits_l225_225575

noncomputable def lcm (a b : ℕ) : ℕ := sorry -- definition for lcm is already in Mathlib
noncomputable def gcd (a b : ℕ) : ℕ := sorry -- definition for gcd is already in Mathlib

theorem gcd_has_at_most_3_digits
  (a b : ℕ)
  (ha : 10^6 ≤ a ∧ a < 10^7)  -- a is a 7-digit integer
  (hb : 10^6 ≤ b ∧ b < 10^7)  -- b is a 7-digit integer
  (hlcm_digits : 10^11 ≤ lcm a b ∧ lcm a b < 10^12)  -- lcm of a and b has 12 digits
  : gcd a b < 10^3 := by
  sorry

end gcd_has_at_most_3_digits_l225_225575


namespace JulioHasMoreSoda_l225_225758

-- Define the number of bottles each person has
def JulioOrangeBottles : ℕ := 4
def JulioGrapeBottles : ℕ := 7
def MateoOrangeBottles : ℕ := 1
def MateoGrapeBottles : ℕ := 3

-- Define the volume of each bottle in liters
def BottleVolume : ℕ := 2

-- Define the total liters of soda each person has
def JulioTotalLiters : ℕ := JulioOrangeBottles * BottleVolume + JulioGrapeBottles * BottleVolume
def MateoTotalLiters : ℕ := MateoOrangeBottles * BottleVolume + MateoGrapeBottles * BottleVolume

-- Prove the difference in total liters of soda between Julio and Mateo
theorem JulioHasMoreSoda : JulioTotalLiters - MateoTotalLiters = 14 := by
  sorry

end JulioHasMoreSoda_l225_225758


namespace number_of_ways_to_assign_guests_l225_225215

theorem number_of_ways_to_assign_guests (friends rooms : ℕ) (h_friends : friends = 5) (h_rooms : rooms = 5) 
  (h_max_per_room : ∀ r, r ∈ finset.range rooms → r ≤ 2) : 
  (number_of_assignments friends rooms h_max_per_room) = 1620 := sorry

noncomputable def number_of_assignments : ℕ → ℕ → (∀ r : ℕ, r < 5 → r ≤ 2) → ℕ
| 5, 5, h_max_per_room := 120 + 600 + 900
| _, _, _ := 0

end number_of_ways_to_assign_guests_l225_225215


namespace Bruce_bought_8_kg_of_grapes_l225_225064

-- Defining the conditions
def rate_grapes := 70
def rate_mangoes := 55
def weight_mangoes := 11
def total_paid := 1165

-- Result to be proven
def cost_mangoes := rate_mangoes * weight_mangoes
def total_cost_grapes (G : ℕ) := rate_grapes * G
def total_cost (G : ℕ) := (total_cost_grapes G) + cost_mangoes

theorem Bruce_bought_8_kg_of_grapes (G : ℕ) (h : total_cost G = total_paid) : G = 8 :=
by
  sorry  -- Proof omitted

end Bruce_bought_8_kg_of_grapes_l225_225064


namespace cost_of_two_other_puppies_l225_225520

theorem cost_of_two_other_puppies (total_cost : ℕ) (sale_price : ℕ) (num_puppies : ℕ) (num_sale_puppies : ℕ) (remaining_puppies_cost : ℕ) :
  total_cost = 800 →
  sale_price = 150 →
  num_puppies = 5 →
  num_sale_puppies = 3 →
  remaining_puppies_cost = (total_cost - num_sale_puppies * sale_price) →
  (remaining_puppies_cost / (num_puppies - num_sale_puppies)) = 175 :=
by
  intros
  sorry

end cost_of_two_other_puppies_l225_225520


namespace binom_10_2_eq_45_l225_225683

theorem binom_10_2_eq_45 :
  binom 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l225_225683


namespace ice_cream_tubs_eaten_l225_225371

-- Conditions
def number_of_pans := 2
def pieces_per_pan := 16
def percentage_eaten_second_pan := 0.75
def scoops_per_tub := 8
def scoops_per_guest := 2
def guests_not_eating_ala_mode := 4

-- Questions
def tubs_of_ice_cream_eaten : Nat :=
  sorry

theorem ice_cream_tubs_eaten :
  tubs_of_ice_cream_eaten = 6 := by
  sorry

end ice_cream_tubs_eaten_l225_225371


namespace initial_average_age_l225_225631

theorem initial_average_age (A : ℝ) (n : ℕ) (h1 : n = 17) (h2 : n * A + 32 = (n + 1) * 15) : A = 14 := by
  sorry

end initial_average_age_l225_225631


namespace greatest_possible_large_chips_l225_225181

theorem greatest_possible_large_chips : 
  ∃ s l p: ℕ, s + l = 60 ∧ s = l + 2 * p ∧ Prime p ∧ l = 28 :=
by
  sorry

end greatest_possible_large_chips_l225_225181


namespace find_smaller_number_l225_225373

theorem find_smaller_number (u v : ℝ) (hu : u > 0) (hv : v > 0)
  (h_ratio : u / v = 3 / 5) (h_sum : u + v = 16) : u = 6 :=
by
  sorry

end find_smaller_number_l225_225373


namespace find_f_2008_l225_225550

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the problem statement with all given conditions
theorem find_f_2008 (h_odd : is_odd f) (h_f2 : f 2 = 0) (h_rec : ∀ x, f (x + 4) = f x + f 4) : f 2008 = 0 := 
sorry

end find_f_2008_l225_225550


namespace gcd_cubed_and_sum_l225_225090

theorem gcd_cubed_and_sum (n : ℕ) (h_pos : 0 < n) (h_gt_square : n > 9) : 
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 := 
sorry

end gcd_cubed_and_sum_l225_225090


namespace probability_product_multiple_of_3_l225_225947

structure Die where
  sides : ℕ
  rolls : ℕ

def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

noncomputable def probability_multiple_of_3_in_rolls (die : Die) : ℚ :=
  1 - (float_of (2/3) ^ die.rolls)

theorem probability_product_multiple_of_3 (die : Die)
  (h1 : die.sides = 6)
  (h2 : die.rolls = 8) :
  probability_multiple_of_3_in_rolls die = 6305 / 6561 :=
  sorry

end probability_product_multiple_of_3_l225_225947


namespace tan_identity_proof_l225_225269

noncomputable def tan_identity : ℝ := Real.tan (π / 9) * Real.tan (2 * (π / 9)) * Real.tan (4 * (π / 9))

theorem tan_identity_proof : tan_identity = 1 := 
by 
  -- Use the polynomial relation: 9t - 84t^3 + 126t^5 - 36t^7 + t^9 = 0
  -- derived from \tan(9θ)
  -- And factor it to get t(t^8 - 84t^6 + 126t^4 - 36t^2 + 1) = 0
  -- For theta = π/9, 2π/9, 4π/9
  -- Use Vieta’s formulas to find the product
  sorry

end tan_identity_proof_l225_225269


namespace ten_percent_of_x_l225_225934

variable (certain_value : ℝ)
variable (x : ℝ)

theorem ten_percent_of_x (h : 3 - (1 / 4) * 2 - (1 / 3) * 3 - (1 / 7) * x = certain_value) :
  0.1 * x = 0.7 * (1.5 - certain_value) := sorry

end ten_percent_of_x_l225_225934


namespace no_t_for_xyz_equal_l225_225314

theorem no_t_for_xyz_equal (t : ℝ) (x y z : ℝ) : 
  (x = 1 - 3 * t) → 
  (y = 2 * t - 3) → 
  (z = 4 * t^2 - 5 * t + 1) → 
  ¬ (x = y ∧ y = z) := 
by
  intro h1 h2 h3 h4
  have h5 : t = 4 / 5 := 
    by linarith [h1, h2, h4]
  rw [h5] at h3
  sorry

end no_t_for_xyz_equal_l225_225314


namespace triangle_area_l225_225823

theorem triangle_area : 
  let line_eq (x y : ℝ) := 3 * x + 2 * y = 12
  let x_intercept := (4 : ℝ)
  let y_intercept := (6 : ℝ)
  ∃ (x y : ℝ), line_eq x y ∧ x = x_intercept ∧ y = y_intercept ∧
  ∃ (area : ℝ), area = 1 / 2 * x * y ∧ area = 12 :=
by
  sorry

end triangle_area_l225_225823


namespace sum_of_inscribed_sphere_volumes_l225_225095

theorem sum_of_inscribed_sphere_volumes :
  let height := 3
  let angle := Real.pi / 3
  let r₁ := height / 3 -- Radius of the first inscribed sphere
  let geometric_ratio := 1 / 3
  let volume (r : ℝ) := (4 / 3) * Real.pi * r^3
  let volumes : ℕ → ℝ := λ n => volume (r₁ * geometric_ratio^(n - 1))
  let total_volume := ∑' n, volumes n
  total_volume = (18 * Real.pi) / 13 :=
by
  sorry

end sum_of_inscribed_sphere_volumes_l225_225095


namespace integer_solutions_abs_inequality_l225_225982

-- Define the condition as a predicate
def abs_inequality_condition (x : ℝ) : Prop := |x - 4| ≤ 3

-- State the proposition
theorem integer_solutions_abs_inequality : ∃ (n : ℕ), n = 7 ∧ ∀ (x : ℤ), abs_inequality_condition x → (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7) :=
sorry

end integer_solutions_abs_inequality_l225_225982


namespace ratio_of_ages_l225_225942

-- Definitions of the conditions
def son_current_age : ℕ := 28
def man_current_age : ℕ := son_current_age + 30
def son_age_in_two_years : ℕ := son_current_age + 2
def man_age_in_two_years : ℕ := man_current_age + 2

-- The theorem
theorem ratio_of_ages : (man_age_in_two_years / son_age_in_two_years) = 2 :=
by
  -- Skipping the proof steps
  sorry

end ratio_of_ages_l225_225942


namespace sector_central_angle_l225_225713

theorem sector_central_angle (r θ : ℝ) (h1 : 2 * r + r * θ = 6) (h2 : 0.5 * r * r * θ = 2) : θ = 1 ∨ θ = 4 :=
sorry

end sector_central_angle_l225_225713


namespace tan_product_pi_nine_l225_225254

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end tan_product_pi_nine_l225_225254


namespace geometric_body_is_cylinder_l225_225039

def top_view_is_circle : Prop := sorry

def is_prism_or_cylinder : Prop := sorry

theorem geometric_body_is_cylinder 
  (h1 : top_view_is_circle) 
  (h2 : is_prism_or_cylinder) 
  : Cylinder := 
sorry

end geometric_body_is_cylinder_l225_225039


namespace largest_divisor_of_n_l225_225651

theorem largest_divisor_of_n (n : ℕ) (h_pos : 0 < n) (h_div : 72 ∣ n^2) : 12 ∣ n :=
by
  sorry

end largest_divisor_of_n_l225_225651


namespace anika_more_than_twice_reeta_l225_225957

theorem anika_more_than_twice_reeta (R A M : ℕ) (h1 : R = 20) (h2 : A + R = 64) (h3 : A = 2 * R + M) : M = 4 :=
by
  sorry

end anika_more_than_twice_reeta_l225_225957


namespace mean_exercise_days_correct_l225_225455

def students_exercise_days : List (Nat × Nat) := 
  [ (2, 0), (4, 1), (5, 2), (7, 3), (5, 4), (3, 5), (1, 6)]

def total_days_exercised : Nat := 
  List.sum (students_exercise_days.map (λ (count, days) => count * days))

def total_students : Nat := 
  List.sum (students_exercise_days.map Prod.fst)

def mean_exercise_days : Float := 
  total_days_exercised.toFloat / total_students.toFloat

theorem mean_exercise_days_correct : Float.round (mean_exercise_days * 100) / 100 = 2.81 :=
by
  sorry -- proof not required

end mean_exercise_days_correct_l225_225455


namespace remainder_4059_div_32_l225_225792

theorem remainder_4059_div_32 : 4059 % 32 = 27 := by
  sorry

end remainder_4059_div_32_l225_225792


namespace sin_double_angle_l225_225700

theorem sin_double_angle (x : ℝ) (h : Real.sin (Real.pi / 4 - x) = 3 / 5) : Real.sin (2 * x) = 7 / 25 := by
  sorry

end sin_double_angle_l225_225700


namespace tickets_sold_l225_225503

theorem tickets_sold (T : ℕ) (h1 : 3 * T / 4 > 0)
    (h2 : 5 * (T / 4) / 9 > 0)
    (h3 : 80 > 0)
    (h4 : 20 > 0) :
    (1 / 4 * T - 5 / 36 * T = 100) -> T = 900 :=
by
  sorry

end tickets_sold_l225_225503


namespace vertex_on_x_axis_l225_225513

theorem vertex_on_x_axis (m : ℝ) : 
  (∃ x : ℝ, x^2 - 8 * x + m = 0) ↔ m = 16 :=
by
  sorry

end vertex_on_x_axis_l225_225513


namespace jills_present_age_l225_225784

-- Define the problem parameters and conditions
variables (H J : ℕ)
axiom cond1 : H + J = 43
axiom cond2 : H - 5 = 2 * (J - 5)

-- State the goal
theorem jills_present_age : J = 16 :=
sorry

end jills_present_age_l225_225784


namespace infinite_triangular_pairs_l225_225785

theorem infinite_triangular_pairs : ∃ (a_i b_i : ℕ → ℕ), (∀ m : ℕ, ∃ n : ℕ, m = n * (n + 1) / 2 ↔ ∃ k : ℕ, a_i k * m + b_i k = k * (k + 1) / 2) ∧ ∀ j : ℕ, ∃ k : ℕ, k > j :=
by {
  sorry
}

end infinite_triangular_pairs_l225_225785


namespace sum_geometric_sequence_first_eight_terms_l225_225085

theorem sum_geometric_sequence_first_eight_terms :
  let a_0 := (1 : ℚ) / 3
  let r := (1 : ℚ) / 3
  let n := 8
  let S_n := a_0 * (1 - r^n) / (1 - r)
  S_n = 6560 / 19683 := 
by
  sorry

end sum_geometric_sequence_first_eight_terms_l225_225085


namespace triangle_side_BC_length_l225_225739

noncomputable def triangle_side_length
  (AB : ℝ) (angle_a : ℝ) (angle_c : ℝ) : ℝ := 
  let sin_a := Real.sin angle_a
  let sin_c := Real.sin angle_c
  (AB * sin_a) / sin_c

theorem triangle_side_BC_length (AB : ℝ) (angle_a angle_c : ℝ) :
  AB = (Real.sqrt 6) / 2 →
  angle_a = (45 * Real.pi / 180) →
  angle_c = (60 * Real.pi / 180) →
  triangle_side_length AB angle_a angle_c = 1 :=
sorry

end triangle_side_BC_length_l225_225739


namespace units_digit_13_pow_2003_l225_225378

theorem units_digit_13_pow_2003 : (13 ^ 2003) % 10 = 7 := by
  sorry

end units_digit_13_pow_2003_l225_225378


namespace jason_seashells_after_giving_l225_225137

-- Define the number of seashells Jason originally found
def original_seashells : ℕ := 49

-- Define the number of seashells Jason gave to Tim
def seashells_given : ℕ := 13

-- Prove that the number of seashells Jason now has is 36
theorem jason_seashells_after_giving : original_seashells - seashells_given = 36 :=
by
  -- This is where the proof would go
  sorry

end jason_seashells_after_giving_l225_225137


namespace gcd_digit_bound_l225_225578

theorem gcd_digit_bound
  (a b : ℕ)
  (h1 : 10^6 ≤ a)
  (h2 : a < 10^7)
  (h3 : 10^6 ≤ b)
  (h4 : b < 10^7)
  (h_lcm : 10^{11} ≤ Nat.lcm a b)
  (h_lcm2 : Nat.lcm a b < 10^{12}) :
  Nat.gcd a b < 10^3 :=
sorry

end gcd_digit_bound_l225_225578


namespace ratio_of_female_to_male_officers_on_duty_l225_225004

theorem ratio_of_female_to_male_officers_on_duty 
    (p : ℝ) (T : ℕ) (F : ℕ) 
    (hp : p = 0.19) (hT : T = 152) (hF : F = 400) : 
    (76 / 76) = 1 :=
by
  sorry

end ratio_of_female_to_male_officers_on_duty_l225_225004


namespace train_length_l225_225953

theorem train_length 
    (t : ℝ) 
    (s_kmh : ℝ) 
    (s_mps : ℝ)
    (h1 : t = 2.222044458665529) 
    (h2 : s_kmh = 162) 
    (h3 : s_mps = s_kmh * (5 / 18))
    (L : ℝ)
    (h4 : L = s_mps * t) : 
  L = 100 := 
sorry

end train_length_l225_225953


namespace max_n_consecutive_sum_2014_l225_225595

theorem max_n_consecutive_sum_2014 : 
  ∃ (k n : ℕ), (2 * k + n - 1) * n = 4028 ∧ n = 53 ∧ k > 0 := sorry

end max_n_consecutive_sum_2014_l225_225595


namespace num_rows_seat_9_people_l225_225848

-- Define the premises of the problem.
def seating_arrangement (x y : ℕ) : Prop := (9 * x + 7 * y = 58)

-- The theorem stating the number of rows seating exactly 9 people.
theorem num_rows_seat_9_people
  (x y : ℕ)
  (h : seating_arrangement x y) :
  x = 1 :=
by
  -- Proof is not required as per the instruction
  sorry

end num_rows_seat_9_people_l225_225848


namespace oil_price_reduction_l225_225390

theorem oil_price_reduction (P P_reduced : ℝ) (h1 : P_reduced = 50) (h2 : 1000 / P_reduced - 5 = 5) :
  ((P - P_reduced) / P) * 100 = 25 := by
  sorry

end oil_price_reduction_l225_225390


namespace centroids_coincide_l225_225540

noncomputable def centroid (A B C : ℂ) : ℂ :=
  (A + B + C) / 3

theorem centroids_coincide (A B C : ℂ) (k : ℝ) (C1 A1 B1 : ℂ)
  (h1 : C1 = k * (B - A) + A)
  (h2 : A1 = k * (C - B) + B)
  (h3 : B1 = k * (A - C) + C) :
  centroid A1 B1 C1 = centroid A B C := by
  sorry

end centroids_coincide_l225_225540


namespace jimmy_points_lost_for_bad_behavior_l225_225140

theorem jimmy_points_lost_for_bad_behavior (points_per_exam : ℕ) (num_exams : ℕ) (points_needed : ℕ)
  (extra_points_allowed : ℕ) (total_points_earned : ℕ) (current_points : ℕ)
  (h1 : points_per_exam = 20) (h2 : num_exams = 3) (h3 : points_needed = 50)
  (h4 : extra_points_allowed = 5) (h5 : total_points_earned = points_per_exam * num_exams)
  (h6 : current_points = points_needed + extra_points_allowed) :
  total_points_earned - current_points = 5 :=
by
  sorry

end jimmy_points_lost_for_bad_behavior_l225_225140


namespace no_term_in_sequence_is_3_alpha_5_beta_l225_225703

theorem no_term_in_sequence_is_3_alpha_5_beta :
  ∀ (v : ℕ → ℕ),
    v 0 = 0 →
    v 1 = 1 →
    (∀ n, 1 ≤ n → v (n + 1) = 8 * v n * v (n - 1)) →
    ∀ n, ∀ (α β : ℕ), α > 0 → β > 0 → v n ≠ 3^α * 5^β := by
  intros v h0 h1 recurrence n α β hα hβ
  sorry

end no_term_in_sequence_is_3_alpha_5_beta_l225_225703


namespace regular_octagon_side_length_l225_225160

theorem regular_octagon_side_length
  (side_length_pentagon : ℕ)
  (total_wire_length : ℕ)
  (side_length_octagon : ℕ) :
  side_length_pentagon = 16 →
  total_wire_length = 5 * side_length_pentagon →
  side_length_octagon = total_wire_length / 8 →
  side_length_octagon = 10 := 
sorry

end regular_octagon_side_length_l225_225160


namespace loaves_count_l225_225815

theorem loaves_count (initial_loaves afternoon_sales evening_delivery end_day_loaves: ℕ)
  (h_initial: initial_loaves = 2355)
  (h_sales: afternoon_sales = 629)
  (h_delivery: evening_delivery = 489)
  (h_end: end_day_loaves = 2215) :
  initial_loaves - afternoon_sales + evening_delivery = end_day_loaves :=
  by {
    rw [h_initial, h_sales, h_delivery, h_end],
    sorry
  }

end loaves_count_l225_225815


namespace isosceles_triangle_base_length_l225_225057

theorem isosceles_triangle_base_length (x : ℝ) (h1 : 2 * x + 2 * x + x = 20) : x = 4 :=
sorry

end isosceles_triangle_base_length_l225_225057


namespace exponentiation_problem_l225_225673

theorem exponentiation_problem : (8^8 / 8^5) * 2^10 * 2^3 = 2^22 := by
  sorry

end exponentiation_problem_l225_225673


namespace betty_total_oranges_l225_225668

-- Definitions for the given conditions
def boxes : ℝ := 3.0
def oranges_per_box : ℝ := 24

-- Theorem statement to prove the correct answer to the problem
theorem betty_total_oranges : boxes * oranges_per_box = 72 := by
  sorry

end betty_total_oranges_l225_225668


namespace prime_exists_solution_l225_225770

theorem prime_exists_solution (p : ℕ) [hp : Fact p.Prime] :
  ∃ n : ℕ, (6 * n^2 + 5 * n + 1) % p = 0 :=
by
  sorry

end prime_exists_solution_l225_225770


namespace solve_for_x_l225_225036

theorem solve_for_x (x : ℝ) (h : 8 / x + 6 = 8) : x = 4 :=
sorry

end solve_for_x_l225_225036


namespace carson_pumps_needed_l225_225415

theorem carson_pumps_needed 
  (full_tire_capacity : ℕ) (flat_tires_count : ℕ) 
  (full_percentage_tire_1 : ℚ) (full_percentage_tire_2 : ℚ)
  (air_per_pump : ℕ) : 
  flat_tires_count = 2 →
  full_tire_capacity = 500 →
  full_percentage_tire_1 = 0.40 →
  full_percentage_tire_2 = 0.70 →
  air_per_pump = 50 →
  let needed_air_flat_tires := flat_tires_count * full_tire_capacity
  let needed_air_tire_1 := (1 - full_percentage_tire_1) * full_tire_capacity
  let needed_air_tire_2 := (1 - full_percentage_tire_2) * full_tire_capacity
  let total_needed_air := needed_air_flat_tires + needed_air_tire_1 + needed_air_tire_2
  let pumps_needed := total_needed_air / air_per_pump
  pumps_needed = 29 := 
by
  intros
  sorry

end carson_pumps_needed_l225_225415


namespace solve_problem_l225_225343

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) :
  f (x^3 + y^3) = (x + y) * ((f x)^2 - (f x) * (f y) + (f (f y))^2)

theorem solve_problem (x : ℝ) : f (1996 * x) = 1996 * f x :=
sorry

end solve_problem_l225_225343


namespace soda_difference_l225_225754

theorem soda_difference :
  let Julio_orange_bottles := 4
  let Julio_grape_bottles := 7
  let Mateo_orange_bottles := 1
  let Mateo_grape_bottles := 3
  let liters_per_bottle := 2
  let Julio_total_liters := Julio_orange_bottles * liters_per_bottle + Julio_grape_bottles * liters_per_bottle
  let Mateo_total_liters := Mateo_orange_bottles * liters_per_bottle + Mateo_grape_bottles * liters_per_bottle
  Julio_total_liters - Mateo_total_liters = 14 := by
    sorry

end soda_difference_l225_225754


namespace vector_sum_is_correct_l225_225456

-- Define the points A, B, and C
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (0, 1)

-- Define the vectors AB and AC
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vectorAC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

-- State the theorem
theorem vector_sum_is_correct : vectorAB + vectorAC = (-3, -1) :=
by
  sorry

end vector_sum_is_correct_l225_225456


namespace cost_price_equals_selling_price_l225_225633

theorem cost_price_equals_selling_price (C : ℝ) (x : ℝ) (h1 : 20 * C = 1.25 * C * x) : x = 16 :=
by
  -- This proof is omitted at the moment
  sorry

end cost_price_equals_selling_price_l225_225633


namespace dice_multiple_3_prob_l225_225948

-- Define the probability calculations for the problem
noncomputable def single_roll_multiple_3_prob: ℝ := 1 / 3
noncomputable def single_roll_not_multiple_3_prob: ℝ := 1 - single_roll_multiple_3_prob
noncomputable def eight_rolls_not_multiple_3_prob: ℝ := (single_roll_not_multiple_3_prob) ^ 8
noncomputable def at_least_one_roll_multiple_3_prob: ℝ := 1 - eight_rolls_not_multiple_3_prob

-- The lean theorem statement
theorem dice_multiple_3_prob : 
  at_least_one_roll_multiple_3_prob = 6305 / 6561 := by 
sorry

end dice_multiple_3_prob_l225_225948


namespace factor_polynomial_l225_225168

theorem factor_polynomial (x y z : ℤ) :
  x * (y - z) ^ 3 + y * (z - x) ^ 3 + z * (x - y) ^ 3 = (x - y) * (y - z) * (z - x) * (x + y + z) := 
by
  sorry

end factor_polynomial_l225_225168


namespace binom_10_2_eq_45_l225_225679

-- Definitions used in the conditions
def binom (n k : ℕ) := n.choose k

-- The statement that needs to be proven
theorem binom_10_2_eq_45 : binom 10 2 = 45 :=
by
  sorry

end binom_10_2_eq_45_l225_225679


namespace area_of_triangle_bounded_by_line_and_axes_l225_225830

theorem area_of_triangle_bounded_by_line_and_axes (x y : ℝ) (hx : 3 * x + 2 * y = 12) :
  ∃ (area : ℝ), area = 12 := by
sorry

end area_of_triangle_bounded_by_line_and_axes_l225_225830


namespace sqrt_2x_plus_y_eq_4_l225_225100

theorem sqrt_2x_plus_y_eq_4 (x y : ℝ) 
  (h1 : (3 * x + 1) = 4) 
  (h2 : (2 * y - 1) = 27) : 
  Real.sqrt (2 * x + y) = 4 := 
by 
  sorry

end sqrt_2x_plus_y_eq_4_l225_225100


namespace tan_product_l225_225236

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end tan_product_l225_225236


namespace simplify_expression_l225_225225

theorem simplify_expression (a : ℤ) : 7 * a - 3 * a = 4 * a :=
by
  sorry

end simplify_expression_l225_225225


namespace oj_fraction_is_11_over_30_l225_225644

-- Define the capacity of each pitcher
def pitcher_capacity : ℕ := 600

-- Define the fraction of orange juice in each pitcher
def fraction_oj_pitcher1 : ℚ := 1 / 3
def fraction_oj_pitcher2 : ℚ := 2 / 5

-- Define the amount of orange juice in each pitcher
def oj_amount_pitcher1 := pitcher_capacity * fraction_oj_pitcher1
def oj_amount_pitcher2 := pitcher_capacity * fraction_oj_pitcher2

-- Define the total amount of orange juice after both pitchers are poured into the large container
def total_oj_amount := oj_amount_pitcher1 + oj_amount_pitcher2

-- Define the total volume of the mixture in the large container
def total_mixture_volume := 2 * pitcher_capacity

-- Define the fraction of the mixture that is orange juice
def oj_fraction_in_mixture := total_oj_amount / total_mixture_volume

-- Prove that the fraction of the mixture that is orange juice is 11/30
theorem oj_fraction_is_11_over_30 : oj_fraction_in_mixture = 11 / 30 := by
  sorry

end oj_fraction_is_11_over_30_l225_225644


namespace calculate_area_of_triangle_tangent_line_l225_225065

noncomputable def areaOfTriangleTangent := 
  let f : ℝ → ℝ := λ x, Real.exp (-x)
  let M := (1 : ℝ, Real.exp (-1))
  let tangentLine := λ x, -Real.exp (-1) * x + 2 * Real.exp (-1)
  let A := (2 : ℝ, 0)
  let B := (0 : ℝ, 2 * Real.exp (-1))
  let C := (0 : ℝ, 0)
  let base := 2 - 0
  let height := 2 * Real.exp (-1)
  (1 / 2) * base * height

theorem calculate_area_of_triangle_tangent_line :
  areaOfTriangleTangent = 2 / Real.exp 1 :=
by
  sorry

end calculate_area_of_triangle_tangent_line_l225_225065


namespace total_frogs_in_pond_l225_225884

def frogsOnLilyPads : ℕ := 5
def frogsOnLogs : ℕ := 3
def babyFrogsOnRock : ℕ := 2 * 12 -- Two dozen

theorem total_frogs_in_pond : frogsOnLilyPads + frogsOnLogs + babyFrogsOnRock = 32 :=
by
  sorry

end total_frogs_in_pond_l225_225884


namespace teachers_students_relationship_l225_225896

variables (m n k l : ℕ)

theorem teachers_students_relationship
  (teachers_count : m > 0)
  (students_count : n > 0)
  (students_per_teacher : k > 0)
  (teachers_per_student : l > 0)
  (h1 : ∀ p ∈ (Finset.range m), (Finset.card (Finset.range k)) = k)
  (h2 : ∀ s ∈ (Finset.range n), (Finset.card (Finset.range l)) = l) :
  m * k = n * l :=
sorry

end teachers_students_relationship_l225_225896


namespace train_length_l225_225663

-- Define the given speeds and time
def train_speed_km_per_h := 25
def man_speed_km_per_h := 2
def crossing_time_sec := 36

-- Convert speeds to m/s
def km_per_h_to_m_per_s (v : ℕ) : ℕ := (v * 1000) / 3600
def train_speed_m_per_s := km_per_h_to_m_per_s train_speed_km_per_h
def man_speed_m_per_s := km_per_h_to_m_per_s man_speed_km_per_h

-- Define the relative speed in m/s
def relative_speed_m_per_s := train_speed_m_per_s + man_speed_m_per_s

-- Theorem to prove the length of the train
theorem train_length : (relative_speed_m_per_s * crossing_time_sec) = 270 :=
by
  -- sorry is used to skip the proof
  sorry

end train_length_l225_225663


namespace tan_product_l225_225275

theorem tan_product : 
  tan (Real.pi / 9) * tan (2 * Real.pi / 9) * tan (4 * Real.pi / 9) = 3 :=
by
  sorry

end tan_product_l225_225275


namespace gcd_digits_le_3_l225_225585

theorem gcd_digits_le_3 (a b : ℕ) (h_a : 10^6 ≤ a < 10^7) (h_b : 10^6 ≤ b < 10^7) (h_lcm : 10^11 ≤ Nat.lcm a b):
  Nat.gcd a b < 1000 := 
sorry

end gcd_digits_le_3_l225_225585


namespace gcd_has_at_most_3_digits_l225_225574

noncomputable def lcm (a b : ℕ) : ℕ := sorry -- definition for lcm is already in Mathlib
noncomputable def gcd (a b : ℕ) : ℕ := sorry -- definition for gcd is already in Mathlib

theorem gcd_has_at_most_3_digits
  (a b : ℕ)
  (ha : 10^6 ≤ a ∧ a < 10^7)  -- a is a 7-digit integer
  (hb : 10^6 ≤ b ∧ b < 10^7)  -- b is a 7-digit integer
  (hlcm_digits : 10^11 ≤ lcm a b ∧ lcm a b < 10^12)  -- lcm of a and b has 12 digits
  : gcd a b < 10^3 := by
  sorry

end gcd_has_at_most_3_digits_l225_225574


namespace find_x_l225_225309

-- Definitions for the vectors and their relationships
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)
def u (x : ℝ) : ℝ × ℝ := (a.1 + 2 * (b x).1, a.2 + 2 * (b x).2)
def v (x : ℝ) : ℝ × ℝ := (2 * a.1 - (b x).1, 2 * a.2 - (b x).2)

-- Given condition that u is parallel to v
def u_parallel_v (x : ℝ) : Prop := u x = v x

-- Prove that the value of x is 1/2
theorem find_x : ∃ x : ℝ, u_parallel_v x ∧ x = 1 / 2 := 
sorry

end find_x_l225_225309


namespace inequality_solution_l225_225476

theorem inequality_solution :
  {x : ℝ // -1 < (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) ∧ (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) < 1} = 
  {x : ℝ // x > 1/6} :=
sorry

end inequality_solution_l225_225476


namespace ed_pets_count_l225_225849

theorem ed_pets_count : 
  let dogs := 2 
  let cats := 3 
  let fish := 2 * (cats + dogs) 
  let birds := dogs * cats 
  dogs + cats + fish + birds = 21 := 
by
  sorry

end ed_pets_count_l225_225849


namespace find_a_l225_225457

theorem find_a 
  (x y a m n : ℝ)
  (h1 : x - 5 / 2 * y + 1 = 0) 
  (h2 : x = m + a) 
  (h3 : y = n + 1)  -- since k = 1, so we replace k with 1
  (h4 : m + a = m + 1 / 2) : 
  a = 1 / 2 := 
by 
  sorry

end find_a_l225_225457


namespace min_value_lemma_min_value_achieved_l225_225536

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((1 - x)^2 + (1 + x)^2)

theorem min_value_lemma : ∀ (x : ℝ), f x ≥ Real.sqrt 5 := 
by
  intro x
  sorry

theorem min_value_achieved : ∃ (x : ℝ), f x = Real.sqrt 5 :=
by
  use 1 / 3
  sorry

end min_value_lemma_min_value_achieved_l225_225536


namespace triangle_sin_double_angle_l225_225315

open Real

theorem triangle_sin_double_angle (A : ℝ) (h : cos (π / 4 + A) = 5 / 13) : sin (2 * A) = 119 / 169 :=
by
  sorry

end triangle_sin_double_angle_l225_225315


namespace coins_player_1_received_l225_225179

def round_table := List Nat
def players := List Nat
def coins_received (table: round_table) (player_idx: Nat) : Nat :=
sorry -- the function to calculate coins received by player's index

-- Define the given conditions
def sectors : round_table := [1, 2, 3, 4, 5, 6, 7, 8, 9]
def num_players := 9
def num_rotations := 11
def player_4 := 4
def player_8 := 8
def player_1 := 1
def coins_player_4 := 90
def coins_player_8 := 35

theorem coins_player_1_received : coins_received sectors player_1 = 57 :=
by
  -- Setup the conditions
  have h1 : coins_received sectors player_4 = 90 := sorry
  have h2 : coins_received sectors player_8 = 35 := sorry
  -- Prove the target statement
  show coins_received sectors player_1 = 57
  sorry

end coins_player_1_received_l225_225179


namespace binom_10_2_eq_45_l225_225676

theorem binom_10_2_eq_45 : Nat.binomial 10 2 = 45 := by
  sorry

end binom_10_2_eq_45_l225_225676


namespace circle_properties_l225_225336

noncomputable def circle_center_and_radius (x y: ℝ) : Prop :=
  (x^2 + 8*x + y^2 - 10*y = 11)

theorem circle_properties :
  (∃ (a b r : ℝ), (a, b) = (-4, 5) ∧ r = 2 * Real.sqrt 13 ∧ circle_center_and_radius x y → a + b + r = 1 + 2 * Real.sqrt 13) :=
  sorry

end circle_properties_l225_225336


namespace technology_courses_correct_l225_225322

variable (m : ℕ)

def subject_courses := m
def arts_courses := subject_courses + 9
def technology_courses := 1 / 3 * arts_courses + 5

theorem technology_courses_correct : technology_courses = 1 / 3 * m + 8 := by
  sorry

end technology_courses_correct_l225_225322


namespace marcus_goal_points_value_l225_225764

-- Definitions based on conditions
def marcus_goals_first_type := 5
def marcus_goals_second_type := 10
def second_type_goal_points := 2
def team_total_points := 70
def marcus_percentage_points := 50

-- Theorem statement
theorem marcus_goal_points_value : 
  ∃ (x : ℕ), 5 * x + 10 * 2 = 35 ∧ 35 = 50 * team_total_points / 100 := 
sorry

end marcus_goal_points_value_l225_225764


namespace YoongiHasSevenPets_l225_225498

def YoongiPets (dogs cats : ℕ) : ℕ := dogs + cats

theorem YoongiHasSevenPets : YoongiPets 5 2 = 7 :=
by
  sorry

end YoongiHasSevenPets_l225_225498


namespace basketball_team_wins_l225_225658

theorem basketball_team_wins (f : ℚ) (h1 : 40 + 40 * f + (40 + 40 * f) = 130) : f = 5 / 8 :=
by
  sorry

end basketball_team_wins_l225_225658


namespace cyclic_identity_l225_225357

theorem cyclic_identity (a b c : ℝ) :
  a * (a - c)^2 + b * (b - c)^2 - (a - c) * (b - c) * (a + b - c) =
  b * (b - a)^2 + c * (c - a)^2 - (b - a) * (c - a) * (b + c - a) ∧
  b * (b - a)^2 + c * (c - a)^2 - (b - a) * (c - a) * (b + c - a) =
  c * (c - b)^2 + a * (a - b)^2 - (c - b) * (a - b) * (c + a - b) := by
sorry

end cyclic_identity_l225_225357


namespace soda_difference_l225_225753

theorem soda_difference :
  let Julio_orange_bottles := 4
  let Julio_grape_bottles := 7
  let Mateo_orange_bottles := 1
  let Mateo_grape_bottles := 3
  let liters_per_bottle := 2
  let Julio_total_liters := Julio_orange_bottles * liters_per_bottle + Julio_grape_bottles * liters_per_bottle
  let Mateo_total_liters := Mateo_orange_bottles * liters_per_bottle + Mateo_grape_bottles * liters_per_bottle
  Julio_total_liters - Mateo_total_liters = 14 := by
    sorry

end soda_difference_l225_225753


namespace rate_percent_l225_225791

noncomputable def calculate_rate (P: ℝ) : ℝ :=
  let I : ℝ := 320
  let t : ℝ := 2
  I * 100 / (P * t)

theorem rate_percent (P: ℝ) (hP: P > 0) : calculate_rate P = 4 := 
by
  sorry

end rate_percent_l225_225791


namespace competition_total_races_l225_225318

theorem competition_total_races (sprinters : ℕ) (sprinters_with_bye : ℕ) (lanes_preliminary : ℕ) (lanes_subsequent : ℕ) 
  (eliminated_per_race : ℕ) (first_round_advance : ℕ) (second_round_advance : ℕ) (third_round_advance : ℕ) 
  : sprinters = 300 → sprinters_with_bye = 16 → lanes_preliminary = 8 → lanes_subsequent = 6 → 
    eliminated_per_race = 7 → first_round_advance = 36 → second_round_advance = 9 → third_round_advance = 2 
    → first_round_races = 36 → second_round_races = 9 → third_round_races = 2 → final_race = 1
    → first_round_races + second_round_races + third_round_races + final_race = 48 :=
by 
  intros sprinters_eq sprinters_with_bye_eq lanes_preliminary_eq lanes_subsequent_eq eliminated_per_race_eq 
         first_round_advance_eq second_round_advance_eq third_round_advance_eq 
         first_round_races_eq second_round_races_eq third_round_races_eq final_race_eq
  sorry

end competition_total_races_l225_225318


namespace words_on_each_page_l225_225209

theorem words_on_each_page (p : ℕ) (h : 150 * p ≡ 198 [MOD 221]) : p = 93 :=
sorry

end words_on_each_page_l225_225209


namespace proof_valid_set_exists_l225_225471

noncomputable def valid_set_exists : Prop :=
∃ (s : Finset ℕ), s.card = 10 ∧ 
(∀ (a b : ℕ), a ∈ s → b ∈ s → a ≠ b → a ≠ b) ∧ 
(∃ (t1 : Finset ℕ), t1 ⊆ s ∧ t1.card = 3 ∧ ∀ n ∈ t1, 5 ∣ n) ∧
(∃ (t2 : Finset ℕ), t2 ⊆ s ∧ t2.card = 4 ∧ ∀ n ∈ t2, 4 ∣ n) ∧
s.sum id < 75

theorem proof_valid_set_exists : valid_set_exists :=
sorry

end proof_valid_set_exists_l225_225471


namespace triangle_trig_problems_l225_225546

open Real

-- Define the main theorem
theorem triangle_trig_problems (A B C a b c : ℝ) (h1: b ≠ 0) 
  (h2: cos A - 2 * cos C ≠ 0) 
  (h3 : (cos A - 2 * cos C) / cos B = (2 * c - a) / b) 
  (h4 : cos B = 1/4)
  (h5 : b = 2) :
  (sin C / sin A = 2) ∧ 
  (2 * a * c * sqrt 15 / 4 = sqrt 15 / 4) :=
by 
  sorry

end triangle_trig_problems_l225_225546


namespace series_sum_l225_225467

noncomputable def S (n : ℕ) : ℝ := 2^(n + 1) + n - 2

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then S 1 else S n - S (n - 1)

theorem series_sum : 
  ∑' i, a i / 4^i = 4 / 3 :=
by 
  sorry

end series_sum_l225_225467


namespace blanch_breakfast_slices_l225_225063

-- Define the initial number of slices
def initial_slices : ℕ := 15

-- Define the slices eaten at different times
def lunch_slices : ℕ := 2
def snack_slices : ℕ := 2
def dinner_slices : ℕ := 5

-- Define the number of slices left
def slices_left : ℕ := 2

-- Calculate the total slices eaten during lunch, snack, and dinner
def total_eaten_ex_breakfast : ℕ := lunch_slices + snack_slices + dinner_slices

-- Define the slices eaten during breakfast
def breakfast_slices : ℕ := initial_slices - total_eaten_ex_breakfast - slices_left

-- The theorem to prove
theorem blanch_breakfast_slices : breakfast_slices = 4 := by
  sorry

end blanch_breakfast_slices_l225_225063


namespace total_frogs_seen_by_hunter_l225_225878

/-- Hunter saw 5 frogs sitting on lily pads in the pond. -/
def initial_frogs : ℕ := 5

/-- Three more frogs climbed out of the water onto logs floating in the pond. -/
def frogs_on_logs : ℕ := 3

/-- Two dozen baby frogs (24 frogs) hopped onto a big rock jutting out from the pond. -/
def baby_frogs : ℕ := 24

/--
The total number of frogs Hunter saw in the pond.
-/
theorem total_frogs_seen_by_hunter : initial_frogs + frogs_on_logs + baby_frogs = 32 := by
sorry

end total_frogs_seen_by_hunter_l225_225878


namespace change_amount_l225_225421

theorem change_amount 
    (tank_capacity : ℕ) 
    (current_fuel : ℕ) 
    (price_per_liter : ℕ) 
    (total_money : ℕ) 
    (full_tank : tank_capacity = 150) 
    (fuel_in_truck : current_fuel = 38) 
    (cost_per_liter : price_per_liter = 3) 
    (money_with_donny : total_money = 350) : 
    total_money - ((tank_capacity - current_fuel) * price_per_liter) = 14 :=
by
sorr

end change_amount_l225_225421


namespace range_of_a_l225_225552

-- Defining the problem conditions
def f (x : ℝ) : ℝ := sorry -- The function f : ℝ → ℝ is defined elsewhere such that its range is [0, 4]
def g (a x : ℝ) : ℝ := a * x - 1

-- Theorem to prove the range of 'a'
theorem range_of_a (a : ℝ) : (a ≥ 1/2) ∨ (a ≤ -1/2) :=
sorry

end range_of_a_l225_225552


namespace kat_average_training_hours_l225_225609

def strength_training_sessions_per_week : ℕ := 3
def strength_training_hour_per_session : ℕ := 1
def strength_training_missed_sessions_per_2_weeks : ℕ := 1

def boxing_training_sessions_per_week : ℕ := 4
def boxing_training_hour_per_session : ℝ := 1.5
def boxing_training_skipped_sessions_per_2_weeks : ℕ := 1

def cardio_workout_sessions_per_week : ℕ := 2
def cardio_workout_minutes_per_session : ℕ := 30

def flexibility_training_sessions_per_week : ℕ := 1
def flexibility_training_minutes_per_session : ℕ := 45

def interval_training_sessions_per_week : ℕ := 1
def interval_training_hour_per_session : ℝ := 1.25 -- 1 hour and 15 minutes 

noncomputable def average_hours_per_week : ℝ :=
  let strength_training_per_week : ℝ := ((5 / 2) * strength_training_hour_per_session)
  let boxing_training_per_week : ℝ := ((7 / 2) * boxing_training_hour_per_session)
  let cardio_workout_per_week : ℝ := (cardio_workout_sessions_per_week * cardio_workout_minutes_per_session / 60)
  let flexibility_training_per_week : ℝ := (flexibility_training_sessions_per_week * flexibility_training_minutes_per_session / 60)
  let interval_training_per_week : ℝ := interval_training_hour_per_session
  strength_training_per_week + boxing_training_per_week + cardio_workout_per_week + flexibility_training_per_week + interval_training_per_week

theorem kat_average_training_hours : average_hours_per_week = 10.75 := by
  unfold average_hours_per_week
  norm_num
  sorry

end kat_average_training_hours_l225_225609


namespace speed_of_stream_l225_225388

theorem speed_of_stream (v : ℝ) (h1 : 22 > 0) (h2 : 8 > 0) (h3 : 216 = (22 + v) * 8) : v = 5 := 
by 
  sorry

end speed_of_stream_l225_225388


namespace find_t_l225_225460

-- Definitions from the given conditions
def earning (hours : ℕ) (rate : ℕ) : ℕ := hours * rate

-- The main theorem based on the translated problem
theorem find_t
  (t : ℕ)
  (h1 : earning (t - 4) (3 * t - 7) = earning (3 * t - 12) (t - 3)) :
  t = 4 := 
sorry

end find_t_l225_225460


namespace evaluate_expression_l225_225672

theorem evaluate_expression :
  -1^2008 + 3*(-1)^2007 + 1^2008 - 2*(-1)^2009 = -5 := 
by
  sorry

end evaluate_expression_l225_225672


namespace inequality_solution_set_l225_225638

noncomputable def solution_set := {x : ℝ | x^2 + 2 * x - 3 ≥ 0}

theorem inequality_solution_set :
  (solution_set = {x : ℝ | x ≤ -3 ∨ x ≥ 1}) :=
sorry

end inequality_solution_set_l225_225638


namespace scientific_notation_conversion_l225_225653

theorem scientific_notation_conversion : 450000000 = 4.5 * 10^8 :=
by
  sorry

end scientific_notation_conversion_l225_225653


namespace fifth_dog_is_older_than_fourth_l225_225562

theorem fifth_dog_is_older_than_fourth :
  ∀ (age_1 age_2 age_3 age_4 age_5 : ℕ),
  (age_1 = 10) →
  (age_2 = age_1 - 2) →
  (age_3 = age_2 + 4) →
  (age_4 = age_3 / 2) →
  (age_5 = age_4 + 20) →
  ((age_1 + age_5) / 2 = 18) →
  (age_5 - age_4 = 20) :=
by
  intros age_1 age_2 age_3 age_4 age_5 h1 h2 h3 h4 h5 h_avg
  sorry

end fifth_dog_is_older_than_fourth_l225_225562


namespace no_real_roots_quadratic_eq_l225_225636

theorem no_real_roots_quadratic_eq :
  ¬ ∃ x : ℝ, 7 * x^2 - 4 * x + 6 = 0 :=
by sorry

end no_real_roots_quadratic_eq_l225_225636


namespace common_ratio_geometric_series_l225_225851

-- Define the terms of the geometric series
def term (n : ℕ) : ℚ :=
  match n with
  | 0     => 7 / 8
  | 1     => -21 / 32
  | 2     => 63 / 128
  | _     => sorry  -- Placeholder for further terms if necessary

-- Define the common ratio
def common_ratio : ℚ := -3 / 4

-- Prove that the common ratio is consistent for the given series
theorem common_ratio_geometric_series :
  ∀ (n : ℕ), term (n + 1) / term n = common_ratio :=
by
  sorry

end common_ratio_geometric_series_l225_225851


namespace quadratic_form_l225_225023

-- Define the constants b and c based on the problem conditions
def b : ℤ := 900
def c : ℤ := -807300

-- Create a statement that represents the proof goal
theorem quadratic_form (c_eq : c = -807300) (b_eq : b = 900) : c / b = -897 :=
by
  sorry

end quadratic_form_l225_225023


namespace gcd_digit_bound_l225_225589

theorem gcd_digit_bound {a b : ℕ} (ha : 10^6 ≤ a ∧ a < 10^7) (hb : 10^6 ≤ b ∧ b < 10^7) (hlcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 10^3 :=
by
  sorry

end gcd_digit_bound_l225_225589


namespace g_prime_positive_l225_225102

noncomputable def f (a x : ℝ) := a * x - a * x ^ 2 - Real.log x

noncomputable def g (a x : ℝ) := -2 * (a * x - a * x ^ 2 - Real.log x) - (2 * a + 1) * x ^ 2 + a * x

def g_zero (a x1 x2 : ℝ) := g a x1 = 0 ∧ g a x2 = 0

def x1_x2_condition (x1 x2 : ℝ) := x1 < x2 ∧ x2 < 4 * x1

theorem g_prime_positive (a x1 x2 : ℝ) (h1 : g_zero a x1 x2) (h2 : x1_x2_condition x1 x2) :
  (deriv (g a) ((2 * x1 + x2) / 3)) > 0 := by
  sorry

end g_prime_positive_l225_225102


namespace prove_additional_minutes_needed_l225_225146

-- Assume the given conditions as definitions in Lean 4
def number_of_classmates := 30
def initial_gathering_time := 120   -- in minutes (2 hours)
def time_per_flower := 10           -- in minutes
def flowers_lost := 3

-- Calculate the flowers gathered initially
def initial_flowers_gathered := initial_gathering_time / time_per_flower

-- Calculate flowers remaining after loss
def flowers_remaining := initial_flowers_gathered - flowers_lost

-- Calculate additional flowers needed
def additional_flowers_needed := number_of_classmates - flowers_remaining

-- Therefore, calculate the additional minutes required to gather the remaining flowers
def additional_minutes_needed := additional_flowers_needed * time_per_flower

theorem prove_additional_minutes_needed :
  additional_minutes_needed = 210 :=
by 
  unfold additional_minutes_needed additional_flowers_needed flowers_remaining initial_flowers_gathered
  sorry

end prove_additional_minutes_needed_l225_225146


namespace propP_necessary_but_not_sufficient_l225_225301

open Function Real

variable (f : ℝ → ℝ)

-- Conditions: differentiable function f and the proposition Q
def diff_and_propQ (h_deriv : Differentiable ℝ f) : Prop :=
∀ x : ℝ, abs (deriv f x) < 2018

-- Proposition P
def propP : Prop :=
∀ x1 x2 : ℝ, x1 ≠ x2 → abs ((f x1 - f x2) / (x1 - x2)) < 2018

-- Final statement
theorem propP_necessary_but_not_sufficient (h_deriv : Differentiable ℝ f) (hQ : diff_and_propQ f h_deriv) : 
  (∀ x1 x2 : ℝ, x1 ≠ x2 → abs ((f x1 - f x2) / (x1 - x2)) < 2018) ∧ 
  ¬(∀ x : ℝ, abs (deriv f x) < 2018 ↔ ∀ x1 x2 : ℝ, x1 ≠ x2 → abs ((f x1 - f x2) / (x1 - x2)) < 2018) :=
by
  sorry

end propP_necessary_but_not_sufficient_l225_225301


namespace bill_difference_is_zero_l225_225522

theorem bill_difference_is_zero
    (a b : ℝ)
    (h1 : 0.25 * a = 5)
    (h2 : 0.15 * b = 3) :
    a - b = 0 := 
by 
  sorry

end bill_difference_is_zero_l225_225522


namespace plot_length_l225_225213

-- Define the conditions
def rent_per_acre_per_month : ℝ := 30
def total_rent_per_month : ℝ := 300
def width_feet : ℝ := 1210
def area_acres : ℝ := 10
def square_feet_per_acre : ℝ := 43560

-- Prove that the length of the plot is 360 feet
theorem plot_length (h1 : rent_per_acre_per_month = 30)
                    (h2 : total_rent_per_month = 300)
                    (h3 : width_feet = 1210)
                    (h4 : area_acres = 10)
                    (h5 : square_feet_per_acre = 43560) :
  (area_acres * square_feet_per_acre) / width_feet = 360 := 
by {
  sorry
}

end plot_length_l225_225213


namespace range_of_t_l225_225993

theorem range_of_t (x y a t : ℝ) 
  (h1 : x + 3 * y + a = 4) 
  (h2 : x - y - 3 * a = 0) 
  (h3 : -1 ≤ a ∧ a ≤ 1) 
  (h4 : t = x + y) : 
  1 ≤ t ∧ t ≤ 3 := 
sorry

end range_of_t_l225_225993


namespace find_a_l225_225305

def f (x : ℝ) : ℝ := |x - 1| - |x + 1|

theorem find_a (a : ℝ) (h : f (f a) = f 9 + 1) : a = -1/4 := 
by 
  sorry

end find_a_l225_225305


namespace dice_product_sum_impossible_l225_225032

theorem dice_product_sum_impossible (d1 d2 d3 d4 : ℕ) (h1 : 1 ≤ d1 ∧ d1 ≤ 6) (h2 : 1 ≤ d2 ∧ d2 ≤ 6) (h3 : 1 ≤ d3 ∧ d3 ≤ 6) (h4 : 1 ≤ d4 ∧ d4 ≤ 6) (hprod : d1 * d2 * d3 * d4 = 180) :
  (d1 + d2 + d3 + d4 ≠ 14) ∧ (d1 + d2 + d3 + d4 ≠ 17) :=
by
  sorry

end dice_product_sum_impossible_l225_225032


namespace factorize_expression_l225_225974

theorem factorize_expression (x : ℝ) : 
  (x^2 + 4)^2 - 16 * x^2 = (x + 2)^2 * (x - 2)^2 := 
by sorry

end factorize_expression_l225_225974


namespace scrooge_mcduck_max_box_l225_225902

-- Define Fibonacci numbers
def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

-- The problem statement: for a given positive integer k (number of coins initially),
-- the maximum box index n into which Scrooge McDuck can place a coin
-- is F_{k+2} - 1.
theorem scrooge_mcduck_max_box (k : ℕ) (h_pos : 0 < k) :
  ∃ n, n = fib (k + 2) - 1 :=
sorry

end scrooge_mcduck_max_box_l225_225902


namespace mike_practice_hours_l225_225349

def weekday_practice_hours_per_day : ℕ := 3
def days_per_weekday_practice : ℕ := 5
def saturday_practice_hours : ℕ := 5
def weeks_until_game : ℕ := 3

def total_weekday_practice_hours : ℕ := weekday_practice_hours_per_day * days_per_weekday_practice
def total_weekly_practice_hours : ℕ := total_weekday_practice_hours + saturday_practice_hours
def total_practice_hours : ℕ := total_weekly_practice_hours * weeks_until_game

theorem mike_practice_hours :
  total_practice_hours = 60 := by
  sorry

end mike_practice_hours_l225_225349


namespace total_distance_AD_l225_225810

theorem total_distance_AD :
  let d_AB := 100
  let d_BC := d_AB + 50
  let d_CD := 2 * d_BC
  d_AB + d_BC + d_CD = 550 := by
  sorry

end total_distance_AD_l225_225810


namespace porter_monthly_earnings_l225_225005

def daily_rate : ℕ := 8

def regular_days : ℕ := 5

def extra_day_rate : ℕ := daily_rate * 3 / 2  -- 50% increase on the daily rate

def weekly_earnings_with_overtime : ℕ := (daily_rate * regular_days) + extra_day_rate

def weeks_in_month : ℕ := 4

theorem porter_monthly_earnings : weekly_earnings_with_overtime * weeks_in_month = 208 :=
by
  sorry

end porter_monthly_earnings_l225_225005


namespace least_integer_sum_of_primes_l225_225033

-- Define what it means to be prime and greater than a number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def greater_than_ten (n : ℕ) : Prop := n > 10

-- Main theorem statement
theorem least_integer_sum_of_primes :
  ∃ n, (∀ p1 p2 p3 p4 : ℕ, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
                        greater_than_ten p1 ∧ greater_than_ten p2 ∧ greater_than_ten p3 ∧ greater_than_ten p4 ∧
                        p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
                        n = p1 + p2 + p3 + p4 → n ≥ 60) ∧
        n = 60 :=
  sorry

end least_integer_sum_of_primes_l225_225033


namespace value_of_m_making_365m_divisible_by_12_l225_225528

theorem value_of_m_making_365m_divisible_by_12
  (m : ℕ)
  (h1 : (3650 + m) % 3 = 0)
  (h2 : (50 + m) % 4 = 0) :
  m = 0 :=
sorry

end value_of_m_making_365m_divisible_by_12_l225_225528


namespace gcd_max_two_digits_l225_225588

theorem gcd_max_two_digits (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (h3 : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 100 :=
by
  -- Definitions
  sorry

end gcd_max_two_digits_l225_225588


namespace tan_product_l225_225273

theorem tan_product : 
  tan (Real.pi / 9) * tan (2 * Real.pi / 9) * tan (4 * Real.pi / 9) = 3 :=
by
  sorry

end tan_product_l225_225273


namespace cube_painting_probability_l225_225972

-- Define the conditions: a cube with six faces, each painted either green or yellow (independently, with probability 1/2)
structure Cube where
  faces : Fin 6 → Bool  -- Let's represent Bool with True for green, False for yellow

def is_valid_arrangement (c : Cube) : Prop :=
  ∃ (color : Bool), 
    (c.faces 0 = color ∧ c.faces 1 = color ∧ c.faces 2 = color ∧ c.faces 3 = color) ∧
    (∀ (i j : Fin 6), i = j ∨ ¬(c.faces i = color ∧ c.faces j = color))

def total_arrangements : ℕ := 2 ^ 6

def suitable_arrangements : ℕ := 20  -- As calculated previously: 2 + 12 + 6 = 20

-- We want to prove that the probability is 5/16
theorem cube_painting_probability :
  (suitable_arrangements : ℚ) / total_arrangements = 5 / 16 := 
by
  sorry

end cube_painting_probability_l225_225972


namespace quadratic_real_solutions_l225_225043

theorem quadratic_real_solutions (x y : ℝ) :
  (∃ z : ℝ, 16 * z^2 + 4 * x * y * z + (y^2 - 3) = 0) ↔ (x ≤ -2 ∨ x ≥ 2) :=
by
  sorry

end quadratic_real_solutions_l225_225043


namespace P_of_7_l225_225153

noncomputable def P (x : ℝ) : ℝ := 12 * (x - 1) * (x - 2) * (x - 3) * (x - 4)^2 * (x - 5)^2 * (x - 6)

theorem P_of_7 : P 7 = 51840 :=
by
  sorry

end P_of_7_l225_225153


namespace smallest_z_l225_225035

theorem smallest_z 
  (x y z : ℕ) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) 
  (h1 : x + y = z) 
  (h2 : x * y < z^2) 
  (ineq : (27^z) * (5^x) > (3^24) * (2^y)) :
  z = 10 :=
by
  sorry

end smallest_z_l225_225035


namespace sum_of_roots_eq_k_div_4_l225_225464

variables {k d y_1 y_2 : ℝ}

theorem sum_of_roots_eq_k_div_4 (h1 : y_1 ≠ y_2)
                                  (h2 : 4 * y_1^2 - k * y_1 = d)
                                  (h3 : 4 * y_2^2 - k * y_2 = d) :
  y_1 + y_2 = k / 4 :=
sorry

end sum_of_roots_eq_k_div_4_l225_225464


namespace students_passed_in_dixon_lecture_l225_225317

theorem students_passed_in_dixon_lecture :
  let ratio_collins := 18 / 30
  let students_dixon := 45
  ∃ y, ratio_collins = y / students_dixon ∧ y = 27 :=
by
  sorry

end students_passed_in_dixon_lecture_l225_225317


namespace sum_of_roots_unique_solution_l225_225968

open Real

def operation (a b : ℝ) : ℝ := a^2 + 2 * a * b - b^2

def f (x : ℝ) : ℝ := operation x 2

theorem sum_of_roots_unique_solution
  (x1 x2 x3 x4 : ℝ)
  (h1 : ∀ x, f x = log (abs (x + 2)) → x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4)
  (h2 : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) :
  x1 + x2 + x3 + x4 = -8 :=
sorry

end sum_of_roots_unique_solution_l225_225968


namespace find_function_solution_l225_225287

def satisfies_condition (f : ℝ → ℝ) :=
  ∀ (x y : ℝ), f (f (x * y)) = |x| * f y + 3 * f (x * y)

theorem find_function_solution (f : ℝ → ℝ) :
  satisfies_condition f → (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 4 * |x|) ∨ (∀ x : ℝ, f x = -4 * |x|) :=
by
  sorry

end find_function_solution_l225_225287


namespace triangle_area_l225_225820

theorem triangle_area : 
  let line_eq (x y : ℝ) := 3 * x + 2 * y = 12
  let x_intercept := (4 : ℝ)
  let y_intercept := (6 : ℝ)
  ∃ (x y : ℝ), line_eq x y ∧ x = x_intercept ∧ y = y_intercept ∧
  ∃ (area : ℝ), area = 1 / 2 * x * y ∧ area = 12 :=
by
  sorry

end triangle_area_l225_225820


namespace goods_train_length_is_420_l225_225054

/-- The man's train speed in km/h. -/
def mans_train_speed_kmph : ℝ := 64

/-- The goods train speed in km/h. -/
def goods_train_speed_kmph : ℝ := 20

/-- The time taken for the trains to pass each other in seconds. -/
def passing_time_s : ℝ := 18

/-- The relative speed of two trains traveling in opposite directions in m/s. -/
noncomputable def relative_speed_mps : ℝ := 
  (mans_train_speed_kmph + goods_train_speed_kmph) * 1000 / 3600

/-- The length of the goods train in meters. -/
noncomputable def goods_train_length_m : ℝ := relative_speed_mps * passing_time_s

/-- The theorem stating the length of the goods train is 420 meters. -/
theorem goods_train_length_is_420 :
  goods_train_length_m = 420 :=
sorry

end goods_train_length_is_420_l225_225054


namespace undefined_expression_values_l225_225857

theorem undefined_expression_values : 
    ∃ x : ℝ, x^2 - 9 = 0 ↔ (x = -3 ∨ x = 3) :=
by
  sorry

end undefined_expression_values_l225_225857


namespace friend_redistribution_l225_225856

-- Definitions of friends' earnings
def earnings := [18, 22, 26, 32, 47]

-- Definition of total earnings
def totalEarnings := earnings.sum

-- Definition of equal share
def equalShare := totalEarnings / earnings.length

-- The amount that the friend who earned 47 needs to redistribute
def redistributionAmount := 47 - equalShare

-- The goal to prove
theorem friend_redistribution:
  redistributionAmount = 18 := by
  sorry

end friend_redistribution_l225_225856


namespace gcd_digits_le_3_l225_225584

theorem gcd_digits_le_3 (a b : ℕ) (h_a : 10^6 ≤ a < 10^7) (h_b : 10^6 ≤ b < 10^7) (h_lcm : 10^11 ≤ Nat.lcm a b):
  Nat.gcd a b < 1000 := 
sorry

end gcd_digits_le_3_l225_225584


namespace tangent_product_eq_three_l225_225261

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end tangent_product_eq_three_l225_225261


namespace range_of_a_l225_225984

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

def no_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c < 0

theorem range_of_a (a : ℝ) :
  no_real_roots 1 (2 * a - 1) 1 ↔ -1 / 2 < a ∧ a < 3 / 2 := 
by sorry

end range_of_a_l225_225984


namespace find_base_l225_225554

noncomputable def log_base (a x : ℝ) := Real.log x / Real.log a

theorem find_base (a : ℝ) (h : 1 < a) :
  (log_base a (2 * a) - log_base a a = 1 / 2) → a = 4 :=
by
  -- skipping the proof
  sorry

end find_base_l225_225554


namespace remainder_14_div_5_l225_225198

theorem remainder_14_div_5 : 14 % 5 = 4 := by
  sorry

end remainder_14_div_5_l225_225198


namespace tan_product_identity_l225_225280

noncomputable def tan_pi_over_n (n : ℕ) : ℝ := Real.tan (Real.pi / n)

theorem tan_product_identity :
  tan_pi_over_n 9 * tan_pi_over_n 4.5 * tan_pi_over_n 2.25 = Real.sqrt 3 := 
sorry

end tan_product_identity_l225_225280


namespace gcd_digit_bound_l225_225577

theorem gcd_digit_bound
  (a b : ℕ)
  (h1 : 10^6 ≤ a)
  (h2 : a < 10^7)
  (h3 : 10^6 ≤ b)
  (h4 : b < 10^7)
  (h_lcm : 10^{11} ≤ Nat.lcm a b)
  (h_lcm2 : Nat.lcm a b < 10^{12}) :
  Nat.gcd a b < 10^3 :=
sorry

end gcd_digit_bound_l225_225577


namespace range_a_if_no_solution_l225_225154

def f (x : ℝ) : ℝ := abs (x - abs (2 * x - 4))

theorem range_a_if_no_solution (a : ℝ) :
  (∀ x : ℝ, f x > 0 → false) → a < 1 :=
by
  sorry

end range_a_if_no_solution_l225_225154


namespace scientific_notation_of_graphene_l225_225722

theorem scientific_notation_of_graphene :
  0.00000000034 = 3.4 * 10^(-10) :=
sorry

end scientific_notation_of_graphene_l225_225722


namespace simplify_expression_l225_225625

-- Define the problem and its conditions
theorem simplify_expression :
  (81 * 10^12) / (9 * 10^4) = 900000000 :=
by
  sorry  -- Proof placeholder

end simplify_expression_l225_225625


namespace find_c_l225_225125

-- Given that the function f(x) = 2^x + c passes through the point (2,5),
-- Prove that c = 1.
theorem find_c (c : ℝ) : (∃ (f : ℝ → ℝ), (∀ x, f x = 2^x + c) ∧ (f 2 = 5)) → c = 1 := by
  sorry

end find_c_l225_225125


namespace gcd_at_most_3_digits_l225_225572

/-- If the least common multiple of two 7-digit integers has 12 digits, 
  then their greatest common divisor has at most 3 digits. -/
theorem gcd_at_most_3_digits (a b : ℕ)
  (h1 : 10^6 ≤ a ∧ a < 10^7)
  (h2 : 10^6 ≤ b ∧ b < 10^7)
  (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b ≤ 999 := 
sorry

end gcd_at_most_3_digits_l225_225572


namespace number_of_comedies_rented_l225_225742

noncomputable def comedies_rented (r : ℕ) (a : ℕ) : ℕ := 3 * a

theorem number_of_comedies_rented (a : ℕ) (h : a = 5) : comedies_rented 3 a = 15 := by
  rw [h]
  exact rfl

end number_of_comedies_rented_l225_225742


namespace rate_per_kg_mangoes_is_55_l225_225670

def total_amount : ℕ := 1125
def rate_per_kg_grapes : ℕ := 70
def weight_grapes : ℕ := 9
def weight_mangoes : ℕ := 9

def cost_grapes := rate_per_kg_grapes * weight_grapes
def cost_mangoes := total_amount - cost_grapes

theorem rate_per_kg_mangoes_is_55 (rate_per_kg_mangoes : ℕ) (h : rate_per_kg_mangoes = cost_mangoes / weight_mangoes) : rate_per_kg_mangoes = 55 :=
by
  -- proof construction
  sorry

end rate_per_kg_mangoes_is_55_l225_225670


namespace max_nine_multiple_l225_225719

theorem max_nine_multiple {a b c n : ℕ} (h1 : Prime a) (h2 : Prime b) (h3 : Prime c) (h4 : 3 < a) (h5 : 3 < b) (h6 : 3 < c) (h7 : 2 * a + 5 * b = c) : 9 ∣ (a + b + c) :=
sorry

end max_nine_multiple_l225_225719


namespace evaluate_expression_l225_225427

theorem evaluate_expression (x : ℤ) (h : x + 1 = 4) : 
  (-3)^3 + (-3)^2 + (-3 * x) + 3 * x + 3^2 + 3^3 = 18 :=
by
  -- Since we know the condition x + 1 = 4
  have hx : x = 3 := by linarith
  -- Substitution x = 3 into the expression
  rw [hx]
  -- The expression after substitution and simplification
  sorry

end evaluate_expression_l225_225427


namespace tan_product_identity_l225_225277

noncomputable def tan_pi_over_n (n : ℕ) : ℝ := Real.tan (Real.pi / n)

theorem tan_product_identity :
  tan_pi_over_n 9 * tan_pi_over_n 4.5 * tan_pi_over_n 2.25 = Real.sqrt 3 := 
sorry

end tan_product_identity_l225_225277


namespace soccer_team_games_played_l225_225661

theorem soccer_team_games_played (t : ℝ) (h1 : 0.40 * t = 63.2) : t = 158 :=
sorry

end soccer_team_games_played_l225_225661


namespace negation_proof_l225_225365

-- Definitions based on conditions
def Line : Type := sorry  -- Define a type for lines (using sorry for now)
def Plane : Type := sorry  -- Define a type for planes (using sorry for now)

-- Condition definition
def is_perpendicular (l : Line) (α : Plane) : Prop := sorry  -- Define what it means for a plane to be perpendicular to a line (using sorry for now)

-- Given condition
axiom condition : ∀ (l : Line), ∃ (α : Plane), is_perpendicular l α

-- Statement to prove
theorem negation_proof : (∃ (l : Line), ∀ (α : Plane), ¬is_perpendicular l α) :=
sorry

end negation_proof_l225_225365


namespace james_nickels_l225_225459

theorem james_nickels (p n : ℕ) (h₁ : p + n = 50) (h₂ : p + 5 * n = 150) : n = 25 :=
by
  -- Skipping the proof since only the statement is required
  sorry

end james_nickels_l225_225459


namespace grocer_pounds_of_bananas_purchased_l225_225809

/-- 
Given:
1. The grocer purchased bananas at a rate of 3 pounds for $0.50.
2. The grocer sold the entire quantity at a rate of 4 pounds for $1.00.
3. The profit from selling the bananas was $11.00.

Prove that the number of pounds of bananas the grocer purchased is 132. 
-/
theorem grocer_pounds_of_bananas_purchased (P : ℕ) 
    (h1 : ∃ P, (3 * P / 0.5) - (4 * P / 1.0) = 11) : 
    P = 132 := 
sorry

end grocer_pounds_of_bananas_purchased_l225_225809


namespace div_powers_same_base_l225_225066

variable (x : ℝ)

theorem div_powers_same_base : x^8 / x^2 = x^6 :=
by
  sorry

end div_powers_same_base_l225_225066


namespace period_of_sin_sub_cos_l225_225197

open Real

theorem period_of_sin_sub_cos :
  ∃ T > 0, ∀ x, sin x - cos x = sin (x + T) - cos (x + T) ∧ T = 2 * π := sorry

end period_of_sin_sub_cos_l225_225197


namespace last_three_digits_of_7_pow_215_l225_225979

theorem last_three_digits_of_7_pow_215 :
  (7 ^ 215) % 1000 = 447 := by
  sorry

end last_three_digits_of_7_pow_215_l225_225979


namespace geometric_progression_common_ratio_l225_225597

theorem geometric_progression_common_ratio (a r : ℝ) 
(h_pos: a > 0)
(h_condition: ∀ n : ℕ, a * r^(n-1) = (a * r^n + a * r^(n+1))^2):
  r = 0.618 :=
sorry

end geometric_progression_common_ratio_l225_225597


namespace find_a4_l225_225452

variable {a : ℕ → ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) * a (n - 1) = a n * a n

def given_sequence_conditions (a : ℕ → ℝ) : Prop :=
  is_geometric_sequence a ∧ a 2 + a 6 = 34 ∧ a 3 * a 5 = 64

-- Statement
theorem find_a4 (a : ℕ → ℝ) (h : given_sequence_conditions a) : a 4 = 8 :=
sorry

end find_a4_l225_225452


namespace value_of_f_g6_minus_g_f6_l225_225299

def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
def g (x : ℝ) : ℝ := x + 4

theorem value_of_f_g6_minus_g_f6 : f (g 6) - g (f 6) = 48 :=
by
  sorry

end value_of_f_g6_minus_g_f6_l225_225299


namespace inequality_absolute_value_l225_225986

theorem inequality_absolute_value (a b : ℝ) (h1 : a < b) (h2 : b < 0) : |a| > -b :=
sorry

end inequality_absolute_value_l225_225986


namespace number_of_integer_length_chords_through_point_l225_225163

theorem number_of_integer_length_chords_through_point 
  (r : ℝ) (d : ℝ) (P_is_5_units_from_center : d = 5) (circle_has_radius_13 : r = 13) :
  ∃ n : ℕ, n = 3 := by
  sorry

end number_of_integer_length_chords_through_point_l225_225163


namespace max_consecutive_irreducible_l225_225506

-- Define what it means for a five-digit number to be irreducible
def is_irreducible (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ ¬∃ x y : ℕ, 100 ≤ x ∧ x < 1000 ∧ 100 ≤ y ∧ y < 1000 ∧ x * y = n

-- Prove the maximum number of consecutive irreducible five-digit numbers is 99
theorem max_consecutive_irreducible : ∃ m : ℕ, m = 99 ∧ 
  (∀ n : ℕ, (n ≤ 99901) → (∀ k : ℕ, (n ≤ k ∧ k < n + m) → is_irreducible k)) ∧
  (∀ x y : ℕ, x > 99 → ∀ n : ℕ, (n ≤ 99899) → (∀ k : ℕ, (n ≤ k ∧ k < n + x) → is_irreducible k) → x = 99) :=
by
  sorry

end max_consecutive_irreducible_l225_225506


namespace dan_baseball_cards_total_l225_225842

-- Define the initial conditions
def initial_baseball_cards : Nat := 97
def torn_baseball_cards : Nat := 8
def sam_bought_cards : Nat := 15
def alex_bought_fraction : Nat := 4
def gift_cards : Nat := 6

-- Define the number of cards    
def non_torn_baseball_cards : Nat := initial_baseball_cards - torn_baseball_cards
def remaining_after_sam : Nat := non_torn_baseball_cards - sam_bought_cards
def remaining_after_alex : Nat := remaining_after_sam - remaining_after_sam / alex_bought_fraction
def final_baseball_cards : Nat := remaining_after_alex + gift_cards

-- The theorem to prove 
theorem dan_baseball_cards_total : final_baseball_cards = 62 := by
  sorry

end dan_baseball_cards_total_l225_225842


namespace count_two_digit_perfect_squares_divisible_by_4_l225_225109

theorem count_two_digit_perfect_squares_divisible_by_4 : 
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = k^2 ∧ k^2 % 4 = 0}.to_finset.card = 3 :=
by
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l225_225109


namespace binom_10_2_eq_45_l225_225678

-- Definitions used in the conditions
def binom (n k : ℕ) := n.choose k

-- The statement that needs to be proven
theorem binom_10_2_eq_45 : binom 10 2 = 45 :=
by
  sorry

end binom_10_2_eq_45_l225_225678


namespace min_cans_needed_l225_225970

theorem min_cans_needed (oz_per_can : ℕ) (total_oz_needed : ℕ) (H1 : oz_per_can = 15) (H2 : total_oz_needed = 150) :
  ∃ n : ℕ, 15 * n ≥ 150 ∧ ∀ m : ℕ, 15 * m ≥ 150 → n ≤ m :=
by
  sorry

end min_cans_needed_l225_225970


namespace largest_negative_root_l225_225977

theorem largest_negative_root : 
  ∃ x : ℝ, (∃ k : ℤ, x = -1/2 + 2 * ↑k) ∧ 
  ∀ y : ℝ, (∃ k : ℤ, (y = -1/2 + 2 * ↑k ∨ y = 1/6 + 2 * ↑k ∨ y = 5/6 + 2 * ↑k)) → y < 0 → y ≤ x :=
sorry

end largest_negative_root_l225_225977


namespace max_value_of_function_l225_225288

open Real

theorem max_value_of_function :
  ∀ x ∈ Icc (0 : ℝ) (π / 2), (λ x, x + 2 * cos x) x ≤ (π / 6 + sqrt 3) :=
sorry

end max_value_of_function_l225_225288


namespace prime_square_minus_one_divisible_by_twelve_l225_225612

theorem prime_square_minus_one_divisible_by_twelve (p : ℕ) (hp : Nat.Prime p) (hp_gt : p > 3) : 12 ∣ (p^2 - 1) :=
by
  sorry

end prime_square_minus_one_divisible_by_twelve_l225_225612


namespace fraction_ordering_l225_225965

theorem fraction_ordering :
  let a := (6 : ℚ) / 22
  let b := (8 : ℚ) / 32
  let c := (10 : ℚ) / 29
  a < b ∧ b < c :=
by
  sorry

end fraction_ordering_l225_225965


namespace major_axis_length_of_ellipse_l225_225718

-- Definition of the conditions
def line (x y : ℝ) : Prop := 2 * x - y + 4 = 0
def ellipse (x y : ℝ) (m : ℝ) : Prop := (x^2) / m + (y^2) / 2 = 1
def is_focus (x y m : ℝ) : Prop := line x y ∧ ellipse x y m

theorem major_axis_length_of_ellipse (m : ℝ) (h₀ : m > 0) :
  (∃ (x y : ℝ), is_focus x y m) → 2 * Real.sqrt 6 = 2 * Real.sqrt m :=
sorry

end major_axis_length_of_ellipse_l225_225718


namespace Donny_change_l225_225418

theorem Donny_change (tank_capacity : ℕ) (initial_fuel : ℕ) (money_available : ℕ) (fuel_cost_per_liter : ℕ) 
  (h1 : tank_capacity = 150) 
  (h2 : initial_fuel = 38) 
  (h3 : money_available = 350) 
  (h4 : fuel_cost_per_liter = 3) : 
  money_available - (tank_capacity - initial_fuel) * fuel_cost_per_liter = 14 := 
by 
  sorry

end Donny_change_l225_225418


namespace gcd_max_digits_l225_225594

theorem gcd_max_digits {a b : ℕ} (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) 
  (h3 : ∃ k, 10^11 ≤ k ∧ k < 10^{12} ∧ k = lcm a b) : 
  (gcd a b) < 10^3 :=
sorry

end gcd_max_digits_l225_225594


namespace sum_of_digits_0_to_2012_l225_225329

-- Define the sum of digits function
def sum_of_digits (n : Nat) : Nat := 
  n.digits 10 |>.sum

-- Define the problem to calculate the sum of all digits from 0 to 2012
def sum_digits_up_to (n : Nat) : Nat := 
  (List.range (n + 1)).map sum_of_digits |>.sum

-- Lean theorem statement to prove the sum of digits from 0 to 2012 is 28077
theorem sum_of_digits_0_to_2012 : sum_digits_up_to 2012 = 28077 := 
  sorry

end sum_of_digits_0_to_2012_l225_225329


namespace combined_avg_score_l225_225458

noncomputable def classA_student_count := 45
noncomputable def classB_student_count := 55
noncomputable def classA_avg_score := 110
noncomputable def classB_avg_score := 90

theorem combined_avg_score (nA nB : ℕ) (avgA avgB : ℕ) 
  (h1 : nA = classA_student_count) 
  (h2 : nB = classB_student_count) 
  (h3 : avgA = classA_avg_score) 
  (h4 : avgB = classB_avg_score) : 
  (nA * avgA + nB * avgB) / (nA + nB) = 99 := 
by 
  rw [h1, h2, h3, h4]
  -- Substitute the values to get:
  -- (45 * 110 + 55 * 90) / (45 + 55) 
  -- = (4950 + 4950) / 100 
  -- = 9900 / 100 
  -- = 99
  sorry

end combined_avg_score_l225_225458


namespace divisible_by_3_l225_225538

theorem divisible_by_3 :
  ∃ n : ℕ, (5 + 2 + n + 4 + 8) % 3 = 0 ∧ n = 2 := 
by
  sorry

end divisible_by_3_l225_225538


namespace Bettina_card_value_l225_225328

theorem Bettina_card_value (x : ℝ) (h₀ : 0 < x) (h₁ : x < π / 2) (h₂ : Real.tan x ≠ 1) (h₃ : Real.sin x ≠ Real.cos x) :
  ∀ {a b c : ℝ}, (a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x) →
                  (b = Real.sin x ∨ b = Real.cos x ∨ b = Real.tan x) →
                  (c = Real.sin x ∨ c = Real.cos x ∨ c = Real.tan x) →
                  a ≠ b → b ≠ c → a ≠ c →
                  (b = Real.cos x) → b = Real.sqrt 3 / 2 := 
  sorry

end Bettina_card_value_l225_225328


namespace ax5_by5_eq_neg1065_l225_225905

theorem ax5_by5_eq_neg1065 (a b x y : ℝ) 
  (h1 : a*x + b*y = 5) 
  (h2 : a*x^2 + b*y^2 = 9) 
  (h3 : a*x^3 + b*y^3 = 20) 
  (h4 : a*x^4 + b*y^4 = 48) 
  (h5 : x + y = -15) 
  (h6 : x^2 + y^2 = 55) : 
  a * x^5 + b * y^5 = -1065 := 
sorry

end ax5_by5_eq_neg1065_l225_225905


namespace find_a_l225_225542

open Real

-- Define the circle equation
def circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 5

-- Define the line equation passing through P(2,2)
def line_through_P (m b x y : ℝ) : Prop := y = m * x + b ∧ (2, 2) = (x, y)

-- Define the line equation ax - y + 1 = 0
def perpendicular_line (a x y : ℝ) : Prop := a * x - y + 1 = 0

theorem find_a : ∃ a : ℝ, ∀ x y m b : ℝ,
    circle x y ∧ line_through_P m b x y ∧
    (line_through_P m b x y → perpendicular_line a x y) → a = 2 :=
by
  intros
  sorry

end find_a_l225_225542


namespace average_ABC_is_three_l225_225641
-- Import the entirety of the Mathlib library

-- Define the required conditions and the theorem to be proved
theorem average_ABC_is_three (A B C : ℝ) 
    (h1 : 2012 * C - 4024 * A = 8048) 
    (h2 : 2012 * B + 6036 * A = 10010) : 
    (A + B + C) / 3 = 3 := 
by
  sorry

end average_ABC_is_three_l225_225641


namespace p_necessary_not_sufficient_q_l225_225702

-- Define the conditions p and q
def p (a : ℝ) : Prop := a < 1
def q (a : ℝ) : Prop := 0 < a ∧ a < 1

-- State the necessary but not sufficient condition theorem
theorem p_necessary_not_sufficient_q (a : ℝ) : p a → q a → p a ∧ ¬∀ (a : ℝ), p a → q a :=
by
  sorry

end p_necessary_not_sufficient_q_l225_225702


namespace remainder_5310_mod8_l225_225366

theorem remainder_5310_mod8 : (53 ^ 10) % 8 = 1 := 
by 
  sorry

end remainder_5310_mod8_l225_225366


namespace sum_geometric_sequence_l225_225082

theorem sum_geometric_sequence (a r : ℝ) (n : ℕ) (h_a : a = 1/3) (h_r : r = 1/3) (h_n : n = 8) :
  let S_n := a * (1 - r^n) / (1 - r) in S_n = 3280/6561 :=
by
  sorry

end sum_geometric_sequence_l225_225082


namespace tubs_of_ice_cream_guests_ate_l225_225372

def pans : Nat := 2
def pieces_per_pan : Nat := 16
def eaten_percentage : Float := 0.75
def scoops_per_tub : Nat := 8
def guests_not_eating : Nat := 4
def scoops_per_guest : Nat := 2

theorem tubs_of_ice_cream_guests_ate :
  let total_pieces := pans * pieces_per_pan
  let eaten_pieces := pieces_per_pan + Nat.floor (eaten_percentage * pieces_per_pan.toReal)
  let guests_ala_mode := eaten_pieces - guests_not_eating
  let total_scoops_eaten := guests_ala_mode * scoops_per_guest
  let tubs_ice_cream := total_scoops_eaten / scoops_per_tub
  tubs_ice_cream = 6 := by
  sorry

end tubs_of_ice_cream_guests_ate_l225_225372


namespace poly_value_at_two_l225_225029

def f (x : ℝ) : ℝ := x^5 + 2 * x^4 + 3 * x^3 + 4 * x^2 + 5 * x + 6

theorem poly_value_at_two : f 2 = 216 :=
by
  unfold f
  norm_num
  sorry

end poly_value_at_two_l225_225029


namespace find_N_l225_225519

theorem find_N
  (N : ℕ)
  (h : (4 / 10 : ℝ) * (16 / (16 + N : ℝ)) + (6 / 10 : ℝ) * (N / (16 + N : ℝ)) = 0.58) :
  N = 144 :=
sorry

end find_N_l225_225519


namespace binomial_10_2_equals_45_l225_225684

open Nat

theorem binomial_10_2_equals_45 : Nat.choose 10 2 = 45 := 
by
  sorry

end binomial_10_2_equals_45_l225_225684


namespace pony_average_speed_l225_225941

theorem pony_average_speed
  (time_head_start : ℝ)
  (time_catch : ℝ)
  (horse_speed : ℝ)
  (distance_covered_by_horse : ℝ)
  (distance_covered_by_pony : ℝ)
  (pony's_head_start : ℝ)
  : (time_head_start = 3) → (time_catch = 4) → (horse_speed = 35) → 
    (distance_covered_by_horse = horse_speed * time_catch) → 
    (pony's_head_start = time_head_start * v) → 
    (distance_covered_by_pony = pony's_head_start + (v * time_catch)) → 
    (distance_covered_by_horse = distance_covered_by_pony) → v = 20 :=
  by 
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end pony_average_speed_l225_225941


namespace pumps_to_fill_tires_l225_225414

-- Definitions based on conditions
def AirPerTire := 500
def PumpVolume := 50
def FlatTires := 2
def PercentFullTire1 := 0.40
def PercentFullTire2 := 0.70

-- The formal statement/proof problem
theorem pumps_to_fill_tires : (FlatTires * AirPerTire + AirPerTire * (1 - PercentFullTire1) + AirPerTire * (1 - PercentFullTire2)) / PumpVolume = 29 := 
by sorry

end pumps_to_fill_tires_l225_225414


namespace p_plus_q_l225_225362

-- Define the problem conditions
def p (x : ℝ) : ℝ := 4 * (x - 2)
def q (x : ℝ) : ℝ := (x + 2) * (x - 2)

-- Main theorem to prove the answer
theorem p_plus_q (x : ℝ) : p x + q x = x^2 + 4 * x - 12 := 
by
  sorry

end p_plus_q_l225_225362


namespace add_base8_numbers_l225_225401

def fromBase8 (n : Nat) : Nat :=
  Nat.digits 8 n |> Nat.ofDigits 8

theorem add_base8_numbers : 
  fromBase8 356 + fromBase8 672 + fromBase8 145 = fromBase8 1477 :=
by
  sorry

end add_base8_numbers_l225_225401


namespace tan_product_l225_225259

noncomputable def tan_pi_nine := Real.tan (Real.pi / 9)
noncomputable def tan_two_pi_nine := Real.tan (2 * Real.pi / 9)
noncomputable def tan_four_pi_nine := Real.tan (4 * Real.pi / 9)

-- Key polynomial condition
def poly := λ t : ℝ, t^8 - 36 * t^6 + 126 * t^4 - 84 * t^2 + 9

-- Condition: the polynomial has roots tan(π/9), tan(2π/9), tan(4π/9)
axiom poly_roots: 
  (poly tan_pi_nine = 0 ∧
   poly tan_two_pi_nine = 0 ∧
   poly tan_four_pi_nine = 0)

-- The proof statement
theorem tan_product : 
  tan_pi_nine * 
  tan_two_pi_nine * 
  tan_four_pi_nine = 3 :=
sorry

end tan_product_l225_225259


namespace tom_books_total_l225_225027

theorem tom_books_total :
  (2 + 6 + 10 + 14 + 18) = 50 :=
by {
  -- Proof steps would go here.
  sorry
}

end tom_books_total_l225_225027


namespace ceil_floor_sum_l225_225071

theorem ceil_floor_sum :
  (Int.ceil (7 / 3 : ℚ)) + (Int.floor (-7 / 3 : ℚ)) = 0 := 
sorry

end ceil_floor_sum_l225_225071


namespace max_questions_wrong_to_succeed_l225_225521

theorem max_questions_wrong_to_succeed :
  ∀ (total_questions : ℕ) (passing_percentage : ℚ),
  total_questions = 50 →
  passing_percentage = 0.75 →
  ∃ (max_wrong : ℕ), max_wrong = 12 ∧
    (total_questions - max_wrong) ≥ passing_percentage * total_questions := by
  intro total_questions passing_percentage h1 h2
  use 12
  constructor
  . rfl
  . sorry  -- Proof omitted

end max_questions_wrong_to_succeed_l225_225521


namespace line_equation_M_l225_225735

theorem line_equation_M (x y : ℝ) : 
  (∃ c1 m1 : ℝ, m1 = 2 / 3 ∧ c1 = 4 ∧ 
  (∃ m2 c2 : ℝ, m2 = 2 * m1 ∧ c2 = (1 / 2) * c1 ∧ y = m2 * x + c2)) → 
  y = (4 / 3) * x + 2 := 
sorry

end line_equation_M_l225_225735


namespace sin_value_given_cos_condition_l225_225863

theorem sin_value_given_cos_condition (theta : ℝ) (h : Real.cos (5 * Real.pi / 12 - theta) = 1 / 3) :
  Real.sin (Real.pi / 12 + theta) = 1 / 3 :=
sorry

end sin_value_given_cos_condition_l225_225863


namespace difference_of_cubes_not_div_by_twice_diff_l225_225007

theorem difference_of_cubes_not_div_by_twice_diff (a b : ℤ) (h_a : a % 2 = 1) (h_b : b % 2 = 1) (h_neq : a ≠ b) :
  ¬ (2 * (a - b)) ∣ ((a^3) - (b^3)) := 
sorry

end difference_of_cubes_not_div_by_twice_diff_l225_225007


namespace Haleigh_needs_leggings_l225_225444

/-- Haleigh's pet animals -/
def dogs : Nat := 4
def cats : Nat := 3
def legs_per_dog : Nat := 4
def legs_per_cat : Nat := 4
def leggings_per_pair : Nat := 2

/-- The proof statement -/
theorem Haleigh_needs_leggings : (dogs * legs_per_dog + cats * legs_per_cat) / leggings_per_pair = 14 := by
  sorry

end Haleigh_needs_leggings_l225_225444


namespace integral_is_Gaussian_l225_225465

noncomputable def isGaussian (X : Ω → ℝ) : Prop :=
sorry

variable {Ω : Type*} {X : Ω → ℝ}

variables (measurable_X : ae_measurable X (measure_space.measure_space Ω)) 
          (finite_integral : ∫⁻ x, |X x| ∂(measure_space.measure_space Ω) < ∞)
          (is_gaussian_system : ∀ ⦃sD : set ℝ⦄ (hsD : is_open sD), measurable_set (λ ω, X ω ∈ sD))

theorem integral_is_Gaussian : isGaussian (λ ω, ∫ t in 0..1, X ω t) :=
sorry

end integral_is_Gaussian_l225_225465


namespace find_c_l225_225630

theorem find_c (a b c : ℝ) (k₁ k₂ : ℝ) 
  (h₁ : a * b = k₁) 
  (h₂ : b * c = k₂) 
  (h₃ : 40 * 5 = k₁) 
  (h₄ : 7 * 10 = k₂) 
  (h₅ : a = 16) : 
  c = 5.6 :=
  sorry

end find_c_l225_225630


namespace option_A_cannot_be_true_l225_225093

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (r : ℝ) -- common ratio for the geometric sequence
variable (n : ℕ) -- number of terms

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

def sum_of_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  S 0 = a 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

theorem option_A_cannot_be_true
  (h_geom : is_geometric_sequence a r)
  (h_sum : sum_of_geometric_sequence a S) :
  a 2016 * (S 2016 - S 2015) ≠ 0 :=
sorry

end option_A_cannot_be_true_l225_225093


namespace evaluate_g_at_neg2_l225_225072

def g (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem evaluate_g_at_neg2 : g (-2) = 11 := by
  sorry

end evaluate_g_at_neg2_l225_225072


namespace value_of_x_l225_225121

variable (w x y : ℝ)

theorem value_of_x 
  (h_avg : (w + x) / 2 = 0.5)
  (h_eq : (7 / w) + (7 / x) = 7 / y)
  (h_prod : w * x = y) :
  x = 0.5 :=
sorry

end value_of_x_l225_225121


namespace total_books_l225_225008

-- Define the conditions
def books_per_shelf : ℕ := 9
def mystery_shelves : ℕ := 6
def picture_shelves : ℕ := 2

-- The proof problem statement
theorem total_books : 
  (mystery_shelves * books_per_shelf) + 
  (picture_shelves * books_per_shelf) = 72 := 
sorry

end total_books_l225_225008


namespace a_range_l225_225992

variables {x a : ℝ}

def p (x : ℝ) : Prop := (4 * x - 3) ^ 2 ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem a_range (h : ∀ x, ¬p x → ¬q x a ∧ (∃ x, q x a ∧ ¬p x)) :
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end a_range_l225_225992


namespace factor_expression_correct_l225_225690

variable (y : ℝ)

def expression := 4 * y * (y + 2) + 6 * (y + 2)

theorem factor_expression_correct : expression y = (y + 2) * (2 * (2 * y + 3)) :=
by
  sorry

end factor_expression_correct_l225_225690


namespace sheepdog_rounded_up_percentage_l225_225227

/-- Carla's sheepdog rounded up a certain percentage of her sheep. We know the remaining 10% of the sheep  wandered off into the hills, which is 9 sheep out in the wilderness. There are 81 sheep in the pen. We need to prove that the sheepdog rounded up 90% of the total number of sheep. -/
theorem sheepdog_rounded_up_percentage (total_sheep pen_sheep wilderness_sheep : ℕ) 
  (h1 : wilderness_sheep = 9) 
  (h2 : pen_sheep = 81) 
  (h3 : wilderness_sheep = total_sheep / 10) :
  (pen_sheep * 100 / total_sheep) = 90 :=
sorry

end sheepdog_rounded_up_percentage_l225_225227


namespace complete_square_proof_l225_225917

def complete_square (x : ℝ) : Prop :=
  x^2 - 2 * x - 8 = 0 -> (x - 1)^2 = 9

theorem complete_square_proof (x : ℝ) :
  complete_square x :=
sorry

end complete_square_proof_l225_225917


namespace parity_of_expression_l225_225888

theorem parity_of_expression (a b c : ℕ) (h_apos : 0 < a) (h_aodd : a % 2 = 1) (h_beven : b % 2 = 0) :
  (3^a + (b+1)^2 * c) % 2 = if c % 2 = 0 then 1 else 0 :=
sorry

end parity_of_expression_l225_225888


namespace rope_length_l225_225649

theorem rope_length (x S : ℝ) (H1 : x + 7 * S = 140)
(H2 : x - S = 20) : x = 35 := by
sorry

end rope_length_l225_225649


namespace elena_snow_removal_l225_225973

theorem elena_snow_removal :
  ∀ (length width depth : ℝ) (compaction_factor : ℝ), 
  length = 30 ∧ width = 3 ∧ depth = 0.75 ∧ compaction_factor = 0.90 → 
  (length * width * depth * compaction_factor = 60.75) :=
by
  intros length width depth compaction_factor h
  obtain ⟨length_eq, width_eq, depth_eq, compaction_factor_eq⟩ := h
  -- Proof steps go here
  sorry

end elena_snow_removal_l225_225973


namespace tan_product_identity_l225_225279

noncomputable def tan_pi_over_n (n : ℕ) : ℝ := Real.tan (Real.pi / n)

theorem tan_product_identity :
  tan_pi_over_n 9 * tan_pi_over_n 4.5 * tan_pi_over_n 2.25 = Real.sqrt 3 := 
sorry

end tan_product_identity_l225_225279


namespace tree_sidewalk_space_l225_225131

theorem tree_sidewalk_space
  (num_trees : ℕ)
  (distance_between_trees : ℝ)
  (total_road_length : ℝ)
  (total_gaps : ℝ)
  (space_each_tree : ℝ)
  (H1 : num_trees = 11)
  (H2 : distance_between_trees = 14)
  (H3 : total_road_length = 151)
  (H4 : total_gaps = (num_trees - 1) * distance_between_trees)
  (H5 : space_each_tree = (total_road_length - total_gaps) / num_trees)
  : space_each_tree = 1 := 
by
  sorry

end tree_sidewalk_space_l225_225131


namespace number_of_terms_in_arithmetic_sequence_l225_225119

theorem number_of_terms_in_arithmetic_sequence : 
  ∀ (a d l : ℕ), a = 20 → d = 5 → l = 150 → 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 27 :=
by
  intros a d l ha hd hl
  use 27
  rw [ha, hd, hl]
  sorry

end number_of_terms_in_arithmetic_sequence_l225_225119


namespace leggings_needed_l225_225442

theorem leggings_needed (dogs : ℕ) (cats : ℕ) (dogs_legs : ℕ) (cats_legs : ℕ) (pair_of_leggings : ℕ) 
                        (hd : dogs = 4) (hc : cats = 3) (hl1 : dogs_legs = 4) (hl2 : cats_legs = 4) (lp : pair_of_leggings = 2)
                        : (dogs * dogs_legs + cats * cats_legs) / pair_of_leggings = 14 :=
by
  sorry

end leggings_needed_l225_225442


namespace quadratic_transformation_l225_225022

theorem quadratic_transformation
    (a b c : ℝ)
    (h : ℝ)
    (cond : ∀ x, a * x^2 + b * x + c = 4 * (x - 5)^2 + 16) :
    (∀ x, 5 * a * x^2 + 5 * b * x + 5 * c = 20 * (x - h)^2 + 80) → h = 5 :=
by
  sorry

end quadratic_transformation_l225_225022


namespace tan_product_eq_three_l225_225232

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end tan_product_eq_three_l225_225232


namespace plums_total_correct_l225_225159

-- Define the number of plums picked by Melanie, Dan, and Sally
def plums_melanie : ℕ := 4
def plums_dan : ℕ := 9
def plums_sally : ℕ := 3

-- Define the total number of plums picked
def total_plums : ℕ := plums_melanie + plums_dan + plums_sally

-- Theorem stating the total number of plums picked
theorem plums_total_correct : total_plums = 16 := by
  sorry

end plums_total_correct_l225_225159


namespace triangle_area_l225_225822

theorem triangle_area : 
  let line_eq (x y : ℝ) := 3 * x + 2 * y = 12
  let x_intercept := (4 : ℝ)
  let y_intercept := (6 : ℝ)
  ∃ (x y : ℝ), line_eq x y ∧ x = x_intercept ∧ y = y_intercept ∧
  ∃ (area : ℝ), area = 1 / 2 * x * y ∧ area = 12 :=
by
  sorry

end triangle_area_l225_225822


namespace average_age_increase_l225_225479

theorem average_age_increase 
    (num_students : ℕ) (avg_age_students : ℕ) (age_staff : ℕ)
    (H1: num_students = 32)
    (H2: avg_age_students = 16)
    (H3: age_staff = 49) : 
    ((num_students * avg_age_students + age_staff) / (num_students + 1) - avg_age_students = 1) :=
by
  sorry

end average_age_increase_l225_225479


namespace rectangular_field_area_l225_225526

noncomputable def length : ℝ := 1.2
noncomputable def width : ℝ := (3/4) * length

theorem rectangular_field_area : (length * width = 1.08) :=
by 
  -- The proof steps would go here
  sorry

end rectangular_field_area_l225_225526


namespace gcd_at_most_3_digits_l225_225573

/-- If the least common multiple of two 7-digit integers has 12 digits, 
  then their greatest common divisor has at most 3 digits. -/
theorem gcd_at_most_3_digits (a b : ℕ)
  (h1 : 10^6 ≤ a ∧ a < 10^7)
  (h2 : 10^6 ≤ b ∧ b < 10^7)
  (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b ≤ 999 := 
sorry

end gcd_at_most_3_digits_l225_225573


namespace volume_of_apple_juice_l225_225806

noncomputable def apple_juice_volume : ℝ :=
  let pi := Real.pi
  let total_ratio := 2 + 5
  let apple_ratio := 2 / total_ratio.toRat
  let radius := 2
  let height := 3
  let total_volume := pi * radius^2 * height
  apple_ratio * total_volume

theorem volume_of_apple_juice : abs (apple_juice_volume - 10.74) < 0.01 := by
  sorry

end volume_of_apple_juice_l225_225806


namespace sum_geometric_sequence_first_eight_terms_l225_225084

theorem sum_geometric_sequence_first_eight_terms :
  let a_0 := (1 : ℚ) / 3
  let r := (1 : ℚ) / 3
  let n := 8
  let S_n := a_0 * (1 - r^n) / (1 - r)
  S_n = 6560 / 19683 := 
by
  sorry

end sum_geometric_sequence_first_eight_terms_l225_225084


namespace centered_hexagonal_seq_l225_225387

def is_centered_hexagonal (a : ℕ) : Prop :=
  ∃ n : ℕ, a = 3 * n^2 - 3 * n + 1

def are_sequences (a b c d : ℕ) : Prop :=
  (b = 2 * a - 1) ∧ (d = c^2) ∧ (a + b = c + d)

theorem centered_hexagonal_seq (a : ℕ) :
  (∃ b c d, are_sequences a b c d) ↔ is_centered_hexagonal a :=
sorry

end centered_hexagonal_seq_l225_225387


namespace two_solutions_exist_l225_225876

theorem two_solutions_exist 
  (a b c : ℝ) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_equation : (1 / a) + (1 / b) + (1 / c) = (1 / (a + b + c))) : 
  ∃ (a' b' c' : ℝ), 
    ((a' = 1/3 ∧ b' = 1/3 ∧ c' = 1/3) ∨ (a' = -1/3 ∧ b' = -1/3 ∧ c' = -1/3)) := 
sorry

end two_solutions_exist_l225_225876


namespace min_expression_value_l225_225001

theorem min_expression_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (z : ℝ) (h3 : x^2 + y^2 = z) :
  (x + 1/y) * (x + 1/y - 2020) + (y + 1/x) * (y + 1/x - 2020) = -2040200 :=
  sorry

end min_expression_value_l225_225001


namespace find_G_8_l225_225903

noncomputable def G : Polynomial ℝ := sorry 

variable (x : ℝ)

theorem find_G_8 :
  G.eval 4 = 8 ∧ 
  (∀ x, (G.eval (2*x)) / (G.eval (x+2)) = 4 - (16 * x) / (x^2 + 2 * x + 2)) →
  G.eval 8 = 40 := 
sorry

end find_G_8_l225_225903


namespace tan_identity_l225_225246

theorem tan_identity :
  tan (π / 9) * tan (2 * π / 9) * tan (4 * π / 9) = 3 :=
sorry

end tan_identity_l225_225246


namespace contrapositive_example_contrapositive_proof_l225_225778

theorem contrapositive_example (x : ℝ) (h : x > 1) : x^2 > 1 := 
sorry

theorem contrapositive_proof (x : ℝ) (h : x^2 ≤ 1) : x ≤ 1 :=
sorry

end contrapositive_example_contrapositive_proof_l225_225778


namespace height_of_parallelogram_l225_225798

theorem height_of_parallelogram (A B h : ℝ) (hA : A = 72) (hB : B = 12) (h_area : A = B * h) : h = 6 := by
  sorry

end height_of_parallelogram_l225_225798


namespace three_pow_zero_l225_225224

theorem three_pow_zero : 3^0 = 1 :=
by sorry

end three_pow_zero_l225_225224


namespace tan_product_l225_225239

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end tan_product_l225_225239


namespace correct_result_value_at_neg_one_l225_225438

theorem correct_result (x : ℝ) (A : ℝ := 3 * x^2 - x + 1) (incorrect : ℝ := 2 * x^2 - 3 * x - 2) :
  (A - (incorrect - A)) = 4 * x^2 + x + 4 :=
by sorry

theorem value_at_neg_one (x : ℝ := -1) (A : ℝ := 3 * x^2 - x + 1) (incorrect : ℝ := 2 * x^2 - 3 * x - 2) :
  (4 * x^2 + x + 4) = 7 :=
by sorry

end correct_result_value_at_neg_one_l225_225438


namespace deal_or_no_deal_min_eliminations_l225_225453

theorem deal_or_no_deal_min_eliminations (n_boxes : ℕ) (n_high_value : ℕ) 
    (initial_count : n_boxes = 26)
    (high_value_count : n_high_value = 9) :
  ∃ (min_eliminations : ℕ), min_eliminations = 8 ∧
    ((n_boxes - min_eliminations - 1) / 2) ≥ n_high_value :=
sorry

end deal_or_no_deal_min_eliminations_l225_225453


namespace cube_paint_probability_l225_225971

/-- 
Each face of a cube is painted either green or yellow, each with probability 1/2. 
The color of each face is determined independently. Prove that the probability 
that the painted cube can be placed on a horizontal surface so that the four 
vertical faces are all the same color is 5/16. 
-/
theorem cube_paint_probability : 
  let cube_faces := [true, false] -- Assuming true is green, false is yellow
  let arrangements := { arr | arr ∈ cube_faces^6 ∧ independent (λ i, arr i) }
  let suitable_arrangements := { arr ∈ arrangements | 
    can_be_placed_on_horizontal_surface_with_same_color_vertical_faces arr }
  ∃ pr, pr = (|suitable_arrangements| : ℚ) / (|arrangements| : ℚ) ∧ pr = 5 / 16
:= sorry

end cube_paint_probability_l225_225971


namespace largest_common_value_l225_225845

theorem largest_common_value :
  ∃ (a : ℕ), (∃ (n m : ℕ), a = 4 + 5 * n ∧ a = 5 + 10 * m) ∧ a < 1000 ∧ a = 994 :=
by {
  sorry
}

end largest_common_value_l225_225845


namespace rocky_run_miles_l225_225743

theorem rocky_run_miles : 
  let day1 := 4 in
  let day2 := 2 * day1 in
  let day3 := 3 * day2 in
  day1 + day2 + day3 = 36 :=
by
  sorry

end rocky_run_miles_l225_225743


namespace correct_system_of_equations_l225_225601

theorem correct_system_of_equations (x y : ℕ) :
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) ↔ 
  (y = 7 * x + 7) ∧ (y = 9 * (x - 1)) :=
by
  sorry

end correct_system_of_equations_l225_225601


namespace find_math_books_l225_225042

theorem find_math_books 
  (M H : ℕ)
  (h1 : M + H = 80)
  (h2 : 4 * M + 5 * H = 390) : 
  M = 10 := 
by 
  sorry

end find_math_books_l225_225042


namespace intersection_points_C1_C2_l225_225872

theorem intersection_points_C1_C2 :
  (∀ t : ℝ, ∃ (ρ θ : ℝ), 
    (ρ^2 - 10 * ρ * Real.cos θ - 8 * ρ * Real.sin θ + 41 = 0) ∧ 
    (ρ = 2 * Real.cos θ) → 
    ((ρ = 2 ∧ θ = 0) ∨ (ρ = Real.sqrt 2 ∧ θ = Real.pi / 4))) :=
sorry

end intersection_points_C1_C2_l225_225872


namespace color_lattice_points_l225_225962

variables {α : Type*} [Fintype α]

def lattice_points (n : ℕ) : Finset (ℕ × ℕ) :=
  (Finset.range n).product (Finset.range n).filter (fun x => x.1 ≠ x.2)

theorem color_lattice_points : 
  ∃ (coloring : (ℕ × ℕ) → Fin 10), 
    (∀ a b c : ℕ, a ≠ b → b ≠ c → 
      (coloring (a, b) ≠ coloring (b, c))) :=
begin
  -- proof here
  sorry
end

end color_lattice_points_l225_225962


namespace largest_proper_divisor_condition_l225_225515

def is_proper_divisor (n k : ℕ) : Prop :=
  k > 1 ∧ k < n ∧ n % k = 0

theorem largest_proper_divisor_condition (n p : ℕ) (hp : is_proper_divisor n p) (hl : ∀ k, is_proper_divisor n k → k ≤ n / p):
  n = 12 ∨ n = 33 :=
by
  -- Placeholder for proof
  sorry

end largest_proper_divisor_condition_l225_225515


namespace true_propositions_count_l225_225873

-- Original Proposition
def P (x y : ℝ) : Prop := x^2 + y^2 = 0 → x = 0 ∧ y = 0

-- Converse Proposition
def Q (x y : ℝ) : Prop := x = 0 ∧ y = 0 → x^2 + y^2 = 0

-- Contrapositive Proposition
def contrapositive_Q_P (x y : ℝ) : Prop := (x ≠ 0 ∨ y ≠ 0) → (x^2 + y^2 ≠ 0)

-- Inverse Proposition
def inverse_P (x y : ℝ) : Prop := (x^2 + y^2 ≠ 0) → (x ≠ 0 ∨ y ≠ 0)

-- Problem Statement
theorem true_propositions_count : ∀ (x y : ℝ),
  P x y ∧ Q x y ∧ contrapositive_Q_P x y ∧ inverse_P x y → 3 = 3 :=
by
  intros x y h
  sorry

end true_propositions_count_l225_225873


namespace vinces_bus_ride_length_l225_225788

theorem vinces_bus_ride_length (zachary_ride : ℝ) (vince_extra : ℝ) (vince_ride : ℝ) :
  zachary_ride = 0.5 →
  vince_extra = 0.13 →
  vince_ride = zachary_ride + vince_extra →
  vince_ride = 0.63 :=
by
  intros hz hv he
  -- proof steps here
  sorry

end vinces_bus_ride_length_l225_225788


namespace alpha_in_second_quadrant_l225_225122

theorem alpha_in_second_quadrant (α : Real) 
  (h1 : Real.sin (2 * α) < 0) 
  (h2 : Real.cos α - Real.sin α < 0) : 
  π / 2 < α ∧ α < π :=
sorry

end alpha_in_second_quadrant_l225_225122


namespace tan_product_equals_three_l225_225244

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end tan_product_equals_three_l225_225244


namespace players_at_least_two_sciences_l225_225222

-- Define the conditions of the problem
def total_players : Nat := 30
def players_biology : Nat := 15
def players_chemistry : Nat := 10
def players_physics : Nat := 5
def players_all_three : Nat := 3

-- Define the main theorem we want to prove
theorem players_at_least_two_sciences :
  (players_biology + players_chemistry + players_physics 
    - players_all_three - total_players) = 9 :=
sorry

end players_at_least_two_sciences_l225_225222


namespace students_more_than_pets_l225_225062

-- Definitions for the conditions
def number_of_classrooms := 5
def students_per_classroom := 22
def rabbits_per_classroom := 3
def hamsters_per_classroom := 2

-- Total number of students in all classrooms
def total_students := number_of_classrooms * students_per_classroom

-- Total number of pets in all classrooms
def total_pets := number_of_classrooms * (rabbits_per_classroom + hamsters_per_classroom)

-- The theorem to prove
theorem students_more_than_pets : 
  total_students - total_pets = 85 :=
by
  sorry

end students_more_than_pets_l225_225062


namespace F_2021_F_integer_F_divisibility_l225_225983

/- Part 1 -/
def F (n : ℕ) : ℕ := 
  let a := n / 1000
  let b := (n % 1000) / 100
  let c := (n % 100) / 10
  let d := n % 10
  let n' := 1000 * c + 100 * d + 10 * a + b
  (n + n') / 101

theorem F_2021 : F 2021 = 41 :=
  sorry

/- Part 2 -/
theorem F_integer (a b c d : ℕ) (ha : 1 ≤ a) (hb : a ≤ 9) (hc : 0 ≤ b) (hd : b ≤ 9)
(hc' : 0 ≤ c) (hd' : c ≤ 9) (hc'' : 0 ≤ d) (hd'' : d ≤ 9) :
  let n := 1000 * a + 100 * b + 10 * c + d
  let n' := 1000 * c + 100 * d + 10 * a + b
  F n = (101 * (10 * a + b + 10 * c + d)) / 101 :=
  sorry

/- Part 3 -/
theorem F_divisibility (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 5) (hb : 5 ≤ b ∧ b ≤ 9) :
  let s := 3800 + 10 * a + b
  let t := 1000 * b + 100 * a + 13
  (3 * F t - F s) % 8 = 0 ↔ s = 3816 ∨ s = 3847 ∨ s = 3829 :=
  sorry

end F_2021_F_integer_F_divisibility_l225_225983


namespace amelia_remaining_money_l225_225221

variable {m b n : ℚ}

theorem amelia_remaining_money (h : (1 / 4) * m = (1 / 2) * n * b) : 
  m - n * b = (1 / 2) * m :=
by
  sorry

end amelia_remaining_money_l225_225221


namespace problem_solution_l225_225123

noncomputable def find_a3_and_sum (a0 a1 a2 a3 a4 a5 : ℝ) : Prop :=
  (∀ x : ℝ, x^5 = a0 + a1 * (x + 2) + a2 * (x + 2)^2 + a3 * (x + 2)^3 + a4 * (x + 2)^4 + a5 * (x + 2)^5) →
  (a3 = 40 ∧ a0 + a1 + a2 + a4 + a5 = -41)

theorem problem_solution {a0 a1 a2 a3 a4 a5 : ℝ} :
  find_a3_and_sum a0 a1 a2 a3 a4 a5 :=
by
  intros h
  sorry

end problem_solution_l225_225123


namespace gcd_at_most_3_digits_l225_225571

/-- If the least common multiple of two 7-digit integers has 12 digits, 
  then their greatest common divisor has at most 3 digits. -/
theorem gcd_at_most_3_digits (a b : ℕ)
  (h1 : 10^6 ≤ a ∧ a < 10^7)
  (h2 : 10^6 ≤ b ∧ b < 10^7)
  (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b ≤ 999 := 
sorry

end gcd_at_most_3_digits_l225_225571


namespace number_of_tables_l225_225860

theorem number_of_tables (last_year_distance : ℕ) (factor : ℕ) 
  (distance_between_table_1_and_3 : ℕ) (number_of_tables : ℕ) :
  (last_year_distance = 300) ∧ 
  (factor = 4) ∧ 
  (distance_between_table_1_and_3 = 400) ∧
  (number_of_tables = ((factor * last_year_distance) / (distance_between_table_1_and_3 / 2)) + 1) 
  → number_of_tables = 7 :=
by
  intros
  sorry

end number_of_tables_l225_225860


namespace complement_of_A_in_U_l225_225344

def U : Set ℕ := {1,3,5,7,9}
def A : Set ℕ := {1,9}
def complement_U_A : Set ℕ := {3,5,7}

theorem complement_of_A_in_U : (U \ A) = complement_U_A := by
  sorry

end complement_of_A_in_U_l225_225344


namespace candies_markus_l225_225157

theorem candies_markus (m k s : ℕ) (h_initial_m : m = 9) (h_initial_k : k = 5) (h_total_s : s = 10) :
  (m + s) / 2 = 12 := by
  sorry

end candies_markus_l225_225157


namespace harmful_bacteria_time_l225_225494

noncomputable def number_of_bacteria (x : ℝ) : ℝ :=
  4000 * 2^x

theorem harmful_bacteria_time :
  ∃ (x : ℝ), number_of_bacteria x > 90000 ∧ x = 4.5 :=
by
  sorry

end harmful_bacteria_time_l225_225494


namespace remaining_money_is_correct_l225_225011

def initial_amount : ℕ := 53
def cost_toy_car : ℕ := 11
def number_toy_cars : ℕ := 2
def cost_scarf : ℕ := 10
def cost_beanie : ℕ := 14
def remaining_money : ℕ := 
  initial_amount - (cost_toy_car * number_toy_cars) - cost_scarf - cost_beanie

theorem remaining_money_is_correct : remaining_money = 7 := by
  sorry

end remaining_money_is_correct_l225_225011


namespace tangent_chord_equation_l225_225920

theorem tangent_chord_equation (x1 y1 x2 y2 : ℝ) :
  (x1^2 + y1^2 = 1) →
  (x2^2 + y2^2 = 1) →
  (2*x1 + 2*y1 + 1 = 0) →
  (2*x2 + 2*y2 + 1 = 0) →
  ∀ (x y : ℝ), 2*x + 2*y + 1 = 0 :=
by
  intros hx1 hy1 hx2 hy2 x y
  exact sorry

end tangent_chord_equation_l225_225920


namespace perfect_squares_two_digit_divisible_by_4_count_l225_225108

-- Define two-digit
def is_two_digit (n : ℤ) : Prop :=
  10 ≤ n ∧ n < 100

-- Define perfect square
def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k^2

-- Define divisible by 4
def divisible_by_4 (n : ℤ) : Prop :=
  n % 4 = 0

-- Define the main statement: number of two-digit perfect squares that are divisible by 4 is 3
theorem perfect_squares_two_digit_divisible_by_4_count :
  { n : ℤ | is_two_digit n ∧ is_perfect_square n ∧ divisible_by_4 n }.size = 3 :=
by sorry

end perfect_squares_two_digit_divisible_by_4_count_l225_225108


namespace triangle_property_l225_225596

theorem triangle_property
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a > b)
  (h2 : a = 5)
  (h3 : c = 6)
  (h4 : Real.sin B = 3 / 5) :
  (b = Real.sqrt 13 ∧ Real.sin A = 3 * Real.sqrt 13 / 13) →
  Real.sin (2 * A + π / 4) = 7 * Real.sqrt 2 / 26 :=
sorry

end triangle_property_l225_225596


namespace induction_proof_l225_225622

def f (n : ℕ) : ℕ := (List.range (2 * n - 1)).sum + n

theorem induction_proof (n : ℕ) (h : n > 0) : f (n + 1) - f n = 8 * n := by
  sorry

end induction_proof_l225_225622


namespace complement_union_eq_l225_225557

open Set

-- Define the universe and sets P and Q
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 3, 5}
def Q : Set ℕ := {1, 2, 4}

-- State the theorem
theorem complement_union_eq :
  ((U \ P) ∪ Q) = {1, 2, 4, 6} := by
  sorry

end complement_union_eq_l225_225557


namespace area_of_triangle_bounded_by_coordinate_axes_and_line_l225_225818

def area_of_triangle (b h : ℕ) : ℕ := (b * h) / 2

theorem area_of_triangle_bounded_by_coordinate_axes_and_line :
  area_of_triangle 4 6 = 12 :=
by
  sorry

end area_of_triangle_bounded_by_coordinate_axes_and_line_l225_225818


namespace car_r_speed_l225_225799

theorem car_r_speed (v : ℝ) (h : 150 / v - 2 = 150 / (v + 10)) : v = 25 :=
sorry

end car_r_speed_l225_225799


namespace average_speed_of_train_l225_225398

theorem average_speed_of_train (x : ℝ) (h1 : x > 0): 
  (3 * x) / ((x / 40) + (2 * x / 20)) = 24 :=
by
  sorry

end average_speed_of_train_l225_225398


namespace cylinder_twice_volume_l225_225541

theorem cylinder_twice_volume :
  let r := 8
  let h1 := 10
  let h2 := 20
  let V := (pi * r^2 * h1)
  let V_desired := 2 * V
  V_desired = pi * r^2 * h2 :=
by
  let r := 8
  let h1 := 10
  let h2 := 20
  let V := (pi * r^2 * h1)
  let V_desired := 2 * V
  show V_desired = pi * r^2 * h2
  sorry

end cylinder_twice_volume_l225_225541


namespace product_of_N_l225_225409

theorem product_of_N (M L : ℝ) (N : ℝ) 
  (h1 : M = L + N) 
  (h2 : ∀ M4 L4 : ℝ, M4 = M - 7 → L4 = L + 5 → |M4 - L4| = 4) :
  N = 16 ∨ N = 8 ∧ (16 * 8 = 128) := 
by 
  sorry

end product_of_N_l225_225409


namespace substring_012_appears_148_times_l225_225424

noncomputable def count_substring_012_in_base_3_concat (n : ℕ) : ℕ :=
  -- The function that counts the "012" substrings in the concatenated base-3 representations
  sorry

theorem substring_012_appears_148_times :
  count_substring_012_in_base_3_concat 728 = 148 :=
  sorry

end substring_012_appears_148_times_l225_225424


namespace landscape_length_l225_225931

-- Define the conditions from the problem
def breadth (b : ℝ) := b > 0
def length_of_landscape (l b : ℝ) := l = 8 * b
def area_of_playground (pg_area : ℝ) := pg_area = 1200
def playground_fraction (A b : ℝ) := A = 8 * b^2
def fraction_of_landscape (pg_area A : ℝ) := pg_area = (1/6) * A

-- Main theorem statement
theorem landscape_length (b l A pg_area : ℝ) 
  (H_b : breadth b) 
  (H_length : length_of_landscape l b)
  (H_pg_area : area_of_playground pg_area)
  (H_pg_fraction : playground_fraction A b)
  (H_pg_landscape_fraction : fraction_of_landscape pg_area A) :
  l = 240 :=
by
  sorry

end landscape_length_l225_225931


namespace proof_problem_l225_225486

def star (a b : ℕ) : ℕ := a - a / b

theorem proof_problem : star 18 6 + 2 * 6 = 27 := 
by
  admit  -- proof goes here

end proof_problem_l225_225486


namespace area_of_triangle_bounded_by_coordinate_axes_and_line_l225_225817

def area_of_triangle (b h : ℕ) : ℕ := (b * h) / 2

theorem area_of_triangle_bounded_by_coordinate_axes_and_line :
  area_of_triangle 4 6 = 12 :=
by
  sorry

end area_of_triangle_bounded_by_coordinate_axes_and_line_l225_225817


namespace inequality_subtraction_l225_225120

theorem inequality_subtraction (a b : ℝ) (h : a < b) : a - 5 < b - 5 := 
by {
  sorry
}

end inequality_subtraction_l225_225120


namespace area_of_triangle_bounded_by_line_and_axes_l225_225828

theorem area_of_triangle_bounded_by_line_and_axes (x y : ℝ) (hx : 3 * x + 2 * y = 12) :
  ∃ (area : ℝ), area = 12 := by
sorry

end area_of_triangle_bounded_by_line_and_axes_l225_225828


namespace sum_of_reciprocals_l225_225368

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 24) : 
  (1 / x + 1 / y = 1 / 2) :=
by 
  sorry

end sum_of_reciprocals_l225_225368


namespace fraction_inequality_l225_225567

variable (a b c : ℝ)

theorem fraction_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : c > a) (h5 : a > b) :
  (a / (c - a)) > (b / (c - b)) := 
sorry

end fraction_inequality_l225_225567


namespace total_money_l225_225400

-- Define the variables A, B, and C as real numbers.
variables (A B C : ℝ)

-- Define the conditions as hypotheses.
def conditions : Prop :=
  A + C = 300 ∧ B + C = 150 ∧ C = 50

-- State the theorem to prove the total amount of money A, B, and C have.
theorem total_money (h : conditions A B C) : A + B + C = 400 :=
by {
  -- This proof is currently omitted.
  sorry
}

end total_money_l225_225400


namespace tan_product_equals_three_l225_225242

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end tan_product_equals_three_l225_225242


namespace units_digit_7_pow_5_l225_225496

theorem units_digit_7_pow_5 : (7 ^ 5) % 10 = 7 := 
by
  sorry

end units_digit_7_pow_5_l225_225496


namespace slices_all_three_toppings_l225_225935

def slices_with_all_toppings (total_slices pepperoni_slices mushroom_slices olive_slices : ℕ) : ℕ := 
  (12 : ℕ)

theorem slices_all_three_toppings
  (total_slices : ℕ)
  (pepperoni_slices : ℕ)
  (mushroom_slices : ℕ)
  (olive_slices : ℕ)
  (h : total_slices = 24)
  (h1 : pepperoni_slices = 12)
  (h2 : mushroom_slices = 14)
  (h3 : olive_slices = 16)
  (hc : total_slices ≥ 0)
  (hc1 : pepperoni_slices ≥ 0)
  (hc2 : mushroom_slices ≥ 0)
  (hc3 : olive_slices ≥ 0) :
  slices_with_all_toppings total_slices pepperoni_slices mushroom_slices olive_slices = 2 :=
  sorry

end slices_all_three_toppings_l225_225935


namespace negation_proof_l225_225708

theorem negation_proof : 
  (¬(∀ x : ℝ, x < 2^x) ↔ ∃ x : ℝ, x ≥ 2^x) :=
by
  sorry

end negation_proof_l225_225708


namespace find_m_values_l225_225710

noncomputable def find_m (a b c m : ℝ) : Prop :=
  (a + b = 4) ∧
  (ab = m) ∧
  (b + c = 8) ∧
  (bc = 5m)

theorem find_m_values (a b c m : ℝ) (h : find_m a b c m) : m = 0 ∨ m = 3 :=
by
  sorry

end find_m_values_l225_225710


namespace average_of_six_numbers_l225_225480

theorem average_of_six_numbers (A : ℝ) (x y z w u v : ℝ)
  (h1 : (x + y + z + w + u + v) / 6 = A)
  (h2 : (x + y) / 2 = 1.1)
  (h3 : (z + w) / 2 = 1.4)
  (h4 : (u + v) / 2 = 5) :
  A = 2.5 :=
by
  sorry

end average_of_six_numbers_l225_225480


namespace digit_arrangements_l225_225898

section
variable (digits : Finset ℕ := {6, 0, 4, 0, 2})

theorem digit_arrangements : 
  (Finset.filter (λ n : List ℕ, n.head ≠ 0) 
      (Finset.image List.ofFn 
        (Finset.univ : Finset (Fin₅ → ℕ))))
      .card = 96 := by
  sorry
end

end digit_arrangements_l225_225898


namespace find_a_l225_225870

-- Given conditions as definitions.
def f (a x : ℝ) := a * x^3
def tangent_line (a : ℝ) (x : ℝ) : ℝ := 3 * x + a - 3

-- Problem statement in Lean 4.
theorem find_a (a : ℝ) (h_tangent : ∀ x : ℝ, f a 1 = 1 ∧ f a 1 = tangent_line a 1) : a = 1 := 
by sorry

end find_a_l225_225870


namespace ratio_c_a_l225_225566

theorem ratio_c_a (a b c : ℚ) (h1 : a * b = 3) (h2 : b * c = 8 / 5) : c / a = 8 / 15 := 
by 
  sorry

end ratio_c_a_l225_225566


namespace diameter_of_circumscribed_circle_l225_225124

noncomputable def circumscribed_circle_diameter (a : ℝ) (A : ℝ) : ℝ :=
  a / Real.sin A

theorem diameter_of_circumscribed_circle :
  circumscribed_circle_diameter 15 (Real.pi / 4) = 15 * Real.sqrt 2 :=
by
  sorry

end diameter_of_circumscribed_circle_l225_225124
