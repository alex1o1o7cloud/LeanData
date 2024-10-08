import Mathlib

namespace operation_result_l17_17460

def operation (a b : Int) : Int :=
  (a + b) * (a - b)

theorem operation_result :
  operation 4 (operation 2 (-1)) = 7 :=
by
  sorry

end operation_result_l17_17460


namespace fabian_total_cost_l17_17556

def mouse_cost : ℕ := 20

def keyboard_cost : ℕ := 2 * mouse_cost

def headphones_cost : ℕ := mouse_cost + 15

def usb_hub_cost : ℕ := 36 - mouse_cost

def total_cost : ℕ := mouse_cost + keyboard_cost + headphones_cost + usb_hub_cost

theorem fabian_total_cost : total_cost = 111 := 
by 
  unfold total_cost mouse_cost keyboard_cost headphones_cost usb_hub_cost
  sorry

end fabian_total_cost_l17_17556


namespace population_net_increase_l17_17964

theorem population_net_increase
  (birth_rate : ℕ) (death_rate : ℕ) (T : ℕ)
  (h1 : birth_rate = 7) (h2 : death_rate = 3) (h3 : T = 86400) :
  (birth_rate - death_rate) * (T / 2) = 172800 :=
by
  sorry

end population_net_increase_l17_17964


namespace correct_statement_D_l17_17338

theorem correct_statement_D : (- 3 / 5 : ℚ) < (- 4 / 7 : ℚ) :=
  by
  -- The proof step is omitted as per the instruction
  sorry

end correct_statement_D_l17_17338


namespace boys_in_school_l17_17735

theorem boys_in_school (x : ℕ) (boys girls : ℕ) (h1 : boys = 5 * x) 
  (h2 : girls = 13 * x) (h3 : girls - boys = 128) : boys = 80 :=
by
  sorry

end boys_in_school_l17_17735


namespace range_of_m_l17_17158

def A : Set ℝ := {x | x^2 - x - 12 < 0}
def B (m : ℝ) : Set ℝ := {x | abs (x - 3) ≤ m}
def p (x : ℝ) : Prop := x ∈ A
def q (x : ℝ) (m : ℝ) : Prop := x ∈ B m

theorem range_of_m (m : ℝ) (hm : m > 0):
  (∀ x, p x → q x m) ↔ (6 ≤ m) := by
  sorry

end range_of_m_l17_17158


namespace pq_condition_l17_17607

theorem pq_condition (p q : ℝ) (h1 : p * q = 16) (h2 : p + q = 10) : (p - q)^2 = 36 :=
by
  sorry

end pq_condition_l17_17607


namespace crabapple_recipients_sequences_l17_17686

-- Define the number of students in Mrs. Crabapple's class
def num_students : ℕ := 12

-- Define the number of class meetings per week
def num_meetings : ℕ := 5

-- Define the total number of different sequences
def total_sequences : ℕ := num_students ^ num_meetings

-- The target theorem to prove
theorem crabapple_recipients_sequences :
  total_sequences = 248832 := by
  sorry

end crabapple_recipients_sequences_l17_17686


namespace trip_to_museum_l17_17986

theorem trip_to_museum (x y z w : ℕ) 
  (h2 : y = 2 * x) 
  (h3 : z = 2 * x - 6) 
  (h4 : w = x + 9) 
  (htotal : x + y + z + w = 75) : 
  x = 12 := 
by 
  sorry

end trip_to_museum_l17_17986


namespace convert_to_rectangular_form_l17_17409

noncomputable def θ : ℝ := 15 * Real.pi / 2

noncomputable def EulerFormula (θ : ℝ) : ℂ := Complex.exp (Complex.I * θ)

theorem convert_to_rectangular_form : EulerFormula θ = Complex.I := by
  sorry

end convert_to_rectangular_form_l17_17409


namespace square_inscribed_in_right_triangle_side_length_l17_17936

theorem square_inscribed_in_right_triangle_side_length
  (A B C X Y Z W : ℝ × ℝ)
  (AB BC AC : ℝ)
  (square_side : ℝ)
  (h : 0 < square_side) :
  -- Define the lengths of sides of the triangle.
  AB = 3 ∧ BC = 4 ∧ AC = 5 ∧

  -- Define the square inscribed in the triangle
  (W.1 - A.1)^2 + (W.2 - A.2)^2 = square_side^2 ∧
  (X.1 - W.1)^2 + (X.2 - W.2)^2 = square_side^2 ∧
  (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = square_side^2 ∧
  (Z.1 - W.1)^2 + (Z.2 - W.2)^2 = square_side^2 ∧
  (Z.1 - C.1)^2 + (Z.2 - C.2)^2 = square_side^2 ∧

  -- Points where square meets triangle sides
  X.1 = A.1 ∧ Z.1 = C.1 ∧ Y.1 = X.1 ∧ W.1 = Z.1 ∧ Z.2 = Y.2 ∧

  -- Right triangle condition
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = AB^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = BC^2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = AC^2 ∧
  
  -- Right angle at vertex B
  (B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0
  →
  -- Prove the side length of the inscribed square
  square_side = 60 / 37 :=
sorry

end square_inscribed_in_right_triangle_side_length_l17_17936


namespace sum_first_six_terms_l17_17728

noncomputable def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_six_terms :
  geometric_series_sum (1/4) (1/4) 6 = 4095 / 12288 :=
by 
  sorry

end sum_first_six_terms_l17_17728


namespace polynomial_is_positive_for_all_x_l17_17844

noncomputable def P (x : ℝ) : ℝ := x^12 - x^9 + x^4 - x + 1

theorem polynomial_is_positive_for_all_x (x : ℝ) : P x > 0 := 
by
  dsimp [P]
  sorry -- Proof is omitted.

end polynomial_is_positive_for_all_x_l17_17844


namespace jorge_land_fraction_clay_rich_soil_l17_17623

theorem jorge_land_fraction_clay_rich_soil 
  (total_acres : ℕ) 
  (yield_good_soil_per_acre : ℕ) 
  (yield_clay_soil_factor : ℕ)
  (total_yield : ℕ) 
  (fraction_clay_rich_soil : ℚ) :
  total_acres = 60 →
  yield_good_soil_per_acre = 400 →
  yield_clay_soil_factor = 2 →
  total_yield = 20000 →
  fraction_clay_rich_soil = 1/3 :=
by
  intro h_total_acres h_yield_good_soil_per_acre h_yield_clay_soil_factor h_total_yield
  -- math proof will be here
  sorry

end jorge_land_fraction_clay_rich_soil_l17_17623


namespace solve_equation_l17_17804

theorem solve_equation :
  ∃ x : Real, (x = 2 ∨ x = (-(1:Real) - Real.sqrt 17) / 2) ∧ (x^2 - |x - 1| - 3 = 0) :=
by
  sorry

end solve_equation_l17_17804


namespace lex_reads_in_12_days_l17_17467

theorem lex_reads_in_12_days
  (total_pages : ℕ)
  (pages_per_day : ℕ)
  (h1 : total_pages = 240)
  (h2 : pages_per_day = 20) :
  total_pages / pages_per_day = 12 :=
by
  sorry

end lex_reads_in_12_days_l17_17467


namespace non_swimmers_play_soccer_percentage_l17_17272

theorem non_swimmers_play_soccer_percentage (N : ℕ) (hN_pos : 0 < N)
 (h1 : (0.7 * N : ℝ) = x)
 (h2 : (0.5 * N : ℝ) = y)
 (h3 : (0.6 * x : ℝ) = z)
 : (0.56 * y = 0.28 * N) := 
 sorry

end non_swimmers_play_soccer_percentage_l17_17272


namespace darryl_books_l17_17078

variable (l m d : ℕ)

theorem darryl_books (h1 : l + m + d = 97) (h2 : l = m - 3) (h3 : m = 2 * d) : d = 20 := 
by
  sorry

end darryl_books_l17_17078


namespace algebraic_expression_value_l17_17213

-- Definitions for the problem conditions
def x := -1
def y := 1 / 2
def expr := 2 * (x^2 - 5 * x * y) - 3 * (x^2 - 6 * x * y)

-- The problem statement to be proved
theorem algebraic_expression_value : expr = 3 :=
by
  sorry

end algebraic_expression_value_l17_17213


namespace total_fencing_cost_l17_17240

-- Definitions of the given conditions
def length : ℝ := 57
def breadth : ℝ := length - 14
def cost_per_meter : ℝ := 26.50

-- Definition of the total cost calculation
def total_cost : ℝ := 2 * (length + breadth) * cost_per_meter

-- Statement of the theorem to be proved
theorem total_fencing_cost :
  total_cost = 5300 := by
  -- Proof is omitted
  sorry

end total_fencing_cost_l17_17240


namespace black_area_after_transformations_l17_17567

theorem black_area_after_transformations :
  let initial_fraction : ℝ := 1
  let transformation_factor : ℝ := 3 / 4
  let number_of_transformations : ℕ := 5
  let final_fraction : ℝ := transformation_factor ^ number_of_transformations
  final_fraction = 243 / 1024 :=
by
  -- Proof omitted
  sorry

end black_area_after_transformations_l17_17567


namespace sum_of_three_integers_l17_17268

def three_positive_integers (x y z : ℕ) : Prop :=
  x + y = 2003 ∧ y - z = 1000

theorem sum_of_three_integers (x y z : ℕ) (h1 : x + y = 2003) (h2 : y - z = 1000) (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) : 
  x + y + z = 2004 := 
by 
  sorry

end sum_of_three_integers_l17_17268


namespace asian_games_discount_equation_l17_17062

variable (a : ℝ)

theorem asian_games_discount_equation :
  168 * (1 - a / 100)^2 = 128 :=
sorry

end asian_games_discount_equation_l17_17062


namespace symmetric_codes_count_l17_17096

def isSymmetric (grid : List (List Bool)) : Prop :=
  -- condition for symmetry: rotational and reflectional symmetry
  sorry

def isValidCode (grid : List (List Bool)) : Prop :=
  -- condition for valid scanning code with at least one black and one white
  sorry

noncomputable def numberOfSymmetricCodes : Nat :=
  -- function to count the number of symmetric valid codes
  sorry

theorem symmetric_codes_count :
  numberOfSymmetricCodes = 62 := 
  sorry

end symmetric_codes_count_l17_17096


namespace marbles_exchange_l17_17694

-- Define the initial number of marbles for Drew and Marcus
variables {D M x : ℕ}

-- Conditions
axiom Drew_initial (D M : ℕ) : D = M + 24
axiom Drew_after_give (D x : ℕ) : D - x = 25
axiom Marcus_after_receive (M x : ℕ) : M + x = 25

-- The goal is to prove: x = 12
theorem marbles_exchange : ∀ {D M x : ℕ}, D = M + 24 ∧ D - x = 25 ∧ M + x = 25 → x = 12 :=
by 
    sorry

end marbles_exchange_l17_17694


namespace petya_vasya_cubic_roots_diff_2014_l17_17989

theorem petya_vasya_cubic_roots_diff_2014 :
  ∀ (p q r : ℚ), ∃ (x1 x2 x3 : ℚ), x1 ≠ 0 ∧ (x1 - x2 = 2014 ∨ x1 - x3 = 2014 ∨ x2 - x3 = 2014) :=
sorry

end petya_vasya_cubic_roots_diff_2014_l17_17989


namespace sqrt_sum_eq_pow_l17_17488

/-- 
For the value \( k = 3/2 \), the expression \( \sqrt{2016} + \sqrt{56} \) equals \( 14^k \)
-/
theorem sqrt_sum_eq_pow (k : ℝ) (h : k = 3 / 2) : 
  (Real.sqrt 2016 + Real.sqrt 56) = 14 ^ k := 
by 
  sorry

end sqrt_sum_eq_pow_l17_17488


namespace abs_sum_inequality_for_all_x_l17_17146

theorem abs_sum_inequality_for_all_x (m : ℝ) :
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ m) ↔ (m ≤ 3) :=
by
  sorry

end abs_sum_inequality_for_all_x_l17_17146


namespace calculate_expression_l17_17927

theorem calculate_expression :
  2 * (-1 / 4) - |1 - Real.sqrt 3| + (-2023)^0 = 3 / 2 - Real.sqrt 3 :=
by
  sorry

end calculate_expression_l17_17927


namespace sufficient_p_wages_l17_17052

variable (S P Q : ℕ)

theorem sufficient_p_wages (h1 : S = 40 * Q) (h2 : S = 15 * (P + Q))  :
  ∃ D : ℕ, S = D * P ∧ D = 24 := 
by
  use 24
  sorry

end sufficient_p_wages_l17_17052


namespace multiple_of_3804_l17_17553

theorem multiple_of_3804 (n : ℕ) (hn : 0 < n) : 
  ∃ k : ℕ, (n^3 - n) * (5^(8*n+4) + 3^(4*n+2)) = k * 3804 :=
by
  sorry

end multiple_of_3804_l17_17553


namespace frac_abs_div_a_plus_one_l17_17332

theorem frac_abs_div_a_plus_one (a : ℝ) (h : a ≠ 0) : abs a / a + 1 = 0 ∨ abs a / a + 1 = 2 :=
by sorry

end frac_abs_div_a_plus_one_l17_17332


namespace min_value_expression_l17_17892

noncomputable def f (t : ℝ) : ℝ :=
  (1 / (t + 1)) + (2 * t / (2 * t + 1))

theorem min_value_expression (x y : ℝ) (h : x * y > 0) :
  ∃ t, (x / y = t) ∧ t > 0 ∧ f t = 4 - 2 * Real.sqrt 2 := 
  sorry

end min_value_expression_l17_17892


namespace length_of_DE_l17_17169

theorem length_of_DE 
  (area_ABC : ℝ) 
  (area_trapezoid : ℝ) 
  (altitude_ABC : ℝ) 
  (h1 : area_ABC = 144) 
  (h2 : area_trapezoid = 96)
  (h3 : altitude_ABC = 24) :
  ∃ (DE_length : ℝ), DE_length = 2 * Real.sqrt 3 := 
sorry

end length_of_DE_l17_17169


namespace pages_left_to_be_read_l17_17299

def total_pages : ℕ := 381
def pages_read : ℕ := 149
def pages_per_day : ℕ := 20
def days_in_week : ℕ := 7

theorem pages_left_to_be_read :
  total_pages - pages_read - (pages_per_day * days_in_week) = 92 := by
  sorry

end pages_left_to_be_read_l17_17299


namespace ab_value_l17_17547

theorem ab_value (a b : ℚ) 
  (h1 : (a + b) ^ 2 + |b + 5| = b + 5) 
  (h2 : 2 * a - b + 1 = 0) : 
  a * b = -1 / 9 :=
by
  sorry

end ab_value_l17_17547


namespace sum_of_all_possible_N_l17_17164

theorem sum_of_all_possible_N
  (a b c : ℕ)
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : c = a + b)
  (h3 : N = a * b * c)
  (h4 : N = 6 * (a + b + c)) :
  N = 156 ∨ N = 96 ∨ N = 84 ∧
  (156 + 96 + 84 = 336) :=
by {
  -- proof will go here
  sorry
}

end sum_of_all_possible_N_l17_17164


namespace time_per_step_l17_17454

def apply_and_dry_time (total_time steps : ℕ) : ℕ :=
  total_time / steps

theorem time_per_step : apply_and_dry_time 120 6 = 20 := by
  -- Proof omitted
  sorry

end time_per_step_l17_17454


namespace part1_part2_part3_l17_17615

noncomputable def f : ℝ → ℝ := sorry -- Given f is a function on ℝ with domain (0, +∞)

axiom domain_pos (x : ℝ) : 0 < x
axiom pos_condition (x : ℝ) (h : 1 < x) : 0 < f x
axiom functional_eq (x y : ℝ) : f (x * y) = f x + f y
axiom specific_value : f (1/3) = -1

-- (1) Prove: f(1/x) = -f(x)
theorem part1 (x : ℝ) (hx : 0 < x) : f (1 / x) = - f x := sorry

-- (2) Prove: f(x) is an increasing function on its domain
theorem part2 (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (h : x1 < x2) : f x1 < f x2 := sorry

-- (3) Prove the range of x for the inequality
theorem part3 (x : ℝ) (hx : 0 < x) (hx2 : 0 < x - 2) : 
  f x - f (1 / (x - 2)) ≥ 2 ↔ 1 + Real.sqrt 10 ≤ x := sorry

end part1_part2_part3_l17_17615


namespace directrix_of_parabola_l17_17683

theorem directrix_of_parabola (p : ℝ) (hp : 0 < p) (h_point : ∃ (x y : ℝ), y^2 = 2 * p * x ∧ (x = 2 ∧ y = 2)) :
  x = -1/2 :=
sorry

end directrix_of_parabola_l17_17683


namespace problem1_problem2_l17_17515

-- Problem 1
theorem problem1 :
  2 * Real.cos (Real.pi / 4) + (Real.pi - Real.sqrt 3)^0 - Real.sqrt 8 = 1 - Real.sqrt 2 := 
by
  sorry

-- Problem 2
theorem problem2 (m : ℝ) (h : m ≠ 1) :
  (2 / (m - 1) + 1) / ((2 * m + 2) / (m^2 - 2 * m + 1)) = (m - 1) / 2 :=
by
  sorry

end problem1_problem2_l17_17515


namespace sum_geometric_series_l17_17915

theorem sum_geometric_series :
  let a := (1 : ℚ) / 5
  let r := (1 : ℚ) / 5
  let n := 8
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 195312 / 781250 := by
    sorry

end sum_geometric_series_l17_17915


namespace Jamir_swims_more_l17_17745

def Julien_distance_per_day : ℕ := 50
def Sarah_distance_per_day (J : ℕ) : ℕ := 2 * J
def combined_distance_per_week (J S M : ℕ) : ℕ := 7 * (J + S + M)

theorem Jamir_swims_more :
  let J := Julien_distance_per_day
  let S := Sarah_distance_per_day J
  ∃ M, combined_distance_per_week J S M = 1890 ∧ (M - S = 20) := by
    let J := Julien_distance_per_day
    let S := Sarah_distance_per_day J
    use 120
    sorry

end Jamir_swims_more_l17_17745


namespace trig_second_quadrant_l17_17147

theorem trig_second_quadrant (α : ℝ) (h1 : α > π / 2) (h2 : α < π) :
  (|Real.sin α| / Real.sin α) - (|Real.cos α| / Real.cos α) = 2 :=
by
  sorry

end trig_second_quadrant_l17_17147


namespace polynomial_solutions_l17_17679

theorem polynomial_solutions :
  (∀ x : ℂ, (x^4 + 2*x^3 + 2*x^2 + 2*x + 1 = 0) ↔ (x = -1 ∨ x = Complex.I ∨ x = -Complex.I)) :=
by
  sorry

end polynomial_solutions_l17_17679


namespace final_value_of_S_is_10_l17_17459

-- Define the initial value of S
def initial_S : ℕ := 1

-- Define the sequence of I values
def I_values : List ℕ := [1, 3, 5]

-- Define the update operation on S
def update_S (S : ℕ) (I : ℕ) : ℕ := S + I

-- Final value of S after all updates
def final_S : ℕ := (I_values.foldl update_S initial_S)

-- The theorem stating that the final value of S is 10
theorem final_value_of_S_is_10 : final_S = 10 :=
by
  sorry

end final_value_of_S_is_10_l17_17459


namespace travel_paths_l17_17212

-- Definitions for conditions
def roads_AB : ℕ := 3
def roads_BC : ℕ := 2

-- The theorem statement
theorem travel_paths : roads_AB * roads_BC = 6 := by
  sorry

end travel_paths_l17_17212


namespace functional_identity_l17_17799

-- Define the set of non-negative integers
def S : Set ℕ := {n | n ≥ 0}

-- Define the function f with the required domain and codomain
def f (n : ℕ) : ℕ := n

-- The hypothesis: the functional equation satisfied by f
axiom functional_equation :
  ∀ m n : ℕ, f (m + f n) = f (f m) + f n

-- The theorem we want to prove
theorem functional_identity (n : ℕ) : f n = n :=
  sorry

end functional_identity_l17_17799


namespace fraction_meaningful_l17_17157

theorem fraction_meaningful (x : ℝ) : (x ≠ -1) ↔ (∃ k : ℝ, k = 1 / (x + 1)) :=
by
  sorry

end fraction_meaningful_l17_17157


namespace find_starting_number_l17_17110

theorem find_starting_number : 
  ∃ x : ℕ, (∃ n : ℕ, n = 21 ∧ (forall k, 1 ≤ k ∧ k ≤ n → x + k*19 ≤ 500) ∧ 
  (forall k, 1 ≤ k ∧ k < n → x + k*19 > 0)) ∧ x = 113 := by {
  sorry
}

end find_starting_number_l17_17110


namespace find_vector_at_t_zero_l17_17945

def vector_at_t (a d : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (a.1 + t*d.1, a.2 + t*d.2)

theorem find_vector_at_t_zero :
  ∃ (a d : ℝ × ℝ),
    vector_at_t a d 1 = (2, 3) ∧
    vector_at_t a d 4 = (8, -5) ∧
    vector_at_t a d 5 = (10, -9) ∧
    vector_at_t a d 0 = (0, 17/3) :=
by
  sorry

end find_vector_at_t_zero_l17_17945


namespace spencer_total_distance_l17_17564

def d1 : ℝ := 1.2
def d2 : ℝ := 0.6
def d3 : ℝ := 0.9
def d4 : ℝ := 1.7
def d5 : ℝ := 2.1
def d6 : ℝ := 1.3
def d7 : ℝ := 0.8

theorem spencer_total_distance : d1 + d2 + d3 + d4 + d5 + d6 + d7 = 8.6 :=
by
  sorry

end spencer_total_distance_l17_17564


namespace smallest_positive_integer_l17_17689

theorem smallest_positive_integer (n : ℕ) :
  (∃ n : ℕ, n > 0 ∧ n % 30 = 0 ∧ n % 40 = 0 ∧ n % 16 ≠ 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 30 = 0 ∧ m % 40 = 0 ∧ m % 16 ≠ 0) → n ≤ m) ↔ n = 120 :=
by
  sorry

end smallest_positive_integer_l17_17689


namespace find_length_of_GH_l17_17490

variable {A B C F G H : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
          [MetricSpace F] [MetricSpace G] [MetricSpace H]

variables (AB BC Res : ℝ)
variables (ratio1 ratio2 : ℝ)
variable (similar : SimilarTriangles A B C F G H)

def length_of_GH (GH : ℝ) : Prop :=
  GH = 15

theorem find_length_of_GH (h1 : AB = 15) (h2 : BC = 25) (h3 : ratio1 = 5) (h4 : ratio2 = 3)
  (h5 : similar) : ∃ GH, length_of_GH GH :=
by
  have ratio : ratio2 / ratio1 = 3 / 5 := by assumption
  sorry

end find_length_of_GH_l17_17490


namespace abc_value_l17_17088

theorem abc_value (a b c : ℂ) (h1 : 2 * a * b + 3 * b = -21)
                   (h2 : 2 * b * c + 3 * c = -21)
                   (h3 : 2 * c * a + 3 * a = -21) :
                   a * b * c = 105.75 := 
sorry

end abc_value_l17_17088


namespace solve_quadratic_l17_17591

theorem solve_quadratic (y : ℝ) :
  y^2 - 3 * y - 10 = -(y + 2) * (y + 6) ↔ (y = -1/2 ∨ y = -2) :=
by
  sorry

end solve_quadratic_l17_17591


namespace future_skyscraper_climb_proof_l17_17261

variable {H_f H_c H_fut : ℝ}

theorem future_skyscraper_climb_proof
  (H_f : ℝ)
  (H_c : ℝ := 3 * H_f)
  (H_fut : ℝ := 1.25 * H_c)
  (T_f : ℝ := 1) :
  (H_fut * T_f / H_f) > 2 * T_f :=
by
  -- specific calculations would go here
  sorry

end future_skyscraper_climb_proof_l17_17261


namespace best_is_man_l17_17046

structure Competitor where
  name : String
  gender : String
  age : Int
  is_twin : Bool

noncomputable def participants : List Competitor := [
  ⟨"man", "male", 30, false⟩,
  ⟨"sister", "female", 30, true⟩,
  ⟨"son", "male", 30, true⟩,
  ⟨"niece", "female", 25, false⟩
]

def are_different_gender (c1 c2 : Competitor) : Bool := c1.gender ≠ c2.gender
def has_same_age (c1 c2 : Competitor) : Bool := c1.age = c2.age

noncomputable def best_competitor : Competitor :=
  let best_candidate := participants[0] -- assuming "man" is the best for example's sake
  let worst_candidate := participants[2] -- assuming "son" is the worst for example's sake
  best_candidate

theorem best_is_man : best_competitor.name = "man" :=
by
  have h1 : are_different_gender (participants[0]) (participants[2]) := by sorry
  have h2 : has_same_age (participants[0]) (participants[2]) := by sorry
  exact sorry

end best_is_man_l17_17046


namespace least_n_froods_l17_17445

def froods_score (n : ℕ) : ℕ := n * (n + 1) / 2
def eating_score (n : ℕ) : ℕ := n ^ 2

theorem least_n_froods :
    ∃ n : ℕ, 0 < n ∧ (froods_score n > eating_score n) ∧ (∀ m : ℕ, 0 < m ∧ m < n → froods_score m ≤ eating_score m) :=
  sorry

end least_n_froods_l17_17445


namespace inequality_am_gm_l17_17339

theorem inequality_am_gm (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (x^4 + y^2) + y / (x^2 + y^4)) ≤ (1 / (x * y)) :=
by
  sorry

end inequality_am_gm_l17_17339


namespace coins_value_percentage_l17_17229

theorem coins_value_percentage :
  let penny_value := 1
  let nickel_value := 5
  let dime_value := 10
  let quarter_value := 25
  let total_value_cents := (1 * penny_value) + (2 * nickel_value) + (1 * dime_value) + (2 * quarter_value)
  (total_value_cents / 100) * 100 = 71 :=
by
  sorry

end coins_value_percentage_l17_17229


namespace white_pairs_coincide_l17_17995

def num_red : Nat := 4
def num_blue : Nat := 4
def num_green : Nat := 2
def num_white : Nat := 6
def red_pairs : Nat := 3
def blue_pairs : Nat := 2
def green_pairs : Nat := 1 
def red_white_pairs : Nat := 2
def green_blue_pairs : Nat := 1

theorem white_pairs_coincide :
  (num_red = 4) ∧ 
  (num_blue = 4) ∧ 
  (num_green = 2) ∧ 
  (num_white = 6) ∧ 
  (red_pairs = 3) ∧ 
  (blue_pairs = 2) ∧ 
  (green_pairs = 1) ∧ 
  (red_white_pairs = 2) ∧ 
  (green_blue_pairs = 1) → 
  4 = 4 :=
by
  sorry

end white_pairs_coincide_l17_17995


namespace number_of_squares_in_grid_l17_17797

-- Grid of size 6 × 6 composed entirely of squares.
def grid_size : Nat := 6

-- Definition of the function that counts the number of squares of a given size in an n × n grid.
def count_squares (n : Nat) (size : Nat) : Nat :=
  (n - size + 1) * (n - size + 1)

noncomputable def total_squares : Nat :=
  List.sum (List.map (count_squares grid_size) (List.range grid_size).tail)  -- Using tail to skip zero size

theorem number_of_squares_in_grid : total_squares = 86 := by
  sorry

end number_of_squares_in_grid_l17_17797


namespace distinct_triplet_inequality_l17_17634

theorem distinct_triplet_inequality (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  abs (a / (b - c)) + abs (b / (c - a)) + abs (c / (a - b)) ≥ 2 := 
sorry

end distinct_triplet_inequality_l17_17634


namespace ken_pencils_kept_l17_17035

-- Define the known quantities and conditions
def initial_pencils : ℕ := 250
def manny_pencils : ℕ := 25
def nilo_pencils : ℕ := manny_pencils * 2
def carlos_pencils : ℕ := nilo_pencils / 2
def tina_pencils : ℕ := carlos_pencils + 10
def rina_pencils : ℕ := tina_pencils - 20

-- Formulate the total pencils given away
def total_given_away : ℕ :=
  manny_pencils + nilo_pencils + carlos_pencils + tina_pencils + rina_pencils

-- Prove the final number of pencils Ken kept.
theorem ken_pencils_kept : initial_pencils - total_given_away = 100 :=
by
  sorry

end ken_pencils_kept_l17_17035


namespace circle_arc_and_circumference_l17_17325

theorem circle_arc_and_circumference (C_X : ℝ) (θ_YOZ : ℝ) (C_D : ℝ) (r_X r_D : ℝ) :
  C_X = 100 ∧ θ_YOZ = 150 ∧ r_X = 50 / π ∧ r_D = 25 / π ∧ C_D = 50 →
  (θ_YOZ / 360) * C_X = 500 / 12 ∧ 2 * π * r_D = C_D :=
by sorry

end circle_arc_and_circumference_l17_17325


namespace find_x_l17_17119

theorem find_x 
  (x : ℝ)
  (h : 0.4 * x + (0.6 * 0.8) = 0.56) : 
  x = 0.2 := sorry

end find_x_l17_17119


namespace max_digit_e_l17_17875

theorem max_digit_e 
  (d e : ℕ) 
  (digits : ∀ (n : ℕ), n ≤ 9) 
  (even_e : e % 2 = 0) 
  (div_9 : (22 + d + e) % 9 = 0) 
  : e ≤ 8 :=
sorry

end max_digit_e_l17_17875


namespace novel_pages_l17_17780

theorem novel_pages (x : ℕ) (pages_per_day_in_reality : ℕ) (planned_days actual_days : ℕ)
  (h1 : planned_days = 20)
  (h2 : actual_days = 15)
  (h3 : pages_per_day_in_reality = x + 20)
  (h4 : pages_per_day_in_reality * actual_days = x * planned_days) :
  x * planned_days = 1200 :=
by
  sorry

end novel_pages_l17_17780


namespace minimal_divisors_at_kth_place_l17_17680

open Nat

theorem minimal_divisors_at_kth_place (n k : ℕ) (hnk : n ≥ k) (S : ℕ) (hS : ∃ d : ℕ, d ≥ n ∧ d = S ∧ ∀ i, i ≤ d → exists m, m = d):
  ∃ (min_div : ℕ), min_div = ⌈ (n : ℝ) / k ⌉ :=
by
  sorry

end minimal_divisors_at_kth_place_l17_17680


namespace BANANA_arrangements_l17_17956

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l17_17956


namespace sum_of_common_ratios_is_five_l17_17361

theorem sum_of_common_ratios_is_five {k p r : ℝ} 
  (h1 : p ≠ r)                       -- different common ratios
  (h2 : k ≠ 0)                       -- non-zero k
  (a2 : ℝ := k * p)                  -- term a2
  (a3 : ℝ := k * p^2)                -- term a3
  (b2 : ℝ := k * r)                  -- term b2
  (b3 : ℝ := k * r^2)                -- term b3
  (h3 : a3 - b3 = 5 * (a2 - b2))     -- given condition
  : p + r = 5 := 
by
  sorry

end sum_of_common_ratios_is_five_l17_17361


namespace recycling_target_l17_17355

/-- Six Grade 4 sections launched a recycling drive where they collect old newspapers to recycle.
Each section collected 280 kilos in two weeks. After the third week, they found that they need 320 kilos more to reach their target.
  How many kilos of the newspaper is their target? -/
theorem recycling_target (sections : ℕ) (kilos_collected_2_weeks : ℕ) (additional_kilos : ℕ) : 
  sections = 6 ∧ kilos_collected_2_weeks = 280 ∧ additional_kilos = 320 → 
  (sections * (kilos_collected_2_weeks / 2) * 3 + additional_kilos) = 2840 :=
by
  sorry

end recycling_target_l17_17355


namespace focal_length_is_correct_l17_17452

def hyperbola_eqn : Prop := (∀ x y : ℝ, (x^2 / 4) - (y^2 / 9) = 1 → True)

noncomputable def focal_length_of_hyperbola : ℝ :=
  2 * Real.sqrt (4 + 9)

theorem focal_length_is_correct : hyperbola_eqn → focal_length_of_hyperbola = 2 * Real.sqrt 13 := by
  intro h
  sorry

end focal_length_is_correct_l17_17452


namespace circle_incircle_tangent_radius_l17_17494

theorem circle_incircle_tangent_radius (r1 r2 r3 : ℕ) (k : ℕ) (h1 : r1 = 1) (h2 : r2 = 4) (h3 : r3 = 9) : 
  k = 11 :=
by
  -- Definitions according to the problem
  let k₁ := r1
  let k₂ := r2
  let k₃ := r3
  -- Hypotheses given by the problem
  have h₁ : k₁ = 1 := h1
  have h₂ : k₂ = 4 := h2
  have h₃ : k₃ = 9 := h3
  -- Prove the radius of the incircle k
  sorry

end circle_incircle_tangent_radius_l17_17494


namespace triangle_solution_l17_17987

noncomputable def solve_triangle (a : ℝ) (α : ℝ) (t : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let s := 75
  let b := 41
  let c := 58
  let β := 43 + 36 / 60 + 10 / 3600
  let γ := 77 + 19 / 60 + 11 / 3600
  ((b, c), (β, γ))

theorem triangle_solution :
  solve_triangle 51 (59 + 4 / 60 + 39 / 3600) 1020 = ((41, 58), (43 + 36 / 60 + 10 / 3600, 77 + 19 / 60 + 11 / 3600)) :=
sorry  

end triangle_solution_l17_17987


namespace deborah_total_cost_l17_17802

-- Standard postage per letter
def stdPostage : ℝ := 1.08

-- Additional charge for international shipping per letter
def intlAdditional : ℝ := 0.14

-- Number of domestic and international letters
def numDomestic : ℕ := 2
def numInternational : ℕ := 2

-- Expected total cost for four letters
def expectedTotalCost : ℝ := 4.60

theorem deborah_total_cost :
  (numDomestic * stdPostage) + (numInternational * (stdPostage + intlAdditional)) = expectedTotalCost :=
by
  -- proof skipped
  sorry

end deborah_total_cost_l17_17802


namespace prime_fraction_identity_l17_17643

theorem prime_fraction_identity : ∀ (p q : ℕ),
  Prime p → Prime q → p = 2 → q = 2 →
  (pq + p^p + q^q) / (p + q) = 3 :=
by
  intros p q hp hq hp2 hq2
  sorry

end prime_fraction_identity_l17_17643


namespace integer_solution_l17_17865

theorem integer_solution (x : ℤ) (h : (Int.natAbs x - 1) * x ^ 2 - 9 = 1) : x = 2 ∨ x = -2 ∨ x = 3 ∨ x = -3 :=
by
  sorry

end integer_solution_l17_17865


namespace calculate_expression_l17_17417

theorem calculate_expression (p q : ℝ) (hp : p + q = 7) (hq : p * q = 12) :
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 3691 := 
by sorry

end calculate_expression_l17_17417


namespace average_of_t_b_c_29_l17_17852
-- Importing the entire Mathlib library

theorem average_of_t_b_c_29 (t b c : ℝ) 
  (h : (t + b + c + 14 + 15) / 5 = 12) : 
  (t + b + c + 29) / 4 = 15 :=
by 
  sorry

end average_of_t_b_c_29_l17_17852


namespace most_probable_hits_l17_17627

variable (n : ℕ) (p : ℝ) (q : ℝ) (k : ℕ)
variable (h1 : n = 5) (h2 : p = 0.6) (h3 : q = 1 - p)

theorem most_probable_hits : k = 3 := by
  -- Define the conditions
  have hp : p = 0.6 := h2
  have hn : n = 5 := h1
  have hq : q = 1 - p := h3

  -- Set the expected value for the number of hits
  let expected := n * p

  -- Use the bounds for the most probable number of successes (k_0)
  have bounds := expected - q ≤ k ∧ k ≤ expected + p

  -- Proof step analysis can go here
  sorry

end most_probable_hits_l17_17627


namespace solve_abs_quadratic_l17_17583

theorem solve_abs_quadratic :
  ∀ x : ℝ, abs (x^2 - 4 * x + 4) = 3 - x ↔ (x = (3 + Real.sqrt 5) / 2 ∨ x = (3 - Real.sqrt 5) / 2) :=
by
  sorry

end solve_abs_quadratic_l17_17583


namespace largest_number_is_A_l17_17345

theorem largest_number_is_A (x y z w: ℕ):
  x = (8 * 9 + 5) → -- 85 in base 9 to decimal
  y = (2 * 6 * 6) → -- 200 in base 6 to decimal
  z = ((6 * 11) + 8) → -- 68 in base 11 to decimal
  w = 70 → -- 70 in base 10 remains 70
  max (max x y) (max z w) = x := -- 77 is the maximum
by
  sorry

end largest_number_is_A_l17_17345


namespace sum_medians_is_64_l17_17120

noncomputable def median (l: List ℝ) : ℝ := sorry  -- Placeholder for median calculation

open List

/-- Define the scores for players A and B as lists of real numbers -/
def player_a_scores : List ℝ := sorry
def player_b_scores : List ℝ := sorry

/-- Prove that the sum of the medians of the scores lists is 64 -/
theorem sum_medians_is_64 : median player_a_scores + median player_b_scores = 64 := sorry

end sum_medians_is_64_l17_17120


namespace fence_perimeter_l17_17605

noncomputable def posts (n : ℕ) := 36
noncomputable def space_between_posts (d : ℕ) := 6
noncomputable def length_is_twice_width (l w : ℕ) := l = 2 * w

theorem fence_perimeter (n d w l perimeter : ℕ)
  (h1 : posts n = 36)
  (h2 : space_between_posts d = 6)
  (h3 : length_is_twice_width l w)
  : perimeter = 216 :=
sorry

end fence_perimeter_l17_17605


namespace updated_mean_of_decremented_observations_l17_17132

theorem updated_mean_of_decremented_observations (n : ℕ) (initial_mean decrement : ℝ)
  (h₀ : n = 50) (h₁ : initial_mean = 200) (h₂ : decrement = 6) :
  ((n * initial_mean) - (n * decrement)) / n = 194 := by
  sorry

end updated_mean_of_decremented_observations_l17_17132


namespace fruit_trees_l17_17290

theorem fruit_trees (total_streets : ℕ) 
  (fruit_trees_every_other : total_streets % 2 = 0) 
  (equal_fruit_trees : ∀ n : ℕ, 3 * n = total_streets / 2) : 
  ∃ n : ℕ, n = total_streets / 6 :=
by
  sorry

end fruit_trees_l17_17290


namespace candy_lasts_for_days_l17_17596

-- Definitions based on conditions
def candy_from_neighbors : ℕ := 75
def candy_from_sister : ℕ := 130
def candy_traded : ℕ := 25
def candy_lost : ℕ := 15
def candy_eaten_per_day : ℕ := 7

-- Total candy calculation
def total_candy : ℕ := candy_from_neighbors + candy_from_sister - candy_traded - candy_lost
def days_candy_lasts : ℕ := total_candy / candy_eaten_per_day

-- Proof statement
theorem candy_lasts_for_days : days_candy_lasts = 23 := by
  -- sorry is used to skip the actual proof
  sorry

end candy_lasts_for_days_l17_17596


namespace midpoint_trajectory_extension_trajectory_l17_17824

-- Define the conditions explicitly

def is_midpoint (M A O : ℝ × ℝ) : Prop :=
  M = ((O.1 + A.1) / 2, (O.2 + A.2) / 2)

def on_circle (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 + P.2 ^ 2 - 8 * P.1 = 0

-- First problem: Trajectory equation of the midpoint M
theorem midpoint_trajectory (M O A : ℝ × ℝ) (hO : O = (0,0)) (hA : on_circle A) (hM : is_midpoint M A O) :
  M.1 ^ 2 + M.2 ^ 2 - 4 * M.1 = 0 :=
sorry

-- Define the condition for N
def extension_point (O A N : ℝ × ℝ) : Prop :=
  (A.1 - O.1) * 2 = N.1 - O.1 ∧ (A.2 - O.2) * 2 = N.2 - O.2

-- Second problem: Trajectory equation of the point N
theorem extension_trajectory (N O A : ℝ × ℝ) (hO : O = (0,0)) (hA : on_circle A) (hN : extension_point O A N) :
  N.1 ^ 2 + N.2 ^ 2 - 16 * N.1 = 0 :=
sorry

end midpoint_trajectory_extension_trajectory_l17_17824


namespace valid_B_sets_l17_17891

def A : Set ℝ := {x | 0 < x ∧ x < 2}

theorem valid_B_sets (B : Set ℝ) : A ∩ B = B ↔ B = ∅ ∨ B = {1} ∨ B = A :=
by
  sorry

end valid_B_sets_l17_17891


namespace problem_decimal_parts_l17_17437

theorem problem_decimal_parts :
  let a := 5 + Real.sqrt 7 - 7
  let b := 5 - Real.sqrt 7 - 2
  (a + b) ^ 2023 = 1 :=
by
  sorry

end problem_decimal_parts_l17_17437


namespace triangle_AB_length_correct_l17_17296

theorem triangle_AB_length_correct (BC AC : Real) (A : Real) 
  (hBC : BC = Real.sqrt 7) 
  (hAC : AC = 2 * Real.sqrt 3) 
  (hA : A = Real.pi / 6) :
  ∃ (AB : Real), (AB = 5 ∨ AB = 1) :=
by
  sorry

end triangle_AB_length_correct_l17_17296


namespace remainder_of_M_mod_1000_l17_17828

def M : ℕ := Nat.choose 9 8

theorem remainder_of_M_mod_1000 : M % 1000 = 9 := by
  sorry

end remainder_of_M_mod_1000_l17_17828


namespace jill_total_tax_percentage_l17_17112

theorem jill_total_tax_percentage (spent_clothing_percent spent_food_percent spent_other_percent tax_clothing_percent tax_food_percent tax_other_percent : ℝ)
  (h1 : spent_clothing_percent = 0.5)
  (h2 : spent_food_percent = 0.25)
  (h3 : spent_other_percent = 0.25)
  (h4 : tax_clothing_percent = 0.1)
  (h5 : tax_food_percent = 0)
  (h6 : tax_other_percent = 0.2) :
  ((spent_clothing_percent * tax_clothing_percent + spent_food_percent * tax_food_percent + spent_other_percent * tax_other_percent) * 100) = 10 :=
by
  sorry

end jill_total_tax_percentage_l17_17112


namespace arithmetic_sequence_problem_l17_17648

theorem arithmetic_sequence_problem (q a₁ a₂ a₃ : ℕ) (a : ℕ → ℕ) (c : ℕ → ℕ) (S T : ℕ → ℕ)
  (h1 : q > 1)
  (h2 : a₁ + a₂ + a₃ = 7)
  (h3 : a₁ + 3 + a₃ + 4 = 6 * a₂) :
  (∀ n : ℕ, a n = 2^(n-1)) ∧ (∀ n : ℕ, T n = (3 * n - 5) * 2^n + 5) :=
by
  sorry

end arithmetic_sequence_problem_l17_17648


namespace arithmetic_sequence_term_l17_17140

variable (a : ℕ → ℕ)
variable (d : ℕ)

-- Conditions
def common_difference := d = 2
def value_a_2007 := a 2007 = 2007

-- Question to be proved
theorem arithmetic_sequence_term :
  common_difference d →
  value_a_2007 a →
  a 2009 = 2011 :=
by
  sorry

end arithmetic_sequence_term_l17_17140


namespace intersects_line_l17_17073

theorem intersects_line (x y : ℝ) : 
  (3 * x + 2 * y = 5) ∧ ((x / 3) + (y / 2) = 1) → ∃ x y : ℝ, (3 * x + 2 * y = 5) ∧ ((x / 3) + (y / 2) = 1) :=
by
  intro h
  sorry

end intersects_line_l17_17073


namespace part1_part2_l17_17014

variable (f : ℝ → ℝ)

-- Conditions
axiom h1 : ∀ x y : ℝ, f (x - y) = f x / f y
axiom h2 : ∀ x : ℝ, f x > 0
axiom h3 : ∀ x y : ℝ, x < y → f x > f y

-- First part: f(0) = 1 and proving f(x + y) = f(x) * f(y)
theorem part1 : f 0 = 1 ∧ (∀ x y : ℝ, f (x + y) = f x * f y) :=
sorry

-- Second part: Given f(-1) = 3, solve the inequality
axiom h4 : f (-1) = 3

theorem part2 : {x : ℝ | (x ≤ 3) ∨ (x ≥ 4)} = {x : ℝ | f (x^2 - 7*x + 10) ≤ f (-2)} :=
sorry

end part1_part2_l17_17014


namespace factorize_quadratic_l17_17427

theorem factorize_quadratic : ∀ x : ℝ, x^2 - 7*x + 10 = (x - 2)*(x - 5) :=
by
  sorry

end factorize_quadratic_l17_17427


namespace race_order_count_l17_17808

-- Define the problem conditions
def participants : List String := ["Harry", "Ron", "Neville", "Hermione"]
def no_ties : Prop := True -- Since no ties are given directly, we denote this as always true for simplicity

-- Define the proof problem statement
theorem race_order_count (h_no_ties : no_ties) : participants.permutations.length = 24 := 
by
  -- Placeholder for proof
  sorry

end race_order_count_l17_17808


namespace sequence_sum_l17_17674

open BigOperators

-- Define the general term
def term (n : ℕ) : ℚ := n * (1 - (1 / n))

-- Define the index range for the sequence
def index_range : Finset ℕ := Finset.range 9 \ {0, 1}

-- Lean statement of the problem
theorem sequence_sum : ∑ n in index_range, term (n + 2) = 45 := by
  sorry

end sequence_sum_l17_17674


namespace cone_radius_l17_17750

theorem cone_radius (h : ℝ) (V : ℝ) (π : ℝ) (r : ℝ)
    (h_def : h = 21)
    (V_def : V = 2199.114857512855)
    (volume_formula : V = (1/3) * π * r^2 * h) : r = 10 :=
by {
  sorry
}

end cone_radius_l17_17750


namespace Brazil_wins_10_l17_17091

/-- In the year 3000, the World Hockey Championship will follow new rules: 12 points will be awarded for a win, 
5 points will be deducted for a loss, and no points will be awarded for a draw. If the Brazilian team plays 
38 matches, scores 60 points, and loses at least once, then the number of wins they can achieve is 10. 
List all possible scenarios and justify why there cannot be any others. -/
theorem Brazil_wins_10 (x y z : ℕ) 
    (h1: x + y + z = 38) 
    (h2: 12 * x - 5 * y = 60) 
    (h3: y ≥ 1)
    (h4: z ≥ 0): 
  x = 10 :=
by
  sorry

end Brazil_wins_10_l17_17091


namespace field_area_l17_17129

-- Define the given conditions and prove the area of the field
theorem field_area (x y : ℕ) 
  (h1 : 2*(x + 20) + 2*y = 2*(2*x + 2*y))
  (h2 : 2*x + 2*(2*y) = 2*x + 2*y + 18) : x * y = 99 := by 
{
  sorry
}

end field_area_l17_17129


namespace find_matrix_calculate_M5_alpha_l17_17594

-- Define the matrix M, eigenvalues, eigenvectors and vector α
def M : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 2], ![3, 2]]
def alpha : Fin 2 → ℝ := ![-1, 1]
def e1 : Fin 2 → ℝ := ![2, 3]
def e2 : Fin 2 → ℝ := ![1, -1]
def lambda1 : ℝ := 4
def lambda2 : ℝ := -1

-- Conditions: eigenvalues and their corresponding eigenvectors
axiom h1 : M.mulVec e1 = lambda1 • e1
axiom h2 : M.mulVec e2 = lambda2 • e2

-- Condition: given vector α
axiom h3 : alpha = - e2

-- Prove that M is the matrix given by the components
theorem find_matrix : M = ![![1, 2], ![3, 2]] :=
sorry

-- Prove that M^5 times α equals the given vector
theorem calculate_M5_alpha : (M^5).mulVec alpha = ![-1, 1] :=
sorry

end find_matrix_calculate_M5_alpha_l17_17594


namespace range_of_m_l17_17580

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : x > 0) (hy : y > 0) (hxy : (1/x) + (4/y) = 1) :
  (x + y > m^2 + 8 * m) → (-9 < m ∧ m < 1) :=
by 
  sorry

end range_of_m_l17_17580


namespace weight_of_each_bag_l17_17644

theorem weight_of_each_bag 
  (total_potatoes_weight : ℕ) (damaged_potatoes_weight : ℕ) 
  (bag_price : ℕ) (total_revenue : ℕ) (sellable_potatoes_weight : ℕ) (number_of_bags : ℕ) 
  (weight_of_each_bag : ℕ) :
  total_potatoes_weight = 6500 →
  damaged_potatoes_weight = 150 →
  sellable_potatoes_weight = total_potatoes_weight - damaged_potatoes_weight →
  bag_price = 72 →
  total_revenue = 9144 →
  number_of_bags = total_revenue / bag_price →
  weight_of_each_bag * number_of_bags = sellable_potatoes_weight →
  weight_of_each_bag = 50 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end weight_of_each_bag_l17_17644


namespace graphs_symmetric_about_a_axis_of_symmetry_l17_17608

def graph_symmetric_about_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a - x) = f (x - a)

theorem graphs_symmetric_about_a (f : ℝ → ℝ) (a : ℝ) :
  ∀ x, f (x - a) = f (a - (x - a)) :=
sorry

theorem axis_of_symmetry (f : ℝ → ℝ) :
  (∀ x : ℝ, f (1 + 2 * x) = f (1 - 2 * x)) →
  ∀ x, f x = f (2 - x) := 
sorry

end graphs_symmetric_about_a_axis_of_symmetry_l17_17608


namespace percentage_increase_area_l17_17931

theorem percentage_increase_area (L W : ℝ) (hL : 0 < L) (hW : 0 < W) :
  let A := L * W
  let A' := (1.35 * L) * (1.35 * W)
  let percentage_increase := ((A' - A) / A) * 100
  percentage_increase = 82.25 :=
by
  sorry

end percentage_increase_area_l17_17931


namespace blisters_on_rest_of_body_l17_17957

theorem blisters_on_rest_of_body (blisters_per_arm total_blisters : ℕ) (h1 : blisters_per_arm = 60) (h2 : total_blisters = 200) : 
  total_blisters - 2 * blisters_per_arm = 80 :=
by {
  -- The proof can be written here
  sorry
}

end blisters_on_rest_of_body_l17_17957


namespace phantom_needs_more_money_l17_17933

def amount_phantom_has : ℤ := 50
def cost_black : ℤ := 11
def count_black : ℕ := 2
def cost_red : ℤ := 15
def count_red : ℕ := 3
def cost_yellow : ℤ := 13
def count_yellow : ℕ := 2

def total_cost : ℤ := cost_black * count_black + cost_red * count_red + cost_yellow * count_yellow
def additional_amount_needed : ℤ := total_cost - amount_phantom_has

theorem phantom_needs_more_money : additional_amount_needed = 43 := by
  sorry

end phantom_needs_more_money_l17_17933


namespace non_congruent_rectangles_count_l17_17796

theorem non_congruent_rectangles_count (h w : ℕ) (P : ℕ) (multiple_of_4: ℕ → Prop) :
  P = 80 →
  w ≥ 1 ∧ h ≥ 1 →
  P = 2 * (w + h) →
  (multiple_of_4 w ∨ multiple_of_4 h) →
  (∀ k, multiple_of_4 k ↔ ∃ m, k = 4 * m) →
  ∃ n, n = 5 :=
by
  sorry

end non_congruent_rectangles_count_l17_17796


namespace no_integer_solution_l17_17175

theorem no_integer_solution :
  ∀ (x : ℤ), ¬ (x^2 + 3 < 2 * x) :=
by
  intro x
  sorry

end no_integer_solution_l17_17175


namespace part1_equation_part2_equation_l17_17362

-- Part (Ⅰ)
theorem part1_equation :
  (- ((-1) ^ 1000) - 2.45 * 8 + 2.55 * (-8) = -41) :=
by
  sorry

-- Part (Ⅱ)
theorem part2_equation :
  ((1 / 6 - 1 / 3 + 0.25) / (- (1 / 12)) = -1) :=
by
  sorry

end part1_equation_part2_equation_l17_17362


namespace cannot_fit_all_pictures_l17_17492

theorem cannot_fit_all_pictures 
  (typeA_capacity : Nat) (typeB_capacity : Nat) (typeC_capacity : Nat)
  (typeA_count : Nat) (typeB_count : Nat) (typeC_count : Nat)
  (total_pictures : Nat)
  (h1 : typeA_capacity = 12)
  (h2 : typeB_capacity = 18)
  (h3 : typeC_capacity = 24)
  (h4 : typeA_count = 6)
  (h5 : typeB_count = 4)
  (h6 : typeC_count = 3)
  (h7 : total_pictures = 480) :
  (typeA_capacity * typeA_count + typeB_capacity * typeB_count + typeC_capacity * typeC_count < total_pictures) :=
  by sorry

end cannot_fit_all_pictures_l17_17492


namespace bamboo_volume_l17_17744

theorem bamboo_volume :
  ∃ (a₁ d a₅ : ℚ), 
  (4 * a₁ + 6 * d = 5) ∧ 
  (3 * a₁ + 21 * d = 4) ∧ 
  (a₅ = a₁ + 4 * d) ∧ 
  (a₅ = 85 / 66) :=
sorry

end bamboo_volume_l17_17744


namespace jackson_meat_left_l17_17665

theorem jackson_meat_left (total_meat : ℕ) (meatballs_fraction : ℚ) (spring_rolls_meat : ℕ) :
  total_meat = 20 →
  meatballs_fraction = 1/4 →
  spring_rolls_meat = 3 →
  total_meat - (meatballs_fraction * total_meat + spring_rolls_meat) = 12 := by
  intros ht hm hs
  sorry

end jackson_meat_left_l17_17665


namespace carrots_left_over_l17_17343

theorem carrots_left_over (c g : ℕ) (h₁ : c = 47) (h₂ : g = 4) : c % g = 3 :=
by
  sorry

end carrots_left_over_l17_17343


namespace alcohol_percentage_solution_x_l17_17963

theorem alcohol_percentage_solution_x :
  ∃ (P : ℝ), 
  (∀ (vol_x vol_y : ℝ), vol_x = 50 → vol_y = 150 →
    ∀ (percent_y percent_new : ℝ), percent_y = 30 → percent_new = 25 →
      ((P / 100) * vol_x + (percent_y / 100) * vol_y) / (vol_x + vol_y) = percent_new) → P = 10 :=
by
  -- Given conditions
  let vol_x := 50
  let vol_y := 150
  let percent_y := 30
  let percent_new := 25

  -- The proof body should be here
  sorry

end alcohol_percentage_solution_x_l17_17963


namespace largest_p_plus_q_l17_17328

-- All required conditions restated as Assumptions
def triangle {R : Type*} [LinearOrderedField R] (p q : R) : Prop :=
  let B : R × R := (10, 15)
  let C : R × R := (25, 15)
  let A : R × R := (p, q)
  let M : R × R := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let area : R := (1 / 2) * abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))
  let median_slope : R := (A.2 - M.2) / (A.1 - M.1)
  area = 100 ∧ median_slope = -3

-- Statement to be proven
theorem largest_p_plus_q {R : Type*} [LinearOrderedField R] (p q : R) :
  triangle p q → p + q = 70 / 3 :=
by
  sorry

end largest_p_plus_q_l17_17328


namespace root_equality_l17_17455

theorem root_equality (p q : ℝ) (h1 : 1 + p + q = (2 - 2 * q) / p) (h2 : 1 + p + q = (1 - p + q) / q) :
  p + q = 1 :=
sorry

end root_equality_l17_17455


namespace hawkeye_remaining_money_l17_17159

-- Define the conditions
def cost_per_charge : ℝ := 3.5
def number_of_charges : ℕ := 4
def budget : ℝ := 20

-- Define the theorem to prove the remaining money
theorem hawkeye_remaining_money : 
  budget - (number_of_charges * cost_per_charge) = 6 := by
  sorry

end hawkeye_remaining_money_l17_17159


namespace Lin_peels_15_potatoes_l17_17659

-- Define the conditions
def total_potatoes : Nat := 60
def homer_rate : Nat := 2 -- potatoes per minute
def christen_rate : Nat := 3 -- potatoes per minute
def lin_rate : Nat := 4 -- potatoes per minute
def christen_join_time : Nat := 6 -- minutes
def lin_join_time : Nat := 9 -- minutes

-- Prove that Lin peels 15 potatoes
theorem Lin_peels_15_potatoes :
  ∃ (lin_potatoes : Nat), lin_potatoes = 15 :=
by
  sorry

end Lin_peels_15_potatoes_l17_17659


namespace create_proper_six_sided_figure_l17_17000

-- Definition of a matchstick configuration
structure MatchstickConfig where
  sides : ℕ
  matchsticks : ℕ

-- Initial configuration: a regular hexagon with 6 matchsticks
def initialConfig : MatchstickConfig := ⟨6, 6⟩

-- Condition: Cannot lay any stick on top of another, no free ends
axiom no_overlap (cfg : MatchstickConfig) : Prop
axiom no_free_ends (cfg : MatchstickConfig) : Prop

-- New configuration after adding 3 matchsticks
def newConfig : MatchstickConfig := ⟨6, 9⟩

-- Theorem stating the possibility to create a proper figure with six sides
theorem create_proper_six_sided_figure : no_overlap newConfig → no_free_ends newConfig → newConfig.sides = 6 :=
by
  sorry

end create_proper_six_sided_figure_l17_17000


namespace canvas_decreased_by_40_percent_l17_17233

noncomputable def canvas_decrease (P C : ℝ) (x d : ℝ) : Prop :=
  (P = 4 * C) ∧
  ((P - 0.60 * P) + (C - (x / 100) * C) = (1 - d / 100) * (P + C)) ∧
  (d = 55.99999999999999)

theorem canvas_decreased_by_40_percent (P C : ℝ) (x d : ℝ) 
  (h : canvas_decrease P C x d) : x = 40 :=
by
  sorry

end canvas_decreased_by_40_percent_l17_17233


namespace sets_of_earrings_l17_17588

namespace EarringsProblem

variables (magnets buttons gemstones earrings : ℕ)

theorem sets_of_earrings (h1 : gemstones = 24)
                         (h2 : gemstones = 3 * buttons)
                         (h3 : buttons = magnets / 2)
                         (h4 : earrings = magnets / 2)
                         (h5 : ∀ n : ℕ, n % 2 = 0 → ∃ k, n = 2 * k) :
  earrings = 8 :=
by
  sorry

end EarringsProblem

end sets_of_earrings_l17_17588


namespace highest_score_is_151_l17_17163

-- Definitions for the problem conditions
def total_runs : ℕ := 2704
def total_runs_excluding_HL : ℕ := 2552

variables (H L : ℕ) 

-- Problem conditions as hypotheses
axiom h1 : H - L = 150
axiom h2 : H + L = 152
axiom h3 : 2704 = 2552 + H + L

-- Proof statement
theorem highest_score_is_151 (H L : ℕ) (h1 : H - L = 150) (h2 : H + L = 152) (h3 : 2704 = 2552 + H + L) : H = 151 :=
by sorry

end highest_score_is_151_l17_17163


namespace floor_factorial_expression_l17_17215

-- Mathematical definitions (conditions)
def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

-- Mathematical proof problem (statement)
theorem floor_factorial_expression :
  Int.floor ((factorial 2007 + factorial 2004 : ℚ) / (factorial 2006 + factorial 2005)) = 2006 :=
sorry

end floor_factorial_expression_l17_17215


namespace downward_parabola_with_symmetry_l17_17518

-- Define the general form of the problem conditions in Lean
theorem downward_parabola_with_symmetry (k : ℝ) :
  ∃ a : ℝ, a < 0 ∧ ∃ h : ℝ, h = 3 ∧ ∃ k : ℝ, k = k ∧ ∃ (y x : ℝ), y = a * (x - h)^2 + k :=
sorry

end downward_parabola_with_symmetry_l17_17518


namespace average_percentage_l17_17258

theorem average_percentage (n1 n2 : ℕ) (s1 s2 : ℕ)
  (h1 : n1 = 15) (h2 : s1 = 80) (h3 : n2 = 10) (h4 : s2 = 90) :
  (n1 * s1 + n2 * s2) / (n1 + n2) = 84 :=
by
  sorry

end average_percentage_l17_17258


namespace simplify_evaluate_expr_l17_17141

theorem simplify_evaluate_expr (x : ℕ) (h : x = 2023) : (x + 1) ^ 2 - x * (x + 1) = 2024 := 
by 
  sorry

end simplify_evaluate_expr_l17_17141


namespace determine_k_l17_17714

theorem determine_k (k : ℝ) :
  (∀ x : ℝ, (x - 3) * (x - 5) = k - 4 * x) ↔ k = 11 :=
by
  sorry

end determine_k_l17_17714


namespace least_number_of_coins_l17_17706

theorem least_number_of_coins (n : ℕ) : 
  (n % 7 = 3) ∧ (n % 5 = 4) ∧ (∀ m : ℕ, (m % 7 = 3) ∧ (m % 5 = 4) → n ≤ m) → n = 24 :=
by
  sorry

end least_number_of_coins_l17_17706


namespace expected_americans_with_allergies_l17_17069

theorem expected_americans_with_allergies (prob : ℚ) (sample_size : ℕ) (h_prob : prob = 1/5) (h_sample_size : sample_size = 250) :
  sample_size * prob = 50 := by
  rw [h_prob, h_sample_size]
  norm_num

#print expected_americans_with_allergies

end expected_americans_with_allergies_l17_17069


namespace harmonic_mean_pairs_l17_17791

open Nat

theorem harmonic_mean_pairs :
  ∃ n, n = 199 ∧ 
  (∀ (x y : ℕ), 0 < x → 0 < y → 
  x < y → (2 * x * y) / (x + y) = 6^10 → 
  x * y - (3^10 * 2^9) * (x - 1) - (3^10 * 2^9) * (y - 1) = 3^20 * 2^18) :=
sorry

end harmonic_mean_pairs_l17_17791


namespace quadratic_root_ratio_l17_17983

theorem quadratic_root_ratio {m p q : ℝ} (h₁ : m ≠ 0) (h₂ : p ≠ 0) (h₃ : q ≠ 0)
  (h₄ : ∀ s₁ s₂ : ℝ, (s₁ + s₂ = -q ∧ s₁ * s₂ = m) →
    (∃ t₁ t₂ : ℝ, t₁ = 3 * s₁ ∧ t₂ = 3 * s₂ ∧ (t₁ + t₂ = -m ∧ t₁ * t₂ = p))) :
  p / q = 27 :=
by
  sorry

end quadratic_root_ratio_l17_17983


namespace sample_size_product_A_l17_17424

theorem sample_size_product_A 
  (ratio_A : ℕ)
  (ratio_B : ℕ)
  (ratio_C : ℕ)
  (total_ratio : ℕ)
  (sample_size : ℕ) 
  (h_ratio : ratio_A = 2 ∧ ratio_B = 3 ∧ ratio_C = 5)
  (h_total_ratio : total_ratio = ratio_A + ratio_B + ratio_C)
  (h_sample_size : sample_size = 80) :
  (80 * (ratio_A : ℚ) / total_ratio) = 16 :=
by
  sorry

end sample_size_product_A_l17_17424


namespace hyperbola_properties_l17_17216

theorem hyperbola_properties :
  let h := -3
  let k := 0
  let a := 5
  let c := Real.sqrt 50
  ∃ b : ℝ, a^2 + b^2 = c^2 ∧ h + k + a + b = 7 :=
by
  sorry

end hyperbola_properties_l17_17216


namespace range_for_a_l17_17671

noncomputable def line_not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (3 * a - 1) * x + (2 - a) * y - 1 = 0 → (x ≥ 0 ∨ y ≥ 0)

theorem range_for_a (a : ℝ) :
  (line_not_in_second_quadrant a) ↔ a ≥ 2 := by
  sorry

end range_for_a_l17_17671


namespace find_third_triangle_angles_l17_17347

-- Define the problem context
variables {A B C : ℝ} -- angles of the original triangle

-- Condition: The sum of the angles in a triangle is 180 degrees
axiom sum_of_angles (a b c : ℝ) : a + b + c = 180

-- Given conditions about the triangle and inscribed circles
def original_triangle (a b c : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 180

def inscribed_circle (a b c : ℝ) : Prop :=
original_triangle a b c

def second_triangle (a b c : ℝ) : Prop :=
inscribed_circle a b c

def third_triangle (a b c : ℝ) : Prop :=
second_triangle a b c

-- Goal: Prove that the angles in the third triangle are 60 degrees each
theorem find_third_triangle_angles (a b c : ℝ) (ha : original_triangle a b c)
  (h_inscribed : inscribed_circle a b c)
  (h_second : second_triangle a b c)
  (h_third : third_triangle a b c) : a = 60 ∧ b = 60 ∧ c = 60 := by
sorry

end find_third_triangle_angles_l17_17347


namespace closest_point_on_line_is_correct_l17_17059

theorem closest_point_on_line_is_correct :
  ∃ (p : ℝ × ℝ), p = (-0.04, -0.28) ∧
  ∃ x : ℝ, p = (x, (3 * x - 1) / 4) ∧
  ∀ q : ℝ × ℝ, (q = (x, (3 * x - 1) / 4) → 
  (dist (2, -3) p) ≤ (dist (2, -3) q)) :=
sorry

end closest_point_on_line_is_correct_l17_17059


namespace length_BE_l17_17575

-- Definitions and Conditions
def is_square (ABCD : Type) (side_length : ℝ) : Prop :=
  side_length = 2

def triangle_area (base : ℝ) (height : ℝ) : ℝ :=
  0.5 * base * height

def rectangle_area (length : ℝ) (width : ℝ) : ℝ :=
  length * width

-- Problem statement in Lean
theorem length_BE 
(ABCD : Type) (side_length : ℝ) 
(JKHG : Type) (BC : ℝ) (x : ℝ) 
(E : Type) (E_on_BC : E) 
(area_fact : rectangle_area BC x = 2 * triangle_area x BC) 
(h1 : is_square ABCD side_length) 
(h2 : BC = 2) : 
x = 1 :=
by {
  sorry
}

end length_BE_l17_17575


namespace problem_solution_l17_17002

theorem problem_solution (x : ℝ) (N : ℝ) (h1 : 625 ^ (-x) + N ^ (-2 * x) + 5 ^ (-4 * x) = 11) (h2 : x = 0.25) :
  N = 25 / 2809 :=
by
  sorry

end problem_solution_l17_17002


namespace circle_center_transformation_l17_17099

def original_center : ℤ × ℤ := (3, -4)

def reflect_x_axis (p : ℤ × ℤ) : ℤ × ℤ := (p.1, -p.2)

def translate_right (p : ℤ × ℤ) (d : ℤ) : ℤ × ℤ := (p.1 + d, p.2)

def final_center : ℤ × ℤ := (8, 4)

theorem circle_center_transformation :
  translate_right (reflect_x_axis original_center) 5 = final_center :=
by
  sorry

end circle_center_transformation_l17_17099


namespace average_of_first_13_even_numbers_l17_17940

-- Definition of the first 13 even numbers
def first_13_even_numbers := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]

-- The sum of the first 13 even numbers
def sum_of_first_13_even_numbers : ℕ := 182

-- The number of these even numbers
def number_of_even_numbers : ℕ := 13

-- The average of the first 13 even numbers
theorem average_of_first_13_even_numbers : (sum_of_first_13_even_numbers / number_of_even_numbers) = 14 := by
  sorry

end average_of_first_13_even_numbers_l17_17940


namespace union_of_sets_l17_17422

theorem union_of_sets (x y : ℕ) (A B : Set ℕ) (hA : A = {x, y}) (hB : B = {x + 1, 5}) (h_inter : A ∩ B = {2}) :
  A ∪ B = {1, 2, 5} :=
by
  sorry

end union_of_sets_l17_17422


namespace solve_eq1_solve_eq2_l17_17139

noncomputable def eq1_solution1 := -2 + Real.sqrt 5
noncomputable def eq1_solution2 := -2 - Real.sqrt 5

noncomputable def eq2_solution1 := 3
noncomputable def eq2_solution2 := 1

theorem solve_eq1 (x : ℝ) :
  x^2 + 4 * x - 1 = 0 → (x = eq1_solution1 ∨ x = eq1_solution2) :=
by
  sorry

theorem solve_eq2 (x : ℝ) :
  (x - 3)^2 + 2 * x * (x - 3) = 0 → (x = eq2_solution1 ∨ x = eq2_solution2) :=
by 
  sorry

end solve_eq1_solve_eq2_l17_17139


namespace subset_intersection_exists_l17_17026

theorem subset_intersection_exists {n : ℕ} (A : Fin (n + 1) → Finset (Fin n)) 
    (h_distinct : ∀ i j : Fin (n + 1), i ≠ j → A i ≠ A j)
    (h_size : ∀ i : Fin (n + 1), (A i).card = 3) : 
    ∃ (i j : Fin (n + 1)), i ≠ j ∧ (A i ∩ A j).card = 1 :=
by
  sorry

end subset_intersection_exists_l17_17026


namespace good_fractions_expression_l17_17842

def is_good_fraction (n : ℕ) (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = n

theorem good_fractions_expression (n : ℕ) (a b : ℕ) :
  n > 1 →
  (∀ a b, b < n → is_good_fraction n a b → ∃ x y, x + y = a / b ∨ x - y = a / b) ↔
  Nat.Prime n :=
by
  sorry

end good_fractions_expression_l17_17842


namespace watermelon_weight_l17_17920

theorem watermelon_weight (B W : ℝ) (n : ℝ) 
  (h1 : B + n * W = 63) 
  (h2 : B + (n / 2) * W = 34) : 
  n * W = 58 :=
sorry

end watermelon_weight_l17_17920


namespace initial_albums_in_cart_l17_17480

theorem initial_albums_in_cart (total_songs : ℕ) (songs_per_album : ℕ) (removed_albums : ℕ) 
  (h_total: total_songs = 42) 
  (h_songs_per_album: songs_per_album = 7)
  (h_removed: removed_albums = 2): 
  (total_songs / songs_per_album) + removed_albums = 8 := 
by
  sorry

end initial_albums_in_cart_l17_17480


namespace percentage_of_women_in_study_group_l17_17165

variable (W : ℝ) -- W is the percentage of women in the study group in decimal form

-- Given conditions as hypotheses
axiom h1 : 0 < W ∧ W <= 1         -- W represents a percentage, so it must be between 0 and 1.
axiom h2 : 0.40 * W = 0.28         -- 40 percent of women are lawyers, and the probability of selecting a woman lawyer is 0.28.

-- The statement to prove
theorem percentage_of_women_in_study_group : W = 0.7 :=
by
  sorry

end percentage_of_women_in_study_group_l17_17165


namespace bob_total_profit_l17_17109

/-- Define the cost of each dog --/
def dog_cost : ℝ := 250.0

/-- Define the number of dogs Bob bought --/
def number_of_dogs : ℕ := 2

/-- Define the total cost of the dogs --/
def total_cost_for_dogs : ℝ := dog_cost * number_of_dogs

/-- Define the selling price of each puppy --/
def puppy_selling_price : ℝ := 350.0

/-- Define the number of puppies --/
def number_of_puppies : ℕ := 6

/-- Define the total revenue from selling the puppies --/
def total_revenue_from_puppies : ℝ := puppy_selling_price * number_of_puppies

/-- Define Bob's total profit from selling the puppies --/
def total_profit : ℝ := total_revenue_from_puppies - total_cost_for_dogs

/-- The theorem stating that Bob's total profit is $1600.00 --/
theorem bob_total_profit : total_profit = 1600.0 := 
by
  /- We leave the proof out as we just need the statement -/
  sorry

end bob_total_profit_l17_17109


namespace sector_central_angle_l17_17772

-- Definitions and constants
def arc_length := 4 -- arc length of the sector in cm
def area := 2       -- area of the sector in cm²

-- The central angle of the sector we want to prove
def theta := 4      -- radian measure of the central angle

-- Main statement to prove
theorem sector_central_angle : 
  ∃ (r : ℝ), (1 / 2) * theta * r^2 = area ∧ theta * r = arc_length :=
by
  -- No proof is required as per the instruction
  sorry

end sector_central_angle_l17_17772


namespace find_a_l17_17430

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(-x) * (1 - a^x)

theorem find_a (a : ℝ) : 
  (∀ x : ℝ, f a (-x) = -f a x) ∧ a > 0 ∧ a ≠ 1 → a = 4 :=
by
  sorry

end find_a_l17_17430


namespace common_difference_is_3_l17_17476

variables {a : ℕ → ℝ} {d a1 : ℝ}

-- Define the arithmetic sequence
def arithmetic_sequence (a_n : ℕ → ℝ) (a1 d : ℝ) : Prop := 
  ∀ n, a_n n = a1 + (n - 1) * d

-- Conditions
def a2_eq : a 2 = 3 := sorry
def a5_eq : a 5 = 12 := sorry

-- Theorem to prove the common difference is 3
theorem common_difference_is_3 :
  ∀ {a : ℕ → ℝ} {a1 d : ℝ},
  (arithmetic_sequence a a1 d)
  → a 2 = 3 
  → a 5 = 12 
  → d = 3 :=
  by
  intros a a1 d h_seq h_a2 h_a5
  sorry

end common_difference_is_3_l17_17476


namespace expression_value_l17_17466

theorem expression_value :
  (2^1006 + 5^1007)^2 - (2^1006 - 5^1007)^2 = 40 * 10^1006 :=
by sorry

end expression_value_l17_17466


namespace smallest_x_l17_17103

theorem smallest_x : ∃ x : ℕ, x + 6721 ≡ 3458 [MOD 12] ∧ x % 5 = 0 ∧ x = 45 :=
by
  sorry

end smallest_x_l17_17103


namespace total_matches_l17_17025

theorem total_matches (home_wins home_draws home_losses rival_wins rival_draws rival_losses : ℕ)
  (H_home_wins : home_wins = 3)
  (H_home_draws : home_draws = 4)
  (H_home_losses : home_losses = 0)
  (H_rival_wins : rival_wins = 2 * home_wins)
  (H_rival_draws : rival_draws = 4)
  (H_rival_losses : rival_losses = 0) :
  home_wins + home_draws + home_losses + rival_wins + rival_draws + rival_losses = 17 :=
by
  sorry

end total_matches_l17_17025


namespace sum_f_sequence_l17_17766

noncomputable def f (x : ℝ) : ℝ := 1 / (4^x + 2)

theorem sum_f_sequence :
  f (1/10) + f (2/10) + f (3/10) + f (4/10) + f (5/10) + f (6/10) + f (7/10) + f (8/10) + f (9/10) = 9 / 4 :=
by {
  sorry
}

end sum_f_sequence_l17_17766


namespace lcm_of_two_numbers_hcf_and_product_l17_17114

theorem lcm_of_two_numbers_hcf_and_product (a b : ℕ) (h_hcf : Nat.gcd a b = 20) (h_prod : a * b = 2560) :
  Nat.lcm a b = 128 :=
by
  sorry

end lcm_of_two_numbers_hcf_and_product_l17_17114


namespace sqrt_eq_pm_4_l17_17658

theorem sqrt_eq_pm_4 : {x : ℝ | x * x = 16} = {4, -4} :=
by sorry

end sqrt_eq_pm_4_l17_17658


namespace largest_prime_factor_of_891_l17_17421

theorem largest_prime_factor_of_891 : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ 891 ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ 891 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_891_l17_17421


namespace find_AB_l17_17926

variables {AB CD AD BC AP PD APD PQ Q: ℝ}

def is_rectangle (ABCD : Prop) := ABCD

variables (P_on_BC : Prop)
variable (BP CP: ℝ)
variable (tan_angle_APD: ℝ)

theorem find_AB (ABCD : Prop) (P_on_BC : Prop) (BP CP: ℝ) (tan_angle_APD: ℝ) : 
  is_rectangle ABCD →
  P_on_BC →
  BP = 24 →
  CP = 12 →
  tan_angle_APD = 2 →
  AB = 27 := 
by
  sorry

end find_AB_l17_17926


namespace point_B_in_fourth_quadrant_l17_17493

theorem point_B_in_fourth_quadrant (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : (b > 0 ∧ a < 0) :=
by {
    sorry
}

end point_B_in_fourth_quadrant_l17_17493


namespace find_f1_increasing_on_positive_solve_inequality_l17_17310

-- Given conditions
axiom f : ℝ → ℝ
axiom domain : ∀ x, 0 < x → true
axiom f4 : f 4 = 1
axiom multiplicative : ∀ x y, 0 < x → 0 < y → f (x * y) = f x + f y
axiom less_than_zero : ∀ x, 0 < x ∧ x < 1 → f x < 0

-- Required proofs
theorem find_f1 : f 1 = 0 := sorry

theorem increasing_on_positive : ∀ x y, 0 < x → 0 < y → x < y → f x < f y := sorry

theorem solve_inequality : {x : ℝ // 3 < x ∧ x ≤ 5} := sorry

end find_f1_increasing_on_positive_solve_inequality_l17_17310


namespace value_2x_y_l17_17836

theorem value_2x_y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y + 5 = 0) : 2*x + y = 0 := 
by
  sorry

end value_2x_y_l17_17836


namespace percentage_carnations_l17_17263

variable (F : ℕ)
variable (H1 : F ≠ 0) -- Non-zero flowers
variable (H2 : ∀ (y : ℕ), 5 * y = F → 2 * y ≠ 0) -- Two fifths of the pink flowers are roses.
variable (H3 : ∀ (z : ℕ), 7 * z = 3 * (F - F / 2 - F / 5) → 6 * z ≠ 0) -- Six sevenths of the red flowers are carnations.
variable (H4 : ∀ (w : ℕ), 5 * w = F → w ≠ 0) -- One fifth of the flowers are yellow tulips.
variable (H5 : 2 * F / 2 = F) -- Half of the flowers are pink.
variable (H6 : ∀ (c : ℕ), 10 * c = F → c ≠ 0) -- Total flowers in multiple of 10

theorem percentage_carnations :
  (exists (pc rc : ℕ), 70 * (pc + rc) = 55 * F) :=
sorry

end percentage_carnations_l17_17263


namespace Prudence_sleep_weeks_l17_17944

def Prudence_sleep_per_week : Nat := 
  let nights_sleep_weekday := 6
  let nights_sleep_weekend := 9
  let weekday_nights := 5
  let weekend_nights := 2
  let naps := 1
  let naps_days := 2
  weekday_nights * nights_sleep_weekday + weekend_nights * nights_sleep_weekend + naps_days * naps

theorem Prudence_sleep_weeks (w : Nat) (h : w * Prudence_sleep_per_week = 200) : w = 4 :=
by
  sorry

end Prudence_sleep_weeks_l17_17944


namespace geometric_arithmetic_sum_l17_17517

theorem geometric_arithmetic_sum {a : Nat → ℝ} {b : Nat → ℝ} 
  (h_geo : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n)
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d)
  (h_condition : a 3 * a 11 = 4 * a 7)
  (h_equal : a 7 = b 7) :
  b 5 + b 9 = 8 :=
sorry

end geometric_arithmetic_sum_l17_17517


namespace quadratic_distinct_real_roots_range_l17_17632

theorem quadratic_distinct_real_roots_range (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 2 * x - 1 = 0 ∧ ∃ y : ℝ, y ≠ x ∧ k * y^2 - 2 * y - 1 = 0) ↔ (k > -1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_distinct_real_roots_range_l17_17632


namespace clock_angle_solution_l17_17663

theorem clock_angle_solution (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 360) :
    (θ = 15) ∨ (θ = 165) :=
by
  sorry

end clock_angle_solution_l17_17663


namespace log_ab_eq_l17_17473

-- Definition and conditions
variables (a b x : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hx : 0 < x)

-- The theorem to prove
theorem log_ab_eq (a b x : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) :
  Real.log (x) / Real.log (a * b) = (Real.log (x) / Real.log (a)) * (Real.log (x) / Real.log (b)) / ((Real.log (x) / Real.log (a)) + (Real.log (x) / Real.log (b))) :=
sorry

end log_ab_eq_l17_17473


namespace problem_statement_l17_17573

noncomputable def g (x : ℝ) : ℝ := 3^(x + 1)

theorem problem_statement (x : ℝ) : g (x + 1) - 2 * g x = g x := by
  -- The proof here is omitted
  sorry

end problem_statement_l17_17573


namespace sequence_n_500_l17_17049

theorem sequence_n_500 (a : ℕ → ℤ) 
  (h1 : a 1 = 1010) 
  (h2 : a 2 = 1011) 
  (h3 : ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = 2 * n + 3) : 
  a 500 = 3003 := 
sorry

end sequence_n_500_l17_17049


namespace simplify_expression_l17_17205

noncomputable def a : ℝ := 2 * Real.sqrt 12 - 4 * Real.sqrt 27 + 3 * Real.sqrt 75 + 7 * Real.sqrt 8 - 3 * Real.sqrt 18
noncomputable def b : ℝ := 4 * Real.sqrt 48 - 3 * Real.sqrt 27 - 5 * Real.sqrt 18 + 2 * Real.sqrt 50

theorem simplify_expression : a * b = 97 := by
  sorry

end simplify_expression_l17_17205


namespace find_angle_C_find_area_of_triangle_l17_17684

-- Given triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively
-- And given conditions: c * cos B = (2a - b) * cos C

variable (a b c : ℝ) (A B C : ℝ)
variable (h1 : c * Real.cos B = (2 * a - b) * Real.cos C)
variable (h2 : c = 2)
variable (h3 : a + b + c = 2 * Real.sqrt 3 + 2)

-- Prove that angle C = π / 3
theorem find_angle_C : C = Real.pi / 3 :=
by sorry

-- Given angle C, side c, and perimeter, prove the area of triangle ABC
theorem find_area_of_triangle (h4 : C = Real.pi / 3) : 
  1 / 2 * a * b * Real.sin C = 2 * Real.sqrt 3 / 3 :=
by sorry

end find_angle_C_find_area_of_triangle_l17_17684


namespace ice_cream_melt_time_l17_17411

theorem ice_cream_melt_time :
  let blocks := 16
  let block_length := 1.0/8.0 -- miles per block
  let distance := blocks * block_length -- in miles
  let speed := 12.0 -- miles per hour
  let time := distance / speed -- in hours
  let time_in_minutes := time * 60 -- converted to minutes
  time_in_minutes = 10 := by sorry

end ice_cream_melt_time_l17_17411


namespace factory_Y_bulbs_proportion_l17_17916

theorem factory_Y_bulbs_proportion :
  (0.60 * 0.59 + 0.40 * P_Y = 0.62) → (P_Y = 0.665) :=
by
  sorry

end factory_Y_bulbs_proportion_l17_17916


namespace simplify_expression_l17_17579

theorem simplify_expression (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) (h4 : x ≠ 5):
  ((x^2 - 4 * x + 3) / (x^2 - 6 * x + 9)) / ((x^2 - 6 * x + 8) / (x^2 - 8 * x + 15)) = 
  (x - 1) * (x - 5) / ((x - 3) * (x - 4) * (x - 2)) :=
sorry

end simplify_expression_l17_17579


namespace units_digit_is_six_l17_17472

theorem units_digit_is_six (n : ℤ) (h : (n^2 / 10 % 10) = 7) : (n^2 % 10) = 6 :=
by sorry

end units_digit_is_six_l17_17472


namespace value_of_4_Y_3_l17_17273

def Y (a b : ℕ) : ℕ := (2 * a ^ 2 - 3 * a * b + b ^ 2) ^ 2

theorem value_of_4_Y_3 : Y 4 3 = 25 := by
  sorry

end value_of_4_Y_3_l17_17273


namespace necessary_but_not_sufficient_condition_l17_17238

-- Let p be the proposition |x| < 2
def p (x : ℝ) : Prop := abs x < 2

-- Let q be the proposition x^2 - x - 2 < 0
def q (x : ℝ) : Prop := x^2 - x - 2 < 0

-- The proof statement
theorem necessary_but_not_sufficient_condition (x : ℝ) : q x → p x ∧ ¬ (p x → q x) := 
sorry

end necessary_but_not_sufficient_condition_l17_17238


namespace maintain_income_with_new_demand_l17_17045

variable (P D : ℝ) -- Original Price and Demand
def new_price := 1.20 * P -- New Price after 20% increase
def new_demand := 1.12 * D -- New Demand after 12% increase due to advertisement
def original_income := P * D -- Original income
def new_income := new_price * new_demand -- New income after changes

theorem maintain_income_with_new_demand :
  ∀ P D : ℝ, P * D = 1.20 * P * 1.12 * (D_new : ℝ) → (D_new = 14/15 * D) :=
by
  intro P D h
  sorry

end maintain_income_with_new_demand_l17_17045


namespace people_per_car_l17_17906

theorem people_per_car (total_people : ℝ) (total_cars : ℝ) (h1 : total_people = 189) (h2 : total_cars = 3.0) : total_people / total_cars = 63 := 
by
  sorry

end people_per_car_l17_17906


namespace sasha_prediction_l17_17236

theorem sasha_prediction (n : ℕ) 
  (white_rook_students : ℕ)
  (black_elephant_students : ℕ)
  (total_games : ℕ) :
  white_rook_students = 15 → 
  black_elephant_students = 20 → 
  total_games = 300 → 
  n = 280 → 
  ∃ s : ℕ, s ≤ white_rook_students ∧ s ≤ black_elephant_students ∧ s * black_elephant_students ≥ total_games - n :=
by
  sorry

end sasha_prediction_l17_17236


namespace average_speed_l17_17423

theorem average_speed (x y : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y)
  (total_time : x / 4 + y / 3 + y / 6 + x / 4 = 5) :
  (2 * (x + y)) / 5 = 4 :=
by
  sorry

end average_speed_l17_17423


namespace undefined_expression_real_val_l17_17938

theorem undefined_expression_real_val (a : ℝ) :
  a = 2 → (a^3 - 8 = 0) :=
by
  intros
  sorry

end undefined_expression_real_val_l17_17938


namespace solve_system_of_equations_l17_17901

theorem solve_system_of_equations (x y m : ℝ) 
  (h1 : 2 * x + y = 7)
  (h2 : x + 2 * y = m - 3) 
  (h3 : x - y = 2) : m = 8 :=
by
  -- Proof part is replaced with sorry as mentioned
  sorry

end solve_system_of_equations_l17_17901


namespace sum_of_digits_of_A15B94_multiple_of_99_l17_17392

theorem sum_of_digits_of_A15B94_multiple_of_99 (A B : ℕ) 
  (hA : A < 10) (hB : B < 10)
  (h_mult_99 : ∃ n : ℕ, (100000 * A + 10000 + 5000 + 100 * B + 90 + 4) = 99 * n) :
  A + B = 8 := 
by
  sorry

end sum_of_digits_of_A15B94_multiple_of_99_l17_17392


namespace anton_has_more_cards_than_ann_l17_17087

-- Define Heike's number of cards
def heike_cards : ℕ := 60

-- Define Anton's number of cards in terms of Heike's cards
def anton_cards (H : ℕ) : ℕ := 3 * H

-- Define Ann's number of cards as equal to Heike's cards
def ann_cards (H : ℕ) : ℕ := H

-- Theorem statement
theorem anton_has_more_cards_than_ann 
  (H : ℕ) (H_equals : H = heike_cards) : 
  anton_cards H - ann_cards H = 120 :=
by
  -- At this point, the actual proof would be inserted.
  sorry

end anton_has_more_cards_than_ann_l17_17087


namespace participants_won_more_than_lost_l17_17283

-- Define the conditions given in the problem
def total_participants := 64
def rounds := 6

-- Define a function that calculates the number of participants reaching a given round
def participants_after_round (n : Nat) (r : Nat) : Nat :=
  n / (2 ^ r)

-- The theorem we need to prove
theorem participants_won_more_than_lost :
  participants_after_round total_participants 2 = 16 :=
by 
  -- Provide a placeholder for the proof
  sorry

end participants_won_more_than_lost_l17_17283


namespace find_original_number_l17_17105

-- Let x be the original number
def maria_operations (x : ℤ) : Prop :=
  (3 * (x - 3) + 3) / 3 = 10

theorem find_original_number (x : ℤ) (h : maria_operations x) : x = 12 :=
by
  sorry

end find_original_number_l17_17105


namespace min_ab_min_inv_a_plus_2_inv_b_max_sqrt_2a_plus_sqrt_b_not_max_a_plus_1_times_b_plus_1_l17_17053

-- Condition definitions
variable {a b : ℝ}
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1)

-- Minimum value of ab is 1/8
theorem min_ab (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = (a * b) ∧ y = 1 / 8 := by
  sorry

-- Minimum value of 1/a + 2/b is 8
theorem min_inv_a_plus_2_inv_b (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = (1 / a + 2 / b) ∧ y = 8 := by
  sorry

-- Maximum value of sqrt(2a) + sqrt(b) is sqrt(2)
theorem max_sqrt_2a_plus_sqrt_b (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = (Real.sqrt (2 * a) + Real.sqrt b) ∧ y = Real.sqrt 2 := by
  sorry

-- Maximum value of (a+1)(b+1) is not 2
theorem not_max_a_plus_1_times_b_plus_1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = ((a + 1) * (b + 1)) ∧ y ≠ 2 := by
  sorry


end min_ab_min_inv_a_plus_2_inv_b_max_sqrt_2a_plus_sqrt_b_not_max_a_plus_1_times_b_plus_1_l17_17053


namespace find_minimum_value_2a_plus_b_l17_17951

theorem find_minimum_value_2a_plus_b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_re_z : (3 * a * b + 2) = 4) : 2 * a + b = (4 * Real.sqrt 3) / 3 :=
sorry

end find_minimum_value_2a_plus_b_l17_17951


namespace sum_of_cubes_l17_17855

theorem sum_of_cubes
  (a b c : ℝ)
  (h₁ : a + b + c = 7)
  (h₂ : ab + ac + bc = 9)
  (h₃ : a * b * c = -18) :
  a^3 + b^3 + c^3 = 100 := by
  sorry

end sum_of_cubes_l17_17855


namespace tangent_line_m_value_l17_17167

theorem tangent_line_m_value : 
  (∀ m : ℝ, ∃ (x y : ℝ), (x = my + 2) ∧ (x + one)^2 + (y + one)^2 = 2) → 
  (m = 1 ∨ m = -7) :=
  sorry

end tangent_line_m_value_l17_17167


namespace system_has_integer_solution_l17_17922

theorem system_has_integer_solution (a b : ℤ) : 
  ∃ x y z t : ℤ, x + y + 2 * z + 2 * t = a ∧ 2 * x - 2 * y + z - t = b :=
by
  sorry

end system_has_integer_solution_l17_17922


namespace upper_limit_b_l17_17247

theorem upper_limit_b (a b : ℤ) (h1 : 6 < a) (h2 : a < 17) (h3 : 3 < b) (h4 : (a : ℚ) / b ≤ 3.75) : b ≤ 4 := by
  sorry

end upper_limit_b_l17_17247


namespace degree_of_minus_5x4y_l17_17368

def degree_of_monomial (coeff : Int) (x_exp y_exp : Nat) : Nat :=
  x_exp + y_exp

theorem degree_of_minus_5x4y : degree_of_monomial (-5) 4 1 = 5 :=
by
  sorry

end degree_of_minus_5x4y_l17_17368


namespace g_1200_value_l17_17009

noncomputable def g : ℝ → ℝ := sorry

-- Assume the given condition as a definition
axiom functional_eq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y

-- Assume the given value of g(1000)
axiom g_1000_value : g 1000 = 4

-- Prove that g(1200) = 10/3
theorem g_1200_value : g 1200 = 10 / 3 := by
  sorry

end g_1200_value_l17_17009


namespace john_spent_fraction_at_arcade_l17_17603

theorem john_spent_fraction_at_arcade 
  (allowance : ℝ) (spent_arcade : ℝ) (spent_candy_store : ℝ) 
  (h1 : allowance = 3.45)
  (h2 : spent_candy_store = 0.92)
  (h3 : 3.45 - spent_arcade - (1/3) * (3.45 - spent_arcade) = spent_candy_store) :
  spent_arcade / allowance = 2.07 / 3.45 :=
by
  sorry

end john_spent_fraction_at_arcade_l17_17603


namespace quadratic_equation_original_eq_l17_17682

theorem quadratic_equation_original_eq :
  ∃ (α β : ℝ), (α + β = 3) ∧ (α * β = -6) ∧ (∀ (x : ℝ), x^2 - 3 * x - 6 = 0 → (x = α ∨ x = β)) :=
sorry

end quadratic_equation_original_eq_l17_17682


namespace numPeopleToLeftOfKolya_l17_17001

-- Definitions based on the conditions.
def peopleToRightOfKolya := 12
def peopleToLeftOfSasha := 20
def peopleToRightOfSasha := 8

-- Theorem statement with the given conditions and conclusion.
theorem numPeopleToLeftOfKolya 
  (h1 : peopleToRightOfKolya = 12)
  (h2 : peopleToLeftOfSasha = 20)
  (h3 : peopleToRightOfSasha = 8) :
  ∃ n, n = 16 :=
by
  -- Proving the theorem will be done here.
  sorry

end numPeopleToLeftOfKolya_l17_17001


namespace students_with_grade_B_and_above_l17_17514

theorem students_with_grade_B_and_above (total_students : ℕ) (percent_below_B : ℕ) 
(h1 : total_students = 60) (h2 : percent_below_B = 40) : 
(total_students * (100 - percent_below_B) / 100) = 36 := by
  sorry

end students_with_grade_B_and_above_l17_17514


namespace gcd_polynomials_l17_17617

theorem gcd_polynomials (b : ℤ) (h : b % 8213 = 0 ∧ b % 2 = 1) :
  Int.gcd (8 * b^2 + 63 * b + 144) (2 * b + 15) = 9 :=
sorry

end gcd_polynomials_l17_17617


namespace train_length_l17_17352

theorem train_length (speed_kph : ℝ) (time_sec : ℝ) (speed_mps : ℝ) (length_m : ℝ) 
  (h1 : speed_kph = 60) 
  (h2 : time_sec = 42) 
  (h3 : speed_mps = speed_kph * 1000 / 3600) 
  (h4 : length_m = speed_mps * time_sec) :
  length_m = 700.14 :=
by
  sorry

end train_length_l17_17352


namespace lab_tech_items_l17_17972

theorem lab_tech_items (num_uniforms : ℕ) (num_coats : ℕ) (num_techs : ℕ) (total_items : ℕ)
  (h_uniforms : num_uniforms = 12)
  (h_coats : num_coats = 6 * num_uniforms)
  (h_techs : num_techs = num_uniforms / 2)
  (h_total : total_items = num_coats + num_uniforms) :
  total_items / num_techs = 14 :=
by
  -- Placeholder for proof, ensuring theorem builds correctly.
  sorry

end lab_tech_items_l17_17972


namespace quadratic_has_solution_zero_l17_17389

theorem quadratic_has_solution_zero (k : ℝ) : 
  (∃ x : ℝ, (k - 2) * x^2 + 3 * x + k^2 - 4 = 0) →
  ((k - 2) ≠ 0) → k = -2 := 
by 
  sorry

end quadratic_has_solution_zero_l17_17389


namespace larger_number_is_eight_l17_17144

variable {x y : ℝ}

theorem larger_number_is_eight (h1 : x - y = 3) (h2 : x^2 - y^2 = 39) : x = 8 :=
by
  sorry

end larger_number_is_eight_l17_17144


namespace x_varies_z_pow_l17_17651

variable (k j : ℝ)
variable (y z : ℝ)

-- Given conditions
def x_varies_y_squared (x : ℝ) := x = k * y^2
def y_varies_z_cuberoot_squared := y = j * z^(2/3)

-- To prove: 
theorem x_varies_z_pow (x : ℝ) (h1 : x_varies_y_squared k y x) (h2 : y_varies_z_cuberoot_squared j z y) : ∃ m : ℝ, x = m * z^(4/3) :=
by
  sorry

end x_varies_z_pow_l17_17651


namespace find_A_l17_17996

-- Define the polynomial and the partial fraction decomposition equation
def polynomial (x : ℝ) : ℝ := x^3 - 3 * x^2 - 13 * x + 15

theorem find_A (A B C : ℝ) (h : ∀ x : ℝ, 1 / polynomial x = A / (x + 3) + B / (x - 1) + C / (x - 1)^2) : 
  A = 1 / 16 :=
sorry

end find_A_l17_17996


namespace compounded_rate_of_growth_l17_17736

theorem compounded_rate_of_growth (k m : ℝ) :
  (1 + k / 100) * (1 + m / 100) - 1 = ((k + m + (k * m / 100)) / 100) :=
by
  sorry

end compounded_rate_of_growth_l17_17736


namespace unique_positive_real_solution_l17_17448

theorem unique_positive_real_solution (x : ℝ) (hx_pos : x > 0) (h_eq : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_real_solution_l17_17448


namespace x1_x2_in_M_l17_17562

-- Definitions of the set M and the condition x ∈ M
def M : Set ℕ := { x | ∃ a b : ℤ, x = a^2 + b^2 }

-- Statement of the problem
theorem x1_x2_in_M (x1 x2 : ℕ) (h1 : x1 ∈ M) (h2 : x2 ∈ M) : (x1 * x2) ∈ M :=
sorry

end x1_x2_in_M_l17_17562


namespace P_eq_CU_M_union_CU_N_l17_17540

open Set

-- Definitions of U, M, N
def U : Set (ℝ × ℝ) := { p | True }
def M : Set (ℝ × ℝ) := { p | p.2 ≠ p.1 }
def N : Set (ℝ × ℝ) := { p | p.2 ≠ -p.1 }
def CU_M : Set (ℝ × ℝ) := { p | p.2 = p.1 }
def CU_N : Set (ℝ × ℝ) := { p | p.2 = -p.1 }

-- Theorem statement
theorem P_eq_CU_M_union_CU_N :
  { p : ℝ × ℝ | p.2^2 ≠ p.1^2 } = CU_M ∪ CU_N :=
sorry

end P_eq_CU_M_union_CU_N_l17_17540


namespace complement_A_possible_set_l17_17731

variable (U A B : Set ℕ)

theorem complement_A_possible_set (hU : U = {1, 2, 3, 4, 5, 6})
  (h_union : A ∪ B = {1, 2, 3, 4, 5}) 
  (h_inter : A ∩ B = {3, 4, 5}) :
  ∃ C, C = U \ A ∧ C = {6} :=
by
  sorry

end complement_A_possible_set_l17_17731


namespace polynomial_root_multiplicity_l17_17751

theorem polynomial_root_multiplicity (A B n : ℤ) (h1 : A + B + 1 = 0) (h2 : (n + 1) * A + n * B = 0) :
  A = n ∧ B = -(n + 1) :=
sorry

end polynomial_root_multiplicity_l17_17751


namespace num_positive_four_digit_integers_of_form_xx75_l17_17393

theorem num_positive_four_digit_integers_of_form_xx75 : 
  ∃ n : ℕ, n = 90 ∧ ∀ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → (∃ x: ℕ, x = 1000 * a + 100 * b + 75 ∧ 1000 ≤ x ∧ x < 10000) → n = 90 :=
sorry

end num_positive_four_digit_integers_of_form_xx75_l17_17393


namespace amount_a_put_in_correct_l17_17469

noncomputable def amount_a_put_in (total_profit managing_fee total_received_by_a profit_remaining: ℝ) : ℝ :=
  let capital_b := 2500
  let a_receives_from_investment := total_received_by_a - managing_fee
  let profit_ratio := a_receives_from_investment / profit_remaining
  profit_ratio * capital_b

theorem amount_a_put_in_correct :
  amount_a_put_in 9600 960 6000 8640 = 3500 :=
by
  dsimp [amount_a_put_in]
  sorry

end amount_a_put_in_correct_l17_17469


namespace area_product_is_2_l17_17747

open Real

-- Definitions for parabola, points, and the condition of dot product
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 * B.1 + A.2 * B.2) = -4

def area (O F P : ℝ × ℝ) : ℝ :=
  0.5 * abs (O.1 * (F.2 - P.2) + F.1 * (P.2 - O.2) + P.1 * (O.2 - F.2))

-- Points A and B are on the parabola and the dot product condition holds
variables (A B : ℝ × ℝ)
variable (H_A_on_parabola : parabola A.1 A.2)
variable (H_B_on_parabola : parabola B.1 B.2)
variable (H_dot_product : dot_product_condition A B)

-- Focus of the parabola
def F : ℝ × ℝ := (1, 0)

-- Origin
def O : ℝ × ℝ := (0, 0)

-- Prove that the product of areas is 2
theorem area_product_is_2 : 
  area O F A * area O F B = 2 :=
sorry

end area_product_is_2_l17_17747


namespace Marge_savings_l17_17741

theorem Marge_savings
  (lottery_winnings : ℝ)
  (taxes_paid : ℝ)
  (student_loan_payment : ℝ)
  (amount_after_taxes : ℝ)
  (amount_after_student_loans : ℝ)
  (fun_money : ℝ)
  (investment : ℝ)
  (savings : ℝ)
  (h_win : lottery_winnings = 12006)
  (h_tax : taxes_paid = lottery_winnings / 2)
  (h_after_tax : amount_after_taxes = lottery_winnings - taxes_paid)
  (h_loans : student_loan_payment = amount_after_taxes / 3)
  (h_after_loans : amount_after_student_loans = amount_after_taxes - student_loan_payment)
  (h_fun : fun_money = 2802)
  (h_savings_investment : amount_after_student_loans - fun_money = savings + investment)
  (h_investment : investment = savings / 5)
  (h_left : amount_after_student_loans - fun_money = 1200) :
  savings = 1000 :=
by
  sorry

end Marge_savings_l17_17741


namespace triangle_ineq_l17_17935

theorem triangle_ineq
  (a b c : ℝ)
  (triangle_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (triangle_ineq : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
by
  sorry

end triangle_ineq_l17_17935


namespace vanessa_missed_days_l17_17861

theorem vanessa_missed_days (V M S : ℕ) 
                           (h1 : V + M + S = 17) 
                           (h2 : V + M = 14) 
                           (h3 : M + S = 12) : 
                           V = 5 :=
sorry

end vanessa_missed_days_l17_17861


namespace decagon_adjacent_probability_l17_17998

noncomputable def probability_adjacent_vertices (total_vertices : ℕ) (adjacent_vertices : ℕ) : ℚ :=
adjacent_vertices / (total_vertices - 1)

theorem decagon_adjacent_probability :
  probability_adjacent_vertices 10 2 = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l17_17998


namespace C_share_correct_l17_17535

def investment_A := 27000
def investment_B := 72000
def investment_C := 81000
def total_profit := 80000

def gcd_investment : ℕ := Nat.gcd investment_A (Nat.gcd investment_B investment_C)
def ratio_A : ℕ := investment_A / gcd_investment
def ratio_B : ℕ := investment_B / gcd_investment
def ratio_C : ℕ := investment_C / gcd_investment
def total_parts : ℕ := ratio_A + ratio_B + ratio_C

def C_share : ℕ := (ratio_C / total_parts) * total_profit

theorem C_share_correct : C_share = 36000 := 
by sorry

end C_share_correct_l17_17535


namespace minimum_value_of_nS_n_l17_17790

noncomputable def a₁ (d : ℝ) : ℝ := -9/2 * d

noncomputable def S (n : ℕ) (d : ℝ) : ℝ :=
  n / 2 * (2 * a₁ d + (n - 1) * d)

theorem minimum_value_of_nS_n :
  S 10 (2/3) = 0 → S 15 (2/3) = 25 → ∃ (n : ℕ), (n * S n (2/3)) = -48 :=
by 
  intros h10 h15
  sorry

end minimum_value_of_nS_n_l17_17790


namespace intersection_height_correct_l17_17746

noncomputable def height_of_intersection (height1 height2 distance : ℝ) : ℝ :=
  let line1 (x : ℝ) := - (height1 / distance) * x + height1
  let line2 (x : ℝ) := - (height2 / distance) * x
  let x_intersect := - (height2 * distance) / (height1 - height2)
  line1 x_intersect

theorem intersection_height_correct :
  height_of_intersection 40 60 120 = 120 :=
by
  sorry

end intersection_height_correct_l17_17746


namespace correct_sequence_is_A_l17_17370

def Step := String
def Sequence := List Step

def correct_sequence : Sequence :=
  ["Buy a ticket", "Wait for the train", "Check the ticket", "Board the train"]

def option_A : Sequence :=
  ["Buy a ticket", "Wait for the train", "Check the ticket", "Board the train"]
def option_B : Sequence :=
  ["Wait for the train", "Buy a ticket", "Board the train", "Check the ticket"]
def option_C : Sequence :=
  ["Buy a ticket", "Wait for the train", "Board the train", "Check the ticket"]
def option_D : Sequence :=
  ["Repair the train", "Buy a ticket", "Check the ticket", "Board the train"]

theorem correct_sequence_is_A :
  correct_sequence = option_A :=
sorry

end correct_sequence_is_A_l17_17370


namespace intersection_of_A_and_B_l17_17291

open Set

-- Definition of set A
def A : Set ℤ := {1, 2, 3}

-- Definition of set B
def B : Set ℤ := {x | x < -1 ∨ 0 < x ∧ x < 2}

-- The theorem to prove A ∩ B = {1}
theorem intersection_of_A_and_B : A ∩ B = {1} := by
  -- Proof logic here
  sorry

end intersection_of_A_and_B_l17_17291


namespace mechanic_hourly_rate_l17_17353

-- Definitions and conditions
def total_bill : ℕ := 450
def parts_charge : ℕ := 225
def hours_worked : ℕ := 5

-- The main theorem to prove
theorem mechanic_hourly_rate : (total_bill - parts_charge) / hours_worked = 45 := by
  sorry

end mechanic_hourly_rate_l17_17353


namespace probability_stopping_in_C_l17_17911

noncomputable def probability_C : ℚ :=
  let P_A := 1 / 5
  let P_B := 1 / 5
  let x := (1 - (P_A + P_B)) / 3
  x

theorem probability_stopping_in_C :
  probability_C = 1 / 5 :=
by
  unfold probability_C
  sorry

end probability_stopping_in_C_l17_17911


namespace closest_to_2010_l17_17697

theorem closest_to_2010 :
  let A := 2008 * 2012
  let B := 1000 * Real.pi
  let C := 58 * 42
  let D := (48.3 ^ 2 - 2 * 8.3 * 48.3 + 8.3 ^ 2)
  abs (2010 - D) < abs (2010 - A) ∧
  abs (2010 - D) < abs (2010 - B) ∧
  abs (2010 - D) < abs (2010 - C) :=
by
  sorry

end closest_to_2010_l17_17697


namespace smallest_b_l17_17197

noncomputable def geometric_sequence : Prop :=
∃ (a b c r : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ b = a * r ∧ c = a * r^2 ∧ a * b * c = 216

theorem smallest_b (a b c r: ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) 
  (h_geom: b = a * r ∧ c = a * r^2 ∧ a * b * c = 216) : b = 6 :=
sorry

end smallest_b_l17_17197


namespace toy_position_from_left_l17_17724

/-- Define the total number of toys -/
def total_toys : ℕ := 19

/-- Define the position of toy (A) from the right -/
def position_from_right : ℕ := 8

/-- Prove the main statement: The position of toy (A) from the left is 12 given the conditions -/
theorem toy_position_from_left : total_toys - position_from_right + 1 = 12 := by
  sorry

end toy_position_from_left_l17_17724


namespace min_value_l17_17862

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + 2*y = 2) : 
  ∃ c : ℝ, c = 2 ∧ ∀ z, (z = (x^2 / (2*y) + 4*(y^2) / x)) → z ≥ c :=
by
  sorry

end min_value_l17_17862


namespace equation_holds_except_two_values_l17_17897

noncomputable def check_equation (a y : ℝ) (h : a ≠ 0) : Prop :=
  (a / (a + y) + y / (a - y)) / (y / (a + y) - a / (a - y)) = -1 ↔ y ≠ a ∧ y ≠ -a

theorem equation_holds_except_two_values (a y: ℝ) (h: a ≠ 0): check_equation a y h := sorry

end equation_holds_except_two_values_l17_17897


namespace modulus_of_z_equals_two_l17_17125

namespace ComplexProblem

open Complex

-- Definition and conditions of the problem
def satisfies_condition (z : ℂ) : Prop :=
  (z + I) * (1 + I) = 1 - I

-- Statement that needs to be proven
theorem modulus_of_z_equals_two (z : ℂ) (h : satisfies_condition z) : abs z = 2 :=
sorry

end ComplexProblem

end modulus_of_z_equals_two_l17_17125


namespace partial_fraction_sum_zero_l17_17629

theorem partial_fraction_sum_zero (A B C D E F : ℝ) :
  (∀ x : ℝ, 1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
     A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
sorry

end partial_fraction_sum_zero_l17_17629


namespace expected_value_is_one_dollar_l17_17203

def star_prob := 1 / 4
def moon_prob := 1 / 2
def sun_prob := 1 / 4

def star_prize := 2
def moon_prize := 4
def sun_penalty := -6

def expected_winnings := star_prob * star_prize + moon_prob * moon_prize + sun_prob * sun_penalty

theorem expected_value_is_one_dollar : expected_winnings = 1 := by
  sorry

end expected_value_is_one_dollar_l17_17203


namespace percent_dimes_value_is_60_l17_17191

variable (nickels dimes : ℕ)
variable (value_nickel value_dime : ℕ)
variable (num_nickels num_dimes : ℕ)

def total_value (n d : ℕ) (v_n v_d : ℕ) := n * v_n + d * v_d

def percent_value_dimes (total d_value : ℕ) := (d_value * 100) / total

theorem percent_dimes_value_is_60 :
  num_nickels = 40 →
  num_dimes = 30 →
  value_nickel = 5 →
  value_dime = 10 →
  percent_value_dimes (total_value num_nickels num_dimes value_nickel value_dime) (num_dimes * value_dime) = 60 := 
by sorry

end percent_dimes_value_is_60_l17_17191


namespace rate_of_increase_twice_l17_17560

theorem rate_of_increase_twice {x : ℝ} (h : (1 + x)^2 = 2) : x = (Real.sqrt 2) - 1 :=
sorry

end rate_of_increase_twice_l17_17560


namespace value_of_z_plus_one_over_y_l17_17276

theorem value_of_z_plus_one_over_y
  (x y z : ℝ)
  (h1 : 0 < x) 
  (h2 : 0 < y) 
  (h3 : 0 < z)
  (h4 : x * y * z = 1)
  (h5 : x + 1 / z = 3)
  (h6 : y + 1 / x = 31) :
  z + 1 / y = 9 / 23 :=
by
  sorry

end value_of_z_plus_one_over_y_l17_17276


namespace inequality_power_cubed_l17_17249

theorem inequality_power_cubed
  (x y a : ℝ)
  (h_condition : (0 < a ∧ a < 1) ∧ a ^ x < a ^ y) : x^3 > y^3 :=
by {
  sorry
}

end inequality_power_cubed_l17_17249


namespace pie_shop_earnings_l17_17104

-- Define the conditions
def price_per_slice : ℕ := 3
def slices_per_pie : ℕ := 10
def number_of_pies : ℕ := 6

-- Calculate the total slices
def total_slices : ℕ := number_of_pies * slices_per_pie

-- Calculate the total earnings
def total_earnings : ℕ := total_slices * price_per_slice

-- State the theorem
theorem pie_shop_earnings : total_earnings = 180 :=
by
  -- Proof can be skipped with a sorry
  sorry

end pie_shop_earnings_l17_17104


namespace exists_five_integers_sum_fifth_powers_no_four_integers_sum_fifth_powers_l17_17340

theorem exists_five_integers_sum_fifth_powers (A B C D E : ℤ) : 
  ∃ (A B C D E : ℤ), 2018 = A^5 + B^5 + C^5 + D^5 + E^5 :=
  by
    sorry

theorem no_four_integers_sum_fifth_powers (A B C D : ℤ) : 
  ¬ ∃ (A B C D : ℤ), 2018 = A^5 + B^5 + C^5 + D^5 :=
  by
    sorry

end exists_five_integers_sum_fifth_powers_no_four_integers_sum_fifth_powers_l17_17340


namespace teams_dig_tunnel_in_10_days_l17_17600

theorem teams_dig_tunnel_in_10_days (hA : ℝ) (hB : ℝ) (work_A : hA = 15) (work_B : hB = 30) : 
  (1 / (1 / hA + 1 / hB)) = 10 := 
by
  sorry

end teams_dig_tunnel_in_10_days_l17_17600


namespace find_number_of_dogs_l17_17477

variables (D P S : ℕ)
theorem find_number_of_dogs (h1 : D = 2 * P) (h2 : P = 2 * S) (h3 : 4 * D + 4 * P + 2 * S = 510) :
  D = 60 := 
sorry

end find_number_of_dogs_l17_17477


namespace license_plate_combinations_l17_17805

-- Definitions of the conditions
def num_consonants : ℕ := 20
def num_vowels : ℕ := 6
def num_digits : ℕ := 10

-- The theorem statement
theorem license_plate_combinations : num_consonants * num_vowels * num_vowels * num_digits = 7200 := by
  sorry

end license_plate_combinations_l17_17805


namespace min_value_of_m_cauchy_schwarz_inequality_l17_17020

theorem min_value_of_m (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m = a + 1 / ((a - b) * b)) : 
  ∃ t, t = 3 ∧ ∀ a b : ℝ, a > b → b > 0 → m = a + 1 / ((a - b) * b) → m ≥ t :=
sorry

theorem cauchy_schwarz_inequality (x y z : ℝ) :
  (x^2 + 4 * y^2 + z^2 = 3) → |x + 2 * y + z| ≤ 3 :=
sorry

end min_value_of_m_cauchy_schwarz_inequality_l17_17020


namespace correct_equation_for_annual_consumption_l17_17287

-- Definitions based on the problem conditions
-- average_monthly_consumption_first_half is the average monthly electricity consumption in the first half of the year, assumed to be x
def average_monthly_consumption_first_half (x : ℝ) := x

-- average_monthly_consumption_second_half is the average monthly consumption in the second half of the year, i.e., x - 2000
def average_monthly_consumption_second_half (x : ℝ) := x - 2000

-- total_annual_consumption is the total annual electricity consumption which is 150000 kWh
def total_annual_consumption (x : ℝ) := 6 * average_monthly_consumption_first_half x + 6 * average_monthly_consumption_second_half x

-- The main theorem statement which we need to prove
theorem correct_equation_for_annual_consumption (x : ℝ) : total_annual_consumption x = 150000 :=
by
  -- equation derivation
  sorry

end correct_equation_for_annual_consumption_l17_17287


namespace sufficient_but_not_necessary_condition_l17_17131

variable (a : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : a = 1) (h2 : |a| = 1) : 
  (a = 1 → |a| = 1) ∧ ¬(|a| = 1 → a = 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l17_17131


namespace solution_y_amount_l17_17416

-- Definitions based on the conditions
def alcohol_content_x : ℝ := 0.10
def alcohol_content_y : ℝ := 0.30
def initial_volume_x : ℝ := 50
def final_alcohol_percent : ℝ := 0.25

-- Function to calculate the amount of solution y needed
def required_solution_y (y : ℝ) : Prop :=
  (alcohol_content_x * initial_volume_x + alcohol_content_y * y) / (initial_volume_x + y) = final_alcohol_percent

theorem solution_y_amount : ∃ y : ℝ, required_solution_y y ∧ y = 150 := by
  sorry

end solution_y_amount_l17_17416


namespace solve_system_l17_17704

theorem solve_system :
  ∃ a b c d e : ℤ, 
    (a * b + a + 2 * b = 78) ∧
    (b * c + 3 * b + c = 101) ∧
    (c * d + 5 * c + 3 * d = 232) ∧
    (d * e + 4 * d + 5 * e = 360) ∧
    (e * a + 2 * e + 4 * a = 192) ∧
    ((a = 8 ∧ b = 7 ∧ c = 10 ∧ d = 14 ∧ e = 16) ∨ (a = -12 ∧ b = -9 ∧ c = -16 ∧ d = -24 ∧ e = -24)) :=
by
  sorry

end solve_system_l17_17704


namespace find_a_l17_17133

noncomputable def f (a x : ℝ) : ℝ := a * x * Real.log x

theorem find_a (a : ℝ) (h : (deriv (f a)) e = 3) : a = 3 / 2 :=
by
-- placeholder for the proof
sorry

end find_a_l17_17133


namespace factorize_expression_l17_17154

theorem factorize_expression (a b : ℝ) : a^2 - a * b = a * (a - b) :=
by sorry

end factorize_expression_l17_17154


namespace download_time_l17_17375

def file_size : ℕ := 90
def rate_first_part : ℕ := 5
def rate_second_part : ℕ := 10
def size_first_part : ℕ := 60

def time_first_part : ℕ := size_first_part / rate_first_part
def size_second_part : ℕ := file_size - size_first_part
def time_second_part : ℕ := size_second_part / rate_second_part
def total_time : ℕ := time_first_part + time_second_part

theorem download_time :
  total_time = 15 := by
  -- sorry can be replaced with the actual proof if needed
  sorry

end download_time_l17_17375


namespace fraction_of_total_students_l17_17801

variables (G B T : ℕ) (F : ℚ)

-- Given conditions
axiom ratio_boys_to_girls : (7 : ℚ) / 3 = B / G
axiom total_students : T = B + G
axiom fraction_equals_two_thirds_girls : (2 : ℚ) / 3 * G = F * T

-- Proof goal
theorem fraction_of_total_students : F = 1 / 5 :=
by
  sorry

end fraction_of_total_students_l17_17801


namespace parallelogram_fourth_vertex_distance_l17_17336

theorem parallelogram_fourth_vertex_distance (d1 d2 d3 d4 : ℝ) (h1 : d1 = 1) (h2 : d2 = 3) (h3 : d3 = 5) :
    d4 = 7 :=
sorry

end parallelogram_fourth_vertex_distance_l17_17336


namespace square_area_l17_17004

theorem square_area (side_length : ℝ) (h : side_length = 10) : side_length * side_length = 100 := by
  sorry

end square_area_l17_17004


namespace problem_statement_l17_17444

theorem problem_statement (x y z : ℝ) (hx : x + y + z = 2) (hxy : xy + xz + yz = -9) (hxyz : xyz = 1) :
  (yz / x) + (xz / y) + (xy / z) = 77 := sorry

end problem_statement_l17_17444


namespace roots_squared_sum_l17_17961

theorem roots_squared_sum {x y : ℝ} (hx : 3 * x^2 - 7 * x + 5 = 0) (hy : 3 * y^2 - 7 * y + 5 = 0) (hxy : x ≠ y) :
  x^2 + y^2 = 19 / 9 :=
sorry

end roots_squared_sum_l17_17961


namespace books_combination_l17_17431

theorem books_combination :
  (Nat.choose 15 3) = 455 := 
sorry

end books_combination_l17_17431


namespace geometric_sequence_a_11_l17_17837

-- Define the geometric sequence with given terms
variable (a : ℕ → ℝ) (q : ℝ)

-- Conditions
def is_geometric_sequence := ∀ n : ℕ, a (n + 1) = a n * q

axiom a_5 : a 5 = -16
axiom a_8 : a 8 = 8

-- Question to prove
theorem geometric_sequence_a_11 (h : is_geometric_sequence a q) : a 11 = -4 := 
sorry

end geometric_sequence_a_11_l17_17837


namespace rise_in_water_level_l17_17077

noncomputable def edge : ℝ := 15.0
noncomputable def base_length : ℝ := 20.0
noncomputable def base_width : ℝ := 15.0
noncomputable def volume_cube : ℝ := edge ^ 3
noncomputable def base_area : ℝ := base_length * base_width

theorem rise_in_water_level :
  (volume_cube / base_area) = 11.25 :=
by
  sorry

end rise_in_water_level_l17_17077


namespace find_quarters_l17_17599

def num_pennies := 123
def num_nickels := 85
def num_dimes := 35
def cost_per_scoop_cents := 300  -- $3 = 300 cents
def num_family_members := 5
def leftover_cents := 48

def total_cost_cents := num_family_members * cost_per_scoop_cents
def total_initial_cents := total_cost_cents + leftover_cents

-- Values of coins in cents
def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25

def total_pennies_value := num_pennies * penny_value
def total_nickels_value := num_nickels * nickel_value
def total_dimes_value := num_dimes * dime_value
def total_initial_excluding_quarters := total_pennies_value + total_nickels_value + total_dimes_value

def total_quarters_value := total_initial_cents - total_initial_excluding_quarters
def num_quarters := total_quarters_value / quarter_value

theorem find_quarters : num_quarters = 26 := by
  sorry

end find_quarters_l17_17599


namespace trace_bag_weight_is_two_l17_17502

-- Define the weights of Gordon's shopping bags
def weight_gordon1 : ℕ := 3
def weight_gordon2 : ℕ := 7

-- Summarize Gordon's total weight
def total_weight_gordon : ℕ := weight_gordon1 + weight_gordon2

-- Provide necessary conditions from problem statement
def trace_bags_count : ℕ := 5
def trace_total_weight : ℕ := total_weight_gordon
def trace_one_bag_weight : ℕ := trace_total_weight / trace_bags_count

theorem trace_bag_weight_is_two : trace_one_bag_weight = 2 :=
by 
  -- Placeholder for proof
  sorry

end trace_bag_weight_is_two_l17_17502


namespace double_inputs_revenue_l17_17687

theorem double_inputs_revenue (A K L : ℝ) (α1 α2 : ℝ) (hα1 : α1 = 0.6) (hα2 : α2 = 0.5) (hα1_bound : 0 < α1 ∧ α1 < 1) (hα2_bound : 0 < α2 ∧ α2 < 1) :
  A * (2 * K) ^ α1 * (2 * L) ^ α2 > 2 * (A * K ^ α1 * L ^ α2) :=
by
  sorry

end double_inputs_revenue_l17_17687


namespace dana_total_earnings_l17_17064

-- Define the constants for Dana's hourly rate and hours worked each day
def hourly_rate : ℝ := 13
def friday_hours : ℝ := 9
def saturday_hours : ℝ := 10
def sunday_hours : ℝ := 3

-- Define the total earnings calculation function
def total_earnings (rate : ℝ) (hours1 hours2 hours3 : ℝ) : ℝ :=
  rate * hours1 + rate * hours2 + rate * hours3

-- The main statement
theorem dana_total_earnings : total_earnings hourly_rate friday_hours saturday_hours sunday_hours = 286 := by
  sorry

end dana_total_earnings_l17_17064


namespace product_of_fractions_is_27_l17_17668

theorem product_of_fractions_is_27 :
  (1/3) * (9/1) * (1/27) * (81/1) * (1/243) * (729/1) = 27 :=
by
  sorry

end product_of_fractions_is_27_l17_17668


namespace number_of_real_solutions_l17_17783

noncomputable def f (x : ℝ) : ℝ :=
  (Finset.range 50).sum (λ n => (n + 1 : ℝ) / (x - (n + 1 : ℝ)))

theorem number_of_real_solutions : ∃ n : ℕ, n = 51 ∧ ∀ x : ℝ, f x = x + 1 ↔ n = 51 :=
by
  sorry

end number_of_real_solutions_l17_17783


namespace rational_k_quadratic_solution_count_l17_17781

theorem rational_k_quadratic_solution_count (N : ℕ) :
  (N = 98) ↔ 
  (∃ (k : ℚ) (x : ℤ), |k| < 500 ∧ (3 * x^2 + k * x + 7 = 0)) :=
sorry

end rational_k_quadratic_solution_count_l17_17781


namespace shop_owner_cheat_selling_percentage_l17_17787

noncomputable def percentage_cheat_buying : ℝ := 12
noncomputable def profit_percentage : ℝ := 40
noncomputable def percentage_cheat_selling : ℝ := 20

theorem shop_owner_cheat_selling_percentage 
  (percentage_cheat_buying : ℝ := 12)
  (profit_percentage : ℝ := 40) :
  percentage_cheat_selling = 20 := 
sorry

end shop_owner_cheat_selling_percentage_l17_17787


namespace find_price_of_turban_l17_17542

-- Define the main variables and conditions
def price_of_turban (T : ℝ) : Prop :=
  ((3 / 4) * 90 + T = 60 + T) → T = 30

-- State the theorem with the given conditions and aim to find T
theorem find_price_of_turban (T : ℝ) (h1 : 90 + T = 120) :  price_of_turban T :=
by
  intros
  sorry


end find_price_of_turban_l17_17542


namespace cabbage_count_l17_17447

theorem cabbage_count 
  (length : ℝ)
  (width : ℝ)
  (density : ℝ)
  (h_length : length = 16)
  (h_width : width = 12)
  (h_density : density = 9) : 
  length * width * density = 1728 := 
by
  rw [h_length, h_width, h_density]
  norm_num
  done

end cabbage_count_l17_17447


namespace select_pairs_eq_l17_17678

open Set

-- Definitions for sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

-- Statement of the theorem
theorem select_pairs_eq :
  {p | p.1 ∈ A ∧ p.2 ∈ B} = {(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)} :=
by sorry

end select_pairs_eq_l17_17678


namespace downstream_speed_l17_17451

-- Define constants based on conditions given 
def V_upstream : ℝ := 30
def V_m : ℝ := 35

-- Define the speed of the stream based on the given conditions and upstream speed
def V_s : ℝ := V_m - V_upstream

-- The downstream speed is the man's speed in still water plus the stream speed
def V_downstream : ℝ := V_m + V_s

-- Theorem to be proved
theorem downstream_speed : V_downstream = 40 :=
by
  -- The actual proof steps are omitted
  sorry

end downstream_speed_l17_17451


namespace intersecting_lines_l17_17729

theorem intersecting_lines (m b : ℝ)
  (h1 : ∀ x, (9 : ℝ) = 2 * m * x + 3 → x = 3)
  (h2 : ∀ x, (9 : ℝ) = 4 * x + b → x = 3) :
  b + 2 * m = -1 :=
sorry

end intersecting_lines_l17_17729


namespace sams_trip_length_l17_17841

theorem sams_trip_length (total_trip : ℚ) 
  (h1 : total_trip / 4 + 24 + total_trip / 6 = total_trip) : 
  total_trip = 288 / 7 :=
by
  -- proof placeholder
  sorry

end sams_trip_length_l17_17841


namespace percentage_of_60_eq_15_l17_17810

-- Conditions provided in the problem
def percentage (p : ℚ) : ℚ := p / 100
def num : ℚ := 60
def fraction_of_num (p : ℚ) (n : ℚ) : ℚ := (percentage p) * n

-- Assertion to be proved
theorem percentage_of_60_eq_15 : fraction_of_num 25 num = 15 := 
by 
  show fraction_of_num 25 60 = 15
  sorry

end percentage_of_60_eq_15_l17_17810


namespace initial_pipes_count_l17_17693

theorem initial_pipes_count (n r : ℝ) 
  (h1 : n * r = 1 / 12) 
  (h2 : (n + 10) * r = 1 / 4) : 
  n = 5 := 
by 
  sorry

end initial_pipes_count_l17_17693


namespace find_angle_D_l17_17654

theorem find_angle_D 
  (A B C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 50) :
  D = 25 := 
by
  sorry

end find_angle_D_l17_17654


namespace min_k_inequality_l17_17626

theorem min_k_inequality (α β : ℝ) (hα : 0 < α) (hα2 : α < 2 * Real.pi / 3)
  (hβ : 0 < β) (hβ2 : β < 2 * Real.pi / 3) :
  4 * Real.cos α ^ 2 + 2 * Real.cos α * Real.cos β + 4 * Real.cos β ^ 2
  - 3 * Real.cos α - 3 * Real.cos β - 6 < 0 :=
by
  sorry

end min_k_inequality_l17_17626


namespace solve_cubic_eq_l17_17782

theorem solve_cubic_eq (z : ℂ) : z^3 = 27 ↔ (z = 3 ∨ z = - (3 / 2) + (3 / 2) * Complex.I * Real.sqrt 3 ∨ z = - (3 / 2) - (3 / 2) * Complex.I * Real.sqrt 3) :=
by
  sorry

end solve_cubic_eq_l17_17782


namespace isosceles_triangle_perimeter_l17_17910

theorem isosceles_triangle_perimeter (a b c : ℝ) (h₀ : a = 5) (h₁ : b = 10) 
  (h₂ : c = 10 ∨ c = 5) (h₃ : a = b ∨ b = c ∨ c = a) 
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 25 := by
  sorry

end isosceles_triangle_perimeter_l17_17910


namespace smallest_n_for_quadratic_factorization_l17_17047

theorem smallest_n_for_quadratic_factorization :
  ∃ (n : ℤ), (∀ A B : ℤ, A * B = 50 → n = 5 * B + A) ∧ (∀ m : ℤ, 
    (∀ A B : ℤ, A * B = 50 → m ≤ 5 * B + A) → n ≤ m) :=
by
  sorry

end smallest_n_for_quadratic_factorization_l17_17047


namespace order_of_a_b_c_l17_17440

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 2 / Real.log 3
noncomputable def c : ℝ := (1 / 2) * (Real.log 5 / Real.log 2)

theorem order_of_a_b_c : a > c ∧ c > b :=
by
  -- proof here
  sorry

end order_of_a_b_c_l17_17440


namespace jessie_final_position_l17_17565

theorem jessie_final_position :
  ∃ y : ℕ,
  (0 + 6 * 4 = 24) ∧
  (y = 24) :=
by
  sorry

end jessie_final_position_l17_17565


namespace find_q_l17_17023

theorem find_q (p q : ℚ) (h1 : 5 * p + 7 * q = 20) (h2 : 7 * p + 5 * q = 26) : q = 5 / 12 := by
  sorry

end find_q_l17_17023


namespace find_extrema_on_interval_l17_17312

noncomputable def y (x : ℝ) := (10 * x + 10) / (x^2 + 2 * x + 2)

theorem find_extrema_on_interval :
  ∃ (min_val max_val : ℝ) (min_x max_x : ℝ), 
    min_val = 0 ∧ min_x = -1 ∧ max_val = 5 ∧ max_x = 0 ∧ 
    (∀ x ∈ Set.Icc (-1 : ℝ) 2, y x ≥ min_val) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 2, y x ≤ max_val) :=
by
  sorry

end find_extrema_on_interval_l17_17312


namespace last_three_digits_of_7_to_50_l17_17357

theorem last_three_digits_of_7_to_50 : (7^50) % 1000 = 991 := 
by 
  sorry

end last_three_digits_of_7_to_50_l17_17357


namespace largest_value_of_x_not_defined_l17_17305

noncomputable def quadratic_formula (a b c : ℝ) : (ℝ × ℝ) :=
  let discriminant := b*b - 4*a*c
  let sqrt_discriminant := Real.sqrt discriminant
  let x1 := (-b + sqrt_discriminant) / (2*a)
  let x2 := (-b - sqrt_discriminant) / (2*a)
  (x1, x2)

noncomputable def largest_root : ℝ :=
  let (x1, x2) := quadratic_formula 4 (-81) 49
  if x1 > x2 then x1 else x2

theorem largest_value_of_x_not_defined :
  largest_root = 19.6255 :=
by
  sorry

end largest_value_of_x_not_defined_l17_17305


namespace find_sample_size_l17_17512

def sample_size (sample : List ℕ) : ℕ :=
  sample.length

theorem find_sample_size :
  sample_size (List.replicate 500 0) = 500 :=
by
  sorry

end find_sample_size_l17_17512


namespace add_two_inequality_l17_17570

theorem add_two_inequality (a b : ℝ) (h : a > b) : a + 2 > b + 2 :=
sorry

end add_two_inequality_l17_17570


namespace solution_set_l17_17005

-- Defining the condition and inequalities:
variable (a x : Real)

-- Condition that a < 0
def condition_a : Prop := a < 0

-- Inequalities in the system
def inequality1 : Prop := x > -2 * a
def inequality2 : Prop := x > 3 * a

-- The solution set we need to prove
theorem solution_set (h : condition_a a) : (inequality1 a x) ∧ (inequality2 a x) ↔ x > -2 * a :=
by
  sorry

end solution_set_l17_17005


namespace value_of_Y_l17_17519

theorem value_of_Y :
  let part1 := 15 * 180 / 100  -- 15% of 180
  let part2 := part1 - part1 / 3  -- one-third less than 15% of 180
  let part3 := 24.5 * (2 * 270 / 3) / 100  -- 24.5% of (2/3 * 270)
  let part4 := (5.4 * 2) / (0.25 * 0.25)  -- (5.4 * 2) / (0.25)^2
  let Y := part2 + part3 - part4
  Y = -110.7 := by
    -- proof skipped
    sorry

end value_of_Y_l17_17519


namespace number_of_female_students_selected_is_20_l17_17210

noncomputable def number_of_female_students_to_be_selected
(total_students : ℕ) (female_students : ℕ) (students_to_be_selected : ℕ) : ℕ :=
students_to_be_selected * female_students / total_students

theorem number_of_female_students_selected_is_20 :
  number_of_female_students_to_be_selected 2000 800 50 = 20 := 
by
  sorry

end number_of_female_students_selected_is_20_l17_17210


namespace probability_tenth_ball_black_l17_17878

theorem probability_tenth_ball_black :
  let total_balls := 30
  let black_balls := 4
  let red_balls := 7
  let yellow_balls := 5
  let green_balls := 6
  let white_balls := 8
  (black_balls / total_balls) = 4 / 30 :=
by sorry

end probability_tenth_ball_black_l17_17878


namespace maximum_sequence_length_l17_17084

theorem maximum_sequence_length
  (seq : List ℚ) 
  (h1 : ∀ i : ℕ, i + 2 < seq.length → (seq.get! i + seq.get! (i+1) + seq.get! (i+2)) < 0)
  (h2 : ∀ i : ℕ, i + 3 < seq.length → (seq.get! i + seq.get! (i+1) + seq.get! (i+2) + seq.get! (i+3)) > 0) 
  : seq.length ≤ 5 := 
sorry

end maximum_sequence_length_l17_17084


namespace pq_sufficient_not_necessary_l17_17218

theorem pq_sufficient_not_necessary (p q : Prop) :
  (¬ (p ∨ q)) → (¬ p ∧ ¬ q) ∧ ¬ ((¬ p ∧ ¬ q) → (¬ (p ∨ q))) :=
sorry

end pq_sufficient_not_necessary_l17_17218


namespace sylvia_time_to_complete_job_l17_17883

theorem sylvia_time_to_complete_job (S : ℝ) (h₁ : 18 ≠ 0) (h₂ : 30 ≠ 0)
  (together_rate : (1 / S) + (1 / 30) = 1 / 18) :
  S = 45 :=
by
  -- Proof will be provided here
  sorry

end sylvia_time_to_complete_job_l17_17883


namespace yolk_count_proof_l17_17487

-- Define the conditions of the problem
def eggs_in_carton : ℕ := 12
def double_yolk_eggs : ℕ := 5
def single_yolk_eggs : ℕ := eggs_in_carton - double_yolk_eggs
def yolks_in_double_yolk_eggs : ℕ := double_yolk_eggs * 2
def yolks_in_single_yolk_eggs : ℕ := single_yolk_eggs
def total_yolks : ℕ := yolks_in_single_yolk_eggs + yolks_in_double_yolk_eggs

-- Stating the theorem to prove the total number of yolks is 17
theorem yolk_count_proof : total_yolks = 17 := 
by
  sorry

end yolk_count_proof_l17_17487


namespace estimate_expr_range_l17_17749

theorem estimate_expr_range :
  5 < (2 * Real.sqrt 5 + 5 * Real.sqrt 2) * Real.sqrt (1 / 5) ∧
  (2 * Real.sqrt 5 + 5 * Real.sqrt 2) * Real.sqrt (1 / 5) < 6 :=
  sorry

end estimate_expr_range_l17_17749


namespace contrapositive_honor_roll_l17_17847

variable (Student : Type) (scores_hundred : Student → Prop) (honor_roll_qualifies : Student → Prop)

theorem contrapositive_honor_roll (s : Student) :
  (¬ honor_roll_qualifies s) → (¬ scores_hundred s) := 
sorry

end contrapositive_honor_roll_l17_17847


namespace similar_terms_solution_l17_17885

theorem similar_terms_solution
  (a b : ℝ)
  (m n x y : ℤ)
  (h1 : m - 1 = n - 2 * m)
  (h2 : m + n = 3 * m + n - 4)
  (h3 : m * x + (n - 2) * y = 24)
  (h4 : 2 * m * x + n * y = 46) :
  x = 9 ∧ y = 2 := by
  sorry

end similar_terms_solution_l17_17885


namespace chris_raisins_nuts_l17_17520

theorem chris_raisins_nuts (R N x : ℝ) 
  (hN : N = 4 * R) 
  (hxR : x * R = 0.15789473684210525 * (x * R + 4 * N)) :
  x = 3 :=
by
  sorry

end chris_raisins_nuts_l17_17520


namespace xy_difference_squared_l17_17816

theorem xy_difference_squared (x y : ℝ) (h1 : x + y = 8) (h2 : x - y = 4) : x^2 - y^2 = 32 :=
by
  -- the proof goes here
  sorry

end xy_difference_squared_l17_17816


namespace arithmetic_sequence_common_difference_l17_17879

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) (h1 : a 1 = 1) (h3 : a 3 = 4) :
  ∃ d : ℤ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ d = 3 / 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l17_17879


namespace words_to_numbers_l17_17456

def word_to_num (w : String) : Float := sorry

theorem words_to_numbers :
  word_to_num "fifty point zero zero one" = 50.001 ∧
  word_to_num "seventy-five point zero six" = 75.06 :=
by
  sorry

end words_to_numbers_l17_17456


namespace isosceles_triangle_leg_length_l17_17075

-- Define the necessary condition for the isosceles triangle
def isosceles_triangle (a b c : ℕ) : Prop :=
  b = c ∧ a + b + c = 16 ∧ a = 4

-- State the theorem we want to prove
theorem isosceles_triangle_leg_length :
  ∃ (b c : ℕ), isosceles_triangle 4 b c ∧ b = 6 :=
by
  -- Formal proof will be provided here
  sorry

end isosceles_triangle_leg_length_l17_17075


namespace intersection_A_B_l17_17640

open Set

def universal_set : Set ℤ := {x | 1 ≤ x ∧ x ≤ 5}
def A : Set ℤ := {1, 2, 3}
def complement_B : Set ℤ := {1, 2}
def B : Set ℤ := universal_set \ complement_B

theorem intersection_A_B : A ∩ B = {3} :=
by
  sorry

end intersection_A_B_l17_17640


namespace total_fencing_cost_l17_17904

-- Conditions
def length : ℝ := 55
def cost_per_meter : ℝ := 26.50

-- We derive breadth from the given conditions
def breadth : ℝ := length - 10

-- Calculate the perimeter of the rectangular plot
def perimeter : ℝ := 2 * (length + breadth)

-- Calculate the total cost of fencing the plot
def total_cost : ℝ := cost_per_meter * perimeter

-- The theorem to prove that total cost is equal to 5300
theorem total_fencing_cost : total_cost = 5300 := by
  -- Calculation goes here
  sorry

end total_fencing_cost_l17_17904


namespace find_eccentricity_find_equation_l17_17243

open Real

-- Conditions for the first question
def is_ellipse (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

def are_focus (a b : ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  F1 = ( - sqrt (a^2 - b^2), 0) ∧ F2 = (sqrt (a^2 - b^2), 0)

def arithmetic_sequence (a b : ℝ) (A B : ℝ × ℝ) (F1 : ℝ × ℝ) : Prop :=
  let dist_AF1 := abs (A.1 - F1.1)
  let dist_BF1 := abs (B.1 - F1.1)
  let dist_AB := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  (dist_AF1 + dist_AB + dist_BF1 = 4 * a) ∧
  (dist_AF1 + dist_BF1 = 2 * dist_AB)

-- Proof statement for the eccentricity
theorem find_eccentricity (a b : ℝ) (F1 F2 A B : ℝ × ℝ)
  (h1 : a > b) (h2 : b > 0) (h3 : is_ellipse a b)
  (h4 : are_focus a b F1 F2)
  (h5 : arithmetic_sequence a b A B F1) :
  ∃ e : ℝ, e = sqrt 2 / 2 :=
sorry

-- Conditions for the second question
def geometric_property (a b : ℝ) (A B P : ℝ × ℝ) : Prop :=
  ∀ x y : ℝ, P = (0, -1) → 
             (x^2 / a^2) + (y^2 / b^2) = 1 → 
             abs ((P.1 - A.1)^2 + (P.2 - A.2)^2) = 
             abs ((P.1 - B.1)^2 + (P.2 - B.2)^2)

-- Proof statement for the equation of the ellipse
theorem find_equation (a b : ℝ) (A B P : ℝ × ℝ)
  (h1 : a = 3 * sqrt 2) (h2 : b = 3) (h3 : P = (0, -1))
  (h4 : is_ellipse a b) (h5 : geometric_property a b A B P) :
  ∃ E : Prop, E = ((x : ℝ) * 2 / 18 + (y : ℝ) * 2 / 9 = 1) :=
sorry

end find_eccentricity_find_equation_l17_17243


namespace deviation_interpretation_l17_17351

variable (average_score : ℝ)
variable (x : ℝ)

-- Given condition
def higher_than_average : Prop := x = average_score + 5

-- To prove
def lower_than_average : Prop := x = average_score - 9

theorem deviation_interpretation (x : ℝ) (h : x = average_score + 5) : x - 14 = average_score - 9 :=
by
  sorry

end deviation_interpretation_l17_17351


namespace pepperoni_crust_ratio_l17_17753

-- Define the conditions as Lean 4 statements
def L : ℕ := 50
def C : ℕ := 2 * L
def D : ℕ := 210
def S : ℕ := L + C + D
def S_E : ℕ := S / 4
def CR : ℕ := 600
def CH : ℕ := 400
def PizzaTotal (P : ℕ) : ℕ := CR + P + CH
def PizzaEats (P : ℕ) : ℕ := (PizzaTotal P) / 5
def JacksonEats : ℕ := 330

theorem pepperoni_crust_ratio (P : ℕ) (h1 : S_E + PizzaEats P = JacksonEats) : P / CR = 1 / 3 :=
by sorry

end pepperoni_crust_ratio_l17_17753


namespace lele_dongdong_meet_probability_l17_17094

-- Define the conditions: distances and speeds
def segment_length : ℕ := 500
def n : ℕ := sorry
def d : ℕ := segment_length * n
def lele_speed : ℕ := 18
def dongdong_speed : ℕ := 24

-- Define times to traverse distance d
def t_L : ℚ := d / lele_speed
def t_D : ℚ := d / dongdong_speed

-- Define the time t when they meet
def t : ℚ := d / (lele_speed + dongdong_speed)

-- Define the maximum of t_L and t_D
def max_t_L_t_D : ℚ := max t_L t_D

-- Define the probability they meet on their way
def P_meet : ℚ := t / max_t_L_t_D

-- The theorem to prove the probability of meeting is 97/245
theorem lele_dongdong_meet_probability : P_meet = 97 / 245 :=
sorry

end lele_dongdong_meet_probability_l17_17094


namespace correct_population_growth_pattern_statement_l17_17060

-- Definitions based on the conditions provided
def overall_population_growth_modern (world_population : ℕ) : Prop :=
  -- The overall pattern of population growth worldwide is already in the modern stage
  sorry

def transformation_synchronized (world_population : ℕ) : Prop :=
  -- The transformation of population growth patterns in countries or regions around the world is synchronized
  sorry

def developed_countries_transformed (world_population : ℕ) : Prop :=
  -- Developed countries have basically completed the transformation of population growth patterns
  sorry

def transformation_determined_by_population_size (world_population : ℕ) : Prop :=
  -- The process of transformation in population growth patterns is determined by the population size of each area
  sorry

-- The statement to be proven
theorem correct_population_growth_pattern_statement (world_population : ℕ) :
  developed_countries_transformed world_population := sorry

end correct_population_growth_pattern_statement_l17_17060


namespace angle_B_degrees_l17_17010

theorem angle_B_degrees (A B C : ℕ) (h1 : A < B) (h2 : B < C) (h3 : 4 * C = 7 * A) (h4 : A + B + C = 180) : B = 59 :=
sorry

end angle_B_degrees_l17_17010


namespace relation_between_y_l17_17991

/-- Definition of the points on the parabola y = -(x-3)^2 - 4 --/
def pointA (y₁ : ℝ) : Prop := y₁ = -(1/4 - 3)^2 - 4
def pointB (y₂ : ℝ) : Prop := y₂ = -(1 - 3)^2 - 4
def pointC (y₃ : ℝ) : Prop := y₃ = -(4 - 3)^2 - 4 

/-- Relationship between y₁, y₂, y₃ for given points on the quadratic function --/
theorem relation_between_y (y₁ y₂ y₃ : ℝ) 
  (hA : pointA y₁)
  (hB : pointB y₂)
  (hC : pointC y₃) : 
  y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

end relation_between_y_l17_17991


namespace marbles_given_to_juan_l17_17414

def initial : ℕ := 776
def left : ℕ := 593

theorem marbles_given_to_juan : initial - left = 183 :=
by sorry

end marbles_given_to_juan_l17_17414


namespace smallest_n_for_inequality_l17_17618

theorem smallest_n_for_inequality (n : ℕ) : 5 + 3 * n > 300 ↔ n = 99 := by
  sorry

end smallest_n_for_inequality_l17_17618


namespace mother_age_l17_17778

theorem mother_age (x : ℕ) (h1 : 3 * x + x = 40) : 3 * x = 30 :=
by
  -- Here we should provide the proof but for now we use sorry to skip it
  sorry

end mother_age_l17_17778


namespace helly_half_planes_helly_convex_polygons_l17_17692

-- Helly's theorem for half-planes
theorem helly_half_planes (n : ℕ) (H : Fin n → Set ℝ) 
  (h : ∀ (i j k : Fin n), (H i ∩ H j ∩ H k).Nonempty) : 
  (⋂ i, H i).Nonempty :=
sorry

-- Helly's theorem for convex polygons
theorem helly_convex_polygons (n : ℕ) (P : Fin n → Set ℝ) 
  (h : ∀ (i j k : Fin n), (P i ∩ P j ∩ P k).Nonempty) : 
  (⋂ i, P i).Nonempty :=
sorry

end helly_half_planes_helly_convex_polygons_l17_17692


namespace number_of_girls_l17_17795

theorem number_of_girls (num_vans : ℕ) (students_per_van : ℕ) (num_boys : ℕ) (total_students : ℕ) (num_girls : ℕ) 
(h1 : num_vans = 5) 
(h2 : students_per_van = 28) 
(h3 : num_boys = 60) 
(h4 : total_students = num_vans * students_per_van) 
(h5 : num_girls = total_students - num_boys) : 
num_girls = 80 :=
by
  sorry

end number_of_girls_l17_17795


namespace circular_paper_pieces_needed_l17_17513

-- Definition of the problem conditions
def side_length_dm := 10
def side_length_cm := side_length_dm * 10
def perimeter_cm := 4 * side_length_cm
def number_of_sides := 4
def semicircles_per_side := 1
def total_semicircles := number_of_sides * semicircles_per_side
def semicircles_to_circles := 2
def total_circles := total_semicircles / semicircles_to_circles
def paper_pieces_per_circle := 20

-- Main theorem stating the problem and the answer.
theorem circular_paper_pieces_needed : (total_circles * paper_pieces_per_circle) = 40 :=
by sorry

end circular_paper_pieces_needed_l17_17513


namespace minimum_distance_after_9_minutes_l17_17171

-- Define the initial conditions and movement rules of the robot
structure RobotMovement :=
  (minutes : ℕ)
  (movesStraight : Bool) -- Did the robot move straight in the first minute
  (speed : ℕ)          -- The speed, which is 10 meters/minute
  (turns : Fin (minutes + 1) → ℤ) -- Turns in degrees (-90 for left, 0 for straight, 90 for right)

-- Define the distance function for the robot movement after given minutes
def distanceFromOrigin (rm : RobotMovement) : ℕ :=
  -- This function calculates the minimum distance from the origin where the details are abstracted
  sorry

-- Define the specific conditions of our problem
def robotMovementExample : RobotMovement :=
  { minutes := 9, movesStraight := true, speed := 10,
    turns := λ i => if i = 0 then 0 else -- no turn in the first minute
                      if i % 2 == 0 then 90 else -90 -- Example turning pattern
  }

-- Statement of the proof
theorem minimum_distance_after_9_minutes :
  distanceFromOrigin robotMovementExample = 10 :=
sorry

end minimum_distance_after_9_minutes_l17_17171


namespace min_value_x_plus_y_l17_17666

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : (2 * x + Real.sqrt (4 * x^2 + 1)) * (Real.sqrt (y^2 + 4) - 2) ≥ y) : 
  x + y >= 2 := 
by
  sorry

end min_value_x_plus_y_l17_17666


namespace Tim_driving_hours_l17_17870

theorem Tim_driving_hours (D T : ℕ) (h1 : T = 2 * D) (h2 : D + T = 15) : D = 5 :=
by
  sorry

end Tim_driving_hours_l17_17870


namespace factor_expression_l17_17959

theorem factor_expression (x : ℝ) : 16 * x^4 - 4 * x^2 = 4 * x^2 * (2 * x + 1) * (2 * x - 1) :=
sorry

end factor_expression_l17_17959


namespace leo_current_weight_l17_17269

theorem leo_current_weight (L K : ℝ) 
  (h1 : L + 10 = 1.5 * K) 
  (h2 : L + K = 140) : 
  L = 80 :=
by 
  sorry

end leo_current_weight_l17_17269


namespace smallest_n_value_existence_l17_17006

-- Define a three-digit positive integer n such that the conditions hold
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def problem_conditions (n : ℕ) : Prop :=
  n % 9 = 3 ∧ n % 6 = 3

-- Main statement: There exists a three-digit positive integer n satisfying the conditions and is equal to 111
theorem smallest_n_value_existence : ∃ n : ℕ, is_three_digit n ∧ problem_conditions n ∧ n = 111 :=
by
  sorry

end smallest_n_value_existence_l17_17006


namespace words_memorized_l17_17464

theorem words_memorized (x y z : ℕ) (h1 : x = 4 * (y + z) / 5) (h2 : x + y = 6 * z / 5) (h3 : 100 < x + y + z ∧ x + y + z < 200) : 
  x + y + z = 198 :=
by
  sorry

end words_memorized_l17_17464


namespace xy_square_diff_l17_17074

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l17_17074


namespace smallest_integral_k_l17_17574

theorem smallest_integral_k (k : ℤ) :
  (297 - 108 * k < 0) ↔ (k ≥ 3) :=
sorry

end smallest_integral_k_l17_17574


namespace truncated_cone_sphere_radius_l17_17072

structure TruncatedCone :=
(base_radius_top : ℝ)
(base_radius_bottom : ℝ)

noncomputable def sphere_radius (c : TruncatedCone) : ℝ :=
  if c.base_radius_top = 24 ∧ c.base_radius_bottom = 6 then 12 else 0

theorem truncated_cone_sphere_radius (c : TruncatedCone) (h_radii : c.base_radius_top = 24 ∧ c.base_radius_bottom = 6) :
  sphere_radius c = 12 :=
by
  sorry

end truncated_cone_sphere_radius_l17_17072


namespace find_5_digit_number_l17_17166

theorem find_5_digit_number {A B C D E : ℕ} 
  (hA_even : A % 2 = 0) 
  (hB_even : B % 2 = 0) 
  (hA_half_B : A = B / 2) 
  (hC_sum : C = A + B) 
  (hDE_prime : Prime (10 * D + E)) 
  (hD_3B : D = 3 * B) : 
  10000 * A + 1000 * B + 100 * C + 10 * D + E = 48247 := 
sorry

end find_5_digit_number_l17_17166


namespace unique_solution_quadratic_l17_17281

theorem unique_solution_quadratic (x : ℚ) (b : ℚ) (h_b_nonzero : b ≠ 0) (h_discriminant_zero : 625 - 36 * b = 0) : 
  (b = 625 / 36) ∧ (x = -18 / 25) → b * x^2 + 25 * x + 9 = 0 :=
by 
  -- We assume b = 625 / 36 and x = -18 / 25
  rintro ⟨hb, hx⟩
  -- Substitute b and x into the quadratic equation and simplify
  rw [hb, hx]
  -- Show the left-hand side evaluates to zero
  sorry

end unique_solution_quadratic_l17_17281


namespace geometric_series_first_term_l17_17699

theorem geometric_series_first_term 
  (S : ℝ) (r : ℝ) (a : ℝ)
  (h_sum : S = 40) (h_ratio : r = 1/4) :
  S = a / (1 - r) → a = 30 := by
  sorry

end geometric_series_first_term_l17_17699


namespace ferry_routes_ratio_l17_17257

theorem ferry_routes_ratio :
  ∀ (D_P D_Q : ℝ) (speed_P time_P speed_Q time_Q : ℝ),
  speed_P = 8 →
  time_P = 3 →
  speed_Q = speed_P + 4 →
  time_Q = time_P + 1 →
  D_P = speed_P * time_P →
  D_Q = speed_Q * time_Q →
  D_Q / D_P = 2 :=
by sorry

end ferry_routes_ratio_l17_17257


namespace find_a9_l17_17903

variable (S : ℕ → ℤ) (a : ℕ → ℤ)
variable (d a1 : ℤ)

def arithmetic_seq (n : ℕ) : ℤ :=
  a1 + ↑n * d

def sum_arithmetic_seq (n : ℕ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

axiom h1 : sum_arithmetic_seq 8 = 4 * arithmetic_seq 3
axiom h2 : arithmetic_seq 7 = -2

theorem find_a9 : arithmetic_seq 9 = -6 :=
by
  sorry

end find_a9_l17_17903


namespace center_of_circle_l17_17380

theorem center_of_circle (x y : ℝ) : 
    (∃ x y : ℝ, x^2 + y^2 = 4*x - 6*y + 9) → (x, y) = (2, -3) := 
by sorry

end center_of_circle_l17_17380


namespace minimum_value_exists_l17_17070

noncomputable def minimized_function (x y : ℝ) : ℝ :=
  3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y + y^3

theorem minimum_value_exists :
  ∃ (x y : ℝ), minimized_function x y = minimized_function (4/3 - 2 * y/3) y :=
sorry

end minimum_value_exists_l17_17070


namespace common_divisors_count_48_80_l17_17278

noncomputable def prime_factors_48 : Nat -> Prop
| n => n = 48

noncomputable def prime_factors_80 : Nat -> Prop
| n => n = 80

theorem common_divisors_count_48_80 :
  let gcd_48_80 := 2^4
  let divisors_of_gcd := [1, 2, 4, 8, 16]
  prime_factors_48 48 ∧ prime_factors_80 80 →
  List.length divisors_of_gcd = 5 :=
by
  intros
  sorry

end common_divisors_count_48_80_l17_17278


namespace union_of_A_and_B_l17_17356

-- Condition definitions
def A : Set ℝ := {x : ℝ | abs (x - 3) < 2}
def B : Set ℝ := {x : ℝ | (x + 1) / (x - 2) ≤ 0}

-- The theorem we need to prove
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 5} :=
by
  -- This is where the proof would go if it were required
  sorry

end union_of_A_and_B_l17_17356


namespace simplify_expression_1_simplify_expression_2_l17_17086

theorem simplify_expression_1 (x y : ℝ) :
  x^2 + 5*y - 4*x^2 - 3*y = -3*x^2 + 2*y :=
sorry

theorem simplify_expression_2 (a b : ℝ) :
  7*a + 3*(a - 3*b) - 2*(b - a) = 12*a - 11*b :=
sorry

end simplify_expression_1_simplify_expression_2_l17_17086


namespace tangent_parabola_line_l17_17970

theorem tangent_parabola_line (a : ℝ) :
  (∃ x : ℝ, ax^2 + 1 = x ∧ ∀ y : ℝ, (y = ax^2 + 1 → y = x)) ↔ a = 1/4 :=
by
  sorry

end tangent_parabola_line_l17_17970


namespace gcd_888_1147_l17_17507

theorem gcd_888_1147 : Nat.gcd 888 1147 = 37 := by
  sorry

end gcd_888_1147_l17_17507


namespace min_value_of_f_l17_17955

noncomputable def f (x : ℝ) : ℝ := 2 + 3 * x + 4 / (x - 1)

theorem min_value_of_f :
  (∀ x : ℝ, x > 1 → f x ≥ (5 + 4 * Real.sqrt 3)) ∧
  (f (1 + 2 * Real.sqrt 3 / 3) = 5 + 4 * Real.sqrt 3) :=
by
  sorry

end min_value_of_f_l17_17955


namespace xy_product_l17_17593

theorem xy_product (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 :=
sorry

end xy_product_l17_17593


namespace sales_fifth_month_l17_17082

-- Definitions based on conditions
def sales1 : ℝ := 5420
def sales2 : ℝ := 5660
def sales3 : ℝ := 6200
def sales4 : ℝ := 6350
def sales6 : ℝ := 8270
def average_sale : ℝ := 6400

-- Lean proof problem statement
theorem sales_fifth_month :
  sales1 + sales2 + sales3 + sales4 + sales6 + s = 6 * average_sale  →
  s = 6500 :=
by
  sorry

end sales_fifth_month_l17_17082


namespace simplify_expression_l17_17912

theorem simplify_expression (a b : ℂ) (x : ℂ) (hb : b ≠ 0) (ha : a ≠ b) (hx : x = a / b) :
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by
  -- Proof goes here
  sorry

end simplify_expression_l17_17912


namespace no_55_rooms_l17_17438

theorem no_55_rooms 
  (count_roses count_carnations count_chrysanthemums : ℕ)
  (rooms_with_CC rooms_with_CR rooms_with_HR : ℕ)
  (at_least_one_bouquet_in_each_room: ∀ (room: ℕ), room > 0)
  (total_rooms : ℕ)
  (h_bouquets : count_roses = 30 ∧ count_carnations = 20 ∧ count_chrysanthemums = 10)
  (h_overlap_conditions: rooms_with_CC = 2 ∧ rooms_with_CR = 3 ∧ rooms_with_HR = 4):
  (total_rooms != 55) :=
sorry

end no_55_rooms_l17_17438


namespace trigonometric_identity_l17_17976

theorem trigonometric_identity (α : ℝ) 
  (h : Real.sin α = 1 / 3) : 
  Real.cos (Real.pi / 2 + α) = - 1 / 3 := 
by
  sorry

end trigonometric_identity_l17_17976


namespace find_S3m_l17_17622
  
-- Arithmetic sequence with given properties
variable (m : ℕ)
variable (S : ℕ → ℕ)
variable (a : ℕ → ℕ)

-- Define the conditions
axiom Sm : S m = 30
axiom S2m : S (2 * m) = 100

-- Problem statement to prove
theorem find_S3m : S (3 * m) = 170 :=
by
  sorry

end find_S3m_l17_17622


namespace total_runs_of_a_b_c_l17_17412

/-- Suppose a, b, and c are the runs scored by three players in a cricket match. The ratios of the runs are given as a : b = 1 : 3 and b : c = 1 : 5. Additionally, c scored 75 runs. Prove that the total runs scored by all of them is 95. -/
theorem total_runs_of_a_b_c (a b c : ℕ) (h1 : a * 3 = b) (h2 : b * 5 = c) (h3 : c = 75) : a + b + c = 95 := 
by sorry

end total_runs_of_a_b_c_l17_17412


namespace select_representatives_l17_17523

theorem select_representatives
  (female_count : ℕ) (male_count : ℕ)
  (female_count_eq : female_count = 4)
  (male_count_eq : male_count = 6) :
  female_count * male_count = 24 := by
  sorry

end select_representatives_l17_17523


namespace radius_increase_50_percent_l17_17997

theorem radius_increase_50_percent 
  (r : ℝ)
  (h1 : 1.5 * r = r + r * 0.5) : 
  (3 * Real.pi * r = 2 * Real.pi * r + (2 * Real.pi * r * 0.5)) ∧
  (2.25 * Real.pi * r^2 = Real.pi * r^2 + (Real.pi * r^2 * 1.25)) := 
sorry

end radius_increase_50_percent_l17_17997


namespace range_of_x_l17_17198

def f (x : ℝ) : ℝ := abs (x - 2)

theorem range_of_x (a b x : ℝ) (a_nonzero : a ≠ 0) (ab_real : a ∈ Set.univ ∧ b ∈ Set.univ) : 
  (|a + b| + |a - b| ≥ |a| • f x) ↔ (0 ≤ x ∧ x ≤ 4) :=
sorry

end range_of_x_l17_17198


namespace sum_of_x_and_y_l17_17195

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 200) (h2 : y = 240) : x + y = 680 :=
by
  sorry

end sum_of_x_and_y_l17_17195


namespace problem_statement_l17_17866

theorem problem_statement {x₁ x₂ : ℝ} (h1 : 3 * x₁^2 - 9 * x₁ - 21 = 0) (h2 : 3 * x₂^2 - 9 * x₂ - 21 = 0) :
  (3 * x₁ - 4) * (6 * x₂ - 8) = -202 := sorry

end problem_statement_l17_17866


namespace solution_set_l17_17288

open Nat

def is_solution (a b c : ℕ) : Prop :=
  a ^ (b + 20) * (c - 1) = c ^ (b + 21) - 1

theorem solution_set (a b c : ℕ) : 
  (is_solution a b c) ↔ ((c = 0 ∧ a = 1) ∨ (c = 1)) := 
sorry

end solution_set_l17_17288


namespace max_expr_on_circle_l17_17032

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 4 * x - 6 * y + 4 = 0

noncomputable def expr (x y : ℝ) : ℝ :=
  3 * x - 4 * y

theorem max_expr_on_circle : 
  ∃ (x y : ℝ), circle_eq x y ∧ ∀ (x' y' : ℝ), circle_eq x' y' → expr x y ≤ expr x' y' :=
sorry

end max_expr_on_circle_l17_17032


namespace angle_mul_add_proof_solve_equation_proof_l17_17207

-- For (1)
def angle_mul_add_example : Prop :=
  let a := 34 * 3600 + 25 * 60 + 20 -- 34°25'20'' to seconds
  let b := 35 * 60 + 42 * 60        -- 35°42' to total minutes
  let result := a * 3 + b * 60      -- Multiply a by 3 and convert b to seconds
  let final_result := result / 3600 -- Convert back to degrees
  final_result = 138 + (58 / 60)

-- For (2)
def solve_equation_example : Prop :=
  ∀ x : ℚ, (x + 1) / 2 - 1 = (2 - 3 * x) / 3 → x = 1 / 9

theorem angle_mul_add_proof : angle_mul_add_example := sorry

theorem solve_equation_proof : solve_equation_example := sorry

end angle_mul_add_proof_solve_equation_proof_l17_17207


namespace inequality_solution_set_l17_17185

theorem inequality_solution_set :
  {x : ℝ | x ≠ 0 ∧ x ≠ 2 ∧ (2 * x / (x - 2) + (x + 3) / (3 * x) ≥ 4)} 
  = {x : ℝ | (0 < x ∧ x ≤ 1/5) ∨ (2 < x ∧ x ≤ 6)} := 
by {
  sorry
}

end inequality_solution_set_l17_17185


namespace value_of_y_l17_17434

theorem value_of_y (x y : ℤ) (h1 : x^2 = y - 5) (h2 : x = 7) : y = 54 := by
  sorry

end value_of_y_l17_17434


namespace red_paint_quarts_l17_17917

theorem red_paint_quarts (r g w : ℕ) (ratio_rw : r * 5 = w * 4) (w_quarts : w = 15) : r = 12 :=
by 
  -- We provide the skeleton of the proof here: the detailed steps are skipped (as instructed).
  sorry

end red_paint_quarts_l17_17917


namespace fish_population_l17_17919

theorem fish_population (x : ℕ) : 
  (1: ℝ) / 45 = (100: ℝ) / ↑x -> x = 1125 :=
by
  sorry

end fish_population_l17_17919


namespace nate_total_distance_l17_17079

def length_field : ℕ := 168
def distance_8s : ℕ := 4 * length_field
def additional_distance : ℕ := 500
def total_distance : ℕ := distance_8s + additional_distance

theorem nate_total_distance : total_distance = 1172 := by
  sorry

end nate_total_distance_l17_17079


namespace original_price_of_shoes_l17_17297

theorem original_price_of_shoes (P : ℝ) (h1 : 0.80 * P = 480) : P = 600 := 
by
  sorry

end original_price_of_shoes_l17_17297


namespace cindy_correct_result_l17_17826

theorem cindy_correct_result (x : ℝ) (h: (x - 7) / 5 = 27) : (x - 5) / 7 = 20 :=
by
  sorry

end cindy_correct_result_l17_17826


namespace find_value_of_f_l17_17040

axiom f : ℝ → ℝ

theorem find_value_of_f :
  (∀ x : ℝ, f (Real.cos x) = Real.sin (3 * x)) →
  f (Real.sin (Real.pi / 9)) = -1 / 2 :=
sorry

end find_value_of_f_l17_17040


namespace time_to_fill_bottle_l17_17149

-- Definitions
def flow_rate := 500 / 6 -- mL per second
def volume := 250 -- mL

-- Target theorem
theorem time_to_fill_bottle (r : ℝ) (v : ℝ) (t : ℝ) (h : r = flow_rate) (h2 : v = volume) : t = 3 :=
by
  sorry

end time_to_fill_bottle_l17_17149


namespace evaluate_expression_l17_17293

theorem evaluate_expression (x : ℕ) (h : x = 5) : 2 * x ^ 2 + 5 = 55 := by
  sorry

end evaluate_expression_l17_17293


namespace exists_i_with_α_close_to_60_l17_17886

noncomputable def α : ℕ → ℝ := sorry  -- Placeholder for the function α

theorem exists_i_with_α_close_to_60 :
  ∃ i : ℕ, abs (α i - 60) < 1
:= sorry

end exists_i_with_α_close_to_60_l17_17886


namespace workbook_problems_l17_17050

theorem workbook_problems (P : ℕ)
  (h1 : (1/2 : ℚ) * P = (1/2 : ℚ) * P)
  (h2 : (1/4 : ℚ) * P = (1/4 : ℚ) * P)
  (h3 : (1/6 : ℚ) * P = (1/6 : ℚ) * P)
  (h4 : ((1/2 : ℚ) * P + (1/4 : ℚ) * P + (1/6 : ℚ) * P + 20 = P)) : 
  P = 240 :=
sorry

end workbook_problems_l17_17050


namespace union_complement_eq_complement_intersection_eq_l17_17726

-- Define the universal set U and sets A, B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 5, 7}

-- Theorem 1: A ∪ (U \ B) = {2, 4, 5, 6}
theorem union_complement_eq : A ∪ (U \ B) = {2, 4, 5, 6} := by
  sorry

-- Theorem 2: U \ (A ∩ B) = {1, 2, 3, 4, 6, 7}
theorem complement_intersection_eq : U \ (A ∩ B) = {1, 2, 3, 4, 6, 7} := by
  sorry

end union_complement_eq_complement_intersection_eq_l17_17726


namespace S_30_zero_l17_17178

variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {n : ℕ} 

-- Definitions corresponding to the conditions
def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a_n n = a1 + d * n

def sum_arithmetic_sequence (S : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, S n = n * a1 + (n * (n - 1) / 2) * d
  
-- The given conditions
axiom S_eq (S_10 S_20 : ℝ) : S 10 = S 20

-- The theorem we need to prove
theorem S_30_zero (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h_seq : arithmetic_sequence a_n)
  (h_sum : sum_arithmetic_sequence S a_n)
  (h_eq : S 10 = S 20) :
  S 30 = 0 :=
sorry

end S_30_zero_l17_17178


namespace factorize_expression_l17_17061

theorem factorize_expression (a b : ℤ) (h1 : 3 * b + a = -1) (h2 : a * b = -18) : a - b = -11 :=
by
  sorry

end factorize_expression_l17_17061


namespace incorrect_conclusion_l17_17924

def y (x : ℝ) : ℝ := -2 * x + 3

theorem incorrect_conclusion : ∀ (x : ℝ), y x = 0 → x ≠ 0 := 
by
  sorry

end incorrect_conclusion_l17_17924


namespace jinsu_work_per_hour_l17_17364

theorem jinsu_work_per_hour (t : ℝ) (h : t = 4) : (1 / t = 1 / 4) :=
by {
    sorry
}

end jinsu_work_per_hour_l17_17364


namespace jack_years_after_son_death_l17_17702

noncomputable def jackAdolescenceTime (L : Real) : Real := (1 / 6) * L
noncomputable def jackFacialHairTime (L : Real) : Real := (1 / 12) * L
noncomputable def jackMarriageTime (L : Real) : Real := (1 / 7) * L
noncomputable def jackSonBornTime (L : Real) (marriageTime : Real) : Real := marriageTime + 5
noncomputable def jackSonLifetime (L : Real) : Real := (1 / 2) * L
noncomputable def jackSonDeathTime (bornTime : Real) (sonLifetime : Real) : Real := bornTime + sonLifetime
noncomputable def yearsAfterSonDeath (L : Real) (sonDeathTime : Real) : Real := L - sonDeathTime

theorem jack_years_after_son_death : 
  yearsAfterSonDeath 84 
    (jackSonDeathTime (jackSonBornTime 84 (jackMarriageTime 84)) (jackSonLifetime 84)) = 4 :=
by
  sorry

end jack_years_after_son_death_l17_17702


namespace problem1_problem2_l17_17759

-- Problem 1
theorem problem1 (f : ℝ → ℝ) (x : ℝ) (h : ∀ x, f x = abs (x - 1)) :
  f x ≥ (1/2) * (x + 1) ↔ (x ≤ 1/3) ∨ (x ≥ 3) :=
sorry

-- Problem 2
theorem problem2 (g : ℝ → ℝ) (A : Set ℝ) (a : ℝ) 
  (h1 : ∀ x, g x = abs (x - a) - abs (x - 2))
  (h2 : A ⊆ Set.Icc (-1 : ℝ) 3) :
  (1 ≤ a ∧ a < 2) ∨ (2 ≤ a ∧ a ≤ 3) :=
sorry

end problem1_problem2_l17_17759


namespace simplify_expression_l17_17581

theorem simplify_expression (x : ℝ) : 7 * x + 15 - 3 * x + 2 = 4 * x + 17 := 
by sorry

end simplify_expression_l17_17581


namespace shop_owner_percentage_profit_l17_17182

theorem shop_owner_percentage_profit
  (cp : ℝ)  -- cost price of 1 kg
  (cheat_buy : ℝ) -- cheat percentage when buying
  (cheat_sell : ℝ) -- cheat percentage when selling
  (h_cp : cp = 100) -- cost price is $100
  (h_cheat_buy : cheat_buy = 15) -- cheat by 15% when buying
  (h_cheat_sell : cheat_sell = 20) -- cheat by 20% when selling
  :
  let weight_bought := 1 + (cheat_buy / 100)
  let weight_sold := 1 - (cheat_sell / 100)
  let real_selling_price_per_kg := cp / weight_sold
  let total_selling_price := weight_bought * real_selling_price_per_kg
  let profit := total_selling_price - cp
  let percentage_profit := (profit / cp) * 100
  percentage_profit = 43.75 := 
by
  sorry

end shop_owner_percentage_profit_l17_17182


namespace union_sets_l17_17122

noncomputable def A : Set ℝ := {x | (x + 1) * (x - 2) < 0}
noncomputable def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}
noncomputable def C : Set ℝ := {x | -1 < x ∧ x ≤ 3}

theorem union_sets (A : Set ℝ) (B : Set ℝ) : (A ∪ B = C) := by
  sorry

end union_sets_l17_17122


namespace equivalent_expr_l17_17148

theorem equivalent_expr (a y : ℝ) (ha : a ≠ 0) (hy : y ≠ a ∧ y ≠ -a) :
  ( (a / (a + y) + y / (a - y)) / ( y / (a + y) - a / (a - y)) ) = -1 :=
by
  sorry

end equivalent_expr_l17_17148


namespace mass_percentage_Ca_in_CaI2_l17_17874

noncomputable def molar_mass_Ca : ℝ := 40.08
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_CaI2 : ℝ := molar_mass_Ca + 2 * molar_mass_I

theorem mass_percentage_Ca_in_CaI2 :
  (molar_mass_Ca / molar_mass_CaI2) * 100 = 13.63 :=
by
  sorry

end mass_percentage_Ca_in_CaI2_l17_17874


namespace number_of_girls_l17_17344

variable (N n g : ℕ)
variable (h1 : N = 1600)
variable (h2 : n = 200)
variable (h3 : g = 95)

theorem number_of_girls (G : ℕ) (h : g * N = G * n) : G = 760 :=
by sorry

end number_of_girls_l17_17344


namespace range_of_a_for_monotonically_decreasing_l17_17028

noncomputable def f (a x: ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - 2 * x

theorem range_of_a_for_monotonically_decreasing (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (1/x - a*x - 2 < 0)) ↔ (a ∈ Set.Ioi (-1)) := 
sorry

end range_of_a_for_monotonically_decreasing_l17_17028


namespace common_ratio_of_geometric_series_l17_17531

theorem common_ratio_of_geometric_series (a1 a2 a3 : ℚ) (h1 : a1 = -4 / 7)
                                         (h2 : a2 = 14 / 3) (h3 : a3 = -98 / 9) :
  ∃ r : ℚ, r = a2 / a1 ∧ r = a3 / a2 ∧ r = -49 / 6 :=
by
  use -49 / 6
  sorry

end common_ratio_of_geometric_series_l17_17531


namespace bottles_per_case_l17_17946

theorem bottles_per_case (total_bottles : ℕ) (total_cases : ℕ) (h1 : total_bottles = 60000) (h2 : total_cases = 12000) :
  total_bottles / total_cases = 5 :=
by
  -- Using the given problem, so steps from the solution are not required here
  sorry

end bottles_per_case_l17_17946


namespace total_passengers_landed_l17_17701

theorem total_passengers_landed (on_time late : ℕ) (h_on_time : on_time = 14507) (h_late : late = 213) :
  on_time + late = 14720 :=
by
  sorry

end total_passengers_landed_l17_17701


namespace not_solvable_equations_l17_17042

theorem not_solvable_equations :
  ¬(∃ x : ℝ, (x - 5) ^ 2 = -1) ∧ ¬(∃ x : ℝ, |2 * x| + 3 = 0) :=
by
  sorry

end not_solvable_equations_l17_17042


namespace find_circle_radius_l17_17280

/-- Eight congruent copies of the parabola y = x^2 are arranged in the plane so that each vertex 
is tangent to a circle, and each parabola is tangent to its two neighbors at an angle of 45°.
Find the radius of the circle. -/

theorem find_circle_radius
  (r : ℝ)
  (h_tangent_to_circle : ∀ (x : ℝ), (x^2 + r) = x → x^2 - x + r = 0)
  (h_single_tangent_point : ∀ (x : ℝ), (x^2 - x + r = 0) → ((1 : ℝ)^2 - 4 * 1 * r = 0)) :
  r = 1/4 :=
by
  -- the proof would go here
  sorry

end find_circle_radius_l17_17280


namespace express_in_scientific_notation_l17_17977

def scientific_notation_of_160000 : Prop :=
  160000 = 1.6 * 10^5

theorem express_in_scientific_notation : scientific_notation_of_160000 :=
  sorry

end express_in_scientific_notation_l17_17977


namespace prove_a_plus_b_l17_17027

-- Defining the function f(x)
def f (a b x: ℝ) : ℝ := a * x^2 + b * x

-- The given conditions
variable (a b : ℝ)
variable (h1 : f a b (a - 1) = f a b (2 * a))
variable (h2 : ∀ x : ℝ, f a b x = f a b (-x))

-- The objective is to show a + b = 1/3
theorem prove_a_plus_b (a b : ℝ) (h1 : f a b (a - 1) = f a b (2 * a)) (h2 : ∀ x : ℝ, f a b x = f a b (-x)) :
  a + b = 1 / 3 := 
sorry

end prove_a_plus_b_l17_17027


namespace proof_problem_l17_17277

-- Define sets A and B according to the given conditions
def A : Set ℝ := { x | x ≥ -1 }
def B : Set ℝ := { x | x > 2 }
def complement_B : Set ℝ := { x | ¬ (x > 2) }  -- Complement of B

-- Remaining intersection expression
def intersect_expr : Set ℝ := { x | x ≥ -1 ∧ x ≤ 2 }

-- Statement to prove
theorem proof_problem : (A ∩ complement_B) = intersect_expr :=
sorry

end proof_problem_l17_17277


namespace find_f_1998_l17_17890

noncomputable def f : ℝ → ℝ := sorry -- Define f as a noncomputable function

theorem find_f_1998 (x : ℝ) (h1 : ∀ x, f (x +1) = f x - 1) (h2 : f 1 = 3997) : f 1998 = 2000 :=
  sorry

end find_f_1998_l17_17890


namespace opposite_difference_five_times_l17_17320

variable (a b : ℤ) -- Using integers for this example

theorem opposite_difference_five_times (a b : ℤ) : (-a - 5 * b) = -(a) - (5 * b) := 
by
  -- The proof details would be filled in here
  sorry

end opposite_difference_five_times_l17_17320


namespace cone_volume_is_3_6_l17_17786

-- Define the given conditions
def is_maximum_volume_cone_with_cutoff (cone_volume cutoff_volume : ℝ) : Prop :=
  cutoff_volume = 2 * cone_volume

def volume_difference (cutoff_volume cone_volume difference : ℝ) : Prop :=
  cutoff_volume - cone_volume = difference

-- The theorem to prove the volume of the cone
theorem cone_volume_is_3_6 
  (cone_volume cutoff_volume difference: ℝ)  
  (h1: is_maximum_volume_cone_with_cutoff cone_volume cutoff_volume)
  (h2: volume_difference cutoff_volume cone_volume 3.6) 
  : cone_volume = 3.6 :=
sorry

end cone_volume_is_3_6_l17_17786


namespace sqrt_164_between_12_and_13_l17_17404

theorem sqrt_164_between_12_and_13 : 12 < Real.sqrt 164 ∧ Real.sqrt 164 < 13 :=
sorry

end sqrt_164_between_12_and_13_l17_17404


namespace zero_in_interval_l17_17720

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem zero_in_interval : ∃ x0, f x0 = 0 ∧ 2 < x0 ∧ x0 < 3 :=
by
  have h_cont : Continuous f := sorry -- f is continuous (can be proven using the continuity of log and linear functions)
  have h_eval1 : f 2 < 0 := sorry -- f(2) = ln(2) - 6 + 4 < 0
  have h_eval2 : f 3 > 0 := sorry -- f(3) = ln(3) - 6 + 6 > 0
  -- By the Intermediate Value Theorem, since f is continuous and changes signs between (2, 3), there exists a zero x0 in (2, 3).
  exact sorry

end zero_in_interval_l17_17720


namespace carol_carrots_l17_17620

def mother_picked := 16
def good_carrots := 38
def bad_carrots := 7
def total_carrots := good_carrots + bad_carrots
def carol_picked : Nat := total_carrots - mother_picked

theorem carol_carrots : carol_picked = 29 := by
  sorry

end carol_carrots_l17_17620


namespace smallest_integer_n_condition_l17_17363

theorem smallest_integer_n_condition :
  (∃ n : ℕ, n > 0 ∧ (∀ (m : ℤ), (1 ≤ m ∧ m ≤ 1992) → (∃ (k : ℤ), (m : ℚ) / 1993 < k / n ∧ k / n < (m + 1 : ℚ) / 1994))) ↔ n = 3987 :=
sorry

end smallest_integer_n_condition_l17_17363


namespace polygon_sides_eq_2023_l17_17092

theorem polygon_sides_eq_2023 (n : ℕ) (h : n - 2 = 2021) : n = 2023 :=
sorry

end polygon_sides_eq_2023_l17_17092


namespace trig_identity_l17_17809

noncomputable def tan_eq_neg_4_over_3 (theta : ℝ) : Prop := 
  Real.tan theta = -4 / 3

theorem trig_identity (theta : ℝ) (h : tan_eq_neg_4_over_3 theta) : 
  (Real.cos (π / 2 + θ) - Real.sin (-π - θ)) / (Real.cos (11 * π / 2 - θ) + Real.sin (9 * π / 2 + θ)) = 8 / 7 :=
by
  sorry

end trig_identity_l17_17809


namespace correct_option_l17_17717

variable (p q : Prop)

/-- If only one of p and q is true, then p or q is a true proposition. -/
theorem correct_option (h : (p ∧ ¬ q) ∨ (¬ p ∧ q)) : p ∨ q :=
by sorry

end correct_option_l17_17717


namespace minimum_AB_l17_17646

noncomputable def shortest_AB (a : ℝ) : ℝ :=
  let x := (Real.sqrt 3) / 4 * a
  x

theorem minimum_AB (a : ℝ) : ∃ x, (x = (Real.sqrt 3) / 4 * a) ∧ ∀ y, (y = (Real.sqrt 3) / 4 * a) → shortest_AB a = x :=
by
  sorry

end minimum_AB_l17_17646


namespace flower_garden_mystery_value_l17_17722

/-- Prove the value of "花园探秘" given the arithmetic sum conditions and unique digit mapping. -/
theorem flower_garden_mystery_value :
  ∀ (shu_hua_hua_yuan : ℕ) (wo_ai_tan_mi : ℕ),
  shu_hua_hua_yuan + 2011 = wo_ai_tan_mi →
  (∃ (hua yuan tan mi : ℕ),
    0 ≤ hua ∧ hua < 10 ∧
    0 ≤ yuan ∧ yuan < 10 ∧
    0 ≤ tan ∧ tan < 10 ∧
    0 ≤ mi ∧ mi < 10 ∧
    hua ≠ yuan ∧ hua ≠ tan ∧ hua ≠ mi ∧
    yuan ≠ tan ∧ yuan ≠ mi ∧ tan ≠ mi ∧
    shu_hua_hua_yuan = hua * 1000 + yuan * 100 + tan * 10 + mi ∧
    wo_ai_tan_mi = 9713) := sorry

end flower_garden_mystery_value_l17_17722


namespace percent_of_d_is_e_l17_17932

variable (a b c d e : ℝ)
variable (h1 : d = 0.40 * a)
variable (h2 : d = 0.35 * b)
variable (h3 : e = 0.50 * b)
variable (h4 : e = 0.20 * c)
variable (h5 : c = 0.30 * a)
variable (h6 : c = 0.25 * b)

theorem percent_of_d_is_e : (e / d) * 100 = 15 :=
by sorry

end percent_of_d_is_e_l17_17932


namespace sum_series_and_convergence_l17_17254

theorem sum_series_and_convergence (x : ℝ) (h : -1 < x ∧ x < 1) :
  ∑' n, (n + 6) * x^(7 * n) = (6 - 5 * x^7) / (1 - x^7)^2 :=
by
  sorry

end sum_series_and_convergence_l17_17254


namespace prove_equation_C_l17_17044

theorem prove_equation_C (m : ℝ) : -(m - 2) = -m + 2 := 
  sorry

end prove_equation_C_l17_17044


namespace intersection_line_l17_17286

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 3*x - y = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + y = 0

-- Define the line that we need to prove as the intersection
def line (x y : ℝ) : Prop := x - 2*y = 0

-- The theorem to prove
theorem intersection_line (x y : ℝ) : circle1 x y ∧ circle2 x y → line x y :=
by
  sorry

end intersection_line_l17_17286


namespace people_remaining_at_end_l17_17307

def total_people_start : ℕ := 600
def girls_start : ℕ := 240
def boys_start : ℕ := total_people_start - girls_start
def boys_left_early : ℕ := boys_start / 4
def girls_left_early : ℕ := girls_start / 8
def total_left_early : ℕ := boys_left_early + girls_left_early
def people_remaining : ℕ := total_people_start - total_left_early

theorem people_remaining_at_end : people_remaining = 480 := by
  sorry

end people_remaining_at_end_l17_17307


namespace find_number_l17_17179

theorem find_number (x : ℝ) (h : x / 2 = x - 5) : x = 10 :=
by
  sorry

end find_number_l17_17179


namespace radius_of_sphere_with_surface_area_4pi_l17_17271

noncomputable def sphere_radius (surface_area: ℝ) : ℝ :=
  sorry

theorem radius_of_sphere_with_surface_area_4pi :
  sphere_radius (4 * Real.pi) = 1 :=
by
  sorry

end radius_of_sphere_with_surface_area_4pi_l17_17271


namespace points_scored_fourth_game_l17_17718

-- Define the conditions
def avg_score_3_games := 18
def avg_score_4_games := 17
def games_played_3 := 3
def games_played_4 := 4

-- Calculate total points after 3 games
def total_points_3_games := avg_score_3_games * games_played_3

-- Calculate total points after 4 games
def total_points_4_games := avg_score_4_games * games_played_4

-- Define a theorem to prove the points scored in the fourth game
theorem points_scored_fourth_game :
  total_points_4_games - total_points_3_games = 14 :=
by
  sorry

end points_scored_fourth_game_l17_17718


namespace divides_y_l17_17410

theorem divides_y
  (x y : ℤ)
  (h1 : 2 * x + 1 ∣ 8 * y) : 
  2 * x + 1 ∣ y :=
sorry

end divides_y_l17_17410


namespace polynomial_factorization_l17_17298

theorem polynomial_factorization :
  ∀ (a b c : ℝ),
    a * (b - c) ^ 4 + b * (c - a) ^ 4 + c * (a - b) ^ 4 =
    (a - b) * (b - c) * (c - a) * (a + b + c) :=
  by
    intro a b c
    sorry

end polynomial_factorization_l17_17298


namespace solution_set_l17_17335

-- Define determinant operation on 2x2 matrices
def determinant (a b c d : ℝ) := a * d - b * c

-- Define the condition inequality
def condition (x : ℝ) : Prop :=
  determinant x 3 (-x) x < determinant 2 0 1 2

-- Prove that the solution to the condition is -4 < x < 1
theorem solution_set : {x : ℝ | condition x} = {x : ℝ | -4 < x ∧ x < 1} :=
by
  sorry

end solution_set_l17_17335


namespace sequence_eq_third_term_l17_17329

theorem sequence_eq_third_term 
  (p : ℤ → ℤ)
  (a : ℕ → ℤ)
  (n : ℕ) (h₁ : n > 2)
  (h₂ : a 2 = p (a 1))
  (h₃ : a 3 = p (a 2))
  (h₄ : ∀ k, 4 ≤ k ∧ k ≤ n → a k = p (a (k - 1)))
  (h₅ : a 1 = p (a n))
  : a 1 = a 3 :=
sorry

end sequence_eq_third_term_l17_17329


namespace soda_cost_proof_l17_17319

-- Define the main facts about the weeds
def weeds_flower_bed : ℕ := 11
def weeds_vegetable_patch : ℕ := 14
def weeds_grass : ℕ := 32 / 2  -- Only half the weeds in the grass

-- Define the earning rate
def earning_per_weed : ℕ := 6

-- Define the total earnings and the remaining money conditions
def total_earnings : ℕ := (weeds_flower_bed + weeds_vegetable_patch + weeds_grass) * earning_per_weed
def remaining_money : ℕ := 147

-- Define the cost of the soda
def cost_of_soda : ℕ := total_earnings - remaining_money

-- Problem statement: Prove that the cost of the soda is 99 cents
theorem soda_cost_proof : cost_of_soda = 99 := by
  sorry

end soda_cost_proof_l17_17319


namespace length_of_square_side_l17_17597

-- Definitions based on conditions
def perimeter_of_triangle : ℝ := 46
def total_perimeter : ℝ := 78
def perimeter_of_square : ℝ := total_perimeter - perimeter_of_triangle

-- Lean statement for the problem
theorem length_of_square_side : perimeter_of_square / 4 = 8 := by
  sorry

end length_of_square_side_l17_17597


namespace miranda_pillows_l17_17585

-- Define the conditions in the problem
def pounds_per_pillow := 2
def feathers_per_pound := 300
def total_feathers := 3600

-- Define the goal in terms of these conditions
def num_pillows : Nat :=
  (total_feathers / feathers_per_pound) / pounds_per_pillow

-- Prove that the number of pillows Miranda can stuff is 6
theorem miranda_pillows : num_pillows = 6 :=
by
  sorry

end miranda_pillows_l17_17585


namespace chord_length_l17_17937

-- Definitions and conditions for the problem
variables (A D B C G E F : Point)

-- Lengths and radii in the problem
noncomputable def radius : Real := 10
noncomputable def AB : Real := 20
noncomputable def BC : Real := 20
noncomputable def CD : Real := 20

-- Centers of circles
variables (O N P : Circle) (AN ND : Real)

-- Tangent properties and intersection points
variable (tangent_AG : Tangent AG P G)
variable (intersect_AG_N : Intersects AG N E F)

-- Given the geometry setup, prove the length of chord EF.
theorem chord_length (EF_length : Real) :
  EF_length = 2 * Real.sqrt 93.75 := sorry

end chord_length_l17_17937


namespace company_picnic_attendance_l17_17015

theorem company_picnic_attendance :
  ∀ (employees men women men_attending women_attending : ℕ)
  (h_employees : employees = 100)
  (h_men : men = 55)
  (h_women : women = 45)
  (h_men_attending: men_attending = 11)
  (h_women_attending: women_attending = 18),
  (100 * (men_attending + women_attending) / employees) = 29 := 
by
  intros employees men women men_attending women_attending 
         h_employees h_men h_women h_men_attending h_women_attending
  sorry

end company_picnic_attendance_l17_17015


namespace rectangle_area_l17_17295

variable (x y : ℕ)

theorem rectangle_area
  (h1 : (x + 3) * (y - 1) = x * y)
  (h2 : (x - 3) * (y + 2) = x * y) :
  x * y = 36 :=
by
  -- Proof omitted
  sorry

end rectangle_area_l17_17295


namespace sufficient_not_necessary_implies_a_lt_1_l17_17100

theorem sufficient_not_necessary_implies_a_lt_1 {x a : ℝ} (h : ∀ x : ℝ, x > 1 → x > a ∧ ¬(x > a → x > 1)) : a < 1 :=
sorry

end sufficient_not_necessary_implies_a_lt_1_l17_17100


namespace average_of_multiples_of_9_l17_17419

-- Define the problem in Lean
theorem average_of_multiples_of_9 :
  let pos_multiples := [9, 18, 27, 36, 45]
  let neg_multiples := [-9, -18, -27, -36, -45]
  (pos_multiples.sum + neg_multiples.sum) / 2 = 0 :=
by
  sorry

end average_of_multiples_of_9_l17_17419


namespace marys_next_birthday_l17_17670

noncomputable def calculate_marys_age (d j s m TotalAge : ℝ) (H1 : j = 1.15 * d) (H2 : s = 1.30 * d) (H3 : m = 1.25 * s) (H4 : j + d + s + m = TotalAge) : ℝ :=
  m + 1

theorem marys_next_birthday (d j s m TotalAge : ℝ) 
  (H1 : j = 1.15 * d)
  (H2 : s = 1.30 * d)
  (H3 : m = 1.25 * s)
  (H4 : j + d + s + m = TotalAge)
  (H5 : TotalAge = 80) :
  calculate_marys_age d j s m TotalAge H1 H2 H3 H4 = 26 :=
sorry

end marys_next_birthday_l17_17670


namespace A_div_B_l17_17968

noncomputable def A : ℝ := 
  ∑' n, if n % 2 = 0 ∧ n % 4 ≠ 0 then 1 / (n:ℝ)^2 else 0

noncomputable def B : ℝ := 
  ∑' n, if n % 4 = 0 then (-1)^(n / 4 + 1) * 1 / (n:ℝ)^2 else 0

theorem A_div_B : A / B = 17 := by
  sorry

end A_div_B_l17_17968


namespace find_n_l17_17929

theorem find_n (n : ℕ) (h : n > 0) : 
  (3^n + 5^n) % (3^(n-1) + 5^(n-1)) = 0 ↔ n = 1 := 
by sorry

end find_n_l17_17929


namespace find_p_l17_17041

-- Define the conditions for the problem.
-- Random variable \xi follows binomial distribution B(n, p).
axiom binomial_distribution (n : ℕ) (p : ℝ) : Type
variables (ξ : binomial_distribution n p)

-- Given conditions: Eξ = 300 and Dξ = 200.
axiom Eξ (ξ : binomial_distribution n p) : ℝ
axiom Dξ (ξ : binomial_distribution n p) : ℝ

-- Given realizations of expectations and variance.
axiom h1 : Eξ ξ = 300
axiom h2 : Dξ ξ = 200

-- Prove that p = 1/3
theorem find_p (n : ℕ) (p : ℝ) (ξ : binomial_distribution n p)
  (h1 : Eξ ξ = 300) (h2 : Dξ ξ = 200) : p = 1 / 3 :=
sorry

end find_p_l17_17041


namespace minimum_value_l17_17474

theorem minimum_value (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
    (h_condition : (1 / a) + (1 / b) + (1 / c) = 9) : 
    a^3 * b^2 * c ≥ 64 / 729 :=
sorry

end minimum_value_l17_17474


namespace probability_two_cards_diff_suits_l17_17755

def prob_two_cards_diff_suits {deck_size suits cards_per_suit : ℕ} (h1 : deck_size = 40) (h2 : suits = 4) (h3 : cards_per_suit = 10) : ℚ :=
  let total_cards := deck_size
  let cards_same_suit := cards_per_suit - 1
  let cards_diff_suit := total_cards - 1 - cards_same_suit 
  cards_diff_suit / (total_cards - 1)

theorem probability_two_cards_diff_suits (h1 : 40 = 40) (h2 : 4 = 4) (h3 : 10 = 10) :
  prob_two_cards_diff_suits h1 h2 h3 = 10 / 13 :=
by
  sorry

end probability_two_cards_diff_suits_l17_17755


namespace one_inch_cubes_with_two_or_more_painted_faces_l17_17442

def original_cube_length : ℕ := 4

def total_one_inch_cubes : ℕ := original_cube_length ^ 3

def corners_count : ℕ := 8

def edges_minus_corners_count : ℕ := 12 * 2

theorem one_inch_cubes_with_two_or_more_painted_faces
  (painted_faces_on_each_face : ∀ i : ℕ, i < total_one_inch_cubes → ℕ) : 
  ∃ n : ℕ, n = corners_count + edges_minus_corners_count ∧ n = 32 := 
by
  simp only [corners_count, edges_minus_corners_count, total_one_inch_cubes]
  sorry

end one_inch_cubes_with_two_or_more_painted_faces_l17_17442


namespace first_month_sale_l17_17201

theorem first_month_sale (sales_2 : ℕ) (sales_3 : ℕ) (sales_4 : ℕ) (sales_5 : ℕ) (sales_6 : ℕ) (average_sale : ℕ) (total_months : ℕ)
  (H_sales_2 : sales_2 = 6927)
  (H_sales_3 : sales_3 = 6855)
  (H_sales_4 : sales_4 = 7230)
  (H_sales_5 : sales_5 = 6562)
  (H_sales_6 : sales_6 = 5591)
  (H_average_sale : average_sale = 6600)
  (H_total_months : total_months = 6) :
  ∃ (sale_1 : ℕ), sale_1 = 6435 :=
by
  -- placeholder for the proof
  sorry

end first_month_sale_l17_17201


namespace find_c_plus_d_l17_17698

def is_smallest_two_digit_multiple_of_5 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = 5 * k ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ ∃ k', m = 5 * k') → n ≤ m

def is_smallest_three_digit_multiple_of_7 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = 7 * k ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ ∃ k', m = 7 * k') → n ≤ m

theorem find_c_plus_d :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end find_c_plus_d_l17_17698


namespace real_values_of_x_l17_17331

theorem real_values_of_x (x : ℝ) (h : x ≠ 4) :
  (x * (x + 1) / (x - 4)^2 ≥ 15) ↔ (x ≤ 3 ∨ (40/7 < x ∧ x < 4) ∨ x > 4) :=
by sorry

end real_values_of_x_l17_17331


namespace range_of_a_l17_17334

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) := 
sorry

end range_of_a_l17_17334


namespace no_polygon_with_1974_diagonals_l17_17337

theorem no_polygon_with_1974_diagonals :
  ¬ ∃ N : ℕ, N * (N - 3) / 2 = 1974 :=
sorry

end no_polygon_with_1974_diagonals_l17_17337


namespace susan_mean_l17_17435

def susan_scores : List ℝ := [87, 90, 95, 98, 100]

theorem susan_mean :
  (susan_scores.sum) / (susan_scores.length) = 94 := by
  sorry

end susan_mean_l17_17435


namespace michael_robots_l17_17688

-- Conditions
def tom_robots := 3
def times_more := 4

-- Theorem to prove
theorem michael_robots : (times_more * tom_robots) + tom_robots = 15 := by
  sorry

end michael_robots_l17_17688


namespace starting_positions_P0_P1024_l17_17111

noncomputable def sequence_fn (x : ℝ) : ℝ := 4 * x / (x^2 + 1)

def find_starting_positions (n : ℕ) : ℕ := 2^n - 2

theorem starting_positions_P0_P1024 :
  ∃ P0 : ℝ, ∀ n : ℕ, P0 = sequence_fn^[n] P0 → P0 = sequence_fn^[1024] P0 ↔ find_starting_positions 1024 = 2^1024 - 2 :=
sorry

end starting_positions_P0_P1024_l17_17111


namespace largest_three_digit_divisible_by_6_l17_17193

-- Defining what it means for a number to be divisible by 6, 2, and 3
def divisible_by (n d : ℕ) : Prop := ∃ k : ℕ, n = d * k

-- Conditions extracted from the problem
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def last_digit_even (n : ℕ) : Prop := (n % 10) % 2 = 0
def sum_of_digits_divisible_by_3 (n : ℕ) : Prop := ((n / 100) + (n / 10 % 10) + (n % 10)) % 3 = 0

-- Define what it means for a number to be divisible by 6 according to the conditions
def divisible_by_6 (n : ℕ) : Prop := last_digit_even n ∧ sum_of_digits_divisible_by_3 n

-- Prove that 996 is the largest three-digit number that satisfies these conditions
theorem largest_three_digit_divisible_by_6 (n : ℕ) : is_three_digit n ∧ divisible_by_6 n → n ≤ 996 :=
by
    sorry

end largest_three_digit_divisible_by_6_l17_17193


namespace larger_integer_is_21_l17_17655

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end larger_integer_is_21_l17_17655


namespace unique_prime_pair_l17_17709

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_prime_pair :
  ∀ p : ℕ, is_prime p ∧ is_prime (p + 1) → p = 2 := by
  sorry

end unique_prime_pair_l17_17709


namespace triangle_ABC_no_common_factor_l17_17151

theorem triangle_ABC_no_common_factor (a b c : ℕ) (h_coprime: Nat.gcd (Nat.gcd a b) c = 1)
  (h_angleB_eq_2angleC : True) (h_b_lt_600 : b < 600) : False :=
by
  sorry

end triangle_ABC_no_common_factor_l17_17151


namespace price_per_glass_first_day_l17_17214

theorem price_per_glass_first_day (O W : ℝ) (P1 P2 : ℝ) 
  (h1 : O = W) 
  (h2 : P2 = 0.40)
  (revenue_eq : 2 * O * P1 = 3 * O * P2) 
  : P1 = 0.60 := 
by 
  sorry

end price_per_glass_first_day_l17_17214


namespace hypotenuse_length_right_triangle_l17_17909

theorem hypotenuse_length_right_triangle :
  ∃ (x : ℝ), (x > 7) ∧ ((x - 7)^2 + x^2 = (x + 2)^2) ∧ (x + 2 = 17) :=
by {
  sorry
}

end hypotenuse_length_right_triangle_l17_17909


namespace B_finishes_work_in_4_days_l17_17222

-- Define the work rates of A and B
def work_rate_A : ℚ := 1 / 5
def work_rate_B : ℚ := 1 / 10

-- Combined work rate when A and B work together
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Work done by A and B in 2 days
def work_done_in_2_days : ℚ := 2 * combined_work_rate

-- Remaining work after 2 days
def remaining_work : ℚ := 1 - work_done_in_2_days

-- Time B needs to finish the remaining work
def time_for_B_to_finish_remaining_work : ℚ := remaining_work / work_rate_B

theorem B_finishes_work_in_4_days : time_for_B_to_finish_remaining_work = 4 := by
  sorry

end B_finishes_work_in_4_days_l17_17222


namespace elves_closed_eyes_l17_17773

theorem elves_closed_eyes :
  ∃ (age: ℕ → ℕ), -- Function assigning each position an age
  (∀ n, 1 ≤ n ∧ n ≤ 100 → (age n < age ((n % 100) + 1) ∧ age n < age (n - 1 % 100 + 1)) ∨
                          (age n > age ((n % 100) + 1) ∧ age n > age (n - 1 % 100 + 1))) :=
by
  sorry

end elves_closed_eyes_l17_17773


namespace divisor_of_number_l17_17544

theorem divisor_of_number (n d q p : ℤ) 
  (h₁ : n = d * q + 3)
  (h₂ : n ^ 2 = d * p + 3) : 
  d = 6 := 
sorry

end divisor_of_number_l17_17544


namespace determine_value_of_expression_l17_17126

theorem determine_value_of_expression (x y : ℤ) (h : y^2 + 4 * x^2 * y^2 = 40 * x^2 + 817) : 4 * x^2 * y^2 = 3484 :=
sorry

end determine_value_of_expression_l17_17126


namespace probability_of_point_within_two_units_l17_17975

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let area_of_circle := 4 * Real.pi
  let area_of_square := 36
  area_of_circle / area_of_square

theorem probability_of_point_within_two_units :
  probability_within_two_units_of_origin = Real.pi / 9 := 
by
  -- The proof steps are omitted as per the requirements
  sorry

end probability_of_point_within_two_units_l17_17975


namespace solution_of_system_l17_17341

theorem solution_of_system : ∃ x y : ℝ, (2 * x + y = 2) ∧ (x - y = 1) ∧ (x = 1) ∧ (y = 0) := 
by
  sorry

end solution_of_system_l17_17341


namespace area_OMVK_l17_17789

def AreaOfQuadrilateral (S_OKSL S_ONAM S_OMVK : ℝ) : ℝ :=
  let S_ABCD := 4 * (S_OKSL + S_ONAM)
  S_ABCD - S_OKSL - 24 - S_ONAM

theorem area_OMVK {S_OKSL S_ONAM : ℝ} (h_OKSL : S_OKSL = 6) (h_ONAM : S_ONAM = 12) : 
  AreaOfQuadrilateral S_OKSL S_ONAM 30 = 30 :=
by
  sorry

end area_OMVK_l17_17789


namespace exist_unique_xy_solution_l17_17396

theorem exist_unique_xy_solution :
  ∃! (x y : ℝ), x^2 + (1 - y)^2 + (x - y)^2 = 1 / 3 ∧ x = 1 / 3 ∧ y = 2 / 3 :=
by
  sorry

end exist_unique_xy_solution_l17_17396


namespace min_bn_of_arithmetic_sequence_l17_17155

theorem min_bn_of_arithmetic_sequence :
  (∃ n : ℕ, 1 ≤ n ∧ b_n = n + 1 + 7 / n ∧ (∀ m : ℕ, 1 ≤ m → b_m ≥ b_n)) :=
sorry

def a_n (n : ℕ) : ℕ :=
  if n = 0 then 0 else n

def S_n (n : ℕ) : ℕ :=
  if n = 0 then 0 else n * (n + 1) / 2

def b_n (n : ℕ) : ℕ :=
  if n = 0 then 0 else (2 * S_n n + 7) / n

end min_bn_of_arithmetic_sequence_l17_17155


namespace total_cows_in_ranch_l17_17785

def WeThePeopleCows : ℕ := 17
def HappyGoodHealthyFamilyCows : ℕ := 3 * WeThePeopleCows + 2

theorem total_cows_in_ranch : WeThePeopleCows + HappyGoodHealthyFamilyCows = 70 := by
  sorry

end total_cows_in_ranch_l17_17785


namespace video_game_price_l17_17819

theorem video_game_price (total_games not_working_games : ℕ) (total_earnings : ℕ)
  (h1 : total_games = 10) (h2 : not_working_games = 2) (h3 : total_earnings = 32) :
  ((total_games - not_working_games) > 0) →
  (total_earnings / (total_games - not_working_games)) = 4 :=
by
  sorry

end video_game_price_l17_17819


namespace sets_of_bleachers_l17_17662

def totalFans : ℕ := 2436
def fansPerSet : ℕ := 812

theorem sets_of_bleachers (n : ℕ) (h : totalFans = n * fansPerSet) : n = 3 :=
by {
    sorry
}

end sets_of_bleachers_l17_17662


namespace intersection_of_A_and_B_l17_17941

def set_A : Set ℝ := {x | x^2 ≤ 4 * x}
def set_B : Set ℝ := {x | |x| ≥ 2}

theorem intersection_of_A_and_B :
  {x | x ∈ set_A ∧ x ∈ set_B} = {x | 2 ≤ x ∧ x ≤ 4} :=
sorry

end intersection_of_A_and_B_l17_17941


namespace vertical_line_division_l17_17264

theorem vertical_line_division (A B C : ℝ × ℝ)
    (hA : A = (0, 2)) (hB : B = (0, 0)) (hC : C = (6, 0))
    (a : ℝ) (h_area_half : 1 / 2 * 6 * 2 / 2 = 3) :
    a = 3 :=
sorry

end vertical_line_division_l17_17264


namespace timesToFillBottlePerWeek_l17_17960

noncomputable def waterConsumptionPerDay : ℕ := 4 * 5
noncomputable def waterConsumptionPerWeek : ℕ := 7 * waterConsumptionPerDay
noncomputable def bottleCapacity : ℕ := 35

theorem timesToFillBottlePerWeek : 
  waterConsumptionPerWeek / bottleCapacity = 4 := 
by
  sorry

end timesToFillBottlePerWeek_l17_17960


namespace min_pairs_with_same_sum_l17_17369

theorem min_pairs_with_same_sum (n : ℕ) (h1 : n > 0) :
  (∀ weights : Fin n → ℕ, (∀ i, weights i ≤ 21) → (∃ i j k l : Fin n,
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    weights i + weights j = weights k + weights l)) ↔ n ≥ 8 :=
by
  sorry

end min_pairs_with_same_sum_l17_17369


namespace prime_iff_good_fractions_l17_17576

def isGoodFraction (n : ℕ) (a b : ℕ) : Prop := a > 0 ∧ b > 0 ∧ (a + b = n)

def canBeExpressedUsingGoodFractions (n : ℕ) (a b : ℕ) : Prop :=
  ∃ (expressedFraction : ℕ → ℕ → Prop), expressedFraction a b ∧
  ∀ x y, expressedFraction x y → isGoodFraction n x y

theorem prime_iff_good_fractions {n : ℕ} (hn : n > 1) :
  Prime n ↔
    ∀ a b : ℕ, b < n → (a > 0 ∧ b > 0) → canBeExpressedUsingGoodFractions n a b :=
sorry

end prime_iff_good_fractions_l17_17576


namespace total_receipts_l17_17019

theorem total_receipts 
  (x y : ℕ) 
  (h1 : x + y = 64)
  (h2 : y ≥ 8) 
  : 3 * x + 4 * y = 200 := 
by
  sorry

end total_receipts_l17_17019


namespace third_diff_n_cube_is_const_6_third_diff_general_form_is_6_l17_17536

-- Define the first finite difference function
def delta (f : ℕ → ℤ) (n : ℕ) : ℤ := f (n + 1) - f n

-- Define the second finite difference using the first
def delta2 (f : ℕ → ℤ) (n : ℕ) : ℤ := delta (delta f) n

-- Define the third finite difference using the second
def delta3 (f : ℕ → ℤ) (n : ℕ) : ℤ := delta (delta2 f) n

-- Prove the third finite difference of n^3 is 6
theorem third_diff_n_cube_is_const_6 :
  delta3 (fun (n : ℕ) => (n : ℤ)^3) = fun _ => 6 := 
by
  sorry

-- Prove the third finite difference of the general form function is 6
theorem third_diff_general_form_is_6 (a b c : ℤ) :
  delta3 (fun (n : ℕ) => (n : ℤ)^3 + a * (n : ℤ)^2 + b * (n : ℤ) + c) = fun _ => 6 := 
by
  sorry

end third_diff_n_cube_is_const_6_third_diff_general_form_is_6_l17_17536


namespace mrs_taylor_total_payment_l17_17817

-- Declaring the price of items and discounts
def price_tv : ℝ := 750
def price_soundbar : ℝ := 300

def discount_tv : ℝ := 0.15
def discount_soundbar : ℝ := 0.10

-- Total number of each items
def num_tv : ℕ := 2
def num_soundbar : ℕ := 3

-- Total cost calculation after discounts
def total_cost_tv := num_tv * price_tv * (1 - discount_tv)
def total_cost_soundbar := num_soundbar * price_soundbar * (1 - discount_soundbar)
def total_cost := total_cost_tv + total_cost_soundbar

-- The theorem we want to prove
theorem mrs_taylor_total_payment : total_cost = 2085 := by
  -- Skipping the proof
  sorry

end mrs_taylor_total_payment_l17_17817


namespace different_books_read_l17_17275

theorem different_books_read (t_books d_books b_books td_same all_same : ℕ)
  (h_t_books : t_books = 23)
  (h_d_books : d_books = 12)
  (h_b_books : b_books = 17)
  (h_td_same : td_same = 3)
  (h_all_same : all_same = 1) : 
  t_books + d_books + b_books - (td_same + all_same) = 48 := 
by
  sorry

end different_books_read_l17_17275


namespace largest_n_rational_sqrt_l17_17485

theorem largest_n_rational_sqrt : ∃ n : ℕ, 
  (∀ k l : ℤ, k = Int.natAbs (Int.sqrt (n - 100)) ∧ l = Int.natAbs (Int.sqrt (n + 100)) → 
  k + l = 100) ∧ 
  (n = 2501) :=
by
  sorry

end largest_n_rational_sqrt_l17_17485


namespace quadratic_completion_l17_17007

theorem quadratic_completion (a b : ℤ) (h_eq : (x : ℝ) → x^2 - 10 * x + 25 = 0) :
  (∃ a b : ℤ, ∀ x : ℝ, (x + a) ^ 2 = b) → a + b = -5 := by
  sorry

end quadratic_completion_l17_17007


namespace distinct_numbers_div_sum_diff_l17_17377

theorem distinct_numbers_div_sum_diff (n : ℕ) : 
  ∃ (numbers : Fin n → ℕ), 
    ∀ i j, i ≠ j → (numbers i + numbers j) % (numbers i - numbers j) = 0 := 
by
  sorry

end distinct_numbers_div_sum_diff_l17_17377


namespace yellow_tickets_needed_l17_17992

def yellow_from_red (r : ℕ) : ℕ := r / 10
def red_from_blue (b : ℕ) : ℕ := b / 10
def blue_needed (current_blue : ℕ) (additional_blue : ℕ) : ℕ := current_blue + additional_blue
def total_blue_from_tickets (y : ℕ) (r : ℕ) (b : ℕ) : ℕ := (y * 10 * 10) + (r * 10) + b

theorem yellow_tickets_needed (y r b additional_blue : ℕ) (h : total_blue_from_tickets y r b + additional_blue = 1000) :
  yellow_from_red (red_from_blue (total_blue_from_tickets y r b + additional_blue)) = 10 := 
by
  sorry

end yellow_tickets_needed_l17_17992


namespace equation_is_hyperbola_l17_17043

-- Define the equation
def equation (x y : ℝ) : Prop :=
  4 * x^2 - 9 * y^2 + 3 * x = 0

-- Theorem stating that the given equation represents a hyperbola
theorem equation_is_hyperbola : ∀ x y : ℝ, equation x y → (∃ A B : ℝ, A * x^2 - B * y^2 = 1) :=
by
  sorry

end equation_is_hyperbola_l17_17043


namespace first_quadrant_solution_l17_17113

theorem first_quadrant_solution (c : ℝ) :
  (∃ x y : ℝ, x - y = 2 ∧ c * x + y = 3 ∧ 0 < x ∧ 0 < y) ↔ -1 < c ∧ c < 3 / 2 :=
by
  sorry

end first_quadrant_solution_l17_17113


namespace simplify_and_evaluate_l17_17106

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 2) :
  ((m ^ 2 - 9) / (m ^ 2 - 6 * m + 9) - 3 / (m - 3)) / (m ^ 2 / (m - 3)) = Real.sqrt 2 / 2 :=
by {
  -- Proof goes here
  sorry
}

end simplify_and_evaluate_l17_17106


namespace max_area_of_triangle_l17_17571

noncomputable def max_area_triangle (a A : ℝ) : ℝ :=
  let bcsinA := sorry
  1 / 2 * bcsinA

theorem max_area_of_triangle (a A : ℝ) (hab : a = 4) (hAa : A = Real.pi / 3) :
  max_area_triangle a A = 4 * Real.sqrt 3 :=
by
  sorry

end max_area_of_triangle_l17_17571


namespace miles_hiked_first_day_l17_17737

theorem miles_hiked_first_day (total_distance remaining_distance : ℕ)
  (h1 : total_distance = 36)
  (h2 : remaining_distance = 27) :
  total_distance - remaining_distance = 9 :=
by
  sorry

end miles_hiked_first_day_l17_17737


namespace sum_of_squares_l17_17980

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 20) (h2 : ab + bc + ca = 5) : a^2 + b^2 + c^2 = 390 :=
by sorry

end sum_of_squares_l17_17980


namespace original_price_of_cycle_l17_17557

theorem original_price_of_cycle (SP : ℝ) (gain_percent : ℝ) (original_price : ℝ) 
  (hSP : SP = 1260) (hgain : gain_percent = 0.40) (h_eq : SP = original_price * (1 + gain_percent)) :
  original_price = 900 :=
by
  sorry

end original_price_of_cycle_l17_17557


namespace inversely_proportional_percentage_change_l17_17974

variable {x y k : ℝ}
variable (a b : ℝ)

/-- Given that x and y are positive numbers and inversely proportional,
if x increases by a% and y decreases by b%, then b = 100a / (100 + a) -/
theorem inversely_proportional_percentage_change
  (hx : 0 < x) (hy : 0 < y) (hinv : y = k / x)
  (ha : 0 < a) (hb : 0 < b)
  (hchange : ((1 + a / 100) * x) * ((1 - b / 100) * y) = k) :
  b = 100 * a / (100 + a) :=
sorry

end inversely_proportional_percentage_change_l17_17974


namespace xy_sum_143_l17_17656

theorem xy_sum_143 (x y : ℕ) (h1 : x < 30) (h2 : y < 30) (h3 : x + y + x * y = 143) (h4 : 0 < x) (h5 : 0 < y) :
  x + y = 22 ∨ x + y = 23 ∨ x + y = 24 :=
by
  sorry

end xy_sum_143_l17_17656


namespace proof_problem_l17_17117

noncomputable def A : Set ℝ := { x | x^2 - 4 = 0 }
noncomputable def B : Set ℝ := { y | ∃ x, y = x^2 - 4 }

theorem proof_problem :
  (A ∩ B = A) ∧ (A ∪ B = B) :=
by {
  sorry
}

end proof_problem_l17_17117


namespace trigonometric_identity_l17_17652

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180))
  = (4 * Real.sin (10 * Real.pi / 180) / Real.sin (40 * Real.pi / 180)) :=
by sorry

end trigonometric_identity_l17_17652


namespace solve_quadratic_equation_l17_17834

theorem solve_quadratic_equation:
  (∀ x : ℝ, (8 * x^2 + 52 * x + 4) / (3 * x + 13) = 2 * x + 3 →
    x = ( -17 + Real.sqrt 569) / 4 ∨ x = ( -17 - Real.sqrt 569) / 4) :=
by
  sorry

end solve_quadratic_equation_l17_17834


namespace liam_finishes_on_wednesday_l17_17561

theorem liam_finishes_on_wednesday :
  let start_day := 3  -- Wednesday, where 0 represents Sunday
  let total_books := 20
  let total_days := (total_books * (total_books + 1)) / 2
  (total_days % 7) = 0 :=
by sorry

end liam_finishes_on_wednesday_l17_17561


namespace find_x_range_l17_17248

theorem find_x_range (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -2) (h3 : 2 * x - 5 > 0) : x > 5 / 2 :=
by
  sorry

end find_x_range_l17_17248


namespace prime_number_p_squared_divides_sum_or_cube_divides_sum_of_cubes_l17_17769

variable {p a b : ℤ}

theorem prime_number_p_squared_divides_sum_or_cube_divides_sum_of_cubes
  (hp : Prime p) (hp_ne_3 : p ≠ 3)
  (h1 : p ∣ (a + b)) (h2 : p^2 ∣ (a^3 + b^3)) :
  p^2 ∣ (a + b) ∨ p^3 ∣ (a^3 + b^3) :=
sorry

end prime_number_p_squared_divides_sum_or_cube_divides_sum_of_cubes_l17_17769


namespace trapezoidal_section_length_l17_17206

theorem trapezoidal_section_length 
  (total_area : ℝ) 
  (rectangular_area : ℝ) 
  (parallel_side1 : ℝ) 
  (parallel_side2 : ℝ) 
  (trapezoidal_area : ℝ)
  (H1 : total_area = 55)
  (H2 : rectangular_area = 30)
  (H3 : parallel_side1 = 3)
  (H4 : parallel_side2 = 6)
  (H5 : trapezoidal_area = total_area - rectangular_area) :
  (trapezoidal_area = 25) → 
  (1/2 * (parallel_side1 + parallel_side2) * L = trapezoidal_area) →
  L = 25 / 4.5 :=
by
  sorry

end trapezoidal_section_length_l17_17206


namespace number_of_prime_factors_30_factorial_l17_17981

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l17_17981


namespace area_of_region_AGF_l17_17925

theorem area_of_region_AGF 
  (ABCD_area : ℝ)
  (hABCD_area : ABCD_area = 160)
  (E F G : ℝ)
  (hE_midpoint : E = (A + B) / 2)
  (hF_midpoint : F = (C + D) / 2)
  (EF_divides : EF_area = ABCD_area / 2)
  (hEF_midpoint : G = (E + F) / 2)
  (AG_divides_upper : AG_area = EF_area / 2) :
  AGF_area = 40 := 
sorry

end area_of_region_AGF_l17_17925


namespace unit_A_saplings_l17_17503

theorem unit_A_saplings 
  (Y B D J : ℕ)
  (h1 : J = 2 * Y + 20)
  (h2 : J = 3 * B + 24)
  (h3 : J = 5 * D - 45)
  (h4 : J + Y + B + D = 2126) :
  J = 1050 :=
by sorry

end unit_A_saplings_l17_17503


namespace prove_R_value_l17_17798

noncomputable def geometric_series (Q : ℕ) : ℕ :=
  (2^(Q + 1) - 1)

noncomputable def R (F : ℕ) : ℝ :=
  Real.sqrt (Real.log (1 + F) / Real.log 2)

theorem prove_R_value :
  let F := geometric_series 120
  R F = 11 :=
by
  sorry

end prove_R_value_l17_17798


namespace part_I_part_II_l17_17770

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x + 5|
def g (x : ℝ) : ℝ := |x - 1| - |2 * x|

-- Part I
theorem part_I : ∀ x : ℝ, g x > -4 → -5 < x ∧ x < -3 :=
by
  sorry

-- Part II
theorem part_II : 
  (∃ x1 x2 : ℝ, f x1 a = g x2) → -6 ≤ a ∧ a ≤ -4 :=
by
  sorry

end part_I_part_II_l17_17770


namespace cost_of_3000_pencils_l17_17953

theorem cost_of_3000_pencils (pencils_per_box : ℕ) (cost_per_box : ℝ) (pencils_needed : ℕ) (unit_cost : ℝ): 
  pencils_per_box = 120 → cost_per_box = 36 → pencils_needed = 3000 → unit_cost = 0.30 →
  (pencils_needed * unit_cost = (3000 : ℝ) * 0.30) :=
by
  intros _ _ _ _
  sorry

end cost_of_3000_pencils_l17_17953


namespace num_ways_choose_pair_of_diff_color_socks_l17_17602

-- Define the numbers of socks of each color
def num_white := 5
def num_brown := 5
def num_blue := 3
def num_black := 3

-- Define the calculation for pairs of different colored socks
def num_pairs_white_brown := num_white * num_brown
def num_pairs_brown_blue := num_brown * num_blue
def num_pairs_white_blue := num_white * num_blue
def num_pairs_white_black := num_white * num_black
def num_pairs_brown_black := num_brown * num_black
def num_pairs_blue_black := num_blue * num_black

-- Define the total number of pairs
def total_pairs := num_pairs_white_brown + num_pairs_brown_blue + num_pairs_white_blue + num_pairs_white_black + num_pairs_brown_black + num_pairs_blue_black

-- The theorem to be proved
theorem num_ways_choose_pair_of_diff_color_socks : total_pairs = 94 := by
  -- Since we do not need to include the proof steps, we use sorry
  sorry

end num_ways_choose_pair_of_diff_color_socks_l17_17602


namespace average_attendance_l17_17123

def monday_attendance := 10
def tuesday_attendance := 15
def wednesday_attendance := 10
def thursday_attendance := 10
def friday_attendance := 10
def total_days := 5

theorem average_attendance :
  (monday_attendance + tuesday_attendance + wednesday_attendance + thursday_attendance + friday_attendance) / total_days = 11 :=
by
  sorry

end average_attendance_l17_17123


namespace total_votes_l17_17405

theorem total_votes (emma_votes : ℕ) (vote_fraction : ℚ) (h_emma : emma_votes = 45) (h_fraction : vote_fraction = 3/7) :
  emma_votes = vote_fraction * 105 :=
by {
  sorry
}

end total_votes_l17_17405


namespace gymnastics_team_l17_17315

def number_of_rows (n m k : ℕ) : Prop :=
  n = k * (2 * m + k - 1) / 2

def members_in_first_row (n m k : ℕ) : Prop :=
  number_of_rows n m k ∧ 16 < k

theorem gymnastics_team : ∃ m k : ℕ, members_in_first_row 1000 m k ∧ k = 25 ∧ m = 28 :=
by
  sorry

end gymnastics_team_l17_17315


namespace g_increasing_in_interval_l17_17533

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / 3) * x^3 - a * x^2 + a * x + 2
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 - 2 * a * x + a
noncomputable def f'' (a : ℝ) (x : ℝ) : ℝ := 2 * x - 2 * a

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f'' a x / x
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := 1 - a / (x^2)

theorem g_increasing_in_interval (a : ℝ) (h : a < 1) :
  ∀ x : ℝ, 1 < x → 0 < g' a x := by
  sorry

end g_increasing_in_interval_l17_17533


namespace egg_weight_probability_l17_17552

theorem egg_weight_probability : 
  let P_lt_30 := 0.3
  let P_30_40 := 0.5
  P_lt_30 + P_30_40 ≤ 1 → (1 - (P_lt_30 + P_30_40) = 0.2) := by
  intro h
  sorry

end egg_weight_probability_l17_17552


namespace gcd_217_155_l17_17242

theorem gcd_217_155 : Nat.gcd 217 155 = 1 := by
  sorry

end gcd_217_155_l17_17242


namespace cannot_be_six_l17_17138

theorem cannot_be_six (n r : ℕ) (h_n : n = 6) : 3 * n ≠ 4 * r :=
by
  sorry

end cannot_be_six_l17_17138


namespace total_income_l17_17965

def ron_ticket_price : ℝ := 2.00
def kathy_ticket_price : ℝ := 4.50
def total_tickets : ℕ := 20
def ron_tickets_sold : ℕ := 12

theorem total_income : ron_tickets_sold * ron_ticket_price + (total_tickets - ron_tickets_sold) * kathy_ticket_price = 60.00 := by
  sorry

end total_income_l17_17965


namespace remainder_7_times_10_pow_20_plus_1_pow_20_mod_9_l17_17624

theorem remainder_7_times_10_pow_20_plus_1_pow_20_mod_9 :
  (7 * 10 ^ 20 + 1 ^ 20) % 9 = 8 :=
by
  -- need to note down the known conditions to help guide proof writing.
  -- condition: 1 ^ 20 = 1
  -- condition: 10 % 9 = 1

  sorry

end remainder_7_times_10_pow_20_plus_1_pow_20_mod_9_l17_17624


namespace value_of_af_over_cd_l17_17016

variable (a b c d e f : ℝ)

theorem value_of_af_over_cd :
  a * b * c = 130 ∧
  b * c * d = 65 ∧
  c * d * e = 500 ∧
  d * e * f = 250 →
  (a * f) / (c * d) = 1 :=
by
  sorry

end value_of_af_over_cd_l17_17016


namespace sampling_scheme_exists_l17_17830

theorem sampling_scheme_exists : 
  ∃ (scheme : List ℕ → List (List ℕ)), 
    ∀ (p : List ℕ), p.length = 100 → (scheme p).length = 20 :=
by
  sorry

end sampling_scheme_exists_l17_17830


namespace temperature_relationship_l17_17173

def temperature (t : ℕ) (T : ℕ) :=
  ∀ t < 10, T = 7 * t + 30

-- Proof not required, hence added sorry.
theorem temperature_relationship (t : ℕ) (T : ℕ) (h : t < 10) :
  temperature t T :=
by {
  sorry
}

end temperature_relationship_l17_17173


namespace alternate_interior_angles_equal_l17_17713

-- Defining the parallel lines and the third intersecting line
def Line : Type := sorry  -- placeholder type for a line

-- Predicate to check if lines are parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Predicate to represent a line intersecting another
def intersects (l1 l2 : Line) : Prop := sorry

-- Function to get interior alternate angles formed by the intersection
def alternate_interior_angles (l1 l2 : Line) (l3 : Line) : Prop := sorry

-- Theorem statement
theorem alternate_interior_angles_equal
  (l1 l2 l3 : Line)
  (h1 : parallel l1 l2)
  (h2 : intersects l3 l1)
  (h3 : intersects l3 l2) :
  alternate_interior_angles l1 l2 l3 :=
sorry

end alternate_interior_angles_equal_l17_17713


namespace bob_cookie_price_same_as_jane_l17_17346

theorem bob_cookie_price_same_as_jane
  (r_jane : ℝ)
  (s_bob : ℝ)
  (dough_jane : ℝ)
  (num_jane_cookies : ℕ)
  (price_jane_cookie : ℝ)
  (total_earning_jane : ℝ)
  (num_cookies_bob : ℝ)
  (price_bob_cookie : ℝ) :
  r_jane = 4 ∧
  s_bob = 6 ∧
  dough_jane = 18 * (Real.pi * r_jane^2) ∧
  price_jane_cookie = 0.50 ∧
  total_earning_jane = 18 * 50 ∧
  num_cookies_bob = dough_jane / s_bob^2 ∧
  total_earning_jane = num_cookies_bob * price_bob_cookie →
  price_bob_cookie = 36 :=
by
  intros
  sorry

end bob_cookie_price_same_as_jane_l17_17346


namespace matrix_det_l17_17985

def matrix := ![
  ![2, -4, 2],
  ![0, 6, -1],
  ![5, -3, 1]
]

theorem matrix_det : Matrix.det matrix = -34 := by
  sorry

end matrix_det_l17_17985


namespace unguarded_area_eq_225_l17_17705

-- Define the basic conditions of the problem in Lean
structure Room where
  side_length : ℕ
  unguarded_fraction : ℚ
  deriving Repr

-- Define the specific room used in the problem
def problemRoom : Room :=
  { side_length := 10,
    unguarded_fraction := 9/4 }

-- Define the expected unguarded area in square meters
def expected_unguarded_area (r : Room) : ℚ :=
  r.unguarded_fraction * (r.side_length ^ 2)

-- Prove that the unguarded area is 225 square meters
theorem unguarded_area_eq_225 (r : Room) (h : r = problemRoom) : expected_unguarded_area r = 225 := by
  -- The proof in this case is omitted.
  sorry

end unguarded_area_eq_225_l17_17705


namespace grain_distance_l17_17669

theorem grain_distance
    (d : ℝ) (v_church : ℝ) (v_cathedral : ℝ)
    (h_d : d = 400) (h_v_church : v_church = 20) (h_v_cathedral : v_cathedral = 25) :
    ∃ x : ℝ, x = 1600 / 9 ∧ v_church * x = v_cathedral * (d - x) :=
by
  sorry

end grain_distance_l17_17669


namespace exercise_l17_17545

theorem exercise (a b : ℕ) (h1 : 656 = 3 * 7^2 + a * 7 + b) (h2 : 656 = 3 * 10^2 + a * 10 + b) : 
  (a * b) / 15 = 1 :=
by
  sorry

end exercise_l17_17545


namespace find_number_l17_17876

theorem find_number (x y a : ℝ) (h₁ : x * y = 1) (h₂ : (a ^ ((x + y) ^ 2)) / (a ^ ((x - y) ^ 2)) = 1296) : a = 6 :=
sorry

end find_number_l17_17876


namespace regular_polygon_sides_l17_17385

theorem regular_polygon_sides (n : ℕ) (h : (n - 2) * 180 / n = 135) : n = 8 := 
by
  sorry

end regular_polygon_sides_l17_17385


namespace equilateral_triangle_circumradius_ratio_l17_17176

variables (B b S s : ℝ)

-- Given two equilateral triangles with side lengths B and b, and respectively circumradii S and s
-- B and b are not equal
-- Prove that S / s = B / b
theorem equilateral_triangle_circumradius_ratio (hBneqb : B ≠ b)
  (hS : S = B * Real.sqrt 3 / 3)
  (hs : s = b * Real.sqrt 3 / 3) : S / s = B / b :=
by
  sorry

end equilateral_triangle_circumradius_ratio_l17_17176


namespace original_average_l17_17708

theorem original_average (A : ℝ) (h : (2 * (12 * A)) / 12 = 100) : A = 50 :=
by
  sorry

end original_average_l17_17708


namespace central_angle_of_sector_l17_17649

theorem central_angle_of_sector (r α : ℝ) (h_arc_length : α * r = 5) (h_area : 0.5 * α * r^2 = 5): α = 5 / 2 := by
  sorry

end central_angle_of_sector_l17_17649


namespace imaginary_part_of_product_l17_17942

def imaginary_unit : ℂ := Complex.I

def z : ℂ := 2 + imaginary_unit

theorem imaginary_part_of_product : (z * imaginary_unit).im = 2 := by
  sorry

end imaginary_part_of_product_l17_17942


namespace mms_pack_count_l17_17301

def mms_per_pack (sundaes_monday : Nat) (mms_monday : Nat) (sundaes_tuesday : Nat) (mms_tuesday : Nat) (packs : Nat) : Nat :=
  (sundaes_monday * mms_monday + sundaes_tuesday * mms_tuesday) / packs

theorem mms_pack_count 
  (sundaes_monday : Nat)
  (mms_monday : Nat)
  (sundaes_tuesday : Nat)
  (mms_tuesday : Nat)
  (packs : Nat)
  (monday_total_mms : sundaes_monday * mms_monday = 240)
  (tuesday_total_mms : sundaes_tuesday * mms_tuesday = 200)
  (total_packs : packs = 11)
  : mms_per_pack sundaes_monday mms_monday sundaes_tuesday mms_tuesday packs = 40 := by
  sorry

end mms_pack_count_l17_17301


namespace points_on_opposite_sides_of_line_l17_17788

theorem points_on_opposite_sides_of_line 
  (a : ℝ) 
  (h : (3 * -3 - 2 * -1 - a) * (3 * 4 - 2 * -6 - a) < 0) : 
  -7 < a ∧ a < 24 :=
sorry

end points_on_opposite_sides_of_line_l17_17788


namespace calculate_otimes_l17_17145

def otimes (x y : ℝ) : ℝ := x^3 - y^2 + x

theorem calculate_otimes (k : ℝ) : 
  otimes k (otimes k k) = -k^6 + 2*k^5 - 3*k^4 + 3*k^3 - k^2 + 2*k := by
  sorry

end calculate_otimes_l17_17145


namespace value_of_expression_l17_17262

theorem value_of_expression 
  (x1 x2 x3 x4 x5 x6 x7 : ℝ)
  (h1 : x1 + 9*x2 + 25*x3 + 49*x4 + 81*x5 + 121*x6 + 169*x7 = 2)
  (h2 : 9*x1 + 25*x2 + 49*x3 + 81*x4 + 121*x5 + 169*x6 + 225*x7 = 24)
  (h3 : 25*x1 + 49*x2 + 81*x3 + 121*x4 + 169*x5 + 225*x6 + 289*x7 = 246) : 
  49*x1 + 81*x2 + 121*x3 + 169*x4 + 225*x5 + 289*x6 + 361*x7 = 668 := 
sorry

end value_of_expression_l17_17262


namespace prob_iff_eq_l17_17265

noncomputable def A (m : ℝ) : Set ℝ := { x | x^2 + m * x + 2 ≥ 0 ∧ x ≥ 0 }
noncomputable def B (m : ℝ) : Set ℝ := { y | ∃ x, x ∈ A m ∧ y = Real.sqrt (x^2 + m * x + 2) }

theorem prob_iff_eq (m : ℝ) : (A m = { y | ∃ x, x ^ 2 + m * x + 2 = y ^ 2 ∧ x ≥ 0 } ↔ m = -2 * Real.sqrt 2) :=
by
  sorry

end prob_iff_eq_l17_17265


namespace prove_a_eq_neg2_solve_inequality_for_a_leq0_l17_17067

-- Problem 1: Proving that a = -2 given the solution set of the inequality
theorem prove_a_eq_neg2 (a : ℝ) (h : ∀ x : ℝ, (-1 < x ∧ x < -1/2) ↔ (ax - 1) * (x + 1) > 0) : a = -2 := sorry

-- Problem 2: Solving the inequality (ax-1)(x+1) > 0 for different conditions on a
theorem solve_inequality_for_a_leq0 (a x : ℝ) (h_a_le_0 : a ≤ 0) : 
  (ax - 1) * (x + 1) > 0 ↔ 
    if a < -1 then -1 < x ∧ x < 1/a
    else if a = -1 then false
    else if -1 < a ∧ a < 0 then 1/a < x ∧ x < -1
    else x < -1 := sorry

end prove_a_eq_neg2_solve_inequality_for_a_leq0_l17_17067


namespace circle_standard_equation_l17_17107

theorem circle_standard_equation (x y : ℝ) (center : ℝ × ℝ) (radius : ℝ) 
  (h_center : center = (2, -1)) (h_radius : radius = 2) :
  (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2 ↔ (x - 2) ^ 2 + (y + 1) ^ 2 = 4 := by
  sorry

end circle_standard_equation_l17_17107


namespace payment_to_z_l17_17864

-- Definitions of the conditions
def x_work_rate := 1 / 15
def y_work_rate := 1 / 10
def total_payment := 720
def combined_work_rate_xy := x_work_rate + y_work_rate
def combined_work_rate_xyz := 1 / 5
def z_work_rate := combined_work_rate_xyz - combined_work_rate_xy
def z_contribution := z_work_rate * 5
def z_payment := z_contribution * total_payment

-- The statement to be proven
theorem payment_to_z : z_payment = 120 := by
  sorry

end payment_to_z_l17_17864


namespace bags_le_40kg_l17_17869

theorem bags_le_40kg (capacity boxes crates sacks box_weight crate_weight sack_weight bag_weight: ℕ)
  (h_capacity: capacity = 13500)
  (h_boxes: boxes = 100)
  (h_crates: crates = 10)
  (h_sacks: sacks = 50)
  (h_box_weight: box_weight = 100)
  (h_crate_weight: crate_weight = 60)
  (h_sack_weight: sack_weight = 50)
  (h_bag_weight: bag_weight = 40) :
  10 = (capacity - (boxes * box_weight + crates * crate_weight + sacks * sack_weight)) / bag_weight := by 
  sorry

end bags_le_40kg_l17_17869


namespace degree_sequence_a_invalid_degree_sequence_b_invalid_degree_sequence_c_invalid_all_sequences_invalid_l17_17382

-- Definition of the "isValidGraph" function based on degree sequences
-- Placeholder for the actual definition
def isValidGraph (degrees : List ℕ) : Prop :=
  sorry

-- Degree sequences given in the problem
def d_a := [8, 6, 5, 4, 4, 3, 2, 2]
def d_b := [7, 7, 6, 5, 4, 2, 2, 1]
def d_c := [6, 6, 6, 5, 5, 3, 2, 2]

-- Statement that proves none of these sequences can form a valid graph
theorem degree_sequence_a_invalid : ¬ isValidGraph d_a :=
  sorry

theorem degree_sequence_b_invalid : ¬ isValidGraph d_b :=
  sorry

theorem degree_sequence_c_invalid : ¬ isValidGraph d_c :=
  sorry

-- Final statement combining all individual proofs
theorem all_sequences_invalid :
  ¬ isValidGraph d_a ∧ ¬ isValidGraph d_b ∧ ¬ isValidGraph d_c :=
  ⟨degree_sequence_a_invalid, degree_sequence_b_invalid, degree_sequence_c_invalid⟩

end degree_sequence_a_invalid_degree_sequence_b_invalid_degree_sequence_c_invalid_all_sequences_invalid_l17_17382


namespace calculation_result_l17_17200

theorem calculation_result :
  3 * 15 + 3 * 16 + 3 * 19 + 11 = 161 :=
sorry

end calculation_result_l17_17200


namespace box_volume_l17_17449

theorem box_volume (L W H : ℝ) (h1 : L * W = 120) (h2 : W * H = 72) (h3 : L * H = 60) : L * W * H = 720 := 
by sorry

end box_volume_l17_17449


namespace max_pencils_l17_17578

theorem max_pencils 
  (p : ℕ → ℝ)
  (h_price1 : ∀ n : ℕ, n ≤ 10 → p n = 0.75 * n)
  (h_price2 : ∀ n : ℕ, n > 10 → p n = 0.75 * 10 + 0.65 * (n - 10))
  (budget : ℝ) (h_budget : budget = 10) :
  ∃ n : ℕ, p n ≤ budget ∧ (∀ m : ℕ, p m ≤ budget → m ≤ 13) :=
by {
  sorry
}

end max_pencils_l17_17578


namespace larger_number_is_30_l17_17973

-- Formalizing the conditions
variables (x y : ℝ)

-- Define the conditions given in the problem
def sum_condition : Prop := x + y = 40
def ratio_condition : Prop := x / y = 3

-- Formalize the problem statement
theorem larger_number_is_30 (h1 : sum_condition x y) (h2 : ratio_condition x y) : x = 30 :=
sorry

end larger_number_is_30_l17_17973


namespace y_value_l17_17371

def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b + d)

theorem y_value (x y : ℤ) (h1 : star 5 0 2 (-2) = (3, -2)) (h2 : star x y 0 3 = (3, -2)) :
  y = -5 :=
sorry

end y_value_l17_17371


namespace quadratic_roots_identity_l17_17954

theorem quadratic_roots_identity (α β : ℝ) (hαβ : α^2 - 3*α - 4 = 0 ∧ β^2 - 3*β - 4 = 0) : 
  α^2 + α*β - 3*α = 0 := 
by 
  sorry

end quadratic_roots_identity_l17_17954


namespace sum_difference_even_odd_l17_17923

-- Define the sum of even integers from 2 to 100
def sum_even (n : ℕ) : ℕ := (n / 2) * (2 + n)

-- Define the sum of odd integers from 1 to 99
def sum_odd (n : ℕ) : ℕ := (n / 2) * (1 + n)

theorem sum_difference_even_odd:
  let a := sum_even 100
  let b := sum_odd 99
  a - b = 50 :=
by
  sorry

end sum_difference_even_odd_l17_17923


namespace ana_final_salary_l17_17867

def initial_salary : ℝ := 2500
def june_raise : ℝ := initial_salary * 0.15
def june_bonus : ℝ := 300
def salary_after_june : ℝ := initial_salary + june_raise + june_bonus
def july_pay_cut : ℝ := salary_after_june * 0.25
def final_salary : ℝ := salary_after_june - july_pay_cut

theorem ana_final_salary :
  final_salary = 2381.25 := by
  -- sorry is used here to skip the proof
  sorry

end ana_final_salary_l17_17867


namespace weight_ratio_l17_17489

noncomputable def students_weight : ℕ := 79
noncomputable def siblings_total_weight : ℕ := 116

theorem weight_ratio (S W : ℕ) (h1 : siblings_total_weight = S + W) (h2 : students_weight = S):
  (S - 5) / (siblings_total_weight - S) = 2 :=
by
  sorry

end weight_ratio_l17_17489


namespace value_at_7_5_l17_17484

def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 2) = -f x
axiom interval_condition (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f x = x

theorem value_at_7_5 : f 7.5 = -0.5 := by
  sorry

end value_at_7_5_l17_17484


namespace necessary_but_not_sufficient_l17_17712

theorem necessary_but_not_sufficient (x : ℝ) : (1 - x) * (1 + |x|) > 0 -> x < 2 :=
by
  sorry

end necessary_but_not_sufficient_l17_17712


namespace last_colored_cell_is_51_50_l17_17984

def last_spiral_cell (width height : ℕ) : ℕ × ℕ :=
  -- Assuming an external or pre-defined process to calculate the last cell for a spiral pattern
  sorry 

theorem last_colored_cell_is_51_50 :
  last_spiral_cell 200 100 = (51, 50) :=
sorry

end last_colored_cell_is_51_50_l17_17984


namespace sin_60_eq_sqrt_three_div_two_l17_17090

theorem sin_60_eq_sqrt_three_div_two :
  Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_60_eq_sqrt_three_div_two_l17_17090


namespace find_greatest_number_l17_17604

def numbers := [0.07, -0.41, 0.8, 0.35, -0.9]

theorem find_greatest_number :
  ∃ x ∈ numbers, x > 0.7 ∧ ∀ y ∈ numbers, y > 0.7 → y = 0.8 :=
by
  sorry

end find_greatest_number_l17_17604


namespace solution_system_of_equations_solution_system_of_inequalities_l17_17873

-- Part 1: System of Equations
theorem solution_system_of_equations (x y : ℚ) :
  (3 * x + 2 * y = 13) ∧ (2 * x + 3 * y = -8) ↔ (x = 11 ∧ y = -10) :=
by
  sorry

-- Part 2: System of Inequalities
theorem solution_system_of_inequalities (y : ℚ) :
  ((5 * y - 2) / 3 - 1 > (3 * y - 5) / 2) ∧ (2 * (y - 3) ≤ 0) ↔ (-5 < y ∧ y ≤ 3) :=
by
  sorry

end solution_system_of_equations_solution_system_of_inequalities_l17_17873


namespace maximize_z_l17_17510

open Real

theorem maximize_z (x y : ℝ) (h1 : x + y ≤ 10) (h2 : 3 * x + y ≤ 18) (h3 : 0 ≤ x) (h4 : 0 ≤ y) :
  (∀ x y, x + y ≤ 10 ∧ 3 * x + y ≤ 18 ∧ 0 ≤ x ∧ 0 ≤ y → x + y / 2 ≤ 7) :=
by
  sorry

end maximize_z_l17_17510


namespace division_and_multiplication_l17_17740

theorem division_and_multiplication (a b c d : ℝ) : (a / b / c * d) = 30 :=
by 
  let a := 120
  let b := 6
  let c := 2
  let d := 3
  sorry

end division_and_multiplication_l17_17740


namespace monthly_income_of_A_l17_17742

theorem monthly_income_of_A (A B C : ℝ)
  (h1 : (A + B) / 2 = 5050)
  (h2 : (B + C) / 2 = 6250)
  (h3 : (A + C) / 2 = 5200) :
  A = 4000 :=
sorry

end monthly_income_of_A_l17_17742


namespace units_digit_of_result_is_3_l17_17675

def hundreds_digit_relation (c : ℕ) (a : ℕ) : Prop :=
  a = 2 * c - 3

def original_number_expression (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

def reversed_number_expression (a b c : ℕ) : ℕ :=
  100 * c + 10 * b + a + 50

def subtraction_result (orig rev : ℕ) : ℕ :=
  orig - rev

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_result_is_3 (a b c : ℕ) (h : hundreds_digit_relation c a) :
  units_digit (subtraction_result (original_number_expression a b c)
                                  (reversed_number_expression a b c)) = 3 :=
by
  sorry

end units_digit_of_result_is_3_l17_17675


namespace distance_T_S_l17_17224

theorem distance_T_S : 
  let P := -14
  let Q := 46
  let S := P + (3 / 4:ℚ) * (Q - P)
  let T := P + (1 / 3:ℚ) * (Q - P)
  S - T = 25 :=
by
  let P := -14
  let Q := 46
  let S := P + (3 / 4:ℚ) * (Q - P)
  let T := P + (1 / 3:ℚ) * (Q - P)
  show S - T = 25
  sorry

end distance_T_S_l17_17224


namespace diff_of_cubes_divisible_by_9_l17_17439

theorem diff_of_cubes_divisible_by_9 (a b : ℤ) : 9 ∣ ((2 * a + 1)^3 - (2 * b + 1)^3) := 
sorry

end diff_of_cubes_divisible_by_9_l17_17439


namespace geometric_first_term_l17_17124

-- Define the conditions
def is_geometric_series (first_term : ℝ) (r : ℝ) (sum : ℝ) : Prop :=
  sum = first_term / (1 - r)

-- Define the main theorem
theorem geometric_first_term (r : ℝ) (sum : ℝ) (first_term : ℝ) 
  (h_r : r = 1/4) (h_S : sum = 80) (h_sum_formula : is_geometric_series first_term r sum) : 
  first_term = 60 :=
by
  sorry

end geometric_first_term_l17_17124


namespace highest_probability_two_out_of_three_probability_l17_17192

structure Student :=
  (name : String)
  (P_T : ℚ)  -- Probability of passing the theoretical examination
  (P_S : ℚ)  -- Probability of passing the social practice examination

noncomputable def P_earn (student : Student) : ℚ :=
  student.P_T * student.P_S

def student_A := Student.mk "A" (5 / 6) (1 / 2)
def student_B := Student.mk "B" (4 / 5) (2 / 3)
def student_C := Student.mk "C" (3 / 4) (5 / 6)

theorem highest_probability : 
  P_earn student_C > P_earn student_B ∧ P_earn student_B > P_earn student_A :=
by sorry

theorem two_out_of_three_probability :
  (1 - P_earn student_A) * P_earn student_B * P_earn student_C +
  P_earn student_A * (1 - P_earn student_B) * P_earn student_C +
  P_earn student_A * P_earn student_B * (1 - P_earn student_C) =
  115 / 288 :=
by sorry

end highest_probability_two_out_of_three_probability_l17_17192


namespace average_age_of_women_l17_17511

variable {A W : ℝ}

theorem average_age_of_women (A : ℝ) (h : 12 * (A + 3) = 12 * A - 90 + W) : 
  W / 3 = 42 := by
  sorry

end average_age_of_women_l17_17511


namespace find_a_of_perpendicular_lines_l17_17730

theorem find_a_of_perpendicular_lines (a : ℝ) :
  let line1 : ℝ := a * x + y - 1
  let line2 : ℝ := 4 * x + (a - 3) * y - 2
  (∀ x y : ℝ, (line1 = 0 → line2 ≠ 0 → line1 * line2 = -1)) → a = 3 / 5 :=
by
  sorry

end find_a_of_perpendicular_lines_l17_17730


namespace find_length_of_rectangular_playground_l17_17771

def perimeter (L B : ℕ) : ℕ := 2 * (L + B)

theorem find_length_of_rectangular_playground (P B : ℕ) (hP : P = 1200) (hB : B = 500) : ∃ L, perimeter L B = P ∧ L = 100 :=
by
  sorry

end find_length_of_rectangular_playground_l17_17771


namespace min_value_geq_four_l17_17470

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  (x + y) / (x * y * z)

theorem min_value_geq_four (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) :
  4 ≤ min_value_expression x y z :=
sorry

end min_value_geq_four_l17_17470


namespace additional_number_is_31_l17_17011

theorem additional_number_is_31
(six_numbers_sum : ℕ)
(seven_numbers_avg : ℕ)
(h1 : six_numbers_sum = 144)
(h2 : seven_numbers_avg = 25)
: ∃ x : ℕ, ((six_numbers_sum + x) / 7 = 25) ∧ x = 31 := 
by
  sorry

end additional_number_is_31_l17_17011


namespace consultation_session_probability_l17_17359

noncomputable def consultation_probability : ℝ :=
  let volume_cube := 3 * 3 * 3
  let volume_valid := 9 - 2 * (1/3 * 2.25 * 1.5)
  volume_valid / volume_cube

theorem consultation_session_probability : consultation_probability = 1 / 4 :=
by
  sorry

end consultation_session_probability_l17_17359


namespace simplify_expr1_simplify_expr2_l17_17453

theorem simplify_expr1 (a b : ℝ) : 6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1 / 2) * a * b) = -a * b :=
by
  sorry

theorem simplify_expr2 (t : ℝ) : -(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l17_17453


namespace farmer_apples_count_l17_17152

-- Definitions from the conditions in step a)
def initial_apples : ℕ := 127
def apples_given_away : ℕ := 88

-- Proof goal from step c)
theorem farmer_apples_count : initial_apples - apples_given_away = 39 :=
by
  sorry

end farmer_apples_count_l17_17152


namespace paper_holes_symmetric_l17_17566

-- Define the initial conditions
def folded_paper : Type := sorry -- Specific structure to represent the paper and its folds

def paper_fold_bottom_to_top (paper : folded_paper) : folded_paper := sorry
def paper_fold_right_half_to_left (paper : folded_paper) : folded_paper := sorry
def paper_fold_diagonal (paper : folded_paper) : folded_paper := sorry

-- Define a function that represents punching a hole near the folded edge
def punch_hole_near_folded_edge (paper : folded_paper) : folded_paper := sorry

-- Initial paper
def initial_paper : folded_paper := sorry

-- Folded and punched paper
def paper_after_folds_and_punch : folded_paper :=
  punch_hole_near_folded_edge (
    paper_fold_diagonal (
      paper_fold_right_half_to_left (
        paper_fold_bottom_to_top initial_paper)))

-- Unfolding the paper
def unfold_diagonal (paper : folded_paper) : folded_paper := sorry
def unfold_right_half (paper : folded_paper) : folded_paper := sorry
def unfold_bottom_to_top (paper : folded_paper) : folded_paper := sorry

def paper_after_unfolding : folded_paper :=
  unfold_bottom_to_top (
    unfold_right_half (
      unfold_diagonal paper_after_folds_and_punch))

-- Definition of hole pattern 'eight_symmetric_holes'
def eight_symmetric_holes (paper : folded_paper) : Prop := sorry

-- The proof problem
theorem paper_holes_symmetric :
  eight_symmetric_holes paper_after_unfolding := sorry

end paper_holes_symmetric_l17_17566


namespace extra_charge_per_wand_l17_17428

theorem extra_charge_per_wand
  (cost_per_wand : ℕ)
  (num_wands : ℕ)
  (total_collected : ℕ)
  (num_wands_sold : ℕ)
  (h_cost : cost_per_wand = 60)
  (h_num_wands : num_wands = 3)
  (h_total_collected : total_collected = 130)
  (h_num_wands_sold : num_wands_sold = 2) :
  ((total_collected / num_wands_sold) - cost_per_wand) = 5 :=
by
  -- Proof goes here
  sorry

end extra_charge_per_wand_l17_17428


namespace equal_savings_l17_17966

theorem equal_savings (U B UE BE US BS : ℕ) (h1 : U / B = 8 / 7) 
                      (h2 : U = 16000) (h3 : UE / BE = 7 / 6) (h4 : US = BS) :
                      US = 2000 ∧ BS = 2000 :=
by
  sorry

end equal_savings_l17_17966


namespace base9_sum_correct_l17_17326

def base9_addition (a b c : ℕ) : ℕ :=
  a + b + c

theorem base9_sum_correct :
  base9_addition (263) (452) (247) = 1073 :=
by sorry

end base9_sum_correct_l17_17326


namespace color_theorem_l17_17757

/-- The only integers \( k \geq 1 \) such that if each integer is colored in one of these \( k \)
colors, there must exist integers \( a_1 < a_2 < \cdots < a_{2023} \) of the same color where the
differences \( a_2 - a_1, a_3 - a_2, \cdots, a_{2023} - a_{2022} \) are all powers of 2 are
\( k = 1 \) and \( k = 2 \). -/
theorem color_theorem : ∀ (k : ℕ), (k ≥ 1) →
  (∀ f : ℕ → Fin k,
    ∃ a : Fin 2023 → ℕ,
    (∀ i : Fin (2023 - 1), ∃ n : ℕ, 2^n = (a i.succ - a i)) ∧
    (∀ i j : Fin 2023, i < j → f (a i) = f (a j)))
  ↔ k = 1 ∨ k = 2 := by
  sorry

end color_theorem_l17_17757


namespace complex_fourth_power_l17_17887

theorem complex_fourth_power (i : ℂ) (hi : i^2 = -1) : (1 - i)^4 = -4 := 
sorry

end complex_fourth_power_l17_17887


namespace gcd_4320_2550_l17_17792

-- Definitions for 4320 and 2550
def a : ℕ := 4320
def b : ℕ := 2550

-- Statement to prove the greatest common factor of a and b is 30
theorem gcd_4320_2550 : Nat.gcd a b = 30 := 
by 
  sorry

end gcd_4320_2550_l17_17792


namespace reimbursement_correct_l17_17174

-- Define the days and miles driven each day
def miles_monday : ℕ := 18
def miles_tuesday : ℕ := 26
def miles_wednesday : ℕ := 20
def miles_thursday : ℕ := 20
def miles_friday : ℕ := 16

-- Define the mileage rate
def mileage_rate : ℝ := 0.36

-- Define the total miles driven
def total_miles_driven : ℕ := miles_monday + miles_tuesday + miles_wednesday + miles_thursday + miles_friday

-- Define the total reimbursement
def reimbursement : ℝ := total_miles_driven * mileage_rate

-- Prove that the reimbursement is $36
theorem reimbursement_correct : reimbursement = 36 := by
  sorry

end reimbursement_correct_l17_17174


namespace solve_problem_l17_17127

theorem solve_problem
    (product_trailing_zeroes : ∃ (x y z w v u p q r : ℕ), (10 ∣ (x * y * z * w * v * u * p * q * r)) ∧ B = 0)
    (digit_sequences : (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9) % 10 = 8 ∧
                       (11 * 12 * 13 * 14 * 15 * 16 * 17 * 18 * 19) % 10 = 4 ∧
                       (21 * 22 * 23 * 24 * 25 * 26 * 27 * 28 * 29) % 10 = 4 ∧
                       (31 * 32 * 33 * 34 * 35) % 10 = 4 ∧
                       A = 2 ∧ B = 0)
    (divisibility_rule_11 : ∀ C D, (71 + C) - (68 + D) = 11 → C - D = -3 ∨ C - D = 8)
    (divisibility_rule_9 : ∀ C D, (139 + C + D) % 9 = 0 → C + D = 5 ∨ C + D = 14)
    (system_of_equations : ∀ C D, (C - D = -3 ∧ C + D = 5) → (C = 1 ∧ D = 4)) :
  A = 2 ∧ B = 0 ∧ C = 1 ∧ D = 4 :=
by
  sorry

end solve_problem_l17_17127


namespace most_people_can_attend_on_most_days_l17_17478

-- Define the days of the week as a type
inductive Day
| Mon | Tues | Wed | Thurs | Fri

open Day

-- Define the availability of each person
def is_available (person : String) (day : Day) : Prop :=
  match person, day with
  | "Anna", Mon => False
  | "Anna", Wed => False
  | "Anna", Fri => False
  | "Bill", Tues => False
  | "Bill", Thurs => False
  | "Bill", Fri => False
  | "Carl", Mon => False
  | "Carl", Tues => False
  | "Carl", Thurs => False
  | "Diana", Wed => False
  | "Diana", Fri => False
  | _, _ => True

-- Prove the result
theorem most_people_can_attend_on_most_days :
  {d : Day | d ∈ [Mon, Tues, Wed]} = {d : Day | ∀p : String, is_available p d → p ∈ ["Bill", "Carl", "Diana"] ∨ p ∉ ["Anna", "Bill"]} :=
sorry

end most_people_can_attend_on_most_days_l17_17478


namespace wang_pens_purchase_l17_17065

theorem wang_pens_purchase :
  ∀ (total_money spent_on_albums pen_cost : ℝ)
  (number_of_pens : ℕ),
  total_money = 80 →
  spent_on_albums = 45.6 →
  pen_cost = 2.5 →
  number_of_pens = 13 →
  (total_money - spent_on_albums) / pen_cost ≥ number_of_pens ∧ 
  (total_money - spent_on_albums) / pen_cost < number_of_pens + 1 :=
by
  intros
  sorry

end wang_pens_purchase_l17_17065


namespace find_z_l17_17327

theorem find_z 
  {x y z : ℕ}
  (hx : x = 4)
  (hy : y = 7)
  (h_least : x - y - z = 17) : 
  z = 14 :=
by
  sorry

end find_z_l17_17327


namespace no_triangle_with_heights_1_2_3_l17_17568

open Real

theorem no_triangle_with_heights_1_2_3 :
  ¬(∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
     ∃ (k : ℝ), k > 0 ∧ 
       a * k = 1 ∧ b * (k / 2) = 2 ∧ c * (k / 3) = 3 ∧
       (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
by 
  sorry

end no_triangle_with_heights_1_2_3_l17_17568


namespace longer_diagonal_of_rhombus_l17_17252

theorem longer_diagonal_of_rhombus (d1 d2 : ℝ) (area : ℝ) (h₁ : d1 = 12) (h₂ : area = 120) :
  d2 = 20 :=
by
  sorry

end longer_diagonal_of_rhombus_l17_17252


namespace trig_triple_angle_l17_17134

theorem trig_triple_angle (θ : ℝ) (h : Real.tan θ = 5) :
  Real.tan (3 * θ) = 55 / 37 ∧
  Real.sin (3 * θ) = 55 * Real.sqrt 1369 / (37 * Real.sqrt 4394) ∨ Real.sin (3 * θ) = -(55 * Real.sqrt 1369 / (37 * Real.sqrt 4394)) ∧
  Real.cos (3 * θ) = Real.sqrt (1369 / 4394) ∨ Real.cos (3 * θ) = -Real.sqrt (1369 / 4394) :=
by
  sorry

end trig_triple_angle_l17_17134


namespace cheese_pops_count_l17_17482

-- Define the number of hotdogs, chicken nuggets, and total portions
def hotdogs : ℕ := 30
def chicken_nuggets : ℕ := 40
def total_portions : ℕ := 90

-- Define the number of bite-sized cheese pops
def cheese_pops : ℕ := total_portions - hotdogs - chicken_nuggets

-- Theorem to prove that the number of bite-sized cheese pops Andrew brought is 20
theorem cheese_pops_count :
  cheese_pops = 20 :=
by
  -- The following proof is omitted
  sorry

end cheese_pops_count_l17_17482


namespace special_set_exists_l17_17884

def exists_special_set : Prop :=
  ∃ S : Finset ℕ, S.card = 4004 ∧ 
  (∀ A : Finset ℕ, A ⊆ S ∧ A.card = 2003 → (A.sum id % 2003 ≠ 0))

-- statement with sorry to skip the proof
theorem special_set_exists : exists_special_set :=
sorry

end special_set_exists_l17_17884


namespace perfect_squares_unique_l17_17555

theorem perfect_squares_unique (n : ℕ) (h1 : ∃ k : ℕ, 20 * n = k^2) (h2 : ∃ p : ℕ, 5 * n + 275 = p^2) :
  n = 125 :=
by
  sorry

end perfect_squares_unique_l17_17555


namespace hyungjun_initial_ribbon_length_l17_17628

noncomputable def initial_ribbon_length (R: ℝ) : Prop :=
  let used_for_first_box := R / 2 + 2000
  let remaining_after_first := R - used_for_first_box
  let used_for_second_box := (remaining_after_first / 2) + 2000
  remaining_after_first - used_for_second_box = 0

theorem hyungjun_initial_ribbon_length : ∃ R: ℝ, initial_ribbon_length R ∧ R = 12000 :=
  by
  exists 12000
  unfold initial_ribbon_length
  simp
  sorry

end hyungjun_initial_ribbon_length_l17_17628


namespace right_triangle_area_l17_17217

theorem right_triangle_area
    (h : ∀ {a b c : ℕ}, a^2 + b^2 = c^2 → c = 13 → a = 5 ∨ b = 5)
    (hypotenuse : ℕ)
    (leg : ℕ)
    (hypotenuse_eq : hypotenuse = 13)
    (leg_eq : leg = 5) : ∃ (area: ℕ), area = 30 :=
by
  -- The proof will go here.
  sorry

end right_triangle_area_l17_17217


namespace marbles_count_l17_17815

-- Define the condition variables
variable (M : ℕ) -- total number of marbles placed on Monday
variable (day2_marbles : ℕ) -- marbles remaining after second day
variable (day3_cleo_marbles : ℕ) -- marbles taken by Cleo on third day

-- Condition definitions
def condition1 : Prop := day2_marbles = 2 * M / 5
def condition2 : Prop := day3_cleo_marbles = (day2_marbles / 2)
def condition3 : Prop := day3_cleo_marbles = 15

-- The theorem to prove
theorem marbles_count : 
  condition1 M day2_marbles → 
  condition2 day2_marbles day3_cleo_marbles → 
  condition3 day3_cleo_marbles → 
  M = 75 :=
by
  intros h1 h2 h3
  sorry

end marbles_count_l17_17815


namespace minjeong_walk_distance_l17_17468

noncomputable def park_side_length : ℕ := 40
noncomputable def square_sides : ℕ := 4

theorem minjeong_walk_distance (side_length : ℕ) (sides : ℕ) (h : side_length = park_side_length) (h2 : sides = square_sides) : 
  side_length * sides = 160 := by
  sorry

end minjeong_walk_distance_l17_17468


namespace range_of_m_l17_17227

theorem range_of_m (m : ℝ) : (∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → (x - 1) * (x - (m - 1)) > 0) → m > 1 :=
by
  intro h
  sorry

end range_of_m_l17_17227


namespace cone_volume_filled_88_8900_percent_l17_17136

noncomputable def cone_volume_ratio_filled_to_two_thirds_height
  (h r : ℝ) (π : ℝ) : ℝ :=
  let V := (1 / 3) * π * r ^ 2 * h
  let V' := (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)
  (V' / V * 100)

theorem cone_volume_filled_88_8900_percent
  (h r π : ℝ) (V V' : ℝ)
  (V_def : V = (1 / 3) * π * r ^ 2 * h)
  (V'_def : V' = (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)):
  cone_volume_ratio_filled_to_two_thirds_height h r π = 88.8900 :=
by
  sorry

end cone_volume_filled_88_8900_percent_l17_17136


namespace kristin_reading_time_l17_17066

-- Definitions
def total_books : Nat := 20
def peter_time_per_book : ℕ := 18
def reading_speed_ratio : Nat := 3

-- Derived Definitions
def kristin_time_per_book : ℕ := peter_time_per_book * reading_speed_ratio
def kristin_books_to_read : Nat := total_books / 2
def kristin_total_time : ℕ := kristin_time_per_book * kristin_books_to_read

-- Statement to be proved
theorem kristin_reading_time :
  kristin_total_time = 540 :=
  by 
    -- Proof would go here, but we are only required to state the theorem
    sorry

end kristin_reading_time_l17_17066


namespace message_forwarding_time_l17_17168

theorem message_forwarding_time :
  ∃ n : ℕ, (∀ m : ℕ, (∀ p : ℕ, (∀ q : ℕ, 1 + (2 * (2 ^ n)) - 1 = 2047)) ∧ n = 10) :=
sorry

end message_forwarding_time_l17_17168


namespace find_overlap_length_l17_17499

-- Define the given conditions
def plank_length : ℝ := 30 -- length of each plank in cm
def number_of_planks : ℕ := 25 -- number of planks
def total_fence_length : ℝ := 690 -- total length of the fence in cm

-- Definition for the overlap length
def overlap_length (y : ℝ) : Prop :=
  total_fence_length = (13 * plank_length) + (12 * (plank_length - 2 * y))

-- Theorem statement to prove the required overlap length
theorem find_overlap_length : ∃ y : ℝ, overlap_length y ∧ y = 2.5 :=
by 
  -- The proof goes here
  sorry

end find_overlap_length_l17_17499


namespace minimum_value_expression_l17_17076

theorem minimum_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : b ≠ 0) :
  (∃ a b c : ℝ, (b > c ∧ c > a) ∧ b ≠ 0 ∧ (a + b) = b - c ∧ (b - c) = c - a ∧ (a - c) = 0 ∧
   ∀ x y z : ℝ, (x = a + b ∧ y = b - c ∧ z = c - a) → 
    (x^2 + y^2 + z^2) / b^2 = 4/3) :=
  sorry

end minimum_value_expression_l17_17076


namespace initial_winnings_l17_17415

theorem initial_winnings (X : ℝ) 
  (h1 : X - 0.25 * X = 0.75 * X)
  (h2 : 0.75 * X - 0.10 * (0.75 * X) = 0.675 * X)
  (h3 : 0.675 * X - 0.15 * (0.675 * X) = 0.57375 * X)
  (h4 : 0.57375 * X = 240) :
  X = 418 := by
  sorry

end initial_winnings_l17_17415


namespace contrapositive_of_square_sum_zero_l17_17988

theorem contrapositive_of_square_sum_zero (a b : ℝ) :
  (a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0 :=
by
  sorry

end contrapositive_of_square_sum_zero_l17_17988


namespace exists_xy_interval_l17_17558

theorem exists_xy_interval (a b : ℝ) : 
  ∃ (x y : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ |x * y - a * x - b * y| ≥ 1 / 3 :=
sorry

end exists_xy_interval_l17_17558


namespace correct_result_value_at_neg_one_l17_17234

theorem correct_result (x : ℝ) (A : ℝ := 3 * x^2 - x + 1) (incorrect : ℝ := 2 * x^2 - 3 * x - 2) :
  (A - (incorrect - A)) = 4 * x^2 + x + 4 :=
by sorry

theorem value_at_neg_one (x : ℝ := -1) (A : ℝ := 3 * x^2 - x + 1) (incorrect : ℝ := 2 * x^2 - 3 * x - 2) :
  (4 * x^2 + x + 4) = 7 :=
by sorry

end correct_result_value_at_neg_one_l17_17234


namespace find_p_q_l17_17672

noncomputable def f (p q : ℝ) (x : ℝ) : ℝ :=
if x < -1 then p * x + q else 5 * x - 10

theorem find_p_q (p q : ℝ) (h : ∀ x, f p q (f p q x) = x) : p + q = 11 :=
sorry

end find_p_q_l17_17672


namespace find_initial_number_l17_17089

theorem find_initial_number (N : ℕ) (k : ℤ) (h : N - 3 = 15 * k) : N = 18 := 
by
  sorry

end find_initial_number_l17_17089


namespace no_real_solutions_sufficient_not_necessary_l17_17939

theorem no_real_solutions_sufficient_not_necessary (m : ℝ) : 
  (|m| < 1) → (m^2 < 4) :=
by
  sorry

end no_real_solutions_sufficient_not_necessary_l17_17939


namespace radius_of_circle_l17_17543

theorem radius_of_circle:
  (∃ (r: ℝ), 
    (∀ (x: ℝ), (x^2 + r - x) = 0 → 1 - 4 * r = 0)
  ) → r = 1 / 4 := 
sorry

end radius_of_circle_l17_17543


namespace apples_in_market_l17_17342

theorem apples_in_market (A O : ℕ) 
    (h1 : A = O + 27) 
    (h2 : A + O = 301) : 
    A = 164 :=
by
  sorry

end apples_in_market_l17_17342


namespace sum_of_fourth_powers_correct_l17_17913

noncomputable def sum_of_fourth_powers (x : ℤ) : ℤ :=
  x^4 + (x+1)^4 + (x+2)^4

theorem sum_of_fourth_powers_correct (x : ℤ) (h : x * (x+1) * (x+2) = 36 * x + 12) : 
  sum_of_fourth_powers x = 98 :=
sorry

end sum_of_fourth_powers_correct_l17_17913


namespace find_triangle_sides_l17_17840

-- Define the variables and conditions
noncomputable def k := 5
noncomputable def c := 12
noncomputable def d := 10

-- Assume the perimeters of the figures
def P1 : ℕ := 74
def P2 : ℕ := 84
def P3 : ℕ := 82

-- Define the equations based on the perimeters
def Equation1 := P2 = P1 + 2 * k
def Equation2 := P3 = P1 + 6 * c - 2 * k

-- The lean theorem proving that the sides of the triangle are as given
theorem find_triangle_sides : 
  (Equation1 ∧ Equation2) →
  (k = 5 ∧ c = 12 ∧ d = 10) :=
by
  sorry

end find_triangle_sides_l17_17840


namespace matt_profit_trade_l17_17156

theorem matt_profit_trade
  (total_cards : ℕ := 8)
  (value_per_card : ℕ := 6)
  (traded_cards_count : ℕ := 2)
  (trade_value_per_card : ℕ := 6)
  (received_cards_count_1 : ℕ := 3)
  (received_value_per_card_1 : ℕ := 2)
  (received_cards_count_2 : ℕ := 1)
  (received_value_per_card_2 : ℕ := 9)
  (profit : ℕ := 3) :
  profit = (received_cards_count_1 * received_value_per_card_1 
           + received_cards_count_2 * received_value_per_card_2) 
           - (traded_cards_count * trade_value_per_card) :=
  by
  sorry

end matt_profit_trade_l17_17156


namespace four_digit_square_l17_17372

/-- A four-digit square number that satisfies the given conditions -/
theorem four_digit_square (a b c d : ℕ) (h₁ : b + c = a) (h₂ : a + c = 10 * d) :
  1000 * a + 100 * b + 10 * c + d = 6241 :=
sorry

end four_digit_square_l17_17372


namespace can_divide_2007_triangles_can_divide_2008_triangles_l17_17085

theorem can_divide_2007_triangles :
  ∃ k : ℕ, 2007 = 9 + 3 * k :=
by
  sorry

theorem can_divide_2008_triangles :
  ∃ m : ℕ, 2008 = 4 + 3 * m :=
by
  sorry

end can_divide_2007_triangles_can_divide_2008_triangles_l17_17085


namespace Lin_trip_time_l17_17142

theorem Lin_trip_time
  (v : ℕ) -- speed on the mountain road in miles per minute
  (h1 : 80 = d_highway) -- Lin travels 80 miles on the highway
  (h2 : 20 = d_mountain) -- Lin travels 20 miles on the mountain road
  (h3 : v_highway = 2 * v) -- Lin drives twice as fast on the highway
  (h4 : 40 = 20 / v) -- Lin spent 40 minutes driving on the mountain road
  : 40 + 80 = 120 :=
by
  -- proof steps would go here
  sorry

end Lin_trip_time_l17_17142


namespace car_maintenance_expense_l17_17225

-- Define constants and conditions
def miles_per_year : ℕ := 12000
def oil_change_interval : ℕ := 3000
def oil_change_price (quarter : ℕ) : ℕ := 
  if quarter = 1 then 55 
  else if quarter = 2 then 45 
  else if quarter = 3 then 50 
  else 40
def free_oil_changes_per_year : ℕ := 1

def tire_rotation_interval : ℕ := 6000
def tire_rotation_cost : ℕ := 40
def tire_rotation_discount : ℕ := 10 -- In percent

def brake_pad_interval : ℕ := 24000
def brake_pad_cost : ℕ := 200
def brake_pad_discount : ℕ := 20 -- In percent
def brake_pad_membership_cost : ℕ := 60
def membership_duration : ℕ := 2 -- In years

def total_annual_expense : ℕ :=
  let oil_changes := (miles_per_year / oil_change_interval) - free_oil_changes_per_year
  let oil_cost := (oil_change_price 2 + oil_change_price 3 + oil_change_price 4) -- Free oil change in Q1
  let tire_rotations := miles_per_year / tire_rotation_interval
  let tire_cost := (tire_rotation_cost * (100 - tire_rotation_discount) / 100) * tire_rotations
  let brake_pad_cost_per_year := (brake_pad_cost * (100 - brake_pad_discount) / 100) / membership_duration
  let membership_cost_per_year := brake_pad_membership_cost / membership_duration
  oil_cost + tire_cost + (brake_pad_cost_per_year + membership_cost_per_year)

-- Assert the proof problem
theorem car_maintenance_expense : total_annual_expense = 317 := by
  sorry

end car_maintenance_expense_l17_17225


namespace symmetry_about_origin_l17_17255

noncomputable def f (x : ℝ) : ℝ := x * Real.log (-x)
noncomputable def g (x : ℝ) : ℝ := x * Real.log x

theorem symmetry_about_origin :
  ∀ x : ℝ, f (-x) = -g (-x) :=
by
  sorry

end symmetry_about_origin_l17_17255


namespace braden_money_box_total_l17_17880

def initial_money : ℕ := 400

def correct_predictions : ℕ := 3

def betting_rules (correct_predictions : ℕ) : ℕ :=
  match correct_predictions with
  | 1 => 25
  | 2 => 50
  | 3 => 75
  | 4 => 200
  | _ => 0

theorem braden_money_box_total:
  let winnings := (betting_rules correct_predictions * initial_money) / 100
  initial_money + winnings = 700 := 
by
  let winnings := (betting_rules correct_predictions * initial_money) / 100
  show initial_money + winnings = 700
  sorry

end braden_money_box_total_l17_17880


namespace periodicity_iff_condition_l17_17270

-- Define the given conditions
variable (f : ℝ → ℝ)
variable (h_even : ∀ x, f (-x) = f x)

-- State the problem
theorem periodicity_iff_condition :
  (∀ x, f (1 - x) = f (1 + x)) ↔ (∀ x, f (x + 2) = f x) :=
sorry

end periodicity_iff_condition_l17_17270


namespace quadratic_inequality_solution_l17_17611

theorem quadratic_inequality_solution {x : ℝ} :
  (x^2 + x - 6 ≤ 0) ↔ (-3 ≤ x ∧ x ≤ 2) :=
by
  sorry

end quadratic_inequality_solution_l17_17611


namespace boys_from_school_A_study_science_l17_17383

theorem boys_from_school_A_study_science (total_boys school_A_percent non_science_boys school_A_boys study_science_boys: ℕ) 
(h1 : total_boys = 300)
(h2 : school_A_percent = 20)
(h3 : non_science_boys = 42)
(h4 : school_A_boys = (school_A_percent * total_boys) / 100)
(h5 : study_science_boys = school_A_boys - non_science_boys) :
(study_science_boys * 100 / school_A_boys) = 30 :=
by
  sorry

end boys_from_school_A_study_science_l17_17383


namespace equal_parallelogram_faces_are_rhombuses_l17_17827

theorem equal_parallelogram_faces_are_rhombuses 
  (a b c : ℝ) 
  (h: a * b = b * c ∧ b * c = a * c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  a = b ∧ b = c :=
sorry

end equal_parallelogram_faces_are_rhombuses_l17_17827


namespace calculate_decimal_sum_and_difference_l17_17223

theorem calculate_decimal_sum_and_difference : 
  (0.5 + 0.003 + 0.070) - 0.008 = 0.565 := 
by 
  sorry

end calculate_decimal_sum_and_difference_l17_17223


namespace total_earnings_l17_17309

theorem total_earnings (L A J M : ℝ) 
  (hL : L = 2000) 
  (hA : A = 0.70 * L) 
  (hJ : J = 1.50 * A) 
  (hM : M = 0.40 * J) 
  : L + A + J + M = 6340 := 
  by 
    sorry

end total_earnings_l17_17309


namespace fraction_sum_le_41_over_42_l17_17818

theorem fraction_sum_le_41_over_42 (a b c : ℕ) (h : 1/a + 1/b + 1/c < 1) : 1/a + 1/b + 1/c ≤ 41/42 :=
sorry

end fraction_sum_le_41_over_42_l17_17818


namespace Jerry_walked_9_miles_l17_17835

theorem Jerry_walked_9_miles (x : ℕ) (h : 2 * x = 18) : x = 9 := 
by
  sorry

end Jerry_walked_9_miles_l17_17835


namespace hyperbola_focal_length_l17_17300

theorem hyperbola_focal_length : 
  (∃ (f : ℝ) (x y : ℝ), (3 * x^2 - y^2 = 3) ∧ (f = 4)) :=
by {
  sorry
}

end hyperbola_focal_length_l17_17300


namespace average_after_17th_inning_l17_17823

variable (A : ℕ)

-- Definition of total runs before the 17th inning
def total_runs_before := 16 * A

-- Definition of new total runs after the 17th inning
def total_runs_after := total_runs_before A + 87

-- Definition of new average after the 17th inning
def new_average := A + 4

-- Definition of new total runs in terms of new average
def new_total_runs := 17 * new_average A

-- The statement we want to prove
theorem average_after_17th_inning : total_runs_after A = new_total_runs A → new_average A = 23 := by
  sorry

end average_after_17th_inning_l17_17823


namespace necessary_but_not_sufficient_l17_17811

theorem necessary_but_not_sufficient (x : ℝ) : (x < 0) -> (x^2 + x < 0 ↔ -1 < x ∧ x < 0) :=
by
  sorry

end necessary_but_not_sufficient_l17_17811


namespace symmetric_points_product_l17_17918

theorem symmetric_points_product (a b : ℝ) 
    (h1 : a + 2 = -4) 
    (h2 : b = 2) : 
    a * b = -12 := 
sorry

end symmetric_points_product_l17_17918


namespace find_a_from_binomial_l17_17491

variable (x : ℝ) (a : ℝ)

def binomial_term (r : ℕ) : ℝ :=
  (Nat.choose 5 r) * ((-a)^r) * x^(5 - 2 * r)

theorem find_a_from_binomial :
  (∃ x : ℝ, ∃ a : ℝ, (binomial_term x a 1 = 10)) → a = -2 :=
by 
  sorry

end find_a_from_binomial_l17_17491


namespace sarahs_brother_apples_l17_17539

theorem sarahs_brother_apples (x : ℝ) (hx : 5 * x = 45.0) : x = 9.0 :=
by
  sorry

end sarahs_brother_apples_l17_17539


namespace finance_to_manufacturing_ratio_l17_17250

theorem finance_to_manufacturing_ratio : 
    let finance_angle := 72
    let manufacturing_angle := 108
    (finance_angle:ℕ) / (Nat.gcd finance_angle manufacturing_angle) = 2 ∧ 
    (manufacturing_angle:ℕ) / (Nat.gcd finance_angle manufacturing_angle) = 3 := 
by 
    sorry

end finance_to_manufacturing_ratio_l17_17250


namespace remainder_expression_l17_17762

theorem remainder_expression (x y u v : ℕ) (hy_pos : y > 0) (h : x = u * y + v) (hv : 0 ≤ v) (hv_lt : v < y) :
  (x + 4 * u * y) % y = v :=
by
  sorry

end remainder_expression_l17_17762


namespace mother_age_twice_xiaoming_in_18_years_l17_17548

-- Definitions based on conditions
def xiaoming_age_now : ℕ := 6
def mother_age_now : ℕ := 30

theorem mother_age_twice_xiaoming_in_18_years : 
    ∀ (n : ℕ), xiaoming_age_now + n = 24 → mother_age_now + n = 2 * (xiaoming_age_now + n) → n = 18 :=
by
  intro n hn hm
  sorry

end mother_age_twice_xiaoming_in_18_years_l17_17548


namespace tangent_and_normal_lines_l17_17463

noncomputable def x (t : ℝ) := 2 * Real.exp t
noncomputable def y (t : ℝ) := Real.exp (-t)

theorem tangent_and_normal_lines (t0 : ℝ) (x0 y0 : ℝ) (m_tangent m_normal : ℝ)
  (hx0 : x0 = x t0)
  (hy0 : y0 = y t0)
  (hm_tangent : m_tangent = -(1 / 2))
  (hm_normal : m_normal = 2) :
  (∀ x y : ℝ, y = m_tangent * x + 2) ∧ (∀ x y : ℝ, y = m_normal * x - 3) :=
by
  sorry

end tangent_and_normal_lines_l17_17463


namespace div_powers_same_base_l17_17614

variable (x : ℝ)

theorem div_powers_same_base : x^8 / x^2 = x^6 :=
by
  sorry

end div_powers_same_base_l17_17614


namespace circumscribed_sphere_radius_l17_17358

noncomputable def radius_of_circumscribed_sphere (a : ℝ) (α : ℝ) : ℝ :=
  a / (3 * Real.sin α)

theorem circumscribed_sphere_radius (a α : ℝ) :
  radius_of_circumscribed_sphere a α = a / (3 * Real.sin α) :=
by
  sorry

end circumscribed_sphere_radius_l17_17358


namespace find_p_of_probability_l17_17350

-- Define the conditions and the problem statement
theorem find_p_of_probability
  (A_red_prob : ℚ := 1/3) -- probability of drawing a red ball from bag A
  (A_to_B_ratio : ℚ := 1/2) -- ratio of number of balls in bag A to bag B
  (combined_red_prob : ℚ := 2/5) -- total probability of drawing a red ball after combining balls
  : p = 13 / 30 := by
  sorry

end find_p_of_probability_l17_17350


namespace peter_class_students_l17_17595

def total_students (students_with_two_hands students_with_one_hand students_with_three_hands : ℕ) : ℕ :=
  students_with_two_hands + students_with_one_hand + students_with_three_hands + 1

theorem peter_class_students
  (students_with_two_hands students_with_one_hand students_with_three_hands : ℕ)
  (total_hands_without_peter : ℕ) :

  students_with_two_hands = 10 →
  students_with_one_hand = 3 →
  students_with_three_hands = 1 →
  total_hands_without_peter = 20 →
  total_students students_with_two_hands students_with_one_hand students_with_three_hands = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end peter_class_students_l17_17595


namespace range_of_a_l17_17196

noncomputable def is_decreasing (a : ℝ) : Prop :=
∀ n : ℕ, 0 < n → n ≤ 6 → (1 - 3 * a) * n + 10 * a > (1 - 3 * a) * (n + 1) + 10 * a ∧ 0 < a ∧ a < 1 ∧ ((1 - 3 * a) * 6 + 10 * a > 1)

theorem range_of_a (a : ℝ) : is_decreasing a ↔ (1/3 < a ∧ a < 5/8) :=
sorry

end range_of_a_l17_17196


namespace problem_126_times_3_pow_6_l17_17549

theorem problem_126_times_3_pow_6 (p : ℝ) (h : 126 * 3^8 = p) : 
  126 * 3^6 = (1 / 9) * p := 
by {
  -- Placeholder for the proof
  sorry
}

end problem_126_times_3_pow_6_l17_17549


namespace ratio_boys_to_girls_l17_17204

variable (g b : ℕ)

theorem ratio_boys_to_girls (h1 : b = g + 9) (h2 : g + b = 25) : b / g = 17 / 8 := by
  -- Proof goes here
  sorry

end ratio_boys_to_girls_l17_17204


namespace find_k_l17_17446

theorem find_k (k : ℝ) (h : ∃ (k : ℝ), 3 = k * (-1) - 2) : k = -5 :=
by
  rcases h with ⟨k, hk⟩
  sorry

end find_k_l17_17446


namespace dennis_took_away_l17_17930

-- Define the initial and remaining number of cards
def initial_cards : ℕ := 67
def remaining_cards : ℕ := 58

-- Define the number of cards taken away
def cards_taken_away (n m : ℕ) : ℕ := n - m

-- Prove that the number of cards taken away is 9
theorem dennis_took_away :
  cards_taken_away initial_cards remaining_cards = 9 :=
by
  -- Placeholder proof
  sorry

end dennis_took_away_l17_17930


namespace verify_equation_l17_17569

theorem verify_equation : (3^2 + 5^2)^2 = 16^2 + 30^2 := by
  sorry

end verify_equation_l17_17569


namespace logarithmic_expression_range_l17_17907

theorem logarithmic_expression_range (a : ℝ) : 
  (a - 2 > 0) ∧ (5 - a > 0) ∧ (a - 2 ≠ 1) ↔ (2 < a ∧ a < 3) ∨ (3 < a ∧ a < 5) := 
by
  sorry

end logarithmic_expression_range_l17_17907


namespace find_value_of_a_3m_2n_l17_17508

variable {a : ℝ} {m n : ℕ}
axiom h1 : a ^ m = 2
axiom h2 : a ^ n = 5

theorem find_value_of_a_3m_2n : a ^ (3 * m - 2 * n) = 8 / 25 := by
  sorry

end find_value_of_a_3m_2n_l17_17508


namespace river_flow_rate_l17_17282

theorem river_flow_rate
  (depth width volume_per_minute : ℝ)
  (h1 : depth = 2)
  (h2 : width = 45)
  (h3 : volume_per_minute = 6000) :
  (volume_per_minute / (depth * width)) * (1 / 1000) * 60 = 4.0002 :=
by
  -- Sorry is used to skip the proof.
  sorry

end river_flow_rate_l17_17282


namespace joan_gave_28_seashells_to_sam_l17_17378

/-- 
Given:
- Joan found 70 seashells on the beach.
- After giving away some seashells, she has 27 left.
- She gave twice as many seashells to Sam as she gave to her friend Lily.

Show that:
- Joan gave 28 seashells to Sam.
-/
theorem joan_gave_28_seashells_to_sam (L S : ℕ) 
  (h1 : S = 2 * L) 
  (h2 : 70 - 27 = 43) 
  (h3 : L + S = 43) :
  S = 28 :=
by
  sorry

end joan_gave_28_seashells_to_sam_l17_17378


namespace christmas_gift_count_l17_17752

theorem christmas_gift_count (initial_gifts : ℕ) (additional_gifts : ℕ) (gifts_to_orphanage : ℕ)
  (h1 : initial_gifts = 77)
  (h2 : additional_gifts = 33)
  (h3 : gifts_to_orphanage = 66) :
  (initial_gifts + additional_gifts - gifts_to_orphanage = 44) :=
by
  sorry

end christmas_gift_count_l17_17752


namespace range_of_x_l17_17496

theorem range_of_x (x : ℝ) : (4 : ℝ)^(2 * x - 1) > (1 / 2) ^ (-x - 4) → x > 2 := by
  sorry

end range_of_x_l17_17496


namespace find_a_value_l17_17425

theorem find_a_value :
  (∀ y : ℝ, y ∈ Set.Ioo (-3/2 : ℝ) 4 → y * (2 * y - 3) < (12 : ℝ)) ↔ (12 = 12) := 
by 
  sorry

end find_a_value_l17_17425


namespace sugar_percentage_l17_17637

theorem sugar_percentage (x : ℝ) (h2 : 50 ≤ 100) (h1 : 1 / 4 * x + 12.5 = 20) : x = 10 :=
by
  sorry

end sugar_percentage_l17_17637


namespace clock_angle_8_30_l17_17500

theorem clock_angle_8_30 
  (angle_per_hour_mark : ℝ := 30)
  (angle_per_minute_mark : ℝ := 6)
  (hour_hand_angle_8 : ℝ := 8 * angle_per_hour_mark)
  (half_hour_movement : ℝ := 0.5 * angle_per_hour_mark)
  (hour_hand_angle_8_30 : ℝ := hour_hand_angle_8 + half_hour_movement)
  (minute_hand_angle_30 : ℝ := 30 * angle_per_minute_mark) :
  abs (hour_hand_angle_8_30 - minute_hand_angle_30) = 75 :=
by
  sorry

end clock_angle_8_30_l17_17500


namespace bowling_ball_volume_l17_17506

open Real

noncomputable def remaining_volume (d_bowling_ball d1 d2 d3 d4 h1 h2 h3 h4 : ℝ) : ℝ :=
  let r_bowling_ball := d_bowling_ball / 2
  let v_bowling_ball := (4/3) * π * (r_bowling_ball ^ 3)
  let v_hole1 := π * ((d1 / 2) ^ 2) * h1
  let v_hole2 := π * ((d2 / 2) ^ 2) * h2
  let v_hole3 := π * ((d3 / 2) ^ 2) * h3
  let v_hole4 := π * ((d4 / 2) ^ 2) * h4
  v_bowling_ball - (v_hole1 + v_hole2 + v_hole3 + v_hole4)

theorem bowling_ball_volume :
  remaining_volume 40 3 3 4 5 10 10 12 8 = 10523.67 * π :=
by
  sorry

end bowling_ball_volume_l17_17506


namespace minimum_value_expression_l17_17279

noncomputable def expr (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2 - 2*x - 2*y + 2) + 
  Real.sqrt (x^2 + y^2 - 2*x + 4*y + 2*Real.sqrt 3*y + 8 + 4*Real.sqrt 3) +
  Real.sqrt (x^2 + y^2 + 8*x + 4*Real.sqrt 3*x - 4*y + 32 + 16*Real.sqrt 3)

theorem minimum_value_expression : (∃ x y : ℝ, expr x y = 3*Real.sqrt 6 + 4*Real.sqrt 2) :=
sorry

end minimum_value_expression_l17_17279


namespace gumballs_in_packages_l17_17241

theorem gumballs_in_packages (total_gumballs : ℕ) (gumballs_per_package : ℕ) (h1 : total_gumballs = 20) (h2 : gumballs_per_package = 5) :
  total_gumballs / gumballs_per_package = 4 :=
by {
  sorry
}

end gumballs_in_packages_l17_17241


namespace complete_square_l17_17458

theorem complete_square (x : ℝ) : (x^2 - 4*x + 2 = 0) → ((x - 2)^2 = 2) :=
by
  intro h
  sorry

end complete_square_l17_17458


namespace kayla_score_fourth_level_l17_17160

theorem kayla_score_fourth_level 
  (score1 score2 score3 score5 score6 : ℕ) 
  (h1 : score1 = 2) 
  (h2 : score2 = 3) 
  (h3 : score3 = 5) 
  (h5 : score5 = 12) 
  (h6 : score6 = 17)
  (h_diff : ∀ n : ℕ, score2 - score1 + n = score3 - score2 + n + 1 ∧ score3 - score2 + n + 2 = score5 - score3 + n + 3 ∧ score5 - score3 + n + 4 = score6 - score5 + n + 5) :
  ∃ score4 : ℕ, score4 = 8 :=
by
  sorry

end kayla_score_fourth_level_l17_17160


namespace sin_x_plus_pi_l17_17863

theorem sin_x_plus_pi {x : ℝ} (hx : Real.sin x = -4 / 5) : Real.sin (x + Real.pi) = 4 / 5 :=
by
  -- Proof steps go here
  sorry

end sin_x_plus_pi_l17_17863


namespace simplify_expression_l17_17577

theorem simplify_expression (b : ℝ) (h1 : b ≠ 1) (h2 : b ≠ 1 / 2) :
  (1 / 2 - 1 / (1 + b / (1 - 2 * b))) = (3 * b - 1) / (2 * (1 - b)) :=
sorry

end simplify_expression_l17_17577


namespace inequality_and_equality_condition_l17_17137

theorem inequality_and_equality_condition (a b : ℝ) :
  a^2 + 4 * b^2 + 4 * b - 4 * a + 5 ≥ 0 ∧ (a^2 + 4 * b^2 + 4 * b - 4 * a + 5 = 0 ↔ (a = 2 ∧ b = -1 / 2)) :=
by
  sorry

end inequality_and_equality_condition_l17_17137


namespace count_odd_expressions_l17_17613

theorem count_odd_expressions : 
  let exp1 := 1^2
  let exp2 := 2^3
  let exp3 := 3^4
  let exp4 := 4^5
  let exp5 := 5^6
  (if exp1 % 2 = 1 then 1 else 0) + 
  (if exp2 % 2 = 1 then 1 else 0) + 
  (if exp3 % 2 = 1 then 1 else 0) + 
  (if exp4 % 2 = 1 then 1 else 0) + 
  (if exp5 % 2 = 1 then 1 else 0) = 3 :=
by 
  sorry

end count_odd_expressions_l17_17613


namespace identity_solution_l17_17743

theorem identity_solution (x : ℝ) :
  ∃ a b : ℝ, (2 * x + a) ^ 3 = 5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x ∧
             a = -1 ∧ b = 1 :=
by
  -- we can skip the proof as this is just a statement
  sorry

end identity_solution_l17_17743


namespace total_area_to_be_painted_l17_17856

theorem total_area_to_be_painted (length width height partition_length partition_height : ℝ) 
(partition_along_length inside_outside both_sides : Bool)
(h1 : length = 15)
(h2 : width = 12)
(h3 : height = 6)
(h4 : partition_length = 15)
(h5 : partition_height = 6) 
(h_partition_along_length : partition_along_length = true)
(h_inside_outside : inside_outside = true)
(h_both_sides : both_sides = true) :
    let end_wall_area := 2 * 2 * width * height
    let side_wall_area := 2 * 2 * length * height
    let ceiling_area := length * width
    let partition_area := 2 * partition_length * partition_height
    (end_wall_area + side_wall_area + ceiling_area + partition_area) = 1008 :=
by
    sorry

end total_area_to_be_painted_l17_17856


namespace proof_system_solution_l17_17450

noncomputable def solve_system : Prop :=
  ∃ x y : ℚ, x + 4 * y = 14 ∧ (x - 3) / 4 - (y - 3) / 3 = 1 / 12 ∧ x = 3 ∧ y = 11 / 4

theorem proof_system_solution : solve_system :=
sorry

end proof_system_solution_l17_17450


namespace solution_to_problem_l17_17187

theorem solution_to_problem (x y : ℕ) (h : (2*x - 5) * (2*y - 5) = 25) : x + y = 10 ∨ x + y = 18 := by
  sorry

end solution_to_problem_l17_17187


namespace remainder_12401_163_l17_17661

theorem remainder_12401_163 :
  let original_number := 12401
  let divisor := 163
  let quotient := 76
  let remainder := 13
  original_number = divisor * quotient + remainder :=
by
  sorry

end remainder_12401_163_l17_17661


namespace bolton_class_students_l17_17848

theorem bolton_class_students 
  (S : ℕ) 
  (H1 : 2/5 < 1)
  (H2 : 1/3 < 1)
  (C1 : (2 / 5) * (S:ℝ) + (2 / 5) * (S:ℝ) = 20) : 
  S = 25 := 
by
  sorry

end bolton_class_students_l17_17848


namespace positive_integer_solutions_equation_l17_17101

theorem positive_integer_solutions_equation (x y : ℕ) (positive_x : x > 0) (positive_y : y > 0) :
  x^2 + 6 * x * y - 7 * y^2 = 2009 ↔ (x = 252 ∧ y = 251) ∨ (x = 42 ∧ y = 35) ∨ (x = 42 ∧ y = 1) :=
sorry

end positive_integer_solutions_equation_l17_17101


namespace jaden_toy_cars_left_l17_17308

-- Definitions for each condition
def initial_toys : ℕ := 14
def purchased_toys : ℕ := 28
def birthday_toys : ℕ := 12
def given_to_sister : ℕ := 8
def given_to_vinnie : ℕ := 3
def traded_lost : ℕ := 5
def traded_received : ℕ := 7

-- The final number of toy cars Jaden has
def final_toys : ℕ :=
  initial_toys + purchased_toys + birthday_toys - given_to_sister - given_to_vinnie + (traded_received - traded_lost)

theorem jaden_toy_cars_left : final_toys = 45 :=
by
  -- The proof will be filled in here 
  sorry

end jaden_toy_cars_left_l17_17308


namespace middle_digit_base5_l17_17190

theorem middle_digit_base5 {M : ℕ} (x y z : ℕ) (hx : 0 ≤ x ∧ x < 5) (hy : 0 ≤ y ∧ y < 5) (hz : 0 ≤ z ∧ z < 5)
    (h_base5 : M = 25 * x + 5 * y + z) (h_base8 : M = 64 * z + 8 * y + x) : y = 0 :=
sorry

end middle_digit_base5_l17_17190


namespace triangle_angle_bisector_segment_length_l17_17180

theorem triangle_angle_bisector_segment_length
  (DE DF EF DG EG : ℝ)
  (h_ratio : DE / 12 = 1 ∧ DF / DE = 4 / 3 ∧ EF / DE = 5 / 3)
  (h_angle_bisector : DG / EG = DE / DF ∧ DG + EG = EF) :
  EG = 80 / 7 :=
by
  sorry

end triangle_angle_bisector_segment_length_l17_17180


namespace four_friends_total_fish_l17_17251

-- Define the number of fish each friend has based on the conditions
def micah_fish : ℕ := 7
def kenneth_fish : ℕ := 3 * micah_fish
def matthias_fish : ℕ := kenneth_fish - 15
def total_three_boys_fish : ℕ := micah_fish + kenneth_fish + matthias_fish
def gabrielle_fish : ℕ := 2 * total_three_boys_fish
def total_fish : ℕ := micah_fish + kenneth_fish + matthias_fish + gabrielle_fish

-- The proof goal
theorem four_friends_total_fish : total_fish = 102 :=
by
  -- We assume the proof steps are correct and leave the proof part as sorry
  sorry

end four_friends_total_fish_l17_17251


namespace ab_value_in_triangle_l17_17763

theorem ab_value_in_triangle (a b c : ℝ) (C : ℝ) (h1 : (a + b)^2 - c^2 = 4) (h2 : C = 60) :
  a * b = 4 / 3 :=
by sorry

end ab_value_in_triangle_l17_17763


namespace harry_total_cost_l17_17226

noncomputable def total_cost : ℝ :=
let small_price := 10
let medium_price := 12
let large_price := 14
let small_topping_price := 1.50
let medium_topping_price := 1.75
let large_topping_price := 2
let small_pizzas := 1
let medium_pizzas := 2
let large_pizzas := 1
let small_toppings := 2
let medium_toppings := 3
let large_toppings := 4
let item_cost : ℝ := (small_pizzas * small_price + medium_pizzas * medium_price + large_pizzas * large_price)
let topping_cost : ℝ := 
  (small_pizzas * small_toppings * small_topping_price) + 
  (medium_pizzas * medium_toppings * medium_topping_price) +
  (large_pizzas * large_toppings * large_topping_price)
let garlic_knots := 2 * 3 -- 2 sets of 5 knots at $3 each
let soda := 2
let replace_total := item_cost + topping_cost
let discounted_total := replace_total - 0.1 * item_cost
let subtotal := discounted_total + garlic_knots + soda
let tax := 0.08 * subtotal
let total_with_tax := subtotal + tax
let tip := 0.25 * total_with_tax
total_with_tax + tip

theorem harry_total_cost : total_cost = 98.15 := by
  sorry

end harry_total_cost_l17_17226


namespace percentage_ethanol_in_fuel_B_l17_17266

-- Definitions from the conditions
def tank_capacity : ℝ := 218
def ethanol_percentage_fuel_A : ℝ := 0.12
def total_ethanol : ℝ := 30
def volume_of_fuel_A : ℝ := 122

-- Expression to calculate ethanol in Fuel A
def ethanol_in_fuel_A : ℝ := ethanol_percentage_fuel_A * volume_of_fuel_A

-- The remaining ethanol in Fuel B = Total ethanol - Ethanol in Fuel A
def ethanol_in_fuel_B : ℝ := total_ethanol - ethanol_in_fuel_A

-- The volume of fuel B used to fill the tank
def volume_of_fuel_B : ℝ := tank_capacity - volume_of_fuel_A

-- Statement to prove:
theorem percentage_ethanol_in_fuel_B : (ethanol_in_fuel_B / volume_of_fuel_B) * 100 = 16 :=
sorry

end percentage_ethanol_in_fuel_B_l17_17266


namespace focus_of_parabola_l17_17479

theorem focus_of_parabola (y : ℝ → ℝ) (h : ∀ x, y x = 16 * x^2) : 
    ∃ p, p = (0, 1/64) := 
by
    existsi (0, 1/64)
    -- The proof would go here, but we are adding sorry to skip it 
    sorry

end focus_of_parabola_l17_17479


namespace absent_children_count_l17_17274

theorem absent_children_count : ∀ (total_children present_children absent_children bananas : ℕ), 
  total_children = 260 → 
  bananas = 2 * total_children → 
  bananas = 4 * present_children → 
  present_children + absent_children = total_children →
  absent_children = 130 :=
by
  intros total_children present_children absent_children bananas h1 h2 h3 h4
  sorry

end absent_children_count_l17_17274


namespace even_three_digit_numbers_less_than_600_l17_17115

def count_even_three_digit_numbers : ℕ :=
  let hundreds_choices := 5
  let tens_choices := 6
  let units_choices := 3
  hundreds_choices * tens_choices * units_choices

theorem even_three_digit_numbers_less_than_600 : count_even_three_digit_numbers = 90 := by
  -- sorry ensures that the statement type checks even without the proof.
  sorry

end even_three_digit_numbers_less_than_600_l17_17115


namespace value_range_of_sum_difference_l17_17024

theorem value_range_of_sum_difference (a b c : ℝ) (h₁ : a < b)
  (h₂ : a + b = b / a) (h₃ : a * b = c / a) (h₄ : a + b > c)
  (h₅ : a + c > b) (h₆ : b + c > a) : 
  ∃ x y, x = 7 / 8 ∧ y = Real.sqrt 5 - 1 ∧ x < a + b - c ∧ a + b - c < y := sorry

end value_range_of_sum_difference_l17_17024


namespace sum_of_sides_eq_l17_17036

open Real

theorem sum_of_sides_eq (a h : ℝ) (α : ℝ) (ha : a > 0) (hh : h > 0) (hα : 0 < α ∧ α < π) :
  ∃ b c : ℝ, b + c = sqrt (a^2 + 2 * a * h * (cos (α / 2) / sin (α / 2))) :=
by
  sorry

end sum_of_sides_eq_l17_17036


namespace correct_choices_l17_17832

theorem correct_choices :
  (∃ u : ℝ × ℝ, (2 * u.1 + u.2 + 3 = 0) → u = (1, -2)) ∧
  ¬ (∀ a : ℝ, (a = -1 ↔ a^2 * x - y + 1 = 0 ∧ x - a * y - 2 = 0) → a = -1) ∧
  ((∃ (l : ℝ) (P : ℝ × ℝ), l = x + y - 6 → P = (2, 4) → 2 + 4 = l) → x + y - 6 = 0) ∧
  ((∃ (m b : ℝ), y = m * x + b → b = -2) → y = 3 * x - 2) :=
sorry

end correct_choices_l17_17832


namespace find_PQ_l17_17228

noncomputable def right_triangle_tan (PQ PR : ℝ) (tan_P : ℝ) (R_right : Prop) : Prop :=
  tan_P = PQ / PR ∧ R_right

theorem find_PQ (PQ PR : ℝ) (tan_P : ℝ) (R_right : Prop)
  (h1 : tan_P = 3 / 2)
  (h2 : PR = 6)
  (h3 : R_right) :
  right_triangle_tan PQ PR tan_P R_right → PQ = 9 :=
by
  sorry

end find_PQ_l17_17228


namespace multiple_of_B_share_l17_17429

theorem multiple_of_B_share (A B C : ℝ) (k : ℝ) 
    (h1 : 3 * A = k * B) 
    (h2 : k * B = 7 * 84) 
    (h3 : C = 84)
    (h4 : A + B + C = 427) :
    k = 4 :=
by
  -- We do not need the detailed proof steps here.
  sorry

end multiple_of_B_share_l17_17429


namespace max_value_expr_l17_17349

theorem max_value_expr (a b c d : ℝ) (ha : -4 ≤ a ∧ a ≤ 4) (hb : -4 ≤ b ∧ b ≤ 4) (hc : -4 ≤ c ∧ c ≤ 4) (hd : -4 ≤ d ∧ d ≤ 4) :
  (a + 2*b + c + 2*d - a*b - b*c - c*d - d*a) ≤ 72 :=
sorry

end max_value_expr_l17_17349


namespace divide_and_add_l17_17394

variable (number : ℝ)

theorem divide_and_add (h : 4 * number = 166.08) : number / 4 + 0.48 = 10.86 := by
  -- assume the proof follows accurately
  sorry

end divide_and_add_l17_17394


namespace average_temperature_l17_17256

theorem average_temperature (temps : List ℕ) (temps_eq : temps = [40, 47, 45, 41, 39]) :
  (temps.sum : ℚ) / temps.length = 42.4 :=
by
  sorry

end average_temperature_l17_17256


namespace problem1_problem2_problem3_l17_17219

-- Definitions of arithmetic and geometric sequences
def arithmetic (a_n : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a_n n = a_n 0 + n * d
def geometric (b_n : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, b_n n = b_n 0 * q ^ n
def E (m p r : ℕ) := m < p ∧ p < r
def common_difference_greater_than_one (m p r : ℕ) := (p - m = r - p) ∧ (p - m > 1)

-- Problem (1)
theorem problem1 (a_n b_n : ℕ → ℝ) (d q : ℝ) (h₁: arithmetic a_n d) (h₂: geometric b_n q) (hne: q ≠ 1 ∧ q ≠ -1)
  (h: a_n 0 + b_n 1 = a_n 1 + b_n 2 ∧ a_n 1 + b_n 2 = a_n 2 + b_n 0) :
  q = -1/2 :=
sorry

-- Problem (2)
theorem problem2 (a_n b_n : ℕ → ℝ) (d q : ℝ) (m p r : ℕ) (h₁: arithmetic a_n d) (h₂: geometric b_n q) (hne: q ≠ 1 ∧ q ≠ -1)
  (hE: E m p r) (hDiff: common_difference_greater_than_one m p r)
  (h: a_n m + b_n p = a_n p + b_n r ∧ a_n p + b_n r = a_n r + b_n m) :
  q = - (1/2)^(1/3) :=
sorry

-- Problem (3)
theorem problem3 (a_n b_n : ℕ → ℝ) (m p r : ℕ) (hE: E m p r)
  (hG: ∀ n : ℕ, b_n n = (-1/2)^((n:ℕ)-1)) (h: a_n m + b_n m = 0 ∧ a_n p + b_n p = 0 ∧ a_n r + b_n r = 0) :
  ∃ (E : ℕ × ℕ × ℕ) (a : ℕ → ℝ), (E = ⟨1, 3, 4⟩ ∧ ∀ n : ℕ, a n = 3/8 * n - 11/8) :=
sorry

end problem1_problem2_problem3_l17_17219


namespace polynomial_no_real_roots_l17_17664

def f (x : ℝ) : ℝ := 4 * x ^ 8 - 2 * x ^ 7 + x ^ 6 - 3 * x ^ 4 + x ^ 2 - x + 1

theorem polynomial_no_real_roots : ∀ x : ℝ, f x > 0 := by
  sorry

end polynomial_no_real_roots_l17_17664


namespace avg_width_is_3_5_l17_17284

def book_widths : List ℚ := [4, (3/4), 1.25, 3, 2, 7, 5.5]

noncomputable def average (l : List ℚ) : ℚ :=
  l.sum / l.length

theorem avg_width_is_3_5 : average book_widths = 23.5 / 7 :=
by
  sorry

end avg_width_is_3_5_l17_17284


namespace wall_clock_ring_interval_l17_17321

theorem wall_clock_ring_interval 
  (n : ℕ)                -- Number of rings in a day
  (total_minutes : ℕ)    -- Total minutes in a day
  (intervals : ℕ) :       -- Number of intervals
  n = 6 ∧ total_minutes = 1440 ∧ intervals = n - 1 ∧ intervals = 5
    → (1440 / intervals = 288 ∧ 288 / 60 = 4∧ 288 % 60 = 48) := sorry

end wall_clock_ring_interval_l17_17321


namespace area_percentage_l17_17080

theorem area_percentage (D_S D_R : ℝ) (h : D_R = 0.8 * D_S) : 
  let R_S := D_S / 2
  let R_R := D_R / 2
  let A_S := π * R_S^2
  let A_R := π * R_R^2
  (A_R / A_S) * 100 = 64 := 
by
  sorry

end area_percentage_l17_17080


namespace pencils_per_box_l17_17285

-- Variables and Definitions based on the problem conditions
def num_boxes : ℕ := 10
def pencils_kept : ℕ := 10
def friends : ℕ := 5
def pencils_per_friend : ℕ := 8

-- Theorem to prove the solution
theorem pencils_per_box (pencils_total : ℕ)
  (h1 : pencils_total = pencils_kept + (friends * pencils_per_friend))
  (h2 : pencils_total = num_boxes * (pencils_total / num_boxes)) :
  (pencils_total / num_boxes) = 5 :=
sorry

end pencils_per_box_l17_17285


namespace hyperbola_asymptotes_l17_17038

theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
                              (h3 : 2 * a = 4) (h4 : 2 * b = 6) : 
                              ∀ x y : ℝ, (y = (3 / 2) * x) ∨ (y = - (3 / 2) * x) := by
  sorry

end hyperbola_asymptotes_l17_17038


namespace ratio_of_second_to_first_l17_17691

theorem ratio_of_second_to_first (A1 A2 A3 : ℕ) (h1 : A1 = 600) (h2 : A3 = A1 + A2 - 400) (h3 : A1 + A2 + A3 = 3200) : A2 / A1 = 2 :=
by
  sorry

end ratio_of_second_to_first_l17_17691


namespace sqrt_sum_eq_five_l17_17872

theorem sqrt_sum_eq_five
  (x : ℝ)
  (h1 : -Real.sqrt 15 ≤ x ∧ x ≤ Real.sqrt 15)
  (h2 : Real.sqrt (25 - x^2) - Real.sqrt (15 - x^2) = 2) :
  Real.sqrt (25 - x^2) + Real.sqrt (15 - x^2) = 5 := by
  sorry

end sqrt_sum_eq_five_l17_17872


namespace cos_double_angle_at_origin_l17_17374

noncomputable def vertex : ℝ × ℝ := (0, 0)
noncomputable def initial_side : ℝ × ℝ := (1, 0)
noncomputable def terminal_side : ℝ × ℝ := (-1, 3)
noncomputable def cos2alpha (v i t : ℝ × ℝ) : ℝ :=
  2 * ((t.1) / (Real.sqrt (t.1 ^ 2 + t.2 ^ 2))) ^ 2 - 1

theorem cos_double_angle_at_origin :
  cos2alpha vertex initial_side terminal_side = -4 / 5 :=
by
  sorry

end cos_double_angle_at_origin_l17_17374


namespace sum_of_possible_a_l17_17843

theorem sum_of_possible_a (a : ℤ) :
  (∃ x : ℕ, x - (2 - a * x) / 6 = x / 3 - 1) →
  a = -19 :=
sorry

end sum_of_possible_a_l17_17843


namespace max_sum_product_l17_17845

theorem max_sum_product (a b c d : ℝ) (h_nonneg: a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) (h_sum: a + b + c + d = 200) : 
  ab + bc + cd + da ≤ 10000 := 
sorry

end max_sum_product_l17_17845


namespace Mr_Mayer_purchase_price_l17_17774

theorem Mr_Mayer_purchase_price 
  (P : ℝ) 
  (H1 : (1.30 * 2) * P = 2600) : 
  P = 1000 := 
by
  sorry

end Mr_Mayer_purchase_price_l17_17774


namespace fred_balloons_remaining_l17_17189

theorem fred_balloons_remaining 
    (initial_balloons : ℕ)         -- Fred starts with these many balloons
    (given_to_sandy : ℕ)           -- Fred gives these many balloons to Sandy
    (given_to_bob : ℕ)             -- Fred gives these many balloons to Bob
    (h1 : initial_balloons = 709) 
    (h2 : given_to_sandy = 221) 
    (h3 : given_to_bob = 153) : 
    (initial_balloons - given_to_sandy - given_to_bob = 335) :=
by
  sorry

end fred_balloons_remaining_l17_17189


namespace chemistry_class_size_l17_17037

theorem chemistry_class_size
  (total_students : ℕ)
  (chem_bio_both : ℕ)
  (bio_students : ℕ)
  (chem_students : ℕ)
  (both_students : ℕ)
  (H1 : both_students = 8)
  (H2 : bio_students + chem_students + both_students = total_students)
  (H3 : total_students = 70)
  (H4 : chem_students = 2 * (bio_students + both_students)) :
  chem_students + both_students = 52 :=
by
  sorry

end chemistry_class_size_l17_17037


namespace number_of_cannoneers_l17_17521

-- Define the variables for cannoneers, women, and men respectively
variables (C W M : ℕ)

-- Define the conditions as assumptions
def conditions : Prop :=
  W = 2 * C ∧
  M = 2 * W ∧
  M + W = 378

-- Prove that the number of cannoneers is 63
theorem number_of_cannoneers (h : conditions C W M) : C = 63 :=
by sorry

end number_of_cannoneers_l17_17521


namespace vector_dot_product_proof_l17_17775

variable (a b : ℝ × ℝ)

def dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2

theorem vector_dot_product_proof
  (h1 : a = (1, -3))
  (h2 : b = (3, 7)) :
  dot_product a b = -18 :=
by 
  sorry

end vector_dot_product_proof_l17_17775


namespace unknown_subtraction_problem_l17_17532

theorem unknown_subtraction_problem (x y : ℝ) (h1 : x = 40) (h2 : x / 4 * 5 + 10 - y = 48) : y = 12 :=
by
  sorry

end unknown_subtraction_problem_l17_17532


namespace problem1_problem2_l17_17681

-- Problem 1
theorem problem1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  a * b + b * c + c * a ≤ 1 / 3 :=
sorry

-- Problem 2
theorem problem2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b) :
  2 * a ^ 3 - b ^ 3 ≥ 2 * a * b ^ 2 - a ^ 2 * b :=
sorry

end problem1_problem2_l17_17681


namespace min_value_fraction_8_l17_17071

noncomputable def min_value_of_fraction (x y: ℝ) : Prop :=
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (x, y - 1)
  let parallel := (3 * (y - 1)) = (-2) * x
  x > 0 ∧ y > 0 ∧ parallel → (∀ z, z = (3 / x) + (2 / y) → z ≥ 8)

theorem min_value_fraction_8 (x y : ℝ) (h_posx : x > 0) (h_posy : y > 0) :
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (x, y - 1)
  let parallel := (3 * (y - 1)) = (-2) * x
  parallel → (3 / x) + (2 / y) ≥ 8 :=
by
  sorry

end min_value_fraction_8_l17_17071


namespace infinite_triangle_area_sum_l17_17323

noncomputable def rectangle_area_sum : ℝ :=
  let AB := 2
  let BC := 1
  let Q₁ := 0.5
  let base_area := (1/2) * Q₁ * (1/4)
  base_area * (1/(1 - 1/4))

theorem infinite_triangle_area_sum :
  rectangle_area_sum = 1/12 :=
by
  sorry

end infinite_triangle_area_sum_l17_17323


namespace S8_is_80_l17_17849

variable {a : ℕ → ℝ} -- sequence definition
variable {S : ℕ → ℝ} -- sum of sequence

-- Conditions
variable (h_seq : ∀ n, a (n + 1) = a n + d) -- arithmetic sequence
variable (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) -- sum of the first n terms
variable (h_cond : a 3 = 20 - a 6) -- given condition

theorem S8_is_80 (h_seq : ∀ n, a (n + 1) = a n + d) (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) (h_cond : a 3 = 20 - a 6) :
  S 8 = 80 :=
sorry

end S8_is_80_l17_17849


namespace sin_value_l17_17183

theorem sin_value (α : ℝ) 
  (h : Real.sin (2 * Real.pi / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5) :
  Real.sin (α + 7 * Real.pi / 6) = -4 / 5 :=
by
  sorry

end sin_value_l17_17183


namespace average_speed_l17_17854

theorem average_speed (D T : ℝ) (hD : D = 200) (hT : T = 6) : D / T = 33.33 := by
  -- Sorry is used to skip the proof, only the statement is provided as per instruction
  sorry

end average_speed_l17_17854


namespace cricket_team_throwers_l17_17527

def cricket_equation (T N : ℕ) := 
  (2 * N / 3 = 51 - T) ∧ (T + N = 58)

theorem cricket_team_throwers : 
  ∃ T : ℕ, ∃ N : ℕ, cricket_equation T N ∧ T = 37 :=
by
  sorry

end cricket_team_throwers_l17_17527


namespace find_a_l17_17943

theorem find_a (a : ℝ) : 
  let A := {1, 2, 3}
  let B := {x : ℝ | x^2 - (a + 1) * x + a = 0}
  A ∪ B = A → a = 1 ∨ a = 2 ∨ a = 3 :=
by
  intros
  sorry

end find_a_l17_17943


namespace martin_walk_distance_l17_17505

-- Define the conditions
def time : ℝ := 6 -- Martin's walking time in hours
def speed : ℝ := 2 -- Martin's walking speed in miles per hour

-- Define the target distance
noncomputable def distance : ℝ := 12 -- Distance from Martin's house to Lawrence's house

-- The theorem to prove the target distance given the conditions
theorem martin_walk_distance : (speed * time = distance) :=
by
  sorry

end martin_walk_distance_l17_17505


namespace hyperbola_condition_l17_17767

theorem hyperbola_condition (m n : ℝ) : 
  (mn < 0) ↔ (∀ x y : ℝ, ∃ k ∈ {a : ℝ | a ≠ 0}, (x^2 / m + y^2 / n = 1)) := sorry

end hyperbola_condition_l17_17767


namespace intersection_y_sum_zero_l17_17871

theorem intersection_y_sum_zero :
  ∀ (x1 y1 x2 y2 : ℝ), (y1 = 2 * x1) ∧ (y1 = 2 / x1) ∧ (y2 = 2 * x2) ∧ (y2 = 2 / x2) →
  (x2 = -x1) ∧ (y2 = -y1) →
  y1 + y2 = 0 :=
by
  sorry

end intersection_y_sum_zero_l17_17871


namespace laborer_income_l17_17895

theorem laborer_income (I : ℕ) (debt : ℕ) 
  (h1 : 6 * I < 420) 
  (h2 : 4 * I = 240 + debt + 30) 
  (h3 : debt = 420 - 6 * I) : 
  I = 69 := by
  sorry

end laborer_income_l17_17895


namespace range_of_m_l17_17365

variable (m : ℝ)

def prop_p : Prop := ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + m*x1 + 1 = 0) ∧ (x2^2 + m*x2 + 1 = 0)

def prop_q : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

theorem range_of_m (h₁ : prop_p m) (h₂ : ¬prop_q m) : m < -2 ∨ m ≥ 3 :=
sorry

end range_of_m_l17_17365


namespace number_of_arrangements_l17_17721

def basil_plants := 2
def aloe_plants := 1
def cactus_plants := 1
def white_lamps := 2
def red_lamps := 2
def total_plants := basil_plants + aloe_plants + cactus_plants
def total_lamps := white_lamps + red_lamps

theorem number_of_arrangements : total_plants = 4 ∧ total_lamps = 4 →
  ∃ n : ℕ, n = 28 :=
by
  intro h
  sorry

end number_of_arrangements_l17_17721


namespace factorial_div_l17_17551

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_div : (factorial 4) / (factorial (4 - 3)) = 24 := by
  sorry

end factorial_div_l17_17551


namespace overtime_hours_proof_l17_17831

-- Define the conditions
variable (regular_pay_rate : ℕ := 3)
variable (regular_hours : ℕ := 40)
variable (overtime_multiplier : ℕ := 2)
variable (total_pay : ℕ := 180)

-- Calculate the regular pay for 40 hours
def regular_pay : ℕ := regular_pay_rate * regular_hours

-- Calculate the extra pay received beyond regular pay
def extra_pay : ℕ := total_pay - regular_pay

-- Calculate overtime pay rate
def overtime_pay_rate : ℕ := overtime_multiplier * regular_pay_rate

-- Calculate the number of overtime hours
def overtime_hours (extra_pay : ℕ) (overtime_pay_rate : ℕ) : ℕ :=
  extra_pay / overtime_pay_rate

-- The theorem to prove
theorem overtime_hours_proof :
  overtime_hours extra_pay overtime_pay_rate = 10 := by
  sorry

end overtime_hours_proof_l17_17831


namespace juan_stamp_cost_l17_17868

-- Defining the prices of the stamps
def price_brazil : ℝ := 0.07
def price_peru : ℝ := 0.05

-- Defining the number of stamps from the 70s and 80s
def stamps_brazil_70s : ℕ := 12
def stamps_brazil_80s : ℕ := 15
def stamps_peru_70s : ℕ := 6
def stamps_peru_80s : ℕ := 12

-- Calculating total number of stamps from the 70s and 80s
def total_stamps_brazil : ℕ := stamps_brazil_70s + stamps_brazil_80s
def total_stamps_peru : ℕ := stamps_peru_70s + stamps_peru_80s

-- Calculating total cost
def total_cost_brazil : ℝ := total_stamps_brazil * price_brazil
def total_cost_peru : ℝ := total_stamps_peru * price_peru

def total_cost : ℝ := total_cost_brazil + total_cost_peru

-- Proof statement
theorem juan_stamp_cost : total_cost = 2.79 :=
by
  sorry

end juan_stamp_cost_l17_17868


namespace product_of_roots_l17_17592

noncomputable def quadratic_equation (x : ℝ) : Prop :=
  (x + 4) * (x - 5) = 22

theorem product_of_roots :
  ∀ x1 x2 : ℝ, quadratic_equation x1 → quadratic_equation x2 → (x1 * x2 = -42) := 
by
  sorry

end product_of_roots_l17_17592


namespace regular_tickets_sold_l17_17199

variables (S R : ℕ) (h1 : S + R = 65) (h2 : 10 * S + 15 * R = 855)

theorem regular_tickets_sold : R = 41 :=
sorry

end regular_tickets_sold_l17_17199


namespace odd_periodic_function_l17_17765

theorem odd_periodic_function (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_period : ∀ x : ℝ, f (x + 5) = f x)
  (h_f1 : f 1 = 1)
  (h_f2 : f 2 = 2) :
  f 3 - f 4 = -1 :=
sorry

end odd_periodic_function_l17_17765


namespace dual_expr_result_solve_sqrt_eq_16_solve_sqrt_rational_eq_4x_l17_17184

-- Question 1
theorem dual_expr_result (m n : ℝ) (h1 : m = 2 - Real.sqrt 3) (h2 : n = 2 + Real.sqrt 3) :
  m * n = 1 :=
sorry

-- Question 2
theorem solve_sqrt_eq_16 (x : ℝ) (h : Real.sqrt (x + 42) + Real.sqrt (x + 10) = 16) :
  x = 39 :=
sorry

-- Question 3
theorem solve_sqrt_rational_eq_4x (x : ℝ) (h : Real.sqrt (4 * x^2 + 6 * x - 5) + Real.sqrt (4 * x^2 - 2 * x - 5) = 4 * x) :
  x = 3 :=
sorry

end dual_expr_result_solve_sqrt_eq_16_solve_sqrt_rational_eq_4x_l17_17184


namespace simplify_and_substitute_l17_17367

theorem simplify_and_substitute (x : ℝ) (h1 : x ≠ 1) (h3 : x ≠ 3) : 
  ((1 - (2 / (x - 1))) * ((x^2 - x) / (x^2 - 6*x + 9))) = (x / (x - 3)) ∧ 
  (2 / (2 - 3)) = -2 := by
  sorry

end simplify_and_substitute_l17_17367


namespace find_y_l17_17495

theorem find_y (y : ℕ) : (8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10) = 2 ^ y → y = 33 := 
by 
  sorry

end find_y_l17_17495


namespace pipe_ratio_l17_17245

theorem pipe_ratio (A B : ℝ) (hA : A = 1 / 12) (hAB : A + B = 1 / 3) : B / A = 3 := by
  sorry

end pipe_ratio_l17_17245


namespace john_probability_l17_17860

/-- John arrives at a terminal which has sixteen gates arranged in a straight line with exactly 50 feet between adjacent gates. His departure gate is assigned randomly. After waiting at that gate, John is informed that the departure gate has been changed to another gate, chosen randomly again. Prove that the probability that John walks 200 feet or less to the new gate is \(\frac{4}{15}\), and find \(4 + 15 = 19\) -/
theorem john_probability :
  let n_gates := 16
  let dist_between_gates := 50
  let max_walk_dist := 200
  let total_possibilities := n_gates * (n_gates - 1)
  let valid_cases :=
    4 * (2 + 2 * (4 - 1))
  let probability_within_200_feet := valid_cases / total_possibilities
  let fraction := probability_within_200_feet * (15 / 4)
  fraction = 1 → 4 + 15 = 19 := by
  sorry -- Proof goes here 

end john_probability_l17_17860


namespace perpendicular_plane_line_sum_l17_17732

theorem perpendicular_plane_line_sum (x y : ℝ)
  (h1 : ∃ k : ℝ, (2, -4 * x, 1) = (6 * k, 12 * k, -3 * k * y))
  : x + y = -2 :=
sorry

end perpendicular_plane_line_sum_l17_17732


namespace minimum_value_expression_l17_17639

theorem minimum_value_expression (α β : ℝ) : (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 18)^2 ≥ 144 :=
by
  sorry

end minimum_value_expression_l17_17639


namespace price_first_oil_l17_17777

theorem price_first_oil (P : ℝ) (h1 : 10 * P + 5 * 66 = 15 * 58.67) : P = 55.005 :=
sorry

end price_first_oil_l17_17777


namespace sequence_problem_l17_17633

theorem sequence_problem 
  (a : ℕ → ℕ) 
  (h1 : a 1 = 5) 
  (h2 : ∀ n : ℕ, a (n + 1) - a n = 3 + 4 * (n - 1)) : 
  a 50 = 4856 :=
sorry

end sequence_problem_l17_17633


namespace gcd_of_324_and_135_l17_17400

theorem gcd_of_324_and_135 : Nat.gcd 324 135 = 27 :=
by
  sorry

end gcd_of_324_and_135_l17_17400


namespace average_expenditure_week_l17_17387

theorem average_expenditure_week (avg_3_days: ℝ) (avg_4_days: ℝ) (total_days: ℝ):
  avg_3_days = 350 → avg_4_days = 420 → total_days = 7 → 
  ((3 * avg_3_days + 4 * avg_4_days) / total_days = 390) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end average_expenditure_week_l17_17387


namespace total_people_ball_l17_17153

theorem total_people_ball (n m : ℕ) (h1 : n + m < 50) (h2 : 3 * n = 20 * m) : n + m = 41 := 
sorry

end total_people_ball_l17_17153


namespace cube_expansion_l17_17821

variable {a b : ℝ}

theorem cube_expansion (a b : ℝ) : (-a * b^2)^3 = -a^3 * b^6 :=
  sorry

end cube_expansion_l17_17821


namespace matrix_inverse_l17_17534

-- Define the given matrix
def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![5, 4], ![-2, 8]]

-- Define the expected inverse matrix
def A_inv_expected : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![1/6, -1/12], ![1/24, 5/48]]

-- The main statement: Prove that the inverse of A is equal to the expected inverse
theorem matrix_inverse :
  A⁻¹ = A_inv_expected := sorry

end matrix_inverse_l17_17534


namespace correct_statements_l17_17711

theorem correct_statements (f : ℝ → ℝ)
  (h_add : ∀ x y : ℝ, f (x + y) = f (x) + f (y))
  (h_pos : ∀ x : ℝ, x > 0 → f (x) > 0) :
  (f 0 ≠ 1) ∧
  (∀ x : ℝ, f (-x) = -f (x)) ∧
  ¬ (∀ x : ℝ, |f (x)| = |f (-x)|) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f (x₁) < f (x₂)) ∧
  ¬ (∀ x : ℝ, f (x) + 1 < f (x + 1)) :=
by
  sorry

end correct_statements_l17_17711


namespace max_value_of_y_over_x_l17_17896

theorem max_value_of_y_over_x
  (x y : ℝ)
  (h1 : x + y ≥ 3)
  (h2 : x - y ≥ -1)
  (h3 : 2 * x - y ≤ 3) :
  (∀ (x y : ℝ), (x + y ≥ 3) ∧ (x - y ≥ -1) ∧ (2 * x - y ≤ 3) → (∀ k, k = y / x → k ≤ 2)) :=
by
  sorry

end max_value_of_y_over_x_l17_17896


namespace half_sum_of_squares_l17_17063

theorem half_sum_of_squares (n m : ℕ) (h : n ≠ m) :
  ∃ a b : ℕ, ( (2 * n)^2 + (2 * m)^2) / 2 = a^2 + b^2 := by
  sorry

end half_sum_of_squares_l17_17063


namespace range_M_l17_17118

theorem range_M (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b < 1) :
  1 < (1 / (1 + a)) + (1 / (1 + b)) ∧ (1 / (1 + a)) + (1 / (1 + b)) < 2 := by
  sorry

end range_M_l17_17118


namespace red_marbles_count_l17_17538

variable (n : ℕ)

-- Conditions
def ratio_green_yellow_red := (3 * n, 4 * n, 2 * n)
def not_red_marbles := 3 * n + 4 * n = 63

-- Goal
theorem red_marbles_count (hn : not_red_marbles n) : 2 * n = 18 :=
by
  sorry

end red_marbles_count_l17_17538


namespace cos_beta_value_l17_17934

theorem cos_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (hα_cos : Real.cos α = 4 / 5) (hαβ_cos : Real.cos (α + β) = -16 / 65) : 
  Real.cos β = 5 / 13 := 
sorry

end cos_beta_value_l17_17934


namespace compute_f_1_g_3_l17_17121

def f (x : ℝ) : ℝ := 2 * x - 5
def g (x : ℝ) : ℝ := x + 2

theorem compute_f_1_g_3 : f (1 + g 3) = 7 := 
by
  -- Proof goes here
  sorry

end compute_f_1_g_3_l17_17121


namespace apples_in_third_basket_l17_17882

theorem apples_in_third_basket (total_apples : ℕ) (x : ℕ) (y : ℕ) 
    (h_total : total_apples = 2014)
    (h_second_basket : 49 + x = total_apples - 2 * y - x - y)
    (h_first_basket : total_apples - 2 * y - x + y = 2 * y)
    : x + y = 655 :=
by
    sorry

end apples_in_third_basket_l17_17882


namespace joe_saves_6000_l17_17993

-- Definitions based on the conditions
def flight_cost : ℕ := 1200
def hotel_cost : ℕ := 800
def food_cost : ℕ := 3000
def money_left : ℕ := 1000

-- Total expenses
def total_expenses : ℕ := flight_cost + hotel_cost + food_cost

-- Total savings
def total_savings : ℕ := total_expenses + money_left

-- The proof statement
theorem joe_saves_6000 : total_savings = 6000 := by
  -- Proof goes here
  sorry

end joe_saves_6000_l17_17993


namespace circle_radius_6_l17_17443

theorem circle_radius_6 (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 10*x + y^2 + 6*y - k = 0 ↔ (x + 5)^2 + (y + 3)^2 = 36) → k = 2 :=
by
  sorry

end circle_radius_6_l17_17443


namespace complete_contingency_table_chi_square_test_certainty_l17_17707

-- Defining the initial conditions given in the problem
def total_students : ℕ := 100
def boys_dislike : ℕ := 10
def girls_like : ℕ := 20
def dislike_probability : ℚ := 0.4

-- Completed contingency table values based on given and inferred values
def boys_total : ℕ := 50
def girls_total : ℕ := 50
def boys_like : ℕ := boys_total - boys_dislike
def girls_dislike : ℕ := 30
def total_like : ℕ := boys_like + girls_like
def total_dislike : ℕ := boys_dislike + girls_dislike

-- Chi-square value from the solution
def K_squared : ℚ := 50 / 3

-- Declaring the proof problem for the completed contingency table
theorem complete_contingency_table :
  boys_total + girls_total = total_students ∧ 
  total_like + total_dislike = total_students ∧ 
  dislike_probability * total_students = total_dislike ∧ 
  boys_like = 40 ∧ 
  girls_dislike = 30 :=
sorry

-- Declaring the proof problem for the chi-square test
theorem chi_square_test_certainty :
  K_squared > 10.828 :=
sorry

end complete_contingency_table_chi_square_test_certainty_l17_17707


namespace average_of_shifted_sample_l17_17616

theorem average_of_shifted_sample (x1 x2 x3 : ℝ) (hx_avg : (x1 + x2 + x3) / 3 = 40) (hx_var : ((x1 - 40) ^ 2 + (x2 - 40) ^ 2 + (x3 - 40) ^ 2) / 3 = 1) : 
  ((x1 + 40) + (x2 + 40) + (x3 + 40)) / 3 = 80 :=
sorry

end average_of_shifted_sample_l17_17616


namespace mall_b_better_for_fewer_than_6_mall_equal_for_6_mall_a_better_for_more_than_6_l17_17524

-- Definitions
def original_price : ℕ := 80
def discount_mallA (n : ℕ) : ℕ := min ((4 * n) * n) (80 * n / 2)
def discount_mallB (n : ℕ) : ℕ := (80 * n * 3) / 10

def total_cost_mallA (n : ℕ) : ℕ := (original_price * n) - discount_mallA n
def total_cost_mallB (n : ℕ) : ℕ := (original_price * n) - discount_mallB n

-- Theorem statements
theorem mall_b_better_for_fewer_than_6 (n : ℕ) (h : n < 6) : total_cost_mallA n > total_cost_mallB n := sorry
theorem mall_equal_for_6 (n : ℕ) (h : n = 6) : total_cost_mallA n = total_cost_mallB n := sorry
theorem mall_a_better_for_more_than_6 (n : ℕ) (h : n > 6) : total_cost_mallA n < total_cost_mallB n := sorry

end mall_b_better_for_fewer_than_6_mall_equal_for_6_mall_a_better_for_more_than_6_l17_17524


namespace range_x_l17_17003

variable {R : Type*} [LinearOrderedField R]

def monotone_increasing_on (f : R → R) (s : Set R) := ∀ ⦃a b⦄, a ≤ b → f a ≤ f b

theorem range_x 
    (f : R → R) 
    (h_mono : monotone_increasing_on f Set.univ) 
    (h_zero : f 1 = 0) 
    (h_ineq : ∀ x, f (x^2 + 3 * x - 3) < 0) :
  ∀ x, -4 < x ∧ x < 1 :=
by 
  sorry

end range_x_l17_17003


namespace joan_missed_games_l17_17921

theorem joan_missed_games :
  ∀ (total_games attended_games missed_games : ℕ),
  total_games = 864 →
  attended_games = 395 →
  missed_games = total_games - attended_games →
  missed_games = 469 :=
by
  intros total_games attended_games missed_games H1 H2 H3
  rw [H1, H2] at H3
  exact H3

end joan_missed_games_l17_17921


namespace distance_ran_by_Juan_l17_17812

-- Definitions based on the condition
def speed : ℝ := 10 -- in miles per hour
def time : ℝ := 8 -- in hours

-- Theorem statement
theorem distance_ran_by_Juan : speed * time = 80 := by
  sorry

end distance_ran_by_Juan_l17_17812


namespace geometric_seq_inequality_l17_17908

theorem geometric_seq_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : b^2 = a * c) : a^2 + b^2 + c^2 > (a - b + c)^2 :=
by
  sorry

end geometric_seq_inequality_l17_17908


namespace mouse_lives_difference_l17_17057

-- Definitions of variables and conditions
def cat_lives : ℕ := 9
def dog_lives : ℕ := cat_lives - 3
def mouse_lives : ℕ := 13

-- Theorem to prove
theorem mouse_lives_difference : mouse_lives - dog_lives = 7 := by
  -- This is where the proof would go, but we use sorry to skip it.
  sorry

end mouse_lives_difference_l17_17057


namespace problem_l17_17021

theorem problem (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 5) 
: (5 * m * r - 2 * n * t) / (7 * n * t - 10 * m * r) = -31 / 56 := 
sorry

end problem_l17_17021


namespace no_form3000001_is_perfect_square_l17_17559

theorem no_form3000001_is_perfect_square (n : ℕ) : 
  ∀ k : ℤ, (3 * 10^n + 1 ≠ k^2) :=
by
  sorry

end no_form3000001_is_perfect_square_l17_17559


namespace muffin_banana_cost_ratio_l17_17481

variables (m b c : ℕ) -- costs of muffin, banana, and cookie respectively
variables (susie_cost calvin_cost : ℕ)

-- Conditions
def susie_cost_eq : Prop := susie_cost = 5 * m + 4 * b + 2 * c
def calvin_cost_eq : Prop := calvin_cost = 3 * (5 * m + 4 * b + 2 * c)
def calvin_cost_eq_reduced : Prop := calvin_cost = 3 * m + 20 * b + 6 * c
def cookie_cost_eq : Prop := c = 2 * b

-- Question and Answer
theorem muffin_banana_cost_ratio
  (h1 : susie_cost_eq m b c susie_cost)
  (h2 : calvin_cost_eq m b c calvin_cost)
  (h3 : calvin_cost_eq_reduced m b c calvin_cost)
  (h4 : cookie_cost_eq b c)
  : m = 4 * b / 3 :=
sorry

end muffin_banana_cost_ratio_l17_17481


namespace min_negative_numbers_l17_17048

theorem min_negative_numbers (a b c d : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : a + b + c < d) (h6 : a + b + d < c) (h7 : a + c + d < b) (h8 : b + c + d < a) :
  3 ≤ (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) + (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0) := 
sorry

end min_negative_numbers_l17_17048


namespace additional_rate_of_interest_l17_17776

variable (P A A' : ℝ) (T : ℕ) (R : ℝ)

-- Conditions
def principal_amount := (P = 8000)
def original_amount := (A = 9200)
def time_period := (T = 3)
def new_amount := (A' = 9440)

-- The Lean statement to prove the additional percentage of interest
theorem additional_rate_of_interest  (P A A' : ℝ) (T : ℕ) (R : ℝ)
    (h1 : principal_amount P)
    (h2 : original_amount A)
    (h3 : time_period T)
    (h4 : new_amount A') :
    (A' - P) / (P * T) * 100 - (A - P) / (P * T) * 100 = 1 :=
by
  sorry

end additional_rate_of_interest_l17_17776


namespace salary_january_l17_17330

theorem salary_january
  (J F M A May : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8600)
  (h3 : May = 6500) :
  J = 4100 :=
by 
  sorry

end salary_january_l17_17330


namespace minimum_value_of_g_gm_equal_10_implies_m_is_5_l17_17900

/-- Condition: Definition of the function y in terms of x and m -/
def y (x m : ℝ) : ℝ := x^2 + m * x - 4

/-- Theorem about finding the minimum value of g(m) -/
theorem minimum_value_of_g (m : ℝ) :
  ∃ g : ℝ, g = (if m ≥ -4 then 2 * m
      else if -8 < m ∧ m < -4 then -m^2 / 4 - 4
      else 4 * m + 12) := by
  sorry

/-- Theorem that if the minimum value of g(m) is 10, then m must be 5 -/
theorem gm_equal_10_implies_m_is_5 :
  ∃ m, (if m ≥ -4 then 2 * m
       else if -8 < m ∧ m < -4 then -m^2 / 4 - 4
       else 4 * m + 12) = 10 := by
  use 5
  sorry

end minimum_value_of_g_gm_equal_10_implies_m_is_5_l17_17900


namespace equation_of_parabola_l17_17653

def parabola_passes_through_point (a h : ℝ) : Prop :=
  2 = a * (8^2) + h

def focus_x_coordinate (a h : ℝ) : Prop :=
  h + (1 / (4 * a)) = 3

theorem equation_of_parabola :
  ∃ (a h : ℝ), parabola_passes_through_point a h ∧ focus_x_coordinate a h ∧
    (∀ x y : ℝ, x = (15 / 256) * y^2 - (381 / 128)) :=
sorry

end equation_of_parabola_l17_17653


namespace ratio_population_A_to_F_l17_17497

variable (F : ℕ)

def population_E := 6 * F
def population_D := 2 * population_E
def population_C := 8 * population_D
def population_B := 3 * population_C
def population_A := 5 * population_B

theorem ratio_population_A_to_F (F_pos : F > 0) :
  population_A F / F = 1440 := by
sorry

end ratio_population_A_to_F_l17_17497


namespace smallest_common_multiple_of_9_and_6_l17_17715

theorem smallest_common_multiple_of_9_and_6 : ∃ (n : ℕ), n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m := 
sorry

end smallest_common_multiple_of_9_and_6_l17_17715


namespace shaded_fraction_in_fourth_square_l17_17526

theorem shaded_fraction_in_fourth_square : 
  ∀ (f : ℕ → ℕ), (f 1 = 1)
  ∧ (f 2 = 3)
  ∧ (f 3 = 5)
  ∧ (f 4 = f 3 + (3 - 1) + (5 - 3))
  ∧ (f 4 * 2 = 14)
  → (f 4 = 7)
  → (f 4 / 16 = 7 / 16) :=
sorry

end shaded_fraction_in_fourth_square_l17_17526


namespace fraction_evaluation_l17_17550

theorem fraction_evaluation :
  let p := 8579
  let q := 6960
  p.gcd q = 1 ∧ (32 / 30 - 30 / 32 + 32 / 29) = p / q :=
by
  sorry

end fraction_evaluation_l17_17550


namespace problem_correct_answer_l17_17143

theorem problem_correct_answer :
  (∀ (P L : Type) (passes_through_point : P → L → Prop) (parallel_to : L → L → Prop),
    (∀ (l₁ l₂ : L) (p : P), passes_through_point p l₁ ∧ ¬ passes_through_point p l₂ → (∃! l : L, passes_through_point p l ∧ parallel_to l l₂)) ->
  (∃ (l₁ l₂ : L) (A : P), passes_through_point A l₁ ∧ ¬ passes_through_point A l₂ ∧ ∃ l : L, passes_through_point A l ∧ parallel_to l l₂) ) :=
sorry

end problem_correct_answer_l17_17143


namespace darius_age_is_8_l17_17018

def age_of_darius (jenna_age darius_age : ℕ) : Prop :=
  jenna_age = darius_age + 5

theorem darius_age_is_8 (jenna_age darius_age : ℕ) (h1 : jenna_age = darius_age + 5) (h2: jenna_age = 13) : 
  darius_age = 8 :=
by
  sorry

end darius_age_is_8_l17_17018


namespace prove_ab_ge_5_l17_17259

theorem prove_ab_ge_5 (a b c : ℕ) (h : ∀ x, x * (a * x) = b * x + c → 0 ≤ x ∧ x ≤ 1) : 5 ≤ a ∧ 5 ≤ b := 
sorry

end prove_ab_ge_5_l17_17259


namespace find_a_l17_17982

theorem find_a (a : ℝ) (h1 : ∀ (x y : ℝ), ax + 2*y - 2 = 0 → (x + y) = 0)
  (h2 : ∀ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 6 → (∃ A B : ℝ × ℝ, A ≠ B ∧ (A = (x, y) ∧ B = (-x, -y))))
  : a = -2 := 
sorry

end find_a_l17_17982


namespace train_length_is_100_meters_l17_17471

-- Definitions of conditions
def speed_kmh := 40  -- speed in km/hr
def time_s := 9  -- time in seconds

-- Conversion factors
def km_to_m := 1000  -- 1 km = 1000 meters
def hr_to_s := 3600  -- 1 hour = 3600 seconds

-- Converting speed from km/hr to m/s
def speed_ms := (speed_kmh * km_to_m) / hr_to_s

-- The proof that the length of the train is 100 meters
theorem train_length_is_100_meters :
  (speed_ms * time_s) = 100 :=
by
  sorry

-- The Lean statement merely sets up the problem as asked.

end train_length_is_100_meters_l17_17471


namespace square_must_rotate_at_least_5_turns_l17_17333

-- Define the square and pentagon as having equal side lengths
def square_sides : Nat := 4
def pentagon_sides : Nat := 5

-- The problem requires us to prove that the square needs to rotate at least 5 full turns
theorem square_must_rotate_at_least_5_turns :
  let lcm := Nat.lcm square_sides pentagon_sides
  lcm / square_sides = 5 :=
by
  -- Proof to be provided
  sorry

end square_must_rotate_at_least_5_turns_l17_17333


namespace y_intercept_of_line_l17_17807

/-- Let m be the slope of a line and (x_intercept, 0) be the x-intercept of the same line.
    If the line passes through the point (3, 0) and has a slope of -3, then its y-intercept is (0, 9). -/
theorem y_intercept_of_line 
    (m : ℝ) (x_intercept : ℝ) (x1 y1 : ℝ)
    (h1 : m = -3)
    (h2 : (x_intercept, 0) = (3, 0)) :
    (0, -m * x_intercept) = (0, 9) :=
by sorry

end y_intercept_of_line_l17_17807


namespace choose_5_starters_including_twins_l17_17441

def number_of_ways_choose_starters (total_players : ℕ) (members_in_lineup : ℕ) (twins1 twins2 : (ℕ × ℕ)) : ℕ :=
1834

theorem choose_5_starters_including_twins :
  number_of_ways_choose_starters 18 5 (1, 2) (3, 4) = 1834 :=
sorry

end choose_5_starters_including_twins_l17_17441


namespace quadratic_inequality_solution_l17_17839

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * a * x + a > 0) ↔ (0 < a ∧ a < 1) :=
by
  sorry

end quadratic_inequality_solution_l17_17839


namespace fraction_not_going_l17_17354

theorem fraction_not_going (S J : ℕ) (h1 : J = (2:ℕ)/3 * S) 
  (h_not_junior : 3/4 * J = 3/4 * (2/3 * S)) 
  (h_not_senior : 1/3 * S = (1:ℕ)/3 * S) :
  3/4 * (2/3 * S) + 1/3 * S = 5/6 * S :=
by 
  sorry

end fraction_not_going_l17_17354


namespace watermelons_left_l17_17764

theorem watermelons_left (initial : ℕ) (eaten : ℕ) (remaining : ℕ) (h1 : initial = 4) (h2 : eaten = 3) : remaining = 1 :=
by
  sorry

end watermelons_left_l17_17764


namespace exists_base_and_digit_l17_17172

def valid_digit_in_base (B : ℕ) (V : ℕ) : Prop :=
  V^2 % B = V ∧ V ≠ 0 ∧ V ≠ 1

theorem exists_base_and_digit :
  ∃ B V, valid_digit_in_base B V :=
by {
  sorry
}

end exists_base_and_digit_l17_17172


namespace arithmetic_sequence_a6_l17_17418

theorem arithmetic_sequence_a6 (a : ℕ → ℝ)
  (h4_8 : ∃ a4 a8, (a 4 = a4) ∧ (a 8 = a8) ∧ a4^2 - 6*a4 + 5 = 0 ∧ a8^2 - 6*a8 + 5 = 0) :
  a 6 = 3 := by 
  sorry

end arithmetic_sequence_a6_l17_17418


namespace part_a_avg_area_difference_part_b_prob_same_area_part_c_expected_value_difference_l17_17267

-- Part (a)
theorem part_a_avg_area_difference : 
  let zahid_avg := (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) / 6
  let yana_avg := (21 / 6)^2
  zahid_avg - yana_avg = 35 / 12 := sorry

-- Part (b)
theorem part_b_prob_same_area :
  let prob_zahid_min n := (13 - 2 * n) / 36
  let prob_same_area := (1 / 36) * ((11 / 36) + (9 / 36) + (7 / 36) + (5 / 36) + (3 / 36) + (1 / 36))
  prob_same_area = 1 / 24 := sorry

-- Part (c)
theorem part_c_expected_value_difference :
  let yana_avg := 49 / 4
  let zahid_avg := (11 / 36 * 1^2 + 9 / 36 * 2^2 + 7 / 36 * 3^2 + 5 / 36 * 4^2 + 3 / 36 * 5^2 + 1 / 36 * 6^2)
  (yana_avg - zahid_avg) = 35 / 9 := sorry

end part_a_avg_area_difference_part_b_prob_same_area_part_c_expected_value_difference_l17_17267


namespace minimal_abs_diff_l17_17647

theorem minimal_abs_diff (a b : ℤ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : a * b - 3 * a + 7 * b = 222) : |a - b| = 54 :=
by
  sorry

end minimal_abs_diff_l17_17647


namespace fractional_exponent_representation_of_sqrt_l17_17177

theorem fractional_exponent_representation_of_sqrt (a : ℝ) : 
  Real.sqrt (a * 3 * a * Real.sqrt a) = a ^ (3 / 4) := 
sorry

end fractional_exponent_representation_of_sqrt_l17_17177


namespace sum_of_consecutive_even_integers_l17_17316

theorem sum_of_consecutive_even_integers (n : ℕ) (h1 : (n - 2) + (n + 2) = 162) (h2 : ∃ k : ℕ, n = k^2) :
  (n - 2) + n + (n + 2) = 243 :=
by
  -- no proof required
  sorry

end sum_of_consecutive_even_integers_l17_17316


namespace scott_monthly_miles_l17_17031

theorem scott_monthly_miles :
  let miles_per_mon_wed := 3
  let mon_wed_days := 3
  let thur_fri_factor := 2
  let thur_fri_days := 2
  let weeks_per_month := 4
  let miles_mon_wed := miles_per_mon_wed * mon_wed_days
  let miles_thur_fri_per_day := thur_fri_factor * miles_per_mon_wed
  let miles_thur_fri := miles_thur_fri_per_day * thur_fri_days
  let miles_per_week := miles_mon_wed + miles_thur_fri
  let total_miles_in_month := miles_per_week * weeks_per_month
  total_miles_in_month = 84 := 
  by
    sorry

end scott_monthly_miles_l17_17031


namespace range_of_a_range_of_f_diff_l17_17397

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + x + 1
noncomputable def f' (a x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 1

theorem range_of_a (a : ℝ) : (∃ x1 x2 : ℝ, f' a x1 = 0 ∧ f' a x2 = 0 ∧ x1 ≠ x2) ↔ (a < -Real.sqrt 3 ∨ a > Real.sqrt 3) :=
by
  sorry

theorem range_of_f_diff (a x1 x2 : ℝ) (h1 : f' a x1 = 0) (h2 : f' a x2 = 0) (h12 : x1 ≠ x2) : 
  0 < f a x1 - f a x2 :=
by
  sorry

end range_of_a_range_of_f_diff_l17_17397


namespace no_integer_solutions_l17_17928

theorem no_integer_solutions (x y : ℤ) :
  ¬ (x^2 + 3 * x * y - 2 * y^2 = 122) :=
sorry

end no_integer_solutions_l17_17928


namespace cos_x_is_necessary_but_not_sufficient_for_sin_x_zero_l17_17432

-- Defining the conditions
def cos_x_eq_one (x : ℝ) : Prop := Real.cos x = 1
def sin_x_eq_zero (x : ℝ) : Prop := Real.sin x = 0

-- Main theorem statement
theorem cos_x_is_necessary_but_not_sufficient_for_sin_x_zero (x : ℝ) : 
  (∀ x, cos_x_eq_one x → sin_x_eq_zero x) ∧ (∃ x, sin_x_eq_zero x ∧ ¬ cos_x_eq_one x) :=
by 
  sorry

end cos_x_is_necessary_but_not_sufficient_for_sin_x_zero_l17_17432


namespace hyperbola_focus_y_axis_l17_17851

theorem hyperbola_focus_y_axis (m : ℝ) :
  (∀ x y : ℝ, (m + 1) * x^2 + (2 - m) * y^2 = 1) → m < -1 :=
sorry

end hyperbola_focus_y_axis_l17_17851


namespace goose_eggs_count_l17_17029

theorem goose_eggs_count (E : ℕ) (h1 : E % 3 = 0) 
(h2 : ((4 / 5) * (1 / 3) * E) * (2 / 5) = 120) : E = 1125 := 
sorry

end goose_eggs_count_l17_17029


namespace max_gcd_13n_plus_4_8n_plus_3_l17_17621

theorem max_gcd_13n_plus_4_8n_plus_3 (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, k = 9 ∧ gcd (13 * n + 4) (8 * n + 3) = k := 
sorry

end max_gcd_13n_plus_4_8n_plus_3_l17_17621


namespace find_unknown_number_l17_17829

theorem find_unknown_number (x n : ℚ) (h1 : n + 7/x = 6 - 5/x) (h2 : x = 12) : n = 5 :=
by
  sorry

end find_unknown_number_l17_17829


namespace puppy_sleep_duration_l17_17054

-- Definitions based on conditions
def connor_sleep : ℕ := 6
def luke_sleep : ℕ := connor_sleep + 2
def puppy_sleep : ℕ := 2 * luke_sleep

-- Theorem stating that the puppy sleeps for 16 hours
theorem puppy_sleep_duration : puppy_sleep = 16 := by
  sorry

end puppy_sleep_duration_l17_17054


namespace tetrad_does_not_have_four_chromosomes_l17_17039

noncomputable def tetrad_has_two_centromeres : Prop := -- The condition: a tetrad has two centromeres
  sorry

noncomputable def tetrad_contains_four_dna_molecules : Prop := -- The condition: a tetrad contains four DNA molecules
  sorry

noncomputable def tetrad_consists_of_two_pairs_of_sister_chromatids : Prop := -- The condition: a tetrad consists of two pairs of sister chromatids
  sorry

theorem tetrad_does_not_have_four_chromosomes 
  (h1: tetrad_has_two_centromeres)
  (h2: tetrad_contains_four_dna_molecules)
  (h3: tetrad_consists_of_two_pairs_of_sister_chromatids) 
  : ¬ (tetrad_has_four_chromosomes : Prop) :=
sorry

end tetrad_does_not_have_four_chromosomes_l17_17039


namespace no_real_x_condition_l17_17952

theorem no_real_x_condition (x : ℝ) : 
(∃ a b : ℕ, 4 * x^5 - 7 = a^2 ∧ 4 * x^13 - 7 = b^2) → false := 
by {
  sorry
}

end no_real_x_condition_l17_17952


namespace anglet_angle_measurement_l17_17401

-- Definitions based on conditions
def anglet_measurement := 1
def sixth_circle_degrees := 360 / 6
def anglets_in_sixth_circle := 6000

-- Lean theorem statement proving the implied angle measurement
theorem anglet_angle_measurement (one_percent : Real := 0.01) :
  (anglets_in_sixth_circle * one_percent * sixth_circle_degrees) = anglet_measurement * 60 := 
  sorry

end anglet_angle_measurement_l17_17401


namespace total_oil_leaked_correct_l17_17994

-- Definitions of given conditions.
def initial_leak_A : ℕ := 6522
def leak_rate_A : ℕ := 257
def time_A : ℕ := 20

def initial_leak_B : ℕ := 3894
def leak_rate_B : ℕ := 182
def time_B : ℕ := 15

def initial_leak_C : ℕ := 1421
def leak_rate_C : ℕ := 97
def time_C : ℕ := 12

-- Total additional leaks calculation.
def additional_leak (rate time : ℕ) : ℕ := rate * time
def additional_leak_A : ℕ := additional_leak leak_rate_A time_A
def additional_leak_B : ℕ := additional_leak leak_rate_B time_B
def additional_leak_C : ℕ := additional_leak leak_rate_C time_C

-- Total leaks from each pipe.
def total_leak_A : ℕ := initial_leak_A + additional_leak_A
def total_leak_B : ℕ := initial_leak_B + additional_leak_B
def total_leak_C : ℕ := initial_leak_C + additional_leak_C

-- Total oil leaked.
def total_oil_leaked : ℕ := total_leak_A + total_leak_B + total_leak_C

-- The proof problem statement.
theorem total_oil_leaked_correct : total_oil_leaked = 20871 := by
  sorry

end total_oil_leaked_correct_l17_17994


namespace area_of_circle_l17_17625

/-- Given a circle with circumference 36π, prove that the area is 324π. -/
theorem area_of_circle (C : ℝ) (hC : C = 36 * π) 
  (h1 : ∀ r : ℝ, C = 2 * π * r → 0 ≤ r)
  (h2 : ∀ r : ℝ, 0 ≤ r → ∃ (A : ℝ), A = π * r^2) :
  ∃ k : ℝ, (A = 324 * π → k = 324) := 
sorry


end area_of_circle_l17_17625


namespace square_of_binomial_l17_17660

theorem square_of_binomial {a r s : ℚ} 
  (h1 : r^2 = a)
  (h2 : 2 * r * s = 18)
  (h3 : s^2 = 16) : 
  a = 81 / 16 :=
by sorry

end square_of_binomial_l17_17660


namespace car_rental_budget_l17_17598

def daily_rental_cost : ℝ := 30.0
def cost_per_mile : ℝ := 0.18
def total_miles : ℝ := 250.0

theorem car_rental_budget : daily_rental_cost + (cost_per_mile * total_miles) = 75.0 :=
by 
  sorry

end car_rental_budget_l17_17598


namespace percentage_passed_in_all_three_subjects_l17_17820

-- Define the given failed percentages as real numbers
def A : ℝ := 0.25  -- 25%
def B : ℝ := 0.48  -- 48%
def C : ℝ := 0.35  -- 35%
def AB : ℝ := 0.27 -- 27%
def AC : ℝ := 0.20 -- 20%
def BC : ℝ := 0.15 -- 15%
def ABC : ℝ := 0.10 -- 10%

-- State the theorem to prove the percentage of students who passed in all three subjects
theorem percentage_passed_in_all_three_subjects : 
  1 - (A + B + C - AB - AC - BC + ABC) = 0.44 :=
by
  sorry

end percentage_passed_in_all_three_subjects_l17_17820


namespace probability_A_does_not_lose_l17_17850

theorem probability_A_does_not_lose (p_tie p_A_win : ℚ) (h_tie : p_tie = 1 / 2) (h_A_win : p_A_win = 1 / 3) :
  p_tie + p_A_win = 5 / 6 :=
by sorry

end probability_A_does_not_lose_l17_17850


namespace equal_share_expense_l17_17311

theorem equal_share_expense (L B C X : ℝ) : 
  let T := L + B + C - X
  let share := T / 3 
  L + (share - L) == (B + C - X - 2 * L) / 3 := 
by
  sorry

end equal_share_expense_l17_17311


namespace g_x_squared_minus_3_l17_17590

theorem g_x_squared_minus_3 (g : ℝ → ℝ)
  (h : ∀ x : ℝ, g (x^2 - 1) = x^4 - 4 * x^2 + 4) :
  ∀ x : ℝ, g (x^2 - 3) = x^4 - 6 * x^2 + 11 :=
by
  sorry

end g_x_squared_minus_3_l17_17590


namespace range_of_x_l17_17221

theorem range_of_x (x : ℝ) : (|x + 1| + |x - 1| = 2) → (-1 ≤ x ∧ x ≤ 1) :=
by
  intro h
  sorry

end range_of_x_l17_17221


namespace sheep_ratio_l17_17969

theorem sheep_ratio (s : ℕ) (h1 : s = 400) (h2 : s / 4 + 150 = s - s / 4) : (s / 4 * 3 - 150) / 150 = 1 :=
by {
  sorry
}

end sheep_ratio_l17_17969


namespace find_age_difference_l17_17673

variable (a b c : ℕ)

theorem find_age_difference (h : a + b = b + c + 20) : c = a - 20 :=
by
  sorry

end find_age_difference_l17_17673


namespace randi_peter_ratio_l17_17232

-- Given conditions
def ray_cents := 175
def cents_per_nickel := 5
def peter_cents := 30
def randi_extra_nickels := 6

-- Define the nickels Ray has
def ray_nickels := ray_cents / cents_per_nickel
-- Define the nickels Peter receives
def peter_nickels := peter_cents / cents_per_nickel
-- Define the nickels Randi receives
def randi_nickels := peter_nickels + randi_extra_nickels
-- Define the cents Randi receives
def randi_cents := randi_nickels * cents_per_nickel

-- The goal is to prove the ratio of the cents given to Randi to the cents given to Peter is 2.
theorem randi_peter_ratio : randi_cents / peter_cents = 2 := by
  sorry

end randi_peter_ratio_l17_17232


namespace fraction_product_l17_17537

theorem fraction_product :
  (7 / 4 : ℚ) * (14 / 35) * (21 / 12) * (28 / 56) * (49 / 28) * (42 / 84) * (63 / 36) * (56 / 112) = (1201 / 12800) := 
by
  sorry

end fraction_product_l17_17537


namespace airplane_seat_difference_l17_17208

theorem airplane_seat_difference (F C X : ℕ) 
    (h1 : 387 = F + 310) 
    (h2 : C = 310) 
    (h3 : C = 4 * F + X) :
    X = 2 :=
by
    sorry

end airplane_seat_difference_l17_17208


namespace toys_produced_each_day_l17_17403

-- Define the conditions
def total_weekly_production : ℕ := 8000
def days_worked_per_week : ℕ := 4
def daily_production : ℕ := total_weekly_production / days_worked_per_week

-- The statement to be proved
theorem toys_produced_each_day : daily_production = 2000 := sorry

end toys_produced_each_day_l17_17403


namespace matthew_hotdogs_l17_17461

-- Definitions based on conditions
def hotdogs_ella_emma : ℕ := 2 + 2
def hotdogs_luke : ℕ := 2 * hotdogs_ella_emma
def hotdogs_hunter : ℕ := (3 * hotdogs_ella_emma) / 2  -- Multiplying by 1.5 

-- Theorem statement to prove the total number of hotdogs
theorem matthew_hotdogs : hotdogs_ella_emma + hotdogs_luke + hotdogs_hunter = 18 := by
  sorry

end matthew_hotdogs_l17_17461


namespace area_of_rectangle_is_588_l17_17794

-- Define the conditions
def radius_of_circle := 7
def width_of_rectangle := 2 * radius_of_circle
def length_to_width_ratio := 3

-- Define the width and length of the rectangle based on the conditions
def width := width_of_rectangle
def length := length_to_width_ratio * width_of_rectangle

-- Define the area of the rectangle
def area_of_rectangle := length * width

-- The theorem to prove
theorem area_of_rectangle_is_588 : area_of_rectangle = 588 :=
by sorry -- Proof is not required

end area_of_rectangle_is_588_l17_17794


namespace quadratic_equation_roots_l17_17509

-- Define the two numbers α and β such that their arithmetic and geometric means are given.
variables (α β : ℝ)

-- Arithmetic mean condition
def arithmetic_mean_condition : Prop := (α + β = 16)

-- Geometric mean condition
def geometric_mean_condition : Prop := (α * β = 225)

-- The quadratic equation with roots α and β
def quadratic_equation (x : ℝ) : ℝ := x^2 - 16 * x + 225

-- The proof statement
theorem quadratic_equation_roots (α β : ℝ) (h1 : arithmetic_mean_condition α β) (h2 : geometric_mean_condition α β) :
  ∃ x : ℝ, quadratic_equation x = 0 :=
sorry

end quadratic_equation_roots_l17_17509


namespace semi_circle_radius_l17_17475

theorem semi_circle_radius (π : ℝ) (hπ : Real.pi = π) (P : ℝ) (hP : P = 180) : 
  ∃ r : ℝ, r = 180 / (π + 2) :=
by
  sorry

end semi_circle_radius_l17_17475


namespace cid_earnings_l17_17486

variable (x : ℕ)
variable (oil_change_price repair_price car_wash_price : ℕ)
variable (cars_repaired cars_washed total_earnings : ℕ)

theorem cid_earnings :
  (oil_change_price = 20) →
  (repair_price = 30) →
  (car_wash_price = 5) →
  (cars_repaired = 10) →
  (cars_washed = 15) →
  (total_earnings = 475) →
  (oil_change_price * x + repair_price * cars_repaired + car_wash_price * cars_washed = total_earnings) →
  x = 5 := by sorry

end cid_earnings_l17_17486


namespace tan_60_eq_sqrt3_l17_17695

theorem tan_60_eq_sqrt3 : Real.tan (Real.pi / 3) = Real.sqrt 3 := 
sorry

end tan_60_eq_sqrt3_l17_17695


namespace nephews_count_l17_17230

theorem nephews_count (a_nephews_20_years_ago : ℕ) (third_now_nephews : ℕ) (additional_nephews : ℕ) :
  a_nephews_20_years_ago = 80 →
  third_now_nephews = 3 →
  additional_nephews = 120 →
  ∃ (a_nephews_now : ℕ) (v_nephews_now : ℕ), a_nephews_now = third_now_nephews * a_nephews_20_years_ago ∧ v_nephews_now = a_nephews_now + additional_nephews ∧ (a_nephews_now + v_nephews_now = 600) :=
by
  sorry

end nephews_count_l17_17230


namespace contradictory_statement_of_p_l17_17162

-- Given proposition p
def p : Prop := ∀ (x : ℝ), x + 3 ≥ 0 → x ≥ -3

-- Contradictory statement of p
noncomputable def contradictory_p : Prop := ∀ (x : ℝ), x + 3 < 0 → x < -3

-- Proof statement
theorem contradictory_statement_of_p : contradictory_p :=
sorry

end contradictory_statement_of_p_l17_17162


namespace simplify_expression_l17_17209

theorem simplify_expression :
  (3 * Real.sqrt 8) / 
  (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7 + Real.sqrt 9) =
  - (2 * Real.sqrt 6 - 2 * Real.sqrt 2 + 2 * Real.sqrt 14) / 5 :=
by
  sorry

end simplify_expression_l17_17209


namespace left_handed_rock_lovers_l17_17950

theorem left_handed_rock_lovers (total_people left_handed rock_music right_dislike_rock x : ℕ) :
  total_people = 30 →
  left_handed = 14 →
  rock_music = 20 →
  right_dislike_rock = 5 →
  (x + (left_handed - x) + (rock_music - x) + right_dislike_rock = total_people) →
  x = 9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end left_handed_rock_lovers_l17_17950


namespace extreme_points_l17_17758

noncomputable def f (a x : ℝ) : ℝ := Real.log (1 / (2 * x)) - a * x^2 + x

theorem extreme_points (
  a : ℝ
) (h : 0 < a ∧ a < (1 : ℝ) / 8) :
  ∃ x1 x2 : ℝ, f a x1 + f a x2 > 3 - 4 * Real.log 2 :=
sorry

end extreme_points_l17_17758


namespace true_propositions_l17_17322

theorem true_propositions :
  (∀ x y, (x * y = 1 → x * y = (x * y))) ∧
  (¬ (∀ (a b : ℝ), (∀ (A B : ℝ), a = b → A = B) ∧ (A = B → a ≠ b))) ∧
  (∀ m : ℝ, (m ≤ 1 → ∃ x : ℝ, x^2 - 2 * x + m = 0)) ↔
    (true ∧ true ∧ true) :=
by sorry

end true_propositions_l17_17322


namespace shark_sightings_in_Daytona_Beach_l17_17504

def CM : ℕ := 7

def DB : ℕ := 3 * CM + 5

theorem shark_sightings_in_Daytona_Beach : DB = 26 := by
  sorry

end shark_sightings_in_Daytona_Beach_l17_17504


namespace population_increase_rate_l17_17877

theorem population_increase_rate (persons : ℕ) (minutes : ℕ) (seconds_per_person : ℕ) 
  (h1 : persons = 240) 
  (h2 : minutes = 60) 
  (h3 : seconds_per_person = (minutes * 60) / persons) 
  : seconds_per_person = 15 :=
by 
  sorry

end population_increase_rate_l17_17877


namespace cost_per_trip_l17_17017

theorem cost_per_trip
  (pass_cost : ℝ)
  (oldest_trips : ℕ)
  (youngest_trips : ℕ)
  (h_pass_cost : pass_cost = 100.0)
  (h_oldest_trips : oldest_trips = 35)
  (h_youngest_trips : youngest_trips = 15) :
  (2 * pass_cost) / (oldest_trips + youngest_trips) = 4.0 :=
by
  sorry

end cost_per_trip_l17_17017


namespace parabola_intersection_difference_l17_17381

noncomputable def parabola1 (x : ℝ) := 3 * x^2 - 6 * x + 6
noncomputable def parabola2 (x : ℝ) := -2 * x^2 + 2 * x + 6

theorem parabola_intersection_difference :
  let a := 0
  let c := 8 / 5
  c - a = 8 / 5 := by
  sorry

end parabola_intersection_difference_l17_17381


namespace find_positive_number_l17_17051

-- The definition to state the given condition
def condition1 (n : ℝ) : Prop := n > 0 ∧ n^2 + n = 245

-- The theorem stating the problem and its solution
theorem find_positive_number (n : ℝ) (h : condition1 n) : n = 14 :=
by sorry

end find_positive_number_l17_17051


namespace max_value_2x_plus_y_l17_17407

theorem max_value_2x_plus_y (x y : ℝ) (h : y^2 / 4 + x^2 / 3 = 1) : 2 * x + y ≤ 4 :=
by
  sorry

end max_value_2x_plus_y_l17_17407


namespace elena_alex_total_dollars_l17_17703

theorem elena_alex_total_dollars :
  (5 / 6 : ℚ) + (7 / 15 : ℚ) = (13 / 10 : ℚ) :=
by
    sorry

end elena_alex_total_dollars_l17_17703


namespace probability_of_finding_last_defective_product_on_fourth_inspection_l17_17304

theorem probability_of_finding_last_defective_product_on_fourth_inspection :
  let total_products := 6
  let qualified_products := 4
  let defective_products := 2
  let probability := (4 / 6) * (3 / 5) * (2 / 4) * (1 / 3) + (4 / 6) * (2 / 5) * (3 / 4) * (1 / 3) + (2 / 6) * (4 / 5) * (3 / 4) * (1 / 3)
  probability = 1 / 5 :=
by
  let total_products := 6
  let qualified_products := 4
  let defective_products := 2
  let probability := (4 / 6) * (3 / 5) * (2 / 4) * (1 / 3) + (4 / 6) * (2 / 5) * (3 / 4) * (1 / 3) + (2 / 6) * (4 / 5) * (3 / 4) * (1 / 3)
  have : probability = 1 / 5 := sorry
  exact this

end probability_of_finding_last_defective_product_on_fourth_inspection_l17_17304


namespace min_value_frac_l17_17979

theorem min_value_frac (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) :
  ∃ x, 0 < x ∧ x < 1 ∧ (∀ y, 0 < y ∧ y < 1 → (a * a / y + b * b / (1 - y)) ≥ (a + b) * (a + b)) ∧ 
       a * a / x + b * b / (1 - x) = (a + b) * (a + b) := 
by {
  sorry
}

end min_value_frac_l17_17979


namespace find_angle_B_and_sin_ratio_l17_17889

variable (A B C a b c : ℝ)
variable (h₁ : a * (Real.sin C - Real.sin A) / (Real.sin C + Real.sin B) = c - b)
variable (h₂ : Real.tan B / Real.tan A + Real.tan B / Real.tan C = 4)

theorem find_angle_B_and_sin_ratio :
  B = Real.pi / 3 ∧ Real.sin A / Real.sin C = (3 + Real.sqrt 5) / 2 ∨ Real.sin A / Real.sin C = (3 - Real.sqrt 5) / 2 :=
by
  sorry

end find_angle_B_and_sin_ratio_l17_17889


namespace sally_baseball_cards_l17_17893

theorem sally_baseball_cards (initial_cards sold_cards : ℕ) (h1 : initial_cards = 39) (h2 : sold_cards = 24) :
  (initial_cards - sold_cards = 15) :=
by
  -- Proof needed
  sorry

end sally_baseball_cards_l17_17893


namespace total_cost_correct_l17_17846

def bun_price : ℝ := 0.1
def buns_count : ℝ := 10
def milk_price : ℝ := 2
def milk_count : ℝ := 2
def egg_price : ℝ := 3 * milk_price

def total_cost : ℝ := (buns_count * bun_price) + (milk_count * milk_price) + egg_price

theorem total_cost_correct : total_cost = 11 := by
  sorry

end total_cost_correct_l17_17846


namespace greater_number_is_64_l17_17317

theorem greater_number_is_64
  (x y : ℕ)
  (h1 : x * y = 2048)
  (h2 : (x + y) - (x - y) = 64)
  (h3 : x > y) :
  x = 64 :=
by
  -- proof to be filled in
  sorry

end greater_number_is_64_l17_17317


namespace band_song_average_l17_17813

/-- 
The school band has 30 songs in their repertoire. 
They played 5 songs in the first set and 7 songs in the second set. 
They will play 2 songs for their encore. 
Assuming the band plays through their entire repertoire, 
how many songs will they play on average in the third and fourth sets?
 -/
theorem band_song_average
    (total_songs : ℕ)
    (first_set_songs : ℕ)
    (second_set_songs : ℕ)
    (encore_songs : ℕ)
    (remaining_sets : ℕ)
    (h_total : total_songs = 30)
    (h_first : first_set_songs = 5)
    (h_second : second_set_songs = 7)
    (h_encore : encore_songs = 2)
    (h_remaining : remaining_sets = 2) :
    (total_songs - (first_set_songs + second_set_songs + encore_songs)) / remaining_sets = 8 := 
by
  -- The proof will go here.
  sorry

end band_song_average_l17_17813


namespace value_of_expression_l17_17554

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : a^2 - b^2 + 6 * b = 9 :=
  sorry

end value_of_expression_l17_17554


namespace platform_length_proof_l17_17306

noncomputable def train_length : ℝ := 1200
noncomputable def time_to_cross_tree : ℝ := 120
noncomputable def time_to_cross_platform : ℝ := 240
noncomputable def speed_of_train : ℝ := train_length / time_to_cross_tree
noncomputable def platform_length : ℝ := 2400 - train_length

theorem platform_length_proof (h1 : train_length = 1200) (h2 : time_to_cross_tree = 120) (h3 : time_to_cross_platform = 240) :
  platform_length = 1200 := by
  sorry

end platform_length_proof_l17_17306


namespace complement_of_union_l17_17859

def A : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def U : Set ℤ := Set.univ

theorem complement_of_union (x : ℤ) : (x ∉ A ∪ B) ↔ ∃ k : ℤ, x = 3 * k := 
by {
  sorry
}

end complement_of_union_l17_17859


namespace find_daily_rate_second_company_l17_17289

def daily_rate_second_company (x : ℝ) : Prop :=
  let total_cost_1 := 21.95 + 0.19 * 150
  let total_cost_2 := x + 0.21 * 150
  total_cost_1 = total_cost_2

theorem find_daily_rate_second_company : daily_rate_second_company 18.95 :=
  by
  unfold daily_rate_second_company
  sorry

end find_daily_rate_second_company_l17_17289


namespace vishal_investment_more_than_trishul_l17_17462

theorem vishal_investment_more_than_trishul :
  ∀ (V T R : ℝ), R = 2000 → T = R - 0.10 * R → V + T + R = 5780 → (V - T) / T * 100 = 10 :=
by
  intros V T R hR hT hSum
  sorry

end vishal_investment_more_than_trishul_l17_17462


namespace quadratic_has_distinct_real_roots_l17_17055

theorem quadratic_has_distinct_real_roots (q : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x^2 + 8 * x + q = 0) ↔ q < 16 :=
by
  -- only the statement is provided, the proof is omitted
  sorry

end quadratic_has_distinct_real_roots_l17_17055


namespace triangle_DEF_rotate_180_D_l17_17150

def rotate_180_degrees_clockwise (E D : (ℝ × ℝ)) : (ℝ × ℝ) :=
  let ED := (D.1 - E.1, D.2 - E.2)
  (E.1 - ED.1, E.2 - ED.2)

theorem triangle_DEF_rotate_180_D (D E F : (ℝ × ℝ))
  (hD : D = (3, 2)) (hE : E = (6, 5)) (hF : F = (6, 2)) :
  rotate_180_degrees_clockwise E D = (9, 8) :=
by
  rw [hD, hE, rotate_180_degrees_clockwise]
  sorry

end triangle_DEF_rotate_180_D_l17_17150


namespace range_of_a_l17_17902

variable (a : ℝ)

def p : Prop := ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ (x : ℝ), x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (h : p a ∧ q a) : a ∈ Set.Iic (-2) ∪ {1} := by
  sorry

end range_of_a_l17_17902


namespace geometric_sequence_a9_l17_17793

theorem geometric_sequence_a9
  (a : ℕ → ℤ)
  (q : ℤ)
  (h1 : a 3 * a 6 = -32)
  (h2 : a 4 + a 5 = 4)
  (hq : ∃ n : ℤ, q = n)
  : a 10 = -256 := 
sorry

end geometric_sequence_a9_l17_17793


namespace area_of_square_with_diagonal_l17_17237

theorem area_of_square_with_diagonal (d : ℝ) (s : ℝ) (hsq : d = s * Real.sqrt 2) (hdiagonal : d = 12 * Real.sqrt 2) : 
  s^2 = 144 :=
by
  -- Proof details would go here.
  sorry

end area_of_square_with_diagonal_l17_17237


namespace ellipse_standard_equation_and_point_l17_17806
  
noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x^2) / 25 + (y^2) / 9 = 1

def exists_dot_product_zero_point (P : ℝ × ℝ) : Prop :=
  let F1 := (-4, 0)
  let F2 := (4, 0)
  (P.1 + 4) * (P.1 - 4) + P.2 * P.2 = 0

theorem ellipse_standard_equation_and_point :
  ∃ (P : ℝ × ℝ), ellipse_equation P.1 P.2 ∧ exists_dot_product_zero_point P ∧ 
    ((P = ((5 * Real.sqrt 7) / 4, 9 / 4)) ∨ (P = (-(5 * Real.sqrt 7) / 4, 9 / 4)) ∨ 
    (P = ((5 * Real.sqrt 7) / 4, -(9 / 4))) ∨ (P = (-(5 * Real.sqrt 7) / 4, -(9 / 4)))) :=
by 
  sorry

end ellipse_standard_equation_and_point_l17_17806


namespace even_function_odd_function_neither_even_nor_odd_function_l17_17853

def f (x : ℝ) : ℝ := 1 + x^2 + x^4
def g (x : ℝ) : ℝ := x + x^3 + x^5
def h (x : ℝ) : ℝ := 1 + x + x^2 + x^3 + x^4

theorem even_function : ∀ x : ℝ, f (-x) = f x :=
by
  sorry

theorem odd_function : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

theorem neither_even_nor_odd_function : ∀ x : ℝ, (h (-x) ≠ h x) ∧ (h (-x) ≠ -h x) :=
by
  sorry

end even_function_odd_function_neither_even_nor_odd_function_l17_17853


namespace quadratic_equation_no_real_roots_l17_17898

theorem quadratic_equation_no_real_roots :
  ∀ (x : ℝ), ¬ (x^2 - 2 * x + 3 = 0) :=
by
  intro x
  sorry

end quadratic_equation_no_real_roots_l17_17898


namespace range_of_k_l17_17784

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x^2 - k*x + 1 > 0) → (-2 < k ∧ k < 2) :=
by
  sorry

end range_of_k_l17_17784


namespace farmer_cows_after_selling_l17_17803

theorem farmer_cows_after_selling
  (initial_cows : ℕ) (new_cows : ℕ) (quarter_factor : ℕ)
  (h_initial : initial_cows = 51)
  (h_new : new_cows = 5)
  (h_quarter : quarter_factor = 4) :
  initial_cows + new_cows - (initial_cows + new_cows) / quarter_factor = 42 :=
by
  sorry

end farmer_cows_after_selling_l17_17803


namespace values_of_x_l17_17756

def P (x : ℝ) : ℝ := x^3 - 5 * x^2 + 8 * x

theorem values_of_x (x : ℝ) :
  P x = P (x + 1) ↔ (x = 1 ∨ x = 4 / 3) :=
by sorry

end values_of_x_l17_17756


namespace expression_equals_36_l17_17235

def k := 13

theorem expression_equals_36 : 13 * (3 - 3 / 13) = 36 := by
  sorry

end expression_equals_36_l17_17235


namespace binary_multiplication_l17_17097

theorem binary_multiplication : (0b1101 * 0b111 = 0b1001111) :=
by {
  -- placeholder for proof
  sorry
}

end binary_multiplication_l17_17097


namespace timothy_read_pages_l17_17128

theorem timothy_read_pages 
    (mon_tue_pages : Nat) (wed_pages : Nat) (thu_sat_pages : Nat) 
    (sun_read_pages : Nat) (sun_review_pages : Nat) : 
    mon_tue_pages = 45 → wed_pages = 50 → thu_sat_pages = 40 → sun_read_pages = 25 → sun_review_pages = 15 →
    (2 * mon_tue_pages + wed_pages + 3 * thu_sat_pages + sun_read_pages + sun_review_pages = 300) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end timothy_read_pages_l17_17128


namespace cups_of_sugar_already_put_in_l17_17194

-- Defining the given conditions
variable (f s x : ℕ)

-- The total flour and sugar required
def total_flour_required := 9
def total_sugar_required := 6

-- Mary needs to add 7 more cups of flour than cups of sugar
def remaining_flour_to_sugar_difference := 7

-- Proof goal: to find how many cups of sugar Mary has already put in
theorem cups_of_sugar_already_put_in (total_flour_remaining : ℕ := 9 - 7)
    (remaining_sugar : ℕ := 9 - 7) 
    (already_added_sugar : ℕ := 6 - 2) : already_added_sugar = 4 :=
by sorry

end cups_of_sugar_already_put_in_l17_17194


namespace equation_of_tangent_line_l17_17399

-- Definitions for the given conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * x
def P : ℝ × ℝ := (-1, 4)
def slope_of_tangent (a : ℝ) (x : ℝ) : ℝ := -6 * x^2 - 2

-- The main theorem to prove the equation of the tangent line
theorem equation_of_tangent_line (a : ℝ) (ha : f a (-1) = 4) :
  8 * x + y + 4 = 0 := by
  sorry

end equation_of_tangent_line_l17_17399


namespace largest_circle_diameter_l17_17962

theorem largest_circle_diameter
  (A : ℝ) (hA : A = 180)
  (w l : ℝ) (hw : l = 3 * w)
  (hA2 : w * l = A) :
  ∃ d : ℝ, d = 16 * Real.sqrt 15 / Real.pi :=
by
  sorry

end largest_circle_diameter_l17_17962


namespace solution_interval_l17_17302

noncomputable def set_of_solutions : Set ℝ :=
  {x : ℝ | 4 * x - 3 < (x - 2) ^ 2 ∧ (x - 2) ^ 2 < 6 * x - 5}

theorem solution_interval :
  set_of_solutions = {x : ℝ | 7 < x ∧ x < 9} := by
  sorry

end solution_interval_l17_17302


namespace john_collects_crabs_l17_17388

-- Definitions for the conditions
def baskets_per_week : ℕ := 3
def crabs_per_basket : ℕ := 4
def price_per_crab : ℕ := 3
def total_income : ℕ := 72

-- Definition for the question
def times_per_week_to_collect (baskets_per_week crabs_per_basket price_per_crab total_income : ℕ) : ℕ :=
  (total_income / price_per_crab) / (baskets_per_week * crabs_per_basket)

-- The theorem statement
theorem john_collects_crabs (h1 : baskets_per_week = 3) (h2 : crabs_per_basket = 4) (h3 : price_per_crab = 3) (h4 : total_income = 72) :
  times_per_week_to_collect baskets_per_week crabs_per_basket price_per_crab total_income = 2 :=
by
  sorry

end john_collects_crabs_l17_17388


namespace proof_of_problem_l17_17402

variable (f : ℝ → ℝ)
variable (h_nonzero : ∀ x, f x ≠ 0)
variable (h_equation : ∀ x y, f (x * y) = y * f x + x * f y)

theorem proof_of_problem :
  f 1 = 0 ∧ f (-1) = 0 ∧ (∀ x, f (-x) = -f x) :=
by
  sorry

end proof_of_problem_l17_17402


namespace product_prices_determined_max_product_A_pieces_l17_17677

theorem product_prices_determined (a b : ℕ) :
  (20 * a + 15 * b = 380) →
  (15 * a + 10 * b = 280) →
  a = 16 ∧ b = 4 :=
by sorry

theorem max_product_A_pieces (x : ℕ) :
  (16 * x + 4 * (100 - x) ≤ 900) →
  x ≤ 41 :=
by sorry

end product_prices_determined_max_product_A_pieces_l17_17677


namespace number_of_boys_in_school_l17_17857

theorem number_of_boys_in_school (B : ℝ) (h1 : 542.0 = B + 155) : B = 387 :=
by
  sorry

end number_of_boys_in_school_l17_17857


namespace min_abs_sum_l17_17303

theorem min_abs_sum (x y z : ℝ) (hx : 0 ≤ x) (hxy : x ≤ y) (hyz : y ≤ z) (hz : z ≤ 4) 
  (hy_eq : y^2 = x^2 + 2) (hz_eq : z^2 = y^2 + 2) : 
  |x - y| + |y - z| = 4 - 2 * Real.sqrt 3 :=
sorry

end min_abs_sum_l17_17303


namespace original_amount_of_money_l17_17376

-- Define the conditions
variables (x : ℕ) -- daily allowance

-- Spending details
def spend_10_days := 6 * 10 - 6 * x
def spend_15_days := 15 * 3 - 3 * x

-- Lean proof statement
theorem original_amount_of_money (h : spend_10_days = spend_15_days) : (6 * 10 - 6 * x) = 30 :=
by
  sorry

end original_amount_of_money_l17_17376


namespace find_extrema_l17_17881

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 6

theorem find_extrema :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → f x ≤ 6) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ f x = 6) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → 2 ≤ f x) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ f x = 2) :=
by sorry

end find_extrema_l17_17881


namespace bird_families_flew_to_Asia_l17_17098

-- Variables/Parameters
variable (A : ℕ) (X : ℕ)
axiom hA : A = 47
axiom hX : X = A + 47

-- Theorem Statement
theorem bird_families_flew_to_Asia : X = 94 :=
by
  sorry

end bird_families_flew_to_Asia_l17_17098


namespace son_daughter_eggs_per_morning_l17_17584

-- Define the given conditions in Lean 4
def trays_per_week : Nat := 2
def eggs_per_tray : Nat := 24
def eggs_per_night_rhea_husband : Nat := 4
def nights_per_week : Nat := 7
def uneaten_eggs_per_week : Nat := 6

-- Define the total eggs bought per week
def total_eggs_per_week : Nat := trays_per_week * eggs_per_tray

-- Define the eggs eaten per week by Rhea and her husband
def eggs_eaten_per_week_rhea_husband : Nat := eggs_per_night_rhea_husband * nights_per_week

-- Prove the number of eggs eaten by son and daughter every morning
theorem son_daughter_eggs_per_morning :
  (total_eggs_per_week - eggs_eaten_per_week_rhea_husband - uneaten_eggs_per_week) = 14 :=
sorry

end son_daughter_eggs_per_morning_l17_17584


namespace maximum_area_of_garden_l17_17748

theorem maximum_area_of_garden (w l : ℝ) 
  (h_perimeter : 2 * w + l = 400) : 
  ∃ (A : ℝ), A = 20000 ∧ A = w * l ∧ l = 400 - 2 * w ∧ ∀ (w' : ℝ) (l' : ℝ),
    2 * w' + l' = 400 → w' * l' ≤ 20000 :=
by
  sorry

end maximum_area_of_garden_l17_17748


namespace find_d_l17_17186

-- Define the conditions
variables (x₀ y₀ c : ℝ)

-- Define the system of equations
def system_of_equations : Prop :=
  x₀ * y₀ = 6 ∧ x₀^2 * y₀ + x₀ * y₀^2 + x₀ + y₀ + c = 2

-- Define the target proof problem
theorem find_d (h : system_of_equations x₀ y₀ c) : x₀^2 + y₀^2 = 69 :=
sorry

end find_d_l17_17186


namespace find_a_minus_c_l17_17822

theorem find_a_minus_c (a b c : ℝ) (h1 : (a + b) / 2 = 110) (h2 : (b + c) / 2 = 170) : a - c = -120 :=
by
  sorry

end find_a_minus_c_l17_17822


namespace comparison_of_square_roots_l17_17013

theorem comparison_of_square_roots (P Q : ℝ) (hP : P = Real.sqrt 2) (hQ : Q = Real.sqrt 6 - Real.sqrt 2) : P > Q :=
by
  sorry

end comparison_of_square_roots_l17_17013


namespace gcd_three_numbers_l17_17779

theorem gcd_three_numbers (a b c : ℕ) (h₁ : a = 13847) (h₂ : b = 21353) (h₃ : c = 34691) : Nat.gcd (Nat.gcd a b) c = 5 := by sorry

end gcd_three_numbers_l17_17779


namespace restaurant_total_cost_l17_17587

theorem restaurant_total_cost :
  let vegetarian_cost := 5
  let chicken_cost := 7
  let steak_cost := 10
  let kids_cost := 3
  let tax_rate := 0.10
  let tip_rate := 0.15
  let num_vegetarians := 3
  let num_chicken_lovers := 4
  let num_steak_enthusiasts := 2
  let num_kids_hot_dog := 3
  let subtotal := (num_vegetarians * vegetarian_cost) + (num_chicken_lovers * chicken_cost) + (num_steak_enthusiasts * steak_cost) + (num_kids_hot_dog * kids_cost)
  let tax := subtotal * tax_rate
  let tip := subtotal * tip_rate
  let total_cost := subtotal + tax + tip
  total_cost = 90 :=
by sorry

end restaurant_total_cost_l17_17587


namespace negative_integer_solution_l17_17292

theorem negative_integer_solution (M : ℤ) (h1 : 2 * M^2 + M = 12) (h2 : M < 0) : M = -4 :=
sorry

end negative_integer_solution_l17_17292


namespace find_other_number_l17_17211

-- Given: 
-- LCM of two numbers is 2310
-- GCD of two numbers is 55
-- One number is 605,
-- Prove: The other number is 210

theorem find_other_number (a b : ℕ) (h_lcm : Nat.lcm a b = 2310) (h_gcd : Nat.gcd a b = 55) (h_b : b = 605) :
  a = 210 :=
sorry

end find_other_number_l17_17211


namespace multiple_of_9_l17_17202

noncomputable def digit_sum (x : ℕ) : ℕ := sorry  -- Placeholder for the digit sum function

theorem multiple_of_9 (n : ℕ) (h1 : digit_sum n = digit_sum (3 * n))
  (h2 : ∀ x, x % 9 = digit_sum x % 9) :
  n % 9 = 0 :=
by
  sorry

end multiple_of_9_l17_17202


namespace represent_1947_as_squares_any_integer_as_squares_l17_17710

theorem represent_1947_as_squares :
  ∃ (a b c : ℤ), 1947 = a * a - b * b - c * c :=
by
  use 488, 486, 1
  sorry

theorem any_integer_as_squares (n : ℤ) :
  ∃ (a b c d : ℤ), n = a * a + b * b + c * c + d * d :=
by
  sorry

end represent_1947_as_squares_any_integer_as_squares_l17_17710


namespace f_plus_one_odd_l17_17949

noncomputable def f : ℝ → ℝ := sorry

theorem f_plus_one_odd (f : ℝ → ℝ)
  (h : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂ + 1) :
  ∀ x : ℝ, f x + 1 = -(f (-x) + 1) :=
sorry

end f_plus_one_odd_l17_17949


namespace total_water_needed_l17_17642

def adults : ℕ := 7
def children : ℕ := 3
def hours : ℕ := 24
def replenish_bottles : ℚ := 14
def water_per_hour_adult : ℚ := 1/2
def water_per_hour_child : ℚ := 1/3

theorem total_water_needed : 
  let total_water_per_hour := (adults * water_per_hour_adult) + (children * water_per_hour_child)
  let total_water := total_water_per_hour * hours 
  let initial_water_needed := total_water - replenish_bottles
  initial_water_needed = 94 := by 
  sorry

end total_water_needed_l17_17642


namespace y_intercept_tangent_line_l17_17838

noncomputable def tangent_line_y_intercept (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ) (htangent: Prop) : ℝ :=
  if r1 = 3 ∧ r2 = 2 ∧ c1 = (3, 0) ∧ c2 = (8, 0) ∧ htangent = true then 6 * Real.sqrt 6 else 0

theorem y_intercept_tangent_line (h : tangent_line_y_intercept 3 2 (3, 0) (8, 0) true = 6 * Real.sqrt 6) :
  tangent_line_y_intercept 3 2 (3, 0) (8, 0) true = 6 * Real.sqrt 6 :=
by
  exact h

end y_intercept_tangent_line_l17_17838


namespace expression_equivalence_l17_17116

theorem expression_equivalence (a b : ℝ) :
  let P := a + b
  let Q := a - b
  (P + Q)^2 / (P - Q)^2 - (P - Q)^2 / (P + Q)^2 = (a^2 + b^2) * (a^2 - b^2) / (a^2 * b^2) :=
by
  sorry

end expression_equivalence_l17_17116


namespace expand_product_l17_17253

theorem expand_product (y : ℝ) : 3 * (y - 6) * (y + 9) = 3 * y^2 + 9 * y - 162 :=
by sorry

end expand_product_l17_17253


namespace remainder_of_sum_l17_17008

theorem remainder_of_sum :
  ((88134 + 88135 + 88136 + 88137 + 88138 + 88139) % 9) = 6 :=
by
  sorry

end remainder_of_sum_l17_17008


namespace largest_square_plots_l17_17390

theorem largest_square_plots (width length pathway_material : Nat) (width_eq : width = 30) (length_eq : length = 60) (pathway_material_eq : pathway_material = 2010) : ∃ (n : Nat), n * (2 * n) = 578 := 
by
  sorry

end largest_square_plots_l17_17390


namespace smallest_e_value_l17_17108

noncomputable def poly := (1, -3, 7, -2/5)

theorem smallest_e_value (a b c d e : ℤ) 
  (h_poly_eq : a * (1)^4 + b * (1)^3 + c * (1)^2 + d * (1) + e = 0)
  (h_poly_eq_2 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
  (h_poly_eq_3 : a * 7^4 + b * 7^3 + c * 7^2 + d * 7 + e = 0)
  (h_poly_eq_4 : a * (-2/5)^4 + b * (-2/5)^3 + c * (-2/5)^2 + d * (-2/5) + e = 0)
  (h_e_positive : e > 0) :
  e = 42 :=
sorry

end smallest_e_value_l17_17108


namespace Hay_s_Linens_sales_l17_17413

theorem Hay_s_Linens_sales :
  ∃ (n : ℕ), 500 ≤ 52 * n ∧ 52 * n ≤ 700 ∧
             ∀ m, (500 ≤ 52 * m ∧ 52 * m ≤ 700) → n ≤ m :=
sorry

end Hay_s_Linens_sales_l17_17413


namespace factor_100_minus_16y2_l17_17739

theorem factor_100_minus_16y2 (y : ℝ) : 100 - 16 * y^2 = 4 * (5 - 2 * y) * (5 + 2 * y) := 
by sorry

end factor_100_minus_16y2_l17_17739


namespace infinitely_many_numbers_composed_of_0_and_1_divisible_by_2017_l17_17408

theorem infinitely_many_numbers_composed_of_0_and_1_divisible_by_2017 :
  ∀ n : ℕ, ∃ m : ℕ, (m ∈ {x | ∀ d ∈ Nat.digits 10 x, d = 0 ∨ d = 1}) ∧ 2017 ∣ m :=
by
  sorry

end infinitely_many_numbers_composed_of_0_and_1_divisible_by_2017_l17_17408


namespace fraction_spent_on_raw_material_l17_17170

variable (C : ℝ)
variable (x : ℝ)

theorem fraction_spent_on_raw_material :
  C - x * C - (1/10) * (C * (1 - x)) = 0.675 * C → x = 1/4 :=
by
  sorry

end fraction_spent_on_raw_material_l17_17170


namespace cube_painting_equiv_1260_l17_17457

def num_distinguishable_paintings_of_cube : Nat :=
  1260

theorem cube_painting_equiv_1260 :
  ∀ (colors : Fin 8 → Color), -- assuming we have a type Color representing colors
    (∀ i j : Fin 6, i ≠ j → colors i ≠ colors j) →  -- each face has a different color
    ∃ f : Cube × Fin 8 → Cube × Fin 8, -- considering symmetry transformations (rotations)
      num_distinguishable_paintings_of_cube = 1260 :=
by
  -- Proof would go here
  sorry

end cube_painting_equiv_1260_l17_17457


namespace parabola_tangent_y_intercept_correct_l17_17700

noncomputable def parabola_tangent_y_intercept (a : ℝ) : Prop :=
  let C := fun x : ℝ => x^2
  let slope := 2 * a
  let tangent_line := fun x : ℝ => slope * (x - a) + C a
  let Q := (0, tangent_line 0)
  Q = (0, -a^2)

-- Statement of the problem as a Lean theorem
theorem parabola_tangent_y_intercept_correct (a : ℝ) (h : a > 0) :
  parabola_tangent_y_intercept a := 
by 
  sorry

end parabola_tangent_y_intercept_correct_l17_17700


namespace general_term_less_than_zero_from_13_l17_17093

-- Define the arithmetic sequence and conditions
def an (n : ℕ) : ℝ := 12 - n

-- Condition: a_3 = 9
def a3_condition : Prop := an 3 = 9

-- Condition: a_9 = 3
def a9_condition : Prop := an 9 = 3

-- Prove the general term of the sequence is 12 - n
theorem general_term (n : ℕ) (h3 : a3_condition) (h9 : a9_condition) :
  an n = 12 - n := 
sorry

-- Prove that the sequence becomes less than 0 starting from the 13th term
theorem less_than_zero_from_13 (h3 : a3_condition) (h9 : a9_condition) :
  ∀ n, n ≥ 13 → an n < 0 :=
sorry

end general_term_less_than_zero_from_13_l17_17093


namespace intersection_of_M_and_N_l17_17761

def M : Set ℝ := {x | 0 < x ∧ x < 3}
def N : Set ℝ := {x | x > 2 ∨ x < -2}
def expected_intersection : Set ℝ := {x | 2 < x ∧ x < 3}

theorem intersection_of_M_and_N : M ∩ N = expected_intersection := by
  sorry

end intersection_of_M_and_N_l17_17761


namespace pigeon_problem_l17_17667

theorem pigeon_problem (x y : ℕ) :
  (1 / 6 : ℝ) * (x + y) = y - 1 ∧ x - 1 = y + 1 → x = 4 ∧ y = 2 :=
by
  sorry

end pigeon_problem_l17_17667


namespace amy_height_l17_17833

variable (A H N : ℕ)

theorem amy_height (h1 : A = 157) (h2 : A = H + 4) (h3 : H = N + 3) :
  N = 150 := sorry

end amy_height_l17_17833


namespace cricket_runs_l17_17563

theorem cricket_runs (x a b c d : ℕ) 
    (h1 : a = 1 * x) 
    (h2 : b = 3 * x) 
    (h3 : c = 5 * x) 
    (h4 : d = 4 * x) 
    (total_runs : 1 * x + 3 * x + 5 * x + 4 * x = 234) :
  a = 18 ∧ b = 54 ∧ c = 90 ∧ d = 72 := by
  sorry

end cricket_runs_l17_17563


namespace expression_value_l17_17318

theorem expression_value (a b c : ℚ) (h₁ : b = 8) (h₂ : c = 5) (h₃ : a * b * c = 2 * (a + b + c) + 14) : 
  (c - a) ^ 2 + b = 8513 / 361 := by 
  sorry

end expression_value_l17_17318


namespace jerry_boxes_l17_17754

theorem jerry_boxes (boxes_sold boxes_left : ℕ) (h₁ : boxes_sold = 5) (h₂ : boxes_left = 5) : (boxes_sold + boxes_left = 10) :=
by
  sorry

end jerry_boxes_l17_17754


namespace dozen_chocolate_bars_cost_l17_17360

theorem dozen_chocolate_bars_cost
  (cost_mag : ℕ → ℝ) (cost_choco_bar : ℕ → ℝ)
  (H1 : cost_mag 1 = 1)
  (H2 : 4 * (cost_choco_bar 1) = 8 * (cost_mag 1)) :
  12 * (cost_choco_bar 1) = 24 := 
sorry

end dozen_chocolate_bars_cost_l17_17360


namespace min_rice_proof_l17_17760

noncomputable def minRicePounds : ℕ := 2

theorem min_rice_proof (o r : ℕ) (h1 : o ≥ 8 + 3 * r / 4) (h2 : o ≤ 5 * r) :
  r ≥ 2 :=
by
  sorry

end min_rice_proof_l17_17760


namespace volume_of_parallelepiped_l17_17501

theorem volume_of_parallelepiped (x y z : ℝ)
  (h1 : (x^2 + y^2) * z^2 = 13)
  (h2 : (y^2 + z^2) * x^2 = 40)
  (h3 : (x^2 + z^2) * y^2 = 45) :
  x * y * z = 6 :=
by 
  sorry

end volume_of_parallelepiped_l17_17501


namespace lines_intersect_at_l17_17727

theorem lines_intersect_at :
  ∃ t u : ℝ, (∃ (x y : ℝ),
    (x = 2 + 3 * t ∧ y = 4 - 2 * t) ∧
    (x = -1 + 6 * u ∧ y = 5 + u) ∧
    (x = 1/5 ∧ y = 26/5)) :=
by
  sorry

end lines_intersect_at_l17_17727


namespace peytons_children_l17_17978

theorem peytons_children (C : ℕ) (juice_per_week : ℕ) (weeks_in_school_year : ℕ) (total_juice_boxes : ℕ) 
  (h1 : juice_per_week = 5) 
  (h2 : weeks_in_school_year = 25) 
  (h3 : total_juice_boxes = 375)
  (h4 : C * (juice_per_week * weeks_in_school_year) = total_juice_boxes) 
  : C = 3 :=
sorry

end peytons_children_l17_17978


namespace mod_residue_l17_17589

theorem mod_residue : (250 * 15 - 337 * 5 + 22) % 13 = 7 := by
  sorry

end mod_residue_l17_17589


namespace probability_of_one_girl_conditional_probability_of_one_girl_given_at_least_one_l17_17825

/- Define number of boys and girls -/
def num_boys : ℕ := 5
def num_girls : ℕ := 3

/- Define number of students selected -/
def num_selected : ℕ := 2

/- Define the total number of ways to select -/
def total_ways : ℕ := Nat.choose (num_boys + num_girls) num_selected

/- Define the number of ways to select exactly one girl -/
def ways_one_girl : ℕ := Nat.choose num_girls 1 * Nat.choose num_boys 1

/- Define the number of ways to select at least one girl -/
def ways_at_least_one_girl : ℕ := total_ways - Nat.choose num_boys num_selected

/- Define the first probability: exactly one girl participates -/
def prob_one_girl : ℚ := ways_one_girl / total_ways

/- Define the second probability: exactly one girl given at least one girl -/
def prob_one_girl_given_at_least_one : ℚ := ways_one_girl / ways_at_least_one_girl

theorem probability_of_one_girl : prob_one_girl = 15 / 28 := by
  sorry

theorem conditional_probability_of_one_girl_given_at_least_one : prob_one_girl_given_at_least_one = 5 / 6 := by
  sorry

end probability_of_one_girl_conditional_probability_of_one_girl_given_at_least_one_l17_17825


namespace dot_not_line_l17_17033

variable (D S DS T : Nat)
variable (h1 : DS = 20) (h2 : S = 36) (h3 : T = 60)
variable (h4 : T = D + S - DS)

theorem dot_not_line : (D - DS) = 24 :=
by
  sorry

end dot_not_line_l17_17033


namespace gcd_factorial_8_10_l17_17012

theorem gcd_factorial_8_10 : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end gcd_factorial_8_10_l17_17012


namespace rectangle_perimeter_eq_26_l17_17716

theorem rectangle_perimeter_eq_26 (a b c W : ℕ) (h_tri : a = 5 ∧ b = 12 ∧ c = 13)
  (h_right_tri : a^2 + b^2 = c^2) (h_W : W = 3) (h_area_eq : 1/2 * (a * b) = (W * L))
  (A L : ℕ) (hA : A = 30) (hL : L = A / W) :
  2 * (L + W) = 26 :=
by
  sorry

end rectangle_perimeter_eq_26_l17_17716


namespace largest_increase_between_2006_and_2007_l17_17528

-- Define the number of students taking the AMC in each year
def students_2002 := 50
def students_2003 := 55
def students_2004 := 63
def students_2005 := 70
def students_2006 := 75
def students_2007_AMC10 := 90
def students_2007_AMC12 := 15

-- Define the total number of students participating in any AMC contest each year
def total_students_2002 := students_2002
def total_students_2003 := students_2003
def total_students_2004 := students_2004
def total_students_2005 := students_2005
def total_students_2006 := students_2006
def total_students_2007 := students_2007_AMC10 + students_2007_AMC12

-- Function to calculate percentage increase
def percentage_increase (old new : ℕ) : ℚ :=
  ((new - old : ℕ) : ℚ) / old * 100

-- Calculate percentage increases between the years
def inc_2002_2003 := percentage_increase total_students_2002 total_students_2003
def inc_2003_2004 := percentage_increase total_students_2003 total_students_2004
def inc_2004_2005 := percentage_increase total_students_2004 total_students_2005
def inc_2005_2006 := percentage_increase total_students_2005 total_students_2006
def inc_2006_2007 := percentage_increase total_students_2006 total_students_2007

-- Prove that the largest percentage increase is between 2006 and 2007
theorem largest_increase_between_2006_and_2007 :
  inc_2006_2007 > inc_2005_2006 ∧
  inc_2006_2007 > inc_2004_2005 ∧
  inc_2006_2007 > inc_2003_2004 ∧
  inc_2006_2007 > inc_2002_2003 := 
by {
  sorry
}

end largest_increase_between_2006_and_2007_l17_17528


namespace leah_coins_worth_89_cents_l17_17095

variables (p n d : ℕ)

theorem leah_coins_worth_89_cents (h1 : p + n + d = 15) (h2 : d - 1 = n) : 
  1 * p + 5 * n + 10 * d = 89 := 
sorry

end leah_coins_worth_89_cents_l17_17095


namespace ab_greater_than_a_plus_b_l17_17719

variable {a b : ℝ}
variables (pos_a : 0 < a) (pos_b : 0 < b) (h : a - b = a / b)

theorem ab_greater_than_a_plus_b : a * b > a + b :=
sorry

end ab_greater_than_a_plus_b_l17_17719


namespace max_number_of_eligible_ages_l17_17391

-- Definitions based on the problem conditions
def average_age : ℝ := 31
def std_dev : ℝ := 5
def acceptable_age_range (a : ℝ) : Prop := 26 ≤ a ∧ a ≤ 36
def has_masters_degree : Prop := 24 ≤ 26  -- simplified for context indicated in problem
def has_work_experience : Prop := 26 ≥ 26

-- Define the maximum number of different ages of the eligible applicants
noncomputable def max_diff_ages : ℕ := 36 - 26 + 1  -- This matches the solution step directly

-- The theorem stating the result
theorem max_number_of_eligible_ages :
  max_diff_ages = 11 :=
by {
  sorry
}

end max_number_of_eligible_ages_l17_17391


namespace find_inverse_of_f_at_4_l17_17465

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^2

-- Statement of the problem
theorem find_inverse_of_f_at_4 : ∃ t : ℝ, f t = 4 ∧ t ≤ 1 ∧ t = -1 := by
  sorry

end find_inverse_of_f_at_4_l17_17465


namespace sufficient_but_not_necessary_condition_for_negative_root_l17_17161

def quadratic_equation (a x : ℝ) : ℝ := a * x^2 + 2 * x + 1

theorem sufficient_but_not_necessary_condition_for_negative_root 
  (a : ℝ) (h : a < 0) : 
  (∃ x : ℝ, quadratic_equation a x = 0 ∧ x < 0) ∧ 
  (∀ a : ℝ, (∃ x : ℝ, quadratic_equation a x = 0 ∧ x < 0) → a ≤ 0) :=
sorry

end sufficient_but_not_necessary_condition_for_negative_root_l17_17161


namespace problem_1_l17_17610

theorem problem_1 :
  (5 / ((1 / (1 * 2)) + (1 / (2 * 3)) + (1 / (3 * 4)) + (1 / (4 * 5)) + (1 / (5 * 6)))) = 6 := by
  sorry

end problem_1_l17_17610


namespace ellipse_distance_pf2_l17_17723

noncomputable def ellipse_focal_length := 2 * Real.sqrt 2
noncomputable def ellipse_equation (a : ℝ) (a_gt_one : a > 1)
  (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  (x^2 / a) + y^2 = 1

theorem ellipse_distance_pf2
  (a : ℝ) (a_gt_one : a > 1)
  (focus_distance : 2 * Real.sqrt (a - 1) = 2 * Real.sqrt 2)
  (F1 F2 P : ℝ × ℝ)
  (on_ellipse : ellipse_equation a a_gt_one P)
  (PF1_eq_two : dist P F1 = 2)
  (a_eq : a = 3) :
  dist P F2 = 2 * Real.sqrt 3 - 2 := 
sorry

end ellipse_distance_pf2_l17_17723


namespace poly_div_factor_l17_17905

theorem poly_div_factor (c : ℚ) : 2 * x + 7 ∣ 8 * x^4 + 27 * x^3 + 6 * x^2 + c * x - 49 ↔
  c = 47.25 :=
  sorry

end poly_div_factor_l17_17905


namespace stratified_sampling_community_A_l17_17239

theorem stratified_sampling_community_A :
  let A_households := 360
  let B_households := 270
  let C_households := 180
  let total_households := A_households + B_households + C_households
  let total_units := 90
  (A_households : ℕ) / total_households * total_units = 40 :=
by
  let A_households := 360
  let B_households := 270
  let C_households := 180
  let total_households := A_households + B_households + C_households
  let total_units := 90
  have : total_households = 810 := by sorry
  have : (A_households : ℕ) / total_households * total_units = 40 := by sorry
  exact this

end stratified_sampling_community_A_l17_17239


namespace sector_angle_l17_17814

theorem sector_angle (r : ℝ) (S_sector : ℝ) (h_r : r = 2) (h_S : S_sector = (2 / 5) * π) : 
  (∃ α : ℝ, S_sector = (1 / 2) * α * r^2 ∧ α = (π / 5)) :=
by
  use π / 5
  sorry

end sector_angle_l17_17814


namespace base_b_square_of_15_l17_17386

theorem base_b_square_of_15 (b : ℕ) (h : (b + 5) * (b + 5) = 4 * b^2 + 3 * b + 6) : b = 8 :=
sorry

end base_b_square_of_15_l17_17386


namespace integer_side_lengths_triangle_l17_17914

theorem integer_side_lengths_triangle :
  ∃ (a b c : ℤ), (abc = 2 * (a - 1) * (b - 1) * (c - 1)) ∧
            (a = 8 ∧ b = 7 ∧ c = 3 ∨ a = 6 ∧ b = 5 ∧ c = 4) := 
by
  sorry

end integer_side_lengths_triangle_l17_17914


namespace parking_lot_perimeter_l17_17738

theorem parking_lot_perimeter (x y: ℝ) 
  (h1: x = (2 / 3) * y)
  (h2: x^2 + y^2 = 400)
  (h3: x * y = 120) :
  2 * (x + y) = 20 * Real.sqrt 5 :=
by
  sorry

end parking_lot_perimeter_l17_17738


namespace triangle_is_obtuse_l17_17324

-- Definitions based on given conditions
def is_obtuse_triangle (a b c : ℝ) : Prop :=
  if a ≥ b ∧ a ≥ c then a^2 > b^2 + c^2
  else if b ≥ a ∧ b ≥ c then b^2 > a^2 + c^2
  else c^2 > a^2 + b^2

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Statement to prove
theorem triangle_is_obtuse : is_triangle 4 6 8 ∧ is_obtuse_triangle 4 6 8 :=
by
  sorry

end triangle_is_obtuse_l17_17324


namespace henry_time_proof_l17_17572

-- Define the time Dawson took to run the first leg of the course
def dawson_time : ℝ := 38

-- Define the average time they took to run a leg of the course
def average_time : ℝ := 22.5

-- Define the time Henry took to run the second leg of the course
def henry_time : ℝ := 7

-- Prove that Henry took 7 seconds to run the second leg
theorem henry_time_proof : 
  (dawson_time + henry_time) / 2 = average_time :=
by
  -- This is where the proof would go
  sorry

end henry_time_proof_l17_17572


namespace remainder_17_pow_2037_mod_20_l17_17030

theorem remainder_17_pow_2037_mod_20:
      (17^1) % 20 = 17 ∧
      (17^2) % 20 = 9 ∧
      (17^3) % 20 = 13 ∧
      (17^4) % 20 = 1 → 
      (17^2037) % 20 = 17 := sorry

end remainder_17_pow_2037_mod_20_l17_17030


namespace solve_for_n_l17_17058

theorem solve_for_n (n : ℤ) (h : (1 : ℤ) / (n + 2) + 2 / (n + 2) + n / (n + 2) = 2) : n = 2 :=
sorry

end solve_for_n_l17_17058


namespace calc_15_op_and_op2_l17_17899

def op1 (x : ℤ) : ℤ := 10 - x
def op2 (x : ℤ) : ℤ := x - 10

theorem calc_15_op_and_op2 :
  op2 (op1 15) = -15 :=
by
  sorry

end calc_15_op_and_op2_l17_17899


namespace decrease_in_length_l17_17606

theorem decrease_in_length (L B : ℝ) (h₀ : L ≠ 0) (h₁ : B ≠ 0)
  (h₂ : ∃ (A' : ℝ), A' = 0.72 * L * B)
  (h₃ : ∃ B' : ℝ, B' = B * 0.9) :
  ∃ (x : ℝ), x = 20 :=
by
  sorry

end decrease_in_length_l17_17606


namespace mary_picked_nine_lemons_l17_17135

def num_lemons_sally := 7
def total_num_lemons := 16
def num_lemons_mary := total_num_lemons - num_lemons_sally

theorem mary_picked_nine_lemons :
  num_lemons_mary = 9 := by
  sorry

end mary_picked_nine_lemons_l17_17135


namespace sum_of_five_consecutive_odd_numbers_l17_17522

theorem sum_of_five_consecutive_odd_numbers (x : ℤ) : 
  (x - 4) + (x - 2) + x + (x + 2) + (x + 4) = 5 * x :=
by
  sorry

end sum_of_five_consecutive_odd_numbers_l17_17522


namespace age_difference_between_brother_and_cousin_l17_17768

-- Define the ages used in the problem 
def Lexie_age : ℕ := 8
def Grandma_age : ℕ := 68
def Brother_age : ℕ := Lexie_age - 6
def Sister_age : ℕ := 2 * Lexie_age
def Uncle_age : ℕ := Grandma_age - 12
def Cousin_age : ℕ := Brother_age + 5

-- The proof problem statement in Lean 4
theorem age_difference_between_brother_and_cousin : 
  Brother_age < Cousin_age ∧ Cousin_age - Brother_age = 5 :=
by
  -- Definitions and imports are done above. The statement below should prove the age difference.
  sorry

end age_difference_between_brother_and_cousin_l17_17768


namespace ratio_equivalence_l17_17373

theorem ratio_equivalence (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : x ≠ z)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h : y / (x - z) = (x + 2 * y) / z ∧ (x + 2 * y) / z = x / (y + z)) :
  x / (y + z) = (2 * y - z) / (y + z) :=
by
  sorry

end ratio_equivalence_l17_17373


namespace marble_189_is_gray_l17_17406

def marble_color (n : ℕ) : String :=
  let cycle_length := 14
  let gray_thres := 5
  let white_thres := 9
  let black_thres := 12
  let position := (n - 1) % cycle_length + 1
  if position ≤ gray_thres then "gray"
  else if position ≤ white_thres then "white"
  else if position ≤ black_thres then "black"
  else "blue"

theorem marble_189_is_gray : marble_color 189 = "gray" :=
by {
  -- We assume the necessary definitions and steps discussed above.
  sorry
}

end marble_189_is_gray_l17_17406


namespace tile_count_l17_17948

theorem tile_count (a : ℕ) (h1 : ∃ b : ℕ, b = 2 * a) (h2 : 2 * (Int.floor (a * Real.sqrt 5)) - 1 = 49) :
  2 * a^2 = 50 :=
by
  sorry

end tile_count_l17_17948


namespace inequality_proof_equality_condition_l17_17348

variable {x y z : ℝ}

def positive_reals (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0

theorem inequality_proof (hxyz : positive_reals x y z) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
sorry -- Proof goes here

theorem equality_condition (hxyz : positive_reals x y z) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ x = 2 * z ∧ y = z :=
sorry -- Proof goes here

end inequality_proof_equality_condition_l17_17348


namespace arithmetic_sequence_sum_l17_17546

noncomputable def a_n (a1 d : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d
noncomputable def S_n (a1 d : ℕ) (n : ℕ) : ℕ := n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_sum (a1 d : ℕ) 
  (h1 : a1 + d = 6) 
  (h2 : (a1 + 2 * d)^2 = a1 * (a1 + 6 * d)) 
  (h3 : d ≠ 0) : 
  S_n a1 d 8 = 88 := 
by 
  sorry

end arithmetic_sequence_sum_l17_17546


namespace xiao_ming_kite_payment_l17_17696

/-- Xiao Ming has multiple 1 yuan, 2 yuan, and 5 yuan banknotes. 
    He wants to buy a kite priced at 18 yuan using no more than 10 of these banknotes
    and must use at least two different denominations.
    Prove that there are exactly 11 different ways he can pay. -/
theorem xiao_ming_kite_payment : 
  ∃ (combinations : Nat), 
    (∀ (c1 c2 c5 : Nat), (c1 * 1 + c2 * 2 + c5 * 5 = 18) → 
    (c1 + c2 + c5 ≤ 10) → 
    ((c1 > 0 ∧ c2 > 0) ∨ (c1 > 0 ∧ c5 > 0) ∨ (c2 > 0 ∧ c5 > 0)) →
    combinations = 11) :=
sorry

end xiao_ming_kite_payment_l17_17696


namespace num_cows_on_farm_l17_17433

variables (D C S : ℕ)

def total_legs : ℕ := 8 * S + 2 * D + 4 * C
def total_heads : ℕ := D + C + S

theorem num_cows_on_farm
  (h1 : S = 2 * D)
  (h2 : total_legs D C S = 2 * total_heads D C S + 72)
  (h3 : D + C + S ≤ 40) :
  C = 30 :=
sorry

end num_cows_on_farm_l17_17433


namespace solve_a_value_l17_17800

theorem solve_a_value (a b k : ℝ) 
  (h1 : a^3 * b^2 = k)
  (h2 : a = 5)
  (h3 : b = 2) :
  ∃ a', b = 8 → a' = 2.5 :=
by
  sorry

end solve_a_value_l17_17800


namespace smallest_multiple_of_40_gt_100_l17_17650

theorem smallest_multiple_of_40_gt_100 :
  ∃ x : ℕ, 0 < x ∧ 40 * x > 100 ∧ ∀ y : ℕ, 0 < y ∧ 40 * y > 100 → x ≤ y → 40 * x = 120 :=
by
  sorry

end smallest_multiple_of_40_gt_100_l17_17650


namespace both_selected_prob_l17_17612

-- Given conditions
def prob_Ram := 6 / 7
def prob_Ravi := 1 / 5

-- The mathematically equivalent proof problem statement
theorem both_selected_prob : (prob_Ram * prob_Ravi) = 6 / 35 := by
  sorry

end both_selected_prob_l17_17612


namespace cord_length_before_cut_l17_17635

-- Definitions based on the conditions
def parts_after_cut := 20
def longest_piece := 8
def shortest_piece := 2
def initial_parts := 19

-- Lean statement to prove the length of the cord before it was cut
theorem cord_length_before_cut : 
  (initial_parts * ((longest_piece / 2) + shortest_piece) = 114) :=
by 
  sorry

end cord_length_before_cut_l17_17635


namespace max_x2_y2_on_circle_l17_17631

noncomputable def max_value_on_circle : ℝ :=
  12 + 8 * Real.sqrt 2

theorem max_x2_y2_on_circle (x y : ℝ) (h : x^2 - 4 * x - 4 + y^2 = 0) : 
  x^2 + y^2 ≤ max_value_on_circle := 
by
  sorry

end max_x2_y2_on_circle_l17_17631


namespace complex_quadrant_l17_17188

open Complex

theorem complex_quadrant
  (z1 z2 z : ℂ) (h1 : z1 = 2 + I) (h2 : z2 = 1 - I) (h3 : z = z1 / z2) :
  0 < z.re ∧ 0 < z.im :=
by
  -- sorry to skip the proof steps
  sorry

end complex_quadrant_l17_17188


namespace problems_per_page_is_eight_l17_17999

noncomputable def totalProblems := 60
noncomputable def finishedProblems := 20
noncomputable def totalPages := 5
noncomputable def problemsLeft := totalProblems - finishedProblems
noncomputable def problemsPerPage := problemsLeft / totalPages

theorem problems_per_page_is_eight :
  problemsPerPage = 8 :=
by
  sorry

end problems_per_page_is_eight_l17_17999


namespace watch_loss_percentage_l17_17395

noncomputable def initial_loss_percentage : ℝ :=
  let CP := 350
  let SP_new := 364
  let delta_SP := 140
  show ℝ from 
  sorry

theorem watch_loss_percentage (CP SP_new delta_SP : ℝ) (h₁ : CP = 350)
  (h₂ : SP_new = 364) (h₃ : delta_SP = 140) : 
  initial_loss_percentage = 36 :=
by
  -- Use the hypothesis and solve the corresponding problem
  sorry

end watch_loss_percentage_l17_17395


namespace speed_of_stream_l17_17582

-- Conditions
variables (v : ℝ) -- speed of the stream in kmph
variables (boat_speed_still_water : ℝ := 10) -- man's speed in still water in kmph
variables (distance : ℝ := 90) -- distance traveled down the stream in km
variables (time : ℝ := 5) -- time taken to travel the distance down the stream in hours

-- Proof statement
theorem speed_of_stream : v = 8 :=
  by
    -- effective speed down the stream = boat_speed_still_water + v
    -- given that distance = speed * time
    -- 90 = (10 + v) * 5
    -- solving for v
    sorry

end speed_of_stream_l17_17582


namespace ball_reaches_less_than_5_l17_17220

noncomputable def height_after_bounces (initial_height : ℕ) (ratio : ℝ) (bounces : ℕ) : ℝ :=
  initial_height * (ratio ^ bounces)

theorem ball_reaches_less_than_5 (initial_height : ℕ) (ratio : ℝ) (k : ℕ) (target_height : ℝ) (stop_height : ℝ) 
  (h_initial : initial_height = 500) (h_ratio : ratio = 0.6) (h_target : target_height = 5) (h_stop : stop_height = 0.1) :
  ∃ n, height_after_bounces initial_height ratio n < target_height ∧ 500 * (0.6 ^ 17) < stop_height := by
  sorry

end ball_reaches_less_than_5_l17_17220


namespace percentage_loss_l17_17294

theorem percentage_loss (CP SP : ℝ) (h₁ : CP = 1400) (h₂ : SP = 1232) :
  ((CP - SP) / CP) * 100 = 12 :=
by
  sorry

end percentage_loss_l17_17294


namespace alex_shirts_count_l17_17314

theorem alex_shirts_count (j a b : ℕ) (h1 : j = a + 3) (h2 : b = j + 8) (h3 : b = 15) : a = 4 :=
by
  sorry

end alex_shirts_count_l17_17314


namespace total_weight_loss_l17_17022

def seth_loss : ℝ := 17.53
def jerome_loss : ℝ := 3 * seth_loss
def veronica_loss : ℝ := seth_loss + 1.56
def seth_veronica_loss : ℝ := seth_loss + veronica_loss
def maya_loss : ℝ := seth_veronica_loss - 0.25 * seth_veronica_loss
def total_loss : ℝ := seth_loss + jerome_loss + veronica_loss + maya_loss

theorem total_weight_loss : total_loss = 116.675 := by
  sorry

end total_weight_loss_l17_17022


namespace find_volume_of_12_percent_solution_l17_17366

variable (x y : ℝ)

theorem find_volume_of_12_percent_solution
  (h1 : x + y = 60)
  (h2 : 0.02 * x + 0.12 * y = 3) :
  y = 18 := 
sorry

end find_volume_of_12_percent_solution_l17_17366


namespace a_gt_abs_b_suff_not_necc_l17_17498

theorem a_gt_abs_b_suff_not_necc (a b : ℝ) (h : a > |b|) : 
  a^2 > b^2 ∧ ∀ a b : ℝ, (a^2 > b^2 → |a| > |b|) → ¬ (a < -|b|) := 
by
  sorry

end a_gt_abs_b_suff_not_necc_l17_17498


namespace sin_cos_sum_l17_17733

theorem sin_cos_sum (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π)
  (h : Real.tan (θ + Real.pi / 4) = 1 / 7) : Real.sin θ + Real.cos θ = -1 / 5 := 
by
  sorry

end sin_cos_sum_l17_17733


namespace ninth_term_of_geometric_sequence_l17_17384

theorem ninth_term_of_geometric_sequence :
  let a1 := (5 : ℚ)
  let r := (3 / 4 : ℚ)
  (a1 * r^8) = (32805 / 65536 : ℚ) :=
by {
  sorry
}

end ninth_term_of_geometric_sequence_l17_17384


namespace quadratic_roots_correct_l17_17244

theorem quadratic_roots_correct (x : ℝ) : (x^2 = 2 * x) ↔ (x = 0 ∨ x = 2) := 
by
  sorry

end quadratic_roots_correct_l17_17244


namespace solve_abs_eq_linear_l17_17858

theorem solve_abs_eq_linear (x : ℝ) (h : |2 * x - 4| = x + 3) : x = 7 :=
sorry

end solve_abs_eq_linear_l17_17858


namespace factorization_m_minus_n_l17_17967

theorem factorization_m_minus_n :
  ∃ (m n : ℤ), (6 * (x:ℝ)^2 - 5 * x - 6 = (6 * x + m) * (x + n)) ∧ (m - n = 5) :=
by {
  sorry
}

end factorization_m_minus_n_l17_17967


namespace pyramid_volume_l17_17586

theorem pyramid_volume
  (FB AC FA FC AB BC : ℝ)
  (hFB : FB = 12)
  (hAC : AC = 4)
  (hFA : FA = 7)
  (hFC : FC = 7)
  (hAB : AB = 7)
  (hBC : BC = 7) :
  (1/3 * AC * (1/2 * FB * 3)) = 24 := by sorry

end pyramid_volume_l17_17586


namespace probability_of_log2N_is_integer_and_N_is_even_l17_17420

-- Defining the range of N as a four-digit number in base four
def is_base4_four_digit (N : ℕ) : Prop := 64 ≤ N ∧ N ≤ 255

-- Defining the condition that log_2 N is an integer
def is_power_of_two (N : ℕ) : Prop := ∃ k : ℕ, N = 2^k

-- Defining the condition that N is even
def is_even (N : ℕ) : Prop := N % 2 = 0

-- Combining all conditions
def meets_conditions (N : ℕ) : Prop := is_base4_four_digit N ∧ is_power_of_two N ∧ is_even N

-- Total number of four-digit numbers in base four
def total_base4_four_digits : ℕ := 192

-- Set of N values that meet the conditions
def valid_N_values : Finset ℕ := {64, 128}

-- The probability calculation
def calculated_probability : ℚ := valid_N_values.card / total_base4_four_digits

-- The final proof statement
theorem probability_of_log2N_is_integer_and_N_is_even : calculated_probability = 1 / 96 :=
by
  -- Prove the equality here (matching the solution given)
  sorry

end probability_of_log2N_is_integer_and_N_is_even_l17_17420


namespace total_cost_price_is_correct_l17_17676

noncomputable def selling_price_before_discount (sp_after_discount : ℝ) (discount_rate : ℝ) : ℝ :=
  sp_after_discount / (1 - discount_rate)

noncomputable def cost_price_from_profit (selling_price : ℝ) (profit_rate : ℝ) : ℝ :=
  selling_price / (1 + profit_rate)

noncomputable def cost_price_from_loss (selling_price : ℝ) (loss_rate : ℝ) : ℝ :=
  selling_price / (1 - loss_rate)

noncomputable def total_cost_price : ℝ :=
  let CP1 := cost_price_from_profit (selling_price_before_discount 600 0.05) 0.25
  let CP2 := cost_price_from_loss 800 0.20
  let CP3 := cost_price_from_profit 1000 0.30 - 50
  CP1 + CP2 + CP3

theorem total_cost_price_is_correct : total_cost_price = 2224.49 :=
  by
  sorry

end total_cost_price_is_correct_l17_17676


namespace proof_l17_17231

-- Define the conditions in Lean
variable {f : ℝ → ℝ}
variable (h1 : ∀ x ∈ (Set.Ioi 0), 0 ≤ f x)
variable (h2 : ∀ x ∈ (Set.Ioi 0), x * f x + f x ≤ 0)

-- Formulate the goal
theorem proof (a b : ℝ) (ha : a ∈ (Set.Ioi 0)) (hb : b ∈ (Set.Ioi 0)) (h : a < b) : 
    b * f a ≤ a * f b :=
by
  sorry  -- Proof omitted

end proof_l17_17231


namespace simplify_expression_l17_17888

theorem simplify_expression :
  ((4 * 7) / (12 * 14)) * ((9 * 12 * 14) / (4 * 7 * 9)) ^ 2 = 1 := 
by
  sorry

end simplify_expression_l17_17888


namespace find_prices_max_sets_of_go_compare_options_l17_17034

theorem find_prices (x y : ℕ) (h1 : 2 * x + 3 * y = 140) (h2 : 4 * x + y = 130) :
  x = 25 ∧ y = 30 :=
by sorry

theorem max_sets_of_go (m : ℕ) (h3 : 25 * (80 - m) + 30 * m ≤ 2250) :
  m ≤ 50 :=
by sorry

theorem compare_options (a : ℕ) :
  (a < 10 → 27 * a < 21 * a + 60) ∧ (a = 10 → 27 * a = 21 * a + 60) ∧ (a > 10 → 27 * a > 21 * a + 60) :=
by sorry

end find_prices_max_sets_of_go_compare_options_l17_17034


namespace arithmetic_sequence_problem_l17_17056

variable (a : ℕ → ℕ)
variable (d : ℕ) -- common difference for the arithmetic sequence
variable (h1 : ∀ n : ℕ, a (n + 1) = a n + d)
variable (h2 : a 1 - a 9 + a 17 = 7)

theorem arithmetic_sequence_problem : a 3 + a 15 = 14 := by
  sorry

end arithmetic_sequence_problem_l17_17056


namespace jake_spent_more_l17_17541

def cost_of_balloons (helium_count : ℕ) (foil_count : ℕ) (helium_price : ℝ) (foil_price : ℝ) : ℝ :=
  helium_count * helium_price + foil_count * foil_price

theorem jake_spent_more 
  (allan_helium : ℕ) (allan_foil : ℕ) (jake_helium : ℕ) (jake_foil : ℕ)
  (helium_price : ℝ) (foil_price : ℝ)
  (h_allan_helium : allan_helium = 2) (h_allan_foil : allan_foil = 3) 
  (h_jake_helium : jake_helium = 4) (h_jake_foil : jake_foil = 2)
  (h_helium_price : helium_price = 1.5) (h_foil_price : foil_price = 2.5) :
  cost_of_balloons jake_helium jake_foil helium_price foil_price - 
  cost_of_balloons allan_helium allan_foil helium_price foil_price = 0.5 := 
by
  sorry

end jake_spent_more_l17_17541


namespace cos_squared_plus_twice_sin_double_alpha_l17_17685

theorem cos_squared_plus_twice_sin_double_alpha (α : ℝ) (h : Real.tan α = 3 / 4) :
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 :=
by
  sorry

end cos_squared_plus_twice_sin_double_alpha_l17_17685


namespace possible_rectangle_areas_l17_17530

def is_valid_pair (a b : ℕ) := 
  a + b = 12 ∧ a > 0 ∧ b > 0

def rectangle_area (a b : ℕ) := a * b

theorem possible_rectangle_areas :
  {area | ∃ (a b : ℕ), is_valid_pair a b ∧ area = rectangle_area a b} 
  = {11, 20, 27, 32, 35, 36} := 
by 
  sorry

end possible_rectangle_areas_l17_17530


namespace biff_hourly_earnings_l17_17083

theorem biff_hourly_earnings:
  let ticket_cost := 11
  let drinks_snacks_cost := 3
  let headphones_cost := 16
  let wifi_cost_per_hour := 2
  let bus_ride_hours := 3
  let total_non_wifi_expenses := ticket_cost + drinks_snacks_cost + headphones_cost
  let total_wifi_cost := bus_ride_hours * wifi_cost_per_hour
  let total_expenses := total_non_wifi_expenses + total_wifi_cost
  ∀ (x : ℝ), 3 * x = total_expenses → x = 12 :=
by sorry -- Proof skipped

end biff_hourly_earnings_l17_17083


namespace radius_of_inscribed_circle_in_rhombus_l17_17068

noncomputable def radius_of_inscribed_circle (d₁ d₂ : ℕ) : ℝ :=
  (d₁ * d₂) / (2 * Real.sqrt ((d₁ / 2) ^ 2 + (d₂ / 2) ^ 2))

theorem radius_of_inscribed_circle_in_rhombus :
  radius_of_inscribed_circle 8 18 = 36 / Real.sqrt 97 :=
by
  -- Skip the detailed proof steps
  sorry

end radius_of_inscribed_circle_in_rhombus_l17_17068


namespace problem_solution_l17_17081

open Real

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) :
  (∃ (C₁ : ℝ), (2 : ℝ)^x + (4 : ℝ)^y = C₁ ∧ C₁ = 2 * sqrt 2) ∧
  (∃ (C₂ : ℝ), 1 / x + 2 / y = C₂ ∧ C₂ = 9) ∧
  (∃ (C₃ : ℝ), x^2 + 4 * y^2 = C₃ ∧ C₃ = 1 / 2) :=
by
  sorry

end problem_solution_l17_17081


namespace other_root_of_quadratic_l17_17260

theorem other_root_of_quadratic (a b c : ℚ) (x₁ x₂ : ℚ) :
  a ≠ 0 →
  x₁ = 4 / 9 →
  (a * x₁^2 + b * x₁ + c = 0) →
  (a = 81) →
  (b = -145) →
  (c = 64) →
  x₂ = -16 / 9
:=
sorry

end other_root_of_quadratic_l17_17260


namespace find_number_l17_17690

theorem find_number 
  (n : ℤ)
  (h1 : n % 7 = 2)
  (h2 : n % 8 = 4)
  (quot_7 : ℤ)
  (quot_8 : ℤ)
  (h3 : n = 7 * quot_7 + 2)
  (h4 : n = 8 * quot_8 + 4)
  (h5 : quot_7 = quot_8 + 7) :
  n = 380 := by
  sorry

end find_number_l17_17690


namespace unique_number_not_in_range_of_g_l17_17529

noncomputable def g (x : ℝ) (a b c d : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_number_not_in_range_of_g 
  (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : g 5 a b c d = 5) (h6 : g 25 a b c d = 25) 
  (h7 : ∀ x, x ≠ -d/c → g (g x a b c d) a b c d = x) :
  ∃ r, r = 15 ∧ ∀ y, g y a b c d ≠ r := 
by
  sorry

end unique_number_not_in_range_of_g_l17_17529


namespace minimum_k_exists_l17_17958

theorem minimum_k_exists :
  ∀ (s : Finset ℝ), s.card = 3 → (∀ (a b : ℝ), a ∈ s → b ∈ s → (|a - b| ≤ (1.5 : ℝ) ∨ |(1 / a) - (1 / b)| ≤ 1.5)) :=
by
  sorry

end minimum_k_exists_l17_17958


namespace wool_usage_l17_17398

def total_balls_of_wool_used (scarves_aaron sweaters_aaron sweaters_enid : ℕ) (wool_per_scarf wool_per_sweater : ℕ) : ℕ :=
  (scarves_aaron * wool_per_scarf) + (sweaters_aaron * wool_per_sweater) + (sweaters_enid * wool_per_sweater)

theorem wool_usage :
  total_balls_of_wool_used 10 5 8 3 4 = 82 :=
by
  -- calculations done in solution steps
  -- total_balls_of_wool_used (10 scarves * 3 balls/scarf) + (5 sweaters * 4 balls/sweater) + (8 sweaters * 4 balls/sweater)
  -- total_balls_of_wool_used (30) + (20) + (32)
  -- total_balls_of_wool_used = 30 + 20 + 32 = 82
  sorry

end wool_usage_l17_17398


namespace necessary_but_not_sufficient_condition_l17_17734

variable (A B C : Set α) (a : α)
variable [Nonempty α]
variable (H1 : ∀ a, a ∈ A ↔ (a ∈ B ∧ a ∈ C))

theorem necessary_but_not_sufficient_condition :
  (a ∈ B → a ∈ A) ∧ ¬(a ∈ A → a ∈ B) :=
by
  sorry

end necessary_but_not_sufficient_condition_l17_17734


namespace solve_for_x_l17_17971

-- Define the custom operation for real numbers
def custom_op (a b c d : ℝ) : ℝ := a * c - b * d

-- The theorem to prove
theorem solve_for_x (x : ℝ) (h : custom_op (-x) 3 (x - 2) (-6) = 10) :
  x = 4 ∨ x = -2 :=
sorry

end solve_for_x_l17_17971


namespace arithmetic_seq_sum_mul_3_l17_17636

-- Definition of the arithmetic sequence
def arithmetic_sequence := [101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121]

-- Prove that 3 times the sum of the arithmetic sequence is 3663
theorem arithmetic_seq_sum_mul_3 : 
  3 * (arithmetic_sequence.sum) = 3663 :=
by
  sorry

end arithmetic_seq_sum_mul_3_l17_17636


namespace inequality_xyz_equality_condition_l17_17725

theorem inequality_xyz (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : 
  x + y + z ≤ 2 + x * y * z :=
sorry

theorem equality_condition (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) :
  (x + y + z = 2 + x * y * z) ↔ (x = 0 ∧ y = 1 ∧ z = 1) ∨ (x = 1 ∧ y = 0 ∧ z = 1) ∨ (x = 1 ∧ y = 1 ∧ z = 0) ∨
                                                  (x = 0 ∧ y = -1 ∧ z = -1) ∨ (x = -1 ∧ y = 0 ∧ z = 1) ∨
                                                  (x = -1 ∧ y = 1 ∧ z = 0) :=
sorry

end inequality_xyz_equality_condition_l17_17725


namespace largest_angle_of_triangle_l17_17947

theorem largest_angle_of_triangle (x : ℝ) (h_ratio : (5 * x) + (6 * x) + (7 * x) = 180) :
  7 * x = 70 := 
sorry

end largest_angle_of_triangle_l17_17947


namespace range_of_a_l17_17601

theorem range_of_a (a : ℝ) (x : ℝ) (h : x^2 + a * x + 1 < 0) : a < -2 ∨ a > 2 :=
sorry

end range_of_a_l17_17601


namespace average_difference_correct_l17_17483

def daily_diff : List ℤ := [15, 0, -15, 25, 5, -5, 10]
def number_of_days : ℤ := 7

theorem average_difference_correct :
  (daily_diff.sum : ℤ) / number_of_days = 5 := by
  sorry

end average_difference_correct_l17_17483


namespace sum_of_midpoints_double_l17_17638

theorem sum_of_midpoints_double (a b c : ℝ) (h : a + b + c = 15) : 
  (a + b) + (a + c) + (b + c) = 30 :=
by
  -- We skip the proof according to the instruction
  sorry

end sum_of_midpoints_double_l17_17638


namespace last_digit_of_exponents_l17_17436

theorem last_digit_of_exponents : 
  (∃k, 2011 = 4 * k + 3 ∧ 
         (2^2011 % 10 = 8) ∧ 
         (3^2011 % 10 = 7)) → 
  ((2^2011 + 3^2011) % 10 = 5) := 
by 
  sorry

end last_digit_of_exponents_l17_17436


namespace exceeds_500_bacteria_l17_17246

noncomputable def bacteria_count (n : Nat) : Nat :=
  4 * 3^n

theorem exceeds_500_bacteria (n : Nat) (h : 4 * 3^n > 500) : n ≥ 6 :=
by
  sorry

end exceeds_500_bacteria_l17_17246


namespace orchestra_ticket_cost_l17_17426

noncomputable def cost_balcony : ℝ := 8  -- cost of balcony tickets
noncomputable def total_sold : ℝ := 340  -- total tickets sold
noncomputable def total_revenue : ℝ := 3320  -- total revenue
noncomputable def extra_balcony : ℝ := 40  -- extra tickets sold for balcony than orchestra

theorem orchestra_ticket_cost (x y : ℝ) (h1 : x + extra_balcony = total_sold)
    (h2 : y = x + extra_balcony) (h3 : x + y = total_sold)
    (h4 : x + cost_balcony * y = total_revenue) : 
    cost_balcony = 8 → x = 12 :=
by
  sorry

end orchestra_ticket_cost_l17_17426


namespace parabola_focus_coordinates_l17_17657

theorem parabola_focus_coordinates :
  ∀ (x y : ℝ), x^2 = 8 * y → ∃ F : ℝ × ℝ, F = (0, 2) :=
  sorry

end parabola_focus_coordinates_l17_17657


namespace part1_purchase_price_part2_minimum_A_l17_17130

section
variables (x y m : ℝ)

-- Part 1: Purchase price per piece
theorem part1_purchase_price (h1 : 10 * x + 15 * y = 3600) (h2 : 25 * x + 30 * y = 8100) :
  x = 180 ∧ y = 120 :=
sorry

-- Part 2: Minimum number of model A bamboo mats
theorem part2_minimum_A (h3 : x = 180) (h4 : y = 120) 
    (h5 : (260 - x) * m + (180 - y) * (60 - m) ≥ 4400) : 
  m ≥ 40 :=
sorry
end

end part1_purchase_price_part2_minimum_A_l17_17130


namespace convert_sq_meters_to_hectares_convert_hours_to_hours_and_minutes_l17_17645

theorem convert_sq_meters_to_hectares :
  (123000 / 10000) = 12.3 :=
by
  sorry

theorem convert_hours_to_hours_and_minutes :
  (4 + 0.25 * 60) = 4 * 60 + 15 :=
by
  sorry

end convert_sq_meters_to_hectares_convert_hours_to_hours_and_minutes_l17_17645


namespace mariela_cards_l17_17102

theorem mariela_cards (cards_after_home : ℕ) (total_cards : ℕ) (cards_in_hospital : ℕ) : 
  cards_after_home = 287 → 
  total_cards = 690 → 
  cards_in_hospital = total_cards - cards_after_home → 
  cards_in_hospital = 403 := 
by 
  intros h1 h2 h3 
  rw [h1, h2] at h3 
  exact h3


end mariela_cards_l17_17102


namespace geometric_sequence_log_sum_l17_17894

noncomputable def log_base_three (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem geometric_sequence_log_sum (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∃ r, ∀ n, a (n + 1) = a n * r)
  (h3 : a 6 * a 7 = 9) :
  log_base_three (a 1) + log_base_three (a 2) + log_base_three (a 3) +
  log_base_three (a 4) + log_base_three (a 5) + log_base_three (a 6) +
  log_base_three (a 7) + log_base_three (a 8) + log_base_three (a 9) +
  log_base_three (a 10) + log_base_three (a 11) + log_base_three (a 12) = 12 :=
  sorry

end geometric_sequence_log_sum_l17_17894


namespace solve_for_c_l17_17641

theorem solve_for_c (a b c : ℝ) (B : ℝ) (ha : a = 4) (hb : b = 2*Real.sqrt 7) (hB : B = Real.pi / 3) : 
  (c^2 - 4*c - 12 = 0) → c = 6 :=
by 
  intro h
  -- Details of the proof would be here
  sorry

end solve_for_c_l17_17641


namespace expression_constant_value_l17_17619

theorem expression_constant_value (a b x y : ℝ) 
  (h_a : a = Real.sqrt (1 + x^2))
  (h_b : b = Real.sqrt (1 + y^2)) 
  (h_xy : x + y = 1) : 
  (a + b + 1) * (a + b - 1) * (a - b + 1) * (-a + b + 1) = 4 := 
by 
  sorry

end expression_constant_value_l17_17619


namespace original_inhabitants_l17_17181

theorem original_inhabitants (X : ℝ) (h : 0.75 * 0.9 * X = 5265) : X = 7800 :=
by
  sorry

end original_inhabitants_l17_17181


namespace find_constant_term_l17_17609

theorem find_constant_term (x y C : ℤ) 
    (h1 : 5 * x + y = 19) 
    (h2 : 3 * x + 2 * y = 10) 
    (h3 : C = x + 3 * y) 
    : C = 1 := 
by 
  sorry

end find_constant_term_l17_17609


namespace ratio_of_squares_l17_17379

theorem ratio_of_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a / b = 1 / 3) :
  (4 * a / (4 * b) = 1 / 3) ∧ (a * a / (b * b) = 1 / 9) :=
by
  sorry

end ratio_of_squares_l17_17379


namespace total_goals_scored_l17_17313

theorem total_goals_scored (g1 t1 g2 t2 : ℕ)
  (h1 : g1 = 2)
  (h2 : g1 = t1 - 3)
  (h3 : t2 = 6)
  (h4 : g2 = t2 - 2) :
  g1 + t1 + g2 + t2 = 17 :=
by
  sorry

end total_goals_scored_l17_17313


namespace hyperbola_asymptotes_l17_17990

-- Define the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 / 4 = 1

-- Define the equations for the asymptotes
def asymptote_pos (x y : ℝ) : Prop := y = 2 * x
def asymptote_neg (x y : ℝ) : Prop := y = -2 * x

-- State the theorem
theorem hyperbola_asymptotes (x y : ℝ) :
  hyperbola_eq x y → (asymptote_pos x y ∨ asymptote_neg x y) := 
by
  sorry

end hyperbola_asymptotes_l17_17990


namespace sum_of_interior_angles_of_pentagon_l17_17630

theorem sum_of_interior_angles_of_pentagon : (5 - 2) * 180 = 540 := 
by
  -- We skip the proof as per instruction
  sorry

end sum_of_interior_angles_of_pentagon_l17_17630


namespace tank_capacity_l17_17525

variable (x : ℝ) -- Total capacity of the tank

theorem tank_capacity (h1 : x / 8 = 120 / (1 / 2 - 1 / 8)) :
  x = 320 :=
by
  sorry

end tank_capacity_l17_17525


namespace fraction_of_orange_juice_in_mixture_l17_17516

theorem fraction_of_orange_juice_in_mixture
  (capacity_pitcher : ℕ)
  (fraction_first_pitcher : ℚ)
  (fraction_second_pitcher : ℚ)
  (condition1 : capacity_pitcher = 500)
  (condition2 : fraction_first_pitcher = 1/4)
  (condition3 : fraction_second_pitcher = 3/7) :
  (125 + 500 * (3/7)) / (2 * 500) = 95 / 280 :=
by
  sorry

end fraction_of_orange_juice_in_mixture_l17_17516
