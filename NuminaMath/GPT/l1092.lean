import Mathlib

namespace find_angle_D_l1092_109255

-- Define the given angles and conditions
def angleA := 30
def angleB (D : ℝ) := 2 * D
def angleC (D : ℝ) := D + 40
def sum_of_angles (A B C D : ℝ) := A + B + C + D = 360

theorem find_angle_D (D : ℝ) (hA : angleA = 30) (hB : angleB D = 2 * D) (hC : angleC D = D + 40) (hSum : sum_of_angles angleA (angleB D) (angleC D) D):
  D = 72.5 :=
by
  -- Proof is omitted
  sorry

end find_angle_D_l1092_109255


namespace a_range_l1092_109203

noncomputable def f (x a : ℝ) : ℝ := |2 * x - 1| + |x - 2 * a|

def valid_a_range (a : ℝ) : Prop :=
∀ x, 1 ≤ x ∧ x ≤ 2 → f x a ≤ 4

theorem a_range (a : ℝ) : valid_a_range a → (1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ) := 
sorry

end a_range_l1092_109203


namespace find_distance_to_place_l1092_109248

noncomputable def distance_to_place (speed_boat : ℝ) (speed_stream : ℝ) (total_time : ℝ) : ℝ :=
  let downstream_speed := speed_boat + speed_stream
  let upstream_speed := speed_boat - speed_stream
  let distance := (total_time * (downstream_speed * upstream_speed)) / (downstream_speed + upstream_speed)
  distance

theorem find_distance_to_place :
  distance_to_place 16 2 937.1428571428571 = 7392.92 :=
by
  sorry

end find_distance_to_place_l1092_109248


namespace polynomial_solution_l1092_109235

noncomputable def is_solution (P : ℝ → ℝ) : Prop :=
  (P 0 = 0) ∧ (∀ x : ℝ, P x = (P (x + 1) + P (x - 1)) / 2)

theorem polynomial_solution (P : ℝ → ℝ) : is_solution P → ∃ a : ℝ, ∀ x : ℝ, P x = a * x :=
  sorry

end polynomial_solution_l1092_109235


namespace width_of_plot_is_60_l1092_109247

-- Defining the conditions
def length_of_plot := 90
def distance_between_poles := 5
def number_of_poles := 60

-- The theorem statement
theorem width_of_plot_is_60 :
  ∃ width : ℕ, 2 * (length_of_plot + width) = number_of_poles * distance_between_poles ∧ width = 60 :=
sorry

end width_of_plot_is_60_l1092_109247


namespace inequality_of_f_on_angles_l1092_109296

noncomputable def f : ℝ → ℝ := sorry -- Define f as a noncomputable function

-- Stating the properties of the function f
axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodic_function : ∀ x : ℝ, f (x + 1) = -f x
axiom decreasing_interval : ∀ x y : ℝ, (-3 ≤ x ∧ x < y ∧ y ≤ -2) → f x > f y

-- Stating the properties of the angles α and β
variables (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) (hαβ : α ≠ β)

-- The proof statement we want to prove
theorem inequality_of_f_on_angles : f (Real.sin α) > f (Real.cos β) :=
sorry -- The proof is omitted

end inequality_of_f_on_angles_l1092_109296


namespace tickets_difference_l1092_109227

-- Definitions of conditions
def tickets_won : Nat := 19
def tickets_for_toys : Nat := 12
def tickets_for_clothes : Nat := 7

-- Theorem statement: Prove that the difference between tickets used for toys and tickets used for clothes is 5
theorem tickets_difference : (tickets_for_toys - tickets_for_clothes = 5) := by
  sorry

end tickets_difference_l1092_109227


namespace solve_for_a_l1092_109208

theorem solve_for_a
  (a x : ℚ)
  (h1 : (2 * a * x + 3) / (a - x) = 3 / 4)
  (h2 : x = 1) : a = -3 :=
by
  sorry

end solve_for_a_l1092_109208


namespace polynomial_roots_l1092_109215

theorem polynomial_roots:
  ∀ x : ℝ, (x^2 - 5*x + 6) * (x - 3) * (2*x - 8) = 0 ↔ x = 2 ∨ x = 3 ∨ x = 4 :=
by
  sorry

end polynomial_roots_l1092_109215


namespace reciprocal_sum_l1092_109282

theorem reciprocal_sum (a b c d : ℚ) (h1 : a = 2) (h2 : b = 5) (h3 : c = 3) (h4 : d = 4) : 
  (a / b + c / d)⁻¹ = (20 : ℚ) / 23 := 
by
  sorry

end reciprocal_sum_l1092_109282


namespace max_sum_a_b_l1092_109260

theorem max_sum_a_b (a b : ℝ) (h : a^2 - a*b + b^2 = 1) : a + b ≤ 2 := 
by sorry

end max_sum_a_b_l1092_109260


namespace only_one_true_l1092_109281

-- Definitions based on conditions
def line := Type
def plane := Type
def parallel (m n : line) : Prop := sorry
def perpendicular (m n : line) : Prop := sorry
def subset (m : line) (alpha : plane) : Prop := sorry

-- Propositions derived from conditions
def prop1 (m n : line) (alpha : plane) : Prop := parallel m alpha ∧ parallel n alpha → ¬ parallel m n
def prop2 (m n : line) (alpha : plane) : Prop := perpendicular m alpha ∧ perpendicular n alpha → parallel m n
def prop3 (m n : line) (alpha beta : plane) : Prop := parallel alpha beta ∧ subset m alpha ∧ subset n beta → parallel m n
def prop4 (m n : line) (alpha beta : plane) : Prop := perpendicular alpha beta ∧ perpendicular m n ∧ perpendicular m alpha → perpendicular n beta

-- Theorem statement that only one proposition is true
theorem only_one_true (m n : line) (alpha beta : plane) :
  (prop1 m n alpha = false) ∧
  (prop2 m n alpha = true) ∧
  (prop3 m n alpha beta = false) ∧
  (prop4 m n alpha beta = false) :=
by sorry

end only_one_true_l1092_109281


namespace barbi_weight_loss_duration_l1092_109278

theorem barbi_weight_loss_duration :
  (∃ x : ℝ, 
    (∃ l_barbi l_luca : ℝ, 
      l_barbi = 1.5 * x ∧ 
      l_luca = 99 ∧ 
      l_luca = l_barbi + 81) ∧
    x = 12) :=
by
  sorry

end barbi_weight_loss_duration_l1092_109278


namespace question_l1092_109207

def N : ℕ := 100101102 -- N should be defined properly but is simplified here for illustration.

theorem question (k : ℕ) (h : N = 100101102502499500) : (3^3 ∣ N) ∧ ¬(3^4 ∣ N) :=
sorry

end question_l1092_109207


namespace percentage_increase_l1092_109273

theorem percentage_increase (D1 D2 : ℕ) (total_days : ℕ) (H1 : D1 = 4) (H2 : total_days = 9) (H3 : D1 + D2 = total_days) : 
  (D2 - D1) / D1 * 100 = 25 := 
sorry

end percentage_increase_l1092_109273


namespace half_of_expression_correct_l1092_109238

theorem half_of_expression_correct :
  (2^12 + 3 * 2^10) / 2 = 2^9 * 7 :=
by
  sorry

end half_of_expression_correct_l1092_109238


namespace cos_beta_value_l1092_109298

open Real

theorem cos_beta_value (α β : ℝ) (h1 : sin α = sqrt 5 / 5) (h2 : sin (α - β) = - sqrt 10 / 10) (h3 : 0 < α ∧ α < π / 2) (h4 : 0 < β ∧ β < π / 2) : cos β = sqrt 2 / 2 :=
by
sorry

end cos_beta_value_l1092_109298


namespace square_floor_tiling_total_number_of_tiles_l1092_109244

theorem square_floor_tiling (s : ℕ) (h : (2 * s - 1 : ℝ) / (s ^ 2 : ℝ) = 0.41) : s = 4 :=
by
  sorry

theorem total_number_of_tiles : 4^2 = 16 := 
by
  norm_num

end square_floor_tiling_total_number_of_tiles_l1092_109244


namespace distance_between_skew_lines_l1092_109251

-- Definitions for the geometric configuration
def AB : ℝ := 4
def AA1 : ℝ := 4
def AD : ℝ := 3

-- Theorem statement to prove the distance between skew lines A1D and B1D1
theorem distance_between_skew_lines:
  ∃ d : ℝ, d = (6 * Real.sqrt 34) / 17 :=
sorry

end distance_between_skew_lines_l1092_109251


namespace polygon_sides_l1092_109286

theorem polygon_sides {n k : ℕ} (h1 : k = n * (n - 3) / 2) (h2 : k = 3 * n / 2) : n = 6 :=
by
  sorry

end polygon_sides_l1092_109286


namespace distance_sum_l1092_109283

theorem distance_sum (a : ℝ) (x y : ℝ) 
  (AB CD : ℝ) (A B C D P Q M N : ℝ)
  (h_AB : AB = 4) (h_CD : CD = 8) 
  (h_M_AB : M = (A + B) / 2) (h_N_CD : N = (C + D) / 2)
  (h_P_AB : P ∈ [A, B]) (h_Q_CD : Q ∈ [C, D])
  (h_x : x = dist P M) (h_y : y = dist Q N)
  (h_y_eq_2x : y = 2 * x) (h_x_eq_a : x = a) :
  x + y = 3 * a := 
by
  sorry

end distance_sum_l1092_109283


namespace trail_mix_total_weight_l1092_109211

def peanuts : ℝ := 0.17
def chocolate_chips : ℝ := 0.17
def raisins : ℝ := 0.08

theorem trail_mix_total_weight :
  peanuts + chocolate_chips + raisins = 0.42 :=
by
  -- The proof would go here
  sorry

end trail_mix_total_weight_l1092_109211


namespace cousin_typing_time_l1092_109284

theorem cousin_typing_time (speed_ratio : ℕ) (my_time_hours : ℕ) (minutes_per_hour : ℕ) (my_time_minutes : ℕ) :
  speed_ratio = 4 →
  my_time_hours = 3 →
  minutes_per_hour = 60 →
  my_time_minutes = my_time_hours * minutes_per_hour →
  ∃ (cousin_time : ℕ), cousin_time = my_time_minutes / speed_ratio := by
  sorry

end cousin_typing_time_l1092_109284


namespace number_of_juniors_l1092_109210

variable (J S x y : ℕ)

-- Conditions given in the problem
axiom total_students : J + S = 40
axiom junior_debate_team : 3 * J / 10 = x
axiom senior_debate_team : S / 5 = y
axiom equal_debate_team : x = y

-- The theorem to prove 
theorem number_of_juniors : J = 16 :=
by
  sorry

end number_of_juniors_l1092_109210


namespace rahim_books_bought_l1092_109265

theorem rahim_books_bought (x : ℕ) 
  (first_shop_cost second_shop_cost total_books : ℕ)
  (avg_price total_spent : ℕ)
  (h1 : first_shop_cost = 1500)
  (h2 : second_shop_cost = 340)
  (h3 : total_books = x + 60)
  (h4 : avg_price = 16)
  (h5 : total_spent = first_shop_cost + second_shop_cost)
  (h6 : avg_price = total_spent / total_books) :
  x = 55 :=
by
  sorry

end rahim_books_bought_l1092_109265


namespace initial_elephants_count_l1092_109224

def exodus_rate : ℕ := 2880
def exodus_time : ℕ := 4
def entrance_rate : ℕ := 1500
def entrance_time : ℕ := 7
def final_elephants : ℕ := 28980

theorem initial_elephants_count :
  final_elephants - (exodus_rate * exodus_time) + (entrance_rate * entrance_time) = 27960 := by
  sorry

end initial_elephants_count_l1092_109224


namespace isosceles_triangle_l1092_109237

theorem isosceles_triangle {a b R : ℝ} {α β : ℝ} 
  (h : a * Real.tan α + b * Real.tan β = (a + b) * Real.tan ((α + β) / 2))
  (ha : a = 2 * R * Real.sin α) (hb : b = 2 * R * Real.sin β) :
  α = β := 
sorry

end isosceles_triangle_l1092_109237


namespace min_contribution_l1092_109217

theorem min_contribution (x : ℝ) (h1 : 0 < x) (h2 : 10 * x = 20) (h3 : ∀ p, p ≠ 1 → p ≠ 2 → p ≠ 3 → p ≠ 4 → p ≠ 5 → p ≠ 6 → p ≠ 7 → p ≠ 8 → p ≠ 9 → p ≠ 10 → p ≤ 11) : 
  x = 2 := sorry

end min_contribution_l1092_109217


namespace distinct_natural_primes_l1092_109299

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem distinct_natural_primes :
  ∃ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 5 ∧
  is_prime (a * b + c * d) ∧
  is_prime (a * c + b * d) ∧
  is_prime (a * d + b * c) := by
  sorry

end distinct_natural_primes_l1092_109299


namespace exists_unique_integer_pair_l1092_109221

theorem exists_unique_integer_pair (a : ℕ) (ha : 0 < a) :
  ∃! (x y : ℕ), 0 < x ∧ 0 < y ∧ x + (x + y - 1) * (x + y - 2) / 2 = a :=
by
  sorry

end exists_unique_integer_pair_l1092_109221


namespace correct_transformation_l1092_109216

theorem correct_transformation (x : ℝ) : (x^2 - 10 * x - 1 = 0) → ((x - 5) ^ 2 = 26) := by
  sorry

end correct_transformation_l1092_109216


namespace find_number_l1092_109270

theorem find_number (N : ℕ) (h : N / 16 = 16 * 8) : N = 2048 :=
sorry

end find_number_l1092_109270


namespace directrix_of_parabola_l1092_109202

noncomputable def parabola_directrix (y : ℝ) (x : ℝ) : Prop :=
  y = 4 * x^2

theorem directrix_of_parabola : ∃ d : ℝ, (parabola_directrix (y := 4) (x := x) → d = -1/16) :=
by
  sorry

end directrix_of_parabola_l1092_109202


namespace work_problem_l1092_109225

theorem work_problem (A B : ℝ) (hA : A = 1/4) (hB : B = 1/12) :
  (2 * (A + B) + 4 * B = 1) :=
by
  -- Work rate of A and B together
  -- Work done in 2 days by both
  -- Remaining work and time taken by B alone
  -- Final Result
  sorry

end work_problem_l1092_109225


namespace range_of_m_l1092_109233

noncomputable def f (x m : ℝ) := Real.exp x * (Real.log x + (1 / 2) * x ^ 2 - m * x)

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 0 < x → ((Real.exp x * ((1 / x) + x - m)) > 0)) → m < 2 := by
  sorry

end range_of_m_l1092_109233


namespace range_of_m_l1092_109250

theorem range_of_m (m : ℝ) :
  (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ↔ 7 < m ∧ m < 24 :=
sorry

end range_of_m_l1092_109250


namespace ratio_of_periods_l1092_109249

variable (I_B T_B : ℝ)
variable (I_A T_A : ℝ)
variable (Profit_A Profit_B TotalProfit : ℝ)
variable (k : ℝ)

-- Define the conditions
axiom h1 : I_A = 3 * I_B
axiom h2 : T_A = k * T_B
axiom h3 : Profit_B = 4500
axiom h4 : TotalProfit = 31500
axiom h5 : Profit_A = TotalProfit - Profit_B

-- The profit shares are proportional to the product of investment and time period
axiom h6 : Profit_A = I_A * T_A
axiom h7 : Profit_B = I_B * T_B

theorem ratio_of_periods : T_A / T_B = 2 := by
  sorry

end ratio_of_periods_l1092_109249


namespace minimum_gb_for_cheaper_plan_l1092_109232

theorem minimum_gb_for_cheaper_plan : ∃ g : ℕ, (g ≥ 778) ∧ 
  (∀ g' < 778, 3000 + (if g' ≤ 500 then 8 * g' else 8 * 500 + 6 * (g' - 500)) ≥ 15 * g') ∧ 
  3000 + (if g ≤ 500 then 8 * g else 8 * 500 + 6 * (g - 500)) < 15 * g :=
by
  sorry

end minimum_gb_for_cheaper_plan_l1092_109232


namespace number_of_girls_l1092_109266

theorem number_of_girls (total_children boys : ℕ) (h1 : total_children = 60) (h2 : boys = 16) : total_children - boys = 44 := by
  sorry

end number_of_girls_l1092_109266


namespace proof_problem_l1092_109218

variable {a b x : ℝ}

theorem proof_problem (h1 : x = b / a) (h2 : a ≠ b) (h3 : a ≠ 0) : 
  (2 * a + b) / (2 * a - b) = (2 + x) / (2 - x) :=
sorry

end proof_problem_l1092_109218


namespace number_of_tickets_bought_l1092_109269

noncomputable def ticketCost : ℕ := 5
noncomputable def popcornCost : ℕ := (80 * ticketCost) / 100
noncomputable def sodaCost : ℕ := (50 * popcornCost) / 100
noncomputable def totalSpent : ℕ := 36
noncomputable def numberOfPopcorns : ℕ := 2 
noncomputable def numberOfSodas : ℕ := 4

theorem number_of_tickets_bought : 
  (totalSpent - (numberOfPopcorns * popcornCost + numberOfSodas * sodaCost)) = 4 * ticketCost :=
by
  sorry

end number_of_tickets_bought_l1092_109269


namespace average_minutes_run_per_day_l1092_109231

theorem average_minutes_run_per_day (f : ℕ) :
  let third_grade_minutes := 12
  let fourth_grade_minutes := 15
  let fifth_grade_minutes := 10
  let third_graders := 4 * f
  let fourth_graders := 2 * f
  let fifth_graders := f
  let total_minutes := third_graders * third_grade_minutes + fourth_graders * fourth_grade_minutes + fifth_graders * fifth_grade_minutes
  let total_students := third_graders + fourth_graders + fifth_graders
  total_minutes / total_students = 88 / 7 :=
by
  sorry

end average_minutes_run_per_day_l1092_109231


namespace solve_for_x_l1092_109213

theorem solve_for_x : ∀ x : ℝ, ( (x * x^(2:ℝ)) ^ (1/6) )^2 = 4 → x = 4 := by
  intro x
  sorry

end solve_for_x_l1092_109213


namespace smallest_of_three_consecutive_odd_numbers_l1092_109275

theorem smallest_of_three_consecutive_odd_numbers (x : ℤ) (h : x + (x + 2) + (x + 4) = 69) : x = 21 :=
sorry

end smallest_of_three_consecutive_odd_numbers_l1092_109275


namespace angle_between_sides_of_triangle_l1092_109280

noncomputable def right_triangle_side_lengths1 : Nat × Nat × Nat := (15, 36, 39)
noncomputable def right_triangle_side_lengths2 : Nat × Nat × Nat := (40, 42, 58)

-- Assuming both triangles are right triangles
def is_right_triangle (a b c : Nat) : Prop := a^2 + b^2 = c^2

theorem angle_between_sides_of_triangle
  (h1 : is_right_triangle 15 36 39)
  (h2 : is_right_triangle 40 42 58) : 
  ∃ (θ : ℝ), θ = 90 :=
by
  sorry

end angle_between_sides_of_triangle_l1092_109280


namespace ellipse_foci_y_axis_l1092_109290

theorem ellipse_foci_y_axis (k : ℝ) :
  (∃ a b : ℝ, a = 15 - k ∧ b = k - 9 ∧ a > 0 ∧ b > 0) ↔ (12 < k ∧ k < 15) :=
by
  sorry

end ellipse_foci_y_axis_l1092_109290


namespace percentage_of_girls_after_changes_l1092_109276

theorem percentage_of_girls_after_changes :
  let boys_classA := 15
  let girls_classA := 20
  let boys_classB := 25
  let girls_classB := 35
  let boys_transferAtoB := 3
  let girls_transferAtoB := 2
  let boys_joiningA := 4
  let girls_joiningA := 6

  let boys_classA_after := boys_classA - boys_transferAtoB + boys_joiningA
  let girls_classA_after := girls_classA - girls_transferAtoB + girls_joiningA
  let boys_classB_after := boys_classB + boys_transferAtoB
  let girls_classB_after := girls_classB + girls_transferAtoB

  let total_students := boys_classA_after + girls_classA_after + boys_classB_after + girls_classB_after
  let total_girls := girls_classA_after + girls_classB_after 

  (total_girls / total_students : ℝ) * 100 = 58.095 := by
  sorry

end percentage_of_girls_after_changes_l1092_109276


namespace john_needs_more_money_l1092_109288

def total_needed : ℝ := 2.50
def current_amount : ℝ := 0.75
def remaining_amount : ℝ := 1.75

theorem john_needs_more_money : total_needed - current_amount = remaining_amount :=
by
  sorry

end john_needs_more_money_l1092_109288


namespace cube_surface_area_726_l1092_109293

noncomputable def cubeSurfaceArea (volume : ℝ) : ℝ :=
  let side := volume^(1 / 3)
  6 * (side ^ 2)

theorem cube_surface_area_726 (h : cubeSurfaceArea 1331 = 726) : cubeSurfaceArea 1331 = 726 :=
by
  sorry

end cube_surface_area_726_l1092_109293


namespace probability_of_not_adjacent_to_edge_is_16_over_25_l1092_109239

def total_squares : ℕ := 100
def perimeter_squares : ℕ := 36
def non_perimeter_squares : ℕ := total_squares - perimeter_squares
def probability_not_adjacent_to_edge : ℚ := non_perimeter_squares / total_squares

theorem probability_of_not_adjacent_to_edge_is_16_over_25 :
  probability_not_adjacent_to_edge = 16 / 25 := by
  sorry

end probability_of_not_adjacent_to_edge_is_16_over_25_l1092_109239


namespace find_9a_value_l1092_109292

theorem find_9a_value (a : ℚ) 
  (h : (4 - a) / (5 - a) = (4 / 5) ^ 2) : 9 * a = 20 :=
by
  sorry

end find_9a_value_l1092_109292


namespace price_of_tea_mixture_l1092_109204

noncomputable def price_of_mixture (price1 price2 price3 : ℝ) (ratio1 ratio2 ratio3 : ℝ) : ℝ :=
  (price1 * ratio1 + price2 * ratio2 + price3 * ratio3) / (ratio1 + ratio2 + ratio3)

theorem price_of_tea_mixture :
  price_of_mixture 126 135 175.5 1 1 2 = 153 := 
by
  sorry

end price_of_tea_mixture_l1092_109204


namespace original_deck_size_l1092_109258

/-- 
Aubrey adds 2 additional cards to a deck and then splits the deck evenly among herself and 
two other players, each player having 18 cards. 
We want to prove that the original number of cards in the deck was 52. 
-/
theorem original_deck_size :
  ∃ (n : ℕ), (n + 2) / 3 = 18 ∧ n = 52 :=
by
  sorry

end original_deck_size_l1092_109258


namespace tetrahedron_edges_sum_of_squares_l1092_109272

-- Given conditions
variables {a b c d e f x y z : ℝ}

-- Mathematical statement
theorem tetrahedron_edges_sum_of_squares :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 4 * (x^2 + y^2 + z^2) :=
sorry

end tetrahedron_edges_sum_of_squares_l1092_109272


namespace problem1_solution_problem2_solution_l1092_109271

-- Problem 1: 
theorem problem1_solution (x : ℝ) (h : 4 * x^2 = 9) : x = 3 / 2 ∨ x = - (3 / 2) := 
by sorry

-- Problem 2: 
theorem problem2_solution (x : ℝ) (h : (1 - 2 * x)^3 = 8) : x = - 1 / 2 := 
by sorry

end problem1_solution_problem2_solution_l1092_109271


namespace marble_problem_l1092_109240

theorem marble_problem (R B : ℝ) 
  (h1 : R + B = 6000) 
  (h2 : (R + B) - |R - B| = 4800) 
  (h3 : B > R) : B = 3600 :=
sorry

end marble_problem_l1092_109240


namespace subset_strict_M_P_l1092_109242

-- Define the set M
def M : Set ℕ := {x | ∃ a : ℕ, a > 0 ∧ x = a^2 + 1}

-- Define the set P
def P : Set ℕ := {y | ∃ b : ℕ, b > 0 ∧ y = b^2 - 4*b + 5}

-- Prove that M is strictly a subset of P
theorem subset_strict_M_P : M ⊆ P ∧ ∃ x ∈ P, x ∉ M :=
by
  sorry

end subset_strict_M_P_l1092_109242


namespace christmas_gift_distribution_l1092_109205

theorem christmas_gift_distribution :
  ∃ n : ℕ, n = 30 ∧ 
  ∃ (gifts : Finset α) (students : Finset β) 
    (distribute : α → β) (a b c d : α),
    a ∈ gifts ∧ b ∈ gifts ∧ c ∈ gifts ∧ d ∈ gifts ∧ gifts.card = 4 ∧
    students.card = 3 ∧ 
    (∀ s ∈ students, ∃ g ∈ gifts, distribute g = s) ∧ 
    distribute a ≠ distribute b :=
sorry

end christmas_gift_distribution_l1092_109205


namespace Wendy_age_l1092_109246

theorem Wendy_age
  (years_as_accountant : ℕ)
  (years_as_manager : ℕ)
  (percent_accounting_related : ℝ)
  (total_accounting_related : ℕ)
  (total_lifespan : ℝ) :
  years_as_accountant = 25 →
  years_as_manager = 15 →
  percent_accounting_related = 0.50 →
  total_accounting_related = years_as_accountant + years_as_manager →
  (total_accounting_related : ℝ) = percent_accounting_related * total_lifespan →
  total_lifespan = 80 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Wendy_age_l1092_109246


namespace simplify_and_evaluate_l1092_109254

theorem simplify_and_evaluate :
  ∀ (x y : ℝ), x = -1/2 → y = 3 → 3 * (2 * x^2 * y - x * y^2) - 2 * (-2 * y^2 * x + x^2 * y) = -3/2 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end simplify_and_evaluate_l1092_109254


namespace equal_distances_l1092_109243

def Point := ℝ × ℝ × ℝ

def dist (p1 p2 : Point) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 + (z1 - z2) ^ 2

def A : Point := (-8, 0, 0)
def B : Point := (0, 4, 0)
def C : Point := (0, 0, -6)
def D : Point := (0, 0, 0)
def P : Point := (-4, 2, -3)

theorem equal_distances : dist P A = dist P B ∧ dist P B = dist P C ∧ dist P C = dist P D :=
by
  sorry

end equal_distances_l1092_109243


namespace f_zero_is_two_l1092_109287

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x1 x2 x3 x4 x5 : ℝ) : 
  f (x1 + x2 + x3 + x4 + x5) = f x1 + f x2 + f x3 + f x4 + f x5 - 8

theorem f_zero_is_two : f 0 = 2 := 
by
  sorry

end f_zero_is_two_l1092_109287


namespace perpendicular_lines_implies_perpendicular_plane_l1092_109267

theorem perpendicular_lines_implies_perpendicular_plane
  (triangle_sides : Line → Prop)
  (circle_diameters : Line → Prop)
  (perpendicular : Line → Line → Prop)
  (is_perpendicular_to_plane : Line → Prop) :
  (∀ l₁ l₂, triangle_sides l₁ → triangle_sides l₂ → perpendicular l₁ l₂ → is_perpendicular_to_plane l₁) ∧
  (∀ l₁ l₂, circle_diameters l₁ → circle_diameters l₂ → perpendicular l₁ l₂ → is_perpendicular_to_plane l₁) :=
  sorry

end perpendicular_lines_implies_perpendicular_plane_l1092_109267


namespace sara_quarters_eq_l1092_109220

-- Define the initial conditions and transactions
def initial_quarters : ℕ := 21
def dad_quarters : ℕ := 49
def spent_quarters : ℕ := 15
def mom_dollars : ℕ := 2
def quarters_per_dollar : ℕ := 4
def amy_quarters (x : ℕ) := x

-- Define the function to compute total quarters
noncomputable def total_quarters (x : ℕ) : ℕ :=
initial_quarters + dad_quarters - spent_quarters + mom_dollars * quarters_per_dollar + amy_quarters x

-- Prove that the total number of quarters matches the expected value
theorem sara_quarters_eq (x : ℕ) : total_quarters x = 63 + x :=
by
  sorry

end sara_quarters_eq_l1092_109220


namespace expand_polynomial_l1092_109209

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l1092_109209


namespace cube_difference_l1092_109234

theorem cube_difference (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 135 :=
sorry

end cube_difference_l1092_109234


namespace more_likely_to_return_to_initial_count_l1092_109212

noncomputable def P_A (a b c d : ℕ) : ℚ :=
(b * (d + 1) + a * (c + 1)) / (50 * 51)

noncomputable def P_A_bar (a b c d : ℕ) : ℚ :=
(b * c + a * d) / (50 * 51)

theorem more_likely_to_return_to_initial_count (a b c d : ℕ) (h1 : a + b = 50) (h2 : c + d = 50) 
  (h3 : b ≥ a) (h4 : d ≥ c - 1) (h5 : a > 0) :
P_A a b c d > P_A_bar a b c d := by
  sorry

end more_likely_to_return_to_initial_count_l1092_109212


namespace market_value_of_stock_l1092_109200

theorem market_value_of_stock 
  (yield : ℝ) 
  (dividend_percentage : ℝ) 
  (face_value : ℝ) 
  (market_value : ℝ) 
  (h1 : yield = 0.10) 
  (h2 : dividend_percentage = 0.07) 
  (h3 : face_value = 100) 
  (h4 : market_value = (dividend_percentage * face_value) / yield) :
  market_value = 70 := by
  sorry

end market_value_of_stock_l1092_109200


namespace nth_smallest_d0_perfect_square_l1092_109223

theorem nth_smallest_d0_perfect_square (n : ℕ) : 
  ∃ (d_0 : ℕ), (∃ v : ℕ, ∀ t : ℝ, (2 * t * t + d_0 = v * t) ∧ (∃ k : ℕ, v = k ∧ k * k = v * v)) 
               ∧ d_0 = 4^(n - 1) := 
by sorry

end nth_smallest_d0_perfect_square_l1092_109223


namespace bus_speed_excluding_stoppages_l1092_109259

theorem bus_speed_excluding_stoppages (v : ℝ) 
  (speed_including_stoppages : ℝ := 45) 
  (stoppage_time : ℝ := 1/6) 
  (h : v * (1 - stoppage_time) = speed_including_stoppages) : 
  v = 54 := 
by 
  sorry

end bus_speed_excluding_stoppages_l1092_109259


namespace correct_option_B_l1092_109229

-- Define decimal representation of the numbers
def dec_13 : ℕ := 13
def dec_25 : ℕ := 25
def dec_11 : ℕ := 11
def dec_10 : ℕ := 10

-- Define binary representation of the numbers
def bin_1101 : ℕ := 2^(3) + 2^(2) + 2^(0)  -- 1*8 + 1*4 + 0*2 + 1*1 = 13
def bin_10110 : ℕ := 2^(4) + 2^(2) + 2^(1)  -- 1*16 + 0*8 + 1*4 + 1*2 + 0*1 = 22
def bin_1011 : ℕ := 2^(3) + 2^(2) + 2^(0)  -- 1*8 + 0*4 + 1*2 + 1*1 = 11
def bin_10 : ℕ := 2^(1)  -- 1*2 + 0*1 = 2

theorem correct_option_B : (dec_13 = bin_1101) := by
  -- Proof is skipped
  sorry

end correct_option_B_l1092_109229


namespace trigonometric_expression_l1092_109241

theorem trigonometric_expression (α : ℝ) (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (3 * Real.cos α + 3 * Real.sin α) = 2 / 3 :=
by
  sorry

end trigonometric_expression_l1092_109241


namespace permutation_sum_l1092_109226

theorem permutation_sum (n : ℕ) (h1 : n + 3 ≤ 2 * n) (h2 : n + 1 ≤ 4) (h3 : n > 0) :
  Nat.factorial (2 * n) / Nat.factorial (2 * n - (n + 3)) + Nat.factorial 4 / Nat.factorial (4 - (n + 1)) = 744 :=
by
  sorry

end permutation_sum_l1092_109226


namespace max_area_perpendicular_l1092_109264

theorem max_area_perpendicular (a b θ : ℝ) (ha : 0 < a) (hb : 0 < b) (hθ : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) : 
  ∃ θ_max, θ_max = Real.pi / 2 ∧ (∀ θ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → 
  (0 < Real.sin θ → (1 / 2) * a * b * Real.sin θ ≤ (1 / 2) * a * b * 1)) :=
sorry

end max_area_perpendicular_l1092_109264


namespace class_funds_l1092_109245

theorem class_funds (total_contribution : ℕ) (students : ℕ) (contribution_per_student : ℕ) (remaining_amount : ℕ) 
    (h1 : total_contribution = 90) 
    (h2 : students = 19) 
    (h3 : contribution_per_student = 4) 
    (h4 : remaining_amount = total_contribution - (students * contribution_per_student)) : 
    remaining_amount = 14 :=
sorry

end class_funds_l1092_109245


namespace biff_break_even_night_hours_l1092_109222

-- Define the constants and conditions
def ticket_cost : ℝ := 11
def snacks_cost : ℝ := 3
def headphones_cost : ℝ := 16
def lunch_cost : ℝ := 8
def dinner_cost : ℝ := 10
def accommodation_cost : ℝ := 35

def total_expenses_without_wifi : ℝ := ticket_cost + snacks_cost + headphones_cost + lunch_cost + dinner_cost + accommodation_cost

def earnings_per_hour : ℝ := 12
def wifi_cost_day : ℝ := 2
def wifi_cost_night : ℝ := 1

-- Define the total expenses with wifi cost variable
def total_expenses (D N : ℝ) : ℝ := total_expenses_without_wifi + (wifi_cost_day * D) + (wifi_cost_night * N)

-- Define the total earnings
def total_earnings (D N : ℝ) : ℝ := earnings_per_hour * (D + N)

-- Prove that the minimum number of hours Biff needs to work at night to break even is 8 hours
theorem biff_break_even_night_hours :
  ∃ N : ℕ, N = 8 ∧ total_earnings 0 N ≥ total_expenses 0 N := 
by 
  sorry

end biff_break_even_night_hours_l1092_109222


namespace min_value_PA_PF_l1092_109201

noncomputable def minimum_value_of_PA_and_PF_minimum 
  (x y : ℝ)
  (A : ℝ × ℝ)
  (F : ℝ × ℝ) : ℝ :=
  if ((A = (-1, 8)) ∧ (F = (0, 1)) ∧ (x^2 = 4 * y)) then 9 else 0

theorem min_value_PA_PF 
  (A : ℝ × ℝ := (-1, 8))
  (F : ℝ × ℝ := (0, 1))
  (P : ℝ × ℝ)
  (hP : P.1^2 = 4 * P.2) :
  minimum_value_of_PA_and_PF_minimum P.1 P.2 A F = 9 :=
by
  sorry

end min_value_PA_PF_l1092_109201


namespace multiplication_results_l1092_109291

theorem multiplication_results
  (h1 : 25 * 4 = 100) :
  25 * 8 = 200 ∧ 25 * 12 = 300 ∧ 250 * 40 = 10000 ∧ 25 * 24 = 600 :=
by
  sorry

end multiplication_results_l1092_109291


namespace sum_of_reciprocals_l1092_109285

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 4 * x * y) : 
  (1 / x) + (1 / y) = 4 :=
by
  sorry

end sum_of_reciprocals_l1092_109285


namespace bottle_caps_found_l1092_109268

theorem bottle_caps_found
  (caps_current : ℕ) 
  (caps_earlier : ℕ) 
  (h_current : caps_current = 32) 
  (h_earlier : caps_earlier = 25) :
  caps_current - caps_earlier = 7 :=
by 
  sorry

end bottle_caps_found_l1092_109268


namespace largest_x_l1092_109261

theorem largest_x (x : ℝ) : 
  (∃ x, (15 * x ^ 2 - 40 * x + 18) / (4 * x - 3) + 6 * x = 7 * x - 2) → 
  (x ≤ 1) := sorry

end largest_x_l1092_109261


namespace triangle_congruence_example_l1092_109263

variable {A B C : Type}
variable (A' B' C' : Type)

def triangle (A B C : Type) : Prop := true

def congruent (t1 t2 : Prop) : Prop := true

variable (P : ℕ)

def perimeter (t : Prop) (p : ℕ) : Prop := true

def length (a b : Type) (l : ℕ) : Prop := true

theorem triangle_congruence_example :
  ∀ (A B C A' B' C' : Type) (h_cong : congruent (triangle A B C) (triangle A' B' C'))
    (h_perimeter : perimeter (triangle A B C) 20)
    (h_AB : length A B 8)
    (h_BC : length B C 5),
    length A C 7 :=
by sorry

end triangle_congruence_example_l1092_109263


namespace standard_equation_of_ellipse_midpoint_of_chord_l1092_109262

variables (a b c : ℝ)
variables (x1 y1 x2 y2 : ℝ)
variables (A B : ℝ × ℝ)

axiom conditions :
  a > b ∧ b > 0 ∧
  (c / a = (Real.sqrt 6) / 3) ∧
  a = Real.sqrt 3 ∧
  a^2 = b^2 + c^2 ∧
  (A = (-1, 0)) ∧ (B = (x2, y2)) ∧
  A ≠ B ∧
  (∃ l : ℝ -> ℝ, l (-1) = 0 ∧ ∀ x, l x = x + 1) ∧
  (∃ x1 x2 y1 y2 : ℝ, x1 + x2 = -3 / 2)

theorem standard_equation_of_ellipse :
  ∃ (e : ℝ), e = 1 ∧ (x1 / 3) + y1 = 1 := sorry

theorem midpoint_of_chord :
  ∃ (m : ℝ × ℝ), m = (-(3 / 4), 1 / 4) := sorry

end standard_equation_of_ellipse_midpoint_of_chord_l1092_109262


namespace num_integer_distance_pairs_5x5_grid_l1092_109257

-- Define the problem conditions
def grid_size : ℕ := 5

-- Define a function to calculate the number of pairs of vertices with integer distances
noncomputable def count_integer_distance_pairs (n : ℕ) : ℕ := sorry

-- The theorem to prove
theorem num_integer_distance_pairs_5x5_grid : count_integer_distance_pairs grid_size = 108 :=
by
  sorry

end num_integer_distance_pairs_5x5_grid_l1092_109257


namespace problem_statement_l1092_109297

open Real

variable (a b c : ℝ)

theorem problem_statement
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h_cond : a + b + c + a * b * c = 4) :
  (1 + a / b + c * a) * (1 + b / c + a * b) * (1 + c / a + b * c) ≥ 27 := 
by
  sorry

end problem_statement_l1092_109297


namespace find_decimal_number_l1092_109230

noncomputable def decimal_number (x : ℝ) : Prop := 
x > 0 ∧ (100000 * x = 5 * (1 / x))

theorem find_decimal_number {x : ℝ} (h : decimal_number x) : x = 1 / (100 * Real.sqrt 2) :=
by
  sorry

end find_decimal_number_l1092_109230


namespace fraction_product_l1092_109219

theorem fraction_product :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := 
by
  -- Detailed proof steps would go here
  sorry

end fraction_product_l1092_109219


namespace find_other_sides_of_triangle_l1092_109236

-- Given conditions
variables (a b c : ℝ) -- side lengths of the triangle
variables (perimeter : ℝ) -- perimeter of the triangle
variables (iso : ℝ → ℝ → ℝ → Prop) -- a predicate to check if a triangle is isosceles
variables (triangle_ineq : ℝ → ℝ → ℝ → Prop) -- another predicate to check the triangle inequality

-- Given facts
axiom triangle_is_isosceles : iso a b c
axiom triangle_perimeter : a + b + c = perimeter
axiom one_side_is_4 : a = 4 ∨ b = 4 ∨ c = 4
axiom perimeter_value : perimeter = 17

-- The mathematically equivalent proof problem
theorem find_other_sides_of_triangle :
  (b = 6.5 ∧ c = 6.5) ∨ (a = 6.5 ∧ c = 6.5) ∨ (a = 6.5 ∧ b = 6.5) :=
sorry

end find_other_sides_of_triangle_l1092_109236


namespace Marcy_120_votes_l1092_109279

-- Definitions based on conditions
def votes (name : String) : ℕ := sorry -- placeholder definition

-- Conditions
def Joey_votes := votes "Joey" = 8
def Jill_votes := votes "Jill" = votes "Joey" + 4
def Barry_votes := votes "Barry" = 2 * (votes "Joey" + votes "Jill")
def Marcy_votes := votes "Marcy" = 3 * votes "Barry"
def Tim_votes := votes "Tim" = votes "Marcy" / 2
def Sam_votes := votes "Sam" = votes "Tim" + 10

-- Theorem to prove
theorem Marcy_120_votes : Joey_votes → Jill_votes → Barry_votes → Marcy_votes → Tim_votes → Sam_votes → votes "Marcy" = 120 := by
  intros
  -- Skipping the proof
  sorry

end Marcy_120_votes_l1092_109279


namespace number_of_multiples_of_15_between_35_and_200_l1092_109294

theorem number_of_multiples_of_15_between_35_and_200 : ∃ n : ℕ, n = 11 ∧ ∃ k : ℕ, k ≤ 200 ∧ k ≥ 35 ∧ (∃ m : ℕ, m < n ∧ 45 + m * 15 = k) :=
by
  sorry

end number_of_multiples_of_15_between_35_and_200_l1092_109294


namespace range_of_2x_plus_y_range_of_c_l1092_109228

open Real

def point_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 2 * y

theorem range_of_2x_plus_y (x y : ℝ) (h : point_on_circle x y) : 
  1 - sqrt 2 ≤ 2 * x + y ∧ 2 * x + y ≤ 1 + sqrt 2 :=
sorry

theorem range_of_c (c : ℝ) : 
  (∀ x y : ℝ, point_on_circle x y → x + y + c > 0) → c ≥ -1 :=
sorry

end range_of_2x_plus_y_range_of_c_l1092_109228


namespace third_recipe_soy_sauce_l1092_109277

theorem third_recipe_soy_sauce :
  let bottle_ounces := 16
  let cup_ounces := 8
  let first_recipe_cups := 2
  let second_recipe_cups := 1
  let total_bottles := 3
  (total_bottles * bottle_ounces) / cup_ounces - (first_recipe_cups + second_recipe_cups) = 3 :=
by
  sorry

end third_recipe_soy_sauce_l1092_109277


namespace cricket_innings_l1092_109289

theorem cricket_innings (n : ℕ) (h1 : (32 * n + 137) / (n + 1) = 37) : n = 20 :=
sorry

end cricket_innings_l1092_109289


namespace jessie_weight_before_jogging_l1092_109252

theorem jessie_weight_before_jogging (current_weight lost_weight : ℕ) 
(hc : current_weight = 67)
(hl : lost_weight = 7) : 
current_weight + lost_weight = 74 := 
by
  -- Here we skip the proof part
  sorry

end jessie_weight_before_jogging_l1092_109252


namespace vacuum_total_time_l1092_109206

theorem vacuum_total_time (x : ℕ) (hx : 2 * x + 5 = 27) :
  27 + x = 38 :=
by
  sorry

end vacuum_total_time_l1092_109206


namespace A_independent_of_beta_l1092_109214

noncomputable def A (alpha beta : ℝ) : ℝ :=
  (Real.sin (alpha + beta) ^ 2) + (Real.sin (beta - alpha) ^ 2) - 
  2 * (Real.sin (alpha + beta)) * (Real.sin (beta - alpha)) * (Real.cos (2 * alpha))

theorem A_independent_of_beta (alpha beta : ℝ) : 
  ∃ (c : ℝ), ∀ beta : ℝ, A alpha beta = c :=
by
  sorry

end A_independent_of_beta_l1092_109214


namespace arithmetic_sequence_a8_l1092_109295

theorem arithmetic_sequence_a8 (a_1 : ℕ) (S_5 : ℕ) (h_a1 : a_1 = 1) (h_S5 : S_5 = 35) : 
    ∃ a_8 : ℕ, a_8 = 22 :=
by
  sorry

end arithmetic_sequence_a8_l1092_109295


namespace age_in_1930_l1092_109274

/-- A person's age at the time of their death (y) was one 31st of their birth year,
and we want to prove the person's age in 1930 (x). -/
theorem age_in_1930 (x y : ℕ) (h : 31 * y + x = 1930) (hx : 0 < x) (hxy : x < y) :
  x = 39 :=
sorry

end age_in_1930_l1092_109274


namespace main_theorem_l1092_109253

/-- A good integer is an integer whose absolute value is not a perfect square. -/
def good (n : ℤ) : Prop := ∀ k : ℤ, k^2 ≠ |n|

/-- Integer m can be represented as a sum of three distinct good integers u, v, w whose product is the square of an odd integer. -/
def special_representation (m : ℤ) : Prop :=
  ∃ u v w : ℤ,
    good u ∧ good v ∧ good w ∧
    (u ≠ v ∧ u ≠ w ∧ v ≠ w) ∧
    (∃ k : ℤ, (u * v * w = k^2 ∧ k % 2 = 1)) ∧
    (m = u + v + w)

/-- All integers m having the property that they can be represented in infinitely many ways as a sum of three distinct good integers whose product is the square of an odd integer are those which are congruent to 3 modulo 4. -/
theorem main_theorem (m : ℤ) : special_representation m ↔ m % 4 = 3 := sorry

end main_theorem_l1092_109253


namespace window_width_l1092_109256

theorem window_width (length area : ℝ) (h_length : length = 6) (h_area : area = 60) :
  area / length = 10 :=
by
  sorry

end window_width_l1092_109256
