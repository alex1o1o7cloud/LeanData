import Mathlib

namespace find_triples_l2291_229196

theorem find_triples (a b c : ℝ) :
  a^2 + b^2 + c^2 = 1 ∧ a * (2 * b - 2 * a - c) ≥ 1/2 ↔ 
  (a = 1 / Real.sqrt 6 ∧ b = 2 / Real.sqrt 6 ∧ c = -1 / Real.sqrt 6) ∨
  (a = -1 / Real.sqrt 6 ∧ b = -2 / Real.sqrt 6 ∧ c = 1 / Real.sqrt 6) := 
by 
  sorry

end find_triples_l2291_229196


namespace series_items_increase_l2291_229100

theorem series_items_increase (n : ℕ) (hn : n ≥ 2) :
  (2^n + 1) - 2^(n-1) - 1 = 2^(n-1) :=
by
  sorry

end series_items_increase_l2291_229100


namespace movie_screening_guests_l2291_229118

theorem movie_screening_guests
  (total_guests : ℕ)
  (women_percentage : ℝ)
  (men_count : ℕ)
  (men_left_fraction : ℝ)
  (children_left_percentage : ℝ)
  (children_count : ℕ)
  (people_left : ℕ) :
  total_guests = 75 →
  women_percentage = 0.40 →
  men_count = 25 →
  men_left_fraction = 1/3 →
  children_left_percentage = 0.20 →
  children_count = total_guests - (round (women_percentage * total_guests) + men_count) →
  people_left = (round (men_left_fraction * men_count)) + (round (children_left_percentage * children_count)) →
  (total_guests - people_left) = 63 :=
by
  intros ht hw hm hf hc hc_count hl
  sorry

end movie_screening_guests_l2291_229118


namespace diff_of_squares_635_615_l2291_229141

theorem diff_of_squares_635_615 : 635^2 - 615^2 = 25000 :=
by
  sorry

end diff_of_squares_635_615_l2291_229141


namespace raghu_investment_l2291_229170

noncomputable def investment_problem (R T V : ℝ) : Prop :=
  V = 1.1 * T ∧
  T = 0.9 * R ∧
  R + T + V = 6358 ∧
  R = 2200

theorem raghu_investment
  (R T V : ℝ)
  (h1 : V = 1.1 * T)
  (h2 : T = 0.9 * R)
  (h3 : R + T + V = 6358) :
  R = 2200 :=
sorry

end raghu_investment_l2291_229170


namespace calc_f_y_eq_2f_x_l2291_229138

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

theorem calc_f_y_eq_2f_x (x : ℝ) (h : -1 < x) (h' : x < 1) :
  f ( (2 * x + x^2) / (1 + 2 * x^2) ) = 2 * f x := by
  sorry

end calc_f_y_eq_2f_x_l2291_229138


namespace area_of_right_triangle_l2291_229131

theorem area_of_right_triangle (m k : ℝ) (hm : 0 < m) (hk : 0 < k) : 
  ∃ A : ℝ, A = (k^2) / (2 * m) :=
by
  sorry

end area_of_right_triangle_l2291_229131


namespace total_distinguishable_triangles_l2291_229110

-- Define number of colors
def numColors : Nat := 8

-- Define center colors
def centerColors : Nat := 3

-- Prove the total number of distinguishable large equilateral triangles
theorem total_distinguishable_triangles : 
  numColors * (numColors + numColors * (numColors - 1) + (numColors.choose 3)) * centerColors = 360 := by
  sorry

end total_distinguishable_triangles_l2291_229110


namespace solution_set_inequality_l2291_229167

variable (f : ℝ → ℝ)
variable (h1 : ∀ x, f (x - 1/2) + f (x + 1) = 0)
variable (h2 : e ^ 3 * f 2018 = 1)
variable (h3 : ∀ x, f x > f'' (-x))
variable (h4 : ∀ x, f x = f (-x))

theorem solution_set_inequality :
  ∀ x, f (x - 1) > 1 / (e ^ x) ↔ x > 3 :=
sorry

end solution_set_inequality_l2291_229167


namespace problem_k_star_k_star_k_l2291_229146

def star (x y : ℝ) : ℝ := 2 * x^2 - y

theorem problem_k_star_k_star_k (k : ℝ) : star k (star k k) = k :=
by
  sorry

end problem_k_star_k_star_k_l2291_229146


namespace arithmetic_sequence_sum_l2291_229173

theorem arithmetic_sequence_sum (a_n : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : S 9 = a_n 4 + a_n 5 + a_n 6 + 72)
  (h2 : ∀ n, S n = n * (a_n 1 + a_n n) / 2)
  (h3 : ∀ n, a_n (n+1) - a_n n = d)
  (h4 : a_n 1 + a_n 9 = a_n 3 + a_n 7)
  (h5 : a_n 3 + a_n 7 = a_n 4 + a_n 6)
  (h6 : a_n 4 + a_n 6 = 2 * a_n 5) : 
  a_n 3 + a_n 7 = 24 := 
sorry

end arithmetic_sequence_sum_l2291_229173


namespace smallest_next_divisor_l2291_229160

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_divisor (a b : ℕ) : Prop := b % a = 0

theorem smallest_next_divisor 
  (m : ℕ) 
  (h1 : 1000 ≤ m ∧ m < 10000) 
  (h2 : is_even m) 
  (h3 : is_divisor 171 m)
  : ∃ k, k > 171 ∧ k = 190 ∧ is_divisor k m := 
by
  sorry

end smallest_next_divisor_l2291_229160


namespace train_and_car_combined_time_l2291_229197

theorem train_and_car_combined_time (car_time : ℝ) (train_time : ℝ) 
  (h1 : train_time = car_time + 2) (h2 : car_time = 4.5) : 
  car_time + train_time = 11 := 
by 
  -- Proof goes here
  sorry

end train_and_car_combined_time_l2291_229197


namespace convex_polyhedron_space_diagonals_l2291_229190

theorem convex_polyhedron_space_diagonals
  (vertices : ℕ)
  (edges : ℕ)
  (faces : ℕ)
  (triangular_faces : ℕ)
  (hexagonal_faces : ℕ)
  (total_faces : faces = triangular_faces + hexagonal_faces)
  (vertices_eq : vertices = 30)
  (edges_eq : edges = 72)
  (triangular_faces_eq : triangular_faces = 32)
  (hexagonal_faces_eq : hexagonal_faces = 12)
  (faces_eq : faces = 44) :
  ((vertices * (vertices - 1)) / 2) - edges - 
  (triangular_faces * 0 + hexagonal_faces * ((6 * (6 - 3)) / 2)) = 255 := by
sorry

end convex_polyhedron_space_diagonals_l2291_229190


namespace arithmetic_geometric_mean_inequality_l2291_229107

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) : 
  (a + b + c) / 3 ≥ (a * b * c) ^ (1 / 3) :=
sorry

end arithmetic_geometric_mean_inequality_l2291_229107


namespace problem1_problem2_problem3_l2291_229152

variable {m n p x : ℝ}

-- Problem 1
theorem problem1 : m^2 * (n - 3) + 4 * (3 - n) = (n - 3) * (m + 2) * (m - 2) := 
sorry

-- Problem 2
theorem problem2 : (p - 3) * (p - 1) + 1 = (p - 2) ^ 2 := 
sorry

-- Problem 3
theorem problem3 (hx : x^2 + x + 1 / 4 = 0) : (2 * x + 1) / (x + 1) + (x - 1) / 1 / (x + 2) / (x^2 + 2 * x + 1) = -1 / 4 :=
sorry

end problem1_problem2_problem3_l2291_229152


namespace quadratic_function_min_value_l2291_229142

theorem quadratic_function_min_value (a b c : ℝ) (h_a : a > 0) (h_b : b ≠ 0) 
(h_f0 : |c| = 1) (h_f1 : |a + b + c| = 1) (h_fn1 : |a - b + c| = 1) :
∃ f : ℝ → ℝ, (∀ x : ℝ, f x = a*x^2 + b*x + c) ∧
  (|f 0| = 1) ∧ (|f 1| = 1) ∧ (|f (-1)| = 1) ∧
  (f 0 = -(5/4) ∨ f 1 = -(5/4) ∨ f (-1) = -(5/4)) :=
by
  sorry

end quadratic_function_min_value_l2291_229142


namespace number_of_arrangements_of_six_students_l2291_229162

/-- A and B cannot stand together -/
noncomputable def arrangements_A_B_not_together (n: ℕ) (A B: ℕ) : ℕ :=
  if n = 6 then 480 else 0

theorem number_of_arrangements_of_six_students :
  arrangements_A_B_not_together 6 1 2 = 480 :=
sorry

end number_of_arrangements_of_six_students_l2291_229162


namespace find_c_l2291_229163

structure ProblemData where
  (r : ℝ → ℝ)
  (s : ℝ → ℝ)
  (h : r (s 3) = 20)

def r (x : ℝ) : ℝ := 5 * x - 10
def s (x : ℝ) (c : ℝ) : ℝ := 4 * x - c

theorem find_c (c : ℝ) (h : (r (s 3 c)) = 20) : c = 6 :=
sorry

end find_c_l2291_229163


namespace inverse_proportion_passes_first_and_third_quadrants_l2291_229149

theorem inverse_proportion_passes_first_and_third_quadrants (m : ℝ) :
  ((∀ x : ℝ, x ≠ 0 → (x > 0 → (m - 3) / x > 0) ∧ (x < 0 → (m - 3) / x < 0)) → m = 5) := 
by 
  sorry

end inverse_proportion_passes_first_and_third_quadrants_l2291_229149


namespace weight_of_new_person_l2291_229112

theorem weight_of_new_person 
  (avg_weight_increase : ℝ)
  (old_weight : ℝ) 
  (num_people : ℕ)
  (new_weight_increase : ℝ)
  (total_weight_increase : ℝ)  
  (W : ℝ)
  (h1 : avg_weight_increase = 1.8)
  (h2 : old_weight = 69)
  (h3 : num_people = 6) 
  (h4 : new_weight_increase = num_people * avg_weight_increase) 
  (h5 : total_weight_increase = new_weight_increase)
  (h6 : W = old_weight + total_weight_increase)
  : W = 79.8 := 
by
  sorry

end weight_of_new_person_l2291_229112


namespace sum_of_terms_in_geometric_sequence_eq_fourteen_l2291_229175

theorem sum_of_terms_in_geometric_sequence_eq_fourteen
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h_a1 : a 1 = 1)
  (h_arith : 4 * a 2 = 2 * a 3 ∧ 2 * a 3 - 4 * a 2 = a 4 - 2 * a 3) :
  a 2 + a 3 + a 4 = 14 :=
sorry

end sum_of_terms_in_geometric_sequence_eq_fourteen_l2291_229175


namespace steve_needs_28_feet_of_wood_l2291_229120

theorem steve_needs_28_feet_of_wood :
  (6 * 4) + (2 * 2) = 28 := by
  sorry

end steve_needs_28_feet_of_wood_l2291_229120


namespace Petya_time_comparison_l2291_229174

-- Define the conditions
variables (a V : ℝ) (hV : V > 0)

noncomputable def T_planned := a / V

noncomputable def T1 := a / (2.5 * V)

noncomputable def T2 := a / (1.6 * V)

noncomputable def T_real := T1 + T2

-- State the main theorem
theorem Petya_time_comparison (ha : a > 0) : T_real a V > T_planned a V :=
by
  -- Proof to be filled in
  sorry

end Petya_time_comparison_l2291_229174


namespace prob_green_ball_l2291_229172

-- Definitions for the conditions
def red_balls_X := 3
def green_balls_X := 7
def total_balls_X := red_balls_X + green_balls_X

def red_balls_YZ := 7
def green_balls_YZ := 3
def total_balls_YZ := red_balls_YZ + green_balls_YZ

-- The probability of selecting any container
def prob_select_container := 1 / 3

-- The probabilities of drawing a green ball from each container
def prob_green_given_X := green_balls_X / total_balls_X
def prob_green_given_YZ := green_balls_YZ / total_balls_YZ

-- The combined probability of selecting a green ball
theorem prob_green_ball : 
  prob_select_container * prob_green_given_X + 
  prob_select_container * prob_green_given_YZ + 
  prob_select_container * prob_green_given_YZ = 13 / 30 := 
  by sorry

end prob_green_ball_l2291_229172


namespace sum_of_first_n_natural_numbers_l2291_229114

theorem sum_of_first_n_natural_numbers (n : ℕ) (h : n * (n + 1) / 2 = 190) : n = 19 :=
sorry

end sum_of_first_n_natural_numbers_l2291_229114


namespace seven_searchlights_shadow_length_l2291_229169

noncomputable def searchlight_positioning (n : ℕ) (angle : ℝ) (shadow_length : ℝ) : Prop :=
  ∃ (positions : Fin n → ℝ × ℝ), ∀ i : Fin n, ∃ shadow : ℝ, shadow = shadow_length ∧
  (∀ j : Fin n, i ≠ j → ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  θ - angle / 2 < θ ∧ θ + angle / 2 > θ → shadow = shadow_length)

theorem seven_searchlights_shadow_length :
  searchlight_positioning 7 (Real.pi / 2) 7000 :=
sorry

end seven_searchlights_shadow_length_l2291_229169


namespace cone_water_volume_percentage_l2291_229101

theorem cone_water_volume_percentage
  (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let full_volume := (1 / 3) * π * r^2 * h
  let water_height := (2 / 3) * h
  let water_radius := (2 / 3) * r
  let water_volume := (1 / 3) * π * water_radius^2 * water_height
  let percentage := (water_volume / full_volume) * 100
  abs (percentage - 29.6296) < 0.0001 :=
by
  let full_volume := (1 / 3) * π * r^2 * h
  let water_height := (2 / 3) * h
  let water_radius := (2 / 3) * r
  let water_volume := (1 / 3) * π * water_radius^2 * water_height
  let percentage := (water_volume / full_volume) * 100
  sorry

end cone_water_volume_percentage_l2291_229101


namespace net_amount_spent_correct_l2291_229199

def trumpet_cost : ℝ := 145.16
def song_book_revenue : ℝ := 5.84
def net_amount_spent : ℝ := 139.32

theorem net_amount_spent_correct : trumpet_cost - song_book_revenue = net_amount_spent :=
by
  sorry

end net_amount_spent_correct_l2291_229199


namespace total_animals_l2291_229105

def pigs : ℕ := 10

def cows : ℕ := 2 * pigs - 3

def goats : ℕ := cows + 6

theorem total_animals : pigs + cows + goats = 50 := by
  sorry

end total_animals_l2291_229105


namespace marbles_sum_l2291_229116

variable {K M : ℕ}

theorem marbles_sum (hFabian_kyle : 15 = 3 * K) (hFabian_miles : 15 = 5 * M) :
  K + M = 8 :=
by
  sorry

end marbles_sum_l2291_229116


namespace no_n_exists_l2291_229130

theorem no_n_exists (n : ℕ) : ¬ ∃ n : ℕ, (n^2 + 6 * n + 2019) % 100 = 0 :=
by {
  sorry
}

end no_n_exists_l2291_229130


namespace general_formula_an_bounds_Mn_l2291_229136

variable {n : ℕ}

-- Define the sequence Sn
def S : ℕ → ℚ := λ n => n * (4 * n - 3) - 2 * n * (n - 1)

-- Define the sequence an based on Sn
def a : ℕ → ℚ := λ n =>
  if n = 0 then 0 else S n - S (n - 1)

-- Define the sequence Mn and the bounds to prove
def M : ℕ → ℚ := λ n => (1 / 4) * (1 - (1 / (4 * n + 1)))

-- Theorem: General formula for the sequence {a_n}
theorem general_formula_an (n : ℕ) (hn : 1 ≤ n) : a n = 4 * n - 3 :=
  sorry

-- Theorem: Bounds for the sequence {M_n}
theorem bounds_Mn (n : ℕ) (hn : 1 ≤ n) : (1 / 5 : ℚ) ≤ M n ∧ M n < (1 / 4) :=
  sorry

end general_formula_an_bounds_Mn_l2291_229136


namespace calculate_sum_calculate_product_l2291_229177

theorem calculate_sum : 13 + (-7) + (-6) = 0 :=
by sorry

theorem calculate_product : (-8) * (-4 / 3) * (-0.125) * (5 / 4) = -5 / 3 :=
by sorry

end calculate_sum_calculate_product_l2291_229177


namespace f_even_l2291_229198

-- Let g(x) = x^3 - x
def g (x : ℝ) : ℝ := x^3 - x

-- Let f(x) = |g(x^2)|
def f (x : ℝ) : ℝ := abs (g (x^2))

-- Prove that f(x) is even, i.e., f(-x) = f(x) for all x
theorem f_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end f_even_l2291_229198


namespace length_of_tube_l2291_229119

/-- Prove that the length of the tube is 1.5 meters given the initial conditions -/
theorem length_of_tube (h1 : ℝ) (m_water : ℝ) (rho : ℝ) (g : ℝ) (p_ratio : ℝ) :
  h1 = 1.5 ∧ m_water = 1000 ∧ rho = 1000 ∧ g = 9.8 ∧ p_ratio = 2 → 
  ∃ h2 : ℝ, h2 = 1.5 :=
by
  sorry

end length_of_tube_l2291_229119


namespace perpendicular_lines_intersection_l2291_229137

theorem perpendicular_lines_intersection (a b c d : ℝ)
    (h_perpendicular : (a / 2) * (-2 / b) = -1)
    (h_intersection1 : a * 2 - 2 * (-3) = d)
    (h_intersection2 : 2 * 2 + b * (-3) = c) :
    d = 12 := 
sorry

end perpendicular_lines_intersection_l2291_229137


namespace isosceles_vertex_angle_l2291_229135

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem isosceles_vertex_angle (a b θ : ℝ)
  (h1 : a = golden_ratio * b) :
  ∃ θ, θ = 36 :=
by
  sorry

end isosceles_vertex_angle_l2291_229135


namespace num_complementary_sets_l2291_229180

-- Definitions for shapes, colors, shades, and patterns
inductive Shape
| circle | square | triangle

inductive Color
| red | blue | green

inductive Shade
| light | medium | dark

inductive Pattern
| striped | dotted | plain

-- Definition of a card
structure Card where
  shape : Shape
  color : Color
  shade : Shade
  pattern : Pattern

-- Condition: Each possible combination is represented once in a deck of 81 cards.
def deck : List Card := sorry -- Construct the deck with 81 unique cards

-- Predicate for complementary sets of three cards
def is_complementary (c1 c2 c3 : Card) : Prop :=
  (c1.shape = c2.shape ∧ c2.shape = c3.shape ∧ c1.shape = c3.shape ∨
   c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape) ∧
  (c1.color = c2.color ∧ c2.color = c3.color ∧ c1.color = c3.color ∨
   c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color) ∧
  (c1.shade = c2.shade ∧ c2.shade = c3.shade ∧ c1.shade = c3.shade ∨
   c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c1.shade ≠ c3.shade) ∧
  (c1.pattern = c2.pattern ∧ c2.pattern = c3.pattern ∧ c1.pattern = c3.pattern ∨
   c1.pattern ≠ c2.pattern ∧ c2.pattern ≠ c3.pattern ∧ c1.pattern ≠ c3.pattern)

-- Statement of the theorem to prove
theorem num_complementary_sets : 
  ∃ (complementary_sets : List (Card × Card × Card)), 
  complementary_sets.length = 5400 ∧
  ∀ (c1 c2 c3 : Card), (c1, c2, c3) ∈ complementary_sets → is_complementary c1 c2 c3 :=
sorry

end num_complementary_sets_l2291_229180


namespace fill_tank_time_l2291_229111

theorem fill_tank_time (t_A t_B : ℕ) (hA : t_A = 20) (hB : t_B = t_A / 4) :
  t_B = 4 := by
  sorry

end fill_tank_time_l2291_229111


namespace boys_play_theater_with_Ocho_l2291_229133

variables (Ocho_friends : ℕ) (half_girls : Ocho_friends / 2 = 4)

theorem boys_play_theater_with_Ocho : (Ocho_friends / 2) = 4 := by
  -- Ocho_friends is the total number of Ocho's friends
  -- half_girls is given as a condition that half of Ocho's friends are girls
  -- thus, we directly use this to conclude that the number of boys is 4
  sorry

end boys_play_theater_with_Ocho_l2291_229133


namespace solve_for_x_l2291_229134

theorem solve_for_x (x : ℚ) : (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ↔ x = -5 / 3 :=
by
  sorry

end solve_for_x_l2291_229134


namespace triangle_angle_conditions_l2291_229164

theorem triangle_angle_conditions
  (a b c : ℝ)
  (α β γ : ℝ)
  (h_triangle : c^2 = a^2 + 2 * b^2 * Real.cos β)
  (h_tri_angles : α + β + γ = 180):
  (γ = β / 2 + 90 ∧ α = 90 - 3 * β / 2 ∧ 0 < β ∧ β < 60) ∨ 
  (α = β / 2 ∧ γ = 180 - 3 * β / 2 ∧ 0 < β ∧ β < 120) :=
sorry

end triangle_angle_conditions_l2291_229164


namespace no_such_n_exists_l2291_229171

theorem no_such_n_exists :
  ¬ ∃ n : ℕ, 0 < n ∧
  (∃ a : ℕ, 2 * n^2 + 1 = a^2) ∧
  (∃ b : ℕ, 3 * n^2 + 1 = b^2) ∧
  (∃ c : ℕ, 6 * n^2 + 1 = c^2) :=
sorry

end no_such_n_exists_l2291_229171


namespace solution_set_of_f_lt_exp_l2291_229186

noncomputable def f : ℝ → ℝ := sorry -- assume f is a differentiable function

-- Define the conditions
axiom h_deriv : ∀ x : ℝ, deriv f x < f x
axiom h_periodic : ∀ x : ℝ, f (x + 2) = f (x - 2)
axiom h_value_at_4 : f 4 = 1

-- The main statement to be proved
theorem solution_set_of_f_lt_exp :
  ∀ x : ℝ, (f x < Real.exp x ↔ x > 0) :=
by
  intro x
  sorry

end solution_set_of_f_lt_exp_l2291_229186


namespace safe_security_system_l2291_229123

theorem safe_security_system (commission_members : ℕ) 
                            (majority_access : ℕ)
                            (max_inaccess_members : ℕ) 
                            (locks : ℕ)
                            (keys_per_member : ℕ) :
  commission_members = 11 →
  majority_access = 6 →
  max_inaccess_members = 5 →
  locks = (Nat.choose 11 5) →
  keys_per_member = (locks * 6) / 11 →
  locks = 462 ∧ keys_per_member = 252 :=
by
  intros
  sorry

end safe_security_system_l2291_229123


namespace total_lotus_flowers_l2291_229145

theorem total_lotus_flowers (x : ℕ) (h1 : x > 0) 
  (c1 : 3 ∣ x)
  (c2 : 5 ∣ x)
  (c3 : 6 ∣ x)
  (c4 : 4 ∣ x)
  (h_total : x = x / 3 + x / 5 + x / 6 + x / 4 + 6) : 
  x = 120 :=
by
  sorry

end total_lotus_flowers_l2291_229145


namespace number_of_rows_with_7_eq_5_l2291_229165

noncomputable def number_of_rows_with_7_people (x y : ℕ) : Prop :=
  7 * x + 6 * (y - x) = 59

theorem number_of_rows_with_7_eq_5 :
  ∃ x y : ℕ, number_of_rows_with_7_people x y ∧ x = 5 :=
by {
  sorry
}

end number_of_rows_with_7_eq_5_l2291_229165


namespace compute_c_plus_d_l2291_229127

variable {c d : ℝ}

-- Define the given polynomial equations
def poly_c (c : ℝ) := c^3 - 21*c^2 + 28*c - 70
def poly_d (d : ℝ) := 10*d^3 - 75*d^2 - 350*d + 3225

theorem compute_c_plus_d (hc : poly_c c = 0) (hd : poly_d d = 0) : c + d = 21 / 2 := sorry

end compute_c_plus_d_l2291_229127


namespace power_sum_is_integer_l2291_229156

theorem power_sum_is_integer (a : ℝ) (n : ℕ) (h_pos : 0 < n)
  (h_k : ∃ k : ℤ, k = a + 1/a) : 
  ∃ m : ℤ, m = a^n + 1/a^n := 
sorry

end power_sum_is_integer_l2291_229156


namespace exp_pi_gt_pi_exp_l2291_229140

theorem exp_pi_gt_pi_exp (h : Real.pi > Real.exp 1) : Real.exp Real.pi > Real.pi ^ Real.exp 1 := by
  sorry

end exp_pi_gt_pi_exp_l2291_229140


namespace find_expression_value_l2291_229139

theorem find_expression_value (x : ℝ) (h : x^2 - 5*x = 14) : 
  (x-1)*(2*x-1) - (x+1)^2 + 1 = 15 := 
by 
  sorry

end find_expression_value_l2291_229139


namespace problem_l2291_229189

open Complex

-- Given condition: smallest positive integer n greater than 3
def smallest_n_gt_3 (n : ℕ) : Prop :=
  n > 3 ∧ ∀ m : ℕ, m > 3 → m < n → False

-- Given condition: equation holds for complex numbers
def equation_holds (a b : ℝ) (n : ℕ) : Prop :=
  (a + b * I)^n + a = (a - b * I)^n + b

-- Proof problem: Given conditions, prove b / a = 1
theorem problem (n : ℕ) (a b : ℝ)
  (h1 : smallest_n_gt_3 n)
  (h2 : 0 < a) (h3 : 0 < b)
  (h4 : equation_holds a b n) :
  b / a = 1 :=
by
  sorry

end problem_l2291_229189


namespace diminishing_allocation_proof_l2291_229124

noncomputable def diminishing_allocation_problem : Prop :=
  ∃ (a b m : ℝ), 
  a = 0.2 ∧
  b * (1 - a)^2 = 80 ∧
  b * (1 - a) + b * (1 - a)^3 = 164 ∧
  b + 80 + 164 = m ∧
  m = 369

theorem diminishing_allocation_proof : diminishing_allocation_problem :=
by
  sorry

end diminishing_allocation_proof_l2291_229124


namespace price_of_red_car_l2291_229104

noncomputable def car_price (total_amount loan_amount interest_rate : ℝ) : ℝ :=
  loan_amount + (total_amount - loan_amount) / (1 + interest_rate)

theorem price_of_red_car :
  car_price 38000 20000 0.15 = 35000 :=
by sorry

end price_of_red_car_l2291_229104


namespace percent_forgot_group_B_l2291_229183

def num_students_group_A : ℕ := 20
def num_students_group_B : ℕ := 80
def percent_forgot_group_A : ℚ := 0.20
def total_percent_forgot : ℚ := 0.16

/--
There are two groups of students in the sixth grade. 
There are 20 students in group A, and 80 students in group B. 
On a particular day, 20% of the students in group A forget their homework, and a certain 
percentage of the students in group B forget their homework. 
Then, 16% of the sixth graders forgot their homework. 
Prove that 15% of the students in group B forgot their homework.
-/
theorem percent_forgot_group_B : 
  let num_forgot_group_A := percent_forgot_group_A * num_students_group_A
  let total_students := num_students_group_A + num_students_group_B
  let total_forgot := total_percent_forgot * total_students
  let num_forgot_group_B := total_forgot - num_forgot_group_A
  let percent_forgot_group_B := (num_forgot_group_B / num_students_group_B) * 100
  percent_forgot_group_B = 15 :=
by {
  sorry
}

end percent_forgot_group_B_l2291_229183


namespace second_player_wins_l2291_229182

def num_of_piles_initial := 3
def total_stones := 10 + 15 + 20
def num_of_piles_final := total_stones
def total_moves := num_of_piles_final - num_of_piles_initial

theorem second_player_wins : total_moves % 2 = 0 :=
sorry

end second_player_wins_l2291_229182


namespace find_apron_cost_l2291_229129

-- Definitions used in the conditions
variables (hand_mitts cost small_knife utensils apron : ℝ)
variables (nieces : ℕ)
variables (total_cost_before_discount total_cost_after_discount : ℝ)

-- Conditions given
def conditions := 
  hand_mitts = 14 ∧ 
  utensils = 10 ∧ 
  small_knife = 2 * utensils ∧
  (total_cost_before_discount : ℝ) = (3 * hand_mitts + 3 * utensils + 3 * small_knife + 3 * apron) ∧
  (total_cost_after_discount : ℝ) = 135 ∧
  total_cost_before_discount * 0.75 = total_cost_after_discount ∧
  nieces = 3

-- Theorem statement (proof problem)
theorem find_apron_cost (h : conditions hand_mitts utensils small_knife apron nieces total_cost_before_discount total_cost_after_discount) : 
  apron = 16 :=
by 
  sorry

end find_apron_cost_l2291_229129


namespace apples_left_is_correct_l2291_229187

-- Definitions for the conditions
def blue_apples : ℕ := 5
def yellow_apples : ℕ := 2 * blue_apples
def total_apples : ℕ := blue_apples + yellow_apples
def apples_given_to_son : ℚ := 1 / 5 * total_apples
def apples_left : ℚ := total_apples - apples_given_to_son

-- The main statement to be proven
theorem apples_left_is_correct : apples_left = 12 := by
  sorry

end apples_left_is_correct_l2291_229187


namespace problem1_problem2_l2291_229113

-- Let's define the first problem statement in Lean
theorem problem1 : 2 - 7 * (-3) + 10 + (-2) = 31 := sorry

-- Let's define the second problem statement in Lean
theorem problem2 : -1^2022 + 24 + (-2)^3 - 3^2 * (-1/3)^2 = 14 := sorry

end problem1_problem2_l2291_229113


namespace angle_Z_is_90_l2291_229188

theorem angle_Z_is_90 (X Y Z : ℝ) (h_sum_XY : X + Y = 90) (h_Y_is_2X : Y = 2 * X) (h_sum_angles : X + Y + Z = 180) : Z = 90 :=
by
  sorry

end angle_Z_is_90_l2291_229188


namespace ring_groups_in_first_tree_l2291_229158

variable (n : ℕ) (y1 y2 : ℕ) (t : ℕ) (groupsPerYear : ℕ := 6)

-- each tree's rings are in groups of 2 fat rings and 4 thin rings, representing 6 years
def group_represents_years : ℕ := groupsPerYear

-- second tree has 40 ring groups, so it is 40 * 6 = 240 years old
def second_tree_groups : ℕ := 40

-- first tree is 180 years older, so its age in years
def first_tree_age : ℕ := (second_tree_groups * groupsPerYear) + 180

-- number of ring groups in the first tree
def number_of_ring_groups_in_first_tree := first_tree_age / groupsPerYear

theorem ring_groups_in_first_tree :
  number_of_ring_groups_in_first_tree = 70 :=
by
  sorry

end ring_groups_in_first_tree_l2291_229158


namespace total_pencils_l2291_229106

def pencils_in_rainbow_box : ℕ := 7
def total_people : ℕ := 8

theorem total_pencils : pencils_in_rainbow_box * total_people = 56 := by
  sorry

end total_pencils_l2291_229106


namespace smallest_N_for_triangle_sides_l2291_229153

theorem smallest_N_for_triangle_sides (a b c : ℝ) (h_triangle : a + b > c) (h_a_ne_b : a ≠ b) : (a^2 + b^2) / c^2 < 1 := 
sorry

end smallest_N_for_triangle_sides_l2291_229153


namespace dot_product_min_value_in_triangle_l2291_229108

noncomputable def dot_product_min_value (a b c : ℝ) (angleA : ℝ) : ℝ :=
  b * c * Real.cos angleA

theorem dot_product_min_value_in_triangle (b c : ℝ) (hyp1 : 0 ≤ b) (hyp2 : 0 ≤ c) 
  (hyp3 : b^2 + c^2 + b * c = 16) (hyp4 : Real.cos (2 * Real.pi / 3) = -1 / 2) : 
  ∃ (p : ℝ), p = dot_product_min_value 4 b c (2 * Real.pi / 3) ∧ p = -8 / 3 :=
by
  sorry

end dot_product_min_value_in_triangle_l2291_229108


namespace sum_a_b_neg1_l2291_229117

-- Define the problem using the given condition
theorem sum_a_b_neg1 (a b : ℝ) (h : |a + 3| + (b - 2) ^ 2 = 0) : a + b = -1 := 
by
  sorry

end sum_a_b_neg1_l2291_229117


namespace work_completion_days_l2291_229151

theorem work_completion_days
    (A : ℝ) (B : ℝ) (h1 : 1 / A + 1 / B = 1 / 10)
    (h2 : B = 35) :
    A = 14 :=
by
  sorry

end work_completion_days_l2291_229151


namespace largest_divisor_of_polynomial_l2291_229147

theorem largest_divisor_of_polynomial (n : ℕ) (h : n % 2 = 0) : 
  105 ∣ (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) :=
sorry

end largest_divisor_of_polynomial_l2291_229147


namespace hourly_wage_l2291_229125

theorem hourly_wage (reps : ℕ) (hours_per_day : ℕ) (days : ℕ) (total_payment : ℕ) :
  reps = 50 →
  hours_per_day = 8 →
  days = 5 →
  total_payment = 28000 →
  (total_payment / (reps * hours_per_day * days) : ℕ) = 14 :=
by
  intros h_reps h_hours_per_day h_days h_total_payment
  -- Now the proof steps can be added here
  sorry

end hourly_wage_l2291_229125


namespace arithmetic_sequence_sum_9_l2291_229185

theorem arithmetic_sequence_sum_9 :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  (∀ n, a n = 2 + n * d) ∧ d ≠ 0 ∧ (2 : ℝ) + 2 * d ≠ 0 ∧ (2 + 5 * d) ≠ 0 ∧ d = 0.5 →
  (2 + 2 * d)^2 = 2 * (2 + 5 * d) →
  (9 * 2 + (9 * 8 / 2) * 0.5) = 36 :=
by
  intros a d h1 h2
  sorry

end arithmetic_sequence_sum_9_l2291_229185


namespace problem_a_l2291_229150

theorem problem_a (x a : ℝ) (h : (x + a) * (x + 2 * a) * (x + 3 * a) * (x + 4 * a) = 3 * a^4) :
  x = (-5 * a + a * Real.sqrt 37) / 2 ∨ x = (-5 * a - a * Real.sqrt 37) / 2 :=
by
  sorry

end problem_a_l2291_229150


namespace average_molecular_weight_benzoic_acid_l2291_229132

def atomic_mass_C : ℝ := (12 * 0.9893) + (13 * 0.0107)
def atomic_mass_H : ℝ := (1 * 0.99985) + (2 * 0.00015)
def atomic_mass_O : ℝ := (16 * 0.99762) + (17 * 0.00038) + (18 * 0.00200)

theorem average_molecular_weight_benzoic_acid :
  (7 * atomic_mass_C) + (6 * atomic_mass_H) + (2 * atomic_mass_O) = 123.05826 :=
by {
  sorry
}

end average_molecular_weight_benzoic_acid_l2291_229132


namespace Tim_marble_count_l2291_229143

theorem Tim_marble_count (Fred_marbles : ℕ) (Tim_marbles : ℕ) (h1 : Fred_marbles = 110) (h2 : Fred_marbles = 22 * Tim_marbles) : 
  Tim_marbles = 5 := 
sorry

end Tim_marble_count_l2291_229143


namespace cone_radius_of_surface_area_and_lateral_surface_unfolds_to_semicircle_l2291_229168

theorem cone_radius_of_surface_area_and_lateral_surface_unfolds_to_semicircle
  (surface_area : ℝ) (lateral_surface_unfolds_to_semicircle : Prop) :
  surface_area = 12 * Real.pi → lateral_surface_unfolds_to_semicircle → ∃ r : ℝ, r = 2 := by
  sorry

end cone_radius_of_surface_area_and_lateral_surface_unfolds_to_semicircle_l2291_229168


namespace same_type_as_target_l2291_229109

-- Definitions of the polynomials
def optionA (a b : ℝ) : ℝ := a^2 * b
def optionB (a b : ℝ) : ℝ := -2 * a * b^2
def optionC (a b : ℝ) : ℝ := a * b
def optionD (a b c : ℝ) : ℝ := a * b^2 * c

-- Definition of the target polynomial type
def target (a b : ℝ) : ℝ := a * b^2

-- Statement: Option B is of the same type as target
theorem same_type_as_target (a b : ℝ) : optionB a b = -2 * target a b := 
sorry

end same_type_as_target_l2291_229109


namespace fraction_distance_walked_by_first_class_l2291_229193

namespace CulturalCenterProblem

def walking_speed : ℝ := 4
def bus_speed_with_students : ℝ := 40
def bus_speed_empty : ℝ := 60

theorem fraction_distance_walked_by_first_class :
  ∃ (x : ℝ), 
    (x / walking_speed) = ((1 - x) / bus_speed_with_students) + ((1 - 2 * x) / bus_speed_empty)
    ∧ x = 5 / 37 :=
by
  sorry

end CulturalCenterProblem

end fraction_distance_walked_by_first_class_l2291_229193


namespace no_such_a_and_sequence_exists_l2291_229166

theorem no_such_a_and_sequence_exists :
  ¬∃ (a : ℝ) (a_pos : 0 < a ∧ a < 1) (a_seq : ℕ → ℝ), (∀ n : ℕ, 0 < a_seq n) ∧ (∀ n : ℕ, 1 + a_seq (n + 1) ≤ a_seq n + (a / (n + 1)) * a_seq n) :=
by
  sorry

end no_such_a_and_sequence_exists_l2291_229166


namespace car_speed_ratio_l2291_229195

theorem car_speed_ratio 
  (t D : ℝ) 
  (v_alpha v_beta : ℝ)
  (H1 : (v_alpha + v_beta) * t = D)
  (H2 : v_alpha * 4 = D - v_alpha * t)
  (H3 : v_beta * 1 = D - v_beta * t) : 
  v_alpha / v_beta = 2 :=
by
  sorry

end car_speed_ratio_l2291_229195


namespace exists_m_with_totient_ratio_l2291_229191

variable (α β : ℝ)

theorem exists_m_with_totient_ratio (h0 : 0 ≤ α) (h1 : α < β) (h2 : β ≤ 1) :
  ∃ m : ℕ, α < (Nat.totient m : ℝ) / m ∧ (Nat.totient m : ℝ) / m < β := 
  sorry

end exists_m_with_totient_ratio_l2291_229191


namespace range_of_a_l2291_229184

theorem range_of_a (a : ℝ) : (∃ x y : ℝ, x - y + 1 = 0 ∧ (x - a)^2 + y^2 = 2) ↔ -3 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l2291_229184


namespace part_a_prob_part_b_expected_time_l2291_229159

/--  
Let total_suitcases be 200.
Let business_suitcases be 10.
Let total_wait_time be 120.
Let arrival_interval be 2.
--/

def total_suitcases : ℕ := 200
def business_suitcases : ℕ := 10
def total_wait_time : ℕ := 120 
def arrival_interval : ℕ := 2

def two_minutes_suitcases : ℕ := total_wait_time / arrival_interval
def prob_last_suitcase_at_n_minutes (n : ℕ) : ℚ := 
  (Nat.choose (two_minutes_suitcases - 1) (business_suitcases - 1) : ℚ) / 
  (Nat.choose total_suitcases business_suitcases : ℚ)

theorem part_a_prob : 
  prob_last_suitcase_at_n_minutes 2 = 
  (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) := sorry

noncomputable def expected_position_last_suitcase (total_pos : ℕ) (suitcases_per_group : ℕ) : ℚ :=
  (total_pos * business_suitcases : ℚ) / (business_suitcases + 1)

theorem part_b_expected_time : 
  expected_position_last_suitcase 201 11 * arrival_interval = 
  (4020 : ℚ) / 11 := sorry

end part_a_prob_part_b_expected_time_l2291_229159


namespace fourth_bus_people_difference_l2291_229157

def bus1_people : Nat := 12
def bus2_people : Nat := 2 * bus1_people
def bus3_people : Nat := bus2_people - 6
def total_people : Nat := 75
def bus4_people : Nat := total_people - (bus1_people + bus2_people + bus3_people)
def difference_people : Nat := bus4_people - bus1_people

theorem fourth_bus_people_difference : difference_people = 9 := by
  -- Proof logic here
  sorry

end fourth_bus_people_difference_l2291_229157


namespace min_regions_l2291_229154

namespace CircleDivision

def k := 12

-- Theorem statement: Given exactly 12 points where at least two circles intersect,
-- the minimum number of regions into which these circles divide the plane is 14.
theorem min_regions (k := 12) : ∃ R, R = 14 :=
by
  let R := 14
  existsi R
  exact rfl

end min_regions_l2291_229154


namespace length_of_first_platform_l2291_229128

-- Definitions corresponding to conditions
def length_train := 310
def time_first_platform := 15
def length_second_platform := 250
def time_second_platform := 20

-- Time-speed relationship
def speed_first_platform (L : ℕ) : ℚ := (length_train + L) / time_first_platform
def speed_second_platform : ℚ := (length_train + length_second_platform) / time_second_platform

-- Theorem to prove length of first platform
theorem length_of_first_platform (L : ℕ) (h : speed_first_platform L = speed_second_platform) : L = 110 :=
by
  sorry

end length_of_first_platform_l2291_229128


namespace corrected_mean_is_40_point_6_l2291_229121

theorem corrected_mean_is_40_point_6 
  (mean_original : ℚ) (num_observations : ℕ) (wrong_observation : ℚ) (correct_observation : ℚ) :
  num_observations = 50 → mean_original = 40 → wrong_observation = 15 → correct_observation = 45 →
  ((mean_original * num_observations + (correct_observation - wrong_observation)) / num_observations = 40.6 : Prop) :=
by intros; sorry

end corrected_mean_is_40_point_6_l2291_229121


namespace amount_received_by_a_l2291_229155

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

end amount_received_by_a_l2291_229155


namespace train_avg_speed_l2291_229122

variable (x : ℝ)

def avg_speed_of_train (x : ℝ) : ℝ := 3

theorem train_avg_speed (h : x > 0) : avg_speed_of_train x / (x / 7.5) = 22.5 :=
  sorry

end train_avg_speed_l2291_229122


namespace problem_solution_l2291_229126

theorem problem_solution (x : ℝ) :
    (x^2 / (x - 2) ≥ (3 / (x + 2)) + (7 / 5)) →
    (x ∈ Set.Ioo (-2 : ℝ) 2 ∪ Set.Ioi (2 : ℝ)) :=
by
  intro h
  sorry

end problem_solution_l2291_229126


namespace contributions_before_john_l2291_229176

theorem contributions_before_john
  (A : ℝ) (n : ℕ)
  (h1 : 1.5 * A = 75)
  (h2 : (n * A + 150) / (n + 1) = 75) :
  n = 3 :=
by
  sorry

end contributions_before_john_l2291_229176


namespace kiwis_to_apples_l2291_229103

theorem kiwis_to_apples :
  (1 / 4) * 20 = 10 → (3 / 4) * 12 * (2 / 5) = 18 :=
by
  sorry

end kiwis_to_apples_l2291_229103


namespace airplane_children_l2291_229144

theorem airplane_children (total_passengers men women children : ℕ) 
    (h1 : total_passengers = 80) 
    (h2 : men = women) 
    (h3 : men = 30) 
    (h4 : total_passengers = men + women + children) : 
    children = 20 := 
by
    -- We need to show that the number of children is 20.
    sorry

end airplane_children_l2291_229144


namespace sum_of_surface_points_l2291_229192

theorem sum_of_surface_points
  (n : ℕ) (h_n : n = 2012) 
  (total_sum : ℕ) (h_total : total_sum = n * 21)
  (matching_points_sum : ℕ) (h_matching : matching_points_sum = (n - 1) * 7)
  (x : ℕ) (h_x_range : 1 ≤ x ∧ x ≤ 6) :
  (total_sum - matching_points_sum + 2 * x = 28177 ∨
   total_sum - matching_points_sum + 2 * x = 28179 ∨
   total_sum - matching_points_sum + 2 * x = 28181 ∨
   total_sum - matching_points_sum + 2 * x = 28183 ∨
   total_sum - matching_points_sum + 2 * x = 28185 ∨
   total_sum - matching_points_sum + 2 * x = 28187) :=
by sorry

end sum_of_surface_points_l2291_229192


namespace total_clothing_donated_l2291_229148

-- Definition of the initial donation by Adam
def adam_initial_donation : Nat := 4 + 4 + 4*2 + 20 -- 4 pairs of pants, 4 jumpers, 4 pajama sets (8 items), 20 t-shirts

-- Adam's friends' total donation
def friends_donation : Nat := 3 * adam_initial_donation

-- Adam's donation after keeping half
def adam_final_donation : Nat := adam_initial_donation / 2

-- Total donation being the sum of Adam's and friends' donations
def total_donation : Nat := adam_final_donation + friends_donation

-- The statement to prove
theorem total_clothing_donated : total_donation = 126 := by
  -- This is skipped as per instructions
  sorry

end total_clothing_donated_l2291_229148


namespace marks_for_correct_answer_l2291_229179

theorem marks_for_correct_answer (x : ℕ) 
  (total_marks : ℤ) (total_questions : ℕ) (correct_answers : ℕ) 
  (wrong_mark : ℤ) (result : ℤ) :
  total_marks = result →
  total_questions = 70 →
  correct_answers = 27 →
  (-1) * (total_questions - correct_answers) = wrong_mark →
  total_marks = (correct_answers : ℤ) * (x : ℤ) + wrong_mark →
  x = 3 := 
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end marks_for_correct_answer_l2291_229179


namespace wendy_walked_l2291_229194

theorem wendy_walked (x : ℝ) (h1 : 19.83 = x + 10.67) : x = 9.16 :=
sorry

end wendy_walked_l2291_229194


namespace sector_area_ratio_l2291_229178

theorem sector_area_ratio (angle_AOE angle_FOB : ℝ) (h1 : angle_AOE = 40) (h2 : angle_FOB = 60) : 
  (180 - angle_AOE - angle_FOB) / 360 = 2 / 9 :=
by
  sorry

end sector_area_ratio_l2291_229178


namespace knives_percentage_l2291_229115

-- Definitions based on conditions
def initial_knives : ℕ := 6
def initial_forks : ℕ := 12
def initial_spoons : ℕ := 3 * initial_knives
def traded_knives : ℕ := 10
def traded_spoons : ℕ := 6

-- Definitions for calculations
def final_knives : ℕ := initial_knives + traded_knives
def final_spoons : ℕ := initial_spoons - traded_spoons
def total_silverware : ℕ := final_knives + final_spoons + initial_forks

-- Theorem to prove the percentage of knives
theorem knives_percentage : (final_knives * 100) / total_silverware = 40 := by
  sorry

end knives_percentage_l2291_229115


namespace remainder_of_product_mod_5_l2291_229181

theorem remainder_of_product_mod_5 :
  (2685 * 4932 * 91406) % 5 = 0 :=
by
  sorry

end remainder_of_product_mod_5_l2291_229181


namespace total_arrangements_correct_adjacent_males_correct_descending_heights_correct_l2291_229161

-- Total number of different arrangements of 3 male students and 2 female students.
def total_arrangements (males females : ℕ) : ℕ :=
  (males + females).factorial

-- Number of arrangements where exactly two male students are adjacent.
def adjacent_males (males females : ℕ) : ℕ :=
  if males = 3 ∧ females = 2 then 72 else 0

-- Number of arrangements where male students of different heights are arranged from tallest to shortest.
def descending_heights (heights : Nat → ℕ) (males females : ℕ) : ℕ :=
  if males = 3 ∧ females = 2 then 20 else 0

-- Theorem statements corresponding to the questions.
theorem total_arrangements_correct : total_arrangements 3 2 = 120 := sorry

theorem adjacent_males_correct : adjacent_males 3 2 = 72 := sorry

theorem descending_heights_correct (heights : Nat → ℕ) : descending_heights heights 3 2 = 20 := sorry

end total_arrangements_correct_adjacent_males_correct_descending_heights_correct_l2291_229161


namespace bruce_purchased_mangoes_l2291_229102

noncomputable def calculate_mango_quantity (grapes_quantity : ℕ) (grapes_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) : ℕ :=
  let cost_of_grapes := grapes_quantity * grapes_rate
  let cost_of_mangoes := total_paid - cost_of_grapes
  cost_of_mangoes / mango_rate

theorem bruce_purchased_mangoes :
  calculate_mango_quantity 8 70 55 1055 = 9 :=
by
  sorry

end bruce_purchased_mangoes_l2291_229102
