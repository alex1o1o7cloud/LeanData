import Mathlib

namespace determine_p_l862_86241

theorem determine_p (p x1 x2 : ℝ) 
  (h_eq : ∀ x, x^2 + p * x + 3 = 0)
  (h_root_relation : x2 = 3 * x1)
  (h_vieta1 : x1 + x2 = -p)
  (h_vieta2 : x1 * x2 = 3) :
  p = 4 ∨ p = -4 := 
sorry

end determine_p_l862_86241


namespace no_three_segments_form_triangle_l862_86206

theorem no_three_segments_form_triangle :
  ∃ (a : Fin 10 → ℕ), ∀ {i j k : Fin 10}, i < j → j < k → a i + a j ≤ a k :=
by
  sorry

end no_three_segments_form_triangle_l862_86206


namespace ticket_is_five_times_soda_l862_86209

variable (p_i p_r : ℝ)

theorem ticket_is_five_times_soda
  (h1 : 6 * p_i + 20 * p_r = 50)
  (h2 : 6 * p_r = p_i + p_r) : p_i = 5 * p_r :=
sorry

end ticket_is_five_times_soda_l862_86209


namespace sequence_a4_value_l862_86290

theorem sequence_a4_value :
  ∃ (a : ℕ → ℕ), (a 1 = 1) ∧ (∀ n, a (n+1) = 2 * a n + 1) ∧ (a 4 = 15) :=
by
  sorry

end sequence_a4_value_l862_86290


namespace math_problem_l862_86217

theorem math_problem : (-1: ℝ)^2 + (1/3: ℝ)^0 = 2 := by
  sorry

end math_problem_l862_86217


namespace tangent_slope_is_4_l862_86203

theorem tangent_slope_is_4 (x y : ℝ) (h_curve : y = x^4) (h_slope : (deriv (fun x => x^4) x) = 4) :
    (x, y) = (1, 1) :=
by
  -- Place proof here
  sorry

end tangent_slope_is_4_l862_86203


namespace factorial_of_6_is_720_l862_86295

theorem factorial_of_6_is_720 : (Nat.factorial 6) = 720 := by
  sorry

end factorial_of_6_is_720_l862_86295


namespace octahedron_plane_pairs_l862_86280

-- A regular octahedron has 12 edges.
def edges_octahedron : ℕ := 12

-- Each edge determines a plane with 8 other edges.
def pairs_with_each_edge : ℕ := 8

-- The number of unordered pairs of edges that determine a plane
theorem octahedron_plane_pairs : (edges_octahedron * pairs_with_each_edge) / 2 = 48 :=
by
  -- sorry is used to skip the proof
  sorry

end octahedron_plane_pairs_l862_86280


namespace exists_point_on_graph_of_quadratic_l862_86242

-- Define the condition for the discriminant to be zero
def is_single_root (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c = 0

-- Define a function representing a quadratic polynomial
def quadratic_poly (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- The main statement
theorem exists_point_on_graph_of_quadratic (b c : ℝ) 
  (h : is_single_root 1 b c) :
  ∃ (p q : ℝ), q = (p^2) / 4 ∧ is_single_root 1 p q :=
sorry

end exists_point_on_graph_of_quadratic_l862_86242


namespace width_of_deck_l862_86216

noncomputable def length : ℝ := 30
noncomputable def cost_per_sqft_construction : ℝ := 3
noncomputable def cost_per_sqft_sealant : ℝ := 1
noncomputable def total_cost : ℝ := 4800
noncomputable def total_cost_per_sqft : ℝ := cost_per_sqft_construction + cost_per_sqft_sealant

theorem width_of_deck (w : ℝ) 
  (h1 : length * w * total_cost_per_sqft = total_cost) : 
  w = 40 := 
sorry

end width_of_deck_l862_86216


namespace wire_leftover_length_l862_86289

-- Define given conditions as variables/constants
def initial_wire_length : ℝ := 60
def side_length : ℝ := 9
def sides_in_square : ℕ := 4

-- Define the theorem: prove leftover wire length is 24 after creating the square
theorem wire_leftover_length :
  initial_wire_length - sides_in_square * side_length = 24 :=
by
  -- proof steps are not required, so we use sorry to indicate where the proof should be
  sorry

end wire_leftover_length_l862_86289


namespace projection_correct_l862_86259

theorem projection_correct :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-1, 3)
  -- Definition of dot product for 2D vectors
  let dot (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2
  -- Definition of projection of a onto b
  let proj := (dot a b / (b.1^2 + b.2^2)) • b
  proj = (-1 / 2, 3 / 2) :=
by
  sorry

end projection_correct_l862_86259


namespace num_sheets_in_stack_l862_86232

-- Definitions coming directly from the conditions
def thickness_ream := 4 -- cm
def num_sheets_ream := 400
def height_stack := 10 -- cm

-- The final proof statement
theorem num_sheets_in_stack : (height_stack / (thickness_ream / num_sheets_ream)) = 1000 :=
by
  sorry

end num_sheets_in_stack_l862_86232


namespace profit_without_discount_l862_86244

noncomputable def cost_price : ℝ := 100
noncomputable def profit_percentage_with_discount : ℝ := 44
noncomputable def discount : ℝ := 4

theorem profit_without_discount (CP MP SP : ℝ) (h_CP : CP = cost_price) (h_pwpd : profit_percentage_with_discount = 44) (h_discount : discount = 4) (h_SP : SP = CP * (1 + profit_percentage_with_discount / 100)) (h_MP : SP = MP * (1 - discount / 100)) :
  ((MP - CP) / CP * 100) = 50 :=
by
  sorry

end profit_without_discount_l862_86244


namespace find_a_l862_86247

noncomputable def hyperbola_eccentricity (a : ℝ) : ℝ := (Real.sqrt (a^2 + 3)) / a

theorem find_a (a : ℝ) (h : a > 0) (hexp : hyperbola_eccentricity a = 2) : a = 1 :=
by
  sorry

end find_a_l862_86247


namespace intersection_of_sets_l862_86229

open Set Real

theorem intersection_of_sets :
  let A := {x : ℝ | x^2 - 2*x - 3 < 0}
  let B := {y : ℝ | ∃ (x : ℝ), y = sin x}
  A ∩ B = Ioc (-1) 1 := by
  sorry

end intersection_of_sets_l862_86229


namespace infinite_coprime_binom_l862_86278

theorem infinite_coprime_binom (k l : ℕ) (hk : k > 0) (hl : l > 0) : 
  ∃ᶠ m in atTop, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1 := by
sorry

end infinite_coprime_binom_l862_86278


namespace pascals_triangle_contains_47_once_l862_86246

theorem pascals_triangle_contains_47_once (n : ℕ) : 
  (∃ k, k ≤ n ∧ Nat.choose n k = 47) ↔ n = 47 := by
  sorry

end pascals_triangle_contains_47_once_l862_86246


namespace fraction_of_earnings_spent_on_candy_l862_86275

theorem fraction_of_earnings_spent_on_candy :
  let candy_bars_cost := 2 * 0.75
  let lollipops_cost := 4 * 0.25
  let total_candy_cost := candy_bars_cost + lollipops_cost
  let earnings_per_driveway := 1.5
  let total_earnings := 10 * earnings_per_driveway
  total_candy_cost / total_earnings = 1 / 6 :=
by
  let candy_bars_cost := 2 * 0.75
  let lollipops_cost := 4 * 0.25
  let total_candy_cost := candy_bars_cost + lollipops_cost
  let earnings_per_driveway := 1.5
  let total_earnings := 10 * earnings_per_driveway
  have h : total_candy_cost / total_earnings = 1 / 6 := by sorry
  exact h

end fraction_of_earnings_spent_on_candy_l862_86275


namespace tan_alpha_value_l862_86225

theorem tan_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin α * Real.cos α = 1 / 4) :
  Real.tan α = 2 - Real.sqrt 3 ∨ Real.tan α = 2 + Real.sqrt 3 :=
sorry

end tan_alpha_value_l862_86225


namespace magnitude_of_difference_between_roots_l862_86215

variable (α β m : ℝ)

theorem magnitude_of_difference_between_roots
    (hαβ_root : ∀ x, x^2 - 2 * m * x + m^2 - 4 = 0 → (x = α ∨ x = β)) :
    |α - β| = 4 := by
  sorry

end magnitude_of_difference_between_roots_l862_86215


namespace min_time_to_complete_tasks_l862_86200

-- Define the conditions as individual time durations for each task in minutes
def bed_making_time : ℕ := 3
def teeth_washing_time : ℕ := 4
def water_boiling_time : ℕ := 10
def breakfast_time : ℕ := 7
def dish_washing_time : ℕ := 1
def backpack_organizing_time : ℕ := 2
def milk_making_time : ℕ := 1

-- Define the total minimum time required to complete all tasks
def min_completion_time : ℕ := 18

-- A theorem stating that given the times for each task, the minimum completion time is 18 minutes
theorem min_time_to_complete_tasks :
  bed_making_time + teeth_washing_time + water_boiling_time + 
  breakfast_time + dish_washing_time + backpack_organizing_time + milk_making_time - 
  (bed_making_time + teeth_washing_time + backpack_organizing_time + milk_making_time) <=
  min_completion_time := by
  sorry

end min_time_to_complete_tasks_l862_86200


namespace dave_spent_on_books_l862_86219

-- Define the cost of books in each category without any discounts or taxes
def cost_animal_books : ℝ := 8 * 10
def cost_outer_space_books : ℝ := 6 * 12
def cost_train_books : ℝ := 9 * 8
def cost_history_books : ℝ := 4 * 15
def cost_science_books : ℝ := 5 * 18

-- Define the discount and tax rates
def discount_animal_books : ℝ := 0.10
def tax_science_books : ℝ := 0.15

-- Apply the discount to animal books
def discounted_cost_animal_books : ℝ := cost_animal_books * (1 - discount_animal_books)

-- Apply the tax to science books
def final_cost_science_books : ℝ := cost_science_books * (1 + tax_science_books)

-- Calculate the total cost of all books after discounts and taxes
def total_cost : ℝ := discounted_cost_animal_books 
                  + cost_outer_space_books
                  + cost_train_books
                  + cost_history_books
                  + final_cost_science_books

theorem dave_spent_on_books : total_cost = 379.5 := by
  sorry

end dave_spent_on_books_l862_86219


namespace log_identity_l862_86291

theorem log_identity :
  (Real.log 25 / Real.log 10) - 2 * (Real.log (1 / 2) / Real.log 10) = 2 :=
by
  sorry

end log_identity_l862_86291


namespace total_songs_sung_l862_86257

def total_minutes := 80
def intermission_minutes := 10
def long_song_minutes := 10
def short_song_minutes := 5

theorem total_songs_sung : 
  (total_minutes - intermission_minutes - long_song_minutes) / short_song_minutes + 1 = 13 := 
by 
  sorry

end total_songs_sung_l862_86257


namespace phantom_additional_money_needed_l862_86274

theorem phantom_additional_money_needed
  (given_money : ℕ)
  (black_inks_cost : ℕ)
  (red_inks_cost : ℕ)
  (yellow_inks_cost : ℕ)
  (blue_inks_cost : ℕ)
  (total_money_needed : ℕ)
  (additional_money_needed : ℕ) :
  given_money = 50 →
  black_inks_cost = 3 * 12 →
  red_inks_cost = 4 * 16 →
  yellow_inks_cost = 3 * 14 →
  blue_inks_cost = 2 * 17 →
  total_money_needed = black_inks_cost + red_inks_cost + yellow_inks_cost + blue_inks_cost →
  additional_money_needed = total_money_needed - given_money →
  additional_money_needed = 126 :=
by
  intros h_given_money h_black h_red h_yellow h_blue h_total h_additional
  sorry

end phantom_additional_money_needed_l862_86274


namespace volume_cube_box_for_pyramid_l862_86264

theorem volume_cube_box_for_pyramid (h_pyramid : height_of_pyramid = 18) 
  (base_side_pyramid : side_of_square_base = 15) : 
  volume_of_box = 18^3 :=
by
  sorry

end volume_cube_box_for_pyramid_l862_86264


namespace sets_equal_l862_86218

def M := { u | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l }
def N := { u | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r }

theorem sets_equal : M = N :=
by sorry

end sets_equal_l862_86218


namespace cost_of_dozen_pens_l862_86255

theorem cost_of_dozen_pens
  (cost_three_pens_five_pencils : ℝ)
  (cost_one_pen : ℝ)
  (pen_to_pencil_ratio : ℝ)
  (h1 : 3 * cost_one_pen + 5 * (cost_three_pens_five_pencils / 8) = 260)
  (h2 : cost_one_pen = 65)
  (h3 : cost_one_pen / (cost_three_pens_five_pencils / 8) = 5/1)
  : 12 * cost_one_pen = 780 := by
    sorry

end cost_of_dozen_pens_l862_86255


namespace sum_first_n_odd_eq_n_squared_l862_86297

theorem sum_first_n_odd_eq_n_squared (n : ℕ) : (Finset.sum (Finset.range n) (fun k => (2 * k + 1)) = n^2) := sorry

end sum_first_n_odd_eq_n_squared_l862_86297


namespace ryan_learning_hours_l862_86277

theorem ryan_learning_hours :
  ∃ hours : ℕ, 
    (∀ e_hrs : ℕ, e_hrs = 2) → 
    (∃ c_hrs : ℕ, c_hrs = hours) → 
    (∀ s_hrs : ℕ, s_hrs = 4) → 
    hours = 4 + 1 :=
by
  sorry

end ryan_learning_hours_l862_86277


namespace sum_of_numbers_facing_up_is_4_probability_l862_86251

-- Definition of a uniform dice with faces numbered 1 to 6
def dice_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Definition of the sample space when the dice is thrown twice
def sample_space : Finset (ℕ × ℕ) := Finset.product dice_faces dice_faces

-- Definition of the event where the sum of the numbers is 4
def event_sum_4 : Finset (ℕ × ℕ) := sample_space.filter (fun pair => pair.1 + pair.2 = 4)

-- The number of favorable outcomes
def favorable_outcomes : ℕ := event_sum_4.card

-- The total number of possible outcomes
def total_outcomes : ℕ := sample_space.card

-- The probability of the event
def probability_event_sum_4 : ℚ := favorable_outcomes / total_outcomes

theorem sum_of_numbers_facing_up_is_4_probability :
  probability_event_sum_4 = 1 / 12 :=
by
  sorry

end sum_of_numbers_facing_up_is_4_probability_l862_86251


namespace breaststroke_hours_correct_l862_86243

namespace Swimming

def total_required_hours : ℕ := 1500
def backstroke_hours : ℕ := 50
def butterfly_hours : ℕ := 121
def monthly_freestyle_sidestroke_hours : ℕ := 220
def months : ℕ := 6

def calculated_total_hours : ℕ :=
  backstroke_hours + butterfly_hours + (monthly_freestyle_sidestroke_hours * months)

def remaining_hours_to_breaststroke : ℕ :=
  total_required_hours - calculated_total_hours

theorem breaststroke_hours_correct :
  remaining_hours_to_breaststroke = 9 :=
by
  sorry

end Swimming

end breaststroke_hours_correct_l862_86243


namespace arithmetic_sequence_and_sum_properties_l862_86288

noncomputable def a_n (n : ℕ) : ℤ := 30 - 2 * n
noncomputable def S_n (n : ℕ) : ℤ := -n^2 + 29 * n

theorem arithmetic_sequence_and_sum_properties :
  (a_n 3 = 24 ∧ a_n 6 = 18) ∧
  (∀ n : ℕ, (S_n n = (n * (a_n 1 + a_n n)) / 2) ∧ ((a_n 3 = 24 ∧ a_n 6 = 18) → ∀ n : ℕ, a_n n = 30 - 2 * n)) ∧
  (S_n 14 = 210) :=
by 
  -- Proof omitted.
  sorry

end arithmetic_sequence_and_sum_properties_l862_86288


namespace number_exceeds_twenty_percent_by_forty_l862_86265

theorem number_exceeds_twenty_percent_by_forty (x : ℝ) (h : x = 0.20 * x + 40) : x = 50 :=
by
  sorry

end number_exceeds_twenty_percent_by_forty_l862_86265


namespace common_number_in_sequences_l862_86234

theorem common_number_in_sequences (n m: ℕ) (a : ℕ)
    (h1 : a = 3 + 8 * n)
    (h2 : a = 5 + 9 * m)
    (h3 : 1 ≤ a ∧ a ≤ 200) : a = 131 :=
by
  sorry

end common_number_in_sequences_l862_86234


namespace probability_of_ace_ten_king_l862_86235

noncomputable def probability_first_ace_second_ten_third_king : ℚ :=
  (4/52) * (4/51) * (4/50)

theorem probability_of_ace_ten_king :
  probability_first_ace_second_ten_third_king = 2/16575 :=
by
  sorry

end probability_of_ace_ten_king_l862_86235


namespace certain_number_is_4_l862_86269

theorem certain_number_is_4 (x y C : ℝ) (h1 : 2 * x - y = C) (h2 : 6 * x - 3 * y = 12) : C = 4 :=
by
  -- Proof goes here
  sorry

end certain_number_is_4_l862_86269


namespace non_negative_dot_product_l862_86284

theorem non_negative_dot_product
  (a b c d e f g h : ℝ) :
  (a * c + b * d ≥ 0) ∨ (a * e + b * f ≥ 0) ∨ (a * g + b * h ≥ 0) ∨
  (c * e + d * f ≥ 0) ∨ (c * g + d * h ≥ 0) ∨ (e * g + f * h ≥ 0) :=
sorry

end non_negative_dot_product_l862_86284


namespace problem_statement_l862_86286

variables {x y x1 y1 a b c d : ℝ}

-- The main theorem statement
theorem problem_statement (h0 : ∀ (x y : ℝ), 6 * y ^ 2 = 2 * x ^ 3 + 3 * x ^ 2 + x) 
                           (h1 : x1 = a * x + b) 
                           (h2 : y1 = c * y + d) 
                           (h3 : y1 ^ 2 = x1 ^ 3 - 36 * x1) : 
                           a + b + c + d = 90 := sorry

end problem_statement_l862_86286


namespace original_number_l862_86292

theorem original_number (x : ℕ) : x * 16 = 3408 → x = 213 := by
  intro h
  sorry

end original_number_l862_86292


namespace probability_distribution_xi_l862_86262

theorem probability_distribution_xi (a : ℝ) (ξ : ℕ → ℝ) (h1 : ξ 1 = a / (1 * 2))
  (h2 : ξ 2 = a / (2 * 3)) (h3 : ξ 3 = a / (3 * 4)) (h4 : ξ 4 = a / (4 * 5))
  (h5 : (ξ 1) + (ξ 2) + (ξ 3) + (ξ 4) = 1) :
  ξ 1 + ξ 2 = 5 / 6 :=
by
  sorry

end probability_distribution_xi_l862_86262


namespace common_difference_of_arithmetic_sequence_l862_86273

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℝ) (d a1 : ℝ) (h1 : a 3 = a1 + 2 * d) (h2 : a 5 = a1 + 4 * d)
  (h3 : a 7 = a1 + 6 * d) (h4 : a 10 = a1 + 9 * d) (h5 : a 13 = a1 + 12 * d) (h6 : (a 3) + (a 5) = 2) (h7 : (a 7) + (a 10) + (a 13) = 9) :
  d = (1 / 3) := by
  sorry

end common_difference_of_arithmetic_sequence_l862_86273


namespace find_correct_four_digit_number_l862_86239

theorem find_correct_four_digit_number (N : ℕ) (misspelledN : ℕ) (misspelled_unit_digit_correction : ℕ) 
  (h1 : misspelledN = (N / 10) * 10 + 6)
  (h2 : N - misspelled_unit_digit_correction = (N / 10) * 10 - 7 + 9)
  (h3 : misspelledN - 57 = 1819) : N = 1879 :=
  sorry


end find_correct_four_digit_number_l862_86239


namespace range_of_a_l862_86222

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (3 * a - 2) * x + 6 * a - 1 else a ^ x

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (3 / 8 ≤ a ∧ a < 2 / 3) :=
by
  sorry

end range_of_a_l862_86222


namespace percent_increase_between_maintenance_checks_l862_86268

theorem percent_increase_between_maintenance_checks (original_time new_time : ℕ) (h_orig : original_time = 50) (h_new : new_time = 60) :
  ((new_time - original_time : ℚ) / original_time) * 100 = 20 := by
  sorry

end percent_increase_between_maintenance_checks_l862_86268


namespace game_remaining_sprite_color_l862_86279

theorem game_remaining_sprite_color (m n : ℕ) : 
  (∀ m n : ℕ, ∃ sprite : String, sprite = if n % 2 = 0 then "Red" else "Blue") :=
by sorry

end game_remaining_sprite_color_l862_86279


namespace problem_statement_l862_86230

def P := {x : ℤ | ∃ k : ℤ, x = 2 * k - 1}
def Q := {y : ℤ | ∃ n : ℤ, y = 2 * n}

theorem problem_statement (x y : ℤ) (hx : x ∈ P) (hy : y ∈ Q) :
  (x + y ∈ P) ∧ (x * y ∈ Q) :=
by
  sorry

end problem_statement_l862_86230


namespace plates_count_l862_86266

theorem plates_count (n : ℕ)
  (h1 : 500 < n)
  (h2 : n < 600)
  (h3 : n % 10 = 7)
  (h4 : n % 12 = 7) : n = 547 :=
sorry

end plates_count_l862_86266


namespace circle_chord_segments_l862_86223

theorem circle_chord_segments (r : ℝ) (ch : ℝ) (a : ℝ) :
  (r = 8) ∧ (ch = 12) ∧ (r^2 - a^2 = 36) →
  a = 2 * Real.sqrt 7 → ∃ (ak bk : ℝ), ak = 8 - 2 * Real.sqrt 7 ∧ bk = 8 + 2 * Real.sqrt 7 :=
by
  sorry

end circle_chord_segments_l862_86223


namespace find_x_l862_86248

noncomputable def a (x : ℝ) : ℝ × ℝ :=
  (Real.cos (3 * x / 2), Real.sin (3 * x / 2))

noncomputable def b (x : ℝ) : ℝ × ℝ :=
  (Real.cos (x / 2), -Real.sin (x / 2))

noncomputable def norm_sq (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

theorem find_x (x : ℝ) :
  (0 ≤ x ∧ x ≤ Real.pi)
  ∧ (norm_sq (a x) + norm_sq (b x) + 2 * ((a x).1 * (b x).1 + (a x).2 * (b x).2) = 1)
  → (x = Real.pi / 3 ∨ x = 2 * Real.pi / 3) :=
by
  intro h
  sorry

end find_x_l862_86248


namespace quadratic_solutions_l862_86227

theorem quadratic_solutions : ∀ x : ℝ, x^2 - 25 = 0 → (x = 5 ∨ x = -5) :=
by
  sorry

end quadratic_solutions_l862_86227


namespace number_of_real_roots_eq_3_eq_m_l862_86210

theorem number_of_real_roots_eq_3_eq_m {x m : ℝ} (h : ∀ x, x^2 - 2 * |x| + 2 = m) : m = 2 :=
sorry

end number_of_real_roots_eq_3_eq_m_l862_86210


namespace initial_average_is_16_l862_86256

def average_of_six_observations (A : ℝ) : Prop :=
  ∃ s : ℝ, s = 6 * A

def new_observation (A : ℝ) (new_obs : ℝ := 9) : Prop :=
  ∃ t : ℝ, t = 7 * (A - 1)

theorem initial_average_is_16 (A : ℝ) (new_obs : ℝ := 9) :
  (average_of_six_observations A) → (new_observation A new_obs) → A = 16 :=
by
  intro h1 h2
  sorry

end initial_average_is_16_l862_86256


namespace fourth_machine_works_for_12_hours_daily_l862_86228

noncomputable def hours_fourth_machine_works (m1_hours m1_production_rate: ℕ) (m2_hours m2_production_rate: ℕ) (price_per_kg: ℕ) (total_earning: ℕ) :=
  let m1_total_production := m1_hours * m1_production_rate
  let m1_total_output := 3 * m1_total_production
  let m1_revenue := m1_total_output * price_per_kg
  let remaining_revenue := total_earning - m1_revenue
  let m2_total_production := remaining_revenue / price_per_kg
  m2_total_production / m2_production_rate

theorem fourth_machine_works_for_12_hours_daily : hours_fourth_machine_works 23 2 (sorry) (sorry) 50 8100 = 12 := by
  sorry

end fourth_machine_works_for_12_hours_daily_l862_86228


namespace perfect_square_l862_86231

-- Define natural numbers m and n and the condition mn ∣ m^2 + n^2 + m
variables (m n : ℕ)

-- Define the condition as a hypothesis
def condition (m n : ℕ) : Prop := (m * n) ∣ (m ^ 2 + n ^ 2 + m)

-- The main theorem statement: if the condition holds, then m is a perfect square
theorem perfect_square (m n : ℕ) (h : condition m n) : ∃ k : ℕ, m = k ^ 2 :=
sorry

end perfect_square_l862_86231


namespace extreme_values_l862_86220

-- Define the function f(x) with symbolic constants a and b
def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x

-- Given conditions
def intersects_at_1_0 (a b : ℝ) : Prop := (f 1 a b = 0)
def derivative_at_1_0 (a b : ℝ) : Prop := (3 - 2 * a - b = 0)

-- Main theorem statement
theorem extreme_values (a b : ℝ) (h1 : intersects_at_1_0 a b) (h2 : derivative_at_1_0 a b) :
  (∀ x, f x a b ≤ 4 / 27) ∧ (∀ x, 0 ≤ f x a b) :=
sorry

end extreme_values_l862_86220


namespace quadratic_has_two_distinct_roots_l862_86237

theorem quadratic_has_two_distinct_roots (a b c α : ℝ) (h : a * (a * α^2 + b * α + c) < 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a*x1^2 + b*x1 + c = 0) ∧ (a*x2^2 + b*x2 + c = 0) ∧ x1 < α ∧ x2 > α :=
sorry

end quadratic_has_two_distinct_roots_l862_86237


namespace Kim_has_4_cousins_l862_86253

noncomputable def pieces_per_cousin : ℕ := 5
noncomputable def total_pieces : ℕ := 20
noncomputable def cousins : ℕ := total_pieces / pieces_per_cousin

theorem Kim_has_4_cousins : cousins = 4 := 
by
  show cousins = 4
  sorry

end Kim_has_4_cousins_l862_86253


namespace rank_from_right_l862_86221

theorem rank_from_right (rank_from_left total_students : ℕ) (h1 : rank_from_left = 5) (h2 : total_students = 10) :
  total_students - rank_from_left + 1 = 6 :=
by 
  -- Placeholder for the actual proof.
  sorry

end rank_from_right_l862_86221


namespace glove_selection_l862_86202

theorem glove_selection :
  let n := 6                -- Number of pairs
  let k := 4                -- Number of selected gloves
  let m := 1                -- Number of matching pairs
  let total_ways := n * 10 * 8 / 2  -- Calculation based on solution steps
  total_ways = 240 := by
  sorry

end glove_selection_l862_86202


namespace meeting_day_correct_l862_86236

noncomputable def smallest_meeting_day :=
  ∀ (players courts : ℕ)
    (initial_reimu_court initial_marisa_court : ℕ),
    players = 2016 →
    courts = 1008 →
    initial_reimu_court = 123 →
    initial_marisa_court = 876 →
    ∀ (winner_moves_to court : ℕ → ℕ),
      (∀ (i : ℕ), 2 ≤ i ∧ i ≤ courts → winner_moves_to i = i - 1) →
      (winner_moves_to 1 = 1) →
      ∀ (loser_moves_to court : ℕ → ℕ),
        (∀ (j : ℕ), 1 ≤ j ∧ j ≤ courts - 1 → loser_moves_to j = j + 1) →
        (loser_moves_to courts = courts) →
        ∃ (n : ℕ), n = 1139

theorem meeting_day_correct : smallest_meeting_day :=
  sorry

end meeting_day_correct_l862_86236


namespace number_of_problems_l862_86261

theorem number_of_problems (Terry_score : ℤ) (points_right : ℤ) (points_wrong : ℤ) (wrong_ans : ℤ) 
  (h_score : Terry_score = 85) (h_points_right : points_right = 4) 
  (h_points_wrong : points_wrong = -1) (h_wrong_ans : wrong_ans = 3) : 
  ∃ (total_problems : ℤ), total_problems = 25 :=
by
  sorry

end number_of_problems_l862_86261


namespace students_more_than_pets_l862_86214

theorem students_more_than_pets
  (students_per_classroom : ℕ)
  (rabbits_per_classroom : ℕ)
  (birds_per_classroom : ℕ)
  (number_of_classrooms : ℕ)
  (total_students : ℕ)
  (total_rabbits : ℕ)
  (total_birds : ℕ)
  (total_pets : ℕ)
  (difference : ℕ)
  : students_per_classroom = 22 → 
    rabbits_per_classroom = 3 → 
    birds_per_classroom = 2 → 
    number_of_classrooms = 5 → 
    total_students = students_per_classroom * number_of_classrooms → 
    total_rabbits = rabbits_per_classroom * number_of_classrooms → 
    total_birds = birds_per_classroom * number_of_classrooms → 
    total_pets = total_rabbits + total_birds → 
    difference = total_students - total_pets →
    difference = 85 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end students_more_than_pets_l862_86214


namespace polynomial_divisible_x_minus_2_l862_86299

theorem polynomial_divisible_x_minus_2 (m : ℝ) : 
  (3 * 2^2 - 9 * 2 + m = 0) → m = 6 :=
by
  sorry

end polynomial_divisible_x_minus_2_l862_86299


namespace find_triangle_side_value_find_triangle_tan_value_l862_86271

noncomputable def triangle_side_value (A B C : ℝ) (a b c : ℝ) : Prop :=
  C = 2 * Real.pi / 3 ∧
  c = 5 ∧
  a = Real.sqrt 5 * b * Real.sin A ∧
  b = 2 * Real.sqrt 15 / 3

noncomputable def triangle_tan_value (B : ℝ) : Prop :=
  Real.tan (B + Real.pi / 4) = 3

theorem find_triangle_side_value (A B C a b c : ℝ) :
  triangle_side_value A B C a b c := by sorry

theorem find_triangle_tan_value (B : ℝ) :
  triangle_tan_value B := by sorry

end find_triangle_side_value_find_triangle_tan_value_l862_86271


namespace triangle_area_ellipse_l862_86293

open Real

noncomputable def ellipse_foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := sqrt (a^2 + b^2)
  ((-c, 0), (c, 0))

theorem triangle_area_ellipse 
  (a : ℝ) (b : ℝ) 
  (h1 : a = sqrt 2) (h2 : b = 1) 
  (F1 F2 : ℝ × ℝ) 
  (hfoci : ellipse_foci a b = (F1, F2))
  (hF2 : F2 = (sqrt 3, 0))
  (A B : ℝ × ℝ)
  (hA : A = (0, -1))
  (hB : B = (0, -1))
  (h_inclination : ∃ θ, θ = pi / 4 ∧ (B.1 - A.1) / (B.2 - A.2) = tan θ) :
  F1 = (-sqrt 3, 0) → 
  1/2 * (B.1 - A.1) * (B.2 - A.2) = 4/3 :=
sorry

end triangle_area_ellipse_l862_86293


namespace fencing_cost_per_foot_is_3_l862_86258

-- Definitions of the constants given in the problem
def side_length : ℕ := 9
def back_length : ℕ := 18
def total_cost : ℕ := 72
def neighbor_behind_rate : ℚ := 1/2
def neighbor_left_rate : ℚ := 1/3

-- The statement to be proved
theorem fencing_cost_per_foot_is_3 : 
  (total_cost / ((2 * side_length + back_length) - 
                (neighbor_behind_rate * back_length) -
                (neighbor_left_rate * side_length))) = 3 := 
by
  sorry

end fencing_cost_per_foot_is_3_l862_86258


namespace average_of_first_12_even_is_13_l862_86238

-- Define the first 12 even numbers
def first_12_even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

-- Define the sum of the first 12 even numbers
def sum_first_12_even : ℕ := first_12_even_numbers.sum

-- Define the number of values
def num_vals : ℕ := first_12_even_numbers.length

-- Define the average calculation
def average_first_12_even : ℕ := sum_first_12_even / num_vals

-- The theorem we want to prove
theorem average_of_first_12_even_is_13 : average_first_12_even = 13 := by
  sorry

end average_of_first_12_even_is_13_l862_86238


namespace total_profit_l862_86213

theorem total_profit (P Q R : ℝ) (profit : ℝ) 
  (h1 : 4 * P = 6 * Q) 
  (h2 : 6 * Q = 10 * R) 
  (h3 : R = 840 / 6) : 
  profit = 4340 :=
sorry

end total_profit_l862_86213


namespace max_distance_with_optimal_tire_swapping_l862_86296

theorem max_distance_with_optimal_tire_swapping
  (front_tires_last : ℕ)
  (rear_tires_last : ℕ)
  (front_tires_last_eq : front_tires_last = 20000)
  (rear_tires_last_eq : rear_tires_last = 30000) :
  ∃ D : ℕ, D = 30000 :=
by
  sorry

end max_distance_with_optimal_tire_swapping_l862_86296


namespace karen_wins_in_race_l862_86224

theorem karen_wins_in_race (w : ℝ) (h1 : w / 45 > 1 / 15) 
    (h2 : 60 * (w / 45 - 1 / 15) = w + 4) : 
    w = 8 / 3 := 
sorry

end karen_wins_in_race_l862_86224


namespace absolute_value_of_h_l862_86276

theorem absolute_value_of_h {h : ℝ} :
  (∀ x : ℝ, (x^2 + 2 * h * x = 3) → (∃ r s : ℝ, r + s = -2 * h ∧ r * s = -3 ∧ r^2 + s^2 = 10)) →
  |h| = 1 :=
by
  sorry

end absolute_value_of_h_l862_86276


namespace cagr_decline_l862_86226

theorem cagr_decline 
  (EV BV : ℝ) (n : ℕ) 
  (h_ev : EV = 52)
  (h_bv : BV = 89)
  (h_n : n = 3)
: ((EV / BV) ^ (1 / n) - 1) = -0.1678 := 
by
  rw [h_ev, h_bv, h_n]
  sorry

end cagr_decline_l862_86226


namespace nuts_consumed_range_l862_86211

def diet_day_nuts : Nat := 1
def normal_day_nuts : Nat := diet_day_nuts + 2

def total_nuts_consumed (start_with_diet_day : Bool) : Nat :=
  if start_with_diet_day then
    (10 * diet_day_nuts) + (9 * normal_day_nuts)
  else
    (10 * normal_day_nuts) + (9 * diet_day_nuts)

def min_nuts_consumed : Nat :=
  Nat.min (total_nuts_consumed true) (total_nuts_consumed false)

def max_nuts_consumed : Nat :=
  Nat.max (total_nuts_consumed true) (total_nuts_consumed false)

theorem nuts_consumed_range :
  min_nuts_consumed = 37 ∧ max_nuts_consumed = 39 := by
  sorry

end nuts_consumed_range_l862_86211


namespace mean_median_difference_is_correct_l862_86270

noncomputable def mean_median_difference (scores : List ℕ) (percentages : List ℚ) : ℚ := sorry

theorem mean_median_difference_is_correct :
  mean_median_difference [60, 75, 85, 90, 100] [15/100, 20/100, 25/100, 30/100, 10/100] = 2.75 :=
sorry

end mean_median_difference_is_correct_l862_86270


namespace smallest_value_of_y_l862_86208

theorem smallest_value_of_y (x y z d : ℝ) (h1 : x = y - d) (h2 : z = y + d) (h3 : x * y * z = 125) (h4 : 0 < x ∧ 0 < y ∧ 0 < z) : y ≥ 5 :=
by
  -- Officially, the user should navigate through the proof, but we conclude with 'sorry' as placeholder
  sorry

end smallest_value_of_y_l862_86208


namespace hexagon_perimeter_sum_l862_86282

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def perimeter : ℝ := 
  distance 0 0 1 2 +
  distance 1 2 3 3 +
  distance 3 3 5 3 +
  distance 5 3 6 1 +
  distance 6 1 4 (-1) +
  distance 4 (-1) 0 0

theorem hexagon_perimeter_sum :
  perimeter = 3 * Real.sqrt 5 + 2 + 2 * Real.sqrt 2 + Real.sqrt 17 := 
sorry

end hexagon_perimeter_sum_l862_86282


namespace problem_statement_l862_86252

noncomputable def nonnegative_reals : Type := {x : ℝ // 0 ≤ x}

theorem problem_statement (x : nonnegative_reals) :
  x.1^(3/2) + 6*x.1^(5/4) + 8*x.1^(3/4) ≥ 15*x.1 ∧
  (x.1^(3/2) + 6*x.1^(5/4) + 8*x.1^(3/4) = 15*x.1 ↔ (x.1 = 0 ∨ x.1 = 1)) :=
by
  sorry

end problem_statement_l862_86252


namespace petya_equals_vasya_l862_86283

def petya_word_count (m : ℕ) : ℕ :=
  sorry -- The actual count of m-letter words with equal T's and O's using letters T, O, W, and N.

def vasya_word_count (m : ℕ) : ℕ :=
  sorry -- The actual count of 2m-letter words with equal T's and O's using only letters T and O.

theorem petya_equals_vasya (m : ℕ) : petya_word_count m = vasya_word_count m :=
  sorry

end petya_equals_vasya_l862_86283


namespace largest_expression_l862_86249

noncomputable def x : ℝ := 10 ^ (-2024 : ℤ)

theorem largest_expression :
  let a := 5 + x
  let b := 5 - x
  let c := 5 * x
  let d := 5 / x
  let e := x / 5
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by
  sorry

end largest_expression_l862_86249


namespace parts_per_hour_l862_86263

theorem parts_per_hour (x y : ℝ) (h₁ : 90 / x = 120 / y) (h₂ : x + y = 35) : x = 15 ∧ y = 20 :=
by
  sorry

end parts_per_hour_l862_86263


namespace athletes_meet_time_number_of_overtakes_l862_86260

-- Define the speeds of the athletes
def speed1 := 155 -- m/min
def speed2 := 200 -- m/min
def speed3 := 275 -- m/min

-- Define the total length of the track
def track_length := 400 -- meters

-- Prove the minimum time for the athletes to meet again is 80/3 minutes
theorem athletes_meet_time (speed1 speed2 speed3 track_length : ℕ) (h1 : speed1 = 155) (h2 : speed2 = 200) (h3 : speed3 = 275) (h4 : track_length = 400) :
  ∃ t : ℚ, t = (80 / 3 : ℚ) :=
by
  sorry

-- Prove the number of overtakes during this time is 13
theorem number_of_overtakes (speed1 speed2 speed3 track_length t : ℕ) (h1 : speed1 = 155) (h2 : speed2 = 200) (h3 : speed3 = 275) (h4 : track_length = 400) (h5 : t = 80 / 3) :
  ∃ n : ℕ, n = 13 :=
by
  sorry

end athletes_meet_time_number_of_overtakes_l862_86260


namespace range_of_independent_variable_l862_86298

theorem range_of_independent_variable (x : ℝ) : 
  (∃ y, y = x / (Real.sqrt (x + 4)) + 1 / (x - 1)) ↔ x > -4 ∧ x ≠ 1 := 
by
  sorry

end range_of_independent_variable_l862_86298


namespace paint_room_alone_l862_86287

theorem paint_room_alone (x : ℝ) (hx : (1 / x) + (1 / 4) = 1 / 1.714) : x = 3 :=
by sorry

end paint_room_alone_l862_86287


namespace polynomial_solution_l862_86250

theorem polynomial_solution (x : ℝ) (h : (2 * x - 1) ^ 2 = 9) : x = 2 ∨ x = -1 :=
by
  sorry

end polynomial_solution_l862_86250


namespace compute_five_fold_application_l862_86272

def f (x : ℤ) : ℤ :=
  if x >= 0 then -(x^3) else x + 10

theorem compute_five_fold_application : f (f (f (f (f 2)))) = -8 := by
  sorry

end compute_five_fold_application_l862_86272


namespace largest_consecutive_odd_numbers_l862_86205

theorem largest_consecutive_odd_numbers (x : ℤ)
  (h : (x + (x + 2) + (x + 4) + (x + 6)) / 4 = 24) : 
  x + 6 = 27 :=
  sorry

end largest_consecutive_odd_numbers_l862_86205


namespace largest_two_digit_with_remainder_2_l862_86240

theorem largest_two_digit_with_remainder_2 (n : ℕ) :
  10 ≤ n ∧ n ≤ 99 ∧ n % 13 = 2 → n = 93 :=
by
  intro h
  sorry

end largest_two_digit_with_remainder_2_l862_86240


namespace pages_per_chapter_l862_86201

-- Definitions based on conditions
def chapters_in_book : ℕ := 2
def days_to_finish : ℕ := 664
def chapters_per_day : ℕ := 332
def total_chapters_read : ℕ := chapters_per_day * days_to_finish

-- Theorem that states the problem
theorem pages_per_chapter : total_chapters_read / chapters_in_book = 110224 :=
by
  -- Proof is omitted
  sorry

end pages_per_chapter_l862_86201


namespace largest_partner_share_l862_86233

-- Definitions for the conditions
def total_profit : ℕ := 48000
def ratio_parts : List ℕ := [2, 4, 5, 3, 6]
def total_ratio_parts : ℕ := ratio_parts.sum
def value_per_part : ℕ := total_profit / total_ratio_parts
def largest_share : ℕ := 6 * value_per_part

-- Statement of the proof problem
theorem largest_partner_share : largest_share = 14400 := by
  -- Insert proof here
  sorry

end largest_partner_share_l862_86233


namespace expression_simplification_l862_86204

theorem expression_simplification :
  (2 ^ 2 / 3 + (-(3 ^ 2) + 5) + (-(3) ^ 2) * ((2 / 3) ^ 2)) = 4 / 3 :=
sorry

end expression_simplification_l862_86204


namespace sum_pos_integers_9_l862_86294

theorem sum_pos_integers_9 (x y z : ℕ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_eq : 30 / 7 = x + 1 / (y + 1 / z)) : x + y + z = 9 :=
sorry

end sum_pos_integers_9_l862_86294


namespace miss_tree_class_children_count_l862_86254

noncomputable def number_of_children (n: ℕ) : ℕ := 7 * n + 2

theorem miss_tree_class_children_count (n : ℕ) :
  (20 < number_of_children n) ∧ (number_of_children n < 30) ∧ 7 * n + 2 = 23 :=
by {
  sorry
}

end miss_tree_class_children_count_l862_86254


namespace full_day_students_l862_86212

def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

theorem full_day_students : 
  (total_students - (total_students * percentage_half_day_students / 100)) = 60 := by
  sorry

end full_day_students_l862_86212


namespace birds_after_changes_are_235_l862_86245

-- Define initial conditions for the problem
def initial_cages : Nat := 15
def parrots_per_cage : Nat := 3
def parakeets_per_cage : Nat := 8
def canaries_per_cage : Nat := 5
def parrots_sold : Nat := 5
def canaries_sold : Nat := 2
def parakeets_added : Nat := 2


-- Define the function to count total birds after the changes
def total_birds_after_changes (initial_cages parrots_per_cage parakeets_per_cage canaries_per_cage parrots_sold canaries_sold parakeets_added : Nat) : Nat :=
  let initial_parrots := initial_cages * parrots_per_cage
  let initial_parakeets := initial_cages * parakeets_per_cage
  let initial_canaries := initial_cages * canaries_per_cage
  
  let final_parrots := initial_parrots - parrots_sold
  let final_parakeets := initial_parakeets + parakeets_added
  let final_canaries := initial_canaries - canaries_sold
  
  final_parrots + final_parakeets + final_canaries

-- Prove that the total number of birds is 235
theorem birds_after_changes_are_235 : total_birds_after_changes 15 3 8 5 5 2 2 = 235 :=
  by 
    -- Proof is omitted as per the instructions
    sorry

end birds_after_changes_are_235_l862_86245


namespace lana_goal_is_20_l862_86285

def muffins_sold_morning := 12
def muffins_sold_afternoon := 4
def muffins_needed_to_goal := 4
def total_muffins_sold := muffins_sold_morning + muffins_sold_afternoon
def lana_goal := total_muffins_sold + muffins_needed_to_goal

theorem lana_goal_is_20 : lana_goal = 20 := by
  sorry

end lana_goal_is_20_l862_86285


namespace product_of_numerator_and_denominator_l862_86207

-- Defining the repeating decimal as a fraction in lowest terms
def repeating_decimal_as_fraction_in_lowest_terms : ℚ :=
  1 / 37

-- Theorem to prove the product of the numerator and the denominator
theorem product_of_numerator_and_denominator :
  (repeating_decimal_as_fraction_in_lowest_terms.num.natAbs *
   repeating_decimal_as_fraction_in_lowest_terms.den) = 37 :=
by
  -- declaration of the needed fact and its direct consequence
  sorry

end product_of_numerator_and_denominator_l862_86207


namespace intersection_single_point_max_PA_PB_l862_86267

-- Problem (1)
theorem intersection_single_point (a : ℝ) :
  (∀ x : ℝ, 2 * a = |x - a| - 1 → x = a) → a = -1 / 2 :=
sorry

-- Problem (2)
theorem max_PA_PB (m : ℝ) (P : ℝ × ℝ) :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, 3)
  P ≠ A ∧ P ≠ B ∧ (P.1 + m * P.2 = 0) ∧ (m * P.1 - P.2 - m + 3 = 0) →
  |dist P A| * |dist P B| ≤ 5 :=
sorry

end intersection_single_point_max_PA_PB_l862_86267


namespace sum_digits_l862_86281

def distinct_digits (a b c d : ℕ) : Prop :=
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d)

def valid_equation (Y E M T : ℕ) : Prop :=
  ∃ (YE ME TTT : ℕ),
    YE = Y * 10 + E ∧
    ME = M * 10 + E ∧
    TTT = T * 111 ∧
    YE < ME ∧
    YE * ME = TTT ∧
    distinct_digits Y E M T

theorem sum_digits (Y E M T : ℕ) :
  valid_equation Y E M T → Y + E + M + T = 21 := 
sorry

end sum_digits_l862_86281
