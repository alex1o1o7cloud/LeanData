import Mathlib

namespace find_original_number_l1707_170768

theorem find_original_number (r : ℝ) (h1 : r * 1.125 - r * 0.75 = 30) : r = 80 :=
by
  sorry

end find_original_number_l1707_170768


namespace find_y_given_x_l1707_170779

-- Let x and y be real numbers
variables (x y : ℝ)

-- Assume x and y are inversely proportional, so their product is a constant C
variable (C : ℝ)

-- Additional conditions from the problem statement
variable (h1 : x + y = 40) (h2 : x - y = 10) (hx : x = 7)

-- Define the goal: y = 375 / 7
theorem find_y_given_x : y = 375 / 7 :=
sorry

end find_y_given_x_l1707_170779


namespace election_majority_l1707_170799

theorem election_majority (V : ℝ) 
  (h1 : ∃ w l : ℝ, w = 0.70 * V ∧ l = 0.30 * V ∧ w - l = 174) : 
  V = 435 :=
by
  sorry

end election_majority_l1707_170799


namespace exists_f_satisfying_iteration_l1707_170710

-- Mathematically equivalent problem statement in Lean 4
theorem exists_f_satisfying_iteration :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, (f^[1995] n) = 2 * n :=
by
  -- Fill in proof here
  sorry

end exists_f_satisfying_iteration_l1707_170710


namespace distinct_positive_integers_exists_l1707_170700

theorem distinct_positive_integers_exists 
(n : ℕ)
(a b : ℕ)
(h1 : a ≠ b)
(h2 : b % a = 0)
(h3 : a > 10^(2 * n - 1) ∧ a < 10^(2 * n))
(h4 : b > 10^(2 * n - 1) ∧ b < 10^(2 * n))
(h5 : ∀ x y : ℕ, a = 10^n * x + y ∧ b = 10^n * y + x ∧ x < y ∧ x / 10^(n - 1) ≠ 0 ∧ y / 10^(n - 1) ≠ 0) :
a = (10^(2 * n) - 1) / 7 ∧ b = 6 * (10^(2 * n) - 1) / 7 := 
by
  sorry

end distinct_positive_integers_exists_l1707_170700


namespace passing_marks_required_l1707_170734

theorem passing_marks_required (T : ℝ)
  (h1 : 0.30 * T + 60 = 0.40 * T)
  (h2 : 0.40 * T = passing_mark)
  (h3 : 0.50 * T - 40 = passing_mark) :
  passing_mark = 240 := by
  sorry

end passing_marks_required_l1707_170734


namespace correct_statement_about_CH3COOK_l1707_170741

def molar_mass_CH3COOK : ℝ := 98  -- in g/mol

def avogadro_number : ℝ := 6.02 * 10^23  -- molecules per mole

def hydrogen_atoms_in_CH3COOK (mol_CH3COOK : ℝ) : ℝ :=
  3 * mol_CH3COOK * avogadro_number

theorem correct_statement_about_CH3COOK (mol_CH3COOK : ℝ) (h: mol_CH3COOK = 1) :
  hydrogen_atoms_in_CH3COOK mol_CH3COOK = 3 * avogadro_number :=
by
  sorry

end correct_statement_about_CH3COOK_l1707_170741


namespace polygon_sides_l1707_170773

theorem polygon_sides (n : ℕ) (h : 180 * (n - 2) = 1620) : n = 11 := 
by 
  sorry

end polygon_sides_l1707_170773


namespace find_x_l1707_170797

open Nat

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_x (x : ℕ) (hx : x > 0) (hprime : is_prime (x^5 + x + 1)) : x = 1 := 
by 
  sorry

end find_x_l1707_170797


namespace expected_total_rainfall_over_week_l1707_170786

noncomputable def daily_rain_expectation : ℝ :=
  (0.5 * 0) + (0.2 * 2) + (0.3 * 5)

noncomputable def total_rain_expectation (days: ℕ) : ℝ :=
  days * daily_rain_expectation

theorem expected_total_rainfall_over_week : total_rain_expectation 7 = 13.3 :=
by 
  -- calculation of expected value here
  -- daily_rain_expectation = 1.9
  -- total_rain_expectation 7 = 7 * 1.9 = 13.3
  sorry

end expected_total_rainfall_over_week_l1707_170786


namespace coefficient_fifth_term_expansion_l1707_170706

theorem coefficient_fifth_term_expansion :
  let a := (2 : ℝ)
  let b := -(1 : ℝ)
  let n := 6
  let k := 4
  Nat.choose n k * (a ^ (n - k)) * (b ^ k) = 60 := by
  -- We can assume x to be any nonzero real, but it is not needed in the theorem itself.
  sorry

end coefficient_fifth_term_expansion_l1707_170706


namespace ratio_traditionalists_progressives_l1707_170732

variables (T P C : ℝ)

-- Conditions from the problem
-- There are 6 provinces and each province has the same number of traditionalists
-- The fraction of the country that is traditionalist is 0.6
def country_conditions (T P C : ℝ) :=
  (6 * T = 0.6 * C) ∧
  (C = P + 6 * T)

-- Theorem that needs to be proven
theorem ratio_traditionalists_progressives (T P C : ℝ) (h : country_conditions T P C) :
  T / P = 1 / 4 :=
by
  -- Setup conditions from the hypothesis h
  rcases h with ⟨h1, h2⟩
  -- Start the proof (Proof content is not required as per instructions)
  sorry

end ratio_traditionalists_progressives_l1707_170732


namespace find_g5_l1707_170774

noncomputable def g (x : ℤ) : ℤ := sorry

axiom condition1 : g 1 > 1
axiom condition2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom condition3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 := 
by
  sorry

end find_g5_l1707_170774


namespace subtraction_of_negatives_l1707_170793

theorem subtraction_of_negatives : (-1) - (-4) = 3 :=
by
  -- Proof goes here.
  sorry

end subtraction_of_negatives_l1707_170793


namespace max_books_per_student_l1707_170756

-- Define the variables and conditions
variables (students : ℕ) (not_borrowed5 borrowed1_20 borrowed2_25 borrowed3_30 borrowed5_20 : ℕ)
variables (avg_books_per_student : ℕ)
variables (remaining_books : ℕ) (max_books : ℕ)

-- Assume given conditions
def conditions : Prop :=
  students = 100 ∧ 
  not_borrowed5 = 5 ∧ 
  borrowed1_20 = 20 ∧ 
  borrowed2_25 = 25 ∧ 
  borrowed3_30 = 30 ∧ 
  borrowed5_20 = 20 ∧ 
  avg_books_per_student = 3

-- Prove the maximum number of books any single student could have borrowed is 50
theorem max_books_per_student (students not_borrowed5 borrowed1_20 borrowed2_25 borrowed3_30 borrowed5_20 avg_books_per_student : ℕ) (max_books : ℕ) :
  conditions students not_borrowed5 borrowed1_20 borrowed2_25 borrowed3_30 borrowed5_20 avg_books_per_student →
  max_books = 50 :=
by
  sorry

end max_books_per_student_l1707_170756


namespace electricity_usage_A_B_l1707_170720

def electricity_cost (x : ℕ) : ℝ :=
  if h₁ : 0 ≤ x ∧ x ≤ 24 then 4.2 * x
  else if h₂ : 24 < x ∧ x ≤ 60 then 5.2 * x - 24
  else if h₃ : 60 < x ∧ x ≤ 100 then 6.6 * x - 108
  else if h₄ : 100 < x ∧ x ≤ 150 then 7.6 * x - 208
  else if h₅ : 150 < x ∧ x ≤ 250 then 8 * x - 268
  else 8.4 * x - 368

theorem electricity_usage_A_B (x : ℕ) (h : electricity_cost x = 486) :
  60 < x ∧ x ≤ 100 ∧ 5 * x = 450 ∧ 2 * x = 180 :=
by
  sorry

end electricity_usage_A_B_l1707_170720


namespace tree_height_at_year_3_l1707_170769

theorem tree_height_at_year_3 :
  ∃ h₃ : ℕ, h₃ = 27 ∧
  (∃ h₇ h₆ h₅ h₄ : ℕ,
   h₇ = 648 ∧
   h₆ = h₇ / 2 ∧
   h₅ = h₆ / 2 ∧
   h₄ = h₅ / 2 ∧
   h₄ = 3 * h₃) :=
by
  sorry

end tree_height_at_year_3_l1707_170769


namespace numeral_of_place_face_value_difference_l1707_170790

theorem numeral_of_place_face_value_difference (P F : ℕ) (H : P - F = 63) (Hface : F = 7) : P = 70 :=
sorry

end numeral_of_place_face_value_difference_l1707_170790


namespace figure_perimeter_l1707_170717

theorem figure_perimeter 
  (side_length : ℕ)
  (inner_large_square_sides : ℕ)
  (shared_edge_length : ℕ)
  (rectangle_dimension_1 : ℕ)
  (rectangle_dimension_2 : ℕ) 
  (h1 : side_length = 2)
  (h2 : inner_large_square_sides = 4)
  (h3 : shared_edge_length = 2)
  (h4 : rectangle_dimension_1 = 2)
  (h5 : rectangle_dimension_2 = 1) : 
  let large_square_perimeter := inner_large_square_sides * side_length
  let horizontal_perimeter := large_square_perimeter - shared_edge_length + rectangle_dimension_1 + rectangle_dimension_2
  let vertical_perimeter := large_square_perimeter
  horizontal_perimeter + vertical_perimeter = 33 := 
by
  sorry

end figure_perimeter_l1707_170717


namespace analysis_duration_unknown_l1707_170760

-- Definitions based on the given conditions
def number_of_bones : Nat := 206
def analysis_duration_per_bone (bone: Nat) : Nat := 5  -- assumed fixed for simplicity
-- Time spent analyzing all bones (which needs more information to be accurately known)
def total_analysis_time (bones_analyzed: Nat) (hours_per_bone: Nat) : Nat := bones_analyzed * hours_per_bone

-- Given the number of bones and duration per bone, there isn't enough information to determine the total analysis duration
theorem analysis_duration_unknown (total_bones : Nat) (duration_per_bone : Nat) (bones_remaining: Nat) (analysis_already_done : Nat) :
  total_bones = number_of_bones →
  (∀ bone, analysis_duration_per_bone bone = duration_per_bone) →
  analysis_already_done ≠ (total_bones - bones_remaining) ->
  ∃ hours_needed, hours_needed = total_analysis_time (total_bones - bones_remaining) duration_per_bone :=
by
  intros
  sorry

end analysis_duration_unknown_l1707_170760


namespace Marissa_sister_height_l1707_170780

theorem Marissa_sister_height (sunflower_height_feet : ℕ) (height_difference_inches : ℕ) :
  sunflower_height_feet = 6 -> height_difference_inches = 21 -> 
  let sunflower_height_inches := sunflower_height_feet * 12
  let sister_height_inches := sunflower_height_inches - height_difference_inches
  let sister_height_feet := sister_height_inches / 12
  let sister_height_remainder_inches := sister_height_inches % 12
  sister_height_feet = 4 ∧ sister_height_remainder_inches = 3 :=
by
  intros
  sorry

end Marissa_sister_height_l1707_170780


namespace area_of_rectangle_l1707_170792

-- Definitions from the conditions
def breadth (b : ℝ) : Prop := b > 0
def length (l b : ℝ) : Prop := l = 3 * b
def perimeter (P l b : ℝ) : Prop := P = 2 * (l + b)

-- The main theorem we are proving
theorem area_of_rectangle (b l : ℝ) (P : ℝ) (h1 : breadth b) (h2 : length l b) (h3 : perimeter P l b) (h4 : P = 96) : l * b = 432 := 
by
  -- Proof steps will go here
  sorry

end area_of_rectangle_l1707_170792


namespace negation_of_existence_l1707_170719

theorem negation_of_existence (h : ¬ (∃ x : ℝ, x^2 - x - 1 > 0)) : ∀ x : ℝ, x^2 - x - 1 ≤ 0 :=
sorry

end negation_of_existence_l1707_170719


namespace number_of_chickens_l1707_170747

theorem number_of_chickens (c k : ℕ) (h1 : c + k = 120) (h2 : 2 * c + 4 * k = 350) : c = 65 :=
by sorry

end number_of_chickens_l1707_170747


namespace modular_inverse_28_mod_29_l1707_170709

theorem modular_inverse_28_mod_29 :
  28 * 28 ≡ 1 [MOD 29] :=
by
  sorry

end modular_inverse_28_mod_29_l1707_170709


namespace interval_length_implies_difference_l1707_170759

theorem interval_length_implies_difference (a b : ℝ) (h : (b - 5) / 3 - (a - 5) / 3 = 15) : b - a = 45 := by
  sorry

end interval_length_implies_difference_l1707_170759


namespace geometric_seq_l1707_170777

def seq (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  a 1 = 3 ∧ (∀ n : ℕ, S (n + 1) + a n = S n + 5 * 4 ^ n)

theorem geometric_seq (a S : ℕ → ℝ) (h : seq a S) :
  ∃ r : ℝ, ∃ a1 : ℝ, (∀ n : ℕ, (a (n + 1) - 4 ^ (n + 1)) = r * (a n - 4 ^ n)) :=
by
  sorry

end geometric_seq_l1707_170777


namespace line_equation_in_slope_intercept_form_l1707_170713

variable {x y : ℝ}

theorem line_equation_in_slope_intercept_form :
  (3 * (x - 2) - 4 * (y - 8) = 0) → (y = (3 / 4) * x + 6.5) :=
by
  intro h
  sorry

end line_equation_in_slope_intercept_form_l1707_170713


namespace smallest_integer_greater_than_sqrt5_plus_sqrt3_pow6_l1707_170737

theorem smallest_integer_greater_than_sqrt5_plus_sqrt3_pow6 :
  ∃ n : ℤ, n = 3323 ∧ n > (Real.sqrt 5 + Real.sqrt 3)^6 ∧ ∀ m : ℤ, m > (Real.sqrt 5 + Real.sqrt 3)^6 → n ≤ m :=
by
  sorry

end smallest_integer_greater_than_sqrt5_plus_sqrt3_pow6_l1707_170737


namespace num_adults_on_field_trip_l1707_170722

-- Definitions of the conditions
def num_vans : Nat := 6
def people_per_van : Nat := 9
def num_students : Nat := 40

-- The theorem to prove
theorem num_adults_on_field_trip : (num_vans * people_per_van) - num_students = 14 := by
  sorry

end num_adults_on_field_trip_l1707_170722


namespace weight_loss_challenge_l1707_170782

noncomputable def percentage_weight_loss (W : ℝ) : ℝ :=
  ((W - (0.918 * W)) / W) * 100

theorem weight_loss_challenge (W : ℝ) (h : W > 0) :
  percentage_weight_loss W = 8.2 :=
by
  sorry

end weight_loss_challenge_l1707_170782


namespace total_amount_l1707_170727

theorem total_amount (x y z : ℝ) 
  (hy : y = 0.45 * x) 
  (hz : z = 0.30 * x) 
  (hy_value : y = 54) : 
  x + y + z = 210 := 
by
  sorry

end total_amount_l1707_170727


namespace john_taking_pictures_years_l1707_170704

-- Definitions based on the conditions
def pictures_per_day : ℕ := 10
def images_per_card : ℕ := 50
def cost_per_card : ℕ := 60
def total_spent : ℕ := 13140
def days_per_year : ℕ := 365

-- Theorem statement
theorem john_taking_pictures_years : total_spent / cost_per_card * images_per_card / pictures_per_day / days_per_year = 3 :=
by
  sorry

end john_taking_pictures_years_l1707_170704


namespace sin_cos_identity_l1707_170702

theorem sin_cos_identity (α : ℝ) (h1 : Real.sin (α - Real.pi / 6) = 1 / 3) :
    Real.sin (2 * α - Real.pi / 6) + Real.cos (2 * α) = 7 / 9 :=
sorry

end sin_cos_identity_l1707_170702


namespace candies_bought_l1707_170775

theorem candies_bought :
  ∃ (S C : ℕ), S + C = 8 ∧ 300 * S + 500 * C = 3000 ∧ C = 3 :=
by
  sorry

end candies_bought_l1707_170775


namespace sum_of_first_five_terms_l1707_170784

theorem sum_of_first_five_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q) -- geometric sequence definition
  (h3 : a 2 + a 5 = 2 * (a 4 + 2)) : 
  S 5 = 62 :=
by
  -- lean tactics would go here to provide the proof
  sorry

end sum_of_first_five_terms_l1707_170784


namespace sum_geom_seq_l1707_170735

theorem sum_geom_seq (S : ℕ → ℝ) (a_n : ℕ → ℝ) (h1 : S 4 ≠ 0) 
  (h2 : S 8 / S 4 = 4) 
  (h3 : ∀ n : ℕ, S n = a_n 0 * (1 - (a_n 1 / a_n 0)^n) / (1 - a_n 1 / a_n 0)) :
  S 12 / S 4 = 13 :=
sorry

end sum_geom_seq_l1707_170735


namespace circles_intersect_l1707_170708

def circle1 (x y : ℝ) := x^2 + y^2 + 2*x + 8*y - 8 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4*x - 4*y - 1 = 0

theorem circles_intersect : 
  (∃ (x y : ℝ), circle1 x y ∧ circle2 x y) := 
sorry

end circles_intersect_l1707_170708


namespace big_SUV_wash_ratio_l1707_170752

-- Defining constants for time taken for various parts of the car
def time_windows : ℕ := 4
def time_body : ℕ := 7
def time_tires : ℕ := 4
def time_waxing : ℕ := 9

-- Time taken to wash one normal car
def time_normal_car : ℕ := time_windows + time_body + time_tires + time_waxing

-- Given total time William spent washing all vehicles
def total_time : ℕ := 96

-- Time taken for two normal cars
def time_two_normal_cars : ℕ := 2 * time_normal_car

-- Time taken for the big SUV
def time_big_SUV : ℕ := total_time - time_two_normal_cars

-- Ratio of time taken to wash the big SUV to the time taken to wash a normal car
def time_ratio : ℕ := time_big_SUV / time_normal_car

theorem big_SUV_wash_ratio : time_ratio = 2 := by
  sorry

end big_SUV_wash_ratio_l1707_170752


namespace price_per_pound_second_coffee_l1707_170705

theorem price_per_pound_second_coffee
  (price_first : ℝ) (total_mix_weight : ℝ) (sell_price_per_pound : ℝ) (each_kind_weight : ℝ) 
  (total_sell_price : ℝ) (total_first_cost : ℝ) (total_second_cost : ℝ) (price_second : ℝ) :
  price_first = 2.15 →
  total_mix_weight = 18 →
  sell_price_per_pound = 2.30 →
  each_kind_weight = 9 →
  total_sell_price = total_mix_weight * sell_price_per_pound →
  total_first_cost = each_kind_weight * price_first →
  total_second_cost = total_sell_price - total_first_cost →
  price_second = total_second_cost / each_kind_weight →
  price_second = 2.45 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end price_per_pound_second_coffee_l1707_170705


namespace cats_left_l1707_170701

theorem cats_left (siamese house persian sold_first sold_second : ℕ) (h1 : siamese = 23) (h2 : house = 17) (h3 : persian = 29) (h4 : sold_first = 40) (h5 : sold_second = 12) :
  siamese + house + persian - sold_first - sold_second = 17 :=
by sorry

end cats_left_l1707_170701


namespace last_number_nth_row_sum_of_nth_row_position_of_2008_l1707_170787

theorem last_number_nth_row (n : ℕ) : 
  ∃ last_number, last_number = 2^n - 1 := by
  sorry

theorem sum_of_nth_row (n : ℕ) : 
  ∃ sum_nth_row, sum_nth_row = 2^(2*n-2) + 2^(2*n-3) - 2^(n-2) := by
  sorry

theorem position_of_2008 : 
  ∃ (row : ℕ) (position : ℕ), row = 11 ∧ position = 2008 - 2^10 + 1 :=
  by sorry

end last_number_nth_row_sum_of_nth_row_position_of_2008_l1707_170787


namespace cans_to_paint_35_rooms_l1707_170748

/-- Paula the painter initially had enough paint for 45 identically sized rooms.
    Unfortunately, she lost five cans of paint, leaving her with only enough paint for 35 rooms.
    Prove that she now uses 18 cans of paint to paint the 35 rooms. -/
theorem cans_to_paint_35_rooms :
  ∀ (cans_per_room : ℕ) (total_cans : ℕ) (lost_cans : ℕ) (rooms_before : ℕ) (rooms_after : ℕ),
  rooms_before = 45 →
  lost_cans = 5 →
  rooms_after = 35 →
  rooms_before - rooms_after = cans_per_room * lost_cans →
  (cans_per_room * rooms_after) / rooms_after = 18 :=
by
  intros
  sorry

end cans_to_paint_35_rooms_l1707_170748


namespace number_of_a_values_l1707_170757

theorem number_of_a_values (a : ℝ) : 
  (∃ a : ℝ, ∃ b : ℝ, a = 0 ∨ a = 1) := sorry

end number_of_a_values_l1707_170757


namespace min_value_of_expression_l1707_170743

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 1) :
  (1 / (2 * a) + 2 / b) = 8 :=
sorry

end min_value_of_expression_l1707_170743


namespace no_line_normal_to_both_curves_l1707_170766

theorem no_line_normal_to_both_curves :
  ¬ ∃ a b : ℝ, ∃ (l : ℝ → ℝ),
    -- normal to y = cosh x at x = a
    (∀ x : ℝ, l x = -1 / (Real.sinh a) * (x - a) + Real.cosh a) ∧
    -- normal to y = sinh x at x = b
    (∀ x : ℝ, l x = -1 / (Real.cosh b) * (x - b) + Real.sinh b) := 
  sorry

end no_line_normal_to_both_curves_l1707_170766


namespace average_increase_l1707_170754

theorem average_increase (A A' : ℕ) (runs_in_17th : ℕ) (total_innings : ℕ) (new_avg : ℕ) 
(h1 : total_innings = 17)
(h2 : runs_in_17th = 87)
(h3 : new_avg = 39)
(h4 : A' = new_avg)
(h5 : 16 * A + runs_in_17th = total_innings * new_avg) 
: A' - A = 3 := by
  sorry

end average_increase_l1707_170754


namespace reflect_point_example_l1707_170730

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def reflect_over_x_axis (P : Point3D) : Point3D :=
  { x := P.x, y := -P.y, z := -P.z }

theorem reflect_point_example :
  reflect_over_x_axis ⟨2, 3, 4⟩ = ⟨2, -3, -4⟩ :=
by
  -- Proof can be filled in here
  sorry

end reflect_point_example_l1707_170730


namespace average_speed_ratio_l1707_170707

theorem average_speed_ratio (t_E t_F : ℝ) (d_B d_C : ℝ) (htE : t_E = 3) (htF : t_F = 4) (hdB : d_B = 450) (hdC : d_C = 300) :
  (d_B / t_E) / (d_C / t_F) = 2 :=
by
  sorry

end average_speed_ratio_l1707_170707


namespace bill_head_circumference_l1707_170755

theorem bill_head_circumference (jack_head_circumference charlie_head_circumference bill_head_circumference : ℝ) :
  jack_head_circumference = 12 →
  charlie_head_circumference = (1 / 2 * jack_head_circumference) + 9 →
  bill_head_circumference = (2 / 3 * charlie_head_circumference) →
  bill_head_circumference = 10 :=
by
  intro hj hc hb
  sorry

end bill_head_circumference_l1707_170755


namespace find_value_of_x2_div_y2_l1707_170772

theorem find_value_of_x2_div_y2 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x ≠ y) (h5 : y ≠ z) (h6 : x ≠ z)
    (h7 : (y^2 / (x^2 - z^2) = (x^2 + y^2) / z^2))
    (h8 : (x^2 + y^2) / z^2 = x^2 / y^2) : x^2 / y^2 = 2 := by
  sorry

end find_value_of_x2_div_y2_l1707_170772


namespace average_payment_l1707_170798

theorem average_payment (total_payments : ℕ) (first_n_payments : ℕ)  (first_payment_amt : ℕ) (remaining_payment_amt : ℕ) 
  (H1 : total_payments = 104)
  (H2 : first_n_payments = 24)
  (H3 : first_payment_amt = 520)
  (H4 : remaining_payment_amt = 615)
  :
  (24 * 520 + 80 * 615) / 104 = 593.08 := 
  by 
    sorry

end average_payment_l1707_170798


namespace johns_original_earnings_l1707_170767

theorem johns_original_earnings (x : ℝ) (h1 : x + 0.5 * x = 90) : x = 60 := 
by
  -- sorry indicates the proof steps are omitted
  sorry

end johns_original_earnings_l1707_170767


namespace triangle_angles_l1707_170770

theorem triangle_angles (second_angle first_angle third_angle : ℝ) 
  (h1 : first_angle = 2 * second_angle)
  (h2 : third_angle = second_angle + 30)
  (h3 : second_angle + first_angle + third_angle = 180) :
  second_angle = 37.5 ∧ first_angle = 75 ∧ third_angle = 67.5 :=
sorry

end triangle_angles_l1707_170770


namespace expand_fraction_product_l1707_170796

theorem expand_fraction_product (x : ℝ) (hx : x ≠ 0) : 
  (3 / 4) * (8 / x^2 - 5 * x^3) = 6 / x^2 - 15 * x^3 / 4 := 
by 
  sorry

end expand_fraction_product_l1707_170796


namespace shape_of_phi_eq_d_in_spherical_coordinates_l1707_170795

theorem shape_of_phi_eq_d_in_spherical_coordinates (d : ℝ) : 
  (∃ (ρ θ : ℝ), ∀ (φ : ℝ), φ = d) ↔ ( ∃ cone_vertex : ℝ × ℝ × ℝ, ∃ opening_angle : ℝ, cone_vertex = (0, 0, 0) ∧ opening_angle = d) :=
sorry

end shape_of_phi_eq_d_in_spherical_coordinates_l1707_170795


namespace part_A_part_B_part_D_l1707_170776

variables (c d : ℤ)

def multiple_of_5 (x : ℤ) : Prop := ∃ k : ℤ, x = 5 * k
def multiple_of_10 (x : ℤ) : Prop := ∃ k : ℤ, x = 10 * k

-- Given conditions
axiom h1 : multiple_of_5 c
axiom h2 : multiple_of_10 d

-- Problems to prove
theorem part_A : multiple_of_5 d := by sorry
theorem part_B : multiple_of_5 (c - d) := by sorry
theorem part_D : multiple_of_5 (c + d) := by sorry

end part_A_part_B_part_D_l1707_170776


namespace gas_cost_per_gallon_l1707_170745

def car_mileage : Nat := 450
def car1_mpg : Nat := 50
def car2_mpg : Nat := 10
def car3_mpg : Nat := 15
def monthly_gas_cost : Nat := 56

theorem gas_cost_per_gallon (car_mileage car1_mpg car2_mpg car3_mpg monthly_gas_cost : Nat)
  (h1 : car_mileage = 450) 
  (h2 : car1_mpg = 50) 
  (h3 : car2_mpg = 10) 
  (h4 : car3_mpg = 15) 
  (h5 : monthly_gas_cost = 56) :
  monthly_gas_cost / ((car_mileage / 3) / car1_mpg + 
                      (car_mileage / 3) / car2_mpg + 
                      (car_mileage / 3) / car3_mpg) = 2 := 
by 
  sorry

end gas_cost_per_gallon_l1707_170745


namespace graph_of_eqn_is_pair_of_lines_l1707_170723

theorem graph_of_eqn_is_pair_of_lines : 
  ∃ (l₁ l₂ : ℝ × ℝ → Prop), 
  (∀ x y, l₁ (x, y) ↔ x = 2 * y) ∧ 
  (∀ x y, l₂ (x, y) ↔ x = -2 * y) ∧ 
  (∀ x y, (x^2 - 4 * y^2 = 0) ↔ (l₁ (x, y) ∨ l₂ (x, y))) :=
by
  sorry

end graph_of_eqn_is_pair_of_lines_l1707_170723


namespace cube_volume_increase_l1707_170763

theorem cube_volume_increase (s : ℝ) (h : s > 0) :
  let new_volume := (1.4 * s) ^ 3
  let original_volume := s ^ 3
  let increase_percentage := ((new_volume - original_volume) / original_volume) * 100
  increase_percentage = 174.4 := by
  sorry

end cube_volume_increase_l1707_170763


namespace quadratic_roots_distinct_real_l1707_170714

theorem quadratic_roots_distinct_real (a b c : ℝ) (h_eq : 2 * a = 2 ∧ 2 * b + -3 = b ∧ 2 * c + 1 = c) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (∀ x : ℝ, (2 * x^2 + (-3) * x + 1 = 0) ↔ (x = x1 ∨ x = x2)) :=
by
  sorry

end quadratic_roots_distinct_real_l1707_170714


namespace initial_nickels_proof_l1707_170725

def initial_nickels (N : ℕ) (D : ℕ) (total_value : ℝ) : Prop :=
  D = 3 * N ∧
  total_value = (N + 2 * N) * 0.05 + 3 * N * 0.10 ∧
  total_value = 9

theorem initial_nickels_proof : ∃ N, ∃ D, (initial_nickels N D 9) → (N = 20) :=
by
  sorry

end initial_nickels_proof_l1707_170725


namespace shaded_area_floor_l1707_170785

noncomputable def area_of_white_quarter_circle : ℝ := Real.pi / 4

noncomputable def area_of_white_per_tile : ℝ := 4 * area_of_white_quarter_circle

noncomputable def area_of_tile : ℝ := 4

noncomputable def shaded_area_per_tile : ℝ := area_of_tile - area_of_white_per_tile

noncomputable def number_of_tiles : ℕ := by
  have floor_area : ℝ := 12 * 15
  have tile_area : ℝ := 2 * 2
  exact Nat.floor (floor_area / tile_area)

noncomputable def total_shaded_area (num_tiles : ℕ) : ℝ := num_tiles * shaded_area_per_tile

theorem shaded_area_floor : total_shaded_area number_of_tiles = 180 - 45 * Real.pi := by
  sorry

end shaded_area_floor_l1707_170785


namespace length_AD_of_circle_l1707_170711

def circle_radius : ℝ := 8
def p_A : Prop := True  -- stand-in for the point A on the circle
def p_B : Prop := True  -- stand-in for the point B on the circle
def dist_AB : ℝ := 10
def p_D : Prop := True  -- stand-in for point D opposite B

theorem length_AD_of_circle 
  (r : ℝ := circle_radius)
  (A B D : Prop)
  (h_AB : dist_AB = 10)
  (h_radius : r = 8)
  (h_opposite : D)
  : ∃ AD : ℝ, AD = Real.sqrt 252.75 :=
sorry

end length_AD_of_circle_l1707_170711


namespace no_adjacent_numbers_differ_by_10_or_multiple_10_l1707_170744

theorem no_adjacent_numbers_differ_by_10_or_multiple_10 :
  ¬ ∃ (f : Fin 25 → Fin 25),
    (∀ n : Fin 25, f (n + 1) - f n = 10 ∨ (f (n + 1) - f n) % 10 = 0) :=
by
  sorry

end no_adjacent_numbers_differ_by_10_or_multiple_10_l1707_170744


namespace num_math_books_l1707_170762

theorem num_math_books (total_books total_cost math_book_cost history_book_cost : ℕ) (M H : ℕ)
  (h1 : total_books = 80)
  (h2 : math_book_cost = 4)
  (h3 : history_book_cost = 5)
  (h4 : total_cost = 368)
  (h5 : M + H = total_books)
  (h6 : math_book_cost * M + history_book_cost * H = total_cost) :
  M = 32 :=
by
  sorry

end num_math_books_l1707_170762


namespace coat_price_calculation_l1707_170778

noncomputable def effective_price (initial_price : ℝ) 
  (reduction1 reduction2 reduction3 : ℝ) 
  (tax1 tax2 tax3 : ℝ) : ℝ :=
  let price_after_first_month := initial_price * (1 - reduction1 / 100) * (1 + tax1 / 100)
  let price_after_second_month := price_after_first_month * (1 - reduction2 / 100) * (1 + tax2 / 100)
  let price_after_third_month := price_after_second_month * (1 - reduction3 / 100) * (1 + tax3 / 100)
  price_after_third_month

noncomputable def total_percent_reduction (initial_price final_price : ℝ) : ℝ :=
  (initial_price - final_price) / initial_price * 100

theorem coat_price_calculation :
  let original_price := 500
  let price_final := effective_price original_price 10 15 20 5 8 6
  let reduction_percentage := total_percent_reduction original_price price_final
  price_final = 367.824 ∧ reduction_percentage = 26.44 :=
by
  sorry

end coat_price_calculation_l1707_170778


namespace expand_polynomial_l1707_170731

theorem expand_polynomial (x : ℝ) : 
  3 * (x - 2) * (x^2 + x + 1) = 3 * x^3 - 3 * x^2 - 3 * x - 6 :=
by
  sorry

end expand_polynomial_l1707_170731


namespace smallest_value_fraction_l1707_170788

theorem smallest_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 6) :
  ∃ k : ℝ, (∀ (x y : ℝ), (-6 ≤ x ∧ x ≤ -3) → (3 ≤ y ∧ y ≤ 6) → k ≤ (x + y) / x) ∧ k = 0 :=
by
  sorry

end smallest_value_fraction_l1707_170788


namespace problem_proof_l1707_170789

theorem problem_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b = ab → a + 4 * b = 9) ∧
  (a + b = 1 → ∀ a b,  2^a + 2^(b + 1) ≥ 4) ∧
  (a + b = ab → 1 / a^2 + 2 / b^2 = 2 / 3) ∧
  (a + b = 1 → ∀ a b,  2 * a / (a + b^2) + b / (a^2 + b) = (2 * Real.sqrt 3 / 3) + 1) :=
by
  sorry

end problem_proof_l1707_170789


namespace burn_down_village_in_1920_seconds_l1707_170758

-- Definitions of the initial conditions
def initial_cottages : Nat := 90
def burn_interval_seconds : Nat := 480
def burn_time_per_unit : Nat := 5
def max_burns_per_interval : Nat := burn_interval_seconds / burn_time_per_unit

-- Recurrence relation for the number of cottages after n intervals
def cottages_remaining (n : Nat) : Nat :=
if n = 0 then initial_cottages
else 2 * cottages_remaining (n - 1) - max_burns_per_interval

-- Time taken to burn all cottages is when cottages_remaining(n) becomes 0
def total_burn_time_seconds (intervals : Nat) : Nat :=
intervals * burn_interval_seconds

-- Main theorem statement
theorem burn_down_village_in_1920_seconds :
  ∃ n, cottages_remaining n = 0 ∧ total_burn_time_seconds n = 1920 := by
  sorry

end burn_down_village_in_1920_seconds_l1707_170758


namespace solve_system_of_equations_l1707_170791

theorem solve_system_of_equations :
  ∃ (x y : ℝ), (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ∧ (x = 5) ∧ (y = 5) :=
by
  sorry

end solve_system_of_equations_l1707_170791


namespace flat_path_time_l1707_170733

/-- Malcolm's walking time problem -/
theorem flat_path_time (x : ℕ) (h1 : 6 + 12 + 6 = 24)
                       (h2 : 3 * x = 24 + 18) : x = 14 := 
by
  sorry

end flat_path_time_l1707_170733


namespace cylinder_surface_area_minimization_l1707_170746

theorem cylinder_surface_area_minimization (S V r h : ℝ) (h₁ : π * r^2 * h = V) (h₂ : r^2 + (h / 2)^2 = S^2) : (h / r) = 2 :=
sorry

end cylinder_surface_area_minimization_l1707_170746


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l1707_170721

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l1707_170721


namespace square_area_l1707_170729

theorem square_area (XY ZQ : ℕ) (inscribed_square : Prop) : (XY = 35) → (ZQ = 65) → inscribed_square → ∃ (a : ℕ), a^2 = 2275 :=
by
  intros hXY hZQ hinscribed
  use 2275
  sorry

end square_area_l1707_170729


namespace largest_constant_D_l1707_170764

theorem largest_constant_D (D : ℝ) 
  (h : ∀ (x y : ℝ), x^2 + y^2 + 4 ≥ D * (x + y)) : 
  D ≤ 2 * Real.sqrt 2 :=
sorry

end largest_constant_D_l1707_170764


namespace fraction_value_l1707_170742

theorem fraction_value :
  (2015^2 : ℤ) / (2014^2 + 2016^2 - 2) = (1 : ℚ) / 2 :=
by
  sorry

end fraction_value_l1707_170742


namespace find_smallest_even_number_l1707_170753

theorem find_smallest_even_number (n : ℕ) (h : n + (n + 2) + (n + 4) = 162) : n = 52 :=
by
  sorry

end find_smallest_even_number_l1707_170753


namespace negative_expression_l1707_170738

noncomputable def U : ℝ := -2.5
noncomputable def V : ℝ := -0.8
noncomputable def W : ℝ := 0.4
noncomputable def X : ℝ := 1.0
noncomputable def Y : ℝ := 2.2

theorem negative_expression :
  (U - V < 0) ∧ ¬(U * V < 0) ∧ ¬((X / V) * U < 0) ∧ ¬(W / (U * V) < 0) ∧ ¬((X + Y) / W < 0) :=
by
  sorry

end negative_expression_l1707_170738


namespace arrangement_possible_l1707_170726

noncomputable def exists_a_b : Prop :=
  ∃ a b : ℝ, a + 2*b > 0 ∧ 7*a + 13*b < 0

theorem arrangement_possible : exists_a_b := by
  sorry

end arrangement_possible_l1707_170726


namespace recipe_sugar_amount_l1707_170739

theorem recipe_sugar_amount (F_total F_added F_additional F_needed S : ℕ)
  (h1 : F_total = 9)
  (h2 : F_added = 2)
  (h3 : F_additional = S + 1)
  (h4 : F_needed = F_total - F_added)
  (h5 : F_needed = F_additional) :
  S = 6 := 
sorry

end recipe_sugar_amount_l1707_170739


namespace time_expression_l1707_170716

theorem time_expression (h V₀ g S V t : ℝ) :
  (V = g * t + V₀) →
  (S = h + (1 / 2) * g * t^2 + V₀ * t) →
  t = (2 * (S - h)) / (V + V₀) :=
by
  intro h_eq v_eq
  sorry

end time_expression_l1707_170716


namespace find_value_of_expression_l1707_170712

theorem find_value_of_expression (x y : ℝ) 
  (h1 : 4 * x + 2 * y = 20)
  (h2 : 2 * x + 4 * y = 16) : 
  4 * x ^ 2 + 12 * x * y + 12 * y ^ 2 = 292 :=
by
  sorry

end find_value_of_expression_l1707_170712


namespace polynomial_coefficients_sum_l1707_170771

theorem polynomial_coefficients_sum :
  let a := -15
  let b := 69
  let c := -81
  let d := 27
  10 * a + 5 * b + 2 * c + d = 60 :=
by
  let a := -15
  let b := 69
  let c := -81
  let d := 27
  sorry

end polynomial_coefficients_sum_l1707_170771


namespace skilled_picker_capacity_minimize_costs_l1707_170715

theorem skilled_picker_capacity (x : ℕ) (h1 : ∀ x : ℕ, ∀ s : ℕ, s = 3 * x) (h2 : 450 * 25 = 3 * x * 25 + 600) :
  s = 30 :=
by
  sorry

theorem minimize_costs (s n m : ℕ)
(h1 : s ≤ 20)
(h2 : n ≤ 15)
(h3 : 600 = s * 30 + n * 10)
(h4 : ∀ y, y = s * 300 + n * 80) :
  m = 15 ∧ s = 15 :=
by
  sorry

end skilled_picker_capacity_minimize_costs_l1707_170715


namespace cost_of_jacket_is_60_l1707_170761

/-- Define the constants from the problem --/
def cost_of_shirt : ℕ := 8
def cost_of_pants : ℕ := 18
def shirts_bought : ℕ := 4
def pants_bought : ℕ := 2
def jackets_bought : ℕ := 2
def carrie_paid : ℕ := 94

/-- Define the problem statement --/
theorem cost_of_jacket_is_60 (total_cost jackets_cost : ℕ) 
    (H1 : total_cost = (shirts_bought * cost_of_shirt) + (pants_bought * cost_of_pants) + jackets_cost)
    (H2 : carrie_paid = total_cost / 2)
    : jackets_cost / jackets_bought = 60 := 
sorry

end cost_of_jacket_is_60_l1707_170761


namespace parabola_circle_intersection_l1707_170718

theorem parabola_circle_intersection :
  (∃ x y : ℝ, y = (x - 2)^2 ∧ x + 1 = (y + 2)^2) →
  (∃ r : ℝ, ∀ x y : ℝ, (y = (x - 2)^2 ∧ x + 1 = (y + 2)^2) →
    (x - 5/2)^2 + (y + 3/2)^2 = r^2 ∧ r^2 = 3/2) :=
by
  intros
  sorry

end parabola_circle_intersection_l1707_170718


namespace money_left_in_wallet_l1707_170751

def initial_amount := 106
def spent_supermarket := 31
def spent_showroom := 49

theorem money_left_in_wallet : initial_amount - spent_supermarket - spent_showroom = 26 := by
  sorry

end money_left_in_wallet_l1707_170751


namespace g_diff_l1707_170736

def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 - 2 * x - 1

theorem g_diff (x h : ℝ) : g (x + h) - g x = h * (6 * x^2 + 6 * x * h + 2 * h^2 + 10 * x + 5 * h - 2) := 
by
  sorry

end g_diff_l1707_170736


namespace Dorothy_found_57_pieces_l1707_170794

def total_pieces_Dorothy_found 
  (B_green B_red R_red R_blue : ℕ)
  (D_red_factor D_blue_factor : ℕ)
  (H1 : B_green = 12)
  (H2 : B_red = 3)
  (H3 : R_red = 9)
  (H4 : R_blue = 11)
  (H5 : D_red_factor = 2)
  (H6 : D_blue_factor = 3) : ℕ := 
  let D_red := D_red_factor * (B_red + R_red)
  let D_blue := D_blue_factor * R_blue
  D_red + D_blue

theorem Dorothy_found_57_pieces 
  (B_green B_red R_red R_blue : ℕ)
  (D_red_factor D_blue_factor : ℕ)
  (H1 : B_green = 12)
  (H2 : B_red = 3)
  (H3 : R_red = 9)
  (H4 : R_blue = 11)
  (H5 : D_red_factor = 2)
  (H6 : D_blue_factor = 3) :
  total_pieces_Dorothy_found B_green B_red R_red R_blue D_red_factor D_blue_factor H1 H2 H3 H4 H5 H6 = 57 := by
    sorry

end Dorothy_found_57_pieces_l1707_170794


namespace quilt_shaded_fraction_l1707_170750

theorem quilt_shaded_fraction (total_squares : ℕ) (fully_shaded : ℕ) (half_shaded_squares : ℕ) (half_shades_per_square: ℕ) : 
  (((fully_shaded) + (half_shaded_squares * half_shades_per_square / 2)) / total_squares) = (1 / 4) :=
by 
  let fully_shaded := 2
  let half_shaded_squares := 4
  let half_shades_per_square := 1
  let total_squares := 16
  sorry

end quilt_shaded_fraction_l1707_170750


namespace system_of_equations_correct_l1707_170783

-- Define the problem conditions
variable (x y : ℝ) -- Define the productivity of large and small harvesters

-- Define the correct system of equations as per the problem
def system_correct : Prop := (2 * (2 * x + 5 * y) = 3.6) ∧ (5 * (3 * x + 2 * y) = 8)

-- State the theorem to prove the correctness of the system of equations under given conditions
theorem system_of_equations_correct (x y : ℝ) : (2 * (2 * x + 5 * y) = 3.6) ∧ (5 * (3 * x + 2 * y) = 8) :=
by
  sorry

end system_of_equations_correct_l1707_170783


namespace grogg_expected_value_l1707_170703

theorem grogg_expected_value (n : ℕ) (p : ℝ) (h_n : 2 ≤ n) (h_p : 0 < p ∧ p < 1) :
  (p + n * p^n * (1 - p) = 1) ↔ (p = 1 / n^(1/n:ℝ)) :=
sorry

end grogg_expected_value_l1707_170703


namespace candy_per_bag_correct_l1707_170749

def total_candy : ℕ := 648
def sister_candy : ℕ := 48
def friends : ℕ := 3
def bags : ℕ := 8

def remaining_candy (total candy_kept : ℕ) : ℕ := total - candy_kept
def candy_per_person (remaining people : ℕ) : ℕ := remaining / people
def candy_per_bag (per_person bags : ℕ) : ℕ := per_person / bags

theorem candy_per_bag_correct :
  candy_per_bag (candy_per_person (remaining_candy total_candy sister_candy) (friends + 1)) bags = 18 :=
by
  sorry

end candy_per_bag_correct_l1707_170749


namespace correct_inequality_l1707_170728

variable (a b c d : ℝ)
variable (h₁ : a > b)
variable (h₂ : b > 0)
variable (h₃ : 0 > c)
variable (h₄ : c > d)

theorem correct_inequality :
  (c / a) - (d / b) > 0 :=
by sorry

end correct_inequality_l1707_170728


namespace three_distinct_numbers_l1707_170781

theorem three_distinct_numbers (s : ℕ) (A : Finset ℕ) (S : Finset ℕ) (hA : A = Finset.range (4 * s + 1) \ Finset.range 1)
  (hS : S ⊆ A) (hcard: S.card = 2 * s + 2) :
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x + y = 2 * z :=
by
  sorry

end three_distinct_numbers_l1707_170781


namespace find_valid_pairs_l1707_170765

theorem find_valid_pairs :
  ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 12 ∧ 1 ≤ b ∧ b ≤ 12 →
  (∃ C : ℤ, ∀ (n : ℕ), 0 < n → (a^n + b^(n+9)) % 13 = C % 13) ↔
  (a, b) = (1, 1) ∨ (a, b) = (4, 4) ∨ (a, b) = (10, 10) ∨ (a, b) = (12, 12) := 
by
  sorry

end find_valid_pairs_l1707_170765


namespace compute_y_geometric_series_l1707_170740

theorem compute_y_geometric_series :
  let S1 := (∑' n : ℕ, (1 / 3)^n)
  let S2 := (∑' n : ℕ, (-1)^n * (1 / 3)^n)
  (S1 * S2 = ∑' n : ℕ, (1 / 9)^n) → 
  S1 = 3 / 2 →
  S2 = 3 / 4 →
  (∑' n : ℕ, (1 / y)^n) = 9 / 8 →
  y = 9 := 
by
  intros S1 S2 h₁ h₂ h₃ h₄
  sorry

end compute_y_geometric_series_l1707_170740


namespace value_of_a_l1707_170724

theorem value_of_a (a x : ℝ) (h : x = 4) (h_eq : x^2 - 3 * x = a^2) : a = 2 ∨ a = -2 :=
by
  -- The proof is omitted, but the theorem statement adheres to the problem conditions and expected result.
  sorry

end value_of_a_l1707_170724
