import Mathlib

namespace proof_problem_l270_270920

def f (x : ℤ) : ℤ := 2 * x - 1
def g (x : ℤ) : ℤ := x^2 + 2 * x + 1

theorem proof_problem : f (g 3) - g (f 3) = -5 := by
  sorry

end proof_problem_l270_270920


namespace find_b_l270_270083

noncomputable def a : ℂ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℂ := sorry

-- Given conditions
axiom sum_eq : a + b + c = 4
axiom prod_pairs_eq : a * b + b * c + c * a = 5
axiom prod_triple_eq : a * b * c = 6

-- Prove that b = 1
theorem find_b : b = 1 :=
by
  -- Proof omitted
  sorry

end find_b_l270_270083


namespace sumata_family_total_miles_l270_270591

theorem sumata_family_total_miles
  (days : ℝ) (miles_per_day : ℝ)
  (h1 : days = 5.0)
  (h2 : miles_per_day = 250) : 
  miles_per_day * days = 1250 := 
by
  sorry

end sumata_family_total_miles_l270_270591


namespace bailey_chew_toys_l270_270017

theorem bailey_chew_toys (dog_treats rawhide_bones: ℕ) (cards items_per_card : ℕ)
  (h1 : dog_treats = 8)
  (h2 : rawhide_bones = 10)
  (h3 : cards = 4)
  (h4 : items_per_card = 5) :
  ∃ chew_toys : ℕ, chew_toys = 2 :=
by
  sorry

end bailey_chew_toys_l270_270017


namespace fraction_zero_numerator_l270_270416

theorem fraction_zero_numerator (x : ℝ) (h₁ : (x^2 - 9) / (x + 3) = 0) (h₂ : x + 3 ≠ 0) : x = 3 :=
sorry

end fraction_zero_numerator_l270_270416


namespace f_correct_l270_270379

noncomputable def f (n : ℕ) : ℕ :=
  if h : n ≥ 15 then (n - 1) / 2
  else if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 2
  else if n = 6 then 4
  else if 7 ≤ n ∧ n ≤ 15 then 7
  else 0

theorem f_correct (n : ℕ) (hn : n ≥ 3) : 
  f n = if n ≥ 15 then (n - 1) / 2
        else if n = 3 then 1
        else if n = 4 then 1
        else if n = 5 then 2
        else if n = 6 then 4
        else if 7 ≤ n ∧ n ≤ 15 then 7
        else 0 := sorry

end f_correct_l270_270379


namespace bell_peppers_needed_l270_270447

-- Definitions based on the conditions
def large_slices_per_bell_pepper : ℕ := 20
def small_pieces_from_half_slices : ℕ := (20 / 2) * 3
def total_slices_and_pieces_per_bell_pepper : ℕ := large_slices_per_bell_pepper / 2 + small_pieces_from_half_slices
def desired_total_slices_and_pieces : ℕ := 200

-- Proving the number of bell peppers needed
theorem bell_peppers_needed : 
  desired_total_slices_and_pieces / total_slices_and_pieces_per_bell_pepper = 5 := 
by 
  -- Add the proof steps here
  sorry

end bell_peppers_needed_l270_270447


namespace cos_segments_ratio_proof_l270_270118

open Real

noncomputable def cos_segments_ratio := 
  let p := 5
  let q := 26
  ∀ x : ℝ, (cos x = cos 50) → (p, q) = (5, 26)

theorem cos_segments_ratio_proof : cos_segments_ratio :=
by 
  sorry

end cos_segments_ratio_proof_l270_270118


namespace problem_l270_270873

noncomputable def f (x a : ℝ) : ℝ := (1/2) * x ^ 2 - x - a * Real.log (x - a)

def monotonicity_f (a : ℝ) : Prop :=
  if a = 0 then
    ∀ x : ℝ, 0 < x → (x < 1 → f x 0 < f (x + 1) 0) ∧ (x > 1 → f x 0 > f (x + 1) 0)
  else if a > 0 then
    ∀ x : ℝ, a < x → (x < a + 1 → f x a < f (x + 1) a) ∧ (x > a + 1 → f x a > f (x + 1) a)
  else if -1 < a ∧ a < 0 then
    ∀ x : ℝ, 0 < x → (x < a + 1 → f x a < f (x + 1) a) ∧ (x > a + 1 → f (x + 1) a > f x a)
  else if a = -1 then
    ∀ x : ℝ, -1 < x → f x (-1) < f (x + 1) (-1)
  else
    ∀ x : ℝ, a < x → (x < 0 → f (x + 1) a > f x a) ∧ (0 < x → f x a > f (x + 1) a)

noncomputable def g (x a : ℝ) : ℝ := f (x + a) a - a * (x + (1/2) * a - 1)

def extreme_points (x₁ x₂ a : ℝ) : Prop :=
  x₁ < x₂ ∧ ∀ x : ℝ, 0 < x → x < 1 → g x a = 0

theorem problem (a : ℝ) (x₁ x₂ : ℝ) (hx : extreme_points x₁ x₂ a) (h_dom : -1/4 < a ∧ a < 0) :
  0 < f x₁ a - f x₂ a ∧ f x₁ a - f x₂ a < 1/2 := sorry

end problem_l270_270873


namespace point_not_in_region_l270_270168

theorem point_not_in_region (A B C D : ℝ × ℝ) :
  (A = (0, 0) ∧ 3 * A.1 + 2 * A.2 < 6) ∧
  (B = (1, 1) ∧ 3 * B.1 + 2 * B.2 < 6) ∧
  (C = (0, 2) ∧ 3 * C.1 + 2 * C.2 < 6) ∧
  (D = (2, 0) ∧ ¬ ( 3 * D.1 + 2 * D.2 < 6 )) :=
by {
  sorry
}

end point_not_in_region_l270_270168


namespace mean_of_xyz_l270_270116

theorem mean_of_xyz (mean7 : ℕ) (mean10 : ℕ) (x y z : ℕ) (h1 : mean7 = 40) (h2 : mean10 = 50) : (x + y + z) / 3 = 220 / 3 :=
by
  have sum7 := 7 * mean7
  have sum10 := 10 * mean10
  have sum_xyz := sum10 - sum7
  have mean_xyz := sum_xyz / 3
  sorry

end mean_of_xyz_l270_270116


namespace scientific_notation_of_100000_l270_270543

theorem scientific_notation_of_100000 :
  100000 = 1 * 10^5 :=
by sorry

end scientific_notation_of_100000_l270_270543


namespace jason_current_cards_l270_270913

-- Define the initial number of Pokemon cards Jason had.
def initial_cards : ℕ := 9

-- Define the number of Pokemon cards Jason gave to his friends.
def given_away : ℕ := 4

-- Prove that the number of Pokemon cards he has now is 5.
theorem jason_current_cards : initial_cards - given_away = 5 := by
  sorry

end jason_current_cards_l270_270913


namespace intersection_A_B_l270_270388

-- Definitions of the sets A and B according to the problem conditions
def A : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}
def B : Set ℝ := {y | ∃ x : ℝ, y = Real.sqrt (-x^2 + 4 * x - 3)}

-- The proof problem statement
theorem intersection_A_B :
  A ∩ B = {y | 0 ≤ y ∧ y ≤ 1} :=
by
  sorry

end intersection_A_B_l270_270388


namespace combined_weight_of_three_parcels_l270_270438

theorem combined_weight_of_three_parcels (x y z : ℕ)
  (h1 : x + y = 112) (h2 : y + z = 146) (h3 : z + x = 132) :
  x + y + z = 195 :=
by
  sorry

end combined_weight_of_three_parcels_l270_270438


namespace negation_of_exists_l270_270234

theorem negation_of_exists (x : ℝ) : ¬(∃ x_0 : ℝ, |x_0| + x_0^2 < 0) ↔ ∀ x : ℝ, |x| + x^2 ≥ 0 :=
by
  sorry

end negation_of_exists_l270_270234


namespace find_divisor_l270_270737

-- Define the problem specifications
def divisor_problem (D Q R d : ℕ) : Prop :=
  D = d * Q + R

-- The specific instance with given values
theorem find_divisor :
  divisor_problem 15968 89 37 179 :=
by
  -- Proof omitted
  sorry

end find_divisor_l270_270737


namespace ratio_squirrels_to_raccoons_l270_270554

def animals_total : ℕ := 84
def raccoons : ℕ := 12
def squirrels : ℕ := animals_total - raccoons

theorem ratio_squirrels_to_raccoons : (squirrels : ℚ) / raccoons = 6 :=
by
  sorry

end ratio_squirrels_to_raccoons_l270_270554


namespace sqrt_57_in_range_l270_270642

theorem sqrt_57_in_range (h1 : 49 < 57) (h2 : 57 < 64) (h3 : 7^2 = 49) (h4 : 8^2 = 64) : 7 < Real.sqrt 57 ∧ Real.sqrt 57 < 8 := by
  sorry

end sqrt_57_in_range_l270_270642


namespace sequence_statements_correct_l270_270670

theorem sequence_statements_correct (S : ℕ → ℝ) (a : ℕ → ℝ) (T : ℕ → ℝ) 
(h_S_nonzero : ∀ n, n > 0 → S n ≠ 0)
(h_S_T_relation : ∀ n, n > 0 → S n + T n = S n * T n) :
  (a 1 = 2) ∧ (∀ n, n > 0 → T n - T (n - 1) = 1) ∧ (∀ n, n > 0 → S n = (n + 1) / n) :=
by
  sorry

end sequence_statements_correct_l270_270670


namespace continuity_at_4_l270_270988

def f (x : ℝ) : ℝ := -2 * x^2 + 9

theorem continuity_at_4 : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 4| < δ → |f x + 23| < ε := by
  sorry

end continuity_at_4_l270_270988


namespace relationship_among_a_b_c_l270_270222

noncomputable def a : ℝ := 3 ^ Real.cos (Real.pi / 6)
noncomputable def b : ℝ := Real.log (Real.sin (Real.pi / 6)) / Real.log (1 / 3)
noncomputable def c : ℝ := Real.log (Real.tan (Real.pi / 6)) / Real.log 2

theorem relationship_among_a_b_c : a > b ∧ b > c := 
by
  sorry

end relationship_among_a_b_c_l270_270222


namespace relationship_between_abc_l270_270662

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

theorem relationship_between_abc (h1 : 2^a = Real.log (1/a) / Real.log 2)
                                 (h2 : Real.log b / Real.log 2 = 2)
                                 (h3 : c = Real.log 2 + Real.log 3 - Real.log 7) :
  b > a ∧ a > c :=
sorry

end relationship_between_abc_l270_270662


namespace father_l270_270931

theorem father's_age_at_middle_son_birth (a b c F : ℕ) 
  (h1 : a = b + c) 
  (h2 : F * a * b * c = 27090) : 
  F - b = 34 :=
by sorry

end father_l270_270931


namespace randy_biscuits_l270_270576

theorem randy_biscuits (F : ℕ) (initial_biscuits mother_biscuits brother_ate remaining_biscuits : ℕ) 
  (h_initial : initial_biscuits = 32)
  (h_mother : mother_biscuits = 15)
  (h_brother : brother_ate = 20)
  (h_remaining : remaining_biscuits = 40)
  : ((initial_biscuits + mother_biscuits + F) - brother_ate) = remaining_biscuits → F = 13 := 
by
  intros h_eq
  sorry

end randy_biscuits_l270_270576


namespace unique_rectangle_Q_l270_270219

noncomputable def rectangle_Q_count (a : ℝ) :=
  let x := (3 * a) / 2
  let y := a / 2
  if x < 2 * a then 1 else 0

-- The main theorem
theorem unique_rectangle_Q (a : ℝ) (h : a > 0) :
  rectangle_Q_count a = 1 :=
sorry

end unique_rectangle_Q_l270_270219


namespace youtube_dislikes_l270_270730

theorem youtube_dislikes (x y : ℕ) 
  (h1 : x = 3 * y) 
  (h2 : x = 100 + 2 * y) 
  (h_y_increased : ∃ y' : ℕ, y' = 3 * y) :
  y' = 300 := by
  sorry

end youtube_dislikes_l270_270730


namespace Jason_seashells_l270_270736

theorem Jason_seashells (initial_seashells given_to_Tim remaining_seashells : ℕ) :
  initial_seashells = 49 → given_to_Tim = 13 → remaining_seashells = initial_seashells - given_to_Tim →
  remaining_seashells = 36 :=
by intros; sorry

end Jason_seashells_l270_270736


namespace train_length_is_correct_l270_270628

noncomputable def speed_kmhr : ℝ := 45
noncomputable def time_sec : ℝ := 30
noncomputable def bridge_length_m : ℝ := 235

noncomputable def speed_ms : ℝ := (speed_kmhr * 1000) / 3600
noncomputable def total_distance_m : ℝ := speed_ms * time_sec
noncomputable def train_length_m : ℝ := total_distance_m - bridge_length_m

theorem train_length_is_correct : train_length_m = 140 :=
by
  -- Placeholder to indicate that a proof should go here
  -- Proof is omitted as per the instructions
  sorry

end train_length_is_correct_l270_270628


namespace area_of_trapezium_l270_270856

def length_parallel_side1 : ℝ := 20
def length_parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 15
def expected_area : ℝ := 285

theorem area_of_trapezium :
  (1 / 2) * (length_parallel_side1 + length_parallel_side2) * distance_between_sides = expected_area :=
by sorry

end area_of_trapezium_l270_270856


namespace pure_imaginary_complex_number_l270_270249

theorem pure_imaginary_complex_number (m : ℝ) (h : (m^2 - 3*m) = 0) :
  (m^2 - 5*m + 6) ≠ 0 → m = 0 :=
by
  intro h_im
  have h_fact : (m = 0) ∨ (m = 3) := by
    sorry -- This is where the factorization steps would go
  cases h_fact with
  | inl h0 =>
    assumption
  | inr h3 =>
    exfalso
    have : (3^2 - 5*3 + 6) = 0 := by
      sorry -- Simplify to check that m = 3 is not a valid solution
    contradiction

end pure_imaginary_complex_number_l270_270249


namespace fraction_difference_l270_270815

-- Definitions for the problem conditions
def repeatingDecimal72 := 8 / 11
def decimal72 := 18 / 25

-- Statement that needs to be proven
theorem fraction_difference : repeatingDecimal72 - decimal72 = 2 / 275 := 
by 
  sorry

end fraction_difference_l270_270815


namespace solve_equation_l270_270305

theorem solve_equation (x : ℝ) (h : x ≠ 1) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) → x = -4 :=
by
  intro hyp
  sorry

end solve_equation_l270_270305


namespace fraction_difference_l270_270818

-- Definitions for the problem conditions
def repeatingDecimal72 := 8 / 11
def decimal72 := 18 / 25

-- Statement that needs to be proven
theorem fraction_difference : repeatingDecimal72 - decimal72 = 2 / 275 := 
by 
  sorry

end fraction_difference_l270_270818


namespace perm_mississippi_l270_270370

theorem perm_mississippi : 
  let n := 11
  let n_S := 4
  let n_I := 4
  let n_P := 2
  let n_M := 1
  ∏ (mississippi_perm := Nat.factorial n / (Nat.factorial n_S * Nat.factorial n_I * Nat.factorial n_P * Nat.factorial n_M)) :=
  mississippi_perm = 34650 :=
by sorry

end perm_mississippi_l270_270370


namespace man_owns_fraction_of_business_l270_270007

theorem man_owns_fraction_of_business
  (x : ℚ)
  (H1 : (3 / 4) * (x * 90000) = 45000)
  (H2 : x * 90000 = y) : 
  x = 2 / 3 := 
by
  sorry

end man_owns_fraction_of_business_l270_270007


namespace color_nat_two_colors_no_sum_power_of_two_l270_270268

theorem color_nat_two_colors_no_sum_power_of_two :
  ∃ (f : ℕ → ℕ), (∀ a b : ℕ, a ≠ b → f a = f b → ∃ c : ℕ, c > 0 ∧ c ≠ 1 ∧ c ≠ 2 ∧ (a + b ≠ 2 ^ c)) :=
sorry

end color_nat_two_colors_no_sum_power_of_two_l270_270268


namespace problem_solution_l270_270190

def problem_conditions : Prop :=
  (∃ (students_total excellent_students: ℕ) 
     (classA_excellent classB_not_excellent: ℕ),
     students_total = 110 ∧
     excellent_students = 30 ∧
     classA_excellent = 10 ∧
     classB_not_excellent = 30)

theorem problem_solution
  (students_total excellent_students: ℕ)
  (classA_excellent classB_not_excellent: ℕ)
  (h : problem_conditions) :
  ∃ classA_not_excellent classB_excellent: ℕ,
    classA_not_excellent = 50 ∧
    classB_excellent = 20 ∧
    ((∃ χ_squared: ℝ, χ_squared = 7.5 ∧ χ_squared > 6.635) → true) ∧
    (∃ selectA selectB: ℕ, selectA = 5 ∧ selectB = 3) :=
by {
  sorry
}

end problem_solution_l270_270190


namespace smallest_term_at_n_is_4_or_5_l270_270423

def a_n (n : ℕ) : ℝ :=
  n^2 - 9 * n - 100

theorem smallest_term_at_n_is_4_or_5 :
  ∃ n, n = 4 ∨ n = 5 ∧ a_n n = min (a_n 4) (a_n 5) :=
by
  sorry

end smallest_term_at_n_is_4_or_5_l270_270423


namespace probability_of_exactly_one_second_class_product_l270_270495

-- Definitions based on the conditions provided
def total_products := 100
def first_class_products := 90
def second_class_products := 10
def selected_products := 4

-- Calculation of the probability
noncomputable def probability : ℚ :=
  (Nat.choose 10 1 * Nat.choose 90 3) / Nat.choose 100 4

-- Statement to prove that the probability is 0.30
theorem probability_of_exactly_one_second_class_product : 
  probability = 0.30 := by
  sorry

end probability_of_exactly_one_second_class_product_l270_270495


namespace probability_of_neither_red_nor_purple_l270_270998

theorem probability_of_neither_red_nor_purple :
  let total_balls := 100
  let white_balls := 20
  let green_balls := 30
  let yellow_balls := 10
  let red_balls := 37
  let purple_balls := 3
  let neither_red_nor_purple_balls := white_balls + green_balls + yellow_balls
  (neither_red_nor_purple_balls : ℝ) / (total_balls : ℝ) = 0.6 :=
by
  sorry

end probability_of_neither_red_nor_purple_l270_270998


namespace example_one_example_two_l270_270164

-- We will define natural numbers corresponding to seven, seventy-seven, and seven hundred seventy-seven.
def seven : ℕ := 7
def seventy_seven : ℕ := 77
def seven_hundred_seventy_seven : ℕ := 777

-- We will define both solutions in the form of equalities producing 100.
theorem example_one : (seven_hundred_seventy_seven / seven) - (seventy_seven / seven) = 100 :=
  by sorry

theorem example_two : (seven * seven) + (seven * seven) + (seven / seven) + (seven / seven) = 100 :=
  by sorry

end example_one_example_two_l270_270164


namespace find_hyperbola_focus_l270_270361

theorem find_hyperbola_focus : ∃ (x y : ℝ), 
  2 * x ^ 2 - 3 * y ^ 2 + 8 * x - 12 * y - 8 = 0 
  → (x, y) = (-2 + (Real.sqrt 30)/3, -2) :=
by
  sorry

end find_hyperbola_focus_l270_270361


namespace S_7_is_28_l270_270460

-- Define the arithmetic sequence and sum of first n terms
def a : ℕ → ℝ := sorry  -- placeholder for arithmetic sequence
def S (n : ℕ) : ℝ := sorry  -- placeholder for the sum of first n terms

-- Given conditions
def a_3 : ℝ := 3
def a_10 : ℝ := 10

-- Define properties of the arithmetic sequence
axiom a_n_property (n : ℕ) : a n = a 1 + (n - 1) * (a 10 - a 3) / (10 - 3)

-- Define the sum of first n terms
axiom sum_property (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Given specific elements of the sequence
axiom a_3_property : a 3 = 3
axiom a_10_property : a 10 = 10

-- The statement to prove
theorem S_7_is_28 : S 7 = 28 :=
sorry

end S_7_is_28_l270_270460


namespace inequality_proof_l270_270573

theorem inequality_proof
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a * b + b * c + c * a = 1) :
  (a + b) / Real.sqrt (a * b * (1 - a * b)) + 
  (b + c) / Real.sqrt (b * c * (1 - b * c)) + 
  (c + a) / Real.sqrt (c * a * (1 - c * a)) ≤ Real.sqrt 2 / (a * b * c) :=
sorry

end inequality_proof_l270_270573


namespace tan_double_angle_l270_270216

theorem tan_double_angle (α : ℝ) (h₁ : Real.sin α = 4/5) (h₂ : α ∈ Set.Ioc (π / 2) π) :
  Real.tan (2 * α) = 24 / 7 := 
  sorry

end tan_double_angle_l270_270216


namespace average_speed_of_car_l270_270339

theorem average_speed_of_car : 
  let d1 := 80
  let d2 := 60
  let d3 := 40
  let d4 := 50
  let d5 := 30
  let d6 := 70
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  total_distance / total_time = 55 := 
by
  let d1 := 80
  let d2 := 60
  let d3 := 40
  let d4 := 50
  let d5 := 30
  let d6 := 70
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  show total_distance / total_time = 55
  sorry

end average_speed_of_car_l270_270339


namespace rectangle_is_square_l270_270617

theorem rectangle_is_square
  (a b: ℝ)  -- rectangle side lengths
  (h: a ≠ b)  -- initial assumption: rectangle not a square
  (shift_perpendicular: ∀ (P Q R S: ℝ × ℝ), (P ≠ Q → Q ≠ R → R ≠ S → S ≠ P) → (∀ (shift: ℝ × ℝ → ℝ × ℝ), ∀ (P₁: ℝ × ℝ), shift P₁ = P₁ + (0, 1) ∨ shift P₁ = P₁ + (1, 0)) → false):
  False := sorry

end rectangle_is_square_l270_270617


namespace fraction_difference_l270_270814

-- Definitions for the problem conditions
def repeatingDecimal72 := 8 / 11
def decimal72 := 18 / 25

-- Statement that needs to be proven
theorem fraction_difference : repeatingDecimal72 - decimal72 = 2 / 275 := 
by 
  sorry

end fraction_difference_l270_270814


namespace number_of_positive_integers_with_at_most_two_digits_l270_270691

theorem number_of_positive_integers_with_at_most_two_digits:
  -- There are 2151 positive integers less than 100000 with at most two different digits
  ∃ (n : ℕ), n = 2151 ∧
    (∀ k : ℕ, k < 100000 → 
      let digits := (nat.digits 10 k).erase_dup 
      in (1 ≤ digits.length ∧ digits.length ≤ 2) → 1 ≤ k ∧ k < 100000) :=
begin
  -- proof would go here
  sorry
end

end number_of_positive_integers_with_at_most_two_digits_l270_270691


namespace min_value_sqrt_expr_l270_270084

theorem min_value_sqrt_expr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  \(\frac{\sqrt{(x^2 + y^2)(4x^2 + y^2)}}{xy}\) ≥ 3 ∧ 
    (\(\frac{\sqrt{(x^2 + y^2)(4x^2 + y^2)}}{xy}\) = 3 ↔ y = x \(\sqrt{2}\)) :=
sorry

end min_value_sqrt_expr_l270_270084


namespace beaker_water_division_l270_270328

-- Given conditions
variable (buckets : ℕ) (bucket_capacity : ℕ) (remaining_water : ℝ)
  (total_buckets : ℕ := 2) (capacity : ℕ := 120) (remaining : ℝ := 2.4)

-- Theorem statement
theorem beaker_water_division (h1 : buckets = total_buckets)
                             (h2 : bucket_capacity = capacity)
                             (h3 : remaining_water = remaining) :
                             (total_water : ℝ := buckets * bucket_capacity + remaining_water ) → 
                             (water_per_beaker : ℝ := total_water / 3) →
                             water_per_beaker = 80.8 :=
by
  -- Skipping the proof steps here, will use sorry
  sorry

end beaker_water_division_l270_270328


namespace melanie_batches_l270_270103

theorem melanie_batches (total_brownies_given: ℕ)
                        (brownies_per_batch: ℕ)
                        (fraction_bake_sale: ℚ)
                        (fraction_container: ℚ)
                        (remaining_brownies_given: ℕ) :
                        brownies_per_batch = 20 →
                        fraction_bake_sale = 3/4 →
                        fraction_container = 3/5 →
                        total_brownies_given = 20 →
                        (remaining_brownies_given / (brownies_per_batch * (1 - fraction_bake_sale) * (1 - fraction_container))) = 10 :=
by
  sorry

end melanie_batches_l270_270103


namespace part_a_part_b_part_c_l270_270997

noncomputable theory

namespace BirthProbability

open Classical

-- Define the various facts and probabilities.
def equally_probable := (1/2 : ℝ)

def probability_one_boy_one_girl : ℝ := 1/2
def probability_given_one_is_boy : ℝ := 2/3
def probability_given_boy_born_on_monday : ℝ := 14/27

-- The first problem
theorem part_a
  (B G : Type)
  [fintype (B × G)]
  (h : ∀ bg : B × G, equally_probable) :
  (probability_one_boy_one_girl = 1/2) :=
sorry

-- The second problem
theorem part_b
  (B G : Type)
  [fintype (B × G)]
  (h : ∀ bg : B × G, (bg.fst = “boy” ∨ bg.snd = “boy”) ∧ equally_probable) :
  (probability_given_one_is_boy = 2/3) :=
sorry

-- The third problem
theorem part_c
  (B G : Type)
  [fintype (B × G)]
  (h : ∀ bg : B × G, (bg.fst = “boy” ∧ bg.snd.born_on(monday) ∧ equally_probable) ∨ (bg.snd = “boy” ∧ bg.snd.born_on(monday) ∧ equally_probable)) :
  (probability_given_boy_born_on_monday = 14/27) :=
sorry -- Placeholder for the detailed complex proof
end BirthProbability

end part_a_part_b_part_c_l270_270997


namespace infinite_solutions_x2_plus_2y2_eq_z2_no_nontrivial_solns_x2_plus_2y2_eq_z2_and_2x2_plus_y2_eq_t2_l270_270112

-- Define x, y, z to be natural numbers
def has_infinitely_many_solutions : Prop :=
  ∃ (x y z : ℕ), x^2 + 2 * y^2 = z^2

-- Prove that there are infinitely many such x, y, z
theorem infinite_solutions_x2_plus_2y2_eq_z2 : has_infinitely_many_solutions :=
  sorry

-- Define x, y, z, t to be integers and non-zero
def no_nontrivial_integer_quadruplets : Prop :=
  ∀ (x y z t : ℤ), 
    (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ t ≠ 0) → 
    ¬((x^2 + 2 * y^2 = z^2) ∧ (2 * x^2 + y^2 = t^2))

-- Prove that no nontrivial integer quadruplets exist
theorem no_nontrivial_solns_x2_plus_2y2_eq_z2_and_2x2_plus_y2_eq_t2 : no_nontrivial_integer_quadruplets :=
  sorry

end infinite_solutions_x2_plus_2y2_eq_z2_no_nontrivial_solns_x2_plus_2y2_eq_z2_and_2x2_plus_y2_eq_t2_l270_270112


namespace quadratic_range_and_value_l270_270230

theorem quadratic_range_and_value (k : ℝ) :
  (∃ x1 x2 : ℝ, (x1^2 + (2 * k - 1) * x1 + k^2 - 1 = 0) ∧
  (x2^2 + (2 * k - 1) * x2 + k^2 - 1 = 0)) →
  k ≤ 5 / 4 ∧ (∀ x1 x2 : ℝ, (x1^2 + (2 * k - 1) * x1 + k^2 - 1 = 0) ∧
  (x2^2 + (2 * k - 1) * x2 + k^2 - 1 = 0) ∧ (x1^2 + x2^2 = 16 + x1 * x2)) → k = -2 :=
by sorry

end quadratic_range_and_value_l270_270230


namespace election_votes_l270_270771

variable (V : ℝ)

theorem election_votes (h1 : 0.70 * V - 0.30 * V = 192) : V = 480 :=
by
  sorry

end election_votes_l270_270771


namespace min_value_sqrt_expr_l270_270085

theorem min_value_sqrt_expr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  \(\frac{\sqrt{(x^2 + y^2)(4x^2 + y^2)}}{xy}\) ≥ 3 ∧ 
    (\(\frac{\sqrt{(x^2 + y^2)(4x^2 + y^2)}}{xy}\) = 3 ↔ y = x \(\sqrt{2}\)) :=
sorry

end min_value_sqrt_expr_l270_270085


namespace correlation_statements_l270_270169

def heavy_snow_predicts_harvest_year (heavy_snow benefits_wheat : Prop) : Prop := benefits_wheat → heavy_snow
def great_teachers_produce_students (great_teachers outstanding_students : Prop) : Prop := great_teachers → outstanding_students
def smoking_is_harmful (smoking harmful_to_health : Prop) : Prop := smoking → harmful_to_health
def magpies_call_signifies_joy (magpies_call joy_signified : Prop) : Prop := joy_signified → magpies_call

theorem correlation_statements (heavy_snow benefits_wheat great_teachers outstanding_students smoking harmful_to_health magpies_call joy_signified : Prop)
  (H1 : heavy_snow_predicts_harvest_year heavy_snow benefits_wheat)
  (H2 : great_teachers_produce_students great_teachers outstanding_students)
  (H3 : smoking_is_harmful smoking harmful_to_health) :
  ¬ magpies_call_signifies_joy magpies_call joy_signified := sorry

end correlation_statements_l270_270169


namespace max_and_min_l270_270026

open Real

-- Define the function
def y (x : ℝ) : ℝ := 3 * x^4 - 6 * x^2 + 4

-- Define the interval
def a : ℝ := -1
def b : ℝ := 3

theorem max_and_min:
  ∃ (xmin xmax : ℝ), xmin = 1 ∧ xmax = 193 ∧
  (∀ x, a ≤ x ∧ x ≤ b → y(xmin) ≤ y x ∧ y x ≤ y(xmax)) :=
by
  sorry

end max_and_min_l270_270026


namespace roots_triple_relation_l270_270511

theorem roots_triple_relation (a b c : ℤ) (α β : ℤ)
    (h_quad : a ≠ 0)
    (h_roots : α + β = -b / a)
    (h_prod : α * β = c / a)
    (h_triple : β = 3 * α) :
    3 * b^2 = 16 * a * c :=
sorry

end roots_triple_relation_l270_270511


namespace nolan_monthly_savings_l270_270281

theorem nolan_monthly_savings (m k : ℕ) (H : 12 * m = 36 * k) : m = 3 * k := 
by sorry

end nolan_monthly_savings_l270_270281


namespace exists_linear_function_second_quadrant_l270_270572

theorem exists_linear_function_second_quadrant (k b : ℝ) (h1 : k > 0) (h2 : b > 0) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = k * x + b) ∧ (∀ x, x < 0 → f x > 0) :=
by
  -- Prove there exists a linear function of the form f(x) = kx + b with given conditions
  -- Skip the proof for now
  sorry

end exists_linear_function_second_quadrant_l270_270572


namespace moving_point_trajectory_and_perpendicularity_l270_270008

theorem moving_point_trajectory_and_perpendicularity :
  (∀ (x y: ℝ), (x ≠ -2 ∧ x ≠ 2) → 
    (let slope_PA := y / (x + 2),
         slope_PB := y / (x - 2) in
       slope_PA * slope_PB = -1/3) →
    (∀ (c d : Point), (non_zero_slope ? ? (line_through Q c)) ->
    (line_through Q (-1, 0) intersect curve_E) = {c, d} &
      (x^2 / 4 + 3*y^2 / 4 = 1  → 
        ∀ (a : Point), a = (-2, 0) → 
          let AC := vector a c,
          let AD := vector a d in
            dot_product AC AD = 0)) :=
sorry

end moving_point_trajectory_and_perpendicularity_l270_270008


namespace solve_problem_l270_270050

variable (a b c x : ℝ)

-- Conditions
def condition1 : Prop := ∀ x, ax^2 + bx + c > 0 ↔ -3 < x ∧ x < 2

-- Statements to prove
def statementA : Prop := a < 0
def statementB : Prop := a + b + c > 0
def statementD : Prop := ∀ x, (cx^2 + bx + a < 0 ↔ (-1/3 < x ∧ x < 1/2))

theorem solve_problem (h1 : condition1)
  (h2 : statementA)
  (h3 : statementB)
  (h4 : statementD) : a < 0 ∧ a + b + c > 0 ∧ ∀ x, (cx^2 + bx + a < 0 ↔ (-1/3 < x ∧ x < 1/2)) :=
by
  sorry

end solve_problem_l270_270050


namespace total_cost_of_gas_l270_270860

theorem total_cost_of_gas :
  ∃ x : ℚ, (4 * (x / 4) - 4 * (x / 7) = 40) ∧ x = 280 / 3 :=
by
  sorry

end total_cost_of_gas_l270_270860


namespace largest_product_of_three_numbers_l270_270474

open Finset

theorem largest_product_of_three_numbers : 
  ∀ (s : Finset ℤ), s = {-3, -2, -1, 4, 5} → 
  (∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b → a ≠ c → b ≠ c → True) →
  ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a * b * c = 30 :=
by
  -- This is the problem statement.
  sorry

end largest_product_of_three_numbers_l270_270474


namespace range_of_m_l270_270047

theorem range_of_m (f : ℝ → ℝ) {m : ℝ} (h_dec : ∀ x y, -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1 ∧ x < y → f x ≥ f y)
  (h_ineq : f (m - 1) > f (2 * m - 1)) : 0 < m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l270_270047


namespace line_tangent_to_circle_l270_270709

theorem line_tangent_to_circle (l : ℝ → ℝ) (P : ℝ × ℝ) 
  (hP1 : P = (0, 1)) (hP2 : ∀ x y : ℝ, x^2 + y^2 = 1 -> l x = y)
  (hTangent : ∀ x y : ℝ, l x = y ↔ x^2 + y^2 = 1 ∧ y = 1):
  l x = 1 := by
  sorry

end line_tangent_to_circle_l270_270709


namespace simplify_fraction_l270_270303

theorem simplify_fraction :
  ( (5^2010)^2 - (5^2008)^2 ) / ( (5^2009)^2 - (5^2007)^2 ) = 25 := by
  sorry

end simplify_fraction_l270_270303


namespace findValuesForFibSequence_l270_270446

noncomputable def maxConsecutiveFibonacciTerms (A B C : ℝ) : ℝ :=
  if A ≠ 0 then 4 else 0

theorem findValuesForFibSequence :
  maxConsecutiveFibonacciTerms (1/2) (-1/2) 2 = 4 ∧ maxConsecutiveFibonacciTerms (1/2) (1/2) 2 = 4 :=
by
  -- This statement will follow from the given conditions and the solution provided.
  sorry

end findValuesForFibSequence_l270_270446


namespace factorize_expression_l270_270023

theorem factorize_expression (x : ℝ) : 
  x^4 + 324 = (x^2 - 18 * x + 162) * (x^2 + 18 * x + 162) := 
sorry

end factorize_expression_l270_270023


namespace vasya_example_fewer_sevens_l270_270153

theorem vasya_example_fewer_sevens : 
  (777 / 7) - (77 / 7) = 100 :=
begin
  sorry
end

end vasya_example_fewer_sevens_l270_270153


namespace ratio_length_to_breadth_l270_270593

theorem ratio_length_to_breadth (l b : ℕ) (h1 : b = 14) (h2 : l * b = 588) : l / b = 3 :=
by
  sorry

end ratio_length_to_breadth_l270_270593


namespace yang_hui_problem_solution_l270_270424

theorem yang_hui_problem_solution (x : ℕ) (h : x * (x - 1) = 650) : x * (x - 1) = 650 :=
by
  exact h

end yang_hui_problem_solution_l270_270424


namespace intersection_A_B_l270_270058

-- Definitions based on conditions
variable (U : Set Int) (A B : Set Int)

#check Set

-- Given conditions
def U_def : Set Int := {-1, 3, 5, 7, 9}
def compl_U_A : Set Int := {-1, 9}
def B_def : Set Int := {3, 7, 9}

-- A is defined as the set difference of U and the complement of A in U
def A_def : Set Int := { x | x ∈ U_def ∧ ¬ (x ∈ compl_U_A) }

-- Theorem stating the intersection of A and B equals {3, 7}
theorem intersection_A_B : A_def ∩ B_def = {3, 7} :=
by
  -- Here would be the proof block, but we add 'sorry' to indicate it is unfinished.
  sorry

end intersection_A_B_l270_270058


namespace repeatingDecimal_exceeds_l270_270834

noncomputable def repeatingDecimalToFraction (d : ℚ) : ℚ := 
    -- Function to convert repeating decimal to fraction
    if d = 0.99999 then 1 else (d * 100 - d) / 99  -- Simplified conversion for demonstration

def decimalToFraction (d : ℚ) : ℚ :=
    -- Function to convert decimal to fraction
    if d = 0.72 then 18 / 25 else 0  -- Replace with actual conversion

theorem repeatingDecimal_exceeds (x y : ℚ) (hx : repeatingDecimalToFraction x = 8/11) (hy : decimalToFraction y = 18/25):
    x - y = 2 / 275 :=
by
    sorry

end repeatingDecimal_exceeds_l270_270834


namespace excess_common_fraction_l270_270806

theorem excess_common_fraction :
  let x := (72 / 10^2 + 7.2 * (10 / 10^2) * (10^{-2} / (1 - 10^{-2}))) in
  x - (72 / 100) = 8 / 1100 :=
by
  sorry

end excess_common_fraction_l270_270806


namespace g_ab_eq_zero_l270_270399

def g (x : ℤ) : ℤ := x^2 - 2013 * x

theorem g_ab_eq_zero (a b : ℤ) (h1 : g a = g b) (h2 : a ≠ b) : g (a + b) = 0 :=
by
  sorry

end g_ab_eq_zero_l270_270399


namespace repeating_decimal_difference_l270_270821

theorem repeating_decimal_difference :
  let x := 72 / 99 in
  let y := 72 / 100 in
  x - y = 2 / 275 := 
by
  -- we will add a proof later
  sorry

end repeating_decimal_difference_l270_270821


namespace excess_common_fraction_l270_270801

theorem excess_common_fraction :
  let x := (72 / 10^2 + 7.2 * (10 / 10^2) * (10^{-2} / (1 - 10^{-2}))) in
  x - (72 / 100) = 8 / 1100 :=
by
  sorry

end excess_common_fraction_l270_270801


namespace fencing_cost_l270_270122

theorem fencing_cost (w : ℝ) (h : ℝ) (p : ℝ) (cost_per_meter : ℝ) 
  (hw : h = w + 10) (perimeter : p = 220) (cost_rate : cost_per_meter = 6.5) : 
  ((p * cost_per_meter) = 1430) := by 
  sorry

end fencing_cost_l270_270122


namespace first_group_men_8_l270_270711

variable (x : ℕ)

theorem first_group_men_8 (h1 : x * 80 = 20 * 32) : x = 8 := by
  -- provide the proof here
  sorry

end first_group_men_8_l270_270711


namespace sally_spent_eur_l270_270301

-- Define the given conditions
def coupon_value : ℝ := 3
def peaches_total_usd : ℝ := 12.32
def cherries_original_usd : ℝ := 11.54
def discount_rate : ℝ := 0.1
def conversion_rate : ℝ := 0.85

-- Define the intermediate calculations
def cherries_discount_usd : ℝ := cherries_original_usd * discount_rate
def cherries_final_usd : ℝ := cherries_original_usd - cherries_discount_usd
def total_usd : ℝ := peaches_total_usd + cherries_final_usd
def total_eur : ℝ := total_usd * conversion_rate

-- The final statement to be proven
theorem sally_spent_eur : total_eur = 19.30 := by
  sorry

end sally_spent_eur_l270_270301


namespace lucas_change_l270_270099

-- Define the initial amount of money Lucas has
def initial_amount : ℕ := 20

-- Define the cost of one avocado
def cost_per_avocado : ℕ := 2

-- Define the number of avocados Lucas buys
def number_of_avocados : ℕ := 3

-- Calculate the total cost of avocados
def total_cost : ℕ := number_of_avocados * cost_per_avocado

-- Calculate the remaining amount of money (change)
def remaining_amount : ℕ := initial_amount - total_cost

-- The proposition to prove: Lucas brings home $14
theorem lucas_change : remaining_amount = 14 := by
  sorry

end lucas_change_l270_270099


namespace total_pie_eaten_l270_270323

theorem total_pie_eaten (s1 s2 s3 : ℚ) (h1 : s1 = 8/9) (h2 : s2 = 5/6) (h3 : s3 = 2/3) :
  s1 + s2 + s3 = 43/18 := by
  sorry

end total_pie_eaten_l270_270323


namespace intersection_point_in_polar_coordinates_l270_270716

theorem intersection_point_in_polar_coordinates (theta : ℝ) (rho : ℝ) (h₁ : theta = π / 3) (h₂ : rho = 2 * Real.cos theta) (h₃ : rho > 0) : rho = 1 :=
by
  -- Proof skipped
  sorry

end intersection_point_in_polar_coordinates_l270_270716


namespace reciprocal_inequalities_l270_270734

theorem reciprocal_inequalities (a b c : ℝ)
  (h1 : -1 < a ∧ a < -2/3)
  (h2 : -1/3 < b ∧ b < 0)
  (h3 : 1 < c) :
  1/c < 1/(b - a) ∧ 1/(b - a) < 1/(a * b) :=
by
  sorry

end reciprocal_inequalities_l270_270734


namespace perfect_square_m_value_l270_270615

theorem perfect_square_m_value (y m : ℤ) (h : ∃ k : ℤ, y^2 - 8 * y + m = (y - k)^2) : m = 16 :=
sorry

end perfect_square_m_value_l270_270615


namespace trig_identity_l270_270863

theorem trig_identity (x : ℝ) (h : Real.sin (x + π / 3) = 1 / 3) : 
  Real.sin ((5 * π) / 3 - x) - Real.cos (2 * x - π / 3) = 4 / 9 := 
by 
  sorry

end trig_identity_l270_270863


namespace eval_f_at_3_l270_270702

-- Define the polynomial function
def f (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 2 * x - 1

-- State the theorem to prove f(3) = 41
theorem eval_f_at_3 : f 3 = 41 :=
by
  -- Proof would go here
  sorry

end eval_f_at_3_l270_270702


namespace boys_to_girls_ratio_l270_270899

theorem boys_to_girls_ratio (T G : ℕ) (h : (1 / 2) * G = (1 / 6) * T) : (T - G) = 2 * G := by
  sorry

end boys_to_girls_ratio_l270_270899


namespace count_positive_integers_with_two_digits_l270_270677

theorem count_positive_integers_with_two_digits : 
  ∃ (count : ℕ), count = 2151 ∧ ∀ n, n < 100000 → (∀ d1 d2 d, d ∈ digits n → d = d1 ∨ d = d2) → n ∈ (range 100000) := sorry

end count_positive_integers_with_two_digits_l270_270677


namespace yellow_balls_count_l270_270338

theorem yellow_balls_count (r y : ℕ) (h1 : r = 9) (h2 : (r : ℚ) / (r + y) = 1 / 3) : y = 18 := 
by
  sorry

end yellow_balls_count_l270_270338


namespace part1_part2_l270_270517

-- Definitions of y1 and y2 based on given conditions
def y1 (x : ℝ) : ℝ := -x + 3
def y2 (x : ℝ) : ℝ := 2 + x

-- Prove for x such that y1 = y2
theorem part1 (x : ℝ) : y1 x = y2 x ↔ x = 1 / 2 := by
  sorry

-- Prove for x such that y1 = 2y2 + 5
theorem part2 (x : ℝ) : y1 x = 2 * y2 x + 5 ↔ x = -2 := by
  sorry

end part1_part2_l270_270517


namespace totalNumberOfBalls_l270_270377

def numberOfBoxes : ℕ := 3
def numberOfBallsPerBox : ℕ := 5

theorem totalNumberOfBalls : numberOfBoxes * numberOfBallsPerBox = 15 := 
by
  sorry

end totalNumberOfBalls_l270_270377


namespace example_one_example_two_l270_270165

-- We will define natural numbers corresponding to seven, seventy-seven, and seven hundred seventy-seven.
def seven : ℕ := 7
def seventy_seven : ℕ := 77
def seven_hundred_seventy_seven : ℕ := 777

-- We will define both solutions in the form of equalities producing 100.
theorem example_one : (seven_hundred_seventy_seven / seven) - (seventy_seven / seven) = 100 :=
  by sorry

theorem example_two : (seven * seven) + (seven * seven) + (seven / seven) + (seven / seven) = 100 :=
  by sorry

end example_one_example_two_l270_270165


namespace unique_plants_count_1320_l270_270600

open Set

variable (X Y Z : Finset ℕ)

def total_plants_X : ℕ := 600
def total_plants_Y : ℕ := 480
def total_plants_Z : ℕ := 420
def shared_XY : ℕ := 60
def shared_YZ : ℕ := 70
def shared_XZ : ℕ := 80
def shared_XYZ : ℕ := 30

theorem unique_plants_count_1320 : X.card = total_plants_X →
                                Y.card = total_plants_Y →
                                Z.card = total_plants_Z →
                                (X ∩ Y).card = shared_XY →
                                (Y ∩ Z).card = shared_YZ →
                                (X ∩ Z).card = shared_XZ →
                                (X ∩ Y ∩ Z).card = shared_XYZ →
                                (X ∪ Y ∪ Z).card = 1320 := 
by {
  sorry
}

end unique_plants_count_1320_l270_270600


namespace magnitude_squared_l270_270411

-- Let z be the complex number 3 + 4i
def z : ℂ := 3 + 4 * Complex.I

-- Prove that the magnitude of z squared equals 25
theorem magnitude_squared : Complex.abs z ^ 2 = 25 := by
  -- The term "by" starts the proof block, and "sorry" allows us to skip the proof details.
  sorry

end magnitude_squared_l270_270411


namespace right_triangle_hypotenuse_l270_270549

theorem right_triangle_hypotenuse (a b : ℝ) (m_a m_b : ℝ)
    (h1 : m_a = Real.sqrt (b^2 + (a / 2)^2))
    (h2 : m_b = Real.sqrt (a^2 + (b / 2)^2))
    (h3 : m_a = Real.sqrt 30)
    (h4 : m_b = 6) :
  Real.sqrt (4 * (a^2 + b^2)) = 2 * Real.sqrt 52.8 :=
by
  sorry

end right_triangle_hypotenuse_l270_270549


namespace first_rectangle_dimensions_second_rectangle_dimensions_l270_270953

theorem first_rectangle_dimensions (x y : ℕ) (h : x * y = 2 * (x + y) + 1) : (x = 7 ∧ y = 3) ∨ (x = 3 ∧ y = 7) :=
sorry

theorem second_rectangle_dimensions (a b : ℕ) (h : a * b = 2 * (a + b) - 1) : (a = 5 ∧ b = 3) ∨ (a = 3 ∧ b = 5) :=
sorry

end first_rectangle_dimensions_second_rectangle_dimensions_l270_270953


namespace find_q_zero_l270_270557

theorem find_q_zero
  (p q r : ℝ → ℝ)  -- Define p, q, r as functions from ℝ to ℝ (since they are polynomials)
  (h1 : ∀ x, r x = p x * q x + 2)  -- Condition 1: r(x) = p(x) * q(x) + 2
  (h2 : p 0 = 6)                   -- Condition 2: constant term of p(x) is 6
  (h3 : r 0 = 5)                   -- Condition 3: constant term of r(x) is 5
  : q 0 = 1 / 2 :=                 -- Conclusion: q(0) = 1/2
sorry

end find_q_zero_l270_270557


namespace bonnie_roark_wire_ratio_l270_270503

theorem bonnie_roark_wire_ratio :
  let bonnie_wire_length := 12 * 8
  let bonnie_cube_volume := 8 ^ 3
  let roark_cube_volume := 2
  let roark_edge_length := 1.5
  let roark_cube_edge_count := 12
  let num_roark_cubes := bonnie_cube_volume / roark_cube_volume
  let roark_wire_per_cube := roark_cube_edge_count * roark_edge_length
  let roark_total_wire := num_roark_cubes * roark_wire_per_cube
  bonnie_wire_length / roark_total_wire = 1 / 48 :=
  by
  sorry

end bonnie_roark_wire_ratio_l270_270503


namespace trapezoid_area_l270_270468

variable (x y : ℝ)

def condition1 : Prop := abs (y - 3 * x) ≥ abs (2 * y + x) ∧ -1 ≤ y - 3 ∧ y - 3 ≤ 1

def condition2 : Prop := (2 * y + y - y + 3 * x) * (2 * y + x + y - 3 * x) ≤ 0 ∧ 2 ≤ y ∧ y ≤ 4

theorem trapezoid_area (h1 : condition1 x y) (h2 : condition2 x y) :
  let A := (3, 2)
  let B := (-1/2, 2)
  let C := (-1, 4)
  let D := (6, 4)
  let S := (1/2) * (2 * (7 + 3.5))
  S = 10.5 :=
sorry

end trapezoid_area_l270_270468


namespace carpenter_job_duration_l270_270619

theorem carpenter_job_duration
  (total_estimate : ℤ)
  (carpenter_hourly_rate : ℤ)
  (assistant_hourly_rate : ℤ)
  (material_cost : ℤ)
  (H1 : total_estimate = 1500)
  (H2 : carpenter_hourly_rate = 35)
  (H3 : assistant_hourly_rate = 25)
  (H4 : material_cost = 720) :
  (total_estimate - material_cost) / (carpenter_hourly_rate + assistant_hourly_rate) = 13 :=
by
  sorry

end carpenter_job_duration_l270_270619


namespace product_of_second_largest_and_second_smallest_l270_270319

theorem product_of_second_largest_and_second_smallest (l : List ℕ) (h : l = [10, 11, 12]) :
  l.nthLe 1 (by norm_num [h]) * l.nthLe 1 (by norm_num [h]) = 121 := 
sorry

end product_of_second_largest_and_second_smallest_l270_270319


namespace sum_of_remainders_l270_270729

theorem sum_of_remainders (p : ℕ) (hp : p > 2) (hp_prime : Nat.Prime p)
    (a : ℕ → ℕ) (ha : ∀ k, a k = k^p % p^2) :
    (Finset.sum (Finset.range (p - 1)) a) = (p^3 - p^2) / 2 :=
by
  sorry

end sum_of_remainders_l270_270729


namespace jordan_no_quiz_probability_l270_270753

theorem jordan_no_quiz_probability (P_quiz : ℚ) (h : P_quiz = 5 / 9) :
  1 - P_quiz = 4 / 9 :=
by
  rw [h]
  exact sorry

end jordan_no_quiz_probability_l270_270753


namespace vasya_example_fewer_sevens_l270_270151

theorem vasya_example_fewer_sevens : 
  (777 / 7) - (77 / 7) = 100 :=
begin
  sorry
end

end vasya_example_fewer_sevens_l270_270151


namespace solve_inequality_l270_270945

theorem solve_inequality :
  {x : ℝ | (x^2 - 9) / (x - 3) > 0} = { x : ℝ | (-3 < x ∧ x < 3) ∨ (x > 3)} :=
by {
  sorry
}

end solve_inequality_l270_270945


namespace machine_production_time_difference_undetermined_l270_270566

theorem machine_production_time_difference_undetermined :
  ∀ (machineP_machineQ_440_hours_diff : ℝ)
    (machineQ_production_rate : ℝ)
    (machineA_production_rate : ℝ),
    machineA_production_rate = 4.000000000000005 →
    machineQ_production_rate = machineA_production_rate * 1.1 →
    machineP_machineQ_440_hours_diff > 0 →
    machineQ_production_rate * machineP_machineQ_440_hours_diff = 440 →
    ∃ machineP_production_rate, 
    ¬(∃ hours_diff : ℝ, hours_diff = 440 / machineP_production_rate - 440 / machineQ_production_rate) := sorry

end machine_production_time_difference_undetermined_l270_270566


namespace S_11_eq_22_l270_270535

variable {S : ℕ → ℕ}

-- Condition: given that S_8 - S_3 = 10
axiom h : S 8 - S 3 = 10

-- Proof goal: we want to show that S_11 = 22
theorem S_11_eq_22 : S 11 = 22 :=
by
  sorry

end S_11_eq_22_l270_270535


namespace Seth_boxes_initially_l270_270940

-- Define the initial conditions
def initial_boxes (x : ℕ) : Prop :=
  ∃ n : ℕ, 4 = (x - 1) / 2 ∧ x = 9

-- Prove that Seth initially bought 9 boxes of oranges
theorem Seth_boxes_initially : initial_boxes 9 :=
by {
  use 9,
  split,
  exact rfl,
  exact rfl,
}

end Seth_boxes_initially_l270_270940


namespace mary_fruits_l270_270928

noncomputable def totalFruitsLeft 
    (initial_apples: ℕ) (initial_oranges: ℕ) (initial_blueberries: ℕ) (initial_grapes: ℕ) (initial_kiwis: ℕ)
    (salad_apples: ℕ) (salad_oranges: ℕ) (salad_blueberries: ℕ)
    (snack_apples: ℕ) (snack_oranges: ℕ) (snack_kiwis: ℕ)
    (given_apples: ℕ) (given_oranges: ℕ) (given_blueberries: ℕ) (given_grapes: ℕ) (given_kiwis: ℕ) : ℕ :=
  let remaining_apples := initial_apples - salad_apples - snack_apples - given_apples
  let remaining_oranges := initial_oranges - salad_oranges - snack_oranges - given_oranges
  let remaining_blueberries := initial_blueberries - salad_blueberries - given_blueberries
  let remaining_grapes := initial_grapes - given_grapes
  let remaining_kiwis := initial_kiwis - snack_kiwis - given_kiwis
  remaining_apples + remaining_oranges + remaining_blueberries + remaining_grapes + remaining_kiwis

theorem mary_fruits :
    totalFruitsLeft 26 35 18 12 22 6 10 8 2 3 1 5 7 4 3 3 = 61 := by
  sorry

end mary_fruits_l270_270928


namespace cross_covers_two_rectangles_l270_270013

def Chessboard := Fin 8 × Fin 8

def is_cross (center : Chessboard) (point : Chessboard) : Prop :=
  (point.1 = center.1 ∧ (point.2 = center.2 - 1 ∨ point.2 = center.2 + 1)) ∨
  (point.2 = center.2 ∧ (point.1 = center.1 - 1 ∨ point.1 = center.1 + 1)) ∨
  (point = center)

def Rectangle_1x3 (rect : Fin 22) : Chessboard → Prop := sorry -- This represents Alina's rectangles
def Rectangle_1x2 (rect : Fin 22) : Chessboard → Prop := sorry -- This represents Polina's rectangles

theorem cross_covers_two_rectangles :
  ∃ center : Chessboard, 
    (∃ (rect_a : Fin 22), ∃ (rect_b : Fin 22), rect_a ≠ rect_b ∧ 
      (∀ p, is_cross center p → Rectangle_1x3 rect_a p) ∧ 
      (∀ p, is_cross center p → Rectangle_1x2 rect_b p)) ∨
    (∃ (rect_a : Fin 22), ∃ (rect_b : Fin 22), rect_a ≠ rect_b ∧ 
      (∀ p, is_cross center p → Rectangle_1x3 rect_a p) ∧ 
      (∀ p, is_cross center p → Rectangle_1x3 rect_b p)) ∨
    (∃ (rect_a : Fin 22), ∃ (rect_b : Fin 22), rect_a ≠ rect_b ∧ 
      (∀ p, is_cross center p → Rectangle_1x2 rect_a p) ∧ 
      (∀ p, is_cross center p → Rectangle_1x2 rect_b p)) :=
sorry

end cross_covers_two_rectangles_l270_270013


namespace minimum_value_l270_270885

noncomputable def condition (x : ℝ) : Prop := (2 * x - 1) / 3 - 1 ≥ x - (5 - 3 * x) / 2

noncomputable def target_function (x : ℝ) : ℝ := abs (x - 1) - abs (x + 3)

theorem minimum_value :
  ∃ x : ℝ, condition x ∧ ∀ y : ℝ, condition y → target_function y ≥ target_function x :=
sorry

end minimum_value_l270_270885


namespace minimum_value_expression_l270_270086

theorem minimum_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  3 ≤ (sqrt ((x^2 + y^2) * (4*x^2 + y^2))) / (x * y) :=
by
  sorry

end minimum_value_expression_l270_270086


namespace car_A_catches_up_l270_270355

variables (v_A v_B t_A : ℝ)

-- The conditions of the problem
def distance : ℝ := 300
def time_car_B : ℝ := t_A + 2
def distance_eq_A : Prop := distance = v_A * t_A
def distance_eq_B : Prop := distance = v_B * time_car_B

-- The final proof problem: Car A catches up with car B 150 kilometers away from city B.
theorem car_A_catches_up (t_A > 0) (v_A > 0) (v_B > 0) :
  distance_eq_A ∧ distance_eq_B → 
  ∃ d : ℝ, d = 150 := 
sorry

end car_A_catches_up_l270_270355


namespace reciprocal_of_neg_2023_l270_270457

theorem reciprocal_of_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l270_270457


namespace actual_distance_traveled_l270_270248

theorem actual_distance_traveled
  (D : ℝ) 
  (H : ∃ T : ℝ, D = 5 * T ∧ D + 20 = 15 * T) : 
  D = 10 :=
by
  sorry

end actual_distance_traveled_l270_270248


namespace volume_of_rectangular_parallelepiped_l270_270925

theorem volume_of_rectangular_parallelepiped (x y z p q r : ℝ) 
  (h1 : p = x * y) 
  (h2 : q = x * z) 
  (h3 : r = y * z) : 
  x * y * z = Real.sqrt (p * q * r) :=
by
  sorry

end volume_of_rectangular_parallelepiped_l270_270925


namespace count_valid_numbers_l270_270682

def is_valid_number (n : ℕ) : Prop :=
  n < 100000 ∧
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ d₁ ≠ d₂ ∧
    (∀ k, (n / 10^k % 10 = d₁ ∨ n / 10^k % 10 = d₂))

theorem count_valid_numbers : 
  ∃ (k : ℕ), k = 2151 ∧ (∀ n, is_valid_number n → n < 100000 → n ≤ k) :=
by
  sorry

end count_valid_numbers_l270_270682


namespace repeating_decimal_exceeds_finite_decimal_by_l270_270808

-- Definitions based on the problem conditions
def repeating_decimal := 8 / 11
def finite_decimal := 18 / 25

-- Statement to be proved
theorem repeating_decimal_exceeds_finite_decimal_by : 
  repeating_decimal - finite_decimal = 2 / 275 :=
by
  -- Skipping the proof itself with 'sorry'
  sorry

end repeating_decimal_exceeds_finite_decimal_by_l270_270808


namespace skipping_rope_equation_correct_l270_270170

-- Definitions of constraints
variable (x : ℕ) -- Number of skips per minute by Xiao Ji
variable (H1 : 0 < x) -- The number of skips per minute by Xiao Ji is positive
variable (H2 : 100 / x * x = 100) -- Xiao Ji skips exactly 100 times

-- Xiao Fan's conditions
variable (H3 : 100 + 20 = 120) -- Xiao Fan skips 20 more times than Xiao Ji
variable (H4 : x + 30 > 0) -- Xiao Fan skips 30 more times per minute than Xiao Ji

-- Prove the equation is correct
theorem skipping_rope_equation_correct :
  100 / x = 120 / (x + 30) :=
by
  sorry

end skipping_rope_equation_correct_l270_270170


namespace excess_common_fraction_l270_270803

theorem excess_common_fraction :
  let x := (72 / 10^2 + 7.2 * (10 / 10^2) * (10^{-2} / (1 - 10^{-2}))) in
  x - (72 / 100) = 8 / 1100 :=
by
  sorry

end excess_common_fraction_l270_270803


namespace find_x_l270_270220

theorem find_x 
  (x : ℝ) 
  (θ : ℝ) 
  (P : ℝ × ℝ) 
  (hP : P = (x, 6)) 
  (hcos : Real.cos θ = -4/5) 
  : x = -8 := 
sorry

end find_x_l270_270220


namespace arithmetic_geometric_inequality_l270_270775

theorem arithmetic_geometric_inequality (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧ ((a + b) / 2 = Real.sqrt (a * b) ↔ a = b) :=
by
  sorry

end arithmetic_geometric_inequality_l270_270775


namespace sum_second_largest_and_smallest_l270_270597

theorem sum_second_largest_and_smallest :
  let numbers := [10, 11, 12, 13, 14]
  ∃ second_largest second_smallest, (List.nthLe numbers 3 sorry = second_largest ∧ List.nthLe numbers 1 sorry = second_smallest ∧ second_largest + second_smallest = 24) :=
sorry

end sum_second_largest_and_smallest_l270_270597


namespace repeatingDecimal_exceeds_l270_270833

noncomputable def repeatingDecimalToFraction (d : ℚ) : ℚ := 
    -- Function to convert repeating decimal to fraction
    if d = 0.99999 then 1 else (d * 100 - d) / 99  -- Simplified conversion for demonstration

def decimalToFraction (d : ℚ) : ℚ :=
    -- Function to convert decimal to fraction
    if d = 0.72 then 18 / 25 else 0  -- Replace with actual conversion

theorem repeatingDecimal_exceeds (x y : ℚ) (hx : repeatingDecimalToFraction x = 8/11) (hy : decimalToFraction y = 18/25):
    x - y = 2 / 275 :=
by
    sorry

end repeatingDecimal_exceeds_l270_270833


namespace cycling_time_difference_l270_270715

-- Definitions from the conditions
def youth_miles : ℤ := 20
def youth_hours : ℤ := 2
def adult_miles : ℤ := 12
def adult_hours : ℤ := 3

-- Conversion from hours to minutes
def hours_to_minutes (hours : ℤ) : ℤ := hours * 60

-- Time per mile calculations
def youth_time_per_mile : ℤ := hours_to_minutes youth_hours / youth_miles
def adult_time_per_mile : ℤ := hours_to_minutes adult_hours / adult_miles

-- The difference in time per mile
def time_difference : ℤ := adult_time_per_mile - youth_time_per_mile

-- Theorem to prove the difference is 9 minutes
theorem cycling_time_difference : time_difference = 9 := by
  -- Proof steps would go here
  sorry

end cycling_time_difference_l270_270715


namespace max_sin_a_l270_270433

theorem max_sin_a (a b : ℝ) (h : Real.sin (a + b) = Real.sin a + Real.sin b) : Real.sin a ≤ 1 :=
by
  sorry

end max_sin_a_l270_270433


namespace sales_difference_greatest_in_june_l270_270590

def percentage_difference (D B : ℕ) : ℚ :=
  if B = 0 then 0 else (↑(max D B - min D B) / ↑(min D B)) * 100

def january : ℕ × ℕ := (8, 5)
def february : ℕ × ℕ := (10, 5)
def march : ℕ × ℕ := (8, 8)
def april : ℕ × ℕ := (4, 8)
def may : ℕ × ℕ := (5, 10)
def june : ℕ × ℕ := (3, 9)

noncomputable
def greatest_percentage_difference_month : String :=
  let jan_diff := percentage_difference january.1 january.2
  let feb_diff := percentage_difference february.1 february.2
  let mar_diff := percentage_difference march.1 march.2
  let apr_diff := percentage_difference april.1 april.2
  let may_diff := percentage_difference may.1 may.2
  let jun_diff := percentage_difference june.1 june.2
  if max jan_diff (max feb_diff (max mar_diff (max apr_diff (max may_diff jun_diff)))) == jun_diff
  then "June" else "Not June"
  
theorem sales_difference_greatest_in_june : greatest_percentage_difference_month = "June" :=
  by sorry

end sales_difference_greatest_in_june_l270_270590


namespace second_sheet_width_l270_270310

theorem second_sheet_width :
  ∃ w : ℝ, (286 = 22 * w + 100) ∧ w = 8.5 :=
by
  -- Proof goes here
  sorry

end second_sheet_width_l270_270310


namespace ratio_PeteHand_to_TracyCartwheel_l270_270295

noncomputable def SusanWalkingSpeed (PeteBackwardSpeed : ℕ) : ℕ :=
  PeteBackwardSpeed / 3

noncomputable def TracyCartwheelSpeed (SusanSpeed : ℕ) : ℕ :=
  SusanSpeed * 2

def PeteHandsWalkingSpeed : ℕ := 2

def PeteBackwardWalkingSpeed : ℕ := 12

theorem ratio_PeteHand_to_TracyCartwheel :
  let SusanSpeed := SusanWalkingSpeed PeteBackwardWalkingSpeed
  let TracySpeed := TracyCartwheelSpeed SusanSpeed
  (PeteHandsWalkingSpeed : ℕ) / (TracySpeed : ℕ) = 1 / 4 :=
by
  sorry

end ratio_PeteHand_to_TracyCartwheel_l270_270295


namespace faster_train_speed_l270_270762

theorem faster_train_speed (dist_between_stations : ℕ) (extra_distance : ℕ) (slower_speed : ℕ) 
  (dist_between_stations_eq : dist_between_stations = 444)
  (extra_distance_eq : extra_distance = 60) 
  (slower_speed_eq : slower_speed = 16) :
  ∃ (faster_speed : ℕ), faster_speed = 21 := by
  sorry

end faster_train_speed_l270_270762


namespace green_and_yellow_peaches_total_is_correct_l270_270176

-- Define the number of red, yellow, and green peaches
def red_peaches : ℕ := 5
def yellow_peaches : ℕ := 14
def green_peaches : ℕ := 6

-- Definition of the total number of green and yellow peaches
def total_green_and_yellow_peaches : ℕ := green_peaches + yellow_peaches

-- Theorem stating that the total number of green and yellow peaches is 20
theorem green_and_yellow_peaches_total_is_correct : total_green_and_yellow_peaches = 20 :=
by 
  sorry

end green_and_yellow_peaches_total_is_correct_l270_270176


namespace total_distance_walked_l270_270104

def distance_to_fountain : ℕ := 30
def number_of_trips : ℕ := 4
def round_trip_distance : ℕ := 2 * distance_to_fountain

theorem total_distance_walked : (number_of_trips * round_trip_distance) = 240 := by
  sorry

end total_distance_walked_l270_270104


namespace batsman_average_after_12th_innings_l270_270770

theorem batsman_average_after_12th_innings 
  (A : ℕ) 
  (total_runs_11_innings : ℕ := 11 * A) 
  (new_average : ℕ := A + 2) 
  (total_runs_12_innings : ℕ := total_runs_11_innings + 92) 
  (increased_average_after_12 : 12 * new_average = total_runs_12_innings) 
  : new_average = 70 := 
by
  -- skipping proof
  sorry

end batsman_average_after_12th_innings_l270_270770


namespace seven_expression_one_seven_expression_two_l270_270136

theorem seven_expression_one : 777 / 7 - 77 / 7 = 100 :=
by sorry

theorem seven_expression_two : 7 * 7 + 7 * 7 + 7 / 7 + 7 / 7 = 100 :=
by sorry

end seven_expression_one_seven_expression_two_l270_270136


namespace perimeter_of_sector_l270_270880

theorem perimeter_of_sector (r : ℝ) (area : ℝ) (perimeter : ℝ) 
  (hr : r = 1) (ha : area = π / 3) : perimeter = (2 * π / 3) + 2 :=
by
  -- You can start the proof here
  sorry

end perimeter_of_sector_l270_270880


namespace max_sum_of_squares_l270_270842

open Real

theorem max_sum_of_squares 
  (a : ℝ) (α : ℝ) 
  (hα : 0 < α ∧ α < π / 2) -- α is an acute angle
  -- Assuming a valid acute-give triangle with sides a, b, c with given angle α
  : ∃ b c, b ^ 2 + c ^ 2 ≤ a ^ 2 / (2 * sin (α / 2) ^ 2) ∧
    (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * cos α) := 
sorry

end max_sum_of_squares_l270_270842


namespace consecutive_even_product_l270_270969

theorem consecutive_even_product (x : ℤ) (h : x * (x + 2) = 224) : x * (x + 2) = 224 := by
  sorry

end consecutive_even_product_l270_270969


namespace common_difference_l270_270461

theorem common_difference (a1 d : ℕ) (S3 : ℕ) (h1 : S3 = 6) (h2 : a1 = 1)
  (h3 : S3 = 3 * (2 * a1 + 2 * d) / 2) : d = 1 :=
by
  sorry

end common_difference_l270_270461


namespace part1_l270_270909

variable (a b c : ℝ) (A B : ℝ)
variable (triangle_abc : Triangle ABC)
variable (cos : ℝ → ℝ)

axiom law_of_cosines : ∀ {a b c A : ℝ}, a^2 = b^2 + c^2 - 2 * b * c * cos A

theorem part1 (h1 : b^2 + 3 * a * c * (a^2 + c^2 - b^2) / (2 * a * c) = 2 * c^2) (h2 : a = c) : A = π / 4 := 
sorry

end part1_l270_270909


namespace triangle_third_side_length_l270_270713

theorem triangle_third_side_length {x : ℝ}
    (h1 : 3 > 0)
    (h2 : 7 > 0)
    (h3 : 3 + 7 > x)
    (h4 : x + 3 > 7)
    (h5 : x + 7 > 3) :
    4 < x ∧ x < 10 := by
  sorry

end triangle_third_side_length_l270_270713


namespace math_problem_l270_270658

theorem math_problem 
  (x y : ℝ) 
  (h1 : x + y = -5) 
  (h2 : x * y = 3) :
  x * Real.sqrt (y / x) + y * Real.sqrt (x / y) = -2 * Real.sqrt 3 := 
sorry

end math_problem_l270_270658


namespace fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l270_270142

theorem fewer_than_ten_sevens_example1 : (777 / 7) - (77 / 7) = 100 :=
  by sorry

theorem fewer_than_ten_sevens_example2 : (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7) = 100 :=
  by sorry

end fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l270_270142


namespace arithmetic_mean_calc_l270_270354

theorem arithmetic_mean_calc (x a : ℝ) (hx : x ≠ 0) (ha : a ≠ 0) :
  ( ( (x + a)^2 / x ) + ( (x - a)^2 / x ) ) / 2 = x + (a^2 / x) :=
sorry

end arithmetic_mean_calc_l270_270354


namespace repeating_decimal_exceeds_finite_decimal_by_l270_270811

-- Definitions based on the problem conditions
def repeating_decimal := 8 / 11
def finite_decimal := 18 / 25

-- Statement to be proved
theorem repeating_decimal_exceeds_finite_decimal_by : 
  repeating_decimal - finite_decimal = 2 / 275 :=
by
  -- Skipping the proof itself with 'sorry'
  sorry

end repeating_decimal_exceeds_finite_decimal_by_l270_270811


namespace slope_angle_tangent_line_l270_270314

open Real

-- Define the function 
def f (x : ℝ) := exp x * cos x

-- The proof statement
theorem slope_angle_tangent_line : 
  let α := arctan 1 in
  0 ≤ α ∧ α < π ∧ tan α = 1 ∧ α = π/4 :=
by
  let α := arctan 1
  have hα : α = π / 4, from arctan_eq_pi_div_4,
  have h_tan : tan α = 1, from tan_arctan one_ne_zero,
  have h_bounds : 0 ≤ α ∧ α < π, from ⟨arctan_nonneg 1 (by norm_num), arctan_lt_pi 1⟩,
  exact ⟨h_bounds.left, h_bounds.right, h_tan, hα⟩

end slope_angle_tangent_line_l270_270314


namespace intersection_A_B_l270_270389

-- Definitions of the sets A and B according to the problem conditions
def A : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}
def B : Set ℝ := {y | ∃ x : ℝ, y = Real.sqrt (-x^2 + 4 * x - 3)}

-- The proof problem statement
theorem intersection_A_B :
  A ∩ B = {y | 0 ≤ y ∧ y ≤ 1} :=
by
  sorry

end intersection_A_B_l270_270389


namespace gcd_seq_coprime_l270_270109

def seq (n : ℕ) : ℕ := 2^(2^n) + 1

theorem gcd_seq_coprime (n k : ℕ) (hnk : n ≠ k) : Nat.gcd (seq n) (seq k) = 1 :=
by
  sorry

end gcd_seq_coprime_l270_270109


namespace complex_number_simplification_l270_270838

theorem complex_number_simplification (i : ℂ) (hi : i^2 = -1) : i - (1 / i) = 2 * i :=
by
  sorry

end complex_number_simplification_l270_270838


namespace sequence_6th_term_sequence_1994th_term_l270_270646

def sequence_term (n : Nat) : Nat := n * (n + 1)

theorem sequence_6th_term:
  sequence_term 6 = 42 :=
by
  -- proof initially skipped
  sorry

theorem sequence_1994th_term:
  sequence_term 1994 = 3978030 :=
by
  -- proof initially skipped
  sorry

end sequence_6th_term_sequence_1994th_term_l270_270646


namespace sum_first_seven_arithmetic_l270_270551

theorem sum_first_seven_arithmetic (a : ℕ) (d : ℕ) (h : a + 3 * d = 3) :
    let a1 := a
    let a2 := a + d
    let a3 := a + 2 * d
    let a4 := a + 3 * d
    let a5 := a + 4 * d
    let a6 := a + 5 * d
    let a7 := a + 6 * d
    a1 + a2 + a3 + a4 + a5 + a6 + a7 = 21 :=
by
  sorry

end sum_first_seven_arithmetic_l270_270551


namespace log_fraction_property_l270_270614

noncomputable def log_base (a N : ℝ) : ℝ := Real.log N / Real.log a

theorem log_fraction_property :
  (log_base 3 4 / log_base 9 8) = 4 / 3 :=
by
  sorry

end log_fraction_property_l270_270614


namespace car_catch_up_distance_l270_270357

-- Define the problem: Prove that Car A catches Car B at the specified location, given the conditions

theorem car_catch_up_distance
  (distance : ℝ := 300)
  (v_A v_B t_A : ℝ)
  (start_A_late : ℝ := 1) -- Car A leaves 1 hour later
  (arrive_A_early : ℝ := 1) -- Car A arrives 1 hour earlier
  (t : ℝ := 2) -- Time when Car A catches up with Car B
  (dist_eq_A : distance = v_A * t_A)
  (dist_eq_B : distance = v_B * (t_A + 2))
  (catch_up_time_eq : v_A * t = v_B * (t + 1)):
  (distance - v_A * t) = 150 := sorry

end car_catch_up_distance_l270_270357


namespace max_sin_a_l270_270432

theorem max_sin_a (a b : ℝ) (h : sin (a + b) = sin a + sin b) : 
  sin a ≤ 1 :=
by
  sorry

end max_sin_a_l270_270432


namespace bobby_total_l270_270634

-- Define the conditions
def initial_candy : ℕ := 33
def additional_candy : ℕ := 4
def chocolate : ℕ := 14

-- Define the total pieces of candy Bobby ate
def total_candy : ℕ := initial_candy + additional_candy

-- Define the total pieces of candy and chocolate Bobby ate
def total_candy_and_chocolate : ℕ := total_candy + chocolate

-- Theorem to prove the total pieces of candy and chocolate Bobby ate
theorem bobby_total : total_candy_and_chocolate = 51 :=
by sorry

end bobby_total_l270_270634


namespace weight_of_new_student_l270_270986

-- Definitions from conditions
def total_weight_19 : ℝ := 19 * 15
def total_weight_20 : ℝ := 20 * 14.9

-- Theorem to prove the weight of the new student
theorem weight_of_new_student : (total_weight_20 - total_weight_19) = 13 := by
  sorry

end weight_of_new_student_l270_270986


namespace students_in_favor_ABC_l270_270194

variables (U A B C : Finset ℕ)

-- Given conditions
axiom total_students : U.card = 300
axiom students_in_favor_A : A.card = 210
axiom students_in_favor_B : B.card = 190
axiom students_in_favor_C : C.card = 160
axiom students_against_all : (U \ (A ∪ B ∪ C)).card = 40

-- Proof goal
theorem students_in_favor_ABC : (A ∩ B ∩ C).card = 80 :=
by {
  sorry
}

end students_in_favor_ABC_l270_270194


namespace sum_of_squares_ge_two_ab_l270_270923

theorem sum_of_squares_ge_two_ab (a b : ℝ) : a^2 + b^2 ≥ 2 * a * b := 
  sorry

end sum_of_squares_ge_two_ab_l270_270923


namespace repeating_seventy_two_exceeds_seventy_two_l270_270825

noncomputable def repeating_decimal (n d : ℕ) : ℚ := n / d

theorem repeating_seventy_two_exceeds_seventy_two :
  repeating_decimal 72 99 - (72 / 100) = (2 / 275) := 
sorry

end repeating_seventy_two_exceeds_seventy_two_l270_270825


namespace longer_piece_length_l270_270337

-- Conditions
def total_length : ℤ := 69
def is_cuts_into_two_pieces (a b : ℤ) : Prop := a + b = total_length
def is_twice_the_length (a b : ℤ) : Prop := a = 2 * b

-- Question: What is the length of the longer piece?
theorem longer_piece_length
  (a b : ℤ) 
  (H1: is_cuts_into_two_pieces a b)
  (H2: is_twice_the_length a b) :
  a = 46 :=
sorry

end longer_piece_length_l270_270337


namespace smallest_number_with_55_divisors_l270_270030

theorem smallest_number_with_55_divisors : ∃ (n : ℕ), (∃ (p : ℕ → ℕ) (k : ℕ → ℕ) (m : ℕ), 
  n = ∏ i in finset.range m, (p i)^(k i) ∧ (∀ i j, i ≠ j → nat.prime (p i) → nat.prime (p j) → p i ≠ p j) ∧ 
  (finset.range m).card = m ∧ 
  (∏ i in finset.range m, (k i + 1) = 55)) ∧ 
  n = 3^4 * 2^10 then n = 82944 :=
by
  sorry

end smallest_number_with_55_divisors_l270_270030


namespace red_gumballs_count_l270_270005

def gumballs_problem (R B G : ℕ) : Prop :=
  B = R / 2 ∧
  G = 4 * B ∧
  R + B + G = 56

theorem red_gumballs_count (R B G : ℕ) (h : gumballs_problem R B G) : R = 16 :=
by
  rcases h with ⟨h1, h2, h3⟩
  sorry

end red_gumballs_count_l270_270005


namespace jongkook_points_l270_270943

-- Define the conditions in the problem
def num_questions_solved_each : ℕ := 18
def shinhye_points : ℕ := 100
def jongkook_correct_6_points : ℕ := 8
def jongkook_correct_5_points : ℕ := 6
def points_per_question_6 : ℕ := 6
def points_per_question_5 : ℕ := 5
def jongkook_wrong_questions : ℕ := num_questions_solved_each - jongkook_correct_6_points - jongkook_correct_5_points

-- Calculate Jongkook's points from correct answers
def jongkook_points_from_6 : ℕ := jongkook_correct_6_points * points_per_question_6
def jongkook_points_from_5 : ℕ := jongkook_correct_5_points * points_per_question_5

-- Calculate total points
def jongkook_total_points : ℕ := jongkook_points_from_6 + jongkook_points_from_5

-- Prove that Jongkook's total points is 78
theorem jongkook_points : jongkook_total_points = 78 :=
by
  sorry

end jongkook_points_l270_270943


namespace hair_cut_first_day_l270_270515

theorem hair_cut_first_day 
  (total_hair_cut : ℝ) 
  (hair_cut_second_day : ℝ) 
  (h_total : total_hair_cut = 0.875) 
  (h_second : hair_cut_second_day = 0.5) : 
  total_hair_cut - hair_cut_second_day = 0.375 := 
  by
  simp [h_total, h_second]
  sorry

end hair_cut_first_day_l270_270515


namespace ancient_chinese_poem_l270_270078

theorem ancient_chinese_poem (x : ℕ) :
  (7 * x + 7 = 9 * (x - 1)) :=
sorry

end ancient_chinese_poem_l270_270078


namespace sqrt_defined_iff_nonneg_l270_270258

theorem sqrt_defined_iff_nonneg (x : ℝ) : (∃ y, y = sqrt (x - 2)) ↔ x ≥ 2 :=
by
  sorry

end sqrt_defined_iff_nonneg_l270_270258


namespace final_speed_train_l270_270776

theorem final_speed_train
  (u : ℝ) (a : ℝ) (t : ℕ) :
  u = 0 → a = 1 → t = 20 → u + a * t = 20 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end final_speed_train_l270_270776


namespace joy_sixth_time_is_87_seconds_l270_270421

def sixth_time (times : List ℝ) (new_median : ℝ) : ℝ :=
  let sorted_times := times |>.insertNth 2 (2 * new_median - times.nthLe 2 sorry)
  2 * new_median - times.nthLe 2 sorry

theorem joy_sixth_time_is_87_seconds (times : List ℝ) (new_median : ℝ) :
  times = [82, 85, 93, 95, 99] → new_median = 90 →
  sixth_time times new_median = 87 :=
by
  intros h_times h_median
  rw [h_times]
  rw [h_median]
  sorry

end joy_sixth_time_is_87_seconds_l270_270421


namespace tangent_circle_locus_l270_270309

-- Definitions for circle C1 and circle C2
def Circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def Circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Definition of being tangent to a circle
def ExternallyTangent (cx cy cr : ℝ) : Prop := (cx - 0)^2 + (cy - 0)^2 = (cr + 1)^2
def InternallyTangent (cx cy cr : ℝ) : Prop := (cx - 3)^2 + (cy - 0)^2 = (3 - cr)^2

-- Definition of locus L where (a,b) are centers of circles tangent to both C1 and C2
def Locus (a b : ℝ) : Prop := 28 * a^2 + 64 * b^2 - 84 * a - 49 = 0

-- The theorem to be proved
theorem tangent_circle_locus (a b r : ℝ) :
  (ExternallyTangent a b r) → (InternallyTangent a b r) → Locus a b :=
by {
  sorry
}

end tangent_circle_locus_l270_270309


namespace complete_square_solution_l270_270135

theorem complete_square_solution (x : ℝ) (h : x^2 - 4 * x + 2 = 0) : (x - 2)^2 = 2 := 
by sorry

end complete_square_solution_l270_270135


namespace prove_functions_same_l270_270513

theorem prove_functions_same (u v : ℝ) (huv : u = v) : 
  (u > 1) → (v > 1) → (Real.sqrt ((u + 1) / (u - 1)) = Real.sqrt ((v + 1) / (v - 1))) :=
by
  sorry

end prove_functions_same_l270_270513


namespace seven_expression_one_seven_expression_two_l270_270138

theorem seven_expression_one : 777 / 7 - 77 / 7 = 100 :=
by sorry

theorem seven_expression_two : 7 * 7 + 7 * 7 + 7 / 7 + 7 / 7 = 100 :=
by sorry

end seven_expression_one_seven_expression_two_l270_270138


namespace desired_overall_percentage_l270_270787

-- Define the scores in the three subjects
def score1 := 50
def score2 := 70
def score3 := 90

-- Define the expected overall percentage
def expected_overall_percentage := 70

-- The main theorem to prove
theorem desired_overall_percentage :
  (score1 + score2 + score3) / 3 = expected_overall_percentage :=
by
  sorry

end desired_overall_percentage_l270_270787


namespace count_at_most_two_different_digits_l270_270698

def at_most_two_different_digits (n : ℕ) : Prop :=
  n < 100000 ∧ (∀ m < n.digits.length, ∀ p < n.digits.length, m ≠ p → n.digits.nth_le m ≠ n.digits.nth_le p) → 
  (∃ d1 d2 : ℕ, n.digits.all (λ d, d = d1 ∨ d = d2))

theorem count_at_most_two_different_digits :
  (nat.filter at_most_two_different_digits (list.range 100000)).length = 2034 :=
begin
  sorry
end

end count_at_most_two_different_digits_l270_270698


namespace percent_of_x_is_y_l270_270704

theorem percent_of_x_is_y (x y : ℝ) (h : 0.25 * (x - y) = 0.15 * (x + y)) : y = 0.25 * x := by
  sorry

end percent_of_x_is_y_l270_270704


namespace g_difference_l270_270537

-- Define the function g(n)
def g (n : ℤ) : ℚ := (1/2 : ℚ) * n^2 * (n + 3)

-- State the theorem
theorem g_difference (s : ℤ) : g s - g (s - 1) = (1/2 : ℚ) * (3 * s - 2) := by
  sorry

end g_difference_l270_270537


namespace g_3_2_plus_g_3_5_l270_270556

def g (x y : ℚ) : ℚ :=
  if x + y ≤ 5 then (x * y - x + 3) / (3 * x) else (x * y - y - 3) / (-3 * y)

theorem g_3_2_plus_g_3_5 : g 3 2 + g 3 5 = 1/5 := by
  sorry

end g_3_2_plus_g_3_5_l270_270556


namespace twelve_xy_leq_fourx_1_y_9y_1_x_l270_270282

theorem twelve_xy_leq_fourx_1_y_9y_1_x
  (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) :=
  sorry

end twelve_xy_leq_fourx_1_y_9y_1_x_l270_270282


namespace simplify_expression_l270_270741

theorem simplify_expression (x : ℝ) : 3 * x + 4 * (2 - x) - 2 * (3 - 2 * x) + 5 * (2 + 3 * x) = 18 * x + 12 :=
by
  sorry

end simplify_expression_l270_270741


namespace arithmetic_mean_of_14_22_36_l270_270976

theorem arithmetic_mean_of_14_22_36 : (14 + 22 + 36) / 3 = 24 := by
  sorry

end arithmetic_mean_of_14_22_36_l270_270976


namespace monotonicity_intervals_range_of_m_l270_270882

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + Real.log x) / x

theorem monotonicity_intervals (m : ℝ) (x : ℝ) (hx : x > 1):
  (m >= 1 → ∀ x' > 1, f m x' ≤ f m x) ∧
  (m < 1 → (∀ x' ∈ Set.Ioo 1 (Real.exp (1 - m)), f m x' > f m x) ∧
            (∀ x' ∈ Set.Ioi (Real.exp (1 - m)), f m x' < f m x)) := by
  sorry

theorem range_of_m (m : ℝ) :
  (∀ x > 1, f m x < m * x) ↔ m ≥ 1/2 := by
  sorry

end monotonicity_intervals_range_of_m_l270_270882


namespace evaluate_expression_l270_270197

noncomputable def expression := 
  (Real.sqrt 3 * Real.tan (Real.pi / 15) - 3) / 
  (4 * (Real.cos (Real.pi / 15))^2 * Real.sin (Real.pi / 15) - 2 * Real.sin (Real.pi / 15))

theorem evaluate_expression : expression = -4 * Real.sqrt 3 :=
  sorry

end evaluate_expression_l270_270197


namespace total_volume_of_cubes_l270_270914

theorem total_volume_of_cubes (Jim_cubes : Nat) (Jim_side_length : Nat) 
    (Laura_cubes : Nat) (Laura_side_length : Nat)
    (h1 : Jim_cubes = 7) (h2 : Jim_side_length = 3) 
    (h3 : Laura_cubes = 4) (h4 : Laura_side_length = 4) : 
    (Jim_cubes * Jim_side_length^3 + Laura_cubes * Laura_side_length^3 = 445) :=
by
  sorry

end total_volume_of_cubes_l270_270914


namespace area_below_line_l270_270975

-- Define the conditions provided in the problem.
def graph_eq (x y : ℝ) : Prop := x^2 - 14*x + 3*y + 70 = 21 + 11*y - y^2
def line_eq (x y : ℝ) : Prop := y = x - 3

-- State the final proof problem which is to find the area under the given conditions.
theorem area_below_line :
  ∃ area : ℝ, area = 8 * Real.pi ∧ 
  (∀ x y, graph_eq x y → y ≤ x - 3 → -area / 2 ≤ y ∧ y ≤ area / 2) := 
sorry

end area_below_line_l270_270975


namespace value_of_x_minus_y_squared_l270_270410

theorem value_of_x_minus_y_squared (x y : ℝ) (hx : x^2 = 4) (hy : y^2 = 9) : 
  ((x - y)^2 = 1) ∨ ((x - y)^2 = 25) :=
sorry

end value_of_x_minus_y_squared_l270_270410


namespace distance_between_5th_and_23rd_red_light_l270_270304

theorem distance_between_5th_and_23rd_red_light :
  let inch_to_feet (inches : ℕ) : ℝ := inches / 12.0
  let distance_in_inches := 40 * 8
  inch_to_feet distance_in_inches = 26.67 :=
by
  sorry

end distance_between_5th_and_23rd_red_light_l270_270304


namespace number_of_terms_arithmetic_sequence_l270_270072

theorem number_of_terms_arithmetic_sequence
  (a₁ d n : ℝ)
  (h1 : a₁ + (a₁ + d) + (a₁ + 2 * d) = 34)
  (h2 : (a₁ + (n-3) * d) + (a₁ + (n-2) * d) + (a₁ + (n-1) * d) = 146)
  (h3 : n / 2 * (2 * a₁ + (n-1) * d) = 390) :
  n = 11 :=
by sorry

end number_of_terms_arithmetic_sequence_l270_270072


namespace integer_to_the_fourth_l270_270705

theorem integer_to_the_fourth (a : ℤ) (h : a = 243) : 3^12 * 3^8 = a^4 :=
by {
  sorry
}

end integer_to_the_fourth_l270_270705


namespace contractor_fired_two_people_l270_270780

theorem contractor_fired_two_people
  (total_days : ℕ) (initial_people : ℕ) (days_worked : ℕ) (fraction_completed : ℚ)
  (remaining_days : ℕ) (people_fired : ℕ)
  (h1 : total_days = 100)
  (h2 : initial_people = 10)
  (h3 : days_worked = 20)
  (h4 : fraction_completed = 1/4)
  (h5 : remaining_days = 75)
  (h6 : remaining_days + days_worked = total_days)
  (h7 : people_fired = initial_people - 8) :
  people_fired = 2 :=
  sorry

end contractor_fired_two_people_l270_270780


namespace more_likely_second_machine_l270_270039

variable (P_B1 : ℝ := 0.8) -- Probability that a part is from the first machine
variable (P_B2 : ℝ := 0.2) -- Probability that a part is from the second machine
variable (P_A_given_B1 : ℝ := 0.01) -- Probability that a part is defective given it is from the first machine
variable (P_A_given_B2 : ℝ := 0.05) -- Probability that a part is defective given it is from the second machine

noncomputable def P_A : ℝ :=
  P_B1 * P_A_given_B1 + P_B2 * P_A_given_B2

noncomputable def P_B1_given_A : ℝ :=
  (P_B1 * P_A_given_B1) / P_A

noncomputable def P_B2_given_A : ℝ :=
  (P_B2 * P_A_given_B2) / P_A

theorem more_likely_second_machine :
  P_B2_given_A > P_B1_given_A :=
by
  sorry

end more_likely_second_machine_l270_270039


namespace ratio_pr_l270_270712

variable (p q r s : ℚ)

def ratio_pq (p q : ℚ) : Prop := p / q = 5 / 4
def ratio_rs (r s : ℚ) : Prop := r / s = 4 / 3
def ratio_sq (s q : ℚ) : Prop := s / q = 1 / 5

theorem ratio_pr (hpq : ratio_pq p q) (hrs : ratio_rs r s) (hsq : ratio_sq s q) : p / r = 75 / 16 := by
  sorry

end ratio_pr_l270_270712


namespace pictures_left_after_deletion_l270_270993

variable (zoo museum deleted : ℕ)

def total_pictures_taken (zoo museum : ℕ) : ℕ := zoo + museum

def pictures_remaining (total deleted : ℕ) : ℕ := total - deleted

theorem pictures_left_after_deletion (h1 : zoo = 50) (h2 : museum = 8) (h3 : deleted = 38) :
  pictures_remaining (total_pictures_taken zoo museum) deleted = 20 :=
by
  sorry

end pictures_left_after_deletion_l270_270993


namespace x_varies_as_z_raised_to_n_power_l270_270539

noncomputable def x_varies_as_cube_of_y (k y : ℝ) : ℝ := k * y ^ 3
noncomputable def y_varies_as_cube_root_of_z (j z : ℝ) : ℝ := j * z ^ (1/3 : ℝ)

theorem x_varies_as_z_raised_to_n_power (k j z : ℝ) :
  ∃ n : ℝ, x_varies_as_cube_of_y k (y_varies_as_cube_root_of_z j z) = (k * j^3) * z ^ n ∧ n = 1 :=
by
  sorry

end x_varies_as_z_raised_to_n_power_l270_270539


namespace correct_options_l270_270052

theorem correct_options (a b c : ℝ) (h1 : ∀ x : ℝ, (a*x^2 + b*x + c > 0) ↔ (-3 < x ∧ x < 2)) :
  (a < 0) ∧ (a + b + c > 0) ∧ (∀ x, (b*x + c > 0) ↔ x > 6) = False ∧ (∀ x, (c*x^2 + b*x + a < 0) ↔ (-1/3 < x ∧ x < 1/2)) :=
by 
  sorry

end correct_options_l270_270052


namespace alpha_plus_beta_l270_270465

theorem alpha_plus_beta (α β : ℝ) (h : ∀ x, (x - α) / (x + β) = (x^2 - 116 * x + 2783) / (x^2 + 99 * x - 4080)) 
: α + β = 115 := 
sorry

end alpha_plus_beta_l270_270465


namespace find_t_when_perpendicular_l270_270060

variable {t : ℝ}

def vector_m (t : ℝ) : ℝ × ℝ := (t + 1, 1)
def vector_n (t : ℝ) : ℝ × ℝ := (t + 2, 2)
def add_vectors (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def sub_vectors (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)

theorem find_t_when_perpendicular : 
  (dot_product (add_vectors (vector_m t) (vector_n t)) (sub_vectors (vector_m t) (vector_n t)) = 0) ↔ t = -3 := by
  sorry

end find_t_when_perpendicular_l270_270060


namespace range_of_a_l270_270534

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 - a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
by
  sorry

end range_of_a_l270_270534


namespace arrangements_mississippi_l270_270364

open Nat

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem arrangements_mississippi : 
  let total := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  factorial total / (factorial m_count * factorial i_count * factorial s_count * factorial p_count) = 34650 :=
by
  -- This is the proof placeholder
  sorry

end arrangements_mississippi_l270_270364


namespace perpendicular_slope_l270_270649

theorem perpendicular_slope (x y : ℝ) : (∃ b : ℝ, 4 * x - 5 * y = 10) → ∃ m : ℝ, m = -5 / 4 :=
by
  intro h
  sorry

end perpendicular_slope_l270_270649


namespace num_digits_less_than_100000_with_two_or_fewer_different_digits_l270_270688

theorem num_digits_less_than_100000_with_two_or_fewer_different_digits:
  let count_single_digit : Nat := 9 * 5
  let count_two_distinct_without_zero : Nat := let comb_9_2 : Nat := 9.choose 2
                                               comb_9_2 * (2 + 6 + 14 + 30)
  let count_two_distinct_with_zero : Nat := let num_ways_with_zero : Nat := 9 * (1 + 3 + 7 + 15)
                                            num_ways_with_zero
  let total_count : Nat := count_single_digit + count_two_distinct_without_zero + count_two_distinct_with_zero
  total_count = 2151 := by
  sorry

end num_digits_less_than_100000_with_two_or_fewer_different_digits_l270_270688


namespace central_angle_nonagon_l270_270308

theorem central_angle_nonagon : (360 / 9 = 40) :=
by
  sorry

end central_angle_nonagon_l270_270308


namespace polygon_sides_l270_270096

theorem polygon_sides {n k : ℕ} (h1 : k = n * (n - 3) / 2) (h2 : k = 3 * n / 2) : n = 6 :=
by
  sorry

end polygon_sides_l270_270096


namespace cube_value_proportional_l270_270781

theorem cube_value_proportional (side_length1 side_length2 : ℝ) (volume1 volume2 : ℝ) (value1 value2 : ℝ) :
  side_length1 = 4 → volume1 = side_length1 ^ 3 → value1 = 500 →
  side_length2 = 6 → volume2 = side_length2 ^ 3 → value2 = value1 * (volume2 / volume1) →
  value2 = 1688 :=
by
  sorry

end cube_value_proportional_l270_270781


namespace temperature_decrease_2C_l270_270073

variable (increase_3 : ℤ := 3)
variable (decrease_2 : ℤ := -2)

theorem temperature_decrease_2C :
  decrease_2 = -2 :=
by
  -- This is where the proof would go
  sorry

end temperature_decrease_2C_l270_270073


namespace fraction_addition_l270_270607

theorem fraction_addition : (1 + 3 + 5)/(2 + 4 + 6) + (2 + 4 + 6)/(1 + 3 + 5) = 25/12 := by
  sorry

end fraction_addition_l270_270607


namespace polygon_sides_l270_270394

theorem polygon_sides {n : ℕ} (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end polygon_sides_l270_270394


namespace fewerSevensCanProduce100_l270_270148

noncomputable def validAs100UsingSevens : (ℕ → ℕ) → Prop := 
  fun (f : ℕ → ℕ) => ∃ (x y : ℕ), (fewer_than_ten_sevens x y) ∧ f x y = 100

theorem fewerSevensCanProduce100 : 
  ∃ (f : ℕ → ℕ), validAs100UsingSevens f :=
by 
  sorry

end fewerSevensCanProduce100_l270_270148


namespace find_m_n_l270_270383

noncomputable def A : Set ℝ := {3, 5}
def B (m n : ℝ) : Set ℝ := {x | x^2 + m * x + n = 0}

theorem find_m_n (m n : ℝ) (h_union : A ∪ B m n = A) (h_inter : A ∩ B m n = {5}) :
  m = -10 ∧ n = 25 :=
by
  sorry

end find_m_n_l270_270383


namespace vasya_100_using_fewer_sevens_l270_270157

-- Definitions and conditions
def seven := 7

-- Theorem to prove
theorem vasya_100_using_fewer_sevens :
  (777 / seven - 77 / seven = 100) ∨
  (seven * seven + seven * seven + seven / seven + seven / seven = 100) :=
by
  sorry

end vasya_100_using_fewer_sevens_l270_270157


namespace probability_angie_carlos_opposite_l270_270192

noncomputable def probability_opposite_seating (n : ℕ) : ℚ :=
  if n = 5 then 1 / 2 else 0

theorem probability_angie_carlos_opposite :
  probability_opposite_seating 5 = 1 / 2 :=
sorry


end probability_angie_carlos_opposite_l270_270192


namespace part1_part2_l270_270228

section
variable (k : ℝ)

/-- Part 1: Range of k -/
def discriminant_eqn (k : ℝ) := (2 * k - 1) ^ 2 - 4 * (k ^ 2 - 1)

theorem part1 (h : discriminant_eqn k ≥ 0) : k ≤ 5 / 4 :=
by sorry

/-- Part 2: Value of k when x₁ and x₂ satisfy the given condition -/
def x1_x2_eqn (k x1 x2 : ℝ) := x1 ^ 2 + x2 ^ 2 = 16 + x1 * x2

def vieta (k : ℝ) (x1 x2 : ℝ) :=
  x1 + x2 = 1 - 2 * k ∧ x1 * x2 = k ^ 2 - 1

theorem part2 (x1 x2 : ℝ) (h1 : vieta k x1 x2) (h2 : x1_x2_eqn k x1 x2) : k = -2 :=
by sorry

end

end part1_part2_l270_270228


namespace monotonic_f_inequality_f_over_h_l270_270231

noncomputable def f (x : ℝ) : ℝ := 1 + (1 / x) + Real.log x + (Real.log x / x)

theorem monotonic_f :
  ∀ x : ℝ, x > 0 → ∃ I : Set ℝ, (I = Set.Ioo 0 x ∨ I = Set.Icc 0 x) ∧ (∀ y ∈ I, y > 0 → f y = f x) :=
by
  sorry

theorem inequality_f_over_h :
  ∀ x : ℝ, x > 1 → (f x) / (Real.exp 1 + 1) > (2 * Real.exp (x - 1)) / (x * Real.exp x + 1) :=
by
  sorry

end monotonic_f_inequality_f_over_h_l270_270231


namespace brownie_cost_l270_270631

theorem brownie_cost (total_money : ℕ) (num_pans : ℕ) (pieces_per_pan : ℕ) (cost_per_piece : ℕ) :
  total_money = 32 → num_pans = 2 → pieces_per_pan = 8 → cost_per_piece = total_money / (num_pans * pieces_per_pan) → 
  cost_per_piece = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end brownie_cost_l270_270631


namespace example_one_example_two_l270_270161

-- We will define natural numbers corresponding to seven, seventy-seven, and seven hundred seventy-seven.
def seven : ℕ := 7
def seventy_seven : ℕ := 77
def seven_hundred_seventy_seven : ℕ := 777

-- We will define both solutions in the form of equalities producing 100.
theorem example_one : (seven_hundred_seventy_seven / seven) - (seventy_seven / seven) = 100 :=
  by sorry

theorem example_two : (seven * seven) + (seven * seven) + (seven / seven) + (seven / seven) = 100 :=
  by sorry

end example_one_example_two_l270_270161


namespace max_omega_for_monotonic_sine_l270_270657

open Real

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x)

theorem max_omega_for_monotonic_sine :
  (∀ x1 x2 : ℝ, 0 ≤ x1 → x1 ≤ x2 → x2 ≤ π / 3 → f (3 / 2) x1 ≤ f (3 / 2) x2) ∧
  (∀ ω : ℝ, (0 < ω) → (∀ x1 x2 : ℝ, 0 ≤ x1 → x1 ≤ x2 → x2 ≤ π / 3 → f ω x1 ≤ f ω x2) → ω ≤ 3 / 2) :=
by sorry

end max_omega_for_monotonic_sine_l270_270657


namespace fraction_difference_is_correct_l270_270796

-- Convert repeating decimal to fraction
def repeating_to_fraction : ℚ := 0.72 + 72 / 9900

-- Convert 0.72 to fraction
def decimal_to_fraction : ℚ := 72 / 100

-- Define the difference between the fractions
def fraction_difference := repeating_to_fraction - decimal_to_fraction

-- Final statement to prove
theorem fraction_difference_is_correct : fraction_difference = 2 / 275 := by
  sorry

end fraction_difference_is_correct_l270_270796


namespace unique_digit_B_l270_270901

open Finset

theorem unique_digit_B : 
  ∀ (A B C D E F : ℕ), 
    (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (A ≠ F) ∧ 
    (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (B ≠ F) ∧ 
    (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) ∧ 
    (D ≠ E) ∧ (D ≠ F) ∧ 
    (E ≠ F) ∧
    (A ∈ range 2 6) ∧ (B ∈ range 2 6) ∧ (C ∈ range 2 6) ∧
    (D ∈ range 2 6) ∧ (E ∈ range 2 6) ∧ (F ∈ range 2 6) ∧
    (A + B + C) + (A + B + E + F) + (C + D + E) + (B + D + F) + (C + F) = 65 
    → B = 4 :=
by sorry

end unique_digit_B_l270_270901


namespace ratio_of_areas_l270_270936

-- Definitions of the perimeters for each region
def perimeter_I : ℕ := 16
def perimeter_II : ℕ := 36
def perimeter_IV : ℕ := 48

-- Define the side lengths based on the given perimeters
def side_length (P : ℕ) : ℕ := P / 4

-- Calculate the areas from the side lengths
def area (s : ℕ) : ℕ := s * s

-- Now we state the theorem
theorem ratio_of_areas : 
  (area (side_length perimeter_II)) / (area (side_length perimeter_IV)) = 9 / 16 := 
by sorry

end ratio_of_areas_l270_270936


namespace olympiad_divisors_l270_270589

theorem olympiad_divisors :
  {n : ℕ | n > 0 ∧ n ∣ (1998 + n)} = {n : ℕ | n > 0 ∧ n ∣ 1998} :=
by {
  sorry
}

end olympiad_divisors_l270_270589


namespace necessary_but_not_sufficient_cond_l270_270710

open Set

variable {α : Type*} (A B C : Set α)

/-- Mathematical equivalent proof problem statement -/
theorem necessary_but_not_sufficient_cond (h1 : A ∪ B = C) (h2 : ¬ B ⊆ A) (hA : A.Nonempty) (hB : B.Nonempty) (hC : C.Nonempty) :
  (∀ x, x ∈ A → x ∈ C) ∧ (∃ y ∈ C, y ∉ A) :=
by
  sorry

end necessary_but_not_sufficient_cond_l270_270710


namespace tenfold_largest_two_digit_number_l270_270316

def largest_two_digit_number : ℕ := 99

theorem tenfold_largest_two_digit_number :
  10 * largest_two_digit_number = 990 :=
by
  sorry

end tenfold_largest_two_digit_number_l270_270316


namespace smallest_number_with_55_divisors_l270_270032

theorem smallest_number_with_55_divisors : ∃ n : ℕ, 
  (number_of_divisors n = 55) ∧ (∀ m : ℕ, number_of_divisors m = 55 → n ≤ m) := 
sorry

end smallest_number_with_55_divisors_l270_270032


namespace find_profit_percentage_l270_270542

theorem find_profit_percentage (h : (m + 8) / (1 - 0.08) = m + 10) : m = 15 := sorry

end find_profit_percentage_l270_270542


namespace count_valid_numbers_l270_270684

def is_valid_number (n : ℕ) : Prop :=
  n < 100000 ∧
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ d₁ ≠ d₂ ∧
    (∀ k, (n / 10^k % 10 = d₁ ∨ n / 10^k % 10 = d₂))

theorem count_valid_numbers : 
  ∃ (k : ℕ), k = 2151 ∧ (∀ n, is_valid_number n → n < 100000 → n ≤ k) :=
by
  sorry

end count_valid_numbers_l270_270684


namespace radius_of_larger_circle_15_l270_270599

def radius_larger_circle (r1 r2 r3 r : ℝ) : Prop :=
  ∃ (A B C O : EuclideanSpace ℝ (Fin 2)), 
    dist A B = r1 + r2 ∧
    dist B C = r2 + r3 ∧
    dist A C = r1 + r3 ∧
    dist O A = r - r1 ∧
    dist O B = r - r2 ∧
    dist O C = r - r3 ∧
    (dist O A + r1 = r ∧
    dist O B + r2 = r ∧
    dist O C + r3 = r)

theorem radius_of_larger_circle_15 :
  radius_larger_circle 10 3 2 15 :=
by
  sorry

end radius_of_larger_circle_15_l270_270599


namespace age_difference_is_36_l270_270134

open Nat

theorem age_difference_is_36 (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
    (h_eq : (10 * a + b) + 8 = 3 * ((10 * b + a) + 8)) :
    (10 * a + b) - (10 * b + a) = 36 :=
by
  sorry

end age_difference_is_36_l270_270134


namespace xiao_wang_ways_to_make_8_cents_l270_270773

theorem xiao_wang_ways_to_make_8_cents :
  (∃ c1 c2 c5 : ℕ, c1 ≤ 8 ∧ c2 ≤ 4 ∧ c5 ≤ 1 ∧ c1 + 2 * c2 + 5 * c5 = 8) → (number_of_ways_to_make_8_cents = 7) :=
sorry

end xiao_wang_ways_to_make_8_cents_l270_270773


namespace part_a_solution_l270_270306

theorem part_a_solution (x y : ℤ) : xy + 3 * x - 5 * y = -3 ↔ 
  (x = 6 ∧ y = -21) ∨ 
  (x = -13 ∧ y = -2) ∨ 
  (x = 4 ∧ y = 15) ∨ 
  (x = 23 ∧ y = -4) ∨ 
  (x = 7 ∧ y = -12) ∨ 
  (x = -4 ∧ y = -1) ∨ 
  (x = 3 ∧ y = 6) ∨ 
  (x = 14 ∧ y = -5) ∨ 
  (x = 8 ∧ y = -9) ∨ 
  (x = -1 ∧ y = 0) ∨ 
  (x = 2 ∧ y = 3) ∨ 
  (x = 11 ∧ y = -6) := 
by sorry

end part_a_solution_l270_270306


namespace compare_values_l270_270868

variable (a b c : ℝ)

noncomputable def a_def : ℝ := 0.1 * Real.exp 0.1
noncomputable def b_def : ℝ := 1 / 9
noncomputable def c_def : ℝ := - Real.log 0.9

theorem compare_values : 
  c < a ∧ a < b := 
begin
  let a := 0.1 * Real.exp 0.1,
  let b := 1 / 9,
  let c := - Real.log 0.9,
  have h1 : c < a, sorry,
  have h2 : a < b, sorry,
  exact ⟨h1, h2⟩
end

end compare_values_l270_270868


namespace product_three_numbers_l270_270755

theorem product_three_numbers 
  (a b c : ℝ)
  (h1 : a + b + c = 30)
  (h2 : a = 3 * (b + c))
  (h3 : b = 5 * c) : 
  a * b * c = 176 := 
by
  sorry

end product_three_numbers_l270_270755


namespace count_at_most_two_different_digits_l270_270697

def at_most_two_different_digits (n : ℕ) : Prop :=
  n < 100000 ∧ (∀ m < n.digits.length, ∀ p < n.digits.length, m ≠ p → n.digits.nth_le m ≠ n.digits.nth_le p) → 
  (∃ d1 d2 : ℕ, n.digits.all (λ d, d = d1 ∨ d = d2))

theorem count_at_most_two_different_digits :
  (nat.filter at_most_two_different_digits (list.range 100000)).length = 2034 :=
begin
  sorry
end

end count_at_most_two_different_digits_l270_270697


namespace identity_problem_l270_270408

theorem identity_problem
  (a b : ℝ)
  (h₁ : a * b = 2)
  (h₂ : a + b = 3) :
  (a - b)^2 = 1 :=
by
  sorry

end identity_problem_l270_270408


namespace determine_top_5_median_required_l270_270076

theorem determine_top_5_median_required (scores : Fin 9 → ℝ) (unique_scores : ∀ (i j : Fin 9), i ≠ j → scores i ≠ scores j) :
  ∃ median,
  (∀ (student_score : ℝ), 
    (student_score > median ↔ ∃ (idx_top : Fin 5), student_score = scores ⟨idx_top.1, sorry⟩)) :=
sorry

end determine_top_5_median_required_l270_270076


namespace cyclist_speed_ratio_l270_270603

theorem cyclist_speed_ratio (v_1 v_2 : ℝ)
  (h1 : v_1 = 2 * v_2)
  (h2 : v_1 + v_2 = 6)
  (h3 : v_1 - v_2 = 2) :
  v_1 / v_2 = 2 := 
sorry

end cyclist_speed_ratio_l270_270603


namespace train_speed_conversion_l270_270186

-- Define the speed of the train in meters per second.
def speed_mps : ℝ := 37.503

-- Definition of the conversion factor between m/s and km/h.
def conversion_factor : ℝ := 3.6

-- Define the expected speed of the train in kilometers per hour.
def expected_speed_kmph : ℝ := 135.0108

-- Prove that the speed in km/h is the expected value.
theorem train_speed_conversion :
  (speed_mps * conversion_factor = expected_speed_kmph) :=
by
  sorry

end train_speed_conversion_l270_270186


namespace expr_value_at_neg2_l270_270980

variable (a b : ℝ)

def expr (x : ℝ) : ℝ := a * x^3 + b * x - 7

theorem expr_value_at_neg2 :
  (expr a b 2 = -19) → (expr a b (-2) = 5) :=
by 
  intro h
  sorry

end expr_value_at_neg2_l270_270980


namespace pattern_equation_l270_270396

theorem pattern_equation (n : ℕ) : n^2 + n = n * (n + 1) := 
  sorry

end pattern_equation_l270_270396


namespace time_to_run_home_l270_270721

-- Define the conditions
def blocks_run_per_time : ℚ := 2 -- Justin runs 2 blocks
def time_per_blocks : ℚ := 1.5 -- in 1.5 minutes
def blocks_to_home : ℚ := 8 -- Justin is 8 blocks from home

-- Define the theorem to prove the time taken for Justin to run home
theorem time_to_run_home : (blocks_to_home / blocks_run_per_time) * time_per_blocks = 6 :=
by
  sorry

end time_to_run_home_l270_270721


namespace solve_for_x_y_l270_270237

noncomputable def x_y_2018_sum (x y : ℝ) : ℝ := x^2018 + y^2018

theorem solve_for_x_y (A B : Set ℝ) (x y : ℝ)
  (hA : A = {x, x * y, x + y})
  (hB : B = {0, |x|, y}) 
  (h : A = B) :
  x_y_2018_sum x y = 2 := 
by
  sorry

end solve_for_x_y_l270_270237


namespace slices_per_person_l270_270578

theorem slices_per_person (total_slices : ℕ) (total_people : ℕ) (h_slices : total_slices = 12) (h_people : total_people = 3) :
  total_slices / total_people = 4 :=
by
  sorry

end slices_per_person_l270_270578


namespace determine_x1_l270_270508

theorem determine_x1
  (x1 x2 x3 x4 : ℝ)
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1/3) :
  x1 = 4/5 :=
by
  sorry

end determine_x1_l270_270508


namespace integer_values_sides_triangle_l270_270125

theorem integer_values_sides_triangle (x : ℝ) (hx_pos : x > 0) (hx1 : x + 15 > 40) (hx2 : x + 40 > 15) (hx3 : 15 + 40 > x) : 
    (∃ (n : ℤ), ∃ (hn : 0 < n) (hn1 : (n : ℝ) = x) (hn2 : 26 ≤ n) (hn3 : n ≤ 54), 
    ∀ (y : ℤ), (26 ≤ y ∧ y ≤ 54) → (∃ (m : ℤ), y = 26 + m ∧ m < 29 ∧ m ≥ 0)) := 
sorry

end integer_values_sides_triangle_l270_270125


namespace clowns_per_mobile_28_l270_270121

def clowns_in_each_mobile (total_clowns num_mobiles : Nat) (h : total_clowns = 140 ∧ num_mobiles = 5) : Nat :=
  total_clowns / num_mobiles

theorem clowns_per_mobile_28 (total_clowns num_mobiles : Nat) (h : total_clowns = 140 ∧ num_mobiles = 5) :
  clowns_in_each_mobile total_clowns num_mobiles h = 28 :=
by
  sorry

end clowns_per_mobile_28_l270_270121


namespace problem_statement_l270_270350

-- Define the necessary and sufficient conditions
def necessary_but_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ (¬ (P → Q))

-- Specific propositions in this scenario
def x_conditions (x : ℝ) : Prop := x^2 - 2 * x - 3 = 0
def x_equals_3 (x : ℝ) : Prop := x = 3

-- Prove the given problem statement
theorem problem_statement (x : ℝ) : necessary_but_not_sufficient (x_conditions x) (x_equals_3 x) :=
  sorry

end problem_statement_l270_270350


namespace farm_cows_l270_270263

theorem farm_cows (c h : ℕ) 
  (legs_eq : 5 * c + 2 * h = 20 + 2 * (c + h)) : 
  c = 6 :=
by 
  sorry

end farm_cows_l270_270263


namespace economist_wins_by_choosing_method_1_l270_270480

variable (n : ℕ) (h_odd : n % 2 = 1) (h_greater_than_4 : n > 4)

-- Condition for Step 1: Lawyer divides coins into a and b
variable (a b : ℕ) (h_a_b : a + b = n) (h_a_2 : a ≥ 2) (h_b_2 : b ≥ 2) (h_a_lt_b : a < b)

-- Condition for Step 2: Economist divides a into x1 and x2, and b into y1 and y2
variable (x1 x2 y1 y2 : ℕ)
variable (h_x1_x2 : x1 + x2 = a) (h_y1_y2 : y1 + y2 = b)
variable (h_x1_1 : x1 ≥ 1) (h_x2_1 : x2 ≥ 1) (h_y1_1 : y1 ≥ 1) (h_y2_1 : y2 ≥ 1)
variable (h_x1_le_x2 : x1 ≤ x2) (h_y1_le_y2 : y1 ≤ y2)

-- Method 1: Economist takes largest and smallest parts

-- Method 2: Economist takes both middle parts

-- Method 3: Economist chooses method 1 or 2 and gives one coin to the lawyer

theorem economist_wins_by_choosing_method_1 :
  economist_strategy n = method1 :=
sorry

end economist_wins_by_choosing_method_1_l270_270480


namespace arrange_MISSISSIPPI_l270_270367

theorem arrange_MISSISSIPPI : 
  let total_letters := 11
  let count_I := 4
  let count_S := 4
  let count_P := 2
  let arrangements := 11.factorial / (count_I.factorial * count_S.factorial * count_P.factorial)
  in arrangements = 34650 :=
by
  have h1 : 11.factorial = 39916800 := by sorry
  have h2 : 4.factorial = 24 := by sorry
  have h3 : 2.factorial = 2 := by sorry
  have h4 : arrangements = 34650 := by sorry
  exact h4

end arrange_MISSISSIPPI_l270_270367


namespace average_of_two_numbers_l270_270954

theorem average_of_two_numbers (A B C : ℝ) (h1 : (A + B + C)/3 = 48) (h2 : C = 32) : (A + B)/2 = 56 := by
  sorry

end average_of_two_numbers_l270_270954


namespace mean_profit_first_15_days_l270_270626

-- Definitions and conditions
def mean_daily_profit_entire_month : ℝ := 350
def total_days_in_month : ℕ := 30
def mean_daily_profit_last_15_days : ℝ := 445

-- Proof statement
theorem mean_profit_first_15_days : 
  (mean_daily_profit_entire_month * (total_days_in_month : ℝ) 
   - mean_daily_profit_last_15_days * 15) / 15 = 255 :=
by
  sorry

end mean_profit_first_15_days_l270_270626


namespace xiaoli_estimate_greater_l270_270982

variable (p q a b : ℝ)

theorem xiaoli_estimate_greater (hpq : p > q) (hq0 : q > 0) (hab : a > b) : (p + a) - (q + b) > p - q := 
by 
  sorry

end xiaoli_estimate_greater_l270_270982


namespace lucas_change_l270_270098

-- Define the initial amount of money Lucas has
def initial_amount : ℕ := 20

-- Define the cost of one avocado
def cost_per_avocado : ℕ := 2

-- Define the number of avocados Lucas buys
def number_of_avocados : ℕ := 3

-- Calculate the total cost of avocados
def total_cost : ℕ := number_of_avocados * cost_per_avocado

-- Calculate the remaining amount of money (change)
def remaining_amount : ℕ := initial_amount - total_cost

-- The proposition to prove: Lucas brings home $14
theorem lucas_change : remaining_amount = 14 := by
  sorry

end lucas_change_l270_270098


namespace compare_a_b_c_l270_270871

noncomputable def a := 0.1 * Real.exp 0.1
def b := 1 / 9
noncomputable def c := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b := by
  have h_c_b : c < b := sorry
  have h_a_b : a < b := sorry
  have h_c_a : c < a := sorry
  exact ⟨h_c_a, h_a_b⟩

end compare_a_b_c_l270_270871


namespace sum_of_roots_eq_a_plus_b_l270_270407

theorem sum_of_roots_eq_a_plus_b (a b : ℝ) 
  (h : ∀ x : ℝ, x^2 - (a + b) * x + (ab + 1) = 0 → (x = a ∨ x = b)) :
  a + b = a + b :=
by sorry

end sum_of_roots_eq_a_plus_b_l270_270407


namespace spoon_less_than_fork_l270_270632

-- Define the initial price of spoon and fork in kopecks
def initial_price (x : ℕ) : Prop :=
  x > 100 -- ensuring the spoon's sale price remains positive

-- Define the sale price of the spoon
def spoon_sale_price (x : ℕ) : ℕ :=
  x - 100

-- Define the sale price of the fork
def fork_sale_price (x : ℕ) : ℕ :=
  x / 10

-- Prove that the spoon's sale price can be less than the fork's sale price
theorem spoon_less_than_fork (x : ℕ) (h : initial_price x) : 
  spoon_sale_price x < fork_sale_price x :=
by
  sorry

end spoon_less_than_fork_l270_270632


namespace eggs_division_l270_270464

theorem eggs_division (n_students n_eggs : ℕ) (h_students : n_students = 9) (h_eggs : n_eggs = 73):
  n_eggs / n_students = 8 ∧ n_eggs % n_students = 1 :=
by
  rw [h_students, h_eggs]
  exact ⟨rfl, rfl⟩

end eggs_division_l270_270464


namespace probability_all_same_flips_l270_270759

noncomputable def four_same_flips_probability : ℚ := 
  (∑' n : ℕ, if n > 0 then (1/2)^(4*n) else 0)

theorem probability_all_same_flips : 
  four_same_flips_probability = 1 / 15 := 
sorry

end probability_all_same_flips_l270_270759


namespace inequality_xy_l270_270290

theorem inequality_xy {x y : ℝ} (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
  sorry

end inequality_xy_l270_270290


namespace fraction_difference_l270_270816

-- Definitions for the problem conditions
def repeatingDecimal72 := 8 / 11
def decimal72 := 18 / 25

-- Statement that needs to be proven
theorem fraction_difference : repeatingDecimal72 - decimal72 = 2 / 275 := 
by 
  sorry

end fraction_difference_l270_270816


namespace carA_catches_up_with_carB_at_150_km_l270_270360

-- Definitions representing the problem's conditions
variable (t_A t_B v_A v_B : ℝ)
variable (distance_A_B : ℝ := 300)
variable (time_diff_start : ℝ := 1)
variable (time_diff_end : ℝ := 1)

-- Assumptions representing the problem's conditions
axiom speed_carA : v_A = distance_A_B / t_A
axiom speed_carB : v_B = distance_A_B / (t_A + 2)
axiom time_relation : t_B = t_A + 2
axiom time_diff_starting : t_A = t_B - 2

-- The statement to be proven: car A catches up with car B 150 km from city B
theorem carA_catches_up_with_carB_at_150_km :
  ∃ t₀ : ℝ, v_A * t₀ = v_B * (t₀ + time_diff_start) ∧ (distance_A_B - v_A * t₀ = 150) :=
sorry

end carA_catches_up_with_carB_at_150_km_l270_270360


namespace seating_arrangement_l270_270206

theorem seating_arrangement (x y z : ℕ) (h1 : z = x + y) (h2 : x*10 + y*9 = 67) : x = 4 :=
by
  sorry

end seating_arrangement_l270_270206


namespace financier_invariant_l270_270905

theorem financier_invariant (D A : ℤ) (hD : D = 1 ∨ D = 10 * (A - 1) + D ∨ D = D - 1 + 10 * A)
  (hA : A = 0 ∨ A = A + 10 * (1 - D) ∨ A = A - 1):
  (D - A) % 11 = 1 := 
sorry

end financier_invariant_l270_270905


namespace perimeter_of_square_l270_270451

-- Defining the context and proving the equivalence.
theorem perimeter_of_square (x y : ℕ) (h : Nat.gcd x y = 3) (area : ℕ) :
  let lcm_xy := Nat.lcm x y
  let side_length := Real.sqrt (20 * lcm_xy)
  let perimeter := 4 * side_length
  perimeter = 24 * Real.sqrt 5 :=
by
  let lcm_xy := Nat.lcm x y
  let side_length := Real.sqrt (20 * lcm_xy)
  let perimeter := 4 * side_length
  sorry

end perimeter_of_square_l270_270451


namespace triangle_obtuse_l270_270221

theorem triangle_obtuse (a b c : ℝ) (A B C : ℝ) 
  (hBpos : 0 < B) 
  (hBpi : B < Real.pi) 
  (sin_C_lt_cos_A_sin_B : Real.sin C / Real.sin B < Real.cos A) 
  (hC_eq : C = A + B) 
  (ha2 : A + B + C = Real.pi) :
  B > Real.pi / 2 := 
sorry

end triangle_obtuse_l270_270221


namespace exists_element_x_l270_270918

open Set

theorem exists_element_x (n : ℕ) (S : Finset (Fin n)) (A : Fin n → Finset (Fin n)) 
  (h_distinct : ∀ i j : Fin n, i ≠ j → A i ≠ A j) : 
  ∃ x ∈ S, ∀ i j : Fin n, i ≠ j → (A i \ {x}) ≠ (A j \ {x}) :=
sorry

end exists_element_x_l270_270918


namespace probability_different_last_digit_l270_270948

open BigOperators

def count_ways_different_last_digit : ℕ :=
  10 * 9 * 8 * 7 * 6

def total_combinations (n k : ℕ) : ℕ :=
  n.choose k

theorem probability_different_last_digit :
  (count_ways_different_last_digit : ℚ) / (total_combinations 90 5 : ℚ) = 252 / 366244 :=
by
  sorry

end probability_different_last_digit_l270_270948


namespace little_john_initial_money_l270_270565

theorem little_john_initial_money :
  let spent := 1.05
  let given_to_each_friend := 2 * 1.00
  let total_spent := spent + given_to_each_friend
  let left := 2.05
  total_spent + left = 5.10 :=
by
  let spent := 1.05
  let given_to_each_friend := 2 * 1.00
  let total_spent := spent + given_to_each_friend
  let left := 2.05
  show total_spent + left = 5.10
  sorry

end little_john_initial_money_l270_270565


namespace triangle_side_c_l270_270075

noncomputable def angle_B_eq_2A (A B : ℝ) := B = 2 * A
noncomputable def side_a_eq_1 (a : ℝ) := a = 1
noncomputable def side_b_eq_sqrt3 (b : ℝ) := b = Real.sqrt 3

noncomputable def find_side_c (A B C a b c : ℝ) :=
  angle_B_eq_2A A B ∧
  side_a_eq_1 a ∧
  side_b_eq_sqrt3 b →
  c = 2

theorem triangle_side_c (A B C a b c : ℝ) : find_side_c A B C a b c :=
by sorry

end triangle_side_c_l270_270075


namespace find_m_l270_270878

noncomputable def f : ℝ → ℝ := sorry

theorem find_m (h₁ : ∀ x : ℝ, f (2 * x + 1) = 3 * x - 2) (h₂ : f 2 = m) : m = -1 / 2 :=
by
  sorry

end find_m_l270_270878


namespace angle_BDC_proof_l270_270545

noncomputable def angle_sum_triangle (angle_A angle_B angle_C : ℝ) : Prop :=
  angle_A + angle_B + angle_C = 180

-- Given conditions
def angle_A : ℝ := 70
def angle_E : ℝ := 50
def angle_C : ℝ := 40

-- The problem of proving that angle_BDC = 20 degrees
theorem angle_BDC_proof (A E C BDC : ℝ) 
  (hA : A = angle_A)
  (hE : E = angle_E)
  (hC : C = angle_C) :
  BDC = 20 :=
  sorry

end angle_BDC_proof_l270_270545


namespace sqrt_domain_l270_270253

theorem sqrt_domain (x : ℝ) : (∃ y, y * y = x - 2) ↔ (x ≥ 2) :=
by sorry

end sqrt_domain_l270_270253


namespace Bryan_did_258_pushups_l270_270794

-- Define the conditions
def sets : ℕ := 15
def pushups_per_set : ℕ := 18
def pushups_fewer_last_set : ℕ := 12

-- Define the planned total push-ups
def planned_total_pushups : ℕ := sets * pushups_per_set

-- Define the actual push-ups in the last set
def last_set_pushups : ℕ := pushups_per_set - pushups_fewer_last_set

-- Define the total push-ups Bryan did
def total_pushups : ℕ := (sets - 1) * pushups_per_set + last_set_pushups

-- The theorem to prove
theorem Bryan_did_258_pushups :
  total_pushups = 258 := by
  sorry

end Bryan_did_258_pushups_l270_270794


namespace total_weeds_correct_l270_270579

def tuesday : ℕ := 25
def wednesday : ℕ := 3 * tuesday
def thursday : ℕ := wednesday / 5
def friday : ℕ := thursday - 10
def total_weeds : ℕ := tuesday + wednesday + thursday + friday

theorem total_weeds_correct : total_weeds = 120 :=
by
  sorry

end total_weeds_correct_l270_270579


namespace cupcakes_left_over_l270_270575

def total_cupcakes := 40
def ms_delmont_class := 18
def mrs_donnelly_class := 16
def ms_delmont := 1
def mrs_donnelly := 1
def school_nurse := 1
def school_principal := 1

def total_given_away := ms_delmont_class + mrs_donnelly_class + ms_delmont + mrs_donnelly + school_nurse + school_principal

theorem cupcakes_left_over : total_cupcakes - total_given_away = 2 := by
  sorry

end cupcakes_left_over_l270_270575


namespace real_roots_of_quadratic_l270_270660

theorem real_roots_of_quadratic (m : ℝ) : ((m - 2) ≠ 0 ∧ (-4 * m + 24) ≥ 0) → (m ≤ 6 ∧ m ≠ 2) := 
by 
  sorry

end real_roots_of_quadratic_l270_270660


namespace inequality_proof_l270_270286

variable (x y : ℝ)
variable (h1 : x ≥ 0)
variable (h2 : y ≥ 0)
variable (h3 : x + y ≤ 1)

theorem inequality_proof (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y ≤ 1) : 
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := 
by 
  sorry

end inequality_proof_l270_270286


namespace sum_of_original_numbers_l270_270439

theorem sum_of_original_numbers :
  ∃ a b : ℚ, a = b + 12 ∧ a^2 + b^2 = 169 / 2 ∧ (a^2)^2 - (b^2)^2 = 5070 ∧ a + b = 5 :=
by
  sorry

end sum_of_original_numbers_l270_270439


namespace orthocenter_of_triangle_ABC_l270_270902

open Point

-- Definition of the points A, B, and C.
def A : Point ℝ := (2, 3, 4)
def B : Point ℝ := (6, 4, 2)
def C : Point ℝ := (4, 5, 6)

-- Definition of the orthocenter H of the triangle ABC.
def H : Point ℝ := (4, 3, 2)

-- Lean statement to prove that H is the orthocenter of triangle ABC.
theorem orthocenter_of_triangle_ABC : 
  orthocenter A B C = H :=
sorry

end orthocenter_of_triangle_ABC_l270_270902


namespace correctly_transformed_equation_l270_270981

theorem correctly_transformed_equation (s a b x y : ℝ) :
  (s = a * b → a = s / b ∧ b ≠ 0) ∧
  (1/2 * x = 8 → x = 16) ∧
  (-x - 1 = y - 1 → x = -y) ∧
  (a = b → a + 3 = b + 3) :=
by
  sorry

end correctly_transformed_equation_l270_270981


namespace sqrt_domain_l270_270254

theorem sqrt_domain (x : ℝ) : (∃ y, y * y = x - 2) ↔ (x ≥ 2) :=
by sorry

end sqrt_domain_l270_270254


namespace equivalence_condition_l270_270385

theorem equivalence_condition (a b c d : ℝ) (h : (a + b) / (b + c) = (c + d) / (d + a)) : 
  a = c ∨ a + b + c + d = 0 :=
sorry

end equivalence_condition_l270_270385


namespace repeating_decimal_exceeds_finite_decimal_by_l270_270810

-- Definitions based on the problem conditions
def repeating_decimal := 8 / 11
def finite_decimal := 18 / 25

-- Statement to be proved
theorem repeating_decimal_exceeds_finite_decimal_by : 
  repeating_decimal - finite_decimal = 2 / 275 :=
by
  -- Skipping the proof itself with 'sorry'
  sorry

end repeating_decimal_exceeds_finite_decimal_by_l270_270810


namespace landscape_breadth_l270_270117

theorem landscape_breadth (L B : ℕ) (h1 : B = 8 * L)
  (h2 : 3200 = 1 / 9 * (L * B))
  (h3 : B * B = 28800) :
  B = 480 := by
  sorry

end landscape_breadth_l270_270117


namespace range_of_k_l270_270541

theorem range_of_k (k : ℝ) : 
  (∃ a b : ℝ, x^2 + ky^2 = 2 ∧ a^2 = 2/k ∧ b^2 = 2 ∧ a > b) → 0 < k ∧ k < 1 :=
by {
  sorry
}

end range_of_k_l270_270541


namespace fewerSevensCanProduce100_l270_270147

noncomputable def validAs100UsingSevens : (ℕ → ℕ) → Prop := 
  fun (f : ℕ → ℕ) => ∃ (x y : ℕ), (fewer_than_ten_sevens x y) ∧ f x y = 100

theorem fewerSevensCanProduce100 : 
  ∃ (f : ℕ → ℕ), validAs100UsingSevens f :=
by 
  sorry

end fewerSevensCanProduce100_l270_270147


namespace inequality_proof_l270_270287

variable (x y : ℝ)
variable (h1 : x ≥ 0)
variable (h2 : y ≥ 0)
variable (h3 : x + y ≤ 1)

theorem inequality_proof (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y ≤ 1) : 
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := 
by 
  sorry

end inequality_proof_l270_270287


namespace minimum_questions_to_determine_village_l270_270598

-- Step 1: Define the types of villages
inductive Village
| A : Village
| B : Village
| C : Village

-- Step 2: Define the properties of residents in each village
def tells_truth (v : Village) (p : Prop) : Prop :=
  match v with
  | Village.A => p
  | Village.B => ¬p
  | Village.C => p ∨ ¬p

-- Step 3: Define the problem context in Lean
theorem minimum_questions_to_determine_village :
    ∀ (tourist_village person_village : Village), ∃ (n : ℕ), n = 4 := by
  sorry

end minimum_questions_to_determine_village_l270_270598


namespace intersection_M_N_l270_270093

-- Given set M defined by the inequality
def M : Set ℝ := {x | x^2 + x - 6 < 0}

-- Given set N defined by the interval
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- The intersection M ∩ N should be equal to the interval [1, 2)
theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_l270_270093


namespace clock_angle_150_at_5pm_l270_270501

theorem clock_angle_150_at_5pm :
  (∀ t : ℕ, (t = 5) ↔ (∀ θ : ℝ, θ = 150 → θ = (30 * t))) := sorry

end clock_angle_150_at_5pm_l270_270501


namespace average_visitors_per_day_l270_270620

theorem average_visitors_per_day
  (sunday_visitors : ℕ := 540)
  (other_days_visitors : ℕ := 240)
  (days_in_month : ℕ := 30)
  (first_day_is_sunday : Bool := true)
  (result : ℕ := 290) :
  let num_sundays := 5
  let num_other_days := days_in_month - num_sundays
  let total_visitors := num_sundays * sunday_visitors + num_other_days * other_days_visitors
  let average_visitors := total_visitors / days_in_month
  average_visitors = result :=
by
  sorry

end average_visitors_per_day_l270_270620


namespace b_money_used_for_10_months_l270_270173

theorem b_money_used_for_10_months
  (a_capital_ratio : ℚ)
  (a_time_used : ℕ)
  (b_profit_share : ℚ)
  (h1 : a_capital_ratio = 1 / 4)
  (h2 : a_time_used = 15)
  (h3 : b_profit_share = 2 / 3) :
  ∃ (b_time_used : ℕ), b_time_used = 10 :=
by
  sorry

end b_money_used_for_10_months_l270_270173


namespace product_of_two_numbers_l270_270462

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 16) (h2 : x^2 + y^2 = 200) : x * y = 28 :=
sorry

end product_of_two_numbers_l270_270462


namespace range_of_a_l270_270250

theorem range_of_a (a : ℝ)
  (h : ∀ x y : ℝ, (4 * x - 3 * y - 2 = 0) → (x^2 + y^2 - 2 * a * x + 4 * y + a^2 - 12 = 0) → x ≠ y) :
  -6 < a ∧ a < 4 :=
by
  sorry

end range_of_a_l270_270250


namespace factorization_of_square_difference_l270_270849

variable (t : ℝ)

theorem factorization_of_square_difference : t^2 - 144 = (t - 12) * (t + 12) := 
sorry

end factorization_of_square_difference_l270_270849


namespace vector_evaluation_l270_270375

-- Define the vectors
def v1 : ℝ × ℝ := (3, -2)
def v2 : ℝ × ℝ := (2, -6)
def v3 : ℝ × ℝ := (0, 3)
def scalar : ℝ := 5
def expected_result : ℝ × ℝ := (-7, 31)

-- Statement to be proved
theorem vector_evaluation : v1 - scalar • v2 + v3 = expected_result :=
by
  sorry

end vector_evaluation_l270_270375


namespace tod_trip_time_l270_270968

noncomputable def total_time (d1 d2 d3 d4 s1 s2 s3 s4 : ℝ) : ℝ :=
  d1 / s1 + d2 / s2 + d3 / s3 + d4 / s4

theorem tod_trip_time :
  total_time 55 95 30 75 40 50 20 60 = 6.025 :=
by 
  sorry

end tod_trip_time_l270_270968


namespace vasya_example_fewer_sevens_l270_270155

theorem vasya_example_fewer_sevens : 
  (777 / 7) - (77 / 7) = 100 :=
begin
  sorry
end

end vasya_example_fewer_sevens_l270_270155


namespace marbles_remaining_l270_270567

theorem marbles_remaining 
  (initial_remaining : ℕ := 400)
  (num_customers : ℕ := 20)
  (marbles_per_customer : ℕ := 15) :
  initial_remaining - (num_customers * marbles_per_customer) = 100 :=
by
  sorry

end marbles_remaining_l270_270567


namespace ratio_of_areas_l270_270761

noncomputable def circumferences_equal_arcs (C1 C2 : ℝ) (k1 k2 : ℕ) : Prop :=
  (k1 : ℝ) / 360 * C1 = (k2 : ℝ) / 360 * C2

theorem ratio_of_areas (C1 C2 : ℝ) (h : circumferences_equal_arcs C1 C2 60 30) :
  (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 1 / 4 :=
by
  sorry

end ratio_of_areas_l270_270761


namespace line_circle_no_intersection_l270_270843

theorem line_circle_no_intersection :
  (∀ (x y : ℝ), 3 * x + 4 * y = 12 ∨ x^2 + y^2 = 4) →
  (∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4) →
  false :=
by
  sorry

end line_circle_no_intersection_l270_270843


namespace number_of_integer_values_of_x_l270_270123

theorem number_of_integer_values_of_x (x : ℕ) (hx1 : x + 15 > 40) (hx2 : x + 40 > 15) (hx3 : 15 + 40 > x) :
  ∃ n : ℕ, n = 29 ∧ ∀ y : ℕ, (26 ≤ y ∧ y ≤ 54) ↔ true :=
by
  sorry

end number_of_integer_values_of_x_l270_270123


namespace first_comparison_second_comparison_l270_270768

theorem first_comparison (x y : ℕ) (h1 : x = 2^40) (h2 : y = 3^28) : x < y := 
by sorry

theorem second_comparison (a b : ℕ) (h3 : a = 31^11) (h4 : b = 17^14) : a < b := 
by sorry

end first_comparison_second_comparison_l270_270768


namespace selectedParticipants_correct_l270_270757

-- Define the random number table portion used in the problem
def randomNumTable := [
  [12, 56, 85, 99, 26, 96, 96, 68, 27, 31, 05, 03, 72, 93, 15, 57, 12, 10, 14, 21, 88, 26, 49, 81, 76]
]

-- Define the conditions
def totalStudents := 247
def selectedStudentsCount := 4
def startingIndexRow := 4
def startingIndexCol := 9
def startingNumber := randomNumTable[0][8]

-- Define the expected selected participants' numbers
def expectedParticipants := [050, 121, 014, 218]

-- The Lean statement that needs to be proved
theorem selectedParticipants_correct : expectedParticipants = [050, 121, 014, 218] := by
  sorry

end selectedParticipants_correct_l270_270757


namespace eta_expectation_and_variance_l270_270528

noncomputable def ξ : Type := sorry

def η := 5 * ξ

theorem eta_expectation_and_variance :
  E(η) = 25 / 2 ∧ D(η) = 125 / 4 := by
sorry

end eta_expectation_and_variance_l270_270528


namespace pencil_and_pen_choice_count_l270_270318

-- Definitions based on the given conditions
def numPencilTypes : Nat := 4
def numPenTypes : Nat := 6

-- Statement we want to prove
theorem pencil_and_pen_choice_count : (numPencilTypes * numPenTypes) = 24 :=
by
  sorry

end pencil_and_pen_choice_count_l270_270318


namespace find_x_l270_270343

theorem find_x (number x : ℝ) (h1 : 24 * number = 173 * x) (h2 : 24 * number = 1730) : x = 10 :=
by
  sorry

end find_x_l270_270343


namespace count_special_integers_l270_270681

theorem count_special_integers : ∃ n : ℕ, n = 2151 ∧ 
  ∀ m : ℕ, (m < 100000 ∧ (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ 
    ((∀ k, k < 5 → m.digit k = d1 ∨ m.digit k = d2) ∨ 
    (∀ k, k < 5 → m.digit k = d1)))) → m ∈ finset.range 100000 :=
by sorry

end count_special_integers_l270_270681


namespace sum_of_cubes_of_consecutive_integers_l270_270963

theorem sum_of_cubes_of_consecutive_integers :
  ∃ (a b c d : ℕ), a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ (a^2 + b^2 + c^2 + d^2 = 9340) ∧ (a^3 + b^3 + c^3 + d^3 = 457064) :=
by
  sorry

end sum_of_cubes_of_consecutive_integers_l270_270963


namespace can_construct_segment_l270_270403

noncomputable def constructSegment (Ω₁ Ω₂ : Set (ℝ × ℝ)) (P : ℝ × ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), A ∈ Ω₁ ∧ B ∈ Ω₂ ∧ (A + B) / 2 = P

theorem can_construct_segment (Ω₁ Ω₂ : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ Ω₁ ∧ B ∈ Ω₂ ∧ (A + B) / 2 = P) :=
sorry

end can_construct_segment_l270_270403


namespace find_a_l270_270874

-- Define the variables
variables (m d a b : ℝ)

-- State the main theorem with conditions
theorem find_a (h : m = d * a * b / (a - b)) (h_ne : m ≠ d * b) : a = m * b / (m - d * b) :=
sorry

end find_a_l270_270874


namespace unique_function_l270_270029

def satisfies_inequality (f : ℝ → ℝ) (k : ℤ) : Prop :=
  ∀ x y z : ℝ, f (x * y) + f (x + z) + k * f x * f (y * z) ≥ k^2

theorem unique_function (k : ℤ) (h : k > 0) :
  ∃! f : ℝ → ℝ, satisfies_inequality f k :=
by
  sorry

end unique_function_l270_270029


namespace sqrt_x_minus_2_meaningful_in_reals_l270_270255

theorem sqrt_x_minus_2_meaningful_in_reals (x : ℝ) : (∃ (y : ℝ), y * y = x - 2) → x ≥ 2 :=
by
  sorry

end sqrt_x_minus_2_meaningful_in_reals_l270_270255


namespace excess_common_fraction_l270_270805

theorem excess_common_fraction :
  let x := (72 / 10^2 + 7.2 * (10 / 10^2) * (10^{-2} / (1 - 10^{-2}))) in
  x - (72 / 100) = 8 / 1100 :=
by
  sorry

end excess_common_fraction_l270_270805


namespace perm_mississippi_l270_270371

theorem perm_mississippi : 
  let n := 11
  let n_S := 4
  let n_I := 4
  let n_P := 2
  let n_M := 1
  ∏ (mississippi_perm := Nat.factorial n / (Nat.factorial n_S * Nat.factorial n_I * Nat.factorial n_P * Nat.factorial n_M)) :=
  mississippi_perm = 34650 :=
by sorry

end perm_mississippi_l270_270371


namespace each_person_ate_2_cakes_l270_270279

def initial_cakes : ℕ := 8
def number_of_friends : ℕ := 4

theorem each_person_ate_2_cakes (h_initial_cakes : initial_cakes = 8)
  (h_number_of_friends : number_of_friends = 4) :
  initial_cakes / number_of_friends = 2 :=
by sorry

end each_person_ate_2_cakes_l270_270279


namespace reciprocal_of_neg_2023_l270_270454

theorem reciprocal_of_neg_2023 : (1 : ℝ) / (-2023) = -(1 / 2023) :=
by 
  sorry

end reciprocal_of_neg_2023_l270_270454


namespace sarah_total_weeds_l270_270581

theorem sarah_total_weeds :
  let tuesday_weeds := 25
  let wednesday_weeds := 3 * tuesday_weeds
  let thursday_weeds := wednesday_weeds / 5
  let friday_weeds := thursday_weeds - 10
  tuesday_weeds + wednesday_weeds + thursday_weeds + friday_weeds = 120 :=
by
  intros
  let tuesday_weeds := 25
  let wednesday_weeds := 3 * tuesday_weeds
  let thursday_weeds := wednesday_weeds / 5
  let friday_weeds := thursday_weeds - 10
  sorry

end sarah_total_weeds_l270_270581


namespace probability_neither_defective_l270_270985

def total_pens : ℕ := 8
def defective_pens : ℕ := 3
def non_defective_pens : ℕ := total_pens - defective_pens
def draw_count : ℕ := 2

def probability_of_non_defective (total : ℕ) (defective : ℕ) (draws : ℕ) : ℚ :=
  let non_defective := total - defective
  (non_defective / total) * ((non_defective - 1) / (total - 1))

theorem probability_neither_defective :
  probability_of_non_defective total_pens defective_pens draw_count = 5 / 14 :=
by sorry

end probability_neither_defective_l270_270985


namespace probability_htth_l270_270077

def probability_of_sequence_HTTH := (1 / 2) * (1 / 2) * (1 / 2) * (1 / 2)

theorem probability_htth : probability_of_sequence_HTTH = 1 / 16 := by
  sorry

end probability_htth_l270_270077


namespace average_speed_last_segment_l270_270427

theorem average_speed_last_segment
  (total_distance : ℕ)
  (total_time : ℕ)
  (speed1 speed2 speed3 : ℕ)
  (last_segment_time : ℕ)
  (average_speed_total : ℕ) :
  total_distance = 180 →
  total_time = 180 →
  speed1 = 40 →
  speed2 = 50 →
  speed3 = 60 →
  average_speed_total = 60 →
  last_segment_time = 45 →
  ∃ (speed4 : ℕ), speed4 = 90 :=
by sorry

end average_speed_last_segment_l270_270427


namespace vasya_100_using_fewer_sevens_l270_270158

-- Definitions and conditions
def seven := 7

-- Theorem to prove
theorem vasya_100_using_fewer_sevens :
  (777 / seven - 77 / seven = 100) ∨
  (seven * seven + seven * seven + seven / seven + seven / seven = 100) :=
by
  sorry

end vasya_100_using_fewer_sevens_l270_270158


namespace part1_part2_l270_270040

open Set

variable (U : Set ℤ := {x | -3 ≤ x ∧ x ≤ 3})
variable (A : Set ℤ := {1, 2, 3})
variable (B : Set ℤ := {-1, 0, 1})
variable (C : Set ℤ := {-2, 0, 2})

theorem part1 : A ∪ (B ∩ C) = {0, 1, 2, 3} := by
  sorry

theorem part2 : A ∩ Uᶜ ∪ (B ∪ C) = {3} := by
  sorry

end part1_part2_l270_270040


namespace unique_root_iff_k_eq_4_l270_270402

theorem unique_root_iff_k_eq_4 (k : ℝ) : 
  (∃! x : ℝ, x^2 - 4 * x + k = 0) ↔ k = 4 := 
by {
  sorry
}

end unique_root_iff_k_eq_4_l270_270402


namespace original_number_of_laborers_l270_270488

theorem original_number_of_laborers 
(L : ℕ) (h1 : L * 15 = (L - 5) * 20) : L = 15 :=
sorry

end original_number_of_laborers_l270_270488


namespace find_a6_l270_270875

variable {a : ℕ → ℝ}
variable {q : ℝ}
variable {a₁ : ℝ}

/-- The sequence is a geometric sequence -/
axiom geom_seq (n : ℕ) : a n = a₁ * q ^ (n - 1)

/-- The sum of the first three terms is 168 -/
axiom sum_of_first_three_terms : a₁ + a₁ * q + a₁ * q ^ 2 = 168

/-- The difference between the 2nd and the 5th terms is 42 -/
axiom difference_a2_a5 : a₁ * q - a₁ * q ^ 4 = 42

theorem find_a6 : a 6 = 3 :=
by
  -- Proof goes here
  sorry

end find_a6_l270_270875


namespace total_net_gain_computation_l270_270280

noncomputable def house1_initial_value : ℝ := 15000
noncomputable def house2_initial_value : ℝ := 20000

noncomputable def house1_selling_price : ℝ := 1.15 * house1_initial_value
noncomputable def house2_selling_price : ℝ := 1.2 * house2_initial_value

noncomputable def house1_buy_back_price : ℝ := 0.85 * house1_selling_price
noncomputable def house2_buy_back_price : ℝ := 0.8 * house2_selling_price

noncomputable def house1_profit : ℝ := house1_selling_price - house1_buy_back_price
noncomputable def house2_profit : ℝ := house2_selling_price - house2_buy_back_price

noncomputable def total_net_gain : ℝ := house1_profit + house2_profit

theorem total_net_gain_computation : total_net_gain = 7387.5 :=
by
  sorry

end total_net_gain_computation_l270_270280


namespace correct_options_l270_270053

theorem correct_options (a b c : ℝ) (h1 : ∀ x : ℝ, (a*x^2 + b*x + c > 0) ↔ (-3 < x ∧ x < 2)) :
  (a < 0) ∧ (a + b + c > 0) ∧ (∀ x, (b*x + c > 0) ↔ x > 6) = False ∧ (∀ x, (c*x^2 + b*x + a < 0) ↔ (-1/3 < x ∧ x < 1/2)) :=
by 
  sorry

end correct_options_l270_270053


namespace waiting_probability_no_more_than_10_seconds_l270_270015

def total_cycle_time : ℕ := 30 + 10 + 40
def proceed_during_time : ℕ := 40 -- green time
def yellow_time : ℕ := 10

theorem waiting_probability_no_more_than_10_seconds :
  (proceed_during_time + yellow_time + yellow_time) / total_cycle_time = 3 / 4 := by
  sorry

end waiting_probability_no_more_than_10_seconds_l270_270015


namespace repeating_seventy_two_exceeds_seventy_two_l270_270829

noncomputable def repeating_decimal (n d : ℕ) : ℚ := n / d

theorem repeating_seventy_two_exceeds_seventy_two :
  repeating_decimal 72 99 - (72 / 100) = (2 / 275) := 
sorry

end repeating_seventy_two_exceeds_seventy_two_l270_270829


namespace number_of_item_B_l270_270583

theorem number_of_item_B
    (x y z : ℕ)
    (total_items total_cost : ℕ)
    (hx_price : 1 ≤ x ∧ x ≤ 100)
    (hy_price : 1 ≤ y ∧ y ≤ 100)
    (hz_price : 1 ≤ z ∧ z ≤ 100)
    (h_total_items : total_items = 100)
    (h_total_cost : total_cost = 100)
    (h_price_equation : (x / 8) + 10 * y = z)
    (h_item_equation : x + y + (total_items - (x + y)) = total_items)
    : total_items - (x + y) = 21 :=
sorry

end number_of_item_B_l270_270583


namespace fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l270_270141

theorem fewer_than_ten_sevens_example1 : (777 / 7) - (77 / 7) = 100 :=
  by sorry

theorem fewer_than_ten_sevens_example2 : (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7) = 100 :=
  by sorry

end fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l270_270141


namespace intervals_monotonicity_a1_range_of_a_for_monotonicity_l270_270673

open Real

noncomputable def f (x a : ℝ) : ℝ := (3 * x) / a - 2 * x^2 + log x

noncomputable def f_prime (x a : ℝ) : ℝ := (3 / a) - 4 * x + 1 / x

theorem intervals_monotonicity_a1 :
  ∀ (x : ℝ), (0 < x ∧ x < 1 → (f_prime x 1) > 0) ∧ (1 < x → (f_prime x 1) < 0) :=
by sorry

noncomputable def h (x : ℝ) : ℝ := 4 * x - 1 / x

theorem range_of_a_for_monotonicity :
  ∀ a : ℝ, (0 < a ≤ 2 / 5) ↔ ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → (f_prime x a) ≥ 0 :=
by sorry

end intervals_monotonicity_a1_range_of_a_for_monotonicity_l270_270673


namespace arrangements_mississippi_l270_270365

open Nat

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem arrangements_mississippi : 
  let total := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  factorial total / (factorial m_count * factorial i_count * factorial s_count * factorial p_count) = 34650 :=
by
  -- This is the proof placeholder
  sorry

end arrangements_mississippi_l270_270365


namespace find_angle_BCD_l270_270547

-- Defining the given conditions in the problem
def angleA : ℝ := 100
def angleD : ℝ := 120
def angleE : ℝ := 80
def angleABC : ℝ := 140
def pentagonInteriorAngleSum : ℝ := 540

-- Statement: Prove that the measure of ∠ BCD is 100 degrees given the conditions
theorem find_angle_BCD (h1 : angleA = 100) (h2 : angleD = 120) (h3 : angleE = 80) 
                       (h4 : angleABC = 140) (h5 : pentagonInteriorAngleSum = 540) :
    (angleBCD : ℝ) = 100 :=
sorry

end find_angle_BCD_l270_270547


namespace parabola_focus_distance_l270_270523

theorem parabola_focus_distance (C : Set (ℝ × ℝ))
  (hC : ∀ x y, (y^2 = x) → (x, y) ∈ C)
  (F : ℝ × ℝ)
  (hF : F = (1/4, 0))
  (A : ℝ × ℝ)
  (hA : A = (x0, y0) ∧ (y0^2 = x0 ∧ (x0, y0) ∈ C))
  (hAF : dist A F = (5/4) * x0) :
  x0 = 1 :=
sorry

end parabola_focus_distance_l270_270523


namespace seven_expression_one_seven_expression_two_l270_270140

theorem seven_expression_one : 777 / 7 - 77 / 7 = 100 :=
by sorry

theorem seven_expression_two : 7 * 7 + 7 * 7 + 7 / 7 + 7 / 7 = 100 :=
by sorry

end seven_expression_one_seven_expression_two_l270_270140


namespace four_consecutive_numbers_l270_270317

theorem four_consecutive_numbers (numbers : List ℝ) (h_distinct : numbers.Nodup) (h_length : numbers.length = 100) :
  ∃ (a b c d : ℝ) (h_seq : ([a, b, c, d] ∈ numbers.cyclicPermutations)), b + c < a + d :=
by
  sorry

end four_consecutive_numbers_l270_270317


namespace inverse_of_f_at_neg2_l270_270894

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Define the property of the inverse function we need to prove
theorem inverse_of_f_at_neg2 : f (-(3/2)) = -2 :=
  by
    -- Placeholder for the proof
    sorry

end inverse_of_f_at_neg2_l270_270894


namespace max_stamps_l270_270331

theorem max_stamps (n friends extra total: ℕ) (h1: friends = 15) (h2: extra = 5) (h3: total < 150) : total ≤ 140 :=
by
  sorry

end max_stamps_l270_270331


namespace area_of_trapezium_l270_270857

def length_parallel_side1 : ℝ := 20
def length_parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 15
def expected_area : ℝ := 285

theorem area_of_trapezium :
  (1 / 2) * (length_parallel_side1 + length_parallel_side2) * distance_between_sides = expected_area :=
by sorry

end area_of_trapezium_l270_270857


namespace speed_of_current_l270_270346

theorem speed_of_current (v : ℝ) : 
  (∀ s, s = 3 → s / (3 - v) = 2.3076923076923075) → v = 1.7 := 
by
  intro h
  sorry

end speed_of_current_l270_270346


namespace negation_of_exists_proposition_l270_270958

theorem negation_of_exists_proposition :
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≠ 0 :=
sorry

end negation_of_exists_proposition_l270_270958


namespace num_digits_less_than_100000_with_two_or_fewer_different_digits_l270_270690

theorem num_digits_less_than_100000_with_two_or_fewer_different_digits:
  let count_single_digit : Nat := 9 * 5
  let count_two_distinct_without_zero : Nat := let comb_9_2 : Nat := 9.choose 2
                                               comb_9_2 * (2 + 6 + 14 + 30)
  let count_two_distinct_with_zero : Nat := let num_ways_with_zero : Nat := 9 * (1 + 3 + 7 + 15)
                                            num_ways_with_zero
  let total_count : Nat := count_single_digit + count_two_distinct_without_zero + count_two_distinct_with_zero
  total_count = 2151 := by
  sorry

end num_digits_less_than_100000_with_two_or_fewer_different_digits_l270_270690


namespace simplify_expression_l270_270584

theorem simplify_expression (x : ℝ) : 5 * x + 7 * x - 3 * x = 9 * x :=
by
  sorry

end simplify_expression_l270_270584


namespace base_case_proof_l270_270971

noncomputable def base_case_inequality := 1 + (1 / (2 ^ 3)) < 2 - (1 / 2)

theorem base_case_proof : base_case_inequality := by
  -- The proof would go here
  sorry

end base_case_proof_l270_270971


namespace find_x_l270_270479

theorem find_x
  (x : ℝ)
  (h : (x + 1) / (x + 5) = (x + 5) / (x + 13)) :
  x = 3 :=
sorry

end find_x_l270_270479


namespace repeatingDecimal_exceeds_l270_270832

noncomputable def repeatingDecimalToFraction (d : ℚ) : ℚ := 
    -- Function to convert repeating decimal to fraction
    if d = 0.99999 then 1 else (d * 100 - d) / 99  -- Simplified conversion for demonstration

def decimalToFraction (d : ℚ) : ℚ :=
    -- Function to convert decimal to fraction
    if d = 0.72 then 18 / 25 else 0  -- Replace with actual conversion

theorem repeatingDecimal_exceeds (x y : ℚ) (hx : repeatingDecimalToFraction x = 8/11) (hy : decimalToFraction y = 18/25):
    x - y = 2 / 275 :=
by
    sorry

end repeatingDecimal_exceeds_l270_270832


namespace find_values_l270_270653

theorem find_values (a b : ℝ) (h : a^2 * b^2 + a^2 + b^2 + 1 = 4 * a * b) :
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) :=
by
  sorry

end find_values_l270_270653


namespace total_length_correct_l270_270344

-- Definitions for the first area's path length and scale.
def first_area_scale : ℕ := 500
def first_area_path_length_inches : ℕ := 6
def first_area_path_length_feet : ℕ := first_area_scale * first_area_path_length_inches

-- Definitions for the second area's path length and scale.
def second_area_scale : ℕ := 1000
def second_area_path_length_inches : ℕ := 3
def second_area_path_length_feet : ℕ := second_area_scale * second_area_path_length_inches

-- Total length represented by both paths in feet.
def total_path_length_feet : ℕ :=
  first_area_path_length_feet + second_area_path_length_feet

-- The Lean theorem proving that the total length is 6000 feet.
theorem total_length_correct : total_path_length_feet = 6000 := by
  sorry

end total_length_correct_l270_270344


namespace problemStatement_l270_270267

-- Define the set of values as a type
structure SetOfValues where
  k : ℤ
  b : ℤ

-- The given sets of values
def A : SetOfValues := ⟨2, 2⟩
def B : SetOfValues := ⟨2, -2⟩
def C : SetOfValues := ⟨-2, -2⟩
def D : SetOfValues := ⟨-2, 2⟩

-- Define the conditions for the function
def isValidSet (s : SetOfValues) : Prop :=
  s.k < 0 ∧ s.b > 0

-- The problem statement: Prove that D is a valid set
theorem problemStatement : isValidSet D := by
  sorry

end problemStatement_l270_270267


namespace intersection_of_sets_l270_270390

theorem intersection_of_sets :
  let A := {y : ℝ | ∃ x : ℝ, y = Real.sin x}
  let B := {y : ℝ | ∃ x : ℝ, y = Real.sqrt (-(x^2 - 4*x + 3))}
  A ∩ B = {y : ℝ | 0 ≤ y ∧ y ≤ 1} :=
by
  sorry

end intersection_of_sets_l270_270390


namespace find_m_l270_270046

theorem find_m (x y m : ℤ) (h1 : x = 2) (h2 : y = -3) (h3 : 3 * x - 4 * (m - 1) * y + 30 = 0) : m = -2 :=
by
  sorry

end find_m_l270_270046


namespace student_percentage_to_pass_l270_270788

/-- A student needs to obtain 50% of the total marks to pass given the conditions:
    1. The student got 200 marks.
    2. The student failed by 20 marks.
    3. The maximum marks are 440. -/
theorem student_percentage_to_pass : 
  ∀ (student_marks : ℕ) (failed_by : ℕ) (max_marks : ℕ),
  student_marks = 200 → failed_by = 20 → max_marks = 440 →
  (student_marks + failed_by) / max_marks * 100 = 50 := 
by
  intros student_marks failed_by max_marks h1 h2 h3
  sorry

end student_percentage_to_pass_l270_270788


namespace range_of_a_l270_270404

theorem range_of_a (a : ℝ) (an bn : ℕ → ℝ)
  (h_an : ∀ n, an n = (-1) ^ (n + 2013) * a)
  (h_bn : ∀ n, bn n = 2 + (-1) ^ (n + 2014) / n)
  (h_condition : ∀ n : ℕ, 1 ≤ n → an n < bn n) :
  -2 ≤ a ∧ a < 1 :=
by
  sorry

end range_of_a_l270_270404


namespace f_1_eq_0_range_x_l270_270561

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined_on_Rstar : ∀ x : ℝ, ¬ (x = 0) → f x = sorry
axiom f_4_eq_1 : f 4 = 1
axiom f_mult : ∀ (x₁ x₂ : ℝ), ¬ (x₁ = 0) → ¬ (x₂ = 0) → f (x₁ * x₂) = f x₁ + f x₂
axiom f_increasing : ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f x₁ < f x₂

theorem f_1_eq_0 : f 1 = 0 := sorry

theorem range_x (x : ℝ) : f (3 * x + 1) + f (2 * x - 6) ≤ 3 → 3 < x ∧ x ≤ 5 := sorry

end f_1_eq_0_range_x_l270_270561


namespace selling_price_same_loss_as_profit_l270_270961

theorem selling_price_same_loss_as_profit (cost_price selling_price_with_profit selling_price_with_loss profit loss : ℝ)
  (h1 : selling_price_with_profit - cost_price = profit)
  (h2 : cost_price - selling_price_with_loss = loss)
  (h3 : profit = loss) :
  selling_price_with_loss = 52 :=
by
  have h4 : selling_price_with_profit = 66 := by sorry
  have h5 : cost_price = 59 := by sorry
  have h6 : profit = 66 - 59 := by sorry
  have h7 : profit = 7 := by sorry
  have h8 : loss = 59 - selling_price_with_loss := by sorry
  have h9 : loss = 7 := by sorry
  have h10 : selling_price_with_loss = 59 - loss := by sorry
  have h11 : selling_price_with_loss = 59 - 7 := by sorry
  have h12 : selling_price_with_loss = 52 := by sorry
  exact h12

end selling_price_same_loss_as_profit_l270_270961


namespace arithmetic_sequence_count_l270_270042

theorem arithmetic_sequence_count :
  ∃! (n a d : ℕ), n ≥ 3 ∧ (n * (2 * a + (n - 1) * d) = 2 * 97^2) :=
sorry

end arithmetic_sequence_count_l270_270042


namespace trajectory_of_Q_existence_of_M_l270_270530

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := (x + 2) ^ 2 + y ^ 2 = 81 / 16
def C2 (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 1 / 16

-- Define the conditions about circle Q
def is_tangent_to_both (Q : ℝ → ℝ → Prop) : Prop :=
  ∃ r : ℝ, (∀ x y : ℝ, Q x y → (x + 2)^2 + y^2 = (r + 9/4)^2) ∧ (∀ x y : ℝ, Q x y → (x - 2)^2 + y^2 = (r + 1/4)^2)

-- Prove the trajectory of the center of Q
theorem trajectory_of_Q (Q : ℝ → ℝ → Prop) (h : is_tangent_to_both Q) :
  ∀ x y : ℝ, Q x y ↔ (x^2 - y^2 / 3 = 1 ∧ x ≥ 1) :=
sorry

-- Prove the existence and coordinates of M
theorem existence_of_M (M : ℝ) (Q : ℝ → ℝ → Prop) (h : is_tangent_to_both Q) :
  ∃ x y : ℝ, (x, y) = (-1, 0) ∧ (∀ x0 y0 : ℝ, Q x0 y0 → ((-y0 / (x0 - 2) = 2 * (y0 / (x0 - M)) / (1 - (y0 / (x0 - M))^2)) ↔ M = -1)) :=
sorry

end trajectory_of_Q_existence_of_M_l270_270530


namespace number_of_integer_values_of_x_l270_270124

theorem number_of_integer_values_of_x (x : ℕ) (hx1 : x + 15 > 40) (hx2 : x + 40 > 15) (hx3 : 15 + 40 > x) :
  ∃ n : ℕ, n = 29 ∧ ∀ y : ℕ, (26 ≤ y ∧ y ≤ 54) ↔ true :=
by
  sorry

end number_of_integer_values_of_x_l270_270124


namespace initial_birds_was_one_l270_270321

def initial_birds (b : Nat) : Prop :=
  b + 4 = 5

theorem initial_birds_was_one : ∃ b, initial_birds b ∧ b = 1 :=
by
  use 1
  unfold initial_birds
  sorry

end initial_birds_was_one_l270_270321


namespace contrapositive_a_eq_b_imp_a_sq_eq_b_sq_l270_270380

theorem contrapositive_a_eq_b_imp_a_sq_eq_b_sq (a b : ℝ) :
  (a = b → a^2 = b^2) ↔ (a^2 ≠ b^2 → a ≠ b) :=
by
  sorry

end contrapositive_a_eq_b_imp_a_sq_eq_b_sq_l270_270380


namespace value_of_f_at_2_l270_270225

-- Given the conditions
variable (f : ℝ → ℝ)
variable (h_mono : Monotone f)
variable (h_cond : ∀ x : ℝ, f (f x - 3^x) = 4)

-- Define the proof goal
theorem value_of_f_at_2 : f 2 = 10 := 
sorry

end value_of_f_at_2_l270_270225


namespace probability_even_sum_between_3_and_15_l270_270113

open Finset

theorem probability_even_sum_between_3_and_15 :
  let S := Icc 3 15
  let pairs := S.product S \ (diagonal S)
  let even_pairs : Finset (ℕ × ℕ) := pairs.filter (λ p, (p.1 + p.2) % 2 = 0)
  let probability : ℚ := even_pairs.card / pairs.card 
  probability = 6 / 13 := 
sorry

end probability_even_sum_between_3_and_15_l270_270113


namespace range_of_a_for_empty_solution_set_l270_270252

theorem range_of_a_for_empty_solution_set : 
  (∀ a : ℝ, (∀ x : ℝ, |x - 4| + |3 - x| < a → false) ↔ a ≤ 1) := 
sorry

end range_of_a_for_empty_solution_set_l270_270252


namespace unique_arrangements_mississippi_l270_270368

open scoped Nat

theorem unique_arrangements_mississippi : 
  let n := 11
  let n_M := 1
  let n_I := 4
  let n_S := 4
  let n_P := 2
  (n.fact / (n_M.fact * n_I.fact * n_S.fact * n_P.fact)) = 34650 :=
  by
  sorry

end unique_arrangements_mississippi_l270_270368


namespace max_sin_a_l270_270434

theorem max_sin_a (a b : ℝ) (h : Real.sin (a + b) = Real.sin a + Real.sin b) : Real.sin a ≤ 1 :=
by
  sorry

end max_sin_a_l270_270434


namespace fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l270_270144

theorem fewer_than_ten_sevens_example1 : (777 / 7) - (77 / 7) = 100 :=
  by sorry

theorem fewer_than_ten_sevens_example2 : (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7) = 100 :=
  by sorry

end fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l270_270144


namespace triangle_area_half_l270_270763

theorem triangle_area_half (AB AC BC : ℝ) (h₁ : AB = 8) (h₂ : AC = BC) (h₃ : AC * AC = AB * AB / 2) (h₄ : AC = BC) : 
  (1 / 2) * (1 / 2 * AB * AB) = 16 :=
  by
  sorry

end triangle_area_half_l270_270763


namespace count_special_integers_l270_270680

theorem count_special_integers : ∃ n : ℕ, n = 2151 ∧ 
  ∀ m : ℕ, (m < 100000 ∧ (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ 
    ((∀ k, k < 5 → m.digit k = d1 ∨ m.digit k = d2) ∨ 
    (∀ k, k < 5 → m.digit k = d1)))) → m ∈ finset.range 100000 :=
by sorry

end count_special_integers_l270_270680


namespace num_integers_two_digits_l270_270695

theorem num_integers_two_digits (n : ℕ) (n < 100000) : 
  ∃ k, k = 2151 ∧ 
  ∀ x, (x < n ∧ (∃ d₁ d₂, ∀ y, y ∈ x.digits 10 → y = d₁ ∨ y = d₂)) → k = 2151 := sorry

end num_integers_two_digits_l270_270695


namespace car_A_catches_up_l270_270356

variables (v_A v_B t_A : ℝ)

-- The conditions of the problem
def distance : ℝ := 300
def time_car_B : ℝ := t_A + 2
def distance_eq_A : Prop := distance = v_A * t_A
def distance_eq_B : Prop := distance = v_B * time_car_B

-- The final proof problem: Car A catches up with car B 150 kilometers away from city B.
theorem car_A_catches_up (t_A > 0) (v_A > 0) (v_B > 0) :
  distance_eq_A ∧ distance_eq_B → 
  ∃ d : ℝ, d = 150 := 
sorry

end car_A_catches_up_l270_270356


namespace complement_of_M_in_U_l270_270727

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - 2*x > 0}
def complement_U_M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem complement_of_M_in_U : (U \ M) = complement_U_M :=
by sorry

end complement_of_M_in_U_l270_270727


namespace factor_expression_l270_270643

theorem factor_expression (x : ℝ) : 72 * x^5 - 90 * x^9 = -18 * x^5 * (5 * x^4 - 4) :=
by
  sorry

end factor_expression_l270_270643


namespace find_square_sum_l270_270089

theorem find_square_sum (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 0) (h5 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 2 / 7 :=
by
  sorry

end find_square_sum_l270_270089


namespace bucket_weight_full_l270_270167

variable (p q x y : ℝ)

theorem bucket_weight_full (h1 : x + (3 / 4) * y = p)
                           (h2 : x + (1 / 3) * y = q) :
  x + y = (1 / 5) * (8 * p - 3 * q) :=
by
  sorry

end bucket_weight_full_l270_270167


namespace triangle_area_sqrt2_div2_find_a_c_l270_270260

  -- Problem 1
  -- Prove the area of triangle ABC is sqrt(2)/2
  theorem triangle_area_sqrt2_div2 {a b c : ℝ} 
    (cond1 : a + (1 / a) = 4 * Real.cos (Real.arccos (a^2 + 1 - c^2) / (2 * a))) 
    (cond2 : b = 1) 
    (cond3 : Real.arcsin (1) = Real.pi / 2) : 
    (1 / 2) * 1 * Real.sqrt 2 = Real.sqrt 2 / 2 := sorry

  -- Problem 2
  -- Prove a = sqrt(7) and c = 2
  theorem find_a_c {a b c : ℝ} 
    (cond1 : a + (1 / a) = 4 * Real.cos (Real.arccos (a^2 + 1 - c^2) / (2 * a))) 
    (cond2 : b = 1) 
    (cond3 : (1 / 2) * a * Real.sin (Real.arcsin (Real.sqrt 3 / a)) = Real.sqrt 3 / 2) : 
    a = Real.sqrt 7 ∧ c = 2 := sorry

  
end triangle_area_sqrt2_div2_find_a_c_l270_270260


namespace alpha_range_theorem_l270_270891

noncomputable def alpha_range (k : ℤ) (α : ℝ) : Prop :=
  2 * k * Real.pi - Real.pi ≤ α ∧ α ≤ 2 * k * Real.pi

theorem alpha_range_theorem (α : ℝ) (k : ℤ) (h : |Real.sin (4 * Real.pi - α)| = Real.sin (Real.pi + α)) :
  alpha_range k α :=
by
  sorry

end alpha_range_theorem_l270_270891


namespace total_amount_l270_270769

-- Definitions directly derived from the conditions in the problem
variable (you_spent friend_spent : ℕ)
variable (h1 : friend_spent = you_spent + 1)
variable (h2 : friend_spent = 8)

-- The goal is to prove that the total amount spent on lunch is $15
theorem total_amount : you_spent + friend_spent = 15 := by
  sorry

end total_amount_l270_270769


namespace students_without_pens_l270_270965

theorem students_without_pens (total_students blue_pens red_pens both_pens : ℕ)
  (h_total : total_students = 40)
  (h_blue : blue_pens = 18)
  (h_red : red_pens = 26)
  (h_both : both_pens = 10) :
  total_students - (blue_pens + red_pens - both_pens) = 6 :=
by
  sorry

end students_without_pens_l270_270965


namespace orthocenter_of_triangle_l270_270903

theorem orthocenter_of_triangle :
  ∀ (A B C H : ℝ × ℝ × ℝ),
    A = (2, 3, 4) → 
    B = (6, 4, 2) → 
    C = (4, 5, 6) → 
    H = (17/53, 152/53, 725/53) → 
    true :=
by sorry

end orthocenter_of_triangle_l270_270903


namespace triangle_area_l270_270789

theorem triangle_area (a b c : ℕ) (h1 : a + b + c = 12) (h2 : a + b > c) (h3 : a + c > b) (h4 : b + c > a) : 
  a = 3 ∧ b = 4 ∧ c = 5 ∨ a = 4 ∧ b = 3 ∧ c = 5 ∨ a = 5 ∧ b = 4 ∧ c = 3 ∨
  a = 5 ∧ b = 3 ∧ c = 4 ∨ a = 4 ∧ b = 5 ∧ c = 3 ∨ a = 3 ∧ b = 5 ∧ c = 4 → 
  (1 / 2 : ℝ) * ↑a * ↑b = 6 := by
  sorry

end triangle_area_l270_270789


namespace grading_ratio_l270_270347

noncomputable def num_questions : ℕ := 100
noncomputable def correct_answers : ℕ := 91
noncomputable def score_received : ℕ := 73
noncomputable def incorrect_answers : ℕ := num_questions - correct_answers
noncomputable def total_points_subtracted : ℕ := correct_answers - score_received
noncomputable def points_per_incorrect : ℚ := total_points_subtracted / incorrect_answers

theorem grading_ratio (h: (points_per_incorrect : ℚ) = 2) :
  2 / 1 = points_per_incorrect / 1 :=
by sorry

end grading_ratio_l270_270347


namespace lucas_change_l270_270101

def initialAmount : ℕ := 20
def costPerAvocado : ℕ := 2
def numberOfAvocados : ℕ := 3

def totalCost : ℕ := numberOfAvocados * costPerAvocado
def change : ℕ := initialAmount - totalCost

theorem lucas_change : change = 14 := by
  sorry

end lucas_change_l270_270101


namespace total_weeds_correct_l270_270580

def tuesday : ℕ := 25
def wednesday : ℕ := 3 * tuesday
def thursday : ℕ := wednesday / 5
def friday : ℕ := thursday - 10
def total_weeds : ℕ := tuesday + wednesday + thursday + friday

theorem total_weeds_correct : total_weeds = 120 :=
by
  sorry

end total_weeds_correct_l270_270580


namespace value_of_A_l270_270774

theorem value_of_A (A B C D : ℕ) (h1 : A * B = 60) (h2 : C * D = 60) (h3 : A - B = C + D) (h4 : A ≠ B) (h5 : A ≠ C) (h6 : A ≠ D) (h7 : B ≠ C) (h8 : B ≠ D) (h9 : C ≠ D) : A = 20 :=
by sorry

end value_of_A_l270_270774


namespace yolkino_to_palkino_distance_l270_270293

theorem yolkino_to_palkino_distance 
  (n : ℕ) 
  (digit_sum : ℕ → ℕ) 
  (h1 : ∀ k : ℕ, k ≤ n → digit_sum k + digit_sum (n - k) = 13) : 
  n = 49 := 
by 
  sorry

end yolkino_to_palkino_distance_l270_270293


namespace range_of_a_l270_270251

def solution_set_non_empty (a : ℝ) : Prop :=
  ∃ x : ℝ, |x - 3| + |x - 4| < a

theorem range_of_a (a : ℝ) : solution_set_non_empty a ↔ a > 1 := sorry

end range_of_a_l270_270251


namespace period_pi_omega_l270_270232

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ :=
  3 * (Real.sin (ω * x)) * (Real.cos (ω * x)) - 4 * (Real.cos (ω * x))^2

theorem period_pi_omega (ω : ℝ) (hω : ω > 0) (period_condition : ∀ x, f x ω = f (x + π) ω)
  (theta : ℝ) (h_f_theta : f theta ω = 1 / 2) :
  f (theta + π / 2) ω + f (theta - π / 4) ω = -13 / 2 :=
by
  sorry

end period_pi_omega_l270_270232


namespace factorization_of_difference_of_squares_l270_270851

theorem factorization_of_difference_of_squares (m : ℝ) : 
  m^2 - 16 = (m + 4) * (m - 4) := 
by 
  sorry

end factorization_of_difference_of_squares_l270_270851


namespace Vann_total_teeth_cleaned_l270_270326

theorem Vann_total_teeth_cleaned :
  let dogs := 7
  let cats := 12
  let pigs := 9
  let horses := 4
  let rabbits := 15
  let dogs_teeth := 42
  let cats_teeth := 30
  let pigs_teeth := 44
  let horses_teeth := 40
  let rabbits_teeth := 28
  (dogs * dogs_teeth) + (cats * cats_teeth) + (pigs * pigs_teeth) + (horses * horses_teeth) + (rabbits * rabbits_teeth) = 1630 :=
by
  sorry

end Vann_total_teeth_cleaned_l270_270326


namespace sum_of_three_digits_eq_nine_l270_270449

def horizontal_segments (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | 1 => 0
  | 2 => 2
  | 3 => 3
  | 4 => 1
  | 5 => 2
  | 6 => 1
  | 7 => 1
  | 8 => 3
  | 9 => 2
  | _ => 0  -- Invalid digit

def vertical_segments (n : ℕ) : ℕ :=
  match n with
  | 0 => 4
  | 1 => 2
  | 2 => 3
  | 3 => 3
  | 4 => 3
  | 5 => 2
  | 6 => 3
  | 7 => 2
  | 8 => 4
  | 9 => 3
  | _ => 0  -- Invalid digit

theorem sum_of_three_digits_eq_nine :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
             (horizontal_segments a + horizontal_segments b + horizontal_segments c = 5) ∧ 
             (vertical_segments a + vertical_segments b + vertical_segments c = 10) ∧
             (a + b + c = 9) :=
sorry

end sum_of_three_digits_eq_nine_l270_270449


namespace percent_voters_for_candidate_A_l270_270261

theorem percent_voters_for_candidate_A (d r i u p_d p_r p_i p_u : ℝ) 
  (hd : d = 0.45) (hr : r = 0.30) (hi : i = 0.20) (hu : u = 0.05)
  (hp_d : p_d = 0.75) (hp_r : p_r = 0.25) (hp_i : p_i = 0.50) (hp_u : p_u = 0.50) :
  d * p_d + r * p_r + i * p_i + u * p_u = 0.5375 :=
by
  sorry

end percent_voters_for_candidate_A_l270_270261


namespace result_of_subtraction_l270_270995

theorem result_of_subtraction (N : ℝ) (h1 : N = 100) : 0.80 * N - 20 = 60 :=
by
  sorry

end result_of_subtraction_l270_270995


namespace k_value_correct_l270_270886

theorem k_value_correct (k : ℚ) : 
  let f (x : ℚ) := 4 * x^2 - 3 * x + 5
  let g (x : ℚ) := x^2 + k * x - 8
  (f 5 - g 5 = 20) -> k = 53 / 5 :=
by
  intro h
  sorry

end k_value_correct_l270_270886


namespace inequality_not_hold_l270_270215

theorem inequality_not_hold (a b : ℝ) (h : a < b ∧ b < 0) : (1 / (a - b) < 1 / a) :=
by
  sorry

end inequality_not_hold_l270_270215


namespace isabella_paintable_area_l270_270717

def total_paintable_area : ℕ :=
  let room1_area := 2 * (14 * 9) + 2 * (12 * 9) - 70
  let room2_area := 2 * (13 * 9) + 2 * (11 * 9) - 70
  let room3_area := 2 * (15 * 9) + 2 * (10 * 9) - 70
  let room4_area := 4 * (12 * 9) - 70
  room1_area + room2_area + room3_area + room4_area

theorem isabella_paintable_area : total_paintable_area = 1502 := by
  sorry

end isabella_paintable_area_l270_270717


namespace count_positive_integers_with_two_digits_l270_270676

theorem count_positive_integers_with_two_digits : 
  ∃ (count : ℕ), count = 2151 ∧ ∀ n, n < 100000 → (∀ d1 d2 d, d ∈ digits n → d = d1 ∨ d = d2) → n ∈ (range 100000) := sorry

end count_positive_integers_with_two_digits_l270_270676


namespace binary_to_decimal_1010101_l270_270510

def bin_to_dec (bin : List ℕ) (len : ℕ): ℕ :=
  List.foldl (λ acc (digit, idx) => acc + digit * 2^idx) 0 (List.zip bin (List.range len))

theorem binary_to_decimal_1010101 : bin_to_dec [1, 0, 1, 0, 1, 0, 1] 7 = 85 :=
by
  simp [bin_to_dec, List.range, List.zip]
  -- Detailed computation can be omitted and sorry used here if necessary
  sorry

end binary_to_decimal_1010101_l270_270510


namespace sum_of_squares_l270_270733

theorem sum_of_squares (a b c d : ℝ) (h1 : b = a + 1) (h2 : c = a + 2) (h3 : d = a + 3) :
  a^2 + b^2 = c^2 + d^2 := by
  sorry

end sum_of_squares_l270_270733


namespace roots_geom_prog_eq_neg_cbrt_c_l270_270739

theorem roots_geom_prog_eq_neg_cbrt_c {a b c : ℝ} (h : ∀ (x1 x2 x3 : ℝ), 
  (x1^3 + a * x1^2 + b * x1 + c = 0) ∧ (x2^3 + a * x2^2 + b * x2 + c = 0) ∧ (x3^3 + a * x3^2 + b * x3 + c = 0) ∧ 
  (∃ (r : ℝ), (x2 = r * x1) ∧ (x3 = r^2 * x1))) : 
  ∃ (x : ℝ), (x^3 = c) ∧ (x = - ((c) ^ (1/3))) :=
by 
  sorry

end roots_geom_prog_eq_neg_cbrt_c_l270_270739


namespace only_B_forms_triangle_l270_270610

/-- Check if a set of line segments can form a triangle --/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem only_B_forms_triangle :
  ¬ can_form_triangle 2 6 3 ∧
  can_form_triangle 6 7 8 ∧
  ¬ can_form_triangle 1 7 9 ∧
  ¬ can_form_triangle (3 / 2) 4 (5 / 2) :=
by
  sorry

end only_B_forms_triangle_l270_270610


namespace max_unique_coin_sums_l270_270341

def coin_values : List ℕ := [1, 1, 1, 5, 5, 10, 10, 50]

def possible_sums : Finset ℕ := (Finset.filter (λ x, x ≠ 0)
 (Finset.map (Function.uncurry (+))
  (Finset.product (Finset.fromList coin_values) (Finset.fromList coin_values))))

theorem max_unique_coin_sums : possible_sums.card = 9 := by sorry

end max_unique_coin_sums_l270_270341


namespace smallest_three_digit_number_with_property_l270_270751

theorem smallest_three_digit_number_with_property : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ 
  (∀ d, (1 ≤ d ∧ d ≤ 1000) → ((d = n + 1 ∨ d = n - 1) → d % 11 = 0)) ∧ 
  n = 120 :=
by
  sorry

end smallest_three_digit_number_with_property_l270_270751


namespace repeating_decimal_difference_l270_270823

theorem repeating_decimal_difference :
  let x := 72 / 99 in
  let y := 72 / 100 in
  x - y = 2 / 275 := 
by
  -- we will add a proof later
  sorry

end repeating_decimal_difference_l270_270823


namespace num_integers_two_digits_l270_270694

theorem num_integers_two_digits (n : ℕ) (n < 100000) : 
  ∃ k, k = 2151 ∧ 
  ∀ x, (x < n ∧ (∃ d₁ d₂, ∀ y, y ∈ x.digits 10 → y = d₁ ∨ y = d₂)) → k = 2151 := sorry

end num_integers_two_digits_l270_270694


namespace relation_y1_y2_y3_l270_270664

-- Definition of being on the parabola
def on_parabola (x : ℝ) (y m : ℝ) : Prop := y = -3*x^2 - 12*x + m

-- The conditions given in the problem
variables {y1 y2 y3 m : ℝ}

-- The points (-3, y1), (-2, y2), (1, y3) are on the parabola given by the equation
axiom h1 : on_parabola (-3) y1 m
axiom h2 : on_parabola (-2) y2 m
axiom h3 : on_parabola (1) y3 m

-- We need to prove the relationship between y1, y2, and y3
theorem relation_y1_y2_y3 : y2 > y1 ∧ y1 > y3 :=
by { sorry }

end relation_y1_y2_y3_l270_270664


namespace Justin_run_home_time_l270_270724

variable (blocksPerMinute : ℝ) (totalBlocks : ℝ)

theorem Justin_run_home_time (h1 : blocksPerMinute = 2 / 1.5) (h2 : totalBlocks = 8) :
  totalBlocks / blocksPerMinute = 6 := by
  sorry

end Justin_run_home_time_l270_270724


namespace min_distance_point_curve_to_line_l270_270412

noncomputable def curve (x : ℝ) : ℝ := x^2 - Real.log x

def line (x y : ℝ) : Prop := x - y - 2 = 0

theorem min_distance_point_curve_to_line :
  ∀ (P : ℝ × ℝ), 
  curve P.1 = P.2 →
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 :=
by
  sorry

end min_distance_point_curve_to_line_l270_270412


namespace find_x_l270_270979

theorem find_x (p q r s x : ℚ) (hpq : p ≠ q) (hq0 : q ≠ 0) 
    (h : (p + x) / (q - x) = r / s) 
    (hp : p = 3) (hq : q = 5) (hr : r = 7) (hs : s = 9) : 
    x = 1/2 :=
by {
  sorry
}

end find_x_l270_270979


namespace squared_expression_l270_270482

variable {x y : ℝ}

theorem squared_expression (x y : ℝ) : (-3 * x^2 * y)^2 = 9 * x^4 * y^2 :=
  by
  sorry

end squared_expression_l270_270482


namespace atleast_one_alarm_rings_on_time_l270_270097

def probability_alarm_A_rings := 0.80
def probability_alarm_B_rings := 0.90

def probability_atleast_one_rings := 1 - (1 - probability_alarm_A_rings) * (1 - probability_alarm_B_rings)

theorem atleast_one_alarm_rings_on_time :
  probability_atleast_one_rings = 0.98 :=
sorry

end atleast_one_alarm_rings_on_time_l270_270097


namespace part_a_part_b_l270_270036

/-- Definition of the sequence of numbers on the cards -/
def card_numbers (n : ℕ) : ℕ :=
  if n = 0 then 1 else (10^(n + 1) - 1) / 9 * 2 + 1

/-- Part (a) statement: Is it possible to choose at least three cards such that 
the sum of the numbers on them equals a number where all digits except one are twos? -/
theorem part_a : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ card_numbers a + card_numbers b + card_numbers c % 10 = 2 ∧ 
  (∀ d, ∃ k ≤ 1, (card_numbers a + card_numbers b + card_numbers c / (10^d)) % 10 = 2) :=
sorry

/-- Part (b) statement: Suppose several cards were chosen such that the sum of the numbers 
on them equals a number where all digits except one are twos. What could be the digit that is not two? -/
theorem part_b (sum : ℕ) :
  (∀ d, sum / (10^d) % 10 = 2) → ((sum % 10 = 0) ∨ (sum % 10 = 1)) :=
sorry

end part_a_part_b_l270_270036


namespace min_value_seq_ratio_l270_270529

-- Define the sequence {a_n} based on the given recurrence relation and initial condition
def seq (n : ℕ) : ℕ := 
  if n = 0 then 0 -- Handling the case when n is 0, though sequence starts from n=1
  else n^2 - n + 15

-- Prove the minimum value of (a_n / n) is 27/4
theorem min_value_seq_ratio : 
  ∃ n : ℕ, n > 0 ∧ seq n / n = 27 / 4 :=
by
  sorry

end min_value_seq_ratio_l270_270529


namespace evaluate_at_3_l270_270384

def f (x : ℕ) : ℕ := x ^ 2

theorem evaluate_at_3 : f 3 = 9 :=
by
  sorry

end evaluate_at_3_l270_270384


namespace jellybean_removal_l270_270132

theorem jellybean_removal 
    (initial_count : ℕ) 
    (first_removal : ℕ) 
    (added_back : ℕ) 
    (final_count : ℕ)
    (initial_count_eq : initial_count = 37)
    (first_removal_eq : first_removal = 15)
    (added_back_eq : added_back = 5)
    (final_count_eq : final_count = 23) :
    (initial_count - first_removal + added_back - final_count) = 4 :=
by 
    sorry

end jellybean_removal_l270_270132


namespace abc_zero_iff_quadratic_identities_l270_270725

variable {a b c : ℝ}

theorem abc_zero_iff_quadratic_identities (h : ¬(a = b ∧ b = c ∧ c = a)) : 
  a + b + c = 0 ↔ a^2 + ab + b^2 = b^2 + bc + c^2 ∧ b^2 + bc + c^2 = c^2 + ca + a^2 :=
by
  sorry

end abc_zero_iff_quadratic_identities_l270_270725


namespace gcd_n_cube_plus_16_n_plus_4_l270_270210

theorem gcd_n_cube_plus_16_n_plus_4 (n : ℕ) (h1 : n > 16) : 
  Nat.gcd (n^3 + 16) (n + 4) = Nat.gcd 48 (n + 4) :=
by
  sorry

end gcd_n_cube_plus_16_n_plus_4_l270_270210


namespace find_p1_plus_q1_l270_270509

noncomputable def p (x : ℤ) := x^4 + 14 * x^2 + 1
noncomputable def q (x : ℤ) := x^4 - 14 * x^2 + 1

theorem find_p1_plus_q1 :
  (p 1) + (q 1) = 4 :=
sorry

end find_p1_plus_q1_l270_270509


namespace most_probable_light_is_green_l270_270353

def duration_red := 30
def duration_yellow := 5
def duration_green := 40
def total_duration := duration_red + duration_yellow + duration_green

def prob_red := duration_red / total_duration
def prob_yellow := duration_yellow / total_duration
def prob_green := duration_green / total_duration

theorem most_probable_light_is_green : prob_green > prob_red ∧ prob_green > prob_yellow := 
  by
  sorry

end most_probable_light_is_green_l270_270353


namespace vacation_books_l270_270405

-- Define the number of mystery, fantasy, and biography novels.
def num_mystery : ℕ := 3
def num_fantasy : ℕ := 4
def num_biography : ℕ := 3

-- Define the condition that we want to choose three books with no more than one from each genre.
def num_books_to_choose : ℕ := 3
def max_books_per_genre : ℕ := 1

-- The number of ways to choose one book from each genre
def num_combinations (m f b : ℕ) : ℕ :=
  m * f * b

-- Prove that the number of possible sets of books is 36
theorem vacation_books : num_combinations num_mystery num_fantasy num_biography = 36 := by
  sorry

end vacation_books_l270_270405


namespace number_of_positive_integers_with_at_most_two_digits_l270_270692

theorem number_of_positive_integers_with_at_most_two_digits:
  -- There are 2151 positive integers less than 100000 with at most two different digits
  ∃ (n : ℕ), n = 2151 ∧
    (∀ k : ℕ, k < 100000 → 
      let digits := (nat.digits 10 k).erase_dup 
      in (1 ≤ digits.length ∧ digits.length ≤ 2) → 1 ≤ k ∧ k < 100000) :=
begin
  -- proof would go here
  sorry
end

end number_of_positive_integers_with_at_most_two_digits_l270_270692


namespace lines_intersection_l270_270675

def intersection_point_of_lines
  (t u : ℚ)
  (x₁ y₁ x₂ y₂ : ℚ)
  (x y : ℚ) : Prop := 
  ∃ (t u : ℚ),
    (x₁ + 3*t = 7 + 6*u) ∧
    (y₁ - 4*t = -5 + 3*u) ∧
    (x = x₁ + 3 * t) ∧ 
    (y = y₁ - 4 * t)

theorem lines_intersection :
  ∀ (t u : ℚ),
    intersection_point_of_lines t u 3 2 7 (-5) (87/11) (-50/11) :=
by
  sorry

end lines_intersection_l270_270675


namespace Integers_and_fractions_are_rational_numbers_l270_270475

-- Definitions from conditions
def is_fraction (x : ℚ) : Prop :=
  ∃a b : ℤ, b ≠ 0 ∧ x = (a : ℚ) / (b : ℚ)

def is_integer (x : ℤ) : Prop := 
  ∃n : ℤ, x = n

def is_rational (x : ℚ) : Prop := 
  ∃a b : ℤ, b ≠ 0 ∧ x = (a : ℚ) / (b : ℚ)

-- The statement to be proven
theorem Integers_and_fractions_are_rational_numbers (x : ℚ) : 
  (∃n : ℤ, x = (n : ℚ)) ∨ is_fraction x ↔ is_rational x :=
by sorry

end Integers_and_fractions_are_rational_numbers_l270_270475


namespace order_of_abc_l270_270063

noncomputable def a : ℝ := (0.3)^3
noncomputable def b : ℝ := (3)^3
noncomputable def c : ℝ := Real.log 0.3 / Real.log 3

theorem order_of_abc : b > a ∧ a > c :=
by
  have ha : a = (0.3)^3 := rfl
  have hb : b = (3)^3 := rfl
  have hc : c = Real.log 0.3 / Real.log 3 := rfl
  sorry

end order_of_abc_l270_270063


namespace expected_value_sum_until_6_l270_270381

noncomputable def die_probability_6 := 3 / 8
noncomputable def die_probability_4 := 1 / 4
noncomputable def die_probability_other := 1 / 20

noncomputable def expected_single_roll_value : ℝ :=
  (1 / 20 ) * 1 + (1 / 20 ) * 2 + (1 / 20 ) * 3 + (1 / 4 ) * 4 + (1 / 20 ) * 5 + (3 / 8 ) * 6

noncomputable def expected_rolls_until_6 : ℝ :=
  1 / die_probability_6

noncomputable def expected_sum_until_6 : ℝ :=
  expected_rolls_until_6 * expected_single_roll_value

theorem expected_value_sum_until_6 :
  expected_sum_until_6 = 9.4 :=
by
  sorry

end expected_value_sum_until_6_l270_270381


namespace reciprocal_of_neg_2023_l270_270456

theorem reciprocal_of_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l270_270456


namespace distance_focus_directrix_l270_270119

theorem distance_focus_directrix (y x : ℝ) (h : y^2 = 2 * x) : x = 1 := 
by 
  sorry

end distance_focus_directrix_l270_270119


namespace volume_of_box_l270_270623

-- Defining the initial parameters of the problem
def length_sheet := 48
def width_sheet := 36
def side_length_cut_square := 3

-- Define the transformed dimensions after squares are cut off
def length_box := length_sheet - 2 * side_length_cut_square
def width_box := width_sheet - 2 * side_length_cut_square
def height_box := side_length_cut_square

-- The target volume of the box
def target_volume := 3780

-- Prove that the volume of the box is equal to the target volume
theorem volume_of_box : length_box * width_box * height_box = target_volume := by
  -- Calculate the expected volume
  -- Expected volume = 42 m * 30 m * 3 m
  -- Which equals 3780 m³
  sorry

end volume_of_box_l270_270623


namespace int_solution_for_system_l270_270645

noncomputable def log6 : ℝ → ℝ := λ x, Real.log x / Real.log 6

theorem int_solution_for_system :
  ∃ x y : ℤ, (x ^ (x - 2 * y) = 36) ∧ (4 * (x - 2 * y) + Real.log x / Real.log 6 = 9) ∧ (x, y) = (6, 2) := 
by
  sorry

end int_solution_for_system_l270_270645


namespace find_side_b_l270_270259

variable {a b c : ℝ} -- sides of the triangle
variable {A B C : ℝ} -- angles of the triangle
variable {area : ℝ}

axiom sides_form_arithmetic_sequence : 2 * b = a + c
axiom angle_B_is_60_degrees : B = Real.pi / 3
axiom area_is_3sqrt3 : area = 3 * Real.sqrt 3
axiom area_formula : area = 1 / 2 * a * c * Real.sin (B)

theorem find_side_b : b = 2 * Real.sqrt 3 := by
  sorry

end find_side_b_l270_270259


namespace greatest_num_of_coins_l270_270198

-- Define the total amount of money Carlos has in U.S. coins.
def total_value : ℝ := 5.45

-- Define the value of each type of coin.
def quarter_value : ℝ := 0.25
def dime_value : ℝ := 0.10
def nickel_value : ℝ := 0.05

-- Define the number of quarters, dimes, and nickels Carlos has.
def num_coins (q : ℕ) := quarter_value * q + dime_value * q + nickel_value * q

-- The main theorem: Carlos can have at most 13 quarters, dimes, and nickels.
theorem greatest_num_of_coins (q : ℕ) :
  num_coins q = total_value → q ≤ 13 :=
sorry

end greatest_num_of_coins_l270_270198


namespace min_value_sin6_cos6_l270_270028

open Real

theorem min_value_sin6_cos6 (x : ℝ) : sin x ^ 6 + 2 * cos x ^ 6 ≥ 2 / 3 :=
by
  sorry

end min_value_sin6_cos6_l270_270028


namespace money_left_after_distributions_and_donations_l270_270784

theorem money_left_after_distributions_and_donations 
  (total_income : ℕ)
  (percent_to_children : ℕ)
  (percent_to_each_child : ℕ)
  (number_of_children : ℕ)
  (percent_to_wife : ℕ)
  (percent_to_orphan_house : ℕ)
  (remaining_income_percentage : ℕ)
  (children_distribution : ℕ → ℕ → ℕ)
  (wife_distribution : ℕ → ℕ)
  (calculate_remaining : ℕ → ℕ → ℕ)
  (calculate_donation : ℕ → ℕ → ℕ)
  (calculate_money_left : ℕ → ℕ → ℕ)
  (income : ℕ := 400000)
  (result : ℕ := 57000) :
  children_distribution percent_to_each_child number_of_children = 60 →
  percent_to_wife = 25 →
  remaining_income_percentage = 15 →
  percent_to_orphan_house = 5 →
  wife_distribution percent_to_wife = 100000 →
  calculate_remaining 100 85 = 15 →
  calculate_donation percent_to_orphan_house (calculate_remaining 100 85 * total_income) = 3000 →
  calculate_money_left (calculate_remaining 100 85 * total_income) 3000 = result →
  total_income = income →
  income - (60 * income / 100 + 25 * income / 100 + 5 * (15 * income / 100) / 100) = result
  :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end money_left_after_distributions_and_donations_l270_270784


namespace compare_a_b_c_l270_270867

noncomputable def a : ℝ := 0.1 * Real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b :=
by
  have : c < b := sorry,
  have : a < b := sorry,
  have : a > c := sorry,
  exact ⟨this.2, this.1⟩

end compare_a_b_c_l270_270867


namespace parabola_intersects_x_axis_l270_270669

theorem parabola_intersects_x_axis {p q x₀ x₁ x₂ : ℝ} (h : ∀ (x : ℝ), x ^ 2 + p * x + q ≠ 0)
    (M_below_x_axis : x₀ ^ 2 + p * x₀ + q < 0)
    (M_at_1_neg2 : x₀ = 1 ∧ (1 ^ 2 + p * 1 + q = -2)) :
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₀ < x₁ → x₁ < x₂) ∧ x₁ = -1 ∧ x₂ = 2 ∨ x₁ = 0 ∧ x₂ = 3) :=
by
  sorry

end parabola_intersects_x_axis_l270_270669


namespace half_radius_of_circle_y_l270_270984

theorem half_radius_of_circle_y 
  (r_x r_y : ℝ) 
  (h₁ : π * r_x^2 = π * r_y^2) 
  (h₂ : 2 * π * r_x = 14 * π) :
  r_y / 2 = 3.5 :=
by {
  sorry
}

end half_radius_of_circle_y_l270_270984


namespace min_value_expression_l270_270608

theorem min_value_expression :
  ∃ x : ℝ, (x+2) * (x+3) * (x+5) * (x+6) + 2024 = 2021.75 :=
sorry

end min_value_expression_l270_270608


namespace relationship_between_x_and_y_l270_270246

theorem relationship_between_x_and_y
  (x y : ℝ)
  (h1 : 2 * x - 3 * y > 6 * x)
  (h2 : 3 * x - 4 * y < 2 * y - x) :
  x < y ∧ x < 0 ∧ y < 0 :=
sorry

end relationship_between_x_and_y_l270_270246


namespace general_term_a_general_term_b_sum_first_n_terms_l270_270057

def a : Nat → Nat
| 0     => 1
| (n+1) => 2 * a n

def b (n : Nat) : Int :=
  3 * (n + 1) - 2

def S (n : Nat) : Int :=
  2^n - (3 * n^2) / 2 + n / 2 - 1

-- We state the theorems with the conditions included.

theorem general_term_a (n : Nat) : a n = 2^(n - 1) := by
  sorry

theorem general_term_b (n : Nat) : b n = 3 * (n + 1) - 2 := by
  sorry

theorem sum_first_n_terms (n : Nat) : 
  (Finset.range n).sum (λ i => a i - b i) = 2^n - (3 * n^2) / 2 + n / 2 - 1 := by
  sorry

end general_term_a_general_term_b_sum_first_n_terms_l270_270057


namespace continuity_at_x_0_l270_270991

def f (x : ℝ) := -2 * x^2 + 9
def x_0 : ℝ := 4

theorem continuity_at_x_0 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x_0| < δ → |f x - f x_0| < ε :=
by
  sorry

end continuity_at_x_0_l270_270991


namespace square_area_of_equal_perimeter_l270_270351

theorem square_area_of_equal_perimeter 
  (side_length_triangle : ℕ) (side_length_square : ℕ) (perimeter_square : ℕ)
  (h1 : side_length_triangle = 20)
  (h2 : perimeter_square = 3 * side_length_triangle)
  (h3 : 4 * side_length_square = perimeter_square) :
  side_length_square ^ 2 = 225 := 
by
  sorry

end square_area_of_equal_perimeter_l270_270351


namespace solution_of_fraction_l270_270415

theorem solution_of_fraction (x : ℝ) (h1 : x^2 - 9 = 0) (h2 : x + 3 ≠ 0) : x = 3 :=
by
  sorry

end solution_of_fraction_l270_270415


namespace length_of_tangent_l270_270200

/-- 
Let O and O1 be the centers of the larger and smaller circles respectively with radii 8 and 3. 
The circles touch each other internally. Let A be the point of tangency and OM be the tangent from center O to the smaller circle. 
Prove that the length of this tangent is 4.
--/
theorem length_of_tangent {O O1 : Type} (radius_large : ℝ) (radius_small : ℝ) (OO1 : ℝ) 
  (OM O1M : ℝ) (h : 8 - 3 = 5) (h1 : OO1 = 5) (h2 : O1M = 3): OM = 4 :=
by
  sorry

end length_of_tangent_l270_270200


namespace repeating_seventy_two_exceeds_seventy_two_l270_270830

noncomputable def repeating_decimal (n d : ℕ) : ℚ := n / d

theorem repeating_seventy_two_exceeds_seventy_two :
  repeating_decimal 72 99 - (72 / 100) = (2 / 275) := 
sorry

end repeating_seventy_two_exceeds_seventy_two_l270_270830


namespace number_of_paths_l270_270636

-- Define the coordinates and the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

def E := (0, 7)
def F := (4, 5)
def G := (9, 0)

-- Define the number of steps required for each path segment
def steps_to_F := 6
def steps_to_G := 10

-- Capture binomial coefficients for the calculated path segments
def paths_E_to_F := binomial steps_to_F 4
def paths_F_to_G := binomial steps_to_G 5

-- Prove the total number of paths from E to G through F
theorem number_of_paths : paths_E_to_F * paths_F_to_G = 3780 :=
by rw [paths_E_to_F, paths_F_to_G]; sorry

end number_of_paths_l270_270636


namespace samantha_total_cost_l270_270939

noncomputable def daily_rental_rate : ℝ := 30
noncomputable def daily_rental_days : ℝ := 3
noncomputable def cost_per_mile : ℝ := 0.15
noncomputable def miles_driven : ℝ := 500

theorem samantha_total_cost :
  (daily_rental_rate * daily_rental_days) + (cost_per_mile * miles_driven) = 165 :=
by
  sorry

end samantha_total_cost_l270_270939


namespace sum_reciprocal_inequality_l270_270041

theorem sum_reciprocal_inequality (p q a b c d e : ℝ) (hp : 0 < p) (ha : p ≤ a) (hb : p ≤ b) (hc : p ≤ c) (hd : p ≤ d) (he : p ≤ e) (haq : a ≤ q) (hbq : b ≤ q) (hcq : c ≤ q) (hdq : d ≤ q) (heq : e ≤ q) :
  (a + b + c + d + e) * (1 / a + 1 / b + 1 / c + 1 / d + 1 / e) ≤ 25 + 6 * ((Real.sqrt (q / p) - Real.sqrt (p / q)) ^ 2) :=
by sorry

end sum_reciprocal_inequality_l270_270041


namespace impossible_to_achieve_25_percent_grape_juice_l270_270604

theorem impossible_to_achieve_25_percent_grape_juice (x y : ℝ) 
  (h1 : ∀ a b : ℝ, (8 / (8 + 32) = 2 / 10) → (6 / (6 + 24) = 2 / 10))
  (h2 : (8 * x + 6 * y) / (40 * x + 30 * y) = 1 / 4) : false :=
by
  sorry

end impossible_to_achieve_25_percent_grape_juice_l270_270604


namespace non_neg_solutions_l270_270208

theorem non_neg_solutions (x y z : ℕ) :
  (x^3 = 2 * y^2 - z) →
  (y^3 = 2 * z^2 - x) →
  (z^3 = 2 * x^2 - y) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by {
  sorry
}

end non_neg_solutions_l270_270208


namespace opposite_of_negative_six_is_six_l270_270594

theorem opposite_of_negative_six_is_six : ∀ (x : ℤ), (-6 + x = 0) → x = 6 :=
by
  intro x hx
  sorry

end opposite_of_negative_six_is_six_l270_270594


namespace gcd_decomposition_l270_270518

open Polynomial

noncomputable def f : Polynomial ℚ := 4 * X ^ 4 - 2 * X ^ 3 - 16 * X ^ 2 + 5 * X + 9
noncomputable def g : Polynomial ℚ := 2 * X ^ 3 - X ^ 2 - 5 * X + 4

theorem gcd_decomposition :
  ∃ (u v : Polynomial ℚ), u * f + v * g = X - 1 :=
sorry

end gcd_decomposition_l270_270518


namespace focus_of_parabola_l270_270363

theorem focus_of_parabola :
  (∀ y : ℝ, x = (1 / 4) * y^2) → (focus = (-1, 0)) := by
  sorry

end focus_of_parabola_l270_270363


namespace example_one_example_two_l270_270162

-- We will define natural numbers corresponding to seven, seventy-seven, and seven hundred seventy-seven.
def seven : ℕ := 7
def seventy_seven : ℕ := 77
def seven_hundred_seventy_seven : ℕ := 777

-- We will define both solutions in the form of equalities producing 100.
theorem example_one : (seven_hundred_seventy_seven / seven) - (seventy_seven / seven) = 100 :=
  by sorry

theorem example_two : (seven * seven) + (seven * seven) + (seven / seven) + (seven / seven) = 100 :=
  by sorry

end example_one_example_two_l270_270162


namespace factory_output_decrease_l270_270987

noncomputable def original_output (O : ℝ) : ℝ :=
  O

noncomputable def increased_output_10_percent (O : ℝ) : ℝ :=
  O * 1.1

noncomputable def increased_output_30_percent (O : ℝ) : ℝ :=
  increased_output_10_percent O * 1.3

noncomputable def percentage_decrease_needed (original new_output : ℝ) : ℝ :=
  ((new_output - original) / new_output) * 100

theorem factory_output_decrease (O : ℝ) : 
  abs (percentage_decrease_needed (original_output O) (increased_output_30_percent O) - 30.07) < 0.01 :=
by
  sorry

end factory_output_decrease_l270_270987


namespace tank_emptying_time_l270_270188

theorem tank_emptying_time
  (initial_volume : ℝ)
  (filling_rate : ℝ)
  (emptying_rate : ℝ)
  (initial_fraction_full : initial_volume = 1 / 5)
  (pipe_a_rate : filling_rate = 1 / 10)
  (pipe_b_rate : emptying_rate = 1 / 6) :
  (initial_volume / (filling_rate - emptying_rate) = 3) :=
by
  sorry

end tank_emptying_time_l270_270188


namespace distribution_count_l270_270203

def num_distributions (novels poetry students : ℕ) : ℕ :=
  -- This is where the formula for counting would go, but we'll just define it as sorry for now
  sorry

theorem distribution_count : num_distributions 3 2 4 = 28 :=
by
  sorry

end distribution_count_l270_270203


namespace tray_contains_40_brownies_l270_270178

-- Definitions based on conditions
def tray_length : ℝ := 24
def tray_width : ℝ := 15
def brownie_length : ℝ := 3
def brownie_width : ℝ := 3

-- The mathematical statement to prove
theorem tray_contains_40_brownies :
  (tray_length * tray_width) / (brownie_length * brownie_width) = 40 :=
by
  sorry

end tray_contains_40_brownies_l270_270178


namespace power_function_properties_l270_270522

theorem power_function_properties (α : ℝ) (h : (3 : ℝ) ^ α = 27) :
  (α = 3) →
  (∀ x : ℝ, (x ^ α) = x ^ 3) ∧
  (∀ x : ℝ, x ^ α = -(((-x) ^ α))) ∧
  (∀ x y : ℝ, x < y → x ^ α < y ^ α) ∧
  (∀ y : ℝ, ∃ x : ℝ, x ^ α = y) :=
by
  sorry

end power_function_properties_l270_270522


namespace repeating_decimal_exceeds_finite_decimal_by_l270_270809

-- Definitions based on the problem conditions
def repeating_decimal := 8 / 11
def finite_decimal := 18 / 25

-- Statement to be proved
theorem repeating_decimal_exceeds_finite_decimal_by : 
  repeating_decimal - finite_decimal = 2 / 275 :=
by
  -- Skipping the proof itself with 'sorry'
  sorry

end repeating_decimal_exceeds_finite_decimal_by_l270_270809


namespace fraction_difference_is_correct_l270_270797

-- Convert repeating decimal to fraction
def repeating_to_fraction : ℚ := 0.72 + 72 / 9900

-- Convert 0.72 to fraction
def decimal_to_fraction : ℚ := 72 / 100

-- Define the difference between the fractions
def fraction_difference := repeating_to_fraction - decimal_to_fraction

-- Final statement to prove
theorem fraction_difference_is_correct : fraction_difference = 2 / 275 := by
  sorry

end fraction_difference_is_correct_l270_270797


namespace musketeer_statements_triplets_count_l270_270270

-- Definitions based on the conditions
def musketeers : Type := { x : ℕ // x < 3 }

def is_guilty (m : musketeers) : Prop := sorry  -- Placeholder for the property of being guilty

def statement (m1 m2 : musketeers) : Prop := sorry  -- Placeholder for the statement made by one musketeer about another

-- Condition that each musketeer makes one statement
def made_statement (m : musketeers) : Prop := sorry

-- Condition that exactly one musketeer lied
def exactly_one_lied : Prop := sorry

-- The final proof problem statement:
theorem musketeer_statements_triplets_count : ∃ n : ℕ, n = 99 :=
  sorry

end musketeer_statements_triplets_count_l270_270270


namespace find_m_for_even_function_l270_270401

def quadratic_function (m : ℝ) (x : ℝ) : ℝ :=
  m * x^2 + (m + 2) * m * x + 2

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem find_m_for_even_function :
  ∃ m : ℝ, is_even_function (quadratic_function m) ∧ m = -2 :=
by
  sorry

end find_m_for_even_function_l270_270401


namespace triangle_enlargement_invariant_l270_270297

theorem triangle_enlargement_invariant (α β γ : ℝ) (h_sum : α + β + γ = 180) (f : ℝ) :
  (α * f ≠ α) ∧ (β * f ≠ β) ∧ (γ * f ≠ γ) → (α * f + β * f + γ * f = 180 * f) → α + β + γ = 180 :=
by
  sorry

end triangle_enlargement_invariant_l270_270297


namespace find_a9_for_geo_seq_l270_270881

noncomputable def geo_seq_a_3_a_13_positive_common_ratio_2 (a_3 a_9 a_13 : ℕ) : Prop :=
  (a_3 * a_13 = 16) ∧ (a_3 > 0) ∧ (a_9 > 0) ∧ (a_13 > 0) ∧ (forall (n₁ n₂ : ℕ), a_9 = a_3 * 2 ^ 6)

theorem find_a9_for_geo_seq (a_3 a_9 a_13 : ℕ) 
  (h : geo_seq_a_3_a_13_positive_common_ratio_2 a_3 a_9 a_13) :
  a_9 = 8 :=
  sorry

end find_a9_for_geo_seq_l270_270881


namespace emily_necklaces_l270_270207

theorem emily_necklaces (total_beads : ℕ) (beads_per_necklace : ℕ) (necklaces_made : ℕ) 
  (h1 : total_beads = 52)
  (h2 : beads_per_necklace = 2)
  (h3 : necklaces_made = total_beads / beads_per_necklace) :
  necklaces_made = 26 :=
by
  rw [h1, h2] at h3
  exact h3

end emily_necklaces_l270_270207


namespace outfit_combinations_l270_270588

theorem outfit_combinations :
  let num_shirts := 5
  let num_pants := 4
  let num_ties := 6 -- 5 ties + no tie option
  let num_belts := 3 -- 2 belts + no belt option
  num_shirts * num_pants * num_ties * num_belts = 360 :=
by
  let num_shirts := 5
  let num_pants := 4
  let num_ties := 6
  let num_belts := 3
  show num_shirts * num_pants * num_ties * num_belts = 360
  sorry

end outfit_combinations_l270_270588


namespace intersection_point_not_on_x_3_l270_270430

noncomputable def f (x : ℝ) : ℝ := (x^2 - 8*x + 15) / (3*x - 6)
noncomputable def g (x : ℝ) : ℝ := (-1/3 * x^2 + 6*x - 6) / (x - 2)

theorem intersection_point_not_on_x_3 : 
  ∃ x y : ℝ, (x ≠ 3) ∧ (f x = g x) ∧ (y = f x) ∧ (x = 11/3 ∧ y = -11/3) :=
by
  sorry

end intersection_point_not_on_x_3_l270_270430


namespace cost_of_plastering_l270_270333

/-- 
Let's define the problem conditions
Length of the tank (in meters)
-/
def tank_length : ℕ := 25

/--
Width of the tank (in meters)
-/
def tank_width : ℕ := 12

/--
Depth of the tank (in meters)
-/
def tank_depth : ℕ := 6

/--
Cost of plastering per square meter (55 paise converted to rupees)
-/
def cost_per_sq_meter : ℝ := 0.55

/--
Prove that the cost of plastering the walls and bottom of the tank is 409.2 rupees
-/
theorem cost_of_plastering (total_cost : ℝ) : 
  total_cost = 409.2 :=
sorry

end cost_of_plastering_l270_270333


namespace sum_max_min_expr_l270_270889

theorem sum_max_min_expr (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) : 
    let expr := (x / |x|) + (|y| / y) - (|x * y| / (x * y))
    max (max expr (expr)) (min expr expr) = -2 :=
sorry

end sum_max_min_expr_l270_270889


namespace problem_statement_l270_270877

theorem problem_statement (a b : ℝ) (h1 : 2^a = 10) (h2 : 5^b = 10) : (1 / a) + (1 / b) = 1 :=
sorry

end problem_statement_l270_270877


namespace problem_l270_270656

noncomputable def f (x : ℝ) (a b : ℝ) := x^2 + a * x + b

theorem problem (a b : ℝ) (h1 : f 1 a b = 0) (h2 : f 2 a b = 0) : f (-1) a b = 6 :=
by
  sorry

end problem_l270_270656


namespace polynomial_identity_l270_270214

theorem polynomial_identity : 
  ∀ x : ℝ, 
    5 * x^3 - 32 * x^2 + 75 * x - 71 = 
    5 * (x - 2)^3 + (-2) * (x - 2)^2 + 7 * (x - 2) - 9 :=
by 
  sorry

end polynomial_identity_l270_270214


namespace remainder_example_l270_270640

def P (x : ℝ) := 8 * x^3 - 20 * x^2 + 28 * x - 26
def D (x : ℝ) := 4 * x - 8

theorem remainder_example : P 2 = 14 :=
by
  sorry

end remainder_example_l270_270640


namespace increasing_interval_l270_270516

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * x^2

theorem increasing_interval :
  ∃ a b : ℝ, (0 < a) ∧ (a < b) ∧ (b = 1/2) ∧ (∀ x : ℝ, a < x ∧ x < b → (deriv f x > 0)) :=
by
  sorry

end increasing_interval_l270_270516


namespace additional_life_vests_needed_l270_270011

def num_students : ℕ := 40
def num_instructors : ℕ := 10
def life_vests_on_hand : ℕ := 20
def percent_students_with_vests : ℕ := 20

def total_people : ℕ := num_students + num_instructors
def students_with_vests : ℕ := (percent_students_with_vests * num_students) / 100
def total_vests_available : ℕ := life_vests_on_hand + students_with_vests

theorem additional_life_vests_needed : 
  total_people - total_vests_available = 22 :=
by 
  sorry

end additional_life_vests_needed_l270_270011


namespace simplify_fraction_l270_270302

theorem simplify_fraction (a b m : ℝ) (h1 : (a / b) ^ m = (a^m) / (b^m)) (h2 : (-1 : ℝ) ^ (0 : ℝ) = 1) :
  ( (81 / 16) ^ (3 / 4) ) - 1 = 19 / 8 :=
by
  sorry

end simplify_fraction_l270_270302


namespace negation_proof_equivalence_l270_270450

theorem negation_proof_equivalence : 
  ¬(∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
sorry

end negation_proof_equivalence_l270_270450


namespace max_covered_squares_by_tetromino_l270_270735

-- Definition of the grid size
def grid_size := (5, 5)

-- Definition of S-Tetromino (Z-Tetromino) coverage covering four contiguous squares
def is_STetromino (coords: List (Nat × Nat)) : Prop := 
  coords.length = 4 ∧ ∃ (x y : Nat), coords = [(x, y), (x, y+1), (x+1, y+1), (x+1, y+2)]

-- Definition of the coverage constraint
def no_more_than_two_tiles (cover: List (Nat × Nat)) : Prop :=
  ∀ (coord: Nat × Nat), cover.count coord ≤ 2

-- Definition of the total tiled squares covered by at least one tile
def tiles_covered (cover: List (Nat × Nat)) : Nat := 
  cover.toFinset.card 

-- Definition of the problem using proof equivalence
theorem max_covered_squares_by_tetromino
  (cover: List (List (Nat × Nat)))
  (H_tiles: ∀ t, t ∈ cover → is_STetromino t)
  (H_coverage: no_more_than_two_tiles (cover.join)) :
  tiles_covered (cover.join) = 24 :=
sorry 

end max_covered_squares_by_tetromino_l270_270735


namespace virginia_more_than_adrienne_l270_270973

def teaching_years (V A D : ℕ) : Prop :=
  V + A + D = 102 ∧ D = 43 ∧ V = D - 9

theorem virginia_more_than_adrienne (V A : ℕ) (h : teaching_years V A 43) : V - A = 9 :=
by
  sorry

end virginia_more_than_adrienne_l270_270973


namespace trapezium_top_width_l270_270748

theorem trapezium_top_width (bottom_width : ℝ) (height : ℝ) (area : ℝ) (top_width : ℝ) 
  (h1 : bottom_width = 8) 
  (h2 : height = 50) 
  (h3 : area = 500) : top_width = 12 :=
by
  -- Definitions
  have h_formula : area = 1 / 2 * (top_width + bottom_width) * height := by sorry
  -- Applying given conditions to the formula
  rw [h1, h2, h3] at h_formula
  -- Solve for top_width
  sorry

end trapezium_top_width_l270_270748


namespace erasers_given_l270_270296

theorem erasers_given (initial final : ℕ) (h1 : initial = 8) (h2 : final = 11) : (final - initial = 3) :=
by
  sorry

end erasers_given_l270_270296


namespace center_of_circle_sum_l270_270641
-- Import the entire library

-- Define the problem using declarations for conditions and required proof
theorem center_of_circle_sum (x y : ℝ) 
  (h : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = 9 → (x = 2) ∧ (y = -3)) : 
  x + y = -1 := 
by 
  sorry 

end center_of_circle_sum_l270_270641


namespace ashok_total_subjects_l270_270630

/-- Ashok secured an average of 78 marks in some subjects. If the average of marks in 5 subjects 
is 74, and he secured 98 marks in the last subject, how many subjects are there in total? -/
theorem ashok_total_subjects (n : ℕ) 
  (avg_all : 78 * n = 74 * (n - 1) + 98) : n = 6 :=
sorry

end ashok_total_subjects_l270_270630


namespace find_e_of_conditions_l270_270960

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := x^3 + d * x^2 + e * x + f

theorem find_e_of_conditions (d e f : ℝ) 
  (h1 : f = 6) 
  (h2 : -d / 3 = -f)
  (h3 : -f = d + e + f - 1) : 
  e = -30 :=
by 
  sorry

end find_e_of_conditions_l270_270960


namespace quadratic_root_solution_l270_270223

theorem quadratic_root_solution (k : ℤ) (a : ℤ) :
  (∀ x, x^2 + k * x - 10 = 0 → x = 2 ∨ x = a) →
  2 + a = -k →
  2 * a = -10 →
  k = 3 ∧ a = -5 :=
by
  sorry

end quadratic_root_solution_l270_270223


namespace translation_theorem_l270_270760

noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + θ)
noncomputable def g (θ : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x - 2 * φ + θ)

theorem translation_theorem
  (θ φ : ℝ)
  (hθ1 : |θ| < Real.pi / 2)
  (hφ1 : 0 < φ)
  (hφ2 : φ < Real.pi)
  (hf : f θ 0 = 1 / 2)
  (hg : g θ φ 0 = 1 / 2) :
  φ = 2 * Real.pi / 3 :=
sorry

end translation_theorem_l270_270760


namespace polynomial_transformation_exists_l270_270933

theorem polynomial_transformation_exists (P : ℝ → ℝ → ℝ) (hP : ∀ x y, P (x - 1) (y - 2 * x + 1) = P x y) :
  ∃ Φ : ℝ → ℝ, ∀ x y, P x y = Φ (y - x^2) := by
  sorry

end polynomial_transformation_exists_l270_270933


namespace ron_pay_cuts_l270_270937

-- Define percentages as decimals
def cut_1 : ℝ := 0.05
def cut_2 : ℝ := 0.10
def cut_3 : ℝ := 0.15
def overall_cut : ℝ := 0.27325

-- Define the total number of pay cuts
def total_pay_cuts : ℕ := 3

noncomputable def verify_pay_cuts (cut_1 cut_2 cut_3 overall_cut : ℝ) (total_pay_cuts : ℕ) : Prop :=
  (((1 - cut_1) * (1 - cut_2) * (1 - cut_3) = (1 - overall_cut)) ∧ (total_pay_cuts = 3))

theorem ron_pay_cuts 
  (cut_1 : ℝ := 0.05)
  (cut_2 : ℝ := 0.10)
  (cut_3 : ℝ := 0.15)
  (overall_cut : ℝ := 0.27325)
  (total_pay_cuts : ℕ := 3) 
  : verify_pay_cuts cut_1 cut_2 cut_3 overall_cut total_pay_cuts :=
by sorry

end ron_pay_cuts_l270_270937


namespace simplify_expression_l270_270442

theorem simplify_expression (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) (h4 : x ≠ 5):
  ((x^2 - 4 * x + 3) / (x^2 - 6 * x + 9)) / ((x^2 - 6 * x + 8) / (x^2 - 8 * x + 15)) = 
  (x - 1) * (x - 5) / ((x - 3) * (x - 4) * (x - 2)) :=
sorry

end simplify_expression_l270_270442


namespace proof_ac_plus_bd_l270_270536

theorem proof_ac_plus_bd (a b c d : ℝ)
  (h1 : a + b + c = 10)
  (h2 : a + b + d = -6)
  (h3 : a + c + d = 0)
  (h4 : b + c + d = 15) :
  ac + bd = -130.111 := 
by
  sorry

end proof_ac_plus_bd_l270_270536


namespace problem1_problem2_problem3_l270_270335

-- Problem (I)
theorem problem1 (x : ℝ) (hx : x > 1) : 2 * Real.log x < x - 1/x :=
sorry

-- Problem (II)
theorem problem2 (a : ℝ) : (∀ t : ℝ, t > 0 → (1 + a / t) * Real.log (1 + t) > a) → 0 < a ∧ a ≤ 2 :=
sorry

-- Problem (III)
theorem problem3 : (9/10 : ℝ)^19 < 1 / (Real.exp 2) :=
sorry

end problem1_problem2_problem3_l270_270335


namespace twelve_xy_leq_fourx_1_y_9y_1_x_l270_270283

theorem twelve_xy_leq_fourx_1_y_9y_1_x
  (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) :=
  sorry

end twelve_xy_leq_fourx_1_y_9y_1_x_l270_270283


namespace initial_pennies_l270_270740

-- Defining the conditions
def pennies_spent : Nat := 93
def pennies_left : Nat := 5

-- Question: How many pennies did Sam have in his bank initially?
theorem initial_pennies : pennies_spent + pennies_left = 98 := by
  sorry

end initial_pennies_l270_270740


namespace reciprocal_of_neg_2023_l270_270455

theorem reciprocal_of_neg_2023 : (1 : ℝ) / (-2023) = -(1 / 2023) :=
by 
  sorry

end reciprocal_of_neg_2023_l270_270455


namespace total_distance_hiked_l270_270952

-- Defining the distances Terrell hiked on Saturday and Sunday
def distance_Saturday : Real := 8.2
def distance_Sunday : Real := 1.6

-- Stating the theorem to prove the total distance
theorem total_distance_hiked : distance_Saturday + distance_Sunday = 9.8 := by
  sorry

end total_distance_hiked_l270_270952


namespace compare_constants_l270_270870

theorem compare_constants :
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  in c < a ∧ a < b :=
by
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  sorry

end compare_constants_l270_270870


namespace expand_expression_l270_270376

theorem expand_expression (x : ℝ) : 16 * (2 * x + 5) = 32 * x + 80 :=
by
  sorry

end expand_expression_l270_270376


namespace alice_needs_more_life_vests_l270_270012

-- Definitions based on the given conditions
def students : ℕ := 40
def instructors : ℕ := 10
def lifeVestsOnHand : ℕ := 20
def percentWithLifeVests : ℚ := 0.20

-- Statement of the problem
theorem alice_needs_more_life_vests :
  let totalPeople := students + instructors
  let lifeVestsBroughtByStudents := (percentWithLifeVests * students).toNat
  let totalLifeVestsAvailable := lifeVestsOnHand + lifeVestsBroughtByStudents
  totalPeople - totalLifeVestsAvailable = 22 :=
by
  sorry

end alice_needs_more_life_vests_l270_270012


namespace cyclist_speed_l270_270708

noncomputable def required_speed (d t : ℝ) : ℝ := d / t

theorem cyclist_speed :
  ∀ (d t : ℝ), 
  (d / 10 = t + 1) → 
  (d / 15 = t - 1) →
  required_speed d t = 12 := 
by
  intros d t h1 h2
  sorry

end cyclist_speed_l270_270708


namespace arithmetic_seq_problem_l270_270560

theorem arithmetic_seq_problem
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a (n + 1) - a n = d)
  (h2 : d > 0)
  (h3 : a 1 + a 2 + a 3 = 15)
  (h4 : a 1 * a 2 * a 3 = 80) :
  a 11 + a 12 + a 13 = 105 :=
sorry

end arithmetic_seq_problem_l270_270560


namespace minimum_value_expression_l270_270087

theorem minimum_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  3 ≤ (sqrt ((x^2 + y^2) * (4*x^2 + y^2))) / (x * y) :=
by
  sorry

end minimum_value_expression_l270_270087


namespace find_a9_l270_270266

variable {a : ℕ → ℤ}  -- Define a as a sequence of integers
variable (d : ℤ) (a3 : ℤ) (a4 : ℤ)

-- Define the specific conditions given in the problem
def arithmetic_sequence_condition (a : ℕ → ℤ) (d : ℤ) (a3 a4 : ℤ) : Prop :=
  a 3 + a 4 = 12 ∧ d = 2

-- Define the arithmetic sequence relation
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

-- Statement to prove
theorem find_a9 
  (a : ℕ → ℤ) (d : ℤ) (a3 a4 : ℤ)
  (h1 : arithmetic_sequence_condition a d a3 a4)
  (h2 : arithmetic_sequence a d) :
  a 9 = 17 :=
sorry

end find_a9_l270_270266


namespace trapezium_area_proof_l270_270852

-- Define the lengths of the parallel sides and the distance between them
def a : ℝ := 20
def b : ℝ := 18
def h : ℝ := 15

-- Define the area of the trapezium
def area_of_trapezium (a b h : ℝ) : ℝ := (1/2) * (a + b) * h

-- State the theorem to be proved
theorem trapezium_area_proof : area_of_trapezium a b h = 285 := by
  sorry

end trapezium_area_proof_l270_270852


namespace blue_stamp_price_l270_270102

theorem blue_stamp_price :
  ∀ (red_stamps blue_stamps yellow_stamps : ℕ) (red_price blue_price yellow_price total_earnings : ℝ),
    red_stamps = 20 →
    blue_stamps = 80 →
    yellow_stamps = 7 →
    red_price = 1.1 →
    yellow_price = 2 →
    total_earnings = 100 →
    (red_stamps * red_price + yellow_stamps * yellow_price + blue_stamps * blue_price = total_earnings) →
    blue_price = 0.80 :=
by
  intros red_stamps blue_stamps yellow_stamps red_price blue_price yellow_price total_earnings
  intros h_red_stamps h_blue_stamps h_yellow_stamps h_red_price h_yellow_price h_total_earnings
  intros h_earning_eq
  sorry

end blue_stamp_price_l270_270102


namespace math_problem_l270_270635

theorem math_problem 
  (a1 : (10^4 + 500) = 100500)
  (a2 : (25^4 + 500) = 390625500)
  (a3 : (40^4 + 500) = 256000500)
  (a4 : (55^4 + 500) = 915062500)
  (a5 : (70^4 + 500) = 24010062500)
  (b1 : (5^4 + 500) = 625+500)
  (b2 : (20^4 + 500) = 160000500)
  (b3 : (35^4 + 500) = 150062500)
  (b4 : (50^4 + 500) = 625000500)
  (b5 : (65^4 + 500) = 1785062500) :
  ( (100500 * 390625500 * 256000500 * 915062500 * 24010062500) / (625+500 * 160000500 * 150062500 * 625000500 * 1785062500) = 240) :=
by
  sorry

end math_problem_l270_270635


namespace find_y_arithmetic_mean_l270_270393

theorem find_y_arithmetic_mean (y : ℝ) 
  (h : (8 + 15 + 20 + 7 + y + 9) / 6 = 12) : 
  y = 13 :=
sorry

end find_y_arithmetic_mean_l270_270393


namespace rectangular_solid_volume_l270_270184

theorem rectangular_solid_volume (a b c : ℝ) 
  (h1 : a * b = 15) 
  (h2 : b * c = 20) 
  (h3 : c * a = 12) : a * b * c = 60 :=
by
  sorry

end rectangular_solid_volume_l270_270184


namespace xyz_inequality_l270_270205

theorem xyz_inequality (x y z : ℝ) (hx : 0 ≤ x) (hx' : x ≤ 1) (hy : 0 ≤ y) (hy' : y ≤ 1) (hz : 0 ≤ z) (hz' : z ≤ 1) :
  (x^2 / (1 + x + x*y*z) + y^2 / (1 + y + x*y*z) + z^2 / (1 + z + x*y*z) ≤ 1) :=
sorry

end xyz_inequality_l270_270205


namespace side_lengths_sum_eq_225_l270_270726

noncomputable def GX (x y z : ℝ) : ℝ :=
  (1 / 3) * (x + y + z) - x

noncomputable def GY (x y z : ℝ) : ℝ :=
  (1 / 3) * (x + y + z) - y

noncomputable def GZ (x y z : ℝ) : ℝ :=
  (1 / 3) * (x + y + z) - z

theorem side_lengths_sum_eq_225
  (x y z : ℝ)
  (h : GX x y z ^ 2 + GY x y z ^ 2 + GZ x y z ^ 2 = 75) :
  (x - y) ^ 2 + (x - z) ^ 2 + (y - z) ^ 2 = 225 := by {
  sorry
}

end side_lengths_sum_eq_225_l270_270726


namespace radius_of_semi_circle_l270_270452

-- Given definitions and conditions
def perimeter : ℝ := 33.934511513692634
def pi_approx : ℝ := 3.141592653589793

-- The formula for the perimeter of a semi-circle
def semi_circle_perimeter (r : ℝ) : ℝ := pi_approx * r + 2 * r

-- The theorem we want to prove
theorem radius_of_semi_circle (r : ℝ) (h: semi_circle_perimeter r = perimeter) : r = 6.6 :=
sorry

end radius_of_semi_circle_l270_270452


namespace factor_difference_of_squares_l270_270848

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l270_270848


namespace solution_of_fraction_l270_270414

theorem solution_of_fraction (x : ℝ) (h1 : x^2 - 9 = 0) (h2 : x + 3 ≠ 0) : x = 3 :=
by
  sorry

end solution_of_fraction_l270_270414


namespace probability_of_two_one_color_and_one_other_color_l270_270618

theorem probability_of_two_one_color_and_one_other_color
    (black_balls white_balls : ℕ)
    (total_drawn : ℕ)
    (draw_two_black_one_white : ℕ)
    (draw_one_black_two_white : ℕ)
    (total_ways : ℕ)
    (favorable_ways : ℕ)
    (probability : ℚ) :
    black_balls = 8 →
    white_balls = 7 →
    total_drawn = 3 →
    draw_two_black_one_white = 196 →
    draw_one_black_two_white = 168 →
    total_ways = 455 →
    favorable_ways = draw_two_black_one_white + draw_one_black_two_white →
    probability = favorable_ways / total_ways →
    probability = 4 / 5 :=
by sorry

end probability_of_two_one_color_and_one_other_color_l270_270618


namespace coin_difference_l270_270570

-- Definitions based on problem conditions
def denominations : List ℕ := [5, 10, 25, 50]
def amount_owed : ℕ := 55

-- Proof statement
theorem coin_difference :
  let min_coins := 1 + 1 -- one 50-cent coin and one 5-cent coin
  let max_coins := 11 -- eleven 5-cent coins
  max_coins - min_coins = 9 :=
by
  -- Proof details skipped
  sorry

end coin_difference_l270_270570


namespace part1_part2_l270_270227

section
variable (k : ℝ)

/-- Part 1: Range of k -/
def discriminant_eqn (k : ℝ) := (2 * k - 1) ^ 2 - 4 * (k ^ 2 - 1)

theorem part1 (h : discriminant_eqn k ≥ 0) : k ≤ 5 / 4 :=
by sorry

/-- Part 2: Value of k when x₁ and x₂ satisfy the given condition -/
def x1_x2_eqn (k x1 x2 : ℝ) := x1 ^ 2 + x2 ^ 2 = 16 + x1 * x2

def vieta (k : ℝ) (x1 x2 : ℝ) :=
  x1 + x2 = 1 - 2 * k ∧ x1 * x2 = k ^ 2 - 1

theorem part2 (x1 x2 : ℝ) (h1 : vieta k x1 x2) (h2 : x1_x2_eqn k x1 x2) : k = -2 :=
by sorry

end

end part1_part2_l270_270227


namespace area_of_pentagon_ABCDE_l270_270448

open Real Set Polynomial

-- Define the shapes and conditions
noncomputable def pentagon_ABCDE : List (Point ℝ) :=
[⟨0, 0⟩, ⟨1, 0⟩, ⟨2 * cos (2 * π / 3), 2 * sin (2 * π / 3)⟩, ⟨3 * cos (4 * π / 3), 3 * sin (4 * π / 3)⟩, ⟨0, sin (2 * π / 3)⟩]

-- Prove the area of the pentagon
theorem area_of_pentagon_ABCDE : 
  Polygon.area pentagon_ABCDE = 5 * sqrt 3 :=
by
  sorry

end area_of_pentagon_ABCDE_l270_270448


namespace solve_log_eq_l270_270742

noncomputable def log3 (x : ℝ) := Real.log x / Real.log 3

theorem solve_log_eq :
  (∃ x : ℝ, log3 ((5 * x + 15) / (7 * x - 5)) + log3 ((7 * x - 5) / (2 * x - 3)) = 3 ∧ x = 96 / 49) :=
by
  sorry

end solve_log_eq_l270_270742


namespace solve_for_x_l270_270166

theorem solve_for_x (x : ℝ) (h : 2 * (1/x + 3/x / 6/x) - 1/x = 1.5) : x = 2 := 
by 
  sorry

end solve_for_x_l270_270166


namespace inequality_proof_l270_270864

-- Define constants
def a : ℝ := 0.1 * Real.exp 0.1
def b : ℝ := 1 / 9
def c : ℝ := -Real.log 0.9

-- Assert the inequalities
theorem inequality_proof : c < a ∧ a < b :=
by sorry

end inequality_proof_l270_270864


namespace selection_methods_l270_270496

/-- Type definition for the workers -/
inductive Worker
  | PliersOnly  : Worker
  | CarOnly     : Worker
  | Both        : Worker

/-- Conditions -/
def num_workers : ℕ := 11
def num_pliers_only : ℕ := 5
def num_car_only : ℕ := 4
def num_both : ℕ := 2
def pliers_needed : ℕ := 4
def car_needed : ℕ := 4

/-- Main statement -/
theorem selection_methods : 
  (num_pliers_only + num_car_only + num_both = num_workers) → 
  (num_pliers_only = 5) → 
  (num_car_only = 4) → 
  (num_both = 2) → 
  (pliers_needed = 4) → 
  (car_needed = 4) → 
  ∃ n : ℕ, n = 185 := 
by 
  sorry -- Proof Skipped

end selection_methods_l270_270496


namespace value_of_a8_l270_270861

theorem value_of_a8 (a : ℕ → ℝ) :
  (1 + x) ^ 10 = a 0 + a 1 * (1 - x) + a 2 * (1 - x) ^ 2 + a 3 * (1 - x) ^ 3 +
  a 4 * (1 - x) ^ 4 + a 5 * (1 - x) ^ 5 + a 6 * (1 - x) ^ 6 + a 7 * (1 - x) ^ 7 + 
  a 8 * (1 - x) ^ 8 + a 9 * (1 - x) ^ 9 + a 10 * (1 - x) ^ 10 → 
  a 8 = 180 :=
by
  sorry

end value_of_a8_l270_270861


namespace b_2016_eq_neg_4_l270_270907

def b : ℕ → ℤ
| 0     => 1
| 1     => 5
| (n+2) => b (n+1) - b n

theorem b_2016_eq_neg_4 : b 2015 = -4 :=
sorry

end b_2016_eq_neg_4_l270_270907


namespace value_of_expression_l270_270243

variables {x y z w : ℝ}

theorem value_of_expression (h1 : 4 * x * z + y * w = 4) (h2 : x * w + y * z = 8) :
  (2 * x + y) * (2 * z + w) = 20 :=
by
  sorry

end value_of_expression_l270_270243


namespace prob_two_consecutive_heads_is_half_l270_270181

noncomputable def prob_at_least_two_consecutive_heads : ℚ :=
  let total_outcomes := 16 in
  let unfavorable_states := 8 in
  let p_no_consecutive_heads := (unfavorable_states : ℚ) / (total_outcomes : ℚ) in
  1 - p_no_consecutive_heads

theorem prob_two_consecutive_heads_is_half :
  prob_at_least_two_consecutive_heads = 1 / 2 :=
by
  sorry

end prob_two_consecutive_heads_is_half_l270_270181


namespace total_distance_of_trail_l270_270300

theorem total_distance_of_trail (a b c d e : ℕ) 
    (h1 : a + b + c = 30) 
    (h2 : b + d = 30) 
    (h3 : d + e = 28) 
    (h4 : a + d = 34) : 
    a + b + c + d + e = 58 := 
sorry

end total_distance_of_trail_l270_270300


namespace odd_function_value_l270_270921

theorem odd_function_value (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_fx : ∀ x : ℝ, x ≤ 0 → f x = 2 * x ^ 2 - x) :
  f 1 = -3 := 
sorry

end odd_function_value_l270_270921


namespace find_k_l270_270897

theorem find_k (x y k : ℤ) 
  (h1 : 2 * x - y = 5 * k + 6) 
  (h2 : 4 * x + 7 * y = k) 
  (h3 : x + y = 2023) : 
  k = 2022 := 
  by 
    sorry

end find_k_l270_270897


namespace crayons_given_l270_270294

theorem crayons_given (initial lost left given : ℕ)
  (h1 : initial = 1453)
  (h2 : lost = 558)
  (h3 : left = 332)
  (h4 : given = initial - left - lost) :
  given = 563 :=
by
  rw [h1, h2, h3] at h4
  exact h4

end crayons_given_l270_270294


namespace no_two_points_same_color_distance_one_l270_270182

/-- Prove that if a plane is colored using seven colors, it is not necessary that there will be two points of the same color exactly 1 unit apart. -/
theorem no_two_points_same_color_distance_one (coloring : ℝ × ℝ → Fin 7) :
  ¬ ∀ (x y : ℝ × ℝ), (dist x y = 1) → (coloring x = coloring y) :=
by
  sorry

end no_two_points_same_color_distance_one_l270_270182


namespace shirt_original_price_l270_270720

theorem shirt_original_price (P : ℝ) : 
  (18 = P * 0.75 * 0.75 * 0.90 * 1.15) → 
  P = 18 / (0.75 * 0.75 * 0.90 * 1.15) :=
by
  intro h
  sorry

end shirt_original_price_l270_270720


namespace no_faces_painted_two_or_three_faces_painted_l270_270320

-- Define the dimensions of the cuboid
def cuboid_length : ℕ := 3
def cuboid_width : ℕ := 4
def cuboid_height : ℕ := 5

-- Define the number of small cubes
def small_cubes_total : ℕ := 60

-- Define the number of small cubes with no faces painted
def small_cubes_no_faces_painted : ℕ := (cuboid_length - 2) * (cuboid_width - 2) * (cuboid_height - 2)

-- Define the number of small cubes with 2 faces painted
def small_cubes_two_faces_painted : ℕ := (cuboid_length - 2) * cuboid_width +
                                          (cuboid_width - 2) * cuboid_length +
                                          (cuboid_height - 2) * cuboid_width

-- Define the number of small cubes with 3 faces painted
def small_cubes_three_faces_painted : ℕ := 8

-- Define the probabilities
def probability_no_faces_painted : ℚ := small_cubes_no_faces_painted / small_cubes_total
def probability_two_or_three_faces_painted : ℚ := (small_cubes_two_faces_painted + small_cubes_three_faces_painted) / small_cubes_total

-- Theorems to prove
theorem no_faces_painted (h : cuboid_length = 3 ∧ cuboid_width = 4 ∧ cuboid_height = 5 ∧ 
                           small_cubes_total = 60 ∧ small_cubes_no_faces_painted = 6) :
  probability_no_faces_painted = 1 / 10 := by
  sorry

theorem two_or_three_faces_painted (h : cuboid_length = 3 ∧ cuboid_width = 4 ∧ cuboid_height = 5 ∧ 
                                    small_cubes_total = 60 ∧ small_cubes_two_faces_painted = 24 ∧
                                    small_cubes_three_faces_painted = 8) :
  probability_two_or_three_faces_painted = 8 / 15 := by
  sorry

end no_faces_painted_two_or_three_faces_painted_l270_270320


namespace count_two_digit_or_less_numbers_l270_270686

theorem count_two_digit_or_less_numbers : 
  let count_single_digit (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let count_two_digit (d₁ d₂ : ℕ) : Bool := (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (d₁ ≠ d₂)
  let count_two_digit_with_zero (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let nums_with_single_digit_count := 45
  let nums_with_exactly_two_digits_count := 1872
  let nums_with_two_digits_including_zero_count := 234
  let total_count := nums_with_single_digit_count + nums_with_exactly_two_digits_count + nums_with_two_digits_including_zero_count
  total_count = 2151 :=
by
  sorry

end count_two_digit_or_less_numbers_l270_270686


namespace find_side_b_l270_270045

variables {A B C a b c x : ℝ}

theorem find_side_b 
  (cos_A : ℝ) (cos_C : ℝ) (a : ℝ) (hcosA : cos_A = 4/5) 
  (hcosC : cos_C = 5/13) (ha : a = 1) : 
  b = 21/13 :=
by
  sorry

end find_side_b_l270_270045


namespace transport_tax_to_be_paid_l270_270273

noncomputable def engine_power : ℕ := 150
noncomputable def tax_rate : ℕ := 20
noncomputable def annual_tax : ℕ := engine_power * tax_rate
noncomputable def months_used : ℕ := 8
noncomputable def prorated_tax : ℕ := (months_used * annual_tax) / 12

theorem transport_tax_to_be_paid : prorated_tax = 2000 := 
by 
  -- sorry is used to skip the proof step
  sorry

end transport_tax_to_be_paid_l270_270273


namespace total_clothes_washed_l270_270506

def number_of_clothing_items (Cally Danny Emily shared_socks : ℕ) : ℕ :=
  Cally + Danny + Emily + shared_socks

theorem total_clothes_washed :
  let Cally_clothes := (10 + 5 + 7 + 6 + 3)
  let Danny_clothes := (6 + 8 + 10 + 6 + 4)
  let Emily_clothes := (8 + 6 + 9 + 5 + 2)
  let shared_socks := (3 + 2)
  number_of_clothing_items Cally_clothes Danny_clothes Emily_clothes shared_socks = 100 :=
by
  sorry

end total_clothes_washed_l270_270506


namespace find_a1_and_d_l270_270422

variable (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) (a5 : ℤ := -1) (a8 : ℤ := 2)

def arithmetic_sequence : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

theorem find_a1_and_d
  (h : arithmetic_sequence a d)
  (h_a5 : a 5 = -1)
  (h_a8 : a 8 = 2) :
  a 1 = -5 ∧ d = 1 :=
by
  sorry

end find_a1_and_d_l270_270422


namespace star_is_addition_l270_270485

variable {α : Type} [AddCommGroup α]

-- Define the binary operation star
variable (star : α → α → α)

-- Define the condition given in the problem
axiom star_condition : ∀ (a b c : α), star (star a b) c = a + b + c

-- Prove that star is the same as usual addition
theorem star_is_addition : ∀ (a b : α), star a b = a + b :=
  sorry

end star_is_addition_l270_270485


namespace continuous_stripe_encircling_tetrahedron_probability_l270_270950

noncomputable def tetrahedron_continuous_stripe_probability : ℚ :=
  let total_combinations := 3^4
  let favorable_combinations := 2 
  favorable_combinations / total_combinations

theorem continuous_stripe_encircling_tetrahedron_probability :
  tetrahedron_continuous_stripe_probability = 2 / 81 :=
by
  -- the proof would be here
  sorry

end continuous_stripe_encircling_tetrahedron_probability_l270_270950


namespace fraction_difference_is_correct_l270_270798

-- Convert repeating decimal to fraction
def repeating_to_fraction : ℚ := 0.72 + 72 / 9900

-- Convert 0.72 to fraction
def decimal_to_fraction : ℚ := 72 / 100

-- Define the difference between the fractions
def fraction_difference := repeating_to_fraction - decimal_to_fraction

-- Final statement to prove
theorem fraction_difference_is_correct : fraction_difference = 2 / 275 := by
  sorry

end fraction_difference_is_correct_l270_270798


namespace matrix_inverse_correct_l270_270025

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, -2], ![5, 3]]

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![3/22, 1/11], ![-5/22, 2/11]]

theorem matrix_inverse_correct : A⁻¹ = A_inv :=
  by
    sorry

end matrix_inverse_correct_l270_270025


namespace expand_and_simplify_l270_270022

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 4) + 6 = x^2 + x - 6 := by
  sorry

end expand_and_simplify_l270_270022


namespace items_in_descending_order_l270_270006

-- Assume we have four real numbers representing the weights of the items.
variables (C S B K : ℝ)

-- The conditions given in the problem.
axiom h1 : S > B
axiom h2 : C + B > S + K
axiom h3 : K + C = S + B

-- Define a predicate to check if the weights are in descending order.
def DescendingOrder (C S B K : ℝ) : Prop :=
  C > S ∧ S > B ∧ B > K

-- The theorem to prove the descending order of weights.
theorem items_in_descending_order : DescendingOrder C S B K :=
sorry

end items_in_descending_order_l270_270006


namespace solution_set_of_inequality_l270_270034

theorem solution_set_of_inequality (x : ℝ) : x < (1 / x) ↔ (x < -1 ∨ (0 < x ∧ x < 1)) :=
by
  sorry

end solution_set_of_inequality_l270_270034


namespace max_min_diff_half_dollars_l270_270571

-- Definitions based only on conditions
variables (a c d : ℕ)

-- Conditions:
def condition1 : Prop := a + c + d = 60
def condition2 : Prop := 5 * a + 25 * c + 50 * d = 1000

-- The mathematically equivalent proof statement
theorem max_min_diff_half_dollars : condition1 a c d → condition2 a c d → (∃ d_min d_max : ℕ, d_min = 0 ∧ d_max = 15 ∧ d_max - d_min = 15) :=
by
  intros
  sorry

end max_min_diff_half_dollars_l270_270571


namespace factorial_mod_eq_l270_270772

theorem factorial_mod_eq :
  63! % 71 = 61! % 71 := 
sorry

end factorial_mod_eq_l270_270772


namespace solve_for_b_l270_270409

variable (a b c d m : ℝ)

theorem solve_for_b (h : m = cadb / (a - b)) : b = ma / (cad + m) :=
sorry

end solve_for_b_l270_270409


namespace repeating_decimal_exceeds_finite_decimal_by_l270_270812

-- Definitions based on the problem conditions
def repeating_decimal := 8 / 11
def finite_decimal := 18 / 25

-- Statement to be proved
theorem repeating_decimal_exceeds_finite_decimal_by : 
  repeating_decimal - finite_decimal = 2 / 275 :=
by
  -- Skipping the proof itself with 'sorry'
  sorry

end repeating_decimal_exceeds_finite_decimal_by_l270_270812


namespace log_arith_example_l270_270174

noncomputable def log10 (x : ℝ) : ℝ := sorry -- Assume the definition of log base 10

theorem log_arith_example : log10 4 + 2 * log10 5 + 8^(2/3) = 6 := 
by
  -- The proof would go here
  sorry

end log_arith_example_l270_270174


namespace bucket_full_weight_l270_270001

variables (x y p q : Real)

theorem bucket_full_weight (h1 : x + (1 / 4) * y = p)
                           (h2 : x + (3 / 4) * y = q) :
    x + y = 3 * q - p :=
by
  sorry

end bucket_full_weight_l270_270001


namespace positive_triple_l270_270299

theorem positive_triple
  (a b c : ℝ)
  (h1 : a + b + c > 0)
  (h2 : ab + bc + ca > 0)
  (h3 : abc > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end positive_triple_l270_270299


namespace angle_C_value_sides_a_b_l270_270898

variables (A B C : ℝ) (a b c : ℝ)

-- First part: Proving the value of angle C
theorem angle_C_value
  (h1 : 2*Real.cos (A/2)^2 + (Real.cos B - Real.sqrt 3 * Real.sin B) * Real.cos C = 1)
  : C = Real.pi / 3 :=
sorry

-- Second part: Proving the values of a and b given c and the area
theorem sides_a_b
  (c : ℝ)
  (h2 : c = 2)
  (h3 : C = Real.pi / 3)
  (area : ℝ)
  (h4 : area = Real.sqrt 3)
  (h5 : 1/2 * a * b * Real.sin C = Real.sqrt 3)
  : a = 2 ∧ b = 2 :=
sorry

end angle_C_value_sides_a_b_l270_270898


namespace Mark_same_color_opposite_foot_l270_270927

variable (shoes : Finset (Σ _ : Fin (14), Bool))

def same_color_opposite_foot_probability (shoes : Finset (Σ _ : Fin (14), Bool)) : ℚ := 
  let total_shoes : ℚ := 28
  let num_black_pairs := 7
  let num_brown_pairs := 4
  let num_gray_pairs := 2
  let num_white_pairs := 1
  let black_pair_prob  := (14 / total_shoes) * (7 / (total_shoes - 1))
  let brown_pair_prob  := (8 / total_shoes) * (4 / (total_shoes - 1))
  let gray_pair_prob   := (4 / total_shoes) * (2 / (total_shoes - 1))
  let white_pair_prob  := (2 / total_shoes) * (1 / (total_shoes - 1))
  black_pair_prob + brown_pair_prob + gray_pair_prob + white_pair_prob

theorem Mark_same_color_opposite_foot (shoes : Finset (Σ _ : Fin (14), Bool)) :
  same_color_opposite_foot_probability shoes = 35 / 189 := 
sorry

end Mark_same_color_opposite_foot_l270_270927


namespace christina_payment_l270_270199

theorem christina_payment :
  let pay_flowers_per_flower := (8 : ℚ) / 3
  let pay_lawn_per_meter := (5 : ℚ) / 2
  let num_flowers := (9 : ℚ) / 4
  let area_lawn := (7 : ℚ) / 3
  let total_payment := pay_flowers_per_flower * num_flowers + pay_lawn_per_meter * area_lawn
  total_payment = 71 / 6 :=
by
  sorry

end christina_payment_l270_270199


namespace tan_half_sum_eq_third_l270_270558

theorem tan_half_sum_eq_third
  (x y : ℝ)
  (h1 : Real.cos x + Real.cos y = 3/5)
  (h2 : Real.sin x + Real.sin y = 1/5) :
  Real.tan ((x + y) / 2) = 1/3 :=
by sorry

end tan_half_sum_eq_third_l270_270558


namespace count_at_most_two_different_digits_l270_270699

def at_most_two_different_digits (n : ℕ) : Prop :=
  n < 100000 ∧ (∀ m < n.digits.length, ∀ p < n.digits.length, m ≠ p → n.digits.nth_le m ≠ n.digits.nth_le p) → 
  (∃ d1 d2 : ℕ, n.digits.all (λ d, d = d1 ∨ d = d2))

theorem count_at_most_two_different_digits :
  (nat.filter at_most_two_different_digits (list.range 100000)).length = 2034 :=
begin
  sorry
end

end count_at_most_two_different_digits_l270_270699


namespace y_expression_l270_270395

theorem y_expression (x y : ℝ) (h : 4 * x + y = 9) : y = 9 - 4 * x := 
by
  sorry

end y_expression_l270_270395


namespace gcd_15_70_l270_270837

theorem gcd_15_70 : Int.gcd 15 70 = 5 := by
  sorry

end gcd_15_70_l270_270837


namespace probability_multiple_of_3_l270_270324

def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

theorem probability_multiple_of_3 : 
  let total_tickets := 27
  let multiples_of_3 := { n : ℕ | 1 ≤ n ∧ n ≤ total_tickets ∧ is_multiple_of_3 n }
  let count_multiples_of_3 := multiplicative_group.orderOf multiples_of_3
in 
  (count_multiples_of_3 / total_tickets : ℚ) = 1 / 3 :=
sorry

end probability_multiple_of_3_l270_270324


namespace continuity_at_4_l270_270989

def f (x : ℝ) : ℝ := -2 * x^2 + 9

theorem continuity_at_4 : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 4| < δ → |f x + 23| < ε := by
  sorry

end continuity_at_4_l270_270989


namespace fraction_difference_l270_270817

-- Definitions for the problem conditions
def repeatingDecimal72 := 8 / 11
def decimal72 := 18 / 25

-- Statement that needs to be proven
theorem fraction_difference : repeatingDecimal72 - decimal72 = 2 / 275 := 
by 
  sorry

end fraction_difference_l270_270817


namespace min_m_min_expression_l270_270845

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Part (Ⅰ)
theorem min_m (m : ℝ) (h : ∃ x₀ : ℝ, f x₀ ≤ m) : m ≥ 2 := sorry

-- Part (Ⅱ)
theorem min_expression (a b : ℝ) (h1 : 3 * a + b = 2) (h2 : 0 < a) (h3 : 0 < b) :
  (1 / (2 * a) + 1 / (a + b)) ≥ 2 := sorry

end min_m_min_expression_l270_270845


namespace problem_statement_l270_270378

noncomputable def given_expression (x y z : ℝ) : ℝ :=
  (45 + (23 / 89) * Real.sin x) * (4 * y^2 - 7 * z^3)

theorem problem_statement : given_expression (Real.pi / 6) 3 (-2) = 4186 := by
  sorry

end problem_statement_l270_270378


namespace solve_for_x_l270_270944

theorem solve_for_x (x : ℚ) (h : 5 * x + 9 * x = 420 - 10 * (x - 4)) : 
  x = 115 / 6 :=
by
  sorry

end solve_for_x_l270_270944


namespace Allan_more_balloons_l270_270629

-- Define the number of balloons that Allan and Jake brought
def Allan_balloons := 5
def Jake_balloons := 3

-- Prove that the number of more balloons that Allan had than Jake is 2
theorem Allan_more_balloons : (Allan_balloons - Jake_balloons) = 2 := by sorry

end Allan_more_balloons_l270_270629


namespace meters_conversion_equivalence_l270_270996

-- Define the conditions
def meters_to_decimeters (m : ℝ) : ℝ := m * 10
def meters_to_centimeters (m : ℝ) : ℝ := m * 100

-- State the problem
theorem meters_conversion_equivalence :
  7.34 = 7 + (meters_to_decimeters 0.3) / 10 + (meters_to_centimeters 0.04) / 100 :=
sorry

end meters_conversion_equivalence_l270_270996


namespace repeating_decimal_difference_l270_270822

theorem repeating_decimal_difference :
  let x := 72 / 99 in
  let y := 72 / 100 in
  x - y = 2 / 275 := 
by
  -- we will add a proof later
  sorry

end repeating_decimal_difference_l270_270822


namespace missing_dimension_of_soap_box_l270_270783

theorem missing_dimension_of_soap_box 
  (volume_carton : ℕ) 
  (volume_soap_box : ℕ)
  (number_of_boxes : ℕ)
  (x : ℕ) 
  (h1 : volume_carton = 25 * 48 * 60) 
  (h2 : volume_soap_box = x * 6 * 5)
  (h3: number_of_boxes = 300)
  (h4 : number_of_boxes * volume_soap_box = volume_carton) : 
  x = 8 := by 
  sorry

end missing_dimension_of_soap_box_l270_270783


namespace part_one_l270_270526

theorem part_one (m : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = m * Real.exp x - x - 2) :
  (∀ x : ℝ, f x > 0) → m > Real.exp 1 :=
sorry

end part_one_l270_270526


namespace value_of_expression_l270_270066

theorem value_of_expression (x : ℝ) (h : x + 3 = 10) : 5 * x + 15 = 50 := by
  sorry

end value_of_expression_l270_270066


namespace sum_of_eight_smallest_multiples_of_12_l270_270978

theorem sum_of_eight_smallest_multiples_of_12 :
  let sum_n := (n : ℕ) → (n * (n + 1)) / 2
  12 * sum_n 8 = 432 :=
by
  sorry

end sum_of_eight_smallest_multiples_of_12_l270_270978


namespace evaluate_expression_l270_270846

theorem evaluate_expression : 101^3 + 3 * 101^2 * 2 + 3 * 101 * 2^2 + 2^3 = 1092727 := by
  sorry

end evaluate_expression_l270_270846


namespace and_false_iff_not_both_true_l270_270241

variable (p q : Prop)

theorem and_false_iff_not_both_true (h : ¬(p ∧ q)) : ¬p ∨ ¬q :=
by
    sorry

end and_false_iff_not_both_true_l270_270241


namespace becky_packs_lunch_days_l270_270131

-- Definitions of conditions
def school_days := 180
def aliyah_packing_fraction := 1 / 2
def becky_relative_fraction := 1 / 2

-- Derived quantities from conditions
def aliyah_pack_days := school_days * aliyah_packing_fraction
def becky_pack_days := aliyah_pack_days * becky_relative_fraction

-- Statement to prove
theorem becky_packs_lunch_days : becky_pack_days = 45 := by
  sorry

end becky_packs_lunch_days_l270_270131


namespace sin_3theta_over_sin_theta_l270_270217

theorem sin_3theta_over_sin_theta (θ : ℝ) (h : Real.tan θ = Real.sqrt 2) : 
  Real.sin (3 * θ) / Real.sin θ = 1 / 3 :=
by
  sorry

end sin_3theta_over_sin_theta_l270_270217


namespace inequality_proof_l270_270654

theorem inequality_proof (a b : ℝ) (h₀ : b > a) (h₁ : ab > 0) : 
  (1 / a > 1 / b) ∧ (a + b < 2 * b) :=
by
  sorry

end inequality_proof_l270_270654


namespace quadratic_has_distinct_real_roots_l270_270441

theorem quadratic_has_distinct_real_roots :
  ∃ (x y : ℝ), x ≠ y ∧ (x^2 - 3 * x - 1 = 0) ∧ (y^2 - 3 * y - 1 = 0) :=
by {
  sorry
}

end quadratic_has_distinct_real_roots_l270_270441


namespace negation_of_universal_proposition_l270_270068

variable (p : Prop)
variable (x : ℝ)

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - 1 > 0)) ↔ (∃ x : ℝ, x^2 - 1 ≤ 0) :=
by sorry

end negation_of_universal_proposition_l270_270068


namespace chef_dressing_total_volume_l270_270486

theorem chef_dressing_total_volume :
  ∀ (V1 V2 : ℕ) (P1 P2 : ℕ) (total_amount : ℕ),
    V1 = 128 →
    V2 = 128 →
    P1 = 8 →
    P2 = 13 →
    total_amount = V1 + V2 →
    total_amount = 256 :=
by
  intros V1 V2 P1 P2 total_amount hV1 hV2 hP1 hP2 h_total
  rw [hV1, hV2, add_comm, add_comm] at h_total
  exact h_total

end chef_dressing_total_volume_l270_270486


namespace compute_expression_l270_270922

theorem compute_expression (p q r : ℝ) 
  (h1 : p + q + r = 6) 
  (h2 : pq + qr + rp = 11) 
  (h3 : pqr = 12) : 
  (pq / r) + (qr / p) + (rp / q) = -23 / 12 := 
sorry

end compute_expression_l270_270922


namespace sam_total_money_spent_l270_270930

def value_of_pennies (n : ℕ) : ℝ := n * 0.01
def value_of_nickels (n : ℕ) : ℝ := n * 0.05
def value_of_dimes (n : ℕ) : ℝ := n * 0.10
def value_of_quarters (n : ℕ) : ℝ := n * 0.25

def total_money_spent : ℝ :=
  (value_of_pennies 5 + value_of_nickels 3) +  -- Monday
  (value_of_dimes 8 + value_of_quarters 4) +   -- Tuesday
  (value_of_nickels 7 + value_of_dimes 10 + value_of_quarters 2) +  -- Wednesday
  (value_of_pennies 20 + value_of_nickels 15 + value_of_dimes 12 + value_of_quarters 6) +  -- Thursday
  (value_of_pennies 45 + value_of_nickels 20 + value_of_dimes 25 + value_of_quarters 10)  -- Friday

theorem sam_total_money_spent : total_money_spent = 14.05 :=
by
  sorry

end sam_total_money_spent_l270_270930


namespace solution_l270_270329

noncomputable def problem : Prop :=
  (2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) = 1 / 2) ∧
  (1 - 2 * Real.sin (Real.pi / 12) ^ 2 ≠ 1 / 2) ∧
  (Real.cos (45 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) - 
   Real.sin (45 * Real.pi / 180) * Real.sin (15 * Real.pi / 180) = 1 / 2) ∧
  ( (Real.tan (77 * Real.pi / 180) - Real.tan (32 * Real.pi / 180)) /
    (2 * (1 + Real.tan (77 * Real.pi / 180) * Real.tan (32 * Real.pi / 180))) = 1 / 2 )

theorem solution : problem :=
  by 
    sorry

end solution_l270_270329


namespace boxes_per_case_l270_270719

theorem boxes_per_case (total_boxes : ℕ) (total_cases : ℕ) (h1 : total_boxes = 24) (h2 : total_cases = 3) : (total_boxes / total_cases) = 8 :=
by 
  sorry

end boxes_per_case_l270_270719


namespace find_x_l270_270245

theorem find_x (x : ℝ) (h : 0.5 * x = 0.05 * 500 - 20) : x = 10 :=
by
  sorry

end find_x_l270_270245


namespace range_of_m_l270_270750

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 3| - |x - 1| ≤ m^2 - 3 * m) ↔ m ≤ -1 ∨ m ≥ 4 :=
by
  sorry

end range_of_m_l270_270750


namespace range_of_m_l270_270400

-- Define the proposition
def P : Prop := ∀ x : ℝ, ∃ m : ℝ, 4^x - 2^(x + 1) + m = 0

-- Given that the negation of P is false
axiom neg_P_false : ¬¬P

-- Prove the range of m
theorem range_of_m : ∀ m : ℝ, (∀ x : ℝ, 4^x - 2^(x + 1) + m = 0) → m ≤ 1 :=
by
  sorry

end range_of_m_l270_270400


namespace twelve_xy_leq_fourx_1_y_9y_1_x_l270_270284

theorem twelve_xy_leq_fourx_1_y_9y_1_x
  (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) :=
  sorry

end twelve_xy_leq_fourx_1_y_9y_1_x_l270_270284


namespace prob_divisible_by_7_or_11_l270_270014

theorem prob_divisible_by_7_or_11 (m n : ℕ) (hm : m = 11) (hn : n = 50) :
  (m + n) = 61 :=
by {
  sorry
}

end prob_divisible_by_7_or_11_l270_270014


namespace slope_perpendicular_l270_270650

theorem slope_perpendicular (x y : ℝ) (h : 4 * x - 5 * y = 10) :
  let m := 4 / 5 in
  -(1 / m) = -5 / 4 :=
by {
  let m := 4 / 5,
  have h1 : 1 / m = 5 / 4 := by sorry,
  exact neg_eq_neg h1,
}

end slope_perpendicular_l270_270650


namespace cricket_initial_overs_l270_270262

-- Definitions based on conditions
def run_rate_initial : ℝ := 3.2
def run_rate_remaining : ℝ := 12.5
def target_runs : ℝ := 282
def remaining_overs : ℕ := 20

-- Mathematical statement to prove
theorem cricket_initial_overs (x : ℝ) (y : ℝ)
    (h1 : y = run_rate_initial * x)
    (h2 : y + run_rate_remaining * remaining_overs = target_runs) :
    x = 10 :=
sorry

end cricket_initial_overs_l270_270262


namespace mark_deposit_amount_l270_270278

-- Define the conditions
def bryans_deposit (M : ℝ) : ℝ := 5 * M - 40
def total_deposit (M : ℝ) : ℝ := M + bryans_deposit M

-- State the theorem
theorem mark_deposit_amount (M : ℝ) (h1: total_deposit M = 400) : M = 73.33 :=
by
  sorry

end mark_deposit_amount_l270_270278


namespace probability_20_correct_l270_270957

noncomputable def probability_sum_20_dodecahedral : ℚ :=
  let num_faces := 12
  let total_outcomes := num_faces * num_faces
  let favorable_outcomes := 5
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_20_correct : probability_sum_20_dodecahedral = 5 / 144 := 
by 
  sorry

end probability_20_correct_l270_270957


namespace min_bottles_l270_270765

theorem min_bottles (a b : ℕ) (h1 : a > b) (h2 : b > 1) : 
  ∃ x : ℕ, x = Nat.ceil (a - a / b) := sorry

end min_bottles_l270_270765


namespace intersection_M_N_l270_270235

def M : Set ℕ := {1, 2, 3, 4}
def N : Set ℕ := {x | x ≥ 3}

theorem intersection_M_N : M ∩ N = {3, 4} := 
by
  sorry

end intersection_M_N_l270_270235


namespace f_eight_l270_270398

noncomputable def f : ℝ → ℝ := sorry -- Defining the function without implementing it here

axiom f_x_neg {x : ℝ} (hx : x < 0) : f x = Real.log (-x) + x
axiom f_symmetric {x : ℝ} (hx : -Real.exp 1 ≤ x ∧ x ≤ Real.exp 1) : f (-x) = -f x
axiom f_periodic {x : ℝ} (hx : x > 1) : f (x + 2) = f x

theorem f_eight : f 8 = 2 - Real.log 2 := 
by
  sorry

end f_eight_l270_270398


namespace sin_max_value_l270_270436

open Real

theorem sin_max_value (a b : ℝ) (h : sin (a + b) = sin a + sin b) :
  sin a ≤ 1 :=
by
  sorry

end sin_max_value_l270_270436


namespace ratio_of_ages_l270_270938

-- Necessary conditions as definitions in Lean
def combined_age (S D : ℕ) : Prop := S + D = 54
def sam_is_18 (S : ℕ) : Prop := S = 18

-- The statement that we need to prove
theorem ratio_of_ages (S D : ℕ) (h1 : combined_age S D) (h2 : sam_is_18 S) : S / D = 1 / 2 := by
  sorry

end ratio_of_ages_l270_270938


namespace evaluate_f_at_3_l270_270872

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  x^7 + a * x^5 + b * x - 5

theorem evaluate_f_at_3 (a b : ℝ)
  (h : f (-3) a b = 5) : f 3 a b = -15 :=
by
  sorry

end evaluate_f_at_3_l270_270872


namespace washing_machine_regular_wash_l270_270187

variable {R : ℕ}

/-- A washing machine uses 20 gallons of water for a heavy wash,
2 gallons of water for a light wash, and an additional light wash
is added when bleach is used. Given conditions:
- Two heavy washes are done.
- Three regular washes are done.
- One light wash is done.
- Two loads are bleached.
- Total water used is 76 gallons.
Prove the washing machine uses 10 gallons of water for a regular wash. -/
theorem washing_machine_regular_wash (h : 2 * 20 + 3 * R + 1 * 2 + 2 * 2 = 76) : R = 10 :=
by
  sorry

end washing_machine_regular_wash_l270_270187


namespace correct_operation_l270_270191

theorem correct_operation (a b : ℝ) : (a * b) - 2 * (a * b) = - (a * b) :=
sorry

end correct_operation_l270_270191


namespace trapezium_area_l270_270855

def length_parallel_side1 : ℝ := 20
def length_parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 15
def expected_area : ℝ := 285

theorem trapezium_area :
  (1/2) * (length_parallel_side1 + length_parallel_side2) * distance_between_sides = expected_area :=
  sorry

end trapezium_area_l270_270855


namespace lowest_possible_students_l270_270179

theorem lowest_possible_students :
  ∃ n : ℕ, (n % 10 = 0 ∧ n % 24 = 0) ∧ n = 120 :=
by
  sorry

end lowest_possible_students_l270_270179


namespace number_of_students_l270_270596

-- Definitions based on conditions
def candy_bar_cost : ℝ := 2
def chips_cost : ℝ := 0.5
def total_cost_per_student : ℝ := candy_bar_cost + 2 * chips_cost
def total_amount : ℝ := 15

-- Statement to prove
theorem number_of_students : (total_amount / total_cost_per_student) = 5 :=
by
  sorry

end number_of_students_l270_270596


namespace joan_spent_on_jacket_l270_270555

def total_spent : ℝ := 42.33
def shorts_spent : ℝ := 15.00
def shirt_spent : ℝ := 12.51
def jacket_spent : ℝ := 14.82

theorem joan_spent_on_jacket :
  total_spent - shorts_spent - shirt_spent = jacket_spent :=
by
  sorry

end joan_spent_on_jacket_l270_270555


namespace number_of_hens_l270_270612

theorem number_of_hens (H C : ℕ) 
  (h1 : H + C = 60) 
  (h2 : 2 * H + 4 * C = 200) : H = 20 :=
sorry

end number_of_hens_l270_270612


namespace inequality_proof_l270_270285

variable (x y : ℝ)
variable (h1 : x ≥ 0)
variable (h2 : y ≥ 0)
variable (h3 : x + y ≤ 1)

theorem inequality_proof (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y ≤ 1) : 
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := 
by 
  sorry

end inequality_proof_l270_270285


namespace best_fit_slope_is_correct_l270_270983

open Real

noncomputable def slope_regression_line (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) :=
  (-2.5 * y1 - 1.5 * y2 + 0.5 * y3 + 3.5 * y4) / 21

theorem best_fit_slope_is_correct (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3) (h3 : x3 < x4)
  (h_arith : (x4 - x3 = 2 * (x3 - x2)) ∧ (x3 - x2 = 2 * (x2 - x1))) :
  slope_regression_line x1 x2 x3 x4 y1 y2 y3 y4 = (-2.5 * y1 - 1.5 * y2 + 0.5 * y3 + 3.5 * y4) / 21 := 
sorry

end best_fit_slope_is_correct_l270_270983


namespace area_smaller_part_l270_270493

theorem area_smaller_part (A B : ℝ) (h₁ : A + B = 500) (h₂ : B - A = (A + B) / 10) : A = 225 :=
by sorry

end area_smaller_part_l270_270493


namespace part1_part2_l270_270659

theorem part1 (a : ℝ) (ha : z = Complex.mk (a^2 - 7*a + 6) (a^2 - 5*a - 6))
  (h_imag : z.re = 0) : a = 1 :=
sorry

theorem part2 (a : ℝ) (ha : z = Complex.mk (a^2 - 7*a + 6) (a^2 - 5*a - 6))
  (h4thQuad : z.re > 0 ∧ z.im < 0) : -1 < a ∧ a < 1 :=
sorry

end part1_part2_l270_270659


namespace simplify_log_expression_eq_2_trig_expression_eq_neg_sqrt3_plus1_div_3_l270_270483

noncomputable def simplify_log_expression : ℝ :=
  log 27 (1/3)^(1/2) + log 10 25 + log 10 4 + 7^(-log 7 2) + (-0.98)^0

theorem simplify_log_expression_eq_2 : simplify_log_expression = 2 :=
  by sorry

variables (α : ℝ) (P : ℝ × ℝ)
def point_on_terminal_side (α : ℝ) (P : ℝ × ℝ) : Prop :=
  P.1 = real.sqrt 2 ∧ P.2 = -real.sqrt 6

theorem trig_expression_eq_neg_sqrt3_plus1_div_3
  (h : point_on_terminal_side α (real.sqrt 2, -real.sqrt 6)) :
  (cos (π / 2 + α) * cos (2π - α) + sin (-α - π / 2) * cos (π - α)) / (sin (π + α) * cos (π / 2 - α))
  = - (real.sqrt 3 + 1) / 3 :=
  by sorry

end simplify_log_expression_eq_2_trig_expression_eq_neg_sqrt3_plus1_div_3_l270_270483


namespace max_choir_members_l270_270749

theorem max_choir_members : 
  ∃ (m : ℕ), 
    (∃ k : ℕ, m = k^2 + 11) ∧ 
    (∃ n : ℕ, m = n * (n + 5)) ∧ 
    (∀ m' : ℕ, 
      ((∃ k' : ℕ, m' = k' * k' + 11) ∧ 
       (∃ n' : ℕ, m' = n' * (n' + 5))) → 
      m' ≤ 266) ∧ 
    m = 266 :=
by sorry

end max_choir_members_l270_270749


namespace find_b_l270_270752

noncomputable def P (x a b c : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + c

theorem find_b (a b c : ℝ) (h1: P 0 a b c = 12)
  (h2: (-c / 2) * 1 = -6)
  (h3: (2 + a + b + c) = -6)
  (h4: a + b + 14 = -6) : b = -56 :=
sorry

end find_b_l270_270752


namespace solution_of_loginequality_l270_270962

-- Define the conditions as inequalities
def condition1 (x : ℝ) : Prop := 2 * x - 1 > 0
def condition2 (x : ℝ) : Prop := -x + 5 > 0
def condition3 (x : ℝ) : Prop := 2 * x - 1 > -x + 5

-- Define the final solution set
def solution_set (x : ℝ) : Prop := (2 < x) ∧ (x < 5)

-- The theorem stating that under the given conditions, the solution set holds
theorem solution_of_loginequality (x : ℝ) : condition1 x ∧ condition2 x ∧ condition3 x → solution_set x :=
by
  intro h
  sorry

end solution_of_loginequality_l270_270962


namespace area_of_triangle_formed_by_medians_l270_270935

variable {a b c m_a m_b m_c Δ Δ': ℝ}

-- Conditions from the problem
axiom rel_sum_of_squares : m_a^2 + m_b^2 + m_c^2 = (3 / 4) * (a^2 + b^2 + c^2)
axiom rel_fourth_powers : m_a^4 + m_b^4 + m_c^4 = (9 / 16) * (a^4 + b^4 + c^4)

-- Statement of the problem to prove
theorem area_of_triangle_formed_by_medians :
  Δ' = (3 / 4) * Δ := sorry

end area_of_triangle_formed_by_medians_l270_270935


namespace count_two_digit_or_less_numbers_l270_270685

theorem count_two_digit_or_less_numbers : 
  let count_single_digit (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let count_two_digit (d₁ d₂ : ℕ) : Bool := (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (d₁ ≠ d₂)
  let count_two_digit_with_zero (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let nums_with_single_digit_count := 45
  let nums_with_exactly_two_digits_count := 1872
  let nums_with_two_digits_including_zero_count := 234
  let total_count := nums_with_single_digit_count + nums_with_exactly_two_digits_count + nums_with_two_digits_including_zero_count
  total_count = 2151 :=
by
  sorry

end count_two_digit_or_less_numbers_l270_270685


namespace permutations_mississippi_l270_270373

theorem permutations_mississippi : 
  let total_letters := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  (Nat.factorial total_letters / (Nat.factorial m_count * Nat.factorial i_count * Nat.factorial s_count * Nat.factorial p_count)) = 34650 := 
by
  sorry

end permutations_mississippi_l270_270373


namespace find_z_l270_270080

/- Definitions of angles and their relationships -/
def angle_sum_triangle (A B C : ℝ) : Prop := A + B + C = 180

/- Given conditions -/
def ABC : ℝ := 75
def BAC : ℝ := 55
def BCA : ℝ := 180 - ABC - BAC  -- This follows from the angle sum property of triangle ABC
def DCE : ℝ := BCA
def CDE : ℝ := 90

/- Prove z given the above conditions -/
theorem find_z : ∃ (z : ℝ), z = 90 - DCE := by
  use 40
  sorry

end find_z_l270_270080


namespace abs_nonneg_rational_l270_270994

theorem abs_nonneg_rational (a : ℚ) : |a| ≥ 0 :=
sorry

end abs_nonneg_rational_l270_270994


namespace prism_ratio_l270_270201

theorem prism_ratio (a b c d : ℝ) (h_d : d = 60) (h_c : c = 104) (h_b : b = 78 * Real.pi) (h_a : a = (4 * Real.pi) / 3) :
  b * c / (a * d) = 8112 / 240 := 
by 
  sorry

end prism_ratio_l270_270201


namespace selection_count_l270_270500

theorem selection_count :
  (Nat.choose 6 3) * (Nat.choose 5 2) = 200 := 
sorry

end selection_count_l270_270500


namespace problem_1_problem_2_l270_270616

-- Statements for our proof problems
theorem problem_1 (a b : ℝ) : a^2 + b^2 ≥ 2 * (2 * a - b) - 5 :=
sorry

theorem problem_2 (a b : ℝ) :
  a^a * b^b ≥ (a * b)^((a + b) / 2) ∧ (a = b ↔ a^a * b^b = (a * b)^((a + b) / 2)) :=
sorry

end problem_1_problem_2_l270_270616


namespace father_l270_270342

theorem father's_age (M F : ℕ) (h1 : M = 2 * F / 5) (h2 : M + 6 = (F + 6) / 2) : F = 30 :=
by
  sorry

end father_l270_270342


namespace shooting_enthusiast_l270_270345

variables {P : ℝ} -- Declare P as a real number

-- Define the conditions where X follows a binomial distribution
def binomial_variance (n : ℕ) (p : ℝ) :=
  n * p * (1 - p)

-- State the theorem in Lean 4
theorem shooting_enthusiast (h : binomial_variance 3 P = 3 / 4) : 
  P = 1 / 2 :=
by
  sorry -- Proof goes here

end shooting_enthusiast_l270_270345


namespace daily_harvest_l270_270120

theorem daily_harvest (sacks_per_section : ℕ) (num_sections : ℕ) 
  (h1 : sacks_per_section = 45) (h2 : num_sections = 8) : 
  sacks_per_section * num_sections = 360 :=
by
  sorry

end daily_harvest_l270_270120


namespace businesses_brandon_can_apply_to_l270_270563

-- Definitions of the given conditions in the problem
variables (x y : ℕ)

-- Define the total, fired, and quit businesses
def total_businesses : ℕ := 72
def fired_businesses : ℕ := 36
def quit_businesses : ℕ := 24

-- Define the unique businesses Brandon can still apply to, considering common businesses and reapplications
def businesses_can_apply_to : ℕ := (12 + x) + y

-- The theorem to prove
theorem businesses_brandon_can_apply_to (x y : ℕ) : businesses_can_apply_to x y = 12 + x + y := by
  unfold businesses_can_apply_to
  sorry

end businesses_brandon_can_apply_to_l270_270563


namespace polyhedron_faces_l270_270020

theorem polyhedron_faces (V E : ℕ) (F T P : ℕ) (h1 : F = 40) (h2 : V - E + F = 2) (h3 : T + P = 40) 
  (h4 : E = (3 * T + 4 * P) / 2) (h5 : V = (160 - T) / 2 - 38) (h6 : P = 3) (h7 : T = 1) :
  100 * P + 10 * T + V = 351 :=
by
  sorry

end polyhedron_faces_l270_270020


namespace repeating_decimal_difference_l270_270820

theorem repeating_decimal_difference :
  let x := 72 / 99 in
  let y := 72 / 100 in
  x - y = 2 / 275 := 
by
  -- we will add a proof later
  sorry

end repeating_decimal_difference_l270_270820


namespace economist_winning_strategy_l270_270481

-- Conditions setup
variables {n a b x1 x2 y1 y2 : ℕ}

-- Definitions according to the conditions
def valid_initial_division (n a b : ℕ) : Prop :=
  n > 4 ∧ n % 2 = 1 ∧ 2 ≤ a ∧ 2 ≤ b ∧ a + b = n ∧ a < b

def valid_further_division (a b x1 x2 y1 y2 : ℕ) : Prop :=
  x1 + x2 = a ∧ x1 ≥ 1 ∧ x2 ≥ 1 ∧ y1 + y2 = b ∧ y1 ≥ 1 ∧ y2 ≥ 1 ∧ x1 ≤ x2 ∧ y1 ≤ y2

-- Methods defined: Assumptions about which parts the economist takes
def method_1 (x1 x2 y1 y2 : ℕ) : ℕ :=
  max x2 y2 + min x1 y1

def method_2 (x1 x2 y1 y2 : ℕ) : ℕ :=
  (x1 + y1) / 2 + (x2 + y2) / 2

def method_3 (x1 x2 y1 y2 : ℕ) : ℕ :=
  max (method_1 x1 x2 y1 y2 - 1) (method_2 x1 x2 y1 y2 - 1) + 1

-- The statement to prove that the economist would choose method 1
theorem economist_winning_strategy :
  ∀ n a b x1 x2 y1 y2,
    valid_initial_division n a b →
    valid_further_division a b x1 x2 y1 y2 →
    n > 4 → n % 2 = 1 →
    (method_1 x1 x2 y1 y2) > (method_2 x1 x2 y1 y2) →
    (method_1 x1 x2 y1 y2) > (method_3 x1 x2 y1 y2) →
    method_1 x1 x2 y1 y2 = max (method_1 x1 x2 y1 y2) (method_2 x1 x2 y1 y2) :=
by
  -- Placeholder for the actual proof
  sorry

end economist_winning_strategy_l270_270481


namespace count_two_digit_or_less_numbers_l270_270687

theorem count_two_digit_or_less_numbers : 
  let count_single_digit (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let count_two_digit (d₁ d₂ : ℕ) : Bool := (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (d₁ ≠ d₂)
  let count_two_digit_with_zero (d : ℕ) : Bool := (1 ≤ d ∧ d ≤ 9)
  let nums_with_single_digit_count := 45
  let nums_with_exactly_two_digits_count := 1872
  let nums_with_two_digits_including_zero_count := 234
  let total_count := nums_with_single_digit_count + nums_with_exactly_two_digits_count + nums_with_two_digits_including_zero_count
  total_count = 2151 :=
by
  sorry

end count_two_digit_or_less_numbers_l270_270687


namespace alpha_beta_square_l270_270701

-- Statement of the problem in Lean 4
theorem alpha_beta_square :
  ∀ (α β : ℝ), (α ≠ β ∧ ∀ x : ℝ, x^2 - 2 * x - 1 = 0 → (x = α ∨ x = β)) → (α - β)^2 = 8 := 
by
  intros α β h
  sorry

end alpha_beta_square_l270_270701


namespace vasya_example_fewer_sevens_l270_270152

theorem vasya_example_fewer_sevens : 
  (777 / 7) - (77 / 7) = 100 :=
begin
  sorry
end

end vasya_example_fewer_sevens_l270_270152


namespace monotonic_increasing_m_ge_neg4_l270_270070

def is_monotonic_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ a → y > x → f y ≥ f x

def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * x - 2

theorem monotonic_increasing_m_ge_neg4 (m : ℝ) :
  is_monotonic_increasing (f m) 2 → m ≥ -4 :=
by
  sorry

end monotonic_increasing_m_ge_neg4_l270_270070


namespace reciprocal_of_neg_2023_l270_270453

theorem reciprocal_of_neg_2023 : (1 : ℝ) / (-2023) = -(1 / 2023) :=
by 
  sorry

end reciprocal_of_neg_2023_l270_270453


namespace boys_girls_relationship_l270_270016

theorem boys_girls_relationship (b g : ℕ) (h1 : b > 0) (h2 : g > 2) (h3 : ∀ n : ℕ, n < b → (n + 1) + 2 ≤ g) (h4 : b + 2 = g) : b = g - 2 := 
by
  sorry

end boys_girls_relationship_l270_270016


namespace kolya_sheets_exceed_500_l270_270428

theorem kolya_sheets_exceed_500 :
  ∃ k : ℕ, (10 + k * (k + 1) / 2 > 500) :=
sorry

end kolya_sheets_exceed_500_l270_270428


namespace find_x_l270_270700

theorem find_x (x : ℝ) (A B : Set ℝ) (hA : A = {1, 4, x}) (hB : B = {1, x^2}) (h_inter : A ∩ B = B) : x = -2 ∨ x = 2 ∨ x = 0 :=
sorry

end find_x_l270_270700


namespace notebook_problem_l270_270105

theorem notebook_problem
    (total_notebooks : ℕ)
    (cost_price_A : ℕ)
    (cost_price_B : ℕ)
    (total_cost_price : ℕ)
    (selling_price_A : ℕ)
    (selling_price_B : ℕ)
    (discount_A : ℕ)
    (profit_condition : ℕ)
    (x y m : ℕ) 
    (h1 : total_notebooks = 350)
    (h2 : cost_price_A = 12)
    (h3 : cost_price_B = 15)
    (h4 : total_cost_price = 4800)
    (h5 : selling_price_A = 20)
    (h6 : selling_price_B = 25)
    (h7 : discount_A = 30)
    (h8 : 12 * x + 15 * y = 4800)
    (h9 : x + y = 350)
    (h10 : selling_price_A * m + selling_price_B * m + (x - m) * selling_price_A * 7 / 10 + (y - m) * cost_price_B - total_cost_price ≥ profit_condition):
    x = 150 ∧ m ≥ 128 :=
by
    sorry

end notebook_problem_l270_270105


namespace inequality_xy_l270_270288

theorem inequality_xy {x y : ℝ} (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
  sorry

end inequality_xy_l270_270288


namespace min_value_fraction_8_l270_270059

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

end min_value_fraction_8_l270_270059


namespace find_fraction_l270_270065

theorem find_fraction (F : ℝ) (N : ℝ) (X : ℝ)
  (h1 : 0.85 * F = 36)
  (h2 : N = 70.58823529411765)
  (h3 : F = 42.35294117647059) :
  X * N = 42.35294117647059 → X = 0.6 :=
by
  sorry

end find_fraction_l270_270065


namespace find_x_l270_270175

theorem find_x (x : ℝ) (h : (x * 74) / 30 = 1938.8) : x = 786 := by
  sorry

end find_x_l270_270175


namespace outdoor_section_length_l270_270786

theorem outdoor_section_length (W : ℝ) (A : ℝ) (hW : W = 4) (hA : A = 24) : ∃ L : ℝ, A = W * L ∧ L = 6 := 
by
  use 6
  sorry

end outdoor_section_length_l270_270786


namespace circle_passing_through_points_eq_l270_270171

theorem circle_passing_through_points_eq :
  let A := (-2, 1)
  let B := (9, 3)
  let C := (1, 7)
  let center := (7/2, 2)
  let radius_sq := 125 / 4
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = radius_sq ↔ 
    (∃ t : ℝ, (x - center.1)^2 + (y - center.2)^2 = t^2) ∧
    ∀ P : ℝ × ℝ, P = A ∨ P = B ∨ P = C → (P.1 - center.1)^2 + (P.2 - center.2)^2 = radius_sq := by sorry

end circle_passing_through_points_eq_l270_270171


namespace anthony_solve_l270_270499

def completing_square (a b c : ℤ) : ℤ :=
  let d := Int.sqrt a
  let e := b / (2 * d)
  let f := (d * e * e - c)
  d + e + f

theorem anthony_solve (d e f : ℤ) (h_d_pos : d > 0)
  (h_eqn : 25 * d * d + 30 * d * e - 72 = 0)
  (h_form : (d * x + e)^2 = f) : 
  d + e + f = 89 :=
by
  have d : ℤ := 5
  have e : ℤ := 3
  have f : ℤ := 81
  sorry

end anthony_solve_l270_270499


namespace trip_duration_exactly_six_hours_l270_270624

theorem trip_duration_exactly_six_hours : 
  ∀ start_time end_time : ℕ,
  (start_time = (8 * 60 + 43 * 60 / 11)) ∧ 
  (end_time = (14 * 60 + 43 * 60 / 11)) → 
  (end_time - start_time) = 6 * 60 :=
by
  sorry

end trip_duration_exactly_six_hours_l270_270624


namespace count_special_integers_l270_270679

theorem count_special_integers : ∃ n : ℕ, n = 2151 ∧ 
  ∀ m : ℕ, (m < 100000 ∧ (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ 
    ((∀ k, k < 5 → m.digit k = d1 ∨ m.digit k = d2) ∨ 
    (∀ k, k < 5 → m.digit k = d1)))) → m ∈ finset.range 100000 :=
by sorry

end count_special_integers_l270_270679


namespace Lin_finishes_reading_on_Monday_l270_270277

theorem Lin_finishes_reading_on_Monday :
  let start_day := "Tuesday"
  let book_days : ℕ → ℕ := fun n => n
  let total_books := 10
  let total_days := (total_books * (total_books + 1)) / 2
  let days_in_a_week := 7
  let finish_day_offset := total_days % days_in_a_week
  let day_names := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  (day_names.indexOf start_day + finish_day_offset) % days_in_a_week = day_names.indexOf "Monday" :=
by
  sorry

end Lin_finishes_reading_on_Monday_l270_270277


namespace blue_marbles_difference_l270_270471

-- Definitions of the conditions
def total_green_marbles := 95

-- Ratios for Jar 1 and Jar 2
def ratio_blue_green_jar1 := (9, 1)
def ratio_blue_green_jar2 := (8, 1)

-- Total number of green marbles in each jar
def green_marbles_jar1 (a : ℕ) := a
def green_marbles_jar2 (b : ℕ) := b

-- Total number of marbles in each jar
def total_marbles_jar1 (a : ℕ) := 10 * a
def total_marbles_jar2 (b : ℕ) := 9 * b

-- Number of blue marbles in each jar
def blue_marbles_jar1 (a : ℕ) := 9 * a
def blue_marbles_jar2 (b : ℕ) := 8 * b

-- Conditions in terms of Lean definitions
theorem blue_marbles_difference:
  ∀ (a b : ℕ), green_marbles_jar1 a + green_marbles_jar2 b = total_green_marbles →
  total_marbles_jar1 a = total_marbles_jar2 b →
  blue_marbles_jar1 a - blue_marbles_jar2 b = 5 :=
by sorry

end blue_marbles_difference_l270_270471


namespace negation_of_proposition_l270_270108

variables (x : ℝ)

def proposition (x : ℝ) : Prop := x > 0 → (x ≠ 2 → (x^3 / (x - 2) > 0))

theorem negation_of_proposition : ∃ x : ℝ, x > 0 ∧ 0 ≤ x ∧ x ≤ 2 :=
by
  sorry

end negation_of_proposition_l270_270108


namespace compare_two_sqrt_three_l270_270841

theorem compare_two_sqrt_three : 2 > Real.sqrt 3 :=
by {
  sorry
}

end compare_two_sqrt_three_l270_270841


namespace repeatingDecimal_exceeds_l270_270835

noncomputable def repeatingDecimalToFraction (d : ℚ) : ℚ := 
    -- Function to convert repeating decimal to fraction
    if d = 0.99999 then 1 else (d * 100 - d) / 99  -- Simplified conversion for demonstration

def decimalToFraction (d : ℚ) : ℚ :=
    -- Function to convert decimal to fraction
    if d = 0.72 then 18 / 25 else 0  -- Replace with actual conversion

theorem repeatingDecimal_exceeds (x y : ℚ) (hx : repeatingDecimalToFraction x = 8/11) (hy : decimalToFraction y = 18/25):
    x - y = 2 / 275 :=
by
    sorry

end repeatingDecimal_exceeds_l270_270835


namespace total_cost_price_l270_270180

variables (C_table C_chair C_shelf : ℝ)

axiom h1 : 1.24 * C_table = 8091
axiom h2 : 1.18 * C_chair = 5346
axiom h3 : 1.30 * C_shelf = 11700

theorem total_cost_price :
  C_table + C_chair + C_shelf = 20055.51 :=
sorry

end total_cost_price_l270_270180


namespace part1_part2_l270_270397

def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) (m : ℝ) : ℝ := -|x + 3| + m
def h (x : ℝ) : ℝ := |x - 1| + |x + 3|

theorem part1 (x : ℝ) : f x + x^2 - 1 > 0 ↔ x > 1 ∨ x < 0 :=
by
  sorry

theorem part2 (m : ℝ) : (∃ x : ℝ, f x < g x m) ↔ m > 4 :=
by
  sorry

end part1_part2_l270_270397


namespace slope_of_tangent_at_4_l270_270315

def f (x : ℝ) : ℝ := x^3 - 7 * x^2 + 1

theorem slope_of_tangent_at_4 : (deriv f 4) = -8 := by
  sorry

end slope_of_tangent_at_4_l270_270315


namespace moles_of_Cl2_combined_l270_270647

theorem moles_of_Cl2_combined (nCH4 : ℕ) (nCl2 : ℕ) (nHCl : ℕ) 
  (h1 : nCH4 = 3) 
  (h2 : nHCl = nCl2) 
  (h3 : nHCl ≤ nCH4) : 
  nCl2 = 3 :=
by
  sorry

end moles_of_Cl2_combined_l270_270647


namespace number_of_blue_parrots_l270_270292

-- Defining the known conditions
def total_parrots : ℕ := 120
def fraction_red : ℚ := 2 / 3
def fraction_green : ℚ := 1 / 6

-- Proving the number of blue parrots given the conditions
theorem number_of_blue_parrots : (1 - (fraction_red + fraction_green)) * total_parrots = 20 := by
  sorry

end number_of_blue_parrots_l270_270292


namespace apples_and_pears_weight_l270_270967

theorem apples_and_pears_weight (apples pears : ℕ) 
    (h_apples : apples = 240) 
    (h_pears : pears = 3 * apples) : 
    apples + pears = 960 := 
  by
  sorry

end apples_and_pears_weight_l270_270967


namespace distance_between_stations_l270_270613

theorem distance_between_stations (x : ℕ) 
  (h1 : ∃ (x : ℕ), ∀ t : ℕ, (t * 16 = x ∧ t * 21 = x + 60)) :
  2 * x + 60 = 444 :=
by sorry

end distance_between_stations_l270_270613


namespace relationship_y_values_l270_270666

theorem relationship_y_values (m y1 y2 y3 : ℝ) 
  (h1 : y1 = -3 * (-3 : ℝ)^2 - 12 * (-3 : ℝ) + m)
  (h2 : y2 = -3 * (-2 : ℝ)^2 - 12 * (-2 : ℝ) + m)
  (h3 : y3 = -3 * (1 : ℝ)^2 - 12 * (1 : ℝ) + m) :
  y2 > y1 ∧ y1 > y3 :=
sorry

end relationship_y_values_l270_270666


namespace solve_equation_l270_270586

theorem solve_equation (x : ℝ) (h : x ≠ -2) : (x = -1/2) ↔ (x / (x + 2) + 1 = 1 / (x + 2)) :=
by
  sorry

end solve_equation_l270_270586


namespace tape_pieces_needed_l270_270489

-- Define the setup: cube edge length and tape width
def edge_length (n : ℕ) : ℕ := n
def tape_width : ℕ := 1

-- Define the statement we want to prove
theorem tape_pieces_needed (n : ℕ) (h₁ : edge_length n > 0) : 2 * n = 2 * (edge_length n) :=
  by
  sorry

end tape_pieces_needed_l270_270489


namespace length_of_goods_train_l270_270622

/-- The length of the goods train given the conditions of the problem --/
theorem length_of_goods_train
  (speed_passenger_train : ℝ) (speed_goods_train : ℝ) 
  (time_taken_to_pass : ℝ) (length_goods_train : ℝ) :
  speed_passenger_train = 80 / 3.6 →  -- Convert 80 km/h to m/s
  speed_goods_train    = 32 / 3.6 →  -- Convert 32 km/h to m/s
  time_taken_to_pass   = 9 →
  length_goods_train   = 280 → 
  length_goods_train = (speed_passenger_train + speed_goods_train) * time_taken_to_pass := by
    sorry

end length_of_goods_train_l270_270622


namespace hexagonal_prism_cross_section_l270_270473

theorem hexagonal_prism_cross_section (n : ℕ) (h₁: n ≥ 3) (h₂: n ≤ 8) : ¬ (n = 9):=
sorry

end hexagonal_prism_cross_section_l270_270473


namespace multiply_fractions_l270_270504

theorem multiply_fractions :
  (2 : ℚ) / 3 * 5 / 7 * 11 / 13 = 110 / 273 :=
by
  sorry

end multiply_fractions_l270_270504


namespace segments_not_arrangeable_l270_270911

theorem segments_not_arrangeable :
  ¬∃ (segments : ℕ → (ℝ × ℝ) × (ℝ × ℝ)), 
    (∀ i, 0 ≤ i → i < 1000 → 
      ∃ j, 0 ≤ j → j < 1000 → 
        i ≠ j ∧
        (segments i).fst.1 > (segments j).fst.1 ∧
        (segments i).fst.2 < (segments j).snd.2 ∧
        (segments i).snd.1 > (segments j).fst.1 ∧
        (segments i).snd.2 < (segments j).snd.2) :=
by
  sorry

end segments_not_arrangeable_l270_270911


namespace cos_identity_proof_l270_270298

noncomputable def cos_eq_half : Prop :=
  (Real.cos (Real.pi / 7) - Real.cos (2 * Real.pi / 7) + Real.cos (3 * Real.pi / 7)) = 1 / 2

theorem cos_identity_proof : cos_eq_half :=
  by sorry

end cos_identity_proof_l270_270298


namespace investment_time_l270_270956

theorem investment_time (P R diff : ℝ) (T : ℕ) 
  (hP : P = 1500)
  (hR : R = 0.10)
  (hdiff : diff = 15)
  (h1 : P * ((1 + R) ^ T - 1) - (P * R * T) = diff) 
  : T = 2 := 
by
  -- proof steps here
  sorry

end investment_time_l270_270956


namespace arrange_MISSISSIPPI_l270_270366

theorem arrange_MISSISSIPPI : 
  let total_letters := 11
  let count_I := 4
  let count_S := 4
  let count_P := 2
  let arrangements := 11.factorial / (count_I.factorial * count_S.factorial * count_P.factorial)
  in arrangements = 34650 :=
by
  have h1 : 11.factorial = 39916800 := by sorry
  have h2 : 4.factorial = 24 := by sorry
  have h3 : 2.factorial = 2 := by sorry
  have h4 : arrangements = 34650 := by sorry
  exact h4

end arrange_MISSISSIPPI_l270_270366


namespace fraction_difference_is_correct_l270_270800

-- Convert repeating decimal to fraction
def repeating_to_fraction : ℚ := 0.72 + 72 / 9900

-- Convert 0.72 to fraction
def decimal_to_fraction : ℚ := 72 / 100

-- Define the difference between the fractions
def fraction_difference := repeating_to_fraction - decimal_to_fraction

-- Final statement to prove
theorem fraction_difference_is_correct : fraction_difference = 2 / 275 := by
  sorry

end fraction_difference_is_correct_l270_270800


namespace polynomial_roots_identity_l270_270934

theorem polynomial_roots_identity {p q α β γ δ : ℝ} 
  (h1 : α^2 + p*α + 1 = 0)
  (h2 : β^2 + p*β + 1 = 0)
  (h3 : γ^2 + q*γ + 1 = 0)
  (h4 : δ^2 + q*δ + 1 = 0) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = q^2 - p^2 :=
sorry

end polynomial_roots_identity_l270_270934


namespace find_n_l270_270244

theorem find_n (n : ℕ) : 5 ^ 29 * 4 ^ 15 = 2 * 10 ^ n → n = 29 :=
by
  sorry

end find_n_l270_270244


namespace pedestrian_speeds_unique_l270_270970

variables 
  (x y : ℝ)
  (d : ℝ := 105)  -- Distance between cities
  (t1 : ℝ := 7.5) -- Time for current speeds
  (t2 : ℝ := 105 / 13) -- Time for adjusted speeds

theorem pedestrian_speeds_unique :
  (x + y = 14) →
  (3 * x + y = 14) →
  x = 6 ∧ y = 8 :=
by
  intros h1 h2
  have : 2 * x = 12 :=
    by ring_nf; sorry
  have hx : x = 6 :=
    by linarith
  have hy : y = 8 :=
    by linarith
  exact ⟨hx, hy⟩

end pedestrian_speeds_unique_l270_270970


namespace f_one_eq_minus_one_third_f_of_a_f_is_odd_l270_270054

noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (2^x + 1)

theorem f_one_eq_minus_one_third : f 1 = -1/3 := 
by sorry

theorem f_of_a (a : ℝ) : f a = (1 - 2^a) / (2^a + 1) := 
by sorry

theorem f_is_odd : ∀ x, f (-x) = -f x := by sorry

end f_one_eq_minus_one_third_f_of_a_f_is_odd_l270_270054


namespace withdraw_from_three_cards_probability_withdraw_all_four_l270_270467

namespace ThiefProblem

-- Definitions of the problem conditions
structure ProblemState where
  cards : Fin 4
  pin_codes : Fin 4
  attempts : Fin 4 → ℕ -- number of attempts made per card

-- Problem (a): Kirpich can always withdraw money from three cards
theorem withdraw_from_three_cards (s : ProblemState) : ∃ (cards_to_succeed : Nat), cards_to_succeed ≥ 3 :=
sorry

-- Problem (b): The probability of withdrawing money from all four cards is 23/24
theorem probability_withdraw_all_four : (23 : ℚ) / 24 =
  1 - ((1/4) * (1/3) * (1/2)) :=
sorry

end ThiefProblem

end withdraw_from_three_cards_probability_withdraw_all_four_l270_270467


namespace shaded_area_of_octagon_l270_270785

noncomputable def areaOfShadedRegion (s : ℝ) (r : ℝ) (theta : ℝ) : ℝ :=
  let n := 8
  let octagonArea := n * 0.5 * s^2 * (Real.sin (Real.pi/n) / Real.sin (Real.pi/(2 * n)))
  let sectorArea := n * 0.5 * r^2 * (theta / (2 * Real.pi))
  octagonArea - sectorArea

theorem shaded_area_of_octagon (h_s : 5 = 5) (h_r : 3 = 3) (h_theta : 45 = 45) :
  areaOfShadedRegion 5 3 (45 * (Real.pi / 180)) = 100 - 9 * Real.pi := by
  sorry

end shaded_area_of_octagon_l270_270785


namespace solution_l270_270703

theorem solution (y q : ℝ) (h1 : |y - 3| = q) (h2 : y < 3) : y - 2 * q = 3 - 3 * q :=
by
  sorry

end solution_l270_270703


namespace number_of_hens_l270_270332

variables (H C : ℕ)

def total_heads (H C : ℕ) : Prop := H + C = 48
def total_feet (H C : ℕ) : Prop := 2 * H + 4 * C = 144

theorem number_of_hens (H C : ℕ) (h1 : total_heads H C) (h2 : total_feet H C) : H = 24 :=
sorry

end number_of_hens_l270_270332


namespace can_be_divided_into_two_triangles_l270_270269

-- Definitions and properties of geometrical shapes
def is_triangle (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 3 ∧ vertices = 3

def is_pentagon (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 5 ∧ vertices = 5

def is_hexagon (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 6 ∧ vertices = 6

def is_heptagon (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 7 ∧ vertices = 7

-- The theorem we need to prove
theorem can_be_divided_into_two_triangles :
  ∀ sides vertices,
  (is_pentagon sides vertices → is_triangle sides vertices ∧ is_triangle sides vertices) ∧
  (is_hexagon sides vertices → is_triangle sides vertices ∧ is_triangle sides vertices) ∧
  (is_heptagon sides vertices → ¬ (is_triangle sides vertices ∧ is_triangle sides vertices)) :=
by sorry

end can_be_divided_into_two_triangles_l270_270269


namespace Eddie_number_divisibility_l270_270514

theorem Eddie_number_divisibility (n: ℕ) (h₁: n = 40) (h₂: n % 5 = 0): n % 2 = 0 := 
by
  sorry

end Eddie_number_divisibility_l270_270514


namespace angles_in_quadrilateral_l270_270540

theorem angles_in_quadrilateral (A B C D : ℝ)
    (h : A / B = 1 / 3 ∧ B / C = 3 / 5 ∧ C / D = 5 / 6)
    (sum_angles : A + B + C + D = 360) :
    A = 24 ∧ D = 144 := 
by
    sorry

end angles_in_quadrilateral_l270_270540


namespace max_sin_a_l270_270431

theorem max_sin_a (a b : ℝ) (h : sin (a + b) = sin a + sin b) : 
  sin a ≤ 1 :=
by
  sorry

end max_sin_a_l270_270431


namespace integer_values_sides_triangle_l270_270126

theorem integer_values_sides_triangle (x : ℝ) (hx_pos : x > 0) (hx1 : x + 15 > 40) (hx2 : x + 40 > 15) (hx3 : 15 + 40 > x) : 
    (∃ (n : ℤ), ∃ (hn : 0 < n) (hn1 : (n : ℝ) = x) (hn2 : 26 ≤ n) (hn3 : n ≤ 54), 
    ∀ (y : ℤ), (26 ≤ y ∧ y ≤ 54) → (∃ (m : ℤ), y = 26 + m ∧ m < 29 ∧ m ≥ 0)) := 
sorry

end integer_values_sides_triangle_l270_270126


namespace ratio_of_capitals_l270_270193

-- Variables for the capitals of Ashok and Pyarelal
variables (A P : ℕ)

-- Given conditions
def total_loss := 670
def pyarelal_loss := 603
def ashok_loss := total_loss - pyarelal_loss

-- Proof statement: the ratio of Ashok's capital to Pyarelal's capital
theorem ratio_of_capitals : ashok_loss * P = total_loss * pyarelal_loss - pyarelal_loss * P → A * pyarelal_loss = P * ashok_loss :=
by
  sorry

end ratio_of_capitals_l270_270193


namespace squared_difference_l270_270240

theorem squared_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 10) : (x - y)^2 = 24 :=
by
  sorry

end squared_difference_l270_270240


namespace sum_eight_smallest_multiples_of_12_l270_270977

theorem sum_eight_smallest_multiples_of_12 :
  let series := (List.range 8).map (λ k => 12 * (k + 1))
  series.sum = 432 :=
by
  sorry

end sum_eight_smallest_multiples_of_12_l270_270977


namespace repeating_seventy_two_exceeds_seventy_two_l270_270828

noncomputable def repeating_decimal (n d : ℕ) : ℚ := n / d

theorem repeating_seventy_two_exceeds_seventy_two :
  repeating_decimal 72 99 - (72 / 100) = (2 / 275) := 
sorry

end repeating_seventy_two_exceeds_seventy_two_l270_270828


namespace carA_catches_up_with_carB_at_150_km_l270_270359

-- Definitions representing the problem's conditions
variable (t_A t_B v_A v_B : ℝ)
variable (distance_A_B : ℝ := 300)
variable (time_diff_start : ℝ := 1)
variable (time_diff_end : ℝ := 1)

-- Assumptions representing the problem's conditions
axiom speed_carA : v_A = distance_A_B / t_A
axiom speed_carB : v_B = distance_A_B / (t_A + 2)
axiom time_relation : t_B = t_A + 2
axiom time_diff_starting : t_A = t_B - 2

-- The statement to be proven: car A catches up with car B 150 km from city B
theorem carA_catches_up_with_carB_at_150_km :
  ∃ t₀ : ℝ, v_A * t₀ = v_B * (t₀ + time_diff_start) ∧ (distance_A_B - v_A * t₀ = 150) :=
sorry

end carA_catches_up_with_carB_at_150_km_l270_270359


namespace sqrt_defined_iff_nonneg_l270_270257

theorem sqrt_defined_iff_nonneg (x : ℝ) : (∃ y, y = sqrt (x - 2)) ↔ x ≥ 2 :=
by
  sorry

end sqrt_defined_iff_nonneg_l270_270257


namespace percentage_increase_of_sides_l270_270746

noncomputable def percentage_increase_in_area (L W : ℝ) (p : ℝ) : ℝ :=
  let A : ℝ := L * W
  let L' : ℝ := L * (1 + p / 100)
  let W' : ℝ := W * (1 + p / 100)
  let A' : ℝ := L' * W'
  ((A' - A) / A) * 100

theorem percentage_increase_of_sides (L W : ℝ) (hL : L > 0) (hW : W > 0) :
    percentage_increase_in_area L W 20 = 44 :=
by
  sorry

end percentage_increase_of_sides_l270_270746


namespace sin_angle_add_pi_over_4_l270_270226

open Real

theorem sin_angle_add_pi_over_4 (α : ℝ) (h1 : (cos α = -3/5) ∧ (sin α = 4/5)) : sin (α + π / 4) = sqrt 2 / 10 :=
by
  sorry

end sin_angle_add_pi_over_4_l270_270226


namespace factorization_of_square_difference_l270_270850

variable (t : ℝ)

theorem factorization_of_square_difference : t^2 - 144 = (t - 12) * (t + 12) := 
sorry

end factorization_of_square_difference_l270_270850


namespace students_problem_count_l270_270322

theorem students_problem_count 
  (x y z q r : ℕ) 
  (H1 : x + y + z + q + r = 30) 
  (H2 : x + 2 * y + 3 * z + 4 * q + 5 * r = 40) 
  (h_y_pos : 1 ≤ y) 
  (h_z_pos : 1 ≤ z) 
  (h_q_pos : 1 ≤ q) 
  (h_r_pos : 1 ≤ r) : 
  x = 26 := 
  sorry

end students_problem_count_l270_270322


namespace compare_sums_of_square_roots_l270_270652

theorem compare_sums_of_square_roots
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (M : ℝ := Real.sqrt a + Real.sqrt b) 
  (N : ℝ := Real.sqrt (a + b)) :
  M > N :=
by
  sorry

end compare_sums_of_square_roots_l270_270652


namespace fraction_difference_l270_270813

-- Definitions for the problem conditions
def repeatingDecimal72 := 8 / 11
def decimal72 := 18 / 25

-- Statement that needs to be proven
theorem fraction_difference : repeatingDecimal72 - decimal72 = 2 / 275 := 
by 
  sorry

end fraction_difference_l270_270813


namespace determine_H_zero_l270_270127

theorem determine_H_zero (E F G H : ℕ) 
  (h1 : E < 10) (h2 : F < 10) (h3 : G < 10) (h4 : H < 10)
  (add_eq : 10 * E + F + 10 * G + E = 10 * H + E)
  (sub_eq : 10 * E + F - (10 * G + E) = E) : 
  H = 0 :=
sorry

end determine_H_zero_l270_270127


namespace mark_less_than_kate_and_laura_l270_270569

theorem mark_less_than_kate_and_laura (K : ℝ) (h : K + 2 * K + 3 * K + 4.5 * K = 360) :
  let Pat := 2 * K
  let Mark := 3 * K
  let Laura := 4.5 * K
  let Combined := K + Laura
  Mark - Combined = -85.72 :=
sorry

end mark_less_than_kate_and_laura_l270_270569


namespace total_hooligans_l270_270904

def hooligans_problem (X Y : ℕ) : Prop :=
  (X * Y = 365) ∧ (X + Y = 78 ∨ X + Y = 366)

theorem total_hooligans (X Y : ℕ) (h : hooligans_problem X Y) : X + Y = 78 ∨ X + Y = 366 :=
  sorry

end total_hooligans_l270_270904


namespace problem_proof_l270_270055

-- Definitions based on conditions
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

-- Theorem to prove
theorem problem_proof (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by
  sorry

end problem_proof_l270_270055


namespace ice_cream_ordering_ways_l270_270074

def number_of_cone_choices : ℕ := 2
def number_of_flavor_choices : ℕ := 4

theorem ice_cream_ordering_ways : number_of_cone_choices * number_of_flavor_choices = 8 := by
  sorry

end ice_cream_ordering_ways_l270_270074


namespace cost_of_outfit_l270_270791

theorem cost_of_outfit (P T J : ℝ) 
  (h1 : 4 * P + 8 * T + 2 * J = 2400)
  (h2 : 2 * P + 14 * T + 3 * J = 2400)
  (h3 : 3 * P + 6 * T = 1500) :
  P + 4 * T + J = 860 := 
sorry

end cost_of_outfit_l270_270791


namespace solve_ab_sum_l270_270538

theorem solve_ab_sum (x a b : ℝ) (ha : ℕ) (hb : ℕ)
  (h1 : a = ha)
  (h2 : b = hb)
  (h3 : x = a + Real.sqrt b)
  (h4 : x^2 + 3 * x + 3 / x + 1 / x^2 = 26) :
  (ha + hb = 5) :=
sorry

end solve_ab_sum_l270_270538


namespace negation_of_proposition_l270_270056

variable (x : ℝ)
variable (p : Prop)

def proposition : Prop := ∀ x > 0, (x + 1) * Real.exp x > 1

theorem negation_of_proposition : ¬ proposition ↔ ∃ x > 0, (x + 1) * Real.exp x ≤ 1 :=
by
  sorry

end negation_of_proposition_l270_270056


namespace change_in_spiders_l270_270018

theorem change_in_spiders 
  (x a y b : ℤ) 
  (h1 : x + a = 20) 
  (h2 : y + b = 23) 
  (h3 : x - b = 5) :
  y - a = 8 := 
by
  sorry

end change_in_spiders_l270_270018


namespace repeatingDecimal_exceeds_l270_270836

noncomputable def repeatingDecimalToFraction (d : ℚ) : ℚ := 
    -- Function to convert repeating decimal to fraction
    if d = 0.99999 then 1 else (d * 100 - d) / 99  -- Simplified conversion for demonstration

def decimalToFraction (d : ℚ) : ℚ :=
    -- Function to convert decimal to fraction
    if d = 0.72 then 18 / 25 else 0  -- Replace with actual conversion

theorem repeatingDecimal_exceeds (x y : ℚ) (hx : repeatingDecimalToFraction x = 8/11) (hy : decimalToFraction y = 18/25):
    x - y = 2 / 275 :=
by
    sorry

end repeatingDecimal_exceeds_l270_270836


namespace num_digits_less_than_100000_with_two_or_fewer_different_digits_l270_270689

theorem num_digits_less_than_100000_with_two_or_fewer_different_digits:
  let count_single_digit : Nat := 9 * 5
  let count_two_distinct_without_zero : Nat := let comb_9_2 : Nat := 9.choose 2
                                               comb_9_2 * (2 + 6 + 14 + 30)
  let count_two_distinct_with_zero : Nat := let num_ways_with_zero : Nat := 9 * (1 + 3 + 7 + 15)
                                            num_ways_with_zero
  let total_count : Nat := count_single_digit + count_two_distinct_without_zero + count_two_distinct_with_zero
  total_count = 2151 := by
  sorry

end num_digits_less_than_100000_with_two_or_fewer_different_digits_l270_270689


namespace time_to_run_home_l270_270722

-- Define the conditions
def blocks_run_per_time : ℚ := 2 -- Justin runs 2 blocks
def time_per_blocks : ℚ := 1.5 -- in 1.5 minutes
def blocks_to_home : ℚ := 8 -- Justin is 8 blocks from home

-- Define the theorem to prove the time taken for Justin to run home
theorem time_to_run_home : (blocks_to_home / blocks_run_per_time) * time_per_blocks = 6 :=
by
  sorry

end time_to_run_home_l270_270722


namespace determine_k_l270_270639

variable (x y z k : ℝ)

theorem determine_k
  (h1 : 9 / (x - y) = 16 / (z + y))
  (h2 : k / (x + z) = 16 / (z + y)) :
  k = 25 := by
  sorry

end determine_k_l270_270639


namespace field_dimension_solution_l270_270004

theorem field_dimension_solution (m : ℤ) (H1 : (3 * m + 11) * m = 100) : m = 5 :=
sorry

end field_dimension_solution_l270_270004


namespace catch_up_time_l270_270291

open Real

/-- Object A moves along a straight line with a velocity v_A(t) = 3t^2 + 1 (m/s),
object B is 5 meters ahead of A and moves with velocity v_B(t) = 10t (m/s).
Prove that the time (t in seconds) it takes for object A to catch up with object B
is t = 5.
-/
theorem catch_up_time :
  let v_A := fun t : ℝ => 3 * t^2 + 1,
      v_B := fun t : ℝ => 10 * t,
      dist_A := fun t : ℝ => (∫ s in 0..t, v_A s),
      dist_B := fun t : ℝ => (∫ s in 0..t, v_B s) + 5 in
  ∃ t : ℝ, dist_A t = dist_B t ∧ t = 5 :=
by 
  -- The proof will involve finding that when the distances are equal, then t = 5
  sorry

end catch_up_time_l270_270291


namespace unique_arrangements_mississippi_l270_270369

open scoped Nat

theorem unique_arrangements_mississippi : 
  let n := 11
  let n_M := 1
  let n_I := 4
  let n_S := 4
  let n_P := 2
  (n.fact / (n_M.fact * n_I.fact * n_S.fact * n_P.fact)) = 34650 :=
  by
  sorry

end unique_arrangements_mississippi_l270_270369


namespace excess_common_fraction_l270_270802

theorem excess_common_fraction :
  let x := (72 / 10^2 + 7.2 * (10 / 10^2) * (10^{-2} / (1 - 10^{-2}))) in
  x - (72 / 100) = 8 / 1100 :=
by
  sorry

end excess_common_fraction_l270_270802


namespace count_ordered_triples_l270_270887

open Nat

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b

theorem count_ordered_triples :
  {t : ℕ × ℕ × ℕ // lcm t.1 t.2.1 = 90 ∧ lcm t.1 t.2.2 = 450 ∧ lcm t.2.1 t.2.2 = 1350}.to_finset.card = 10 :=
by {
  sorry
}

end count_ordered_triples_l270_270887


namespace randys_trip_length_l270_270110

theorem randys_trip_length
  (trip_length : ℚ)
  (fraction_gravel : trip_length = (1 / 4) * trip_length)
  (middle_miles : 30 = (7 / 12) * trip_length)
  (fraction_dirt : trip_length = (1 / 6) * trip_length) :
  trip_length = 360 / 7 :=
by
  sorry

end randys_trip_length_l270_270110


namespace find_arithmetic_progression_areas_l270_270910

noncomputable def triangle_ABC : ℝ × ℝ × ℝ := (5, 8, 7)

def area_arithmetic_progression (AB BC AC : ℝ) (AOB_area AOC_area BOC_area : ℝ) : Prop :=
  (AOC_area = 10 * Real.sqrt 3 / 3) ∧ 
  (AOB_area = 5 * (AOC_area) / 7) ∧ 
  (BOC_area = 10 * Real.sqrt 3 - AOB_area - AOC_area) ∧ 
  (AOB_area, AOC_area, BOC_area).antisymm = [5 * Real.sqrt 3 / 21, 10 * Real.sqrt 3 / 3, 15 * Real.sqrt 3 / 21]

theorem find_arithmetic_progression_areas :
  ∃ (AOB_area AOC_area BOC_area : ℝ),
    area_arithmetic_progression 5 8 7 AOB_area AOC_area BOC_area :=
begin
  -- Here we can outline the proof steps if necessary
  sorry
end

end find_arithmetic_progression_areas_l270_270910


namespace allocate_to_Team_A_l270_270951

theorem allocate_to_Team_A (x : ℕ) :
  31 + x = 2 * (50 - x) →
  x = 23 :=
by
  sorry

end allocate_to_Team_A_l270_270951


namespace vasya_100_using_fewer_sevens_l270_270159

-- Definitions and conditions
def seven := 7

-- Theorem to prove
theorem vasya_100_using_fewer_sevens :
  (777 / seven - 77 / seven = 100) ∨
  (seven * seven + seven * seven + seven / seven + seven / seven = 100) :=
by
  sorry

end vasya_100_using_fewer_sevens_l270_270159


namespace find_abc_l270_270919

open Real

theorem find_abc {a b c : ℝ}
  (h1 : b + c = 16)
  (h2 : c + a = 17)
  (h3 : a + b = 18) :
  a * b * c = 606.375 :=
sorry

end find_abc_l270_270919


namespace triangle_acute_l270_270892

theorem triangle_acute
  (A B C : ℝ)
  (h_sum : A + B + C = 180)
  (h_ratio : A / B = 2 / 3 ∧ B / C = 3 / 4) :
  A < 90 ∧ B < 90 ∧ C < 90 :=
by
  -- proof goes here
  sorry

end triangle_acute_l270_270892


namespace smaller_number_of_two_digits_product_3774_l270_270595

theorem smaller_number_of_two_digits_product_3774 (a b : ℕ) (ha : 9 < a ∧ a < 100) (hb : 9 < b ∧ b < 100) (h : a * b = 3774) : a = 51 ∨ b = 51 :=
by
  sorry

end smaller_number_of_two_digits_product_3774_l270_270595


namespace find_missing_number_l270_270484

theorem find_missing_number (x : ℝ) (h : 1 / ((1 / 0.03) + (1 / x)) = 0.02775) : abs (x - 0.370) < 0.001 := by
  sorry

end find_missing_number_l270_270484


namespace probability_not_snowing_l270_270128

  -- Define the probability that it will snow tomorrow
  def P_snowing : ℚ := 2 / 5

  -- Define the probability that it will not snow tomorrow
  def P_not_snowing : ℚ := 1 - P_snowing

  -- Theorem stating the required proof
  theorem probability_not_snowing : P_not_snowing = 3 / 5 :=
  by 
    -- Proof would go here
    sorry
  
end probability_not_snowing_l270_270128


namespace reciprocals_expression_eq_zero_l270_270064

theorem reciprocals_expression_eq_zero {m n : ℝ} (h : m * n = 1) : (2 * m - 2 / n) * (1 / m + n) = 0 :=
by
  sorry

end reciprocals_expression_eq_zero_l270_270064


namespace yonderland_license_plates_l270_270348

/-!
# Valid License Plates in Yonderland

A valid license plate in Yonderland consists of three letters followed by four digits. 

We are tasked with determining the number of valid license plates possible under this format.
-/

def num_letters : ℕ := 26
def num_digits : ℕ := 10
def letter_combinations : ℕ := num_letters ^ 3
def digit_combinations : ℕ := num_digits ^ 4
def total_combinations : ℕ := letter_combinations * digit_combinations

theorem yonderland_license_plates : total_combinations = 175760000 := by
  sorry

end yonderland_license_plates_l270_270348


namespace polynomial_horner_form_operations_l270_270972

noncomputable def horner_eval (coeffs : List ℕ) (x : ℕ) : ℕ :=
  coeffs.foldr (fun a acc => a + acc * x) 0

theorem polynomial_horner_form_operations :
  let p := [1, 1, 2, 3, 4, 5]
  let x := 2
  horner_eval p x = ((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1 ∧
  (∀ x, x = 2 → (((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1 =  5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + 1 * x + 1)) ∧ 
  (∃ m a, m = 5 ∧ a = 5) := sorry

end polynomial_horner_form_operations_l270_270972


namespace intersection_of_sets_l270_270094

def setA : Set ℝ := { x | x^2 - 4*x + 3 < 0 }
def setB : Set ℝ := { x | 2*x - 3 > 0 }

theorem intersection_of_sets : setA ∩ setB = { x : ℝ | x > 3/2 ∧ x < 3 } :=
  by sorry

end intersection_of_sets_l270_270094


namespace percent_change_area_decrease_l270_270472

theorem percent_change_area_decrease (L W : ℝ) (hL : L > 0) (hW : W > 0) :
    let A_initial := L * W
    let L_new := 1.60 * L
    let W_new := 0.40 * W
    let A_new := L_new * W_new
    let percent_change := (A_new - A_initial) / A_initial * 100
    percent_change = -36 :=
by
  sorry

end percent_change_area_decrease_l270_270472


namespace quadratic_has_real_root_l270_270890

theorem quadratic_has_real_root {b : ℝ} :
  ∃ x : ℝ, x^2 + b*x + 25 = 0 ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_has_real_root_l270_270890


namespace int_product_negative_max_negatives_l270_270071

theorem int_product_negative_max_negatives (n : ℤ) (hn : n ≤ 9) (hp : n % 2 = 1) :
  ∃ m : ℤ, n + m = m ∧ m ≥ 0 :=
by
  use 9
  sorry

end int_product_negative_max_negatives_l270_270071


namespace broken_stick_triangle_probability_l270_270605

noncomputable def probability_of_triangle (x y z : ℕ) : ℚ := sorry

theorem broken_stick_triangle_probability :
  ∀ x y z : ℕ, (x < y + z ∧ y < x + z ∧ z < x + y) → probability_of_triangle x y z = 1 / 4 := 
by
  sorry

end broken_stick_triangle_probability_l270_270605


namespace sin_max_value_l270_270435

open Real

theorem sin_max_value (a b : ℝ) (h : sin (a + b) = sin a + sin b) :
  sin a ≤ 1 :=
by
  sorry

end sin_max_value_l270_270435


namespace probability_of_one_red_ball_is_one_third_l270_270264

-- Define the number of red and black balls
def red_balls : Nat := 2
def black_balls : Nat := 4
def total_balls : Nat := red_balls + black_balls

-- Define the probability calculation
def probability_red_ball : ℚ := red_balls / (red_balls + black_balls)

-- State the theorem
theorem probability_of_one_red_ball_is_one_third :
  probability_red_ball = 1 / 3 :=
by
  sorry

end probability_of_one_red_ball_is_one_third_l270_270264


namespace find_t_l270_270552

noncomputable def a_n (n : ℕ) : ℝ := 1 * (2 : ℝ)^(n-1)

noncomputable def S_3n (n : ℕ) : ℝ := (1 - (2 : ℝ)^(3 * n)) / (1 - 2)

noncomputable def a_n_cubed (n : ℕ) : ℝ := (a_n n)^3

noncomputable def T_n (n : ℕ) : ℝ := (1 - (a_n_cubed 2)^n) / (1 - (a_n_cubed 2))

theorem find_t (n : ℕ) : S_3n n = 7 * T_n n :=
by
  sorry

end find_t_l270_270552


namespace max_and_min_l270_270027

open Real

-- Define the function
def y (x : ℝ) : ℝ := 3 * x^4 - 6 * x^2 + 4

-- Define the interval
def a : ℝ := -1
def b : ℝ := 3

theorem max_and_min:
  ∃ (xmin xmax : ℝ), xmin = 1 ∧ xmax = 193 ∧
  (∀ x, a ≤ x ∧ x ≤ b → y(xmin) ≤ y x ∧ y x ≤ y(xmax)) :=
by
  sorry

end max_and_min_l270_270027


namespace repeating_seventy_two_exceeds_seventy_two_l270_270826

noncomputable def repeating_decimal (n d : ℕ) : ℚ := n / d

theorem repeating_seventy_two_exceeds_seventy_two :
  repeating_decimal 72 99 - (72 / 100) = (2 / 275) := 
sorry

end repeating_seventy_two_exceeds_seventy_two_l270_270826


namespace sin_alpha_cos_half_beta_minus_alpha_l270_270382

open Real

noncomputable def problem_condition (α β : ℝ) : Prop :=
  0 < α ∧ α < π / 2 ∧
  0 < β ∧ β < π / 2 ∧
  sin (π / 3 - α) = 3 / 5 ∧
  cos (β / 2 - π / 3) = 2 * sqrt 5 / 5

theorem sin_alpha (α β : ℝ) (h : problem_condition α β) : 
  sin α = (4 * sqrt 3 - 3) / 10 := sorry

theorem cos_half_beta_minus_alpha (α β : ℝ) (h : problem_condition α β) :
  cos (β / 2 - α) = 11 * sqrt 5 / 25 := sorry

end sin_alpha_cos_half_beta_minus_alpha_l270_270382


namespace banana_cost_l270_270374

/-- If 4 bananas cost $20, then the cost of one banana is $5. -/
theorem banana_cost (total_cost num_bananas : ℕ) (cost_per_banana : ℕ) 
  (h : total_cost = 20 ∧ num_bananas = 4) : cost_per_banana = 5 := by
  sorry

end banana_cost_l270_270374


namespace largest_additional_license_plates_l270_270196

theorem largest_additional_license_plates :
  let original_first_set := 5
  let original_second_set := 3
  let original_third_set := 4
  let original_total := original_first_set * original_second_set * original_third_set

  let new_set_case1 := original_first_set * (original_second_set + 2) * original_third_set
  let new_set_case2 := original_first_set * (original_second_set + 1) * (original_third_set + 1)

  let new_total := max new_set_case1 new_set_case2

  new_total - original_total = 40 :=
by
  let original_first_set := 5
  let original_second_set := 3
  let original_third_set := 4
  let original_total := original_first_set * original_second_set * original_third_set

  let new_set_case1 := original_first_set * (original_second_set + 2) * original_third_set
  let new_set_case2 := original_first_set * (original_second_set + 1) * (original_third_set + 1)

  let new_total := max new_set_case1 new_set_case2

  sorry

end largest_additional_license_plates_l270_270196


namespace car_catch_up_distance_l270_270358

-- Define the problem: Prove that Car A catches Car B at the specified location, given the conditions

theorem car_catch_up_distance
  (distance : ℝ := 300)
  (v_A v_B t_A : ℝ)
  (start_A_late : ℝ := 1) -- Car A leaves 1 hour later
  (arrive_A_early : ℝ := 1) -- Car A arrives 1 hour earlier
  (t : ℝ := 2) -- Time when Car A catches up with Car B
  (dist_eq_A : distance = v_A * t_A)
  (dist_eq_B : distance = v_B * (t_A + 2))
  (catch_up_time_eq : v_A * t = v_B * (t + 1)):
  (distance - v_A * t) = 150 := sorry

end car_catch_up_distance_l270_270358


namespace factor_expression_l270_270644

theorem factor_expression (x : ℝ) : 72 * x^5 - 90 * x^9 = -18 * x^5 * (5 * x^4 - 4) :=
by
  sorry

end factor_expression_l270_270644


namespace ratio_surface_area_volume_l270_270602

theorem ratio_surface_area_volume (a b : ℕ) (h1 : a^3 = 6 * b^2) (h2 : 6 * a^2 = 6 * b) : 
  (6 * a^2) / (b^3) = 7776 :=
by
  sorry

end ratio_surface_area_volume_l270_270602


namespace range_of_a_l270_270896

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x + a| + |x - 2| + a < 2010) ↔ a < 1006 :=
sorry

end range_of_a_l270_270896


namespace max_cookies_andy_can_eat_l270_270601

theorem max_cookies_andy_can_eat (A B C : ℕ) (hB_pos : B > 0) (hC_pos : C > 0) (hB : B ∣ A) (hC : C ∣ A) (h_sum : A + B + C = 36) :
  A ≤ 30 := by
  sorry

end max_cookies_andy_can_eat_l270_270601


namespace cherry_tomatoes_weight_l270_270544

def kilogram_to_grams (kg : ℕ) : ℕ := kg * 1000

theorem cherry_tomatoes_weight (kg_tomatoes : ℕ) (extra_tomatoes_g : ℕ) : kg_tomatoes = 2 → extra_tomatoes_g = 560 → kilogram_to_grams kg_tomatoes + extra_tomatoes_g = 2560 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end cherry_tomatoes_weight_l270_270544


namespace person_B_processes_components_l270_270107

theorem person_B_processes_components (x : ℕ) (h1 : ∀ x, x > 0 → x + 2 > 0) 
(h2 : ∀ x, x > 0 → (25 / (x + 2)) = (20 / x)) :
  x = 8 := sorry

end person_B_processes_components_l270_270107


namespace relation_y1_y2_y3_l270_270663

-- Definition of being on the parabola
def on_parabola (x : ℝ) (y m : ℝ) : Prop := y = -3*x^2 - 12*x + m

-- The conditions given in the problem
variables {y1 y2 y3 m : ℝ}

-- The points (-3, y1), (-2, y2), (1, y3) are on the parabola given by the equation
axiom h1 : on_parabola (-3) y1 m
axiom h2 : on_parabola (-2) y2 m
axiom h3 : on_parabola (1) y3 m

-- We need to prove the relationship between y1, y2, and y3
theorem relation_y1_y2_y3 : y2 > y1 ∧ y1 > y3 :=
by { sorry }

end relation_y1_y2_y3_l270_270663


namespace gcd_n_cube_plus_16_n_plus_4_l270_270211

theorem gcd_n_cube_plus_16_n_plus_4 (n : ℕ) (h1 : n > 16) : 
  Nat.gcd (n^3 + 16) (n + 4) = Nat.gcd 48 (n + 4) :=
by
  sorry

end gcd_n_cube_plus_16_n_plus_4_l270_270211


namespace Mike_watches_TV_every_day_l270_270929

theorem Mike_watches_TV_every_day :
  (∃ T : ℝ, 
  (3 * (T / 2) + 7 * T = 34) 
  → T = 4) :=
by
  let T := 4
  sorry

end Mike_watches_TV_every_day_l270_270929


namespace tom_has_1_dollar_left_l270_270469

/-- Tom has $19 and each folder costs $2. After buying as many folders as possible,
Tom will have $1 left. -/
theorem tom_has_1_dollar_left (initial_money : ℕ) (folder_cost : ℕ) (folders_bought : ℕ) (money_left : ℕ) 
  (h1 : initial_money = 19)
  (h2 : folder_cost = 2)
  (h3 : folders_bought = initial_money / folder_cost)
  (h4 : money_left = initial_money - folders_bought * folder_cost) :
  money_left = 1 :=
by
  -- proof will be provided here
  sorry

end tom_has_1_dollar_left_l270_270469


namespace LeRoy_should_pay_30_l270_270204

/-- Define the empirical amounts paid by LeRoy and Bernardo, and the total discount. -/
def LeRoy_paid : ℕ := 240
def Bernardo_paid : ℕ := 360
def total_discount : ℕ := 60

/-- Define total expenses pre-discount. -/
def total_expenses : ℕ := LeRoy_paid + Bernardo_paid

/-- Define total expenses post-discount. -/
def adjusted_expenses : ℕ := total_expenses - total_discount

/-- Define each person's adjusted share. -/
def each_adjusted_share : ℕ := adjusted_expenses / 2

/-- Define the amount LeRoy should pay Bernardo. -/
def leroy_to_pay : ℕ := each_adjusted_share - LeRoy_paid

/-- Prove that LeRoy should pay Bernardo $30 to equalize their expenses post-discount. -/
theorem LeRoy_should_pay_30 : leroy_to_pay = 30 :=
by 
  -- Proof goes here...
  sorry

end LeRoy_should_pay_30_l270_270204


namespace reduced_price_per_dozen_apples_l270_270185

variables (P R : ℝ) 

theorem reduced_price_per_dozen_apples (h₁ : R = 0.70 * P) 
  (h₂ : (30 / P + 54) * R = 30) :
  12 * R = 2 := 
sorry

end reduced_price_per_dozen_apples_l270_270185


namespace find_salary_B_l270_270313

def salary_A : ℕ := 8000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000
def avg_salary : ℕ := 8000

theorem find_salary_B (S_B : ℕ) :
  (salary_A + S_B + salary_C + salary_D + salary_E) / 5 = avg_salary ↔ S_B = 5000 := by
  sorry

end find_salary_B_l270_270313


namespace vendor_apples_sold_l270_270494

theorem vendor_apples_sold (x : ℝ) (h : 0.15 * (1 - x / 100) + 0.50 * (1 - x / 100) * 0.85 = 0.23) : x = 60 :=
sorry

end vendor_apples_sold_l270_270494


namespace dot_product_calculation_l270_270531

def vector := (ℤ × ℤ)

def dot_product (v1 v2 : vector) : ℤ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a : vector := (1, 3)
def b : vector := (-1, 2)

def scalar_mult (c : ℤ) (v : vector) : vector :=
  (c * v.1, c * v.2)

def vector_add (v1 v2 : vector) : vector :=
  (v1.1 + v2.1, v1.2 + v2.2)

theorem dot_product_calculation :
  dot_product (vector_add (scalar_mult 2 a) b) b = 15 := by
  sorry

end dot_product_calculation_l270_270531


namespace y2_minus_x2_l270_270879

theorem y2_minus_x2 (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) (h1 : 56 ≤ x + y) (h2 : x + y ≤ 59) (h3 : 9 < 10 * x) (h4 : 10 * x < 91 * y) : y^2 - x^2 = 177 :=
by
  sorry

end y2_minus_x2_l270_270879


namespace max_distance_point_circle_l270_270672

open Real

noncomputable def distance (P C : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)

theorem max_distance_point_circle :
  let C : ℝ × ℝ := (1, 2)
  let P : ℝ × ℝ := (3, 3)
  let r : ℝ := 2
  let max_distance : ℝ := distance P C + r
  ∃ M : ℝ × ℝ, distance P M = max_distance ∧ (M.1 - 1)^2 + (M.2 - 2)^2 = r^2 :=
by
  sorry

end max_distance_point_circle_l270_270672


namespace transport_tax_correct_l270_270272

-- Define the conditions
def car_horsepower : ℕ := 150
def tax_rate : ℕ := 20
def tax_period_months : ℕ := 8

-- Define the function to calculate the annual tax
def annual_transport_tax (horsepower : ℕ) (rate : ℕ) : ℕ :=
  horsepower * rate

-- Define the function to prorate the annual tax
def prorated_tax (annual_tax : ℕ) (months : ℕ) : ℕ :=
  (annual_tax * months) / 12

-- The proof problem: Prove the amount of transport tax Ivan needs to pay
theorem transport_tax_correct :
  let annual_tax := annual_transport_tax car_horsepower tax_rate in
  let prorated_tax := prorated_tax annual_tax tax_period_months in
  prorated_tax = 2000 :=
by 
  sorry

end transport_tax_correct_l270_270272


namespace dye_jobs_scheduled_l270_270577

noncomputable def revenue_from_haircuts (n : ℕ) : ℕ := n * 30
noncomputable def revenue_from_perms (n : ℕ) : ℕ := n * 40
noncomputable def revenue_from_dye_jobs (n : ℕ) : ℕ := n * (60 - 10)
noncomputable def total_revenue (haircuts perms dye_jobs : ℕ) (tips : ℕ) : ℕ :=
  revenue_from_haircuts haircuts + revenue_from_perms perms + revenue_from_dye_jobs dye_jobs + tips

theorem dye_jobs_scheduled : 
  (total_revenue 4 1 dye_jobs 50 = 310) → (dye_jobs = 2) := 
by
  sorry

end dye_jobs_scheduled_l270_270577


namespace first_three_decimal_digits_of_x_l270_270974

noncomputable def x : ℝ := (10^100 + 1) ^ (5 / 3)

theorem first_three_decimal_digits_of_x : (floor ((x - floor x) * 1000)) = 666 := 
sorry

end first_three_decimal_digits_of_x_l270_270974


namespace intersection_M_N_l270_270236

def M (x : ℝ) : Prop := (2 - x) / (x + 1) ≥ 0
def N (y : ℝ) : Prop := ∃ x : ℝ, y = Real.log x

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {y : ℝ | N y} = {x : ℝ | -1 < x ∧ x ≤ 2} := by
  sorry

end intersection_M_N_l270_270236


namespace smallest_angle_range_l270_270714

theorem smallest_angle_range {A B C : ℝ} (hA : 0 < A) (hABC : A + B + C = 180) (horder : A ≤ B ∧ B ≤ C) :
  0 < A ∧ A ≤ 60 := by
  sorry

end smallest_angle_range_l270_270714


namespace radius_of_larger_circle_l270_270115

theorem radius_of_larger_circle
  (r r_s : ℝ)
  (h1 : r_s = 2)
  (h2 : π * r^2 = 4 * π * r_s^2) :
  r = 4 :=
by
  sorry

end radius_of_larger_circle_l270_270115


namespace inequality_xy_l270_270289

theorem inequality_xy {x y : ℝ} (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
  sorry

end inequality_xy_l270_270289


namespace x_coordinate_of_P_l270_270079

noncomputable section

open Real

-- Define the standard properties of the parabola and point P
def parabola (p : ℝ) (x y : ℝ) := (y ^ 2 = 4 * x)

def distance (P F : ℝ × ℝ) : ℝ := 
  sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)

-- Position of the focus for the given parabola y^2 = 4x; Focus F(1, 0)
def focus : ℝ × ℝ := (1, 0)

-- The given conditions translated into Lean form
def on_parabola (x y : ℝ) := parabola 2 x y ∧ distance (x, y) focus = 5

-- The theorem we need to prove: If point P satisfies these conditions, then its x-coordinate is 4
theorem x_coordinate_of_P (P : ℝ × ℝ) (h : on_parabola P.1 P.2) : P.1 = 4 :=
by
  sorry

end x_coordinate_of_P_l270_270079


namespace max_largest_integer_l270_270069

theorem max_largest_integer (A B C D E : ℕ) (h₀ : A ≤ B) (h₁ : B ≤ C) (h₂ : C ≤ D) (h₃ : D ≤ E) 
(h₄ : (A + B + C + D + E) = 225) (h₅ : E - A = 10) : E = 215 :=
sorry

end max_largest_integer_l270_270069


namespace add_to_1_eq_62_l270_270327

theorem add_to_1_eq_62 :
  let y := 5 * 12 / (180 / 3)
  ∃ x, y + x = 62 ∧ x = 61 :=
by
  sorry

end add_to_1_eq_62_l270_270327


namespace vector_sum_magnitude_eq_2_or_5_l270_270532

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := 1
noncomputable def c : ℝ := 3
def equal_angles (θ : ℝ) := θ = 120 ∨ θ = 0

theorem vector_sum_magnitude_eq_2_or_5
  (a_mag : ℝ := a)
  (b_mag : ℝ := b)
  (c_mag : ℝ := c)
  (θ : ℝ)
  (Hθ : equal_angles θ) :
  (|a_mag| = 1) ∧ (|b_mag| = 1) ∧ (|c_mag| = 3) →
  (|a_mag + b_mag + c_mag| = 2 ∨ |a_mag + b_mag + c_mag| = 5) :=
by
  sorry

end vector_sum_magnitude_eq_2_or_5_l270_270532


namespace total_payment_correct_l270_270470

def payment_y : ℝ := 318.1818181818182
def payment_ratio : ℝ := 1.2
def payment_x : ℝ := payment_ratio * payment_y
def total_payment : ℝ := payment_x + payment_y

theorem total_payment_correct :
  total_payment = 700.00 :=
sorry

end total_payment_correct_l270_270470


namespace graphs_intersection_count_l270_270947

theorem graphs_intersection_count (g : ℝ → ℝ) (hg : Function.Injective g) :
  ∃ S : Finset ℝ, (∀ x ∈ S, g (x^3) = g (x^5)) ∧ S.card = 3 :=
by
  sorry

end graphs_intersection_count_l270_270947


namespace circle_symmetric_point_l270_270387

theorem circle_symmetric_point (a b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + a * x - 2 * y + b = 0 → x = 2 ∧ y = 1) ∧
  (∀ x y : ℝ, (x, y) ∈ { (px, py) | px = 2 ∧ py = 1 ∨ x + y - 1 = 0 } → x^2 + y^2 + a * x - 2 * y + b = 0) →
  a = 0 ∧ b = -3 := 
by {
  sorry
}

end circle_symmetric_point_l270_270387


namespace variance_of_numbers_l270_270521

noncomputable def variance (s : List ℕ) : ℚ :=
  let mean := (s.sum : ℚ) / s.length
  let sqDiffs := s.map (λ n => (n - mean) ^ 2)
  sqDiffs.sum / s.length

def avg_is_34 (s : List ℕ) : Prop := (s.sum : ℚ) / s.length = 34

theorem variance_of_numbers (x : ℕ) 
  (h : avg_is_34 [31, 38, 34, 35, x]) : variance [31, 38, 34, 35, x] = 6 := 
by
  sorry

end variance_of_numbers_l270_270521


namespace g_range_l270_270239

noncomputable section

-- Define the function f(x) = sin(ωx + φ)
def f (ω φ x : ℝ) := Real.sin (ω * x + φ)

-- Conditions
axiom omega_positive : ∀ (ω : ℝ), ω > 0
axiom phi_range : ∀ (φ : ℝ), 0 < φ ∧ φ < Real.pi / 2
axiom passes_through_point : ∀ (φ : ℝ), f 2 φ 0 = 1 / 2
axiom y_difference : ∀ (x1 y1 x2 y2 : ℝ), (f 2 (Real.pi / 6) x1 - f 2 (Real.pi / 6) x2).abs = 2
axiom min_x1_x2_distance : ∀ (x1 x2 : ℝ), (x1 - x2).abs = Real.pi / 2

-- In triangle ABC
axiom triangle_condition : ∀ (A B C a b c : ℝ), 
  2 * Real.sin A * Real.sin C + Real.cos (2 * B) = 1

-- Define the function g(B)
def g (B : ℝ) := sqrt 3 * (f 2 (Real.pi / 6) B) + f 2 (Real.pi / 6) (B + Real.pi / 4)

-- And its range
theorem g_range : ∀ B : ℝ, 0 < B ∧ B ≤ Real.pi / 3 → 0 ≤ g B ∧ g B ≤ 2 := sorry

end g_range_l270_270239


namespace jill_food_percentage_l270_270568

theorem jill_food_percentage (total_amount : ℝ) (tax_rate_clothing tax_rate_other_items spent_clothing_rate spent_other_rate spent_total_tax_rate : ℝ) : 
  spent_clothing_rate = 0.5 →
  spent_other_rate = 0.25 →
  tax_rate_clothing = 0.1 →
  tax_rate_other_items = 0.2 →
  spent_total_tax_rate = 0.1 →
  (spent_clothing_rate * tax_rate_clothing * total_amount) + (spent_other_rate * tax_rate_other_items * total_amount) = spent_total_tax_rate * total_amount →
  (1 - spent_clothing_rate - spent_other_rate) * total_amount / total_amount = 0.25 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end jill_food_percentage_l270_270568


namespace area_of_sector_l270_270419

def radius : ℝ := 5
def central_angle : ℝ := 2

theorem area_of_sector : (1 / 2) * radius^2 * central_angle = 25 := by
  sorry

end area_of_sector_l270_270419


namespace size_of_angle_C_l270_270519

theorem size_of_angle_C 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a = 5) 
  (h2 : b + c = 2 * a) 
  (h3 : 3 * Real.sin A = 5 * Real.sin B) : 
  C = 2 * Real.pi / 3 := 
sorry

end size_of_angle_C_l270_270519


namespace initial_blocks_l270_270189

variable (x : ℕ)

theorem initial_blocks (h : x + 30 = 65) : x = 35 := by
  sorry

end initial_blocks_l270_270189


namespace intersection_A_B_l270_270044

open Set

variable (l : ℝ)

def A := {x : ℝ | x > l}
def B := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_A_B (h₁ : l = 1) :
  A l ∩ B = {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l270_270044


namespace correct_product_l270_270633

theorem correct_product : 
  (0.0063 * 3.85 = 0.024255) :=
sorry

end correct_product_l270_270633


namespace smallest_number_with_55_divisors_l270_270033

theorem smallest_number_with_55_divisors :
  ∃ n : ℕ, (n = 3^4 * 2^{10}) ∧ (nat.count_divisors n = 55) :=
by
  have n : ℕ := 3^4 * 2^{10}
  exact ⟨n, ⟨rfl, nat.count_divisors_eq_count_divisors 3 4 2 10⟩⟩
  sorry

end smallest_number_with_55_divisors_l270_270033


namespace yan_distance_ratio_l270_270172

-- Define conditions
variable (x z w: ℝ)  -- x: distance from Yan to his home, z: distance from Yan to the school, w: Yan's walking speed
variable (h1: z / w = x / w + (x + z) / (5 * w))  -- Both choices require the same amount of time

-- The ratio of Yan's distance from his home to his distance from the school is 2/3
theorem yan_distance_ratio :
    x / z = 2 / 3 :=
by
  sorry

end yan_distance_ratio_l270_270172


namespace S9_equals_27_l270_270895

variables {a : ℕ → ℤ} {S : ℕ → ℤ} {d : ℤ}

-- (Condition 1) The sequence is an arithmetic sequence: a_{n+1} = a_n + d
axiom arithmetic_seq : ∀ n : ℕ, a (n + 1) = a n + d

-- (Condition 2) The sum S_n is the sum of the first n terms of the sequence
axiom sum_first_n_terms : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- (Condition 3) Given a_1 = 2 * a_3 - 3
axiom given_condition : a 1 = 2 * a 3 - 3

-- Prove that S_9 = 27
theorem S9_equals_27 : S 9 = 27 :=
by
  sorry

end S9_equals_27_l270_270895


namespace trapezium_area_l270_270854

def length_parallel_side1 : ℝ := 20
def length_parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 15
def expected_area : ℝ := 285

theorem trapezium_area :
  (1/2) * (length_parallel_side1 + length_parallel_side2) * distance_between_sides = expected_area :=
  sorry

end trapezium_area_l270_270854


namespace quadratic_range_and_value_l270_270229

theorem quadratic_range_and_value (k : ℝ) :
  (∃ x1 x2 : ℝ, (x1^2 + (2 * k - 1) * x1 + k^2 - 1 = 0) ∧
  (x2^2 + (2 * k - 1) * x2 + k^2 - 1 = 0)) →
  k ≤ 5 / 4 ∧ (∀ x1 x2 : ℝ, (x1^2 + (2 * k - 1) * x1 + k^2 - 1 = 0) ∧
  (x2^2 + (2 * k - 1) * x2 + k^2 - 1 = 0) ∧ (x1^2 + x2^2 = 16 + x1 * x2)) → k = -2 :=
by sorry

end quadratic_range_and_value_l270_270229


namespace cindy_same_color_probability_l270_270778

def box := ["red", "red", "red", "blue", "blue", "blue", "green", "yellow"]

def alice_draws (b : List String) : Finset (Finset String) := 
  Finset.powersetLen 3 (Finset.ofList b)

def bob_draws (a_draws b : List String) : Finset (Finset String) :=
  let remaining := b.filter (λ x, ¬ a_draws.contains x)
  Finset.powersetLen 2 (Finset.ofList remaining)

def cindy_draws (a_draws b_draws b : List String) : Finset (Finset String) :=
  let remaining := b.filter (λ x, ¬ a_draws.contains x ∧ ¬ b_draws.contains x)
  Finset.powersetLen 2 (Finset.ofList remaining)

def is_same_color (drawn : Finset String) : Bool :=
  ∀ x ∈ drawn, x = drawn.choose (by sorry)

theorem cindy_same_color_probability :
  let α := algebra_map (Fin ℕ) ℚ -- Probability ratio
  let a_draws := alice_draws box
  let favorable_ways := a_draws.sum (λ a, (bob_draws a box).sum (λ b, cindy_draws a b box).count (λ c, is_same_color c))
  let total_ways := a_draws.card * (bob_draws (a_draws.choose sorry) box).card * (cindy_draws (a_draws.choose sorry) ((bob_draws (a_draws.choose sorry) box).choose sorry) box).card
  α (favorable_ways) / α (total_ways) = 1 / 35 := 
sorry

end cindy_same_color_probability_l270_270778


namespace possible_values_of_ABCD_l270_270992

noncomputable def discriminant (a b c : ℕ) : ℕ :=
  b^2 - 4*a*c

theorem possible_values_of_ABCD 
  (A B C D : ℕ)
  (AB BC CD : ℕ)
  (hAB : AB = 10*A + B)
  (hBC : BC = 10*B + C)
  (hCD : CD = 10*C + D)
  (h_no_9 : A ≠ 9 ∧ B ≠ 9 ∧ C ≠ 9 ∧ D ≠ 9)
  (h_leading_nonzero : A ≠ 0)
  (h_quad1 : discriminant A B CD ≥ 0)
  (h_quad2 : discriminant A BC D ≥ 0)
  (h_quad3 : discriminant AB C D ≥ 0) :
  ABCD = 1710 ∨ ABCD = 1810 :=
sorry

end possible_values_of_ABCD_l270_270992


namespace roast_cost_l270_270916

-- Given conditions as described in the problem.
def initial_money : ℝ := 100
def cost_vegetables : ℝ := 11
def money_left : ℝ := 72
def total_spent : ℝ := initial_money - money_left

-- The cost of the roast that we need to prove. We expect it to be €17.
def cost_roast : ℝ := total_spent - cost_vegetables

-- The theorem that states the cost of the roast given the conditions.
theorem roast_cost :
  cost_roast = 100 - 72 - 11 := by
  -- skipping the proof steps with sorry
  sorry

end roast_cost_l270_270916


namespace compare_abc_l270_270865
open Real

noncomputable def a : ℝ := 0.1 * exp(0.1)
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - log 0.9

theorem compare_abc : c < a ∧ a < b := by
  sorry

end compare_abc_l270_270865


namespace original_number_of_cats_l270_270790

theorem original_number_of_cats (C : ℕ) : 
  (C - 600) / 2 = 600 → C = 1800 :=
by
  sorry

end original_number_of_cats_l270_270790


namespace absent_children_l270_270932

/-- On a school's annual day, sweets were to be equally distributed amongst 112 children. 
But on that particular day, some children were absent. Thus, the remaining children got 6 extra sweets. 
Each child was originally supposed to get 15 sweets. Prove that 32 children were absent. -/
theorem absent_children (A : ℕ) 
  (total_children : ℕ := 112) 
  (sweets_per_child : ℕ := 15) 
  (extra_sweets : ℕ := 6)
  (absent_eq : (total_children - A) * (sweets_per_child + extra_sweets) = total_children * sweets_per_child) : 
  A = 32 := 
by
  sorry

end absent_children_l270_270932


namespace fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l270_270143

theorem fewer_than_ten_sevens_example1 : (777 / 7) - (77 / 7) = 100 :=
  by sorry

theorem fewer_than_ten_sevens_example2 : (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7) = 100 :=
  by sorry

end fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l270_270143


namespace compare_constants_l270_270869

noncomputable def a : ℝ := 0.1 * real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - real.log 0.9

theorem compare_constants : c < a ∧ a < b :=
by sorry

end compare_constants_l270_270869


namespace find_n_l270_270487

theorem find_n 
  (molecular_weight : ℕ)
  (atomic_weight_Al : ℕ)
  (weight_OH : ℕ)
  (n : ℕ) 
  (h₀ : molecular_weight = 78)
  (h₁ : atomic_weight_Al = 27) 
  (h₂ : weight_OH = 17)
  (h₃ : molecular_weight = atomic_weight_Al + n * weight_OH) : 
  n = 3 := 
by 
  -- the proof is omitted
  sorry

end find_n_l270_270487


namespace find_a_find_b_find_p_find_k_l270_270067

-- SG. 1
theorem find_a (a : ℝ) : (∃ t : ℝ, 2 * a * t^2 + 12 * t + 9 = 0 /\ ∀ t : ℝ, 2 * a * t^2 + 12 * t + 9 = 0 -> t = -6 / a) -> a = 2 :=
sorry

-- SG. 2
theorem find_b (b a : ℝ) (h1 : a = 2) : (∀ x y : ℝ, a * x + b * y = 1 -> 4 * x + 18 * y = 3) -> b = 9 :=
sorry

-- SG. 3
noncomputable def nth_prime (n : ℕ) : ℕ := Nat.factor (#2 ..)
theorem find_p (p b : ℕ) (h1 : b = 9) : nth_prime b = p -> p = 23 :=
by {intro hp, rw [hp], simp [nth_prime]}

-- SG. 4
theorem find_k (k : ℝ) (θ : ℝ) : (k = (4 * sin θ + 3 * cos θ) / (2 * sin θ - cos θ)) ∧ (tan θ = 3) -> k = 3 :=
sorry

end find_a_find_b_find_p_find_k_l270_270067


namespace problem_statement_l270_270437

noncomputable def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 1}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 2}
noncomputable def complement_U (s : Set ℝ) : Set ℝ := {x | x ∈ U ∧ x ∉ s}
noncomputable def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

theorem problem_statement : intersection N (complement_U M) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end problem_statement_l270_270437


namespace hotel_room_assignment_even_hotel_room_assignment_odd_l270_270546

def smallest_n_even (k : ℕ) (m : ℕ) (h1 : k = 2 * m) : ℕ :=
  100 * (m + 1)

def smallest_n_odd (k : ℕ) (m : ℕ) (h1 : k = 2 * m + 1) : ℕ :=
  100 * (m + 1) + 1

theorem hotel_room_assignment_even (k m : ℕ) (h1 : k = 2 * m) :
  ∃ n, n = smallest_n_even k m h1 ∧ n >= 100 :=
  by
  sorry

theorem hotel_room_assignment_odd (k m : ℕ) (h1 : k = 2 * m + 1) :
  ∃ n, n = smallest_n_odd k m h1 ∧ n >= 100 :=
  by
  sorry

end hotel_room_assignment_even_hotel_room_assignment_odd_l270_270546


namespace gcd_n3_plus_16_n_plus_4_l270_270213

/-- For a given positive integer n > 2^4, the greatest common divisor of n^3 + 16 and n + 4 is 1. -/
theorem gcd_n3_plus_16_n_plus_4 (n : ℕ) (h : n > 2^4) : Nat.gcd (n^3 + 16) (n + 4) = 1 := by
  sorry

end gcd_n3_plus_16_n_plus_4_l270_270213


namespace remaining_thumbtacks_in_each_can_l270_270731

-- Definitions based on the conditions:
def total_thumbtacks : ℕ := 450
def num_cans : ℕ := 3
def thumbtacks_per_board_tested : ℕ := 1
def total_boards_tested : ℕ := 120

-- Lean 4 Statement

theorem remaining_thumbtacks_in_each_can :
  ∀ (initial_thumbtacks_per_can remaining_thumbtacks_per_can : ℕ),
  initial_thumbtacks_per_can = (total_thumbtacks / num_cans) →
  remaining_thumbtacks_per_can = (initial_thumbtacks_per_can - (thumbtacks_per_board_tested * total_boards_tested)) →
  remaining_thumbtacks_per_can = 30 :=
by
  sorry

end remaining_thumbtacks_in_each_can_l270_270731


namespace excess_common_fraction_l270_270804

theorem excess_common_fraction :
  let x := (72 / 10^2 + 7.2 * (10 / 10^2) * (10^{-2} / (1 - 10^{-2}))) in
  x - (72 / 100) = 8 / 1100 :=
by
  sorry

end excess_common_fraction_l270_270804


namespace intersection_of_sets_l270_270391

theorem intersection_of_sets :
  let A := {y : ℝ | ∃ x : ℝ, y = Real.sin x}
  let B := {y : ℝ | ∃ x : ℝ, y = Real.sqrt (-(x^2 - 4*x + 3))}
  A ∩ B = {y : ℝ | 0 ≤ y ∧ y ≤ 1} :=
by
  sorry

end intersection_of_sets_l270_270391


namespace bridge_weight_requirement_l270_270915

def weight_soda_can : ℕ := 12
def weight_empty_soda_can : ℕ := 2
def num_soda_cans : ℕ := 6

def weight_empty_other_can : ℕ := 3
def num_other_cans : ℕ := 2

def wind_force_eq_soda_cans : ℕ := 2

def total_weight_bridge_must_hold : ℕ :=
  weight_soda_can * num_soda_cans + weight_empty_soda_can * num_soda_cans +
  weight_empty_other_can * num_other_cans +
  wind_force_eq_soda_cans * (weight_soda_can + weight_empty_soda_can)

theorem bridge_weight_requirement :
  total_weight_bridge_must_hold = 118 :=
by
  unfold total_weight_bridge_must_hold weight_soda_can weight_empty_soda_can num_soda_cans
    weight_empty_other_can num_other_cans wind_force_eq_soda_cans
  sorry

end bridge_weight_requirement_l270_270915


namespace number_of_BMWs_sold_l270_270491

-- Defining the percentages of Mercedes, Toyota, and Acura cars sold
def percentageMercedes : ℕ := 18
def percentageToyota  : ℕ := 25
def percentageAcura   : ℕ := 15

-- Defining the total number of cars sold
def totalCars : ℕ := 250

-- The theorem to be proved
theorem number_of_BMWs_sold : (totalCars * (100 - (percentageMercedes + percentageToyota + percentageAcura)) / 100) = 105 := by
  sorry -- Proof to be filled in later

end number_of_BMWs_sold_l270_270491


namespace initial_investment_B_l270_270777

theorem initial_investment_B (A_initial : ℝ) (B : ℝ) (total_profit : ℝ) (A_profit : ℝ) 
(A_withdraw : ℝ) (B_advance : ℝ) : 
  A_initial = 3000 → B_advance = 1000 → A_withdraw = 1000 → total_profit = 756 → A_profit = 288 → 
  (8 * A_initial + 4 * (A_initial - A_withdraw)) / (8 * B + 4 * (B + B_advance)) = A_profit / (total_profit - A_profit) → 
  B = 4000 := 
by 
  intros h1 h2 h3 h4 h5 h6 
  sorry

end initial_investment_B_l270_270777


namespace geometric_sequence_general_term_l270_270906

noncomputable def general_term (n : ℕ) := (1 / 2) ^ (n - 1)

theorem geometric_sequence_general_term (a : ℕ → ℚ) (q : ℚ)
  (h_seq : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 + a 4 = 5 / 8)
  (h3 : a 3 = 1 / 4)
  (hq : q < 1) :
  ∀ n, a n = (1 / 2) ^ (n - 1) :=
by 
  sorry

end geometric_sequence_general_term_l270_270906


namespace distance_from_stream_to_meadow_l270_270425

noncomputable def distance_from_car_to_stream : ℝ := 0.2
noncomputable def distance_from_meadow_to_campsite : ℝ := 0.1
noncomputable def total_distance_hiked : ℝ := 0.7

theorem distance_from_stream_to_meadow : 
  (total_distance_hiked - distance_from_car_to_stream - distance_from_meadow_to_campsite = 0.4) :=
by
  sorry

end distance_from_stream_to_meadow_l270_270425


namespace egyptians_panamanians_l270_270758

-- Given: n + m = 12 and (n(n-1))/2 + (m(m-1))/2 = 31 and n > m
-- Prove: n = 7 and m = 5

theorem egyptians_panamanians (n m : ℕ) (h1 : n + m = 12) (h2 : n > m) 
(h3 : n * (n - 1) / 2 + m * (m - 1) / 2 = 31) :
  n = 7 ∧ m = 5 := 
by
  sorry

end egyptians_panamanians_l270_270758


namespace find_n_l270_270418

theorem find_n (x : ℝ) (hx : x > 0) (h : x / n + x / 25 = 0.24000000000000004 * x) : n = 5 :=
sorry

end find_n_l270_270418


namespace min_value_of_sum_l270_270926

noncomputable def min_value_x_3y (x y : ℝ) : ℝ :=
  x + 3 * y

theorem min_value_of_sum (x y : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0) 
  (cond : 1 / (x + 1) + 1 / (y + 1) = 1 / 2) :
  x + 3 * y ≥ 4 + 4 * Real.sqrt 3 :=
  sorry

end min_value_of_sum_l270_270926


namespace quadratic_real_roots_l270_270037

theorem quadratic_real_roots (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2 * x + 1 = 0) ↔ (a ≤ 1 ∧ a ≠ 0) :=
by
  sorry

end quadratic_real_roots_l270_270037


namespace bertha_daughters_no_daughters_l270_270195

theorem bertha_daughters_no_daughters (daughters granddaughters: ℕ) (no_great_granddaughters: granddaughters = 5 * daughters) (total_women: 8 + granddaughters = 48) :
  8 + granddaughters = 48 :=
by {
  sorry
}

end bertha_daughters_no_daughters_l270_270195


namespace solution_of_system_l270_270512

noncomputable def system_of_equations (x y : ℝ) :=
  x = 1.12 * y + 52.8 ∧ x = y + 50

theorem solution_of_system : 
  ∃ (x y : ℝ), system_of_equations x y ∧ y = -23.33 ∧ x = 26.67 :=
by
  sorry

end solution_of_system_l270_270512


namespace radius_of_circle_from_chord_and_line_l270_270265

theorem radius_of_circle_from_chord_and_line (r : ℝ) (t θ : ℝ) 
    (param_line : ℝ × ℝ) (param_circle : ℝ × ℝ)
    (chord_length : ℝ) 
    (h1 : param_line = (3 + 3 * t, 1 - 4 * t))
    (h2 : param_circle = (r * Real.cos θ, r * Real.sin θ))
    (h3 : chord_length = 4) 
    : r = Real.sqrt 13 :=
sorry

end radius_of_circle_from_chord_and_line_l270_270265


namespace jason_initial_quarters_l270_270718

theorem jason_initial_quarters (q_d q_n q_i : ℕ) (h1 : q_d = 25) (h2 : q_n = 74) :
  q_i = q_n - q_d → q_i = 49 :=
by
  sorry

end jason_initial_quarters_l270_270718


namespace inequality_solution_set_l270_270048

theorem inequality_solution_set (a b c : ℝ)
  (h1 : ∀ x, (ax^2 + bx + c > 0 ↔ -3 < x ∧ x < 2)) :
  (a < 0) ∧ (a + b + c > 0) ∧ (∀ x, ¬ (bx + c > 0 ↔ x > 6)) ∧ (∀ x, (cx^2 + bx + a < 0 ↔ -1/3 < x ∧ x < 1/2)) :=
by
  sorry

end inequality_solution_set_l270_270048


namespace total_people_l270_270900

-- Define the conditions as constants
def B : ℕ := 50
def S : ℕ := 70
def B_inter_S : ℕ := 20

-- Total number of people in the group
theorem total_people : B + S - B_inter_S = 100 := by
  sorry

end total_people_l270_270900


namespace correct_calculation_l270_270609

theorem correct_calculation (a b : ℝ) :
  (6 * a - 5 * a ≠ 1) ∧
  (a + 2 * a^2 ≠ 3 * a^3) ∧
  (- (a - b) = -a + b) ∧
  (2 * (a + b) ≠ 2 * a + b) :=
by 
  sorry

end correct_calculation_l270_270609


namespace Justin_run_home_time_l270_270723

variable (blocksPerMinute : ℝ) (totalBlocks : ℝ)

theorem Justin_run_home_time (h1 : blocksPerMinute = 2 / 1.5) (h2 : totalBlocks = 8) :
  totalBlocks / blocksPerMinute = 6 := by
  sorry

end Justin_run_home_time_l270_270723


namespace parts_processed_per_hour_l270_270476

theorem parts_processed_per_hour (x : ℕ) (y : ℕ) (h1 : y = x + 10) (h2 : 150 / y = 120 / x) :
  x = 40 ∧ y = 50 :=
by {
  sorry
}

end parts_processed_per_hour_l270_270476


namespace quadrilateral_segments_condition_l270_270625

-- Define the lengths and their conditions
variables {a b c d : ℝ}

-- Define the main theorem with necessary and sufficient conditions
theorem quadrilateral_segments_condition (h_sum : a + b + c + d = 1.5)
    (h_order : a ≤ b) (h_order2 : b ≤ c) (h_order3 : c ≤ d) (h_ratio : d ≤ 3 * a) :
    (a ≥ 0.25 ∧ d < 0.75) ↔ (a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a) :=
by {
  sorry -- proof is omitted
}

end quadrilateral_segments_condition_l270_270625


namespace repeatingDecimal_exceeds_l270_270831

noncomputable def repeatingDecimalToFraction (d : ℚ) : ℚ := 
    -- Function to convert repeating decimal to fraction
    if d = 0.99999 then 1 else (d * 100 - d) / 99  -- Simplified conversion for demonstration

def decimalToFraction (d : ℚ) : ℚ :=
    -- Function to convert decimal to fraction
    if d = 0.72 then 18 / 25 else 0  -- Replace with actual conversion

theorem repeatingDecimal_exceeds (x y : ℚ) (hx : repeatingDecimalToFraction x = 8/11) (hy : decimalToFraction y = 18/25):
    x - y = 2 / 275 :=
by
    sorry

end repeatingDecimal_exceeds_l270_270831


namespace solve_inequality_l270_270444

theorem solve_inequality (a x : ℝ) :
  ((x - a) * (x - 2 * a) < 0) ↔ 
  ((a < 0 ∧ 2 * a < x ∧ x < a) ∨ (a = 0 ∧ false) ∨ (a > 0 ∧ a < x ∧ x < 2 * a)) :=
by sorry

end solve_inequality_l270_270444


namespace number_of_clerks_l270_270955

theorem number_of_clerks 
  (num_officers : ℕ) 
  (num_clerks : ℕ) 
  (avg_salary_staff : ℕ) 
  (avg_salary_officers : ℕ) 
  (avg_salary_clerks : ℕ)
  (h1 : avg_salary_staff = 90)
  (h2 : avg_salary_officers = 600)
  (h3 : avg_salary_clerks = 84)
  (h4 : num_officers = 2)
  : num_clerks = 170 :=
sorry

end number_of_clerks_l270_270955


namespace minimum_value_4x_minus_y_l270_270668

theorem minimum_value_4x_minus_y (x y : ℝ) (h1 : x - y ≥ 0) (h2 : x + y - 4 ≥ 0) (h3 : x ≤ 4) :
  ∃ (m : ℝ), m = 6 ∧ ∀ (x' y' : ℝ), (x' - y' ≥ 0) → (x' + y' - 4 ≥ 0) → (x' ≤ 4) → 4 * x' - y' ≥ m :=
by
  sorry

end minimum_value_4x_minus_y_l270_270668


namespace find_m_n_sum_product_l270_270667

noncomputable def sum_product_of_roots (m n : ℝ) : Prop :=
  (m^2 - 4*m - 12 = 0) ∧ (n^2 - 4*n - 12 = 0) 

theorem find_m_n_sum_product (m n : ℝ) (h : sum_product_of_roots m n) :
  m + n + m * n = -8 :=
by 
  sorry

end find_m_n_sum_product_l270_270667


namespace min_value_expr_l270_270559

theorem min_value_expr (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (hxyz : x * y * z = 1) :
  2 * x^2 + 8 * x * y + 6 * y^2 + 16 * y * z + 3 * z^2 ≥ 24 :=
by
  sorry

end min_value_expr_l270_270559


namespace total_number_of_balls_l270_270177

theorem total_number_of_balls 
(b : ℕ) (P_blue : ℚ) (h1 : b = 8) (h2 : P_blue = 1/3) : 
  ∃ g : ℕ, b + g = 24 := by
  sorry

end total_number_of_balls_l270_270177


namespace ab2c_value_l270_270661

theorem ab2c_value (a b c : ℚ) (h₁ : |a + 1| + (b - 2)^2 = 0) (h₂ : |c| = 3) :
  a + b + 2 * c = 7 ∨ a + b + 2 * c = -5 := sorry

end ab2c_value_l270_270661


namespace part1_part2_l270_270276

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 2| + |x - a|

-- Prove part 1: For all x in ℝ, log(f(x, -8)) ≥ 1
theorem part1 : ∀ x : ℝ, Real.log (f x (-8)) ≥ 1 :=
by 
  sorry

-- Prove part 2: For all x in ℝ, if f(x,a) ≥ a, then a ≤ 1
theorem part2 (a : ℝ) : (∀ x : ℝ, f x a ≥ a) → a ≤ 1 :=
by
  sorry

end part1_part2_l270_270276


namespace cdf_of_Z_pdf_of_Z_l270_270674

noncomputable def f1 (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 2 then 0.5 else 0

noncomputable def f2 (y : ℝ) : ℝ :=
  if 0 < y ∧ y < 2 then 0.5 else 0

noncomputable def G (z : ℝ) : ℝ :=
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z^2 / 8
  else if 2 < z ∧ z ≤ 4 then 1 - (4 - z)^2 / 8
  else 1

noncomputable def g (z : ℝ) : ℝ :=
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z / 4
  else if 2 < z ∧ z ≤ 4 then 1 - z / 4
  else 0

theorem cdf_of_Z (z : ℝ) : G z = 
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z^2 / 8
  else if 2 < z ∧ z ≤ 4 then 1 - (4 - z)^2 / 8
  else 1 := sorry

theorem pdf_of_Z (z : ℝ) : g z = 
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z / 4
  else if 2 < z ∧ z ≤ 4 then 1 - z / 4
  else 0 := sorry

end cdf_of_Z_pdf_of_Z_l270_270674


namespace trapezium_area_proof_l270_270853

-- Define the lengths of the parallel sides and the distance between them
def a : ℝ := 20
def b : ℝ := 18
def h : ℝ := 15

-- Define the area of the trapezium
def area_of_trapezium (a b h : ℝ) : ℝ := (1/2) * (a + b) * h

-- State the theorem to be proved
theorem trapezium_area_proof : area_of_trapezium a b h = 285 := by
  sorry

end trapezium_area_proof_l270_270853


namespace domain_of_function_l270_270202

theorem domain_of_function (x : ℝ) : (|x - 2| + |x + 2| ≠ 0) := 
sorry

end domain_of_function_l270_270202


namespace range_of_m_l270_270876

-- Definitions of propositions
def is_circle (m : ℝ) : Prop :=
  ∃ x y : ℝ, (x - m)^2 + y^2 = 2 * m - m^2 ∧ 2 * m - m^2 > 0

def is_hyperbola_eccentricity_in_interval (m : ℝ) : Prop :=
  1 < Real.sqrt (1 + m / 5) ∧ Real.sqrt (1 + m / 5) < 2

-- Proving the main statement
theorem range_of_m (m : ℝ) (h1 : is_circle m ∨ is_hyperbola_eccentricity_in_interval m)
  (h2 : ¬ (is_circle m ∧ is_hyperbola_eccentricity_in_interval m)) : 2 ≤ m ∧ m < 15 :=
sorry

end range_of_m_l270_270876


namespace like_terms_sum_l270_270062

theorem like_terms_sum (m n : ℕ) (a b : ℝ) 
  (h₁ : 5 * a^m * b^3 = 5 * a^m * b^3) 
  (h₂ : -4 * a^2 * b^(n-1) = -4 * a^2 * b^(n-1)) 
  (h₃ : m = 2) (h₄ : 3 = n - 1) : m + n = 6 := by
  sorry

end like_terms_sum_l270_270062


namespace function_increasing_and_decreasing_intervals_l270_270883

noncomputable def f (x : ℝ) : ℝ := x ^ 3 - (1 / 3) * x ^ 2 - (1 / 2) * x

theorem function_increasing_and_decreasing_intervals :
  let a := 1 / 3
  let b := -1 / 2
  ∃ f : ℝ → ℝ, 
    (∀ x : ℝ, f x = x ^ 3 - 3 * a * x ^ 2 + 2 * b * x) ∧ 
    f 1 = -1 ∧ 
    f' 1 = 0 ∧
    ((∀ x < -1 / 3, f' x > 0) ∧ 
     (∀ x > 1, f' x > 0) ∧ 
     (∀ x ∈ (-1 / 3, 1), f' x < 0)) := 
by 
  sorry

end function_increasing_and_decreasing_intervals_l270_270883


namespace expected_value_counter_l270_270003

theorem expected_value_counter :
  let E (n : ℕ) : ℚ := 1 - (1 / (2 ^ n))
  let m := 1023
  let n := 1024
  assert : E 10 = (1023 / 1024)
  assert : Nat.gcd m n = 1
  100 * m + n = 103324 :=
by sorry

end expected_value_counter_l270_270003


namespace characterization_of_M_l270_270091

noncomputable def M : Set ℂ := {z : ℂ | (z - 1) ^ 2 = Complex.abs (z - 1) ^ 2}

theorem characterization_of_M : M = {z : ℂ | ∃ r : ℝ, z = r} :=
by
  sorry

end characterization_of_M_l270_270091


namespace number_of_proper_subsets_l270_270524

-- Define the universal set A and prove that the number of proper subsets of A is 6
theorem number_of_proper_subsets (A : Finset ℕ) (h : A = {0, 1, 2}) : A.card = 3 ∧ (A.powerset.card - 2) = 6 :=
by
  sorry

end number_of_proper_subsets_l270_270524


namespace seth_initial_boxes_l270_270941

-- Definitions based on conditions:
def remaining_boxes_after_giving_half (initial_boxes : ℕ) : ℕ :=
  let boxes_after_giving_to_mother := initial_boxes - 1
  let remaining_boxes := boxes_after_giving_to_mother / 2
  remaining_boxes

-- Main problem statement to prove.
theorem seth_initial_boxes (initial_boxes : ℕ) (remaining_boxes : ℕ) :
  remaining_boxes_after_giving_half initial_boxes = remaining_boxes ->
  remaining_boxes = 4 ->
  initial_boxes = 9 := 
by
  intros h1 h2
  sorry

end seth_initial_boxes_l270_270941


namespace flush_probability_l270_270606

noncomputable def probability_flush : ℚ :=
  4 * Nat.choose 13 5 / Nat.choose 52 5

theorem flush_probability :
  probability_flush = 33 / 16660 :=
by
  sorry

end flush_probability_l270_270606


namespace problem_solved_prob_l270_270325

theorem problem_solved_prob (pA pB : ℝ) (HA : pA = 1 / 3) (HB : pB = 4 / 5) :
  ((1 - (1 - pA) * (1 - pB)) = 13 / 15) :=
by
  sorry

end problem_solved_prob_l270_270325


namespace sufficient_but_not_necessary_condition_l270_270311

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 3 → x^2 - 2 * x > 0) ∧ ¬ (x^2 - 2 * x > 0 → x > 3) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l270_270311


namespace women_in_business_class_l270_270942

theorem women_in_business_class 
  (total_passengers : ℕ) 
  (percent_women : ℝ) 
  (percent_women_in_business : ℝ) 
  (H1 : total_passengers = 300)
  (H2 : percent_women = 0.70)
  (H3 : percent_women_in_business = 0.08) : 
  ∃ (num_women_business_class : ℕ), num_women_business_class = 16 := 
by
  sorry

end women_in_business_class_l270_270942


namespace distinct_arrays_for_48_chairs_with_conditions_l270_270002

theorem distinct_arrays_for_48_chairs_with_conditions : 
  ∃ n : ℕ, n = 7 ∧ 
    ∀ (m r c : ℕ), 
      m = 48 ∧ 
      2 ≤ r ∧ 
      2 ≤ c ∧ 
      r * c = m ↔ 
      (∃ (k : ℕ), 
         ((k = (m / r) ∧ r * (m / r) = m) ∨ (k = (m / c) ∧ c * (m / c) = m)) ∧ 
         r * c = m) → 
    n = 7 :=
by
  sorry

end distinct_arrays_for_48_chairs_with_conditions_l270_270002


namespace rectangle_perimeter_is_104_l270_270009

noncomputable def perimeter_of_rectangle (b : ℝ) (h1 : b > 0) (h2 : 3 * b * b = 507) : ℝ :=
  2 * (3 * b) + 2 * b

theorem rectangle_perimeter_is_104 {b : ℝ} (h1 : b > 0) (h2 : 3 * b * b = 507) :
  perimeter_of_rectangle b h1 h2 = 104 :=
by
  sorry

end rectangle_perimeter_is_104_l270_270009


namespace repeating_decimal_difference_l270_270824

theorem repeating_decimal_difference :
  let x := 72 / 99 in
  let y := 72 / 100 in
  x - y = 2 / 275 := 
by
  -- we will add a proof later
  sorry

end repeating_decimal_difference_l270_270824


namespace find_base_17_digit_l270_270224

theorem find_base_17_digit (a : ℕ) (h1 : 0 ≤ a ∧ a < 17) 
  (h2 : (25 + a) % 16 = 0) : a = 7 :=
sorry

end find_base_17_digit_l270_270224


namespace fewerSevensCanProduce100_l270_270150

noncomputable def validAs100UsingSevens : (ℕ → ℕ) → Prop := 
  fun (f : ℕ → ℕ) => ∃ (x y : ℕ), (fewer_than_ten_sevens x y) ∧ f x y = 100

theorem fewerSevensCanProduce100 : 
  ∃ (f : ℕ → ℕ), validAs100UsingSevens f :=
by 
  sorry

end fewerSevensCanProduce100_l270_270150


namespace system_of_linear_eq_l270_270330

theorem system_of_linear_eq :
  ∃ (x y : ℝ), x + y = 5 ∧ y = 2 :=
sorry

end system_of_linear_eq_l270_270330


namespace locus_centers_of_tangent_circles_l270_270747

theorem locus_centers_of_tangent_circles (a b : ℝ) :
  (x^2 + y^2 = 1) ∧ ((x - 1)^2 + (y -1)^2 = 81) →
  (a^2 + b^2 - (2 * a * b) / 63 - (66 * a) / 63 - (66 * b) / 63 + 17 = 0) :=
by
  sorry

end locus_centers_of_tangent_circles_l270_270747


namespace map_area_ratio_l270_270129

theorem map_area_ratio (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  ¬ ((l * w) / ((500 * l) * (500 * w)) = 1 / 500) :=
by
  -- The proof will involve calculations showing the true ratio is 1/250000
  sorry

end map_area_ratio_l270_270129


namespace multiply_fractions_l270_270505

theorem multiply_fractions :
  (2 : ℚ) / 3 * 5 / 7 * 11 / 13 = 110 / 273 :=
by
  sorry

end multiply_fractions_l270_270505


namespace min_total_cost_at_n_equals_1_l270_270587

-- Define the conditions and parameters
variables (a : ℕ) -- The total construction area
variables (n : ℕ) -- The number of floors

-- Definitions based on the given problem conditions
def land_expropriation_cost : ℕ := 2388 * a
def construction_cost (n : ℕ) : ℕ :=
  if n = 0 then 0 else if n = 1 then 455 * a else (455 * n * a + 30 * (n-2) * (n-1) / 2 * a)

-- Total cost including land expropriation and construction costs
def total_cost (n : ℕ) : ℕ := land_expropriation_cost a + construction_cost a n

-- The minimum total cost occurs at n = 1
theorem min_total_cost_at_n_equals_1 :
  ∃ n, n = 1 ∧ total_cost a n = 2788 * a :=
by sorry

end min_total_cost_at_n_equals_1_l270_270587


namespace female_officers_count_l270_270478

theorem female_officers_count
  (total_on_duty : ℕ)
  (on_duty_females : ℕ)
  (total_female_officers : ℕ)
  (h1 : total_on_duty = 240)
  (h2 : on_duty_females = total_on_duty / 2)
  (h3 : on_duty_females = (40 * total_female_officers) / 100) : 
  total_female_officers = 300 := 
by
  sorry

end female_officers_count_l270_270478


namespace percentage_of_students_passed_l270_270550

theorem percentage_of_students_passed
  (students_failed : ℕ)
  (total_students : ℕ)
  (H_failed : students_failed = 260)
  (H_total : total_students = 400)
  (passed := total_students - students_failed) :
  (passed * 100 / total_students : ℝ) = 35 := 
by
  -- proof steps would go here
  sorry

end percentage_of_students_passed_l270_270550


namespace fewerSevensCanProduce100_l270_270146

noncomputable def validAs100UsingSevens : (ℕ → ℕ) → Prop := 
  fun (f : ℕ → ℕ) => ∃ (x y : ℕ), (fewer_than_ten_sevens x y) ∧ f x y = 100

theorem fewerSevensCanProduce100 : 
  ∃ (f : ℕ → ℕ), validAs100UsingSevens f :=
by 
  sorry

end fewerSevensCanProduce100_l270_270146


namespace simplify_neg_cube_square_l270_270585

theorem simplify_neg_cube_square (a : ℝ) : (-a^3)^2 = a^6 :=
by
  sorry

end simplify_neg_cube_square_l270_270585


namespace quadratic_coefficients_l270_270732

theorem quadratic_coefficients (b c : ℝ) :
  (∀ x : ℝ, |x + 4| = 3 ↔ x^2 + bx + c = 0) → (b = 8 ∧ c = 7) :=
by
  sorry

end quadratic_coefficients_l270_270732


namespace distance_between_points_l270_270081

theorem distance_between_points :
  let A : ℝ × ℝ × ℝ := (1, -2, 3)
  let B : ℝ × ℝ × ℝ := (1, 2, 3)
  let C : ℝ × ℝ × ℝ := (1, 2, -3)
  dist B C = 6 :=
by
  sorry

end distance_between_points_l270_270081


namespace count_valid_n_l270_270497

theorem count_valid_n:
  ( ∃ f: ℕ → ℕ, ∀ n, (0 < n ∧ n < 2012 → 7 ∣ (2^n - n^2) ↔ 7 ∣ (f n)) ∧ f 2012 = 576) → 
  ∃ valid_n_count: ℕ, valid_n_count = 576 := 
sorry

end count_valid_n_l270_270497


namespace num_integers_two_digits_l270_270696

theorem num_integers_two_digits (n : ℕ) (n < 100000) : 
  ∃ k, k = 2151 ∧ 
  ∀ x, (x < n ∧ (∃ d₁ d₂, ∀ y, y ∈ x.digits 10 → y = d₁ ∨ y = d₂)) → k = 2151 := sorry

end num_integers_two_digits_l270_270696


namespace linear_function_equality_l270_270949

theorem linear_function_equality (f : ℝ → ℝ) (hf : ∀ x, f (3 * (f x)⁻¹ + 5) = f x)
  (hf1 : f 1 = 5) : f 2 = 3 :=
sorry

end linear_function_equality_l270_270949


namespace least_number_divisible_l270_270766

theorem least_number_divisible (n : ℕ) (h1 : n % 7 = 4) (h2 : n % 9 = 4) (h3 : n % 18 = 4) : n = 130 := sorry

end least_number_divisible_l270_270766


namespace problem_l270_270655

noncomputable def f (x : ℝ) (a b : ℝ) := x^2 + a * x + b

theorem problem (a b : ℝ) (h1 : f 1 a b = 0) (h2 : f 2 a b = 0) : f (-1) a b = 6 :=
by
  sorry

end problem_l270_270655


namespace gcd_180_126_l270_270024

theorem gcd_180_126 : Nat.gcd 180 126 = 18 := by
  sorry

end gcd_180_126_l270_270024


namespace integer_solutions_x2_minus_y2_equals_12_l270_270959

theorem integer_solutions_x2_minus_y2_equals_12 : 
  ∃! (s : Finset (ℤ × ℤ)), (∀ (xy : ℤ × ℤ), xy ∈ s → xy.1^2 - xy.2^2 = 12) ∧ s.card = 4 :=
sorry

end integer_solutions_x2_minus_y2_equals_12_l270_270959


namespace correct_statements_count_l270_270218

theorem correct_statements_count (x : ℝ) :
  let inverse := (x > 0) → (x^2 > 0)
  let converse := (x^2 ≤ 0) → (x ≤ 0)
  let contrapositive := (x ≤ 0) → (x^2 ≤ 0)
  (∃ p : Prop, p = inverse ∨ p = converse ∧ p) ↔ 
  ¬ contrapositive →
  2 = 2 :=
by
  sorry

end correct_statements_count_l270_270218


namespace find_a_value_l270_270908

noncomputable def find_a (a : ℝ) : Prop :=
  (a > 0) ∧ (1 / 3 = 2 / a)

theorem find_a_value (a : ℝ) (h : find_a a) : a = 6 :=
sorry

end find_a_value_l270_270908


namespace mother_returns_to_freezer_l270_270793

noncomputable def probability_return_to_freezer : ℝ :=
  1 - ((5 / 17) * (4 / 16) * (3 / 15) * (2 / 14) * (1 / 13))

theorem mother_returns_to_freezer :
  abs (probability_return_to_freezer - 0.99979) < 0.00001 :=
by
    sorry

end mother_returns_to_freezer_l270_270793


namespace fraction_difference_is_correct_l270_270799

-- Convert repeating decimal to fraction
def repeating_to_fraction : ℚ := 0.72 + 72 / 9900

-- Convert 0.72 to fraction
def decimal_to_fraction : ℚ := 72 / 100

-- Define the difference between the fractions
def fraction_difference := repeating_to_fraction - decimal_to_fraction

-- Final statement to prove
theorem fraction_difference_is_correct : fraction_difference = 2 / 275 := by
  sorry

end fraction_difference_is_correct_l270_270799


namespace minimum_perimeter_l270_270095

def fractional_part (x : ℚ) : ℚ := x - x.floor

-- Define l, m, n being sides of the triangle with l > m > n
variables (l m n : ℤ)

-- Defining conditions as Lean predicates
def triangle_sides (l m n : ℤ) : Prop := l > m ∧ m > n

def fractional_part_condition (l m n : ℤ) : Prop :=
  fractional_part (3^l / 10^4) = fractional_part (3^m / 10^4) ∧
  fractional_part (3^m / 10^4) = fractional_part (3^n / 10^4)

-- Prove the minimum perimeter is 3003 given above conditions
theorem minimum_perimeter (l m n : ℤ) :
  triangle_sides l m n →
  fractional_part_condition l m n →
  l + m + n = 3003 :=
by
  intros h_sides h_fractional
  sorry

end minimum_perimeter_l270_270095


namespace gcd_n3_plus_16_n_plus_4_l270_270212

/-- For a given positive integer n > 2^4, the greatest common divisor of n^3 + 16 and n + 4 is 1. -/
theorem gcd_n3_plus_16_n_plus_4 (n : ℕ) (h : n > 2^4) : Nat.gcd (n^3 + 16) (n + 4) = 1 := by
  sorry

end gcd_n3_plus_16_n_plus_4_l270_270212


namespace determine_angle_A_max_triangle_area_l270_270420

-- Conditions: acute triangle with sides opposite to angles A, B, C as a, b, c.
variables {A B C a b c : ℝ}
-- Given condition on angles.
axiom angle_condition : 1 + (Real.sqrt 3 / 3) * Real.sin (2 * A) = 2 * Real.sin ((B + C) / 2) ^ 2 
-- Circumcircle radius
axiom circumcircle_radius : Real.pi > A ∧ A > 0 

-- Question I: Determine angle A
theorem determine_angle_A : A = Real.pi / 3 :=
by sorry

-- Given radius of the circumcircle
noncomputable def R := 2 * Real.sqrt 3 

-- Maximum area of triangle ABC
theorem max_triangle_area (a b c : ℝ) : ∃ area, area = 9 * Real.sqrt 3 :=
by sorry

end determine_angle_A_max_triangle_area_l270_270420


namespace inequality_solution_set_l270_270049

theorem inequality_solution_set (a b c : ℝ)
  (h1 : ∀ x, (ax^2 + bx + c > 0 ↔ -3 < x ∧ x < 2)) :
  (a < 0) ∧ (a + b + c > 0) ∧ (∀ x, ¬ (bx + c > 0 ↔ x > 6)) ∧ (∀ x, (cx^2 + bx + a < 0 ↔ -1/3 < x ∧ x < 1/2)) :=
by
  sorry

end inequality_solution_set_l270_270049


namespace continuity_at_x_0_l270_270990

def f (x : ℝ) := -2 * x^2 + 9
def x_0 : ℝ := 4

theorem continuity_at_x_0 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x_0| < δ → |f x - f x_0| < ε :=
by
  sorry

end continuity_at_x_0_l270_270990


namespace min_value_expression_l270_270392

theorem min_value_expression (x : ℝ) (h : x > 3) : x + 4 / (x - 3) ≥ 7 :=
sorry

end min_value_expression_l270_270392


namespace cake_divided_into_equal_parts_l270_270779

theorem cake_divided_into_equal_parts (cake_weight : ℕ) (pierre : ℕ) (nathalie : ℕ) (parts : ℕ) 
  (hw_eq : cake_weight = 400)
  (hp_eq : pierre = 100)
  (pn_eq : pierre = 2 * nathalie)
  (parts_eq : cake_weight / nathalie = parts)
  (hparts_eq : parts = 8) :
  cake_weight / nathalie = 8 := 
by
  sorry

end cake_divided_into_equal_parts_l270_270779


namespace range_of_k_l270_270035

theorem range_of_k (k : ℝ) : ((∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ (-4 < k ∧ k ≤ 0)) :=
sorry

end range_of_k_l270_270035


namespace greatest_value_y_l270_270406

theorem greatest_value_y (y : ℝ) (hy : 11 = y^2 + 1/y^2) : y + 1/y ≤ Real.sqrt 13 :=
sorry

end greatest_value_y_l270_270406


namespace stagePlayRolesAssignment_correct_l270_270627

noncomputable def stagePlayRolesAssignment : ℕ :=
  let male_roles : ℕ := 4 * 3 -- ways to assign male roles
  let female_roles : ℕ := 5 * 4 -- ways to assign female roles
  let either_gender_roles : ℕ := 5 * 4 * 3 -- ways to assign either-gender roles
  male_roles * female_roles * either_gender_roles -- total assignments

theorem stagePlayRolesAssignment_correct : stagePlayRolesAssignment = 14400 := by
  sorry

end stagePlayRolesAssignment_correct_l270_270627


namespace permutations_mississippi_l270_270372

theorem permutations_mississippi : 
  let total_letters := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  (Nat.factorial total_letters / (Nat.factorial m_count * Nat.factorial i_count * Nat.factorial s_count * Nat.factorial p_count)) = 34650 := 
by
  sorry

end permutations_mississippi_l270_270372


namespace vertex_of_given_function_l270_270756

-- Definition of the given quadratic function
def given_function (x : ℝ) : ℝ := 2 * (x - 4) ^ 2 + 5

-- Definition of the vertex coordinates
def vertex_coordinates : ℝ × ℝ := (4, 5)

-- Theorem stating the vertex coordinates of the function
theorem vertex_of_given_function : (0, given_function 4) = vertex_coordinates :=
by 
  -- Placeholder for the proof
  sorry

end vertex_of_given_function_l270_270756


namespace fraction_of_book_finished_l270_270946

variables (x y : ℝ)

theorem fraction_of_book_finished (h1 : x = y + 90) (h2 : x + y = 270) : x / 270 = 2 / 3 :=
by sorry

end fraction_of_book_finished_l270_270946


namespace valid_three_digit_card_numbers_count_l270_270466

def card_numbers : List (ℕ × ℕ) := [(0, 1), (2, 3), (4, 5), (7, 8)]

def is_valid_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 -- Ensures it's three digits

def three_digit_numbers : List ℕ := 
  [201, 210, 102, 120, 301, 310, 103, 130, 401, 410, 104, 140,
   501, 510, 105, 150, 601, 610, 106, 160, 701, 710, 107, 170,
   801, 810, 108, 180, 213, 231, 312, 321, 413, 431, 512, 521,
   613, 631, 714, 741, 813, 831, 214, 241, 315, 351, 415, 451,
   514, 541, 615, 651, 716, 761, 815, 851, 217, 271, 317, 371,
   417, 471, 517, 571, 617, 671, 717, 771, 817, 871, 217, 271,
   321, 371, 421, 471, 521, 571, 621, 671, 721, 771, 821, 871]

def count_valid_three_digit_numbers : ℕ :=
  three_digit_numbers.length

theorem valid_three_digit_card_numbers_count :
    count_valid_three_digit_numbers = 168 :=
by
  -- proof goes here
  sorry

end valid_three_digit_card_numbers_count_l270_270466


namespace reciprocal_of_neg_2023_l270_270458

theorem reciprocal_of_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l270_270458


namespace compute_expression_l270_270088

theorem compute_expression (x y z : ℝ) (h₀ : x ≠ y) (h₁ : y ≠ z) (h₂ : z ≠ x) (h₃ : x + y + z = 3) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = 9 / (2 * (x^2 + y^2 + z^2)) - 1 / 2 :=
by
  sorry

end compute_expression_l270_270088


namespace cupcakes_left_l270_270574

def total_cupcakes : ℕ := 40
def students_class_1 : ℕ := 18
def students_class_2 : ℕ := 16
def additional_individuals : ℕ := 4

theorem cupcakes_left (total_cupcakes students_class_1 students_class_2 additional_individuals : ℕ) :
  total_cupcakes - students_class_1 - students_class_2 - additional_individuals = 2 :=
by
  have h1 : total_cupcakes - students_class_1 = 22 := by sorry
  have h2 : 22 - students_class_2 = 6 := by sorry
  have h3 : 6 - additional_individuals = 2 := by sorry
  exact h3

end cupcakes_left_l270_270574


namespace cube_volume_in_pyramid_l270_270183

-- Definition for the conditions and parameters of the problem
def pyramid_condition (base_length : ℝ) (triangle_side : ℝ) : Prop :=
  base_length = 2 ∧ triangle_side = 2 * Real.sqrt 2

-- Definition for the cube's placement and side length condition inside the pyramid
def cube_side_length (s : ℝ) : Prop :=
  s = (Real.sqrt 6 / 3)

-- The final Lean statement proving the volume of the cube
theorem cube_volume_in_pyramid (base_length triangle_side s : ℝ) 
  (h_base_length : base_length = 2)
  (h_triangle_side : triangle_side = 2 * Real.sqrt 2)
  (h_cube_side_length : s = (Real.sqrt 6 / 3)) :
  (s ^ 3) = (2 * Real.sqrt 6 / 9) := 
by
  -- Using the given conditions to assert the conclusion
  rw [h_cube_side_length]
  have : (Real.sqrt 6 / 3) ^ 3 = 2 * Real.sqrt 6 / 9 := sorry
  exact this

end cube_volume_in_pyramid_l270_270183


namespace solve_eq1_solve_eq2_l270_270209

theorem solve_eq1 (x : ℝ) : 3 * (x - 2) ^ 2 = 27 ↔ (x = 5 ∨ x = -1) :=
by
  sorry

theorem solve_eq2 (x : ℝ) : (x + 5) ^ 3 + 27 = 0 ↔ x = -8 :=
by
  sorry

end solve_eq1_solve_eq2_l270_270209


namespace coda_password_combinations_l270_270507

open BigOperators

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13 ∨ n = 17 ∨ n = 19 
  ∨ n = 23 ∨ n = 29

def is_power_of_two (n : ℕ) : Prop :=
  n = 2 ∨ n = 4 ∨ n = 8 ∨ n = 16

def is_multiple_of_three (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n ≥ 1 ∧ n ≤ 30

def count_primes : ℕ :=
  10
def count_powers_of_two : ℕ :=
  4
def count_multiples_of_three : ℕ :=
  10

theorem coda_password_combinations : count_primes * count_powers_of_two * count_multiples_of_three = 400 := by
  sorry

end coda_password_combinations_l270_270507


namespace paint_remaining_after_two_days_l270_270782

-- Define the conditions
def original_paint_amount := 1
def paint_used_day1 := original_paint_amount * (1/4)
def remaining_paint_after_day1 := original_paint_amount - paint_used_day1
def paint_used_day2 := remaining_paint_after_day1 * (1/2)
def remaining_paint_after_day2 := remaining_paint_after_day1 - paint_used_day2

-- Theorem to be proved
theorem paint_remaining_after_two_days :
  remaining_paint_after_day2 = (3/8) * original_paint_amount := sorry

end paint_remaining_after_two_days_l270_270782


namespace find_number_l270_270477

theorem find_number (x : ℝ) (h : 3 * (2 * x + 5) = 105) : x = 15 :=
by
  sorry

end find_number_l270_270477


namespace smallest_nat_with_55_divisors_l270_270031

open BigOperators

theorem smallest_nat_with_55_divisors :
  ∃ (n : ℕ), 
    (∃ (f : ℕ → ℕ) (primes : Finset ℕ),
      (∀ p ∈ primes, Nat.Prime p) ∧ 
      (primes.Sum (λ p => p ^ (f p))) = n ∧
      ((primes.Sum (λ p => f p + 1)) = 55)) ∧
    (∀ m, 
      (∃ (f_m : ℕ → ℕ) (primes_m : Finset ℕ),
        (∀ p ∈ primes_m, Nat.Prime p) ∧ 
        (primes_m.Sum (λ p => p ^ (f_m p))) = m ∧
        ((primes_m.Sum (λ p => f_m p + 1)) = 55)) → 
      n ≤ m) ∧
  n = 3^4 * 2^10 := 
begin
  sorry
end

end smallest_nat_with_55_divisors_l270_270031


namespace solve_quadratic_l270_270443

theorem solve_quadratic (y : ℝ) :
  y^2 - 3 * y - 10 = -(y + 2) * (y + 6) ↔ (y = -1/2 ∨ y = -2) :=
by
  sorry

end solve_quadratic_l270_270443


namespace bamboo_fifth_section_volume_l270_270745

theorem bamboo_fifth_section_volume
  (a₁ q : ℝ)
  (h1 : a₁ * (a₁ * q) * (a₁ * q^2) = 3)
  (h2 : (a₁ * q^6) * (a₁ * q^7) * (a₁ * q^8) = 9) :
  a₁ * q^4 = Real.sqrt 3 :=
sorry

end bamboo_fifth_section_volume_l270_270745


namespace orchid_bushes_after_planting_l270_270463

def total_orchid_bushes (current_orchids new_orchids : Nat) : Nat :=
  current_orchids + new_orchids

theorem orchid_bushes_after_planting :
  ∀ (current_orchids new_orchids : Nat), current_orchids = 22 → new_orchids = 13 → total_orchid_bushes current_orchids new_orchids = 35 :=
by
  intros current_orchids new_orchids h_current h_new
  rw [h_current, h_new]
  exact rfl

end orchid_bushes_after_planting_l270_270463


namespace total_money_9pennies_4nickels_3dimes_l270_270061

def value_of_pennies (num_pennies : ℕ) : ℝ := num_pennies * 0.01
def value_of_nickels (num_nickels : ℕ) : ℝ := num_nickels * 0.05
def value_of_dimes (num_dimes : ℕ) : ℝ := num_dimes * 0.10

def total_value (pennies nickels dimes : ℕ) : ℝ :=
  value_of_pennies pennies + value_of_nickels nickels + value_of_dimes dimes

theorem total_money_9pennies_4nickels_3dimes :
  total_value 9 4 3 = 0.59 :=
by 
  sorry

end total_money_9pennies_4nickels_3dimes_l270_270061


namespace integer_to_the_fourth_l270_270706

theorem integer_to_the_fourth (a : ℤ) (h : a = 243) : 3^12 * 3^8 = a^4 :=
by {
  sorry
}

end integer_to_the_fourth_l270_270706


namespace tan_half_prod_eq_sqrt3_l270_270888

theorem tan_half_prod_eq_sqrt3 (a b : ℝ) (h : 7 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b + 1) = 0) :
  ∃ (xy : ℝ), xy = Real.tan (a / 2) * Real.tan (b / 2) ∧ (xy = Real.sqrt 3 ∨ xy = -Real.sqrt 3) :=
by
  sorry

end tan_half_prod_eq_sqrt3_l270_270888


namespace range_a_l270_270520

variable (a : ℝ)

def p := (∀ x : ℝ, x^2 + x + a > 0)
def q := ∃ x y : ℝ, x^2 - 2 * a * x + 1 ≤ y

theorem range_a :
  ({a : ℝ | (p a ∧ ¬q a) ∨ (¬p a ∧ q a)} = {a : ℝ | a < -1} ∪ {a : ℝ | 1 / 4 < a ∧ a < 1}) := 
by
  sorry

end range_a_l270_270520


namespace permutation_divisible_by_7_l270_270271

open Int

theorem permutation_divisible_by_7 (n : ℕ) (h : n ≥ 2) :
  ∃ p : List ℕ, p ~ List.range (n + 1) ∧ (p.foldr (λ (d m : ℕ), d + m * 10) 0) ≡ 0 [MOD 7] :=
  sorry

end permutation_divisible_by_7_l270_270271


namespace arithmetic_sequence_difference_l270_270744

noncomputable def arithmetic_difference (d: ℚ) (b₁: ℚ) : Prop :=
  (50 * b₁ + ((50 * 49) / 2) * d = 150) ∧
  (50 * (b₁ + 50 * d) + ((50 * 149) / 2) * d = 250)

theorem arithmetic_sequence_difference {d b₁ : ℚ} (h : arithmetic_difference d b₁) :
  (b₁ + d) - b₁ = (200 / 1295) :=
by
  sorry

end arithmetic_sequence_difference_l270_270744


namespace total_revenue_correct_l270_270133

-- Definitions based on the problem conditions
def price_per_kg_first_week : ℝ := 10
def quantity_sold_first_week : ℝ := 50
def discount_percentage : ℝ := 0.25
def multiplier_next_week : ℝ := 3

-- Derived definitions
def revenue_first_week := quantity_sold_first_week * price_per_kg_first_week
def quantity_sold_second_week := multiplier_next_week * quantity_sold_first_week
def discounted_price_per_kg := price_per_kg_first_week * (1 - discount_percentage)
def revenue_second_week := quantity_sold_second_week * discounted_price_per_kg
def total_revenue := revenue_first_week + revenue_second_week

-- The theorem that needs to be proven
theorem total_revenue_correct : total_revenue = 1625 := 
by
  sorry

end total_revenue_correct_l270_270133


namespace fewerSevensCanProduce100_l270_270149

noncomputable def validAs100UsingSevens : (ℕ → ℕ) → Prop := 
  fun (f : ℕ → ℕ) => ∃ (x y : ℕ), (fewer_than_ten_sevens x y) ∧ f x y = 100

theorem fewerSevensCanProduce100 : 
  ∃ (f : ℕ → ℕ), validAs100UsingSevens f :=
by 
  sorry

end fewerSevensCanProduce100_l270_270149


namespace min_x_prime_sum_l270_270728

theorem min_x_prime_sum (x y : ℕ) (h : 3 * x^2 = 5 * y^4) :
  ∃ a b c d : ℕ, x = a^b * c^d ∧ (a + b + c + d = 11) := 
by sorry

end min_x_prime_sum_l270_270728


namespace value_of_a2_b2_c2_l270_270090

noncomputable def nonzero_reals := { x : ℝ // x ≠ 0 }

theorem value_of_a2_b2_c2 (a b c : nonzero_reals) (h1 : (a : ℝ) + (b : ℝ) + (c : ℝ) = 0) 
  (h2 : (a : ℝ)^3 + (b : ℝ)^3 + (c : ℝ)^3 = (a : ℝ)^7 + (b : ℝ)^7 + (c : ℝ)^7) : 
  (a : ℝ)^2 + (b : ℝ)^2 + (c : ℝ)^2 = 6 / 7 :=
by
  sorry

end value_of_a2_b2_c2_l270_270090


namespace areas_of_triangles_l270_270564

open EuclideanGeometry

variables {A B C D E F M P R S L N K : Point ℝ}

-- Given 
-- Intersecting points:
axiom h1 : Line_thru B C ∩ Line_thru E D = P
axiom h2 : Line_thru E D ∩ Line_thru A F = R
axiom h3 : Line_thru A F ∩ Line_thru D C = S
axiom h4 : Line_thru A B ∩ Line_thru C D = L
axiom h5 : Line_thru C D ∩ Line_thru E F = N
axiom h6 : Line_thru E F ∩ Line_thru A B = K

-- Equilateral triangles:
axiom h7 : EquilateralTriangle [K, L, N]
axiom h8 : EquilateralTriangle [S, R, P]
axiom h9 : CongruentTriangle [K, L, N] [S, R, P]

-- Sum of distances:
axiom h10 : (distance_to_line M B C) + (distance_to_line M E D) + (distance_to_line M A F) = 
            (distance_to_line M A B) + (distance_to_line M C D) + (distance_to_line M A F)

-- Sum of areas:
axiom h11 : area A M B + area C M D + area F M E + area B M C + area D M E + area A M E = (1/2) * area A B C D E F
axiom h12 : area A M B + area E M D = area B M C + area M E F = area C M D + area A M F = (1/3) * area A B C D E F

-- Prove:
theorem areas_of_triangles : 
  area M F E = 6 ∧ 
  area B M C = 6 ∧ 
  area D M E = 9 ∧ 
  area A M F = 3 :=
sorry

end areas_of_triangles_l270_270564


namespace solve_problem_l270_270051

variable (a b c x : ℝ)

-- Conditions
def condition1 : Prop := ∀ x, ax^2 + bx + c > 0 ↔ -3 < x ∧ x < 2

-- Statements to prove
def statementA : Prop := a < 0
def statementB : Prop := a + b + c > 0
def statementD : Prop := ∀ x, (cx^2 + bx + a < 0 ↔ (-1/3 < x ∧ x < 1/2))

theorem solve_problem (h1 : condition1)
  (h2 : statementA)
  (h3 : statementB)
  (h4 : statementD) : a < 0 ∧ a + b + c > 0 ∧ ∀ x, (cx^2 + bx + a < 0 ↔ (-1/3 < x ∧ x < 1/2)) :=
by
  sorry

end solve_problem_l270_270051


namespace area_of_enclosed_shape_l270_270307

noncomputable def enclosed_area : ℝ := 
∫ x in (0 : ℝ)..(2/3 : ℝ), (2 * x - 3 * x^2)

theorem area_of_enclosed_shape : enclosed_area = 4 / 27 := by
  sorry

end area_of_enclosed_shape_l270_270307


namespace factor_difference_of_squares_l270_270847

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l270_270847


namespace rich_knight_l270_270743

-- Definitions for the problem
inductive Status
| knight  -- Always tells the truth
| knave   -- Always lies

def tells_truth (s : Status) : Prop := 
  s = Status.knight

def lies (s : Status) : Prop := 
  s = Status.knave

def not_poor (s : Status) : Prop := 
  s = Status.knight ∨ s = Status.knave -- Knights can either be poor or wealthy

def wealthy (s : Status) : Prop :=
  s = Status.knight

-- Statement to be proven
theorem rich_knight (s : Status) (h_truth : tells_truth s) (h_not_poor : not_poor s) : wealthy s :=
by
  sorry

end rich_knight_l270_270743


namespace f_2016_eq_one_third_l270_270233

noncomputable def f (x : ℕ) : ℝ := sorry

axiom f_one : f 1 = 2
axiom f_recurrence : ∀ x : ℕ, f (x + 1) = (1 + f x) / (1 - f x)

theorem f_2016_eq_one_third : f 2016 = 1 / 3 := sorry

end f_2016_eq_one_third_l270_270233


namespace max_sides_of_convex_polygon_with_4_obtuse_l270_270707

theorem max_sides_of_convex_polygon_with_4_obtuse (n : ℕ) (hn : n ≥ 3) :
  (∃ k : ℕ, k = 4 ∧
    ∀ θ : Fin n → ℝ, 
      (∀ p, θ p > 90 ∧ ∃ t, θ t = 180 ∨ θ t < 90 ∨ θ t = 90) →
      4 = k →
      n ≤ 7
  ) :=
sorry

end max_sides_of_convex_polygon_with_4_obtuse_l270_270707


namespace vasya_100_using_fewer_sevens_l270_270160

-- Definitions and conditions
def seven := 7

-- Theorem to prove
theorem vasya_100_using_fewer_sevens :
  (777 / seven - 77 / seven = 100) ∨
  (seven * seven + seven * seven + seven / seven + seven / seven = 100) :=
by
  sorry

end vasya_100_using_fewer_sevens_l270_270160


namespace log_expression_simplification_l270_270525

open Real

theorem log_expression_simplification (p q r s t z : ℝ) :
  log (p / q) + log (q / r) + log (r / s) - log (p * t / (s * z)) = log (z / t) :=
  sorry

end log_expression_simplification_l270_270525


namespace find_fahrenheit_l270_270764

variable (F : ℝ)
variable (C : ℝ)

theorem find_fahrenheit (h : C = 40) (h' : C = 5 / 9 * (F - 32)) : F = 104 := by
  sorry

end find_fahrenheit_l270_270764


namespace smallest_n_divisible_by_23_l270_270859

theorem smallest_n_divisible_by_23 :
  ∃ n : ℕ, (n^3 + 12 * n^2 + 15 * n + 180) % 23 = 0 ∧
            ∀ m : ℕ, (m^3 + 12 * m^2 + 15 * m + 180) % 23 = 0 → n ≤ m :=
sorry

end smallest_n_divisible_by_23_l270_270859


namespace perpendicular_slope_l270_270648

def line_slope (A B : ℚ) (x y : ℚ) : ℚ := A * x - B * y

def is_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

theorem perpendicular_slope : 
  ∃ m : ℚ, let slope_given_line := (4 : ℚ) / (5 : ℚ) in 
    is_perpendicular m slope_given_line ∧ m = - (5 : ℚ) / (4 : ℚ) := 
by 
  sorry

end perpendicular_slope_l270_270648


namespace repeating_seventy_two_exceeds_seventy_two_l270_270827

noncomputable def repeating_decimal (n d : ℕ) : ℚ := n / d

theorem repeating_seventy_two_exceeds_seventy_two :
  repeating_decimal 72 99 - (72 / 100) = (2 / 275) := 
sorry

end repeating_seventy_two_exceeds_seventy_two_l270_270827


namespace blue_balls_taken_out_l270_270964

theorem blue_balls_taken_out :
  ∃ x : ℕ, (0 ≤ x ∧ x ≤ 7) ∧ (7 - x) / (15 - x) = 1 / 3 ∧ x = 3 :=
sorry

end blue_balls_taken_out_l270_270964


namespace proposition_only_A_l270_270334

def is_proposition (statement : String) : Prop := sorry

def statement_A : String := "Red beans grow in the southern country"
def statement_B : String := "They sprout several branches in spring"
def statement_C : String := "I hope you pick more"
def statement_D : String := "For these beans symbolize longing"

theorem proposition_only_A :
  is_proposition statement_A ∧
  ¬is_proposition statement_B ∧
  ¬is_proposition statement_C ∧
  ¬is_proposition statement_D := 
sorry

end proposition_only_A_l270_270334


namespace range_of_a_l270_270738

open Real

theorem range_of_a (a : ℝ) :
  (0 < a ∧ a < 6) ∨ (a ≥ 5 ∨ a ≤ 1) ∧ ¬((0 < a ∧ a < 6) ∧ (a ≥ 5 ∨ a ≤ 1)) ↔ 
  (a ≥ 6 ∨ a ≤ 0 ∨ (1 < a ∧ a < 5)) :=
by sorry

end range_of_a_l270_270738


namespace find_full_haired_dogs_l270_270912

-- Definitions of the given conditions
def minutes_per_short_haired_dog : Nat := 10
def short_haired_dogs : Nat := 6
def total_time_minutes : Nat := 4 * 60
def twice_as_long (n : Nat) : Nat := 2 * n

-- Define the problem
def full_haired_dogs : Nat :=
  let short_haired_total_time := short_haired_dogs * minutes_per_short_haired_dog
  let remaining_time := total_time_minutes - short_haired_total_time
  remaining_time / (twice_as_long minutes_per_short_haired_dog)

-- Theorem statement
theorem find_full_haired_dogs : 
  full_haired_dogs = 9 :=
by
  sorry

end find_full_haired_dogs_l270_270912


namespace count_positive_integers_with_two_digits_l270_270678

theorem count_positive_integers_with_two_digits : 
  ∃ (count : ℕ), count = 2151 ∧ ∀ n, n < 100000 → (∀ d1 d2 d, d ∈ digits n → d = d1 ∨ d = d2) → n ∈ (range 100000) := sorry

end count_positive_integers_with_two_digits_l270_270678


namespace sarah_total_weeds_l270_270582

theorem sarah_total_weeds :
  let tuesday_weeds := 25
  let wednesday_weeds := 3 * tuesday_weeds
  let thursday_weeds := wednesday_weeds / 5
  let friday_weeds := thursday_weeds - 10
  tuesday_weeds + wednesday_weeds + thursday_weeds + friday_weeds = 120 :=
by
  intros
  let tuesday_weeds := 25
  let wednesday_weeds := 3 * tuesday_weeds
  let thursday_weeds := wednesday_weeds / 5
  let friday_weeds := thursday_weeds - 10
  sorry

end sarah_total_weeds_l270_270582


namespace stella_glasses_count_l270_270445

-- Definitions for the conditions
def dolls : ℕ := 3
def clocks : ℕ := 2
def price_per_doll : ℕ := 5
def price_per_clock : ℕ := 15
def price_per_glass : ℕ := 4
def total_cost : ℕ := 40
def profit : ℕ := 25

-- The proof statement
theorem stella_glasses_count (dolls clocks price_per_doll price_per_clock price_per_glass total_cost profit : ℕ) :
  (dolls * price_per_doll + clocks * price_per_clock) + profit + total_cost = total_cost + profit → 
  (dolls * price_per_doll + clocks * price_per_clock) + profit + total_cost - (dolls * price_per_doll + clocks * price_per_clock) = price_per_glass * 5 :=
sorry

end stella_glasses_count_l270_270445


namespace exponent_multiplication_l270_270336

theorem exponent_multiplication :
  (10 ^ 10000) * (10 ^ 8000) = 10 ^ 18000 :=
by
  sorry

end exponent_multiplication_l270_270336


namespace conical_surface_radius_l270_270247

theorem conical_surface_radius (r : ℝ) :
  (2 * Real.pi * r = 5 * Real.pi) → r = 2.5 :=
by
  sorry

end conical_surface_radius_l270_270247


namespace base8_to_base10_conversion_l270_270340

theorem base8_to_base10_conversion : 
  let n := 432
  let base := 8
  let result := 282
  (2 * base^0 + 3 * base^1 + 4 * base^2) = result := 
by
  let n := 2 * 8^0 + 3 * 8^1 + 4 * 8^2
  have h1 : n = 2 + 24 + 256 := by sorry
  have h2 : 2 + 24 + 256 = 282 := by sorry
  exact Eq.trans h1 h2


end base8_to_base10_conversion_l270_270340


namespace path_length_cube_dot_l270_270490

-- Define the edge length of the cube
def edge_length : ℝ := 2

-- Define the distance of the dot from the center of the top face
def dot_distance_from_center : ℝ := 0.5

-- Define the number of complete rolls
def complete_rolls : ℕ := 2

-- Calculate the constant c such that the path length of the dot is c * π
theorem path_length_cube_dot : ∃ c : ℝ, dot_distance_from_center = 2.236 :=
by
  sorry

end path_length_cube_dot_l270_270490


namespace a7_is_1_S2022_is_4718_l270_270498

def harmonious_progressive (a : ℕ → ℕ) : Prop :=
  ∀ p q : ℕ, p > 0 → q > 0 → a p = a q → a (p + 1) = a (q + 1)

variables (a : ℕ → ℕ) (S : ℕ → ℕ)

axiom harmonious_seq : harmonious_progressive a
axiom a1 : a 1 = 1
axiom a2 : a 2 = 2
axiom a4 : a 4 = 1
axiom a6_plus_a8 : a 6 + a 8 = 6

theorem a7_is_1 : a 7 = 1 := sorry

theorem S2022_is_4718 : S 2022 = 4718 := sorry

end a7_is_1_S2022_is_4718_l270_270498


namespace infinite_series_sum_l270_270019

theorem infinite_series_sum :
  ∑' n : ℕ, (1 / (n.succ * (n.succ + 2))) = 3 / 4 :=
by sorry

end infinite_series_sum_l270_270019


namespace playback_methods_proof_l270_270000

/-- A TV station continuously plays 5 advertisements, consisting of 3 different commercial advertisements
and 2 different Olympic promotional advertisements. The requirements are:
  1. The last advertisement must be an Olympic promotional advertisement.
  2. The 2 Olympic promotional advertisements can be played consecutively.
-/
def number_of_playback_methods (commercials olympics: ℕ) (last_ad_olympic: Bool) (olympics_consecutive: Bool) : ℕ :=
  if commercials = 3 ∧ olympics = 2 ∧ last_ad_olympic ∧ olympics_consecutive then 36 else 0

theorem playback_methods_proof :
  number_of_playback_methods 3 2 true true = 36 := by
  sorry

end playback_methods_proof_l270_270000


namespace repeating_decimal_exceeds_finite_decimal_by_l270_270807

-- Definitions based on the problem conditions
def repeating_decimal := 8 / 11
def finite_decimal := 18 / 25

-- Statement to be proved
theorem repeating_decimal_exceeds_finite_decimal_by : 
  repeating_decimal - finite_decimal = 2 / 275 :=
by
  -- Skipping the proof itself with 'sorry'
  sorry

end repeating_decimal_exceeds_finite_decimal_by_l270_270807


namespace hyperbola_asymptote_l270_270884

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) : 
  (∀ x y, 3 * x + 2 * y = 0 ∨ 3 * x - 2 * y = 0) →
  (∀ x y, y * y = 9 * (x * x / (a * a) - 1)) →
  a = 2 :=
by
  intros asymptote_constr hyp
  sorry

end hyperbola_asymptote_l270_270884


namespace vasya_100_using_fewer_sevens_l270_270156

-- Definitions and conditions
def seven := 7

-- Theorem to prove
theorem vasya_100_using_fewer_sevens :
  (777 / seven - 77 / seven = 100) ∨
  (seven * seven + seven * seven + seven / seven + seven / seven = 100) :=
by
  sorry

end vasya_100_using_fewer_sevens_l270_270156


namespace prove_expression_l270_270839

def given_expression : ℤ := -4 + 6 / (-2)

theorem prove_expression : given_expression = -7 := 
by 
  -- insert proof here
  sorry

end prove_expression_l270_270839


namespace find_a_l270_270893

theorem find_a (a x y : ℝ)
    (h1 : a * x - 5 * y = 5)
    (h2 : x / (x + y) = 5 / 7)
    (h3 : x - y = 3) :
    a = 3 := 
by 
  sorry

end find_a_l270_270893


namespace average_percentage_score_is_71_l270_270548

-- Define the number of students.
def number_of_students : ℕ := 150

-- Define the scores and their corresponding frequencies.
def scores_and_frequencies : List (ℕ × ℕ) :=
  [(100, 10), (95, 20), (85, 45), (75, 30), (65, 25), (55, 15), (45, 5)]

-- Define the total points scored by all students.
def total_points_scored : ℕ := 
  scores_and_frequencies.foldl (λ acc pair => acc + pair.1 * pair.2) 0

-- Define the average percentage score.
def average_score : ℚ := total_points_scored / number_of_students

-- Statement of the proof problem.
theorem average_percentage_score_is_71 :
  average_score = 71.0 := by
  sorry

end average_percentage_score_is_71_l270_270548


namespace circle_equation_from_tangents_and_parabola_l270_270527

/-- Given the parabola x^2 = 4y and the point H(1, -1), prove that the equation of the circle with the segment AB as its diameter, where A and B are points of intersection between the parabola and the tangent lines through H, is (x - 1)^2 + (y - 3/2)^2 = 25/4. -/
theorem circle_equation_from_tangents_and_parabola (x y : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ, x1^2 = 4 * y1 ∧ x2^2 = 4 * y2 ∧
      (x1 - 2 * y1 + 2 = 0 ∧ x2 - 2 * y2 + 2 = 0) ∧
      (1 - 2 * (-1) + 2 = 0 ∧ (x1^2 - 0) + (x2^2 - 0 = 5))
  </} ></-- +
  (x - 1)^2 + (y - (3/2))^2 = 25 / 4 :=}


end circle_equation_from_tangents_and_parabola_l270_270527


namespace solve_fraction_zero_l270_270130

theorem solve_fraction_zero (x : ℝ) (h1 : (x^2 - 25) / (x + 5) = 0) (h2 : x ≠ -5) : x = 5 :=
sorry

end solve_fraction_zero_l270_270130


namespace Rick_is_three_times_Sean_l270_270111

-- Definitions and assumptions
def Fritz_money : ℕ := 40
def Sean_money : ℕ := (Fritz_money / 2) + 4
def total_money : ℕ := 96

-- Rick's money can be derived from total_money - Sean_money
def Rick_money : ℕ := total_money - Sean_money

-- Claim to be proven
theorem Rick_is_three_times_Sean : Rick_money = 3 * Sean_money := 
by 
  -- Proof steps would go here
  sorry

end Rick_is_three_times_Sean_l270_270111


namespace four_digit_greater_than_three_digit_l270_270492

theorem four_digit_greater_than_three_digit (n m : ℕ) (h₁ : 1000 ≤ n ∧ n ≤ 9999) (h₂ : 100 ≤ m ∧ m ≤ 999) : n > m :=
sorry

end four_digit_greater_than_three_digit_l270_270492


namespace random_events_l270_270349

def is_random_event_1 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  ∃ c d : ℝ, c > 0 → d < 0 → a + d < 0 ∨ b + c > 0

def is_random_event_2 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  ∃ c d : ℝ, c > 0 → d < 0 → a - d > 0 ∨ b - c < 0

def is_impossible_event_3 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  a * b > 0

def is_certain_event_4 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  a / b < 0

theorem random_events (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  is_random_event_1 a b ha hb ∧ is_random_event_2 a b ha hb :=
by
  sorry

end random_events_l270_270349


namespace example_one_example_two_l270_270163

-- We will define natural numbers corresponding to seven, seventy-seven, and seven hundred seventy-seven.
def seven : ℕ := 7
def seventy_seven : ℕ := 77
def seven_hundred_seventy_seven : ℕ := 777

-- We will define both solutions in the form of equalities producing 100.
theorem example_one : (seven_hundred_seventy_seven / seven) - (seventy_seven / seven) = 100 :=
  by sorry

theorem example_two : (seven * seven) + (seven * seven) + (seven / seven) + (seven / seven) = 100 :=
  by sorry

end example_one_example_two_l270_270163


namespace lucas_change_l270_270100

def initialAmount : ℕ := 20
def costPerAvocado : ℕ := 2
def numberOfAvocados : ℕ := 3

def totalCost : ℕ := numberOfAvocados * costPerAvocado
def change : ℕ := initialAmount - totalCost

theorem lucas_change : change = 14 := by
  sorry

end lucas_change_l270_270100


namespace embankment_building_l270_270082

theorem embankment_building (days : ℕ) (workers_initial : ℕ) (workers_later : ℕ) (embankments : ℕ) :
  workers_initial = 75 → days = 4 → embankments = 2 →
  (∀ r : ℚ, embankments = workers_initial * r * days →
            embankments = workers_later * r * 5) :=
by
  intros h75 hd4 h2 r hr
  sorry

end embankment_building_l270_270082


namespace sin_theta_eq_neg_one_ninth_l270_270862

theorem sin_theta_eq_neg_one_ninth 
(θ : ℝ)
(h : Real.cos (Real.pi / 4 - θ / 2) = 2 / 3) :
  Real.sin θ = -1 / 9 := 
by
  sorry

end sin_theta_eq_neg_one_ninth_l270_270862


namespace charlie_extra_fee_l270_270840

-- Conditions
def data_limit_week1 : ℕ := 2 -- in GB
def data_limit_week2 : ℕ := 3 -- in GB
def data_limit_week3 : ℕ := 2 -- in GB
def data_limit_week4 : ℕ := 1 -- in GB

def additional_fee_week1 : ℕ := 12 -- dollars per GB
def additional_fee_week2 : ℕ := 10 -- dollars per GB
def additional_fee_week3 : ℕ := 8 -- dollars per GB
def additional_fee_week4 : ℕ := 6 -- dollars per GB

def data_used_week1 : ℕ := 25 -- in 0.1 GB
def data_used_week2 : ℕ := 40 -- in 0.1 GB
def data_used_week3 : ℕ := 30 -- in 0.1 GB
def data_used_week4 : ℕ := 50 -- in 0.1 GB

-- Additional fee calculation
def extra_data_fee := 
  let extra_data_week1 := max (data_used_week1 - data_limit_week1 * 10) 0
  let extra_fee_week1 := extra_data_week1 * additional_fee_week1 / 10
  let extra_data_week2 := max (data_used_week2 - data_limit_week2 * 10) 0
  let extra_fee_week2 := extra_data_week2 * additional_fee_week2 / 10
  let extra_data_week3 := max (data_used_week3 - data_limit_week3 * 10) 0
  let extra_fee_week3 := extra_data_week3 * additional_fee_week3 / 10
  let extra_data_week4 := max (data_used_week4 - data_limit_week4 * 10) 0
  let extra_fee_week4 := extra_data_week4 * additional_fee_week4 / 10
  extra_fee_week1 + extra_fee_week2 + extra_fee_week3 + extra_fee_week4

-- The math proof problem
theorem charlie_extra_fee : extra_data_fee = 48 := sorry

end charlie_extra_fee_l270_270840


namespace math_problem_l270_270092

open Real

noncomputable def f (x : ℝ) : ℝ := sin (π * x / 2) ^ 2
noncomputable def g (x : ℝ) : ℝ := 1 - abs (x - 1)

theorem math_problem (x : ℝ) (k : ℤ) :
  (∀ x, f (x) = f (2 * k - x)) ∧
  (∀ x, 1 < x ∧ x < 2 → ∃ x1 x2, x1 < x2 ∧ f x1 > f x ∧ f x2 < f x) ∧
  (∀ x, f (x - 1) + g (x - 1) = f (1 - x) + g (1 - x)) ∧
  (∀ x, f (x) ≤ 1 ∧ g (x) ≤ 1 ∧ f 1 + g 1 = 2)
:=
begin
  -- Proof for each statement here
  sorry,
end

end math_problem_l270_270092


namespace range_of_g_l270_270924

def f (x : ℝ) : ℝ := 4 * x + 1

def g (x : ℝ) : ℝ := f (f (f (f (x))))

theorem range_of_g : ∀ x, 0 ≤ x ∧ x ≤ 3 → 85 ≤ g x ∧ g x ≤ 853 :=
by
  intro x h
  sorry

end range_of_g_l270_270924


namespace domain_of_h_l270_270638

open Real

theorem domain_of_h : ∀ x : ℝ, |x - 5| + |x + 2| ≠ 0 := by
  intro x
  sorry

end domain_of_h_l270_270638


namespace div_condition_l270_270043

theorem div_condition
  (a b : ℕ)
  (h₁ : a < 1000)
  (h₂ : b ≠ 0)
  (h₃ : b ∣ a ^ 21)
  (h₄ : b ^ 10 ∣ a ^ 21) :
  b ∣ a ^ 2 :=
sorry

end div_condition_l270_270043


namespace number_of_positive_integers_with_at_most_two_digits_l270_270693

theorem number_of_positive_integers_with_at_most_two_digits:
  -- There are 2151 positive integers less than 100000 with at most two different digits
  ∃ (n : ℕ), n = 2151 ∧
    (∀ k : ℕ, k < 100000 → 
      let digits := (nat.digits 10 k).erase_dup 
      in (1 ≤ digits.length ∧ digits.length ≤ 2) → 1 ≤ k ∧ k < 100000) :=
begin
  -- proof would go here
  sorry
end

end number_of_positive_integers_with_at_most_two_digits_l270_270693


namespace initial_speed_l270_270426

variable (v : ℝ)
variable (h1 : (v / 2) + 2 * v = 75)

theorem initial_speed (v : ℝ) (h1 : (v / 2) + 2 * v = 75) : v = 30 :=
sorry

end initial_speed_l270_270426


namespace seven_expression_one_seven_expression_two_l270_270137

theorem seven_expression_one : 777 / 7 - 77 / 7 = 100 :=
by sorry

theorem seven_expression_two : 7 * 7 + 7 * 7 + 7 / 7 + 7 / 7 = 100 :=
by sorry

end seven_expression_one_seven_expression_two_l270_270137


namespace fraction_difference_is_correct_l270_270795

-- Convert repeating decimal to fraction
def repeating_to_fraction : ℚ := 0.72 + 72 / 9900

-- Convert 0.72 to fraction
def decimal_to_fraction : ℚ := 72 / 100

-- Define the difference between the fractions
def fraction_difference := repeating_to_fraction - decimal_to_fraction

-- Final statement to prove
theorem fraction_difference_is_correct : fraction_difference = 2 / 275 := by
  sorry

end fraction_difference_is_correct_l270_270795


namespace linlin_speed_l270_270440

theorem linlin_speed (distance time : ℕ) (q_speed linlin_speed : ℕ)
  (h1 : distance = 3290)
  (h2 : time = 7)
  (h3 : q_speed = 70)
  (h4 : distance = (q_speed + linlin_speed) * time) : linlin_speed = 400 :=
by sorry

end linlin_speed_l270_270440


namespace median_of_roller_coaster_times_l270_270754

theorem median_of_roller_coaster_times:
  let data := [80, 85, 90, 125, 130, 135, 140, 145, 195, 195, 210, 215, 240, 245, 300, 305, 315, 320, 325, 330, 300]
  ∃ median_time, median_time = 210 ∧
    (∀ t ∈ data, t ≤ median_time ↔ index_of_median = 11) :=
by
  sorry

end median_of_roller_coaster_times_l270_270754


namespace distribution_schemes_l270_270114

theorem distribution_schemes 
    (total_professors : ℕ)
    (high_schools : Finset ℕ) 
    (A : ℕ) 
    (B : ℕ) 
    (C : ℕ)
    (D : ℕ)
    (cond1 : total_professors = 6) 
    (cond2 : A = 1)
    (cond3 : B ≥ 1)
    (cond4 : C ≥ 1)
    (D' := (total_professors - A - B - C)) 
    (cond5 : D' ≥ 1) : 
    ∃ N : ℕ, N = 900 := by
  sorry

end distribution_schemes_l270_270114


namespace boxes_with_neither_l270_270502

def total_boxes : ℕ := 15
def boxes_with_crayons : ℕ := 9
def boxes_with_markers : ℕ := 6
def boxes_with_both : ℕ := 4

theorem boxes_with_neither : total_boxes - (boxes_with_crayons + boxes_with_markers - boxes_with_both) = 4 := by
  sorry

end boxes_with_neither_l270_270502


namespace solution_Y_required_l270_270917

theorem solution_Y_required (V_total V_ratio_Y : ℝ) (h_total : V_total = 0.64) (h_ratio : V_ratio_Y = 3 / 8) : 
  (0.64 * (3 / 8) = 0.24) :=
by
  sorry

end solution_Y_required_l270_270917


namespace chord_length_range_l270_270671

variable {x y : ℝ}

def center : ℝ × ℝ := (4, 5)
def radius : ℝ := 13
def point : ℝ × ℝ := (1, 1)
def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 169

-- statement: prove the range of |AB| for specific conditions
theorem chord_length_range :
  ∀ line : (ℝ × ℝ) → (ℝ × ℝ) → Prop,
  (line center point → line (x, y) (x, y) ∧ circle_eq x y)
  → 24 ≤ abs (dist (x, y) (x, y)) ∧ abs (dist (x, y) (x, y)) ≤ 26 :=
by
  sorry

end chord_length_range_l270_270671


namespace find_fifth_day_sales_l270_270352

-- Define the variables and conditions
variables (x : ℝ)
variables (a : ℝ := 100) (b : ℝ := 92) (c : ℝ := 109) (d : ℝ := 96) (f : ℝ := 96) (g : ℝ := 105)
variables (mean : ℝ := 100.1)

-- Define the mean condition which leads to the proof of x
theorem find_fifth_day_sales : (a + b + c + d + x + f + g) / 7 = mean → x = 102.7 := by
  intro h
  -- Proof goes here
  sorry

end find_fifth_day_sales_l270_270352


namespace expand_polynomial_l270_270021

theorem expand_polynomial (t : ℝ) : (2 * t^3 - 3 * t + 2) * (-3 * t^2 + 3 * t - 5) = 
  -6 * t^5 + 6 * t^4 - t^3 + 3 * t^2 + 21 * t - 10 :=
by
  sorry

end expand_polynomial_l270_270021


namespace diamonds_in_G_10_l270_270637

-- Define the sequence rule for diamonds in Gn
def diamonds_in_G (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 10
  else 2 + 2 * n^2 + 2 * n - 4

-- The main theorem to prove that the number of diamonds in G₁₀ is 218
theorem diamonds_in_G_10 : diamonds_in_G 10 = 218 := by
  sorry

end diamonds_in_G_10_l270_270637


namespace vasya_example_fewer_sevens_l270_270154

theorem vasya_example_fewer_sevens : 
  (777 / 7) - (77 / 7) = 100 :=
begin
  sorry
end

end vasya_example_fewer_sevens_l270_270154


namespace eq_margin_l270_270413

variables (C S n : ℝ) (M : ℝ)

theorem eq_margin (h : M = 1 / n * (2 * C - S)) : M = S / (n + 2) :=
sorry

end eq_margin_l270_270413


namespace man_cannot_row_against_stream_l270_270621

theorem man_cannot_row_against_stream (rate_in_still_water speed_with_stream : ℝ)
  (h_rate : rate_in_still_water = 1)
  (h_speed_with : speed_with_stream = 6) :
  ¬ ∃ (speed_against_stream : ℝ), speed_against_stream = rate_in_still_water - (speed_with_stream - rate_in_still_water) :=
by
  sorry

end man_cannot_row_against_stream_l270_270621


namespace kate_average_speed_correct_l270_270274

noncomputable def kate_average_speed : ℝ :=
  let biking_time_hours := 20 / 60
  let walking_time_hours := 60 / 60
  let jogging_time_hours := 40 / 60
  let biking_distance := 20 * biking_time_hours
  let walking_distance := 4 * walking_time_hours
  let jogging_distance := 6 * jogging_time_hours
  let total_distance := biking_distance + walking_distance + jogging_distance
  let total_time_hours := biking_time_hours + walking_time_hours + jogging_time_hours
  total_distance / total_time_hours

theorem kate_average_speed_correct : kate_average_speed = 9 :=
by
  sorry

end kate_average_speed_correct_l270_270274


namespace exists_integer_coordinates_l270_270553

theorem exists_integer_coordinates :
  ∃ (x y : ℤ), (x^2 + y^2) = 2 * 2017^2 + 2 * 2018^2 :=
by
  sorry

end exists_integer_coordinates_l270_270553


namespace proof_problem_l270_270866

-- Define the constants a, b, c
def a := 0.1 * Real.exp 0.1
def b := 1 / 9
def c := -Real.log 0.9

-- State the theorem to prove
theorem proof_problem : c < a ∧ a < b := 
by
  sorry

end proof_problem_l270_270866


namespace pages_needed_l270_270611

def total_new_cards : ℕ := 8
def total_old_cards : ℕ := 10
def cards_per_page : ℕ := 3

theorem pages_needed (h : total_new_cards = 8) (h2 : total_old_cards = 10) (h3 : cards_per_page = 3) : 
  (total_new_cards + total_old_cards) / cards_per_page = 6 := by 
  sorry

end pages_needed_l270_270611


namespace calculate_total_money_l270_270533

noncomputable def cost_per_gumdrop : ℕ := 4
noncomputable def number_of_gumdrops : ℕ := 20
noncomputable def total_money : ℕ := 80

theorem calculate_total_money : 
  cost_per_gumdrop * number_of_gumdrops = total_money := 
by
  sorry

end calculate_total_money_l270_270533


namespace age_of_youngest_child_l270_270459

theorem age_of_youngest_child (x : ℕ) (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 70) : x = 8 :=
by
  sorry

end age_of_youngest_child_l270_270459


namespace proportion_fourth_number_l270_270242

theorem proportion_fourth_number (x y : ℝ) (h_x : x = 0.6) (h_prop : 0.75 / x = 10 / y) : y = 8 :=
by
  sorry

end proportion_fourth_number_l270_270242


namespace expression_value_l270_270844

theorem expression_value :
  (100 - (3000 - 300) + (3000 - (300 - 100)) = 200) := by
  sorry

end expression_value_l270_270844


namespace notebooks_type_A_count_minimum_profit_m_l270_270106

def total_notebooks := 350
def costA := 12
def costB := 15
def total_cost := 4800

def selling_priceA := 20
def selling_priceB := 25
def discountA := 0.7
def profit_min := 2348

-- Prove the number of type A notebooks is 150
theorem notebooks_type_A_count (x y : ℕ) (h1 : x + y = total_notebooks)
    (h2 : costA * x + costB * y = total_cost) : x = 150 := by
  sorry

-- Prove the minimum value of m is 111 such that profit is not less than 2348
theorem minimum_profit_m (m : ℕ) (profit : ℕ)
    (h : profit = (m * selling_priceA + m * selling_priceB  + (150 - m) * (selling_priceA * discountA).toNat + (200 - m) * costB - total_cost))
    (h_prof : profit >= profit_min) : m >= 111 := by
  sorry

end notebooks_type_A_count_minimum_profit_m_l270_270106


namespace company_pays_per_box_per_month_l270_270999

/-
  Given:
  - The dimensions of each box are 15 inches by 12 inches by 10 inches
  - The total volume occupied by all boxes is 1,080,000 cubic inches
  - The total cost for record storage per month is $480

  Prove:
  - The company pays $0.80 per box per month for record storage
-/

theorem company_pays_per_box_per_month :
  let length := 15
  let width := 12
  let height := 10
  let box_volume := length * width * height
  let total_volume := 1080000
  let total_cost := 480
  let num_boxes := total_volume / box_volume
  let cost_per_box_per_month := total_cost / num_boxes
  cost_per_box_per_month = 0.80 :=
by
  let length := 15
  let width := 12
  let height := 10
  let box_volume := length * width * height
  let total_volume := 1080000
  let total_cost := 480
  let num_boxes := total_volume / box_volume
  let cost_per_box_per_month := total_cost / num_boxes
  sorry

end company_pays_per_box_per_month_l270_270999


namespace count_valid_numbers_l270_270683

def is_valid_number (n : ℕ) : Prop :=
  n < 100000 ∧
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ d₁ ≠ d₂ ∧
    (∀ k, (n / 10^k % 10 = d₁ ∨ n / 10^k % 10 = d₂))

theorem count_valid_numbers : 
  ∃ (k : ℕ), k = 2151 ∧ (∀ n, is_valid_number n → n < 100000 → n ≤ k) :=
by
  sorry

end count_valid_numbers_l270_270683


namespace area_of_roof_l270_270312

def roof_area (w l : ℕ) : ℕ := l * w

theorem area_of_roof :
  ∃ (w l : ℕ), l = 4 * w ∧ l - w = 45 ∧ roof_area w l = 900 :=
by
  -- Defining witnesses for width and length
  use 15, 60
  -- Splitting the goals for clarity
  apply And.intro
  -- Proving the first condition: l = 4 * w
  · show 60 = 4 * 15
    rfl
  apply And.intro
  -- Proving the second condition: l - w = 45
  · show 60 - 15 = 45
    rfl
  -- Proving the area calculation: roof_area w l = 900
  · show roof_area 15 60 = 900
    rfl

end area_of_roof_l270_270312


namespace solve_triangle_problem_l270_270238
noncomputable def triangle_problem (A B C a b c : ℝ) (area : ℝ) : Prop :=
  (2 * c * Real.sin B * Real.cos A - b * Real.sin C = 0) ∧
  area = Real.sqrt 3 ∧ 
  b + c = 5 →
  (A = Real.pi / 3) ∧ (a = Real.sqrt 13)

-- Lean statement for the proof problem
theorem solve_triangle_problem 
  (A B C a b c : ℝ) 
  (h1 : 2 * c * Real.sin B * Real.cos A - b * Real.sin C = 0)
  (h2 : 1/2 * b * c * Real.sin A = Real.sqrt 3)
  (h3 : b + c = 5) :
  A = Real.pi / 3 ∧ a = Real.sqrt 13 :=
sorry

end solve_triangle_problem_l270_270238


namespace square_area_l270_270792

theorem square_area (x : ℝ) 
  (h1 : 5 * x - 18 = 27 - 4 * x) 
  (side_length : ℝ := 5 * x - 18) : 
  side_length ^ 2 = 49 := 
by 
  sorry

end square_area_l270_270792


namespace intersection_complement_l270_270562

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4, 5}

-- Define set M
def M : Set ℕ := {0, 3, 5}

-- Define set N
def N : Set ℕ := {1, 4, 5}

-- Define the complement of N in U
def complement_U_N : Set ℕ := U \ N

-- The main theorem statement
theorem intersection_complement : M ∩ complement_U_N = {0, 3} :=
by
  -- The proof would go here
  sorry

end intersection_complement_l270_270562


namespace total_ladybugs_l270_270966

theorem total_ladybugs (leaves : Nat) (ladybugs_per_leaf : Nat) (total_ladybugs : Nat) : 
  leaves = 84 → 
  ladybugs_per_leaf = 139 → 
  total_ladybugs = leaves * ladybugs_per_leaf → 
  total_ladybugs = 11676 := by
  intros h1 h2 h3
  rw [h1, h2] at h3
  assumption

end total_ladybugs_l270_270966


namespace customer_survey_response_l270_270010

theorem customer_survey_response (N : ℕ)
  (avg_income : ℕ → ℕ)
  (avg_all : avg_income N = 45000)
  (avg_top10 : avg_income 10 = 55000)
  (avg_others : avg_income (N - 10) = 42500) :
  N = 50 := 
sorry

end customer_survey_response_l270_270010


namespace repeating_decimal_difference_l270_270819

theorem repeating_decimal_difference :
  let x := 72 / 99 in
  let y := 72 / 100 in
  x - y = 2 / 275 := 
by
  -- we will add a proof later
  sorry

end repeating_decimal_difference_l270_270819


namespace seven_expression_one_seven_expression_two_l270_270139

theorem seven_expression_one : 777 / 7 - 77 / 7 = 100 :=
by sorry

theorem seven_expression_two : 7 * 7 + 7 * 7 + 7 / 7 + 7 / 7 = 100 :=
by sorry

end seven_expression_one_seven_expression_two_l270_270139


namespace ordered_pair_l270_270767

-- Definitions
def P (x : ℝ) := x^4 - 8 * x^3 + 20 * x^2 - 34 * x + 15
def D (k : ℝ) (x : ℝ) := x^2 - 3 * x + k
def R (a : ℝ) (x : ℝ) := x + a

-- Hypothesis
def condition (k a : ℝ) : Prop := ∀ x : ℝ, P x % D k x = R a x

-- Theorem
theorem ordered_pair (k a : ℝ) (h : condition k a) : (k, a) = (5, 15) := 
  sorry

end ordered_pair_l270_270767


namespace sqrt_x_minus_2_meaningful_in_reals_l270_270256

theorem sqrt_x_minus_2_meaningful_in_reals (x : ℝ) : (∃ (y : ℝ), y * y = x - 2) → x ≥ 2 :=
by
  sorry

end sqrt_x_minus_2_meaningful_in_reals_l270_270256


namespace smallest_m_l270_270275

noncomputable def fractional_part (x : ℝ) : ℝ :=
  x - ⌊x⌋

noncomputable def f (x : ℝ) : ℝ :=
  abs (3 * fractional_part x - 1.5)

theorem smallest_m (m : ℤ) (h1 : ∀ x : ℝ, m^2 * f (x * f x) = x → True) : ∃ m, m = 8 :=
by
  have h2 : ∀ m : ℤ, (∃ (s : ℕ), s ≥ 1008 ∧ (m^2 * abs (3 * fractional_part (s * abs (1.5 - 3 * (fractional_part s) )) - 1.5) = s)) → m = 8
  {
    sorry
  }
  sorry

end smallest_m_l270_270275


namespace not_periodic_cos_add_cos_sqrt2_l270_270362

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.cos (x * Real.sqrt 2)

theorem not_periodic_cos_add_cos_sqrt2 :
  ¬(∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) :=
sorry

end not_periodic_cos_add_cos_sqrt2_l270_270362


namespace binomial_constant_term_l270_270858

theorem binomial_constant_term : 
  ∃ (c : ℚ), (x : ℝ) → (x^2 + (1 / (2 * x)))^6 = c ∧ c = 15 / 16 := by
  sorry

end binomial_constant_term_l270_270858


namespace relationship_y_values_l270_270665

theorem relationship_y_values (m y1 y2 y3 : ℝ) 
  (h1 : y1 = -3 * (-3 : ℝ)^2 - 12 * (-3 : ℝ) + m)
  (h2 : y2 = -3 * (-2 : ℝ)^2 - 12 * (-2 : ℝ) + m)
  (h3 : y3 = -3 * (1 : ℝ)^2 - 12 * (1 : ℝ) + m) :
  y2 > y1 ∧ y1 > y3 :=
sorry

end relationship_y_values_l270_270665


namespace infinitely_many_solutions_l270_270592

def sum_of_first_n (x : ℕ) : ℕ := x * (x + 1) / 2

theorem infinitely_many_solutions : 
  ∃ (f : ℕ → ℕ × ℕ), 
  ∀ k : ℕ, 
  let (x_k, y_k) := f k in 
  sum_of_first_n x_k = y_k * y_k := 
sorry

end infinitely_many_solutions_l270_270592


namespace fraction_zero_numerator_l270_270417

theorem fraction_zero_numerator (x : ℝ) (h₁ : (x^2 - 9) / (x + 3) = 0) (h₂ : x + 3 ≠ 0) : x = 3 :=
sorry

end fraction_zero_numerator_l270_270417


namespace probability_A_C_winning_l270_270038

-- Definitions based on the conditions given
def students := ["A", "B", "C", "D"]

def isDistictPositions (x y : String) : Prop :=
  x ≠ y

-- Lean statement for the mathematical problem
theorem probability_A_C_winning :
  ∃ (P : ℚ), P = 1/6 :=
by
  sorry

end probability_A_C_winning_l270_270038


namespace kyle_lift_weight_l270_270429

theorem kyle_lift_weight (this_year_weight last_year_weight : ℕ) 
  (h1 : this_year_weight = 80) 
  (h2 : this_year_weight = 3 * last_year_weight) : 
  (this_year_weight - last_year_weight) = 53 := by
  sorry

end kyle_lift_weight_l270_270429


namespace fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l270_270145

theorem fewer_than_ten_sevens_example1 : (777 / 7) - (77 / 7) = 100 :=
  by sorry

theorem fewer_than_ten_sevens_example2 : (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7) = 100 :=
  by sorry

end fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l270_270145


namespace geometric_sequence_a6_l270_270651

theorem geometric_sequence_a6
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (a1 : a 1 = 1)
  (S3 : S 3 = 7 / 4)
  (sum_S3 : S 3 = a 1 + a 1 * a 2 + a 1 * (a 2)^2) :
  a 6 = 1 / 32 := by
  sorry

end geometric_sequence_a6_l270_270651


namespace min_value_F_l270_270386

theorem min_value_F :
  ∀ (x y : ℝ), (x^2 + y^2 - 2*x - 2*y + 1 = 0) → (x + 1) / y ≥ 3 / 4 :=
by
  intro x y h
  sorry

end min_value_F_l270_270386
