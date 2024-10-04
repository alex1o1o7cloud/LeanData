import Mathlib

namespace find_eccentricity_l773_773040

noncomputable def ellipse_eccentricity (m : ℝ) (c : ℝ) (a : ℝ) : ℝ :=
  c / a

theorem find_eccentricity
  (m : ℝ) (c := Real.sqrt 2) (a := 3 * Real.sqrt 2 / 2)
  (h1 : 2 * m^2 - (m + 1) = 2)
  (h2 : m > 0) :
  ellipse_eccentricity m c a = 2 / 3 :=
by sorry

end find_eccentricity_l773_773040


namespace average_members_remaining_l773_773116

theorem average_members_remaining :
  let initial_members := [7, 8, 10, 13, 6, 10, 12, 9]
  let members_leaving := [1, 2, 1, 2, 1, 2, 1, 2]
  let remaining_members := List.map (λ (x, y) => x - y) (List.zip initial_members members_leaving)
  let total_remaining := List.foldl Nat.add 0 remaining_members
  let num_families := initial_members.length
  total_remaining / num_families = 63 / 8 := by
    sorry

end average_members_remaining_l773_773116


namespace train_crossing_time_l773_773347

theorem train_crossing_time (train_length : ℕ) (bridge_length : ℕ) (speed_kmph : ℕ) : ℕ :=
  let total_distance := train_length + bridge_length
  let speed_mps := (speed_kmph * 1000) / 3600
  let time_seconds := total_distance / speed_mps
  if train_length = 160 ∧ bridge_length = 215 ∧ speed_kmph = 45 then
    time_seconds = 30
  else
    sorry

#print axioms train_crossing_time

end train_crossing_time_l773_773347


namespace find_b_magnitude_l773_773483

noncomputable def vector_a : ℝ × ℝ := (1, 2)
def projection_length : ℝ := 2 * Real.sqrt 5
noncomputable def diff_norm : ℝ := Real.sqrt 10

theorem find_b_magnitude (b : ℝ × ℝ)
  (h1 : b.1 * vector_a.1 + b.2 * vector_a.2 = 2 * Real.sqrt 5 * Real.sqrt (vector_a.1^2 + vector_a.2^2))
  (h2 : Real.sqrt ((vector_a.1 - b.1)^2 + (vector_a.2 - b.2)^2) = Real.sqrt 10) :
  Real.sqrt (b.1^2 + b.2^2) = 5 :=
sorry

end find_b_magnitude_l773_773483


namespace enter_exit_ways_eq_sixteen_l773_773260

theorem enter_exit_ways_eq_sixteen (n : ℕ) (h : n = 4) : n * n = 16 :=
by sorry

end enter_exit_ways_eq_sixteen_l773_773260


namespace students_no_A_l773_773865

theorem students_no_A
  (total_students : ℕ)
  (A_in_English : ℕ)
  (A_in_math : ℕ)
  (A_in_both : ℕ)
  (total_students_eq : total_students = 40)
  (A_in_English_eq : A_in_English = 10)
  (A_in_math_eq : A_in_math = 18)
  (A_in_both_eq : A_in_both = 6) :
  total_students - ((A_in_English + A_in_math) - A_in_both) = 18 :=
by
  sorry

end students_no_A_l773_773865


namespace triangle_inequality_l773_773146

theorem triangle_inequality (S R r : ℝ) (h : S^2 = 2 * R^2 + 8 * R * r + 3 * r^2) : 
  R / r = 2 ∨ R / r ≥ Real.sqrt 2 + 1 := 
by 
  sorry

end triangle_inequality_l773_773146


namespace slips_drawn_prob_even_l773_773395

theorem slips_drawn_prob_even :
∀ (n : ℕ),
    let P_n := ((5/10) * (4/9) * (3/8) * (2/7) * (1/6)).pow n in
    P_n = 0.023809523809523808 → n = 4 := 
by
  sorry

end slips_drawn_prob_even_l773_773395


namespace oak_grove_public_library_books_l773_773259

theorem oak_grove_public_library_books
  (total_books : ℕ)
  (school_books : ℕ)
  (h_total_books : total_books = 7092)
  (h_school_books : school_books = 5106) :
  total_books - school_books = 1986 :=
by
  rw [h_total_books, h_school_books]
  exact rfl

end oak_grove_public_library_books_l773_773259


namespace cycling_time_l773_773979

-- Define the necessary conditions
def length_breadth_ratio (L B : ℕ) : Prop := (B = 3 * L)
def park_area (L B : ℕ) : Prop := (30000 = L * B)
def cycling_speed := 12 * 1000 / 3600  -- 12 km/hr in meters per second

-- State the problem and the corresponding proof
theorem cycling_time {L B : ℕ} (h1 : length_breadth_ratio L B) (h2 : park_area L B) : 
  (800 / cycling_speed) / 60 = 4 :=
by 
  -- Preamble for the calculations needed
  have h3 : B = 3 * L := h1,
  have h4 : 30000 = L * (3 * L) := h2,
  have h5 : 30000 = 3 * L^2 := by rwa [mul_assoc] at h4,
  have h6 : L^2 = 10000 := by linarith only [h5],
  have h7 : L = 100 := by rw [←Nat.sqrt_eq, Nat.pow_two] at h6,
  have h8 : (200 + 600) = 800 := by norm_num,
  have h9 : cycling_speed = 10 / 3 := by norm_num [cycling_speed],
  have h10 : 800 / (10 / 3) = 240 := by norm_num,
  have h11 : 240 / 60 = 4 := by norm_num,
  rw [h11],
  sorry  -- Proof continues here as needed

end cycling_time_l773_773979


namespace distinct_prime_factors_of_A_l773_773150

noncomputable def A : ℕ := ∏ (d : ℕ) in (finset.filter (λ x, 60 % x = 0) (finset.range 61)), d

theorem distinct_prime_factors_of_A : (finset.filter (nat.prime) (nat.factorization A).support).card = 3 :=
sorry

end distinct_prime_factors_of_A_l773_773150


namespace girls_picked_more_l773_773891

variable (N I A V : ℕ)

theorem girls_picked_more (h1 : N > A) (h2 : N > V) (h3 : N > I)
                         (h4 : I ≥ A) (h5 : I ≥ V) (h6 : A > V) :
  N + I > A + V := by
  sorry

end girls_picked_more_l773_773891


namespace convex_tetrahedral_angle_if_and_only_if_l773_773577

/-
  This statement defines the edge generators of a convex tetrahedral angle
  as generators of a cone if and only if the sums of the opposite dihedral 
  angles of the tetrahedral angle are equal.
-/
theorem convex_tetrahedral_angle_if_and_only_if (SA SB SC SD : Type*) 
  (dihedral_angle : SA → SA → SA → SA → Real) 
  (α β γ δ : Real) :
  (∃ (g : SA → ConicalSection), is_generating_line g SA ∧ is_generating_line g SB ∧ is_generating_line g SC ∧ is_generating_line g SD) 
  ↔ 
  (dihedral_angle SA SB + dihedral_angle SC SD = dihedral_angle SB SC + dihedral_angle SD SA) := 
sorry

end convex_tetrahedral_angle_if_and_only_if_l773_773577


namespace hannah_total_savings_l773_773077

theorem hannah_total_savings :
  let a1 := 4
  let a2 := 2 * a1
  let a3 := 2 * a2
  let a4 := 2 * a3
  let a5 := 20
  a1 + a2 + a3 + a4 + a5 = 80 :=
by
  sorry

end hannah_total_savings_l773_773077


namespace hari_contribution_l773_773194

theorem hari_contribution 
    (P_investment : ℕ) (P_time : ℕ) (H_time : ℕ) (profit_ratio : ℚ)
    (investment_ratio : P_investment * P_time / (Hari_contribution * H_time) = profit_ratio) :
    Hari_contribution = 10080 :=
by
    have P_investment := 3920
    have P_time := 12
    have H_time := 7
    have profit_ratio := (2 : ℚ) / 3
    sorry

end hari_contribution_l773_773194


namespace find_cheesecake_price_l773_773120

def price_of_cheesecake (C : ℝ) (coffee_price : ℝ) (discount_rate : ℝ) (final_price : ℝ) : Prop :=
  let original_price := coffee_price + C
  let discounted_price := discount_rate * original_price
  discounted_price = final_price

theorem find_cheesecake_price : ∃ C : ℝ,
  price_of_cheesecake C 6 0.75 12 ∧ C = 10 :=
by
  sorry

end find_cheesecake_price_l773_773120


namespace calculate_inradius_of_right_triangle_l773_773693

noncomputable def right_triangle {a b c : ℕ} (h : a^2 + b^2 = c^2) :=
  {sides : ℕ × ℕ × ℕ // sides = (a, b, c) ∧ a < b ∧ b < c}

noncomputable def inradius {a b c : ℕ} (h : a^2 + b^2 = c^2) : ℝ :=
  let s := (a + b + c) / 2 in (s - a) * (s - b) * (s - c) / s

theorem calculate_inradius_of_right_triangle :
  ∀ (a b c : ℕ) (h : a^2 + b^2 = c^2), (a, b, c) = (5, 12, 13) → inradius h = 2 :=
by
  intros a b c h sides
  sorry

end calculate_inradius_of_right_triangle_l773_773693


namespace ABC_books_sold_eq_4_l773_773715

/-- "TOP" book cost in dollars --/
def TOP_price : ℕ := 8

/-- "ABC" book cost in dollars --/
def ABC_price : ℕ := 23

/-- Number of "TOP" books sold --/
def TOP_books_sold : ℕ := 13

/-- Difference in earnings in dollars --/
def earnings_difference : ℕ := 12

/-- Prove the number of "ABC" books sold --/
theorem ABC_books_sold_eq_4 (x : ℕ) (h : TOP_books_sold * TOP_price - x * ABC_price = earnings_difference) : x = 4 :=
by
  sorry

end ABC_books_sold_eq_4_l773_773715


namespace scheduling_plans_count_l773_773706

theorem scheduling_plans_count :
  let employees := {A, B, C, D, E, F, G} in
  let days := {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday} in
  let schedules : employees → days := sorry in
  (∑ (s : schedules), 
    ((s A = Monday ∨ s A = Tuesday) ∧
    (s B ≠ Tuesday) ∧
    (s C = Friday) ∧
    (∀ (e1 e2 : employees), e1 ≠ e2 → s e1 ≠ s e2)))  
  = 216 :=
begin
  -- proof will go here
  sorry
end

end scheduling_plans_count_l773_773706


namespace number_of_integer_solutions_l773_773490

theorem number_of_integer_solutions :
  { x : ℝ | (x - 3)^(36 - x^2) = 1 }.count = 4 :=
by
  sorry

end number_of_integer_solutions_l773_773490


namespace solution_set_inequality_l773_773626

theorem solution_set_inequality (x : ℝ) : 
  (2 < 1 / (x - 1) ∧ 1 / (x - 1) < 3) ↔ (4 / 3 < x ∧ x < 3 / 2) := 
by
  sorry

end solution_set_inequality_l773_773626


namespace problem_b_minus_a_equals_neg_two_l773_773913

theorem problem_b_minus_a_equals_neg_two (a b : ℝ) (A B : Set ℝ) 
  (h1 : A = {1, a + b, a}) (h2 : B = {0, b / a, b}) 
  (h3 : A = B) : b - a = -2 * a :=
begin
  -- Proof goes here
  sorry
end

end problem_b_minus_a_equals_neg_two_l773_773913


namespace problem_part_one_problem_part_two_l773_773369

theorem problem_part_one : 23 - 17 - (-6) + (-16) = -4 :=
by
  sorry

theorem problem_part_two : 0 - 32 / ((-2)^3 - (-4)) = 8 :=
by
  sorry

end problem_part_one_problem_part_two_l773_773369


namespace number_of_selection_methods_l773_773117

-- Define the types and conditions required for the proof
variables {Student : Type} {Boys : Finset Student} {Girls : Finset Student} (boyA girlB : Student)

-- Assume there are exactly 5 boys and 4 girls
-- And the total number of students is 9
axiom boys_count : Boys.card = 5
axiom girls_count : Girls.card = 4
axiom total_count : Boys ∪ Girls = {x | true}.to_finset ∧ (Boys ∪ Girls).card = 9

-- Additional conditions mentioned in the problem:
axiom boyA_condition : boyA ∈ Boys
axiom girlB_condition : girlB ∈ Girls

-- Define the question as a theorem
theorem number_of_selection_methods :
  (∃ S ⊆ (Boys ∪ Girls), S.card = 4 ∧ (∃ b ∈ Boys, b ∈ S) ∧ (∃ g ∈ Girls, g ∈ S) ∧ (boyA ∈ S ∨ girlB ∈ S)) = 86 := 
sorry

end number_of_selection_methods_l773_773117


namespace labor_union_trees_l773_773679

theorem labor_union_trees (x : ℕ) :
  (∃ t : ℕ, t = 2 * x + 21) ∧ (∃ t' : ℕ, t' = 3 * x - 24) →
  2 * x + 21 = 3 * x - 24 :=
by
  sorry

end labor_union_trees_l773_773679


namespace triangular_number_difference_30_28_l773_773362

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem triangular_number_difference_30_28 : triangular_number 30 - triangular_number 28 = 59 := 
by
  sorry

end triangular_number_difference_30_28_l773_773362


namespace divide_set_into_disjoint_subsets_l773_773083

theorem divide_set_into_disjoint_subsets {α : Type} [Fintype α] (h : Fintype.card α = 6) :
  (∃ (A B C : Finset α), A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧ 
                         A ∪ B ∪ C = Finset.univ ∧ 
                         (A.card = 2 ∧ B.card = 2 ∧ C.card = 2)) →
  fintype.card (quotient (λ (P1 P2 : {s // s.card = 2} × {s // s.card = 2} × {s // s.card = 2}), 
    ∃ σ : Sym (Fin 3), let ⟨⟨a1, a2⟩, a3⟩ := P1, ⟨⟨b1, b2⟩, b3⟩ := P2 in 
    (a1 = b1 ∘ σ) ∧ (a2 = b2 ∘ σ) ∧ (a3 = b3 ∘ σ))) = 15 :=
by
  sorry

end divide_set_into_disjoint_subsets_l773_773083


namespace calories_consumed_l773_773898

-- Define the conditions
def pages_written : ℕ := 12
def pages_per_donut : ℕ := 2
def calories_per_donut : ℕ := 150

-- Define the theorem to be proved
theorem calories_consumed (pages_written : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  (pages_written / pages_per_donut) * calories_per_donut

-- Ensure the theorem corresponds to the correct answer
example : calories_consumed 12 2 150 = 900 := by
  sorry

end calories_consumed_l773_773898


namespace find_initial_terms_l773_773181

noncomputable def sequence (a₁ a₂ a₃ : ℕ) (n : ℕ) : ℕ :=
  nat.rec_on n a₁ (nat.rec_on n a₂ (nat.rec_on n a₃ (λ n₁ fn₁, nat.rec_on n₁ a₃ (λ n₂ fn₂, fn₂ (fn₁ + 2 * sequence a₁ a₂ a₃ n₁)))))

theorem find_initial_terms : ∃ a₁ a₂ a₃ : ℕ, 
  sequence a₁ a₂ a₃ 6 = 2288 ∧ 
  a₁ = 5 ∧ a₂ = 1 ∧ a₃ = 2 :=
by
  sorry

end find_initial_terms_l773_773181


namespace has_max_and_min_values_iff_monotonic_on_interval_l773_773472

def f (a : ℝ) (x : ℝ) := -x^2 + a*x - real.log x

-- Question 1: Prove the necessary and sufficient condition for f(x) to have both a maximum and a minimum value.
theorem has_max_and_min_values_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (0 < x1) ∧ (0 < x2) ∧ f a x1 = f a x1 ) ↔ (a > 2) :=
sorry

-- Question 2: Prove the range of values for a when f(x) is monotonic on the interval [1, 2].
theorem monotonic_on_interval (a : ℝ) :
  (∀ x ∈ set.Icc 1 2, (∃ c : ℝ, x ∈ set.Icc 1 c ∧ (∀ y ∈ set.Icc 1 c, deriv (f a) y ≥ 0) ∨ (∀ y ∈ set.Icc 1 c, deriv (f a) y ≤ 0))) ↔ (a ≤ 2 ∨ a ≥ 3) :=
sorry

end has_max_and_min_values_iff_monotonic_on_interval_l773_773472


namespace simplify_and_evaluate_expr_l773_773943

theorem simplify_and_evaluate_expr (x : Real) (h : x = Real.sqrt 3 - 1) :
  1 - (x / (x + 1)) / (x / (x ^ 2 - 1)) = 3 - Real.sqrt 3 :=
sorry

end simplify_and_evaluate_expr_l773_773943


namespace menu_choices_l773_773428

theorem menu_choices :
  let lunchChinese := 5 
  let lunchJapanese := 4 
  let dinnerChinese := 3 
  let dinnerJapanese := 5 
  let lunchOptions := lunchChinese + lunchJapanese
  let dinnerOptions := dinnerChinese + dinnerJapanese
  lunchOptions * dinnerOptions = 72 :=
by
  let lunchChinese := 5
  let lunchJapanese := 4
  let dinnerChinese := 3
  let dinnerJapanese := 5
  let lunchOptions := lunchChinese + lunchJapanese
  let dinnerOptions := dinnerChinese + dinnerJapanese
  have h : lunchOptions * dinnerOptions = 72 :=
    by 
      sorry
  exact h

end menu_choices_l773_773428


namespace circle_area_below_line_l773_773646

theorem circle_area_below_line : 
  let circle_eq := (x y : ℝ) → (x-2)^2 + (y+3)^2 = 16
  let line_eq := (y : ℝ) → y = -1
  ∃ area, area = (40 * Real.pi) / 3 + 2 * Real.sqrt(3) * Real.pi
  ∧ ∀ x y, circle_eq x y → y < -1 → Area of circle below line_eq = area :=
begin
  sorry
end

end circle_area_below_line_l773_773646


namespace shaded_area_is_correct_l773_773531

-- Let n be the number 2015 represented using shaded regions within a grid of 1x1 squares.
-- We are tasked with proving the area of these shaded regions, given the possible configurations of lines.

-- Define the number 2015 written using shaded squares.
def number_with_shaded_squares (n : ℕ) : Prop :=
  n = 2015

-- Define the grid property.
def grid_1x1 : Prop :=
  ∀ (squares : ℕ), squares = 1

-- The configurations of the lines forming the shaded regions can be horizontal, vertical, or diagonals.
def line_configuration : Prop :=
  ∀ (lines : string), lines = "horizontal" ∨ lines = "vertical" ∨ lines = "diagonal"

-- The total area of the shaded regions needs to be calculated.
theorem shaded_area_is_correct :
  ∀ (n : ℕ), number_with_shaded_squares n → grid_1x1 → line_configuration → (area : ℝ), area = 47.5 :=
by {
  intros,
  sorry
}

end shaded_area_is_correct_l773_773531


namespace find_total_price_l773_773976

-- Define the cost parameters
variables (sugar_price salt_price : ℝ)

-- Define the given conditions
def condition_1 : Prop := 2 * sugar_price + 5 * salt_price = 5.50
def condition_2 : Prop := sugar_price = 1.50

-- Theorem to be proven
theorem find_total_price (h1 : condition_1 sugar_price salt_price) (h2 : condition_2 sugar_price) : 
  3 * sugar_price + 1 * salt_price = 5.00 :=
by
  sorry

end find_total_price_l773_773976


namespace sum_of_last_two_digits_l773_773652

theorem sum_of_last_two_digits (x y : ℕ) : 
  x = 8 → y = 12 → (x^25 + y^25) % 100 = 0 := 
by
  intros hx hy
  sorry

end sum_of_last_two_digits_l773_773652


namespace find_width_of_smaller_cuboid_l773_773080

def volume_cuboid (l w h : ℝ) : ℝ := l * w * h

theorem find_width_of_smaller_cuboid :
  ∃ w : ℝ, volume_cuboid 5 w 3 = 2 :=
begin
  let larger_cube_volume := volume_cuboid 18 15 2,
  let smaller_cubes_volume := 18 * volume_cuboid 5 _ 3,
  have h : 270 * 2 = larger_cube_volume,
  sorry,
end

end find_width_of_smaller_cuboid_l773_773080


namespace system_of_equations_solve_l773_773068

theorem system_of_equations_solve (x y : ℝ) 
  (h1 : 2 * x + y = 5)
  (h2 : x + 2 * y = 4) :
  x + y = 3 :=
by
  sorry

end system_of_equations_solve_l773_773068


namespace correct_option_l773_773659

-- Define the operations as boolean values
def optionA (a : ℝ) : Prop := a^4 * a^2 = a^8
def optionB (a : ℝ) : Prop := (a^2)^3 = a^6
def optionC (a : ℝ) : Prop := a^3 + a^3 = a^6
def optionD (a : ℝ) : Prop := (-2 * a)^3 = 8 * a^3

theorem correct_option : ∀ a : ℝ, optionB a :=
by
  intro a
  simp only [optionB, pow_mul]
  refl

end correct_option_l773_773659


namespace sine_of_angle_l773_773846

theorem sine_of_angle (α : Real) (h1 : cos α = -3/5) (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : sin α = -4/5 :=
by
  sorry

end sine_of_angle_l773_773846


namespace number_of_girls_l773_773243

theorem number_of_girls (B G : ℕ) (ratio_condition : B = G / 2) (total_condition : B + G = 90) : 
  G = 60 := 
by
  -- This is the problem statement, with conditions and required result.
  sorry

end number_of_girls_l773_773243


namespace fractions_order_l773_773722

theorem fractions_order : (23 / 18) < (21 / 16) ∧ (21 / 16) < (25 / 19) :=
by
  sorry

end fractions_order_l773_773722


namespace triangle_properties_l773_773857

theorem triangle_properties (a b c : ℝ) (h1 : a / b = 5 / 12) (h2 : b / c = 12 / 13) (h3 : a + b + c = 60) :
  (a^2 + b^2 = c^2) ∧ ((1 / 2) * a * b > 100) :=
by
  sorry

end triangle_properties_l773_773857


namespace problem_l773_773799

-- Let's define the conditions and the proof statement
variable (x a : ℤ) -- Define x and a as integer variables
variable (b : ℤ) -- Define b as an integer variable
hypothesis (h : x < a ∧ a < 0) -- Define the hypothesis that x < a < 0
hypothesis (h1 : b = x^2 - a^2) -- Define the hypothesis that b = x^2 - a^2

-- The goal is to prove that b > 0 under these conditions
theorem problem (x a b : ℤ) (h : x < a ∧ a < 0) (h1 : b = x^2 - a^2) : b > 0 := by
  sorry -- Placeholder for the proof

end problem_l773_773799


namespace mia_stops_in_quarter_C_l773_773936

def track_circumference : ℕ := 100 -- The circumference of the track in feet.
def total_distance_run : ℕ := 10560 -- The total distance Mia runs in feet.

-- Define the function to determine the quarter of the circle Mia stops in.
def quarter_mia_stops : ℕ :=
  let quarters := track_circumference / 4 -- Each quarter's length.
  let complete_laps := total_distance_run / track_circumference
  let remaining_distance := total_distance_run % track_circumference
  if remaining_distance < quarters then 1 -- Quarter A
  else if remaining_distance < 2 * quarters then 2 -- Quarter B
  else if remaining_distance < 3 * quarters then 3 -- Quarter C
  else 4 -- Quarter D

theorem mia_stops_in_quarter_C : quarter_mia_stops = 3 := by
  sorry

end mia_stops_in_quarter_C_l773_773936


namespace seulgi_stack_higher_l773_773086

-- Define the conditions
def num_red_boxes : ℕ := 15
def num_yellow_boxes : ℕ := 20
def height_red_box : ℝ := 4.2
def height_yellow_box : ℝ := 3.3

-- Define the total height for each stack
def total_height_hyunjeong : ℝ := num_red_boxes * height_red_box
def total_height_seulgi : ℝ := num_yellow_boxes * height_yellow_box

-- Lean statement to prove the comparison of their heights
theorem seulgi_stack_higher : total_height_seulgi > total_height_hyunjeong :=
by
  -- Proof will be inserted here
  sorry

end seulgi_stack_higher_l773_773086


namespace dagger_computation_l773_773974

def dagger (m n p q : ℕ) (hn : n ≠ 0) (hm : m ≠ 0) : ℚ :=
  (m^2 * p * (q / n)) + ((p : ℚ) / m)

theorem dagger_computation :
  dagger 5 9 6 2 (by norm_num) (by norm_num) = 518 / 15 :=
sorry

end dagger_computation_l773_773974


namespace parallel_condition_sufficient_not_necessary_l773_773839

noncomputable def a (x : ℝ) : ℝ × ℝ := (1, x - 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x + 1, 3)

theorem parallel_condition_sufficient_not_necessary (x : ℝ) :
  (x = 2) → (a x = b x) ∨ (a (-2) = b (-2)) :=
by sorry

end parallel_condition_sufficient_not_necessary_l773_773839


namespace track_length_eq_900_l773_773716

/-- 
Bruce and Bhishma are running on a circular track. 
The speed of Bruce is 30 m/s and that of Bhishma is 20 m/s.
They start from the same point at the same time in the same direction.
They meet again for the first time after 90 seconds. 
Prove that the length of the track is 900 meters.
-/
theorem track_length_eq_900 :
  let speed_bruce := 30 -- [m/s]
  let speed_bhishma := 20 -- [m/s]
  let time_meet := 90 -- [s]
  let distance_bruce := speed_bruce * time_meet
  let distance_bhishma := speed_bhishma * time_meet
  let track_length := distance_bruce - distance_bhishma
  track_length = 900 :=
by
  let speed_bruce := 30
  let speed_bhishma := 20
  let time_meet := 90
  let distance_bruce := speed_bruce * time_meet
  let distance_bhishma := speed_bhishma * time_meet
  let track_length := distance_bruce - distance_bhishma
  have : track_length = 900 := by
    sorry
  exact this

end track_length_eq_900_l773_773716


namespace john_overall_profit_l773_773903

noncomputable def purchase_price_grinder := 15000
noncomputable def purchase_price_mobile := 10000
noncomputable def loss_percentage_grinder := 0.04
noncomputable def profit_percentage_mobile := 0.10

noncomputable def loss_grinder := loss_percentage_grinder * purchase_price_grinder
noncomputable def selling_price_grinder := purchase_price_grinder - loss_grinder

noncomputable def profit_mobile := profit_percentage_mobile * purchase_price_mobile
noncomputable def selling_price_mobile := purchase_price_mobile + profit_mobile

noncomputable def total_purchase_price := purchase_price_grinder + purchase_price_mobile
noncomputable def total_selling_price := selling_price_grinder + selling_price_mobile

noncomputable def overall_profit := total_selling_price - total_purchase_price

theorem john_overall_profit : overall_profit = 400 := 
by 
  -- Here we would place the proof steps, but it is omitted as per instructions
  sorry

end john_overall_profit_l773_773903


namespace quadrilateral_opposite_sides_equal_l773_773596

variables {A B C D O : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited O]
variables (angle_B angle_D : ℚ) (midpoint : O) (bisection : AC = AC) (angle_eq : angle_B = angle_D)
variables (AB BC CD DA : ℚ)

theorem quadrilateral_opposite_sides_equal 
    (h1 : angle_eq)
    (h2 : bisection) : 
    AB = CD ∧ BC = DA :=
sorry

end quadrilateral_opposite_sides_equal_l773_773596


namespace housewife_saved_approx_12_12_percent_l773_773687

noncomputable def percentage_saved (amount_saved sale_price: ℝ) : ℝ :=
  let original_price := sale_price + amount_saved
  (amount_saved / original_price) * 100

theorem housewife_saved_approx_12_12_percent :
  percentage_saved 4 29 ≈ 12.12 :=
by
  have h_orig_price : 29 + 4 = 33 := by norm_num
  have h_percent_saved : percentage_saved 4 29 = (4 / 33) * 100 := by
    -- Calculate percentage saved
    rw [h_orig_price, percentage_saved]
    norm_num
  -- Approximate the percentage saved
  have h_approx : (4 / 33) * 100 ≈ 12.12 := by norm_num
  -- Combine the facts
  rw h_percent_saved
  exact h_approx
  sorry

end housewife_saved_approx_12_12_percent_l773_773687


namespace limit_of_growth_series_l773_773689

theorem limit_of_growth_series :
  let series := 2 + (∑' n : ℕ, (1 / (2^(n+1)))) + sqrt (3) * (∑' n : ℕ, (1 / (3^(n+1))))
  in series = 3 + (1 / 2) * sqrt (3) :=
by
  sorry

end limit_of_growth_series_l773_773689


namespace constant_term_binom_expansion_l773_773602

theorem constant_term_binom_expansion : 
  let general_term (r : ℕ) : ℤ := (-1)^r * (Nat.choose 6 r) * x^(6 - 2*r) in
  ∃ r : ℕ, 6 - 2 * r = 0 ∧ general_term r = -20 :=
begin
  sorry
end

end constant_term_binom_expansion_l773_773602


namespace intersect_points_l773_773335

-- Given definitions
def hyperbola : Set (ℝ × ℝ) := {p | p.1 ^ 2 - p.2 ^ 2 = 1}
def intersects_hyperbola (L : Set (ℝ × ℝ)) : ℝ := 
  (L ∩ hyperbola).finite.to_finset.card

-- Assume lines used
variable {L1 L2 : Set (ℝ × ℝ)}

-- Conditions
def line_circle_intersect_at_one (L1 : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) : Prop :=
  ∃ p ∈ L1 ∩ C, (L1 ∩ C).finite.to_finset.card = 1

def line_intersects_hyperbola (L : Set (ℝ × ℝ)) : Prop := 
  ∃ p ∈ L ∩ hyperbola, 
    ([p].to_finset ⊆ (L ∩ hyperbola).finite.to_finset ∧
     (L ∩ hyperbola).finite.to_finset.card ≥ 1)

def two_lines_not_tangent_to_hyperbola (L1 L2 : Set (ℝ × ℝ)) : Prop := 
  ∃ p₁ p₂ ∈ (L1 ∩ hyperbola).finite.to_finset.to_List, 
    (L1 ∩ hyperbola).finite.to_finset.card = 2 ∧ 
  ∃ q₁ q₂ ∈ (L2 ∩ hyperbola).finite.to_finset.to_List, 
    (L2 ∩ hyperbola).finite.to_finset.card = 2

-- Theorem statement
theorem intersect_points : 
  line_circle_intersect_at_one L1 circle →
  line_intersects_hyperbola L2 →
  line_intersects_hyperbola L1 →
  two_lines_not_tangent_to_hyperbola L1 L2 →
  intersects_hyperbola L1 + intersects_hyperbola L2 = 3 ∨
  intersects_hyperbola L1 + intersects_hyperbola L2 = 4 := 
sorry

end intersect_points_l773_773335


namespace find_ellipse_equation_l773_773104

noncomputable def ellipse_standard_equation (a b : ℝ) : Prop :=
(∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → x = 4 ∧ y = 0)

theorem find_ellipse_equation :
  ∃ (a b : ℝ), (ellipse_standard_equation a b)
  ∧ ((a^2 = 16 ∧ b^2 = 4 ∧ (c : ℝ) = 2 * a * sqrt ((a^2 - b^2))/((a^2 + b^2))) 
  ∨ (a^2 = 16 ∧ b^2 = 64 ∧ (c : ℝ) = 2 * b * sqrt ((b^2 - a^2))/((a^2 + b^2)))) := sorry

end find_ellipse_equation_l773_773104


namespace MrsLacson_sweet_potatoes_and_pumpkins_l773_773933

theorem MrsLacson_sweet_potatoes_and_pumpkins
    (total_harvested : ℝ := 80)
    (sold_to_adams : ℝ := 20)
    (sold_to_lenon : ℝ := 15)
    (traded_for_pumpkins : ℝ := 10)
    (number_of_pumpkins : ℝ := 5)
    (weight_per_pumpkin : ℝ := 3)
    (donation_percentage : ℝ := 5 / 100)
    (average_weight_per_sweet_potato : ℝ := 200 / 1000)
    : 
    let remaining_before_donation := total_harvested - (sold_to_adams + sold_to_lenon + traded_for_pumpkins)
    let donation := donation_percentage * remaining_before_donation
    let rounded_donation := real.ceil donation / 1 
    let number_of_unsold_sweet_potatoes := remaining_before_donation - rounded_donation
    let total_weight_of_pumpkins := number_of_pumpkins * weight_per_pumpkin
    in
    number_of_unsold_sweet_potatoes = 33 ∧ total_weight_of_pumpkins = 15 :=
by
  sorry

end MrsLacson_sweet_potatoes_and_pumpkins_l773_773933


namespace regular_ngon_unique_acute_triangle_l773_773342

theorem regular_ngon_unique_acute_triangle (n : ℕ) (h1 : n = 2017)
  (regular_ngon : ∀ i j, i ≠ j → ¬diagonal_intersects_other_diag (diagonal i j) (diagonal k l) ) :
  ∃! (t : triangle), acute_triangle t :=
sorry

end regular_ngon_unique_acute_triangle_l773_773342


namespace area_of_plot_is_correct_l773_773688

-- Define the side length of the square plot
def side_length : ℝ := 50.5

-- Define the area of the square plot
def area_of_square (s : ℝ) : ℝ := s * s

-- Theorem stating that the area of a square plot with side length 50.5 m is 2550.25 m²
theorem area_of_plot_is_correct : area_of_square side_length = 2550.25 := by
  sorry

end area_of_plot_is_correct_l773_773688


namespace sticker_distribution_probability_l773_773703

theorem sticker_distribution_probability :
  let p := 32
  let q := 50050
  p + q = 50082 :=
sorry

end sticker_distribution_probability_l773_773703


namespace gcd_lcm_product_48_75_l773_773005

theorem gcd_lcm_product_48_75 : 
  let a := 48
  let b := 75
  let gcd_ab := Nat.gcd a b
  let lcm_ab := Nat.lcm a b
  gcd_ab * lcm_ab = 3600 :=
by
  let a := 48
  let b := 75
  let gcd_ab := Nat.gcd a b
  let lcm_ab := Nat.lcm a b
  calc
    gcd_ab * lcm_ab = 3 * 1200 := sorry
                  ... = 3600 := by norm_num

end gcd_lcm_product_48_75_l773_773005


namespace find_polynomials_l773_773272

-- Define polynomials with integer coefficients
variables {R : Type*} [CommRing R]

-- Assuming the main hypothesis on f, g, h, r, s with a prime p
def congruent_polynomials (p : ℤ) [Fact (Nat.Prime p)] (f g h r s : R[X]) :=
  (r * f + s * g ≡ 1 [ℤ] % p) ∧ (f * g ≡ h [ℤ] % p)

theorem find_polynomials (p : ℤ) [Fact (Nat.Prime p)]
  (f g h r s : R[X])
  (H : congruent_polynomials p f g h r s) :
  ∀ (n : ℕ) (hn : 0 < n), ∃ (F G : R[X]),
    (F ≡ f [ℤ] % p) ∧ (G ≡ g [ℤ] % p) ∧ (F * G ≡ h [ℤ] % (p^n)) :=
by
  skip -- Proof not required
  sorry

end find_polynomials_l773_773272


namespace shaded_region_perimeter_l773_773130

noncomputable def circumference (r : ℝ) := 2 * Real.pi * r
noncomputable def arc_length (r : ℝ) (theta : ℝ) := theta / 360 * circumference r

theorem shaded_region_perimeter :
  ∀ r : ℝ, 
  (circumference r = 24) →
  (arc_length r 90 = 6) →
  (3 * arc_length r 90 = 18) :=
by
  intros r hc ha
  rw [arc_length, circumference] at ha
  norm_num at ha
  exact ha.symm ▸ by norm_num

end shaded_region_perimeter_l773_773130


namespace simplify_fraction_l773_773584

theorem simplify_fraction (a b : ℤ) (h : a = 2^6 + 2^4) (h1 : b = 2^5 - 2^2) : 
  (a / b : ℚ) = 20 / 7 := by
  sorry

end simplify_fraction_l773_773584


namespace minimum_quadratic_expression_l773_773650

theorem minimum_quadratic_expression : ∃ (x : ℝ), (∀ y : ℝ, y^2 - 6*y + 5 ≥ -4) ∧ (x^2 - 6*x + 5 = -4) :=
by
  sorry

end minimum_quadratic_expression_l773_773650


namespace sequence_sum_2000_l773_773368

-- Define the sequence using the given pattern
def sequence (n : ℕ) : ℤ :=
  if (n % 5) = 1 ∨ (n % 5) = 4 ∨ (n % 5) = 0 then (n : ℤ)
  else if (n % 5) = 2 ∨ (n % 5) = 3 then -(n : ℤ)
  else 0

-- Define the function to calculate the sum of the sequence up to a certain term
def sequence_sum (n : ℕ) : ℤ :=
  (Finset.range (n + 1)).sum sequence

-- The main proof statement: the sum of the sequence from 1 to 2000 is 400000
theorem sequence_sum_2000 : sequence_sum 2000 = 400000 := 
  sorry

end sequence_sum_2000_l773_773368


namespace distance_between_planes_l773_773411

theorem distance_between_planes :
  let a1 := 3
  let b1 := -1
  let c1 := 2
  let d1 := -4
  let a2 := 3
  let b2 := -1
  let c2 := 2
  let d2 := 7 / 2
  let point := (0, -4, 0)
  let dist := abs (a2 * point.1 + b2 * point.2 + c2 * point.3 + d2) / real.sqrt (a2 ^ 2 + b2 ^ 2 + c2 ^ 2)
  dist = 3 / real.sqrt 14 :=
by
  sorry

end distance_between_planes_l773_773411


namespace feasible_schemes_l773_773264

def inv_prop_func (x : ℝ) : ℝ := 6 / x

def translate_upwards (f : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := f x + k

def translate_right_then_downwards (f : ℝ → ℝ) (a b x : ℝ) : ℝ := f (x - a) - b

def reflect_across_line_y_eq (f : ℝ → ℝ) (y0 : ℝ) (x : ℝ) : ℝ := 2 * y0 - f x

def reflect_across_line_x_eq (f : ℝ → ℝ) (x0 x : ℝ) : ℝ := f (2 * x0 - x)

def translate_right (x0 : ℝ × ℝ) (dx : ℝ) : ℝ × ℝ := (x0.1 + dx, x0.2)

theorem feasible_schemes :
  ((translate_right_then_downwards inv_prop_func 2 2 3 = 4) ∧
   (reflect_across_line_y_eq inv_prop_func 3 3 = 4)) :=
by
  sorry

end feasible_schemes_l773_773264


namespace pages_read_tonight_l773_773001

def sum_of_digits (n: ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem pages_read_tonight :
  let pages_3_nights_ago := 20
  let pages_2_nights_ago := 20^2 + 5
  let pages_last_night := sum_of_digits pages_2_nights_ago * 3
  let total_pages := 500
  total_pages - (pages_3_nights_ago + pages_2_nights_ago + pages_last_night) = 48 :=
by
  sorry

end pages_read_tonight_l773_773001


namespace slant_asymptote_value_l773_773346

theorem slant_asymptote_value :
  ∀ (x : ℝ), (lim (fun y => (3 * x^2 + 2 * x - 5) / (x - 4)) at_top = 3 * x + 14) →
  (m = 3) → (b = 14) → (m + b = 17) :=
begin
  intros x hl hm hb,
  calc
    m + b = 3 + 14 : by {rw [hm, hb]}
        ... = 17    : by norm_num,
end

end slant_asymptote_value_l773_773346


namespace train_crossing_time_l773_773348

theorem train_crossing_time (train_length : ℕ) (bridge_length : ℕ) (speed_kmph : ℕ) : ℕ :=
  let total_distance := train_length + bridge_length
  let speed_mps := (speed_kmph * 1000) / 3600
  let time_seconds := total_distance / speed_mps
  if train_length = 160 ∧ bridge_length = 215 ∧ speed_kmph = 45 then
    time_seconds = 30
  else
    sorry

#print axioms train_crossing_time

end train_crossing_time_l773_773348


namespace find_x_l773_773751

variable (c d : ℝ)

theorem find_x (x : ℝ) (h : x^2 + 4 * c^2 = (3 * d - x)^2) : 
  x = (9 * d^2 - 4 * c^2) / (6 * d) :=
sorry

end find_x_l773_773751


namespace total_units_is_34_l773_773570

-- Define the number of units on the first floor
def first_floor_units : Nat := 2

-- Define the number of units on the remaining floors (each floor) and number of such floors
def other_floors_units : Nat := 5
def number_of_other_floors : Nat := 3

-- Define the total number of floors per building
def total_floors : Nat := 4

-- Calculate the total units in one building
def units_in_one_building : Nat := first_floor_units + other_floors_units * number_of_other_floors

-- The number of buildings
def number_of_buildings : Nat := 2

-- Calculate the total number of units in both buildings
def total_units : Nat := units_in_one_building * number_of_buildings

-- Prove the total units is 34
theorem total_units_is_34 : total_units = 34 := by
  sorry

end total_units_is_34_l773_773570


namespace CircleTouchesSidesOfAngle_l773_773138

theorem CircleTouchesSidesOfAngle
    {A B : Point} 
    {r R : ℝ}
    (h1 : Circle A r)
    (h2 : Circle B R)
    (h3 : CirclesTangent h1 h2)
    (h4 : CircleTouchesAngleSides h1)
    (h5 : CircleTouchesAngleSides h2)
    : CircleTouchesAngleSides (Circle ((A+B)/2) (dist A B / 2)) :=
    sorry

end CircleTouchesSidesOfAngle_l773_773138


namespace number_of_correct_statements_is_two_l773_773470

-- Definitions of the given statements
def statement1_negation_is_correct : Prop :=
  (∀ x y, (x = 0 ∨ y = 0) → (x * y = 0)) ↔ (∀ x y, (x ≠ 0 ∧ y ≠ 0) → (x * y ≠ 0))

def line_perpendicular_cond (a : ℝ) : Prop :=
  (a^2 = 4) ↔ (a = 2 ∨ a = -2)

def statement3_negation_is_correct : Prop :=
  ¬ (∀ x : ℝ, x - real.log x > 0) ↔ (∃ x_0 : ℝ, x_0 - real.log x_0 ≤ 0)

def zero_function_interval : Prop :=
  ∃ x : ℝ, (-1 < x ∧ x < 0 ∧ exp x + x = 0)

-- Main theorem stating the number of correct statements
theorem number_of_correct_statements_is_two :
  ¬statement1_negation_is_correct ∧ 
  ¬line_perpendicular_cond 2 ∧ 
  statement3_negation_is_correct ∧ 
  zero_function_interval →
  2 = 2 := 
by {
  sorry
}

end number_of_correct_statements_is_two_l773_773470


namespace probability_not_less_than_4_probability_less_than_20_l773_773392

-- Define the box with balls
inductive BallColor
| black
| red
| white

def box : List BallColor := [BallColor.black, BallColor.black, BallColor.black, BallColor.red, BallColor.red, BallColor.white]

-- Section (I)
def draw_two_balls (balls : List BallColor) : List (BallColor × BallColor) :=
  List.product balls balls

def reward_amount (draw : BallColor × BallColor) : ℕ :=
  match draw with
  | (BallColor.black, BallColor.black) => 2
  | (BallColor.black, BallColor.red) => 3
  | (BallColor.black, BallColor.white) => 4
  | (BallColor.red, BallColor.red) => 4
  | (BallColor.red, BallColor.white) => 5
  | (BallColor.white, BallColor.white) => 6
  | _ => 0 -- other cases, should not happen

def event_A (balls : List BallColor) : List (BallColor × BallColor) :=
  balls.product balls |>.filter (fun draw => reward_amount draw = 4)

def event_B (balls : List BallColor) : List (BallColor × BallColor) :=
  balls.product balls |>.filter (fun draw => reward_amount draw = 5)

theorem probability_not_less_than_4 (balls : List BallColor) (h : balls = box) : 
  (event_A balls.length + event_B balls.length) / (draw_two_balls balls).length = 2 / 5 := sorry

-- Section (II)
def draw_one_ball (balls : List BallColor) : List BallColor := balls

def reward_amount_2 (draw1 draw2 : BallColor) : ℕ :=
  match (draw1, draw2) with
  | (BallColor.red, BallColor.red) => 20
  | _ => less_than_20 draw1 draw2

def less_than_20 (draw1 draw2 : BallColor): ℕ :=
  match (draw1, draw2) with
  | (BallColor.black, _) => 5  
  | (_, BallColor.black) => 5
  | (BallColor.white, _) => 5
  | (_, BallColor.white) => 5
  | (BallColor.red, BallColor.black) => 10
  | (BallColor.red, BallColor.white) => 10
  | (BallColor.black, BallColor.red) => 10
  | (BallColor.white, BallColor.red) => 10
  | _ => 0 -- other cases, not considered in problem 

def event_C (balls : List BallColor) : List (BallColor × BallColor) :=
  balls.product balls |>.filter (fun draw => reward_amount_2 draw.fst draw.snd = 20)

theorem probability_less_than_20 (balls : List BallColor) (h : balls = box) :
  1 - event_C balls.length / (draw_two_balls balls).length = 8 / 9 := sorry

end probability_not_less_than_4_probability_less_than_20_l773_773392


namespace maximize_sequence_length_l773_773726

theorem maximize_sequence_length :
  ∃ y, y = 1545 ∧ y ∈ ℕ ∧
       (32500 - 21*y > 0) ∧ (34*y - 52500 > 0) :=
by
  sorry

end maximize_sequence_length_l773_773726


namespace find_unique_f_l773_773401

theorem find_unique_f (f : ℝ → ℝ) (h : ∀ x y z : ℝ, f (x * y) + f (x * z) ≥ f (x) * f (y * z) + 1) : 
    ∀ x : ℝ, f x = 1 :=
by
  sorry

end find_unique_f_l773_773401


namespace common_sale_days_once_l773_773322

def is_multiple_of (n k : ℕ) : Prop := ∃ m, k = n * m

def bookstore_sale_days : List ℕ :=
  [4, 8, 12, 16, 20, 24, 28]

def shoe_store_sale_days : List ℕ :=
  [2, 9, 16, 23, 30]

theorem common_sale_days_once :
  (bookstore_sale_days.filter (λ d, d ∈ shoe_store_sale_days)).length = 1 :=
by
  sorry

end common_sale_days_once_l773_773322


namespace weight_of_bowling_ball_l773_773220

-- We define the given conditions
def bowlingBalls := 10
def canoes := 4
def weightOfCanoe := 35

-- We state the theorem we want to prove
theorem weight_of_bowling_ball:
    (canoes * weightOfCanoe) / bowlingBalls = 14 :=
by
  -- Additional needed definitions
  let weightOfCanoes := canoes * weightOfCanoe
  have weightEquality : weightOfCanoes = 140 := by sorry  -- Calculating the total weight of the canoes
  -- Final division to find the weight of one bowling ball
  have weightOfOneBall := weightEquality / bowlingBalls
  show weightOfOneBall = 14 from sorry
  sorry

end weight_of_bowling_ball_l773_773220


namespace factorial_sum_div_l773_773724

theorem factorial_sum_div : ((8.factorial + 9.factorial) / 6.factorial) = 560 := by
  sorry

end factorial_sum_div_l773_773724


namespace rhombus_area_l773_773247

theorem rhombus_area (side diagonal₁ : ℝ) (h_side : side = 20) (h_diagonal₁ : diagonal₁ = 16) : 
  ∃ (diagonal₂ : ℝ), (2 * diagonal₂ * diagonal₂ + 8 * 8 = side * side) ∧ 
  (1 / 2 * diagonal₁ * diagonal₂ = 64 * Real.sqrt 21) := by
  sorry

end rhombus_area_l773_773247


namespace jean_total_calories_l773_773896

-- Define the conditions
def pages_per_donut : ℕ := 2
def written_pages : ℕ := 12
def calories_per_donut : ℕ := 150

-- Define the question as a theorem
theorem jean_total_calories : (written_pages / pages_per_donut) * calories_per_donut = 900 := by
  sorry

end jean_total_calories_l773_773896


namespace percentage_more_l773_773184

variables (J T M : ℝ)

-- Conditions
def Tim_income : Prop := T = 0.90 * J
def Mary_income : Prop := M = 1.44 * J

-- Theorem to be proved
theorem percentage_more (h1 : Tim_income J T) (h2 : Mary_income J M) :
  ((M - T) / T) * 100 = 60 :=
sorry

end percentage_more_l773_773184


namespace smallest_sum_minimum_l773_773785

noncomputable def smallest_sum (x y : ℕ) : ℕ :=
if h₁ : x ≠ y ∧ (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) then x + y else 0

theorem smallest_sum_minimum (x y : ℕ) (h₁ : x ≠ y) (h₂ : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) :
  smallest_sum x y = 96 := sorry

end smallest_sum_minimum_l773_773785


namespace range_of_k_range_of_a_l773_773828

-- Condition for the function to pass through the first and third quadrants.
def passes_through_first_and_third_quadrants (k : ℝ) : Prop :=
  k > 4

-- Condition that given points are in the first quadrant and y1 < y2
def valid_points (a y1 y2 : ℝ) : Prop :=
  a > 0 ∧ y1 < y2 ∧ y2 = (k-4)/(2a + 1) ∧ y1 = (k-4)/(a + 5)

-- Main theorem to prove range of k
theorem range_of_k (k : ℝ) : passes_through_first_and_third_quadrants k → k > 4 :=
by
  sorry

-- Main theorem to prove range of a
theorem range_of_a (a y1 y2 k : ℝ) : valid_points a y1 y2 → (0 < a ∧ a < 4) :=
by
  sorry

end range_of_k_range_of_a_l773_773828


namespace explicit_expression_for_f_l773_773110

def f (x : ℝ) : ℝ := sorry

theorem explicit_expression_for_f :
  (∀ x : ℝ, f(x + 3) = 2 * x - 1) → (∀ x : ℝ, f(x) = 2 * x - 7) :=
by
  intro h
  funext x
  sorry

end explicit_expression_for_f_l773_773110


namespace solve_perimeter_l773_773616

noncomputable def ellipse_perimeter_proof : Prop :=
  let a := 4
  let b := Real.sqrt 7
  let c := 3
  let F1 := (-c, 0)
  let F2 := (c, 0)
  let ellipse_eq (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 7) = 1
  ∀ (A B : ℝ×ℝ), 
    (ellipse_eq A.1 A.2) ∧ (ellipse_eq B.1 B.2) ∧ (∃ l : ℝ, l ≠ 0 ∧ ∀ t : ℝ, (A = (F1.1 + t * l, F1.2 + t * l)) ∨ (B = (F1.1 + t * l, F1.2 + t * l))) 
    → ∃ P : ℝ, P = 16

theorem solve_perimeter : ellipse_perimeter_proof := sorry

end solve_perimeter_l773_773616


namespace minimum_distance_sum_parabola_l773_773774

theorem minimum_distance_sum_parabola :
  ∃ P : ℝ × ℝ, (P.snd^2 = 4 * P.fst) ∧
  (let F := (1 : ℝ, 0 : ℝ),
       Q := (2 : ℝ, 1 : ℝ),
       distance_PF := dist P F,
       distance_PQ := dist P Q in
  distance_PF + distance_PQ = 3) :=
sorry

end minimum_distance_sum_parabola_l773_773774


namespace distinct_terms_in_expansion_l773_773842

theorem distinct_terms_in_expansion :
  let n1 := 2 -- number of terms in (x + y)
  let n2 := 3 -- number of terms in (a + b + c)
  let n3 := 3 -- number of terms in (d + e + f)
  (n1 * n2 * n3) = 18 :=
by
  sorry

end distinct_terms_in_expansion_l773_773842


namespace binom_15_13_eq_105_l773_773374

theorem binom_15_13_eq_105 : Nat.choose 15 13 = 105 := by
  sorry

end binom_15_13_eq_105_l773_773374


namespace math_proof_l773_773375

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def binom (n k : Nat) : Nat :=
  (factorial n) / ((factorial k) * (factorial (n - k)))

theorem math_proof :
  binom 20 6 * factorial 6 = 27907200 :=
by
  sorry

end math_proof_l773_773375


namespace domain_of_f_l773_773734

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 3 * x + 2)

theorem domain_of_f :
  {x : ℝ | (x < 1) ∨ (1 < x ∧ x < 2) ∨ (x > 2)} = 
  {x : ℝ | f x ≠ 0} :=
sorry

end domain_of_f_l773_773734


namespace total_legs_correct_l773_773930

def num_ants : ℕ := 12
def num_spiders : ℕ := 8
def legs_per_ant : ℕ := 6
def legs_per_spider : ℕ := 8
def total_legs := num_ants * legs_per_ant + num_spiders * legs_per_spider

theorem total_legs_correct : total_legs = 136 :=
by
  sorry

end total_legs_correct_l773_773930


namespace min_cost_of_container_l773_773263

noncomputable def min_total_cost (a b : ℝ) : ℝ :=
  let volume := 4
  let height := 1
  let base_cost := 20
  let side_cost := 10
  let S := a * b
  let y := base_cost * S + side_cost * (2 * (a + b))
  y

theorem min_cost_of_container : ∃ (a b : ℝ), a * b = 4 ∧ min_total_cost a b = 160 :=
by 
  existsi (2 : ℝ), existsi (2 : ℝ)
  split
  . exact (4 : ℝ)
  . rfl
  sorry

end min_cost_of_container_l773_773263


namespace triangle_cosine_relation_l773_773537

theorem triangle_cosine_relation 
  (A B C a b c : ℝ) 
  (hA: A ≠ 0 ∧ A ≠ π) -- Ensure the angles are non-degenerate and within a valid range for a triangle
  (hB: B ≠ 0 ∧ B ≠ π)
  (hC: C ≠ 0 ∧ C ≠ π)
  (hA_BC: A + B + C = π) -- Sum of angles in a triangle
  (ha: a = 2*R*sin(A))
  (hb: b = 2*R*sin(B))
  (hc: c = 2*R*sin(C))
  (R: ℝ) -- Circumradius
  :
  a * Real.cos C + c * Real.cos A = b :=
sorry

end triangle_cosine_relation_l773_773537


namespace correct_operation_l773_773295

-- Definitions of the expressions
def A : Prop := sqrt (1 / 2) = 2 * sqrt 2
def B : Prop := sqrt (15 / 2) = (1 / 2) * sqrt 30
def C : Prop := sqrt (37 / 4) = 3.5
def D : Prop := sqrt (8 / 3) = (2 / 3) * sqrt 3

-- Main theorem stating B is the only correct equation
theorem correct_operation :
  ¬ A ∧ B ∧ ¬ C ∧ ¬ D :=
by sorry

end correct_operation_l773_773295


namespace geometric_sequence_sum_property_l773_773161

theorem geometric_sequence_sum_property (a : ℕ → ℝ) (r : ℝ) (n : ℕ) (h_geom : ∀ (m : ℕ), a (m + 1) = a m * r) :
  let X := (finset.range (n+1)).sum (λ k, a k)
  let Y := (finset.range (2*n+1)).sum (λ k, a k)
  let Z := (finset.range (3*n+1)).sum (λ k, a k)
  Y * (Y - X) = X * (Z - X) :=
by
  skip
  let X := (finset.range (n+1)).sum (λ k, a k)
  let Y := (finset.range (2*n+1)).sum (λ k, a k)
  let Z := (finset.range (3*n+1)).sum (λ k, a k)
  sorry

end geometric_sequence_sum_property_l773_773161


namespace range_of_k_l773_773510

-- Define the line and the circle
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 3
def circle (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 2) ^ 2 = 4

-- Define the condition for intersection and chord length
def intersects_at_points_MN (k : ℝ) : Prop :=
  let d := (abs (k * 0 + 3 - 2) / sqrt (1 + k ^ 2)) in
  d <= 1

-- Main theorem to prove
theorem range_of_k (k : ℝ) : intersects_at_points_MN k → (k <= 0) :=
by sorry

end range_of_k_l773_773510


namespace root_in_interval_l773_773315

noncomputable def f (x: ℝ) : ℝ := Real.exp x + 1/2 * x - 2

theorem root_in_interval : 
  (∀ x y : ℝ, x < y → f x < f y) ∧
  f (1/2) < 0 ∧ 
  f 1 > 0 →
  ∃ x : ℝ, 1/2 < x ∧ x < 1 ∧ f x = 0 :=
begin
  assume h,
  sorry
end

end root_in_interval_l773_773315


namespace vector_magnitude_parallel_l773_773212

open Real

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1^2 + v.2^2)

def vectors_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

def m : ℝ × ℝ := (-1, 2)
def n (b : ℝ) : ℝ × ℝ := (2, b)

theorem vector_magnitude_parallel (b : ℝ) (h_parallel : vectors_parallel m (n b)) :
  magnitude (m.1 - n(b).1, m.2 - n(b).2) = 3 * sqrt 5 :=
by
  sorry

end vector_magnitude_parallel_l773_773212


namespace square_difference_example_l773_773287

theorem square_difference_example : 601^2 - 599^2 = 2400 := 
by sorry

end square_difference_example_l773_773287


namespace inequality_for_specific_n_l773_773732

theorem inequality_for_specific_n :
  ∀ (n : ℕ), n > 1 →
  (∀ (x : Fin n → ℝ), (∑ i, x i^2) ≥ x (⟨n - 1, by linarith⟩) * (∑ i in Finset.range (n - 1), x i))
  ↔ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5  :=
by sorry

end inequality_for_specific_n_l773_773732


namespace inequality_half_l773_773093

-- The Lean 4 statement for the given proof problem
theorem inequality_half (a b : ℝ) (h : a > b) : (a / 2 > b / 2) :=
sorry


end inequality_half_l773_773093


namespace min_elements_in_B_l773_773549

theorem min_elements_in_B (A : Set ℝ) (hA : A.card = 11) : 
    ∃ B : Set ℝ, B = {uv | u v : ℝ, u ∈ A, v ∈ A, u ≠ v} ∧ B.card ≥ 17 :=
by sorry

end min_elements_in_B_l773_773549


namespace number_of_elements_in_C_l773_773065

theorem number_of_elements_in_C (A B : set ℕ) (hA : A = {0, 2, 3, 4, 5, 7}) (hB : B = {1, 2, 3, 4, 6}) :
  (∃ (C : set ℕ), C = {x | x ∈ A ∧ x ∉ B} ∧ C.card = 3) :=
sorry

end number_of_elements_in_C_l773_773065


namespace rodney_correct_guess_probability_l773_773940

noncomputable def rodney_probability : ℚ :=
  let tens_digits := {d | d ∈ {6, 8}}
  let unit_digits := {d | d % 2 = 1}
  let eligible_digits := {10 * t + u | t ∈ tens_digits, u ∈ unit_digits}
  if 50 < eligible_digits.min
    then 1 / eligible_digits.card
    else 0

theorem rodney_correct_guess_probability :
  rodney_probability = 1 / 10 :=
by
  sorry

end rodney_correct_guess_probability_l773_773940


namespace integral_example_l773_773398

theorem integral_example : ∫ x in 2..4, (1/x + x) = Real.log 2 + 6 :=
by 
  sorry

end integral_example_l773_773398


namespace find_AD_l773_773072

variable {V : Type} [AddCommGroup V] [Module ℝ V]

variables (A B C D : V)
variables (a b : V)

-- Given conditions
def AB := a
def AC := b
def BD_split := ∃ t : ℝ, BD = t • DC ∧ t = 2

-- Rewrite the construction for AD based on the conditions
theorem find_AD
  (AB AC : V)
  (BD_split : exists t : ℝ, BD = t • (C - D) ∧ t = 2)
  : (A + (2 / 3) • (C - A)) = (1 / 3) • a + (2 / 3) • b := 
sorry

end find_AD_l773_773072


namespace smallest_t_for_r_eq_sin_theta_plot_whole_circle_l773_773238

theorem smallest_t_for_r_eq_sin_theta_plot_whole_circle :
    ∃ t, (∀ θ, 0 ≤ θ ∧ θ ≤ t → r(θ) = r(θ + π)) ∧ 
         (∀ t', (∀ θ, 0 ≤ θ ∧ θ ≤ t' → r(θ) = r(θ + π)) → t ≤ t') :=
by
  let r (θ : ℝ) := sin θ
  use π
  -- Details of proof are skipped
  sorry

end smallest_t_for_r_eq_sin_theta_plot_whole_circle_l773_773238


namespace solve_problem_l773_773367

noncomputable def cubic_root (x : ℝ) : ℝ := x^(1 / 3)

noncomputable def sqrt_cube_root_sqrt_small_number : ℝ :=
  cubic_root (Real.sqrt 0.000008)

theorem solve_problem : Real.to_nearest_thousandth sqrt_cube_root_sqrt_small_number = 0.141 :=
  sorry

end solve_problem_l773_773367


namespace find_ellipse_standard_eq_find_line_eq_and_max_area_l773_773039

section EllipseProblem

def focus_pt : ℝ × ℝ := (0, sqrt 3)
def passing_pt : ℝ × ℝ := (1 / 2, sqrt 3)
def line_eq_params {k t : ℝ} : Prop := 
  (k = sqrt 5 ∧ t = (3 * sqrt 5) / 5) ∨ (k = -sqrt 5 ∧ t = -(3 * sqrt 5) / 5)

theorem find_ellipse_standard_eq (a b c : ℝ) (ha : a = 2) (hb : b = 1) :
  c = sqrt 3 → a^2 - b^2 = c^2 → 
  (passing_pt.snd^2 / a^2 + passing_pt.fst^2 / b^2 = 1) →
  ∀ x y : ℝ, (y^2 / 4 + x^2 = 1) := by
  intros h_focus h_diff_eq h_passing_eq
  sorry

theorem find_line_eq_and_max_area (a b c : ℝ) 
  (ha : a = 2) (hb : b = 1) (A : ℝ × ℝ := (1,0)) :
  (focus_pt = (0, sqrt 3)) →
  (passing_pt = (1/2, sqrt 3)) →
  (forall x y, y^2 / 4 + x^2 = 1) →
  (line_eq_params k t) →
  (|AM| = |AN|) ∧ (AM ⊥ AN) →
  ∃ k t : ℝ, line_eq_params k t ∧ max_area := (64 / 25) := by
  intros h_focus h_passing_eq h_ellipse_eq h_line cond_eq
  sorry

end EllipseProblem

end find_ellipse_standard_eq_find_line_eq_and_max_area_l773_773039


namespace simplify_expression_l773_773206

variable (a : ℝ)

theorem simplify_expression : (a + 1)^2 - a^2 = 2a + 1 :=
by sorry

end simplify_expression_l773_773206


namespace stratified_sampling_l773_773353

theorem stratified_sampling :
  let total_population := 28 + 56 + 84
  let sampling_ratio := 36 / total_population 
  sampling_ratio * 28 = 6 ∧
  sampling_ratio * 56 = 12 ∧
  sampling_ratio * 84 = 18 := 
begin
  simp [total_population, sampling_ratio],
  norm_num, 
end

end stratified_sampling_l773_773353


namespace angle_and_maximum_area_l773_773135

-- Definitions based on conditions
variables (A B C : ℝ) (a b c : ℝ)
-- Law of cosines for angle A (assume condition from problem)
axiom cos_law_for_angle_A : a*cosC + √3*a*sinC - b - c = 0

-- Correct answer: A = π / 3 and maximum area for given a
theorem angle_and_maximum_area (h_cos_law : cos_law_for_angle_A)
  (h_a : a = √3) : 
  (A = π / 3) ∧ (∀ b c : ℝ, MaximumArea := A * b * c / 2, MaximumArea = 3√3 / 4) :=
by
  split
  -- Prove A = π / 3
  sorry,
  -- Prove the maximum area = 3√3 / 4
  sorry

end angle_and_maximum_area_l773_773135


namespace book_has_125_pages_l773_773321

-- Define the number of pages in each chapter
def chapter1_pages : ℕ := 66
def chapter2_pages : ℕ := 35
def chapter3_pages : ℕ := 24

-- Define the total number of pages in the book
def total_pages : ℕ := chapter1_pages + chapter2_pages + chapter3_pages

-- State the theorem to prove that the total number of pages is 125
theorem book_has_125_pages : total_pages = 125 := 
by 
  -- The proof is omitted for the purpose of this task
  sorry

end book_has_125_pages_l773_773321


namespace triangle_incircle_radius_hypotenuse_l773_773670

theorem triangle_incircle_radius_hypotenuse
  (A B C I K L M N : Type)
  (ABC_triangle : is_right_triangle A B C)
  (angle_B : ∠B = 90)
  (incircle_gamma : incircle Γ ABC_triangle)
  (touches_K : touches incircle_gamma AB K)
  (touches_L : touches incircle_gamma BC L)
  (I_is_incenter : is_incenter I Γ ABC_triangle)
  (M_on_AB : is_on_line M AB)
  (N_on_BC : is_on_line N BC)
  (MK : distance M K = 225)
  (NL : distance N L = 64)
  (MN_parallel_AC : is_parallel MN AC) :
  radius Γ = 120 ∧ length AC = 680 := 
sorry

end triangle_incircle_radius_hypotenuse_l773_773670


namespace sufficient_but_not_necessary_l773_773441

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a > 2 → a^2 > 2 * a) ∧ ¬(a^2 > 2 * a → a > 2) :=
by
  sorry

end sufficient_but_not_necessary_l773_773441


namespace baronies_neighbors_impossible_l773_773241

theorem baronies_neighbors_impossible: 
  ∀ (G : SimpleGraph (Fin 19)), (∀ (v : Fin 19), G.degree v = 1 ∨ G.degree v = 5 ∨ G.degree v = 9) → False := 
by 
  sorry

end baronies_neighbors_impossible_l773_773241


namespace min_frac_sum_min_frac_sum_achieved_l773_773021

theorem min_frac_sum (a b : ℝ) (h₁ : 2 * a + 3 * b = 6) (h₂ : 0 < a) (h₃ : 0 < b) :
  (2 / a + 3 / b) ≥ 25 / 6 :=
by sorry

theorem min_frac_sum_achieved :
  (2 / (6 / 5) + 3 / (6 / 5)) = 25 / 6 :=
by sorry


end min_frac_sum_min_frac_sum_achieved_l773_773021


namespace routes_MN_is_8_l773_773644

-- Define the nodes in the path
inductive Node
| M | N | A | B | C | D | X

open Node

def routes : Node → Node → Nat
| M, N := 8 -- This is what we need to prove
| C, N := 1
| D, N := 1
| A, N := 2 -- 1 (C to N) + 1 (D to N)
| B, N := 4 -- 1 (direct B to N) + 2 (A to N) + 1 (C to N)
| X, N := 2 -- 2 (A to N)
| _, _ := 0 -- other transitions are either not allowed or not relevant for N

theorem routes_MN_is_8 : routes M N = 8 :=
by 
  sorry -- No proof required

end routes_MN_is_8_l773_773644


namespace part1_part2_l773_773030

def is_regressive_sequence (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, x n + x (n + 2) - x (n + 1) = x m

theorem part1 (a : ℕ → ℝ) (h : ∀ n : ℕ, a n = 3 ^ n) :
  ¬ is_regressive_sequence a := by
  sorry

theorem part2 (b : ℕ → ℝ) (h_reg : is_regressive_sequence b) (h_inc : ∀ n : ℕ, b n < b (n + 1)) :
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d := by
  sorry

end part1_part2_l773_773030


namespace eval_g_at_8_l773_773503

-- Definition of the function g
def g (x : ℝ) : ℝ := (7 * x + 3) / (x - 2)

-- The proof statement
theorem eval_g_at_8 : g 8 = 59 / 6 := by
  sorry

end eval_g_at_8_l773_773503


namespace matt_homework_time_l773_773566

theorem matt_homework_time (total_time : ℕ) (math_percentage science_percentage history_percentage english_percentage : ℕ) 
    (remaining_subjects_minimum : ℕ) 
    (H_total_time : total_time = 150)
    (H_math_percentage : math_percentage = 20)
    (H_science_percentage : science_percentage = 25)
    (H_history_percentage : history_percentage = 10)
    (H_english_percentage : english_percentage = 15)
    (H_remaining_subjects_minimum : remaining_subjects_minimum = 30) : 
    let math_time := math_percentage * total_time / 100
    let science_time := science_percentage * total_time / 100
    let history_time := history_percentage * total_time / 100
    let english_time := english_percentage * total_time / 100
    let known_subjects_time := math_time + science_time + history_time + english_time
    let remaining_time := total_time - known_subjects_time in
    remaining_time - remaining_subjects_minimum = 15 := 
by {
  let math_time := 0.2 * total_time,
  let science_time := 0.25 * total_time,
  let history_time := 0.1 * total_time,
  let english_time := 0.15 * total_time,
  let known_subjects_time := math_time + science_time + history_time + english_time,
  let remaining_time := total_time - known_subjects_time,
  have H_known_subjects_time : known_subjects_time = 105,
  { calc
      known_subjects_time = 30 + 37.5 + 15 + 22.5 : by {simp [math_time, science_time, history_time, english_time]}
                          ... = 105 : by norm_num
  },
  have H_remaining_time : remaining_time = 45,
  { calc
      remaining_time = total_time - known_subjects_time : by simp [remaining_time]
                   ... = 150 - 105 : by rw [H_total_time, H_known_subjects_time]
                   ... = 45 : by norm_num
  },
  calc
    remaining_time - remaining_subjects_minimum = 45 - 30 : by rw [H_remaining_time, H_remaining_subjects_minimum]
                                    ... = 15 : by norm_num
}

end matt_homework_time_l773_773566


namespace general_formulas_sum_first_n_c_terms_l773_773796

variables (a b : ℕ → ℕ) (d q : ℕ)
variables (S : ℕ → ℕ) (c T : ℕ → ℕ)

-- Conditions
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n+1) - a n = d

def geometric_sequence (b : ℕ → ℕ) : Prop :=
  ∃ q, ∀ n, b (n+1) = q * b n

axiom a1_b1_equals_2 : a 1 = 2 ∧ b 1 = 2
axiom a2_plus_b4_equals_21 : a 2 + b 4 = 21
axiom b4_minus_S3_equals_1 : b 4 - S 3 = 1

-- Sum of first n terms of the arithmetic sequence
def Sn (n : ℕ) : ℕ := 5 * n

-- Problem to prove
theorem general_formulas :
  arithmetic_sequence a → geometric_sequence b →
  a 1 = 2 → b 1 = 2 → a 2 + b 4 = 21 → b 4 - Sn 3 = 1 → 
  (∀ n, a n = 3 * n - 1) ∧ (∀ n, b n = 2^n) :=
by sorry

theorem sum_first_n_c_terms (n : ℕ) :
  (∀ n, a n = 3 * n - 1) → (∀ n, b n = 2^n) →
  ∃ T, T n = (3 * n - 4) * 2^(n + 1) + 8 :=
by sorry

end general_formulas_sum_first_n_c_terms_l773_773796


namespace smallest_possible_sum_l773_773789

-- Defining the conditions for x and y.
variables (x y : ℕ)

-- We need a theorem to formalize our question with the given conditions.
theorem smallest_possible_sum (hx : x > 0) (hy : y > 0) (hne : x ≠ y) (hxy : 1/x + 1/y = 1/24) : x + y = 100 :=
by
  sorry

end smallest_possible_sum_l773_773789


namespace coefficient_x4_is_160_l773_773131

noncomputable def coefficient_of_x4_in_expansion : ℕ :=
  let expr := (1 + X) * (1 + 2 * X)^5 in
  expr.coeff 4

theorem coefficient_x4_is_160 :
  coefficient_of_x4_in_expansion = 160 :=
sorry

end coefficient_x4_is_160_l773_773131


namespace volume_equivalence_l773_773256

variables {V : Type*} [add_comm_group V] [module ℝ V] [finite_dimensional ℝ V] 
variables (a b c : V)
noncomputable def volume_original : ℝ := abs (a ⬝ (b × c))

noncomputable def volume_new : ℝ := 
abs ((2 • a - b) ⬝ ((b + 4 • c) × (c + 5 • a)))

theorem volume_equivalence 
  (h : volume_original a b c = 6) : 
  volume_new a b c = 108 :=
sorry

end volume_equivalence_l773_773256


namespace remove_red_balls_l773_773188

theorem remove_red_balls (total_balls red_percentage remaining_red_percentage : ℕ) 
  (h1 : total_balls = 600) 
  (h2 : red_percentage = 70) 
  (h3 : remaining_red_percentage = 60) :
  let initial_red_balls := red_percentage * total_balls / 100 in 
  let initial_blue_balls := total_balls - initial_red_balls in
  ∃ (x : ℕ), (initial_red_balls - x) * 100 / (total_balls - x) = remaining_red_percentage :=
begin
  sorry
end

end remove_red_balls_l773_773188


namespace line_BF_passes_through_circumcenter_l773_773551

variables {A B C H D E F O : Type} [MetricAffineSpace ℝ A B C]

-- Conditions
def is_acute_angled_triangle (A B C : Type) [MetricAffineSpace ℝ A B C] : Prop :=
 ∃ (H : Type), altitude B A C H ∧ is_midpoint D A B ∧ is_midpoint E A C ∧ reflection H E D F

-- Proof problem statement
theorem line_BF_passes_through_circumcenter
  (A B C : Type) [MetricAffineSpace ℝ A B C]
  (h : is_acute_angled_triangle A B C)
  (O : Type) [IsCircumcenter ℝ A B C O] :
  passes_through (line B F) O :=
by
  sorry -- Proof goes here

end line_BF_passes_through_circumcenter_l773_773551


namespace points_scores_l773_773515

-- Definitions for the teams and their scores
def Teams := {A, B, C, D, E}
variable (points : Teams → ℕ)

-- Conditions
axiom participated_once : ∀ t : Teams, ∀ t' : Teams, t ≠ t' → plays_with t t'
axiom points_awarded : 
  ∀ t1 t2 : Teams, ∃ p1 p2, (t1 ≠ t2) 
  ∧ (p1 = 3 ∧ p2 = 0 ∨ p2 = 3 ∧ p1 = 0 ∨ p1 = 1 ∧ p2 = 1) 
  ∧ (points t1 = p1 + points t1 ∧ points t2 = p2 + points t2)

axiom all_diff_points : injective points

axiom A_most_points : 
  ∀ t : Teams, t ≠ Teams.A → points t < points Teams.A

axiom A_lost_to_B : 
  points Teams.A = points Teams.A - 3 + points Teams.B

axiom B_C_no_losses : 
  ∀ t1 t2 : Teams, (t1 = Teams.B ∨ t1 = Teams.C) ∧ t2 ≠ t1 → (3 = 1 ∨ t1 = t2)

axiom C_fewer_points_D : 
  points Teams.C < points Teams.D

-- The theorem statement
theorem points_scores :
  points Teams.A = 7 
  ∧ points Teams.B = 6 
  ∧ points Teams.C = 4 
  ∧ points Teams.D = 5
  ∧ points Teams.E = 2 :=
sorry

end points_scores_l773_773515


namespace solution_to_fx_eq_10_l773_773921

def f (x : ℝ) : ℝ :=
  if x < -1 then
    5 * x + 10
  else
    x^2 + 8 * x + 15

theorem solution_to_fx_eq_10 :
  (∀ x : ℝ, f x = 10 → x = -4 + Real.sqrt 11 ∨ x = -4 - Real.sqrt 11) :=
by
  intros x h
  sorry

end solution_to_fx_eq_10_l773_773921


namespace correct_statements_B_and_C_l773_773793

-- Given real numbers a, b, c satisfying the conditions
variables (a b c : ℝ)
variables (h1 : a > b)
variables (h2 : b > c)
variables (h3 : a + b + c = 0)

theorem correct_statements_B_and_C : (a - c > 2 * b) ∧ (a ^ 2 > b ^ 2) :=
by
  sorry

end correct_statements_B_and_C_l773_773793


namespace total_spent_amount_l773_773224

def admission_tickets : ℝ := 45
def discount : ℝ := 0.10 * admission_tickets
def discounted_price : ℝ := admission_tickets - discount
def food_cost : ℝ := discounted_price - 13
def food_tax : ℝ := 0.08 * food_cost
def total_food_cost : ℝ := food_cost + food_tax
def transportation : ℝ := 25
def souvenirs : ℝ := 40
def games : ℝ := 28

theorem total_spent_amount :
  (discounted_price + total_food_cost + transportation + souvenirs + games) = 163.20 :=
by
  let admission_tickets := 45
  let discount := 0.10 * admission_tickets
  let discounted_price := admission_tickets - discount
  let food_cost := discounted_price - 13
  let food_tax := 0.08 * food_cost
  let total_food_cost := food_cost + food_tax
  let transportation := 25
  let souvenirs := 40
  let games := 28
  sorry

end total_spent_amount_l773_773224


namespace find_cos_C_find_sin_B_and_area_l773_773512

noncomputable theory

variables {α : Type*} [linear_ordered_field α]

def triangle_sides (a b c : α) := a^2 + b^2 + c^2 - 2 * a * b * c = 0

def cosine_rule {A B C : α} : Prop :=
  ∀ (a b c : α), cos B + (cos A - 2 * sin A) * cos C = 0

-- Part I
theorem find_cos_C {α : Type*} [linear_ordered_field α] 
  (A B C: α) (h: cos B + (cos A - 2 * sin A) * cos C = 0) :
  cos C = 5⁻¹ * (sqrt 5) :=
sorry

-- Part II
theorem find_sin_B_and_area {α : Type*} [linear_ordered_field α] 
  (a b c A B C: α) 
  (ha: a = sqrt 5)
  (hCM: CM = sqrt 2)
  (hcosC: cos C = 5⁻¹ * (sqrt 5)):
  sin B = 5⁻¹ * (sqrt 5) 
  ∧ area ABC = 1 :=
sorry

end find_cos_C_find_sin_B_and_area_l773_773512


namespace maximum_area_of_triangle_ABC_l773_773887

-- Define the conditions
def pa_distance : ℝ := 6
def pb_distance : ℝ := 7
def pc_distance : ℝ := 10
def angle_A : ℝ := 60

-- Define the maximum area theorem
theorem maximum_area_of_triangle_ABC :
  ∃ (ABC : ℝ), (∀ (PA PB PC : point),
  PA.distance = pa_distance ∧
  PB.distance = pb_distance ∧
  PC.distance = pc_distance ∧
  angle_A = 60 ∧
  area_triangle ABC ≤ 36 + 22 * sqrt 3 :=
sorry

end maximum_area_of_triangle_ABC_l773_773887


namespace prime_in_range_l773_773501

theorem prime_in_range (p: ℕ) (h_prime: Nat.Prime p) (h_int_roots: ∃ a b: ℤ, a ≠ b ∧ a + b = -p ∧ a * b = -520 * p) : 11 < p ∧ p ≤ 21 := 
by
  sorry

end prime_in_range_l773_773501


namespace cost_for_23_days_l773_773106

-- Define the cost structure
def costFirstWeek : ℕ → ℝ := λ days => if days <= 7 then days * 18 else 7 * 18
def costAdditionalDays : ℕ → ℝ := λ days => if days > 7 then (days - 7) * 14 else 0

-- Total cost equation
def totalCost (days : ℕ) : ℝ := costFirstWeek days + costAdditionalDays days

-- Declare the theorem to prove
theorem cost_for_23_days : totalCost 23 = 350 := by
  sorry

end cost_for_23_days_l773_773106


namespace max_val_of_g_l773_773416

def g (x : ℝ) : ℝ := 4 * x - x ^ 4

theorem max_val_of_g : ∃ x ∈ Icc (-1 : ℝ) 2, g x = 3 :=
by
  use 1
  split
  { norm_num }
  { rfl }

end max_val_of_g_l773_773416


namespace find_number_of_cups_l773_773572

theorem find_number_of_cups (a C B : ℝ) (h1 : a * C + 2 * B = 12.75) (h2 : 2 * C + 5 * B = 14.00) (h3 : B = 1.5) : a = 3 :=
by
  sorry

end find_number_of_cups_l773_773572


namespace polynomial_asymptotes_l773_773968

theorem polynomial_asymptotes (A B C : ℤ)
  (h_denom : ∀ x, x ∉ {-3, 0, 4} → x^3 + A * x^2 + B * x + C ≠ 0) :
  A = -1 ∧ B = -12 ∧ C = 0 ∧ A + B + C = -13 :=
by {
  sorry
}

end polynomial_asymptotes_l773_773968


namespace administrators_in_sample_l773_773306

theorem administrators_in_sample :
  let total_employees := 160
  let salespeople := 104
  let administrators := 32
  let logistics := 24
  let sample_size := 20
  let proportion_admin := administrators / total_employees
  let admin_in_sample := sample_size * proportion_admin
  admin_in_sample = 4 :=
by
  intros
  sorry

end administrators_in_sample_l773_773306


namespace students_neither_football_nor_cricket_l773_773192

theorem students_neither_football_nor_cricket 
  (total_students : ℕ) 
  (football_players : ℕ) 
  (cricket_players : ℕ) 
  (both_players : ℕ) 
  (H1 : total_students = 410) 
  (H2 : football_players = 325) 
  (H3 : cricket_players = 175) 
  (H4 : both_players = 140) :
  total_students - (football_players + cricket_players - both_players) = 50 :=
by
  sorry

end students_neither_football_nor_cricket_l773_773192


namespace sin_double_angle_identity_l773_773023

theorem sin_double_angle_identity (α : ℝ) (h : sin (α - π / 3) = 2 / 3 + sin α) : 
  sin (2 * α + π / 6) = -1 / 9 :=
by
  sorry

end sin_double_angle_identity_l773_773023


namespace part1_part2_part3_l773_773053

-- Define the function f
noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  Real.log x - m * x^2 + (1 - 2 * m) * x + 1

-- Part 1: Value of m and the extreme value of f
theorem part1 (h : f 1 m = -1) : m = 1 ∧ ∃ x : ℝ, f x 1 = (1 / 4) - Real.log 2 := 
sorry

-- Part 2: Monotonicity of the function
theorem part2 (m : ℝ) : 
  (m ≤ 0 → ∀ x > 0, 0 ≤ deriv (λ x, f x m) x) ∧
  (m > 0 → ∀ x > 0, if x < 1 / (2 * m) then 0 ≤ deriv (λ x, f x m) x else deriv (λ x, f x m) x ≤ 0) := 
sorry

-- Part 3: Smallest integer value of m
theorem part3 : (∀ x > 0, f x m ≤ 0) → ∃ m : ℤ, m = 1 := 
sorry

end part1_part2_part3_l773_773053


namespace compute_z_l773_773502

theorem compute_z :
  let z := (Real.log 4 / Real.log 3) * (Real.log 5 / Real.log 4) * (Real.log 6 / Real.log 5) *
           (Real.log 7 / Real.log 6) * (Real.log 8 / Real.log 7) * (Real.log 9 / Real.log 8) *
           (Real.log 10 / Real.log 9) * (Real.log 11 / Real.log 10) * (Real.log 12 / Real.log 11) *
           (Real.log 13 / Real.log 12) * (Real.log 14 / Real.log 13) * (Real.log 15 / Real.log 14) *
           (Real.log 16 / Real.log 15) * (Real.log 17 / Real.log 16) * (Real.log 18 / Real.log 17) *
           (Real.log 19 / Real.log 18) * (Real.log 20 / Real.log 19)
  in z = (Real.log 20 / Real.log 3) :=
by
  sorry

end compute_z_l773_773502


namespace range_of_a_for_increasing_f_l773_773815

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then x^2 + a * x - 2 else -a^x

-- Define the conditions and the statement to prove
def increasing_on_positive_reals (a : ℝ) : Prop :=
0 < a ∧ a < 1 ∧ -a ≥ a - 1 ∧ ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f a x₁ ≤ f a x₂

theorem range_of_a_for_increasing_f :
  {a : ℝ | a ∈ set.Ioo 0 (1 / 2)} = 
  {a : ℝ | increasing_on_positive_reals a} :=
sorry

end range_of_a_for_increasing_f_l773_773815


namespace parallel_line_distance_l773_773480

/-- Given a line y = (2/3)x + 5, a line L parallel to this line and 3 units away 
is y = (2/3)x + (5 + sqrt(13)) or y = (2/3)x + (5 - sqrt(13)).
-/
theorem parallel_line_distance (x : ℝ) (y : ℝ) :
  (∃ c : ℝ, y = (2/3) * x + c ∧ |c - 5| = sqrt(13)) ↔
  (y = (2/3) * x + (5 + sqrt(13)) ∨ y = (2/3) * x + (5 - sqrt(13))) :=
sorry

end parallel_line_distance_l773_773480


namespace pyramid_sphere_circumscribe_iff_circle_base_circumscribe_l773_773195

theorem pyramid_sphere_circumscribe_iff_circle_base_circumscribe
    (P : Point) (A : Finset Point) :
  (∃ S : Sphere, S.circumscribes (P :: A.to_list)) ↔ (∃ C : Circle, C.circumscribes A.to_list) :=
sorry

end pyramid_sphere_circumscribe_iff_circle_base_circumscribe_l773_773195


namespace sum_even_numbers_from_2_to_60_l773_773719

noncomputable def sum_even_numbers_seq : ℕ :=
  let a₁ := 2
  let d := 2
  let aₙ := 60
  let n := (aₙ - a₁) / d + 1
  n / 2 * (a₁ + aₙ)

theorem sum_even_numbers_from_2_to_60:
  sum_even_numbers_seq = 930 :=
by
  sorry

end sum_even_numbers_from_2_to_60_l773_773719


namespace intersection_of_A_and_B_l773_773794

def A : Set ℝ := {x | 1 < x ∧ x < 7}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x ≤ 5} := by
  sorry

end intersection_of_A_and_B_l773_773794


namespace find_n_l773_773218

-- Defining necessary conditions and declarations
def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def sumOfDigits (n : ℕ) : ℕ :=
  n / 100 + (n / 10) % 10 + n % 10

def productOfDigits (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem find_n (n : ℕ) (s : ℕ) (p : ℕ) 
  (h1 : isThreeDigit n) 
  (h2 : isPerfectSquare n) 
  (h3 : sumOfDigits n = s) 
  (h4 : productOfDigits n = p) 
  (h5 : 10 ≤ s ∧ s < 100)
  (h6 : ∀ m : ℕ, isThreeDigit m → isPerfectSquare m → sumOfDigits m = s → productOfDigits m = p → (m = n → false))
  (h7 : ∃ m : ℕ, isThreeDigit m ∧ isPerfectSquare m ∧ sumOfDigits m = s ∧ productOfDigits m = p ∧ (∃ k : ℕ, k ≠ m → true)) :
  n = 841 :=
sorry

end find_n_l773_773218


namespace EF_parallel_AX_l773_773389

open EuclideanGeometry

variables {P : Type*} [MetricSpace P] [EuclideanSpace P]

-- Configuration of points
variables (A B B' C C' E F X : P)
variables (ℓ : Line P)

-- Conditions
variables 
  (h1 : B ∈ ℓ)
  (h2 : B' ∈ ℓ)
  (h3 : C ∈ ℓ)
  (h4 : C' ∈ ℓ)
  (hA : A ∉ ℓ)
  (hE : LineThrough B (parallelOf (LineThrough A B')) ∩ (LineThrough A C))
  (hF : LineThrough C (parallelOf (LineThrough A C')) ∩ (LineThrough A B))
  (hX : Circle (TriangleCircumcircle A B C) ∩ Circle (TriangleCircumcircle A B' C'))
  (hX_neq_A : A ≠ X)

/- 
 = h_parallel : EF ∥ AX
-/
theorem EF_parallel_AX
  (h_parallel : IsParallel (LineThrough E F) (LineThrough A X)) : IsParallel (LineThrough E F) (LineThrough A X) :=
sorry

end EF_parallel_AX_l773_773389


namespace cookie_cost_1_l773_773141

theorem cookie_cost_1 (C : ℝ) 
  (h1 : ∀ c, c > 0 → 1.2 * c = c + 0.2 * c)
  (h2 : 50 * (1.2 * C) = 60) :
  C = 1 :=
by
  sorry

end cookie_cost_1_l773_773141


namespace smaller_circle_area_l773_773661

theorem smaller_circle_area (P A A' B B' : Point) (r1 r2 : ℝ)
  (h_external_tangent : circles_are_externally_tangent P A A' B B' r1 r2) 
  (h_common_tangents : common_tangents P A A' B B') 
  (h_tangent_length : PA = AB := 4) : 
    circle_area r2 = 2 * π :=
by
  -- We need to prove that the area of the smaller circle with radius r2 is 2π
  sorry

end smaller_circle_area_l773_773661


namespace inverse_of_h_l773_773554

def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := 3 * x - 4
def h (x : ℝ) : ℝ := f (g x)

theorem inverse_of_h : function.inverse h = (fun x => (x + 13) / 12) :=
by
  sorry

end inverse_of_h_l773_773554


namespace boxes_arrangement_l773_773258

open Nat

theorem boxes_arrangement :
  let total_ways := binomial 7 4
  let invalid_ways := 2 * 3 * 2 * binomial 4 2
  total_ways * binomial 5 2 - invalid_ways = 336 := 
by
  sorry

end boxes_arrangement_l773_773258


namespace log16_2_eq_1_over_4_log16_8_eq_3_over_4_l773_773399

theorem log16_2_eq_1_over_4 (log_eq_div : ∀ (b a : ℕ), ∀ (c : ℝ), b = 2^c → real.log a / real.log b = real.log a / (c * real.log 2)) :
  real.log 2 / (4 * real.log 2) = 1 / 4 :=
by 
  have h16 : 16 = 2^4 := by norm_num
  rw [←log_eq_div 16 2 4 h16]
  exact div_eq_iff (by norm_num).mpr rfl

theorem log16_8_eq_3_over_4 (log_eq_div : ∀ (b a : ℕ), ∀ (c : ℝ), b = 2^c → real.log a / real.log b = real.log a / (c * real.log 2)) :
  real.log 8 / (4 * real.log 2) = 3 / 4 :=
by 
  have h16 : 16 = 2^4 := by norm_num
  have h8 : 8 = 2^3 := by norm_num
  rw [←log_eq_div 16 8 4 h16]
  rw [←log_eq_div 8 8 3 h8]
  norm_num
  rw [mul_assoc, ←div_div, div_self (by norm_num : real.log 2 ≠ 0), one_div, mul_one]
  norm_num


end log16_2_eq_1_over_4_log16_8_eq_3_over_4_l773_773399


namespace find_a_l773_773066

variable {R : Type*} [LinearOrder R] [RealSemiring R] [TrigonometricSemiring R]

noncomputable def P (a : R) : Set R := {x | x ≤ a}

noncomputable def Q : Set R := {y | ∃ θ : R, y = Real.sin θ}

theorem find_a (a : R) (h : P a ⊇ Q) : 1 ≤ a :=
begin
  sorry
end

end find_a_l773_773066


namespace main_theorem_l773_773529

noncomputable def parabola : ℝ → ℝ :=
  λ x, x^2 - x - 6

def intersection_points : set (ℝ × ℝ) :=
  {(-2, 0), (3, 0), (0, -6)}

def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - x + 5y - 6 = 0

def point_on_circle (p : ℝ × ℝ) : Prop :=
  circle_equation p.1 p.2

def tangents_perpendicular (A B C : ℝ × ℝ) : Prop :=
  let (cx, cy) := C in
  let (ax, ay) := A in
  let (bx, by) := B in
  (ay - cy) * (by - cy) + (ax - cx) * (bx - cx) = 0

def line_equation (k : ℝ) (y_intercept : ℝ) (x y : ℝ) : Prop :=
  y = k * x + y_intercept

theorem main_theorem :
  (∀ p : ℝ × ℝ, p ∈ intersection_points → point_on_circle p) ∧
  (∃ A B : ℝ × ℝ, A ≠ B ∧ point_on_circle A ∧ point_on_circle B ∧ 
  ∃ l : ℝ → ℝ, line_equation l (-2) 4 3 7 ∧ tangents_perpendicular A B (-0.5, -2.5)) →
  ∃ l : ℝ → ℝ, (line_equation l (-2) ∨ line_equation l 7) :=
sorry

end main_theorem_l773_773529


namespace angle_between_a_b_is_pi_over_3_l773_773111

variable {V : Type} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def angle_between_vectors (a b : V) : ℝ :=
  Real.arccos ((inner a b) / (‖a‖ * ‖b‖))

theorem angle_between_a_b_is_pi_over_3 
  (a b : V) 
  (ha : ‖a‖ = 2) 
  (hb : ‖b‖ = 4) 
  (hab : ⟪a, b⟫ = 4) : 
  angle_between_vectors a b = π / 3 := 
by 
  sorry

end angle_between_a_b_is_pi_over_3_l773_773111


namespace prob_odd_even_subset_m1_expectation_m2_l773_773014

-- Definitions related to the problem
def non_empty_subsets (s : Finset ℕ) : Finset (Finset ℕ) := s.powerset.filter (λ t => t.nonempty)

-- Problem 1: Prove the probability for m = 1
theorem prob_odd_even_subset_m1 : 
  let s := {1, 2, 3, 4, 5}
  let subsets := non_empty_subsets s
  let odd_subsets := subsets.filter (λ t => ∀ x ∈ t, x % 2 = 1)
  let even_subsets := subsets.filter (λ t => ∀ x ∈ t, x % 2 = 0)
  let both_odd_even_subsets := subsets.card - odd_subsets.card - even_subsets.card
  let P_A := (both_odd_even_subsets : ℝ) / subsets.card
  P_A = 21 / 31 := 
sorry

-- Problem 2: Prove the expectation for m = 2
theorem expectation_m2 :
  let s := {1, 2, 3, 4, 5}
  let subsets := non_empty_subsets s
  let diff_num_elements (subset1 subset2 : Finset ℕ) : ℕ := abs (subset1.card - subset2.card)
  let possible_diff := [0, 1, 2, 3, 4]

  -- Probabilities
  let P_0 := (5 * 4 + 10 * 9 + 10 * 9 + 5 * 4) / (31 * 30)
  let P_1 := 2 * (5 * 10 + 10 * 10 + 10 * 5 + 5) / (31 * 30)
  let P_2 := 2 * (5 * 10 + 10 * 10 + 10 * 5 + 5) / (31 * 30)
  let P_3 := 2 * (5 * 5 + 10 * 1) / (31 * 30)
  let P_4 := 2 * (5 * 1) / (31 * 30)
  
  let E_ξ := (0 * P_0 + 1 * P_1 + 2 * P_2 + 3 * P_3 + 4 * P_4)
  E_ξ = 110 / 93 := 
sorry

end prob_odd_even_subset_m1_expectation_m2_l773_773014


namespace f_at_1_over_11_l773_773911

noncomputable def f : (ℝ → ℝ) := sorry

axiom f_domain : ∀ x, 0 < x → 0 < f x

axiom f_eq : ∀ x y, 0 < x → 0 < y → 10 * ((x + y) / (x * y)) = (f x) * (f y) - f (x * y) - 90

theorem f_at_1_over_11 : f (1 / 11) = 21 := by
  -- proof is omitted
  sorry

end f_at_1_over_11_l773_773911


namespace greatest_value_a_maximum_value_a_l773_773414

-- Define the quadratic polynomial
def quadratic (a : ℝ) : ℝ := -a^2 + 9 * a - 20

-- The statement to be proven:
theorem greatest_value_a : ∀ a : ℝ, (quadratic a ≥ 0) → a ≤ 5 := 
sorry

theorem maximum_value_a : quadratic 5 = 0 :=
sorry

end greatest_value_a_maximum_value_a_l773_773414


namespace cats_dogs_ratio_l773_773623

open Real

theorem cats_dogs_ratio (c d : ℕ) (h1 : d.to_nat = (7 * (21 / 7)).to_nat) : d = 15 := by
  sorry

end cats_dogs_ratio_l773_773623


namespace contradiction_method_assumption_l773_773291

theorem contradiction_method_assumption (a b c : ℝ) :
  (¬(a > 0 ∨ b > 0 ∨ c > 0) → false) :=
sorry

end contradiction_method_assumption_l773_773291


namespace hyperbola_properties_l773_773203

theorem hyperbola_properties : 
  let hyperbola := { (x, y) | x^2 - y^2 / 2 = 1 }
  let ellipse := { (x, y) | x^2 / 4 + y^2 = 1 }
  let hyperbola_asymptotes := { y = sqrt 2 * x, y = - sqrt 2 * x }
  let other_hyperbola_asymptotes := { y = sqrt 2 * x, y = - sqrt 2 * x }
  let foci_hyperbola := (± sqrt 3, 0)
  let foci_ellipse := (± sqrt 3, 0)
  let point_on_hyperbola (P : ℝ × ℝ) := P ∈ hyperbola
  let f1 := (- sqrt 3, 0)
  let f2 := (sqrt 3, 0)
  let distance (A B : ℝ × ℝ) := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
in
  (∀ P, point_on_hyperbola P → distance P f1 = 2 * distance P f2 → ∠ f1 P f2 = π / 3) ∧
  (foci_hyperbola = foci_ellipse) ∧
  (hyperbola_asymptotes = other_hyperbola_asymptotes) ∧
  ¬ (∀ (P : ℝ × ℝ), point_on_hyperbola P → (distance P f2 = 4))
:= sorry

end hyperbola_properties_l773_773203


namespace calories_consumed_Jean_l773_773901

def donuts_per_page (pages : ℕ) : ℕ := pages / 2

def calories_per_donut : ℕ := 150

def total_calories (pages : ℕ) : ℕ :=
  let donuts := donuts_per_page pages
  donuts * calories_per_donut

theorem calories_consumed_Jean (h1 : ∀ pages, donuts_per_page pages = pages / 2)
  (h2 : calories_per_donut = 150)
  (h3 : total_calories 12 = 900) :
  total_calories 12 = 900 := by
  sorry

end calories_consumed_Jean_l773_773901


namespace evaluate_expression_l773_773746

theorem evaluate_expression : 
  (Real.sqrt 3 + 3 + (1 / (Real.sqrt 3 + 3))^2 + 1 / (3 - Real.sqrt 3)) = Real.sqrt 3 + 3 + 5 / 6 := by
  sorry

end evaluate_expression_l773_773746


namespace power_mod_equiv_l773_773280

theorem power_mod_equiv : 7^150 % 12 = 1 := 
  by
  sorry

end power_mod_equiv_l773_773280


namespace distinct_sums_not_always_distinct_sums_l773_773548

theorem distinct_sums (k n : ℕ) (a b c : Fin n → ℝ) (h1 : k ≥ 3) 
  (h2 : n > Nat.choose k 3) 
  (h3 : ∀ i j : Fin n, i ≠ j → a i ≠ a j ∧ b i ≠ b j ∧ c i ≠ c j) :
  (∃ (t : Finset ℝ), t.card ≥ k + 1 ∧ 
  (∀ i : Fin n, ∃ s : Finset ℝ, s = {a i + b i, a i + c i, b i + c i} ⊆ t)) :=
sorry

theorem not_always_distinct_sums (k : ℕ) (n : ℕ) (h1 : k ≥ 3)
  (h2 : n = Nat.choose k 3) :
  ¬ (∀ (a b c : Fin n → ℝ), 
    (∀ i j : Fin n, i ≠ j → a i ≠ a j ∧ b i ≠ b j ∧ c i ≠ c j) →
    (∃ (t : Finset ℝ), t.card ≥ k + 1 ∧ 
    (∀ i : Fin n, ∃ s : Finset ℝ, s = {a i + b i, a i + c i, b i + c i} ⊆ t))) :=
sorry

end distinct_sums_not_always_distinct_sums_l773_773548


namespace total_units_is_34_l773_773571

-- Define the number of units on the first floor
def first_floor_units : Nat := 2

-- Define the number of units on the remaining floors (each floor) and number of such floors
def other_floors_units : Nat := 5
def number_of_other_floors : Nat := 3

-- Define the total number of floors per building
def total_floors : Nat := 4

-- Calculate the total units in one building
def units_in_one_building : Nat := first_floor_units + other_floors_units * number_of_other_floors

-- The number of buildings
def number_of_buildings : Nat := 2

-- Calculate the total number of units in both buildings
def total_units : Nat := units_in_one_building * number_of_buildings

-- Prove the total units is 34
theorem total_units_is_34 : total_units = 34 := by
  sorry

end total_units_is_34_l773_773571


namespace largest_sum_of_digits_l773_773097

theorem largest_sum_of_digits (a b c : ℕ) (y : ℕ) 
  (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) 
  (h4 : 0.1 * a + 0.01 * b + 0.001 * c = 1 / y) 
  (h5 : y ∈ {2, 3, 5, 7}) 
  : a + b + c = 5 := 
sorry

end largest_sum_of_digits_l773_773097


namespace find_floor_abs_S_l773_773948

noncomputable def S (x : Fin 1000 → ℝ) : ℝ := ∑ i, x i

theorem find_floor_abs_S {x : Fin 1000 → ℝ}
  (h : ∀ i, x i + (i : ℝ) + 1 = S x + 1001) :
  ⌊|S x|⌋ = 501 := 
sorry

end find_floor_abs_S_l773_773948


namespace smallest_integer_k_l773_773992

def seq (a : ℕ → ℝ) :=
  a 0 = 1 ∧
  a 1 = real.rpow 2 (1/19) ∧
  ∀ n ≥ 2, a n = a (n-1) * (a (n-2))^2

theorem smallest_integer_k
  (a : ℕ → ℝ)
  (h_seq : seq a)
  : ∃ k, (∀ n, (1 ≤ n ∧ n ≤ k) → is_integer (a n)) → is_integer (∏ i in (finset.range k).erase 0, a (i + 1)) ∧ k = 17 :=
by sorry

end smallest_integer_k_l773_773992


namespace compute_expression_l773_773723

theorem compute_expression :
  (-9 * 5 - (-7 * -2) + (-11 * -4)) = -15 :=
by
  sorry

end compute_expression_l773_773723


namespace exists_mn_square_l773_773432

theorem exists_mn_square (k : ℕ) (hk : k ∈ {0, 3, 4, 6, 7}) :
  ∃ m n s : ℕ, m > 0 ∧ n > 0 ∧ s > 0 ∧ 3^m + 3^n + k = s^2 := by
    sorry

end exists_mn_square_l773_773432


namespace number_of_correct_propositions_l773_773813

-- Definitions of the propositions as conditions
def proposition1 (l1 l2 : Line) (p : Plane) : Prop :=
  (intersect l1 l2) -> (intersect (proj l1 p) (proj l2 p))

def proposition2 (l1 l2 : Line) (p : Plane) : Prop :=
  (parallel (proj l1 p) (proj l2 p)) -> (parallel l1 l2 ∨ skew l1 l2)

def proposition3 (l1 l2 : Line) (p : Plane) : Prop :=
  (perpendicular l1 l2) ∧ (perpendicular p p) -> (parallel l2 p)

-- The proof statement specifying the number of correct propositions
theorem number_of_correct_propositions (a b : Line) (α : Plane) :
  ¬ proposition1 a b α ∧ proposition2 a b α ∧ ¬ proposition3 a α := by
  sorry

end number_of_correct_propositions_l773_773813


namespace boat_speed_correct_l773_773319

-- Definitions based on conditions
def stream_speed : ℝ := 4
def downstream_time : ℝ := 3
def downstream_distance : ℝ := 84

-- The speed of the boat in still water
def boat_speed_in_still_water : ℝ := 24

-- The statement we need to prove
theorem boat_speed_correct :
  let V_b := boat_speed_in_still_water in
  downstream_distance = (V_b + stream_speed) * downstream_time →
  V_b = 24 :=
by
  intros h
  sorry

end boat_speed_correct_l773_773319


namespace divide_edges_into_three_equal_segments_l773_773377

structure Tetrahedron (V : Type*) :=
(A B C D : V)

def centroid {V : Type*} [AddCommGroup V] [Module ℝ V] (t : Tetrahedron V) : V :=
(1 / 4 : ℝ) • (t.A + t.B + t.C + t.D)

def divides_in_ratio {V : Type*} [AddCommGroup V] [Module ℝ V] (P Q : V) (α β : ℝ) : Prop :=
∃ K : V, K = (α / (α + β)) • P + (β / (α + β)) • Q

def parallel_to_face {V : Type*} [AddCommGroup V] [Module ℝ V] (plane_point face_points : V × V × V) : Prop :=
∃ (a b c d : ℝ), a • plane_point.1 + b • plane_point.2 + c • plane_point.2 = d

theorem divide_edges_into_three_equal_segments {V : Type*} [AddCommGroup V] [Module ℝ V]
(t : Tetrahedron V) :
let S := centroid t in
let ASₐ := centroid ⟨t.B, t.C, t.D⟩ in
let K := (4 / 9 : ℝ) • t.A + (5 / 9 : ℝ) • S in
let plane_K_BCD := parallel_to_face (K, (t.B, t.C, t.D)) in
let B₁ := (1 / 3 : ℝ) • t.A + (2 / 3 : ℝ) • t.B in
let C₁ := (1 / 3 : ℝ) • t.A + (2 / 3 : ℝ) • t.C in
let D₁ := (1 / 3 : ℝ) • t.A + (2 / 3 : ℝ) • t.D in
(K = B₁ ∧ K = C₁ ∧ K = D₁) :=
sorry

end divide_edges_into_three_equal_segments_l773_773377


namespace polynomial_equality_l773_773211

theorem polynomial_equality (x : ℂ) (h1 : x ^ 2017 - 3 * x + 3 = 0) (h2 : x ≠ 1) : 
  x ^ 2016 + x ^ 2015 + ... + x + 1 = 3 := 
sorry

end polynomial_equality_l773_773211


namespace total_legs_of_collection_l773_773927

theorem total_legs_of_collection (spiders ants : ℕ) (legs_per_spider legs_per_ant : ℕ)
  (h_spiders : spiders = 8) (h_ants : ants = 12)
  (h_legs_per_spider : legs_per_spider = 8) (h_legs_per_ant : legs_per_ant = 6) :
  (spiders * legs_per_spider + ants * legs_per_ant) = 136 :=
by
  sorry

end total_legs_of_collection_l773_773927


namespace zero_function_l773_773731

noncomputable def satisfies_conditions (f : ℝ → ℝ) :=
  (∀ (a b : ℝ), a < b → ∃ I : ℝ, (∀ t, t ∈ set.Icc a b → f t ≤ I)) ∧
  (∀ (x : ℝ) (n : ℕ), 1 ≤ n → f x = (n / 2) * ∫ (t : ℝ) in (set.Ioc (x - (1 / ↑n)) (x + (1 / ↑n))), f t)

theorem zero_function :
  ∀ f : ℝ → ℝ,
    satisfies_conditions f →
    (∀ x : ℝ, f x = 0) :=
by
  intros
  sorry

end zero_function_l773_773731


namespace total_units_l773_773186

theorem total_units (A B C: ℕ) (hA: A = 2 + 4 + 6 + 8 + 10 + 12) (hB: B = A) (hC: C = 3 + 5 + 7 + 9) : 
  A + B + C = 108 := 
sorry

end total_units_l773_773186


namespace find_x_l773_773283

theorem find_x (x : ℝ) : 65 + 5 * 12 / (180 / x) = 66 → x = 3 := 
by
  intro h
  have h1 : 60 / (180 / x) = 1 := by
    calc 
      60 / (180 / x) = (60 * x) / 180 : by
        rw div_eq_mul_one_div
        rw one_div_div
      ... = 60 * (x / 180) : by
        rw mul_div_comm
      ... = 1 : sorry
  exact sorry

end find_x_l773_773283


namespace diagonals_from_vertex_of_polygon_l773_773253

theorem diagonals_from_vertex_of_polygon (n : ℕ) (h1 : (n - 2) * 180 = 540) : 
  (n = 5) → 2 :=
begin
  sorry
end

end diagonals_from_vertex_of_polygon_l773_773253


namespace sum_sequence_formula_l773_773236

variable (a₁ : ℝ) (n : ℕ)
hypothesis h₁ : a₁ > 1

-- Sequence definition
def a (k : ℕ) : ℝ :=
  if k = 0 then a₁ else
  2 ^ (k - 1) * (a₁ - 1) + 1

-- Sum of the first n terms of the sequence
def S_n (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, a k

theorem sum_sequence_formula : S_n n = (a₁ - 1) * (2 ^ n - 1) + n :=
  sorry

end sum_sequence_formula_l773_773236


namespace maximum_area_of_rectangle_with_given_perimeter_l773_773907

theorem maximum_area_of_rectangle_with_given_perimeter {x y : ℕ} (h₁ : 2 * x + 2 * y = 160) : 
  (∃ x y : ℕ, 2 * x + 2 * y = 160 ∧ x * y = 1600) := 
sorry

end maximum_area_of_rectangle_with_given_perimeter_l773_773907


namespace part_I_part_II_l773_773041

noncomputable def omega : ℝ := 2
noncomputable def f (x : ℝ) : ℝ := sin (omega * x - (π / 6)) - 1 / 2

theorem part_I (k : ℤ) :
  ∀ x, (k : ℝ) * π + π / 3 ≤ x ∧ x ≤ (k : ℝ) * π + 5 * π / 6 → 
  f' x < 0 :=
sorry

theorem part_II (A : ℝ) (b : ℝ) (c : ℝ) (S : ℝ) (a : ℝ) :
  (f A = 1 / 2) ∧ (c = 3) ∧ (S = 3 * sqrt 3) →
  (b = 4) →
  a = sqrt 13 :=
sorry

end part_I_part_II_l773_773041


namespace price_of_three_kg_sugar_and_one_kg_salt_is_five_dollars_l773_773978

-- Defining the known quantities
def price_of_two_kg_sugar_and_five_kg_salt : ℝ := 5.50
def price_per_kg_sugar : ℝ := 1.50

-- Defining the variables for the proof
def price_per_kg_salt := (price_of_two_kg_sugar_and_five_kg_salt - 2 * price_per_kg_sugar) / 5

def price_of_three_kg_sugar_and_one_kg_salt := 3 * price_per_kg_sugar + price_per_kg_salt

-- The theorem stating the result
theorem price_of_three_kg_sugar_and_one_kg_salt_is_five_dollars :
  price_of_three_kg_sugar_and_one_kg_salt = 5.00 :=
by
  -- Calculate intermediary values for sugar and salt costs
  let price_of_two_kg_sugar := 2 * price_per_kg_sugar
  let price_of_five_kg_salt := price_of_two_kg_sugar_and_five_kg_salt - price_of_two_kg_sugar
  let price_per_kg_salt' := price_of_five_kg_salt / 5

  -- Calculate final price for verification
  let price_of_three_kg_sugar := 3 * price_per_kg_sugar
  let final_price := price_of_three_kg_sugar + price_per_kg_salt'

  -- Assert the final price is $5.00
  have h1 : price_of_two_kg_sugar = 3.00 := by sorry
  have h2 : price_of_five_kg_salt = 2.50 := by sorry
  have h3 : price_per_kg_salt' = 0.50 := by sorry
  have h4 : final_price = 5.00 := by sorry

  -- Conclude the proof
  exact h4

end price_of_three_kg_sugar_and_one_kg_salt_is_five_dollars_l773_773978


namespace calculate_star_value_l773_773010

def star (a b : ℝ) (h : a ≠ b) : ℝ := (a + b) / (a - b)

theorem calculate_star_value : (star (star 1 3 (by norm_num)) 5 (by norm_num)) = -3 / 7 :=
by sorry

end calculate_star_value_l773_773010


namespace carnation_second_bouquet_l773_773266

/-- Trevor buys three bouquets of carnations.
The first included 9 carnations, and the second included some carnations.
The third bouquet had 13 carnations. If the average number of carnations in the bouquets is 12,
prove that the second bouquet has 14 carnations.
-/
theorem carnation_second_bouquet (x : ℕ) (h₁ : 9 + x + 13 = 3 * 12) : x = 14 :=
by
  have h₂ : 9 + 13 = 22 := by norm_num
  rw [h₂] at h₁
  linarith

end carnation_second_bouquet_l773_773266


namespace ellipse_eccentricity_and_equation_l773_773801

theorem ellipse_eccentricity_and_equation
  (a b : ℝ) (h : a > b) (h1 : b > 0) 
  (M : ℝ × ℝ)
  (hM1 : M.1 = 2 * a / 3) 
  (hM2 : M.2 = b / 3)
  (hM_slope : ((M.2 - 0) / (M.1 - 0)) = (√5 / 10))
  (N : ℝ × ℝ)
  (hN1 : N.1 = -a / 2)
  (hN2 : N.2 = b / 2)
  (S : ℝ × ℝ)
  (hS : S.2 = 13 / 2):
  (∃ e, e = 2 * √5 / 5) ∧ 
  (∃ E : ℝ × ℝ → Prop, E = λ p, (p.1 ^ 2 / 45 + p.2 ^ 2 / 9 = 1)) :=
by
  sorry

end ellipse_eccentricity_and_equation_l773_773801


namespace boat_speed_still_water_l773_773320

theorem boat_speed_still_water (V_b V_c : ℝ) (h1 : 45 / (V_b - V_c) = t) (h2 : V_b = 12)
(h3 : V_b + V_c = 15):
  V_b = 12 :=
by
  sorry

end boat_speed_still_water_l773_773320


namespace number_of_five_digit_sum_two_l773_773249

theorem number_of_five_digit_sum_two : 
  (∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (n.digits.sum = 2)) = 5 := 
sorry

end number_of_five_digit_sum_two_l773_773249


namespace inequality_div_half_l773_773095

theorem inequality_div_half (a b : ℝ) (h : a > b) : (a / 2 > b / 2) :=
sorry

end inequality_div_half_l773_773095


namespace number_of_small_slices_l773_773361

-- Define the given conditions
variables (S L : ℕ)
axiom total_slices : S + L = 5000
axiom total_revenue : 150 * S + 250 * L = 1050000

-- State the problem we need to prove
theorem number_of_small_slices : S = 1500 :=
by sorry

end number_of_small_slices_l773_773361


namespace finite_set_elements_at_least_half_m_l773_773148

theorem finite_set_elements_at_least_half_m (m : ℕ) (A : Finset ℤ) (B : ℕ → Finset ℤ) 
  (hm : 2 ≤ m) 
  (hB : ∀ k : ℕ, 1 ≤ k → k ≤ m → (B k).sum id = (m : ℤ) ^ k) : 
  ∃ n : ℕ, (A.card ≥ n) ∧ (n ≥ m / 2) :=
by
  sorry

end finite_set_elements_at_least_half_m_l773_773148


namespace original_height_is_100_l773_773665

theorem original_height_is_100 :
  ∃ h : ℝ, (
    let h₁ := h in
    let h₂ := (1/2) * h₁ in
    let h₃ := (1/2) * h₂ in
    h₁ + h₂ + h₂ + h₃ + h₃ = 250
  ) → h = 100 :=
by
  sorry

end original_height_is_100_l773_773665


namespace regression_line_interpretation_l773_773990

/-- For the regression line equation y = 256 + 3x, where y is the cost per ton 
    of pig iron (in yuan) and x is the scrap rate (in %), 
    prove that for every 1% increase in the scrap rate, 
    the cost of pig iron per ton increases by 3 yuan. -/
theorem regression_line_interpretation (x : ℝ) (y : ℝ) 
  (h : y = 256 + 3 * x) : ∀ Δx : ℝ, Δy : ℝ, Δx = 1 → Δy = 3 :=
by
  sorry

end regression_line_interpretation_l773_773990


namespace max_odd_partial_sums_l773_773171

def is_permutation (l1 l2 : List ℕ) : Prop :=
  l1.length = l2.length ∧ ∀ x, l1.count x = l2.count x

def partial_sums (l : List ℕ) : List ℕ :=
  l.scanl (· + ·) 0 |>.tail

def max_odd_count (l : List ℕ) : ℕ :=
  (partial_sums l).countp (λ n, n % 2 = 1)

theorem max_odd_partial_sums :
  ∀ (a : List ℕ), 
    is_permutation a (List.range 2014).map (· + 1) →
    max_odd_count a = 1511 := by
  sorry

end max_odd_partial_sums_l773_773171


namespace tan_alpha_plus_pi_cos_alpha_minus_pi_div_two_sin_alpha_plus_3pi_div_two_l773_773795

open Real

theorem tan_alpha_plus_pi (α : ℝ)
  (h1 : sin α = 3 / 5)
  (h2 : π / 2 < α ∧ α < π) :
  tan (α + π) = -3 / 4 :=
sorry

theorem cos_alpha_minus_pi_div_two_sin_alpha_plus_3pi_div_two (α : ℝ)
  (h1 : sin α = 3 / 5)
  (h2 : π / 2 < α ∧ α < π) :
  cos (α - π / 2) * sin (α + 3 * π / 2) = 12 / 25 :=
sorry

end tan_alpha_plus_pi_cos_alpha_minus_pi_div_two_sin_alpha_plus_3pi_div_two_l773_773795


namespace unique_solution_for_f_l773_773402

noncomputable def f : ℝ+ → ℝ+ := sorry

axiom condition1 (x y : ℝ+) : f(x * f(y)) = y * f(x)
axiom condition2 : tendsto f at_top (𝓝 0)

theorem unique_solution_for_f : ∀ x : ℝ+, f(x) = 1 / x := sorry

end unique_solution_for_f_l773_773402


namespace f_1_eq_f_even_f_range_l773_773234

noncomputable def D : Set ℝ := {x | x ≠ 0}

def f : ℝ → ℝ
axiom f_mul (x1 x2 : ℝ) (h1 : x1 ∈ D) (h2 : x2 ∈ D) : f (x1 * x2) = f x1 + f x2

theorem f_1_eq : f 1 = 0 := sorry

theorem f_even : ∀ x, f (-x) = f x := sorry

theorem f_range {x : ℝ}
  (h4 : f 4 = 3)
  (h_bound : ∀ x, f (x - 2) + f (x + 1) ≤ 3)
  (h_inc : ∀ x1 x2, 0 < x1 ∧ x1 < x2 → f x1 ≤ f x2) :
  x ∈ [-2, -1) ∪ (-1, 2) ∪ (2, 3] := sorry

end f_1_eq_f_even_f_range_l773_773234


namespace no_common_multiples_of_3_l773_773942

-- Define the sets X and Y
def SetX : Set ℤ := {n | 1 ≤ n ∧ n ≤ 24 ∧ n % 2 = 1}
def SetY : Set ℤ := {n | 0 ≤ n ∧ n ≤ 40 ∧ n % 2 = 0}

-- Define the condition for being a multiple of 3
def isMultipleOf3 (n : ℤ) : Prop := n % 3 = 0

-- Define the intersection of SetX and SetY that are multiples of 3
def intersectionMultipleOf3 : Set ℤ := {n | n ∈ SetX ∧ n ∈ SetY ∧ isMultipleOf3 n}

-- Prove that the set is empty
theorem no_common_multiples_of_3 : intersectionMultipleOf3 = ∅ := by
  sorry

end no_common_multiples_of_3_l773_773942


namespace part1_part2_l773_773438

noncomputable def A := 
  let a : ℝ := arbitrary
  let b : ℝ := arbitrary
  let B : ℝ := arbitrary
  let m := (a, sqrt (3 : ℝ) * b)
  let n := (Real.cos (π / 2 - B), Real.cos (π - A))
  let dot_product := m.1 * n.1 + m.2 * n.2
  if dot_product = 0
  then π / 3
  else arbitrary

noncomputable def a (A : ℝ) (c : ℝ) (area : ℝ) : ℝ :=
  if A = π / 3 ∧ c = 3 ∧ area = 3 * sqrt (3 : ℝ) / 2
  then sqrt (7 : ℝ)
  else arbitrary

theorem part1 (a b : ℝ) (A B : ℝ) (h : (a, sqrt (3 : ℝ) * b) ⬝ (Real.cos (π / 2 - B), Real.cos (π - A)) = 0) : 
  A = π / 3 :=
sorry

theorem part2 (A c : ℝ) (area : ℝ) (h₁ : A = π / 3) (h₂ : c = 3) (h₃ : area = 3 * sqrt (3 : ℝ) / 2) : 
  a A c area = sqrt(7) :=
sorry

end part1_part2_l773_773438


namespace bob_hair_length_l773_773615

theorem bob_hair_length (h_0 : ℝ) (r : ℝ) (t : ℝ) (months_per_year : ℝ) (h : ℝ) :
  h_0 = 6 ∧ r = 0.5 ∧ t = 5 ∧ months_per_year = 12 → h = h_0 + r * months_per_year * t :=
sorry

end bob_hair_length_l773_773615


namespace proof_equivalent_l773_773063

variable (a b x : ℝ)

def proposition_p := ∀ (a b : ℝ), a > b → a^2 ≤ b^2
def proposition_q := ∀ (x : ℝ), x^2 + 2 < 3x

theorem proof_equivalent :
  ¬ proposition_p ∧ ¬ proposition_q :=
by {
  sorry
}

end proof_equivalent_l773_773063


namespace rainfall_thursday_l773_773139

theorem rainfall_thursday : 
  let monday_rain := 0.9
  let tuesday_rain := monday_rain - 0.7
  let wednesday_rain := tuesday_rain * 1.5
  let thursday_rain := wednesday_rain * 0.8
  thursday_rain = 0.24 :=
by
  sorry

end rainfall_thursday_l773_773139


namespace molecular_weight_BaO_is_correct_l773_773755

-- Define the atomic weights
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight of BaO as the sum of atomic weights of Ba and O
def molecular_weight_BaO := atomic_weight_Ba + atomic_weight_O

-- Theorem stating the molecular weight of BaO
theorem molecular_weight_BaO_is_correct : molecular_weight_BaO = 153.33 := by
  -- Proof can be filled in
  sorry

end molecular_weight_BaO_is_correct_l773_773755


namespace percent_singles_l773_773393

theorem percent_singles (total_hits home_runs triples doubles : ℕ) 
  (h_total: total_hits = 50) 
  (h_hr: home_runs = 3) 
  (h_tr: triples = 2) 
  (h_double: doubles = 8) : 
  100 * (total_hits - (home_runs + triples + doubles)) / total_hits = 74 := 
by
  -- proofs
  sorry

end percent_singles_l773_773393


namespace stories_in_building_l773_773323

-- Definitions of the conditions
def apartments_per_floor := 4
def people_per_apartment := 2
def total_people := 200

-- Definition of people per floor
def people_per_floor := apartments_per_floor * people_per_apartment

-- The theorem stating the desired conclusion
theorem stories_in_building :
  total_people / people_per_floor = 25 :=
by
  -- Insert the proof here
  sorry

end stories_in_building_l773_773323


namespace total_games_played_l773_773301

theorem total_games_played (R G : ℕ) :
  (0.85 * 100).toNat + (0.50 * (R : ℝ)).toNat = (0.70 * (100 + R : ℝ)).toNat → 
  G = 100 + R →
  G = 175 :=
by
  intros h1 h2
  dsimp at *
  sorry

end total_games_played_l773_773301


namespace seq_22_unique_l773_773493

def valid_seq (n : ℕ) (seq : list ℕ) : Prop :=
  seq.length = n ∧
  seq.head = 0 ∧
  seq.last = 0 ∧
  ∀ i, i < seq.length - 1 → seq.nth i = 0 → seq.nth (i+1) ≠ 0 ∧
  seq.filter_indices (λ l, l = [1, 1, 1]).length = 1

theorem seq_22_unique : ∃! seq : list ℕ, valid_seq 22 seq := 
  sorry

end seq_22_unique_l773_773493


namespace cos_A_in_triangle_l773_773452

noncomputable def cos_A (A B : ℝ) (AC BC : ℝ) : ℝ := sorry

theorem cos_A_in_triangle (A B : ℝ) (AC BC : ℝ) (CD : TrianglePoint) (h1 : B = 2 * A)
  (h2 : area_ratio (triangle_area (A C B)) (cd_divides_area_ratio CD (4 : 3))) :
  cos_A A B AC BC = 2 / 3 := by
    sorry

end cos_A_in_triangle_l773_773452


namespace additional_chair_frequency_l773_773633

theorem additional_chair_frequency 
  (workers : ℕ)
  (chairs_per_worker_per_hour : ℕ)
  (hours : ℕ)
  (total_chairs : ℕ) 
  (additional_chairs_rate : ℕ)
  (h_workers : workers = 3) 
  (h_chairs_per_worker : chairs_per_worker_per_hour = 4) 
  (h_hours : hours = 6 ) 
  (h_total_chairs : total_chairs = 73) :
  additional_chairs_rate = 6 :=
by
  sorry

end additional_chair_frequency_l773_773633


namespace solve_sqrt_equation_l773_773405

theorem solve_sqrt_equation (x : ℝ) (hx : x ≥ 2) :
  (\sqrt(x + 5 - 6 * \sqrt(x - 2)) + \sqrt(x + 12 - 8 * \sqrt(x - 2)) = 2) ↔ (x = 11 ∨ x = 27) :=
by sorry

end solve_sqrt_equation_l773_773405


namespace area_of_intersection_is_807_point_5_l773_773642

noncomputable def area_of_intersection_of_congruent_triangles 
  (E F C D : Point) (CD : Line) 
  (hCD : length CD = 12) 
  (hDE : length (segment D E) = 15) 
  (hCF : length (segment C F) = 15) 
  (hEC : length (segment E C) = 20) 
  (hFD : length (segment F D) = 20) 
  (hCongruent : cong_triangl C D E C F D): 
  ℝ :=
  2 * sqrt (23.5 * (23.5 - 12) * (23.5 - 15) * (23.5 - 20))

theorem area_of_intersection_is_807_point_5 
  (E F C D : Point) (CD : Line) 
  (hCD : length CD = 12) 
  (hDE : length (segment D E) = 15) 
  (hCF : length (segment C F) = 15) 
  (hEC : length (segment E C) = 20) 
  (hFD : length (segment F D) = 20) 
  (hCongruent : cong_triangl C D E C F D): 
  area_of_intersection_of_congruent_triangles E F C D CD hCD hDE hCF hEC hFD hCongruent = 807.5 :=
  sorry

end area_of_intersection_is_807_point_5_l773_773642


namespace min_possible_value_l773_773385

noncomputable def min_sum_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : Prop :=
  ∀ (x y z : ℝ), 0 < x → 0 < y → 0 < z → 
    x = a / (3 * b) → y = (b ^ 2) / (6 * c ^ 2) → z = (c ^ 3) / (9 * a ^ 3) →
      x + y + z ≥ 3 / real.cbrt 162

theorem min_possible_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min_sum_value a b c ha hb hc :=
  sorry

end min_possible_value_l773_773385


namespace new_paint_intensity_l773_773947

-- Definition of the given conditions
def original_paint_intensity : ℝ := 0.15
def replacement_paint_intensity : ℝ := 0.25
def fraction_replaced : ℝ := 1.5
def original_volume : ℝ := 100

-- Proof statement
theorem new_paint_intensity :
  (original_volume * original_paint_intensity + original_volume * fraction_replaced * replacement_paint_intensity) /
  (original_volume + original_volume * fraction_replaced) = 0.21 :=
by
  sorry

end new_paint_intensity_l773_773947


namespace cookie_radius_l773_773223

theorem cookie_radius (x y : ℝ) (h : x^2 + y^2 + 36 = 6 * x + 24 * y) : 
  ∃ (r : ℝ), r = 3 * Real.sqrt 13 := 
sorry

end cookie_radius_l773_773223


namespace range_of_t_l773_773622

theorem range_of_t (t : ℝ) (A : set ℝ) : A = {1, t} → (t ≠ 1) :=
by
  intro h
  have : 1 ≠ t := by simp [set.ext_iff] at h; exact h 1 ⟨1, by simp⟩
  sorry

end range_of_t_l773_773622


namespace change_for_50_cents_l773_773085

-- Definitions for the problem conditions
def is_standard_coin (c : ℕ) : Prop := 
  c = 1 ∨ c = 5 ∨ c = 10 ∨ c = 25

-- Definition to check if a list of coins adds up to 50 cents
def makes_fifty_cents (coins : List ℕ) : Prop :=
  coins.filter (≠ 50).sum = 50 ∧ ∀ c ∈ coins, is_standard_coin c

-- The theorem transforming the problem into Lean 4 statement
theorem change_for_50_cents : 
  ∃ (ways : ℕ), ways = 43 ∧ 
  (∀ (coins : List ℕ), makes_fifty_cents coins → coins.permutations.count = ways) :=
sorry

end change_for_50_cents_l773_773085


namespace pass_rate_is_correct_and_average_score_is_correct_l773_773739

noncomputable def pass_rate (scores : List ℝ) (standard : ℝ) : ℝ :=
  let passing_scores := scores.filter (λ s => s <= 0)
  (passing_scores.length.toFloat / scores.length.toFloat) * 100

noncomputable def average_score (scores : List ℝ) (standard : ℝ) : ℝ :=
  (scores.sum + standard * scores.length) / scores.length

theorem pass_rate_is_correct_and_average_score_is_correct :
  let scores := [1, -0.2, 0.4, 0, -1, 0.2, -0.1, -0.3]
  let standard := 18
  pass_rate scores standard = 62.5 ∧ average_score scores standard = 18 :=
by
  sorry

end pass_rate_is_correct_and_average_score_is_correct_l773_773739


namespace locus_of_midpoint_is_rhombus_l773_773604

/--
The cube ABCDA'B'C'D' has upper face ABCD and lower face A'B'C'D', with A directly 
above A' and so on. The point X moves at constant speed along the perimeter of ABCD, 
and the point Y moves at the same speed along the perimeter of B'C'C'B. X leaves A 
towards B at the same moment as Y leaves B' towards C'. Prove that the locus of the 
midpoint of XY is the rhombus CUVW where:
- C = (1, 1, 0)
- U = (0.5, 0.5, 0)
- V = (1, 0.5, 0.5)
- W = (1, 0.5, 0.5)
-/
theorem locus_of_midpoint_is_rhombus :
  let A := (0, 0, 0)
  let B := (1, 0, 0)
  let C := (1, 1, 0)
  let D := (0, 1, 0)
  let A' := (0, 0, 1)
  let B' := (1, 0, 1)
  let C' := (1, 1, 1)
  let D' := (0, 1, 1)
  let U := (0.5, 0.5, 0)
  let V := (1, 0.5, 0.5)
  let W := (1, 0.5, 0.5)
  -- Prove the locus of midpoint of XY forms rhombus CUVW
  (∀ t ∈ [0,1], midpoint (t, 0, 0) (1, t, 1) = ((t + 1) / 2, t / 2, 1 / 2)) ∧
  (∀ t ∈ [0,1], midpoint (1, t, 0) (1, 1, 1 - t) = (1, (t + 1) / 2, (1 - t) / 2)) ∧
  (∀ t ∈ [0,1], midpoint (1 - t, 1, 0) (1, 1 - t, 0) = (1 - t / 2, 1 - t / 2, 0)) ∧
  (∀ t ∈ [0,1], midpoint (0, 1 - t, 0) (1, 0, t) = (1 / 2, (1 - t) / 2, t / 2)) ∧
  locus_of_midpoint_is_rhombus (C as (1, 1, 0)) (U as (0.5, 0.5, 0)) (V as (1, 0.5, 0.5)) (W as (1, 0.5, 0.5))
  := sorry

end locus_of_midpoint_is_rhombus_l773_773604


namespace seconds_hand_revolution_l773_773254

theorem seconds_hand_revolution (revTimeSeconds revTimeMinutes : ℕ) : 
  (revTimeSeconds = 60) ∧ (revTimeMinutes = 1) :=
sorry

end seconds_hand_revolution_l773_773254


namespace susan_spaces_to_win_l773_773214

def spaces_in_game : ℕ := 48
def first_turn_movement : ℤ := 8
def second_turn_movement : ℤ := 2 - 5
def third_turn_movement : ℤ := 6

def total_movement : ℤ :=
  first_turn_movement + second_turn_movement + third_turn_movement

def spaces_to_win (spaces_in_game : ℕ) (total_movement : ℤ) : ℤ :=
  spaces_in_game - total_movement

theorem susan_spaces_to_win : spaces_to_win spaces_in_game total_movement = 37 := by
  sorry

end susan_spaces_to_win_l773_773214


namespace count_three_digit_numbers_between_l773_773497

theorem count_three_digit_numbers_between 
  (a b : ℕ) 
  (ha : a = 137) 
  (hb : b = 285) : 
  ∃ n, n = (b - a - 1) + 1 := 
sorry

end count_three_digit_numbers_between_l773_773497


namespace max_area_equilateral_triangle_l773_773993

/-- The sides of rectangle ABCD have lengths 13 and 14. An equilateral triangle is drawn so that no point of the triangle lies outside ABCD. The maximum possible area of such a triangle can be written in the form p * sqrt q - r, where p, q, and r are positive integers, and q is not divisible by the square of any prime number. Prove that p + q + r = 732. -/
theorem max_area_equilateral_triangle (A B C D : ℝ) (AB BC : ℝ) (p q r : ℕ) 
  (h1 : AB = 13) (h2 : BC = 14)
  (h3 : q > 0) (h4 : ∀ (n : ℕ), (n * n ∣ q) → n = 1)
  (h5 : ∃ (s : ℝ), AB = s * sqrt q - r):
  p + q + r = 732 := 
sorry

end max_area_equilateral_triangle_l773_773993


namespace height_of_scale_model_l773_773713

-- Define the conditions
def scale_ratio : ℝ := 1 / 25
def actual_height : ℝ := 305
def model_height : ℝ := actual_height * scale_ratio

-- The question we want to prove, rounded to the nearest whole number
theorem height_of_scale_model : Real.round model_height = 12 :=
by
  sorry

end height_of_scale_model_l773_773713


namespace find_angle_B_l773_773711

noncomputable def circumradius := sorry
noncomputable def projection := sorry
noncomputable def midpoint := sorry
noncomputable def angle := sorry

variables (A B C K M O : Type) (R : ℝ)
variables [circumcenter : circumradius A B C O R]
variables [projectionOK : projection O (angle B) K]
variables [midpointACM : midpoint A C M]
variables [angle_in_range : ∀ {X Y Z} (a : angle X Y Z), 30 < a ∧ a < 90]

theorem find_angle_B (h₁ : 2 * (KM : ℝ) = R) : angle B = 60 :=
sorry

end find_angle_B_l773_773711


namespace triangle_isosceles_at_F_l773_773158

theorem triangle_isosceles_at_F 
  (A B C D E F : Type) 
  [concyclic {A, B, C, D}] 
  [intersection (line A B) (line C D) E]
  [tangent (circumcircle A D E) D (line D F)] 
  [intersection (line B C) (line D F) F] 
  : is_isosceles (triangle D C F) F := 
sorry

end triangle_isosceles_at_F_l773_773158


namespace height_increase_of_cylinder_l773_773267

theorem height_increase_of_cylinder {r h : ℝ} (r_incr : ℝ) (h_incr : ℝ) (V1 V2 : ℝ) :
  r = 5 → h = 4 → r_incr = 2 →
  V1 = real.pi * (r + r_incr) ^ 2 * h → V2 = real.pi * r^2 * (h + h_incr) →
  V1 = V2 →
  h_incr = 96 / 25 :=
by {
  intros hr hh hr_incr hV1 hV2 heqV,
  sorry
}

end height_increase_of_cylinder_l773_773267


namespace find_positive_number_l773_773805

theorem find_positive_number (a : ℝ) (n : ℝ) 
  (h : 2 * a - 3 = real.sqrt n ∨ 3 * a - 22 = real.sqrt n ) :
  n = 49 := by
  sorry

end find_positive_number_l773_773805


namespace constant_term_expr_l773_773457

noncomputable def a : ℝ := ∫ x in -1..1, real.sqrt (1 - x^2)

def expr (x : ℝ) : ℝ := ((a / real.pi) * x - (1 / x)) ^ 6

theorem constant_term_expr : constant_term (expr x) = -5/2 := 
sorry

end constant_term_expr_l773_773457


namespace slopes_product_ab_fixed_point_l773_773027

noncomputable def ellipse_conditions := sorry

-- QUESTIONS TO PROVE

-- 1. Prove k1 * k2 = -1/3
theorem slopes_product : 
  ∀ (a b : ℝ) (e : ℝ) (M A B : Point) (k1 k2 : ℝ)
  (Ma_geq_B : a > b > 0) (ell_eq : e = Real.sqrt(6) / 3) 
  (ellipse_eq : M ∈ Ellipse a b C) (M_coords : M = (0, 1))
  (symmetric_AB : symmetric_about_origin A B) (slope_sum : k1 + k2 = 3),
  k1 * k2 = -1/3 :=
by sorry

-- 2. Prove Line AB passes through the fixed point (-2/3, -1)
theorem ab_fixed_point :
  ∀ (a b : ℝ) (e : ℝ) (M A B : Point) (k1 k2 : ℝ)
  (Ma_geq_B : a > b > 0) (ell_eq : e = Real.sqrt(6) / 3) 
  (ellipse_eq : M ∈ Ellipse a b C) (M_coords : M = (0, 1))
  (symmetric_AB : symmetric_about_origin A B) (slope_sum : k1 + k2 = 3),
  Line_through A B (-2/3, -1) :=
by sorry


end slopes_product_ab_fixed_point_l773_773027


namespace axis_of_symmetry_cosine_value_l773_773019

def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sqrt 3 * Real.cos x)
def vec_b (x : ℝ) : ℝ × ℝ := (Real.cos x, -Real.cos x)
def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2 + (Real.sqrt 3 / 2)

theorem axis_of_symmetry (x : ℝ) (k : ℤ) : x = 5 * Real.pi / 12 + k * Real.pi / 2 :=
sorry

variable {x1 x2 : ℝ}

theorem cosine_value (h1 : 0 < x1) 
  (h2 : x1 < 5 * Real.pi / 12) 
  (h3 : 5 * Real.pi / 12 < x2) 
  (h4 : x2 < 2 * Real.pi / 3)
  (hx1 : f x1 = 1 / 3)
  (hx2 : f x2 = 1 / 3) : Real.cos (x1 - x2) = 1 / 3 :=
sorry

end axis_of_symmetry_cosine_value_l773_773019


namespace greatest_int_less_than_200_with_gcd_18_eq_9_l773_773648

theorem greatest_int_less_than_200_with_gcd_18_eq_9 :
  ∃ n, n < 200 ∧ Int.gcd n 18 = 9 ∧ ∀ m, m < 200 ∧ Int.gcd m 18 = 9 → m ≤ n :=
sorry

end greatest_int_less_than_200_with_gcd_18_eq_9_l773_773648


namespace probability_product_positive_correct_l773_773270

noncomputable def probability_product_positive : ℚ :=
  let length_total := 45
  let length_negative := 30
  let length_positive := 15
  let prob_negative := (length_negative : ℚ) / length_total
  let prob_positive := (length_positive : ℚ) / length_total
  let prob_product_positive := prob_negative^2 + prob_positive^2
  prob_product_positive

theorem probability_product_positive_correct :
  probability_product_positive = 5 / 9 :=
by
  sorry

end probability_product_positive_correct_l773_773270


namespace smallest_in_consecutive_odds_with_median_and_greatest_l773_773421

-- Define the set of consecutive odd integers
def is_odd (n : ℤ) : Prop := n % 2 = 1

def consecutive_odds (a b : ℤ) (S : set ℤ) : Prop :=
  ∀ x ∈ S, is_odd x ∧ ∃ (m : ℤ), S = {m - (b - m), m - (b - m + 2), ..., m + 2, m, ..., b - 2, b}

-- Define the conditions
def median (S : set ℤ) (m : ℤ) : Prop :=
  ∃ (T U : set ℤ), S = T ∪ U ∧ ∀ t ∈ T, t < m ∧ ∀ u ∈ U, u > m

def greatest (S : set ℤ) (g : ℤ) : Prop :=
  ∀ s ∈ S, s ≤ g ∧ g ∈ S

def smallest (S : set ℤ) (s : ℤ) : Prop :=
  ∀ x ∈ S, s ≤ x ∧ s ∈ S

-- Define the lean statement
theorem smallest_in_consecutive_odds_with_median_and_greatest {S : set ℤ} :
  consecutive_odds (133 : ℤ) (167 : ℤ) S ∧ median S 150 ∧ greatest S 167 → smallest S 133 :=
sorry

end smallest_in_consecutive_odds_with_median_and_greatest_l773_773421


namespace crocodiles_count_l773_773523

-- Definitions of constants
def alligators : Nat := 23
def vipers : Nat := 5
def total_dangerous_animals : Nat := 50

-- Theorem statement
theorem crocodiles_count :
  total_dangerous_animals - alligators - vipers = 22 :=
by
  sorry

end crocodiles_count_l773_773523


namespace percentage_increase_l773_773957

theorem percentage_increase (N P : ℕ) (h1 : N = 40)
       (h2 : (N + (P / 100) * N) - (N - (30 / 100) * N) = 22) : P = 25 :=
by 
  have p1 := h1
  have p2 := h2
  sorry

end percentage_increase_l773_773957


namespace task_completion_time_and_last_tasks_l773_773909

def task_condition (N : ℕ) := 
  ∀ k : ℕ, 1 ≤ k → k ≤ N → (∀ d : ℕ, d ∣ k → d ≠ k → (d = 1 ∨ d ≤ k - 1))

def height (n : ℕ) : ℕ :=
  n.factorization.values.sum

theorem task_completion_time_and_last_tasks 
  (N : ℕ)
  (hN : N = 2017)
  (h1 : task_condition N ∧ ∀ i, (1 ≤ i ∧ i ≤ N ∧ ∀ j, j ∣ i → ∀ k, j ≠ k → k ≠ i → k ∣ i → k < j)) :
  ∃ t : ℕ, t = 2 ∧ (∀ i : ℕ, i ∈ { k | 1 ≤ k ∧ k ≤ N ∧ height k = 1 } → prime i) :=
by 
  sorry

end task_completion_time_and_last_tasks_l773_773909


namespace reciprocal_of_2023_l773_773983

theorem reciprocal_of_2023 : 1 / 2023 = (1 : ℚ) / 2023 :=
by sorry

end reciprocal_of_2023_l773_773983


namespace system_of_equations_l773_773070

theorem system_of_equations (x y : ℝ) 
  (h1 : 2 * x + y = 5) 
  (h2 : x + 2 * y = 4) : 
  x + y = 3 :=
sorry

end system_of_equations_l773_773070


namespace sin_A_value_l773_773128

theorem sin_A_value (A B C : Type) [Triangle A B C] (h_angle_B : angle B = 90) (h_relation : 4 * sin A = 5 * cos A) :
  sin A = 5 / sqrt 41 * (sqrt 41 / 41) :=
by
  sorry

end sin_A_value_l773_773128


namespace ratio_of_increase_proof_l773_773341

variable (T : ℕ) -- Total marks of all the pupils initially
variable (A : ℕ) -- Original average marks

-- Define the number of pupils and the wrongly entered marks
def num_pupils : ℕ := 80
def wrong_mark : ℕ := 85
def correct_mark : ℕ := 45

-- Total increase in marks
def increase_in_marks : ℕ := wrong_mark - correct_mark

-- New total marks after wrong entry
def new_total_marks : ℕ := T + increase_in_marks

-- Original average marks
def original_average_marks : ℕ := T / num_pupils

-- New average marks after wrong entry
def new_average_marks : ℕ := new_total_marks / num_pupils

-- Increase in average marks
def increase_in_average_marks : ℕ := new_average_marks - original_average_marks

-- Ratio of the increase in average marks to the original average marks
def ratio_of_increase (A : ℕ) : ℚ := (1 / 2 : ℚ) / A

theorem ratio_of_increase_proof (T A : ℕ) (h : A = T / num_pupils) :
  ratio_of_increase A = 1 / (2 * A) :=
by
  unfold ratio_of_increase
  sorry

end ratio_of_increase_proof_l773_773341


namespace mango_coconut_ratio_l773_773578

open Function

theorem mango_coconut_ratio
  (mango_trees : ℕ)
  (coconut_trees : ℕ)
  (total_trees : ℕ)
  (R : ℚ)
  (H1 : mango_trees = 60)
  (H2 : coconut_trees = R * 60 - 5)
  (H3 : total_trees = 85)
  (H4 : total_trees = mango_trees + coconut_trees) :
  R = 1/2 :=
by
  sorry

end mango_coconut_ratio_l773_773578


namespace correct_operation_l773_773294

-- Definitions of the expressions
def A : Prop := sqrt (1 / 2) = 2 * sqrt 2
def B : Prop := sqrt (15 / 2) = (1 / 2) * sqrt 30
def C : Prop := sqrt (37 / 4) = 3.5
def D : Prop := sqrt (8 / 3) = (2 / 3) * sqrt 3

-- Main theorem stating B is the only correct equation
theorem correct_operation :
  ¬ A ∧ B ∧ ¬ C ∧ ¬ D :=
by sorry

end correct_operation_l773_773294


namespace lattice_points_count_l773_773129

theorem lattice_points_count : ∃ n : ℕ, n = 16 ∧ 
  ∀ (x y : ℤ), ((|x| - 1)^2 + (|y| - 1)^2 < 2) ↔ (x, y) ∈ finset.univ.filter (λ p, (p.1, p.2) ∈ ℤ × ℤ) := sorry

end lattice_points_count_l773_773129


namespace value_decrease_proof_l773_773860

noncomputable def value_comparison (diana_usd : ℝ) (etienne_eur : ℝ) (eur_to_usd : ℝ) : ℝ :=
  let etienne_usd := etienne_eur * eur_to_usd
  let percentage_decrease := ((diana_usd - etienne_usd) / diana_usd) * 100
  percentage_decrease

theorem value_decrease_proof :
  value_comparison 700 300 1.5 = 35.71 :=
by
  sorry

end value_decrease_proof_l773_773860


namespace exists_zero_point_in_interval_l773_773753

-- Let f be the given function
def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem exists_zero_point_in_interval : ∃ c ∈ Ioo (2 : ℝ) (3 : ℝ), f c = 0 := by
  sorry

end exists_zero_point_in_interval_l773_773753


namespace not_all_yellow_l773_773357

def color := ℕ
def green : color := 0
def red : color := 1
def yellow : color := 2

noncomputable def next_color (left: color) (right: color) : color :=
  if left = right then left
  else if (left = green ∧ right = red) ∨ (left = red ∧ right = green) then yellow
  else if (left = green ∧ right = yellow) ∨ (left = yellow ∧ right = green) then red
  else green

noncomputable def next_colors (lights: list color) : list color := 
  list.map₂ next_color (lights.insert_nth (lights.length - 1) (lights.head_nth 0)) lights

theorem not_all_yellow : 
  ∃ (n : ℕ) (next_colors : ℕ → list color) (current : list color),
    n = 1998 ∧ 
    current.head_nth 0 = red ∧ 
    current.tail.all (= green) ∧ 
    ∀ i : ℕ, current = next_colors i → ∃ j : ℕ, next_colors j ↔ next_colors j.head_nth 0 ≠ yellow 
    :=
by
  sorry

end not_all_yellow_l773_773357


namespace diagonal_sums_difference_l773_773378

def original_matrix : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![21, 22, 23], ![24, 25, 26], ![27, 28, 29]]

def transformed_matrix : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![21, 22, 23], ![26, 25, 24], ![27, 28, 29]]

def main_diagonal_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  m 0 0 + m 1 1 + m 2 2

def anti_diagonal_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  m 0 2 + m 1 1 + m 2 0

theorem diagonal_sums_difference :
  |main_diagonal_sum transformed_matrix - anti_diagonal_sum transformed_matrix| = 0 :=
by
  sorry

end diagonal_sums_difference_l773_773378


namespace min_value_fraction_l773_773767

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  ∃ c, (c = 9) ∧ (∀ x y, (x > 0 ∧ y > 0 ∧ x + y = 1) → (1/x + 4/y ≥ c)) :=
by
  sorry

end min_value_fraction_l773_773767


namespace range_of_f_l773_773757

noncomputable def f (t : ℝ) : ℝ := (t^2 + (1/2)*t) / (t^2 + 1)

theorem range_of_f : Set.Icc (-1/4 : ℝ) (1/4) = Set.range f :=
by
  sorry

end range_of_f_l773_773757


namespace solve_quadratic_equation_l773_773587

theorem solve_quadratic_equation (x : ℝ) : 
  2 * x^2 - 4 * x = 6 - 3 * x ↔ (x = -3/2 ∨ x = 2) :=
by
  sorry

end solve_quadratic_equation_l773_773587


namespace probability_two_of_three_same_color_l773_773675

/-- A bag contains 2 white balls, 3 black balls, and 4 red balls. Prove that the probability of drawing 3 balls such that exactly 2 of them are the same color is 55/84. --/
theorem probability_two_of_three_same_color :
  let total_balls := 9,
      choose_three := (9.choose 3).toRat,
      prob_two_white := (2.choose 2).toRat * (7.choose 1).toRat / choose_three,
      prob_two_black := (3.choose 2).toRat * (6.choose 1).toRat / choose_three,
      prob_two_red := (4.choose 2).toRat * (5.choose 1).toRat / choose_three in
    (prob_two_white + prob_two_black + prob_two_red) = (55 / 84 : ℚ) :=
by
  sorry

end probability_two_of_three_same_color_l773_773675


namespace determine_a_l773_773832

def P : Set ℝ := {1, 2}
def Q (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

theorem determine_a (a : ℝ) : P ∪ Q a = P ↔ a ∈ {0, -2, -1} := by sorry

end determine_a_l773_773832


namespace find_yellow_shells_l773_773217

-- Define the conditions
def total_shells : ℕ := 65
def purple_shells : ℕ := 13
def pink_shells : ℕ := 8
def blue_shells : ℕ := 12
def orange_shells : ℕ := 14

-- Define the result as the proof goal
theorem find_yellow_shells (total_shells purple_shells pink_shells blue_shells orange_shells : ℕ) : 
  total_shells = 65 →
  purple_shells = 13 →
  pink_shells = 8 →
  blue_shells = 12 →
  orange_shells = 14 →
  65 - (13 + 8 + 12 + 14) = 18 :=
by
  intros
  sorry

end find_yellow_shells_l773_773217


namespace inequality_solution_equality_condition_l773_773176

variables {n k : ℕ} (a : ℕ → ℝ)
hypothesis (h_n : n ≥ 2) (h_k : k ≥ 1)
hypothesis (h_nonneg : ∀ i, 1 ≤ i ∧ i ≤ n → a i ≥ 0)
hypothesis (h_sum : (∑ i in range n, a (i + 1)) = n)
noncomputable def cyclic_a := a ⟨_, nat.mod_lt _ (nat.succ_pos n)⟩
hypothesis (h_cyclic : a (n + 1) = a 1)

theorem inequality_solution :
  (∑ i in range n, (a i + 1)^(2 * k) / (a (i + 1) + 1)^k) ≥ n * 2^k :=
sorry

theorem equality_condition :
  (∑ i in range n, (a i + 1)^(2 * k) / (a (i + 1) + 1)^k) = n * 2^k ↔ ∀ i, a i = 1 :=
sorry

end inequality_solution_equality_condition_l773_773176


namespace parallel_lines_slope_l773_773032

theorem parallel_lines_slope (a : ℝ) (h1 : ∀ x y, ax + 3*y + 1 = 0) (h2 : ∀ x y, 2*x + (a + 1)*y + 1 = 0) 
  (h_parallel: (∃ x, 3*x + 1 ≠ 0) ∧ (∃ x y, (a*x + 3*y + 1 = 0 ∧ 2*x + (a + 1)*y + 1 = 0))) :
  a = -3 :=
sorry

end parallel_lines_slope_l773_773032


namespace parallel_AB_XY_l773_773178

variables {k : ℝ}
def A : ℝ × ℝ := (-6, 2)
def B : ℝ × ℝ := (2, -6)
def X : ℝ × ℝ := (0, 10)
def Y : ℝ × ℝ := (18, k)

-- Define a function to calculate the slope
def slope (P Q : ℝ × ℝ) : ℝ :=
  (Q.2 - P.2) / (Q.1 - P.1)

theorem parallel_AB_XY : (slope A B) = (slope X Y) → k = -8 := by
  sorry

end parallel_AB_XY_l773_773178


namespace no_n_consecutive_composite_l773_773404

theorem no_n_consecutive_composite (n : ℕ) (h : n > 0) : 
  (¬ ∃ k : ℕ, (∀ i : ℕ, 0 ≤ i ∧ i < n → Nat.not_prime (k + i)) ∧ k + n < Nat.factorial n) ↔ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 :=
by
  sorry

end no_n_consecutive_composite_l773_773404


namespace find_a_plus_t_l773_773812

variable (n : ℝ) (a : ℝ) (t : ℝ)

theorem find_a_plus_t (h₁ : n >= 2) (h₂ : ∀ n, a = n ∧ t = n^2 - 1)
  (h₃ : √(n + a / t) = n * √(a / t)) :
  a + t = n^2 + n - 1 :=
sorry

end find_a_plus_t_l773_773812


namespace digit_7_occurrences_l773_773082

theorem digit_7_occurrences : 
  let count_units := 80,
      count_tens := 8 * 10,
      count_hundreds := 100
  in count_units + count_tens + count_hundreds = 260 :=
by
  sorry

end digit_7_occurrences_l773_773082


namespace areaOfPolarCurve_l773_773718

noncomputable def polarArea (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ φ in a..b, (f φ)^2

def polarEq (φ : ℝ) : ℝ := 1 + Real.sqrt 2 * Real.sin φ

theorem areaOfPolarCurve :
  polarArea polarEq (-Real.pi / 2) (Real.pi / 2) = 2 * Real.pi :=
by
  sorry

end areaOfPolarCurve_l773_773718


namespace gp_condition_necessity_l773_773797

theorem gp_condition_necessity {a b c : ℝ} 
    (h_gp: ∃ r: ℝ, b = a * r ∧ c = a * r^2 ) : b^2 = a * c :=
by
  sorry

end gp_condition_necessity_l773_773797


namespace range_of_m_real_roots_l773_773109

theorem range_of_m_real_roots (m : ℝ) : 
  (∀ x : ℝ, ∃ k l : ℝ, k = 2*x ∧ l = m - x^2 ∧ k^2 - 4*l ≥ 0) ↔ m ≤ 1 := 
sorry

end range_of_m_real_roots_l773_773109


namespace door_height_is_six_l773_773959

def height_of_door
  (length width room_height : ℝ)
  (cost_per_sq_foot total_cost : ℝ)
  (door_width window_height window_width : ℝ)
  (num_windows : ℝ) : ℝ :=
  let wall_area := 2 * (length * room_height) + 2 * (width * room_height)
  let window_area := num_windows * (window_height * window_width)
  let area_to_paint := total_cost / cost_per_sq_foot
  let door_height := (wall_area - window_area - area_to_paint) / door_width
  door_height

theorem door_height_is_six
  (length width room_height h door_width window_height window_width : ℝ)
  (cost_per_sq_foot total_cost : ℝ)
  (num_windows : ℝ)
  (h_eq : h = height_of_door length width room_height cost_per_sq_foot total_cost door_width window_height window_width num_windows) :
  h = 6 := by
  sorry

end door_height_is_six_l773_773959


namespace minimum_zeros_l773_773125

def min_zeros_in_grid (a : ℕ → ℕ → ℤ) : ℕ :=
  if h1 : (∀ i : fin 15, ∑ j, a i j ≤ 0) ∧ (∀ j : fin 15, 0 ≤ ∑ i, a i j) 
  then 15 
  else 0

theorem minimum_zeros (a : ℕ → ℕ → ℤ) 
  (h1 : ∀ (i : fin 15), ∑ j, a i j ≤ 0) 
  (h2 : ∀ (j : fin 15), 0 ≤ ∑ i, a i j) : 
  min_zeros_in_grid a = 15 :=
begin
  sorry
end

end minimum_zeros_l773_773125


namespace part1_part2_l773_773034

-- Define the sets A and B based on the conditions
def A (a : ℝ) : Set ℝ := {x : ℝ | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5}
def B : Set ℝ := {x : ℝ | x < -1 ∨ x > 16}

-- Prove that if A is non-empty, then a ≥ 6
theorem part1 (a : ℝ) (h : A a ≠ ∅) : a ≥ 6 := 
sorry

-- Prove that if A ⊆ B, then a ∈ (-∞, 6) ∪ (15/2, ∞)
theorem part2 (a : ℝ) (h : A a ⊆ B) : a ∈ Set.Ioo (-∞) 6 ∪ Set.Ioo (15 / 2) ∞ := 
sorry

end part1_part2_l773_773034


namespace correct_statement_is_b_l773_773296

theorem correct_statement_is_b : 
  (∀ a b : ℚ, (b ≠ 0 ∧ a = ⅔ b) → a * b = 1) ∧
  (∀ x y : ℚ, (y = -x) → x = -y) ∧
  (∀ z : ℤ, (-|z| = -z) → z < 0) → 
  opposite (-1/3) = 1/3 :=
by sorry

end correct_statement_is_b_l773_773296


namespace prairie_total_area_l773_773686

theorem prairie_total_area (acres_dust_storm : ℕ) (acres_untouched : ℕ) (h₁ : acres_dust_storm = 64535) (h₂ : acres_untouched = 522) : acres_dust_storm + acres_untouched = 65057 :=
by
  sorry

end prairie_total_area_l773_773686


namespace trapezoid_midpoint_dist_l773_773174

open Real EuclideanGeometry

noncomputable def midpoint (p₁ p₂ : Point ℝ) : Point ℝ :=
  ⟨(p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2⟩

variable 
  (A B C D I J : Point ℝ)
  (AB_CD_parallel : parallel (lineThrough AB A B) (lineThrough CD C D))
  (angle_sum : angle D A B + angle A B C = π / 2)
  (I_midpoint : I = midpoint A B)
  (J_midpoint : J = midpoint C D)

theorem trapezoid_midpoint_dist (trapezoid_ABCD : Trapezoid A B C D) :
  2 * distance I J = abs (distance A B - distance C D) :=
sorry

end trapezoid_midpoint_dist_l773_773174


namespace triangle_area_l773_773666

def base : ℝ := 8
def height : ℝ := 4

theorem triangle_area : (base * height) / 2 = 16 := by
  sorry

end triangle_area_l773_773666


namespace cot_sum_arccot_roots_l773_773165

noncomputable def roots : Fin 15 → ℂ := by
  sorry

theorem cot_sum_arccot_roots :
  (∀ (k : Fin 15), roots k ^ 15 - 3 * roots k ^ 14 + 6 * roots k ^ 13 - 10 * roots k ^ 12 + 
  -- add rest of the polynomial terms here
  225 = 0) → 
  Real.cot (∑ k in Finset.range 15, Real.arccot (roots ⟨k, sorry⟩)) = 10 / 9 :=
by sorry

end cot_sum_arccot_roots_l773_773165


namespace calories_consumed_l773_773897

-- Define the conditions
def pages_written : ℕ := 12
def pages_per_donut : ℕ := 2
def calories_per_donut : ℕ := 150

-- Define the theorem to be proved
theorem calories_consumed (pages_written : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  (pages_written / pages_per_donut) * calories_per_donut

-- Ensure the theorem corresponds to the correct answer
example : calories_consumed 12 2 150 = 900 := by
  sorry

end calories_consumed_l773_773897


namespace find_second_term_l773_773424

-- Define the elements of the sequence and the conditions
def sequence (n : ℕ) : ℤ :=
  match n with
  | 0 => 8  -- first term
  | 1 => x  -- second term
  | 2 => 2  -- third term
  | 3 => -4 -- fourth term
  | 4 => -12 -- fifth term
  | _ => 0  -- out of defined range for simplicity

-- Define the pattern of the sequence
def pattern (a b : ℤ) (n : ℕ) : Prop :=
  (b - a = -2 * (n + 1))

-- State that the sequence follows the pattern
axiom seq_pattern (n : ℕ) : n < 4 → pattern (sequence n) (sequence (n + 1)) n

-- Prove the second term is 4 given the conditions
theorem find_second_term : sequence 1 = 4 :=
by {
  -- Use the given patterns and conditions to show that the second term is 4
  have h1 : pattern (sequence 0) (sequence 1) 0 := seq_pattern 0 (by linarith),
  cases h1,
  -- Use the pattern condition to solve for x
  linarith,
}

end find_second_term_l773_773424


namespace geometric_series_sum_condition_l773_773869

def geometric_series_sum (a q n : ℕ) : ℕ := a * (1 - q^n) / (1 - q)

theorem geometric_series_sum_condition (S : ℕ → ℕ) (a : ℕ) (q : ℕ) (h1 : a = 1) 
  (h2 : ∀ n, S n = geometric_series_sum a q n)
  (h3 : S 7 - 4 * S 6 + 3 * S 5 = 0) : 
  S 4 = 40 := 
by 
  sorry

end geometric_series_sum_condition_l773_773869


namespace find_salary_month_l773_773598

variable (J F M A May : ℝ)

def condition_1 : Prop := (J + F + M + A) / 4 = 8000
def condition_2 : Prop := (F + M + A + May) / 4 = 8450
def condition_3 : Prop := J = 4700
def condition_4 (X : ℝ) : Prop := X = 6500

theorem find_salary_month (J F M A May : ℝ) 
  (h1 : condition_1 J F M A) 
  (h2 : condition_2 F M A May) 
  (h3 : condition_3 J) 
  : ∃ M : ℝ, condition_4 May :=
by sorry

end find_salary_month_l773_773598


namespace part1_part2_l773_773814

-- Defining the given function f(x) with conditions
noncomputable def f (x : ℝ) : ℝ :=
  let A := 1
  let B := 1
  A * Real.sin (3 * x + Real.pi / 6) + B

-- The maximum and minimum values of f(x)
def f_max := 2
def f_min := 0

-- Prove that f(7π/18) = 1 - √3/2
theorem part1 :
  f (7 * Real.pi / 18) = 1 - Real.sqrt(3) / 2 :=
sorry

-- Define the function g(x) as described in the problem
noncomputable def g (x : ℝ) : ℝ :=
  Real.sqrt 2 * Real.sin (3 * x - Real.pi / 3) + Real.sqrt 2

-- Prove the solutions to the equation g(x) = √2/2
theorem part2 (k : ℤ) :
  g x = Real.sqrt(2) / 2 ↔ ∃ k : ℤ, x = 2 / 3 * k * Real.pi + Real.pi / 2 ∨ x = 2 / 3 * k * Real.pi + 13 * Real.pi / 18 :=
sorry

end part1_part2_l773_773814


namespace min_profit_stationery_l773_773327

noncomputable def profit (x : ℝ) : ℝ :=
  let y := -2 * x + 60
  in y * (x - 10)

theorem min_profit_stationery :
  ∃ p : ℝ, p = 128 ∧ (∀ x : ℝ, 15 ≤ x ∧ x ≤ 26 → profit x ≥ p) :=
begin
  sorry
end

end min_profit_stationery_l773_773327


namespace sum_of_seven_consecutive_integers_l773_773592

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
  sorry

end sum_of_seven_consecutive_integers_l773_773592


namespace painted_cube_faces_same_color_probability_eq_one_l773_773394

noncomputable def cube_face_color_probability : ℚ := 2 / 3

noncomputable def painted_cube_problem : ℚ :=
  let p_red := 2 / 3 in
  let p_blue := 1 / 3 in
  let p_three_faces_same_color := 1 in
  p_three_faces_same_color

theorem painted_cube_faces_same_color_probability_eq_one :
  painted_cube_problem = 1 :=
  sorry

end painted_cube_faces_same_color_probability_eq_one_l773_773394


namespace problem_a_problem_b_problem_c_problem_d_problem_e_l773_773204

section problem_a
  -- Conditions
  def rainbow_russian_first_letters_sequence := ["к", "о", "ж", "з", "г", "с", "ф"]
  
  -- Theorem (question == answer)
  theorem problem_a : rainbow_russian_first_letters_sequence[4] = "г" ∧
                      rainbow_russian_first_letters_sequence[5] = "с" ∧
                      rainbow_russian_first_letters_sequence[6] = "ф" :=
  by
    -- Skip proof: sorry
    sorry
end problem_a

section problem_b
  -- Conditions
  def russian_alphabet_alternating_sequence := ["а", "в", "г", "ё", "ж", "з", "л", "м", "н", "о", "п", "т", "у"]
 
  -- Theorem (question == answer)
  theorem problem_b : russian_alphabet_alternating_sequence[10] = "п" ∧
                      russian_alphabet_alternating_sequence[11] = "т" ∧
                      russian_alphabet_alternating_sequence[12] = "у" :=
  by
    -- Skip proof: sorry
    sorry
end problem_b

section problem_c
  -- Conditions
  def russian_number_of_letters_sequence := ["один", "четыре", "шесть", "пять", "семь", "восемь"]
  
  -- Theorem (question == answer)
  theorem problem_c : russian_number_of_letters_sequence[4] = "семь" ∧
                      russian_number_of_letters_sequence[5] = "восемь" :=
  by
    -- Skip proof: sorry
    sorry
end problem_c

section problem_d
  -- Conditions
  def approximate_symmetry_letters_sequence := ["Ф", "Х", "Ш", "В"]

  -- Theorem (question == answer)
  theorem problem_d : approximate_symmetry_letters_sequence[3] = "В" :=
  by
    -- Skip proof: sorry
    sorry
end problem_d

section problem_e
  -- Conditions
  def russian_loops_in_digit_sequence := ["0", "д", "т", "ч", "п", "ш", "с", "в", "д"]

  -- Theorem (question == answer)
  theorem problem_e : russian_loops_in_digit_sequence[7] = "в" ∧
                      russian_loops_in_digit_sequence[8] = "д" :=
  by
    -- Skip proof: sorry
    sorry
end problem_e

end problem_a_problem_b_problem_c_problem_d_problem_e_l773_773204


namespace vector_properties_l773_773114

-- Definitions of vectors
def vec_a : ℝ × ℝ := (3, 11)
def vec_b : ℝ × ℝ := (-1, -4)
def vec_c : ℝ × ℝ := (1, 3)

-- Dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Linear combination of vector scaling and addition
def vec_sub_scal (u v : ℝ × ℝ) (k : ℝ) : ℝ × ℝ := (u.1 - k * v.1, u.2 - k * v.2)

-- Check if two vectors are parallel
def parallel (u v : ℝ × ℝ) : Prop := u.1 / v.1 = u.2 / v.2

-- Lean statement for the proof problem
theorem vector_properties :
  dot_product vec_a vec_b = -47 ∧
  vec_sub_scal vec_a vec_b 2 = (5, 19) ∧
  dot_product (vec_b.1 + vec_c.1, vec_b.2 + vec_c.2) vec_c ≠ 0 ∧
  parallel (vec_sub_scal vec_a vec_c 1) vec_b :=
by sorry

end vector_properties_l773_773114


namespace ratio_of_areas_l773_773333

def area_equilateral_triangle (s : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * s^2

theorem ratio_of_areas :
  let A_large := area_equilateral_triangle 12
  let A_small := area_equilateral_triangle 6
  let A_remaining := A_large - A_small
  (A_small / A_remaining) = 1 / 3 := by
  sorry

end ratio_of_areas_l773_773333


namespace p_sufficient_not_necessary_q_l773_773022

variable {x : ℝ}

def p : Prop := 2 < x ∧ x < 4
def q : Prop := x < -3 ∨ x > 2

theorem p_sufficient_not_necessary_q : (p → q) ∧ ¬(q → p) :=
by sorry

end p_sufficient_not_necessary_q_l773_773022


namespace zeros_of_g_l773_773818

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else Real.log x / Real.log 2

noncomputable def g (x : ℝ) : ℝ :=
  f (f x) + 1

theorem zeros_of_g : setOf (λ x, g x = 0).card = 4 := by
  sorry

end zeros_of_g_l773_773818


namespace number_of_squares_in_region_l773_773496

theorem number_of_squares_in_region :
  let region := {p : ℝ × ℝ | p.2 = 2 * p.1 ∨ p.2 = 1 ∨ p.1 = 6}
  ∃ (squares : finset (set (ℝ × ℝ))),
    ∀ square ∈ squares,
      (∀ vertex ∈ square, vertex.2 = floor vertex.2) ∧
      (∀ vertex ∈ square, vertex.1 = ceil vertex.1) ∧
      set.subset square region ∧
      (finset.card squares = 5) := by
  sorry

end number_of_squares_in_region_l773_773496


namespace min_abs_diff_value_l773_773091

noncomputable def min_abs_diff (x y : ℝ) : ℝ :=
  |x| - |y|

theorem min_abs_diff_value (x y : ℝ) 
  (h : log 4 (x + 2 * y) + log 4 (x - 2 * y) = 1) : 
  ∃ x y : ℝ, min_abs_diff x y = sqrt 3 :=
by
  sorry

end min_abs_diff_value_l773_773091


namespace total_employees_l773_773525

variable (E : ℕ) -- E is the total number of employees

-- Conditions given in the problem
variable (male_fraction : ℚ := 0.45) -- 45% of the total employees are males
variable (males_below_50 : ℕ := 1170) -- 1170 males are below 50 years old
variable (males_total : ℕ := 2340) -- Total number of male employees

-- Condition derived from the problem (calculation of total males)
lemma male_employees_equiv (h : males_total = 2 * males_below_50) : males_total = 2340 :=
  by sorry

-- Main theorem
theorem total_employees (h : male_fraction * E = males_total) : E = 5200 :=
  by sorry

end total_employees_l773_773525


namespace towers_remainder_mod_100_l773_773328

/-- Define edge lengths and the number of cubes. --/
def edge_lengths : List ℕ := [1, 2, 3, 4, 5, 6, 7]
def num_cubes : ℕ := 7

/-- Define the number of different towers that can be constructed. --/
def num_towers (n : ℕ) : ℕ :=
  if h : n = num_cubes then
    64 -- Based on the solution to the problem
  else
    0 -- Placeholder for other values not directly used

/-- Proving the remainder when num_towers is divided by 100 --/
theorem towers_remainder_mod_100 : num_towers num_cubes % 100 = 64 :=
by
  unfold num_towers
  rw if_pos rfl
  exact rfl

end towers_remainder_mod_100_l773_773328


namespace figure_50_unit_squares_l773_773565

-- Definitions reflecting the conditions from step A
def f (n : ℕ) := (1/2 : ℚ) * n^3 + (7/2 : ℚ) * n + 1

theorem figure_50_unit_squares : f 50 = 62676 := by
  sorry

end figure_50_unit_squares_l773_773565


namespace reciprocal_of_2023_l773_773988

theorem reciprocal_of_2023 : (2023 : ℝ)⁻¹ = 1 / 2023 :=
by
  sorry

end reciprocal_of_2023_l773_773988


namespace domain_of_logarithmic_function_l773_773230

theorem domain_of_logarithmic_function :
  ∀ x : ℝ, (3^x - 2^x > 0 ↔ x > 0) :=
by
sorry

end domain_of_logarithmic_function_l773_773230


namespace jonah_raisins_l773_773144

variable (y : ℝ)

theorem jonah_raisins :
  (y + 0.4 = 0.7) → (y = 0.3) :=
  by
  intro h
  sorry

end jonah_raisins_l773_773144


namespace log_base_a_of_square_l773_773089

theorem log_base_a_of_square (a : ℝ) (N : ℝ) (h1 : N = a^2) (h2 : a > 0) (h3 : a ≠ 1) : log a N = 2 :=
sorry

end log_base_a_of_square_l773_773089


namespace probability_of_prime_or_odd_is_half_l773_773276

-- Define the list of sections on the spinner
def sections : List ℕ := [3, 6, 1, 4, 8, 10, 2, 7]

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Bool :=
  if n < 2 then false else List.foldr (λ p b => b && (n % p ≠ 0)) true (List.range (n - 2) |>.map (λ x => x + 2))

-- Define a function to check if a number is odd
def is_odd (n : ℕ) : Bool := n % 2 ≠ 0

-- Define the condition of being either prime or odd
def is_prime_or_odd (n : ℕ) : Bool := is_prime n || is_odd n

-- List of favorable outcomes where the number is either prime or odd
def favorable_outcomes : List ℕ := sections.filter is_prime_or_odd

-- Calculate the probability
def probability_prime_or_odd : ℚ := (favorable_outcomes.length : ℚ) / (sections.length : ℚ)

-- Statement to prove the probability is 1/2
theorem probability_of_prime_or_odd_is_half : probability_prime_or_odd = 1 / 2 := by
  sorry

end probability_of_prime_or_odd_is_half_l773_773276


namespace inequality_proof_l773_773555

theorem inequality_proof (n : ℕ) (h : n ≥ 2) : 
  n * ((1 + n : ℝ) ^ (1 / n : ℝ) - 1) < (∑ i in Finset.range (n+1), 1 / (i+1)) ∧
  (∑ i in Finset.range (n+1), 1 / (i+1)) < n - (n - 1) * (1 / n : ℝ) ^ (-1 / ((n - 1) : ℝ)) :=
sorry

end inequality_proof_l773_773555


namespace triangle_area_from_curve_l773_773494

-- Definition of the curve
def curve (x : ℝ) : ℝ := (x - 2)^2 * (x + 3)

-- Area calculation based on intercepts
theorem triangle_area_from_curve : 
  (1 / 2) * (2 - (-3)) * (curve 0) = 30 :=
by
  sorry

end triangle_area_from_curve_l773_773494


namespace problem_which_function_has_same_shape_as_5x2_l773_773657

def has_same_shape (f g : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = a * g x

theorem problem_which_function_has_same_shape_as_5x2 :
  has_same_shape (λ x, -5 * x^2 + 2) (λ x, 5 * x^2) :=
by
  use -1
  intro x
  simp
  sorry

end problem_which_function_has_same_shape_as_5x2_l773_773657


namespace factorize_b5_minus_b_neg5_l773_773048

theorem factorize_b5_minus_b_neg5 (b : ℝ) (hb : b ≠ 0) : 
  b^5 - b^(-5) = (b - b^(-1)) * (b^4 + b^2 + 1 + b^(-2) + b^(-4)) := 
by
  sorry

end factorize_b5_minus_b_neg5_l773_773048


namespace sandy_salary_increase_l773_773545

-- Define the conditions for the problem
variables (S : ℝ) (P : ℝ)

-- Conditions
def last_year_saved := 0.1 * S
def this_year_salary := S * (1 + P / 100)
def this_year_saved := 0.06 * this_year_salary

-- Statement of the problem
theorem sandy_salary_increase :
  this_year_saved S P = 0.66 * last_year_saved S → P = 10 := by
  sorry

end sandy_salary_increase_l773_773545


namespace ratio_of_squares_l773_773912

theorem ratio_of_squares (A B C D E F G H : ℝ × ℝ) (s : ℝ) (h_side : s = 4)
  (h_square : 
    A = (0, 0) ∧ B = (4, 0) ∧ C = (4, 4) ∧ D = (0, 4) ∧
    E = (2, 2 * (real.sqrt 3) / 3) ∧ F = (4 - (2 * (real.sqrt 3) / 3), 2) ∧ 
    G = (2, 4 - (2 * (real.sqrt 3) / 3)) ∧ H = (2 * (real.sqrt 3) / 3, 2)) :
  (area (square E F G H)) / (area (square A B C D)) = 2 / 3 :=
  sorry

end ratio_of_squares_l773_773912


namespace fraction_covered_by_small_circles_l773_773645

-- Definition of the problem conditions
def side_length : ℝ := 2
def radius_small_circle : ℝ := side_length / 2
def circumradius_pentagon : ℝ := side_length / (2 * Real.cos (Real.pi / 5))
def radius_large_circle : ℝ := circumradius_pentagon + radius_small_circle
def area_small_circle : ℝ := Real.pi * radius_small_circle^2
def area_five_small_circles : ℝ := 5 * area_small_circle
def area_large_circle : ℝ := Real.pi * radius_large_circle^2
def fraction_area_covered : ℝ := area_five_small_circles / area_large_circle

-- The statement to be proved
theorem fraction_covered_by_small_circles : fraction_area_covered = 0.8 :=
by
  sorry

end fraction_covered_by_small_circles_l773_773645


namespace hyperbola_representation_l773_773655

variable (x y : ℝ)

/--
Given the equation (x - y)^2 = 3(x^2 - y^2), we prove that
the resulting graph represents a hyperbola.
-/
theorem hyperbola_representation :
  (x - y)^2 = 3 * (x^2 - y^2) →
  ∃ A B C : ℝ, A ≠ 0 ∧ (x^2 + x * y - 2 * y^2 = 0) ∧ (A = 1) ∧ (B = 1) ∧ (C = -2) ∧ (B^2 - 4*A*C > 0) :=
by
  sorry

end hyperbola_representation_l773_773655


namespace company_picnic_attendance_l773_773863

/-
Given:
- IT Department: 40% men, 25% women
- HR Department: 30% men, 20% women
- Marketing Department: 30% men, 55% women
- Attendance rates:
  - IT Department: 25% men, 60% women
  - HR Department: 30% men, 50% women
  - Marketing Department: 10% men, 45% women

Prove that the total percentage of employees who attended the annual company picnic is 71.75%.
-/

theorem company_picnic_attendance :
  let men_IT := 0.40
  let women_IT := 0.25
  let men_HR := 0.30
  let women_HR := 0.20
  let men_Marketing := 0.30
  let women_Marketing := 0.55
  let att_IT_men := 0.25 * 0.40
  let att_IT_women := 0.60 * 0.25
  let att_HR_men := 0.30 * 0.30
  let att_HR_women := 0.50 * 0.20
  let att_Marketing_men := 0.10 * 0.30
  let att_Marketing_women := 0.45 * 0.55
  let total_att_IT := att_IT_men + att_IT_women
  let total_att_HR := att_HR_men + att_HR_women
  let total_att_Marketing := att_Marketing_men + att_Marketing_women
  let total_att := total_att_IT + total_att_HR + total_att_Marketing
  total_att = 0.7175 :=
begin
  sorry
end

end company_picnic_attendance_l773_773863


namespace calories_consumed_Jean_l773_773902

def donuts_per_page (pages : ℕ) : ℕ := pages / 2

def calories_per_donut : ℕ := 150

def total_calories (pages : ℕ) : ℕ :=
  let donuts := donuts_per_page pages
  donuts * calories_per_donut

theorem calories_consumed_Jean (h1 : ∀ pages, donuts_per_page pages = pages / 2)
  (h2 : calories_per_donut = 150)
  (h3 : total_calories 12 = 900) :
  total_calories 12 = 900 := by
  sorry

end calories_consumed_Jean_l773_773902


namespace remainder_when_200_divided_by_k_l773_773011

theorem remainder_when_200_divided_by_k (k : ℕ) (hk_pos : 0 < k)
  (h₁ : 125 % (k^3) = 5) : 200 % k = 0 :=
sorry

end remainder_when_200_divided_by_k_l773_773011


namespace part1_l773_773826

theorem part1 (k : ℝ) : (∀ x : ℝ, x ≠ 0 → (k-4) / x > 0 ↔ k > 4) :=
begin
  sorry
end

end part1_l773_773826


namespace intersection_M_N_l773_773562

def M : Set ℝ := {x | x < 2016}

def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l773_773562


namespace find_angle_A_l773_773115

theorem find_angle_A (a b : ℝ) (B A : ℝ)
  (h1 : a = 2) 
  (h2 : b = Real.sqrt 3) 
  (h3 : B = Real.pi / 3) : 
  A = Real.pi / 2 := 
sorry

end find_angle_A_l773_773115


namespace ratio_size12_to_size6_l773_773600

-- Definitions based on conditions
def cheerleaders_size2 : ℕ := 4
def cheerleaders_size6 : ℕ := 10
def total_cheerleaders : ℕ := 19
def cheerleaders_size12 : ℕ := total_cheerleaders - (cheerleaders_size2 + cheerleaders_size6)

-- Proof statement
theorem ratio_size12_to_size6 : cheerleaders_size12.toFloat / cheerleaders_size6.toFloat = 1 / 2 := sorry

end ratio_size12_to_size6_l773_773600


namespace total_blood_cells_correct_l773_773354

-- Define the number of blood cells in the first and second samples.
def sample_1_blood_cells : ℕ := 4221
def sample_2_blood_cells : ℕ := 3120

-- Define the total number of blood cells.
def total_blood_cells : ℕ := sample_1_blood_cells + sample_2_blood_cells

-- Theorem stating the total number of blood cells based on the conditions.
theorem total_blood_cells_correct : total_blood_cells = 7341 :=
by
  -- Proof is omitted
  sorry

end total_blood_cells_correct_l773_773354


namespace smallest_integer_k_no_real_roots_l773_773758

def quadratic_no_real_roots (a b c : ℝ) : Prop := b^2 - 4 * a * c < 0

theorem smallest_integer_k_no_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, quadratic_no_real_roots (2 * k - 1) (-8) 6) ∧ (k = 2) :=
by
  sorry

end smallest_integer_k_no_real_roots_l773_773758


namespace susan_remaining_spaces_to_win_l773_773216

/-- Susan's board game has 48 spaces. She makes three moves:
 1. She moves forward 8 spaces
 2. She moves forward 2 spaces and then back 5 spaces
 3. She moves forward 6 spaces
 Prove that the remaining spaces she has to move to reach the end is 37.
-/
theorem susan_remaining_spaces_to_win :
  let total_spaces := 48
  let first_turn := 8
  let second_turn := 2 - 5
  let third_turn := 6
  let total_moved := first_turn + second_turn + third_turn
  total_spaces - total_moved = 37 :=
by
  sorry

end susan_remaining_spaces_to_win_l773_773216


namespace solve_quadratic_equation_l773_773588

theorem solve_quadratic_equation (x : ℝ) : 
  2 * x^2 - 4 * x = 6 - 3 * x ↔ (x = -3/2 ∨ x = 2) :=
by
  sorry

end solve_quadratic_equation_l773_773588


namespace sum_of_squares_l773_773745

theorem sum_of_squares :
  ∀ (ABC AD1E1 AD1E2 AD2E3 AD2E4 : triangle) (a b c d e f : ℝ),
  (ABC.side_length = √121) →
  (AD1E1.congruent ABC ∧ AD1E2.congruent ABC ∧ AD2E3.congruent ABC ∧ AD2E4.congruent ABC) →
  (BD1.length = BD2.length = √25) →
   (∑ k in [1..4], (CEk)^2 = 968) :=
by
sorry

end sum_of_squares_l773_773745


namespace right_triangle_segment_equality_l773_773871

-- Definitions of the points and conditions
variables {A B C D F E L : Type} [MetricSpace A]
variable {ℝ : Type} [Real ℝ]
class right_triangle (triangle : Type) :=
  (right_angle : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (angle_C_90 : ∠ C = 90º)

-- Exists height and equilateral triangles
variable (H1 : ∃ D, is_height C D A B ∧ lies_in_half_plane C F A B ∧ lies_in_half_plane C E A B)
variable (H2 : ∃ F, is_equilateral_triangle A D F)
variable (H3 : ∃ E, is_equilateral_triangle D B E)
variable (H4 : ∃ L, line_intersect_segment D E A C L)

-- The goal
theorem right_triangle_segment_equality :
  ∀ {A B C D F E L : Type} [MetricSpace A] [Real ℝ],
  right_triangle (triangle A B C) →
  (is_height C D A B ∧ lies_in_half_plane C F A B ∧ lies_in_half_plane C E A B) →
  (is_equilateral_triangle A D F) →
  (is_equilateral_triangle D B E) →
  (line_intersect_segment D E A C L) →
  distance F L = distance C L + distance L D :=
by sorry

end right_triangle_segment_equality_l773_773871


namespace min_swipes_to_clear_checkerboard_l773_773637

theorem min_swipes_to_clear_checkerboard (n : ℕ) : 
  ∀ (chips : ℕ), chips = (n + 1)^2 → 
  (∀ (swipe1 swipe2 : ℕ), 
    (swipe1 + swipe2 = (n^2 + n)) ∧ 
    (swipe1: ℕ → nat.succ const 1) ∧ 
    swipe2: ℕ → 
    ((adjacent_or_diagonal))) :=
sorry

end min_swipes_to_clear_checkerboard_l773_773637


namespace inscribed_equal_angles_regular_polygon_circumscribed_equal_sides_regular_polygon_l773_773690

-- Definitions to set up the conditions
variables {n : ℕ} (hn : Odd n) (angles_equal : ∀ i j, ∡ A_i = ∡ A_j) 
          (sides_equal : ∀ i j, side A_i A_{i+1} = side A_j A_{j+1})

-- Definitions for inscribed and circumscribed polygons
variables (inscribed : ∃ (O : point), ∀ k, distance O A_k = radius) 
          (circumscribed : ∃ (I : point), is_tangent I A_k A_{k+1})

-- Part (a) statement: 
theorem inscribed_equal_angles_regular_polygon :  
  inscribed → angles_equal → RegularPolygon A_1 A_n :=
by
  intro h_inscribed h_angles_equal
  sorry

-- Part (b) statement: 
theorem circumscribed_equal_sides_regular_polygon :  
  circumscribed → sides_equal → RegularPolygon A_1 A_n :=
by
  intro h_circumscribed h_sides_equal
  sorry

end inscribed_equal_angles_regular_polygon_circumscribed_equal_sides_regular_polygon_l773_773690


namespace baseball_bats_against_lefties_l773_773318

-- Define the conditions
variables (L R : ℕ)
variable h1 : L + R = 600
variable h2 : 0.25 * L + 0.35 * R = 192

-- State the theorem
theorem baseball_bats_against_lefties : L = 180 :=
by
  sorry

end baseball_bats_against_lefties_l773_773318


namespace no_solution_abs_eq_l773_773585

theorem no_solution_abs_eq (x : ℝ) (h : x > 0) : |x + 4| = 3 - x → false :=
by
  sorry

end no_solution_abs_eq_l773_773585


namespace next_performance_together_in_90_days_l773_773396

theorem next_performance_together_in_90_days :
  Nat.lcm (Nat.lcm 5 6) (Nat.lcm 9 10) = 90 := by
  sorry

end next_performance_together_in_90_days_l773_773396


namespace tan_double_angle_l773_773046

theorem tan_double_angle (θ : ℝ) (h1 : θ = Real.arctan (-2)) : Real.tan (2 * θ) = 4 / 3 :=
by
  sorry

end tan_double_angle_l773_773046


namespace beavers_working_on_home_l773_773313

noncomputable def initial_beavers : ℝ := 2.0
noncomputable def additional_beavers : ℝ := 1.0

theorem beavers_working_on_home : initial_beavers + additional_beavers = 3.0 :=
by
  sorry

end beavers_working_on_home_l773_773313


namespace perimeter_bound_l773_773340

variables (R S : ℝ)
variable (n : ℕ)
variable (A : fin n → ℝ × ℝ)
variable (B : fin n → ℝ × ℝ)
variable (O : ℝ × ℝ)

-- Conditions: A polygon of area S inscribed in a circle of radius R and containing the center O
-- Perimeter formula of the convex hull of points Bi is ≥ 2S / R

def area_of_polygon (A : fin n → ℝ × ℝ) : ℝ := sorry
def perimeter (B : fin n → ℝ × ℝ) : ℝ := sorry

theorem perimeter_bound (h1 : area_of_polygon A = S) (h2 : ∀ i, dist O (A i) = R):
  perimeter B ≥ (2 * S) / R := sorry

end perimeter_bound_l773_773340


namespace parabola_intersection_count_l773_773427

theorem parabola_intersection_count :
  ∃ (count : ℕ), count = 37 ∧ 
                  ∀ k : ℤ, (-4 ≤ k ∧ k ≤ 32) ↔
                            ∃ x : ℝ, (- (1/8) * x^2 + 4 = x^2 - k) ∧
                                      (- (1/8) * x^2 + 4 ≥ 0) ∧ (x^2 - k ≥ 0) :=
by 
  use 37
  split
  sorry
  intros k
  split <;> intros
  sorry

end parabola_intersection_count_l773_773427


namespace correct_sqrt_expression_l773_773292

theorem correct_sqrt_expression : 
  (∃ (a b : ℝ), a = sqrt (15 / 2) ∧ b = (1 / 2) * sqrt 30 ∧ a = b) :=
by
  sorry

end correct_sqrt_expression_l773_773292


namespace determine_x_l773_773737

theorem determine_x (x : ℝ) : (∀ y : ℝ, 10 * x * y - 15 * y + 2 * x - 3 = 0) → x = 3 / 2 :=
by
  intro h
  have : ∀ y : ℝ, (5 * y + 1) * (2 * x - 3) = 0 := 
    sorry
  have : (2 * x - 3) = 0 := 
    sorry
  show x = 3 / 2
  sorry

end determine_x_l773_773737


namespace proof_problem_l773_773844

noncomputable def a : ℝ := 0.85 * 250
noncomputable def b : ℝ := 0.75 * 180
noncomputable def c : ℝ := 0.90 * 320

theorem proof_problem :
  (a - b = 77.5) ∧ (77.5 < c) :=
by
  sorry

end proof_problem_l773_773844


namespace banish_all_bad_wizards_at_most_one_good_l773_773235

theorem banish_all_bad_wizards_at_most_one_good :
  ∀ (wizards : List Wizard)
  (good_wizard : Wizard → Prop)
  (bad_wizard : Wizard → Prop)
  (answers : Wizard → Wizard → Bool),
  (∀ w, good_wizard w ↔ ¬ bad_wizard w) →
  (∀ w, good_wizard w → ∀ v, answers w v = true ↔ good_wizard v) →
  (List.length wizards = 2015) →
  ∃ banishment_strategy : (List Wizard → Wizard) → Prop,
  (∀ wizards_updated, (List.length wizards_updated < 2015) →
  List.All bad_wizard wizards_updated ∨
  (List.ExactlyOne good_wizard wizards →
   List.All bad_wizard (List.RemoveOnce wizards_updated (banishment_strategy wizards_updated)))) :=
by
  sorry

end banish_all_bad_wizards_at_most_one_good_l773_773235


namespace find_a_l773_773475
open Real

noncomputable def f (a x : ℝ) := x * sin x + a * x

theorem find_a (a : ℝ) : (deriv (f a) (π / 2) = 1) → a = 0 := by
  sorry

end find_a_l773_773475


namespace distinct_prime_factors_of_A_l773_773149

noncomputable def A : ℕ := ∏ (d : ℕ) in (finset.filter (λ x, 60 % x = 0) (finset.range 61)), d

theorem distinct_prime_factors_of_A : (finset.filter (nat.prime) (nat.factorization A).support).card = 3 :=
sorry

end distinct_prime_factors_of_A_l773_773149


namespace labor_union_trees_l773_773678

theorem labor_union_trees (x : ℕ) :
  (∃ t : ℕ, t = 2 * x + 21) ∧ (∃ t' : ℕ, t' = 3 * x - 24) →
  2 * x + 21 = 3 * x - 24 :=
by
  sorry

end labor_union_trees_l773_773678


namespace reciprocal_2023_l773_773984

def reciprocal (x : ℕ) := 1 / x

theorem reciprocal_2023 : reciprocal 2023 = 1 / 2023 :=
by
  sorry

end reciprocal_2023_l773_773984


namespace function_symmetric_l773_773074

-- Given the function f
def f (x : ℝ) : ℝ := if 0 < x then x^2 - 2*x else x^2 + 2*x

-- Prove that for x < 0, f(x) = x^2 + 2x
theorem function_symmetric {x : ℝ} (h₁ : f (x - 1) = f (2 - x)) (h₂ : 0 < x ∧ f x = x^2 - 2*x) :
  x < 0 → f x = x^2 + 2*x :=
by
  intro hx
  sorry

end function_symmetric_l773_773074


namespace domain_of_f_l773_773004

def f (x : ℝ) : ℝ := Real.tan (Real.arcsin (x^3))

theorem domain_of_f :
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 ↔ Real.arcsin (x^3) ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2) :=
by
  sorry

end domain_of_f_l773_773004


namespace monotone_increasing_interval_l773_773618

open Real

noncomputable def f (x : ℝ) : ℝ := ln (x^2 - 2 * x - 8)

theorem monotone_increasing_interval :
  ∀ x, x ∈ (Ioi 4) → Monotone (fun y => f (x + y)) := 
by
  sorry

end monotone_increasing_interval_l773_773618


namespace tess_distance_graph_l773_773593

open Real

theorem tess_distance_graph (X Y Z : Point) (triangle_equilateral : equilateral_triangle X Y Z)
    (distance_function : ℝ)
    (start : distance_function 0 = 0)
    (path_XY : ∀ t ∈ [0, 1/3], increasing (distance_function t))
    (path_YZ : ∀ t ∈ (1/3, 2/3], (decreasing (distance_function t) ∨ increasing (distance_function t)))
    (path_ZX : ∀ t ∈ (2/3, 1], decreasing (distance_function t))
    (periodicity : distance_function 1 = 0) :
  correct_graph distance_function = GraphF := 
begin
  sorry
end

end tess_distance_graph_l773_773593


namespace nat_not_exceeding_factorial_sum_l773_773939

open Nat

theorem nat_not_exceeding_factorial_sum (n : ℕ) (a : ℕ) 
  (h : a ≤ factorial n) : 
  ∃ S : Finset ℕ, (∑ x in S, x = a) ∧ (S.card ≤ n) ∧ (∀ x ∈ S, x ∣ factorial n) :=
begin
  sorry
end

end nat_not_exceeding_factorial_sum_l773_773939


namespace problem_solution_l773_773720

theorem problem_solution :
  3 * 995 + 4 * 996 + 5 * 997 + 6 * 998 + 7 * 999 - 4985 * 3 = 9980 := 
  by
  sorry

end problem_solution_l773_773720


namespace find_interest_rate_A_l773_773334

-- Definitions of the conditions
def principal := 2000
def interest_rate_C := 11.5 / 100
def time := 3
def B_gain := 90

-- Definition for the interest earned by B from C
def interest_from_C := principal * interest_rate_C * time

-- Definition for the interest paid by B to A
def interest_paid_to_A := interest_from_C - B_gain

-- Definition of the interest formula used to find R (the interest rate A lends to B)
def interest_rate_A (R : ℝ) := (principal * (R / 100) * time)

-- The main theorem: proving that the interest rate A lends to B is 10%
theorem find_interest_rate_A : ∃ (R : ℝ), interest_rate_A R = interest_paid_to_A ∧ R = 10 := by
  sorry

end find_interest_rate_A_l773_773334


namespace domain_of_f_l773_773233

noncomputable def domain_f : Set ℝ := {x : ℝ | 3^x - 2^x > 0}

theorem domain_of_f :
  domain_f = {x : ℝ | 0 < x} :=
sorry

end domain_of_f_l773_773233


namespace find_e_of_T_shaped_squares_division_l773_773583

theorem find_e_of_T_shaped_squares_division (e : ℝ) :
  (∃ e, 
     let area_total := 7 
     let area_half := area_total / 2 
     let slope := 3 / (4 - e)
     let line_eq := λ x, slope * (x - e)
     let triangle_base := 4 - e
     let triangle_height := 3 
     let triangle_area := 1/2 * triangle_base * triangle_height 
     triangle_area = area_half 
  ) ↔ e = 5 / 3 :=
begin
  sorry
end

end find_e_of_T_shaped_squares_division_l773_773583


namespace cheapest_lamp_cost_l773_773435

/--
Frank wants to buy a new lamp for his bedroom. The cost of the cheapest lamp is some amount, and the most expensive in the store is 3 times more expensive. Frank has $90, and if he buys the most expensive lamp available, he would have $30 remaining. Prove that the cost of the cheapest lamp is $20.
-/
theorem cheapest_lamp_cost (c most_expensive : ℝ) (h_cheapest_lamp : most_expensive = 3 * c) 
(h_frank_money : 90 - most_expensive = 30) : c = 20 := 
sorry

end cheapest_lamp_cost_l773_773435


namespace total_apartment_units_l773_773568

-- Define the number of apartment units on different floors
def units_first_floor := 2
def units_other_floors := 5
def num_other_floors := 3
def num_buildings := 2

-- Calculation of total units in one building
def units_one_building := units_first_floor + num_other_floors * units_other_floors

-- Calculation of total units in all buildings
def total_units := num_buildings * units_one_building

-- The theorem to prove
theorem total_apartment_units : total_units = 34 :=
by
  sorry

end total_apartment_units_l773_773568


namespace problem_l773_773768

theorem problem (a b : ℝ) (h : {a, b / a, 1} = {a^2, a + b, 0}) : a^2023 + b^2024 = -1 := by
  sorry

end problem_l773_773768


namespace square_difference_l773_773288

theorem square_difference : (601^2 - 599^2 = 2400) :=
by {
  -- Placeholder for the proof
  sorry
}

end square_difference_l773_773288


namespace solve_quadratic_eq_l773_773185

theorem solve_quadratic_eq : ∃ (a b : ℕ), a = 145 ∧ b = 7 ∧ a + b = 152 ∧ 
  ∀ x, x = Real.sqrt a - b → x^2 + 14 * x = 96 :=
by 
  use 145, 7
  simp
  sorry

end solve_quadratic_eq_l773_773185


namespace remainder_7_pow_150_mod_12_l773_773277

theorem remainder_7_pow_150_mod_12 :
  (7^150) % 12 = 1 := sorry

end remainder_7_pow_150_mod_12_l773_773277


namespace mean_median_mode_relation_l773_773078

theorem mean_median_mode_relation (xs : List ℕ) (h : xs = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4]) : 
  let mean := (xs.sum : ℚ) / xs.length
  let median := (xs.nthLe (xs.length / 2 - 1) sorry + xs.nthLe (xs.length / 2) sorry : ℚ) / 2
  let mode := [2, 3]
  mode.head < mean ∧ mean = median :=
by
  sorry

end mean_median_mode_relation_l773_773078


namespace equal_segments_in_square_l773_773639

/-
Given the following conditions:
1. A square named ABCD.
2. An arbitrary point X inside the square.
3. Two mutually perpendicular lines passing through X.
4. These lines intersect the opposite sides of the square at points P and Q along one line, and R and S along the other.

Prove that the segments PQ and RS confined within the square are equal.
-/
theorem equal_segments_in_square
    (A B C D X P Q R S : ℝ × ℝ)
    (h_square : (A.1, A.2) = (A.1, B.2) ∧ (B.1, B.2) = (C.1, B.2) ∧ (C.1, C.2) = (C.1, D.2) ∧ (D.1, D.2) = (A.1, D.2))
    (h_inside : X.1 > A.1 ∧ X.1 < C.1 ∧ X.2 > A.2 ∧ X.2 < C.2)
    (h_intersections : (P.1, P.2, Q.1, Q.2, R.1, R.2, S.1, S.2))
    (h_perpendicular_lines : ((P.1 - X.1) * (R.1 - X.1) + (P.2 - X.2) * (R.2 - X.2) = 0 ∨
                             (P.1 - X.1) * (S.1 - X.1) + (P.2 - X.2) * (S.2 - X.2) = 0) ∧
                            ((Q.1 - X.1) * (S.1 - X.1) + (Q.2 - X.2) * (S.2 - X.2) = 0 ∨
                             (Q.1 - X.1) * (R.1 - X.1) + (Q.2 - X.2) * (R.2 - X.2) = 0)) :
    dist P Q = dist R S :=
by
  sorry

end equal_segments_in_square_l773_773639


namespace number_of_combinations_with_odd_sum_l773_773061

open Finset

/-- The set of numbers from 1 to 9 -/
def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- The set of odd numbers in A -/
def odd_numbers : Finset ℕ := {1, 3, 5, 7, 9}

/-- The set of even numbers in A -/
def even_numbers : Finset ℕ := {2, 4, 6, 8}

/-- The number of ways to select four numbers from A such that the sum is odd -/
theorem number_of_combinations_with_odd_sum : 
  (choose 5 1) * (choose 4 3) + (choose 5 3) * (choose 4 1) = 60 :=
by 
  simp [choose]
  sorry

end number_of_combinations_with_odd_sum_l773_773061


namespace Sn_equilateral_triangle_l773_773782

theorem Sn_equilateral_triangle (n : ℕ) (h : n > 0) :
  let A := 0, B := (1, 0 : ℝ), C := (1/2, √3/2 : ℝ)
  let P : ℕ → ℝ × ℝ := λ k, ((1 - k / n) * 1 + (k / n) * 1, (1 - k / n) * 0 + (k / n) * (√3/2))
  let S_n := ∑ k in finset.range (n - 1), (B.1 - A) * (P k).1 + (P k).1 * (P (k + 1)).1
  (S_n = (5 * n^2 - 2) / (6 * n)) :=
sorry

end Sn_equilateral_triangle_l773_773782


namespace find_a_of_inequality_solution_set_l773_773858

theorem find_a_of_inequality_solution_set :
  (∃ (a : ℝ), (∀ (x : ℝ), |2*x - a| + a ≤ 4 ↔ -1 ≤ x ∧ x ≤ 2) ∧ a = 1) :=
by sorry

end find_a_of_inequality_solution_set_l773_773858


namespace odd_function_f_minus_one_l773_773308

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 4 * x + (f 0 - 4 * 0)
  else -f (-x)

theorem odd_function_f_minus_one :
  (∃ m : ℝ, ∀ x : ℝ, f x = if x ≥ 0 then 4 * x + m else -f (-x)) →
  (f 0 = 0) →
  f (-1) = -3 :=
by
  sorry

end odd_function_f_minus_one_l773_773308


namespace distance_between_parallel_lines_l773_773410

theorem distance_between_parallel_lines : 
  ∀ (x y : ℝ), 
  (3 * x - 4 * y - 3 = 0) ∧ (6 * x - 8 * y + 5 = 0) → 
  ∃ d : ℝ, d = 11 / 10 :=
by
  sorry

end distance_between_parallel_lines_l773_773410


namespace identify_quadratic_function_l773_773658

-- Define the functions
def fA (x : ℝ) : ℝ := x - 2
def fB (x : ℝ) : ℝ := x^2
def fC (x : ℝ) : ℝ := x^2 - (x + 1)^2
def fD (x : ℝ) : ℝ := 2 / x^2

-- Define what it means to be a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c)

-- Theorem stating that fB is the quadratic function among the given options
theorem identify_quadratic_function :
  is_quadratic fB ∧ ¬ is_quadratic fA ∧ ¬ is_quadratic fC ∧ ¬ is_quadratic fD :=
by
  sorry

end identify_quadratic_function_l773_773658


namespace existence_of_8_segments_l773_773025

theorem existence_of_8_segments (segments : Fin 50 → Segment) :
  (∃ (s : Finset (Fin 50)), s.card = 8 ∧ ∀ (i j : Fin 50), i ∈ s → j ∈ s → i ≠ j → disjoint (segments i) (segments j)) ∨
  (∃ (s : Finset (Fin 50)), s.card = 8 ∧ ∃ p, ∀ i ∈ s, p ∈ segments i) := 
sorry

end existence_of_8_segments_l773_773025


namespace calories_consumed_Jean_l773_773900

def donuts_per_page (pages : ℕ) : ℕ := pages / 2

def calories_per_donut : ℕ := 150

def total_calories (pages : ℕ) : ℕ :=
  let donuts := donuts_per_page pages
  donuts * calories_per_donut

theorem calories_consumed_Jean (h1 : ∀ pages, donuts_per_page pages = pages / 2)
  (h2 : calories_per_donut = 150)
  (h3 : total_calories 12 = 900) :
  total_calories 12 = 900 := by
  sorry

end calories_consumed_Jean_l773_773900


namespace num_five_digit_numbers_is_correct_l773_773840

-- Define the set of digits and their repetition as given in the conditions
def digits : Multiset ℕ := {1, 3, 3, 5, 8}

-- Calculate the permutation with repetitions
noncomputable def num_five_digit_numbers : ℕ := (digits.card.factorial) / 
  (Multiset.count 1 digits).factorial / 
  (Multiset.count 3 digits).factorial / 
  (Multiset.count 5 digits).factorial / 
  (Multiset.count 8 digits).factorial

-- Theorem stating the final result
theorem num_five_digit_numbers_is_correct : num_five_digit_numbers = 60 :=
by
  -- Proof is omitted
  sorry

end num_five_digit_numbers_is_correct_l773_773840


namespace recip_sum_focus_l773_773467

-- Given definitions from the problem.
def parametric_ellipse (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

def ellipse : Set (ℝ × ℝ) :=
  {p | (p.1 ^ 2) / 4 + (p.2 ^ 2) / 3 = 1}

def right_focus : ℝ × ℝ :=
  (1, 0)

def l : Set (ℝ × ℝ) :=
  {p | p.1 = 1}

def distances (M N F : ℝ × ℝ) : ℝ × ℝ :=
  (Real.sqrt ((M.1 - F.1) ^ 2 + (M.2 - F.2) ^ 2),
   Real.sqrt ((N.1 - F.1) ^ 2 + (N.2 - F.2) ^ 2))

noncomputable def recip_sum (M N F : ℝ × ℝ) : ℝ :=
  let (m, n) := distances M N F in (1 / m + 1 / n)

-- The Lean theorem statement to prove.
theorem recip_sum_focus (M N : ℝ × ℝ) (hM : M ∈ ellipse) (hN : N ∈ ellipse) (hM_line : M ∈ l) (hN_line : N ∈ l) :
  recip_sum M N right_focus = 4 / 3 :=
sorry

end recip_sum_focus_l773_773467


namespace DF_is_5_point_5_l773_773020

variables {A B C D E F : Type}
variables (congruent : triangle A B C ≃ triangle D E F)
variables (ac_length : AC = 5.5)

theorem DF_is_5_point_5 : DF = 5.5 :=
by
  -- skipped proof
  sorry

end DF_is_5_point_5_l773_773020


namespace line_contains_point_l773_773736

theorem line_contains_point {
    k : ℝ
} :
  (2 - k * 3 = -4 * 1) → k = 2 :=
by
  sorry

end line_contains_point_l773_773736


namespace possible_question_l773_773859

-- We represent the concept of Ilya always telling the truth and giving different answers 
-- by setting up necessary predicates and showing a proof outline.

-- Definitions:
-- Ilya always tells the truth (condition 1)
axiom Ilya_tells_truth : Prop

-- When asked the same question twice, he gave different answers (condition 2)
axiom Different_answers (q1 q2 : Prop) : (q1 ≠ q2)

-- We need to prove that the question "How many questions have I already asked you?" 
-- can cause different answers if asked twice.
theorem possible_question (I_truthful : Ilya_tells_truth) (diff_a : ∀ q, Different_answers q q) :
  ∃ (q : Prop), I_truthful → diff_a q := 
begin
  sorry
end

end possible_question_l773_773859


namespace nested_root_expression_l773_773717

theorem nested_root_expression :
  (∛0.000008) ^ (1/4 : ℚ) = (2 : ℝ) ^ (1/4 : ℚ) / (10 : ℝ) ^ (1/2 : ℚ) := 
by 
{
  sorry,
}

end nested_root_expression_l773_773717


namespace find_number_l773_773102

theorem find_number (x : ℤ) :
  45 - (x - (37 - (15 - 18))) = 57 → x = 28 :=
by
  sorry

end find_number_l773_773102


namespace find_x_l773_773654

theorem find_x (x : ℝ) (h : 3 * x = 36 - x + 16) : x = 13 :=
by
  sorry

end find_x_l773_773654


namespace inequality_div_half_l773_773096

theorem inequality_div_half (a b : ℝ) (h : a > b) : (a / 2 > b / 2) :=
sorry

end inequality_div_half_l773_773096


namespace inequality_proof_l773_773193

theorem inequality_proof
  (a b c : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : c^2 + a * b = a^2 + b^2) :
  c^2 + a * b ≤ a * c + b * c := sorry

end inequality_proof_l773_773193


namespace longest_perimeter_l773_773673

theorem longest_perimeter 
  (x : ℝ) (h : x > 1)
  (pA : ℝ := 4 + 6 * x)
  (pB : ℝ := 2 + 10 * x)
  (pC : ℝ := 7 + 5 * x)
  (pD : ℝ := 6 + 6 * x)
  (pE : ℝ := 1 + 11 * x) :
  pE > pA ∧ pE > pB ∧ pE > pC ∧ pE > pD :=
by
  sorry

end longest_perimeter_l773_773673


namespace parabola_directrix_distance_l773_773960

theorem parabola_directrix_distance :
  let P := parabola {y : ℝ | ∃ x : ℝ, y^2 = 4 * x}
  let F := (1 : ℝ, 0 : ℝ)
  let dir := { x : ℝ | x = -1 }
  distance F dir = 2 :=
by sorry

end parabola_directrix_distance_l773_773960


namespace triangle_largest_angle_l773_773875

theorem triangle_largest_angle (a b c : ℝ) (h1 : 10 * a / 2 = 24 * b / 2)
                               (h2 : 24 * b / 2 = 15 * c / 2)
                               (h3 : a + b > c)
                               (h4 : b + c > a)
                               (h5 : c + a > b) :
  let angle := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) in
  120° < angle ∧ angle < 150° := 
sorry

end triangle_largest_angle_l773_773875


namespace dessert_menu_count_is_324_l773_773324

-- Define the days of the week
inductive Day : Type
| Sunday : Day
| Monday : Day
| Tuesday : Day
| Wednesday : Day
| Thursday : Day
| Friday : Day
| Saturday : Day

-- Define the dessert options
inductive Dessert : Type
| cake : Dessert
| pie : Dessert
| ice_cream : Dessert
| pudding : Dessert

-- Define the condition and questions
def isValidDessertMenu (dessertMenu : Day → Dessert) : Prop :=
  (dessertMenu Day.Monday = Dessert.cake) ∧
  (dessertMenu Day.Thursday = Dessert.ice_cream) ∧
  (∀ d : Day, match d with
              | Day.Sunday => true
              | Day.Monday => dessertMenu d ≠ dessertMenu Day.Sunday
              | Day.Tuesday => (dessertMenu d ≠ dessertMenu Day.Monday)
              | Day.Wednesday => (dessertMenu d ≠ dessertMenu Day.Tuesday)
              | Day.Thursday => (dessertMenu d ≠ dessertMenu Day.Wednesday)
              | Day.Friday => (dessertMenu d ≠ dessertMenu Day.Thursday)
              | Day.Saturday => (dessertMenu d ≠ dessertMenu Day.Friday)
              end)

-- Prove the total valid dessert menus is 324
theorem dessert_menu_count_is_324 : ∃ (dessertMenu : Day → Dessert), 
  isValidDessertMenu dessertMenu ∧ (4 * 1 * 3 * 3 * 1 * 3 * 3 = 324) :=
sorry

end dessert_menu_count_is_324_l773_773324


namespace number_and_sum_of_g_16_l773_773163

theorem number_and_sum_of_g_16 (g : ℕ → ℕ)
  (h : ∀ (a b : ℕ), 2 * g (a^2 + b^2) = g a ^ 2 + g b ^ 2) :
  let n := {y | ∃ a b : ℕ, g(a^2 + b^2) = y}.finite.to_finset.card in
  let s := {y | ∃ a b : ℕ, g(a^2 + b^2) = y}.sum id in
  n * s = 99 :=
by
  sorry

end number_and_sum_of_g_16_l773_773163


namespace range_scalar_product_l773_773517

noncomputable def cartesian_coord_A (θ : ℝ) (ρ : ℝ) : ℝ × ℝ :=
(ρ * cos θ, ρ * sin θ)

noncomputable def parametric_ellipse (θ : ℝ) : ℝ × ℝ :=
(√3 * cos θ, sin θ)

def scalar_product_AE_AF (θ : ℝ) : ℝ :=
5 - 3 * cos θ - 2 * sin θ

theorem range_scalar_product :
  ∃ α : ℝ, (tan α = (2 / 3)) ∧ 
  (∀ θ : ℝ, let s := scalar_product_AE_AF θ
  in s ≥ 5 - √13 ∧ s ≤ 5 + √13) :=
sorry

end range_scalar_product_l773_773517


namespace constant_term_in_binomial_expansion_l773_773733

theorem constant_term_in_binomial_expansion :
  let a := (3 : ℝ)
  let b := (-1 : ℝ)
  let n := 6
  let const_term := binomial_coefficient n 3 * 3^(n-3) * (-1)^3
  (3x - x⁻¹) ^ n = const_term ↔ const_term = -540 :=
by
  sorry

end constant_term_in_binomial_expansion_l773_773733


namespace matrix_multiplication_correct_l773_773376

-- Define the matrices
def A : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![2, 0, -3],
    ![1, 3, -2],
    ![0, 2, 4]
  ]

def B : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![1, -1, 0],
    ![0, 2, -1],
    ![3, 0, 1]
  ]

def C : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![-7, -2, -3],
    ![-5, 5, -5],
    ![12, 4, 2]
  ]

-- Proof statement that multiplication of A and B gives C
theorem matrix_multiplication_correct : A * B = C := 
by
  sorry

end matrix_multiplication_correct_l773_773376


namespace sufficient_but_not_necessary_condition_l773_773440

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 2 → a^2 > 2 * a) ∧ ¬(a^2 > 2 * a → a > 2) :=
begin
  sorry
end

end sufficient_but_not_necessary_condition_l773_773440


namespace probability_S7_eq_3_l773_773870

theorem probability_S7_eq_3 :
  let a_n (n : ℕ) : ℤ := if n % 3 = 0 then -1 else 1 
  let S_n (n : ℕ) : ℤ := ∑ i in finset.range n, a_n i
  let probability (x : ℕ) (y : ℕ) := nat.choose 7 5 * (1/3)^5 * (2/3)^2 
  (S_n 7 = 3) → probability 5 2 = C_7^5 \frac{1}{3}^5 \frac{2}{3}^2 :=
by
  sorry

end probability_S7_eq_3_l773_773870


namespace susan_spaces_to_win_l773_773213

def spaces_in_game : ℕ := 48
def first_turn_movement : ℤ := 8
def second_turn_movement : ℤ := 2 - 5
def third_turn_movement : ℤ := 6

def total_movement : ℤ :=
  first_turn_movement + second_turn_movement + third_turn_movement

def spaces_to_win (spaces_in_game : ℕ) (total_movement : ℤ) : ℤ :=
  spaces_in_game - total_movement

theorem susan_spaces_to_win : spaces_to_win spaces_in_game total_movement = 37 := by
  sorry

end susan_spaces_to_win_l773_773213


namespace max_knights_is_8_l773_773742

def Person : Type := { is_knight : Bool, number : ℕ, statement1 : ℕ, statement2 : ℕ }

def valid_knight (p : Person) : Prop :=
  if p.is_knight then
    (p.statement1 < p.number ∨ p.statement2 < p.number) ∧
    (p.statement1 > p.number ∨ p.statement2 > p.number)
  else
    (p.statement1 ≥ p.number ∧ p.statement2 ≥ p.number) ∧
    (p.statement1 ≤ p.number ∧ p.statement2 ≤ p.number)

def max_knights : ℕ → Prop :=
  λ n, ∃ (ps : List Person), ps.length = 10 ∧
        (∑ p in ps, if p.is_knight then 1 else 0) = n ∧
        (∀ p in ps, valid_knight p)

theorem max_knights_is_8 : max_knights 8 :=
sorry

end max_knights_is_8_l773_773742


namespace passenger_capacity_passenger_capacity_at_5_max_profit_l773_773997

section SubwayProject

-- Define the time interval t and the passenger capacity function p(t)
def p (t : ℕ) : ℕ :=
  if 2 ≤ t ∧ t < 10 then 300 + 40 * t - 2 * t^2
  else if 10 ≤ t ∧ t ≤ 20 then 500
  else 0

-- Define the net profit function Q(t)
def Q (t : ℕ) : ℚ :=
  if 2 ≤ t ∧ t < 10 then (8 * p t - 2656) / t - 60
  else if 10 ≤ t ∧ t ≤ 20 then (1344 : ℚ) / t - 60
  else 0

-- Statement 1: Prove the correct expression for p(t) and its value at t = 5
theorem passenger_capacity (t : ℕ) (ht1 : 2 ≤ t) (ht2 : t ≤ 20) :
  (p t = if 2 ≤ t ∧ t < 10 then 300 + 40 * t - 2 * t^2 else 500) :=
sorry

theorem passenger_capacity_at_5 : p 5 = 450 :=
sorry

-- Statement 2: Prove the time interval t and the maximum value of Q(t)
theorem max_profit : ∃ t : ℕ, 2 ≤ t ∧ t ≤ 10 ∧ Q t = 132 ∧ (∀ u : ℕ, 2 ≤ u ∧ u ≤ 10 → Q u ≤ Q t) :=
sorry

end SubwayProject

end passenger_capacity_passenger_capacity_at_5_max_profit_l773_773997


namespace integer_solutions_count_l773_773489

theorem integer_solutions_count :
  {x : ℤ | (x - 3) ^ (36 - x ^ 2) = 1}.to_finset.card = 4 := 
by
  sorry

end integer_solutions_count_l773_773489


namespace solution_set_l773_773407

open Real

noncomputable def condition (x : ℝ) := x ≥ 2

noncomputable def eq_1 (x : ℝ) := sqrt (x + 5 - 6 * sqrt (x - 2)) + sqrt (x + 12 - 8 * sqrt (x - 2)) = 2

theorem solution_set :
  {x : ℝ | condition x ∧ eq_1 x} = {x : ℝ | 11 ≤ x ∧ x ≤ 18} :=
by sorry

end solution_set_l773_773407


namespace no_intersection_of_ellipses_l773_773386

theorem no_intersection_of_ellipses :
  (∀ (x y : ℝ), (9*x^2 + y^2 = 9) ∧ (x^2 + 16*y^2 = 16) → false) :=
sorry

end no_intersection_of_ellipses_l773_773386


namespace min_value_of_expression_l773_773166

open Real

noncomputable def condition (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ (1 / (x + 2) + 1 / (y + 2) = 1 / 4)

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 4) :
  2 * x + 3 * y = 5 + 4 * sqrt 3 :=
sorry

end min_value_of_expression_l773_773166


namespace expected_tie_moments_at_10_l773_773269

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def indicator_probability (k : ℕ) : ℝ :=
  (binomial_coefficient (2 * k) k) / (2:ℝ)^(2 * k)

def expected_tie_moments (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), indicator_probability k

theorem expected_tie_moments_at_10 :
  expected_tie_moments 5 = 1.70703125 :=
by
  sorry

end expected_tie_moments_at_10_l773_773269


namespace monthlyShoeSales_l773_773924

-- Defining the conditions
def pairsSoldLastWeek := 27
def pairsSoldThisWeek := 12
def pairsNeededToMeetGoal := 41

-- Defining the question as a statement to prove
theorem monthlyShoeSales : pairsSoldLastWeek + pairsSoldThisWeek + pairsNeededToMeetGoal = 80 := by
  sorry

end monthlyShoeSales_l773_773924


namespace annual_revenue_increase_l773_773677

structure Factory := 
  (daily_production : ℕ)
  (grade_a_threshold : ℝ)
  (price_a : ℝ)
  (price_b : ℝ)
  (cqi_values : List ℝ)
  (cqi_mean : ℝ)
  (cqi_variance : ℝ)
  (improvement_cost : ℝ)
  (improvement_increase : ℝ)
  (days_per_year : ℕ)

noncomputable def improved_mean (factory : Factory) : ℝ :=
  factory.cqi_mean + factory.improvement_increase

noncomputable def revenue (units : ℕ) (price : ℝ) (prob : ℝ) : ℝ :=
  units * price * prob

noncomputable def probabilities (values : List ℝ) (threshold : ℝ) : (ℝ × ℝ) :=
  let a_count := (values.countp (λ x => x ≥ threshold))
  let count := values.length
  (a_count / count, (count - a_count) / count)

theorem annual_revenue_increase (factory : Factory) :
  -- Calculations for daily revenue before improvement
  let (pA_before, pB_before) := probabilities factory.cqi_values factory.grade_a_threshold in
  let daily_revenue_before := revenue factory.daily_production factory.price_a pA_before + 
                              revenue factory.daily_production factory.price_b pB_before in
  -- Improved CQI values
  let improved_cqi_values := factory.cqi_values.map (λ x => x + factory.improvement_increase) in
  -- Calculations for daily revenue after improvement
  let (pA_after, pB_after) := probabilities improved_cqi_values factory.grade_a_threshold in
  let daily_revenue_after := revenue factory.daily_production factory.price_a pA_after + 
                             revenue factory.daily_production factory.price_b pB_after in
  -- Annual revenue increase
  let annual_increase := (daily_revenue_after - daily_revenue_before) * factory.days_per_year - factory.improvement_cost in
  -- Proving the annual revenue increase and mean, variance adjustments
  annual_increase = 15625 ∧ improved_mean factory = 10.02 ∧ factory.cqi_variance = 0.045 := by
  sorry

end annual_revenue_increase_l773_773677


namespace actual_average_speed_l773_773685

theorem actual_average_speed 
  (v t : ℝ)
  (h : v * t = (v + 21) * (2/3) * t) : 
  v = 42 :=
by
  sorry

end actual_average_speed_l773_773685


namespace calorie_difference_l773_773738

-- Definitions and conditions
def calories_per_serving_X : ℕ := 5
def calories_per_serving_Y : ℕ := 8
def ratio_XY : ℕ × ℕ := (2, 3)
def lunch_calories := (2 * calories_per_serving_X) + (3 * calories_per_serving_Y) -- 34
def max_calories_per_dietitian : ℕ := 25

-- Consumed calories by each dietitian
def calories_A : ℝ := (3 / 4 : ℝ) * lunch_calories
def calories_B : ℝ := (5 / 6 : ℝ) * lunch_calories
def calories_C : ℝ := (1 / 2 : ℝ) * lunch_calories
def total_calories_consumed : ℝ := calories_A + calories_B + calories_C

-- Total recommended calories for three dietitians
def total_recommended_calories : ℕ := 3 * max_calories_per_dietitian

-- Theorem statement: prove the calorie difference
theorem calorie_difference : total_calories_consumed = total_recommended_calories - 4.17 :=
by
  -- The proof would go here
  sorry

end calorie_difference_l773_773738


namespace cos_pi_minus_theta_l773_773808

theorem cos_pi_minus_theta (x y : ℝ) (h1 : x = 4) (h2 : y = -3) (r : ℝ) 
  (h3 : r = Real.sqrt (x^2 + y^2)) :
  let θ := Real.arccos (x / r) in
  (r = 5) ∧ (cos (π - θ) = -4 / 5) :=
by
  -- Using the given conditions
  have hx : x = 4 := h1
  have hy : y = -3 := h2
  have hr : r = Real.sqrt (4^2 + (-3)^2) := h3

  -- Calculate r
  rw [hx, hy] at hr
  have r_value : r = 5 := by linarith

  -- Define θ
  let θ := Real.arccos (x / r)

  -- Find cos(θ)
  have hθ' : cos θ = x / r := Real.cos_arccos (by rw [hx, r_value]; exact le_of_lt (lt_add_iff_pos_left (x / 5)).1 (by norm_num))
  rw [hx, r_value] at hθ'
  have hθ : cos θ = 4 / 5 := hθ'
  
  -- Calculate cos(π - θ)
  have h_cos_pi_minus_theta : cos (π - θ) = -cos θ := Real.cos_pi_sub (by linarith)
  rw [hθ] at h_cos_pi_minus_theta

  -- Prove the final result
  split
  . exact r_value
  . exact h_cos_pi_minus_theta
  done

end cos_pi_minus_theta_l773_773808


namespace min_value_of_k_l773_773886

theorem min_value_of_k 
  (AB CD : ℝ) 
  (E F : ℝ → ℝ) 
  (area_trapezoid : ℝ) 
  (k : ℕ) :
  CD = 2 * AB →
  AB = 2 * a →
  k > 0 →
  (let h := some_height in
  let area_triangle_CDG := (1 / 2) * 4 * a * (1 / 2) * h in
  let area_quadrilateral_AEGF := (1 / 2) * 2 * a * h in
  area_triangle_CDG - area_quadrilateral_AEGF = k / 24) →
  (let trapezoid_integral_area := 3 * a * h in
  (area_trapezoid = trapezoid_integral_area) →
  ∀ ah,
    3 * (k / 24)) ≤ ah →
  k % 8 = 0 → 
  k = 8)
:= 
sorry

end min_value_of_k_l773_773886


namespace problem_correct_statements_l773_773579

noncomputable def f (x : ℝ) : ℝ := real.sqrt (-x^2 + 2*x + 3)

theorem problem_correct_statements :
  (∀ x, -1 ≤ x → x ≤ 3 → f(x) ≥ 0) ∧
  ((∀ x, -1 < x → x < 1 → f(x) < f(x + ε)) ∧
  (∀ x, 1 < x → x < 3 → f(x) > f(x + ε)) ∧
  (∀ x, 1 ≤ x → 3 ≤ x → f(x) ≤ 2)) :=
begin
  sorry
end

end problem_correct_statements_l773_773579


namespace dan_initial_money_l773_773727

theorem dan_initial_money (cost_candy : ℕ) (cost_chocolate : ℕ) (total_spent: ℕ) (hc : cost_candy = 7) (hch : cost_chocolate = 6) (hs : total_spent = 13) 
  (h : total_spent = cost_candy + cost_chocolate) : total_spent = 13 := by
  sorry

end dan_initial_money_l773_773727


namespace triangle_properties_l773_773003

-- Definitions for vertices A, B, C
def A := (2 : ℝ, -3 : ℝ)
def B := (-1 : ℝ, 4 : ℝ)
def C := (4 : ℝ, -2 : ℝ)

-- Proof statement for area and length
theorem triangle_properties :
  let v := (2 - 4, -3 - (-2)) in
  let w := (-1 - 4, 4 - (-2)) in
  let area := real.abs ((v.1 * w.2) - (v.2 * w.1)) / 2 in
  let AC := real.sqrt ((2 - 4)^2 + (-3 + 2)^2) in
  area = 17 / 2 ∧ AC = real.sqrt 5 :=
by
  sorry

end triangle_properties_l773_773003


namespace line_and_ellipse_intersection_l773_773134

theorem line_and_ellipse_intersection :
  let l := λ (θ : ℝ), ⟨1 / sin (θ - π / 4), θ⟩ in
  let C := λ (θ : ℝ), ⟨2 * cos θ, sqrt 3 * sin θ⟩ in
  (∀ θ, l θ = (∃ (x y : ℝ), x - y + sqrt 2 = 0)) ∧
  (∀ θ, C θ = (∃ (x y : ℝ), (x / 2) ^ 2 + (y / sqrt 3) ^ 2 = 1)) ∧
  let M := ⟨√2 / 2, √2 + √2 / 2⟩ in
  let N := ⟨-√2 / 2, √2 - √2 / 2⟩ in
  let P := (0, √2) in
  abs (dist P M) + abs (dist P N) = 4 * sqrt 30 / 7 :=
begin
  sorry
end

end line_and_ellipse_intersection_l773_773134


namespace ella_dog_food_l773_773527

theorem ella_dog_food (d : ℕ) : 
  (∀ (d : ℕ), 1000 = (20 * d) + (80 * d) → d = 10) := 
begin
  intro d,
  intro h,
  have h1 : (20 * d) + (80 * d) = 100 * d := by ring,
  rw h1 at h,
  have h2 : 100 * d = 1000 := h,
  exact eq_of_mul_eq_mul_right (by norm_num) h2
end

end ella_dog_food_l773_773527


namespace vertex_of_parabola_range_of_a_l773_773829

-- Problem (1): Prove the coordinates of the vertex when a = 1
theorem vertex_of_parabola (a : ℝ) (ha : a = 1) :
  let y := λ x : ℝ, x^2 - 2 * a * x + a^2 + 2 * a in
  (∃ x_vertex y_vertex, (y x_vertex = y_vertex) ∧ (x_vertex = 1) ∧ (y_vertex = 2)) := 
by {
  sorry
}

-- Problem (2): Prove the range of a such that PQ decreases for m < 3
theorem range_of_a (a : ℝ) (ha : a > 0) :
  (forall m < 3, (m - a - 1)^2 + 1 < (m-1 - a - 1)^2 + 1 -> m >= a + 1) ↔ (a >= 2) :=
by {
  sorry
}

end vertex_of_parabola_range_of_a_l773_773829


namespace sum_squares_of_distances_l773_773352

theorem sum_squares_of_distances
  (s d : ℝ)
  (h1 : ∀ (A B C : Type), ∃ (D1 D2 : Type) (E1 E2 E3 E4 : Type), 
      (BD1 = BD2 = d) ∧ d < s ∧
      (∃ (tri1 : Triangle ABC), congruent tri1 (triangle AD1E1)) ∧ 
      (∃ (tri2 : Triangle ABC), congruent tri2 (triangle AD1E2)) ∧ 
      (∃ (tri3 : Triangle ABC), congruent tri3 (triangle AD2E3)) ∧ 
      (∃ (tri4 : Triangle ABC), congruent tri4 (triangle AD2E4)) ∧
      BD1 = BD2 ∧ BD1 < s ∧ s > d):
  ∑ k in finset.range 4, (CE_k ^ 2) = 12 * (s ^ 2) := 
sorry

end sum_squares_of_distances_l773_773352


namespace coefficient_of_x4y2_in_expansion_l773_773601

theorem coefficient_of_x4y2_in_expansion 
  (binom : ∀ n k : ℕ, ℕ)
  (expansion_term : ∀ (n r : ℕ) (x y : ℕ), ℕ) 
  (h_binom : ∀ r, binom 5 r = Nat.choose 5 r) :
  expansion_term 5 r 3 (-1) 2 :=
  sorry

end coefficient_of_x4y2_in_expansion_l773_773601


namespace distance_from_top_to_bottom_l773_773699

-- Define the initial conditions
def top_ring_outer_diameter := 30
def top_ring_thickness := 2
def bottom_ring_outer_diameter := 10
def diameter_decrement := 2
def thickness_decrement := 0.1

def rings : ℕ := (top_ring_outer_diameter - bottom_ring_outer_diameter) / diameter_decrement + 1

def calc_internal_diameter (outer_diameter thickness : ℝ) : ℝ :=
  outer_diameter - 2 * thickness

def internal_diameters_sum : ℝ :=
  let seq := (list.range rings).map (λ i, calc_internal_diameter (top_ring_outer_diameter - diameter_decrement * i) (top_ring_thickness - thickness_decrement * i))
  seq.sum

def total_distance := internal_diameters_sum + top_ring_thickness + (top_ring_thickness - thickness_decrement * (rings - 1))

theorem distance_from_top_to_bottom : total_distance = 117 := by
  -- Proof will be added here
  sorry

end distance_from_top_to_bottom_l773_773699


namespace f_f_one_ninth_l773_773823

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3 else 2 ^ x

theorem f_f_one_ninth :
  f (f (1 / 9)) = 1 / 4 :=
by
  sorry

end f_f_one_ninth_l773_773823


namespace jean_total_calories_l773_773895

-- Define the conditions
def pages_per_donut : ℕ := 2
def written_pages : ℕ := 12
def calories_per_donut : ℕ := 150

-- Define the question as a theorem
theorem jean_total_calories : (written_pages / pages_per_donut) * calories_per_donut = 900 := by
  sorry

end jean_total_calories_l773_773895


namespace complex_subtraction_l773_773855

theorem complex_subtraction 
  (z1 : ℂ) (z2 : ℂ) (h1 : z1 = 3 + 4 * Complex.i) (h2 : z2 = 1 + 2 * Complex.i) : 
  z1 - z2 = 2 + 2 * Complex.i := 
by 
  sorry

end complex_subtraction_l773_773855


namespace calories_consumed_l773_773899

-- Define the conditions
def pages_written : ℕ := 12
def pages_per_donut : ℕ := 2
def calories_per_donut : ℕ := 150

-- Define the theorem to be proved
theorem calories_consumed (pages_written : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  (pages_written / pages_per_donut) * calories_per_donut

-- Ensure the theorem corresponds to the correct answer
example : calories_consumed 12 2 150 = 900 := by
  sorry

end calories_consumed_l773_773899


namespace negation_irrational_square_rational_is_irrational_square_l773_773973
open Classical

variable {α : Type}

def rational (x : α) [LinearOrderedField α] : Prop := ∃ (r s : ℤ), s ≠ 0 ∧ x = r / s
def irrational (x : α) [LinearOrderedField α] : Prop := ¬ rational x

theorem negation_irrational_square_rational_is_irrational_square :
  ¬ (∃ (x : α) [LinearOrderedField α], irrational x ∧ rational (x * x)) ↔ 
  ∀ (y : α) [LinearOrderedField α], irrational y → irrational (y * y) := 
by sorry

end negation_irrational_square_rational_is_irrational_square_l773_773973


namespace height_of_tree_in_kilmer_park_l773_773613

-- Define the initial conditions
def initial_height_ft := 52
def growth_per_year_ft := 5
def years := 8
def ft_to_inch := 12

-- Define the expected result in inches
def expected_height_inch := 1104

-- State the problem as a theorem
theorem height_of_tree_in_kilmer_park :
  (initial_height_ft + growth_per_year_ft * years) * ft_to_inch = expected_height_inch :=
by
  sorry

end height_of_tree_in_kilmer_park_l773_773613


namespace jamies_score_l773_773522

def quiz_score (correct incorrect unanswered : ℕ) : ℚ :=
  (correct * 2) + (incorrect * (-0.5)) + (unanswered * 0.25)

theorem jamies_score :
  quiz_score 16 10 4 = 28 :=
by
  sorry

end jamies_score_l773_773522


namespace final_price_of_coat_after_discounts_l773_773707

def original_price : ℝ := 120
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.20

theorem final_price_of_coat_after_discounts : 
    (1 - second_discount) * (1 - first_discount) * original_price = 72 := 
by
    sorry

end final_price_of_coat_after_discounts_l773_773707


namespace infinite_product_l773_773239

noncomputable def sequence : ℕ → ℝ
| 0       := 1 / 2
| (n + 1) := Real.sqrt ((1 + sequence n) / 2)

theorem infinite_product : 
  ∏' n, sequence n = (3 * Real.sqrt 3) / (4 * Real.pi) :=
sorry

end infinite_product_l773_773239


namespace tangent_line_at_point_tangent_line_perpendicular_to_l773_773822

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 1

theorem tangent_line_at_point (x y : ℝ) (h : (x, y) = (1, -1)) :
    let m := 3 * x^2 - 1 in
    y - (-1) = m * (x - 1) ∧ m = 2 :=
by
  -- proof goes here
  sorry

theorem tangent_line_perpendicular_to (x y : ℝ) (h : f'(x) = 2) (h₁ : y = 2 * x - c) :
    ∃ m ∈ {1, -1}, f m = y := 
by
  -- proof goes here
  sorry

end tangent_line_at_point_tangent_line_perpendicular_to_l773_773822


namespace evaluate_floor_ceil_sum_l773_773000

/- 
  Definitions of floor and ceiling functions along with specific values.
  We will use these definitions to prove the problem statement.
-/

def floor (x : ℝ) : ℤ := Int.floor x
def ceil (x : ℝ) : ℤ := Int.ceil x

def x1 : ℝ := 0.999
def x2 : ℝ := 2.001

theorem evaluate_floor_ceil_sum : floor x1 + ceil x2 = 3 := 
by
  sorry

end evaluate_floor_ceil_sum_l773_773000


namespace a_beats_b_by_meters_l773_773119
noncomputable def speed (distance : ℕ) (time : ℕ) : ℚ := distance / time

theorem a_beats_b_by_meters :
  let distance_A := 1000 in
  let time_A := 92 in
  let time_margin := 8 in
  let speed_A := speed distance_A time_A in

  (speed_A * time_margin : ℝ).toInt = 87 :=
by
  -- the proof here is omitted
  sorry

end a_beats_b_by_meters_l773_773119


namespace avg_salary_correct_l773_773952

-- Definitions
-- There are 14 workers, 7 of whom are technicians and the rest are non-technicians
def num_workers : ℕ := 14
def num_technicians : ℕ := 7
def num_non_technicians : ℕ := num_workers - num_technicians

-- Average salaries
def avg_salary_technicians : ℝ := 10000
def avg_salary_non_technicians : ℝ := 6000

-- Total salaries
def total_salary_technicians : ℝ := num_technicians * avg_salary_technicians
def total_salary_non_technicians : ℝ := num_non_technicians * avg_salary_non_technicians
def total_salary_workers : ℝ := total_salary_technicians + total_salary_non_technicians

-- Average salary of all workers
def avg_salary_workers : ℝ := total_salary_workers / num_workers

-- Theorem Statement
theorem avg_salary_correct : avg_salary_workers = 8000 := 
by
  sorry

end avg_salary_correct_l773_773952


namespace intersection_M_N_l773_773831

noncomputable def set_M : Set ℝ := {x | x^2 - 3 * x - 4 ≤ 0}
noncomputable def set_N : Set ℝ := {x | Real.log x ≥ 0}

theorem intersection_M_N :
  {x | x ∈ set_M ∧ x ∈ set_N} = {x | 1 ≤ x ∧ x ≤ 4} :=
sorry

end intersection_M_N_l773_773831


namespace probability_at_least_one_defective_l773_773465

open Finset

noncomputable def probability_defective (n : ℕ) (k : ℕ) (d : ℕ) : ℚ :=
  let total := choose n k
  let non_defective := choose (n - d) k
  1 - (non_defective : ℚ) / total

theorem probability_at_least_one_defective :
  let n := 10
  let k := 3
  let d := 3
  at_probability_least_one_defective (n k d) = 17 / 24 :=
by
  sorry

end probability_at_least_one_defective_l773_773465


namespace triangle_trigonometry_problem_l773_773031

theorem triangle_trigonometry_problem
  (A B C : ℝ)
  (h1 : sin A = cos B)
  (h2 : cos B = tan C)
  (h3 : A + B + C = π) :
  cos A ^ 3 + cos A ^ 2 - cos A = 1 / 2 :=
by
  sorry

end triangle_trigonometry_problem_l773_773031


namespace train_length_l773_773103

noncomputable def speed_kmph : ℝ := 80
noncomputable def time_seconds : ℝ := 5
noncomputable def speed_mps : ℝ := (speed_kmph * 1000) / 3600
noncomputable def distance : ℝ := speed_mps * time_seconds

theorem train_length :
  distance ≈ 111.1 :=
by sorry

end train_length_l773_773103


namespace terminating_decimal_count_l773_773426

theorem terminating_decimal_count :
  (finset.card (finset.filter (λ n : ℕ, n % 3 = 0) (finset.range 301))) = 100 :=
sorry

end terminating_decimal_count_l773_773426


namespace problem_statement_l773_773445

variable {x : ℝ}

def p : Prop := (x - 1) * (x - 2) ≤ 0
def q : Prop := Real.log (x + 1) / Real.log 2 ≥ 1

theorem problem_statement : (∀ x, p → q) ∧ ¬ (∀ x, q → p) := by
  sorry

end problem_statement_l773_773445


namespace total_legs_of_collection_l773_773928

theorem total_legs_of_collection (spiders ants : ℕ) (legs_per_spider legs_per_ant : ℕ)
  (h_spiders : spiders = 8) (h_ants : ants = 12)
  (h_legs_per_spider : legs_per_spider = 8) (h_legs_per_ant : legs_per_ant = 6) :
  (spiders * legs_per_spider + ants * legs_per_ant) = 136 :=
by
  sorry

end total_legs_of_collection_l773_773928


namespace round_to_nearest_thousandth_l773_773363

theorem round_to_nearest_thousandth (x : ℝ) (h₁ : x = 0.06398) (h₂ : ∃ d, x = d * 0.001 ∧ d % 1 = 0.06398) : round(x * 1000) / 1000 = 0.064 := 
by
  sorry

end round_to_nearest_thousandth_l773_773363


namespace probabilistic_dice_problem_l773_773373

noncomputable def fair_die : ℙ (fin 6) :=
  λ x, if x.1 = 5 then (1 / 6 : ℚ) else (1 / 6 : ℚ)

noncomputable def biased_die : ℙ (fin 6) :=
  λ x, if x.1 = 5 then (1 / 2 : ℚ) else (1 / 10 : ℚ)

theorem probabilistic_dice_problem :
  let p := 325
  let q := 656
  let prob_six_fifth_roll_given_two_sixes_in_first_four := (p : ℚ) / (q : ℚ)
  in p + q = 981 :=
begin
  -- Proof goes here.
  sorry,
end

end probabilistic_dice_problem_l773_773373


namespace must_divisor_of_a_l773_773553

-- The statement
theorem must_divisor_of_a (a b c d : ℕ) (h1 : Nat.gcd a b = 18)
    (h2 : Nat.gcd b c = 45) (h3 : Nat.gcd c d = 60) (h4 : 90 < Nat.gcd d a ∧ Nat.gcd d a < 120) : 
    5 ∣ a := 
sorry

end must_divisor_of_a_l773_773553


namespace total_students_count_l773_773122

-- Definitions for the conditions
def ratio_girls_to_boys (g b : ℕ) : Prop := g * 4 = b * 3
def boys_count : ℕ := 28

-- Theorem to prove the total number of students
theorem total_students_count {g : ℕ} (h : ratio_girls_to_boys g boys_count) : g + boys_count = 49 :=
sorry

end total_students_count_l773_773122


namespace determine_operations_and_calculate_l773_773781

theorem determine_operations_and_calculate {
  -- Conditions: two operations and given results table
  assume op1, op2 : Rat → Rat → Rat,
  op1 = (λ x y, x * y),
  op2 = (λ x y, x + y),
  assume a b c d e f g h : Rat,
  a = 3 + 1 / 11,
  b = 3 + 15 / 17,
  c = 2,
  d = 2,
  e = 13,
  f = 5,
  g = 2,
  h = 1,
  h1 : op2 (op1 a b) d = e,
  h2 : op2 (op1 c d) h = f
  : 
  let A = op2 (op1 g (3 + 15 / 17)) 1,
      B = op2 (op1 (3 + 1 / 11) g) 1
  in 
  A + B = 2982 / 187 := 
  sorry⟩
endmodule

end determine_operations_and_calculate_l773_773781


namespace cos_pi_minus_alpha_l773_773454

open Real

variable (α : ℝ)

theorem cos_pi_minus_alpha (h1 : 0 < α ∧ α < π / 2) (h2 : sin α = 4 / 5) : cos (π - α) = -3 / 5 := by
  sorry

end cos_pi_minus_alpha_l773_773454


namespace ellipse_and_triangle_l773_773780

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
(a > b ∧ b > 0 ∧ a = 2 * √3 ∧ b = 2 ∧ (∃ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1))

noncomputable def triangle_area (P A B : ℝ × ℝ) : ℝ :=
let (x1, y1) := A in let (x2, y2) := B in
let |AB| := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) in
let d := |(-3 - x1) * (y2 - y1) - (2 - y1) * (x2 - x1)| / real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) in
1 / 2 * |AB| * d

theorem ellipse_and_triangle :
(∀ a b : ℝ, ellipse_equation a b) →
triangle_area (-3, 2) (-3, -1) (0, 2) = 9 / 2 :=
by
  sorry

end ellipse_and_triangle_l773_773780


namespace infinite_lines_through_point_P_60_degree_l773_773853

theorem infinite_lines_through_point_P_60_degree (L : Line) (P : Point) (h : P ∉ L) : 
  ∃ᶠ M : Line, P ∈ M ∧ ∠(M, L) = 60 :=
sorry

end infinite_lines_through_point_P_60_degree_l773_773853


namespace find_precy_age_l773_773513

-- Defining the given conditions as Lean definitions
def alex_current_age : ℕ := 15
def alex_age_in_3_years : ℕ := alex_current_age + 3
def alex_age_a_year_ago : ℕ := alex_current_age - 1
axiom precy_current_age : ℕ
axiom in_3_years : alex_age_in_3_years = 3 * (precy_current_age + 3)
axiom a_year_ago : alex_age_a_year_ago = 7 * (precy_current_age - 1)

-- Stating the equivalent proof problem
theorem find_precy_age : precy_current_age = 3 :=
by
  sorry

end find_precy_age_l773_773513


namespace greatest_exponent_three_factorial_l773_773852

theorem greatest_exponent_three_factorial (m : ℕ) : 
  (∃ k : ℕ, m = 3^k ∧ ∃ (e : ℕ), (m ∣ nat.factorial 22) ∧ e = k ∧ (∀ k' : ℕ, m = 3^k' → m ∣ nat.factorial 22 → k' ≤ e)) → 
  m = 3^9 := 
by sorry

end greatest_exponent_three_factorial_l773_773852


namespace find_beta_l773_773090

variable (α β : ℝ)

theorem find_beta 
  (h1 : Real.cos α = 1 / 7)
  (h2 : Real.cos (α + β) = -11 / 14)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : Real.pi / 2 < α + β ∧ α + β < Real.pi) : β = Real.pi / 3 := sorry

end find_beta_l773_773090


namespace class_groups_l773_773514

open Nat

def combinations (n k : ℕ) : ℕ :=
  n.choose k

theorem class_groups (boys girls : ℕ) (group_size : ℕ) :
  boys = 9 → girls = 12 → group_size = 3 →
  (combinations boys 1 * combinations girls 2) + (combinations boys 2 * combinations girls 1) = 1026 :=
by
  intros
  sorry

end class_groups_l773_773514


namespace riding_owners_ratio_l773_773326

theorem riding_owners_ratio :
  ∃ (R W : ℕ), (R + W = 16) ∧ (4 * R + 6 * W = 80) ∧ (R : ℚ) / 16 = 1/2 :=
by
  sorry

end riding_owners_ratio_l773_773326


namespace magnitude_a_add_b_l773_773838

-- Define the vectors a and b
def vector_a (x : ℝ) := (x, 3 : ℝ)
def vector_b := (2, -2 : ℝ)

-- Define the dot product function for 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define vector addition for 2D vectors
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

-- Define the magnitude of a 2D vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

-- The theorem to be proved
theorem magnitude_a_add_b {x : ℝ} (h : dot_product (vector_a x) vector_b = 0) :
  magnitude (vector_add (vector_a x) vector_b) = real.sqrt 26 :=
by
  sorry

end magnitude_a_add_b_l773_773838


namespace sum_of_two_numbers_l773_773621

variables {x y : ℝ}

theorem sum_of_two_numbers (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 22 :=
sorry

end sum_of_two_numbers_l773_773621


namespace least_months_exceed_tripled_borrowed_l773_773893

theorem least_months_exceed_tripled_borrowed :
  ∃ t : ℕ, (1.03 : ℝ)^t > 3 ∧ ∀ n < t, (1.03 : ℝ)^n ≤ 3 :=
sorry

end least_months_exceed_tripled_borrowed_l773_773893


namespace regular_price_correct_l773_773701

noncomputable def regular_price_of_one_tire (x : ℝ) : Prop :=
  3 * x + 5 - 10 = 302

theorem regular_price_correct (x : ℝ) : regular_price_of_one_tire x → x = 307 / 3 := by
  intro h
  sorry

end regular_price_correct_l773_773701


namespace range_k_domain_f_l773_773856

theorem range_k_domain_f :
  (∀ x : ℝ, x^2 - 6*k*x + k + 8 ≥ 0) ↔ (-8/9 ≤ k ∧ k ≤ 1) :=
sorry

end range_k_domain_f_l773_773856


namespace houston_bus_passes_six_dallas_buses_l773_773366

-- Define the conditions and parameters in the problem
def buses_passed : ℕ :=
  let travel_time := 6
  let dallas_buses := list.range (12 + 1)  -- Buses leaving Dallas every hour from 6:00 AM to 12:00 PM (inclusive)
  let houston_bus_departure := 12         -- Houston bus leaves at 12:00 PM
  let intersections := dallas_buses.filter (λ t, t + travel_time >= houston_bus_departure && t < houston_bus_departure)
  intersections.length

-- The theorem we aim to prove, based on the given conditions
theorem houston_bus_passes_six_dallas_buses : buses_passed = 6 :=
sorry -- Proof is omitted

end houston_bus_passes_six_dallas_buses_l773_773366


namespace no_primes_between_factorial_minus_m_and_factorial_minus_one_l773_773761

theorem no_primes_between_factorial_minus_m_and_factorial_minus_one (m : ℤ) (h : m > 2) : 
  ∀ n, m! - m ≤ n ∧ n < m! - 1 → ¬ prime n :=
by
  sorry

end no_primes_between_factorial_minus_m_and_factorial_minus_one_l773_773761


namespace ab_range_l773_773107

theorem ab_range (a b : ℝ) : (a + b = 1/2) → ab ≤ 1/16 :=
by
  sorry

end ab_range_l773_773107


namespace general_term_sum_formula_T_l773_773777

noncomputable def a_sequence (n : ℕ) : ℕ :=
  if n = 0 then 1 else 3 ^ (n - 1)

def c (n : ℕ) := real.log 3 (a_sequence (2 * n))

def b (n : ℕ) := 1 / (c n * c (n + 1))

noncomputable def T (n : ℕ) : ℝ :=
  (1 / 4) * (4 / 3 - 1 / (2 * n + 1) - 1 / (2 * n + 3))

theorem general_term (n : ℕ) : a_sequence n = 3 ^ (n - 1) := sorry

theorem sum_formula_T (n : ℕ) : 
  T n = (1 / 4) * (4 / 3 - 1 / (2 * n + 1) - 1 / (2 * n + 3)) := sorry

end general_term_sum_formula_T_l773_773777


namespace smallest_sum_minimum_l773_773784

noncomputable def smallest_sum (x y : ℕ) : ℕ :=
if h₁ : x ≠ y ∧ (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) then x + y else 0

theorem smallest_sum_minimum (x y : ℕ) (h₁ : x ≠ y) (h₂ : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) :
  smallest_sum x y = 96 := sorry

end smallest_sum_minimum_l773_773784


namespace triangle_area_quadrilateral_area_polygon_area_l773_773951

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

-- Part (a): Triangle area
theorem triangle_area (m : ℕ) : 
  let A1 := (fib (m+1), fib (m+2)),
      A2 := (fib (m+3), fib (m+4)),
      A3 := (fib (m+5), fib (m+6)) 
  in 1/2 * abs (A1.1*A2.2 + A2.1*A3.2 + A3.1*A1.2 - A1.2*A2.1 - A2.2*A3.1 - A3.2*A1.1) = 0.5 := 
sorry

-- Part (b): Quadrilateral area
theorem quadrilateral_area (m : ℕ) : 
  let A1 := (fib (m+1), fib (m+2)),
      A2 := (fib (m+3), fib (m+4)),
      A3 := (fib (m+5), fib (m+6)),
      A4 := (fib (m+7), fib (m+8)) 
  in 1/2 * abs (A1.1*A2.2 + A2.1*A3.2 + A3.1*A4.2 + A4.1*A1.2 - A1.2*A2.1 - A2.2*A3.1 - A3.2*A4.1 - A4.2*A1.1) = 2.5 := 
sorry

-- Part (c): Polygon area
theorem polygon_area (n m: ℕ) (hn : n ≥ 3) : 
  let vertices := λ i, (fib (m + 2*i - 1), fib (m + 2*i)) in
  1/2 * abs (∑ i in finset.range n, vertices i.1 * vertices ((i+1)%n).2 - vertices i.2 * vertices ((i+1)%n).1) = abs (fib (2*n - 2) - n + 1) / 2 := 
sorry

end triangle_area_quadrilateral_area_polygon_area_l773_773951


namespace min_x_plus_y_l773_773786

theorem min_x_plus_y (x y : ℕ) (hxy : x ≠ y) (h : (1/x : ℝ) + 1/y = 1/24) : x + y = 98 :=
sorry

end min_x_plus_y_l773_773786


namespace find_S_n_find_range_of_d_l773_773778

variables 
  (a_n : ℕ → ℝ) 
  (S_n : ℕ → ℝ) 
  (d : ℝ) 
  (c_n : ℕ → ℝ)
  [h1 : ∀ n, a_n n = -1 + (n-1) * d]
  (h2 : d > 1)

-- (Ⅰ)
theorem find_S_n 
  (h3 : S_n 4 - 2 * (a_n 2) * (a_n 3) + 6 = 0) : 
  ∀ n, S_n n = (3 * n^2 - 5 * n) / 2 :=
  sorry

-- (Ⅱ)
theorem find_range_of_d 
  (h4 : ∀ n, (a_n n + c_n n) * (a_n (n+2) + 15 * c_n n) = (a_n (n+1) + 4 * c_n n)^2): 
  1 < d ∧ d ≤ 2 :=
  sorry

end find_S_n_find_range_of_d_l773_773778


namespace find_A_and_B_l773_773388

theorem find_A_and_B : 
  ∃ A B : ℝ, 
    (A = 6.5 ∧ B = 0.5) ∧
    (∀ x : ℝ, (8 * x - 17) / ((3 * x + 5) * (x - 3)) = A / (3 * x + 5) + B / (x - 3)) :=
by
  sorry

end find_A_and_B_l773_773388


namespace problem_statement_l773_773486

variable (a b c : ℝ)

theorem problem_statement 
  (h1 : ab / (a + b) = 1 / 3)
  (h2 : bc / (b + c) = 1 / 4)
  (h3 : ca / (c + a) = 1 / 5) :
  abc / (ab + bc + ca) = 1 / 6 := 
sorry

end problem_statement_l773_773486


namespace find_phi_l773_773824

theorem find_phi (φ : ℝ) (h₀ : 0 ≤ φ ∧ φ ≤ π)
  (h₁ : sin (2 * (π / 3) + φ) = cos (π / 3)) : φ = π / 6 :=
begin
  sorry
end

end find_phi_l773_773824


namespace find_n_l773_773563

theorem find_n :
  ∃ (n : ℕ+), (∃ (x0 : ℝ), x0 ∈ set.Ioo n (n + 1) ∧ x0 ^ 3 = (1 / 2) ^ (x0 - 2)) ∧ n = 1 :=
begin
  sorry
end

end find_n_l773_773563


namespace probability_of_selecting_at_least_one_female_l773_773013

theorem probability_of_selecting_at_least_one_female :
  let total_students := 5
  let total_ways := nat.choose total_students 2
  let male_students := 3
  let female_students := 2
  let ways_2_males := nat.choose male_students 2
  (1 - ((ways_2_males : ℚ) / (total_ways : ℚ))) = 7 / 10 := by
  sorry

end probability_of_selecting_at_least_one_female_l773_773013


namespace convex_polygons_count_l773_773750

theorem convex_polygons_count (n : ℕ) (h : n = 15) :
  let total_subsets := 2^n,
      zero_point_subsets := nat.choose n 0,
      one_point_subsets := nat.choose n 1,
      two_point_subsets := nat.choose n 2,
      three_point_subsets := nat.choose n 3,
      subsets_with_less_than_four := zero_point_subsets + one_point_subsets + two_point_subsets + three_point_subsets,
      distinct_polygons := total_subsets - subsets_with_less_than_four
  in distinct_polygons = 32192 :=
by
  -- proof here
  sorry

end convex_polygons_count_l773_773750


namespace exists_yk_bound_l773_773455

variable (y : ℕ → ℝ) (n : ℕ)

noncomputable def sum_up_to (f : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in (Finset.range (n + 1)), f i

theorem exists_yk_bound
  (h1 : sum_up_to (fun i => (y i) ^ 3) n = 0) :
  ∃ k ∈ (Finset.range (n + 1)), |y k| ≥ real.sqrt 27 / 2 ^ ((1 : ℝ) / 4) * |sum_up_to y n| / n :=
by
  sorry

end exists_yk_bound_l773_773455


namespace max_value_of_f_l773_773413

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x) + Real.cos (Real.sin x)

theorem max_value_of_f : ∀ x : ℝ, f x ≤ Real.sin 1 + 1 ∧ (f 0 = Real.sin 1 + 1) :=
by
  intro x
  sorry

end max_value_of_f_l773_773413


namespace race_outcomes_210_l773_773708

-- Define the participants
def participants : List String := ["Abe", "Bobby", "Charles", "Devin", "Edwin", "Fern", "Grace"]

-- The question is to prove the number of different 1st-2nd-3rd place outcomes is 210.
theorem race_outcomes_210 (h : participants.length = 7) : (7 * 6 * 5 = 210) :=
  by sorry

end race_outcomes_210_l773_773708


namespace three_similar_1995_digit_numbers_l773_773728

noncomputable def similar_numbers_exist : Prop := 
  ∃ (A B C: Nat), 
    (A.to_digits = B.to_digits.perm) ∧ -- A and B are similar (permuted versions of each other)
    (C.to_digits.nodup ∧ C.to_digits.all (λ x, x ∈ [4, 5, 9])) ∧ -- C and its digits are within [4, 5, 9]
    (A.to_digits.countp (λ x, x = 0) = 0) ∧ -- No zero in A's digits
    (B.to_digits.countp (λ x, x = 0) = 0) ∧ -- No zero in B's digits
    (C.to_digits.countp (λ x, x = 0) = 0) ∧ -- No zero in C's digits
    (A + B = C) ∧ -- A and B sum to C
    (A.to_digits.length = 1995) ∧ -- A has 1995 digits
    (B.to_digits.length = 1995) ∧ -- B has 1995 digits
    (C.to_digits.length = 1995) -- C has 1995 digits

theorem three_similar_1995_digit_numbers : similar_numbers_exist := 
  sorry

end three_similar_1995_digit_numbers_l773_773728


namespace exists_k_homomorphism_l773_773436

variable {G : Type*} [Group G]
variable (φ : G → G)
variable (h : ∀ {a b c d e f : G}, a * b * c = 1 → d * e * f = 1 → φ a * φ b * φ c = φ d * φ e * φ f)

theorem exists_k_homomorphism : ∃ k : G, ∀ x y : G, k * φ (x * y) = k * φ x * k * φ y :=
by
  let k := φ 1⁻¹
  use k
  intro x y
  -- Here you would complete the proof
  sorry

end exists_k_homomorphism_l773_773436


namespace brock_buys_7_cookies_l773_773925

variable (cookies_total : ℕ)
variable (sold_to_stone : ℕ)
variable (left_after_sale : ℕ)
variable (cookies_brock_buys : ℕ)
variable (cookies_katy_buys : ℕ)

theorem brock_buys_7_cookies
  (h1 : cookies_total = 5 * 12)
  (h2 : sold_to_stone = 2 * 12)
  (h3 : left_after_sale = 15)
  (h4 : cookies_total - sold_to_stone - (cookies_brock_buys + cookies_katy_buys) = left_after_sale)
  (h5 : cookies_katy_buys = 2 * cookies_brock_buys) :
  cookies_brock_buys = 7 :=
by
  -- Proof is skipped
  sorry

end brock_buys_7_cookies_l773_773925


namespace find_edge_DA_l773_773885

noncomputable def pyramid_edge (AB BC CD : ℝ) (perpendicular : Prop (line_through_points AC BD = ⊥)) : ℝ :=
  1  -- The correct answer by given solution

theorem find_edge_DA 
  (AB BC CD : ℝ) 
  (AB_eq : AB = 7) 
  (BC_eq : BC = 8) 
  (CD_eq : CD = 4)
  (perpendicular : line_through_points AC BD = ⊥)
  : DA = pyramid_edge AB BC CD perpendicular := 
by 
  rw [AB_eq, BC_eq, CD_eq]
  exact eq.refl (1)


end find_edge_DA_l773_773885


namespace sum_of_reciprocal_roots_of_polynomial_l773_773380

theorem sum_of_reciprocal_roots_of_polynomial (b : ℕ → ℂ) (h : ∀ n, b n ∈ (Polroots (X^1009 + X^1008 + X^1007 + X^1006 + X^1005 + X^1004 + X^1003 + X^1002 + X - (671 : ℂ)))) :
  (∑ n in finset.range 1009, (1 : ℂ) / (1 - b n)) = 1515 :=
sorry

end sum_of_reciprocal_roots_of_polynomial_l773_773380


namespace labor_union_tree_equation_l773_773680

theorem labor_union_tree_equation (x : ℕ) : 2 * x + 21 = 3 * x - 24 := 
sorry

end labor_union_tree_equation_l773_773680


namespace david_total_cost_correct_l773_773676

def base_cost : ℝ := 25
def included_texts : ℕ := 200
def included_talk_time_hours : ℝ := 40
def extra_texts_cost_cents : ℝ := 0.03
def extra_talk_time_cost_cents_per_minute : ℝ := 0.15
def included_data_gb : ℝ := 3
def extra_data_cost : ℝ := 10
def texts_sent_in_february : ℕ := 250
def talk_time_in_february_hours : ℝ := 42
def data_used_in_february_gb : ℝ := 4

def extra_texts_cost : ℝ :=
  if texts_sent_in_february > included_texts then
    (texts_sent_in_february - included_texts) * extra_texts_cost_cents
  else
    0

def extra_talk_time_cost : ℝ :=
  if talk_time_in_february_hours > included_talk_time_hours then
    (talk_time_in_february_hours - included_talk_time_hours) * 60 * extra_talk_time_cost_cents_per_minute
  else
    0

def extra_data_cost : ℝ :=
  if data_used_in_february_gb > included_data_gb then
    (data_used_in_february_gb - included_data_gb) * extra_data_cost
  else
    0

def total_cost : ℝ :=
  base_cost + extra_texts_cost + extra_talk_time_cost + extra_data_cost

theorem david_total_cost_correct :
  total_cost = 54.50 := by
  sorry

end david_total_cost_correct_l773_773676


namespace prism_faces_and_vertices_l773_773028

def prism_properties (edges : ℕ) (x : ℕ) (y : ℕ) : Prop :=
  -- Defining the property of the prism having 21 edges and calculating x and y
  edges = 21 ∧ x = 9 ∧ y = 14

theorem prism_faces_and_vertices (edges x y : ℕ) (h : prism_properties edges x y) : 3 * x - 2 * y = -1 :=
by
  -- By the given conditions, proving the value of 3x - 2y
  sorry

end prism_faces_and_vertices_l773_773028


namespace find_m_for_equal_roots_l773_773884

def roots_equal (x m : ℝ) (m_ne_two : m ≠ 2) (x_ne_two : x ≠ 2) : Prop :=
  (x * (x - 2) - (m + 2)) / ((x - 2) * (m - 2)) = x / m ∧
  (x^2 - x - (m^2 + 2 * m + 2) = 0 ∨ (m^2 + 2 * m + 2 = 0))

theorem find_m_for_equal_roots : ∃ m : ℝ, roots_equal x m (by norm_num) (by norm_num) → m = -3 / 2 :=
sorry

end find_m_for_equal_roots_l773_773884


namespace find_a_b_value_l773_773036

theorem find_a_b_value (a b : ℝ) (h₁ : {1, a, b / a} = {0, a^2, a + b}) : a ^ 2016 + b ^ 2016 = 1 :=
by
  sorry

end find_a_b_value_l773_773036


namespace problem_proof_l773_773741

noncomputable def binomial (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.choose n k else 0

noncomputable def probability_ratio_pq : ℕ :=
let p := binomial 10 2 * binomial 30 2 * binomial 28 2
let q := binomial 30 3 * binomial 27 3 * binomial 24 3 * binomial 21 3 * binomial 18 3 * binomial 15 3 * binomial 12 3 * binomial 9 3 * binomial 6 3 * binomial 3 3
p / (q / (binomial 30 3 * binomial 27 3 * binomial 24 3 * binomial 21 3 * binomial 18 3 * binomial 15 3 * binomial 12 3 * binomial 9 3 * binomial 6 3 * binomial 3 3))

theorem problem_proof :
  probability_ratio_pq = 7371 :=
sorry

end problem_proof_l773_773741


namespace angle_equality_of_triangle_l773_773355

theorem angle_equality_of_triangle (A B C P Q : Point) (h₁ : ∠PBA = ∠QBC) (h₂ : ∠PCA = ∠QCB) : ∠PAB = ∠QAC := 
sorry

end angle_equality_of_triangle_l773_773355


namespace reciprocal_2023_l773_773985

def reciprocal (x : ℕ) := 1 / x

theorem reciprocal_2023 : reciprocal 2023 = 1 / 2023 :=
by
  sorry

end reciprocal_2023_l773_773985


namespace find_b_squared_l773_773964

theorem find_b_squared
    (b : ℝ)
    (c_ellipse c_hyperbola a_ellipse a2_hyperbola b2_hyperbola : ℝ)
    (h1: a_ellipse^2 = 25)
    (h2 : b2_hyperbola = 9 / 4)
    (h3 : a2_hyperbola = 4)
    (h4 : c_hyperbola = Real.sqrt (a2_hyperbola + b2_hyperbola))
    (h5 : c_ellipse = c_hyperbola)
    (h6 : b^2 = a_ellipse^2 - c_ellipse^2)
: b^2 = 75 / 4 :=
sorry

end find_b_squared_l773_773964


namespace evaluate_expression_l773_773747

theorem evaluate_expression : 2 + 3 / (4 + 5 / 6) = 76 / 29 := by
  sorry

end evaluate_expression_l773_773747


namespace imaginary_unit_equation_l773_773922

theorem imaginary_unit_equation (a : ℝ) : 
  (1 - (complex.I : ℂ) = (2 + a * complex.I) / (1 + complex.I)) → a = 0 := by
s.t

#eval sorry

end imaginary_unit_equation_l773_773922


namespace trip_duration_l773_773265

-- Definitions of distances and time
variables (d1 d2 T : ℝ)

-- Conditions
def tom_harry_initial_travel : ℝ := d1 / 30
def harry_walk : ℝ := (150 - d1) / 4
def harry_total_time : ℝ := tom_harry_initial_travel + harry_walk

def tom_backtrack_pickup : ℝ := d2 / 30
def tom_total_time : ℝ := tom_harry_initial_travel + tom_backtrack_pickup + (150 - (d1 - d2)) / 30

def dick_walk : ℝ := (d1 - d2) / 4
def dick_ride : ℝ := (150 - (d1 - d2)) / 30
def dick_total_time : ℝ := dick_walk + dick_ride

-- Statement to prove
theorem trip_duration (h1: harry_total_time = T) (h2: tom_total_time = T) (h3: dick_total_time = T) : 
  T = 18 :=
sorry

end trip_duration_l773_773265


namespace trigonometric_identities_l773_773456

noncomputable def tan (θ : ℝ) : ℝ := Real.tan θ
noncomputable def sin (θ : ℝ) : ℝ := Real.sin θ
noncomputable def cos (θ : ℝ) : ℝ := Real.cos θ

theorem trigonometric_identities (θ : ℝ) (h_tan : tan θ = 2) (h_identity : sin θ ^ 2 + cos θ ^ 2 = 1) :
    ((sin θ = 2 * Real.sqrt 5 / 5 ∧ cos θ = Real.sqrt 5 / 5) ∨ (sin θ = -2 * Real.sqrt 5 / 5 ∧ cos θ = -Real.sqrt 5 / 5)) ∧
    ((4 * sin θ - 3 * cos θ) / (6 * cos θ + 2 * sin θ) = 1 / 2) :=
by
  sorry

end trigonometric_identities_l773_773456


namespace part_one_min_f_value_part_two_range_a_l773_773050

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |x + a|

theorem part_one_min_f_value (x : ℝ) (a : ℝ) (h : a = 1) : f x a ≥ (3/2) :=
  sorry

theorem part_two_range_a (a : ℝ) : (11/2 < a) ∧ (a < 4.5) :=
  sorry

end part_one_min_f_value_part_two_range_a_l773_773050


namespace product_not_perfect_square_l773_773934

theorem product_not_perfect_square :
  ¬ ∃ n : ℕ, n^2 = (2021^1004) * (6^3) :=
by
  sorry

end product_not_perfect_square_l773_773934


namespace quadratic_vertex_correct_l773_773431

theorem quadratic_vertex_correct:
  ∀ x : ℝ, (let y := (x - 1)^2 + 3 in y) = (x - 1)^2 + 3 →
  ∃ h k : ℝ, h = 1 ∧ k = 3 ∧ y = (x - h)^2 + k :=
by 
  sorry

end quadratic_vertex_correct_l773_773431


namespace permutations_of_four_digits_l773_773140

theorem permutations_of_four_digits : 
  ∃ (s : Finset (Fin 4)) (h : s = {0, 1, 2, 3}), (s.card.factorial = 24) :=
begin
  let s := {0, 1, 2, 3},
  have hs : s.card = 4,
  {
    exact Finset.card_fin 4,
  },
  use s,
  split,
  exact rfl,
  simp,
  rw hs,
  exact dec_trivial,
end

end permutations_of_four_digits_l773_773140


namespace fx_fixed_point_l773_773966

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  log a (1 - x) + 5

theorem fx_fixed_point (a : ℝ) (ha_pos : 0 < a) (ha_ne : a ≠ 1) :
  f a 0 = 5 :=
by
  have : log a 1 = 0 := log_base_one ha_pos ha_ne
  calc
    f a 0 = log a (1 - 0) + 5 : rfl
        ... = log a 1 + 5       : by rw [sub_zero]
        ... = 0 + 5             : by rw [this]
        ... = 5                 : by rw [zero_add]

end fx_fixed_point_l773_773966


namespace find_f_m_l773_773476

noncomputable def f (x : ℝ) := x^5 + Real.tan x - 3

theorem find_f_m (m : ℝ) (h : f (-m) = -2) : f m = -4 :=
sorry

end find_f_m_l773_773476


namespace gcd_pow_sub_one_l773_773273

theorem gcd_pow_sub_one (a b : ℕ) 
  (h_a : a = 2^2004 - 1) 
  (h_b : b = 2^1995 - 1) : 
  Int.gcd a b = 511 :=
by
  sorry

end gcd_pow_sub_one_l773_773273


namespace sum_of_m_and_n_l773_773073

noncomputable theory

open Complex

theorem sum_of_m_and_n (m n : ℂ) (h1 : m ≠ n ∧ m ≠ 0 ∧ n ≠ 0) (h2 : {m, n} = {m^2, n^2}) : 
  m + n = -1 := 
sorry

end sum_of_m_and_n_l773_773073


namespace probability_union_intersections_l773_773105

def prob_a : ℚ := 2/5
def prob_b : ℚ := 2/5
def prob_c : ℚ := 1/5
def prob_d : ℚ := 1/3

def independent (p : ℚ) (q : ℚ) : Bool :=
  true  -- This is a placeholder for the independent condition of events

theorem probability_union_intersections :
  independent prob_a prob_b → 
  independent prob_c prob_d →
  let p_ab := prob_a * prob_b in
  let p_cd := prob_c * prob_d in
  let p_ab_cd := p_ab * p_cd in
  p_ab + p_cd - p_ab_cd = 27/125 := 
by
  intro h_ab h_cd 
  let p_ab := prob_a * prob_b
  let p_cd := prob_c * prob_d
  let p_ab_cd := p_ab * p_cd
  have h1 : p_ab = 4 / 25 := by sorry
  have h2 : p_cd = 1 / 15 := by sorry
  have h3 : p_ab_cd = 4 / 375 := by sorry
  have h4 : p_ab + p_cd - p_ab_cd = 27 / 125 := by sorry
  exact h4

end probability_union_intersections_l773_773105


namespace complex_expression_value_l773_773037

noncomputable def value_of_expression (a : ℝ) : ℂ := (a + complex.I ^ 2016) / (1 + complex.I)

theorem complex_expression_value :
  ∀ (a : ℝ),
  (a ^ 2 - 1 = 0) →
  value_of_expression a = 1 - complex.I := by
  intros a ha
  sorry

end complex_expression_value_l773_773037


namespace solve_equation_l773_773946

theorem solve_equation : ∀ x : ℝ, 4 * x + 4 - x - 2 * x + 2 - 2 - x + 2 + 6 = 0 → x = 0 :=
by 
  intro x h
  sorry

end solve_equation_l773_773946


namespace star_property_l773_773429

-- Define the custom operation
def star (a b : ℝ) : ℝ := a^2 - b

-- Define the property to prove
theorem star_property (x y : ℝ) : star (x - y) (x + y) = x^2 - x - 2 * x * y + y^2 - y :=
by sorry

end star_property_l773_773429


namespace problem_l773_773112

def rel1 (a : ℕ) : Prop := a = 1
def rel2 (b : ℕ) : Prop := b ≠ 1
def rel3 (c : ℕ) : Prop := c = 2
def rel4 (d : ℕ) : Prop := d ≠ 4

def correct_combination (a b c d : ℕ) : Prop :=
  {a, b, c, d} = {1, 2, 3, 4} ∧
  ((rel1 a ∧ ¬rel2 b ∧ ¬rel3 c ∧ ¬rel4 d) ∨ 
   (¬rel1 a ∧ rel2 b ∧ ¬rel3 c ∧ ¬rel4 d) ∨ 
   (¬rel1 a ∧ ¬rel2 b ∧ rel3 c ∧ ¬rel4 d) ∨ 
   (¬rel1 a ∧ ¬rel2 b ∧ ¬rel3 c ∧ rel4 d))

theorem problem (a b c d : ℕ) (h : correct_combination a b c d) : 1000 * a + 100 * b + 10 * c + d = 1342 :=
sorry

end problem_l773_773112


namespace circumcenter_AEF_lies_on_AB_l773_773669

open EuclideanGeometry

variables {P Q : Type} [EuclideanGeometry P] [EuclideanGeometry Q]

theorem circumcenter_AEF_lies_on_AB
  (A B C H E F : P)
  (hABC : ∠A B C < 90 ∧ ∠B A C < 90 ∧ ∠C A B < 90 ∧ side_eq B A C B)
  (hH_orthocenter : orthocenter A B C H)
  (hE_reflection : reflection_line (altitude A H) C E)
  (hF_intersection : between E H F  ∧ between A C F) :
  on_line (circumcenter A E F) (line A B) := sorry

end circumcenter_AEF_lies_on_AB_l773_773669


namespace factorize_expression_l773_773749

theorem factorize_expression (x : ℝ) : x * (x - 3) - x + 3 = (x - 1) * (x - 3) :=
by
  sorry

end factorize_expression_l773_773749


namespace line_through_points_l773_773484

theorem line_through_points 
  (A1 B1 A2 B2 : ℝ) 
  (h₁ : A1 * -7 + B1 * 9 = 1) 
  (h₂ : A2 * -7 + B2 * 9 = 1) :
  ∃ (k : ℝ), ∀ (x y : ℝ), (A1, B1) ≠ (A2, B2) → y = k * x + (B1 - k * A1) → -7 * x + 9 * y = 1 :=
sorry

end line_through_points_l773_773484


namespace candle_height_half_time_l773_773702
open Nat

theorem candle_height_half_time :
  let height := 100
  let burn_time (k : ℕ) := 5 * k
  let total_burn_time := ∑ k in range (height + 1), burn_time k
  let half_burn_time := total_burn_time / 2
  let m := Nat.floor (sqrt (2 * half_burn_time / 5))

  total_burn_time = 25250 →
  half_burn_time = 12625 →
  (∑ k in range (m + 1), burn_time k ≤ half_burn_time ∧
  half_burn_time < ∑ k in range (m + 2), burn_time k) →
  m = 73 →
  (height - m) = 27 
:= sorry

end candle_height_half_time_l773_773702


namespace comparison_of_constants_l773_773444

theorem comparison_of_constants (a b c : ℝ) (h₁ : a = Real.log 10) (h₂ : b = Real.sqrt Real.exp 1) (h₃ : c = 2) :
  a > c ∧ c > b :=
by
  sorry

end comparison_of_constants_l773_773444


namespace range_of_a_0_l773_773298

variable (a_0 : ℝ)
variable (h : a_0 = 9.3 ∨ (a_0 > 9.3 - 0.5 ∧ a_0 < 9.3 + 0.5))

theorem range_of_a_0 : 8.8 ≤ a_0 ∧ a_0 ≤ 9.8 :=
by {
  cases h with heq hineq,
  {
    rw heq,
    exact ⟨by norm_num, by norm_num⟩,
  },
  {
    exact ⟨by linarith, by linarith⟩,
  }
}

end range_of_a_0_l773_773298


namespace reflection_twice_is_identity_l773_773160

-- Define the reflection matrix R over the vector (1, 2)
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  -- Note: The specific definition of the reflection matrix over (1, 2) is skipped as we only need the final proof statement.
  sorry

-- Assign the reflection matrix R to variable R
def R := reflection_matrix

-- Prove that R^2 = I
theorem reflection_twice_is_identity : R * R = 1 := by
  sorry

end reflection_twice_is_identity_l773_773160


namespace bacteria_growth_l773_773599

theorem bacteria_growth (d : ℕ) (t : ℕ) (initial final : ℕ) 
  (h_doubling : d = 4) 
  (h_initial : initial = 500) 
  (h_final : final = 32000) 
  (h_ratio : final / initial = 2^6) :
  t = d * 6 → t = 24 :=
by
  sorry

end bacteria_growth_l773_773599


namespace number_of_distinct_prime_factors_l773_773153

-- Given: A is the product of the divisors of 60.
-- Prove: The number of distinct prime factors of A is 3.

theorem number_of_distinct_prime_factors (A : ℕ) 
  (h1 : A = ∏ d in (finset.filter (λ d, d ∣ 60) (finset.range (60 + 1))), d) :
  (factors_multiset A).to_finset.card = 3 := 
sorry

end number_of_distinct_prime_factors_l773_773153


namespace min_value_of_S_l773_773169

variable (x : ℝ)
def S (x : ℝ) : ℝ := (x - 10)^2 + (x + 5)^2

theorem min_value_of_S : ∀ x : ℝ, S x ≥ 112.5 :=
by
  sorry

end min_value_of_S_l773_773169


namespace can_determine_own_grades_l773_773668

variable (students : Type) (grades : students → Prop)
variable [decidable_pred grades]   -- Assuming decidable grades (Excellent or Good)
variable (甲 乙 丙 丁 : students)
variable (excellent good : Prop)

-- Conditions as given
axiom H1 : (∃ 甲 乙 丙 丁 : students, grades 甲 = excellent ∧ grades 乙 = excellent ∧ grades 丙 = good ∧ grades 丁 = good) ∨
           (∃ 甲 乙 丙 丁 : students, grades 甲 = excellent ∧ grades 乙 = good ∧ grades 丙 = excellent ∧ grades 丁 = good) ∨
           (∃ 甲 乙 丙 丁 : students, grades 甲 = good ∧ grades 乙 = excellent ∧ grades 丙 = good ∧ grades 丁 = excellent) ∨
           (∃ 甲 乙 丙 丁 : students, grades 甲 = good ∧ grades 乙 = good ∧ grades 丙 = excellent ∧ grades 丁 = excellent)
axiom H2 : (∃ 乙 丙 : students, grades 乙 ≠ grades 丙)  -- 甲 sees the grades of 乙 and 丙 and can't determine his own grade
axiom H3 : ∀ 乙 丙 : students, (grades 乙 = excellent ∨ grades 乙 = good) → (grades 丙 = excellent ∨ grades 丙 = good)
axiom H4 : ∀ 丁 甲 : students, (grades 丁 = excellent ∨ grades 丁 = good) → (grades 甲 = excellent ∨ grades 甲 = good)

-- Question/Proof goal
theorem can_determine_own_grades : ∀ 乙 丁 : students, (graded 乙) → (graded 丁) :=
by
  intro 乙 丁
  sorry

end can_determine_own_grades_l773_773668


namespace find_sum_of_coordinates_l773_773026

noncomputable theory
open_locale big_operators

def point := (ℝ × ℝ)

def A : point := (-1, 2)
def B : point := (3, -1)
def D : point := (5, 7)

def is_diagonal (A D : point) (B C : point) : Prop :=
  let M_AD := ((A.1 + D.1) / 2, (A.2 + D.2) / 2) in
  let M_BC := ((B.1 + C.1) / 2, (B.2 + C.2) / 2) in
  M_AD = M_BC

def is_perpendicular (A B C : point) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2) in
  let BC := (C.1 - B.1, C.2 - B.2) in
  AB.1 * BC.1 + AB.2 * BC.2 = 0

def sum_of_coordinates (C : point) : ℝ :=
  C.1 + C.2

theorem find_sum_of_coordinates :
  ∃ (C : point), is_diagonal A D B C ∧ is_perpendicular A B C ∧ sum_of_coordinates C = 9 :=
sorry

end find_sum_of_coordinates_l773_773026


namespace wholesale_cost_is_200_l773_773344

variable (W R E : ℝ)

def retail_price (W : ℝ) : ℝ := 1.20 * W

def employee_price (R : ℝ) : ℝ := 0.75 * R

-- Main theorem stating that given the retail and employee price formulas and the employee paid amount,
-- the wholesale cost W is equal to 200.
theorem wholesale_cost_is_200
  (hR : R = retail_price W)
  (hE : E = employee_price R)
  (heq : E = 180) :
  W = 200 :=
by
  sorry

end wholesale_cost_is_200_l773_773344


namespace fractional_parts_subtraction_l773_773559

theorem fractional_parts_subtraction (a b: ℝ)
  (h_a: a = fract (sqrt (3 + sqrt 5) - sqrt (3 - sqrt 5)))
  (h_b: b = fract (sqrt (6 + 3 * sqrt 3) - sqrt (6 - 3 * sqrt 3))) :
  (2 / b) - (1 / a) = sqrt 6 - sqrt 2 + 1 :=
by
  sorry

end fractional_parts_subtraction_l773_773559


namespace three_digit_multiples_of_24_l773_773081

theorem three_digit_multiples_of_24 : 
  let lower_bound := 100
  let upper_bound := 999
  let div_by := 24
  let first := lower_bound + (div_by - lower_bound % div_by) % div_by
  let last := upper_bound - (upper_bound % div_by)
  ∃ n : ℕ, (n + 1) = (last - first) / div_by + 1 := 
sorry

end three_digit_multiples_of_24_l773_773081


namespace size_relationship_l773_773098

def a : ℝ := (-2023)^0
def b : ℝ := -1 / 10
def c : ℝ := (-5 / 3)^2

theorem size_relationship : c > a ∧ a > b :=
by
  sorry

end size_relationship_l773_773098


namespace george_small_pizza_count_l773_773697

def slices_per_small_pizza := 4
def slices_per_large_pizza := 8
def num_large_pizzas := 2
def george_slices := 3
def bob_slices := george_slices + 1
def susie_slices := bob_slices / 2
def bill_slices := 3
def fred_slices := 3
def mark_slices := 3
def slices_left_over := 10

theorem george_small_pizza_count : 
  let total_consumed := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices,
      total_slices := total_consumed + slices_left_over,
      large_pizza_slices := num_large_pizzas * slices_per_large_pizza,
      small_pizza_slices := total_slices - large_pizza_slices
  in small_pizza_slices / slices_per_small_pizza = 3 :=
sorry

end george_small_pizza_count_l773_773697


namespace number_of_correct_statements_l773_773712

theorem number_of_correct_statements :
  let statement1 := "The axis of symmetry of an isosceles triangle is the perpendicular bisector of the base"
  let statement2 := "The diagonals of a rhombus are equal and bisect each other"
  let statement3 := "A quadrilateral whose four interior angles are all equal is a rectangle"
  let statement4 := "The quadrilateral formed by sequentially connecting the midpoints of the sides of a quadrilateral with equal diagonals is a rhombus"
  let statement5 := "A quadrilateral whose diagonals are perpendicular and equal is a square"
  let conditions := [statement1, statement2, statement3, statement4, statement5]
  3
:= sorry

end number_of_correct_statements_l773_773712


namespace expected_zeroes_in_string_l773_773137

theorem expected_zeroes_in_string (B : ℕ) (A : ℚ) (h1 : B = 4) (h2 : A = 1/5) :
  ∀ E : ℚ, E = B / 2 :=
by
  intro E
  rw [h1]
  norm_num
  exact rfl

end expected_zeroes_in_string_l773_773137


namespace evaluate_g2_plus_g_neg2_l773_773173

def g (x : ℝ) : ℝ := 3 * x ^ 6 + 5 * x ^ 4 - 6 * x ^ 2 + 7

theorem evaluate_g2_plus_g_neg2 : g 2 = 4 → g 2 + g (-2) = 8 :=
by
  intros h
  rw [← h]
  -- Since g is an even function, g(-x) = g(x)
  have h_even : g (-2) = g 2 := by
    simp [g]
  rw [h_even]
  rw [h, add_self_eq_double]
  simp
  exact Ne.symm (@zero_ne_two ℝ _ _)

end evaluate_g2_plus_g_neg2_l773_773173


namespace sequence_properties_l773_773237

noncomputable def a_n (n : ℕ) : ℝ := (1 + 1 / n) ^ (n + 1)

theorem sequence_properties :
  (∀ n : ℕ, a_n n > a_n (n + 1)) ∧
  (∀ n : ℕ, a_n n > real.exp 1) :=
by
  sorry

end sequence_properties_l773_773237


namespace part1_part2_l773_773054

noncomputable def f (a x : ℝ) : ℝ := a * x * (Real.log x - 1) - x^2

theorem part1 (a : ℝ) (x1 x2: ℝ) (h1 : x1 < x2) (h2 : a * Real.log x1 - 2 * x1 = 0) 
  (h3 : a * Real.log x2 - 2 * x2 = 0) : 
  2 * Real.exp(1) < a :=
sorry

theorem part2 (a : ℝ) (x1 x2 : ℝ) (λ : ℝ) (h1 : x1 < x2) 
  (h2 : a * Real.log x1 - 2 * x1 = 0) (h3 : a * Real.log x2 - 2 * x2 = 0)
  (h4 : Real.log x1 + λ * Real.log x2 > 1 + λ) :
  1 ≤ λ :=
sorry

end part1_part2_l773_773054


namespace quadratic_integer_roots_l773_773586

theorem quadratic_integer_roots (a b x : ℤ) :
  (∀ x₁ x₂ : ℤ, x₁ + x₂ = -b / a ∧ x₁ * x₂ = b / a → (x₁ = x₂ ∧ x₁ = -2 ∧ b = 4 * a) ∨ (x = -1 ∧ a = 0 ∧ b ≠ 0) ∨ (x = 0 ∧ a ≠ 0 ∧ b = 0)) :=
sorry

end quadratic_integer_roots_l773_773586


namespace ao_minus_bc_eq_ob_l773_773800

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (A B C D O : V)

-- Conditions
def is_rectangle (A B C D : V) : Prop :=
  let AB := B - A in
  let BC := C - B in
  let CD := D - C in
  let DA := A - D in
  AB = CD ∧ BC = DA ∧ (∃ O : V, (O = (A + C) / 2) ∧ (O = (B + D) / 2))

-- Theorem Statement
theorem ao_minus_bc_eq_ob 
  (h : is_rectangle A B C D)
  (hO : O = (A + C) / 2)
  (hO_bisects : O = (B + D) / 2) :
  (O - A) - (C - B) = (O - B) :=
sorry

end ao_minus_bc_eq_ob_l773_773800


namespace remainder_of_square_l773_773177

variable (N X : Set ℤ)
variable (k : ℤ)

/-- Given any n in set N and any x in set X, where dividing n by x gives a remainder of 3,
prove that the remainder of n^2 divided by x is 9 mod x. -/
theorem remainder_of_square (n x : ℤ) (hn : n ∈ N) (hx : x ∈ X)
  (h : ∃ k, n = k * x + 3) : (n^2) % x = 9 % x :=
by
  sorry

end remainder_of_square_l773_773177


namespace is_condition_B_an_algorithm_l773_773660

-- Definitions of conditions A, B, C, D
def condition_A := "At home, it is generally the mother who cooks"
def condition_B := "The steps to cook rice include washing the pot, rinsing the rice, adding water, and heating"
def condition_C := "Cooking outdoors is called camping cooking"
def condition_D := "Rice is necessary for cooking"

-- Definition of being considered an algorithm
def is_algorithm (s : String) : Prop :=
  s = condition_B  -- Based on the analysis that condition_B meets the criteria of an algorithm

-- The proof statement to show that condition_B can be considered an algorithm
theorem is_condition_B_an_algorithm : is_algorithm condition_B :=
by
  sorry

end is_condition_B_an_algorithm_l773_773660


namespace sum_of_solutions_sum_is_zero_l773_773284

theorem sum_of_solutions (x : ℝ) (h : |x^2 - 10 * x + 30| = 4) : false :=
begin
  sorry
end

theorem sum_is_zero : ∑ x in { x : ℝ | |x^2 - 10 * x + 30| = 4 }.to_finset, x = 0 :=
begin
  apply finset.sum_empty,
  sorry
end

end sum_of_solutions_sum_is_zero_l773_773284


namespace cubic_root_form_l773_773956

theorem cubic_root_form (d e f : ℕ) (x : ℝ) :
  (27 * x ^ 3 - 12 * x ^ 2 - 12 * x - 4 = 0) →
  x = (real.cbrt d + real.cbrt e + 2) / f →
  d + e + f = 780 :=
sorry

end cubic_root_form_l773_773956


namespace coloring_count_l773_773084

/--
Consider the integers from 2 to 10. We want to count the number of ways to paint each of these integers with one of three colors: red, green, or blue.

The painting should satisfy the following conditions:
1. Consecutive integers cannot have the same color.
2. An integer must have a different color from each of its proper divisors.

The final result should prove that the number of ways to paint these integers is 192.
-/
theorem coloring_count :
  let colors := {red, green, blue},
      proper_divisors := λ n, {d | d < n ∧ n % d = 0},
      constraint := λ n m, n ≠ m ∧ (m = n + 1 ∨ m ∈ proper_divisors n)
  in
  count_valid_colorings 2 10 colors constraint = 192 :=
sorry

end coloring_count_l773_773084


namespace grayson_unanswered_l773_773075

noncomputable def unanswered_questions : ℕ :=
  let total_questions := 200
  let first_set_questions := 50
  let first_set_time := first_set_questions * 1 -- 1 minute per question
  let second_set_questions := 50
  let second_set_time := second_set_questions * (90 / 60) -- convert 90 seconds to minutes
  let third_set_questions := 25
  let third_set_time := third_set_questions * 2 -- 2 minutes per question
  let total_answered_time := first_set_time + second_set_time + third_set_time
  let total_time_available := 4 * 60 -- 4 hours in minutes 
  let unanswered := total_questions - (first_set_questions + second_set_questions + third_set_questions)
  unanswered

theorem grayson_unanswered : unanswered_questions = 75 := 
by 
  sorry

end grayson_unanswered_l773_773075


namespace measure_of_arc_CD_l773_773864

-- Definition: A circle P with a chord CD and the angle ∠CBD is 35 degrees
variables {P : Type} [ring P] [module P] [metric_space P]
variable {C D B : P}
variable (circle_P : Circle P)
variable (CD : segment circle_P)
variable (angle_CBD : ℝ)
variable (angle_CBD_eq : angle_CBD = 35)

-- Theorem: Prove that the measure of arc CD is 70 degrees
theorem measure_of_arc_CD (h : ∠CBD = 35) : measure_arc (CD) = 70 :=
sorry

end measure_of_arc_CD_l773_773864


namespace find_number_eq_l773_773848

theorem find_number_eq : ∃ x : ℚ, (35 / 100) * x = (25 / 100) * 40 ∧ x = 200 / 7 :=
by
  sorry

end find_number_eq_l773_773848


namespace range_of_a_l773_773792

noncomputable def has_real_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 - a*x + 1 = 0 ∧ y^2 - a*y + 1 = 0

def holds_for_all_x (a : ℝ) : Prop :=
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → a^2 - 3*a - x + 1 ≤ 0

theorem range_of_a (a : ℝ) :
  (¬ ((has_real_roots a) ∧ (holds_for_all_x a))) ∧ (¬ (¬ (holds_for_all_x a))) → (1 ≤ a ∧ a < 2) :=
by
  sorry

end range_of_a_l773_773792


namespace parabola_equation_origin_l773_773630

theorem parabola_equation_origin (x0 : ℝ) :
  ∃ (p : ℝ), (p > 0) ∧ (x0^2 = 2 * p * 2) ∧ (p = 2) ∧ (x0^2 = 4 * 2) := 
by 
  sorry

end parabola_equation_origin_l773_773630


namespace remainder_of_polynomial_division_l773_773281

-- Definitions based on conditions in the problem
def polynomial (x : ℝ) : ℝ := 8 * x^4 - 22 * x^3 + 9 * x^2 + 10 * x - 45

def divisor (x : ℝ) : ℝ := 4 * x - 8

-- Proof statement as per the problem equivalence
theorem remainder_of_polynomial_division : polynomial 2 = -37 := by
  sorry

end remainder_of_polynomial_division_l773_773281


namespace housewife_spending_l773_773692

theorem housewife_spending (P R A : ℝ) (h1 : R = 34.2) (h2 : R = 0.8 * P) (h3 : A / R - A / P = 4) :
  A = 683.45 :=
by
  sorry

end housewife_spending_l773_773692


namespace sales_price_l773_773245

/-
Define the variables involved according to the conditions given:
  C : Cost price
  G : Gross profit
  S : Sales price
-/
variables (C G S : ℝ)

-- Conditions
def condition1 := G = 1.70 * C
def condition2 := G = 51
def condition3 := S = C + G

-- Proposition to be proved
theorem sales_price (h1 : condition1) (h2 : condition2) (h3 : condition3) : S = 81 := 
by sorry

end sales_price_l773_773245


namespace evan_chen_problem_l773_773147

theorem evan_chen_problem
  (P : ℚ[X]) (a b c x : ℚ)
  (hP : P = X^3 + 27 * X^2 + 199 * X + 432)
  (ha : P.eval (-a) = 0)
  (hb : P.eval (-b) = 0)
  (hc : P.eval (-c) = 0)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x)
  (h : sqrt ((a + b + c) / x) = sqrt ((b + c + x) / a) + sqrt ((c + a + x) / b) + sqrt ((a + b + x) / c))
  : let m := 432 in let n := 415 in m + n = 847 := 
by simp -- simplifying lemma, actual proof skipped

end evan_chen_problem_l773_773147


namespace exists_six_points_configuration_l773_773540

-- Define the condition of six points each connected to exactly four others
def connects_four_others (G : Finset (Finset ℕ)) : Prop :=
  ∀ (p ∈ G), p.card = 4

-- Define the condition of non-intersecting segments
def non_intersecting (G : Finset (Finset ℕ)) : Prop :=
  -- A function to implement non-intersecting condition; assumption here as a placeholder
  true

-- Define the condition of having exactly six points
def six_points (G : Finset (Finset ℕ)) : Prop :=
  G.card = 6

-- Prove that it's possible to have such a configuration satisfying all conditions
theorem exists_six_points_configuration :
  ∃ G : Finset (Finset ℕ), six_points G ∧ connects_four_others G ∧ non_intersecting G :=
  sorry

end exists_six_points_configuration_l773_773540


namespace average_weight_stem_leaf_plot_l773_773629

theorem average_weight_stem_leaf_plot :
  let weights := [25, 30, 31, 32, 34, 35, 37, 39, 40, 41, 42, 44, 45, 48] in
  (∑ i in weights, i) / weights.length = 35.5 :=
by
  let weights := [25, 30, 31, 32, 34, 35, 37, 39, 40, 41, 42, 44, 45, 48]
  have length_check : weights.length = 14 := rfl
  have sum_check : ∑ i in weights, i = 497 := rfl
  sorry

end average_weight_stem_leaf_plot_l773_773629


namespace vector_parallel_eq_l773_773836

theorem vector_parallel_eq (m : ℝ) : 
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (3, -2)
  a.1 * b.2 = a.2 * b.1 -> m = -6 := 
by 
  sorry

end vector_parallel_eq_l773_773836


namespace correct_sequence_of_propositions_l773_773835

def parallel (v w : vector) : Prop :=
  ∃ (λ : Real), v = λ • w

def perpendicular (v w : vector) : Prop :=
  ∀ λ : Real, v = λ • w → v = 0

theorem correct_sequence_of_propositions 
  (a b : line) 
  (α : plane)
  (e1 e2 n : vector)
  (h1 : e1 ∥ e2)
  (h2 : e1 ∥ n)
  (h3 : e1 ∥ n ∧ e1 ∥ e2 → a ∥ b) 
  (h4 : e1 ∥ n ∧ b ∉ α ∧ e1 ⊥ e2 → b ∥ α)
  (h5 : e1 ∥ e2 ∧ e1 ∥ n → b ⊥ α)
  : true :=
begin
  -- The hypotheses imply the correct sequence of true propositions.
  -- Hypotheses indicate B: ②, ③, ④ as the true sequence.
  sorry
end

end correct_sequence_of_propositions_l773_773835


namespace goldenRabbitCard_count_l773_773862

def isGoldenRabbitCard (num : ℕ) : Prop :=
  let digits := (num % 10, (num / 10) % 10, (num / 100) % 10, (num / 1000) % 10)
  digits.1 = 6 ∨ digits.1 = 8 ∨ digits.2 = 6 ∨ digits.2 = 8 ∨ digits.3 = 6 ∨ digits.3 = 8 ∨ digits.4 = 6 ∨ digits.4 = 8

def countGoldenRabbitCards : ℕ :=
  (List.range 10000).countp isGoldenRabbitCard

theorem goldenRabbitCard_count : countGoldenRabbitCards = 5904 := by
  sorry

end goldenRabbitCard_count_l773_773862


namespace road_trip_people_count_l773_773743

theorem road_trip_people_count
  (hours : ℕ)
  (water_per_hour : ℝ)
  (total_bottles : ℕ)
  (total_hours : hours = 16)
  (water_consumption : water_per_hour = 1/2)
  (total_water_needed : total_bottles = 32) :
  (total_bottles : ℝ) / (total_hours * water_per_hour) = 4 := 
by
  sorry

end road_trip_people_count_l773_773743


namespace x2_x3_sum_lt_two_l773_773473

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (2 * x^2 - x - 1)

def monotonic_intervals : (set ℝ × set ℝ × set ℝ) :=
  ( { x | x < -2 }, { x | -2 < x ∧ x < 1/2 }, { x | x > 1/2 } )

def g (x : ℝ) (a : ℝ) : ℝ := abs (f x) - a

def range_a : set ℝ := { a | (9 / Real.exp 2 < a) ∧ (a < Real.sqrt (Real.exp 1)) }

theorem x2_x3_sum_lt_two (x1 x2 x3 a : ℝ)
  (h1 : x1 < x2) (h2 : x2 < x3)
  (h3 : g x1 a = 0) (h4 : g x2 a = 0) (h5 : g x3 a = 0)
  (h6 : a ∈ range_a) : x2 + x3 < 2 := sorry

end x2_x3_sum_lt_two_l773_773473


namespace valid_two_digit_even_numbers_l773_773002

/-- A number in the range 10 to 99 is even. -/
def is_even (n : ℕ) : Prop :=
  n % 2 = 0

/-- Split a two-digit number into tens and units digits. -/
def split_digits (n : ℕ) : ℕ × ℕ :=
  (n / 10, n % 10)

/-- The sum of the digits of a two-digit number n. -/
def sum_of_digits (n : ℕ) : ℕ :=
  let (t, u) := split_digits n
  t + u

/-- The correct list of valid two-digit numbers under the given conditions. -/
def valid_numbers : List ℕ :=
  [70, 80, 90, 62, 72, 82, 92, 84, 94]

/-- An equivalent math proof problem in Lean 4 statement. -/
theorem valid_two_digit_even_numbers : 
  {n : ℕ | is_even n ∧ 10 ≤ n ∧ n ≤ 99 ∧ 6 < sum_of_digits n ∧ n / 10 ≥ (n % 10) + 4} = { 70, 80, 90, 62, 72, 82, 92, 84, 94 }.to_finset :=
by
  sorry 

end valid_two_digit_even_numbers_l773_773002


namespace ribbon_left_after_gifts_l773_773905

-- Define the initial conditions
def total_ribbon : ℕ := 18
def gifts : ℕ := 6
def ribbon_per_gift : ℕ := 2

-- State the theorem
theorem ribbon_left_after_gifts (total_ribbon = 18) (gifts = 6) (ribbon_per_gift = 2) : 
  total_ribbon - (gifts * ribbon_per_gift) = 6 := 
by
  sorry

end ribbon_left_after_gifts_l773_773905


namespace area_of_triangle_OPQ_l773_773530

-- Define the points and their relationships
variables {O A B P Q : Type} [MetricSpace O] [MetricSpace A] [MetricSpace B] [MetricSpace P] [MetricSpace Q]

-- Conditions given in the problem statement
-- OAB is a triangle
def angle_AOB_90_degrees (O A B : Type) [MetricSpace O] [MetricSpace A] [MetricSpace B] : Prop := 
  ∠AOB = 90

-- Length OB is 13 cm
def length_OB (O B : Type) [MetricSpace O] [MetricSpace B] : Prop := 
  dist O B = 13

-- Proportional segments on AB
def segment_proportions (O A B P Q : Type) [MetricSpace O] [MetricSpace A] [MetricSpace B] [MetricSpace P] [MetricSpace Q] : Prop := 
  let x := dist A B in 
  26 * dist A P = 22 * dist P Q ∧ 22 * dist P Q = 11 * dist Q B

-- Vertical height from P to Q is 4 cm
def vertical_height_PQ (P Q : Type) [MetricSpace P] [MetricSpace Q] : Prop := 
  dist P Q = 4

-- Theorem to prove
theorem area_of_triangle_OPQ 
  (O A B P Q : Type) 
  [MetricSpace O] [MetricSpace A] [MetricSpace B] [MetricSpace P] [MetricSpace Q] 
  (h1 : angle_AOB_90_degrees O A B)
  (h2 : length_OB O B)
  (h3 : segment_proportions O A B P Q)
  (h4 : vertical_height_PQ P Q) : 
  area O P Q = 26 := 
sorry

end area_of_triangle_OPQ_l773_773530


namespace monotonically_increasing_range_of_a_l773_773478

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4 * x - 5)

theorem monotonically_increasing_range_of_a :
  ∀ (a : ℝ), (∀ x : ℝ, x > a → f x > f a) ↔ a ≥ 5 :=
by
  intro a
  unfold f
  sorry

end monotonically_increasing_range_of_a_l773_773478


namespace solve_quadratic_equation_l773_773590

theorem solve_quadratic_equation (x : ℝ) : 
  (2 * x^2 - 4 * x = 6 - 3 * x) ↔ 
  (x = -3/2 ∨ x = 2) :=
by
  sorry

end solve_quadratic_equation_l773_773590


namespace determine_a_l773_773833

def P : Set ℝ := {1, 2}
def Q (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

theorem determine_a (a : ℝ) : P ∪ Q a = P ↔ a ∈ {0, -2, -1} := by sorry

end determine_a_l773_773833


namespace tickets_sold_at_door_l773_773226

theorem tickets_sold_at_door (A D : ℕ) 
    (h1 : A + D = 800)
    (h2 : 14.50 * A + 22.00 * D = 16640) 
    : D = 672 :=
by
  sorry

end tickets_sold_at_door_l773_773226


namespace min_x_plus_y_l773_773787

theorem min_x_plus_y (x y : ℕ) (hxy : x ≠ y) (h : (1/x : ℝ) + 1/y = 1/24) : x + y = 98 :=
sorry

end min_x_plus_y_l773_773787


namespace distances_less_than_PiQ_l773_773546

theorem distances_less_than_PiQ (n : ℕ) (h : n ≥ 12) (P : Fin n → EuclideanSpace ℝ (Fin 2)) (Q : EuclideanSpace ℝ (Fin 2)) (distinct_points : Function.Injective P) :
  ∃ i : Fin n, (↑n / 6 - 1) ≤ (Finset.filter (λ j, j ≠ i) Finset.univ).card (λ j, (P j -ᵥ P i).norm < (Q -ᵥ P i).norm) :=
sorry

end distances_less_than_PiQ_l773_773546


namespace find_m_l773_773499

theorem find_m (x y m : ℤ) (h1 : x = 3) (h2 : y = 1) (h3 : x - m * y = 1) : m = 2 :=
by
  -- Proof goes here
  sorry

end find_m_l773_773499


namespace sum_of_set_A_l773_773830

def A (a0 a1 a2 a3 : ℕ) : ℕ :=
  a0 + a1 * 2 + a2 * 2^2 + a3 * 2^3

theorem sum_of_set_A :
  (∀ a0 a1 a2 a3, a0 ∈ {1, 2} ∧ a1 ∈ {0, 1, 2} ∧ a2 ∈ {0, 1, 2} ∧ a3 ∈ {0, 1, 2} → 
  ∑ (a0, a1, a2, a3) in ({1, 2} × {0, 1, 2} × {0, 1, 2} × {0, 1, 2} : Finset (ℕ × ℕ × ℕ × ℕ)), A a0 a1 a2 a3) = 837 :=
by
-- proof will be here
sorry

end sum_of_set_A_l773_773830


namespace quarters_percentage_value_l773_773299

theorem quarters_percentage_value (dimes quarters : Nat) (value_dime value_quarter : Nat) (total_value quarter_value : Nat)
(h_dimes : dimes = 30)
(h_quarters : quarters = 40)
(h_value_dime : value_dime = 10)
(h_value_quarter : value_quarter = 25)
(h_total_value : total_value = dimes * value_dime + quarters * value_quarter)
(h_quarter_value : quarter_value = quarters * value_quarter) :
(quarter_value : ℚ) / (total_value : ℚ) * 100 = 76.92 := 
sorry

end quarters_percentage_value_l773_773299


namespace number_of_possible_radii_l773_773721

theorem number_of_possible_radii (r : ℕ) (h1 : r < 100) (h2 : 200 % r = 0) : 
  ∃ n, n = 8 := 
begin
  sorry
end

end number_of_possible_radii_l773_773721


namespace part_I_part_II_l773_773817

open Real

-- Part I
theorem part_I (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = exp x + a * exp (-x) - 2 * x)
  (h_odd : ∀ x : ℝ, f (-x) = - (f x)) : a = -1 ∧ ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by
  sorry

-- Part II
theorem part_II (b : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_f : ∀ x : ℝ, f x = exp x - exp (-x) - 2 * x)
  (h_g : ∀ x : ℝ, g x = f (2 * x) - 4 * b * f x)
  (h_pos : ∀ x : ℝ, x > 0 → g x > 0) : b ≤ 2 := by
  sorry

end part_I_part_II_l773_773817


namespace sin_cos_identity_l773_773766

theorem sin_cos_identity (a : ℝ) (h : Real.sin (π - a) = -2 * Real.sin (π / 2 + a)) : 
  Real.sin a * Real.cos a = -2 / 5 :=
by
  sorry

end sin_cos_identity_l773_773766


namespace total_apartment_units_l773_773569

-- Define the number of apartment units on different floors
def units_first_floor := 2
def units_other_floors := 5
def num_other_floors := 3
def num_buildings := 2

-- Calculation of total units in one building
def units_one_building := units_first_floor + num_other_floors * units_other_floors

-- Calculation of total units in all buildings
def total_units := num_buildings * units_one_building

-- The theorem to prove
theorem total_apartment_units : total_units = 34 :=
by
  sorry

end total_apartment_units_l773_773569


namespace area_of_woods_l773_773632

def width := 8 -- the width in miles
def length := 3 -- the length in miles
def area (w : Nat) (l : Nat) : Nat := w * l -- the area function for a rectangle

theorem area_of_woods : area width length = 24 := by
  sorry

end area_of_woods_l773_773632


namespace fib_mod_3_l773_773575

def fibonacci : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

theorem fib_mod_3 (k : ℕ) : fibonacci (4 * k) % 3 = 0 :=
sorry

end fib_mod_3_l773_773575


namespace find_minimum_part2_i_part2_ii_l773_773051

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 + 4 / x^3

theorem find_minimum : ∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → f y ≥ f x :=
sorry

variables {x1 x2 : ℝ}
hypothesis h1 : f x1 = f x2
hypothesis h2 : x1 < x2

theorem part2_i : x1^3 + (2 - x1)^3 < x1^4 + (2 - x1)^4 :=
sorry

theorem part2_ii : x1 + x2 > 2 :=
sorry

end find_minimum_part2_i_part2_ii_l773_773051


namespace algebra_problem_l773_773500

theorem algebra_problem (a b c d x : ℝ) (h1 : a = -b) (h2 : c * d = 1) (h3 : |x| = 3) : 
  (a + b) / 2023 + c * d - x^2 = -8 := by
  sorry

end algebra_problem_l773_773500


namespace seconds_in_day_misfortune_l773_773240

theorem seconds_in_day_misfortune (minutes_in_day : ℕ) (seconds_in_minute : ℕ)
  (h1 : minutes_in_day = 77) (h2 : seconds_in_minute = 91) : 
  let minutes_in_hour := Nat.gcd 77 91 in
  let hours_in_day := minutes_in_day / minutes_in_hour in
  let seconds_in_hour := minutes_in_hour * seconds_in_minute in
  hours_in_day * seconds_in_hour = 1001 :=
by
  sorry

end seconds_in_day_misfortune_l773_773240


namespace plotted_points_on_parabola_l773_773762

theorem plotted_points_on_parabola (t : ℝ) : 
  let x := 3^t - 4 in 
  let y := 9^t - 7 * 3^t + 2 in
  y = x^2 + x - 10 := 
by 
  let x := 3^t - 4 
  let y := 9^t - 7 * 3^t + 2 
  have h1 : 3^t = x + 4 := sorry
  have h2 : 9^t = (3^t)^2 := sorry
  have h3 : y = (x + 4)^2 - 7 * (x + 4) + 2, by rw [h1, h2]; sorry
  have h4 : y = x^2 + 8*x + 16 - 7*x - 28 + 2 := sorry
  have h5 : y = x^2 + x - 10 := sorry
  exact h5

end plotted_points_on_parabola_l773_773762


namespace reciprocal_of_2023_l773_773981

theorem reciprocal_of_2023 : 1 / 2023 = (1 : ℚ) / 2023 :=
by sorry

end reciprocal_of_2023_l773_773981


namespace equal_sum_split_count_l773_773526

namespace Problem1

def numbers := list.range' 1 12

def sums_of_squares (l : list ℕ) : ℕ :=
l.map (λ x, x*x ) |>.sum

theorem equal_sum_split_count :
  (∃ (A B : finset ℕ), (A ∪ B = finset.univ ∧ A ∩ B = ∅) ∧ sums_of_squares A.val = 325 ∧ sums_of_squares B.val = 325) → 5 :=
sorry

end Problem1

end equal_sum_split_count_l773_773526


namespace number_of_distinct_prime_factors_l773_773152

-- Given: A is the product of the divisors of 60.
-- Prove: The number of distinct prime factors of A is 3.

theorem number_of_distinct_prime_factors (A : ℕ) 
  (h1 : A = ∏ d in (finset.filter (λ d, d ∣ 60) (finset.range (60 + 1))), d) :
  (factors_multiset A).to_finset.card = 3 := 
sorry

end number_of_distinct_prime_factors_l773_773152


namespace region_area_l773_773647

theorem region_area : 
  (∃ (x y : ℝ), abs (4 * x - 16) + abs (3 * y + 9) ≤ 6) →
  (∀ (A : ℝ), (∀ x y : ℝ, abs (4 * x - 16) + abs (3 * y + 9) ≤ 6 → 0 ≤ A ∧ A = 6)) :=
by
  intro h exist_condtion
  sorry

end region_area_l773_773647


namespace probability_at_least_three_passes_in_four_attempts_l773_773694

theorem probability_at_least_three_passes_in_four_attempts :
  let p := 4 / 5
  ∑ k in {3, 4}, (nat.choose 4 k) * (p ^ k) * ((1 - p) ^ (4 - k)) = 512 / 625 := by
  sorry

end probability_at_least_three_passes_in_four_attempts_l773_773694


namespace integer_solutions_count_l773_773488

theorem integer_solutions_count :
  {x : ℤ | (x - 3) ^ (36 - x ^ 2) = 1}.to_finset.card = 4 := 
by
  sorry

end integer_solutions_count_l773_773488


namespace isosceles_triangle_l773_773640

noncomputable def is_isosceles (A B C P D E : Point) :=
  ∃ (l : ℝ), P ∈ line_segment(A, C) ∧ P ∈ line_segment(B, C) ∧
  is_parallel(line_segment(P, D), line_segment(A, C)) ∧
  is_parallel(line_segment(P, E), line_segment(B, C)) ∧
  len_segment(P, D) = l ∧ len_segment(P, E) = l

theorem isosceles_triangle
  {Triangle : Type*} (A B C P D E : Point)
  (h1 : P ∈ line_segment(A, C) ∨ P ∈ line_segment(C, extension(A, B)))
  (h2 : is_parallel(line_segment(P, D), line_segment(A, C)))
  (h3 : is_parallel(line_segment(P, E), line_segment(B, C)))
  (h4 : len_segment(P, D) = len_segment(P, E)) :
  is_isosceles(A, B, C, P, D, E) :=
begin
  sorry
end

end isosceles_triangle_l773_773640


namespace find_original_selling_price_l773_773303

variable (x : ℝ) (discount_rate : ℝ) (final_price : ℝ)

def original_selling_price_exists (x : ℝ) (discount_rate : ℝ) (final_price : ℝ) : Prop :=
  (x * (1 - discount_rate) = final_price) → (x = 700)

theorem find_original_selling_price
  (discount_rate : ℝ := 0.20)
  (final_price : ℝ := 560) :
  ∃ x : ℝ, original_selling_price_exists x discount_rate final_price :=
by
  use 700
  sorry

end find_original_selling_price_l773_773303


namespace smallest_x_solution_l773_773422

theorem smallest_x_solution :
  ∃ x : ℝ, x * |x| + 3 * x = 5 * x + 2 ∧ (∀ y : ℝ, y * |y| + 3 * y = 5 * y + 2 → x ≤ y)
:=
sorry

end smallest_x_solution_l773_773422


namespace father_l773_773113

theorem father's_age :
  ∃ (S F : ℕ), 2 * S + F = 70 ∧ S + 2 * F = 95 ∧ F = 40 :=
by
  sorry

end father_l773_773113


namespace solve_quadratic_inequality_l773_773628

theorem solve_quadratic_inequality :
  {x : ℝ | -x^2 + 3 * x + 28 ≤ 0} = set.Iic (-4) ∪ set.Ici 7 :=
by
  sorry

end solve_quadratic_inequality_l773_773628


namespace zachary_pushups_l773_773300

theorem zachary_pushups (david_pushups : ℕ) (h1 : david_pushups = 44) (h2 : ∀ z : ℕ, z = david_pushups + 7) : z = 51 :=
by
  sorry

end zachary_pushups_l773_773300


namespace distinct_prime_factors_of_A_l773_773151

noncomputable def A : ℕ := ∏ (d : ℕ) in (finset.filter (λ x, 60 % x = 0) (finset.range 61)), d

theorem distinct_prime_factors_of_A : (finset.filter (nat.prime) (nat.factorization A).support).card = 3 :=
sorry

end distinct_prime_factors_of_A_l773_773151


namespace solution_l773_773709

noncomputable def problem : Prop :=
  let num_apprentices := 200
  let num_junior := 20
  let num_intermediate := 60
  let num_senior := 60
  let num_technician := 40
  let num_senior_technician := 20
  let total_technician := num_technician + num_senior_technician
  let sampling_ratio := 10 / num_apprentices
  
  -- Number of technicians (including both technician and senior technicians) in the exchange group
  let num_technicians_selected := total_technician * sampling_ratio

  -- Probability Distribution of X
  let P_X_0 := 7 / 24
  let P_X_1 := 21 / 40
  let P_X_2 := 7 / 40
  let P_X_3 := 1 / 120

  -- Expected value of X
  let E_X := (0 * P_X_0) + (1 * P_X_1) + (2 * P_X_2) + (3 * P_X_3)
  E_X = 9 / 10

theorem solution : problem :=
  sorry

end solution_l773_773709


namespace solve_x_eq_l773_773945

theorem solve_x_eq : ∃ x : ℝ, (5 + 3.5 * x = 2 * x - 25 + x) ∧ x = -60 :=
by
  use -60
  split
  { linarith }
  { refl }

end solve_x_eq_l773_773945


namespace compute_series_value_l773_773919

theorem compute_series_value :
  let n := 1990 in
  (1 / 2^n : ℝ) * ((finset.range (n / 2 + 1)).sum (λ k, (-1)^k * 3^k * (nat.choose n (2 * k)))) = -1 / 2 :=
by sorry

end compute_series_value_l773_773919


namespace range_of_k_range_of_a_l773_773827

-- Condition for the function to pass through the first and third quadrants.
def passes_through_first_and_third_quadrants (k : ℝ) : Prop :=
  k > 4

-- Condition that given points are in the first quadrant and y1 < y2
def valid_points (a y1 y2 : ℝ) : Prop :=
  a > 0 ∧ y1 < y2 ∧ y2 = (k-4)/(2a + 1) ∧ y1 = (k-4)/(a + 5)

-- Main theorem to prove range of k
theorem range_of_k (k : ℝ) : passes_through_first_and_third_quadrants k → k > 4 :=
by
  sorry

-- Main theorem to prove range of a
theorem range_of_a (a y1 y2 k : ℝ) : valid_points a y1 y2 → (0 < a ∧ a < 4) :=
by
  sorry

end range_of_k_range_of_a_l773_773827


namespace yield_is_eight_percent_l773_773316

noncomputable def par_value : ℝ := 100
noncomputable def annual_dividend : ℝ := 0.12 * par_value
noncomputable def market_value : ℝ := 150
noncomputable def yield_percentage : ℝ := (annual_dividend / market_value) * 100

theorem yield_is_eight_percent : yield_percentage = 8 := 
by 
  sorry

end yield_is_eight_percent_l773_773316


namespace sum_of_roots_l773_773653

theorem sum_of_roots : 
  ∀ (x : ℝ), x^2 - 5 * x + 10 = 0 → x = 2 ∨ x =  he sorry :=
sorry

end sum_of_roots_l773_773653


namespace new_person_weight_l773_773667

theorem new_person_weight 
  (increase_avg: ℝ)
  (n: ℕ)
  (w_old: ℝ)
  (w_total: ℝ):
  increase_avg = 5.5 →
  n = 9 →
  w_old = 86 →
  w_total = n * increase_avg →
  ∃ w_new: ℝ, w_new = w_old + w_total :=
by
  intros h1 h2 h3 h4
  use w_old + w_total
  sorry

end new_person_weight_l773_773667


namespace roots_decomp_l773_773556

noncomputable def polynomial := λ (x : ℝ), x^3 - 15 * x^2 + 50 * x - 60

theorem roots_decomp (p q r A B C : ℝ) 
  (h_roots : polynomial p = 0 ∧ polynomial q = 0 ∧ polynomial r = 0)
  (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (h_decomp : ∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r → (1 / polynomial s) = (A / (s - p)) + (B / (s - q)) + (C / (s - r))) :
  1 / A + 1 / B + 1 / C = 135 := 
sorry

end roots_decomp_l773_773556


namespace stamps_ratio_l773_773261

theorem stamps_ratio (orig_stamps_P : ℕ) (addie_stamps : ℕ) (final_stamps_P : ℕ) 
  (h₁ : orig_stamps_P = 18) (h₂ : addie_stamps = 72) (h₃ : final_stamps_P = 36) :
  (final_stamps_P - orig_stamps_P) / addie_stamps = 1 / 4 :=
by {
  sorry
}

end stamps_ratio_l773_773261


namespace g_at_3_l773_773609

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_at_3 (h : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = 2) : g 3 = 0 := by
  sorry

end g_at_3_l773_773609


namespace solution_set_l773_773408

open Real

noncomputable def condition (x : ℝ) := x ≥ 2

noncomputable def eq_1 (x : ℝ) := sqrt (x + 5 - 6 * sqrt (x - 2)) + sqrt (x + 12 - 8 * sqrt (x - 2)) = 2

theorem solution_set :
  {x : ℝ | condition x ∧ eq_1 x} = {x : ℝ | 11 ≤ x ∧ x ≤ 18} :=
by sorry

end solution_set_l773_773408


namespace a2_a8_sum_l773_773462

variable {a : ℕ → ℝ}  -- Define the arithmetic sequence a

-- Conditions:
axiom arithmetic_sequence (n : ℕ) : a (n + 1) - a n = a 1 - a 0
axiom a1_a9_sum : a 1 + a 9 = 8

-- Theorem stating the question and the answer
theorem a2_a8_sum : a 2 + a 8 = 8 :=
by
  sorry

end a2_a8_sum_l773_773462


namespace largest_n_factorial_expression_l773_773209

theorem largest_n_factorial_expression :
  ∃ (n : ℕ), (∀ (k : ℕ), k > 0 → n! = (k + (n - 4))! / (k - 1)!) ∧ n = 23 :=
begin
  sorry
end

end largest_n_factorial_expression_l773_773209


namespace solution_set_of_inequality_l773_773759

theorem solution_set_of_inequality :
  {x : ℝ | 4*x^2 - 9*x > 5} = {x : ℝ | x < -1/4} ∪ {x : ℝ | x > 5} :=
by
  sorry

end solution_set_of_inequality_l773_773759


namespace tangent_line_at_1_minimum_h_l773_773180

noncomputable def e := Real.exp 1

def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x ^ 2 - e * x - 2
def f' (x : ℝ) (a : ℝ) : ℝ := Real.exp x - 2 * a * x - e

theorem tangent_line_at_1 (x y : ℝ) (a : ℝ) (h_a : a = 1) (h_x : x = 1) (h_f1 : f x a = -3) (h_f'1 : f' x a = -2) :
  2 * x + y + 1 = 0 :=
  sorry

theorem minimum_h (a : ℝ) (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ 1) :
  let h := f' x a in
  if a ≤ 1/2 then h ≥ 1 - e
  else if a > e/2 then h ≥ -2 * a
  else h ≥ 2 * a - 2 * a * Real.log (2 * a) - e :=
  sorry

end tangent_line_at_1_minimum_h_l773_773180


namespace they_met_on_wednesday_l773_773938

theorem they_met_on_wednesday 
  (D : ℕ) -- total distance of princess's journey
  (d : ℕ) -- number of days traveled by princess to cover 1/5 of her journey
  (prince_days : ℕ) -- days traveled by prince
  (total_days : ℕ) -- total days from meeting to arrival at castle
  (start_day : ℕ) -- start day of princess's journey (Friday)
  (meet_day : ℕ) -- day they met
  
  (h1 : prince_days = 2)
  (h2 : d = 2)
  (h3 : total_days = 11)
  (h4 : start_day = 5) -- Assume 5 represents Friday as start day
  (h5 : meet_day = start_day + 2) -- After 2 days
  
  : meet_day % 7 = 3 :=
by
  sorry

end they_met_on_wednesday_l773_773938


namespace seating_possible_l773_773314

-- Define the conditions of the problem
def participant := ℕ  -- or use an appropriate definition if considering a more complex participant structure

structure Olympiad :=
(participants : Finset participant)
(acquainted : participant → participant → Prop)
(circles : Finset participant → Prop)
(no_inter_circle : ∀ (c : Finset participant), circles c → 
  ∀ (x : participant), x ∉ c → ¬ (∃ y ∈ c, acquainted x y))
(num_participants : participants.card = 2018)

theorem seating_possible : ∃ (rooms : Finset (Finset participant)), rooms.card = 90 ∧
  (∀ c ∈ (Finset.powersetForall (λ r, r.card > 0) → Finset.card c), Olympiad.circles c → 
  ∀ r ∈ rooms, ¬ c ⊆ r) :=
sorry

end seating_possible_l773_773314


namespace correct_statements_l773_773466

noncomputable def y_eq := λ x a : ℝ, (2 * x + a : ℝ)^(1/2)
noncomputable def k_n := λ n a : ℝ, 1 / (2 * n + a)^(1/2)
noncomputable def x_n := λ n a : ℝ, -n - a
noncomputable def y_n := λ n a : ℝ, (n + a) / (2 * n + a)^(1/2)
noncomputable def s_n := λ {α : Type*} [add_monoid α] (n : ℚ) (a : ℚ), 
  (finset.range n).sum (λ k, 1 / (2 * k + a)^(1/2))

theorem correct_statements (n : ℕ) (a : ℝ) (h₀ : a > 0) (h₁ : |x_n 0 1| = |y_n 0 1|) :
  (a = 1) ∧
  ((n > 0) → (y_n n 1 = 2 * (3 : ℝ)^(1/2) / 3)) ∧
  ((n > 0) → (k_n n 1 > 2^(1/2) * real.sin (1 / (2 * n + 1)^(1/2)))) → false ∧
  ((n > 0) → (s_n ∑  k_n n 1 < 2^(1/2) * ((n + 1)^(1/2) - 1))) :=
begin
  sorry
end

end correct_statements_l773_773466


namespace ratio_of_squares_l773_773624

theorem ratio_of_squares (a b c : ℤ) (h : 72 / 98 = (a * real.sqrt b) / c) : a = 6 ∧ b = 1 ∧ c = 7 ∧ a + b + c = 14 :=
by
  sorry

end ratio_of_squares_l773_773624


namespace sum_of_first_15_terms_l773_773132

theorem sum_of_first_15_terms (a : ℕ → ℝ) (r : ℝ)
    (h_geom : ∀ n, a (n + 1) = a n * r)
    (h1 : a 1 + a 2 + a 3 = 1)
    (h2 : a 4 + a 5 + a 6 = -2) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 +
   a 10 + a 11 + a 12 + a 13 + a 14 + a 15) = 11 :=
sorry

end sum_of_first_15_terms_l773_773132


namespace polygon_divisible_by_zigzag_cut_l773_773892

-- Definitions and conditions
structure Polygon (P : Type) :=
(vertices : Set P)
(grid_aligned : ∀ v ∈ vertices, ∃ x y : ℕ, v = (x, y))

structure ZigzagCut (C : Type) :=
(segments : List (P × P))
(follows_grid : ∀ (s ∈ segments), ∃ x y : ℕ, (s.1 = (x, y) ∧ s.2 = (x + 1, y + 1)) ∨ (s.2 = (x, y) ∧ s.1 = (x - 1, y - 1)))
(half_length_small_segments : ∀ (s ∈ segments), let (p1, p2) := s in (dist p1 p2) = 0.5 * (dist p1' p2'))

-- Placeholder for distance function assuming Euclidean distance
noncomputable def dist {P} [Dist P] : P → P → ℝ := sorry

-- The condition that the cut starts and ends at the polygon boundary
structure ValidCut (P C : Type) extends Polygon P, ZigzagCut C :=
(within_polygon : ∀ (s ∈ C.segments), ∃ (p ∈ P.vertices), s = (p, p) ∨ s = ∅)
(boundary_touches_only_ends : ∃ (p1 p2 ∈ P.vertices), (p1 ∈ C.segments) ∧ (p2 ∈ C.segments))

-- The theorem to prove
theorem polygon_divisible_by_zigzag_cut (P C : Type) [Dist P] [ValidCut P C] : 
  ∃ (cut : ZigzagCut C), 
    (∀ (p q ∈ P.vertices), p ≠ q → (dist p q) = 0) →
    (cut.within_polygon ∨ cut.boundary_touches_only_ends) :=
sorry

end polygon_divisible_by_zigzag_cut_l773_773892


namespace third_even_number_sequence_l773_773423

theorem third_even_number_sequence (x : ℕ) (h_even : x % 2 = 0) (h_sum : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) = 180) : x + 4 = 30 :=
by
  sorry

end third_even_number_sequence_l773_773423


namespace min_x_plus_y_l773_773788

theorem min_x_plus_y (x y : ℕ) (hxy : x ≠ y) (h : (1/x : ℝ) + 1/y = 1/24) : x + y = 98 :=
sorry

end min_x_plus_y_l773_773788


namespace triangle_angle_C_min_value_expression_l773_773890

theorem triangle_angle_C
  (a b c A B C : ℝ) :
  1 + (Real.tan C / Real.tan B) = 2 * a / b →
  Real.cos C = 1 / 2 :=
sorry

theorem min_value_expression
  (a b c : ℝ) :
  (a + b)^2 - c^2 = 4 →
  (∃ b', b' = b ∧ Real.inv b' = 2) →
  (∀ b', Real.inv b' = 2 → -4 ≤ (Real.inv (b' * b')) - 3 * a) :=
sorry

end triangle_angle_C_min_value_expression_l773_773890


namespace average_children_in_families_with_children_l773_773949

/-- Let there be 10 families, each with an average of 2 children, and exactly 2 of these families are childless.
    Prove that the average number of children in the families with children is 2.5. -/
theorem average_children_in_families_with_children :
  ∀ (num_families : ℕ) (avg_children_per_family : ℕ) (num_childless_families : ℕ),
    num_families = 10 →
    avg_children_per_family = 2 →
    num_childless_families = 2 →
    (num_families * avg_children_per_family) / (num_families - num_childless_families : ℤ) = 2.5 :=
by
  intros num_families avg_children_per_family num_childless_families h1 h2 h3
  sorry

end average_children_in_families_with_children_l773_773949


namespace sufficient_but_not_necessary_l773_773442

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a > 2 → a^2 > 2 * a) ∧ ¬(a^2 > 2 * a → a > 2) :=
by
  sorry

end sufficient_but_not_necessary_l773_773442


namespace f_at_3_l773_773509

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x - 1

-- The theorem to prove
theorem f_at_3 : f 3 = 5 := sorry

end f_at_3_l773_773509


namespace trigonometric_expression_range_l773_773889

theorem trigonometric_expression_range 
  (A B C a b c : ℝ)
  (ha2b : 2 * a * cos C = 2 * b - c)
  (h1 : A + B + C = π)
  (h2 : sin A / a = sin B / b)
  (h3 : sin A / a = sin C / c)
  : -1 < (1 - 2 * cos 2 * C / (1 + tan C)) ∧ (1 - 2 * cos 2 * C / (1 + tan C)) ≤ sqrt 2 :=
by
  sorry

end trigonometric_expression_range_l773_773889


namespace y_is_less_than_x_by_9444_percent_l773_773338

theorem y_is_less_than_x_by_9444_percent (x y : ℝ) (h : x = 18 * y) : (x - y) / x * 100 = 94.44 :=
by
  sorry

end y_is_less_than_x_by_9444_percent_l773_773338


namespace max_ranked_players_in_tournament_l773_773521

theorem max_ranked_players_in_tournament :
  ∃ (k : ℕ), k ≤ 30 ∧ k * 0.6 * 29 ≤ 435 ∧ k = 24 :=
begin
  use 24,
  split,
  { -- k ≤ 30
    exact nat.le_of_lt (by norm_num), },
  split,
  { -- k * 17.4 ≤ 435
    calc
      24 * (0.6 * 29) = 24 * 17.4 : by norm_num
      ... ≤ 435 : by norm_num, },
  { -- k = 24
    refl, }
end

end max_ranked_players_in_tournament_l773_773521


namespace number_of_five_digit_sum_two_l773_773250

theorem number_of_five_digit_sum_two : 
  (∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (n.digits.sum = 2)) = 5 := 
sorry

end number_of_five_digit_sum_two_l773_773250


namespace max_sum_three_largest_angles_of_pentagon_l773_773880

-- Defining the internal angles and the conditions
variables (x y : ℝ)
def angle_sum_pentagon := 3 * x + 2 * y = 540
def similar_triangles := x + (x + d) + (x + 2 * d) = 180

-- The theorem to prove
theorem max_sum_three_largest_angles_of_pentagon (hx : angle_sum_pentagon x y) (hd : similar_triangles x) : 
  100 + 120 + 120 = 340 := 
by 
  sorry

end max_sum_three_largest_angles_of_pentagon_l773_773880


namespace weight_of_bowling_ball_l773_773219

-- We define the given conditions
def bowlingBalls := 10
def canoes := 4
def weightOfCanoe := 35

-- We state the theorem we want to prove
theorem weight_of_bowling_ball:
    (canoes * weightOfCanoe) / bowlingBalls = 14 :=
by
  -- Additional needed definitions
  let weightOfCanoes := canoes * weightOfCanoe
  have weightEquality : weightOfCanoes = 140 := by sorry  -- Calculating the total weight of the canoes
  -- Final division to find the weight of one bowling ball
  have weightOfOneBall := weightEquality / bowlingBalls
  show weightOfOneBall = 14 from sorry
  sorry

end weight_of_bowling_ball_l773_773219


namespace even_periodic_log_function_inequality_l773_773803

theorem even_periodic_log_function_inequality :
  (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) (h_periodic : ∀ x, f x = f (x + 2))
  (h_def : ∀ x, 0 < x ∧ x < 1 → f x = Real.log x / Real.log 0.5) :
  f (-1/2) > f (7/5) ∧ f (7/5) > f (4/3) :=
by
  sorry

end even_periodic_log_function_inequality_l773_773803


namespace find_c_value_l773_773965

theorem find_c_value (p q : ℕ) (h_simplest_form : Nat.gcd p q = 1)
                     (h_interval : 7 / 10 < p / q ∧ p / q < 11 / 15)
                     (h_minimal_q : ∀ r : ℕ, (7 / 10 < p / r ∧ p / r < 11 / 15) → q ≤ r) :
  c = p * q := by
  sorry

# Given information encoded in Lean as assertions:
# h_simplest_form: gcd(p, q) = 1
# h_interval: 7/10 < p/q < 11/15
# h_minimal_q: q is the smallest possible integer q for which the fraction p/q is in its simplest form and lies in the interval.

end find_c_value_l773_773965


namespace books_borrowed_after_lunchtime_l773_773076

variable (B0 : ℕ) (B_lb : ℕ) (A : ℕ) (B_e : ℕ)

theorem books_borrowed_after_lunchtime : B0 = 100 → B_lb = 50 → A = 40 → B_e = 60 → 90 - B_e = 30 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  rfl

end books_borrowed_after_lunchtime_l773_773076


namespace exists_k_with_three_different_real_roots_exists_k_with_two_different_real_roots_l773_773430

noncomputable def equation (x : ℝ) (k : ℝ) := x^2 - 2 * |x| - (2 * k + 1)^2

theorem exists_k_with_three_different_real_roots :
  ∃ k : ℝ, ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ equation x1 k = 0 ∧ equation x2 k = 0 ∧ equation x3 k = 0 :=
sorry

theorem exists_k_with_two_different_real_roots :
  ∃ k : ℝ, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ equation x1 k = 0 ∧ equation x2 k = 0 :=
sorry

end exists_k_with_three_different_real_roots_exists_k_with_two_different_real_roots_l773_773430


namespace eight_pow_neg_x_eq_one_eighth_l773_773504

theorem eight_pow_neg_x_eq_one_eighth (x : ℝ) (h : 8^(2 * x) = 64) : 8^(-x) = 1 / 8 :=
sorry

end eight_pow_neg_x_eq_one_eighth_l773_773504


namespace P_inter_Q_l773_773064

def P : set ℝ := { x | |x| * |x - 1| ≤ 1 }

def Q : set ℕ := { x | true }

theorem P_inter_Q : P ∩ Q = {0, 1, 2} :=
by {
  sorry
}

end P_inter_Q_l773_773064


namespace inequality_sin_tan_l773_773087

theorem inequality_sin_tan (x y : ℝ) :
  (sin (real.pi * 50/180))^x - (tan (real.pi * 50/180))^x ≤ (sin (real.pi * 50/180))^(-y) - (tan (real.pi * 50/180))^(-y) →
  x + y ≥ 0 :=
by
  sorry

end inequality_sin_tan_l773_773087


namespace math_pages_l773_773199

def total_pages := 7
def reading_pages := 2

theorem math_pages : total_pages - reading_pages = 5 := by
  sorry

end math_pages_l773_773199


namespace square_difference_example_l773_773286

theorem square_difference_example : 601^2 - 599^2 = 2400 := 
by sorry

end square_difference_example_l773_773286


namespace parents_survey_l773_773874

theorem parents_survey (W M : ℚ) 
  (h1 : 3/4 * W + 9/10 * M = 84) 
  (h2 : W + M = 100) :
  W = 40 :=
by
  sorry

end parents_survey_l773_773874


namespace eqD_is_linear_l773_773656

-- Definitions for the given equations
def eqA (x y : ℝ) : Prop := 3 * x - 2 * y = 1
def eqB (x : ℝ) : Prop := 1 + (1 / x) = x
def eqC (x : ℝ) : Prop := x^2 = 9
def eqD (x : ℝ) : Prop := 2 * x - 3 = 5

-- Definition of a linear equation in one variable
def isLinear (eq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, (∀ x : ℝ, eq x ↔ a * x + b = c)

-- Theorem stating that eqD is a linear equation
theorem eqD_is_linear : isLinear eqD :=
  sorry

end eqD_is_linear_l773_773656


namespace train_length_is_300_l773_773351

noncomputable def length_of_train (V L : ℝ) : Prop :=
  (L = V * 18) ∧ (L + 500 = V * 48)

theorem train_length_is_300
  (V : ℝ) (L : ℝ) (h : length_of_train V L) : L = 300 :=
by
  sorry

end train_length_is_300_l773_773351


namespace density_conversion_to_carats_per_cubic_inch_l773_773606

noncomputable def density_conversion_constant : ℝ := (3.5 * 5 * (2.54)^3)

theorem density_conversion_to_carats_per_cubic_inch :
  density_conversion_constant ≈ 287 :=
by
  sorry

end density_conversion_to_carats_per_cubic_inch_l773_773606


namespace valid_start_days_l773_773908

-- Define the days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Define function to get the redemption day after a number of weeks
def redemption_day (start : Day) (weeks : ℕ) : Day :=
  match start with
  | Sunday => Day.casesOn weeks Sunday Monday Tuesday Wednesday Thursday Friday Saturday
  | Monday => Day.casesOn weeks Monday Tuesday Wednesday Thursday Friday Saturday Sunday
  | Tuesday => Day.casesOn weeks Tuesday Wednesday Thursday Friday Saturday Sunday Monday
  | Wednesday => Day.casesOn weeks Wednesday Thursday Friday Saturday Sunday Monday Tuesday
  | Thursday => Day.casesOn weeks Thursday Friday Saturday Sunday Monday Tuesday Wednesday
  | Friday => Day.casesOn weeks Friday Saturday Sunday Monday Tuesday Wednesday Thursday
  | Saturday => Day.casesOn weeks Saturday Sunday Monday Tuesday Wednesday Thursday Friday

-- Prove that the starting days are Sunday, Monday, Tuesday, or Thursday
theorem valid_start_days (start : Day) :
  (redemption_day start 0 ≠ Wednesday) ∧
  (redemption_day start 0 ≠ Saturday) ∧
  (redemption_day start 1 ≠ Wednesday) ∧
  (redemption_day start 1 ≠ Saturday) ∧
  (redemption_day start 2 ≠ Wednesday) ∧
  (redemption_day start 2 ≠ Saturday) ∧
  (redemption_day start 3 ≠ Wednesday) ∧
  (redemption_day start 3 ≠ Saturday) ∧
  (redemption_day start 4 ≠ Wednesday) ∧
  (redemption_day start 4 ≠ Saturday) ∧
  (redemption_day start 5 ≠ Wednesday) ∧
  (redemption_day start 5 ≠ Saturday) ∧
  (redemption_day start 6 ≠ Wednesday) ∧
  (redemption_day start 6 ≠ Saturday) ∧
  (redemption_day start 7 ≠ Wednesday) ∧
  (redemption_day start 7 ≠ Saturday) ↔
  start = Sunday ∨
  start = Monday ∨
  start = Tuesday ∨
  start = Thursday :=
sorry

end valid_start_days_l773_773908


namespace matrix_sum_ge_half_n_sq_l773_773168

open Matrix

theorem matrix_sum_ge_half_n_sq {n : ℕ} (A : Matrix (Fin n) (Fin n) ℕ) 
  (h : ∀ i j, A i j = 0 → (∑ k, A i k + ∑ k, A k j) ≥ n) : 
  (∑ i j, A i j) ≥ n^2 / 2 :=
sorry

end matrix_sum_ge_half_n_sq_l773_773168


namespace similar_triangle_side_l773_773345

theorem similar_triangle_side (a b : ℝ) (hypotenuse1 hypotenuse2 : ℝ) (scaling_factor : ℝ) 
  (h1 : a = 15) (h2 : hypotenuse1 = 17) (h3 : hypotenuse2 = 34) 
  (h4 : similar_triangles : ∀ (k : ℝ), scaling_factor = hypotenuse2 / hypotenuse1 := 34 / 17)
  (h5 : scaling_factor = 2) :
  similar_triangles * a = 30 :=
by
  sorry

end similar_triangle_side_l773_773345


namespace find_x_when_y1_eq_y2_find_x_range_y1_gt_y2_case1_find_x_range_y1_gt_y2_case2_l773_773770

variable {a x : ℝ}

-- Defining conditions
def y1 (a x : ℝ) : ℝ := log a (3 * x + 1)
def y2 (a x : ℝ) : ℝ := log a (-3 * x)
def a_pos (a : ℝ) : Prop := a > 0
def a_not_one (a : ℝ) : Prop := a ≠ 1

-- Statement 1: Finding x when y1 = y2
theorem find_x_when_y1_eq_y2 (h1 : a_pos a) (h2 : a_not_one a) (h3 : y1 a x = y2 a x) : x = -1 / 6 := 
sorry

-- Statement 2: Find the range of x when y1 > y2
theorem find_x_range_y1_gt_y2_case1 (h1 : a_pos a) (h2 : a < 1) (h3 : y1 a x > y2 a x) : -1 / 3 < x ∧ x < -1 / 6 := 
sorry

theorem find_x_range_y1_gt_y2_case2 (h1 : a_pos a) (h2 : 1 < a) (h3 : y1 a x > y2 a x) : -1 / 6 < x ∧ x < 0 := 
sorry

end find_x_when_y1_eq_y2_find_x_range_y1_gt_y2_case1_find_x_range_y1_gt_y2_case2_l773_773770


namespace cost_of_fencing_per_meter_l773_773242

-- Define the conditions given in the problem
def length_plot : ℝ := 57
def breadth_plot : ℝ := length_plot - 14
def total_cost : ℝ := 5300

-- Calculate the perimeter
def perimeter_plot : ℝ := 2 * (length_plot + breadth_plot)

-- Define the cost of fencing per meter
def cost_per_meter : ℝ := total_cost / perimeter_plot

-- The theorem stating the problem
theorem cost_of_fencing_per_meter : cost_per_meter = 26.5 := by
  unfold cost_per_meter perimeter_plot breadth_plot length_plot total_cost
  -- Perform calculations (skipped)
  sorry

end cost_of_fencing_per_meter_l773_773242


namespace theta_constant_implies_cylinder_l773_773528

noncomputable def shape_given_theta_constant (ρ θ φ : ℝ) (c : ℝ) : Prop :=
θ = c

theorem theta_constant_implies_cylinder (ρ θ φ c : ℝ) :
  shape_given_theta_constant ρ θ φ c → 
  is_cylinder ρ θ φ :=
sorry

end theta_constant_implies_cylinder_l773_773528


namespace range_of_x_l773_773443

theorem range_of_x (a b x : ℝ) (h1 : a + b = 1) (h2 : 0 < a) (h3 : 0 < b) :
  (1 / a + 4 / b ≥ |2 * x - 1| - |x + 1|) → (-7 ≤ x ∧ x ≤ 11) :=
by
  -- we provide the exact statement we aim to prove.
  sorry

end range_of_x_l773_773443


namespace convex_polygon_sides_eq_49_l773_773866

theorem convex_polygon_sides_eq_49 
  (n : ℕ)
  (hn : n > 0) 
  (h : (n * (n - 3)) / 2 = 23 * n) : n = 49 :=
sorry

end convex_polygon_sides_eq_49_l773_773866


namespace catFinishesOnMondayNextWeek_l773_773580

def morningConsumptionDaily (day : String) : ℚ := if day = "Wednesday" then 1 / 3 else 1 / 4
def eveningConsumptionDaily : ℚ := 1 / 6

def totalDailyConsumption (day : String) : ℚ :=
  morningConsumptionDaily day + eveningConsumptionDaily

-- List of days in order
def week : List String := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

-- Total food available initially
def totalInitialFood : ℚ := 8

-- Function to calculate total food consumed until a given day
def foodConsumedUntil (day : String) : ℚ :=
  week.takeWhile (· != day) |>.foldl (λ acc d => acc + totalDailyConsumption d) 0

-- Function to determine the day when 8 cans are completely consumed
def finishingDay : String :=
  match week.find? (λ day => foodConsumedUntil day + totalDailyConsumption day = totalInitialFood) with
  | some day => day
  | none => "Monday"  -- If no exact match is found in the first week, it is Monday of the next week

theorem catFinishesOnMondayNextWeek :
  finishingDay = "Monday" := by
  sorry

end catFinishesOnMondayNextWeek_l773_773580


namespace number_of_distinct_prime_factors_l773_773154

-- Given: A is the product of the divisors of 60.
-- Prove: The number of distinct prime factors of A is 3.

theorem number_of_distinct_prime_factors (A : ℕ) 
  (h1 : A = ∏ d in (finset.filter (λ d, d ∣ 60) (finset.range (60 + 1))), d) :
  (factors_multiset A).to_finset.card = 3 := 
sorry

end number_of_distinct_prime_factors_l773_773154


namespace area_between_concentric_circles_l773_773999

theorem area_between_concentric_circles 
  {O G E F : Type*} [MetricSpace O] [MetricSpace G] [MetricSpace E] [MetricSpace F] 
  (radius_outer : ℝ) (chord_len : ℝ) (tangent_len : ℝ) 
  (h_outer_radius : radius_outer = 12) 
  (h_chord_leng : chord_len = 20) 
  (h_tangent_point : tangent_len = Real.sqrt 44) :
  ∀ (r_outer r_inner : ℝ), 
  (π * r_outer^2 - π * r_inner^2 = 100 * π) :=
begin
  intros r_outer r_inner,
  sorry
end

end area_between_concentric_circles_l773_773999


namespace distance_origin_to_midpoint_l773_773519

/-- 
  Define the points A and B and the corresponding midpoint function.
  Then show the distance from the origin to this midpoint is 0.
-/
def pointA : ℝ × ℝ := (-6, 8)
def pointB : ℝ × ℝ := (6, -8)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def distance_from_origin (point : ℝ × ℝ) : ℝ :=
  Real.sqrt (point.1 ^ 2 + point.2 ^ 2)

theorem distance_origin_to_midpoint : 
  distance_from_origin (midpoint pointA pointB) = 0 :=
by
  sorry

end distance_origin_to_midpoint_l773_773519


namespace partition_rectangle_l773_773763

theorem partition_rectangle (n : ℕ) :
  (1 ≤ n ∧ n ≤ 998 ∨ n ≥ 3990) →
  ∃ strips : list ℕ, strips.nodup ∧ (∀ k ∈ strips, k > 0) ∧ (list.sum strips = 1995 * n) ∧
  (∀ k ∈ strips, ∃ m ≤ 1995, strips.count m * m = 1995 * n) :=
by
  sorry

end partition_rectangle_l773_773763


namespace product_divisors_60_prime_factors_l773_773156

theorem product_divisors_60_prime_factors : 
  ∃ (A : ℕ), (A = ∏ d in (Finset.filter (λ d, d ∣ 60) (Finset.range (60+1))), d) ∧ 
             (nat.factors A).to_finset.card = 3 := 
begin
  sorry
end

end product_divisors_60_prime_factors_l773_773156


namespace all_numbers_rational_l773_773297

-- Define the mathematical operations for the problem
def fourth_root (x : ℝ) : ℝ := x ^ (1 / 4)
def square_root (x : ℝ) : ℝ := x ^ (1 / 2)
def cube_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem all_numbers_rational :
    (∃ x1 : ℚ, fourth_root 81 = x1) ∧
    (∃ x2 : ℚ, square_root 0.64 = x2) ∧
    (∃ x3 : ℚ, cube_root 0.001 = x3) ∧
    (∃ x4 : ℚ, (cube_root 8) * (square_root ((0.25)⁻¹)) = x4) :=
  sorry

end all_numbers_rational_l773_773297


namespace smallest_n_for_club_l773_773683

namespace ClubProblem

def mutual_acquaintance_exists (G : Type) [SimpleGraph G] (n : ℕ) : Prop :=
  ∃ (A B C D : G), A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ A ≠ C ∧ B ≠ D ∧
                    G.adj A B ∧ G.adj B C ∧ G.adj C D ∧ G.adj D A ∧ G.adj A C ∧ G.adj B D

theorem smallest_n_for_club (n people : ℕ) (H : people = 99) (condition : ∀ p : Fin people, ∃ (k : Fin people), k.1 > n) :
  mutual_acquaintance_exists (Fin people) 66 :=
sorry

end ClubProblem

end smallest_n_for_club_l773_773683


namespace validSetsCount_l773_773015

-- Define the set from which numbers are chosen
def set123456789 : Set ℕ := {2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define the condition of having number 6 in the set
def includesSix (s : Set ℕ) : Prop := 6 ∈ s

-- Define the condition of having distinct elements
def distinctElements (s : Set ℕ) : Prop := s.card = 3 ∧ (∀ a b, a ∈ s → b ∈ s → a ≠ b → a ≠ b)

-- Define the condition of summing to 18
def sumToEighteen (s : Set ℕ) : Prop := s.sum id = 18

-- Define what it means to be a valid set
def isValidSet (s : Set ℕ) : Prop := 
  includesSix s ∧
  distinctElements s ∧
  sumToEighteen s

-- Prove that there are exactly 4 such sets
theorem validSetsCount : {s : Set ℕ | s ⊆ set123456789 ∧ isValidSet s}.card = 4 := 
by sorry

end validSetsCount_l773_773015


namespace min_value_expression_l773_773417

theorem min_value_expression : ∃ x y z : ℝ, (3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + z^2 + 6 * z + 10) = -7 / 2 :=
by sorry

end min_value_expression_l773_773417


namespace incorrect_major_premise_l773_773339

-- Definitions
def is_extremum (f : ℝ → ℝ) (x_0 : ℝ) : Prop :=
  (∀ x, (x < x_0 → f(x) > f(x_0)) ∧ (x > x_0 → f(x) > f(x_0))) ∨
  (∀ x, (x < x_0 → f(x) < f(x_0)) ∧ (x > x_0 → f(x) < f(x_0)))

-- Problem statement
theorem incorrect_major_premise : ¬ ∀ (f : ℝ → ℝ) (hf : Differentiable ℝ f) (x_0 : ℝ), f' x_0 = 0 → is_extremum f x_0 :=
by
  sorry

end incorrect_major_premise_l773_773339


namespace find_cost_per_day_first_week_l773_773954

noncomputable def cost_per_day_first_week (x : ℝ) := 
  let first_week_days := 7
  let additional_day_cost := 13
  let total_days := 23
  let total_cost := 334
  in (7 * x) + ((total_days - first_week_days) * additional_day_cost) = total_cost

theorem find_cost_per_day_first_week : 
  ∃ x : ℝ, cost_per_day_first_week x ∧ x = 18 :=
by
  sorry

end find_cost_per_day_first_week_l773_773954


namespace complex_conjugate_of_z_l773_773771

noncomputable def z : ℂ := (7 + 5*I) / (1 + I)

theorem complex_conjugate_of_z
  (h : z * (1 + I) = 7 + 5*I) :
  conj z = 6 + I := sorry

end complex_conjugate_of_z_l773_773771


namespace inclination_angle_l773_773479

theorem inclination_angle (a : ℝ) (ha : a > 0) :
  ∃ α : ℝ, α = π - arctan (-a) ∧ (∃ k : ℝ, k = -a ∧ tan α = k) :=
sorry

end inclination_angle_l773_773479


namespace number_of_ears_pierced_l773_773542

-- Definitions for the conditions
def nosePiercingPrice : ℝ := 20
def earPiercingPrice := nosePiercingPrice + 0.5 * nosePiercingPrice
def totalAmountMade : ℝ := 390
def nosesPierced : ℕ := 6
def totalFromNoses := nosesPierced * nosePiercingPrice
def totalFromEars := totalAmountMade - totalFromNoses

-- The proof statement
theorem number_of_ears_pierced : totalFromEars / earPiercingPrice = 9 := by
  sorry

end number_of_ears_pierced_l773_773542


namespace part_one_part_two_part_three_l773_773967

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + real.log x

theorem part_one (a : ℝ) :
  (2 - 2 * a + 1 = -2) → (a = 5 / 2) :=
by
  sorry

theorem part_two (a : ℝ) :
  (∀ x > 0, 2 * x - 2 * a + 1 / x) →
  ((a ≤ real.sqrt 2 → ∀ x > 0, 2 * x - 2 * a + 1 / x ≥ 0) ∧
   (a > real.sqrt 2 → ∃ x > 0, 2 * x - 2 * a + 1 / x = 0)) :=
by
  sorry

theorem part_three (a : ℝ) :
  (∀ x ∈ set.Ioo 0 real.exp(1), 2 * x * real.log x ≥ -x^2 + a * x - 3) →
  (a ≤ 4) :=
by
  sorry

end part_one_part_two_part_three_l773_773967


namespace problem_statement_l773_773463

-- Definition of geometric sequence sum forming arithmetic sequence with a fixed element
def is_geom_seq (a : ℕ → ℝ) (s : ℕ → ℝ) : Prop :=
  ∀ n, s n = (range (n + 1)).sum a

def is_arith_seq (x y z : ℝ) : Prop :=
  2 * y = x + z

-- Given conditions
def cond1 (a : ℕ → ℝ) (s : ℕ → ℝ) : Prop :=
  is_geom_seq a s ∧ is_arith_seq (2 : ℝ) (s (2 : ℕ)) (2.0 : ℝ)

-- Proving statements
theorem problem_statement :
  ∃ a s, cond1 a s ∧
  (∀ n, s n = 2 ^ n + (-(2 : ℝ)) / 2) ∧
  (∀ n, a n = 2 ^ (n - 1)) ∧
  (∀ n, (range (n + 1)).sum (λ k, (2 * k - 1) * a k) = 3 + (2 * n - 3) * 2 ^ n) :=
by {
  sorry -- Proof steps would be provided here
}

end problem_statement_l773_773463


namespace valid_approach_order_l773_773926

section approach_order

variables (order : list char) -- Representing the order as a list of characters ['M', 'G', 'B']
variables (M_last : order = ['G', 'X', 'M'] → False) -- M_1: If M is last, G is not first
variables (M_first : order = ['M', 'X', 'G'] → False) -- M_2: If M is first, G is not last
variables (B_last : order = ['X', 'G', 'B'] → (order ≠ ['X', 'G', 'M'])) -- B_1: If B is last, M does not follow G
variables (B_first : order = ['B', 'X', 'M'] → False) -- B_2: If B is first, M does not precede G
variables (G_middle : order = ['X', 'G', 'X'] → (order ≠ ['M', 'G', 'B'] ∧ order ≠ ['G', 'B', 'M'])) -- G_1: If G is neither first nor last, M does not precede B

theorem valid_approach_order : order = ['M', 'G', 'B'] ∨ order = ['B', 'G', 'M'] :=
sorry

end approach_order

end valid_approach_order_l773_773926


namespace a_share_proof_l773_773674
noncomputable def a_share_of_profits (a_initial_investment b_initial_investment a_withdrawal b_advance total_profits : ℝ) :=
  let a_investment_6_months := a_initial_investment * 6
  let a_investment_8_months := (a_initial_investment - a_withdrawal) * 8
  let a_total_investment := a_investment_6_months + a_investment_8_months

  let b_investment_6_months := b_initial_investment * 6
  let b_investment_8_months := (b_initial_investment + b_advance) * 8
  let b_total_investment := b_investment_6_months + b_investment_8_months

  let ratio := a_total_investment / b_total_investment

  total_profits / (1 + 1 / ratio)

theorem a_share_proof :
  a_share_of_profits 5000 7000 1500 2000 1384 ≈ 466.95 := by
  sorry

end a_share_proof_l773_773674


namespace sum_ck_squared_l773_773383
noncomputable def c (k : ℕ) : ℚ :=
have hyp : (k + 1 : ℚ) ≠ 0 := by norm_num,
k + 1 / (k + 1 + 1 / (k + 2 + 1 / (k + 3 + ...)))

theorem sum_ck_squared (n : ℕ) (hn : n > 2) : 
  ∑ k in finset.range n, (c k)^2 = n * (2 * n^2 + 4 * n + 3) / 6 := sorry

end sum_ck_squared_l773_773383


namespace adults_on_bus_l773_773634

def total_passengers : ℕ := 360
def fraction_children : ℚ := 3 / 7
def fraction_adults : ℚ := 1 - fraction_children

theorem adults_on_bus : (total_passengers * fraction_adults).natAbs = 205 := by
  sorry

end adults_on_bus_l773_773634


namespace number_of_small_jars_l773_773304

theorem number_of_small_jars (S L : ℕ) (h1 : S + L = 100) (h2 : 3 * S + 5 * L = 376) : S = 62 := 
sorry

end number_of_small_jars_l773_773304


namespace slope_product_is_constant_quadrilateral_OAPB_is_parallelogram_l773_773468

-- Define the Ellipse and Line passing through given conditions
variables {m : ℝ} (h_m : m > 0)
def ellipse (x y : ℝ) := 9 * x^2 + y^2 = m^2
def line (k b : ℝ) (h_k : k ≠ 0) (h_b : b ≠ 0) := ∃ x y : ℝ, y = k * x + b

-- Question 1: Prove the product of the slopes is -9
theorem slope_product_is_constant {k b : ℝ} (h_k : k ≠ 0) (h_b : b ≠ 0) :
  ∀ x_M y_M : ℝ, ∀ x y : ℝ, line k b h_k h_b →
    ellipse x_M y_M →
    ellipse x y →
    let slope_OM := y_M / x_M
    let slope_l := k
    in slope_OM * slope_l = -9 :=
sorry

-- Question 2: Prove conditions for OAPB being a parallelogram
theorem quadrilateral_OAPB_is_parallelogram {
  k b : ℝ} (h_k : k ≠ 0) (h_b : b ≠ 0) :
  line k b h_k h_b →
  ∃ l_p m_p : ℝ, l_p = m / 3 ∧ m_p = m →
  let slope_l := k
  in (slope_l = 4 - real.sqrt 7 ∨ slope_l = 4 + real.sqrt 7) :=
sorry

end slope_product_is_constant_quadrilateral_OAPB_is_parallelogram_l773_773468


namespace inheritance_calculation_l773_773544

theorem inheritance_calculation
  (x : ℝ)
  (h1 : 0.25 * x + 0.15 * (0.75 * x) = 14000) :
  x = 38600 := by
  sorry

end inheritance_calculation_l773_773544


namespace find_a_and_union_set_l773_773550

theorem find_a_and_union_set (a : ℝ) 
  (A : Set ℝ) (B : Set ℝ) 
  (hA : A = {-3, a + 1}) 
  (hB : B = {2 * a - 1, a ^ 2 + 1}) 
  (h_inter : A ∩ B = {3}) : 
  a = 2 ∧ A ∪ B = {-3, 3, 5} :=
by
  sorry

end find_a_and_union_set_l773_773550


namespace evaluate_expression_l773_773397

noncomputable def ceil_div (a b : ℚ) : ℤ := ⌈a / b⌉

theorem evaluate_expression :
  let x := 43
  let y := 13
  let z1 := 45
  let z2 := 29
  let a1 := 56
  let b1 := 13
  let a2 := 13 * 29
  let b2 := 45 in
  (ceil_div x y - ceil_div z1 z2) / (ceil_div a1 b1 + ceil_div a2 b2) = 1 / 7 :=
by
  sorry

end evaluate_expression_l773_773397


namespace a_lt_zero_not_necessary_nor_sufficient_l773_773996

theorem a_lt_zero_not_necessary_nor_sufficient (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x - 1 < 0) → (a < 0) ↔ False :=
begin
  sorry
end

end a_lt_zero_not_necessary_nor_sufficient_l773_773996


namespace choose_subsets_condition_l773_773564

noncomputable def numberOfWays (I : Set ℕ) : ℕ := 
  let A_subsets := I.powerset
  let B_subsets := I.powerset
  let validPairs := (A_subsets ×ˢ B_subsets).filter 
    (λ ⟨A,B⟩, A.nonempty ∧ B.nonempty ∧ (∀ a ∈ A, ∀ b ∈ B, a < b))
  validPairs.card

theorem choose_subsets_condition (I : Set ℕ) (hI : I = {1, 2, 3, 4, 5}) : 
  numberOfWays I = 49 :=
by
  sorry

end choose_subsets_condition_l773_773564


namespace coefficient_x3_expansion_l773_773384

theorem coefficient_x3_expansion :
  ∀ {R : Type} [CommRing R], (1 + (X : R[X])^2) * (1 - X)^5 = R[X].sum (= term_with_coeff (-15 : R) X^3) :=
by
  sorry

end coefficient_x3_expansion_l773_773384


namespace arithmetic_sequence_formula_min_lambda_l773_773998

noncomputable def a_n (n : ℕ) : ℝ := (1 / 2 : ℝ) * n + (5 / 4 : ℝ)
noncomputable def T_n (n : ℕ) : ℝ := 16 * n / (7 * (2 * n + 7))
def S_4 : ℝ := 10
def a1 : ℝ := 7 / 4
def d : ℝ := 1 / 2

theorem arithmetic_sequence_formula
  (h1 : S_4 = 4 * a1 + 6 * d)
  (h2 : (a1 + 7 * d) ^ 2 = a1 * (a1 + 28 * d))
  : ∀ n : ℕ, a_n n = (1 / 2 : ℝ) * n + (5 / 4 : ℝ) :=
sorry

theorem min_lambda
  (h : ∀ n : ℕ, 7 * T_n n <= λ * (a_n (n + 1) + 1))
  : λ ≥ (256 / 225 : ℝ) :=
sorry

end arithmetic_sequence_formula_min_lambda_l773_773998


namespace someone_answers_no_l773_773191

theorem someone_answers_no (n : ℕ) (knight liar : ℕ) (seated : ℕ): 
(seated = 100) → 
(knight + liar = 100) →
(knight ≥ 1 ∧ liar ≥ 1) →
(∀ i : ℕ, i < 100 → 
  let left := (i + 90) % 100 in
  let right := (i + 10) % 100 in
  if knight_p i 
  then ∑ j in finset.range(21), cond (knight_p ((i + j - 10 + 100) % 100)) 1 0 > 10
  else ∑ j in finset.range(21), cond (liar_p ((i + j - 10 + 100) % 100)) 1 0 > 10) →
false :=
sorry

end someone_answers_no_l773_773191


namespace area_of_S3_is_16_over_9_l773_773450

-- Define the conditions as Lean definitions and the final proof statement.
def S1_area : ℝ := 36
def S1_side : ℝ := real.sqrt S1_area
def trisected (side_length : ℝ) : ℝ := side_length / 3
def diagonal (side_length : ℝ) : ℝ := side_length * real.sqrt 2
def S2_side : ℝ := trisected (diagonal S1_side)
def S2_area : ℝ := S2_side ^ 2
def S3_side : ℝ := trisected (diagonal S2_side)
def S3_area : ℝ := S3_side ^ 2

theorem area_of_S3_is_16_over_9 : S3_area = 16 / 9 := by
  sorry

end area_of_S3_is_16_over_9_l773_773450


namespace fly_distance_to_ceiling_l773_773643

theorem fly_distance_to_ceiling :
  ∀ (x y z : ℝ), 
  (x = 3) → 
  (y = 4) → 
  (z * z + 25 = 49) →
  z = 2 * Real.sqrt 6 :=
by
  sorry

end fly_distance_to_ceiling_l773_773643


namespace height_of_tree_in_8_years_in_inches_l773_773611

theorem height_of_tree_in_8_years_in_inches 
  (initial_height : ℕ) (annual_growth : ℕ) (years : ℕ) (feet_to_inches : ℕ) 
  (h_initial_height : initial_height = 52) 
  (h_annual_growth : annual_growth = 5) 
  (h_years : years = 8) 
  (h_feet_to_inches : feet_to_inches = 12) : 
  let total_growth := annual_growth * years in
  let final_height_in_feet := initial_height + total_growth in
  final_height_in_feet * feet_to_inches = 1104 :=
by
  sorry

end height_of_tree_in_8_years_in_inches_l773_773611


namespace shift_graph_f_shifted_function_l773_773811

open Real

noncomputable def mat_determinant (a₁ a₂ a₃ a₄ : ℝ) : ℝ := a₁ * a₄ - a₂ * a₃

noncomputable def function_f (x : ℝ) : ℝ :=
  mat_determinant (sin (π - x)) (sqrt 3) (cos (π + x)) 1

theorem shift_graph_f (x : ℝ) : 
  let f := function_f in 
  f x = 2 * sin (x + π / 3) := 
sorry

theorem shifted_function (x : ℝ) : 
  let g := (λ x, function_f (x - π / 3)) in 
  g x = 2 * sin x := 
sorry

end shift_graph_f_shifted_function_l773_773811


namespace exists_f_l773_773391

noncomputable def f : ℕ+ → ℕ+
| ⟨n⟩ =>
  if n % 2 = 1 then
    let k := (n - 1) / 4
    if n % 4 = 1 then ⟨4*k + 3⟩ else ⟨8*k + 2⟩
  else
    let n' := n / 2
    ⟨2 * (f ⟨n'⟩).val⟩

theorem exists_f : ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, f (f n) = ⟨2 * n.val⟩ :=
begin
  use f,
  intro n,
  sorry
end

end exists_f_l773_773391


namespace correct_sqrt_expression_l773_773293

theorem correct_sqrt_expression : 
  (∃ (a b : ℝ), a = sqrt (15 / 2) ∧ b = (1 / 2) * sqrt 30 ∧ a = b) :=
by
  sorry

end correct_sqrt_expression_l773_773293


namespace type1_type2_nsum_equiv_l773_773359

-- Definitions for Type 1 and Type 2 n-sums:

def is_type1_nsum (n : ℕ) (a : List ℕ) : Prop :=
  (a.sum = n) ∧ 
  (∀ i : ℕ, 0 ≤ i ∧ i < a.length - 2 → a.nth i > a.nth (i + 1) + a.nth (i + 2)) ∧ 
  (∀ j : ℕ, 0 ≤ j ∧ j = a.length - 2 → a.nth j > a.nth (j + 1))

def g_sequence : List ℕ := 
  let f (g : List ℕ) : List ℕ := g ++ [g.sum + 1]
  List.range n.foldl f [1, 2]

def is_type2_nsum (n : ℕ) (b : List ℕ) : Prop :=
  (b.sum = n) ∧ 
  (∀ i : ℕ, 0 ≤ i ∧ i < b.length - 1 → b.nth i ≥ b.nth (i + 1)) ∧ 
  (∀ x ∈ b, x ∈ g_sequence) ∧ 
  ∃ k : ℕ, b.head = g_sequence.nth k

-- Theorem to prove their equality in terms of count
theorem type1_type2_nsum_equiv (n : ℕ) (n_ge_one : n ≥ 1) : 
  ( ∃ A : Finset (List ℕ), 
    ( A.filter (is_type1_nsum n) ).card = 
    ( A.filter (is_type2_nsum n) ).card ) :=
  sorry

end type1_type2_nsum_equiv_l773_773359


namespace max_possible_x_l773_773167

theorem max_possible_x (x y z : ℝ) 
  (h1 : 3 * x + 2 * y + z = 10)
  (h2 : x * y + x * z + y * z = 6) :
  x ≤ 2 * Real.sqrt 5 / 5 :=
sorry

end max_possible_x_l773_773167


namespace polygon_area_is_odd_l773_773518

structure Polygon100 :=
  (vertices : Fin 100 → (ℤ × ℤ))
  (edgesParallel : ∀ i : Fin 100, 
      let (x1, y1) := vertices i in
      let (x2, y2) := vertices (i + 1) in (* Addition mod 100 *)
      (x1 = x2 ∧ abs (y1 - y2) % 2 = 1) ∨
      (y1 = y2 ∧ abs (x1 - x2) % 2 = 1))

def area (P : Polygon100) : ℤ := 
  let vertices := P.vertices 
  1/2 * abs (Finset.sum (Finset.finRange 100) (fun  i =>
    let (x_i, y_i) := vertices i
    let (x_next, y_next) := vertices ((i + 1) % 100)
    x_i * y_next - y_i * x_next))

theorem polygon_area_is_odd (P : Polygon100) : 
  2 ∣ area P := 
sorry

end polygon_area_is_odd_l773_773518


namespace total_amount_200_after_3_years_l773_773845

/-- Given conditions -/
def P₁ : ℝ := 150
def T₁ : ℝ := 6
def A₁ : ℝ := 210

def P₂ : ℝ := 200
def T₂ : ℝ := 3

/-- Simple interest formula -/
def interest (P R T : ℝ) : ℝ := (P * R * T) / 100

theorem total_amount_200_after_3_years :
  ∃ R : ℝ, A₁ = P₁ + interest P₁ R T₁ →
  P₂ + interest P₂ R T₂ = 240 :=
begin
  sorry
end

end total_amount_200_after_3_years_l773_773845


namespace weight_of_one_bowling_ball_l773_773221

def weight_of_one_canoe : ℕ := 35

def ten_bowling_balls_equal_four_canoes (W: ℕ) : Prop :=
  ∀ w, (10 * w = 4 * W)

theorem weight_of_one_bowling_ball (W: ℕ) (h : W = weight_of_one_canoe) : 
  (10 * 14 = 4 * W) → 14 = 140 / 10 :=
by
  intros H
  sorry

end weight_of_one_bowling_ball_l773_773221


namespace cost_per_quart_l773_773437

-- Defining the conditions
def pool_length : ℝ := 10
def pool_width : ℝ := 8
def pool_depth : ℝ := 6
def chlorine_per_cubic_feet : ℝ := 1 / 120
def total_amount_spent : ℝ := 12

-- Defining the volume of the pool
def pool_volume : ℝ := pool_length * pool_width * pool_depth

-- Defining the number of quarts needed
def quarts_needed : ℝ := pool_volume * chlorine_per_cubic_feet

-- The proof problem to be solved
theorem cost_per_quart : (total_amount_spent / quarts_needed) = 3 := by
  sorry

end cost_per_quart_l773_773437


namespace part1_part2_l773_773451

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1 + Real.log x) / x
noncomputable def g (x : ℝ) : ℝ := 3 * Real.exp (1 - x) + 1

theorem part1 (a : ℝ) (x : ℝ) (h : 0 < x) : (∀ x > 0, f a x ≤ Real.exp 1) → a ≤ 1 := 
sorry

theorem part2 (a : ℝ) : (∃! x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f a x1 = g x1 ∧ f a x2 = g x2 ∧ f a x3 = g x3) → a = 3 :=
sorry

end part1_part2_l773_773451


namespace find_n_l773_773752

theorem find_n : ∃ n : ℤ, -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * Real.pi / 180) = Real.cos (612 * Real.pi / 180) :=
by
  use -18
  split
  linarith
  split
  linarith
  -- This is where the proof that Real.sin (-18 * Real.pi / 180) = Real.cos (612 * Real.pi / 180) would go
  sorry

end find_n_l773_773752


namespace complement_of_angle_A_l773_773018

theorem complement_of_angle_A (A : ℝ) (h : A = 76) : 90 - A = 14 := by
  sorry

end complement_of_angle_A_l773_773018


namespace P_neg_2015_zero_l773_773910

noncomputable def polynomial_nonconstant : Type := sorry
noncomputable def P : polynomial_nonconstant := sorry

theorem P_neg_2015_zero (P : polynomial_nonconstant) (pos_coeffs : ∀ i, P.coeff i > 0)
  (non_constant : P.degree > 0)
  (div_condition : ∀ n : ℕ, P.eval n ∣ P.eval (P.eval n - 2015)) :
  P.eval (-2015) = 0 :=
sorry

end P_neg_2015_zero_l773_773910


namespace number_of_schools_in_lansing_l773_773906

theorem number_of_schools_in_lansing (total_students : ℝ) (average_students_per_school : ℝ) (h1 : total_students = 247.0) (h2 : average_students_per_school = 9.88) : total_students / average_students_per_school ≈ 25 :=
by
  sorry

end number_of_schools_in_lansing_l773_773906


namespace certain_number_is_310_l773_773282

theorem certain_number_is_310 (x : ℤ) (h : 3005 - x + 10 = 2705) : x = 310 :=
by
  sorry

end certain_number_is_310_l773_773282


namespace domain_of_function_l773_773607

theorem domain_of_function :
  (∀ x : ℝ, (2 - x) ≥ 0 ∧ (2x - 1) / (3 - x) > 0 ↔ (1/2 < x ∧ x ≤ 2)) :=
by
  sorry

end domain_of_function_l773_773607


namespace value_of_a_minus_b_l773_773904

-- Define the problem condition using Lean definitions
noncomputable def identity_holds (a b : ℚ) (x : ℚ) : Prop :=
  (a / (10^x - 3) + b / (10^x + 4) = (3 * 10^x - 1) / ((10^x - 3) * (10^x + 4)))

-- Define the theorem stating that a - b = -5/7
theorem value_of_a_minus_b (a b : ℚ) (x : ℚ) (hx : 0 < x) (h : identity_holds a b x) :
  a - b = -5/7 :=
sorry

end value_of_a_minus_b_l773_773904


namespace length_of_side_l773_773520

theorem length_of_side {r : ℝ} (h1 : r = 8) (h2 : ∀ (A B C : ℝ), is_right_triangle A B C) :
  ∃ (AB : ℝ), AB = 16 + 16 * Real.sqrt 3 :=
by
  use 16 + 16 * Real.sqrt 3
  sorry

end length_of_side_l773_773520


namespace no_nonzero_integer_solution_l773_773197

theorem no_nonzero_integer_solution 
(a b c n : ℤ) (h : 6 * (6 * a ^ 2 + 3 * b ^ 2 + c ^ 2) = 5 * n ^ 2) : 
a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := 
sorry

end no_nonzero_integer_solution_l773_773197


namespace animals_feet_l773_773337

theorem animals_feet (heads hens : ℕ) (H1 : heads = 44) (H2 : hens = 18) : 2 * hens + 4 * (heads - hens) = 140 :=
by
  rw [H1, H2]
  simp
  exact rfl

end animals_feet_l773_773337


namespace curve_properties_l773_773062

-- Definition of the parametric curve C
def curveC (α : ℝ) : ℝ × ℝ :=
  (3 + sqrt 5 * cos α, 1 + sqrt 5 * sin α)

-- The polar equation of the line
def polar_line (θ ρ : ℝ) : Prop :=
  sin θ - cos θ = 1 / ρ

-- The standard form of the circle
def circle_eq (x y : ℝ) : Prop :=
  (x - 3) ^ 2 + (y - 1) ^ 2 = 5

-- Converted polar equation of the circle
def polar_circle_eq (ρ θ : ℝ) : Prop :=
  ρ ^ 2 - 6 * ρ * cos θ - 2 * ρ * sin θ + 5 = 0

-- The distance from the center (3, 1) to the line
def distance (x1 y1 B C : ℝ) : ℝ :=
  abs (B * x1 + C * y1 + 1) / sqrt (B ^ 2 + C ^ 2)

-- Goal statement to prove the two properties
theorem curve_properties :
  (∀ α, circle_eq (3 + sqrt 5 * cos α) (1 + sqrt 5 * sin α)) ∧
  (∀ (θ ρ), polar_circle_eq ρ θ ↔ (∃ α : ℝ, (3 + sqrt 5 * cos α = ρ * cos θ ∧ 1 + sqrt 5 * sin α = ρ * sin θ))) ∧
  (distance 3 1 (-1) 1 = 3 * sqrt 2 / 2 ∧ chord_length 3 1 (3 * sqrt 2 / 2) sqrt 5 = sqrt 2) :=
by
  sorry

end curve_properties_l773_773062


namespace f_neg1_greater_f_1_l773_773459

-- Given that f is differentiable on R and f(x) = x^3 + 2xf'(2)
noncomputable def f (x : ℝ) : ℝ := 
  x^3 + 2 * x * (deriv f 2)

-- Proof statement that f(-1) > f(1)
theorem f_neg1_greater_f_1
  (h_diff : ∀ x : ℝ, Differentiable ℝ f)
  (h_eq : ∀ x : ℝ, f x = x^3 + 2 * x * deriv f 2) :
  f (-1) > f 1 :=
by
  sorry

end f_neg1_greater_f_1_l773_773459


namespace hyperbola_real_axis_length_l773_773415

theorem hyperbola_real_axis_length (x y : ℝ) :
  x^2 - y^2 / 9 = 1 → 2 = 2 :=
by
  sorry

end hyperbola_real_axis_length_l773_773415


namespace product_divisors_60_prime_factors_l773_773155

theorem product_divisors_60_prime_factors : 
  ∃ (A : ℕ), (A = ∏ d in (Finset.filter (λ d, d ∣ 60) (Finset.range (60+1))), d) ∧ 
             (nat.factors A).to_finset.card = 3 := 
begin
  sorry
end

end product_divisors_60_prime_factors_l773_773155


namespace reciprocal_sum_eq_l773_773917

-- Define the given pyramid and points with their relationships and metrics
structure Pyramid (α β d : ℝ) (O P A B C Q R S : Type) := 
  (center_base : O = center_triangle A B C) 
  (lateral_edge_angle : ∀ (P1 P2 : Type), angle (distance P1 P) (distance P2 P) = α)
  (plane_angle : angle (line PC) (plane PAB) = β)
  (dist_to_faces : ∀ (face : Type), distance O (plane face) = d)
  (intersects_PC : (line Q P ∩ line O) = S)
  (intersects_PA : (line R P ∩ line A) = Q)
  (intersects_PB : (line S P ∩ line B) = R)

-- Define the theorem to be proven
theorem reciprocal_sum_eq (α β d : ℝ) (O P A B C Q R S : Type) [Pyramid α β d O P A B C Q R S] : 
  (1 / (distance P Q)) + (1 / (distance P R)) + (1 / (distance P S)) = (sin β / d) :=
sorry

end reciprocal_sum_eq_l773_773917


namespace power_function_value_at_4_l773_773460

theorem power_function_value_at_4 :
  (∃ a : ℝ, ∀ x : ℝ, f x = x^a ∧ f (real.sqrt 2) = 2) → f 4 = 16 :=
  sorry

end power_function_value_at_4_l773_773460


namespace num_arithmetic_sequences_l773_773387

theorem num_arithmetic_sequences (d : ℕ) (x : ℕ)
  (h_sum : 8 * x + 28 * d = 1080)
  (h_no180 : ∀ i, x + i * d ≠ 180)
  (h_pos : ∀ i, 0 < x + i * d)
  (h_less160 : ∀ i, x + i * d < 160)
  (h_not_equiangular : d ≠ 0) :
  ∃ n : ℕ, n = 3 :=
by sorry

end num_arithmetic_sequences_l773_773387


namespace BXMY_cyclic_l773_773547

theorem BXMY_cyclic
  {A B C M P Q X Y : Type*}
  [geometry A B C M P Q X Y]
  (hM_midpoint : (AC M) ∧ (AC / 2))
  (hP_on_AM : P ∈ AM)
  (hQ_on_CM : Q ∈ CM)
  (hPQ : PQ = AC / 2)
  (hX_circum_ABQ : X ≠ B ∧ (X ∈ circle A B Q))
  (hY_circum_BCP : Y ≠ B ∧ (Y ∈ circle B C P)) :
  cyclic_quadrilateral B X M Y :=
sorry

end BXMY_cyclic_l773_773547


namespace median_of_81_consecutive_integers_l773_773248

theorem median_of_81_consecutive_integers (a : ℤ) (h_sum : ∑ i in finset.range 81, (a + i) = 3^9) : 
  (a + 40) = 243 :=
begin
  -- Utilize the sum formula for arithmetic sequences and given conditions to prove the median
  sorry
end

end median_of_81_consecutive_integers_l773_773248


namespace intersection_A_B_range_of_a_l773_773474

open Set Real

def f (x : ℝ) := sqrt (6 - 2 * x) + log (x + 2)

def A : Set ℝ := { x | -2 < x ∧ x ≤ 3 }
def B : Set ℝ := { x | x > 3 ∨ x < 2 }
def C (a : ℝ) : Set ℝ := { x | x < 2 * a + 1 }

theorem intersection_A_B :
  define f's domain as A ∧ define B ⧐ as B :=
  A ∩ B = { x | -2 < x ∧ x < 2 } :=
by sorry

theorem range_of_a (a : ℝ) :
  (B ∩ C a) = C a → a ≤ 1 / 2 :=
by sorry

end intersection_A_B_range_of_a_l773_773474


namespace problem1_l773_773371

theorem problem1 : 2 * Real.sin (Real.pi / 3) - 3 * Real.tan (Real.pi / 6) = 0 := by
  sorry

end problem1_l773_773371


namespace sum_of_special_factorial_last_two_digits_eq_five_l773_773285

-- Define a function to get the last two digits of a factorial
def last_two_digits_of_factorial (n : ℕ) : ℕ :=
  (∏ i in Finset.range (n + 1), i) % 100

-- Specific factorials as per the problem
def factorial_values : List ℕ := [1!, 1!, 2!, 3!, 5!, 8!, 13!, 21!, 34!, 55!, 55!]

-- Define the main function summing the last two digits of these factorials
def sum_of_last_two_digits (values : List ℕ) : ℕ :=
  List.sum (values.map last_two_digits_of_factorial)

-- The proof problem statement
theorem sum_of_special_factorial_last_two_digits_eq_five :
  sum_of_last_two_digits factorial_values = 5 :=
by
  sorry

end sum_of_special_factorial_last_two_digits_eq_five_l773_773285


namespace find_intersection_A_B_find_range_t_l773_773834

-- Define sets A, B, C
def A : Set ℝ := {y | ∃ x, (1 ≤ x ∧ x ≤ 2) ∧ y = 2^x}
def B : Set ℝ := {x | 0 < Real.log x ∧ Real.log x < 1}
def C (t : ℝ) : Set ℝ := {x | t + 1 < x ∧ x < 2 * t}

-- Theorem 1: Finding A ∩ B
theorem find_intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < Real.exp 1} := 
by
  sorry

-- Theorem 2: If A ∩ C = C, find the range of values for t
theorem find_range_t (t : ℝ) (h : A ∩ C t = C t) : t ≤ 2 :=
by
  sorry

end find_intersection_A_B_find_range_t_l773_773834


namespace best_fitting_model_l773_773879

theorem best_fitting_model (R2_1 R2_2 R2_3 R2_4 : ℝ)
  (h1 : R2_1 = 0.98)
  (h2 : R2_2 = 0.80)
  (h3 : R2_3 = 0.50)
  (h4 : R2_4 = 0.25) :
  R2_1 = 0.98 ∧ R2_1 > R2_2 ∧ R2_1 > R2_3 ∧ R2_1 > R2_4 :=
by { sorry }

end best_fitting_model_l773_773879


namespace solve_eq1_solve_eq2_solve_eq3_l773_773207

theorem solve_eq1 : ∃ x : ℝ, (5 * x + 2) * (4 - x) = 0 ↔ (x = -2/5 ∨ x = 4) :=
begin
  sorry
end

theorem solve_eq2 : ∃ x : ℝ, 2 * x^2 - 1 = 4 * x ↔ (x = (2 + Real.sqrt 6) / 2 ∨ x = (2 - Real.sqrt 6) / 2) :=
begin
  sorry
end

theorem solve_eq3 : ∃ x : ℝ, 2 * (x - 3)^2 = x^2 - 9 ↔ (x = 3 ∨ x = 9) :=
begin
  sorry
end

end solve_eq1_solve_eq2_solve_eq3_l773_773207


namespace cube_root_sqrt_64_eq_2_l773_773605

theorem cube_root_sqrt_64_eq_2 : Real.cbrt (Real.sqrt 64) = 2 := 
by 
  sorry

end cube_root_sqrt_64_eq_2_l773_773605


namespace quadrilateral_is_parallelogram_l773_773576

variables {A B C D O : Type} [AffineSpace ℝ A] [Affine ℝ A O]

-- Conditions
variables (midline_passes_through_intersection : ∃ (O : A), midpoint ℝ A B O ∧ midpoint ℝ O C D)
variables (intersection_is_midpoint : midpoint ℝ A C O ∧ midpoint ℝ B D O)

-- Theorem statement
theorem quadrilateral_is_parallelogram (ABCD : Convex ℝ := {A, B, C, D})
  (midline_passes_through_intersection : ∃ (O : A), midpoint ℝ A O ∧ midpoint ℝ O C)
  (intersection_is_midpoint : midpoint ℝ A C O ∧ midpoint ℝ B D O) :
  isParallelogram ABCD :=
  sorry

end quadrilateral_is_parallelogram_l773_773576


namespace extra_apples_l773_773246

theorem extra_apples (redApples : ℕ) (greenApples : ℕ) (students : ℕ) :
  redApples = 25 → greenApples = 17 → students = 10 → redApples + greenApples - students = 32 :=
by 
  intros h_red h_green h_students
  rw [h_red, h_green, h_students]
  norm_num
  sorry

end extra_apples_l773_773246


namespace unique_positive_real_solution_l773_773843

theorem unique_positive_real_solution :
  ∃! x : ℝ, 0 < x ∧ (x^8 + 5 * x^7 + 10 * x^6 + 2023 * x^5 - 2021 * x^4 = 0) := sorry

end unique_positive_real_solution_l773_773843


namespace count_3_digit_integers_with_product_36_l773_773079

theorem count_3_digit_integers_with_product_36 : 
  ∃ n, n = 21 ∧ 
         (∀ d1 d2 d3 : ℕ, 
           1 ≤ d1 ∧ d1 ≤ 9 ∧ 
           1 ≤ d2 ∧ d2 ≤ 9 ∧ 
           1 ≤ d3 ∧ d3 ≤ 9 ∧
           d1 * d2 * d3 = 36 → 
           (d1 ≠ 0 ∨ d2 ≠ 0 ∨ d3 ≠ 0)) := sorry

end count_3_digit_integers_with_product_36_l773_773079


namespace volleyball_tournament_l773_773123

theorem volleyball_tournament (n : ℕ) (teams : Finset (Fin n)) (played : ∀ (a b : teams), a ≠ b → a ≠ b → Prop)  
(cycle_condition : ∃ (a b c : teams), played a b ∧ played b c ∧ played c a) : 
∀ (wins : teams → ℕ), (∃ x y, x ≠ y ∧ wins x = wins y) :=
by
  sorry

end volleyball_tournament_l773_773123


namespace find_f_m_l773_773477

noncomputable def f (x : ℝ) := x^5 + Real.tan x - 3

theorem find_f_m (m : ℝ) (h : f (-m) = -2) : f m = -4 :=
sorry

end find_f_m_l773_773477


namespace shaded_area_correct_l773_773532

theorem shaded_area_correct :
  ∀ (height length : ℕ) (sh_height1 sh_length1 sh_height2 sh_length2 sh_height3 sh_length3 : ℕ),
  height = 5 →
  length = 12 →
  sh_height1 = 2 →
  sh_length1 = 3 →
  sh_height2 = 2 →
  sh_length2 = 4 →
  sh_height3 = 1 →
  sh_length3 = 5 →
  let total_area := height * length in
  let unshaded_area := (1 / 2) * sh_length1 * (sh_height1 - 1) + (1 / 2) * sh_length2 * (sh_height2 - 1) + (1 / 2) * sh_length3 * (sh_height3 - 1) in
  let shaded_area := total_area - unshaded_area in
  shaded_area = 51.5 :=
by
  intros height length sh_height1 sh_length1 sh_height2 sh_length2 sh_height3 sh_length3
  sorry

end shaded_area_correct_l773_773532


namespace intersection_point_finv_l773_773970

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 4 * x + b

theorem intersection_point_finv (a b : ℤ) : 
  (∀ x : ℝ, f (f x b) b = x) → 
  (∀ y : ℝ, f (f y b) b = y) → 
  (f (-4) b = a) → 
  (f a b = -4) → 
  a = -4 := 
by
  intros
  sorry

end intersection_point_finv_l773_773970


namespace part1_l773_773825

theorem part1 (k : ℝ) : (∀ x : ℝ, x ≠ 0 → (k-4) / x > 0 ↔ k > 4) :=
begin
  sorry
end

end part1_l773_773825


namespace poly_simplification_l773_773944

-- Defining the polynomials
def poly1 : ℕ → ℕ → ℤ
| 9, 0 := 4 -- 4x^9
| 0, 8 := 3 -- 3y^8
| 7, 0 := 5 -- 5x^7
| _, _ := 0

def poly2 : ℕ → ℕ → ℤ
| 10, 0 := 2 -- 2x^10
| 9, 0 := 6 -- 6x^9
| 0, 8 := 1 -- y^8
| 7, 0 := 4 -- 4x^7
| 0, 4 := 2 -- 2y^4
| 1, 0 := 7 -- 7x
| 0, 0 := 9 -- 9 (constant term)
| _, _ := 0

-- The target simplified polynomial
def target_poly : ℕ → ℕ → ℤ
| 10, 0 := 2 -- 2x^10
| 9, 0 := 10 -- 10x^9
| 0, 8 := 4 -- 4y^8
| 7, 0 := 9 -- 9x^7
| 0, 4 := 2 -- 2y^4
| 1, 0 := 7 -- 7x
| 0, 0 := 9 -- 9 (constant term)
| _, _ := 0

-- Statement of the proof
theorem poly_simplification :
  (λ n m, poly1 n m + poly2 n m) = target_poly :=
by {
  -- We need to check that for every (n, m), the sums match the target polynomial
  sorry
}

end poly_simplification_l773_773944


namespace exist_x_y_satisfy_condition_l773_773196

theorem exist_x_y_satisfy_condition (f g : ℝ → ℝ) (h1 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≥ 0) (h2 : ∀ y, 0 ≤ y ∧ y ≤ 1 → g y ≥ 0) :
  ∃ (x : ℝ), ∃ (y : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ |f x + g y - x * y| ≥ 1 / 4 :=
by
  sorry

end exist_x_y_satisfy_condition_l773_773196


namespace triangle_area_increase_l773_773851

theorem triangle_area_increase 
  (a b : ℝ) (θ : ℝ) : 
  let original_area := (1 / 2) * a * b * Real.sin θ,
      new_area := (1 / 2) * (3 * a) * (2 * b) * Real.sin θ in
  new_area = 6 * original_area := 
by sorry

end triangle_area_increase_l773_773851


namespace percent_increase_l773_773662

theorem percent_increase (S_new : ℝ) (ΔS : ℝ) (h1 : S_new = 90000) (h2 : ΔS = 25000) : 
  let S_orig := S_new - ΔS in
  let percent_increase := (ΔS / S_orig) * 100 in
  percent_increase ≈ 38.46 :=
by
  sorry

end percent_increase_l773_773662


namespace sequence_sum_a100_l773_773775

-- Definitions from conditions
def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 3 ∧ ∀ n ≥ 2, a (n + 1) = a n - a (n - 1)

def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in Finset.range (n + 1), a i

-- Theorem to be proved
theorem sequence_sum_a100 (a : ℕ → ℤ) (h_seq : sequence a) : a 100 = -1 ∧ S a 100 = 5 :=
  sorry

end sequence_sum_a100_l773_773775


namespace magnitude_of_z_l773_773773

open Complex

theorem magnitude_of_z {z : ℂ} (h : z * (1 + I) = 1 - I) : abs z = 1 :=
sorry

end magnitude_of_z_l773_773773


namespace jean_total_calories_l773_773894

-- Define the conditions
def pages_per_donut : ℕ := 2
def written_pages : ℕ := 12
def calories_per_donut : ℕ := 150

-- Define the question as a theorem
theorem jean_total_calories : (written_pages / pages_per_donut) * calories_per_donut = 900 := by
  sorry

end jean_total_calories_l773_773894


namespace height_of_tree_in_8_years_in_inches_l773_773610

theorem height_of_tree_in_8_years_in_inches 
  (initial_height : ℕ) (annual_growth : ℕ) (years : ℕ) (feet_to_inches : ℕ) 
  (h_initial_height : initial_height = 52) 
  (h_annual_growth : annual_growth = 5) 
  (h_years : years = 8) 
  (h_feet_to_inches : feet_to_inches = 12) : 
  let total_growth := annual_growth * years in
  let final_height_in_feet := initial_height + total_growth in
  final_height_in_feet * feet_to_inches = 1104 :=
by
  sorry

end height_of_tree_in_8_years_in_inches_l773_773610


namespace general_term_formula_l773_773310

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ d, ∀ n, a (n + 1) = a n + d

theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h_arith : is_arithmetic_sequence a)
  (h1 : a 1 + a 5 = (2 / 7) * (a 3) ^ 2) (h2 : S 7 = 63) :
  ∀ n, a n = 2 * n + 1 := by
  sorry

end general_term_formula_l773_773310


namespace lcm_a_c_336_l773_773044

-- Define lcm and lcm conditions
variables {a b c : ℕ}

def lcm (x y : ℕ) : ℕ := (x * y) / Nat.gcd x y

-- State the problem conditions
def condition1 : Prop := lcm a b = 16
def condition2 : Prop := lcm b c = 21

-- State the theorem to be proven
theorem lcm_a_c_336 (ha : condition1) (hb : condition2) : lcm a c = 336 :=
  sorry

end lcm_a_c_336_l773_773044


namespace range_of_t_value_of_t_at_special_condition_max_value_of_m_l773_773819

noncomputable def f (x : ℝ) (t : ℝ) : ℝ := (x^3 - 6 * x^2 + 3 * x + t) * exp(x)

-- Theorem 1: Range of t
theorem range_of_t (t : ℝ): -8 < t ∧ t < 24 := sorry

-- Definitions for conditions on extrema
variables {a b c t : ℝ}

-- Theorem 2: Value of t when a + c = 2b^2
theorem value_of_t_at_special_condition 
  (h1 : a < b) (h2 : b < c) (h3 : a + c = 2 * b^2) 
  (h_extrema : ∀ x, f x t = 0 → x = a ∨ x = b ∨ x = c) : t = 8 := sorry

-- Theorem 3: Maximum value of m
theorem max_value_of_m (m : ℕ) : 
  (∃ t ∈ Icc 0 2, ∀ x ∈ Icc 1 ↑m, f x t ≤ x) → m = 5 := sorry

end range_of_t_value_of_t_at_special_condition_max_value_of_m_l773_773819


namespace divisor_exists_l773_773561

theorem divisor_exists (n k : ℕ) (hn_pos : 0 < n) (hk_pos : 0 < k) 
  (hn_odd : Odd n) (h_divisors_odd : Odd ((finset.filter (fun d => d ≤ k) (finset.filter (fun d => d \in (finset.divisors (2 * n))) finset.range.fin (k + 1))).card)) : 
  ∃ d, d ∣ (2 * n) ∧ k < d ∧ d ≤ 2 * k :=
begin
  sorry
end

end divisor_exists_l773_773561


namespace incorrect_conclusion_l773_773765

theorem incorrect_conclusion (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : 1/a < 1/b ∧ 1/b < 0) : ¬ (ab > b^2) :=
by
  { sorry }

end incorrect_conclusion_l773_773765


namespace sum_powers_divisible_by_p_l773_773916

theorem sum_powers_divisible_by_p {p k : ℕ} (hp : Nat.Prime p) (hk : 0 ≤ k):
  p ∣ (1 + 2 + ... + p) ^ k ↔ (k = 0 ∨ (k ≥ 1 ∧ ¬(p-1 ∣ k))) :=
sorry

end sum_powers_divisible_by_p_l773_773916


namespace gcd_abcd_dcba_eq_one_l773_773229

theorem gcd_abcd_dcba_eq_one (a : ℤ) (b : ℤ) (c : ℤ) (d : ℤ)
    (h_b : b = a^2 + 1)
    (h_c : c = a^2 + 2)
    (h_d : d = a^2 + 3) :
  ∃ k : ℤ, ∀ a b c d, b = a^2 + 1 → c = a^2 + 2 → d = a^2 + 3 →
  k = 1000 * a + 100 * b + 10 * c + d + 1000 * d + 100 * c + 10 * b + a ∧
  gcd (1000 * a + 100 * b + 10 * c + d + 1000 * d + 100 * c + 10 * b + a) (1000 * d + 100 * c + 10 * b + a + 1000 * a + 100 * b + 10 * c + d) = 1 :=
begin
  sorry
end

end gcd_abcd_dcba_eq_one_l773_773229


namespace weight_of_one_bowling_ball_l773_773222

def weight_of_one_canoe : ℕ := 35

def ten_bowling_balls_equal_four_canoes (W: ℕ) : Prop :=
  ∀ w, (10 * w = 4 * W)

theorem weight_of_one_bowling_ball (W: ℕ) (h : W = weight_of_one_canoe) : 
  (10 * 14 = 4 * W) → 14 = 140 / 10 :=
by
  intros H
  sorry

end weight_of_one_bowling_ball_l773_773222


namespace horse_revolutions_l773_773332

noncomputable def circumference (radius: ℝ): ℝ := 2 * Real.pi * radius
noncomputable def total_distance (circumference: ℝ, revolutions: ℕ): ℝ := circumference * revolutions
noncomputable def required_revolutions (total_distance: ℝ, circumference: ℝ): ℕ := total_distance / circumference

theorem horse_revolutions
  (r1 r2: ℝ) (revs1: ℕ)
  (circumference1 circumference2: ℝ)
  (distance1 distance2: ℝ)
  (target_distance: ℝ):
  r1 = 30 → r2 = 10 →
  circumference1 = circumference r1 → 
  circumference2 = circumference r2 →
  distance1 = total_distance circumference1 revs1 →
  target_distance = 2 * distance1 →
  distance2 = target_distance →
  required_revolutions distance2 circumference2 = 240 :=
by
  intros
  sorry

end horse_revolutions_l773_773332


namespace minimum_value_of_a2_plus_b2_l773_773729

theorem minimum_value_of_a2_plus_b2 :
  ∀ (a b : ℝ), (∀ x : ℝ, abs (x - (a + b - 3)) < a + b ↔ -3 < x ∧ x < 3) → a^2 + b^2 = 9 / 2 :=
by
  intro a b h
  -- Based on the given conditions and interval translation
  have h1 : 2 * (a + b) - 3 = 3 := by
  {
    sorry -- Placeholder for detailed proof steps
  }
  have h2 : a + b = 3 := by
  {
    have := h1,
    linarith,
    sorry -- Placeholder for detailed proof steps
  }
  -- Basic inequalities to find the minimum value
  have h3 : a^2 + b^2 ≥ (a + b)^2 / 2 := by
  {
    calc
      a^2 + b^2 = (a + b)^2 - 2 * a * b : by sorry -- Placeholder for proof steps
      ... ≥ (a + b)^2 / 2         : by sorry -- Placeholder for proof steps
  }
  exact eq.symm (by sorry) -- Placeholder for concluding proof steps
  sorry -- Overall finish for proof

end minimum_value_of_a2_plus_b2_l773_773729


namespace estimated_red_balls_in_bag_l773_773878

theorem estimated_red_balls_in_bag (total_balls : Nat) (red_draws : Nat) (total_draws : Nat) 
  (red_probability : ℚ) (h_total_balls : total_balls = 10) 
  (h_repeated_draws : red_draws = 200) 
  (h_total_draws : total_draws = 1000)
  (h_probability : red_probability = red_draws / total_draws) : 
  (total_balls * red_probability).toNat = 2 :=
by
  sorry

end estimated_red_balls_in_bag_l773_773878


namespace volume_of_sphere_eq_4_sqrt3_pi_l773_773807

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r ^ 3

theorem volume_of_sphere_eq_4_sqrt3_pi
  (r : ℝ) (h : 4 * Real.pi * r ^ 2 = 2 * Real.sqrt 3 * Real.pi * (2 * r)) :
  volume_of_sphere r = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end volume_of_sphere_eq_4_sqrt3_pi_l773_773807


namespace problem_l773_773099

def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ :=
  2 * Real.sin (ω * x + φ) - 3

theorem problem (ω φ : ℝ) (hω : ω > 0)
  (hf_eq : ∀ x : ℝ, f ω φ (x + π / 6) = f ω φ (π / 3 - x)) :
  f ω φ (π / 4) ∈ {-1, -5} :=
by sorry

end problem_l773_773099


namespace symmetric_point_y_axis_l773_773603

theorem symmetric_point_y_axis (x y : ℝ) (hx : x = -3) (hy : y = 2) :
  (-x, y) = (3, 2) :=
by
  sorry

end symmetric_point_y_axis_l773_773603


namespace trig_eq_solutions_l773_773672

open Real

theorem trig_eq_solutions (x : ℝ) :
  2 * sin x ^ 3 + 2 * sin x ^ 2 * cos x - sin x * cos x ^ 2 - cos x ^ 3 = 0 ↔
  (∃ n : ℤ, x = -π / 4 + n * π) ∨ (∃ k : ℤ, x = arctan (sqrt 2 / 2) + k * π) ∨ (∃ m : ℤ, x = -arctan (sqrt 2 / 2) + m * π) :=
by
  sorry

end trig_eq_solutions_l773_773672


namespace num_possible_homework_situations_l773_773636

theorem num_possible_homework_situations (students : ℕ) (teachers : ℕ) (h_students : students = 4) (h_teachers : teachers = 3) :
  teachers ^ students = 3 ^ 4 :=
by
  rw [h_students, h_teachers]
  exact (3^4)

end num_possible_homework_situations_l773_773636


namespace sixth_sample_is_623_l773_773329

-- Define the random number table rows provided
def random_table_row_4_to_6 :=
  ["32211834297864540732524206443812234356773578905642",
   "84421253313457860736253007328623457889072368960804",
   "32567808436789535577348994837522535578324577892345"]

-- Define the function to read the samples from the table
def read_samples (start_row : Nat) (start_col : Nat) (n : Nat) (table : List String) : List String :=
  table.drop (start_row - 1)           -- Drop rows to start from start_row
        .flat_map (λ row => row.reverse.to_list) -- Convert rows to list of characters, reversed
        .drop (start_col - 1)           -- Drop columns to start from start_col
        .chunks 3                       -- Divide into chunks of 3 (each sample number)
        .take n                        -- Take n samples
        .map (λ chunk => chunk.asString) -- Convert list of characters to string

-- The theorem statement
theorem sixth_sample_is_623 : read_samples 5 6 6 random_table_row_4_to_6 = ["253", "313", "457", "007", "328", "623"] :=
  by  -- Reading samples from the random number table from the 5th row and 6th column
    sorry

end sixth_sample_is_623_l773_773329


namespace closest_point_on_line_to_given_point_l773_773419

noncomputable def closest_point (p : ℝ × ℝ × ℝ) (line_pt : ℝ × ℝ × ℝ) (line_dir : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  let t := ((p.1 - line_pt.1) * line_dir.1 + (p.2 - line_pt.2) * line_dir.2 + (p.3 - line_pt.3) * line_dir.3) /
            (line_dir.1 * line_dir.1 + line_dir.2 * line_dir.2 + line_dir.3 * line_dir.3) in
  (line_pt.1 + t * line_dir.1, line_pt.2 + t * line_dir.2, line_pt.3 + t * line_dir.3)

theorem closest_point_on_line_to_given_point :
  closest_point (1, 4, 5) (3, 0, 2) (1, 5, -2) = (105 / 31, 60 / 31, 50 / 31) := 
sorry

end closest_point_on_line_to_given_point_l773_773419


namespace maximum_AB_l773_773133

-- Definitions and conditions
def point := (ℝ × ℝ)
def circle (center : point) (radius : ℝ) : set point :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def A_curve : set point :=
  circle (3, 4) 2

def B_curve : set point :=
  circle (0, 0) 1

def distance (p1 p2 : point) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Proving the maximum distance
theorem maximum_AB : ∀ A ∈ A_curve, ∀ B ∈ B_curve, distance A B ≤ 8 :=
by
  intros A A_in A_curve B B_in B_curve
  sorry

end maximum_AB_l773_773133


namespace equipment_rental_ratio_l773_773330

theorem equipment_rental_ratio 
  (cost_actors : ℤ) 
  (num_people cost_food_per_person : ℤ) 
  (selling_price profit : ℤ)
  (h1 : cost_actors = 1200)
  (h2 : num_people = 50)
  (h3 : cost_food_per_person = 3)
  (h4 : selling_price = 10000)
  (h5 : profit = 5950)
  : (2700 / (cost_actors + (num_people * cost_food_per_person)) = 2) :=
by 
  let total_food_cost := num_people * cost_food_per_person;
  have h_total_food_cost : total_food_cost = 150 := by 
    sorry,
  let combined_cost := cost_actors + total_food_cost;
  have h_combined_cost : combined_cost = 1350 := by 
    sorry,
  let total_cost := combined_cost + 2700;
  have h_total_cost : total_cost = 4050 := by 
    sorry,
  have h_total_cost_from_profit : total_cost = selling_price - profit := by 
    sorry,
  have h_equate_total_cost : combined_cost + 2700 = 4050 := by 
    sorry,
  have h_equipment_rental : 2700 = 2700 := by 
    sorry,
  rw h_equate_total_cost,
  have h_ratio : 2700 / combined_cost = 2 := by 
    sorry,
  exact h_ratio

end equipment_rental_ratio_l773_773330


namespace garden_perimeter_l773_773620

theorem garden_perimeter (L B : ℕ) (hL : L = 205) (hB : B = 95) : 2 * (L + B) = 600 := 
by 
  rw [hL, hB]
  norm_num

end garden_perimeter_l773_773620


namespace pentagon_edges_same_color_l773_773029

-- Definitions for vertices of the pentagons and their connecting edges
inductive Color
| Red
| Blue

-- Defining the edges with their colors
def edge_color (i j : ℕ) : Color := sorry  -- This will be specified assuming the constraints

-- Propositions resulting from the constraints
axiom triangle_condition (i j k : ℕ) (hi : i ≠ j) (hj : j ≠ k) (hk : k ≠ i) :
  ¬ (edge_color i j = edge_color j k ∧ edge_color j k = edge_color k i)

-- Statement that needs to be proved
theorem pentagon_edges_same_color :
  (∀ i j : ℕ, i ≠ j → (i, j) ∈ {(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)} →
   (edge_color i j = Color.Red ∨ edge_color i j = Color.Blue)) →
  (∀ i j : ℕ, (i, j) ∈ {(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)} →
   edge_color i j = edge_color 1 2) :=
sorry

end pentagon_edges_same_color_l773_773029


namespace transform_area_l773_773159

noncomputable def matrix := ![[3, 2], [4, 3]]

theorem transform_area (R : Type*) [measure_space R] (μ : measure R) (hR : μ set.univ = 9) :
  let A := matrix in
  let det_A := (3 * 3) - (2 * 4) in
  det_A = 1 →
  (μ (A '' set.univ)) = 9 :=
by
  -- Define the matrix A
  let A := matrix

  -- Compute the determinant of A
  let det_A := (3 * 3) - (2 * 4)

  -- Given condition: det_A = 1
  intro h_det
  have h_det_eq : det_A = 1,
  exact h_det

  -- Resulting transformed area
  have h_transformed_area : μ (A '' set.univ) = det_A * μ set.univ,
  sorry  -- Proof goes here

  -- Substitute det_A = 1 and μ set.univ = 9
  rw [h_det_eq, hR],
  exact rfl

end transform_area_l773_773159


namespace AM_BN_product_l773_773461

theorem AM_BN_product (x y k: ℝ) (A B M N F : ℝ × ℝ)
  (h_parabola : ∀ x, (x, (x^2 : ℝ)/4) ∈ {p : ℝ × ℝ | p.fst^2 = 4 * p.snd})
  (h_circle : ∀ x y, (x, y) ∈ {p : ℝ × ℝ | p.fst^2 + p.snd^2 - 2 * p.snd = 0})
  (h_line : ∀ x, (x, k * x + 1) ∈ {p : ℝ × ℝ | p.snd = k * p.fst + 1})
  (h_focus : F = (0, 1))
  (h_intersection_parabola : ∃ A B, A ≠ B ∧ A.1 ^ 2 = 4 * A.2 ∧ B.1 ^ 2 = 4 * B.2 ∧ A.2 = k * A.1 + 1 ∧ B.2 = k * B.1 + 1)
  (h_intersection_circle : ∃ M N, M ≠ N ∧ M ∈ {p : ℝ × ℝ | p.fst^2 + p.snd^2 - 2 * p.snd = 0} ∧ N ∈ {p : ℝ × ℝ | p.fst^2 + p.snd^2 - 2 * p.snd = 0})
  (h_side: ∃ M, M.1 ≥ 0)
  (h_dist: ∀ M N, M ∈ {p : ℝ × ℝ | p.fst^2 + p.snd^2 - 2 * p.snd = 0} ∧ N ∈ {p : ℝ × ℝ | p.fst^2 + p.snd^2 - 2 * p.snd = 0} 
    → |(M.1 - F.1)| = |(N.1 - F.1)| ∧ |(M.2 - F.2)| = |(N.2 - F.2)|)
  : |dist A M| * |dist B N| = 1 := sorry

end AM_BN_product_l773_773461


namespace labor_union_tree_equation_l773_773681

theorem labor_union_tree_equation (x : ℕ) : 2 * x + 21 = 3 * x - 24 := 
sorry

end labor_union_tree_equation_l773_773681


namespace area_ratio_of_squares_l773_773881

theorem area_ratio_of_squares (a b c d i j k l : ℝ) (A B C D I J K L : ℝ → ℝ) :
  A (16) -- Each side of square ABCD is 16 units
  → I (3 * B) -- Point I on side AB with AI = 3 * IB
  → area_IJKL = 32 -- Area of square IJKL is 32 square units
  → area_ABCD = 256 -- Area of square ABCD is 256 square units
  → ratio_areas = (area_IJKL / area_ABCD) -- Ratio of areas
  → ratio_areas = 1 / 8 :=  -- Required ratio to prove
sorry

end area_ratio_of_squares_l773_773881


namespace reciprocal_of_2023_l773_773987

theorem reciprocal_of_2023 : (2023 : ℝ)⁻¹ = 1 / 2023 :=
by
  sorry

end reciprocal_of_2023_l773_773987


namespace smallest_possible_sum_l773_773790

-- Defining the conditions for x and y.
variables (x y : ℕ)

-- We need a theorem to formalize our question with the given conditions.
theorem smallest_possible_sum (hx : x > 0) (hy : y > 0) (hne : x ≠ y) (hxy : 1/x + 1/y = 1/24) : x + y = 100 :=
by
  sorry

end smallest_possible_sum_l773_773790


namespace distance_trains_meet_l773_773271

-- Define the conditions and question of the math problem in Lean 4.
def length_train1 : ℝ := 100 -- Length of train 1 in meters
def length_train2 : ℝ := 200 -- Length of train 2 in meters

def speed_train1_kmh : ℝ := 54 -- Speed of train 1 in km/h
def speed_train2_kmh : ℝ := 72 -- Speed of train 2 in km/h

def meeting_time : ℝ := 23.99808015358771 -- Time in seconds when trains meet

-- Convert speeds from km/h to m/s
def speed_train1_ms : ℝ := speed_train1_kmh * 1000 / 3600
def speed_train2_ms : ℝ := speed_train2_kmh * 1000 / 3600

-- Calculate the relative speed and distance
def relative_speed : ℝ := speed_train1_ms + speed_train2_ms
def distance_between_trains : ℝ := relative_speed * meeting_time

-- The theorem to be proven: the distance is approximately 839.9328053815769 meters
theorem distance_trains_meet : abs (distance_between_trains - 839.9328053815769) < 1e-6 := by
  sorry

end distance_trains_meet_l773_773271


namespace dogs_in_pet_shop_l773_773664

variable (D C B : ℕ) (x : ℕ)

theorem dogs_in_pet_shop
  (h1 : D = 3 * x)
  (h2 : C = 7 * x)
  (h3 : B = 12 * x)
  (h4 : D + B = 375) :
  D = 75 :=
by
  sorry

end dogs_in_pet_shop_l773_773664


namespace linda_coats_l773_773923

variable (wall_area : ℝ) (cover_per_gallon : ℝ) (gallons_bought : ℝ)

theorem linda_coats (h1 : wall_area = 600)
                    (h2 : cover_per_gallon = 400)
                    (h3 : gallons_bought = 3) :
  (gallons_bought / (wall_area / cover_per_gallon)) = 2 :=
by
  sorry

end linda_coats_l773_773923


namespace problem_complement_intersection_l773_773071

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

def complement (U M : Set ℕ) : Set ℕ := {x ∈ U | x ∉ M}

theorem problem_complement_intersection :
  (complement U M) ∩ N = {3} :=
by
  sorry

end problem_complement_intersection_l773_773071


namespace tan_beta_minus_2alpha_l773_773035

def tan (x : Real) : Real := Real.sin x / Real.cos x

theorem tan_beta_minus_2alpha (α β : Real)
  (h₁ : tan α = 1/2)
  (h₂ : tan (α - β) = -1/3) :
  tan (β - 2 * α) = -1/7 :=
by
  sorry

end tan_beta_minus_2alpha_l773_773035


namespace minimize_fence_length_l773_773541

theorem minimize_fence_length :
  ∃ (x y : ℝ), x * y = 294 ∧ 3 * x + 2 * y = 3 * 14 + 2 * 21 :=
begin
  sorry
end

end minimize_fence_length_l773_773541


namespace new_container_holds_48_l773_773704

-- Definitions of the conditions
def original_volume (r h : ℝ) : ℝ := π * r^2 * h
def new_volume (r h : ℝ) : ℝ := π * (2 * r)^2 * (4 * h)

-- Given condition
axiom original_volume_is_3 (r h : ℝ) : original_volume r h = 3

-- Theorem statement to prove the new volume
theorem new_container_holds_48 (r h : ℝ) (h0 : original_volume r h = 3) : new_volume r h = 48 :=
by
  -- proof steps here
  sorry

end new_container_holds_48_l773_773704


namespace phone_bill_percentage_increase_l773_773016

theorem phone_bill_percentage_increase
  (usual_monthly_bill yearly_total next_year_total : ℝ)
  (months_in_year : ℕ) 
  (usual_monthly_bill_eq : usual_monthly_bill = 50)
  (months_in_year_eq : months_in_year = 12)
  (next_year_total_eq : next_year_total = 660)
  (yearly_total_eq : yearly_total = usual_monthly_bill * months_in_year) :
  ((next_year_total - yearly_total) / yearly_total) * 100 = 10 := 
by 
  -- Definitions
  have h1 : yearly_total = 50 * 12, from calc
    yearly_total = usual_monthly_bill * months_in_year : yearly_total_eq
             ... = 50 * 12 : by rw [usual_monthly_bill_eq, months_in_year_eq],
  have h2 : yearly_total = 600, by norm_num,
  have h3 : next_year_total - yearly_total = 60, by linarith [yearly_total_eq, next_year_total_eq],
  have h4 : (next_year_total - yearly_total) / yearly_total = 0.1, by linarith [h3, h1, h2],
   -- Conclusion
  linarith [h4]  -- Same as multiplying by 100 and verifying the value

end phone_bill_percentage_increase_l773_773016


namespace total_pieces_of_junk_mail_l773_773336

def pieces_per_block : ℕ := 48
def num_blocks : ℕ := 4

theorem total_pieces_of_junk_mail : (pieces_per_block * num_blocks) = 192 := by
  sorry

end total_pieces_of_junk_mail_l773_773336


namespace prime_factors_count_N_l773_773841

noncomputable def num_prime_factors (N : ℕ) : ℕ := 
if H : N > 1 then (⟨λ x, x.factors.to_finset.size⟩ : Prime)

theorem prime_factors_count_N (N : ℕ)
  (h : log 2 (log 3 (log 7 (log 11 N))) = 7) : num_prime_factors N = 1 :=
sorry

end prime_factors_count_N_l773_773841


namespace sum_of_first_1000_natural_numbers_divisible_by_143_l773_773198

theorem sum_of_first_1000_natural_numbers_divisible_by_143 :
  let S := (1000 * (1000 + 1)) / 2
  in S = 500500 ∧ 500500 % 143 = 0 :=
by {
  let S := (1000 * (1000 + 1)) / 2,
  show S = 500500 ∧ 500500 % 143 = 0,
  sorry
}

end sum_of_first_1000_natural_numbers_divisible_by_143_l773_773198


namespace sum_1_to_n_sum_S_n_k_l773_773760

-- Part (a)
theorem sum_1_to_n (n : ℕ) : 
  (∑ i in Finset.range n + 1, (i + 1) * (n - i)) = (n * (n + 1) * (n + 2)) / 6 :=
sorry

-- Part (b)
theorem sum_S_n_k (n k : ℕ) : 
  let S := (∑ i in Finset.range k +1, (∏ j in Finset.range i +1, (j+1) * (n-(k-i)))) in
  S = (nat.factorial k)^2 * nat.choose (n + k + 1) (2 * k + 1) :=
sorry

end sum_1_to_n_sum_S_n_k_l773_773760


namespace interval_of_decrease_for_f_x_minus_1_l773_773508

-- Define the function f' based on the given condition.
def f' (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the condition for decrease.
def decreasing_interval (f' : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a < x ∧ x < b → f' x < 0

-- State that the interval of decrease for f(x-1) is (2, 4).
theorem interval_of_decrease_for_f_x_minus_1 : 
  decreasing_interval (λ x, (x + 1)^2 - 4*(x + 1) + 3) 2 4 :=
by
  sorry

end interval_of_decrease_for_f_x_minus_1_l773_773508


namespace characterize_strictly_negative_l773_773498

-- Assume we are working within the context of $p$-adic integers
def negative_padic (p : ℕ) (a : ℤ_[p]) : Prop :=
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → a.digit n = p - 1

-- The main theorem statement in Lean 4
theorem characterize_strictly_negative (p : ℕ) (a : ℤ_[p]) : Prop :=
  negative_padic p a ↔ (a < 0 ∧ ∀ n : ℕ, ∃ m : ℕ, m ≥ n → a.digit m = p - 1)

-- Add the "sorry" to skip the proof
by sorry

end characterize_strictly_negative_l773_773498


namespace xiaoming_interview_pass_probability_l773_773325

theorem xiaoming_interview_pass_probability :
  let p_correct := 0.7
  let p_fail_per_attempt := 1 - p_correct
  let p_fail_all_attempts := p_fail_per_attempt ^ 3
  let p_pass_interview := 1 - p_fail_all_attempts
  p_pass_interview = 0.973 := by
    let p_correct := 0.7
    let p_fail_per_attempt := 1 - p_correct
    let p_fail_all_attempts := p_fail_per_attempt ^ 3
    let p_pass_interview := 1 - p_fail_all_attempts
    sorry

end xiaoming_interview_pass_probability_l773_773325


namespace value_of_f_log_half_24_l773_773505

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_log_half_24 :
  (∀ x : ℝ, f x * -1 = f (-x)) → -- Condition 1: f(x) is an odd function.
  (∀ x : ℝ, f (x + 1) = f (x - 1)) → -- Condition 2: f(x + 1) = f(x - 1).
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x = 2^x - 2) → -- Condition 3: For 0 < x < 1, f(x) = 2^x - 2.
  f (Real.logb 0.5 24) = 1 / 2 := 
sorry

end value_of_f_log_half_24_l773_773505


namespace power_mod_equiv_l773_773279

theorem power_mod_equiv : 7^150 % 12 = 1 := 
  by
  sorry

end power_mod_equiv_l773_773279


namespace eighth_arithmetic_term_l773_773625

theorem eighth_arithmetic_term (a₂ a₁₄ a₈ : ℚ) 
  (h2 : a₂ = 8 / 11)
  (h14 : a₁₄ = 9 / 13) :
  a₈ = 203 / 286 :=
by
  sorry

end eighth_arithmetic_term_l773_773625


namespace sum_of_solutions_l773_773042

noncomputable def is_even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)
noncomputable def is_monotonic (f : ℝ → ℝ) := ∀ ⦃x y⦄, 0 < x → x < y → f x ≤ f y

theorem sum_of_solutions (f : ℝ → ℝ) (even_f : is_even_function f) (monotonic_f : is_monotonic f) :
    (∑ x in { x : ℝ | f x = f ((x + 1) / (x + 2)) }, x) = -4 :=
sorry

end sum_of_solutions_l773_773042


namespace acute_triangles_no_more_than_three_quarters_l773_773516

theorem acute_triangles_no_more_than_three_quarters 
  {n : ℕ} (h : n > 3) 
  (no_three_collinear : ∀ (p₁ p₂ p₃ : ℝ × ℝ), p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₃ ≠ p₁ → ¬ collinear ℝ (λ i, [p₁, p₂, p₃]) i) :
  let points := fin n → ℝ × ℝ in
  (number_of_acute_triangles points) ≤ 3 / 4 * (number_of_triangles points) := 
sorry

end acute_triangles_no_more_than_three_quarters_l773_773516


namespace class_average_l773_773302

theorem class_average (n : ℕ) (h1 : 20 * n = 20 * 80) (h2 : 50 * n = 50 * 60) (h3 : 30 * n = 30 * 40) :
  (20 * 80 + 50 * 60 + 30 * 40) / 100 = 58 :=
by
  have : 20 * 80 + 50 * 60 + 30 * 40 = 5800, sorry
  have : 5800 / 100 = 58, sorry
  sorry

end class_average_l773_773302


namespace cos_alpha_minus_beta_cos_alpha_plus_beta_l773_773309

variables (α β : Real) (h1 : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2)
           (h2 : Real.tan α * Real.tan β = 13/7)
           (h3 : Real.sin (α - β) = sqrt 5 / 3)

-- Part (1): Prove that cos (α - β) = 2/3
theorem cos_alpha_minus_beta : Real.cos (α - β) = 2 / 3 := by
  have h := h1
  have h := h2
  have h := h3
  sorry

-- Part (2): Prove that cos (α + β) = -1/5
theorem cos_alpha_plus_beta : Real.cos (α + β) = -1 / 5 := by
  have h := h1
  have h := h2
  have h := h3
  sorry

end cos_alpha_minus_beta_cos_alpha_plus_beta_l773_773309


namespace remainder_equivalence_l773_773290

theorem remainder_equivalence (x : ℕ) (r : ℕ) (hx_pos : 0 < x) 
  (h1 : ∃ q1, 100 = q1 * x + r) (h2 : ∃ q2, 197 = q2 * x + r) : 
  r = 3 :=
by
  sorry

end remainder_equivalence_l773_773290


namespace percentage_increase_l773_773868

variables {a b : ℝ} -- Assuming a and b are real numbers

-- Define the conditions explicitly
def initial_workers := a
def workers_left := b
def remaining_workers := a - b

-- Define the theorem for percentage increase in daily performance
theorem percentage_increase (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
  (100 * b) / (a - b) = (100 * a * b) / (a * (a - b)) :=
by
  sorry -- Proof will be filled in as needed

end percentage_increase_l773_773868


namespace range_of_a_l773_773962

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x ∈ set.Icc (-1 : ℝ) 1 ∧ (x^2 + 2 * x - a = 0)) → (-1 ≤ a ∧ a ≤ 3) := by
  sorry

end range_of_a_l773_773962


namespace ellipse_problem_l773_773779

theorem ellipse_problem :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    ((∀ x y : ℝ, (x = -a ∧ y = 0 ∧
    x - √2 * y + √2 = 0) ∨ (x = 0 ∧ y = b ∧
    x - √2 * y + √2 = 0)) →
    (a = √2 ∧ b = 1 ∧ ∀ x y : ℝ, (x^2 / 2 + y^2 = 1))) ∧
    (∃ x0 y0 : ℝ, (x0 - √2 * y0 + √2 = 0) ∧ 
    ((x0 = (√14 - √2) / 3 ∧ y0 = (2 + √7) / 3) ∨
    (x0 = (-√14 - √2) / 3 ∧ y0 = (2 - √7) / 3))) :=
begin
  sorry
end

end ellipse_problem_l773_773779


namespace system_of_equations_l773_773069

theorem system_of_equations (x y : ℝ) 
  (h1 : 2 * x + y = 5) 
  (h2 : x + 2 * y = 4) : 
  x + y = 3 :=
sorry

end system_of_equations_l773_773069


namespace triangle_area_l773_773821

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + 2 * sqrt 3 * cos x^2 - sqrt 3

theorem triangle_area
  (A : ℝ) (b c : ℝ)
  (h1 : f A = 1)
  (h2 : b * c = 2) 
  (h3 : (b * cos A) * (c * cos A) = sqrt 2) : 
  (1 / 2 * b * c * sin A = sqrt 2 / 2) := 
sorry

end triangle_area_l773_773821


namespace coefficient_x4_in_expansion_l773_773806

theorem coefficient_x4_in_expansion:
  (∑ i in Finset.range (6), (binomial 5 i) * (if i = 2 then 3 else -1) * (if i = 4 then 1 else 0)) = 25 :=
by
  sorry

end coefficient_x4_in_expansion_l773_773806


namespace tessa_needs_4_apples_l773_773950

variables (initial_apples given_apples left_apples needed_apples : ℝ)

axiom initial_apples_def : initial_apples = 10.0
axiom given_apples_def : given_apples = 5.0
axiom left_apples_def : left_apples = 11.0

noncomputable def total_apples : ℝ := initial_apples + given_apples
noncomputable def needed_apples_def : needed_apples = total_apples - left_apples

theorem tessa_needs_4_apples : needed_apples = 4.0 :=
by 
  rw [needed_apples_def, total_apples, initial_apples_def, given_apples_def, left_apples_def]
  sorry

end tessa_needs_4_apples_l773_773950


namespace count_three_digit_integers_divisible_by_11_and_7_l773_773492

theorem count_three_digit_integers_divisible_by_11_and_7 : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ n % 77 = 0 } → (card { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ n % 77 = 0 }) = 11 :=
by
  sorry

end count_three_digit_integers_divisible_by_11_and_7_l773_773492


namespace sum_first_n_terms_l773_773228

noncomputable theory

-- Definitions from problem conditions
def common_difference_not_zero (d : ℤ) : Prop := d ≠ 0

def a3_geometric_mean (a1 d : ℤ) : Prop := (a1 + 2 * d) ^ 2 = (a1 + d) * (a1 + 4 * d)

def geometric_sequence (a1 d : ℤ) (k : ℕ → ℕ) : Prop :=
  ∀ n, a1 + (k n - 1) * d = d * 3^(n - 1)

-- Statement to prove
theorem sum_first_n_terms
  (a1 d : ℤ) (k : ℕ → ℕ) (T : ℕ → ℤ) :
  common_difference_not_zero d →
  a3_geometric_mean a1 d →
  geometric_sequence a1 d k →
  T = λ n, (3 ^ (n + 2) - 9 + 2 * n) / 2 →
  2 * T n + 9 = 3 ^ (n + 2) + 2 * n :=
by 
  intros h1 h2 h3 hT
  sorry

end sum_first_n_terms_l773_773228


namespace perimeter_of_figure_l773_773381

theorem perimeter_of_figure :
  let side_length := 1
  let horizontal_perimeter := 4 * side_length
  let vertical_perimeter := 2 * side_length
  let triangle_perimeter := 2 * side_length
  in
  horizontal_perimeter + vertical_perimeter + triangle_perimeter = 8 :=
by
  let side_length := 1
  let horizontal_perimeter := 4 * side_length
  let vertical_perimeter := 2 * side_length
  let triangle_perimeter := 2 * side_length
  show horizontal_perimeter + vertical_perimeter + triangle_perimeter = 8
  sorry

end perimeter_of_figure_l773_773381


namespace range_m_l773_773816

noncomputable def f (x m : ℝ) : ℝ := real.sqrt (real.log x + x + m)

def y_curve (x : ℝ) : ℝ := ((1 - real.exp 1) / 2) * real.cos x + (1 + real.exp 1) / 2

theorem range_m :
  (∃ (x0 y0 : ℝ), y0 = y_curve x0 ∧ f (f y0) 0 = y0) →
  ∀ (m : ℝ), 0 ≤ m ∧ m ≤ real.exp 2 - real.exp 1 - 1 :=
sorry

end range_m_l773_773816


namespace susan_remaining_spaces_to_win_l773_773215

/-- Susan's board game has 48 spaces. She makes three moves:
 1. She moves forward 8 spaces
 2. She moves forward 2 spaces and then back 5 spaces
 3. She moves forward 6 spaces
 Prove that the remaining spaces she has to move to reach the end is 37.
-/
theorem susan_remaining_spaces_to_win :
  let total_spaces := 48
  let first_turn := 8
  let second_turn := 2 - 5
  let third_turn := 6
  let total_moved := first_turn + second_turn + third_turn
  total_spaces - total_moved = 37 :=
by
  sorry

end susan_remaining_spaces_to_win_l773_773215


namespace find_A_l773_773597

def area_square (s : ℕ) : ℕ := s * s

def area_rectangle (w l : ℕ) : ℕ := w * l

theorem find_A : ∀ (A : ℕ), 
  let s := 12 in 
  let initial_area := area_square s in 
  let new_width := s + 3 in 
  let new_length := s - A in 
  initial_area = 144 ∧
  area_rectangle new_width new_length = 120 -> 
  A = 4 := 
by
  intros A
  let s := 12
  let initial_area := area_square s 
  let new_width := s + 3 
  let new_length := s - A 
  assume h,
  sorry

end find_A_l773_773597


namespace total_pieces_gum_is_correct_l773_773671

-- Define the number of packages and pieces per package
def packages : ℕ := 27
def pieces_per_package : ℕ := 18

-- Define the total number of pieces of gum Robin has
def total_pieces_gum : ℕ :=
  packages * pieces_per_package

-- State the theorem and proof obligation
theorem total_pieces_gum_is_correct : total_pieces_gum = 486 := by
  -- Proof omitted
  sorry

end total_pieces_gum_is_correct_l773_773671


namespace find_k_l773_773358

variable (S v : ℝ)

theorem find_k (hv : v > 0) (hs : S > 0) : let k := 100 * ( (S / v - S / (1.25*v)) / (S / v) )
 in k = 20 := by
  have : 1.25 = 5 / 4 := by norm_num
  have h1 : S / (5 / 4 * v) = 4 / 5 * (S / v) := by
    rw [mul_comm (5/4), ← div_div, div_eq_mul_inv, inv_div, one_mul]
  have h2 : S / v - 4 / 5 * (S / v) = S / v * (1 - 4 / 5) := by
    ring
  have h3: (1 - 4 / 5) = 1 / 5 := by norm_num
  have h4 : (S / v * (1 / 5)) / (S / v) = 1 / 5 := by
    rw [mul_comm (1 / 5), mul_div_cancel]; exact ne_of_gt hv
  have h5 : 100 * (1 / 5) = 20 := by norm_num
  rw [h1, h2, h3, h4, h5]
  exact rfl

end find_k_l773_773358


namespace price_of_three_kg_sugar_and_one_kg_salt_is_five_dollars_l773_773977

-- Defining the known quantities
def price_of_two_kg_sugar_and_five_kg_salt : ℝ := 5.50
def price_per_kg_sugar : ℝ := 1.50

-- Defining the variables for the proof
def price_per_kg_salt := (price_of_two_kg_sugar_and_five_kg_salt - 2 * price_per_kg_sugar) / 5

def price_of_three_kg_sugar_and_one_kg_salt := 3 * price_per_kg_sugar + price_per_kg_salt

-- The theorem stating the result
theorem price_of_three_kg_sugar_and_one_kg_salt_is_five_dollars :
  price_of_three_kg_sugar_and_one_kg_salt = 5.00 :=
by
  -- Calculate intermediary values for sugar and salt costs
  let price_of_two_kg_sugar := 2 * price_per_kg_sugar
  let price_of_five_kg_salt := price_of_two_kg_sugar_and_five_kg_salt - price_of_two_kg_sugar
  let price_per_kg_salt' := price_of_five_kg_salt / 5

  -- Calculate final price for verification
  let price_of_three_kg_sugar := 3 * price_per_kg_sugar
  let final_price := price_of_three_kg_sugar + price_per_kg_salt'

  -- Assert the final price is $5.00
  have h1 : price_of_two_kg_sugar = 3.00 := by sorry
  have h2 : price_of_five_kg_salt = 2.50 := by sorry
  have h3 : price_per_kg_salt' = 0.50 := by sorry
  have h4 : final_price = 5.00 := by sorry

  -- Conclude the proof
  exact h4

end price_of_three_kg_sugar_and_one_kg_salt_is_five_dollars_l773_773977


namespace inequality_solution_set_l773_773627

theorem inequality_solution_set (x : ℝ) : 2^(x-2) > 1 ↔ x > 2 :=
by sorry

end inequality_solution_set_l773_773627


namespace function_properties_l773_773403

noncomputable def f : ℕ → ℕ+ := sorry

def k : ℕ+ := sorry

def p : ℕ+ := sorry

axiom prime_p : Prime p

axiom condition1 : ∀ n : ℕ+, n ≥ k → f (n + p) = f n

axiom condition2 : ∀ m n : ℕ+, m ∣ n → f (m + 1) ∣ (f(n) + 1)

theorem function_properties : 
(∀ n : ℕ+, n ≥ k → n % p ≠ 1 → f n = 1) ∧
(∀ n : ℕ+, n ≥ k → n % p = 1 → f n = 2) ∧
(∀ n : ℕ+, 1 < n → n < k → n % p = 1 → (f n = 1 ∨ f n = 2)) ∧
(∃ f1 : ℕ+, f f1 ∣ (f 1 + 1)) :=
by
  sorry

end function_properties_l773_773403


namespace promenade_and_tornado_liars_count_l773_773533

theorem promenade_and_tornado_liars_count :
  ∃ L : ℕ, L = 1010 ∧ 
    ∀ (N : ℕ) (D : ℕ) (S : ℕ → Prop),
    N = 2020 ∧ 
    D = 1011 ∧ 
    (∀ d : ℕ, 1 ≤ d ∧ d ≤ N → (S d) = ("If you do not count me, there are more liars than half among the remaining deputies")) ∧ 
    (∀ d : ℕ, (S d) → (d = 1010)) :=
sorry

end promenade_and_tornado_liars_count_l773_773533


namespace performance_arrangement_l773_773696

def num_ways_to_arrange_performances : ℕ :=
  let perms := finEnum.enumerate (Fin 6) -- Total arrangements of 6 performances
  perms.filter (λ perm,
    perm.head ≠ skit # and
    (0 until 4).all (λ i, perm.nth (i + 1) ≠ perm.nth i + 1) # all adjacent checks 
  ).size 

theorem performance_arrangement (first_not_skit : ∀ (arr : List ℕ), arr.head ≠ 5)
 (no_adj_sings : ∀ (arr : List ℕ), ∀ (i : ℕ), i < arr.length - 1 → arr[i]! > 1 ∨ arr[i + 1]! < 1 ∨ arr[i]! < 3 ∨ arr[i + 1]! > 2)
: num_ways_to_arrange_performances = 408 :=
sorry

end performance_arrangement_l773_773696


namespace sum_of_powers_of_a_eq_zero_l773_773695

variable (a b : ℝ)

theorem sum_of_powers_of_a_eq_zero
  (h1 : {a, b / a, 1} = {a^2, a + b, 0})
  (h2 : a ≠ 0)
  (h3 : a ≠ 1) :
  (∑ i in (range 2013), a^i) = 0 := sorry

end sum_of_powers_of_a_eq_zero_l773_773695


namespace solve_inequality_l773_773208

open Set

theorem solve_inequality (x : ℝ) (h : -3 * x^2 + 5 * x + 4 < 0 ∧ x > 0) : x ∈ Ioo 0 1 := by
  sorry

end solve_inequality_l773_773208


namespace remainder_when_divided_by_7_l773_773244

theorem remainder_when_divided_by_7 :
  let a := -1234
  let b := 1984
  let c := -1460
  let d := 2008
  (a * b * c * d) % 7 = 0 :=
by
  sorry

end remainder_when_divided_by_7_l773_773244


namespace calories_in_lemonade_l773_773434

theorem calories_in_lemonade :
  let lemon_juice := 100
  let sugar := 100
  let water := 400
  let total_weight := lemon_juice + sugar + water
  let lemon_juice_calories := 25 * lemon_juice / 100
  let sugar_calories := 386 * sugar / 100
  let total_calories := lemon_juice_calories + sugar_calories
  let desired_weight := 200
  let calories_per_gram := total_calories / total_weight
  let calories_in_desired_weight := calories_per_gram * desired_weight
  calories_in_desired_weight = 137 := by
    let lemon_juice := 100
    let sugar := 100
    let water := 400
    let total_weight := lemon_juice + sugar + water
    let lemon_juice_calories := 25 * lemon_juice / 100
    let sugar_calories := 386 * sugar / 100
    let total_calories := lemon_juice_calories + sugar_calories
    let desired_weight := 200
    let calories_per_gram := total_calories / total_weight
    let calories_in_desired_weight := calories_per_gram * desired_weight
    show calories_in_desired_weight = 137 from sorry

end calories_in_lemonade_l773_773434


namespace abhay_speed_l773_773124

variables (A S : ℝ)

theorem abhay_speed (h1 : 24 / A = 24 / S + 2) (h2 : 24 / (2 * A) = 24 / S - 1) : A = 12 :=
by {
  sorry
}

end abhay_speed_l773_773124


namespace find_remainder_l773_773961

theorem find_remainder :
  ∀ (D d q r : ℕ), 
    D = 18972 → 
    d = 526 → 
    q = 36 → 
    D = d * q + r → 
    r = 36 :=
by 
  intros D d q r hD hd hq hEq
  sorry

end find_remainder_l773_773961


namespace equal_perpendicular_diagonals_semicircles_equal_perpendicular_diagonals_fullcircles_l773_773582

noncomputable theory
open_locale classical

variables (A B C D O1 O2 O3 O4 : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
          [metric_space O1] [metric_space O2] [metric_space O3] [metric_space O4]

-- Define the quadrilateral and centers
def quadrilateral (ABCD : Type) := ∃ (A B C D : Type), true
def semicircle_center (A B O : Type) := ∃ (O : Type), true
def fullcircle_center (A B O : Type) := ∃ (O : Type), true

theorem equal_perpendicular_diagonals_semicircles 
  (ABCD : Type) (O1 O2 O3 O4 : Type)
  [quadrilateral ABCD]
  [semicircle_center A B O1] [semicircle_center B C O2]
  [semicircle_center C D O3] [semicircle_center D A O4] :
  (∃ (P Q R S : Type), true) → (P Q R S : Type) :=
sorry

theorem equal_perpendicular_diagonals_fullcircles 
  (ABCD : Type) (O1 O2 O3 O4 : Type)
  [quadrilateral ABCD]
  [fullcircle_center A B O1] [fullcircle_center B C O2]
  [fullcircle_center C D O3] [fullcircle_center D A O4] :
  (∃ (P Q R S : Type), true) → (P Q R S : Type) :=
sorry

end equal_perpendicular_diagonals_semicircles_equal_perpendicular_diagonals_fullcircles_l773_773582


namespace electronics_weight_l773_773980

theorem electronics_weight (B C E : ℝ) (h1 : B / C = 5 / 4) (h2 : B / E = 5 / 2) (h3 : B / (C - 9) = 10 / 4) : E = 9 := 
by 
  sorry

end electronics_weight_l773_773980


namespace train_cross_bridge_time_l773_773705

-- Definitions based on conditions
def carriages := 35
def length_each_carriage := 75 -- in meters
def length_engine := 75 -- in meters
def speed_kmph := 80 -- in kmph
def bridge_length_km := 5 -- in km

-- Conversion factors
def km_to_m := 1000 -- conversion factor from kilometers to meters
def hr_to_sec := 3600 -- conversion factor from hours to seconds

-- Total lengths and speeds
def total_train_length := (carriages * length_each_carriage) + length_engine -- sum of carriages and engine length in meters
def bridge_length := bridge_length_km * km_to_m -- length of the bridge in meters
def total_distance := total_train_length + bridge_length -- total distance to cover in meters
def speed_mps := (speed_kmph * km_to_m) / hr_to_sec -- speed in meters per second

-- Time calculations
def time_seconds := total_distance / speed_mps -- time in seconds
def time_minutes := time_seconds / 60 -- time in minutes

-- Main theorem statement
theorem train_cross_bridge_time : time_minutes ≈ 5.77 := 
sorry -- proof goes here

end train_cross_bridge_time_l773_773705


namespace number_of_correct_statements_is_three_l773_773433

variables {V : Type} [AddCommGroup V] [Module ℝ V]

-- Definitions of the conditions in the problem
def basis_transformation (a b c : V) (h : LinearIndependent ℝ ![a, b, c]) : 
  LinearIndependent ℝ ![a + b, a - b, c] := 
  sorry  -- We're not proving this, just stating it

def coplanar_vectors (u v : V) : Prop := ∃ s t : ℝ, s • u + t • v = (0 : V)

def parallel_lines (a b : V) : Prop := ∃ k : ℝ, a = k • b

def parallel_planes (u v : V) : u = (1 : ℝ, 2, -2) → v = (-2 : ℝ, -4, 4) → u = (-(2 : ℝ)) • v

-- Everything above here has defined the conditions
-- Now we formulate the theorem we aim to prove

/-- 
Theorem: The number of correct statements among the given four statements is 3.
-/
theorem number_of_correct_statements_is_three 
  (a b c : V) (h₁ : LinearIndependent ℝ ![a, b, c])
  (h₂ : basis_transformation a b c h₁ = false)
  (h₃ : ∀ (u v : V), coplanar_vectors u v)
  (h₄ : ∀ (l m : V), parallel_lines l m ↔ parallel_lines l m)
  (h₅ : ∀ (u v : V), parallel_planes u v u v) :
  3 = 3 :=
by {
  sorry
}

end number_of_correct_statements_is_three_l773_773433


namespace nine_odot_three_l773_773638

-- Defining the operation based on the given conditions
axiom odot_def (a b : ℕ) : ℕ

axiom odot_eq_1 : odot_def 2 4 = 8
axiom odot_eq_2 : odot_def 4 6 = 14
axiom odot_eq_3 : odot_def 5 3 = 13
axiom odot_eq_4 : odot_def 8 7 = 23

-- Proving that 9 ⊙ 3 = 21
theorem nine_odot_three : odot_def 9 3 = 21 := 
by
  sorry

end nine_odot_three_l773_773638


namespace total_peaches_l773_773941

-- Definitions of conditions
def initial_peaches : ℕ := 13
def picked_peaches : ℕ := 55

-- Proof problem statement
theorem total_peaches : initial_peaches + picked_peaches = 68 :=
by
  -- Including sorry to skip the actual proof
  sorry

end total_peaches_l773_773941


namespace solve_quadratic_equation_l773_773589

theorem solve_quadratic_equation (x : ℝ) : 
  (2 * x^2 - 4 * x = 6 - 3 * x) ↔ 
  (x = -3/2 ∨ x = 2) :=
by
  sorry

end solve_quadratic_equation_l773_773589


namespace max_sum_in_circles_l773_773714

-- Define the regions and their values
inductive Region
| A | B | C | D | E | F | G deriving DecidableEq

-- Define the circles and how they intersect
def Circle1 : Set Region := {Region.A, Region.B, Region.E}
def Circle2 : Set Region := {Region.A, Region.C, Region.F}
def Circle3 : Set Region := {Region.A, Region.D, Region.G}

-- Define the sum of values within a circle
def sum_values (values : Region → ℕ) (circle : Set Region) : ℕ :=
  circle.to_finset.sum (λ r => values r)

-- Define the condition to be proved:
theorem max_sum_in_circles (values : Region → ℕ) :
  (∀ circle ∈ [{Circle1}, {Circle2}, {Circle3}], sum_values values circle = 15) →
  False :=
sorry

end max_sum_in_circles_l773_773714


namespace find_least_n_l773_773162

def sequence (a : ℕ → ℕ) : Prop :=
  (a 15 = 20) ∧ ∀ n > 15, a n = 50 * a (n - 1) + n

theorem find_least_n : ∃ n : ℕ, n > 15 ∧ (∀ a : ℕ → ℕ, sequence a → a n % 35 = 0) ∧ ∀ m : ℕ, (m > 15 ∧ (∀ a : ℕ → ℕ, sequence a → a m % 35 = 0)) → n ≤ m := sorry

end find_least_n_l773_773162


namespace probability_of_abs_diff_greater_one_l773_773201

noncomputable def fair_die := uniformly_distributed (finset.range 1 7)

noncomputable def random_number (die_result : ℕ) : ℝ :=
  if die_result = 1 ∨ die_result = 2 then uniformly_between 0 2
  else if die_result = 3 ∨ die_result = 4 then
         if coin_flip = heads then 0 else 2
  else 1

def P_abs_diff_greater_one (x y : ℝ) : Prop :=
  |x - y| > 1

def question_probability : Prop :=
  let x := random_number (classical.some fair_die)
  let y := random_number (classical.some fair_die)
  probability (P_abs_diff_greater_one x y) = 1 / 3

theorem probability_of_abs_diff_greater_one : question_probability := 
sorry

end probability_of_abs_diff_greater_one_l773_773201


namespace Grunters_win_all_five_games_probability_l773_773225

theorem Grunters_win_all_five_games_probability :
  (let p_win := 3 / 5 in
   let p_all_wins := p_win ^ 5 in
   p_all_wins = 0.07776) :=
by sorry

end Grunters_win_all_five_games_probability_l773_773225


namespace ratio_of_areas_l773_773175

def supports (x y z a b c : ℝ) : Prop :=
  ( if x ≥ a then 1 else 0 ) +
  ( if y ≥ b then 1 else 0 ) +
  ( if z ≥ c then 1 else 0 ) = 2

def in_T (x y z : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 1

def T : Set (ℝ × ℝ × ℝ) :=
  {p | in_T p.1 p.2.1 p.2.2}

def S : Set (ℝ × ℝ × ℝ) :=
  {p | in_T p.1 p.2.1 p.2.2 ∧ supports p.1 p.2.1 p.2.2 (1/4) (1/5) (1/6)}

theorem ratio_of_areas :
  real.volume S / real.volume T = 2149 / 3600 := sorry

end ratio_of_areas_l773_773175


namespace height_of_tree_in_kilmer_park_l773_773612

-- Define the initial conditions
def initial_height_ft := 52
def growth_per_year_ft := 5
def years := 8
def ft_to_inch := 12

-- Define the expected result in inches
def expected_height_inch := 1104

-- State the problem as a theorem
theorem height_of_tree_in_kilmer_park :
  (initial_height_ft + growth_per_year_ft * years) * ft_to_inch = expected_height_inch :=
by
  sorry

end height_of_tree_in_kilmer_park_l773_773612


namespace height_of_each_step_l773_773143

-- Define the number of steps in each staircase
def first_staircase_steps : ℕ := 20
def second_staircase_steps : ℕ := 2 * first_staircase_steps
def third_staircase_steps : ℕ := second_staircase_steps - 10

-- Define the total steps climbed
def total_steps_climbed : ℕ := first_staircase_steps + second_staircase_steps + third_staircase_steps

-- Define the total height climbed
def total_height_climbed : ℝ := 45

-- Prove the height of each step
theorem height_of_each_step : (total_height_climbed / total_steps_climbed) = 0.5 := by
  sorry

end height_of_each_step_l773_773143


namespace sufficient_but_not_necessary_condition_l773_773439

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 2 → a^2 > 2 * a) ∧ ¬(a^2 > 2 * a → a > 2) :=
begin
  sorry
end

end sufficient_but_not_necessary_condition_l773_773439


namespace first_group_checked_correctly_l773_773400

-- Define the given conditions
def total_factories : ℕ := 169
def checked_by_second_group : ℕ := 52
def remaining_unchecked : ℕ := 48

-- Define the number of factories checked by the first group
def checked_by_first_group : ℕ := total_factories - checked_by_second_group - remaining_unchecked

-- State the theorem to be proved
theorem first_group_checked_correctly : checked_by_first_group = 69 :=
by
  -- The proof is not provided, use sorry to skip the proof steps
  sorry

end first_group_checked_correctly_l773_773400


namespace total_legs_correct_l773_773929

def num_ants : ℕ := 12
def num_spiders : ℕ := 8
def legs_per_ant : ℕ := 6
def legs_per_spider : ℕ := 8
def total_legs := num_ants * legs_per_ant + num_spiders * legs_per_spider

theorem total_legs_correct : total_legs = 136 :=
by
  sorry

end total_legs_correct_l773_773929


namespace city_tax_problem_l773_773511

theorem city_tax_problem :
  ∃ (x y : ℕ), 
    ((x + 3000) * (y - 10) = x * y) ∧
    ((x - 1000) * (y + 10) = x * y) ∧
    (x = 3000) ∧
    (y = 20) ∧
    (x * y = 60000) :=
by
  sorry

end city_tax_problem_l773_773511


namespace emily_pen_selections_is_3150_l773_773744

open Function

noncomputable def emily_pen_selections : ℕ :=
  (Nat.choose 10 4) * (Nat.choose 6 2)

theorem emily_pen_selections_is_3150 : emily_pen_selections = 3150 :=
by
  sorry

end emily_pen_selections_is_3150_l773_773744


namespace circumference_of_given_circle_l773_773955

theorem circumference_of_given_circle :
  let circle_eq := λ (x y : ℝ), x^2 + y^2 - 2*x + 6*y + 8 = 0 in
  ∃ (C : ℝ), C = 2 * real.sqrt 2 * real.pi ∧
             ∀ x y : ℝ, circle_eq x y ↔ (x - 1)^2 + (y + 3)^2 = 2 :=
by
  sorry

end circumference_of_given_circle_l773_773955


namespace angle_ACD_l773_773538

theorem angle_ACD {α β δ : Type*} [LinearOrderedField α] [CharZero α] (ABC DAB DBA : α)
  (h1 : ABC = 60) (h2 : BAC = 80) (h3 : DAB = 10) (h4 : DBA = 20):
  ACD = 30 := by
  sorry

end angle_ACD_l773_773538


namespace mo_tea_vs_hot_chocolate_l773_773190

theorem mo_tea_vs_hot_chocolate (n : ℕ) 
    (h1 : Mo drinks exactly n cups of hot chocolate on rainy days)
    (h2 : Mo drinks exactly 3 cups of tea on non-rainy days)
    (h3 : Last week Mo drank a total of 20 cups of tea and hot chocolate together)
    (h4 : Mo drank more tea cups than hot chocolate cups last week)
    (h5 : There were 2 rainy days last week) :
     (3 * 5 - 2 * (n : ℕ)) = 11 := 
by 
  sorry

end mo_tea_vs_hot_chocolate_l773_773190


namespace sequence_sum_l773_773877

theorem sequence_sum (a : ℕ → ℝ) (the_seq_is_arithmetic : ∀ n, a (n + 1) + a (n - 1) = 2 * a n)
  (specific_relation : ∀ n, a (n + 1) = a n ^ 2 - a (n - 1)) 
  (a_not_zero : ∀ n, a n ≠ 0) : 
  (∑ i in finset.range 2016, a i) = 4032 :=
by 
  sorry

end sequence_sum_l773_773877


namespace cost_price_computer_table_before_assembly_fee_l773_773619

noncomputable def compute_cost_price := sorry

theorem cost_price_computer_table_before_assembly_fee :
  let price_with_tax_discount := 3800
  let discount := 0.85
  let sales_tax := 1.08
  let assembly_fee := 200
  let chair_cost := 1000
  let chair_final_price := chair_cost * sales_tax
  let correct_cost_price := 1905.49 in
  let computer_table_price (C : ℝ) := (1.45 * C + assembly_fee) * sales_tax * discount + chair_final_price in
  computer_table_price correct_cost_price = price_with_tax_discount :=
begin
  sorry
end

end cost_price_computer_table_before_assembly_fee_l773_773619


namespace factor_theorem_example_l773_773101

theorem factor_theorem_example (k : ℤ) :
  ∃ k : ℤ, (∀ m: ℤ, (m - 8) ∣ (m^2 - k * m - 24)) → k = 5 :=
begin
  intro h,
  sorry
end

end factor_theorem_example_l773_773101


namespace Mrs_Early_speed_l773_773932

noncomputable def speed_to_reach_on_time (distance : ℝ) (ideal_time : ℝ) : ℝ := distance / ideal_time

theorem Mrs_Early_speed:
  ∃ (d t : ℝ), 
    (d = 50 * (t + 5/60)) ∧ 
    (d = 80 * (t - 7/60)) ∧ 
    (speed_to_reach_on_time d t = 59) := sorry

end Mrs_Early_speed_l773_773932


namespace length_of_chord_MN_l773_773971

theorem length_of_chord_MN 
  (m n : ℝ)
  (h1 : ∃ (M N : ℝ × ℝ), M ≠ N ∧ M.1 * M.1 + M.2 * M.2 + m * M.1 + n * M.2 - 4 = 0 ∧ N.1 * N.1 + N.2 * N.2 + m * N.1 + n * N.2 - 4 = 0 
    ∧ N.2 = M.1 ∧ N.1 = M.2) 
  (h2 : x + y = 0)
  : length_of_chord = 4 := sorry

end length_of_chord_MN_l773_773971


namespace smallest_positive_period_ratio_b_a_l773_773820

noncomputable def f (x : Real) : Real := 
  2 * Real.cos x ^ 2 - 2 * Real.sqrt 3 * Real.sin x * Real.sin (x - Real.pi / 2)

theorem smallest_positive_period : (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∃ x, f (x + T') ≠ f x) → T ≤ T') := 
by 
  sorry

theorem ratio_b_a (a b c A B C : ℝ) (hA : A = Real.pi / 3) (hB : B = Real.pi / 4) 
  (hS : Real.abs (S) = 1 / 4 * (a ^ 2 + c ^ 2 - b ^ 2)) : b / a = Real.sqrt 6 / 3 :=
by 
  sorry

end smallest_positive_period_ratio_b_a_l773_773820


namespace stand_in_line_with_block_l773_773126

theorem stand_in_line_with_block :
  let total_ways := 4! -- Total permutations of 4 students
  let block_ways := 2! * 3! -- Ways to arrange with 3 students in a block
  total_ways - block_ways = 12 := 
begin
  sorry
end

end stand_in_line_with_block_l773_773126


namespace least_number_with_remainder_l773_773274

theorem least_number_with_remainder (n : ℕ) (d₁ d₂ d₃ d₄ r : ℕ) 
  (h₁ : d₁ = 5) (h₂ : d₂ = 6) (h₃ : d₃ = 9) (h₄ : d₄ = 12) (hr : r = 184) :
  (∀ d, d ∈ [d₁, d₂, d₃, d₄] → n % d = r % d) → n = 364 := 
sorry

end least_number_with_remainder_l773_773274


namespace total_trees_planted_l773_773312

theorem total_trees_planted (x : ℕ) (workers_A workers_B total_workers : ℕ) (length_B_eq_4_A : workers_B * x - 1 = 4 * (workers_A * x - 1)) :
    19 * 3 = 57 :=
by
  have h1 : workers_A = 4 := rfl
  have h2 : workers_B = 15 := rfl
  have h3 : total_workers = 19 := rfl
  have h4 : workers_A + workers_B = total_workers := by rw [h1, h2]; exact rfl
  have hx : x = 3 := by 
    rw [h1, h2] at length_B_eq_4_A;
    linarith
  rw hx 
  exact rfl

end total_trees_planted_l773_773312


namespace jonah_added_yellow_raisins_l773_773145

variable (y : ℝ)

theorem jonah_added_yellow_raisins (h : y + 0.4 = 0.7) : y = 0.3 := by
  sorry

end jonah_added_yellow_raisins_l773_773145


namespace number_of_integer_solutions_l773_773491

theorem number_of_integer_solutions :
  { x : ℝ | (x - 3)^(36 - x^2) = 1 }.count = 4 :=
by
  sorry

end number_of_integer_solutions_l773_773491


namespace problem_solution_exists_l773_773382

theorem problem_solution_exists (x : ℝ) (h : ∃ x, 2 * (3 * 5 - x) - x = -8) : x = 10 :=
sorry

end problem_solution_exists_l773_773382


namespace remainder_7_pow_150_mod_12_l773_773278

theorem remainder_7_pow_150_mod_12 :
  (7^150) % 12 = 1 := sorry

end remainder_7_pow_150_mod_12_l773_773278


namespace inequality_half_l773_773094

-- The Lean 4 statement for the given proof problem
theorem inequality_half (a b : ℝ) (h : a > b) : (a / 2 > b / 2) :=
sorry


end inequality_half_l773_773094


namespace nonneg_real_inequality_l773_773210

theorem nonneg_real_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
    a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 :=
sorry

end nonneg_real_inequality_l773_773210


namespace range_c_l773_773769

theorem range_c (c : ℝ) :
  c > 0 ∧ c ≠ 1 →
  (∀ x : ℝ, monotone_decreasing (λ x, c^x)) →
  (∀ x : ℝ, x > 1/2 → deriv (λ x, x^2 - 2 * c * x + 1) x > 0) →
  ¬ ((∀ x : ℝ, monotone_decreasing (λ x, c^x)) ∧ (∀ x : ℝ, x > 1/2 → deriv (λ x, x^2 - 2 * c * x + 1) x > 0)) →
  ((∀ x : ℝ, monotone_decreasing (λ x, c^x)) ∨ (∀ x : ℝ, x > 1/2 → deriv (λ x, x^2 - 2 * c * x + 1) x > 0)) →
  (1 / 2 < c ∧ c < 1) :=
by
  sorry

end range_c_l773_773769


namespace total_population_l773_773873

-- Define the predicates for g, b, and s based on t
variables (g b t s : ℕ)

-- The conditions given in the problem
def condition1 : Prop := g = 4 * t
def condition2 : Prop := b = 6 * g
def condition3 : Prop := s = t / 2

-- The theorem stating the total population is equal to (59 * t) / 2
theorem total_population (g b t s : ℕ) (h1 : condition1 g t) (h2 : condition2 b g) (h3 : condition3 s t) :
  b + g + t + s = 59 * t / 2 :=
by sorry

end total_population_l773_773873


namespace tan_2alpha_and_cos_beta_l773_773017

theorem tan_2alpha_and_cos_beta
    (α β : ℝ)
    (h1 : 0 < β ∧ β < α ∧ α < (Real.pi / 2))
    (h2 : Real.sin α = (4 * Real.sqrt 3) / 7)
    (h3 : Real.cos (β - α) = 13 / 14) :
    Real.tan (2 * α) = -(8 * Real.sqrt 3) / 47 ∧ Real.cos β = 1 / 2 := by
  sorry

end tan_2alpha_and_cos_beta_l773_773017


namespace praveen_hari_profit_ratio_l773_773937

theorem praveen_hari_profit_ratio
  (praveen_capital : ℕ := 3360)
  (hari_capital : ℕ := 8640)
  (time_praveen_invested : ℕ := 12)
  (time_hari_invested : ℕ := 7)
  (praveen_shares_full_time : ℕ := praveen_capital * time_praveen_invested)
  (hari_shares_full_time : ℕ := hari_capital * time_hari_invested)
  (gcd_common : ℕ := Nat.gcd praveen_shares_full_time hari_shares_full_time) :
  (praveen_shares_full_time / gcd_common) * 2 = 2 ∧ (hari_shares_full_time / gcd_common) * 2 = 3 := by
    sorry

end praveen_hari_profit_ratio_l773_773937


namespace no_conditional_statements_l773_773469

def requires_conditional_statements_1 (x : ℝ) : Prop := ∀ y : ℝ, y = -x → true

def requires_conditional_statements_2 : Prop := 4 * Real.sqrt 6 > 0

def requires_conditional_statements_3 (a b c : ℝ) : Prop := ¬ (a ≤ b ∧ b ≤ c ∨ a ≤ c ∧ c ≤ b ∨ b ≤ a ∧ a ≤ c ∨ b ≤ c ∧ c ≤ a ∨ c ≤ a ∧ a ≤ b ∨ c ≤ b ∧ b ≤ a)

def requires_conditional_statements_4 (x : ℝ) : Prop := (x ≥ 0 ∧ (x - 1) > 0) ∨ (x < 0 ∧ (x + 2) > 0)

theorem no_conditional_statements (h1 : ∀ x : ℝ, requires_conditional_statements_1 x)
  (h2 : requires_conditional_statements_2)
  (h3 : ∀ a b c : ℝ, requires_conditional_statements_3 a b c)
  (h4 : ∀ x : ℝ, requires_conditional_statements_4 x) : 
  \#(λ (q : Nat), q = (if (h1 ∨ h2) then 1 else 0) + 
  (if (h3 ∨ h4) then 0 else 1)) = 2 :=
sorry

end no_conditional_statements_l773_773469


namespace perimeter_of_triangle_l773_773536

variable {A B C D : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variable (ABC : Triangle B C D)
variable (D : Point B C) (A : Point)
variable [has_sqrt : has_sqrt ℝ]

-- Conditions
def BC_length_eq_one (ABC : Triangle B C D) :=
  dist ABC.B ABC.C = 1

def exists_unique_D (A B C D : Point) :=
  ∃! D, dist D A ^ 2 = dist D B * dist D C

-- The proof statement
theorem perimeter_of_triangle (ABC : Triangle B C D) (hBC : BC_length_eq_one ABC) (hD : exists_unique_D A B C D) : 
  perimeter ABC = sqrt 2 + 1 :=
sorry

end perimeter_of_triangle_l773_773536


namespace voting_result_l773_773262

-- Definitions for the candidates
inductive Candidate
| Dupont | Dubois | Durand
open Candidate

-- Definitions for the brothers
inductive Brother
| Pierre | Jean | Jacques
open Brother

-- Functions defining who each brother votes for
def votes_for : Brother → Candidate → Prop
| Brother.Pierre, Candidate.Dupont => 
  ∀ (vote_Jean vote_Jacques : Candidate), 
    (vote_Jean = Candidate.Dubois → votes_for Brother.Pierre Candidate.Dupont) ∧
    (vote_Jean = Candidate.Durand → votes_for Brother.Pierre Candidate.Dubois) ∧
    (vote_Jacques = Candidate.Dupont → votes_for Brother.Pierre Candidate.Durand)
| Brother.Jean, Candidate.Duran => 
  ∀ (vote_Pierre vote_Jacques : Candidate), 
    (vote_Pierre = Candidate.Duran → ¬(votes_for Brother.Jean Candidate.Dupont)) ∧
    (vote_Jacques = Candidate.Dubois → votes_for Brother.Jean Candidate.Dupont)
| Brother.Jacques, Candidate.Dupont => 
  ∀ (vote_Pierre : Candidate), 
    (vote_Pierre = Candidate.Dupont → ¬(votes_for Brother.Jacques Candidate.Durand))
| _, _ => false

-- The proof statement that under specified conditions the brothers voted as identified.
theorem voting_result :
  votes_for Brother.Pierre Candidate.Dubois ∧
  votes_for Brother.Jean Candidate.Duran ∧
  votes_for Brother.Jacques Candidate.Dupont :=
by
  sorry

end voting_result_l773_773262


namespace train_cross_bridge_time_l773_773349

/--
A train 160 meters long is travelling at 45 km/hr and can cross a bridge of 215 meters in a certain amount of time. 
We are to prove it takes the train 30 seconds to cross the bridge.
-/
theorem train_cross_bridge_time :
  let train_length := 160 in
  let bridge_length := 215 in
  let speed_km_hr := 45 in
  let total_distance := train_length + bridge_length in
  let speed_m_s := (speed_km_hr * 1000) / 3600 in
  (total_distance : ℝ) / (speed_m_s : ℝ) = 30 := 
by
  sorry

end train_cross_bridge_time_l773_773349


namespace integer_roots_k_values_y1_y2_squared_sum_k_0_y1_y2_squared_sum_k_neg1_l773_773047

theorem integer_roots_k_values (k : ℤ) :
  (∀ x : ℤ, k * x ^ 2 + (2 * k - 1) * x + k - 1 = 0) →
  k = 0 ∨ k = -1 :=
sorry

theorem y1_y2_squared_sum_k_0 (m y1 y2: ℝ) :
  (m > -2) →
  (k = 0) →
  ((k - 1) * y1 ^ 2 - 3 * y1 + m = 0) →
  ((k - 1) * y2 ^ 2 - 3 * y2 + m = 0) →
  y1^2 + y2^2 = 9 + 2 * m :=
sorry

theorem y1_y2_squared_sum_k_neg1 (m y1 y2: ℝ) :
  (m > -2) →
  (k = -1) →
  ((k - 1) * y1 ^ 2 - 3 * y1 + m = 0) →
  ((k - 1) * y2 ^ 2 - 3 * y2 + m = 0) →
  y1^2 + y2^2 = 9 / 4 + m :=
sorry

end integer_roots_k_values_y1_y2_squared_sum_k_0_y1_y2_squared_sum_k_neg1_l773_773047


namespace value_ab_plus_a_plus_b_l773_773552

noncomputable def polynomial : Polynomial ℝ := Polynomial.C (-1) + Polynomial.X * Polynomial.C (-1) + Polynomial.X^2 * Polynomial.C (-4) + Polynomial.X^4

theorem value_ab_plus_a_plus_b {a b : ℝ} (h : polynomial.eval a = 0 ∧ polynomial.eval b = 0) : a * b + a + b = -1 / 2 :=
sorry

end value_ab_plus_a_plus_b_l773_773552


namespace inequality_holds_l773_773092

theorem inequality_holds (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) : a^2 + b^2 ≥ 2 :=
sorry

end inequality_holds_l773_773092


namespace smallest_possible_sum_l773_773791

-- Defining the conditions for x and y.
variables (x y : ℕ)

-- We need a theorem to formalize our question with the given conditions.
theorem smallest_possible_sum (hx : x > 0) (hy : y > 0) (hne : x ≠ y) (hxy : 1/x + 1/y = 1/24) : x + y = 100 :=
by
  sorry

end smallest_possible_sum_l773_773791


namespace find_S9_l773_773883

-- Definitions for the problem condition
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- Constant definitions from the problem
constant a : ℕ → ℤ 
constant S : ℕ → ℤ 
constant a5 : a 5 = 6

-- Main theorem to prove
theorem find_S9 (arith : arithmetic_seq a) (sum_def : sum_first_terms a S) : S 9 = 54 :=
sorry

end find_S9_l773_773883


namespace cone_volume_l773_773631

/-- 
Given that the volume of a cylinder is 72π cm^3,
prove that the volume of a cone with the same radius and height 
as the cylinder is 24π cm^3.
-/
theorem cone_volume (V_cyl : ℝ) (h_cyl : V_cyl = 72 * Real.pi) : 
  ∃ V_cone : ℝ, V_cone = 24 * Real.pi :=
by
  have V_cone := (1 / 3 : ℝ) * V_cyl
  use V_cone
  rw [h_cyl]
  norm_num
  sorry

end cone_volume_l773_773631


namespace point_A_coordinates_l773_773969

variable (a x y : ℝ)

def f (a x : ℝ) : ℝ := (a^2 - 1) * (x^2 - 1) + (a - 1) * (x - 1)

theorem point_A_coordinates (h1 : ∃ t : ℝ, ∀ x : ℝ, f a x = t * x + t) (h2 : x = 0) : (0, 2) = (0, f a 0) :=
by
  sorry

end point_A_coordinates_l773_773969


namespace cost_of_books_purchasing_plans_l773_773189

theorem cost_of_books (x y : ℕ) (h1 : 4 * x + 2 * y = 480) (h2 : 2 * x + 3 * y = 520) : x = 50 ∧ y = 140 :=
by
  -- proof can be filled in later
  sorry

theorem purchasing_plans (a b : ℕ) (h_total_cost : 50 * a + 140 * (20 - a) ≤ 1720) (h_quantity : a ≤ 2 * (20 - b)) : (a = 12 ∧ b = 8) ∨ (a = 13 ∧ b = 7) :=
by
  -- proof can be filled in later
  sorry

end cost_of_books_purchasing_plans_l773_773189


namespace smallest_sum_minimum_l773_773783

noncomputable def smallest_sum (x y : ℕ) : ℕ :=
if h₁ : x ≠ y ∧ (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) then x + y else 0

theorem smallest_sum_minimum (x y : ℕ) (h₁ : x ≠ y) (h₂ : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) :
  smallest_sum x y = 96 := sorry

end smallest_sum_minimum_l773_773783


namespace find_M4_l773_773471

noncomputable def M (M_0 : ℝ) (t : ℝ) : ℝ :=
  M_0 * (1.2 ^ (- t / 2))

theorem find_M4 (M_0 : ℝ) (H : - (1 / 2) * real.log 1.2 * M_0 * 1.2 ^ (-1) = -10 * real.log 1.2) :
  M M_0 4 = 50 / 3 :=
by
  sorry

end find_M4_l773_773471


namespace reciprocal_2023_l773_773986

def reciprocal (x : ℕ) := 1 / x

theorem reciprocal_2023 : reciprocal 2023 = 1 / 2023 :=
by
  sorry

end reciprocal_2023_l773_773986


namespace base_solution_l773_773425

theorem base_solution (b : ℕ) : 161_b + 134_b = 315_b → b = 8 :=
sorry

end base_solution_l773_773425


namespace count_five_digit_integers_with_sum_of_digits_two_l773_773251

/-- 
  Problem statement:
  Prove that there are exactly 5 five-digit positive integers whose digits sum up to 2,
  and do not start with zero.
-/

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := List.ofFn (fun i => Nat.digit n i) in
  List.sum digits

theorem count_five_digit_integers_with_sum_of_digits_two :
  ∃ (S : Finset ℕ), S.card = 5 ∧ ∀ n ∈ S, n >= 10000 ∧ n < 100000 ∧ sum_of_digits n = 2 :=
sorry

end count_five_digit_integers_with_sum_of_digits_two_l773_773251


namespace find_principal_l773_773756

-- Given conditions
def annual_interest_rate : ℝ := 0.05
def time_in_years : ℝ := 2.4
def final_amount : ℝ := 896

-- Required to find the principal
def principal : ℝ := final_amount / (1 + annual_interest_rate * time_in_years)

-- The proof goal
theorem find_principal (R T A : ℝ) 
  (hR : R = annual_interest_rate) 
  (hT : T = time_in_years) 
  (hA : A = final_amount) : 
  principal = 800 :=
by
  rw [hR, hT, hA]
  dsimp [principal, annual_interest_rate, time_in_years, final_amount]
  norm_num
  rfl
-- Here we use sorry to skip the proof but have set the structure
-- sorry

end find_principal_l773_773756


namespace min_PQ_distance_l773_773854

-- Define the curves
def curve1 (x : ℝ) : ℝ := -x^2 - 1
def curve2 (y : ℝ) : ℝ := 1 + y^2

-- Define the distance formula between points
def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the points P and Q according to the curves
def P_def (x : ℝ) : ℝ × ℝ := (x, curve1 x)
def Q_def (y : ℝ) : ℝ × ℝ := (curve2 y, y)

-- Define the main theorem statement
theorem min_PQ_distance : 
  ∃ x y : ℝ, (x, -x^2 - 1) = P_def x ∧ (1 + y^2, y) = Q_def y ∧ distance (P_def x) (Q_def y) = 3 * Real.sqrt 2 / 4 := sorry

end min_PQ_distance_l773_773854


namespace cameras_not_in_both_l773_773581

-- Definitions for the given conditions
def shared_cameras : ℕ := 12
def sarah_cameras : ℕ := 24
def mike_unique_cameras : ℕ := 9

-- The proof statement
theorem cameras_not_in_both : (sarah_cameras - shared_cameras) + mike_unique_cameras = 21 := by
  sorry

end cameras_not_in_both_l773_773581


namespace fleet_arrangement_l773_773331

theorem fleet_arrangement :
  let C (n k : ℕ) := Nat.choose n k in
  let P (n : ℕ) := n! in
  let choose_and_arrange (n k : ℕ) := C n k * P k in
  let total_ways := choose_and_arrange 7 4 in
  let ways_A_only := choose_and_arrange 6 3 in
  let ways_B_only := choose_and_arrange 6 3 in
  let ways_A_and_B := choose_and_arrange 5 2 * (P 3 * P 2 / 2) in
  let arrangements := total_ways - ways_A_and_B in
  arrangements = 600 :=
begin
  -- We'll refine the exact steps inside the begin-end block
  sorry
end

end fleet_arrangement_l773_773331


namespace domain_of_function_l773_773735

theorem domain_of_function :
  {x : ℝ | x >= 4} ∩ {x : ℝ | x ≠ 5} = {x : ℝ | x ∈ [4, 5) ∪ (5, +∞)} := 
by {
  sorry
}

end domain_of_function_l773_773735


namespace range_f_in_interval_l773_773058

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)
noncomputable def g (ω φ x : ℝ) : ℝ := f ω φ (x + π / 3)

theorem range_f_in_interval (ω φ : ℝ) (hω : ω > 0) (hφ : abs φ < π / 2)
  (h_dist_sym_axes : ∃ T, T = π ∧ (fun x => f ω φ x) x = f ω φ (x + T)) 
  (h_even_g : ∀ x, g ω φ x = g ω φ (-x)) :
  (∀ x, x ∈ Icc (-π / 6) (π / 6) → f 2 (-π / 6) x ∈ Icc (-2 : ℝ) 1) :=
sorry

end range_f_in_interval_l773_773058


namespace rectangle_ratio_l773_773202

theorem rectangle_ratio (AB BC BD FM θ : ℝ) (ABCD_congruent_ABEF: ℝ) (D_AB_E_dihedral: ℝ) (M_midpoint_AB: ℝ)
  (sin_theta : sin θ = sqrt 78 / 9) :
  AB / BC = sqrt 2 / 2 :=
by sorry

end rectangle_ratio_l773_773202


namespace square_difference_l773_773289

theorem square_difference : (601^2 - 599^2 = 2400) :=
by {
  -- Placeholder for the proof
  sorry
}

end square_difference_l773_773289


namespace focus_with_greater_x_l773_773364

-- Definitions for given conditions
def center : ℝ × ℝ := (3, -2)
def semi_major_axis : ℝ := 3.1
def semi_minor_axis : ℝ := 3

-- The focus with greater x-coordinate according to the solution is (3.39, -2)
theorem focus_with_greater_x :
  let f := (√((semi_major_axis)^2 - (semi_minor_axis)^2) / 2)
  in (center.1 + f, center.2) = (3.39, -2) :=
by
  sorry

end focus_with_greater_x_l773_773364


namespace solution_set_of_quadratic_inequality_l773_773994

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | 2 - x - x^2 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
sorry

end solution_set_of_quadratic_inequality_l773_773994


namespace even_func_root_sum_l773_773798

theorem even_func_root_sum (f : ℝ → ℝ) 
  (hf : ∀ x, f x = f (-x)) 
  (hf_roots : ∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0) : 
  let roots := {r : ℝ | f r = 0} in 
  ∑ r in roots, r = 0 := 
by
  -- proof would go here
  sorry

end even_func_root_sum_l773_773798


namespace triangle_area_ratio_l773_773006

theorem triangle_area_ratio
  (A B C X : Type)
  (AB BC AC : ℝ)
  (hAB : AB = 40)
  (hBC : BC = 45)
  (hAC : AC = 50)
  (h_bisect : CX bisects ∠ACB) :
  (area_of_triangle B C X / area_of_triangle A C X) = 9 / 10 := by
  sorry

end triangle_area_ratio_l773_773006


namespace cookie_cost_1_l773_773142

theorem cookie_cost_1 (C : ℝ) 
  (h1 : ∀ c, c > 0 → 1.2 * c = c + 0.2 * c)
  (h2 : 50 * (1.2 * C) = 60) :
  C = 1 :=
by
  sorry

end cookie_cost_1_l773_773142


namespace max_g_10_l773_773164

theorem max_g_10 (g : ℝ → ℝ) (h_poly : ∃ (n : ℕ) (b : ℕ → ℝ), g = λ x, ∑ i in (finset.range (n + 1)), b i * x^i) 
  (h_nonneg : ∀ x, 0 ≤ g x )
  (h1 : g 5 = 80)
  (h2 : g 20 = 2560) : g 10 ≤ 452 :=
sorry

end max_g_10_l773_773164


namespace max_slope_of_tangent_line_l773_773055

def f (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d
def curve (x : ℝ) : ℝ := x^2 * sin x + x * cos x

theorem max_slope_of_tangent_line :
  (∃ (p : ℝ), p ∈ set.Icc (π/3) (2*π/3) ∧ curve p = f p) ∧ f(0) = 0 ∧
  f'(0) = 0 ∧ a < 0 ∧ b > 0 →
  (∀ x, f' x ≤ f' (π/2)) → (f'(π/2) = 3π/4) :=
sorry

end max_slope_of_tangent_line_l773_773055


namespace sin_double_angle_l773_773507

theorem sin_double_angle (α : ℝ) (h : sin α = -2 * cos α) : sin (2 * α) = -4 / 5 :=
sorry

end sin_double_angle_l773_773507


namespace find_n_given_sum_l773_773024

noncomputable def geometric_sequence_general_term (n : ℕ) : ℝ :=
  if n ≥ 2 then 2^(2 * n - 3) else 0

def b_n (n : ℕ) : ℝ :=
  2 * n - 3

def sum_b_n (n : ℕ) : ℝ :=
  n^2 - 2 * n

theorem find_n_given_sum : ∃ n : ℕ, sum_b_n n = 360 :=
  by { use 20, sorry }

end find_n_given_sum_l773_773024


namespace average_marks_of_failed_boys_l773_773524

def total_boys : ℕ := 120
def average_marks_all_boys : ℝ := 35
def number_of_passed_boys : ℕ := 100
def average_marks_passed_boys : ℝ := 39
def number_of_failed_boys : ℕ := total_boys - number_of_passed_boys

noncomputable def total_marks_all_boys : ℝ := average_marks_all_boys * total_boys
noncomputable def total_marks_passed_boys : ℝ := average_marks_passed_boys * number_of_passed_boys
noncomputable def total_marks_failed_boys : ℝ := total_marks_all_boys - total_marks_passed_boys
noncomputable def average_marks_failed_boys : ℝ := total_marks_failed_boys / number_of_failed_boys

theorem average_marks_of_failed_boys :
  average_marks_failed_boys = 15 :=
by
  -- The proof can be filled in here
  sorry

end average_marks_of_failed_boys_l773_773524


namespace log_27_gt_point_53_l773_773307

open Real

theorem log_27_gt_point_53 :
  log 27 > 0.53 :=
by
  sorry

end log_27_gt_point_53_l773_773307


namespace decreasing_interval_l773_773617

def f (x : ℝ) := Real.log (x^2 - x - 2)

theorem decreasing_interval :
  ∀ x : ℝ, x < -1 → f(x) < f(x + 1) ∧ (x > 2 → f(x) < f(x + 1)) :=
by
  sorry

end decreasing_interval_l773_773617


namespace sum_of_digits_next_palindrome_l773_773691

def is_palindrome (n : ℕ) : Prop :=
  let digits := (n.to_string.to_list)
  digits = digits.reverse

theorem sum_of_digits_next_palindrome :
  ∃ n : ℕ, n > 13931 ∧ is_palindrome n ∧ (n.digits.sum = 10) :=
sorry

end sum_of_digits_next_palindrome_l773_773691


namespace integer_solutions_abs_sum_eq_n_l773_773487

theorem integer_solutions_abs_sum_eq_n (n : ℕ) : 
  { p : ℤ × ℤ // |p.1| + |p.2| = n }.card = 4 * n :=
by sorry

end integer_solutions_abs_sum_eq_n_l773_773487


namespace footballers_squad_numbers_l773_773574

theorem footballers_squad_numbers (n k : ℕ) (forwards goalkeepers : Finset ℕ) (goals : Finset (ℕ × ℕ)) :
  forwards.card + goalkeepers.card = n ∧ goals.card = k ∧
  ∀ (g ∈ goals), (∃ f ∈ forwards, ∃ gk ∈ goalkeepers, g = (f, gk)) →
  ∃ (squad_numbers : ℕ → ℕ), squad_numbers ∈ Finset.range n ∧ 
   (∀ (f gk) ∈ goals, (f, gk) = g → abs (squad_numbers f - squad_numbers gk) > n - k) :=
sorry

end footballers_squad_numbers_l773_773574


namespace john_friends_count_l773_773183

-- Define the initial conditions
def initial_amount : ℚ := 7.10
def cost_of_sweets : ℚ := 1.05
def amount_per_friend : ℚ := 1.00
def remaining_amount : ℚ := 4.05

-- Define the intermediate values
def after_sweets : ℚ := initial_amount - cost_of_sweets
def given_away : ℚ := after_sweets - remaining_amount

-- Define the final proof statement
theorem john_friends_count : given_away / amount_per_friend = 2 :=
by
  sorry

end john_friends_count_l773_773183


namespace problem_conditions_l773_773059

variable (x : ℝ) (a : ℝ)

def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a * x - 1

-- Conditions
theorem problem_conditions (hx : 0 < x) (hyp : f 1 1 = 0) :
  e^(-x) + x * (Real.log x + x - 1) ≥ 0 :=
by sorry

end problem_conditions_l773_773059


namespace unique_01_sequence_l773_773920

open Nat

-- Define the condition: n is a nonzero natural number
variable (n : ℕ) (hn : n > 0)

-- Define the function for the sequence with the described property
def unique_sequence (n : ℕ) : List (List ℕ) := 
  List.range (n + 1).map (λ k => List.replicate (n - k) 0 ++ List.replicate k 1)

-- Define the main proposition
theorem unique_01_sequence : 
  ∃! (a : ℕ → ℕ), 
    (∀ i, 0 ≤ i ∧ i ≤ n^2 - n → 
      (List.range (n+1).map (λ j => a (i + 1 + j))).sum < 
      (List.range (n+1).map (λ j => a (i + 1 + n + j))).sum) :=
sorry

end unique_01_sequence_l773_773920


namespace magnitude_of_z_l773_773772

open Complex

noncomputable def z : ℂ := (1 - I) / (1 + I) + 2 * I

theorem magnitude_of_z : Complex.abs z = 1 := by
  sorry

end magnitude_of_z_l773_773772


namespace men_in_second_scenario_l773_773849

-- Definitions based on conditions
def men1 := 4
def hours_day1 := 10
def days_week := 7
def earnings1 := 1400
def earnings2 := 1890.0000000000002
def hours_day2 := 6

-- Total man-hours in scenario 1
def total_man_hours1 := men1 * hours_day1 * days_week

-- Rate per man-hour
def rate_per_man_hour := earnings1 / total_man_hours1

-- Total man-hours needed in scenario 2
def total_man_hours2 := earnings2 / rate_per_man_hour

-- Number of men in scenario 2
def men2 := total_man_hours2 / (hours_day2 * days_week)

-- Theorem proving the number of men in the second scenario is 9
theorem men_in_second_scenario : men2 = 9 := by
  sorry

end men_in_second_scenario_l773_773849


namespace sergeant_reaches_end_in_minimum_moves_l773_773635

def trench_positions := Fin 9
def niche_positions := Fin 3

structure TrenchState where
  people : Fin 12 → Option (Fin 10)  -- Represents positions 0-8 for trench, 9-11 for niches

def initial_trench_state : TrenchState :=
  ⟨ fun n => if n < 9 then some n.toNat else none ⟩

def move (state : TrenchState) (from to : Fin 12) : TrenchState :=
  { state with people := state.people ∘ fun n =>
    if n == from then none
    else if n == to then state.people from
    else state.people n }

def sergeant_final_position (state : TrenchState) : Prop :=
  state.people 8 = some 0  -- Sergeant (represented by 0) should be at position 8 (end of trench)

noncomputable def valid_moves : List (Fin 12 × Fin 12) :=
  [(1, 10), (2, 1), (3, 2), (0, 10), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7), (10, 8), (7, 8), (6, 7), (8, 6), (6, 7), (5, 6), (7, 6)]

noncomputable def move_sequence (s₀ : TrenchState) (moves : List (Fin 12 × Fin 12)) : TrenchState :=
  moves.foldl (fun s (p : Fin 12 × Fin 12) => move s p.1 p.2) s₀

theorem sergeant_reaches_end_in_minimum_moves :
  sergeant_final_position (move_sequence initial_trench_state valid_moves) :=
by
  sorry

end sergeant_reaches_end_in_minimum_moves_l773_773635


namespace find_line_equation_l773_773447

-- Define the general equation of the line with slope 2 and y-intercept m
def line_equation (m : ℝ) : ∀ x, 2 * x + m = sorry

-- Define the equation of the hyperbola
def hyperbola : ∀ x y, x^2 / 3 - y^2 / 2 = 1 := sorry

-- Define the condition that the distance between intersection points A and B is sqrt(6)
def distance_condition (x₁ y₁ x₂ y₂ : ℝ) : (x₁ - x₂)^2 + (y₁ - y₂)^2 = 6 := sorry

-- Main statement to prove
theorem find_line_equation (m : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
     hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ 
     distance_condition x₁ y₁ x₂ y₂ ∧
     ∀ x, y₁ = line_equation m x ∧ y₂ = line_equation m x
  ) →
  (∀ m, m^2 = 15 → line_equation m = y = 2x + sqrt(15) ∨ line_equation m = y = 2x - sqrt(15)) :=
sorry

end find_line_equation_l773_773447


namespace product_divisors_60_prime_factors_l773_773157

theorem product_divisors_60_prime_factors : 
  ∃ (A : ℕ), (A = ∏ d in (Finset.filter (λ d, d ∣ 60) (Finset.range (60+1))), d) ∧ 
             (nat.factors A).to_finset.card = 3 := 
begin
  sorry
end

end product_divisors_60_prime_factors_l773_773157


namespace second_order_derivative_correct_l773_773420

noncomputable def second_order_derivative : ℝ :=
  (1 + 3 * t^2) * (1 + t^2)

theorem second_order_derivative_correct (t : ℝ) :
  ∃ (y_xx : ℝ), y_xx = second_order_derivative :=
begin
  use (1 + 3 * t^2) * (1 + t^2),
  refl,
end

end second_order_derivative_correct_l773_773420


namespace sum_converges_to_half_l773_773007

theorem sum_converges_to_half :
  ∀ n : ℕ, ∑ k in finset.range (n+1), ((3^k) / (9^k - 1)) ≤ 0.5 :=
sorry

end sum_converges_to_half_l773_773007


namespace tank_emptying_time_l773_773370

theorem tank_emptying_time (
  (fill_time_no_leak : ℝ) (fill_time_with_leak : ℝ) 
  (h_fill_time_no_leak : fill_time_no_leak = 10) 
  (h_fill_time_with_leak : fill_time_with_leak = 11) 
) : 
  let R := 1 / fill_time_no_leak in
  let R_effective := 1 / fill_time_with_leak in
  let L := R - R_effective in
  let T := 1 / L in
  T = 109.89 :=
by {
  have h_R : R = 0.1,
  by {
    rw h_fill_time_no_leak,
    show R = 1 / 10,
    norm_num,
  },
  have h_R_effective : R_effective = 0.0909, 
  by {
    rw h_fill_time_with_leak,
    show R_effective = 1 / 11,
    norm_num,
  },
  have h_L : L = 0.0091, 
  by {
    show L = 0.1 - 0.0909,
    norm_num,
  },
  have h_T : T = 1 / L, 
  by {
    show T = 1 / 0.0091,
    norm_num,
  },
  show T = 109.89,
  by {
    rw h_T,
    norm_num,
  },
  sorry
}

end tank_emptying_time_l773_773370


namespace mohan_least_cookies_l773_773567

theorem mohan_least_cookies :
  ∃ b : ℕ, 
    b % 6 = 5 ∧
    b % 8 = 3 ∧
    b % 9 = 6 ∧
    b = 59 :=
by
  sorry

end mohan_least_cookies_l773_773567


namespace cyclic_quad_angle_ABC_gt_120_l773_773867

open EuclideanGeometry

theorem cyclic_quad_angle_ABC_gt_120
  (A B C D : Point)
  (h_cyclic : CyclicQuadrilateral A B C D)
  (h_eq : dist A B = dist A D)
  (h_ineq : dist C D > dist A B + dist B C) :
  angle A B C > 120 := 
begin
  sorry
end

end cyclic_quad_angle_ABC_gt_120_l773_773867


namespace bus_stop_time_per_hour_l773_773748

-- Define the conditions
def speed_excluding_stoppages := 65 -- km/h
def speed_including_stoppages := 48 -- km/h
def distance := 17 -- km, the distance not covered due to stoppages

-- Define the conversion from hours to minutes
def km_per_minute (km_per_hour : ℕ) : ℚ := km_per_hour / 60

-- Main statement of the Lean theorem
theorem bus_stop_time_per_hour : 
  let time := distance / (km_per_minute speed_excluding_stoppages) in 
  time ≈ 15.7 :=
sorry

end bus_stop_time_per_hour_l773_773748


namespace cos_alpha_of_point_l773_773448

theorem cos_alpha_of_point (-4 3) : ∃ (α : ℝ), let P := (-4, 3) in
    P.1^2 + P.2^2 = 5^2 ∧ cos α = -4 / 5 :=
begin
  -- We've set up the terms, using the given condition, and now state that
  -- under these conditions, the cosine of the angle is -4/5. 
  sorry
end

end cos_alpha_of_point_l773_773448


namespace unique_two_digit_numbers_count_l773_773049

theorem unique_two_digit_numbers_count :
  ∃ n : ℕ, n = 12 ∧ 
    (∀ a b : ℕ, a ∈ {3, 5, 7, 8} ∧ b ∈ {3, 5, 7, 8} ∧ a ≠ b → (10 * a + b) > 9) := 
begin
  use 12,
  split,
  { refl, },
  { intros a b ha hb hab,
    rcases ha with rfl | rfl | rfl | rfl;
    rcases hb with rfl | rfl | rfl | rfl;
    repeat {norm_num at hab, contradiction},
    all_goals { norm_num },
  },
  sorry
end


end unique_two_digit_numbers_count_l773_773049


namespace min_value_x_y_l773_773506

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) : x + y ≥ 16 :=
sorry

end min_value_x_y_l773_773506


namespace correct_regression_equation_l773_773991

variable (x y : ℝ)

-- Assume that y is negatively correlated with x
axiom negative_correlation : x * y ≤ 0

-- The candidate regression equations
def regression_A : ℝ := -2 * x - 100
def regression_B : ℝ := 2 * x - 100
def regression_C : ℝ := -2 * x + 100
def regression_D : ℝ := 2 * x + 100

-- Prove that the correct regression equation reflecting the negative correlation is regression_C
theorem correct_regression_equation : regression_C x = -2 * x + 100 := by
  sorry

end correct_regression_equation_l773_773991


namespace num_pairs_satisfying_eq_l773_773418

theorem num_pairs_satisfying_eq :
  ∃ n : ℕ, (n = 256) ∧ (∀ x y : ℤ, x^2 + x * y = 30000000 → true) :=
sorry

end num_pairs_satisfying_eq_l773_773418


namespace hyperbola_eccentricity_l773_773060

-- Define the conditions
variables (a b : ℝ) (c e : ℝ)
variable ha : a > 0
variable hb : b > 0
variable h1 : c^2 = a^2 + b^2

-- Define the hyperbola and the point M
def hyperbola_eq (x y : ℝ) : Prop := (x^2) / a^2 - (y^2) / b^2 = 1

def point_M (M : ℝ × ℝ) : Prop := 
  ∃ (x y : ℝ), hyperbola_eq x y ∧ M = (c / 2, (sqrt 3) * c / 2)

theorem hyperbola_eccentricity :
  ∀ (M : ℝ × ℝ), point_M M → e = sqrt 3 + 1 :=
by
  intro M hM
  sorry

end hyperbola_eccentricity_l773_773060


namespace passes_through_point_l773_773847

theorem passes_through_point (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : (1, 1) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1))} :=
by
  sorry

end passes_through_point_l773_773847


namespace find_a_of_even_function_l773_773179

-- Define the function f
def f (x a : ℝ) := (x + 1) * (x + a)

-- State the theorem to be proven
theorem find_a_of_even_function (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = -1 :=
by
  -- The actual proof goes here
  sorry

end find_a_of_even_function_l773_773179


namespace find_total_price_l773_773975

-- Define the cost parameters
variables (sugar_price salt_price : ℝ)

-- Define the given conditions
def condition_1 : Prop := 2 * sugar_price + 5 * salt_price = 5.50
def condition_2 : Prop := sugar_price = 1.50

-- Theorem to be proven
theorem find_total_price (h1 : condition_1 sugar_price salt_price) (h2 : condition_2 sugar_price) : 
  3 * sugar_price + 1 * salt_price = 5.00 :=
by
  sorry

end find_total_price_l773_773975


namespace mean_age_of_children_l773_773595

theorem mean_age_of_children :
  let ages := [8, 8, 12, 12, 10, 14]
  let n := ages.length
  let sum_ages := ages.foldr (· + ·) 0
  let mean_age := sum_ages / n
  mean_age = 10 + 2 / 3 :=
by
  sorry

end mean_age_of_children_l773_773595


namespace complex_division_l773_773915

theorem complex_division : (let i := Complex.I in (i - 2) / i = 1 + 2 * i) := by
  sorry

end complex_division_l773_773915


namespace simplify_and_evaluate_expr_l773_773205

-- Define the expressions and condition
def expr (a : ℝ) : ℝ := (a / (a + 2)) - ((a + 3) / (a^2 - 4)) / ((2 * a + 6) / (2 * a^2 - 8 * a + 8))

def a_value : ℝ := abs (-6) - (1 / 2)⁻¹

-- The theorem we want to prove
theorem simplify_and_evaluate_expr : expr a_value = 1 / 3 := by
  sorry

end simplify_and_evaluate_expr_l773_773205


namespace lexie_paintings_l773_773182

theorem lexie_paintings (n m : ℕ) (h1 : n = 4) (h2 : m = 8) : n * m = 32 := by
  rw [h1, h2]
  norm_num

end lexie_paintings_l773_773182


namespace angle_ECD_measure_l773_773861

-- Define the vertices A, B, C, D, E
variables {A B C D E : Type}

-- Define angles and their measures
variables [inner_product_space ℝ B]
variables {m : A → B → A → ℝ}

-- Conditions from the problem
variable h1 : ∀ (x y z : A), AC = BC
variable h2 : m D C B = 60
variable h3 : is_perpendicular C D A B

-- Goal
theorem angle_ECD_measure : m E C D = 60 :=
by sorry

end angle_ECD_measure_l773_773861


namespace distance_between_A_and_B_l773_773412

def point (α : Type) := ℝ × ℝ

def distance (p1 p2 : point ℝ) : ℝ := 
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

axiom pointA : point ℝ := (4, -6)
axiom pointB : point ℝ := (-8, 5)

theorem distance_between_A_and_B : distance pointA pointB = Real.sqrt 265 := by
  sorry

end distance_between_A_and_B_l773_773412


namespace total_right_handed_players_l773_773935

-- Define the number of players, throwers, and fraction of left-handed non-throwers.
variables (P : ℕ) (T : ℕ) (one_third : ℚ)

-- Assume required conditions.
axiom total_players_seventy : P = 70
axiom throwers_fifty_two : T = 52
axiom one_third_fraction : one_third = 1 / 3

-- Define the statement to prove the total number of right-handed players.
theorem total_right_handed_players (P T : ℕ) (one_third : ℚ)
  (h1 : P = 70) (h2 : T = 52) (h3 : one_third = 1 / 3) : 
  P - T = 18 ∧ (1 / 3:ℚ * 18 = 6) ∧ (18 - 6 = 12) ∧ (T + 12 = 64) :=
by
  sorry

end total_right_handed_players_l773_773935


namespace ellipse_standard_equation_and_circle_property_l773_773045

theorem ellipse_standard_equation_and_circle_property :
  (∃ (x y : ℝ), (∃ c : ℝ, c = 2 * Real.sqrt 2) ∧ (6 = 2 * 3) ∧
                 (x^2 / 9 + y^2 = 1)) ∧ 
  (let line_equation := λ x : ℝ, x + 2 in
  ∃ (x₁ y₁ x₂ y₂ : ℝ), (y₁ = line_equation x₁) ∧ (y₂ = line_equation x₂) ∧ 
                       (x₁^2 / 9 + y₁^2 = 1) ∧ (x₂^2 / 9 + y₂^2 = 1) ∧
                       ((x₁ * x₂ + y₁ * y₂) + (x₁ + x₂) + 4 ≠ 0)) := sorry

end ellipse_standard_equation_and_circle_property_l773_773045


namespace projection_of_AB_on_AC_l773_773453

open Matrix

noncomputable def point := ℝ × ℝ × ℝ

def A := (1 : ℝ, -1 : ℝ, 2 : ℝ)
def B := (5 : ℝ, -6 : ℝ, 2 : ℝ)
def C := (1 : ℝ, 3 : ℝ, -1 : ℝ)

def vector_sub (p q : point) : point :=
  (p.1 - q.1, p.2 - q.2, p.3 - q.3)

def dot_product (u v : point) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def norm (v : point) : ℝ :=
  Real.sqrt (dot_product v v)

def projection (u v : point) : ℝ :=
  (dot_product u v) / (norm v)

theorem projection_of_AB_on_AC :
  projection (vector_sub B A) (vector_sub C A) = -4 :=
by
  sorry

end projection_of_AB_on_AC_l773_773453


namespace combination_indices_l773_773088
open Nat

theorem combination_indices (x : ℕ) (h : choose 18 x = choose 18 (3 * x - 6)) : x = 3 ∨ x = 6 :=
by
  sorry

end combination_indices_l773_773088


namespace system_of_equations_solve_l773_773067

theorem system_of_equations_solve (x y : ℝ) 
  (h1 : 2 * x + y = 5)
  (h2 : x + 2 * y = 4) :
  x + y = 3 :=
by
  sorry

end system_of_equations_solve_l773_773067


namespace hyperbola_focus_coordinates_l773_773614

theorem hyperbola_focus_coordinates :
  let a := 7
  let b := 11
  let h := 5
  let k := -3
  let c := Real.sqrt (a^2 + b^2)
  (∃ x y : ℝ, (x = h + c ∧ y = k) ∧ (∀ x' y', (x' = h + c ∧ y' = k) ↔ (x = x' ∧ y = y'))) :=
by
  sorry

end hyperbola_focus_coordinates_l773_773614


namespace margaret_total_cost_l773_773012

noncomputable def cost_margaret_spends_on_bread
  (num_people : ℕ)
  (sandwiches_per_person : ℕ)
  (num_bread_types : ℕ)
  (mini_croissants_price : ℕ → ℕ)
  (ciabatta_rolls_price : ℕ → ℕ)
  (multigrain_bread_price : ℕ → ℕ)
  (discount_threshold : ℕ)
  (discount_rate : ℚ)
  (tax_rate : ℚ)
  (packs_needed : ℕ → ℕ) : ℚ :=
let total_sandwiches := num_people * sandwiches_per_person in
let sandwiches_per_type := total_sandwiches / num_bread_types in
let total_cost_before_discount := 
  (packs_needed mini_croissants_price sandwiches_per_type) * (mini_croissants_price 12) +
  (packs_needed ciabatta_rolls_price sandwiches_per_type) * (ciabatta_rolls_price 10) +
  (packs_needed multigrain_bread_price sandwiches_per_type) * (multigrain_bread_price 20) in
let discounted_cost := if total_cost_before_discount ≥ discount_threshold 
                       then total_cost_before_discount * (1 - discount_rate)
                       else total_cost_before_discount in
let total_cost_after_tax := discounted_cost * (1 + tax_rate) in
total_cost_after_tax

def margaret_condition := 
  cost_margaret_spends_on_bread 24 2 3 
    (λ n, if n = 12 then 8 else 0) 
    (λ n, if n = 10 then 9 else 0)
    (λ n, if n = 20 then 7 else 0)
    50
    0.1
    0.07
    (λ price sandwiches_per_type, if price sandwiches_per_type >= sandwiches_per_type then 2 else 0)

theorem margaret_total_cost : margaret_condition = 51.36 := 
by sorry

end margaret_total_cost_l773_773012


namespace smallest_positive_debt_l773_773268

theorem smallest_positive_debt :
  ∃ (D : ℕ) (p g : ℤ), 0 < D ∧ D = 350 * p + 240 * g ∧ D = 10 := sorry

end smallest_positive_debt_l773_773268


namespace train_cross_time_l773_773136

noncomputable def train_length : ℝ := 317.5
noncomputable def train_speed_kph : ℝ := 153.3
noncomputable def convert_speed_to_mps (speed_kph : ℝ) : ℝ :=
  (speed_kph * 1000) / 3600

noncomputable def train_speed_mps : ℝ := convert_speed_to_mps train_speed_kph
noncomputable def time_to_cross_pole (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

theorem train_cross_time :
  time_to_cross_pole train_length train_speed_mps = 7.456 :=
by 
  -- This is where the proof would go
  sorry

end train_cross_time_l773_773136


namespace solve_problem_l773_773043

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := (2^x + b) / (2^x + 1)

theorem solve_problem :
  (∀ x : ℝ, f x b = -f (-x) b) →
  f 0 b = 0 →
  f 2x - 1 = 1 - (2 / (2^x + 1)) ∧
  (∀ x : ℝ,
    (f (2 * x + 1) (-1) + f x (-1) < 0) = (x < -1 / 3)) :=
by
  sorry

end solve_problem_l773_773043


namespace max_expression_value_l773_773275

open Real

theorem max_expression_value : 
  ∃ q : ℝ, ∀ q : ℝ, -3 * q ^ 2 + 18 * q + 5 ≤ 32 ∧ (-3 * (3 ^ 2) + 18 * 3 + 5 = 32) :=
by
  sorry

end max_expression_value_l773_773275


namespace complex_ineq_proof_l773_773914

theorem complex_ineq_proof
  (a b x y : ℝ)
  (h₁ : x^2 + y^2 ≤ 1)
  (h₂ : a^2 + b^2 ≤ 2) :
  |b * (x^2 - y^2) + 2 * a * x * y| ≤ √2 := 
sorry

end complex_ineq_proof_l773_773914


namespace problem_inequality_l773_773033

theorem problem_inequality (n : ℕ) (a : ℕ → ℝ)
  (h1 : a 1 = 0)
  (h2 : ∀ k, 2 ≤ k → k ≤ n → |a k| = |a (k - 1) + 1|) :
  (1 / n) * (∑ k in Finset.range n.succ, a (k + 1)) ≥ -1 / 2 :=
sorry

end problem_inequality_l773_773033


namespace point_outside_circle_l773_773804

theorem point_outside_circle (P Q : ℝ → ℝ) (r : ℝ) (hPQ : dist P Q = 2) (hr : r = 1.5) : dist P Q > r := 
by
  rw [hPQ, hr]
  norm_num
  exact sorry

end point_outside_circle_l773_773804


namespace complement_U_A_l773_773810

def U : Set ℝ := Set.univ

def A : Set ℝ := { x | |x - 1| > 1 }

theorem complement_U_A : (U \ A) = { x | 0 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end complement_U_A_l773_773810


namespace excelling_in_mathematics_or_physics_excelling_in_mathematics_or_physics_or_chemistry_l773_773118

variable (A₁ A₂ A₃ : Set Nat)
variable (n₁ n₂ n₃ n₁₂ n₂₃ n₃₁ n₁₂₃ : Nat)

axiom h1 : |A₁| = 30
axiom h2 : |A₂| = 28
axiom h3 : |A₃| = 25
axiom h4 : |A₁ ∩ A₂| = 20
axiom h5 : |A₂ ∩ A₃| = 16
axiom h6 : |A₃ ∩ A₁| = 17
axiom h7 : |A₁ ∩ A₂ ∩ A₃| = 10

theorem excelling_in_mathematics_or_physics :
  |A₁ ∪ A₂| = 38 := by sorry

theorem excelling_in_mathematics_or_physics_or_chemistry :
  |A₁ ∪ A₂ ∪ A₃| = 40 := by sorry

end excelling_in_mathematics_or_physics_excelling_in_mathematics_or_physics_or_chemistry_l773_773118


namespace factorial_sum_div_l773_773725

theorem factorial_sum_div : ((8.factorial + 9.factorial) / 6.factorial) = 560 := by
  sorry

end factorial_sum_div_l773_773725


namespace ThaboRatio_l773_773594

-- Define the variables
variables (P_f P_nf H_nf : ℕ)

-- Define the conditions as hypotheses
def ThaboConditions := P_f + P_nf + H_nf = 280 ∧ P_nf = H_nf + 20 ∧ H_nf = 55

-- State the theorem we want to prove
theorem ThaboRatio (h : ThaboConditions P_f P_nf H_nf) : (P_f / P_nf) = 2 :=
by sorry

end ThaboRatio_l773_773594


namespace median_salary_is_25000_l773_773684

def Position := String

def employees : List (Position × Nat × Int) :=
  [ ("CEO", 1, 140000),
    ("General Manager", 7, 95000),
    ("Senior Manager", 8, 80000),
    ("Manager", 4, 55000),
    ("Staff", 43, 25000) ]

theorem median_salary_is_25000 :
  let num_employees := employees.foldl (fun acc (_, count, _) => acc + count) 0
  let median_position := (num_employees + 1) / 2
  let cumulative_counts := employees.scanl (fun acc (_, count, _) => acc + count) 0
  let median_salary := employees.foldl (fun acc (pos, count, salary) =>
      if acc.fst then (acc.fst, acc.snd)
      else if acc.snd + count >= median_position then (true, salary)
      else (false, acc.snd + count)
    ) (false, 0)
  median_salary.snd = 25000 :=
by
  -- We specify the cumulative counts which are correct
  have cumulative_counts : List Nat := [0, 1, 8, 16, 20, 63]
  -- Calculate number of employees, which is 63
  have num_employees : Nat := 63
  -- Calculate median position, which is 32
  have median_position : Nat := 32
  -- Locate the median salary according to the cumulative counts
  have median_salary : Int := 25000
  exact rfl

end median_salary_is_25000_l773_773684


namespace coordinates_change_l773_773464

variable {V : Type} [AddCommGroup V] [Module ℝ V]
variable {a b c p : V}
variable hb : LinearIndependent ℝ ![a, b, c]
variable hb' : LinearIndependent ℝ ![a + b, a - b, c]
variable hp : p = 1 • a - 2 • b + 3 • c

theorem coordinates_change :
  ∃ x y z : ℝ, p = x • (a + b) + y • (a - b) + z • c ∧ x = -1/2 ∧ y = 3/2 ∧ z = 3 :=
by {
  -- Proof to be completed.
  sorry
}

end coordinates_change_l773_773464


namespace Toby_second_part_distance_l773_773641

noncomputable def total_time_journey (distance_unloaded_second: ℝ) : ℝ :=
  18 + (distance_unloaded_second / 20) + 8 + 7

theorem Toby_second_part_distance:
  ∃ d : ℝ, total_time_journey d = 39 ∧ d = 120 :=
by
  use 120
  unfold total_time_journey
  sorry

end Toby_second_part_distance_l773_773641


namespace avg_price_of_racket_l773_773698

theorem avg_price_of_racket (total_revenue : ℝ) (pairs_sold : ℝ) (h1 : total_revenue = 686) (h2 : pairs_sold = 70) : 
  total_revenue / pairs_sold = 9.8 := by
  sorry

end avg_price_of_racket_l773_773698


namespace acme_corp_five_letter_words_l773_773356

open Finset

theorem acme_corp_five_letter_words (A E I O U Y : Nat) (A_limit : A = 3) (other_limit : E = 5 ∧ I = 5 ∧ O = 5 ∧ U = 5 ∧ Y = 5) :
  (A + E + I + O + U + Y = 28) →
  (number_of_valid_words : Nat := (5 ^ 5) + (5 * 5 ^ 4) + (10 * 5 ^ 3) + (10 * 5 ^ 2)) = 7750 :=
by
  sorry

end acme_corp_five_letter_words_l773_773356


namespace magnitude_of_root_of_quadratic_eq_l773_773458

open Complex

theorem magnitude_of_root_of_quadratic_eq (z : ℂ) 
  (h : z^2 - (2 : ℂ) * z + 2 = 0) : abs z = Real.sqrt 2 :=
by 
  sorry

end magnitude_of_root_of_quadratic_eq_l773_773458


namespace campers_in_morning_correct_l773_773591

-- Definitions
def campers_morning := 36
def campers_afternoon := 13
def campers_evening := 49
def campers_total := 98

-- Hypothesis
def total_campers_eq := campers_morning + campers_afternoon + campers_evening = campers_total

-- Goal
theorem campers_in_morning_correct : campers_morning = 36 :=
by
  unfold campers_morning campers_afternoon campers_evening campers_total total_campers_eq
  norm_num
  sorry

end campers_in_morning_correct_l773_773591


namespace difference_is_1365_l773_773958

-- Define the conditions as hypotheses
def difference_between_numbers (L S : ℕ) : Prop :=
  L = 1637 ∧ L = 6 * S + 5

-- State the theorem to prove the difference is 1365
theorem difference_is_1365 {L S : ℕ} (h₁ : L = 1637) (h₂ : L = 6 * S + 5) :
  L - S = 1365 :=
by
  sorry

end difference_is_1365_l773_773958


namespace length_AB_l773_773121

namespace RightTriangleMedians

variables (A B C : Type) [RealScalar A] [RealScalar B] [RealScalar C]
variables {AM BN : ℝ}

-- Definitions derived from conditions
noncomputable def is_right_triangle (ABC : Triangle) : Prop :=
  ABC.angle C = 90

noncomputable def median_from_A_to_BC (A : Point) (BC : Segment) := 
  is_median A BC

noncomputable def median_from_B_to_AC (B : Point) (AC : Segment) := 
  is_median B AC

-- Given conditions
axiom length_AM : AM = 5
axiom length_BN : BN = 3 * (sqrt 5)

-- Proof problem
theorem length_AB (ABC : Triangle) : 
  is_right_triangle ABC → 
  median_from_A_to_BC A ABC.BC → 
  median_from_B_to_AC B ABC.AC →
  length_AM = 5 → 
  length_BN = 3 * (sqrt 5) → 
  ∃ (AB : ℝ), AB = 2 * (sqrt 14) :=
sorry

end RightTriangleMedians

end length_AB_l773_773121


namespace minimum_value_l773_773057

theorem minimum_value (x : ℝ) (hx : 0 ≤ x) : ∃ y : ℝ, y = x^2 - 6 * x + 8 ∧ (∀ t : ℝ, 0 ≤ t → y ≤ t^2 - 6 * t + 8) :=
sorry

end minimum_value_l773_773057


namespace proj_u_on_t_l773_773485

-- Given vectors
def u : ℝ × ℝ := (4, -3)
def t : ℝ × ℝ := (-6, 8)

-- Compute projections
def proj (u t : ℝ × ℝ) : ℝ × ℝ :=
  let dot (x y : ℝ × ℝ) := x.1 * y.1 + x.2 * y.2
  let scalar := dot u t / dot t t
  (scalar * t.1, scalar * t.2)

-- The equivalent proof goal
theorem proj_u_on_t : 
  proj u t = (2.88, -3.84) :=
by
  sorry

end proj_u_on_t_l773_773485


namespace probability_at_least_one_white_ball_l773_773317

/-!
Given conditions:
- A bag contains a total of 6 black and white balls of the same size.
- The probability of drawing a black ball from the bag is 2/3.

Prove that:
The probability of drawing at least one white ball when two balls are randomly drawn from the bag is 3/5.
-/

theorem probability_at_least_one_white_ball :
  ∀ (total_balls black_balls : ℕ) (P_black : ℚ),
  total_balls = 6 →
  black_balls / total_balls.to_rat = 2 / 3 →
  (C black_balls 1 * C (total_balls - black_balls) 1 + C (total_balls - black_balls) 2) / C total_balls 2 = 3 / 5 :=
begin
  sorry
end

end probability_at_least_one_white_ball_l773_773317


namespace intersection_sets_l773_773482

theorem intersection_sets (x : ℝ) :
  let M := {x | 2 * x - x^2 ≥ 0 }
  let N := {x | -1 < x ∧ x < 1}
  M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_sets_l773_773482


namespace inequality_solution_l773_773995

-- Define the inequality condition
def fraction_inequality (x : ℝ) : Prop :=
  (3 * x - 1) / (x - 2) ≤ 0

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  1 / 3 ≤ x ∧ x < 2

-- The theorem to prove that the inequality's solution matches the given solution set
theorem inequality_solution (x : ℝ) (h : fraction_inequality x) : solution_set x :=
  sorry

end inequality_solution_l773_773995


namespace jelly_bean_ratio_l773_773187

theorem jelly_bean_ratio 
  (Napoleon_jelly_beans : ℕ)
  (Sedrich_jelly_beans : ℕ)
  (Mikey_jelly_beans : ℕ)
  (h1 : Napoleon_jelly_beans = 17)
  (h2 : Sedrich_jelly_beans = Napoleon_jelly_beans + 4)
  (h3 : Mikey_jelly_beans = 19) :
  2 * (Napoleon_jelly_beans + Sedrich_jelly_beans) / Mikey_jelly_beans = 4 := 
sorry

end jelly_bean_ratio_l773_773187


namespace triangle_area_when_a_is_one_range_of_a_if_inequality_holds_l773_773056

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a * x - 1

theorem triangle_area_when_a_is_one :
  let a := 1
  let tangent := (y : ℝ) → (Real.exp 1 + 1) * y - (Math.exp 1) in
  let x_intercept := 1 / (Real.exp 1 + 1) in
  let y_intercept := -1 in
  1 / 2 * x_intercept * y_intercept = 1 / (2 * (Real.exp 1 + 1)) :=
by sorry

theorem range_of_a_if_inequality_holds :
  (∀ x : ℝ, (0 < x ∧ x < 1) → f x a ≥ x ^ 2) → a ≥ 2 - Real.exp 1 :=
by sorry

end triangle_area_when_a_is_one_range_of_a_if_inequality_holds_l773_773056


namespace center_of_circle_polar_coords_l773_773534

theorem center_of_circle_polar_coords :
  ∀ (θ : ℝ), ∃ (ρ : ℝ), (ρ, θ) = (2, Real.pi) ∧ ρ = - 4 * Real.cos θ := 
sorry

end center_of_circle_polar_coords_l773_773534


namespace perpendicular_squares_equal_sum_l773_773539

theorem perpendicular_squares_equal_sum {A B C P A1 B1 C1 : Point}
  (h1 : is_perpendicular A1 B C P)
  (h2 : is_perpendicular B1 C A P)
  (h3 : is_perpendicular C1 A B P)
  (h4 : (dist A C1) ^ 2 + (dist P C1) ^ 2 = (dist A B1) ^ 2 + (dist P B1) ^ 2)
  (h5 : (dist B A1) ^ 2 + (dist P A1) ^ 2 = (dist B C1) ^ 2 + (dist P C1) ^ 2)
  (h6 : (dist C B1) ^ 2 + (dist P B1) ^ 2 = (dist C A1) ^ 2 + (dist P A1) ^ 2) :
  (dist A C1) ^ 2 + (dist B A1) ^ 2 + (dist C B1) ^ 2 = (dist A B1) ^ 2 + (dist B C1) ^ 2 + (dist C A1) ^ 2 :=
by
  sorry

end perpendicular_squares_equal_sum_l773_773539


namespace min_area_ratio_l773_773809

def Triangle (α β γ : ℝ) : Type :=
  {α β γ : ℝ // α + β + γ = 180 ∧ α > 0 ∧ β > 0 ∧ γ > 0}

noncomputable def isosceles_right_triangle (α β γ : ℝ) (hαβγ : α + β + γ = 180) : Prop :=
  (α = 90 ∧ β = γ) ∨ (β = 90 ∧ α = γ) ∨ (γ = 90 ∧ α = β)

noncomputable def area_of_triangle (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def triangle_areas_relation (t1 t2 : Triangle) (PQR_on_ABC : ∀ t1.vertices_on t2) : ℝ :=
  area_of_triangle t1.a t1.b t1.c t1.ha t1.hb t1.hc / area_of_triangle t2.a t2.b t2.c t2.ha t2.hb t2.hc

theorem min_area_ratio:
  ∀ (α1 β1 γ1 α2 β2 γ2 : ℝ)
  (h_tri_PQR : Triangle α1 β1 γ1)
  (h_tri_ABC : Triangle α2 β2 γ2),
  isosceles_right_triangle α1 β1 γ1 →
  isosceles_right_triangle α2 β2 γ2 →
  (∀ PQR_on_ABC : ∀ h_tri_PQR.vertices_on h_tri_ABC),
  triangle_areas_relation h_tri_PQR h_tri_ABC PQR_on_ABC = 1 / 5 :=
sorry

end min_area_ratio_l773_773809


namespace inequality_proof_l773_773446

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2 / 3 :=
sorry

end inequality_proof_l773_773446


namespace unique_polynomial_g_l773_773560

noncomputable def f : ℕ → ℕ
| 0       := 0
| (n + 1) := n + 1 - f (f n)

theorem unique_polynomial_g (g : ℝ → ℝ) :
  (∀ n : ℕ, f n = ⌊ g n ⌋) →
  ∃ a b : ℝ, (∀ x : ℝ, g x = a * (x + 1) + b) ∧ (a = (sqrt 5 - 1) / 2) ∧ (b = (sqrt 5 - 1) / 2) :=
begin
  sorry
end

end unique_polynomial_g_l773_773560


namespace even_sum_probability_l773_773764

theorem even_sum_probability:
  let tiles := (1: ℕ) :: (2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: 11 :: 12 :: 13 :: 14 :: []) in
  let n := 14 in
  let players := 3 in
  let tiles_per_player := 3 in
  let total_combinations := choose n tiles_per_player * choose (n - tiles_per_player) tiles_per_player * choose (n - 2 * tiles_per_player) tiles_per_player in
  let even_condition := 
    (choose 7 3) * (choose 7 2 * choose 5 2 * choose 4 1 * choose 3 1) * players in 
  let probability := even_condition / total_combinations in
  let (m, n) := nat.gcdext (even_condition : ℕ) (total_combinations : ℕ) in
  m + n = 1379 :=
by
  sorry

end even_sum_probability_l773_773764


namespace Alice_more_nickels_l773_773710

-- Define quarters each person has
def Alice_quarters (q : ℕ) : ℕ := 10 * q + 2
def Bob_quarters (q : ℕ) : ℕ := 2 * q + 10

-- Prove that Alice has 40(q - 1) more nickels than Bob
theorem Alice_more_nickels (q : ℕ) : 
  (5 * (Alice_quarters q - Bob_quarters q)) = 40 * (q - 1) :=
by
  sorry

end Alice_more_nickels_l773_773710


namespace mengers_theorem_l773_773557

-- Definitions:
variables {V : Type*} (G : Graph V) (A B : set V) (k : ℕ)

-- Conditions:
def minimum_vertex_separation (G : Graph V) (A B : set V) : ℕ := sorry

def maximum_disjoint_AB_paths (G : Graph V) (A B : set V) : ℕ := sorry

theorem mengers_theorem (G : Graph V) (A B : set V) :
  minimum_vertex_separation G A B = maximum_disjoint_AB_paths G A B := sorry

end mengers_theorem_l773_773557


namespace percentage_of_sikh_boys_l773_773872

def total_boys := 850
def percent_muslim := 0.44
def percent_hindu := 0.28
def other_community_boys := 153

theorem percentage_of_sikh_boys : 
  (total_boys - (percent_muslim * total_boys).toInt - (percent_hindu * total_boys).toInt - other_community_boys).toFloat / total_boys * 100 = 10 :=
by
  sorry

end percentage_of_sikh_boys_l773_773872


namespace right_angled_triangle_with_different_colors_exists_l773_773740

theorem right_angled_triangle_with_different_colors_exists
  (grid_point_color : ℕ × ℕ → ℕ)
  (used_colors : {c : ℕ | c = 1 ∨ c = 2 ∨ c = 3})
  (all_colors_used : ∀ (c : {c | c = 1 ∨ c = 2 ∨ c = 3}), ∃ (p : ℕ × ℕ), grid_point_color p = c)
  : ∃ (A B C : ℕ × ℕ),
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    right_angle_triangle A B C ∧
    (grid_point_color A, grid_point_color B, grid_point_color C).pairwise (≠) :=
sorry

end right_angled_triangle_with_different_colors_exists_l773_773740


namespace train_cross_bridge_time_l773_773350

/--
A train 160 meters long is travelling at 45 km/hr and can cross a bridge of 215 meters in a certain amount of time. 
We are to prove it takes the train 30 seconds to cross the bridge.
-/
theorem train_cross_bridge_time :
  let train_length := 160 in
  let bridge_length := 215 in
  let speed_km_hr := 45 in
  let total_distance := train_length + bridge_length in
  let speed_m_s := (speed_km_hr * 1000) / 3600 in
  (total_distance : ℝ) / (speed_m_s : ℝ) = 30 := 
by
  sorry

end train_cross_bridge_time_l773_773350


namespace billy_videos_within_limit_l773_773365

def total_videos_watched_within_time_limit (time_limit : ℕ) (video_time : ℕ) (search_time : ℕ) (break_time : ℕ) (num_trials : ℕ) (videos_per_trial : ℕ) (categories : ℕ) (videos_per_category : ℕ) : ℕ :=
  let total_trial_time := videos_per_trial * video_time + search_time + break_time
  let total_category_time := videos_per_category * video_time
  let full_trial_time := num_trials * total_trial_time
  let full_category_time := categories * total_category_time
  let total_time := full_trial_time + full_category_time
  let non_watching_time := search_time * num_trials + break_time * (num_trials - 1)
  let available_time := time_limit - non_watching_time
  let max_videos := available_time / video_time
  max_videos

theorem billy_videos_within_limit : total_videos_watched_within_time_limit 90 4 3 5 5 15 2 10 = 13 := by
  sorry

end billy_videos_within_limit_l773_773365


namespace range_of_m_l773_773108

theorem range_of_m (a : ℝ) (m : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (hx : ∃ x < 0, a^x = 3 * m - 2) :
  1 < m :=
sorry

end range_of_m_l773_773108


namespace maximum_rooks_on_chessboard_l773_773649

-- Define the condition of the chessboard
def Chessboard : Type := { b : Fin 8 × Fin 8 // b.1 < 8 ∧ b.2 < 8 }

-- Define the relationship of attack between two rooks
def attacks (r1 r2 : Chessboard) : Prop :=
  r1.val.1 = r2.val.1 ∨ r1.val.2 = r2.val.2

-- Define a valid placement of rooks
def is_valid (placement : Set Chessboard) : Prop :=
  ∀ r1 r2 ∈ placement, (r1 ≠ r2) → attacks r1 r2 → False ∧
  ∀ r ∈ placement, ∃ r1 r2 ∈ placement, r ≠ r1 ∧ r ≠ r2 ∧ attacks r r1 ∧ attacks r r2

theorem maximum_rooks_on_chessboard : ∃ (placement : Set Chessboard), 
  is_valid placement ∧ cardinality placement = 10 := sorry

end maximum_rooks_on_chessboard_l773_773649


namespace reciprocal_of_2023_l773_773982

theorem reciprocal_of_2023 : 1 / 2023 = (1 : ℚ) / 2023 :=
by sorry

end reciprocal_of_2023_l773_773982


namespace triangle_area_from_curve_l773_773495

-- Definition of the curve
def curve (x : ℝ) : ℝ := (x - 2)^2 * (x + 3)

-- Area calculation based on intercepts
theorem triangle_area_from_curve : 
  (1 / 2) * (2 - (-3)) * (curve 0) = 30 :=
by
  sorry

end triangle_area_from_curve_l773_773495


namespace find_possible_m_values_l773_773535

-- Definitions of points and distances
def point (x y z : ℝ) := (x, y, z)
def distance (p q : ℝ × ℝ × ℝ) : ℝ := (Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 + (p.3 - q.3) ^ 2))

-- Given conditions
def A := point 2 2 2
noncomputable def distance_A_to_plane := 1
def B (m : ℝ) := point m 0 0
noncomputable def distance_B_to_plane := 4

-- Proof goal
theorem find_possible_m_values (m : ℝ) :
  ∃ m, (√((m - 2)^2 + 2^2 + 2^2) = 3) ↔ (m = 1 ∨ m = 3) :=
sorry

end find_possible_m_values_l773_773535


namespace cone_volume_eq_l773_773255

-- Define the given conditions
def V_cylinder := 72 * Real.pi
def height_cylinder (r : ℝ) := 2 * r

-- Define the volume formula for a cone and a cylinder
def volume_cylinder (r h : ℝ) := Real.pi * r^2 * h
def volume_cone (r h : ℝ) := (1 / 3) * Real.pi * r^2 * h

-- Prove that the volume of the cone is 144π cubic cm
theorem cone_volume_eq (r h : ℝ) 
  (H_cylinder : volume_cylinder r h = V_cylinder)
  (H_height : h = height_cylinder r) :
  volume_cone r h = 144 * Real.pi := 
by 
  sorry -- Proof is omitted, placeholder for the actual proof.

end cone_volume_eq_l773_773255


namespace cost_of_computer_game_is_90_l773_773931

-- Define the costs of individual items
def polo_shirt_price : ℕ := 26
def necklace_price : ℕ := 83
def rebate : ℕ := 12
def total_cost_after_rebate : ℕ := 322

-- Define the number of items
def polo_shirt_quantity : ℕ := 3
def necklace_quantity : ℕ := 2
def computer_game_quantity : ℕ := 1

-- Calculate the total cost before rebate
def total_cost_before_rebate : ℕ :=
  total_cost_after_rebate + rebate

-- Calculate the total cost of polo shirts and necklaces
def total_cost_polo_necklaces : ℕ :=
  (polo_shirt_quantity * polo_shirt_price) + (necklace_quantity * necklace_price)

-- Define the unknown cost of the computer game
def computer_game_price : ℕ :=
  total_cost_before_rebate - total_cost_polo_necklaces

-- Prove the cost of the computer game
theorem cost_of_computer_game_is_90 : computer_game_price = 90 := by
  -- The following line is a placeholder for the actual proof
  sorry

end cost_of_computer_game_is_90_l773_773931


namespace sum_binomials_eq_two_pow_weighted_sum_binomials_eq_squared_weighted_sum_binomials_eq_l773_773311

open Nat

/-
Prove: C(n, 0) + C(n, 1) + ... + C(n, n) = 2^n using the Binomial Theorem.
-/
theorem sum_binomials_eq_two_pow (n : ℕ) : 
  ∑ k in finset.range (n + 1), nat.choose n k = 2^n :=
by
  sorry

/-
Prove: 1 * C(n, 1) + 2 * C(n, 2) + ... + n * C(n, n) = n * 2^(n-1) using the Binomial Theorem.
-/
theorem weighted_sum_binomials_eq (n : ℕ) : 
  ∑ k in finset.range (n + 1), k * (nat.choose n k) = n * 2^(n - 1) :=
by
  sorry

/-
Prove: 1^2 * C(n, 1) + 2^2 * C(n, 2) + ... + n^2 * C(n, n) = (n^2 + n) 2^(n-2) using the Binomial Theorem.
-/
theorem squared_weighted_sum_binomials_eq (n : ℕ) : 
  ∑ k in finset.range (n + 1), k^2 * (nat.choose n k) = (n^2 + n) * 2^(n - 2) :=
by
  sorry

end sum_binomials_eq_two_pow_weighted_sum_binomials_eq_squared_weighted_sum_binomials_eq_l773_773311


namespace smallest_n_to_sum_one_with_doubly_special_l773_773372

def is_doubly_special (x : ℝ) : Prop :=
  ∃ (y : ℕ → ℕ), (∀ i, y i = 0 ∨ y i = 5) ∧ ((∑ i, y i * 10^(-i) : ℝ) = x)
  
theorem smallest_n_to_sum_one_with_doubly_special : ∃ n, (∀ (a : ℕ → ℝ), (∀ i < n, is_doubly_special (a i)) → (∑ i in finset.range n, a i = 1)) ∧ ¬(∃ k, k < n ∧ (∀ (b : ℕ → ℝ), (∀ i < k, is_doubly_special (b i)) → (∑ i in finset.range k, b i = 1))) ∧ n = 2 :=
by sorry

end smallest_n_to_sum_one_with_doubly_special_l773_773372


namespace sum_of_midpoint_coordinates_l773_773608

theorem sum_of_midpoint_coordinates : 
  let (x1, y1) := (4, 7)
  let (x2, y2) := (10, 19)
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2)
  let sum_of_coordinates := midpoint.fst + midpoint.snd
  sum_of_coordinates = 20 := sorry

end sum_of_midpoint_coordinates_l773_773608


namespace reciprocal_of_2023_l773_773989

theorem reciprocal_of_2023 : (2023 : ℝ)⁻¹ = 1 / 2023 :=
by
  sorry

end reciprocal_of_2023_l773_773989


namespace negation_of_universal_l773_773972

theorem negation_of_universal {x : ℝ} : ¬ (∀ x > 0, x^2 - x ≤ 0) ↔ ∃ x > 0, x^2 - x > 0 :=
by
  sorry

end negation_of_universal_l773_773972


namespace one_even_one_odd_polynomial_l773_773558

open Polynomial

noncomputable def P (x : ℤ) := a_n * x^n + a_(n-1) * x^(n-1) + ... + a_1 * x + a_0
noncomputable def Q (x : ℤ) := b_m * x^m + b_(m-1) * x^(m-1) + ... + b_1 * x + b_0

theorem one_even_one_odd_polynomial (P Q : Polynomial ℤ)
  (h1 : ∀ k, coeff (P * Q) k % 2 = 0)
  (h2 : ∃ k, (coeff (P * Q) k % 4) ≠ 0) : 
  (∀ i, coeff P i % 2 = 0) ∨ (∀ i, coeff Q i % 2 = 0) ∧ 
  (∃ j, (coeff P j % 2 = 1) ∨ (coeff Q j % 2 = 1)) :=
sorry

end one_even_one_odd_polynomial_l773_773558


namespace part1_part2_part3_l773_773776

open Nat

def S (n : ℕ) : ℕ := (1 / 2 : ℚ) * n ^ 2 + (9 / 2 : ℚ) * n

def a (n : ℕ) : ℕ := if n = 1 then 5 else n + 4

def c (n : ℕ) : ℚ := 1 / ((2 * a n - 9) * (2 * a n - 7))
def T (n : ℕ) : ℚ := (1 / 2 : ℚ) * (1 - 1 / (2 * n + 1))

def f (n : ℕ) : ℕ :=
  if ∃ k : ℕ, n = 2 * k - 1 then a n else 3 * a n - 13

theorem part1 (n : ℕ) (h : 0 < n) : a n = n + 4 := sorry

theorem part2 (n : ℕ) (h : 0 < n) : ∃ k : ℕ, T n > k / 2017 ↔ k = 672 := sorry

theorem part3 (m : ℕ) (h : 0 < m) : ¬ (f (m + 15) = 5 * f m) := sorry

end part1_part2_part3_l773_773776


namespace tangent_length_range_l773_773882

noncomputable def circle_center := (1, 1 : ℝ)
noncomputable def circle_radius : ℝ := 1
noncomputable def line_eq (P : ℝ × ℝ) : Prop := 3 * P.1 + 4 * P.2 + 3 = 0
noncomputable def circle_eq (P : ℝ × ℝ) : Prop := (P.1 - 1) ^ 2 + (P.2 - 1) ^ 2 = 1

theorem tangent_length_range :
  ∀ P : ℝ × ℝ, line_eq P →
  ∃ A B : ℝ × ℝ, circle_eq A ∧ circle_eq B ∧
  tangent_from_point_to_circle P A ∧ tangent_from_point_to_circle P B ∧
  √3 ≤ dist A B ∧ dist A B < 2 :=
by sorry

end tangent_length_range_l773_773882


namespace functional_equation_solution_l773_773730

noncomputable def f (x : ℚ) := x⁻¹

theorem functional_equation_solution (f : ℚ⁺ → ℚ⁺) 
  (h : ∀ x y : ℚ⁺, f (f x ^ 2 * y) = x ^ 3 * f (x * y)) : 
  ∀ x : ℚ⁺, f x = x⁻¹ :=
begin
  sorry
end

end functional_equation_solution_l773_773730


namespace values_of_f_range_of_x_l773_773172

noncomputable
def f : ℝ → ℝ := sorry -- Assume the existence of such a function.

axiom f_defined_for_pos (x : ℝ) : x > 0 → ∃ y : ℝ, y = f(x)
axiom f_of_2 : f 2 = 1
axiom f_multiplicative (x y : ℝ) : x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_increasing (x y : ℝ) : x > 0 → y > 0 → (f x > f y ↔ x > y)

theorem values_of_f : f 1 = 0 ∧ f 4 = 2 := by
  sorry

theorem range_of_x (x : ℝ) : 3 < x ∧ x ≤ 4 ↔ f x + f (x - 3) ≤ 2 := by
  sorry

end values_of_f_range_of_x_l773_773172


namespace work_equivalence_l773_773100

variable (m d r : ℕ)

theorem work_equivalence (h : d > 0) : (m * d) / (m + r^2) = d := sorry

end work_equivalence_l773_773100


namespace volume_of_reservoir_proof_relationship_Q_t_proof_min_hourly_drainage_proof_min_time_to_drain_proof_l773_773343

noncomputable def volume_of_reservoir (drain_rate : ℝ) (time_to_drain : ℝ) : ℝ :=
  drain_rate * time_to_drain

theorem volume_of_reservoir_proof :
  volume_of_reservoir 8 6 = 48 :=
by
  sorry

noncomputable def relationship_Q_t (volume : ℝ) (t : ℝ) : ℝ :=
  volume / t

theorem relationship_Q_t_proof :
  ∀ (t : ℝ), relationship_Q_t 48 t = 48 / t :=
by
  intro t
  sorry

noncomputable def min_hourly_drainage (volume : ℝ) (time : ℝ) : ℝ :=
  volume / time

theorem min_hourly_drainage_proof :
  min_hourly_drainage 48 5 = 9.6 :=
by
  sorry

theorem min_time_to_drain_proof :
  ∀ (max_capacity : ℝ), relationship_Q_t 48 max_capacity = 12 → 48 / 12 = 4 :=
by
  intro max_capacity h
  sorry

end volume_of_reservoir_proof_relationship_Q_t_proof_min_hourly_drainage_proof_min_time_to_drain_proof_l773_773343


namespace rectangle_side_excess_l773_773127

theorem rectangle_side_excess
  (L W : ℝ)  -- length and width of the rectangle
  (x : ℝ)   -- percentage in excess for the first side
  (h1 : 0.95 * (L * (1 + x / 100) * W) = 1.102 * (L * W)) :
  x = 16 :=
by
  sorry

end rectangle_side_excess_l773_773127


namespace question_solution_l773_773008

-- Definitions and conditions corresponding to the problem
def seats := ℕ -- representing the seats as natural numbers
def total_seats := 5
def opposite (a b : seats) := (b = (a + total_seats / 2) % total_seats)
def two_seats_away (a b : seats) := (b = (a + 2) % total_seats) ∨ (b = (a + total_seats - 2) % total_seats)
def next_to (a b : seats) := (b = (a + 1) % total_seats) ∨ (b = (a + total_seats - 1) % total_seats)

theorem question_solution (Chloe Emma David Alan Bella : seats)
  (h1 : opposite Chloe Emma)
  (h2 : two_seats_away Alan David)
  (h3 : ¬ next_to Alan Emma)
  (h4 : next_to Bella Emma) :
  (pred Alan total_seats) = Chloe :=
sorry

end question_solution_l773_773008


namespace value_of_k_l773_773038

noncomputable def problem : Prop :=
  ∀ (a x : ℝ), (a < 0) ∧ (0 < x) → (x^2 + (3 - a) * x + 3 - 2 * a^2 < 3 * exp(x))

theorem value_of_k : problem :=
  begin
    sorry
  end

end value_of_k_l773_773038


namespace max_f_on_interval_l773_773754

noncomputable def f (x : ℝ) : ℝ := -x + 1 / x

theorem max_f_on_interval : 
  ∃ x ∈ Icc (-2 : ℝ) (-1 / 3), 
    (∀ y ∈ Icc (-2 : ℝ) (-1 / 3), f y ≤ f x) ∧ f x = 3 / 2 :=
by
  use -2
  split
  · exact set.left_mem_Icc.2 (by linarith)
  split
  · intro y hy
    have h_der : ∀ x ∈ Ioo (-2 : ℝ) (-1 / 3), deriv f x < 0 :=
      λ x hx, by
        dsimp only [f]
        have h_dx : deriv (-x + x⁻¹) x = -1 - x⁻² := deriv_add (deriv_const (-1)) (deriv_inv' x)
        rw h_dx
        exact add_neg_of_neg_of_neg (by linarith) (inv_sq<0)
    apply continuous_on_Icc_bounds_of_deriv_neg
    · exact continuous_on_of_real ((continuous_real_of_real : continuous f).continuous_on)
    · exact h_der
  · sorry

end max_f_on_interval_l773_773754


namespace triangle_angle_bisectors_l773_773227

theorem triangle_angle_bisectors (A B C L M: Point) 
  (R: ℝ) (circumcircle: Circle) 
  (h1: internal_angle_bisector C A B L) 
  (h2: external_angle_bisector C A B M) 
  (h3: CL = CM) 
  (h4: circumference triangle ABC circumcircle R): 
  AC^2 + BC^2 = 4 * R^2 :=
sorry

end triangle_angle_bisectors_l773_773227


namespace dividend_approx_l773_773573

variable (Q R : Nat) (D : Real)

def dividend :=
  D * Q + R

theorem dividend_approx (Q : Nat) (R : Nat) (D : Real)
  (hQ : Q = 89) (hR : R = 14) (hD : D = 154.75280898876406) :
  dividend D Q R ≈ 13787 := by
  simp [dividend, hQ, hR, hD]
  sorry

end dividend_approx_l773_773573


namespace formula_for_a_min_m_sum_formula_l773_773052

-- Define the sequence (a_n)
def f (x : ℚ) : ℚ := (2 * x) / (3 * x + 2)

-- The sequence a_n where a_1 = 1 and a_(n+1) = f(a_n)
def a : ℕ → ℚ
| 0       := 1
| (n + 1) := f (a n)

-- Prove the general formula for the sequence a_n
theorem formula_for_a (n : ℕ) : a n = 2 / (3 * n - 1) := sorry

-- Define b_n = a_n * a_(n+1)
def b (n : ℕ) : ℚ := (a n) * (a (n + 1))

-- Define S_n as the sum of the first n terms of b_n
def S (n : ℕ) : ℚ := (Finset.range n).sum b

-- Prove that for all positive integers n, S_n < (m - 2016) / 2 implies m ≥ 2019
theorem min_m {n : ℕ} (hn : 0 < n) (hS : S n < (m - 2016) / 2) : 2019 ≤ m := sorry

-- Define b_n = (1 / a_n) * 2^n
def b' (n : ℕ) : ℚ := (1 / (a n)) * 2^n

-- Define S'_n as the sum of the first n terms of b'_n
def S' (n : ℕ) : ℚ := (Finset.range n).sum b'

-- Prove the formula for S'_n
theorem sum_formula (n : ℕ) : S' n = (3 * n - 4) * 2^n + 4 := sorry

end formula_for_a_min_m_sum_formula_l773_773052


namespace opposite_direction_vector_x_value_l773_773837

theorem opposite_direction_vector_x_value (x : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (x, 1)) 
  (hb : b = (4, x)) 
  (opposite : ∃ λ : ℝ, λ < 0 ∧ a = λ • b) : 
  x = -2 :=
sorry

end opposite_direction_vector_x_value_l773_773837


namespace sum_of_radii_is_one_l773_773682

noncomputable def r_1 : ℝ := 1 / 6
noncomputable def r_2 : ℝ := 1 / 3
noncomputable def r_3 : ℝ := 1 / 2

theorem sum_of_radii_is_one : r_1 + r_2 + r_3 = 1 := 
by
  have h1 : r_1 = 1 / 6 := rfl
  have h2 : r_2 = 1 / 3 := rfl
  have h3 : r_3 = 1 / 2 := rfl
  calc 
    r_1 + r_2 + r_3
      = 1 / 6 + 1 / 3 + 1 / 2 : by rw [h1, h2, h3]
  ... = 1 : by norm_num

end sum_of_radii_is_one_l773_773682


namespace inverse_function_f_5_l773_773802

theorem inverse_function_f_5 : (∃ (f : ℝ → ℝ) (f_inv : ℝ → ℝ), (∀ x, f (f_inv x) = x) ∧ (∀ y, f_inv (f y) = y) ∧ (f = fun x => 2 * x - 1)) → ∃ (a : ℝ), a = 3 ∧ (∃ f_inv : ℝ → ℝ, f_inv 5 = a) :=
by
  intro h,
  sorry

end inverse_function_f_5_l773_773802


namespace fibonacci_expression_equality_l773_773963

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

-- Statement to be proven
theorem fibonacci_expression_equality :
  (fibonacci 0 * fibonacci 2 + fibonacci 1 * fibonacci 3 + fibonacci 2 * fibonacci 4 +
  fibonacci 3 * fibonacci 5 + fibonacci 4 * fibonacci 6 + fibonacci 5 * fibonacci 7)
  - (fibonacci 1 ^ 2 + fibonacci 2 ^ 2 + fibonacci 3 ^ 2 + fibonacci 4 ^ 2 + fibonacci 5 ^ 2 + fibonacci 6 ^ 2)
  = 0 :=
by
  sorry

end fibonacci_expression_equality_l773_773963


namespace count_five_digit_integers_with_sum_of_digits_two_l773_773252

/-- 
  Problem statement:
  Prove that there are exactly 5 five-digit positive integers whose digits sum up to 2,
  and do not start with zero.
-/

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := List.ofFn (fun i => Nat.digit n i) in
  List.sum digits

theorem count_five_digit_integers_with_sum_of_digits_two :
  ∃ (S : Finset ℕ), S.card = 5 ∧ ∀ n ∈ S, n >= 10000 ∧ n < 100000 ∧ sum_of_digits n = 2 :=
sorry

end count_five_digit_integers_with_sum_of_digits_two_l773_773252


namespace sum_of_three_digit_numbers_not_in_simplest_form_l773_773651

open Nat

theorem sum_of_three_digit_numbers_not_in_simplest_form :
  ∑ n in (Finset.filter (λ n, gcd (3 * n + 2) (5 * n + 1) = 7) (Finset.range' 100 900)), n = 70950 := 
sorry

end sum_of_three_digit_numbers_not_in_simplest_form_l773_773651


namespace vector_add_eq_l773_773888

open Real EuclideanGeometry

variables {A B C : Point} (E F P : Point)
variables (k1 k2 k3 k4 : ℝ)
variables (x y z : ℝ)
axiom hp : k1 + k2 + k3 = 1
axiom h1 : k1 = (12 : ℝ) / 19
axiom h2 : k2 = (3 : ℝ) / 19
axiom h3 : k3 = (4 : ℝ) / 19
axiom h4 : E = k1 • A + k3 • C
axiom h5 : F = k2 • A + k4 • B
axiom h6 : P = (1 : ℝ) / 4 * (k1 • E + k2 • B) + (3 : ℝ) / 4 * (k3 • F + k4 • C)

theorem vector_add_eq (A B C E F P : Point) (hx : x = 12 / 19) (hy : y = 3 / 19) (hz : z = 4 / 19) :
  P = x • A + y • B + z • C :=
by
  sorry

end vector_add_eq_l773_773888


namespace sum_of_angles_pentagon_triangle_l773_773379

theorem sum_of_angles_pentagon_triangle 
  (P : Type) [Polygon P]
  (T : Type) [Triangle T]
  (angles_P : Finset ℝ) (angles_T : Finset ℝ)
  (hP : ∑ x in angles_P, x = (5 - 2) * 180)
  (hT : ∑ y in angles_T, y = 180) :
  ∑ x in angles_P, x + ∑ y in angles_T, y = 720 := 
 by
    sorry

end sum_of_angles_pentagon_triangle_l773_773379


namespace num_digits_concatenated_l773_773481

theorem num_digits_concatenated (m n : ℕ) : 
  (number_of_digits (5^1971 * 2^1971) = 1972) :=
begin
 sorry
end

end num_digits_concatenated_l773_773481


namespace domain_of_logarithmic_function_l773_773231

theorem domain_of_logarithmic_function :
  ∀ x : ℝ, (3^x - 2^x > 0 ↔ x > 0) :=
by
sorry

end domain_of_logarithmic_function_l773_773231


namespace largest_unrepresentable_integer_l773_773170

theorem largest_unrepresentable_integer 
  (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (gcd_ab : Nat.gcd a b = 1) (gcd_bc : Nat.gcd b c = 1) (gcd_ca : Nat.gcd c a = 1) : 
  ¬ ∃ (x y z : ℕ), (bcx + cay + abz = 2abc - ab - bc - ca) where
  bcx := b * c * x
  cay := c * a * y
  abz := a * b * z
  2abc := 2 * a * b * c
  ab := a * b
  bc := b * c
  ca := c * a :=
sorry

end largest_unrepresentable_integer_l773_773170


namespace domain_of_f_l773_773232

noncomputable def domain_f : Set ℝ := {x : ℝ | 3^x - 2^x > 0}

theorem domain_of_f :
  domain_f = {x : ℝ | 0 < x} :=
sorry

end domain_of_f_l773_773232


namespace highest_power_of_7_dividing_64_factorial_l773_773663

theorem highest_power_of_7_dividing_64_factorial : ∃ k : ℕ, (7 ^ k ∣ nat.factorial 64) ∧ (∀ m : ℕ, 7 ^ (m + 1) ∣ nat.factorial 64 → m < k) ∧ k = 10 :=
by
  sorry

end highest_power_of_7_dividing_64_factorial_l773_773663


namespace yuna_has_most_apples_l773_773543

def apples_count_jungkook : ℕ :=
  6 / 3

def apples_count_yoongi : ℕ :=
  4

def apples_count_yuna : ℕ :=
  5

theorem yuna_has_most_apples : apples_count_yuna > apples_count_yoongi ∧ apples_count_yuna > apples_count_jungkook :=
by
  sorry

end yuna_has_most_apples_l773_773543


namespace inequality_always_holds_l773_773009

theorem inequality_always_holds (a : ℝ) (h : a ≥ -2) : ∀ (x : ℝ), x^2 + a * |x| + 1 ≥ 0 :=
by
  sorry

end inequality_always_holds_l773_773009


namespace butter_mixture_price_l773_773200

theorem butter_mixture_price :
  let cost1 := 48 * 150
  let cost2 := 36 * 125
  let cost3 := 24 * 100
  let revenue1 := cost1 + cost1 * (20 / 100)
  let revenue2 := cost2 + cost2 * (30 / 100)
  let revenue3 := cost3 + cost3 * (50 / 100)
  let total_weight := 48 + 36 + 24
  (revenue1 + revenue2 + revenue3) / total_weight = 167.5 :=
by
  sorry

end butter_mixture_price_l773_773200


namespace bond_paper_cost_l773_773360

/-!
# Bond Paper Cost Calculation

This theorem calculates the total cost to buy the required amount of each type of bond paper, given the specified conditions.
-/

def cost_of_ream (sheets_per_ream : ℤ) (cost_per_ream : ℤ) (required_sheets : ℤ) : ℤ :=
  let reams_needed := (required_sheets + sheets_per_ream - 1) / sheets_per_ream
  reams_needed * cost_per_ream

theorem bond_paper_cost :
  let total_sheets := 5000
  let required_A := 2500
  let required_B := 1500
  let remaining_sheets := total_sheets - required_A - required_B
  let cost_A := cost_of_ream 500 27 required_A
  let cost_B := cost_of_ream 400 24 required_B
  let cost_C := cost_of_ream 300 18 remaining_sheets
  cost_A + cost_B + cost_C = 303 := 
by
  sorry

end bond_paper_cost_l773_773360


namespace axis_of_symmetry_of_quadratic_l773_773953

theorem axis_of_symmetry_of_quadratic (m : ℝ) :
  (∀ x : ℝ, -x^2 + 2 * m * x - m^2 + 3 = -x^2 + 2 * m * x - m^2 + 3) ∧ (∃ x : ℝ, x + 2 = 0) → m = -2 :=
by
  sorry

end axis_of_symmetry_of_quadratic_l773_773953


namespace sum_of_digits_l773_773876

theorem sum_of_digits (x : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (h : 10 * x + 6 * x = 16) : x + 6 * x = 7 :=
by
  -- The proof is omitted
  sorry

end sum_of_digits_l773_773876


namespace conclusion_1_conclusion_2_conclusion_3_conclusion_4_l773_773449

variable (a : ℕ → ℝ)

-- Conditions
def sequence_positive : Prop :=
  ∀ n, a n > 0

def recurrence_relation : Prop :=
  ∀ n, a (n + 1) ^ 2 - a (n + 1) = a n

-- Correct conclusions to prove:

-- Conclusion ①
theorem conclusion_1 (h1 : sequence_positive a) (h2 : recurrence_relation a) :
  ∀ n ≥ 2, a n > 1 := 
sorry

-- Conclusion ②
theorem conclusion_2 (h1 : sequence_positive a) (h2 : recurrence_relation a) :
  ¬∀ n, a n = a (n + 1) := 
sorry

-- Conclusion ③
theorem conclusion_3 (h1 : sequence_positive a) (h2 : recurrence_relation a) (h3 : 0 < a 1 ∧ a 1 < 2) :
  ∀ n, a (n + 1) > a n :=
sorry

-- Conclusion ④
theorem conclusion_4 (h1 : sequence_positive a) (h2 : recurrence_relation a) (h4 : a 1 > 2) :
  ∀ n ≥ 2, 2 < a n ∧ a n < a 1 :=
sorry

end conclusion_1_conclusion_2_conclusion_3_conclusion_4_l773_773449


namespace array_is_natural_sequence_l773_773409

-- Define the sequence and the conditions
def isSequence (a : List ℕ) := a.length = n ∧ (∀ i j, i < j → a[i] < a[j])

def M (a : List ℕ) (d : ℕ) : ℚ :=
  (List.foldl (λ acc x => acc * (x + d)) 1 a.map (λ x => x + d)) / (List.foldl (λ acc x => acc * x) 1 a)

-- Prove that the correct sequence is (1, 2, ..., n)
theorem array_is_natural_sequence (n : ℕ) (a : List ℕ) (h_seq : isSequence a)
  (h_cond : ∀ d : ℕ, M a d ∈ ℤ) :
  a = List.range (n+1) := 
sorry

end array_is_natural_sequence_l773_773409


namespace one_greater_than_17_over_10_l773_773918

theorem one_greater_than_17_over_10 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a + b + c = a * b * c) : 
  a > 17 / 10 ∨ b > 17 / 10 ∨ c > 17 / 10 :=
by
  sorry

end one_greater_than_17_over_10_l773_773918


namespace x_minus_y_eq_11_l773_773850

theorem x_minus_y_eq_11 (x y : ℝ) (h : |x - 6| + |y + 5| = 0) : x - y = 11 := by
  sorry

end x_minus_y_eq_11_l773_773850


namespace shara_savings_l773_773700

theorem shara_savings 
  (original_price : ℝ)
  (discount1 : ℝ := 0.08)
  (discount2 : ℝ := 0.05)
  (sales_tax : ℝ := 0.06)
  (final_price : ℝ := 184)
  (h : (original_price * (1 - discount1) * (1 - discount2) * (1 + sales_tax)) = final_price) :
  original_price - final_price = 25.78 :=
sorry

end shara_savings_l773_773700


namespace mutually_exclusive_not_complementary_l773_773390

universe u
variables {person : Type u} {card : Type u} [inhabited person] [inhabited card]

noncomputable def distribute_cards (cards : list card) (people : list person) : Prop :=
cards.length = people.length → ∃ (f : person → card), function.bijective f ∧ ∀ p, f p ∈ cards

noncomputable def event_A_gets_clubs (f : person → card) (A : person) (clubs : card) : Prop := f A = clubs
noncomputable def event_B_gets_clubs (f : person → card) (B : person) (clubs : card) : Prop := f B = clubs

theorem mutually_exclusive_not_complementary 
  (hearts spades diamonds clubs : card)
  (A B C D : person)
  (f : person → card) 
  (dist : distribute_cards [hearts, spades, diamonds, clubs] [A, B, C, D]) :
  event_A_gets_clubs f A clubs ∧ event_B_gets_clubs f B clubs ↔ false ∧ 
  ∃ C D, event_B_gets_clubs f B clubs ∧ ¬ event_A_gets_clubs f A clubs := 
by 
  sorry

end mutually_exclusive_not_complementary_l773_773390


namespace solve_sqrt_equation_l773_773406

theorem solve_sqrt_equation (x : ℝ) (hx : x ≥ 2) :
  (\sqrt(x + 5 - 6 * \sqrt(x - 2)) + \sqrt(x + 12 - 8 * \sqrt(x - 2)) = 2) ↔ (x = 11 ∨ x = 27) :=
by sorry

end solve_sqrt_equation_l773_773406


namespace candies_per_child_rounded_l773_773257

/-- There are 15 pieces of candy divided equally among 7 children. The number of candies per child, rounded to the nearest tenth, is 2.1. -/
theorem candies_per_child_rounded :
  let candies := 15
  let children := 7
  Float.round (candies / children * 10) / 10 = 2.1 :=
by
  sorry

end candies_per_child_rounded_l773_773257


namespace find_tan_EAM_l773_773305

-- Define the regular hexagon properties
structure hexagon (α : Type*) :=
  (A B C D E F : α)
  (regular : ∀ (X Y Z : α), (X,Y,Z) ∈ {(A,B,C), (B,C,D), (C,D,E), (D,E,F), (E,F,A), (F,A,B)})
  (mirror_inner_surface : bool)

variables {α : Type*} [field α]

-- Define points' coordinates in the given coordinate system for simplicity
noncomputable def A : (α × α) := (1/2 : α, (sqrt 3 / 2 : α))
noncomputable def B : (α × α) := (1 : α, 0)
noncomputable def C : (α × α) := (1/2 : α, -(sqrt 3 / 2 : α))
noncomputable def D : (α × α) := (-1 : α, 0)
noncomputable def E : (α × α) := (-1/2 : α, (sqrt 3 / 2 : α))
noncomputable def F : (α × α) := (-1/2 : α, -(sqrt 3 / 2 : α))

-- Define the tangent function
noncomputable def tangent (θ : α) := real.tan θ

-- Define the problem in the lean statement
theorem find_tan_EAM 
  (M N : α × α)
  (hex : hexagon α)
  (on_sides : M.1 ∈ [A.1, B.1] ∧ N.1 ∈ [B.1, C.1])
  (light_path : N.1 = D.1)
  : tangent (angle A E M) = 1 / 3 / sqrt 3 := 
sorry

end find_tan_EAM_l773_773305
