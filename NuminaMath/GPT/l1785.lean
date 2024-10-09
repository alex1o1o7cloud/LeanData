import Mathlib

namespace log_expression_equals_l1785_178532

noncomputable def expression (x y : ℝ) : ℝ :=
  (Real.log x^2) / (Real.log y^10) *
  (Real.log y^3) / (Real.log x^7) *
  (Real.log x^4) / (Real.log y^8) *
  (Real.log y^6) / (Real.log x^9) *
  (Real.log x^11) / (Real.log y^5)

theorem log_expression_equals (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  expression x y = (1 / 15) * Real.log y / Real.log x :=
sorry

end log_expression_equals_l1785_178532


namespace angle_is_120_degrees_l1785_178530

-- Define the magnitudes of vectors a and b and their dot product
def magnitude_a : ℝ := 10
def magnitude_b : ℝ := 12
def dot_product_ab : ℝ := -60

-- Define the angle between vectors a and b
def angle_between_vectors (θ : ℝ) : Prop :=
  magnitude_a * magnitude_b * Real.cos θ = dot_product_ab

-- Prove that the angle θ is 120 degrees
theorem angle_is_120_degrees : angle_between_vectors (2 * Real.pi / 3) :=
by 
  unfold angle_between_vectors
  sorry

end angle_is_120_degrees_l1785_178530


namespace trigonometric_identity_l1785_178559

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - 3 * Real.cos α) = 3 := 
by 
  sorry

end trigonometric_identity_l1785_178559


namespace dog_older_than_max_by_18_l1785_178554

-- Definition of the conditions
def human_to_dog_years_ratio : ℕ := 7
def max_age : ℕ := 3
def dog_age_in_human_years : ℕ := 3

-- Translate the question: How much older, in dog years, will Max's dog be?
def age_difference_in_dog_years : ℕ :=
  dog_age_in_human_years * human_to_dog_years_ratio - max_age

-- The proof statement
theorem dog_older_than_max_by_18 : age_difference_in_dog_years = 18 := by
  sorry

end dog_older_than_max_by_18_l1785_178554


namespace find_interest_rate_l1785_178589

-- Define the given conditions
def initial_investment : ℝ := 2200
def additional_investment : ℝ := 1099.9999999999998
def total_investment : ℝ := initial_investment + additional_investment
def desired_income : ℝ := 0.06 * total_investment
def income_from_additional_investment : ℝ := 0.08 * additional_investment
def income_from_initial_investment (r : ℝ) : ℝ := initial_investment * r

-- State the proof problem
theorem find_interest_rate (r : ℝ) 
    (h : desired_income = income_from_additional_investment + income_from_initial_investment r) :
    r = 0.05 :=
sorry

end find_interest_rate_l1785_178589


namespace train_travel_time_l1785_178579

theorem train_travel_time
  (a : ℝ) (s : ℝ) (t : ℝ)
  (ha : a = 3)
  (hs : s = 27)
  (h0 : ∀ t, 0 ≤ t) :
  t = Real.sqrt 18 :=
by
  sorry

end train_travel_time_l1785_178579


namespace directrix_of_parabola_l1785_178570

theorem directrix_of_parabola (a b c : ℝ) (parabola_eqn : ∀ x : ℝ, y = 3 * x^2 - 6 * x + 2)
  (vertex : ∃ h k : ℝ, h = 1 ∧ k = -1)
  : ∃ y : ℝ, y = -13 / 12 := 
sorry

end directrix_of_parabola_l1785_178570


namespace painted_cube_faces_l1785_178511

theorem painted_cube_faces (a : ℕ) (h : 2 < a) :
  ∃ (one_face two_faces three_faces : ℕ),
  (one_face = 6 * (a - 2) ^ 2) ∧
  (two_faces = 12 * (a - 2)) ∧
  (three_faces = 8) := by
  sorry

end painted_cube_faces_l1785_178511


namespace necessary_but_not_sufficient_for_p_l1785_178581

variable {p q r : Prop}

theorem necessary_but_not_sufficient_for_p 
  (h₁ : p → q) (h₂ : ¬ (q → p)) 
  (h₃ : q → r) (h₄ : ¬ (r → q)) 
  : (r → p) ∧ ¬ (p → r) :=
sorry

end necessary_but_not_sufficient_for_p_l1785_178581


namespace triangle_angles_median_bisector_altitude_l1785_178508

theorem triangle_angles_median_bisector_altitude {α β γ : ℝ} 
  (h : α + β + γ = 180) 
  (median_angle_condition : α / 4 + β / 4 + γ / 4 = 45) -- Derived from 90/4 = 22.5
  (median_from_C : 4 * α = γ) -- Given condition that angle is divided into 4 equal parts
  (median_angle_C : γ = 90) -- Derived that angle @ C must be right angle (90°)
  (sum_angles_C : α + β = 90) : 
  α = 22.5 ∧ β = 67.5 ∧ γ = 90 :=
by
  sorry

end triangle_angles_median_bisector_altitude_l1785_178508


namespace quadratic_real_roots_range_l1785_178569

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3 * x + m = 0) → m ≤ 9 / 4 :=
by
  sorry

end quadratic_real_roots_range_l1785_178569


namespace expression_never_equals_33_l1785_178574

theorem expression_never_equals_33 (x y : ℤ) : 
  x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 := 
sorry

end expression_never_equals_33_l1785_178574


namespace actual_time_when_watch_reads_11_pm_is_correct_l1785_178516

-- Define the conditions
def noon := 0 -- Time when Cassandra sets her watch to the correct time
def actual_time_2_pm := 120 -- 2:00 PM in minutes
def watch_time_2_pm := 113.2 -- 1:53 PM and 12 seconds in minutes (113 minutes + 0.2 minutes)

-- Define the goal
def actual_time_watch_reads_11_pm := 731.25 -- 12:22 PM and 15 seconds in minutes from noon

-- Provide the theorem statement without proof
theorem actual_time_when_watch_reads_11_pm_is_correct :
  actual_time_watch_reads_11_pm = 731.25 :=
sorry

end actual_time_when_watch_reads_11_pm_is_correct_l1785_178516


namespace find_number_l1785_178548

variable (N : ℕ)

theorem find_number (h : 6 * ((N / 8) + 8 - 30) = 12) : N = 192 := 
by
  sorry

end find_number_l1785_178548


namespace pyramid_surface_area_l1785_178512

theorem pyramid_surface_area (base_edge volume : ℝ)
  (h_base_edge : base_edge = 1)
  (h_volume : volume = 1) :
  let height := 3
  let slant_height := Real.sqrt (9.25)
  let base_area := base_edge * base_edge
  let lateral_area := 4 * (1 / 2 * base_edge * slant_height)
  let total_surface_area := base_area + lateral_area
  total_surface_area = 7.082 :=
by
  sorry

end pyramid_surface_area_l1785_178512


namespace hare_overtakes_tortoise_l1785_178506

noncomputable def hare_distance (t: ℕ) : ℕ := 
  if t ≤ 5 then 10 * t
  else if t ≤ 20 then 50
  else 50 + 20 * (t - 20)

noncomputable def tortoise_distance (t: ℕ) : ℕ :=
  2 * t

theorem hare_overtakes_tortoise : 
  ∃ t : ℕ, t ≤ 60 ∧ hare_distance t = tortoise_distance t ∧ 60 - t = 22 :=
sorry

end hare_overtakes_tortoise_l1785_178506


namespace compute_expression_l1785_178585

theorem compute_expression (x : ℤ) (h : x = 6) :
  ((x^9 - 24 * x^6 + 144 * x^3 - 512) / (x^3 - 8) = 43264) :=
by
  sorry

end compute_expression_l1785_178585


namespace geometric_sequence_product_l1785_178583

theorem geometric_sequence_product :
  ∀ (a : ℕ → ℝ), (∀ n, a n > 0) →
  (∃ (a_1 a_99 : ℝ), (a_1 + a_99 = 10) ∧ (a_1 * a_99 = 16) ∧ a 1 = a_1 ∧ a 99 = a_99) →
  a 20 * a 50 * a 80 = 64 :=
by
  intro a hpos hex
  sorry

end geometric_sequence_product_l1785_178583


namespace polynomial_value_at_2008_l1785_178551

def f (a₀ a₁ a₂ a₃ a₄ : ℝ) (x : ℝ) : ℝ := a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4

theorem polynomial_value_at_2008 (a₀ a₁ a₂ a₃ a₄ : ℝ) (h₁ : a₄ ≠ 0)
  (h₀₃ : f a₀ a₁ a₂ a₃ a₄ 2003 = 24)
  (h₀₄ : f a₀ a₁ a₂ a₃ a₄ 2004 = -6)
  (h₀₅ : f a₀ a₁ a₂ a₃ a₄ 2005 = 4)
  (h₀₆ : f a₀ a₁ a₂ a₃ a₄ 2006 = -6)
  (h₀₇ : f a₀ a₁ a₂ a₃ a₄ 2007 = 24) :
  f a₀ a₁ a₂ a₃ a₄ 2008 = 274 :=
by sorry

end polynomial_value_at_2008_l1785_178551


namespace percentage_is_50_l1785_178533

theorem percentage_is_50 (P : ℝ) (h1 : P = 0.20 * 15 + 47) : P = 50 := 
by
  -- skip the proof
  sorry

end percentage_is_50_l1785_178533


namespace age_difference_l1785_178562

theorem age_difference (a1 a2 a3 a4 x y : ℕ) 
  (h1 : (a1 + a2 + a3 + a4 + x) / 5 = 28)
  (h2 : ((a1 + 1) + (a2 + 1) + (a3 + 1) + (a4 + 1) + y) / 5 = 30) : 
  y - (x + 1) = 5 := 
by
  sorry

end age_difference_l1785_178562


namespace least_positive_integer_not_representable_as_fraction_l1785_178522

theorem least_positive_integer_not_representable_as_fraction : 
  ¬ ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ (2^a - 2^b) / (2^c - 2^d) = 11 :=
sorry

end least_positive_integer_not_representable_as_fraction_l1785_178522


namespace brownie_leftover_is_zero_l1785_178550

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

end brownie_leftover_is_zero_l1785_178550


namespace knight_reachability_l1785_178501

theorem knight_reachability (p q : ℕ) (hpq_pos : 0 < p ∧ 0 < q) :
  (p + q) % 2 = 1 ∧ Nat.gcd p q = 1 ↔
  ∀ x y x' y', ∃ k h n m, x' = x + k * p + h * q ∧ y' = y + n * p + m * q :=
by
  sorry

end knight_reachability_l1785_178501


namespace converse_inverse_contrapositive_l1785_178534

theorem converse (x y : ℤ) : (x = 3 ∧ y = 2) → (x + y = 5) :=
by sorry

theorem inverse (x y : ℤ) : (x + y ≠ 5) → (x ≠ 3 ∨ y ≠ 2) :=
by sorry

theorem contrapositive (x y : ℤ) : (¬ (x = 3 ∧ y = 2)) → (¬ (x + y = 5)) :=
by sorry

end converse_inverse_contrapositive_l1785_178534


namespace value_of_a_minus_b_l1785_178563

variables (a b : ℝ)

theorem value_of_a_minus_b (h1 : abs a = 3) (h2 : abs b = 5) (h3 : a > b) : a - b = 8 :=
sorry

end value_of_a_minus_b_l1785_178563


namespace find_ratio_squares_l1785_178566

variables (x y z a b c : ℝ)

theorem find_ratio_squares 
  (h1 : x / a + y / b + z / c = 5) 
  (h2 : a / x + b / y + c / z = 0) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 25 := 
sorry

end find_ratio_squares_l1785_178566


namespace converse_of_statement_l1785_178542

theorem converse_of_statement (x y : ℝ) :
  (¬ (x = 0 ∧ y = 0)) → (x^2 + y^2 ≠ 0) :=
by {
  sorry
}

end converse_of_statement_l1785_178542


namespace abc_sum_eq_sqrt34_l1785_178538

noncomputable def abc_sum (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 16)
                          (h2 : ab + bc + ca = 9)
                          (h3 : a^2 + b^2 = 10)
                          (h4 : 0 ≤ a) (h5 : 0 ≤ b) (h6 : 0 ≤ c) : ℝ :=
a + b + c

theorem abc_sum_eq_sqrt34 (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 16)
  (h2 : ab + bc + ca = 9)
  (h3 : a^2 + b^2 = 10)
  (h4 : 0 ≤ a)
  (h5 : 0 ≤ b)
  (h6 : 0 ≤ c) :
  abc_sum a b c h1 h2 h3 h4 h5 h6 = Real.sqrt 34 :=
by
  sorry

end abc_sum_eq_sqrt34_l1785_178538


namespace rhombus_longer_diagonal_l1785_178549

theorem rhombus_longer_diagonal
  (a b : ℝ)
  (ha : a = 65) -- Side length
  (hb : b = 28) -- Half the length of the shorter diagonal
  : 2 * Real.sqrt (a^2 - b^2) = 118 :=  -- Statement of the longer diagonal length
by
  sorry -- Proof goes here

end rhombus_longer_diagonal_l1785_178549


namespace john_income_increase_l1785_178546

noncomputable def net_percentage_increase (initial_income : ℝ) (final_income_before_bonus : ℝ) (monthly_bonus : ℝ) (tax_deduction_rate : ℝ) : ℝ :=
  let weekly_bonus := monthly_bonus / 4
  let final_income_before_taxes := final_income_before_bonus + weekly_bonus
  let tax_deduction := tax_deduction_rate * final_income_before_taxes
  let net_final_income := final_income_before_taxes - tax_deduction
  ((net_final_income - initial_income) / initial_income) * 100

theorem john_income_increase :
  net_percentage_increase 40 60 100 0.10 = 91.25 := by
  sorry

end john_income_increase_l1785_178546


namespace brads_zip_code_l1785_178547

theorem brads_zip_code (A B C D E : ℕ) (h1 : A + B + C + D + E = 20)
                        (h2 : B = A + 1) (h3 : C = A)
                        (h4 : D = 2 * A) (h5 : D + E = 13)
                        (h6 : Nat.Prime (A*10000 + B*1000 + C*100 + D*10 + E)) :
                        A*10000 + B*1000 + C*100 + D*10 + E = 34367 := 
sorry

end brads_zip_code_l1785_178547


namespace points_opposite_sides_of_line_l1785_178580

theorem points_opposite_sides_of_line (a : ℝ) :
  (1 + 1 - a) * (2 - 1 - a) < 0 ↔ 1 < a ∧ a < 2 :=
by sorry

end points_opposite_sides_of_line_l1785_178580


namespace negation_of_existence_l1785_178527

theorem negation_of_existence :
  ¬ (∃ x : ℝ, x^2 > 2) ↔ ∀ x : ℝ, x^2 ≤ 2 :=
by
  sorry

end negation_of_existence_l1785_178527


namespace daily_evaporation_rate_l1785_178578

theorem daily_evaporation_rate (initial_amount : ℝ) (period : ℕ) (percentage_evaporated : ℝ) (h_initial : initial_amount = 10) (h_period : period = 50) (h_percentage : percentage_evaporated = 4) : 
  (percentage_evaporated / 100 * initial_amount) / period = 0.008 :=
by
  -- Ensures that the conditions translate directly into the Lean theorem statement
  rw [h_initial, h_period, h_percentage]
  -- Insert the required logical proof here
  sorry

end daily_evaporation_rate_l1785_178578


namespace points_eq_l1785_178560

-- Definition of the operation 
def star (a b : ℝ) : ℝ := a^2 * b + a * b^2

-- The property we want to prove
theorem points_eq : {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1} =
    {p : ℝ × ℝ | p.1 = 0} ∪ {p : ℝ × ℝ | p.2 = 0} ∪ {p : ℝ × ℝ | p.1 + p.2 = 0} :=
by
  sorry

end points_eq_l1785_178560


namespace profit_percentage_is_correct_l1785_178576

-- Define the conditions
variables (market_price_per_pen : ℝ) (discount_percentage : ℝ) (total_pens_bought : ℝ) (cost_pens_market_price : ℝ)
variables (cost_price_per_pen : ℝ) (selling_price_per_pen : ℝ) (profit_per_pen : ℝ) (profit_percent : ℝ)

-- Conditions
def condition_1 : market_price_per_pen = 1 := by sorry
def condition_2 : discount_percentage = 0.01 := by sorry
def condition_3 : total_pens_bought = 80 := by sorry
def condition_4 : cost_pens_market_price = 36 := by sorry

-- Definitions based on conditions
def cost_price_per_pen_def : cost_price_per_pen = cost_pens_market_price / total_pens_bought := by sorry
def selling_price_per_pen_def : selling_price_per_pen = market_price_per_pen * (1 - discount_percentage) := by sorry
def profit_per_pen_def : profit_per_pen = selling_price_per_pen - cost_price_per_pen := by sorry
def profit_percent_def : profit_percent = (profit_per_pen / cost_price_per_pen) * 100 := by sorry

-- The statement to prove
theorem profit_percentage_is_correct : profit_percent = 120 :=
by
  have h1 : cost_price_per_pen = 36 / 80 := by sorry
  have h2 : selling_price_per_pen = 1 * (1 - 0.01) := by sorry
  have h3 : profit_per_pen = 0.99 - 0.45 := by sorry
  have h4 : profit_percent = (0.54 / 0.45) * 100 := by sorry
  sorry

end profit_percentage_is_correct_l1785_178576


namespace find_k_slope_eq_l1785_178552

theorem find_k_slope_eq :
  ∃ k: ℝ, (∃ k: ℝ, ((k - 4) / 7 = (-2 - k) / 14) → k = 2) :=
by
  sorry

end find_k_slope_eq_l1785_178552


namespace find_jack_euros_l1785_178545

theorem find_jack_euros (E : ℕ) (h1 : 45 + 2 * E = 117) : E = 36 :=
by
  sorry

end find_jack_euros_l1785_178545


namespace problems_per_worksheet_l1785_178598

theorem problems_per_worksheet (total_worksheets : ℕ) (graded_worksheets : ℕ) (remaining_problems : ℕ)
    (h1 : total_worksheets = 16) (h2 : graded_worksheets = 8) (h3 : remaining_problems = 32) :
    remaining_problems / (total_worksheets - graded_worksheets) = 4 :=
by
  sorry

end problems_per_worksheet_l1785_178598


namespace sin6_add_3sin2_cos2_add_cos6_eq_one_iff_eq_l1785_178521

-- Define the real interval [0, π/2]
def interval_0_pi_over_2 (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ Real.pi / 2

-- Define the proposition to be proven
theorem sin6_add_3sin2_cos2_add_cos6_eq_one_iff_eq (a b : ℝ) 
  (ha : interval_0_pi_over_2 a) (hb : interval_0_pi_over_2 b) :
  (Real.sin a)^6 + 3 * (Real.sin a)^2 * (Real.cos b)^2 + (Real.cos b)^6 = 1 ↔ a = b :=
by
  sorry

end sin6_add_3sin2_cos2_add_cos6_eq_one_iff_eq_l1785_178521


namespace line_tangent_to_parabola_l1785_178504

theorem line_tangent_to_parabola (k : ℝ) (x₀ y₀ : ℝ) 
  (h₁ : y₀ = k * x₀ - 2) 
  (h₂ : x₀^2 = 4 * y₀) 
  (h₃ : ∀ x y, (x = x₀ ∧ y = y₀) → (k = (1/2) * x₀)) :
  k = Real.sqrt 2 ∨ k = -Real.sqrt 2 := 
sorry

end line_tangent_to_parabola_l1785_178504


namespace sum_of_p_and_q_l1785_178597

-- Definitions for points and collinearity condition
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := {x := 1, y := 3, z := -2}
def B : Point3D := {x := 2, y := 5, z := 1}
def C (p q : ℝ) : Point3D := {x := p, y := 7, z := q - 2}

def collinear (A B C : Point3D) : Prop :=
  ∃ (k : ℝ), B.x - A.x = k * (C.x - A.x) ∧ B.y - A.y = k * (C.y - A.y) ∧ B.z - A.z = k * (C.z - A.z)

theorem sum_of_p_and_q (p q : ℝ) (h : collinear A B (C p q)) : p + q = 9 := by
  sorry

end sum_of_p_and_q_l1785_178597


namespace sum_ratios_eq_l1785_178509

-- Define points A, B, C, D, E, and G as well as their relationships
variables {A B C D E G : Type}

-- Given conditions
axiom BD_2DC : ∀ {BD DC : ℝ}, BD = 2 * DC
axiom AE_3EB : ∀ {AE EB : ℝ}, AE = 3 * EB
axiom AG_2GD : ∀ {AG GD : ℝ}, AG = 2 * GD

-- Mass assumptions for the given problem
noncomputable def mC := 1
noncomputable def mB := 2
noncomputable def mD := mB + 2 * mC  -- mD = B's mass + 2*C's mass
noncomputable def mA := 1
noncomputable def mE := 3 * mA + mB  -- mE = 3A's mass + B's mass
noncomputable def mG := 2 * mA + mD  -- mG = 2A's mass + D's mass

-- Ratios defined according to the problem statement
noncomputable def ratio1 := (1 : ℝ) / mE
noncomputable def ratio2 := mD / mA
noncomputable def ratio3 := mD / mG

-- The Lean theorem to state the problem and correct answer
theorem sum_ratios_eq : ratio1 + ratio2 + ratio3 = (73 / 15 : ℝ) :=
by
  unfold ratio1 ratio2 ratio3
  sorry

end sum_ratios_eq_l1785_178509


namespace drawing_at_least_one_red_is_certain_l1785_178553

-- Defining the balls and box conditions
structure Box :=
  (red_balls : ℕ) 
  (yellow_balls : ℕ) 

-- Let the box be defined as having 3 red balls and 2 yellow balls
def box : Box := { red_balls := 3, yellow_balls := 2 }

-- Define the event of drawing at least one red ball
def at_least_one_red (draws : ℕ) (b : Box) : Prop :=
  ∀ drawn_yellow, drawn_yellow < draws → drawn_yellow < b.yellow_balls

-- The conclusion we want to prove
theorem drawing_at_least_one_red_is_certain : at_least_one_red 3 box :=
by 
  sorry

end drawing_at_least_one_red_is_certain_l1785_178553


namespace fixed_point_of_f_l1785_178528

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + Real.logb a (|x + 1|)

theorem fixed_point_of_f (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : f a 0 = 1 :=
by
  sorry

end fixed_point_of_f_l1785_178528


namespace assignment_ways_l1785_178513

-- Definitions
def graduates := 5
def companies := 3

-- Statement to be proven
theorem assignment_ways :
  ∃ (ways : ℕ), ways = 150 :=
sorry

end assignment_ways_l1785_178513


namespace sculpture_plus_base_height_l1785_178567

def height_sculpture_feet : Nat := 2
def height_sculpture_inches : Nat := 10
def height_base_inches : Nat := 4

def height_sculpture_total_inches : Nat := height_sculpture_feet * 12 + height_sculpture_inches
def height_total_inches : Nat := height_sculpture_total_inches + height_base_inches

theorem sculpture_plus_base_height :
  height_total_inches = 38 := by
  sorry

end sculpture_plus_base_height_l1785_178567


namespace tan_beta_l1785_178586

open Real

variable (α β : ℝ)

theorem tan_beta (h₁ : tan α = 1/3) (h₂ : tan (α + β) = 1/2) : tan β = 1/7 :=
by sorry

end tan_beta_l1785_178586


namespace molecular_weight_correct_l1785_178587

-- Define the atomic weights of the elements.
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms for each element in the compound.
def number_of_C : ℕ := 7
def number_of_H : ℕ := 6
def number_of_O : ℕ := 2

-- Define the molecular weight calculation.
def molecular_weight : ℝ := 
  (number_of_C * atomic_weight_C) +
  (number_of_H * atomic_weight_H) +
  (number_of_O * atomic_weight_O)

-- Step to prove that molecular weight is equal to 122.118 g/mol.
theorem molecular_weight_correct : molecular_weight = 122.118 := by
  sorry

end molecular_weight_correct_l1785_178587


namespace geometric_sequence_sum_l1785_178518

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : q = 2) (h3 : a 0 + a 1 + a 2 = 21) : 
  a 2 + a 3 + a 4 = 84 :=
sorry

end geometric_sequence_sum_l1785_178518


namespace fill_tank_time_l1785_178565

-- Definitions based on provided conditions
def pipeA_time := 60 -- Pipe A fills the tank in 60 minutes
def pipeB_time := 40 -- Pipe B fills the tank in 40 minutes

-- Theorem statement
theorem fill_tank_time (T : ℕ) : 
  (T / 2) / pipeB_time + (T / 2) * (1 / pipeA_time + 1 / pipeB_time) = 1 → 
  T = 48 :=
by
  intro h
  sorry

end fill_tank_time_l1785_178565


namespace complement_intersection_eq_l1785_178500

def U : Set ℕ := {0, 1, 2, 3}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem complement_intersection_eq :
  (U \ A) ∩ B = {3} :=
by
  sorry

end complement_intersection_eq_l1785_178500


namespace calc_x2015_l1785_178588

noncomputable def f (x a : ℝ) : ℝ := x / (a * (x + 2))

theorem calc_x2015 (a x x_0 : ℝ) (x_seq : ℕ → ℝ)
  (h_unique: ∀ x, f x a = x → x = 0) 
  (h_a_val: a = 1 / 2)
  (h_f_x0: f x_0 a = 1 / 1008)
  (h_seq: ∀ n, x_seq (n + 1) = f (x_seq n) a)
  (h_x0_val: x_seq 0 = x_0):
  x_seq 2015 = 1 / 2015 :=
by
  sorry

end calc_x2015_l1785_178588


namespace solve_k_n_l1785_178524
-- Import the entire Mathlib

-- Define the theorem statement
theorem solve_k_n (k n : ℕ) (hk : k > 0) (hn : n > 0) : k^2 - 2016 = 3^n ↔ k = 45 ∧ n = 2 :=
  by sorry

end solve_k_n_l1785_178524


namespace stone_solution_l1785_178514

noncomputable def stone_problem : Prop :=
  ∃ y : ℕ, (∃ x z : ℕ, x + y + z = 100 ∧ x + 10 * y + 50 * z = 500) ∧
    ∀ y1 y2 : ℕ, (∃ x1 z1 : ℕ, x1 + y1 + z1 = 100 ∧ x1 + 10 * y1 + 50 * z1 = 500) ∧
                (∃ x2 z2 : ℕ, x2 + y2 + z2 = 100 ∧ x2 + 10 * y2 + 50 * z2 = 500) →
                y1 = y2

theorem stone_solution : stone_problem :=
sorry

end stone_solution_l1785_178514


namespace value_of_a_l1785_178535

theorem value_of_a (a : ℕ) (h : a ^ 3 = 21 * 35 * 45 * 35) : a = 105 :=
by
  sorry

end value_of_a_l1785_178535


namespace proof_of_k_values_l1785_178503

noncomputable def problem_statement : Prop :=
  ∀ k : ℝ,
    (∃ a b : ℝ, (6 * a^2 + 5 * a + k = 0 ∧ 6 * b^2 + 5 * b + k = 0 ∧ a ≠ b ∧
    |a - b| = 3 * (a^2 + b^2))) ↔ (k = 1 ∨ k = -20.717)

theorem proof_of_k_values : problem_statement :=
by sorry

end proof_of_k_values_l1785_178503


namespace sum_of_digits_eq_28_l1785_178536

theorem sum_of_digits_eq_28 (A B C D E : ℕ) 
  (hA : 0 ≤ A ∧ A ≤ 9) 
  (hB : 0 ≤ B ∧ B ≤ 9) 
  (hC : 0 ≤ C ∧ C ≤ 9) 
  (hD : 0 ≤ D ∧ D ≤ 9) 
  (hE : 0 ≤ E ∧ E ≤ 9) 
  (unique_digits : (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (C ≠ D) ∧ (C ≠ E) ∧ (D ≠ E)) 
  (h : (10 * A + B) * (10 * C + D) = 111 * E) : 
  A + B + C + D + E = 28 :=
sorry

end sum_of_digits_eq_28_l1785_178536


namespace circle_radius_l1785_178539

/-- Consider a square ABCD with a side length of 4 cm. A circle touches the extensions 
of sides AB and AD. From point C, two tangents are drawn to this circle, 
and the angle between the tangents is 60 degrees. -/
theorem circle_radius (side_length : ℝ) (angle_between_tangents : ℝ) : 
  side_length = 4 ∧ angle_between_tangents = 60 → 
  ∃ (radius : ℝ), radius = 4 * (Real.sqrt 2 + 1) :=
by
  sorry

end circle_radius_l1785_178539


namespace gcd_lcm_of_a_b_l1785_178557

def a := 1560
def b := 1040

theorem gcd_lcm_of_a_b :
  (Nat.gcd a b = 520) ∧ (Nat.lcm a b = 1560) :=
by
  -- Proof is omitted.
  sorry

end gcd_lcm_of_a_b_l1785_178557


namespace find_surcharge_l1785_178596

-- The property tax in 1996 is increased by 6% over the 1995 tax.
def increased_tax (T_1995 : ℝ) : ℝ := T_1995 * 1.06

-- Petersons' property tax for the year 1995 is $1800.
def T_1995 : ℝ := 1800

-- The Petersons' 1996 tax totals $2108.
def T_1996 : ℝ := 2108

-- Additional surcharge for a special project.
def surcharge (T_1996 : ℝ) (increased_tax : ℝ) : ℝ := T_1996 - increased_tax

theorem find_surcharge : surcharge T_1996 (increased_tax T_1995) = 200 := by
  sorry

end find_surcharge_l1785_178596


namespace frogs_per_fish_per_day_l1785_178537

theorem frogs_per_fish_per_day
  (f g n F : ℕ)
  (h1 : f = 30)
  (h2 : g = 15)
  (h3 : n = 9)
  (h4 : F = 32400) :
  F / f / (n * g) = 8 := by
  sorry

end frogs_per_fish_per_day_l1785_178537


namespace Janet_horses_l1785_178505

theorem Janet_horses (acres : ℕ) (gallons_per_acre : ℕ) (spread_acres_per_day : ℕ) (total_days : ℕ)
  (gallons_per_day_per_horse : ℕ) (total_gallons_needed : ℕ) (total_gallons_spread : ℕ) (horses : ℕ) :
  acres = 20 ->
  gallons_per_acre = 400 ->
  spread_acres_per_day = 4 ->
  total_days = 25 ->
  gallons_per_day_per_horse = 5 ->
  total_gallons_needed = acres * gallons_per_acre ->
  total_gallons_spread = spread_acres_per_day * gallons_per_acre * total_days ->
  horses = total_gallons_needed / (gallons_per_day_per_horse * total_days) ->
  horses = 64 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end Janet_horses_l1785_178505


namespace find_number_l1785_178543

theorem find_number (x : ℚ) : (35 / 100) * x = (20 / 100) * 50 → x = 200 / 7 :=
by
  intros h
  sorry

end find_number_l1785_178543


namespace complete_square_eq_l1785_178507

theorem complete_square_eq (x : ℝ) : (x^2 - 6 * x - 5 = 0) -> (x - 3)^2 = 14 :=
by
  intro h
  sorry

end complete_square_eq_l1785_178507


namespace maple_taller_than_birch_l1785_178541

def birch_tree_height : ℚ := 49 / 4
def maple_tree_height : ℚ := 102 / 5

theorem maple_taller_than_birch : maple_tree_height - birch_tree_height = 163 / 20 :=
by
  sorry

end maple_taller_than_birch_l1785_178541


namespace most_stable_performance_l1785_178525

theorem most_stable_performance 
    (S_A S_B S_C S_D : ℝ)
    (h_A : S_A = 0.54) 
    (h_B : S_B = 0.61) 
    (h_C : S_C = 0.7) 
    (h_D : S_D = 0.63) :
    S_A <= S_B ∧ S_A <= S_C ∧ S_A <= S_D :=
by {
  sorry
}

end most_stable_performance_l1785_178525


namespace lattice_points_count_l1785_178561

-- A definition of lattice points and bounded region
def is_lattice_point (p : ℤ × ℤ) : Prop := true

def in_region (p : ℤ × ℤ) : Prop :=
  let (x, y) := p
  (y = abs x ∨ y = -x^2 + 4*x + 6) ∧ (y ≤ abs x ∧ y ≤ -x^2 + 4*x + 6)

-- The target statement to prove
theorem lattice_points_count : ∃ n, n = 23 ∧ ∀ p : ℤ × ℤ, is_lattice_point p → in_region p := sorry

end lattice_points_count_l1785_178561


namespace rotation_image_of_D_l1785_178540

def rotate_90_clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, -p.fst)

theorem rotation_image_of_D :
  rotate_90_clockwise (-3, 2) = (2, 3) :=
by
  sorry

end rotation_image_of_D_l1785_178540


namespace exists_x_odd_n_l1785_178593

theorem exists_x_odd_n (n : ℤ) (h : n % 2 = 1) : 
  ∃ x : ℤ, n^2 ∣ x^2 - n*x - 1 := by
  sorry

end exists_x_odd_n_l1785_178593


namespace smallest_coterminal_angle_pos_radians_l1785_178577

theorem smallest_coterminal_angle_pos_radians :
  ∀ (θ : ℝ), θ = -560 * (π / 180) → ∃ α : ℝ, α > 0 ∧ α = (8 * π) / 9 ∧ (∃ k : ℤ, θ + 2 * k * π = α) :=
by
  sorry

end smallest_coterminal_angle_pos_radians_l1785_178577


namespace quadratic_real_roots_l1785_178594

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end quadratic_real_roots_l1785_178594


namespace total_visitors_400_l1785_178544

variables (V E U : ℕ)

def visitors_did_not_enjoy_understand (V : ℕ) := 3 * V / 4 + 100 = V
def visitors_enjoyed_equal_understood (E U : ℕ) := E = U
def total_visitors_satisfy_34 (V E : ℕ) := 3 * V / 4 = E

theorem total_visitors_400
  (h1 : ∀ V, visitors_did_not_enjoy_understand V)
  (h2 : ∀ E U, visitors_enjoyed_equal_understood E U)
  (h3 : ∀ V E, total_visitors_satisfy_34 V E) :
  V = 400 :=
by { sorry }

end total_visitors_400_l1785_178544


namespace mother_used_eggs_l1785_178556

variable (initial_eggs : ℕ) (eggs_after_chickens : ℕ) (chickens : ℕ) (eggs_per_chicken : ℕ) (current_eggs : ℕ)

theorem mother_used_eggs (h1 : initial_eggs = 10)
                        (h2 : chickens = 2)
                        (h3 : eggs_per_chicken = 3)
                        (h4 : current_eggs = 11)
                        (eggs_laid : ℕ)
                        (h5 : eggs_laid = chickens * eggs_per_chicken)
                        (eggs_used : ℕ)
                        (h6 : eggs_after_chickens = initial_eggs - eggs_used + eggs_laid)
                        : eggs_used = 7 :=
by
  -- proof steps go here
  sorry

end mother_used_eggs_l1785_178556


namespace population_exceeds_l1785_178592

theorem population_exceeds (n : ℕ) : (∃ n, 4 * 3^n > 200) ∧ ∀ m, m < n → 4 * 3^m ≤ 200 := by
  sorry

end population_exceeds_l1785_178592


namespace odd_expression_divisible_by_48_l1785_178510

theorem odd_expression_divisible_by_48 (x : ℤ) (h : Odd x) : 48 ∣ (x^3 + 3*x^2 - x - 3) :=
  sorry

end odd_expression_divisible_by_48_l1785_178510


namespace product_lcm_gcd_eq_product_original_numbers_l1785_178595

theorem product_lcm_gcd_eq_product_original_numbers :
  let a := 12
  let b := 18
  (Int.gcd a b) * (Int.lcm a b) = a * b :=
by
  sorry

end product_lcm_gcd_eq_product_original_numbers_l1785_178595


namespace sequence_geometric_l1785_178590

theorem sequence_geometric (a : ℕ → ℕ) (n : ℕ) (hn : 0 < n):
  (a 1 = 1) →
  (∀ n, 0 < n → a (n + 1) = 2 * a n) →
  a n = 2^(n-1) :=
by
  intros
  sorry

end sequence_geometric_l1785_178590


namespace remainder_of_polynomial_division_l1785_178523

theorem remainder_of_polynomial_division :
  Polynomial.eval 2 (8 * X^3 - 22 * X^2 + 30 * X - 45) = -9 :=
by {
  sorry
}

end remainder_of_polynomial_division_l1785_178523


namespace volume_tetrahedron_ABCD_l1785_178515

noncomputable def volume_of_tetrahedron (AB CD distance angle : ℝ) : ℝ :=
  (1 / 3) * ((1 / 2) * AB * CD * Real.sin angle) * distance

theorem volume_tetrahedron_ABCD :
  volume_of_tetrahedron 1 (Real.sqrt 3) 2 (Real.pi / 3) = 1 / 2 :=
by
  unfold volume_of_tetrahedron
  sorry

end volume_tetrahedron_ABCD_l1785_178515


namespace max_pasture_area_l1785_178573

/-- A rectangular sheep pasture is enclosed on three sides by a fence, while the fourth side uses the 
side of a barn that is 500 feet long. The fence costs $10 per foot, and the total budget for the 
fence is $2000. Determine the length of the side parallel to the barn that will maximize the pasture area. -/
theorem max_pasture_area (length_barn : ℝ) (cost_per_foot : ℝ) (budget : ℝ) :
  length_barn = 500 ∧ cost_per_foot = 10 ∧ budget = 2000 → 
  ∃ x : ℝ, x = 100 ∧ (∀ y : ℝ, y ≥ 0 → 
    (budget / cost_per_foot) ≥ 2*y + x → 
    (y * x ≤ y * 100)) :=
by
  sorry

end max_pasture_area_l1785_178573


namespace determine_s_plus_u_l1785_178591

theorem determine_s_plus_u (p r s u : ℂ) (q t : ℂ) (h₁ : q = 5)
    (h₂ : t = -p - r) (h₃ : p + q * I + r + s * I + t + u * I = 4 * I) : s + u = -1 :=
by
  sorry

end determine_s_plus_u_l1785_178591


namespace sum_of_reciprocals_is_one_l1785_178584

theorem sum_of_reciprocals_is_one (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / (x : ℚ)) + (1 / (y : ℚ)) + (1 / (z : ℚ)) = 1 ↔ (x, y, z) = (2, 4, 4) ∨ 
                                                    (x, y, z) = (2, 3, 6) ∨ 
                                                    (x, y, z) = (3, 3, 3) :=
by 
  sorry

end sum_of_reciprocals_is_one_l1785_178584


namespace one_greater_than_one_l1785_178529

theorem one_greater_than_one (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * b * c = 1)
  (h5 : a + b + c > 1/a + 1/b + 1/c) : a > 1 ∨ b > 1 ∨ c > 1 :=
by
  sorry

end one_greater_than_one_l1785_178529


namespace number_of_Cl_atoms_l1785_178526

def atomic_weight_H : ℝ := 1
def atomic_weight_Cl : ℝ := 35.5
def atomic_weight_O : ℝ := 16

def H_atoms : ℕ := 1
def O_atoms : ℕ := 2
def total_molecular_weight : ℝ := 68

theorem number_of_Cl_atoms :
  (total_molecular_weight - (H_atoms * atomic_weight_H + O_atoms * atomic_weight_O)) / atomic_weight_Cl = 1 :=
by
  -- proof to show this holds
  sorry

end number_of_Cl_atoms_l1785_178526


namespace min_value_of_a_plus_2b_l1785_178558

theorem min_value_of_a_plus_2b (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 2 / a + 1 / b = 1) : a + 2 * b = 4 :=
sorry

end min_value_of_a_plus_2b_l1785_178558


namespace option_A_is_linear_equation_l1785_178599

-- Definitions for considering an equation being linear in two variables
def is_linear_equation (e : Prop) : Prop :=
  ∃ (a b c : ℝ), e = (a = b + c) ∧ a ≠ 0 ∧ b ≠ 0

-- The given equation in option A
def Eq_A : Prop := ∀ (x y : ℝ), (2 * y - 1) / 5 = 2 - (3 * x - 2) / 4

-- Proof problem statement
theorem option_A_is_linear_equation : is_linear_equation Eq_A :=
sorry

end option_A_is_linear_equation_l1785_178599


namespace triangle_inequality_x_values_l1785_178502

theorem triangle_inequality_x_values :
  {x : ℕ | 1 ≤ x ∧ x < 14} = {x | x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9 ∨ x = 10 ∨ x = 11 ∨ x = 12 ∨ x = 13} :=
  by
    sorry

end triangle_inequality_x_values_l1785_178502


namespace train_speed_ratio_l1785_178572

variable (V1 V2 : ℝ)

theorem train_speed_ratio (H1 : V1 * 4 = D1) (H2 : V2 * 36 = D2) (H3 : D1 / D2 = 1 / 9) :
  V1 / V2 = 1 := 
by
  sorry

end train_speed_ratio_l1785_178572


namespace binary_multiplication_correct_l1785_178582

-- Define binary numbers as strings to directly use them in Lean
def binary_num1 : String := "1111"
def binary_num2 : String := "111"

-- Define a function to convert binary strings to natural numbers
def binary_to_nat (s : String) : Nat :=
  s.foldl (fun acc c => acc * 2 + (if c = '1' then 1 else 0)) 0

-- Define the target multiplication result
def binary_product_correct : Nat :=
  binary_to_nat "1001111"

theorem binary_multiplication_correct :
  binary_to_nat binary_num1 * binary_to_nat binary_num2 = binary_product_correct :=
by
  sorry

end binary_multiplication_correct_l1785_178582


namespace inequality_proof_l1785_178571

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_ineq : a + b + c > 1 / a + 1 / b + 1 / c) : 
  a + b + c ≥ 3 / (a * b * c) :=
sorry

end inequality_proof_l1785_178571


namespace construct_one_degree_l1785_178531

theorem construct_one_degree (theta : ℝ) (h : theta = 19) : 1 = 19 * theta - 360 :=
by
  -- Proof here will be filled
  sorry

end construct_one_degree_l1785_178531


namespace sqrt_expr_is_599_l1785_178564

theorem sqrt_expr_is_599 : Real.sqrt ((26 * 25 * 24 * 23) + 1) = 599 := by
  sorry

end sqrt_expr_is_599_l1785_178564


namespace remainder_when_n_plus_5040_divided_by_7_l1785_178520

theorem remainder_when_n_plus_5040_divided_by_7 (n : ℤ) (h: n % 7 = 2) : (n + 5040) % 7 = 2 :=
by
  sorry

end remainder_when_n_plus_5040_divided_by_7_l1785_178520


namespace cos_minus_sin_l1785_178568

theorem cos_minus_sin (α : ℝ) (h1 : Real.sin (2 * α) = 1 / 4) (h2 : Real.pi / 4 < α ∧ α < Real.pi / 2) : 
  Real.cos α - Real.sin α = - (Real.sqrt 3) / 2 :=
sorry

end cos_minus_sin_l1785_178568


namespace cannot_tile_regular_pentagon_l1785_178517

theorem cannot_tile_regular_pentagon :
  ¬ (∃ n : ℕ, 360 % (180 - (360 / 5 : ℕ)) = 0) :=
by sorry

end cannot_tile_regular_pentagon_l1785_178517


namespace Amanda_lost_notebooks_l1785_178519

theorem Amanda_lost_notebooks (initial_notebooks ordered additional_notebooks remaining_notebooks : ℕ)
  (h1 : initial_notebooks = 10)
  (h2 : ordered = 6)
  (h3 : remaining_notebooks = 14) :
  initial_notebooks + ordered - remaining_notebooks = 2 := by
sorry

end Amanda_lost_notebooks_l1785_178519


namespace daily_evaporation_l1785_178575

variable (initial_water : ℝ) (percentage_evaporated : ℝ) (days : ℕ)
variable (evaporation_amount : ℝ)

-- Given conditions
def conditions_met : Prop :=
  initial_water = 10 ∧ percentage_evaporated = 0.4 ∧ days = 50

-- Question: Prove the amount of water evaporated each day is 0.08
theorem daily_evaporation (h : conditions_met initial_water percentage_evaporated days) :
  evaporation_amount = (initial_water * percentage_evaporated) / days :=
sorry

end daily_evaporation_l1785_178575


namespace number_of_team_members_l1785_178555

-- Let's define the conditions.
def packs : ℕ := 3
def pouches_per_pack : ℕ := 6
def total_pouches : ℕ := packs * pouches_per_pack
def coaches : ℕ := 3
def helpers : ℕ := 2
def total_people (members : ℕ) : ℕ := members + coaches + helpers

-- Prove the number of members on the baseball team.
theorem number_of_team_members (members : ℕ) (h : total_people members = total_pouches) : members = 13 :=
by
  sorry

end number_of_team_members_l1785_178555
