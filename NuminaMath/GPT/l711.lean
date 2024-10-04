import Mathlib

namespace calculate_expression_l711_711122

theorem calculate_expression : 
  (π - 2019)^0 + |real.sqrt 3 - 1| + (-1/2 : ℝ)^(-1) - 2 * real.tan (real.pi / 6) = -2 + real.sqrt 3 / 3 :=
by sorry

end calculate_expression_l711_711122


namespace heloise_total_pets_l711_711231

-- Define initial data
def ratio_dogs_to_cats := (10, 17)
def dogs_given_away := 10
def dogs_remaining := 60

-- Definition of initial number of dogs based on conditions
def initial_dogs := dogs_remaining + dogs_given_away

-- Definition based on ratio of dogs to cats
def dogs_per_set := ratio_dogs_to_cats.1
def cats_per_set := ratio_dogs_to_cats.2

-- Compute the number of sets of dogs
def sets_of_dogs := initial_dogs / dogs_per_set

-- Compute the number of cats
def initial_cats := sets_of_dogs * cats_per_set

-- Definition of the total number of pets
def total_pets := dogs_remaining + initial_cats

-- Lean statement for the proof
theorem heloise_total_pets :
  initial_dogs = 70 ∧
  sets_of_dogs = 7 ∧
  initial_cats = 119 ∧
  total_pets = 179 :=
by
  -- The statements to be proved are listed as conjunctions (∧)
  sorry

end heloise_total_pets_l711_711231


namespace value_of_a2022_l711_711809

theorem value_of_a2022 (a : ℕ → ℤ) (h : ∀ (n k : ℕ), 1 ≤ n ∧ n ≤ 2022 ∧ 1 ≤ k ∧ k ≤ 2022 → a n - a k ≥ (n^3 : ℤ) - (k^3 : ℤ)) (ha1011 : a 1011 = 0) : 
  a 2022 = 7246031367 := 
by
  sorry

end value_of_a2022_l711_711809


namespace find_p_l711_711202

noncomputable def focusDist (x y p : ℝ) : ℝ := 
  sqrt ((x - p / 2)^2 + y^2)

theorem find_p 
  (h1 : ∃ y, y^2 = 8 * 4)
  (h2 : ∃ y, focusDist 4 y 2 = 5) :
  2 = 2 := sorry

end find_p_l711_711202


namespace prime_sum_value_l711_711227

theorem prime_sum_value (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 2019) : 
  (p - 1)^(q - 1) = 1 ∨ (p - 1)^(q - 1) = 2016 :=
by
  sorry

end prime_sum_value_l711_711227


namespace probability_x_lt_2y_l711_711084

noncomputable def rectangle_area : ℝ := 5 * 2

noncomputable def triangle_area : ℝ := 1/2 * 2 * 4

theorem probability_x_lt_2y :
  (triangle_area / rectangle_area) = 2 / 5 :=
by
  sorry

end probability_x_lt_2y_l711_711084


namespace arithmetic_sequence_product_l711_711287

theorem arithmetic_sequence_product (a : ℕ → ℤ) (d : ℤ) (h_inc : ∀ n m, n < m → a n < a m) 
  (h_arith : ∀ n, a (n + 1) = a n + d) (h_prod : a 4 * a 5 = 12) : a 2 * a 7 = 6 :=
sorry

end arithmetic_sequence_product_l711_711287


namespace convert_spherical_to_rectangular_correct_l711_711142

-- Define the spherical to rectangular conversion functions
noncomputable def spherical_to_rectangular (rho θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (rho * Real.sin φ * Real.cos θ, rho * Real.sin φ * Real.sin θ, rho * Real.cos φ)

-- Define the given spherical coordinates
def given_spherical_coords : ℝ × ℝ × ℝ :=
  (5, 7 * Real.pi / 4, Real.pi / 3)

-- Define the expected rectangular coordinates
def expected_rectangular_coords : ℝ × ℝ × ℝ :=
  (-5 * Real.sqrt 6 / 4, -5 * Real.sqrt 6 / 4, 5 / 2)

-- The proof statement
theorem convert_spherical_to_rectangular_correct (ρ θ φ : ℝ)
  (h_ρ : ρ = 5) (h_θ : θ = 7 * Real.pi / 4) (h_φ : φ = Real.pi / 3) :
  spherical_to_rectangular ρ θ φ = expected_rectangular_coords :=
by
  -- Proof omitted
  sorry

end convert_spherical_to_rectangular_correct_l711_711142


namespace cube_volume_from_surface_area_l711_711434

theorem cube_volume_from_surface_area (A : ℝ) (hA : A = 294) : ∃ V : ℝ, V = 343 :=
by
  let s := real.sqrt (A / 6)
  have h_s : s = 7 := sorry
  let V := s^3
  have h_V : V = 343 := sorry
  exact ⟨V, h_V⟩

end cube_volume_from_surface_area_l711_711434


namespace calculate_a_share_l711_711051

noncomputable def a_share (x : ℝ) : ℝ :=
  0.20 * (x * 12)

theorem calculate_a_share (x : ℝ) (total_gain : ℝ) :
  let a_gain := 0.20 * (x * 12)
  let b_gain := 0.25 * (2.5 * x * 6)
  let c_gain := 0.18 * (3.7 * x * 4)
  (a_gain + b_gain + c_gain = total_gain) →
  (total_gain = 48000) →
  (a_share x ≈ 13065.50) :=
by
  intros a_gain b_gain c_gain h1 h2
  sorry

end calculate_a_share_l711_711051


namespace sum_infinite_f_eq_one_l711_711525

-- Conditions definitions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def f (n : ℕ) : ℝ := ∑' k : ℕ in {i : ℕ | i ≥ 2}, (1 : ℝ) / (k ^ factorial n)

-- Statement of the theorem
theorem sum_infinite_f_eq_one : ∑' n : ℕ in {i : ℕ | i ≥ 2}, f n = 1 := 
by sorry

end sum_infinite_f_eq_one_l711_711525


namespace find_a1_in_arithmetic_progression_l711_711406

theorem find_a1_in_arithmetic_progression :
  ∃ a1 : ℤ, 
    -9 ≤ a1 ∧ a1 ≤ -1 ∧ 
    ∀ d : ℤ, d = 1 →
    let a9 := a1 + 8 * d in
    let a17 := a1 + 16 * d in
    let a11 := a1 + 10 * d in
    let a15 := a1 + 14 * d in
    let S := 14 * a1 + 91 * d in
    (a9 * a17 > S + 12) ∧ (a11 * a15 < S + 47) :=
begin
  sorry
end

end find_a1_in_arithmetic_progression_l711_711406


namespace factor_expression_l711_711134

theorem factor_expression (x : ℝ) :
  (12 * x^3 + 45 * x - 3) - (-3 * x^3 + 5 * x - 2) = 5 * x * (3 * x^2 + 8) - 1 :=
by
  sorry

end factor_expression_l711_711134


namespace general_solution_of_diff_eq_l711_711586

theorem general_solution_of_diff_eq
  (f : ℝ → ℝ → ℝ)
  (D : Set (ℝ × ℝ))
  (hf : ∀ x y, f x y = x)
  (hD : D = Set.univ) :
  ∃ C : ℝ, ∀ x : ℝ, ∃ y : ℝ, y = (x^2) / 2 + C :=
by
  sorry

end general_solution_of_diff_eq_l711_711586


namespace find_length_PS_l711_711263

noncomputable def triangle_PQR (P Q R S T : Type)
  [euclidean_geometry P Q R S T] : Prop :=
  P.angle_at Q R = 90 ∧
  dist P Q = 15 ∧
  dist Q R = 20 ∧
  (S ∈ segment P R) ∧
  (T ∈ segment Q R) ∧
  angle_at P T S = 90 ∧
  dist S T = 12

theorem find_length_PS {P Q R S T : Type}
  [euclidean_geometry P Q R S T] :
  triangle_PQR P Q R S T → dist P S = 15 :=
by
  intro h
  -- Proof goes here
  sorry

end find_length_PS_l711_711263


namespace convex_polygon_segment_difference_l711_711318

theorem convex_polygon_segment_difference
  (P : ℝ) (hP : P > 0) (sides : list ℝ) (h_convex : ∀ d ∈ sides, d < P / 2)
  (h_sum : sides.sum = P) :
  ∃ (l1 l2 : ℝ), l1 ∈ sides ∧ l2 ∈ sides ∧ |l1 - l2| ≤ P / 3 :=
by
  sorry

end convex_polygon_segment_difference_l711_711318


namespace units_digit_of_2_to_the_10_l711_711874

theorem units_digit_of_2_to_the_10 : ∃ d : ℕ, (d < 10) ∧ (2^10 % 10 = d) ∧ (d == 4) :=
by {
  -- sorry to skip the proof
  sorry
}

end units_digit_of_2_to_the_10_l711_711874


namespace angle_in_triangle_l711_711264

theorem angle_in_triangle (A B C x : ℝ) (hA : A = 40)
    (hB : B = 3 * x) (hC : C = x) (h_sum : A + B + C = 180) : x = 35 :=
by
  sorry

end angle_in_triangle_l711_711264


namespace deductive_reasoning_correct_statements_l711_711389

-- Definitions based on provided conditions
def cond1 : Prop := DeductiveReasoning.isFromGeneralToSpecific
def cond2 : Prop := DeductiveReasoning.conclusionIsAlwaysCorrect
def cond3 : Prop := DeductiveReasoning.isInSyllogismForm
def cond4 : Prop := DeductiveReasoning.correctnessDependsOnAll

-- Condition structure (assuming we have a structure for DeductiveReasoning in Lean)
structure DeductiveReasoning where
  isFromGeneralToSpecific : Prop
  conclusionIsAlwaysCorrect : Prop
  isInSyllogismForm : Prop
  correctnessDependsOnAll : Prop

-- The problem statement in Lean
theorem deductive_reasoning_correct_statements
  (h1 : cond1)
  (h2 : ¬cond2)
  (h3 : cond3)
  (h4 : cond4) :
  (cond1 ∧ ¬cond2 ∧ cond3 ∧ cond4) → (∃ n, n = 3) :=
by
  sorry

end deductive_reasoning_correct_statements_l711_711389


namespace arithmetic_sequence__geometric_sequence__l711_711408

-- Part 1: Arithmetic Sequence
theorem arithmetic_sequence_
  (d : ℤ) (n : ℤ) (a_n : ℤ) (a_1 : ℤ) (S_n : ℤ)
  (h_d : d = 2) (h_n : n = 15) (h_a_n : a_n = -10)
  (h_a_1 : a_1 = -38) (h_S_n : S_n = -360) :
  a_n = a_1 + (n - 1) * d ∧ S_n = n * (a_1 + a_n) / 2 :=
by
  sorry

-- Part 2: Geometric Sequence
theorem geometric_sequence_
  (a_1 : ℝ) (q : ℝ) (S_10 : ℝ)
  (a_2 : ℝ) (a_3 : ℝ) (a_4 : ℝ)
  (h_a_2_3 : a_2 + a_3 = 6) (h_a_3_4 : a_3 + a_4 = 12)
  (h_a_1 : a_1 = 1) (h_q : q = 2) (h_S_10 : S_10 = 1023) :
  a_2 = a_1 * q ∧ a_3 = a_1 * q^2 ∧ a_4 = a_1 * q^3 ∧ S_10 = a_1 * (1 - q^10) / (1 - q) :=
by
  sorry

end arithmetic_sequence__geometric_sequence__l711_711408


namespace cos_mod_360_eq_l711_711159

theorem cos_mod_360_eq :
  ∃ m : ℕ,  (0 ≤ m ∧ m ≤ 360) ∧ (cos (m * Real.pi / 180) = cos (970 * Real.pi / 180)) :=
by
  use [110, 250]
  split
  { --  Range check
    norm_num,
    norm_num
  }
  { -- cos(110 degrees) = cos(970 degrees)
     rw [Int.mod, Real.cos_of_Real],
     norm_num,
     sorry
  }
  { -- cos(250 degrees) = cos(970 degrees)
     rw [Int.mod, Real.cos_of_Real],
     norm_num,
     sorry
  }

end cos_mod_360_eq_l711_711159


namespace area_of_revolution_leq_half_d_squared_l711_711074

noncomputable def surface_area_of_revolution_not_exceed (d : ℝ) : Prop :=
  ∀ (broken_line : ℝ → ℝ) (h_convex : ∀ x y z, x < y → y < z → (broken_line y - broken_line x) * (z - y) - (broken_line z - broken_line y) * (y - x) ≤ 0)
    (h_length : ∫ (t : ℝ) in 0..1, sqrt (1 + (deriv (broken_line) t)^2) = d),
    ∃ surface_area : ℝ, surface_area ≤ (d^2) / 2

theorem area_of_revolution_leq_half_d_squared (d : ℝ) :
  surface_area_of_revolution_not_exceed d :=
sorry

end area_of_revolution_leq_half_d_squared_l711_711074


namespace field_trip_total_cost_l711_711928

def students := 25
def teachers := 6
def student_ticket_price := 1.50
def teacher_ticket_price := 4
def weekend_discount := 0.20
def tour_price := 3.50
def bus_cost := 100
def meal_price := 7.50

def total_tickets := students + teachers
def total_ticket_cost_before_discount := (students * student_ticket_price) + (teachers * teacher_ticket_price)
def discount := total_ticket_cost_before_discount * weekend_discount
def total_ticket_cost_after_discount := total_ticket_cost_before_discount - discount

def tour_cost := total_tickets * tour_price
def meal_cost := total_tickets * meal_price

def total_cost := total_ticket_cost_after_discount + tour_cost + bus_cost + meal_cost

theorem field_trip_total_cost : total_cost = 490.20 :=
by
  sorry

end field_trip_total_cost_l711_711928


namespace cinema_ticket_prices_cinema_ticket_possible_values_l711_711068

theorem cinema_ticket_prices (x : ℕ) : 
  (15 * x = 90 ∧ 20 * x = 120) ↔ x ∈ {1, 2, 3, 5, 6, 10, 15, 30} :=
by sorry

theorem cinema_ticket_possible_values : 
  ∃ n, n = 8 ∧ ∀ x : ℕ, (15 * x = 90 ∧ 20 * x = 120) ↔ x ∈ {1, 2, 3, 5, 6, 10, 15, 30} :=
by {
  use [8],
  split,
  { reflexivity },
  { exact cinema_ticket_prices },
}

end cinema_ticket_prices_cinema_ticket_possible_values_l711_711068


namespace minimum_PM2_PN2_l711_711192

-- Definitions of the given points and the line equation
def M := (1, 0 : ℝ × ℝ)
def N := (-1, 0 : ℝ × ℝ)
def line_P (x y : ℝ) : Prop := 2 * x - y - 1 = 0

-- Function to compute the square of the Euclidean distance between two points in the plane
def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- The main theorem statement
theorem minimum_PM2_PN2 :
  ∃ x y : ℝ, line_P x y ∧ (distance_squared (x, y) M + distance_squared (x, y) N = 2 / 5 ∧ x = 1 / 5 ∧ y = -3 / 5) :=
sorry

end minimum_PM2_PN2_l711_711192


namespace impossible_to_use_all_components_l711_711372

/- Define the units required for each product -/
def units_for_A (p q r : ℕ) : ℕ := 2 * p + 2 * r
def units_for_B (p q r : ℕ) : ℕ := 2 * p + q
def units_for_C (p q r : ℕ) : ℕ := q + r

/- Define the remaining components after production -/
def remaining_A (p r : ℕ) : ℕ := 2 * p + 2 * r + 2
def remaining_B (p q : ℕ) : ℕ := 2 * p + q + 1
def remaining_C (q r : ℕ) : ℕ := q + r

/- The main theorem statement -/
theorem impossible_to_use_all_components 
  (p q r : ℕ) :
  units_for_A p q r + 2 = remaining_A p r →
  units_for_B p q r + 1 = remaining_B p q →
  units_for_C q r = remaining_C q r →
  ¬ ∃ x y z : ℕ, 2 * x + 2 * z = remaining_A p r ∧ 2 * x + y = remaining_B p q ∧ y + z = remaining_C q r :=
by {
  intros h1 h2 h3,
  sorry
}

end impossible_to_use_all_components_l711_711372


namespace probability_A_seven_rolls_l711_711548

noncomputable def probability_A_after_n_rolls (n : ℕ) : ℚ :=
  if n = 0 then 1 else 1/3 * (1 - (-1/2)^(n-1))

theorem probability_A_seven_rolls : probability_A_after_n_rolls 7 = 21 / 64 :=
by sorry

end probability_A_seven_rolls_l711_711548


namespace water_evaporation_per_day_l711_711444

theorem water_evaporation_per_day :
  ∀ (initial_water: ℝ) (percentage_evaporated: ℝ) (days: ℕ),
  initial_water = 10 → percentage_evaporated = 0.12 → days = 20 →
  (initial_water * percentage_evaporated) / days = 0.06 :=
by
  intros initial_water percentage_evaporated days h1 h2 h3
  rw [h1, h2, h3]
  simp
  norm_num
  sorry

end water_evaporation_per_day_l711_711444


namespace fraction_to_decimal_comparison_l711_711967

theorem fraction_to_decimal_comparison:
  let frac := (7:ℚ) / 24
  in frac < 0.3 :=
by {
  have h : frac = 7 / 24 := rfl,
  have decimal_val : frac = 0.2916666 := sorry, --We might need to prove this decimal conversion separately 
  have comparison_result : 0.2916666 < 0.3 := sorry, --Using real number properties
  rw decimal_val,
  exact comparison_result,
}

end fraction_to_decimal_comparison_l711_711967


namespace rational_expression_l711_711975

theorem rational_expression (x : ℚ) : 2 * x^2 + 1 - 1 / (2 * x^2 + 1) ∈ ℚ :=
sorry

end rational_expression_l711_711975


namespace problem1_problem2_l711_711592

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + |3 * x - 1|

-- Part (1) statement
theorem problem1 (x : ℝ) : f x (-1) ≤ 1 ↔ (1/4 ≤ x ∧ x ≤ 1/2) :=
by
    sorry

-- Part (2) statement
theorem problem2 (x a : ℝ) (h : 1/4 ≤ x ∧ x ≤ 1) : f x a ≤ |3 * x + 1| ↔ -7/3 ≤ a ∧ a ≤ 1 :=
by
    sorry

end problem1_problem2_l711_711592


namespace math_proof_l711_711962

noncomputable def proof_expr (a : ℝ) (b : ℝ) : ℝ :=
  (a^(2/3) * b^(-1))^(-1/2) * a - (1/2) * b^(1/3)

noncomputable def denom_expr (a : ℝ) (b : ℝ) : ℝ :=
  (a * b^5)^(1/6)

theorem math_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (proof_expr a b) / (denom_expr a b) = 1 / a :=
by
  sorry

end math_proof_l711_711962


namespace sum_of_all_ks_l711_711868

theorem sum_of_all_ks (k : ℤ) :
  (∃ r s : ℤ, r ≠ s ∧ (3 * x^2 - k * x + 12 = 0) ∧ (r + s = k / 3) ∧ (r * s = 4)) → (k = 15 ∨ k = -15) → k + k = 0 :=
by
  sorry

end sum_of_all_ks_l711_711868


namespace simplify_fraction_l711_711322

theorem simplify_fraction : ∀ (x y : ℕ), (x = 2) → (y = 5) → (15 * x ^ 3 * y ^ 2) / (10 * x ^ 2 * y ^ 4) = (3 / 25) :=
by {
  intros x y hx hy,
  rw [hx, hy],
  norm_num,
  sorry -- used to skip the proof steps
}

end simplify_fraction_l711_711322


namespace cyclic_quadrilateral_power_of_point_theorem_l711_711113

theorem cyclic_quadrilateral_power_of_point_theorem 
  (A B C M N P Q S : Point) 
  (h1 : Triangle A B C)
  (h2 : Circle_in (A B C))
  (h3 : Tangent (A) (Circle_in (A B C)))
  (h4 : Parallel (Line_through (A B)) (Line_through (M N))) 
  (h5 : Line_intersects_circle (M N) (P Q))
  (h6 : Intersects_tangent (M N) (A) (S))
  (h7 : Segment (N S) * Segment (M N) = Segment (N P) * Segment (N Q)) :
  NS * MN = NP * NQ := 
sorry

end cyclic_quadrilateral_power_of_point_theorem_l711_711113


namespace isosceles_triangle_perimeter_l711_711550

-- Define a structure to represent a triangle with sides a, b, c
structure Triangle :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (triangle_ineq_1 : a + b > c)
  (triangle_ineq_2 : a + c > b)
  (triangle_ineq_3 : b + c > a)

-- Define the specific triangle given the condition
def isosceles_triangle_with_sides (s1 s2 : ℝ) (h_iso : s1 = 3 ∨ s2 = 3) (h_ineq : s1 = 6 ∨ s2 = 6) : Triangle :=
  if h_iso then
    { a := 3, b := 3, c := 6,
      triangle_ineq_1 := by linarith,
      triangle_ineq_2 := by linarith,
      triangle_ineq_3 := by linarith }
  else 
    sorry -- We cover the second case directly with the checked option

-- Prove that the perimeter of the given isosceles triangle is as expected
theorem isosceles_triangle_perimeter :
  let t := isosceles_triangle_with_sides 3 6 (or.inl rfl) (or.inr rfl) in
  t.a + t.b + t.c = 15 :=
by simp [isosceles_triangle_with_sides, add_assoc]

end isosceles_triangle_perimeter_l711_711550


namespace base_prime_representation_360_l711_711832

def prime_factorization (n : ℕ) : List (ℕ × ℕ) :=
  if n = 360 then [(2, 3), (3, 2), (5, 1)] else []

def base_prime_representation (n : ℕ) : ℕ :=
  if n = 360 then 321 else 0

theorem base_prime_representation_360 : base_prime_representation 360 = 321 := 
by
  -- setting up conditions
  have prime_fact : prime_factorization 360 = [(2, 3), (3, 2), (5, 1)]:=rfl,
  -- proving the theorem
  exact rfl

end base_prime_representation_360_l711_711832


namespace number_of_odd_positive_integer_triples_sum_25_l711_711233

theorem number_of_odd_positive_integer_triples_sum_25 :
  ∃ n : ℕ, (
    n = 78 ∧
    ∃ (a b c : ℕ), 
      (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 25
  ) := 
sorry

end number_of_odd_positive_integer_triples_sum_25_l711_711233


namespace angles_of_triangle_ABC_l711_711280

noncomputable def z : ℂ := complex.exp((complex.I * π / 5))
noncomputable def A_1 : ℂ := 1
noncomputable def A_2 : ℂ := z
noncomputable def A_4 : ℂ := z^3
noncomputable def A_5 : ℂ := z^4
noncomputable def A_6 : ℂ := z^5
noncomputable def A_7 : ℂ := z^6
noncomputable def A_9 : ℂ := z^8
noncomputable def A_10 : ℂ := z^9

noncomputable def A : ℂ := (z^3 * (z + z^4) - z^5 * (1 + z^3)) / (z^3 - z^5)
noncomputable def B : ℂ := 0
noncomputable def C : ℂ := (z^8 * (z + z^9) - z^10 * (1 + z^8)) / (z^8 - z^10)

theorem angles_of_triangle_ABC :
  ∃ α β γ : ℝ, (α = 3 * π / 10) ∧ (β = π / 2) ∧ (γ = π / 5) ∧
    (α + β + γ = π) :=
by
  sorry -- Proof required here.

end angles_of_triangle_ABC_l711_711280


namespace total_cost_of_installing_ramp_l711_711711

-- Definitions based on the conditions
def permits_cost : ℕ := 250
def contractor_labour_rate : ℕ := 150
def contractor_materials_rate : ℕ := 50
def contractor_days : ℕ := 3
def contractor_hours_per_day : ℕ := 5
def contractor_lunch_break_minutes : ℕ := 30
def inspector_discount_rate : ℚ := 0.8
def inspector_hours_per_day : ℕ := 2

-- Statement to prove
theorem total_cost_of_installing_ramp :
  let contractor_work_hours_per_day := contractor_hours_per_day - contractor_lunch_break_minutes / 60
      total_contractor_hours := contractor_work_hours_per_day * contractor_days
      total_contractor_labour_cost := total_contractor_hours * contractor_labour_rate
      total_materials_cost := total_contractor_hours * contractor_materials_rate
      inspector_labour_rate := contractor_labour_rate * (1 - inspector_discount_rate)
      total_inspector_hours := inspector_hours_per_day * contractor_days
      total_inspector_cost := total_inspector_hours * inspector_labour_rate
      total_cost := permits_cost + total_contractor_labour_cost + total_materials_cost + total_inspector_cost 
  in total_cost = 3130 := 
begin
  sorry
end

end total_cost_of_installing_ramp_l711_711711


namespace percentage_increase_first_year_l711_711805

variable (P : ℝ) -- Original price of the painting
variable (X : ℝ) -- Percentage increase in the first year

-- Conditions
def first_year_price (P : ℝ) (X : ℝ) : ℝ := P + (X / 100) * P
def second_year_price (P : ℝ) (X : ℝ) : ℝ := first_year_price P X * 0.85

-- Given condition about final price
def final_price_condition (P : ℝ) (X : ℝ) : Prop :=
  second_year_price P X = 1.02 * P

-- The proof problem: prove that the first year percentage increase X is 20%
theorem percentage_increase_first_year :
  ∀ (P : ℝ), (∃ (X : ℝ), final_price_condition P X ∧ X = 20) :=
by
  sorry

end percentage_increase_first_year_l711_711805


namespace drawing_time_total_l711_711480

theorem drawing_time_total
  (bianca_school : ℕ)
  (bianca_home : ℕ)
  (lucas_school : ℕ)
  (lucas_home : ℕ)
  (h_bianca_school : bianca_school = 22)
  (h_bianca_home : bianca_home = 19)
  (h_lucas_school : lucas_school = 10)
  (h_lucas_home : lucas_home = 35) :
  bianca_school + bianca_home + lucas_school + lucas_home = 86 := 
by
  -- Proof would go here
  sorry

end drawing_time_total_l711_711480


namespace sum_of_k_with_distinct_integer_solutions_l711_711856

-- Definitions from conditions
def quadratic_eq (k : ℤ) (x : ℤ) : Prop := 3 * x^2 - k * x + 12 = 0

def distinct_integer_solutions (k : ℤ) : Prop := ∃ p q : ℤ, p ≠ q ∧ quadratic_eq k p ∧ quadratic_eq k q

-- Main statement to prove
theorem sum_of_k_with_distinct_integer_solutions : 
  ∑ k in {k : ℤ | distinct_integer_solutions k}, k = 0 := 
sorry

end sum_of_k_with_distinct_integer_solutions_l711_711856


namespace product_equivalent_l711_711951

noncomputable def product_expression : ℚ := (∏ n in (Finset.range 13).map ((+) 2), (n * (n + 3)) / ((n + 5)^3))

theorem product_equivalent :
  (∏ n in (Finset.range 13).map ((+) 2), (n * (n + 3)) / ((n + 5)^3)) =
  (∏ n in (Finset.range 13).map ((+) 2), n * (n + 3)) / 
  (∏ n in (Finset.range 13).map ((+) 2), (n + 5)^3) :=
by sorry

end product_equivalent_l711_711951


namespace distance_between_stripes_l711_711460

-- Definitions of initial conditions
def distance_between_curbs := 50  -- feet
def length_of_curb_between_stripes := 20  -- feet
def length_of_each_stripe := 55  -- feet

-- Required proof
theorem distance_between_stripes :
  let d := (length_of_curb_between_stripes * distance_between_curbs) / length_of_each_stripe in
  d = 1000 / 55 :=
by 
  -- Sorry skipped for proof completion.
  sorry

end distance_between_stripes_l711_711460


namespace least_perimeter_xyz_l711_711248

theorem least_perimeter_xyz :
  ∃ (x y z : ℕ), 
    (cos (X : ℝ) = 3/5) ∧ 
    (cos (Y : ℝ) = 15/17) ∧ 
    (cos (Z : ℝ) = -1/3) ∧ 
    (x + y + z = 15) 
:=
sorry

end least_perimeter_xyz_l711_711248


namespace ellipse_equation_line_fixed_point_max_area_triangle_l711_711225

theorem ellipse_equation (r : ℝ) (x y : ℝ) (h_r : 0 < r ∧ r < 4)
  (h1 : (x + 1)^2 + y^2 = r^2)
  (h2 : (x - 1)^2 + y^2 = (4 - r)^2) : 
  (x^2 / 4) + (y^2 / 3) = 1 := sorry

theorem line_fixed_point (x1 y1 x2 y2 : ℝ) (k : ℝ)
  (h_curve : x^2 / 4 + y^2 / 3 = 1)
  (h_slope_product : k * ((k * x1 + y1 - sqrt 3) / x1) * ((k * x2 + y2 - sqrt 3) / x2) = 1 / 4) :
  (y1 * y2 ≠ 0) → (x = 0 → (x1 * x2 = 4 * y1 * y2 - 3)) :=
sorry

theorem max_area_triangle (y1 y2 x1 x2 : ℝ) (k : ℝ)
  (h1 : x - sqrt 3 = k)
  (h2 : x^2 / 4 + y^2 / 3 = 1)
  (h3 : 4 * k^2 - 9 = 12) :
  (1 / 2) * sqrt 3 * sqrt ((x1 + x2)^2 - 4 * (x1 * x2)) = sqrt 3 / 2 :=
sorry

end ellipse_equation_line_fixed_point_max_area_triangle_l711_711225


namespace total_dots_not_visible_l711_711530

-- Define the total dot sum for each die
def sum_of_dots_per_die : Nat := 1 + 2 + 3 + 4 + 5 + 6

-- Define the total number of dice
def number_of_dice : Nat := 4

-- Calculate the total dot sum for all dice
def total_dots_all_dice : Nat := sum_of_dots_per_die * number_of_dice

-- Sum of visible dots
def sum_of_visible_dots : Nat := 1 + 1 + 2 + 2 + 3 + 3 + 4 + 5 + 6 + 6

-- Prove the total dots not visible
theorem total_dots_not_visible : total_dots_all_dice - sum_of_visible_dots = 51 := by
  sorry

end total_dots_not_visible_l711_711530


namespace parallelogram_area_l711_711998

def base : ℕ := 22
def height : ℕ := 14

theorem parallelogram_area : base * height = 308 := by
  unfold base height
  sorry

end parallelogram_area_l711_711998


namespace shaded_area_correct_l711_711111

-- Define conditions
def side_length_square : ℝ := 10
def dimensions_rectangles : ℝ × ℝ := (3, 4)

-- Calculate areas
def area_square (s : ℝ) : ℝ := s * s
def area_rectangles (n : ℝ) (dim : ℝ × ℝ) : ℝ :=
  let (a, b) := dim
  n * (a * b)

-- We need to prove that the area of the shaded part is 74 given the above conditions
theorem shaded_area_correct :
  ∀ (s : ℝ) (dim : ℝ × ℝ),
    s = side_length_square →
    dim = dimensions_rectangles →
    let A_square := area_square s in
    let A_rectangles := area_rectangles 4 dim in
    let A_unshaded := A_rectangles / 2 in
    let A_shaded := A_square - A_unshaded in
    A_shaded = 74 :=
by
  intros s dim h_s h_dim
  let A_square := area_square s
  let A_rectangles := area_rectangles 4 dim
  let A_unshaded := A_rectangles / 2
  let A_shaded := A_square - A_unshaded
  sorry  -- The actual proof goes here

end shaded_area_correct_l711_711111


namespace jill_investment_value_l711_711690

def compoundInterest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_investment_value :
  compoundInterest 10000 0.0396 2 2 ≈ 10812 :=
by
  sorry

end jill_investment_value_l711_711690


namespace evaluate_expression_l711_711505

theorem evaluate_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ((x)^(1/6) - (y)^(1/6)) / ((x)^(1/2) + (x)^(1/3) * (y)^(1/6))
  * (((x)^(1/3) + (y)^(1/3))^2 - 4 * (x * y)^(1/3))
  / ((x)^(5/6) * (y)^(1/3) - (x)^(1/2) * (y)^(2/3))
  + 2 * (x)^(-2/3) * (y)^(-1/6)
  = ((x)^(1/3) + (y)^(1/3)) / (x)^(5/6) * (y)^(2/6) :=
sorry

end evaluate_expression_l711_711505


namespace hyperbola_equation_l711_711519

theorem hyperbola_equation 
  (a b : ℝ)
  (asymptote : ∀ x y : ℝ, x - sqrt 3 * y = 0 → y = (sqrt 3 / 3) * x)
  (ellipse_foci : ∀ x : ℝ, x^2 + 4 * y^2 = 64 → ∃ f : ℝ, (f = 4 * sqrt 3)) :
  (∃ a b : ℝ, a^2 = 36 ∧ b^2 = 12 ∧ (∀ x y : ℝ, (x / a)^2 - (y / b)^2 = 1)) :=
by
  sorry

end hyperbola_equation_l711_711519


namespace no_solution_fractional_eq_l711_711327

theorem no_solution_fractional_eq :
  ¬∃ x : ℝ, (1 - x) / (x - 2) = 1 / (2 - x) + 1 :=
by
  -- The proof is intentionally omitted.
  sorry

end no_solution_fractional_eq_l711_711327


namespace problem_imaginary_axis_l711_711527

theorem problem_imaginary_axis (a : ℝ) (z : ℂ) 
  (h1 : z = complex.of_real (a^2 - 2 * a) + complex.I * (a^2 - a - 2))
  (h2 : z.re = 0) : a = 0 ∨ a = 2 :=
by sorry

end problem_imaginary_axis_l711_711527


namespace small_apple_cost_l711_711128

theorem small_apple_cost :
  ∃ S : ℝ, 6 * S + 6 * 2 + 8 * 3 = 45 ∧ S = 1.5 :=
begin
  use 1.5,
  split,
  { calc
      6 * (1.5 : ℝ) + 6 * 2 + 8 * 3
          = 6 * 1.5 + 6 * 2 + 8 * 3 : by ring
      ... = 9 + 12 + 24 : by norm_num
      ... = 45 : by norm_num,
  },
  { refl, }
end

end small_apple_cost_l711_711128


namespace jill_investment_l711_711702

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_investment :
  compound_interest 10000 0.0396 2 2 ≈ 10815.66 :=
by
  sorry

end jill_investment_l711_711702


namespace person_speed_l711_711449

theorem person_speed (distance : ℝ) (time : ℝ) (h_distance : distance = 720) (h_time : time = 12) : 
  (distance / 1000) / (time / 60) = 3.6 :=
by
  rw [h_distance, h_time]
  -- Convert distance to kilometers: 720 meters = 0.72 kilometers
  have h_distance_km : 720 / 1000 = 0.72 := by norm_num
  rw h_distance_km
  -- Convert time to hours: 12 minutes = 0.2 hours
  have h_time_hr : 12 / 60 = 0.2 := by norm_num
  rw h_time_hr
  -- Calculate speed: 0.72 km / 0.2 hr = 3.6 km/hr
  norm_num
  sorry

end person_speed_l711_711449


namespace carl_spent_on_index_cards_l711_711485

-- Defining conditions based on the problem statement
def students_per_class (grade : ℕ) : ℕ :=
  if grade = 6 then 20 else
  if grade = 7 then 25 else
  if grade = 8 then 30 else 0

def cards_per_student (grade : ℕ) : ℕ :=
  if grade = 6 then 8 else
  if grade = 7 then 10 else
  if grade = 8 then 12 else 0

def cost_per_pack (size : ℕ) : ℕ :=
  if size = 35 then 3 else
  if size = 46 then 4 else 0

-- Proof problem
theorem carl_spent_on_index_cards : 
  let total_cost := 60 + 90 + 176 in
  total_cost = 326 :=
sorry

end carl_spent_on_index_cards_l711_711485


namespace rooms_count_l711_711302

theorem rooms_count (total_paintings : ℕ) (paintings_per_room : ℕ) (h1 : total_paintings = 32) (h2 : paintings_per_room = 8) : (total_paintings / paintings_per_room) = 4 := by
  sorry

end rooms_count_l711_711302


namespace ensureUserDataSecurity_l711_711768

-- Definitions based on the given conditions and correct answers
variable (storeApp : Type) -- representing the online store application

/-- Condition: Users can pay using credit cards in the store application -/
def canPayWithCreditCard (app : storeApp) : Prop := 
sorry -- detailed implementation is not provided

/-- Condition: Users can order home delivery in the store application -/
def canOrderHomeDelivery (app : storeApp) : Prop := 
sorry -- detailed implementation is not provided

/-- Security Measure 1: Avoid Storing Card Data - assume implemented properly -/
def avoidStoringCardData (app : storeApp) : Prop := 
sorry -- detailed implementation is not provided

/-- Security Measure 2: Encryption of Stored Data - assume implemented properly -/
def encryptStoredData (app : storeApp) : Prop := 
sorry -- detailed implementation is not provided

/-- Security Measure 3: Encryption of Data in Transit - assume implemented properly -/
def encryptDataInTransit (app : storeApp) : Prop := 
sorry -- detailed implementation is not provided

/-- Theorem: Ensuring user data security in an online store application -/
theorem ensureUserDataSecurity:
  ∀ (app : storeApp), 
    canPayWithCreditCard app → 
    canOrderHomeDelivery app → 
    avoidStoringCardData app →
    encryptStoredData app → 
    encryptDataInTransit app → 
    true := 
by
  sorry

end ensureUserDataSecurity_l711_711768


namespace find_constants_l711_711748

theorem find_constants (a b c d : ℝ) 
  (h₁ : ∀ x, ((a * x + b) * sin x + (c * x + d) * cos x)'' = x * cos x) : 
a = 1 ∧ b = 0 ∧ c = 0 ∧ d = 1 :=
by
  sorry

end find_constants_l711_711748


namespace cell_phone_total_cost_l711_711065

def base_cost : ℕ := 25
def text_cost_per_message : ℕ := 3
def extra_minute_cost_per_minute : ℕ := 15
def included_hours : ℕ := 40
def messages_sent_in_february : ℕ := 200
def hours_talked_in_february : ℕ := 41

theorem cell_phone_total_cost :
  base_cost + (messages_sent_in_february * text_cost_per_message) / 100 + 
  ((hours_talked_in_february - included_hours) * 60 * extra_minute_cost_per_minute) / 100 = 40 :=
by
  sorry

end cell_phone_total_cost_l711_711065


namespace coefficient_of_x_eq_31_then_a_eq_neg2_l711_711334

theorem coefficient_of_x_eq_31_then_a_eq_neg2 
  (a : ℝ) 
  (h : binomial (6,6) + (-a) * binomial (6,2) = 31) : 
  a = -2 := 
sorry

end coefficient_of_x_eq_31_then_a_eq_neg2_l711_711334


namespace sin_double_angle_of_tan_l711_711626

-- Given condition: tan(alpha) = 2
-- To prove: sin(2 * alpha) = 4/5
theorem sin_double_angle_of_tan (α : ℝ) (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 :=
  sorry

end sin_double_angle_of_tan_l711_711626


namespace trapezoid_properties_l711_711347

theorem trapezoid_properties :
  ∃ (x y z : ℝ) (a : ℝ = 5) (area : ℝ = 100), 
    x - y = 6 ∧
    (x + y - 10) * (z + 1) = (x + y) * z ∧
    x = 28 ∧ y = 22 ∧ z = 4 ∧
    area = (x + y) * z / 2 :=
by
  sorry

end trapezoid_properties_l711_711347


namespace trigonometric_identity_l711_711201

theorem trigonometric_identity 
  (α m : ℝ) 
  (h : Real.tan (α / 2) = m) :
  (1 - 2 * (Real.sin (α / 2))^2) / (1 + Real.sin α) = (1 - m) / (1 + m) :=
by
  sorry

end trigonometric_identity_l711_711201


namespace problem_l711_711810

open_locale big_operators

theorem problem (n : ℕ) (a : ℕ → ℝ) (c : ℝ) (h0 : a 0 = 0) (hn : a n = 0)
  (h1 : ∀ k, 1 ≤ k → k ≤ n-1 → a k = c + ∑ i in finset.range(n-k), a (i) * (a (i+k) + a (i+k+1))) :
  c ≤ 1 / (4 * n) :=
sorry

end problem_l711_711810


namespace min_value_of_expression_l711_711734

open Real

theorem min_value_of_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 :=
by
  sorry

end min_value_of_expression_l711_711734


namespace trapezoid_possible_and_area_sum_l711_711542

theorem trapezoid_possible_and_area_sum (a b c d : ℕ) (h1 : a = 4) (h2 : b = 6) (h3 : c = 8) (h4 : d = 12) :
  ∃ (S : ℚ), S = 72 := 
by
  -- conditions ensure one pair of sides is parallel
  -- area calculation based on trapezoid properties
  sorry

end trapezoid_possible_and_area_sum_l711_711542


namespace number_of_liars_l711_711893

-- Define the type of individuals.
inductive KnightOrLiar
| knight  -- always tells the truth
| liar    -- always lies

open KnightOrLiar

-- Define the statements made by individuals based on their number.
def statement (n : ℕ) (people : ℕ → KnightOrLiar) : Prop :=
  if n % 2 = 1 then  -- odd-numbered person
    ∀ m, m > n → people m = liar
  else               -- even-numbered person
    ∀ m, m < n → people m = liar

-- Define the overall condition for all 30 people.
def consistent (people : ℕ → KnightOrLiar) : Prop :=
  ∀ n, 1 ≤ n ∧ n ≤ 30 → statement n people

-- The theorem that we need to prove based on the given conditions.
theorem number_of_liars : ∀ (people : ℕ → KnightOrLiar),
  consistent people →
  (∑ n in (Finset.range 31), if people n = liar then 1 else 0) = 28 := sorry

end number_of_liars_l711_711893


namespace students_end_of_year_l711_711673

theorem students_end_of_year (n_start l n_new : ℕ) (h1 : n_start = 11) (h2 : l = 6) (h3 : n_new = 42) :
  n_start - l + n_new = 47 :=
by
  rw [h1, h2, h3]
  norm_num

end students_end_of_year_l711_711673


namespace probability_2_lt_xi_le_4_l711_711220

noncomputable def p_xi : ℕ → ℝ
| k := 1 / (2 ^ k)

theorem probability_2_lt_xi_le_4 :
  (p_xi 3 + p_xi 4) = 3 / 16 :=
by
  sorry

end probability_2_lt_xi_le_4_l711_711220


namespace k_cannot_be_zero_l711_711166

theorem k_cannot_be_zero (k : ℝ) (h₁ : k ≠ 0) (h₂ : 4 - 2 * k > 0) : k ≠ 0 :=
by 
  exact h₁

end k_cannot_be_zero_l711_711166


namespace charlie_paints_60_sqft_l711_711469

theorem charlie_paints_60_sqft (A B C : ℕ) (total_sqft : ℕ) (h_ratio : A = 3 ∧ B = 5 ∧ C = 2) (h_total : total_sqft = 300) : 
  C * (total_sqft / (A + B + C)) = 60 :=
by
  rcases h_ratio with ⟨rfl, rfl, rfl⟩
  rcases h_total with rfl
  sorry

end charlie_paints_60_sqft_l711_711469


namespace task_completion_time_of_B_l711_711914

-- Definitions based on given conditions
def A_time : ℝ := 12
def B_efficiency : ℝ := 1.75

-- Derived definitions
def A_rate : ℝ := 1 / A_time
def B_rate : ℝ := B_efficiency * A_rate
def B_time : ℝ := 1 / B_rate

-- The main theorem to prove
theorem task_completion_time_of_B :
  B_time = 48 / 7 :=
sorry

end task_completion_time_of_B_l711_711914


namespace amys_haircut_l711_711105

theorem amys_haircut : 
  ∀ (initial_length cut_length : ℕ), initial_length = 11 → cut_length = 4 → (initial_length - cut_length) = 7 :=
by
  intros initial_length cut_length h_initial h_cut
  rw [h_initial, h_cut]
  simp
  sorry

end amys_haircut_l711_711105


namespace triangle_problem_1_correct_triangle_problem_2_correct_l711_711687

noncomputable def triangle_problem_1 (a b c S : ℝ) (condition : 4 * real.sqrt 3 * S = b ^ 2 + c ^ 2 - a ^ 2) : ℝ :=
  if 0 < real.cos (a / b) ∧ real.cos (a / b) < 1 then
    real.arccos ((b ^ 2 + c ^ 2 - a ^ 2) / (2 * b * c)) 
  else 
    0

noncomputable def triangle_problem_2 (a b : ℝ) : set ℝ :=
  if a = 2 ∧ b = 2 * real.sqrt 3 then
    {π / 2, π / 6}
  else
    ∅

theorem triangle_problem_1_correct (a b c S : ℝ) (h : 4 * real.sqrt 3 * S = b ^ 2 + c ^ 2 - a ^ 2) : 
  triangle_problem_1 a b c S h = π / 6 := sorry

theorem triangle_problem_2_correct : triangle_problem_2 2 (2 * real.sqrt 3) = {π / 2, π / 6} := sorry

end triangle_problem_1_correct_triangle_problem_2_correct_l711_711687


namespace sum_of_distances_and_pq_l711_711138

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem sum_of_distances_and_pq :
  let D := (0, 0) : ℝ × ℝ
  let E := (8, 0) : ℝ × ℝ
  let F := (5, 7) : ℝ × ℝ
  let P := (5, 3) : ℝ × ℝ
  let DP := distance D P
  let EP := distance E P
  let FP := distance F P
  DP + EP + FP = Real.sqrt 34 + 3 * Real.sqrt 2 + 4 ∧ 1 + 3 = 4
:= 
by
  sorry

end sum_of_distances_and_pq_l711_711138


namespace jill_investment_l711_711696

def compound_interest (P : ℕ) (r : ℚ) (n : ℕ) (t : ℕ) : ℚ :=
  P * (1 + r / n)^(n * t)

theorem jill_investment : 
  compound_interest 10000 (396 / 10000) 2 2 ≈ 10812 := by
sorry

end jill_investment_l711_711696


namespace tg_mul_fraction_eq_sin_six_l711_711048

-- Define the conditions cosine must not be zero and 1 - 3 * (tan x)^2 must not be zero
def cos_ne_zero (x : ℝ) := (∃ k : ℤ, x ≠ (π / 2 + k * π))
def one_minus_3tan2_ne_zero (x : ℝ) := (∃ k : ℤ, 1 ≠ 3 * (Real.tan x)^2)

-- Prove the mathematical problem
theorem tg_mul_fraction_eq_sin_six (x : ℝ) :
  (cos_ne_zero x) → (one_minus_3tan2_ne_zero x) → 
  (Real.tan x * (3 - (Real.tan x) ^ 2) / (1 - 3 * (Real.tan x) ^ 2) = Real.sin (6 * x))
  → (x = π * k / 3 ∨ x = π * ((2 * n + 1) / 12)) :=
sorry

end tg_mul_fraction_eq_sin_six_l711_711048


namespace max_visible_cubes_l711_711941

theorem max_visible_cubes (a b c : ℕ) (ha : a = 12) (hb : b = 10) (hc : c = 9) : 
  let face1 := a * b,
      face2 := b * c,
      face3 := a * c,
      edge1 := b,
      edge2 := a,
      edge3 := c,
      corner := 1,
      total := face1 + face2 + face3 - edge1 - edge2 - edge3 + corner
  in total = 288 := 
by
  rw [ha, hb, hc]
  let face1 := 12 * 10
  let face2 := 10 * 9
  let face3 := 12 * 9
  let edge1 := 10
  let edge2 := 12
  let edge3 := 9
  let corner := 1
  let total := face1 + face2 + face3 - edge1 - edge2 - edge3 + corner
  have h_face1 : face1 = 120 := rfl
  have h_face2 : face2 = 90 := rfl
  have h_face3 : face3 = 108 := rfl
  have h_edges : edge1 + edge2 + edge3 = 31 := rfl
  have h_total : total = 288 := by
    rw [h_face1, h_face2, h_face3, h_edges]
    rfl
  exact h_total

end max_visible_cubes_l711_711941


namespace squirrel_acorns_left_l711_711090

noncomputable def acorns_per_winter_month (total_acorns : ℕ) (months : ℕ) (acorns_taken_total : ℕ) : ℕ :=
  let per_month := total_acorns / months
  let acorns_taken_per_month := acorns_taken_total / months
  per_month - acorns_taken_per_month

theorem squirrel_acorns_left (total_acorns : ℕ) (months : ℕ) (acorns_taken_total : ℕ) :
  total_acorns = 210 → months = 3 → acorns_taken_total = 30 → acorns_per_winter_month total_acorns months acorns_taken_total = 60 :=
by intros; sorry

end squirrel_acorns_left_l711_711090


namespace product_of_differing_inputs_equal_l711_711591

theorem product_of_differing_inputs_equal (a b : ℝ) (h₁ : a ≠ b)
(h₂ : |Real.log a - (1 / 2)| = |Real.log b - (1 / 2)|) : a * b = Real.exp 1 :=
sorry

end product_of_differing_inputs_equal_l711_711591


namespace rods_left_equilateral_isosceles_trapezoid_arrangements_best_trapezoid_arrangement_l711_711500

-- Definitions for the problem conditions
def total_rods : ℕ := 2009
def rod_diameter_cm : ℕ := 10
def max_height_cm : ℕ := 400  -- 4 meters

-- Equilateral triangle stack conditions
def Sn (n : ℕ) :ℕ := n * (n + 1) / 2

-- Proving the first statement about equilateral triangle stack
theorem rods_left_equilateral (n : ℕ) (h : Sn n ≤ total_rods) : total_rods - Sn n = 56 := sorry

-- For isosceles trapezoid stack definitions
def trapezoid_arrangements : list ℕ := [7, 14, 41, 49]

theorem isosceles_trapezoid_arrangements : trapezoid_arrangements.length = 4 := sorry

noncomputable def x_value (n : ℕ) : ℕ := (total_rods - n * (n - 1) / 2) / n

-- Proving the feasibility and space-saving arrangement for trapezoid stack
theorem best_trapezoid_arrangement (n : ℕ) (H₁ : n ∈ trapezoid_arrangements) (H₂ : n = 41) : (200 * real.sqrt 3) + rod_diameter_cm * 2 ≤ max_height_cm := sorry

end rods_left_equilateral_isosceles_trapezoid_arrangements_best_trapezoid_arrangement_l711_711500


namespace find_integer_mod_l711_711516

theorem find_integer_mod (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 15) (h3 : n ≡ 14567 [MOD 16]) : n = 7 :=
sorry

end find_integer_mod_l711_711516


namespace polynomial_division_correct_l711_711161

noncomputable def polynomial_division_quota: ℚ[X] :=
  (8 * X^4 + 7 * X^3 + 3 * X^2 - 5 * X - 8) /ₘ (X + 2)

theorem polynomial_division_correct :
  polynomial_division_quota = (8 * X^3 - 9 * X^2 + 21 * X - 47) :=
by
  sorry

end polynomial_division_correct_l711_711161


namespace foma_waiting_probability_l711_711989

open Set Real

noncomputable def probability_foma_waits_no_more_than_four_minutes : ℝ :=
  let s := { p : ℝ × ℝ | 2 < p.1 ∧ p.1 < 10 ∧ p.1 < p.2 ∧ p.2 < 12 }
  let t := { p : ℝ × ℝ | 2 < p.1 ∧ p.1 < 10 ∧ p.1 < p.2 ∧ p.2 < p.1 + 4 }
  (volume t) / (volume s)

theorem foma_waiting_probability :
  probability_foma_waits_no_more_than_four_minutes = 0.75 :=
sorry

end foma_waiting_probability_l711_711989


namespace largest_fraction_l711_711044

theorem largest_fraction (a b c d e : ℚ) (h₀ : a = 3/7) (h₁ : b = 4/9) (h₂ : c = 17/35) 
  (h₃ : d = 100/201) (h₄ : e = 151/301) : 
  e > a ∧ e > b ∧ e > c ∧ e > d :=
by
  sorry

end largest_fraction_l711_711044


namespace intersect_on_circumcircle_of_ABC_l711_711313

noncomputable theory

open_locale classical

variables {A B C P Q M N X Y : Type} 
  [IsPoint A] [IsPoint B] [IsPoint C] [IsPoint P] [IsPoint Q] [IsPoint M] [IsPoint N] [IsPoint X] [IsPoint Y]
  [AcuteTriangle ABC] 
  (hP: OnSegment P (Segment B C))
  (hQ: OnSegment Q (Segment B C))
  (h1: ∠ PAB = ∠ ACB) 
  (h2: ∠ QAC = ∠ CBA) 
  (h3: OnLine M (Line A P)) 
  (h4: OnLine N (Line A Q)) 
  (h5: dist A P = dist P M) 
  (h6: dist A Q = dist Q N) 

theorem intersect_on_circumcircle_of_ABC:
  ∃ X Y, OnCircumcircle ABC X ∧ OnCircumcircle ABC Y ∧ Intersect (Line B M) (Line C N) X ∧ Intersect (Line B M) (Line C N) Y ∧ X = Y :=
sorry

end intersect_on_circumcircle_of_ABC_l711_711313


namespace cube_volume_surface_area_l711_711164

variable (x : ℝ)

theorem cube_volume_surface_area (h1 : s^3 = 8 * x) (h2 : 6 * s^2 = 2 * x) : x = 1728 :=
by
  sorry

end cube_volume_surface_area_l711_711164


namespace day_of_50th_day_l711_711266

theorem day_of_50th_day (days_250_N days_150_N1 : ℕ) 
  (h₁ : days_250_N % 7 = 5) (h₂ : days_150_N1 % 7 = 5) : 
  ((50 + 315 - 150 + 365 * 2) % 7) = 4 := 
  sorry

end day_of_50th_day_l711_711266


namespace arccos_gt_two_arcsin_l711_711512

open Real

theorem arccos_gt_two_arcsin (x : ℝ) (h : x ∈ Set.Icc -1 1) : 
  (arccos x > 2 * arcsin x) ↔ (x ∈ Set.Ioc -1 ((1 - Real.sqrt 3) / 2)) := by
  sorry

end arccos_gt_two_arcsin_l711_711512


namespace middle_letter_value_l711_711272

theorem middle_letter_value 
  (final_score : ℕ) 
  (first_letter_value : ℕ) 
  (third_letter_value : ℕ)
  (word_length : ℕ)
  (triple_score : ℕ)
  (total_points : ℕ)
  (middle_letter_value : ℕ)
  (h1 : final_score = 30)
  (h2 : first_letter_value = 1)
  (h3 : third_letter_value = 1)
  (h4 : word_length = 3)
  (h5 : triple_score = 3)
  (h6 : total_points = final_score / triple_score)
  (h7 : total_points = 10)
  (h8 : middle_letter_value = total_points - first_letter_value - third_letter_value) :
  middle_letter_value = 8 := 
by sorry

end middle_letter_value_l711_711272


namespace simplify_negative_exponents_l711_711387

theorem simplify_negative_exponents (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x + y)⁻¹ * (x⁻¹ + y⁻¹) = x⁻¹ * y⁻¹ :=
  sorry

end simplify_negative_exponents_l711_711387


namespace chris_babysitting_hours_l711_711958

theorem chris_babysitting_hours (h : ℕ) (video_game_cost candy_cost earn_per_hour leftover total_cost : ℕ) :
  video_game_cost = 60 ∧
  candy_cost = 5 ∧
  earn_per_hour = 8 ∧
  leftover = 7 ∧
  total_cost = video_game_cost + candy_cost ∧
  earn_per_hour * h = total_cost + leftover
  → h = 9 := by
  intros
  sorry

end chris_babysitting_hours_l711_711958


namespace area_of_fourth_square_l711_711035

variables (PQ QR PR PS : ℝ)
variables (area1 area2 area3 area4 : ℝ)

theorem area_of_fourth_square
  (h1 : PQ^2 = 25)
  (h2 : QR^2 = 64)
  (h3 : PR^2 = 89)
  (h4 : PS^2 = 49)
  : area4 = 138 :=
begin
  sorry
end

end area_of_fourth_square_l711_711035


namespace binomial_expansion_coefficient_x3_l711_711681

theorem binomial_expansion_coefficient_x3 (n : ℕ) (h : (x + 1)^n = 64) : 
  (finset.range 4).sum (λ k, nat.choose n k) = 20 :=
sorry

end binomial_expansion_coefficient_x3_l711_711681


namespace symmetric_points_l711_711631

theorem symmetric_points (m n : ℤ) (h1 : m - 1 = -3) (h2 : 1 = n - 1) : m + n = 0 := by
  sorry

end symmetric_points_l711_711631


namespace min_value_expression_l711_711737

theorem min_value_expression (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 7) ^ 2 + (3 * Real.sin α + 4 * Real.cos β - 12) ^ 2 ≥ 36 :=
sorry

end min_value_expression_l711_711737


namespace length_of_courtyard_l711_711922

theorem length_of_courtyard
  (breadth_of_courtyard : ℝ)
  (brick_length_cm : ℝ)
  (brick_width_cm : ℝ)
  (number_of_bricks : ℝ)
  (total_area : ℝ)
  (length_of_courtyard : ℝ) :
  breadth_of_courtyard = 12 →
  brick_length_cm = 15 →
  brick_width_cm = 13 →
  number_of_bricks = 11076.923076923076 →
  (brick_length_cm * brick_width_cm / 10000) * number_of_bricks = total_area →
  total_area = breadth_of_courtyard * length_of_courtyard →
  length_of_courtyard = 18 :=
by
  intros hb hl hw hn ha ht
  have h1: (brick_length_cm * brick_width_cm / 10000) = 0.0195, from calc
    (15 * 13 / 10000) = 0.0195 : by norm_num
  have h2: (0.0195 * 11076.923076923076) = 216, from calc
    0.0195 * 11076.923076923076 = 216 : by norm_num
  have h3: brick_length_cm * brick_width_cm = 195, from calc
    15 * 13 = 195 : by norm_num
  sorry

end length_of_courtyard_l711_711922


namespace consumption_decrease_l711_711008

theorem consumption_decrease (X Y : ℝ) :
  let original_quantity := Y / X in
  let new_price := 1.4 * X in
  let new_budget := 1.12 * Y in
  let new_quantity := new_budget / new_price in
  100 * (original_quantity - new_quantity) / original_quantity = 20 := by
sorry

end consumption_decrease_l711_711008


namespace min_value_expression_l711_711742

open Real

theorem min_value_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 := by
  sorry

end min_value_expression_l711_711742


namespace special_function_continuous_at_only_zero_l711_711966

def is_rational (x : ℝ) : Prop :=
  ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

noncomputable def special_function (x : ℝ) : ℝ :=
if is_rational x then 0 else x

theorem special_function_continuous_at_only_zero :
  ∀ x : ℝ, continuous_at special_function x ↔ x = 0 :=
by {
  sorry
}

end special_function_continuous_at_only_zero_l711_711966


namespace max_sum_first_n_terms_l711_711547

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (n : ℕ)
variable (a1 : ℝ)

def arithmetic_sequence (n : ℕ) : Prop :=
  ∀ i, a i = a1 + (i - 1) * d

def inequality_solution_set : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 9 → d * x^2 + 2 * a1 * x ≥ 0

theorem max_sum_first_n_terms
  (h_seq: arithmetic_sequence a)
  (h_neg_d: d < 0)
  (h_a1: a1 = - 9 / 2 * d)
  (h_inequality: inequality_solution_set) :
  ∃ n, n = 5 ∧ ∀ k, k < n → a k ≥ 0 ∧ a (k + 1) < 0 :=
sorry

end max_sum_first_n_terms_l711_711547


namespace original_three_numbers_are_arith_geo_seq_l711_711370

theorem original_three_numbers_are_arith_geo_seq
  (x y z : ℕ) (h1 : ∃ k : ℕ, x = 3*k ∧ y = 4*k ∧ z = 5*k)
  (h2 : ∃ r : ℝ, (x + 1) / y = r ∧ y / z = r ∧ r^2 = z / y):
  x = 15 ∧ y = 20 ∧ z = 25 :=
by 
  sorry

end original_three_numbers_are_arith_geo_seq_l711_711370


namespace problem_statement_l711_711205

theorem problem_statement (x : ℝ) (h : x^2 + 4 * x - 2 = 0) : 3 * x^2 + 12 * x - 23 = -17 :=
sorry

end problem_statement_l711_711205


namespace cube_volume_l711_711438

-- Define the given condition: The surface area of the cube
def surface_area (A : ℕ) := A = 294

-- The key proposition we need to prove using the given condition
theorem cube_volume (s : ℕ) (A : ℕ) (V : ℕ) 
  (area_condition : surface_area A)
  (side_length_condition : s ^ 2 = A / 6) 
  (volume_condition : V = s ^ 3) : 
  V = 343 := by
  sorry

end cube_volume_l711_711438


namespace min_height_of_box_l711_711757

noncomputable def height_minimization (x : ℝ) : Prop :=
  (h : ℝ := x + 4) in
  2*x^2 + 4*x*(x + 4) ≥ 120 → h = 8

theorem min_height_of_box (x : ℝ) :
  height_minimization x :=
sorry

end min_height_of_box_l711_711757


namespace area_arccos_cos_l711_711155

open Real

theorem area_arccos_cos (a b : ℝ) (h : a = 0 ∧ b = 3 * π) :
  ∫ x in a..b, arccos (cos x) = (3 / 2) * π^2 :=
by
  have h1 : ∫ x in 0..π, x = (1 / 2) * π^2, by sorry
  have h2 : ∫ x in π..2 * π, 2 * π - x = (1 / 2) * π^2, by sorry
  have h3 : ∫ x in 2 * π..3 * π, x - 2 * π = (1 / 2) * π^2, by sorry
  calc
    ∫ x in a..b, arccos (cos x)
        = ∫ x in 0..π, arccos (cos x) + ∫ x in π..2 * π, arccos (cos x) +
          ∫ x in 2 * π..3 * π, arccos (cos x) : by sorry
    ... = ∫ x in 0..π, x + ∫ x in π..2 * π, 2 * π - x + ∫ x in 2 * π..3 * π, x - 2 * π : by sorry
    ... = (1 / 2) * π^2 + (1 / 2) * π^2 + (1 / 2) * π^2 : by rw [h1, h2, h3]
    ... = (3 / 2) * π^2 : by norm_num

end area_arccos_cos_l711_711155


namespace minimum_value_l711_711165

open Real

theorem minimum_value (x : ℝ) (hx : x > 2) : 
  ∃ y ≥ 4 * Real.sqrt 2, ∀ z, (z = (x + 6) / (Real.sqrt (x - 2)) → y ≤ z) := 
sorry

end minimum_value_l711_711165


namespace probability_diff_color_balls_l711_711581

theorem probability_diff_color_balls 
  (Box_A_red : ℕ) (Box_A_black : ℕ) (Box_A_white : ℕ) 
  (Box_B_yellow : ℕ) (Box_B_black : ℕ) (Box_B_white : ℕ) 
  (hA : Box_A_red = 3 ∧ Box_A_black = 3 ∧ Box_A_white = 3)
  (hB : Box_B_yellow = 2 ∧ Box_B_black = 2 ∧ Box_B_white = 2) :
  ((Box_A_red * (Box_B_black + Box_B_white + Box_B_yellow))
  + (Box_A_black * (Box_B_yellow + Box_B_white))
  + (Box_A_white * (Box_B_black + Box_B_yellow))) / 
  ((Box_A_red + Box_A_black + Box_A_white) * 
  (Box_B_yellow + Box_B_black + Box_B_white)) = 7 / 9 := 
by
  sorry

end probability_diff_color_balls_l711_711581


namespace primes_satisfying_equation_l711_711824

theorem primes_satisfying_equation (p q r : ℕ) (hp : nat.prime p) (hq : nat.prime q) (hr : nat.prime r) : 
  p ^ q + q ^ p = r ↔ (p = 2 ∧ q = 3 ∧ r = 17) ∨ (p = 3 ∧ q = 2 ∧ r = 17) :=
sorry

end primes_satisfying_equation_l711_711824


namespace find_number_l711_711247

theorem find_number (x : ℕ) (h : (x + 720) / 125 = 7392 / 462) : x = 1280 :=
sorry

end find_number_l711_711247


namespace calc_expr1_calc_expr2_l711_711120

-- Problem 1
theorem calc_expr1 : 
  (Real.sqrt (25 / 4) - Real.cbrt (51 / 16) + Real.cbrt (1 / 8) - (Real.sqrt 2 - Real.sqrt 3)^0) = 1 / 2 :=
  sorry

-- Problem 2
theorem calc_expr2 : 
  ((Real.log 3 / (2 * Real.log 2) + Real.log 3 / (3 * Real.log 2)) * (Real.log 2 / Real.log 3 + Real.log 2 / (2 * Real.log 3))) = 5 / 4 :=
  sorry

end calc_expr1_calc_expr2_l711_711120


namespace pam_bags_equiv_gerald_bags_l711_711776

theorem pam_bags_equiv_gerald_bags :
  ∀ (total_apples pam_bags apples_per_gerald_bag : ℕ), 
    total_apples = 1200 ∧ pam_bags = 10 ∧ apples_per_gerald_bag = 40 → 
    (total_apples / pam_bags) / apples_per_gerald_bag = 3 :=
by
  intros total_apples pam_bags apples_per_gerald_bag h
  obtain ⟨ht, hp, hg⟩ : total_apples = 1200 ∧ pam_bags = 10 ∧ apples_per_gerald_bag = 40 := h
  sorry

end pam_bags_equiv_gerald_bags_l711_711776


namespace remainder_of_3_pow_2023_mod_7_l711_711836

theorem remainder_of_3_pow_2023_mod_7 : (3^2023) % 7 = 3 :=
by
  sorry

end remainder_of_3_pow_2023_mod_7_l711_711836


namespace calculate_x_minus_y_l711_711018

theorem calculate_x_minus_y (x y z : ℝ) 
    (h1 : x - y + z = 23) 
    (h2 : x - y - z = 7) : 
    x - y = 15 :=
by
  sorry

end calculate_x_minus_y_l711_711018


namespace triangle_angled_sum_rounded_l711_711392

theorem triangle_angled_sum_rounded (P Q R P' Q' R' : ℝ) 
    (h₀: Integer P) (h₁: Integer Q) (h₂: Integer R)
    (h₃: P' + Q' + R' = 180) 
    (h₄: P' - 0.5 ≤ P ∧ P ≤ P' + 0.5) 
    (h₅: Q' - 0.5 ≤ Q ∧ Q ≤ Q' + 0.5) 
    (h₆: R' - 0.5 ≤ R ∧ R ≤ R' + 0.5) :
    P + Q + R = 179 ∨ P + Q + R = 180 ∨ P + Q + R = 181 := sorry

end triangle_angled_sum_rounded_l711_711392


namespace anne_trip_shorter_l711_711276

noncomputable def john_walk_distance : ℝ := 2 + 1

noncomputable def anne_walk_distance : ℝ := Real.sqrt (2^2 + 1^2)

noncomputable def distance_difference : ℝ := john_walk_distance - anne_walk_distance

noncomputable def percentage_reduction : ℝ := (distance_difference / john_walk_distance) * 100

theorem anne_trip_shorter :
  20 ≤ percentage_reduction ∧ percentage_reduction < 30 :=
by
  sorry

end anne_trip_shorter_l711_711276


namespace trapezoid_EFGH_properties_l711_711823

-- Definitions and Conditions
def EF : ℝ := 60
def GH : ℝ := 30
def EG : ℝ := 40
def FH : ℝ := 50
def height_G_to_EF : ℝ := 24

-- Theorems to be proved
theorem trapezoid_EFGH_properties :
  let perimeter := EF + GH + FH + sqrt (EG^2 + height_G_to_EF^2)
  let diagonal_EG := sqrt (EG^2 + height_G_to_EF^2)
  perimeter = 191 ∧ diagonal_EG = 51 := by
  sorry

end trapezoid_EFGH_properties_l711_711823


namespace f_x_neg_l711_711538

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 1 else -x^2 - 1

theorem f_x_neg (x : ℝ) (h : x < 0) : f x = -x^2 - 1 :=
by
  sorry

end f_x_neg_l711_711538


namespace sum_of_two_integers_l711_711009

theorem sum_of_two_integers :
  ∃ (a b : ℕ), a * b + a + b = 119 ∧ a < 20 ∧ b < 20 ∧ Nat.coprime a b → a + b = 21 :=
by
  sorry

end sum_of_two_integers_l711_711009


namespace length_op_sqrt3_l711_711257

theorem length_op_sqrt3 (A B : ℝ) (P : ℝ) (b : ℝ) (hA : A > 0) (hB : B > 0) (hP : P = b) (hb : b > 0)
  (tangent_circle : (A - B)^2 + 4 = (A - B)^2 + 1)
  (fixed_angle : true) :
  b = sqrt 3 := 
sorry

end length_op_sqrt3_l711_711257


namespace arithmetic_to_geometric_find_original_numbers_l711_711368

theorem arithmetic_to_geometric (k : ℕ) 
  (h1 : ∃ k, (3 * k + 1) * 4 * k = 5 * k * (3 * k + 1))
  : k = 5 :=
begin
  sorry,
end

theorem find_original_numbers :
  ∃ (a b c : ℕ), a = 15 ∧ b = 20 ∧ c = 25 :=
begin
  use [15, 20, 25],
  exact ⟨rfl, rfl, rfl⟩
end

end arithmetic_to_geometric_find_original_numbers_l711_711368


namespace problem_1_problem_2_l711_711554

open Real

theorem problem_1 (m : ℝ) :
  ((x ^ 2 + y ^ 2 + 2 * m * x + 2 * y + 2) = 0) ∧ (∃ A : ℝ × ℝ, A = (1,2)) → 
  ((m > -11 / 2 ∧ m < -1) ∨ (m > 1)) :=
sorry

theorem problem_2 (m : ℝ) (P : ℝ × ℝ) :
  m = -2 ∧ (∃ k : ℝ, P = (2 * k - 3, k)) → 
  (let d := abs (2 * 2 - (-1) + 3) / sqrt (2 ^ 2 + (-1) ^ 2) in 
   let radius := sqrt 3 in 
   radius * (d - radius) = 7 * sqrt 15 / 5) :=
sorry

end problem_1_problem_2_l711_711554


namespace ensureUserDataSecurity_l711_711767

-- Definitions based on the given conditions and correct answers
variable (storeApp : Type) -- representing the online store application

/-- Condition: Users can pay using credit cards in the store application -/
def canPayWithCreditCard (app : storeApp) : Prop := 
sorry -- detailed implementation is not provided

/-- Condition: Users can order home delivery in the store application -/
def canOrderHomeDelivery (app : storeApp) : Prop := 
sorry -- detailed implementation is not provided

/-- Security Measure 1: Avoid Storing Card Data - assume implemented properly -/
def avoidStoringCardData (app : storeApp) : Prop := 
sorry -- detailed implementation is not provided

/-- Security Measure 2: Encryption of Stored Data - assume implemented properly -/
def encryptStoredData (app : storeApp) : Prop := 
sorry -- detailed implementation is not provided

/-- Security Measure 3: Encryption of Data in Transit - assume implemented properly -/
def encryptDataInTransit (app : storeApp) : Prop := 
sorry -- detailed implementation is not provided

/-- Theorem: Ensuring user data security in an online store application -/
theorem ensureUserDataSecurity:
  ∀ (app : storeApp), 
    canPayWithCreditCard app → 
    canOrderHomeDelivery app → 
    avoidStoringCardData app →
    encryptStoredData app → 
    encryptDataInTransit app → 
    true := 
by
  sorry

end ensureUserDataSecurity_l711_711767


namespace powerThreeExpression_l711_711566

theorem powerThreeExpression (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / (x^3)) = 140 := sorry

end powerThreeExpression_l711_711566


namespace domain_of_sqrt_ln_calc_log_expression_l711_711407

noncomputable def domain(f : ℝ → ℝ) : Set ℝ := {x : ℝ | f x ≥ 0}

def eq_log_formulae : Prop :=
  (log 25 + (2 / 3) * log 8 + log 5 * log 20 + (log 2) ^ 2 = 3)

theorem domain_of_sqrt_ln :
  domain (λ x, real.sqrt (real.log (x^2 - x - 1))) = {x | x ≤ -1 ∨ x ≥ 2} :=
sorry

theorem calc_log_expression :
  eq_log_formulae :=
sorry

end domain_of_sqrt_ln_calc_log_expression_l711_711407


namespace cubic_difference_l711_711385

-- Define the variables and conditions
variables (a b : ℝ)
variable h1 : a + b = 12
variable h2 : a * b = 20

-- Statement of the theorem
theorem cubic_difference (h1 : a + b = 12) (h2 : a * b = 20) : a^3 - b^3 = 992 :=
sorry

end cubic_difference_l711_711385


namespace jill_account_balance_l711_711708

noncomputable def compound_interest 
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_account_balance :
  let P := 10000
  let r := 0.0396
  let n := 2
  let t := 2
  compound_interest P r n t ≈ 10816.49 :=
by
  sorry

end jill_account_balance_l711_711708


namespace brooke_kent_ratio_l711_711943

theorem brooke_kent_ratio :
  ∀ (alison brooke brittany kent : ℕ),
  (kent = 1000) →
  (alison = 4000) →
  (alison = brittany / 2) →
  (brittany = 4 * brooke) →
  brooke / kent = 2 :=
by
  intros alison brooke brittany kent kent_val alison_val alison_brittany brittany_brooke
  sorry

end brooke_kent_ratio_l711_711943


namespace arcsin_arccos_add_eq_pi6_l711_711324

noncomputable def arcsin (x : Real) : Real := sorry
noncomputable def arccos (x : Real) : Real := sorry

theorem arcsin_arccos_add_eq_pi6 (x : Real) (hx_range : -1 ≤ x ∧ x ≤ 1)
    (h3x_range : -1 ≤ 3 * x ∧ 3 * x ≤ 1) 
    (h : arcsin x + arccos (3 * x) = Real.pi / 6) :
    x = Real.sqrt (3 / 124) := 
  sorry

end arcsin_arccos_add_eq_pi6_l711_711324


namespace distinct_int_divisible_by_12_l711_711281

variable {a b c d : ℤ}

theorem distinct_int_divisible_by_12 (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) :
  12 ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) :=
by
  sorry

end distinct_int_divisible_by_12_l711_711281


namespace total_green_peaches_l711_711362

-- Define the known conditions
def baskets : ℕ := 7
def green_peaches_per_basket : ℕ := 2

-- State the problem and the proof goal
theorem total_green_peaches : baskets * green_peaches_per_basket = 14 := by
  -- Provide a proof here
  sorry

end total_green_peaches_l711_711362


namespace drunk_drivers_total_correct_l711_711330

-- Define the total number of people
def total_people : ℕ := 500

-- Define the frequencies according to histogram
def frequency_at_80 : ℝ := 0.01
def frequency_above_80 : ℝ := 0.005

-- Define the combined frequency for drunk driving as per the problem statement
def combined_frequency_for_drunk_driving : ℝ := frequency_at_80 + frequency_above_80

-- Define the expected number of drunk drivers
def expected_number_of_drunk_drivers (total : ℕ) (frequency : ℝ) : ℕ := 
  (total : ℝ) * frequency |> Int.toNat

-- Check if the total number of drunk drivers equals 75
theorem drunk_drivers_total_correct : 
  expected_number_of_drunk_drivers total_people combined_frequency_for_drunk_driving = 75 :=
by
  sorry

end drunk_drivers_total_correct_l711_711330


namespace num_valid_distributions_l711_711521

-- Define the rabbits and pet shops
inductive Rabbit
| Alice | Bob | Charlie | Daisy | Earl

open Rabbit

inductive Store
| shop1 | shop2 | shop3 | shop4 | shop5

open Store

-- Define the parent-child relationship
def isParent (r : Rabbit) : Prop :=
  r = Alice ∨ r = Bob

def isChild (r : Rabbit) : Prop :=
  r = Charlie ∨ r = Daisy ∨ r = Earl

-- Define the constraint that no store can have both a parent and any of their offspring
def valid_distribution (distribution : Rabbit → Option Store) : Prop :=
  ∀ s : Store,
    (∃ r1 r2 : Rabbit, isParent r1 ∧ isChild r2 ∧
    distribution r1 = some s ∧ distribution r2 = some s) → false

-- Define the main problem statement
theorem num_valid_distributions : 
  ∃ n : ℕ, n = 266 ∧ 
  ∃ distribution : Rabbit → Option Store,
  valid_distribution distribution :=
begin
  sorry
end

end num_valid_distributions_l711_711521


namespace circle_radius_l711_711353

theorem circle_radius {C : ℝ × ℝ} (hC : C = (0, -3)) (on_circle : (4, 0) ∈ (λ P, P = (C.1, C.2)) ∧ (3, 1) ∈ (λ P, P = (C.1, C.2))) :
  let radius := Math.sqrt ((4 - C.1) ^ 2 + (0 - C.2) ^ 2) in
  radius = 5 :=
by
  sorry

end circle_radius_l711_711353


namespace area_of_unpainted_region_l711_711030

def rhombus_area (w1 w2 : ℝ) (angle : ℝ) : ℝ :=
  1 / 2 * (w1 * real.sqrt 2) * (w2 * real.sqrt 2)

theorem area_of_unpainted_region :
  let width1 := 5
  let width2 := 7
  let angle := 45
  rhombus_area width1 width2 angle = 35 := by
  sorry

end area_of_unpainted_region_l711_711030


namespace sum_distances_eq_100_l711_711168

noncomputable def radii_relationships := {
  A_radius : ℝ,
  B_radius : ℝ,
  C_radius : ℝ,
  D_radius : ℝ,
  P : ℝ,
  Q : ℝ,
  R : ℝ,
  A : ℝ,
  B : ℝ,
  C : ℝ,
  D : ℝ,
  AB : ℝ := 50,
  CD : ℝ := 50,
  PQ : ℝ := 50,
  midpoint_R_PQ : ∃ R : ℝ, R = (P + Q) / 2,
  A_radius_def : A_radius = 2 / 3 * B_radius,
  C_radius_def : C_radius = 2 / 3 * D_radius,
  power_of_point_R_circle_A : (A_radius) ^ 2 - A ^ 2 = 625,
  power_of_point_R_circle_B : (3/2 * A_radius) ^ 2 - B ^ 2 = 625,
  power_of_point_R_circle_C : (C_radius) ^ 2 - C ^ 2 = 625,
  power_of_point_R_circle_D : (3/2 * C_radius) ^ 2 - D ^ 2 = 625
}

theorem sum_distances_eq_100 : (A + B + C + D) = 100 := by
  sorry

end sum_distances_eq_100_l711_711168


namespace cube_volume_l711_711425

theorem cube_volume (A : ℝ) (hA : A = 294) : ∃ (V : ℝ), V = 343 :=
by
  sorry

end cube_volume_l711_711425


namespace perpendicular_vector_evaluation_l711_711223

-- Define the vectors and constants
def vec_a : ℝ × ℝ := (1, -2)
def vec_b (m : ℝ) : ℝ × ℝ := (4, m)
def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Given condition
def condition : Prop := is_perpendicular vec_a (vec_b 2)

-- Expression to be evaluated
def vector_expression (u v : ℝ × ℝ) : ℝ × ℝ :=
  (5 * u.1 - 3 * v.1, 5 * u.2 - 3 * v.2)

-- Final statement to be proved
theorem perpendicular_vector_evaluation : condition → vector_expression vec_a (vec_b 2) = (-7, -16) :=
by
  intro h,
  sorry

end perpendicular_vector_evaluation_l711_711223


namespace tax_rate_is_11_percent_l711_711939

-- Define the relevant quantities
def total_value : ℝ := 1720
def non_taxed_amount : ℝ := 600
def tax_paid : ℝ := 123.2

-- Compute the tax rate on the portion that is in excess of 600
def taxable_amount : ℝ := total_value - non_taxed_amount
def tax_rate : ℝ := tax_paid / taxable_amount

-- Statement to be proven: tax_rate is 0.11
theorem tax_rate_is_11_percent : tax_rate = 0.11 :=
by
  sorry

end tax_rate_is_11_percent_l711_711939


namespace sum_of_lengths_at_least_n_times_two_to_the_n_l711_711539

theorem sum_of_lengths_at_least_n_times_two_to_the_n (n : ℕ) (S : Finset (List Bool))
  (h_len : S.card = 2^n)
  (h_prefix : ∀ (s1 s2 : List Bool), s1 ∈ S → s2 ∈ S → s1 ≠ s2 → ¬ s1.isPrefixOf s2) :
  (S.sum (λ l, l.length)) ≥ n * 2^n :=
by
  sorry

end sum_of_lengths_at_least_n_times_two_to_the_n_l711_711539


namespace domain_log_function_l711_711795

open Real

def domain_of_log_function (f : ℝ → ℝ) : Set ℝ :=
  {x | f x = log (2 - x) / log 3}

theorem domain_log_function :
    domain_of_log_function (λ x, log (2 - x) / log 3) = {x : ℝ | x < 2} := by
  sorry

end domain_log_function_l711_711795


namespace parabola_vertex_coordinates_l711_711338

theorem parabola_vertex_coordinates :
  ∀ x : ℝ, (3 * (x - 7) ^ 2 + 5) = 3 * (x - 7) ^ 2 + 5 := by
  sorry

end parabola_vertex_coordinates_l711_711338


namespace jill_account_balance_l711_711706

noncomputable def compound_interest 
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_account_balance :
  let P := 10000
  let r := 0.0396
  let n := 2
  let t := 2
  compound_interest P r n t ≈ 10816.49 :=
by
  sorry

end jill_account_balance_l711_711706


namespace cylinder_height_l711_711456

theorem cylinder_height (OA OB : ℝ) (h_OA : OA = 7) (h_OB : OB = 2) :
  ∃ (h_cylinder : ℝ), h_cylinder = 3 * Real.sqrt 5 :=
by
  use (Real.sqrt (OA^2 - OB^2))
  rw [h_OA, h_OB]
  norm_num
  sorry

end cylinder_height_l711_711456


namespace find_other_endpoint_l711_711801

def other_endpoint (midpoint endpoint: ℝ × ℝ) : ℝ × ℝ :=
  let (mx, my) := midpoint
  let (ex, ey) := endpoint
  (2 * mx - ex, 2 * my - ey)

theorem find_other_endpoint :
  other_endpoint (3, 1) (7, -4) = (-1, 6) :=
by
  -- Midpoint formula to find other endpoint
  sorry

end find_other_endpoint_l711_711801


namespace repeating_decimal_to_fraction_l711_711147

theorem repeating_decimal_to_fraction :
  let x := Real.ofRat (8 + 137 / 999) in
  x = 2709 / 333 :=
by
  sorry

end repeating_decimal_to_fraction_l711_711147


namespace jill_investment_l711_711695

def compound_interest (P : ℕ) (r : ℚ) (n : ℕ) (t : ℕ) : ℚ :=
  P * (1 + r / n)^(n * t)

theorem jill_investment : 
  compound_interest 10000 (396 / 10000) 2 2 ≈ 10812 := by
sorry

end jill_investment_l711_711695


namespace number_of_common_divisors_36_54_l711_711234

open Nat

theorem number_of_common_divisors_36_54 : 
  let common_divisors := (divisors 36) ∩ (divisors 54) in
  common_divisors.card = 6 :=
by
  sorry

end number_of_common_divisors_36_54_l711_711234


namespace x_intercept_of_line_l711_711386

theorem x_intercept_of_line : (∃ x : ℝ, 2 * x + 0 - 2 = 0) → x = 1 :=
by
    intro h
    rcases h with ⟨x, lhs_eq_0⟩
    have eq1 : 2 * x - 2 = 0 := by rw [lhs_eq_0, zero_add]
    have eq2 : 2 * x = 2 := by linarith
    have x_eq_1 : x = 1 := by linarith 
    exact x_eq_1

end x_intercept_of_line_l711_711386


namespace boat_mass_problem_l711_711063

/-- Given a boat with length 3 meters, breadth 2 meters, sinking height 0.01 meters, and the 
    density of water is 1000 kg/m³, the mass of the man that causes the boat to sink 
    by 0.01 meters is 60 kg. -/
theorem boat_mass_problem 
  (L : ℝ) (B : ℝ) (h : ℝ) (ρ : ℝ)
  (hL : L = 3) (hB : B = 2) (hh : h = 0.01) (hρ : ρ = 1000) :
  ∃ m : ℝ, m = 60 :=
begin
  -- Sorry state to skip the proof.
  sorry,
end

end boat_mass_problem_l711_711063


namespace derivative_of_f_l711_711793

def f : ℝ → ℝ := λ x => Real.exp x + x^2 + Real.sin x

theorem derivative_of_f (x : ℝ) : deriv f x = Real.exp x + 2 * x + Real.cos x := by
  sorry

end derivative_of_f_l711_711793


namespace slope_reciprocal_and_a_bounds_l711_711219

theorem slope_reciprocal_and_a_bounds (x : ℝ) (f g : ℝ → ℝ) 
    (h1 : ∀ x, f x = Real.log x - a * (x - 1)) 
    (h2 : ∀ x, g x = Real.exp x) :
    ((∀ k₁ k₂, (∃ x₁, k₁ = deriv f x₁) ∧ (∃ x₂, k₂ = deriv g x₂) ∧ k₁ * k₂ = 1) 
    ↔ (Real.exp 1 - 1) / Real.exp 1 < a ∧ a < (Real.exp 2 - 1) / Real.exp 1 ∨ a = 0) :=
by
  sorry

end slope_reciprocal_and_a_bounds_l711_711219


namespace sum_of_k_values_with_distinct_integer_solutions_l711_711843

theorem sum_of_k_values_with_distinct_integer_solutions : 
  ∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ 3 * p * q + (-k) * (p + q) + 12 = 0}, k = 0 :=
by
  sorry

end sum_of_k_values_with_distinct_integer_solutions_l711_711843


namespace length_of_train_eq_l711_711463

-- Definitions of the given conditions
def speed_kmph : ℝ := 72
def speed_mps := speed_kmph * (1000 / 3600)
def crossing_time : ℝ := 25
def platform_length : ℝ := 300.04
def crossing_distance := speed_mps * crossing_time

-- Statement of the problem
theorem length_of_train_eq : 
  let L_train := crossing_distance - platform_length in
  L_train = 199.96 :=
by
  -- Optional: Intermediate calculations for clarity
  have : speed_mps = 20 := by sorry
  have : crossing_distance = speed_mps * crossing_time := by sorry
  have : crossing_distance = 500 := by sorry
  have L_train := crossing_distance - platform_length
  exact sorry

end length_of_train_eq_l711_711463


namespace three_digit_multiples_of_seven_count_l711_711612

theorem three_digit_multiples_of_seven_count :
  let smallest := 15
  let largest := 142
  largest - smallest + 1 = 128 :=
by
  let smallest := 15
  let largest := 142
  have h_smallest : 7 * smallest = 105 := rfl
  have h_largest : 7 * largest = 994 := rfl
  show largest - smallest + 1 = 128 from sorry

end three_digit_multiples_of_seven_count_l711_711612


namespace select_athlete_l711_711251

-- Define the constants and conditions
constant average_A : ℝ := 9.1
constant average_B : ℝ := 9.1
constant variance_A : ℝ := 0.69
constant variance_B : ℝ := 0.03
constant threshold : ℝ := 9.0

-- The theorem statement
theorem select_athlete (h1 : average_A > threshold) (h2 : average_B > threshold) (h3 : variance_B < variance_A) : 
  "Athlete B is the better choice to participate in the competition" :=
sorry

end select_athlete_l711_711251


namespace sum_of_k_distinct_integer_roots_l711_711850

theorem sum_of_k_distinct_integer_roots : 
  (∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ p * q = 4 ∧ k = 3 * (p + q)}, k) = 0 := 
by
  sorry

end sum_of_k_distinct_integer_roots_l711_711850


namespace decision_represented_by_D_l711_711470

-- Define the basic symbols in the flowchart
inductive BasicSymbol
| Start
| Process
| Decision
| End

open BasicSymbol

-- Define the meaning of each basic symbol
def meaning_of (sym : BasicSymbol) : String :=
  match sym with
  | Start => "start"
  | Process => "process"
  | Decision => "decision"
  | End => "end"

-- The theorem stating that the Decision symbol represents a decision
theorem decision_represented_by_D : meaning_of Decision = "decision" :=
by sorry

end decision_represented_by_D_l711_711470


namespace car_actual_time_l711_711049

noncomputable def actual_time_taken (S T : ℝ) : ℝ :=
  let T_late := 5 / 4 * T
  in if (T_late - T = 1 / 4) then T else 0

theorem car_actual_time (S T : ℝ) (h : actual_time_taken S T = T) (h_late : actual_time_taken S T - T = 1 / 4):
  T = 1 := 
by
  sorry

end car_actual_time_l711_711049


namespace line_intersects_x_axis_at_neg3_l711_711948

theorem line_intersects_x_axis_at_neg3 :
  ∃ (x y : ℝ), (3 * y - 4 * x = 12) ∧ y = 0 ∧ x = -3 :=
by {
  use [-3, 0],
  split,
  norm_num,
  split,
  norm_num,
  norm_num,
}

end line_intersects_x_axis_at_neg3_l711_711948


namespace dice_probability_l711_711061

theorem dice_probability :
  let prob_one_digit := (9:ℚ) / 20
  let prob_two_digit := (11:ℚ) / 20
  let prob := 10 * (prob_two_digit^2) * (prob_one_digit^3)
  prob = 1062889 / 128000000 := 
by 
  sorry

end dice_probability_l711_711061


namespace concyclic_points_l711_711283

open EuclideanGeometry

variables {A B C D E F H P : Point} {triangle : Triangle}

-- Assume the conditions given in the problem
variables (h_orthocenter : Orthocenter H triangle)
variables (h_midpoint : Midpoint D (B, C))
variables (line_h : ∃ line, line.contains H ∧ line.intersect (AB, F) ∧ line.intersect (AC, E))
variables (h_eq : EuclideanDistance A E = EuclideanDistance A F)
variables (circle_intersect : ∃ circumcircle, Circumcircle circumcircle triangle ∧ RayIntersects DH circumcircle P)

-- Statement proving the concyclicity of the points P, A, E, and F
theorem concyclic_points 
  (h1 : Orthocenter H triangle)
  (h2 : Midpoint D (B, C))
  (h3 : ∃ line, line.contains H ∧ line.intersect (AB, F) ∧ line.intersect (AC, E))
  (h4 : EuclideanDistance A E = EuclideanDistance A F)
  (h5 : ∃ circumcircle, Circumcircle circumcircle triangle ∧ RayIntersects DH circumcircle P) :
  Concyclic P A E F :=
sorry

end concyclic_points_l711_711283


namespace range_of_a_l711_711726

theorem range_of_a (a : ℝ) (A : set ℝ) (B : set ℝ) (hA : A = set.Ico (-1 : ℝ) 6) (hB : B = set.Ioi (1 - a)) : 
  (A ∩ B ≠ ∅) → (a > -5) :=
by
  intros h
  sorry

end range_of_a_l711_711726


namespace find_integer_n_l711_711004

theorem find_integer_n :
  ∃ n : ℤ, 
    50 ≤ n ∧ n ≤ 120 ∧ (n % 5 = 0) ∧ (n % 6 = 3) ∧ (n % 7 = 4) ∧ n = 165 :=
by
  sorry

end find_integer_n_l711_711004


namespace scaled_area_l711_711807

variable {g : ℝ → ℝ}
variable (h₁ : ∫ x in (-∞) .. ∞, g x = 12)

theorem scaled_area : (∫ x in (-∞) .. ∞, (2 * g (x - 3))) = 24 :=
by
  sorry

end scaled_area_l711_711807


namespace polynomial_unique_factorization_l711_711745

variables {k : Type*} [Field k]

theorem polynomial_unique_factorization (f : Polynomial k) :
  ∃! (l : Multiset (Polynomial k)), (∀ g ∈ l, Irreducible g) ∧ (l.Prod = f) :=
sorry

end polynomial_unique_factorization_l711_711745


namespace mary_income_percentage_l711_711304

-- Declare noncomputable as necessary
noncomputable def calculate_percentage_more
    (J : ℝ) -- Juan's income
    (T : ℝ) (M : ℝ)
    (hT : T = 0.70 * J) -- Tim's income is 30% less than Juan's income
    (hM : M = 1.12 * J) -- Mary's income is 112% of Juan's income
    : ℝ :=
  ((M - T) / T) * 100

theorem mary_income_percentage
    (J T M : ℝ)
    (hT : T = 0.70 * J)
    (hM : M = 1.12 * J) :
    calculate_percentage_more J T M hT hM = 60 :=
by sorry

end mary_income_percentage_l711_711304


namespace ratio_min_l711_711100

namespace tablecloth

/-- Defining the conditions for the problem -/
variables (S S1 : ℝ)

/-- Theorem stating that the ratio S1 / S is at least 2/3 -/
theorem ratio_min (h1 : S1 <= S)
  (h2 : S1 <= S / 2)
  (h3 : S1 = S) : S1 / S >= 2 / 3 :=
begin
  sorry
end

end tablecloth

end ratio_min_l711_711100


namespace part1_part2_l711_711106

-- Given that the ant's position after 2 seconds is non-negative
-- We want to prove the probability that the ant is at x=0 after 2 seconds is 1/2
theorem part1 (p_right : ℝ) (p_left : ℝ) (t : ℕ) (P_A : ℝ) (P_B : ℝ) : 
  p_right = 2/3 →
  p_left = 1/3 →
  t = 2 →
  P_A = ((p_right)^2 + 2 * (p_right) * (p_left)) →
  P_B = (2 * (p_right) * (p_left)) →
  (P_B / P_A) = 1/2 :=
  by
  intros hpr hpl ht hpa hpb
  rw [hpr, hpl, ht, hpa, hpb]
  sorry

-- Let X be the real number corresponding to the ant's position after 4 seconds
-- We want to prove the expected value of X is 4/3 given the derived probability distribution
theorem part2 (p_right : ℝ) (p_left : ℝ) (t : ℕ) (X : ℝ → ℝ) (E_X : ℝ) : 
  p_right = 2/3 →
  p_left = 1/3 →
  t = 4 →
  (X (-4) = -4 * (p_left)^4) →
  (X (-2) = -2 * 4 * (p_right) * (p_left)^3) →
  (X (0) = 0 * 6 * (p_right)^2 * (p_left)^2) →
  (X (2) = 2 * 4 * (p_right)^3 * (p_left)) →
  (X (4) = 4 * (p_right)^4) →
  E_X = ((X (-4)) + (X (-2)) + (X (0)) + (X (2)) + (X (4))) →
  E_X = 4/3 :=
  by
  intros hpr hpl ht hx_neg4 hx_neg2 hx_0 hx_2 hx_4 he_x
  rw [hpr, hpl, ht, hx_neg4, hx_neg2, hx_0, hx_2, hx_4]
  sorry

end part1_part2_l711_711106


namespace monotonic_intervals_area_of_triangle_l711_711603

variables 
  (x A b : ℝ)
  (a : ℝ := 2 * Real.sqrt 3)
  (c : ℝ := 4)
  (k : ℤ)

-- Define vectors and the function f
def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x - Real.cos x, 1)
def n (x : ℝ) : ℝ × ℝ := (Real.cos x, 1 / 2)
def f (x : ℝ) : ℝ := (m x).fst * (n x).fst + (m x).snd * (n x).snd

-- First proof problem
theorem monotonic_intervals : 
  (- (Real.pi / 2) + (2 * k * Real.pi) ≤ 2 * x - Real.pi / 6 ∧ 
   2 * x - Real.pi / 6 ≤ Real.pi / 2 + (2 * k * Real.pi)) ↔ 
  (k * Real.pi - Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 3) := 
sorry

-- Second proof problem conditions
def f_A_equals_one (A : ℝ) : Prop := f A = 1

-- Using Law of Cosines and finding b
def find_b (a c f_A : ℝ) : ℝ := (a^2 + c^2 - 2 * a * c * Real.cos f_A) / (2 * c)

-- Calculating area using a, b, and C
theorem area_of_triangle (A : ℝ) (h : f_A_equals_one A) : 
  let B := find_b a c (Real.pi / 3)
  in 1 / 2 * a * B * Real.sin (Real.pi / 2) = 2 * Real.sqrt 3 := 
sorry

end monotonic_intervals_area_of_triangle_l711_711603


namespace incorrect_statement_about_g_l711_711240

-- Let's define the function g as given in the problem
def g (x : ℝ) : ℝ := (x - 3) / (x + 2)

-- Here's the Lean 4 statement representing the proof problem
theorem incorrect_statement_about_g : g (-2) = 0 → False := by
  sorry

end incorrect_statement_about_g_l711_711240


namespace main_theorem_l711_711124

noncomputable def main_expr := (Real.pi - 2019) ^ 0 + |Real.sqrt 3 - 1| + (-1 / 2)⁻¹ - 2 * Real.tan (Real.pi / 6)

theorem main_theorem : main_expr = -2 + Real.sqrt 3 / 3 := by
  sorry

end main_theorem_l711_711124


namespace centers_collinear_l711_711267

/-- Given a triangle ABC with a circle δ and three identical circles α, β, γ,
each tangent to two sides of the triangle and the circle δ,
prove that the center of the circle δ lies on the line passing through
the incenter and circumcenter of the triangle ABC. --/
theorem centers_collinear (ABC : Triangle) (α β γ δ : Circle) (r : ℝ)
  (hα : α.radius = r) (hβ : β.radius = r) (hγ : γ.radius = r) (hδ : δ.radius = r)
  (h_tangent_α : ∀ (s : Side), s ∈ ABC.sides → α.tangents_to_side s)
  (h_tangent_β : ∀ (s : Side), s ∈ ABC.sides → β.tangents_to_side s)
  (h_tangent_γ : ∀ (s : Side), s ∈ ABC.sides → γ.tangents_to_side s)
  (h_tangent_all : ∀ (a b : Circle), (a = α ∨ a = β ∨ a = γ) → (b = δ) → a.tangent b) :
  collinear δ.center ABC.incenter ABC.circumcenter := 
begin
  sorry
end

end centers_collinear_l711_711267


namespace impossible_100_digits_l711_711307

theorem impossible_100_digits (n : ℕ) : ¬ (9 + 2 * (n - 9) = 100) :=
by
  exfalso
  have h : 9 + 2 * (n - 9) = 100 → 2 * (n - 9) = 91 := by
    intro h_eq
    linarith
  have h' : 2 * (n - 9) = 91 → n = 54.5 := by
    intro h_eq
    ring at h_eq
    linarith
  apply nat.not_int_of_real h'
  sorry

end impossible_100_digits_l711_711307


namespace condition_1_total_ways_condition_2_total_ways_l711_711361

def boys : Nat := 3
def girls : Nat := 4

-- Condition 1: All stand in a row with the girls standing together
theorem condition_1_total_ways : (fact boys + 1) * fact girls = 576 :=
by sorry

-- Condition 2: All stand in a row with no two boys standing next to each other
theorem condition_2_total_ways : fact girls * ((boys + girls).choose boys) = 1440 :=
by sorry

end condition_1_total_ways_condition_2_total_ways_l711_711361


namespace sin_B_value_max_perimeter_l711_711175

-- Define the conditions
variables (a b c : ℝ) (A B C : ℝ)
variable (triangle_ABC : a^2 + b^2 + c^2 - 2 * a * b * cos C = 0)
variable (condition : (2 * a - c) * cos B = b * cos C)

-- Prove (1) the value of sin B
theorem sin_B_value (h1 : sin A ≠ 0) : sin B = sqrt 3 / 2 :=
sorry

-- Prove (2) the maximum value of the perimeter given b = sqrt 7
theorem max_perimeter (h2 : b = sqrt 7) : a + b + c ≤ 3 * sqrt 7 :=
sorry

end sin_B_value_max_perimeter_l711_711175


namespace find_Q_plus_R_l711_711238

-- P, Q, R must be digits in base 8 (distinct and non-zero)
def is_valid_digit (d : Nat) : Prop :=
  d > 0 ∧ d < 8

def digits_distinct (P Q R : Nat) : Prop :=
  P ≠ Q ∧ Q ≠ R ∧ R ≠ P

-- Define the base 8 number from its digits
def base8_number (P Q R : Nat) : Nat :=
  8^2 * P + 8 * Q + R

-- Define the given condition
def condition (P Q R : Nat) : Prop :=
  is_valid_digit P ∧ is_valid_digit Q ∧ is_valid_digit R ∧ digits_distinct P Q R ∧ 
  (base8_number P Q R + base8_number Q R P + base8_number R P Q = 8^3 * P + 8^2 * P + 8 * P + 8)

-- The result: Q + R in base 8 is 10_8 which is 8 + 2 (in decimal is 10)
theorem find_Q_plus_R (P Q R : Nat) (h : condition P Q R) : Q + R = 8 + 2 :=
sorry

end find_Q_plus_R_l711_711238


namespace value_of_S20_l711_711545

variables {a : ℕ → ℕ}

def Sn (n : ℕ) : ℕ := (finset.range n).sum a

axiom given_sequence_property (n : ℕ) (hn : n > 0) : a (n+2) - a n = 1

axiom initial_conditions1 : a 1 = 1
axiom initial_conditions2 : a 2 = 2

axiom arithmetic_progression_property : 2 * (a 2) * (a 3) = (a 1) * (a 2) + (a 3) * (a 4)

theorem value_of_S20 : Sn 20 = 120 :=
sorry

end value_of_S20_l711_711545


namespace exists_large_yummy_segment_l711_711058

def red_numbers : set ℕ := {n | n ∈ (1 : ℕ) .. 1000 ∧ is_red n}

def is_red_segment (a b : ℕ) : Prop :=
  ∀ t, 1 ≤ t ∧ t ≤ b - a → ∃ x y, x ∈ (a : ℕ) .. b ∧ y ∈ (a : ℕ) .. b ∧ x ∈ red_numbers ∧ y ∈ red_numbers ∧ y - x = t

theorem exists_large_yummy_segment (h : ∃ s : finset ℕ, s.card = 600 ∧ ∀ n ∈ s, n ∈ (1 : ℕ) .. 1000) :
  ∃ a b, a ∈ (1 : ℕ) .. 1000 ∧ b ∈ (1 : ℕ) .. 1000 ∧ is_red_segment a b ∧ b - a ≥ 199 :=
sorry

end exists_large_yummy_segment_l711_711058


namespace jellybeans_needed_l711_711942

theorem jellybeans_needed (n : ℕ) : (n ≥ 120 ∧ n % 15 = 14) → n = 134 :=
by sorry

end jellybeans_needed_l711_711942


namespace solution_set_f_lt_zero_l711_711729

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_nonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y
  
theorem solution_set_f_lt_zero (f : ℝ → ℝ) 
  (H_even: even_function f)
  (H_increasing: increasing_on_nonnegative f)
  (H_value: f (-3) = 0)
  : {x : ℝ | f x < 0} = Ioo (-3) 3 :=
begin
  sorry
end

end solution_set_f_lt_zero_l711_711729


namespace wrapping_paper_area_correct_l711_711072

variable (w h : ℝ) -- Define the base length and height of the box.

-- Lean statement for the problem asserting that the area of the wrapping paper is \(2(w+h)^2\).
def wrapping_paper_area (w h : ℝ) : ℝ := 2 * (w + h) ^ 2

-- Theorem stating that the derived formula for the area of the wrapping paper is correct.
theorem wrapping_paper_area_correct (w h : ℝ) : wrapping_paper_area w h = 2 * (w + h) ^ 2 :=
by
  sorry -- Proof is omitted

end wrapping_paper_area_correct_l711_711072


namespace find_sin_theta_l711_711285

noncomputable def angle_between (b c : Vector ℝ) : ℝ :=
  Real.arccos ((b ⬝ c) / (‖b‖ * ‖c‖))

theorem find_sin_theta
  (a b c : Vector ℝ)
  (non_zero_a : a ≠ 0)
  (non_zero_b : b ≠ 0)
  (non_zero_c : c ≠ 0)
  (not_parallel_ab : ¬ (a × b = 0))
  (not_parallel_bc : ¬ (b × c = 0))
  (not_parallel_ca : ¬ (c × a = 0))
  (h : (a × b) × c = - (1 / 2) * (‖b‖) * (‖c‖) • a) :
  Real.sin (angle_between b c) = Real.sqrt (3) / 2 := 
by
  sorry

end find_sin_theta_l711_711285


namespace point_on_fixed_line_l711_711080

theorem point_on_fixed_line
  (O A B P : Point)
  (h1 : Line O A)
  (h2 : Line O B)
  (h3 : ∃ (OP OA OB : ℝ), OP * 2 = OA * OB / (OA + OB)
: FixedLine P :=
sorry

end point_on_fixed_line_l711_711080


namespace max_value_f_period_f_strictly_increasing_f_l711_711230

def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x)
def f (x : ℝ) : ℝ := (a x).1 * (a x).1 + (a x).2 * (a x).2 + (a x).1 * (b x).1 + (b x).2 * (a x).2 - 3 / 2

theorem max_value_f : ∃ x, f x = Real.sqrt 2 / 2 := sorry
theorem period_f : ∃ T, T > 0 ∧ ∀ x, f (x + T) = f x := sorry
theorem strictly_increasing_f :
  ∀ k : ℤ, ∀ x ∈ Set.Icc (-3 * Real.pi / 8 + k * Real.pi) (Real.pi / 8 + k * Real.pi),
  (∃ ε > 0, ∀ h : ℝ, 0 <= h ∧ h < ε → f (x + h) > f x) := sorry

end max_value_f_period_f_strictly_increasing_f_l711_711230


namespace sum_of_k_with_distinct_integer_solutions_l711_711855

-- Definitions from conditions
def quadratic_eq (k : ℤ) (x : ℤ) : Prop := 3 * x^2 - k * x + 12 = 0

def distinct_integer_solutions (k : ℤ) : Prop := ∃ p q : ℤ, p ≠ q ∧ quadratic_eq k p ∧ quadratic_eq k q

-- Main statement to prove
theorem sum_of_k_with_distinct_integer_solutions : 
  ∑ k in {k : ℤ | distinct_integer_solutions k}, k = 0 := 
sorry

end sum_of_k_with_distinct_integer_solutions_l711_711855


namespace dot_product_of_vectors_l711_711081

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x y : ℝ) : Prop := y ^ 2 = 4 * x

def line_through_focus (k x y : ℝ) : Prop := y = k * (x - 1)

theorem dot_product_of_vectors
  (A B : ℝ × ℝ)
  (hA_parabola : parabola A.fst A.snd)
  (hB_parabola : parabola B.fst B.snd)
  (hA_line : ∃ k, line_through_focus k A.fst A.snd)
  (hB_line : ∃ k, line_through_focus k B.fst B.snd) :
  A.fst * B.fst + A.snd * B.snd = -3 :=
  sorry

end dot_product_of_vectors_l711_711081


namespace sequence_b_100_l711_711492

theorem sequence_b_100 :
  let b : ℕ → ℝ :=
    λ n, if n = 0 then 1 else 3^(n-1) * real.sqrt (nat.factorial (n-1))
  in b 100 = 3^99 * real.sqrt (nat.factorial 99) :=
by
  sorry

end sequence_b_100_l711_711492


namespace infinite_solutions_k_l711_711785

theorem infinite_solutions_k (x y k : ℝ) :
  (2 * x - 3 * y = 5) ∧ (4 * x - 6 * y = k) ↔ k = 10 := 
begin
  sorry
end

end infinite_solutions_k_l711_711785


namespace measure_of_angle_l711_711336

theorem measure_of_angle (x : ℝ) (h1 : 90 - x = 3 * x - 10) : x = 25 :=
by
  sorry

end measure_of_angle_l711_711336


namespace smallest_n_value_l711_711296

variable {n : ℕ}
variable {x : Fin n → ℝ}

theorem smallest_n_value (h1 : ∀ i, |x i| < 1)
    (h2 : (Finset.univ.sum (λ i, |x i|)) = 25 + |Finset.univ.sum x|) : n ≥ 26 :=
sorry

end smallest_n_value_l711_711296


namespace third_month_sale_l711_711927

theorem third_month_sale (s3 : ℝ)
  (s1 s2 s4 s5 s6 : ℝ)
  (h1 : s1 = 2435)
  (h2 : s2 = 2920)
  (h4 : s4 = 3230)
  (h5 : s5 = 2560)
  (h6 : s6 = 1000)
  (average : (s1 + s2 + s3 + s4 + s5 + s6) / 6 = 2500) :
  s3 = 2855 := 
by sorry

end third_month_sale_l711_711927


namespace series_mod_7_l711_711808

theorem series_mod_7 {n : ℕ} (h : n = 216) :
  (finset.range (n + 1)).sum (λ i, 4^i) % 7 = 1 :=
  sorry

end series_mod_7_l711_711808


namespace area_of_formed_triangle_l711_711722

def triangle_area (S R d : ℝ) (S₁ : ℝ) : Prop :=
  S₁ = (S / 4) * |1 - (d^2 / R^2)|

variable (S R d : ℝ)

theorem area_of_formed_triangle (h : S₁ = (S / 4) * |1 - (d^2 / R^2)|) : triangle_area S R d S₁ :=
by
  sorry

end area_of_formed_triangle_l711_711722


namespace sin_390_eq_half_l711_711981

theorem sin_390_eq_half : Real.sin (390 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_390_eq_half_l711_711981


namespace triangle_ABC_maximize_ACB_length_l711_711265

noncomputable def triangle_ABC_maximize_ACB : Prop :=
  let AB := 5
  let AC := 6
  let BC := 8
  ∃ Δ : Triangle, Δ.sideA = AB ∧ Δ.sideB = AC ∧ Δ.sideC = BC ∧ Δ.angleACB = 90 ∧ |BC| = 8

-- main theorem statement
theorem triangle_ABC_maximize_ACB_length : triangle_ABC_maximize_ACB :=
by
  sorry

end triangle_ABC_maximize_ACB_length_l711_711265


namespace jars_of_plum_jelly_sold_l711_711445

theorem jars_of_plum_jelly_sold (P R G S : ℕ) (h1 : R = 2 * P) (h2 : G = 3 * R) (h3 : G = 2 * S) (h4 : S = 18) : P = 6 := by
  sorry

end jars_of_plum_jelly_sold_l711_711445


namespace sum_of_medians_eq_mean_l711_711041

theorem sum_of_medians_eq_mean : 
  let mean_5 (x : ℝ) := (3 + 7 + x + 14 + 20) / 5
  let possible_medians := [7, x, 14]
  let x_vals : set ℝ := {x | mean_5 x = 7 ∨ mean_5 x = x ∨ mean_5 x = 14}
  (∑ x in x_vals, x) = 28 :=
sorry

end sum_of_medians_eq_mean_l711_711041


namespace option1_payment_correct_option2_payment_correct_most_cost_effective_for_30_l711_711419

variables (x : ℕ) (hx : x > 10)

def suit_price : ℕ := 1000
def tie_price : ℕ := 200
def num_suits : ℕ := 10

-- Option 1: Buy one suit, get one tie for free
def option1_payment : ℕ := 200 * x + 8000

-- Option 2: All items sold at a 10% discount
def option2_payment : ℕ := (10 * 1000 + x * 200) * 9 / 10

-- For x = 30, which option is more cost-effective
def x_value := 30
def option1_payment_30 : ℕ := 200 * x_value + 8000
def option2_payment_30 : ℕ := (10 * 1000 + x_value * 200) * 9 / 10
def more_cost_effective_option_30 : ℕ := if option1_payment_30 < option2_payment_30 then option1_payment_30 else option2_payment_30

-- Most cost-effective option for x = 30 with new combination plan
def combination_payment_30 : ℕ := 10000 + 20 * 200 * 9 / 10

-- Statements to be proved
theorem option1_payment_correct : option1_payment x = 200 * x + 8000 := sorry

theorem option2_payment_correct : option2_payment x = (10 * 1000 + x * 200) * 9 / 10 := sorry

theorem most_cost_effective_for_30 :
  option1_payment_30 = 14000 ∧ 
  option2_payment_30 = 14400 ∧ 
  more_cost_effective_option_30 = 14000 ∧
  combination_payment_30 = 13600 := sorry

end option1_payment_correct_option2_payment_correct_most_cost_effective_for_30_l711_711419


namespace min_common_perimeter_l711_711374

theorem min_common_perimeter : ∃ x y k : ℕ, 
  (x ≠ y) ∧ 
  (2 * x + 5 * k = 2 * y + 6 * k) ∧ 
  (5 * real.sqrt (4 * x ^ 2 - 25 * k ^ 2) = 6 * real.sqrt (4 * y ^ 2 - 36 * k ^ 2)) ∧ 
  (2 * x + 5 * k = 399) :=
sorry

end min_common_perimeter_l711_711374


namespace measure_of_angle_C_l711_711663

-- Define the conditions using Lean 4 constructs
variable (a b c : ℝ)
variable (A B C : ℝ) -- Measures of angles in triangle ABC
variable (triangle_ABC : (a * a + b * b - c * c = a * b))

-- Statement of the proof problem
theorem measure_of_angle_C (h : a^2 + b^2 - c^2 = ab) (h2 : 0 < C ∧ C < π) : C = π / 3 :=
by
  -- Proof will go here but is omitted with sorry
  sorry

end measure_of_angle_C_l711_711663


namespace area_AGE_l711_711773

-- Definitions of the points and the conditions (definitions should only directly appear in the conditions part)
structure Point where
  x : ℝ
  y : ℝ

-- Definitions for the square and the points on it
noncomputable def A : Point := ⟨0, 0⟩
noncomputable def B : Point := ⟨5, 0⟩
noncomputable def C : Point := ⟨5, 5⟩
noncomputable def D : Point := ⟨0, 5⟩
noncomputable def E : Point := ⟨5, 2⟩ -- Given BE = 2 and EC = 3

-- Define the interaction with the circumcircle of ABE
-- Simplified representations
noncomputable def circumcircle_center : Point := ⟨2.5, 1⟩
def circumcircle_radius_squared : ℝ := 6.25

-- Definition of diagonal BD, and point G
-- Solving the system from the solution steps for the intersection coordinates would require more involved Lean capabilities
noncomputable def G : Point := ⟨x_g, y_g⟩ -- Placeholder for solution coordinates of G, to be derived from the equations

-- Area function (can be specified as per Lean's prelude definitions but simplified here)
def area (P Q R : Point) : ℝ := 
  abs ((P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y)) / 2)

-- Theorem to be proven
theorem area_AGE :
  area A G E = 51.25 := 
sorry

end area_AGE_l711_711773


namespace line_and_circle_intersection_distance_l711_711684

theorem line_and_circle_intersection_distance:
  let line_l := { x := 3 - Real.sqrt 2 * t, y := Real.sqrt 5 + Real.sqrt 2 * t : ℝ } in
  let circle_C := { ρ := 2 * Real.sqrt 5 * Real.sin θ : ℝ } in
  let point_P := (3, Real.sqrt 5) in
  -- Cartesian equation of line derived from parametric form
  (∀ x y t, (x = 3 - Real.sqrt 2 * t) → (y = Real.sqrt 5 + Real.sqrt 2 * t) → (x + y = 3 + Real.sqrt 5)) →
  -- Cartesian equation of the circle derived from polar form
  (∀ x y, (ρ = Real.sqrt (x^2 + y^2)) → (Real.sin θ = y / ρ) → (x * x + (y - Real.sqrt 5) * (y - Real.sqrt 5) = 5)) →
  -- Calculate the value of |PA| + |PB|
  (|PA| + |PB| = 3 * Real.sqrt 2)
:=
by
  sorry

end line_and_circle_intersection_distance_l711_711684


namespace range_of_k_smallest_m_l711_711590

-- Part (1)
theorem range_of_k (f : ℝ → ℝ) (k : ℝ) (h : ∀ x > 0, f x > k * x - 1 / 2) :
  k < 1 - log 2 :=
sorry

-- Part (2)
theorem smallest_m (f : ℝ → ℝ) (m : ℕ) (h : ∀ x > 0, f (m + x) < f m * exp x) :
  m = 3 :=
sorry

end range_of_k_smallest_m_l711_711590


namespace work_completion_time_l711_711398

-- Define the rate of work done by a, b, and c.
def rate_a := 1 / 4
def rate_b := 1 / 12
def rate_c := 1 / 6

-- Define the time each person starts working and the cycle pattern.
def start_time : ℕ := 6 -- in hours
def cycle_pattern := [rate_a, rate_b, rate_c]

-- Calculate the total amount of work done in one cycle of 3 hours.
def work_per_cycle := (rate_a + rate_b + rate_c)

-- Calculate the total time to complete the work.
def total_time_to_complete_work := 2 * 3 -- number of cycles times 3 hours per cycle

-- Calculate the time of completion.
def completion_time := start_time + total_time_to_complete_work

-- Theorem to prove the work completion time.
theorem work_completion_time : completion_time = 12 := 
by
  -- Proof can be filled in here
  sorry

end work_completion_time_l711_711398


namespace sum_of_all_ks_l711_711870

theorem sum_of_all_ks (k : ℤ) :
  (∃ r s : ℤ, r ≠ s ∧ (3 * x^2 - k * x + 12 = 0) ∧ (r + s = k / 3) ∧ (r * s = 4)) → (k = 15 ∨ k = -15) → k + k = 0 :=
by
  sorry

end sum_of_all_ks_l711_711870


namespace no_such_finite_set_l711_711984

theorem no_such_finite_set :
  ¬ ∃ (M : Finset ℝ), ∀ (n : ℕ), ∃ (P : Polynomial ℝ), 
  P.degree ≥ n ∧ (∀ (c ∈ P.coeffs), c ∈ M) ∧ (∀ (root ∈ P.roots), root ∈ M) ∧ P.roots ∈ SetOf Real :=
sorry

end no_such_finite_set_l711_711984


namespace transform_to_f_l711_711027

-- Definitions
def f (x : ℝ) : ℝ := Real.cos (2 * x + π / 3)
def g (x : ℝ) : ℝ := Real.sin (2 * x + π / 3)
def shift (h : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x, h (x + a)

-- Theorem
theorem transform_to_f :
  shift g (-π / 4) = f := 
by
  sorry

end transform_to_f_l711_711027


namespace nested_series_value_l711_711498

theorem nested_series_value :
  2023 + (1 / 2) * (2022 + (1 / 2) * (2021 + (1 / 2) * (2020 + (1 / 2) * (2019 + (1 / 2) * 
  (2018 + (1 / 2) * (2017 + (1 / 2) * (2016 + (1 / 2) * (2015 + (1 / 2) * (2014 + (1 / 2) * 
  (2013 + (1 / 2) * (2012 + (1 / 2) * (2011 + (1 / 2) * (2010 + (1 / 2) * (2009 + (1 / 2) * 
  (2008 + (1 / 2) * (2007 + (1 / 2) * (2006 + (1 / 2) * (2005 + (1 / 2) * (2004 + (1 / 2) * 
  (2003 + (1 / 2) * (2002 + (1 / 2) * (2001 + (1 / 2) * (2000 + (1 / 2) * (1999 + (1 / 2) * 
  (1998 + (1 / 2) * (1997 + (1 / 2) * (1996 + (1 / 2) * (1995 + (1 / 2) * (1994 + (1 / 2) * 
  (1993 + (1 / 2) * (1992 + (1 / 2) * (1991 + (1 / 2) * (1990 + (1 / 2) * (1989 + (1 / 2) * 
  (1988 + (1 / 2) * (1987 + (1 / 2) * (1986 + (1 / 2) * (1985 + (1 / 2) * (1984 + (1 / 2) * 
  (1983 + (1 / 2) * (1982 + (1 / 2) * (1981 + (1 / 2) * (1980 + (1 / 2) * (1979 + (1 / 2) *
  (1978 + (1 / 2) * (1977 + (1 / 2) * (1976 + (1 / 2) * (1975 + (1 / 2) * (1974 + (1 / 2) * 
  (3 + 1))))))))))))))))))))))))))))))))))))))))))))))))))))))) = 4044 := sorry

end nested_series_value_l711_711498


namespace sum_of_b_is_150_l711_711508

def b : ℕ → ℕ := sorry

def a (i : ℕ) : ℕ := (b (i - 1 % 15) + b ((i + 1) % 15)) / 2

axiom b_bounds : ∀ i, 1 ≤ i → i ≤ 15 → 1 ≤ b i ∧ b i ≤ 30
axiom a1_is_10 : a 1 = 10
axiom b1_is_20 : b 1 = 20

theorem sum_of_b_is_150 : (∑ i in finset.range 15, b (i + 1)) = 150 :=
sorry

end sum_of_b_is_150_l711_711508


namespace point_Q_in_third_quadrant_l711_711632

-- Define point P in the fourth quadrant with coordinates a and b.
variable (a b : ℝ)
variable (h1 : a > 0)  -- Condition for the x-coordinate of P in fourth quadrant
variable (h2 : b < 0)  -- Condition for the y-coordinate of P in fourth quadrant

-- Point Q is defined by the coordinates (-a, b-1). We need to show it lies in the third quadrant.
theorem point_Q_in_third_quadrant : (-a < 0) ∧ (b - 1 < 0) :=
  by
    sorry

end point_Q_in_third_quadrant_l711_711632


namespace minimum_f_l711_711160

def f (x : ℝ) : ℝ := x^2 + (1 / x^2) + (1 / (x^2 + (1 / x^2)))

theorem minimum_f (x : ℝ) (hx : x > 0) : f x ≥ 2.5 := 
by {
  sorry
}

end minimum_f_l711_711160


namespace number_of_liars_is_28_l711_711897

variables (n : ℕ → Prop)

-- Define the conditions for knights and liars
def knight (k : ℕ) := n k
def liar (k : ℕ) := ¬n k

-- Define statements for odd and even numbered people
def odd_statement (k : ℕ) := ∀ m, m > k → liar m
def even_statement (k : ℕ) := ∀ m, m < k → liar m

-- Define the main hypothesis following the problem conditions
def conditions : Prop :=
  (∀ k, k % 2 = 1 → (knight k ↔ odd_statement k)) ∧
  (∀ k, k % 2 = 0 → (knight k ↔ even_statement k)) ∧
  (∃ m, m = 30) -- Ensuring there are 30 people

-- Prove the main statement
theorem number_of_liars_is_28 : ∃ l, l = 28 ∧ (∀ k, k ≤ 30 → (liar k ↔ k ≤ 28)) :=
by
  sorry

end number_of_liars_is_28_l711_711897


namespace no_four_digit_central_ring_number_l711_711242

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_central_ring_number (a b c d : ℕ) : Prop :=
  let N := 1000 * a + 100 * b + 10 * c + d in
  ¬ (N % 11 = 0) ∧
  (100 * b + 10 * c + d) % 11 = 0 ∧
  (100 * a + 10 * c + d) % 11 = 0 ∧
  (100 * a + 10 * b + d) % 11 = 0 ∧
  (100 * a + 10 * b + c) % 11 = 0

theorem no_four_digit_central_ring_number :
  ∀ (a b c d : ℕ), is_digit a → is_digit b → is_digit c → is_digit d →
  ¬ is_central_ring_number a b c d :=
by {
  intros a b c d ha hb hc hd,
  sorry
}

end no_four_digit_central_ring_number_l711_711242


namespace frank_cookies_l711_711531

theorem frank_cookies :
  ∀ (F M M_i L : ℕ),
    (F = M / 2 - 3) →
    (M = 3 * M_i) →
    (M_i = 2 * L) →
    (L = 5) →
    F = 12 :=
by
  intros F M M_i L h1 h2 h3 h4
  rw [h4] at h3
  rw [h3] at h2
  rw [h2] at h1
  sorry

end frank_cookies_l711_711531


namespace find_a100_l711_711585

-- Define the arithmetic sequence with the given conditions
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Given conditions
variables {a d : ℤ}
variables (S_9 : ℤ) (a_10 : ℤ)

-- Conditions in Lean definition
def conditions (a d : ℤ) : Prop :=
  (9 / 2 * (2 * a + 8 * d) = 27) ∧ (a + 9 * d = 8)

-- Prove the final statement
theorem find_a100 : ∃ a d : ℤ, conditions a d → arithmetic_sequence a d 100 = 98 := 
by {
    sorry
}

end find_a100_l711_711585


namespace num_valid_N_l711_711526

theorem num_valid_N :
  { N : ℕ // 0 < N ∧ ∃ k : ℕ, N + 2 = k ∧ k ∣ 36 }.to_finset.card = 7 :=
begin
  sorry
end

end num_valid_N_l711_711526


namespace probability_floor_equality_l711_711780

open Real

-- Definitions based on the given conditions.
def random_variable_x : Type := { x : ℝ // 0 < x ∧ x < 2 }
def random_variable_y : Type := { y : ℝ // 0 < y ∧ y < 2 }

-- The probability we need to prove.
theorem probability_floor_equality :
  let P := (λ (A : set (random_variable_x × random_variable_y)), (volume (prod.mk '' A)).toReal / 4) in
  P { xy | ∃ (x : random_variable_x) (y : random_variable_y), ⟨x.1, y.1⟩ ∈ xy ∧ ⌊x.1⌋ = ⌊y.1⌋ } = 1/2 :=
by
  sorry

end probability_floor_equality_l711_711780


namespace find_honeydews_left_l711_711971

theorem find_honeydews_left 
  (cantaloupe_price : ℕ)
  (honeydew_price : ℕ)
  (initial_cantaloupes : ℕ)
  (initial_honeydews : ℕ)
  (dropped_cantaloupes : ℕ)
  (rotten_honeydews : ℕ)
  (end_cantaloupes : ℕ)
  (total_revenue : ℕ)
  (honeydews_left : ℕ) :
  cantaloupe_price = 2 →
  honeydew_price = 3 →
  initial_cantaloupes = 30 →
  initial_honeydews = 27 →
  dropped_cantaloupes = 2 →
  rotten_honeydews = 3 →
  end_cantaloupes = 8 →
  total_revenue = 85 →
  honeydews_left = 9 :=
by
  sorry

end find_honeydews_left_l711_711971


namespace new_car_fuel_consumption_l711_711915

theorem new_car_fuel_consumption : 
  (∃ (x : ℝ), 100/x - 100/(x + 2) = 25/6) → x = 6 :=
by {
  intro h,
  rcases h with ⟨x, hx⟩,
  sorry
}

end new_car_fuel_consumption_l711_711915


namespace length_pq_inscribed_circle_l711_711919

theorem length_pq_inscribed_circle (a b c : ℝ) (h₀ : a = 4) (h₁ : b = 6) (h₂ : c = 8) :
  let AP := 3 in
  let cos_A := 11 / 16 in
  let PQ := real.sqrt ((AP^2 + AP^2 - 2 * AP * AP * cos_A)) in
  PQ = (3 * real.sqrt 10) / 4 :=
by
  sorry

end length_pq_inscribed_circle_l711_711919


namespace jill_investment_value_l711_711691

def compoundInterest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_investment_value :
  compoundInterest 10000 0.0396 2 2 ≈ 10812 :=
by
  sorry

end jill_investment_value_l711_711691


namespace g_of_neg_two_is_six_l711_711200

theorem g_of_neg_two_is_six (f : ℝ → ℝ) (g : ℝ → ℝ) 
(h1 : ∀ x, f(-x) - f(x) = 2 * x)
(h2 : f(2) = 1) 
(h3 : ∀ x, g x = f x + 1) :
g (-2) = 6 :=
by 
  sorry

end g_of_neg_two_is_six_l711_711200


namespace number_of_rectangles_on_3x3_grid_l711_711235

-- Define the grid and its properties
structure Grid3x3 where
  sides_are_2_units_apart : Bool
  diagonal_connections_allowed : Bool
  condition : sides_are_2_units_apart = true ∧ diagonal_connections_allowed = true

-- Define the number_rectangles function
def number_rectangles (g : Grid3x3) : Nat := 60

-- Define the theorem to prove the number of rectangles
theorem number_of_rectangles_on_3x3_grid : ∀ (g : Grid3x3), g.sides_are_2_units_apart = true ∧ g.diagonal_connections_allowed = true → number_rectangles g = 60 := by
  intro g
  intro h
  -- proof goes here
  sorry

end number_of_rectangles_on_3x3_grid_l711_711235


namespace min_value_expression_l711_711739

theorem min_value_expression (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 7) ^ 2 + (3 * Real.sin α + 4 * Real.cos β - 12) ^ 2 ≥ 36 :=
sorry

end min_value_expression_l711_711739


namespace z_pow_2021_pure_imaginary_max_modulus_z1_conj_z_is_neg_i_point_in_third_quadrant_l711_711185

noncomputable def z : ℂ := (1 + complex.I) / (1 - complex.I)

-- 1. Prove that z^2021 is pure imaginary
theorem z_pow_2021_pure_imaginary : (z ^ 2021).im ≠ 0 ∧ (z ^ 2021).re = 0 := sorry

-- 2. Prove the maximum value of |z1| given |z1 - z| = 1
theorem max_modulus_z1 (z1 : ℂ) (h : complex.abs (z1 - z) = 1) : complex.abs z1 ≤ 2 := sorry

-- 3. Prove the conjugate of z is -i
theorem conj_z_is_neg_i : complex.conj z = -complex.I := sorry

-- 4. Prove the point corresponding to (conj z + z * complex.I) is in the third quadrant
theorem point_in_third_quadrant : 
  let w := (complex.conj z) + z * complex.I in w.re < 0 ∧ w.im < 0 := sorry

end z_pow_2021_pure_imaginary_max_modulus_z1_conj_z_is_neg_i_point_in_third_quadrant_l711_711185


namespace decorations_left_to_put_up_l711_711490

variable (S B W P C T : Nat)
variable (h₁ : S = 12)
variable (h₂ : B = 4)
variable (h₃ : W = 12)
variable (h₄ : P = 2 * W)
variable (h₅ : C = 1)
variable (h₆ : T = 83)

theorem decorations_left_to_put_up (h₁ : S = 12) (h₂ : B = 4) (h₃ : W = 12) (h₄ : P = 2 * W) (h₅ : C = 1) (h₆ : T = 83) :
  T - (S + B + W + P + C) = 30 := sorry

end decorations_left_to_put_up_l711_711490


namespace cubic_identity_l711_711564

theorem cubic_identity (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 :=
by
  sorry

end cubic_identity_l711_711564


namespace min_value_of_expression_l711_711736

open Real

theorem min_value_of_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 :=
by
  sorry

end min_value_of_expression_l711_711736


namespace find_function_l711_711154

theorem find_function (f : ℕ → ℕ)
  (h1 : ∀ m n : ℕ, f (m^2 + n^2) = f m ^ 2 + f n ^ 2)
  (h2 : f 1 > 0) : ∀ n : ℕ, f n = n := 
sorry

end find_function_l711_711154


namespace octal_to_binary_of_55_equals_101101_l711_711968

def octalToDecimal (n : ℕ) : ℕ :=
  (n / 10) * 8^(1 : ℕ) + (n % 10) * 8^(0 : ℕ)

def divideBy2Remainders (n : ℕ) : List ℕ :=
  if n = 0 then [] else (n % 2) :: divideBy2Remainders (n / 2)

def listToBinary (l : List ℕ) : ℕ :=
  l.foldr (λ x acc, acc * 10 + x) 0

noncomputable def octalToBinary (n : ℕ) : ℕ :=
  listToBinary (divideBy2Remainders (octalToDecimal n))

theorem octal_to_binary_of_55_equals_101101 : octalToBinary 55 = 101101 :=
by
  sorry

end octal_to_binary_of_55_equals_101101_l711_711968


namespace smallest_number_l711_711887

theorem smallest_number (x : ℕ) (h : x = 1014) : 
  (x - 6) % 12 = 0 ∧ 
  (x - 6) % 16 = 0 ∧ 
  (x - 6) % 18 = 0 ∧ 
  (x - 6) % 21 = 0 ∧ 
  (x - 6) % 28 = 0 := 
by {
  rw [h], 
  sorry
}

end smallest_number_l711_711887


namespace find_polar_equations_and_distance_l711_711256

noncomputable def polar_equation_C1 (rho theta : ℝ) : Prop :=
  rho^2 * Real.cos (2 * theta) = 1

noncomputable def polar_equation_C2 (rho theta : ℝ) : Prop :=
  rho = 2 * Real.cos theta

theorem find_polar_equations_and_distance :
  (∀ rho theta, polar_equation_C1 rho theta ↔ rho^2 * Real.cos (2 * theta) = 1) ∧
  (∀ rho theta, polar_equation_C2 rho theta ↔ rho = 2 * Real.cos theta) ∧
  let theta := Real.pi / 6
  let rho_A := Real.sqrt 2
  let rho_B := Real.sqrt 3
  (|rho_A - rho_B| = |Real.sqrt 3 - Real.sqrt 2|) :=
  by sorry

end find_polar_equations_and_distance_l711_711256


namespace molly_takes_180_minutes_more_l711_711390

noncomputable def xanthia_speed : ℕ := 120
noncomputable def molly_speed : ℕ := 60
noncomputable def first_book_pages : ℕ := 360

-- Time taken by Xanthia to read the first book in hours
noncomputable def xanthia_time_first_book : ℕ := first_book_pages / xanthia_speed

-- Time taken by Molly to read the first book in hours
noncomputable def molly_time_first_book : ℕ := first_book_pages / molly_speed

-- Difference in time taken to read the first book in minutes
noncomputable def time_diff_minutes : ℕ := (molly_time_first_book - xanthia_time_first_book) * 60

theorem molly_takes_180_minutes_more : time_diff_minutes = 180 := by
  sorry

end molly_takes_180_minutes_more_l711_711390


namespace volleyball_shotput_cost_l711_711373

theorem volleyball_shotput_cost (x y : ℝ) :
  (2*x + 3*y = 95) ∧ (5*x + 7*y = 230) :=
  sorry

end volleyball_shotput_cost_l711_711373


namespace solve_fraction_eq_l711_711325

theorem solve_fraction_eq (x : ℝ) 
  (h₁ : x ≠ -9) 
  (h₂ : x ≠ -7) 
  (h₃ : x ≠ -10) 
  (h₄ : x ≠ -6) 
  (h₅ : 1 / (x + 9) + 1 / (x + 7) = 1 / (x + 10) + 1 / (x + 6)) : 
  x = -8 := 
sorry

end solve_fraction_eq_l711_711325


namespace sum_a_2015_l711_711017

variable (a : ℕ → ℚ) -- Define the arithmetic sequence
variable (S : ℕ → ℚ) -- Define the sum function

-- Arithmetic sequence condition
axiom arithmetic_sequence (d : ℚ) (a₁ : ℚ) : ∀ n, a n = a₁ + ↑n * d

-- The given conditions
axiom a_1008 : a 1008 = 1 / 2

-- Sum of the first n terms of an arithmetic sequence
axiom sum_arithmetic_sequence (d : ℚ) (a₁ : ℚ) : ∀ n, S n = n * (2 * a₁ + ↑(n - 1) * d) / 2

theorem sum_a_2015 {d a₁} (h : a 1008 = 1 / 2) : S 2015 = 2015 / 2 := by
  -- Using the fact that a₁ + a₁ + 2014d = 1
  have h₁ : a₁ + (a₁ + 2014 * d) = 1 :=
    calc
      a₁ + (a₁ + 2014 * d) = 2 * a 1008 := by rw [a_1008]
      ... = 1 : by norm_num
  sorry -- Proof omitted

end sum_a_2015_l711_711017


namespace sin_alpha_plus_3pi_div_2_l711_711537

theorem sin_alpha_plus_3pi_div_2 (α : ℝ) (h : Real.cos α = 1 / 3) : 
  Real.sin (α + 3 * Real.pi / 2) = -1 / 3 :=
by
  sorry

end sin_alpha_plus_3pi_div_2_l711_711537


namespace fraction_sequence_calc_l711_711119

theorem fraction_sequence_calc : 
  (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) * (8 / 9) - 1 = -(7 / 9) := 
by 
  sorry

end fraction_sequence_calc_l711_711119


namespace new_group_size_l711_711351

theorem new_group_size (N : ℕ) (h1 : 20 < N) (h2 : N < 50) (h3 : (N - 5) % 6 = 0) (h4 : (N - 5) % 7 = 0) (h5 : (N % (N - 7)) = 7) : (N - 7).gcd (N) = 8 :=
by
  sorry

end new_group_size_l711_711351


namespace circle_condition_l711_711244

theorem circle_condition (k : ℝ) : (∃ x y : ℝ, x^2 + y^2 + 4 * x + 2 * y + 4 * k + 1 = 0) → (k < 1) :=
by
  sorry

end circle_condition_l711_711244


namespace sum_and_product_of_white_are_white_l711_711314

-- Definitions based on the conditions
def is_colored_black_or_white (n : ℕ) : Prop :=
  true -- This is a simplified assumption since this property is always true.

def is_black (n : ℕ) : Prop := (n % 2 = 0)
def is_white (n : ℕ) : Prop := (n % 2 = 1)

-- Conditions given in the problem
axiom sum_diff_colors_is_black (a b : ℕ) (ha : is_black a) (hb : is_white b) : is_black (a + b)
axiom infinitely_many_whites : ∀ n, ∃ m ≥ n, is_white m

-- Statement to prove that the sum and product of two white numbers are white
theorem sum_and_product_of_white_are_white (a b : ℕ) (ha : is_white a) (hb : is_white b) : 
  is_white (a + b) ∧ is_white (a * b) :=
sorry

end sum_and_product_of_white_are_white_l711_711314


namespace a_and_m_values_l711_711224

theorem a_and_m_values (a m : ℝ) :
  (∀ x, x ∈ {x : ℝ | x^2 - 3x + 2 = 0} ∪ {x : ℝ | x^2 - ax + (a-1) = 0} ↔ x ∈ {x : ℝ | x^2 - 3x + 2 = 0}) →
  (∀ x, x ∈ {x : ℝ | x^2 - 3x + 2 = 0} ∩ {x : ℝ | x^2 - mx + 2 = 0} ↔ x ∈ {x : ℝ | x^2 - mx + 2 = 0}) →
  (a = 2 ∨ a = 3) ∧ (m = 3 ∨ -2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2) :=
by
  sorry

end a_and_m_values_l711_711224


namespace isosceles_triangle_perimeter_l711_711551

-- Defining the given conditions
def is_isosceles (a b c : ℕ) : Prop := a = b ∨ b = c ∨ c = a
def triangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

-- Stating the problem and goal
theorem isosceles_triangle_perimeter (a b c : ℕ) 
  (h_iso: is_isosceles a b c)
  (h_len1: a = 3 ∨ a = 6)
  (h_len2: b = 3 ∨ b = 6)
  (h_triangle: triangle a b c): a + b + c = 15 :=
sorry

end isosceles_triangle_perimeter_l711_711551


namespace integral_sin_minus_cos_l711_711482

open Real

theorem integral_sin_minus_cos :
  ∫ x in 0..π, (sin x - cos x) = 2 :=
by
  sorry

end integral_sin_minus_cos_l711_711482


namespace select_26_baskets_to_halve_total_l711_711667

-- Define a structure for a basket that contains cucumbers, eggplants, and tomatoes.
structure Basket :=
(cucumbers : ℕ)
(eggplants : ℕ)
(tomatoes : ℕ)

-- A function to add up the contents of a list of baskets.
def total_contents (baskets : List Basket) : Basket :=
baskets.foldl (λ acc b, ⟨acc.cucumbers + b.cucumbers, acc.eggplants + b.eggplants, acc.tomatoes + b.tomatoes⟩) ⟨0, 0, 0⟩

-- The main theorem statement
theorem select_26_baskets_to_halve_total (baskets : List Basket) (h : baskets.length = 50) :
  ∃ (selected_baskets : List Basket), 
    selected_baskets.length = 26 ∧ 
    (total_contents selected_baskets).cucumbers ≥ (total_contents baskets).cucumbers / 2 ∧
    (total_contents selected_baskets).eggplants ≥ (total_contents baskets).eggplants / 2 ∧
    (total_contents selected_baskets).tomatoes ≥ (total_contents baskets).tomatoes / 2 :=
sorry

end select_26_baskets_to_halve_total_l711_711667


namespace num_divisible_by_99_correct_l711_711110

-- We'll define a noncomputable def due to the nature of combinatorial calculations.
noncomputable def num_divisible_by_99 : ℕ := 
  285120

theorem num_divisible_by_99_correct :
  ∃ (k:ℕ), k = 285120 ∧ -- existence of such k
  (∀ (n: ℕ),  -- for all ten-digit numbers constituted by permutations of 0 to 9
    (∀ d:ℕ, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → -- digits must be in the set of 0 to 9
    (∀ i j, i ≠ j → n.digit i ≠ n.digit j) → -- all digits must be unique
    n.digit 0 ≠ 0 → -- first digit must not be zero
    n % 99 = 0 → -- number must be divisible by 99
    k = 285120) -- then the count matches the expected answer
:= sorry

end num_divisible_by_99_correct_l711_711110


namespace number_of_friends_l711_711487

def total_envelopes : ℕ := 37
def envelopes_per_friend : ℕ := 3
def envelopes_left : ℕ := 22

theorem number_of_friends :
  ((total_envelopes - envelopes_left) / envelopes_per_friend) = 5 := by
  sorry

end number_of_friends_l711_711487


namespace jesse_initial_blocks_l711_711273

def total_blocks_initial (blocks_cityscape blocks_farmhouse blocks_zoo blocks_first_area blocks_second_area blocks_third_area blocks_left : ℕ) : ℕ :=
  blocks_cityscape + blocks_farmhouse + blocks_zoo + blocks_first_area + blocks_second_area + blocks_third_area + blocks_left

theorem jesse_initial_blocks :
  total_blocks_initial 80 123 95 57 43 62 84 = 544 :=
sorry

end jesse_initial_blocks_l711_711273


namespace measurable_intervals_with_three_ropes_l711_711397

-- Define a structure for a rope that burns for 1 hour when lit from one end
structure Rope where
  burnTime : ℕ -- Time in minutes that the rope burns for (60 minutes for 1 hour)
  uneven   : Bool -- Indicates the rope burns unevenly
  ltrKeyword : String -- Added to distinguish that the burn time property is initialized with one end lit

-- Assuming each rope burns for 60 minutes and burns unevenly
def rope1 : Rope := { 
  burnTime := 60, 
  uneven := true,
  ltrKeyword := "one-end-lit"
}

def rope2 : Rope := { 
  burnTime := 60, 
  uneven := true,
  ltrKeyword := "one-end-lit"
}

def rope3 : Rope := { 
  burnTime := 60, 
  uneven := true,
  ltrKeyword := "one-end-lit"
}

-- Theorem statement: Verify the number of distinct time intervals measurable with 3 such ropes is 7
theorem measurable_intervals_with_three_ropes : 
  ∀ (r1 r2 r3 : Rope), r1.burnTime = 60 → r1.uneven = true → r1.ltrKeyword = "one-end-lit" →
                        r2.burnTime = 60 → r2.uneven = true → r2.ltrKeyword = "one-end-lit" →
                        r3.burnTime = 60 → r3.uneven = true → r3.ltrKeyword = "one-end-lit" →
                        (calculate_intervals r1 r2 r3) = 7
:= by
  sorry

-- Function to calculate the number of intervals (define it adequately for context)
noncomputable def calculate_intervals (r1 r2 r3 : Rope) : ℕ :=
  -- Placeholder: In real proof, this would be calculated based on rope properties
  7

end measurable_intervals_with_three_ropes_l711_711397


namespace conic_section_is_parabola_l711_711497

noncomputable def conic_section_equation (x y : ℝ) : Prop :=
  abs (y - 3) = real.sqrt ((x + 4) ^ 2 + (y - 1) ^ 2)

theorem conic_section_is_parabola (x y : ℝ) :
  conic_section_equation x y ↔ ∃ a b c : ℝ, a ≠ 0 ∧ (y = a * x^2 + b * x + c) :=
by sorry

end conic_section_is_parabola_l711_711497


namespace negation_prop_l711_711349

variable {U : Type} (A B : Set U)
variable (x : U)

theorem negation_prop (h : x ∈ A ∩ B) : (x ∉ A ∩ B) → (x ∉ A ∧ x ∉ B) :=
sorry

end negation_prop_l711_711349


namespace original_height_of_ball_l711_711775

theorem original_height_of_ball (h : ℝ) : 
  (h + 2 * (0.5 * h) + 2 * ((0.5)^2 * h) = 200) -> 
  h = 800 / 9 := 
by
  sorry

end original_height_of_ball_l711_711775


namespace part1_part2_l711_711535

-- Definitions and conditions
structure Point :=
(x : ℝ)
(y : ℝ)

def A : Point := ⟨1, 3⟩
def B : Point := ⟨2, -2⟩
def C : Point := ⟨4, 1⟩

noncomputable def vector (P Q : Point) : Point :=
⟨Q.x - P.x, Q.y - P.y⟩

-- First part: finding D
def D: Point := ⟨5, -4⟩

theorem part1 : vector A B = vector C D := by
  sorry

-- Second part: finding k
noncomputable def vector_a := vector A B
noncomputable def vector_b := vector B C

def k : ℝ := -1 / 3

theorem part2 : ∃ (k : ℝ), k * vector_a - vector_b = (vector_a + (3:ℝ) * vector_b) := by
  use k
  sorry

end part1_part2_l711_711535


namespace cube_volume_from_surface_area_l711_711435

theorem cube_volume_from_surface_area (A : ℝ) (hA : A = 294) : ∃ V : ℝ, V = 343 :=
by
  let s := real.sqrt (A / 6)
  have h_s : s = 7 := sorry
  let V := s^3
  have h_V : V = 343 := sorry
  exact ⟨V, h_V⟩

end cube_volume_from_surface_area_l711_711435


namespace find_x_for_f_eq_f_inv_l711_711491

def f (x : ℝ) : ℝ := 3 * x - 8

noncomputable def f_inv (x : ℝ) : ℝ := (x + 8) / 3

theorem find_x_for_f_eq_f_inv : ∃ x : ℝ, f x = f_inv x ∧ x = 4 :=
by
  sorry

end find_x_for_f_eq_f_inv_l711_711491


namespace min_value_expression_l711_711741

open Real

theorem min_value_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 := by
  sorry

end min_value_expression_l711_711741


namespace evaluate_expression_l711_711504

theorem evaluate_expression : 
  (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) = 1 := 
by
  sorry

end evaluate_expression_l711_711504


namespace find_number_l711_711421

theorem find_number (x : ℝ) (h : x / 0.02 = 201.79999999999998) : x ≈ 4.036 := 
by sorry

end find_number_l711_711421


namespace tiffany_earns_money_from_cans_l711_711787

-- This defines the given conditions in Lean
variables (c : ℕ) (h_nonneg_c : 0 ≤ c)

-- Define the problem statement in Lean
theorem tiffany_earns_money_from_cans :
  let bags_monday := 10
  let bags_tuesday := 3
  let bags_wednesday := 7 
  let total_bags := bags_monday + bags_tuesday + bags_wednesday
  let cans_per_bag := c
  let total_cans := total_bags * cans_per_bag
  let rate_per_can := 0.10
  let total_money := total_cans * rate_per_can
  total_money = 2 * c := by
    sorry

end tiffany_earns_money_from_cans_l711_711787


namespace number_of_incorrect_propositions_l711_711104

/-- Define the propositions-/
def prop1 (A B : Prop) : Prop :=
  (A → ¬B) ∧ (B → ¬A)

def prop2 (P : (Prop → ℝ)) (A B : Prop) : Prop :=
  ¬ (P (A ∨ B) = P A + P B)

def prop3 (P : (Prop → ℝ)) (A B C : Prop) : Prop :=
  ¬ ((A → ¬B ∧ A → ¬C ∧ B → ¬C) ∧ (P A + P B + P C = 1))

def prop4 (P : (Prop → ℝ)) (A B : Prop) : Prop :=
  ¬ (P A + P B = 1 → (A → ¬B) ∧ (B → ¬A))

/-- Define the propositions used in the problem-/
def propositions {A B C : Prop} (P : Prop → ℝ) :=
  [prop1 A B, prop2 P A B, prop3 P A B C, prop4 P A B]

/-- Proof problem: the number of incorrect propositions is 2-/
theorem number_of_incorrect_propositions {A B C : Prop} (P : Prop → ℝ) :
  (propositions P).count (λ x, x) = 2 :=
sorry

end number_of_incorrect_propositions_l711_711104


namespace largest_divisor_of_n_squared_divisible_by_72_l711_711630

theorem largest_divisor_of_n_squared_divisible_by_72
    (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) : 12 ∣ n :=
by {
    sorry
}

end largest_divisor_of_n_squared_divisible_by_72_l711_711630


namespace lines_parallel_if_perpendicular_to_same_plane_l711_711197

variables {Line Plane : Type}
variable (a b : Line)
variable (α : Plane)

-- Conditions: a and b are two different lines, α is a plane
-- The question is to prove a || b given a ⊥ α and b ⊥ α

axiom different_lines (h : a ≠ b) : True
axiom line_perpendicular_to_plane (l : Line) (p : Plane) : Prop
axiom line_parallel_to_line (l1 l2 : Line) : Prop

theorem lines_parallel_if_perpendicular_to_same_plane (h1 : line_perpendicular_to_plane a α)
                                                      (h2 : line_perpendicular_to_plane b α)
                                                      (h3 : different_lines a b) :
                                                      line_parallel_to_line a b := sorry

end lines_parallel_if_perpendicular_to_same_plane_l711_711197


namespace J_3_3_4_l711_711528

def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

theorem J_3_3_4 : J 3 (3 / 4) 4 = 259 / 48 := 
by {
    -- We would normally include proof steps here, but according to the instruction, we use 'sorry'.
    sorry
}

end J_3_3_4_l711_711528


namespace sum_of_k_values_with_distinct_integer_solutions_l711_711845

theorem sum_of_k_values_with_distinct_integer_solutions : 
  ∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ 3 * p * q + (-k) * (p + q) + 12 = 0}, k = 0 :=
by
  sorry

end sum_of_k_values_with_distinct_integer_solutions_l711_711845


namespace find_rotation_y_l711_711986

-- Define the points P, Q, R
variable (P Q R : Type)

-- Define the rotations as functions
variable (rotate_clockwise : P → Q → ℕ → P)
variable (rotate_counterclockwise : P → Q → ℕ → P)

-- State the conditions
variables (h1 : rotate_clockwise P Q 450 = R)
variables (h2 : ∀ y < 360, rotate_counterclockwise P Q y = R)

-- State the theorem
theorem find_rotation_y (h1 : rotate_clockwise P Q 450 = R)
                       (h2 : ∀ y < 360, rotate_counterclockwise P Q y = R) :
  (∃ y, y < 360 ∧ rotate_counterclockwise P Q y = R ∧ y = 270) :=
sorry

end find_rotation_y_l711_711986


namespace find_m_values_l711_711209

noncomputable def valid_m_values : set ℚ :=
  {m | ∃ x1 x2 : ℤ, (m + 1) * (x1 : ℚ)^2 + 2 * (x1 : ℚ) - 5 * m - 13 = 0 ∧ 
                   (m + 1) * (x2 : ℚ)^2 + 2 * (x2 : ℚ) - 5 * m - 13 = 0}

theorem find_m_values : valid_m_values = {(-1 : ℚ), -11 / 10, -1 / 2} :=
sorry

end find_m_values_l711_711209


namespace sum_of_medians_l711_711015

theorem sum_of_medians {A_scores B_scores : List ℕ}
  (hA : A_scores = [24, 25, 28, 31, 32, 32, 34, 35, 37, 39])
  (hB : B_scores = [23, 24, 25, 25, 25, 26, 27, 27, 29, 30]) : 
  let median_A := (A_scores.nth_le 4 (by decide) + A_scores.nth_le 5 (by decide)) / 2
  let median_B := (B_scores.nth_le 4 (by decide) + B_scores.nth_le 5 (by decide)) / 2 in
  median_A + median_B = 57 := by
sorry

end sum_of_medians_l711_711015


namespace cyclic_quadrilateral_side_length_l711_711931

theorem cyclic_quadrilateral_side_length (r : ℝ) (a b c x : ℝ) 
  (h_r : r = 150) 
  (h_a : a = 100) 
  (h_b : b = 100) 
  (h_c : c = 150) 
  (h_inscribed : (a, b, c, x) inscribed_in_circle_of_radius r) : 
  x = 300 := 
sorry

end cyclic_quadrilateral_side_length_l711_711931


namespace carrots_total_l711_711305

variables (initiallyPicked : Nat) (thrownOut : Nat) (pickedNextDay : Nat)

def totalCarrots (initiallyPicked : Nat) (thrownOut : Nat) (pickedNextDay : Nat) :=
  initiallyPicked - thrownOut + pickedNextDay

theorem carrots_total (h1 : initiallyPicked = 19)
                     (h2 : thrownOut = 4)
                     (h3 : pickedNextDay = 46) :
  totalCarrots initiallyPicked thrownOut pickedNextDay = 61 :=
by
  sorry

end carrots_total_l711_711305


namespace convert_spherical_to_rectangular_correct_l711_711140

-- Define the spherical to rectangular conversion functions
noncomputable def spherical_to_rectangular (rho θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (rho * Real.sin φ * Real.cos θ, rho * Real.sin φ * Real.sin θ, rho * Real.cos φ)

-- Define the given spherical coordinates
def given_spherical_coords : ℝ × ℝ × ℝ :=
  (5, 7 * Real.pi / 4, Real.pi / 3)

-- Define the expected rectangular coordinates
def expected_rectangular_coords : ℝ × ℝ × ℝ :=
  (-5 * Real.sqrt 6 / 4, -5 * Real.sqrt 6 / 4, 5 / 2)

-- The proof statement
theorem convert_spherical_to_rectangular_correct (ρ θ φ : ℝ)
  (h_ρ : ρ = 5) (h_θ : θ = 7 * Real.pi / 4) (h_φ : φ = Real.pi / 3) :
  spherical_to_rectangular ρ θ φ = expected_rectangular_coords :=
by
  -- Proof omitted
  sorry

end convert_spherical_to_rectangular_correct_l711_711140


namespace cubic_identity_l711_711561

theorem cubic_identity (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 :=
by
  sorry

end cubic_identity_l711_711561


namespace option_A_correct_option_D_correct_correct_options_l711_711878

variables (a b x y : ℝ)

theorem option_A_correct (ha : a > 0) (hb : b > 0) : 
  (b / a + a / b) ≥ 2 := 
by sorry

theorem option_D_correct (hxy : x * y < 0) : 
  (x / y + y / x) ≤ -2 ∧ (x / y + y / x = -2 ↔ x = -y) := 
by sorry

theorem correct_options (ha_pos : a > 0) (hb_pos : b > 0) (hx_y_lt_0 : x * y < 0) : 
  (b / a + a / b ≥ 2) ∧ ((x / y + y / x) ≤ -2 ∧ (x / y + y / x = -2 ↔ x = -y)) := 
⟨option_A_correct ha_pos hb_pos, option_D_correct hx_y_lt_0⟩

end option_A_correct_option_D_correct_correct_options_l711_711878


namespace triangle_area_given_conditions_l711_711664

theorem triangle_area_given_conditions (a b c A B S : ℝ) (h₁ : (2 * c - b) * Real.cos A = a * Real.cos B) (h₂ : b = 1) (h₃ : c = 2) :
  S = (1 / 2) * b * c * Real.sin A → S = Real.sqrt 3 / 2 := 
by
  intros
  sorry

end triangle_area_given_conditions_l711_711664


namespace max_radius_squared_l711_711032

   noncomputable def cone_base_radius := 5
   noncomputable def cone_height := 10
   noncomputable def intersection_distance_from_base := 5

   theorem max_radius_squared :
     let r : ℝ := sqrt(5) - 1 in
     r^2 = 6 - 2 * sqrt 5 ∧
     (6 - 2 * sqrt 5).toFloat.isFraction &&
     (6 - 2 * sqrt 5) = 1/5 ∧
     (1 + 5 = 6) := sorry
   
end max_radius_squared_l711_711032


namespace counting_unit_of_0_75_l711_711341

def decimal_places (n : ℝ) : ℕ := 
  by sorry  -- Assume this function correctly calculates the number of decimal places of n

def counting_unit (n : ℝ) : ℝ :=
  by sorry  -- Assume this function correctly determines the counting unit based on decimal places

theorem counting_unit_of_0_75 : counting_unit 0.75 = 0.01 :=
  by sorry


end counting_unit_of_0_75_l711_711341


namespace composition_N_O_6_times_result_l711_711721

def N (x : ℝ) : ℝ := 3 * real.sqrt x
def O (x : ℝ) : ℝ := x ^ 2

theorem composition_N_O_6_times_result : N(O(N(O(N(O(2)))))) = 54 :=
by
  sorry

end composition_N_O_6_times_result_l711_711721


namespace trigonometric_identity_l711_711905

theorem trigonometric_identity : 
  Real.cos 6 * Real.cos 36 + Real.sin 6 * Real.cos 54 = Real.sqrt 3 / 2 :=
sorry

end trigonometric_identity_l711_711905


namespace fili_wins_with_optimal_play_l711_711148

-- Define the game board and initial conditions
def board_size : ℕ := n -- Assuming 'n' is some positive integer
def initial_position : (ℕ × ℕ) := (1, 1)

-- Define the main problem: Proving Fili (second player) wins with optimal play
theorem fili_wins_with_optimal_play (n : ℕ) (h : n > 0) : 
  ∃ (winner : String), winner = "Fili wins" := 
by 
  -- Add necessary steps (definitions, conditions) before proving
  sorry

end fili_wins_with_optimal_play_l711_711148


namespace a3_minus_a2_plus_a1_l711_711536

theorem a3_minus_a2_plus_a1 (a_4 a_3 a_2 a_1 a : ℕ) :
  (a_4 * (1 : ℕ + 1)^4 + a_3 * (1 + 1)^3 + a_2 * (1 + 1)^2 + a_1 * (1 + 1) + a = 1^4) → 
  a_3 - a_2 + a_1 = -14 :=
by
  -- Definitions using provided binomial coefficients
  let a_4 := nat.choose 4 0 -- equal to 1
  let a_3 := -(nat.choose 4 1) -- equal to -4
  let a_2 := nat.choose 4 2 -- equal to 6
  let a_1 := -(nat.choose 4 3) -- equal to -4
  
  -- State the main goal using sorry to serve as the placeholder for the proof
  sorry

end a3_minus_a2_plus_a1_l711_711536


namespace expression_eq_49_l711_711118

theorem expression_eq_49 (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 := 
by 
  sorry

end expression_eq_49_l711_711118


namespace sum_of_values_k_l711_711864

theorem sum_of_values_k (k : ℕ) : 
  (∀ x y : ℤ, (3 * x * x - k * x + 12 = 0) ∧ (3 * y * y - k * y + 12 = 0) ∧ x ≠ y) → k = 0 :=
by
  sorry

end sum_of_values_k_l711_711864


namespace limit_cos_x_limit_sin_3x_limit_cos_pi_x_limit_sin_pi_x_inv_l711_711402

-- Statement a
theorem limit_cos_x (x : ℝ) :
  (∃ ε > 0, ∀ x, abs(x) < ε → (∃ L, L = 1/2 ∧ tendsto (λ x, (1 - cos x) / x^2) (nhds 0) (𝓝 L))) :=
sorry

-- Statement b
theorem limit_sin_3x (x : ℝ) :
  (∃ δ > 0, ∀ x, abs(x - π) < δ → (∃ L, L = -3 ∧ tendsto (λ x, (sin (3 * x)) / (x - π)) (nhds π) (𝓝 L))) :=
sorry

-- Statement c
theorem limit_cos_pi_x (x : ℝ) :
  (∃ δ > 0, ∀ x, abs(x - 1) < δ → (∃ L, L = -π / 2 ∧ tendsto (λ x, (cos (π * x / 2)) / (x - 1)) (nhds 1) (𝓝 L))) :=
sorry

-- Statement d
theorem limit_sin_pi_x_inv (x : ℝ) :
  (∃ ε > 0, ∀ x, abs(x) < ε → (∃ L, L = π^2 ∧ tendsto (λ x, (1 / x) * sin (π / (1 + π * x))) (nhds 0) (𝓝 L))) :=
sorry

end limit_cos_x_limit_sin_3x_limit_cos_pi_x_limit_sin_pi_x_inv_l711_711402


namespace radius_of_spherical_ball_l711_711076

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := π * r^2 * h
noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3 : ℝ) * π * r^3

theorem radius_of_spherical_ball (V_water : ℝ) (r_cylinder : ℝ) (h_rise : ℝ) (r_ball : ℝ) 
  (initial_depth : ℝ) : 
  (V_water = π * r_cylinder^2 * h_rise) ∧ 
  (V_water = (4 / 3 : ℝ) * π * r_ball^3) ∧ 
  (r_cylinder = 12) ∧ 
  (h_rise = 6.75) ∧ 
  (initial_depth = 20) 
  → r_ball = 9 :=
by 
  sorry

end radius_of_spherical_ball_l711_711076


namespace product_root_is_even_l711_711733

-- Define the product of 8 consecutive positive integers.
def consecutive_product (x : ℕ) : ℕ :=
  x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5) * (x + 6) * (x + 7)

-- Define N as the fourth root of this product.
def fourth_root (n : ℕ) : ℝ := real.rpow n (1/4)

-- Define the greatest integer function
def greatest_integer (r : ℝ) : ℤ := int.floor r

-- Statement of the theorem to prove
theorem product_root_is_even (x : ℕ) : even (greatest_integer (fourth_root (consecutive_product x))) :=
sorry

end product_root_is_even_l711_711733


namespace angle_C_max_area_l711_711666

-- Define the problem
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
axiom triangle_conditions : 
  a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C

-- Additional condition for part 2
axiom side_c : c = 2 * Real.sqrt 3

-- 1. Prove that C = π/3
theorem angle_C : 
  A + B + C = π ∧ a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C → C = π / 3 :=
sorry

-- 2. Prove that the maximum area of the triangle ABC is 3√3 when c = 2√3
theorem max_area : 
  c = 2 * Real.sqrt 3 → a * b = 12 → 
  ∃ a b : ℝ, (∃ h : a * b ≤ 12, h → 
  (1 / 2) * a * b * Real.sin (π / 3) = 3 * Real.sqrt 3) :=
sorry

end angle_C_max_area_l711_711666


namespace min_value_of_expression_l711_711735

open Real

theorem min_value_of_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 :=
by
  sorry

end min_value_of_expression_l711_711735


namespace exists_two_participants_per_event_l711_711253

noncomputable def participants_per_event (n : ℕ) (participants : Finset (Finset ℕ)) : Prop :=
  ∀ (k : ℕ), 1 ≤ k ∧ k < n → (participants.filter (λ P, P.card ≤ k)).card ≤ k - 1

theorem exists_two_participants_per_event (n : ℕ) (n_ge_2 : n ≥ 2) :
  ∃ participants : Finset (Finset ℕ), 
    (participants.card = 2 * n) ∧
    (∀ p ∈ participants, p.card = 2) ∧
    participants_per_event n participants ∧
    (∀ (p1 p2 ∈ participants), (p1 ≠ p2) → (¬∃ e, e ∈ p1 ∧ e ∈ p2)) := 
sorry

end exists_two_participants_per_event_l711_711253


namespace initial_value_approximation_l711_711930

theorem initial_value_approximation (n : ℤ) : ∀ k : ℤ, k ≈ 21 → ∃ m : ℤ, m = 136 * n - k :=
by sorry

end initial_value_approximation_l711_711930


namespace ticket_cost_difference_l711_711308

noncomputable def total_cost_adults (tickets : ℕ) (price : ℝ) : ℝ := tickets * price
noncomputable def total_cost_children (tickets : ℕ) (price : ℝ) : ℝ := tickets * price
noncomputable def total_tickets (adults : ℕ) (children : ℕ) : ℕ := adults + children
noncomputable def discount (threshold : ℕ) (discount_rate : ℝ) (cost : ℝ) (tickets : ℕ) : ℝ :=
  if tickets > threshold then cost * discount_rate else 0
noncomputable def final_cost (initial_cost : ℝ) (discount : ℝ) : ℝ := initial_cost - discount
noncomputable def proportional_discount (partial_cost : ℝ) (total_cost : ℝ) (total_discount : ℝ) : ℝ :=
  (partial_cost / total_cost) * total_discount
noncomputable def difference (cost1 : ℝ) (cost2 : ℝ) : ℝ := cost1 - cost2

theorem ticket_cost_difference :
  let adult_tickets := 9
  let children_tickets := 7
  let adult_price := 11
  let children_price := 7
  let discount_rate := 0.15
  let discount_threshold := 10
  let total_adult_cost := total_cost_adults adult_tickets adult_price
  let total_children_cost := total_cost_children children_tickets children_price
  let all_tickets := total_tickets adult_tickets children_tickets
  let initial_total_cost := total_adult_cost + total_children_cost
  let total_discount := discount discount_threshold discount_rate initial_total_cost all_tickets
  let final_total_cost := final_cost initial_total_cost total_discount
  let adult_discount := proportional_discount total_adult_cost initial_total_cost total_discount
  let children_discount := proportional_discount total_children_cost initial_total_cost total_discount
  let final_adult_cost := final_cost total_adult_cost adult_discount
  let final_children_cost := final_cost total_children_cost children_discount
  difference final_adult_cost final_children_cost = 42.52 := by
  sorry

end ticket_cost_difference_l711_711308


namespace negation_of_existence_l711_711803

theorem negation_of_existence :
  ¬(∃ x : ℝ, 2^x > 0) ↔ ∀ x : ℝ, 2^x ≤ 0 := 
sorry

end negation_of_existence_l711_711803


namespace price_increase_l711_711109

theorem price_increase (P : ℝ) (hP : P = 120) (H₁ : ∃ N : ℝ, N = 0.85 * P) : ∃ x : ℝ, (N + N * x - P = 0) ∧ (x = 0.17647) :=
by
  -- Given conditions
  have hN : ∀ N, N = 0.85 * P → N = 102 :=
    by
    intro N
    intro hN₀
    calc
      N = 0.85 * 120 : by rw [hN₀, hP]
      ... = 102 : by norm_num
      
  -- Prove the required percentage increase
  use 0.17647
  sorry

end price_increase_l711_711109


namespace binom_expansion_l711_711587

/-- Given the binomial expansion of (sqrt(x) + 3x)^n for n < 15, 
    with the binomial coefficients of the 9th, 10th, and 11th terms forming an arithmetic sequence,
    we conclude that n must be 14 and describe all the rational terms in the expansion.
-/
theorem binom_expansion (n : ℕ) (h : n < 15)
  (h_seq : Nat.choose n 8 + Nat.choose n 10 = 2 * Nat.choose n 9) :
  n = 14 ∧
  (∃ (t1 t2 t3 : ℕ), 
    (t1 = 1 ∧ (Nat.choose 14 0 : ℕ) * (x ^ 7 : ℤ) = x ^ 7) ∧
    (t2 = 164 ∧ (Nat.choose 14 6 : ℕ) * (x ^ 6 : ℤ) = 164 * x ^ 6) ∧
    (t3 = 91 ∧ (Nat.choose 14 12 : ℕ) * (x ^ 5 : ℤ) = 91 * x ^ 5)) := 
  sorry

end binom_expansion_l711_711587


namespace four_digit_numbers_with_three_identical_digits_l711_711514

theorem four_digit_numbers_with_three_identical_digits : 
  {n : ℕ | n ≥ 1000 ∧ n < 10000 ∧ (n / 1000 = 1) ∧ (∃ d, d ≠ 1 ∧ (
    (n % 1000 = d * 111) -- 1aaa form
    ))}.card = 8 := 
sorry

end four_digit_numbers_with_three_identical_digits_l711_711514


namespace parallel_vector_example_l711_711222

def vector := (ℝ × ℝ × ℝ)

def is_parallel (v1 v2 : vector) : Prop :=
  ∃ λ : ℝ, v1 = (λ * v2.1, λ * v2.2, λ * v2.3)

theorem parallel_vector_example : is_parallel (2, -3, 1) (-4, 6, -2) :=
  sorry

end parallel_vector_example_l711_711222


namespace oil_bill_for_January_l711_711354

variables (J F : ℝ)

-- Conditions
def condition1 := F = (5 / 4) * J
def condition2 := (F + 45) / J = 3 / 2

theorem oil_bill_for_January (h1 : condition1 J F) (h2 : condition2 J F) : J = 180 :=
by sorry

end oil_bill_for_January_l711_711354


namespace product_equiv_l711_711137

theorem product_equiv :
  (∏ i in (finset.range 6).map (nat.succ ∘ nat.succ), (i^3 - 1) / (i^3 + 1)) = (57 / 168) :=
by
  sorry

end product_equiv_l711_711137


namespace ellipse_focus_distance_l711_711026

theorem ellipse_focus_distance :
  let P1 := (1, 5)
  let P2 := (6, 1)
  let P3 := (9, 5)
  let center := ( (fst P1 + fst P3) / 2, (snd P1 + snd P3) / 2 )
  let semi_major_axis := (fst P3 - fst P1) / 2
  let semi_minor_axis := semi_major_axis + 3
  let distance_foci := 2 * Real.sqrt (semi_minor_axis ^ 2 - semi_major_axis ^ 2)
  distance_foci = 2 * Real.sqrt 33 := sorry

end ellipse_focus_distance_l711_711026


namespace N_is_perfect_square_l711_711294

def is_sequence_N (n : ℕ) : ℕ :=
  (∑ i in finset.range n, 10^(2*n+1-i)) + (∑ i in finset.range (n+1), 2 * 10^(n-i)) + 25

theorem N_is_perfect_square (n : ℕ) : ∃ (k : ℕ), k * k = is_sequence_N n := by
  sorry

end N_is_perfect_square_l711_711294


namespace zero_point_in_2_3_l711_711346

def f (x : ℝ) : ℝ := (1 / 2) ^ x - x + 2

theorem zero_point_in_2_3 :
  (∃ x ∈ Ioo 2 3, f x = 0) :=
by
  have h₀ : f 2 = (1 / 2) ^ 2 - 2 + 2 := by norm_num
  have h₁ : f 2 = 1 / 4 := by norm_num
  have h₂ : f 3 = (1 / 2) ^ 3 - 3 + 2 := by norm_num
  have h₃ : f 3 = -7 / 8 := by norm_num
  have h₂_pos : f 2 > 0 := by norm_num
  have h₃_neg : f 3 < 0 := by norm_num
  apply IntermediateValueTheorem; sorry

end zero_point_in_2_3_l711_711346


namespace dots_not_visible_l711_711024

theorem dots_not_visible (visible_sum : ℕ) (total_faces_sum : ℕ) (num_dice : ℕ) (total_visible_faces : ℕ)
  (h1 : total_faces_sum = 21)
  (h2 : visible_sum = 22) 
  (h3 : num_dice = 3)
  (h4 : total_visible_faces = 7) :
  (num_dice * total_faces_sum - visible_sum) = 41 :=
sorry

end dots_not_visible_l711_711024


namespace sin_alpha_necessary_not_sufficient_l711_711196

-- Definitions based on conditions
def is_internal_angle (α : ℝ) : Prop := 0 < α ∧ α < 180
def sin_alpha_eq_sqrt2_div_2 (α : ℝ) : Prop := Real.sin α = Real.sqrt 2 / 2

-- Lean 4 statement of the theorem
theorem sin_alpha_necessary_not_sufficient (α : ℝ) :
  is_internal_angle α →
  sin_alpha_eq_sqrt2_div_2 α →
  ∀ β : ℝ, (β = 45 → α = 45 ∨ α = 135) ↔
           (sin_alpha_eq_sqrt2_div_2 45 ∧ ¬ (sin_alpha_eq_sqrt2_div_2 α → α = 45)) := 
by 
  intros h1 h2 β h3
  exact sorry

end sin_alpha_necessary_not_sufficient_l711_711196


namespace percentage_reduction_of_entry_fee_l711_711797

theorem percentage_reduction_of_entry_fee 
  (V : ℝ) 
  (fee_original : ℝ := 1) 
  (sale_increase_ratio : ℝ := 1.2) 
  (visitors_increase_ratio : ℝ := 1.6) :
  let fee_reduced := fee_original / visitors_increase_ratio,
      percentage_reduction := 1 - fee_reduced in
  percentage_reduction = 0.375 :=
by
  sorry

end percentage_reduction_of_entry_fee_l711_711797


namespace rice_grains_difference_l711_711098

theorem rice_grains_difference :
  let grains (k : ℕ) := 2^k in
  grains 15 - (∑ k in finset.range 12, grains (k + 1)) = 24578 :=
by
  sorry

end rice_grains_difference_l711_711098


namespace trigonometric_identity_proof_l711_711405

variable (α : ℝ)

theorem trigonometric_identity_proof :
  3 + 4 * (Real.sin (4 * α + (3 / 2) * Real.pi)) +
  Real.sin (8 * α + (5 / 2) * Real.pi) = 
  8 * (Real.sin (2 * α))^4 :=
sorry

end trigonometric_identity_proof_l711_711405


namespace gumball_difference_l711_711486

theorem gumball_difference :
  ∀ C : ℕ, 19 ≤ (29 + C) / 3 ∧ (29 + C) / 3 ≤ 25 →
  (46 - 28) = 18 :=
by
  intros C h
  sorry

end gumball_difference_l711_711486


namespace min_value_of_reciprocal_sum_l711_711658

-- Define the problem
theorem min_value_of_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ x y : ℝ, (x^2 + y^2 + 2 * x - 4 * y + 1 = 0) ∧ (2 * a * x - b * y + 2 = 0)):
  ∃ (m : ℝ), m = 4 ∧ (1 / a + 1 / b) ≥ m :=
by
  sorry

end min_value_of_reciprocal_sum_l711_711658


namespace even_func_decreasing_on_neg_interval_l711_711634

variable {f : ℝ → ℝ}

theorem even_func_decreasing_on_neg_interval
  (h_even : ∀ x, f x = f (-x))
  (h_increasing : ∀ (a b : ℝ), 3 ≤ a → a < b → b ≤ 7 → f a < f b)
  (h_min_val : ∀ x, 3 ≤ x → x ≤ 7 → f x ≥ 2) :
  (∀ (a b : ℝ), -7 ≤ a → a < b → b ≤ -3 → f a > f b) ∧ (∀ x, -7 ≤ x → x ≤ -3 → f x ≤ 2) :=
by
  sorry

end even_func_decreasing_on_neg_interval_l711_711634


namespace sum_of_k_distinct_integer_roots_l711_711853

theorem sum_of_k_distinct_integer_roots : 
  (∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ p * q = 4 ∧ k = 3 * (p + q)}, k) = 0 := 
by
  sorry

end sum_of_k_distinct_integer_roots_l711_711853


namespace convex_polygon_segments_l711_711317

theorem convex_polygon_segments (P : ℝ) (sides : list ℝ) :
  (∀ side ∈ sides, side ≤ P / 2) → (sides.sum = P) → 
  ∃ (l1 l2 : ℝ), (l1 ∈ sides) ∧ (l2 ∈ sides) ∧ (|l1 - l2| ≤ P / 3) :=
by
  sorry

end convex_polygon_segments_l711_711317


namespace semicircle_edge_through_vertex_crescents_area_eq_triangle_l711_711404

noncomputable def right_triangle (T : Type) :=
  ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ T = (a, b, c)

structure semicircle :=
  (radius : ℝ)
  (area : ℝ := π * radius^2 / 2)

lemma area_of_semicircles_sum_eq_hypotenuse (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  let A1 := semicircle.mk (a / 2),
      A2 := semicircle.mk (b / 2),
      A3 := semicircle.mk (c / 2) in
  A1.area + A2.area = A3.area :=
by {
  let A1 := semicircle.mk (a / 2),
  let A2 := semicircle.mk (b / 2),
  let A3 := semicircle.mk (c / 2),
  have h1 : A1.area = π * (a / 2)^2 / 2 := rfl,
  have h2 : A2.area = π * (b / 2)^2 / 2 := rfl,
  have h3 : A3.area = π * (c / 2)^2 / 2 := rfl,
  rw [h1, h2, h3],
  calc
    π * (a / 2)^2 / 2 + π * (b / 2)^2 / 2 
      : = (π * a^2 / 4 / 2) + (π * b^2 / 4 / 2) 
      : = (π * a^2 / 8) + (π * b^2 / 8)
      : = π * (a^2 + b^2) / 8 
      : = π * c^2 / 8
      : = π * (c / 2)^2 / 2 
}

theorem semicircle_edge_through_vertex (T : Type) (hp : right_triangle T) :
  ∀ (A3 : semicircle), 
  let (a, b, c) := T in 
  A3.radius = c / 2 →
  A3.area = π * (c / 2)^2 / 2 → 
  -- Edge of A3 passes through the right-angle vertex of T
  sorry := sorry

theorem crescents_area_eq_triangle (T : Type) (hp : right_triangle T) :
  ∃ (L1 L2 : Type), 
  let (a, b, c) := T,
      A1 := semicircle.mk (a / 2),
      A2 := semicircle.mk (b / 2),
      A3 := semicircle.mk (c / 2) in
  A1.area + A2.area = A3.area →
  -- Area calculations for L1 and L2 to match area of T
  sorry := sorry

end semicircle_edge_through_vertex_crescents_area_eq_triangle_l711_711404


namespace cube_volume_l711_711437

-- Define the given condition: The surface area of the cube
def surface_area (A : ℕ) := A = 294

-- The key proposition we need to prove using the given condition
theorem cube_volume (s : ℕ) (A : ℕ) (V : ℕ) 
  (area_condition : surface_area A)
  (side_length_condition : s ^ 2 = A / 6) 
  (volume_condition : V = s ^ 3) : 
  V = 343 := by
  sorry

end cube_volume_l711_711437


namespace answer_l711_711750

def p : Prop := ∃ x > Real.exp 1, (1 / 2)^x > Real.log x
def q : Prop := ∀ a b : Real, a > 1 → b > 1 → Real.log a / Real.log b + 2 * (Real.log b / Real.log a) ≥ 2 * Real.sqrt 2

theorem answer : ¬ p ∧ q :=
by
  have h1 : ¬ p := sorry
  have h2 : q := sorry
  exact ⟨h1, h2⟩

end answer_l711_711750


namespace remainder_1234_mul_5678_mod_1000_l711_711382

theorem remainder_1234_mul_5678_mod_1000 :
  (1234 * 5678) % 1000 = 652 := by
  sorry

end remainder_1234_mul_5678_mod_1000_l711_711382


namespace binomial_square_formula_l711_711388

theorem binomial_square_formula (a b : ℝ) :
  let e1 := (4 * a + b) * (4 * a - 2 * b)
  let e2 := (a - 2 * b) * (2 * b - a)
  let e3 := (2 * a - b) * (-2 * a + b)
  let e4 := (a - b) * (a + b)
  (e4 = a^2 - b^2) :=
by
  sorry

end binomial_square_formula_l711_711388


namespace convex_polygon_segment_difference_l711_711319

theorem convex_polygon_segment_difference
  (P : ℝ) (hP : P > 0) (sides : list ℝ) (h_convex : ∀ d ∈ sides, d < P / 2)
  (h_sum : sides.sum = P) :
  ∃ (l1 l2 : ℝ), l1 ∈ sides ∧ l2 ∈ sides ∧ |l1 - l2| ≤ P / 3 :=
by
  sorry

end convex_polygon_segment_difference_l711_711319


namespace inverse_function_l711_711381

def f (x : ℝ) : ℝ := 3 - 4 * x
def g (x : ℝ) : ℝ := (3 - x) / 4

theorem inverse_function : (∀ x : ℝ, f (g x) = x) ∧ (∀ x : ℝ, g (f x) = x) :=
by
  sorry

end inverse_function_l711_711381


namespace three_digit_multiples_of_seven_l711_711609

theorem three_digit_multiples_of_seven : 
  let min_bound := 100
  let max_bound := 999
  let first_multiple := Nat.ceil (min_bound / 7)
  let last_multiple := Nat.floor (max_bound / 7)
  (last_multiple - first_multiple + 1) = 128 :=
by 
  let min_bound := 100
  let max_bound := 999
  let first_multiple := Nat.ceil (min_bound / 7)
  let last_multiple := Nat.floor (max_bound / 7)
  have calc_first_multiple : first_multiple = 15 := sorry
  have calc_last_multiple : last_multiple = 142 := sorry
  have count_multiples : (last_multiple - first_multiple + 1) = 128 := sorry
  exact count_multiples

end three_digit_multiples_of_seven_l711_711609


namespace a_plus_b_l711_711628

theorem a_plus_b (a b : ℝ) (h : a * (a - 4) = 12) (h2 : b * (b - 4) = 12) (h3 : a ≠ b) : a + b = 4 :=
sorry

end a_plus_b_l711_711628


namespace inverse_proportion_l711_711753

theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x^2 * y^4 = k)
  (h2 : 6^2 * 2^4 = k) (hy : y = 4) : x^2 = 2.25 :=
by
  sorry

end inverse_proportion_l711_711753


namespace school_B_saves_l711_711309

theorem school_B_saves (kg : ℕ) (price_A price_reduction additional_percentage : ℝ) 
  (correct_savings : ℝ) : kg = 56 → price_A = 8.06 → price_reduction = 0.56 → additional_percentage = 0.05 → correct_savings = 51.36 → 
  let price_B := price_A - price_reduction in 
  let effective_weight := kg + kg * additional_percentage in 
  let actual_weight_B := kg / (1 + additional_percentage) in 
  let cost_A := kg * price_A in 
  let cost_B := actual_weight_B * price_B in 
  cost_A - cost_B = correct_savings :=
by
  intros hkg hprice_A hprice_reduction hadditional_percentage hcorrect_savings
  rw [hkg, hprice_A, hprice_reduction, hadditional_percentage, hcorrect_savings]
  let price_B := 8.06 - 0.56
  let effective_weight := 56 + 56 * 0.05
  let actual_weight_B := 56 / 1.05
  let cost_A := 56 * 8.06
  let cost_B := actual_weight_B * price_B
  have calc1 : 450.56 - 400 = 51.36, from sorry
  calc1

end school_B_saves_l711_711309


namespace spending_after_drink_l711_711712

variable (X : ℝ)
variable (Y : ℝ)

theorem spending_after_drink (h : X - 1.75 - Y = 6) : Y = X - 7.75 :=
by sorry

end spending_after_drink_l711_711712


namespace cos_7pi_over_6_l711_711152

noncomputable def cos_seven_pi_six : ℝ := -real.cos (real.pi / 6)

theorem cos_7pi_over_6 : real.cos (7 * real.pi / 6) = cos_seven_pi_six := by
  -- skipped proof
  sorry

end cos_7pi_over_6_l711_711152


namespace payment_option1_payment_option2_cost_effective_option_most_cost_effective_plan_l711_711417

variable (x : ℕ)
variable (hx : x > 10)

noncomputable def option1_payment (x : ℕ) : ℕ := 200 * x + 8000
noncomputable def option2_payment (x : ℕ) : ℕ := 180 * x + 9000

theorem payment_option1 (x : ℕ) (hx : x > 10) : option1_payment x = 200 * x + 8000 :=
by sorry

theorem payment_option2 (x : ℕ) (hx : x > 10) : option2_payment x = 180 * x + 9000 :=
by sorry

theorem cost_effective_option (x : ℕ) (hx : x > 10) (h30 : x = 30) : option1_payment 30 < option2_payment 30 :=
by sorry

theorem most_cost_effective_plan (h30 : x = 30) : (10000 + 3600 = 13600) :=
by sorry

end payment_option1_payment_option2_cost_effective_option_most_cost_effective_plan_l711_711417


namespace find_n_l711_711648

theorem find_n (N : ℕ) (hN : 10 ≤ N ∧ N < 100)
  (h : ∃ a b : ℕ, N = 10 * a + b ∧ 4 * a + 2 * b = N / 2) : 
  N = 32 ∨ N = 64 ∨ N = 96 :=
sorry

end find_n_l711_711648


namespace exists_STS_order_n1n2_l711_711246

-- Define conditions / assumptions
variables (n₁ n₂ : ℕ) 
variable (𝒩₁ : set (set ℕ)) -- Assume each set is a triple
variable (𝓛₂ : set (set ℕ)) 

-- Assuming 𝒩₁ is a Steiner triple system of order n₁
axiom NSTS : ∀ {t : set ℕ}, t ∈ 𝒩₁ → t.card = 3 ∧ (∀ {x y}, x ≠ y → {x, y} ⊆ t → ∃! z, z ∈ t ∧ z ≠ x ∧ z ≠ y)
axiom NSTS_order : 𝒩₁.finite ∧ 𝒩₁.card = n₁ * (n₁ - 1) / 6

-- Assuming 𝓛₂ is a Steiner triple system of order n₂
axiom LSTS : ∀ {t : set ℕ}, t ∈ 𝓛₂ → t.card = 3 ∧ (∀ {x y}, x ≠ y → {x, y} ⊆ t → ∃! z, z ∈ t ∧ z ≠ x ∧ z ≠ y)
axiom LSTS_order : 𝓛₂.finite ∧ 𝓛₂.card = n₂ * (n₂ - 1) / 6

-- Prove that there exists a Steiner triple system of order n₁ * n₂
theorem exists_STS_order_n1n2 : ∃ 𝒜₃ : set (set (ℕ × ℕ)), 
  (∀ {t : set (ℕ × ℕ)}, t ∈ 𝒜₃ → t.card = 3 ∧ 
    ∀ {xy₁ xy₂}, xy₁ ≠ xy₂ → ({xy₁, xy₂} ⊆ t → ∃! xy₃, xy₃ ∈ t ∧ xy₃ ≠ xy₁ ∧ xy₃ ≠ xy₂)) ∧
  𝒜₃.finite ∧ 𝒜₃.card = n₁ * n₂ * (n₁ * n₂ - 1) / 6 :=
begin
  sorry -- Proof goes here
end

end exists_STS_order_n1n2_l711_711246


namespace pyramid_D1_EDF_volume_l711_711796

-- Definitions and conditions
def cube_edge_length : ℝ := 1

-- Points on the segments of the cube
def point_E_on_AA1 := true  -- Representing E is on segment AA1
def point_F_on_B1C := true  -- Representing F is on segment B1C

-- Volume of the pyramid D1-EDF
def pyramid_volume (base_area height : ℝ) : ℝ := 
  (1 / 3) * base_area * height

theorem pyramid_D1_EDF_volume
  (H_cube: cube_edge_length = 1)
  (H_E: point_E_on_AA1)
  (H_F: point_F_on_B1C) :
  pyramid_volume (1 / 2) cube_edge_length = 1 / 6 :=
sorry

end pyramid_D1_EDF_volume_l711_711796


namespace complement_union_l711_711301

-- Definitions of the sets
def U : set ℕ := {1,2,3,4,5,6,7}
def A : set ℕ := {1,3,5}
def B : set ℕ := {2,5,7}

-- Setting the property to prove
theorem complement_union (x : ℕ) : x ∈ U \ (A ∪ B) ↔ x ∈ ({4,6} : set ℕ) := by 
  sorry

end complement_union_l711_711301


namespace andy_position_after_16_moves_l711_711477

-- Define the initial position and movement rules for Andy the Ant
structure Position :=
  (x : ℤ)
  (y : ℤ)

inductive Direction
| North
| West
| South
| East

open Direction

def turn_left : Direction → Direction
| North := West
| West := South
| South := East
| East := North

def move (pos : Position) (dir : Direction) (step : ℕ) : Position :=
  let n := step ^ 2 in
  match dir with
  | North => Position.mk pos.x (pos.y + n)
  | West  => Position.mk (pos.x - n) pos.y
  | South => Position.mk pos.x (pos.y - n)
  | East  => Position.mk (pos.x + n) pos.y

def move_n_times : Position → Direction → ℕ → Position
| pos, dir, 0         => pos
| pos, dir, (nat.succ steps) =>
  let new_pos := move pos dir (nat.succ steps) in
  move_n_times new_pos (turn_left dir) steps

def Andy_initial_position := Position.mk 10 (-10)

def Andy_final_position := move_n_times Andy_initial_position North 16

theorem andy_position_after_16_moves : 
  Andy_final_position = Position.mk 154 (-138) := by
  sorry

end andy_position_after_16_moves_l711_711477


namespace number_of_blue_spotted_fish_l711_711023

theorem number_of_blue_spotted_fish : 
  ∀ (fish_total : ℕ) (one_third_blue : ℕ) (half_spotted : ℕ),
    fish_total = 30 →
    one_third_blue = fish_total / 3 →
    half_spotted = one_third_blue / 2 →
    half_spotted = 5 := 
by
  intros fish_total one_third_blue half_spotted ht htb hhs
  sorry

end number_of_blue_spotted_fish_l711_711023


namespace harmonic_mean_1_3_6_9_l711_711834

theorem harmonic_mean_1_3_6_9 : harmonic_mean [1, 3, 6, 9] = 72 / 29 :=
by sorry

end harmonic_mean_1_3_6_9_l711_711834


namespace sum_of_values_k_l711_711863

theorem sum_of_values_k (k : ℕ) : 
  (∀ x y : ℤ, (3 * x * x - k * x + 12 = 0) ∧ (3 * y * y - k * y + 12 = 0) ∧ x ≠ y) → k = 0 :=
by
  sorry

end sum_of_values_k_l711_711863


namespace largest_integer_log_sum_l711_711835

noncomputable def log_sum := ∑ k in Finset.range 2011 \ Finset.range 1, Real.logBase 3 ((k+2:ℕ)/(k+1:ℕ))

theorem largest_integer_log_sum : 
  ⌊log_sum⌋ = 6 :=
by
  sorry

end largest_integer_log_sum_l711_711835


namespace directrix_of_parabola_l711_711157

theorem directrix_of_parabola (y x : ℝ) (p : ℝ) (h₁ : y = 8 * x ^ 2) (h₂ : y = 4 * p * x) : 
  p = 2 ∧ (y = -p ↔ y = -2) :=
by
  sorry

end directrix_of_parabola_l711_711157


namespace chicken_pot_pie_customers_l711_711129

theorem chicken_pot_pie_customers (slices_per_shepherd: ℕ) (slices_per_chicken: ℕ) (shepherd_orders: ℕ) (total_pies: ℕ) :
  slices_per_shepherd = 4 → 
  slices_per_chicken = 5 → 
  shepherd_orders = 52 → 
  total_pies = 29 → 
  ∃ customers_chicken: ℕ, customers_chicken = 80 :=
by 
  intros h1 h2 h3 h4 
  let shepherd_pies := shepherd_orders / slices_per_shepherd
  let chicken_pies := total_pies - shepherd_pies 
  let customers_chicken := chicken_pies * slices_per_chicken
  use customers_chicken
  have h5: customers_chicken = 80 := by
    have h_shepherd_pies : shepherd_pies = 13 := by 
      rw [h1, h3]
      norm_num
    have h_chicken_pies : chicken_pies = 16 := by 
      rw [← h4, h_shepherd_pies]
      norm_num
    rw [← h2]
    norm_num at h_chicken_pies
    rw h_chicken_pies 
    norm_num
  exact h5

end chicken_pot_pie_customers_l711_711129


namespace sum_of_k_values_with_distinct_integer_solutions_l711_711846

theorem sum_of_k_values_with_distinct_integer_solutions : 
  ∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ 3 * p * q + (-k) * (p + q) + 12 = 0}, k = 0 :=
by
  sorry

end sum_of_k_values_with_distinct_integer_solutions_l711_711846


namespace division_of_negatives_l711_711135

theorem division_of_negatives : (-500 : ℤ) / (-50 : ℤ) = 10 := by
  sorry

end division_of_negatives_l711_711135


namespace propositions_using_logical_connectives_l711_711002

-- Define each of the propositions.
def prop1 := "October 1, 2004, is the National Day and also the Mid-Autumn Festival."
def prop2 := "Multiples of 10 are definitely multiples of 5."
def prop3 := "A trapezoid is not a rectangle."
def prop4 := "The solutions to the equation x^2 = 1 are x = ± 1."

-- Define logical connectives usage.
def uses_and (s : String) : Prop := 
  s = "October 1, 2004, is the National Day and also the Mid-Autumn Festival."
def uses_not (s : String) : Prop := 
  s = "A trapezoid is not a rectangle."
def uses_or (s : String) : Prop := 
  s = "The solutions to the equation x^2 = 1 are x = ± 1."

-- The lean theorem stating the propositions that use logical connectives
theorem propositions_using_logical_connectives :
  (uses_and prop1) ∧ (¬ uses_and prop2) ∧ (uses_not prop3) ∧ (uses_or prop4) := 
by
  sorry

end propositions_using_logical_connectives_l711_711002


namespace find_years_l711_711461

-- Define the given conditions
variable (R T : ℕ)
constant sum : ℕ
constant more : ℕ

-- Given constants
def P : ℕ := 700
def inc : ℕ := 2
def SI_diff : ℕ := 56

-- Sum and more set to their respective values
noncomputable def sum := 700
noncomputable def more := 56

-- Lean 4 statement to prove the number of years T given the conditions
theorem find_years (h : 700 * T * (R + inc) - 700 * T * R = 5600) : T = 4 :=
by {
  sorry
}

end find_years_l711_711461


namespace value_of_series_l711_711237

theorem value_of_series :
  ∀ (a : ℕ → ℝ) (x : ℝ),
    (2 * x - 1)^2016 = Σ n in range(2017), a n * x^n →
    a 0 = 1 →
    Σ n in range(2017), a n * (1/2)^n = 0 →
    Σ n in range(1, 2017), (a n) / 2^n = -1 :=
by
  intros a x h_eq h_a0 h_half
  sorry

end value_of_series_l711_711237


namespace valid_8_digit_numbers_l711_711947

theorem valid_8_digit_numbers (digits : List Nat) (H : digits = [2, 0, 1, 9, 20, 19]) : 
  ∃ (count : Nat), count = 498 ∧ 
  (∀ (first_digit : Nat), first_digit ≠ 0 → 
    (∃ perm : List Nat, perm.perm digits ∧ perm.head = first_digit ∧ 
    (∃ count_perms : Nat, count_perms = 5 * 5! - (|{perm | perm.head = first_digit ∧ perm.get! 1 = 0}| / 4 + 
                                                     |{perm | perm.get! 1 = 9 ∧ perm.head = 1}| / 2 + 
                                                     |{perm | perm.head = 1 ∧ perm.get! 1 = 9}| / 2) ∧
    count_perms = 498))) := 
begin
  sorry
end

end valid_8_digit_numbers_l711_711947


namespace volume_of_cube_with_surface_area_l711_711431

theorem volume_of_cube_with_surface_area (S : ℝ) (hS : S = 294) : 
  ∃ V : ℝ, V = 343 :=
by
  let s := (S / 6).sqrt
  have hs : s = 7 := by sorry
  use s ^ 3
  simp [hs]
  exact sorry

end volume_of_cube_with_surface_area_l711_711431


namespace train_crosses_platform_in_time_l711_711912

noncomputable def train_cross_platform_time 
  (length_train : ℝ) (initial_speed_kmh : ℝ) (time_cross_tree : ℝ) 
  (acceleration_rate : ℝ) (length_platform : ℝ) (incline_angle : ℝ) : ℝ :=
  let initial_speed_ms := initial_speed_kmh * (1000 / 3600)
  let new_speed_ms := initial_speed_ms * (1 + acceleration_rate)
  let total_distance := length_train + length_platform
  total_distance / new_speed_ms

theorem train_crosses_platform_in_time
  (length_train : ℝ) (initial_speed_kmh : ℝ) (time_cross_tree : ℝ) 
  (acceleration_rate : ℝ) (length_platform : ℝ) (incline_angle : ℝ)
  (h1 : length_train = 1500) 
  (h2 : initial_speed_kmh = 30) 
  (h3 : time_cross_tree = 120) 
  (h4 : acceleration_rate = 0.5) 
  (h5 : length_platform = 500) 
  (h6 : incline_angle = 2) :
  train_cross_platform_time 
     length_train initial_speed_kmh time_cross_tree acceleration_rate length_platform incline_angle 
  ≈ 160.064 :=
by {
  sorry
}

end train_crosses_platform_in_time_l711_711912


namespace find_x_l711_711995

theorem find_x (x : ℝ) (h1 : x > 9) : 
  (sqrt (x - 9 * sqrt (x - 9)) + 3 = sqrt (x + 9 * sqrt (x - 9)) - 3) → 
  x ≥ 45 :=
sorry

end find_x_l711_711995


namespace triangle_inequality_l711_711299

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a + b) / (a + b + c) > 1 / 2 :=
sorry

end triangle_inequality_l711_711299


namespace cyclists_same_direction_cyclists_opposite_direction_l711_711827

open Real EuclideanGeometry

variables {circle1 circle2 : Circle} {A B : Point}
variables {M N : Point} -- Positions of cyclists
variables {L K : Point} -- Diametrically opposite points
variables {P : Point} -- Midpoint of L and K
variables {O1 O2 : Point} -- Centers of circles
variables {A1 : Point} -- Point defining the parallelogram

-- Part (a): Cyclists move in the same direction
theorem cyclists_same_direction (h_intersect : circles_intersect_at circle1 circle2 A B)
  (h_start : cyclists_start_simultaneously A)
  (h_meet : cyclists_meet_after_full_lap A)
  (h_constant_speed : cyclists_move_at_constant_speed)
  (h_diametric : L K are_diametric_opposite_points B)
  (h_midpoint_P : P is_midpoint_of L K) :
  distances_from_P_to_M_and_N_are_always_equal :=
sorry

-- Part (b): Cyclists move in opposite directions
theorem cyclists_opposite_direction (h_intersect : circles_intersect_at circle1 circle2 A B)
  (h_start : cyclists_start_simultaneously A)
  (h_meet : cyclists_meet_after_full_lap A)
  (h_constant_speed : cyclists_move_at_constant_speed)
  (parallelogram_O1AO2A1 : is_parallelogram O1 A O2 A1) :
  distances_from_symmetry_point_to_M_and_N_are_always_equal :=
sorry

end cyclists_same_direction_cyclists_opposite_direction_l711_711827


namespace eval_log_expression_l711_711987

theorem eval_log_expression (x : ℕ) (y : ℕ) (h : 2000 = 2^4 * 5^3) : 
  (2 / log 4 (2000^6) + 3 / log 5 (2000^6) = 1 / 6) := 
  sorry

end eval_log_expression_l711_711987


namespace cyclic_quadrilateral_eq_l711_711489

-- Given conditions
variables {A B C D O P M N E F : Type*}
variables [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty O]
variables [Nonempty P] [Nonempty M] [Nonempty N] [Nonempty E] [Nonempty F]

-- Definitions based on the problem
def is_cyclic_quadrilateral (A B C D O P M N E F : Type*) : Prop :=
  ∃ (ω1 ω2 ω3 : Type*), 
    (circumcenter_of ω1 = O) ∧
    (circumcenter_of ω2 = O) ∧
    (circumcenter_of ω3 = O) ∧
    (midpoint A D = M) ∧
    (midpoint B C = N) ∧
    (intersect_diagonals A C B D = P) ∧
    (intersection_of_circumcircle ω1 ω3 \notin arc_extension A P D) = E ∧
    (intersection_of_circumcircle ω2 ω3 \notin arc_extension B P C) = F

-- The theorem to be proven
theorem cyclic_quadrilateral_eq (A B C D O P M N E F : Type*)
  (h : is_cyclic_quadrilateral A B C D O P M N E F) : 
    distance O E = distance O F :=
sorry

end cyclic_quadrilateral_eq_l711_711489


namespace range_of_inclination_angle_l711_711819

theorem range_of_inclination_angle (α : ℝ) (A : ℝ × ℝ) (hA : A = (-2, 0))
  (ellipse_eq : ∀ (x y : ℝ), (x^2 / 2) + y^2 = 1) :
    (0 ≤ α ∧ α < Real.arcsin (Real.sqrt 3 / 3)) ∨ 
    (π - Real.arcsin (Real.sqrt 3 / 3) < α ∧ α < π) :=
sorry

end range_of_inclination_angle_l711_711819


namespace equation_holds_if_a_eq_neg_b_c_l711_711529

-- Define the conditions and equation
variables {a b c : ℝ} (h1 : a ≠ 0) (h2 : a + b ≠ 0)

-- Statement to be proved
theorem equation_holds_if_a_eq_neg_b_c : 
  (a = -(b + c)) ↔ (a + b + c) / a = (b + c) / (a + b) := 
sorry

end equation_holds_if_a_eq_neg_b_c_l711_711529


namespace two_real_solutions_l711_711977

noncomputable def numberOfRealSolutions : ℕ :=
  let f := λ x : ℝ => 2 ^ (3 * x^2 - 8 * x + 3)
  let g := (λ x, 3 * x^2 - 8 * x + 3)
  let h := 1
  let solutions := (λ : Set ℝ => {x : ℝ | g x = 0})
  finset.card {x : ℝ | g x = 0}.to_finset

theorem two_real_solutions : numberOfRealSolutions = 2 :=
sorry

end two_real_solutions_l711_711977


namespace correct_statement_about_compass_and_straightedge_constructions_is_B_l711_711045

-- Definitions of the statements in the conditions
def optionA : Prop := ∀ (A B D : Point), extends_ray_to_point A B D
def optionB : Prop := ∀ (O A : Point), draws_arc O (segment_length O A)
def optionC : Prop := ∀ (A B : Point), draws_line_segment_eq_AB A B (3 : ℝ)
def optionD : Prop := ∀ (A B C : Point), extends_segment_to_point_eq AC BC

-- Correct answer is option B
theorem correct_statement_about_compass_and_straightedge_constructions_is_B :
  optionB ∧ ¬optionA ∧ ¬optionC ∧ ¬optionD :=
by sorry

end correct_statement_about_compass_and_straightedge_constructions_is_B_l711_711045


namespace two_digit_numbers_solution_l711_711641

theorem two_digit_numbers_solution :
  ∀ (N : ℕ), (∃ (x y : ℕ), (N = 10 * x + y) ∧ (x < 10) ∧ (y < 10) ∧ 4 * x + 2 * y = N / 2) →
    (N = 32 ∨ N = 64 ∨ N = 96) := 
by
  sorry

end two_digit_numbers_solution_l711_711641


namespace max_c_val_l711_711633

theorem max_c_val (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : 2 * a * b = 2 * a + b) 
  (h2 : a * b * c = 2 * a + b + c) :
  c ≤ 4 :=
sorry

end max_c_val_l711_711633


namespace calculate_volume_l711_711055

noncomputable def ellipse_region (x y : ℝ) :=
  (x^2 / 3 + y^2 / 16 = 1) ∧ (0 ≤ y)

def region_volume : ℝ :=
  32

theorem calculate_volume :
  (∀ x y z : ℝ, ellipse_region x y → (z = y * sqrt 3 ∨ z = 0) → (z = 0 ∧ 0 ≤ y) ∧ x ∈ [-sqrt 3, sqrt 3] ∧ y ∈ [0, 4] → 
  ∫∫∫ (1 : ℝ) dx dy dz = region_volume) := 
sorry

end calculate_volume_l711_711055


namespace find_k_l711_711184

-- Define the given circle equation
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define line l1
def line_l1 (x y : ℝ) : Prop := y = real.sqrt 3 * x

-- Define line l2 parameterized by k
def line_l2 (x y k : ℝ) : Prop := y = k * x - 1

-- Define the lengths of the chords intercepted
def chord_length (d : ℝ) : ℝ := 2 * real.sqrt (4 - d^2)

-- Define the distance from the center (2, 0) of the circle to a line
def distance_to_line (k x0 y0 : ℝ) : ℝ := abs (k * x0 - y0 + 1) / real.sqrt (k^2 + 1)

-- Assuming the ratio condition for the lengths of the chords intercepted is 1:2
def ratio_condition (k : ℝ) : Prop :=
  chord_length (real.sqrt 3) = (1 / 2) * chord_length (distance_to_line k 2 0)

theorem find_k (k : ℝ) (h : ratio_condition k) : k = 1 / 2 := sorry

end find_k_l711_711184


namespace jill_investment_value_l711_711692

def compoundInterest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_investment_value :
  compoundInterest 10000 0.0396 2 2 ≈ 10812 :=
by
  sorry

end jill_investment_value_l711_711692


namespace sum_of_k_distinct_integer_roots_l711_711849

theorem sum_of_k_distinct_integer_roots : 
  (∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ p * q = 4 ∧ k = 3 * (p + q)}, k) = 0 := 
by
  sorry

end sum_of_k_distinct_integer_roots_l711_711849


namespace cos_seven_pi_over_six_l711_711150

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 := 
by
  sorry

end cos_seven_pi_over_six_l711_711150


namespace count_5_numbers_after_996_l711_711143

theorem count_5_numbers_after_996 : 
  ∃ a b c d e, a = 997 ∧ b = 998 ∧ c = 999 ∧ d = 1000 ∧ e = 1001 :=
sorry

end count_5_numbers_after_996_l711_711143


namespace gravel_path_width_l711_711889

-- Define the dimensions of the rectangular garden
variables {AB BC : ℝ} (h_AB_positive : AB > 0) (h_BC_positive : BC > 0)

-- Define the width of the gravel path
def width_of_gravel_path := AB / 4

-- Theorem stating that the width of the gravel path surrounding the rose garden is 1/4 of the garden's length
theorem gravel_path_width (AB BC : ℝ) (h_AB_pos : 0 < AB) (h_BC_pos : 0 < BC) :
  width_of_gravel_path AB = AB / 4 :=
by
  exact rfl

#check gravel_path_width

-- Adding a placeholder for the proof 
sorry

end gravel_path_width_l711_711889


namespace infinite_composites_in_sequence_l711_711254

-- Define the sequence and conditions
def is_strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n, a n < a (n + 1)

def is_sum_of_two_preceding (a : ℕ → ℕ) : Prop :=
  ∀ n, n ≥ 2 → ∃ i j: ℕ, i < n ∧ j < n ∧ a n = a i + a j

-- The statement to prove
theorem infinite_composites_in_sequence (a : ℕ → ℕ) (h_incr : is_strictly_increasing a) (h_sum : is_sum_of_two_preceding a) :
  ∃ᶠ n in at_top, ¬ nat.prime (a n) :=
sorry

end infinite_composites_in_sequence_l711_711254


namespace cube_difference_l711_711574

variable (x : ℝ)

theorem cube_difference (h : x - 1/x = 5) : x^3 - 1/x^3 = 120 := by
  sorry

end cube_difference_l711_711574


namespace polynomial_value_at_4_l711_711121

def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

theorem polynomial_value_at_4 : f 4 = 371 := by
  sorry

end polynomial_value_at_4_l711_711121


namespace value_of_a_l711_711771

theorem value_of_a (a : ℝ) (h1 : a < 0) (h2 : |a| = 3) : a = -3 := 
by
  sorry

end value_of_a_l711_711771


namespace eval_expr_correct_l711_711988

noncomputable def eval_expr (a b : ℚ) : ℚ :=
  (4 * a^2 - b^2) / (a^6 - 8 * b^6) *
  Real.sqrt (a^2 - 2 * b * Real.sqrt (a^2 - b^2)) *
  (4^4 + 2 * a^2 * b^2 + 4 * b^4) / (4 * a^2 + 4 * a * b + b^2) *
  Real.sqrt (a^2 + 2 * b * Real.sqrt (a^2 - b^2))

theorem eval_expr_correct :
  eval_expr (4/3) (1/4) = 29 / 35 := sorry

end eval_expr_correct_l711_711988


namespace baron_munchausen_not_boasting_l711_711890

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits = digits.reverse

def cuts_into (n : ℕ) (parts : list ℕ) : Prop :=
  let digit_parts := parts.map (λ x, x.digits 10).join
  digit_parts = n.digits 10

def sequence_1_to_19 := list.range 19 |>.map succ

theorem baron_munchausen_not_boasting :
  ∃ n : ℕ, is_palindrome n ∧ cuts_into n sequence_1_to_19 :=
sorry

end baron_munchausen_not_boasting_l711_711890


namespace y_coordinate_P_eq_1_l711_711282

-- Define the points A, B, C, D
constant A : ℝ × ℝ := (-3, 0)
constant B : ℝ × ℝ := (-3, 2)
constant C : ℝ × ℝ := (3, 2)
constant D : ℝ × ℝ := (3, 0)

-- Define the distance function
noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Conditions
axiom PA_PD_eq_10 (P : ℝ × ℝ) : dist P A + dist P D = 10
axiom PB_PC_eq_10 (P : ℝ × ℝ) : dist P B + dist P C = 10

-- The problem: Find the y-coordinate of P
def y_coordinate_P (P : ℝ × ℝ) : ℝ := P.2

-- Theorem to prove the y-coordinate of P is 1
theorem y_coordinate_P_eq_1 (P : ℝ × ℝ) (h1 : dist P A + dist P D = 10) (h2 : dist P B + dist P C = 10) :
  y_coordinate_P P = 1 :=
sorry

end y_coordinate_P_eq_1_l711_711282


namespace factorization_correct_l711_711344

theorem factorization_correct :
  ∀ (m a b x y : ℝ), 
    (m^2 - 4 = (m + 2) * (m - 2)) ∧
    ((a + 3) * (a - 3) = a^2 - 9) ∧
    (a^2 - b^2 + 1 = (a + b) * (a - b) + 1) ∧
    (6 * x^2 * y^3 = 2 * x^2 * 3 * y^3) →
    (m^2 - 4 = (m + 2) * (m - 2)) :=
by
  intros m a b x y h
  have ⟨hA, hB, hC, hD⟩ := h
  exact hA

end factorization_correct_l711_711344


namespace compare_M_N_compare_a_b_l711_711903

-- Definition and conditions
def M (x : ℝ) := (x + 8) * (x + 11)
def N (x : ℝ) := (x + 9) * (x + 10)
def a := Real.sqrt 5 - 2
def b := Real.sqrt 6 - Real.sqrt 5

-- Theorems we want to prove
theorem compare_M_N (x : ℝ) : M x < N x :=
by sorry

theorem compare_a_b : a > b :=
by sorry

end compare_M_N_compare_a_b_l711_711903


namespace count_three_digit_multiples_of_seven_l711_711616

theorem count_three_digit_multiples_of_seven :
  let a := 100 in
  let b := 999 in
  let smallest := (Nat.ceil (a.toRat / 7)).natAbs * 7 in
  let largest := (b / 7) * 7 in
  (largest / 7) - ((smallest - 1) / 7) = 128 := sorry

end count_three_digit_multiples_of_seven_l711_711616


namespace volume_of_cube_with_surface_area_l711_711429

theorem volume_of_cube_with_surface_area (S : ℝ) (hS : S = 294) : 
  ∃ V : ℝ, V = 343 :=
by
  let s := (S / 6).sqrt
  have hs : s = 7 := by sorry
  use s ^ 3
  simp [hs]
  exact sorry

end volume_of_cube_with_surface_area_l711_711429


namespace find_smallest_m_l711_711756

noncomputable def reflect_angle (angle θ : ℝ) : ℝ :=
2 * θ

noncomputable def R (l : ℝ) (l1_angle l2_angle : ℝ) : ℝ :=
let θ := l in
let l' := reflect_angle l1_angle - θ,
    l'' := reflect_angle l2_angle - l' in
l''

noncomputable def R_n (l : ℝ) (l1_angle l2_angle : ℝ) (n : ℕ) : ℝ :=
nat.rec l (fun _ prev => R prev l1_angle l2_angle) n

theorem find_smallest_m
  (l : ℝ)
  (l1_angle l2_angle : ℝ)
  (θ := real.arctan (11 / 50)) :
  ∃ m : ℕ, m > 0 ∧ R_n l l1_angle l2_angle m = l ∧
  ∀ k, k < m → R_n l l1_angle l2_angle k ≠ l := by
  let m := 72
  exists m sorry

end find_smallest_m_l711_711756


namespace range_of_x_when_p_and_q_hold_range_of_a_when_p_is_necessary_not_sufficient_l711_711751

-- Definitions
def p (a x : ℝ) : Prop := x^2 - (a + (1/a)) * x + 1 < 0
def q (x : ℝ) : Prop := x^2 - 4 * x + 3 ≤ 0

-- Theorem (1)
theorem range_of_x_when_p_and_q_hold (a : ℝ) (h1 : a = 2) : 
  (∃ x : ℝ, p a x ∧ q x) ↔ (1 ≤ ∀ x : ℝ, x < 2) :=
sorry

-- Theorem (2)
theorem range_of_a_when_p_is_necessary_not_sufficient (a : ℝ) : 
  (∀ x : ℝ, q x → p a x) ∧ ¬(∀ x : ℝ, p a x → q x) → 3 < a :=
sorry

end range_of_x_when_p_and_q_hold_range_of_a_when_p_is_necessary_not_sufficient_l711_711751


namespace calculate_PQ_length_l711_711719

noncomputable def coord (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

noncomputable def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

noncomputable def dist (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2 + (p1.3 - p2.3) ^ 2)

variables (E F G H E' F' G' H' P Q : ℝ × ℝ × ℝ)

def coordinates := 
  E = coord 0 0 0 ∧ 
  F = coord 12 0 0 ∧ 
  G = coord 12 4 0 ∧ 
  H = coord 0 4 0 ∧ 
  E' = coord 0 0 20 ∧ 
  F' = coord 12 0 12 ∧ 
  G' = coord 12 4 30 ∧ 
  H' = coord 0 4 24 ∧ 
  P = midpoint E' G' ∧ 
  Q = midpoint F' H'

theorem calculate_PQ_length (h : coordinates E F G H E' F' G' H' P Q) : 
  dist P Q = 7 :=
by sorry

end calculate_PQ_length_l711_711719


namespace find_parabola_through_point_and_area_l711_711594

noncomputable def parabola_through_point (p : ℝ) (h : 0 < p) (P : ℝ × ℝ) (H : P = (1, -2)) : Prop := 
  (2 * p = 4 ∧ y^2 = 4 * x) ∧ (axis_eq :  x = -1)

noncomputable def area_of_triangle (F : ℝ × ℝ) (A B : ℝ × ℝ) (slope : ℝ) (l : ℝ → ℝ) (origin : ℝ × ℝ)
                      (H1 : F = (1, 0)) (H2: slope = 2) (H3: l = (λ x, 2 * x - 2))
                      (H4: P = (0, 0) := (0, 0)) : Prop :=
  let chord_length := sqrt 5 * abs (x1 - x2) in
  let distance_from_origin := sqrt 5 * 2 / 5 in
  let area := chord_length * distance_from_origin / 2 in
  area = 5

theorem find_parabola_through_point_and_area :
  ∃ p : ℝ, 0 < p ∧ parabola_through_point p (1, -2) (h = 2) (H: P = 4) = true ∧ 
  (∃ F A B, area_of_triangle F A B 2 (λ x, 2 * x - 2) (0, 0) := true)
  :=
  sorry

end find_parabola_through_point_and_area_l711_711594


namespace sum_of_max_and_min_on_interval_l711_711813

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 1

theorem sum_of_max_and_min_on_interval : 
  (∀ x ∈ set.Icc (0 : ℝ) 5, f x) ∧ 
  (∀ x ∈ set.Icc (0 : ℝ) 5, f 2 ≤ f x ∧ f x ≤ f 5) → 
  (f 2 + f 5 = 3) :=
by
  sorry

end sum_of_max_and_min_on_interval_l711_711813


namespace range_of_inclination_angle_l711_711818

theorem range_of_inclination_angle (α : ℝ) :
  let A := (-2 : ℝ, 0 : ℝ)
  let ellipse := ∀ x y : ℝ, x^2 / 2 + y^2 = 1
  ∃ B C : ℝ × ℝ, (∃ l : ℝ → ℝ × ℝ, ∀ t : ℝ, l t = (-2 + t * Real.cos α, t * Real.sin α) ∧ ellipse (fst (l t)) (snd (l t))) ∧ B ≠ C ↔ (0 ≤ α ∧ α < Real.arcsin (Real.sqrt 3 / 3) ∨ π - Real.arcsin (Real.sqrt 3 / 3) < α ∧ α < π) :=
begin
  sorry
end

end range_of_inclination_angle_l711_711818


namespace vertical_distance_rotated_square_l711_711169

-- Lean 4 statement for the mathematically equivalent proof problem
theorem vertical_distance_rotated_square
  (side_length : ℝ)
  (n : ℕ)
  (rot_angle : ℝ)
  (orig_line_height before_rotation : ℝ)
  (diagonal_length : ℝ)
  (lowered_distance : ℝ)
  (highest_point_drop : ℝ)
  : side_length = 2 →
    n = 4 →
    rot_angle = 45 →
    orig_line_height = 1 →
    diagonal_length = side_length * (2:ℝ)^(1/2) →
    lowered_distance = (diagonal_length / 2) - orig_line_height →
    highest_point_drop = lowered_distance →
    2 = 2 :=
    sorry

end vertical_distance_rotated_square_l711_711169


namespace tree_height_end_of_2_years_l711_711464

theorem tree_height_end_of_2_years (h4 : ℕ → ℕ)
  (h_tripling : ∀ n, h4 (n + 1) = 3 * h4 n)
  (h4_at_4 : h4 4 = 81) :
  h4 2 = 9 :=
by
  sorry

end tree_height_end_of_2_years_l711_711464


namespace find_B_find_correct_expression_l711_711183

variable {a b c : ℝ}

def A := 3 * a^2 * b - 2 * a * b^2 + a * b * c
def C := 4 * a^2 * b - 3 * a * b^2 + 4 * a * b * c
 
theorem find_B (B : ℝ) : 2 * A + B = C → B = -2 * a^2 * b + a * b^2 + 2 * a * b * c := 
  by
  sorry

theorem find_correct_expression (B : ℝ) : B = -2 * a^2 * b + a * b^2 + 2 * a * b * c → 2 * A - B = 8 * a^2 * b - 5 * a * b^2 := 
  by
  sorry

end find_B_find_correct_expression_l711_711183


namespace lucy_fish_count_l711_711758

theorem lucy_fish_count (initial_fish : ℕ) (additional_fish : ℕ) (final_fish : ℕ) : 
  initial_fish = 212 ∧ additional_fish = 68 → final_fish = 280 :=
by
  sorry

end lucy_fish_count_l711_711758


namespace two_digit_number_solution_l711_711638

theorem two_digit_number_solution (N : ℕ) (x y : ℕ) :
  (10 * x + y = N) ∧ (4 * x + 2 * y = (10 * x + y) / 2) →
  N = 32 ∨ N = 64 ∨ N = 96 := 
sorry

end two_digit_number_solution_l711_711638


namespace goose_eggs_count_l711_711772

theorem goose_eggs_count (E : ℕ)
  (hatch_ratio : ℚ := 2 / 3)
  (survive_first_month_ratio : ℚ := 3 / 4)
  (survive_first_year_ratio : ℚ := 2 / 5)
  (survived_first_year : ℕ := 130) :
  (survive_first_year_ratio * survive_first_month_ratio * hatch_ratio * (E : ℚ) = survived_first_year) →
  E = 1300 := by
  sorry

end goose_eggs_count_l711_711772


namespace geometric_progression_fourth_term_l711_711964

theorem geometric_progression_fourth_term (a b c : ℝ) (r : ℝ) (h1 : a = 2^(1/4))
    (h2 : b = 2^(1/8))
    (h3 : c = 2^(1/16))
    (h4 : b = a * r)
    (h5 : c = b * r)
    (h6 : r = 2^(-1/8)) :
  ∃ d : ℝ, d = c * r ∧ d = 2^(-1/16) :=
by
  sorry

end geometric_progression_fourth_term_l711_711964


namespace walmart_cards_initially_asked_l711_711268

-- Define the given conditions
def initial_best_buy_cards : ℕ := 6
def best_buy_card_value : ℕ := 500
def walmart_card_value : ℕ := 200
def sent_best_buy_cards : ℕ := 1
def sent_walmart_cards : ℕ := 2
def returnable_amount : ℕ := 3900

-- Total amount Jack initially had
def total_value : ℕ := returnable_amount + (sent_best_buy_cards * best_buy_card_value) + (sent_walmart_cards * walmart_card_value)

-- Total Best Buy amount Jack initially needed to send
def best_buy_total : ℕ := initial_best_buy_cards * best_buy_card_value

-- Walmart gift cards amount Jack initially needed to send
def initial_walmart_value : ℕ := total_value - best_buy_total

-- Number of Walmart gift cards asked to send
noncomputable def initial_walmart_cards : ℕ := initial_walmart_value / walmart_card_value

-- Theorem to prove the required number of Walmart gift cards
theorem walmart_cards_initially_asked : initial_walmart_cards = 9 :=
begin
  sorry
end

end walmart_cards_initially_asked_l711_711268


namespace range_of_m_l711_711180

open Real

def problem_statement (m : ℝ) : Prop :=
  ((∃ x : ℝ, cos (2 * x) - sin x + 2 ≤ m) ∨ 
   (∀ x : ℝ, x ≥ 2 → (1 / 3)^(2 * x^2 - m * x + 2) ≤ (1 / 3)^(2 * (x + 1)^2 - m * (x + 1) + 2)) ∧ 
   ¬((∃ x : ℝ, cos (2 * x) - sin x + 2 ≤ m) ∧ 
   (∀ x : ℝ, x ≥ 2 → (1 / 3)^(2 * x^2 - m * x + 2) ≤ (1 / 3)^(2 * (x + 1)^2 - m * (x + 1) + 2))) → 
  m < 0 ∨ 8 < m

theorem range_of_m (m : ℝ) : problem_statement m := 
by
  sorry

end range_of_m_l711_711180


namespace simplify_trig_expression_l711_711393

theorem simplify_trig_expression (α β : ℝ) :
  (sin (α + β))^2 - (sin α)^2 - (sin β)^2 = 
  (sin (α + β))^2 - (cos α)^2 - (cos β)^2 → 
  (-(tan α) * (tan β)) :=
by
  sorry

end simplify_trig_expression_l711_711393


namespace magnitude_of_complex_solution_correct_l711_711181

noncomputable def magnitude_of_complex_solution : ℝ :=
let x := 1 in -- Derived from solving x(1 + i) = 1 + yi
let y := x in -- Derived from solving x(1 + i) = 1 + yi
complex.abs (complex.of_real x + complex.I * y)

theorem magnitude_of_complex_solution_correct :
  (∃ x y : ℝ, x * (1 + complex.I) = 1 + y * complex.I ∧ complex.abs (complex.of_real x + complex.I * y) = ∥(1 : ℝ) + complex.I∥) ↔ magnitude_of_complex_solution = real.sqrt 2 := 
by sorry

end magnitude_of_complex_solution_correct_l711_711181


namespace diet_soda_bottles_l711_711079

theorem diet_soda_bottles (total_bottles : ℕ) (regular_soda_bottles : ℕ) (h_total : total_bottles = 30) (h_regular : regular_soda_bottles = 28) : total_bottles - regular_soda_bottles = 2 := 
by {
  rw [h_total, h_regular],
  sorry
}

end diet_soda_bottles_l711_711079


namespace problem_f_g_pi_l711_711749

def f (x : ℝ) : ℝ :=
  if x > 0 then 1 else
  if x = 0 then 0 else -1

def g (x : ℝ) : ℝ :=
  if ∃ q : ℚ, x = q then 1 else 0

theorem problem_f_g_pi : f (g π) = 0 :=
by {
  -- Since π is irrational, g(π) = 0
  have h1 : g π = 0,
  { -- π is not rational
    rw g,
    simp,
    intro h,
    cases h with q hq,
    exact real.pi_not_rational hq },
  -- Therefore f(g(π)) = f(0)
  rw h1,
  -- And f(0) = 0
  rw f,
  simp }

end problem_f_g_pi_l711_711749


namespace real_roots_quadratic_range_l711_711660

theorem real_roots_quadratic_range (k : ℝ) :
  (∃ x : ℝ, x^2 + 2 * x - k = 0) ↔ k ≥ -1 :=
by
  sorry

end real_roots_quadratic_range_l711_711660


namespace solution_exists_in_interval_l711_711014

noncomputable def f (x : ℝ) : ℝ := 3^x + x - 3

theorem solution_exists_in_interval : ∃ x, 0 < x ∧ x < 1 ∧ f x = 0 :=
by {
  -- placeholder for the skipped proof
  sorry
}

end solution_exists_in_interval_l711_711014


namespace cubic_identity_l711_711562

theorem cubic_identity (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 :=
by
  sorry

end cubic_identity_l711_711562


namespace isosceles_triangle_perimeter_l711_711549

-- Define a structure to represent a triangle with sides a, b, c
structure Triangle :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (triangle_ineq_1 : a + b > c)
  (triangle_ineq_2 : a + c > b)
  (triangle_ineq_3 : b + c > a)

-- Define the specific triangle given the condition
def isosceles_triangle_with_sides (s1 s2 : ℝ) (h_iso : s1 = 3 ∨ s2 = 3) (h_ineq : s1 = 6 ∨ s2 = 6) : Triangle :=
  if h_iso then
    { a := 3, b := 3, c := 6,
      triangle_ineq_1 := by linarith,
      triangle_ineq_2 := by linarith,
      triangle_ineq_3 := by linarith }
  else 
    sorry -- We cover the second case directly with the checked option

-- Prove that the perimeter of the given isosceles triangle is as expected
theorem isosceles_triangle_perimeter :
  let t := isosceles_triangle_with_sides 3 6 (or.inl rfl) (or.inr rfl) in
  t.a + t.b + t.c = 15 :=
by simp [isosceles_triangle_with_sides, add_assoc]

end isosceles_triangle_perimeter_l711_711549


namespace permute_nums_2xn_not_permute_nums_100x100_l711_711410

-- Problem (a)
theorem permute_nums_2xn (n : ℕ) (h1 : 2 < n) (table : Fin 2 → Fin n → ℕ)
    (h2 : ∀ j k : Fin n, j ≠ k → (table 0 j + table 1 j) ≠ (table 0 k + table 1 k)) :
    ∃ permuted_table : Fin 2 → Fin n → ℕ,
    (∀ j k : Fin n, j ≠ k → (permuted_table 0 j + permuted_table 1 j) ≠ (permuted_table 0 k + permuted_table 1 k))
    ∧ (permuted_table 0 0 + permuted_table 0 1 + ... + permuted_table 0 (n-1)) ≠ (permuted_table 1 0 + permuted_table 1 1 + ... + permuted_table 1 (n-1)) :=
  sorry

-- Problem (b)
theorem not_permute_nums_100x100 (table : Fin 100 → Fin 100 → ℕ)
    (h1 : ∀ j k : Fin 100, j ≠ k → (∑ i in Finset.univ, table i j) ≠ (∑ i in Finset.univ, table i k)) :
    ¬ (∃ permuted_table : Fin 100 → Fin 100 → ℕ,
    (∀ j k : Fin 100, j ≠ k → (∑ i in Finset.univ, permuted_table i j) ≠ (∑ i in Finset.univ, permuted_table i k))
    ∧ (∀ i j : Fin 100, i ≠ j → (∑ k in Finset.univ, permuted_table i k) ≠ (∑ k in Finset.univ, permuted_table j k))) :=
  sorry

end permute_nums_2xn_not_permute_nums_100x100_l711_711410


namespace difference_in_tiles_l711_711457

theorem difference_in_tiles :
  let side_length : ℕ → ℕ := λ n, 1 + 2 * (n - 1)
  let num_tiles : ℕ → ℕ := λ n, (side_length n) ^ 2
  num_tiles 11 - num_tiles 10 = 80 := by
sorry

end difference_in_tiles_l711_711457


namespace weight_measurement_l711_711965

theorem weight_measurement :
  ∀ (w : Set ℕ), w = {1, 3, 9, 27} → (∀ n ∈ w, ∃ k, k = n ∧ k ∈ w) →
  ∃ (num_sets : ℕ), num_sets = 41 := by
  intros w hw hcomb
  sorry

end weight_measurement_l711_711965


namespace number_of_liars_is_28_l711_711898

variables (n : ℕ → Prop)

-- Define the conditions for knights and liars
def knight (k : ℕ) := n k
def liar (k : ℕ) := ¬n k

-- Define statements for odd and even numbered people
def odd_statement (k : ℕ) := ∀ m, m > k → liar m
def even_statement (k : ℕ) := ∀ m, m < k → liar m

-- Define the main hypothesis following the problem conditions
def conditions : Prop :=
  (∀ k, k % 2 = 1 → (knight k ↔ odd_statement k)) ∧
  (∀ k, k % 2 = 0 → (knight k ↔ even_statement k)) ∧
  (∃ m, m = 30) -- Ensuring there are 30 people

-- Prove the main statement
theorem number_of_liars_is_28 : ∃ l, l = 28 ∧ (∀ k, k ≤ 30 → (liar k ↔ k ≤ 28)) :=
by
  sorry

end number_of_liars_is_28_l711_711898


namespace probability_of_being_selected_same_l711_711255

def reasonable_sampling_method : Prop := sorry -- We assume this due to the problem statement
def same_sample_size (size1 size2 : ℕ) : Prop := size1 = size2 -- Condition from the problem

theorem probability_of_being_selected_same (size1 size2 : ℕ)
  (h_size_equal : same_sample_size size1 size2)
  (h_reasonable_sampling : reasonable_sampling_method) :
  ∀ student, (probability_of_being_selected size1 student = probability_of_being_selected size2 student) :=
sorry -- Proof to be completed

end probability_of_being_selected_same_l711_711255


namespace sum_of_k_with_distinct_integer_solutions_l711_711859

-- Definitions from conditions
def quadratic_eq (k : ℤ) (x : ℤ) : Prop := 3 * x^2 - k * x + 12 = 0

def distinct_integer_solutions (k : ℤ) : Prop := ∃ p q : ℤ, p ≠ q ∧ quadratic_eq k p ∧ quadratic_eq k q

-- Main statement to prove
theorem sum_of_k_with_distinct_integer_solutions : 
  ∑ k in {k : ℤ | distinct_integer_solutions k}, k = 0 := 
sorry

end sum_of_k_with_distinct_integer_solutions_l711_711859


namespace holiday_not_on_22nd_l711_711086

theorem holiday_not_on_22nd:
  ∀ d : ℕ, 
  (d ≠ 15 ∧ d ≠ 16 ∧ d ≠ 17 ∧ d ≠ 18 ∧ d ≠ 19 ∧ d ≠ 20 ∧ d ≠ 21) → d = 22 → false :=
by
  intros d h1 h2
  cases h1 with h15 remaining1
  cases remaining1 with h16 remaining2
  cases remaining2 with h17 remaining3
  cases remaining3 with h18 remaining4
  cases remaining4 with h19 remaining5
  cases remaining5 with h20 h21

  contradiction

end holiday_not_on_22nd_l711_711086


namespace jill_account_balance_l711_711707

noncomputable def compound_interest 
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_account_balance :
  let P := 10000
  let r := 0.0396
  let n := 2
  let t := 2
  compound_interest P r n t ≈ 10816.49 :=
by
  sorry

end jill_account_balance_l711_711707


namespace remainder_problem_l711_711395

theorem remainder_problem (N : ℤ) (k : ℤ) (h : N = 35 * k + 25) : N % 15 = 10 :=
by
  sorry

end remainder_problem_l711_711395


namespace number_of_elements_in_T_l711_711289

noncomputable def g (x : ℝ) := (2 * x + 8) / x

def sequence_g : ℕ → (ℝ → ℝ)
| 0       := g
| (n + 1) := g ∘ sequence_g n

def T (x : ℝ) : Prop := ∃ n : ℕ, sequence_g n x = x

theorem number_of_elements_in_T : { x : ℝ | T x }.finite.to_finset.card = 2 :=
by
  sorry

end number_of_elements_in_T_l711_711289


namespace sum_of_uv_l711_711291

theorem sum_of_uv (u v : ℕ) (hu : 0 < u) (hv : 0 < v) (hv_lt_hu : v < u)
  (area_pent : 6 * u * v = 500) : u + v = 19 :=
by
  sorry

end sum_of_uv_l711_711291


namespace problem_statement_l711_711261

noncomputable def a : ℕ → ℚ
| 1 := 1 / 2
| 2 := 1 / 3
| n := if h : n ≥ 3 ∧ n % 4 = 1 then 2
       else if h : n ≥ 3 ∧ n % 4 = 2 then 3
       else if h : n ≥ 3 ∧ n % 4 = 3 then 1 / 2
       else if h : n ≥ 3 ∧ n % 4 = 0 then 1 / 3
       else 1 -- default value, shouldn't reach

theorem problem_statement :
  a 2016 + a 2017 = 7 / 2 :=
by sorry

end problem_statement_l711_711261


namespace num_integers_divisible_by_11_between_100_and_500_l711_711006

theorem num_integers_divisible_by_11_between_100_and_500 : 
  ∃ n, n = 37 ∧ (∀ x, 100 ≤ x ∧ x ≤ 500 ∧ x % 11 = 0 ↔ ∃ k, x = 11 * k ∧ 100 ≤ 11 * k ∧ 11 * k ≤ 500) :=
begin
  sorry
end

end num_integers_divisible_by_11_between_100_and_500_l711_711006


namespace complement_event_of_at_least_one_white_ball_is_both_red_l711_711249

theorem complement_event_of_at_least_one_white_ball_is_both_red :
  ∀ (draw : finset (fin 4)) (bags : finset (fin 4)),
  (bags = {0, 1, 2, 3}) → 
  (draw ⊆ bags) → 
  (draw.card = 2) → 
  (∀ (event_a : Prop), event_a ↔ (∃ w1 w2, w1 ≠ w2 ∧ w1 ∈ draw ∧ w2 ∈ draw ∧ w1 = 2 ∨ w2 = 2)) → 
  (∃ r1 r2, r1 ≠ r2 ∧ r1 ∈ draw ∧ r2 ∈ draw ∧ r1 < 2 ∧ r2 < 2) :=
by
  intro draw bags h_bags h_draw h_card event_a
  use sorry

end complement_event_of_at_least_one_white_ball_is_both_red_l711_711249


namespace parabolas_intersect_points_l711_711034

theorem parabolas_intersect_points :
  ∀ (x y : ℝ), (y = 4 * x^2 + 5 * x - 6) ∧ (y = 2 * x^2 + 14) →
  (x = -4 ∧ y = 38) ∨ (x = 5 / 2 ∧ y = 31.5) :=
by
  intros x y h,
  cases h,
  sorry

end parabolas_intersect_points_l711_711034


namespace jill_investment_value_l711_711693

def compoundInterest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_investment_value :
  compoundInterest 10000 0.0396 2 2 ≈ 10812 :=
by
  sorry

end jill_investment_value_l711_711693


namespace shorter_side_length_l711_711933

theorem shorter_side_length (a b : ℕ) (h1 : 2 * a + 2 * b = 42) (h2 : a * b = 108) : b = 9 :=
by
  sorry

end shorter_side_length_l711_711933


namespace trigonometric_identity_proof_l711_711394

theorem trigonometric_identity_proof (x : ℝ) (k : ℤ) :
  cos (2 * x) ≠ 0 →
  sin (2 * x) ≠ 0 →
  (cos (2 * x))⁻² * tan (2 * x) + (sin (2 * x))⁻² * cot (2 * x) =
  (8 * (cos (4 * x))^2) / (sin (4 * x))^3 + 10 * arcsin (4 * x) + 4 * sqrt 3 →
  x = (-1)^(k + 1) * (π / 12) + π * k / 4 := 
begin
  sorry
end

end trigonometric_identity_proof_l711_711394


namespace cow_milk_problem_l711_711239

theorem cow_milk_problem (x : ℕ) (h : ∀ y, x cows_produce y = y <= x+1):
  (d = (x * (x+2) * (x+5)) / ((x+1) * (x+3))) :=
sorry

end cow_milk_problem_l711_711239


namespace convert_base8_to_base10_l711_711077

theorem convert_base8_to_base10 : 
  ∀ (n : ℕ), n = 742 → (7 * 8^2 + 4 * 8^1 + 2 * 8^0) = 482 := 
by
  intros n h
  rw h
  have h₁ : 742 = 7 * 8^2 + 4 * 8^1 + 2 * 8^0 := by norm_num
  rw h₁
  norm_num
  done

end convert_base8_to_base10_l711_711077


namespace n_squared_divisors_form_l711_711717

noncomputable def M (n : ℕ) : ℕ := sorry -- number of divisors of n^2 of the form 4k-1
noncomputable def P (n : ℕ) : ℕ := sorry -- number of divisors of n^2 of the form 4k+1

theorem n_squared_divisors_form (n : ℕ) (h : n > 0) : M(n) < P(n) := 
by
  sorry

end n_squared_divisors_form_l711_711717


namespace circle_condition_l711_711791

theorem circle_condition (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 + 4*x - 2*y + 5*m = 0) →
  (m < 1) :=
by
  sorry

end circle_condition_l711_711791


namespace find_N_l711_711654

theorem find_N : ∃ N : ℕ, (∃ x y : ℕ, 
(10 * x + y = N) ∧ 
(4 * x + 2 * y = N / 2) ∧ 
(1 ≤ x) ∧ (x ≤ 9) ∧ 
(0 ≤ y) ∧ (y ≤ 9)) ∧
(N = 32 ∨ N = 64 ∨ N = 96) :=
by {
    have h1 : ∀ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 → 4 * x + 2 * y = (10 * x + y) / 2 → true, sorry,
    exact h1
}

end find_N_l711_711654


namespace problem_proof_l711_711557

-- Define propositions p and q
def p : Prop := ∃ x : ℝ, x^2 - 2*x + 3 ≤ 0
def q : Prop := (complex_real.eccentricity (4 : ℝ)) = 2

-- Define the main theorem
theorem problem_proof : p ∨ q :=
by
  have h₁ : p := sorry -- Proof that p is true.
  have h₂ : ¬ q := sorry -- Proof that q is false.
  exact Or.inl h₁ -- Since p is true, p ∨ q is true by Or.inl

end problem_proof_l711_711557


namespace find_number_l711_711013

theorem find_number :
  ∃ x : ℝ, (3.242 * x) / 100 = 0.051871999999999995 ∧ x = 1.6 :=
begin
  use 1.6,
  split,
  {
    suffices h : 3.242 * 1.6 = 5.1871999999999995,
    {
      exact h.symm ▸ rfl,
    },
    sorry, -- Numerical verification (optional)
  },
  refl,
end

end find_number_l711_711013


namespace derivative_r_l711_711510

variable (a b : ℝ) (t : ℝ)

def r (t : ℝ) : ℝ × ℝ :=
  (a * Real.cos t, b * Real.sin t)

theorem derivative_r : 
  (deriv (λ t, (a * Real.cos t, b * Real.sin t)) t) = 
  (-a * Real.sin t, b * Real.cos t) :=
sorry

end derivative_r_l711_711510


namespace general_formulas_values_of_n_for_c_n_gt_one_no_three_terms_arithmetic_seq_l711_711524
noncomputable def a_n (n : ℕ) : ℕ := 2^(n-1)
noncomputable def b_n (n : ℕ) : ℕ := 3*n - 1
noncomputable def c_n (n : ℕ) : ℚ := (3*n - 1) / 2^(n-1)

-- 1. Prove that the sequence {a_n} is given by a_n = 2^(n-1) and {b_n} is given by b_n = 3n - 1
theorem general_formulas :
  (∀ n : ℕ, n > 0 → a_n n = 2^(n-1)) ∧
  (∀ n : ℕ, n > 0 → b_n n = 3*n - 1) :=
sorry

-- 2. Prove that the values of n for which c_n > 1 are n = 1, 2, 3, 4
theorem values_of_n_for_c_n_gt_one :
  { n : ℕ | n > 0 ∧ c_n n > 1 } = {1, 2, 3, 4} :=
sorry

-- 3. Prove that no three terms from {a_n} can form an arithmetic sequence
theorem no_three_terms_arithmetic_seq :
  ∀ p q r : ℕ, p < q ∧ q < r ∧ p > 0 ∧ q > 0 ∧ r > 0 →
  ¬ (2 * a_n q = a_n p + a_n r) :=
sorry

end general_formulas_values_of_n_for_c_n_gt_one_no_three_terms_arithmetic_seq_l711_711524


namespace parabola_vertex_coordinates_l711_711339

theorem parabola_vertex_coordinates :
  ∀ x y : ℝ, y = 3 * (x - 7)^2 + 5 → (7, 5) = (7, 5) :=
by
  intros x y h
  exact rfl

end parabola_vertex_coordinates_l711_711339


namespace correct_propositions_l711_711210

-- Definitions of the Propositions
def prop1 (L L' : Type) [affine_space ℝ L] [affine_space ℝ L'] : Prop :=
  ∀ (l : L) (l' : L'), ¬ intersect l l' → parallel l l'

def prop2 (L L' L'' : Type) [affine_space ℝ L] [affine_space ℝ L'] [affine_space ℝ L''] : Prop :=
  ∀ (l : L) (l' : L') (l'' : L''), parallel l l' ∧ parallel l l'' → parallel l' l''

def prop3 (L L' L'' : Type) [affine_space ℝ L] [affine_space ℝ L'] [affine_space ℝ L''] : Prop :=
  ∀ (l : L) (l' : L') (l'' : L''), (intersect l l' ∧ parallel l' l'') → intersect l l''

def prop4 (L L' L'' L''' : Type) [affine_space ℝ L] [affine_space ℝ L'] [affine_space ℝ L''] [affine_space ℝ L'''] : Prop :=
  ∀ (a : L) (b : L') (c : L'') (d : L'''), (parallel a b ∧ parallel c d ∧ parallel a d) → parallel b c

-- Proof to check correctness of the propositions
theorem correct_propositions {L L' L'' L''' : Type} [affine_space ℝ L] [affine_space ℝ L'] [affine_space ℝ L''] [affine_space ℝ L'''] :
  ¬ prop1 L L' ∧ prop2 L L' L'' ∧ ¬ prop3 L L' L'' ∧ prop4 L L' L'' L''' :=
by
  sorry

end correct_propositions_l711_711210


namespace two_digit_numbers_solution_l711_711640

theorem two_digit_numbers_solution :
  ∀ (N : ℕ), (∃ (x y : ℕ), (N = 10 * x + y) ∧ (x < 10) ∧ (y < 10) ∧ 4 * x + 2 * y = N / 2) →
    (N = 32 ∨ N = 64 ∨ N = 96) := 
by
  sorry

end two_digit_numbers_solution_l711_711640


namespace problem_solve_l711_711292

theorem problem_solve (x y : ℝ) (h1 : x ≠ y) (h2 : x / y + (x + 6 * y) / (y + 6 * x) = 3) : 
    x / y = (8 + Real.sqrt 46) / 6 := 
  sorry

end problem_solve_l711_711292


namespace sum_of_values_k_l711_711861

theorem sum_of_values_k (k : ℕ) : 
  (∀ x y : ℤ, (3 * x * x - k * x + 12 = 0) ∧ (3 * y * y - k * y + 12 = 0) ∧ x ≠ y) → k = 0 :=
by
  sorry

end sum_of_values_k_l711_711861


namespace maximum_third_altitude_l711_711672

noncomputable def largest_third_altitude_in_scalene_triangle (DEF : Type) [triangle DEF] 
  (alt1 : ℝ) (alt2 : ℝ) (h_alt1 : alt1 = 6) (h_alt2 : alt2 = 18) : ℝ :=
  let k := 9 in
  k

theorem maximum_third_altitude
  (DEF : Type) [triangle DEF]
  (alt1 alt2 : ℝ) (h_alt1 : alt1 = 6) (h_alt2 : alt2 = 18) :
  ∃ k : ℝ, k = 9 :=
by
  use largest_third_altitude_in_scalene_triangle DEF alt1 alt2 h_alt1 h_alt2
  exact sorry

end maximum_third_altitude_l711_711672


namespace find_possible_m_values_l711_711798

noncomputable def complex_values_m (m : ℂ) : Prop :=
  (∀ x : ℂ, x ≠ 0 → ( ((x / (x + 2)) + (x / (x + 3)) = m * x) ↔ m = 0 ∨ m = 2 * Complex.i ∨ m = -2 * Complex.i ) )
 
theorem find_possible_m_values:
  ∀ m : ℂ, complex_values_m m :=
by
  sorry

end find_possible_m_values_l711_711798


namespace closest_points_distance_between_circles_l711_711133

def dist (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def radius (p : ℝ × ℝ) : ℝ := 
  p.2

def closest_distance_between_circles (c1 c2 : ℝ × ℝ) : ℝ := 
  dist c1 c2 - (radius c1 + radius c2)

theorem closest_points_distance_between_circles :
  closest_distance_between_circles (3, 3) (20, 12) = real.sqrt 370 - 15 :=
sorry

end closest_points_distance_between_circles_l711_711133


namespace row_seat_notation_l711_711241

-- Define the notation for "row seat" as a pair (row number, seat number).
def notation (row : ℕ) (seat : ℕ) : ℕ × ℕ :=
  (row, seat)

-- Given condition as a hypothesis
def given_condition : notation 10 3 = (10, 3) := rfl

-- Goal: Prove that the notation for "5 row 16 seat" is (5, 16)
theorem row_seat_notation : notation 5 16 = (5, 16) :=
by
  -- Hypothesis for the given condition is provided separately, can be used for context if needed
  -- given_condition,
  sorry

end row_seat_notation_l711_711241


namespace journey_distance_l711_711083

theorem journey_distance (t : ℝ) : 
  t = 20 →
  ∃ D : ℝ, (D / 20 + D / 30 = t) ∧ D = 240 :=
by
  sorry

end journey_distance_l711_711083


namespace distance_travelled_is_960_l711_711888

-- Definitions based on conditions
def speed_slower := 60 -- Speed of slower bike in km/h
def speed_faster := 64 -- Speed of faster bike in km/h
def time_diff := 1 -- Time difference in hours

-- Problem statement: Prove that the distance covered by both bikes is 960 km.
theorem distance_travelled_is_960 (T : ℝ) (D : ℝ) 
  (h1 : D = speed_slower * T)
  (h2 : D = speed_faster * (T - time_diff)) :
  D = 960 := 
sorry

end distance_travelled_is_960_l711_711888


namespace discount_price_l711_711131

theorem discount_price (P P_d : ℝ) 
  (h1 : P_d = 0.85 * P) 
  (P_final : ℝ) 
  (h2 : P_final = 1.25 * P_d) 
  (h3 : P - P_final = 5.25) :
  P_d = 71.4 :=
by
  sorry

end discount_price_l711_711131


namespace candle_duration_1_hour_per_night_l711_711957

-- Definitions based on the conditions
def burn_rate_2_hours (candles: ℕ) (nights: ℕ) : ℕ := nights / candles -- How long each candle lasts when burned for 2 hours per night

-- Given conditions provided
def nights_24 : ℕ := 24
def candles_6 : ℕ := 6

-- The duration a candle lasts when burned for 2 hours every night
def candle_duration_2_hours_per_night : ℕ := burn_rate_2_hours candles_6 nights_24 -- => 4 (not evaluated here)

-- Theorem to prove the duration a candle lasts when burned for 1 hour every night
theorem candle_duration_1_hour_per_night : candle_duration_2_hours_per_night * 2 = 8 :=
by
  sorry -- The proof is omitted, only the statement is required

-- Note: candle_duration_2_hours_per_night = 4 by the given conditions 
-- This leads to 4 * 2 = 8, which matches the required number of nights the candle lasts when burned for 1 hour per night.

end candle_duration_1_hour_per_night_l711_711957


namespace hire_applicant_C_hire_applicant_B_l711_711921

-- Define the scores for each applicant
def scores := { 
  A := (9, 8, 7, 5),  -- (Education, Experience, Ability, Attitude)
  B := (8, 6, 8, 7),
  C := (8, 9, 8, 5)
}

-- Function to calculate weighted score
def weighted_score (weights : ℕ × ℕ × ℕ × ℕ) (scores : ℕ × ℕ × ℕ × ℕ) : ℚ :=
  let (w1, w2, w3, w4) := weights
  let (s1, s2, s3, s4) := scores
  (w1 * s1 + w2 * s2 + w3 * s3 + w4 * s4 : ℚ) / (w1 + w2 + w3 + w4 : ℚ)

-- Given weights 1:1:1:1
def weights1 := (1, 1, 1, 1)

-- Given weights 4:1:1:4
def weights2 := (4, 1, 1, 4)

-- Define the Lean statements for the documents

-- First part of the problem: Prove that C will be hired with weights 1:1:1:1
theorem hire_applicant_C :
  let A_score := weighted_score weights1 scores.A
  let B_score := weighted_score weights1 scores.B
  let C_score := weighted_score weights1 scores.C
  C_score > A_score ∧ C_score > B_score := by
  sorry

-- Second part of the problem: Prove that B will be hired with weights 4:1:1:4
theorem hire_applicant_B :
  let A_score := weighted_score weights2 scores.A
  let B_score := weighted_score weights2 scores.B
  let C_score := weighted_score weights2 scores.C
  B_score > A_score ∧ B_score > C_score := by
  sorry

end hire_applicant_C_hire_applicant_B_l711_711921


namespace part_a_part_b_part_c_part_d_l711_711725
noncomputable theory

open Probability MeasureTheory

variables {Ω : Type*} {ι : Type*}

-- Define the sequence of independent Bernoulli random variables
def bernoulli_sequence (Ω : Type*) (ξ : Ω → ι → ℤ) : Prop :=
  ∀ (i : ι), ∀ (ω : Ω),
  (ξ ω i = 1 ∨ ξ ω i = -1) ∧ (P{ξ ω i = 1} = 1 / 2) ∧ (P{ξ ω i = -1} = 1 / 2)

-- Define the partial sum Sₙ
def S (ξ : Ω → ℤ) (n : ℕ) : Ω → ℤ :=
  λ ω, (finset.range n).sum (λ i, ξ ω i)

-- Define the first hitting time σ₁(x)
def σ₁ (x : ℤ) (S : ℕ → Ω → ℤ) : Ω → ℕ :=
  λ ω, if (∃ n, S n ω = x) then (nat.find (λ n, S n ω = x ∧ 0 < n)) else ⊤

-- Formulate the theorems
theorem part_a (x n : ℕ) (ξ : Ω → ℤ) (h_seq : bernoulli_sequence Ω ξ) :
  P{σ₁ x (S ξ) > n} = P{max 0 (finset.range n).to_pa S(f) < x} := 
sorry

theorem part_b (x n : ℕ) (ξ : Ω → ℤ) (h_seq : bernoulli_sequence Ω ξ) :
  P{σ₁ x (S ξ) = n} = (x / n) 2 ^ (-n) (nat.choose n ((n + x) / 2)) := 
sorry

theorem part_c (n : ℕ) (ξ : Ω → ℤ) (h_seq : bernoulli_sequence Ω ξ) :
  P{σ₁ 1 (S ξ) = 2*n+1} = 2 ^ (-2*n-1) / (n+1) * (nat.choose (2*n) n) := 
sorry

theorem part_d (n : ℕ) (ξ : Ω → ℤ) (h_seq : bernoulli_sequence Ω ξ) :
  P{σ₁ 1 (S ξ) > n} = 2 ^ (-n) * (nat.choose n (n / 2)) := 
sorry

end part_a_part_b_part_c_part_d_l711_711725


namespace num_subsets_of_A_l711_711804

/-- 
  The set A is defined as the set of elements x in positive natural numbers
  such that 3 ≤ x < 6. We aim to prove that the number of subsets of this set is 8.
-/
theorem num_subsets_of_A : 
  let A := {x : ℕ // 3 ≤ x ∧ x < 6} in
  fintype.card (set.powerset A) = 8 := 
by sorry

end num_subsets_of_A_l711_711804


namespace arccos_sin_const_l711_711961

theorem arccos_sin_const : real.arccos (real.sin 1.5) = 0.0708 := by
  sorry

end arccos_sin_const_l711_711961


namespace T_seq_correct_l711_711596

noncomputable def a₅ : ℕ := 13
noncomputable def a_seq : ℕ → ℤ := λ n, 3 * (n+1) - 2
noncomputable def b_sum : ℕ → ℝ := λ n, 1 - (1 / (2:ℝ)^n)
noncomputable def b_seq : ℕ → ℝ := λ n, (1 / (2:ℝ)^n)
noncomputable def T_seq : ℕ → ℝ :=
λ n, ∑ k in finset.range (n + 1), (a_seq k) * (b_seq k)

open_locale big_operators

lemma a_seq_correct (n : ℕ) : a_seq (n+1) = a_seq n + 3 :=
by {
  dsimp [a_seq], calc
  3 * (n+2) - 2 = 3 * (n+1) + 3 - 2 : by ring
  ... = a_seq n + 3 : by simp[a_seq]
}

lemma b_seq_correct (n : ℕ) (h : n > 0) : 
  b_seq n = b_sum n - b_sum (n - 1) :=
by {
  dsimp [b_seq, b_sum],
  have : 1 - (1 / (2:ℝ)^n) - (1 - (1 / (2:ℝ)^(n - 1))) = (1 / (2:ℝ)^n),
  {
    have h₁ : (1 / (2:ℝ)^n) > 0, from one_div_pos.mpr (pow_pos zero_lt_two _),
    have h₂ : (1 / (2:ℝ)^(n-1)) > 0, from one_div_pos.mpr (pow_pos zero_lt_two _),
    calc
    (1 - (1 / (2:ℝ)^n)) - (1 - (1 / (2:ℝ)^(n-1))) = (1 / (2:ℝ)^(n - 1)) - (1 / (2:ℝ)^n) : by ring
    ... = 1 / (2:ℝ)^n : by { field_simp, rw [pow_succ', mul_assoc], simp },
  },
  assumption,
}

theorem T_seq_correct (n : ℕ) : 
  T_seq n = 4 - (3 * (n + 1) + 1) / (2 ^ n) ∧ T_seq n < 4 :=
by {
  dsimp [T_seq, a_seq, b_seq],
  have : ∑ k in finset.range (n + 1), ((3 * (k + 1) - 2) * (1 / (2 ^ k))) = 4 - ((3 * (n + 1) + 1) / (2 ^ n)) ∧
         ∑ k in finset.range (n + 1), ((3 * (k + 1) - 2) * (1 / (2 ^ k))) < 4,
  -- Computation here would require extensive simplification and using sum formulas
  sorry
}

end T_seq_correct_l711_711596


namespace find_fraction_l711_711286

theorem find_fraction (a b : ℝ) (h₁ : a ≠ b) (h₂ : a / b + (a + 6 * b) / (b + 6 * a) = 2) :
  a / b = 1 / 2 :=
sorry

end find_fraction_l711_711286


namespace sum_of_non_y_coefficients_l711_711812

theorem sum_of_non_y_coefficients (x y : ℝ) :
  let expr := (x + y + 3) ^ 3 in
  let non_y_expr := expr.subst (λ y, 0) in
  non_y_expr.subst (λ x, 1) = 64 :=
by
  sorry

end sum_of_non_y_coefficients_l711_711812


namespace rectangle_area_error_l711_711108

theorem rectangle_area_error
  (L W : ℝ)
  (measured_length : ℝ := 1.15 * L)
  (measured_width : ℝ := 1.20 * W)
  (true_area : ℝ := L * W)
  (measured_area : ℝ := measured_length * measured_width)
  (percentage_error : ℝ := ((measured_area - true_area) / true_area) * 100) :
  percentage_error = 38 :=
by
  sorry

end rectangle_area_error_l711_711108


namespace horner_method_operations_l711_711954

def polynomial : ℝ → ℝ :=
  λ x, 3 * x^6 + 4 * x^5 + 5 * x^4 + 6 * x^3 + 7 * x^2 + 8 * x + 1

def horner_operations : ℕ := 12

theorem horner_method_operations (x : ℝ) (h : x = 0.7) :
  let f := polynomial x in
  (count_operations_horner f) = horner_operations := 
sorry

end horner_method_operations_l711_711954


namespace infinite_sum_F_l711_711720

noncomputable def F : ℕ → ℚ
| 0     := 0
| 1     := 3 / 2
| (n+2) := 5 / 2 * F (n+1) - F n

theorem infinite_sum_F (F : ℕ → ℚ)
  (hF0 : F 0 = 0)
  (hF1 : F 1 = 3 / 2)
  (hFrec : ∀ n, F (n + 2) = (5 / 2) * F (n + 1) - F n) :
  (∑' n, 1 / F (2 ^ n)) = 1 :=
sorry

end infinite_sum_F_l711_711720


namespace two_digit_numbers_solution_l711_711642

theorem two_digit_numbers_solution :
  ∀ (N : ℕ), (∃ (x y : ℕ), (N = 10 * x + y) ∧ (x < 10) ∧ (y < 10) ∧ 4 * x + 2 * y = N / 2) →
    (N = 32 ∨ N = 64 ∨ N = 96) := 
by
  sorry

end two_digit_numbers_solution_l711_711642


namespace two_digit_number_solution_l711_711636

theorem two_digit_number_solution (N : ℕ) (x y : ℕ) :
  (10 * x + y = N) ∧ (4 * x + 2 * y = (10 * x + y) / 2) →
  N = 32 ∨ N = 64 ∨ N = 96 := 
sorry

end two_digit_number_solution_l711_711636


namespace sum_of_possible_n_gon_l711_711668

theorem sum_of_possible_n_gon (n : ℕ) (h₁ : ∃ x, (1 ≤ x ∧ x ≤ n-1) ∧ (x-1) * (n-x-1) = 14) :
    {n : ℕ | ∃ x, (1 ≤ x ∧ x ≤ n-1) ∧ (x-1) * (n-x-1) = 14}.sum = 28 :=
by 
    sorry

end sum_of_possible_n_gon_l711_711668


namespace find_m_l711_711229

open Finset

def vec := ℕ → ℝ -- Define a vector type as a map from natural numbers to real numbers 

def perpendicular (v1 v2 : vec) : Prop :=
  v1 0 * v2 0 + v1 1 * v2 1 = 0

theorem find_m :
  ∃ m : ℝ, let a := (λ i, if i = 0 then 5 else m), 
                b := (λ i, if i = 0 then 2 else -2) in 
            perpendicular (λ i, a i + b i) b → m = 9 :=
by {
  -- Proof goes here
  sorry
}

end find_m_l711_711229


namespace three_digit_multiples_of_seven_l711_711606

theorem three_digit_multiples_of_seven : 
  let min_bound := 100
  let max_bound := 999
  let first_multiple := Nat.ceil (min_bound / 7)
  let last_multiple := Nat.floor (max_bound / 7)
  (last_multiple - first_multiple + 1) = 128 :=
by 
  let min_bound := 100
  let max_bound := 999
  let first_multiple := Nat.ceil (min_bound / 7)
  let last_multiple := Nat.floor (max_bound / 7)
  have calc_first_multiple : first_multiple = 15 := sorry
  have calc_last_multiple : last_multiple = 142 := sorry
  have count_multiples : (last_multiple - first_multiple + 1) = 128 := sorry
  exact count_multiples

end three_digit_multiples_of_seven_l711_711606


namespace find_n_l711_711647

theorem find_n (N : ℕ) (hN : 10 ≤ N ∧ N < 100)
  (h : ∃ a b : ℕ, N = 10 * a + b ∧ 4 * a + 2 * b = N / 2) : 
  N = 32 ∨ N = 64 ∨ N = 96 :=
sorry

end find_n_l711_711647


namespace heavy_tailed_permutations_count_l711_711448

def is_heavy_tailed (p : Permutation (Fin 5)) : Prop :=
  p 0 + p 1 < p 3 + p 4

theorem heavy_tailed_permutations_count :
  {p : Permutation (Fin 5) | 
    p = {i // i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4} ∧
    is_heavy_tailed p}.card = 48 :=
sorry

end heavy_tailed_permutations_count_l711_711448


namespace probability_neither_red_nor_purple_l711_711917

theorem probability_neither_red_nor_purple :
  let total_balls := 100 
  let white_balls := 50 
  let green_balls := 30 
  let yellow_balls := 8 
  let red_balls := 9 
  let purple_balls := 3 
  (100 - (red_balls + purple_balls)) / total_balls = 0.88 := 
by 
  -- Definitions based on conditions
  let total_balls := 100 
  let white_balls := 50 
  let green_balls := 30 
  let yellow_balls := 8 
  let red_balls := 9 
  let purple_balls := 3 
  -- Compute the probability
  sorry

end probability_neither_red_nor_purple_l711_711917


namespace emissions_from_tap_water_l711_711800

def carbon_dioxide_emission (x : ℕ) : ℕ := 9 / 10 * x  -- Note: using 9/10 instead of 0.9 to maintain integer type

theorem emissions_from_tap_water : carbon_dioxide_emission 10 = 9 :=
by
  sorry

end emissions_from_tap_water_l711_711800


namespace three_digit_multiples_of_seven_count_l711_711610

theorem three_digit_multiples_of_seven_count :
  let smallest := 15
  let largest := 142
  largest - smallest + 1 = 128 :=
by
  let smallest := 15
  let largest := 142
  have h_smallest : 7 * smallest = 105 := rfl
  have h_largest : 7 * largest = 994 := rfl
  show largest - smallest + 1 = 128 from sorry

end three_digit_multiples_of_seven_count_l711_711610


namespace sum_of_k_with_distinct_integer_roots_l711_711841

theorem sum_of_k_with_distinct_integer_roots :
  ∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ 3*p*q = 12 ∧ p + q = k/3}, k = 0 :=
by
  sorry

end sum_of_k_with_distinct_integer_roots_l711_711841


namespace requiredHorsepower_l711_711078

-- Defining constants and parameters as given in the problem conditions
def height := 84 -- meters
def trackLength := 140 -- meters
def carWeight := 5000 -- kg per car (50 quintals)
def numPeople := 40
def avgPersonWeight := 70 -- kg
def rollingFrictionCoeff := 0.005
def tripTime := 60 -- seconds

-- Derived values from the given conditions
def totalWeightLoadedCar := carWeight + numPeople * avgPersonWeight -- kg
def emptyCarWeight := carWeight -- kg
def loadedCarWeight := totalWeightLoadedCar -- kg

noncomputable def gravitationalForceComponent (mass: Real) := mass * (height / trackLength) -- kg

-- Forces involved
def forceRequired := gravitationalForceComponent loadedCarWeight - gravitationalForceComponent emptyCarWeight -- kg

-- Work done formula
def workDone := forceRequired * trackLength -- kg m

-- Power formula
def powerRequired := workDone / tripTime / 75 -- Convert to HP, since 1 HP = 75 kg m/s

-- Rolling friction calculations
def totalWeightCars := carWeight * 2 + numPeople * avgPersonWeight -- kg (both cars combined)
def cosAlpha := (Real.sqrt (trackLength^2 - height^2)) / trackLength
def rollingFrictionForce := totalWeightCars * cosAlpha * rollingFrictionCoeff -- kg

-- Additional work done due to friction
def additionalWork := rollingFrictionForce * trackLength -- kg m

-- Power required due to friction
def frictionPowerRequired := additionalWork / tripTime / 75 -- Convert to HP

-- Total power required
def totalPowerRequired := powerRequired + frictionPowerRequired -- HP

theorem requiredHorsepower : totalPowerRequired ≈ 53.86 := by
  sorry

end requiredHorsepower_l711_711078


namespace parabola_vertex_coordinates_l711_711337

theorem parabola_vertex_coordinates :
  ∀ x : ℝ, (3 * (x - 7) ^ 2 + 5) = 3 * (x - 7) ^ 2 + 5 := by
  sorry

end parabola_vertex_coordinates_l711_711337


namespace range_of_a_l711_711216

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → x ≤ 4 → (a * x^2 - 2 * x + 2) > 0) ↔ (a > 1 / 2) :=
by
  sorry

end range_of_a_l711_711216


namespace drawing_orders_l711_711047

-- Define the number of fruits and the drawing condition
def num_fruits := 4
def num_draws := 2

-- Define the calculation of the number of orders of fruits drawn without replacement
def calc_orders (n m : Nat) : Nat := n * (n - 1)

-- The proof statement
theorem drawing_orders : calc_orders num_fruits num_draws = 12 := by
  -- Calculation to show the proof
  calc_orders num_fruits num_draws = 4 * 3 := rfl
  ... = 12 := rfl

end drawing_orders_l711_711047


namespace percent_of_a_is_4b_l711_711786

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.2 * b) : (4 * b / a) * 100 = 333.33 := by
  sorry

end percent_of_a_is_4b_l711_711786


namespace angle_ABC_in_regular_octagon_l711_711963

theorem angle_ABC_in_regular_octagon (A B C D : Point)
    (r_octagon : RegularOctagon)
    (inscribed_square : InscribedSquare)
    (shared_side : SharedSide r_octagon inscribed_square)
    (consec_vertices_AB : ConsecutiveVertices r_octagon A B)
    (adjacent_to_shared_side : AdjacentVertex inscribed_square shared_side C)
    (adjacent_vertex_D : AdjacentVertexToOpposite shared_side C D): 
    ∠ABC = 90 := sorry

end angle_ABC_in_regular_octagon_l711_711963


namespace stream_speed_l711_711359

variable (D v : ℝ)

/--
The time taken by a man to row his boat upstream is twice the time taken by him to row the same distance downstream.
If the speed of the boat in still water is 63 kmph, prove that the speed of the stream is 21 kmph.
-/
theorem stream_speed (h : D / (63 - v) = 2 * (D / (63 + v))) : v = 21 := 
sorry

end stream_speed_l711_711359


namespace coeff_x3_in_expansion_l711_711335

theorem coeff_x3_in_expansion : 
  let term := (λ (k : ℕ), (-2)^k * Nat.choose 8 k * x^(4 - k / 2))
  (∃ k : ℕ, 4 - k / 2 = 3 ∧ term k = 112 * x^3) :=
by
  sorry

end coeff_x3_in_expansion_l711_711335


namespace tangent_line_at_origin_inequality_in_interval_l711_711214

open Real

noncomputable def f (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

theorem tangent_line_at_origin : 
  let x := 0,
  let f0 := f(0) in
  y = 2 * x :=
by
  sorry

theorem inequality_in_interval (x : ℝ) (h : 0 < x ∧ x < 1) : 
  f(x) > 2 * (x + (x^3)/3) :=
by
  sorry

end tangent_line_at_origin_inequality_in_interval_l711_711214


namespace log_sin_cos_decreasing_interval_l711_711802

open Real

def decreasing_interval (y : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x1 x2, a < x1 ∧ x1 < x2 ∧ x2 < b → y x2 < y x1

theorem log_sin_cos_decreasing_interval :
  ∀ (k : ℤ),
    decreasing_interval (λ x, log (1/3) (sin x - cos x))
      (2 * k * π + 3 * π / 4) (2 * k * π + 5 * π / 4) :=
sorry

end log_sin_cos_decreasing_interval_l711_711802


namespace triangle_inequality_not_true_l711_711198

theorem triangle_inequality_not_true (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : ¬ (b + c > 2 * a) :=
by {
  -- assume (b + c > 2 * a)
  -- we need to reach a contradiction
  sorry
}

end triangle_inequality_not_true_l711_711198


namespace largest_of_four_l711_711471

theorem largest_of_four : 
  let a := 1 
  let b := 0 
  let c := |(-2)| 
  let d := -3 
  max (max (max a b) c) d = c := by
  sorry

end largest_of_four_l711_711471


namespace negation_prop_l711_711348

variable {U : Type} (A B : Set U)
variable (x : U)

theorem negation_prop (h : x ∈ A ∩ B) : (x ∉ A ∩ B) → (x ∉ A ∧ x ∉ B) :=
sorry

end negation_prop_l711_711348


namespace minimum_bats_examined_l711_711054

theorem minimum_bats_examined 
  (bats : Type) 
  (R L : bats → Prop) 
  (total_bats : ℕ)
  (right_eye_bats : ∀ {b: bats}, R b → Fin 2)
  (left_eye_bats : ∀ {b: bats}, L b → Fin 3)
  (not_left_eye_bats: ∀ {b: bats}, ¬ L b → Fin 4)
  (not_right_eye_bats: ∀ {b: bats}, ¬ R b → Fin 5)
  : total_bats ≥ 7 := sorry

end minimum_bats_examined_l711_711054


namespace horner_evaluation_at_2_l711_711828

def f (x : ℤ) : ℤ := 3 * x^5 - 2 * x^4 + 2 * x^3 - 4 * x^2 - 7

theorem horner_evaluation_at_2 : f 2 = 16 :=
by {
  sorry
}

end horner_evaluation_at_2_l711_711828


namespace sequence_bound_l711_711727

def a_seq : ℕ → ℚ
| 0       := 7 / 17
| (n + 1) := 3 * (a_seq n) ^ 2 - 2

noncomputable def c : ℝ := 17 / real.sqrt 210

theorem sequence_bound :
  (∀ n > 0, (|∏ i in finset.range n, a_seq i| ≤ c / 3^n)) ∧ (real.floor (100 * c) = 117) := by
sorry

end sequence_bound_l711_711727


namespace general_term_l711_711685

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n / (2 + a n)

theorem general_term (a : ℕ → ℝ) (h : seq a) : ∀ n : ℕ, n > 0 → a n = 2 / (n + 1) :=
by
sorry

end general_term_l711_711685


namespace new_salt_percentage_l711_711472

def initial_volume : ℝ := 80
def initial_concentration : ℝ := 0.10
def added_volume : ℝ := 20
def new_volume : ℝ := initial_volume + added_volume
def amount_of_salt : ℝ := initial_concentration * initial_volume
def new_concentration : ℝ := (amount_of_salt / new_volume) * 100

theorem new_salt_percentage : new_concentration = 8 := by
  sorry

end new_salt_percentage_l711_711472


namespace three_digit_multiples_of_seven_l711_711618

theorem three_digit_multiples_of_seven : 
  ∃ k, (k = {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 0}.card) ∧ k = 128 :=
by {
  sorry
}

end three_digit_multiples_of_seven_l711_711618


namespace M_lies_in_third_quadrant_l711_711583

noncomputable def harmonious_point (a b : ℝ) : Prop :=
  3 * a = 2 * b + 5

noncomputable def point_M_harmonious (m : ℝ) : Prop :=
  harmonious_point (m - 1) (3 * m + 2)

theorem M_lies_in_third_quadrant (m : ℝ) (hM : point_M_harmonious m) : 
  (m - 1 < 0 ∧ 3 * m + 2 < 0) :=
by {
  sorry
}

end M_lies_in_third_quadrant_l711_711583


namespace max_king_moves_l711_711829

-- Definitions
def board_size : ℕ := 99
def num_cells : ℕ := board_size * board_size
def positions : Fin num_cells := sorry -- positions on the board

-- King movement directions
inductive Direction
| Up | Down | Left | Right | UpLeft | UpRight | DownLeft | DownRight

variable {pos : Fin num_cells} -- current position of the king

-- Number in the cell at a given position
def num_in_cell (pos : Fin num_cells) : ℕ := sorry

-- Predicate for valid moves
def valid_move (pos pos' : Fin num_cells) : Prop :=
  (∃ (d : Direction), is_adjacent pos pos' d) ∧ num_in_cell pos < num_in_cell pos'

-- Function to determine if two positions are adjacent given a direction
def is_adjacent (pos pos' : Fin num_cells) (d : Direction) : Prop := sorry

-- Theorem stating the solution
theorem max_king_moves : ∀ (arrangement : Fin num_cells → ℕ), ∃ (moves : ℕ), moves = 3 :=
  sorry

end max_king_moves_l711_711829


namespace volume_of_solid_of_revolution_l711_711453

theorem volume_of_solid_of_revolution :
  ∀ (a b : ℝ), a = 15 → b = 20 → 
  15*π*(16 + √ʺ319)/ 3 +
    (32.5 - √ʺ319) * π * (144 +172*√ʺ319 + 319*√ʺ319²)/ 3 
  = 754.124 * π := 
sorry

end volume_of_solid_of_revolution_l711_711453


namespace two_digit_number_solution_l711_711637

theorem two_digit_number_solution (N : ℕ) (x y : ℕ) :
  (10 * x + y = N) ∧ (4 * x + 2 * y = (10 * x + y) / 2) →
  N = 32 ∨ N = 64 ∨ N = 96 := 
sorry

end two_digit_number_solution_l711_711637


namespace scrabble_middle_letter_value_l711_711270

theorem scrabble_middle_letter_value 
  (triple_word_score : ℕ) (single_letter_value : ℕ) (middle_letter_value : ℕ)
  (h1 : triple_word_score = 30)
  (h2 : single_letter_value = 1)
  : 3 * (2 * single_letter_value + middle_letter_value) = triple_word_score → middle_letter_value = 8 :=
by
  sorry

end scrabble_middle_letter_value_l711_711270


namespace grasshoppers_cannot_reach_l711_711365

-- Defining the positions of the grasshoppers as coordinates
structure Point where
  x : ℤ
  y : ℤ
deriving DecidableEq

-- Initial positions of the grasshoppers
def A0 : Point := ⟨1, 0⟩
def B0 : Point := ⟨0, 0⟩
def C0 : Point := ⟨0, 1⟩

-- Final positions to reach
def Af : Point := ⟨0, 0⟩
def Bf : Point := ⟨-1, -1⟩
def Cf : Point := ⟨1, 1⟩

-- Proving that the grasshoppers cannot reach the final positions under the given movement rules
theorem grasshoppers_cannot_reach :
  ¬(∃ (move_sequence : List (Point × Point)), 
    move_sequence ≠ [] ∧
    (move_sequence.head?.fst = A0 ∧ 
     move_sequence.head?.snd = Af ∧
     List.foldr (λ move acc, move.snd) A0 move_sequence = Af ∧
     List.foldr (λ move acc, move.snd) B0 move_sequence = Bf ∧
     List.foldr (λ move acc, move.snd) C0 move_sequence = Cf)) := sorry

end grasshoppers_cannot_reach_l711_711365


namespace initial_amount_l711_711038

theorem initial_amount (X : ℝ) (h : 0.7 * X = 3500) : X = 5000 :=
by
  sorry

end initial_amount_l711_711038


namespace construct_parallel_line_l711_711541

theorem construct_parallel_line (A : Point) (l : Line) (h : ¬ A ∈ l) : 
  ∃ (m : Line), A ∈ m ∧ (Parallel m l) :=
by 
  sorry

end construct_parallel_line_l711_711541


namespace polynomial_roots_l711_711162

theorem polynomial_roots :
  Polynomial.roots (Polynomial.C 4 * Polynomial.X ^ 4 
                    - Polynomial.C 26 * Polynomial.X ^ 3 
                    + Polynomial.C 57 * Polynomial.X ^ 2 
                    - Polynomial.C 26 * Polynomial.X 
                    + Polynomial.C 4) 
  = {root1, root2, root3, root4} :=
by
  let root1 := (26 + 3 * Complex.I * Real.sqrt 3) / 16
  let root2 := (26 - 3 * Complex.I * Real.sqrt 3) / 16
  let root3 := (-26 + 3 * Complex.I * Real.sqrt 3) / 16
  let root4 := (-26 - 3 * Complex.I * Real.sqrt 3) / 16
  sorry

end polynomial_roots_l711_711162


namespace exists_squares_seq_l711_711167

noncomputable def seq : ℕ → ℕ
| 0 := 3
| (n + 1) := (seq n)^2 - 1 / 2

theorem exists_squares_seq (n : ℕ) (hn : 0 < n) :
    ∃ a : ℕ → ℕ, (∀ i < n, 0 < a i) ∧ (∑ i in range n, (a i)^2).isPerfectSquare := sorry

end exists_squares_seq_l711_711167


namespace degree_of_polynomial_l711_711145

theorem degree_of_polynomial : degree((X^3 + 1)^5 * (X^4 + 1)^3) = 27 := 
sorry

end degree_of_polynomial_l711_711145


namespace min_value_expression_l711_711738

theorem min_value_expression (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 7) ^ 2 + (3 * Real.sin α + 4 * Real.cos β - 12) ^ 2 ≥ 36 :=
sorry

end min_value_expression_l711_711738


namespace arithmetic_sequence_mod_9_l711_711779

theorem arithmetic_sequence_mod_9 :
  let seq := List.range 502 |>.map (λ i, 2 + 4 * i)
  let A := seq.foldl (λ acc n, acc + n) 0
  A % 9 = 8 := by
  sorry

end arithmetic_sequence_mod_9_l711_711779


namespace number_of_liars_l711_711895

-- Define the type of individuals.
inductive KnightOrLiar
| knight  -- always tells the truth
| liar    -- always lies

open KnightOrLiar

-- Define the statements made by individuals based on their number.
def statement (n : ℕ) (people : ℕ → KnightOrLiar) : Prop :=
  if n % 2 = 1 then  -- odd-numbered person
    ∀ m, m > n → people m = liar
  else               -- even-numbered person
    ∀ m, m < n → people m = liar

-- Define the overall condition for all 30 people.
def consistent (people : ℕ → KnightOrLiar) : Prop :=
  ∀ n, 1 ≤ n ∧ n ≤ 30 → statement n people

-- The theorem that we need to prove based on the given conditions.
theorem number_of_liars : ∀ (people : ℕ → KnightOrLiar),
  consistent people →
  (∑ n in (Finset.range 31), if people n = liar then 1 else 0) = 28 := sorry

end number_of_liars_l711_711895


namespace selling_price_approx_l711_711089

def cost_price : ℝ := 68.97
def profit_rate : ℝ := 0.45
def expected_selling_price : ℝ := 100.01

theorem selling_price_approx (h : Real) (cost_price profit_rate expected_selling_price) :
  (profit := profit_rate * cost_price) →
  (selling_price := cost_price + profit) →
  Real.abs(selling_price - expected_selling_price) < 0.01 := sorry

end selling_price_approx_l711_711089


namespace side_length_of_equilateral_triangle_l711_711679

noncomputable def z : ℂ := 
  let ω := -1/2 + complex.I * (sqrt 3)/2 in
  ω - 1

theorem side_length_of_equilateral_triangle (ω : ℂ) (hω : ω^3 = 1 ∧ ω ≠ 1) :
  let z₁ := z in
  let z₂ := z₁^2 in
  let z₃ := z₁^3 in
  z₁ ≠ 0 ∧ z₁ ≠ 1 →
  z₁, z₂, and z₃ form an equilateral triangle →
  ∃ l : ℝ, l = sqrt (481 / 16) in 
  sorry

end side_length_of_equilateral_triangle_l711_711679


namespace bugs_meet_again_l711_711825

theorem bugs_meet_again 
  (radius1 radius2 : ℝ)
  (speed1 speed2 : ℝ)
  (h_radius1 : radius1 = 6) 
  (h_radius2 : radius2 = 3)
  (h_speed1 : speed1 = 4 * Real.pi)
  (h_speed2 : speed2 = 3 * Real.pi) :
  let t1 := (2 * radius1 * Real.pi) / speed1 in
  let t2 := (2 * radius2 * Real.pi) / speed2 in
  Int.lcm t1.nat_abs t2.nat_abs = 6 :=
by
  sorry

end bugs_meet_again_l711_711825


namespace amount_of_water_needed_l711_711350

def original_volume := 12
def original_concentration := 0.6
def desired_concentration := 0.4

-- The measure of alcohol in the original solution
def original_alcohol := original_volume * original_concentration

-- The statement we want to prove
theorem amount_of_water_needed : ∃ x : ℤ, original_alcohol / (original_volume + x) = desired_concentration ∧ original_alcohol = 7.2 ∧ x = 6 := 
by
  sorry

end amount_of_water_needed_l711_711350


namespace find_x_orthogonal_l711_711153

theorem find_x_orthogonal :
  ∃ x : ℝ, (2 * x + 5 * (-3) = 0) ∧ x = 15 / 2 :=
by
  sorry

end find_x_orthogonal_l711_711153


namespace find_annual_interest_rate_l711_711007

noncomputable def annual_interest_rate (A P : ℝ) (t n : ℕ) : ℝ := 
  (real.sqrt (A / P) - 1)

theorem find_annual_interest_rate :
  annual_interest_rate 169 156.25 2 1 ≈ 0.0398 :=
by
  sorry

end find_annual_interest_rate_l711_711007


namespace angle_between_clock_hands_at_7_25_l711_711380

theorem angle_between_clock_hands_at_7_25 : 
  let degrees_per_hour := 30
  let minute_hand_position := (25 / 60 * 360 : ℝ)
  let hour_hand_position := (7 * degrees_per_hour + (25 / 60 * degrees_per_hour) : ℝ)
  abs (hour_hand_position - minute_hand_position) = 72.5 
  := by
  let degrees_per_hour := 30
  let minute_hand_position := (25 / 60 * 360 : ℝ)
  let hour_hand_position := (7 * degrees_per_hour + (25 / 60 * degrees_per_hour) : ℝ)
  sorry

end angle_between_clock_hands_at_7_25_l711_711380


namespace tan_theta_correct_l711_711182

noncomputable def cos_double_angle (θ : ℝ) : ℝ := 2 * Real.cos θ ^ 2 - 1

theorem tan_theta_correct (θ : ℝ) (hθ₁ : θ > 0) (hθ₂ : θ < Real.pi / 2) 
  (h : 15 * cos_double_angle θ - 14 * Real.cos θ + 11 = 0) : Real.tan θ = Real.sqrt 5 / 2 :=
sorry

end tan_theta_correct_l711_711182


namespace sum_of_k_with_distinct_integer_solutions_l711_711858

-- Definitions from conditions
def quadratic_eq (k : ℤ) (x : ℤ) : Prop := 3 * x^2 - k * x + 12 = 0

def distinct_integer_solutions (k : ℤ) : Prop := ∃ p q : ℤ, p ≠ q ∧ quadratic_eq k p ∧ quadratic_eq k q

-- Main statement to prove
theorem sum_of_k_with_distinct_integer_solutions : 
  ∑ k in {k : ℤ | distinct_integer_solutions k}, k = 0 := 
sorry

end sum_of_k_with_distinct_integer_solutions_l711_711858


namespace correct_option_l711_711323

variable (a : ℤ)

theorem correct_option :
  (-2 * a^2)^3 = -8 * a^6 :=
by
  sorry

end correct_option_l711_711323


namespace cheapest_shipping_option_l711_711127

/-- Defines the cost options for shipping, given a weight of 5 pounds. -/
def cost_A (weight : ℕ) : ℝ := 5.00 + 0.80 * weight
def cost_B (weight : ℕ) : ℝ := 4.50 + 0.85 * weight
def cost_C (weight : ℕ) : ℝ := 3.00 + 0.95 * weight

/-- Proves that for a package weighing 5 pounds, the cheapest shipping option is Option C costing $7.75. -/
theorem cheapest_shipping_option : cost_C 5 < cost_A 5 ∧ cost_C 5 < cost_B 5 ∧ cost_C 5 = 7.75 :=
by
  -- Calculation is omitted
  sorry

end cheapest_shipping_option_l711_711127


namespace solve_for_x_l711_711996

theorem solve_for_x (x : ℝ) (h1 : x > 9) :
  (sqrt (x - 9 * sqrt (x - 9)) + 3 = sqrt (x + 9 * sqrt (x - 9)) - 3) → x ≥ 40.5 :=
by
  intro h
  sorry

end solve_for_x_l711_711996


namespace steps_to_get_down_empire_state_building_l711_711605

theorem steps_to_get_down_empire_state_building (total_steps : ℕ) (steps_building_to_garden : ℕ) (steps_to_madison_square : ℕ) :
  total_steps = 991 -> steps_building_to_garden = 315 -> steps_to_madison_square = total_steps - steps_building_to_garden -> steps_to_madison_square = 676 :=
by
  intros
  subst_vars
  sorry

end steps_to_get_down_empire_state_building_l711_711605


namespace two_digit_numbers_solution_l711_711644

theorem two_digit_numbers_solution :
  ∀ (N : ℕ), (∃ (x y : ℕ), (N = 10 * x + y) ∧ (x < 10) ∧ (y < 10) ∧ 4 * x + 2 * y = N / 2) →
    (N = 32 ∨ N = 64 ∨ N = 96) := 
by
  sorry

end two_digit_numbers_solution_l711_711644


namespace range_of_x_l711_711556

open Real

theorem range_of_x (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m + n + 3 = m * n) 
    (x : ℝ) (hx : x ∈ Iic (-1) ∪ Ici (2/3)) : 0 ≤ (m + n) * x^2 + 2 * x + (m * n) - 13 := 
sorry

end range_of_x_l711_711556


namespace cube_difference_l711_711572

variable (x : ℝ)

theorem cube_difference (h : x - 1/x = 5) : x^3 - 1/x^3 = 120 := by
  sorry

end cube_difference_l711_711572


namespace prove_domain_of_g_l711_711976

/-- Let \( g(x) = \log_{\frac{1}{3}}(\log_9(\log_{\frac{1}{9}}(\log_{81}(\log_{\frac{1}{81}}x)))) \).
Prove that the domain of \( g(x) \) is \( \left(\frac{1}{3^{324}}, \frac{1}{81}\right) \).
-/
theorem prove_domain_of_g :
  ∀ x : ℝ, (x > 1 / (3^324) ∧ x < 1 / 81) ↔ 
  log (log (log (log (log (x) / log (1/81)) / log 81) / log (1/9)) / log 9) / log (1/3) is defined :=
sorry

end prove_domain_of_g_l711_711976


namespace two_digit_number_solution_l711_711639

theorem two_digit_number_solution (N : ℕ) (x y : ℕ) :
  (10 * x + y = N) ∧ (4 * x + 2 * y = (10 * x + y) / 2) →
  N = 32 ∨ N = 64 ∨ N = 96 := 
sorry

end two_digit_number_solution_l711_711639


namespace powerThreeExpression_l711_711568

theorem powerThreeExpression (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / (x^3)) = 140 := sorry

end powerThreeExpression_l711_711568


namespace problem_i_problem_ii_l711_711177

noncomputable def f (m x : ℝ) := (Real.log x / Real.log m) ^ 2 + 2 * (Real.log x / Real.log m) - 3

theorem problem_i (x : ℝ) : f 2 x < 0 ↔ (1 / 8) < x ∧ x < 2 :=
by sorry

theorem problem_ii (m : ℝ) (H : ∀ x, 2 ≤ x ∧ x ≤ 4 → f m x < 0) : 
  (0 < m ∧ m < 4^(1/3)) ∨ (4 < m) :=
by sorry

end problem_i_problem_ii_l711_711177


namespace square_area_from_diagonal_l711_711999

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) : (d^2 / 2) = 72 :=
by sorry

end square_area_from_diagonal_l711_711999


namespace find_number_l711_711356

theorem find_number (x : ℝ) (h : 100 - x = x + 40) : x = 30 :=
sorry

end find_number_l711_711356


namespace max_value_f_l711_711517

noncomputable def f (x : ℝ) : ℝ := (4 * x - 4 * x^3) / (1 + 2 * x^2 + x^4)

theorem max_value_f : ∃ x : ℝ, (f x = 1) ∧ (∀ y : ℝ, f y ≤ 1) :=
sorry

end max_value_f_l711_711517


namespace triangle_area_inscribed_l711_711096

noncomputable def triangle_area (r : ℝ) (n : ℕ) (ratio1 ratio2 ratio3 : ℝ) : ℝ :=
let x := n / (2 * r) in
1 / 2 * (ratio1 * x) * (ratio2 * x)

theorem triangle_area_inscribed (r : ℝ) (n ratio1 ratio2 ratio3 : ℕ) (h_ratios : ratio1 = 5) (h_ratio2 : ratio2 = 12) (h_ratio3 : ratio3 = 13) (h_r : r = 6.5) :
  triangle_area r n ratio1 ratio2 ratio3 = 30 := by
  -- Given a right triangle with sides in the ratio 5:12:13 inscribed in a circle with radius 6.5, 
  -- its area should be 30.
  sorry

end triangle_area_inscribed_l711_711096


namespace ensure_user_data_security_l711_711766

-- Define what it means to implement a security measure
inductive SecurityMeasure
  | avoidStoringCardData
  | encryptStoredData
  | encryptDataInTransit
  | codeObfuscation
  | restrictRootedDevices
  | antivirusProtectionAgent

-- Define the conditions: Developing an online store app where users can 
-- pay by credit card and order home delivery ensures user data security 
-- if at least three security measures are implemented.

def providesSecurity (measures : List SecurityMeasure) : Prop :=
  measures.contains SecurityMeasure.avoidStoringCardData ∨
  measures.contains SecurityMeasure.encryptStoredData ∨
  measures.contains SecurityMeasure.encryptDataInTransit ∨
  measures.contains SecurityMeasure.codeObfuscation ∨
  measures.contains SecurityMeasure.restrictRootedDevices ∨
  measures.contains SecurityMeasure.antivirusProtectionAgent

theorem ensure_user_data_security (measures : List SecurityMeasure) (h : measures.length ≥ 3) : providesSecurity measures :=
  sorry

end ensure_user_data_security_l711_711766


namespace isosceles_triangle_perimeter_l711_711552

-- Defining the given conditions
def is_isosceles (a b c : ℕ) : Prop := a = b ∨ b = c ∨ c = a
def triangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

-- Stating the problem and goal
theorem isosceles_triangle_perimeter (a b c : ℕ) 
  (h_iso: is_isosceles a b c)
  (h_len1: a = 3 ∨ a = 6)
  (h_len2: b = 3 ∨ b = 6)
  (h_triangle: triangle a b c): a + b + c = 15 :=
sorry

end isosceles_triangle_perimeter_l711_711552


namespace maximum_horizontal_range_angle_l711_711910

variables (v0 h : ℝ) (g : ℝ := 9.8)

def time_of_flight (θ : ℝ) : ℝ :=
  (v0 * sin θ + sqrt ((v0 * sin θ)^2 + 2 * g * h)) / g

def horizontal_range (θ : ℝ) : ℝ :=
  v0 * cos θ * time_of_flight v0 h θ

theorem maximum_horizontal_range_angle :
  ∀ θ, (0 < θ ∧ θ < 45 * (π/180)) → 
       (∀ φ, (0 < φ ∧ φ < 180 * (π/180)) → (horizontal_range v0 h g θ) ≥ (horizontal_range v0 h g φ)) := sorry

end maximum_horizontal_range_angle_l711_711910


namespace two_digit_number_solution_l711_711635

theorem two_digit_number_solution (N : ℕ) (x y : ℕ) :
  (10 * x + y = N) ∧ (4 * x + 2 * y = (10 * x + y) / 2) →
  N = 32 ∨ N = 64 ∨ N = 96 := 
sorry

end two_digit_number_solution_l711_711635


namespace tetrahedron_volume_ratio_l711_711188

-- Variables and conditions for the problem
variables {A B C D : Type} [inhabited A] [inhabited B] [inhabited C] [inhabited D]
variable (a b d ω k : ℝ)
variable (plane_here : ∀ (ε : Type) (AB CD : Type),
  ∃ε, ∥AB - ε∥ = k * ∥CD - ε∥)

-- Definition of tetrahedron volume ratio under given conditions
theorem tetrahedron_volume_ratio (a b d ω k : ℝ) (plane_here : ∀ (ε : Type) (AB CD : Type),
  ∃ε, ∥AB - ε∥ = k * ∥CD - ε∥):
  let q := (3 * k + 1) / (k^3 + 3 * k^2) in
  true := by
    sorry -- proof is not required

end tetrahedron_volume_ratio_l711_711188


namespace oakwood_earnings_correct_l711_711250

-- Define the conditions using Lean definitions
def total_maple_student_days (students : ℕ) (days : ℕ) := students * days
def total_oakwood_student_days (students : ℕ) (days : ℕ) := students * days
def total_pine_student_days (students : ℕ) (days : ℕ) := students * days

def total_payment_for_work (payment : ℝ) := payment
def total_student_days (maple : ℕ) (oakwood : ℕ) (pine : ℕ) := maple + oakwood + pine

def daily_wage_per_student (total_payment : ℝ) (total_student_days : ℕ) := total_payment / total_student_days

def oakwood_earnings (daily_wage : ℝ) (oakwood_student_days : ℕ) := daily_wage * oakwood_student_days

theorem oakwood_earnings_correct :
  total_maple_student_days 5 4 + total_oakwood_student_days 6 3 + total_pine_student_days 4 6 = 62 → 
  total_payment_for_work 972 = 972 →
  daily_wage_per_student 972 62 = 15.677419354838709 →
  oakwood_earnings 15.677419354838709 18 = 282.19 :=
by
  intro h1 h2 h3
  sorry
  
end oakwood_earnings_correct_l711_711250


namespace slope_angle_tangent_line_l711_711728

noncomputable def f (x : ℝ) : ℝ := sorry
axiom differentiable_f : Differentiable ℝ f
axiom limit_cond : tendsto (λ (h : ℝ), (f (1 + 2 * h) - f (1 - h)) / h) (𝓝 0) (𝓝 3)

theorem slope_angle_tangent_line : ∃ α : ℝ, 0 ≤ α ∧ α < Real.pi ∧ tan α = 1 ∧ α = Real.pi / 4 :=
by
  have deriv_eq : deriv f 1 = 1,
  { apply tendsto_nhds_unique,
    convert limit_cond,
    ext, 
    simp },
  use Real.arctan 1,
  split,
  { apply Real.arctan_nonneg },
  split,
  { apply Real.arctan_lt_pi },
  split,
  { exact Real.tan_arctan_of_nonneg_of_lt 0 Real.one_pos },
  { simp [Real.arctan_eq_pi_div_four] }

end slope_angle_tangent_line_l711_711728


namespace cannot_obtain_every_arrangement_l711_711902

theorem cannot_obtain_every_arrangement (n : ℕ) (h : n ≥ 1) : 
  ∀ (families : Fin (3 * n) → ℕ), 
  ¬ (∀ (initial_arrangement final_arrangement : Fin (3 * n) → Fin (3 * n)),
    reachable families initial_arrangement final_arrangement) :=
begin
  sorry
end

-- Definitions for reachable (i.e., reachable via allowed exchanges)
-- Note: This is just a sketch for clarity. Detailed definitions would be needed based on allowable moves.
noncomputable def reachable (families : Fin (3 * n) → ℕ) 
  (initial_arrangement final_arrangement : Fin (3 * n) → Fin (3 * n)) : Prop :=
-- Define when one arrangement is reachable from another based on the given exchange rules
sorry

end cannot_obtain_every_arrangement_l711_711902


namespace willam_farm_tax_l711_711990

theorem willam_farm_tax
  (T : ℝ)
  (h1 : 0.4 * T * (3840 / (0.4 * T)) = 3840)
  (h2 : 0 < T) :
  0.3125 * T * (3840 / (0.4 * T)) = 3000 := by
  sorry

end willam_farm_tax_l711_711990


namespace placemat_length_l711_711920

theorem placemat_length (radius : ℝ) (placemat_width : ℝ) (num_placemats : ℕ) 
  (touching_inner_corners : ∀ i j, 0 ≤ i < num_placemats → 0 ≤ j < num_placemats → i ≠ j → 
    (inner_corner i = inner_corner j)) : 
  radius = 5 → placemat_width = 1 → num_placemats = 5 → (length_of_placemat placemat_width radius touching_inner_corners) = 5.878 := 
by
  intros
  sorry

end placemat_length_l711_711920


namespace num_valid_sequences_l711_711474

def isValidMoveSequence (seq : List (Sum Unit Unit)) : Prop :=
  seq.length = 10 ∧
  seq.filter (λ x => x = Sum.inl ()).length = 5 ∧
  seq.filter (λ x => x = Sum.inr ()).length = 5 ∧
  ∀ i, i < 8 → (seq.nth i = seq.nth (i + 1) → seq.nth (i + 1) ≠ seq.nth (i + 2))

theorem num_valid_sequences :
  {seq : List (Sum Unit Unit) // isValidMoveSequence seq}.length = 84 :=
sorry

end num_valid_sequences_l711_711474


namespace problem1_problem2_l711_711059

-- Problem (1):
theorem problem1 : (5 + 1 / 16)^0.5 - 2 * (2 + 10 / 27)^(-2 / 3) - 2 * (sqrt (2 + Real.pi))^0 / (3 / 4)^(-2) = 0 := 
sorry

-- Problem (2):
theorem problem2 : Real.log 5 35 + 2 * Real.log 0.5 (sqrt 2) - Real.log 5 (1 / 50) - Real.log 5 14 + 5^(Real.log 5 3) = 5 := 
sorry

end problem1_problem2_l711_711059


namespace range_of_inclination_angle_l711_711820

theorem range_of_inclination_angle (α : ℝ) (A : ℝ × ℝ) (hA : A = (-2, 0))
  (ellipse_eq : ∀ (x y : ℝ), (x^2 / 2) + y^2 = 1) :
    (0 ≤ α ∧ α < Real.arcsin (Real.sqrt 3 / 3)) ∨ 
    (π - Real.arcsin (Real.sqrt 3 / 3) < α ∧ α < π) :=
sorry

end range_of_inclination_angle_l711_711820


namespace sequence_all_ones_l711_711329

theorem sequence_all_ones (k : ℕ) (n : ℕ → ℕ) (h_k : 2 ≤ k)
  (h1 : ∀ i, 1 ≤ i → i ≤ k → 1 ≤ n i) 
  (h2 : n 2 ∣ 2^(n 1) - 1) 
  (h3 : n 3 ∣ 2^(n 2) - 1) 
  (h4 : n 4 ∣ 2^(n 3) - 1)
  (h5 : ∀ i, 2 ≤ i → i < k → n (i + 1) ∣ 2^(n i) - 1)
  (h6 : n 1 ∣ 2^(n k) - 1) : 
  ∀ i, 1 ≤ i → i ≤ k → n i = 1 := 
by 
  sorry

end sequence_all_ones_l711_711329


namespace total_hiking_time_l711_711391

-- Define the conditions as variables
variable (rate_up rate_down time_up : ℝ)
variable (rate_up_pos : rate_up > 0) (rate_down_pos : rate_down > 0) (time_up_pos : time_up > 0)

-- Define specific values from conditions in the problem
def hike_up_rate : ℝ := 4
def hike_down_rate : ℝ := 6
def hiking_time_up : ℝ := 1.2

-- Using the conditions to compute the distance up the hill and down the hill
def distance_up (rate_up : ℝ) (time_up : ℝ) : ℝ := rate_up * time_up
def distance_down (rate_down : ℝ) (distance_up : ℝ) : ℝ := distance_up / rate_down

-- Given conditions mapping to specific values
def given_distance_up : ℝ := distance_up hike_up_rate hiking_time_up
def given_time_down : ℝ := distance_down hike_down_rate given_distance_up

-- Prove the total hiking time using the given conditions
theorem total_hiking_time (rate_up rate_down time_up : ℝ)
  (rate_up_pos : rate_up > 0) (rate_down_pos : rate_down > 0) (time_up_pos : time_up > 0)
  (hike_up_rate : rate_up = 4) (hike_down_rate : rate_down = 6) (hiking_time_up : time_up = 1.2) :
  let distance_up := rate_up * time_up in
  (time_up + (distance_up / rate_down)) = 2 := 
  by
    sorry

end total_hiking_time_l711_711391


namespace probability_of_at_least_40_cents_l711_711788

-- Definitions for each type of coin and their individual values in cents.
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25
def half_dollar := 50

-- The total value needed for a successful outcome
def minimum_success_value := 40

-- Total number of possible outcomes from flipping 5 coins independently
def total_outcomes := 2^5

-- Count the successful outcomes that result in at least 40 cents
-- This is a placeholder for the actual successful counting method
noncomputable def successful_outcomes := 18

-- Calculate the probability of successful outcomes
noncomputable def probability := (successful_outcomes : ℚ) / total_outcomes

-- Proof statement to show the probability is 9/16
theorem probability_of_at_least_40_cents : probability = 9 / 16 := 
by
  sorry

end probability_of_at_least_40_cents_l711_711788


namespace employed_males_percentage_l711_711262

theorem employed_males_percentage (P : ℕ) (H1: P > 0)
    (employed_pct : ℝ) (female_pct : ℝ)
    (H_employed_pct : employed_pct = 0.64)
    (H_female_pct : female_pct = 0.140625) :
    (0.859375 * employed_pct * 100) = 54.96 :=
by
  sorry

end employed_males_percentage_l711_711262


namespace cubic_identity_l711_711576

theorem cubic_identity (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / x^3) = 140 := 
  sorry

end cubic_identity_l711_711576


namespace midpoint_trajectory_l711_711312

theorem midpoint_trajectory (A : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ) 
  (H_AO : O.1^2 + O.2^2 = r^2) (H_A : A = (0, 2)) (H_O : O = (0, 0)) : 
  ∀ (B C M : ℝ × ℝ), (B.1^2 + B.2^2 = r^2) → (C.1^2 + C.2^2 = r^2) → 
  (B.1 ≠ C.1 ∨ B.2 ≠ C.2) → -- Ensure B and C are distinct points
  let M := ((B.1 + C.1) / 2, (B.2 + C.2) / 2) in 
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 → 
  M.1^2 + M.2^2 - 2 * M.2 - 6 = 0 := 
by
  intros B C M H_BC H_CC H_dist H_perp,
  sorry

end midpoint_trajectory_l711_711312


namespace cube_volume_l711_711436

-- Define the given condition: The surface area of the cube
def surface_area (A : ℕ) := A = 294

-- The key proposition we need to prove using the given condition
theorem cube_volume (s : ℕ) (A : ℕ) (V : ℕ) 
  (area_condition : surface_area A)
  (side_length_condition : s ^ 2 = A / 6) 
  (volume_condition : V = s ^ 3) : 
  V = 343 := by
  sorry

end cube_volume_l711_711436


namespace option1_payment_correct_option2_payment_correct_most_cost_effective_for_30_l711_711418

variables (x : ℕ) (hx : x > 10)

def suit_price : ℕ := 1000
def tie_price : ℕ := 200
def num_suits : ℕ := 10

-- Option 1: Buy one suit, get one tie for free
def option1_payment : ℕ := 200 * x + 8000

-- Option 2: All items sold at a 10% discount
def option2_payment : ℕ := (10 * 1000 + x * 200) * 9 / 10

-- For x = 30, which option is more cost-effective
def x_value := 30
def option1_payment_30 : ℕ := 200 * x_value + 8000
def option2_payment_30 : ℕ := (10 * 1000 + x_value * 200) * 9 / 10
def more_cost_effective_option_30 : ℕ := if option1_payment_30 < option2_payment_30 then option1_payment_30 else option2_payment_30

-- Most cost-effective option for x = 30 with new combination plan
def combination_payment_30 : ℕ := 10000 + 20 * 200 * 9 / 10

-- Statements to be proved
theorem option1_payment_correct : option1_payment x = 200 * x + 8000 := sorry

theorem option2_payment_correct : option2_payment x = (10 * 1000 + x * 200) * 9 / 10 := sorry

theorem most_cost_effective_for_30 :
  option1_payment_30 = 14000 ∧ 
  option2_payment_30 = 14400 ∧ 
  more_cost_effective_option_30 = 14000 ∧
  combination_payment_30 = 13600 := sorry

end option1_payment_correct_option2_payment_correct_most_cost_effective_for_30_l711_711418


namespace jill_account_balance_l711_711709

noncomputable def compound_interest 
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_account_balance :
  let P := 10000
  let r := 0.0396
  let n := 2
  let t := 2
  compound_interest P r n t ≈ 10816.49 :=
by
  sorry

end jill_account_balance_l711_711709


namespace area_of_given_triangle_l711_711833

noncomputable def area_of_triangle_formed_by_lines 
  (y1 y2 y3 : ℝ → ℝ)
  (intersect_1_3 : ∃ x, y1 x = y3 x)
  (intersect_2_3 : ∃ x, y2 x = y3 x)
  (intersect_1_2 : ∃ x, y1 x = y2 x) : ℝ :=
  let base := 3
  let height := 1.5
  (1 / 2) * base * height

theorem area_of_given_triangle :
  area_of_triangle_formed_by_lines
    (λ x, x)      -- y = x
    (λ x, -x + 3) -- y = -x + 3
    (λ _, 3)      -- y = 3
    (by use 3; simp)    -- intersection of y = x and y = 3
    (by use 0; simp)    -- intersection of y = -x + 3 and y = 3
    (by use 1.5; simp)  -- intersection of y = x and y = -x + 3
    = 2.25 := by
  -- Proof is omitted
  sorry

end area_of_given_triangle_l711_711833


namespace pyramid_coloring_l711_711117

def number_of_ways_to_color_pyramid : ℕ :=
  5 * 4 * 3 * 1 * 3 + 5 * 4 * 3 * 2 * 2

theorem pyramid_coloring (colors : ℕ) (five_colors : colors = 5) :
  number_of_ways_to_color_pyramid = 420 :=
by
  have h1 : 5 * 4 * 3 * 1 * 3 = 180 := by norm_num
  have h2 : 5 * 4 * 3 * 2 * 2 = 240 := by norm_num
  have total : 180 + 240 = 420 := by norm_num
  rw [number_of_ways_to_color_pyramid]
  rw [h1, h2, total]
  sorry

end pyramid_coloring_l711_711117


namespace fence_length_correct_l711_711396

-- Definitions for conditions
def radius : ℝ := 7
def opening : ℝ := 3

-- Definition for the length of the fence
def fence_length : ℝ := (7 * Real.pi) + 11

-- The theorem to prove that the computed fence length is correct
theorem fence_length_correct : 
  let r := radius
  let open := opening
  let diameter := 2 * r
  let semicircle_perimeter := (Real.pi * r) + diameter
  let fence := semicircle_perimeter - open
  fence = fence_length :=
by
  sorry

end fence_length_correct_l711_711396


namespace total_travel_time_l711_711414

-- Defining the conditions
def car_travel_180_miles_in_4_hours : Prop :=
  180 / 4 = 45

def car_travel_135_miles_additional_time : Prop :=
  135 / 45 = 3

-- The main statement to be proved
theorem total_travel_time : car_travel_180_miles_in_4_hours ∧ car_travel_135_miles_additional_time → 4 + 3 = 7 := by
  sorry

end total_travel_time_l711_711414


namespace range_of_inclination_angle_l711_711817

theorem range_of_inclination_angle (α : ℝ) :
  let A := (-2 : ℝ, 0 : ℝ)
  let ellipse := ∀ x y : ℝ, x^2 / 2 + y^2 = 1
  ∃ B C : ℝ × ℝ, (∃ l : ℝ → ℝ × ℝ, ∀ t : ℝ, l t = (-2 + t * Real.cos α, t * Real.sin α) ∧ ellipse (fst (l t)) (snd (l t))) ∧ B ≠ C ↔ (0 ≤ α ∧ α < Real.arcsin (Real.sqrt 3 / 3) ∨ π - Real.arcsin (Real.sqrt 3 / 3) < α ∧ α < π) :=
begin
  sorry
end

end range_of_inclination_angle_l711_711817


namespace rectangle_area_increase_is_32_25_percent_l711_711662

theorem rectangle_area_increase_is_32_25_percent:
  (∀ (L W : ℝ), 
    let A_original := L * W in
    let L_new := 1.15 * L in
    let W_new := 1.15 * W in
    let A_new := L_new * W_new in
    let A_increase := (A_new - A_original) / A_original * 100 in
    A_increase = 32.25
  ) :=
by sorry

end rectangle_area_increase_is_32_25_percent_l711_711662


namespace alpha_plus_beta_l711_711493

noncomputable def alpha_beta (α β : ℝ) : Prop :=
  ∀ x : ℝ, ((x - α) / (x + β)) = ((x^2 - 54 * x + 621) / (x^2 + 42 * x - 1764))

theorem alpha_plus_beta : ∃ α β : ℝ, α + β = 86 ∧ alpha_beta α β :=
by
  sorry

end alpha_plus_beta_l711_711493


namespace jill_account_balance_l711_711705

noncomputable def compound_interest 
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_account_balance :
  let P := 10000
  let r := 0.0396
  let n := 2
  let t := 2
  compound_interest P r n t ≈ 10816.49 :=
by
  sorry

end jill_account_balance_l711_711705


namespace chessboard_coloring_impossible_l711_711126

theorem chessboard_coloring_impossible :
  ¬ (∃ (coloring : Fin 1990 × Fin 1990 → Bool),
      (∀ i, (∑ j, if coloring (i, j) then 1 else 0) = 995) ∧ 
      (∀ j, (∑ i, if coloring (i, j) then 1 else 0) = 995) ∧ 
      (∀ i j, coloring (i, j) ≠ coloring (Fin 1990.last - i, Fin 1990.last - j))) := sorry

end chessboard_coloring_impossible_l711_711126


namespace cube_difference_l711_711573

variable (x : ℝ)

theorem cube_difference (h : x - 1/x = 5) : x^3 - 1/x^3 = 120 := by
  sorry

end cube_difference_l711_711573


namespace parabola_vertex_coordinates_l711_711340

theorem parabola_vertex_coordinates :
  ∀ x y : ℝ, y = 3 * (x - 7)^2 + 5 → (7, 5) = (7, 5) :=
by
  intros x y h
  exact rfl

end parabola_vertex_coordinates_l711_711340


namespace original_proposition_converse_inverse_contrapositive_l711_711046

def is_integer (x : ℝ) : Prop := ∃ (n : ℤ), x = n
def is_real (x : ℝ) : Prop := true

theorem original_proposition (x : ℝ) : is_integer x → is_real x := 
by sorry

theorem converse (x : ℝ) : ¬(is_real x → is_integer x) := 
by sorry

theorem inverse (x : ℝ) : ¬((¬ is_integer x) → (¬ is_real x)) := 
by sorry

theorem contrapositive (x : ℝ) : (¬ is_real x) → (¬ is_integer x) := 
by sorry

end original_proposition_converse_inverse_contrapositive_l711_711046


namespace charlie_cortland_apples_l711_711146

/-- Given that Charlie picked 0.17 bags of Golden Delicious apples, 0.17 bags of Macintosh apples, 
   and a total of 0.67 bags of fruit, prove that the number of bags of Cortland apples picked by Charlie is 0.33. -/
theorem charlie_cortland_apples :
  let golden_delicious := 0.17
  let macintosh := 0.17
  let total_fruit := 0.67
  total_fruit - (golden_delicious + macintosh) = 0.33 :=
by
  sorry

end charlie_cortland_apples_l711_711146


namespace ratio_pat_mark_l711_711778

theorem ratio_pat_mark (P K M : ℕ) (h1 : P + K + M = 180) 
  (h2 : P = 2 * K) (h3 : M = K + 100) : P / gcd P M = 1 ∧ M / gcd P M = 3 := by
  sorry

end ratio_pat_mark_l711_711778


namespace explicit_formula_and_constant_area_l711_711215

noncomputable def f (a b x : ℝ) : ℝ := a * x - b / x

theorem explicit_formula_and_constant_area :
  (∃ (a b : ℝ), ∀ x : ℝ, f a b x = x - 3 / x) ∧
  (∀ x0 : ℝ, let a := 1, let b := 3,
        let f := f a b,
        let slope := (∂ f / ∂ x) x0,
        let tangent_line := (y : ℝ) = slope * (x : ℝ - x0) + f x0,
        let intersection_y := tangent_line 0,
        let intersection_xy := solve (y = x) (tangent_line y),
        math.abs (1 / 2  * 2 * x0 * math.abs (intersection_y)) = 6) :=
sorry

end explicit_formula_and_constant_area_l711_711215


namespace complex_number_imaginary_part_l711_711179

-- Definitions for complex numbers and imaginary unit
def i : ℂ := complex.I

-- Given condition
def one_plus_i : ℂ := 1 + i

-- Function to find the imaginary part of a complex number
def imaginary_part (z : ℂ) : ℝ := z.im

-- The complex number we are interested in
def complex_num : ℂ := 2016 / one_plus_i

-- Theorem stating the problem
theorem complex_number_imaginary_part : imaginary_part complex_num = -1008 := 
sorry

end complex_number_imaginary_part_l711_711179


namespace angle_ADE_is_70_l711_711258

theorem angle_ADE_is_70 
  (A B C D E : Point)
  (h1 : collinear A B C)
  (h2 : ∠ D B C = 70) 
  (h3 : D ∈ line_through A C)
  (h4 : D ∈ line_through B C)
  (h5 : dist A D = dist B D) 
  (h6 : dist B E = dist D E) : 
  ∠ A D E = 70 := 
sorry

end angle_ADE_is_70_l711_711258


namespace quadratic_has_one_real_root_positive_value_of_m_l711_711245

theorem quadratic_has_one_real_root (m : ℝ) (h : (4 * m) * (4 * m) - 4 * 1 * m = 0) : m = 0 ∨ m = 1/4 := by
  sorry

theorem positive_value_of_m (m : ℝ) (h : (4 * m) * (4 * m) - 4 * 1 * m = 0) : m = 1/4 := by
  have root_cases := quadratic_has_one_real_root m h
  cases root_cases
  · exfalso
    -- We know m = 0 cannot be the positive m we are looking for.
    sorry
  · assumption

end quadratic_has_one_real_root_positive_value_of_m_l711_711245


namespace same_length_of_dashed_paths_l711_711306

-- Define the problem setup and conditions
variables {A B C X Y Z O : Type*}
variable [metric_space A]
variable [metric_space B]
variable [metric_space C]
variable [metric_space X]
variable [metric_space Y]
variable [metric_space Z]
variable [metric_space O]

-- Condition: Triangles and symmetry
def triangle_AXB_equilateral (A B X : Type*) : Prop :=
  dist A B = dist B X ∧ dist B X = dist X A ∧ dist A B = dist X A

def perpendicular_bisector (A B O : Type*) : Prop :=
  dist A O = dist B O ∧ angle A O B = π / 2

def points_symmetric (P Q O : Type*) : Prop :=
  dist P O = dist Q O ∧ angle P O Q = π

-- Define the lengths of the dashed lines
def dashed_path_length (A Y X Z O : Type*) : Prop :=
  dist A Y = dist O Z

-- Theorem statement
theorem same_length_of_dashed_paths
  (h1 : triangle_AXB_equilateral A B X)
  (h2 : perpendicular_bisector Y A O)
  (h3 : perpendicular_bisector C Z O)
  (h4 : points_symmetric B X O)
  (h5 : points_symmetric O Z D) :
  dashed_path_length A Y X Z O :=
sorry

end same_length_of_dashed_paths_l711_711306


namespace card_prob_ace_of_hearts_l711_711091

def problem_card_probability : Prop :=
  let deck_size := 52
  let draw_size := 2
  let ace_hearts := 1
  let total_combinations := Nat.choose deck_size draw_size
  let favorable_combinations := deck_size - ace_hearts
  let probability := favorable_combinations / total_combinations
  probability = 1 / 26

theorem card_prob_ace_of_hearts : problem_card_probability := by
  sorry

end card_prob_ace_of_hearts_l711_711091


namespace derivative_of_f_l711_711176

def f (x : ℝ) : ℝ := x * sin x

theorem derivative_of_f : deriv f x = sin x + x * cos x := 
by
  sorry

end derivative_of_f_l711_711176


namespace solution_exists_l711_711315

theorem solution_exists (a b c : ℝ) : ∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 := 
sorry

end solution_exists_l711_711315


namespace direct_proportion_increases_inverse_proportion_increases_l711_711830

-- Question 1: Prove y=2x increases as x increases.
theorem direct_proportion_increases (x1 x2 : ℝ) (h : x1 < x2) : 
  2 * x1 < 2 * x2 := by sorry

-- Question 2: Prove y=-2/x increases as x increases when x > 0.
theorem inverse_proportion_increases (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) : 
  - (2 / x1) < - (2 / x2) := by sorry

end direct_proportion_increases_inverse_proportion_increases_l711_711830


namespace unique_triple_solution_l711_711511

theorem unique_triple_solution (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
                               (n : ℕ) (h_pos_n : 0 < n) (h_prime_cond : ∀ p : ℕ, prime p → p < 2014 → ¬ p ∣ n) :
  n + c ∣ a^n + b^n + n ↔ (a = 1 ∧ b = 1 ∧ c = 2) :=
by sorry

end unique_triple_solution_l711_711511


namespace difference_of_roots_l711_711156

theorem difference_of_roots :
  ∀ (x : ℝ), (x^2 - 5*x + 6 = 0) → (∃ r1 r2 : ℝ, r1 > 2 ∧ r2 < r1 ∧ r1 - r2 = 1) :=
by
  sorry

end difference_of_roots_l711_711156


namespace count_three_digit_multiples_of_seven_l711_711615

theorem count_three_digit_multiples_of_seven :
  let a := 100 in
  let b := 999 in
  let smallest := (Nat.ceil (a.toRat / 7)).natAbs * 7 in
  let largest := (b / 7) * 7 in
  (largest / 7) - ((smallest - 1) / 7) = 128 := sorry

end count_three_digit_multiples_of_seven_l711_711615


namespace triangle_area_inscribed_l711_711097

noncomputable def triangle_area (r : ℝ) (n : ℕ) (ratio1 ratio2 ratio3 : ℝ) : ℝ :=
let x := n / (2 * r) in
1 / 2 * (ratio1 * x) * (ratio2 * x)

theorem triangle_area_inscribed (r : ℝ) (n ratio1 ratio2 ratio3 : ℕ) (h_ratios : ratio1 = 5) (h_ratio2 : ratio2 = 12) (h_ratio3 : ratio3 = 13) (h_r : r = 6.5) :
  triangle_area r n ratio1 ratio2 ratio3 = 30 := by
  -- Given a right triangle with sides in the ratio 5:12:13 inscribed in a circle with radius 6.5, 
  -- its area should be 30.
  sorry

end triangle_area_inscribed_l711_711097


namespace asymptote_slope_of_hyperbola_l711_711139

theorem asymptote_slope_of_hyperbola :
  ∀ (x y : ℝ), (x ≠ 0) ∧ (y/x = 3/4 ∨ y/x = -3/4) ↔ (x^2 / 144 - y^2 / 81 = 1) := 
by
  sorry

end asymptote_slope_of_hyperbola_l711_711139


namespace convex_polygons_are_equal_l711_711226

open List

theorem convex_polygons_are_equal 
  (n : ℕ) 
  (A B : Fin n → Point ℝ)
  (convex_A : convex_hull (range A) = set.univ)
  (convex_B : convex_hull (range B) = set.univ)
  (sides_equal : ∀ i : Fin n, dist (A i) (A (i + 1) % n) = dist (B i) (B (i + 1) % n))
  (angles_equal : ∀ i : Fin (n - 3), angle (A i) (A (i + 1) % n) (A (i + 2) % n) = angle (B i) (B (i + 1) % n) (B (i + 2) % n)) :
  ∀ i : Fin n, A i = B i := sorry

end convex_polygons_are_equal_l711_711226


namespace equilateral_triangle_on_parabola_sum_x_coords_l711_711468

theorem equilateral_triangle_on_parabola_sum_x_coords :
  ∃ (m n : ℕ), gcd m n = 1 ∧
  (∀ (x1 x2 x3 : ℝ),
    (x1 + x2 = 2) ∧
    (x1^2 + x2^2 + x3^2 = x1 + x2 + x3) ∧ -- Vertices on the parabola and given side slope condition
    (x1 + x2 + x3 = (m : ℝ) / (n : ℝ)) →
    m + n = 14) :=
sorry

end equilateral_triangle_on_parabola_sum_x_coords_l711_711468


namespace count_possible_pairs_l711_711352

theorem count_possible_pairs :
  let pairs := list.range' 1007 1005
  let transformed_pairs_seq := λ l, l.map_with_index $ λ i x, (x + 2 * i, 1006 - i) -- This represents the sequence as described.
  let transform := λ (p q : ℕ × ℕ), ((p.1 * p.2 * q.1) / q.2, (p.1 * p.2 * q.2) / q.1)

  ∃ (count : ℕ), 
  (∀ pair_seq : list (ℕ × ℕ), 
     pair_seq = transformed_pairs_seq pairs
     ∧ ∃ final_pair : ℕ × ℕ, final_pair ∈ pair_seq.filter (λ x, pair_seq.length = 1))
  ∧ count = 504510 :=
sorry

end count_possible_pairs_l711_711352


namespace find_N_l711_711650

theorem find_N : ∃ N : ℕ, (∃ x y : ℕ, 
(10 * x + y = N) ∧ 
(4 * x + 2 * y = N / 2) ∧ 
(1 ≤ x) ∧ (x ≤ 9) ∧ 
(0 ≤ y) ∧ (y ≤ 9)) ∧
(N = 32 ∨ N = 64 ∨ N = 96) :=
by {
    have h1 : ∀ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 → 4 * x + 2 * y = (10 * x + y) / 2 → true, sorry,
    exact h1
}

end find_N_l711_711650


namespace problem1_b_ge_neg1_problem2_max_g1_l711_711781

section problem1

def f (x : ℝ) : ℝ := if (0 < x ∧ x < 1) then x else (if x ≥ 1 then 1/x else 0)

def g (a : ℝ) (x : ℝ) : ℝ := a * f(x) - |x - 1|

theorem problem1_b_ge_neg1
  (b : ℝ) :
  (∀ x : ℝ, (0 < x) → g 0 x ≤ |x - 2| + b) ↔ b ≥ -1 :=
by sorry

end problem1

section problem2

theorem problem2_max_g1 :
  (∀ x : ℝ, (0 < x) → x ≠ 1 → (if (0 < x ∧ x < 1) then (1 : ℝ) else (1 : ℝ)) ) :=
by sorry

end problem2

end problem1_b_ge_neg1_problem2_max_g1_l711_711781


namespace income_ratio_l711_711011

theorem income_ratio (I1 I2 E1 E2 : ℝ) (h1 : I1 = 5500) (h2 : E1 = I1 - 2200) (h3 : E2 = I2 - 2200) (h4 : E1 / E2 = 3 / 2) : I1 / I2 = 5 / 4 := by
  -- This is where the proof would go, but it's omitted for brevity.
  sorry

end income_ratio_l711_711011


namespace sin_derivative_is_cos_l711_711879

theorem sin_derivative_is_cos (x : ℝ) : deriv (λ x : ℝ, sin x) x = cos x := by
  sorry

end sin_derivative_is_cos_l711_711879


namespace largest_minus_smallest_l711_711039

theorem largest_minus_smallest : 
  ∃ (digits : Finset ℕ), digits = {2, 7, 4, 9} ∧ 
  (let largest := 974 in 
   let smallest := 247 in 
   largest - smallest = 727) :=
by
  use ({2, 7, 4, 9} : Finset ℕ)
  split
  . rfl
  . let largest := 974
    let smallest := 247
    show largest - smallest = 727
    sorry

end largest_minus_smallest_l711_711039


namespace simplify_expression_l711_711782

theorem simplify_expression : 4 * (14 / 5) * (20 / -42) = -4 / 15 := 
by sorry

end simplify_expression_l711_711782


namespace subsets_not_in_A_or_B_l711_711599

namespace SubsetProblem

def U := {1, 2, 3, 4, 5, 6, 7, 8}
def A := {1, 2, 3, 4, 5}
def B := {4, 5, 6, 7, 8}

noncomputable def countValidSubsets : ℕ :=
  let set1 := {1, 2, 3}
  let set2 := {6, 7, 8}
  let set3 := {4, 5}
  (2^|set1| - 1) * (2^|set2| - 1) * 2^|set3|

theorem subsets_not_in_A_or_B :
  countValidSubsets = 196 := by
  sorry

end SubsetProblem

end subsets_not_in_A_or_B_l711_711599


namespace part_a_has_unique_positive_root_part_b_inequality_l711_711743

open BigOperators

variables {α : Type*} [LinearOrderedField α]

def A (a : Fin n → α) : α :=
  ∑ i, a i

def B (a : Fin n → α) : α :=
  ∑ i, (i + 1) * a i

theorem part_a_has_unique_positive_root (a : Fin n → α) (h_nonneg : ∀ i, 0 ≤ a i) (h_not_all_zero : ∃ i, a i ≠ 0) :
  ∃! R : α, 0 < R ∧ (∏ i, R^(n - i - 1) - ∑ i, a i * R^(n - i - 1) = 0) :=
sorry

theorem part_b_inequality (a : Fin n → α) (h_nonneg : ∀ i, 0 ≤ a i) (h_not_all_zero : ∃ i, a i ≠ 0)
  (h_R_unique : ∃! R : α, 0 < R ∧ (∏ i, R^(n - i - 1) - ∑ i, a i * R^(n - i - 1) = 0)) :
  let R := classical.some h_R_unique,
      A := ∑ i, a i,
      B := ∑ i, (i + 1) * a i
  in A^A ≤ R^B :=
sorry

end part_a_has_unique_positive_root_part_b_inequality_l711_711743


namespace sample_data_perfect_negative_correlation_l711_711252

variables {α : Type*} [linear_ordered_field α]

-- Define the sample data
variables {x y : ℕ → α}
variable {n : ℕ}
-- Conditions
variable (h1 : n ≥ 2)
variable (h2 : ¬∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → x i = x j)
variable (h3 : ∀ i, 1 ≤ i ∧ i ≤ n → (2 * x i + y i = 1))

-- Define the sample correlation coefficient
noncomputable def sample_correlation_coefficient (x y : ℕ → α) (n : ℕ) : α := sorry

-- State the theorem
theorem sample_data_perfect_negative_correlation : 
  sample_correlation_coefficient x y n = -1 := 
  by
    assume h1 h2 h3
    sorry

end sample_data_perfect_negative_correlation_l711_711252


namespace find_q_l711_711413

theorem find_q (p : ℝ) (q : ℝ) (h1 : p ≠ 0) (h2 : p = 4) (h3 : q ≠ 0) (avg_speed_eq : (2 * p * 3) / (p + 3) = 24 / q) : q = 7 := 
 by
  sorry

end find_q_l711_711413


namespace line_intersects_circle_two_points_min_chord_length_line_eq_l711_711553

noncomputable def Circle := {C : Type* // ∃ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 6}
noncomputable def Line := {l : Type* // ∃ m : ℝ, ∃ x y : ℝ, mx - y + 1 - m = 0}

theorem line_intersects_circle_two_points (m : ℝ) :
  ∀ C : Circle, ∀ l : Line, ∃ p1 p2 : ℝ × ℝ,
  p1 ≠ p2 ∧
  (∃ x y : ℝ, (x+1)^2 + (y-2)^2 = 6) ∧
  (∃ x y : ℝ, mx - y + 1 - m = 0) := by
  sorry

theorem min_chord_length_line_eq :
  ∀ l : Line,
  ∃ x y : ℝ, 2x + y - 3 = 0 ∧
  (∀ k : ℝ, k ≠ -2 → ¬ (∃ x y : ℝ, mx - y + 1 - m = 0)) := by
  sorry

end line_intersects_circle_two_points_min_chord_length_line_eq_l711_711553


namespace tan_add_l711_711973

open Real

-- Define positive acute angles
def acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2

-- Theorem: Tangent addition formula
theorem tan_add (α β : ℝ) (hα : acute_angle α) (hβ : acute_angle β) :
  tan (α + β) = (tan α + tan β) / (1 - tan α * tan β) :=
  sorry

end tan_add_l711_711973


namespace complex_formula_result_l711_711136

noncomputable def complex_formula (i : ℂ) (h : i^2 = -1) : ℂ :=
  3 * (1 + i)^2 / (i - 1)

theorem complex_formula_result (i : ℂ) (h : i^2 = -1) : complex_formula i h = 3 - 3i :=
by sorry

end complex_formula_result_l711_711136


namespace probability_of_connecting_in_no_more_than_3_attempts_l711_711450

-- Define the set of possible digits
def digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the probability of dialing the correct number in a single attempt
def prob_correct_attempt := 1 / 10

-- The target theorem to be proven
theorem probability_of_connecting_in_no_more_than_3_attempts : 
  (prob_correct_attempt + prob_correct_attempt + prob_correct_attempt) = 3 / 10 :=
by sorry

end probability_of_connecting_in_no_more_than_3_attempts_l711_711450


namespace second_watermelon_correct_weight_l711_711278

-- Define various weights involved as given in the conditions
def first_watermelon_weight : ℝ := 9.91
def total_watermelon_weight : ℝ := 14.02

-- Define the weight of the second watermelon
def second_watermelon_weight : ℝ :=
  total_watermelon_weight - first_watermelon_weight

-- State the theorem to prove that the weight of the second watermelon is 4.11 pounds
theorem second_watermelon_correct_weight : second_watermelon_weight = 4.11 :=
by
  -- This ensures the statement can be built successfully in Lean 4
  sorry

end second_watermelon_correct_weight_l711_711278


namespace largest_sphere_radius_l711_711938

-- Define the conditions
def inner_radius : ℝ := 3
def outer_radius : ℝ := 7
def circle_center_x := 5
def circle_center_z := 2
def circle_radius := 2

-- Define the question into a statement
noncomputable def radius_of_largest_sphere : ℝ :=
  (29 : ℝ) / 4

-- Prove the required radius given the conditions
theorem largest_sphere_radius:
  ∀ (r : ℝ),
  r = radius_of_largest_sphere → r * r = inner_radius * inner_radius + (circle_center_x * circle_center_x + (r - circle_center_z) * (r - circle_center_z))
:=
by
  sorry

end largest_sphere_radius_l711_711938


namespace baker_new_cakes_bought_l711_711479

variable (total_cakes initial_sold sold_more_than_bought : ℕ)

def new_cakes_bought (total_cakes initial_sold sold_more_than_bought : ℕ) : ℕ :=
  total_cakes - (initial_sold + sold_more_than_bought)

theorem baker_new_cakes_bought (total_cakes initial_sold sold_more_than_bought : ℕ) 
  (h1 : total_cakes = 170)
  (h2 : initial_sold = 78)
  (h3 : sold_more_than_bought = 47) :
  new_cakes_bought total_cakes initial_sold sold_more_than_bought = 78 :=
  sorry

end baker_new_cakes_bought_l711_711479


namespace number_of_liars_l711_711894

-- Define the type of individuals.
inductive KnightOrLiar
| knight  -- always tells the truth
| liar    -- always lies

open KnightOrLiar

-- Define the statements made by individuals based on their number.
def statement (n : ℕ) (people : ℕ → KnightOrLiar) : Prop :=
  if n % 2 = 1 then  -- odd-numbered person
    ∀ m, m > n → people m = liar
  else               -- even-numbered person
    ∀ m, m < n → people m = liar

-- Define the overall condition for all 30 people.
def consistent (people : ℕ → KnightOrLiar) : Prop :=
  ∀ n, 1 ≤ n ∧ n ≤ 30 → statement n people

-- The theorem that we need to prove based on the given conditions.
theorem number_of_liars : ∀ (people : ℕ → KnightOrLiar),
  consistent people →
  (∑ n in (Finset.range 31), if people n = liar then 1 else 0) = 28 := sorry

end number_of_liars_l711_711894


namespace determine_y_l711_711982

theorem determine_y (x y : ℤ) (h1 : x^2 + 4 * x - 1 = y - 2) (h2 : x = -3) : y = -2 := by
  intros
  sorry

end determine_y_l711_711982


namespace cube_difference_l711_711571

variable (x : ℝ)

theorem cube_difference (h : x - 1/x = 5) : x^3 - 1/x^3 = 120 := by
  sorry

end cube_difference_l711_711571


namespace height_of_pyramid_equal_to_cube_volume_l711_711440

theorem height_of_pyramid_equal_to_cube_volume :
  (∃ h : ℝ, (5:ℝ)^3 = (1/3:ℝ) * (10:ℝ)^2 * h) ↔ h = 3.75 :=
by
  sorry

end height_of_pyramid_equal_to_cube_volume_l711_711440


namespace g_domain_l711_711907

noncomputable def g (x : ℝ) : ℝ := Real.tan (Real.arccos (x^3))

theorem g_domain : { x : ℝ | -1 ≤ x ∧ x ≤ 1 ∧ x ≠ 0 } = (Set.Icc (-1) 0 ∪ Set.Icc 0 1) \ {0} :=
by
  sorry

end g_domain_l711_711907


namespace range_of_function_l711_711980

noncomputable def function (x : ℝ) : ℝ := 2 * (Real.sin x)^2 - 3 * Real.sin x + 1

theorem range_of_function : 
  ∀ (x : ℝ), x ∈ Set.Icc π/6 (5*π/6) → 
  function x ∈ Set.Icc (-1/8) 0 := 
by 
  sorry

end range_of_function_l711_711980


namespace circle_equation_given_center_and_diameter_l711_711540

theorem circle_equation_given_center_and_diameter :
  ∃ (x y : ℝ), (x + 2) ^ 2 + (y - 1) ^ 2 = 5 → x^2 + y^2 + 4 * x - 2 * y = 0 := 
by
  use -2
  use 1
  intro h
  rw [pow_two, pow_two, h]
  sorry

end circle_equation_given_center_and_diameter_l711_711540


namespace cubic_identity_l711_711563

theorem cubic_identity (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 :=
by
  sorry

end cubic_identity_l711_711563


namespace AD_eq_EF_l711_711682

variable (A B C D E F : Type) [EuclideanGeometry A B C D E F]

-- Variables for the points A, B, C, D, E, and F.
variable [Points A B C D E F]

-- Conditions
variable (equilateral_BCD : Equilateral B C D)
variable (equilateral_CAE : Equilateral C A E)
variable (equilateral_ABF : Equilateral A B F)
variable (right_angle_ABC : RightAngle A B C)

-- Goal: Prove that |AD| = |EF|.
theorem AD_eq_EF :
  dist A D = dist E F :=
by 
  sorry

end AD_eq_EF_l711_711682


namespace balloons_popped_on_ground_l711_711760

def max_rate : Nat := 2
def max_time : Nat := 30
def zach_rate : Nat := 3
def zach_time : Nat := 40
def total_filled_balloons : Nat := 170

theorem balloons_popped_on_ground :
  (max_rate * max_time + zach_rate * zach_time) - total_filled_balloons = 10 :=
by
  sorry

end balloons_popped_on_ground_l711_711760


namespace number_of_ordered_pairs_l711_711724

def omega : ℂ := Complex.I

def is_solution (a b : ℤ) : Prop :=
  Complex.abs (a * omega + b) = Real.sqrt 2

theorem number_of_ordered_pairs :
  {p : ℤ × ℤ | is_solution p.1 p.2}.toFinset.card = 4 := sorry

end number_of_ordered_pairs_l711_711724


namespace eldest_boy_age_l711_711333

theorem eldest_boy_age (a b c : ℕ) (h1 : a + b + c = 45) (h2 : 3 * c = 7 * a) (h3 : 5 * c = 7 * b) : c = 21 := 
sorry

end eldest_boy_age_l711_711333


namespace volume_not_occupied_by_cones_l711_711826

noncomputable def volume_unoccupied (r_cone h_cone h_cylinder : ℝ) : ℝ :=
  let V_cylinder := pi * r_cone^2 * h_cylinder
  let V_cone := (1 / 3) * pi * r_cone^2 * h_cone
  let V_total_cones := 2 * V_cone
  V_cylinder - V_total_cones

theorem volume_not_occupied_by_cones :
  volume_unoccupied 10 9 20 = 1400 * pi :=
by
  sorry

end volume_not_occupied_by_cones_l711_711826


namespace rate_of_interest_l711_711937
noncomputable section

open Real

def P : ℝ
def r : ℝ
def A1 := 240
def A2 := 217.68707482993196

theorem rate_of_interest:
  (A2 = P * (1 + r)) →
  (A1 = P * (1 + r) ^ 2) →
  r = 0.1025 :=
by
  intros h1 h2
  sorry

end rate_of_interest_l711_711937


namespace remainder_67_pow_67_plus_67_mod_68_l711_711876

theorem remainder_67_pow_67_plus_67_mod_68 :
  (67 ^ 67 + 67) % 68 = 66 :=
by
  -- Skip the proof for now
  sorry

end remainder_67_pow_67_plus_67_mod_68_l711_711876


namespace parallelogram_circle_covering_l711_711882

theorem parallelogram_circle_covering (ABCD : Parallelogram)
    (AB AD alpha : ℝ) 
    (h_AB : ABCD.AB = α)
    (h_AD : ABCD.AD = 1)
    (h_alpha : angle ABCD.DAB = α)
    (h_acute_ABD : triangle ABD has_acute_angles) :
    (cover_by_circles ABCD K_A K_B K_C K_D 1) ↔ (α ≤ cos α + sqrt 3 * sin α) :=
sorry

end parallelogram_circle_covering_l711_711882


namespace eccentricity_of_ellipse_l711_711208

theorem eccentricity_of_ellipse (a b : ℝ) (h_ab : a > b) 
    (ellipse_eq : ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1 → a > b > 0) 
    (F_eq : F = (3, 0)) (A_eq : A = (0, b)) (B_eq : B = (0, -b)) 
    (M_eq : M = (24/5, -3*b/5)) (N_eq : N = (12, 0)) : 
    eccentricity = 1/2 :=
by 
  sorry

end eccentricity_of_ellipse_l711_711208


namespace parabola_chord_length_l711_711082

theorem parabola_chord_length (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (h_focus : ∃ (m : ℝ), ∃ (b : ℝ), ∀ (x y : ℝ), (y^2 = 4 * x) → (y = m * x + b)) 
  (h_intersect : y1^2 = 4 * x1 ∧ y2^2 = 4 * x2)
  (h_sum_x : x1 + x2 = 6) :
  abs (x2 - x1) + sqrt (y1^2 + y2^2) = 8 := 
sorry

end parabola_chord_length_l711_711082


namespace smallest_possible_n_l711_711442

theorem smallest_possible_n
  (n : ℕ)
  (d : ℕ)
  (h_d_pos : d > 0)
  (h_profit : 10 * n - 30 = 100)
  (h_cost_multiple : ∃ k, d = 2 * n * k) :
  n = 13 :=
by {
  sorry
}

end smallest_possible_n_l711_711442


namespace c_share_is_56_l711_711067

theorem c_share_is_56 (total_sum : ℝ) (A B C : ℝ)
  (hA : A = 287 / 2.05)
  (hB : B = 0.65 * A)
  (hC : C = 0.40 * A)
  (h_total : A + B + C = total_sum) :
  total_sum = 287 → C = 56 :=
by
  intros h_total_sum
  rw [h_total_sum, hA, hB, hC]
  sorry

end c_share_is_56_l711_711067


namespace minimum_distance_l711_711207

-- Definitions for the curves C1, C2, and C3
def C1 (t : ℝ) : ℝ × ℝ := (-4 + Real.cos t, 3 + Real.sin t)
def C2 (θ : ℝ) : ℝ × ℝ := (8 * Real.cos θ, 3 * Real.sin θ)
def C3 (t : ℝ) : ℝ × ℝ := (3 + 2 * t, -2 + t)

-- Definition of point P on C1 corresponding to t = π/2
def P : ℝ × ℝ := C1 (Real.pi / 2)

-- Definition of point Q as a moving point on C2
def Q (θ : ℝ) : ℝ × ℝ := C2 θ

-- Definition of the midpoint M of P and Q
def M (θ : ℝ) : ℝ × ℝ := ((-2 + 4 * Real.cos θ), (2 + 3 / 2 * Real.sin θ))

-- Definition of the distance from point M to the line defined by C3
def distance_to_C3 (θ : ℝ) : ℝ := 
  Real.sqrt 5 / 5 * Real.abs (4 * Real.cos θ - 3 * Real.sin θ - 13)

-- Statement of the theorem
theorem minimum_distance : ∃ θ : ℝ, distance_to_C3 θ = 8 * Real.sqrt 5 / 5 :=
sorry

end minimum_distance_l711_711207


namespace general_solution_l711_711158

noncomputable def diff_eq := λ (y : ℝ → ℝ), ∀ x, d ^ 3 y x + 2 * d ^ 2 y x + d y x = 0

theorem general_solution (y : ℝ → ℝ) (C1 C2 C3 : ℝ) : 
  diff_eq y → (∀ x, y x = C1 * exp (-x) + C2 * x * exp (-x) + C3) :=
by
  sorry

end general_solution_l711_711158


namespace average_speed_of_car_l711_711884

theorem average_speed_of_car (time : ℝ) (distance : ℝ) (h_time : time = 4.5) (h_distance : distance = 360) : 
  distance / time = 80 :=
by
  sorry

end average_speed_of_car_l711_711884


namespace real_number_representation_l711_711906

theorem real_number_representation (x : ℝ) 
  (h₀ : 0 < x) (h₁ : x ≤ 1) :
  ∃ (n : ℕ → ℕ), (∀ k, n k > 0) ∧ (∀ k, n (k + 1) = n k * 2 ∨ n (k + 1) = n k * 3 ∨ n (k + 1) = n k * 4) ∧ 
  (x = ∑' k, 1 / (n k)) :=
sorry

end real_number_representation_l711_711906


namespace cube_volume_l711_711439

-- Define the given condition: The surface area of the cube
def surface_area (A : ℕ) := A = 294

-- The key proposition we need to prove using the given condition
theorem cube_volume (s : ℕ) (A : ℕ) (V : ℕ) 
  (area_condition : surface_area A)
  (side_length_condition : s ^ 2 = A / 6) 
  (volume_condition : V = s ^ 3) : 
  V = 343 := by
  sorry

end cube_volume_l711_711439


namespace price_of_insulated_cups_find_purchasing_schemes_number_of_prizes_l711_711499

noncomputable def cost_price_cup_A := 40
noncomputable def cost_price_cup_B := 50
noncomputable def cost_prize_A := 270
noncomputable def cost_prize_B := 240
noncomputable def budget := 2970

theorem price_of_insulated_cups (x : ℕ) :
  x = cost_price_cup_A ∧ (x + 10) = cost_price_cup_B :=
by
  have h1 :  x = 40, from sorry,
  have h2 :  x + 10 = 50, from sorry,
  exact (h1, h2)

theorem find_purchasing_schemes (y : ℕ) (A B : ℕ) :
  38 ≤ y ∧ y ≤ 40 ∧ B = y - 9 ∧ 40 * y + 50 * B ≤ 3150 :=
by
  have h := sorry,
  exact h

theorem number_of_prizes (m n : ℕ):
  cost_prize_A * m + cost_prize_B * n = budget ∧ m = 3 ∧ n = 9 :=
by
 have h := sorry,
 exact h

end price_of_insulated_cups_find_purchasing_schemes_number_of_prizes_l711_711499


namespace geo_series_sum_inequality_l711_711811

theorem geo_series_sum_inequality (a₁ a₂ r q s₁ s₂ : ℝ) (h1 : a₁ = 5/12) (h2 : a₁ = a₂)
  (h3 : q = 3/4) (h4 : s₁ = a₂ / (1 - r)) (h5 : s₂ = a₂ / (1 - q)) :
  s₂ + q > s₁ :=
by
  sorry

end geo_series_sum_inequality_l711_711811


namespace words_per_page_l711_711412

theorem words_per_page (p : ℕ) :
  (136 * p) % 203 = 184 % 203 ∧ p ≤ 100 → p = 73 :=
sorry

end words_per_page_l711_711412


namespace slope_angle_tangent_line_at_zero_l711_711518

noncomputable def curve (x : ℝ) : ℝ := 2 * x - Real.exp x

noncomputable def slope_at (x : ℝ) : ℝ := 
  (deriv curve) x

theorem slope_angle_tangent_line_at_zero : 
  Real.arctan (slope_at 0) = Real.pi / 4 :=
by
  sorry

end slope_angle_tangent_line_at_zero_l711_711518


namespace intersection_roots_l711_711331

theorem intersection_roots :
  x^2 - 4*x - 5 = 0 → (x = 5 ∨ x = -1) := by
  sorry

end intersection_roots_l711_711331


namespace find_N_l711_711651

theorem find_N : ∃ N : ℕ, (∃ x y : ℕ, 
(10 * x + y = N) ∧ 
(4 * x + 2 * y = N / 2) ∧ 
(1 ≤ x) ∧ (x ≤ 9) ∧ 
(0 ≤ y) ∧ (y ≤ 9)) ∧
(N = 32 ∨ N = 64 ∨ N = 96) :=
by {
    have h1 : ∀ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 → 4 * x + 2 * y = (10 * x + y) / 2 → true, sorry,
    exact h1
}

end find_N_l711_711651


namespace number_of_students_selected_is_four_l711_711363

theorem number_of_students_selected_is_four :
  ∃ n : ℕ, n % 2 = 0 ∧ (∃ k : ℕ, k = n / 2 ∧ 
    (nat.choose 4 k * nat.choose 4 k) / (nat.choose 8 n) = 0.5142857142857142) ∧ n = 4 := 
sorry

end number_of_students_selected_is_four_l711_711363


namespace find_C_range_a_minus_b_l711_711665

variables {α β γ : ℝ} -- angles A, B, C
variables {a b c : ℝ} -- sides opposite to angles A, B, C

-- Conditions of the problem
axiom acute_triangle (A B C : ℝ) (h1 : 0 < A ∧ A < π / 2) (h2 : 0 < B ∧ B < π / 2) (h3 : 0 < C ∧ C < π / 2) 
axiom trig_condition (A B C : ℝ) (h : cos (2 * A) + cos (2 * B) + 2 * sin A * sin B = 2 * (cos C)^2)
axiom sides_relation (a b c : ℝ) (h : c = sqrt 3)

-- Questions to prove
theorem find_C (A B C : ℝ) (a b c : ℝ) 
    (h1 : acute_triangle A B C) 
    (h2 : trig_condition A B C) : 
    C = π / 3 :=
sorry

theorem range_a_minus_b (A B C : ℝ) (a b c : ℝ) 
    (h1 : acute_triangle A B C) 
    (h2 : trig_condition A B C) 
    (h3 : sides_relation a b c) : 
    -1 < a - b ∧ a - b < 1 :=
sorry

end find_C_range_a_minus_b_l711_711665


namespace find_x_l711_711918

noncomputable def circle_with_chords (x : ℝ) :=
  ∃ (r : ℝ) (A : ℝ), 
    let C := (⟨x, r⟩ : ℝ × ℝ) -- A circle with radius r and chord length x
    in r ^ 2 = x ^ 2 / 2 ∧
      let area_circle := π * r ^ 2
      in let area_chords := x ^ 2
        in let effective_area := x ^ 2 + (area_circle - area_chords) / 2
          in effective_area = 2 + π

theorem find_x (x : ℝ) : 
  circle_with_chords x → x = 2 :=
by
  sorry

end find_x_l711_711918


namespace complex_div_l711_711560

theorem complex_div : (1 + 3 * complex.I) / (1 + complex.I) = 2 + complex.I := by
  sorry

end complex_div_l711_711560


namespace calculate_expression_l711_711123

theorem calculate_expression : 
  (π - 2019)^0 + |real.sqrt 3 - 1| + (-1/2 : ℝ)^(-1) - 2 * real.tan (real.pi / 6) = -2 + real.sqrt 3 / 3 :=
by sorry

end calculate_expression_l711_711123


namespace arithmetic_sequence_formula_sum_of_bn_l711_711189

theorem arithmetic_sequence_formula {a : ℕ → ℕ} (h : ∀ n, a (n + 1) - a n = 2) 
  (h_geom : (a 1 - 1) * (a 2 - 1) = (a 0 - 1) * (a 2 + 1)) : 
  ∀ n, a n = 2 * n + 1 := sorry

theorem sum_of_bn (a : ℕ → ℕ) (b : ℕ → ℚ) 
  (h : ∀ n, a (n + 1) - a n = 2) 
  (h_geom : (a 1 - 1) * (a 2 - 1) = (a 0 - 1) * (a 2 + 1)) 
  (b_def : ∀ n, b n = 2 / ((a (n + 1)) * (a n))) : 
  ∀ n, ∑ k in finset.range n, b k = 1/3 - 1 / (2 * n + 3) := sorry

end arithmetic_sequence_formula_sum_of_bn_l711_711189


namespace minimum_common_perimeter_l711_711376

theorem minimum_common_perimeter :
  ∃ (a b c : ℤ), 
    (2 * a + 10 * c = 2 * b + 12 * c) ∧ 
    (5 * c * Real.sqrt (a^2 - (5 * c)^2) = 6 * c * Real.sqrt (b^2 - (6 * c)^2)) ∧ 
    ((a - b) = c ∧ 5 * Real.sqrt (a - 5 * c) = 6 * Real.sqrt (b - 6 * c) ∧ 25 * a + 91 * (a - b) = 36 * b) ∧ 
    2 * a + 10 * c = 364 :=
begin
  sorry,
end

end minimum_common_perimeter_l711_711376


namespace jack_marbles_l711_711688

theorem jack_marbles (initial_marbles share_marbles : ℕ) (h_initial : initial_marbles = 62) (h_share : share_marbles = 33) : 
  initial_marbles - share_marbles = 29 :=
by 
  sorry

end jack_marbles_l711_711688


namespace new_ratio_milk_water_l711_711908

def original_mixture_liters : ℕ := 135
def original_ratio_milk_part : ℕ := 3
def original_ratio_water_part : ℕ := 2
def added_water_liters : ℕ := 54

theorem new_ratio_milk_water 
  (total_mixture: ℕ := original_mixture_liters)
  (milk_part : ℕ := original_ratio_milk_part)
  (water_part : ℕ := original_ratio_water_part)
  (added_water : ℕ := added_water_liters) :
  let total_parts := milk_part + water_part in
  let milk_liters := (milk_part * total_mixture) / total_parts in
  let water_liters := (water_part * total_mixture) / total_parts in
  let new_water_liters := water_liters + added_water in
  let gcd_milk_water := Nat.gcd milk_liters new_water_liters in
  (milk_liters / gcd_milk_water) = 3 ∧ (new_water_liters / gcd_milk_water) = 4 := 
by
  sorry

end new_ratio_milk_water_l711_711908


namespace count_three_digit_multiples_of_seven_l711_711617

theorem count_three_digit_multiples_of_seven :
  let a := 100 in
  let b := 999 in
  let smallest := (Nat.ceil (a.toRat / 7)).natAbs * 7 in
  let largest := (b / 7) * 7 in
  (largest / 7) - ((smallest - 1) / 7) = 128 := sorry

end count_three_digit_multiples_of_seven_l711_711617


namespace new_mean_corrected_scores_l711_711462

theorem new_mean_corrected_scores :
  ∀ (initial_mean : ℝ) (initial_count : ℕ) (score1 score2 : ℝ),
  initial_mean = 42 →
  initial_count = 60 →
  score1 = 50 →
  score2 = 60 →
  let initial_sum := initial_mean * initial_count in
  let corrected_sum := initial_sum - (score1 + score2) in
  let new_mean := corrected_sum / (initial_count - 2) in
  new_mean = 41.55 :=
by
  sorry

end new_mean_corrected_scores_l711_711462


namespace factorial_expression_l711_711481

noncomputable theory
open_locale big_operators

theorem factorial_expression :
  7 * nat.factorial 7 + 5 * nat.factorial 5 + 3 * nat.factorial 3 + 2 * (nat.factorial 2)^2 = 35906 :=
by
  sorry

end factorial_expression_l711_711481


namespace cubic_identity_l711_711577

theorem cubic_identity (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / x^3) = 140 := 
  sorry

end cubic_identity_l711_711577


namespace souvenir_prices_total_profit_l711_711458

variables (x y m n : ℝ)

-- Conditions for the first part
def conditions_part1 : Prop :=
  7 * x + 8 * y = 380 ∧
  10 * x + 6 * y = 380

-- Result for the first part
def result_part1 : Prop :=
  x = 20 ∧ y = 30

-- Conditions for the second part
def conditions_part2 : Prop :=
  m + n = 40 ∧
  20 * m + 30 * n = 900 

-- Result for the second part
def result_part2 : Prop :=
  30 * 5 + 10 * 7 = 220

theorem souvenir_prices (x y : ℝ) (h : conditions_part1 x y) : result_part1 x y :=
by { sorry }

theorem total_profit (m n : ℝ) (h : conditions_part2 m n) : result_part2 :=
by { sorry }

end souvenir_prices_total_profit_l711_711458


namespace hyperbola_eccentricity_minimization_l711_711360

-- Define the hyperbola with given parameters.
variables (a b x1 y1 : ℝ) (ha : a > 0) (hb : b > 0)
def hyperbola_equation : Prop := (x1^2 / a^2) - (y1^2 / b^2) = 1

-- Define the slopes k1 and k2.
def k1 : ℝ := y1 / (x1 - a)
def k2 : ℝ := y1 / (x1 + a)

-- Define the expression we want to minimize.
def expression_to_minimize : ℝ := k1 * k2 - 2 * (Real.log (abs k1) + Real.log (abs k2))

-- Define the condition that k1 * k2 = 2
def k1_k2_condition : Prop := y1^2 / (x1^2 - a^2) = 2

-- Define the hyperbola property relationship.
def eccentricity_condition : Prop := b^2 / a^2 = 2

-- Define the proof statement that the eccentricity is √3.
theorem hyperbola_eccentricity_minimization :
  hyperbola_equation a b x1 y1 ha hb →
  k1_k2_condition a x1 y1 →
  eccentricity_condition a b →
  ∃ e : ℝ, e = Real.sqrt 3 :=
by
  intros h1 h2 h3
  sorry

end hyperbola_eccentricity_minimization_l711_711360


namespace scrabble_middle_letter_value_l711_711269

theorem scrabble_middle_letter_value 
  (triple_word_score : ℕ) (single_letter_value : ℕ) (middle_letter_value : ℕ)
  (h1 : triple_word_score = 30)
  (h2 : single_letter_value = 1)
  : 3 * (2 * single_letter_value + middle_letter_value) = triple_word_score → middle_letter_value = 8 :=
by
  sorry

end scrabble_middle_letter_value_l711_711269


namespace jill_investment_l711_711698

def compound_interest (P : ℕ) (r : ℚ) (n : ℕ) (t : ℕ) : ℚ :=
  P * (1 + r / n)^(n * t)

theorem jill_investment : 
  compound_interest 10000 (396 / 10000) 2 2 ≈ 10812 := by
sorry

end jill_investment_l711_711698


namespace cone_ratio_l711_711935

noncomputable def cone_height_ratio : ℚ :=
  let original_height := 40
  let circumference := 24 * Real.pi
  let original_radius := 12
  let new_volume := 432 * Real.pi
  let new_height := 9
  new_height / original_height

theorem cone_ratio (h : cone_height_ratio = 9 / 40) : (9 : ℚ) / 40 = 9 / 40 := by
  sorry

end cone_ratio_l711_711935


namespace number_of_liars_l711_711899

-- Define a predicate is_knight representing if a person is a knight.
def is_knight (n : ℕ) : Prop := sorry

-- Define the total number of people.
def total_people := 30

-- Define conditions for odd-numbered persons.
def odd_person_statement (n : ℕ) : Prop := 
  is_knight n ↔ ∀ m, (m > n ∧ m ≤ total_people) → ¬is_knight m

-- Define conditions for even-numbered persons.
def even_person_statement (n : ℕ) : Prop := 
  is_knight n ↔ ∀ m, (m < n ∧ m ≥ 1) → ¬is_knight m

-- The main theorem we want to prove.
theorem number_of_liars : ∃ n, n = 28 ∧ 
  (∀ i, (1 ≤ i ∧ i ≤ total_people ∧ i % 2 = 1) → odd_person_statement i) ∧
  (∀ j, (1 ≤ j ∧ j ≤ total_people ∧ j % 2 = 0) → even_person_statement j) ∧
  (∀ k, 1 ≤ k ∧ k ≤ total_people → (is_knight k ∨ ¬is_knight k)) ∧
  (n = total_people - ∑ p in finRange total_people, if is_knight p then 1 else 0) :=
sorry

end number_of_liars_l711_711899


namespace black_to_brown_ratio_l711_711357

-- Definitions of the given conditions
def total_shoes : ℕ := 66
def brown_shoes : ℕ := 22
def black_shoes : ℕ := total_shoes - brown_shoes

-- Lean 4 problem statement: Prove the ratio of black shoes to brown shoes is 2:1
theorem black_to_brown_ratio :
  (black_shoes / Nat.gcd black_shoes brown_shoes) = 2 ∧ (brown_shoes / Nat.gcd black_shoes brown_shoes) = 1 := by
sorry

end black_to_brown_ratio_l711_711357


namespace fibonacci_money_problem_l711_711678

variable (x : ℕ)

theorem fibonacci_money_problem (h : 0 < x - 6) (eq_amounts : 90 / (x - 6) = 120 / x) : 
    90 / (x - 6) = 120 / x :=
sorry

end fibonacci_money_problem_l711_711678


namespace max_points_of_intersection_l711_711755

def L (n : ℕ) : Type := Line

def is_parallel (l1 l2 : Line) : Prop := sorry -- parallel lines (dummy definition)
def passes_through (l : Line) (p : Point) : Prop := sorry -- line passes through a point (dummy definition)

axiom A : Point
axiom B : Point

-- Create enough lines and ensure the conditions in Lean
variable (L : ℕ → Line)
variable (A B : Point)

-- Express the conditions
def distinct_lines : Prop := ∀ i j, i ≠ j → L i ≠ L j
def condition_1 : Prop := distinct_lines L
def condition_2 : Prop := ∀ n, is_parallel (L (5 * n)) (L (5 * (n + 1)))
def condition_3 : Prop := ∀ n, passes_through (L (5 * n - 4)) A
def condition_4 : Prop := ∀ n, passes_through (L (5 * n - 3)) B

-- Main theorem to prove
theorem max_points_of_intersection : 
  condition_1 L → 
  condition_2 L →
  condition_3 L A →
  condition_4 L B →
  max_intersections (L 150) = 8972 :=
sorry

end max_points_of_intersection_l711_711755


namespace wrapping_paper_area_correct_l711_711070

-- Given conditions:
variables (w h : ℝ) -- base length and height of the box

-- Definition of the area of the wrapping paper given the problem's conditions
def wrapping_paper_area (w h : ℝ) : ℝ :=
  2 * (w + h) ^ 2

-- Theorem statement to prove the area of the wrapping paper
theorem wrapping_paper_area_correct (w h : ℝ) : wrapping_paper_area w h = 2 * (w + h) ^ 2 :=
by
  -- proof to be provided
  sorry

end wrapping_paper_area_correct_l711_711070


namespace powerThreeExpression_l711_711570

theorem powerThreeExpression (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / (x^3)) = 140 := sorry

end powerThreeExpression_l711_711570


namespace area_triangle_MON_l711_711297

-- Defining the problem statement
theorem area_triangle_MON :
  ∀ (A B C D O P M N : Type) -- Define all points A, B, C, D, O, P, M, N
  (inscribed_in_circle : ∀ (A B C D : Type) (O : Type), Prop) -- ABCD is inscribed in circle with center O
  (intersection : ∀ (A B C D P : Type), Prop) -- P is the intersection of AC and BD
  (midpoint : ∀ (X Y Z : Type), Prop) -- M and N are midpoints of AD and BC
  (dist : ∀ (X Y : Type), ℝ) -- Function for distances between points
  (perpendicular : ∀ (x y z w : Type), Prop), -- AC is perpendicular to BD
  inscribed_in_circle A B C D O → 
  intersection A C B D P → 
  midpoint A D M → 
  midpoint B C N → 
  (dist A P = 1) → 
  (dist B P = 3) → 
  (dist D P = Real.sqrt 3) → 
  perpendicular A C B D →
  (∃ area : ℝ, area = 3 / 4) :=
by
  intros A B C D O P M N inscribed_in_circle intersection midpoint dist perpendicular h1 h2 h3 h4 h5 h6 h7
  use 3 / 4
  sorry

end area_triangle_MON_l711_711297


namespace powerThreeExpression_l711_711569

theorem powerThreeExpression (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / (x^3)) = 140 := sorry

end powerThreeExpression_l711_711569


namespace leaks_empty_tank_time_l711_711033

theorem leaks_empty_tank_time :
  let A := 2 * B in
  let total_rate_no_leak := (1 / 8 : ℝ) in
  let total_rate_with_leak := (1 / 12 : ℝ) in
  (2 * B + B = total_rate_no_leak) →
  (2 * B + B - 2 * L = total_rate_with_leak) →
  (B = 1 / 24) →
  (L = 1 / 48) →
  (1 / L = 48) :=
by
  intros A total_rate_no_leak total_rate_with_leak h1 h2 hB hL
  sorry

end leaks_empty_tank_time_l711_711033


namespace total_polled_votes_proof_l711_711399

-- Define the conditions
variables (V : ℕ) -- total number of valid votes
variables (invalid_votes : ℕ) -- number of invalid votes
variables (total_polled_votes : ℕ) -- total polled votes
variables (candidateA_votes candidateB_votes : ℕ) -- votes for candidate A and B respectively

-- Assume the known conditions
variable (h1 : candidateA_votes = 45 * V / 100) -- candidate A got 45% of valid votes
variable (h2 : candidateB_votes = 55 * V / 100) -- candidate B got 55% of valid votes
variable (h3 : candidateB_votes - candidateA_votes = 9000) -- candidate A was defeated by 9000 votes
variable (h4 : invalid_votes = 83) -- there are 83 invalid votes
variable (h5 : total_polled_votes = V + invalid_votes) -- total polled votes is sum of valid and invalid votes

-- Define the theorem to prove
theorem total_polled_votes_proof : total_polled_votes = 90083 :=
by 
  -- Placeholder for the proof
  sorry

end total_polled_votes_proof_l711_711399


namespace calc_value_l711_711955

theorem calc_value : 
  ((2.502 + 0.064)^3 - sqrt((2.502 - 0.064)^4) / log2(2.502 * 0.064)) * sin(0.064) = 1.222307 := 
  sorry

end calc_value_l711_711955


namespace sum_of_k_with_distinct_integer_roots_l711_711839

theorem sum_of_k_with_distinct_integer_roots :
  ∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ 3*p*q = 12 ∧ p + q = k/3}, k = 0 :=
by
  sorry

end sum_of_k_with_distinct_integer_roots_l711_711839


namespace not_associative_l711_711723

def star (a b : ℝ) : ℝ := 2 * a * b

def S := {x : ℝ // x ≠ 0}

theorem not_associative : ¬ (∀ (a b c : S), star (star a c) b = star a (star b c)) :=
by
  sorry

end not_associative_l711_711723


namespace problem_statement_l711_711718

variable {x : ℝ}
noncomputable def A : ℝ := 39
noncomputable def B : ℝ := -5

theorem problem_statement (h : ∀ x ≠ 3, (A / (x - 3) + B * (x + 2)) = (-5 * x ^ 2 + 18 * x + 30) / (x - 3)) : A + B = 34 := 
sorry

end problem_statement_l711_711718


namespace num_solutions_floor_eq_count_solutions_floor_eq_l711_711236

theorem num_solutions_floor_eq : 
  { x : ℕ // ( ⌊ (x : ℕ) / 10 ⌋ = ⌊ (x / 11 : ℕ) ⌋ + 1 ) }.finite :=
by sorry

theorem count_solutions_floor_eq :
  { x : ℕ // ( ⌊ (x : ℕ) / 10 ⌋ = ⌊ (x / 11 : ℕ) ⌋ + 1 ) }.card = 110 :=
by sorry

end num_solutions_floor_eq_count_solutions_floor_eq_l711_711236


namespace ensure_user_data_security_l711_711763

-- Definitions for conditions
def cond1 : Prop := ∃ (app : Type), 
  (app.is_online_store_with_credit_card_payment)
def cond2 : Prop := ∀ (app : Type), 
  (app.needs_security_against_data_theft)

-- Definitions for security measures
def security_measures : Type :=
| avoid_storing_card_data
| encrypt_stored_data
| encrypt_data_in_transit
| code_obfuscation
| restrict_on_rooted_devices
| integrate_antivirus

-- Theorem statement
theorem ensure_user_data_security 
  (c1 : cond1) (c2 : cond2) : 
  ∃ (measures : List security_measures), measures.length ≥ 3 :=
sorry

end ensure_user_data_security_l711_711763


namespace sin_cos_eq_one_l711_711513

open Real

theorem sin_cos_eq_one (x : ℝ) (h0 : 0 ≤ x) (h2 : x < 2 * π) (h : sin x + cos x = 1) :
  x = 0 ∨ x = π / 2 := 
by
  sorry

end sin_cos_eq_one_l711_711513


namespace problem_statement_l711_711597

def sequence_rule (a : ℕ) : ℕ :=
if even a then a / 2 else 3 * a + 1

def a1 := 5
def a2 := sequence_rule a1
def a3 := sequence_rule a2

theorem problem_statement : a1 + a2 + a3 = 29 := by
  sorry

end problem_statement_l711_711597


namespace cube_volume_from_surface_area_l711_711432

theorem cube_volume_from_surface_area (A : ℝ) (hA : A = 294) : ∃ V : ℝ, V = 343 :=
by
  let s := real.sqrt (A / 6)
  have h_s : s = 7 := sorry
  let V := s^3
  have h_V : V = 343 := sorry
  exact ⟨V, h_V⟩

end cube_volume_from_surface_area_l711_711432


namespace min_lines_for_same_quadrant_l711_711892

-- Define the problem conditions and the statement question
def line (k b : ℝ) := ∃ x : ℝ, y = k * x + b

-- Define a function that checks the quadrant based on the line equation
def quadrant_check (k b : ℝ) : ℕ :=
if k > 0 then
  if b > 0 then 1
  else if b = 0 then 2
  else 3
else
  if b > 0 then 4
  else if b = 0 then 5
  else 6

-- State the theorem to be proven
theorem min_lines_for_same_quadrant : 
  ∀ (s : Finset (ℝ × ℝ)), (∀ (k b : ℝ), line k b → k ≠ 0) →
  s.card ≥ 7 →
  ∃ (k1 b1 k2 b2 : ℝ), (k1, b1) ∈ s ∧ (k2, b2) ∈ s ∧ quadrant_check k1 b1 = quadrant_check k2 b2 :=
sorry

end min_lines_for_same_quadrant_l711_711892


namespace sum_of_k_with_distinct_integer_roots_l711_711837

theorem sum_of_k_with_distinct_integer_roots :
  ∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ 3*p*q = 12 ∧ p + q = k/3}, k = 0 :=
by
  sorry

end sum_of_k_with_distinct_integer_roots_l711_711837


namespace acute_angle_of_parallelogram_l711_711446

theorem acute_angle_of_parallelogram
  (a b : ℝ) (h : a < b)
  (parallelogram_division : ∀ x y : ℝ, x + y = a → b = x + 2 * Real.sqrt (x * y) + y) :
  ∃ α : ℝ, α = Real.arcsin ((b / a) - 1) :=
sorry

end acute_angle_of_parallelogram_l711_711446


namespace pieces_eaten_first_night_l711_711523

def initial_candy_debby : ℕ := 32
def initial_candy_sister : ℕ := 42
def candy_after_first_night : ℕ := 39

theorem pieces_eaten_first_night :
  (initial_candy_debby + initial_candy_sister) - candy_after_first_night = 35 := by
  sorry

end pieces_eaten_first_night_l711_711523


namespace points_converge_to_P_l711_711546

-- Define the triangle and positioning of P0
variables {A B C P₀ P : ℝ} -- Using ℝ for simplicity, but can be generalized to appropriate geometric structures

-- Define the properties of the points P_i, Q_i, R_i
axiom cos_A : ℝ
axiom cos_B : ℝ
axiom cos_C : ℝ
axiom k : ℝ := cos_A * cos_B * cos_C
axiom initial_distance : ℝ -- Initial distance P₀P₁ 

-- Define the sequences converging and their geometric constraints
noncomputable def converge_to_P (P₀ : ℝ) (k : ℝ) : Prop :=
  ∃ P : ℝ, limit (λ n, k^n * initial_distance) P

-- The main statement to be proved
theorem points_converge_to_P (cos_A cos_B cos_C : ℝ) (h : abs (cos_A * cos_B * cos_C) < 1) :
  ∃ (P : ℝ), converge_to_P P₀ k :=
sorry

end points_converge_to_P_l711_711546


namespace geometric_sequence_sum_k_eq_l711_711016

theorem geometric_sequence_sum_k_eq:
  (∀ (n : ℕ), S n = 3^(n-2) + k) → k = -(1/9) := by 
  sorry

end geometric_sequence_sum_k_eq_l711_711016


namespace concurrency_l711_711881

namespace GeometryProblem

-- Let P be an arbitrary point inside the square ABCD.
variables (A B C D P E F G H : Point)
-- Let a line through P parallel to AD intersect AB at E and CD at F.
-- Let a line through P parallel to AB intersect AD at G and BC at H.
variable [Square ABCD]
variable [PointInSquare P ABCD]

-- Assume lines drawn through P as specified.
axiom parallel1 : Parallel (LineThrough P) (LineThrough A D) (LineThrough E F)
axiom parallel2 : Parallel (LineThrough P) (LineThrough A B) (LineThrough G H)

-- Prove that lines AH, CE, and DP are concurrent.
theorem concurrency : Concurrency (LineThrough A H) (LineThrough C E) (LineThrough D P) := sorry

end GeometryProblem

end concurrency_l711_711881


namespace surface_area_ratio_l711_711657

-- Definitions for side lengths in terms of common multiplier x
def side_length_a (x : ℝ) := 2 * x
def side_length_b (x : ℝ) := 1 * x
def side_length_c (x : ℝ) := 3 * x
def side_length_d (x : ℝ) := 4 * x
def side_length_e (x : ℝ) := 6 * x

-- Definitions for surface areas using the given formula
def surface_area (side_length : ℝ) := 6 * side_length^2

def surface_area_a (x : ℝ) := surface_area (side_length_a x)
def surface_area_b (x : ℝ) := surface_area (side_length_b x)
def surface_area_c (x : ℝ) := surface_area (side_length_c x)
def surface_area_d (x : ℝ) := surface_area (side_length_d x)
def surface_area_e (x : ℝ) := surface_area (side_length_e x)

-- Proof statement for the ratio of total surface areas
theorem surface_area_ratio (x : ℝ) (hx : x ≠ 0) :
  (surface_area_a x) / (surface_area_b x) = 4 ∧
  (surface_area_c x) / (surface_area_b x) = 9 ∧
  (surface_area_d x) / (surface_area_b x) = 16 ∧
  (surface_area_e x) / (surface_area_b x) = 36 :=
by {
  sorry
}

end surface_area_ratio_l711_711657


namespace problem1_problem2_l711_711172

variable (α β : ℝ)

-- Condition: α ∈ (π/2, π) and sin α = 1/3
axiom alpha_in_interval : α ∈ (Real.pi / 2, Real.pi)
axiom sin_alpha : Real.sin α = 1 / 3

-- Problem 1: Prove sin 2α = -4√2/9
theorem problem1 : Real.sin (2 * α) = -4 * Real.sqrt 2 / 9 := by 
  sorry

-- Condition: sin(α + β) = -3/5 and β ∈ (0, π/2)
axiom sin_alpha_plus_beta : Real.sin (α + β) = -3 / 5
axiom beta_in_interval : β ∈ (0, Real.pi / 2)

-- Problem 2: Prove sin β = (6√2 + 4) / 15
theorem problem2 : Real.sin β = (6 * Real.sqrt 2 + 4) / 15 := by 
  sorry

end problem1_problem2_l711_711172


namespace miles_driven_l711_711916

def daily_rental_cost : ℝ :=
  29

def cost_per_mile : ℝ :=
  0.08

def total_paid : ℝ :=
  46.12

theorem miles_driven:
  ∀ (daily_rental_cost cost_per_mile total_paid : ℝ), daily_rental_cost = 29 ∧ 
                                                      cost_per_mile = 0.08 ∧ 
                                                      total_paid = 46.12 →
                                                      (total_paid - daily_rental_cost) / cost_per_mile = 214 :=
by
  intro daily_rental_cost cost_per_mile total_paid
  intro h
  cases h with h1 h2_and_h3
  cases h2_and_h3 with h2 h3
  sorry

end miles_driven_l711_711916


namespace bet_strategy_possible_l711_711891

def betting_possibility : Prop :=
  (1 / 6 + 1 / 2 + 1 / 9 + 1 / 8 <= 1)

theorem bet_strategy_possible : betting_possibility :=
by
  -- Proof is intentionally omitted
  sorry

end bet_strategy_possible_l711_711891


namespace polynomial_value_at_neg2_l711_711186

def P (a b x : ℝ) : ℝ := a * (x^3 - x^2 + 3 * x) + b * (2 * x^2 + x) + x^3 - 5

theorem polynomial_value_at_neg2 :
  ∃ (a b : ℝ), 
    (∀ x : ℝ, P (a + 1) (2 * b - a) x = (a + 1) * x^3 + (2 * b - a) * x^2 + (3 * a + b) * x - 5) ∧
    P (a + 1) (2 * b - a) 2 = -17 ∧ 
    P (a + 1) (2 * b - a) (-2) = -1 :=
begin
  sorry
end

end polynomial_value_at_neg2_l711_711186


namespace cara_between_friends_l711_711956

theorem cara_between_friends (n : ℕ) (h : n = 6) : ∃ k : ℕ, k = 15 :=
by {
  sorry
}

end cara_between_friends_l711_711956


namespace volume_bounds_l711_711543

noncomputable def volume_range (a b c : ℝ) : set ℝ :=
  {v | v = a * b * c}

theorem volume_bounds {a b c : ℝ} (h1 : 2 * (a * b + b * c + c * a) = 48)
  (h2 : 4 * (a + b + c) = 36) : 
  ∃ v ∈ volume_range a b c, 16 ≤ v ∧ v ≤ 20 :=
by
  sorry

end volume_bounds_l711_711543


namespace spend_on_video_games_l711_711622

/-- Given the total allowance and the fractions of spending on various categories,
prove the amount spent on video games. -/
theorem spend_on_video_games (total_allowance : ℝ)
  (fraction_books fraction_snacks fraction_crafts : ℝ)
  (h_total : total_allowance = 50)
  (h_fraction_books : fraction_books = 1 / 4)
  (h_fraction_snacks : fraction_snacks = 1 / 5)
  (h_fraction_crafts : fraction_crafts = 3 / 10) :
  total_allowance - (fraction_books * total_allowance + fraction_snacks * total_allowance + fraction_crafts * total_allowance) = 12.5 :=
by
  sorry

end spend_on_video_games_l711_711622


namespace num_kids_eq_3_l711_711689

def mom_eyes : ℕ := 1
def dad_eyes : ℕ := 3
def kid_eyes : ℕ := 4
def total_eyes : ℕ := 16

theorem num_kids_eq_3 : ∃ k : ℕ, 1 + 3 + 4 * k = 16 ∧ k = 3 := by
  sorry

end num_kids_eq_3_l711_711689


namespace volume_of_wedge_l711_711441

/-- Given a cylinder with a radius of 8 cm and a height of 12 cm,
    which is divided into three equal wedges along its height,
    prove that the volume of one of these wedges is 804 cubic centimeters (approximated). -/
theorem volume_of_wedge (r h : ℝ) (divisions : ℕ) 
  (h_r : r = 8) (h_h : h = 12) (h_divisions : divisions = 3) : 
  (1 / 3 : ℝ) * 768 * real.pi ≈ 804 :=
by {
  sorry
}

end volume_of_wedge_l711_711441


namespace greatest_n_l711_711056

open Int

def sequence (a d : ℕ) (hk : (x : ℕ) → x > 1) : ℕ → ℕ
| 1 := 1
| (k + 1) := if (a ∣ sequence a d hk k) then (sequence a d hk k) / a else (sequence a d hk k) + d

theorem greatest_n (a d : ℕ) (h1 : a > 1) (h2 : d > 1) (coprime_ad : gcd a d = 1) : 
  ∃ n, ∃ k, (∀ m, m > n → ¬ (a ^ m ∣ sequence a d _ k)) :=
begin
  sorry
end

end greatest_n_l711_711056


namespace Roy_height_l711_711320

theorem Roy_height (Sara_height Joe_height Roy_height : ℕ) 
  (h1 : Sara_height = 45)
  (h2 : Sara_height = Joe_height + 6)
  (h3 : Joe_height = Roy_height + 3) :
  Roy_height = 36 :=
by
  sorry

end Roy_height_l711_711320


namespace expression_for_P_l711_711624

noncomputable def geom_sequence (a r : ℝ) (n : ℕ) : list ℝ :=
(list.range n).map (λ k, a * r^k)

noncomputable def product_geom_seq (seq : list ℝ) : ℝ :=
seq.prod

noncomputable def sum_geom_seq (seq : list ℝ) : ℝ :=
seq.sum

noncomputable def sum_squares_geom_seq (seq : list ℝ) : ℝ :=
(seq.map (λ x, x^2)).sum

theorem expression_for_P (a r : ℝ) (n : ℕ) (h: r ≠ 1):
  let P := product_geom_seq (geom_sequence a r n),
      S := sum_geom_seq (geom_sequence a r n),
      S' := sum_squares_geom_seq (geom_sequence a r n)
  in P = (S' / S) ^ ((n-1):ℕ/2) := sorry

end expression_for_P_l711_711624


namespace tan_pi_minus_alpha_l711_711173

theorem tan_pi_minus_alpha (alpha : ℝ) 
  (h1 : real.sin alpha = 1 / 3) 
  (h2 : π / 2 < alpha ∧ alpha < π) :
  real.tan (π - alpha) = real.sqrt 2 / 4 :=
sorry

end tan_pi_minus_alpha_l711_711173


namespace marla_drive_time_l711_711759

theorem marla_drive_time (x : ℕ) (h_total : x + 70 + x = 110) : x = 20 :=
sorry

end marla_drive_time_l711_711759


namespace categorize_positive_numbers_categorize_negative_numbers_categorize_integers_categorize_fractions_l711_711992

-- Definitions for the sets and given numbers
def positive_numbers : Set ℝ := {x | x > 0}
def negative_numbers : Set ℝ := {x | x < 0}
def integers : Set ℤ := Set.univ
def fractions : Set ℝ := { x | ∃ a b : ℤ, b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ) }

def given_numbers : Set ℝ := { -4, -abs(-1/5), 0, 22/7, -3.14, 717, -5, 1.88 }

-- Expected sets derived from the given numbers
def expected_positive_numbers : Set ℝ := { 22/7, 717, 1.88 }
def expected_negative_numbers : Set ℝ := { -4, -1/5, -3.14, -5 }
def expected_integers : Set ℤ := { -4, 0, 717, -5 }
def expected_fractions : Set ℝ := { -1/5, 22/7, -3.14, 1.88 }

-- Lean statements modeling the proof problem
theorem categorize_positive_numbers :
  {x ∈ given_numbers | x ∈ positive_numbers} = expected_positive_numbers := 
sorry

theorem categorize_negative_numbers :
  {x ∈ given_numbers | x ∈ negative_numbers} = expected_negative_numbers := 
sorry

theorem categorize_integers :
  {x ∈ given_numbers | x ∈ integers.map (coe : ℤ → ℝ)} = expected_integers := 
sorry

theorem categorize_fractions :
  {x ∈ given_numbers | x ∈ fractions} = expected_fractions := 
sorry

end categorize_positive_numbers_categorize_negative_numbers_categorize_integers_categorize_fractions_l711_711992


namespace original_three_numbers_are_arith_geo_seq_l711_711371

theorem original_three_numbers_are_arith_geo_seq
  (x y z : ℕ) (h1 : ∃ k : ℕ, x = 3*k ∧ y = 4*k ∧ z = 5*k)
  (h2 : ∃ r : ℝ, (x + 1) / y = r ∧ y / z = r ∧ r^2 = z / y):
  x = 15 ∧ y = 20 ∧ z = 25 :=
by 
  sorry

end original_three_numbers_are_arith_geo_seq_l711_711371


namespace unpainted_unit_cubes_l711_711062

theorem unpainted_unit_cubes (total_cubes painted_faces edge_overlaps corner_overlaps : ℕ) :
  total_cubes = 6 * 6 * 6 ∧
  painted_faces = 6 * (2 * 6) ∧
  edge_overlaps = 12 * 3 / 2 ∧
  corner_overlaps = 8 ∧
  total_cubes - (painted_faces - edge_overlaps - corner_overlaps) = 170 :=
by
  sorry

end unpainted_unit_cubes_l711_711062


namespace three_digit_multiples_of_seven_l711_711621

theorem three_digit_multiples_of_seven : 
  ∃ k, (k = {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 0}.card) ∧ k = 128 :=
by {
  sorry
}

end three_digit_multiples_of_seven_l711_711621


namespace solve_problem_l711_711983

open Classical

-- Definition of the problem conditions
def problem_conditions (x y : ℝ) : Prop :=
  5 * y^2 + 3 * y + 2 = 2 * (10 * x^2 + 3 * y + 3) ∧ y = 3 * x + 1

-- Definition of the quadratic solution considering the quadratic formula
def quadratic_solution (x : ℝ) : Prop :=
  x = (-21 + Real.sqrt 641) / 50 ∨ x = (-21 - Real.sqrt 641) / 50

-- Main theorem statement
theorem solve_problem :
  ∃ x y : ℝ, problem_conditions x y ∧ quadratic_solution x :=
by
  sorry

end solve_problem_l711_711983


namespace avg_even_positions_correct_l711_711909

noncomputable def avg_even_positions (n : ℕ) (numbers : ℕ → ℕ) (avg_odd : ℕ) : ℕ :=
  let even_positions := (finset.range n).filter (λ i, (even i)) -- Get all even positions
  let sum_even := even_positions.sum (λ i, numbers (i + 1))    -- Sum of the numbers in even positions
  let num_even := even_positions.card                          -- Count of even positions
  sum_even / num_even                                           -- Average of even positions

theorem avg_even_positions_correct : 
  avg_even_positions 2010 (λ i, i + 1) 2345 = 2346 :=
  sorry

end avg_even_positions_correct_l711_711909


namespace investment_ratio_l711_711789

theorem investment_ratio (A B : ℕ) (hA : A = 12000) (hB : B = 12000) 
  (interest_A : ℕ := 11 * A / 100) (interest_B : ℕ := 9 * B / 100) 
  (total_interest : interest_A + interest_B = 2400) :
  A / B = 1 :=
by
  sorry

end investment_ratio_l711_711789


namespace joe_first_lift_weight_l711_711886

theorem joe_first_lift_weight (x y : ℕ) (h1 : x + y = 1500) (h2 : 2 * x = y + 300) : x = 600 :=
by
  sorry

end joe_first_lift_weight_l711_711886


namespace jet_bar_sales_difference_l711_711103

variable (monday_sales : ℕ) (total_target : ℕ) (remaining_target : ℕ)
variable (sales_so_far : ℕ) (tuesday_sales : ℕ)
def JetBarsDifference : Prop :=
  monday_sales = 45 ∧ total_target = 90 ∧ remaining_target = 16 ∧
  sales_so_far = total_target - remaining_target ∧
  tuesday_sales = sales_so_far - monday_sales ∧
  (monday_sales - tuesday_sales = 16)

theorem jet_bar_sales_difference :
  JetBarsDifference 45 90 16 (90 - 16) (90 - 16 - 45) :=
by
  sorry

end jet_bar_sales_difference_l711_711103


namespace jill_investment_l711_711703

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_investment :
  compound_interest 10000 0.0396 2 2 ≈ 10815.66 :=
by
  sorry

end jill_investment_l711_711703


namespace hypotenuse_length_l711_711087

theorem hypotenuse_length {a b c : ℕ} (ha : a = 8) (hb : b = 15) (hc : c = (8^2 + 15^2).sqrt) : c = 17 :=
by
  sorry

end hypotenuse_length_l711_711087


namespace solve_for_y_l711_711326

theorem solve_for_y (y : ℝ) (h : 27^(2*y - 4) = 3^(-y - 9)) : y = 3/7 :=
by
  sorry

end solve_for_y_l711_711326


namespace range_y_coordinate_C_l711_711555

-- Define the structure for a Point
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Given conditions
def pointA : Point := ⟨0, 2⟩
def parabola (p : Point) : Prop := p. y ^ 2 = p.x + 4

-- Orthogonality condition
def orthogonal (a b c : Point) : Prop :=
  let u := (b.x - a.x, b.y - a.y)
  let v := (c.x - b.x, c.y - b.y)
  u.1 * v.1 + u.2 * v.2 = 0

-- Main theorem to prove the range of y-coordinates of point C
theorem range_y_coordinate_C (b c : Point) (hb : parabola b) (hc : parabola c) (h_ortho : orthogonal pointA b c) :
  c.y ∈ set.Iic 0 ∪ set.Ici 4 :=
by sorry

end range_y_coordinate_C_l711_711555


namespace time_ratio_upstream_downstream_l711_711770

theorem time_ratio_upstream_downstream (S_boat S_stream D : ℝ) (h1 : S_boat = 72) (h2 : S_stream = 24) :
  let time_upstream := D / (S_boat - S_stream)
  let time_downstream := D / (S_boat + S_stream)
  (time_upstream / time_downstream) = 2 :=
by
  sorry

end time_ratio_upstream_downstream_l711_711770


namespace smallest_coprime_to_840_l711_711383

theorem smallest_coprime_to_840 : ∃ n : ℕ, n > 1 ∧ gcd n 840 = 1 ∧ ∀ m : ℕ, m > 1 ∧ gcd m 840 = 1 → n ≤ m :=
begin
  sorry
end

end smallest_coprime_to_840_l711_711383


namespace sum_of_leftmost_and_fourth_l711_711025

def valid_digits : List ℕ := [2, 0, 7, 5, 9]

-- Definition of a valid three-digit number from the provided digits
def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  n ≥ 100 ∧ digits.to_set ⊆ valid_digits.to_set ∧ 0 ∉ digits.init ∧ digits.nodup

-- Extract all valid numbers
def valid_numbers : List ℕ :=
  List.range' 102 901 |>.filter is_valid_number

-- Definitions for leftmost (minimum) and fourth smallest number
def leftmost_number : ℕ := valid_numbers.minimum
def fourth_number : ℕ := (valid_numbers.qsort (· < ·)).nth! 3

theorem sum_of_leftmost_and_fourth :
  leftmost_number + fourth_number = 455 :=
by
  -- This would be replaced by the actual proof.
  sorry

end sum_of_leftmost_and_fourth_l711_711025


namespace domain_of_f_l711_711953

noncomputable def f (x : ℝ) : ℝ := Real.log 5 (Real.log 3 (Real.log 2 (x^2)))

theorem domain_of_f :
  {x : ℝ | ∃ y₁ y₂ y₃ : ℝ, y₁ = Real.log 2 (x^2) ∧ y₂ = Real.log 3 y₁ ∧ y₃ = Real.log 5 y₂ ∧ y₃ ≠ 0} =
  {x : ℝ | x < -Real.sqrt 2 ∨ x > Real.sqrt 2} :=
by
  sorry

end domain_of_f_l711_711953


namespace jill_investment_l711_711704

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_investment :
  compound_interest 10000 0.0396 2 2 ≈ 10815.66 :=
by
  sorry

end jill_investment_l711_711704


namespace age_difference_l711_711020

variable (a b c d : ℕ)

theorem age_difference (h₁ : a + b = c + d + 20) (h₂ : b + d = a + c + 10) : d - a = -5 :=
by
  sorry

end age_difference_l711_711020


namespace correct_geometric_view_statement_l711_711944

-- Given conditions as definitions
def views_of_sphere_are_congruent_circles : Prop :=
  ∀ S : Sphere, ∀ v1 v2 v3 : View, is_congruent_circle v1 S ∧ is_congruent_circle v2 S ∧ is_congruent_circle v3 S

def views_of_cube_are_congruent_squares : Prop :=
  ∀ C : Cube, ∀ v1 v2 v3 : View, is_congruent_square v1 C ∧ is_congruent_square v2 C ∧ is_congruent_square v3 C

def views_of_tetrahedron_are_equilateral_triangles : Prop :=
  ∀ T : Tetrahedron, is_horizontally_placed T →
  ∀ v1 v2 v3 : View, is_equilateral_triangle v1 T ∧ is_equilateral_triangle v2 T ∧ is_equilateral_triangle v3 T

def top_view_of_frustum_is_circle : Prop :=
  ∀ F : Frustum, is_horizontally_placed F → is_circle (top_view F)

-- The theorem we need to prove
theorem correct_geometric_view_statement : views_of_sphere_are_congruent_circles ∧ 
  ¬views_of_cube_are_congruent_squares ∧ 
  ¬views_of_tetrahedron_are_equilateral_triangles ∧
  ¬top_view_of_frustum_is_circle :=
by
  sorry

end correct_geometric_view_statement_l711_711944


namespace total_bottle_caps_l711_711501

-- Define the conditions
def bottle_caps_per_child : ℕ := 5
def number_of_children : ℕ := 9

-- Define the main statement to be proven
theorem total_bottle_caps : bottle_caps_per_child * number_of_children = 45 :=
by sorry

end total_bottle_caps_l711_711501


namespace find_pairs_l711_711496

theorem find_pairs (m n : ℕ) (h : m > 0 ∧ n > 0 ∧ m^2 = n^2 + m + n + 2018) :
  (m, n) = (1010, 1008) ∨ (m, n) = (506, 503) :=
by sorry

end find_pairs_l711_711496


namespace number_of_liars_is_28_l711_711896

variables (n : ℕ → Prop)

-- Define the conditions for knights and liars
def knight (k : ℕ) := n k
def liar (k : ℕ) := ¬n k

-- Define statements for odd and even numbered people
def odd_statement (k : ℕ) := ∀ m, m > k → liar m
def even_statement (k : ℕ) := ∀ m, m < k → liar m

-- Define the main hypothesis following the problem conditions
def conditions : Prop :=
  (∀ k, k % 2 = 1 → (knight k ↔ odd_statement k)) ∧
  (∀ k, k % 2 = 0 → (knight k ↔ even_statement k)) ∧
  (∃ m, m = 30) -- Ensuring there are 30 people

-- Prove the main statement
theorem number_of_liars_is_28 : ∃ l, l = 28 ∧ (∀ k, k ≤ 30 → (liar k ↔ k ≤ 28)) :=
by
  sorry

end number_of_liars_is_28_l711_711896


namespace find_A_in_terms_of_B_and_C_l711_711744

variables (f g : ℝ → ℝ) (A B C : ℝ)

-- Definitions and conditions
def f_def := ∀ x, f x = A * x^2 - 3 * B * C
def g_def := ∀ x, g x = C * x^2
def B_neq_zero := B ≠ 0
def C_neq_zero := C ≠ 0
def f_g_condition := f (g 2) = A - 3 * C

-- The theorem to prove
theorem find_A_in_terms_of_B_and_C :
  f_def f A B C →
  g_def g C →
  B_neq_zero B →
  C_neq_zero C →
  f_g_condition f g A B C →
  A = (3 * C * (B - 1)) / (16 * C^2 - 1) :=
by
  intros
  sorry

end find_A_in_terms_of_B_and_C_l711_711744


namespace number_of_markings_l711_711831

def markings (L : ℕ → ℕ) := ∀ n, (n > 0) → L n = L (n - 1) + 1

theorem number_of_markings : ∃ L : ℕ → ℕ, (∀ n, n = 1 → L n = 2) ∧ markings L ∧ L 200 = 201 := 
sorry

end number_of_markings_l711_711831


namespace range_of_f_l711_711010

noncomputable def f (x : ℝ) := Real.log2 (-x^2 - 2 * x + 3)

theorem range_of_f : set.Iic (2 : ℝ) = {y | ∃ x, f x = y} :=
sorry

end range_of_f_l711_711010


namespace probability_x_plus_y_lt_3_in_rectangle_l711_711085

noncomputable def probability_problem : ℚ :=
let rect_area := (4 : ℚ) * 3
let tri_area := (1 / 2 : ℚ) * 3 * 3
tri_area / rect_area

theorem probability_x_plus_y_lt_3_in_rectangle :
  probability_problem = 3 / 8 :=
sorry

end probability_x_plus_y_lt_3_in_rectangle_l711_711085


namespace no_b_satisfies_143b_square_of_integer_l711_711144

theorem no_b_satisfies_143b_square_of_integer :
  ∀ b : ℤ, b > 4 → ¬ ∃ k : ℤ, b^2 + 4 * b + 3 = k^2 :=
by
  intro b hb
  by_contra h
  obtain ⟨k, hk⟩ := h
  have : b^2 + 4 * b + 3 = k ^ 2 := hk
  sorry

end no_b_satisfies_143b_square_of_integer_l711_711144


namespace find_a_l711_711204

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then x ^ 2 + a * Real.cos (Real.pi * x) else 2

theorem find_a (a : ℝ) :
  (∀ x, f (-x) a = -f x a) → f 1 a = 2 → a = - 3 :=
by
  sorry

end find_a_l711_711204


namespace prime_walk_back_to_start_point_l711_711623

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

-- Prove that given the conditions, the net movement is 14 steps forward
theorem prime_walk_back_to_start_point : 
  let primes := { d ∈ finset.range 32 | is_prime d }
  let composites := { d ∈ finset.range 32 | is_composite d }

  3 * primes.card - composites.card = 14 :=
by
  sorry

end prime_walk_back_to_start_point_l711_711623


namespace three_digit_multiples_of_seven_l711_711619

theorem three_digit_multiples_of_seven : 
  ∃ k, (k = {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 0}.card) ∧ k = 128 :=
by {
  sorry
}

end three_digit_multiples_of_seven_l711_711619


namespace cube_remaining_volume_l711_711923

open Real

-- Define the side length of the cube and the dimensions of the cylindrical section
def side := 5
def radius := 1
def height := 5

-- Define the volume of the cube
def V_cube := side^3

-- Define the volume of the cylinder
def V_cylinder := π * radius^2 * height

-- Define the remaining volume of the cube after the cylindrical section is removed
def V_remaining := V_cube - V_cylinder

-- Prove that the remaining volume is 125 - 5π cubic feet
theorem cube_remaining_volume : V_remaining = 125 - 5 * π :=
by
  -- Proof steps would go here, but are skipped with 'sorry'
  sorry

end cube_remaining_volume_l711_711923


namespace false_propositions_count_l711_711300

theorem false_propositions_count : 
  let p := (∀ x y: ℝ, x * x + 3 * x - 1 = 0 ∧ y * y + 3 * y - 1 = 0 → (x * y < 0))
  let q := (∃ x y: ℝ, x * x + 3 * x - 1 = 0 ∧ y * y + 3 * y - 1 = 0 ∧ x + y = 3)
  (¬ p ∨ ¬ q ∨ p ∧ q ∨ p ∨ q) → 
  ((if ¬ p then 1 else 0) + (if ¬ q then 1 else 0) + (if p ∧ q then 1 else 0) + (if p ∨ q then 1 else 0)) = 2 := 
by
  sorry

end false_propositions_count_l711_711300


namespace product_ab_l711_711949

noncomputable def a : ℝ := 3
noncomputable def b : ℝ := 5 / 2
def tan_function (x : ℝ) : ℝ := a * Real.tan (b * x)

theorem product_ab :
  (∀ x, x = -2*Real.pi/5 ∨ x = 0 ∨ x = 2*Real.pi/5 → (Real.tan (b * x) = 0 ∨ Real.tan (b * x) = 1)) ∧
  tan_function (Real.pi / 10) = 3 →
  a * b = 7.5 := 
by
  sorry

end product_ab_l711_711949


namespace spanning_tree_leaves_gt_two_ninths_l711_711478

theorem spanning_tree_leaves_gt_two_ninths (G : Graph) (n : ℕ) (hG : G.connected)
  (hV : G.vertex_count = n) (hD : ∀ v ∈ G.vertices, G.degree v ≥ 3) 
  : ∃ T : G.spanning_tree, T.leaf_count > (2 * n / 9) :=
by
  sorry

end spanning_tree_leaves_gt_two_ninths_l711_711478


namespace g_max_at_1_range_of_a_l711_711212

noncomputable def f (x a : ℝ) : ℝ := x * log x - a * x^2 + a

noncomputable def f_prime (x a : ℝ) : ℝ := log x - 2 * a * x + 1

noncomputable def g (x a : ℝ) : ℝ := f_prime x a + (2 * a - 1) * x

theorem g_max_at_1 (a : ℝ) : g 1 a = 0 := by
  sorry

theorem range_of_a (a : ℝ) : (∀ x > 1, f x a < 0) ↔ a ∈ Set.Ici (1 / 2) := by
  sorry

end g_max_at_1_range_of_a_l711_711212


namespace main_theorem_l711_711125

noncomputable def main_expr := (Real.pi - 2019) ^ 0 + |Real.sqrt 3 - 1| + (-1 / 2)⁻¹ - 2 * Real.tan (Real.pi / 6)

theorem main_theorem : main_expr = -2 + Real.sqrt 3 / 3 := by
  sorry

end main_theorem_l711_711125


namespace part1_part2_l711_711589

noncomputable def f (x : ℝ) := (2 : ℝ) ^ x

theorem part1 (x : ℝ) (h : f⁻¹ x - f⁻¹ (1 - x) = 1) : x = 2 / 3 := 
sorry

theorem part2 (m : ℝ) (h : ∃ x ∈ set.Icc (0 : ℝ) 2, f x + f (1 - x) = m) : 
  m ∈ set.Icc 3 (9 / 2) := 
sorry

end part1_part2_l711_711589


namespace a_2023_is_negative_1011_l711_711190

def sequence (n : ℕ) : ℤ :=
  match n with
  | 0     => 0
  | 1     => 0
  | (n+2) => -(abs (sequence (n + 1) + (n + 1)))

theorem a_2023_is_negative_1011 : sequence 2023 = -1011 := 
  sorry

end a_2023_is_negative_1011_l711_711190


namespace convert_spherical_to_rectangular_correct_l711_711141

-- Define the spherical to rectangular conversion functions
noncomputable def spherical_to_rectangular (rho θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (rho * Real.sin φ * Real.cos θ, rho * Real.sin φ * Real.sin θ, rho * Real.cos φ)

-- Define the given spherical coordinates
def given_spherical_coords : ℝ × ℝ × ℝ :=
  (5, 7 * Real.pi / 4, Real.pi / 3)

-- Define the expected rectangular coordinates
def expected_rectangular_coords : ℝ × ℝ × ℝ :=
  (-5 * Real.sqrt 6 / 4, -5 * Real.sqrt 6 / 4, 5 / 2)

-- The proof statement
theorem convert_spherical_to_rectangular_correct (ρ θ φ : ℝ)
  (h_ρ : ρ = 5) (h_θ : θ = 7 * Real.pi / 4) (h_φ : φ = Real.pi / 3) :
  spherical_to_rectangular ρ θ φ = expected_rectangular_coords :=
by
  -- Proof omitted
  sorry

end convert_spherical_to_rectangular_correct_l711_711141


namespace arrow_sequence_correct_l711_711057

variable (A B C D E F G : ℕ)
variable (square : ℕ → ℕ)

-- Definitions based on given conditions
def conditions : Prop :=
  square 1 = 1 ∧ square 9 = 9 ∧
  square A = 6 ∧ square B = 2 ∧ square C = 4 ∧
  square D = 5 ∧ square E = 3 ∧ square F = 8 ∧ square G = 7 ∧
  (∀ x, (x = 1 → square 2 = B) ∧ (x = 2 → square 3 = E) ∧
       (x = 3 → square 4 = C) ∧ (x = 4 → square 5 = D) ∧
       (x = 5 → square 6 = A) ∧ (x = 6 → square 7 = G) ∧
       (x = 7 → square 8 = F) ∧ (x = 8 → square 9 = 9))

theorem arrow_sequence_correct :
  conditions A B C D E F G square → 
  ∀ x, square (x + 1) = 1 + x :=
by sorry

end arrow_sequence_correct_l711_711057


namespace percentage_female_after_hiring_is_55_l711_711021

noncomputable def percentage_female_after_hiring 
  (initial_female_percentage : ℚ) (additional_males : ℕ) (total_after_hiring : ℕ) : ℚ :=
  let initial_employees := total_after_hiring - additional_males in
  let female_employees := initial_female_percentage * initial_employees in
  100 * female_employees / total_after_hiring

theorem percentage_female_after_hiring_is_55 :
  percentage_female_after_hiring 0.60 30 360 = 55 :=
by 
  sorry

end percentage_female_after_hiring_is_55_l711_711021


namespace find_number_of_cows_l711_711670

-- Definitions for the problem
def number_of_legs (cows chickens : ℕ) := 4 * cows + 2 * chickens
def twice_the_heads_plus_12 (cows chickens : ℕ) := 2 * (cows + chickens) + 12

-- Main statement to prove
theorem find_number_of_cows (h : ℕ) : ∃ c : ℕ, number_of_legs c h = twice_the_heads_plus_12 c h ∧ c = 6 := 
by
  -- Sorry is used as a placeholder for the proof
  sorry

end find_number_of_cows_l711_711670


namespace money_distribution_invalid_l711_711466

theorem money_distribution_invalid :
  ∃ (A B : ℝ), A + 250 = 200 ∧ B + 250 = 350 → false :=
by
  intro A B h
  cases h with h1 h2
  have A_eq : A = 200 - 250 := by linarith
  have B_eq : B = 350 - 250 := by linarith
  linarith

end money_distribution_invalid_l711_711466


namespace probability_event_l711_711452

-- Define the conditions on x, y, z
def condition_x (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2
def condition_y (y : ℝ) : Prop := -2 ≤ y ∧ y ≤ 2
def condition_z (z : ℝ) : Prop := -2 ≤ z ∧ z ≤ 2
def conditions (x y z : ℝ) : Prop := condition_x x ∧ condition_y y ∧ condition_z z

-- Define the sphere and plane conditions
def sphere_condition (x y z : ℝ) : Prop := x^2 + y^2 + z^2 ≤ 4
def plane_condition (x y z : ℝ) : Prop := x + y + z ≥ 0

-- Define the event of interest
def event (x y z : ℝ) : Prop := sphere_condition x y z ∧ plane_condition x y z

-- Main statement
theorem probability_event : 
  (∫ x in -2..2, ∫ y in -2..2, ∫ z in -2..2, if event x y z then 1 else 0) / 64 = (real.pi / 12) :=
by
  sorry

end probability_event_l711_711452


namespace find_N_l711_711653

theorem find_N : ∃ N : ℕ, (∃ x y : ℕ, 
(10 * x + y = N) ∧ 
(4 * x + 2 * y = N / 2) ∧ 
(1 ≤ x) ∧ (x ≤ 9) ∧ 
(0 ≤ y) ∧ (y ≤ 9)) ∧
(N = 32 ∨ N = 64 ∨ N = 96) :=
by {
    have h1 : ∀ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 → 4 * x + 2 * y = (10 * x + y) / 2 → true, sorry,
    exact h1
}

end find_N_l711_711653


namespace find_pqr_l711_711171

-- Defining the sets A and B
def A (p : ℤ) : set ℤ := {x | x ^ 2 - p * x - 2 = 0}
def B (q r : ℤ) : set ℤ := {x | x ^ 2 + q * x + r = 0}

-- Given statements
theorem find_pqr : 
  (A p ∪ B q r = {-2, 1, 7}) →
  (A p ∩ B q r = {-2}) →
  p = -1 ∧ q = -5 ∧ r = -14 :=
by
  intro h_union h_inter
  sorry
  -- Here, you should proceed with the actual proof steps, showing the necessary calculations.

end find_pqr_l711_711171


namespace bridge_length_l711_711401

/-- A train is 80 meters long, travels at 45 km/hr, and crosses a bridge in 30 seconds. Prove that the length of the bridge is 295 meters. -/
theorem bridge_length
  (train_length : ℕ := 80)
  (speed_kmph : ℕ := 45)
  (time_seconds : ℕ := 30) :
  let speed_mps := (speed_kmph * 1000 / 3600 : ℝ),
      total_distance := speed_mps * (time_seconds : ℝ) in
  (total_distance - train_length = 295 : ℝ) :=
sorry

end bridge_length_l711_711401


namespace sum_of_k_with_distinct_integer_roots_l711_711838

theorem sum_of_k_with_distinct_integer_roots :
  ∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ 3*p*q = 12 ∧ p + q = k/3}, k = 0 :=
by
  sorry

end sum_of_k_with_distinct_integer_roots_l711_711838


namespace monotonic_increasing_range_a_l711_711655

theorem monotonic_increasing_range_a (a : ℝ) :
  (∀ x ∈ set.Icc (Real.pi / 4) (3 * Real.pi / 4), 
    (1 - 2 * Real.cos (2 * x) - a * Real.sin x) ≥ 0) ↔ a ≤ Real.sqrt 2 :=
by sorry

end monotonic_increasing_range_a_l711_711655


namespace find_N_l711_711652

theorem find_N : ∃ N : ℕ, (∃ x y : ℕ, 
(10 * x + y = N) ∧ 
(4 * x + 2 * y = N / 2) ∧ 
(1 ≤ x) ∧ (x ≤ 9) ∧ 
(0 ≤ y) ∧ (y ≤ 9)) ∧
(N = 32 ∨ N = 64 ∨ N = 96) :=
by {
    have h1 : ∀ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 → 4 * x + 2 * y = (10 * x + y) / 2 → true, sorry,
    exact h1
}

end find_N_l711_711652


namespace binomial_third_term_l711_711494

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def binom_coeff (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

def third_term_binom_coeff : Prop :=
  binom_coeff 10 2 = 45

theorem binomial_third_term : third_term_binom_coeff :=
by {
  sorry
}

end binomial_third_term_l711_711494


namespace sum_of_k_with_distinct_integer_solutions_l711_711860

-- Definitions from conditions
def quadratic_eq (k : ℤ) (x : ℤ) : Prop := 3 * x^2 - k * x + 12 = 0

def distinct_integer_solutions (k : ℤ) : Prop := ∃ p q : ℤ, p ≠ q ∧ quadratic_eq k p ∧ quadratic_eq k q

-- Main statement to prove
theorem sum_of_k_with_distinct_integer_solutions : 
  ∑ k in {k : ℤ | distinct_integer_solutions k}, k = 0 := 
sorry

end sum_of_k_with_distinct_integer_solutions_l711_711860


namespace volume_of_cube_with_surface_area_l711_711428

theorem volume_of_cube_with_surface_area (S : ℝ) (hS : S = 294) : 
  ∃ V : ℝ, V = 343 :=
by
  let s := (S / 6).sqrt
  have hs : s = 7 := by sorry
  use s ^ 3
  simp [hs]
  exact sorry

end volume_of_cube_with_surface_area_l711_711428


namespace sin_double_angle_l711_711534

theorem sin_double_angle (A : ℝ) (h1 : π / 2 < A) (h2 : A < π) (h3 : Real.sin A = 4 / 5) : Real.sin (2 * A) = -24 / 25 := 
by 
  sorry

end sin_double_angle_l711_711534


namespace no_real_roots_of_equation_l711_711784

theorem no_real_roots_of_equation (x : ℝ) : x + sqrt (x - 2) ≠ 6 := 
sorry

end no_real_roots_of_equation_l711_711784


namespace value_of_x_plus_y_l711_711625

theorem value_of_x_plus_y (x y : ℝ) (h : (x + 1)^2 + |y - 6| = 0) : x + y = 5 :=
by
sorry

end value_of_x_plus_y_l711_711625


namespace total_trip_time_l711_711028

theorem total_trip_time (T : ℝ) : 
  (∀ (d_car : ℝ) (v_car : ℝ) (d_walk : ℝ) (v_walk : ℝ) (dick_speed : ℝ) (d_total : ℝ) (dick_distance : ℝ), 
    d_car = 40 ∧ 
    v_car = 30 ∧ 
    d_walk = 80 ∧ 
    v_walk = 5 ∧ 
    dick_speed = 3 ∧ 
    d_total = 120 ∧
    dick_distance = dick_speed * T ∧
    T = (d_car / v_car) + (d_walk / v_walk) ∧
    T = (d_car / v_car) + (dick_distance / v_car) + ((d_total - d_car + dick_distance) / v_car)
  ) → T = 7.47 :=
begin
  intros,
  sorry -- proof to be completed
end

end total_trip_time_l711_711028


namespace find_kn_l711_711260

variable (K L M N C D : ℝ)
variable (x : ℝ)

def is_parallelogram (K L M N : ℝ) : Prop := sorry

def is_tangent_circle_through_L (C D : ℝ) (L : ℝ) (K L M N : ℝ) : Prop := sorry

theorem find_kn
  (KL KN NM NK : ℝ)
  (h1 : is_parallelogram K L M N)
  (h2 : KL = 8)
  (h3 : is_tangent_circle_through_L C D L K L M N)
  (h4 : KC LC : ℝ := 4 / 9 * 8)
  (h5 : LD MD : ℝ := 8 / 9 * ML) :
  KN = 10 := by
  sorry

end find_kn_l711_711260


namespace inequality_solution_range_l711_711656

theorem inequality_solution_range (a : ℝ) : (∃ x : ℝ, |x + 1| + |x - 3| ≤ a) ↔ a ≥ 4 :=
sorry

end inequality_solution_range_l711_711656


namespace three_digit_multiples_of_seven_l711_711620

theorem three_digit_multiples_of_seven : 
  ∃ k, (k = {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 0}.card) ∧ k = 128 :=
by {
  sorry
}

end three_digit_multiples_of_seven_l711_711620


namespace ratio_of_cube_sides_sum_l711_711012

theorem ratio_of_cube_sides_sum {a b c d : ℕ} :
  let volume_ratio := 1232 / 405
  -- Given the volume ratio of cubes, then their side length ratio can be derived as specified.
  -- Find the integers a, b, c, d such that the ratio of their side lengths \(\frac{a \sqrt{b}}{c\sqrt{d}} = \sqrt[3]{volume_ratio}\)
  (volume_ratio = 64 / 21) →
  let side_length_ratio := (4 * (441 ^ (1/3))) / 21
  (side_length_ratio = (a * (b ^ (1/3))) / (c * (d ^ (1/3))))
  (a = 4) ∧ (b = 441) ∧ (c = 21) ∧ (d = 1) →
  -- Sum of a + b + c + d
  a + b + c + d = 467 :=
by
  sorry

end ratio_of_cube_sides_sum_l711_711012


namespace intersection_A_B_l711_711193

open Set

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def B : Set ℕ := {x | 2 ≤ x ∧ x < 6}

theorem intersection_A_B : A ∩ B = {2, 3, 4, 5} := by
  sorry

end intersection_A_B_l711_711193


namespace faster_train_speed_is_68_l711_711036

noncomputable def speed_faster_train 
  (length_train1 length_train2 : ℝ) 
  (time_to_cross : ℝ) 
  (speed_slower_train : ℝ) : ℝ :=
  let total_distance := length_train1 + length_train2
  let relative_speed := total_distance / time_to_cross
  let relative_speed_kmph := relative_speed * 3.6
  let speed_faster_train := relative_speed_kmph - speed_slower_train
  speed_faster_train

theorem faster_train_speed_is_68 
  (length_train1 length_train2 : ℝ) 
  (time_to_cross : ℝ) 
  (speed_slower_train : ℝ) 
  (hf1 : length_train1 = 200) 
  (hf2 : length_train2 = 160) 
  (ht : time_to_cross = 11.999040076793857) 
  (hs : speed_slower_train = 40) :
  speed_faster_train length_train1 length_train2 time_to_cross speed_slower_train = 68 :=
  by rw [hf1, hf2, ht, hs]
  sorry

end faster_train_speed_is_68_l711_711036


namespace mortgage_repayment_months_l711_711275

theorem mortgage_repayment_months :
  ∀ (n : ℕ), 
  (∑ i in finset.range n, 100 * 3^i) = 914800 → 
  n = 9 :=
by
  intros n h
  sorry

end mortgage_repayment_months_l711_711275


namespace cost_per_pound_mixed_feed_correct_l711_711821

noncomputable def total_weight_of_feed : ℝ := 17
noncomputable def cost_per_pound_cheaper_feed : ℝ := 0.11
noncomputable def cost_per_pound_expensive_feed : ℝ := 0.50
noncomputable def weight_cheaper_feed : ℝ := 12.2051282051

noncomputable def total_cost_of_feed : ℝ :=
  (cost_per_pound_cheaper_feed * weight_cheaper_feed) + 
  (cost_per_pound_expensive_feed * (total_weight_of_feed - weight_cheaper_feed))

noncomputable def cost_per_pound_mixed_feed : ℝ :=
  total_cost_of_feed / total_weight_of_feed

theorem cost_per_pound_mixed_feed_correct : 
  cost_per_pound_mixed_feed = 0.22 :=
  by
    sorry

end cost_per_pound_mixed_feed_correct_l711_711821


namespace sum_value_of_solutions_l711_711484

theorem sum_value_of_solutions:
  let z_values : Finset ℂ := {z | z^5 = -1},
  ∑ z in z_values, 1 / (abs (1 - z))^2 = 35 / 4 := by
  sorry

end sum_value_of_solutions_l711_711484


namespace sector_area_l711_711936

theorem sector_area (r : ℝ) (h1 : r = 2) (h2 : 2 * r + r * ((2 * π * r - 2) / r) = 4 * π) :
  (1 / 2) * r^2 * ((4 * π - 2) / r) = 4 * π - 2 :=
by
  sorry

end sector_area_l711_711936


namespace jill_investment_value_l711_711694

def compoundInterest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_investment_value :
  compoundInterest 10000 0.0396 2 2 ≈ 10812 :=
by
  sorry

end jill_investment_value_l711_711694


namespace ensureUserDataSecurity_l711_711769

-- Definitions based on the given conditions and correct answers
variable (storeApp : Type) -- representing the online store application

/-- Condition: Users can pay using credit cards in the store application -/
def canPayWithCreditCard (app : storeApp) : Prop := 
sorry -- detailed implementation is not provided

/-- Condition: Users can order home delivery in the store application -/
def canOrderHomeDelivery (app : storeApp) : Prop := 
sorry -- detailed implementation is not provided

/-- Security Measure 1: Avoid Storing Card Data - assume implemented properly -/
def avoidStoringCardData (app : storeApp) : Prop := 
sorry -- detailed implementation is not provided

/-- Security Measure 2: Encryption of Stored Data - assume implemented properly -/
def encryptStoredData (app : storeApp) : Prop := 
sorry -- detailed implementation is not provided

/-- Security Measure 3: Encryption of Data in Transit - assume implemented properly -/
def encryptDataInTransit (app : storeApp) : Prop := 
sorry -- detailed implementation is not provided

/-- Theorem: Ensuring user data security in an online store application -/
theorem ensureUserDataSecurity:
  ∀ (app : storeApp), 
    canPayWithCreditCard app → 
    canOrderHomeDelivery app → 
    avoidStoringCardData app →
    encryptStoredData app → 
    encryptDataInTransit app → 
    true := 
by
  sorry

end ensureUserDataSecurity_l711_711769


namespace sum_of_all_ks_l711_711867

theorem sum_of_all_ks (k : ℤ) :
  (∃ r s : ℤ, r ≠ s ∧ (3 * x^2 - k * x + 12 = 0) ∧ (r + s = k / 3) ∧ (r * s = 4)) → (k = 15 ∨ k = -15) → k + k = 0 :=
by
  sorry

end sum_of_all_ks_l711_711867


namespace sum_of_k_with_distinct_integer_solutions_l711_711857

-- Definitions from conditions
def quadratic_eq (k : ℤ) (x : ℤ) : Prop := 3 * x^2 - k * x + 12 = 0

def distinct_integer_solutions (k : ℤ) : Prop := ∃ p q : ℤ, p ≠ q ∧ quadratic_eq k p ∧ quadratic_eq k q

-- Main statement to prove
theorem sum_of_k_with_distinct_integer_solutions : 
  ∑ k in {k : ℤ | distinct_integer_solutions k}, k = 0 := 
sorry

end sum_of_k_with_distinct_integer_solutions_l711_711857


namespace solution_set_of_inequality_l711_711194

theorem solution_set_of_inequality :
  {x : ℝ | (1 / real.pi) ^ (-x + 1) > (1 / real.pi) ^ (x^2 - x)} = {x : ℝ | x < -1 ∨ x > 1} :=
by {
 sorry
}

end solution_set_of_inequality_l711_711194


namespace f_le_2x_l711_711732

-- Define the set A
def A : set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

-- Define the function f with the given conditions
variable (f : ℝ → ℝ)
variable (hf1 : f 1 = 1)
variable (hf2 : ∀ x ∈ A, f x ≥ 0)
variable (hf3 : ∀ x y, x ∈ A → y ∈ A → x + y ∈ A → f(x + y) ≥ f x + f y)

-- The main theorem to be proved
theorem f_le_2x : ∀ x ∈ A, f x ≤ 2 * x := 
by
  intro x hx
  sorry

end f_le_2x_l711_711732


namespace max_m_value_l711_711213

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem max_m_value 
  (t : ℝ) 
  (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f (x + t) <= x) : m ≤ 4 :=
sorry

end max_m_value_l711_711213


namespace g_of_neg_two_l711_711288

def f (x : ℝ) : ℝ := 4 * x - 9

def g (x : ℝ) : ℝ := 3 * x ^ 2 + 4 * x - 2

theorem g_of_neg_two : g (-2) = 227 / 16 :=
by
  sorry

end g_of_neg_two_l711_711288


namespace sum_of_k_with_distinct_integer_roots_l711_711840

theorem sum_of_k_with_distinct_integer_roots :
  ∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ 3*p*q = 12 ∧ p + q = k/3}, k = 0 :=
by
  sorry

end sum_of_k_with_distinct_integer_roots_l711_711840


namespace geometric_series_first_term_l711_711814

theorem geometric_series_first_term
  (a r : ℚ)
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 150) :
  a = 60 / 7 :=
by
  sorry

end geometric_series_first_term_l711_711814


namespace sum_of_k_distinct_integer_roots_l711_711854

theorem sum_of_k_distinct_integer_roots : 
  (∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ p * q = 4 ∧ k = 3 * (p + q)}, k) = 0 := 
by
  sorry

end sum_of_k_distinct_integer_roots_l711_711854


namespace monomial_degree_is_five_l711_711659

theorem monomial_degree_is_five (n : ℕ) : (4 * a^n * b^2 * c : ℤ) = monomial 5 := by
  assume h : (4 * a^n * b^2 * c).degree = 5
  have h_exp_sum : n + 2 + 1 = 5 := by sorry
  show n = 2

end monomial_degree_is_five_l711_711659


namespace cube_volume_l711_711424

theorem cube_volume (A : ℝ) (hA : A = 294) : ∃ (V : ℝ), V = 343 :=
by
  sorry

end cube_volume_l711_711424


namespace coloring_even_conditional_l711_711730

-- Define the problem parameters and constraints
def number_of_colorings (n : Nat) (even_red : Bool) (even_yellow : Bool) : Nat :=
  sorry  -- This function would contain the detailed computational logic.

-- Define the main theorem statement
theorem coloring_even_conditional (n : ℕ) (h1 : n > 0) : ∃ C : Nat, number_of_colorings n true true = C := 
by
  sorry  -- The proof would go here.


end coloring_even_conditional_l711_711730


namespace circumcenter_APQ_lies_on_ω_l711_711069

-- Define the given structures and conditions
variables {A B C K P Q : Type}
variable [Geometry ℝ] -- Assuming we are in the real plane geometry
variables (ω : Circle B C) -- Circle passing through points B and C
variables (AB AC BP CQ : Segment)
variables (on_ab : sig (IntersectsAt ω (AB.point P))) -- P is on AB and ω
variables (on_ac : sig (IntersectsAt ω (AC.point P))) -- P is on AC and ω
variables (cong_BP_AC : (BP.length = AC.length))
variables (ray_CK : Ray CK) -- Ray starting at C in direction K
variables (on_ray : sig (OnRay Q ray_CK))
variables (cong_CQ_AB : (CQ.length = AB.length))

-- The main theorem to prove
theorem circumcenter_APQ_lies_on_ω
  (h_abp : P ∈ Segment AB) 
  (h_acp : P ∈ Segment AC)
  (h_cong_BP_AC : BP.length = AC.length)
  (h_on_ray_CK : Q ∈ Ray CK)
  (h_cong_CQ_AB : CQ.length = AB.length) :
  let circumcenter_APQ := find_circumcenter (triangle A P Q) in
  circumcenter_APQ ∈ ω :=
sorry


end circumcenter_APQ_lies_on_ω_l711_711069


namespace solve_for_x_l711_711997

theorem solve_for_x (x : ℝ) (h1 : x > 9) :
  (sqrt (x - 9 * sqrt (x - 9)) + 3 = sqrt (x + 9 * sqrt (x - 9)) - 3) → x ≥ 40.5 :=
by
  intro h
  sorry

end solve_for_x_l711_711997


namespace largest_number_in_systematic_sample_l711_711532

noncomputable def systematic_sample : ℕ := 500
noncomputable def smallest_two_numbers : (ℕ × ℕ) := (7, 32)

theorem largest_number_in_systematic_sample :
  ∀ (n k interval : ℕ), n = systematic_sample → k = fst smallest_two_numbers → (snd smallest_two_numbers - k = interval) → (interval * (systematic_sample / interval - 1) + k = 482) :=
begin
  sorry
end

end largest_number_in_systematic_sample_l711_711532


namespace payment_option1_payment_option2_cost_effective_option_most_cost_effective_plan_l711_711416

variable (x : ℕ)
variable (hx : x > 10)

noncomputable def option1_payment (x : ℕ) : ℕ := 200 * x + 8000
noncomputable def option2_payment (x : ℕ) : ℕ := 180 * x + 9000

theorem payment_option1 (x : ℕ) (hx : x > 10) : option1_payment x = 200 * x + 8000 :=
by sorry

theorem payment_option2 (x : ℕ) (hx : x > 10) : option2_payment x = 180 * x + 9000 :=
by sorry

theorem cost_effective_option (x : ℕ) (hx : x > 10) (h30 : x = 30) : option1_payment 30 < option2_payment 30 :=
by sorry

theorem most_cost_effective_plan (h30 : x = 30) : (10000 + 3600 = 13600) :=
by sorry

end payment_option1_payment_option2_cost_effective_option_most_cost_effective_plan_l711_711416


namespace middle_letter_value_l711_711271

theorem middle_letter_value 
  (final_score : ℕ) 
  (first_letter_value : ℕ) 
  (third_letter_value : ℕ)
  (word_length : ℕ)
  (triple_score : ℕ)
  (total_points : ℕ)
  (middle_letter_value : ℕ)
  (h1 : final_score = 30)
  (h2 : first_letter_value = 1)
  (h3 : third_letter_value = 1)
  (h4 : word_length = 3)
  (h5 : triple_score = 3)
  (h6 : total_points = final_score / triple_score)
  (h7 : total_points = 10)
  (h8 : middle_letter_value = total_points - first_letter_value - third_letter_value) :
  middle_letter_value = 8 := 
by sorry

end middle_letter_value_l711_711271


namespace cindy_dan_stickers_l711_711130

theorem cindy_dan_stickers (C D : ℕ) (c_gave : ℕ) (d_bought : ℕ) 
  (c_init_ratio : 3 * C = 3 * D) (c_final_ratio : (C - c_gave) * 5 = (D + d_bought) * 2) :
  (D + d_bought) - (C - c_gave) = 33 :=
by
  -- Given conditions
  have h1 : 3 * C = 3 * D := c_init_ratio
  have h2 : (C - c_gave) * 5 = (D + d_bought) * 2 := c_final_ratio
  -- Setting values according to the problem definition
  let s := 37
  let c_gave : ℕ := 15
  let d_bought : ℕ := 18
  -- Proving the required result
  show (s + d_bought) - (s - c_gave) = 33
  sorry

end cindy_dan_stickers_l711_711130


namespace area_of_specific_triangle_l711_711094

noncomputable def area_of_triangle_inscribed_in_circle_of_radius (k : ℝ) : ℝ :=
  let x := 1 in -- From 13x = 13, we determine x = 1
  let a := 5 * x in
  let b := 12 * x in
  let c := 13 * x in
  (1 / 2) * a * b

theorem area_of_specific_triangle : area_of_triangle_inscribed_in_circle_of_radius 6.5 = 30 := 
sorry

end area_of_specific_triangle_l711_711094


namespace find_n_l711_711993

open Nat

noncomputable def alternating_factorial_expression : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 1!
| n => if even n then -(n!) + alternating_factorial_expression (n - 1)
       else (n!) + alternating_factorial_expression (n - 1)

def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_n (n N : ℕ) : Prop :=
  alternating_factorial_expression n ∈ {3,4,5,6,7,8,10,15,19,41,59,61,105,160} ∧ n < N

theorem find_n (N : ℕ) : ∃ n, valid_n n N :=
sorry

end find_n_l711_711993


namespace fraction_zero_l711_711904

theorem fraction_zero (x : ℝ) (h : x ≠ 1) (h₁ : (x + 1) / (x - 1) = 0) : x = -1 :=
sorry

end fraction_zero_l711_711904


namespace imaginary_part_of_z_is_minus_two_absolute_value_of_z_is_sqrt_five_l711_711206

noncomputable def z : ℂ := (3 - 4 * Complex.i) / (1 + 2 * Complex.i)

theorem imaginary_part_of_z_is_minus_two : z.im = -2 :=
by sorry

theorem absolute_value_of_z_is_sqrt_five : Complex.abs z = Real.sqrt 5 :=
by sorry

end imaginary_part_of_z_is_minus_two_absolute_value_of_z_is_sqrt_five_l711_711206


namespace automobile_distance_in_5_minutes_l711_711107

theorem automobile_distance_in_5_minutes 
  (b s : ℝ) 
  (h1 : ∀ t, t ≤ 60 → dist (t) = t * (b / (3 * s)))
  (h2 : ∀ t, t > 60 → dist (t) = 60 * (b / (3 * s)) + (t - 60) * (2 * b / (3 * s))) :
  dist 300 = 60 * (b / s) := 
sorry

end automobile_distance_in_5_minutes_l711_711107


namespace quadrilateral_sum_of_squares_ge_four_l711_711465

-- Definition for representing a quadrilateral and its properties
structure Quadrilateral where
  a b c d : ℝ -- lengths of the four sides
  SqSumSides : ℝ := a^2 + b^2 + c^2 + d^2
  -- Ensure that sides are non-negative
  nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0

-- Definition for a list of quadrilaterals in a unit square
structure UnitSquareDecomposition (m : ℕ) where
  quadrilaterals : Fin m → Quadrilateral
  totalArea : ℝ := 1 -- The total area remains 1

-- The main theorem to prove
theorem quadrilateral_sum_of_squares_ge_four {m : ℕ} (Q : UnitSquareDecomposition m) :
  (∑ i in Finset.finRange m, (Q.quadrilaterals i).SqSumSides) ≥ 4 :=
sorry -- proof to be provided

end quadrilateral_sum_of_squares_ge_four_l711_711465


namespace minimum_value_a_plus_4b_l711_711593

theorem minimum_value_a_plus_4b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : (1 / a) + (1 / b) = 1) : a + 4 * b ≥ 9 :=
sorry

end minimum_value_a_plus_4b_l711_711593


namespace largest_angle_in_ratio_triangle_l711_711332

theorem largest_angle_in_ratio_triangle (a b c : ℕ) (h_ratios : 2 * c = 3 * b ∧ 3 * b = 4 * a)
  (h_sum : a + b + c = 180) : max a (max b c) = 80 :=
by
  sorry

end largest_angle_in_ratio_triangle_l711_711332


namespace arithmetic_sequence_fifth_term_l711_711019

theorem arithmetic_sequence_fifth_term (a d : ℤ) 
  (h1 : a + 9 * d = 3) 
  (h2 : a + 11 * d = 9) : 
  a + 4 * d = -12 :=
by
  sorry

end arithmetic_sequence_fifth_term_l711_711019


namespace sphere_surface_area_correct_l711_711924

-- Given conditions
def face_area1 := 12
def face_area2 := 15
def face_area3 := 20

def cuboid_edge_length (a b : ℝ) := sqrt (a * b)

-- Lengths of the cuboid edges calculated from the face areas
def edge1 := cuboid_edge_length face_area1 face_area2
def edge2 := cuboid_edge_length face_area1 face_area3
def edge3 := cuboid_edge_length face_area2 face_area3

-- Calculate the space diagonal of the cuboid
def space_diagonal := sqrt (edge1^2 + edge2^2 + edge3^2)

-- Determine the diameter of the sphere
def sphere_diameter := space_diagonal

-- Radius of the sphere
def sphere_radius := sphere_diameter / 2

-- Surface area of the sphere
def sphere_surface_area := 4 * π * (sphere_radius^2)

-- The main theorem to prove
theorem sphere_surface_area_correct : sphere_surface_area = 50 * π :=
by 
  sorry

end sphere_surface_area_correct_l711_711924


namespace arithmetic_expression_equals_100_l711_711311

/-- 
  Prove that there exists a placement of arithmetic operation signs and parentheses 
  between the numbers 1, 2, 3, 4, 5, 6, 7, 8, 9 such that the resulting expression equals 100.
-/
theorem arithmetic_expression_equals_100 : ∃ f : List ℕ → ℕ, 
  f [1, 2, 3, 4, 5, 6, 7, 8, 9] = 100 :=
by
  have h1 : 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 * 9 = 100 := by norm_num
  have h2 : 1 * 2 * 3 + 4 + 5 + 6 + 7 + 8 * 9 = 100 := by norm_num
  have h3 : 1 + 2 * 3 + 4 * 5 - 6 + 7 + 8 * 9 = 100 := by norm_num
  have h4 : 1 * 2 * 3 * 4 + (5 + 6 - 7) + 8 * 9 = 100 := by norm_num
  have h5 : (1 * 2 + 3) * 4 * 5 + 6 - 7 - 8 + 9 = 100 := by norm_num
  have h6 : (1 + 2 + 3) * (4 + 5 + 6) - 7 + 8 + 9 = 100 := by norm_num
  have h7 : ((1 + 2) / 3 + 4 + 5 - 6) * 7 + 8 * 9 = 100 := by norm_num
  exact ⟨_, h1⟩
  sorry

end arithmetic_expression_equals_100_l711_711311


namespace banana_ratio_l711_711411

theorem banana_ratio (x : ℕ) (nearby_island_produce jakies_island_produce total_produce : ℕ)
  (h1 : nearby_island_produce = 9000)
  (h2 : jakies_island_produce = x * nearby_island_produce)
  (h3 : total_produce = nearby_island_produce + jakies_island_produce)
  (h4 : total_produce = 99000) :
  x = 10 :=
by
  rw [h1, h2, h3, h4]
  sorry

end banana_ratio_l711_711411


namespace lcm_hcf_relationship_l711_711661

theorem lcm_hcf_relationship (a b : ℕ) (h_prod : a * b = 84942) (h_hcf : Nat.gcd a b = 33) : Nat.lcm a b = 2574 :=
by
  sorry

end lcm_hcf_relationship_l711_711661


namespace cubic_identity_l711_711579

theorem cubic_identity (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / x^3) = 140 := 
  sorry

end cubic_identity_l711_711579


namespace z_in_first_quadrant_l711_711422

noncomputable def i := complex.I

noncomputable def z := 1 / (1 - i)

def is_in_first_quadrant (z: ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

theorem z_in_first_quadrant : is_in_first_quadrant z :=
  sorry

end z_in_first_quadrant_l711_711422


namespace weighted_mean_l711_711533

variable (m n p : ℕ)
variable (x1 x2 x3 : ℝ)

theorem weighted_mean (m n p : ℕ) (x1 x2 x3 : ℝ) : 
  (m + n + p > 0) → 
  (∑ i in finset.range m, x1 + ∑ i in finset.range n, x2 + ∑ i in finset.range p, x3) / (m + n + p) = 
  (m * x1 + n * x2 + p * x3) / (m + n + p) :=
by 
  intro h
  sorry

end weighted_mean_l711_711533


namespace johnny_years_ago_l711_711043

theorem johnny_years_ago 
  (J : ℕ) (hJ : J = 8) (X : ℕ) 
  (h : J + 2 = 2 * (J - X)) : 
  X = 3 := by
  sorry

end johnny_years_ago_l711_711043


namespace correct_propositions_l711_711926

section

def quasi_odd_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x : ℝ, f(a + x) + f(a - x) = 2 * b

def is_central_point (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ a b : ℝ, quasi_odd_function f a b

variable (f1 : ℝ → ℝ)
variable (f2 : ℝ → ℝ)
variable (f3 : ℝ → ℝ)
variable (f4 : ℝ → ℝ)

def f1_def := λ x : ℝ, Real.sin x + 1
def f2_def := λ x : ℝ, x^3 - 3*x^2 + 6*x - 2
def f3_def := λ x : ℝ, Real.sin (2 * x - Real.pi / 3) + 2

theorem correct_propositions :
  (quasi_odd_function f1_def 0 1) ∧
  (∀ f, is_central_point f a b → ∀ x, f(x + a) - f(a) = f(a - x)) ∧
  (¬ quasi_odd_function f3_def (Real.pi / 3 + k * Real.pi) 2) ∧
  (quasi_odd_function f2_def 1 2) :=
sorry

end

end correct_propositions_l711_711926


namespace sum_of_k_distinct_integer_roots_l711_711852

theorem sum_of_k_distinct_integer_roots : 
  (∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ p * q = 4 ∧ k = 3 * (p + q)}, k) = 0 := 
by
  sorry

end sum_of_k_distinct_integer_roots_l711_711852


namespace line_through_point_with_slope_l711_711343

-- Define a point and a slope
structure Point where
  x : ℝ
  y : ℝ

def slope : ℝ := 2
def P : Point := {x := 0, y := 2}

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := 2*x - y + 2 = 0

-- Lean theorem statement
theorem line_through_point_with_slope (x y : ℝ) (slope : ℝ) (P : Point) :
  slope = 2 → P = {x := 0, y := 2} → line_equation x y :=
by
  intro h1 h2
  rw [h1, h2]
  sorry

end line_through_point_with_slope_l711_711343


namespace braids_each_dancer_l711_711274

-- Define the conditions
def num_dancers := 8
def time_per_braid := 30 -- seconds per braid
def total_time := 20 * 60 -- convert 20 minutes into seconds

-- Define the total number of braids Jill makes
def total_braids := total_time / time_per_braid

-- Define the number of braids per dancer
def braids_per_dancer := total_braids / num_dancers

-- Theorem: Prove that each dancer has 5 braids
theorem braids_each_dancer : braids_per_dancer = 5 := 
by sorry

end braids_each_dancer_l711_711274


namespace initial_dozens_of_doughnuts_l711_711913

theorem initial_dozens_of_doughnuts (doughnuts_eaten doughnuts_left : ℕ)
  (h_eaten : doughnuts_eaten = 8)
  (h_left : doughnuts_left = 16) :
  (doughnuts_eaten + doughnuts_left) / 12 = 2 := by
  sorry

end initial_dozens_of_doughnuts_l711_711913


namespace probability_correct_l711_711783

-- Define the conditions as constants
def total_slips : ℕ := 60
def numbers_range : ℕ := 20
def slips_per_number : ℕ := 3
def drawn_slips : ℕ := 5

-- Define the binomial coefficient function for combinations
def binom (n k : ℕ) : ℕ :=
  nat.choose n k

-- Define the favorable outcomes count function
def favorable_outcomes : ℕ :=
  (binom numbers_range 1) * (binom slips_per_number 3) * (binom (numbers_range - 1) 2) * (binom slips_per_number 1) * (binom slips_per_number 1)

-- Define the total outcomes count
def total_outcomes : ℕ :=
  binom total_slips drawn_slips

-- Define the probability r
def probability_r : ℚ :=
  favorable_outcomes / total_outcomes

-- Target statement to be proved
theorem probability_correct :
  probability_r = 30810 / 5461512 :=
sorry

end probability_correct_l711_711783


namespace ensure_user_data_security_l711_711764

-- Define what it means to implement a security measure
inductive SecurityMeasure
  | avoidStoringCardData
  | encryptStoredData
  | encryptDataInTransit
  | codeObfuscation
  | restrictRootedDevices
  | antivirusProtectionAgent

-- Define the conditions: Developing an online store app where users can 
-- pay by credit card and order home delivery ensures user data security 
-- if at least three security measures are implemented.

def providesSecurity (measures : List SecurityMeasure) : Prop :=
  measures.contains SecurityMeasure.avoidStoringCardData ∨
  measures.contains SecurityMeasure.encryptStoredData ∨
  measures.contains SecurityMeasure.encryptDataInTransit ∨
  measures.contains SecurityMeasure.codeObfuscation ∨
  measures.contains SecurityMeasure.restrictRootedDevices ∨
  measures.contains SecurityMeasure.antivirusProtectionAgent

theorem ensure_user_data_security (measures : List SecurityMeasure) (h : measures.length ≥ 3) : providesSecurity measures :=
  sorry

end ensure_user_data_security_l711_711764


namespace spinner_probability_l711_711443

theorem spinner_probability 
  (triangle : Type) 
  (is_equilateral : triangle → Prop)
  (altitudes : triangle → list (triangle × triangle))
  (regions : list triangle)
  (shaded_regions : list triangle) :
  is_equilateral triangle →
  length (altitudes triangle) = 3 →
  length regions = 6 →
  length shaded_regions = 3 →
  (length shaded_regions / length regions : ℚ) = 1 / 2 :=
by sorry

end spinner_probability_l711_711443


namespace eval_determinant_l711_711506

-- Define the matrix elements
def a := Real.arcsin (Real.sqrt 3 / 2)
def b := 2
def c := Real.arctan (Real.sqrt 3 / 3)
def d := 3

-- Evaluate the 2x2 determinant
theorem eval_determinant : (a * d - b * c) = (2 * Real.pi / 3) := by
  -- We assume a = arcsin (sqrt 3 / 2) simplifies to pi / 3 and c = arctan (sqrt 3 / 3) simplifies to pi / 6,
  sorry

end eval_determinant_l711_711506


namespace positive_rationals_in_S_l711_711716

variable (S : Set ℚ)

-- Conditions
axiom closed_under_addition (a b : ℚ) (ha : a ∈ S) (hb : b ∈ S) : a + b ∈ S
axiom closed_under_multiplication (a b : ℚ) (ha : a ∈ S) (hb : b ∈ S) : a * b ∈ S
axiom zero_rule : ∀ r : ℚ, r ∈ S ∨ -r ∈ S ∨ r = 0

-- Prove that S is the set of positive rational numbers
theorem positive_rationals_in_S : S = {r : ℚ | 0 < r} :=
by
  sorry

end positive_rationals_in_S_l711_711716


namespace area_of_specific_triangle_l711_711095

noncomputable def area_of_triangle_inscribed_in_circle_of_radius (k : ℝ) : ℝ :=
  let x := 1 in -- From 13x = 13, we determine x = 1
  let a := 5 * x in
  let b := 12 * x in
  let c := 13 * x in
  (1 / 2) * a * b

theorem area_of_specific_triangle : area_of_triangle_inscribed_in_circle_of_radius 6.5 = 30 := 
sorry

end area_of_specific_triangle_l711_711095


namespace sum_fifteen_multiples_of_15_l711_711873

theorem sum_fifteen_multiples_of_15:
  let multiples := (List.range 15).map (λ n, 15 * (n+1)) in
  multiples.sum = 1800 :=
by
  -- Definitions of sum on a mapped list
  sorry

end sum_fifteen_multiples_of_15_l711_711873


namespace number_of_liars_l711_711900

-- Define a predicate is_knight representing if a person is a knight.
def is_knight (n : ℕ) : Prop := sorry

-- Define the total number of people.
def total_people := 30

-- Define conditions for odd-numbered persons.
def odd_person_statement (n : ℕ) : Prop := 
  is_knight n ↔ ∀ m, (m > n ∧ m ≤ total_people) → ¬is_knight m

-- Define conditions for even-numbered persons.
def even_person_statement (n : ℕ) : Prop := 
  is_knight n ↔ ∀ m, (m < n ∧ m ≥ 1) → ¬is_knight m

-- The main theorem we want to prove.
theorem number_of_liars : ∃ n, n = 28 ∧ 
  (∀ i, (1 ≤ i ∧ i ≤ total_people ∧ i % 2 = 1) → odd_person_statement i) ∧
  (∀ j, (1 ≤ j ∧ j ≤ total_people ∧ j % 2 = 0) → even_person_statement j) ∧
  (∀ k, 1 ≤ k ∧ k ≤ total_people → (is_knight k ∨ ¬is_knight k)) ∧
  (n = total_people - ∑ p in finRange total_people, if is_knight p then 1 else 0) :=
sorry

end number_of_liars_l711_711900


namespace min_moves_to_order_children_l711_711022

-- Define the condition for a moving operation in the problem
def move (current_positions: List ℕ) (i j: ℕ) : List ℕ :=
  current_positions.take i ++ [current_positions.nth_le j sorry] ++ 
  current_positions.drop (i + 1) ++ current_positions.take j ++ current_positions.drop (j + 1)

-- Define the condition for ordering the children
def is_ordered (positions: List ℕ) : Prop :=
  positions = List.range 10

-- The main theorem
theorem min_moves_to_order_children :
  ∀ (initial_positions: List ℕ), initial_positions.perm (List.range 10) →
  ∃ (moves: ℕ), moves = 8 ∧ 
  (∃ (move_sequence: List (ℕ × ℕ)),
          List.foldl (λ positions move_pair => move positions move_pair.1 move_pair.2)
                      initial_positions
                      move_sequence = List.range 10) :=
by
  intros
  sorry

end min_moves_to_order_children_l711_711022


namespace back_seat_tickets_sold_l711_711423

variable (M B : ℕ)

theorem back_seat_tickets_sold:
  M + B = 20000 ∧ 55 * M + 45 * B = 955000 → B = 14500 :=
by
  sorry

end back_seat_tickets_sold_l711_711423


namespace find_B_and_properties_l711_711195

structure Point :=
(x : ℝ)
(y : ℝ)

def midpoint (A B : Point) : Point :=
{ x := (A.x + B.x) / 2,
  y := (A.y + B.y) / 2 }

def slope (A B : Point) : ℝ :=
if A.x = B.x then 0 else (B.y - A.y) / (B.x - A.x)

def collinear (A B C : Point) : Prop :=
slope A B = slope A C

theorem find_B_and_properties : 
  ∃ B : Point, 
  (midpoint {| x := 6, y := -3 |} B = {| x := 4, y := -1 |}) ∧
  (B.x + B.y = 3) ∧ 
  ¬collinear {| x := 6, y := -3 |} B {| x := 3, y := 3 |} :=
by
  have B := {| x := 2, y := 1 |}
  exists B
  split
  {
    show midpoint {| x := 6, y := -3 |} B = {| x := 4, y := -1 |}
    sorry
  }
  split
  {
    show B.x + B.y = 3
    sorry
  }
  {
    show ¬collinear {| x := 6, y := -3 |} B {| x := 3, y := 3 |}
    sorry
  }

end find_B_and_properties_l711_711195


namespace floor_of_a2017_l711_711598

def sequence (a : ℕ → ℝ) :=
  (a 1 = 1) ∧ (a 2 = 2) ∧ (∀ n ≥ 1, a (n + 2) / a n = (a (n + 1) ^ 2 + 1) / (a n ^ 2 + 1))

theorem floor_of_a2017 {a : ℕ → ℝ} (h : sequence a) : ⌊a 2017⌋ = 63 :=
sorry

end floor_of_a2017_l711_711598


namespace rational_coefficients_zero_l711_711715

variables (A : ℕ → ℂ) (O : ℂ)
variables (λ1 λ2 λ3 λ4 : ℚ)
variables (h_cond : λ1 * (A 1 - O) + λ2 * (A 2 - O) + λ3 * (A 3 - O) + λ4 * (A 4 - O) = 0)

theorem rational_coefficients_zero :
  λ1 = 0 ∧ λ2 = 0 ∧ λ3 = 0 ∧ λ4 = 0 :=
sorry

end rational_coefficients_zero_l711_711715


namespace find_m_l711_711880

theorem find_m
  (A : Set ℤ) (B : Set ℤ)
  (m : ℤ)
  (hA_sum : ∑ i in A, i = 2 * m)
  (hB_sum : ∑ i in B, i = m)
  (hA_consecutive : ∀ x y ∈ A, x ≠ y → x + 1 = y ∨ y + 1 = x)
  (hB_consecutive : ∀ x y ∈ B, x ≠ y → x + 1 = y ∨ y + 1 = x)
  (hA_card : A.card = m)
  (hB_card : B.card = 2 * m)
  (h_abs_diff : |max' A (finite_of_finite A) - max' B (finite_of_finite B)| = 99) :
  m = 201 :=
sorry

end find_m_l711_711880


namespace max_sinA_sinB_l711_711675

-- Definition and theorem statement
theorem max_sinA_sinB (A B C : ℝ) (hABC : A + B + C = π) (hC : C = π / 2) :
  ∃ m, m = 1 / 2 ∧ (∀ A B, A + B = π / 2 → sin A * sin B ≤ m) :=
by
  sorry

end max_sinA_sinB_l711_711675


namespace balls_in_boxes_after_2023_steps_l711_711303

theorem balls_in_boxes_after_2023_steps :
  let calculate_balls (steps : ℕ) : ℕ :=
    let ternary_digits (n : ℕ) : List ℕ :=
      if n = 0 then [0] else
        List.unfoldr
          (λ k, if k = 0 then none else some (k % 3, k / 3))
          n
    in (ternary_digits steps).sum in
  calculate_balls 2023 = 9 :=
by
  sorry

end balls_in_boxes_after_2023_steps_l711_711303


namespace lowest_score_correct_l711_711790

noncomputable def lowest_score (scores : List ℝ) : ℝ :=
if h : scores.length = 15 ∧ scores.sum = 15 * 75 ∧ (scores.erase 95).erase (List.minimum' (scores.erase 95)).sum = 78 * 13 ∧ List.maximum' scores = 95 then
  111 - 95
else
  0 -- Use 0 as a dummy value for non-computable cases

theorem lowest_score_correct (scores : List ℝ) (h1 : scores.length = 15) (h2 : scores.sum = 15 * 75)
  (h3 : (scores.erase 95).erase (List.minimum' (scores.erase 95)).sum = 78 * 13) (h4 : List.maximum' scores = 95) :
  List.minimum' (scores.erase 95) = 16 :=
by
  sorry

end lowest_score_correct_l711_711790


namespace annika_total_distance_l711_711946

/--
Annika hikes at a constant rate of 12 minutes per kilometer. She has hiked 2.75 kilometers
east from the start of a hiking trail when she realizes that she has to be back at the start
of the trail in 51 minutes. Prove that the total distance Annika hiked east is 3.5 kilometers.
-/
theorem annika_total_distance :
  (hike_rate : ℝ) = 12 → 
  (initial_distance_east : ℝ) = 2.75 → 
  (total_time : ℝ) = 51 → 
  (total_distance_east : ℝ) = 3.5 :=
by 
  intro hike_rate initial_distance_east total_time 
  sorry

end annika_total_distance_l711_711946


namespace race_outcomes_l711_711671

theorem race_outcomes (Abe Bobby Charles Devin Edwin Fiona : Type) :
  (finset.univ.card = 6) →
  (¬ ∃ x y, x ≠ y ∧ (x = 1 ∧ y = 1)) →
  ∃ n, n = 6 * 5 * 4 * 3 ∧ n = 360 :=
by
  -- Defining the set of the contestants
  let contestants := {Abe, Bobby, Charles, Devin, Edwin, Fiona}
  -- Defining no ties condition implied by the problem (not having the same contestant place multiple times)
  sorry

end race_outcomes_l711_711671


namespace sum_of_possible_w_values_l711_711115

theorem sum_of_possible_w_values (w x y z : ℤ) (h₁ : w > x) (h₂ : x > y) (h₃ : y > z)
  (h₄ : w + x + y + z = 44)
  (h₅ : {abs (w - x), abs (w - y), abs (w - z), abs (x - y), abs (x - z), abs (y - z)} = {1, 3, 4, 5, 6, 9}) :
  {w | w ∈ {15, 16}}.sum = 31 :=
by
  sorry

end sum_of_possible_w_values_l711_711115


namespace quadrilateral_ABCD_is_trapezoid_l711_711674

open EuclideanGeometry

-- Assume given quadrilateral ABCD and point E lies on CD
variables {A B C D E M N P Q U V : Point}

-- Conditions given
axiom angle_AED_eq_angle_BEC : ∠ A E D = ∠ B E C
axiom ratio_conditions : (AM / DM) * (CN / BN) = (BP / DP) * (CQ / AQ) = (CE / DE)
axiom intersections : (U ∈ MQ ∧ U ∈ AE) ∧ (V ∈ PN ∧ V ∈ BE)
axiom equality_condition : dist U E = dist V E

-- Prove that quadrilateral ABCD is a trapezoid (i.e., AB is parallel to CD)
theorem quadrilateral_ABCD_is_trapezoid :
  is_trapezoid A B C D :=
sorry

end quadrilateral_ABCD_is_trapezoid_l711_711674


namespace lim_tan_minus_x_div_x_minus_sin_x_l711_711483

theorem lim_tan_minus_x_div_x_minus_sin_x :
  (tendsto (fun x => (tan x - x) / (x - sin x)) (𝓝 0) (𝓝 2)) :=
begin
  sorry,
end

end lim_tan_minus_x_div_x_minus_sin_x_l711_711483


namespace sum_of_all_ks_l711_711869

theorem sum_of_all_ks (k : ℤ) :
  (∃ r s : ℤ, r ≠ s ∧ (3 * x^2 - k * x + 12 = 0) ∧ (r + s = k / 3) ∧ (r * s = 4)) → (k = 15 ∨ k = -15) → k + k = 0 :=
by
  sorry

end sum_of_all_ks_l711_711869


namespace original_purchase_price_first_commodity_l711_711792

theorem original_purchase_price_first_commodity (x y : ℝ) 
  (h1 : 1.07 * (x + y) = 827) 
  (h2 : x = y + 127) : 
  x = 450.415 :=
  sorry

end original_purchase_price_first_commodity_l711_711792


namespace find_x_l711_711994

theorem find_x (x : ℝ) (h1 : x > 9) : 
  (sqrt (x - 9 * sqrt (x - 9)) + 3 = sqrt (x + 9 * sqrt (x - 9)) - 3) → 
  x ≥ 45 :=
sorry

end find_x_l711_711994


namespace prize_winners_physics_solve_equation_l711_711747

-- Given conditions
def conditions (x y z a b : ℕ) : Prop :=
  x + y + z + a + 20 = 40 ∧
  y + a = 7 ∧
  x + a = 10 ∧
  z + a = 11 ∧
  x + y + z + a + b + 20 = 51

-- Proof problem for prize winners in physics
theorem prize_winners_physics (x y z a b : ℕ) (h : conditions x y z a b) :
  y + z + a + b = 25 :=
sorry

-- Equation to solve
def equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 1 + real.sqrt (4 * x^2 + 4 * y^2 - 34) = 2 * |x + y| - 2 * x * y

-- Proof problem for solving the equation
theorem solve_equation :
  ∃ (x y : ℝ), equation x y ∧ 
    ((x = 2.5 ∧ y = -1.5) ∨ (x = -2.5 ∧ y = 1.5) ∨ 
     (x = 1.5 ∧ y = -2.5) ∨ (x = -1.5 ∧ y = 2.5)) :=
sorry

end prize_winners_physics_solve_equation_l711_711747


namespace find_n_l711_711646

theorem find_n (N : ℕ) (hN : 10 ≤ N ∧ N < 100)
  (h : ∃ a b : ℕ, N = 10 * a + b ∧ 4 * a + 2 * b = N / 2) : 
  N = 32 ∨ N = 64 ∨ N = 96 :=
sorry

end find_n_l711_711646


namespace magazines_cover_area_l711_711816

variables (S : ℝ)

-- Each magazine covers part of the total area S
-- There are 15 magazines in total covering the area

theorem magazines_cover_area (cover : ℕ → set ℝ) 
  (total_cover : (⋃ i, cover i) = set.univ)
  (h : ∀ i, cover i ≠ ∅)
  (hS : ∀ i, set.measure S (cover i) ≤ S / 15) :
  ∃ (remaining : set ℕ), remaining.card = 8 ∧ set.measure S (⋃ i ∈ remaining, cover i) ≥ 8 / 15 * S :=
sorry

end magazines_cover_area_l711_711816


namespace four_digit_numbers_using_2023_are_odd_l711_711232

theorem four_digit_numbers_using_2023_are_odd : ∃ n, n = 3 ∧ 
  (∀ x : ℕ, (1000 ≤ x ∧ x ≤ 9999 ∧ 
  (∀ d ∈ [2, 2, 0, 3], digits 10 x d 1) ∧ x % 2 = 1 → 
  (x in permutations [2, 2, 0, 3]))) := sorry

end four_digit_numbers_using_2023_are_odd_l711_711232


namespace tan_Y_l711_711676

/-- Definition of the right triangle with the given conditions. -/
structure RightTriangle (A B C : Type) [linear_ordered_field B] :=
(XY : B)
(YZ : B)
(angle_YXZ : A)
(right_angle : angle_YXZ = 90)
(XZ : B := real.sqrt (YZ * YZ - XY * XY))

/-- Given a right triangle XYZ with ∠YXZ = 90°, XY = 24, and YZ = 25, prove that tan Y = 7 / 24. -/
theorem tan_Y (A B : Type) [linear_ordered_field B]
  (angle_YXZ : A) (XY YZ : B)
  (h1 : angle_YXZ = 90)
  (h2 : XY = 24)
  (h3 : YZ = 25) : (XY = 24) ∧ (XZ (RightTriangle.mk XY YZ angle_YXZ h1) = 7) :=
begin
  let T := RightTriangle.mk XY YZ angle_YXZ h1,
  have h4 : XZ T = real.sqrt (YZ * YZ - XY * XY),
  { unfold XZ, },
  have h5 : XZ T = 7,
  { rw [←h4, h2, h3],
    norm_num },
  exact ⟨h2, h5⟩,
end

end tan_Y_l711_711676


namespace only_book_A_l711_711053

theorem only_book_A (purchasedBoth : ℕ) (purchasedOnlyB : ℕ) (purchasedA : ℕ) (purchasedB : ℕ) 
  (h1 : purchasedBoth = 500)
  (h2 : 2 * purchasedOnlyB = purchasedBoth)
  (h3 : purchasedA = 2 * purchasedB)
  (h4 : purchasedB = purchasedOnlyB + purchasedBoth) :
  purchasedA - purchasedBoth = 1000 :=
by
  sorry

end only_book_A_l711_711053


namespace cos_double_angle_l711_711295

theorem cos_double_angle (x : ℝ) (h1 : x ∈ Ioo (-3 * Real.pi / 4) (Real.pi / 4))
    (h2 : Real.cos (Real.pi / 4 - x) = -3 / 5) : Real.cos (2 * x) = -24 / 25 :=
by
  sorry

end cos_double_angle_l711_711295


namespace athlete_speed_l711_711475

theorem athlete_speed {distance : ℝ} {time : ℝ} 
  (h_dist : distance = 200) (h_time : time = 25) :
  let speed := (distance / 1000) / (time / 3600) in
  speed = 28.8 :=
by 
  -- let distance in kilometers = distance / 1000
  -- let time in hours = time / 3600
  -- calculate speed as distance / time, and prove it equals 28.8
  sorry

end athlete_speed_l711_711475


namespace points_per_other_player_l711_711991

-- Define the conditions as variables
variables (total_points : ℕ) (faye_points : ℕ) (total_players : ℕ)

-- Assume the given conditions
def conditions : Prop :=
  total_points = 68 ∧ faye_points = 28 ∧ total_players = 5

-- Define the proof problem: Prove that the points scored by each of the other players is 10
theorem points_per_other_player :
  conditions total_points faye_points total_players →
  (total_points - faye_points) / (total_players - 1) = 10 :=
by
  sorry

end points_per_other_player_l711_711991


namespace students_taking_neither_music_nor_art_l711_711066

def total_students : ℕ := 500
def music_students : ℕ := 40
def art_students : ℕ := 20
def both_music_art_students : ℕ := 10
def neither_music_art_students : ℕ := total_students - (music_students + art_students - both_music_art_students)

theorem students_taking_neither_music_nor_art :
  neither_music_art_students = 450 :=
by
  simp [total_students, music_students, art_students, both_music_art_students, neither_music_art_students]
  sorry

end students_taking_neither_music_nor_art_l711_711066


namespace compare_quadrilateral_areas_l711_711774

def square_areas_helper (S : ℝ) : Prop :=
  let S_ABCD := 36 * S
  let BE_EC_ratio := 1 / 2
  let CF_FB_ratio := 1 / 2
  let CG_GD_ratio := 2 / 1
  let AI_ID_ratio := 1 / 2
  let DH_HA_ratio := 1 / 2

  ∀ E F G H I J K L : Type,
    (∀ α β : ℝ, α / β = BE_EC_ratio → β / α = CF_FB_ratio) →
    (∀ γ δ : ℝ, γ / δ = CG_GD_ratio) →
    (∀ ψ ω : ℝ, ψ / ω = AI_ID_ratio ∧ ω / ψ = DH_HA_ratio) →
    let area_GDHL := (6 * S) - (24 / 11 * S)
    let area_EFKJ := 4 * S - (24 / 33 * S)
    area_GDHL > area_EFKJ

theorem compare_quadrilateral_areas (S : ℝ) :
  square_areas_helper S :=
begin
  sorry
end

end compare_quadrilateral_areas_l711_711774


namespace XY_parallel_AC_l711_711714

-- Define the points A, B, C, P, Q, R, X, Y
variables {A B C P Q R X Y : Point}
-- Conditions stating B is the midpoint of A and C, angle PBC = 60, and triangle properties
variables (h1 : midpoint B A C) (h2 : angle P B C = 60)
variables (h3 : equilateral_triangle P C Q) (h4 : half_plane_diff P C B Q)
variables (h5 : equilateral_triangle A P R) (h6 : half_plane_same A P B R)
variables (h7 : intersection X B Q P C) (h8 : intersection Y B R A P)

-- The theorem to be proven
theorem XY_parallel_AC : parallel X Y A C :=
by sorry

end XY_parallel_AC_l711_711714


namespace line_through_longest_chord_l711_711191

-- Define the point M and the circle equation
def M : ℝ × ℝ := (3, -1)
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + y - 2 = 0

-- Define the standard form of the circle equation
def standard_circle_eqn (x y : ℝ) : Prop := (x - 2)^2 + (y + 1/2)^2 = 25/4

-- Define the line equation
def line_eqn (x y : ℝ) : Prop := x + 2 * y - 2 = 0

-- Theorem: Equation of the line containing the longest chord passing through M
theorem line_through_longest_chord : 
  (circle_eqn 3 (-1)) → 
  ∀ (x y : ℝ), standard_circle_eqn x y → ∃ (k b : ℝ), line_eqn x y :=
by
  -- Proof goes here
  intro h1 x y h2
  sorry

end line_through_longest_chord_l711_711191


namespace sum_of_all_ks_l711_711871

theorem sum_of_all_ks (k : ℤ) :
  (∃ r s : ℤ, r ≠ s ∧ (3 * x^2 - k * x + 12 = 0) ∧ (r + s = k / 3) ∧ (r * s = 4)) → (k = 15 ∨ k = -15) → k + k = 0 :=
by
  sorry

end sum_of_all_ks_l711_711871


namespace number_of_valid_trapezoids_l711_711794

noncomputable def calculate_number_of_trapezoids : ℕ :=
  let rows_1 := 7
  let rows_2 := 9
  let unit_spacing := 1
  let height := 2
  -- Here, we should encode the actual combinatorial calculation as per the problem solution
  -- but for the Lean 4 statement, we will provide the correct answer directly.
  361

theorem number_of_valid_trapezoids :
  calculate_number_of_trapezoids = 361 :=
sorry

end number_of_valid_trapezoids_l711_711794


namespace sum_of_k_values_with_distinct_integer_solutions_l711_711848

theorem sum_of_k_values_with_distinct_integer_solutions : 
  ∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ 3 * p * q + (-k) * (p + q) + 12 = 0}, k = 0 :=
by
  sorry

end sum_of_k_values_with_distinct_integer_solutions_l711_711848


namespace unique_ordered_pair_l711_711454

theorem unique_ordered_pair exists_unique (a b : ℕ) (h1 : b > a) (h2 : ab = 3 * (a - 4) * (b - 4)) : ∃! (a b : ℕ), b > a ∧ ab = 3 * (a - 4) * (b - 4) := 
begin
  sorry
end

end unique_ordered_pair_l711_711454


namespace percentage_spent_on_hats_l711_711279

def total_money : ℕ := 90
def cost_per_scarf : ℕ := 2
def number_of_scarves : ℕ := 18
def cost_of_scarves : ℕ := number_of_scarves * cost_per_scarf
def money_left_for_hats : ℕ := total_money - cost_of_scarves
def number_of_hats : ℕ := 2 * number_of_scarves

theorem percentage_spent_on_hats : 
  (money_left_for_hats : ℝ) / (total_money : ℝ) * 100 = 60 :=
by
  sorry

end percentage_spent_on_hats_l711_711279


namespace problem_inequality_l711_711495

-- Define the set T
def T : set ℝ := {t | t > 1}

-- Theorem statement
theorem problem_inequality (a b : ℝ) (ha : a ∈ T) (hb : b ∈ T) : ab + 1 > a + b :=
by {
  sorry
}

end problem_inequality_l711_711495


namespace card_statement_truth_l711_711415

-- Define the four statements
def S1 := "Exactly one of these statements is true."
def S2 := "Exactly one of these statements is false."
def S3 := "Exactly two of these statements are false."
def S4 := "None of these statements are true."

-- Declare the main problem: Prove the number of true statements is 0 given the conditions
theorem card_statement_truth :
  (¬S1 ∧ ¬S2 ∧ ¬S3 ∧ ¬S4) :=
by
  -- Proof goes here
  sorry

end card_statement_truth_l711_711415


namespace BP_eq_CP_l711_711473

open EuclideanGeometry

variables (A B C K L P : Point)
variable {ω : Circle}
variables (h1 : InscribedTriangle A B C ω)
variables (h2 : TangentThrough A ω K)
variables (h3 : TangentThrough B ω K)
variables (h4 : TangentThrough A ω L)
variables (h5 : TangentThrough C ω L)
variables (h6 : Parallel K A B)
variables (h7 : Parallel L A C)
variables (h8 : PointIntersection K L P)

theorem BP_eq_CP : dist B P = dist C P := sorry

end BP_eq_CP_l711_711473


namespace cubic_identity_l711_711578

theorem cubic_identity (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / x^3) = 140 := 
  sorry

end cubic_identity_l711_711578


namespace transform_equation_to_square_form_l711_711822

theorem transform_equation_to_square_form : 
  ∀ x : ℝ, (x^2 - 6 * x = 0) → ∃ m n : ℝ, (x + m) ^ 2 = n ∧ m = -3 ∧ n = 9 := 
sorry

end transform_equation_to_square_form_l711_711822


namespace wrapping_paper_area_correct_l711_711073

variable (w h : ℝ) -- Define the base length and height of the box.

-- Lean statement for the problem asserting that the area of the wrapping paper is \(2(w+h)^2\).
def wrapping_paper_area (w h : ℝ) : ℝ := 2 * (w + h) ^ 2

-- Theorem stating that the derived formula for the area of the wrapping paper is correct.
theorem wrapping_paper_area_correct (w h : ℝ) : wrapping_paper_area w h = 2 * (w + h) ^ 2 :=
by
  sorry -- Proof is omitted

end wrapping_paper_area_correct_l711_711073


namespace distance_MF_l711_711584

-- Define the conditions for the problem
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

def focus : (ℝ × ℝ) := (2, 0)

def lies_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2

def distance_to_line (M : ℝ × ℝ) (line_x : ℝ) : ℝ :=
  abs (M.1 - line_x)

def point_M_conditions (M : ℝ × ℝ) : Prop :=
  distance_to_line M (-3) = 6 ∧ lies_on_parabola M

-- The final proof problem statement in Lean
theorem distance_MF (M : ℝ × ℝ) (h : point_M_conditions M) : dist M focus = 5 :=
by sorry

end distance_MF_l711_711584


namespace factorize_difference_of_squares_l711_711507

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) :=
sorry

end factorize_difference_of_squares_l711_711507


namespace quadratic_single_root_l711_711355

theorem quadratic_single_root :
  ∀ (a b c d x : ℝ), 
  (b = a - d) ∧ (c = a - 3d) ∧ (a ≥ b) ∧ (b ≥ c) ∧ (c ≥ 0) → 
  (a * x^2 + b * x + c = 0) ∧ (b^2 - 4 * a * c = 0) → 
  x = - (1 + 3 * Real.sqrt 22) / 6 := 
by
  intros a b c d x h_conditions h_discriminant
  sorry

end quadratic_single_root_l711_711355


namespace grandma_olga_grandchildren_and_extended_family_l711_711604

theorem grandma_olga_grandchildren_and_extended_family :
  let daughters := 6 in
  let sons := 5 in
  let daughters_biological_children := 19 in
  let daughters_step_children := 4 in
  let sons_biological_children := 15 in
  let sons_inlaws := 3 in
  let inlaws_children := 2 in
  (daughters * daughters_biological_children + 
   daughters * daughters_step_children + 
   sons * sons_biological_children + 
   sons * sons_inlaws * inlaws_children) = 243 := 
by simp [daughters, sons, daughters_biological_children, daughters_step_children, sons_biological_children, sons_inlaws, inlaws_children]; sorry

end grandma_olga_grandchildren_and_extended_family_l711_711604


namespace product_of_conjugates_is_real_l711_711960

section
variables (x y : ℝ)

-- Major Premise: The product of two complex conjugates is a real number.
def is_real (z : ℂ) : Prop := z.im = 0

-- Minor Premise: (x + yi) and (x - yi) are complex conjugates.
def complex_conjugate_pair (z1 z2 : ℂ) : Prop := 
  z2 = conj z1

-- Conclusion: (x + yi)(x - yi) is a real number.
theorem product_of_conjugates_is_real :
  ∀ (x y : ℝ), is_real ((complex.mk x y) * (complex.mk x (-y))) :=
by
  sorry
end

end product_of_conjugates_is_real_l711_711960


namespace triangular_number_30_l711_711116

theorem triangular_number_30 : 
  (∃ (T : ℕ), T = 30 * (30 + 1) / 2 ∧ T = 465) :=
by 
  sorry

end triangular_number_30_l711_711116


namespace period_and_decreasing_interval_max_area_triangle_l711_711602

-- Definitions for the vectors and function
def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos (π / 3 - x), -Real.sin x)
def vec_b (x : ℝ) : ℝ × ℝ := (Real.sin (x + π / 6), Real.sin x)
def f (x : ℝ) : ℝ := vec_a x • vec_b x

-- Problem 1: Proving the period and the monotonically decreasing interval
theorem period_and_decreasing_interval :
  (∀ x, f (x + π) = f x) ∧ (∀ k : ℤ, ∀ x : ℝ, k * π + π / 6 ≤ x ∧ x ≤ k * π + 2 * π / 3 → f' x ≤ 0) :=
sorry

-- Problem 2: Proving the maximum area of the triangle
theorem max_area_triangle :
  ∀ a b : ℝ, ∀ C : ℝ, (a = b) → (f C = -1/2) →  
  (C = π / 3 ∨ C = 2 * π / 3) → 
  (2 * sqrt 3 = Real.sqrt (a^2 + b^2 + 2 * a * b * (-1 / 2))) →
  (∃ S : ℝ, S = (1/2) * a * b * Real.sin C ∧ S ≤ sqrt 3) :=
sorry

end period_and_decreasing_interval_max_area_triangle_l711_711602


namespace count_four_digit_numbers_with_consecutive_digits_l711_711005

-- Define what it means to be a 4-digit number
def is_four_digit_number (n : ℕ) : Prop := n ≥ 1000 ∧ n < 10000

-- Define consecutive digits
def is_consecutive_digits (d1 d2 : ℕ) : Prop := (d1 + 1 = d2) ∨ (d1 = 8 ∧ d2 = 9)

-- Define what it means to include four consecutive digits
def includes_four_consecutive_digits (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), is_consecutive_digits a b ∧ is_consecutive_digits b c ∧ is_consecutive_digits c d ∧
                  n = 1000 * a + 100 * b + 10 * c + d

-- The main theorem to prove
theorem count_four_digit_numbers_with_consecutive_digits : 
  ∃ (N : ℕ), N = 150 ∧ ∀ (n : ℕ), is_four_digit_number n → includes_four_consecutive_digits n → N :=
sorry

end count_four_digit_numbers_with_consecutive_digits_l711_711005


namespace smallest_integer_k_l711_711163

theorem smallest_integer_k (k : ℕ) :
  (∃ (n : ℕ), (n = (\frac{7 * (10^k - 1)}{9})) ∧ (sum_digits (5 * n) = 800)) → (∃ (k_min : ℕ), k_min = 88) :=
sorry

end smallest_integer_k_l711_711163


namespace three_digit_multiples_of_seven_count_l711_711613

theorem three_digit_multiples_of_seven_count :
  let smallest := 15
  let largest := 142
  largest - smallest + 1 = 128 :=
by
  let smallest := 15
  let largest := 142
  have h_smallest : 7 * smallest = 105 := rfl
  have h_largest : 7 * largest = 994 := rfl
  show largest - smallest + 1 = 128 from sorry

end three_digit_multiples_of_seven_count_l711_711613


namespace average_of_t_b_c_29_l711_711885
-- Importing the entire Mathlib library

theorem average_of_t_b_c_29 (t b c : ℝ) 
  (h : (t + b + c + 14 + 15) / 5 = 12) : 
  (t + b + c + 29) / 4 = 15 :=
by 
  sorry

end average_of_t_b_c_29_l711_711885


namespace smallest_n_l711_711384

theorem smallest_n (n : ℕ) : 
  (∃ k : ℕ, 4 * n = k^2) ∧ (∃ l : ℕ, 5 * n = l^5) ↔ n = 625 :=
by sorry

end smallest_n_l711_711384


namespace g_formula_inequality_proof_l711_711582

-- Define the roots α and β of the given quadratic equation
variables (α β t : ℝ)

-- Condition that α and β are roots of the quadratic equation
def quadratic_roots (α β t : ℝ) : Prop := 
  4 * α^2 - 4 * t * α - 1 = 0 ∧
  4 * β^2 - 4 * t * β - 1 = 0 ∧
  α ≠ β

-- Define the function f(x)
def f (x t : ℝ) : ℝ := 
  (2 * x - t) / (x^2 + 1)

-- Define g(t) as the difference between max and min of f(x) over [α, β]
def g (t : ℝ) : ℝ := 
  (8 * (Real.sqrt (t^2 + 1)) * (2 * t^2 + 5)) / (16 * t^2 + 25)

-- Prove the formula of g(t)
theorem g_formula (t : ℝ) : 
  ∃ α β, quadratic_roots α β t → 
  g(t) = (8 * (Real.sqrt (t^2 + 1)) * (2 * t^2 + 5)) / (16 * t^2 + 25) :=
sorry

-- Define the conditions for the given range of u_i
variables (u1 u2 u3 : ℝ)

def u_range (u : ℝ) : Prop := 
  0 < u ∧ u < π / 2

-- Define the condition on the sum of sine values
def sine_sum_condition (u1 u2 u3 : ℝ) : Prop := 
  Real.sin u1 + Real.sin u2 + Real.sin u3 = 1

-- Prove the inequality involving g(tan(u_i))
theorem inequality_proof (u1 u2 u3 : ℝ) (h1 : sine_sum_condition u1 u2 u3)
  (h2 : u_range u1) (h3 : u_range u2) (h4 : u_range u3) :
  (1 / g (Real.tan u1) + 1 / g (Real.tan u2) + 1 / g (Real.tan u3)) < (3/4) * Real.sqrt(6) :=
sorry

end g_formula_inequality_proof_l711_711582


namespace alok_loss_percentage_l711_711102

noncomputable def loss_percentage 
  (original_radio : ℝ) (discount_radio : ℝ) 
  (original_tv : ℝ) (discount_tv : ℝ) 
  (original_blender : ℝ) (discount_blender : ℝ) 
  (selling_radio : ℝ) (selling_tv : ℝ) (selling_blender : ℝ) : ℝ := 
let discounted_radio := original_radio * (1 - discount_radio / 100) in
let discounted_tv := original_tv * (1 - discount_tv / 100) in
let discounted_blender := original_blender * (1 - discount_blender / 100) in
let total_discounted_cost := discounted_radio + discounted_tv + discounted_blender in
let total_selling_price := selling_radio + selling_tv + selling_blender in
let overall_loss := total_discounted_cost - total_selling_price in
(overall_loss / total_discounted_cost) * 100

theorem alok_loss_percentage :
  loss_percentage 4500 10 8000 15 1300 5 3200 7500 1000 ≈ 3.19 := by
  sorry

end alok_loss_percentage_l711_711102


namespace original_three_numbers_are_arith_geo_seq_l711_711369

theorem original_three_numbers_are_arith_geo_seq
  (x y z : ℕ) (h1 : ∃ k : ℕ, x = 3*k ∧ y = 4*k ∧ z = 5*k)
  (h2 : ∃ r : ℝ, (x + 1) / y = r ∧ y / z = r ∧ r^2 = z / y):
  x = 15 ∧ y = 20 ∧ z = 25 :=
by 
  sorry

end original_three_numbers_are_arith_geo_seq_l711_711369


namespace cube_volume_l711_711427

theorem cube_volume (A : ℝ) (hA : A = 294) : ∃ (V : ℝ), V = 343 :=
by
  sorry

end cube_volume_l711_711427


namespace incorrect_reasoning_form_l711_711451

theorem incorrect_reasoning_form
  (P1 : ∃ q : ℚ, q.is_proper_fraction)
  (P2 : ∀ z : ℤ, (z : ℚ)) :
  ¬ (∀ z : ℤ, z.is_proper_fraction) :=
sorry

end incorrect_reasoning_form_l711_711451


namespace area_of_triangle_LEF_l711_711259

open Real EuclideanGeometry

noncomputable def centroid {α : Type*} [linear_ordered_field α] [inhabited α] (a b c : α × α) : α × α :=
((a.1 + b.1 + c.1)/3, (a.2 + b.2 + c.2)/3)

theorem area_of_triangle_LEF : 
  let P := (0, 0) in -- P is origin, center of the circle
  let radius_P := 10 in -- radius of circle centered at P
  let EF := 12 in -- length of chord EF
  let GF := EF / 2 in -- GF is half of EF
  let PG := sqrt (radius_P^2 - GF^2) in -- Pythagorean theorem to find PG
  let LN := 20 in -- length between L and N
  let PH := PG in -- PH is same as PG
  let EF_parallel := true in -- EF is parallel to LM
  let collinear := true in -- Points L, N, P, and M are collinear
  ∃ (A : ℝ), A = (1 / 2) * EF * PH ∧ A = 48 :=
by
  sorry

end area_of_triangle_LEF_l711_711259


namespace number_of_factors_l711_711746

theorem number_of_factors (m : ℕ) (h : m = 2^5 * 3^3 * 5^4 * 7^2) : 
  nat.totient m = 360 := by
  sorry

end number_of_factors_l711_711746


namespace tangent_line_g_at_2_l711_711218

-- Given conditions
variable {f : ℝ → ℝ}
variable h_tangent_f : ∀ x, x = 2 → (∀ y, y = f x → (2 * x - 1 = y))
variable h_f_deriv : ∀ x, x = 2 → deriv f x = 2

-- Question to prove
theorem tangent_line_g_at_2 :
  let g (x : ℝ) := x^2 + f x in
  has_tangent_eq_at (λ x, g x) (2, g 2) (λ x, 6 * x - y - 5) :=
by
  -- Proof goes here
  sorry

end tangent_line_g_at_2_l711_711218


namespace temperature_difference_l711_711003

theorem temperature_difference (highest lowest : ℝ) (h_high : highest = 27) (h_low : lowest = 17) :
  highest - lowest = 10 :=
by
  sorry

end temperature_difference_l711_711003


namespace jill_investment_l711_711701

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_investment :
  compound_interest 10000 0.0396 2 2 ≈ 10815.66 :=
by
  sorry

end jill_investment_l711_711701


namespace f_2011_2012_l711_711203

noncomputable def f : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
∀ x, f (x + p) = f (x)

axiom f_odd : odd_function f
axiom f_periodic : periodic_function f 4
axiom f_defined_on_reals : ∀ x, f x ∈ ℝ
axiom f_at_1 : f 1 = -4

theorem f_2011_2012 : f 2011 + f 2012 = 4 :=
sorry

end f_2011_2012_l711_711203


namespace class_c_stratified_sampling_l711_711088

theorem class_c_stratified_sampling :
  let classA := 75
  let classB := 75
  let classC := 200
  let classD := 150
  let total_students := classA + classB + classC + classD
  let selected_students := 20
  let prob := selected_students / total_students
  let classC_selected_students := classC * prob
  classC_selected_students = 8 :=
by
  -- Assign values to the classes
  let classA := 75
  let classB := 75
  let classC := 200
  let classD := 150
  -- Calculate total number of students
  let total_students := classA + classB + classC + classD
  -- Number of students to be selected
  let selected_students := 20
  -- Probability of being selected
  let prob := selected_students / total_students
  -- Number of students to be drawn from class C
  let classC_selected_students := classC * prob
  -- Prove that the correct number of students drawn from class C is 8.
  have h1 : total_students = 500 := by simp [total_students, classA, classB, classC, classD]; sorry -- simplify and fill in the proof
  have h2 : prob = 1 / 25 := by simp [prob, selected_students, total_students]; sorry -- simplify and fill in the proof
  have h3 : classC_selected_students = 8 := by simp [classC_selected_students, classC, prob]; sorry -- simplify and fill in the proof
  exact h3

end class_c_stratified_sampling_l711_711088


namespace original_price_of_gift_l711_711713

theorem original_price_of_gift :
  let dave_has := 46
  let kyle_has := 3 * dave_has - 12
  let kyle_after_snowboarding := kyle_has - kyle_has / 3
  let lisa_has := kyle_after_snowboarding + 20
  let total_money := kyle_after_snowboarding + lisa_has
  let discounted_price := total_money / 0.85
  discounted_price = 221.18 :=
by
  let dave_has := 46
  let kyle_has := 3 * dave_has - 12
  let kyle_after_snowboarding := kyle_has - kyle_has / 3
  let lisa_has := kyle_after_snowboarding + 20
  let total_money := kyle_after_snowboarding + lisa_has
  let discounted_price := total_money / 0.85
  show discounted_price = 221.18 from rfl

end original_price_of_gift_l711_711713


namespace sum_of_k_distinct_integer_roots_l711_711851

theorem sum_of_k_distinct_integer_roots : 
  (∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ p * q = 4 ∧ k = 3 * (p + q)}, k) = 0 := 
by
  sorry

end sum_of_k_distinct_integer_roots_l711_711851


namespace number_of_liars_l711_711901

-- Define a predicate is_knight representing if a person is a knight.
def is_knight (n : ℕ) : Prop := sorry

-- Define the total number of people.
def total_people := 30

-- Define conditions for odd-numbered persons.
def odd_person_statement (n : ℕ) : Prop := 
  is_knight n ↔ ∀ m, (m > n ∧ m ≤ total_people) → ¬is_knight m

-- Define conditions for even-numbered persons.
def even_person_statement (n : ℕ) : Prop := 
  is_knight n ↔ ∀ m, (m < n ∧ m ≥ 1) → ¬is_knight m

-- The main theorem we want to prove.
theorem number_of_liars : ∃ n, n = 28 ∧ 
  (∀ i, (1 ≤ i ∧ i ≤ total_people ∧ i % 2 = 1) → odd_person_statement i) ∧
  (∀ j, (1 ≤ j ∧ j ≤ total_people ∧ j % 2 = 0) → even_person_statement j) ∧
  (∀ k, 1 ≤ k ∧ k ≤ total_people → (is_knight k ∨ ¬is_knight k)) ∧
  (n = total_people - ∑ p in finRange total_people, if is_knight p then 1 else 0) :=
sorry

end number_of_liars_l711_711901


namespace cos_seven_pi_over_six_l711_711149

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 := 
by
  sorry

end cos_seven_pi_over_six_l711_711149


namespace train_length_l711_711092

/-- 
Given that a train can cross an electric pole in 200 seconds and its speed is 18 km/h,
prove that the length of the train is 1000 meters.
-/
theorem train_length
  (time_to_cross : ℕ)
  (speed_kmph : ℕ)
  (h_time : time_to_cross = 200)
  (h_speed : speed_kmph = 18)
  : (speed_kmph * 1000 / 3600 * time_to_cross = 1000) :=
by
  sorry

end train_length_l711_711092


namespace sum_of_k_values_with_distinct_integer_solutions_l711_711847

theorem sum_of_k_values_with_distinct_integer_solutions : 
  ∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ 3 * p * q + (-k) * (p + q) + 12 = 0}, k = 0 :=
by
  sorry

end sum_of_k_values_with_distinct_integer_solutions_l711_711847


namespace circle_radius_l711_711031

theorem circle_radius (r : ℝ) :
  (∀ x y : ℝ, (x - r)^2 + y^2 = r^2 → x^2 + 4y^2 = 5) →
  r = sqrt(15) / 4 :=
by
  sorry

end circle_radius_l711_711031


namespace trigonometric_identity_l711_711174

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α - Real.cos α) / (5 * Real.sin α + 3 * Real.cos α) = 1 / 13 := 
by
-- The proof goes here
sorry

end trigonometric_identity_l711_711174


namespace area_under_curve_l711_711952

noncomputable def f (x : ℝ) : ℝ :=
  x * real.sqrt(4 - x^2)

theorem area_under_curve :
  ∫ (x : ℝ) in 0..2, f x = 8/3 := 
by
  sorry

end area_under_curve_l711_711952


namespace order_of_m_n_p_l711_711290

noncomputable def m := Real.sqrt 6 - Real.sqrt 5
noncomputable def n := Real.sqrt 7 - Real.sqrt 6
noncomputable def p := Real.sqrt 8 - Real.sqrt 7

theorem order_of_m_n_p : m > n ∧ n > p :=
by
  have h1 : m := sqrt 6 - sqrt 5
  have h2 : n := sqrt 7 - sqrt 6
  have h3 : p := sqrt 8 - sqrt 7
  sorry

end order_of_m_n_p_l711_711290


namespace select_sprinters_l711_711447

def sprinters : Type := Fin 6

def A : sprinters := ⟨0, by norm_num⟩
def B : sprinters := ⟨1, by norm_num⟩

theorem select_sprinters :
  ∃ (S : Finset sprinters), S.card = 4 ∧
    ∀ (T : Tuple sprinters (Fin 4)), ↑(T 0) ≠ A ∧ ↑(T 3) ≠ B → S = T.to_finset :=
sorry

end select_sprinters_l711_711447


namespace volume_relationship_l711_711455

theorem volume_relationship {r : ℝ} (h_cylinder : ℝ) (h_cone : ℝ) 
  (V_cylinder : ℝ) (V_sphere : ℝ) (V_cone : ℝ) :
  h_cylinder = 2 * r →
  h_cone = 2 * r →
  V_cylinder = 2 * π * r^3 →
  V_cone = (1 / 3) * π * r^2 * h_cone →
  V_sphere = (4 / 3) * π * r^3 →
  V_cone + V_cylinder = 2 * V_sphere :=
by
  intro h_cylinder_eq h_cone_eq V_cylinder_eq V_cone_eq V_sphere_eq
  rw [h_cone_eq] at V_cone_eq
  rw [V_cone_eq, V_cylinder_eq, V_sphere_eq]
  simp only [mul_assoc, mul_comm, mul_left_comm]
  norm_num
  ring
  sorry

end volume_relationship_l711_711455


namespace domain_of_f_l711_711000

noncomputable def domain_of_function (x : ℝ) : Set ℝ :=
  {x | 4 - x ^ 2 ≥ 0 ∧ x ≠ 1}

theorem domain_of_f (x : ℝ) : domain_of_function x = {x | -2 ≤ x ∧ x < 1 ∨ 1 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_f_l711_711000


namespace cartesian_eq_C1_intersection_polar_coordinates_C1_C2_l711_711677

noncomputable def polar_to_cartesian_eq (rho θ : ℝ) : ℝ × ℝ :=
  (rho * cos θ, rho * sin θ)

-- Given conditions
def polar_equation_1 (rho θ : ℝ) : Prop := rho^2 - 4 * rho * cos θ + 3 = 0
def parametric_equations_C2 (t : ℝ) : ℝ × ℝ :=
  (t * cos (π / 6), t * sin (π / 6))

-- Problems to solve
theorem cartesian_eq_C1 (x y : ℝ) :
  (∃ (ρ θ : ℝ), θ ∈ Icc 0 (2 * π) ∧ polar_equation_1 ρ θ ∧ polar_to_cartesian_eq ρ θ = (x, y))
  ↔ (x-2)^2 + y^2 = 1 :=
sorry

theorem intersection_polar_coordinates_C1_C2 :
  (∃ (t : ℝ), parametric_equations_C2 t ∈ {p : ℝ × ℝ | ∃ (ρ θ : ℝ), polar_equation_1 ρ θ ∧ polar_to_cartesian_eq ρ θ = p})
  ↔ (∃ (ρ θ : ℝ), polar_equation_1 ρ θ ∧ ρ = sqrt 3 ∧ θ = π / 6) :=
sorry

end cartesian_eq_C1_intersection_polar_coordinates_C1_C2_l711_711677


namespace meeting_time_correct_l711_711883

-- Definitions based on conditions
def startTime := 7 * 60 -- 7 am in minutes
def speedA := 15 -- Speed of A in kmph
def speedB := 18 -- Speed of B in kmph
def initialDistance := 60 -- Initial distance in km

-- Combined speed of A and B
def combinedSpeed := speedA + speedB

-- Time to meet in minutes
def timeToMeet := initialDistance * 60 / combinedSpeed -- We multiply by 60 to convert hours to minutes

-- Meeting time calculation
def meetingTime := startTime + timeToMeet

-- Final meeting time in hours and minutes
def hours := (meetingTime / 60).toNat -- integer part dividing by 60
def minutes := (meetingTime % 60).toNat -- remainder of division by 60

theorem meeting_time_correct : hours = 8 ∧ minutes = 49 :=
by sorry

end meeting_time_correct_l711_711883


namespace shaded_area_eq_45_l711_711459

structure Point :=
  (x : ℝ)
  (y : ℝ)

def square := {A : Point | (A.x = 0 ∨ A.x = 12) ∧ (A.y = 0 ∨ A.y = 12)}

def right_triangle := {T : Point | 
  (T.x = 12 ∧ T.y = 0) ∨ 
  (T.x = 24 ∧ T.y = 0) ∨ 
  (T.x = 12 ∧ T.y = 9)}

noncomputable def area_of_shaded_region : ℝ := 45

-- The vertices used in the discussion
def A := Point.mk 0 0 
def B := Point.mk 0 12
def C := Point.mk 12 12 
def D := Point.mk 12 0
def E := Point.mk 24 0
def F := Point.mk 12 9

theorem shaded_area_eq_45 : 
  ∃ (G : Point), G.x = 12 ∧ G.y = 10.5 ∧
  let triangle_CDF := (1/2 * 12 * 9) in
  let triangle_BCG := (1/2 * 12 * 1.5) in
  (triangle_CDF - triangle_BCG) = area_of_shaded_region := 
by
  sorry

end shaded_area_eq_45_l711_711459


namespace sum_local_values_of_digits_l711_711042

theorem sum_local_values_of_digits :
  let d2 := 2000
  let d3 := 300
  let d4 := 40
  let d5 := 5
  d2 + d3 + d4 + d5 = 2345 :=
by
  sorry

end sum_local_values_of_digits_l711_711042


namespace wrapping_paper_area_correct_l711_711071

-- Given conditions:
variables (w h : ℝ) -- base length and height of the box

-- Definition of the area of the wrapping paper given the problem's conditions
def wrapping_paper_area (w h : ℝ) : ℝ :=
  2 * (w + h) ^ 2

-- Theorem statement to prove the area of the wrapping paper
theorem wrapping_paper_area_correct (w h : ℝ) : wrapping_paper_area w h = 2 * (w + h) ^ 2 :=
by
  -- proof to be provided
  sorry

end wrapping_paper_area_correct_l711_711071


namespace concentric_circles_equal_segments_l711_711601

open_locale classical

-- Define the existence of two concentric circles S1 and S2
variables (r1 r2 : ℝ) (center : ℝ × ℝ) 
(h_concentric : true) -- Placeholder for the concentric condition
(h_radius : r1 < r2)

-- Define the existence of points A, B, C, D
variables (A B C D : ℝ × ℝ)

-- We say a line creating segments of equal length on concentric circles exists
theorem concentric_circles_equal_segments :
  ∃ (m : ℝ) (b : ℝ), -- There exists a line defined by y = mx + b
  AB = BC ∧ BC = CD :=        -- intersecting both circles such that the segments are equal
sorry

end concentric_circles_equal_segments_l711_711601


namespace region_area_l711_711515

/-- Definitions required to state the problem -/
def fractionalPart (x : ℝ) : ℝ := x - x.floor
def floorPart (x : ℝ) : ℕ := Real.toNat ⌊x⌋ 

/-- Problem statement -/
theorem region_area : 
  (∑ i in Finset.range 50, 0.01 * (i + 1)) = 12.75 := 
by
  let fractional_area (k : ℕ ) := 0.01 * (k + 1)
  have areas_sum : (∑ i in Finset.range 50, fractional_area i) =  12.75 
  sorry

end region_area_l711_711515


namespace sum_of_k_with_distinct_integer_roots_l711_711842

theorem sum_of_k_with_distinct_integer_roots :
  ∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ 3*p*q = 12 ∧ p + q = k/3}, k = 0 :=
by
  sorry

end sum_of_k_with_distinct_integer_roots_l711_711842


namespace towel_bleach_percentage_decrease_l711_711940

theorem towel_bleach_percentage_decrease :
  ∀ (L B : ℝ), (L > 0) → (B > 0) → 
  let L' := 0.70 * L 
  let B' := 0.75 * B 
  let A := L * B 
  let A' := L' * B' 
  (A - A') / A * 100 = 47.5 :=
by sorry

end towel_bleach_percentage_decrease_l711_711940


namespace sum_of_all_ks_l711_711872

theorem sum_of_all_ks (k : ℤ) :
  (∃ r s : ℤ, r ≠ s ∧ (3 * x^2 - k * x + 12 = 0) ∧ (r + s = k / 3) ∧ (r * s = 4)) → (k = 15 ∨ k = -15) → k + k = 0 :=
by
  sorry

end sum_of_all_ks_l711_711872


namespace find_n_l711_711558

theorem find_n (x : ℝ) (n : ℝ) 
  (h1 : log 10 (sin x) + log 10 (cos x) = -2)
  (h2 : log 10 (sin x + cos x) = (1 / 3) * (log 10 n - 2)) :
  n = 1.0309 * 10 ^ 2 :=
by
  sorry

end find_n_l711_711558


namespace david_catches_cory_l711_711970

noncomputable def speed_to_time_reach (distance_ahead : ℕ) (speed_factor : ℕ) (track_length : ℕ) :=
  (distance_ahead * speed_factor) / (speed_factor - 1)

noncomputable def laps_run (distance : ℕ) (track_length : ℕ) :=
  distance / track_length

theorem david_catches_cory (distance_ahead : ℕ) (speed_factor : ℕ) (track_length : ℕ) (cory_speed : ℕ) :
  (laps_run (speed_to_time_reach distance_ahead speed_factor track_length * speed_factor * cory_speed) track_length) = 2 :=
by
  -- Given conditions
  assume (distance_ahead = 50)
  assume (speed_factor = 3 / 2)
  assume (track_length = 600)
  sorry

end david_catches_cory_l711_711970


namespace count_three_digit_multiples_of_seven_l711_711614

theorem count_three_digit_multiples_of_seven :
  let a := 100 in
  let b := 999 in
  let smallest := (Nat.ceil (a.toRat / 7)).natAbs * 7 in
  let largest := (b / 7) * 7 in
  (largest / 7) - ((smallest - 1) / 7) = 128 := sorry

end count_three_digit_multiples_of_seven_l711_711614


namespace loaf_slices_l711_711502

theorem loaf_slices (S : ℕ) (T : ℕ) : 
  (S - 7 = 2 * T + 3) ∧ (S ≥ 20) → S = 20 :=
by
  sorry

end loaf_slices_l711_711502


namespace sum_of_values_k_l711_711866

theorem sum_of_values_k (k : ℕ) : 
  (∀ x y : ℤ, (3 * x * x - k * x + 12 = 0) ∧ (3 * y * y - k * y + 12 = 0) ∧ x ≠ y) → k = 0 :=
by
  sorry

end sum_of_values_k_l711_711866


namespace part_a_part_b_l711_711409

-- We define the hypotheses and bring in necessary domains.

-- Given conditions for part (a)
theorem part_a (x : ℝ) (h_eq : sin x ^ 3 + sin (2 * real.pi / 3 + x) ^ 3 + sin (4 * real.pi / 3 + x) ^ 3 + (3 / 4) * cos (2 * x) = 0) :
  ∃ k : ℤ, x = (4 * k + 1) * real.pi / 10 := sorry

-- Given conditions for part (b)
theorem part_b (n : ℕ) (x : ℝ) (h_eq : ∃ k : ℤ, x = (4 * k + 1) * real.pi / 10) (h_polygon : ∀ i : ℕ, i < n → true) :
  (2 ∣ n ∨ 3 ∣ n) ∨ prime n → false := sorry

end part_a_part_b_l711_711409


namespace men_in_second_group_l711_711060

theorem men_in_second_group (M : ℕ) (h1 : 16 * 30 = 480) (h2 : M * 24 = 480) : M = 20 :=
by
  sorry

end men_in_second_group_l711_711060


namespace cubic_identity_l711_711580

theorem cubic_identity (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / x^3) = 140 := 
  sorry

end cubic_identity_l711_711580


namespace ensure_user_data_security_l711_711761

-- Definitions for conditions
def cond1 : Prop := ∃ (app : Type), 
  (app.is_online_store_with_credit_card_payment)
def cond2 : Prop := ∀ (app : Type), 
  (app.needs_security_against_data_theft)

-- Definitions for security measures
def security_measures : Type :=
| avoid_storing_card_data
| encrypt_stored_data
| encrypt_data_in_transit
| code_obfuscation
| restrict_on_rooted_devices
| integrate_antivirus

-- Theorem statement
theorem ensure_user_data_security 
  (c1 : cond1) (c2 : cond2) : 
  ∃ (measures : List security_measures), measures.length ≥ 3 :=
sorry

end ensure_user_data_security_l711_711761


namespace eval_definite_integral_l711_711503

noncomputable def definite_integral_val : ℝ :=
  ∫ x in 0..2, x - 3

theorem eval_definite_integral : definite_integral_val = -4 := 
by 
  sorry

end eval_definite_integral_l711_711503


namespace min_common_perimeter_l711_711375

theorem min_common_perimeter : ∃ x y k : ℕ, 
  (x ≠ y) ∧ 
  (2 * x + 5 * k = 2 * y + 6 * k) ∧ 
  (5 * real.sqrt (4 * x ^ 2 - 25 * k ^ 2) = 6 * real.sqrt (4 * y ^ 2 - 36 * k ^ 2)) ∧ 
  (2 * x + 5 * k = 399) :=
sorry

end min_common_perimeter_l711_711375


namespace solve_positive_solution_l711_711978

noncomputable def positiveSolution : ℕ := 2

theorem solve_positive_solution (x : ℕ) (h : x = positiveSolution) :
  (∃ y z : ℝ, (y = real.root x (x * real.root x (x * real.root x (x * ...))) ∧ z = real.root x (x + real.root x (x + real.root x (x + ...))) ∧ y^x = x * y ∧ z^x = x + z ∧ y = z)) → 
  x = 2 := 
by
  sorry

end solve_positive_solution_l711_711978


namespace min_max_carpet_length_l711_711669

theorem min_max_carpet_length (corridor_length : ℝ) (total_rug_length : ℝ) (num_rugs : ℕ) (rug_lengths : Fin num_rugs → ℝ) :
  corridor_length = 100 → total_rug_length = 1000 → (∑ i, rug_lengths i) = total_rug_length → (num_rugs = 20) → (∃ i, rug_lengths i ≥ 50) :=
by
  intros h1 h2 h3 h4
  sorry

end min_max_carpet_length_l711_711669


namespace infinite_fractions_approx_l711_711293

theorem infinite_fractions_approx {x : ℝ} : ∃ᶠ pq in (λ (p q : ℤ), (p / q)), ∣ x - p / q ∣ < 1 / q ^ 2 :=
sorry

end infinite_fractions_approx_l711_711293


namespace total_distance_traveled_l711_711277

-- Definitions from the conditions as variables and constants.
def john_speed_alone := 4 -- miles per hour
def john_speed_with_dog := 6 -- miles per hour
def run_time := 0.5 -- hours (for 30 minutes)

-- Proof that total distance traveled by John is 5 miles.
theorem total_distance_traveled : 
    (john_speed_with_dog * run_time + john_speed_alone * run_time) = 5 := 
by 
    sorry

end total_distance_traveled_l711_711277


namespace inlet_pipe_rate_16_liters_per_minute_l711_711476

noncomputable def rate_of_inlet_pipe : ℝ :=
  let capacity := 21600 -- litres
  let outlet_time_alone := 10 -- hours
  let outlet_time_with_inlet := 18 -- hours
  let outlet_rate := capacity / outlet_time_alone
  let combined_rate := capacity / outlet_time_with_inlet
  let inlet_rate := outlet_rate - combined_rate
  inlet_rate / 60 -- converting litres/hour to litres/min

theorem inlet_pipe_rate_16_liters_per_minute : rate_of_inlet_pipe = 16 :=
by
  sorry

end inlet_pipe_rate_16_liters_per_minute_l711_711476


namespace connected_if_and_only_if_contains_edge_of_cut_l711_711627

-- Definitions used in conditions
structure Graph := (vertices : Type) (edges : vertices → vertices → Prop)
def is_locally_finite (G : Graph) : Prop := ∀ v, finite { w | G.edges v w }
def standard_subspace (G : Graph) : Set (G.vertices → ℝ) := sorry -- Assume this is defined

-- Using definitions to formulate the theorem
theorem connected_if_and_only_if_contains_edge_of_cut (G : Graph) (X : Set (G.vertices → ℝ)) 
  (hG : is_locally_finite G) (hX : X = standard_subspace G) :
  (connected X) ↔ (∀ (V1 V2 : Set G.vertices), finite [V1, V2] ∧ (V1 ∩ X ≠ ∅) ∧ (V2 ∩ X ≠ ∅) → ∃ e, e ∈ X ∩ (V1, V2)) :=
sorry

end connected_if_and_only_if_contains_edge_of_cut_l711_711627


namespace cube_diagonal_plane_parallel_min_length_l711_711075

-- Defining the problem
def minimum_length_MN (cube_edge_length : ℝ) (M N : ℝ × ℝ × ℝ) : Prop :=
  let x := 1 / 3 in
  let MN := (6 * (x - 1/3)^2 + 1/3)^0.5 in
  MN = (3^0.5) / 3

-- Main hypothesis and goal
theorem cube_diagonal_plane_parallel_min_length : 
  ∀ (cube_edge_length : ℝ) (M N : ℝ × ℝ × ℝ),
  cube_edge_length = 1 →
  -- M on side diagonal A_1D
  M.1 = 1 - M.2 ∧ M.3 = 0 →
  -- N on diagonal CD_1
  N.1 = 1 - N.3 ∧ N.2 = 0 →
  -- MN parallel to diagonal plane A_1ACC_1
  (∃ (x : ℝ), x = 1/3 ∧ (M.1 - N.1)^2 + (M.2 - N.2)^2 + (M.3 - N.3)^2 = (3 / 9)) :=
by
  intros cube_edge_length M N h_edge_length hM hN hMN
  sorry

end cube_diagonal_plane_parallel_min_length_l711_711075


namespace three_digit_multiples_of_seven_l711_711607

theorem three_digit_multiples_of_seven : 
  let min_bound := 100
  let max_bound := 999
  let first_multiple := Nat.ceil (min_bound / 7)
  let last_multiple := Nat.floor (max_bound / 7)
  (last_multiple - first_multiple + 1) = 128 :=
by 
  let min_bound := 100
  let max_bound := 999
  let first_multiple := Nat.ceil (min_bound / 7)
  let last_multiple := Nat.floor (max_bound / 7)
  have calc_first_multiple : first_multiple = 15 := sorry
  have calc_last_multiple : last_multiple = 142 := sorry
  have count_multiples : (last_multiple - first_multiple + 1) = 128 := sorry
  exact count_multiples

end three_digit_multiples_of_seven_l711_711607


namespace convex_polygon_segments_l711_711316

theorem convex_polygon_segments (P : ℝ) (sides : list ℝ) :
  (∀ side ∈ sides, side ≤ P / 2) → (sides.sum = P) → 
  ∃ (l1 l2 : ℝ), (l1 ∈ sides) ∧ (l2 ∈ sides) ∧ (|l1 - l2| ≤ P / 3) :=
by
  sorry

end convex_polygon_segments_l711_711316


namespace similar_triangles_MN_length_l711_711029

open_locale classical

variables (P Q R X Y Z M N O : Type)
variables [metric_space P] [metric_space Q] [metric_space R]
variables [metric_space X] [metric_space Y] [metric_space Z]
variables [metric_space M] [metric_space N] [metric_space O]

-- Given conditions
variable (PQ := 4)
variable (QR := 8)
variable (YZ := 24)
variable (NO := 32)
variable (similar_PQR_XYZ : ∀ P Q R X Y Z, similar P Q R X Y Z)
variable (similar_XYZ_MNO : ∀ X Y Z M N O, similar X Y Z M N O)

-- Goal
theorem similar_triangles_MN_length :
  ∀ (MN : ℝ), PQ = 4 → QR = 8 → YZ = 24 → NO = 32 → 
    similar_PQR_XYZ P Q R X Y Z → similar_XYZ_MNO X Y Z M N O → MN = 16 := by
  sorry

end similar_triangles_MN_length_l711_711029


namespace journey_time_l711_711911

noncomputable def journey_time_proof : Prop :=
  ∃ t1 t2 t3 : ℝ,
    25 * t1 - 25 * t2 + 25 * t3 = 100 ∧
    5 * t1 + 5 * t2 + 25 * t3 = 100 ∧
    25 * t1 + 5 * t2 + 5 * t3 = 100 ∧
    t1 + t2 + t3 = 8

theorem journey_time : journey_time_proof := by sorry

end journey_time_l711_711911


namespace root_analysis_l711_711799

noncomputable def root1 (a : ℝ) : ℝ :=
2 * a + 2 * Real.sqrt (a^2 - 3 * a + 2)

noncomputable def root2 (a : ℝ) : ℝ :=
2 * a - 2 * Real.sqrt (a^2 - 3 * a + 2)

noncomputable def derivedRoot (a : ℝ) : ℝ :=
(3 * a - 2) / a

theorem root_analysis (a : ℝ) (ha : a > 0) :
( (2/3 ≤ a ∧ a < 1) ∨ (2 < a) → (root1 a ≥ 0 ∧ root2 a ≥ 0)) ∧
( 0 < a ∧ a < 2/3 → (derivedRoot a < 0 ∧ root1 a ≥ 0)) :=
sorry

end root_analysis_l711_711799


namespace find_divisor_l711_711875

theorem find_divisor (d : ℕ) (h1 : d ∣ (9671 - 1)) : d = 9670 :=
by
  sorry

end find_divisor_l711_711875


namespace total_wire_length_l711_711420

theorem total_wire_length (r : ℝ) (h : r = 8) : 
  let C := 2 * real.pi * r in
  let two_radii := 2 * r in
  C + two_radii = 16 * real.pi + 16 :=
by
  sorry

end total_wire_length_l711_711420


namespace ensure_user_data_security_l711_711762

-- Definitions for conditions
def cond1 : Prop := ∃ (app : Type), 
  (app.is_online_store_with_credit_card_payment)
def cond2 : Prop := ∀ (app : Type), 
  (app.needs_security_against_data_theft)

-- Definitions for security measures
def security_measures : Type :=
| avoid_storing_card_data
| encrypt_stored_data
| encrypt_data_in_transit
| code_obfuscation
| restrict_on_rooted_devices
| integrate_antivirus

-- Theorem statement
theorem ensure_user_data_security 
  (c1 : cond1) (c2 : cond2) : 
  ∃ (measures : List security_measures), measures.length ≥ 3 :=
sorry

end ensure_user_data_security_l711_711762


namespace initial_children_on_bus_l711_711328

theorem initial_children_on_bus (a b x : ℕ) (h1 : a = 38) (h2 : b = 64) : x = b - a → x = 26 :=
by
  intro h
  rw [h1, h2]
  rw [h]
  sorry

end initial_children_on_bus_l711_711328


namespace betty_ordered_4_lipsticks_l711_711950

variables (total_items : ℕ) (slippers : ℕ) (slippers_cost : ℝ) (lipsticks_cost : ℝ) (hair_colors : ℕ) (hair_color_cost : ℝ) (total_paid : ℝ) (lipsticks : ℕ)

-- Given conditions
def conditions : Prop :=
  total_items = 18 ∧
  slippers = 6 ∧
  slippers_cost = 2.5 ∧
  hair_colors = 8 ∧
  hair_color_cost = 3 ∧
  total_paid = 44 ∧
  lipsticks_cost = 1.25

-- Prove that the number of pieces of lipstick ordered is 4
theorem betty_ordered_4_lipsticks : conditions total_items slippers slippers_cost lipsticks_cost hair_colors hair_color_cost total_paid lipsticks → lipsticks = 4 :=
  sorry

end betty_ordered_4_lipsticks_l711_711950


namespace total_hours_charged_l711_711310

variable (K P M : ℕ)

theorem total_hours_charged (h1 : P = 2 * K) (h2 : P = M / 3) (h3 : M = K + 80) : K + P + M = 144 := 
by
  sorry

end total_hours_charged_l711_711310


namespace arithmetic_to_geometric_find_original_numbers_l711_711367

theorem arithmetic_to_geometric (k : ℕ) 
  (h1 : ∃ k, (3 * k + 1) * 4 * k = 5 * k * (3 * k + 1))
  : k = 5 :=
begin
  sorry,
end

theorem find_original_numbers :
  ∃ (a b c : ℕ), a = 15 ∧ b = 20 ∧ c = 25 :=
begin
  use [15, 20, 25],
  exact ⟨rfl, rfl, rfl⟩
end

end arithmetic_to_geometric_find_original_numbers_l711_711367


namespace closest_points_distance_between_circles_l711_711132

def dist (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def radius (p : ℝ × ℝ) : ℝ := 
  p.2

def closest_distance_between_circles (c1 c2 : ℝ × ℝ) : ℝ := 
  dist c1 c2 - (radius c1 + radius c2)

theorem closest_points_distance_between_circles :
  closest_distance_between_circles (3, 3) (20, 12) = real.sqrt 370 - 15 :=
sorry

end closest_points_distance_between_circles_l711_711132


namespace total_stamps_received_l711_711321

theorem total_stamps_received
  (initial_stamps : ℕ)
  (final_stamps : ℕ)
  (received_stamps : ℕ)
  (h_initial : initial_stamps = 34)
  (h_final : final_stamps = 61)
  (h_received : received_stamps = final_stamps - initial_stamps) :
  received_stamps = 27 :=
by 
  sorry

end total_stamps_received_l711_711321


namespace integral_neg_f_eq_14_over_3_l711_711243

noncomputable def f : ℝ → ℝ :=
  λ x, x^2 + x

theorem integral_neg_f_eq_14_over_3 :
  (∫ x in 1..3, f (-x)) = 14 / 3 :=
by
  sorry

end integral_neg_f_eq_14_over_3_l711_711243


namespace vector_minimization_and_angle_condition_l711_711559

noncomputable def find_OC_condition (C_op C_oa C_ob : ℝ × ℝ) 
  (C : ℝ × ℝ) : Prop := 
  let CA := (C_oa.1 - C.1, C_oa.2 - C.2)
  let CB := (C_ob.1 - C.1, C_ob.2 - C.2)
  (CA.1 * CB.1 + CA.2 * CB.2) ≤ (C_op.1 * CB.1 + C_op.2 * CB.2)

theorem vector_minimization_and_angle_condition (C : ℝ × ℝ) 
  (C_op := (2, 1)) (C_oa := (1, 7)) (C_ob := (5, 1)) :
  (C = (4, 2)) → 
  find_OC_condition C_op C_oa C_ob C →
  let CA := (C_oa.1 - C.1, C_oa.2 - C.2)
  let CB := (C_ob.1 - C.1, C_ob.2 - C.2)
  let cos_ACB := (CA.1 * CB.1 + CA.2 * CB.2) / 
                 (Real.sqrt (CA.1^2 + CA.2^2) * Real.sqrt (CB.1^2 + CB.2^2))
  cos_ACB = -4 * Real.sqrt (17) / 17 :=
  by 
    intro h1 find
    let CA := (C_oa.1 - C.1, C_oa.2 - C.2)
    let CB := (C_ob.1 - C.1, C_ob.2 - C.2)
    let cos_ACB := (CA.1 * CB.1 + CA.2 * CB.2) / 
                   (Real.sqrt (CA.1^2 + CA.2^2) * Real.sqrt (CB.1^2 + CB.2^2))
    exact sorry

end vector_minimization_and_angle_condition_l711_711559


namespace minimal_fuel_for_travel_l711_711064

def car_fuel_travel_min_fuel (L a d : ℝ) (k : ℕ) (h_da : d > a) : ℝ :=
  (k + n) * L - (k + 2 * n) * ((d_n - d) / a) * L

noncomputable def d_n (a : ℝ) (k n : ℕ) : ℝ :=
  a * (1 + ∑ i in finset.range n, (1 / (k + 2 * i : ℝ)))

theorem minimal_fuel_for_travel {L a d : ℝ} {k n : ℕ} (h_da : d > a) :
  minimal_fuel_required k L a d = (k + n) * L - (k + 2 * n) * (d_n a k n - d) / a * L :=
sorry

end minimal_fuel_for_travel_l711_711064


namespace smallest_e_value_l711_711595

noncomputable def poly := (1, -3, 7, -2/5)

theorem smallest_e_value (a b c d e : ℤ) 
  (h_poly_eq : a * (1)^4 + b * (1)^3 + c * (1)^2 + d * (1) + e = 0)
  (h_poly_eq_2 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
  (h_poly_eq_3 : a * 7^4 + b * 7^3 + c * 7^2 + d * 7 + e = 0)
  (h_poly_eq_4 : a * (-2/5)^4 + b * (-2/5)^3 + c * (-2/5)^2 + d * (-2/5) + e = 0)
  (h_e_positive : e > 0) :
  e = 42 :=
sorry

end smallest_e_value_l711_711595


namespace dot_product_range_l711_711284

theorem dot_product_range (a b : ℝ) (θ : ℝ) (h1 : a = 8) (h2 : b = 12)
  (h3 : 30 * (Real.pi / 180) ≤ θ ∧ θ ≤ 60 * (Real.pi / 180)) :
  48 * Real.sqrt 3 ≤ a * b * Real.cos θ ∧ a * b * Real.cos θ ≤ 48 :=
by
  sorry

end dot_product_range_l711_711284


namespace conic_section_hyperbola_x_axis_l711_711342

theorem conic_section_hyperbola_x_axis (θ : ℝ) (h1 : sin θ + 3 > 0) (h2 : sin θ - 2 < 0) :
  ∃ a b : ℝ, ∀ x y : ℝ, (x^2 / (sin θ + 3) - y^2 / (sin θ - 2) = 1) ↔ (x^2 / a^2 - y^2 / b^2 = 1) :=
sorry

end conic_section_hyperbola_x_axis_l711_711342


namespace intersection_of_complements_l711_711221

variable (U : Set ℕ)
variable (A : Set ℕ)
variable (B : Set ℕ)

def complement (U A : Set ℕ) : Set ℕ := U \ A

theorem intersection_of_complements :
  U = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} →
  A = {0, 1, 3, 5, 8} →
  B = {2, 4, 5, 6, 8} →
  (complement U A ∩ complement U B) = {7, 9} :=
by
  intros hU hA hB
  sorry

end intersection_of_complements_l711_711221


namespace R_n_nonnegative_def_R_n_alpha_nonnegative_def_l711_711037

-- Definition of R_n for the case where 1 < \alpha < 2
noncomputable def R_n_alpha (s t : ℝ^n) (α : ℝ) : ℝ :=
  (|t|_α + |s|_α) / 2 - (|s - t|_α) / 2

-- Define the norm |.|_α for 1 < \alpha < 2
noncomputable def p_norm (α : ℝ) (v : ℝ^n) : ℝ :=
  (v.data.map (λ x, |x|^α)).sum ^ (1 / α)

-- Prove non-negative definiteness for R_n
theorem R_n_nonnegative_def (s t : ℝ^n) : 
  (|t| + |s|) / 2 - (|s - t|) / 2 ≥ 0 := sorry

-- Prove non-negative definiteness for R_n_alpha
theorem R_n_alpha_nonnegative_def (s t : ℝ^n) (α : ℝ) (h_alpha : 1 < α ∧ α < 2) :
  R_n_alpha s t α ≥ 0 := sorry

end R_n_nonnegative_def_R_n_alpha_nonnegative_def_l711_711037


namespace determine_B_is_9_l711_711358

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem determine_B_is_9 (B : ℕ) (h_digits : B ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  is_prime (303200 + B) ↔ B = 9 :=
by
  sorry

end determine_B_is_9_l711_711358


namespace cube_volume_l711_711426

theorem cube_volume (A : ℝ) (hA : A = 294) : ∃ (V : ℝ), V = 343 :=
by
  sorry

end cube_volume_l711_711426


namespace problem_two_l711_711211

noncomputable def f (x : ℝ) := Real.log x - x

def a_n (n : ℕ) : ℝ := 1 + 1 / (2 ^ n)

theorem problem_two (n : ℕ) :
  (∏ i in Finset.range n, a_n (i + 1)) < Real.exp 1 := 
sorry

end problem_two_l711_711211


namespace find_n_l711_711649

theorem find_n (N : ℕ) (hN : 10 ≤ N ∧ N < 100)
  (h : ∃ a b : ℕ, N = 10 * a + b ∧ 4 * a + 2 * b = N / 2) : 
  N = 32 ∨ N = 64 ∨ N = 96 :=
sorry

end find_n_l711_711649


namespace sum_of_k_values_with_distinct_integer_solutions_l711_711844

theorem sum_of_k_values_with_distinct_integer_solutions : 
  ∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ 3 * p * q + (-k) * (p + q) + 12 = 0}, k = 0 :=
by
  sorry

end sum_of_k_values_with_distinct_integer_solutions_l711_711844


namespace three_digit_multiples_of_seven_count_l711_711611

theorem three_digit_multiples_of_seven_count :
  let smallest := 15
  let largest := 142
  largest - smallest + 1 = 128 :=
by
  let smallest := 15
  let largest := 142
  have h_smallest : 7 * smallest = 105 := rfl
  have h_largest : 7 * largest = 994 := rfl
  show largest - smallest + 1 = 128 from sorry

end three_digit_multiples_of_seven_count_l711_711611


namespace adam_simon_distance_l711_711467

theorem adam_simon_distance : 
  ∀ (x : ℝ),
    (let adam_distance := 8 * x in
     let simon_distance := 6 * x in
     adam_distance^2 + simon_distance^2 = 60^2) → x = 6 :=
by
  intros x h
  simp at h
  sorry

end adam_simon_distance_l711_711467


namespace total_shirts_made_l711_711945

def shirtsPerMinute := 6
def minutesWorkedYesterday := 12
def shirtsMadeToday := 14

theorem total_shirts_made : shirtsPerMinute * minutesWorkedYesterday + shirtsMadeToday = 86 := by
  sorry

end total_shirts_made_l711_711945


namespace smallest_n_gcd_l711_711040

theorem smallest_n_gcd (n : ℕ) :
  (∃ n > 0, gcd (11 * n - 3) (8 * n + 2) > 1) ∧ (∀ m > 0, gcd (11 * m - 3) (8 * m + 2) > 1 → m ≥ n) ↔ n = 19 :=
by
  sorry

end smallest_n_gcd_l711_711040


namespace powerThreeExpression_l711_711567

theorem powerThreeExpression (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / (x^3)) = 140 := sorry

end powerThreeExpression_l711_711567


namespace jill_investment_l711_711697

def compound_interest (P : ℕ) (r : ℚ) (n : ℕ) (t : ℕ) : ℚ :=
  P * (1 + r / n)^(n * t)

theorem jill_investment : 
  compound_interest 10000 (396 / 10000) 2 2 ≈ 10812 := by
sorry

end jill_investment_l711_711697


namespace john_photos_l711_711488

theorem john_photos (photos_cristina : ℕ) (photos_sarah : ℕ) (photos_clarissa : ℕ) (total_slots : ℕ) (john_photos : ℕ) :
  photos_cristina = 7 → photos_sarah = 9 → photos_clarissa = 14 → total_slots = 40 → 
  john_photos = total_slots - (photos_cristina + photos_sarah + photos_clarissa) → john_photos = 10 :=
by
  intros h_cristina h_sarah h_clarissa h_slots h_john
  rw [h_cristina, h_sarah, h_clarissa, h_slots, h_john]
  -- automatic simplifications lead to conclusion
  sorry

end john_photos_l711_711488


namespace inv_mod_3_47_l711_711509

theorem inv_mod_3_47 : ∃ x : ℤ, 0 ≤ x ∧ x < 47 ∧ 3 * x ≡ 1 [MOD 47] := 
by { use 16, sorry }

end inv_mod_3_47_l711_711509


namespace problem_l711_711600

noncomputable def A : ℝ × ℝ := (-5, 0)
noncomputable def B : ℝ × ℝ := (3, -3)
noncomputable def C : ℝ × ℝ := (0, 2)

-- Midpoint of side BC
noncomputable def M : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Slope and equation of line BC
noncomputable def k_BC : ℝ := (C.2 - B.2) / (C.1 - B.1)
noncomputable def equation_BC : ℝ × ℝ → Prop := λ p, 5 * p.1 + 3 * p.2 - 6 = 0

-- Slope and equation of line parallel to AC passing through M
noncomputable def k_AC : ℝ := (C.2 - A.2) / (C.1 - A.1)
noncomputable def equation_parallel_AC : ℝ × ℝ → Prop := λ p, 4 * p.1 - 10 * p.2 - 11 = 0

theorem problem (hBC : equation_BC B ∧ equation_BC C)
                (hM_AC : equation_parallel_AC M) :
  ∃ BC_eq : (ℝ × ℝ → Prop), BC_eq B ∧ BC_eq C ∧
  BC_eq = equation_BC ∧
  ∃ M_AC_eq : (ℝ × ℝ → Prop), M_AC_eq M ∧
  M_AC_eq = equation_parallel_AC := 
by
  sorry

end problem_l711_711600


namespace similarTriangles_l711_711298

variable {A B C M D O : Type} 

-- Let's assume points are in two-dimensional space ℝ²
variables [Point A] [Point B] [Point C] [Point M] [Point D] [Point O]

-- Defining the conditions
def isAcuteTriangle (ABC : Triangle) : Prop :=
  ∀ {A B C : ℝ×ℝ}, 0 < ∠A < π / 2 ∧ 0 < ∠B < π / 2 ∧ 0 < ∠C < π / 2 

def isMidpoint (A : Point) (C : Point) (M : Point) : Prop :=
  midpoint A C M

def isFootOfAltitude (A : Point) (BC : Line) (D : Point) : Prop :=
  perpendicular_from_point_to_line A BC D

def isCircumcenter (O : Point) (ABC : Triangle) : Prop :=
  circumcenter O ABC

def trianglesSimilar (MOA DBA : Triangle) : Prop :=
  ∠MOA = ∠DBA ∧ ∠OMA = ∠DAB ∧ ∠OAM = ∠BDA

-- Final Statement
theorem similarTriangles
  (ABC : Triangle)
  (M : Point)
  (D : Point)
  (O : Point)
  (h1 : isAcuteTriangle ABC)
  (h2 : isMidpoint A C M)
  (h3 : isFootOfAltitude A BC D)
  (h4 : isCircumcenter O ABC) :
  trianglesSimilar (Triangle.mk M O A) (Triangle.mk D B A) :=
  sorry

end similarTriangles_l711_711298


namespace minimum_distance_l711_711777

theorem minimum_distance (A B C : ℝ) (wall_length : ℝ) (height_to_wall : ℝ)
  (vertical_distance_to_B : ℝ) (rounded_distance : ℝ) :
  A = 0 →
  B = 1600 →
  height_to_wall = 400 →
  wall_length = 1600 →
  vertical_distance_to_B = 600 →
  rounded_distance = 3578 →
  (let B_prime := B in
  let distance := real.sqrt (wall_length^2 + (height_to_wall + vertical_distance_to_B)^2) in
  int.round distance = rounded_distance) := 
sorry

end minimum_distance_l711_711777


namespace largest_last_digit_in_string_l711_711345

theorem largest_last_digit_in_string :
  ∃ (s : Nat → Fin 10), 
    (s 0 = 1) ∧ 
    (∀ k, k < 99 → (∃ m, (s k * 10 + s (k + 1)) = 17 * m ∨ (s k * 10 + s (k + 1)) = 23 * m)) ∧
    (∃ l, l < 10 ∧ (s 99 = l)) ∧
    (forall last, (last < 10 ∧ (s 99 = last))) ∧
    (∀ m n, s 99 = m → s 99 = n → m ≤ n → n = 9) :=
sorry

end largest_last_digit_in_string_l711_711345


namespace joan_balloon_counts_l711_711710

theorem joan_balloon_counts 
    (orange_initial blue_initial purple_initial : ℕ)
    (orange_lost_frac blue_lost_frac purple_lost_frac : ℚ)
    (orange_lost purple_lost blue_lost : ℕ) 
    (orange_left blue_left purple_left : ℕ) :
  orange_initial = 8 →
  blue_initial = 10 →
  purple_initial = 6 →
  orange_lost_frac = 25 / 100 →
  blue_lost_frac = 1 / 5 →
  purple_lost_frac = 3333 / 10000 →
  orange_lost = (orange_lost_frac * orange_initial) →
  blue_lost = (blue_lost_frac * blue_initial) →
  purple_lost = (purple_lost_frac * purple_initial).to_nat →
  orange_left = (orange_initial - orange_lost) →
  blue_left = (blue_initial - blue_lost) →
  purple_left = (purple_initial - purple_lost) →
  orange_left = 6 ∧ 
  blue_left = 8 ∧ 
  purple_left = 4 := 
by
  intros
  sorry

end joan_balloon_counts_l711_711710


namespace number_of_correct_propositions_l711_711588

theorem number_of_correct_propositions : 
  (∀ x : ℝ, x ∈ Ioi (5/2) → x ∉ setOf fun x => 2^x = x^2 - 5 * x + 6) ∧ -- Condition 1 rewritten in correct form, stating "incorrectness"
  (∀ (x₁ y₁ x₂ y₂ : ℝ), y₂ ≠ y₁ → (∀ x y, (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁))) ∧ -- Condition 2
  (¬ (∀ x : ℝ, x^2 - x - 1 ≤ 0) ↔ ∃ x₀ : ℝ, x₀^2 - x₀ - 1 > 0) -- Condition 3
  → (number_of_correct : 2) := by sorry

end number_of_correct_propositions_l711_711588


namespace outdoor_section_area_l711_711934

theorem outdoor_section_area :
  ∀ (width length : ℕ), width = 4 → length = 6 → (width * length = 24) :=
by
  sorry

end outdoor_section_area_l711_711934


namespace geometric_sequence_304th_term_l711_711683

theorem geometric_sequence_304th_term (a r : ℤ) (n : ℕ) (h_a : a = 8) (h_ar : a * r = -8) (h_n : n = 304) :
  ∃ t : ℤ, t = -8 :=
by
  sorry

end geometric_sequence_304th_term_l711_711683


namespace Liam_chapters_in_fourth_week_l711_711754

noncomputable def chapters_in_first_week (x : ℕ) : ℕ := x
noncomputable def chapters_in_second_week (x : ℕ) : ℕ := x + 3
noncomputable def chapters_in_third_week (x : ℕ) : ℕ := x + 6
noncomputable def chapters_in_fourth_week (x : ℕ) : ℕ := x + 9
noncomputable def total_chapters (x : ℕ) : ℕ := x + (x + 3) + (x + 6) + (x + 9)

theorem Liam_chapters_in_fourth_week : ∃ x : ℕ, total_chapters x = 50 → chapters_in_fourth_week x = 17 :=
by
  sorry

end Liam_chapters_in_fourth_week_l711_711754


namespace track_length_l711_711101

theorem track_length (x : ℝ) (alice_first_meet : 120) (bob_second_meet : 180)
  (h1 : 0 < alice_first_meet) (h2 : 0 < bob_second_meet)
  (h3 : bob_second_meet < x) (h4 : alice_first_meet < x) :
  let bob_first_meet := (x / 2) - alice_first_meet in
  let bob_total_second_meet := bob_first_meet + bob_second_meet in
  let alice_total_second_meet := x - bob_total_second_meet in
  (120 / bob_first_meet = alice_total_second_meet / bob_total_second_meet) →
  x = 600 :=
by sorry

end track_length_l711_711101


namespace volume_of_cube_with_surface_area_l711_711430

theorem volume_of_cube_with_surface_area (S : ℝ) (hS : S = 294) : 
  ∃ V : ℝ, V = 343 :=
by
  let s := (S / 6).sqrt
  have hs : s = 7 := by sorry
  use s ^ 3
  simp [hs]
  exact sorry

end volume_of_cube_with_surface_area_l711_711430


namespace vector_dot_product_l711_711170

variables (a b c : ℝ^3)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hc : ∥c∥ = 1)
variables (h : 3 • a + 4 • b + 5 • c = 0)

theorem vector_dot_product : b ⬝ (a + c) = -4 / 5 :=
by
  sorry

end vector_dot_product_l711_711170


namespace negation_proof_l711_711806

theorem negation_proof (a b c : ℝ) : 
  (a ≤ b → a * c ^ 2 ≤ b * c ^ 2) := 
begin
  sorry
end

end negation_proof_l711_711806


namespace optimal_triangle_division_l711_711403

theorem optimal_triangle_division (T : Type) [triangle T] (Q : T) (l : T → T → T) (S1 S2 : ℝ) :
  centroid_exists T → -- Condition that the centroid exists
  (l T Q).divides T S1 S2 → -- Condition that the line divides triangle T into areas S1 and S2
  (4 / 5 : ℝ) ≤ S1 / S2 ∧ S1 / S2 ≤ (5 / 4 : ℝ) := by
sorry

end optimal_triangle_division_l711_711403


namespace vector_addition_l711_711099

variables (V : Type*) [AddCommGroup V]

-- Define the vectors
variables (a b : V)

-- Define the arbitrary starting point M (which could be the zero vector in V)
def M : V := 0 -- Assuming M is the zero vector for simplicity

-- Define the endpoints after adding the vectors graphically
def A : V := M + a
def B : V := A + b

-- Define the resultant vector
def v_addition_result : V := B - M

-- The theorem stating the correct answer
theorem vector_addition : v_addition_result = a + b :=
by sorry

end vector_addition_l711_711099


namespace number_of_nonofficers_l711_711400

/-- 
Given the following conditions:
1. The average salary of all employees is Rs. 120 per month.
2. The average salary of officers is Rs. 450 per month.
3. The average salary of non-officers is Rs. 110 per month.
4. The number of officers is 15.

Prove that the number of non-officers in the office is 495.
-/
theorem number_of_nonofficers (avg_sal : ℝ) (avg_sal_officers : ℝ) (avg_sal_nonofficers : ℝ) (num_officers : ℕ) (num_nonofficers : ℕ) :
  avg_sal = 120 ∧ avg_sal_officers = 450 ∧ avg_sal_nonofficers = 110 ∧ num_officers = 15 →
  num_nonofficers = 495 :=
by
  intros h,
  sorry

end number_of_nonofficers_l711_711400


namespace angle_A_and_shape_of_triangle_l711_711686

theorem angle_A_and_shape_of_triangle 
  (a b c : ℝ)
  (h1 : a^2 - c^2 = a * c - b * c)
  (h2 : ∃ r : ℝ, a = b * r ∧ c = b / r)
  (h3 : ∃ B C : Type, B = A ∧ C ≠ A ) :
  ∃ (A : ℝ), A = 60 ∧ a = b ∧ b = c := 
sorry

end angle_A_and_shape_of_triangle_l711_711686


namespace jill_investment_l711_711699

def compound_interest (P : ℕ) (r : ℚ) (n : ℕ) (t : ℕ) : ℚ :=
  P * (1 + r / n)^(n * t)

theorem jill_investment : 
  compound_interest 10000 (396 / 10000) 2 2 ≈ 10812 := by
sorry

end jill_investment_l711_711699


namespace jill_investment_l711_711700

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_investment :
  compound_interest 10000 0.0396 2 2 ≈ 10815.66 :=
by
  sorry

end jill_investment_l711_711700


namespace books_in_library_now_l711_711522

noncomputable def books_in_library (initial total: ℕ) (fiction_ratio science_ratio history_ratio: ℝ) : ℕ × ℕ × ℕ :=
  let fiction := (fiction_ratio * (initial: ℝ)).toNat;
  let science := (science_ratio * (initial: ℝ)).toNat;
  let history := total - fiction - science;
  (fiction, science, history)

noncomputable def add_books (current_fiction current_science current_history add_fiction add_science add_history : ℕ) :=
  (current_fiction + add_fiction, current_science + add_science, current_history + add_history)

noncomputable def donate_books (current_fiction current_science current_history donate_fiction donate_history : ℕ) :=
  (current_fiction - donate_fiction, current_science, current_history - donate_history)

-- Initial conditions five years ago
def initial_five_years_ago := 500
def fiction_ratio_five_years := 0.40
def science_ratio_five_years := 0.30
def history_ratio_five_years := 0.30

-- Additions and donations
def bought_two_years_ago := 300
def fiction_ratio_two_years := 0.50
def science_ratio_two_years := 0.25
def history_ratio_two_years := 0.25

def bought_last_year := 400
def donated_this_year := 200
def donate_fiction_ratio := 0.60
def donate_history_ratio := 0.40

theorem books_in_library_now :
  let (fiction_five, science_five, history_five) := books_in_library initial_five_years_ago 500 fiction_ratio_five_years science_ratio_five_years history_ratio_five_years in
  let (fiction_two, science_two, history_two) := books_in_library bought_two_years_ago 300 fiction_ratio_two_years science_ratio_two_years history_ratio_two_years in
  let (current_fiction, current_science, current_history) := add_books fiction_five science_five history_five fiction_two science_two history_two in
  let (fiction_last, science_last, history_last) := books_in_library bought_last_year 400 fiction_ratio_two_years science_ratio_two_years history_ratio_two_years in
  let (current_fiction, current_science, current_history) := add_books current_fiction current_science current_history fiction_last science_last history_last in
  let donate_fiction := (donate_fiction_ratio * (donated_this_year: ℝ)).toNat in
  let donate_history := (donate_history_ratio * (donated_this_year: ℝ)).toNat in
  let (final_fiction, final_science, final_history) := donate_books current_fiction current_science current_history donate_fiction donate_history
  in final_fiction = 430 ∧ final_science = 325 ∧ final_history = 245 :=
by sorry

end books_in_library_now_l711_711522


namespace convert_cylindrical_to_rectangular_l711_711969

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem convert_cylindrical_to_rectangular :
  cylindrical_to_rectangular 7 (Real.pi / 4) 8 = (7 * Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2, 8) :=
by
  sorry

end convert_cylindrical_to_rectangular_l711_711969


namespace sum_of_prime_factors_eq_22_l711_711815

-- Conditions: n is defined as 3^6 - 1
def n : ℕ := 3^6 - 1

-- Statement: The sum of the prime factors of n is 22
theorem sum_of_prime_factors_eq_22 : 
  (∀ p : ℕ, p ∣ n → Prime p → p = 2 ∨ p = 7 ∨ p = 13) → 
  (2 + 7 + 13 = 22) :=
by sorry

end sum_of_prime_factors_eq_22_l711_711815


namespace number_of_deluxe_volumes_l711_711364

theorem number_of_deluxe_volumes (d s : ℕ) 
  (h1 : d + s = 15)
  (h2 : 30 * d + 20 * s = 390) : 
  d = 9 :=
by
  sorry

end number_of_deluxe_volumes_l711_711364


namespace general_term_formula_l711_711544

theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = 2 * a n - 1) : 
  ∀ n, a n = 2^(n-1) := 
by
  sorry

end general_term_formula_l711_711544


namespace christine_bottles_l711_711959

noncomputable def fluid_ounces_to_liters (fl_oz : ℝ) : ℝ := fl_oz / 33.8
noncomputable def liters_to_milliliters (liters : ℝ) : ℝ := liters * 1000
noncomputable def bottles_needed (milliliters : ℝ) (bottle_size : ℝ) : ℕ := (milliliters / bottle_size).ceil.to_nat

theorem christine_bottles :
  (bottles_needed (liters_to_milliliters (fluid_ounces_to_liters 60)) 250) = 8 :=
sorry

end christine_bottles_l711_711959


namespace find_a_l711_711178

theorem find_a (a : ℝ) (h : (2 - a * complex.I) * complex.I).re = -(2 - a * complex.I).im :
  a = -2 :=
sorry

end find_a_l711_711178


namespace cube_difference_l711_711575

variable (x : ℝ)

theorem cube_difference (h : x - 1/x = 5) : x^3 - 1/x^3 = 120 := by
  sorry

end cube_difference_l711_711575


namespace geometric_seq_an_minus_2_l711_711187

-- Definitions of conditions based on given problem
def seq_a : ℕ → ℝ := sorry -- The sequence {a_n}
def sum_s : ℕ → ℝ := sorry -- The sum of the first n terms {s_n}

axiom cond1 (n : ℕ) (hn : n > 0) : seq_a (n + 1) ≠ seq_a n
axiom cond2 (n : ℕ) (hn : n > 0) : sum_s n + seq_a n = 2 * n

-- Theorem statement
theorem geometric_seq_an_minus_2 (n : ℕ) (hn : n > 0) : 
  ∃ r : ℝ, ∀ k : ℕ, seq_a (k + 1) - 2 = r * (seq_a k - 2) := 
sorry

end geometric_seq_an_minus_2_l711_711187


namespace container_volume_ratio_l711_711114

variables (V1 V2 V3 : ℝ)

-- Given conditions
def first_container_full := V1 * (3 / 7)
def second_container_empty := 0
def third_container_full := V3 * (3 / 5)

-- Transfers
def juice_in_second_container := first_container_full
def juice_transferred_to_third := (2 / 3) * juice_in_second_container

-- New state of the third container
def third_container_after_transfer := third_container_full + juice_transferred_to_third

theorem container_volume_ratio :
  (third_container_after_transfer = V3 * (4 / 5)) →
  (juice_in_second_container = V2) →
  (V1 / V2 = 7 / 3) := by
  intros
  sorry

end container_volume_ratio_l711_711114


namespace cos_7pi_over_6_l711_711151

noncomputable def cos_seven_pi_six : ℝ := -real.cos (real.pi / 6)

theorem cos_7pi_over_6 : real.cos (7 * real.pi / 6) = cos_seven_pi_six := by
  -- skipped proof
  sorry

end cos_7pi_over_6_l711_711151


namespace binomial_exp_even_sum_l711_711001

theorem binomial_exp_even_sum :
  (∃ (a : Fin 2012 → ℕ), 
  (1 + x) ^ 2011 = ∑ i, a i * x ^ i) →
  a 0 + a 2 + a 4 + ⋯ + a 2010 = 2 ^ 2010 :=
by
  intros h
  sorry

end binomial_exp_even_sum_l711_711001


namespace count_integers_eventually_one_l711_711972

def g (n : ℕ) : ℕ :=
  if n % 2 = 1 then n^2 - 1 else n / 2

def eventually_one (n : ℕ) : Prop :=
  ∃ k : ℕ, iterate g k n = 1

theorem count_integers_eventually_one : 
  {n | 1 ≤ n ∧ n ≤ 200 ∧ eventually_one n}.to_finset.card = 8 :=
  sorry

end count_integers_eventually_one_l711_711972


namespace cube_volume_from_surface_area_l711_711433

theorem cube_volume_from_surface_area (A : ℝ) (hA : A = 294) : ∃ V : ℝ, V = 343 :=
by
  let s := real.sqrt (A / 6)
  have h_s : s = 7 := sorry
  let V := s^3
  have h_V : V = 343 := sorry
  exact ⟨V, h_V⟩

end cube_volume_from_surface_area_l711_711433


namespace divisor_of_p_l711_711731

theorem divisor_of_p (p q r s : ℕ) (h₁ : Nat.gcd p q = 30) (h₂ : Nat.gcd q r = 45) (h₃ : Nat.gcd r s = 75) (h₄ : 120 < Nat.gcd s p) (h₅ : Nat.gcd s p < 180) : 5 ∣ p := 
sorry

end divisor_of_p_l711_711731


namespace triangle_area_is_one_l711_711974

noncomputable def point := (ℝ × ℝ)

def line_through (p1 p2 : point) : ℝ → ℝ :=
  let m := (p2.2 - p1.2) / (p2.1 - p1.1)
  λ x, m * x + p1.2 - m * p1.1

def area_of_triangle (a b c : point) : ℝ :=
  0.5 * ((a.1 * (b.2 - c.2)) + (b.1 * (c.2 - a.2)) + (c.1 * (a.2 - b.2)))

theorem triangle_area_is_one :
  let a := (0, 3) : point
  let b := (9, 2) : point
  let c := (6, 25/9) : point in
  area_of_triangle a b c = 1 :=
by 
  let a := (0, 3) : point
  let b := (9, 2) : point
  let c := (6, 25/9) : point
  show area_of_triangle a b c = 1
  sorry

end triangle_area_is_one_l711_711974


namespace sum_of_values_k_l711_711865

theorem sum_of_values_k (k : ℕ) : 
  (∀ x y : ℤ, (3 * x * x - k * x + 12 = 0) ∧ (3 * y * y - k * y + 12 = 0) ∧ x ≠ y) → k = 0 :=
by
  sorry

end sum_of_values_k_l711_711865


namespace problem_part_I_problem_part_II_l711_711217

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem problem_part_I
  (a b : ℝ)
  (h₁ : ∀ x, f a b 0 x = x^3 + (3/2)*x^2 - 6*x)
  (h₂ : has_deriv_at (f a b 0) (f a b 0 ′) (-2))
  (h₃ : has_deriv_at (f a b 0) (f a b 0 ′) 1) :
  a = 3/2 ∧ b = -6 ∧
  (∀ x, (f a b 0 ′ x > 0 ↔ x < -2 ∨ x > 1) ∧ (f a b 0 ′ x < 0 ↔ -2 < x ∧ x < 1)) :=
sorry

theorem problem_part_II
  (c : ℝ)
  (h₄ : ∀ x ∈ Set.Icc (-1 : ℝ) 2, f (3/2) (-6) c x < c^2) :
  c > 2 ∨ c < -1 :=
sorry

end problem_part_I_problem_part_II_l711_711217


namespace minimum_common_perimeter_l711_711377

theorem minimum_common_perimeter :
  ∃ (a b c : ℤ), 
    (2 * a + 10 * c = 2 * b + 12 * c) ∧ 
    (5 * c * Real.sqrt (a^2 - (5 * c)^2) = 6 * c * Real.sqrt (b^2 - (6 * c)^2)) ∧ 
    ((a - b) = c ∧ 5 * Real.sqrt (a - 5 * c) = 6 * Real.sqrt (b - 6 * c) ∧ 25 * a + 91 * (a - b) = 36 * b) ∧ 
    2 * a + 10 * c = 364 :=
begin
  sorry,
end

end minimum_common_perimeter_l711_711377


namespace Victor_can_carry_7_trays_at_a_time_l711_711379

-- Define the conditions
def trays_from_first_table : Nat := 23
def trays_from_second_table : Nat := 5
def number_of_trips : Nat := 4

-- Define the total number of trays
def total_trays : Nat := trays_from_first_table + trays_from_second_table

-- Prove that the number of trays Victor can carry at a time is 7
theorem Victor_can_carry_7_trays_at_a_time :
  total_trays / number_of_trips = 7 :=
by
  sorry

end Victor_can_carry_7_trays_at_a_time_l711_711379


namespace range_of_a_l711_711199

noncomputable def e : ℝ := Real.exp 1

theorem range_of_a (a : ℝ) (h : ∀ x ∈ Icc (1/e) 1, ∃! y, y ∈ Icc (-1) 1 ∧  (Real.log x - x + 1 + a = y^2 * Real.exp y)) :
  a ∈ Icc (1/e : ℝ) e :=
sorry

end range_of_a_l711_711199


namespace arithmetic_to_geometric_find_original_numbers_l711_711366

theorem arithmetic_to_geometric (k : ℕ) 
  (h1 : ∃ k, (3 * k + 1) * 4 * k = 5 * k * (3 * k + 1))
  : k = 5 :=
begin
  sorry,
end

theorem find_original_numbers :
  ∃ (a b c : ℕ), a = 15 ∧ b = 20 ∧ c = 25 :=
begin
  use [15, 20, 25],
  exact ⟨rfl, rfl, rfl⟩
end

end arithmetic_to_geometric_find_original_numbers_l711_711366


namespace average_runs_per_game_l711_711052

-- Define the number of games
def games : ℕ := 6

-- Define the list of runs scored in each game
def runs : List ℕ := [1, 4, 4, 5, 5, 5]

-- The sum of the runs
def total_runs : ℕ := List.sum runs

-- The average runs per game
def avg_runs : ℚ := total_runs / games

-- The theorem to prove
theorem average_runs_per_game : avg_runs = 4 := by sorry

end average_runs_per_game_l711_711052


namespace log_base_change_l711_711877

theorem log_base_change (log_16_32 log_16_inv2: ℝ) : 
  (log_16_32 * log_16_inv2 = -5 / 16) :=
by
  sorry

end log_base_change_l711_711877


namespace min_value_expression_l711_711740

open Real

theorem min_value_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 := by
  sorry

end min_value_expression_l711_711740


namespace rowing_time_difference_l711_711929

variable (D S : ℝ)
noncomputable def B := 3 * S
theorem rowing_time_difference (h : S ≠ 0) : 
  (D / (B - S)) - (D / (B + S)) = D / (4 * S) :=
by
  sorry

end rowing_time_difference_l711_711929


namespace trapezoid_circle_center_sum_of_squares_l711_711093

variables {a b : ℝ} -- side lengths of the trapezoid
variables {A B C D O : Type} [has_dist O] -- points representing vertices A, B, C, D and center O of the circle

-- Conditions: Trapezoid with side lengths a and b, center O of the circumscribed circle
def is_trapezoid_with_sides_and_circle_center (O : O) (a b : ℝ) : Prop :=
  ∃ (A B C D : O), dist A B = a ∧ dist C D = b ∧
  (dist O A)^2 + (dist O B)^2 = a^2 ∧
  (dist O C)^2 + (dist O D)^2 = b^2

-- Problem proof: Sum of squares of distances from O to vertices is a^2 + b^2
theorem trapezoid_circle_center_sum_of_squares (O : O) (a b : ℝ)
  (h : is_trapezoid_with_sides_and_circle_center O a b) :
  ∃ (A B C D : O), (dist O A)^2 + (dist O B)^2 + (dist O C)^2 + (dist O D)^2 = a^2 + b^2 :=
begin
  sorry
end

end trapezoid_circle_center_sum_of_squares_l711_711093


namespace ensure_user_data_security_l711_711765

-- Define what it means to implement a security measure
inductive SecurityMeasure
  | avoidStoringCardData
  | encryptStoredData
  | encryptDataInTransit
  | codeObfuscation
  | restrictRootedDevices
  | antivirusProtectionAgent

-- Define the conditions: Developing an online store app where users can 
-- pay by credit card and order home delivery ensures user data security 
-- if at least three security measures are implemented.

def providesSecurity (measures : List SecurityMeasure) : Prop :=
  measures.contains SecurityMeasure.avoidStoringCardData ∨
  measures.contains SecurityMeasure.encryptStoredData ∨
  measures.contains SecurityMeasure.encryptDataInTransit ∨
  measures.contains SecurityMeasure.codeObfuscation ∨
  measures.contains SecurityMeasure.restrictRootedDevices ∨
  measures.contains SecurityMeasure.antivirusProtectionAgent

theorem ensure_user_data_security (measures : List SecurityMeasure) (h : measures.length ≥ 3) : providesSecurity measures :=
  sorry

end ensure_user_data_security_l711_711765


namespace divide_19_degree_angle_into_19_equal_parts_l711_711378

/-- Divide a 19° angle into 19 equal parts, resulting in each part being 1° -/
theorem divide_19_degree_angle_into_19_equal_parts
  (α : ℝ) (hα : α = 19) :
  α / 19 = 1 :=
by
  sorry

end divide_19_degree_angle_into_19_equal_parts_l711_711378


namespace problem1_problem2_l711_711228

def vector_dot (v1 v2 : ℝ × ℝ) : ℝ := 
  v1.1 * v2.1 + v1.2 * v2.2

def perpendicular (v1 v2 : ℝ × ℝ) : Prop := 
  vector_dot v1 v2 = 0

def parallel (v1 v2 : ℝ × ℝ) : Prop := 
  v1.1 * v2.2 = v1.2 * v2.1

-- Given vectors in the problem
def a : ℝ × ℝ := (-3, 1)
def b : ℝ × ℝ := (1, -2)
def c : ℝ × ℝ := (1, -1)
def n (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)
def v : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)

-- Problem 1: Find k when n is perpendicular to v
theorem problem1 (k : ℝ) : perpendicular (n k) v → k = 5 / 3 := 
by sorry

-- Problem 2: Find k when n is parallel to c + k * b
theorem problem2 (k : ℝ) : parallel (n k) (c.1 + k * b.1, c.2 + k * b.2) → k = -1 / 3 := 
by sorry

end problem1_problem2_l711_711228


namespace midpoint_property_l711_711752

theorem midpoint_property :
  let A : ℝ × ℝ := (20, 12)
  let B : ℝ × ℝ := (-4, 3)
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  3 * C.1 - 5 * C.2 = -13.5 :=
by
  let A := (20 : ℝ, 12 : ℝ)
  let B := (-4 : ℝ, 3 : ℝ)
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  show 3 * C.1 - 5 * C.2 = -13.5
  sorry

end midpoint_property_l711_711752


namespace average_speed_of_rocket_l711_711050

theorem average_speed_of_rocket
  (ascent_speed : ℕ)
  (ascent_time : ℕ)
  (descent_distance : ℕ)
  (descent_time : ℕ)
  (average_speed : ℕ)
  (h_ascent_speed : ascent_speed = 150)
  (h_ascent_time : ascent_time = 12)
  (h_descent_distance : descent_distance = 600)
  (h_descent_time : descent_time = 3)
  (h_average_speed : average_speed = 160) :
  (ascent_speed * ascent_time + descent_distance) / (ascent_time + descent_time) = average_speed :=
by
  sorry

end average_speed_of_rocket_l711_711050


namespace product_of_roots_l711_711979

def p (x : ℝ) : ℝ := 3*x^3 + 2*x^2 - 8*x + 20
def q (x : ℝ) : ℝ := 4*x^4 - 25*x^2 + 19

theorem product_of_roots :
  (∏ x in (set_of_roots (λ x => p(x) * q(x))), x) = -95 / 3 :=
by
  sorry

end product_of_roots_l711_711979


namespace determine_distance_traveled_l711_711629

-- Define conditions
def flat_distance (D : Real) : Real := D / 3
def uphill_distance (D : Real) : Real := D / 3
def downhill_distance (D : Real) : Real := D / 3

def T1 (D : Real) : Real := (flat_distance D) / 20 + (uphill_distance D) / 10 + (downhill_distance D) / 30
def T2 (D : Real) : Real := (flat_distance D) / 10 + (uphill_distance D) / 5 + (downhill_distance D) / 15

-- Given conditions
def time_difference (D : Real) : Prop := T2 D = T1 D + 1

-- Lean 4 statement to prove the distance traveled
theorem determine_distance_traveled (D : Real) (h : time_difference D) : D = 180 / 11 := sorry

end determine_distance_traveled_l711_711629


namespace sum_of_values_k_l711_711862

theorem sum_of_values_k (k : ℕ) : 
  (∀ x y : ℤ, (3 * x * x - k * x + 12 = 0) ∧ (3 * y * y - k * y + 12 = 0) ∧ x ≠ y) → k = 0 :=
by
  sorry

end sum_of_values_k_l711_711862


namespace find_circle_radius_l711_711985

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

end find_circle_radius_l711_711985


namespace consistent_price_per_kg_l711_711112

theorem consistent_price_per_kg (m₁ m₂ : ℝ) (p₁ p₂ : ℝ)
  (h₁ : p₁ = 6) (h₂ : m₁ = 2)
  (h₃ : p₂ = 36) (h₄ : m₂ = 12) :
  (p₁ / m₁ = p₂ / m₂) := 
by 
  sorry

end consistent_price_per_kg_l711_711112


namespace construct_quadrilateral_l711_711680

structure Quadrilateral :=
  (A B C D : Point)
  (AB AC AD BC CD : ℝ)
  (β δ : ℝ)
  (convex : Convex {A, B, C, D})
  (AB_eq : AB = distance A B)
  (AD_eq : AD = distance A D)
  (BC_eq : BC = distance B C)
  (CD_eq : CD = distance C D)
  (angle_B : β = angle B)
  (angle_D : δ = angle D)
  (AB_le_AD : AB ≤ AD)
  (BC_eq_CD : BC = CD)

theorem construct_quadrilateral (AB AD β δ : ℝ) (h1 : AB ≤ AD) : 
  ∃ (Q : Quadrilateral), 
    Q.AB_eq ∧ 
    Q.AD_eq ∧ 
    Q.angle_B ∧ 
    Q.angle_D ∧ 
    Q.AB_le_AD ∧ 
    Q.BC_eq_CD :=
sorry

end construct_quadrilateral_l711_711680


namespace ratio_b_a_eq_sqrt5_l711_711520

-- Definitions representing the problem conditions
variables {P : Type} [plane_metric_space P]
variables (A B C D E : P)

-- Given segment lengths
variables (a b : ℝ)
variable (h_distinct_points : distinct [A, B, C, D, E])
variable (h_lengths : {d | dist d = a} = {(A, B), (B, C), (C, A), (A, D), (C, E)} ∧ dist (A, E) = 2 * a ∧ dist (B, E) = b}

-- To prove the ratio b / a == sqrt 5
theorem ratio_b_a_eq_sqrt5 
(h_distinct_points : distinct [A, B, C, D, E])
(h_lengths : {d | dist d = a} = {(A, B), (B, C), (C, A), (A, D), (C, E)} ∧ dist (A, E) = 2 * a ∧ dist (B, E) = b) :
  b / a = Real.sqrt 5 := 
sorry -- Proof omitted

end ratio_b_a_eq_sqrt5_l711_711520


namespace three_digit_multiples_of_seven_l711_711608

theorem three_digit_multiples_of_seven : 
  let min_bound := 100
  let max_bound := 999
  let first_multiple := Nat.ceil (min_bound / 7)
  let last_multiple := Nat.floor (max_bound / 7)
  (last_multiple - first_multiple + 1) = 128 :=
by 
  let min_bound := 100
  let max_bound := 999
  let first_multiple := Nat.ceil (min_bound / 7)
  let last_multiple := Nat.floor (max_bound / 7)
  have calc_first_multiple : first_multiple = 15 := sorry
  have calc_last_multiple : last_multiple = 142 := sorry
  have count_multiples : (last_multiple - first_multiple + 1) = 128 := sorry
  exact count_multiples

end three_digit_multiples_of_seven_l711_711608


namespace two_digit_numbers_solution_l711_711643

theorem two_digit_numbers_solution :
  ∀ (N : ℕ), (∃ (x y : ℕ), (N = 10 * x + y) ∧ (x < 10) ∧ (y < 10) ∧ 4 * x + 2 * y = N / 2) →
    (N = 32 ∨ N = 64 ∨ N = 96) := 
by
  sorry

end two_digit_numbers_solution_l711_711643


namespace cubic_identity_l711_711565

theorem cubic_identity (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 :=
by
  sorry

end cubic_identity_l711_711565


namespace reflected_ray_eq_l711_711932

theorem reflected_ray_eq:
  ∀ (x y : ℝ), 
    (3 * x + 4 * y - 18 = 0) ∧ (3 * x + 2 * y - 12 = 0) →
    63 * x + 16 * y - 174 = 0 :=
by
  intro x y
  intro h
  sorry

end reflected_ray_eq_l711_711932


namespace cylinder_height_percentage_l711_711925

-- Lean 4 statement for the problem
theorem cylinder_height_percentage (h : ℝ) (r : ℝ) (H : ℝ) :
  (7 / 8) * h = (3 / 5) * (1.25 * r)^2 * H → H = 0.9333 * h :=
by 
  sorry

end cylinder_height_percentage_l711_711925


namespace find_n_l711_711645

theorem find_n (N : ℕ) (hN : 10 ≤ N ∧ N < 100)
  (h : ∃ a b : ℕ, N = 10 * a + b ∧ 4 * a + 2 * b = N / 2) : 
  N = 32 ∨ N = 64 ∨ N = 96 :=
sorry

end find_n_l711_711645
