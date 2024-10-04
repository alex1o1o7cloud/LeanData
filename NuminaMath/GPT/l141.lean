import Mathlib

namespace order_of_a_b_c_l141_141299

def a := Real.exp ( 0.3 * Real.ln 7 )
def b := 0.3 ^ 7
def c := Real.ln 0.3

theorem order_of_a_b_c : c < b ∧ b < a :=
by {
  -- proof goes here
  sorry
}

end order_of_a_b_c_l141_141299


namespace range_of_a_l141_141565

theorem range_of_a (a : ℝ) :
  (∃ x ∈ (0, 1), 2 * a * x^2 - x - 1 = 0) ↔ a ∈ Set.Ioi 1 :=
sorry

end range_of_a_l141_141565


namespace length_of_AC_eq_sqrt3_l141_141980

-- Define the triangle and its properties
variable (A B C: Point)
variable (angle_B: ℝ)
variable (BC: ℝ)
variable (area_ABC: ℝ)

-- Given conditions
axiom BC_eq_2 : BC = 2
axiom angle_B_eq_60 : angle_B = π/3 -- 60 degrees in radians
axiom area_ABC_eq_sqrt3_div_2 : area_ABC = (Real.sqrt 3) / 2

-- Definition of side_AC
noncomputable def side_AC (A B C: Point) : ℝ := sorry

-- The theorem to prove
theorem length_of_AC_eq_sqrt3 :
  BC = 2 →
  angle_B = π/3 →
  area_ABC = (Real.sqrt 3) / 2 →
  (side_AC A B C) = Real.sqrt 3 :=
by
  intros
  have hBC := BC_eq_2
  have hB := angle_B_eq_60
  have hArea := area_ABC_eq_sqrt3_div_2
  sorry

end length_of_AC_eq_sqrt3_l141_141980


namespace sum_T_10_20_31_l141_141842

-- Definition of S_n
def S (n : ℕ) : ℤ :=
  (List.range n).sum (λ i, (-1 : ℤ) ^ i * (i + 1))

-- Definition of T_n as twice S_n
def T (n : ℕ) : ℤ := 2 * S n

-- Prove the sum of T_10, T_20 and T_31 is 2
theorem sum_T_10_20_31 : T 10 + T 20 + T 31 = 2 := by 
  sorry

end sum_T_10_20_31_l141_141842


namespace b_parallel_c_projection_c_on_a_is_correct_l141_141507

-- Definitions of the given vectors
def a : ℝ × ℝ × ℝ := (1, 0, 1)
def b : ℝ × ℝ × ℝ := (-1, 2, -3)
def c : ℝ × ℝ × ℝ := (2, -4, 6)

-- Definition of dot product for 3D vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Prove that vectors b and c are parallel
theorem b_parallel_c : ∃ k : ℝ, c = (k * b.1, k * b.2, k * b.3) :=
  sorry

-- Definition of projection of c onto a
def projection (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dp := dot_product u v
  let u_norm_sq := dot_product u u
  let scale := dp / u_norm_sq
  (scale * u.1, scale * u.2, scale * u.3)

-- Prove that the projection of c onto a is (4, 0, 4)
theorem projection_c_on_a_is_correct : projection c a = (4, (0:ℝ), 4) :=
  sorry

end b_parallel_c_projection_c_on_a_is_correct_l141_141507


namespace mint_plants_initially_zero_l141_141889

theorem mint_plants_initially_zero
  (initial_basils : ℕ = 3)
  (initial_parsley : ℕ = 1)
  (additional_basil : ℕ = 1)
  (final_plants : ℕ = 5)
  (rabbit_ate_all_mint : ℕ) :
  initial_basils + initial_parsley + additional_basil + rabbit_ate_all_mint = final_plants →
  rabbit_ate_all_mint = 0 :=
by
  sorry

end mint_plants_initially_zero_l141_141889


namespace total_pieces_of_art_l141_141041

theorem total_pieces_of_art (pieces_on_display sculptures_not_on_display photographs_on_display pieces_in_storage : ℕ) :
    (2 / 5 : ℝ) * pieces_on_display = 600 →
    (1 / 5 : ℝ) * pieces_on_display + (2 / 7 : ℝ) * pieces_in_storage = 1500 →
    pieces_on_display = 1500 →
    pieces_in_storage = 4200 →
    pieces_on_display + pieces_in_storage = 5700 :=
by
  intros h1 h2 h3 h4,
  sorry

end total_pieces_of_art_l141_141041


namespace infinite_ways_to_bisect_parallelogram_area_l141_141090

theorem infinite_ways_to_bisect_parallelogram_area (P : Type) [parallelogram P] :
  ∃ (f : P → P), set.infinite {l : line P | bisects_area P l} :=
begin
  sorry
end

end infinite_ways_to_bisect_parallelogram_area_l141_141090


namespace tan_of_theta_minus_pi_div_4_l141_141905

noncomputable def vector_a (θ : ℝ) : ℝ × ℝ := (Real.sin θ, 2)
noncomputable def vector_b (θ : ℝ) : ℝ × ℝ := (Real.cos θ, 1)

def collinear (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem tan_of_theta_minus_pi_div_4 (θ : ℝ) 
  (h₁ : ∃ (k : ℝ), vector_a θ = (k * vector_b θ).fst ∧ 
                    (2 : ℝ) = k * (1 : ℝ)) : 
  Real.tan (θ - Real.pi / 4) = 1 / 3 := 
sorry

end tan_of_theta_minus_pi_div_4_l141_141905


namespace determinant_of_matrixA_l141_141843

def matrixA : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![3, 0, -2],
  ![5, 6, -4],
  ![1, 3, 7]
]

theorem determinant_of_matrixA : Matrix.det matrixA = 144 := by
  sorry

end determinant_of_matrixA_l141_141843


namespace domain_of_composite_function_l141_141919

theorem domain_of_composite_function {f : ℝ → ℝ} :
  (∀ x, (1 < x ∧ x < 3) → ∃ y, f y = x) → (∀ x, (1 < 2x - 1 ∧ 2x - 1 < 3) ↔ (1 < x ∧ x < 2)) :=
by
  intro h x
  split
  · intro H
    have H1 : 1 < 2x - 1 := H.1
    have H2 : 2x - 1 < 3 := H.2
    calc
      1 < 2x - 1 := H1
      2 < 2x := by linarith
      1 < x := by linarith
    have : 2x < 4 := by linarith
    have : x < 2 := by linarith
    exact ⟨this_left, this_right⟩
  · intro H
    have : 1 < x := H.1
    have : x < 2 := H.2
    calc
      1 < 2x := by linarith
      2x - 1 < 4 - 1 := by linarith
    exact ⟨this_left, this_right⟩

end domain_of_composite_function_l141_141919


namespace reducibility_implies_divisibility_l141_141666

theorem reducibility_implies_divisibility
  (a b c d l k : ℤ)
  (p q : ℤ)
  (h1 : a * l + b = k * p)
  (h2 : c * l + d = k * q) :
  k ∣ (a * d - b * c) :=
sorry

end reducibility_implies_divisibility_l141_141666


namespace circle_parabola_intersection_l141_141403

theorem circle_parabola_intersection (b : ℝ) : 
  b = 25 / 12 → 
  ∃ (r : ℝ) (cx : ℝ), 
  (∃ p1 p2 : ℝ × ℝ, 
    (p1.2 = 3/4 * p1.1 + b ∧ p2.2 = 3/4 * p2.1 + b) ∧ 
    (p1.2 = 3/4 * p1.1^2 ∧ p2.2 = 3/4 * p2.1^2) ∧ 
    (p1 ≠ (0, 0) ∧ p2 ≠ (0, 0))) ∧ 
  (cx^2 + b^2 = r^2) := 
by 
  sorry

end circle_parabola_intersection_l141_141403


namespace snail_movement_96_days_l141_141423

-- Assuming the necessary definitions and lemmas for handling series and summations are present in Mathlib

def snail_daily_movement (n : ℕ) : ℚ := (1 / n) - (1 / (n + 1))

def snail_total_movement (days : ℕ) : ℚ :=
  ∑ i in Finset.range days, snail_daily_movement (i + 1)

theorem snail_movement_96_days : snail_total_movement 96 = 96 / 97 :=
by
  sorry

end snail_movement_96_days_l141_141423


namespace prime_probability_l141_141338

theorem prime_probability (P : Finset ℕ) : 
  (P = {p | p ≤ 30 ∧ Nat.Prime p}).card = 10 → 
  (Finset.Icc 1 30).card = 30 →
  ((Finset.Icc 1 30).card).choose 2 = 435 →
  (P.card).choose 2 = 45 →
  45 / 435 = 1 / 29 :=
by
  intros hP hIcc30 hChoose30 hChoosePrime
  sorry

end prime_probability_l141_141338


namespace solve_quadratic_eqn_l141_141311

theorem solve_quadratic_eqn (x : ℝ) : 3 * x ^ 2 = 27 ↔ x = 3 ∨ x = -3 :=
by
  sorry

end solve_quadratic_eqn_l141_141311


namespace expected_value_of_coins_flip_l141_141414

noncomputable def expected_value : ℝ :=
  let penny_value := 1 / 2 * 1
  let nickel_value := 1 / 2 * 5
  let dime_value := 1 / 2 * 10
  let quarter_value := 1 / 2 * 25
  let half_dollar_value := 1 / 2 * 50
  in penny_value + nickel_value + dime_value + quarter_value + half_dollar_value

theorem expected_value_of_coins_flip : expected_value = 45.5 := by
  sorry

end expected_value_of_coins_flip_l141_141414


namespace hyperbola_triangle_perimeter_l141_141914

theorem hyperbola_triangle_perimeter :
  let C : set (ℝ × ℝ) := {p | (p.1)^2 / 9 - (p.2)^2 / 16 = 1}
  let F : ℝ × ℝ := (-5, 0)
  let points_on_hyperbola (P Q : ℝ × ℝ) := P ∈ C ∧ Q ∈ C ∧ P.1 > 0 ∧ Q.1 > 0 ∧
    P.1 = 5 ∧ Q.1 = 5 ∧ P.2 = 4 * Real.sqrt 7 ∧ Q.2 = -4 * Real.sqrt 7
  let length_PQ : ℝ := 16
  let point_A : ℝ × ℝ := (5, 0)
  (A_on_PQ : ∃ P Q, points_on_hyperbola P Q ∧ point_A.1 = P.1 ∧ point_A.2 = P.2) →
  let PF := Real.sqrt ((F.1 + 5)^2 + (4 * Real.sqrt 7)^2)
  let QF := PF
  let perimeter_PFQ := length_PQ + PF + QF
  perimeter_PFQ = 48 :=
by
  sorry

end hyperbola_triangle_perimeter_l141_141914


namespace exponent_property_l141_141161

variable {a : ℝ} {m n : ℕ}

theorem exponent_property (h1 : a^m = 2) (h2 : a^n = 3) : a^(2*m + n) = 12 :=
sorry

end exponent_property_l141_141161


namespace imaginary_part_of_fraction_l141_141486

theorem imaginary_part_of_fraction : 
  let z := (1 - complex.I) / (1 + complex.I) in 
  complex.im z = -1 :=
by
  let z : ℂ := (1 - complex.I) / (1 + complex.I)
  sorry

end imaginary_part_of_fraction_l141_141486


namespace determinant_is_one_l141_141862

def matrix := λ (α β γ : ℝ), matrix.reindex (fin ∅) (fin ∅) 
  ![
    ![real.cos (α + γ) * real.cos β, real.cos (α + γ) * real.sin β, -real.sin (α + γ)],
    ![-real.sin β, real.cos β, 0],
    ![real.sin (α + γ) * real.cos β, real.sin (α + γ) * real.sin β, real.cos (α + γ)]
  ]

theorem determinant_is_one (α β γ: ℝ):
  matrix.det (matrix α β γ) = 1 := 
by
  sorry

end determinant_is_one_l141_141862


namespace sum_integers_minus50_to_80_l141_141368

theorem sum_integers_minus50_to_80 : (Finset.sum (Finset.range (80 + 51)) - Finset.sum (Finset.range 50)) = 1965 :=
by
  sorry

end sum_integers_minus50_to_80_l141_141368


namespace rectangle_percentage_excess_l141_141193

variable (L W : ℝ) -- The lengths of the sides of the rectangle
variable (x : ℝ) -- The percentage excess for the first side (what we want to prove)

theorem rectangle_percentage_excess 
    (h1 : W' = W * 0.95)                    -- Condition: second side is taken with 5% deficit
    (h2 : L' = L * (1 + x/100))             -- Condition: first side is taken with x% excess
    (h3 : A = L * W)                        -- Actual area of the rectangle
    (h4 : 1.064 = (L' * W') / A) :           -- Condition: error percentage in the area is 6.4%
    x = 12 :=                                -- Proof that x equals 12
sorry

end rectangle_percentage_excess_l141_141193


namespace one_third_recipe_ingredients_l141_141415

noncomputable def cups_of_flour (f : ℚ) := (f : ℚ)
noncomputable def cups_of_sugar (s : ℚ) := (s : ℚ)
def original_recipe_flour := (27 / 4 : ℚ)  -- mixed number 6 3/4 converted to improper fraction
def original_recipe_sugar := (5 / 2 : ℚ)  -- mixed number 2 1/2 converted to improper fraction

theorem one_third_recipe_ingredients :
  cups_of_flour (original_recipe_flour / 3) = (9 / 4) ∧
  cups_of_sugar (original_recipe_sugar / 3) = (5 / 6) :=
by
  sorry

end one_third_recipe_ingredients_l141_141415


namespace sum_of_series_7_l141_141496

noncomputable def sum_series_in_base7 : nat :=
  let n := 24 in -- 33_7 in base 10
  let S := n * (n + 1) / 2 in
  let S_base10 := 300 in
  let S_base7 := 606 in
  S_base7

theorem sum_of_series_7 (S : nat) : S = sum_series_in_base7 :=
by sorry

end sum_of_series_7_l141_141496


namespace sandy_bought_6_more_fish_l141_141260

theorem sandy_bought_6_more_fish (initial_fish : ℕ) (final_fish : ℕ) (h1 : initial_fish = 26) (h2 : final_fish = 32) :
  final_fish - initial_fish = 6 :=
by {
  rw [h1, h2], -- apply the given conditions
  simp, -- simplify the equation
  sorry -- complete the proof
}

end sandy_bought_6_more_fish_l141_141260


namespace acute_triangle_product_inequality_l141_141190

theorem acute_triangle_product_inequality
  (ABC : Triangle)
  (acute : is_acute_angled ABC)
  (O : Point)
  (circumcenter_ABC : is_circumcenter O ABC)
  (R : ℝ)
  (radius_ABC : is_circumradius R ABC)
  (A' B' C' : Point)
  (A'_intersection : is_intersection_of_extension_and_circumcircle O A' (circumcircle B O C))
  (B'_intersection : is_intersection_of_extension_and_circumcircle O B' (circumcircle A O C))
  (C'_intersection : is_intersection_of_extension_and_circumcircle O C' (circumcircle A O B)) :
  OA' * OB' * OC' ≥ 8 * R ^ 3 := 
sorry

end acute_triangle_product_inequality_l141_141190


namespace ernie_circles_l141_141821

theorem ernie_circles (boxes_per_circle_ali boxes_per_circle_ernie total_boxes ali_circles : ℕ)
  (h1: boxes_per_circle_ali = 8)
  (h2: boxes_per_circle_ernie = 10)
  (h3: total_boxes = 80)
  (h4: ali_circles = 5) : 
  (total_boxes - ali_circles * boxes_per_circle_ali) / boxes_per_circle_ernie = 4 :=
by
  sorry

end ernie_circles_l141_141821


namespace min_sum_value_l141_141070

theorem min_sum_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b) + b / (6 * c) + c / (9 * a)) ≥ (1 / real.cbrt 2) := 
begin
  -- Here the proof would go
  sorry,
end

end min_sum_value_l141_141070


namespace minimum_omega_l141_141616

noncomputable section

-- Define the function f and its properties
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.cos (ω * x + φ)

-- Assumptions based on the given conditions
variables {ω φ : ℝ}
variables (T : ℝ) (hω_pos : 0 < ω) (hφ_range : 0 < φ ∧ φ < π)
variables (hT : f T ω φ = Real.sqrt 3 / 2)
variables (hx_zero : f (π / 9) ω φ = 0)

-- Prove the minimum value of ω is 3
theorem minimum_omega : ω = 3 := sorry

end minimum_omega_l141_141616


namespace sum_f_1_to_2014_eq_neg_807_l141_141238

noncomputable def f : ℝ → ℝ :=
λ x, if -3 < x ∧ x <= -1 then x else if -1 < x ∧ x <= 2 then (x - 1) ^ 2 else f (x % 5)

lemma periodic (x : ℝ) : f(x + 5) = f(x) := sorry

lemma f_interval_1 (x : ℝ) (h : -3 < x ∧ x <= -1) : f(x) = x := sorry

lemma f_interval_2 (x : ℝ) (h : -1 < x ∧ x <= 2) : f(x) = (x - 1) ^ 2 := sorry

theorem sum_f_1_to_2014_eq_neg_807 : 
  (∑ i in finset.range 2014 \ finset.range 1, f (i : ℝ)) = -807 :=
sorry

end sum_f_1_to_2014_eq_neg_807_l141_141238


namespace physics_textbooks_same_box_l141_141676

-- Definition of the box capacities and the textbooks.
def box_capacities := [4, 5, 3, 4]
def total_textbooks := 16
def physics_textbooks := 4

-- Prove the probability in question.
theorem physics_textbooks_same_box :
  let total_ways := (nat.choose 16 4) * (nat.choose 12 5) * (nat.choose 7 3) * (nat.choose 4 4)
      favorable_ways := (nat.choose 12 0) + (nat.choose 12 1) + (nat.choose 12 0)
      probability := (14 : ℚ) / (total_ways : ℚ)
  in
  ∃ m n : ℕ, nat.gcd m n = 1 ∧ probability = (3 : ℚ) / (286 : ℚ) ∧ m + n = 289 :=
begin
  -- Calculation details skipped
  sorry
end

end physics_textbooks_same_box_l141_141676


namespace definite_integral_eq_zero_l141_141384

noncomputable def integrand (x : ℝ) : ℝ :=
  (4 * Real.sqrt (1 - x) - Real.sqrt (2 * x + 1)) / ((Real.sqrt (2 * x + 1) + 4 * Real.sqrt (1 - x)) * (2 * x + 1)^2)

theorem definite_integral_eq_zero :
  ∫ x in 0..1, integrand x = 0 := 
by
  sorry

end definite_integral_eq_zero_l141_141384


namespace Marge_personal_spending_l141_141241

-- Definitions based on the problem statement
def initial_winnings := 50000

def taxes_percentage := 0.60
def mortgage_percentage := 0.50
def retirement_fund_percentage := 0.40
def retirement_fund_interest_rate := 0.05
def college_fund_percentage := 0.25

def savings := 1500
def stock_market_investment_percentage := 0.60
def stock_market_return_rate := 0.07

-- Lean theorem statement
theorem Marge_personal_spending:
  let after_taxes := initial_winnings * (1 - taxes_percentage),
      after_mortgage := after_taxes * (1 - mortgage_percentage),
      after_retirement_fund := after_mortgage * (1 - retirement_fund_percentage),
      after_college_fund := after_retirement_fund * (1 - college_fund_percentage),
      stock_market_investment := savings * stock_market_investment_percentage,
      growth_retirement := (after_retirement_fund * retirement_fund_percentage) * retirement_fund_interest_rate,
      growth_stock_market := stock_market_investment * stock_market_return_rate
  in
  after_college_fund + (savings - stock_market_investment) + growth_retirement + growth_stock_market = 5363 :=
sorry

end Marge_personal_spending_l141_141241


namespace volume_ratio_regular_tetrahedron_to_cube_l141_141060

noncomputable def volume_tetrahedron (s : ℝ) : ℝ := (s^3 * real.sqrt 2) / 12

noncomputable def centroid_to_vertex_ratio : ℝ := 2 / 3

noncomputable def cube_side_length (s : ℝ) : ℝ := s * real.sqrt 6 / 9

noncomputable def volume_cube (s : ℝ) : ℝ := (cube_side_length s)^3

theorem volume_ratio_regular_tetrahedron_to_cube (s : ℝ) : 
  let VT := volume_tetrahedron s,
      VC := volume_cube s in
  VT / VC = 27 / 8 ∧ (nat.gcd 27 8 = 1) → 27 + 8 = 35 :=
by
  intros VT VC h
  sorry

end volume_ratio_regular_tetrahedron_to_cube_l141_141060


namespace parabola_shape_of_vertices_l141_141610

def vertices_parabola (a t c : ℝ) : ℝ × ℝ :=
  let x_v := -t / (2 * a)
  let y_v := a * x_v^2 + t * x_v + c
  (x_v, y_v)

theorem parabola_shape_of_vertices : 
  ∀ (t : ℝ), 
    let (a, c) := (2 : ℝ, 6 : ℝ) in
    let (x_v, y_v) := vertices_parabola a t c in
    y_v = -2 * x_v^2 + 6 :=
by
  intros t
  let a := 2 : ℝ
  let c := 6 : ℝ
  let x_v := -t / (2 * a)
  let y_v := a * x_v^2 + t * x_v + c
  have x_v_eq : x_v = -t / 4 := by sorry
  have y_v_eq : y_v = -2 * x_v^2 + 6 := by sorry
  exact y_v_eq

end parabola_shape_of_vertices_l141_141610


namespace necessary_condition_l141_141527

theorem necessary_condition (x : ℝ) (h : (x-1) * (x-2) ≤ 0) : x^2 - 3 * x ≤ 0 :=
sorry

end necessary_condition_l141_141527


namespace greatest_value_of_x_l141_141178

theorem greatest_value_of_x (x : ℤ) (h : 2.134 * 10^x < 240000) : x ≤ 5 :=
by sorry

end greatest_value_of_x_l141_141178


namespace symmetric_point_equation_l141_141536

theorem symmetric_point_equation
  (x y : ℝ)
  (hx : x^2 - y^2 = 1)
  (hx' : x^2 + y^2 ≠ 0)
  (symmetry_condition : ∀ t : ℝ, t * (x^2 + y^2) = 1) :
  x^2 - y^2 = (x^2 + y^2)^2 :=
begin
  sorry
end

end symmetric_point_equation_l141_141536


namespace find_a_quadratic_trinomials_l141_141869

theorem find_a_quadratic_trinomials :
  ∃ a : ℝ, 
    (∀ f g : polynomial ℝ, 
      f = polynomial.C (4 * a) + polynomial.C (-6) * polynomial.X + polynomial.X ^ 2 ∧
      g = polynomial.C 6 + polynomial.C a * polynomial.X + polynomial.X ^ 2 ∧
      f.discriminant > 0 ∧
      g.discriminant > 0 ∧
      (let x1, x2 := ((-(-6)) ± (sqrt ((-6)^2 - 4 * 1 * (4 * a)))) / (2 * 1),
           y1, y2 := ((-a) ± (sqrt (a^2 - 4 * 1 * 6))) / (2 * 1)
       in x1^2 + x2^2 = y1^2 + y2^2)
    ) ∧ a = -12 :=
  sorry

end find_a_quadratic_trinomials_l141_141869


namespace B_necessary_not_sufficient_for_A_l141_141269

def A (x : ℝ) : Prop := 0 < x ∧ x < 5
def B (x : ℝ) : Prop := |x - 2| < 3

theorem B_necessary_not_sufficient_for_A (x : ℝ) :
  (A x → B x) ∧ (∃ x, B x ∧ ¬ A x) :=
by
  sorry

end B_necessary_not_sufficient_for_A_l141_141269


namespace find_p_l141_141995

/-- Given the points Q(0, 15), A(3, 15), B(15, 0), O(0, 0), and C(0, p).
The area of triangle ABC is given as 45.
We need to prove that p = 11.25. -/
theorem find_p (ABC_area : ℝ) (p : ℝ) (h : ABC_area = 45) :
  p = 11.25 :=
by
  sorry

end find_p_l141_141995


namespace volume_of_cone_l141_141567

noncomputable def cone_volume (lateral_area : ℝ) (angle: ℝ): ℝ :=
let r := lateral_area / (20 * Mathlib.pi * Math.cos angle) in
let l := r * (Math.tan angle) in
let h := Math.sqrt (l^2 - r^2) in
(1/3) * Mathlib.pi * r^2 * h

theorem volume_of_cone (lateral_area : ℝ) (angle: ℝ) (h_angle: angle = Mathlib.arccos (4/5)) 
(h_lateral_area: lateral_area = 20 * Mathlib.pi): cone_volume lateral_area angle = 16 * Mathlib.pi :=
by
  sorry

end volume_of_cone_l141_141567


namespace find_x_l141_141560

noncomputable def leastCommonMultiple (a b : ℕ) : ℕ :=
  a * b / (Nat.gcd a b)

noncomputable def lcm_of_10_to_15 : ℕ :=
  leastCommonMultiple 10 (leastCommonMultiple 11 (leastCommonMultiple 12 (leastCommonMultiple 13 (leastCommonMultiple 14 15))))

theorem find_x :
  (lcm_of_10_to_15 / 2310 = 26) := by
  sorry

end find_x_l141_141560


namespace fractions_are_integers_l141_141246

theorem fractions_are_integers (x y : ℕ) 
    (h : ∃ k : ℤ, (x^2 - 1) / (y + 1) + (y^2 - 1) / (x + 1) = k) :
    ∃ u v : ℤ, (x^2 - 1) = u * (y + 1) ∧ (y^2 - 1) = v * (x + 1) := 
by
  sorry

end fractions_are_integers_l141_141246


namespace conical_well_volume_l141_141460

noncomputable def volume_of_conical_well (d1 d2 h : ℝ) : ℝ :=
  let R := d1 / 2
  let r := d2 / 2
  (1 / 3) * Real.pi * h * (R^2 + r^2 + R * r)

theorem conical_well_volume :
  volume_of_conical_well 6 3 24 ≈ 395.84 :=
by
  -- We state that given the defined parameters,
  -- the computed volume is approximately equal to 395.84 cubic meters.
  sorry

end conical_well_volume_l141_141460


namespace winner_more_votes_l141_141200

variable (totalStudents : ℕ) (votingPercentage : ℤ) (winnerPercentage : ℤ) (loserPercentage : ℤ)

theorem winner_more_votes
    (h1 : totalStudents = 2000)
    (h2 : votingPercentage = 25)
    (h3 : winnerPercentage = 55)
    (h4 : loserPercentage = 100 - winnerPercentage)
    (h5 : votingStudents = votingPercentage * totalStudents / 100)
    (h6 : winnerVotes = winnerPercentage * votingStudents / 100)
    (h7 : loserVotes = loserPercentage * votingStudents / 100)
    : winnerVotes - loserVotes = 50 := by
  sorry

end winner_more_votes_l141_141200


namespace sufficient_but_not_necessary_condition_l141_141127

def circle (x y : ℝ) : Prop := x^2 + y^2 = 2

def line (x y a : ℝ) : Prop := x - 2 * y + a = 0

def orthogonal_vectors (x1 y1 x2 y2 : ℝ) :=
  x1 * x2 + y1 * y2 = 0

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  a = Real.sqrt 5 →
  (∃ x1 y1 x2 y2 : ℝ, circle x1 y1 ∧ circle x2 y2 ∧ line x1 y1 a ∧ line x2 y2 a ∧ orthogonal_vectors x1 y1 x2 y2) :=
sorry

end sufficient_but_not_necessary_condition_l141_141127


namespace correct_option_l141_141774

/-- Define sets of line segments --/
def segments (A B C D : list ℕ) :=
  A = [1, 3, 4] ∧ B = [2, 2, 7] ∧ C = [4, 5, 7] ∧ D = [3, 3, 6]

/-- Define the triangle inequality theorem --/
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Prove that only option C satisfies the triangle inequality --/
theorem correct_option (A B C D : list ℕ) (h : segments A B C D) :
  triangle_inequality C[0] C[1] C[2] =
  true :=
begin
  sorry
end

end correct_option_l141_141774


namespace value_of_a_l141_141719

-- Declare and define the given conditions.
def line1 (y : ℝ) := y = 13
def line2 (x t y : ℝ) := y = 3 * x + t

-- Define the proof statement.
theorem value_of_a (a b t : ℝ) (h1 : line1 b) (h2 : line2 a t b) (ht : t = 1) : a = 4 :=
by
  sorry

end value_of_a_l141_141719


namespace most_likely_outcome_l141_141049

/-- Definition of being a boy or girl -/
inductive Gender
| boy : Gender
| girl : Gender

/-- Probability of each gender -/
def prob_gender (g : Gender) : ℚ :=
  1 / 2

/-- Calculate the probability of all children being a specific gender -/
def prob_all_same_gender (n : ℕ) (g : Gender) : ℚ :=
  (prob_gender g) ^ n

/-- Number of ways to choose k girls out of n children -/
def num_ways_k_girls_out_of_n (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Calculate the probability of having exactly k girls out of n children -/
def prob_exactly_k_girls (n k : ℕ) : ℚ :=
  (num_ways_k_girls_out_of_n n k) * (prob_gender Gender.girl) ^ k * (prob_gender Gender.boy) ^ (n - k)

theorem most_likely_outcome :
  let A := prob_all_same_gender 8 Gender.boy,
      B := prob_all_same_gender 8 Gender.girl,
      C := prob_exactly_k_girls 8 4,
      D := prob_exactly_k_girls 8 6 + prob_exactly_k_girls 8 2 in
  C > A ∧ C > B ∧ C > D :=
by
  sorry

end most_likely_outcome_l141_141049


namespace probability_college_graduate_with_degree_l141_141381

theorem probability_college_graduate_with_degree
  (G C N : ℕ)
  (h1 : G * 8 = N)
  (h2 : C * 3 = 2 * N) :
  G = 3 → C = 16 → N = 24 → 
  (G.toRat / (G + C).toRat = 3 / 19) :=
by
  sorry

end probability_college_graduate_with_degree_l141_141381


namespace sequence_initial_term_l141_141304

theorem sequence_initial_term :
  (∀ n ≥ 2, (finset.range n).sum (λ i, a (i + 1)) = n ^ 2 * a n) ∧ a 63 = 1 → a 1 = 2016 :=
begin
  apologize,
end

end sequence_initial_term_l141_141304


namespace rhombus_midpoint_distance_squared_l141_141195

theorem rhombus_midpoint_distance_squared :
  ∀ (A B C D X Y : ℝ × ℝ)
    (hAB : |A - B| = 13)
    (hBC : |B - C| = 13)
    (hCD : |C - D| = 13)
    (hDA : |D - A| = 13)
    (hBAC : ∠BAC = 120)
    (hX : X = midpoint A B)
    (hY : Y = midpoint C D), 
  distance X Y ^ 2 = 126.75 := by
  sorry

end rhombus_midpoint_distance_squared_l141_141195


namespace probability_two_primes_1_to_30_l141_141344

def is_prime (n : ℕ) : Prop := 
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13 ∨ n = 17 ∨ n = 19 ∨ n = 23 ∨ n = 29

def count_combinations (n k : ℕ) : ℕ := nat.choose n k

theorem probability_two_primes_1_to_30 :
  let total_combinations := count_combinations 30 2 in
  let prime_combinations := count_combinations 10 2 in
  total_combinations = 435 ∧ 
  prime_combinations = 45 ∧ 
  (prime_combinations : ℚ) / total_combinations = 15 / 145 :=
by { sorry }

end probability_two_primes_1_to_30_l141_141344


namespace number_of_true_propositions_l141_141929

theorem number_of_true_propositions:
  (∀ x : ℝ, x^2 + 1 > 0) ∧
  (¬ ∀ x : ℕ, x^4 ≥ 1) ∧
  (∃ x : ℤ, x^3 < 1) ∧
  (∀ x : ℚ, x^2 ≠ 2) →
  3 = 3 :=
by
  intros h,
  sorry

end number_of_true_propositions_l141_141929


namespace intersection_of_A_and_B_l141_141519

def A : Set ℤ := { -3, -1, 0, 1 }
def B : Set ℤ := { x | (-2 < x) ∧ (x < 1) }

theorem intersection_of_A_and_B : A ∩ B = { -1, 0 } := by
  sorry

end intersection_of_A_and_B_l141_141519


namespace minimum_omega_is_3_l141_141626

variable (f : ℝ → ℝ)
variable (ω φ T : ℝ)
variable (x : ℝ)

noncomputable def minimum_omega (ω : ℝ) : ℝ := 3 -- Temporary noncomputable definition to hold the value for min omega

theorem minimum_omega_is_3 :
  (∀ x, f x = Real.cos (ω * x + φ)) →
  ω > 0 →
  0 < φ ∧ φ < Real.pi →
  T = (2 * Real.pi) / ω →
  f T = Real.sqrt 3 / 2 →
  f (Real.pi / 9) = 0 →
  ∃(ω_min : ℝ), ω_min = 3 :=
by
  intros h0 h1 h2 h3 h4 h5
  use 3
  sorry -- Proof of the exact answer

end minimum_omega_is_3_l141_141626


namespace linguistic_sequence_n_fact_moves_l141_141397

theorem linguistic_sequence_n_fact_moves (n : ℕ) (w : Fin n → Char) 
  (h_unique : ∀ i j, i ≠ j → w i ≠ w j) 
  (h_reversal : ∃ f : ℕ → List (Fin n), ∀ k, 
    (f (k + 1) = (reverse (f k).take (f k).index) ++ (drop (f k).index (f k)) → 
    ∀ j < k, f (k + 1) ≠ f j)) : 
  (∃ seq : List (List (Fin n)), 
    (∀ i < seq.length, seq.nodup) ∧ seq.length = nat.factorial n) :=
by
  sorry

end linguistic_sequence_n_fact_moves_l141_141397


namespace brass_players_10_l141_141722

theorem brass_players_10:
  ∀ (brass woodwind percussion : ℕ),
    brass + woodwind + percussion = 110 →
    percussion = 4 * woodwind →
    woodwind = 2 * brass →
    brass = 10 :=
by
  intros brass woodwind percussion h1 h2 h3
  sorry

end brass_players_10_l141_141722


namespace calculate_expression_l141_141837

theorem calculate_expression :
  let s := λ (n : ℕ), (nat.iterate (λ x, x + 1) n 1) in
  (1 * 2 * (1 + 2) - 2 * 3 * (2 + 3) + 3 * 4 * (3 + 4) - 
   ... + s 2019 * (s 2019 + 1) * (s 2019 + s 2019 + 1)) = 8242405980 :=
sorry

end calculate_expression_l141_141837


namespace num_pos_int_div_condition_l141_141500

theorem num_pos_int_div_condition : ∃ n : ℕ, ∀ (n < 40 ∧ n > 0), (∃ k : ℕ, n = k * (40 - n)) ↔ (n = 39 ∨ n = 38 ∨ n = 36 ∨ n = 35 ∨ n = 32 ∨ n = 30 ∨ n = 20) ∧ 
    (count (λ n, ∃ k : ℕ, n = k * (40 - n)) (list.range 40)) = 7 := 
sorry

end num_pos_int_div_condition_l141_141500


namespace snail_movement_96_days_l141_141424

-- Assuming the necessary definitions and lemmas for handling series and summations are present in Mathlib

def snail_daily_movement (n : ℕ) : ℚ := (1 / n) - (1 / (n + 1))

def snail_total_movement (days : ℕ) : ℚ :=
  ∑ i in Finset.range days, snail_daily_movement (i + 1)

theorem snail_movement_96_days : snail_total_movement 96 = 96 / 97 :=
by
  sorry

end snail_movement_96_days_l141_141424


namespace red_marbles_more_than_yellow_l141_141740

-- Define the conditions
def total_marbles : ℕ := 19
def yellow_marbles : ℕ := 5
def ratio_parts_blue : ℕ := 3
def ratio_parts_red : ℕ := 4

-- Prove the number of more red marbles than yellow marbles is equal to 3
theorem red_marbles_more_than_yellow :
  let remainder_marbles := total_marbles - yellow_marbles
  let total_parts := ratio_parts_blue + ratio_parts_red
  let marbles_per_part := remainder_marbles / total_parts
  let blue_marbles := ratio_parts_blue * marbles_per_part
  let red_marbles := ratio_parts_red * marbles_per_part
  red_marbles - yellow_marbles = 3 := by
  sorry

end red_marbles_more_than_yellow_l141_141740


namespace sums_of_rows_columns_diagonals_equal_l141_141993

theorem sums_of_rows_columns_diagonals_equal
  (a b c d e f g h i : ℤ)
  (H : a ∈ {-1, 0, 1} ∧ b ∈ {-1, 0, 1} ∧ c ∈ {-1, 0, 1} ∧
       d ∈ {-1, 0, 1} ∧ e ∈ {-1, 0, 1} ∧ f ∈ {-1, 0, 1} ∧
       g ∈ {-1, 0, 1} ∧ h ∈ {-1, 0, 1} ∧ i ∈ {-1, 0, 1}):
  ∃ x y, x ∈ {a + b + c, d + e + f, g + h + i,
               a + d + g, b + e + h, c + f + i,
               a + e + i, c + e + g} ∧ y ∈ {a + b + c, d + e + f, g + h + i,
                                             a + d + g, b + e + h, c + f + i,
                                             a + e + i, c + e + g} ∧ x = y ∧ x ≠ y :=
sorry

end sums_of_rows_columns_diagonals_equal_l141_141993


namespace remainder_97_pow_50_mod_100_l141_141366

theorem remainder_97_pow_50_mod_100 :
  (97 ^ 50) % 100 = 49 := 
by
  sorry

end remainder_97_pow_50_mod_100_l141_141366


namespace negation_of_p_is_neg_p_l141_141169

-- Define the proposition p
def p : Prop := ∀ x : ℝ, x > 3 → x^3 - 27 > 0

-- Define the negation of proposition p
def neg_p : Prop := ∃ x : ℝ, x > 3 ∧ x^3 - 27 ≤ 0

-- The Lean statement that proves the problem
theorem negation_of_p_is_neg_p : ¬ p ↔ neg_p := by
  sorry

end negation_of_p_is_neg_p_l141_141169


namespace min_omega_is_three_l141_141634

open Real

noncomputable def min_omega {ω : ℝ} {ϕ : ℝ} (f : ℝ → ℝ) : ℝ :=
  if f = (λ x, cos (ω * x + ϕ)) then
    if 0 < ω ∧ 0 < ϕ ∧ ϕ < π ∧
       (∃ (T : ℝ), T = 2 * π / ω ∧ f T = sqrt 3 / 2) ∧
       f (π / 9) = 0 then
      ω
    else 0
  else 0

theorem min_omega_is_three :
  let f : ℝ → ℝ := λ x, cos (3 * x + π / 6) in
  min_omega f = 3 := by
  sorry

end min_omega_is_three_l141_141634


namespace probability_two_primes_1_to_30_l141_141345

def is_prime (n : ℕ) : Prop := 
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13 ∨ n = 17 ∨ n = 19 ∨ n = 23 ∨ n = 29

def count_combinations (n k : ℕ) : ℕ := nat.choose n k

theorem probability_two_primes_1_to_30 :
  let total_combinations := count_combinations 30 2 in
  let prime_combinations := count_combinations 10 2 in
  total_combinations = 435 ∧ 
  prime_combinations = 45 ∧ 
  (prime_combinations : ℚ) / total_combinations = 15 / 145 :=
by { sorry }

end probability_two_primes_1_to_30_l141_141345


namespace perimeter_rectangle_l141_141395

theorem perimeter_rectangle (a b : ℕ) 
  (h₁ : a * b = 2018^2 - 2000^2) 
  (h₂ : |a - b| < 40) : 
  2 * (a + b) = 1076 := 
sorry

end perimeter_rectangle_l141_141395


namespace ratio_of_goals_l141_141467

-- The conditions
def first_period_goals_kickers : ℕ := 2
def second_period_goals_kickers := 4
def first_period_goals_spiders := first_period_goals_kickers / 2
def second_period_goals_spiders := 2 * second_period_goals_kickers
def total_goals := first_period_goals_kickers + second_period_goals_kickers + first_period_goals_spiders + second_period_goals_spiders

-- The ratio to prove
def ratio_goals : ℕ := second_period_goals_kickers / first_period_goals_kickers

theorem ratio_of_goals : total_goals = 15 → ratio_goals = 2 := by
  intro h
  sorry

end ratio_of_goals_l141_141467


namespace cos_plus_sin_eq_neg_sqrt_two_l141_141523

theorem cos_plus_sin_eq_neg_sqrt_two 
    (α k : ℝ) 
    (h_roots : ∀ x, x^2 - k * x + k^2 - 3 = 0 → (x = tan α ∨ x = 1 / tan α)) 
    (h_alpha : 3 * π < α ∧ α < 7 / 2 * π) : 
    cos α + sin α = -√2 := 
sorry

end cos_plus_sin_eq_neg_sqrt_two_l141_141523


namespace probability_both_numbers_prime_l141_141356

open Finset

def primes_between_1_and_30 : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_both_numbers_prime :
  (∃ (a b : ℕ), a ≠ b ∧ a ∈ primes_between_1_and_30 ∧ b ∈ primes_between_1_and_30 ∧
  (45 / 435) = (1 / 9)) :=
by
  have h_prime_count : ∃ (s : Finset ℕ), s.card = 10 ∧ s = primes_between_1_and_30 := sorry,
  have h_total_pairs : ∃ (n : ℕ), n = 435 := sorry,
  have h_prime_pairs : ∃ (m : ℕ), m = 45 := sorry,
  have h_fraction : (45 : ℚ) / 435 = (1 : ℚ) / 9 := sorry,
  exact ⟨45, 435, by simp [h_prime_pairs, h_total_pairs, h_fraction]⟩

end probability_both_numbers_prime_l141_141356


namespace tennis_players_cardinality_l141_141189

def sports_club : Type := sorry
def members_playing_something := 35 - 5
def badminton_players := {15}
def both_players := {3}
def tennis_players := badminton_players ∪ both_players -- Union of only badminton players and both players

theorem tennis_players_cardinality :
  tennis_players.card = 18 := 
by {
  let total_members := 35,
  let non_players := 5,
  let badminton := 15,
  let both := 3,
  let only_badminton := badminton - both,
  let playing_something := total_members - non_players,
  let only_tennis := playing_something - only_badminton - both,
  exact only_tennis + both
  sorry
}

end tennis_players_cardinality_l141_141189


namespace solve_inequality_find_minimum_l141_141388

def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

theorem solve_inequality (x : ℝ) : f(x) > 2 ↔ { Solution set from solution steps }
  sorry

theorem find_minimum : ∃ m : ℝ, ∀ x : ℝ, f(x) ≥ m ∧ (∀ y : ℝ, f(y) = m → y = {Some specific value of x(s)})
  sorry

end solve_inequality_find_minimum_l141_141388


namespace fixed_point_all_circles_circles_tangent_to_given_circle_l141_141927

noncomputable def circle_eq (a x y : ℝ) : Prop :=
x^2 + y^2 - 4*a*x + 2*a*y + 20*a - 20 = 0

noncomputable def fixed_point (x y : ℝ) : Prop :=
x = 4 ∧ y = -2

noncomputable def circle_4 : Prop :=
∀ (a : ℝ), circle_eq a 4 (-2)

noncomputable def tangent_circle (a : ℝ) : Prop :=
∀ x y : ℝ, (circle_eq a x y ↔ (x^2 + y^2 = 4))

theorem fixed_point_all_circles :
  circle_4 := 
by
  sorry

theorem circles_tangent_to_given_circle :
  ∃ a : ℝ, (a = 1 + (sqrt 5 / 5) ∨ a = 1 - (sqrt 5 / 5)) ∧ tangent_circle a :=
by
  sorry

end fixed_point_all_circles_circles_tangent_to_given_circle_l141_141927


namespace minimum_omega_value_l141_141642

noncomputable def min_omega (ω T x φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π) (hT : T = 2 * π / ω) 
  (hfT : Real.cos (ω * T + φ) = √3 / 2) (hx : Real.cos (ω * x + φ) = 0) : ℝ :=
  3

theorem minimum_omega_value (ω T x φ : ℝ)
  (hω : ω > 0)
  (hφ : 0 < φ ∧ φ < π)
  (hT : T = 2 * π / ω)
  (hfT : Real.cos (ω * T + φ) = √3 / 2)
  (hx : Real.cos (ω * x + φ) = 0)
  (hx_val : x = π / 9)
  (T_val : T = 2 * π / ω) :
  ω = min_omega ω T (π / 9) (π / 6) :=
begin
  sorry -- Proof steps are skipped as instructed.
end

end minimum_omega_value_l141_141642


namespace statistical_hypothesis_independence_l141_141587

def independence_test_statistical_hypothesis (A B: Prop) (independence_test: Prop) : Prop :=
  (independence_test ∧ A ∧ B) → (A = B)

theorem statistical_hypothesis_independence (A B: Prop) (independence_test: Prop) :
  (independence_test ∧ A ∧ B) → (A = B) :=
by
  sorry

end statistical_hypothesis_independence_l141_141587


namespace minimum_ab_l141_141903

theorem minimum_ab {n : ℤ} (h_n : n ≥ 3) 
(a b : ℝ) 
(h_a : a > 0) (h_b : b > 0)
(h_rect : ∃ (rect : set (ℤ × ℤ)), 
  (∀ (x y : ℤ), (x, y) ∈ rect → 
    (∀ (i j : ℤ), i ∈ {x - 1, x, x + 1} ∧ j ∈ {y - 1, y, y + 1} → 
      (((i, j) = (x, y) ∧ rect i j = 0 ∨ rect i j = 1) ∧ 
       ((i, j) ≠ (x, y) → ¬rect i j = rect x y))) ∧
  (card {z | isolated rect z} ≥ n^2 - n)) : 
a + b ≥ 4 * n :=
begin
  sorry
end

end minimum_ab_l141_141903


namespace min_value_fraction_l141_141123

theorem min_value_fraction {a b c : ℝ} (h1 : c > 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : 4 * a^2 - 2 * a * b + 4 * b^2 - c = 0) :
  ∃ (a b : ℝ), (|2 * a + b| = max |2 * a + b|) ∧ 4 * a^2 - 2 * a * b + 4 * b^2 - c = 0 ∧ min (3 / a - 4 / b + 5 / c) = -2 :=
sorry

end min_value_fraction_l141_141123


namespace range_of_a_l141_141925

-- Definitions and conditions based on problem statement
def point (x y : ℝ) := (x, y)

def circle_eq (x y : ℝ) := x^2 + y^2 + 2*x - 4*y + 3 = 0

def point_A (a : ℝ) := point 0 a

def condition_1 (x y : ℝ) : Prop := circle_eq x y

def condition_2 (a : ℝ) : Prop := a > 0

def condition_3 (x y a : ℝ) : Prop := ∃ (x y : ℝ), circle_eq x y ∧ ((x - 0)^2 + (y - a)^2 = 2 * (x^2 + y^2))

-- The proof problem statement
theorem range_of_a (a : ℝ) : 
  (∃ (x y : ℝ), condition_1 x y ∧ (condition_3 x y a)) ∧ condition_2 a → 
  sqrt 3 ≤ a ∧ a ≤ 4 + sqrt 19 :=
sorry

end range_of_a_l141_141925


namespace total_lives_correct_l141_141002

namespace VideoGame

def num_friends : ℕ := 8
def lives_each : ℕ := 8

def total_lives (n : ℕ) (l : ℕ) : ℕ := n * l 

theorem total_lives_correct : total_lives num_friends lives_each = 64 := by
  sorry

end total_lives_correct_l141_141002


namespace product_of_roots_l141_141551

theorem product_of_roots : (∃ x : ℝ, ((x + 3) * (x - 5) = 24)) → ∀ a b : ℝ, polynomial.roots (polynomial.X ^ 2 - 2 * polynomial.X - 39) = [a, b] → a * b = -39 := 
sorry

end product_of_roots_l141_141551


namespace square_position_after_transformations_l141_141016

-- Definitions for transformations
def square_pos (n : ℕ) : string :=
match n % 8 with
| 0 => "ABCD"
| 1 => "DABC"
| 2 => "BDAC"
| 3 => "ACBD"
| 4 => "CABD"
| 5 => "DCBA"
| 6 => "CDAB"
| 7 => "BADC"
| _ => "DBCA"
end

-- Main theorem statement
theorem square_position_after_transformations (n : ℕ) (h : n = 2010) :
  square_pos n = "BDAC" :=
by
  simp [square_pos, h]
  sorry

end square_position_after_transformations_l141_141016


namespace minimum_omega_value_l141_141648

noncomputable def min_omega (ω T x φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π) (hT : T = 2 * π / ω) 
  (hfT : Real.cos (ω * T + φ) = √3 / 2) (hx : Real.cos (ω * x + φ) = 0) : ℝ :=
  3

theorem minimum_omega_value (ω T x φ : ℝ)
  (hω : ω > 0)
  (hφ : 0 < φ ∧ φ < π)
  (hT : T = 2 * π / ω)
  (hfT : Real.cos (ω * T + φ) = √3 / 2)
  (hx : Real.cos (ω * x + φ) = 0)
  (hx_val : x = π / 9)
  (T_val : T = 2 * π / ω) :
  ω = min_omega ω T (π / 9) (π / 6) :=
begin
  sorry -- Proof steps are skipped as instructed.
end

end minimum_omega_value_l141_141648


namespace red_marbles_more_than_yellow_l141_141741

-- Define the conditions
def total_marbles : ℕ := 19
def yellow_marbles : ℕ := 5
def ratio_parts_blue : ℕ := 3
def ratio_parts_red : ℕ := 4

-- Prove the number of more red marbles than yellow marbles is equal to 3
theorem red_marbles_more_than_yellow :
  let remainder_marbles := total_marbles - yellow_marbles
  let total_parts := ratio_parts_blue + ratio_parts_red
  let marbles_per_part := remainder_marbles / total_parts
  let blue_marbles := ratio_parts_blue * marbles_per_part
  let red_marbles := ratio_parts_red * marbles_per_part
  red_marbles - yellow_marbles = 3 := by
  sorry

end red_marbles_more_than_yellow_l141_141741


namespace proof_problem_l141_141592

noncomputable def problem_statement (a b : ℝ) (A : ℝ) (β γ : ℝ) (c : ℝ) : Prop :=
  (vector_parallel (a, sqrt 3 * b) (cos A, sin β)) →
  (a = sqrt 7) →
  (b = 2) →
  (vector_parallel (a, sqrt 3 * b) (cos A, sin β)) →
  A = π / 3 ∧ (1 / 2 * b * c * sin A = 3 * sqrt 3 / 2)

theorem proof_problem :
  ∃ (a b : ℝ) (A : ℝ) (β γ : ℝ) (c : ℝ), a = sqrt 7 ∧ b = 2 ∧
  problem_statement a b A β γ c :=
sorry

end proof_problem_l141_141592


namespace percent_of_value_and_divide_l141_141762

theorem percent_of_value_and_divide (x : ℝ) (y : ℝ) (z : ℝ) (h : x = 1/300 * 180) (h1 : y = x / 6) : 
  y = 0.1 := 
by
  sorry

end percent_of_value_and_divide_l141_141762


namespace smallest_period_of_f_max_min_values_of_f_in_interval_l141_141953

open Real

def vec_a (x : ℝ) : ℝ × ℝ := (sin x, -2 * cos x)
def vec_b (x : ℝ) : ℝ × ℝ := (sin x + sqrt 3 * cos x, -cos x)

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

def f (x : ℝ) : ℝ := dot_product (vec_a x) (vec_b x)

theorem smallest_period_of_f : (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∀ T' > 0, T' < T → ∃ x, f (x + T') ≠ f x) := sorry

theorem max_min_values_of_f_in_interval : 
  ∀ x ∈ Icc 0 (π / 2), f x ≥ 1 ∧ f x ≤ 5 / 2 := sorry

end smallest_period_of_f_max_min_values_of_f_in_interval_l141_141953


namespace line_intersection_l141_141407

noncomputable def line1 (t : ℚ) : ℚ × ℚ := (1 - 2 * t, 4 + 3 * t)
noncomputable def line2 (u : ℚ) : ℚ × ℚ := (5 + u, 2 + 6 * u)

theorem line_intersection :
  ∃ t u : ℚ, line1 t = (21 / 5, -4 / 5) ∧ line2 u = (21 / 5, -4 / 5) :=
sorry

end line_intersection_l141_141407


namespace trig_expression_evaluation_l141_141451

theorem trig_expression_evaluation :
  let cos_45 := (Real.sqrt 2) / 2
  let sin_60 := (Real.sqrt 3) / 2
  let tan_45 := 1
  (Real.sqrt 2) * cos_45 - (sin_60 ^ 2) + tan_45 = 5 / 4 :=
by
  let cos_45 := (Real.sqrt 2) / 2
  let sin_60 := (Real.sqrt 3) / 2
  let tan_45 := 1
  have h1 : (Real.sqrt 2) * cos_45 = 1 := by
    calc (Real.sqrt 2) * cos_45
       = (Real.sqrt 2) * ((Real.sqrt 2) / 2) : by rw cos_45
    ... = (Real.sqrt 2 * Real.sqrt 2) / 2 : by ring
    ... = 2 / 2 : by rw Real.sqrt_mul_self
    ... = 1 : by ring
  have h2 : sin_60 ^ 2 = 3 / 4 := by
    calc sin_60 ^ 2
       = ((Real.sqrt 3) / 2) ^ 2 : by rw sin_60
    ... = (Real.sqrt 3) ^ 2 / 4 : by ring
    ... = 3 / 4 : by rw Real.sqrt_mul_self
  have h3 : tan_45 = 1 := by rw tan_45
  show (Real.sqrt 2) * cos_45 - sin_60 ^ 2 + tan_45 = 5 / 4 from
    calc (Real.sqrt 2) * cos_45 - sin_60 ^ 2 + tan_45
       = 1 - 3 / 4 + 1 : by rw [h1, h2, h3]
    ... = 5 / 4 : by ring

end trig_expression_evaluation_l141_141451


namespace percent_of_g_is_a_l141_141706

-- Definitions of the seven consecutive numbers
def consecutive_7_avg_9 (a b c d e f g : ℝ) : Prop :=
  a + b + c + d + e + f + g = 63

def is_median (d : ℝ) : Prop :=
  d = 9

def express_numbers (a b c d e f g : ℝ) : Prop :=
  a = d - 3 ∧ b = d - 2 ∧ c = d - 1 ∧ d = d ∧ e = d + 1 ∧ f = d + 2 ∧ g = d + 3

-- Main statement asserting the percentage relationship
theorem percent_of_g_is_a (a b c d e f g : ℝ) (h_avg : consecutive_7_avg_9 a b c d e f g)
  (h_median : is_median d) (h_express : express_numbers a b c d e f g) :
  (a / g) * 100 = 50 := by
  sorry

end percent_of_g_is_a_l141_141706


namespace geometric_mean_4_16_l141_141529

theorem geometric_mean_4_16 (x : ℝ) (h : x^2 = 4 * 16) : x = 8 ∨ x = -8 :=
sorry

end geometric_mean_4_16_l141_141529


namespace incorrect_statement_in_bio_experiment_l141_141373

def biological_experiment_statements (A_correct B_correct C_correct D_incorrect : Prop) : Prop :=
  A_correct ∧ B_correct ∧ C_correct ∧ ¬D_incorrect

theorem incorrect_statement_in_bio_experiment
    (A_correct : The container for making fruit wine needs to be vented in a timely manner)
    (B_correct : When selecting specific microorganisms, it is necessary to prepare selective culture media)
    (C_correct : The MS medium needs to consider the amount of plant hormones)
    (D_incorrect : ¬ The streak plate method can be used for the separation and counting of microorganisms) :
    biological_experiment_statements A_correct B_correct C_correct D_incorrect :=
by
  sorry

end incorrect_statement_in_bio_experiment_l141_141373


namespace problem_statement_l141_141543

variable {a : ℝ}

def P := ∀ x : ℝ, log 2 (x^2 + x + a) > 0

def Q := ∃ x_0 ∈ Icc (-2 : ℝ) 2, 2^a ≤ 2^x_0

theorem problem_statement (hP : P) (hQ : Q) : (5 / 4 : ℝ) < a ∧ a ≤ 2 := 
sorry

end problem_statement_l141_141543


namespace find_x_squared_plus_y_squared_l141_141552

theorem find_x_squared_plus_y_squared
  (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + x + y = 110) : 
  x^2 + y^2 = 80 :=
begin
  sorry
end

end find_x_squared_plus_y_squared_l141_141552


namespace lcm_48_75_l141_141874

theorem lcm_48_75 : Nat.lcm 48 75 = 1200 := by
  sorry

end lcm_48_75_l141_141874


namespace intersection_of_cube_with_plane_l141_141761

theorem intersection_of_cube_with_plane (P : set (set ℝ)) :
  (∀ polygon ∈ P, polygon ≠ ∅ ∧ 
    (∀ edge ∈ polygon, edge ⊆ cube ∧ is_face_cube edge)) ∧
  (∀ polygon ∈ P, the maximum_sides polygon <= 6) ∧
  (∀ polygon ∈ P, edges_on_opposite_faces polygon are_parallel) →
  (P = {equilateral_triangle, square, regular_pentagon}) :=
by
  sorry

end intersection_of_cube_with_plane_l141_141761


namespace domain_of_ln_function_l141_141080

theorem domain_of_ln_function (x : ℝ) : 3 - 4 * x > 0 ↔ x < 3 / 4 := 
by
  sorry

end domain_of_ln_function_l141_141080


namespace remaining_eggs_l141_141032

theorem remaining_eggs (dozens : ℕ) (eggs_per_dozen : ℕ) (used_ratio : ℚ) (broken_eggs : ℕ) :
  dozens = 6 →
  eggs_per_dozen = 12 →
  used_ratio = 1 / 2 →
  broken_eggs = 15 →
  let total_eggs := dozens * eggs_per_dozen in
  let used_eggs := total_eggs * used_ratio in
  let remaining_eggs := total_eggs - used_eggs in
  let final_eggs := remaining_eggs - broken_eggs in
  final_eggs = 21 :=
by
  intros hd hz hr hb
  rw [hd, hz, hr, hb]
  let total_eggs := 6 * 12
  let used_eggs := total_eggs * (1 / 2 : ℚ)
  let remaining_eggs := total_eggs - used_eggs
  have h1 : remaining_eggs = 36 := by norm_num
  let final_eggs := remaining_eggs - 15
  have h2 : final_eggs = 21 := by norm_num
  exact h2

end remaining_eggs_l141_141032


namespace find_number_of_women_l141_141393

-- Define the work rate variables and the equations from conditions
variables (m w : ℝ) (x : ℝ)

-- Define the first condition
def condition1 : Prop := 3 * m + x * w = 6 * m + 2 * w

-- Define the second condition
def condition2 : Prop := 4 * m + 2 * w = (5 / 7) * (3 * m + x * w)

-- The theorem stating that, given the above conditions, x must be 23
theorem find_number_of_women (hmw : m = 7 * w) (h1 : condition1 m w x) (h2 : condition2 m w x) : x = 23 :=
sorry

end find_number_of_women_l141_141393


namespace stopping_distance_l141_141006

-- Define the initial conditions and constants
def initial_speed : ℝ := 20
def braking_factor : ℝ := 0.2
def gravity : ℝ := 9.8

-- Define the acceleration due to braking
def acceleration : ℝ := -braking_factor * gravity

-- Define the equation for velocity as a function of time
def velocity (t : ℝ) : ℝ := initial_speed + acceleration * t

-- Define the equation for distance as a function of time
def distance (t : ℝ) : ℝ := initial_speed * t + 0.5 * acceleration * t^2

-- The problem statement to prove: the stopping distance
theorem stopping_distance : ∃ t s, (velocity t = 0) ∧ (distance t = s) ∧ (s ≈ 102) :=
by
  let t := initial_speed / -acceleration
  let s := distance t
  use [t, s]
  split
  · simp [velocity, t]
    sorry
  · split
    · simp [distance, t, s]
      sorry
    · simp [s]
      sorry

end stopping_distance_l141_141006


namespace max_y_intercept_l141_141541

def hyperbola_equation : String := "x^2 / 5 - y^2 = 1"

structure Ellipse (a b : ℝ) :=
(eq : String := "x^2 / a^2 + y^2 / b^2 = 1")

noncomputable def ellipse_focus_condition (a : ℝ) (b : ℝ) (c : ℝ) : Prop :=
  a = Real.sqrt 6 ∧ c = Real.sqrt 5 ∧ b = Real.sqrt (6 - 5)

noncomputable def reciprocal_eccentricities (e_ellipse e_hyperbola : ℝ) : Prop :=
  e_ellipse = 1 / e_hyperbola

def MN_condition (MN : ℝ) : Prop :=
  MN = 4 * Real.sqrt 3 / 3

theorem max_y_intercept (m : ℝ) : 
  ∃ a b c e_ellipse e_hyperbola (M N : ℝ × ℝ),
    ellipse_focus_condition a b c ∧
    reciprocal_eccentricities e_ellipse e_hyperbola ∧
    MN_condition (real.dist M N) →
    m = 5 / 3 :=
sorry

end max_y_intercept_l141_141541


namespace rotated_angle_remains_60_degrees_l141_141296

theorem rotated_angle_remains_60_degrees (A B C : Type) [metric_space A] 
  (angle_ACB original rotated : ℝ)
  (h0 : angle_ACB = 60)
  (h1 : rotated = 600):
  original = 60 → (original - 2 * 360) % 360 = original :=
by
  sorry

end rotated_angle_remains_60_degrees_l141_141296


namespace girls_maple_grove_correct_l141_141445

variables (total_students boys girls pine_ridge_students maple_grove_students boys_pine_ridge : ℕ)
variables (girls_maple_grove : ℕ)

-- Conditions
def conditions : Prop :=
  total_students = 150 ∧ 
  boys = 82 ∧ 
  girls = 68 ∧ 
  pine_ridge_students = 70 ∧ 
  maple_grove_students = 80 ∧ 
  boys_pine_ridge = 36 ∧ 
  girls_maple_grove = girls - (pine_ridge_students - boys_pine_ridge)

-- Question and Answer translated to a proposition
def proof_problem : Prop :=
  conditions total_students boys girls pine_ridge_students maple_grove_students boys_pine_ridge girls_maple_grove → 
  girls_maple_grove = 34

-- Statement
theorem girls_maple_grove_correct : proof_problem total_students boys girls pine_ridge_students maple_grove_students boys_pine_ridge girls_maple_grove :=
by {
  sorry -- Proof omitted
}

end girls_maple_grove_correct_l141_141445


namespace minimum_value_of_f_l141_141878

def operation (a b : ℝ) : ℝ :=
  if a * b ≤ 0 then a * b else -a / b

def f (x : ℝ) : ℝ :=
  operation x (real.exp x)

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ -1 / real.exp 1) ∧ f x = -1 / real.exp 1 :=
begin
  sorry
end

end minimum_value_of_f_l141_141878


namespace tangent_line_through_point_and_circle_l141_141483

noncomputable def tangent_line_equation : String :=
  "y - 1 = 0"

theorem tangent_line_through_point_and_circle :
  ∀ (line_eq: String), 
  (∀ (x y: ℝ), (x - 1) ^ 2 + y ^ 2 = 1 ∧ (x, y) = (1, 1) → y - 1 = 0) →
  line_eq = tangent_line_equation :=
by
  intro line_eq h
  sorry

end tangent_line_through_point_and_circle_l141_141483


namespace guess_six_digit_number_l141_141964

noncomputable def six_digit_number : Type := {n : ℕ // 100000 ≤ n ∧ n < 1000000}

theorem guess_six_digit_number (n : six_digit_number) : ∃ m : ℕ, (m < 1000000 ∧ 100000 ≤ m) ∧ by sorry :=
begin
  -- The sequence of yes-or-no questions verifies each bit of the six-digit number's binary representation.
  sorry
end

end guess_six_digit_number_l141_141964


namespace find_length_BC_l141_141979

variable {V : Type _} [InnerProductSpace ℝ V]

theorem find_length_BC 
  (A B C : V) 
  (h₁ : ∥B - A∥ = 2) 
  (h₂ : ∥C - A∥ = 2) 
  (h₃ : ⟪B - A, C - A⟫ = 1) : 
  ∥C - B∥ = Real.sqrt 6 := 
sorry

end find_length_BC_l141_141979


namespace extended_pattern_ratio_l141_141864

noncomputable def original_black_tiles : ℕ := 12
noncomputable def original_white_tiles : ℕ := 24
noncomputable def original_total_tiles : ℕ := 36
noncomputable def extended_total_tiles : ℕ := 64
noncomputable def border_black_tiles : ℕ := 24 /- The new border adds 24 black tiles -/
noncomputable def extended_black_tiles : ℕ := 36
noncomputable def extended_white_tiles := original_white_tiles

theorem extended_pattern_ratio :
  (extended_black_tiles : ℚ) / extended_white_tiles = 3 / 2 :=
by
  sorry

end extended_pattern_ratio_l141_141864


namespace treasure_ark_code_correct_l141_141267

def treasureArkTable : List (List ℕ) :=
  [ [5, 9, 4, 9, 4, 1],
    [6, 3, 7, 3, 4, 8],
    [8, 2, 4, 2, 5, 5],
    [7, 4, 5, 7, 5, 2],
    [2, 7, 6, 1, 2, 8],
    [5, 2, 3, 6, 7, 1] ]

def groupsOfThreeSumFourteen (table : List (List ℕ)) :=
  -- This function will define the positions of all groups of three that sum to 14
  [ ([1, 2], [1, 2], [1, 2]), -- different set of positions corresponding to groups cols
    ([2, 3], [2, 4], [2, 5]), -- 7 + 3 + 4, 8 + 2 + 4, 7 + 5 + 2, 7 + 4 + 5, 7 + 6 + 1, etc.
    -- similarly for all other valid groups
  ]

def sumOfRemainingNumbers : List (List ℕ) := do
  let markedIndices := groupsOfThreeSumFourteen treasureArkTable
  -- Assuming table and markedIndices is properly evaluated to remove marked elements

theorem treasure_ark_code_correct :
  let remainingNumbers := sumOfRemainingNumbers
  List.sum (List.join remainingNumbers) = 29 :=
by
  sorry

end treasure_ark_code_correct_l141_141267


namespace instantaneous_velocity_at_t4_l141_141396

open Real

noncomputable def position (t : ℝ) : ℝ := 3 * t^2 + t + 4

def velocity (t : ℝ) : ℝ := deriv position t

theorem instantaneous_velocity_at_t4 : velocity 4 = 25 :=
by
  sorry

end instantaneous_velocity_at_t4_l141_141396


namespace min_omega_is_three_l141_141633

open Real

noncomputable def min_omega {ω : ℝ} {ϕ : ℝ} (f : ℝ → ℝ) : ℝ :=
  if f = (λ x, cos (ω * x + ϕ)) then
    if 0 < ω ∧ 0 < ϕ ∧ ϕ < π ∧
       (∃ (T : ℝ), T = 2 * π / ω ∧ f T = sqrt 3 / 2) ∧
       f (π / 9) = 0 then
      ω
    else 0
  else 0

theorem min_omega_is_three :
  let f : ℝ → ℝ := λ x, cos (3 * x + π / 6) in
  min_omega f = 3 := by
  sorry

end min_omega_is_three_l141_141633


namespace ratio_of_conjugate_is_imaginary_unit_l141_141522

theorem ratio_of_conjugate_is_imaginary_unit
  (z : ℂ)
  (conj_z : ℂ)
  (h₁ : conj_z = conj z)
  (h₂ : z + conj_z = 4)
  (h₃ : z * conj_z = 8) :
  conj_z / z = Complex.I ∨ conj_z / z = -Complex.I :=
sorry

end ratio_of_conjugate_is_imaginary_unit_l141_141522


namespace rook_paths_l141_141062

theorem rook_paths (n : ℕ) : 
  ∃ (C : ℕ), C = (1 / n : ℚ) * (Nat.binomial (2 * n - 2) (n - 1)) := 
by 
  use (Nat.fact (2 * n - 2) / (Nat.fact (n - 1) * Nat.fact (n - 1))) / n
  sorry

end rook_paths_l141_141062


namespace students_play_alto_saxophone_l141_141050

def roosevelt_high_school :=
  let total_students := 600
  let marching_band_students := total_students / 5
  let brass_instrument_students := marching_band_students / 2
  let saxophone_students := brass_instrument_students / 5
  let alto_saxophone_students := saxophone_students / 3
  alto_saxophone_students

theorem students_play_alto_saxophone :
  roosevelt_high_school = 4 :=
  by
    sorry

end students_play_alto_saxophone_l141_141050


namespace square_perimeter_l141_141017

theorem square_perimeter (s : ℝ) (h1 : ∀ r : ℝ, r = 2 * (s + s / 4)) (r_perimeter_eq_40 : r = 40) :
  4 * s = 64 := by
  sorry

end square_perimeter_l141_141017


namespace max_value_a7_b7_c7_d7_l141_141690

-- Assume a, b, c, d are real numbers such that a^6 + b^6 + c^6 + d^6 = 64
-- Prove that the maximum value of a^7 + b^7 + c^7 + d^7 is 128
theorem max_value_a7_b7_c7_d7 (a b c d : ℝ) (h : a^6 + b^6 + c^6 + d^6 = 64) : 
  ∃ a b c d, a^6 + b^6 + c^6 + d^6 = 64 ∧ a^7 + b^7 + c^7 + d^7 = 128 :=
by sorry

end max_value_a7_b7_c7_d7_l141_141690


namespace sum_dihedral_angles_eq_two_pi_l141_141604

-- Definitions and assumptions based on the problem statement:
structure CylinderPlaneGeometry (O A B C : Type) [point: O] [point: A] [point: B] [point: C] :=
  (is_midpoint : ∃ (axis : Type) [line : axis], O = midpoint axis)
  (is_diametrically_opposite : ∃ (base : Type) [circle : base], A ∈ base ∧ B ∈ base ∧ diametrically_opposite A B)
  (C_not_in_plane_OAB : C ∉ plane_of O A B)

-- Problem statement:
theorem sum_dihedral_angles_eq_two_pi (O A B C : Type) [point: O] [point: A] [point: B] [point: C]
  [geom : CylinderPlaneGeometry O A B C] : 
  dihedral_angle_sum O A B C = 2 * π :=
sorry

end sum_dihedral_angles_eq_two_pi_l141_141604


namespace max_value_a7_b7_c7_d7_l141_141691

-- Assume a, b, c, d are real numbers such that a^6 + b^6 + c^6 + d^6 = 64
-- Prove that the maximum value of a^7 + b^7 + c^7 + d^7 is 128
theorem max_value_a7_b7_c7_d7 (a b c d : ℝ) (h : a^6 + b^6 + c^6 + d^6 = 64) : 
  ∃ a b c d, a^6 + b^6 + c^6 + d^6 = 64 ∧ a^7 + b^7 + c^7 + d^7 = 128 :=
by sorry

end max_value_a7_b7_c7_d7_l141_141691


namespace polynomial_characterization_l141_141480

noncomputable def polynomial_form {P : Polynomial ℤ} (n : ℕ) : Prop :=
  ∃ {d : ℕ} (hd : odd d) (a r s : ℤ) (hcoeff : P.coeff = a * r^d),
    ∀ (m : ℕ) (x : Fin m → ℕ), (∀ i j : Fin m, (1 / 2 : ℝ) < (P.eval (x i).toRat / P.eval (x j).toRat) ∧ (P.eval (x i) / P.eval x.toRat) ^ d ∈ ℚ)

theorem polynomial_characterization (P : Polynomial ℤ) :
  (∀ (m : ℕ), ∃ (x : Fin m → ℕ), 
    (∀ i j : Fin m, (1 / 2 : ℝ) < (P.eval (x i).toRat / P.eval (x j).toRat) ∧ (P.eval (x i) / P.eval (x j)) ^ d ∈ ℚ)) 
  → ∃ (a r s : ℤ) (d : ℕ) (hd : odd d), 
      ∀ x, P.eval x = a * ((r * x) + s) ^ d := 
sorry

end polynomial_characterization_l141_141480


namespace ninth_term_is_55_l141_141831

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- Though the formal sequence starts with n=1, we define sequence(0) for completion.
  | 1 => 3
  | (n + 1) => sequence n + (2 + n)

theorem ninth_term_is_55 : sequence 9 = 55 :=
  sorry

end ninth_term_is_55_l141_141831


namespace no_equal_area_point_l141_141187

theorem no_equal_area_point {A B C M P Q : ℝ → Prop} :
    (∃ x y z w : ℝ, 
        0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w 
        ∧ is_right_triangle A B C
        ∧ on_hypotenuse M B C
        ∧ projections P Q M A B C
        ∧ (1 / 2) * B * P * P * M = (1 / 2) * M * Q * C * Q
        ∧ (1 / 2) * B * P * P * M = A * Q * P * M)
    → False := sorry

end no_equal_area_point_l141_141187


namespace xiao_li_max_prob_interview_xiao_li_xiao_wang_prob_and_expected_value_l141_141794

noncomputable def xiao_li_passing_prob_B : ℝ := 2 / 3
noncomputable def xiao_li_passing_prob_C : ℝ := 1 / 3
noncomputable def xiao_li_passing_prob_D : ℝ := 1 / 2
noncomputable def xiao_wang_passing_prob : ℝ := 2 / 3

/-- Xiao Li should choose test locations B and D to maximize the probability of being eligible for an interview. -/
theorem xiao_li_max_prob_interview :
  max (xiao_li_passing_prob_B * xiao_li_passing_prob_C)
      (max (xiao_li_passing_prob_B * xiao_li_passing_prob_D)
           (xiao_li_passing_prob_C * xiao_li_passing_prob_D)) 
  = xiao_li_passing_prob_B * xiao_li_passing_prob_D := 
sorry

/-- Determine the probability distribution for the random variable ξ and its expected value. -/
theorem xiao_li_xiao_wang_prob_and_expected_value (B C D : Prop) [Indep of B C D] :
  let ξ := B.CD.in_front_of_xiao_wang_pass_events (B_intersects_xiao_li_C : bool) in
    ξ = 0 → P (ξ = 0) = 2 / 81
    ∧ ξ = 1 → P (ξ = 1) = 13 / 81
    ∧ ξ = 2 → P (ξ = 2) = 10 / 27
    ∧ ξ = 3 → P (ξ = 3) = 28 / 81
    ∧ ξ = 4 → P (ξ = 4) = 8 / 81
    ∧ E ξ = 7 / 3 := 
sorry

end xiao_li_max_prob_interview_xiao_li_xiao_wang_prob_and_expected_value_l141_141794


namespace answer_neither_question_l141_141165

-- Definitions for the given probabilities
variables (P_A P_B P_A_and_B : ℝ)

-- Conditions from the problem
def cond1 := P_A = 0.75
def cond2 := P_B = 0.35
def cond3 := P_A_and_B = 0.30

-- Calculate P(A ∪ B) using the principle of inclusion-exclusion
def P_A_or_B : ℝ := P_A + P_B - P_A_and_B

-- Define the complement probability P(N)
def P_N : ℝ := 1 - P_A_or_B

-- The theorem to be proven
theorem answer_neither_question (h1 : cond1) (h2 : cond2) (h3 : cond3) : P_N = 0.20 :=
by sorry

end answer_neither_question_l141_141165


namespace minimum_omega_is_3_l141_141625

variable (f : ℝ → ℝ)
variable (ω φ T : ℝ)
variable (x : ℝ)

noncomputable def minimum_omega (ω : ℝ) : ℝ := 3 -- Temporary noncomputable definition to hold the value for min omega

theorem minimum_omega_is_3 :
  (∀ x, f x = Real.cos (ω * x + φ)) →
  ω > 0 →
  0 < φ ∧ φ < Real.pi →
  T = (2 * Real.pi) / ω →
  f T = Real.sqrt 3 / 2 →
  f (Real.pi / 9) = 0 →
  ∃(ω_min : ℝ), ω_min = 3 :=
by
  intros h0 h1 h2 h3 h4 h5
  use 3
  sorry -- Proof of the exact answer

end minimum_omega_is_3_l141_141625


namespace rotated_angle_remains_60_degrees_l141_141295

theorem rotated_angle_remains_60_degrees (A B C : Type) [metric_space A] 
  (angle_ACB original rotated : ℝ)
  (h0 : angle_ACB = 60)
  (h1 : rotated = 600):
  original = 60 → (original - 2 * 360) % 360 = original :=
by
  sorry

end rotated_angle_remains_60_degrees_l141_141295


namespace hadassah_additional_paintings_l141_141152

noncomputable def hadassah_initial_paintings : ℕ := 12
noncomputable def hadassah_initial_hours : ℕ := 6
noncomputable def hadassah_total_hours : ℕ := 16

theorem hadassah_additional_paintings 
  (initial_paintings : ℕ)
  (initial_hours : ℕ)
  (total_hours : ℕ) :
  initial_paintings = hadassah_initial_paintings →
  initial_hours = hadassah_initial_hours →
  total_hours = hadassah_total_hours →
  let additional_hours := total_hours - initial_hours
  let painting_rate := initial_paintings / initial_hours
  let additional_paintings := painting_rate * additional_hours
  additional_paintings = 20 :=
by
  sorry

end hadassah_additional_paintings_l141_141152


namespace sales_volume_at_240_selling_price_for_profit_9000_l141_141790

-- Definitions for conditions
def initialPrice : ℝ := 260
def initialVolume : ℝ := 45
def decreaseRate : ℝ := 7.5
def costPerTon : ℝ := 100
def totalCostPerTon : ℝ := costPerTon

-- Problem 1: Monthly sales volume when price is 240 yuan
theorem sales_volume_at_240 :
  ∀ (price : ℝ), price = 240 → 
  let decrease := initialPrice - price in
  let volumeIncrease := decreaseRate * (decrease / 10) in
  let finalVolume := initialVolume + volumeIncrease in
  finalVolume = 60 :=
by
  intros price h
  let decrease := initialPrice - price
  let volumeIncrease := decreaseRate * (decrease / 10)
  let finalVolume := initialVolume + volumeIncrease
  have := Eq.subst h.succ_inj'.symm rfl
  rw [has_eq_eq_eq.eq, this] at finalVolume
  exact sorry

-- Problem 2: Selling prices for monthly profit of 9000 yuan
theorem selling_price_for_profit_9000 :
  ∀ (x : ℝ), 
  ((x - totalCostPerTon) * (initialVolume + decreaseRate * ((initialPrice - x) / 10))) = 9000 → 
  (x = 200 ∨ x = 220) :=
by
  intros x h
  have : (x - totalCostPerTon) * (initialVolume + decreaseRate * ((initialPrice - x) / 10)) = 9000 :=
    by rw [←h]
  exact sorry

end sales_volume_at_240_selling_price_for_profit_9000_l141_141790


namespace sum_of_coordinates_l141_141922

theorem sum_of_coordinates (g : ℝ → ℝ) (h : g 7 = 10) :
  let x := 7 / 3
  let y := 17 / 3
  x + y = 8 :=
by
  let x := 7 / 3
  let y := 17 / 3
  have hg : 2 * g (3 * x) - 3 = 17 := by
    rw [← mul_div_cancel' 7 (ne_of_gt (by norm_num : 3 > 0)), mul_comm, mul_assoc, mul_inv_cancel, mul_comm, h]
    simp
  have hx : 3 * y = 17 := by
    rw [← hg]
    simp
  have hy : y = 17 / 3 := by
    rw [← div_eq_iff' (by norm_num : 3 ≠ 0)]
    exact hx
  exact calc
    x + y = 7 / 3 + 17 / 3 : by simp [x, y]
        ... = (7 + 17) / 3 : by rw [add_div]
        ... = 24 / 3 : by norm_num
        ... = 8 : by norm_num

end sum_of_coordinates_l141_141922


namespace minimum_omega_value_l141_141647

noncomputable def min_omega (ω T x φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π) (hT : T = 2 * π / ω) 
  (hfT : Real.cos (ω * T + φ) = √3 / 2) (hx : Real.cos (ω * x + φ) = 0) : ℝ :=
  3

theorem minimum_omega_value (ω T x φ : ℝ)
  (hω : ω > 0)
  (hφ : 0 < φ ∧ φ < π)
  (hT : T = 2 * π / ω)
  (hfT : Real.cos (ω * T + φ) = √3 / 2)
  (hx : Real.cos (ω * x + φ) = 0)
  (hx_val : x = π / 9)
  (T_val : T = 2 * π / ω) :
  ω = min_omega ω T (π / 9) (π / 6) :=
begin
  sorry -- Proof steps are skipped as instructed.
end

end minimum_omega_value_l141_141647


namespace num_zeros_f_l141_141852

noncomputable def f : ℝ → ℝ := λ x, (x - 3) * Real.exp x

theorem num_zeros_f :
  ∃! x ∈ Set.Ioi (0 : ℝ), f x = 0 :=
sorry

end num_zeros_f_l141_141852


namespace melanie_gave_8_dimes_l141_141675

theorem melanie_gave_8_dimes
  (initial_dimes : ℕ)
  (additional_dimes : ℕ)
  (current_dimes : ℕ)
  (given_away_dimes : ℕ) :
  initial_dimes = 7 →
  additional_dimes = 4 →
  current_dimes = 3 →
  given_away_dimes = (initial_dimes + additional_dimes - current_dimes) →
  given_away_dimes = 8 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end melanie_gave_8_dimes_l141_141675


namespace intersection_A_B_l141_141670

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def intersection (S T : Set ℝ) : Set ℝ := {x | x ∈ S ∧ x ∈ T}

theorem intersection_A_B :
  (A ∩ B) = {x | 0 < x ∧ x ≤ 2} := by
  sorry

end intersection_A_B_l141_141670


namespace most_accurate_value_l141_141578

noncomputable def K : ℝ := 5.12718
noncomputable def error : ℝ := 0.00457
noncomputable def K_upper : ℝ := K + error
noncomputable def K_lower : ℝ := K - error

theorem most_accurate_value :
  (K_upper ≥ 5.1 ∧ K_upper < 5.2) ∧ (K_lower ≥ 5.1 ∧ K_lower < 5.2) → 5.1 :=
by
  sorry

end most_accurate_value_l141_141578


namespace distance_between_lines_l141_141132

noncomputable def max_min_distance (a b c : ℝ) (h1 : a + b = -1) (h2 : a * b = c) (hc : 0 ≤ c ∧ c ≤ 1/8) : ℝ × ℝ :=
  let d_sq := (1 - 4 * c) / 2 in
  (real.sqrt (d_sq.max (1/4)), real.sqrt (d_sq.min (1/2)))

theorem distance_between_lines (a b c : ℝ) (h1 : a + b = -1) (h2 : a * b = c) (hc : 0 ≤ c ∧ c ≤ 1/8) :
  max_min_distance a b c h1 h2 hc = (real.sqrt (1/2), 1/2) := by
  sorry

end distance_between_lines_l141_141132


namespace no_solution_sqrt_x_plus_1_minus_sqrt_x_minus_1_no_solution_sqrt_x_minus_sqrt_x_minus_sqrt_1_minus_x_l141_141262

theorem no_solution_sqrt_x_plus_1_minus_sqrt_x_minus_1 :
  ∀ x : ℝ, sqrt (x + 1) - sqrt (x - 1) ≠ 0 :=
by
  sorry

theorem no_solution_sqrt_x_minus_sqrt_x_minus_sqrt_1_minus_x :
  ∀ x : ℝ, x ≤ 1 → sqrt x - sqrt (x - sqrt (1 - x)) ≠ 1 :=
by
  sorry

end no_solution_sqrt_x_plus_1_minus_sqrt_x_minus_1_no_solution_sqrt_x_minus_sqrt_x_minus_sqrt_1_minus_x_l141_141262


namespace cos_2alpha_plus_pi_over_6_l141_141526

theorem cos_2alpha_plus_pi_over_6 (α : ℝ) (h₁ : sin α = sqrt 10 / 10) (h₂ : 0 < α ∧ α < π / 2) :
  cos (2 * α + π / 6) = (4 * sqrt 3 - 3) / 10 :=
by
  sorry

end cos_2alpha_plus_pi_over_6_l141_141526


namespace tickets_left_unsold_l141_141094

theorem tickets_left_unsold : 
  let total_tickets := 30 * 100 in
  let fourth_graders := 0.30 * total_tickets in
  let remaining_after_fourth := total_tickets - fourth_graders in
  let fifth_graders := 0.50 * remaining_after_fourth in
  let remaining_after_fifth := remaining_after_fourth - fifth_graders in
  let sixth_graders := 100 in
  let tickets_left := remaining_after_fifth - sixth_graders in
  tickets_left = 950 :=
by
  sorry

end tickets_left_unsold_l141_141094


namespace min_S_of_sum_1995_l141_141510

theorem min_S_of_sum_1995 (a : Fin 10 → ℕ) (h_distinct : Function.Injective a) (h_sum : (Finset.univ.sum a) = 1995) :
  6044 ≤ Finset.univ.sum (λ i : Fin 10, a i * a ((i + 1) % 10)) :=
sorry

end min_S_of_sum_1995_l141_141510


namespace sqrt8_sub_sqrt2_sub_sqrt1div3_mul_sqrt6_sqrt15_div_sqrt3_add_sqrt5_sub1_sq_l141_141055

theorem sqrt8_sub_sqrt2_sub_sqrt1div3_mul_sqrt6 : 
  (Nat.sqrt 8 - Nat.sqrt 2 - (Nat.sqrt (1 / 3) * Nat.sqrt 6) = 0) :=
by
  sorry

theorem sqrt15_div_sqrt3_add_sqrt5_sub1_sq : 
  (Nat.sqrt 15 / Nat.sqrt 3 + (Nat.sqrt 5 - 1) ^ 2 = 6 - Nat.sqrt 5) :=
by
  sorry

end sqrt8_sub_sqrt2_sub_sqrt1div3_mul_sqrt6_sqrt15_div_sqrt3_add_sqrt5_sub1_sq_l141_141055


namespace max_stamps_with_100_dollars_l141_141571

/-- max_stamps function calculates the maximum number of stamps that can be purchased with a given budget, price per stamp, and conditional discount --/
def max_stamps (budget : ℝ) (price_per_stamp : ℝ) (discount_threshold : ℝ) (discount_rate : ℝ) : ℝ :=
  if budget <= discount_threshold * price_per_stamp then
    budget / price_per_stamp
  else
    ((discount_threshold * price_per_stamp) + ((budget - (discount_threshold * price_per_stamp)) / (price_per_stamp * (1 - discount_rate))))

theorem max_stamps_with_100_dollars : max_stamps 100 0.5 100 0.1 = 200 :=
sorry

end max_stamps_with_100_dollars_l141_141571


namespace hyperbola_eccentricity_l141_141126

theorem hyperbola_eccentricity {a b : ℝ} (ha : a > 0) (hb : b > 0)
  (h_line : ∀ x : ℝ, ∃ y : ℝ, y = x + 2)
  (h_hyperbola : ∀ x y : ℝ, (x / a)^2 - (y / b)^2 = 1)
  (h_midpoint : ∃ (x1 x2 y1 y2 : ℝ), (x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = 3 ∧ (h_line x1 = y1) ∧ (h_line x2 = y2) ∧ (h_hyperbola x1 y1 = 1) ∧ (h_hyperbola x2 y2 = 1)) :
  let c := sqrt (a^2 + b^2) in
  c / a = 2 :=
by
  sorry

end hyperbola_eccentricity_l141_141126


namespace total_product_l141_141037

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 12 
  else if n % 2 = 0 then 4 
  else 0 

def allie_rolls : List ℕ := [2, 6, 3, 1, 6]
def betty_rolls : List ℕ := [4, 6, 3, 5]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem total_product : total_points allie_rolls * total_points betty_rolls = 1120 := sorry

end total_product_l141_141037


namespace no_diagonal_between_two_nonadjacent_vertices_l141_141985

theorem no_diagonal_between_two_nonadjacent_vertices (n : ℕ) (h₀ : n ≥ 4) 
  (polygon : Type) [convex_polygon polygon] [vertices polygon n] 
  (h₁ : ∀ (d₁ d₂ : diagonal polygon), d₁ ≠ d₂ → (∀ {v₁ v₂ v₃}, adj v₁ v₂ ∧ adj v₂ v₃ → same_diagonal d₁ d₂))
  : ∃ (v₁ v₂ : vertex polygon), ¬(adj v₁ v₂) ∧ ¬(exists (d : diagonal polygon), connects d v₁ v₂) :=
sorry

end no_diagonal_between_two_nonadjacent_vertices_l141_141985


namespace min_m_value_inequality_x2y2z_l141_141909

theorem min_m_value (a b : ℝ) (h1 : a * b > 0) (h2 : a^2 * b = 2) : 
  ∃ (m : ℝ), m = a * b + a^2 ∧ m = 3 :=
sorry

theorem inequality_x2y2z 
  (t : ℝ) (ht : t = 3) (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x^2 + y^2 + z^2 = t / 3) : 
  |x + 2 * y + 2 * z| ≤ 3 :=
sorry

end min_m_value_inequality_x2y2z_l141_141909


namespace vector_properties_l141_141504

open Real

def vector_parallel (v w : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, w = (k * v.1, k * v.2, k * v.3)

def vector_projection (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let dot_uu := u.1 * u.1 + u.2 * u.2 + u.3 * u.3
  let k := dot_uv / dot_uu
  (k * u.1, k * u.2, k * u.3)

theorem vector_properties :
  let a := (1, 0, 1)
  let b := (-1, 2, -3)
  let c := (2, -4, 6)
  in vector_parallel b c ∧ vector_projection a c = (4, 0, 4) :=
by
  let a := (1, 0, 1)
  let b := (-1, 2, -3)
  let c := (2, -4, 6)
  sorry

end vector_properties_l141_141504


namespace equal_angles_proof_l141_141053

noncomputable def compute_equal_angles := 
  let t_positions := [t | t in [0, 2], 697 * t % 1 = 0 ∧ 730 * t % 1 = 0 ∧ -1427 * t % 1 = 0]
  let valid_positions := t_positions.filter (t ≠ 0 ∧ t ≠ 1 ∧ t ≠ 2 ∧ t ≠ 1/2 ∧ t ≠ 3/2)
  2 * valid_positions.length - 2 = 5700

theorem equal_angles_proof : compute_equal_angles = 5700 :=
sorry

end equal_angles_proof_l141_141053


namespace find_legs_of_triangle_l141_141214

-- Definition of the problem conditions
def right_triangle (x y : ℝ) := x * y = 200 ∧ 4 * (y - 4) = 8 * (x - 8)

-- Theorem we want to prove
theorem find_legs_of_triangle : 
  ∃ (x y : ℝ), right_triangle x y ∧ ((x = 40 ∧ y = 5) ∨ (x = 10 ∧ y = 20)) :=
by
  sorry

end find_legs_of_triangle_l141_141214


namespace correct_propositions_l141_141941

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 3

theorem correct_propositions : 
  let proposition1 := (∀ x, deriv f 1 = -3) 
                   ∧ (f 1 = 1) 
                   ∧ (∀ x y, (3 * x + y - 4 = 0) ↔ (y = -3 * (x - 1) + 1)),
      proposition2 := ∃ x1 x2 x3, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1,
      proposition4 := ∀ x, f (2 - x) = f x 
  in proposition1 
     ∧ proposition2 
     ∧ ¬ (∀ x, x = 2 → has_deriv_at f 0 x ∧ ∀ y, f y ≤ f x)
     ∧ proposition4 := 
by
  sorry

end correct_propositions_l141_141941


namespace license_plate_combinations_l141_141447

-- Definition for the conditions of the problem
def num_license_plate_combinations : ℕ :=
  let num_letters := 26
  let num_digits := 10
  let choose_two_distinct_letters := (num_letters * (num_letters - 1)) / 2
  let arrange_pairs := 2
  let choose_positions := 6
  let digit_permutations := num_digits ^ 2
  choose_two_distinct_letters * arrange_pairs * choose_positions * digit_permutations

-- The theorem we are proving
theorem license_plate_combinations :
  num_license_plate_combinations = 390000 :=
by
  -- The proof would be provided here.
  sorry

end license_plate_combinations_l141_141447


namespace determine_x_squared_plus_y_squared_l141_141849

theorem determine_x_squared_plus_y_squared :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (x * y + x + y = 119) ∧ ((x^2 * y + x * y^2) = 1680) ∧ (x^2 + y^2 = 1057) :=
begin
  sorry
end

end determine_x_squared_plus_y_squared_l141_141849


namespace hyperbola_eccentricity_l141_141728

noncomputable def point (α : Type) := (α × α)
noncomputable def focus {α : Type} (c : α) : point α := (c, 0)
noncomputable def hyperbola {α : Type} [linear_ordered_field α] (x y a b : α) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)
noncomputable def circle {α : Type} [linear_ordered_field α] (x y c b : α) : Prop :=
  ((x - c / 3)^2 + y^2 = b^2 / 9)
axiom tangent {α : Type} [linear_ordered_field α] (P F Q : point α) : Prop 
axiom vector_relation {α : Type} [linear_ordered_field α] (P Q F : point α) : Prop :=
  (2 * (fst Q - fst P), 2 * (snd Q - snd P)) = (fst F - fst Q, snd F - snd Q)

theorem hyperbola_eccentricity {α : Type} [linear_ordered_field α] (a b c : α) (x y : α) 
  (P Q F : point α) : 
  focus c = F →
  hyperbola x y a b →
  circle (fst Q) (snd Q) c b →
  tangent P F Q →
  vector_relation P Q F →
  (c = sqrt 5 * a) →
  true :=
sorry

end hyperbola_eccentricity_l141_141728


namespace fluorescent_tubes_count_l141_141319

theorem fluorescent_tubes_count 
  (x y : ℕ)
  (h1 : x + y = 13)
  (h2 : x / 3 + y / 2 = 5) : x = 9 :=
by
  sorry

end fluorescent_tubes_count_l141_141319


namespace M_inter_N_eq_23_l141_141389

variable (α : Type) [Preorder α] [BoundedOrder α]

def M (x : α) : Prop := 2 < x

def N (x : α) : Prop := x ≤ 3

theorem M_inter_N_eq_23 : {x | M x} ∩ {x | N x} = {x | 2 < x ∧ x ≤ 3} :=
by
  sorry

end M_inter_N_eq_23_l141_141389


namespace area_of_shaded_region_l141_141463

-- Definitions of the radii
def r₁ : ℝ := 5
def r₂ : ℝ := 4
def d : ℝ := r₁ + r₂ -- The distance between the centers due to tangency

-- Definition of the radius of the larger circle
def R : ℝ := d + r₂

-- Area calculations
def area_large_circle : ℝ := Real.pi * R^2
def area_small_circles : ℝ := Real.pi * r₁^2 + Real.pi * r₂^2

-- The statement of the theorem
theorem area_of_shaded_region : area_large_circle - area_small_circles = 128 * Real.pi := 
by sorry

end area_of_shaded_region_l141_141463


namespace find_a_b_fx_ge_gx_all_x_number_of_real_roots_l141_141898

-- Define the functions f and g
def f (x : ℝ) (a : ℝ) : ℝ := a * x^2 - x
def g (x : ℝ) (b : ℝ) : ℝ := b * Real.log x

-- Problem part (Ⅰ)
theorem find_a_b (a b : ℝ) (h1 : f 1 a = g 1 b) (h2 : Deriv f 1 a = Deriv g 1 b) : a = 1 ∧ b = 1 := by
  sorry

-- Problem part (Ⅱ)
theorem fx_ge_gx_all_x (a b : ℝ) (h1 : f 1 a = g 1 b) (h2 : Deriv f 1 a = Deriv g 1 b)
(h3 : a = 1) (h4 : b = 1) : ∀ x > 0, f x a ≥ g x b := by
  sorry

-- Problem part (Ⅲ)
theorem number_of_real_roots (n : ℝ) (hn : n ≥ 6) (a : ℝ) (b : ℝ)
(h1 : f 1 a = g 1 b) (h2 : Deriv f 1 a = Deriv g 1 b)
(h3 : a = 1) (h4 : b = 1) : ∃! x ∈ Ioo 1 (Real.exp n), f x a + x = n * g x b := by
  sorry

end find_a_b_fx_ge_gx_all_x_number_of_real_roots_l141_141898


namespace correct_equation_l141_141992

-- Definitions of the conditions
def contributes_5_coins (x : ℕ) (P : ℕ) : Prop :=
  5 * x + 45 = P

def contributes_7_coins (x : ℕ) (P : ℕ) : Prop :=
  7 * x + 3 = P

-- Mathematical proof problem
theorem correct_equation 
(x : ℕ) (P : ℕ) (h1 : contributes_5_coins x P) (h2 : contributes_7_coins x P) : 
5 * x + 45 = 7 * x + 3 := 
by
  sorry

end correct_equation_l141_141992


namespace original_group_men_l141_141003

-- Let's define the parameters of the problem
def original_days := 55
def absent_men := 15
def completed_days := 60

-- We need to show that the number of original men (x) is 180
theorem original_group_men (x : ℕ) (h : x * original_days = (x - absent_men) * completed_days) : x = 180 :=
by
  sorry

end original_group_men_l141_141003


namespace minimum_omega_value_l141_141649

noncomputable def min_omega (ω T x φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π) (hT : T = 2 * π / ω) 
  (hfT : Real.cos (ω * T + φ) = √3 / 2) (hx : Real.cos (ω * x + φ) = 0) : ℝ :=
  3

theorem minimum_omega_value (ω T x φ : ℝ)
  (hω : ω > 0)
  (hφ : 0 < φ ∧ φ < π)
  (hT : T = 2 * π / ω)
  (hfT : Real.cos (ω * T + φ) = √3 / 2)
  (hx : Real.cos (ω * x + φ) = 0)
  (hx_val : x = π / 9)
  (T_val : T = 2 * π / ω) :
  ω = min_omega ω T (π / 9) (π / 6) :=
begin
  sorry -- Proof steps are skipped as instructed.
end

end minimum_omega_value_l141_141649


namespace race_result_130m_l141_141780

theorem race_result_130m (d : ℕ) (t_a t_b: ℕ) (a_speed b_speed : ℚ) (d_a_t : ℚ) (d_b_t : ℚ) (distance_covered_by_B_in_20_secs : ℚ) :
  d = 130 →
  t_a = 20 →
  t_b = 25 →
  a_speed = (↑d) / t_a →
  b_speed = (↑d) / t_b →
  d_a_t = a_speed * t_a →
  d_b_t = b_speed * t_b →
  distance_covered_by_B_in_20_secs = b_speed * 20 →
  (d - distance_covered_by_B_in_20_secs = 26) :=
by
  sorry

end race_result_130m_l141_141780


namespace equal_areas_of_triangles_in_circle_l141_141594

noncomputable theory

open_locale classical

variables {A B C D E F Q O : Type*} [euclidean_space A] [euclidean_space B] [euclidean_space C]

def points_on_circle (O : Type*) (points : set Type*) : Prop :=
  ∃ (r : ℝ) (center : O), ∀ (p : Type*), p ∈ points → dist p center = r

def perpendicular (line1 line2 : Type*) : Prop :=
  ∃ (p : Type*), p ∈ line1 ∧ p ∈ line2 ∧ ∀ (x : Type*), x ∈ line1 → ∃ y ∈ line2, angle p x y = 90

axiom triangle (A B C : Type*) : Prop

axiom area_eq_of_areas (ABC QBE QCF : Type*) : Prop 

theorem equal_areas_of_triangles_in_circle 
  (triangle_ABC : triangle A B C)
  (circle_O : points_on_circle O {B, C})
  (intersect_E : E ∈ (AB))
  (intersect_F : F ∈ (AC))
  (perpendicular_AD : perpendicular (A, D) (BC))
  (intersect_D : D ∈ circle_ABC)
  (intersect_Q : Q ∈ (DO))
  (on_circle : points_on_circle ABC (Γ))
  : area_eq_of_areas (QBE) (QCF) :=
sorry

end equal_areas_of_triangles_in_circle_l141_141594


namespace same_cost_for_same_sheets_l141_141429

def John's_Photo_World_cost (x : ℕ) : ℝ := 2.75 * x + 125
def Sam's_Picture_Emporium_cost (x : ℕ) : ℝ := 1.50 * x + 140

theorem same_cost_for_same_sheets :
  ∃ (x : ℕ), John's_Photo_World_cost x = Sam's_Picture_Emporium_cost x ∧ x = 12 :=
by
  sorry

end same_cost_for_same_sheets_l141_141429


namespace area_of_semicircle_l141_141394

theorem area_of_semicircle (w : ℝ) (h : ℝ) (hw : w = 1) (hh : h = 3):
  let d := Math.sqrt (w^2 + h^2)
  let r := d / 2
  (π * (r^2) / 2 = (13 * π / 8)) :=
  sorry

end area_of_semicircle_l141_141394


namespace find_a1_l141_141306

-- Define the sequence satisfying the given conditions
def sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 2, (finset.range n).sum a = n^2 * a n

-- Given information: a_63 = 1 and this sequence property
theorem find_a1 (a : ℕ → ℕ) (h_seq : sequence a) (h : a 63 = 1) : a 1 = 2016 :=
begin
  sorry
end

end find_a1_l141_141306


namespace particle_probability_l141_141010

def P : ℕ × ℕ → ℚ
| (0, 0) := 1
| (x, 0) := 0
| (0, y) := 0
| (x + 1, y + 1) := (P (x, y + 1) + P (x + 1, y) + P (x, y)) / 3

theorem particle_probability (x y : ℕ) (h : (x, y) = (3, 5)) :
  let p := 320
  let q := 7
  P (3, 5) = p / (3^q) ∧ p % 3 ≠ 0 ∧ (p + q = 327) :=
by 
  have h1 : P (3, 5) = 320 / 2187,
  -- provide the missing reasoning and recursive evaluations
  sorry

end particle_probability_l141_141010


namespace intersection_unique_one_point_l141_141502

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 7 * x + a
noncomputable def g (x : ℝ) : ℝ := -3 * x^2 + 5 * x - 6

theorem intersection_unique_one_point (a : ℝ) :
  (∃ x y, y = f x a ∧ y = g x) ↔ a = 3 := by
  sorry

end intersection_unique_one_point_l141_141502


namespace arc_MTN_range_l141_141202

theorem arc_MTN_range
  {a b : ℝ}
  (h : a > 0 ∧ b > 0)
  (ΔABC : Type)
  (right_triangle : ∃ A B C : ΔABC, right_angle ∠ABC)
  (circle : Type)
  (radius_eq : ∃ R : ℝ, R = (1 / 2) * (real.sqrt (a^2 + b^2)))
  (T : ΔABC → circle → Type) -- T represents the tangential point on hypotenuse
  (M N : ΔABC → circle → Type) -- M and N represent intersection points
  (arc_varies : ∃ n : ℝ, ∀ T M N, n ∈ set.Icc 0 120) : 
  true :=
sorry

end arc_MTN_range_l141_141202


namespace general_term_a_n_sum_T_2n_l141_141514

-- Define the sequence a_n and its properties
def a_n (n : ℕ) : ℝ := if n > 0 then 2 * ((1/2)^(n-1)) else 0

-- Define the sum S_n
def S_n (n : ℕ) : ℝ := 4 - a_n n

-- Define the sequence b_n based on whether n is odd or even
def b_n (n : ℕ) : ℝ :=
  if n % 2 = 1 then log (1 / 2) (a_n n)
  else a_n n

-- Define the sum of the first 2n terms T_{2n}
def T_2n (n : ℕ) : ℝ :=
  (∑ i in Finset.range (2*n), b_n (i + 1))

theorem general_term_a_n (n : ℕ) (hn : 0 < n) :
  a_n n = (1 / 2)^(n - 2) :=
by
  -- The proof is skipped here.
  sorry

theorem sum_T_2n (n : ℕ) :
  T_2n n = n^2 - 2*n + (4/3) * (1 - (1/4)^n) :=
by
  -- The proof is skipped here.
  sorry

end general_term_a_n_sum_T_2n_l141_141514


namespace volume_of_glass_l141_141760

theorem volume_of_glass (bottle_start_volume : ℝ) (bottle_end_fraction : ℝ) (glass_fraction : ℝ) (glass_volume : ℝ) :
  bottle_start_volume = 1.5 →
  bottle_end_fraction = 3 / 4 →
  glass_fraction = 3 / 4 →
  (bottle_start_volume - (bottle_end_fraction * bottle_start_volume)) = (glass_fraction * glass_volume) →
  glass_volume = 0.5 :=
begin
  intros h1 h2 h3 h4,
  sorry
end

end volume_of_glass_l141_141760


namespace limit_sequence_equiv_l141_141386

noncomputable def problem_limit_sequence : Real :=
  lim (fun n : ℕ => ((2 * n + 1) ^ 3 - (2 * n + 3) ^ 3) / ((2 * n + 1) ^ 2 + (2 * n + 3) ^ 2))

theorem limit_sequence_equiv (L : Real) (hL : L = problem_limit_sequence) :
  L = -15 / 8 := sorry

end limit_sequence_equiv_l141_141386


namespace sub_question_1_sub_question_2_sub_question_3_l141_141140

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := log (4^x + 1) / log 2 + k * x

-- Proof for sub-question 1
theorem sub_question_1 (k : ℝ) (h : ∀ x : ℝ, f (-x) k = f x k) : k = -1 :=
sorry

noncomputable def g (x : ℝ) : ℝ := log (4^x + 1) / log 2 - 2 * x

-- Proof for sub-question 2
theorem sub_question_2 (a : ℝ) : (∀ x : ℝ, g x > 0) -> a ∈ set.Iic (0 : ℝ) :=
sorry

noncomputable def h (x : ℝ) (m : ℝ) : ℝ := 2^(f x (-1) + x) + m * 2^x - 1

-- Proof for sub-question 3
theorem sub_question_3 (m : ℝ) (h : ∀ x ∈ set.Icc (0 : ℝ) (log 3 / log 2), h x m >= 0) : m = -1 :=
sorry

end sub_question_1_sub_question_2_sub_question_3_l141_141140


namespace geom_inequality_l141_141923

noncomputable def geom_seq_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then n * a1 else a1 * (1 - q^n) / (1 - q)

theorem geom_inequality (a1 q : ℝ) (h_q : q ≠ 0) :
  (a1 * (a1 * q^2)) > 0 :=
by
  sorry

end geom_inequality_l141_141923


namespace jet_distance_l141_141004

theorem jet_distance (distance : ℝ) (time : ℝ) (new_time : ℝ) :
  distance = 580 → time = 2 → new_time = 10 → (distance / time) * new_time = 2900 :=
by
  intros h_distance h_time h_new_time
  simp [h_distance, h_time, h_new_time]
  norm_num
  sorry

end jet_distance_l141_141004


namespace proof_problem_l141_141732

-- Problem Statement: Given x satisfies the equation
def problem_statement (x : ℝ) : Prop :=
  x + 1 / x = 3 → x^10 - 6 * x^6 + x^2 = -328 * x^2

axiom x_value (x : ℝ) : x + 1 / x = 3

theorem proof_problem {x : ℝ} (h : x_value x) : problem_statement x := by
  sorry

end proof_problem_l141_141732


namespace ratio_second_week_to_first_week_l141_141247

def cases_week_1 : ℤ := 5000
def cases_week_2 : ℤ := 1250
def cases_week_3 : ℤ := cases_week_2 + 2000
def total_cases := cases_week_1 + cases_week_2 + cases_week_3

theorem ratio_second_week_to_first_week : 
  total_cases = 9500 → 
  (cases_week_2 : ℚ) / cases_week_1 = 1 / 4 :=
by
  intros h_total_cases
  have h_cases_week_2 : cases_week_2 = 1250,
  sorry
  have h_cases_week_1 : cases_week_1 = 5000,
  sorry
  rw [h_cases_week_2, h_cases_week_1],
  norm_num

end ratio_second_week_to_first_week_l141_141247


namespace slope_of_line_l141_141854

theorem slope_of_line (x y : ℝ) (h : x / 4 + y / 3 = 1) : ∀ m : ℝ, (y = m * x + 3) → m = -3/4 :=
by
  sorry

end slope_of_line_l141_141854


namespace trig_identity_l141_141164

variable (α : ℝ)
variable (h : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
variable (h₁ : Real.sin α = 4 / 5)

theorem trig_identity : Real.sin (α + Real.pi / 4) + Real.cos (α + Real.pi / 4) = -3 * Real.sqrt 2 / 5 := 
by 
  sorry

end trig_identity_l141_141164


namespace find_x_l141_141978

/-
If two minus the reciprocal of (3 - x) equals the reciprocal of (2 + x), 
then x equals (1 + sqrt(15)) / 2 or (1 - sqrt(15)) / 2.
-/
theorem find_x (x : ℝ) :
  (2 - (1 / (3 - x)) = (1 / (2 + x))) → 
  (x = (1 + Real.sqrt 15) / 2 ∨ x = (1 - Real.sqrt 15) / 2) :=
by 
  sorry

end find_x_l141_141978


namespace probability_of_meeting_l141_141359

noncomputable def meeting_probability : ℝ := 
  let f := λ (x y : ℝ), |x - y| ≤ (1 / 4)
  let P := MeasureTheory.Measure.univ_measure
  MeasureTheory.Probability.ProbabilityMeasure.measure (MeasureTheory.Interval.unitSquare) {p : ℝ × ℝ | f p.1 p.2}

theorem probability_of_meeting : meeting_probability = 7 / 16 :=
by
  sorry

end probability_of_meeting_l141_141359


namespace integer_solution_f_g_f_eq_g_f_g_l141_141145

def f (x: ℤ): ℤ := x^2 + 4 * x + 3
def g (x: ℤ): ℤ := x^2 + 2 * x - 1

theorem integer_solution_f_g_f_eq_g_f_g : ∀ x: ℤ, f(g(f(x))) = g(f(g(x))) ↔ x = -2 := by
  sorry

end integer_solution_f_g_f_eq_g_f_g_l141_141145


namespace max_n_for_triangle_l141_141987

theorem max_n_for_triangle (a b c : ℝ) (A B C : ℝ) (h₁ : a = BC) (h₂ : b = AC) (h₃ : c = AB)
  (h₄ : ∠A + ∠C = 2 * ∠B) :
  a^4 + c^4 ≤ 2 * b^4 := 
sorry

end max_n_for_triangle_l141_141987


namespace div_by_self_condition_l141_141374

theorem div_by_self_condition (n : ℤ) (h : n^2 + 1 ∣ n) : n = 0 :=
by sorry

end div_by_self_condition_l141_141374


namespace mark_trees_total_l141_141242

def mark_trees (current_trees new_trees : Nat) : Nat :=
  current_trees + new_trees

theorem mark_trees_total (x y : Nat) (h1 : x = 13) (h2 : y = 12) :
  mark_trees x y = 25 :=
by
  rw [h1, h2]
  sorry

end mark_trees_total_l141_141242


namespace midpoint_trajectory_l141_141518

-- Define the data for the problem
variable (P : (ℝ × ℝ)) (Q : ℝ × ℝ)
variable (M : ℝ × ℝ)
variable (x y : ℝ)
variable (hQ : Q = (2*x - 2, 2*y)) -- Definition of point Q based on midpoint M
variable (hC : (Q.1)^2 + (Q.2)^2 = 1) -- Q moves on the circle x^2 + y^2 = 1

-- Define the proof problem
theorem midpoint_trajectory (P : (ℝ × ℝ)) (hP : P = (2, 0)) (M : ℝ × ℝ) (hQ : Q = (2*M.1 - 2, 2*M.2))
  (hC : (Q.1)^2 + (Q.2)^2 = 1) : 4*(M.1 - 1)^2 + 4*(M.2)^2 = 1 := by
  sorry

end midpoint_trajectory_l141_141518


namespace vector_properties_l141_141505

open Real

def vector_parallel (v w : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, w = (k * v.1, k * v.2, k * v.3)

def vector_projection (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let dot_uu := u.1 * u.1 + u.2 * u.2 + u.3 * u.3
  let k := dot_uv / dot_uu
  (k * u.1, k * u.2, k * u.3)

theorem vector_properties :
  let a := (1, 0, 1)
  let b := (-1, 2, -3)
  let c := (2, -4, 6)
  in vector_parallel b c ∧ vector_projection a c = (4, 0, 4) :=
by
  let a := (1, 0, 1)
  let b := (-1, 2, -3)
  let c := (2, -4, 6)
  sorry

end vector_properties_l141_141505


namespace decimal_digits_mod_l141_141847

theorem decimal_digits_mod (a b c : ℕ) (h : 37 ∣ (a * 10^4005 * (10^2001 - 1) // 9 + b * 10^2002 + c * 10^2001 * (10^2001 - 1) // 9)) :
  b = a + c :=
sorry

end decimal_digits_mod_l141_141847


namespace leading_digits_of_powers_of_two_are_non_periodic_l141_141254

theorem leading_digits_of_powers_of_two_are_non_periodic : 
  ∀ n : ℕ, ∃ m : ℕ, leading_digit (2 ^ (2 ^ n)) ≠ leading_digit (2 ^ (2 ^ (n + m))) :=
sorry

end leading_digits_of_powers_of_two_are_non_periodic_l141_141254


namespace simplify_and_evaluate_expression_l141_141266

theorem simplify_and_evaluate_expression (x y : ℝ) (hx : x = -1) (hy : y = -1) :
  (5 * x ^ 2 - 2 * (3 * y ^ 2 + 6 * x) + (2 * y ^ 2 - 5 * x ^ 2)) = 8 :=
by
  sorry

end simplify_and_evaluate_expression_l141_141266


namespace number_of_paths_l141_141472

-- Define the cube structure
structure CubeGraph := 
  (vertices : Fin 8) 
  (edges : Fin 12)

-- Define that Erin starts at a given corner (vertex 0) of a cube.
def initial_vertex : Fin 8 := 0

-- Define the condition for Erin to visit every vertex exactly once.
def visit_every_vertex_once (path : List (Fin 8)) : Prop :=
  ∀ v : Fin 8, v ∈ path ∧ path.Nodup

-- Define the condition of crawling exactly 7 edges.
def exact_seven_edges (path : List (Fin 8)) : Prop :=
  path.length = 8

-- Define the condition that she cannot return to the starting point along an edge.
def no_return_to_start (path : List (Fin 8)) : Prop :=
  path.head = initial_vertex ∧ path.last ≠ initial_vertex

-- Prove the number of such valid paths is 6.
theorem number_of_paths : ∃ (paths : List (Fin 8)), 
  visit_every_vertex_once paths ∧
  exact_seven_edges paths ∧ 
  no_return_to_start paths ∧ 
  paths.length = 6 := 
sorry

end number_of_paths_l141_141472


namespace parametric_second_derivative_l141_141493

noncomputable def second_derivative_parametric (t : ℝ) (h1 : t > 1) (h2 : t < 1) : ℝ :=
  let x := sqrt (t - 1)
  let y := t / sqrt (1 - t)
  have h : t ≠ 1 := by { intro h3, linarith }
  have dx_dt := (1 / (2 * sqrt (t - 1))) : ℝ := by sorry
  have dy_dt := (2 - t) / (2 * (1 - t) ^ (3 / 2)) : ℝ := by sorry
  have dy_dx := ((2 - t) sqrt (t - 1) / (1 - t) ^ (3 / 2)) : ℝ := by sorry
  have d2y_dx2 := 2 / (1 - t) ^ (3 / 2) : ℝ := by sorry
  d2y_dx2

theorem parametric_second_derivative {t : ℝ} (ht : t > 1 ∧ t < 1) :
  second_derivative_parametric t ht.1 ht.2 = 2 / sqrt ((1 - t) ^ 3) := by sorry

end parametric_second_derivative_l141_141493


namespace probability_both_numbers_prime_l141_141358

open Finset

def primes_between_1_and_30 : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_both_numbers_prime :
  (∃ (a b : ℕ), a ≠ b ∧ a ∈ primes_between_1_and_30 ∧ b ∈ primes_between_1_and_30 ∧
  (45 / 435) = (1 / 9)) :=
by
  have h_prime_count : ∃ (s : Finset ℕ), s.card = 10 ∧ s = primes_between_1_and_30 := sorry,
  have h_total_pairs : ∃ (n : ℕ), n = 435 := sorry,
  have h_prime_pairs : ∃ (m : ℕ), m = 45 := sorry,
  have h_fraction : (45 : ℚ) / 435 = (1 : ℚ) / 9 := sorry,
  exact ⟨45, 435, by simp [h_prime_pairs, h_total_pairs, h_fraction]⟩

end probability_both_numbers_prime_l141_141358


namespace second_printer_time_l141_141795

noncomputable def rate_first_printer : ℝ := 800 / 10

noncomputable def combined_rate : ℝ := 800 / 3

theorem second_printer_time :
  let rate_second_printer := combined_rate - rate_first_printer in
  let time_second_printer_to_print_800_flyers := 800 / rate_second_printer in
  time_second_printer_to_print_800_flyers = 30 / 7 :=
by
  let rate_second_printer := combined_rate - rate_first_printer
  have h1: rate_second_printer = 560 / 3 := by sorry
  have h2: time_second_printer_to_print_800_flyers = 800 / rate_second_printer := rfl
  have h3: 800 / rate_second_printer = 30 / 7 := by sorry
  exact Eq.trans h2 h3

end second_printer_time_l141_141795


namespace minimum_omega_is_3_l141_141623

variable (f : ℝ → ℝ)
variable (ω φ T : ℝ)
variable (x : ℝ)

noncomputable def minimum_omega (ω : ℝ) : ℝ := 3 -- Temporary noncomputable definition to hold the value for min omega

theorem minimum_omega_is_3 :
  (∀ x, f x = Real.cos (ω * x + φ)) →
  ω > 0 →
  0 < φ ∧ φ < Real.pi →
  T = (2 * Real.pi) / ω →
  f T = Real.sqrt 3 / 2 →
  f (Real.pi / 9) = 0 →
  ∃(ω_min : ℝ), ω_min = 3 :=
by
  intros h0 h1 h2 h3 h4 h5
  use 3
  sorry -- Proof of the exact answer

end minimum_omega_is_3_l141_141623


namespace probability_of_two_primes_l141_141350

-- Define the set of integers from 1 to 30
def finite_set : set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the set of prime numbers from 1 to 30
def primes_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Define the probability of choosing two different primes
def probability_two_primes : ℚ :=
  (fintype.card {p : ℕ × ℕ // p.1 < p.2 ∧ p.1 ∈ primes_set ∧ p.2 ∈ primes_set}) /
  (fintype.card {p : ℕ × ℕ // p.1 < p.2 ∧ p.1 ∈ finite_set ∧ p.2 ∈ finite_set})

-- Prove that the probability is 1/29
theorem probability_of_two_primes :
  probability_two_primes = 1 / 29 :=
sorry

end probability_of_two_primes_l141_141350


namespace separate_balls_with_one_weighing_l141_141038

theorem separate_balls_with_one_weighing :
  ∃ (n : ℕ), n = 1 ∧
    let balls : list (ℕ × ℝ) := (list.repeat (10: ℝ) 1000).map (prod.mk 10) ++ (list.repeat (9.9: ℝ) 1000).map (prod.mk 9.9)
    in ∃ (b1 b2 : list (ℕ × ℝ)), (b1.length = 1000 ∧ b2.length = 1000 ∧ b1.sum (λ p, p.1 * p.2) ≠ b2.sum (λ p, p.1 * p.2)) :=
begin
  let balls : list (ℕ × ℝ) := (list.repeat (10: ℝ) 1000).map (prod.mk 10) ++ (list.repeat (9.9: ℝ) 1000).map (prod.mk 9.9),
  existsi 1,
  split,
  { refl, },
  { sorry, }
end

end separate_balls_with_one_weighing_l141_141038


namespace pentagon_angle_E_l141_141436

theorem pentagon_angle_E {ABCDE : Type*} [pentagon ABCDE] 
  (h1 : side_length_eq ABCDE)
  (h2 : ∠A = 120 ∧ ∠B = 120) : 
  ∠E = 90 := 
sorry

end pentagon_angle_E_l141_141436


namespace fifth_term_arithmetic_sequence_l141_141716

theorem fifth_term_arithmetic_sequence (x y : ℝ) (a₁ a₂ a₃ a₄ : ℝ) 
  (h1 : a₁ = x + 2 * y) 
  (h2 : a₂ = x - 2 * y) 
  (h3 : a₃ = x + y) 
  (h4 : a₄ = x - y) : 
  a₂ - a₁ = -4 * y → 
  a₃ - a₂ = -4 * y → 
  a₄ - a₃ = -4 * y → 
  let a₅ := a₄ - 4 * y 
  in a₅ = x :=
sorry

end fifth_term_arithmetic_sequence_l141_141716


namespace integer_b_divides_polynomial_l141_141096

theorem integer_b_divides_polynomial:
  ∃ (q : ℤ[X]), (∃ b : ℤ, (∀ x : ℤ, ((↑x^2 - 2 * x + b) * q = x^13 - x + 60)) → b = 2) :=
sorry

end integer_b_divides_polynomial_l141_141096


namespace exp_add_exp_nat_mul_l141_141685

noncomputable def Exp (z : ℝ) : ℝ := Real.exp z

theorem exp_add (a b x : ℝ) :
  Exp ((a + b) * x) = Exp (a * x) * Exp (b * x) := sorry

theorem exp_nat_mul (x : ℝ) (k : ℕ) :
  Exp (k * x) = (Exp x) ^ k := sorry

end exp_add_exp_nat_mul_l141_141685


namespace grid_coloring_probability_sum_l141_141860

theorem grid_coloring_probability_sum : 
  ∃ m n : ℕ, 
    Nat.gcd m n = 1 ∧ 
    (m + n = 929) ∧ 
    (m / n = 417 / 512) :=
by
  sorry

end grid_coloring_probability_sum_l141_141860


namespace function_identity_l141_141479

-- Definitions of the problem
def f (n : ℕ) : ℕ := sorry

-- Main theorem to prove
theorem function_identity (f : ℕ → ℕ) 
  (h : ∀ m n : ℕ, m > 0 → n > 0 → f (m + n) * f (m - n) = f (m * m)) : 
  ∀ n : ℕ, n > 0 → f n = 1 := 
sorry

end function_identity_l141_141479


namespace parametric_second_derivative_l141_141492

noncomputable def second_derivative_parametric (t : ℝ) (h1 : t > 1) (h2 : t < 1) : ℝ :=
  let x := sqrt (t - 1)
  let y := t / sqrt (1 - t)
  have h : t ≠ 1 := by { intro h3, linarith }
  have dx_dt := (1 / (2 * sqrt (t - 1))) : ℝ := by sorry
  have dy_dt := (2 - t) / (2 * (1 - t) ^ (3 / 2)) : ℝ := by sorry
  have dy_dx := ((2 - t) sqrt (t - 1) / (1 - t) ^ (3 / 2)) : ℝ := by sorry
  have d2y_dx2 := 2 / (1 - t) ^ (3 / 2) : ℝ := by sorry
  d2y_dx2

theorem parametric_second_derivative {t : ℝ} (ht : t > 1 ∧ t < 1) :
  second_derivative_parametric t ht.1 ht.2 = 2 / sqrt ((1 - t) ^ 3) := by sorry

end parametric_second_derivative_l141_141492


namespace Joe_ag_is_38_l141_141217

-- Define Joe and Jane's ages as real numbers
variables (J A : ℝ)

-- The conditions from the problem
def conditions := (J + A = 54) ∧ (J - A = 22)

-- The statement to prove Joe's age
theorem Joe_ag_is_38 (h : conditions J A) : J = 38 :=
by
  cases h with h1 h2
  sorry

end Joe_ag_is_38_l141_141217


namespace average_number_of_ducks_l141_141814

def average_ducks (A E K : ℕ) : ℕ :=
  (A + E + K) / 3

theorem average_number_of_ducks :
  ∀ (A E K : ℕ), A = 2 * E → E = K - 45 → A = 30 → average_ducks A E K = 35 :=
by 
  intros A E K h1 h2 h3
  sorry

end average_number_of_ducks_l141_141814


namespace find_a1_l141_141305

-- Define the sequence satisfying the given conditions
def sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 2, (finset.range n).sum a = n^2 * a n

-- Given information: a_63 = 1 and this sequence property
theorem find_a1 (a : ℕ → ℕ) (h_seq : sequence a) (h : a 63 = 1) : a 1 = 2016 :=
begin
  sorry
end

end find_a1_l141_141305


namespace limit_a_n_l141_141661

variable (α : ℝ) (hα : α > 0) (a : ℕ → ℝ)
noncomputable def a_n (n : ℕ) : ℝ := (α + n / n) ^ n

theorem limit_a_n (hα : α > 0) : 
  e^α < limit (nat.irange (λ n, a n) at_top) < e^(α + 1) := sorry

end limit_a_n_l141_141661


namespace solve_for_a_l141_141129

theorem solve_for_a (x a : ℤ) (h : 2 * x - a - 5 = 0) (hx : x = 3) : a = 1 :=
by sorry

end solve_for_a_l141_141129


namespace area_trDEF_l141_141181

noncomputable def area_of_triangle (DE EF DF : ℝ) : ℝ :=
  let s := (DE + EF + DF) / 2
  sqrt (s * (s - DE) * (s - EF) * (s - DF))

theorem area_trDEF (DE EF DF: ℝ) (hDE: DE = 31) (hEF: EF = 31) (hDF: DF = 40) : 
  area_of_triangle DE EF DF = 474 := by
  rw [hDE, hEF, hDF]
  sorry

end area_trDEF_l141_141181


namespace which_polygon_covers_ground_l141_141372

def is_tessellatable (n : ℕ) : Prop :=
  let interior_angle := (n - 2) * 180 / n
  360 % interior_angle = 0

theorem which_polygon_covers_ground :
  is_tessellatable 6 ∧ ¬is_tessellatable 5 ∧ ¬is_tessellatable 8 ∧ ¬is_tessellatable 12 :=
by
  sorry

end which_polygon_covers_ground_l141_141372


namespace sum_powers_of_5_mod_7_l141_141769

theorem sum_powers_of_5_mod_7 : 
  (∑ k in Finset.range 1001, 5^k) % 7 = 4 :=
sorry

end sum_powers_of_5_mod_7_l141_141769


namespace solve_complex_eq_l141_141851

-- Define complex number and its conjugate
def z (a b : ℝ) := a + b * complex.i
def z_conj (a b : ℝ) := a - b * complex.i

-- Define the problem statement as a theorem
theorem solve_complex_eq (a b : ℝ) (hz : 3 * z a b - 4 * z_conj a b = 5 + 24 * complex.i) :
  z a b = -5 + (24 / 7) * complex.i :=
by
  sorry

end solve_complex_eq_l141_141851


namespace find_m_minus_n_l141_141509

theorem find_m_minus_n (m n : ℤ) (h1 : |m| = 14) (h2 : |n| = 23) (h3 : m + n > 0) : m - n = -9 ∨ m - n = -37 := 
sorry

end find_m_minus_n_l141_141509


namespace inv_101_mod_102_l141_141865

theorem inv_101_mod_102 : (101 : ℤ) * 101 ≡ 1 [MOD 102] := by
  sorry

end inv_101_mod_102_l141_141865


namespace solve_for_x_l141_141068

noncomputable def f1 (x : ℝ) : ℝ :=
  (4 * x - 3) / (4 * x + 2)

noncomputable def fn : ℕ → ℝ → ℝ
| 1, x := f1 x
| n + 1, x := f1 (fn n x)

theorem solve_for_x : ∃ x : ℝ, fn 1001 x = x - 2 :=
by
  sorry

end solve_for_x_l141_141068


namespace second_order_derivative_y_l141_141495

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

def x (t : ℝ) : ℝ := sqrt (t - 1)
def y (t : ℝ) : ℝ := t / sqrt (1 - t)

theorem second_order_derivative_y''_xx (t : ℝ) (h1 : t > 1) (h2 : t < 1) : 
  let x := sqrt (t - 1)
  let y := t / sqrt (1 - t)
  y''_xx = (2 : ℝ) / sqrt ((1 - t)^3) :=
begin
  sorry
end

end second_order_derivative_y_l141_141495


namespace total_play_time_in_hours_l141_141777

def football_time : ℕ := 60
def basketball_time : ℕ := 60

theorem total_play_time_in_hours : (football_time + basketball_time) / 60 = 2 := by
  sorry

end total_play_time_in_hours_l141_141777


namespace sum_of_angles_in_equilateral_triangle_l141_141191

open Real

def equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

def segment_division (B C A1 A2 : ℝ × ℝ) : Prop :=
  dist B A1 = dist A1 A2 ∧ dist A1 A2 = dist A2 C

def ratio_division (A C B1 : ℝ × ℝ) : Prop :=
  dist A B1 / dist B1 C = 1/2

theorem sum_of_angles_in_equilateral_triangle (A B C A1 A2 B1 : ℝ × ℝ)
  (h1 : equilateral_triangle A B C)
  (h2 : segment_division B C A1 A2)
  (h3 : ratio_division A C B1) :
  ∠ A A1 B1 + ∠ A A2 B1 = 30 :=
sorry

end sum_of_angles_in_equilateral_triangle_l141_141191


namespace minimum_omega_formula_l141_141655

noncomputable def minimumOmega (f : ℝ → ℝ) (x T ω φ : ℝ) : ℝ :=
  if (f = λ x, Real.cos (ω * x + φ)) 
     ∧ (ω > 0) 
     ∧ (0 < φ < π)
     ∧ (f T = √3 / 2)
     ∧ (f (π / 9) = 0) 
     ∧ (T = 2 * π / ω) 
  then 3 else 0

theorem minimum_omega_formula {f : ℝ → ℝ} {x T ω φ : ℝ} :
  (f = λ x, Real.cos (ω * x + φ)) 
  ∧ (ω > 0) 
  ∧ (0 < φ < π)
  ∧ (f T = √3 / 2)
  ∧ (f (π / 9) = 0)
  ∧ (T = 2 * π / ω) 
  → minimumOmega f x T ω φ = 3 :=
by
  sorry

end minimum_omega_formula_l141_141655


namespace angle_between_vectors_l141_141546

variables (a b : ℝ^2)

-- Given conditions
axiom length_a : ‖a‖ = 3
axiom length_b : ‖b‖ = 2
axiom dot_eqn : (a + b) • (a - 2 • b) = 4

-- Proof goal: the angle between vectors a and b is 2π/3.
theorem angle_between_vectors : 
  let angle := Real.arccos ((a • b) / (‖a‖ * ‖b‖)) in 
  angle = 2 * Real.pi / 3 :=
by 
  -- We skip the proof as it's not required
  sorry

end angle_between_vectors_l141_141546


namespace arcade_playtime_l141_141599

noncomputable def cost_per_six_minutes : ℝ := 0.50
noncomputable def total_spent : ℝ := 15
noncomputable def minutes_per_interval : ℝ := 6
noncomputable def minutes_per_hour : ℝ := 60

theorem arcade_playtime :
  (total_spent / cost_per_six_minutes) * minutes_per_interval / minutes_per_hour = 3 :=
by
  sorry

end arcade_playtime_l141_141599


namespace linear_eq_exp_l141_141971

theorem linear_eq_exp (a b : ℤ) (h : (x^(a - 3) + y^(b - 1) = 0) → Linear (x^(a - 3) + y^(b - 1))) : 
  a = 4 ∧ b = 2 :=
sorry

end linear_eq_exp_l141_141971


namespace solve_a_plus_b_plus_c_l141_141273

theorem solve_a_plus_b_plus_c 
  (f : ℝ → ℝ)
  (a b c : ℝ)
  (h1 : ∀ x, f(x + 5) = 4 * x^2 + 9 * x + 6)
  (h2 : ∀ x, f x = a * x^2 + b * x + c) :
  a + b + c = 34 := 
sorry

end solve_a_plus_b_plus_c_l141_141273


namespace probability_two_primes_1_to_30_l141_141347

def is_prime (n : ℕ) : Prop := 
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13 ∨ n = 17 ∨ n = 19 ∨ n = 23 ∨ n = 29

def count_combinations (n k : ℕ) : ℕ := nat.choose n k

theorem probability_two_primes_1_to_30 :
  let total_combinations := count_combinations 30 2 in
  let prime_combinations := count_combinations 10 2 in
  total_combinations = 435 ∧ 
  prime_combinations = 45 ∧ 
  (prime_combinations : ℚ) / total_combinations = 15 / 145 :=
by { sorry }

end probability_two_primes_1_to_30_l141_141347


namespace equation_descr_circle_l141_141078

theorem equation_descr_circle : ∀ (x y : ℝ), (x - 0) ^ 2 + (y - 0) ^ 2 = 25 → ∃ (c : ℝ × ℝ) (r : ℝ), c = (0, 0) ∧ r = 5 ∧ ∀ (p : ℝ × ℝ), (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 :=
by
  sorry

end equation_descr_circle_l141_141078


namespace minimum_value_l141_141489

open Real

theorem minimum_value (x : ℝ) (h : 0 < x) : 
  ∃ y, (∀ z > 0, 3 * sqrt z + 2 / z ≥ y) ∧ y = 5 := by
  sorry

end minimum_value_l141_141489


namespace find_c_l141_141159

theorem find_c (c : ℝ) (h : Real.log c 27 = 0.75) : c = 81 := 
by
  sorry

end find_c_l141_141159


namespace minimize_sum_at_11_l141_141915

def arithmetic_sequence := ℕ → ℝ
def sum_of_first_n_terms (a : arithmetic_sequence) (n : ℕ) : ℝ := 
  if n = 0 then 0 else (n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1))

axiom S1_lt_0 (a : arithmetic_sequence) : sum_of_first_n_terms a 1 < 0
axiom S21_S25_condition (a : arithmetic_sequence) : 
  2 * sum_of_first_n_terms a 21 + sum_of_first_n_terms a 25 = 0

theorem minimize_sum_at_11 (a : arithmetic_sequence)
  (h₁ : S1_lt_0 a) (h₂ : S21_S25_condition a) : 
  ∀ n, sum_of_first_n_terms a n = sum_of_first_n_terms a 11 := 
by sorry

end minimize_sum_at_11_l141_141915


namespace minimum_omega_value_l141_141644

noncomputable def min_omega (ω T x φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π) (hT : T = 2 * π / ω) 
  (hfT : Real.cos (ω * T + φ) = √3 / 2) (hx : Real.cos (ω * x + φ) = 0) : ℝ :=
  3

theorem minimum_omega_value (ω T x φ : ℝ)
  (hω : ω > 0)
  (hφ : 0 < φ ∧ φ < π)
  (hT : T = 2 * π / ω)
  (hfT : Real.cos (ω * T + φ) = √3 / 2)
  (hx : Real.cos (ω * x + φ) = 0)
  (hx_val : x = π / 9)
  (T_val : T = 2 * π / ω) :
  ω = min_omega ω T (π / 9) (π / 6) :=
begin
  sorry -- Proof steps are skipped as instructed.
end

end minimum_omega_value_l141_141644


namespace sequence_initial_term_l141_141303

theorem sequence_initial_term :
  (∀ n ≥ 2, (finset.range n).sum (λ i, a (i + 1)) = n ^ 2 * a n) ∧ a 63 = 1 → a 1 = 2016 :=
begin
  apologize,
end

end sequence_initial_term_l141_141303


namespace distance_covered_l141_141079

theorem distance_covered 
  (T : ℝ) (S : ℝ) (hT : T = 36) (hS : S = 10) :
  let T_hours := T / 60 in
  let D := S * T_hours in
  D = 6 := 
by
  -- proof section
  sorry

end distance_covered_l141_141079


namespace part1_part2_l141_141138

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.log x - a * x ^ 2 + 1

theorem part1 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ (0 < a ∧ a < 1) :=
sorry

theorem part2 (a : ℝ) :
  (∃ α β m : ℝ, 1 ≤ α ∧ α ≤ 4 ∧ 1 ≤ β ∧ β ≤ 4 ∧ β - α = 1 ∧ f α a = m ∧ f β a = m) ↔ 
  (Real.log 4 / 3 * (2 / 7) ≤ a ∧ a ≤ Real.log 2 * (2 / 3)) :=
sorry

end part1_part2_l141_141138


namespace pepperoni_ratio_l141_141218

-- Definition of the problem's conditions
def total_pepperoni_slices : ℕ := 40
def slice_given_to_jelly_original : ℕ := 10
def slice_fallen_off : ℕ := 1

-- Our goal is to prove that the ratio is 3:10
theorem pepperoni_ratio (total_pepperoni_slices : ℕ) (slice_given_to_jelly_original : ℕ) (slice_fallen_off : ℕ) :
  (slice_given_to_jelly_original - slice_fallen_off) / (total_pepperoni_slices - slice_given_to_jelly_original) = 3 / 10 :=
by
  sorry

end pepperoni_ratio_l141_141218


namespace find_currents_l141_141994

-- Given conditions
def ε : ℝ := 69 -- EMF in volts
def R : ℝ := 10 -- Resistor value in ohms
def R1 : ℝ := R
def R2 : ℝ := 3 * R
def R3 : ℝ := 3 * R
def R4 : ℝ := 4 * R

-- Proof of required currents
theorem find_currents :
  let U13 := (ε * (3/4 * R)) / ((3/4 * R) + (12/7 * R)),
      I1 := U13 / R1,
      U24 := ε - U13,
      I2 := U24 / R2,
      I := I1 - I2
  in I1 = 2.1 ∧ I = 0.5 := by
  sorry

end find_currents_l141_141994


namespace integral_f_eq_six_plus_pi_l141_141282

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 - x else if x ≤ 2 then real.sqrt (4 - x^2) else 0
  
theorem integral_f_eq_six_plus_pi :
  ∫ x in (-2 : ℝ)..2, f x = 6 + real.pi := by
  sorry

end integral_f_eq_six_plus_pi_l141_141282


namespace tan_half_angle_in_fourth_quadrant_l141_141669

theorem tan_half_angle_in_fourth_quadrant (a : ℝ) (ha: a ∈ set.Ioc (3 * real.pi / 2) (2 * real.pi)) : 
  real.tan (a / 2) < 0 := 
sorry

end tan_half_angle_in_fourth_quadrant_l141_141669


namespace soldier_score_9_points_l141_141858

-- Define the conditions and expected result in Lean 4
theorem soldier_score_9_points (shots : List ℕ) :
  shots.length = 10 ∧
  (∀ shot ∈ shots, shot = 7 ∨ shot = 8 ∨ shot = 9 ∨ shot = 10) ∧
  shots.count 10 = 4 ∧
  shots.sum = 90 →
  shots.count 9 = 3 :=
by 
  sorry

end soldier_score_9_points_l141_141858


namespace integral_x2_plus_sin_l141_141835

theorem integral_x2_plus_sin (f : ℝ → ℝ) (a b : ℝ) :
  ∫ x in -1..1, (x^2 + sin x) = 2 / 3 :=
by
  have H₁ : ∫ x in -1..1, x^2 = 2 / 3, sorry
  have H₂ : ∫ x in -1..1, sin x = 0, sorry
  calc ∫ x in -1..1, (x^2 + sin x)
      = ∫ x in -1..1, x^2 + ∫ x in -1..1, sin x : by sorry
  ... = 2 / 3 + 0 : by sorry
  ... = 2 / 3 : by sorry

end integral_x2_plus_sin_l141_141835


namespace factor_expression_l141_141100

theorem factor_expression (x : ℝ) : 6 * x ^ 3 - 54 * x = 6 * x * (x + 3) * (x - 3) :=
by {
  sorry
}

end factor_expression_l141_141100


namespace prob_both_primes_l141_141342

-- Define the set of integers from 1 through 30
def int_set : Set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the set of prime numbers between 1 and 30
def primes_between_1_and_30 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Calculate the number of ways to choose two distinct elements from a set
noncomputable def combination (n k : ℕ) : ℕ := if k > n then 0 else n.choose k

-- Define the probabilities
noncomputable def prob_primes : ℚ :=
  (combination 10 2) / (combination 30 2)

-- State the theorem to prove
theorem prob_both_primes : prob_primes = 10 / 87 := by
  sorry

end prob_both_primes_l141_141342


namespace repeating_decimals_product_l141_141476

noncomputable def repeating_decimal_12 : ℚ :=
  let x := 0.1212121212... in
  (12 : ℚ) / (99 : ℚ)

noncomputable def repeating_decimal_3 : ℚ :=
  let y := 0.33333... in
  (1 : ℚ) / (3 : ℚ)

theorem repeating_decimals_product : (repeating_decimal_12 * repeating_decimal_3) = (4 / 99 : ℚ) :=
by {
  -- Proof goes here
  sorry
}

end repeating_decimals_product_l141_141476


namespace prime_probability_l141_141334

theorem prime_probability (P : Finset ℕ) : 
  (P = {p | p ≤ 30 ∧ Nat.Prime p}).card = 10 → 
  (Finset.Icc 1 30).card = 30 →
  ((Finset.Icc 1 30).card).choose 2 = 435 →
  (P.card).choose 2 = 45 →
  45 / 435 = 1 / 29 :=
by
  intros hP hIcc30 hChoose30 hChoosePrime
  sorry

end prime_probability_l141_141334


namespace hyperbola_equation_Q_on_fixed_circle_l141_141540

-- Define the hyperbola and necessary conditions
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / (3 * a^2) = 1

-- Given conditions
variables (a : ℝ) (h_pos : a > 0)
variables (F1 F2 : ℝ × ℝ)
variables (dist_F2_asymptote : ℝ) (h_dist : dist_F2_asymptote = sqrt 3)
variables (left_vertex : ℝ × ℝ) (right_branch_intersect : ℝ × ℝ)
variables (line_x_half : ℝ × ℝ)
variables (line_PF2 : ℝ × ℝ)
variables (point_Q : ℝ × ℝ)

-- Prove that the equation of the hyperbola is correct
theorem hyperbola_equation :
  hyperbola a x y ↔ x^2 - y^2 / 3 = 1 :=
sorry

-- Prove that point Q lies on a fixed circle
theorem Q_on_fixed_circle :
  dist point_Q F2 = 4 :=
sorry

end hyperbola_equation_Q_on_fixed_circle_l141_141540


namespace volume_of_cone_l141_141566

noncomputable def cone_volume (lateral_area : ℝ) (angle: ℝ): ℝ :=
let r := lateral_area / (20 * Mathlib.pi * Math.cos angle) in
let l := r * (Math.tan angle) in
let h := Math.sqrt (l^2 - r^2) in
(1/3) * Mathlib.pi * r^2 * h

theorem volume_of_cone (lateral_area : ℝ) (angle: ℝ) (h_angle: angle = Mathlib.arccos (4/5)) 
(h_lateral_area: lateral_area = 20 * Mathlib.pi): cone_volume lateral_area angle = 16 * Mathlib.pi :=
by
  sorry

end volume_of_cone_l141_141566


namespace minimum_omega_is_3_l141_141628

variable (f : ℝ → ℝ)
variable (ω φ T : ℝ)
variable (x : ℝ)

noncomputable def minimum_omega (ω : ℝ) : ℝ := 3 -- Temporary noncomputable definition to hold the value for min omega

theorem minimum_omega_is_3 :
  (∀ x, f x = Real.cos (ω * x + φ)) →
  ω > 0 →
  0 < φ ∧ φ < Real.pi →
  T = (2 * Real.pi) / ω →
  f T = Real.sqrt 3 / 2 →
  f (Real.pi / 9) = 0 →
  ∃(ω_min : ℝ), ω_min = 3 :=
by
  intros h0 h1 h2 h3 h4 h5
  use 3
  sorry -- Proof of the exact answer

end minimum_omega_is_3_l141_141628


namespace radical_axis_of_three_spheres_l141_141383

theorem radical_axis_of_three_spheres {C1 C2 C3 : ℝ × ℝ × ℝ} {R1 R2 R3 : ℝ}
  (h1 : (C1 ≠ C2 ∧ C1 ≠ C3 ∧ C2 ≠ C3) ∧ ¬(C1, C2, C3 are_collinear)) :
  ∃ L, ∀ P : ℝ × ℝ × ℝ, power P C1 R1 = power P C2 R2 ∧ power P C2 R2 = power P C3 R3 ↔ P ∈ L ∧ L is_perpendicular_to (plane_through [C1, C2, C3]) :=
sorry

end radical_axis_of_three_spheres_l141_141383


namespace max_angle_oma_l141_141109

theorem max_angle_oma (M : ℝ × ℝ) (h : M.1^2 + M.2^2 = 4) :
  let A := (sqrt 3, 0)
  let O := (0, 0) in
  ∠ O M A ≤ π / 3 :=
sorry

end max_angle_oma_l141_141109


namespace isosceles_triangle_condition_l141_141989

open Classical

-- Definitions from conditions
variable {α : Type*} [EuclideanGeometry α]

variable (A B C D E : Point α)

-- Given conditions
def isosceles_triangle (ABC : Triangle α) : Prop := 
∀ (A B C : Point α), ∃ (D E : Point α), 
  ABC.is_isosceles A B C ∧ 
  CD.is_angle_bisector C A B ∧ 
  E lies_on (AC.intersection_of_perpendicular_through D CD) ∧
  (ABC.AX).length = 2 * (ABC.AD).length

-- Conclusion to prove 
theorem isosceles_triangle_condition (ABC : Triangle α) :
  isosceles_triangle ABC → (EC.length = 2 * AD.length) :=
sorry

end isosceles_triangle_condition_l141_141989


namespace no_swap_possible_l141_141412

structure Point where
  x : ℚ
  y : ℚ

-- Define the initial positions of the needle's endpoints
def A : Point := ⟨0, 0⟩
def B : Point := ⟨1, 0⟩

-- Define the weight function
def weight (p : Point) : ℚ := p.x + 2 * Real.sqrt 2 * p.y

-- Define what it means for the endpoints to be swapped
def swapped (p1 p2 : Point) : Prop := (p1 = B ∧ p2 = A)

-- Define the rotation by 45 degrees around a point
def rotate45 (p center : Point) : Point :=
  let (x, y) := (p.x - center.x, p.y - center.y)
  in ⟨center.x + (x * Real.cos (π / 4) - y * Real.sin (π / 4)),
      center.y + (x * Real.sin (π / 4) + y * Real.cos (π / 4))⟩

theorem no_swap_possible : ¬ ∃ (n : ℕ) (seq : Fin n → Point) (is_rotation : ∀ i, seq (i+1) = rotate45 (seq i) A ∨ seq (i+1) = rotate45 (seq i) B), swapped (seq n 0) (seq n 1) :=
by
  sorry -- Proof is omitted

end no_swap_possible_l141_141412


namespace sequence_sum_2017_l141_141516

noncomputable def a_sequence (m n : ℕ) : ℕ → ℤ
| 0 => 0
| 1 => m
| 2 => n
| k + 2 => a_sequence (k + 1) - a_sequence k

def sum_sequence (f : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum f

theorem sequence_sum_2017 (m n : ℕ) :
  sum_sequence (a_sequence m n) 2017 = m := by
  sorry

end sequence_sum_2017_l141_141516


namespace students_not_excelling_in_either_sport_l141_141188

-- Given conditions
variable (n n_B n_S n_B_and_S : ℕ)
variable (h1 : n = 40)
variable (h2 : n_B = 12)
variable (h3 : n_S = 18)
variable (h4 : n_B_and_S = 6)

-- Theorem statement based on given conditions and required proof
theorem students_not_excelling_in_either_sport : 
  n - (n_B + n_S - n_B_and_S) = 16 :=
by
  -- Translate the conditions directly into Lean 4 language
  rw [←h1, ←h2, ←h3, ←h4]
  -- Now we should have the desired equation and add sorry since proof is not expected
  sorry

end students_not_excelling_in_either_sport_l141_141188


namespace negation_of_proposition_l141_141725

variable (a b : ℝ)

theorem negation_of_proposition :
  (¬ (a * b = 0 → a = 0 ∨ b = 0)) ↔ (a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) :=
by
  sorry

end negation_of_proposition_l141_141725


namespace stationery_difference_l141_141888

theorem stationery_difference :
  let georgia := 25
  let lorene := 3 * georgia
  lorene - georgia = 50 :=
by
  let georgia := 25
  let lorene := 3 * georgia
  show lorene - georgia = 50
  sorry

end stationery_difference_l141_141888


namespace opposite_of_neg_2022_l141_141298

theorem opposite_of_neg_2022 : -(-2022) = 2022 :=
by
  sorry

end opposite_of_neg_2022_l141_141298


namespace find_a_for_quadratic_trinomials_l141_141870

theorem find_a_for_quadratic_trinomials :
  ∃ (a : ℝ), (∀ x : ℝ, x^2 - 6*x + 4*a = 0 → (x - 3 + real.sqrt(9 - 4*4*a)); (x - 3 - real.sqrt(9 - 4*4*a))) ∧
              (∀ y : ℝ, y^2 + a*y + 6 = 0 → (y - -a/2 + real.sqrt(a^2 - 4*6)/2); (y - -a/2 - real.sqrt(a^2 - 4*6)/2)) ∧
              (((6^2 - 8*a) = (a^2 - 12))) := 
begin 
  use -12,
  sorry
end

end find_a_for_quadratic_trinomials_l141_141870


namespace minimum_omega_formula_l141_141654

noncomputable def minimumOmega (f : ℝ → ℝ) (x T ω φ : ℝ) : ℝ :=
  if (f = λ x, Real.cos (ω * x + φ)) 
     ∧ (ω > 0) 
     ∧ (0 < φ < π)
     ∧ (f T = √3 / 2)
     ∧ (f (π / 9) = 0) 
     ∧ (T = 2 * π / ω) 
  then 3 else 0

theorem minimum_omega_formula {f : ℝ → ℝ} {x T ω φ : ℝ} :
  (f = λ x, Real.cos (ω * x + φ)) 
  ∧ (ω > 0) 
  ∧ (0 < φ < π)
  ∧ (f T = √3 / 2)
  ∧ (f (π / 9) = 0)
  ∧ (T = 2 * π / ω) 
  → minimumOmega f x T ω φ = 3 :=
by
  sorry

end minimum_omega_formula_l141_141654


namespace prob_log_interval_l141_141998

noncomputable def geometric_probability (a b x : ℝ) := 
  (b - a) / x

theorem prob_log_interval :
  let a := 0 in
  let b := 4 in
  let f (x : ℝ) : ℝ := log 0.5 (x + 0.5) in
  let A := {x : ℝ | 0 ≤ x ∧ x ≤ 4} in
  let B := {x : ℝ | -1 ≤ f(x) ∧ f(x) ≤ 1} in
  geometric_probability (0:ℝ) (3/2:ℝ) (4:ℝ) = 3 / 8 :=
by
  sorry

end prob_log_interval_l141_141998


namespace problem1_problem2_problem3_l141_141103

noncomputable def f (x a : ℝ) : ℝ := log 2 (1 / x + a)

-- Problem 1
theorem problem1 (x : ℝ) (h₁ : 0 < x ∧ x < 1) :
  f x 1 > 1 :=
sorry

-- Problem 2
theorem problem2 (x : ℝ) (ha : a = 0 ∨ a = -1/4) : 
  f x a + log 2 (x^2) = 0 → ∃! x, f x a + log 2 (x^2) = 0 :=
sorry

-- Problem 3
theorem problem3 (a : ℝ) (ha : a > 0) :
  (∀ t ∈ set.Icc (1 / 2) 1, (∀ x ∈ set.Icc t (t + 1), f x a) - (∀ x ∈ set.Icc t (t + 1), f x a) ≤ 1) → 
  a ≥ 2 / 3 :=
sorry

end problem1_problem2_problem3_l141_141103


namespace area_triangle_equality_l141_141234

theorem area_triangle_equality
  (A B C D E F : ℝ)
  (is_rectangle : quadrilateral A B C D)
  (on_sides : E ∈ segment B C ∧ F ∈ segment C D)
  (is_equilateral : equilateral_triangle A E F) :
  area_triangle B C F = area_triangle A B E + area_triangle A F D :=
sorry

end area_triangle_equality_l141_141234


namespace M_eq_N_l141_141309

-- Define the sets M and N
def M : Set ℤ := {u | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l}
def N : Set ℤ := {u | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r}

-- Prove that M equals N
theorem M_eq_N : M = N := 
by {
  sorry
}

end M_eq_N_l141_141309


namespace find_AD_l141_141210

-- Define the triangle with required properties
variables {A B C D : Type*} [linear_ordered_field A]
variables {a b : A} (ABC_condition : ¬is_same_sides A B C)
variables (angle_bisector_cond : angle_bisector (angle A B C) D)
variables (condition1 : (AB - BD) = a)
variables (condition2 : (AC + CD) = b)

-- Theorem stating the desired outcome
theorem find_AD
  (ABC_condition : ¬is_same_sides A B C)
  (angle_bisector_cond : angle_bisector (angle A B C) D)
  (condition1 : (AB - BD) = a)
  (condition2 : (AC + CD) = b) :
  AD = real.sqrt (a * b) :=
sorry

end find_AD_l141_141210


namespace red_more_than_yellow_l141_141743

-- Define the total number of marbles
def total_marbles : ℕ := 19

-- Define the number of yellow marbles
def yellow_marbles : ℕ := 5

-- Calculate the number of remaining marbles
def remaining_marbles : ℕ := total_marbles - yellow_marbles

-- Define the ratio of blue to red marbles
def blue_ratio : ℕ := 3
def red_ratio : ℕ := 4

-- Calculate the sum of ratio parts
def sum_ratio : ℕ := blue_ratio + red_ratio

-- Calculate the number of shares per ratio part
def share_per_part : ℕ := remaining_marbles / sum_ratio

-- Calculate the number of red marbles
def red_marbles : ℕ := red_ratio * share_per_part

-- Theorem to prove: the difference between red marbles and yellow marbles is 3
theorem red_more_than_yellow : red_marbles - yellow_marbles = 3 :=
by
  sorry

end red_more_than_yellow_l141_141743


namespace derivative_at_pi_over_3_l141_141133

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sqrt 3 * Real.sin x

theorem derivative_at_pi_over_3 : 
  (deriv f) (Real.pi / 3) = 0 := 
by 
  sorry

end derivative_at_pi_over_3_l141_141133


namespace smallest_integer_five_consecutive_sum_2025_l141_141317

theorem smallest_integer_five_consecutive_sum_2025 :
  ∃ n : ℤ, 5 * n + 10 = 2025 ∧ n = 403 :=
by
  sorry

end smallest_integer_five_consecutive_sum_2025_l141_141317


namespace calc_miscellaneous_collective_expenses_l141_141001

def individual_needed_amount : ℕ := 450
def additional_needed_amount : ℕ := 475
def total_students : ℕ := 6
def first_day_amount : ℕ := 600
def second_day_amount : ℕ := 900
def third_day_amount : ℕ := 400
def days : ℕ := 4

def total_individual_goal : ℕ := individual_needed_amount + additional_needed_amount
def total_students_goal : ℕ := total_individual_goal * total_students
def total_first_3_days : ℕ := first_day_amount + second_day_amount + third_day_amount
def total_next_4_days : ℕ := (total_first_3_days / 2) * days
def total_raised : ℕ := total_first_3_days + total_next_4_days

def miscellaneous_collective_expenses : ℕ := total_raised - total_students_goal

theorem calc_miscellaneous_collective_expenses : miscellaneous_collective_expenses = 150 := by
  sorry

end calc_miscellaneous_collective_expenses_l141_141001


namespace minimum_omega_formula_l141_141653

noncomputable def minimumOmega (f : ℝ → ℝ) (x T ω φ : ℝ) : ℝ :=
  if (f = λ x, Real.cos (ω * x + φ)) 
     ∧ (ω > 0) 
     ∧ (0 < φ < π)
     ∧ (f T = √3 / 2)
     ∧ (f (π / 9) = 0) 
     ∧ (T = 2 * π / ω) 
  then 3 else 0

theorem minimum_omega_formula {f : ℝ → ℝ} {x T ω φ : ℝ} :
  (f = λ x, Real.cos (ω * x + φ)) 
  ∧ (ω > 0) 
  ∧ (0 < φ < π)
  ∧ (f T = √3 / 2)
  ∧ (f (π / 9) = 0)
  ∧ (T = 2 * π / ω) 
  → minimumOmega f x T ω φ = 3 :=
by
  sorry

end minimum_omega_formula_l141_141653


namespace boat_speed_in_still_water_l141_141735

theorem boat_speed_in_still_water (x : ℕ) 
  (h1 : x + 17 = 77) (h2 : x - 17 = 43) : x = 60 :=
by
  sorry

end boat_speed_in_still_water_l141_141735


namespace find_length_of_MN_l141_141575

theorem find_length_of_MN (A B C M N : ℝ × ℝ)
  (AB AC : ℝ) (M_midpoint : M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (N_midpoint : N = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))
  (length_AB : abs (B.1 - A.1) + abs (B.2 - A.2) = 15)
  (length_AC : abs (C.1 - A.1) + abs (C.2 - A.2) = 20) :
  abs (N.1 - M.1) + abs (N.2 - M.2) = 40 / 3 := sorry

end find_length_of_MN_l141_141575


namespace perimeter_of_square_is_64_l141_141020

noncomputable def side_length_of_square (s : ℝ) :=
  let rect_height := s
  let rect_width := s / 4
  let perimeter_of_rectangle := 2 * (rect_height + rect_width)
  perimeter_of_rectangle = 40

theorem perimeter_of_square_is_64 (s : ℝ) (h1 : side_length_of_square s) : 4 * s = 64 :=
by
  sorry

end perimeter_of_square_is_64_l141_141020


namespace find_certain_number_l141_141556

open Real

noncomputable def certain_number (x : ℝ) : Prop :=
  0.75 * x = 0.50 * 900

theorem find_certain_number : certain_number 600 :=
by
  dsimp [certain_number]
  -- We need to show that 0.75 * 600 = 0.50 * 900
  sorry

end find_certain_number_l141_141556


namespace planes_fit_in_hangar_l141_141828

theorem planes_fit_in_hangar (hangar_length plane_length : ℕ) (cond_hangar : hangar_length = 300) (cond_plane : plane_length = 40) : (hangar_length / plane_length) = 7 :=
by
  rw [cond_hangar, cond_plane]
  -- Prove that 300 / 40 = 7 when using integer division.
  sorry

end planes_fit_in_hangar_l141_141828


namespace number_of_true_propositions_l141_141931

open Classical

-- Define each proposition as a term or lemma in Lean
def prop1 : Prop := ∀ x : ℝ, x^2 + 1 > 0
def prop2 : Prop := ∀ x : ℕ, x^4 ≥ 1
def prop3 : Prop := ∃ x : ℤ, x^3 < 1
def prop4 : Prop := ∀ x : ℚ, x^2 ≠ 2

-- The main theorem statement that the number of true propositions is 3 given the conditions
theorem number_of_true_propositions : (prop1 ∧ prop3 ∧ prop4) ∧ ¬prop2 → 3 = 3 := by
  sorry

end number_of_true_propositions_l141_141931


namespace find_sequences_l141_141124

-- definitions for sequences a_n and b_n with given conditions
def geometric_seq (a₁ : ℕ) (q : ℕ) : ℕ → ℕ
| 0     := a₁
| (n+1) := geometric_seq a₁ q n * q

def arithmetic_seq (b₁ : ℕ) (d : ℕ) : ℕ → ℕ
| 0     := b₁
| (n+1) := arithmetic_seq b₁ d n + d

-- conditions
axiom condition_1 : geometric_seq 2 2 0 = 2
axiom condition_2 {b₁ : ℕ} {d : ℕ} : 
  ∀ n, let S_n := ∑ i in finset.range (n+1), (arithmetic_seq b₁ d i) in true
axiom condition_3 : geometric_seq 2 2 0 + geometric_seq 2 2 1 = 6
axiom condition_4 {b₁ : ℕ} {d : ℕ} : 2 * (arithmetic_seq b₁ d 0) + (geometric_seq 2 2 2) = (arithmetic_seq b₁ d 3)
axiom condition_5 {b₁ : ℕ} {d : ℕ} : 
  let S_3 := ∑ i in finset.range 3, (arithmetic_seq b₁ d i) in S_3 = 3 * geometric_seq 2 2 1

-- goal
theorem find_sequences :
  (geometric_seq 2 2 n = 2^n) ∧ 
  (arithmetic_seq 1 3 n = 3 * n - 2) ∧ 
  (let c_n n := arithmetic_seq 1 3 (geometric_seq 2 2 n) in 
     ∑ i in finset.range n, c_n i = 6 * 2^n - 2 * n - 6) :=
sorry

end find_sequences_l141_141124


namespace new_acute_angle_ACB_l141_141289

theorem new_acute_angle_ACB (θ : ℝ) (rot_deg : ℝ) (h1 : θ = 60) (h2 : rot_deg = 600) :
  (new_acute_angle (rotate_ray θ rot_deg)) = 0 := 
sorry

end new_acute_angle_ACB_l141_141289


namespace EF_parallel_plane_ABC1D1_distance_EF_to_plane_ABC1D1_l141_141897

variables (a : ℝ) (A B C D A1 B1 C1 D1 E F: ℝ^3)

-- Hypotheses
hypothesis cube : ∃ (a : ℝ), True
hypothesis h_E_on_A1B : E = A1 + 1 / 3 • (B - A1)
hypothesis h_F_on_B1D1 : F = B1 + 1 / 3 • (D1 - B1)

-- Prove EF is parallel to the plane ABC1D1
theorem EF_parallel_plane_ABC1D1
  (EF : ℝ^3) (AB : ℝ^3) (BC1 : ℝ^3) (n : ℝ^3) : 
  EF = F - E ∧
  AB = B - A ∧ 
  BC1 = C1 - B ∧ 
  n = AB × BC1 ∧
  n ≠ 0 →
  EF ∙ n = 0 := 
sorry

-- Find the distance from EF to the plane ABC1D1
theorem distance_EF_to_plane_ABC1D1
  (EF : ℝ^3) (n : ℝ^3) (BE : ℝ^3) :
  BE = E - B →
  ∃ d : ℝ, d = (| BE ∙ n | / ∥ n ∥) :=
sorry

end EF_parallel_plane_ABC1D1_distance_EF_to_plane_ABC1D1_l141_141897


namespace comparison_of_x_powers_l141_141893

theorem comparison_of_x_powers (x : ℝ) (h : 0 < x) (h'_1 : x < 1) :
    let a := x^2
    let b := 1 / x
    let c := Real.sqrt x
  in b > c ∧ c > a :=
by
  sorry

end comparison_of_x_powers_l141_141893


namespace coef_x4_in_binomial_expansion_l141_141104

noncomputable def a : ℝ :=
  4 * ∫ x in 0..(Real.pi / 2), Real.cos (2 * x + (Real.pi / 6))

theorem coef_x4_in_binomial_expansion (a_val : a = -2) :
  let b := (-2)^2 * Nat.choose 5 2 in
  b = 40 := by
  sorry

end coef_x4_in_binomial_expansion_l141_141104


namespace truck_sand_amount_l141_141026

theorem truck_sand_amount (initial_sand loss_sand final_sand : ℝ) (h1 : initial_sand = 4.1) (h2 : loss_sand = 2.4) :
  initial_sand - loss_sand = final_sand ↔ final_sand = 1.7 := 
by
  sorry

end truck_sand_amount_l141_141026


namespace baker_has_4_ovens_l141_141787

theorem baker_has_4_ovens (b : ℕ) (bk_hr : ℕ) (hr_wd : ℕ) (hr_we : ℕ) (wk_w : ℕ) (wks : ℕ) (total_lvs : ℕ) :
  (∃ (n : ℕ), b = 5 ∧ bk_hr = (hr_wd * 5 + hr_we * 2) ∧ wk_w = (bk_hr * 5) ∧ total_lvs = (n * (wk_w * 3)) ∧ n = 4) :=
begin
  -- Given conditions
  have h1 : b = 5,
  { exact rfl },
  have h2 : bk_hr = 29,
  { exact rfl },
  have h3 : wk_w = (bk_hr * 5),
  { exact rfl },
  have h4 : total_lvs = 1740,
  { exact rfl },
  
  -- Proof
  use 4,
  split, exact rfl,
  split, exact rfl,
  split, exact rfl,
  exact rfl,
end

end baker_has_4_ovens_l141_141787


namespace average_number_of_ducks_l141_141813

def average_ducks (A E K : ℕ) : ℕ :=
  (A + E + K) / 3

theorem average_number_of_ducks :
  ∀ (A E K : ℕ), A = 2 * E → E = K - 45 → A = 30 → average_ducks A E K = 35 :=
by 
  intros A E K h1 h2 h3
  sorry

end average_number_of_ducks_l141_141813


namespace isosceles_triangle_vertex_angle_l141_141168

theorem isosceles_triangle_vertex_angle (exterior_angle : ℝ) (h1 : exterior_angle = 40) : 
  ∃ vertex_angle : ℝ, vertex_angle = 140 :=
by
  sorry

end isosceles_triangle_vertex_angle_l141_141168


namespace each_niece_gets_13_l141_141757

-- Define the conditions
def total_sandwiches : ℕ := 143
def number_of_nieces : ℕ := 11

-- Prove that each niece can get 13 ice cream sandwiches
theorem each_niece_gets_13 : total_sandwiches / number_of_nieces = 13 :=
by
  -- Proof omitted
  sorry

end each_niece_gets_13_l141_141757


namespace problem_l141_141913

variable (a b x y : ℝ)

theorem problem (h₁ : a + b = 2) (h₂ : x + y = 2) (h₃ : ax + by = 5) :
  (a^2 + b^2) * x * y + a * b * (x^2 + y^2) = -5 := by
sorry

end problem_l141_141913


namespace spinner_probability_l141_141768

open Nat Set Real

def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(m ∣ n)

def countFavNumbers : ℕ :=
  list.length ([1, 2, 3, 4, 5, 6, 7, 8].filter (λ n, isPrime n ∨ (4 ∣ n)))

def totalNumbers : ℕ := 8

def probability : ℚ := countFavNumbers / totalNumbers

theorem spinner_probability : probability = 3 / 4 := by
  sorry

end spinner_probability_l141_141768


namespace find_AB_l141_141208

-- Given conditions
variables {A B C : Type*} [metric_space A] [metric_space B] [metric_space C]
variables (BC AC AB : ℝ)
variables (angle_A : real.angle)
variables (tan_C cos_B : ℝ)

-- Assume the given values and properties
axiom angle_A_right : angle_A = 90
axiom BC_val : BC = 20
axiom tan_cos_relation : tan_C = 3 * cos_B
axiom tan_C_def : tan_C = AB / AC
axiom cos_B_def : cos_B = AB / BC

-- Prove the required length of AB
theorem find_AB : AB = (40 * real.sqrt 2) / 3 :=
by
  sorry

end find_AB_l141_141208


namespace marble_color_197_l141_141802

-- Define the types and properties of the marbles
inductive Color where
  | red | blue | green

-- Define a function to find the color of the nth marble in the cycle pattern
def colorOfMarble (n : Nat) : Color :=
  let cycleLength := 15
  let positionInCycle := n % cycleLength
  if positionInCycle < 6 then Color.red  -- first 6 marbles are red
  else if positionInCycle < 11 then Color.blue  -- next 5 marbles are blue
  else Color.green  -- last 4 marbles are green

-- The theorem asserting the color of the 197th marble
theorem marble_color_197 : colorOfMarble 197 = Color.red :=
sorry

end marble_color_197_l141_141802


namespace same_cost_for_same_sheets_l141_141428

def John's_Photo_World_cost (x : ℕ) : ℝ := 2.75 * x + 125
def Sam's_Picture_Emporium_cost (x : ℕ) : ℝ := 1.50 * x + 140

theorem same_cost_for_same_sheets :
  ∃ (x : ℕ), John's_Photo_World_cost x = Sam's_Picture_Emporium_cost x ∧ x = 12 :=
by
  sorry

end same_cost_for_same_sheets_l141_141428


namespace rotated_angle_remains_60_degrees_l141_141294

theorem rotated_angle_remains_60_degrees (A B C : Type) [metric_space A] 
  (angle_ACB original rotated : ℝ)
  (h0 : angle_ACB = 60)
  (h1 : rotated = 600):
  original = 60 → (original - 2 * 360) % 360 = original :=
by
  sorry

end rotated_angle_remains_60_degrees_l141_141294


namespace range_of_a_l141_141977

open Real

def f (a x : ℝ) := a * x + 2 * a + 1

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Icc (-1) (1), f a x > 0) ∧ (∃ x ∈ Icc (-1) (1), f a x < 0) ↔ (-1 < a ∧ a < -1 / 3) :=
by
  sorry

end range_of_a_l141_141977


namespace length_of_second_train_l141_141756

noncomputable def length_of_first_train : ℝ := 156.62
noncomputable def speed_of_first_train_kmph : ℝ := 30
noncomputable def speed_of_second_train_kmph : ℝ := 36
noncomputable def time_to_cross_seconds : ℝ := 13.996334838667455

theorem length_of_second_train :
  let relative_speed_mps := (speed_of_first_train_kmph + speed_of_second_train_kmph) * 1000 / 3600
  let total_distance := length_of_first_train + 100.05
  let expected_distance := relative_speed_mps * time_to_cross_seconds
  total_distance = expected_distance
  → 100.05 = expected_distance - length_of_first_train :=
by 
  intro relative_speed_mps total_distance expected_distance h,
  sorry

end length_of_second_train_l141_141756


namespace major_axis_length_l141_141718

theorem major_axis_length :
  ∀ {x y : ℝ}, (x^2 / 49 + y^2 / 81 = 1) → 
  (∃ a : ℝ, a = 9 ∧ 2 * a = 18) :=
by 
  intros x y h
  use 9
  split
  rfl
  norm_num
  rfl

end major_axis_length_l141_141718


namespace club_sum_eq_67_l141_141877

def club_f (x : ℝ) : ℝ := (x^3 + x^4) / 2

theorem club_sum_eq_67 : club_f 1 + club_f 2 + club_f 3 = 67 := 
by 
  sorry

end club_sum_eq_67_l141_141877


namespace gcd_1043_2295_eq_1_l141_141873

theorem gcd_1043_2295_eq_1 : Nat.gcd 1043 2295 = 1 := by
  sorry

end gcd_1043_2295_eq_1_l141_141873


namespace solution1_solution2_l141_141054

-- Define the first expression
def expr1 := (1 : ℝ) * (0.25 : ℝ)^(1/2) - (-2 * (3 / 7)^0)^2 * ((-2) ^ 3)^(4 / 3) + (Real.sqrt 2 - 1)^(-1) - 2^(1 / 2)

-- Define the second expression
def expr2 := (Real.log 5 5)^2 + Real.log 2 2 * Real.log 5 10

theorem solution1 : expr1 = -125 / 2 :=
by
  sorry

theorem solution2 : expr2 = 3 :=
by
  sorry

end solution1_solution2_l141_141054


namespace log_product_eq_four_l141_141969

theorem log_product_eq_four (k x : ℝ) (hk : k > 0) (hk1 : k ≠ 1) (hx : x > 0) (h : log k x * log 3 k = 4) : x = 81 :=
by
  sorry

end log_product_eq_four_l141_141969


namespace trig_identity_proof_l141_141841

noncomputable def cos_30 := Real.cos (Real.pi / 6)
noncomputable def sin_60 := Real.sin (Real.pi / 3)
noncomputable def sin_30 := Real.sin (Real.pi / 6)
noncomputable def cos_60 := Real.cos (Real.pi / 3)

theorem trig_identity_proof :
  (1 - (1 / cos_30)) * (1 + (2 / sin_60)) * (1 - (1 / sin_30)) * (1 + (2 / cos_60)) = (25 - 10 * Real.sqrt 3) / 3 := by
  sorry

end trig_identity_proof_l141_141841


namespace negative_expressions_l141_141287

-- Define the approximated values for P, Q, R, S, and T
def P : ℝ := 3.5
def Q : ℝ := 1.1
def R : ℝ := -0.1
def S : ℝ := 0.9
def T : ℝ := 1.5

-- State the theorem to be proved
theorem negative_expressions : 
  (R / (P * Q) < 0) ∧ ((S + T) / R < 0) :=
by
  sorry

end negative_expressions_l141_141287


namespace students_dislike_food_options_l141_141185

variables (students : nat)
variables (like_french_fries like_burgers like_pizza like_tacos : nat)
variables (like_french_burgers like_french_pizza like_french_tacos : nat)
variables (like_burgers_pizza like_burgers_tacos like_pizza_tacos : nat)
variables (like_french_burgers_pizza like_french_burgers_tacos : nat)
variables (like_french_pizza_tacos like_burgers_pizza_tacos : nat)
variables (like_all_four : nat)

-- Exact values from the problem setup
variables (h_students : students = 35)
variables (h_like_french_fries : like_french_fries = 20)
variables (h_like_burgers : like_burgers = 15)
variables (h_like_pizza : like_pizza = 18)
variables (h_like_tacos : like_tacos = 12)
variables (h_like_french_burgers : like_french_burgers = 10)
variables (h_like_french_pizza : like_french_pizza = 8)
variables (h_like_french_tacos : like_french_tacos = 6)
variables (h_like_burgers_pizza : like_burgers_pizza = 7)
variables (h_like_burgers_tacos : like_burgers_tacos = 5)
variables (h_like_pizza_tacos : like_pizza_tacos = 9)
variables (h_like_french_burgers_pizza : like_french_burgers_pizza = 4)
variables (h_like_french_burgers_tacos : like_french_burgers_tacos = 3)
variables (h_like_french_pizza_tacos : like_french_pizza_tacos = 2)
variables (h_like_burgers_pizza_tacos : like_burgers_pizza_tacos = 1)
variables (h_like_all_four : like_all_four = 1)

theorem students_dislike_food_options : 
  students - 
  (like_french_fries + like_burgers + like_pizza + like_tacos
   - like_french_burgers - like_french_pizza - like_french_tacos
   - like_burgers_pizza - like_burgers_tacos - like_pizza_tacos
   + like_french_burgers_pizza + like_french_burgers_tacos
   + like_french_pizza_tacos + like_burgers_pizza_tacos
   - like_all_four) = 6 := by {
  -- assume the given hypothesis
  rw [h_students, h_like_french_fries, h_like_burgers, h_like_pizza, h_like_tacos,
      h_like_french_burgers, h_like_french_pizza, h_like_french_tacos,
      h_like_burgers_pizza, h_like_burgers_tacos, h_like_pizza_tacos,
      h_like_french_burgers_pizza, h_like_french_burgers_tacos,
      h_like_french_pizza_tacos, h_like_burgers_pizza_tacos, h_like_all_four],
  -- compute the numbers
  norm_num,
}

end students_dislike_food_options_l141_141185


namespace prob_both_primes_l141_141339

-- Define the set of integers from 1 through 30
def int_set : Set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the set of prime numbers between 1 and 30
def primes_between_1_and_30 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Calculate the number of ways to choose two distinct elements from a set
noncomputable def combination (n k : ℕ) : ℕ := if k > n then 0 else n.choose k

-- Define the probabilities
noncomputable def prob_primes : ℚ :=
  (combination 10 2) / (combination 30 2)

-- State the theorem to prove
theorem prob_both_primes : prob_primes = 10 / 87 := by
  sorry

end prob_both_primes_l141_141339


namespace correct_option_l141_141775

/-- Define sets of line segments --/
def segments (A B C D : list ℕ) :=
  A = [1, 3, 4] ∧ B = [2, 2, 7] ∧ C = [4, 5, 7] ∧ D = [3, 3, 6]

/-- Define the triangle inequality theorem --/
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Prove that only option C satisfies the triangle inequality --/
theorem correct_option (A B C D : list ℕ) (h : segments A B C D) :
  triangle_inequality C[0] C[1] C[2] =
  true :=
begin
  sorry
end

end correct_option_l141_141775


namespace determine_n_l141_141791

theorem determine_n (n : ℕ) (h1 : 20 * n / 99 = 4.04) : n = 20 :=
by
  sorry

end determine_n_l141_141791


namespace minimum_omega_is_3_l141_141624

variable (f : ℝ → ℝ)
variable (ω φ T : ℝ)
variable (x : ℝ)

noncomputable def minimum_omega (ω : ℝ) : ℝ := 3 -- Temporary noncomputable definition to hold the value for min omega

theorem minimum_omega_is_3 :
  (∀ x, f x = Real.cos (ω * x + φ)) →
  ω > 0 →
  0 < φ ∧ φ < Real.pi →
  T = (2 * Real.pi) / ω →
  f T = Real.sqrt 3 / 2 →
  f (Real.pi / 9) = 0 →
  ∃(ω_min : ℝ), ω_min = 3 :=
by
  intros h0 h1 h2 h3 h4 h5
  use 3
  sorry -- Proof of the exact answer

end minimum_omega_is_3_l141_141624


namespace exists_non_convex_polyhedron_with_square_faces_l141_141891

-- Define a structure for polyhedra
structure Polyhedron :=
  (faces : list square)
  (is_non_convex : Prop)

def square : Type := 
{side_length : ℝ}

noncomputable def rhombicDodecahedronOfSecondKind : Polyhedron := 
{
  faces := list.repeat ({ side_length := 1 } : square) 12,
  is_non_convex := sorry
}

theorem exists_non_convex_polyhedron_with_square_faces :
  ∃ P : Polyhedron, P.is_non_convex ∧ (∀ f ∈ P.faces, f = { side_length := 1 } : square) :=
  ⟨rhombicDodecahedronOfSecondKind, sorry⟩ 

end exists_non_convex_polyhedron_with_square_faces_l141_141891


namespace bananas_in_blue_basket_l141_141327

def blue_basket_has_8_fruits : Prop := (blue_basket_fruits = 8)
def blue_basket_has_4_apples : Prop := (blue_basket_apples = 4)
def red_basket_holds_half_fruits : Prop := (red_basket_fruits = blue_basket_fruits / 2)

def number_of_bananas_in_blue_basket : ℕ := blue_basket_fruits - blue_basket_apples

theorem bananas_in_blue_basket :
  blue_basket_has_8_fruits ∧ blue_basket_has_4_apples → number_of_bananas_in_blue_basket = 4 := 
by
  intros,
  obtain ⟨blue_basket_fruits_eq, blue_basket_apples_eq⟩ := ‹blue_basket_has_8_fruits ∧ blue_basket_has_4_apples›,
  have h1 : blue_basket_fruits = 8 := blue_basket_fruits_eq,
  have h2 : blue_basket_apples = 4 := blue_basket_apples_eq,
  show number_of_bananas_in_blue_basket =  4,
  sorry

end bananas_in_blue_basket_l141_141327


namespace second_valid_number_is_176_l141_141194

def random_number_table := [84, 42, 17, 53, 31, 57, 24, 55, 0, 88, 77, 4, 74, 17, 67, 21, 76, 33, 50, 25, 83, 92, 12, 6, 76]

def is_valid_student_number (n : Nat) : Prop :=
  1 ≤ n ∧ n ≤ 200

noncomputable def second_random_student_number (table : List Nat) : Nat := 
  let valid_numbers := table.filter is_valid_student_number
  valid_numbers.tail.head -- Get the second valid number from the list

theorem second_valid_number_is_176 :
  second_random_student_number random_number_table = 176 :=
  sorry

end second_valid_number_is_176_l141_141194


namespace minimum_omega_is_3_l141_141631

variable (f : ℝ → ℝ)
variable (ω φ T : ℝ)
variable (x : ℝ)

noncomputable def minimum_omega (ω : ℝ) : ℝ := 3 -- Temporary noncomputable definition to hold the value for min omega

theorem minimum_omega_is_3 :
  (∀ x, f x = Real.cos (ω * x + φ)) →
  ω > 0 →
  0 < φ ∧ φ < Real.pi →
  T = (2 * Real.pi) / ω →
  f T = Real.sqrt 3 / 2 →
  f (Real.pi / 9) = 0 →
  ∃(ω_min : ℝ), ω_min = 3 :=
by
  intros h0 h1 h2 h3 h4 h5
  use 3
  sorry -- Proof of the exact answer

end minimum_omega_is_3_l141_141631


namespace inverse_of_2x_plus_3_l141_141717

noncomputable def inverse_function (y : ℝ) : ℝ := log y

theorem inverse_of_2x_plus_3 (x : ℝ) (h : x ≥ 0) : 
  Function.invFun (λ x : ℝ, 2^x + 3) ≤ (λ y : ℝ, log (y - 3)) :=
by
  sorry

end inverse_of_2x_plus_3_l141_141717


namespace wide_flags_made_l141_141846

theorem wide_flags_made
  (initial_fabric : ℕ) (square_flag_side : ℕ) (wide_flag_width : ℕ) (wide_flag_height : ℕ)
  (tall_flag_width : ℕ) (tall_flag_height : ℕ) (made_square_flags : ℕ) (made_tall_flags : ℕ)
  (remaining_fabric : ℕ) (used_fabric_for_small_flags : ℕ) (used_fabric_for_tall_flags : ℕ)
  (used_fabric_for_wide_flags : ℕ) (wide_flag_area : ℕ) :
    initial_fabric = 1000 →
    square_flag_side = 4 →
    wide_flag_width = 5 →
    wide_flag_height = 3 →
    tall_flag_width = 3 →
    tall_flag_height = 5 →
    made_square_flags = 16 →
    made_tall_flags = 10 →
    remaining_fabric = 294 →
    used_fabric_for_small_flags = 256 →
    used_fabric_for_tall_flags = 150 →
    used_fabric_for_wide_flags = initial_fabric - remaining_fabric - (used_fabric_for_small_flags + used_fabric_for_tall_flags) →
    wide_flag_area = wide_flag_width * wide_flag_height →
    (used_fabric_for_wide_flags / wide_flag_area) = 20 :=
by
  intros; 
  sorry

end wide_flags_made_l141_141846


namespace solution_set_inequality_l141_141310

theorem solution_set_inequality (x : ℝ) : 4 * x < 3 * x + 2 → x < 2 :=
by
  intro h
  -- Add actual proof here, but for now; we use sorry
  sorry

end solution_set_inequality_l141_141310


namespace find_a_b_l141_141907

theorem find_a_b (a b : ℝ) :
  let P := (1, 4) in
  let C := λ x y, x^2 + y^2 + 2 * a * x - 4 * y + b = 0 in
  (C 1 4) ∧
  (∃ P' : ℝ × ℝ, P' = (1 - 3, 2 - 1) ∧ C (-1) 2) →
  a = -1 ∧ b = 1 :=
by
  sorry

end find_a_b_l141_141907


namespace projection_of_vector_3_4_on_1_2_l141_141729

def vector_projection (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / magnitude_b

theorem projection_of_vector_3_4_on_1_2 :
  vector_projection (3, 4) (1, 2) = 11 * real.sqrt 5 / 5 :=
by
  sorry

end projection_of_vector_3_4_on_1_2_l141_141729


namespace average_price_correct_l141_141259

-- Define the conditions
def books_shop1 : ℕ := 65
def price_shop1 : ℕ := 1480
def books_shop2 : ℕ := 55
def price_shop2 : ℕ := 920

-- Define the total books and total price based on conditions
def total_books : ℕ := books_shop1 + books_shop2
def total_price : ℕ := price_shop1 + price_shop2

-- Define the average price based on total books and total price
def average_price : ℕ := total_price / total_books

-- Theorem stating the average price per book Sandy paid
theorem average_price_correct : average_price = 20 :=
  by
  sorry

end average_price_correct_l141_141259


namespace perfect_squares_and_cubes_below_1000_l141_141748

/-- Prove there are exactly three natural numbers (excluding zero) that are both perfect squares and perfect cubes within 1000. -/
theorem perfect_squares_and_cubes_below_1000 :
  {n : ℕ | n > 0 ∧ ∃ k : ℕ, n = k^6 ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end perfect_squares_and_cubes_below_1000_l141_141748


namespace hyperbola_equation_proof_l141_141533

def equation_of_ellipse (x y : ℝ) : Prop := (x^2) / 8 + (y^2) / 5 = 1

def hyperbola_foci (x : ℝ) : Prop := x = -2 * real.sqrt 2 ∨ x = 2 * real.sqrt 2

def hyperbola_vertices (x : ℝ) : Prop := x = -real.sqrt 3 ∨ x = real.sqrt 3

theorem hyperbola_equation_proof :
  (∀ x y, equation_of_ellipse x y) →
  (∀ x, hyperbola_foci x) →
  (∀ x, hyperbola_vertices x) →
  ∃ a b : ℝ, a = real.sqrt 3 ∧ b^2 = (2 * real.sqrt 2)^2 - (real.sqrt 3)^2 ∧
    (∀ x y, (x^2) / a^2 - (y^2) / b^2 = 1) :=
by
  intros _ _ _
  use (real.sqrt 3)
  use real.sqrt 5
  split
  { refl }
  split
  { simp [pow_two, mul_add, mul_comm, mul_assoc], ring }
  { intros x y
    sorry
  }

end hyperbola_equation_proof_l141_141533


namespace apples_to_pears_l141_141572

theorem apples_to_pears :
  (∀ (apples oranges pears : ℕ),
  12 * apples = 6 * oranges →
  3 * oranges = 5 * pears →
  24 * apples = 20 * pears) :=
by
  intros apples oranges pears h₁ h₂
  sorry

end apples_to_pears_l141_141572


namespace modulus_of_complex_l141_141122

theorem modulus_of_complex (a b : ℝ) (h : b^2 + (4 : ℝ) * b + 4 + (0 : ℝ) * a = 0) (h' : b + (a : ℂ) * I = 0) : |complex.mk a b| = 2 * real.sqrt 2 :=
by 
  sorry

end modulus_of_complex_l141_141122


namespace triangle_equilateral_l141_141966

noncomputable def is_equilateral_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = b ∧ b = c ∧ A = B ∧ B = C

theorem triangle_equilateral 
  (a b c A B C : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * a * b * c) 
  (h2 : Real.sin A = 2 * Real.sin B * Real.cos C) : 
  is_equilateral_triangle a b c A B C :=
sorry

end triangle_equilateral_l141_141966


namespace melanie_total_weight_is_correct_l141_141674

-- Define quantities in their respective units
def brie_ounces        : Real := 8
def bread_pounds       : Real := 1
def tomatoes_pounds    : Real := 1
def zucchini_pounds    : Real := 2
def chicken_pounds     : Real := 1.5
def raspberries_ounces : Real := 8
def blueberries_ounces : Real := 8
def asparagus_grams    : Real := 500
def oranges_grams      : Real := 1000
def olive_oil_milliliters : Real := 750

-- Define conversion factors
def gram_to_pound      : Real := 0.00220462
def ounce_to_pound     : Real := 0.0625
def liter_to_pound     : Real := 2.20462

-- Convert items to pounds
def brie_pounds        : Real := brie_ounces * ounce_to_pound
def raspberries_pounds : Real := raspberries_ounces * ounce_to_pound
def blueberries_pounds : Real := blueberries_ounces * ounce_to_pound
def asparagus_pounds   : Real := asparagus_grams * gram_to_pound
def oranges_pounds     : Real := oranges_grams * gram_to_pound
def olive_oil_pounds   : Real := (olive_oil_milliliters / 1000) * liter_to_pound

-- Compute the total weight in pounds
noncomputable def total_weight_pounds : Real := 
  brie_pounds + bread_pounds + tomatoes_pounds + zucchini_pounds + chicken_pounds + 
  raspberries_pounds + blueberries_pounds + asparagus_pounds + oranges_pounds + olive_oil_pounds

-- State the theorem
theorem melanie_total_weight_is_correct : total_weight_pounds = 11.960895 :=
by
  sorry

end melanie_total_weight_is_correct_l141_141674


namespace average_problem_l141_141270

noncomputable def avg2 (a b : ℚ) := (a + b) / 2
noncomputable def avg3 (a b c : ℚ) := (a + b + c) / 3

theorem average_problem :
  avg3 (avg3 2 2 1) (avg2 1 2) 1 = 25 / 18 :=
by
  sorry

end average_problem_l141_141270


namespace sum_of_valid_integers_l141_141172

theorem sum_of_valid_integers :
  let valid_a (a : ℤ) := a ≤ 2 ∧ a < 2 ∧ a ≠ -2 in
  (Finset.sum (Finset.filter valid_a (Finset.Icc (-4) 2))
  (fun x => x)) = -1 :=
by
  sorry

end sum_of_valid_integers_l141_141172


namespace tate_total_years_l141_141276

-- Define the conditions
def high_school_years : Nat := 3
def gap_years : Nat := 2
def bachelor_years : Nat := 2 * high_school_years
def certification_years : Nat := 1
def work_experience_years : Nat := 1
def master_years : Nat := bachelor_years / 2
def phd_years : Nat := 3 * (high_school_years + bachelor_years + master_years)

-- Define the total years Tate spent
def total_years : Nat :=
  high_school_years + gap_years +
  bachelor_years + certification_years +
  work_experience_years + master_years + phd_years

-- State the theorem
theorem tate_total_years : total_years = 52 := by
  sorry

end tate_total_years_l141_141276


namespace sufficient_not_necessary_condition_l141_141528

theorem sufficient_not_necessary_condition (x : ℝ) (p : |x| < 1) (q : x^2 - 2x - 3 < 0) : 
  (∀ x, p x → q x) ∧ ¬ (∀ x, q x → p x) :=
sorry

end sufficient_not_necessary_condition_l141_141528


namespace proof_problem_l141_141895

-- Define the problem conditions
def alpha (a : ℝ) := a ∈ set.Ioc (π/2) π
def cond (α : ℝ) : Prop := 3 * Real.cos (2 * α) - Real.sin α = 2

-- The theorem we want to prove
theorem proof_problem (α : ℝ) (h1 : alpha α) (h2 : cond α) : 
  Real.tan (π - α) = Real.sqrt 2 / 4 :=
sorry

end proof_problem_l141_141895


namespace base6_subtraction_l141_141477

-- Define the base-6 representation of numbers
def from_base6 (lst : List ℕ) : ℕ :=
  lst.reverse.enum_from 0 |>.foldl (λ acc (p : ℕ × ℕ) => acc + p.snd * 6 ^ p.fst) 0

-- Convert the given base-6 numbers to base-10 for computation
noncomputable def four_three_one_6 : ℕ := from_base6 [4, 3, 1]
noncomputable def two_five_four_6 : ℕ := from_base6 [2, 5, 4]

-- Expected result in base-6
def one_three_three_6 : ℕ := from_base6 [1, 3, 3]

-- The main theorem statement
theorem base6_subtraction : (four_three_one_6 - two_five_four_6 = one_three_three_6) :=
by
  sorry

end base6_subtraction_l141_141477


namespace trig_identity_l141_141158

theorem trig_identity (α : ℝ) 
  (h : (cos (2 * α)) / (sin (α - (π / 4))) = - (sqrt 2 / 2)) :
  cos α + sin α = 1 / 2 :=
sorry

end trig_identity_l141_141158


namespace min_omega_is_three_l141_141640

open Real

noncomputable def min_omega {ω : ℝ} {ϕ : ℝ} (f : ℝ → ℝ) : ℝ :=
  if f = (λ x, cos (ω * x + ϕ)) then
    if 0 < ω ∧ 0 < ϕ ∧ ϕ < π ∧
       (∃ (T : ℝ), T = 2 * π / ω ∧ f T = sqrt 3 / 2) ∧
       f (π / 9) = 0 then
      ω
    else 0
  else 0

theorem min_omega_is_three :
  let f : ℝ → ℝ := λ x, cos (3 * x + π / 6) in
  min_omega f = 3 := by
  sorry

end min_omega_is_three_l141_141640


namespace parallel_lines_touching_circles_l141_141333

theorem parallel_lines_touching_circles
  {O₁ O₂ K A B C D : Type}
  (circle_1 : circle O₁)
  (circle_2 : circle O₂)
  (r₁ : ℝ)
  (r₂ : ℝ)
  (tangency : K ∈ circle_1 ∩ circle_2)
  (line_K_A_B : is_line_through K A B)
  (line_K_C_D : is_line_through K C D)
  (A_on_circle_1 : A ∈ circle_1)
  (B_on_circle_1 : B ∈ circle_1)
  (C_on_circle_2 : C ∈ circle_2)
  (D_on_circle_2 : D ∈ circle_2) :
  AB ∥ CD := 
sorry

end parallel_lines_touching_circles_l141_141333


namespace exists_linear_function_intersecting_negative_axes_l141_141252

theorem exists_linear_function_intersecting_negative_axes :
  ∃ (k b : ℝ), k < 0 ∧ b < 0 ∧ (∃ x, k * x + b = 0 ∧ x < 0) ∧ (k * 0 + b < 0) :=
by
  sorry

end exists_linear_function_intersecting_negative_axes_l141_141252


namespace intersection_point_t_bound_l141_141119

theorem intersection_point_t_bound (α β t : ℝ) (h1 : sin β = (sin α)^3.sqrt3 + t^3.sqrt3)
  (h2 : sin β = 3 * t * (sin α)^2 + (3 * t^2 + 1) * sin α + t) : abs t ≤ 1 := 
sorry

end intersection_point_t_bound_l141_141119


namespace interchangeable_propositions_13_l141_141558

def interchangeable_proposition (p: Prop) : Prop :=
  ∀ (line plane : Type) (P: plane → Prop) (L: line → Prop), 
  p ↔ (p ∧ ∀ x : plane, L x ∨ ∀ x : line, P x)

def prop1 : Prop := 
  ∀ (line plane : Type) (p: line → plane → Prop), 
  (∀ l1 l2 : line, ∀ p1 : plane, p l1 p1 ∧ p l2 p1 → l1 = l2)

def prop2 : Prop := 
  ∀ (plane1 plane2 plane3 : Type) (p: plane → plane → Prop), 
  (∀ p1 p2 : plane1, p1 = p2 ∧ p p1 p2 → p1 = p2)

def prop3 : Prop := 
  ∀ (line1 line2 : Type) (p: line → line → Prop), 
  (∀ l1 l2 : line1, ∀ l3 : line2, p l1 l2 ∧ p l1 l3 → l2 = l3)

def prop4 : Prop := 
  ∀ (line plane : Type) (p: line → plane → Prop), 
  (∀ l1 : line, ∀ p1 p2 : plane, (p l1 p1) ∧ (p l1 p2) → p1 = p2)

theorem interchangeable_propositions_13 : 
  interchangeable_proposition prop1 ∧ interchangeable_proposition prop3 ∧ 
  ¬ interchangeable_proposition prop2 ∧ ¬ interchangeable_proposition prop4 := 
sorry

end interchangeable_propositions_13_l141_141558


namespace sqrt_inequality_l141_141095

theorem sqrt_inequality {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  sqrt (4 * a + 1) + sqrt (4 * b + 1) + sqrt (4 * c + 1) < 5 :=
sorry

end sqrt_inequality_l141_141095


namespace not_exists_Q_l141_141542

def P (x : ℚ) : ℚ := 
  ∑ i in finset.range 101, (i + 1) * x ^ i

def is_permutation (P Q : ℚ → ℚ) :=
  ∃ σ : fin (101) → fin (101), bijective σ ∧ 
    (∀ i : fin 101, (P.coeff i) = (Q.coeff (σ i)))

theorem not_exists_Q (Q : ℚ → ℚ) :
  ¬ is_permutation P Q ∨ (∃ k : ℕ, k ≥ 2 ∧ ¬ 2020 ∣ (P k - Q k)) :=
sorry

end not_exists_Q_l141_141542


namespace average_operating_time_l141_141830

-- Definition of problem conditions
def cond1 : Nat := 5 -- originally had 5 air conditioners
def cond2 : Nat := 6 -- after installing 1 more
def total_hours : Nat := 24 * 5 -- total operating hours allowable in 24 hours

-- Formalize the average operating time calculation
theorem average_operating_time : (total_hours / cond2) = 20 := by
  sorry

end average_operating_time_l141_141830


namespace chord_length_is_two_l141_141804

noncomputable def chord_length_on_circle 
  (line_passes_origin : Prop)
  (line_angle : ℝ)
  (circle_center : ℝ × ℝ)
  (circle_radius : ℝ)
  (line_eq : ℝ → ℝ → Prop) : ℝ :=
if line_passes_origin
   ∧ line_angle = 30
   ∧ circle_center = (0, 2)
   ∧ circle_radius = 2
   ∧ line_eq = λ x y, y = (Real.sqrt 3 / 3) * x then
   2
else
   0

theorem chord_length_is_two : 
  chord_length_on_circle 
    (line_passes_origin := true) 
    (line_angle := 30) 
    (circle_center := (0, 2)) 
    (circle_radius := 2)
    (line_eq := λ x y, y = (Real.sqrt 3 / 3) * x) = 2 :=
by
  sorry

end chord_length_is_two_l141_141804


namespace hyperbola_coeff_sum_l141_141461

theorem hyperbola_coeff_sum :
  (∃ A B C D E F : ℤ,
    ∀ t : ℝ, 
      let x := (3 * (Real.cos t + 2)) / (3 + Real.sin t),
          y := (4 * (Real.sin t - 4)) / (3 + Real.sin t)
      in
      A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0
      ∧ Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.gcd (Int.natAbs D) (Int.gcd (Int.natAbs E) (Int.natAbs F))))) = 1)
  → |9| + |24| + |16| + |(-12)| + |16| + |4| = 106 :=
sorry

end hyperbola_coeff_sum_l141_141461


namespace find_distance_OM_l141_141899

-- Definitions
def ellipse : Set (ℝ × ℝ) := {p : ℝ × ℝ | let (x, y) := p in (x^2 / 25 + y^2 / 16 = 1)}

def focus_F := (3 : ℝ, 0 : ℝ)

def is_midpoint (P F M : ℝ × ℝ) : Prop :=
  M = ((P.1 + F.1) / 2, (P.2 + F.2) / 2)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem
theorem find_distance_OM (P M : ℝ × ℝ)
  (hP : P ∈ ellipse)
  (hPF : distance P focus_F = 2)
  (hM : is_midpoint P focus_F M) : distance (0, 0) M = 4 := by
  sorry

end find_distance_OM_l141_141899


namespace person_savings_l141_141596

theorem person_savings
    (income expenditure : ℕ)
    (salary side_business interest total_income: ℕ)
    (salary_ratio side_business_ratio interest_ratio total_ratio: ℕ)
    (tax_salary tax_side_business: ℕ)
    (net_salary net_side_business net_income expenditure_ratio savings: ℕ) 
    (H1 : income = 10000)
    (H2 : expenditure_ratio = 8/10)
    (H3 : salary_ratio / side_business_ratio / interest_ratio = 5/3/2)
    (H4 : salary + side_business + interest = total_income)
    (H5 : salary_ratio + side_business_ratio + interest_ratio = total_ratio)
    (H6 : total_income / total_ratio = 1000)
    (H7 : salary = 5000)
    (H8 : side_business = 3000)
    (H9 : interest = 2000)
    (H10 : tax_salary = 15/100 * salary)
    (H11 : tax_side_business = 10/100 * side_business)
    (H12 : net_salary = salary - tax_salary)
    (H13 : net_side_business = side_business - tax_side_business)
    (H14 : net_income = net_salary + net_side_business + interest)
    (H15 : expenditure = total_income * expenditure_ratio)
    (H16 : savings = net_income - expenditure) :
    savings = 950 :=
begin
  sorry
end

end person_savings_l141_141596


namespace number_of_ordered_triples_l141_141490

theorem number_of_ordered_triples (a b c : ℤ) : 
  ∃ (n : ℕ), -31 <= a ∧ a <= 31 ∧ -31 <= b ∧ b <= 31 ∧ -31 <= c ∧ c <= 31 ∧ 
  (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a + b + c > 0) ∧ n = 117690 :=
by sorry

end number_of_ordered_triples_l141_141490


namespace triangle_angle_bisector_ratio_l141_141591

-- Lean statement for the given proof problem with conditions and the answer.
theorem triangle_angle_bisector_ratio
  (X Y Z Q F G : Type)
  (XY XZ YZ : ℝ)
  (angle_bisectors_at : X → Y → Z → Q → F → G → Prop)
  (intersect_at_Q : X → F → G → Prop)
  (h1 : XY = 8)
  (h2 : XZ = 6)
  (h3 : YZ = 4)
  (h4 : angle_bisectors_at X Y Z Q F G)
  (h5 : intersect_at_Q X F G Q):
  ∃ (YQ QG : ℝ), YQ / QG = 7 / 3 := 
sorry

end triangle_angle_bisector_ratio_l141_141591


namespace complex_log_def_l141_141549

noncomputable def complex_log (z : ℂ) (k : ℤ) : ℂ := 
  complex.log z.abs + complex.I * (z.arg + 2 * k * real.pi)

theorem complex_log_def (z : ℂ) : 
  ∃ k : ℤ, complex.exp (complex_log z k) = z := 
begin
  use 0, -- Example to show existence, using k = 0
  unfold complex_log,
  sorry
end

end complex_log_def_l141_141549


namespace unique_integers_sum_21_l141_141749

theorem unique_integers_sum_21 :
  ∃ (a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ),
    (0 ≤ a_2 ∧ a_2 < 2) ∧ (0 ≤ a_3 ∧ a_3 < 3) ∧ (0 ≤ a_4 ∧ a_4 < 4) ∧
    (0 ≤ a_5 ∧ a_5 < 5) ∧ (0 ≤ a_6 ∧ a_6 < 6) ∧ (0 ≤ a_7 ∧ a_7 < 7) ∧
    (0 ≤ a_8 ∧ a_8 < 8) ∧
    (8 / 13 : ℚ) = (a_2 / 2! + a_3 / 3! + a_4 / 4! + a_5 / 5! + a_6 / 6! + a_7 / 7! + a_8 / 8!) ∧
    (a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 = 21) :=
sorry

end unique_integers_sum_21_l141_141749


namespace divide_one_meter_into_100_parts_l141_141856

theorem divide_one_meter_into_100_parts :
  (1 / 100 : ℝ) = 1 / 100 := 
by
  sorry

end divide_one_meter_into_100_parts_l141_141856


namespace sum_of_digits_from_1_to_2008_l141_141316

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def sum_of_digits_range (m n : ℕ) : ℕ :=
  (range (n - m + 1)).sum (λ i, sum_of_digits (i + m))

theorem sum_of_digits_from_1_to_2008 :
  sum_of_digits_range 1 2008 = 28054 :=
by
  sorry

end sum_of_digits_from_1_to_2008_l141_141316


namespace cyclic_quadrilateral_BXZY_l141_141595

theorem cyclic_quadrilateral_BXZY 
  (A B C X Y Z : Type) 
  [is_triangle A B C]
  (X_on_AB : X ∈ segment A B)
  (Y_on_BC : Y ∈ segment B C)
  (AY_CY_eq : AY = CY)
  (AB_CZ_eq : AB = CZ)
  (AY_intersects_CX_at_Z : ∃ Z, intersect (segment A Y) (segment C X) Z) :
  cyclic B X Z Y :=
sorry

end cyclic_quadrilateral_BXZY_l141_141595


namespace find_a_for_decreasing_interval_l141_141173

theorem find_a_for_decreasing_interval :
  ∃ a : ℝ, ∀ x : ℝ, (f'(x) = 0 ↔ (x = -1 ∨ x = 4)) :=
begin
  let f : ℝ → ℝ := λ x, (1 / 3) * x^3 - (3 / 2) * x^2 + a * x + 4,
  let f' : ℝ → ℝ := λ x, x^2 - 3 * x + a,
  use -4,
  intros x,
  split,
  { intro h,
    -- Show that the roots are exactly -1 or 4 when a = -4
    sorry },
  { intro h,
    -- Show that if x = -1 or x = 4, then f'(x) = 0
    sorry }
end

end find_a_for_decreasing_interval_l141_141173


namespace simplify_expression_l141_141265

theorem simplify_expression :
  (Real.cbrt 8 - Real.sqrt (25 / 2))^2 = (33 - 20 * Real.sqrt 2) / 2 := 
by 
  sorry

end simplify_expression_l141_141265


namespace three_digit_perfect_cubes_divisible_by_8_l141_141960

theorem three_digit_perfect_cubes_divisible_by_8 :
  set.card {x : ℕ | 100 ≤ x ∧ x ≤ 999 ∧ (∃ n : ℕ, x = 8 * n^3)} = 2 :=
sorry

end three_digit_perfect_cubes_divisible_by_8_l141_141960


namespace rowing_upstream_distance_l141_141411

theorem rowing_upstream_distance (b s d : ℝ) (h_stream_speed : s = 5)
    (h_downstream_distance : 60 = (b + s) * 3)
    (h_upstream_time : d = (b - s) * 3) : 
    d = 30 := by
  have h_b : b = 15 := by
    linarith [h_downstream_distance, h_stream_speed]
  rw [h_b, h_stream_speed] at h_upstream_time
  linarith [h_upstream_time]

end rowing_upstream_distance_l141_141411


namespace radius_of_roots_circle_l141_141822

theorem radius_of_roots_circle (z : ℂ) (hz : (z - 2)^6 = 64 * z^6) : ∃ r : ℝ, r = 2 / 3 :=
by
  sorry

end radius_of_roots_circle_l141_141822


namespace second_order_derivative_y_l141_141494

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

def x (t : ℝ) : ℝ := sqrt (t - 1)
def y (t : ℝ) : ℝ := t / sqrt (1 - t)

theorem second_order_derivative_y''_xx (t : ℝ) (h1 : t > 1) (h2 : t < 1) : 
  let x := sqrt (t - 1)
  let y := t / sqrt (1 - t)
  y''_xx = (2 : ℝ) / sqrt ((1 - t)^3) :=
begin
  sorry
end

end second_order_derivative_y_l141_141494


namespace tangent_line_exists_l141_141223

noncomputable def tangent_line_problem := ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  Int.gcd (Int.gcd a b) c = 1 ∧ 
  (∀ x y : ℝ, a * x + b * (x^2 + 52 / 25) = c ∧ a * (y^2 + 81 / 16) + b * y = c) ∧ 
  a + b + c = 168

theorem tangent_line_exists : tangent_line_problem := by
  sorry

end tangent_line_exists_l141_141223


namespace probability_event_l141_141807

-- Define the rectangle and vertices
def rect (x y : ℝ) : Prop := 
    0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 2

-- Define the event x < 2y
def event (x y : ℝ) : Prop := 
    x < 2 * y

-- Probability that a point (x,y) randomly picked from the rectangle satisfies x < 2y
theorem probability_event (x y : ℝ) (hx : rect x y) : 
    (∃ (∅ x' y'), rect x' y' ∧ event x' y') → 
    ∃ P, P = 1 / 2 := 
by 
    sorry

end probability_event_l141_141807


namespace major_snow_shadow_length_l141_141584

theorem major_snow_shadow_length :
  ∃ (a1 d : ℝ), 
  (3 * a1 + 12 * d = 16.5) ∧ 
  (12 * a1 + 66 * d = 84) ∧
  (a1 + 11 * d = 12.5) := 
sorry

end major_snow_shadow_length_l141_141584


namespace exp_ge_add_one_exp_plus_sin_cos_ge_2_minus_ax_l141_141116

-- Part 1: Prove that ∀ x, e^x ≥ x + 1
theorem exp_ge_add_one (x : ℝ) : exp x ≥ x + 1 :=
sorry

-- Part 2: Prove that ∀ a ∈ ℝ, (∀ x ≥ 0, e^x + sin x + cos x - 2 - ax ≥ 0) ↔ (a ≤ 2)
theorem exp_plus_sin_cos_ge_2_minus_ax (a : ℝ) : 
  (∀ x, x ≥ 0 → exp x + sin x + cos x - 2 - a * x ≥ 0) ↔ (a ≤ 2) :=
sorry

end exp_ge_add_one_exp_plus_sin_cos_ge_2_minus_ax_l141_141116


namespace coins_dimes_count_l141_141375

theorem coins_dimes_count :
  ∃ (p n d q : ℕ), 
    p + n + d + q = 10 ∧ 
    p + 5 * n + 10 * d + 25 * q = 110 ∧ 
    p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 2 ∧ d = 5 :=
by {
    sorry
}

end coins_dimes_count_l141_141375


namespace f_f_4_eq_1_l141_141934

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 x

theorem f_f_4_eq_1 : f (f 4) = 1 := by
  sorry

end f_f_4_eq_1_l141_141934


namespace smallest_n_no_two_powers_sum_to_square_l141_141086

theorem smallest_n_no_two_powers_sum_to_square :
  ∃ n : ℕ, n > 1 ∧ (∀ m k : ℕ, ¬(∃ a : ℕ, n^m + n^k = a^2)) ∧ n = 4 :=
begin
  sorry
end

end smallest_n_no_two_powers_sum_to_square_l141_141086


namespace eval_custom_op_l141_141174

def custom_op (a b : ℤ) : ℤ := 2 * b + 5 * a - a^2 - b

theorem eval_custom_op : custom_op 3 4 = 10 :=
by
  sorry

end eval_custom_op_l141_141174


namespace arithmetic_sequence_50th_term_l141_141177

-- Define the arithmetic sequence parameters
def first_term : Int := 2
def common_difference : Int := 5

-- Define the formula to calculate the n-th term of the sequence
def nth_term (n : Nat) : Int :=
  first_term + (n - 1) * common_difference

-- Prove that the 50th term of the sequence is 247
theorem arithmetic_sequence_50th_term : nth_term 50 = 247 :=
  by
  -- Proof goes here
  sorry

end arithmetic_sequence_50th_term_l141_141177


namespace minimum_construction_cost_l141_141064

def construction_cost (x : ℝ) : ℝ :=
  480 + 320 * x + (640 / x)

theorem minimum_construction_cost :
  ∃ x : ℝ, x > 0 ∧ construction_cost x = 1276.8 := 
  sorry

end minimum_construction_cost_l141_141064


namespace find_f_neg_two_l141_141110

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (a b : ℝ) : f(a + b) = f(a) + f(b)
axiom f_of_two : f(2) = 1

theorem find_f_neg_two : f(-2) = -1 := by
  sorry

end find_f_neg_two_l141_141110


namespace sec_150_eq_neg_2_sqrt_3_div_3_l141_141075

def sec (θ : ℝ) : ℝ := 1 / cos θ

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l141_141075


namespace solve_inequality_l141_141734

theorem solve_inequality (x : ℝ) (h : 1 / (x - 1) < -1) : 0 < x ∧ x < 1 :=
sorry

end solve_inequality_l141_141734


namespace train_length_l141_141817

noncomputable def convert_speed (v_kmh : ℝ) : ℝ :=
  v_kmh * (5 / 18)

def length_of_train (speed_mps : ℝ) (time_sec : ℝ) : ℝ :=
  speed_mps * time_sec

theorem train_length (v_kmh : ℝ) (t_sec : ℝ) (length_m : ℝ) :
  v_kmh = 60 →
  t_sec = 45 →
  length_m = 750 →
  length_of_train (convert_speed v_kmh) t_sec = length_m :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end train_length_l141_141817


namespace complex_set_property_l141_141881

variable {a b c d : ℂ}
variable {S : Set ℂ}

theorem complex_set_property (hS : ∀ x y ∈ S, x * y ∈ S)
  (ha : a = 1)
  (hb : b ^ 2 = 1)
  (hc : c ^ 2 = b)
  (HS : S = {a, b, c, d}) :
  b + c + d = -1 :=
sorry

end complex_set_property_l141_141881


namespace pentagon_angle_E_l141_141435

theorem pentagon_angle_E {ABCDE : Type*} [pentagon ABCDE] 
  (h1 : side_length_eq ABCDE)
  (h2 : ∠A = 120 ∧ ∠B = 120) : 
  ∠E = 90 := 
sorry

end pentagon_angle_E_l141_141435


namespace ratios_bound_l141_141660

axiom triangle {A B C : Type} (A B C : A) : Prop
axiom incenter {A B C I : Type} {A B C : A} : Prop
axiom angle_bisectors {A B C A' B' C' : Type} {A B C : A} : Prop

theorem ratios_bound (A B C I A' B' C' : Type) 
  (h1 : triangle A B C) 
  (h2 : incenter I A B C) 
  (h3 : angle_bisectors A B C A' B' C') :
  1 / 4 ≤ (AI • BI • CI) / (AA' • BB' • CC') ∧ (AI • BI • CI) / (AA' • BB' • CC') ≤ 8 / 27 :=
sorry

end ratios_bound_l141_141660


namespace percent_increase_sales_l141_141379

-- Define constants for sales
def sales_last_year : ℕ := 320
def sales_this_year : ℕ := 480

-- Define the percent increase formula
def percent_increase (old_value new_value : ℕ) : ℚ :=
  ((new_value - old_value) / old_value) * 100

-- Prove the percent increase from last year to this year is 50%
theorem percent_increase_sales : percent_increase sales_last_year sales_this_year = 50 := by
  sorry

end percent_increase_sales_l141_141379


namespace wise_men_avoid_execution_l141_141302

-- Definitions for colors
inductive Color
| White
| Blue
| Red

-- Number of wise men
constant num_wise_men : Nat := 100

-- The theorem to prove that 99 wise men will avoid execution.
theorem wise_men_avoid_execution :
  ∃ strategy : (fin num_wise_men → list Color → Color),
    ∀ hats : vector Color num_wise_men,
      let results := vector.map₂ (λ i (_, hat), 
        if i = 0 then 
          strategy 0 (hats.to_list.tail)
        else 
          strategy i ((hats.drop (i + 1)).to_list)) 
        (vector.range num_wise_men).val hats.val in
      (vector.length (vector.filter (λ pair, pair.fst = pair.snd) (results.zip hats))).val = num_wise_men - 1 :=
sorry

end wise_men_avoid_execution_l141_141302


namespace problem_solution_l141_141331

noncomputable def triangle_points (A B C: Point) : Prop :=
  side_length A B = 13 ∧ 
  side_length B C = 14 ∧ 
  side_length C A = 15

noncomputable def circumcircle (ABC : Triangle) : Circle

noncomputable def orthocenter (ABC : Triangle) : Point

noncomputable def circumcircle_intersection (ABC : Triangle) (H : Point) : Point

noncomputable def intersect (line1 line2: Line) : Point 

noncomputable def side_length (A B : Point) : ℝ

noncomputable def area_quad (H D X F : Point) : ℝ

theorem problem_solution :
  ∀ (A B C H D G X F: Point),
  triangle_points A B C →
  let Γ := circumcircle (A, B, C) in
  let H := orthocenter (A, B, C) in
  let D := circumcircle_intersection (A, H, Γ) in
  let G := circumcircle_intersection (B, H, Γ) in
  let F := intersect (B, H) (A, C) in
  let X := intersect (G, D) (A, C) in
  area_quad H D X F ≤ 24 :=
sorry

end problem_solution_l141_141331


namespace quadratic_inequality_l141_141946

theorem quadratic_inequality (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 2 * x + 1 < 0) → a < 1 := 
by
  sorry

end quadratic_inequality_l141_141946


namespace max_value_of_sum_max_value_achievable_l141_141692

theorem max_value_of_sum (a b c d : ℝ) 
  (h : a^6 + b^6 + c^6 + d^6 = 64) : a^7 + b^7 + c^7 + d^7 ≤ 128 :=
sorry

theorem max_value_achievable : ∃ a b c d : ℝ,
  a^6 + b^6 + c^6 + d^6 = 64 ∧ a^7 + b^7 + c^7 + d^7 = 128 :=
sorry

end max_value_of_sum_max_value_achievable_l141_141692


namespace measure_of_angle_E_l141_141439

theorem measure_of_angle_E
  (ABCDE : Type)
  (pentagon : ∀ (P : fin 5 → ABCDE), convex_poly P)
  (equal_sides : ∀ (P : fin 5 → ABCDE), sides_equal_length P)
  (angle_A_eq_120 : ∀ (P : fin 5 → ABCDE), angle P 0 = 120)
  (angle_B_eq_120 : ∀ (P : fin 5 → ABCDE), angle P 1 = 120) :
  ∃ P : fin 5 → ABCDE, angle P 4 = 120 := 
sorry

end measure_of_angle_E_l141_141439


namespace possible_degrees_of_remainder_l141_141370

theorem possible_degrees_of_remainder (p : Polynomial ℝ) (h : p = 3 * X^3 - 5 * X^2 + 2 * X - 8) :
  ∃ d : Finset ℕ, d = {0, 1, 2} :=
by
  sorry

end possible_degrees_of_remainder_l141_141370


namespace cylinder_volume_surface_area_eq_specific_cylinder_dimensions_identical_cylinders_l141_141900

noncomputable def cylinder_dimensions (r m : ℝ) : ℝ × ℝ :=
  ( (sqrt(r^2 + 4*m*r) - r) / 2, 4*m*r^2 / ((sqrt(r^2 + 4*m*r) - r)^2) )

theorem cylinder_volume_surface_area_eq
  (r m : ℝ) (r_1 m_1 : ℝ) (h1: r_1 = (sqrt(r^2 + 4*m*r) - r) / 2) (h2: m_1 = 4*m*r^2 / ((sqrt(r^2 + 4*m*r) - r)^2)) :
  (2 * Real.pi * r^2 + 2 * Real.pi * r * m = 2 * Real.pi * r_1^2 + 2 * Real.pi * r_1 * m_1) ∧
  (Real.pi * r^2 * m = Real.pi * r_1^2 * m_1) :=
by
  sorry

theorem specific_cylinder_dimensions :
  cylinder_dimensions 8 48 = (16, 12) :=
by
  sorry

theorem identical_cylinders (r m : ℝ) (h : r = m / 2) :
  cylinder_dimensions r m = (r, m) :=
by
  sorry

end cylinder_volume_surface_area_eq_specific_cylinder_dimensions_identical_cylinders_l141_141900


namespace unique_triangle_determination_l141_141261

theorem unique_triangle_determination 
  (angles_determine : ∀ {a b c : ℝ}, is_triangle (a, b, c) → is_triangle ((a + b + c), (a - b + c), (a + b - c)) ↔ a * a + b * b = c * c)
  (ratios_determine : ∀ {a b c : ℝ}, (a/b) = (b/c) → is_equilateral_triangle a b c)
  (sum_of_angles : ∀ {α β γ : ℝ}, α + β + γ = π)
  (angle_side_insufficient : ∀ {α a : ℝ}, is_triangle_with_angle_opposite α a → model_triangle α a) :
  ∀ {m s : ℝ}, m/s ≠ 1 → ¬ uniquely_determined_triangle (m, s, m/s) :=
by
  sorry

end unique_triangle_determination_l141_141261


namespace N_8_12_eq_288_l141_141724

-- Definitions for various polygonal numbers
def N3 (n : ℕ) : ℕ := n * (n + 1) / 2
def N4 (n : ℕ) : ℕ := n^2
def N5 (n : ℕ) : ℕ := 3 * n^2 / 2 - n / 2
def N6 (n : ℕ) : ℕ := 2 * n^2 - n

-- General definition conjectured
def N (n k : ℕ) : ℕ := (k - 2) * n^2 / 2 + (4 - k) * n / 2

-- The problem statement to prove N(8, 12) == 288
theorem N_8_12_eq_288 : N 8 12 = 288 := by
  -- We would need the proofs for the definitional equalities and calculation here
  sorry

end N_8_12_eq_288_l141_141724


namespace ratio_of_counters_l141_141258

theorem ratio_of_counters (C_K M_K C_total M_ratio : ℕ)
  (h1 : C_K = 40)
  (h2 : M_K = 50)
  (h3 : M_ratio = 4 * M_K)
  (h4 : C_total = C_K + M_ratio)
  (h5 : C_total = 320) :
  C_K ≠ 0 → (320 - M_ratio) / C_K = 3 :=
by
  sorry

end ratio_of_counters_l141_141258


namespace triangle_division_congruent_l141_141986

theorem triangle_division_congruent 
  (A B C D K N P M : Point)
  (h_parallel : Parallel (Line.mk A B) (Line.mk C D))
  (h_acute_ABC : AcuteAngle (Angle.mk A B C))
  (h_acute_BAD : AcuteAngle (Angle.mk B A D))
  (h_K_intersection : Intersection K (Line.mk A C) (Line.mk B D))
  (h_NP_parallel : Parallel (Line.mk N P) (Line.mk A B))
  (h_NKP_collinear : Collinear N K P)
  (h_M_midpoint : Midpoint M A B) :
  ∃ X₁ X₂ X₃ X₄ Y₁ Y₂ Y₃ Y₄ : Triangle,
    DivideInto (Triangle.mk A B C) [X₁, X₂, X₃, X₄] ∧
    DivideInto (Triangle.mk A B D) [Y₁, Y₂, Y₃, Y₄] ∧
    ∀ i, i ∈ {0, 1, 2, 3} → Congruent (X i) (Y i) :=
sorry

end triangle_division_congruent_l141_141986


namespace inserting_eights_is_composite_l141_141726

theorem inserting_eights_is_composite (n : ℕ) : ¬ Nat.Prime (2000 * 10^n + 8 * ((10^n - 1) / 9) + 21) := 
by sorry

end inserting_eights_is_composite_l141_141726


namespace eccentricity_range_of_ellipse_l141_141170

theorem eccentricity_range_of_ellipse
  (a b c e : ℝ)
  (h1 : a > b > 0)
  (h2 : a^2 = b^2 + c^2)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ∧ (x^2 + y^2 = (b / 2 + c)^2)) :
  (sqrt 5 / 5 < e) ∧ (e < 3 / 5) :=
sorry

end eccentricity_range_of_ellipse_l141_141170


namespace students_behind_yoongi_l141_141391

theorem students_behind_yoongi (total_students jungkoo_position students_between_jungkook_yoongi : ℕ) 
    (h1 : total_students = 20)
    (h2 : jungkoo_position = 3)
    (h3 : students_between_jungkook_yoongi = 5) : 
    (total_students - (jungkoo_position + students_between_jungkook_yoongi + 1)) = 11 :=
by
  sorry

end students_behind_yoongi_l141_141391


namespace geometric_seq_a9_l141_141997

theorem geometric_seq_a9 
  (a : ℕ → ℤ)  -- The sequence definition
  (h_geometric : ∀ n : ℕ, a (n+1) = a 1 * (a 2 ^ n) / a 1 ^ n)  -- Geometric sequence property
  (h_a1 : a 1 = 2)  -- Given a₁ = 2
  (h_a5 : a 5 = 18)  -- Given a₅ = 18
: a 9 = 162 := sorry

end geometric_seq_a9_l141_141997


namespace sum_of_digits_2008_l141_141314

theorem sum_of_digits_2008 :
  (Finset.range 2008).sum (λ n, n.digits 10).sum = 28054 :=
sorry

end sum_of_digits_2008_l141_141314


namespace sum_of_values_l141_141405

def f (x : ℝ) : ℝ := sorry -- The actual function definition is not known.

theorem sum_of_values (T : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → 3 * f(x) + f(1 / x) = 6 * x + 3) →
  (∃ x : ℝ, f(x) = 1000 ∧ T = ∑ roots of the equation) →
  abs (T - 444) < 1 :=
by
  sorry

end sum_of_values_l141_141405


namespace book_cost_in_usd_l141_141434

-- Definition of given conditions
def cost_in_gbp : ℝ := 25
def conversion_factor_gbp_to_usd : ℝ := 1 / 0.75

-- Target statement we need to prove
theorem book_cost_in_usd : 
  (cost_in_gbp * conversion_factor_gbp_to_usd).round(2) = 33.33 :=
by
  sorry

end book_cost_in_usd_l141_141434


namespace problem_l141_141912

variable (a b x y : ℝ)

theorem problem (h₁ : a + b = 2) (h₂ : x + y = 2) (h₃ : ax + by = 5) :
  (a^2 + b^2) * x * y + a * b * (x^2 + y^2) = -5 := by
sorry

end problem_l141_141912


namespace T_21_value_l141_141515

noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, 2 * a (n + 1) = a n + a (n + 2)

def a_n : ℕ → ℤ :=
λ n, 2 * n - 1

def b_n (n : ℕ) : ℤ :=
(2 * n - 1) - (-1)^n * n

def T_n (n : ℕ) : ℤ :=
(n^2 - n) - ((-1)^n - 1)/4 - ((-1)^n * n)/2

lemma seq_conditions_holds :
  (a_n 3 = 5) ∧ (a_n 2 + a_n 6 = 14) ∧ is_arithmetic_sequence a_n :=
by {
  split,
  { unfold a_n, norm_num },
  split,
  { unfold a_n, norm_num },
  { unfold is_arithmetic_sequence a_n, intro n, norm_num },
  sorry
}

theorem T_21_value :
  T_21 = 425 + 3 / 4 :=
by {
  unfold T_n,
  unfold b_n,
  norm_num,
  sorry
}

end T_21_value_l141_141515


namespace perpendicular_line_in_plane_l141_141879

theorem perpendicular_line_in_plane {l : Line} {α : Plane} : 
  ∃ m : Line, m ∈ α ∧ m ⟂ l :=
sorry

end perpendicular_line_in_plane_l141_141879


namespace unicorn_problem_solution_l141_141818

def rope_length_on_tower(a b c : ℕ) (h1 : c.prime) : Prop :=
  a = 90 ∧ b = 156 ∧ c = 3 ∧ (30 - (6 + 2 * Real.sqrt 21)).to_rat = (a - Real.sqrt b) / c

theorem unicorn_problem_solution :
  ∃ (a b c : ℕ), rope_length_on_tower a b c ∧ a + b + c = 249 :=
by
  use 90, 156, 3
  split
  · simp [rope_length_on_tower]
    apply prime_three
  · rfl

-- Sorry added to skip the actual proof steps
sorry

end unicorn_problem_solution_l141_141818


namespace max_size_S_family_l141_141171

/-- An S family (Sperner family) is a family of subsets where no one subset is contained within another. -/
def S_family {α : Type*} (𝒜 : set (set α)) : Prop :=
  ∀ (A B : set α), A ∈ 𝒜 → B ∈ 𝒜 → A ⊆ B → A = B

/-- X is a set with elements {1, ..., n}. -/
def X (n : ℕ) : set (ℕ) := { i | 1 ≤ i ∧ i ≤ n }

/-- The main theorem: maximum size of an S family of subsets of X is given by binomial coefficient C(n, floor n/2). -/
theorem max_size_S_family (n : ℕ) (𝒜 : set (set (X n))) (h𝒜 : S_family 𝒜) :
  finset.card 𝒜.to_finset ≤ nat.choose n (n / 2) :=
sorry

end max_size_S_family_l141_141171


namespace transaction_gain_per_year_l141_141806

def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / (n : ℝ)) ^ (n * t)

theorem transaction_gain_per_year :
  let borrowed_amount := 8000
  let borrowed_rate := 0.04
  let borrowed_compound_periods := 4
  let borrow_duration := 2
  let lent_amount := 8000
  let lent_rate := 0.06
  let lent_compound_periods := 2
  let lend_duration := 2
  in
  let amount_owed := compound_interest borrowed_amount borrowed_rate borrowed_compound_periods borrow_duration
  let amount_earned := compound_interest lent_amount lent_rate lent_compound_periods lend_duration
  let total_gain := amount_earned - amount_owed
  let gain_per_year := total_gain / lend_duration
  gain_per_year ≈ 170.61 :=
by
  sorry

end transaction_gain_per_year_l141_141806


namespace count_exactly_one_zero_in_1_to_3050_l141_141958

theorem count_exactly_one_zero_in_1_to_3050 : 
  (Finset.filter 
    (λ n : ℕ, (Nat.digits 10 n).filter (λ d, d = 0) = [0]) 
    (Finset.range 3051)).card = 657 :=
by
  sorry

end count_exactly_one_zero_in_1_to_3050_l141_141958


namespace number_of_groups_correct_l141_141371

-- Define the conditions
def tallest_height : ℝ := 175
def shortest_height : ℝ := 150
def class_width : ℝ := 3

-- Define the range of heights
def height_range : ℝ := tallest_height - shortest_height

-- Define the number of groups needed as the ceiling of the range divided by the class width
def number_of_groups : ℝ := (height_range / class_width).ceil

-- State the theorem
theorem number_of_groups_correct : number_of_groups = 9 := sorry

end number_of_groups_correct_l141_141371


namespace man_twice_son_age_in_years_l141_141008

theorem man_twice_son_age_in_years :
  let S := 33 in
  let M := S + 35 in
  ∃ (Y : ℕ), M + Y = 2 * (S + Y) ∧ Y = 2 :=
by
  let S := 33
  let M := S + 35
  use 2
  split
  · calc
      M + 2 = 33 + 35 + 2 := by rfl
           ... = 68 + 2 := rfl
           ... = 70 := by norm_num
      2 * (S + 2) = 2 * (33 + 2) := by rfl
                 ... = 2 * 35 := rfl
                 ... = 70 := by norm_num
  · rfl

end man_twice_son_age_in_years_l141_141008


namespace constant_term_in_expansion_l141_141710

noncomputable def binomial_expansion : List (ℤ → ℤ) := sorry

theorem constant_term_in_expansion (f : ℤ -> ℤ) ((x ^ 2 + 2) * (1/x^2 - 1)^5) : f (binomial_expansion) = 3 :=
  sorry

end constant_term_in_expansion_l141_141710


namespace max_non_similar_sequences_correct_l141_141602

open Function

def is_sequence (N : ℕ) (s : list ℤ) : Prop :=
  s.length = N ∧ ∀ e ∈ s, e = 1 ∨ e = -1

def is_similar (N : ℕ) (s1 s2 : list ℤ) : Prop := 
  ∃ f : ℕ → ℕ, 
    (∀ i, f i ≤ 5) ∧
    ((∀ i, f i ≠ 0 → s1[i] = -s2[i]) ∨ (f i = 0 → s1[i] = s2[i])) 

def max_non_similar_sequences (N : ℕ) : ℕ :=
  if N < 5 then 0
  else 16

theorem max_non_similar_sequences_correct (N : ℕ) (h : N ≥ 5) :
  max_non_similar_sequences N = 16 :=
by sorry

end max_non_similar_sequences_correct_l141_141602


namespace tangent_line_at_1_range_of_m_l141_141067

noncomputable def f (x: ℝ) : ℝ := x^2 + x
noncomputable def g (x m: ℝ) : ℝ := (1/3) * x^3 - 2 * x + m

theorem tangent_line_at_1 :
  let x := (1 : ℝ) in
  let y := f x in
  let slope := (2 * x + 1 : ℝ) in
  (3 * x - slope * y - 1 = 0) :=
sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, -4 ≤ x → x ≤ 4 → f x ≥ g x m) → m ≤ -5 / 3 :=
sorry

end tangent_line_at_1_range_of_m_l141_141067


namespace ball_distribution_l141_141156

theorem ball_distribution :
  let balls := 5
  let boxes := 3
  let at_least_one_box_contains_2_balls := 
    ∃ (distribution : Finset (Finset (Fin balls)) -> Prop),
      ∃ d ∈ distribution, (∃ b ∈ d, b.card = 2) ∧
                           (∀ b ∈ d, b.card ≤ boxes)
  in
  ∃ (number_of_ways : ℕ), 
  number_of_ways = 810 ∧ number_of_ways =  choose 5 2 * 3 * 3^3 
:= sorry

end ball_distribution_l141_141156


namespace wastewater_volume_2013_l141_141996

variable (x_2013 x_2014 : ℝ)
variable (condition1 : x_2014 = 38000)
variable (condition2 : x_2014 = 1.6 * x_2013)

theorem wastewater_volume_2013 : x_2013 = 23750 := by
  sorry

end wastewater_volume_2013_l141_141996


namespace polynomial_irreducible_l141_141232

def is_monic (P : Polynomial ℤ) : Prop :=
  P.leadingCoeff = 1

def coefficients_except_leading_divisible_by (P : Polynomial ℤ) (p : ℤ) : Prop :=
  ∀ (i : ℕ), i < P.natDegree → p ∣ P.coeff i

def constant_term_not_divisible_by_square (P : Polynomial ℤ) (p : ℤ) : Prop :=
  ¬(p^2 ∣ P.coeff 0)

theorem polynomial_irreducible (P : Polynomial ℤ) (p : ℤ) :
  is_monic P →
  coefficients_except_leading_divisible_by P p →
  constant_term_not_divisible_by_square P p →
  irreducible P :=
by
  sorry

end polynomial_irreducible_l141_141232


namespace special_natural_sum_l141_141430

def is_special_natural (n : ℕ) : Prop :=
  n % 2 = 1 ∧ n % 3 = 1 ∧ n % 4 = 1 ∧ n % 5 = 1 ∧ n % 7 = 1

noncomputable def smallest_special_3_digit : ℕ :=
  Inf {n | is_special_natural n ∧ 100 ≤ n ∧ n < 1000}

noncomputable def largest_special_3_digit : ℕ :=
  Sup {n | is_special_natural n ∧ 100 ≤ n ∧ n < 1000}

theorem special_natural_sum : smallest_special_3_digit + largest_special_3_digit = 1262 :=
sorry

end special_natural_sum_l141_141430


namespace food_duration_after_increase_l141_141328

theorem food_duration_after_increase:
  (initial_men : ℕ) (initial_days : ℕ) (additional_men : ℕ) (days_consumed : ℕ)
  (more_days : ℕ) (total_man_days : ℕ) (remaining_man_days : ℕ)
  (increased_men : ℕ) :
  initial_men = 760 → initial_days = 22 → additional_men = 40 → days_consumed = 2 →
  total_man_days = initial_men * initial_days →
  remaining_man_days = total_man_days - (initial_men * days_consumed) →
  increased_men = initial_men + additional_men →
  more_days = remaining_man_days / increased_men →
  more_days = 19 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end food_duration_after_increase_l141_141328


namespace total_students_like_sports_l141_141184

def Total_students := 30

def B : ℕ := 12
def C : ℕ := 10
def S : ℕ := 8
def BC : ℕ := 4
def BS : ℕ := 3
def CS : ℕ := 2
def BCS : ℕ := 1

theorem total_students_like_sports : 
  (B + C + S - (BC + BS + CS) + BCS = 22) := by
  sorry

end total_students_like_sports_l141_141184


namespace vieta_two_product_roots_l141_141230

-- Define the polynomial and conditions
def poly := (λ x : ℝ, 2 * x^3 - 5 * x^2 + 7 * x - 3)

-- Assume p, q, r are roots of the polynomial
def is_root (x : ℝ) := poly x = 0
variables {p q r : ℝ}
hypothesis (h1 : is_root p)
hypothesis (h2 : is_root q)
hypothesis (h3 : is_root r)

-- Use Vieta's formulas to define sum and product of roots
def sum_roots := p + q + r
def prod_roots_two_at_a_time := p * q + q * r + r * p

-- Using Vieta's formulas, we need to prove that
theorem vieta_two_product_roots : prod_roots_two_at_a_time = 7 / 2 :=
by sorry

end vieta_two_product_roots_l141_141230


namespace two_digit_number_with_tens_5_l141_141027

-- Definitions and conditions
variable (A : Nat)

-- Problem statement as a Lean theorem
theorem two_digit_number_with_tens_5 (hA : A < 10) : (10 * 5 + A) = 50 + A := by
  sorry

end two_digit_number_with_tens_5_l141_141027


namespace closest_point_in_plane_l141_141491

noncomputable def closest_point (x y z : ℚ) : Prop :=
  ∃ (t : ℚ), 
    x = 2 + 2 * t ∧ 
    y = 3 - 3 * t ∧ 
    z = 1 + 4 * t ∧ 
    2 * (2 + 2 * t) - 3 * (3 - 3 * t) + 4 * (1 + 4 * t) = 40

theorem closest_point_in_plane :
  closest_point (92 / 29) (16 / 29) (145 / 29) :=
by
  sorry

end closest_point_in_plane_l141_141491


namespace star_perimeter_difference_zero_l141_141605

-- Define the property of an equiangular hexagon
def equiangular_hexagon (a b c d e f : ℝ) : Prop :=
  a + b + c + d + e + f = 1 ∧
  ∀ i j k, {
    let θ := (i, j, k) Θ angles_of_hexagon (a, b, c, d, e, f) in
    θ = 120
  }

-- Statement about the difference in star perimeter values
theorem star_perimeter_difference_zero (a b c d e f : ℝ)
  (h : equiangular_hexagon a b c d e f) :
  (max_star_perimeter a b c d e f - min_star_perimeter a b c d e f) = 0 :=
sorry

end star_perimeter_difference_zero_l141_141605


namespace largest_n_value_l141_141764

theorem largest_n_value :
  ∃ n, n < 100000 ∧ (5 * (n - 3) ^ 6 - 2 * n ^ 3 + 20 * n - 35) % 7 = 0 ∧ ∀ m, m < 100000 ∧ (5 * (m - 3) ^ 6 - 2 * m ^ 3 + 20 * m - 35) % 7 = 0 → m ≤ n :=
begin
  use 99998,
  split,
  { exact nat.lt_succ_self 99998 },
  split,
  { norm_num, },
  { intros m hm,
    exact le_of_lt (lt_of_lt_of_le hm (nat.le_succ_self 99998)) },
end

end largest_n_value_l141_141764


namespace least_n_factorial_5250_l141_141765

theorem least_n_factorial_5250 (n : ℕ) (h : 5250 = 2 * 3^2 * 5^ 2 * 7) : 
  (∀ k ≤ n, 5250 ∣ fact k) ↔ n = 10 :=
by sorry

end least_n_factorial_5250_l141_141765


namespace rationalize_denominator_correct_l141_141689

noncomputable def rationalize_denominator : ℚ :=
  let num : ℚ := 2 + real.sqrt 5
  let denom : ℚ := 3 - real.sqrt 5
  let conjugate : ℚ := 3 + real.sqrt 5
  let result := (num * conjugate) / (denom * conjugate)
  let A : ℚ := 11 / 4
  let B : ℚ := 5 / 4
  let C : ℚ := 5
  A * B * C

theorem rationalize_denominator_correct : rationalize_denominator = 275 / 16 :=
by
  sorry

end rationalize_denominator_correct_l141_141689


namespace shaded_area_equilateral_triangle_l141_141603

theorem shaded_area_equilateral_triangle
  (O A B C D E F G H I : Point)
  (ABC : Triangle) 
  (h1 : is_equilateral_triangle ABC)
  (h2 : centroid O ABC)
  (h3 : trisects_sides O ABC D E F G H I)
  (h4 : area ABC = 1 / π) :
  shaded_area O D E F G H I = 2 * sqrt 3 / 27 :=
sorry

end shaded_area_equilateral_triangle_l141_141603


namespace solution_l141_141701

def f (x : ℝ) : ℝ := (2 * x) / 3 + 2
def g (x : ℝ) : ℝ := 5 - 2 * x

theorem solution :
  ∃ a : ℝ, f (g a) = 4 ∧ a = 1 := by
  use 1
  simp [f, g]
  norm_num
  sorry

end solution_l141_141701


namespace renata_final_money_l141_141248

-- Defining the initial condition and the sequence of financial transactions.
def initial_money := 10
def donation := 4
def prize := 90
def slot_loss1 := 50
def slot_loss2 := 10
def slot_loss3 := 5
def water_cost := 1
def lottery_ticket_cost := 1
def lottery_prize := 65

-- Prove that given all these transactions, the final amount of money is $94.
theorem renata_final_money :
  initial_money 
  - donation 
  + prize 
  - slot_loss1 
  - slot_loss2 
  - slot_loss3 
  - water_cost 
  - lottery_ticket_cost 
  + lottery_prize 
  = 94 := 
by
  sorry

end renata_final_money_l141_141248


namespace cone_cylinder_volume_ratio_l141_141796

-- Define the variables and conditions
def r := 6 -- radius of the cylinder and the cone
def h_cylinder := 18 -- height of the cylinder
def h_cone := 1 / 3 * h_cylinder -- height of the cone

-- Calculate the volumes
def V_cylinder := π * r^2 * h_cylinder
def V_cone := 1 / 3 * π * r^2 * h_cone

-- Statement to be proved
theorem cone_cylinder_volume_ratio :
  V_cone / V_cylinder = 1 / 9 :=
sorry

end cone_cylinder_volume_ratio_l141_141796


namespace problem_statement_l141_141150

theorem problem_statement (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (D : ℕ) (M : ℕ) (h_gcd : D = Nat.gcd (Nat.gcd a b) c) (h_lcm : M = Nat.lcm (Nat.lcm a b) c) :
  ((D * M = a * b * c) ∧ ((Nat.gcd a b = 1) ∧ (Nat.gcd b c = 1) ∧ (Nat.gcd a c = 1) → (D * M = a * b * c))) :=
by sorry

end problem_statement_l141_141150


namespace quadratic_equation_with_given_means_l141_141562

theorem quadratic_equation_with_given_means (a b : ℝ)
  (h1: (a + b) / 2 = 8)
  (h2: Real.sqrt (a * b) = 12) : 
  ∀ x : ℝ, x^2 - (a + b) * x + a * b = x^2 - 16 * x + 144 :=
by
  intro x
  rw [h1, h2]
  sorry

end quadratic_equation_with_given_means_l141_141562


namespace sum_of_digits_of_product_45_40_l141_141738

theorem sum_of_digits_of_product_45_40 :
  let n := 45 * 40,
      digits := [1, 8, 0, 0],
      sum_digits := digits.sum in
  sum_digits = 9 :=
by
  let n := 45 * 40;
  let digits := [1, 8, 0, 0];
  let sum_digits := digits.sum;
  show sum_digits = 9;
  sorry

end sum_of_digits_of_product_45_40_l141_141738


namespace limit_of_sequence_is_unique_l141_141253

-- Define the sequence and the limit conditions
noncomputable def sequence : ℕ → ℝ := sorry

def is_limit (a : ℝ) (seq : ℕ → ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n > N, |seq n - a| < ε

theorem limit_of_sequence_is_unique (a b : ℝ) :
  (∃ (seq : ℕ → ℝ), (is_limit a seq) ∧ (is_limit b seq) ∧ a ≠ b) → False :=
by
  sorry

end limit_of_sequence_is_unique_l141_141253


namespace sequence_limit_l141_141682

-- Define the sequence
def a_n (n : ℕ) : ℝ := (2 * n + 3) / (n + 5)

-- Define the limit value
def a : ℝ := 2

-- Formal statement of the theorem to prove
theorem sequence_limit :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε := by
  sorry

end sequence_limit_l141_141682


namespace range_of_a_l141_141108

-- Given f(x) = (x + a / x - 1) * exp(x)
def f (x a : ℝ) : ℝ := (x + a / x - 1) * Real.exp x

-- Definition of the derivative f'(x)
def f_prime (x a : ℝ) : ℝ := 
  ((x^3 + a * x - a) / x^2) * Real.exp x

-- Function h(x) used in the density of proofs
def h (x a : ℝ) : ℝ := x^3 + a * x - a

-- Derivative h'(x)
def h_prime (x a : ℝ) : ℝ := 3 * x^2 + a

-- Statement of the problem
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 3 ∧ f_prime x a = 0) → 
  -27 < a ∧ a < -3 :=
sorry

end range_of_a_l141_141108


namespace proposition_1_proposition_2_l141_141880

-- Proposition 1: ∀ a, b, c ∈ ℝ, c ≠ 0 implies (ac^2 > bc^2 → a > b)
theorem proposition_1 (a b c : ℝ) (hc_ne_zero : c ≠ 0) :
  ac^2 > bc^2 → a > b := by
  sorry

-- Proposition 2: ∀ a, b, c, d ∈ ℝ, (a > b ∧ c > d → a + c > b + d)
theorem proposition_2 (a b c d : ℝ) :
  a > b ∧ c > d → a + c > b + d := by
  sorry

end proposition_1_proposition_2_l141_141880


namespace maximize_expression_l141_141770

theorem maximize_expression (x y : ℝ) : 
  let expr := 2005 - (x + y) ^ 2 in
  is_max (λ (u v: ℝ), 2005 - (u + v) ^ 2) (x, y) → x = -y :=
by
  sorry

end maximize_expression_l141_141770


namespace least_remaining_marbles_l141_141326

/-- 
There are 60 identical marbles forming a tetrahedral pile.
The formula for the number of marbles in a tetrahedral pile up to the k-th level is given by:
∑_(i=1)^k (i * (i + 1)) / 6 = k * (k + 1) * (k + 2) / 6.

We must show that the least number of remaining marbles when 60 marbles are used to form the pile is 4.
-/
theorem least_remaining_marbles : ∃ k : ℕ, (60 - k * (k + 1) * (k + 2) / 6) = 4 :=
by
  sorry

end least_remaining_marbles_l141_141326


namespace probability_king_even_coords_2008_l141_141249

noncomputable def king_probability_even_coords (turns : ℕ) : ℝ :=
  let p_stay := 0.4
  let p_edge := 0.1
  let p_diag := 0.05
  if turns = 2008 then
    (5 ^ 2008 + 1) / (2 * 5 ^ 2008)
  else
    0 -- default value for other cases

theorem probability_king_even_coords_2008 :
  king_probability_even_coords 2008 = (5 ^ 2008 + 1) / (2 * 5 ^ 2008) :=
by
  sorry

end probability_king_even_coords_2008_l141_141249


namespace charlie_more_snowballs_l141_141457

section
variables (charlie_snowballs lucy_snowballs : ℕ)
variables (h_charlie : charlie_snowballs = 50)
variables (h_lucy : lucy_snowballs = 19)

theorem charlie_more_snowballs : (charlie_snowballs - lucy_snowballs) = 31 :=
by {
  rw [h_charlie, h_lucy],
  norm_num,
}
end

end charlie_more_snowballs_l141_141457


namespace circle_equation_l141_141872

theorem circle_equation (C : Type) [metric_space C] [normed_group C] [normed_space ℝ C]
  (center p : C)
  (hx : center = (1, 1) : C)
  (hp : p = (1, 0) : C)
  (h : dist center p = 1) :
  ∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 1 :=
by sorry

end circle_equation_l141_141872


namespace number_of_three_digit_cubes_divisible_by_8_l141_141961

-- Definitions based on given conditions
def three_digit_perfect_cubes_divisible_by_8 (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = 8 * k^3

-- The final theorem statement.
theorem number_of_three_digit_cubes_divisible_by_8 : 
  {n : ℕ | three_digit_perfect_cubes_divisible_by_8 n}.finite  ∧ 
  {n : ℕ | three_digit_perfect_cubes_divisible_by_8 n}.to_finset.card = 2 :=
by 
  sorry

end number_of_three_digit_cubes_divisible_by_8_l141_141961


namespace sum_of_digits_2008_l141_141313

theorem sum_of_digits_2008 :
  (Finset.range 2008).sum (λ n, n.digits 10).sum = 28054 :=
sorry

end sum_of_digits_2008_l141_141313


namespace example_is_Lipschitz_min_Lipschitz_constant_sqrt_l141_141848

-- Definitions for the conditions
def Lipschitz (f : ℝ → ℝ) (k : ℝ) (D : set ℝ) : Prop :=
  ∀ x1 x2 ∈ D, |f x1 - f x2| ≤ k * |x1 - x2|

-- Part 1: Verifying the function f(x) = x with Lipschitz constant k = 1 on D = ℝ
theorem example_is_Lipschitz (D : set ℝ) (hD : ∀ x, x ∈ D) :
  Lipschitz (λ x, x) 1 D :=
sorry

-- Part 2: Finding the minimum Lipschitz constant for the function f(x) = √(x+1) on [0, +∞)
theorem min_Lipschitz_constant_sqrt (k : ℝ) :
  (∀ x1 x2 ∈ set.Ici 0, |sqrt (x1 + 1) - sqrt (x2 + 1)| ≤ k * |x1 - x2|) → k ≥ 1/2 :=
sorry

end example_is_Lipschitz_min_Lipschitz_constant_sqrt_l141_141848


namespace problem_statement_l141_141196

noncomputable def curve_1_cartesian : Prop :=
  ∀ (ρ θ : ℝ), ρ = real.cos θ - real.sin θ → ρ^2 = ρ * real.cos θ - ρ * real.sin θ

noncomputable def curve_2_parametric (t : ℝ) : (ℝ × ℝ) :=
  ((1 / 2) - (real.sqrt 2 / 2) * t, (real.sqrt 2 / 2) * t)

noncomputable def curve_intersection_distance (t1 t2 : ℝ) : ℝ :=
  ∥(t1 - t2)∥

theorem problem_statement :
  (∀ ρ θ, ρ = real.cos θ - real.sin θ → (ρ^2 = ρ * real.cos θ - ρ * real.sin θ) → (∃ \(x, y) : ℝ^2,
    x^2 + y^2 - x + y = 0)) ∧
  (∀ t₁ t₂, t₁^2 + real.sqrt 2 / 2 * t₁ - 1 / 4 = 0 ∧ t₂^2 + real.sqrt 2 / 2 * t₂ - 1 / 4 = 0 →
    curve_intersection_distance t₁ t₂ = real.sqrt 6 / 2) :=
by sorry

end problem_statement_l141_141196


namespace min_omega_is_three_l141_141638

open Real

noncomputable def min_omega {ω : ℝ} {ϕ : ℝ} (f : ℝ → ℝ) : ℝ :=
  if f = (λ x, cos (ω * x + ϕ)) then
    if 0 < ω ∧ 0 < ϕ ∧ ϕ < π ∧
       (∃ (T : ℝ), T = 2 * π / ω ∧ f T = sqrt 3 / 2) ∧
       f (π / 9) = 0 then
      ω
    else 0
  else 0

theorem min_omega_is_three :
  let f : ℝ → ℝ := λ x, cos (3 * x + π / 6) in
  min_omega f = 3 := by
  sorry

end min_omega_is_three_l141_141638


namespace find_AB_l141_141209

-- Given conditions
variables {A B C : Type*} [metric_space A] [metric_space B] [metric_space C]
variables (BC AC AB : ℝ)
variables (angle_A : real.angle)
variables (tan_C cos_B : ℝ)

-- Assume the given values and properties
axiom angle_A_right : angle_A = 90
axiom BC_val : BC = 20
axiom tan_cos_relation : tan_C = 3 * cos_B
axiom tan_C_def : tan_C = AB / AC
axiom cos_B_def : cos_B = AB / BC

-- Prove the required length of AB
theorem find_AB : AB = (40 * real.sqrt 2) / 3 :=
by
  sorry

end find_AB_l141_141209


namespace cost_of_shirts_l141_141497

theorem cost_of_shirts : 
  let shirt1 := 15
  let shirt2 := 15
  let shirt3 := 15
  let shirt4 := 20
  let shirt5 := 20
  shirt1 + shirt2 + shirt3 + shirt4 + shirt5 = 85 := 
by
  sorry

end cost_of_shirts_l141_141497


namespace joint_savings_amount_l141_141601

def kimmie_earnings : ℝ := 1950
def zahra_earnings : ℝ := kimmie_earnings - (2/3) * kimmie_earnings
def layla_earnings : ℝ := (9/4) * kimmie_earnings

def kimmie_savings : ℝ := 0.35 * kimmie_earnings
def zahra_savings : ℝ := 0.40 * zahra_earnings
def layla_savings : ℝ := 0.30 * layla_earnings

def total_savings : ℝ := kimmie_savings + zahra_savings + layla_savings

theorem joint_savings_amount : total_savings = 2258.75 := 
by 
  sorry

end joint_savings_amount_l141_141601


namespace ascending_order_l141_141508

open Real

noncomputable def a := log 64 / log 16
noncomputable def b := log 0.2 / log 10
noncomputable def c := 2^(0.2)

theorem ascending_order : b < c ∧ c < a :=
by
  have h1 : a = 3 / 2 := by
    sorry
  
  have h2 : b < 0 := by
    sorry

  have h3 : 1 < c ∧ c < sqrt 2 := by
    sorry

  split
  case left =>
    have h4 : b < 1 := by
      sorry

    exact lt_of_lt_of_le h2 (le_of_lt h4)
  case right =>
    have h5 : 1 < c := h3.left
    have h6 : c < 3 / 2 := by
      sorry

    exact lt_trans h5 h6

end ascending_order_l141_141508


namespace min_omega_is_three_l141_141636

open Real

noncomputable def min_omega {ω : ℝ} {ϕ : ℝ} (f : ℝ → ℝ) : ℝ :=
  if f = (λ x, cos (ω * x + ϕ)) then
    if 0 < ω ∧ 0 < ϕ ∧ ϕ < π ∧
       (∃ (T : ℝ), T = 2 * π / ω ∧ f T = sqrt 3 / 2) ∧
       f (π / 9) = 0 then
      ω
    else 0
  else 0

theorem min_omega_is_three :
  let f : ℝ → ℝ := λ x, cos (3 * x + π / 6) in
  min_omega f = 3 := by
  sorry

end min_omega_is_three_l141_141636


namespace sequence_equiv_l141_141901

theorem sequence_equiv (n : ℕ) (hn : n > 0) : ∃ (p : ℕ), p > 0 ∧ (4 * p + 5 = (3^n)^2) :=
by
  sorry

end sequence_equiv_l141_141901


namespace correct_option_l141_141776

theorem correct_option :
  ¬(mode_of_data_4_4_6_7_9_6_is_4) ∧
  ¬(std_dev_squared_is_variance) ∧
  (std_dev_3_5_7_9_is_half_of_6_10_14_18) ∧
  (area_of_rectangles_equals_frequency) :=
by
  sorry

-- Definitions based on conditions in part a)

def mode_of_data_4_4_6_7_9_6_is_4 : Prop :=
  ∀ mode : ℕ, mode ∈ {4, 6}

def std_dev_squared_is_variance : Prop :=
  ∀ (X : Type) [has_variance X], 
    std_dev X = variance X

def std_dev_3_5_7_9_is_half_of_6_10_14_18 : Prop :=
  let data1 := [3, 5, 7, 9]
  let data2 := [6, 10, 14, 18]
  std_dev data1 = 0.5 * std_dev data2

def area_of_rectangles_equals_frequency : Prop :=
  ∀ (f : ℕ → ℝ) (freq_histogram : ℕ → ℝ),
    ∑ i, f i * freq_histogram i = 1

#check correct_option

end correct_option_l141_141776


namespace combined_area_of_two_circles_l141_141363

open Real

def distance (a b : ℝ × ℝ) : ℝ :=
  sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

def area_of_circle (radius : ℝ) : ℝ :=
  π * radius^2

theorem combined_area_of_two_circles :
  (distance (1, 2) (5, -2) = 4 * sqrt 2) →
  let radius1 := 4 * sqrt 2 in
  let radius2 := radius1 / 2 in
  area_of_circle radius1 + area_of_circle radius2 = 40 * π :=
by
  intros h_distance,
  let radius1 := 4 * sqrt 2,
  let radius2 := radius1 / 2,
  have area1 := area_of_circle radius1,
  have area2 := area_of_circle radius2,
  have h_area1 : area1 = 32 * π := by 
    rw [area_of_circle, mul_pow, sqrt_sq_eq_abs, abs_of_pos (mul_pos four_pos sqrt_two_pos)],
    linarith,
  have h_area2 : area2 = 8 * π := by 
    rw [area_of_circle, mul_pow, sqrt_sq_eq_abs, abs_of_pos (mul_pos two_pos sqrt_two_pos)],
    linarith,
  rw [h_area1, h_area2],
  linarith,
sorry

end combined_area_of_two_circles_l141_141363


namespace fourth_equation_l141_141677

theorem fourth_equation : 
  ∑ i in (finset.range 6), i^3 = (∑ i in (finset.range 6), i)^2 :=
by 
  sorry

end fourth_equation_l141_141677


namespace triangle_perimeter_l141_141921

theorem triangle_perimeter (a b c : ℝ) 
  (h1 : a = 3) 
  (h2 : b = 5) 
  (hc : c ^ 2 - 3 * c = c - 3) 
  (h3 : 3 + 3 > 5) 
  (h4 : 3 + 5 > 3) 
  (h5 : 5 + 3 > 3) : 
  a + b + c = 11 :=
by
  sorry

end triangle_perimeter_l141_141921


namespace water_pool_amount_is_34_l141_141453

noncomputable def water_in_pool_after_five_hours : ℕ :=
let water_first_hour := 8,
    water_next_two_hours := 2 * 10,
    water_fourth_hour := 14,
    total_water_added := water_first_hour + water_next_two_hours + water_fourth_hour,
    water_leak_fifth_hour := 8
in total_water_added - water_leak_fifth_hour

theorem water_pool_amount_is_34 : water_in_pool_after_five_hours = 34 := by
  sorry

end water_pool_amount_is_34_l141_141453


namespace largest_7_10_triple_example_7_10_triple_l141_141452

def is_7_10_triple (M : ℕ) : Prop :=
  let M_base7_digits := Int.digits 7 M
  let M_base10_from_base7 := Nat.ofDigits 10 M_base7_digits
  M_base10_from_base7 = 3 * M

theorem largest_7_10_triple : ∀ M : ℕ, is_7_10_triple M → M ≤ 1422 :=
by
  sorry

theorem example_7_10_triple : is_7_10_triple 1422 :=
by
  sorry

end largest_7_10_triple_example_7_10_triple_l141_141452


namespace positive_integer_divisibility_l141_141884

theorem positive_integer_divisibility (n : ℕ) (h : n > 0) : (n^2 + 1 ∣ n + 1) ↔ n = 1 := 
begin
  sorry
end

end positive_integer_divisibility_l141_141884


namespace minimum_omega_value_l141_141643

noncomputable def min_omega (ω T x φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π) (hT : T = 2 * π / ω) 
  (hfT : Real.cos (ω * T + φ) = √3 / 2) (hx : Real.cos (ω * x + φ) = 0) : ℝ :=
  3

theorem minimum_omega_value (ω T x φ : ℝ)
  (hω : ω > 0)
  (hφ : 0 < φ ∧ φ < π)
  (hT : T = 2 * π / ω)
  (hfT : Real.cos (ω * T + φ) = √3 / 2)
  (hx : Real.cos (ω * x + φ) = 0)
  (hx_val : x = π / 9)
  (T_val : T = 2 * π / ω) :
  ω = min_omega ω T (π / 9) (π / 6) :=
begin
  sorry -- Proof steps are skipped as instructed.
end

end minimum_omega_value_l141_141643


namespace area_triangle_equality_l141_141235

theorem area_triangle_equality
  (A B C D E F : ℝ)
  (is_rectangle : quadrilateral A B C D)
  (on_sides : E ∈ segment B C ∧ F ∈ segment C D)
  (is_equilateral : equilateral_triangle A E F) :
  area_triangle B C F = area_triangle A B E + area_triangle A F D :=
sorry

end area_triangle_equality_l141_141235


namespace find_roots_l141_141084

theorem find_roots (x : ℝ) :
  5 * x^4 - 28 * x^3 + 46 * x^2 - 28 * x + 5 = 0 → x = 3.2 ∨ x = 0.8 ∨ x = 1 :=
by
  intro h
  sorry

end find_roots_l141_141084


namespace probability_of_two_primes_l141_141352

-- Define the set of integers from 1 to 30
def finite_set : set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the set of prime numbers from 1 to 30
def primes_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Define the probability of choosing two different primes
def probability_two_primes : ℚ :=
  (fintype.card {p : ℕ × ℕ // p.1 < p.2 ∧ p.1 ∈ primes_set ∧ p.2 ∈ primes_set}) /
  (fintype.card {p : ℕ × ℕ // p.1 < p.2 ∧ p.1 ∈ finite_set ∧ p.2 ∈ finite_set})

-- Prove that the probability is 1/29
theorem probability_of_two_primes :
  probability_two_primes = 1 / 29 :=
sorry

end probability_of_two_primes_l141_141352


namespace number_of_integer_solutions_l141_141548

theorem number_of_integer_solutions :
  { x : ℤ | (x^2 + x - 3)^(x-1) = 1 }.to_finset.card = 3 := 
sorry

end number_of_integer_solutions_l141_141548


namespace one_cow_drink_time_l141_141988

theorem one_cow_drink_time (a b c : ℝ) (h1 : a + 3 * c = 3 * 17 * b) 
                           (h2 : a + 30 * c = 30 * 2 * b) : ∃ x : ℝ, a + x * c = x * b ∧ x = 75 :=
by
  -- Variables and conditions
  let eq1 := h1    -- a + 3c = 51b
  let eq2 := h2    -- a + 30c = 60b
  
  -- Solving the system of equations to find c
  have h_sub := eq2 - eq1, -- (a + 30c) - (a + 3c) = 60b - 51b
  have h_c : 27 * c = 9 * b := by { simp [h_sub], },
  have h_c_solved : c = 1/3 * b := by { field_simp, linarith, },
  
  -- Substituting back to find a
  have h_a := eq1.subst h_c_solved,
  have h_b : a = 50 * b := by { field_simp, linarith, },
  
  -- Setting up and solving the equation for one cow
  let x := 75,
  use x,
  split,
  { 
    -- Proof for a + x*c = x*b
    calc
      a + x * c = 50 * b + 75 * (1/3 * b) : by simp [h_b, h_c_solved]
             ... = 50 * b + 25 * b       : by { ring, }
             ... = 75 * b                : by { ring, }
             ... = x * b                 : by { simp, }
  },
  { 
    -- Proof for x = 75
    refl, 
  }

end one_cow_drink_time_l141_141988


namespace prime_divisibility_l141_141274

theorem prime_divisibility 
  (p a b c : ℤ) 
  (hp_prime : prime p)
  (h1 : 6 ∣ (p + 1))
  (h2 : p ∣ a + b + c)
  (h3 : p ∣ a^4 + b^4 + c^4) :
  p ∣ a ∧ p ∣ b ∧ p ∣ c :=
by
  sorry

end prime_divisibility_l141_141274


namespace milk_students_l141_141999

theorem milk_students (T : ℕ) (h1 : (1 / 4) * T = 80) : (3 / 4) * T = 240 := by
  sorry

end milk_students_l141_141999


namespace winner_votes_more_than_loser_l141_141199

-- Define the initial conditions.
def winner_percentage : ℝ := 0.55
def total_students : ℕ := 2000
def voting_percentage : ℝ := 0.25

-- Define the number of students who voted
def num_voted := total_students * (voting_percentage : ℕ)

-- Define the votes received by the winner and the loser,
-- based on the percentage of the total votes.
def winner_votes := num_voted * winner_percentage
def loser_votes := num_voted * (1 - winner_percentage)

-- Define the difference in votes between the winner and the loser.
def vote_difference := winner_votes - loser_votes

-- The target theorem to prove.
theorem winner_votes_more_than_loser : vote_difference = 50 := by
  sorry

end winner_votes_more_than_loser_l141_141199


namespace value_of_z_l141_141555

theorem value_of_z (x y z : ℝ) (h1 : x + y = 6) (h2 : z^2 = x * y - 9) : z = 0 :=
by
  sorry

end value_of_z_l141_141555


namespace calculate_possible_change_l141_141239

structure ChangeProblem where
  (change : ℕ)
  (h1 : change < 100)
  (h2 : ∃ (q : ℕ), change = 25 * q + 10 ∧ q ≤ 3)
  (h3 : ∃ (d : ℕ), change = 10 * d + 20 ∧ d ≤ 9)

theorem calculate_possible_change (p1 p2 p3 p4 : ChangeProblem) :
  p1.change + p2.change + p3.change = 180 :=
by
  sorry

end calculate_possible_change_l141_141239


namespace leo_weight_l141_141166

-- Definitions from the conditions
variable (L K J M : ℝ)

-- Conditions 
def condition1 : Prop := L + 15 = 1.60 * K
def condition2 : Prop := L + 15 = 0.40 * J
def condition3 : Prop := J = K + 25
def condition4 : Prop := M = K - 20
def condition5 : Prop := L + K + J + M = 350

-- Final statement to prove based on the conditions
theorem leo_weight (h1 : condition1 L K) (h2 : condition2 L J) (h3 : condition3 J K) 
                   (h4 : condition4 M K) (h5 : condition5 L K J M) : L = 110.22 :=
by 
  sorry

end leo_weight_l141_141166


namespace triangle_external_angle_bisector_l141_141180

theorem triangle_external_angle_bisector (A B C D : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (a b c d : Point A)
  (BC_len_eq : dist b c = 2 * dist b d)
  (angle_BAD_eq : dist a d = x)
  (angle_DAC_eq : dist d c = y)
  (angle_BCA_eq : dist b c = z) :
  x = z - y :=
sorry

end triangle_external_angle_bisector_l141_141180


namespace dog_food_consumption_per_meal_l141_141834

theorem dog_food_consumption_per_meal
  (dogs : ℕ) (meals_per_day : ℕ) (total_food_kg : ℕ) (days : ℕ)
  (h_dogs : dogs = 4) (h_meals_per_day : meals_per_day = 2)
  (h_total_food_kg : total_food_kg = 100) (h_days : days = 50) :
  (total_food_kg * 1000 / days / meals_per_day / dogs) = 250 :=
by
  sorry

end dog_food_consumption_per_meal_l141_141834


namespace remainder_40_l141_141413

-- Definitions from conditions
def number : ℕ := 220040
def a : ℕ := 555
def b : ℕ := 445
def sum_ab : ℕ := a + b        -- Sum of 555 and 445
def diff_ab : ℕ := a - b       -- Difference between 555 and 445
def quotient : ℕ := 2 * diff_ab -- Quotient is 2 times their difference

-- Theorem statement
theorem remainder_40 : number % sum_ab = 40 := by
  have h1 : sum_ab = 1000 := rfl
  have h2 : diff_ab = 110 := rfl
  have h3 : quotient = 220 := rfl
  have h4 : number % sum_ab = number - sum_ab * quotient := by sorry
  have h5 : number - sum_ab * quotient = 220040 - 1000 * 220 := rfl
  have h6 : 220040 - 1000 * 220 = 40 := by sorry
  rw [h4, h5, h6]
  exact rfl

end remainder_40_l141_141413


namespace math_problem_l141_141911

theorem math_problem
  (a b x y : ℝ)
  (h1 : a + b = 2)
  (h2 : x + y = 2)
  (h3 : ax + by = 5) :
  (a^2 + b^2) * x * y + a * b * (x^2 + y^2) = -5 :=
by
  sorry

end math_problem_l141_141911


namespace alpha_value_l141_141102

theorem alpha_value (α β : ℝ) (h1 : tan β = 1 / 2) (h2 : tan (α - β) = 1 / 3) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) : 
  α = π / 4 :=
by
  sorry

end alpha_value_l141_141102


namespace no_three_perfect_squares_l141_141667

theorem no_three_perfect_squares (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ¬(∃ k₁ k₂ k₃ : ℕ, k₁^2 = a^2 + b + c ∧ k₂^2 = b^2 + c + a ∧ k₃^2 = c^2 + a + b) :=
sorry

end no_three_perfect_squares_l141_141667


namespace sum_ge_sum_of_product_conditions_l141_141702

open Finset

variable {α : Type*} [LinearOrderedField α] 

theorem sum_ge_sum_of_product_conditions (n : ℕ)
  (a b : Fin n → α)
  (ha : ∀ i : Fin n, i.succ < n → a i ≥ a i.succ)
  (h₀ : ∀ i : Fin n, a i > 0)
  (hb1 : b 0 ≥ a 0)
  (hbp : ∀ k : Fin (n - 1), ∏ i in range (k + 1), b i ≥ ∏ i in range (k + 1), a i) :
  (∑ i in range n, b i) ≥ ∑ i in range n, a i := 
by
  sorry

end sum_ge_sum_of_product_conditions_l141_141702


namespace eight_pow_2012_mod_10_l141_141131

theorem eight_pow_2012_mod_10 : (8 ^ 2012) % 10 = 2 :=
by {
  sorry
}

end eight_pow_2012_mod_10_l141_141131


namespace math_problem_l141_141387

noncomputable def a : ℕ := 1265
noncomputable def b : ℕ := 168
noncomputable def c : ℕ := 21
noncomputable def d : ℕ := 6
noncomputable def e : ℕ := 3

theorem math_problem : 
  ( ( b / 100 : ℚ ) * (a ^ 2 / c) / (d - e ^ 2) : ℚ ) = -42646.27 :=
by sorry

end math_problem_l141_141387


namespace minimum_omega_l141_141622

noncomputable section

-- Define the function f and its properties
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.cos (ω * x + φ)

-- Assumptions based on the given conditions
variables {ω φ : ℝ}
variables (T : ℝ) (hω_pos : 0 < ω) (hφ_range : 0 < φ ∧ φ < π)
variables (hT : f T ω φ = Real.sqrt 3 / 2)
variables (hx_zero : f (π / 9) ω φ = 0)

-- Prove the minimum value of ω is 3
theorem minimum_omega : ω = 3 := sorry

end minimum_omega_l141_141622


namespace part_a_part_b_l141_141221

noncomputable def a : ℕ → ℝ
| 1       := 5
| (n + 1) := (a n ^ (n - 1) + 2^(n-1) + 2 * 3^(n-1))^(1 / (n + 1))

theorem part_a : ∀ n ≥ 1, a n = (2^n + 3^n)^(1 / n) := sorry

theorem part_b : ∀ n ≥ 1, a n > a (n + 1) := sorry

end part_a_part_b_l141_141221


namespace product_of_first_three_terms_l141_141714

/--
  The eighth term of an arithmetic sequence is 20.
  If the difference between two consecutive terms is 2,
  prove that the product of the first three terms of the sequence is 480.
-/
theorem product_of_first_three_terms (a d : ℕ) (h_d : d = 2) (h_eighth_term : a + 7 * d = 20) :
  (a * (a + d) * (a + 2 * d) = 480) :=
by
  sorry

end product_of_first_three_terms_l141_141714


namespace steiner_symmetrization_of_convex_is_convex_l141_141684

-- Define convex_polygon type to represent convex polygons
structure ConvexPolygon :=
  (vertices : Set Point) -- Assume we have a type Point
  (is_convex : ∀ A B ∈ vertices, ∀ P ∈ line_segment(A, B), P ∈ vertices)

-- Define Steiner symmetrization function
def steiner_symmetrization (M : ConvexPolygon) (l : Line) : ConvexPolygon := sorry

-- Proof statement
theorem steiner_symmetrization_of_convex_is_convex (M : ConvexPolygon) (l : Line) :
  let M' := steiner_symmetrization(M, l)
  ∀ A B ∈ M'.vertices, ∀ P ∈ line_segment(A, B), P ∈ M'.vertices :=
begin
  -- proof goes here
  sorry
end

end steiner_symmetrization_of_convex_is_convex_l141_141684


namespace minimum_omega_formula_l141_141651

noncomputable def minimumOmega (f : ℝ → ℝ) (x T ω φ : ℝ) : ℝ :=
  if (f = λ x, Real.cos (ω * x + φ)) 
     ∧ (ω > 0) 
     ∧ (0 < φ < π)
     ∧ (f T = √3 / 2)
     ∧ (f (π / 9) = 0) 
     ∧ (T = 2 * π / ω) 
  then 3 else 0

theorem minimum_omega_formula {f : ℝ → ℝ} {x T ω φ : ℝ} :
  (f = λ x, Real.cos (ω * x + φ)) 
  ∧ (ω > 0) 
  ∧ (0 < φ < π)
  ∧ (f T = √3 / 2)
  ∧ (f (π / 9) = 0)
  ∧ (T = 2 * π / ω) 
  → minimumOmega f x T ω φ = 3 :=
by
  sorry

end minimum_omega_formula_l141_141651


namespace maximum_moves_equiv_to_43690_l141_141406

-- Definitions directly from the conditions.
def card_state := list bool -- True represents black-side-up, False represents red-side-up

-- Initial sequence of cards as given in the problem
def initial_sequence : card_state := [true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false]

-- Function representing a valid move: flipping a consecutive sequence starting from a black card and followed by red cards.
def valid_move (seq : card_state) : card_state :=
  let n := seq.indexOf False -- Find the first occurrence of a red card
  let flipped := seq.take (n + 1) |>.map bnot -- Flip the sequence up to the first red card
  flipped ++ seq.drop (n + 1)

-- Function to count the maximum number of valid moves
def max_moves (init_seq : card_state) : ℕ :=
  let rec aux seq count :=
    if seq = [] then count
    else aux (valid_move seq) (count + 1)
  aux init_seq 0

-- The main theorem which states the maximum possible number of moves.
theorem maximum_moves_equiv_to_43690 : max_moves initial_sequence = 43690 := by
  sorry

end maximum_moves_equiv_to_43690_l141_141406


namespace integral_abs_sin_from_0_to_2pi_l141_141073

theorem integral_abs_sin_from_0_to_2pi : ∫ x in (0 : ℝ)..(2 * Real.pi), |Real.sin x| = 4 := 
by
  sorry

end integral_abs_sin_from_0_to_2pi_l141_141073


namespace only_set_c_forms_triangle_l141_141772

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem only_set_c_forms_triangle  : 
  ¬ satisfies_triangle_inequality 1 3 4 ∧
  ¬ satisfies_triangle_inequality 2 2 7 ∧
  satisfies_triangle_inequality 4 5 7 ∧
  ¬ satisfies_triangle_inequality 3 3 6 :=
by 
  unfold satisfies_triangle_inequality,
  -- Proof omitted
  sorry

end only_set_c_forms_triangle_l141_141772


namespace length_BE_of_equilateral_triangle_l141_141469

theorem length_BE_of_equilateral_triangle
  (A B C D E : Type)
  [has_dist A B C]
  [has_perpendicular A BC]
  [is_midpoint E AD]
  (side_length : ℝ)
  (hABC_eq : ∀ {x y z : Type}, equilateral x y z → ∀ s, has_dist x y = s ∧ has_dist y z = s ∧ has_dist z x = s)
  (h_perpendicular : has_perpendicular A BC )
  (h_midpoint_D : D = midpoint B C)
  (h_midpoint_E : E = midpoint A D)
  (side_length_value : side_length = 12) :
  has_dist B E = sqrt 63 :=
by
  sorry

end length_BE_of_equilateral_triangle_l141_141469


namespace eggs_remaining_l141_141029

noncomputable def eggsLeftOnShelf : ℕ := by
  let totalEggs := 6 * 12
  let usedEggs := totalEggs / 2
  let remainingEggs := usedEggs - 15
  exact remainingEggs

theorem eggs_remaining (totalEggs : ℕ) (usedEggs : ℕ) (brokenEggs : ℕ) :
  totalEggs = 6 * 12 →
  usedEggs = totalEggs / 2 →
  brokenEggs = 15 →
  remainingEggs = usedEggs - brokenEggs →
  remainingEggs = 21 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact eq.refl 21

end eggs_remaining_l141_141029


namespace problem1_problem2_problem3_l141_141948

-- Definitions and conditions
def a : ℕ → ℝ
| 0 := 1
| (n+1) := (2^(n+1) * a n) / (a n + 2^n)

def b (n : ℕ) : ℝ :=
- n * (n + 1) * a n

def T (n : ℕ) : ℝ :=
∑ i in (finset.range n), b i

-- Statement (1)
theorem problem1 : ∀ n : ℕ, (n > 0) → (2^n / a n - 2^(n-1) / a (n-1) = 1) := 
begin
  assume n hn,
  sorry
end

-- Statement (2)
theorem problem2 : ∀ n : ℕ, T n = (1 - n) * 2^(n+1) - 2 :=
begin
  assume n,
  sorry
end

-- Statement (3)
theorem problem3 : ∀ n m : ℕ, (n > 0) → T n + (n + m) * (n + 2) * a (n+1) < 0 → m ≤ -1 :=
begin
  assume n m hn hTn,
  sorry
end

end problem1_problem2_problem3_l141_141948


namespace diagonal_difference_is_six_l141_141059

def matrixOriginal : list (list ℕ) := 
  [[1, 2, 3, 4, 5],
  [10, 11, 12, 13, 14],
  [19, 20, 21, 22, 23],
  [28, 29, 30, 31, 32],
  [37, 38, 39, 40, 41]]

def matrixReversed : list (list ℕ) :=
  [[1, 2, 3, 4, 5],
  [14, 13, 12, 11, 10],
  [19, 20, 21, 22, 23],
  [28, 29, 30, 31, 32],
  [41, 40, 39, 38, 37]]

def mainDiagonalSum (matrix : list (list ℕ)) : ℕ :=
  matrix[0][0] + matrix[1][1] + matrix[2][2] + matrix[3][3] + matrix[4][4]

def antiDiagonalSum (matrix : list (list ℕ)) : ℕ :=
  matrix[0][4] + matrix[1][3] + matrix[2][2] + matrix[3][1] + matrix[4][0]

theorem diagonal_difference_is_six :
  abs ((antiDiagonalSum matrixReversed) - (mainDiagonalSum matrixReversed)) = 6 := by
  sorry

end diagonal_difference_is_six_l141_141059


namespace problem1_problem2_problem3_l141_141938

noncomputable def f (x a : ℝ) : ℝ := abs x * (x - a)

-- 1. Prove a = 0 if f(x) is odd
theorem problem1 (h: ∀ x : ℝ, f (-x) a = -f x a) : a = 0 :=
sorry

-- 2. Prove a ≤ 0 if f(x) is increasing on the interval [0, 2]
theorem problem2 (h: ∀ x y : ℝ, 0 ≤ x → x ≤ y → y ≤ 2 → f x a ≤ f y a) : a ≤ 0 :=
sorry

-- 3. Prove there exists an a < 0 such that the maximum value of f(x) on [-1, 1/2] is 2, and find a = -3
theorem problem3 (h: ∃ a : ℝ, a < 0 ∧ ∀ x : ℝ, -1 ≤ x → x ≤ 1/2 → f x a ≤ 2 ∧ ∃ x : ℝ, -1 ≤ x → x ≤ 1/2 → f x a = 2) : a = -3 :=
sorry

end problem1_problem2_problem3_l141_141938


namespace alexander_first_gallery_pictures_l141_141033

def pictures_for_new_galleries := 5 * 2
def pencils_for_new_galleries := pictures_for_new_galleries * 4
def total_exhibitions := 1 + 5
def pencils_for_signing := total_exhibitions * 2
def total_pencils := 88
def pencils_for_first_gallery := total_pencils - pencils_for_new_galleries - pencils_for_signing
def pictures_for_first_gallery := pencils_for_first_gallery / 4

theorem alexander_first_gallery_pictures : pictures_for_first_gallery = 9 :=
by
  sorry

end alexander_first_gallery_pictures_l141_141033


namespace minimum_omega_l141_141621

noncomputable section

-- Define the function f and its properties
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.cos (ω * x + φ)

-- Assumptions based on the given conditions
variables {ω φ : ℝ}
variables (T : ℝ) (hω_pos : 0 < ω) (hφ_range : 0 < φ ∧ φ < π)
variables (hT : f T ω φ = Real.sqrt 3 / 2)
variables (hx_zero : f (π / 9) ω φ = 0)

-- Prove the minimum value of ω is 3
theorem minimum_omega : ω = 3 := sorry

end minimum_omega_l141_141621


namespace find_angle_C_l141_141982

noncomputable def solve_triangle_ABC (a b c : ℝ) (angle_B : ℝ) : ℝ :=
if h : b > c then
  let sin_B := Real.sin angle_B,
      sin_C := (c * sin_B) / b in
  Real.arcsin sin_C
else
  0

theorem find_angle_C (a b c : ℝ) (h1 : c = Real.sqrt 2) (h2 : b = Real.sqrt 6) (h3 : Real.sin 120 = (Real.sqrt 3) / 2) : solve_triangle_ABC a b c 120 = 30 :=
by
  sorry

end find_angle_C_l141_141982


namespace parametric_line_angle_45_l141_141175

theorem parametric_line_angle_45 (t : ℝ) :
  let x := 1 + t,
      y := 2 + t,
      m := 1 in
  ∃ θ : ℝ, θ = 45 ∧ tan θ = m := 
by
  sorry

end parametric_line_angle_45_l141_141175


namespace effectiveness_A_effectiveness_comparison_B_l141_141759

variables (num_total : ℕ)
variables (num_vaccinated_A num_placebo_A infected_vaccinated_A infected_placebo_A : ℕ)
variables (infected_vaccinated_B infected_placebo_B : ℕ)

-- Conditions
def num_total_cond : Prop := num_total = 30000
def num_vaccinated_A_cond : Prop := num_vaccinated_A = num_total / 2
def num_placebo_A_cond : Prop := num_placebo_A = num_total / 2
def infected_vaccinated_A_cond : Prop := infected_vaccinated_A = 50
def infected_placebo_A_cond : Prop := infected_placebo_A = 500
def infected_placebo_B_cond : Prop := infected_placebo_B = 15000  -- Assuming given data for placebo group B

-- Effectiveness formula
def effectiveness (p q : ℚ) : ℚ := (1 - p / q) * 100

-- Calculate p and q for brand A
def p_A : ℚ := infected_vaccinated_A / num_vaccinated_A
def q_A : ℚ := infected_placebo_A / num_placebo_A

-- Statement 1: Vaccine effectiveness of brand A
theorem effectiveness_A : num_total_cond → num_vaccinated_A_cond → num_placebo_A_cond → infected_vaccinated_A_cond → infected_placebo_A_cond →
  effectiveness p_A q_A = 90 := by
  sorry

-- Calculate inequality for brand B
def p_B : ℚ := infected_vaccinated_B / num_vaccinated_A
def q_B : ℚ := infected_placebo_B / num_placebo_A

-- Define effectiveness condition for brand B
def effectiveness_condition_B : Prop := effectiveness p_B q_B > 90

-- Statement 2: Determine if infection number condition is necessary
theorem effectiveness_comparison_B : num_total_cond → num_vaccinated_A_cond → num_placebo_A_cond → infected_placebo_B_cond → 
  effectiveness_condition_B → 
  ¬(infected_vaccinated_B < infected_vaccinated_A) ↔ True := by
  sorry

end effectiveness_A_effectiveness_comparison_B_l141_141759


namespace prob_log_is_integer_l141_141098

/-- Given a set of numbers {2, 3, 8, 9}, if two distinct numbers are chosen and labeled as a and b, 
    the probability that log_a b is an integer is 1/6. -/
theorem prob_log_is_integer : 
  let S := {2, 3, 8, 9}
  ∧ (∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ log a b ∈ ℤ) → 
  (finset.card (({(2, 8), (3, 9)} : finset (ℕ × ℕ))) / finset.card (finset.powerset_len 2 S)) = 1 / 6 :=
sorry

end prob_log_is_integer_l141_141098


namespace calculation_result_l141_141785

theorem calculation_result:
  5 * 301 + 4 * 301 + 3 * 301 + 300 = 3912 :=
by
  sorry

end calculation_result_l141_141785


namespace sin_eq_neg_one_l141_141106

theorem sin_eq_neg_one
  (α : ℝ)
  (h : sin(α) * sin(π / 3 - α) = 3 * cos(α) * sin(α + π / 6)) :
  sin(2 * α + π / 6) = -1 :=
by sorry

end sin_eq_neg_one_l141_141106


namespace area_of_square_l141_141783

theorem area_of_square (p : ℝ) (h : p ≥ 0) :
  let perimeter := 12 * p in
  let side_length := perimeter / 4 in
  let area := side_length * side_length in
  area = 9 * p^2 :=
by
  sorry

end area_of_square_l141_141783


namespace midpoints_altitudes_collinear_iff_right_triangle_l141_141255

theorem midpoints_altitudes_collinear_iff_right_triangle
  (a b c : ℂ) (h_a_unit : abs a = 1) (h_b_unit : abs b = 1) (h_c_unit : abs c = 1) 
  (orthocenter : ℂ) (H_altitude : orthocenter = a + b + c - a * conj b * conj c) :
  ((midpoint_altitude a b c = midpoint_altitude b a c ∧ 
    midpoint_altitude b a c = midpoint_altitude c a b ∧ 
    midpoint_altitude c a b = midpoint_altitude a b c) ↔ 
   a + b = 0 ∨ b + c = 0 ∨ c + a = 0) :=
sorry

-- Helper function to calculate midpoint of altitudes
def midpoint_altitude (a b c : ℂ) : ℂ :=
(3 * a + b + c - b * conj c * conj a) / 4

end midpoints_altitudes_collinear_iff_right_triangle_l141_141255


namespace smallest_n_divisible_11_remainder_1_l141_141365

theorem smallest_n_divisible_11_remainder_1 :
  ∃ (n : ℕ), (n % 2 = 1) ∧ (n % 3 = 1) ∧ (n % 4 = 1) ∧ (n % 5 = 1) ∧ (n % 7 = 1) ∧ (n % 11 = 0) ∧ 
    (∀ m : ℕ, (m % 2 = 1) ∧ (m % 3 = 1) ∧ (m % 4 = 1) ∧ (m % 5 = 1) ∧ (m % 7 = 1) ∧ (m % 11 = 0) → 2521 ≤ m) :=
by
  sorry

end smallest_n_divisible_11_remainder_1_l141_141365


namespace sufficient_but_not_necessary_condition_l141_141524

variable {f : ℝ → ℝ}

def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem sufficient_but_not_necessary_condition (f_odd : is_odd f) (x1 x2 : ℝ) :
  (x1 + x2 = 0 → f(x1) + f(x2) = 0) ∧ ¬(f(x1) + f(x2) = 0 → x1 + x2 = 0) := by
  sorry

end sufficient_but_not_necessary_condition_l141_141524


namespace ticket_cost_l141_141051

noncomputable def adult_ticket_price : ℝ := 4.5
noncomputable def child_ticket_price : ℝ := adult_ticket_price / 3

theorem ticket_cost (adult_price child_price : ℝ) 
  (h1 : child_price = adult_price / 3) 
  (h2 : 3 * adult_price + 5 * child_price = 21) : 
  7 * adult_price + 4 * child_price = 37.5 :=
by
  rw [h1, ←cast_eq_of_eq],
  have h3 : adult_price = 4.5,
  { calc
      3 * adult_price + 5 * (adult_price / 3) = 21 : by simp [h1, h2]
      ... = 21 : by norm_num [adult_price] },
  rw [h3],
  calc
    7 * 4.5 + 4 * (4.5 / 3) = 7 * 4.5 + 4 * 1.5 : by simp
    ... = 31.5 + 6 : by norm_num
    ... = 37.5 : by norm_num

end ticket_cost_l141_141051


namespace find_value_of_y_l141_141968

theorem find_value_of_y (y : ℝ) (h : 9 / y^2 = y / 81) : y = 9 := 
by {
  sorry
}

end find_value_of_y_l141_141968


namespace transformation_correct_l141_141066

-- Define the transformations f and g
def f (m n : ℤ) : ℤ × ℤ := (m, -n)
def g (m n : ℤ) : ℤ × ℤ := (-m, -n)

-- State the theorem to be proved
theorem transformation_correct : g (f (-2, 3)).1 (f (-2, 3)).2 = (2, 3) :=
by
  -- Use the definitions of f and g to transform the coordinates
  have f_result := f (-2, 3)
  have g_result := g f_result.1 f_result.2
  show g_result = (2, 3)
  sorry

end transformation_correct_l141_141066


namespace largest_k_condition_l141_141488

theorem largest_k_condition (a : ℕ → ℝ) (h1 : ∀ n, 0 < a n)
(h2 : summable (λ n, a n)) :
  ∃ k : ℝ, (0 < k) ∧ (∀ b : ℕ → ℝ, ∑' n, b n * a n ≠ ∑' n, a n / (n : ℝ)^k) ↔ k = 1 / 2 :=
sorry

end largest_k_condition_l141_141488


namespace red_cards_in_B_eq_black_cards_in_C_l141_141984

theorem red_cards_in_B_eq_black_cards_in_C (A B C : Type) 
  (deck : set (fin 52)) (is_red : ∀ (card : fin 52), Prop) (is_black : ∀ (card : fin 52), Prop)
  (placed_in_A : fin 52 → bool) (placed_in_B : fin 52 → bool) (placed_in_C : fin 52 → bool) :
  (forall card in deck, (placed_in_A card = true → 
                        (is_red card ∧ placed_in_B card = true) ∨
                        (is_black card ∧ placed_in_C card = true))) →
  (forall card1 card2 in deck, (placed_in_A card1 = true ∧ placed_in_A card2 = true) →
                               (is_red card1 ∧ placed_in_B card2 = true) ∨
                               (is_black card1 ∧ placed_in_C card2 = true)) →
  (∑ card in deck, is_red card ∧ placed_in_B card = true) = 
  (∑ card in deck, is_black card ∧ placed_in_C card = true) :=
by
  sorry

end red_cards_in_B_eq_black_cards_in_C_l141_141984


namespace unique_ab_not_determined_l141_141229

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^2 + b * x - Real.sqrt 2

theorem unique_ab_not_determined :
  ∀ (a b : ℝ), a > 0 → b > 0 → 
  f a b (f a b (Real.sqrt 2)) = 1 → False := 
by
  sorry

end unique_ab_not_determined_l141_141229


namespace smallest_n_rotation_matrix_l141_141088

def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ √2/2, √2/2], ![-√2/2, √2/2]]

def I2 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], ![0, 1]]

theorem smallest_n_rotation_matrix :
  ∃ (n : ℕ), 0 < n ∧ (A^n = I2) ∧ (∀ m : ℕ, 0 < m ∧ (A^m = I2) → n ≤ m) ∧ n = 8 :=
  sorry

end smallest_n_rotation_matrix_l141_141088


namespace probability_both_numbers_prime_l141_141357

open Finset

def primes_between_1_and_30 : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_both_numbers_prime :
  (∃ (a b : ℕ), a ≠ b ∧ a ∈ primes_between_1_and_30 ∧ b ∈ primes_between_1_and_30 ∧
  (45 / 435) = (1 / 9)) :=
by
  have h_prime_count : ∃ (s : Finset ℕ), s.card = 10 ∧ s = primes_between_1_and_30 := sorry,
  have h_total_pairs : ∃ (n : ℕ), n = 435 := sorry,
  have h_prime_pairs : ∃ (m : ℕ), m = 45 := sorry,
  have h_fraction : (45 : ℚ) / 435 = (1 : ℚ) / 9 := sorry,
  exact ⟨45, 435, by simp [h_prime_pairs, h_total_pairs, h_fraction]⟩

end probability_both_numbers_prime_l141_141357


namespace complex_number_quadrant_l141_141537

def complex_quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "first"
  else if z.re < 0 ∧ z.im > 0 then "second"
  else if z.re < 0 ∧ z.im < 0 then "third"
  else if z.re > 0 ∧ z.im < 0 then "fourth"
  else "axis"

theorem complex_number_quadrant : 
  let z := (2 - I) / (1 + I)
  in complex_quadrant z = "fourth" :=
by
  let z := (2 - I) / (1 + I)
  sorry

end complex_number_quadrant_l141_141537


namespace brass_players_count_l141_141721

def marching_band_size : ℕ := 110
def woodwinds (b : ℕ) : ℕ := 2 * b
def percussion (w : ℕ) : ℕ := 4 * w
def total_members (b : ℕ) : ℕ := b + woodwinds b + percussion (woodwinds b)

theorem brass_players_count : ∃ b : ℕ, total_members b = marching_band_size ∧ b = 10 :=
by
  sorry

end brass_players_count_l141_141721


namespace range_of_a_for_monotonicity_l141_141975

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + 1 / x

def f_prime (a : ℝ) (x : ℝ) : ℝ :=
  a / x - 1 / (x^2)

theorem range_of_a_for_monotonicity :
  (∀ x > (1 / 2), f_prime a x ≥ 0) ↔ (a ∈ Set.Ici 2) := by
  sorry

end range_of_a_for_monotonicity_l141_141975


namespace compare_sqrt_fractions_l141_141459

theorem compare_sqrt_fractions : (sqrt 5 - 1) / 2 > 3 / 5 :=
by {
  sorry
}

end compare_sqrt_fractions_l141_141459


namespace wire_bent_square_area_l141_141819

noncomputable def area_of_square (r : ℝ) (π : ℝ) : ℝ :=
  let circumference := 2 * π * r
  let side_length := circumference / 4
  side_length ^ 2

theorem wire_bent_square_area : area_of_square 56 real.pi = 784 * real.pi := 
by
  sorry

end wire_bent_square_area_l141_141819


namespace triangle_area_l141_141077

theorem triangle_area {a b m : ℝ} (h1 : a = 27) (h2 : b = 29) (h3 : m = 26) : 
  ∃ (area : ℝ), area = 270 :=
by
  sorry

end triangle_area_l141_141077


namespace quadratic_inequality_solution_l141_141163

theorem quadratic_inequality_solution
  (x : ℝ)
  (h : x^2 - 5 * x + 6 < 0) :
  2 < x ∧ x < 3 :=
by sorry

end quadratic_inequality_solution_l141_141163


namespace brass_players_count_l141_141720

def marching_band_size : ℕ := 110
def woodwinds (b : ℕ) : ℕ := 2 * b
def percussion (w : ℕ) : ℕ := 4 * w
def total_members (b : ℕ) : ℕ := b + woodwinds b + percussion (woodwinds b)

theorem brass_players_count : ∃ b : ℕ, total_members b = marching_band_size ∧ b = 10 :=
by
  sorry

end brass_players_count_l141_141720


namespace exists_set_with_power_sum_form_l141_141857

theorem exists_set_with_power_sum_form :
  ∃ (M : set ℕ), M.card = 1992 ∧ 
  (∀ m ∈ M, ∃ k l : ℕ, l ≥ 2 ∧ m = k ^ l) ∧ 
  (∀ (S : finset ℕ), S ⊆ M → ∃ k l : ℕ, l ≥ 2 ∧ S.sum id = k ^ l) :=
sorry

end exists_set_with_power_sum_form_l141_141857


namespace minimum_omega_l141_141619

noncomputable section

-- Define the function f and its properties
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.cos (ω * x + φ)

-- Assumptions based on the given conditions
variables {ω φ : ℝ}
variables (T : ℝ) (hω_pos : 0 < ω) (hφ_range : 0 < φ ∧ φ < π)
variables (hT : f T ω φ = Real.sqrt 3 / 2)
variables (hx_zero : f (π / 9) ω φ = 0)

-- Prove the minimum value of ω is 3
theorem minimum_omega : ω = 3 := sorry

end minimum_omega_l141_141619


namespace min_omega_is_three_l141_141635

open Real

noncomputable def min_omega {ω : ℝ} {ϕ : ℝ} (f : ℝ → ℝ) : ℝ :=
  if f = (λ x, cos (ω * x + ϕ)) then
    if 0 < ω ∧ 0 < ϕ ∧ ϕ < π ∧
       (∃ (T : ℝ), T = 2 * π / ω ∧ f T = sqrt 3 / 2) ∧
       f (π / 9) = 0 then
      ω
    else 0
  else 0

theorem min_omega_is_three :
  let f : ℝ → ℝ := λ x, cos (3 * x + π / 6) in
  min_omega f = 3 := by
  sorry

end min_omega_is_three_l141_141635


namespace eggs_remaining_l141_141030

noncomputable def eggsLeftOnShelf : ℕ := by
  let totalEggs := 6 * 12
  let usedEggs := totalEggs / 2
  let remainingEggs := usedEggs - 15
  exact remainingEggs

theorem eggs_remaining (totalEggs : ℕ) (usedEggs : ℕ) (brokenEggs : ℕ) :
  totalEggs = 6 * 12 →
  usedEggs = totalEggs / 2 →
  brokenEggs = 15 →
  remainingEggs = usedEggs - brokenEggs →
  remainingEggs = 21 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact eq.refl 21

end eggs_remaining_l141_141030


namespace simplify_fraction_l141_141697

variable {F : Type*} [Field F]

theorem simplify_fraction (a b : F) (h: a ≠ -1) :
  b / (a * b + b) = 1 / (a + 1) :=
by
  sorry

end simplify_fraction_l141_141697


namespace adjacent_cells_sum_divisible_by_4_l141_141183

theorem adjacent_cells_sum_divisible_by_4 : 
    ∃ (i j : ℕ) (n1 n2 : ℕ), 
    (i < 2006) ∧ (j < 2006) ∧ 
    ((i < 2005 ∧ (n1 = (i * 2006 + j + 1) ∧ n2 = (i * 2006 + j + 2)) ∨
    (j < 2005 ∧ (n1 = (i * 2006 + j + 1) ∧ n2 = ((i + 1) * 2006 + j + 1)))) ∧ 
    (n1 + n2) % 4 = 0 :=
begin
    sorry
end

end adjacent_cells_sum_divisible_by_4_l141_141183


namespace minimum_omega_value_l141_141646

noncomputable def min_omega (ω T x φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π) (hT : T = 2 * π / ω) 
  (hfT : Real.cos (ω * T + φ) = √3 / 2) (hx : Real.cos (ω * x + φ) = 0) : ℝ :=
  3

theorem minimum_omega_value (ω T x φ : ℝ)
  (hω : ω > 0)
  (hφ : 0 < φ ∧ φ < π)
  (hT : T = 2 * π / ω)
  (hfT : Real.cos (ω * T + φ) = √3 / 2)
  (hx : Real.cos (ω * x + φ) = 0)
  (hx_val : x = π / 9)
  (T_val : T = 2 * π / ω) :
  ω = min_omega ω T (π / 9) (π / 6) :=
begin
  sorry -- Proof steps are skipped as instructed.
end

end minimum_omega_value_l141_141646


namespace volume_not_occupied_by_cones_l141_141752

noncomputable def radius := 14 -- base radius of the cones and cylinder, in cm
noncomputable def height_cone := 14 -- height of each cone, in cm
noncomputable def height_cylinder := 35 -- height of the cylinder, in cm

noncomputable def vol_cylinder := π * (radius ^ 2) * height_cylinder -- volume of the cylinder
noncomputable def vol_cone := (1 / 3:ℝ) * π * (radius ^ 2) * height_cone -- volume of one cone
noncomputable def vol_total_cones := 3 * vol_cone -- total volume of the three cones
noncomputable def vol_remaining := vol_cylinder - vol_total_cones -- volume of the cylinder not occupied by the cones

theorem volume_not_occupied_by_cones :
  vol_remaining = 4088 * π := 
by
  sorry

end volume_not_occupied_by_cones_l141_141752


namespace candy_bar_ratio_l141_141824

/-- Amanda had 7 candy bars initially and gave 3 to her sister. The next day, she bought 30 more candy bars and kept 22 candy bars in total.
Prove that the ratio of the number of candy bars Amanda gave her sister the second time to the number she gave the first time is 4:1. --/
theorem candy_bar_ratio :
  let initial := 7
  let given1 := 3
  let bought := 30
  let kept := 22
  let remaining := initial - given1
  let total := remaining + bought
  let given2 := total - kept
  (given2 / given1) = 4 :=
by 
  -- definitions from conditions mapped here for clarity
  let initial := 7
  let given1 := 3
  let bought := 30
  let kept := 22
  let remaining := initial - given1
  let total := remaining + bought
  let given2 := total - kept
  have h1 : remaining = initial - given1 by sorry
  have h2 : total = remaining + bought by sorry
  have h3 : given2 = total - kept by sorry
  have h4 : (given2 / given1) = 4 by sorry
  exact h4

end candy_bar_ratio_l141_141824


namespace projection_onto_plane_equals_l141_141082

-- Define the problem conditions
def plane_normal : ℝ × ℝ × ℝ := (6, -2, 8)
def vector_v : ℝ × ℝ × ℝ := (2, 3, 1)
def desired_projection : ℝ × ℝ × ℝ := (1.1923, 3.2692, -0.0769)

-- Define the theorem to prove the projection of vector_v onto the plane with normal vector plane_normal is as desire_projection.
theorem projection_onto_plane_equals : 
  let dot_prod (a b : ℝ × ℝ × ℝ) := a.1 * b.1 + a.2 * b.2 + a.3 * b.3 in
  let scalar_mult (c : ℝ) (v : ℝ × ℝ × ℝ) := (c * v.1, c * v.2, c * v.3) in
  let vector_sub (a b : ℝ × ℝ × ℝ) := (a.1 - b.1, a.2 - b.2, a.3 - b.3) in
  scalar_mult (dot_prod vector_v plane_normal / dot_prod plane_normal plane_normal) plane_normal = vector_sub vector_v desired_projection := 
by
  -- The proof is omitted for now.
  sorry

end projection_onto_plane_equals_l141_141082


namespace logarithm_identity_l141_141683

theorem logarithm_identity (b c : ℝ) (h1 : (2 * log 2 + log 3) / log 3 = b) (h2 : log 2 / log 3 = c) :
    3^b + 2 = 7 * 3^c := 
by
  sorry

end logarithm_identity_l141_141683


namespace alpha_solution_l141_141917

theorem alpha_solution
  (α : ℝ)
  (h1 : α ∈ Ioo 0 real.pi)
  (h2 : sin (α + real.pi / 6) ^ 2 + cos (α - real.pi / 3) ^ 2 = 3 / 2) :
  α = real.pi / 6 ∨ α = real.pi / 2 :=
by sorry

end alpha_solution_l141_141917


namespace larger_number_l141_141318

theorem larger_number (t a b : ℝ) (h1 : a + b = t) (h2 : a ^ 2 - b ^ 2 = 208) (ht : t = 104) :
  a = 53 :=
by
  sorry

end larger_number_l141_141318


namespace sector_circumradius_l141_141420

-- Given definitions and conditions
variable (θ : ℝ)

def radius_of_circle := 4 -- radius of the original circle
def is_obtuse (θ : ℝ) : Prop := θ > Mathlib.pi / 2 ∧ θ < Mathlib.pi

-- The statement to prove
theorem sector_circumradius (h : is_obtuse θ) : 
  let R := 2 * (1 / Real.sin(θ / 2)) in
  R = 2 * Real.csc(θ / 2) :=
by
  sorry

end sector_circumradius_l141_141420


namespace cone_volume_l141_141569

theorem cone_volume (lateral_area : ℝ) (angle : ℝ) 
  (h₀ : lateral_area = 20 * Real.pi)
  (h₁ : angle = Real.arccos (4/5)) : 
  (1/3) * Real.pi * (4^2) * 3 = 16 * Real.pi :=
by
  sorry

end cone_volume_l141_141569


namespace angle_AOB_is_90_degrees_l141_141945

-- Definitions for parabola and line
def parabola (M : ℝ × ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), M (x, y) ↔ y ^ 2 = 3 * x

def line (l : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ) (t : ℝ), l (x, y) ↔ x = t * y + 3

-- The main theorem
theorem angle_AOB_is_90_degrees (M : ℝ × ℝ → Prop) (l : ℝ → ℝ → Prop) (A B : ℝ × ℝ) :
  parabola M →
  line l →
  l 3 0 →
  ∃ x₁ y₁ x₂ y₂, M (x₁, y₁) ∧ M (x₂, y₂) ∧ l x₁ y₁ ∧ l x₂ y₂
  → ∠ (0, 0) A B = 90 :=
by
  -- Define points A and B as intersection points
  intro hM hl h_line
  cases' h_line with t ht
  apply Exists.intro ht
  apply Exists.intro ht
  apply Exists.intro ht
  apply Exists.intro ht
  sorry -- skipping the detailed proof steps as per instructions

end angle_AOB_is_90_degrees_l141_141945


namespace pages_read_l141_141600

theorem pages_read (a b c d e f : ℕ) (h1 : a = 10) (h2 : b = 20) (h3 : c = 25)
  (h4 : d = 30) (h5 : e = 35) (h6 : f = 40) (p1 p2 p3 p4 p5 p6: ℕ)
  (hb1 : p1 = 15) (hb2 : p2 = 18) (hb3 : p3 = 21) (hb4 : p5 = 24)
  (hb5 : p6 = 27) (hb7 : p7 = 30) :
  (a + b + c + d + e + f + hb1 + hb2 + hb3 + hb4 + hb5 + hb7 = 295) :=
by
  rw [h1, h2, h3, h4, h5, h6, hb1, hb2, hb3, hb4, hb5, hb7]
  exact rfl

end pages_read_l141_141600


namespace multiples_of_7_between_15_and_200_l141_141154

theorem multiples_of_7_between_15_and_200 : ∃ n : ℕ, n = 26 ∧ ∃ (a₁ a_n d : ℕ), 
  a₁ = 21 ∧ a_n = 196 ∧ d = 7 ∧ (a₁ + (n - 1) * d = a_n) := 
by
  sorry

end multiples_of_7_between_15_and_200_l141_141154


namespace area_of_plot_l141_141245

def central_square_area : ℕ := 64

def common_perimeter : ℕ := 32

-- This statement formalizes the proof problem: "The area of Mrs. Lígia's plot is 256 m² given the provided conditions."
theorem area_of_plot (a b : ℕ) 
  (h1 : a * a = central_square_area)
  (h2 : b = a) 
  (h3 : 4 * a = common_perimeter)  
  (h4 : ∀ (x y : ℕ), x + y = 16)
  (h5 : ∀ (x : ℕ), x + a = 16) 
  : a * 16 = 256 :=
sorry

end area_of_plot_l141_141245


namespace probability_sum_10_three_dice_l141_141176

theorem probability_sum_10_three_dice : 
  let die_faces := {1, 2, 3, 4, 5}
  let total_outcomes := 5^3
  let favorable_outcomes := 
    { (a, b, c) ∈ (die_faces × die_faces × die_faces) | a + b + c = 10 }.card
  let probability := favorable_outcomes / total_outcomes
  probability = 21 / 125 :=
by
  sorry

end probability_sum_10_three_dice_l141_141176


namespace at_least_one_miss_l141_141576

variables (p q : Prop)

-- Proposition stating the necessary and sufficient condition.
theorem at_least_one_miss : ¬(p ∧ q) ↔ (¬p ∨ ¬q) :=
by sorry

end at_least_one_miss_l141_141576


namespace root_interval_sum_l141_141974

theorem root_interval_sum (a b : Int) (h1 : b - a = 1) (h2 : ∃ x, a < x ∧ x < b ∧ (x^3 - x + 1) = 0) : a + b = -3 := 
sorry

end root_interval_sum_l141_141974


namespace vinnie_makes_more_l141_141680

-- Define the conditions
def paul_tips : ℕ := 14
def vinnie_tips : ℕ := 30

-- Define the theorem to prove
theorem vinnie_makes_more :
  vinnie_tips - paul_tips = 16 := by
  sorry

end vinnie_makes_more_l141_141680


namespace alpine_school_math_students_l141_141832

theorem alpine_school_math_students (total_players : ℕ) (physics_players : ℕ) (both_players : ℕ) :
  total_players = 15 → physics_players = 9 → both_players = 4 → 
  ∃ math_players : ℕ, math_players = total_players - (physics_players - both_players) + both_players := by
  sorry

end alpine_school_math_students_l141_141832


namespace tan_of_17pi_over_4_l141_141478

theorem tan_of_17pi_over_4 :
  tan (17 * Real.pi / 4) = 1 := by
  -- Conditions
  -- periodicity of tan function (not explicitly needed for the form of the statement)
  sorry

end tan_of_17pi_over_4_l141_141478


namespace lion_king_cost_l141_141704

theorem lion_king_cost
  (LK_earned : ℕ := 200) -- The Lion King earned 200 million
  (LK_profit : ℕ := 190) -- The Lion King profit calculated from half of Star Wars' profit
  (SW_cost : ℕ := 25)    -- Star Wars cost 25 million
  (SW_earned : ℕ := 405) -- Star Wars earned 405 million
  (SW_profit : SW_earned - SW_cost = 380) -- Star Wars profit
  (LK_profit_from_SW : LK_profit = 1/2 * (SW_earned - SW_cost)) -- The Lion King profit calculation
  (LK_cost : ℕ := LK_earned - LK_profit) -- The Lion King cost calculation
  : LK_cost = 10 := 
sorry

end lion_king_cost_l141_141704


namespace largest_log_iterations_l141_141464

noncomputable def U : ℕ → ℕ
| 0       := 3
| (n + 1) := 3 ^ U n

-- Define C to be (U 10) ^ (U 10)
def C : ℕ := U 10 ^ U 10

-- Define D to be (U 10) ^ C
def D : ℕ := U 10 ^ C

-- Define the statement that needs to be proved
theorem largest_log_iterations :
  (∃ k : ℕ, ∀ m < k, (nat.iterate (λ x => nat.log 3 x) m D > 1)) ∧
    ¬ (∃ k : ℕ, ∀ m < k + 1, (nat.iterate (λ x => nat.log 3 x) m D > 1)) :=
sorry

end largest_log_iterations_l141_141464


namespace tile_probability_l141_141755

-- Define the conditions and the target to be proven
theorem tile_probability:
  let A := set.range (λ n : ℕ, n + 1) ∩ (set.Ico 1 26) in
  let B := set.range (λ n : ℕ, n + 15) ∩ (set.Ico 15 45) in
  let prob_A := (∑ x in A, if x < 18 then 1 else 0) / card A.to_finset.card in
  let prime (n : ℕ) := n ≠ 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n in
  let prob_B := (∑ x in B, if prime x ∨ x > 40 then 1 else 0) / card B.to_finset.card in
  prob_A * prob_B = 17/75 := 
sorry

end tile_probability_l141_141755


namespace find_a_l141_141926

theorem find_a (a : ℝ) :
  (∀ x, y = ax^2) → 
  (∀ x, 2x - y - 6 = 0) → 
  (∀ x, (deriv (λ x, ax^2)) = 2a) → 
  (∀ x, deriv (λ x, ax^2) evaluated_at x = 2) → 
  a = 1 :=
by 
  intros h_curve h_line h_deriv1 h_deriv2
  sorry

end find_a_l141_141926


namespace seatingArrangements_l141_141876

def isValidArrangement (arrangement : List ℕ) : Prop :=
  ∃ count_matching_seats : ℕ, count_matching_seats = (List.filter (λ x, arrangement.get! x = x + 1) (List.range 5)).length
  ∧ count_matching_seats ≤ 2

def countValidArrangements : ℕ :=
  List.length (List.filter isValidArrangement (List.permutations (List.range 5)))

theorem seatingArrangements : countValidArrangements = 110 :=
by
  sorry

end seatingArrangements_l141_141876


namespace problem_solution_l141_141332

-- Conditions definitions
variables {A B C G P Q : Point}
variables (h1 : equilateral A B C) -- triangle ABC is equilateral
variables (h2 : side_length A B 6) -- side lengths AB = BC = CA = 6
variables (h3 : centroid G A B C) -- G is the centroid
variables (h4 : on_line_segment P A B) -- P lies on side AB
variables (h5 : on_line_segment Q A C) -- Q lies on side AC
variables (h6 : AP > AQ) -- with AP > AQ
variables (h7 : cyclic_quadrilateral A G P Q) -- AGPQ is cyclic
variables (h8 : area_triangle G P Q 6) -- area of triangle GPQ is 6

-- Proof statement
theorem problem_solution : ∃ a b c : ℕ, BQ_length B Q = a * sqrt b / c ∧ ¬ (∃ p, p^2 ∣ b) ∧ a + b + c = 5 :=
by sorry

end problem_solution_l141_141332


namespace expressions_equal_iff_l141_141970

theorem expressions_equal_iff (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_d : 1 < d) :
  (sqrt (a^2 + (b^d / c)) = a^d * sqrt (b / c)) ↔ (c = (a^(2*d) * b - b^d) / a^2) :=
begin
  sorry
end

end expressions_equal_iff_l141_141970


namespace jane_bought_no_bagels_l141_141468

noncomputable def total_cost (b m : ℕ) : ℕ :=
  55 * b + 80 * m

theorem jane_bought_no_bagels (b m : ℕ) :
  b + m = 6 →
  (∃ (k : ℕ), total_cost b m = 100 * k) →
  b = 0 :=
by
  intros h1 h2
  have h3 : m = 6 - b :=
    by linarith
  rw [h3] at h2
  simp at h2
  sorry

end jane_bought_no_bagels_l141_141468


namespace locus_traces_sphere_passing_through_O_l141_141231

noncomputable def locus_of_C (O X : Point) (R : ℝ) (sphere : Sphere) (plane : Plane) (C : Point) : Prop :=
  let CO := dist O C
      XO := dist X O
      AO := R
  in  CO * XO = R * R 

noncomputable def locus_of_points (O X : Point) (R : ℝ) (sphere : Sphere) (plane : Plane) : Set Point :=
  {C | ∃ (C' : Point), locus_of_C O X R sphere plane C' ∧ dist O C = dist O C'}

theorem locus_traces_sphere_passing_through_O (O X : Point) (R : ℝ) (sphere : Sphere) (plane : Plane) :
  ∀ X ∈ plane, ∃ C, C ∈ locus_of_points O X R sphere plane ∧ dist O C = R := 
sorry

end locus_traces_sphere_passing_through_O_l141_141231


namespace f_mono_increasing_f_range_l141_141137

noncomputable def f (x : ℝ) := 2 * cos x * (sqrt 3 * sin x - cos x) + 1

-- Prove the function is monotonically increasing in the interval [-π/6 + kπ, π/3 + kπ]
theorem f_mono_increasing (k : ℤ) :
  monotone_on f (Icc (-π / 6 + k * π) (π / 3 + k * π)) := sorry

-- Prove the range of the function in the interval [0, π/2] is [-1, 2]
theorem f_range (x : ℝ) (hx : 0 <= x ∧ x <= π / 2) :
  -1 <= f x ∧ f x <= 2 := sorry

end f_mono_increasing_f_range_l141_141137


namespace number_of_solutions_l141_141462

def g (x : ℝ) : ℝ :=
if -5 ≤ x ∧ x ≤ -1 then -x^2 + 1
else if -1 < x ∧ x < 2 then 2 * x + 3
else if 2 ≤ x ∧ x ≤ 5 then -x + 4
else 0  -- For conditions outside the given range.

theorem number_of_solutions : 
  { x : ℝ | -5 ≤ x ∧ x ≤ 5 ∧ g (g x) = 1 }.card = 2 :=
by sorry

end number_of_solutions_l141_141462


namespace no_such_n_l141_141499

-- Definition of P(n) as greatest prime factor of n
def P (n : ℕ) : ℕ := sorry 

-- Proposition stating that there are no positive integers n that satisfy the conditions
theorem no_such_n (n : ℕ) (h₁ : 1 < n) (h₂ : P(n) = √n) (h₃ : P(n + 60) = √(n + 60)) : false :=
by
    sorry

end no_such_n_l141_141499


namespace summation_equals_2529000_l141_141450

def summation_problem : ℕ :=
  (∑ i in Finset.range 150.succ, ∑ j in Finset.range 120.succ, (i + j + 5))

theorem summation_equals_2529000 : summation_problem = 2_529_000 := by
  sorry

end summation_equals_2529000_l141_141450


namespace equal_sum_of_red_and_blue_segments_l141_141784

theorem equal_sum_of_red_and_blue_segments
  (O1 O2 O3 : Point)
  (r : ℝ)
  (h1 : radius O1 = r)
  (h2 : radius O2 = r)
  (h3 : radius O3 = r)
  (tangent_segments_form_hexagon : tangent_segments_form_hexagon O1 O2 O3)
  (hexagon_has_alternating_colors : hexagon_has_alternating_colors O1 O2 O3):
  sum_of_red_segments O1 O2 O3 = sum_of_blue_segments O1 O2 O3 :=
by
  sorry

end equal_sum_of_red_and_blue_segments_l141_141784


namespace expression_is_perfect_square_l141_141855

-- Definitions for the expressions
def exprA := 3^6 * 7^7 * 8^8
def exprB := 3^8 * 7^6 * 8^7
def exprC := 3^7 * 7^8 * 8^6
def exprD := 3^7 * 7^7 * 8^8
def exprE := 3^8 * 7^8 * 8^8

-- The question translated to a Lean statement that asserts which expression is a perfect square
theorem expression_is_perfect_square : ∃ (e = exprE), ∀ (n : ℕ), ∃ (m : ℕ), e = m^2 := 
sorry

end expression_is_perfect_square_l141_141855


namespace segments_equal_if_parallel_l141_141991

variables {A B C D E F : ℝ^3}

-- Defining the conditions
def parallel (u v : ℝ^3) : Prop := ∃ k : ℝ, u = k • v

-- Assume the segments are defined by the points
def segment (P Q : ℝ^3) : ℝ^3 := Q - P

-- Main theorem statement
theorem segments_equal_if_parallel
  (distinct_points : A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ D ≠ E ∧ E ≠ F ∧ F ≠ D ∧ A ≠ D ∧ B ≠ E ∧ C ≠ F)
  (h1 : parallel (segment A B) (segment D E))
  (h2 : parallel (segment B C) (segment E F))
  (h3 : parallel (segment C D) (segment F A)) :
  (segment A B = segment D E) ∧
  (segment B C = segment E F) ∧
  (segment C D = segment F A) :=
by sorry

end segments_equal_if_parallel_l141_141991


namespace find_monotonic_interval_find_max_k_l141_141143

-- Given function and its derivative condition
def given_function (a b : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + b * x ^ 2 - 5 * x + (1 / 3)
def given_derivative (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x ^ 2 + 2 * b * x - 5

-- Conditions
def maximum_condition (a b : ℝ) : Prop := given_derivative a b (-1) = 0 ∧ given_function a b (-1) = 3

-- First question: Interval where the function is monotonically decreasing.
def monotonic_decrease_interval (a b : ℝ) : Set ℝ := {x | given_derivative a b x < 0}

-- Second question: Maximum k value condition
def max_k_condition (x k : ℝ) (a b : ℝ) : Prop := given_derivative a b x > k * (x * Real.log x - 2) - 5 * x - 9

-- Prove these statements
theorem find_monotonic_interval (a b : ℝ) (h : maximum_condition a b) : 
  monotonic_decrease_interval a b = Set.Ioo (-1 : ℝ) 5 := sorry

theorem find_max_k (a b : ℝ) (h : maximum_condition a b) : 
  ∃ (k_max : ℕ), ∀ (k : ℕ), k ≤ k_max ∧ ∀ (x : ℝ), x > 0 → max_k_condition x k a b := sorry

end find_monotonic_interval_find_max_k_l141_141143


namespace min_value_of_expression_l141_141530

-- Let a and b be positive real numbers
variables {a b : ℝ}
-- Condition: a > b
axiom h1 : a > b
-- Condition: ab = 1/2
axiom h2 : a * b = 1 / 2

-- Question: Prove the minimum value of the given expression is 2√5
theorem min_value_of_expression (h1 : a > b) (h2 : a * b = 1 / 2) : 
  ∃ m : ℝ, m = 2 * Real.sqrt 5 ∧ ∀ a b : ℝ, (h1 → h2 → (4 * a^2 + b^2 + 3) / (2 * a - b) ≥ m) :=
sorry

end min_value_of_expression_l141_141530


namespace mr_yadav_savings_l141_141244

theorem mr_yadav_savings
  (S : ℝ)  -- Mr. Yadav's monthly salary
  (h1 : 0.5 * 0.4 * S = 3900)  -- 50% of the remaining 40% salary is 3900
  (monthly_savings : ℝ := 0.2 * S):  -- definition of monthly savings
  S = 19500 →  -- inferred monthly salary
  (yearly_savings : ℝ := monthly_savings * 12):  -- definition of yearly savings
  yearly_savings = 46800 :=  -- proving statement
by
  sorry

end mr_yadav_savings_l141_141244


namespace possible_values_of_cubic_sum_l141_141219

theorem possible_values_of_cubic_sum (x y z : ℂ) (h1 : (Matrix.of ![
    ![x, y, z],
    ![y, z, x],
    ![z, x, y]
  ] ^ 2 = 3 • (1 : Matrix (Fin 3) (Fin 3) ℂ))) (h2 : x * y * z = -1) :
  x^3 + y^3 + z^3 = -3 + 3 * Real.sqrt 3 ∨ x^3 + y^3 + z^3 = -3 - 3 * Real.sqrt 3 := by
  sorry

end possible_values_of_cubic_sum_l141_141219


namespace prob_both_primes_l141_141343

-- Define the set of integers from 1 through 30
def int_set : Set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the set of prime numbers between 1 and 30
def primes_between_1_and_30 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Calculate the number of ways to choose two distinct elements from a set
noncomputable def combination (n k : ℕ) : ℕ := if k > n then 0 else n.choose k

-- Define the probabilities
noncomputable def prob_primes : ℚ :=
  (combination 10 2) / (combination 30 2)

-- State the theorem to prove
theorem prob_both_primes : prob_primes = 10 / 87 := by
  sorry

end prob_both_primes_l141_141343


namespace greatest_possible_x_l141_141781

theorem greatest_possible_x (x : ℤ) (h : 2.13 * 10 ^ x < 2100) : 
  x = 2 :=
by
  sorry

end greatest_possible_x_l141_141781


namespace range_of_a_l141_141942

def f(x : ℝ) : ℝ := x + 4 / x

def g(x a : ℝ) : ℝ := 2^x + a

theorem range_of_a (a : ℝ) :
  (∀ x1 ∈ (set.Icc (1/2 : ℝ) 3), ∃ x2 ∈ (set.Icc 2 3), f x1 ≥ g x2 a) ↔ a ≤ 0 :=
sorry

end range_of_a_l141_141942


namespace trig_identity_l141_141101

open Real

theorem trig_identity (α : ℝ) (h : tan α = -1/2) : 1 - sin (2 * α) = 9/5 := 
  sorry

end trig_identity_l141_141101


namespace find_a_2016_l141_141588

-- Define the sequence according to the given recurrence relation
def a : ℕ → ℚ
| 0 := -2
| (n + 1) := (1 + a n) / (1 - a n)

-- State the main theorem to be proved
theorem find_a_2016 : a 2015 = 3 := 
sorry

end find_a_2016_l141_141588


namespace proof_correct_choices_l141_141573

noncomputable def vector_a := (3 : ℝ, -1 : ℝ)
noncomputable def vector_b := (1 : ℝ, -1 : ℝ)
noncomputable def vector_c := (1 : ℝ, 2 : ℝ)

def statement_A : Prop :=
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 4)

def statement_B : Prop :=
  let diff := (vector_a.1 - 2 * vector_b.1, vector_a.2 - 2 * vector_b.2) in
  (real.sqrt (diff.1 ^ 2 + diff.2 ^ 2) = real.sqrt 2)

def statement_C : Prop :=
  let diff := (vector_a.1 - vector_c.1, vector_a.2 - vector_c.2) in
  (diff.1 * vector_b.2 = diff.2 * vector_b.1)

def statement_D : Prop :=
  let sum := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2) in
  (sum.1 * vector_c.1 + sum.2 * vector_c.2 = 0)

def correct_choices : Prop :=
  statement_A ∧ statement_B ∧ statement_D ∧ ¬statement_C

theorem proof_correct_choices : correct_choices :=
by
  sorry

end proof_correct_choices_l141_141573


namespace cube_root_of_64_l141_141279

theorem cube_root_of_64 :
  ∛(64) = 4 :=
by {
  have cond : 64 = 2 ^ 6 := by norm_num,
  rw [cond],
  rw [real.cbrt_pow 2, real.cbrt_nat_eq_pow],
  norm_num,
  sorry
}

end cube_root_of_64_l141_141279


namespace train_length_l141_141431

theorem train_length :
  (∃ (L : ℝ), (L / 30 = (L + 2500) / 120) ∧ L = 75000 / 90) :=
sorry

end train_length_l141_141431


namespace max_m_value_l141_141511

theorem max_m_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
                    (h_sum : a + b + c = 1) : 
  (sqrt (4 * a + 1) + sqrt (4 * b + 1) + sqrt (4 * c + 1)) ≥ 2 + sqrt 5 :=
by
  sorry

end max_m_value_l141_141511


namespace sum_tangent_products_l141_141448

theorem sum_tangent_products : 
  let tg := Real.tan 
  in (∑ k in Finset.range 2019, tg (k * Real.pi / 43) * tg ((k + 1) * Real.pi / 43)) = -2021 :=
sorry

end sum_tangent_products_l141_141448


namespace probability_both_numbers_prime_l141_141354

open Finset

def primes_between_1_and_30 : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_both_numbers_prime :
  (∃ (a b : ℕ), a ≠ b ∧ a ∈ primes_between_1_and_30 ∧ b ∈ primes_between_1_and_30 ∧
  (45 / 435) = (1 / 9)) :=
by
  have h_prime_count : ∃ (s : Finset ℕ), s.card = 10 ∧ s = primes_between_1_and_30 := sorry,
  have h_total_pairs : ∃ (n : ℕ), n = 435 := sorry,
  have h_prime_pairs : ∃ (m : ℕ), m = 45 := sorry,
  have h_fraction : (45 : ℚ) / 435 = (1 : ℚ) / 9 := sorry,
  exact ⟨45, 435, by simp [h_prime_pairs, h_total_pairs, h_fraction]⟩

end probability_both_numbers_prime_l141_141354


namespace find_alpha_beta_l141_141121

theorem find_alpha_beta (α β : ℝ) 
  (h1 : cos (π / 2 - α) = sqrt 2 * cos (3 * π / 2 + β)) 
  (h2 : sqrt 3 * sin (3 * π / 2 - α) = - sqrt 2 * sin (π / 2 + β)) 
  (h3 : 0 < α ∧ α < π)
  (h4 : 0 < β ∧ β < π) :
  α = 3 * π / 4 ∧ β = 5 * π / 6 := 
sorry

end find_alpha_beta_l141_141121


namespace num_integers_satisfying_conditions_l141_141547

theorem num_integers_satisfying_conditions : 
  ∃ n : ℕ, 
    (120 < n) ∧ (n < 250) ∧ (n % 5 = n % 7) :=
sorry

axiom num_integers_with_conditions : ℕ
@[simp] lemma val_num_integers_with_conditions : num_integers_with_conditions = 25 :=
sorry

end num_integers_satisfying_conditions_l141_141547


namespace spike_crickets_hunted_morning_l141_141700

def crickets_hunted_in_morning (C : ℕ) (total_daily_crickets : ℕ) : Prop :=
  4 * C = total_daily_crickets

theorem spike_crickets_hunted_morning (C : ℕ) (total_daily_crickets : ℕ) :
  total_daily_crickets = 20 → crickets_hunted_in_morning C total_daily_crickets → C = 5 :=
by
  intros h1 h2
  sorry

end spike_crickets_hunted_morning_l141_141700


namespace skips_per_meter_l141_141272

variable (a b c d e f g h : ℕ)

theorem skips_per_meter 
  (hops_skips : a * skips = b * hops)
  (jumps_hops : c * jumps = d * hops)
  (leaps_jumps : e * leaps = f * jumps)
  (leaps_meters : g * leaps = h * meters) :
  1 * skips = (g * b * f * d) / (a * e * h * c) * skips := 
sorry

end skips_per_meter_l141_141272


namespace find_k_for_slope_l141_141803

theorem find_k_for_slope :
  ∃ k : ℝ, (let x1 := -5
               y1 := k
               x2 := 13
               y2 := -7
               slope := -1 / 2 in
             (y2 - y1) / (x2 - x1) = slope) → k = 2 :=
sorry

end find_k_for_slope_l141_141803


namespace perimeter_of_square_is_64_l141_141019

noncomputable def side_length_of_square (s : ℝ) :=
  let rect_height := s
  let rect_width := s / 4
  let perimeter_of_rectangle := 2 * (rect_height + rect_width)
  perimeter_of_rectangle = 40

theorem perimeter_of_square_is_64 (s : ℝ) (h1 : side_length_of_square s) : 4 * s = 64 :=
by
  sorry

end perimeter_of_square_is_64_l141_141019


namespace M_lt_N_l141_141513

-- Define the arithmetic sequence {a_n} for non-zero common difference d
variable (a : ℕ → ℝ) (d : ℝ)
axiom d_nonzero : d ≠ 0

-- Define the terms M and N
def M (n : ℕ) : ℝ := a n * a (n + 3)
def N (n : ℕ) : ℝ := a (n + 1) * a (n + 2)

-- Lean statement to prove the relationship M < N
theorem M_lt_N (n : ℕ) : M a d n < N a d n :=
by {
  -- Proof is to be filled
  sorry
}

end M_lt_N_l141_141513


namespace minimum_omega_l141_141618

noncomputable section

-- Define the function f and its properties
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.cos (ω * x + φ)

-- Assumptions based on the given conditions
variables {ω φ : ℝ}
variables (T : ℝ) (hω_pos : 0 < ω) (hφ_range : 0 < φ ∧ φ < π)
variables (hT : f T ω φ = Real.sqrt 3 / 2)
variables (hx_zero : f (π / 9) ω φ = 0)

-- Prove the minimum value of ω is 3
theorem minimum_omega : ω = 3 := sorry

end minimum_omega_l141_141618


namespace mark_tylenol_intake_l141_141243

-- Define the conditions
def tablets_per_dose : ℕ := 2
def mg_per_tablet : ℕ := 500
def hours_per_dose : ℕ := 4
def total_hours : ℕ := 12

-- The mathematical equivalent proof problem
theorem mark_tylenol_intake :
  let doses := total_hours / hours_per_dose in
  let total_tablets := doses * tablets_per_dose in
  let total_mg := total_tablets * mg_per_tablet in
  let total_grams := total_mg / 1000 in
  total_grams = 3 :=
by
  sorry

end mark_tylenol_intake_l141_141243


namespace ticket_price_profit_condition_maximize_profit_at_7_point_5_l141_141688

-- Define the ticket price increase and the total profit function
def ticket_price (x : ℝ) := (10 + x) * (500 - 20 * x)

-- Prove that the function equals 6000 at x = 10 and x = 25
theorem ticket_price_profit_condition (x : ℝ) :
  ticket_price x = 6000 ↔ (x = 10 ∨ x = 25) :=
by sorry

-- Prove that m = 7.5 maximizes the profit
def profit (m : ℝ) := -20 * m^2 + 300 * m + 5000

theorem maximize_profit_at_7_point_5 (m : ℝ) :
  m = 7.5 ↔ (∀ m, profit 7.5 ≥ profit m) :=
by sorry

end ticket_price_profit_condition_maximize_profit_at_7_point_5_l141_141688


namespace partition_value_l141_141000

variable {a m n p x k l : ℝ}

theorem partition_value :
  (m * (a - n * x) = k * (a - n * x)) ∧
  (n * x = l * x) ∧
  (a - x = p * (a - m * (a - n * x)))
  → x = (a * (m * p - p + 1)) / (n * m * p + 1) :=
by
  sorry

end partition_value_l141_141000


namespace relay_race_last_year_distance_l141_141503

noncomputable def last_years_race_distance (d : ℕ) : Prop := 
  ∃ L : ℕ, ∃ N : ℕ,
    4 * L = N ∧
    N = 5 * d ∧
    L = 250

-- The distance between the tables.
def table_distance : ℕ := 200

-- The main theorem to prove.
theorem relay_race_last_year_distance : 
  (∃ d : ℕ,
    (d = 400 / 2) ∧
    last_years_race_distance d) := 
by
  use table_distance
  split
  -- Proof skipped.
  sorry

end relay_race_last_year_distance_l141_141503


namespace min_numbers_crossed_out_l141_141766

theorem min_numbers_crossed_out {S : set ℕ} (hS : S = {n | 1 ≤ n ∧ n ≤ 1982}) :
  ∃ N ⊆ S, (∀ a b c ∈ N, a * b ≠ c) ∧ (card (S \ N) = 43) :=
sorry

end min_numbers_crossed_out_l141_141766


namespace remove_blue_to_get_80_percent_red_l141_141045

-- Definitions from the conditions
def total_balls : ℕ := 150
def red_balls : ℕ := 60
def initial_blue_balls : ℕ := total_balls - red_balls
def desired_percentage_red : ℤ := 80

-- Lean statement of the proof problem
theorem remove_blue_to_get_80_percent_red :
  ∃ (x : ℕ), (x ≤ initial_blue_balls) ∧ (red_balls * 100 = desired_percentage_red * (total_balls - x)) → x = 75 := sorry

end remove_blue_to_get_80_percent_red_l141_141045


namespace minimum_omega_formula_l141_141652

noncomputable def minimumOmega (f : ℝ → ℝ) (x T ω φ : ℝ) : ℝ :=
  if (f = λ x, Real.cos (ω * x + φ)) 
     ∧ (ω > 0) 
     ∧ (0 < φ < π)
     ∧ (f T = √3 / 2)
     ∧ (f (π / 9) = 0) 
     ∧ (T = 2 * π / ω) 
  then 3 else 0

theorem minimum_omega_formula {f : ℝ → ℝ} {x T ω φ : ℝ} :
  (f = λ x, Real.cos (ω * x + φ)) 
  ∧ (ω > 0) 
  ∧ (0 < φ < π)
  ∧ (f T = √3 / 2)
  ∧ (f (π / 9) = 0)
  ∧ (T = 2 * π / ω) 
  → minimumOmega f x T ω φ = 3 :=
by
  sorry

end minimum_omega_formula_l141_141652


namespace smallest_number_among_neg2_neg1_0_pi_l141_141827

/-- The smallest number among -2, -1, 0, and π is -2. -/
theorem smallest_number_among_neg2_neg1_0_pi : min (min (min (-2 : ℝ) (-1)) 0) π = -2 := 
sorry

end smallest_number_among_neg2_neg1_0_pi_l141_141827


namespace range_of_r_l141_141093

def is_odd_composite (n : ℕ) : Prop :=
  ¬ Prime n ∧ ∃ d : ℕ, d ∣ n ∧ 1 < d ∧ d < n ∧ n % 2 = 1

def r (n : ℕ) : ℕ :=
  if is_odd_composite n then
    (Multiset.map (λ x : ℕ × ℕ => x.1 * x.2) (n.factors.multiplicity.prod)).sum
  else 0

theorem range_of_r (S : Set ℕ) :
  S = {m : ℕ | m > 5 ∧ m ≠ 7} ↔ (∀ n : ℕ, is_odd_composite n → r n ∈ S) :=
sorry

end range_of_r_l141_141093


namespace problem_induction_l141_141360

theorem problem_induction (n : ℕ) (hn : n > 0) :
  ∑ k in Finset.range n, 1 / ((2 * k + 1) * (2 * k + 3)) = n / (2 * n + 1) :=
sorry

end problem_induction_l141_141360


namespace min_omega_is_three_l141_141632

open Real

noncomputable def min_omega {ω : ℝ} {ϕ : ℝ} (f : ℝ → ℝ) : ℝ :=
  if f = (λ x, cos (ω * x + ϕ)) then
    if 0 < ω ∧ 0 < ϕ ∧ ϕ < π ∧
       (∃ (T : ℝ), T = 2 * π / ω ∧ f T = sqrt 3 / 2) ∧
       f (π / 9) = 0 then
      ω
    else 0
  else 0

theorem min_omega_is_three :
  let f : ℝ → ℝ := λ x, cos (3 * x + π / 6) in
  min_omega f = 3 := by
  sorry

end min_omega_is_three_l141_141632


namespace students_walk_fraction_l141_141833

theorem students_walk_fraction :
  (1 - (1/3 + 1/5 + 1/10 + 1/15)) = 3/10 :=
by sorry

end students_walk_fraction_l141_141833


namespace probability_of_two_primes_l141_141349

-- Define the set of integers from 1 to 30
def finite_set : set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the set of prime numbers from 1 to 30
def primes_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Define the probability of choosing two different primes
def probability_two_primes : ℚ :=
  (fintype.card {p : ℕ × ℕ // p.1 < p.2 ∧ p.1 ∈ primes_set ∧ p.2 ∈ primes_set}) /
  (fintype.card {p : ℕ × ℕ // p.1 < p.2 ∧ p.1 ∈ finite_set ∧ p.2 ∈ finite_set})

-- Prove that the probability is 1/29
theorem probability_of_two_primes :
  probability_two_primes = 1 / 29 :=
sorry

end probability_of_two_primes_l141_141349


namespace ellipse_properties_l141_141920

section
variable {a b c : ℝ}
variable {x y : ℝ}
variable {A : ℝ × ℝ}
variable {l : ℝ × ℝ → Prop}

-- Conditions
def is_eccentricity (e : ℝ) := e = (Real.sqrt (a^2 - b^2) / a)
def passes_through (A : ℝ × ℝ) := (A.1^2 / a^2) + (A.2^2 / b^2) = 1
def line_l (l : ℝ × ℝ → Prop) := ∀ x y, l (x, y) ↔ x - y + 2 = 0

-- Main statement
theorem ellipse_properties (h_eccentricity : is_eccentricity (Real.sqrt 3 / 2))
    (h_passes_through : passes_through (-2, 1))
    (h_line : line_l l) :
  (∃ a b,
    ((∀ x y, (x^2 / 8) + (y^2 / 2) = 1)) ∧
    let P := (@SetOf (ℝ) (λ x, l (x, ((x + 2))))) in
    let Q := (@SetOf (ℝ) (λ x, l (x, ((x + 2))))) in
    let PQ := (fun P Q => Real.dist P Q) in
    Real.dist (0, 0) (P ∪ Q) = Real.sqrt 2 ∧ ∃ area : ℝ, area = (4 * (Real.sqrt 6)) / 5 ) :=
sorry

end ellipse_properties_l141_141920


namespace interval_length_difference_l141_141286

-- Given conditions
def interval_length (x1 x2 : ℝ) : ℝ := x2 - x1

def y (x : ℝ) : ℝ := 4 ^ |x|

-- Given domain and range conditions for y(x)
variable (a b : ℝ)
axiom domain : ∀ x, a ≤ x ∧ x ≤ b → x ≥ -1 ∧ x ≤ 1
axiom range : ∀ x, a ≤ x ∧ x ≤ b → 1 ≤ y x ∧ y x ≤ 4

-- Question: Prove the difference between max and min lengths
theorem interval_length_difference : interval_length (-1) 1 - interval_length 0 1 = 1 := by
  sorry

end interval_length_difference_l141_141286


namespace num_subsets_with_even_is_24_l141_141949

def A : Set ℕ := {1, 2, 3, 4, 5}
def odd_subsets_count : ℕ := 2^3

theorem num_subsets_with_even_is_24 : 
  let total_subsets := 2^5
  total_subsets - odd_subsets_count = 24 := by
  sorry

end num_subsets_with_even_is_24_l141_141949


namespace constant_term_in_binomial_expansion_l141_141711

theorem constant_term_in_binomial_expansion :
  (binomial (6 : ℕ) 2 = 15) :=
by
  -- Given the conditions and the problem statement
  sorry

-- Definition of binomial coefficient (combinatorial number)
def binomial (n k : ℕ) : ℕ := nat.choose n k

end constant_term_in_binomial_expansion_l141_141711


namespace water_pool_amount_is_34_l141_141454

noncomputable def water_in_pool_after_five_hours : ℕ :=
let water_first_hour := 8,
    water_next_two_hours := 2 * 10,
    water_fourth_hour := 14,
    total_water_added := water_first_hour + water_next_two_hours + water_fourth_hour,
    water_leak_fifth_hour := 8
in total_water_added - water_leak_fifth_hour

theorem water_pool_amount_is_34 : water_in_pool_after_five_hours = 34 := by
  sorry

end water_pool_amount_is_34_l141_141454


namespace max_small_rectangles_l141_141956

theorem max_small_rectangles (w_large h_large w_small h_small real: ℝ) 
  (hW: w_large = 50) (hH: h_large = 90) (hws: w_small = 1) (hhs: h_small = 10 * real.sqrt 2) : 
  ⌊w_large * h_large / (w_small * h_small)⌋ = 318 :=
by
  sorry

end max_small_rectangles_l141_141956


namespace quadratic_inequality_l141_141908

theorem quadratic_inequality (a : ℝ) (h : 0 ≤ a ∧ a < 4) : ∀ x : ℝ, a * x^2 - a * x + 1 > 0 :=
by
  sorry

end quadratic_inequality_l141_141908


namespace part_I_part_II_l141_141144

open Real

def f (x : ℝ) := exp x - 2

theorem part_I (x : ℝ) (hx : 0 < x) : f(x) > x - 1 ∧ x - 1 ≥ log x :=
by 
  sorry

def g (x : ℝ) (m : ℤ) := exp x - log x - 2 - m

theorem part_II (m : ℤ) : (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ g x₁ m = 0 ∧ g x₂ m = 0) ↔ m = 1 :=
by 
  sorry

end part_I_part_II_l141_141144


namespace mt_mul_equals_neg4_l141_141662

noncomputable def g : ℝ → ℝ := sorry

axiom g_property : ∀ x y : ℝ, g (g x + y) = g x + g (g y + g (-x)) - x

theorem mt_mul_equals_neg4 : 
  let m := 1 in
  let t := g 4 in 
  m * t = -4 :=
by
  sorry

end mt_mul_equals_neg4_l141_141662


namespace AO_eq_OB_l141_141280

variable (A B C D O : Type)
variable [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup O]
variable [ConvexHull A] [ConvexHull B] [ConvexHull C] [ConvexHull D] [ConvexHull O]

axiom intersect_diagonals : ∃ O : A, ∃ A B C D : Type, ConvexHull (Set A) ∧ ConvexHull (Set B) 
  ∧ ConvexHull (Set C) ∧ ConvexHull (Set D) 

axiom perimeter_ABC_eq_ABD : ∃ (a b c d : Type), (a + b + c = a + d + b)

axiom perimeter_ACD_eq_BCD : ∃ (a b c d : Type), (a + c + d = b + d + c)

theorem AO_eq_OB :
  ∀ (A B C D O : Type) (a b c d : Type),
  ∃ O : A, ∃ A B C D : Type, ConvexHull (Set A) ∧ ConvexHull (Set B) 
  ∧ ConvexHull (Set C) ∧ ConvexHull (Set D) →
  a + b + c = a + d + b →
  a + c + d = b + d + c →
  O = B := 
sorry

end AO_eq_OB_l141_141280


namespace no_prime_triangle_sides_l141_141228

theorem no_prime_triangle_sides
  (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c)
  (x y z : ℕ) (hx : x = b + c - a) (hy : y = c + a - b) (hz : z = a + b - c)
  (hz2 : z^2 = y) (hx_sqrt : sqrt x - sqrt y = 2) :
  ¬ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
by
  sorry

end no_prime_triangle_sides_l141_141228


namespace lattice_points_count_l141_141801

-- Definition of a lattice point
structure LatticePoint where
  x : ℤ
  y : ℤ

-- Given endpoints of the line segment
def point1 : LatticePoint := ⟨5, 13⟩
def point2 : LatticePoint := ⟨38, 214⟩

-- Function to count lattice points on the line segment given the endpoints
def countLatticePoints (p1 p2 : LatticePoint) : ℕ := sorry

-- The proof statement
theorem lattice_points_count :
  countLatticePoints point1 point2 = 4 := sorry

end lattice_points_count_l141_141801


namespace margaret_time_hour_l141_141240

theorem margaret_time_hour (name_length : ℕ) (rearrangements_per_minute : ℕ) :
  name_length = 7 ∧ rearrangements_per_minute = 20 →
  (∑ i in finset.range 1 name_length.succ, i) / (60 * rearrangements_per_minute) = 4.2 :=
by
  sorry

end margaret_time_hour_l141_141240


namespace trisecting_accuracy_l141_141687

theorem trisecting_accuracy (r : ℝ) (O D E C A_1 B_1 A_2 B_2 A B T : ℝ) (h a b : ℝ) 
  (γ Θ δ : ℝ) 
  (angle_COD α β : ℝ) 
  (h_def : h = r * sin γ)
  (a_def : a = r + (2 * r * cos α))
  (b_def : b = r + (5 * r * cos β))
  (δ_def : δ = (2 * γ / 3) - Θ)
  (cos α_eq: cos α = (1 + 2 * cos γ) / √(5 + 4 * cos γ))
  (cos β_eq: cos β = (3 + 5 * cos γ) / √(34 + 30 * cos γ))
  (sin α_eq: sin α = (2 * sin γ) / √(5 + 4 * cos γ))
  (sin β_eq: sin β = (5 * sin γ) / √(34 + 30 * cos γ))
  ) :
  γ ≤ π / 2 → δ ≤ 0.0006 := 
by
  -- proof omitted
  sorry

end trisecting_accuracy_l141_141687


namespace angle_cosine_condition_l141_141589

-- Definitions for the triangle and angles
variables {A B : Real}

-- Conditions: ABC is a triangle with A > B
axiom triangle_ABC (A B : Real) : 0 < A ∧ A < π ∧ 0 < B ∧ B < π
axiom angle_relation (A B : Real) : A > B

-- Theorem statement: A > B is necessary and sufficient for cos A < cos B
theorem angle_cosine_condition (A B : Real) (h_triangle : triangle_ABC A B) (h_angle : angle_relation A B) : (A > B) ↔ (Real.cos A < Real.cos B) :=
sorry

end angle_cosine_condition_l141_141589


namespace minimum_omega_l141_141615

noncomputable section

-- Define the function f and its properties
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.cos (ω * x + φ)

-- Assumptions based on the given conditions
variables {ω φ : ℝ}
variables (T : ℝ) (hω_pos : 0 < ω) (hφ_range : 0 < φ ∧ φ < π)
variables (hT : f T ω φ = Real.sqrt 3 / 2)
variables (hx_zero : f (π / 9) ω φ = 0)

-- Prove the minimum value of ω is 3
theorem minimum_omega : ω = 3 := sorry

end minimum_omega_l141_141615


namespace max_possible_sum_l141_141048

/-- There are three intersecting circles creating seven regions. We need to fill the integers from
0 to 6 into these regions such that the sum of the four numbers in each circle is equal.
Prove that the maximum possible sum of the four numbers in each circle is 15. -/
theorem max_possible_sum : 
  ∃ (regions : Fin 7 → Fin 7) (circle_A circle_B circle_C : Fin 7 → Finset (Fin 7)), 
    (∀ i, circle_A i ∈ Finset.range 7) ∧ 
    (∀ i, circle_B i ∈ Finset.range 7) ∧ 
    (∀ i, circle_C i ∈ Finset.range 7) ∧
    (Finset.univ.sum (regions ∘ circle_A)) = 
    (Finset.univ.sum (regions ∘ circle_B)) ∧ 
    (Finset.univ.sum (regions ∘ circle_C)) ∧ 
    (Finset.univ.sum (regions ∘ circle_A)) = 15 :=
sorry

end max_possible_sum_l141_141048


namespace alisa_not_cleverest_inessa_cleverest_l141_141753

section foxes

-- Define the foxes.
inductive Fox : Type
| Alisa : Fox
| Larisa : Fox
| Inessa : Fox

open Fox

-- Predicate representing the cleverest fox.
def cleverest : Fox → Prop

-- Liars and truth-tellers based on who is the cleverest.
axiom Alisa_false : ¬cleverest Alisa → (¬ cleverer_than Alisa Larisa)
axiom Larisa_true : ¬cleverest Alisa → ¬cleverest Alisa
axiom Inessa_true : ¬cleverest Alisa → cleverer_than Alisa Inessa

-- Conditions
axiom cleverest_lies : ∀ f : Fox, cleverest f → liar f
axiom others_tell_truth : ∀ f : Fox, ¬ cleverest f → ¬ liar f

-- First part (a)
theorem alisa_not_cleverest : ¬ cleverest Alisa := sorry

-- Second part (b)
theorem inessa_cleverest : cleverest Inessa := sorry

end foxes

end alisa_not_cleverest_inessa_cleverest_l141_141753


namespace snail_distance_after_96_days_l141_141422

/-- On the n-th day, the snail moves forward by 1/n meters and backward by 1/(n+1) meters. -/
/-- We want to prove that the total distance the snail is from the starting point after 96 days is 96/97 meters. -/
theorem snail_distance_after_96_days :
  (∑ n in nat.range 96, (1 / (n + 1 : ℝ) - 1 / (n + 2 : ℝ))) = 96 / 97 := sorry

end snail_distance_after_96_days_l141_141422


namespace equal_charges_at_x_l141_141427

theorem equal_charges_at_x (x : ℝ) : 
  (2.75 * x + 125 = 1.50 * x + 140) → (x = 12) := 
by
  sorry

end equal_charges_at_x_l141_141427


namespace slices_of_pie_served_today_l141_141418

theorem slices_of_pie_served_today (lunch_slices : ℕ) (dinner_slices : ℕ) :
  lunch_slices = 7 → dinner_slices = 5 → lunch_slices + dinner_slices = 12 :=
by
  intros hLunch hDinner
  rw [hLunch, hDinner]
  rfl

end slices_of_pie_served_today_l141_141418


namespace ticks_to_burrs_ratio_l141_141046

theorem ticks_to_burrs_ratio 
  (number_of_burrs : ℕ) 
  (total_foreign_objects : ℕ) 
  (h : number_of_burrs = 12) 
  (h2 : total_foreign_objects = 84) : 
  let T := total_foreign_objects - number_of_burrs in 
  T / number_of_burrs = 6 := 
by
  sorry

end ticks_to_burrs_ratio_l141_141046


namespace same_function_set_C_l141_141825

-- Definitions of functions in Set C
def f_C (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 else -1

def g_C (x : ℝ) : ℝ :=
  if x ≠ 0 then x / abs x else 1

-- Theorem stating that f_C(x) and g_C(x) represent the same function
theorem same_function_set_C : ∀ (x : ℝ), f_C x = g_C x := 
  by
  intro x
  -- Here the proof would go
  sorry

end same_function_set_C_l141_141825


namespace ratio_addition_l141_141973

variable (a b c : ℝ)

-- Conditions
local notation "condition1" := b / a = 3
local notation "condition2" := c / b = 2

-- Theorem statement
theorem ratio_addition (h1 : condition1) (h2 : condition2) : (a + b) / (b + c) = 4 / 9 := by
  sorry

end ratio_addition_l141_141973


namespace difference_of_fractions_l141_141382

theorem difference_of_fractions (a : ℝ) (b : ℝ) (h₁ : a = 7000) (h₂ : b = 1/10) :
  (a * b - a * (0.1 / 100)) = 693 :=
by 
  sorry

end difference_of_fractions_l141_141382


namespace unique_solution_to_function_equation_l141_141076

theorem unique_solution_to_function_equation (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x + y) = x + f(f(y))) → (∀ x : ℝ, f(x) = x) :=
by
  sorry

end unique_solution_to_function_equation_l141_141076


namespace barbara_stuffed_animals_count_l141_141052

theorem barbara_stuffed_animals_count :
  ∃ B : ℕ, 
    (∃ T : ℕ, 
      T = 2 * B ∧ 
      2 * B + 1.5 * T = 45) ↔ B = 9 :=
by sorry

end barbara_stuffed_animals_count_l141_141052


namespace probability_two_primes_1_to_30_l141_141346

def is_prime (n : ℕ) : Prop := 
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13 ∨ n = 17 ∨ n = 19 ∨ n = 23 ∨ n = 29

def count_combinations (n k : ℕ) : ℕ := nat.choose n k

theorem probability_two_primes_1_to_30 :
  let total_combinations := count_combinations 30 2 in
  let prime_combinations := count_combinations 10 2 in
  total_combinations = 435 ∧ 
  prime_combinations = 45 ∧ 
  (prime_combinations : ℚ) / total_combinations = 15 / 145 :=
by { sorry }

end probability_two_primes_1_to_30_l141_141346


namespace equation_of_ellipse_maximum_area_l141_141203

-- Define the conditions
def eccentricity_of_ellipse : Prop :=
  ∃ (a b : ℝ), a > b ∧ b ≥ 1 ∧ ∀ x y, (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1 ∧ 
                (a^2 - b^2) / a^2 = 3/4 ∧ 
                ∃ Mx My, (Mx^2) / (a ^ 2) + (My^2) / (b^2) = 1 ∧ (sqrt ((Mx - 0)^2 + (My - 3)^2)) = 4

def point_A : Prop := ∃ (A : ℝ × ℝ), A = (0, 1 / 16)

def point_N_on_parabola (t : ℝ) : Prop := ∃ (N : ℝ × ℝ), N = (t, t^2)

def tangent_intersects_ellipse (t : ℝ) : Prop :=
  let N := (t, t^2) in
  ∃ (B C : ℝ × ℝ),
    ∃ x_c : ℝ,
      ∃ y_c : ℝ, 
        B ≠ C ∧ 
        (B.2 - N.2 = 2 * t * (B.1 - t)) ∧ 
        (C.2 - N.2 = 2 * t * (C.1 - t)) ∧
        (x_c * x_c) / 4 + y_c * y_c = 1 ∧ 
        sqrt (1 + 4 * t^2) * sqrt ((4 * t ^ 4 - 4) / (1 + 16 * t ^ 2) ≥ 
        4 * sqrt(16 * t^2 + 64 - (t ^ 4 - 16 * t ^ 2 - 1)) / (1 + 16 * t ^ 2) ≥ 0

theorem equation_of_ellipse : eccentricity_of_ellipse → 
  ∀ (a b : ℝ), (a > b ∧ b ≥ 1 ∧ ∀ x y, (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1 ∧ 
                      (a^2 - b^2) / a^2 = 3/4) → 
                (b^2 = 1 ∧ a^2 = 4) → (∀ x y, (x^2 / 4 + y^2 = 1)) := 
begin
  sorry
end

theorem maximum_area :
  ∀ (t : ℝ), (point_A ∧ point_N_on_parabola t ∧ tangent_intersects_ellipse t) → 
    ∃ A B C : ℝ × ℝ, 
      let area := (4 * sqrt(1 + 4 * t ^ 2) * sqrt(-t ^ 4 + 16 * t ^ 2 + 1) / (1 + 16 * t ^ 2)) / 2 in
        area = sqrt(65) / 8  :=
begin
  sorry
end

end equation_of_ellipse_maximum_area_l141_141203


namespace triangle_area_partition_l141_141251

variables {A B C A1 O : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
variables {p a b c : ℝ}

def is_bisector (A B C : Type) (A1 : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
(h : MetricSpace.dist A A1 = (b + c - a) / 2) : Prop :=
  ∃ (l_a : Line), l_a.perpendicular_to (angle_bisector A B C) ∧ l_a.contains A1

theorem triangle_area_partition
  {A B C A1 : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
  {p a b c : ℝ} (h_bis : is_bisector A B C A1 h)
  (h_lines : ∃ (l_b l_c : Line), l_b.perpendicular_to (angle_bisector B A C) ∧ l_c.perpendicular_to (angle_bisector C A B))
  (h_division : divides_triangle A B C l_a l_b l_c) :
  ∃ Δ1 Δ2 Δ3 Δ4: Triangle, Δ1.area + Δ2.area + Δ3.area = Δ4) :=
sorry

end triangle_area_partition_l141_141251


namespace number_of_ways_to_write_52_as_sum_of_two_primes_l141_141580

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_two_primes (n : ℕ) : ℕ :=
  (Finset.filter (λ p, is_prime p ∧ is_prime (n - p)) (Finset.range n)).card / 2

theorem number_of_ways_to_write_52_as_sum_of_two_primes : sum_of_two_primes 52 = 3 :=
by
  sorry

end number_of_ways_to_write_52_as_sum_of_two_primes_l141_141580


namespace line_x_intercept_l141_141805

theorem line_x_intercept {x1 y1 x2 y2 : ℝ} (h : (x1, y1) = (4, 6)) (h2 : (x2, y2) = (8, 2)) :
  ∃ x : ℝ, (y1 - y2) / (x1 - x2) * x + 6 - ((y1 - y2) / (x1 - x2)) * 4 = 0 ∧ x = 10 :=
by
  sorry

end line_x_intercept_l141_141805


namespace evaluate_expression_l141_141449

theorem evaluate_expression : (120 / 6 * 2 / 3 = (40 / 3)) := 
by sorry

end evaluate_expression_l141_141449


namespace find_value_of_y_l141_141967

theorem find_value_of_y (y : ℝ) (h : 9 / y^2 = y / 81) : y = 9 := 
by {
  sorry
}

end find_value_of_y_l141_141967


namespace apples_per_box_l141_141746

theorem apples_per_box (crates_apples : ℕ) (num_crates : ℕ) (rotten_apples : ℕ) (num_boxes : ℕ) 
  (h_crates_apples : crates_apples = 42) 
  (h_num_crates : num_crates = 12)
  (h_rotten_apples : rotten_apples = 4)
  (h_num_boxes : num_boxes = 50) :
  (num_crates * crates_apples - rotten_apples) / num_boxes = 10 := 
by
  sorry

end apples_per_box_l141_141746


namespace weekly_car_mileage_l141_141323

-- Definitions of the conditions
def dist_school := 2.5 
def dist_market := 2 
def school_days := 4
def school_trips_per_day := 2
def market_trips_per_week := 1

-- Proof statement
theorem weekly_car_mileage : 
  4 * 2 * (2.5 * 2) + (1 * (2 * 2)) = 44 :=
by
  -- The goal is to prove that 4 days of 2 round trips to school plus 1 round trip to market equals 44 miles
  sorry

end weekly_car_mileage_l141_141323


namespace part1_solution_part2_solution_l141_141233

open Finset

-- Defining the problem for Part (1)
noncomputable def problem_part1 (points : Finset ℕ) (segments : Finset (ℕ × ℕ)) : Prop :=
  let four_points : Finset (Finset ℕ) := powersetLen 4 points
  ∃ four_point_set ∈ four_points, ∀ {a b : ℕ}, 
    a ∈ four_point_set → b ∈ four_point_set → a ≠ b → (a, b) ∈ segments ∨ (b, a) ∈ segments

-- Defining the problem for Part (2)
noncomputable def problem_part2 (points : Finset ℕ) (segments : Finset (ℕ × ℕ)) : Prop :=
  let four_points : Finset (Finset ℕ) := powersetLen 4 points
  ¬ ∃ four_point_set ∈ four_points, ∀ {a b : ℕ}, 
    a ∈ four_point_set → b ∈ four_point_set → a ≠ b → (a, b) ∈ segments ∨ (b, a) ∈ segments

-- Main theorem statements for each part
theorem part1_solution : problem_part1 (range 6) (range 6.succ.product (range 6.succ)).erase (0, 1) :=
sorry

theorem part2_solution : ∃ segments, card segments = 12 ∧ problem_part2 (range 6) segments :=
sorry

end part1_solution_part2_solution_l141_141233


namespace tank_capacity_l141_141005

/--
A leak in the bottom of a tank can empty the full tank in 8 hours.
An inlet pipe fills water at the rate of 6 litres a minute.
When the tank is full, the inlet is opened, and due to the leak, the tank is empty in 12 hours.
Prove that the capacity of the tank is 1728 litres.
-/
theorem tank_capacity : 
  ∃ (C : ℕ), 
    (∀ (t : ℕ), (∃ (L : ℕ), L = C / 8 ∧ t = 8) ∧ 
                 (∃ (I : ℕ), I = 6 * 60 ∧ t = 1) ∧ 
                 (∃ (N : ℕ), N = C / 12 ∧ t = 12) → 
                 360 - C / 8 = C / 12) ∧ 
    C = 1728 :=
begin
  sorry
end

end tank_capacity_l141_141005


namespace part_i_l141_141220

def T : ℕ → ℕ 
| (2 * k)     := k
| (2 * k + 1) := 2 * k + 2

def T_iter : ℕ → ℕ → ℕ
| 0     n := n
| (k+1) n := T_iter k (T n)

theorem part_i (n : ℕ) : ∃ k : ℕ, T_iter k n = 1 := 
sorry

end part_i_l141_141220


namespace circle_standard_equation_l141_141736

theorem circle_standard_equation (x y : ℝ) (center : ℝ × ℝ) (radius : ℝ) 
  (h_center : center = (2, -1)) (h_radius : radius = 2) :
  (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2 ↔ (x - 2) ^ 2 + (y + 1) ^ 2 = 4 := by
  sorry

end circle_standard_equation_l141_141736


namespace tiling_impossible_with_one_2x2_tile_lost_l141_141277

variable {Box : Type}
variable [bottom_of_box : Bottom Box]
variable [two_by_two_tile : TwoByTwo Tile]
variable [one_by_four_tile : OneByFour Tile]

/- 
  We assume the bottom of a box is initially tiled with 2x2 and 1x4 tiles, 
  and one 2x2 tile is lost and one 1x4 tile is retrieved as a replacement.
-/
def impossible_to_tile (box : Box) : Prop :=
  ∀ arrangement : Arrangement,
  (arrangement.is_tiled_with two_by_two_tile one_by_four_tile) → 
  ∃ lost_tile : TwoByTwo Tile, ∃ retrieved_tile : OneByFour Tile,
    (arrangement.lost_tile = lost_tile) ∧ 
    (arrangement.retrieved_tile = retrieved_tile) →
    false

theorem tiling_impossible_with_one_2x2_tile_lost 
  (box : Box) (arrangement : Arrangement) 
  (lost_tile : TwoByTwo Tile) (retrieved_tile : OneByFour Tile) :
  arrangement.is_tiled_with two_by_two_tile one_by_four_tile →
  arrangement.lost_tile = lost_tile →
  arrangement.retrieved_tile = retrieved_tile →
  impossible_to_tile box :=
by 
  sorry

end tiling_impossible_with_one_2x2_tile_lost_l141_141277


namespace only_set_c_forms_triangle_l141_141773

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem only_set_c_forms_triangle  : 
  ¬ satisfies_triangle_inequality 1 3 4 ∧
  ¬ satisfies_triangle_inequality 2 2 7 ∧
  satisfies_triangle_inequality 4 5 7 ∧
  ¬ satisfies_triangle_inequality 3 3 6 :=
by 
  unfold satisfies_triangle_inequality,
  -- Proof omitted
  sorry

end only_set_c_forms_triangle_l141_141773


namespace b_parallel_c_projection_c_on_a_is_correct_l141_141506

-- Definitions of the given vectors
def a : ℝ × ℝ × ℝ := (1, 0, 1)
def b : ℝ × ℝ × ℝ := (-1, 2, -3)
def c : ℝ × ℝ × ℝ := (2, -4, 6)

-- Definition of dot product for 3D vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Prove that vectors b and c are parallel
theorem b_parallel_c : ∃ k : ℝ, c = (k * b.1, k * b.2, k * b.3) :=
  sorry

-- Definition of projection of c onto a
def projection (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dp := dot_product u v
  let u_norm_sq := dot_product u u
  let scale := dp / u_norm_sq
  (scale * u.1, scale * u.2, scale * u.3)

-- Prove that the projection of c onto a is (4, 0, 4)
theorem projection_c_on_a_is_correct : projection c a = (4, (0:ℝ), 4) :=
  sorry

end b_parallel_c_projection_c_on_a_is_correct_l141_141506


namespace goods_train_pass_time_l141_141009

def speed_mans_train_kmph : ℝ := 40
def speed_goods_train_kmph : ℝ := 72
def length_goods_train_m : ℝ := 280
def relative_speed_mps (speed1_kmph speed2_kmph : ℝ) : ℝ :=
  (speed1_kmph + speed2_kmph) * (1000 / 3600)

theorem goods_train_pass_time :
  let speed_mans_train_mps := speed_mans_train_kmph * (1000 / 3600),
      speed_goods_train_mps := speed_goods_train_kmph * (1000 / 3600),
      relative_speed := relative_speed_mps speed_mans_train_kmph speed_goods_train_kmph in
  length_goods_train_m / relative_speed ≈ 9 :=
by
  sorry

end goods_train_pass_time_l141_141009


namespace num_not_nice_l141_141501

open Nat

def is_nice (N k : ℕ) : Prop :=
  ∃ a : ℕ, a > 0 ∧ (∃ p : ℕ → Prop, (∀ i, p i → i ∈ finset.range a) ∧ (∏ i in finset.range a, (k * i + 1) = N))

lemma three_nice (N : ℕ) : is_nice N 3 ↔ N % 3 = 1 := 
by sorry

lemma five_nice (N : ℕ) : is_nice N 5 ↔ N % 5 = 1 := 
by sorry

theorem num_not_nice (n : ℕ) (h₁ : n < 500) : 
  266 = (finset.range n).filter (λ x, ¬(is_nice x 3 ∨ is_nice x 5)).card := 
by sorry

end num_not_nice_l141_141501


namespace eight_sided_die_red_faces_l141_141844

theorem eight_sided_die_red_faces :
  let faces := Finset.range 8
  let pairs := faces.product faces
  let valid_pairs := pairs.filter (λ p, (p.1 ≠ p.2) ∧ (p.1 + p.2 ≠ 6) ∧ (¬ odd (p.1 + p.2)))

  valid_pairs.card = 13 := sorry

end eight_sided_die_red_faces_l141_141844


namespace non_empty_prime_subsets_count_l141_141155

-- Definition of the set S
def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Definition of primes in S
def prime_subset_S : Set ℕ := {x ∈ S | Nat.Prime x}

-- The statement to prove
theorem non_empty_prime_subsets_count : 
  ∃ n, n = 15 ∧ ∀ T ⊆ prime_subset_S, T ≠ ∅ → ∃ m, n = 2^m - 1 := 
by
  sorry

end non_empty_prime_subsets_count_l141_141155


namespace halfway_between_one_third_and_one_eighth_l141_141081

theorem halfway_between_one_third_and_one_eighth : (1/3 + 1/8) / 2 = 11 / 48 :=
by
  -- The proof goes here
  sorry

end halfway_between_one_third_and_one_eighth_l141_141081


namespace find_white_cars_l141_141672

-- Defining the conditions given in the problem
def rent_white_per_minute : ℝ := 2 
def rent_red_per_minute : ℝ := 3
def red_cars : ℕ := 3
def rental_time_minutes : ℕ := 180
def total_earnings : ℝ := 2340

-- Defining the number of white cars available to rent
noncomputable def white_cars_available (W : ℝ) :=
  rent_red_per_minute * red_cars * rental_time_minutes + W * rent_white_per_minute * rental_time_minutes = total_earnings

-- The theorem we aim to prove
theorem find_white_cars : ∃ W : ℝ, white_cars_available W ∧ W = 2 :=
begin
  use 2,
  unfold white_cars_available,
  simp [rent_red_per_minute, red_cars, rental_time_minutes, rent_white_per_minute, total_earnings],
  norm_num,
end

end find_white_cars_l141_141672


namespace imaginary_part_condition_l141_141563

theorem imaginary_part_condition (z : ℂ) (h : z * complex.I = 2 + z) : z.im = -1 := 
sorry

end imaginary_part_condition_l141_141563


namespace time_difference_l141_141215

-- Define the conditions
def time_to_nile_delta : Nat := 4
def number_of_alligators : Nat := 7
def combined_walking_time : Nat := 46

-- Define the mathematical statement we want to prove
theorem time_difference (x : Nat) :
  4 + 7 * (time_to_nile_delta + x) = combined_walking_time → x = 2 :=
by
  sorry

end time_difference_l141_141215


namespace projection_matrix_correct_l141_141225

open Matrix

-- Define the vector v0
def v0 : Vector 2 := ![v₀1, v₀2] -- Assume values v₀1, v₀2

-- Define the vector v1: projection of v0 onto (2, 2)
def proj_v1 := (1 / 2) • ((2 + 2)⁻¹) • (Matrix.vec![2, 2])

-- Define the projection matrix onto (1, 0)
def projMatrix_1_0 := λ (M : Matrix (Fin 2) (Fin 2) ℝ), (Matrix.vec![![1, 0], ![0, 0]])

-- Define the projection matrix onto (1/sqrt(2), 1/sqrt(2))
def projMatrix_2_2 := λ (M : Matrix (Fin 2) (Fin 2) ℝ), (Matrix.vec![![0.5, 0.5], ![0.5, 0.5]])

-- The matrix that takes v0 to v2 is the product of these two projection matrices
def transformationMatrix := projMatrix_1_0 ⬝ projMatrix_2_2

-- Prove the matrix correct transformation
theorem projection_matrix_correct :
  transformationMatrix = Matrix.vec![![1 / 2, 1 / 2], ![0, 0]] :=
  by
    sorry

end projection_matrix_correct_l141_141225


namespace sequence_ratio_proof_l141_141147

variable {a : ℕ → ℤ}

-- Sequence definition
axiom a₁ : a 1 = 3
axiom a_recurrence : ∀ n : ℕ, a (n + 1) = 4 * a n + 3

-- The theorem to be proved
theorem sequence_ratio_proof (n : ℕ) : (a (n + 1) + 1) / (a n + 1) = 4 := by
  sorry

end sequence_ratio_proof_l141_141147


namespace determine_m_n_l141_141118

theorem determine_m_n 
  {a b c d m n : ℕ} 
  (h₁ : a + b + c + d = m^2)
  (h₂ : a^2 + b^2 + c^2 + d^2 = 1989)
  (h₃ : max (max a b) (max c d) = n^2) 
  : m = 9 ∧ n = 6 := by 
  sorry

end determine_m_n_l141_141118


namespace relationship_among_a_b_c_l141_141612

noncomputable def a : ℝ := Real.pi ^ 0.3
noncomputable def b : ℝ := Real.log 3 / Real.log Real.pi -- This uses the change of base formula
noncomputable def c : ℝ := 1 -- from 3°, but we know 3° = 1 in this context (as it's equivalent to one full turn in modular arithmetic)

theorem relationship_among_a_b_c : a > c ∧ c > b := by
  sorry

end relationship_among_a_b_c_l141_141612


namespace union_sets_l141_141148

def A : Set ℝ := { x | (x - 1) / (x + 3) > 0 }
def B : Set ℝ := { y | ∃ x : ℝ, y = Real.sqrt (4 - x^2) }

theorem union_sets :
  A ∪ B = { x | x ∈ set.Iio (-3) ∪ set.Ici (0) } :=
by sorry

end union_sets_l141_141148


namespace maximal_k_of_2012_gon_l141_141417

theorem maximal_k_of_2012_gon :
  ∃ (k : ℕ), (k ≤ 2012) ∧ (∀ (s : finset ℕ), 
  (s ⊆ finset.range 2012) → 
  (k = s.card) → 
  ∃ (a b c d : ℕ), ((a ∈ s) ∧ (b ∈ s) ∧ (c ∈ s) ∧ (d ∈ s) ∧ 
  (a + b ≡ c + d [MOD 2012]) ∧ (a ≠ c ∧ b ≠ d)) ∧ 
  (∀ (x1 x2 x3 x4 : ℕ), (x1 ∈ s) ∧ (x2 ∈ s) ∧ (x3 ∈ s) ∧ (x4 ∈ s) → 
  ((x1 + x2 ≡ x3 + x4 [MOD 2012]) → ((x1 = x3 ∧ x2 = x4) ∨ (x1 = x4 ∧ x2 = x3))))) := 
  1509 := 
begin
  sorry
end

end maximal_k_of_2012_gon_l141_141417


namespace choir_arrangement_l141_141792

/-- There are 4 possible row-lengths for arranging 90 choir members such that each row has the same
number of individuals and the number of members per row is between 6 and 15. -/
theorem choir_arrangement (x : ℕ) (h : 6 ≤ x ∧ x ≤ 15 ∧ 90 % x = 0) :
  x = 6 ∨ x = 9 ∨ x = 10 ∨ x = 15 :=
by
  sorry

end choir_arrangement_l141_141792


namespace range_of_a_l141_141083

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → |2 - x| + |x + 1| ≤ a) ↔ a ≥ 9 :=
by
  sorry

end range_of_a_l141_141083


namespace number_of_candies_picked_up_l141_141963

-- Definitions of the conditions
def num_sides_decagon := 10
def diagonals_from_one_vertex (n : Nat) : Nat := n - 3

-- The theorem stating the number of candies Hyeonsu picked up
theorem number_of_candies_picked_up : diagonals_from_one_vertex num_sides_decagon = 7 := by
  sorry

end number_of_candies_picked_up_l141_141963


namespace area_relation_for_square_or_rectangle_area_relation_for_rectangle_l141_141362

open EuclideanGeometry

-- Define a square (or rectangle) inscribed in a circle
def square_inscribed_circle (A B C D P : Point) : Prop :=
  square A B C D ∧ PointOnCircle P (circumcircle A B C D)

-- Define the triangles formed by point P and the vertices of the square (or rectangle)
def triangles_areas (A B C D P : Point) : (ℝ × ℝ × ℝ × ℝ) :=
  (area (triangle P A B), area (triangle P B C), area (triangle P C D), area (triangle P D A))

-- Define the largest triangle area
def largest_triangle_area (A B C D P : Point) : ℝ :=
  max (max (area (triangle P A B)) (area (triangle P B C))) (max (area (triangle P C D)) (area (triangle P D A)))

-- Define the sum of the areas of the other three triangles 
def sum_of_other_triangles_areas (A B C D P : Point) : ℝ :=
  let (a1, a2, a3, a4) := triangles_areas A B C D P in a1 + a2 + a3 + a4 - largest_triangle_area A B C D P

-- The theorem to be proved
theorem area_relation_for_square_or_rectangle (A B C D P : Point) :
  square_inscribed_circle A B C D P → 
  largest_triangle_area A B C D P = sum_of_other_triangles_areas A B C D P :=
by 
  intros h, sorry

-- The assertion is that the relation holds true for a square and a rectangle
theorem area_relation_for_rectangle (A B C D P : Point) :
  rectangle_inscribed_circle A B C D P → 
  largest_triangle_area A B C D P = sum_of_other_triangles_areas A B C D P :=
by 
  intros h, sorry

end area_relation_for_square_or_rectangle_area_relation_for_rectangle_l141_141362


namespace simplify_expr1_simplify_expr2_l141_141698

variable (a b m n : ℝ)

theorem simplify_expr1 : 2 * a - 6 * b - 3 * a + 9 * b = -a + 3 * b := by
  sorry

theorem simplify_expr2 : 2 * (3 * m^2 - m * n) - m * n + m^2 = 7 * m^2 - 3 * m * n := by
  sorry

end simplify_expr1_simplify_expr2_l141_141698


namespace maximum_value_of_k_l141_141557

theorem maximum_value_of_k (x y k : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < k)
    (h4 : 4 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) : k ≤ 1.5 :=
by
  sorry

end maximum_value_of_k_l141_141557


namespace riku_stickers_l141_141695

theorem riku_stickers (kristoff_stickers : ℕ) (riku_multiplier : ℕ) (h_kristoff : kristoff_stickers = 85) (h_riku_multiplier : riku_multiplier = 25) : 
    (25 * 85 = 2125) :=
by
    have h_riku_stickers : riku_multiplier * kristoff_stickers = 2125 := by sorry
    exact h_riku_stickers

end riku_stickers_l141_141695


namespace incorrect_statement_l141_141063

-- Define the conditions as statements
def axiom_true : Prop := 
  ∀ (A : Prop), ¬ (∃ proof : A, proof)

def true_from_false_premises : Prop := 
  ∀ (P Q : Prop), (¬P → Q) → False

def definitions_needed : Prop := 
  ∀ (term : String), ¬ (∅ ∋ term)

def multiple_valid_proofs : Prop :=
  ∃ (T : Prop), ∀ (method : T → Prop), method T

-- Identify which statement is false
theorem incorrect_statement : 
  ¬ true_from_false_premises := 
  by 
    sorry

end incorrect_statement_l141_141063


namespace tan_alpha_plus_pi_div_four_l141_141151

theorem tan_alpha_plus_pi_div_four
  (α : ℝ)
  (a : ℝ × ℝ := (3, 4))
  (b : ℝ × ℝ := (Real.sin α, Real.cos α))
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) :
  Real.tan (α + Real.pi / 4) = 7 := by
  sorry

end tan_alpha_plus_pi_div_four_l141_141151


namespace three_circles_one_common_point_three_circles_two_common_points_three_circles_no_three_common_points_three_circles_no_four_common_points_l141_141838

theorem three_circles_one_common_point :
  ∃ (C1 C2 C3 : set (ℝ × ℝ)), (∃ p : ℝ × ℝ, p ∈ C1 ∧ p ∈ C2 ∧ p ∈ C3) :=
sorry

theorem three_circles_two_common_points :
  ∃ (C1 C2 C3 : set (ℝ × ℝ)), ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ (p1 ∈ C1 ∧ p1 ∈ C2 ∧ p1 ∈ C3) ∧ (p2 ∈ C1 ∧ p2 ∈ C2 ∧ p2 ∈ C3) :=
sorry

theorem three_circles_no_three_common_points :
  ∀ (C1 C2 C3 : set (ℝ × ℝ)), ¬ (∃ p1 p2 p3 : ℝ × ℝ, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ (p1 ∈ C1 ∧ p1 ∈ C2 ∧ p1 ∈ C3) ∧ (p2 ∈ C1 ∧ p2 ∈ C2 ∧ p2 ∈ C3) ∧ (p3 ∈ C1 ∧ p3 ∈ C2 ∧ p3 ∈ C3)) :=
sorry

theorem three_circles_no_four_common_points :
  ∀ (C1 C2 C3 : set (ℝ × ℝ)), ¬ (∃ p1 p2 p3 p4 : ℝ × ℝ, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p4 ∧ (p1 ∈ C1 ∧ p1 ∈ C2 ∧ p1 ∈ C3) ∧ (p2 ∈ C1 ∧ p2 ∈ C2 ∧ p2 ∈ C3) ∧ (p3 ∈ C1 ∧ p3 ∈ C2 ∧ p3 ∈ C3) ∧ (p4 ∈ C1 ∧ p4 ∈ C2 ∧ p4 ∈ C3)) :=
sorry

end three_circles_one_common_point_three_circles_two_common_points_three_circles_no_three_common_points_three_circles_no_four_common_points_l141_141838


namespace triangle_ABC_proof_l141_141213

variable {A B C a b c : ℝ}
variable (angle_A_pos : 0 < A) (angle_A_lt_pi : A < π)
variable (angle_B_pos : 0 < B) (angle_B_lt_pi : B < π)
variable (angle_C_pos : 0 < C) (angle_C_lt_pi : C < π)

theorem triangle_ABC_proof
  (h1 : 2 * sin B * sin C + cos B + 2 * cos (B + C) = 0)
  (h2 : sin B ≠ 1)
  (h3 : 5 * sin B = 3 * sin A)
  (h4 : 1/2 * a * b * sin C = 15 * sqrt 3 / 4)
  (h5 : a = 5) (h6 : b = 3) : 
  C = 2 * Real.pi / 3 ∧ a + b + c = 15 :=
by
  sorry

end triangle_ABC_proof_l141_141213


namespace correct_propositions_l141_141928

-- Given propositions
def proposition1 : Prop := ∃ (points : Set Point), points.card = 3 → plane.determined_by points
def proposition2 : Prop := ∃ (a b c d : Point), is_trapezoid a b c d ∧ (determines_plane a b c d)
def proposition3 : Prop := 
  ∀ (l1 l2 l3 : Line), 
    pairwise_intersect l1 l2 l3 → 
    (∃ (planes : Set Plane), planes.card ≤ 3 ∧ determined_by l1 l2 l3 planes)
def proposition4 : Prop := 
  ∀ (points : Set Point) (plane1 plane2 : Plane), 
    points.card = 3 ∧ points ⊆ plane1 ∧ points ⊆ plane2 → plane1 = plane2

-- Definitions related to geometric concepts
variable {Point : Type}
variable {Plane : Type}
variable {Line : Type}

-- Determine if a set of points determine a plane
def plane.determined_by (points : Set Point) : Prop := sorry

-- Determine if four points form a trapezoid
def is_trapezoid (a b c d : Point) : Prop := sorry

-- Determine if four points determine a plane
def determines_plane (a b c d : Point) : Prop := sorry

-- Check if lines intersect pairwise
def pairwise_intersect (l1 l2 l3 : Line) : Prop := sorry

-- Determine if lines determine a set of planes
def determined_by (l1 l2 l3 : Line) (planes : Set Plane) : Prop := sorry

-- The main statement to prove propositions 2 and 3 are correct
theorem correct_propositions : 
  (proposition2 ∧ proposition3) ∧ ¬proposition1 ∧ ¬proposition4 := 
by 
  sorry

end correct_propositions_l141_141928


namespace average_score_of_male_students_standard_deviation_of_all_students_l141_141023

def students : ℕ := 5
def total_average_score : ℝ := 80
def male_student_variance : ℝ := 150
def female_student1_score : ℝ := 85
def female_student2_score : ℝ := 75
def male_student_average_score : ℝ := 80 -- From solution step (1)
def total_standard_deviation : ℝ := 10 -- From solution step (2)

theorem average_score_of_male_students :
  (students = 5) →
  (total_average_score = 80) →
  (male_student_variance = 150) →
  (female_student1_score = 85) →
  (female_student2_score = 75) →
  male_student_average_score = 80 :=
by sorry

theorem standard_deviation_of_all_students :
  (students = 5) →
  (total_average_score = 80) →
  (male_student_variance = 150) →
  (female_student1_score = 85) →
  (female_student2_score = 75) →
  total_standard_deviation = 10 :=
by sorry

end average_score_of_male_students_standard_deviation_of_all_students_l141_141023


namespace investment_value_after_five_years_l141_141443

theorem investment_value_after_five_years :
  let initial_investment := 10000
  let year1 := initial_investment * (1 - 0.05) * (1 + 0.02)
  let year2 := year1 * (1 + 0.10) * (1 + 0.02)
  let year3 := year2 * (1 + 0.04) * (1 + 0.02)
  let year4 := year3 * (1 - 0.03) * (1 + 0.02)
  let year5 := year4 * (1 + 0.08) * (1 + 0.02)
  year5 = 12570.99 :=
  sorry

end investment_value_after_five_years_l141_141443


namespace solve_for_n_l141_141699

theorem solve_for_n (n : ℝ) (h : 0.05 * n + 0.1 * (30 + n) - 0.02 * n = 15.5) : n = 96 := 
by 
  sorry

end solve_for_n_l141_141699


namespace second_term_geometric_series_l141_141829

theorem second_term_geometric_series (a r S : ℝ) (h1 : r = 1 / 4) (h2 : S = 48) (h3 : S = a / (1 - r)) :
  a * r = 9 :=
by
  -- Sorry is used to finalize the theorem without providing a proof here
  sorry

end second_term_geometric_series_l141_141829


namespace twenty_kopeck_greater_than_ten_kopeck_l141_141709

-- Definitions of the conditions
variables (x y z : ℕ)
axiom total_coins : x + y + z = 30 
axiom total_value : 10 * x + 15 * y + 20 * z = 500 

-- The proof statement
theorem twenty_kopeck_greater_than_ten_kopeck : z > x :=
sorry

end twenty_kopeck_greater_than_ten_kopeck_l141_141709


namespace student_l141_141820

noncomputable def allowance_after_video_games (A : ℝ) : ℝ := (3 / 7) * A

noncomputable def allowance_after_comic_books (remaining_after_video_games : ℝ) : ℝ := (3 / 5) * remaining_after_video_games

noncomputable def allowance_after_trading_cards (remaining_after_comic_books : ℝ) : ℝ := (5 / 8) * remaining_after_comic_books

noncomputable def last_allowance (remaining_after_trading_cards : ℝ) : ℝ := remaining_after_trading_cards

theorem student's_monthly_allowance (A : ℝ) (h1 : last_allowance (allowance_after_trading_cards (allowance_after_comic_books (allowance_after_video_games A))) = 1.20) :
  A = 7.47 := 
sorry

end student_l141_141820


namespace different_shapes_and_colors_l141_141039

theorem different_shapes_and_colors (jugs : Fin 40 → Type) (shapes : jugs → Type) (colors : jugs → Type) :
  (∃ j1 j2 : jugs, shapes j1 ≠ shapes j2) →
  (∃ j3 j4 : jugs, colors j3 ≠ colors j4) →
  (∃ j5 j6 : jugs, shapes j5 ≠ shapes j6 ∧ colors j5 ≠ colors j6) :=
by
  intros hshapes hcolors
  -- Add proof steps here
  sorry

end different_shapes_and_colors_l141_141039


namespace problem_statement_l141_141808

theorem problem_statement 
  (d e f : ℝ)
  (h1 : g(0) = 5)
  (h2 : g(2) = 3)
  (hg : g x = d * x^2 + e * x + f) :
  d + e + 3 * f = 14 :=
by {
  sorry
}

end problem_statement_l141_141808


namespace maddie_total_payment_l141_141673

-- Define costs and discounts
def cost_makeup_palettes : ℝ := 3 * 15
def discount_makeup_palettes : ℝ := cost_makeup_palettes * 0.20
def final_cost_makeup_palettes : ℝ := cost_makeup_palettes - discount_makeup_palettes

def cost_lipsticks : ℝ := 4 * 2.50
def discount_lipsticks : ℝ := cost_lipsticks - 2.50

def cost_hair_color : ℝ := 3 * 4
def discount_hair_color : ℝ := 4 * 0.10
def final_cost_hair_color : ℝ := cost_hair_color - discount_hair_color

def initial_subtotal : ℝ := final_cost_makeup_palettes + discount_lipsticks + final_cost_hair_color

-- Storewide discount
def storewide_discount : ℝ := if initial_subtotal > 50 then initial_subtotal * 0.10 else if initial_subtotal >= 30 then initial_subtotal * 0.05 else 0

def subtotal_after_storewide_discount : ℝ := initial_subtotal - storewide_discount

-- Reward points
def reward_points_discount : ℝ := 5
def subtotal_after_reward_points : ℝ := subtotal_after_storewide_discount - reward_points_discount

-- Sales tax calculation
def tax_on_first_25 : ℝ := min 25 subtotal_after_reward_points * 0.05
def remaining_amount : ℝ := max 0 (subtotal_after_reward_points - 25)
def tax_on_remaining : ℝ := remaining_amount * 0.08
def total_tax : ℝ := tax_on_first_25 + tax_on_remaining

-- Total amount paid
def total_amount_paid : ℝ := subtotal_after_reward_points + total_tax

theorem maddie_total_payment :
  total_amount_paid ≈ 47.41 :=
  by sorry

end maddie_total_payment_l141_141673


namespace triangle_side_length_l141_141205

theorem triangle_side_length (a b c : ℝ) (B : ℝ) (ha : a = 2) (hB : B = 60) (hc : c = 3) :
  b = Real.sqrt 7 :=
by
  sorry

end triangle_side_length_l141_141205


namespace number_of_det_values_l141_141071

theorem number_of_det_values (n : ℕ) 
  (A : Matrix (Fin n) (Fin n) ℝ) 
  (h : A^3 - A^2 - 3 * A + (2 : ℝ) • (1 : Matrix (Fin n) (Fin n) ℝ) = 0) : 
  (∃ k, k = finset.card (finset.univ.image (λ v : Fin (n + 2), v))) :=
begin
  sorry
end

end number_of_det_values_l141_141071


namespace complex_z_solution_l141_141465

theorem complex_z_solution (z : ℂ) (hz : 3 * z - 4 * conj z = 5 + 12 * complex.I) : 
  z = -5 + (12 / 7) * complex.I :=
begin
  -- proof goes here, but it is not required as stated
  sorry
end

end complex_z_solution_l141_141465


namespace spy_detection_prob_l141_141812

theorem spy_detection_prob {s : ℝ} (h_s : s = 10) :
  let forest_square := s * s
  let one_rdf_not_working := true
  let total_rdfs := 4
  let working_rdfs := total_rdfs - 1
  let detection_condition := λ rdfs : ℕ, rdfs >= 2
  let coverage_radius := s
  let operational_rdfs_coverage := function {coord : ℝ × ℝ, rdfs : ℕ} 
                                      -> if rdfs >= 2 then true else false
  operational_rdfs_coverage (0, 0) working_rdfs = true 
  → operational_rdfs_coverage (s, 0) working_rdfs = true 
  → operational_rdfs_coverage (0, s) working_rdfs = true ↔
  working_rdfs < total_rdfs - 2 → 
  ℙ{ not operational_rdfs_coverage } = 0.087 :=
begin
  sorry
end

end spy_detection_prob_l141_141812


namespace pos_diff_roots_l141_141012

variable (x : ℝ)

-- Define the quadratic equation condition
def quadratic_eq := 2 * x^2 - 8 * x - 22 = 0

-- State the positive difference between the roots
theorem pos_diff_roots {x₁ x₂ : ℝ} (h : quadratic_eq x₁ ∧ quadratic_eq x₂) : abs (x₁ - x₂) = 2 * sqrt 15 :=
sorry

end pos_diff_roots_l141_141012


namespace successive_product_l141_141300

theorem successive_product (n : ℤ) (h : n * (n + 1) = 4160) : n = 64 :=
sorry

end successive_product_l141_141300


namespace average_salary_increase_l141_141705

theorem average_salary_increase :
  let avg_salary := 1200
  let num_employees := 20
  let manager_salary := 3300
  let new_num_people := num_employees + 1
  let total_salary := num_employees * avg_salary
  let new_total_salary := total_salary + manager_salary
  let new_avg_salary := new_total_salary / new_num_people
  let increase := new_avg_salary - avg_salary
  increase = 100 :=
by
  sorry

end average_salary_increase_l141_141705


namespace sum_of_digits_from_1_to_2008_l141_141315

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def sum_of_digits_range (m n : ℕ) : ℕ :=
  (range (n - m + 1)).sum (λ i, sum_of_digits (i + m))

theorem sum_of_digits_from_1_to_2008 :
  sum_of_digits_range 1 2008 = 28054 :=
by
  sorry

end sum_of_digits_from_1_to_2008_l141_141315


namespace gina_dimes_l141_141890

theorem gina_dimes (d n : ℕ) (h1 : d + n = 50) (h2 : 0.10 * d + 0.05 * n = 4.30) : d = 36 :=
by
  sorry

end gina_dimes_l141_141890


namespace f_2010_equals_2010_l141_141125

-- Define the function f and the conditions
def f : ℝ → ℝ := sorry

axiom cond1 : ∀ x : ℝ, f(x + 1) = f(x) + f(1)

axiom cond2 : f(1) = 1

-- The goal to prove
theorem f_2010_equals_2010 : f(2010) = 2010 :=
by
  sorry

end f_2010_equals_2010_l141_141125


namespace cone_volume_l141_141568

theorem cone_volume (lateral_area : ℝ) (angle : ℝ) 
  (h₀ : lateral_area = 20 * Real.pi)
  (h₁ : angle = Real.arccos (4/5)) : 
  (1/3) * Real.pi * (4^2) * 3 = 16 * Real.pi :=
by
  sorry

end cone_volume_l141_141568


namespace absent_children_on_teachers_day_l141_141250

theorem absent_children_on_teachers_day (A : ℕ) (h1 : ∀ n : ℕ, n = 190)
(h2 : ∀ s : ℕ, s = 38) (h3 : ∀ extra : ℕ, extra = 14) :
  (190 - A) * 38 = 190 * 24 → A = 70 :=
by
  sorry

end absent_children_on_teachers_day_l141_141250


namespace rectangular_cuboid_edges_l141_141712

theorem rectangular_cuboid_edges 
  (d a : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < d)
  (h3 : ∃ (e1 e2 e3 : ℝ), (e1 + e3 = 2 * e2) ∧ 
     (e1^2 + e2^2 + e3^2 = d^2) ∧ 
     (2 * (e1 * e2 + e2 * e3 + e1 * e3) = 2 * a^2)) :
  ∃ (e1 e2 e3 : ℝ), e1 = (1/3 * real.sqrt (d^2 + 2 * a^2)) - real.sqrt ((d^2 - a^2)/3) ∧
                     e2 = (1/3 * real.sqrt (d^2 + 2 * a^2)) ∧
                     e3 = (1/3 * real.sqrt (d^2 + 2 * a^2)) + real.sqrt ((d^2 - a^2)/3) :=
sorry

end rectangular_cuboid_edges_l141_141712


namespace students_play_both_football_and_tennis_l141_141983

theorem students_play_both_football_and_tennis 
  (T : ℕ) (F : ℕ) (L : ℕ) (N : ℕ) (B : ℕ)
  (hT : T = 38) (hF : F = 26) (hL : L = 20) (hN : N = 9) :
  B = F + L - (T - N) → B = 17 :=
by 
  intros h
  rw [hT, hF, hL, hN] at h
  exact h

end students_play_both_football_and_tennis_l141_141983


namespace distance_between_vertices_l141_141482

theorem distance_between_vertices (a b : ℝ) (h1 : a^2 = 144) (h2 : b^2 = 36) : 
  let distance := 2 * real.sqrt a^2 in distance = 24 :=
by
  sorry

end distance_between_vertices_l141_141482


namespace amanda_debt_45_hours_l141_141823

def earnings_per_hour (hour : ℕ) : ℕ :=
  match hour % 6 with
  | 1 => 2
  | 2 => 4
  | 3 => 6
  | 4 => 8
  | 5 => 10
  | 0 => 12
  | _ => 0  -- This case is actually unreachable, but Lean requires it to be complete.

def total_earnings (hours : ℕ) : ℕ :=
  (List.range hours).map (λ h => earnings_per_hour (h + 1)).sum

theorem amanda_debt_45_hours : total_earnings 45 = 306 :=
by
  -- TODO: Add the detailed proof here.
  sorry

end amanda_debt_45_hours_l141_141823


namespace pool_water_left_l141_141455

theorem pool_water_left 
  (h1_rate: ℝ) (h1_time: ℝ)
  (h2_rate: ℝ) (h2_time: ℝ)
  (h4_rate: ℝ) (h4_time: ℝ)
  (leak_loss: ℝ)
  (h1_rate_eq: h1_rate = 8)
  (h1_time_eq: h1_time = 1)
  (h2_rate_eq: h2_rate = 10)
  (h2_time_eq: h2_time = 2)
  (h4_rate_eq: h4_rate = 14)
  (h4_time_eq: h4_time = 1)
  (leak_loss_eq: leak_loss = 8) :
  (h1_rate * h1_time) + (h2_rate * h2_time) + (h2_rate * h2_time) + (h4_rate * h4_time) - leak_loss = 34 :=
by
  rw [h1_rate_eq, h1_time_eq, h2_rate_eq, h2_time_eq, h4_rate_eq, h4_time_eq, leak_loss_eq]
  norm_num
  sorry

end pool_water_left_l141_141455


namespace problem_statement_l141_141390

-- Definitions
def MagnitudeEqual : Prop := (2.4 : ℝ) = (2.40 : ℝ)
def CountUnit2_4 : Prop := (0.1 : ℝ) = 2.4 / 24
def CountUnit2_40 : Prop := (0.01 : ℝ) = 2.40 / 240

-- Theorem statement
theorem problem_statement : MagnitudeEqual ∧ CountUnit2_4 ∧ CountUnit2_40 → True := by
  intros
  sorry

end problem_statement_l141_141390


namespace dodecagon_product_is_zero_l141_141013

-- Definition of the regular dodecagon vertices
def vertices_dodecagon : Fin₁₂ → ℂ 
| 0 => 1  -- Q₁ at (1,0)
| 6 => -3  -- Q₇ at (-3,0)
| n => sorry  -- Remaining vertices not specified in the conditions directly

-- The main statement that needs to be proven
theorem dodecagon_product_is_zero :
  (∏ n in Fin₁₂, vertices_dodecagon n) = 0 :=
sorry

end dodecagon_product_is_zero_l141_141013


namespace sum_of_x_y_l141_141553

theorem sum_of_x_y :
  ∀ (x y : ℚ), (1 / x + 1 / y = 4) → (1 / x - 1 / y = -8) → x + y = -1 / 3 := 
by
  intros x y h1 h2
  sorry

end sum_of_x_y_l141_141553


namespace area_difference_l141_141402

noncomputable def area_circle (r : ℝ) : ℝ := π * r^2

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ := (sqrt 3 / 4) * s^2

theorem area_difference :
  let r := 2 in
  let s := 4 in
  let area_circle := area_circle r in
  let area_triangle := area_equilateral_triangle s in
  area_circle - area_triangle = 4 * (π - sqrt 3) :=
by
  sorry

end area_difference_l141_141402


namespace arithmetic_sequence_sum_eq_4038_l141_141307

noncomputable def arithmetic_sequence_sum {a : ℕ → ℝ} (f : ℝ → ℝ) (n : ℕ) : Prop :=
  (∃ d, d ≠ 0 ∧ ∃ a₁ ∈ set.Icc (0 : ℝ) 4,
  (∀ i, 1 ≤ i ∧ i ≤ n → a i ∈ set.Icc (0 : ℝ) 4 ∧ a (i + 1) = a i + d) ∧
  (∑ i in finset.range n, f (a (i + 1))) = 0)

theorem arithmetic_sequence_sum_eq_4038
  {a : ℕ → ℝ}
  (f : ℝ → ℝ)
  (h : f = λ x, 3 * real.sin (real.pi / 4 * x - real.pi / 2))
  (h_cond : arithmetic_sequence_sum f 2019) :
  ∑ i in finset.range 2019, a (i + 1) = 4038 :=
sorry

end arithmetic_sequence_sum_eq_4038_l141_141307


namespace winner_more_votes_l141_141201

variable (totalStudents : ℕ) (votingPercentage : ℤ) (winnerPercentage : ℤ) (loserPercentage : ℤ)

theorem winner_more_votes
    (h1 : totalStudents = 2000)
    (h2 : votingPercentage = 25)
    (h3 : winnerPercentage = 55)
    (h4 : loserPercentage = 100 - winnerPercentage)
    (h5 : votingStudents = votingPercentage * totalStudents / 100)
    (h6 : winnerVotes = winnerPercentage * votingStudents / 100)
    (h7 : loserVotes = loserPercentage * votingStudents / 100)
    : winnerVotes - loserVotes = 50 := by
  sorry

end winner_more_votes_l141_141201


namespace min_k_for_inequalities_l141_141112

theorem min_k_for_inequalities :
  ∃ (x : ℕ → ℝ) (k : ℕ), (0 < x 1 ∧ 0 < x 2 ∧ ... ∧ 0 < x k) ∧
  (∑ i in finset.range k, (x i)^2 < 1/2 * ∑ i in finset.range k, x i) ∧
  (∑ i in finset.range k, x i < 1/2 * ∑ i in finset.range k, (x i)^3) ∧
  k = 516 :=
sorry

end min_k_for_inequalities_l141_141112


namespace exists_infinitely_many_n_with_increasing_ω_l141_141227

open Nat

/--
  Let ω(n) represent the number of distinct prime factors of a natural number n (where n > 1).
  Prove that there exist infinitely many n such that ω(n) < ω(n + 1) < ω(n + 2).
-/
theorem exists_infinitely_many_n_with_increasing_ω (ω : ℕ → ℕ) (hω : ∀ (n : ℕ), n > 1 → ∃ k, ω k < ω (k + 1) ∧ ω (k + 1) < ω (k + 2)) :
  ∃ (infinitely_many : ℕ → Prop), ∀ N : ℕ, ∃ n : ℕ, N < n ∧ infinitely_many n :=
by
  sorry

end exists_infinitely_many_n_with_increasing_ω_l141_141227


namespace theater_roles_assignment_l141_141024

open Nat

noncomputable def factorial : Nat → Nat
| 0     => 1
| (n + 1) => (n + 1) * factorial n

def permutations (n k : Nat) : Nat :=
factorial n / factorial (n - k)

theorem theater_roles_assignment : 
  let male_roles := permutations 7 3 in
  let female_roles := permutations 4 2 in
  let either_gender_roles := 6 in
  male_roles * female_roles * either_gender_roles = 15120 :=
by
  unfold permutations factorial
  simp
  sorry

end theater_roles_assignment_l141_141024


namespace ellipse_focal_length_l141_141484

theorem ellipse_focal_length :
  let a_squared := 20
    let b_squared := 11
    let c := Real.sqrt (a_squared - b_squared)
    let focal_length := 2 * c
  11 * x^2 + 20 * y^2 = 220 →
  focal_length = 6 :=
by
  sorry

end ellipse_focal_length_l141_141484


namespace measure_of_angle_E_l141_141438

theorem measure_of_angle_E
  (ABCDE : Type)
  (pentagon : ∀ (P : fin 5 → ABCDE), convex_poly P)
  (equal_sides : ∀ (P : fin 5 → ABCDE), sides_equal_length P)
  (angle_A_eq_120 : ∀ (P : fin 5 → ABCDE), angle P 0 = 120)
  (angle_B_eq_120 : ∀ (P : fin 5 → ABCDE), angle P 1 = 120) :
  ∃ P : fin 5 → ABCDE, angle P 4 = 120 := 
sorry

end measure_of_angle_E_l141_141438


namespace regular_polygon_sides_l141_141167

theorem regular_polygon_sides (D : ℕ) (h : D = 30) :
  ∃ n : ℕ, D = n * (n - 3) / 2 ∧ n = 9 :=
by
  use 9
  rw [h]
  norm_num
  sorry

end regular_polygon_sides_l141_141167


namespace min_value_perpendicular_vectors_l141_141954

theorem min_value_perpendicular_vectors (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (hperp : x + 3 * y = 1) : (1 / x + 1 / (3 * y)) = 4 :=
by sorry

end min_value_perpendicular_vectors_l141_141954


namespace hexagon_angle_F_l141_141036

theorem hexagon_angle_F (AB BC CD DE EF FA : ℝ) (angle_A angle_B angle_C : ℝ)
  (h₁ : AB = BC) (h₂ : BC = CD) (h₃ : CD = DE) (h₄ : DE = EF) (h₅ : EF = FA) (h₆ : FA = AB)
  (h₇ : angle_A = 90) (h₈ : angle_B = 90) (h₉ : angle_C = 90) : 
  let angle_F := 150 in 
  true := 
by 
  sorry

end hexagon_angle_F_l141_141036


namespace number_of_three_digit_cubes_divisible_by_8_l141_141962

-- Definitions based on given conditions
def three_digit_perfect_cubes_divisible_by_8 (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = 8 * k^3

-- The final theorem statement.
theorem number_of_three_digit_cubes_divisible_by_8 : 
  {n : ℕ | three_digit_perfect_cubes_divisible_by_8 n}.finite  ∧ 
  {n : ℕ | three_digit_perfect_cubes_divisible_by_8 n}.to_finset.card = 2 :=
by 
  sorry

end number_of_three_digit_cubes_divisible_by_8_l141_141962


namespace average_width_of_books_l141_141216

def widths : list ℝ := [3, 0.75, 1.2, 4, 9, 0.5, 8]

def average_width (l : list ℝ) : ℝ :=
  l.sum / l.length

theorem average_width_of_books :
  average_width widths ≈ 3.78 :=
begin
  -- Definitions and calculations go here
  sorry
end

end average_width_of_books_l141_141216


namespace three_digit_perfect_cubes_divisible_by_8_l141_141959

theorem three_digit_perfect_cubes_divisible_by_8 :
  set.card {x : ℕ | 100 ≤ x ∧ x ≤ 999 ∧ (∃ n : ℕ, x = 8 * n^3)} = 2 :=
sorry

end three_digit_perfect_cubes_divisible_by_8_l141_141959


namespace modulus_imaginary_unit_div_l141_141525

def imaginary_unit (i : ℂ) : Prop := i = complex.I

theorem modulus_imaginary_unit_div (i : ℂ) (h : imaginary_unit i) : complex.abs ((1 + i) / i) = real.sqrt 2 :=
by
  sorry

end modulus_imaginary_unit_div_l141_141525


namespace tangent_line_at_origin_number_of_zeros_of_F_l141_141135

noncomputable def f (x : ℝ) : ℝ := (2 * real.exp x) / x

theorem tangent_line_at_origin (x0 : ℝ) (a : ℝ) :
    (∀ x : ℝ, 2 * (real.exp x * x - real.exp x) / x^2 = (real.exp x / x) / x) → x0 = 2 :=
sorry

noncomputable def F (x b : ℝ) : ℝ := (real.exp x) / x - b * x

theorem number_of_zeros_of_F (b : ℝ) : 
  (if b ≤ 0 then ∀ x : ℝ, F x b ≠ 0
   else if 0 < b ∧ b < (real.exp 2) / 4 then ∃! x : ℝ, F x b = 0
   else if b = (real.exp 2) / 4 then ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ F x₁ b = 0 ∧ F x₂ b = 0
   else ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ F x₁ b = 0 ∧ F x₂ b = 0 ∧ F x₃ b = 0) :=
sorry

end tangent_line_at_origin_number_of_zeros_of_F_l141_141135


namespace new_acute_angle_is_60_degrees_l141_141291

theorem new_acute_angle_is_60_degrees (A B C : Point) (h: angle A C B = 60) :
  let θ := 600 in
  let effective_rotation := θ % 360 in
  let final_angle := effective_rotation - h in  -- this needs to consider full mod
    180 - (180 - final_angle % 360) = 60 :=
by
  let θ := 600
  let effective_rotation := θ % 360
  let final_angle := effective_rotation - h
  have eq1 : effective_rotation = 240 := by sorry -- {600 % 360 = 240}
  have eq2 : final_angle = 180 := by sorry -- {240 - 60 = 180}
  have eq3 : 180 - final_angle % 360 = 120 := by sorry -- {180 - 180 % 360 = 120}
  have eq4 : 180 - 120 = 60 := by sorry
  show 180 - (180 - (240 - 60) % 360 = 60)

end new_acute_angle_is_60_degrees_l141_141291


namespace sum_of_coeffs_except_constant_term_l141_141737

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem sum_of_coeffs_except_constant_term : 
  let f := (fun x => (2 / Real.sqrt x - x) ^ 9)
  ∑ i in finset.range(10), (binomial_coeff 9 i * (2 ^ (9 - i)) * (-1) ^ i) + 5376 = 5377 :=
by
  sorry

end sum_of_coeffs_except_constant_term_l141_141737


namespace example_problem_l141_141134

def f (x : ℝ) : ℝ :=
  if x > 0 then x ^ 2 + 1
  else if x = 0 then Real.pi
  else 0

theorem example_problem : f (f (f (-2016))) = Real.pi ^ 2 + 1 :=
by
  /- Proof steps omitted -/
  sorry

end example_problem_l141_141134


namespace retailer_profit_percentage_l141_141378

def market_price := 1   -- assuming the market price of each pen is $1
def cost_price_36_pens := 36 * market_price -- $36 for 36 pens
def cost_price_100_pens := cost_price_36_pens   -- retailer buys 100 pens for $36
def selling_price_100_pens := 100 * market_price -- $100 without discount
def discount_percentage := 0.01
def discount_amount := discount_percentage * selling_price_100_pens -- discount amount $1
def selling_price_after_discount := selling_price_100_pens - discount_amount -- $99
def profit := selling_price_after_discount - cost_price_100_pens -- $63
def profit_percentage := (profit / cost_price_100_pens) * 100 -- 175%

theorem retailer_profit_percentage :
  profit_percentage = 175 := sorry

end retailer_profit_percentage_l141_141378


namespace no_tangent_range_of_k_l141_141943

-- Define f(x) = x / ln x and the domain of interest
def f (x : ℝ) : ℝ := x / (Real.log x)
def g (x k : ℝ) : ℝ := k * (x - 1)

-- Problem 1 statement
theorem no_tangent (k : ℝ) :
  ¬ ∃ x ∈ ((Set.Ioo 0 1) ∪ (Set.Ioi 1)), 
  Real.diffableAt ℝ f x ∧
  let m := deriv f x
  let tangent_line := m * (x - x) + f x
  tangent_line = g x k := 
sorry

-- Problem 2 statement
noncomputable def k_min := min ((Real.exp 1 - 0.5) / (Real.exp 1 - 1)) 
                                ((0.5 * (Real.exp 2) - 0.5) / (Real.exp 2 - 1))

theorem range_of_k (x : ℝ) (k : ℝ) (hx : x ∈ Set.Icc (Real.exp 1) (Real.exp 2)) : 
  f x ≤ g x k + 1/2 ↔ k ≥ k_min :=
sorry

end no_tangent_range_of_k_l141_141943


namespace four_digit_number_count_l141_141955

theorem four_digit_number_count :
  ∀ (a b c d : ℕ), 
    a ∈ {1, 2, 3, 4} ∧ b ∈ {1, 2, 3, 4} ∧ c ∈ {1, 2, 3, 4} ∧ d ∈ {1, 2, 3, 4} →
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a →
    (a ≤ b ∧ a ≤ c ∧ a ≤ d) →
    (finset.univ.filter (λ (x : ℕ × ℕ × ℕ × ℕ), 
      x.1.1 ∈ {1, 2, 3, 4} ∧ x.1.2 ∈ {1, 2, 3, 4} ∧ x.2.1 ∈ {1, 2, 3, 4} ∧ x.2.2 ∈ {1, 2, 3, 4} ∧
      x.1.1 ≠ x.1.2 ∧ x.1.2 ≠ x.2.1 ∧ x.2.1 ≠ x.2.2 ∧ x.2.2 ≠ x.1.1 ∧
      x.1.1 ≤ x.1.2 ∧ x.1.1 ≤ x.2.1 ∧ x.1.1 ≤ x.2.2)
      .card) = 28 :=
by
  sorry

end four_digit_number_count_l141_141955


namespace xiaoyu_money_left_l141_141400

def box_prices (x y z : ℝ) : Prop :=
  2 * x + 5 * y = z + 3 ∧ 5 * x + 2 * y = z - 3

noncomputable def money_left (x y z : ℝ) : ℝ :=
  z - 7 * x
  
theorem xiaoyu_money_left (x y z : ℝ) (hx : box_prices x y z) :
  money_left x y z = 7 := by
  sorry

end xiaoyu_money_left_l141_141400


namespace determine_k_l141_141937

   variable {k : ℝ} (h1 : k > 0)

   def f (x : ℝ) := k * x^3 - 3 * (k + 1) * x^2 - k^2 + 1

   theorem determine_k (h2 : ∀ x ∈ set.Ioo (0 : ℝ) 4, 
                          3 * k * x^2 - 6 * (k + 1) * x < 0) : 
     k = 1 :=
   by
   sorry
   
end determine_k_l141_141937


namespace probability_both_in_photo_correct_l141_141577

noncomputable def probability_both_in_photo (lap_time_Emily : ℕ) (lap_time_John : ℕ) (observation_start : ℕ) (observation_end : ℕ) : ℚ := 
  let GCD := Nat.gcd lap_time_Emily lap_time_John
  let cycle_time := lap_time_Emily * lap_time_John / GCD
  let visible_time := 2 * min (lap_time_Emily / 3) (lap_time_John / 3)
  visible_time / cycle_time

theorem probability_both_in_photo_correct : 
  probability_both_in_photo 100 75 900 1200 = 1 / 6 :=
by
  -- Use previous calculations and observations here to construct the proof.
  -- sorry is used to indicate that proof steps are omitted.
  sorry

end probability_both_in_photo_correct_l141_141577


namespace product_of_first_three_terms_l141_141715

/--
  The eighth term of an arithmetic sequence is 20.
  If the difference between two consecutive terms is 2,
  prove that the product of the first three terms of the sequence is 480.
-/
theorem product_of_first_three_terms (a d : ℕ) (h_d : d = 2) (h_eighth_term : a + 7 * d = 20) :
  (a * (a + d) * (a + 2 * d) = 480) :=
by
  sorry

end product_of_first_three_terms_l141_141715


namespace max_remainder_l141_141786

-- Definition of the problem
def max_remainder_condition (x : ℕ) (y : ℕ) : Prop :=
  x % 7 = y

theorem max_remainder (y : ℕ) :
  (max_remainder_condition (7 * 102 + y) y ∧ y < 7) → (y = 6 ∧ 7 * 102 + 6 = 720) :=
by
  sorry

end max_remainder_l141_141786


namespace parabola_focus_distance_l141_141532

theorem parabola_focus_distance (m p : ℝ)
  (h_parabola : ∃ y, y^2 = 2 * p * 1)
  (h_distance : real.sqrt((1 - p/2)^2 + m^2) = 5)
  (h_p_pos : p > 0) :
  m = 4 ∨ m = -4 := 
sorry

end parabola_focus_distance_l141_141532


namespace expected_num_red_light_l141_141022

noncomputable def jiaRedLightExpectedValue : ℝ :=
  let n := 3
  let p := 2 / 5
  n * p

theorem expected_num_red_light
  (n : ℕ := 3)
  (p : ℝ := 2 / 5)
  (ξ : ℕ → ℝ := λ k, if k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 then real.to_ennreal 1 else 0)
  (hx : ξ 0 + ξ 1 + ξ 2 + ξ 3 = 1)
  (hx_eq : ξ 0 = real.to_ennreal 1 - 3 * p ∧
            ξ 1 = 3 * p * (1 - p)^2 ∧
            ξ 2 = 3 * p^2 * (1 - p) ∧
            ξ 3 = p^3) :
  (∑ k in Finset.range (3 + 1), k * ξ k) = 3 * (2/5) := sorry

end expected_num_red_light_l141_141022


namespace length_of_crease_l141_141416

theorem length_of_crease 
  (width length : ℝ) 
  (fold_point_distance : ℝ) 
  (θ : ℝ) 
  (h_width : width = 8) 
  (h_length : length = 10) 
  (h_fold_point_distance : fold_point_distance = 2) :
  let L := 2 * (Real.sec θ) * (Real.csc θ)
  in L = 2 * (Real.sec θ) * (Real.csc θ) := 
by
  intros
  sorry

end length_of_crease_l141_141416


namespace total_molecular_weight_correct_l141_141767

-- Defining the molecular weights of elements
def mol_weight_C : ℝ := 12.01
def mol_weight_H : ℝ := 1.01
def mol_weight_Cl : ℝ := 35.45
def mol_weight_O : ℝ := 16.00

-- Defining the number of moles of compounds
def moles_C2H5Cl : ℝ := 15
def moles_O2 : ℝ := 12

-- Calculating the molecular weights of compounds
def mol_weight_C2H5Cl : ℝ := (2 * mol_weight_C) + (5 * mol_weight_H) + mol_weight_Cl
def mol_weight_O2 : ℝ := 2 * mol_weight_O

-- Calculating the total weight of each compound
def total_weight_C2H5Cl : ℝ := moles_C2H5Cl * mol_weight_C2H5Cl
def total_weight_O2 : ℝ := moles_O2 * mol_weight_O2

-- Defining the final total weight
def total_weight : ℝ := total_weight_C2H5Cl + total_weight_O2

-- Statement to prove
theorem total_molecular_weight_correct :
  total_weight = 1351.8 := by
  sorry

end total_molecular_weight_correct_l141_141767


namespace prime_sum_and_difference_l141_141105

theorem prime_sum_and_difference (m n p : ℕ) (hmprime : Nat.Prime m) (hnprime : Nat.Prime n) (hpprime: Nat.Prime p)
  (h1: m > n)
  (h2: n > p)
  (h3 : m + n + p = 74) 
  (h4 : m - n - p = 44) : 
  m = 59 ∧ n = 13 ∧ p = 2 :=
by
  sorry

end prime_sum_and_difference_l141_141105


namespace factor_expression_l141_141099

theorem factor_expression (x : ℝ) : 6 * x ^ 3 - 54 * x = 6 * x * (x + 3) * (x - 3) :=
by {
  sorry
}

end factor_expression_l141_141099


namespace find_x_l141_141906

-- Let A, B, C, M, and O be points in the space
variables {O A B C M : Type}
-- Assume that M lies in the plane determined by points A, B, and C
variable [plane : plane ABC]

theorem find_x {x : ℝ} 
  (h : ∀ (O : Type), 
    vector.exists l $ ∃ n1 n2 n3 : ℝ, n1 + n2 + n3 = 1 ∧
     λ u v w i j k, (u = x * i) + (v = (1 / 3) * j) + (w = (1 / 3) * k)) :
  x = 1 / 3 :=
by
  sorry

end find_x_l141_141906


namespace cot_difference_isosceles_triangle_l141_141590

theorem cot_difference_isosceles_triangle (A B C D: Point) 
(h_triangle: Triangle A B C) 
(h_isosceles: AB = AC) 
(h_median: isMedian A D B C)
(h_angle: ∠ADP = 60) 
: |cot B - cot C| = 2 * sqrt 3 / 3 :=
sorry

end cot_difference_isosceles_triangle_l141_141590


namespace measure_angle_A_correct_l141_141579

noncomputable def measure_of_angle_A (a b : ℝ) (θ : ℝ) : ℝ :=
if 2 * a * (Real.sin θ) = Real.sqrt 3 * b then 60 else 0

theorem measure_angle_A_correct (A B C : Type) [EuclideanGeometry A B C]
  (a b : ℝ) (A_angle B_angle : Angle) (h1 : Triangle A B C)
  (h2 : Acute A B C) (h3 : 2 * a * (sin B_angle) = Real.sqrt 3 * b) :
  ∃ A_angle, A_angle.val = 60 :=
begin
  sorry
end

end measure_angle_A_correct_l141_141579


namespace probability_of_two_primes_l141_141351

-- Define the set of integers from 1 to 30
def finite_set : set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the set of prime numbers from 1 to 30
def primes_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Define the probability of choosing two different primes
def probability_two_primes : ℚ :=
  (fintype.card {p : ℕ × ℕ // p.1 < p.2 ∧ p.1 ∈ primes_set ∧ p.2 ∈ primes_set}) /
  (fintype.card {p : ℕ × ℕ // p.1 < p.2 ∧ p.1 ∈ finite_set ∧ p.2 ∈ finite_set})

-- Prove that the probability is 1/29
theorem probability_of_two_primes :
  probability_two_primes = 1 / 29 :=
sorry

end probability_of_two_primes_l141_141351


namespace chord_division_ratio_l141_141281

theorem chord_division_ratio (R AB PO DP PC x AP PB : ℝ)
  (hR : R = 11)
  (hAB : AB = 18)
  (hPO : PO = 7)
  (hDP : DP = R - PO)
  (hPC : PC = R + PO)
  (hPower : AP * PB = DP * PC)
  (hChord : AP + PB = AB) :
  AP = 12 ∧ PB = 6 ∨ AP = 6 ∧ PB = 12 :=
by
  -- Structure of the theorem is provided.
  -- Proof steps are skipped and marked with sorry.
  sorry

end chord_division_ratio_l141_141281


namespace problem_statement_l141_141607

def T : Set ℤ :=
  {n^2 + (n+2)^2 + (n+4)^2 | n : ℤ }

theorem problem_statement :
  (∀ x ∈ T, ¬ (4 ∣ x)) ∧ (∃ x ∈ T, 13 ∣ x) :=
by
  sorry

end problem_statement_l141_141607


namespace magnitude_of_a_l141_141918

-- Define the unit vectors e1 and e2
variables (e1 e2 : ℝ → ℝ → ℝ)

-- Assume the conditions given in the problem
axiom unit_vector (e : ℝ → ℝ → ℝ) : e 0 1 = 1
axiom angle_condition (e1 e2 : ℝ → ℝ → ℝ) (α : ℝ) : (cos α = 1/3)

-- Define the vector a
def a (e1 e2 : ℝ → ℝ → ℝ) : ℝ → ℝ := λ t, 3 * e1 t 1 - 2 * e2 t 1

-- The goal is to show |a| = 3
theorem magnitude_of_a (e1 e2 : ℝ → ℝ → ℝ) (α : ℝ) (h_e1 : unit_vector e1) (h_e2 : unit_vector e2) 
(h_angle : angle_condition e1 e2 α) : real.sqrt ((3^2) + (2^2) - 12) = 3 := 
by sorry

end magnitude_of_a_l141_141918


namespace solve_a_power_l141_141275

variable (a : ℝ)

theorem solve_a_power (h : 5 = a + a⁻¹) : a^4 + a⁻⁴ = 527 :=
by
  sorry

end solve_a_power_l141_141275


namespace minimum_omega_l141_141620

noncomputable section

-- Define the function f and its properties
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.cos (ω * x + φ)

-- Assumptions based on the given conditions
variables {ω φ : ℝ}
variables (T : ℝ) (hω_pos : 0 < ω) (hφ_range : 0 < φ ∧ φ < π)
variables (hT : f T ω φ = Real.sqrt 3 / 2)
variables (hx_zero : f (π / 9) ω φ = 0)

-- Prove the minimum value of ω is 3
theorem minimum_omega : ω = 3 := sorry

end minimum_omega_l141_141620


namespace min_second_smallest_element_l141_141408

def median (l : List ℕ) : Option ℕ :=
  if l.length % 2 = 1 then some (l.sort.nth (l.length / 2)).getD 0 else none

def mean (l : List ℕ) : Option ℚ :=
  if l.length = 0 then none else some (l.foldr (· + ·) 0 / l.length)

theorem min_second_smallest_element : 
  ∃ (a : List ℕ), a.length = 7 ∧ allPositive a ∧ median a = some 4 ∧ mean a = some 12 ∧ a[1] = 23 :=
by
  sorry

def allPositive (l : List ℕ) : Prop :=
  l.all (λ x => x > 0)

end min_second_smallest_element_l141_141408


namespace smallest_n_exists_l141_141091

theorem smallest_n_exists (n : ℕ) (n > 0) :
  (∃ x : Fin n → ℝ, (finset.univ.sum x) = 500 ∧ finset.univ.sum (λ i, (x i)^4) = 160000) ↔
  n = 290 :=
by
  sorry

end smallest_n_exists_l141_141091


namespace inverse_of_f_inverse_of_f_inv_l141_141285

noncomputable def f (x : ℝ) : ℝ := 3^(x - 1) + 1

noncomputable def f_inv (x : ℝ) : ℝ := 1 + Real.log x / Real.log 3

theorem inverse_of_f (x : ℝ) (hx : x > 1) : f_inv (f x) = x :=
by
  sorry

theorem inverse_of_f_inv (x : ℝ) (hx : x > 1) : f (f_inv x) = x :=
by
  sorry

end inverse_of_f_inverse_of_f_inv_l141_141285


namespace area_triangle_AOB_constant_l141_141731

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the origin point O
def origin (p : ℝ × ℝ) : Prop :=
  p = (0, 0)

-- Define line l' and condition that it intersects curve C at points A and B
def line_l' (k m : ℝ) (P : ℝ × ℝ) : Prop :=
  P.2 = k * P.1 + m ∧ k ≠ 0

-- Define the condition for points A and B
def points_A_B_on_C (k m x1 y1 x2 y2 : ℝ) : Prop :=
  line_l' k m (x1, y1) ∧ curve_C x1 y1 ∧
  line_l' k m (x2, y2) ∧ curve_C x2 y2

-- Define the parallelogram condition
def parallelogram_OAPB (x0 y0 x1 y1 x2 y2 : ℝ) : Prop :=
  x0 = x1 + x2 ∧ y0 = y1 + y2 ∧ ∃ P, curve_C x0 y0

-- Prove that the area of triangle AOB is a constant
theorem area_triangle_AOB_constant
  (k m x1 y1 x2 y2 x0 y0 : ℝ)
  (h1 : points_A_B_on_C k m x1 y1 x2 y2)
  (h2 : parallelogram_OAPB x0 y0 x1 y1 x2 y2)
  (h3 : origin (0, 0)) :
  ∃ s : ℝ, s = 3 / 2 :=
by
  sorry

end area_triangle_AOB_constant_l141_141731


namespace probability_two_primes_1_to_30_l141_141348

def is_prime (n : ℕ) : Prop := 
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13 ∨ n = 17 ∨ n = 19 ∨ n = 23 ∨ n = 29

def count_combinations (n k : ℕ) : ℕ := nat.choose n k

theorem probability_two_primes_1_to_30 :
  let total_combinations := count_combinations 30 2 in
  let prime_combinations := count_combinations 10 2 in
  total_combinations = 435 ∧ 
  prime_combinations = 45 ∧ 
  (prime_combinations : ℚ) / total_combinations = 15 / 145 :=
by { sorry }

end probability_two_primes_1_to_30_l141_141348


namespace remainder_6n_mod_4_l141_141369

theorem remainder_6n_mod_4 (n : ℕ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := by
  sorry

end remainder_6n_mod_4_l141_141369


namespace complex_number_first_quadrant_a_zero_l141_141896

theorem complex_number_first_quadrant_a_zero 
  (a : ℝ) (z : ℂ) (h1 : (1-ℂ.I) * z = a*ℂ.I + 1)  
  (h2 : 0 < z.re ∧ 0 < z.im) :
  a = 0 :=
sorry

end complex_number_first_quadrant_a_zero_l141_141896


namespace uncle_jerry_total_tomatoes_l141_141758

def day1_tomatoes : ℕ := 120
def day2_tomatoes : ℕ := day1_tomatoes + 50
def day3_tomatoes : ℕ := 2 * day2_tomatoes
def total_tomatoes : ℕ := day1_tomatoes + day2_tomatoes + day3_tomatoes

theorem uncle_jerry_total_tomatoes : total_tomatoes = 630 := by
  sorry

end uncle_jerry_total_tomatoes_l141_141758


namespace find_ellipse_equation_l141_141583

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

theorem find_ellipse_equation :
  ∃ a b : ℝ, (a = 8) ∧ (b = sqrt 32) ∧ ellipse_equation a b :=
by
  use 8, sqrt 32
  split
  { refl }
  split
  { exact eq.symm (real.sq_sqrt (by norm_num : 32 ≥ 0)) }
  exact λ x y, by
    field_simp
    sorry

end find_ellipse_equation_l141_141583


namespace complement_union_eq_l141_141608

namespace SetComplementUnion

-- Defining the universal set U, set M and set N.
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- Proving the desired equality
theorem complement_union_eq :
  (U \ M) ∪ N = {x | x > -1} :=
sorry

end SetComplementUnion

end complement_union_eq_l141_141608


namespace original_amount_eq_48_l141_141778

theorem original_amount_eq_48 (x : ℝ) :
  let money_after_first_loss := (2/3) * x
  let money_after_first_win := money_after_first_loss + 10
  let money_after_second_loss := (2/3) * money_after_first_win
  let final_amount := money_after_second_loss + 20
    in final_amount = x → x = 48 :=
by
  intros
  sorry

end original_amount_eq_48_l141_141778


namespace max_value_of_f_l141_141142

noncomputable def f (a x : ℝ) :=
  a^2 * Real.sin (2 * x) + (a - 2) * Real.cos (2 * x)

theorem max_value_of_f (a : ℝ) (hf : ∀ x x', x + x' = -Real.pi / 4 → f a x = f a x') :
  ∃ M, ∀ x, f a x ≤ M ∧ (∃ x₀, f a x₀ = M) :=
by
  use 4 * Real.sqrt 2
  intro x
  split
  · sorry
  · use -Real.pi / 8
    sorry

end max_value_of_f_l141_141142


namespace car_weight_in_pounds_l141_141747

def kg_to_pounds (weight_kg : ℝ) : ℝ :=
  weight_kg / 0.454

theorem car_weight_in_pounds (w_kg : ℝ) (conversion_factor : ℝ) (expected_w_pounds : ℕ) : 
  conversion_factor = 0.454 → w_kg = 1250 → expected_w_pounds = 2753 → 
  Int.round (kg_to_pounds w_kg) = expected_w_pounds :=
by
  intros h1 h2 h3
  rw [h1, h2]
  sorry

end car_weight_in_pounds_l141_141747


namespace minimum_omega_is_3_l141_141630

variable (f : ℝ → ℝ)
variable (ω φ T : ℝ)
variable (x : ℝ)

noncomputable def minimum_omega (ω : ℝ) : ℝ := 3 -- Temporary noncomputable definition to hold the value for min omega

theorem minimum_omega_is_3 :
  (∀ x, f x = Real.cos (ω * x + φ)) →
  ω > 0 →
  0 < φ ∧ φ < Real.pi →
  T = (2 * Real.pi) / ω →
  f T = Real.sqrt 3 / 2 →
  f (Real.pi / 9) = 0 →
  ∃(ω_min : ℝ), ω_min = 3 :=
by
  intros h0 h1 h2 h3 h4 h5
  use 3
  sorry -- Proof of the exact answer

end minimum_omega_is_3_l141_141630


namespace part_I_part_II_l141_141128

open Nat

section ArithmeticSequence

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ → ℤ
| 1     := -3  -- Prove that a_1 is -3
| 2     := -1
| (n+1) := a n + 2  -- Common difference is 2

-- Assume sequence {b_n} satisfies the given recurrence relation
def b (n : ℕ) : ℕ → ℤ
| 1     := 1
| 3     := 1
| (n+1) := b n + a (n+1)  -- Difference relation

-- Math problem statements
theorem part_I : a 1 = -3 := sorry

theorem part_II (n : ℕ) : b n = n^2 - 4*n + 4 := sorry

end ArithmeticSequence

end part_I_part_II_l141_141128


namespace minimum_omega_is_3_l141_141629

variable (f : ℝ → ℝ)
variable (ω φ T : ℝ)
variable (x : ℝ)

noncomputable def minimum_omega (ω : ℝ) : ℝ := 3 -- Temporary noncomputable definition to hold the value for min omega

theorem minimum_omega_is_3 :
  (∀ x, f x = Real.cos (ω * x + φ)) →
  ω > 0 →
  0 < φ ∧ φ < Real.pi →
  T = (2 * Real.pi) / ω →
  f T = Real.sqrt 3 / 2 →
  f (Real.pi / 9) = 0 →
  ∃(ω_min : ℝ), ω_min = 3 :=
by
  intros h0 h1 h2 h3 h4 h5
  use 3
  sorry -- Proof of the exact answer

end minimum_omega_is_3_l141_141629


namespace total_points_P_l141_141117

noncomputable def count_points : ℕ :=
  let A := (1 : ℝ, 1 : ℝ) in
  let circle_center_O := { p : ℝ × ℝ | (p.1)^2 + (p.2)^2 = (1 : ℝ) * (1 : ℝ) } in
  let circle_center_A := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = (1 : ℝ)^2 } in
  let perp_bisector_OA := { p : ℝ × ℝ | p.1 = 0 } in
  (circle_center_O ∩ {p : ℝ × ℝ | p.2 = 0}).to_finset.card +
  (circle_center_A ∩ {p : ℝ × ℝ | p.2 = 0}).to_finset.card +
  2 * (perp_bisector_OA ∩ {p : ℝ × ℝ | p.2 = 0}).to_finset.card

theorem total_points_P (A : ℝ × ℝ) (hA1 : A = (1, 1)) : count_points = 4 := sorry

end total_points_P_l141_141117


namespace find_angle_DAE_l141_141204

noncomputable def triangle ABC := sorry

def angle (A B C : triangle ABC) : ℝ := sorry

def perpendicular_foot (A B C : triangle ABC) : triangle ABC := sorry

def circumcenter (A B C : triangle ABC) : triangle ABC := sorry

def other_end_of_diameter (A : triangle ABC) (circumcenter : triangle ABC) : triangle ABC := sorry

theorem find_angle_DAE (A B C D O E : triangle ABC)
  (h1 : angle A C B = 60)
  (h2 : angle C B A = 50)
  (h3 : D = perpendicular_foot A B C)
  (h4 : O = circumcenter A B C)
  (h5 : E = other_end_of_diameter A O) :
  angle D A E = 70 := 
sorry

end find_angle_DAE_l141_141204


namespace prob_both_primes_l141_141340

-- Define the set of integers from 1 through 30
def int_set : Set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the set of prime numbers between 1 and 30
def primes_between_1_and_30 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Calculate the number of ways to choose two distinct elements from a set
noncomputable def combination (n k : ℕ) : ℕ := if k > n then 0 else n.choose k

-- Define the probabilities
noncomputable def prob_primes : ℚ :=
  (combination 10 2) / (combination 30 2)

-- State the theorem to prove
theorem prob_both_primes : prob_primes = 10 / 87 := by
  sorry

end prob_both_primes_l141_141340


namespace pentagon_angle_E_l141_141437

theorem pentagon_angle_E {ABCDE : Type*} [pentagon ABCDE] 
  (h1 : side_length_eq ABCDE)
  (h2 : ∠A = 120 ∧ ∠B = 120) : 
  ∠E = 90 := 
sorry

end pentagon_angle_E_l141_141437


namespace male_listeners_l141_141409

structure Survey :=
  (males_dont_listen : Nat)
  (females_listen : Nat)
  (total_listeners : Nat)
  (total_dont_listen : Nat)

def number_of_females_dont_listen (s : Survey) : Nat :=
  s.total_dont_listen - s.males_dont_listen

def number_of_males_listen (s : Survey) : Nat :=
  s.total_listeners - s.females_listen

theorem male_listeners (s : Survey) (h : s = { males_dont_listen := 85, females_listen := 75, total_listeners := 180, total_dont_listen := 160 }) :
  number_of_males_listen s = 105 :=
by
  sorry

end male_listeners_l141_141409


namespace equivalent_proof_problem_l141_141047

noncomputable def proof_problem :=
  let A B C F G M N : Point
  let α : Angle
  let circumcircle : Circle
  let BD CE DE CF BG : Line
  in 
  ∃ (t1 t2 t3 t4 t5 t6 t7 : Triangle), 
  t1.angles.ABC ∈ α.val ∧ 
  (α ≠ 60) ∧ 
  (Line.is_tangent BD circumcircle B) ∧
  (Line.is_tangent CE circumcircle C) ∧
  (BD = CE) ∧
  (BD = BC) ∧
  (DE.intersects (Line.extend AB) F) ∧
  (DE.intersects (Line.extend AC) G) ∧
  (CF.intersects BD M) ∧
  (CE.intersects BG N) ∧
  (⊥ AM AN)

theorem equivalent_proof_problem (A B C M N : Point)
  (circumcircle : Circle)
  (BD CE DE CF BG : Line)
  (F G : Point)
  (α : Angle)
  (t1 t2 t3 t4 t5 t6 t7 : Triangle)
  (h1 : t1.angles.ABC ∈ α.val)
  (h2 : α ≠ 60)
  (h3 : Line.is_tangent BD circumcircle B)
  (h4 : Line.is_tangent CE circumcircle C)
  (h5 : BD = CE)
  (h6 : BD = BC)
  (h7 : DE.intersects (Line.extend AB) F)
  (h8 : DE.intersects (Line.extend AC) G)
  (h9 : CF.intersects BD M)
  (h10 : CE.intersects BG N)
: AM = AN := sorry

end equivalent_proof_problem_l141_141047


namespace probability_both_numbers_prime_l141_141355

open Finset

def primes_between_1_and_30 : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_both_numbers_prime :
  (∃ (a b : ℕ), a ≠ b ∧ a ∈ primes_between_1_and_30 ∧ b ∈ primes_between_1_and_30 ∧
  (45 / 435) = (1 / 9)) :=
by
  have h_prime_count : ∃ (s : Finset ℕ), s.card = 10 ∧ s = primes_between_1_and_30 := sorry,
  have h_total_pairs : ∃ (n : ℕ), n = 435 := sorry,
  have h_prime_pairs : ∃ (m : ℕ), m = 45 := sorry,
  have h_fraction : (45 : ℚ) / 435 = (1 : ℚ) / 9 := sorry,
  exact ⟨45, 435, by simp [h_prime_pairs, h_total_pairs, h_fraction]⟩

end probability_both_numbers_prime_l141_141355


namespace ratio_of_hardback_books_is_two_to_one_l141_141446

noncomputable def ratio_of_hardback_books : ℕ :=
  let sarah_paperbacks := 6
  let sarah_hardbacks := 4
  let brother_paperbacks := sarah_paperbacks / 3
  let total_books_brother := 10
  let brother_hardbacks := total_books_brother - brother_paperbacks
  brother_hardbacks / sarah_hardbacks

theorem ratio_of_hardback_books_is_two_to_one : 
  ratio_of_hardback_books = 2 :=
by
  sorry

end ratio_of_hardback_books_is_two_to_one_l141_141446


namespace prime_probability_l141_141335

theorem prime_probability (P : Finset ℕ) : 
  (P = {p | p ≤ 30 ∧ Nat.Prime p}).card = 10 → 
  (Finset.Icc 1 30).card = 30 →
  ((Finset.Icc 1 30).card).choose 2 = 435 →
  (P.card).choose 2 = 45 →
  45 / 435 = 1 / 29 :=
by
  intros hP hIcc30 hChoose30 hChoosePrime
  sorry

end prime_probability_l141_141335


namespace minimum_omega_is_3_l141_141627

variable (f : ℝ → ℝ)
variable (ω φ T : ℝ)
variable (x : ℝ)

noncomputable def minimum_omega (ω : ℝ) : ℝ := 3 -- Temporary noncomputable definition to hold the value for min omega

theorem minimum_omega_is_3 :
  (∀ x, f x = Real.cos (ω * x + φ)) →
  ω > 0 →
  0 < φ ∧ φ < Real.pi →
  T = (2 * Real.pi) / ω →
  f T = Real.sqrt 3 / 2 →
  f (Real.pi / 9) = 0 →
  ∃(ω_min : ℝ), ω_min = 3 :=
by
  intros h0 h1 h2 h3 h4 h5
  use 3
  sorry -- Proof of the exact answer

end minimum_omega_is_3_l141_141627


namespace club_with_two_thirds_participation_l141_141470

variables {Student Club : Type} (students : Finset Student) (clubs : Finset Club)
variables (participates : Student → Club → Prop)

-- Condition 1: Each student participates in no more than two clubs.
def at_most_two_clubs (participates : Student → Club → Prop) (student : Student) : Prop :=
  (students.filter (λ c, participates student c)).card ≤ 2

-- Condition 2: For any pair of students, there exists a club they both participate in.
def exists_common_club (participates : Student → Club → Prop) : Prop :=
  ∀ (s1 s2 : Student), ∃ c : Club, participates s1 c ∧ participates s2 c

-- Theorem to prove that there is a club with at least 2/3 of the students participating.
theorem club_with_two_thirds_participation :
  (∀ s, at_most_two_clubs participates s) →
  exists_common_club participates →
  ∃ c : Club, (students.filter (λ s, participates s c)).card ≥ (2 * students.card) / 3 :=
begin
  sorry
end

end club_with_two_thirds_participation_l141_141470


namespace remaining_eggs_l141_141031

theorem remaining_eggs (dozens : ℕ) (eggs_per_dozen : ℕ) (used_ratio : ℚ) (broken_eggs : ℕ) :
  dozens = 6 →
  eggs_per_dozen = 12 →
  used_ratio = 1 / 2 →
  broken_eggs = 15 →
  let total_eggs := dozens * eggs_per_dozen in
  let used_eggs := total_eggs * used_ratio in
  let remaining_eggs := total_eggs - used_eggs in
  let final_eggs := remaining_eggs - broken_eggs in
  final_eggs = 21 :=
by
  intros hd hz hr hb
  rw [hd, hz, hr, hb]
  let total_eggs := 6 * 12
  let used_eggs := total_eggs * (1 / 2 : ℚ)
  let remaining_eggs := total_eggs - used_eggs
  have h1 : remaining_eggs = 36 := by norm_num
  let final_eggs := remaining_eggs - 15
  have h2 : final_eggs = 21 := by norm_num
  exact h2

end remaining_eggs_l141_141031


namespace part_of_share_sold_l141_141410

-- Define the conditions
def lot_value : ℝ := 9200
def man_share_fraction : ℝ := 1 / 2
def sale_amount : ℝ := 460

-- Calculate the total value of the man's share
def man_share_value : ℝ := man_share_fraction * lot_value

-- Calculate the fraction of his share sold
def share_sold_fraction : ℝ := sale_amount / man_share_value

-- State the theorem
theorem part_of_share_sold : share_sold_fraction = 1 / 10 := by
  sorry

end part_of_share_sold_l141_141410


namespace simplify_expression_sum_of_coefficients_l141_141271

theorem simplify_expression (k : ℝ) (h : k ≠ 0) : (8 * k + 3 + 6 * k^2) + (5 * k^2 + 4 * k + 7) = 11 * k^2 + 12 * k + 10 :=
by sorry

theorem sum_of_coefficients : 11 + 12 + 10 = 33 :=
by rfl

example (k : ℝ) (h : k ≠ 0) : (8 * k + 3 + 6 * k^2 + 5 * k^2 + 4 * k + 7) = 11 * k^2 + 12 * k + 10 ∧ (11 + 12 + 10 = 33) :=
by exact ⟨simplify_expression k h, sum_of_coefficients⟩

end simplify_expression_sum_of_coefficients_l141_141271


namespace vertical_asymptote_at_5_l141_141089

theorem vertical_asymptote_at_5 (num denom : ℤ → ℤ) (x : ℤ)
  (h_denom : denom x = x - 5)
  (h_num : num x = x^3 + 3 * x^2 + 2 * x + 9)
  (h_asymptote : denom 5 = 0 ∧ num 5 ≠ 0) :
  ∃ x, x = 5 ∧ denom x = 0 ∧ num x ≠ 0 :=
by
  use 5
  split
  exact rfl
  split
  exact h_asymptote.1
  exact h_asymptote.2

end vertical_asymptote_at_5_l141_141089


namespace polynomial_coeff_sum_l141_141550

theorem polynomial_coeff_sum :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) (x : ℝ),
    (3 * x - 1) ^ 5 = a_0 + a_1 * x + a_2 * x ^ 2 + a_3 * x ^ 3 + a_4 * x ^ 4 + a_5 * x ^ 5 →
    a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 240 :=
by
  -- FILL IN THE PROOF LATER
  intros,
  sorry

end polynomial_coeff_sum_l141_141550


namespace find_S12_l141_141545

def sequence (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2^(n/2)
  else 2 * n - 1

def sum_sequence (n : ℕ) : ℕ :=
  (finset.range n).sum sequence

theorem find_S12 : sum_sequence 12 = 1443 :=
by sorry

end find_S12_l141_141545


namespace find_c_quadratic_solution_l141_141520

theorem find_c_quadratic_solution (c : ℝ) :
  (Polynomial.eval (-5) (Polynomial.C (-45) + Polynomial.X * Polynomial.C c + Polynomial.X^2) = 0) →
  c = -4 :=
by 
  intros h
  sorry

end find_c_quadratic_solution_l141_141520


namespace find_a_quadratic_trinomials_l141_141868

theorem find_a_quadratic_trinomials :
  ∃ a : ℝ, 
    (∀ f g : polynomial ℝ, 
      f = polynomial.C (4 * a) + polynomial.C (-6) * polynomial.X + polynomial.X ^ 2 ∧
      g = polynomial.C 6 + polynomial.C a * polynomial.X + polynomial.X ^ 2 ∧
      f.discriminant > 0 ∧
      g.discriminant > 0 ∧
      (let x1, x2 := ((-(-6)) ± (sqrt ((-6)^2 - 4 * 1 * (4 * a)))) / (2 * 1),
           y1, y2 := ((-a) ± (sqrt (a^2 - 4 * 1 * 6))) / (2 * 1)
       in x1^2 + x2^2 = y1^2 + y2^2)
    ) ∧ a = -12 :=
  sorry

end find_a_quadratic_trinomials_l141_141868


namespace triangle_angle_not_greater_than_60_l141_141686

theorem triangle_angle_not_greater_than_60 (A B C : ℝ) (h1 : A + B + C = 180) :
  A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 :=
sorry -- proof by contradiction to be implemented here

end triangle_angle_not_greater_than_60_l141_141686


namespace determinant_matrix_l141_141863

/-- Given a 3x3 matrix where the elements are variables x and y, prove that its determinant equals x^2 + y^2. -/
theorem determinant_matrix (x y : ℝ) : 
  Det (Matrix.of ![ ![1, x, y], ![1, x - y, y], ![1, x, y - x] ]) = x^2 + y^2 :=
by
  sorry

end determinant_matrix_l141_141863


namespace cassie_height_in_cm_correct_l141_141056

-- Cassie's height in inches
def cassie_height_in_inches : ℝ := 68
-- Conversion factor from inches to centimeters
def inch_to_cm_conversion_factor : ℝ := 2.54
-- Cassie's height in centimeters by conversion and rounding to the nearest tenth
def cassie_height_in_cm_rounded : ℝ := 172.7

theorem cassie_height_in_cm_correct :
  (cassie_height_in_inches * inch_to_cm_conversion_factor).round * 0.1 = cassie_height_in_cm_rounded :=
by
  sorry

end cassie_height_in_cm_correct_l141_141056


namespace magical_stack_card_count_l141_141708

noncomputable def total_cards {n : ℕ} (cards : Fin (2 * n) → ℕ) (pile_A pile_B : Fin n → ℕ) : Prop :=
  let restack (A B : List ℕ) := List.interleave A B
  pile_A = List.range n + 1 ∧
  pile_B = List.range (n, 2 * n) + 1 ∧
  let orig_stack := List.range (1, 2 * n + 1) in
  let restacked := restack pile_B pile_A in
  ∃ (k : ℕ), (k ≠ 0) ∧ (k < 2 * n) ∧ orig_stack[k] = 157 ∧ restacked[k] = 157

theorem magical_stack_card_count {n : ℕ} (cards : Fin (2 * n) → ℕ) (pile_A pile_B : Fin n → ℕ) :
  (total_cards cards pile_A pile_B) → 2 * n = 470 :=
by
  sorry

end magical_stack_card_count_l141_141708


namespace min_omega_is_three_l141_141639

open Real

noncomputable def min_omega {ω : ℝ} {ϕ : ℝ} (f : ℝ → ℝ) : ℝ :=
  if f = (λ x, cos (ω * x + ϕ)) then
    if 0 < ω ∧ 0 < ϕ ∧ ϕ < π ∧
       (∃ (T : ℝ), T = 2 * π / ω ∧ f T = sqrt 3 / 2) ∧
       f (π / 9) = 0 then
      ω
    else 0
  else 0

theorem min_omega_is_three :
  let f : ℝ → ℝ := λ x, cos (3 * x + π / 6) in
  min_omega f = 3 := by
  sorry

end min_omega_is_three_l141_141639


namespace maximize_exponential_sum_l141_141611

theorem maximize_exponential_sum (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 ≤ 4) : 
  e^a + e^b + e^c + e^d ≤ 4 * Real.exp 1 := 
sorry

end maximize_exponential_sum_l141_141611


namespace john_debt_exceeds_threshold_l141_141598

theorem john_debt_exceeds_threshold : 
  ∃ (t : ℕ), (1.06 ^ t > 3) ∧ (∀ (n : ℕ), n < t → 1.06 ^ n ≤ 3) :=
sorry

end john_debt_exceeds_threshold_l141_141598


namespace problem_solution_l141_141141

-- Define the function f
def f (x : ℝ) : ℝ := log 2 (4 * x) * log 2 (2 * x)

-- Define the range of t
def range_of_t : Set ℝ := { t | -2 ≤ t ∧ t ≤ 2 }

-- Define the minimum and maximum values of f and their corresponding x values
theorem problem_solution :
  (∀ x, x ∈ Icc (1/4) 4 → f x = 0 → (x = 1/4 ∨ x = 4)) ∧
  (∃ t : ℝ, t ∈ range_of_t ∧ f (2^t) = -1/4 ∧ 2^t = 1/4) ∧
  (∃ t : ℝ, t ∈ range_of_t ∧ f (2^t) = 12 ∧ 2^t = 4) :=
by
  sorry

end problem_solution_l141_141141


namespace number_of_true_propositions_l141_141932

open Classical

-- Define each proposition as a term or lemma in Lean
def prop1 : Prop := ∀ x : ℝ, x^2 + 1 > 0
def prop2 : Prop := ∀ x : ℕ, x^4 ≥ 1
def prop3 : Prop := ∃ x : ℤ, x^3 < 1
def prop4 : Prop := ∀ x : ℚ, x^2 ≠ 2

-- The main theorem statement that the number of true propositions is 3 given the conditions
theorem number_of_true_propositions : (prop1 ∧ prop3 ∧ prop4) ∧ ¬prop2 → 3 = 3 := by
  sorry

end number_of_true_propositions_l141_141932


namespace frank_oranges_correct_l141_141471

def betty_oranges : ℕ := 12
def sandra_oranges : ℕ := 3 * betty_oranges
def emily_oranges : ℕ := 7 * sandra_oranges
def frank_oranges : ℕ := 5 * emily_oranges

theorem frank_oranges_correct : frank_oranges = 1260 := by
  sorry

end frank_oranges_correct_l141_141471


namespace train_crosses_bridge_in_12_2_seconds_l141_141380

def length_of_train : ℕ := 110
def speed_of_train_kmh : ℕ := 72
def length_of_bridge : ℕ := 134

def speed_of_train_ms : ℚ := speed_of_train_kmh * (1000 : ℚ) / (3600 : ℚ)
def total_distance : ℕ := length_of_train + length_of_bridge

noncomputable def time_to_cross_bridge : ℚ := total_distance / speed_of_train_ms

theorem train_crosses_bridge_in_12_2_seconds : time_to_cross_bridge = 12.2 := by
  sorry

end train_crosses_bridge_in_12_2_seconds_l141_141380


namespace rearrange_digits_2022_l141_141192

theorem rearrange_digits_2022 :
  let digits : List Nat := [2, 0, 2, 2],
      valid_permutations := (List.permutations digits).filter (λ l => l.head! = 2 ∧ l[3]! ≠ 0),
      valid_count := valid_permutations.length
  in valid_count = 3 :=
by 
  let digits := [2, 0, 2, 2]
  let valid_permutations := (List.permutations digits).filter (λ l => l.head! = 2 ∧ l[3]! ≠ 0)
  let valid_count := valid_permutations.length
  sorry

end rearrange_digits_2022_l141_141192


namespace new_acute_angle_ACB_l141_141288

theorem new_acute_angle_ACB (θ : ℝ) (rot_deg : ℝ) (h1 : θ = 60) (h2 : rot_deg = 600) :
  (new_acute_angle (rotate_ray θ rot_deg)) = 0 := 
sorry

end new_acute_angle_ACB_l141_141288


namespace adults_multiple_of_four_l141_141043

def event := Type

parameter (children : ℕ)
parameter (tables : ℕ)
parameter (adults : ℕ)
hypothesis h_child : children = 20
hypothesis h_tables : tables = 4

theorem adults_multiple_of_four (children tables adults : ℕ) (h_child : children = 20) (h_tables : tables = 4) : 
  ∃ (a : ℕ), adults = 4 * a :=
by
  sorry

end adults_multiple_of_four_l141_141043


namespace minimum_omega_formula_l141_141656

noncomputable def minimumOmega (f : ℝ → ℝ) (x T ω φ : ℝ) : ℝ :=
  if (f = λ x, Real.cos (ω * x + φ)) 
     ∧ (ω > 0) 
     ∧ (0 < φ < π)
     ∧ (f T = √3 / 2)
     ∧ (f (π / 9) = 0) 
     ∧ (T = 2 * π / ω) 
  then 3 else 0

theorem minimum_omega_formula {f : ℝ → ℝ} {x T ω φ : ℝ} :
  (f = λ x, Real.cos (ω * x + φ)) 
  ∧ (ω > 0) 
  ∧ (0 < φ < π)
  ∧ (f T = √3 / 2)
  ∧ (f (π / 9) = 0)
  ∧ (T = 2 * π / ω) 
  → minimumOmega f x T ω φ = 3 :=
by
  sorry

end minimum_omega_formula_l141_141656


namespace seymour_flats_of_petunias_l141_141696

theorem seymour_flats_of_petunias
  (P : ℕ) -- Number of flats of petunias
  (H1 : ∀ (P : ℕ), Seymour_pneeds_flat (8 * 8)) -- Fertilizer needed per flat of petunias: 8 petunias/flat * 8 ounces/petunia = 64 ounces/flat
  (H2 : 3 * 6 * 3 = 54) -- Fertilizer needed for roses: 3 flats * 6 roses/flat * 3 ounces/rose = 54 ounces
  (H3 : 2 * 2 = 4) -- Fertilizer needed for Venus flytraps: 2 Venus flytraps * 2 ounces/Venus flytrap = 4 ounces
  (Htotal : 64 * P + 54 + 4 = 314) : -- Total fertilizer required: 64 ounces/flat * P + 54 ounces for roses + 4 ounces for Venus flytraps = 314 ounces
  P = 4 :=
sorry

end seymour_flats_of_petunias_l141_141696


namespace budget_left_equals_16_l141_141034

def initial_budget : ℤ := 200
def expense_shirt : ℤ := 30
def expense_pants : ℤ := 46
def expense_coat : ℤ := 38
def expense_socks : ℤ := 11
def expense_belt : ℤ := 18
def expense_shoes : ℤ := 41

def total_expenses : ℤ := 
  expense_shirt + expense_pants + expense_coat + expense_socks + expense_belt + expense_shoes

def budget_left : ℤ := initial_budget - total_expenses

theorem budget_left_equals_16 : 
  budget_left = 16 := by
  sorry

end budget_left_equals_16_l141_141034


namespace bug_visits_all_vertices_in_three_moves_l141_141788

-- Define the vertices of the tetrahedron
inductive Vertex
  | A | B | C | D

-- Define the move relation between vertices (adjacent vertices)
def move : Vertex -> Vertex -> Prop
| Vertex.A, Vertex.B => true
| Vertex.A, Vertex.C => true
| Vertex.A, Vertex.D => true
| Vertex.B, Vertex.A => true
| Vertex.B, Vertex.C => true
| Vertex.B, Vertex.D => true
| Vertex.C, Vertex.A => true
| Vertex.C, Vertex.B => true
| Vertex.C, Vertex.D => true
| Vertex.D, Vertex.A => true
| Vertex.D, Vertex.B => true
| Vertex.D, Vertex.C => true
| _, _ => false

-- Define a function for the probability that a bug starting at a vertex visits all vertices within 3 moves
noncomputable def probability_visits_all_vertices : ℚ :=
  -- Placeholder for the actual computation
  (2 / 9 : ℚ)

-- Proof statement
theorem bug_visits_all_vertices_in_three_moves :
  probability_visits_all_vertices = (2 / 9 : ℚ) :=
sorry

end bug_visits_all_vertices_in_three_moves_l141_141788


namespace fractional_equation_positive_root_l141_141564

theorem fractional_equation_positive_root (a : ℝ) (ha : ∃ x : ℝ, x > 0 ∧ (6 / (x - 2) - 1 = a * x / (2 - x))) : a = -3 :=
by
  sorry

end fractional_equation_positive_root_l141_141564


namespace tangent_line_equation_range_of_m_l141_141940

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a*x^2 - 6*x + 1

theorem tangent_line_equation (a : ℝ) (ha : 2 * a - 3 = -6) :
  ∃ y : ℝ, y = 12 * 1 + 2 * f 1 a - 1 := sorry

theorem range_of_m (a : ℝ) (ha : 2 * a - 3 = -6) :
  ∀ m : ℝ, 
  (∃ x₁ x₂ x₃ : ℝ, x₁ ∈ Icc (-2 : ℝ) (4 : ℝ) ∧ x₂ ∈ Icc (-2 : ℝ) (4 : ℝ) ∧ 
  x₃ ∈ Icc (-2 : ℝ) (4 : ℝ) ∧ g x₁ = 0 ∧ g x₂ = 0 ∧ g x₃ = 0) ↔ 
  m ∈ set.Ico (-1 : ℝ) (9 / 2) := sorry

end tangent_line_equation_range_of_m_l141_141940


namespace max_cars_quotient_l141_141679

theorem max_cars_quotient :
  let M := 4000 in
  (M / 10 = 400) :=
  by
  let M := 4000
  have h : M / 10 = 400 := sorry
  exact h

end max_cars_quotient_l141_141679


namespace length_AB_l141_141582

-- Defining the parabola condition
def parabola_condition (P F: ℝ × ℝ) (l: ℝ) : Prop :=
  (P.fst + 1)^2 + P.snd^2 = (P.fst - 1)^2 + P.snd^2

-- Defining the direction vector condition
def direction_vector_condition (d: ℝ × ℝ) (p: ℝ × ℝ) (A B: ℝ × ℝ) : Prop :=
  ∃ k: ℝ, (B.snd = p.snd + 2 * (B.fst - p.fst))

-- Combine all conditions and prove |AB| == 5
theorem length_AB (P F A B: ℝ × ℝ) (l: ℝ) (d: ℝ × ℝ) (AB_len: ℝ) :
  parabola_condition P F l ∧
  direction_vector_condition d (1, 0) A B ∧
  line_AB_intersects_parabola A B ∧
  (√((B.fst - A.fst)^2 + (B.snd - A.snd)^2) = AB_len)
  → (AB_len = 5) :=
sorry

end length_AB_l141_141582


namespace correct_statements_are_one_l141_141040

theorem correct_statements_are_one :
  let S1 := "Alternate interior angles are equal"
  let S2 := "Consecutive interior angles are supplementary"
  let S3 := "Vertical angles are equal"
  let S4 := "Adjacent supplementary angles are equal"
  let S5 := "Corresponding angles are equal"
  (is_correct (S1) = false) ∧
  (is_correct (S2) = false) ∧
  (is_correct (S3) = true) ∧
  (is_correct (S4) = false) ∧
  (is_correct (S5) = false) :=
by
  sorry

end correct_statements_are_one_l141_141040


namespace smallest_natural_with_properties_l141_141085

theorem smallest_natural_with_properties :
  ∃ n : ℕ, (∃ N : ℕ, n = 10 * N + 6) ∧ 4 * (10 * N + 6) = 6 * 10^(5 : ℕ) + N ∧ n = 153846 := sorry

end smallest_natural_with_properties_l141_141085


namespace bales_in_barn_l141_141329

/-- Define the total number of bales in the barn after a week given initial bales and daily additions -/
theorem bales_in_barn (initial_bales : ℕ) (day1_bales : ℕ) (day2_bales : ℕ) (day3_bales : ℕ)
  (day4_bales : ℕ) (day5_bales : ℕ) (day6_bales : ℕ) (day7_bales : ℕ) :
  initial_bales = 28 →
  day1_bales = 10 →
  day2_bales = 15 →
  day3_bales = 8 →
  day4_bales = 20 →
  day5_bales = 12 →
  day6_bales = 4 →
  day7_bales = 18 →
  initial_bales + day1_bales + day2_bales + day3_bales + day4_bales + day5_bales + day6_bales + day7_bales = 115 :=
by
  intros h₀ h₁ h₂ h₃ h₄ h₅ h₆ h₇
  rw [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇]
  norm_num

end bales_in_barn_l141_141329


namespace complement_of_set_A_l141_141950

open Set

variable {R : Type*} [LinearOrder R]

def set_A (x : R) : Prop := (x ≥ 3) ∨ (x < -1)

theorem complement_of_set_A : ∀ x : R, (¬ set_A x) ↔ (-1 ≤ x ∧ x < 3) := by
  intros x
  sorry

end complement_of_set_A_l141_141950


namespace gcd_7429_13356_l141_141485

theorem gcd_7429_13356 : Nat.gcd 7429 13356 = 1 := by
  sorry

end gcd_7429_13356_l141_141485


namespace correct_operation_l141_141771

theorem correct_operation (a : ℝ) : 2 * a^3 / a^2 = 2 * a := 
sorry

end correct_operation_l141_141771


namespace range_of_h_l141_141136

noncomputable def f (x : ℝ) : ℝ :=
  sqrt 3 * sin (2 * x) + 2 * (cos x) ^ 2 - 1

def M_t (t : ℝ) : ℝ :=
  ⨆ x in set.Icc t (t + π/4), f x

def m_t (t : ℝ) : ℝ :=
  ⨅ x in set.Icc t (t + π/4), f x

def h (t : ℝ) : ℝ := M_t t - m_t t

theorem range_of_h (t : ℝ) (ht : t ∈ set.Icc (π/12) (5 * π / 12)) :
  set.range (h ∘ λ t, t ∈ set.Icc (π/12) (5 * π / 12)) = set.Icc 1 (2 * sqrt 2) :=
sorry

end range_of_h_l141_141136


namespace derivative_at_zero_l141_141613

variable (n : ℕ)

def f (x : ℕ) : ℕ := (List.range (n+1)).foldr (λ i acc, acc * (x + i)) 1

theorem derivative_at_zero : (f n)' 0 = n! := sorry

end derivative_at_zero_l141_141613


namespace tan_alpha_value_l141_141521

theorem tan_alpha_value (α : ℝ) (h : (sin α - 2 * cos α) / (3 * sin α + 5 * cos α) = -5) : tan α = -23 / 16 :=
by
  sorry

end tan_alpha_value_l141_141521


namespace commonly_used_compound_bar_charts_l141_141278

theorem commonly_used_compound_bar_charts : 
  ∃ (types : List String), types = ["vertical", "horizontal"] :=
by
  exists ["vertical", "horizontal"]
  sorry

end commonly_used_compound_bar_charts_l141_141278


namespace permutations_sat_conditions_l141_141058

theorem permutations_sat_conditions : 
  ∃ (p : SymmetricGroup 6), 
    (p (Fin 6) ![0, 1, 2, 3, 4, 5]) ![0, 1, 2, 3, 4, 5] 
    = ![a, b, c, x, y, z] 
    ∧ a < b ∧ b < c ∧ x < y ∧ y < z ∧ a < x ∧ b < y ∧ c < z := 
by 
  sorry

end permutations_sat_conditions_l141_141058


namespace total_taxi_charge_l141_141597

-- Definitions directly from conditions
def initial_fee : ℝ := 2.25
def additional_charge_per_increment : ℝ := 0.3
def increment_in_miles : ℝ := 2 / 5
def trip_distance : ℝ := 3.6

-- Formal proof statement
theorem total_taxi_charge 
  (initial_fee : ℝ) 
  (additional_charge_per_increment : ℝ) 
  (increment_in_miles : ℝ) 
  (trip_distance : ℝ) :
  initial_fee = 2.25 → 
  additional_charge_per_increment = 0.3 → 
  increment_in_miles = 2 / 5 → 
  trip_distance = 3.6 →
  initial_fee + (trip_distance / increment_in_miles) * additional_charge_per_increment = 7.65 :=
by
  intros h1 h2 h3 h4
  sorry

end total_taxi_charge_l141_141597


namespace find_length_of_crease_l141_141990

open Real

-- Definitions for conditions
def isosceles_triangle (A B C : Point) : Prop := dist A B = dist A C
def on_line (A' B C : Point) : Prop := (∃ t : ℝ, B + t * (C - B) = A')

-- Problem Statement
theorem find_length_of_crease
  (A B C A' P Q : Point)
  (h_iso : isosceles_triangle A B C)
  (h_on_line : on_line A' B C)
  (h_ba' : dist B A' = 2)
  (h_a'c : dist A' C = 3)
  : dist P Q = (7 * sqrt 21) / 20 :=
sorry

end find_length_of_crease_l141_141990


namespace find_x_l141_141972

def average_percentage_50_percent (x : ℝ) : Prop :=
    (0.2 * 80 + 0.5 * x + 0.3 * 40 = 58)

theorem find_x : ∃ x : ℝ, average_percentage_50_percent x ∧ x = 60 :=
by
  use 60
  simp [average_percentage_50_percent]
  norm_num
  sorry

end find_x_l141_141972


namespace max_divisible_num_l141_141466

theorem max_divisible_num (n : ℕ) (h : n = 31)
    (G1 G2 : set ℕ)
    (h_disjoint : ∀ x ∈ G1, x ∈ G2 → false)
    (h_union : G1 ∪ G2 = {k | 2 ≤ k ∧ k ≤ n})
    (h_prod : ∀ x y ∈ G1, x * y ∉ G1 ∧ ∀ x y ∈ G2, x * y ∉ G2)
    (h_square : ∀ x ∈ G1, x^2 ∉ G1 ∧ ∀ x ∈ G2, x^2 ∉ G2) :
  n = 31 :=
sorry

end max_divisible_num_l141_141466


namespace f_at_2007_l141_141554

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 : ∀ x : ℝ, f(x + 4) ≤ f(x) + 4
axiom condition2 : ∀ x : ℝ, f(x + 2) ≥ f(x) + 2
axiom f_at_3 : f(3) = 4

theorem f_at_2007 : f(2007) = 2008 :=
by
  sorry

end f_at_2007_l141_141554


namespace cosine_of_angle_l141_141574

variables (a b : ℝ) (vec_a vec_b : EuclideanSpace ℝ (Fin 3))
variable (ab_dot : vec_a ∙ vec_b = 3)

def magnitude (v : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  Real.sqrt (v ∙ v)

def cosine_of_angle_between (u v : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  (u ∙ v) / ((magnitude u) * (magnitude v))

theorem cosine_of_angle
  (h₁ : magnitude vec_a = 2)
  (h₂ : magnitude vec_b = 2 * Real.sqrt 3)
  (h₃ : ab_dot) :
  cosine_of_angle_between vec_b (vec_b - vec_a) = 3 * Real.sqrt 30 / 20 := 
sorry

end cosine_of_angle_l141_141574


namespace smallest_n_l141_141703

theorem smallest_n (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 2012) :
  ∃ m n : ℕ, a.factorial * b.factorial * c.factorial = m * 10 ^ n ∧ ¬ (10 ∣ m) ∧ n = 501 :=
by
  sorry

end smallest_n_l141_141703


namespace mutually_exclusive_A_B_mutually_exclusive_A_C_not_complementary_B_D_complementary_C_D_l141_141797

noncomputable def uniform_die := {n : ℕ | 1 ≤ n ∧ n ≤ 6 }

def event_A (n : ℕ) := n = 4
def event_B (n : ℕ) := n = 1 ∨ n = 3 ∨ n = 5
def event_C (n : ℕ) := n < 4
def event_D (n : ℕ) := n > 3

theorem mutually_exclusive_A_B :
  ∀ (n : ℕ), n ∈ uniform_die → ¬(event_A n ∧ event_B n) :=
by
  intros n h
  unfold event_A event_B uniform_die
  sorry

theorem mutually_exclusive_A_C :
  ∀ (n : ℕ), n ∈ uniform_die → ¬(event_A n ∧ event_C n) :=
by
  intros n h
  unfold event_A event_C uniform_die
  sorry

theorem not_complementary_B_D :
  ∃ (n : ℕ), n ∈ uniform_die ∧ (event_B n ∧ ¬ event_D n) :=
by
  unfold event_B event_D uniform_die
  sorry

theorem complementary_C_D :
  ∀ (n : ℕ), n ∈ uniform_die → (event_C n ↔ ¬ event_D n) :=
by 
  intros n h
  unfold event_C event_D uniform_die
  sorry

end mutually_exclusive_A_B_mutually_exclusive_A_C_not_complementary_B_D_complementary_C_D_l141_141797


namespace Carla_total_cooking_time_l141_141839

theorem Carla_total_cooking_time :
  let waffles_time := 10
  let steak_time := 6
  let chili_time := 20
  let fries_time := 15
  let num_steaks := 8
  let num_waffles := 5
  let num_chili := 3
  let num_fries := 4
  (num_steaks * steak_time + num_waffles * waffles_time + num_chili * chili_time + num_fries * fries_time) = 218 :=
by {
  let waffles_time := 10
  let steak_time := 6
  let chili_time := 20
  let fries_time := 15
  let num_steaks := 8
  let num_waffles := 5
  let num_chili := 3
  let num_fries := 4
  calc
    (num_steaks * steak_time + num_waffles * waffles_time + num_chili * chili_time + num_fries * fries_time)
    = (8 * 6 + 5 * 10 + 3 * 20 + 4 * 15) : by rfl
    ... = 48 + 50 + 60 + 60 : by rfl
    ... = 218 : by rfl
}

end Carla_total_cooking_time_l141_141839


namespace matrix_pow_expression_l141_141224

def A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 4, 2; 0, 2, 3; 0, 0, 1]
def I : Matrix (Fin 3) (Fin 3) ℝ := 1

theorem matrix_pow_expression :
  A^5 - 3 • A^4 = !![0, 4, 2; 0, -1, 3; 0, 0, -2] := by
  sorry

end matrix_pow_expression_l141_141224


namespace pencil_notebook_cost_l141_141021

theorem pencil_notebook_cost (p n : ℝ)
  (h1 : 9 * p + 10 * n = 5.35)
  (h2 : 6 * p + 4 * n = 2.50) :
  24 * 0.9 * p + 15 * n = 9.24 :=
by 
  sorry

end pencil_notebook_cost_l141_141021


namespace CartesianEquationOfC_distance_PQ_l141_141861

noncomputable def cartesian_equation_of_C: Set (ℝ × ℝ) :=
  {p | (p.1 - 2)^2 + p.2^2 = 4}

noncomputable def parametric_equation_of_line (t : ℝ) : ℝ × ℝ :=
  (-1 + (√3 / 2) * t, (1 / 2) * t)

theorem CartesianEquationOfC:
  ∀ (p : ℝ × ℝ), p ∈ cartesian_equation_of_C ↔ ∃ (θ : ℝ), p = (4 * cos θ, 4 * sin θ) :=
by 
  sorry

theorem distance_PQ : 
  ∃ (t1 t2 : ℝ), 
    t1 * t1 - 3 * √3 * t1 + 5 = 0 ∧ 
    t2 * t2 - 3 * √3 * t2 + 5 = 0 ∧ 
    |(parametric_equation_of_line t1).1 - (parametric_equation_of_line t2).1| = √7 :=
by
  sorry

end CartesianEquationOfC_distance_PQ_l141_141861


namespace total_handshakes_is_correct_l141_141186

-- Definitions of the problem setup
def num_students : ℕ := 40
def sequence (i : ℕ) : ℕ := (i % 4) + 1
def turns_around (i : ℕ) : Prop := sequence i = 3

-- The handshakes function should count the total handshakes described.
noncomputable def total_handshakes : ℕ :=
  have initial_handshakes := num_students / 4 -- initial pairs of 3s and 4s
  let rounds_hands := (num_students / 4) * (num_students / 4 / 2) * 2 + (num_students / 4).sum_nat 0 in
  rounds_hands + sum (list.range ((num_students / 4) -1 + 1))  -- All pairs through rounds

-- Statement of the theorem
theorem total_handshakes_is_correct : total_handshakes = 175 :=
  sorry

end total_handshakes_is_correct_l141_141186


namespace probability_of_two_primes_l141_141353

-- Define the set of integers from 1 to 30
def finite_set : set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the set of prime numbers from 1 to 30
def primes_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Define the probability of choosing two different primes
def probability_two_primes : ℚ :=
  (fintype.card {p : ℕ × ℕ // p.1 < p.2 ∧ p.1 ∈ primes_set ∧ p.2 ∈ primes_set}) /
  (fintype.card {p : ℕ × ℕ // p.1 < p.2 ∧ p.1 ∈ finite_set ∧ p.2 ∈ finite_set})

-- Prove that the probability is 1/29
theorem probability_of_two_primes :
  probability_two_primes = 1 / 29 :=
sorry

end probability_of_two_primes_l141_141353


namespace no_periodic_word_l141_141061

-- Define the sequence and initial conditions
def W : ℕ → String
| 0 => "A"
| 1 => "B"
| n + 2 => W n ++ W (n + 1)

-- Helper function to count occurrences of 'A' in a string
def countA (s : String) : Nat :=
  s.toList.count (λ c => c = 'A')

-- Helper function to count occurrences of 'B' in a string
def countB (s : String) : Nat :=
  s.toList.count (λ c => c = 'B')

theorem no_periodic_word : ¬ ∃ (n : ℕ) (P : String), P ≠ "" ∧ W n = P ++ P :=
sorry

end no_periodic_word_l141_141061


namespace round_trip_time_l141_141312

-- Given conditions
def boat_speed : ℝ := 16  -- speed of the boat in standing water (kmph)
def stream_speed : ℝ := 2  -- speed of the stream (kmph)
def distance_to_place : ℝ := 7380  -- distance to the place (km)

-- Derived definitions
def downstream_speed : ℝ := boat_speed + stream_speed  -- speed downstream
def upstream_speed : ℝ := boat_speed - stream_speed  -- speed upstream

-- Time calculations
def time_downstream : ℝ := distance_to_place / downstream_speed  -- time downstream
def time_upstream : ℝ := distance_to_place / upstream_speed  -- time upstream

-- Total time for the round trip
def total_time : ℝ := time_downstream + time_upstream

-- Statement of the problem to be proved
theorem round_trip_time :
  total_time = 937.14 := by
  sorry

end round_trip_time_l141_141312


namespace train_passing_time_l141_141025

/-- The problem defines a train of length 110 meters traveling at 40 km/hr, 
    passing a man who is running at 5 km/hr in the opposite direction.
    We want to prove that the time it takes for the train to pass the man is 8.8 seconds. -/
theorem train_passing_time :
  ∀ (train_length : ℕ) (train_speed man_speed : ℕ), 
  train_length = 110 → train_speed = 40 → man_speed = 5 →
  (∃ time : ℚ, time = 8.8) :=
by
  intros train_length train_speed man_speed h_train_length h_train_speed h_man_speed
  sorry

end train_passing_time_l141_141025


namespace red_points_on_bigger_circle_l141_141793

noncomputable def numberOfRedPoints (n : ℕ) : ℕ :=
  Int.floor (n * Real.sqrt 2) + 1

theorem red_points_on_bigger_circle (n : ℕ) :
  numberOfRedPoints n = Int.floor (n * Real.sqrt 2) + 1 := by
  sorry

end red_points_on_bigger_circle_l141_141793


namespace triangle_ratio_l141_141981

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ)
  (h_triangle : A + B + C = π)
  (h_angleA : A = 2 * π / 3)
  (h_side_a : a = sqrt 3 * c) :
  a / b = sqrt 3 :=
by 
  sorry

end triangle_ratio_l141_141981


namespace prob_both_primes_l141_141341

-- Define the set of integers from 1 through 30
def int_set : Set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the set of prime numbers between 1 and 30
def primes_between_1_and_30 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Calculate the number of ways to choose two distinct elements from a set
noncomputable def combination (n k : ℕ) : ℕ := if k > n then 0 else n.choose k

-- Define the probabilities
noncomputable def prob_primes : ℚ :=
  (combination 10 2) / (combination 30 2)

-- State the theorem to prove
theorem prob_both_primes : prob_primes = 10 / 87 := by
  sorry

end prob_both_primes_l141_141341


namespace four_Y_one_l141_141160

def Y (a b : ℕ) : ℕ := 3 * (a^2 - 2 * a * b + b^2)

theorem four_Y_one : Y 4 1 = 27 := 
by 
  unfold Y
  simp
  norm_num
  sorry

end four_Y_one_l141_141160


namespace arithmetic_sequence_general_term_sum_of_terms_l141_141114

/-- Define an arithmetic sequence (a_n) and its sum (S_n). -/
def a_n (n : ℕ) := 2 * n - 10
def S_n (n : ℕ) := n * (2 * n - 18) / 2

theorem arithmetic_sequence_general_term (a_3 a_1 S_3 : ℤ) (h1 : a_3 - a_1 = 4) (h2 : S_3 = -18) :
    a_n (3 : ℕ) - a_n (1 : ℕ) = 4 ∧ S_n (3 : ℕ) = -18 →
    a_n (0 : ℕ) = -10 ∧ a_n (1 : ℕ) = -8 ∧ a_n (2 : ℕ) = -6 :=
by sorry

theorem sum_of_terms (k : ℤ) (h : S_n k = -14) : k = 2 ∨ k = 7 :=
by sorry

end arithmetic_sequence_general_term_sum_of_terms_l141_141114


namespace find_f_2015_l141_141883

noncomputable def f : ℕ+ → ℕ+
axiom h1 : f 1 ≠ 1
axiom h2 : ∀ n : ℕ+, f n + f (n + 1) + f (f n) = 3 * n + 1

theorem find_f_2015 (h1 : f 1 ≠ 1) (h2 : ∀ n : ℕ+, f n + f (n + 1) + f (f n) = 3 * n + 1) : f 2015 = 2016 :=
sorry

end find_f_2015_l141_141883


namespace circumcircle_locus_l141_141733

variables {A P B C : Type*}
variables [plane A P the_points B C]

theorem circumcircle_locus (AB BC AP : ℝ) (h_non_collinear : ¬ collinear A B C) :
  AP ≥ max (AB / 2) (BC / 2) :=
sorry

end circumcircle_locus_l141_141733


namespace area_A_l141_141264

-- Definitions for the quadrilateral and points
variables (A B C D A' B' C' D' : Point)
variables (AB BC CD DA : ℝ) (area_ABCD : ℝ)

-- Conditions
def quadrilateral_ABCD_Cond : Prop :=
  convex_quadrilateral A B C D ∧
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
  dist A B = 5 ∧ dist B B' = 5 ∧
  dist B C = 6 ∧ dist C C' = 6 ∧
  dist C D = 7 ∧ dist D D' = 7 ∧
  dist D A = 8 ∧ dist A A' = 8 ∧
  area_ABCD = 12

-- The problem statement to prove
theorem area_A'B'C'D'_eq_36 (h : quadrilateral_ABCD_Cond) : 
  area A' B' C' D' = 36 :=
sorry

end area_A_l141_141264


namespace compute_integer_x_l141_141965

/-- Problem statement:
Given the equation 1 * 100 + 2 * 99 + 3 * 98 + ... + 99 * 2 + 100 * 1 = 100 * 50 * x,
prove that the solution for the integer x is 34. -/
theorem compute_integer_x (x : ℤ) :
  (∑ n in Finset.range 100, (n + 1) * (100 - n)) = 100 * 50 * x → x = 34 :=
by
  intro h
  -- Assuming h is ∑_{n=1}^{100} n(101 - n) = 100 * 50 * x
  sorry

end compute_integer_x_l141_141965


namespace number_of_valid_x_values_l141_141153

theorem number_of_valid_x_values : 
  let x := λ n : ℕ, 3 ≤ n / 100 ∧ n / 100 < 5 ∧ n ≤ 166 in
  ∃ y : ℕ, y = 166 - 34 + 1 ∧ y = 133 :=
begin
  sorry
end

end number_of_valid_x_values_l141_141153


namespace largest_angle_is_right_angle_l141_141727

theorem largest_angle_is_right_angle (u : ℝ) (h1 : 0 < 3u - 2) (h2 : 0 < 3u + 2) (h3 : 0 < 2 * u) :
  is_triangle (sqrt (3 * u - 2)) (sqrt (3 * u + 2)) (2 * sqrt u) ∧
  (3 * u - 2) + (3 * u + 2) = 4 * u →
  ∠(sqrt (3 * u - 2)) (sqrt (3 * u + 2)) (2 * sqrt u) = 90 :=
sorry

end largest_angle_is_right_angle_l141_141727


namespace find_x_floor_eq_8_l141_141481

theorem find_x_floor_eq_8 (x : ℝ) : (floor (x * (floor x - 1)) = 8) ↔ (4 ≤ x ∧ x < 4.5) :=
by
  sorry

end find_x_floor_eq_8_l141_141481


namespace minimum_omega_formula_l141_141650

noncomputable def minimumOmega (f : ℝ → ℝ) (x T ω φ : ℝ) : ℝ :=
  if (f = λ x, Real.cos (ω * x + φ)) 
     ∧ (ω > 0) 
     ∧ (0 < φ < π)
     ∧ (f T = √3 / 2)
     ∧ (f (π / 9) = 0) 
     ∧ (T = 2 * π / ω) 
  then 3 else 0

theorem minimum_omega_formula {f : ℝ → ℝ} {x T ω φ : ℝ} :
  (f = λ x, Real.cos (ω * x + φ)) 
  ∧ (ω > 0) 
  ∧ (0 < φ < π)
  ∧ (f T = √3 / 2)
  ∧ (f (π / 9) = 0)
  ∧ (T = 2 * π / ω) 
  → minimumOmega f x T ω φ = 3 :=
by
  sorry

end minimum_omega_formula_l141_141650


namespace triangle_is_isosceles_or_right_l141_141212

theorem triangle_is_isosceles_or_right (A B C a b : ℝ) (h : a * Real.cos (π - A) + b * Real.sin (π / 2 + B) = 0)
  (h1 : a = 2 * R * Real.sin A) 
  (h2 : b = 2 * R * Real.sin B) : 
  (A = B ∨ A + B = π / 2) := 
sorry

end triangle_is_isosceles_or_right_l141_141212


namespace storks_initial_count_l141_141750

theorem storks_initial_count :
  ∃ S : ℕ, ∃ B : ℕ, B = 3 ∧ B + 4 = S + 2 ∧ S = 5 :=
by
  use 5
  use 3
  split
  . refl
  split
  . norm_num
  . refl

end storks_initial_count_l141_141750


namespace range_of_a_for_monotonic_function_l141_141939

theorem range_of_a_for_monotonic_function :
  ∀ (a : ℝ), (∀ x y : ℝ, 1 ≤ x → x ≤ 2 → 1 ≤ y → y ≤ 2 → x ≤ y → (f(x) ≤ f(y) ∨ f(y) ≤ f(x)))
  ↔ (a ≤ 1 ∨ a ≥ 2) :=
by
  let f := λ x : ℝ, x^2 - 2 * a * x - 3
  sorry

end range_of_a_for_monotonic_function_l141_141939


namespace prime_set_min_four_primes_l141_141782

noncomputable def prime_set_q (q : Set ℕ) : Prop :=
  (∀ a ∈ q, Nat.Prime a) ∧
  (Σ q % 2 = 0) ∧
  (∃ x ∈ q, x = 3)

theorem prime_set_min_four_primes (q : Set ℕ) (x : ℕ) :
  prime_set_q q →
  (∀ y, y ∈ q → Nat.Prime y) →
  (Σ q - Σ x = 0 → x ∈ q) →
  (x = 3) →
  q.card ≥ 4 :=
sorry

end prime_set_min_four_primes_l141_141782


namespace existence_of_finite_linear_intervals_l141_141222

noncomputable def linear_piecewise_function (f : ℝ → ℝ) (K : ℝ) :=
∀ (x y : ℝ), x ∈ [0,1] ∧ y ∈ [0,1] → |f x - f y| ≤ K * |x - y|

theorem existence_of_finite_linear_intervals
  (f : ℝ → ℝ)
  (K : ℝ)
  (h1 : K > 0)
  (lip : linear_piecewise_function f K)
  (h2 : ∀ r ∈ (set.Icc 0 1 : set ℝ), ∃ (a b : ℤ), f r = a + b * r) :
  ∃ (n : ℕ) (I : fin n → set ℝ), (∀ i, I i ⊆ [0, 1]) ∧ (∀ i, is_interval (I i)) ∧ (∀ i, is_linear_on f (I i)) ∧ (set.Icc 0 1 ⊆ ⋃ i, I i) :=
sorry

end existence_of_finite_linear_intervals_l141_141222


namespace red_more_than_yellow_l141_141742

-- Define the total number of marbles
def total_marbles : ℕ := 19

-- Define the number of yellow marbles
def yellow_marbles : ℕ := 5

-- Calculate the number of remaining marbles
def remaining_marbles : ℕ := total_marbles - yellow_marbles

-- Define the ratio of blue to red marbles
def blue_ratio : ℕ := 3
def red_ratio : ℕ := 4

-- Calculate the sum of ratio parts
def sum_ratio : ℕ := blue_ratio + red_ratio

-- Calculate the number of shares per ratio part
def share_per_part : ℕ := remaining_marbles / sum_ratio

-- Calculate the number of red marbles
def red_marbles : ℕ := red_ratio * share_per_part

-- Theorem to prove: the difference between red marbles and yellow marbles is 3
theorem red_more_than_yellow : red_marbles - yellow_marbles = 3 :=
by
  sorry

end red_more_than_yellow_l141_141742


namespace find_a_from_max_area_l141_141120

theorem find_a_from_max_area (a : ℝ) (h : a > 0) :
  (∀ P : ℝ × ℝ, (P.fst^2 + (P.snd - 3)^2 = a^2) → ∀ A B : ℝ × ℝ,
    A = (2, 0) → B = (-2, 0) → 
    (∃ d : ℝ, d = 3 ∧ ∃ d_max : ℝ, d_max = d + a ∧
    (∀ (area : ℝ), area = 1/2 * 4 * d_max →
    area = 8))
  ) → a = 1 := 
begin
  sorry
end

end find_a_from_max_area_l141_141120


namespace divisible_check_l141_141284

theorem divisible_check (n : ℕ) (h : n = 287) : 
  ¬ (n % 3 = 0) ∧  ¬ (n % 4 = 0) ∧  ¬ (n % 5 = 0) ∧ ¬ (n % 6 = 0) ∧ (n % 7 = 0) := 
by {
  sorry
}

end divisible_check_l141_141284


namespace integral_correct_l141_141057

noncomputable def integral_expr : ℝ → ℝ :=
  λ x, ∫ (2 * x ^ 3 + x + 1) / ((x + 1) * x ^ 3) dx

theorem integral_correct :
  ∀ C : ℝ, integral_expr x = 2 * ln (abs (x + 1)) - 1 / (2 * x ^ 2) + C :=
by 
  sorry

end integral_correct_l141_141057


namespace range_of_F_l141_141882

theorem range_of_F (A B C : ℝ) (h1 : 0 < A) (h2 : A ≤ B) (h3 : B ≤ C) (h4 : C < π / 2) :
  1 + (Real.sqrt 2) / 2 < (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) ∧
  (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) < 2 :=
  sorry

end range_of_F_l141_141882


namespace prime_probability_l141_141336

theorem prime_probability (P : Finset ℕ) : 
  (P = {p | p ≤ 30 ∧ Nat.Prime p}).card = 10 → 
  (Finset.Icc 1 30).card = 30 →
  ((Finset.Icc 1 30).card).choose 2 = 435 →
  (P.card).choose 2 = 45 →
  45 / 435 = 1 / 29 :=
by
  intros hP hIcc30 hChoose30 hChoosePrime
  sorry

end prime_probability_l141_141336


namespace trig_product_identity_l141_141840

-- Define the specific trigonometric angle variables
def pi_over_12 := Real.pi / 12
def five_pi_over_12 := 5 * Real.pi / 12
def seven_pi_over_12 := 7 * Real.pi / 12
def eleven_pi_over_12 := 11 * Real.pi / 12

theorem trig_product_identity :
  (1 + Real.sin pi_over_12) * (1 + Real.sin five_pi_over_12) *
  (1 + Real.sin seven_pi_over_12) * (1 + Real.sin eleven_pi_over_12) = 1 / 16 :=
by {
  sorry
}

end trig_product_identity_l141_141840


namespace sum_of_first_10_terms_l141_141585

variable {a : ℕ → ℝ} -- Define a sequence a_n

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_first_10_terms (h_arith : arithmetic_sequence a)
  (h_condition : a 2 + a 9 = 2) : 
  (∑ i in finset.range 10, a (i + 1)) = 10 :=
sorry

end sum_of_first_10_terms_l141_141585


namespace philip_car_mileage_typical_week_l141_141321

-- Definitions for distances and frequencies
def distance_to_school := 2.5
def distance_to_market := 2
def school_round_trips_per_day := 2
def school_days_per_week := 4
def market_trips_per_week := 1

-- Calculate total weekly mileage
def total_weekly_mileage := 
  let school_trip := distance_to_school * 2
  let market_trip := distance_to_market * 2
  let weekly_school_miles := school_trip * school_round_trips_per_day * school_days_per_week
  let weekly_market_miles := market_trip * market_trips_per_week
  weekly_school_miles + weekly_market_miles

-- Theorem statement for the proof problem
theorem philip_car_mileage_typical_week : total_weekly_mileage = 44 := by
  sorry

end philip_car_mileage_typical_week_l141_141321


namespace prob_leq_zero_l141_141534

noncomputable def normal_distribution (μ σ : ℝ) := measure_theory.probability_measure (measure_theory.gaussian μ σ)

variables {σ : ℝ} (ξ : ℝ → ℝ) (h₁ : ∀ x, ξ x = measure_theory.expectation (measure_theory.gaussian 2 σ)) (h₂ : measure_theory.measure (set.Iic ξ 4) = 0.84)

theorem prob_leq_zero : measure_theory.measure (set.Iic ξ 0) = 0.16 :=
  sorry

end prob_leq_zero_l141_141534


namespace minimum_omega_formula_l141_141657

noncomputable def minimumOmega (f : ℝ → ℝ) (x T ω φ : ℝ) : ℝ :=
  if (f = λ x, Real.cos (ω * x + φ)) 
     ∧ (ω > 0) 
     ∧ (0 < φ < π)
     ∧ (f T = √3 / 2)
     ∧ (f (π / 9) = 0) 
     ∧ (T = 2 * π / ω) 
  then 3 else 0

theorem minimum_omega_formula {f : ℝ → ℝ} {x T ω φ : ℝ} :
  (f = λ x, Real.cos (ω * x + φ)) 
  ∧ (ω > 0) 
  ∧ (0 < φ < π)
  ∧ (f T = √3 / 2)
  ∧ (f (π / 9) = 0)
  ∧ (T = 2 * π / ω) 
  → minimumOmega f x T ω φ = 3 :=
by
  sorry

end minimum_omega_formula_l141_141657


namespace math_problem_l141_141910

theorem math_problem
  (a b x y : ℝ)
  (h1 : a + b = 2)
  (h2 : x + y = 2)
  (h3 : ax + by = 5) :
  (a^2 + b^2) * x * y + a * b * (x^2 + y^2) = -5 :=
by
  sorry

end math_problem_l141_141910


namespace complement_intersection_l141_141952

noncomputable def U : Set ℤ := {-1, 0, 2}
noncomputable def A : Set ℤ := {-1, 2}
noncomputable def B : Set ℤ := {0, 2}
noncomputable def C_U_A : Set ℤ := U \ A

theorem complement_intersection :
  (C_U_A ∩ B) = {0} :=
by {
  -- sorry to skip the proof part as per instruction
  sorry
}

end complement_intersection_l141_141952


namespace relationship_between_A_and_B_l141_141892

theorem relationship_between_A_and_B (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  let A := a^2
  let B := 2 * a - 1
  A > B :=
by
  let A := a^2
  let B := 2 * a - 1
  sorry

end relationship_between_A_and_B_l141_141892


namespace area_ratio_in_triangle_l141_141593

theorem area_ratio_in_triangle (A B C D E : Type*) [EuclideanGeometry] [Triangle A B C]
  (h1 : AC = 3 * AB)       -- Given condition: AC = 3 * AB
  (h2 : AD_bisects_BAC)    -- Given condition: AD bisects angle BAC
  (h3 : D_on_BC)           -- Given condition: D lies on BC
  (h4 : E_perpendicular_C_to_AD) -- Given condition: E is the foot of the perpendicular from C to AD
  : ∃ r : ℝ, r = 1/3 ∧ ratio_areas ABD CDE = r := sorry

end area_ratio_in_triangle_l141_141593


namespace correct_choice_is_B_l141_141441

def draw_ray := "Draw ray OP=3cm"
def connect_points := "Connect points A and B"
def draw_midpoint := "Draw the midpoint of points A and B"
def draw_distance := "Draw the distance between points A and B"

-- Mathematical function to identify the correct statement about drawing
def correct_drawing_statement (s : String) : Prop :=
  s = connect_points

theorem correct_choice_is_B :
  correct_drawing_statement connect_points :=
by
  sorry

end correct_choice_is_B_l141_141441


namespace polynomial_real_root_condition_l141_141069

theorem polynomial_real_root_condition (b : ℝ) :
    (∃ x : ℝ, x^4 + b * x^3 + x^2 + b * x - 1 = 0) ↔ (b ≥ 1 / 2) :=
by sorry

end polynomial_real_root_condition_l141_141069


namespace equal_charges_at_x_l141_141426

theorem equal_charges_at_x (x : ℝ) : 
  (2.75 * x + 125 = 1.50 * x + 140) → (x = 12) := 
by
  sorry

end equal_charges_at_x_l141_141426


namespace num_correct_statements_is_three_l141_141538

noncomputable def check_statements : Nat :=
  let s1 : Prop := ∅ ⊆ ({0} : Set Nat)
  let s2 : Prop := (√2 : ℝ) ∈ (Set ℚ)
  let s3 : Prop := (3 : ℤ) ∈ ({3, -3} : Set ℤ)
  let s4 : Prop := (0 : ℤ) ∈ (Set ℤ)
  let correct_count := [s1, s3, s4].filter id |>.length
  correct_count

theorem num_correct_statements_is_three : check_statements = 3 := by
  sorry

end num_correct_statements_is_three_l141_141538


namespace triangle_area_l141_141433

-- Define the vertices of the triangle
def point_A : (ℝ × ℝ) := (0, 0)
def point_B : (ℝ × ℝ) := (8, -3)
def point_C : (ℝ × ℝ) := (4, 7)

-- Function to compute the area of a triangle given its vertices
def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Conjecture the area of triangle ABC is 30.0 square units
theorem triangle_area : area_of_triangle point_A point_B point_C = 30.0 := by
  sorry

end triangle_area_l141_141433


namespace engineer_check_progress_after_200_days_l141_141042

/-- Prove that the engineer checked the progress after 200 days --/
theorem engineer_check_progress_after_200_days :
  ∃ x : ℝ, 
    let total_length : ℝ := 15;
    let total_days : ℝ := 300;
    let initial_men : ℝ := 35;
    let completed_length : ℝ := 2.5;
    let extra_men : ℝ := 52.5;
    let new_total_men : ℝ := initial_men + extra_men;
    let initial_rate : ℝ := total_length / total_days;
    let work_done_in_x_days : ℝ := x * initial_rate / initial_men;
    let remaining_work : ℝ := total_length - completed_length;
    let remaining_days : ℝ := total_days - x;
    let new_rate : ℝ := new_total_men * initial_rate / initial_men in
    completed_length / x = work_done_in_x_days / x ∧
    remaining_work / remaining_days = new_total_men / initial_men * initial_rate / initial_men ∧
    total_length - completed_length = remaining_work ∧ 
    remaining_days = total_days - x ∧
    x = 200 :=
sorry

end engineer_check_progress_after_200_days_l141_141042


namespace actual_rankings_l141_141885

-- Define the teams
inductive Team
| A | B | C | D

open Team

-- Define the ranking function and the conditions
def ranking : Team → ℕ :=
| B => 1
| A => 2
| C => 3
| D => 4

axiom Saiyangyang : (ranking A = 1 ∨ ranking D = 4) ∧ ¬(ranking A = 1 ∧ ranking D = 4)
axiom Xiyangyang : (ranking D = 2 ∨ ranking C = 3) ∧ ¬(ranking D = 2 ∧ ranking C = 3)
axiom Feiyangyang : (ranking C = 2 ∨ ranking B = 1) ∧ ¬(ranking C = 2 ∧ ranking B = 1)

theorem actual_rankings : ∀ t, ranking t = 
    match t with 
    | B => 1
    | A => 2
    | C => 3
    | D => 4 :=
by
  intros
  sorry

end actual_rankings_l141_141885


namespace brass_players_10_l141_141723

theorem brass_players_10:
  ∀ (brass woodwind percussion : ℕ),
    brass + woodwind + percussion = 110 →
    percussion = 4 * woodwind →
    woodwind = 2 * brass →
    brass = 10 :=
by
  intros brass woodwind percussion h1 h2 h3
  sorry

end brass_players_10_l141_141723


namespace max_value_of_sum_max_value_achievable_l141_141693

theorem max_value_of_sum (a b c d : ℝ) 
  (h : a^6 + b^6 + c^6 + d^6 = 64) : a^7 + b^7 + c^7 + d^7 ≤ 128 :=
sorry

theorem max_value_achievable : ∃ a b c d : ℝ,
  a^6 + b^6 + c^6 + d^6 = 64 ∧ a^7 + b^7 + c^7 + d^7 = 128 :=
sorry

end max_value_of_sum_max_value_achievable_l141_141693


namespace volume_of_prism_is_correct_l141_141197

-- Conditions: each square has side length 2
def side_length : ℝ := 2

-- Defining area of square and triangle based on given length
def area_of_square : ℝ := side_length ^ 2
def area_of_triangle : ℝ := (1/2) * (side_length / 2) * (side_length / 2)

-- Base area of the pentagon consists of a square minus a right-angled triangle
def base_area_of_pentagon : ℝ := area_of_square - area_of_triangle

-- The height of the prism is given in the problem
def height_of_prism : ℝ := side_length

-- Volume of the pentagonal prism
def volume_of_pentagonal_prism : ℝ := base_area_of_pentagon * height_of_prism

-- Problem statement: prove that the volume of the pentagonal prism equals 7
theorem volume_of_prism_is_correct : volume_of_pentagonal_prism = 7 := by
  sorry

end volume_of_prism_is_correct_l141_141197


namespace philip_car_mileage_typical_week_l141_141320

-- Definitions for distances and frequencies
def distance_to_school := 2.5
def distance_to_market := 2
def school_round_trips_per_day := 2
def school_days_per_week := 4
def market_trips_per_week := 1

-- Calculate total weekly mileage
def total_weekly_mileage := 
  let school_trip := distance_to_school * 2
  let market_trip := distance_to_market * 2
  let weekly_school_miles := school_trip * school_round_trips_per_day * school_days_per_week
  let weekly_market_miles := market_trip * market_trips_per_week
  weekly_school_miles + weekly_market_miles

-- Theorem statement for the proof problem
theorem philip_car_mileage_typical_week : total_weekly_mileage = 44 := by
  sorry

end philip_car_mileage_typical_week_l141_141320


namespace length_AB_of_triangle_ABC_l141_141206

theorem length_AB_of_triangle_ABC 
  (A B C : Type) [euclidean_geometry A B C]
  (angle_A_eq_90 : ∠ A = 90)
  (BC_eq_20 : BC = 20)
  (tan_C_eq_3_cos_B : tan C = 3 * cos B) : 
  AB = 40 * sqrt 2 / 3 :=
sorry

end length_AB_of_triangle_ABC_l141_141206


namespace num_correct_propositions_l141_141933

open_locale vector_space

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (p : V → V → Prop)
variables (a b c : V)
variables (A B C D : V)

/-- Condition 1: Parallelism is transitive if no vector is zero -/
def condition1 := ∀ (a b c : V), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) → (p a b ∧ p b c → p a c)

/-- Condition 2: Non-collinear vectors are both non-zero -/
def condition2 := ∀ (a b : V), ¬p a b → (a ≠ 0 ∧ b ≠ 0)

/-- Condition 3: Triangle rule of vectors -/
def condition3 := A ≠ B ∧ B ≠ C ∧ C ≠ A → A + B + C = 0

/-- Condition 4: Parallelogram condition for vectors -/
def condition4 := (p A B ↔ p C D) → ∃ x y z w : V, x ≠ y ∧ y ≠ z ∧ z ≠ w ∧ (x + z = 0)

theorem num_correct_propositions : 
  (condition1 p a b c) ∧ (condition2 p a b) ∧ (condition3) ∧ (condition4) → 
  2 = 3 := 
  sorry

end num_correct_propositions_l141_141933


namespace number_of_true_propositions_l141_141930

theorem number_of_true_propositions:
  (∀ x : ℝ, x^2 + 1 > 0) ∧
  (¬ ∀ x : ℕ, x^4 ≥ 1) ∧
  (∃ x : ℤ, x^3 < 1) ∧
  (∀ x : ℚ, x^2 ≠ 2) →
  3 = 3 :=
by
  intros h,
  sorry

end number_of_true_propositions_l141_141930


namespace new_acute_angle_ACB_l141_141290

theorem new_acute_angle_ACB (θ : ℝ) (rot_deg : ℝ) (h1 : θ = 60) (h2 : rot_deg = 600) :
  (new_acute_angle (rotate_ray θ rot_deg)) = 0 := 
sorry

end new_acute_angle_ACB_l141_141290


namespace independence_test_applicable_l141_141330

-- Define the conditions given in the problem
structure PopulationData :=
  (total_male : ℕ)
  (sick_male : ℕ)
  (total_female : ℕ)
  (sick_female : ℕ)

-- Given conditions
def problem_data : PopulationData :=
  { total_male := 55,
    sick_male := 24,
    total_female := 34,
    sick_female := 8 }

-- Theorem statement
theorem independence_test_applicable (data : PopulationData) : 
  data = problem_data → "Independence test is the appropriate data analysis method" :=
by
  intro h,
  sorry

end independence_test_applicable_l141_141330


namespace coordinates_of_N_l141_141531

-- Define the points and conditions
structure Point where
  x : ℝ
  y : ℝ

def length (A B : Point) : ℝ :=
  Real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

theorem coordinates_of_N (M N : Point) :
  M.x = 2 ∧ M.y = -4 ∧ length M N = 5 ∧ N.y = M.y → (N.x = -3 ∧ N.y = -4) ∨ (N.x = 7 ∧ N.y = -4) :=
by
  intro h
  cases h with hx1 hrest
  cases hrest with hy1 hrest
  cases hrest with hl hye
  sorry

end coordinates_of_N_l141_141531


namespace triangle_PNM_perimeter_approx19_l141_141713

-- Define the conditions given in the problem.
def circle_center : Point := sorry -- Define the center O of the circle.
def radius : ℝ := 8
def angle_QOP : Angle := π / 2 -- 90 degrees
def midpoint_M (P O : Point) : Point := midpoint P O -- Assume a function midpoint
def lies_on_arc (N : Point) (P Q : Point) : Prop := sorry -- Define the arc PQ
def perpendicular (MN OP : Line) : Prop := isPerp MN OP -- Assume a function isPerp

noncomputable def perimeter_of_trianglePNM_closest_to_target 
    (P Q O N M : Point)
    (h1 : dist O P = radius)
    (h2 : dist O Q = radius)
    (h3 : angle O Q P = angle_QOP)
    (h4 : M = midpoint_M P O)
    (h5 : lies_on_arc N P Q)
    (h6 : perpendicular (segment M N) (segment O P)) : 
    Prop :=
  abs ((dist P N + dist N M + dist M P) - 19) < 1

-- Statement of the proof problem
theorem triangle_PNM_perimeter_approx19
    (P Q O N M: Point)
    (h1 : dist O P = radius)
    (h2 : dist O Q = radius)
    (h3 : angle O Q P = angle_QOP)
    (h4 : M = midpoint_M P O)
    (h5 : lies_on_arc N P Q)
    (h6 : perpendicular (segment M N) (segment O P)) :
    perimeter_of_trianglePNM_closest_to_target P Q O N M h1 h2 h3 h4 h5 h6 :=
sorry

end triangle_PNM_perimeter_approx19_l141_141713


namespace square_perimeter_l141_141018

theorem square_perimeter (s : ℝ) (h1 : ∀ r : ℝ, r = 2 * (s + s / 4)) (r_perimeter_eq_40 : r = 40) :
  4 * s = 64 := by
  sorry

end square_perimeter_l141_141018


namespace minimum_gloves_needed_l141_141283

-- Definitions based on conditions:
def participants : Nat := 43
def gloves_per_participant : Nat := 2

-- Problem statement proving the minimum number of gloves needed
theorem minimum_gloves_needed : participants * gloves_per_participant = 86 := by
  -- sorry allows us to omit the proof, focusing only on the formal statement
  sorry

end minimum_gloves_needed_l141_141283


namespace length_AB_of_triangle_ABC_l141_141207

theorem length_AB_of_triangle_ABC 
  (A B C : Type) [euclidean_geometry A B C]
  (angle_A_eq_90 : ∠ A = 90)
  (BC_eq_20 : BC = 20)
  (tan_C_eq_3_cos_B : tan C = 3 * cos B) : 
  AB = 40 * sqrt 2 / 3 :=
sorry

end length_AB_of_triangle_ABC_l141_141207


namespace most_likely_gender_distribution_l141_141751

theorem most_likely_gender_distribution :
  let outcomes := ["all_3_boys", "all_3_girls", "2_girls_1_boy", "2_boys_1_girl"]
  ∃ (prob : string → ℚ),
    (∀ x ∈ outcomes, prob x = 
      if x = "all_3_boys" ∨ x = "all_3_girls" then
        1/8
      else if x = "2_girls_1_boy" ∨ x = "2_boys_1_girl" then
        3/8
      else
        0) 
    ∧
    (prob "2_girls_1_boy" = 3/8 ∧ prob "2_boys_1_girl" = 3/8
    ∧ prob "all_3_boys" < 3/8 ∧ prob "all_3_girls" < 3/8) :=
by
  sorry

end most_likely_gender_distribution_l141_141751


namespace eq_of_ellipse_eq_of_line_no_three_distinct_points_l141_141517

-- Define given conditions for the ellipse
def focal_length := 2 * sqrt 3
def a := 2
def b := 1
def c := sqrt 3

-- Define the equation of the standard ellipse
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

-- Definition of having property H
def has_property_H (l : ℝ → ℝ × ℝ → Prop) : Prop :=
  ∃ A B M : ℝ × ℝ,
    l A ∧ l B ∧ ellipse_eq M.1 M.2 ∧ M = ((3 : ℝ) / (5 : ℝ)) • A + ((4 : ℝ) / (5 : ℝ)) • B

-- Prove the equation of the ellipse
theorem eq_of_ellipse : ellipse_eq = λ x y, (x^2 / 4) + y^2 = 1 := by
  sorry

-- Prove equations for line l
theorem eq_of_line {l : ℝ → ℝ × ℝ → Prop} (hl : ∀ x, l (x, y).1 (x, y).2)
  (h_perpendicular : ∀ p, (l (p.1, p.2)).1 p.2 = p.1) :
  (l = λ p, p.1 = sqrt 2) ∨ (l = λ p, p.1 = -sqrt 2) := by
  sorry

-- Prove the non-existence of three distinct points with property H
theorem no_three_distinct_points (P Q R : ℝ × ℝ) (hP : ellipse_eq P.1 P.2)
  (hQ : ellipse_eq Q.1 Q.2) (hR : ellipse_eq R.1 R.2)
  (hPQ : has_property_H (λ p, ∃ q, q = (P, Q)))
  (hQR : has_property_H (λ p, ∃ q, q = (Q, R)))
  (hRP : has_property_H (λ p, ∃ q, q = (R, P))) :
  False := by
  sorry

end eq_of_ellipse_eq_of_line_no_three_distinct_points_l141_141517


namespace average_ducks_l141_141816

theorem average_ducks (a e k : ℕ) 
  (h1 : a = 2 * e) 
  (h2 : e = k - 45) 
  (h3 : a = 30) :
  (a + e + k) / 3 = 35 :=
by
  sorry

end average_ducks_l141_141816


namespace g_g_g_20_l141_141668

def g (x : ℝ) : ℝ :=
if x < 10 then x^2 - 9 else x - 18

theorem g_g_g_20 : g (g (g 20)) = 16 := by
  sorry

end g_g_g_20_l141_141668


namespace least_four_digit_integer_has_3_7_11_as_factors_l141_141364

theorem least_four_digit_integer_has_3_7_11_as_factors :
  ∃ x : ℕ, (1000 ≤ x ∧ x < 10000) ∧ (3 ∣ x) ∧ (7 ∣ x) ∧ (11 ∣ x) ∧ x = 1155 := by
  sorry

end least_four_digit_integer_has_3_7_11_as_factors_l141_141364


namespace condition_perpendicular_lines_to_plane_necessary_not_sufficient_l141_141111

variables {L : Line} {a : Plane}

def is_perpendicular_to_countless_lines_in_plane
(line L : Line) (plane a : Plane) : Prop := 
  ∃ (lines : set Line), ∀ l ∈ lines, l ∈ a ∧ L ⊥ l

def is_perpendicular_to_plane
(line L : Line) (plane a : Plane) : Prop := 
  ∀ (l : Line), l ∈ a → L ⊥ l

theorem condition_perpendicular_lines_to_plane_necessary_not_sufficient : 
  (is_perpendicular_to_countless_lines_in_plane L a) → 
  (is_perpendicular_to_plane L a) ↔ false := 
sorry

end condition_perpendicular_lines_to_plane_necessary_not_sufficient_l141_141111


namespace f_value_pi_6_l141_141139

def f (x : ℝ) (ω : ℝ) (φ : ℝ) := 2 * Real.sin (ω * x + φ)

theorem f_value_pi_6 (ω φ : ℝ) (h : ∀ x, f (π / 6 + x) ω φ = f (π / 6 - x) ω φ) : 
  f (π / 6) ω φ = 2 ∨ f (π / 6) ω φ = -2 :=
by
  sorry

end f_value_pi_6_l141_141139


namespace parallel_lines_from_perpendicular_to_plane_l141_141092

variables {α : Type*} (plane : α → Prop) (line m n : α → α → Prop) 

-- Conditions definitions
def angles_equal_with_plane (m n : α → α → Prop) (p : α → Prop) : Prop := 
  ∃ (angle : α → ℝ), ∀ (a b : α), plane (a) ∧ plane(b) ∧ m a b ∧ n a b  → angle a = angle b 

def parallel_to_plane (l : α → α → Prop) (p : α → Prop) : Prop :=
  ∀ (a b : α), p(a) ∧ l(a, b) → p(b)

def perpendicular_to_line (m : α → α → Prop) (n : α → α → Prop) : Prop :=
  ∀ (a b : α), m(a, b) ∧ m(a, b) → n(b, a)

def perpendicular_to_plane (l : α → α → Prop) (p : α → Prop) : Prop :=
  ∀ (a b : α), l(a, b) → ¬ p(a) ∧ ¬ p(b)

def parallel_lines (m n : α → α → Prop) : Prop :=
  ∀ (a b c d : α), m(a,b) → n(c,d) → a ≠ b → c ≠ d → (m(a,d) ∧ n(a, d))

-- Lean statement translating the proof problem
theorem parallel_lines_from_perpendicular_to_plane (plane : α → Prop) (m n : α → α → Prop) :
  (perpendicular_to_plane m plane) ∧ (perpendicular_to_plane n plane)  → (parallel_lines m n) := by
  sorry

end parallel_lines_from_perpendicular_to_plane_l141_141092


namespace ratio_side_lengths_sum_l141_141301

theorem ratio_side_lengths_sum (a b c : ℕ) 
  (h1 : (sqrt 75 / sqrt 128 = a * sqrt b / c)) 
  (h2 : (∀ x, x ∣ 75 * 128 → (x ∣ 128 * (a^2 * b - c^2))) ) : 
  a + b + c = 27 := 
sorry

end ratio_side_lengths_sum_l141_141301


namespace curve_C_parametric_eqn_max_value_3x_4y_l141_141944

open Real

noncomputable def polar_eq_ρ := λ (θ : ℝ), 2 * (sin θ + cos θ + 1 / (ρ θ))

noncomputable def parametric_eqn_C (θ : ℝ) : ℝ × ℝ := (1 + 2 * cos θ, 1 + 2 * sin θ)

theorem curve_C_parametric_eqn :
  ∀ θ, parametric_eqn_C θ = (1 + 2 * cos θ, 1 + 2 * sin θ) :=
by sorry

theorem max_value_3x_4y :
  let P := parametric_eqn_C
  in ∀ θ ∈ [0, 2 * π), 3 * (P θ).1 + 4 * (P θ).2 ≤ 17 :=
by sorry

end curve_C_parametric_eqn_max_value_3x_4y_l141_141944


namespace weekly_car_mileage_l141_141322

-- Definitions of the conditions
def dist_school := 2.5 
def dist_market := 2 
def school_days := 4
def school_trips_per_day := 2
def market_trips_per_week := 1

-- Proof statement
theorem weekly_car_mileage : 
  4 * 2 * (2.5 * 2) + (1 * (2 * 2)) = 44 :=
by
  -- The goal is to prove that 4 days of 2 round trips to school plus 1 round trip to market equals 44 miles
  sorry

end weekly_car_mileage_l141_141322


namespace fraction_of_white_surface_area_l141_141404

def larger_cube_edge : ℕ := 4
def number_of_smaller_cubes : ℕ := 64
def number_of_white_cubes : ℕ := 8
def number_of_red_cubes : ℕ := 56
def total_surface_area : ℕ := 6 * (larger_cube_edge * larger_cube_edge)
def minimized_white_surface_area : ℕ := 7

theorem fraction_of_white_surface_area :
  minimized_white_surface_area % total_surface_area = 7 % 96 :=
by
  sorry

end fraction_of_white_surface_area_l141_141404


namespace snail_distance_after_96_days_l141_141421

/-- On the n-th day, the snail moves forward by 1/n meters and backward by 1/(n+1) meters. -/
/-- We want to prove that the total distance the snail is from the starting point after 96 days is 96/97 meters. -/
theorem snail_distance_after_96_days :
  (∑ n in nat.range 96, (1 / (n + 1 : ℝ) - 1 / (n + 2 : ℝ))) = 96 / 97 := sorry

end snail_distance_after_96_days_l141_141421


namespace student_average_marks_first_exam_l141_141182

theorem student_average_marks_first_exam :
  ∀ (x : ℕ), 
  (∀ (y z : ℕ), 0.55 * 500 = y ∧ 100 = z → 3 * 500 = 1500 ∧ 0.4 * 1500 = 600 → x + y + z = 600) →
  x = 225 :=
by
  intro x
  intro h
  specialize h 275 100
  have hy : 0.55 * 500 = 275 := by norm_num
  have hz : 100 = 100 := by norm_num
  have htotal : 3 * 500 = 1500 := by norm_num
  have htotal_percent : 0.4 * 1500 = 600 := by norm_num
  specialize h hy hz htotal htotal_percent
  exact eq_of_heq h

end student_average_marks_first_exam_l141_141182


namespace construct_asymptotes_l141_141130

variable (O : Point) (x y : Line) (P1 P2 : Point)
variable (P1_ne_P2 : P1 ≠ P2)
variable (P1_ne_O : P1 ≠ O)
variable (P2_ne_O : P2 ≠ O)
variable (not_parallel_x : ¬ Parallel (P1P2) x)
variable (not_parallel_y : ¬ Parallel (P1P2) y)
variable (not_through_O : ¬ Contains O P1P2)

theorem construct_asymptotes : AsymptotesExist O x y P1 P2 :=
by sorry

end construct_asymptotes_l141_141130


namespace average_ducks_l141_141815

theorem average_ducks (a e k : ℕ) 
  (h1 : a = 2 * e) 
  (h2 : e = k - 45) 
  (h3 : a = 30) :
  (a + e + k) / 3 = 35 :=
by
  sorry

end average_ducks_l141_141815


namespace find_ck_l141_141308

theorem find_ck 
  (d r : ℕ)                -- d : common difference, r : common ratio
  (k : ℕ)                  -- k : integer such that certain conditions hold
  (hn2 : (k-2) > 0)        -- ensure (k-2) > 0
  (hk1 : (k+1) > 0)        -- ensure (k+1) > 0
  (h1 : 1 + (k-3) * d + r^(k-3) = 120) -- c_{k-1} = 120
  (h2 : 1 + k * d + r^k = 1200) -- c_{k+1} = 1200
  : (1 + (k-1) * d + r^(k-1)) = 263 := -- c_k = 263
sorry

end find_ck_l141_141308


namespace sufficient_condition_l141_141146

theorem sufficient_condition (a : ℝ) :
  (∀ x ∈ Set.Icc (-4 : ℝ) 2, (1/2 : ℝ) * x^2 - a ≥ 0) → a ≤ 0 :=
by
  sorry

end sufficient_condition_l141_141146


namespace problem_solution_l141_141581

-- Conditions
variables {a b c : ℝ} (P F₁ F₂ : Point) (k m : ℝ)
-- Configuration of ellipse C: x²/a² + y²/b² = 1, a > b > 0
def ellipse_C := {P : Point | P.1^2 / (a^2) + P.2^2 / (b^2) = 1 ∧ a > b ∧ b > 0}
-- Eccentricity condition: c/a = √3/2, and relation a² - c² = b²
def eccentricity_condition := (c / a = √3 / 2) ∧ (a^2 - c^2 = b^2)
-- Circle intersection conditions
def circle_intersection := (circle F₁ 3 ∩ circle F₂ 1).nonempty ∧ (circle F₁ 3 ∩ circle F₂ 1) ⊆ ellipse_C

-- Translated Proof Problem
theorem problem_solution :
  ∃ a b c, (a = 2) ∧ (b = 1) ∧
  (ellipse_C ⟨a, b, 0⟩) ∧
  (eccentricity_condition ⟨a, b, c⟩) ∧
  (circle_intersection) ∧
  (∃ (P : Point), (P ∈ ellipse_C ⟨a, b, 0⟩) →
   let Q := {Q : Point | Q ∈ ellipse_C ⟨4*a, 4*b, 0⟩} in
   | OQ / OP | = 2) ∧
  (maximum_area_triangle_ABQ = 6√3) :=
begin
  sorry,
end

end problem_solution_l141_141581


namespace solve_for_C_and_D_l141_141097

theorem solve_for_C_and_D (C D : ℚ) (h1 : 2 * C + 3 * D + 4 = 31) (h2 : D = C + 2) :
  C = 21 / 5 ∧ D = 31 / 5 :=
by
  sorry

end solve_for_C_and_D_l141_141097


namespace minimum_distance_of_M_to_N_l141_141570

theorem minimum_distance_of_M_to_N :
  ∀ (x y : ℝ), 
  (sqrt ((x + 3)^2 + y^2) + sqrt ((x - 3)^2 + y^2) = 10) → 
  ∃ M : ℝ × ℝ, M ∈ set_of (λ p : ℝ × ℝ, (√((p.1 + 3)^2 + p.2^2) + √((p.1 - 3)^2 + p.2^2) = 10) ) → 
  ∃ N : ℝ × ℝ, N = (-6, 0) → 
  (inf_dist (λ p : ℝ × ℝ, (√((p.1 + 3)^2 + p.2^2) + √((p.1 - 3)^2 + p.2^2) = 10) (-6, 0) = 1) := by
  sorry

end minimum_distance_of_M_to_N_l141_141570


namespace truncatedPyramidVolume_l141_141810

noncomputable def volumeOfTruncatedPyramid (R : ℝ) : ℝ :=
  let h := R * Real.sqrt 3 / 2
  let S_lower := 3 * R^2 * Real.sqrt 3 / 2
  let S_upper := 3 * R^2 * Real.sqrt 3 / 8
  let sqrt_term := Real.sqrt (S_lower * S_upper)
  (1/3) * h * (S_lower + S_upper + sqrt_term)

theorem truncatedPyramidVolume (R : ℝ) (h := R * Real.sqrt 3 / 2)
  (S_lower := 3 * R^2 * Real.sqrt 3 / 2)
  (S_upper := 3 * R^2 * Real.sqrt 3 / 8)
  (V := (1/3) * h * (S_lower + S_upper + Real.sqrt (S_lower * S_upper))) :
  volumeOfTruncatedPyramid R = 21 * R^3 / 16 := by
  sorry

end truncatedPyramidVolume_l141_141810


namespace measure_of_angle_E_l141_141440

theorem measure_of_angle_E
  (ABCDE : Type)
  (pentagon : ∀ (P : fin 5 → ABCDE), convex_poly P)
  (equal_sides : ∀ (P : fin 5 → ABCDE), sides_equal_length P)
  (angle_A_eq_120 : ∀ (P : fin 5 → ABCDE), angle P 0 = 120)
  (angle_B_eq_120 : ∀ (P : fin 5 → ABCDE), angle P 1 = 120) :
  ∃ P : fin 5 → ABCDE, angle P 4 = 120 := 
sorry

end measure_of_angle_E_l141_141440


namespace john_frank_age_ratio_l141_141886

theorem john_frank_age_ratio
  (F J : ℕ)
  (h1 : F + 4 = 16)
  (h2 : J - F = 15)
  (h3 : ∃ k : ℕ, J + 3 = k * (F + 3)) :
  (J + 3) / (F + 3) = 2 :=
by
  sorry

end john_frank_age_ratio_l141_141886


namespace trig_identity_proofs_l141_141894

theorem trig_identity_proofs (α : ℝ) 
  (h : Real.sin α + Real.cos α = 1 / 5) :
  (Real.sin α - Real.cos α = 7 / 5 ∨ Real.sin α - Real.cos α = -7 / 5) ∧
  (Real.sin α ^ 3 + Real.cos α ^ 3 = 37 / 125) :=
by
  sorry

end trig_identity_proofs_l141_141894


namespace alcohol_water_ratio_l141_141779

theorem alcohol_water_ratio (alcohol water : ℝ) (h_alcohol : alcohol = 3 / 5) (h_water : water = 2 / 5) :
  alcohol / water = 3 / 2 :=
by 
  sorry

end alcohol_water_ratio_l141_141779


namespace range_of_x_l141_141498

theorem range_of_x (a : ℝ) (h : -1 ≤ a ∧ a ≤ 1) (x : ℝ) : 
  (x ^ 2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (x < 1 ∨ x > 3) :=
by 
  sorry

end range_of_x_l141_141498


namespace find_integer_divisible_by_18_l141_141866

theorem find_integer_divisible_by_18 :
  ∃ (n : ℕ), (676 ≤ n ∧ n ≤ 702) ∧ (18 ∣ n) ∧ n = 702 :=
by
  have h_range: 676 ≤ (702 : ℕ) := by norm_num
  have h_div: 18 ∣ 702 := by norm_num
  use 702
  exact ⟨⟨h_range,le_refl 702⟩,⟨h_div,rfl⟩⟩
  sorry

end find_integer_divisible_by_18_l141_141866


namespace train_passing_pole_l141_141432

variables (v L t_platform D_platform t_pole : ℝ)
variables (H1 : L = 500)
variables (H2 : t_platform = 100)
variables (H3 : D_platform = L + 500)
variables (H4 : t_platform = D_platform / v)

theorem train_passing_pole :
  t_pole = L / v := 
sorry

end train_passing_pole_l141_141432


namespace max_b_value_l141_141902

theorem max_b_value (b c a x y : ℝ) (F1 F2 : ℝ × ℝ)
  (ellipse : x ^ 2 + (y ^ 2 / b ^ 2) = 1)
  (foci_distance : (F2.1 - F1.1) = 2 * c)
  (point_P_condition : P : ℝ × ℝ)
  (mean_cond : (P = (x, y)) ∧ ((|P.1 - F1.1| + |P.1 - F2.1|) / 2 = x - 1 / c))
  (range_c : \( c \geq \frac{1}{2} \))
  : b ≤ Real.sqrt 3 / 2 := sorry

end max_b_value_l141_141902


namespace monotonic_intervals_range_of_a_if_fx0_negative_l141_141539

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x + (1 + a) / x - a * Real.log x

theorem monotonic_intervals (a : ℝ) (h : a > -1) :
  (∀ x : ℝ, 0 < x ∧ x < 1 + a → f x a = x + (1 + a) / x - a * Real.log x ∧ f' x < 0) ∧
  (∀ x : ℝ, x > 1 + a → f x a = x + (1 + a) / x - a * Real.log x ∧ f' x > 0) :=
sorry

theorem range_of_a_if_fx0_negative (a : ℝ) (h : a > -1) :
  (∃ x0 : ℝ, x0 ∈ Set.Icc 1 (Real.exp 1) ∧ f x0 a < 0) ↔ a ∈ Set.Ioi ((Real.exp 2 + 1) / (Real.exp 1 - 1)) :=
sorry

end monotonic_intervals_range_of_a_if_fx0_negative_l141_141539


namespace limit_at_minus_one_third_l141_141681

theorem limit_at_minus_one_third : 
  ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧ 
  (∀ (x : ℝ), 0 < |x + 1 / 3| ∧ |x + 1 / 3| < δ → 
  |(9 * x^2 - 1) / (x + 1 / 3) + 6| < ε) :=
sorry

end limit_at_minus_one_third_l141_141681


namespace limit_sequence_equiv_l141_141385

noncomputable def problem_limit_sequence : Real :=
  lim (fun n : ℕ => ((2 * n + 1) ^ 3 - (2 * n + 3) ^ 3) / ((2 * n + 1) ^ 2 + (2 * n + 3) ^ 2))

theorem limit_sequence_equiv (L : Real) (hL : L = problem_limit_sequence) :
  L = -15 / 8 := sorry

end limit_sequence_equiv_l141_141385


namespace find_n_l141_141935

noncomputable def f (x : ℝ) : ℝ := (Real.log 3 / Real.log 2)^x + x - Real.log (2 : ℝ) / Real.log (3 : ℝ)

theorem find_n (a b n : ℤ) (hb : 3^b = 2) (ha : 2^a = 3) (hroot : ∃ x_b : ℝ, x_b ∈ (n, n+1) ∧ f x_b = 0) : n = -1 :=
sorry

end find_n_l141_141935


namespace egg_problem_l141_141007

theorem egg_problem :
  ∃ (N F E : ℕ), N + F + E = 100 ∧ 5 * N + F + E / 2 = 100 ∧ (N = F ∨ N = E ∨ F = E) ∧ N = 10 ∧ F = 10 ∧ E = 80 :=
by
  sorry

end egg_problem_l141_141007


namespace total_cost_l141_141324

variable (a b : ℝ)

theorem total_cost (ha : a ≥ 0) (hb : b ≥ 0) : 3 * a + 4 * b = 3 * a + 4 * b :=
by sorry

end total_cost_l141_141324


namespace number_of_integer_pairs_l141_141226

theorem number_of_integer_pairs :
  let 𝜔 : ℂ := complex.of_real (-1 / 2) + complex.I * (real.sqrt 3) / 2
  in ∀ a b : ℤ, 
    (𝜔 ^ 2 + 𝜔 + 1 = 0 ∧ complex.abs (a * 𝜔 + b) = 2) ↔ (a^2 - a * b + b^2 = 4) :=
begin
  sorry
end

end number_of_integer_pairs_l141_141226


namespace minimum_omega_value_l141_141641

noncomputable def min_omega (ω T x φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π) (hT : T = 2 * π / ω) 
  (hfT : Real.cos (ω * T + φ) = √3 / 2) (hx : Real.cos (ω * x + φ) = 0) : ℝ :=
  3

theorem minimum_omega_value (ω T x φ : ℝ)
  (hω : ω > 0)
  (hφ : 0 < φ ∧ φ < π)
  (hT : T = 2 * π / ω)
  (hfT : Real.cos (ω * T + φ) = √3 / 2)
  (hx : Real.cos (ω * x + φ) = 0)
  (hx_val : x = π / 9)
  (T_val : T = 2 * π / ω) :
  ω = min_omega ω T (π / 9) (π / 6) :=
begin
  sorry -- Proof steps are skipped as instructed.
end

end minimum_omega_value_l141_141641


namespace monotone_f_range_l141_141976

noncomputable def f (a x : ℝ) : ℝ :=
  x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotone_f_range (a : ℝ) :
  (∀ x : ℝ, (1 - (2 / 3) * Real.cos (2 * x) + a * Real.cos x) ≥ 0) ↔ (-1 / 3 ≤ a ∧ a ≤ 1 / 3) := 
sorry

end monotone_f_range_l141_141976


namespace perimeter_of_rectangular_field_l141_141707

theorem perimeter_of_rectangular_field (L B : ℝ) 
    (h1 : B = 0.60 * L) 
    (h2 : L * B = 37500) : 
    2 * L + 2 * B = 800 :=
by 
  -- proof goes here
  sorry

end perimeter_of_rectangular_field_l141_141707


namespace orignal_prices_dont_match_discounted_prices_l141_141444

theorem orignal_prices_dont_match_discounted_prices :
  let prices := [15, 20, 25, 30, 35, 40] in
  let total := prices.sum in
  total ≠ 160 ∧ 
  0.75 * total = 120 → false :=
by
  let prices := [15, 20, 25, 30, 35, 40]
  let total := prices.sum
  have h1 : total = 15 + 20 + 25 + 30 + 35 + 40 := sorry
  have h2 : total = 165 := sorry
  have h3 : 160 ≠ 165 := sorry
  have h4 : 0.75 * total = 0.75 * 165 := sorry
  have h5 : 0.75 * 165 = 123.75 := sorry
  have h6 : 123.75 ≠ 120 := sorry
  exact false.elim (h6 h4)

end orignal_prices_dont_match_discounted_prices_l141_141444


namespace line_intersects_hyperbola_l141_141512

variables (a b : ℝ) (h : a ≠ 0) (k : b ≠ 0)

def line (x y : ℝ) := a * x - y + b = 0

def hyperbola (x y : ℝ) := x^2 / (|a| / |b|) - y^2 / (|b| / |a|) = 1

theorem line_intersects_hyperbola :
  ∃ x y : ℝ, line a b x y ∧ hyperbola a b x y := 
sorry

end line_intersects_hyperbola_l141_141512


namespace min_omega_is_three_l141_141637

open Real

noncomputable def min_omega {ω : ℝ} {ϕ : ℝ} (f : ℝ → ℝ) : ℝ :=
  if f = (λ x, cos (ω * x + ϕ)) then
    if 0 < ω ∧ 0 < ϕ ∧ ϕ < π ∧
       (∃ (T : ℝ), T = 2 * π / ω ∧ f T = sqrt 3 / 2) ∧
       f (π / 9) = 0 then
      ω
    else 0
  else 0

theorem min_omega_is_three :
  let f : ℝ → ℝ := λ x, cos (3 * x + π / 6) in
  min_omega f = 3 := by
  sorry

end min_omega_is_three_l141_141637


namespace proof_problem_1_proof_problem_2_l141_141947

section

variable (a : ℝ) 
-- Proposition p
def p := ∀ x : ℝ, x ≥ 1 → x - x^2 ≤ a

-- Proposition q
def q := ∃ (x : ℝ), x^2 - a * x + 1 = 0

-- Define proof problem 1
theorem proof_problem_1 (h : ¬¬p) : 0 ≤ a :=
    sorry

-- Define proof problem 2
theorem proof_problem_2 (hpqfalse : ¬(p ∧ q)) (hpqtrue : p ∨ q) : a ≤ -2 ∨ (0 ≤ a ∧ a < 2) :=
    sorry

end

end proof_problem_1_proof_problem_2_l141_141947


namespace sequence_properties_l141_141115

def f (x : ℝ) : ℝ := x^3 + 3 * x

variables {a_5 a_8 : ℝ}
variables {S_12 : ℝ}

axiom a5_condition : (a_5 - 1)^3 + 3 * a_5 = 4
axiom a8_condition : (a_8 - 1)^3 + 3 * a_8 = 2

theorem sequence_properties : (a_5 > a_8) ∧ (S_12 = 12) :=
by {
  sorry
}

end sequence_properties_l141_141115


namespace neha_amount_removed_l141_141392

theorem neha_amount_removed (N S M : ℝ) (x : ℝ) (total_amnt : ℝ) (M_val : ℝ) (ratio2 : ℝ) (ratio8 : ℝ) (ratio6 : ℝ) :
  total_amnt = 1100 →
  M_val = 102 →
  ratio2 = 2 →
  ratio8 = 8 →
  ratio6 = 6 →
  (M - 4 = ratio6 * x) →
  (S - 8 = ratio8 * x) →
  (N - (N - (ratio2 * x)) = ratio2 * x) →
  (N + S + M = total_amnt) →
  (N - 32.66 = N - (ratio2 * (total_amnt - M_val - 8 - 4) / (ratio2 + ratio8 + ratio6))) →
  N - (N - (ratio2 * ((total_amnt - M_val - 8 - 4) / (ratio2 + ratio8 + ratio6)))) = 826.70 :=
by
  intros
  sorry

end neha_amount_removed_l141_141392


namespace part1_part2_l141_141544

-- Define the predicate for the inequality
def prop (x m : ℝ) : Prop := x^2 - 2 * m * x - 3 * m^2 < 0

-- Define the set A
def A (m : ℝ) : Prop := m < -2 ∨ m > 2 / 3

-- Define the predicate for the other inequality
def prop_B (x a : ℝ) : Prop := x^2 - 2 * a * x + a^2 - 1 < 0

-- Define the set B in terms of a
def B (x a : ℝ) : Prop := a - 1 < x ∧ x < a + 1

-- Define the propositions required in the problem
theorem part1 (m : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → prop x m) ↔ A m :=
sorry

theorem part2 (a : ℝ) :
  (∀ x, B x a → A x) ∧ (∃ x, A x ∧ ¬ B x a) ↔ (a ≤ -3 ∨ a ≥ 5 / 3) :=
sorry

end part1_part2_l141_141544


namespace number_of_exact_values_l141_141297

theorem number_of_exact_values (r : ℝ) (a b c d : ℕ) :
  0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
  r = 0.abcd ∧ 0.3667 ≤ r ∧ r ≤ 0.4499 ∧ (∀ n ∈ { 1, 2 }, ∀ d ∈ ℕ, ¬(r - (n/d) < abs(r - 0.4))) →
  r ∈ set.range (λ (abcd : ℝ), 0.3667 ≤ abcd ∧ abcd ≤ 0.4499) → 
  ∃ (r_set : ℕ → ℝ), (set.range r_set).count = 833 :=
by sorry

end number_of_exact_values_l141_141297


namespace pagesWithText_l141_141398

def bookPagesTotal := 98
def pagesImages := bookPagesTotal / 2
def pagesIntroduction := 11
def remainingPages := bookPagesTotal - pagesImages - pagesIntroduction
def pagesBlank := remainingPages / 2
def pagesText := remainingPages - pagesBlank

theorem pagesWithText (bookPagesTotal = 98) (pagesImages = bookPagesTotal / 2) (pagesIntroduction = 11) :
  pagesText = 19 := by
  sorry

end pagesWithText_l141_141398


namespace equation_of_curve_under_transformation_l141_141904

open Matrix

def M : Matrix (Fin 2) (Fin 2) ℚ := !![1, 0; 0, 2]
def N : Matrix (Fin 2) (Fin 2) ℚ := !![1/2, 0; 0, 1]

theorem equation_of_curve_under_transformation : 
  ∀ (x y : ℝ), 
  (M ⬝ N) = !![1/2, 0; 0, 2] →
  y = 2 * Real.sin (2 * x) :=
by 
  intros x y h
  sorry

end equation_of_curve_under_transformation_l141_141904


namespace minimum_omega_l141_141617

noncomputable section

-- Define the function f and its properties
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.cos (ω * x + φ)

-- Assumptions based on the given conditions
variables {ω φ : ℝ}
variables (T : ℝ) (hω_pos : 0 < ω) (hφ_range : 0 < φ ∧ φ < π)
variables (hT : f T ω φ = Real.sqrt 3 / 2)
variables (hx_zero : f (π / 9) ω φ = 0)

-- Prove the minimum value of ω is 3
theorem minimum_omega : ω = 3 := sorry

end minimum_omega_l141_141617


namespace find_phi_l141_141609

/-- Let P be the product of the roots of the polynomial z^7 - z^5 + z^4 - z^2 + 1 = 0
that have a positive imaginary part, and suppose that P = r * (cos(phi) + i * sin(phi)),
where r > 0 and 0 <= phi < 360. -/
theorem find_phi :
  let P := ∏ (z : ℂ) in {z | z^7 - z^5 + z^4 - z^2 + 1 = 0 ∧ z.im > 0}, z 
  ∃ (phi : ℝ), 
    P = P.abs * (Complex.exp (phi * Complex.I)) ∧ 
    0 ≤ phi ∧ phi < 360 := 
sorry

end find_phi_l141_141609


namespace find_S2011_l141_141236

def is_arithmetic_seq (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

def partial_sum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ k in (Finset.range (n + 1)), a k

noncomputable def S (a : ℕ → ℚ) (n : ℕ) : ℚ := partial_sum a n

theorem find_S2011 (a : ℕ → ℚ) 
  (h_arith : is_arithmetic_seq a)
  (h_cond : S a 2010 - S a 1 = 1) : S a 2011 = 2011 / 2009 := 
sorry

end find_S2011_l141_141236


namespace largest_k_sum_of_consecutive_l141_141487

theorem largest_k_sum_of_consecutive (k : ℕ) (n : ℕ) :
  (k ≤ 250) ∧ (5^7 = (range k).sum + k * n) :=
sorry

end largest_k_sum_of_consecutive_l141_141487


namespace sum_f_sqrt_l141_141162

def f (x : ℝ) : ℝ := (x^2) / (1 + x^2)

theorem sum_f_sqrt (n : ℕ) (h : n ≥ 1):
  f (sqrt 1) + (∑ i in (range n).filter (λ i, i ≠ 1), f (sqrt i) + f (sqrt (1/i))) = n - 1/2 :=
by
  sorry

end sum_f_sqrt_l141_141162


namespace different_possible_schedules_l141_141419

theorem different_possible_schedules (classes : List String)
  (h1 : classes = ["Chinese", "Mathematics", "English", "Physics", "Chemistry", "Physical Education"])
  (h2 : ∀ sched : List String, sched.length = 6 → 
    (sched.head ≠ "Physical Education") ∧ (sched.nth 3 ≠ some "Mathematics")) :
  ∃ (n : ℕ), n = 504 ∧ 
    (∃ scheds : Finset (List String), scheds.card = n ∧ 
    ∀ sched ∈ scheds, sched.head ≠ "Physical Education" ∧ sched.nth 3 ≠ some "Mathematics") :=
sorry

end different_possible_schedules_l141_141419


namespace exists_integer_with_2002_palindromes_l141_141263

def is_palindrome (b : ℕ) (N : ℕ) : Prop :=
  let digits := Nat.digits b N
  digits = digits.reverse

theorem exists_integer_with_2002_palindromes :
  ∃ N : ℤ, ∃ (b_set : Set ℕ), 
    (∀ b ∈ b_set, (N.valOfDigit b).digits.length = 3
      ∧ is_palindrome b N.valOfDigit b)
    ∧ b_set.card ≥ 2002
:= 
sorry

end exists_integer_with_2002_palindromes_l141_141263


namespace inradius_of_right_triangle_l141_141179

variables {A B C E F : Type}
variable [MetricSpace A]
variable [MetricSpace B]
variable [MetricSpace C]

noncomputable def inradius (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] := sorry
def angle_bisectors_projections (C : Type) (A B E F : Type) := sorry

theorem inradius_of_right_triangle
  (ABC : Type) [MetricSpace ABC]
  (angle_C : ∀ (α β : ℝ), α = β)
  (projections : angle_bisectors_projections (C) (A) (B) (E) (F))
  : ∃ (r : ℝ), ∥E - F∥ = r :=
sorry

end inradius_of_right_triangle_l141_141179


namespace minimum_omega_value_l141_141645

noncomputable def min_omega (ω T x φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π) (hT : T = 2 * π / ω) 
  (hfT : Real.cos (ω * T + φ) = √3 / 2) (hx : Real.cos (ω * x + φ) = 0) : ℝ :=
  3

theorem minimum_omega_value (ω T x φ : ℝ)
  (hω : ω > 0)
  (hφ : 0 < φ ∧ φ < π)
  (hT : T = 2 * π / ω)
  (hfT : Real.cos (ω * T + φ) = √3 / 2)
  (hx : Real.cos (ω * x + φ) = 0)
  (hx_val : x = π / 9)
  (T_val : T = 2 * π / ω) :
  ω = min_omega ω T (π / 9) (π / 6) :=
begin
  sorry -- Proof steps are skipped as instructed.
end

end minimum_omega_value_l141_141645


namespace parallel_sufficient_not_necessary_l141_141113

noncomputable def is_perpendicular_to_plane (l : StraightLine) (α : Plane) : Prop :=
  ∀ m : StraightLine, m ⊆ α → l ⊥ m

noncomputable def is_perpendicular_to_line (l m : StraightLine) : Prop :=
  l ⊥ m

noncomputable def is_parallel (α β : Plane) : Prop :=
  ∀ l : StraightLine, l ⊆ α → ∃ m : StraightLine, m ⊆ β ∧ l ∥ m

theorem parallel_sufficient_not_necessary (l m : StraightLine) (α β : Plane) :
  (is_perpendicular_to_plane l α ∧ m ⊆ β) →
  (is_parallel α β → is_perpendicular_to_line l m) ∧
  (is_perpendicular_to_line l m → ¬ is_parallel α β) :=
by
  sorry

end parallel_sufficient_not_necessary_l141_141113


namespace small_z_value_l141_141586

noncomputable def w (n : ℕ) := n
noncomputable def x (n : ℕ) := n + 1
noncomputable def y (n : ℕ) := n + 2
noncomputable def z (n : ℕ) := n + 4

theorem small_z_value (n : ℕ) 
  (h : w n ^ 3 + x n ^ 3 + y n ^ 3 = z n ^ 3)
  : z n = 9 :=
sorry

end small_z_value_l141_141586


namespace boys_girls_ratio_l141_141730

theorem boys_girls_ratio (b : ℕ) (G : ℕ) : b = 12 → (2 * G = 3 * b) → G = 18 :=
by
  intros hb h
  rw [hb] at h
  exact (nat.mul_right_inj (nat.zero_lt_succ 1)).1 (by linarith : 2 * G = 36)
  sorry

end boys_girls_ratio_l141_141730


namespace buying_more_costs_less_buy_101_to_save_l141_141028

-- Define the cost function based on the number of notebooks
def notebook_cost (n : ℕ) : ℝ :=
  if n > 100 then 2.2 * n else 2.3 * n

-- Prove that it is possible for buying more notebooks to cost less than buying fewer notebooks.
theorem buying_more_costs_less : ∃ n m : ℕ, n < m ∧ notebook_cost m < notebook_cost n :=
by {
  use [100, 101],
  simp [notebook_cost],
  norm_num,
  linarith,
}

-- Prove that if 100 notebooks are needed, buying 101 notebooks saves money.
theorem buy_101_to_save : notebook_cost 100 > notebook_cost 101 :=
by {
  simp [notebook_cost],
  norm_num,
  linarith,
}

end buying_more_costs_less_buy_101_to_save_l141_141028


namespace prime_probability_l141_141337

theorem prime_probability (P : Finset ℕ) : 
  (P = {p | p ≤ 30 ∧ Nat.Prime p}).card = 10 → 
  (Finset.Icc 1 30).card = 30 →
  ((Finset.Icc 1 30).card).choose 2 = 435 →
  (P.card).choose 2 = 45 →
  45 / 435 = 1 / 29 :=
by
  intros hP hIcc30 hChoose30 hChoosePrime
  sorry

end prime_probability_l141_141337


namespace num_points_exists_l141_141850

theorem num_points_exists (n : ℕ) (hn : n > 3) :
  (∃ (A : Fin n → ℝ × ℝ) (r : Fin n → ℝ), 
  (∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k → 
    ¬ collinear ({A i, A j, A k} : Set (ℝ × ℝ))) ∧ 
  (∀ i j k : Fin n, i < j → j < k → 
    area_triangle (A i) (A j) (A k) = r i + r j + r k)) ↔ n = 4 := sorry

end num_points_exists_l141_141850


namespace math_problem_l141_141951

open Set

def A (a : ℝ) : Set ℝ := { x | x^2 - x - a = 0 }
def B : Set ℝ := {2, -5}
def U (a : ℝ) : Set ℝ := A a ∪ B

theorem math_problem (a : ℝ) (hA : 2 ∈ A a) :
  (a = 2 ∧ A a = { -1, 2 }) ∧ ((U a).compl (A a) ∪ (U a).compl (B) = { -5, -1 }) :=
by 
  sorry

end math_problem_l141_141951


namespace find_a_for_quadratic_trinomials_l141_141871

theorem find_a_for_quadratic_trinomials :
  ∃ (a : ℝ), (∀ x : ℝ, x^2 - 6*x + 4*a = 0 → (x - 3 + real.sqrt(9 - 4*4*a)); (x - 3 - real.sqrt(9 - 4*4*a))) ∧
              (∀ y : ℝ, y^2 + a*y + 6 = 0 → (y - -a/2 + real.sqrt(a^2 - 4*6)/2); (y - -a/2 - real.sqrt(a^2 - 4*6)/2)) ∧
              (((6^2 - 8*a) = (a^2 - 12))) := 
begin 
  use -12,
  sorry
end

end find_a_for_quadratic_trinomials_l141_141871


namespace trig_expression_is_neg_one_l141_141377

theorem trig_expression_is_neg_one (α : ℝ) :
    (cos (8 * α) * tan (4 * α) - sin (8 * α)) * (cos (8 * α) * cot (4 * α) + sin (8 * α)) = -1 :=
by
  sorry

end trig_expression_is_neg_one_l141_141377


namespace units_digit_of_expression_l141_141074

noncomputable def sqrt_196 : ℝ := real.sqrt 196

noncomputable def expr : ℕ :=
  let A := 12 + sqrt_196
  let B := 12 - sqrt_196
  let term1 := (A : ℤ)^14
  let term2 := (B : ℤ)^14
  let term3 := (A : ℤ)^101
  ((term1 + term2 + term3) % 10).to_nat

theorem units_digit_of_expression :
  expr = 2 :=
by
  have h1 : sqrt_196 = 14 := by sorry
  have hA : (12 + sqrt_196 : ℤ) = 26 := by sorry
  have hB : (12 - sqrt_196 : ℤ) = 10 := by sorry
  rw [h1, hA, hB]
  have term1_mod10 : (26^14 % 10).to_nat = 6 := by sorry
  have term2_mod10 : (10^14 % 10).to_nat = 0 := by sorry
  have term3_mod10 : (26^101 % 10).to_nat = 6 := by sorry
  rw [term1_mod10, term2_mod10, term3_mod10]
  calc
    ((6 + 0 + 6) % 10).to_nat = (12 % 10).to_nat := by rw add_comm
    ... = 2 := by norm_num

end units_digit_of_expression_l141_141074


namespace nine_digit_palindromes_count_l141_141957

theorem nine_digit_palindromes_count :
  let digits := [2, 2, 3, 3, 3, 5, 5, 5, 5]
  let palindrome_form := list.permutations digits
  palindrome_form.count (λ p, p.take 5 = p.drop 4.reverse) = 420 :=
sorry

end nine_digit_palindromes_count_l141_141957


namespace algebraic_expression_for_A_l141_141157

variable {x y A : ℝ}

theorem algebraic_expression_for_A
  (h : (3 * x + 2 * y) ^ 2 = (3 * x - 2 * y) ^ 2 + A) :
  A = 24 * x * y :=
sorry

end algebraic_expression_for_A_l141_141157


namespace moles_CH3COOH_equiv_l141_141875

theorem moles_CH3COOH_equiv (moles_NaOH moles_NaCH3COO : ℕ)
    (h1 : moles_NaOH = 1)
    (h2 : moles_NaCH3COO = 1) :
    moles_NaOH = moles_NaCH3COO :=
by
  sorry

end moles_CH3COOH_equiv_l141_141875


namespace tan_inequality_iff_l141_141853

open Real

noncomputable def tan_inequality_solution_set (x : ℝ) : Prop :=
∃ k : ℤ, -π/2 + k * π < x ∧ x ≤ π/3 + k * π

theorem tan_inequality_iff (x : ℝ) :
  (tan(x) - sqrt(3) ≤ 0) ↔ tan_inequality_solution_set x :=
sorry

end tan_inequality_iff_l141_141853


namespace starts_net_profit_from_third_year_option_1_more_profitable_l141_141399

-- Total Net Profit Calculation
def net_profit (n : ℕ) : ℝ :=
  1.00 * n - (0.24 * ↑n + ↑(n * (n - 1)) * 0.04) - 1.44

-- Condition 1: Business starts making a net profit from the third year onwards
theorem starts_net_profit_from_third_year : ∀ n : ℕ, n ≥ 3 → net_profit n > 0 :=
by {
  assume n hn,
  -- the quadratic inequality -4n^2 + 80n - 144 > 0 for n >= 3
  sorry
}

-- Average annual profit calculation
def avg_annual_profit (n : ℕ) : ℝ :=
  net_profit n / ↑n

-- Option ①: Total returns when avg annual profit is maximized
def option_1_total_returns : ℝ :=
  let avg_profit := avg_annual_profit 6 in
  avg_profit * 6 + 0.96

-- Option ②: Total returns when net profit is maximized
def option_2_total_returns : ℝ :=
  let max_net_profit := net_profit 10 in
  max_net_profit + 0.32

-- Condition 2: Option ① is more profitable considering total net profit
theorem option_1_more_profitable : option_1_total_returns = 288 ∧ option_2_total_returns = 288 :=
by {
  -- We need to prove the numerical equality based on the profit formulas derived.
  sorry
}

end starts_net_profit_from_third_year_option_1_more_profitable_l141_141399


namespace angle_of_unused_sector_l141_141887

noncomputable def cone_volume (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

theorem angle_of_unused_sector (r_paper : ℝ) (r_cone : ℝ) (v_cone : ℝ) (angle_total : ℝ) :
  (r_paper = 20) ∧ (r_cone = 15) ∧ (v_cone = 675 * π) ∧ (angle_total = 360) →
  ∃ angle_unused : ℝ, angle_unused = 90 :=
by
  sorry

end angle_of_unused_sector_l141_141887


namespace steven_weight_l141_141845

theorem steven_weight (danny_weight : ℝ) (steven_more : ℝ) (steven_weight : ℝ) 
  (h₁ : danny_weight = 40) 
  (h₂ : steven_more = 0.2 * danny_weight) 
  (h₃ : steven_weight = danny_weight + steven_more) : 
  steven_weight = 48 := 
  by 
  sorry

end steven_weight_l141_141845


namespace minimum_omega_formula_l141_141658

noncomputable def minimumOmega (f : ℝ → ℝ) (x T ω φ : ℝ) : ℝ :=
  if (f = λ x, Real.cos (ω * x + φ)) 
     ∧ (ω > 0) 
     ∧ (0 < φ < π)
     ∧ (f T = √3 / 2)
     ∧ (f (π / 9) = 0) 
     ∧ (T = 2 * π / ω) 
  then 3 else 0

theorem minimum_omega_formula {f : ℝ → ℝ} {x T ω φ : ℝ} :
  (f = λ x, Real.cos (ω * x + φ)) 
  ∧ (ω > 0) 
  ∧ (0 < φ < π)
  ∧ (f T = √3 / 2)
  ∧ (f (π / 9) = 0)
  ∧ (T = 2 * π / ω) 
  → minimumOmega f x T ω φ = 3 :=
by
  sorry

end minimum_omega_formula_l141_141658


namespace find_z_l141_141659

theorem find_z (z : ℂ) (h : (3 + z) * complex.i = 1) : 
  z = -3 - complex.i :=
by
  sorry

end find_z_l141_141659


namespace problem_l141_141376

open Function

variable (f : ℝ → ℝ)

-- Conditions
axiom one_to_one : Injective f
axiom strictly_increasing : ∀ x y : ℝ, x < y → f(x) < f(y)

-- Definition of solution sets
def P := { x : ℝ | x = f x }
def Q := { x : ℝ | x = f (f x) }

-- The theorem to prove
theorem problem (h1 : Injective f) (h2 : ∀ x y : ℝ, x < y → f x < f y) : P f = Q f :=
sorry

end problem_l141_141376


namespace eulers_polyhedron_theorem_l141_141473

theorem eulers_polyhedron_theorem 
  (V E F t h : ℕ) (T H : ℕ) :
  (F = 30) →
  (t = 20) →
  (h = 10) →
  (T = 3) →
  (H = 2) →
  (E = (3 * t + 6 * h) / 2) →
  (V - E + F = 2) →
  100 * H + 10 * T + V = 262 :=
by
  intros F_eq t_eq h_eq T_eq H_eq E_eq euler_eq
  rw [F_eq, t_eq, h_eq, T_eq, H_eq, E_eq] at *
  sorry

end eulers_polyhedron_theorem_l141_141473


namespace clock_angle_3_40_l141_141763

theorem clock_angle_3_40 : 
  let h := 3
      m := 40
      angle := |(60 * h - 11 * m) / 2|
  in min angle (180 - angle) = 50 := by
  sorry

end clock_angle_3_40_l141_141763


namespace max_value_expression_correct_l141_141664

noncomputable def max_value_expression (z : ℂ) (hz : |z| = 1) : ℝ :=
  |(z + complex.I) / (z + 2)|

theorem max_value_expression_correct (z : ℂ) (hz : |z| = 1) : 
  max_value_expression z hz = (2 * real.sqrt 5) / 3 := 
by 
  sorry

end max_value_expression_correct_l141_141664


namespace spider_travel_distance_l141_141425

theorem spider_travel_distance (r : ℝ) (d₁ d₃ : ℝ) (d₂ : ℝ)
  (h_r : r = 60)
  (h_d₁ : d₁ = 2 * r)
  (h_d₃ : d₃ = 70)
  (h_pythagorean : d₁^2 = d₂^2 + d₃^2) :
  d₁ + d₂ + d₃ = 287.47 :=
by
  have h_diameter : d₁ = 120 := by linarith [h_r, h_d₁]
  have h_d₂_calc : d₂ = Real.sqrt (d₁^2 - d₃^2) := by sorry
  have h_sqrt : Real.sqrt (120^2 - 70^2) = 97.47 := by sorry
  rw [h_diameter, h_d₂_calc, h_sqrt]
  linarith


end spider_travel_distance_l141_141425


namespace geometric_sequence_general_term_sum_of_sequence_n_terms_l141_141535
-- Importing the necessary library

-- Problem (1): Prove the general term formula of the sequence
theorem geometric_sequence_general_term :
  ∀ (n : ℕ), (∀ k : ℕ, a_n > 0)  -- All terms are positive
  ∧ a_2 = 2
  ∧ (∀ a1 q : ℝ, q > 0 ∧ q*q - q - 2 = 0 ∧ a_3 = 2 + 2 * a1 → a_n = 2^(n-1)) := 
sorry

-- Importing the necessary library

-- Problem (2): Prove the sum of the first n terms of the sequence
theorem sum_of_sequence_n_terms :
  ∀ (a : ℕ → ℝ), 
  (∀ k : ℕ, a k > 0)  -- All terms are positive
  ∧ a 2 = 2
  ∧ (∀ n : ℕ, (λ n, 2 * n - 1) / a n = (2 * n - 1) / 2^(n-1))
  ∧ (∀ S_n : ℕ → ℝ, S_n = λ n, 1 + (3 / 2) + (5 / 4) + ... + ((2 * n - 1) / 2^(n-1))
  → S_n = 6 - (2 * n + 3) / 2^(n-1)) := 
sorry

end geometric_sequence_general_term_sum_of_sequence_n_terms_l141_141535


namespace min_value_x_plus_3y_min_value_xy_l141_141107

variable {x y : ℝ}

theorem min_value_x_plus_3y (hx : x > 0) (hy : y > 0) (h : 1/x + 3/y = 1) : x + 3 * y ≥ 16 :=
sorry

theorem min_value_xy (hx : x > 0) (hy : y > 0) (h : 1/x + 3/y = 1) : x * y ≥ 12 :=
sorry

end min_value_x_plus_3y_min_value_xy_l141_141107


namespace base_5_to_base_7_equivalence_l141_141065

theorem base_5_to_base_7_equivalence : 
  ∀ (n : ℕ), base5_to_nat 412 = n → nat_to_base7 n = 212 :=
by
  intros n h
  sorry

-- Definitions for base5_to_nat and nat_to_base7
def base5_to_nat (n : ℕ) : ℕ :=
  4 * 5^2 + 1 * 5^1 + 2 * 5^0

def nat_to_base7 (n : ℕ) : ℕ :=
  2 * 7^2 + 1 * 7^1 + 2 * 7^0

end base_5_to_base_7_equivalence_l141_141065


namespace non_constant_function_inequality_l141_141867

theorem non_constant_function_inequality :
  ∀ a : ℝ, (∃ f : (0:ℝ) → ℝ, (∃ c1 c2 : (0:ℝ), c1 ≠ c2) ∧ 
  (∀ x y : (0:ℝ), a + f (x + y - x * y) + f x * f y ≤ f x + f y)) ↔ a < ¼ :=
sorry

end non_constant_function_inequality_l141_141867


namespace find_number_of_eggs_l141_141475

namespace HalloweenCleanup

def eggs (E : ℕ) (seconds_per_egg : ℕ) (minutes_per_roll : ℕ) (total_time : ℕ) (num_rolls : ℕ) : Prop :=
  seconds_per_egg = 15 ∧
  minutes_per_roll = 30 ∧
  total_time = 225 ∧
  num_rolls = 7 ∧
  E * (seconds_per_egg / 60) + num_rolls * minutes_per_roll = total_time

theorem find_number_of_eggs : ∃ E : ℕ, eggs E 15 30 225 7 :=
  by
    use 60
    unfold eggs
    simp
    exact sorry

end HalloweenCleanup

end find_number_of_eggs_l141_141475


namespace height_of_smaller_cone_l141_141798

theorem height_of_smaller_cone (h_frustum : ℝ) (area_lower_base area_upper_base : ℝ) 
  (h_frustum_eq : h_frustum = 18) 
  (area_lower_base_eq : area_lower_base = 144 * Real.pi) 
  (area_upper_base_eq : area_upper_base = 16 * Real.pi) : 
  ∃ (x : ℝ), x = 9 :=
by
  -- Definitions and assumptions go here
  sorry

end height_of_smaller_cone_l141_141798


namespace expression_equals_1390_l141_141836

theorem expression_equals_1390 :
  (25 + 15 + 8) ^ 2 - (25 ^ 2 + 15 ^ 2 + 8 ^ 2) = 1390 := 
by
  sorry

end expression_equals_1390_l141_141836


namespace volume_space_correct_l141_141014

open Real

-- Definitions based on the given conditions
def base_radius : ℝ := 4
def sphere_radius : ℝ := 7
def inscribed_cylinder_height : ℝ := 6 * sqrt 5

-- Volumes computations
def volume_sphere : ℝ := (4 / 3) * π * (sphere_radius)^3
def volume_cylinder : ℝ := π * (base_radius)^2 * inscribed_cylinder_height

-- Volume of the space outside the cylinder and inside the sphere
def volume_space : ℝ := volume_sphere - volume_cylinder

-- The value X as a common fraction
def X : ℝ := 728 / 3

-- The theorem statement to prove
theorem volume_space_correct : volume_space = X * π := by
  sorry

end volume_space_correct_l141_141014


namespace problem_statement_l141_141325

-- Definitions based on problem conditions
axiom Person : Type
axiom A B : Person
axiom is_boy : Person → Prop
axiom is_girl : Person → Prop
axiom arrangements1 : Nat
axiom arrangements2 : Nat
axiom arrangements3 : Nat
axiom arrangements4 : Nat

--). Defining the conditions
def condition1 : Prop := 
  arrangements1 = 2160

def condition2 : Prop := 
  arrangements2 = 720

def condition3 : Prop :=
  arrangements3 = 144

def condition4 : Prop :=
  arrangements4 = 720

-- Final theorem statement combining all conditions
theorem problem_statement :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 :=
begin
  split,
  { exact sorry },
  split,
  { exact sorry },
  split,
  { exact sorry },
  { exact sorry },
end

end problem_statement_l141_141325


namespace grassy_pathway_area_correct_l141_141809

-- Define the dimensions of the plot and the pathway width
def length_plot : ℝ := 15
def width_plot : ℝ := 10
def width_pathway : ℝ := 2

-- Define the required areas
def total_area : ℝ := (length_plot + 2 * width_pathway) * (width_plot + 2 * width_pathway)
def plot_area : ℝ := length_plot * width_plot
def grassy_pathway_area : ℝ := total_area - plot_area

-- Prove that the area of the grassy pathway is 116 m²
theorem grassy_pathway_area_correct : grassy_pathway_area = 116 := by
  sorry

end grassy_pathway_area_correct_l141_141809


namespace total_oil_leakage_l141_141044

def oil_leaked_before : ℕ := 6522
def oil_leaked_during : ℕ := 5165
def total_oil_leaked : ℕ := oil_leaked_before + oil_leaked_during

theorem total_oil_leakage : total_oil_leaked = 11687 := by
  sorry

end total_oil_leakage_l141_141044


namespace find_y_l141_141924

variable (θ : ℝ) (P : ℝ × ℝ)

-- Conditions
def vertex_origin : Prop := P = (4, P.2)
def initial_side_nonneg_x : Prop := θ = 0 ∨ θ = π / 2 ∨ θ = π ∨ θ = 3 * π / 2
def on_terminal_side : Prop := ∃ θ, θ ≠ π / 2 ∧ θ ≠ 3 * π / 2 ∧ sin θ = -2 * sqrt 5 / 5 ∧ P.1 = 4 ∧ P.2 / sqrt ((P.1)^2 + (P.2)^2) = sin θ

theorem find_y : vertex_origin P → initial_side_nonneg_x θ → on_terminal_side θ P → P.2 = -8 :=
by
  intros
  sorry

end find_y_l141_141924


namespace cos_sum_l141_141859

-- Definitions based on conditions
def equalCircles (ω1 ω2 : Circle) : Prop := 
  ω1.radius = ω2.radius ∧ (ω2.center ∈ ω1) ∧ (ω1.center ∈ ω2)

def inscribedTriangle (ABC : Triangle) (ω1 : Circle) : Prop := 
  ABC.A ∈ ω1 ∧ ABC.B ∈ ω1 ∧ ABC.C ∈ ω1

def tangentLines (C : Point) (AC BC : Line) (ω2 : Circle) : Prop := 
  AC.isTangentTo ω2 ∧ BC.isTangentTo ω2

-- Definition of proof problem based on the given problem and its conditions
theorem cos_sum (ω1 ω2 : Circle) (ABC : Triangle) (AC BC : Line) :
  equalCircles ω1 ω2 →
  inscribedTriangle ABC ω1 →
  tangentLines ABC.C AC BC ω2 →
  (cos (angle ABC.A) + cos (angle ABC.B) = 1) :=
by 
  sorry

end cos_sum_l141_141859


namespace triangle_area_l141_141211

theorem triangle_area 
  (DE EL EF : ℝ)
  (hDE : DE = 14)
  (hEL : EL = 9)
  (hEF : EF = 17)
  (DL : ℝ)
  (hDL : DE^2 = DL^2 + EL^2)
  (hDL_val : DL = Real.sqrt 115):
  (1/2) * EF * DL = 17 * Real.sqrt 115 / 2 :=
by
  -- Sorry, as the proof is not required.
  sorry

end triangle_area_l141_141211


namespace product_evaluation_l141_141474

-- Define the conditions and the target expression
def product (a : ℕ) : ℕ := (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a

-- Main theorem statement
theorem product_evaluation : product 7 = 5040 :=
by
  -- Lean usually requires some import from the broader Mathlib to support arithmetic simplifications
  sorry

end product_evaluation_l141_141474


namespace car_wash_earnings_l141_141800

noncomputable def initialFriends := 6
noncomputable def remainingFriends := 5
noncomputable def totalCost := 1700
noncomputable def extraCost := 40

theorem car_wash_earnings :
  ∃ (carWashEarnings : ℝ), 
    let costPerPerson := totalCost / initialFriends in
    let newCostPerPerson := costPerPerson + extraCost in
    let totalPaidByRemainingFriends := newCostPerPerson * remainingFriends in
    totalPaidByRemainingFriends + carWashEarnings = totalCost ∧ 
    carWashEarnings = 83.35 :=
sorry

end car_wash_earnings_l141_141800


namespace calculate_total_cost_l141_141811

noncomputable def Length1 : ℝ := 5.5
noncomputable def Width1 : ℝ := 3.75
noncomputable def Length2 : ℝ := 4.25
noncomputable def Width2 : ℝ := 2.5
noncomputable def CostPerSqMeterA : ℝ := 1000
noncomputable def CostPerSqMeterB : ℝ := 1200
noncomputable def PercentageA : ℝ := 0.60
noncomputable def PercentageB : ℝ := 0.40

theorem calculate_total_cost :
  let Area1 := Length1 * Width1,
      Area2 := Length2 * Width2,
      TotalArea := Area1 + Area2,
      AreaA := TotalArea * PercentageA,
      AreaB := TotalArea * PercentageB,
      CostA := AreaA * CostPerSqMeterA,
      CostB := AreaB * CostPerSqMeterB,
      TotalCost := CostA + CostB
  in TotalCost = 33750 := by
sorry

end calculate_total_cost_l141_141811


namespace new_acute_angle_is_60_degrees_l141_141293

theorem new_acute_angle_is_60_degrees (A B C : Point) (h: angle A C B = 60) :
  let θ := 600 in
  let effective_rotation := θ % 360 in
  let final_angle := effective_rotation - h in  -- this needs to consider full mod
    180 - (180 - final_angle % 360) = 60 :=
by
  let θ := 600
  let effective_rotation := θ % 360
  let final_angle := effective_rotation - h
  have eq1 : effective_rotation = 240 := by sorry -- {600 % 360 = 240}
  have eq2 : final_angle = 180 := by sorry -- {240 - 60 = 180}
  have eq3 : 180 - final_angle % 360 = 120 := by sorry -- {180 - 180 % 360 = 120}
  have eq4 : 180 - 120 = 60 := by sorry
  show 180 - (180 - (240 - 60) % 360 = 60)

end new_acute_angle_is_60_degrees_l141_141293


namespace range_of_t_l141_141149

-- Define the sets A and B based on the conditions
def A : Set ℝ := { x | 1/4 ≤ 2^x ∧ 2^x ≤ 1/2 }
def B (t : ℝ) : Set ℝ := { x | x^2 - 2 * t * x + 1 ≤ 0 }

-- Define the theorem to state the condition and conclusion
theorem range_of_t (t : ℝ) : (A ⊆ B t) → t ∈ Iic (-5/4) :=
by
  sorry

end range_of_t_l141_141149


namespace factorial_ratio_integer_l141_141916

theorem factorial_ratio_integer (m n : ℕ) : ∃ k : ℕ, (2 * m)! * (2 * n)! = k * (m! * n! * (m + n)!) :=
sorry

end factorial_ratio_integer_l141_141916


namespace exists_diametric_pair_l141_141401

def diametric (P : ℕ → ℕ) (i j : ℕ) : Prop := (P i + 20) % 40 = j

theorem exists_diametric_pair 
  (P : ℕ → Bool)
  (w_count : ∑ i in Finset.range 40, if P i then 1 else 0 = 25)
  (b_count : ∑ i in Finset.range 40, if ¬ P i then 1 else 0 = 15) :
  ∃ i, P i ∧ ¬ P ((i + 20) % 40) := 
sorry

end exists_diametric_pair_l141_141401


namespace total_over_or_underweight_is_8kg_total_selling_price_is_5384_point_8_yuan_l141_141745

-- Definitions based on conditions
def standard_weight : ℝ := 25
def weight_diffs : List ℝ := [-3, -2, -2, -2, -2, -1.5, -1.5, 0, 0, 0, 1, 1, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
def price_per_kg : ℝ := 10.6

-- Problem 1
theorem total_over_or_underweight_is_8kg :
  (weight_diffs.sum = 8) := 
  sorry

-- Problem 2
theorem total_selling_price_is_5384_point_8_yuan :
  (20 * standard_weight + 8) * price_per_kg = 5384.8 :=
  sorry

end total_over_or_underweight_is_8kg_total_selling_price_is_5384_point_8_yuan_l141_141745


namespace pool_water_left_l141_141456

theorem pool_water_left 
  (h1_rate: ℝ) (h1_time: ℝ)
  (h2_rate: ℝ) (h2_time: ℝ)
  (h4_rate: ℝ) (h4_time: ℝ)
  (leak_loss: ℝ)
  (h1_rate_eq: h1_rate = 8)
  (h1_time_eq: h1_time = 1)
  (h2_rate_eq: h2_rate = 10)
  (h2_time_eq: h2_time = 2)
  (h4_rate_eq: h4_rate = 14)
  (h4_time_eq: h4_time = 1)
  (leak_loss_eq: leak_loss = 8) :
  (h1_rate * h1_time) + (h2_rate * h2_time) + (h2_rate * h2_time) + (h4_rate * h4_time) - leak_loss = 34 :=
by
  rw [h1_rate_eq, h1_time_eq, h2_rate_eq, h2_time_eq, h4_rate_eq, h4_time_eq, leak_loss_eq]
  norm_num
  sorry

end pool_water_left_l141_141456


namespace number_of_incorrect_statements_l141_141826

def statement1_condition : Prop := 
  ∀ (dataset : list ℝ) (c : ℝ), list.variance (dataset.map (λ x, x + c)) = list.variance dataset

def statement2_condition : Prop := 
  ∀ (x : ℝ), let y := 3 - 5 * x in y > y + 5

def statement3_condition : Prop := 
  ∀ (x̄ ȳ : ℝ) (b a : ℝ) (data : list (ℝ × ℝ)), 
  list.foldr (λ p (acc : ℝ × ℝ), (acc.1 + p.1, acc.2 + p.2)) (0,0) data = (x̄, ȳ) → 
  (∃ (y : ℝ → ℝ), y = λ x, b * x + a)

def statement4_condition : Prop := 
  ¬(∀ (smoke lung_disease : Prop), Prob (smoke ∧ lung_disease) = 0.99 → 
  Prob (smoke) = 0.99)

theorem number_of_incorrect_statements : 
  {s1:Prop // s1 = statement1_condition → false} +
  {s2:Prop // s2 = statement2_condition → true} + 
  {s3:Prop // s3 = statement3_condition → false} + 
  {s4:Prop // s4 = statement4_condition → true} → 
  2 := by
sorry

end number_of_incorrect_statements_l141_141826


namespace account_balance_after_transfer_l141_141458

def account_after_transfer (initial_balance transfer_amount : ℕ) : ℕ :=
  initial_balance - transfer_amount

theorem account_balance_after_transfer :
  account_after_transfer 27004 69 = 26935 :=
by
  sorry

end account_balance_after_transfer_l141_141458


namespace min_x2_y2_z2_l141_141257

variables {x y z a : ℝ}

-- Given the condition
def condition : Prop := x + 2*y + 3*z = a

theorem min_x2_y2_z2 (h : condition) : x^2 + y^2 + z^2 ≥ a^2 / 14 := by
  sorry

end min_x2_y2_z2_l141_141257


namespace general_formula_for_a_l141_141671

/-- Defining the sequence of x_n where n ≥ 1, and x1 = 0, x2 = a where a > 0. -/
def x (n : ℕ) : ℝ
| 0 := 0  -- this is actually x1, indexed as 0 for easier function definition
| 1 := a
| (n + 2) := (x (n + 1) + x n) / 2

/-- Defining the sequence a_n where a_n = x_{n+1} - x_n. -/
def a_n (n : ℕ) : ℝ :=
x (n + 1) - x n

/-- Main theorem: the general term formula for the sequence {a_n}. -/
theorem general_formula_for_a (n : ℕ) : a_n n = (-(1/2)^(n-1)) * a := 
sorry

end general_formula_for_a_l141_141671


namespace red_more_than_yellow_l141_141744

-- Define the total number of marbles
def total_marbles : ℕ := 19

-- Define the number of yellow marbles
def yellow_marbles : ℕ := 5

-- Calculate the number of remaining marbles
def remaining_marbles : ℕ := total_marbles - yellow_marbles

-- Define the ratio of blue to red marbles
def blue_ratio : ℕ := 3
def red_ratio : ℕ := 4

-- Calculate the sum of ratio parts
def sum_ratio : ℕ := blue_ratio + red_ratio

-- Calculate the number of shares per ratio part
def share_per_part : ℕ := remaining_marbles / sum_ratio

-- Calculate the number of red marbles
def red_marbles : ℕ := red_ratio * share_per_part

-- Theorem to prove: the difference between red marbles and yellow marbles is 3
theorem red_more_than_yellow : red_marbles - yellow_marbles = 3 :=
by
  sorry

end red_more_than_yellow_l141_141744


namespace tangent_line_at_1_f_geq_x_minus_1_min_value_a_l141_141936

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- 1. Proof that the equation of the tangent line at the point (1, f(1)) is y = x - 1
theorem tangent_line_at_1 :
  ∃ k b, (k = 1 ∧ b = -1 ∧ (∀ x, (f x - k * x - b) = 0)) :=
sorry

-- 2. Proof that f(x) ≥ x - 1 for all x in (0, +∞)
theorem f_geq_x_minus_1 :
  ∀ x, 0 < x → f x ≥ x - 1 :=
sorry

-- 3. Proof that the minimum value of a such that f(x) ≥ ax² + 2/a for all x in (0, +∞) is -e³
theorem min_value_a :
  ∃ a, (∀ x, 0 < x → f x ≥ a * x^2 + 2 / a) ∧ (a = -Real.exp 3) :=
sorry

end tangent_line_at_1_f_geq_x_minus_1_min_value_a_l141_141936


namespace problem_statement_l141_141237

noncomputable def f : ℝ → ℝ := sorry  -- Define f as a noncomputable function to accommodate the problem constraints

variables (a : ℝ)

theorem problem_statement (periodic_f : ∀ x, f (x + 3) = f x)
    (odd_f : ∀ x, f (-x) = -f x)
    (ineq_f1 : f 1 < 1)
    (eq_f2 : f 2 = (2*a-1)/(a+1)) :
    a < -1 ∨ 0 < a :=
by
  sorry

end problem_statement_l141_141237


namespace hexagon_digit_assignment_l141_141694

-- Define the problem
theorem hexagon_digit_assignment :
  ∃ (assignments : Finset (Fin 7)), 
    assignments.card = 7 ∧
    (∀ (x y z: ℕ) (h₁: x ∈ assignments) (h₂: y ∈ assignments) (h₃: z ∈ assignments), x + y + z = x) →
    assignments.card = 144 :=
begin
  sorry
end

end hexagon_digit_assignment_l141_141694


namespace find_angle_A_l141_141559

theorem find_angle_A (A B C : ℝ)
  (h1 : C = 2 * B)
  (h2 : B = A / 3)
  (h3 : A + B + C = 180) : A = 90 :=
by
  sorry

end find_angle_A_l141_141559


namespace red_marbles_more_than_yellow_l141_141739

-- Define the conditions
def total_marbles : ℕ := 19
def yellow_marbles : ℕ := 5
def ratio_parts_blue : ℕ := 3
def ratio_parts_red : ℕ := 4

-- Prove the number of more red marbles than yellow marbles is equal to 3
theorem red_marbles_more_than_yellow :
  let remainder_marbles := total_marbles - yellow_marbles
  let total_parts := ratio_parts_blue + ratio_parts_red
  let marbles_per_part := remainder_marbles / total_parts
  let blue_marbles := ratio_parts_blue * marbles_per_part
  let red_marbles := ratio_parts_red * marbles_per_part
  red_marbles - yellow_marbles = 3 := by
  sorry

end red_marbles_more_than_yellow_l141_141739


namespace new_acute_angle_is_60_degrees_l141_141292

theorem new_acute_angle_is_60_degrees (A B C : Point) (h: angle A C B = 60) :
  let θ := 600 in
  let effective_rotation := θ % 360 in
  let final_angle := effective_rotation - h in  -- this needs to consider full mod
    180 - (180 - final_angle % 360) = 60 :=
by
  let θ := 600
  let effective_rotation := θ % 360
  let final_angle := effective_rotation - h
  have eq1 : effective_rotation = 240 := by sorry -- {600 % 360 = 240}
  have eq2 : final_angle = 180 := by sorry -- {240 - 60 = 180}
  have eq3 : 180 - final_angle % 360 = 120 := by sorry -- {180 - 180 % 360 = 120}
  have eq4 : 180 - 120 = 60 := by sorry
  show 180 - (180 - (240 - 60) % 360 = 60)

end new_acute_angle_is_60_degrees_l141_141292


namespace alice_distance_from_start_l141_141035

def feet_per_meter : ℝ := 3.28084
def north_distance_m : ℝ := 30
def east_distance_f : ℝ := 40
def south_distance_m : ℝ := 15
def additional_south_f : ℝ := 50

def north_distance_f := north_distance_m * feet_per_meter
def south_distance_f := south_distance_m * feet_per_meter + additional_south_f
def net_south_distance_f := south_distance_f - north_distance_f

theorem alice_distance_from_start : 
    (sqrt ((east_distance_f)^2 + (net_south_distance_f)^2) ≈ 40) :=
by
  sorry

lemma feet_perapprox_eq (a b : ℝ) : a ≈ b ↔ abs (a - b) < 0.01 :=
by
  sorry

end alice_distance_from_start_l141_141035


namespace nearest_integer_sum_of_x_l141_141799

noncomputable def f : ℝ → ℝ := sorry

axiom func_condition (x : ℝ) (hx : x ≠ 0) : 3 * f x + 2 * f (1 / x) = 6 * x + 3

theorem nearest_integer_sum_of_x (S : ℝ) (hx : ∀ x : ℝ, f x = 2010 ↔ x = sorry) :
  S ≈ 558 :=
begin
  -- Note: proof will involve solving the quadratic equation derived in the solution,
  -- and using Vieta's formulas to establish the sum of the roots.
  sorry,
end

end nearest_integer_sum_of_x_l141_141799


namespace minimum_omega_l141_141614

noncomputable section

-- Define the function f and its properties
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.cos (ω * x + φ)

-- Assumptions based on the given conditions
variables {ω φ : ℝ}
variables (T : ℝ) (hω_pos : 0 < ω) (hφ_range : 0 < φ ∧ φ < π)
variables (hT : f T ω φ = Real.sqrt 3 / 2)
variables (hx_zero : f (π / 9) ω φ = 0)

-- Prove the minimum value of ω is 3
theorem minimum_omega : ω = 3 := sorry

end minimum_omega_l141_141614


namespace area_of_rectangle_ABC_discribed_in_triangle_XYZ_l141_141754

/-
Conditions:
1. Triangle XYZ has an altitude from point X to side YZ measuring 9 cm.
2. The length of side YZ is 15 cm.
3. Rectangle ABCD is inscribed in triangle XYZ.
4. Side AD of the rectangle lies along side YZ of the triangle.
5. The length of segment AB is one-third the length of segment AD.
Conclusion:
The area of rectangle ABCD is 675/16 cm².
-/

theorem area_of_rectangle_ABC_discribed_in_triangle_XYZ
    (X Y Z A B C D : Type)
    (h1 : XY.Height = 9)
    (h2 : YZ.Length = 15)
    (h3 : rectangle_inscribed_in_triangle ABCD XYZ)
    (h4 : side_AD_on_side_YZ AD YZ)
    (h5 : length_AB_eq_one_third_AD (length AB) (length AD))
    : area ABCD = 675 / 16 := 
sorry

end area_of_rectangle_ABC_discribed_in_triangle_XYZ_l141_141754


namespace pies_sold_l141_141072

-- Define the conditions in Lean
def num_cakes : ℕ := 453
def price_per_cake : ℕ := 12
def total_earnings : ℕ := 6318
def price_per_pie : ℕ := 7

-- Define the problem
theorem pies_sold (P : ℕ) (h1 : num_cakes * price_per_cake + P * price_per_pie = total_earnings) : P = 126 := 
by 
  sorry

end pies_sold_l141_141072


namespace find_number_l141_141561

theorem find_number (n x : ℤ) (h1 : n * x + 3 = 10 * x - 17) (h2 : x = 4) : n = 5 :=
by
  sorry

end find_number_l141_141561


namespace A_beats_B_by_28_meters_l141_141789

theorem A_beats_B_by_28_meters :
  ∀ (dA dB tA tB : ℕ), dA = 224 → dB = 224 → tA = 28 → tB = 32 →
  let speedA := dA / tA in
  let speedB := dB / tB in
  let distB_in_A_time := speedB * tA in
  dA - distB_in_A_time = 28 :=
by {
  intros dA dB tA tB h_dA h_dB h_tA h_tB,
  rw [h_dA, h_dB, h_tA, h_tB],
  let speedA := (224 / 28 : ℕ),
  let speedB := (224 / 32 : ℕ),
  let distB_in_A_time := speedB * 28,
  have h_speedA : speedA = 8 := by norm_num,
  have h_speedB : speedB = 7 := by norm_num,
  rw [h_speedA, h_speedB],
  let distB_in_A_time := (7 * 28 : ℕ),
  have h_distB_in_A_time : distB_in_A_time = 196 := by norm_num,
  rw h_distB_in_A_time,
  norm_num,
}

end A_beats_B_by_28_meters_l141_141789


namespace sum_of_digits_N_l141_141606

-- Define a function to compute the least common multiple (LCM) of a list of numbers
def lcm_list (xs : List ℕ) : ℕ :=
  xs.foldr Nat.lcm 1

-- The set of numbers less than 8
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7]

-- The LCM of numbers less than 8
def N_lcm : ℕ := lcm_list nums

-- The second smallest positive integer that is divisible by every positive integer less than 8
def N : ℕ := 2 * N_lcm

-- Function to compute the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Prove that the sum of the digits of N is 12
theorem sum_of_digits_N : sum_of_digits N = 12 :=
by
  -- Necessary proof steps will be filled here
  sorry

end sum_of_digits_N_l141_141606


namespace major_axis_double_minor_axis_l141_141011

-- Define the radius of the right circular cylinder.
def cylinder_radius := 2

-- Define the minor axis length based on the cylinder's radius.
def minor_axis_length := 2 * cylinder_radius

-- Define the major axis length as double the minor axis length.
def major_axis_length := 2 * minor_axis_length

-- State the theorem to prove the major axis length.
theorem major_axis_double_minor_axis : major_axis_length = 8 := by
  sorry

end major_axis_double_minor_axis_l141_141011


namespace sum_of_non_negative_reals_l141_141663

theorem sum_of_non_negative_reals (n : ℕ) (h1 : 1 ≤ n)
  (x : Fin (n + 2) → ℝ) 
  (h2 : ∀ i : Fin (n + 1), 0 ≤ x i) 
  (h3 : ∀ i : Fin n, x i * x (i + 1) - (x (i - 1))^2 ≥ 1) :
  ∑ i in Finset.range (n+2), x i > (2 * n / 3) ^ (3 / 2) := 
sorry

end sum_of_non_negative_reals_l141_141663


namespace winner_votes_more_than_loser_l141_141198

-- Define the initial conditions.
def winner_percentage : ℝ := 0.55
def total_students : ℕ := 2000
def voting_percentage : ℝ := 0.25

-- Define the number of students who voted
def num_voted := total_students * (voting_percentage : ℕ)

-- Define the votes received by the winner and the loser,
-- based on the percentage of the total votes.
def winner_votes := num_voted * winner_percentage
def loser_votes := num_voted * (1 - winner_percentage)

-- Define the difference in votes between the winner and the loser.
def vote_difference := winner_votes - loser_votes

-- The target theorem to prove.
theorem winner_votes_more_than_loser : vote_difference = 50 := by
  sorry

end winner_votes_more_than_loser_l141_141198


namespace exsphere_identity_l141_141256

-- Given definitions for heights and radii
variables {h1 h2 h3 h4 r1 r2 r3 r4 : ℝ}

-- Definition of the relationship that needs to be proven
theorem exsphere_identity 
  (h1 h2 h3 h4 r1 r2 r3 r4 : ℝ) :
  2 * (1 / h1 + 1 / h2 + 1 / h3 + 1 / h4) = 1 / r1 + 1 / r2 + 1 / r3 + 1 / r4 := 
sorry

end exsphere_identity_l141_141256


namespace sum_positive_real_numbers_eq_2_l141_141665

/--
Let \(S\) be the sum of all positive real numbers \(x\) for which \(x^{2^x} = 4^{x^2}\).
The value of \(S\) is 2.
-/
theorem sum_positive_real_numbers_eq_2 :
  let S := (finset.filter (λ x : ℝ, 0 < x) (finset.range 10)) -- Adjust range as needed.
  finset.sum S (λ x, if x^(2^x) = 4^(x^2) then x else 0) = 2 :=
by
  sorry

end sum_positive_real_numbers_eq_2_l141_141665


namespace perfect_factorial_only_six_l141_141442

-- Define a perfect number
def is_perfect (n : ℕ) : Prop :=
  n > 1 ∧ (∑ d in (finset.filter (λ x, x ≠ n) (finset.divisors n)), d) = n

-- Define a factorial number
def is_factorial (n : ℕ) : Prop :=
  ∃ m : ℕ, n = (finset.range (m + 1)).prod nat.succ

-- The theorem to be proved
theorem perfect_factorial_only_six : ∀ n : ℕ, is_perfect n ∧ is_factorial n → n = 6 := 
by
  intros,
  sorry -- Proof is omitted as required.

end perfect_factorial_only_six_l141_141442


namespace triangle_product_area_perimeter_l141_141678

theorem triangle_product_area_perimeter :
  let P := (1, 5)
  let Q := (5, 5)
  let R := (1, 1)
  let PQ := (Q.1 - P.1 : ℝ)
  let PR := (P.2 - R.2 : ℝ)
  let QR := real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  let area := 0.5 * PQ * PR
  let perimeter := PQ + PR + QR
  area * perimeter = 64 + 32 * real.sqrt 2 := 
by
  sorry

end triangle_product_area_perimeter_l141_141678


namespace cone_volume_of_sector_l141_141015

theorem cone_volume_of_sector (r : ℝ) (theta : ℝ) (h : ℝ) (V : ℝ)
  (radius_condition : r = 3)
  (angle_condition : theta = 120 * (π / 180)) -- angle converted to radians
  (arc_length_condition : theta * r = 2 * π)
  (base_radius_condition : 2 * π = 2 * π * r) -- circumference of base
  (height_condition : h = sqrt (r ^ 2 - 1 ^ 2))
  (volume_formula : V = (1 / 3) * π * r * r * h) :
  V = (2 * sqrt 2 / 3) * π :=
sorry

end cone_volume_of_sector_l141_141015


namespace triangle_construction_cases_l141_141361

def triangle_construction (a b γ : ℝ) : Prop :=
(a ≥ 0 ∧ b ≥ 0 ∧ (a = b ∨ (a - b) * γ ≠ 0))

theorem triangle_construction_cases (a b γ : ℝ) :
  triangle_construction a b γ →
  ((a - b) * γ > 0 → ∃! (ABC : Triangle), 
    ABC.BC = a ∧ ABC.AC = b ∧ ABC.∠B - ABC.∠A = γ) ∧
  ((a = b ∧ γ = 0) → ∃ (ABC : Triangle), 
    ∀ABC' : Triangle, (ABC.BC = a ∧ ABC.AC = b ∧ ABC.∠B - ABC.∠A = γ → ABC ≅ ABC')) ∧
  (((a - b) * γ ≤ 0 ∧ (a ≠ b ∨ γ ≠ 0)) → ¬ ∃ (ABC : Triangle), 
    ABC.BC = a ∧ ABC.AC = b ∧ ABC.∠B - ABC.∠A = γ) :=
by
  sorry

end triangle_construction_cases_l141_141361


namespace sum_valid_x_l141_141367

theorem sum_valid_x (x : ℝ) (hx : sqrt ((x - 5)^2) = 9) (hne : x ≠ -4) : x = 14 :=
by {
  have h1 : (x - 5) = 9 ∨ (x - 5) = -9,
  { rw [sqrt_eq_r_pow, pow_two, pow_two, <-function.funext_iff, <-and.congr_right_iff] at hx,
    exact hx },
  cases h1,
  { rw h1,
    exact 14 },
  { rw h1 at hne,
    contradiction }
}

end sum_valid_x_l141_141367


namespace largest_possible_median_of_list_of_eleven_l141_141268

open List

-- Define the original list and the auxiliary definition
def originalList : List ℕ := [3, 5, 9, 1, 6, 2]

-- Define a function to calculate the median of a list of exactly eleven integers
def medianOfListOfEleven (lst : List ℕ) : ℕ :=
  if h : lst.length = 11 then
    let sorted_lst := sort (· ≤ ·) lst
    sorted_lst.nthLe 5 (by simp [h])
  else
    0 -- Return 0 or any placeholder value if the list is not of length 11

-- State the theorem to be proven 
theorem largest_possible_median_of_list_of_eleven : 
  ∃ (lst : List ℕ), originalList ⊆ lst ∧ lst.length = 11 ∧ (∀ x ∈ lst, x > 0) ∧ medianOfListOfEleven lst = 10 :=
sorry

end largest_possible_median_of_list_of_eleven_l141_141268


namespace smallest_n_for_107n_same_last_two_digits_l141_141087

theorem smallest_n_for_107n_same_last_two_digits :
  ∃ n : ℕ, n > 0 ∧ (107 * n) % 100 = n % 100 ∧ n = 50 :=
by {
  sorry
}

end smallest_n_for_107n_same_last_two_digits_l141_141087
