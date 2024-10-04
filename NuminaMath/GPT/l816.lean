import Mathlib

namespace correct_structure_l816_816437

-- Definitions for the conditions regarding flowchart structures
def loop_contains_conditional : Prop := ∀ (loop : Prop), ∃ (conditional : Prop), conditional ∧ loop
def unique_flowchart_for_boiling_water : Prop := ∀ (flowcharts : Prop), ∃! (boiling_process : Prop), flowcharts ∧ boiling_process
def conditional_does_not_contain_sequential : Prop := ∀ (conditional : Prop), ∃ (sequential : Prop), ¬ (conditional ∧ sequential)
def conditional_must_contain_loop : Prop := ∀ (conditional : Prop), ∃ (loop : Prop), conditional ∧ loop

-- The proof statement
theorem correct_structure (A B C D : Prop) (hA : A = loop_contains_conditional) 
  (hB : B = unique_flowchart_for_boiling_water) 
  (hC : C = conditional_does_not_contain_sequential) 
  (hD : D = conditional_must_contain_loop) : 
  A = loop_contains_conditional ∧ ¬ B ∧ ¬ C ∧ ¬ D :=
by {
  sorry
}

end correct_structure_l816_816437


namespace necessary_but_not_sufficient_intersection_l816_816301

theorem necessary_but_not_sufficient_intersection
  (a b : ℝ)
  (h_outside : a^2 + b^2 > 1) :
  (∀ (x y : ℝ), x^2 + y^2 = 1 → ax + by + 2 = 0 → False) ∧ ¬ ( ∀ (x y : ℝ), ax + by + 2 = 0 → x^2 + y^2 = 1 ) := by
  sorry

end necessary_but_not_sufficient_intersection_l816_816301


namespace value_of_N_l816_816199

theorem value_of_N (N : ℕ) (h : (20 / 100) * N = (60 / 100) * 2500) : N = 7500 :=
by {
  sorry
}

end value_of_N_l816_816199


namespace twentieth_number_l816_816407

-- Defining the conditions and goal
theorem twentieth_number :
  ∃ x : ℕ, x % 8 = 5 ∧ x % 3 = 2 ∧ (∃ n : ℕ, x = 5 + 24 * n) ∧ x = 461 := 
sorry

end twentieth_number_l816_816407


namespace speed_against_current_l816_816053

theorem speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) (man_speed_against_current : ℝ) 
  (h : speed_with_current = 12) (h1 : current_speed = 2) : man_speed_against_current = 8 :=
by
  sorry

end speed_against_current_l816_816053


namespace pencil_count_l816_816935

theorem pencil_count 
  (cindi_spent : ℝ)
  (pencil_cost : ℝ)
  (marcia_factor : ℝ)
  (donna_factor : ℝ)
  (bob_extra : ℕ)
  (hc : cindi_spent = 18.75)
  (hp : pencil_cost = 0.25)
  (hm : marcia_factor = 1.5)
  (hd : donna_factor = 4)
  (hb : bob_extra = 20)
  :
  let cindi_pencils := (cindi_spent / pencil_cost).toNat,
      marcia_pencils := (cindi_pencils * marcia_factor).toNat,
      donna_pencils := marcia_pencils * donna_factor.toNat,
      bob_pencils := cindi_pencils + bob_extra 
  in  donna_pencils + marcia_pencils + bob_pencils = 655 :=
by
  sorry

end pencil_count_l816_816935


namespace max_integer_solutions_l816_816913

-- Definition of a semi-centered polynomial
def semi_centred_polynomial (p : ℤ → ℤ) : Prop :=
  (∀ n : ℤ, p n ∈ ℤ) ∧ (p 50 = 50)

-- The theorem statement
theorem max_integer_solutions (p : ℤ → ℤ) (hq : semi_centred_polynomial p) :
  (∃ n : ℕ, ∀ k : ℤ, p k = k^2 → k ∈ (finset.range n) ∧ n = 8) :=
sorry

end max_integer_solutions_l816_816913


namespace probability_odd_product_l816_816954

theorem probability_odd_product :
  let box1 := [1, 2, 3, 4]
  let box2 := [1, 2, 3, 4]
  let total_outcomes := 4 * 4
  let favorable_outcomes := [(1,1), (1,3), (3,1), (3,3)]
  let num_favorable := favorable_outcomes.length
  (num_favorable / total_outcomes : ℚ) = 1 / 4 := 
by
  sorry

end probability_odd_product_l816_816954


namespace differentiation_operations_incorrect_l816_816877

theorem differentiation_operations_incorrect :
  ¬ ((deriv^[2] (fun x : ℝ => cos x) = fun x => sin x) ∨
     (deriv^[2] (fun x : ℝ => log (2 * x)) = fun x => 1 / x) ∨
     (deriv^[2] (fun x : ℝ => 3^x) = fun x => 3^x * log 3) ∨
     (deriv^[2] (fun x : ℝ => x^2 * exp x) = fun x => 2 * x * exp x)) :=
by sorry

end differentiation_operations_incorrect_l816_816877


namespace initial_oranges_is_sum_l816_816409

-- Define the number of oranges taken by Jonathan
def oranges_taken : ℕ := 45

-- Define the number of oranges left in the box
def oranges_left : ℕ := 51

-- The theorem states that the initial number of oranges is the sum of the oranges taken and those left
theorem initial_oranges_is_sum : oranges_taken + oranges_left = 96 := 
by 
  -- This is where the proof would go
  sorry

end initial_oranges_is_sum_l816_816409


namespace sequence_50th_term_l816_816709

def sequence (n : ℕ) : ℕ := 2 + 4 * (n - 1)

theorem sequence_50th_term : sequence 50 = 198 := by
  sorry

end sequence_50th_term_l816_816709


namespace prime_quadruples_unique_l816_816887

noncomputable def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → (m = 1 ∨ m = n)

theorem prime_quadruples_unique (p q r n : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) (hn : n > 0)
  (h_eq : p^2 = q^2 + r^n) :
  (p, q, r, n) = (3, 2, 5, 1) ∨ (p, q, r, n) = (5, 3, 2, 4) :=
by
  sorry

end prime_quadruples_unique_l816_816887


namespace limit_of_sequence_l816_816931

open Real Filter

theorem limit_of_sequence :
  tendsto (λ n : ℕ, ( (2 * n - 1) / (2 * n + 1) ) ^ (n + 1)) at_top (𝓝 (1 / exp 1)) :=
sorry

end limit_of_sequence_l816_816931


namespace proof_problem_part1_proof_problem_part2_l816_816736

variable (a b c : ℝ)
variable (B : ℝ)
variable (A : ℝ)

-- Conditions
def condition1 : Prop := a^2 + c^2 - b^2 = ac
def condition2 : Prop := a = 8 * sqrt 3
def condition3 : Prop := A = real.arccos (3/5)

-- Expected results
def expected_B : Prop := B = real.pi / 3
def expected_b : Prop := b = 15

-- Proof problem
theorem proof_problem_part1 (h1 : condition1) : expected_B :=
sorry

theorem proof_problem_part2 (h2 : condition2) (h3 : condition3) : expected_b :=
sorry

end proof_problem_part1_proof_problem_part2_l816_816736


namespace sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816214

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816214


namespace earrings_gumballs_ratio_l816_816747

theorem earrings_gumballs_ratio:
  ∃ x : ℕ, (3 * 42 = 27 + 9 * x + 9 * (x - 1)) ∧ (ratio (6 : ℕ) (3 : ℕ) = 2) :=
by
  sorry

end earrings_gumballs_ratio_l816_816747


namespace cos_315_eq_sqrt2_div_2_l816_816529

theorem cos_315_eq_sqrt2_div_2 :
  let θ := 315.0
  in θ = 360.0 - 45.0 ∧ (cos 45 * (π / 180)) = (Real.sqrt 2 / 2) →
  (cos (θ * (π / 180))) = (Real.sqrt 2 / 2) :=
by
  intros θ
  -- Non-necessary proof omitted
  sorry

end cos_315_eq_sqrt2_div_2_l816_816529


namespace number_of_solutions_eq_40600_l816_816629

def f (n : ℤ) : ℤ := Int.ceil (197 * n / 200) - Int.floor (200 * n / 203)

theorem number_of_solutions_eq_40600 :
  {n : ℤ | 1 + Int.floor (200 * n / 203) = Int.ceil (197 * n / 200)}.to_finset.card = 40600 :=
sorry

end number_of_solutions_eq_40600_l816_816629


namespace smallest_sample_number_l816_816081

theorem smallest_sample_number (n total_products num_samples interval sample_num k : ℕ) (h1 : n = total_products)
    (h2 : num_samples = 30) (h3 : interval = total_products / num_samples) (h4 : sample_num = 105)
    (h5 : sample_num ∈ set_of (λ x, ∃ k : ℕ, x = k * interval)) : 
    ∃ m : ℕ, sample_num - (m - 1) * interval < sample_num ∧ sample_num - (m - 1) * interval = 96 := 
  sorry

end smallest_sample_number_l816_816081


namespace cost_to_cover_wall_with_tiles_l816_816868

/--
There is a wall in the shape of a rectangle with a width of 36 centimeters (cm) and a height of 72 centimeters (cm).
On this wall, you want to attach tiles that are 3 centimeters (cm) and 4 centimeters (cm) in length and width, respectively,
without any empty space. If it costs 2500 won per tile, prove that the total cost to cover the wall is 540,000 won.

Conditions:
- width_wall = 36
- height_wall = 72
- width_tile = 3
- height_tile = 4
- cost_per_tile = 2500

Target:
- Total_cost = 540,000 won
-/
theorem cost_to_cover_wall_with_tiles :
  let width_wall := 36
  let height_wall := 72
  let width_tile := 3
  let height_tile := 4
  let cost_per_tile := 2500
  let area_wall := width_wall * height_wall
  let area_tile := width_tile * height_tile
  let number_of_tiles := area_wall / area_tile
  let total_cost := number_of_tiles * cost_per_tile
  total_cost = 540000 := by
  sorry

end cost_to_cover_wall_with_tiles_l816_816868


namespace sin_minus_cos_l816_816221

variable (θ : ℝ)
hypothesis (h1 : 0 < θ ∧ θ < π / 2)
hypothesis (h2 : Real.tan θ = 1 / 3)

theorem sin_minus_cos (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := 
by 
  sorry

end sin_minus_cos_l816_816221


namespace equilateral_triangle_perimeter_l816_816854

theorem equilateral_triangle_perimeter (a P : ℕ) 
  (h1 : 2 * a + 10 = 40)  -- Condition: perimeter of isosceles triangle is 40
  (h2 : P = 3 * a) :      -- Definition of perimeter of equilateral triangle
  P = 45 :=               -- Expected result
by
  sorry

end equilateral_triangle_perimeter_l816_816854


namespace find_k_l816_816201

theorem find_k (k : ℤ) (h1 : |k| = 1) (h2 : k - 1 ≠ 0) : k = -1 :=
by
  sorry

end find_k_l816_816201


namespace line_parallel_unique_a_l816_816295

theorem line_parallel_unique_a (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y + a + 3 = 0 → x + (a + 1)*y + 4 = 0) → a = -2 :=
  by
  sorry

end line_parallel_unique_a_l816_816295


namespace snug_point_area_l816_816748

def circle_center := ℝ × ℝ
def radius := ℝ
def point := ℝ × ℝ

structure Circle :=
(center : circle_center)
(r : radius)

structure SnugPointRegion :=
(circle : Circle)
(pointH : point)
(OH_eq_6 : dist circle.center pointH = 6)

def area_snug_point_region (SPR: SnugPointRegion) : ℝ :=
  20 * π

theorem snug_point_area (SPR: SnugPointRegion) : 
  area_snug_point_region SPR = 20 * π :=
sorry

end snug_point_area_l816_816748


namespace original_number_is_five_l816_816433

theorem original_number_is_five (x : ℝ) : (x / 4 * 12 - 6 = 9) -> x = 5 :=
by
  intro h,
  have h1 : x / 4 * 12 = 15,
  { sorry, },
  have h2 : x / 4 = 5 / 4,
  { sorry, },
  have h3 : x = 5,
  { sorry, },
  exact h3

end original_number_is_five_l816_816433


namespace cos_315_proof_l816_816554

noncomputable def cos_315_eq : Prop :=
  let theta := 45 in
  let angle := 360 - theta in
  ∃ c : real, (cos 45) = (sqrt 2 / 2) ∧ (cos (360 - theta) = cos theta) ∧ 315 = angle ∧ (forall a : real, cos a = cos (360 - a)) ∧ (315 ∈ (4th quadrant)) ∧ (cos 315 = cos 45)

theorem cos_315_proof : (cos 315 = sqrt 2 / 2) :=
begin
  -- We start by defining basic facts
  -- 315 = 360 - 45
  -- Cosine is positive in the fourth quadrant, known value \cos 45 = sqrt 2 / 2
  -- Cosine property \cos (360 - \theta) = \cos \theta
  let theta := 45,
  let angle := 360 - theta,
  have H1 : cos theta = sqrt 2 / 2,
  {
    -- known fact
    sorry,
  },
  have H2 : cos (360 - theta) = cos theta,
  {
    -- cosine property
    sorry,
  },
  have H3 : angle = 315,
  {
    -- subtraction identity
    sorry,
  },
  rw H3,
  rw H2,
  rw H1,
  refl,
end

end cos_315_proof_l816_816554


namespace max_value_of_f_l816_816586

-- Define the function f(t)
def f (A B t : Real) : Real :=
  A * Real.sin t + B * Real.sin (2 * t)

-- Define the maximum function considering the given condition
def f_max (A B : Real) : Real :=
  ±(Real.sqrt (((3 * A ± (Real.sqrt (A^2 + 32 * B^2)))^3 * (-A ± Real.sqrt (A^2 + 32 * B^2)))) / (32 * B))

theorem max_value_of_f (A B : Real) (hA : A ≠ 0) (hB : B ≠ 0) :
  ∃ t : Real, f A B t = f_max A B := 
sorry

end max_value_of_f_l816_816586


namespace cos_315_eq_l816_816557

theorem cos_315_eq :
  let Q := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))
  in Q = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) →
     Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  intro hQ
  sorry

end cos_315_eq_l816_816557


namespace proof_by_contradiction_conditions_l816_816434

theorem proof_by_contradiction_conditions:
  (∃ (neg_conclusion known_conditions ax_thms_defs original_conclusion : Prop),
    (neg_conclusion ∧ known_conditions ∧ ax_thms_defs) → False)
:= sorry

end proof_by_contradiction_conditions_l816_816434


namespace balanced_strings_recurrence_base_case_b1_base_case_b2_l816_816152

-- Delta function implementation
def delta (s : String) : Int :=
  let diamonds := s.toList.count (λ c => c = '♦')
  let circles := s.toList.count (λ c => c = '○')
  diamonds - circles

-- Balanced condition
def is_balanced (s : String) : Prop :=
  (∀ t, t.isSubstringOf s -> -2 ≤ delta t ∧ delta t ≤ 2)

-- Base cases
def b_1 := 2
def b_2 := 4

-- Recurrence relation definition
def bn : ℕ → ℕ
| 0 => 0
| 1 => b_1
| 2 => b_2
| (n+2) => 2 * bn n + 2

theorem balanced_strings_recurrence (n : ℕ) :
  bn (n + 2) = 2 * bn n + 2 := sorry

theorem base_case_b1 : bn 1 = 2 := sorry
theorem base_case_b2 : bn 2 = 4 := sorry

end balanced_strings_recurrence_base_case_b1_base_case_b2_l816_816152


namespace find_k_l816_816682

def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x^2 + 2 * x + 5
def g (x : ℝ) (k : ℝ) : ℝ := x^3 - (k + 1) * x^2 - 7 * x - 8

theorem find_k (k : ℝ) (h : f 5 - g 5 k = 24) : k = -16.36 := by
  sorry

end find_k_l816_816682


namespace longer_side_is_40_l816_816978

-- Given the conditions
variable (small_rect_width : ℝ) (small_rect_length : ℝ)
variable (num_rects : ℕ)

-- Conditions 
axiom rect_width_is_10 : small_rect_width = 10
axiom length_is_twice_width : small_rect_length = 2 * small_rect_width
axiom four_rectangles : num_rects = 4

-- Prove length of the longer side of the large rectangle
theorem longer_side_is_40 :
  small_rect_width = 10 → small_rect_length = 2 * small_rect_width → num_rects = 4 →
  (2 * small_rect_length) = 40 := sorry

end longer_side_is_40_l816_816978


namespace card_area_after_shortening_l816_816796

theorem card_area_after_shortening 
  (length : ℕ) (width : ℕ) (area_after_shortening : ℕ) 
  (h_initial : length = 8) (h_initial_width : width = 3)
  (h_area_shortened_by_2 : area_after_shortening = 15) :
  (length - 2) * width = 8 :=
by
  -- Original dimensions
  let original_length := 8
  let original_width := 3
  -- Area after shortening one side by 2 inches
  let area_after_shortening_width := (original_length) * (original_width - 2)
  let area_after_shortening_length := (original_length - 2) * (original_width)
  sorry

end card_area_after_shortening_l816_816796


namespace imaginary_part_z1_mul_z2_l816_816695

def z1 : ℂ := ⟨1, -1⟩
def z2 : ℂ := ⟨2, 4⟩

theorem imaginary_part_z1_mul_z2 : (z1 * z2).im = 2 := by
  sorry

end imaginary_part_z1_mul_z2_l816_816695


namespace airplane_speed_l816_816911

noncomputable def distance : ℝ := 378.6   -- Distance in km
noncomputable def time : ℝ := 693.5       -- Time in seconds

noncomputable def altitude : ℝ := 10      -- Altitude in km
noncomputable def earth_radius : ℝ := 6370 -- Earth's radius in km

noncomputable def speed : ℝ := distance / time * 3600  -- Speed in km/h
noncomputable def adjusted_speed : ℝ := speed * (earth_radius + altitude) / earth_radius

noncomputable def min_distance : ℝ := 378.6 - 0.03     -- Minimum possible distance in km
noncomputable def max_distance : ℝ := 378.6 + 0.03     -- Maximum possible distance in km
noncomputable def min_time : ℝ := 693.5 - 1.5          -- Minimum possible time in s
noncomputable def max_time : ℝ := 693.5 + 1.5          -- Maximum possible time in s

noncomputable def max_speed : ℝ := max_distance / min_time * 3600 -- Max speed with uncertainty
noncomputable def min_speed : ℝ := min_distance / max_time * 3600 -- Min speed with uncertainty

theorem airplane_speed :
  1960 < adjusted_speed ∧ adjusted_speed < 1970 :=
by
  sorry

end airplane_speed_l816_816911


namespace overall_percent_profit_correct_l816_816061

noncomputable def cost_book_a : ℝ := 50
noncomputable def cost_book_b : ℝ := 70
noncomputable def markup_percentage_a : ℝ := 0.45
noncomputable def markup_percentage_b : ℝ := 0.63
noncomputable def discount_percentage_a : ℝ := 0.18
noncomputable def discount_percentage_b : ℝ := 0.22
noncomputable def copies_sold_a : ℕ := 5
noncomputable def copies_sold_b : ℕ := 8

noncomputable def selling_price_before_discount_a : ℝ :=
  cost_book_a * (1 + markup_percentage_a)
noncomputable def selling_price_before_discount_b : ℝ :=
  cost_book_b * (1 + markup_percentage_b)

noncomputable def selling_price_after_discount_a : ℝ :=
  selling_price_before_discount_a * (1 - discount_percentage_a)
noncomputable def selling_price_after_discount_b : ℝ :=
  selling_price_before_discount_b * (1 - discount_percentage_b)

noncomputable def total_revenue_a : ℝ :=
  copies_sold_a * selling_price_after_discount_a
noncomputable def total_revenue_b : ℝ :=
  copies_sold_b * selling_price_after_discount_b

noncomputable def total_revenue : ℝ :=
  total_revenue_a + total_revenue_b

noncomputable def total_cost_a : ℝ :=
  copies_sold_a * cost_book_a
noncomputable def total_cost_b : ℝ :=
  copies_sold_b * cost_book_b

noncomputable def total_cost : ℝ :=
  total_cost_a + total_cost_b

noncomputable def profit : ℝ :=
  total_revenue - total_cost

noncomputable def overall_percent_profit : ℝ :=
  (profit / total_cost) * 100

theorem overall_percent_profit_correct : 
  (overall_percent_profit ≈ 24.60) :=
by sorry

end overall_percent_profit_correct_l816_816061


namespace range_of_m_l816_816179

theorem range_of_m (m : ℝ) : 
  ((m - 1) * x^2 - 4 * x + 1 = 0) → 
  ((20 - 4 * m ≥ 0) ∧ (m ≠ 1)) :=
by
  sorry

end range_of_m_l816_816179


namespace last_three_digits_of_11_pow_210_l816_816115

theorem last_three_digits_of_11_pow_210 : (11 ^ 210) % 1000 = 601 :=
by sorry

end last_three_digits_of_11_pow_210_l816_816115


namespace complex_subtraction_example_l816_816431

theorem complex_subtraction_example :
  ∀ (z : ℂ), (5 - 3 * complex.i) - z = -1 + 4 * complex.i → z = 6 - 7 * complex.i :=
begin
  intros z h,
  sorry
end

end complex_subtraction_example_l816_816431


namespace max_points_for_teams_l816_816716

theorem max_points_for_teams (N : ℕ) (teams : ℕ) (points_per_win points_per_draw points_per_loss : ℕ) : 
  (teams = 15) ∧ (points_per_win = 3) ∧ (points_per_draw = 1) ∧ (points_per_loss = 0) ∧
  ∃ S F, (S.card = 6) ∧ (F.card = 9) ∧ 
  (∀ s ∈ S, s.points ≥ N) → N ≤ 34 :=
by
  sorry

end max_points_for_teams_l816_816716


namespace min_minutes_for_PlanD_cheaper_l816_816070

def PlanC_cost (x : ℕ) : ℕ := 15 * x

def PlanD_cost (x : ℕ) : ℕ := 
  if x ≤ 100 then 2500 + 4 * x 
  else 2900 + 5 * (x - 100)

theorem min_minutes_for_PlanD_cheaper (x : ℕ) : 291 ≤ x → PlanD_cost x < PlanC_cost x :=
by {
  intro h,
  unfold PlanC_cost PlanD_cost,
  split_ifs,
  { sorry }, -- Case when x ≤ 100
  { sorry }  -- Case when x > 100
}

end min_minutes_for_PlanD_cheaper_l816_816070


namespace max_x_ineql_l816_816118

noncomputable def maxXThatSatisfiesInequality (x : ℝ) : ℝ :=
  if x = 1 then 1 else 0

theorem max_x_ineql (x : ℝ) : 
  (6 + 5 * x + x^2) * sqrt (2 * x^2 - x^3 - x) ≤ 0 ↔ x ≤ 1 :=
begin
  sorry
end

end max_x_ineql_l816_816118


namespace anticipated_margin_l816_816897

noncomputable def anticipated_profit_margin (original_purchase_price : ℝ) (decrease_percentage : ℝ) (profit_margin_increase : ℝ) (selling_price : ℝ) : ℝ :=
original_purchase_price * (1 + profit_margin_increase / 100)

theorem anticipated_margin (x : ℝ) (original_purchase_price_decrease : ℝ := 0.064) (profit_margin_increase : ℝ := 8) (selling_price : ℝ) :
  selling_price = original_purchase_price * (1 + x / 100) ∧ selling_price = (1 - original_purchase_price_decrease) * (1 + (x + profit_margin_increase) / 100) →
  true :=
by
  sorry

end anticipated_margin_l816_816897


namespace correct_statements_l816_816149

def seq := ℕ → ℝ
def a (n : ℕ) : ℝ := if n = 0 then 1 else a (n - 1) - (1 / 2) * (a (n - 1))^2

theorem correct_statements (n : ℕ) (h_pos : 0 < n) :
  (∀ n : ℕ, 0 < a n ∧ a n ≤ 1) ∧ 
  (¬ ∑ i in range n + 1, a i < 2) ∧
  (∀ n : ℕ, a n ≤ 2 / (n + 1)) ∧ 
  (∀ n : ℕ, a n ≥ (1 / 2)^(n - 1)) :=
by
  sorry

end correct_statements_l816_816149


namespace angle_sum_around_point_l816_816429

theorem angle_sum_around_point (y : ℝ) (h : 170 + y + y = 360) : y = 95 := 
sorry

end angle_sum_around_point_l816_816429


namespace solution_x_y_l816_816776

theorem solution_x_y :
  ∃ x y : ℝ, y = x^2 ∧ 4 * x - 3 = 9 * (y - 7) ∧ 
             (x ≈ 2.81) ∧ (y ≈ 7.92) := 
begin
  sorry
end

end solution_x_y_l816_816776


namespace student_total_marks_l816_816062

variable (M P C : ℕ)

theorem student_total_marks :
  C = P + 20 ∧ (M + C) / 2 = 25 → M + P = 30 :=
by
  sorry

end student_total_marks_l816_816062


namespace problem_1_problem_2_l816_816676

noncomputable def U := Set ℝ
def A (a : ℝ) := {x : ℝ | ∃ y : ℝ, y = 1 / (Real.sqrt (a - x))}
def B := {x : ℝ | x^2 - x - 6 = 0}

theorem problem_1 (a : ℝ) (h : a = -1) : A a ∩ B = {-2} :=
by sorry

theorem problem_2 (a : ℝ) (h : (U \ A a) ∩ B = ∅) : 3 < a :=
by sorry

end problem_1_problem_2_l816_816676


namespace appoint_positions_l816_816425

theorem appoint_positions:
  let candidates := ['A', 'B', 'C', 'D', 'E']
  let positions := ['class_president', 'vice_president', 'secretary']
  let restrict_pos := [('A', 'class_president'), ('B', 'vice_president'), ('C', 'secretary')]
  (number_of_ways_to_appoint := 
    perm_count_ways(candidates, positions, restrict_pos) = 32) :=
sorry

end appoint_positions_l816_816425


namespace parabola_focus_and_directrix_proof_l816_816963

noncomputable def parabola_focus_and_directrix (p : ℝ) (h : 2 * p = 8) : (ℝ × ℝ) × (ℝ → Prop) :=
  let focus := (0, p / 2)
  let directrix := λ y, y = -p / 2
  ((0, 2), λ y, y = -2)

theorem parabola_focus_and_directrix_proof :
  parabola_focus_and_directrix 4 (by norm_num : 2 * 4 = 8) = ((0, 2), λ y, y = -2) := 
  sorry

end parabola_focus_and_directrix_proof_l816_816963


namespace sin_minus_cos_l816_816268

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end sin_minus_cos_l816_816268


namespace pentagon_area_correct_l816_816056

-- Define the given conditions
def pentagon_sides : List ℕ := [9, 16, 30, 40, 41]

-- Assume right-angled triangle removed has legs 9, 40 and hypotenuse 41
def leg_u : ℕ := 9
def leg_v : ℕ := 40
def hypotenuse_t : ℕ := nat.sqrt (leg_u ^ 2 + leg_v ^ 2)

-- Conditions provided as assignments
def side_r : ℕ := 30
def side_s : ℕ := 40
def side_p : ℕ := 16
def side_q : ℕ := 9

-- Define the calculated areas
noncomputable def area_rectangle: ℕ := side_r * side_s
noncomputable def area_triangle: ℕ := (leg_u * leg_v) / 2
noncomputable def area_pentagon: ℕ := area_rectangle - area_triangle

-- Theorem statement for proof
theorem pentagon_area_correct : area_pentagon = 1020 := by
  -- auto-generated by translation steps
  sorry

end pentagon_area_correct_l816_816056


namespace find_a_l816_816981

/-- 
Given sets A and B defined by specific quadratic equations, 
if A ∪ B = A, then a ∈ (-∞, 0).
-/
theorem find_a :
  ∀ (a : ℝ),
    (A = {x : ℝ | x^2 - 3 * x + 2 = 0}) →
    (B = {x : ℝ | x^2 - 2 * a * x + a^2 - a = 0}) →
    (A ∪ B = A) →
    a < 0 :=
by
  sorry

end find_a_l816_816981


namespace comparison_problem_l816_816986

/-- Define conditions -/
def a : ℝ := Real.sin 4
def b : ℝ := Real.log 15 / Real.log 2 - Real.log 5 / Real.log 2
def c : ℝ := 2^(-0.1)

/-- Proof problem -/
theorem comparison_problem :
  (b > c) ∧ (c > a) :=
by
  sorry

end comparison_problem_l816_816986


namespace find_solutions_l816_816606

noncomputable def solutions_to_equation : set Complex :=
  {x | x^4 - 16 = 0}

theorem find_solutions : solutions_to_equation = {2, -2, Complex.I * 2, -Complex.I * 2} :=
sorry

end find_solutions_l816_816606


namespace no_solutions_exist_l816_816122

theorem no_solutions_exist : ¬ ∃ (x y z : ℝ), x + y = 3 ∧ xy - z^2 = 2 :=
by sorry

end no_solutions_exist_l816_816122


namespace hannah_flour_calculation_l816_816193

theorem hannah_flour_calculation :
  ∀ (bananas mushRatio flourRatio: ℕ), 
    3 = flourRatio →
    4 = mushRatio →
    20 = bananas →
    (bananas / mushRatio) * flourRatio = 15 :=
by
  intros bananas mushRatio flourRatio h_flour h_mush h_bananas
  rw [h_flour, h_mush, h_bananas]
  sorry

end hannah_flour_calculation_l816_816193


namespace imaginary_part_z1_mul_z2_l816_816694

def z1 : ℂ := ⟨1, -1⟩
def z2 : ℂ := ⟨2, 4⟩

theorem imaginary_part_z1_mul_z2 : (z1 * z2).im = 2 := by
  sorry

end imaginary_part_z1_mul_z2_l816_816694


namespace cos_315_is_sqrt2_div_2_l816_816522

noncomputable def cos_315_eq : Prop := 
  real.cos (315 * real.pi / 180) = real.sqrt 2 / 2

theorem cos_315_is_sqrt2_div_2 : cos_315_eq := by 
  sorry

end cos_315_is_sqrt2_div_2_l816_816522


namespace Q_eq_G_l816_816188

def P := {y | ∃ x, y = x^2 + 1}
def Q := {y : ℝ | ∃ x, y = x^2 + 1}
def E := {x : ℝ | ∃ y, y = x^2 + 1}
def F := {(x, y) | y = x^2 + 1}
def G := {x : ℝ | x ≥ 1}

theorem Q_eq_G : Q = G := by
  sorry

end Q_eq_G_l816_816188


namespace max_line_segments_l816_816718

theorem max_line_segments (P : Finset (EuclideanSpace ℝ 2)) (hP_card : P.card = 100)
  (h_dist : ∀ p1 p2 ∈ P, p1 ≠ p2 → dist p1 p2 ≥ 3)
  (h_exact3 : ∀ p1 p2 ∈ P, dist p1 p2 = 3 → true) :
  ∃ (l : Finset (Finset (EuclideanSpace ℝ 2))), l.card ≤ 300 ∧
    ∀ (e ∈ l), ∃ p1 p2, e = {p1, p2} ∧ p1 ∈ P ∧ p2 ∈ P ∧ dist p1 p2 = 3 :=
by
  sorry

end max_line_segments_l816_816718


namespace elimination_tournament_games_l816_816484

theorem elimination_tournament_games (n : ℕ) (h : n = 17) : 
  single_elimination_num_games n = 16 := 
begin
  sorry
end

def single_elimination_num_games (n : ℕ) : ℕ :=
n - 1

end elimination_tournament_games_l816_816484


namespace batsman_running_percentage_l816_816442

theorem batsman_running_percentage (total_runs boundary_runs six_runs : ℕ) 
  (h1 : total_runs = 120) (h2 : boundary_runs = 3 * 4) (h3 : six_runs = 8 * 6) : 
  (total_runs - (boundary_runs + six_runs)) * 100 / total_runs = 50 := 
sorry

end batsman_running_percentage_l816_816442


namespace task1_probability_l816_816880

noncomputable def P_Task1 : ℝ :=
  let P_Task2 : ℝ := 3 / 5
  let P_Task1_and_not_Task2 : ℝ := 0.25
  let P_not_Task2 : ℝ := 1 - P_Task2
  have h1 : P_Task1_and_not_Task2 = P_Task1 * P_not_Task2 := 
    by sorry
  have h2 : P_not_Task2 = 2 / 5 := 
    by sorry
  (P_Task1_and_not_Task2 / (2 / 5))

theorem task1_probability : P_Task1 = 0.625 := 
  by
    -- use h1 and h2 here after defining P_Task1_and_not_Task2 and P_not_Task2 appropriately
    sorry

end task1_probability_l816_816880


namespace inequality_must_hold_l816_816288

theorem inequality_must_hold (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := 
sorry

end inequality_must_hold_l816_816288


namespace letters_with_both_l816_816310

-- Definitions based on conditions
def L := 24   -- Letters with a straight line but no dot
def Total := 40  -- Total number of letters
def D := 5    -- Letters with a dot but no straight line

-- Proof statement
theorem letters_with_both : (∃ DL : ℕ, Total = D + L + DL ∧ DL = 11) :=
by
  sorry

end letters_with_both_l816_816310


namespace hypotenuse_length_l816_816397

theorem hypotenuse_length (a c : ℝ) (h1 : 2a + c = 4 + 4 * Real.sqrt 2) (h2 : c^2 = 2 * a^2) : c = 4 := 
by
  sorry

end hypotenuse_length_l816_816397


namespace initial_students_per_class_l816_816047

theorem initial_students_per_class
  (S : ℕ) 
  (parents chaperones left_students left_chaperones : ℕ)
  (teachers remaining_individuals : ℕ)
  (h1 : parents = 5)
  (h2 : chaperones = 2)
  (h3 : left_students = 10)
  (h4 : left_chaperones = 2)
  (h5 : teachers = 2)
  (h6 : remaining_individuals = 15)
  (h7 : 2 * S + parents + teachers - left_students - left_chaperones = remaining_individuals) :
  S = 10 :=
by
  sorry

end initial_students_per_class_l816_816047


namespace ratio_no_retirement_plan_l816_816713

theorem ratio_no_retirement_plan (num_men num_women : ℕ)
  (percent_women_no_plan percent_men_with_plan : ℚ)
  (total_men total_women : ℕ)
  (h_men : num_men = 120)
  (h_women : num_women = 180)
  (h_percent_women_no_plan : percent_women_no_plan = 0.60)
  (h_percent_men_with_plan : percent_men_with_plan = 0.40) :
  let total_workers := num_men + num_women,
      num_workers_no_plan := 450, -- from the solution 40% of 180 = W
      num_workers_with_plan := 200, -- from the solution 60% of 120 = R
      ratio := (num_workers_no_plan : ℚ) / (num_workers_no_plan + num_workers_with_plan)
  in ratio = 9 / 13 :=
by
  sorry

end ratio_no_retirement_plan_l816_816713


namespace complement_of_angle_l816_816656

def degrees (d m s : Nat) : ℝ :=
  d + m / 60 + s / 3600

theorem complement_of_angle : 
  ∀ (α : ℝ), α = degrees 36 14 25 → 90 - α = degrees 53 45 35 :=
by 
  intro α h
  sorry

end complement_of_angle_l816_816656


namespace exists_point_E_l816_816035

variables {O A B C D P E : Type} [circle γ O A B]
variables [midpoint_of B D C] [intersection_of AC DO P]

theorem exists_point_E :
  ∃ E : Point, (E ∈ AB) ∧ (P ∈ circle_with_diameter A E) := 
sorry

end exists_point_E_l816_816035


namespace cats_in_shelter_after_events_l816_816505

theorem cats_in_shelter_after_events :
  let initial_cats := 12
  let already_had_cats := initial_cats / 2
  let total_before_events := initial_cats + already_had_cats
  let after_adoption := total_before_events - 3
  let after_kittens := after_adoption + 5
  let after_missing_pet := after_kittens - 1
  in after_missing_pet = 19 :=
by
  let initial_cats := 12
  let already_had_cats := initial_cats / 2
  let total_before_events := initial_cats + already_had_cats
  let after_adoption := total_before_events - 3
  let after_kittens := after_adoption + 5
  let after_missing_pet := after_kittens - 1
  show after_missing_pet = 19
  sorry

end cats_in_shelter_after_events_l816_816505


namespace maximize_tables_l816_816420

theorem maximize_tables (wood_total : ℝ) (wood_per_tabletop : ℝ) (wood_per_leg : ℝ)
  (max_tabletops : ℝ) (max_legs : ℝ) (tabletops_needed : ℝ) (legs_needed : ℝ) :
  wood_total = 12 →
  wood_per_tabletop = 1 / 20 →
  wood_per_leg = 1 / 400 →
  max_tabletops = wood_total / wood_per_tabletop →
  max_legs = wood_total / wood_per_leg →
  tabletops_needed = 10 →
  legs_needed = 2 →
  (wood_per_tabletop * tabletops_needed + wood_per_leg * (legs_needed * 4) = 12) :=
begin
  sorry -- Proof steps go here
end

end maximize_tables_l816_816420


namespace minimum_value_f_two_tangents_through_A_l816_816669

noncomputable def f (x : ℝ) : ℝ := x * (Real.log x + 1)

theorem minimum_value_f :
  (∀ x : ℝ, 0 < x → f x ≥ -Real.exp (-2)) ∧ (∃ x : ℝ, 0 < x ∧ f x = -Real.exp (-2)) :=
by {
  sorry
}

variable (a b : ℝ) (h1 : 0 < b) (h2 : b < a * Real.log a + a)

theorem two_tangents_through_A :
  ∀ x : ℝ, e^{-2} < x → ∃ x₀ x₁ : ℝ, x₀ ≠ x₁ ∧ 
           (b - f x₀ = f' x₀ * (a - x₀)) ∧ 
           (b - f x₁ = f' x₁ * (a - x₁)) :=
by {
  sorry
}

lemma derivative_f (x : ℝ) : deriv f x = Real.log x + 2 := by {
  sorry
}

end minimum_value_f_two_tangents_through_A_l816_816669


namespace paper_clips_collected_l816_816361

theorem paper_clips_collected (boxes paper_clips_per_box total_paper_clips : ℕ) 
  (h1 : boxes = 9) 
  (h2 : paper_clips_per_box = 9) 
  (h3 : total_paper_clips = boxes * paper_clips_per_box) : 
  total_paper_clips = 81 :=
by {
  sorry
}

end paper_clips_collected_l816_816361


namespace fixed_point_and_min_radius_l816_816842

theorem fixed_point_and_min_radius :
  (∀ k : ℝ, ∃ (x y : ℝ), y = k * x - k + 1 ∧ x = 1 ∧ y = 1) ∧
  (∀ k : ℝ, ∀ (r : ℝ), (∃ (x y : ℝ), y = k * x - k + 1 ∧ x^2 + y^2 = r^2) → r ≥ sqrt 2) :=
by
  sorry

end fixed_point_and_min_radius_l816_816842


namespace max_product_lmbda2_lmbda3_l816_816305

variables {A B C P E F : Type} [AffineSpace A B C P E F] 
variables {x y : ℝ}
variables (AE EB AF FC : ℝ)

-- Assume AE:EB = 1:2 and AF:FC = 1:2
def AE.EB_ratio (hAE.EB : AE / EB = 1 / 2) : Prop := AE = EB * (1 / 2)
def AF.FC_ratio (hAF.FC : AF / FC = 1 / 2) : Prop := AF = FC * (1 / 2)

-- Assume areas of triangles given points P, B, and C
variables (S S1 S2 S3 : ℝ)
def λ1 : ℝ := S1 / S
def λ2 : ℝ := S2 / S
def λ3 : ℝ := S3 / S

axiom areas_relation : λ1 + λ2 + λ3 = 1

-- Assume point P lies on EF
variable (hP_on_EF : ∃ t : ℝ, t ∈ Icc (0 : ℝ) 1 ∧ P = t • E + (1 - t) • F)

-- Condition given in the problem statement
axiom condition_PA : ∀ ⦃PA PB PC : B⦄, PA = - (x • PB + y • PC)

theorem max_product_lmbda2_lmbda3 :  AE.EB_ratio AE EB ∧ AF.FC_ratio AF FC ∧ areas_relation λ1 λ2 λ3 ∧ hP_on_EF P E F 
 → 2 * x + y = 3 / 4 := 
sorry

end max_product_lmbda2_lmbda3_l816_816305


namespace find_point_D_l816_816154

structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define points A, B, and C
def A : Point := ⟨1, -1⟩
def B : Point := ⟨2, 2⟩
def C : Point := ⟨3, 0⟩

-- Function to calculate the slope of a line segment given two points
def slope (p1 p2 : Point) : ℝ :=
  if p1.x = p2.x then 0 -- Vertical line, slope undefined
  else (p2.y - p1.y) / (p2.x - p1.x)

-- Define the problem conditions as Lean hypothesis
theorem find_point_D (D : Point) :
  slope C D * slope A B = -1 ∧ slope D A = slope C B → D = ⟨0, 1⟩ :=
by
  -- Proof goes here
  sorry

end find_point_D_l816_816154


namespace option_A_option_B_option_C_option_D_l816_816685

variables (m n a : ℝ^3)
hypothesis (m_nonzero : m ≠ 0)
hypothesis (n_nonzero : n ≠ 0)
hypothesis (a_nonzero : a ≠ 0)

theorem option_A : 0 * m = 0 := by
  sorry

theorem option_B (h : a * (n • m) = n * (a • m)) : False := by
  sorry

theorem option_C (h : m • a = n • a) : m = n := by
  sorry

theorem option_D (h_collinear : ∃ k : ℝ, m = k • n) (h_same_direction : 0 < ((m • n) / (|m| * |n|))) : 
  proj(m, n) = (|m| / |n|) • n := by
  sorry

end option_A_option_B_option_C_option_D_l816_816685


namespace total_shaded_area_l816_816423

theorem total_shaded_area (side_len : ℝ) (segment_len : ℝ) (h : ℝ) :
  side_len = 8 ∧ segment_len = 1 ∧ 0 ≤ h ∧ h ≤ 8 →
  (segment_len * h / 2 + segment_len * (side_len - h) / 2) = 4 := 
by
  intro h_cond
  rcases h_cond with ⟨h_side_len, h_segment_len, h_nonneg, h_le⟩
  -- Directly state the simplified computation
  sorry

end total_shaded_area_l816_816423


namespace sin_minus_cos_l816_816240

noncomputable def theta_condition (θ : ℝ) : Prop := (0 < θ) ∧ (θ < π / 2) ∧ (Real.tan θ = 1 / 3)

theorem sin_minus_cos (θ : ℝ) (hθ : theta_condition θ) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_minus_cos_l816_816240


namespace train_cross_pole_time_approx_l816_816489

-- Define the conditions
def speed_kmh := 30 -- Speed in km/hr
def length_m := 75  -- Length of the train in meters

-- Convert speed from km/hr to m/s
def speed_ms := speed_kmh * (1000.0 / 3600.0)

-- Stating the problem in Lean
theorem train_cross_pole_time_approx :
  let time := length_m / speed_ms in
  time ≈ 9.00 := 
by
  have h_speed_ms : speed_ms = speed_kmh * (1000.0 / 3600.0) := by rfl
  have h_length_m : length_m = 75 := by rfl
  have h_time : time = length_m / speed_ms := by rfl
  sorry

end train_cross_pole_time_approx_l816_816489


namespace full_house_plus_prob_correct_l816_816007

-- Define the standard deck of 52 cards and the full house plus hand criteria
def deck_size := 52
def hand_size := 6
def trio_size := 3
def pair_size := 2
def full_house_plus_cards (cards : Set ℕ) : Prop := 
  ∃ (r1 r2 : ℕ) (c1 c2 c3 c4 : ℕ), 
      -- r1 and r2 are distinct ranks
      r1 ≠ r2 ∧
      -- c1, c2, c3 form a trio of rank r1
      cards.count r1 = 3 ∧ 
      -- c4, c5 form a pair of rank r2
      cards.count r2 = 2 ∧ 
      -- c6 is an additional card not forming another pair or four of a kind with trio and pair
      hand_size - trio_size - pair_size = 1 

-- Number of ways to choose 6 cards from 52 cards
noncomputable def total_ways_to_choose_hand := Nat.choose deck_size hand_size

-- Number of favorable ways to draw a full house plus hand
noncomputable def favorable_full_house_plus :=
  13 * 4 * 12 * 6 * 44

-- Probability of drawing a full house plus hand
noncomputable def full_house_plus_probability : ℚ :=
  favorable_full_house_plus / total_ways_to_choose_hand

theorem full_house_plus_prob_correct :
  full_house_plus_probability = 82368 / 10179260 := sorry

end full_house_plus_prob_correct_l816_816007


namespace sin_minus_cos_l816_816256

theorem sin_minus_cos (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < (Real.pi / 2)) (hθ3 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := by
  sorry

end sin_minus_cos_l816_816256


namespace power_of_log_eq_l816_816985

theorem power_of_log_eq (a : ℝ) (h : a = Real.log 4 / Real.log 3) : 3^(2 * a) = 16 :=
  sorry

end power_of_log_eq_l816_816985


namespace xy_squares_l816_816291

theorem xy_squares (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 4) : x^2 - y^2 = 40 := 
by 
  sorry

end xy_squares_l816_816291


namespace length_of_base_l816_816087

variables {A B C : Type} [MetricSpace A B C]

structure IsoscelesTriangle (A B C : A) :=
(isosceles : ∃AB AC : ℝ, AB = AC)
(double_length : ∃AB BC : ℝ, AB = 2 * BC)
(length_AB : ∃AB : ℝ, AB = 10)

theorem length_of_base (h : IsoscelesTriangle A B C) : BC = 5 := by
  sorry

end length_of_base_l816_816087


namespace cos_315_proof_l816_816548

noncomputable def cos_315_eq : Prop :=
  let theta := 45 in
  let angle := 360 - theta in
  ∃ c : real, (cos 45) = (sqrt 2 / 2) ∧ (cos (360 - theta) = cos theta) ∧ 315 = angle ∧ (forall a : real, cos a = cos (360 - a)) ∧ (315 ∈ (4th quadrant)) ∧ (cos 315 = cos 45)

theorem cos_315_proof : (cos 315 = sqrt 2 / 2) :=
begin
  -- We start by defining basic facts
  -- 315 = 360 - 45
  -- Cosine is positive in the fourth quadrant, known value \cos 45 = sqrt 2 / 2
  -- Cosine property \cos (360 - \theta) = \cos \theta
  let theta := 45,
  let angle := 360 - theta,
  have H1 : cos theta = sqrt 2 / 2,
  {
    -- known fact
    sorry,
  },
  have H2 : cos (360 - theta) = cos theta,
  {
    -- cosine property
    sorry,
  },
  have H3 : angle = 315,
  {
    -- subtraction identity
    sorry,
  },
  rw H3,
  rw H2,
  rw H1,
  refl,
end

end cos_315_proof_l816_816548


namespace necessary_but_not_sufficient_l816_816296

-- Define the propositions P and Q
def P (a b : ℝ) : Prop := a^2 + b^2 > 2 * a * b
def Q (a b : ℝ) : Prop := abs (a + b) < abs a + abs b

-- Define the conditions for P and Q
def condition_for_P (a b : ℝ) : Prop := a ≠ b
def condition_for_Q (a b : ℝ) : Prop := a * b < 0

-- Define the statement
theorem necessary_but_not_sufficient (a b : ℝ) :
  (P a b → Q a b) ∧ ¬ (Q a b → P a b) :=
by
  sorry

end necessary_but_not_sufficient_l816_816296


namespace rational_numbers_l816_816094

-- Definitions for each of the given numbers
def number1 := Real.sqrt ((Real.pi ^ 3) ^ (2 / 3))
def number2 := Real.root (1.25) 3
def number3 := Real.root (0.001) 4
def number4 := Real.root (1) 3 * Real.sqrt ((0.04) ^ (-1))
def number5 := Real.sqrt 25

-- Stating the mathematical problem
theorem rational_numbers :
  Real.is_rational number3 ∧ 
  Real.is_rational number4 ∧
  Real.is_rational number5 ∧
  ¬ Real.is_rational number1 ∧
  ¬ Real.is_rational number2 :=
by {
  sorry -- proof details are omitted
}

end rational_numbers_l816_816094


namespace baker_cake_remains_l816_816509

theorem baker_cake_remains (initial_sold : ℕ) (initial_made : ℕ) (h_sold : initial_sold = 44) (h_made : initial_made = 48) : initial_made - initial_sold = 4 :=
by
  rw [h_sold, h_made]
  simp
  sorry

end baker_cake_remains_l816_816509


namespace smallest_mu_exists_l816_816587

theorem smallest_mu_exists (a b c d : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) :
  ∃ μ : ℝ, μ = (3 / 2) - (3 / (4 * Real.sqrt 2)) ∧ 
    (a^3 + b^3 + c^3 + d^3 ≥ a^2 * b + μ * b^2 * c + c^2 * d) :=
by
  sorry

end smallest_mu_exists_l816_816587


namespace coefficient_of_x4_in_expansion_l816_816326

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
if h : k ≤ n then nat.choose n k else 0

theorem coefficient_of_x4_in_expansion : 
  (∃ (f g : ℤ) (n : ℕ) (x : α) (α : Type) (coeff : ℤ), 
    (f = 2 ∧ g = (-1/2)) ∧ 
    (n = 6) ∧ 
    (coeff = binomial_coefficient 6 4 * f^(6-4) * g^4 * (2^2)) ∧ 
    (coeff = 60)) :=
by
  sorry

end coefficient_of_x4_in_expansion_l816_816326


namespace necessary_sufficient_condition_for_areas_l816_816314

def circle_area_condition (r : ℝ) (φ : ℝ) : Prop :=
  0 < φ ∧ φ < π / 4 ∧
  (φ * r^2 + (1/2) * r^2 * sin φ = φ * r^2 + (1/2) * r^2 * sin (2 * φ))

theorem necessary_sufficient_condition_for_areas (r : ℝ) (φ : ℝ) :
  0 < φ ∧ φ < π / 4 → circle_area_condition r φ ↔ φ = 0 :=
by { sorry }

end necessary_sufficient_condition_for_areas_l816_816314


namespace optionA_optionB_optionC_optionD_correct_options_l816_816153

noncomputable def z1 : ℂ := 1 - 3 * Complex.I
noncomputable def z2 : ℂ := (2 - Complex.I) ^ 2
noncomputable def z3 : ℂ := (8 + 10 * Complex.I) / (1 + Complex.I)

theorem optionA : Complex.conj (z1 + z2) = 4 + 7 * Complex.I :=
by sorry

theorem optionB : ∃ r : ℝ, (z1.re = 1) ∧ (z2.re = 3) ∧ (z3.re = 9) ∧ (3 / 1 = r) ∧ (9 / 3 = r) :=
by sorry

theorem optionC : Real.sqrt 10 * Complex.abs z1 = 2 * Complex.abs z2 :=
by sorry

theorem optionD : ¬(∃ d : ℝ, (-3 = -4 + d) ∧ (-4 = 1 + d)) :=
by sorry

theorem correct_options : (optionA ∧ optionB ∧ optionC) ∧ ¬optionD :=
by sorry

end optionA_optionB_optionC_optionD_correct_options_l816_816153


namespace compute_infinite_series_l816_816759

noncomputable def infinite_series (c d : ℝ) (hcd : c > d) : ℝ :=
  ∑' n, 1 / (((n - 1 : ℝ) * c - (n - 2 : ℝ) * d) * (n * c - (n - 1 : ℝ) * d))

theorem compute_infinite_series (c d : ℝ) (hcd : c > d) :
  infinite_series c d hcd = 1 / ((c - d) * d) :=
by
  sorry

end compute_infinite_series_l816_816759


namespace max_value_ahn_operation_l816_816068

theorem max_value_ahn_operation :
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (300 - n)^2 - 10 = 39990 :=
by
  sorry

end max_value_ahn_operation_l816_816068


namespace graveyard_bones_l816_816320

theorem graveyard_bones : 
  let total_skeletons := 36
  let adult_women_skeletons := total_skeletons / 3
  let adult_men_skeletons := total_skeletons / 4
  let children_skeletons := total_skeletons / 6
  let teenagers_skeletons := total_skeletons / 9
  let elderly_skeletons := total_skeletons - (adult_women_skeletons + adult_men_skeletons + children_skeletons + teenagers_skeletons)

  let adult_women_bones := 22
  let adult_men_bones := adult_women_bones + 6
  let children_bones := adult_women_bones - 12
  let teenagers_bones := children_bones * 2
  let elderly_bones := adult_men_bones - 3

  let total_bones := (
    adult_women_skeletons * adult_women_bones + 
    adult_men_skeletons * adult_men_bones + 
    children_skeletons * children_bones + 
    teenagers_skeletons * teenagers_bones + 
    elderly_skeletons * elderly_bones
  )
in total_bones = 781 :=
begin
  sorry
end

end graveyard_bones_l816_816320


namespace min_modulus_of_complex_l816_816762

noncomputable def minimal_modulus (z : ℂ) : ℝ :=
  if h : (|z - 5 * complex.I| + |z - 6| = 7) then min |z| else 0

theorem min_modulus_of_complex (z : ℂ) (h : |z - 5 * complex.I| + |z - 6| = 7) : |z| = 30 / real.sqrt 61 := by
  sorry

end min_modulus_of_complex_l816_816762


namespace average_student_headcount_l816_816089

theorem average_student_headcount (headcount_03_04 headcount_04_05 : ℕ) 
  (h1 : headcount_03_04 = 10500) 
  (h2 : headcount_04_05 = 10700) : 
  (headcount_03_04 + headcount_04_05) / 2 = 10600 := 
by
  sorry

end average_student_headcount_l816_816089


namespace largest_n_divisible_l816_816009

theorem largest_n_divisible (n : ℕ) (h : (n : ℤ) > 0) : 
  (n^3 + 105) % (n + 12) = 0 ↔ n = 93 :=
sorry

end largest_n_divisible_l816_816009


namespace calc_man_dividend_l816_816051

noncomputable def calc_dividend (investment : ℝ) (face_value : ℝ) (premium : ℝ) (dividend_percent : ℝ) : ℝ :=
  let cost_per_share := face_value * (1 + premium / 100)
  let number_of_shares := investment / cost_per_share
  let dividend_per_share := dividend_percent / 100 * face_value
  let total_dividend := dividend_per_share * number_of_shares
  total_dividend

theorem calc_man_dividend :
  calc_dividend 14400 100 20 5 = 600 :=
by
  sorry

end calc_man_dividend_l816_816051


namespace cos_315_eq_sqrt2_div2_l816_816539

theorem cos_315_eq_sqrt2_div2 : Real.cos (315 * Real.pi / 180) = sqrt 2 / 2 := 
sorry

end cos_315_eq_sqrt2_div2_l816_816539


namespace forest_farm_stabilization_l816_816907

theorem forest_farm_stabilization :
  ∃ N: ℕ, ∀ n: ℕ, (n > N) → 
  let f (t: ℕ) := (t * 9 / 10 + 2500) in
  abs ((f (f t)) - 25000) < ε :=
by sorry

end forest_farm_stabilization_l816_816907


namespace ludwig_weekly_salary_is_55_l816_816783

noncomputable def daily_salary : ℝ := 10
noncomputable def full_days : ℕ := 4
noncomputable def half_days : ℕ := 3
noncomputable def half_day_salary := daily_salary / 2

theorem ludwig_weekly_salary_is_55 :
  (full_days * daily_salary + half_days * half_day_salary = 55) := by
  sorry

end ludwig_weekly_salary_is_55_l816_816783


namespace ineq_i_d_l816_816969

def d (n : ℕ) (p : Fin n → Fin n) : ℤ :=
  ∑ i in Finset.univ, |p i - i|

def i (n : ℕ) (p : Fin n → Fin n) : ℕ :=
  Finset.card {pair ∈ Finset.univ.product Finset.univ | pair.1 < pair.2 ∧ p pair.1 > p pair.2}

theorem ineq_i_d (n : ℕ) (p : Fin n → Fin n) : 
  i n p ≤ d n p :=
sorry

end ineq_i_d_l816_816969


namespace sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816207

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816207


namespace fraction_meaningful_l816_816856

theorem fraction_meaningful (x : ℝ) : (∃ z, z = 3 / (x - 4)) ↔ x ≠ 4 :=
by
  sorry

end fraction_meaningful_l816_816856


namespace car_speed_to_keep_window_dry_l816_816889

theorem car_speed_to_keep_window_dry :
  ∀ (v : ℝ) (alpha : ℝ), v = 2 ∧ α = 60 * Real.pi / 180 → 
  (∃ u : ℝ, u = 2 / Real.sqrt 3) :=
by
  intros v alpha h
  cases h
  use 2 / Real.sqrt 3
  sorry

end car_speed_to_keep_window_dry_l816_816889


namespace factorization_of_polynomial_l816_816024

noncomputable def p (x : ℤ) : ℤ := x^15 + x^10 + x^5 + 1
noncomputable def f (x : ℤ) : ℤ := x^3 + x^2 + x + 1
noncomputable def g (x : ℤ) : ℤ := x^12 - x^11 + x^9 - x^8 + x^6 - x^5 + x^3 - x^2 + x - 1

theorem factorization_of_polynomial : ∀ x : ℤ, p x = f x * g x :=
by sorry

end factorization_of_polynomial_l816_816024


namespace find_divisor_l816_816436

theorem find_divisor (d x k j : ℤ) (h₁ : x = k * d + 5) (h₂ : 7 * x = j * d + 8) : d = 11 :=
sorry

end find_divisor_l816_816436


namespace earnings_percentage_difference_l816_816030

-- Defining the conditions
def MikeEarnings : ℕ := 12
def PhilEarnings : ℕ := 6

-- Proving the percentage difference
theorem earnings_percentage_difference :
  ((MikeEarnings - PhilEarnings: ℕ) * 100 / MikeEarnings = 50) :=
by 
  sorry

end earnings_percentage_difference_l816_816030


namespace sin_minus_cos_l816_816260

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end sin_minus_cos_l816_816260


namespace find_constant_l816_816445

variable {R : Type} [LinearOrderedField R]

-- Define the function f
variable (f : R → R)

-- Define the constant c
variable (c : R)

-- Condition 1: f(x) + c * f(8 - x) = x for all real numbers x
def condition1 : Prop := ∀ x : R, f(x) + c * f(8 - x) = x

-- Condition 2: f(2) = 2
def condition2 : Prop := f(2) = 2

-- The theorem to prove: c = 3 given the conditions
theorem find_constant (h1 : condition1 f c) (h2 : condition2 f) : c = 3 :=
sorry

end find_constant_l816_816445


namespace polynomial_roots_not_integers_l816_816749

noncomputable def polynomial_integers_not_divisible (f : ℤ[X]) (k : ℕ) (n : ℤ) : Prop :=
  ∀ i : ℕ, i < k → ¬ (k ∣ (f.eval (n + i)))

theorem polynomial_roots_not_integers (f : ℤ[X]) (k : ℕ) (n : ℤ) (hk : 0 < k)
  (h : polynomial_integers_not_divisible f k n) :
  ∀ x : ℤ, ¬ (f.eval x = 0) :=
begin
  sorry
end

end polynomial_roots_not_integers_l816_816749


namespace max_frac_sum_l816_816995

theorem max_frac_sum {n : ℕ} (h_n : n > 1) :
  ∀ (a b c d : ℕ), (a + c ≤ n) ∧ (b > 0) ∧ (d > 0) ∧
  (a * d + b * c < b * d) → 
  ↑a / ↑b + ↑c / ↑d ≤ (1 - 1 / ( ⌊(2*n : ℝ)/3 + 1/6⌋₊ + 1) * ( ⌊(2*n : ℝ)/3 + 1/6⌋₊ * (n - ⌊(2*n : ℝ)/3 + 1/6⌋₊) + 1)) :=
by sorry

end max_frac_sum_l816_816995


namespace new_pressure_of_helium_l816_816507

noncomputable def helium_pressure (p V p' V' : ℝ) (k : ℝ) : Prop :=
  p * V = k ∧ p' * V' = k

theorem new_pressure_of_helium :
  ∀ (p V p' V' k : ℝ), 
  p = 8 ∧ V = 3.5 ∧ V' = 7 ∧ k = 28 →
  helium_pressure p V p' V' k →
  p' = 4 :=
by
  intros p V p' V' k h1 h2
  sorry

end new_pressure_of_helium_l816_816507


namespace y_increase_for_x_increase_l816_816325

def slope_change (dx dy x_increase : ℕ) : ℕ :=
  (x_increase / dx) * dy

theorem y_increase_for_x_increase :
  ∀ (dx dy x_increase : ℕ), dx = 5 ∧ dy = 11 ∧ x_increase = 15 → slope_change dx dy x_increase = 33 :=
by
  intros dx dy x_increase h
  cases h with hdx rest
  cases rest with hdy hx_increase
  rw [hdx, hdy, hx_increase]
  unfold slope_change
  norm_num
  -- proof omitted
  sorry

end y_increase_for_x_increase_l816_816325


namespace chocolate_distribution_sum_of_digits_l816_816345

theorem chocolate_distribution_sum_of_digits : 
  let N := (3! * Nat.choose 8 1 * Nat.choose 7 2 * Nat.choose 5 5) +
           (3! * Nat.choose 8 1 * Nat.choose 7 3 * Nat.choose 4 4) in
  (N.digits.sum) = 24 :=
by
  sorry

end chocolate_distribution_sum_of_digits_l816_816345


namespace groups_of_men_and_women_l816_816324

theorem groups_of_men_and_women :
  let men := 4 in
  let women := 5 in
  let group1 := 2 in
  let group2 := 3 in
  let group3 := 4 in
  (((nat.choose men 1) * (nat.choose women 1)) * 
   ((nat.choose (men - 1) 1) * (nat.choose (women - 1) 2)) *
   ((nat.choose (men - 2) 2) * (nat.choose (women - 3) 2))) = 360 :=
by
  sorry

end groups_of_men_and_women_l816_816324


namespace tim_balloon_count_l816_816084

theorem tim_balloon_count (Dan_balloons : ℕ) (h1 : Dan_balloons = 59) (Tim_balloons : ℕ) (h2 : Tim_balloons = 11 * Dan_balloons) : Tim_balloons = 649 :=
sorry

end tim_balloon_count_l816_816084


namespace equiv_triangle_sum_of_squares_l816_816591

theorem equiv_triangle_sum_of_squares :
  ∀ (A B C D1 D2 E1 E2 E3 E4 : Point)
    (s : ℝ),
    (equilateral_triangle A B C) →
    (side_length A B C = sqrt 150) →
    (congruent (triangle A D1 E1) (triangle A B C)) →
    (congruent (triangle A D1 E2) (triangle A B C)) →
    (congruent (triangle A D2 E3) (triangle A B C)) →
    (congruent (triangle A D2 E4) (triangle A B C)) →
    (dist B D1 = sqrt 15) →
    (dist B D2 = sqrt 15) →
    (∑ k in [E1, E2, E3, E4].map (λ E, (dist C E)^2) = 1200) :=
by
  intros A B C D1 D2 E1 E2 E3 E4 s h_equilateral h_side h_cong1 h_cong2 h_cong3 h_cong4 h_BD1 h_BD2
  sorry

end equiv_triangle_sum_of_squares_l816_816591


namespace find_solutions_of_x4_minus_16_l816_816621

noncomputable def solution_set : Set Complex :=
  {2, -2, Complex.I * 2, -Complex.I * 2}

theorem find_solutions_of_x4_minus_16 :
  {x : Complex | x^4 - 16 = 0} = solution_set :=
by
  sorry

end find_solutions_of_x4_minus_16_l816_816621


namespace sin_minus_cos_l816_816251

theorem sin_minus_cos (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < (Real.pi / 2)) (hθ3 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := by
  sorry

end sin_minus_cos_l816_816251


namespace sin_minus_cos_l816_816266

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end sin_minus_cos_l816_816266


namespace unique_polynomial_l816_816630

noncomputable def q (x : ℝ) : ℝ := 9 * x^3 - 13.5

theorem unique_polynomial (q : ℝ → ℝ)
  (h : ∀ x : ℝ, q(x^3) - q(x^3 - 3) = (q(x))^2 + 18) :
  q = λ x, 9 * x^3 - 13.5 :=
by
  sorry

end unique_polynomial_l816_816630


namespace Mike_practice_hours_per_weekday_l816_816788

theorem Mike_practice_hours_per_weekday :
  (exists (weekday_hours : ℕ), 
    let total_practice_hours := 60,
        weeks := 3,
        saturdays := weeks,
        weekend_hours := saturdays * 5,
        weekdays := (weeks * 6) - saturdays,
        remaining_weekday_hours := total_practice_hours - weekend_hours,
        weekday_hours := remaining_weekday_hours / weekdays in 
    weekday_hours = 3) :=
begin
  sorry
end

end Mike_practice_hours_per_weekday_l816_816788


namespace percentage_spent_on_medicines_l816_816377

theorem percentage_spent_on_medicines 
  (monthly_salary : ℝ) (spent_on_food_percentage : ℝ) (saved_percentage : ℝ) (actual_savings : ℝ) 
  (monthly_salary_eq : monthly_salary = 15000)
  (spent_on_food_eq : spent_on_food_percentage = 0.4)
  (saved_percentage_eq : saved_percentage = 0.6)
  (actual_savings_eq : actual_savings = 4320) :
  let spent_on_food := spent_on_food_percentage * monthly_salary,
      remaining_amount := monthly_salary - spent_on_food,
      calculated_savings := saved_percentage * remaining_amount,
      spent_on_medicines := calculated_savings - actual_savings,
      spent_on_medicines_percentage := (spent_on_medicines / monthly_salary) * 100
  in spent_on_medicines_percentage = 7.2 := 
by {
  sorry
}

end percentage_spent_on_medicines_l816_816377


namespace find_coeff_l816_816183

noncomputable def f (x a b : ℝ) := Real.exp x * (a * x + b) - x^2 - 4 * x

theorem find_coeff (a b : ℝ)
  (h₁ : f 0 a b = 4)
  (h₂ : HasDerivAt (λ x, f x a b) 4 0) :
  a = 4 ∧ b = 4 :=
sorry

end find_coeff_l816_816183


namespace sum_of_b_terms_l816_816674

noncomputable def sequence_a (n : ℕ) : ℕ :=
if n = 1 then 1 else 2 ^ (n - 1)

noncomputable def sequence_b (n : ℕ) : ℕ :=
3 + ∑ k in Finset.range (n - 1), (2 ^ k)

theorem sum_of_b_terms (n : ℕ) (h_pos: 0 < n) :
  sequence_b n = 2 ^ (n - 1) + 2 :=
by
  sorry

end sum_of_b_terms_l816_816674


namespace Tim_marble_count_l816_816636

theorem Tim_marble_count (Fred_marbles : ℕ) (Tim_marbles : ℕ) (h1 : Fred_marbles = 110) (h2 : Fred_marbles = 22 * Tim_marbles) : 
  Tim_marbles = 5 := 
sorry

end Tim_marble_count_l816_816636


namespace sin_minus_cos_eq_l816_816229

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end sin_minus_cos_eq_l816_816229


namespace D_cows_grazed_l816_816458

-- Defining the given conditions:
def A_cows := 24
def A_months := 3
def A_rent := 1440

def B_cows := 10
def B_months := 5

def C_cows := 35
def C_months := 4

def D_months := 3

def total_rent := 6500

-- Calculate the cost per cow per month (CPCM)
def CPCM := A_rent / (A_cows * A_months)

-- Proving the number of cows D grazed
theorem D_cows_grazed : ∃ x : ℕ, (x * D_months * CPCM + A_rent + (B_cows * B_months * CPCM) + (C_cows * C_months * CPCM) = total_rent) ∧ x = 21 := by
  sorry

end D_cows_grazed_l816_816458


namespace rhombus_side_length_l816_816704

-- Definitions
def is_rhombus_perimeter (P s : ℝ) : Prop := P = 4 * s

-- Theorem to prove
theorem rhombus_side_length (P : ℝ) (hP : P = 4) : ∃ s : ℝ, is_rhombus_perimeter P s ∧ s = 1 :=
by
  sorry

end rhombus_side_length_l816_816704


namespace cos_315_eq_sqrt2_div2_l816_816536

theorem cos_315_eq_sqrt2_div2 : Real.cos (315 * Real.pi / 180) = sqrt 2 / 2 := 
sorry

end cos_315_eq_sqrt2_div2_l816_816536


namespace binom_mod_1000_l816_816937

theorem binom_mod_1000 {S : ℤ} (hS : S = (1 + 1) ^ 2021 + (1 + exp(2 * π * I / 3)) ^ 2021 + (1 + exp(-2 * π * I / 3)) ^ 2021)
  (h_modulo : S % 1000 = 472 - 2) :
  (∑ k in Finset.range(674), (Nat.choose 2021 (3 * k))) % 1000 = 490 :=
by
  sorry

end binom_mod_1000_l816_816937


namespace total_figurines_l816_816067

theorem total_figurines:
  let basswood_blocks := 25
  let butternut_blocks := 30
  let aspen_blocks := 35
  let oak_blocks := 40
  let cherry_blocks := 45
  let basswood_figs_per_block := 3
  let butternut_figs_per_block := 4
  let aspen_figs_per_block := 2 * basswood_figs_per_block
  let oak_figs_per_block := 5
  let cherry_figs_per_block := 7
  let basswood_total := basswood_blocks * basswood_figs_per_block
  let butternut_total := butternut_blocks * butternut_figs_per_block
  let aspen_total := aspen_blocks * aspen_figs_per_block
  let oak_total := oak_blocks * oak_figs_per_block
  let cherry_total := cherry_blocks * cherry_figs_per_block
  let total_figs := basswood_total + butternut_total + aspen_total + oak_total + cherry_total
  total_figs = 920 := by sorry

end total_figurines_l816_816067


namespace monomials_like_terms_l816_816300

theorem monomials_like_terms (a b : ℕ) (h1 : 3 = a) (h2 : 4 = 2 * b) : a = 3 ∧ b = 2 :=
by
  sorry

end monomials_like_terms_l816_816300


namespace service_charge_per_plane_l816_816059

-- Define the conditions given in the problem
def num_planes : ℕ := 4
def num_smaller_planes : ℕ := 2
def num_larger_planes : ℕ := 2
def capacity_smaller_tank : ℕ := 60 -- in liters
def capacity_larger_tank : ℕ := capacity_smaller_tank + capacity_smaller_tank / 2
def fuel_cost_per_liter : ℕ := 50 -- in cents
def total_cost : ℕ := 55000 -- in cents ($550)

-- Define the total fuel needed
def fuel_needed_smaller_planes : ℕ := num_smaller_planes * capacity_smaller_ttank
def fuel_needed_larger_planes : ℕ := num_larger_planes * capacity_larger_ttank
def total_fuel_needed : ℕ := fuel_needed_smaller_planes + fuel_needed_larger_planes
def total_fuel_cost : ℕ := total_fuel_needed * fuel_cost_per_liter

-- Define the proof problem
theorem service_charge_per_plane :
  (total_cost - total_fuel_cost) / num_planes = 10000 := sorry 

end service_charge_per_plane_l816_816059


namespace midpoint_coordinates_l816_816112

theorem midpoint_coordinates :
  let A := (2 : ℤ, 9 : ℤ)
  let B := (8 : ℤ, -3 : ℤ)
  let midpoint (P Q : ℤ × ℤ) : ℤ × ℤ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  midpoint A B = (5, 3) :=
by
  sorry

end midpoint_coordinates_l816_816112


namespace actual_percent_profit_correct_l816_816919

def CP : ℝ := 100
def LP : ℝ := CP + 0.40 * CP
def SP : ℝ := LP - 0.05 * LP

def profit : ℝ := SP - CP
def percent_profit : ℝ := (profit / CP) * 100

theorem actual_percent_profit_correct : percent_profit = 33 := by
  sorry

end actual_percent_profit_correct_l816_816919


namespace divide_triangle_into_isosceles_l816_816810

theorem divide_triangle_into_isosceles (n : ℕ) (h : n ≥ 4) :
  ∃ (isosceles_triangles : list triangle), isosceles_triangles.length = n ∧ 
  (∀ t ∈ isosceles_triangles, is_isosceles t) :=
sorry

end divide_triangle_into_isosceles_l816_816810


namespace sin_minus_cos_theta_l816_816247

theorem sin_minus_cos_theta (θ : ℝ) (h₀ : 0 < θ ∧ θ < (π / 2)) 
  (h₁ : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -√10 / 5 :=
by
  sorry

end sin_minus_cos_theta_l816_816247


namespace binary_to_base4_l816_816579

theorem binary_to_base4 (n : ℕ) (h : n = 0b11010010) : n = 0b3102_base4 := by
  sorry

end binary_to_base4_l816_816579


namespace Fk_same_implies_eq_l816_816347

def Q (n: ℕ) : ℕ :=
  -- Implementation of the square part of n
  sorry

def N (n: ℕ) : ℕ :=
  -- Implementation of the non-square part of n
  sorry

def Fk (k: ℕ) (n: ℕ) : ℕ :=
  -- Implementation of Fk function calculating the smallest positive integer bigger than kn such that Fk(n) * n is a perfect square
  sorry

theorem Fk_same_implies_eq (k: ℕ) (n m: ℕ) (hk: 0 < k) : Fk k n = Fk k m → n = m :=
  sorry

end Fk_same_implies_eq_l816_816347


namespace trig_identity_l816_816286

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end trig_identity_l816_816286


namespace original_perimeter_l816_816058

variable (a b : ℕ)
variable (h : a + b = 129)

theorem original_perimeter (a b : ℕ) (h : a + b = 129) : 2 * (a + b) = 258 :=
by
  rw [h]
  exact rfl

end original_perimeter_l816_816058


namespace tina_hourly_wage_l816_816419

-- Define the conditions and the proof problem
theorem tina_hourly_wage (H : ℝ)
  (regular_hours_per_day : ℝ := 8)
  (total_days : ℝ := 5)
  (daily_work_hours : ℝ := 10)
  (total_earnings : ℝ := 990)
  (overtime_multiplier : ℝ := 1.5) :
  -- Definition of total regular hours worked in a week
  let regular_hours_week := total_days * regular_hours_per_day in
  -- Definition of over time hours worked per day
  let overtime_hours_per_day := daily_work_hours - regular_hours_per_day in
  -- Definition of total overtime hours worked in a week
  let overtime_hours_week := total_days * overtime_hours_per_day in
  -- Definition of the total earnings from regular and overtime hours
  let total_earnings_calculated := 
    (regular_hours_week * H) + (overtime_hours_week * (H * overtime_multiplier)) in
  -- Establish that the total earnings calculated equals the given total earnings of $990
  total_earnings_calculated = total_earnings → 
  -- Prove that Tina's hourly wage is $18
  H = 18 := 
sorry -- proof is not required as per the instructions


end tina_hourly_wage_l816_816419


namespace sin_minus_cos_l816_816238

noncomputable def theta_condition (θ : ℝ) : Prop := (0 < θ) ∧ (θ < π / 2) ∧ (Real.tan θ = 1 / 3)

theorem sin_minus_cos (θ : ℝ) (hθ : theta_condition θ) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_minus_cos_l816_816238


namespace fish_eaten_by_new_fish_l816_816746

def initial_original_fish := 14
def added_fish := 2
def exchange_new_fish := 3
def total_fish_now := 11

theorem fish_eaten_by_new_fish : initial_original_fish - (total_fish_now - exchange_new_fish) = 6 := by
  -- This is where the proof would go
  sorry

end fish_eaten_by_new_fish_l816_816746


namespace no_solutions_for_log_inequality_l816_816134

theorem no_solutions_for_log_inequality :
  ∀ x : ℕ, 50 < x ∧ x < 70 → ¬ (log 2 (x - 50) + log 2 (70 - x) < 3) :=
by sorry

end no_solutions_for_log_inequality_l816_816134


namespace exists_finite_set_with_parallel_lines_l816_816950

theorem exists_finite_set_with_parallel_lines :
  ∃ (M : Finset (ℝ × ℝ × ℝ)),
    (¬ ∀ (A B : (ℝ × ℝ × ℝ)), A ∈ M → B ∈ M → (A.2 = B.2 ∧ A.1 = B.1))
    ∧ (∀ (A B : (ℝ × ℝ × ℝ)), A ∈ M → B ∈ M →
         ∃ (C D : (ℝ × ℝ × ℝ)), C ∈ M ∧ D ∈ M ∧ 
         ((A.1 - B.1) * (C.2 - D.2) = (A.2 - B.2) * (C.1 - D.1) ∧
         ¬(A = C ∧ B = D ∧ A = D ∧ B = C))) :=
  sorry

end exists_finite_set_with_parallel_lines_l816_816950


namespace philippe_wins_l816_816374

/-- Philippe and Emmanuel have 2018 cards numbered 1 to 2018. They take turns, each picking one card 
from the set, with Philippe starting. The player with an even sum of card numbers wins. Prove that 
Philippe can always ensure his sum is even, hence he wins. -/
theorem philippe_wins :
  ∀ cards : finset ℕ, cards = finset.range 2019 \ {0} →
  ∀ hands : ℕ → list ℕ,  
  (∃ p_hand e_hand : list ℕ, ∀ p_sum e_sum : ℕ, 
    list.sum p_hand = p_sum ∧ list.sum e_hand = e_sum ∧ 
    length p_hand = 1009 ∧ length e_hand = 1009 ∧ 
    (p_sum % 2 = 0) ∧ (p_hand.length = e_hand.length + 1) → p_sum = 1018581
  ) 
sorry

end philippe_wins_l816_816374


namespace divisors_of_square_l816_816054

theorem divisors_of_square (n : ℕ) (h : ∃ p q : ℕ, (n = p^3 ∨ (n = p * q ∧ p ≠ q) ∧ nat.prime p ∧ nat.prime q)) :
  ∃ m : ℕ, m = 7 ∨ m = 9 :=
by
  sorry

end divisors_of_square_l816_816054


namespace sum_of_money_l816_816838

-- Define the simple interest function
def simple_interest (P R T : ℝ) : ℝ :=
  P * R * T / 100

-- Define the compound interest function
def compound_interest (P R T : ℝ) : ℝ :=
  P * (1 + R / 100) ^ T - P

-- Given conditions
def P : ℝ := 1800
def R : ℝ := 10
def T : ℝ := 2
def difference : ℝ := 18

-- The main theorem statement
theorem sum_of_money :
  compound_interest P R T - simple_interest P R T = difference → P = 1800 :=
by
  sorry

end sum_of_money_l816_816838


namespace ludwig_weekly_earnings_l816_816784

theorem ludwig_weekly_earnings :
  (7 = 7) ∧
  (∀ day : ℕ, day ∈ {5, 6, 7} → (1 / 2) = 1 / 2) ∧
  (daily_salary = 10) →
  (weekly_earnings = 55) :=
by
  sorry

end ludwig_weekly_earnings_l816_816784


namespace find_other_two_sides_l816_816652

structure IsoscelesTriangle (a b c : ℝ) :=
(perimeter : ℝ)
(side : ℝ)
(is_isosceles : (a = b ∨ a = c ∨ b = c))
(one_side : a = 3 ∨ b = 3 ∨ c = 3)
(perimeter_eq : a + b + c = perimeter)
(peri_value : perimeter = 12)

theorem find_other_two_sides :
  ∃ (x y : ℝ),
  (IsoscelesTriangle x y 3).perimeter_eq
  ∧ (IsoscelesTriangle x y 3).peri_value
  ∧ (IsoscelesTriangle x y 3).is_isosceles
  ∧ x = y
  ∧ x = 4.5
  ∧ y = 4.5 := by
  sorry

end find_other_two_sides_l816_816652


namespace road_repair_inches_l816_816044

-- Variables for lengths of sections
def length_A := 3000
def length_B := 2500
def length_C := 1500

-- Variables for percentage completion of sections
def percent_A := 0.65
def percent_B := 0.40
def percent_C := 0.50

-- Number of inches repaved before today
def repaved_before_today :=
  (percent_A * length_A) +
  (percent_B * length_B) +
  (percent_C * length_C)

-- Theorem stating the problem
theorem road_repair_inches : repaved_before_today = 3700 := by
  sorry

end road_repair_inches_l816_816044


namespace price_per_potato_bundle_l816_816045

theorem price_per_potato_bundle :
  (let total_potatoes := 250
       potatoes_per_bundle := 25
       total_carrots := 320
       carrots_per_bundle := 20
       price_per_carrot_bundle := 2
       total_revenue := 51 in
   let number_of_potato_bundles := total_potatoes / potatoes_per_bundle
       number_of_carrot_bundles := total_carrots / carrots_per_bundle
       revenue_from_carrots := number_of_carrot_bundles * price_per_carrot_bundle
       revenue_from_potatoes := total_revenue - revenue_from_carrots
       price_per_potato_bundle := revenue_from_potatoes / number_of_potato_bundles in
   price_per_potato_bundle = 1.9) :=
by
  sorry

end price_per_potato_bundle_l816_816045


namespace seventh_term_arith_seq_l816_816401

/-- 
The seventh term of an arithmetic sequence given that the sum of the first five terms 
is 15 and the sixth term is 7.
-/
theorem seventh_term_arith_seq (a d : ℚ) 
  (h1 : 5 * a + 10 * d = 15) 
  (h2 : a + 5 * d = 7) : 
  a + 6 * d = 25 / 3 := 
sorry

end seventh_term_arith_seq_l816_816401


namespace cos_315_eq_sqrt2_div2_l816_816538

theorem cos_315_eq_sqrt2_div2 : Real.cos (315 * Real.pi / 180) = sqrt 2 / 2 := 
sorry

end cos_315_eq_sqrt2_div2_l816_816538


namespace slope_angle_tangent_line_sqrt_x_at_1div4_is_pi_div_4_l816_816399

noncomputable def tangent_slope_angle_at_point : ℝ :=
  let y := λ x : ℝ, real.sqrt x
  let derivative := (λ x, (1/2) * (x^(1/2))⁻¹)
  let tangent_slope := derivative (1/4)
  let angle := real.arctan tangent_slope
  angle

theorem slope_angle_tangent_line_sqrt_x_at_1div4_is_pi_div_4 :
  tangent_slope_angle_at_point = real.pi / 4 :=
sorry

end slope_angle_tangent_line_sqrt_x_at_1div4_is_pi_div_4_l816_816399


namespace area_of_polar_curve_l816_816329

-- Given condition describing the curve in polar coordinates
def polar_curve (theta : ℝ) : ℝ := 4 * Real.cos theta

-- Statement of the problem: Proving the area enclosed by the curve is 4π
theorem area_of_polar_curve :
  (∫ θ in 0..2 * Real.pi, (polar_curve θ)^2 / 2) = 4 * Real.pi :=
by
  sorry

end area_of_polar_curve_l816_816329


namespace magnitude_of_z_l816_816834

noncomputable def wz : ℂ := 24 - 16 * complex.I
noncomputable def w_mag : ℝ := real.sqrt 52 

theorem magnitude_of_z (w z : ℂ) (hwz : w * z = wz) (hw : complex.abs w = w_mag) : 
  complex.abs z = 4 :=
by
  sorry

end magnitude_of_z_l816_816834


namespace no_solutions_xyz_l816_816124

theorem no_solutions_xyz : ∀ (x y z : ℝ), x + y = 3 → xy - z^2 = 2 → false := by
  intros x y z h1 h2
  sorry

end no_solutions_xyz_l816_816124


namespace only_perfect_square_up_to_2013_l816_816752

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def sequence_a : ℕ → ℕ
| 0       := 1
| (n + 1) := ⌊sequence_a n + (sequence_a n).sqrt + (1 / 2 : ℚ)⌋₊

theorem only_perfect_square_up_to_2013 :
  ∀ n : ℕ, n ≤ 2013 → is_perfect_square (sequence_a n) → n = 1 := 
sorry

end only_perfect_square_up_to_2013_l816_816752


namespace bridge_length_equals_42_l816_816466

open Nat

-- Define the conditions
def total_path_length : ℕ := 900
def distance_between_poles : ℕ := 6
def total_fence_poles : ℕ := 286

-- Define the property we want to prove
theorem bridge_length_equals_42 :
  let poles_one_side := total_fence_poles / 2 in
  let fenced_length_one_side := poles_one_side * distance_between_poles in
  let bridge_length := total_path_length - fenced_length_one_side in
  bridge_length = 42 :=
by
  let poles_one_side := total_fence_poles / 2
  let fenced_length_one_side := poles_one_side * distance_between_poles
  let bridge_length := total_path_length - fenced_length_one_side
  show bridge_length = 42
  sorry

end bridge_length_equals_42_l816_816466


namespace trig_identity_l816_816285

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end trig_identity_l816_816285


namespace exists_positive_M_l816_816298

open Set

noncomputable def f (x : ℝ) : ℝ := sorry

theorem exists_positive_M 
  (h₁ : ∀ x ∈ Ioo (0 : ℝ) 1, f x > 0)
  (h₂ : ∀ x ∈ Ioo (0 : ℝ) 1, f (2 * x / (1 + x^2)) = 2 * f x) :
  ∃ M > 0, ∀ x ∈ Ioo (0 : ℝ) 1, f x ≤ M :=
sorry

end exists_positive_M_l816_816298


namespace assembling_ratio_l816_816363

def t_onions := 20
def t_garlic_peppers := (1 / 4 : ℝ) * t_onions
def t_kneading := 30
def t_resting := 2 * t_kneading
def t_other_tasks := t_onions + t_garlic_peppers + t_kneading + t_resting
def t_total := 124
def t_assembling := t_total - t_other_tasks
def t_combined_kneading_resting := t_kneading + t_resting

theorem assembling_ratio : t_assembling / t_combined_kneading_resting = (1 / 10 : ℝ) :=
by
  sorry

end assembling_ratio_l816_816363


namespace total_money_spent_l816_816036

theorem total_money_spent 
  (num_friends : ℕ)
  (tshirt_orig_price hat_orig_price : ℚ)
  (tshirt_discount hat_discount : ℚ) :
  num_friends = 4 →
  tshirt_orig_price = 20 →
  tshirt_discount = 0.40 →
  hat_orig_price = 15 →
  hat_discount = 0.60 →
  (num_friends * ((tshirt_orig_price * (1 - tshirt_discount)) + (hat_orig_price * (1 - hat_discount)))) = 72 := 
by
  intros h1 h2 h3 h4 h5
  calc
  4 * (20 * (1 - 0.40) + 15 * (1 - 0.60)) 
  = 4 * (20 * 0.60 + 15 * 0.40) : by rw [h1, h2, h3, h4, h5]
  ... = 4 * (12 + 6) : by rw [mul_comm 20, mul_comm 15, mul_comm 12, mul_comm 6]
  ... = 4 * 18 : by norm_num
  ... = 72 : by norm_num

end total_money_spent_l816_816036


namespace find_f_prime_2014_l816_816987

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 2) * x^2 + 2 * x * f' 2014 + 2014 * log x

theorem find_f_prime_2014 :
  (∀ x : ℝ, 0 < x →
  f x = (1 / 2) * x^2 + 2 * x * f' 2014 + 2014 * log x) →
  f' 2014 = -2015 :=
by
  sorry

end find_f_prime_2014_l816_816987


namespace probability_two_even_dice_l816_816929

open ProbabilityTheory

theorem probability_two_even_dice : 
  let p_even := 1 / 2,
      p_odd := 1 / 2,
      number_of_ways := Nat.choose 4 2,
      probability_per_way := p_even ^ 2 * p_odd ^ 2,
      total_probability := (number_of_ways : ℚ) * probability_per_way in
  total_probability = 3 / 8 :=
by
  let p_even := 1 / 2
  let p_odd := 1 / 2
  let number_of_ways := Nat.choose 4 2
  let probability_per_way := p_even ^ 2 * p_odd ^ 2
  let total_probability := (number_of_ways : ℚ) * probability_per_way
  have probability_calculation : total_probability = 3 / 8 := by
    sorry
  exact probability_calculation

end probability_two_even_dice_l816_816929


namespace tan_triple_angle_formula_l816_816205

variable (θ : ℝ)
variable (h : Real.tan θ = 4)

theorem tan_triple_angle_formula : Real.tan (3 * θ) = 52 / 47 :=
by
  sorry  -- Proof is omitted

end tan_triple_angle_formula_l816_816205


namespace patsy_needs_more_appetizers_l816_816806

def appetizers_per_guest := 6
def number_of_guests := 30
def deviled_eggs := 3 -- dozens
def pigs_in_a_blanket := 2 -- dozens
def kebabs := 2 -- dozens

theorem patsy_needs_more_appetizers :
  let total_required := appetizers_per_guest * number_of_guests,
      total_made := (deviled_eggs + pigs_in_a_blanket + kebabs) * 12,
      total_needed := total_required - total_made
  in total_needed / 12 = 8 := sorry

end patsy_needs_more_appetizers_l816_816806


namespace determine_valid_n_l816_816768

def f (x : Nat) : Nat := x^2 + x + 1

def is_valid_n (n : Nat) : Prop :=
  ∀ k : Nat, k > 0 → k ∣ n → f(k) ∣ f(n)

theorem determine_valid_n (n : Nat) :
  is_valid_n n ↔ 
  (n = 1 ∨ 
  (∃ p : Nat, Prime p ∧ p % 3 = 1 ∧ n = p) ∨ 
  (∃ p : Nat, Prime p ∧ p ≠ 3 ∧ n = p^2)) :=
by
  sorry

end determine_valid_n_l816_816768


namespace pages_in_book_l816_816789

theorem pages_in_book 
  (x : ℕ)
  (h1 : ∃ (x : ℕ), ∀ (pages_left : ℕ), 
    pages_left = 85 → 
    pages_left = 
    (let first_day_left := x - (1/6 * x + 10) in
     let second_day_left := first_day_left - (1/5 * first_day_left + 20) in
     let third_day_left := second_day_left - (1/4 * second_day_left + 25) in
     third_day_left)) : 
  x = 262 :=
sorry

end pages_in_book_l816_816789


namespace only_one_correct_guess_l816_816720

-- Define the contestants
inductive Contestant : Type
| person : ℕ → Contestant

def A_win_first (c: Contestant) : Prop :=
c = Contestant.person 4 ∨ c = Contestant.person 5

def B_not_win_first (c: Contestant) : Prop :=
c ≠ Contestant.person 3 

def C_win_first (c: Contestant) : Prop :=
c = Contestant.person 1 ∨ c = Contestant.person 2 ∨ c = Contestant.person 6

def D_not_win_first (c: Contestant) : Prop :=
c ≠ Contestant.person 4 ∧ c ≠ Contestant.person 5 ∧ c ≠ Contestant.person 6

-- The main theorem: Only one correct guess among A, B, C, and D
theorem only_one_correct_guess (win: Contestant) :
  (A_win_first win ↔ false) ∧ (B_not_win_first win ↔ false) ∧ (C_win_first win ↔ false) ∧ D_not_win_first win
:=
by
  sorry

end only_one_correct_guess_l816_816720


namespace A_oplus_B_eq_l816_816971

def set_diff (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
def symm_diff (M N : Set ℝ) : Set ℝ := set_diff M N ∪ set_diff N M
def A : Set ℝ := {y | ∃ x:ℝ, y = 3^x}
def B : Set ℝ := {y | ∃ x:ℝ, y = -(x-1)^2 + 2}

theorem A_oplus_B_eq : symm_diff A B = {y | y ≤ 0} ∪ {y | y > 2} := by {
  sorry
}

end A_oplus_B_eq_l816_816971


namespace sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816212

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816212


namespace cos_315_is_sqrt2_div_2_l816_816517

noncomputable def cos_315_eq : Prop := 
  real.cos (315 * real.pi / 180) = real.sqrt 2 / 2

theorem cos_315_is_sqrt2_div_2 : cos_315_eq := by 
  sorry

end cos_315_is_sqrt2_div_2_l816_816517


namespace zuminglish_words_mod_1000_l816_816308

/-- Number of 8-letter valid Zuminglish words modulo 1000 -/
def num_valid_zuminglish_words_mod_1000 : ℕ :=
  let a : ℕ → ℕ
    | 2 => 4
    | n + 1 => 2 * (a n + c n)
  and b : ℕ → ℕ
    | 2 => 2
    | n + 1 => a n
  and c : ℕ → ℕ
    | 2 => 2
    | n + 1 => 2 * b n
  in (a 8 + b 8 + c 8) % 1000

theorem zuminglish_words_mod_1000 : 
  num_valid_zuminglish_words_mod_1000 = 368 :=
by
  sorry

end zuminglish_words_mod_1000_l816_816308


namespace find_f3_l816_816589

theorem find_f3 
  (b c : ℝ)
  (h1 : ∀ x, f(x) = x^2 + b * x + c)
  (h2 : (4 + 2 * b + c) + (16 + 4 * b + c) = 12138) :
  f(3) = 6068 :=
by
  sorry

end find_f3_l816_816589


namespace angle_subtended_less_than_pi_over_3_l816_816824

-- Defines a regular tetrahedron with vertices A, B, C, and D
structure RegularTetrahedron (A B C D : Point) : Prop :=
(euclidean_geometry.is_regular : is_regular_tetra A B C D)

-- Defines points P and Q inside the regular tetrahedron
structure PointsInTetrahedron {A B C D : Point} (tetra : RegularTetrahedron A B C D) (P Q : Point) : Prop :=
(P_in_tetra : P ∈ interior (tetrahedron A B C D))
(Q_in_tetra : Q ∈ interior (tetrahedron A B C D))

-- The main theorem
theorem angle_subtended_less_than_pi_over_3 {A B C D P Q : Point} 
  (regular_tetra : RegularTetrahedron A B C D) 
  (points_in_tetra : PointsInTetrahedron regular_tetra P Q) :
  euclidean.angle P A Q < π / 3 := 
sorry

end angle_subtended_less_than_pi_over_3_l816_816824


namespace dumbbell_distribution_impossible_l816_816923

theorem dumbbell_distribution_impossible :
  let W := [4, 5, 6, 9, 10, 11, 14, 19, 23, 24] in
  let total_weight := 125 in
  ∀ W1 W2 W3 : ℕ, 
    W1 + W2 + W3 = total_weight → 
    W1 = W2 / 2 → 
    W2 = W3 / 2 → 
    False :=
by
  sorry

end dumbbell_distribution_impossible_l816_816923


namespace trigonometric_identity_l816_816272

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end trigonometric_identity_l816_816272


namespace total_students_are_45_l816_816496

theorem total_students_are_45 (burgers hot_dogs students : ℕ)
  (h1 : burgers = 30)
  (h2 : burgers = 2 * hot_dogs)
  (h3 : students = burgers + hot_dogs) : students = 45 :=
sorry

end total_students_are_45_l816_816496


namespace siding_cost_l816_816378

-- Define the dimensions of the wall and roof faces
def wall_length : ℝ := 10
def wall_height : ℝ := 8
def roof_length : ℝ := 10
def roof_height : ℝ := 6
def num_roofs : ℝ := 2

-- Calculate the areas of the wall and the roofs
def wall_area : ℝ := wall_length * wall_height
def roof_area : ℝ := roof_length * roof_height
def total_roof_area : ℝ := num_roofs * roof_area
def total_area : ℝ := wall_area + total_roof_area

-- Dimensions and cost of one siding section
def siding_length : ℝ := 10
def siding_height : ℝ := 15
def cost_per_section : ℝ := 30.50
def siding_area : ℝ := siding_length * siding_height

-- Calculate the number of sections required
def num_sections : ℝ := real.ceil (total_area / siding_area)

-- Calculate the total cost
def total_cost : ℝ := num_sections * cost_per_section

-- Prove that the total cost is 61.00
theorem siding_cost : total_cost = 61.00 := by
  -- Proof omitted (replace 'sorry' with actual proof steps)
  sorry

end siding_cost_l816_816378


namespace find_second_number_l816_816885

-- Defining the ratios and sum condition
def ratio (a b c : ℕ) := 5*a = 3*b ∧ 3*b = 4*c

theorem find_second_number (a b c : ℕ) (h_ratio : ratio a b c) (h_sum : a + b + c = 108) : b = 27 :=
by
  sorry

end find_second_number_l816_816885


namespace volume_of_rect_prism_l816_816440

/-- 
  Given a rectangular prism box with length x + 3 units, 
  width x - 3 units, and height x^2 + 9 units, 
  prove that the number of positive integer values of x such that the volume of the box is 
  less than 500 units is 1.
-/
theorem volume_of_rect_prism (x : ℕ) (h_pos : 0 < x) :
  (x+3) * (x-3) * (x^2 + 9) < 500 → { n : ℕ | (x+3) * (x-3) * (x^2 + 9) < 500 }.to_finset.card = 1 := 
sorry

end volume_of_rect_prism_l816_816440


namespace roots_sum_of_quadratic_solve_quadratic_l816_816178

theorem roots_sum_of_quadratic (a b c : ℝ) (h: a ≠ 0) (h_eq : a * x^2 + b * x + c = 0) :
    b^2 - 4 * a * c ≥ 0 → 
    let x1 := (-b + real.sqrt(b^2 - 4 * a * c)) / (2 * a),
        x2 := (-b - real.sqrt(b^2 - 4 * a * c)) / (2 * a) in
    x1 + x2 = -b / a :=
sorry

theorem solve_quadratic:
    let a := 1,
        b := 1,
        c := -2 in
    roots_sum_of_quadratic a b c (by norm_num) (by norm_num : a * x^2 + b * x + c = 0)
        (by norm_num : b^2 - 4 * a * c ≥ 0) = -1 :=
sorry

end roots_sum_of_quadratic_solve_quadratic_l816_816178


namespace range_of_m_l816_816303

theorem range_of_m {x y : ℝ} (hx : 0 < x) (hy : 0 < y)
  (h_cond : 1/x + 4/y = 1) : 
  (∃ x y, 0 < x ∧ 0 < y ∧ 1/x + 4/y = 1 ∧ x + y/4 < m^2 + 3 * m) ↔
  (m < -4 ∨ 1 < m) := 
sorry

end range_of_m_l816_816303


namespace weight_of_new_person_l816_816031

-- Define the given conditions
variables (avg_increase : ℝ) (num_people : ℕ) (replaced_weight : ℝ)
variable (new_weight : ℝ)

-- These are the conditions directly from the problem
axiom avg_weight_increase : avg_increase = 4.5
axiom number_of_people : num_people = 6
axiom person_to_replace_weight : replaced_weight = 75

-- Mathematical equivalent of the proof problem
theorem weight_of_new_person :
  new_weight = replaced_weight + avg_increase * num_people := 
sorry

end weight_of_new_person_l816_816031


namespace number_of_ways_to_choose_vessels_l816_816990

theorem number_of_ways_to_choose_vessels :
  let capacities := (Finset.range 25).map (λ n, n + 1) in
  ∃ s ⊆ capacities, s.card = 10 ∧ ∀ a b ∈ s, a ≠ b → gcd a b = 1 → s.card = 16 :=
by 
  sorry

end number_of_ways_to_choose_vessels_l816_816990


namespace circle_equation_l816_816304

theorem circle_equation (x y : ℝ) (h : ∀ x y : ℝ, x^2 + y^2 ≥ 64) :
  x^2 + y^2 - 64 = 0 ↔ x = 0 ∧ y = 0 :=
by
  sorry

end circle_equation_l816_816304


namespace total_blue_marbles_l816_816786

noncomputable theory

def blue_marbles (jenny_blue mary_blue anie_blue tom_blue : ℕ) : ℕ :=
  jenny_blue + mary_blue + anie_blue + tom_blue

theorem total_blue_marbles (jenny_red jenny_blue mary_red mary_blue anie_red anie_blue tom_red tom_blue : ℕ) 
(hj : jenny_red = 30)
(hj2 : jenny_blue = 25)
(hm1 : mary_red = 2 * jenny_red)
(hm2 : mary_blue = anie_blue / 2)
(ha1 : anie_red = mary_red + 20)
(ha2 : anie_blue = 2 * jenny_blue)
(ht1 : tom_red = anie_red + 10)
(ht2 : tom_blue = mary_blue) :
  blue_marbles jenny_blue mary_blue anie_blue tom_blue = 125 := 
sorry

end total_blue_marbles_l816_816786


namespace area_of_feasible_region_l816_816910

theorem area_of_feasible_region :
  (∃ k m : ℝ, (∀ x y : ℝ,
    (kx - y + 1 ≥ 0 ∧ kx - my ≤ 0 ∧ y ≥ 0) ↔
    (x - y + 1 ≥ 0 ∧ x + y ≤ 0 ∧ y ≥ 0)) ∧
    k = 1 ∧ m = -1) →
  ∃ a : ℝ, a = 1 / 4 :=
by sorry

end area_of_feasible_region_l816_816910


namespace greatest_k_dividing_abcdef_l816_816765

theorem greatest_k_dividing_abcdef {a b c d e f : ℤ}
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = f^2) :
  ∃ k, (∀ a b c d e f, a^2 + b^2 + c^2 + d^2 + e^2 = f^2 → k ∣ (a * b * c * d * e * f)) ∧ k = 24 :=
sorry

end greatest_k_dividing_abcdef_l816_816765


namespace right_triangle_checker_l816_816503

-- Define sets of numbers as tuples
def setA := (0.3, 0.4, 0.5)
def setB := (12, 16, 20)
def setC := (1, Real.sqrt 2, Real.sqrt 3)
def setD := (11, 40, 41)

-- Pythagorean condition function
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Main theorem statement
theorem right_triangle_checker :
  ¬(is_right_triangle setD.1 setD.2 setD.2) ∧
  is_right_triangle setA.1 setA.2 setA.2 ∧
  is_right_triangle setB.1 setB.2 setB.2 ∧
  is_right_triangle setC.1 setC.2 setC.2 := 
sorry

end right_triangle_checker_l816_816503


namespace even_function_increasing_relationship_l816_816691

variables {ℝ : Type*} [ordered_ring ℝ] (f : ℝ → ℝ)

-- Condition: f is an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Condition: f is increasing on (-∞, -1]
def increasing_on_negative_domain (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → x₂ ≤ -1 → f x₁ < f x₂

-- Goal: Prove the relationship f(-3) < f(2) < f(-1)
theorem even_function_increasing_relationship
  (hf_even : even_function f)
  (hf_increasing : increasing_on_negative_domain f) :
  f (-3) < f 2 ∧ f 2 < f (-1) :=
begin
  sorry
end

end even_function_increasing_relationship_l816_816691


namespace slices_remaining_l816_816797

-- Given conditions
def initial_slices : ℕ := 24
def breakfast_percentage : ℚ := 0.4
def lunch_fraction : ℚ := 1 / 6

-- Statement of the problem
theorem slices_remaining : true := by
  let eaten_for_breakfast := (0.4 * initial_slices).floor -- 40% of 24 slices rounded down
  let remaining_after_breakfast := initial_slices - eaten_for_breakfast
  let used_for_lunch := (1 / 6 * remaining_after_breakfast).floor -- 1/6th of remaining slices rounded down
  let final_slices_remaining := remaining_after_breakfast - used_for_lunch

  -- Final number of slices should be 13
  have : final_slices_remaining = 13, from sorry
  trivial

end slices_remaining_l816_816797


namespace jed_cards_after_4_weeks_l816_816743

theorem jed_cards_after_4_weeks :
  ∀ (get_cards_per_week give_cards_every_two_weeks start_cards target_cards : ℕ),
    get_cards_per_week = 6 →
    give_cards_every_two_weeks = 2 →
    start_cards = 20 →
    target_cards = 40 →
    ∃ (n : ℕ), n = 4 ∧ start_cards + n * get_cards_per_week - (n / 2) * give_cards_every_two_weeks = target_cards :=
by
  intros get_cards_per_week give_cards_every_two_weeks start_cards target_cards
  assume h_get_cards h_give_cards h_start_cards h_target_cards
  use 4
  split
  · rfl
  · sorry

end jed_cards_after_4_weeks_l816_816743


namespace cone_height_l816_816899

theorem cone_height (V : ℝ) (h : ℝ) (r : ℝ) (vertex_angle : ℝ) 
  (H1 : V = 16384 * Real.pi)
  (H2 : vertex_angle = 90) 
  (H3 : V = (1 / 3) * Real.pi * r^2 * h)
  (H4 : h = r) : 
  h = 36.6 :=
by
  sorry

end cone_height_l816_816899


namespace prop_A_is_correct_prop_B_is_correct_prop_C_is_incorrect_prop_D_is_incorrect_l816_816634

noncomputable def has_fixed_point (f : ℝ → ℝ) : Set ℝ :=
{x : ℝ | f x = x}

noncomputable def prop_A : Prop :=
  ∃! x : ℝ, x > 0 ∧ f x = 1 + Real.log x

noncomputable def prop_B : Prop :=
  ∃ a b : ℝ, a ≠ b ∧ f a = abs(2 - 1/a) ∧ f b = abs(2 - 1/b)

noncomputable def prop_C : Prop :=
  ∃ x : ℝ, ∀ n : ℤ, f x ≠ 1/3 * Real.tan (x + n * Real.pi)

noncomputable def prop_D : Prop :=
  ¬ ∃ x : ℝ, f x = Real.exp x - 1/2 * x^2 - 1

theorem prop_A_is_correct : prop_A := sorry

theorem prop_B_is_correct : prop_B := sorry

theorem prop_C_is_incorrect : ¬prop_C := sorry

theorem prop_D_is_incorrect : ¬prop_D := sorry

end prop_A_is_correct_prop_B_is_correct_prop_C_is_incorrect_prop_D_is_incorrect_l816_816634


namespace greatest_power_of_3_factor_l816_816426

theorem greatest_power_of_3_factor (n : ℕ) (h : n = 1503) :
  ∃ k : ℕ, 6^n - 3^n = 3^k * (2^n - 1) ∧ k = n :=
by {
  use n,
  rw [h, show 6 = 2 * 3 by norm_num, ←pow_mul, pow_mul, pow_mul],
  simp,
  sorry
}

end greatest_power_of_3_factor_l816_816426


namespace smallest_n_satisfying_conditions_l816_816427

theorem smallest_n_satisfying_conditions : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ x : ℕ, 3 * n = x^4) ∧ (∃ y : ℕ, 2 * n = y^5) ∧ n = 432 :=
by
  sorry

end smallest_n_satisfying_conditions_l816_816427


namespace expected_value_of_winnings_l816_816791

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_composite (n : ℕ) : Prop :=
  n = 4 ∨ n = 6

def loses_money (n : ℕ) : Prop :=
  n = 1 ∨ n = 8

def winnings (n : ℕ) : ℤ :=
  if is_prime n then n else if loses_money n then -4 else 0

def probability (n : ℤ) : ℚ := 1 / 8

def expected_value : ℚ :=
  (∑ n in finset.range 9, (winnings n) * probability n)

theorem expected_value_of_winnings :
  expected_value = 9 / 8 :=
  sorry

end expected_value_of_winnings_l816_816791


namespace total_number_of_balls_is_twelve_l816_816319

noncomputable def num_total_balls (a : ℕ) : Prop :=
(3 : ℚ) / a = (25 : ℚ) / 100

theorem total_number_of_balls_is_twelve : num_total_balls 12 :=
by sorry

end total_number_of_balls_is_twelve_l816_816319


namespace inequality_range_l816_816702

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → m ∈ Icc (-3 : ℝ) 5 :=
by
  sorry

end inequality_range_l816_816702


namespace max_value_of_x2_y3_z_l816_816761

theorem max_value_of_x2_y3_z (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum : x + y + z = 1) : x^2 * y^3 * z ≤ 1 / 3888 :=
begin
  -- Proof omitted
  sorry
end

end max_value_of_x2_y3_z_l816_816761


namespace find_angle_B_find_sin_2C_l816_816333

-- Define variables and their domains/conditions
variables {A B C : ℝ}

-- Define the given conditions
def condition1 : Prop := cos C + (cos A - (sqrt 3) * sin A) * cos B = 0
def condition2 : Prop := sin (A - (π / 3)) = 3 / 5

-- Define the first proof problem
theorem find_angle_B (h1 : condition1) : B = π / 3 :=
sorry

-- Define the second proof problem
theorem find_sin_2C (h1 : condition1) (h2 : condition2) : sin (2 * C) = (24 + 7 * sqrt 3) / 50 :=
sorry

end find_angle_B_find_sin_2C_l816_816333


namespace smallest_n_satisfying_conditions_l816_816745

variable (n : ℕ)
variable (h1 : 100 ≤ n ∧ n < 1000)
variable (h2 : (n + 7) % 6 = 0)
variable (h3 : (n - 5) % 9 = 0)

theorem smallest_n_satisfying_conditions : n = 113 := by
  sorry

end smallest_n_satisfying_conditions_l816_816745


namespace lent_amount_C_is_correct_l816_816050

def principal_A := 4000
def time_A := 2
def time_C := 4
def rate := 0.1375
def total_interest := 2200

theorem lent_amount_C_is_correct : 
  ∃ (X : ℝ), (principal_A * rate * time_A + X * rate * time_C = total_interest) ∧ (X = 2000) := by
  sorry

end lent_amount_C_is_correct_l816_816050


namespace students_like_basketball_or_cricket_or_both_l816_816315

open Set

variable (students: Type)

variable (likes_basketball likes_cricket: students → Prop)

theorem students_like_basketball_or_cricket_or_both
  (hb : (univ.filter likes_basketball).to_finset.card = 7)
  (hc : (univ.filter likes_cricket).to_finset.card = 8)
  (hbc: ((univ.filter (λ s, likes_basketball s ∧ likes_cricket s)).to_finset.card = 5)):
  ((univ.filter (λ s, likes_basketball s ∨ likes_cricket s)).to_finset.card) = 10 := 
by
  sorry

end students_like_basketball_or_cricket_or_both_l816_816315


namespace pentagon_angles_l816_816727

theorem pentagon_angles (M T H A S : ℝ) 
  (h1 : M = T) 
  (h2 : T = H) 
  (h3 : A + S = 180) 
  (h4 : M + A + T + H + S = 540) : 
  H = 120 := 
by 
  -- The proof would be inserted here.
  sorry

end pentagon_angles_l816_816727


namespace find_linear_combination_l816_816287

variable (a b c : ℝ)

theorem find_linear_combination (h1 : a + 2 * b - 3 * c = 4)
                               (h2 : 5 * a - 6 * b + 7 * c = 8) :
  9 * a + 2 * b - 5 * c = 24 :=
sorry

end find_linear_combination_l816_816287


namespace hyperbola_eccentricity_correct_l816_816673

noncomputable def hyperbola_eccentricity (a b m : ℝ) (h_a : a > 0) (h_b : b > 0) (h_m : m ≠ 0) : ℝ :=
  let A := (m * a / (3 * b - a), m * b / (3 * b - a))
  let B := (-m * a / (3 * b + a), m * b / (3 * b + a))
  let P := (m, 0)
  let midpoint := (m * a ^ 2 / (9 * b ^ 2 - a ^ 2), 3 * m * b ^ 2 / (9 * b ^ 2 - a ^ 2))
  let e := (real.sqrt (a ^ 2 + b ^ 2)) / a
  if |((midpoint.2 - P.2) / (midpoint.1 - P.1))| = 3 ∧ a = 2 * b then e else 0

theorem hyperbola_eccentricity_correct (a b m : ℝ) (h_a : a > 0) (h_b : b > 0) (h_m : m ≠ 0) :
  hyperbola_eccentricity a b m h_a h_b h_m = real.sqrt 5 / 2 := sorry

end hyperbola_eccentricity_correct_l816_816673


namespace cos_315_deg_l816_816544

noncomputable def cos_315 : ℝ :=
  Real.cos (315 * Real.pi / 180)

theorem cos_315_deg : cos_315 = Real.sqrt 2 / 2 :=
by
  sorry

end cos_315_deg_l816_816544


namespace sin_minus_cos_l816_816259

theorem sin_minus_cos (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < (Real.pi / 2)) (hθ3 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := by
  sorry

end sin_minus_cos_l816_816259


namespace cos_315_eq_sqrt2_div2_l816_816532

theorem cos_315_eq_sqrt2_div2 : Real.cos (315 * Real.pi / 180) = sqrt 2 / 2 := 
sorry

end cos_315_eq_sqrt2_div2_l816_816532


namespace defective_items_draw_l816_816708

theorem defective_items_draw (total_products : ℕ) (defective_products : ℕ) :
  total_products = 100 ∧ defective_products = 2 →
  (Nat.choose 100 3 - Nat.choose 98 3 = 9472) :=
by
  intro h
  cases h with h_total h_def
  rw [h_total, h_def]
  sorry

end defective_items_draw_l816_816708


namespace sin_minus_cos_eq_l816_816225

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end sin_minus_cos_eq_l816_816225


namespace wheels_in_garage_l816_816416

-- Definitions of the entities within the problem
def cars : Nat := 2
def car_wheels : Nat := 4

def riding_lawnmower : Nat := 1
def lawnmower_wheels : Nat := 4

def bicycles : Nat := 3
def bicycle_wheels : Nat := 2

def tricycle : Nat := 1
def tricycle_wheels : Nat := 3

def unicycle : Nat := 1
def unicycle_wheels : Nat := 1

-- The total number of wheels in the garage
def total_wheels :=
  (cars * car_wheels) +
  (riding_lawnmower * lawnmower_wheels) +
  (bicycles * bicycle_wheels) +
  (tricycle * tricycle_wheels) +
  (unicycle * unicycle_wheels)

-- The theorem we wish to prove
theorem wheels_in_garage : total_wheels = 22 := by
  sorry

end wheels_in_garage_l816_816416


namespace undefined_values_l816_816088

-- Define the expression to check undefined values
noncomputable def is_undefined (x : ℝ) : Prop :=
  x^3 - 9 * x = 0

-- Statement: For which real values of x is the expression undefined?
theorem undefined_values (x : ℝ) : is_undefined x ↔ x = 0 ∨ x = -3 ∨ x = 3 :=
sorry

end undefined_values_l816_816088


namespace principal_sum_l816_816839

noncomputable def diff_simple_compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
(P * ((1 + r / 100)^t) - P) - (P * r * t / 100)

theorem principal_sum (P : ℝ) (r : ℝ) (t : ℝ) (h : diff_simple_compound_interest P r t = 631) (hr : r = 10) (ht : t = 2) :
    P = 63100 := by
  sorry

end principal_sum_l816_816839


namespace part1_sequences_valid_part2_no_infinite_sequence_l816_816150

def valid_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → (∑ i in range (n + 1), a i)^2 = ∑ i in range (n + 1), (a i)^3

theorem part1_sequences_valid (a : ℕ → ℝ) (h : valid_sequence a) :
  a 0 = 1 ∧ ((a 1 = 2 ∧ (a 2 = 3 ∨ a 2 = -2)) ∨ (a 1 = -1 ∧ a 2 = 1)) :=
sorry

theorem part2_no_infinite_sequence (a : ℕ → ℝ) (h : valid_sequence a) (h2 : a 2012 = -2012) :
  false :=
sorry

end part1_sequences_valid_part2_no_infinite_sequence_l816_816150


namespace max_elements_of_set_T_l816_816918

theorem max_elements_of_set_T (T : Set ℕ) 
  (h1 : ∀ x ∈ T, x > 0)  -- positive integers
  (h2 : ∀ x ∈ T, (∑ y in T \ {x}, y) % (T.card - 1) = 0)  -- arithmetic mean condition
  (h3 : 1 ∈ T)  -- 1 is in T
  (h4 : T.max = 1989)  -- 1989 is the largest element
  : T.card ≤ 29 := 
by
  sorry

end max_elements_of_set_T_l816_816918


namespace trig_identity_l816_816284

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end trig_identity_l816_816284


namespace original_garden_side_length_l816_816191

theorem original_garden_side_length (a : ℝ) (h : (a + 3)^2 = 2 * a^2 + 9) : a = 6 :=
by
  sorry

end original_garden_side_length_l816_816191


namespace successful_candidate_is_D_l816_816043

section Recruitment

-- Definitions of the candidates
inductive Candidate
| A | B | C | D | E

open Candidate

-- Given conditions
def conditions : Prop :=
  (∃ masters : fin 5 → Candidate, 
     ∃ doctoral : fin 5 → Candidate,
     masters 0 ≠ masters 1 ∧
     doctoral 0 ≠ doctoral 1 ∧ doctoral 1 ≠ doctoral 2 ∧
     masters 0 ≠ doctoral 0 ∧ masters 0 ≠ doctoral 1) ∧
  (∃ under30 : fin 5 → Candidate,
     ∃ over30 : fin 5 → Candidate,
     under30 0 ≠ under30 1 ∧ under30 1 ≠ under30 2 ∧
     under30 0 ≠ under30 2 ∧
     over30 0 ≠ over30 1) ∧
  (A = under30 0 ∧ C = under30 1) ∧
  (D = over30 0 ∧ E = under30 2) ∧
  (B = doctoral 0 ∧ E = doctoral 1) ∧
  (C = masters 0 ∧ D = doctoral 2) ∧
  (D = over30 0 ∧ masters 1 = D)

-- Goal: To prove the successful candidate is D
theorem successful_candidate_is_D : conditions → ∃ candidate, candidate = D :=
by
  intros h
  existsi D
  have h1 :  candidate = D := sorry
  assumption

end Recruitment

end successful_candidate_is_D_l816_816043


namespace total_chickens_on_farm_l816_816906

noncomputable def total_chickens (H R : ℕ) : ℕ := H + R

theorem total_chickens_on_farm (H R : ℕ) (h1 : H = 9 * R - 5) (h2 : H = 67) : total_chickens H R = 75 := 
by
  sorry

end total_chickens_on_farm_l816_816906


namespace angle_equality_l816_816936

-- Definitions for points A, B, X, Y, P, Q and circles o1, o2
-- The original problem uses the Greek alphabetic angles which we shall directly 
-- convert to their corresponding Lean definitions.
def circles_tangent (o1 o2 : Circle) (A B X Y P Q : Point) : Prop :=
  tangent o1 (Line.through A B) A ∧
  tangent o2 (Line.through A B) B ∧
  intersects o1 o2 X ∧
  intersects o1 o2 Y ∧
  closer_to_line X (Line.through A B) Y ∧
  line_through A X ∩ o2 = P ∧ X ≠ P ∧
  tangent_to_circle o2 P ∩ (Line.through A B) = Q

theorem angle_equality 
  {o1 o2 : Circle} {A B X Y P Q : Point}
  (h : circles_tangent o1 o2 A B X Y P Q) : 
  angle X Y B = angle B Y Q := 
sorry

end angle_equality_l816_816936


namespace sufficient_but_not_necessary_for_perpendicular_l816_816189

variables {R : Type*} [LinearOrderedField R]

-- Definition of perpendicular lines l1 and l2
def perp_lines (l1 l2 : R → R) : Prop :=
  ∃ m1 m2 : R, (l1 = λ x, m1 * x) ∧ (l2 = λ x, m2 * x) ∧ (m1 * m2 = -1)

-- Define the theorem
theorem sufficient_but_not_necessary_for_perpendicular (l1 l2 : R → R) :
  (∃ m1 m2 : R, (l1 = λ x, m1 * x) ∧ (l2 = λ x, m2 * x) ∧ m1 * m2 = -1) →
  perp_lines l1 l2 :=
begin
  sorry
end

end sufficient_but_not_necessary_for_perpendicular_l816_816189


namespace determine_polynomial_l816_816601

theorem determine_polynomial (p : ℝ → ℝ) (h : ∀ x : ℝ, 1 + p x = (p (x - 1) + p (x + 1)) / 2) :
  ∃ b c : ℝ, ∀ x : ℝ, p x = x^2 + b * x + c := by
  sorry

end determine_polynomial_l816_816601


namespace range_of_a_l816_816181

def f : ℝ → ℝ
| x := if x > 0 then log x / log (1/3) else 2^x

theorem range_of_a (a : ℝ) (h : f a > 1/2) : a ∈ set.Ioo (-1) (real.sqrt 3 / 3) :=
by sorry

end range_of_a_l816_816181


namespace Ajay_walk_time_given_distance_l816_816069

def AjayWalkingTime (distance : ℝ) (rate : ℝ) : ℝ := distance / rate

theorem Ajay_walk_time_given_distance 
  (distance : ℝ) (hours : ℝ) (rate : ℝ) 
  (h1 : rate = 6) (h2 : hours = 11.666666666666666) : 
  AjayWalkingTime distance rate = hours :=
by
  sorry

end Ajay_walk_time_given_distance_l816_816069


namespace trig_identity_l816_816279

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end trig_identity_l816_816279


namespace distinct_ordered_pairs_50_l816_816177

theorem distinct_ordered_pairs_50 (a b : ℕ) (h1 : a + b = 50) (h2 : 0 < a) (h3 : 0 < b) : 
    ({p : ℕ × ℕ | p.1 + p.2 = 50 ∧ 0 < p.1 ∧ 0 < p.2}.to_list.length = 49) :=
sorry

end distinct_ordered_pairs_50_l816_816177


namespace arc_length_condition_l816_816941

open Real

noncomputable def hyperbola_eq (a b x y: ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

theorem arc_length_condition (a b r: ℝ) (h1: hyperbola_eq a b 2 1) (h2: r > 0)
  (h3: ∃ x y, x^2 + y^2 = r^2 ∧ hyperbola_eq a b x y) :
  r > 2 * sqrt 2 :=
sorry

end arc_length_condition_l816_816941


namespace count_hexagons_in_diagram_l816_816683

-- Conditions definitions
def hexagons_diagram : Prop := 
  -- Placeholder definition representing the conditions for the diagram 
  -- consisting of hexagons formed by six adjacent triangles and larger hexagons formed 
  -- by combining smaller hexagons.
  sorry  

-- Proof problem
theorem count_hexagons_in_diagram : hexagons_diagram → 12 = 12 :=
by
  intro h,
  sorry -- proof here

end count_hexagons_in_diagram_l816_816683


namespace find_solutions_of_x4_minus_16_l816_816618

noncomputable def solution_set : Set Complex :=
  {2, -2, Complex.I * 2, -Complex.I * 2}

theorem find_solutions_of_x4_minus_16 :
  {x : Complex | x^4 - 16 = 0} = solution_set :=
by
  sorry

end find_solutions_of_x4_minus_16_l816_816618


namespace solve_problem_l816_816679

open Real

noncomputable def problem_statement : Prop :=
  ∃ (p q : ℝ), 1 < p ∧ p < q ∧ (1 / p + 1 / q = 1) ∧ (p * q = 8) ∧ (q = 4 + 2 * sqrt 2)
  
theorem solve_problem : problem_statement :=
sorry

end solve_problem_l816_816679


namespace product_remainder_l816_816632

theorem product_remainder (n : ℕ) (h1 : (n = 4)) (h2 : ∀ i : ℕ, 0 ≤ i ∧ i < 20 → n + i*10 = n):
  (4^2 * 14^2 * 24^2 * ... * 194^2) % 5 = 1 :=
by
  sorry

end product_remainder_l816_816632


namespace tank_empty_time_l816_816049

theorem tank_empty_time
  (V : ℕ)                           -- Volume of the tank in liters.
  (L_rate : ℤ)                      -- Leak rate in liters per hour.
  (I_rate_per_minute : ℤ)           -- Inlet rate in liters per minute.
  (leak_time : ℤ) (L_eq : L_rate = V / leak_time) -- Leak empties the tank in 4 hours.
  (inlet_time : ℤ) (I_eq : inlet_time = 60)  -- Convert inlet rate from per minute to per hour.
  (I_rate : ℤ) (I_rate_def : I_rate = I_rate_per_minute * inlet_time) -- Inlet rate in liters per hour.
  (net_rate : ℤ) (net_rate_def : net_rate = L_rate - I_rate) -- Net rate of emptying in liters per hour.
  (target_time : ℤ) (target_time_eq : target_time = V / net_rate) -- Target time to empty the tank.

  (Volume_val : V = 2160)                 -- Cistern holds 2160 litres.

  (leak_time_val : leak_time = 4)         -- Leak empties tank in 4 hours.
  (I_rate_per_minute_val : I_rate_per_minute = 6) -- Inlet pipe rate: 6 litres a minute.

  : target_time = 12 := by
  rw [Volume_val, leak_time_val, I_rate_per_minute_val] at L_eq I_rate_def net_rate_def target_time_eq,
  have L_rate_val : L_rate = 540, from calc
    L_rate = V / leak_time : L_eq
        ... = 2160 / 4   : by rw Volume_val; exact rfl
        ... = 540       : by norm_num,
  
  have I_rate_val : I_rate = 360, from calc
    I_rate = I_rate_per_minute * inlet_time : I_rate_def
          ... = 6 * 60                       : by norm_num
          ... = 360                          : by norm_num,

  have net_rate_val : net_rate = 180, from calc
    net_rate = L_rate - I_rate : net_rate_def
            ... = 540 - 360    : by rw [L_rate_val, I_rate_val]; exact rfl
            ... = 180          : by norm_num,

  calc
    target_time = V / net_rate       : target_time_eq
             ... = 2160 / 180        : by rw [Volume_val, net_rate_val]; exact rfl
             ... = 12               : by norm_num

end tank_empty_time_l816_816049


namespace winning_lemma_l816_816493

noncomputable def winning_probability (p q : ℝ) : ℝ := p^2 * (3 - 2*p) / (1 - 2*p + 4*p^2 - 2*p^3)

theorem winning_lemma : 
(p q : ℝ) (h_pq : q = 1 - p) (h_p : p = 0.1) → winning_probability p q = 0.033 := 
by
  sorry

end winning_lemma_l816_816493


namespace radio_price_position_l816_816075

def price_positions (n : ℕ) (total_items : ℕ) (rank_lowest : ℕ) : Prop :=
  rank_lowest = total_items - n + 1

theorem radio_price_position :
  ∀ (n total_items rank_lowest : ℕ),
    total_items = 34 →
    rank_lowest = 21 →
    price_positions n total_items rank_lowest →
    n = 14 :=
by
  intros n total_items rank_lowest h_total h_rank h_pos
  rw [h_total, h_rank] at h_pos
  sorry

end radio_price_position_l816_816075


namespace cos_315_eq_sqrt2_div_2_l816_816571

theorem cos_315_eq_sqrt2_div_2 : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
sorry

end cos_315_eq_sqrt2_div_2_l816_816571


namespace solve_x4_eq_16_l816_816615

theorem solve_x4_eq_16 (x : ℂ) : x^4 - 16 = 0 ↔ x = 2 ∨ x = -2 ∨ x = 2 * complex.I ∨ x = -2 * complex.I :=
by
  sorry

end solve_x4_eq_16_l816_816615


namespace probability_is_7_over_26_l816_816366

section VowelProbability

def num_students : Nat := 26

def is_vowel (c : Char) : Bool :=
  c = 'A' || c = 'E' || c = 'I' || c = 'O' || c = 'U' || c = 'Y' || c = 'W'

def num_vowels : Nat := 7

def probability_of_vowel_initials : Rat :=
  (num_vowels : Nat) / (num_students : Nat)

theorem probability_is_7_over_26 :
  probability_of_vowel_initials = 7 / 26 := by
  sorry

end VowelProbability

end probability_is_7_over_26_l816_816366


namespace solve_equation_l816_816383

theorem solve_equation (x : ℝ) (h : x ≠ 2) : 
  2 / (x - 2) = (1 + x) / (x - 2) + 1 → x = 3 / 2 := by
  sorry

end solve_equation_l816_816383


namespace equilateral_triangle_of_roots_of_unity_l816_816770

open Complex

/-- Given three distinct non-zero complex numbers z1, z2, z3 such that z1 * z2 = z3 ^ 2 and z2 * z3 = z1 ^ 2.
Prove that if z2 = z1 * alpha, then alpha is a cube root of unity and the points corresponding to z1, z2, z3
form an equilateral triangle in the complex plane -/
theorem equilateral_triangle_of_roots_of_unity {z1 z2 z3 : ℂ} (h1 : z1 ≠ 0) (h2 : z2 ≠ 0) (h3 : z3 ≠ 0)
  (h_distinct : z1 ≠ z2 ∧ z2 ≠ z3 ∧ z1 ≠ z3)
  (h1_2 : z1 * z2 = z3 ^ 2) (h2_3 : z2 * z3 = z1 ^ 2) (alpha : ℂ) (hz2 : z2 = z1 * alpha) :
  alpha^3 = 1 ∧ ∃ (w1 w2 w3 : ℂ), (w1 = z1) ∧ (w2 = z2) ∧ (w3 = z3) ∧ ((w1, w2, w3) = (z1, z1 * α, z3) 
  ∨ (w1, w2, w3) = (z3, z1, z1 * α) ∨ (w1, w2, w3) = (z1 * α, z3, z1)) 
  ∧ dist w1 w2 = dist w2 w3 ∧ dist w2 w3 = dist w3 w1 := sorry

end equilateral_triangle_of_roots_of_unity_l816_816770


namespace find_theta_colinear_l816_816190

theorem find_theta_colinear (θ : ℝ) 
  (h1 : (2 * Real.cos θ, 2 * Real.sin θ) = (3, Real.sqrt 3) • λ := ∃ k : ℝ, (2 * Real.cos θ, 2 * Real.sin θ) = k • (3, Real.sqrt 3))
  (h2 : 0 ≤ θ ∧ θ < 2 * Real.pi) : 
  θ = Real.pi / 6 ∨ θ = 7 * Real.pi / 6 :=
by
  sorry

end find_theta_colinear_l816_816190


namespace sufficient_not_necessary_condition_l816_816455

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a = 1 → (a - 1) * (a - 2) = 0) ∧ ¬ (∀ (a : ℝ), (a - 1) * (a - 2) = 0 → a = 1) :=
by {
  split,
  { intro h,
    rw h,
    ring, },
  { intro h,
    have h1 : (a - 1) * (a - 2) = 0 → a ≠ 2,
    { intro h2,
      simp at h2,
      cases h2,
      { exfalso,
        exact h2,
      },
      { exact h2, } },
    have h2: (a = 1 ∨ a = 2) → ¬ (a = 2 → a = 1) → (a = 2),
    { intro h3,
      cases h3,
      { exfalso,
        exact h h3, },
      { exact h3, } },
    simp at *,
    exact h2 h, },
  sorry
}

end sufficient_not_necessary_condition_l816_816455


namespace eccentricity_proof_l816_816650

noncomputable def eccentricity_of_ellipse
  (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (c : ℝ)
  (h3 : b^2 = a * c) 
  (h4 : c^2 = a^2 - b^2) :
  ℝ :=
  (real.sqrt 5 - 1) / 2

theorem eccentricity_proof
  (a b c : ℝ) 
  (h1 : 0 < b)
  (h2 : b < a) 
  (h3 : b^2 = a * c) 
  (h4 : c^2 = a^2 - b^2 ) :
  eccentricity_of_ellipse a b h1 h2 c h3 h4 = (real.sqrt 5 - 1) / 2 :=
  sorry

end eccentricity_proof_l816_816650


namespace simplify_expression_l816_816827

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end simplify_expression_l816_816827


namespace arithmetic_and_geometric_sequences_sum_of_c_n_l816_816649

-- Given conditions
variables {a : ℕ → ℤ} {b : ℕ → ℤ} {c : ℕ → ℚ}
variables {d : ℤ} (hd : d > 1)
variables (hS10 : ∑ i in finset.range 10, a i = 100)
variables (h_seq1 : b 0 = a 0) (h_seq2 : b 1 = 2)
variables (q_eq_d : b 1 = b 0 * d)

-- Definitions from conditions
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_seq (b : ℕ → ℤ) (q : ℤ) : Prop :=
  ∀ n : ℕ, b (n + 1) = b n * q

def c_n (a : ℕ → ℤ) (b : ℕ → ℤ) (n : ℕ) : ℚ :=
  (a n - 2) / (b n : ℚ)

def sum_Tn (c : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ i in finset.range n, c i

-- Proofs required
theorem arithmetic_and_geometric_sequences (h_arith : arithmetic_seq a d)
  (h_geo : geometric_seq b d) : 
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, b n = 2 ^ n) :=
sorry

theorem sum_of_c_n (h_arith : arithmetic_seq a d)
  (h_geo : geometric_seq b d) :
  ∀ n, sum_Tn (c_n a b) n = 2 - (2 * n + 1) / (2 ^ (n - 1)) :=
sorry

end arithmetic_and_geometric_sequences_sum_of_c_n_l816_816649


namespace sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816211

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816211


namespace marla_adds_blue_paint_l816_816364

variable (M B : ℝ)

theorem marla_adds_blue_paint :
  (20 = 0.10 * M) ∧ (B = 0.70 * M) → B = 140 := 
by 
  sorry

end marla_adds_blue_paint_l816_816364


namespace no_x2_term_a_eq_1_l816_816635

theorem no_x2_term_a_eq_1 (a : ℝ) : 
  (∀ x : ℝ, (x^2 + a * x + 1) * (x^2 - 3 * a + 2) = x^4 + bx^3 + cx + d) →
  c = 0 →
  a = 1 :=
sorry

end no_x2_term_a_eq_1_l816_816635


namespace trigonometric_identity_l816_816275

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end trigonometric_identity_l816_816275


namespace overall_average_mark_l816_816029

theorem overall_average_mark :
  ∀ (students : ℕ × ℕ × ℕ × ℕ) (mean_marks : ℕ × ℕ × ℕ × ℕ),
    students = (55, 35, 45, 42) →
    mean_marks = (50, 60, 55, 45) →
    let total_marks := (55 * 50) + (35 * 60) + (45 * 55) + (42 * 45)
    in
    let total_students := 55 + 35 + 45 + 42
    in
    (total_marks / total_students : ℤ) = 52.09 :=
by
  intros students mean_marks hs hm
  rw hs at *
  rw hm at *
  let total_marks := (55 * 50) + (35 * 60) + (45 * 55) + (42 * 45)
  let total_students := 55 + 35 + 45 + 42
  have h : total_marks = 9215 := rfl
  have h2 : total_students = 177 := rfl
  rw [h, h2]
  iterate { sorry }

end overall_average_mark_l816_816029


namespace solve_cosine_equation_l816_816026

theorem solve_cosine_equation (x : ℝ) : 
  (∃ k : ℤ, x = (π * (6 * k + 1)) / 30 ∨ x = (π * (6 * k - 1)) / 30) ↔
  (cos (8 * x) + 3 * cos (4 * x) + 3 * cos (2 * x) = 8 * cos x * (cos (3 * x))^3 - 0.5) :=
by 
  sorry

end solve_cosine_equation_l816_816026


namespace difference_of_squares_l816_816861

theorem difference_of_squares (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 16) : x^2 - y^2 = 960 :=
by
  sorry

end difference_of_squares_l816_816861


namespace first_investment_exceeds_two_million_l816_816467

theorem first_investment_exceeds_two_million 
  (initial_investment : ℝ)
  (increase_rate : ℝ)
  (n : ℕ)
  (year : ℕ)
  (lg1_12 : ℝ)
  (lg1_3 : ℝ)
  (lg2 : ℝ)
  (h_initial : initial_investment = 1.3)
  (h_rate : increase_rate = 0.12)
  (h_year : year = 2016)
  (h_lg1_12 : lg1_12 = 0.05)
  (h_lg1_3 : lg1_3 = 0.11)
  (h_lg2 : lg2 = 0.30) :
  initial_investment * (1 + increase_rate) ^ (n - year) > 2 → n = 2020 :=
by
  assume hinv : initial_investment * (1 + increase_rate) ^ (n - year) > 2
  sorry

end first_investment_exceeds_two_million_l816_816467


namespace discount_difference_l816_816143

-- Definitions based on given conditions
def original_bill : ℝ := 8000
def single_discount_rate : ℝ := 0.30
def first_successive_discount_rate : ℝ := 0.26
def second_successive_discount_rate : ℝ := 0.05

-- Calculations based on conditions
def single_discount_final_amount := original_bill * (1 - single_discount_rate)
def first_successive_discount_final_amount := original_bill * (1 - first_successive_discount_rate)
def complete_successive_discount_final_amount := 
  first_successive_discount_final_amount * (1 - second_successive_discount_rate)

-- Proof statement
theorem discount_difference :
  single_discount_final_amount - complete_successive_discount_final_amount = 24 := 
  by
    -- Proof to be provided
    sorry

end discount_difference_l816_816143


namespace total_price_increase_percentage_l816_816690

theorem total_price_increase_percentage 
    (P : ℝ) 
    (h1 : P > 0) 
    (P_after_first_increase : ℝ := P * 1.2) 
    (P_after_second_increase : ℝ := P_after_first_increase * 1.15) :
    ((P_after_second_increase - P) / P) * 100 = 38 :=
by
  sorry

end total_price_increase_percentage_l816_816690


namespace exists_relatively_prime_integer_in_consecutive_set_l816_816376

theorem exists_relatively_prime_integer_in_consecutive_set :
  ∀ (A : set ℤ), (∀ n : ℤ, (n ∈ A ↔ n ≥ 0 ∧ n < 16)) →
  (∃ a ∈ A, ∀ b ∈ A, a ≠ b → gcd a b = 1) :=
by
  sorry

end exists_relatively_prime_integer_in_consecutive_set_l816_816376


namespace apple_boxes_calculation_l816_816869

theorem apple_boxes_calculation : 
  let apples_per_crate := 250
  let crates_delivered := 20
  let rotten_apples := 320
  let apples_per_box := 25
  let total_apples := apples_per_crate * crates_delivered
  let good_apples := total_apples - rotten_apples
  let boxes := good_apples / apples_per_box
  (boxes.floor = 187) := 
by {
  sorry
}

end apple_boxes_calculation_l816_816869


namespace fly_path_max_length_correct_l816_816048

noncomputable def fly_path_max_length (side_length : ℝ) : ℝ :=
  if side_length = 2 then 4 * Real.sqrt 3 + 8 * Real.sqrt 2 + 4 else 0

theorem fly_path_max_length_correct :
  fly_path_max_length 2 = 4 * Real.sqrt 3 + 8 * Real.sqrt 2 + 4 :=
by
  rw [fly_path_max_length]
  simp
  sorry

end fly_path_max_length_correct_l816_816048


namespace unique_prime_n_l816_816107

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_prime_n (n : ℕ)
  (h1 : isPrime n)
  (h2 : isPrime (n^2 + 10))
  (h3 : isPrime (n^2 - 2))
  (h4 : isPrime (n^3 + 6))
  (h5 : isPrime (n^5 + 36)) : n = 7 :=
by
  sorry

end unique_prime_n_l816_816107


namespace midpoint_coordinates_l816_816113

theorem midpoint_coordinates :
  let A := (2 : ℤ, 9 : ℤ)
  let B := (8 : ℤ, -3 : ℤ)
  let midpoint (P Q : ℤ × ℤ) : ℤ × ℤ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  midpoint A B = (5, 3) :=
by
  sorry

end midpoint_coordinates_l816_816113


namespace cos_315_proof_l816_816551

noncomputable def cos_315_eq : Prop :=
  let theta := 45 in
  let angle := 360 - theta in
  ∃ c : real, (cos 45) = (sqrt 2 / 2) ∧ (cos (360 - theta) = cos theta) ∧ 315 = angle ∧ (forall a : real, cos a = cos (360 - a)) ∧ (315 ∈ (4th quadrant)) ∧ (cos 315 = cos 45)

theorem cos_315_proof : (cos 315 = sqrt 2 / 2) :=
begin
  -- We start by defining basic facts
  -- 315 = 360 - 45
  -- Cosine is positive in the fourth quadrant, known value \cos 45 = sqrt 2 / 2
  -- Cosine property \cos (360 - \theta) = \cos \theta
  let theta := 45,
  let angle := 360 - theta,
  have H1 : cos theta = sqrt 2 / 2,
  {
    -- known fact
    sorry,
  },
  have H2 : cos (360 - theta) = cos theta,
  {
    -- cosine property
    sorry,
  },
  have H3 : angle = 315,
  {
    -- subtraction identity
    sorry,
  },
  rw H3,
  rw H2,
  rw H1,
  refl,
end

end cos_315_proof_l816_816551


namespace sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816206

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816206


namespace third_level_score_l816_816344

-- Define the conditions
variable (x : ℕ)

def first_level_score : ℕ := 2
def second_level_score : ℕ := 3
def fourth_level_score : ℕ := 8
def fifth_level_score : ℕ := 12
def sixth_level_score : ℕ := 17

-- Define the increases based on the given conditions and pattern
def increase_1_to_2 : ℕ := second_level_score - first_level_score
def increase_2_to_4 : ℕ := fourth_level_score - second_level_score
def increase_4_to_5 : ℕ := fifth_level_score - fourth_level_score
def increase_5_to_6 : ℕ := sixth_level_score - fifth_level_score

-- Assuming the pattern of increases: +1, +2, +3, +4, +5
def expected_increase_2_to_3 : ℕ := 2

-- Main statement to be proved
theorem third_level_score : x = 5 :=
by
  have first_to_second_increase : increase_1_to_2 = 1 := by rfl
  have second_to_fourth_average_increase : increase_2_to_4 = 5 := by rfl
  have fourth_to_fifth_increase : increase_4_to_5 = 4 := by rfl
  have fifth_to_sixth_increase : increase_5_to_6 = 5 := by rfl

  -- Calculate the expected score for the third level
  have third_level_points : x = second_level_score + expected_increase_2_to_3 := by rfl
  show x = 5, from third_level_points

-- Add sorry to skip the proof
sorry

end third_level_score_l816_816344


namespace fx_periodicity_fx_value_at_2_l816_816641

noncomputable def f (x : ℝ) : ℝ := sorry

theorem fx_periodicity (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f(x + 1) = 2 - f(x)) (h2 : ∀ x : ℝ, f(x + 2) = 2 - f(-x)) :
  ∀ x : ℝ, f(x + 2) = f(x) :=
by
  sorry

theorem fx_value_at_2 (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f(x + 1) = 2 - f(x)) (h2 : ∀ x : ℝ, f(x + 2) = 2 - f(-x)) :
  f(2) = 1 :=
by
  sorry

end fx_periodicity_fx_value_at_2_l816_816641


namespace speech_competition_arrangements_l816_816721

noncomputable def students : Type := {A, B, C, D, E, F : String}
open List Function

-- Define the property for consecutive positions
def consecutive (l : List String) (x y : String) : Prop :=
  ∃ i : ℕ, i < l.length - 1 ∧ (l.nthLe i (by linarith) = x ∧ l.nthLe (i + 1) (by linarith) = y) ∨ (l.nthLe i (by linarith) = y ∧ l.nthLe (i + 1) (by linarith) = x)

-- Define the property for A not being first or last
def A_not_first_last (l : List String) : Prop :=
  l.length = 6 ∧ l.head ≠ "A" ∧ l.last ≠ "A"

-- The main theorem stating the problem
theorem speech_competition_arrangements :
  ∃ l : List String, officials l ∧ consecutive l "B" "C" :=
begin
  sorry
end

end speech_competition_arrangements_l816_816721


namespace problem_l816_816502

structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

def NotParallel (v1 v2 : Vector3D) : Prop := ¬ ∃ k : ℝ, v2 = ⟨k * v1.x, k * v1.y, k * v1.z⟩

def a : Vector3D := ⟨1, 2, -2⟩
def b : Vector3D := ⟨-2, -4, 4⟩
def c : Vector3D := ⟨1, 0, 0⟩
def d : Vector3D := ⟨-3, 0, 0⟩
def g : Vector3D := ⟨-2, 3, 5⟩
def h : Vector3D := ⟨16, 24, 40⟩
def e : Vector3D := ⟨2, 3, 0⟩
def f : Vector3D := ⟨0, 0, 0⟩

theorem problem : NotParallel g h := by
  sorry

end problem_l816_816502


namespace hexagonal_prism_min_cut_l816_816083

-- We formulate the problem conditions and the desired proof
def minimum_edges_to_cut (total_edges : ℕ) (uncut_edges : ℕ) : ℕ :=
  total_edges - uncut_edges

theorem hexagonal_prism_min_cut :
  minimum_edges_to_cut 18 7 = 11 :=
by
  sorry

end hexagonal_prism_min_cut_l816_816083


namespace greg_less_jacob_l816_816804

/-- Given:
1. Patrick has 4 hours less than twice the amount of time that Greg has left.
2. Jacob has 18 hours left to finish his homework.
3. The total of all their hours left to finish their homework is 50 hours.
Prove:
Greg has 6 hours less than Jacob.
-/
theorem greg_less_jacob : ∃ (G : ℕ), Patrick_hours G = 2 * G - 4 ∧
                                       Jacob_hours = 18 ∧
                                       Total_hours G (2 * G - 4) 18 = 50 ∧
                                       Jacob_hours - G = 6 :=
by
  -- Definitions for conditions
  def Patrick_hours (G : ℕ) : ℕ := 2 * G - 4
  def Jacob_hours : ℕ := 18
  def Total_hours (G P J : ℕ) : ℕ := G + P + J
  sorry

end greg_less_jacob_l816_816804


namespace max_x_ineql_l816_816117

noncomputable def maxXThatSatisfiesInequality (x : ℝ) : ℝ :=
  if x = 1 then 1 else 0

theorem max_x_ineql (x : ℝ) : 
  (6 + 5 * x + x^2) * sqrt (2 * x^2 - x^3 - x) ≤ 0 ↔ x ≤ 1 :=
begin
  sorry
end

end max_x_ineql_l816_816117


namespace speed_of_center_rod_l816_816920

/-- Define the conditions as hypotheses -/
variables (v1 v2 : ℝ)
axioms (h_v1 : v1 = 10) (h_v2 : v2 = 6)

/-- Theorem statement -/
theorem speed_of_center_rod : 
  let v_vert_A := real.sqrt (v1^2 - v2^2) in
  let v_vert_B := 1/2 * v_vert_A in
  let v_center := real.sqrt (v2^2 + v_vert_B^2) in
  v_center = 7.2 :=
by
  sorry

end speed_of_center_rod_l816_816920


namespace combination_10_choose_4_l816_816339

theorem combination_10_choose_4 : nat.choose 10 4 = 210 := by 
  sorry

end combination_10_choose_4_l816_816339


namespace cookie_distribution_probability_l816_816895

theorem cookie_distribution_probability :
  let total_cookies := 12
  let types := 3
  let each_type := 4
  let children := 4
  let cookies_per_child := 3
  let p := 72
  let q := 1925
  let probability := ⟨p, q⟩ 

  (4 * 4 * 4 / (total_cookies choose cookies_per_child)) *
  (3 * 3 * 3 / ((total_cookies - cookies_per_child) choose cookies_per_child)) *
  (2 * 2 * 2 / ((total_cookies - 2 * cookies_per_child) choose cookies_per_child)) *
  1 = probability ∧ Nat.gcd p q = 1 ∧
  p + q = 1997 :=
by
  sorry

end cookie_distribution_probability_l816_816895


namespace computer_rental_fees_l816_816481

noncomputable def hourlyCharges (A B C T: ℝ) : Prop :=
  (A = 1.4 * B) ∧
  (B * (T + 20) = 550) ∧
  (C = 0.75 * B) ∧
  (T_C = T + 10) ∧
  (T_B - T_C = 5) ∧
  (A * T = 550) ∧
  (C * (T + 10) = 550)

theorem computer_rental_fees :
  ∃ (A B C T: ℝ),
  hourlyCharges A B C T ∧
  (A ≈ 11) ∧
  (B ≈ 7.86) ∧
  (C ≈ 5.90)
:= sorry

end computer_rental_fees_l816_816481


namespace remainder_poly1_poly2_l816_816014

noncomputable def poly1 : Polynomial ℤ := 3 * X^5 + 5 * X^4 - 13 * X^3 - 7 * X^2 + 52 * X - 34
noncomputable def poly2 : Polynomial ℤ := X^3 + 6 * X^2 + 5 * X - 7
noncomputable def remainder : Polynomial ℤ := 50 * X^3 + 79 * X^2 - 39 * X - 34

theorem remainder_poly1_poly2 :
  polynomial.divMod poly1 poly2 = (0, remainder) :=
sorry

end remainder_poly1_poly2_l816_816014


namespace trigonometric_identity_l816_816277

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end trigonometric_identity_l816_816277


namespace bee_population_reaches_one_fourth_l816_816042

-- Define the exponential decay condition.
def exponential_decay (P : ℝ) (r : ℝ) (d : ℕ) : ℝ :=
  P * (r ^ d)

-- Define the problem condition.
def bee_population_condition (P : ℝ) [hP : 0 < P] : Prop :=
  ∃ d : ℕ, exponential_decay P 0.994 d ≈ (P / 4) ∧ d ≈ 231

-- The main statement of our proof problem.
theorem bee_population_reaches_one_fourth (P : ℝ) [hP : 0 < P] : bee_population_condition P :=
sorry

end bee_population_reaches_one_fourth_l816_816042


namespace trigonometric_identity_l816_816276

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end trigonometric_identity_l816_816276


namespace fifth_term_of_arithmetic_sequence_is_minus_three_l816_816403

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

theorem fifth_term_of_arithmetic_sequence_is_minus_three (a d : ℤ) :
  (arithmetic_sequence a d 11 = 25) ∧ (arithmetic_sequence a d 12 = 29) →
  (arithmetic_sequence a d 4 = -3) :=
by 
  intros h
  sorry

end fifth_term_of_arithmetic_sequence_is_minus_three_l816_816403


namespace option_a_option_b_option_c_option_d_l816_816021

theorem option_a (x y : ℝ) : (x + 2 * y) * (x - 2 * y) = x^2 - 2 * y^2 → false :=
by
  intro h
  calc (x + 2 * y) * (x - 2 * y) = x^2 - 4 * y^2 : by ring
  ... = x^2 - 2 * y^2 : by none

theorem option_b (x y : ℝ) : (x - y) * (-x - y) = -x^2 - y^2 → false :=
by
  intro h
  calc (x - y) * (-x - y) = -(x^2 - y^2) : by ring
  ... = y^2 - x^2 : by none
  ... = -x^2 - y^2 : by none

theorem option_c (x y : ℝ) : (x - y)^2 = x^2 - 2 * x * y + y^2 :=
by
  calc (x - y)^2 = (x - y) * (x - y) : by ring
  ... = x^2 - 2 * x * y + y^2 : by ring

theorem option_d (x y : ℝ) : (x + y)^2 = x^2 + y^2 → false :=
by
  intro h
  calc (x + y)^2 = (x + y) * (x + y) : by ring
  ... = x^2 + 2 * x * y + y^2 : by ring
  ... = x^2 + y^2 : by none

end option_a_option_b_option_c_option_d_l816_816021


namespace people_per_seat_eq_three_l816_816309

noncomputable def seats_on_left := 15
noncomputable def seats_on_right := seats_on_left - 3
noncomputable def back_seat_capacity := 9
noncomputable def total_capacity := 90
noncomputable def x := 3

-- The number of people each seat can hold
theorem people_per_seat_eq_three :
  ∀ x : ℕ,
    15 * x + (15 - 3) * x + 9 = 90 → x = 3 :=
by
  intro x
  assume h : 15 * x + (15 - 3) * x + 9 = 90
  -- We state that x must equal 3
  have : x = 3,
  sorry
  exact this

end people_per_seat_eq_three_l816_816309


namespace complex_magnitude_power_six_l816_816932

theorem complex_magnitude_power_six :
  ∃ c : ℝ, abs ((1 : ℂ) + (real.sqrt 2 : ℂ) * I) ^ 6 = c ∧ c = 27 :=
by
  sorry

end complex_magnitude_power_six_l816_816932


namespace number_of_small_cubes_with_one_side_painted_l816_816582

noncomputable def small_cubes_with_one_side_painted (edge_large_cube : ℝ) (edge_small_cube : ℝ) (faces_painted : ℕ) : ℕ :=
  let cubes_per_edge := edge_large_cube / edge_small_cube in
  let grid_per_face := (cubes_per_edge.to_nat - 2) * (cubes_per_edge.to_nat - 2) in
  grid_per_face * faces_painted

theorem number_of_small_cubes_with_one_side_painted :
  small_cubes_with_one_side_painted 10 2 6 = 54 :=
by {
  unfold small_cubes_with_one_side_painted,
  norm_num,
  sorry
}

end number_of_small_cubes_with_one_side_painted_l816_816582


namespace no_solutions_xyz_l816_816125

theorem no_solutions_xyz : ∀ (x y z : ℝ), x + y = 3 → xy - z^2 = 2 → false := by
  intros x y z h1 h2
  sorry

end no_solutions_xyz_l816_816125


namespace area_of_region_equals_area_of_triangle_l816_816575

-- Define the isosceles right triangle with given legs and hypotenuse
structure Triangle :=
(a : ℝ)  -- Length of the legs
(h : a > 0)  -- Positive length constraint

-- Define semicircle with the diameter BC
def semicircle_radius (t : Triangle) : ℝ :=
(t.a * real.sqrt 2) / 2

-- Define the quarter-circle inscribed inside the triangle tangent to sides AB and AC
def quarter_circle_radius (t : Triangle) : ℝ :=
t.a / real.sqrt 2

-- Define the area of the triangle
def area_of_triangle (t : Triangle) : ℝ :=
(t.a ^ 2) / 2

-- Define the area of the semicircle
def area_of_semicircle (t : Triangle) : ℝ :=
π * (semicircle_radius t) ^ 2 / 2

-- Define the area of the quarter-circle
def area_of_quarter_circle (t : Triangle) : ℝ :=
π * (quarter_circle_radius t) ^ 2 / 4

-- Define the area of triangle OBC
def area_of_OBC (t : Triangle) : ℝ :=
(t.a ^ 2) / 4

-- Define the area of the region enclosed between the arc and the semicircle
def area_of_region (t : Triangle) : ℝ :=
area_of_semicircle t - area_of_quarter_circle t - area_of_OBC t

-- Main theorem statement
theorem area_of_region_equals_area_of_triangle (t : Triangle) :
  area_of_region t = area_of_triangle t :=
sorry

end area_of_region_equals_area_of_triangle_l816_816575


namespace incorrect_statement_l816_816740

noncomputable def f (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

theorem incorrect_statement
  (a b c : ℤ) (h₀ : a ≠ 0)
  (h₁ : 2 * a + b = 0)
  (h₂ : f a b c 1 = 3)
  (h₃ : f a b c 2 = 8) :
  ¬ (f a b c (-1) = 0) :=
sorry

end incorrect_statement_l816_816740


namespace range_of_a_l816_816670

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ -1 → f x a ≥ a) ↔ -2 ≤ a ∧ a ≤ (-1 + Real.sqrt 5) / 2 := 
sorry

end range_of_a_l816_816670


namespace vector_magnitude_proof_l816_816165

variables (a b : ℝ)
variables (angle_45 : ℝ := real.pi / 4)
variables (a_magnitude : ℝ := real.sqrt 2)
variables (b_magnitude : ℝ := 3)

theorem vector_magnitude_proof
(h_angle : real.cos angle_45 = real.sqrt 2 / 2)
(h_a_mag : real.abs a = a_magnitude)
(h_b_mag : real.abs b = b_magnitude)
(h_dot_product : a * b = 3) :
  real.abs (2 * a - b) = real.sqrt 5 :=
sorry

end vector_magnitude_proof_l816_816165


namespace num_functions_fixed_point_difference_correct_l816_816133

noncomputable def num_functions_fixed_point_difference 
  (p : ℕ) [Fact (Nat.Prime p)] 
  (a e : ℕ) (hp : 2 ≤ a) (he : 1 ≤ e) 
  (h_div : ∃ i b, a = p^i * b ∧ ¬ p ∣ b) : 
  ℕ :=
  let ⟨i, b, ha, hb⟩ := h_div in
  p^(e * (a - p^i))

theorem num_functions_fixed_point_difference_correct 
  (p : ℕ) [Fact (Nat.Prime p)] 
  (a e : ℕ) (hp : 2 ≤ a) (he : 1 ≤ e) 
  (h_div : ∃ i b, a = p^i * b ∧ ¬ p ∣ b) :
  ∃ f : (ℤ / a) → (ℤ / (p^e)), ∃ k ≥ 1, ∆^k f = f → 
  (num_functions_fixed_point_difference p a e hp he h_div = p^(e * (a - p^i))) := 
  sorry

end num_functions_fixed_point_difference_correct_l816_816133


namespace trigonometric_identity_l816_816274

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end trigonometric_identity_l816_816274


namespace trigonometric_identity_l816_816269

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end trigonometric_identity_l816_816269


namespace find_m_find_a₁_l816_816994

theorem find_m 
  (a₃ : ℚ) 
  (S₃ : ℚ) 
  (h₁ : a₃ = 3 / 2) 
  (h₂ : S₃ = 9 / 2) :
  (m : ℚ) → m = 3 * Real.sqrt(3) / 2 ∨ m = - 3 * Real.sqrt(3) / 2 :=
by
  sorry

theorem find_a₁
  (a₃ : ℚ)
  (S₃ : ℚ)
  (h₁ : a₃ = 3 / 2)
  (h₂ : S₃ = 9 / 2) 
  (q : ℚ)
  (h₃ : q = 1 ∨ ∃ a₁ : ℚ, a₁ * q ^ 2 = 3 / 2 ∧ a₁ * (1 - q ^ 3) / (1 - q) = 9 / 2) :
  (a₁ : ℚ) → a₁ = 3 / 2 ∨ a₁ = 6 :=
by
  sorry

end find_m_find_a₁_l816_816994


namespace cos_315_eq_sqrt2_div2_l816_816537

theorem cos_315_eq_sqrt2_div2 : Real.cos (315 * Real.pi / 180) = sqrt 2 / 2 := 
sorry

end cos_315_eq_sqrt2_div2_l816_816537


namespace increase_in_breadth_l816_816393

theorem increase_in_breadth (L B : ℝ) (hL : ∀ L', L' = 1.11 * L)
  (hA : ∀ A', A' = 1.3542 * (L * B)) :
  ∃ p : ℝ, p = 22 :=
by
  let L' := 1.11 * L
  let A' := 1.3542 * (L * B)
  have H : 1.11 * (1 + p / 100) = 1.3542
  have hp : p = 22
  use hp
  sorry

end increase_in_breadth_l816_816393


namespace solve_quadratic_eq_l816_816514

theorem solve_quadratic_eq (x : ℝ) (h : (x + 5) ^ 2 = 16) : x = -1 ∨ x = -9 :=
sorry

end solve_quadratic_eq_l816_816514


namespace f_l816_816700

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^4 + b * x^2 - x

-- Define the derivative f'(x)
def f' (a b x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x - 1

-- Problem statement: Prove that f'(-1) = -5 given the conditions
theorem f'_neg_one_value (a b : ℝ) (h : f' a b 1 = 3) : f' a b (-1) = -5 :=
by
  -- Placeholder for the proof
  sorry

end f_l816_816700


namespace xavier_covers_distance_in_48_minutes_l816_816439

-- Definitions of the constants
def initialSpeed : ℝ := 60  -- in kmph
def speedIncrease : ℝ := 10  -- in kmph every 12 minutes
def intervalTime : ℝ := 12 / 60  -- 12 minutes in hours
def totalDistance : ℝ := 60  -- the distance between point P and point Q in km
def totalTime : ℝ := 48 / 60  -- total time taken in hours (which we need to prove)

-- Definition of the function to calculate distance after intervals
def distance (n : ℕ) : ℝ :=
  (initialSpeed + (n : ℝ) * speedIncrease) * intervalTime

-- Definition to calculate the total distance covered after n intervals
def totalDistanceCovered (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i => distance i)

-- Theorem that states Xavier takes 48 minutes (0.8 hours) to cover 60 km
theorem xavier_covers_distance_in_48_minutes :
  totalDistanceCovered 4 = totalDistance := sorry

end xavier_covers_distance_in_48_minutes_l816_816439


namespace distance_between_points_l816_816626

theorem distance_between_points : 
  let p1 := (5 : ℝ, 5 : ℝ) 
  let p2 := (0 : ℝ, 0 : ℝ) 
  dist p1 p2 = 5 * Real.sqrt 2 :=
by
  sorry

end distance_between_points_l816_816626


namespace sin_lt_alpha_lt_tan_l816_816017

open Real

theorem sin_lt_alpha_lt_tan {α : ℝ} (hα1 : 0 < α) (hα2 : α < π / 2) : sin α < α ∧ α < tan α := by
  sorry

end sin_lt_alpha_lt_tan_l816_816017


namespace difference_of_squares_l816_816862

theorem difference_of_squares (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 16) : x^2 - y^2 = 960 :=
by
  sorry

end difference_of_squares_l816_816862


namespace sin_minus_cos_l816_816239

noncomputable def theta_condition (θ : ℝ) : Prop := (0 < θ) ∧ (θ < π / 2) ∧ (Real.tan θ = 1 / 3)

theorem sin_minus_cos (θ : ℝ) (hθ : theta_condition θ) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_minus_cos_l816_816239


namespace quadratic_inequality_l816_816705

variable {a b c : ℝ}

-- Define the parabola function
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions
axiom h1 : ∀ x, -2 < x ∧ x < 4 → a * x^2 + b * x + c > 0
axiom h2 : ∀ x, x ≤ -2 ∨ x ≥ 4 → a * x^2 + b * x + c ≤ 0

-- Given conditions (a, b, c) forming specific roots
axiom a_neg : a < 0
axiom b_eq : b = -2 * a
axiom c_eq : c = -8 * a

-- To prove the specific inequality
theorem quadratic_inequality : f 2 > f (-1) ∧ f (-1) > f 5 :=
by
  sorry

end quadratic_inequality_l816_816705


namespace exists_good_subset_M_l816_816355

def M := {1, 2, ... 20}

def is_9_element_subset (s : Finset ℕ) : Prop :=
  s.card = 9 ∧ s ⊆ M

def f (s : Finset ℕ) (h : is_9_element_subset s) : ℕ :=
  sorry -- Function definition omitted for the sake of the statement.

noncomputable def exists_10_element_good_subset (M : Set ℕ) : Prop :=
  ∃ T : Finset ℕ,
    T.card = 10 ∧ T ⊆ M ∧ 
    ∀ k ∈ T, f (T.erase k) sorry ≠ k

theorem exists_good_subset_M : exists_10_element_good_subset M :=
  sorry

end exists_good_subset_M_l816_816355


namespace halfway_between_ratios_l816_816853

theorem halfway_between_ratios :
  let a := (1 : ℚ) / 8
  let b := (1 : ℚ) / 3
  (a + b) / 2 = 11 / 48 := by
  sorry

end halfway_between_ratios_l816_816853


namespace smallest_y_for_perfect_cube_l816_816939

theorem smallest_y_for_perfect_cube (y : ℕ) : 
  let x := 9 * 36 * 54 in
  (∃ y : ℕ, y > 0 ∧ ∀ a b : ℕ, x * y = a^3 * b^3) → y = 9 :=
by sorry

end smallest_y_for_perfect_cube_l816_816939


namespace line_MN_parallel_angle_bisector_A_l816_816801

variables {A B C E F N M : Point}
variable (triangle_ABC : Triangle A B C)
variable (mid_M_EF : M = midpoint E F)
variables (segment_AE_BC : A E = distance B C)
variables (segment_BF_AC : B F = distance A C)
variables (circle_N_touches_BF : Circle touches N B F)
variables (circle_N_touches_BC : Circle touches N B C)
variables (circle_N_touches_extension_AC : Circle touches N (extension A C))

theorem line_MN_parallel_angle_bisector_A
  (h_TRIANGLE : Triangle A B C)
  (h_M_midpoint : M = midpoint E F)
  (h_AE_BC : distance A E = distance B C)
  (h_BF_AC : distance B F = distance A C)
  (h_Circle_touches_BF : Circle touches segment N B F)
  (h_Circle_touches_BC : Circle touches segment N B C)
  (h_Circle_touches_extension_AC : Circle touches extension A C) :
  parallel (line M N) (angle_bisector (angle A)) :=
sorry

end line_MN_parallel_angle_bisector_A_l816_816801


namespace determine_fx_l816_816392

def f (x : ℝ) : ℝ := sorry

theorem determine_fx (x : ℝ) (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) :
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x ≥ y → f((x + y) / 2) = f(x) * real.sin α + f(y) * (1 - real.sin α)) → 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f(x) = x) :=
  by
    intro h
    sorry

end determine_fx_l816_816392


namespace odd_function_among_options_l816_816071

-- Defining the given functions
def fA (x : ℝ) : ℝ := x^2 * sin x
def fB (x : ℝ) : ℝ := x^2 * cos x
def fC (x : ℝ) : ℝ := |log x|
def fD (x : ℝ) : ℝ := 2^(-x)

-- Defining the property of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Defining the main statement to be proven
theorem odd_function_among_options :
  is_odd_function fA := 
sorry

end odd_function_among_options_l816_816071


namespace complex_in_third_quadrant_l816_816729

-- Define the complex number and the multiplication by i
def complex_num := -2 + 3 * complex.I
def multiplied_by_i := complex.I * complex_num

-- Define the quadrants
def third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- State the theorem
theorem complex_in_third_quadrant : third_quadrant multiplied_by_i :=
by 
  sorry -- Skipping the proof

end complex_in_third_quadrant_l816_816729


namespace part_I_part_II_l816_816140

variable (α : ℝ)

-- Given condition
def tan_alpha : ℝ := 3
def condition := tan α = tan_alpha

-- First part of the proof
theorem part_I (h : condition) : (4 * sin α - 2 * cos α) / (5 * cos α + 3 * sin α) = 5 / 7 :=
by sorry

-- Second part of the proof
theorem part_II (h : condition) : (sin α + cos α) ^ 2 = 8 / 5 :=
by sorry

end part_I_part_II_l816_816140


namespace lunch_combinations_l816_816461

theorem lunch_combinations :
  let num_meat := 4 in
  let num_veg := 7 in
  let choose := @nat.choose in
  (choose num_meat 2 * choose num_veg 2) + 
  (choose num_meat 1 * choose num_veg 2) = 210 :=
by
  sorry

end lunch_combinations_l816_816461


namespace dividend_calculation_l816_816875

theorem dividend_calculation (divisor quotient remainder : ℤ) (h1 : divisor = 800) (h2 : quotient = 594) (h3 : remainder = -968) :
  (divisor * quotient + remainder) = 474232 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end dividend_calculation_l816_816875


namespace distinct_colorings_l816_816898

def sections : ℕ := 6
def red_count : ℕ := 3
def blue_count : ℕ := 1
def green_count : ℕ := 1
def yellow_count : ℕ := 1

def permutations_without_rotation : ℕ := Nat.factorial sections / 
  (Nat.factorial red_count * Nat.factorial blue_count * Nat.factorial green_count * Nat.factorial yellow_count)

def rotational_symmetry : ℕ := permutations_without_rotation / sections

theorem distinct_colorings (rotational_symmetry) : rotational_symmetry = 20 :=
  sorry

end distinct_colorings_l816_816898


namespace segment_lengths_l816_816360

-- Define the primary entities in our problem
variables (O E F G H J M : ℝ)
-- Circle with radius 10 units
def radius := 10
-- EF and GH are perpendicular diameters
def perpendicular (EF GH : ℝ) := ∀ t, E = O - radius * t ∧ F = O + radius * t ∧ 
{} G = O - radius * t ∧ H = O + radius * t
-- Chord EJ intersects GH at M, and EJ = 12 units
def chord_intersects (EJ M GH : ℝ) := (EJ / 2) = MJ ∧ M ∈ GH

theorem segment_lengths (EJ: ℝ) (EF GH M J: ℝ) (h₁ : EF ⊥ GH) 
(h₂ : radius = 10) (h₃ : EJ = 12 ∧ (MJ = 6)):
∃ (GM MH: ℝ), GM = 2 ∧ MH = 18 :=
by
  sorry

end segment_lengths_l816_816360


namespace g_not_even_or_odd_l816_816741

def g (x : ℝ) : ℝ := ⌊x⌋ - x + 1

theorem g_not_even_or_odd : ¬(∀ x, g x = g (-x)) ∧ ¬(∀ x, g (-x) = -g x) := by
  intro h
  sorry

end g_not_even_or_odd_l816_816741


namespace final_result_l816_816351

def P (x : ℝ) : ℝ := 3 * Real.sqrt x
def Q (x : ℝ) : ℝ := x^2 + 1

theorem final_result : P (Q (P (Q (P (Q 4))))) = 3 * Real.sqrt 1387 := by
  sorry

end final_result_l816_816351


namespace cos_315_is_sqrt2_div_2_l816_816521

noncomputable def cos_315_eq : Prop := 
  real.cos (315 * real.pi / 180) = real.sqrt 2 / 2

theorem cos_315_is_sqrt2_div_2 : cos_315_eq := by 
  sorry

end cos_315_is_sqrt2_div_2_l816_816521


namespace perfect_square_trinomial_l816_816864

theorem perfect_square_trinomial (a b c : ℤ) (f : ℤ → ℤ) (h : ∀ x : ℤ, f x = a * x^2 + b * x + c) :
  ∃ d e : ℤ, ∀ x : ℤ, f x = (d * x + e) ^ 2 :=
sorry

end perfect_square_trinomial_l816_816864


namespace fraction_A_to_BC_l816_816463

noncomputable def total_amount : ℝ := 1080
noncomputable def A_amt : ℝ := ?
noncomputable def B_amt : ℝ := A_amt - 30
noncomputable def C_amt : ℝ := 2 / 7 * (A_amt + C_amt) - A_amt

theorem fraction_A_to_BC (T A_amt B_amt C_amt : ℝ) (x : ℝ) :
  T = 1080 ->
  A_amt = x * (B_amt + C_amt) ->
  B_amt = (2 / 7) * (A_amt + C_amt) ->
  A_amt = B_amt + 30 ->
  A_amt + B_amt + C_amt = T ->
  x = 1 / 3 :=
by
  intros hT hA hB hC hSum
  -- you can proceed to solving here, but for now we leave a placeholder
  sorry

end fraction_A_to_BC_l816_816463


namespace chocolate_distribution_l816_816342

theorem chocolate_distribution
  (total_chocolate : ℚ)
  (num_piles : ℕ)
  (piles_given_to_shaina : ℕ)
  (weight_each_pile : ℚ)
  (weight_of_shaina_piles : ℚ)
  (h1 : total_chocolate = 72 / 7)
  (h2 : num_piles = 6)
  (h3 : piles_given_to_shaina = 2)
  (h4 : weight_each_pile = total_chocolate / num_piles)
  (h5 : weight_of_shaina_piles = piles_given_to_shaina * weight_each_pile) :
  weight_of_shaina_piles = 24 / 7 := by
  sorry

end chocolate_distribution_l816_816342


namespace sufficient_but_not_necessary_condition_l816_816982

theorem sufficient_but_not_necessary_condition (θ : ℝ) (h : |θ - π / 12| < π / 12) :
  θ ∈ ℝ ∧ |\theta - π / 12| < π / 12 → sin θ < 1 / 2 :=
sorry

end sufficient_but_not_necessary_condition_l816_816982


namespace best_fit_log2_l816_816312

noncomputable def f1 (x : ℝ) : ℝ := 2 ^ x
noncomputable def f2 (x : ℝ) : ℝ := x ^ 2 - 1
noncomputable def f3 (x : ℝ) : ℝ := 2 * x - 2
noncomputable def f4 (x : ℝ) : ℝ := Real.log2 x

def data_points : List (ℝ × ℝ) := [(0.50, -0.99), (2.01, 0.98)]

def is_best_fit (f : ℝ → ℝ) (points : List (ℝ × ℝ)) : Prop := 
  ∀ p ∈ points, abs (f p.1 - p.2) < abs ((λ (x : ℝ), 2 ^ x) p.1 - p.2) ∧
                abs (f p.1 - p.2) < abs ((λ (x : ℝ), x ^ 2 - 1) p.1 - p.2) ∧
                abs (f p.1 - p.2) < abs ((λ (x : ℝ), 2 * x - 2) p.1 - p.2)

theorem best_fit_log2 : is_best_fit (λ x, Real.log2 x) data_points := sorry

end best_fit_log2_l816_816312


namespace cos_315_eq_sqrt2_div_2_l816_816570

theorem cos_315_eq_sqrt2_div_2 : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
sorry

end cos_315_eq_sqrt2_div_2_l816_816570


namespace max_value_of_omega_increasing_l816_816821

theorem max_value_of_omega_increasing :
  ∀ (f g : ℝ → ℝ) (ω : ℝ), (ω > 0) →
  (∀ x, f x = 2 * real.sin (ω * x + (real.pi / 4))) →
  (∀ x, g x = 2 * real.sin (ω * (x - (real.pi / (4 * ω)))) → g x) →
  (∀ x, ∀ a b : ℝ, (∀ y, a ≤ y → y ≤ b → g y ≤ g (y + real.pi)) → (a = -real.pi / 6) → (b = real.pi / 3) →
  ∃ ω_max : ℝ, ω_max = (3 / 2)) :=
begin
  intros f g ω hω hf hg ha hb,
  rw ← hf at hg,
  sorry
end

end max_value_of_omega_increasing_l816_816821


namespace operation_B_correct_operation_C_correct_l816_816878

theorem operation_B_correct (x y : ℝ) : (-3 * x * y) ^ 2 = 9 * x ^ 2 * y ^ 2 :=
  sorry

theorem operation_C_correct (x y : ℝ) (h : x ≠ y) : 
  (x - y) / (2 * x * y - x ^ 2 - y ^ 2) = 1 / (y - x) :=
  sorry

end operation_B_correct_operation_C_correct_l816_816878


namespace monotonic_interval_y_l816_816852

def monotone_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y, x < y ∧ x ∈ I ∧ y ∈ I → f x < f y

theorem monotonic_interval_y (x : ℝ) : monotone_increasing (λ x, 2*x + sin x) set.univ :=
sorry

end monotonic_interval_y_l816_816852


namespace distinct_ordered_pairs_50_l816_816176

theorem distinct_ordered_pairs_50 (a b : ℕ) (h1 : a + b = 50) (h2 : 0 < a) (h3 : 0 < b) : 
    ({p : ℕ × ℕ | p.1 + p.2 = 50 ∧ 0 < p.1 ∧ 0 < p.2}.to_list.length = 49) :=
sorry

end distinct_ordered_pairs_50_l816_816176


namespace circle_area_pqr_l816_816002

variable (P Q R M : Type) [MetricSpace P] (PQ PR QR PM : ℝ)
variable (PQR_right : ∃ (PQ PR QR : ℝ), PQ = 8 ∧ PR = 6 ∧ QR = Real.sqrt (PQ ^ 2 + PR ^ 2))
variable (M_midpoint : ∃ (M : P), ∀ (a b : P), M = (a + b) / 2)

theorem circle_area_pqr : 
  ∃ (A : ℝ), A = 25 * Real.pi := by
  sorry

end circle_area_pqr_l816_816002


namespace sin_minus_cos_l816_816215

variable (θ : ℝ)
hypothesis (h1 : 0 < θ ∧ θ < π / 2)
hypothesis (h2 : Real.tan θ = 1 / 3)

theorem sin_minus_cos (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := 
by 
  sorry

end sin_minus_cos_l816_816215


namespace prove_inequality_l816_816161

-- The conditions of the problem are given as follows:
variable {f : ℝ → ℝ} (h_diff : ∀ x ∈ Ioo 0 (π / 2), DifferentiableAt ℝ f x)
variable (h_ineq : ∀ x ∈ Ioo 0 (π / 2), (f' x / tan x) < f x)

-- The theorem statement:
theorem prove_inequality (h : ∀ x ∈ Ioo 0 (π / 2), (f' x / tan x) < f x) :
  f (π / 3) < sqrt 3 * f (π / 6) := 
sorry

end prove_inequality_l816_816161


namespace border_pieces_count_l816_816483

def pieces_per_side : Nat := 12

def number_of_border_pieces (n : Nat) : Nat :=
  let top := n
  let bottom := n
  let left := n - 2
  let right := n - 2
  top + bottom + left + right

theorem border_pieces_count : number_of_border_pieces pieces_per_side = 44 := 
by 
  simp [number_of_border_pieces, pieces_per_side]
  sorry

end border_pieces_count_l816_816483


namespace systematic_sampling_l816_816835

theorem systematic_sampling :
  ∃ a, a = [3, 13, 23, 33, 43, 53] ∧ (∀ (i : ℕ), i ∈ List.range 6 → a.nth i = some (3 + 10 * i)) :=
by
  sorry

end systematic_sampling_l816_816835


namespace problem_solution_l816_816356

theorem problem_solution
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + 2) * (b + 2) ≥ c * d :=
sorry

end problem_solution_l816_816356


namespace distinct_ordered_pairs_eq_49_l816_816172

open Nat

theorem distinct_ordered_pairs_eq_49 (a b : ℕ) (h1 : a + b = 50) (h2 : a > 0) (h3 : b > 0) :
  num_solutions (λ p : ℕ × ℕ, p.1 + p.2 = 50 ∧ p.1 > 0 ∧ p.2 > 0) = 49 :=
sorry

end distinct_ordered_pairs_eq_49_l816_816172


namespace unique_sequence_l816_816599

/-- Define an infinite sequence of positive real numbers -/
def infinite_sequence (X : ℕ → ℝ) : Prop :=
  ∀ n, 0 < X n

/-- Define the recurrence relation for the sequence -/
def recurrence_relation (X : ℕ → ℝ) : Prop :=
  ∀ n, X (n + 2) = (1 / 2) * (1 / X (n + 1) + X n)

/-- Prove that the only infinite sequence satisfying the recurrence relation is the constant sequence 1 -/
theorem unique_sequence (X : ℕ → ℝ) (h_seq : infinite_sequence X) (h_recur : recurrence_relation X) :
  ∀ n, X n = 1 :=
by
  sorry

end unique_sequence_l816_816599


namespace final_answer_l816_816843

noncomputable def f (x : ℝ) : ℝ := sorry

axiom even_fun : ∀ x : ℝ, f (-x) = f x
axiom odd_fun : ∀ x : ℝ, f (x - 1) = -f (-x - 1)
axiom defined_on_0_1 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f x = Real.log 2024 (1 / x)

-- Prove the following:
theorem final_answer : f 2025 + f (-1 / 2024) = 1 :=
sorry

end final_answer_l816_816843


namespace wolf_share_calculation_l816_816836

def total_royalty : ℝ := sorry -- total royalty is not given explicitly
def remaining_royalty : ℝ := 2100 -- remaining royalty handed to the Wolf

-- define equal distribution shares for each book
def three_little_pigs_share_per_author : ℝ := total_royalty / 4
def three_little_pigs_remaining_after_nafnaf : ℝ := total_royalty * 3 / 4
def red_riding_hood_share_per_author : ℝ := total_royalty / 3

-- calculate Wolf’s share
def wolf_share_pigs : ℝ := three_little_pigs_remaining_after_nafnaf / 3
def wolf_share_hood : ℝ := red_riding_hood_share_per_author

-- Wolf's total share
def wolf_total_share : ℝ := wolf_share_pigs + wolf_share_hood

-- proof problem statement
theorem wolf_share_calculation : wolf_total_share = 700 := by
  sorry

example (total_royalty: ℝ) (remaining_royalty : ℝ) (three_little_pigs_share_per_author : ℝ) 
  (three_little_pigs_remaining_after_nafnaf : ℝ) (red_riding_hood_share_per_author : ℝ) 
  (wolf_share_pigs : ℝ) (wolf_share_hood : ℝ) (wolf_total_share : ℝ) : 
  wolf_total_share = 700 :=
begin
  sorry
end

end wolf_share_calculation_l816_816836


namespace sin_minus_cos_l816_816253

theorem sin_minus_cos (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < (Real.pi / 2)) (hθ3 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := by
  sorry

end sin_minus_cos_l816_816253


namespace find_AE_l816_816734

noncomputable def triangle_ABC (A B C : Type) :=
(Bertrand_Condition A B C 8 7 6)

def point_on_segment (B C : Type) (E : Type) :=
(segment_contains_point B C E)

def angle_relation (A B C E : Type) :=
(angle_multiplication (angle_subsegments B A E) 3 (angle_subsegments E A C))

theorem find_AE 
  (A B C E : Type)
  (hab : AB = 8) 
  (hbc : BC = 7) 
  (hca : CA = 6) 
  (he : point_on_segment BC E)
  (hae : angle_relation A B C E) :
  4 * AE^2 = 135 :=
by
  sorry

end find_AE_l816_816734


namespace g_range_g_undefined_at_one_g_no_other_values_l816_816600

noncomputable def g (x : ℝ) : ℝ :=
  Real.arctan x + Real.arctan ((1 + x) / (1 - x))

theorem g_range (x : ℝ) (hx : x < 1) : g x = π / 4 :=
  sorry

theorem g_undefined_at_one : ¬(∃ g, g (1 : ℝ) = g) :=
  sorry

theorem g_no_other_values (x : ℝ) (hx : x > 1) : ¬(∃ g, g x = (π / 4)) :=
  sorry

end g_range_g_undefined_at_one_g_no_other_values_l816_816600


namespace proof_problem_l816_816350

noncomputable def A : Set ℝ := { x | x^2 - 4 = 0 }
noncomputable def B : Set ℝ := { y | ∃ x, y = x^2 - 4 }

theorem proof_problem :
  (A ∩ B = A) ∧ (A ∪ B = B) :=
by {
  sorry
}

end proof_problem_l816_816350


namespace minimum_value_expression_l816_816760

theorem minimum_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    ∃ t, t = x + y ∧ x^2 + y^2 + 4 / (x + y)^2 = 2 * real.sqrt 2 :=
begin
  sorry
end

end minimum_value_expression_l816_816760


namespace problem_l816_816290

def f (a : ℕ) : ℕ := a + 3
def F (a b : ℕ) : ℕ := b^2 + a

theorem problem : F 4 (f 5) = 68 := by sorry

end problem_l816_816290


namespace min_focal_length_hyperbola_l816_816184

theorem min_focal_length_hyperbola (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b - c = 2) : 
  2*c ≥ 4 + 4 * Real.sqrt 2 := 
sorry

end min_focal_length_hyperbola_l816_816184


namespace angle_between_unit_vectors_l816_816658

variables {V : Type*} [inner_product_space ℝ V]

theorem angle_between_unit_vectors (a b : V) 
  (ha : ∥a∥ = 1) (hb : ∥b∥ = 1)
  (h : ⟪2 • a + b, a - 2 • b⟫ = -3 * real.sqrt 3 / 2) :
  real.angle (a) (b) = real.pi / 6 :=
begin
  sorry
end

end angle_between_unit_vectors_l816_816658


namespace find_b_l816_816711

variables (A B C : Type) -- these can be considered as abstract types representing the vertices of the triangle

structure Triangle where
  a b c : ℝ -- sides opposite to angles A, B, and C
  θ : ℝ -- angle in radians opposite side a
  area : ℝ -- Area of the triangle

def cos_component (tri : Triangle) : Prop :=
  tri.θ = real.arccos (1 / 3)

def side_a (tri : Triangle) : Prop :=
  tri.a = 2

def triangle_area (tri : Triangle) : Prop :=
  tri.area = real.sqrt 2

def value_of_b (tri : Triangle) : Prop :=
  tri.b = real.sqrt 3

theorem find_b (tri : Triangle) (h1 : cos_component tri) (h2 : side_a tri) (h3 : triangle_area tri) : value_of_b tri :=
  sorry

end find_b_l816_816711


namespace trajectory_of_Q_l816_816980

noncomputable def polar_chord_extension (rho theta : ℝ) : Prop :=
  -- Given condition: the circle equation in polar form
  let rho0 := 2 * Real.cos theta in
  -- Condition for extending OP to Q
  ∃ Q_rho, rho = 5 * (Real.cos theta) ∧ (2 / (rho - rho0)) = (2 / 3)
  

theorem trajectory_of_Q (rho theta : ℝ) :
  (polar_chord_extension rho theta) → 
  ρ = 5 * Real.cos(θ) ∧ (set.univ : set (ℝ × ℝ)) = {(x, y) | x^2 + y^2 = 25} :=
sorry

end trajectory_of_Q_l816_816980


namespace find_solutions_of_x4_minus_16_l816_816617

noncomputable def solution_set : Set Complex :=
  {2, -2, Complex.I * 2, -Complex.I * 2}

theorem find_solutions_of_x4_minus_16 :
  {x : Complex | x^4 - 16 = 0} = solution_set :=
by
  sorry

end find_solutions_of_x4_minus_16_l816_816617


namespace fraction_of_each_task_left_after_14_days_l816_816038

variables (A_rate B_rate C_rate : ℚ)
variables (first_task second_task : ℚ)

def work_rate_A := 1 / 15  -- work rate of A (task per day)
def work_rate_B := 1 / 20  -- work rate of B (task per day)
def work_rate_C := 1 / 25  -- work rate of C (task per day)

def combined_work_rate_AB := work_rate_A + work_rate_B
def combined_work_rate_ABC := work_rate_A + work_rate_B + work_rate_C

def work_done_AB_7_days := 7 * combined_work_rate_AB
def work_done_C_7_days := 7 * work_rate_C

def fraction_first_task_left_after_7_days := 1 - work_done_AB_7_days / first_task
def fraction_second_task_left_after_7_days := 1 - work_done_C_7_days / second_task

def total_work_done_by_ABC_in_7_days := 7 * combined_work_rate_ABC
def fraction_second_task_after_14_days := if total_work_done_by_ABC_in_7_days >= fraction_second_task_left_after_7_days
                                          then 0
                                          else fraction_second_task_left_after_7_days - total_work_done_by_ABC_in_7_days

theorem fraction_of_each_task_left_after_14_days :
  fraction_first_task_left_after_7_days = 11 / 60 ∧ fraction_second_task_after_14_days = 0 := by
  sorry

end fraction_of_each_task_left_after_14_days_l816_816038


namespace find_d_l816_816598

noncomputable def d : ℝ := 3.44

theorem find_d :
  (∃ x : ℝ, (3 * x^2 + 19 * x - 84 = 0) ∧ x = ⌊d⌋) ∧
  (∃ y : ℝ, (5 * y^2 - 26 * y + 12 = 0) ∧ y = d - ⌊d⌋) →
  d = 3.44 :=
by
  sorry

end find_d_l816_816598


namespace problem1_problem2_problem3_problem4_l816_816832

theorem problem1 (x : ℝ) : (|3*x - 5| + 4 = 8) → (x = 3 ∨ x = 1/3) :=
by
  sorry

theorem problem2 (x : ℝ) : (|4*x - 3| - 2 = 3*x + 4) → (x = 9 ∨ x = -3/7) :=
by 
  sorry

theorem problem3 (x : ℝ) : (|x - |2*x + 1|| = 3) → (x = 2 ∨ x = -4/3) :=
by 
  sorry

theorem problem4 (x : ℝ) : (|2*x - 1| + |x - 2| = |x + 1|) → (1/2 ≤ x ∧ x ≤ 2) :=
by 
  sorry

end problem1_problem2_problem3_problem4_l816_816832


namespace sally_cards_sum_l816_816817

noncomputable def red_cards : List ℕ := [1, 2, 3, 4, 5, 6]
noncomputable def blue_cards : List ℕ := [5, 6, 7, 8, 9]

def alternates (stack : List ℕ) : Prop :=
  ∀ i, i < stack.length - 1 → (stack.nth_le i sorry % 2 ≠ stack.nth_le (i + 1) sorry % 2)

def divides_neighbour (red blue : ℕ) : Prop :=
  red ∣ blue

def middle_three_sum (stack : List ℕ) : ℕ :=
  stack.nth_le (stack.length / 2 - 1) sorry +
  stack.nth_le (stack.length / 2) sorry +
  stack.nth_le (stack.length / 2 + 1) sorry

theorem sally_cards_sum :
  ∃ stack : List ℕ, 
  stack.head = 6 ∧
  alternates stack ∧
  (∀ i, (i < stack.length / 2 → divides_neighbour stack.nth_le (2 * i) sorry stack.nth_le (2 * i + 1) sorry) ∧
        (i > stack.length / 2 → divides_neighbour stack.nth_le (2 * i - 1) sorry stack.nth_le (2 * i) sorry)) ∧
  middle_three_sum stack = 19 :=
sorry

end sally_cards_sum_l816_816817


namespace sum_first_nine_arithmetic_sequence_l816_816723

theorem sum_first_nine_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ)
(h_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * d) 
(h_condition : 2 * a 3 + a 9 = 33) : 
(\sum i in finset.range 9, a i.succ) = 99 :=
sorry

end sum_first_nine_arithmetic_sequence_l816_816723


namespace find_angle_y_l816_816731

theorem find_angle_y (m n : Line) (A B C : Point) 
  (angle_A : ∠ A = 50) (angle_B : ∠ B = 90) (angle_C : ∠ C = 40) 
  (parallel_mn : m ∥ n) : 
  (y : ℝ) = 90 :=
by
  -- Formal proof would go here
  sorry

end find_angle_y_l816_816731


namespace distinct_domino_arrangements_l816_816365

-- Define the grid dimensions and the step requirements.
def grid_width : ℕ := 4
def grid_height : ℕ := 6
def right_steps : ℕ := 3
def down_steps : ℕ := 5
def total_steps : ℕ := right_steps + down_steps

-- The number of dominoes used to traverse the grid.
def domino_count : ℕ := 4

-- Helper function to compute the binomial coefficient.
def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

-- Main theorem statement.
theorem distinct_domino_arrangements :
  binomial (total_steps) (right_steps) = 56 :=
by
  -- Here we would include the proof, but the statement itself suffices for now.
  sorry

end distinct_domino_arrangements_l816_816365


namespace trigonometric_identity_l816_816273

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end trigonometric_identity_l816_816273


namespace distinct_ordered_pairs_50_l816_816175

theorem distinct_ordered_pairs_50 (a b : ℕ) (h1 : a + b = 50) (h2 : 0 < a) (h3 : 0 < b) : 
    ({p : ℕ × ℕ | p.1 + p.2 = 50 ∧ 0 < p.1 ∧ 0 < p.2}.to_list.length = 49) :=
sorry

end distinct_ordered_pairs_50_l816_816175


namespace speed_of_stream_l816_816028

theorem speed_of_stream (b s : ℝ) (h1 : 75 = 5 * (b + s)) (h2 : 45 = 5 * (b - s)) : s = 3 :=
by
  have eq1 : b + s = 15 := by linarith [h1]
  have eq2 : b - s = 9 := by linarith [h2]
  have b_val : b = 12 := by linarith [eq1, eq2]
  linarith 

end speed_of_stream_l816_816028


namespace cos_315_eq_sqrt2_div_2_l816_816531

theorem cos_315_eq_sqrt2_div_2 :
  let θ := 315.0
  in θ = 360.0 - 45.0 ∧ (cos 45 * (π / 180)) = (Real.sqrt 2 / 2) →
  (cos (θ * (π / 180))) = (Real.sqrt 2 / 2) :=
by
  intros θ
  -- Non-necessary proof omitted
  sorry

end cos_315_eq_sqrt2_div_2_l816_816531


namespace victor_weekly_earnings_l816_816005

def wage_per_hour : ℕ := 12
def hours_monday : ℕ := 5
def hours_tuesday : ℕ := 6
def hours_wednesday : ℕ := 7
def hours_thursday : ℕ := 4
def hours_friday : ℕ := 8

def earnings_monday := hours_monday * wage_per_hour
def earnings_tuesday := hours_tuesday * wage_per_hour
def earnings_wednesday := hours_wednesday * wage_per_hour
def earnings_thursday := hours_thursday * wage_per_hour
def earnings_friday := hours_friday * wage_per_hour

def total_earnings := earnings_monday + earnings_tuesday + earnings_wednesday + earnings_thursday + earnings_friday

theorem victor_weekly_earnings : total_earnings = 360 := by
  sorry

end victor_weekly_earnings_l816_816005


namespace min_k_plus_p_is_19199_l816_816948

noncomputable def find_min_k_plus_p : ℕ :=
  let D := 1007
  let domain_len := 1 / D
  let min_k : ℕ := 19  -- Minimum k value for which domain length condition holds, found via problem conditions
  let p_for_k (k : ℕ) : ℕ := (D * (k^2 - 1)) / k
  let k_plus_p (k : ℕ) : ℕ := k + p_for_k k
  k_plus_p min_k

theorem min_k_plus_p_is_19199 : find_min_k_plus_p = 19199 :=
  sorry

end min_k_plus_p_is_19199_l816_816948


namespace simplified_expression_l816_816019

theorem simplified_expression :
  (sqrt 5 * 5^(1/2) + 18 / 3 * 3 - 8^(3/2) + 10 - 2^3) = -7 :=
by
  sorry

end simplified_expression_l816_816019


namespace nocks_to_nicks_equivalence_l816_816687

-- Definitions based on conditions
def nicks := ℝ
def nacks := ℝ
def nocks := ℝ

axiom nicks_to_nacks : 5 * nicks = 3 * nacks
axiom nacks_to_nocks : 4 * nacks = 5 * nocks

-- Statement to prove the equivalence
theorem nocks_to_nicks_equivalence : ∀ (n : nocks), 40 * nocks = (160 / 3) * nicks :=
by
  -- skip proof
  sorry

end nocks_to_nicks_equivalence_l816_816687


namespace cos_315_eq_l816_816563

theorem cos_315_eq :
  let Q := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))
  in Q = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) →
     Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  intro hQ
  sorry

end cos_315_eq_l816_816563


namespace cos_315_eq_l816_816559

theorem cos_315_eq :
  let Q := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))
  in Q = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) →
     Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  intro hQ
  sorry

end cos_315_eq_l816_816559


namespace number_of_proper_subsets_of_A_l816_816707

-- Define the universal set A
def A : set ℕ := {0, 1, 2}

-- Define what it means to be a proper subset
def is_proper_subset (A B : set ℕ) : Prop := A ⊂ B

-- State that given the set A = {0, 1, 2}, the number of proper subsets is 7
theorem number_of_proper_subsets_of_A : finset.card (finset.filter (λ s, is_proper_subset s A) (finset.powerset A.to_finset)) = 7 := sorry

end number_of_proper_subsets_of_A_l816_816707


namespace ellipse_equation_l816_816457

theorem ellipse_equation (P : ℝ × ℝ) (d1 d2 : ℝ) 
  (h1 : d1 = (4 * sqrt 5) / 3) (h2 : d2 = (2 * sqrt 5) / 3)
  (h3 : ∃ f : ℝ × ℝ, f ∈ { (c, 0) | c ∈ ℝ } ∪ { (0, c) | c ∈ ℝ }):
  (∃ a b : ℝ, (a = sqrt 5) ∧ (b = sqrt (a^2 - (d1^2 - d2^2))) ∧ 
    ((P.1)^2 / a^2 + (P.2)^2 / b^2 = 1 ∨ (P.1)^2 / b^2 + (P.2)^2 / a^2 = 1)) :=
by
  sorry

end ellipse_equation_l816_816457


namespace imaginary_part_z1z2_l816_816696

open Complex

-- Define the complex numbers z1 and z2
def z1 : ℂ := (1 : ℂ) - I
def z2 : ℂ := (2 : ℂ) + 4 * I

-- Define the product of z1 and z2
def z1z2 : ℂ := z1 * z2

-- State the theorem that the imaginary part of z1z2 is 2
theorem imaginary_part_z1z2 : z1z2.im = 2 := by
  -- Proof steps would go here
  sorry

end imaginary_part_z1z2_l816_816696


namespace sin_double_angle_identity_l816_816142
noncomputable theory

open Real

theorem sin_double_angle_identity (α : ℝ) (h : sin (α + π/3) = 3/5) :
  sin (2 * α + π / 6) = -7 / 25 := sorry

end sin_double_angle_identity_l816_816142


namespace bryan_skittles_l816_816077

theorem bryan_skittles (ben_mms : ℕ) (ben_mms_eq_20 : ben_mms = 20) (bryan_skittles_more : ∀ ben_mms, bryan_skittles = ben_mms + 30) : bryan_skittles = 50 :=
by
  rw [ben_mms_eq_20]
  sorry

end bryan_skittles_l816_816077


namespace find_solutions_l816_816604

noncomputable def solutions_to_equation : set Complex :=
  {x | x^4 - 16 = 0}

theorem find_solutions : solutions_to_equation = {2, -2, Complex.I * 2, -Complex.I * 2} :=
sorry

end find_solutions_l816_816604


namespace minimal_rope_cost_l816_816386

theorem minimal_rope_cost :
  let pieces_needed := 10
  let length_per_piece := 6 -- inches
  let total_length_needed := pieces_needed * length_per_piece -- inches
  let one_foot_length := 12 -- inches
  let cost_six_foot_rope := 5 -- dollars
  let cost_one_foot_rope := 1.25 -- dollars
  let six_foot_length := 6 * one_foot_length -- inches
  let one_foot_total_cost := (total_length_needed / one_foot_length) * cost_one_foot_rope
  let six_foot_total_cost := cost_six_foot_rope
  total_length_needed <= six_foot_length ∧ six_foot_total_cost < one_foot_total_cost →
  six_foot_total_cost = 5 := sorry

end minimal_rope_cost_l816_816386


namespace base_prime_representation_540_l816_816573

/-- The base prime representation of 540 is 131 -/
theorem base_prime_representation_540 : base_prime_repr 540 = "131" := sorry

end base_prime_representation_540_l816_816573


namespace mean_is_greater_than_median_by_2_over_5_l816_816633

-- Definitions of the histogram data
def num_students : ℕ := 20
def histogram : list (ℕ × ℕ) :=
  [(0, 2), (1, 3), (2, 6), (3, 5), (4, 2), (5, 2)]

-- Mean number of days missed
def mean_days_missed : ℚ :=
  (2 * 0 + 3 * 1 + 6 * 2 + 5 * 3 + 2 * 4 + 2 * 5) / num_students

-- Median number of days missed
def median_days_missed : ℚ := 2

-- Prove that mean - median equals the expected difference
theorem mean_is_greater_than_median_by_2_over_5 :
  mean_days_missed - median_days_missed = 2 / 5 :=
by
  sorry

end mean_is_greater_than_median_by_2_over_5_l816_816633


namespace sin_minus_cos_l816_816254

theorem sin_minus_cos (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < (Real.pi / 2)) (hθ3 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := by
  sorry

end sin_minus_cos_l816_816254


namespace proof_problem_l816_816728

noncomputable def parametric_line : ℝ → ℝ × ℝ := λ t, (3 - (Real.sqrt 2 / 2 * t), Real.sqrt 5 + (Real.sqrt 2 / 2 * t))

noncomputable def polar_circle : ℝ → ℝ := λ θ, 2 * Real.sqrt 5 * Real.sin θ

def distance_from_center_to_line (l : ℝ → ℝ × ℝ) : ℝ :=
  let center := (0 : ℝ, Real.sqrt 5) in
  let line := l 0 in
  (abs (fst center + snd center - Real.sqrt 5 - 3)) / Real.sqrt 2

def sum_distances (l : ℝ → ℝ × ℝ) (P : ℝ × ℝ) : ℝ :=
  let intersection_roots : list ℝ := [/*calculation of roots*/] in
  intersection_roots.map (abs ∘ id).sum

theorem proof_problem (t : ℝ) :
  let P := (3 : ℝ, Real.sqrt 5) in
  let expected_distance := 3 * Real.sqrt 2 / 2 in
  let expected_sum := 3 * Real.sqrt 2 in

  distance_from_center_to_line parametric_line = expected_distance ∧
  sum_distances parametric_line P = expected_sum := by
  sorry

end proof_problem_l816_816728


namespace true_equiv_4_l816_816758

variables {Point : Type} [MetricSpace Point]

structure Line :=
(dir : Point)

structure Plane :=
(normal : Point)

def Line.parallel (l1 l2 : Line) := -- Placeholder definition
sorry

def Line.perpendicular (l1 l2 : Line) := -- Placeholder definition
sorry

def Plane.parallel (p1 p2 : Plane) := -- Placeholder definition
sorry

def Plane.perpendicular (p1 p2 : Plane) := -- Placeholder definition
sorry

def Line.in_plane (l : Line) (p : Plane) := -- Placeholder definition
sorry

axiom Line_dir_is_normal (l : Line) (p : Plane) : Line.perpendicular l (Plane.normal p) → False -- Placeholder axiom

theorem true_equiv_4 {a b : Line} {α β : Plane} :
  (Line.perpendicular a (Plane.normal α)) →
  (Line.perpendicular b (Plane.normal β)) →
  (Plane.perpendicular α β) →
  (Line.perpendicular a b) :=
  -- The proof does not need to be provided
  sorry

end true_equiv_4_l816_816758


namespace cone_height_l816_816900

theorem cone_height (V : ℝ) (h : ℝ) (r : ℝ) (vertex_angle : ℝ) 
  (H1 : V = 16384 * Real.pi)
  (H2 : vertex_angle = 90) 
  (H3 : V = (1 / 3) * Real.pi * r^2 * h)
  (H4 : h = r) : 
  h = 36.6 :=
by
  sorry

end cone_height_l816_816900


namespace sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816213

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816213


namespace solve_x4_minus_16_eq_0_l816_816611

open Complex  -- Open the complex number notation

theorem solve_x4_minus_16_eq_0 :
  {x : ℂ | x^4 = 16} = {2, -2, 2 * Complex.I, -2 * Complex.I} :=
by sorry

end solve_x4_minus_16_eq_0_l816_816611


namespace coordinates_in_second_quadrant_l816_816332

section 
variable (x y : ℝ)
variable (hx : x = -7)
variable (hy : y = 4)
variable (quadrant : x < 0 ∧ y > 0)
variable (distance_x : |y| = 4)
variable (distance_y : |x| = 7)

theorem coordinates_in_second_quadrant :
  (x, y) = (-7, 4) := by
  sorry
end

end coordinates_in_second_quadrant_l816_816332


namespace faye_team_size_l816_816957

theorem faye_team_size (total_points faye_points points_per_other_player : ℕ) (h1 : total_points = 68) (h2 : faye_points = 28) (h3 : points_per_other_player = 8) : 
  ((total_points - faye_points) / points_per_other_player) + 1 = 6 :=
by {
  rw [h1, h2, h3],
  have : (68 - 28) / 8 = 5 := by {
    norm_num
  },
  rw this,
  norm_num,
  sorry
}

end faye_team_size_l816_816957


namespace find_q_l816_816106

theorem find_q (q : ℝ) : 15^3 = 10^2 / 5 * 3^(12 * q) → q = 1 / 4 :=
by
  intros h
  sorry

end find_q_l816_816106


namespace cos_315_eq_sqrt2_div_2_l816_816525

theorem cos_315_eq_sqrt2_div_2 :
  let θ := 315.0
  in θ = 360.0 - 45.0 ∧ (cos 45 * (π / 180)) = (Real.sqrt 2 / 2) →
  (cos (θ * (π / 180))) = (Real.sqrt 2 / 2) :=
by
  intros θ
  -- Non-necessary proof omitted
  sorry

end cos_315_eq_sqrt2_div_2_l816_816525


namespace equilateral_triangles_count_l816_816847

open Real

noncomputable def count_equilateral_triangles : ℕ :=
sorry

theorem equilateral_triangles_count :
  let lines := [λ k, λ x, k : ℝ → ℝ, λ k, λ x, sqrt 3 * x + k : ℝ → ℝ, λ k, λ x, -sqrt 3 * x + k : ℝ → ℝ] in
  let ks := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] in
  count_equilateral_triangles = 160 :=
sorry

end equilateral_triangles_count_l816_816847


namespace construct_triangle_l816_816943

variables {R : Type*} [Field R]

-- Define points A = (x1, y1) and B = (x3, y3)
structure Point :=
(x : R) 
(y : R)

-- Define the reflection of point A over a given line to obtain point A'
def reflect_over_line (A : Point) (a b c : R) : Point :=
let x' := A.x - (2 * a * (a * A.x + b * A.y + c)) / (a^2 + b^2) in
let y' := A.y - (2 * b * (a * A.x + b * A.y + c)) / (a^2 + b^2) in
⟨x', y'⟩

-- Define the line through two points
def line_through_points (P Q : Point) : R × R × R :=
let a := Q.y - P.y in
let b := P.x - Q.x in
let c := P.y * Q.x - P.x * Q.y in
(a, b, c)

-- Define the intersection of two lines
def line_intersection (l1 l2 : R × R × R) : Point :=
let ⟨a1, b1, c1⟩ := l1 in
let ⟨a2, b2, c2⟩ := l2 in
let determinant := a1 * b2 - a2 * b1 in
let x := (b1 * c2 - b2 * c1) / determinant in
let y := (a2 * c1 - a1 * c2) / determinant in
⟨x, y⟩

-- Define the conditions and main statement that proves question == answer given conditions
theorem construct_triangle
  (A B : Point)
  (a b c : R) :
  let A' := reflect_over_line A a b c in
  let line_AB' := line_through_points A' B in
  let bisector_line := (a, b, c) in
  let C := line_intersection line_AB' bisector_line in
  true :=
begin
  sorry
end

end construct_triangle_l816_816943


namespace error_percentage_is_95_l816_816914

theorem error_percentage_is_95 
  (y : ℝ) (h : y > 0) : 
  let correct_result := 5 * y,
      mistaken_result := y / 4,
      error := |correct_result - mistaken_result|,
      error_percentage := (error / correct_result) * 100
  in
  error_percentage = 95 :=
sorry

end error_percentage_is_95_l816_816914


namespace projections_on_circle_l816_816643

variables {n : ℕ} (P : fin n → ℝ × ℝ × ℝ) (A : ℝ × ℝ × ℝ) (H : ℝ × ℝ × ℝ)
variables (HP : ∀ i : fin n, P i ∈ sphere (0, 0, 0) 1)
variables (H_HA : orthogonal_projection H (plane (P 0, P 1)) = A)
variables (H_Hi : ∀ i : fin n, orthogonal_projection H (line_through A (P i)) = H)

theorem projections_on_circle (H_i : fin n → ℝ × ℝ × ℝ) :
  (∀ i : fin n, H_i i = orthogonal_projection H (line_through A (P i))) →
  ∃ (C : ℝ × ℝ × ℝ) (r : ℝ), ∀ i : fin n, dist C (H_i i) = r :=
by
  -- proof goes here
  sorry

end projections_on_circle_l816_816643


namespace max_M_inequality_l816_816642

theorem max_M_inequality (n : ℕ) (hpos : n > 0) : 
  ∃ M, (∀ (a : fin n -> ℕ),
     let sqrt_floor := λ i : fin n, int.floor (sqrt (a i))
     in ∑ i, sqrt_floor i 
        ≥ int.floor (sqrt (∑ i, a i + finset.min' (finset.univ.map a))) 
  ) ∧ M = int.floor (n * (n - 1) / 3) :=
sorry

end max_M_inequality_l816_816642


namespace solve_x4_minus_16_eq_0_l816_816609

open Complex  -- Open the complex number notation

theorem solve_x4_minus_16_eq_0 :
  {x : ℂ | x^4 = 16} = {2, -2, 2 * Complex.I, -2 * Complex.I} :=
by sorry

end solve_x4_minus_16_eq_0_l816_816609


namespace possible_values_of_x_l816_816157

theorem possible_values_of_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 10) (h2 : y + 1 / x = 5 / 12) : x = 4 ∨ x = 6 :=
by
  sorry

end possible_values_of_x_l816_816157


namespace convert_binary_and_subtract_l816_816580

theorem convert_binary_and_subtract :
  (let binary := "101101";
       decimal := (1 * 2^0 + 0 * 2^1 + 1 * 2^2 + 1 * 2^3 + 0 * 2^4 + 1 * 2^5) in
     decimal - 5 = 40) :=
by
  sorry

end convert_binary_and_subtract_l816_816580


namespace one_cow_one_bag_days_l816_816800

-- Definitions based on conditions in a)
def cows : ℕ := 60
def bags : ℕ := 75
def days_total : ℕ := 45

-- Main statement for the proof problem
theorem one_cow_one_bag_days : 
  (cows : ℝ) * (bags : ℝ) / (days_total : ℝ) = 1 / 36 := 
by
  sorry   -- Proof placeholder

end one_cow_one_bag_days_l816_816800


namespace ice_cream_sandwiches_each_l816_816004

theorem ice_cream_sandwiches_each (total_ice_cream_sandwiches : ℕ) (number_of_nieces : ℕ) 
  (h1 : total_ice_cream_sandwiches = 143) (h2 : number_of_nieces = 11) : 
  total_ice_cream_sandwiches / number_of_nieces = 13 :=
by
  sorry

end ice_cream_sandwiches_each_l816_816004


namespace find_ordered_pair_l816_816362

variables {A B Q : Type} -- Points A, B, Q
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup Q]
variables {a b q : A} -- Vectors at points A, B, Q
variables (r : ℝ) -- Ratio constant

-- Define the conditions from the original problem
def ratio_aq_qb (A B Q : Type) [AddCommGroup A] [AddCommGroup B] [AddCommGroup Q] (a b q : A) (r : ℝ) :=
  r = 7 / 2

-- Define the goal theorem using the conditions above
theorem find_ordered_pair (h : ratio_aq_qb A B Q a b q r) : 
  q = (7 / 9) • a + (2 / 9) • b :=
sorry

end find_ordered_pair_l816_816362


namespace breakfast_total_correct_l816_816583

-- Define the costs of individual items
def cost_of_toast := 1
def cost_of_egg := 3
def cost_of_coffee := 2
def cost_of_juice := 1.5
def special_offer_cost := 3.5

-- Number of items consumed by each person
def dale_toast := 2
def dale_eggs := 2
def dale_coffee := 1

def andrew_toast := 1
def andrew_eggs := 2
def andrew_juice := 1

def melanie_toast := 3
def melanie_eggs := 1
def melanie_juice := 2

def kevin_toast := 4
def kevin_eggs := 3
def kevin_coffee := 2

-- Service charge percentage
def service_charge := 0.10

-- Define a function to calculate the total cost
noncomputable def total_breakfast_cost : ℝ :=
  let dale_cost := (dale_toast * cost_of_toast) + (dale_eggs * cost_of_egg) + (dale_coffee * cost_of_coffee)
  let andrew_cost := (andrew_toast * cost_of_toast) + (andrew_eggs * cost_of_egg) + (andrew_juice * cost_of_juice)
  let melanie_cost := (melanie_toast * cost_of_toast) + (melanie_eggs * cost_of_egg) + (melanie_juice * cost_of_juice)
  let kevin_cost := (kevin_toast * cost_of_toast) + (kevin_eggs * cost_of_egg) + special_offer_cost
  let subtotal := dale_cost + andrew_cost + melanie_cost + kevin_cost
  let total := subtotal + (subtotal * service_charge)
  total

-- Define the theorem to prove the total breakfast cost
theorem breakfast_total_correct : total_breakfast_cost = 48.40 := by
  sorry

end breakfast_total_correct_l816_816583


namespace distance_focus_to_asymptote_l816_816627

def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

def a : ℝ := 4
def b : ℝ := 3
def c : ℝ := 5
def asymptote (x y : ℝ) : Prop :=
  3 * x + 4 * y = 0

def focus : ℝ × ℝ := (5, 0)

theorem distance_focus_to_asymptote : 
  let (x1, y1) := focus in
  let (A, B, C) := (3, 4, 0) in
  hyperbola_equation 5 0 -> 
  (sorry : ℝ) -- sorry represents where you would typically place the proof
    = (abs (A * x1 + B * y1 + C) / sqrt (A^2 + B^2)) := by
      sorry

end distance_focus_to_asymptote_l816_816627


namespace sine_symmetry_axis_l816_816846

theorem sine_symmetry_axis : ∃ x, ∀ y,  y = sin x → x = π / 2 :=
by
  sorry

end sine_symmetry_axis_l816_816846


namespace number_of_crystals_in_container_l816_816369

-- Define the dimensions of the energy crystal
def length_crystal := 30
def width_crystal := 25
def height_crystal := 5

-- Define the dimensions of the cubic container
def side_container := 27

-- Volume of the cubic container
def volume_container := side_container ^ 3

-- Volume of the energy crystal
def volume_crystal := length_crystal * width_crystal * height_crystal

-- Proof statement
theorem number_of_crystals_in_container :
  volume_container / volume_crystal ≥ 5 :=
sorry

end number_of_crystals_in_container_l816_816369


namespace prob_multiple_42_inscribed_ball_is_zero_l816_816098

theorem prob_multiple_42_inscribed_ball_is_zero :
  let factors_of_200 := { n | n ∣ 200 }
  ∃ (A : set ℕ), A = { n | n ≤ 200 ∧ 42 ∣ n } ∧ (A ∩ factors_of_200 = ∅) →
  let total_balls := factors_of_200.card
  total_balls > 0 →
  0 = 0 :=
begin
  sorry
end

end prob_multiple_42_inscribed_ball_is_zero_l816_816098


namespace total_perimeter_of_triangle_and_circle_l816_816478

def right_triangle (a b c: ℝ) : Prop :=
  a^2 + b^2 = c^2

def inscribed_circle_radius (a b c r: ℝ) : Prop :=
  let s := (a + b + c) / 2 in
  (s * r = (1/2) * a * b) ∧ (s = (a + b + c) / 2)

theorem total_perimeter_of_triangle_and_circle :
  ∀ (a b c r: ℝ), 
  right_triangle a b c → 
  inscribed_circle_radius a b c r → 
  a = 3 → 
  b = 4 → 
  r = 1 → 
  a + b + c + 2 * real.pi * r = 12 + 2 * real.pi :=
by
  sorry

end total_perimeter_of_triangle_and_circle_l816_816478


namespace train_cross_pole_time_approx_l816_816490

-- Define the conditions
def speed_kmh := 30 -- Speed in km/hr
def length_m := 75  -- Length of the train in meters

-- Convert speed from km/hr to m/s
def speed_ms := speed_kmh * (1000.0 / 3600.0)

-- Stating the problem in Lean
theorem train_cross_pole_time_approx :
  let time := length_m / speed_ms in
  time ≈ 9.00 := 
by
  have h_speed_ms : speed_ms = speed_kmh * (1000.0 / 3600.0) := by rfl
  have h_length_m : length_m = 75 := by rfl
  have h_time : time = length_m / speed_ms := by rfl
  sorry

end train_cross_pole_time_approx_l816_816490


namespace range_s_l816_816086

def s (n : ℕ) : ℕ :=
  if n = 1 then 0
  else primeDivisors n |>.sum

theorem range_s : {y | ∃ n : ℕ, n > 1 ∧ s n = y} = {y | y ≥ 2} :=
by
  sorry

end range_s_l816_816086


namespace proof_case_a_proof_case_b_l816_816813

noncomputable def proof_problem_a (x y z p q : ℝ) (n : ℕ) 
  (h1 : y = x^n + p*x + q) 
  (h2 : z = y^n + p*y + q) 
  (h3 : x = z^n + p*z + q) : Prop :=
  x^2 * y + y^2 * z + z^2 * x >= x^2 * z + y^2 * x + z^2 * y

theorem proof_case_a (x y z p q : ℝ) 
  (h1 : y = x^2 + p*x + q) 
  (h2 : z = y^2 + p*y + q) 
  (h3 : x = z^2 + p*z + q) : 
  proof_problem_a x y z p q 2 h1 h2 h3 := 
sorry

theorem proof_case_b (x y z p q : ℝ) 
  (h1 : y = x^2010 + p*x + q) 
  (h2 : z = y^2010 + p*y + q) 
  (h3 : x = z^2010 + p*z + q) : 
  proof_problem_a x y z p q 2010 h1 h2 h3 := 
sorry

end proof_case_a_proof_case_b_l816_816813


namespace find_monday_temperature_l816_816390

theorem find_monday_temperature
  (M T W Th F : ℤ)
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : F = 35) :
  M = 43 :=
by
  sorry

end find_monday_temperature_l816_816390


namespace secretary_longest_time_l816_816837

theorem secretary_longest_time (h_ratio : ∃ x : ℕ, ∃ y : ℕ, ∃ z : ℕ, y = 2 * x ∧ z = 3 * x ∧ (5 * x = 40)) :
  5 * x = 40 := sorry

end secretary_longest_time_l816_816837


namespace inconsistent_assignment_l816_816492

theorem inconsistent_assignment (A : ℝ) (h1 : A = 15) (h2 : A = -A + 5) : false := by
  have h3 : 15 = -15 + 5 := by
    rw [←h1]
    exact h2
  rw [neg_add_eq_sub] at h3
  linarith
  sorry

end inconsistent_assignment_l816_816492


namespace cos_315_eq_sqrt2_div_2_l816_816564

theorem cos_315_eq_sqrt2_div_2 : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
sorry

end cos_315_eq_sqrt2_div_2_l816_816564


namespace smallest_five_digit_divisible_by_smallest_odd_primes_l816_816127

def smallest_odd_primes : List ℕ := [3, 5, 7]

def lcm_list (lst : List ℕ) : ℕ :=
  lst.foldr LCM 1

def smallest_five_digit_multiple (n : ℕ) : ℕ :=
  let lower_bound := 10000
  let m := (lower_bound + n - 1) / n
  n * m

theorem smallest_five_digit_divisible_by_smallest_odd_primes :
  smallest_five_digit_multiple (lcm_list smallest_odd_primes) = 10080 :=
by
  sorry

end smallest_five_digit_divisible_by_smallest_odd_primes_l816_816127


namespace solution_correct_l816_816185

-- Define the given conditions
def line1 (x : ℝ) (b : ℝ) : ℝ := -3 * x + b
def line2 (x : ℝ) (k : ℝ) : ℝ := -k * x + 1

-- Define the intersection point
def intersection_point : (ℝ × ℝ) := (1, -2)

-- Define the system of equations
def system_of_equations (x y b k : ℝ) : Prop :=
  (3 * x + y = b) ∧ (k * x + y = 1)

-- Prove that the intersection point satisfies the system of equations given the b and k values from the intersection condition
theorem solution_correct :
  ∃ b k, system_of_equations 1 (-2) b k :=
by
  use 1, 3
  split
  sorry

end solution_correct_l816_816185


namespace min_moves_to_sort_l816_816348

def num_moves_to_sort_chips (k : ℕ) (h : k ≥ 1) : ℕ := sorry

theorem min_moves_to_sort (k : ℕ) (hk : k ≥ 1) : 
  ∀ (initial_sequence : list ℕ), 
    (initial_sequence.length = 4 * k) ∧ 
    (list.count 1 initial_sequence = 2 * k) ∧ 
    (list.count 0 initial_sequence = 2 * k) → 
    (num_moves_to_sort_chips k hk) <= k :=
sorry

end min_moves_to_sort_l816_816348


namespace solve_x4_minus_16_eq_0_l816_816608

open Complex  -- Open the complex number notation

theorem solve_x4_minus_16_eq_0 :
  {x : ℂ | x^4 = 16} = {2, -2, 2 * Complex.I, -2 * Complex.I} :=
by sorry

end solve_x4_minus_16_eq_0_l816_816608


namespace impossible_to_place_consecutive_numbers_l816_816934

theorem impossible_to_place_consecutive_numbers :
  ∀ (a : ℕ), ¬ (∃ (f : Fin 24 → ℕ), (∀ i : Fin 24, f i = a + i) ∧ (∀ (squares : list (Fin 24 → Fin 4)) (o : Fin 4), (∑ p in squares[o], f p) = (∑ p in squares[(o + 1) % 4], f p))) :=
by
  sorry 

end impossible_to_place_consecutive_numbers_l816_816934


namespace no_solutions_exist_l816_816123

theorem no_solutions_exist : ¬ ∃ (x y z : ℝ), x + y = 3 ∧ xy - z^2 = 2 :=
by sorry

end no_solutions_exist_l816_816123


namespace single_real_solution_eq_l816_816135

theorem single_real_solution_eq (k : ℝ) :
  (∀ x : ℝ, (3*x + 5)*(x - 6) = -34 + k*x → (∃! x : ℝ, x)) ↔ (k = -13 + 4*real.sqrt 3 ∨ k = -13 - 4*real.sqrt 3) :=
sorry

end single_real_solution_eq_l816_816135


namespace cos_315_deg_l816_816541

noncomputable def cos_315 : ℝ :=
  Real.cos (315 * Real.pi / 180)

theorem cos_315_deg : cos_315 = Real.sqrt 2 / 2 :=
by
  sorry

end cos_315_deg_l816_816541


namespace sin_minus_cos_theta_l816_816248

theorem sin_minus_cos_theta (θ : ℝ) (h₀ : 0 < θ ∧ θ < (π / 2)) 
  (h₁ : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -√10 / 5 :=
by
  sorry

end sin_minus_cos_theta_l816_816248


namespace enclosure_largest_side_l816_816781

theorem enclosure_largest_side (l w : ℕ) (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 3600) : l = 60 :=
by
  sorry

end enclosure_largest_side_l816_816781


namespace probability_even_product_l816_816100

theorem probability_even_product:
  (∀ (box1 box2 : set ℕ), 
    (box1 = {1, 2, 3} ∧ box2 = {1, 2, 3}) → 
    ∃ (outcomes : finset (ℕ × ℕ)), 
      (outcomes = 
        { (1,1), (1,2), (1,3), 
          (2,1), (2,2), (2,3), 
          (3,1), (3,2), (3,3) }) ∧
      (∃ (favorable_outcomes : finset (ℕ × ℕ)),
        (favorable_outcomes = 
          { (2,1), (2,2), (1,2), 
            (3,2), (2,3) }) ∧
          (favorable_outcomes.card / outcomes.card : ℚ) = 5 / 9)) :=
begin
  sorry
end

end probability_even_product_l816_816100


namespace num_ordered_pairs_l816_816169

theorem num_ordered_pairs :
  ∃ n : ℕ, n = 49 ∧ ∀ (a b : ℕ), a + b = 50 → 0 < a ∧ 0 < b → (1 ≤ a ∧ a < 50) :=
by
  sorry

end num_ordered_pairs_l816_816169


namespace acute_angle_probability_correct_l816_816349

noncomputable def acute_angle_probability (n : ℕ) (n_ge_4 : n ≥ 4) : ℝ :=
  (n * (n - 2)) / (2 ^ (n-1))

theorem acute_angle_probability_correct (n : ℕ) (h : n ≥ 4) (P : Fin n → ℝ) -- P represents points on the circle
    (uniformly_distributed : ∀ i, P i ∈ Set.Icc (0 : ℝ) 1) : 
    acute_angle_probability n h = (n * (n - 2)) / (2 ^ (n-1)) := 
  sorry

end acute_angle_probability_correct_l816_816349


namespace num_subsets_with_property_P_is_133_l816_816764

-- Define the set S
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define the property P
def has_property_P (T : Finset ℕ) : Prop :=
  2 ≤ T.card ∧ (∀ x y ∈ T, x ≠ y → |x - y| > 1)

-- Define the set of all subsets of S with property P
def subsets_with_property_P : Finset (Finset ℕ) :=
  (Finset.powerset S).filter has_property_P

-- Define the number of such subsets
def num_subsets_with_property_P : ℕ :=
  subsets_with_property_P.card

-- Problem statement: Prove that the number of such subsets equals 133
theorem num_subsets_with_property_P_is_133 : num_subsets_with_property_P = 133 :=
  sorry

end num_subsets_with_property_P_is_133_l816_816764


namespace fraction_to_terminating_decimal_l816_816104

theorem fraction_to_terminating_decimal : (21 : ℚ) / 40 = 0.525 := 
by
  sorry

end fraction_to_terminating_decimal_l816_816104


namespace sin_minus_cos_theta_l816_816244

theorem sin_minus_cos_theta (θ : ℝ) (h₀ : 0 < θ ∧ θ < (π / 2)) 
  (h₁ : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -√10 / 5 :=
by
  sorry

end sin_minus_cos_theta_l816_816244


namespace tangent_line_correct_l816_816114

noncomputable def f (x : ℝ) : ℝ := x^2 * (x - 2) + 1

-- The point of tangency
def tangent_point : ℝ × ℝ := (1, f 1)

-- The derivative of the function
noncomputable def f_prime (x : ℝ) : ℝ := 3 * x^2 - 4 * x

-- The slope of the tangent line at x = 1
noncomputable def slope_at_one : ℝ := f_prime 1

-- The equation of the tangent line at the point
def tangent_line_equation (x y : ℝ) : Prop := x - y - 1 = 0

-- The main theorem: Proving that the tangent line equation at the specified point is as given
theorem tangent_line_correct : tangent_line_equation (1 : ℝ) ((slope_at_one * ((1 : ℝ) - 1)) + f 1) :=
by
  -- We use the function definitions and compute the correct values to show that the tangent line equation is correct
  sorry

end tangent_line_correct_l816_816114


namespace travel_time_l816_816475

theorem travel_time (speed distance : ℕ) (h_speed : speed = 100) (h_distance : distance = 500) :
  distance / speed = 5 := by
  sorry

end travel_time_l816_816475


namespace total_rice_in_pounds_l816_816815

-- Define the given conditions
def num_containers : ℕ := 4
def rice_per_container_ounces : ℕ := 70
def ounces_per_pound : ℕ := 16

-- The statement we need to prove
theorem total_rice_in_pounds :
  (num_containers * rice_per_container_ounces) / ounces_per_pound = 17.5 := 
  by
    sorry

end total_rice_in_pounds_l816_816815


namespace ellipse_major_axis_length_l816_816651

theorem ellipse_major_axis_length 
  (F1 : ℝ × ℝ) (F2 : ℝ × ℝ) 
  (hF1 : F1 = (1, 1)) (hF2 : F2 = (5, 2)) 
  (tangent_x_axis : ∀ x, F1.snd = 0 ∨ F2.snd = 0):
  let c := Real.sqrt ((F2.1 - F1.1)^2 + (F2.2 - F1.2)^2) / 2 in
  let b := F1.2 in
  let a := Real.sqrt (b^2 + c^2) in
    2 * a = Real.sqrt 21 :=
by
  simp [F1, F2, hF1, hF2]
  sorry

end ellipse_major_axis_length_l816_816651


namespace imaginary_part_z1z2_l816_816697

open Complex

-- Define the complex numbers z1 and z2
def z1 : ℂ := (1 : ℂ) - I
def z2 : ℂ := (2 : ℂ) + 4 * I

-- Define the product of z1 and z2
def z1z2 : ℂ := z1 * z2

-- State the theorem that the imaginary part of z1z2 is 2
theorem imaginary_part_z1z2 : z1z2.im = 2 := by
  -- Proof steps would go here
  sorry

end imaginary_part_z1z2_l816_816697


namespace rhombus_area_in_rectangle_l816_816574

theorem rhombus_area_in_rectangle :
  ∀ (l w : ℝ), 
  (∀ (A B C D : ℝ), 
    (2 * w = l) ∧ 
    (l * w = 72) →
    let diag1 := w 
    let diag2 := l 
    (1/2 * diag1 * diag2 = 36)) :=
by
  intros
  sorry

end rhombus_area_in_rectangle_l816_816574


namespace inequality_proof_l816_816812

theorem inequality_proof (x1 x2 y1 y2 z1 z2 : ℝ)
  (hx1 : x1 > 0) (hx2 : x2 > 0)
  (hx1y1 : x1 * y1 - z1^2 > 0) (hx2y2 : x2 * y2 - z2^2 > 0) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 - z2)^2) ≤ 1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
sorry

end inequality_proof_l816_816812


namespace sqrt_64_eq_8_l816_816513

theorem sqrt_64_eq_8 : Real.sqrt 64 = 8 := 
by
  sorry

end sqrt_64_eq_8_l816_816513


namespace equivalent_annual_compounding_rate_l816_816951

-- Define annual interest rate and quarterly compounding
def annual_interest_rate : ℝ := 0.10
def quarterly_rate : ℝ := annual_interest_rate / 4

-- Define the equivalent annual compounding rate
def equivalent_annual_rate : ℝ := (1 + quarterly_rate) ^ 4

-- Define the annual compounding rate r as a percentage
def r : ℝ := (equivalent_annual_rate - 1) * 100

-- State the theorem with the given conditions and question
theorem equivalent_annual_compounding_rate :
  round r = 10.38 := by
  sorry

end equivalent_annual_compounding_rate_l816_816951


namespace determinant_roots_cubic_l816_816766

theorem determinant_roots_cubic (a b c s p q : ℝ)
  (h1 : ∀ x : ℝ, Polynomial.eval x (Polynomial.C q + Polynomial.X * (Polynomial.C p + Polynomial.X * (Polynomial.C s + Polynomial.X))) = 0 ↔ x = a ∨ x = b ∨ x = c) :
  Matrix.det !![!![a, 1, 1], !![1, b, 1], !![1, 1, c]] = -q - s + 2 :=
by
  -- Proof is omitted
  sorry

end determinant_roots_cubic_l816_816766


namespace card_probability_l816_816499

theorem card_probability :
  let n := 120,
  let multiples_2 := n / 2,
  let multiples_3 := n / 3,
  let multiples_5 := n / 5,
  let multiples_6 := n / 6,
  let multiples_10 := n / 10,
  let multiples_15 := n / 15,
  let multiples_30 := n / 30,
  let total_count := multiples_2 + multiples_3 + multiples_5 - multiples_6 - multiples_10 - multiples_15 + multiples_30,
  total_count / n = 11 / 15 :=
by
  let n := 120
  let multiples_2 := n / 2
  let multiples_3 := n / 3
  let multiples_5 := n / 5
  let multiples_6 := n / 6
  let multiples_10 := n / 10
  let multiples_15 := n / 15
  let multiples_30 := n / 30
  let total_count := multiples_2 + multiples_3 + multiples_5 - multiples_6 - multiples_10 - multiples_15 + multiples_30
  calc
    total_count / n = 11 / 15 : by sorry

end card_probability_l816_816499


namespace twenty_fourth_decimal_l816_816886

theorem twenty_fourth_decimal (n : ℕ) (h : n = 24) : 
  (decimal_of_fraction (35 / 36) n) = 2 := 
sorry

def decimal_of_fraction (f : ℚ) (n : ℕ) : ℕ := sorry

end twenty_fourth_decimal_l816_816886


namespace positive_difference_of_squares_l816_816860

theorem positive_difference_of_squares 
  (a b : ℕ)
  (h1 : a + b = 60)
  (h2 : a - b = 16) : a^2 - b^2 = 960 :=
by
  sorry

end positive_difference_of_squares_l816_816860


namespace cos_315_deg_l816_816540

noncomputable def cos_315 : ℝ :=
  Real.cos (315 * Real.pi / 180)

theorem cos_315_deg : cos_315 = Real.sqrt 2 / 2 :=
by
  sorry

end cos_315_deg_l816_816540


namespace triangle_side_squares_l816_816814

theorem triangle_side_squares (a b c : ℝ) (acute right obtuse : Prop)
  (h_inscribed : a^2 + b^2 + c^2 = 4 * (sin α)^2 + 4 * (sin β)^2 + 4 * (sin γ)^2)
  (h_acute : α < 90 ∧ β < 90 ∧ γ < 90)
  (h_right : α = 90 ∨ β = 90 ∨ γ = 90)
  (h_obtuse : α > 90 ∨ β > 90 ∨ γ > 90):
  a^2 + b^2 + c^2 = 
  if acute then 8 + 4 * cos α * cos β * cos γ else
  if right then 8 else
  if obtuse then 8-4
  := sorry

end triangle_side_squares_l816_816814


namespace fifteenth_number_with_digit_sum_15_is_294_l816_816779

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def numbers_with_digit_sum (s : ℕ) : List ℕ :=
  List.filter (λ n => digit_sum n = s) (List.range (10 ^ 3)) -- Assume a maximum of 3-digit numbers

def fifteenth_number_with_digit_sum (s : ℕ) : ℕ :=
  (numbers_with_digit_sum s).get! 14 -- Get the 15th element (0-indexed)

theorem fifteenth_number_with_digit_sum_15_is_294 : fifteenth_number_with_digit_sum 15 = 294 :=
by
  sorry -- Proof is omitted

end fifteenth_number_with_digit_sum_15_is_294_l816_816779


namespace sufficient_condition_for_parallel_lines_l816_816453

theorem sufficient_condition_for_parallel_lines (a : ℝ) :
  (a = 3 → ∃ b c : ℝ, b ≠ c ∧ parallel (line a 2 3) (line 1 (a-1) 1)) ∧
  (∀ b : ℝ, parallel (line a 2 3) (line 1 (b-1) 1) → a = 3) :=
sorry

end sufficient_condition_for_parallel_lines_l816_816453


namespace polar_coordinates_of_2_neg2_l816_816944

noncomputable def rect_to_polar_coord (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let theta := if y < 0 
                then 2 * Real.pi - Real.arctan (x / (-y)) 
                else Real.arctan (y / x)
  (r, theta)

theorem polar_coordinates_of_2_neg2 :
  rect_to_polar_coord 2 (-2) = (2 * Real.sqrt 2, 7 * Real.pi / 4) :=
by 
  sorry

end polar_coordinates_of_2_neg2_l816_816944


namespace separator_count_l816_816040

-- Define a data structure and conditions
structure Point := (x : ℝ) (y : ℝ)

def no_three_collinear (P : Finset Point) : Prop := 
  ∀ (p1 p2 p3 : Point), {p1, p2, p3} ⊆ P → 
  ¬(p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y) = 0)

def no_four_concyclic (P : Finset Point) : Prop :=
  ∀ (p1 p2 p3 p4 : Point), {p1, p2, p3, p4} ⊆ P → 
  ¬(∃ (a b c d e : ℝ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∨ e ≠ 0) ∧ 
    ∀ p : Point, a * p.x ^ 2 + b * p.y ^ 2 + c * p.x + d * p.y + e = 0)

def separator (P : Finset Point) (C : Set Point) : Prop :=
  ∃ (p1 p2 p3 : Point) (p4 p5 : Point), 
    {p1, p2, p3, p4, p5} = P ∧
    C = {p1, p2, p3} ∧
    (∃ circle : { q | q ∈ C}, p4 ∈ interior circle ∧ p5 ∉ interior circle)

-- The main theorem statement
theorem separator_count (P : Finset Point) (h₁ : no_three_collinear P) (h₂ : no_four_concyclic P) (h₃ : P.card = 5) : 
  ∃ (C : Set Point), (separator P C) ∧ C.to_finset.card = 4 :=
sorry -- Proof goes here

end separator_count_l816_816040


namespace chris_earnings_total_l816_816438

-- Define the conditions
variable (hours_week1 hours_week2 : ℕ) (wage_per_hour earnings_diff : ℝ)
variable (hours_week1_val : hours_week1 = 18)
variable (hours_week2_val : hours_week2 = 30)
variable (earnings_diff_val : earnings_diff = 65.40)
variable (constant_wage : wage_per_hour > 0)

-- Theorem statement
theorem chris_earnings_total (total_earnings : ℝ) :
  hours_week2 - hours_week1 = 12 →
  wage_per_hour = earnings_diff / 12 →
  total_earnings = (hours_week1 + hours_week2) * wage_per_hour →
  total_earnings = 261.60 :=
by
  intros h1 h2 h3
  sorry

end chris_earnings_total_l816_816438


namespace parallelogram_area_correct_l816_816623

-- Define the base and height of the parallelogram
def base : ℝ := 32
def height : ℝ := 22

-- Define the area of a parallelogram
def parallelogram_area (b h : ℝ) : ℝ := b * h

-- The theorem to be proved
theorem parallelogram_area_correct (b h : ℝ) (hb : b = 32) (hh : h = 22) : parallelogram_area b h = 704 :=
by
  -- Using the values provided
  rw [hb, hh]
  -- Checking if it matches the expected area
  calc parallelogram_area 32 22 = 32 * 22 : rfl
                             ... = 704 : by norm_num

end parallelogram_area_correct_l816_816623


namespace smaller_square_dimensions_l816_816921

noncomputable def original_square_side_length : ℝ := 12
def smaller_square_side_length : ℝ := original_square_side_length / 2
def smaller_square_area : ℝ := smaller_square_side_length ^ 2

theorem smaller_square_dimensions :
  smaller_square_area = 36 ∧ smaller_square_side_length = 6 :=
by {
  sorry
}

end smaller_square_dimensions_l816_816921


namespace AP_eq_BC_l816_816451

variables (A B C M N P : Type) [MetricSpace A]
variables [MetricSpace B] [MetricSpace C]
variables [MetricSpace M] [MetricSpace N] [MetricSpace P]
variables (ABC : Triangle A B C) (midpoint_M : IsMidpoint M A C)
variables (midpoint_N : IsMidpoint N A B) (P_on_BM : IsOnLine P B M)
variables (not_on_CN : ¬IsOnLine P C N) (PC_eq_2PN : dist P C = 2 * dist P N)

theorem AP_eq_BC : dist A P = dist B C :=
by
  sorry

end AP_eq_BC_l816_816451


namespace find_smallest_n_l816_816099

def num_boxes : ℕ := 2011
def red_marbles_per_box : ℕ := 2

def white_marbles_in_box (k : ℕ) : ℕ := k + 1

def Q (n : ℕ) : ℚ := (4 : ℚ) / ((n + 2) * (n + 3))

theorem find_smallest_n :
  ∃ n : ℕ, Q(n) < 1 / 4022 ∧ (∀ m : ℕ, m < n → Q(m) ≥ 1 / 4022) := sorry

end find_smallest_n_l816_816099


namespace number_of_coprime_integers_in_range_l816_816197

open Nat

def isCoprimeTo16 (a : ℕ) : Prop :=
  gcd a 16 = 1

def countCoprimeTo16 : ℕ :=
  (Finset.range 16).filter isCoprimeTo16 |>.card

theorem number_of_coprime_integers_in_range :
  countCoprimeTo16 = 8 :=
by
  sorry

end number_of_coprime_integers_in_range_l816_816197


namespace find_sum_of_a_b_c_l816_816833

theorem find_sum_of_a_b_c (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
(h4 : (a + b + c) ^ 3 - a ^ 3 - b ^ 3 - c ^ 3 = 210) : a + b + c = 11 :=
sorry

end find_sum_of_a_b_c_l816_816833


namespace base5_division_l816_816595

-- Given conditions in decimal:
def n1_base10 : ℕ := 214
def n2_base10 : ℕ := 7

-- Convert the result back to base 5
def result_base5 : ℕ := 30  -- since 30 in decimal is 110 in base 5

theorem base5_division (h1 : 1324 = 214) (h2 : 12 = 7) : 1324 / 12 = 110 :=
by {
  -- these conditions help us bridge to the proof (intentionally left unproven here)
  sorry
}

end base5_division_l816_816595


namespace min_value_x_squared_y_cubed_z_l816_816772

theorem min_value_x_squared_y_cubed_z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
(h : 1 / x + 1 / y + 1 / z = 9) : x^2 * y^3 * z ≥ 729 / 6912 :=
sorry

end min_value_x_squared_y_cubed_z_l816_816772


namespace smallest_percent_coffee_tea_l816_816105

theorem smallest_percent_coffee_tea (C T : ℝ) (hC : C = 50) (hT : T = 60) : 
  ∃ x, x = C + T - 100 ∧ x = 10 :=
by
  sorry

end smallest_percent_coffee_tea_l816_816105


namespace find_number_of_divisible_by_38_l816_816072

def is_divisible (a b : Int) : Prop := ∃ k, a = k * b

def is_divisible_by_38 (n : Int) : Prop :=
  is_divisible n 38

def num_divisible_by_38 (lst : List Int) : Nat :=
  lst.countp is_divisible_by_38

theorem find_number_of_divisible_by_38 :
  let lst := [3624, 36024, 360924, 3609924, 36099924, 360999924, 3609999924]
  num_divisible_by_38 lst = 6 :=
by
  sorry

end find_number_of_divisible_by_38_l816_816072


namespace scientific_notation_l816_816076

theorem scientific_notation (n : ℕ) (a : ℝ) (h1 : (1:ℝ) ≤ a)
  (h2 : a < 10) (h3 : n = 6) : 1700000 = a * 10^n :=
by 
  use 1.7, 
  use 6, 
  sorry

end scientific_notation_l816_816076


namespace trig_identity_l816_816278

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end trig_identity_l816_816278


namespace area_difference_l816_816328

variable (E A B C D F : Type)
variable [Geometry E A B C D F]

-- Conditions
axiom angle_EAB_right : right_angle E A B
axiom angle_ABC_right : right_angle A B C
axiom AB_length : AB = 5
axiom BC_length : BC = 3
axiom AE_length : AE = 7
axiom AC_BE_intersect_D : intersect AC BE D
axiom BF_perpendicular_AE : perpendicular BF AE
axiom BF_length : BF = 2

-- Statement of the problem
theorem area_difference (h : right_angle E A B ∧ right_angle A B C ∧ AB = 5 ∧ BC = 3 ∧ AE = 7 ∧ intersect AC BE D ∧ perpendicular BF AE ∧ BF = 2) :
  area (triangle ADE) - area (triangle BDC) = 10 :=
by {
  sorry
}

end area_difference_l816_816328


namespace sin_minus_cos_l816_816262

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end sin_minus_cos_l816_816262


namespace quadratic_polynomial_has_root_l816_816628

theorem quadratic_polynomial_has_root :
  ∃ (p : Polynomial ℝ), (p.monic ∧ p.coeff 2 = 1) ∧ p.coeff 1 = -6 ∧ p.coeff 0 = 13 ∧
  (p.eval (3 - 2 * complex.I) = 0) :=
by
  use Polynomial.X^2 - Polynomial.C 6 * Polynomial.X + Polynomial.C 13
  simp [Polynomial.monic]
  split
  exact Polynomial.leading_coeff_monic (Polynomial.X^2 - Polynomial.C 6 * Polynomial.X + Polynomial.C 13)
  split
  simp
  split
  simp
  rw [Polynomial.eval_sub, Polynomial.eval_add, Polynomial.eval_C, Polynomial.eval_X,
    Polynomial.eval_mul, Polynomial.eval_I, Polynomial.eval_pow, Polynomial.eval_map]
  norm_num
  sorry

end quadratic_polynomial_has_root_l816_816628


namespace probability_even_or_less_than_3_l816_816882

theorem probability_even_or_less_than_3 :
  (favorable_outcomes : finset ℕ) := {1, 2, 4, 6},
  (total_outcomes : finset ℕ) := {1, 2, 3, 4, 5, 6},
  (probability : ℚ) := favorable_outcomes.card / total_outcomes.card,
  probability = 2 / 3 :=
by
  sorry

end probability_even_or_less_than_3_l816_816882


namespace hannah_flour_calculation_l816_816192

theorem hannah_flour_calculation :
  ∀ (bananas mushRatio flourRatio: ℕ), 
    3 = flourRatio →
    4 = mushRatio →
    20 = bananas →
    (bananas / mushRatio) * flourRatio = 15 :=
by
  intros bananas mushRatio flourRatio h_flour h_mush h_bananas
  rw [h_flour, h_mush, h_bananas]
  sorry

end hannah_flour_calculation_l816_816192


namespace find_f_of_9_l816_816662

theorem find_f_of_9 (α : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x ^ α)
  (h2 : f 2 = Real.sqrt 2) :
  f 9 = 3 :=
sorry

end find_f_of_9_l816_816662


namespace seq_1964_l816_816644

theorem seq_1964 (a : ℕ → ℤ) 
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h3 : a 3 = -1)
  (h4 : ∀ n ≥ 4, a n = a (n - 1) * a (n - 3)) :
  a 1964 = -1 :=
by {
  sorry
}

end seq_1964_l816_816644


namespace linear_inequality_inequality_l816_816203

theorem linear_inequality_inequality (k : ℝ) (x : ℝ) : 
  (k - 1) * x ^ |k| + 3 ≥ 0 → is_linear x (k - 1) * x ^ |k| + 3 → k = -1 :=
by 
  sorry

end linear_inequality_inequality_l816_816203


namespace curve_crosses_itself_at_point_l816_816928

theorem curve_crosses_itself_at_point :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ t₁^2 - 4 = t₂^2 - 4 ∧ t₁^3 - 6 * t₁ + 4 = t₂^3 - 6 * t₂ + 4 ∧ t₁^2 - 4 = 2 ∧ t₁^3 - 6 * t₁ + 4 = 4 :=
by 
  sorry

end curve_crosses_itself_at_point_l816_816928


namespace distribute_points_optimally_l816_816639

theorem distribute_points_optimally :
  ∃ (n : Fin 30 → ℕ),
    (∀ i j, i ≠ j → n i ≠ n j) ∧       -- All group sizes are distinct
    (∑ i, n i = 1989) ∧                -- Sum of sizes equals 1989
    (∀ k : Fin 29, abs ((n (k+1)).val - (n k).val) ≤ 2) :=
begin
  sorry
end

end distribute_points_optimally_l816_816639


namespace sheila_hourly_earnings_l816_816447

def sheila_hours_per_day (day : String) : Nat :=
  if day = "Monday" ∨ day = "Wednesday" ∨ day = "Friday" then 8
  else if day = "Tuesday" ∨ day = "Thursday" then 6
  else 0

def sheila_weekly_hours : Nat :=
  sheila_hours_per_day "Monday" +
  sheila_hours_per_day "Tuesday" +
  sheila_hours_per_day "Wednesday" +
  sheila_hours_per_day "Thursday" +
  sheila_hours_per_day "Friday"

def sheila_weekly_earnings : Nat := 468

theorem sheila_hourly_earnings :
  sheila_weekly_earnings / sheila_weekly_hours = 13 :=
by
  sorry

end sheila_hourly_earnings_l816_816447


namespace length_of_CD_l816_816313

theorem length_of_CD
  (S : ℝ) -- Placeholder for the center of the circle (can be non-specific)
  (R : ℝ) (SX : ℝ) (AB : ℝ) (X : Bool := true) -- Assume X represents perpendicularity
  (H_radius : R = 52)
  (H_SX : SX = 25)
  (H_AB : AB = 96)
  (H_perpendicular : X = true)
  : ℝ := 100

end length_of_CD_l816_816313


namespace emily_euros_contribution_l816_816510

-- Declare the conditions as a definition
def conditions : Prop :=
  ∃ (cost_of_pie : ℝ) (emily_usd : ℝ) (berengere_euros : ℝ) (exchange_rate : ℝ),
    cost_of_pie = 15 ∧
    emily_usd = 10 ∧
    berengere_euros = 3 ∧
    exchange_rate = 1.1

-- Define the proof problem based on the conditions and required contribution
theorem emily_euros_contribution : conditions → (∃ emily_euros_more : ℝ, emily_euros_more = 3) :=
by
  intro h
  sorry

end emily_euros_contribution_l816_816510


namespace remainder_of_sum_of_primes_is_eight_l816_816876

-- Define the first eight primes and their sum
def firstEightPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]
def sumFirstEightPrimes : ℕ := 77

-- Define the ninth prime
def ninthPrime : ℕ := 23

-- Theorem stating the equivalence
theorem remainder_of_sum_of_primes_is_eight :
  (sumFirstEightPrimes % ninthPrime) = 8 := by
  sorry

end remainder_of_sum_of_primes_is_eight_l816_816876


namespace no_3_mod_4_prime_divisor_no_nat_sol_x2_minus_y3_eq_7_l816_816823

-- Problem Part 1: No divisor congruent to 3 modulo 4
theorem no_3_mod_4_prime_divisor (x : ℤ) (p : ℕ) (hp_prime : nat.prime p) (hdiv : p ∣ (x^2 + 1)) : ¬ (p % 4 = 3) := 
sorry

-- Problem Part 2: No (x, y) ∈ ℕ² such that x^2 - y^3 = 7
theorem no_nat_sol_x2_minus_y3_eq_7 : ¬ (∃ (x y : ℕ), x^2 - y^3 = 7) :=
sorry

end no_3_mod_4_prime_divisor_no_nat_sol_x2_minus_y3_eq_7_l816_816823


namespace problem_m_n_l816_816367

theorem problem_m_n (m n : ℝ) (h1 : m * n = 1) (h2 : m^2 + n^2 = 3) (h3 : m^3 + n^3 = 44 + n^4) (h4 : m^5 + 5 = 11) : m^9 + n = -29 :=
sorry

end problem_m_n_l816_816367


namespace maximum_candies_l816_816406

theorem maximum_candies : 
  ∃ (max_candies : ℕ), 
    let initial_numbers := mul 1 45 in
    (∀ k ∈ initial_numbers, k = 1) → -- all initial numbers are 1
    (∀ t, by {assume erased_1 erased_2 ∈ initial_numbers, erased_1 + erased_2 ∉ initial_numbers}) → 
    max_candies = 990 := 
by
  sorry

end maximum_candies_l816_816406


namespace squirrel_cones_l816_816424

theorem squirrel_cones :
  ∃ (x y : ℕ), 
    x + y < 25 ∧ 
    2 * x > y + 26 ∧ 
    2 * y > x - 4 ∧
    x = 17 ∧ 
    y = 7 :=
by
  sorry

end squirrel_cones_l816_816424


namespace winning_percentage_l816_816724

theorem winning_percentage (total_votes winner_votes : ℕ) 
  (h1 : winner_votes = 1344) 
  (h2 : winner_votes - 288 = total_votes - winner_votes) : 
  (winner_votes * 100 / total_votes = 56) :=
sorry

end winning_percentage_l816_816724


namespace local_minimum_at_neg_one_l816_816774

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem local_minimum_at_neg_one : 
  ∀ f : ℝ → ℝ, f = λ x, x * Real.exp x → 
  ∃ δ > 0, ∀ h, h > 0 → h < δ →
  f (-1 + h) ≥ f (-1) ∧ f (-1 - h) ≥ f (-1) :=
begin
  sorry
end

end local_minimum_at_neg_one_l816_816774


namespace cos_315_eq_l816_816558

theorem cos_315_eq :
  let Q := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))
  in Q = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) →
     Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  intro hQ
  sorry

end cos_315_eq_l816_816558


namespace time_to_pass_platform_l816_816027

-- Given conditions
def train_length : ℝ := 1200 -- meters
def tree_pass_time : ℝ := 120 -- seconds
def platform_length : ℝ := 600 -- meters

-- Speed of the train
def train_speed : ℝ := train_length / tree_pass_time -- meters per second

-- Combined distance to pass
def total_distance : ℝ := train_length + platform_length -- meters

-- Prove the time taken to pass the platform
theorem time_to_pass_platform : (total_distance / train_speed) = 180 := 
by 
  sorry

end time_to_pass_platform_l816_816027


namespace mean_yoga_practice_days_l816_816794

noncomputable def mean_number_of_days (counts : List ℕ) (days : List ℕ) : ℚ :=
  let total_days := List.zipWith (λ c d => c * d) counts days |>.sum
  let total_students := counts.sum
  total_days / total_students

def counts : List ℕ := [2, 4, 5, 3, 2, 1, 3]
def days : List ℕ := [1, 2, 3, 4, 5, 6, 7]

theorem mean_yoga_practice_days : mean_number_of_days counts days = 37 / 10 := 
by 
  sorry

end mean_yoga_practice_days_l816_816794


namespace semicircle_circumference_approx_54_l816_816396

noncomputable def side_of_square (perimeter_rectangle : ℝ) : ℝ := perimeter_rectangle / 4

noncomputable def circumference_of_semicircle (diameter : ℝ) : ℝ := (Real.pi * diameter) / 2 + diameter 

theorem semicircle_circumference_approx_54 (len : ℝ) (bre : ℝ) :
  let perimeter_rectangle := 2 * (len + bre)
  let side := side_of_square (perimeter_rectangle)
  let diameter := side
  circumference_of_semicircle(diameter) ≈ 54.002745 :=
by
  -- Set given values of length and breadth
  have len_val : len = 22 := by sorry
  have bre_val : bre = 20 := by sorry

  -- Establish perimeter of the rectangle
  let perimeter_rectangle := 2 * (len + bre)
  have perimeter_rectangle_val : perimeter_rectangle = 84 := by sorry

  -- Derive the side of the square
  have side_val : side = 21 := by sorry

  -- Derive the diameter equal to side length
  let diameter := side
  have diameter_val : diameter = 21 := by sorry

  -- Derive the circumference of the semicircle
  have circumference_val : 
    circumference_of_semicircle (diameter) ≈ 54.002745 :=
    by
      calc (Real.pi * diameter) / 2 + diameter
       _ = ((Real.pi * 21) / 2) + 21 := by congr; sorry
       _ ≈ 54.002745 := by sorry -- Use approximation property

  exact circumference_val

end semicircle_circumference_approx_54_l816_816396


namespace sin_minus_cos_l816_816265

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end sin_minus_cos_l816_816265


namespace probability_of_draw_l816_816459

-- Define the probabilities as given conditions
def P_A : ℝ := 0.4
def P_A_not_losing : ℝ := 0.9

-- Define the probability of drawing
def P_draw : ℝ :=
  P_A_not_losing - P_A

-- State the theorem to be proved
theorem probability_of_draw : P_draw = 0.5 := by
  sorry

end probability_of_draw_l816_816459


namespace sin_minus_cos_l816_816261

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end sin_minus_cos_l816_816261


namespace total_sum_is_2560_l816_816675

def M : set ℕ := { x | 1 ≤ x ∧ x ≤ 10 }

def sum_of_subsets (A : set ℕ) : ℤ := 
  A.to_finset.sum (λ k : ℕ, (-1)^k * k)

def total_sum_of_all_subsets (S : set ℕ) : ℤ :=
  (S.nonempty_subsets.to_finset.sum (λ A, sum_of_subsets A))

theorem total_sum_is_2560 : total_sum_of_all_subsets M = 2560 := 
sorry

end total_sum_is_2560_l816_816675


namespace min_value_is_2_sqrt_2_l816_816983

noncomputable def min_value (a b : ℝ) : ℝ :=
  a^2 + b^2 / (a - b)

theorem min_value_is_2_sqrt_2 (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : a * b = 1) : 
  min_value a b = 2 * Real.sqrt 2 := 
sorry

end min_value_is_2_sqrt_2_l816_816983


namespace trig_values_l816_816659

theorem trig_values
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 4)
  (hβ : π / 4 < β ∧ β < π / 2)
  (h_sin_cos_α : sin α + cos α = (3 * real.sqrt 5) / 5)
  (h_sin_beta_pi4_minus : sin (β - π / 4) = 3 / 5) :
  (sin (2 * α) = 4 / 5 ∧ tan (2 * α) = 4 / 3) ∧ cos (α + 2 * β) = - (11 * real.sqrt 5) / 25 :=
by
  sorry

end trig_values_l816_816659


namespace mean_days_correct_l816_816307

noncomputable def mean_days (a1 a2 a3 a4 a5 d1 d2 d3 d4 d5 : ℕ) : ℚ :=
  (a1 * d1 + a2 * d2 + a3 * d3 + a4 * d4 + a5 * d5 : ℚ) / (a1 + a2 + a3 + a4 + a5)

theorem mean_days_correct : mean_days 2 4 5 7 4 1 2 4 5 6 = 4.05 := by
  sorry

end mean_days_correct_l816_816307


namespace find_tip_rate_l816_816340

noncomputable def hourly_wage : ℝ := 4.00
def shifts : ℕ := 3
def hours_per_shift : ℕ := 8
def total_hours : ℕ := shifts * hours_per_shift
def averaged_orders_per_hour : ℝ := 40
def total_earnings : ℝ := 240

theorem find_tip_rate :
    let total_wage_earnings := (total_hours : ℝ) * hourly_wage,
        tip_earnings := total_earnings - total_wage_earnings,
        total_orders_cost := (total_hours : ℝ) * averaged_orders_per_hour,
        tip_rate := (tip_earnings / total_orders_cost) * 100 in
    tip_rate = 15 :=
by
    sorry

end find_tip_rate_l816_816340


namespace ellipse_foci_distance_l816_816506

noncomputable def distance_between_foci (a b : ℝ) : ℝ :=
  2 * real.sqrt (a ^ 2 - b ^ 2)

theorem ellipse_foci_distance :
  ∀ (a b : ℝ), a = 5 → b = 3 → distance_between_foci a b = 8 :=
by
  intros a b ha hb
  rw [ha, hb]
  unfold distance_between_foci
  norm_num
  -- This line is to acknowledge the proof is not provided
  sorry

end ellipse_foci_distance_l816_816506


namespace minimize_time_l816_816066

noncomputable def timeA (x : ℕ) [fact (1 ≤ x)][fact (x ≤ 49)] : ℝ := 90 / x

noncomputable def timeB (x : ℕ) [fact (1 ≤ x)][fact (x ≤ 49)] : ℝ := 50 / (50 - x)

noncomputable def timeTotal (x : ℕ) [fact (1 ≤ x)][fact (x ≤ 49)] : ℝ :=
  max (timeA x) (timeB x)

theorem minimize_time : 
  ∃ x : ℕ, (1 ≤ x) ∧ (x ≤ 49) ∧ ∀ y : ℕ, (1 ≤ y) ∧ (y ≤ 49) → timeTotal x ≤ timeTotal y ∧ x = 32 :=
begin
  sorry
end

end minimize_time_l816_816066


namespace no_maximum_quotient_minimum_quotient_two_digit_minimum_quotient_general_l816_816095

-- Definition of a natural number with at least two digits
def at_least_two_digits (n : ℕ) : Prop :=
  n ≥ 10

-- Definition of the sum of digits function
def digit_sum (n : ℕ) : ℕ :=
  (n.to_digits 10).foldl (λ a x => a + x) 0

-- Proposition stating there is no upper bound for the quotient for numbers of the form 10^k
theorem no_maximum_quotient (k : ℕ) (hk : k ≥ 2) : 
  ∀ m : ℕ, ∃ n : ℕ, at_least_two_digits n ∧ n = 10^k ∧ m < n / digit_sum n :=
sorry

-- Proposition stating the minimum quotient for a two-digit number
theorem minimum_quotient_two_digit : 
  ∃ n : ℕ, at_least_two_digits n ∧ n < 100 ∧ n / digit_sum n = 19 / 10 :=
sorry

-- Proposition for general n-digit number's quotient
theorem minimum_quotient_general (n : ℕ) (hn : n ≥ 2) : 
  ∃ A : ℕ, at_least_two_digits A ∧ (gcd' (digit_sum A) A) != zeroMod → 
  (10^(n-1)) ≤ A ∧ (A / digit_sum A) ≤ 1.9 :=
sorry

end no_maximum_quotient_minimum_quotient_two_digit_minimum_quotient_general_l816_816095


namespace percentage_increase_numerator_l816_816703

variable (N D : ℝ) (P : ℝ)
variable (h1 : N / D = 0.75)
variable (h2 : (N * (1 + P / 100)) / (D * 0.92) = 15 / 16)

theorem percentage_increase_numerator :
  P = 15 :=
by
  sorry

end percentage_increase_numerator_l816_816703


namespace cos_315_eq_sqrt2_div2_l816_816534

theorem cos_315_eq_sqrt2_div2 : Real.cos (315 * Real.pi / 180) = sqrt 2 / 2 := 
sorry

end cos_315_eq_sqrt2_div2_l816_816534


namespace Joey_downhill_speed_l816_816341

theorem Joey_downhill_speed
  (Route_length : ℝ) (Time_uphill : ℝ) (Speed_uphill : ℝ) (Overall_average_speed : ℝ) (Extra_time_due_to_rain : ℝ) :
  Route_length = 5 →
  Time_uphill = 1.25 →
  Speed_uphill = 4 →
  Overall_average_speed = 6 →
  Extra_time_due_to_rain = 0.25 →
  ((2 * Route_length) / Overall_average_speed - Time_uphill - Extra_time_due_to_rain) * (Route_length / (2 * Route_length / Overall_average_speed - Time_uphill - Extra_time_due_to_rain)) = 30 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Joey_downhill_speed_l816_816341


namespace sin_minus_cos_l816_816255

theorem sin_minus_cos (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < (Real.pi / 2)) (hθ3 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := by
  sorry

end sin_minus_cos_l816_816255


namespace length_of_chord_16_l816_816469

theorem length_of_chord_16 :
  let F := (1, 0) in
  let k := Real.tan (Real.pi / 6) in
  let line_eq := λ (x : ℝ), k * (x - 1) in
  let parab_eq := λ (x y : ℝ), y^2 = 4 * x in
  ∃ (A B : ℝ × ℝ), 
    (parab_eq A.1 A.2) ∧ 
    (parab_eq B.1 B.2) ∧ 
    (A ≠ B) ∧ 
    (line_eq A.1 = A.2) ∧ 
    (line_eq B.1 = B.2) ∧
    (Real.sqrt (1 + k^2) * Real.abs (A.1 - B.1) = 16) := sorry

end length_of_chord_16_l816_816469


namespace sin_minus_cos_l816_816263

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end sin_minus_cos_l816_816263


namespace f_lambda_constant_l816_816449

theorem f_lambda_constant (ABCD : Type) (E F A B C D : ABCD)
  (AE EB CF FD : ℝ) (λ : ℝ) (h1 : AE/EB = λ) (h2 : CF/FD = λ) (h3 : 0 < λ) 
  (h4 : ∀ (x y z : ABCD), x ≠ y → y ≠ z → z ≠ x → False) 
  (alpha_lambda beta_lambda : ℝ) 
  (h5 : alpha_lambda = 90) (h6 : beta_lambda = 0) :
  (f_lambda : ℝ) = alpha_lambda + beta_lambda := by
    have h7 : f_lambda = 90 := rfl
    exact h7

end f_lambda_constant_l816_816449


namespace total_ways_to_choose_president_and_vice_president_of_same_gender_l816_816803

theorem total_ways_to_choose_president_and_vice_president_of_same_gender :
  let boys := 12
  let girls := 12
  (boys * (boys - 1) + girls * (girls - 1)) = 264 :=
by
  sorry

end total_ways_to_choose_president_and_vice_president_of_same_gender_l816_816803


namespace part_a_part_b_l816_816346

def matrix_pair_in_F (A B : Matrix (Fin 2) (Fin 2) ℤ) : Prop :=
  ∃ (k : ℕ) (C : Fin k → Matrix (Fin 2) (Fin 2) ℤ), (∀ i, C i = A ∨ C i = B) ∧ (1 < k) ∧ (Matrix.mul_vec C = 0)

def k_min (A B : Matrix (Fin 2) (Fin 2) ℤ) : ℕ :=
  Nat.find (matrix_pair_in_F A B)

theorem part_a (A B : Matrix (Fin 2) (Fin 2) ℤ) (p : ℕ) (hp : 0 < p) 
  (hF : matrix_pair_in_F A B)
  (hAk : k_min A B = p + 2)
  (detA : A.det = 0)
  (detB : B.det ≠ 0) :
  A ⬝ B ^ p ⬝ A = 0 := sorry

theorem part_b (k : ℕ) (hk : 3 ≤ k) :
  ∃ (A B : Matrix (Fin 2) (Fin 2) ℤ), matrix_pair_in_F A B ∧ k_min A B = k := sorry

end part_a_part_b_l816_816346


namespace sin_minus_cos_eq_l816_816232

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end sin_minus_cos_eq_l816_816232


namespace segment_bisection_l816_816850

variables {P : Type} [euclidean_geometry P] (O A B C K L M : P)
variables [incidence_geometry P] [hO : center O]

-- Given Conditions
def tangent_to_circle (O A : P) : Prop :=
  tangent_point O (to_circle O) A

def parallel (l₁ l₂ : line P) : Prop :=
  parallel_lines l₁ l₂

def intersects (l : line P) (C : conic_section P) (p : P) : Prop :=
  point_on_conic_section p C ∧ line_meets_conic_section l C

def intersection (l₁ l₂ : line P) (M : P) : Prop :=
  point_on_line M l₁ ∧ point_on_line M l₂

-- Theorem to be proved
theorem segment_bisection
  (h1 : tangent_to_circle O A)
  (h2 : parallel (line_through A O) (line_through B C))
  (h3 : intersects (line_through O B) (circle_centered_at O) K)
  (h4 : intersects (line_through O C) (circle_centered_at O) L)
  (h5 : intersection (line_through K L) (line_through A O) M) :
  distance O M = distance M A :=
begin
  sorry
end

end segment_bisection_l816_816850


namespace area_ratio_of_subdivided_triangle_l816_816370

theorem area_ratio_of_subdivided_triangle (T : Triangle) (h_eq : is_equilateral T)
(points : set Point) (h_med : ∀p ∈ points, ∃ m : Median, divides_in_ratio p m (3, 1)) :
  area (triangle_with_points points) = area T / 64 :=
sorry

end area_ratio_of_subdivided_triangle_l816_816370


namespace range_fx_l816_816949

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem range_fx : Set.range (λ x : ℝ, x + Real.sin x) = Set.Icc 0 (2 * Real.pi) := by
  sorry

end range_fx_l816_816949


namespace find_f3_l816_816162

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem find_f3 
  (hf : is_odd f) 
  (hg : is_even g) 
  (h : ∀ x, f x + g x = 1 / (x - 1)) : 
  f 3 = 3 / 8 :=
by 
  sorry

end find_f3_l816_816162


namespace log_216_equals_3_log_6_l816_816456

theorem log_216_equals_3_log_6 :
  ∀ (log : ℕ → ℝ), log 216 = 3 * log 6 → ∏ x in [1,1], log x = 1

end log_216_equals_3_log_6_l816_816456


namespace planes_determined_13_l816_816778

-- Definitions for the conditions.
variable (a b : Set Point)
variable (skew : SkewLines a b)
variable (points_on_a : Finset.Point a) -- 6 points on line a
variable (points_on_b : Finset.Point b) -- 7 points on line b

noncomputable def num_planes : Nat :=
  (points_on_a.card + points_on_b.card : Nat)

theorem planes_determined_13 (ha : points_on_a.card = 6) (hb : points_on_b.card = 7) :
  num_planes a b points_on_a points_on_b = 13 := by
  sorry

end planes_determined_13_l816_816778


namespace minimum_groups_l816_816482

theorem minimum_groups (athletes groups max_athletes_per_group : ℕ) (h₁ : athletes = 30) (h₂ : max_athletes_per_group = 12) (h₃ : ∃ y, y ∣ athletes ∧ y ≤ max_athletes_per_group) : 
  ∃ x, x = athletes / max_athletes_per_group ∧ x = 3 :=
by {
  have h₄ := nat.divisors athletes, -- Get the list of divisors of 30
  sorry
}

end minimum_groups_l816_816482


namespace find_solutions_of_x4_minus_16_l816_816620

noncomputable def solution_set : Set Complex :=
  {2, -2, Complex.I * 2, -Complex.I * 2}

theorem find_solutions_of_x4_minus_16 :
  {x : Complex | x^4 - 16 = 0} = solution_set :=
by
  sorry

end find_solutions_of_x4_minus_16_l816_816620


namespace rectangle_area_l816_816465

-- Define the basic setup and geometric conditions
variables {A B C D M N : Point}

-- Define the rectangle ABCD
def is_rectangle (A B C D : Point) : Prop :=
  -- Define the properties of the rectangle here
  sorry

-- Define the circle passing through point C that touches sides AB and AD at points M and N
def circle_touches_sides (C M N : Point) (AB AD : Line) : Prop :=
  -- Define the properties of the circle here
  sorry

-- Define the distance property from point C to line segment MN
def distance_from_point_to_line_seg (C N M : Point) (dist: ℝ) : Prop :=
  -- Define the distance property here
  sorry

-- The theorem we want to prove
theorem rectangle_area 
  (h1 : is_rectangle A B C D)
  (h2 : circle_touches_sides C M N (line_through A B) (line_through A D))
  (h3 : distance_from_point_to_line_seg C M N 5) :
  area_of_rectangle A B C D = 25 :=
sorry

end rectangle_area_l816_816465


namespace find_c_find_area_l816_816737

-- Step 1: Define the conditions for part (1)
def triangle_condition (a b c A B C : ℝ) : Prop :=
  4 * a * real.cos B = c^2 - 4 * b * real.cos A

-- Statement for part (1)
theorem find_c (a b A B : ℝ) (C : Type) [triangle_condition a b 4 A B C] :
  4 = 4 :=
sorry

-- Step 2: Define the conditions for part (2)
def triangle_condition2 (a b A B : ℝ) (C : ℝ) (sum_cond : a + b = 4 * real.sqrt 2) : Prop :=
  C = real.pi / 3 ∧ a + b = 4 * real.sqrt 2

-- Statement for part (2)
theorem find_area (a b : ℝ) (sum_cond : a + b = 4 * real.sqrt 2) (C : ℝ) (area_calc : triangle_condition2 a b 0 0 C sum_cond) :
  (1 / 2) * a * b * real.sin (real.pi / 3) = (4 * real.sqrt 3) / 3 :=
sorry

end find_c_find_area_l816_816737


namespace sum_of_solutions_eq_twenty_sum_of_solutions_main_proof_l816_816130

theorem sum_of_solutions_eq_twenty : 
  ∀ x : ℝ, x ≠ 0 → (x^2 - 13 * x + 36) / x = 7 → x = 18 ∨ x = 2 :=
begin
  sorry
end

theorem sum_of_solutions : (18 + 2) = 20 :=
by linarith

theorem main_proof : 
  ∀ (x : ℝ) (h₁ : x ≠ 0) (h₂ : (x^2 - 13 * x + 36) / x = 7), 18 + 2 = 20 :=
begin
  assume x h₁ h₂,
  exact sum_of_solutions,
end

end sum_of_solutions_eq_twenty_sum_of_solutions_main_proof_l816_816130


namespace final_winning_percentage_of_team_l816_816712

theorem final_winning_percentage_of_team
  (initial_matches_played : ℕ)
  (initial_winning_percentage : ℕ)
  (additional_matches_won : ℕ)
  : initial_matches_played = 120 ->
    initial_winning_percentage = 20 ->
    additional_matches_won = 80 ->
    let initial_matches_won := (initial_winning_percentage * initial_matches_played) / 100 in
    let total_matches_won := initial_matches_won + additional_matches_won in
    let total_matches_played := initial_matches_played + additional_matches_won in
    (total_matches_won * 100) / total_matches_played = 52 :=
by
  intros h1 h2 h3,
  rw [h1, h2, h3],
  unfold initial_matches_won total_matches_won total_matches_played,
  norm_num,
  sorry -- Proof steps go here

end final_winning_percentage_of_team_l816_816712


namespace rhombus_area_from_polynomial_l816_816624

noncomputable def complex_roots_rhombus_area : ℂ :=
  let p := (complex.abs (⟨39, -12⟩ : ℂ));
  in 2 * p * p

theorem rhombus_area_from_polynomial :
  let poly_tuples := [((1 : ℂ) ^ 4), (-4 : ℂ) * (1 : ℂ) ^ 3, (13 - 4 * (complex.I : ℂ)) * (1 : ℂ) ^ 2, - (10 + 14 * (complex.I : ℂ)) * (1 : ℂ), (39 - 12 * (complex.I : ℂ))]
  let roots := (complex_roots_rhombus_area)
  (∀ z, z ∈ roots → (z ^ 4 - 4 * z ^ 3 + (13 - 4 * complex.I) * z ^ 2 - (10 + 14 * complex.I) * z + (39 - 12 * complex.I) = 0)) → 
  (complex.abs (⟨39, -12⟩ : ℂ)) = real.sqrt 1665 → 
  let area := 2 * (real.sqrt 1665) * (real.sqrt 1665)
  area = 3330 :=
by sorry

end rhombus_area_from_polynomial_l816_816624


namespace tangent_normal_line_l816_816974

section 

variables {t : ℝ}
def x_eq := sin t
def y_eq := cos (2 * t)
def t0 : ℝ := (Real.pi / 6)
def x0 : ℝ := sin (Real.pi / 6)
def y0 : ℝ := cos (Real.pi / 3)
def y0' : ℝ := -2 * sin (Real.pi / 6)

theorem tangent_normal_line :
  (∀ x y : ℝ, y = -2 * x + 1.5)
  ∧ (∀ x y : ℝ, y = 1 / 2 * x + 0.25) :=
by
  sorry

end

end tangent_normal_line_l816_816974


namespace patsy_needs_more_appetizers_l816_816805

def appetizers_per_guest := 6
def number_of_guests := 30
def deviled_eggs := 3 -- dozens
def pigs_in_a_blanket := 2 -- dozens
def kebabs := 2 -- dozens

theorem patsy_needs_more_appetizers :
  let total_required := appetizers_per_guest * number_of_guests,
      total_made := (deviled_eggs + pigs_in_a_blanket + kebabs) * 12,
      total_needed := total_required - total_made
  in total_needed / 12 = 8 := sorry

end patsy_needs_more_appetizers_l816_816805


namespace find_solutions_l816_816603

noncomputable def solutions_to_equation : set Complex :=
  {x | x^4 - 16 = 0}

theorem find_solutions : solutions_to_equation = {2, -2, Complex.I * 2, -Complex.I * 2} :=
sorry

end find_solutions_l816_816603


namespace complex_conjugate_point_l816_816391

def i : ℂ := complex.I

theorem complex_conjugate_point :
  let z := (1 + 2 * i) / i
  let z_conj := complex.conj z
  (z_conj.re, z_conj.im) = (2, 1) :=
by
  sorry

end complex_conjugate_point_l816_816391


namespace ludwig_weekly_salary_is_55_l816_816782

noncomputable def daily_salary : ℝ := 10
noncomputable def full_days : ℕ := 4
noncomputable def half_days : ℕ := 3
noncomputable def half_day_salary := daily_salary / 2

theorem ludwig_weekly_salary_is_55 :
  (full_days * daily_salary + half_days * half_day_salary = 55) := by
  sorry

end ludwig_weekly_salary_is_55_l816_816782


namespace division_in_base_5_l816_816594

noncomputable def base5_quick_divide : nat := sorry

theorem division_in_base_5 (a b quotient : ℕ) (h1 : a = 1324) (h2 : b = 12) (h3 : quotient = 111) :
  ∃ c : ℕ, c = quotient ∧ a / b = quotient :=
by
  sorry

end division_in_base_5_l816_816594


namespace finite_triples_satisfying_eq_l816_816825

theorem finite_triples_satisfying_eq :
  { (a, b, c) : ℕ × ℕ × ℕ // a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 2009 * (a + b + c) }.finite :=
sorry

end finite_triples_satisfying_eq_l816_816825


namespace find_largest_base_l816_816964

noncomputable def base_b_digit_sum (b : ℕ) : ℕ :=
  ((b + 2) ^ 4).digits b.sum

theorem find_largest_base (b : ℕ) :
  (∀ b > 8, base_b_digit_sum b ≠ 36) ∧ (base_b_digit_sum 8 ≠ 36) := sorry

end find_largest_base_l816_816964


namespace side_length_of_square_l816_816015

-- Mathematical definitions and conditions
def square_area (side : ℕ) : ℕ := side * side

theorem side_length_of_square {s : ℕ} (h : square_area s = 289) : s = 17 :=
sorry

end side_length_of_square_l816_816015


namespace cost_of_each_bar_l816_816953

variables (c : ℝ) (bars_sold : ℕ) (total_earned : ℝ)

-- Given conditions
def bars_total := 8
def bars_remaining := 3
def bars_sold := bars_total - bars_remaining
def total_earned := 20

-- Statement to prove
theorem cost_of_each_bar (h : bars_sold * c = total_earned) : c = 4 :=
sorry

end cost_of_each_bar_l816_816953


namespace sin_minus_cos_l816_816219

variable (θ : ℝ)
hypothesis (h1 : 0 < θ ∧ θ < π / 2)
hypothesis (h2 : Real.tan θ = 1 / 3)

theorem sin_minus_cos (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := 
by 
  sorry

end sin_minus_cos_l816_816219


namespace cos_315_is_sqrt2_div_2_l816_816523

noncomputable def cos_315_eq : Prop := 
  real.cos (315 * real.pi / 180) = real.sqrt 2 / 2

theorem cos_315_is_sqrt2_div_2 : cos_315_eq := by 
  sorry

end cos_315_is_sqrt2_div_2_l816_816523


namespace triangle_RSC_l816_816873

def Point := (ℝ, ℝ)

def triangle (A B C : Point) : Prop :=
(A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

def isVerticalLineIntersection (R S : Point) : Prop :=
R.1 = S.1

def pointInLineSegment (P A B : Point) : Prop :=
∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P.1 = (1 - t) * A.1 + t * B.1 ∧ P.2 = (1 - t) * A.2 + t * B.2

def areaOfTriangle (A B C : Point) : ℝ :=
0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

def isoscelesRightTriangle (R S C : Point) : Prop :=
(R.1 - S.1)^2 + (R.2 - S.2)^2 = (R.1 - C.1)^2 + (R.2 - C.2)^2 ∧
(S.1 - C.1)^2 + (S.2 - C.2)^2 = (R.1 - S.1)^2 + (R.2 - S.2)^2

theorem triangle_RSC :
  ∃ R : Point, ∃ S : Point,
    let A := (0, 5)
    let B := (3, 0)
    let C := (9, 0) in
    triangle A B C ∧
    pointInLineSegment R A C ∧
    pointInLineSegment S B C ∧
    isVerticalLineIntersection R S ∧
    areaOfTriangle R S C = 10 ∧
    isoscelesRightTriangle R S C ∧
    abs (R.1 - R.2) = 13 / 3 :=
sorry

end triangle_RSC_l816_816873


namespace cos_probability_l816_816820

noncomputable def probability_cosine (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 2 * ℝ.pi ∧ (cos x ≥ sqrt 2 / 2) then 1 / (2 * ℝ.pi) else 0

theorem cos_probability : 
  ∫ x in 0..(2 * ℝ.pi), probability_cosine x = 1 / 4 :=
by
  -- Proof goes here
  sorry

end cos_probability_l816_816820


namespace num_good_circles_for_5_points_l816_816452

-- Definitions for the conditions
def points : Type := ℝ × ℝ  -- type representing points in a plane

-- Assuming no three points are collinear and no four points are concyclic
def no_three_collinear (pts: list points) : Prop :=
  ∀ (a b c: points), a ∈ pts → b ∈ pts → c ∈ pts → (a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  ¬ (collinear a b c)

def no_four_concyclic (pts: list points) : Prop :=
  ∀ (a b c d: points), a ∈ pts → b ∈ pts → c ∈ pts → d ∈ pts → 
  (a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ d) → ¬ (cyclic a b c d)

-- Assuming collinear and cyclic properties
def collinear (a b c: points) : Prop :=
  let (x1, y1) := a in
  let (x2, y2) := b in
  let (x3, y3) := c in
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

def cyclic (a b c d: points) : Prop := 
  sorry  -- complex, geometry-related condition that can be defined further if needed

-- The number of "good" circles for \(5\) points
def good_circle_count (pts: list points) : ℕ := sorry  -- function which will count good circles

-- The theorem proving the required number of good circles is 4
theorem num_good_circles_for_5_points (pts: list points) 
  (h_len : pts.length = 5)
  (h_nc : no_three_collinear pts)
  (h_cc : no_four_concyclic pts) : 
  good_circle_count pts = 4 := 
  sorry  -- proof goes here

end num_good_circles_for_5_points_l816_816452


namespace number_of_children_proof_l816_816046

noncomputable def number_of_children (x : ℕ) : Prop :=
  84000 / x + 3500 = 84000 / (x - 2)

theorem number_of_children_proof : ∃ (x : ℕ), number_of_children x ∧ x = 8 :=
by
  use 8
  unfold number_of_children
  -- Equate the left-hand side and the right-hand side of the equation
  have : (84000 : ℝ) / 8 + 3500 = (84000 : ℝ) / 6 := sorry
  exact this

end number_of_children_proof_l816_816046


namespace sin_minus_cos_theta_l816_816243

theorem sin_minus_cos_theta (θ : ℝ) (h₀ : 0 < θ ∧ θ < (π / 2)) 
  (h₁ : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -√10 / 5 :=
by
  sorry

end sin_minus_cos_theta_l816_816243


namespace james_ride_time_l816_816742

theorem james_ride_time (distance speed : ℝ) (h_distance : distance = 200) (h_speed : speed = 25) : distance / speed = 8 :=
by
  rw [h_distance, h_speed]
  norm_num

end james_ride_time_l816_816742


namespace transformed_data_properties_l816_816647

theorem transformed_data_properties (x : Fin 5 → ℝ) (mean_var_x : (∑ i : Fin 5, x i) = 10 ∧ (∑ i : Fin 5, (x i - 2) ^ 2) = 5 / 3) :
  let y := (λ i : Fin 5, 3 * x i - 2)
  in (∑ i : Fin 5, y i) = 20 ∧ (∑ i : Fin 5, (y i - 4) ^ 2) = 15 := sorry

end transformed_data_properties_l816_816647


namespace age_in_1900_l816_816306

theorem age_in_1900 
  (x y : ℕ)
  (H1 : y = 29 * x)
  (H2 : 1901 ≤ y + x ∧ y + x ≤ 1930) :
  1900 - y = 44 := 
sorry

end age_in_1900_l816_816306


namespace find_angle_A_condition_find_area_range_l816_816738

variable {A B C a b c S : ℝ}

-- Part (1): Prove angle A
theorem find_angle_A_condition (A B C a b c : ℝ) : 
  (sqrt 3 * b * sin (B + C) + a * cos B = c) → 
  A = π / 6 :=
  sorry

-- Part (2): Prove range of the area of triangle 
theorem find_area_range (A B C a b c S : ℝ) : 
  b = 6 → 
  0 < B ∧ B < π / 2 ∧ 
  A = π / 6 →
  (9 * sqrt 3 / 2 < S ∧ S < 6 * sqrt 3) :=
  sorry

end find_angle_A_condition_find_area_range_l816_816738


namespace product_of_decimals_l816_816631

theorem product_of_decimals (a b : ℚ) (h1 : a = 0.3) (h2 : b = 0.7) : a * b = 0.21 := by
  -- Since we handle the proof using rational numbers in Lean, we specify the following
  have ha : a = (3 : ℚ) / 10 := by norm_num [h1]
  have hb : b = (7 : ℚ) / 10 := by norm_num [h2]
  calc
    a * b = (3 / 10) * (7 / 10) : by rw [ha, hb]
       ... = 21 / 100 : by norm_num
       ... = 0.21 : by norm_num
  -- Proof completion
  sorry

end product_of_decimals_l816_816631


namespace point_A_coordinates_minimum_length_AB_l816_816145

open Function Real

noncomputable def parabola : Type := { p : ℝ // p > 0 }
def focus (p : parabola) : ℝ × ℝ := (p.val / 2, 0)
def directrix (p : parabola) (x : ℝ) : Prop := x = -p.val / 2
def on_parabola (p : parabola) (A : ℝ × ℝ) : Prop := (A.snd)^2 = 4 * (A.fst)

def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.fst - B.fst)^2 + (A.snd - B.snd)^2)

def intersects_focus (l : line) (F : ℝ × ℝ) : Prop := l.eval F.fst = F.snd
def intersects_parabola (l : line) (p : parabola) (P : ℝ × ℝ) : Prop := l.eval P.fst = P.snd ∧ on_parabola p P

-- 1. Coordinates of point A
theorem point_A_coordinates (p : parabola) (l : line) (A : ℝ × ℝ) (F : ℝ × ℝ) 
  (h_focus : F = focus p) (h_p : p.val = 2) (h_intersect : intersects_focus l F) 
  (h_on_parabola : intersects_parabola l p A) (h_distance : distance A F = 4) : 
  A = (3, 2 * Real.sqrt 3) ∨ A = (3, -2 * Real.sqrt 3) := 
sorry

-- 2. Minimum length of segment AB
theorem minimum_length_AB (p : parabola) (l : line) (A B : ℝ × ℝ) (F : ℝ × ℝ) 
  (h_focus : F = focus p) (h_p : p.val = 2) (h_intersect_A : intersects_parabola l p A) 
  (h_intersect_B : intersects_parabola l p B) : 
  (distance A B = 4 ∨ ∀ AB. distance A B > 4) := 
sorry

end point_A_coordinates_minimum_length_AB_l816_816145


namespace find_angle_E_l816_816992

noncomputable def convex_hexagon :=
universe u
class Hexagon (α : Type u) :=
(is_convex : Prop)
(eq_sides : Prop)
(angles : α → Prop)

open Hexagon

axiom ABCDEF: Type
variables {A B C D E F : ABCDEF}

axiom Hex : Hexagon ABCDEF
axiom hex_convex : Hex.is_convex
axiom hex_eq_sides : Hex.eq_sides
axiom angle_A : Hex.angles (134 : ℝ)
axiom angle_B : Hex.angles (106 : ℝ)
axiom angle_C : Hex.angles (134 : ℝ)

theorem find_angle_E (hex_convex : Hex.is_convex) 
    (hex_eq_sides : Hex.eq_sides) 
    (angle_A : Hex.angles (134 : ℝ)) 
    (angle_B : Hex.angles (106 : ℝ)) 
    (angle_C : Hex.angles (134 : ℝ)) : 
    Hex.angles (134 : ℝ) :=
by
  -- Proof logic goes here
  sorry

end find_angle_E_l816_816992


namespace ludwig_weekly_earnings_l816_816785

theorem ludwig_weekly_earnings :
  (7 = 7) ∧
  (∀ day : ℕ, day ∈ {5, 6, 7} → (1 / 2) = 1 / 2) ∧
  (daily_salary = 10) →
  (weekly_earnings = 55) :=
by
  sorry

end ludwig_weekly_earnings_l816_816785


namespace perimeter_inequality_l816_816754

variable (r : ℝ) (P : Point)
variable (A1 A2 A3 A4 : Point)
variable (C1 : Circle P r) (C2 : Circle P (2 * r))
variable {B1 B2 B3 B4 : Point}

-- Definitions of segments given points
def segment_length (x y : Point) : ℝ := dist x y

-- Definitions of perimeters of quadrilaterals
def perimeter_quad (a b c d : Point) : ℝ :=
  (segment_length a b) + (segment_length b c) + (segment_length c d) + (segment_length d a)

theorem perimeter_inequality
  (hC1 : Inscribed A1 A2 A3 A4 C1)
  (hC2B1 : LineThrough A4 A1 ∩ C2 = B1)
  (hC2B2 : LineThrough A1 A2 ∩ C2 = B2)
  (hC2B3 : LineThrough A2 A3 ∩ C2 = B3)
  (hC2B4 : LineThrough A3 A4 ∩ C2 = B4) :
  2 * perimeter_quad A1 A2 A3 A4 ≤ perimeter_quad B1 B2 B3 B4 ∧
  (perimeter_quad B1 B2 B3 B4 = 2 * perimeter_quad A1 A2 A3 A4 ↔ 
   is_square A1 A2 A3 A4) := sorry

end perimeter_inequality_l816_816754


namespace problem_statement_l816_816430

def common_remainder_is_8 : Prop :=
  let x := 3374 in
  x % 9 = 8 ∧ x % 11 = 8 ∧ x % 17 = 8

theorem problem_statement : common_remainder_is_8 :=
sorry

end problem_statement_l816_816430


namespace quadratic_eq_rational_coeffs_with_root_l816_816958

theorem quadratic_eq_rational_coeffs_with_root (x : ℚ) (a b c : ℚ) (h1 : a = 1) (h2 : b = 6) (h3 : c = -11) (h4 : x = 2 * real.sqrt 5 - 3) :
    a * x^2 + b * x + c = 0 :=
by {
    sorry
}

end quadratic_eq_rational_coeffs_with_root_l816_816958


namespace vector_dot_self_l816_816689

variable (v : ℝ^3) -- Assuming v is in 3-dimensional space for concreteness

theorem vector_dot_self (h : ‖v‖ = 7) : v.dot v = 49 :=
by sorry

end vector_dot_self_l816_816689


namespace speed_conversion_l816_816462

noncomputable def kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

theorem speed_conversion (h : kmh_to_ms 1 = 1000 / 3600) :
  kmh_to_ms 1.7 = 0.4722 :=
by sorry

end speed_conversion_l816_816462


namespace sin_minus_cos_l816_816236

noncomputable def theta_condition (θ : ℝ) : Prop := (0 < θ) ∧ (θ < π / 2) ∧ (Real.tan θ = 1 / 3)

theorem sin_minus_cos (θ : ℝ) (hθ : theta_condition θ) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_minus_cos_l816_816236


namespace more_plastic_pipe_l816_816057

variable (m_copper m_plastic : Nat)
variable (total_cost cost_per_meter : Nat)

-- Conditions
variable (h1 : m_copper = 10)
variable (h2 : cost_per_meter = 4)
variable (h3 : total_cost = 100)
variable (h4 : m_copper * cost_per_meter + m_plastic * cost_per_meter = total_cost)

-- Proof that the number of more meters of plastic pipe bought compared to the copper pipe is 5
theorem more_plastic_pipe :
  m_plastic - m_copper = 5 :=
by
  -- Since proof is not required, we place sorry here.
  sorry

end more_plastic_pipe_l816_816057


namespace determine_values_l816_816667

-- Define variables and conditions
variable {x v w y z : ℕ}

-- Define the conditions
def condition1 := v * x = 8 * 9
def condition2 := y^2 = x^2 + 81
def condition3 := z^2 = 20^2 - x^2
def condition4 := w^2 = 8^2 + v^2
def condition5 := v * 20 = y * 8

-- Theorem to prove
theorem determine_values : 
  x = 12 ∧ y = 15 ∧ z = 16 ∧ v = 6 ∧ w = 10 :=
by
  -- Insert necessary logic or 
  -- produce proof steps here
  sorry

end determine_values_l816_816667


namespace football_throwing_distance_l816_816373

theorem football_throwing_distance 
  (T : ℝ)
  (yards_per_throw_at_T : ℝ)
  (yards_per_throw_at_80 : ℝ)
  (throws_on_Saturday : ℕ)
  (throws_on_Sunday : ℕ)
  (saturday_distance sunday_distance : ℝ)
  (total_distance : ℝ) :
  yards_per_throw_at_T = 20 →
  yards_per_throw_at_80 = 40 →
  throws_on_Saturday = 20 →
  throws_on_Sunday = 30 →
  saturday_distance = throws_on_Saturday * yards_per_throw_at_T →
  sunday_distance = throws_on_Sunday * yards_per_throw_at_80 →
  total_distance = saturday_distance + sunday_distance →
  total_distance = 1600 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end football_throwing_distance_l816_816373


namespace cos_315_eq_sqrt2_div_2_l816_816527

theorem cos_315_eq_sqrt2_div_2 :
  let θ := 315.0
  in θ = 360.0 - 45.0 ∧ (cos 45 * (π / 180)) = (Real.sqrt 2 / 2) →
  (cos (θ * (π / 180))) = (Real.sqrt 2 / 2) :=
by
  intros θ
  -- Non-necessary proof omitted
  sorry

end cos_315_eq_sqrt2_div_2_l816_816527


namespace range_of_m_l816_816359

noncomputable def f (x : ℝ) : ℝ := 4 * sin x * (sin (π / 4 + x / 2))^2 + cos (2 * x)

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (π / 6 ≤ x ∧ x ≤ 2 * π / 3) → |f x - m| < 2) ↔ (0 < m ∧ m < 5) :=
by
  sorry

end range_of_m_l816_816359


namespace franks_clock_midnight_l816_816979

variable (actualTimeRead : ℕ) [noncomputable]

/-- Define the conditions --/
def clock_initial_actual_time := 8 * 60  -- 8:00 AM in minutes
def clock_reported_time_pm := 4 * 60 + 42 -- 4:42 PM in minutes
def actual_time_pm := 17 * 60 -- 5:00 PM in minutes

def time_loss_rate := (actual_time_pm - clock_initial_actual_time) / (clock_initial_actual_time - clock_reported_time_pm)

/-- Define the proof statement: actual time is 12:33 AM when Frank's clock reads midnight --/
theorem franks_clock_midnight (actual_midnight_time : ℕ) [noncomputable] : 
  (clock_initial_actual_time + 16 * 60 + 33) = 993 → actual_midnight_time = 993 :=
by sorry

end franks_clock_midnight_l816_816979


namespace more_chickens_than_chicks_l816_816866

-- Let's define the given conditions
def total : Nat := 821
def chicks : Nat := 267

-- The statement we need to prove
theorem more_chickens_than_chicks : (total - chicks) - chicks = 287 :=
by
  -- This is needed for the proof and not part of conditions
  -- Add sorry as a placeholder for proof steps 
  sorry

end more_chickens_than_chicks_l816_816866


namespace dice_prob_part1_dice_prob_part2_l816_816816

theorem dice_prob_part1 (A : set (ℕ × ℕ)) (h : A = {(x, y) | x + y ∈ {2, 3, 4}}) :
  (∃ p : ℚ, p = 1 / 6 ∧ probability A p) := 
begin
  sorry
end

theorem dice_prob_part2 (B : set (ℕ × ℕ)) (p : ℕ × ℕ → ℝ) :
  (∃ m : ℕ, (∀ a b : ℕ, (a, b) ∈ B → a + b = m) ∧ probability B (6 / 36) ∧ m = 7) :=
begin
  sorry
end

end dice_prob_part1_dice_prob_part2_l816_816816


namespace problem_statement_l816_816893

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x

theorem problem_statement :
  (∀ x, f(-x) = f(x)) ∧
  (∃ p, ∀ x, f(x - p) = -f(x + p)) ∧
  (∀ {x y : ℝ}, -Real.pi / 2 < x → x < y → y < Real.pi / 2 → f x < f y) :=
by sorry

end problem_statement_l816_816893


namespace thirteen_y_minus_x_l816_816032

theorem thirteen_y_minus_x (x y : ℤ) (hx1 : x = 11 * y + 4) (hx2 : 2 * x = 8 * (3 * y) + 3) : 13 * y - x = 1 :=
by
  sorry

end thirteen_y_minus_x_l816_816032


namespace tangent_line_at_point_2_extreme_values_f2_l816_816182

noncomputable def f1 (x : ℝ) : ℝ := x^3 - (3/2 : ℝ) * x^2 + 1
noncomputable def f2 (x : ℝ) : ℝ := (1/3 : ℝ) * x^3 - (3/2 : ℝ) * x^2 + 1

-- Part (I)
theorem tangent_line_at_point_2 :
  let f := f1 in
  ∃ m b, m = 6 ∧ b = -9 ∧ (∀ x, x ≠ 2 → (f x - (m * (x - 2) + f 2)) / (x - 2) - 6 = 0) :=
sorry

-- Part (II)
theorem extreme_values_f2 :
  let f := f2 in
  (∃ x_max, x_max = 0 ∧ f x_max = 1) ∧
  (∃ x_min, x_min = 3 ∧ f x_min = -7/2) :=
sorry

end tangent_line_at_point_2_extreme_values_f2_l816_816182


namespace final_clothes_count_l816_816500

-- Define the conditions as functions or values in Lean
def initial_clothes := 300
def donate_first_orphanage (total_clothes : ℕ) := (5 * total_clothes) / 100
def donate_second_orphanage (total_clothes : ℕ) := 3 * donate_first_orphanage total_clothes
def bought_clothes := 20
def throw_away_percentage (remaining_clothes : ℕ) := (10 * remaining_clothes) / 100

-- Main theorem stating the answer
theorem final_clothes_count : 
  let remaining_after_donation := initial_clothes - donate_first_orphanage initial_clothes - donate_second_orphanage initial_clothes in
  let remaining_after_buying := remaining_after_donation + bought_clothes in
  remaining_after_buying - throw_away_percentage remaining_after_buying = 234 :=
by
  sorry

end final_clothes_count_l816_816500


namespace solve_x4_minus_16_eq_0_l816_816607

open Complex  -- Open the complex number notation

theorem solve_x4_minus_16_eq_0 :
  {x : ℂ | x^4 = 16} = {2, -2, 2 * Complex.I, -2 * Complex.I} :=
by sorry

end solve_x4_minus_16_eq_0_l816_816607


namespace value_of_a_l816_816294

theorem value_of_a (a b : ℤ) (h : (∀ x, x^2 - x - 1 = 0 → a * x^17 + b * x^16 + 1 = 0)) : a = 987 :=
by 
  sorry

end value_of_a_l816_816294


namespace ball_problem_l816_816316

theorem ball_problem (a : ℕ) (h1 : 3 / a = 0.25) : a = 12 :=
by sorry

end ball_problem_l816_816316


namespace unique_sum_of_squares_power_of_two_l816_816807

theorem unique_sum_of_squares_power_of_two (n : ℕ) :
  ∃! (a b : ℕ), 2^n = a^2 + b^2 := 
sorry

end unique_sum_of_squares_power_of_two_l816_816807


namespace sally_jane_difference_l816_816818

theorem sally_jane_difference: 
  let x := (2 / 3 : ℚ)
  let y := (3 / 2 : ℚ)
  let sally_result := x * y
  let jane_result := x + y
  jane_result - sally_result = (7/6 : ℚ) := 
by
  skip_proof -- this is an illustrative placeholder for the solution
  sorry

end sally_jane_difference_l816_816818


namespace stability_and_statistics_l816_816022

-- Define the data set
def dataSet : List ℤ := [1, 2, 5, 5, 5, 3, 3]

-- Variances of sets A and B
def varianceA : ℚ := 0.01
def varianceB : ℚ := 0.1

-- The Lean statement encompassing the required proofs
theorem stability_and_statistics :
  (∃ mode, mode ∈ dataSet ∧ (∀ x, count x dataSet <= count mode dataSet) ∧ mode ≠ 3) ∧
  (let mean := (dataSet.sum : ℚ) / dataSet.length in mean ≠ 3) ∧
  (varianceA < varianceB) :=
by
  sorry

end stability_and_statistics_l816_816022


namespace correct_distance_expression_l816_816840

theorem correct_distance_expression
  (d : ℝ → ℝ)
  (A O : ℝ)
  (t : ℝ)
  (h1 : dist A O = 5)
  (h2 : ∀ t, rot_unif_around_O t A O)
  (h3 : A_at_12_at_t0 A O)
  (h4 : period_second_hand = 60) :
  d(t) = 10 * abs (sin (π / 60 * t)) :=
sorry

end correct_distance_expression_l816_816840


namespace trig_inequality_l816_816809

theorem trig_inequality (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) :
  1 / 2 < (sqrt 3 / 2) * sin α + (1 / 2) * cos α ∧
  (sqrt 3 / 2) * sin α + (1 / 2) * cos α ≤ 1 :=
  sorry

end trig_inequality_l816_816809


namespace sin_minus_cos_theta_l816_816245

theorem sin_minus_cos_theta (θ : ℝ) (h₀ : 0 < θ ∧ θ < (π / 2)) 
  (h₁ : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -√10 / 5 :=
by
  sorry

end sin_minus_cos_theta_l816_816245


namespace find_good_pairs_count_l816_816970

open_locale polynomial

noncomputable def is_p_good (P : ℚ[X]) (p : ℕ) : Prop :=
∃ (a b c : ℕ), 0 ≤ a ∧ a < b ∧ b < c ∧ (c : ℚ) < p / 3 ∧ 
  p ∣ (P.eval a).num ∧ p ∣ (P.eval b).num ∧ p ∣ (P.eval c).num

noncomputable def polynomial_p_good_for_infinite_primes (P : ℚ[X]) : Prop :=
∃ᶠ p in filter.at_top, nat.prime p ∧ is_p_good P p

noncomputable def polynomial_good_pairs_count (n : ℕ) : Prop :=
∃ (r s : ℚ), let P := (X ^ 3 + 10 * X ^ 2 + polynomial.C r * X + polynomial.C s : ℚ[X]) in 
  polynomial_p_good_for_infinite_primes P ∧ 
  n = 12

theorem find_good_pairs_count : polynomial_good_pairs_count 12 :=
  sorry

end find_good_pairs_count_l816_816970


namespace trig_identity_l816_816282

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end trig_identity_l816_816282


namespace average_of_middle_three_numbers_l816_816389

/-- Given five distinct positive whole numbers such that their average is 5, 
    and the difference between the largest and smallest number is maximized.
    Prove that the average of the three middle numbers is 3. -/
theorem average_of_middle_three_numbers (a b c d e : ℕ) (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
    (h2 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0)
    (h3 : (a + b + c + d + e) = 25)
    (h4 : ∃ (w x : ℕ), (w ∈ {a, b, c, d, e} ∧ x ∈ {a, b, c, d, e}) ∧ (w - x = 14)) :
  ((a + b + c + d + e - max (max (max (max a b) c) d) (min (min (min (min a b) c) d))) / 3) = 3 := 
sorry

end average_of_middle_three_numbers_l816_816389


namespace sin_minus_cos_eq_l816_816231

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end sin_minus_cos_eq_l816_816231


namespace max_take_home_pay_at_5000_dollars_l816_816715

noncomputable def income_tax (x : ℕ) : ℕ :=
  if x ≤ 5000 then x * 5 / 100
  else 250 + 10 * ((x - 5000 / 1000) - 5) ^ 2

noncomputable def take_home_pay (y : ℕ) : ℕ :=
  y - income_tax y

theorem max_take_home_pay_at_5000_dollars : ∀ y : ℕ, take_home_pay y ≤ take_home_pay 5000 := by
  sorry

end max_take_home_pay_at_5000_dollars_l816_816715


namespace cos_315_deg_l816_816543

noncomputable def cos_315 : ℝ :=
  Real.cos (315 * Real.pi / 180)

theorem cos_315_deg : cos_315 = Real.sqrt 2 / 2 :=
by
  sorry

end cos_315_deg_l816_816543


namespace probability_complement_occurs_k_times_in_n_trials_l816_816714

theorem probability_complement_occurs_k_times_in_n_trials
  (p : ℝ) (n k : ℕ) :
  p ≥ 0 ∧ p ≤ 1 →
  (nat.choose n k * (1 - p)^k * p^(n - k)) = (nat.choose n k * (1 - p)^k * p^(n - k)) := 
sorry

end probability_complement_occurs_k_times_in_n_trials_l816_816714


namespace triangle_area_correct_l816_816625

noncomputable def triangle_area (a b c : ℝ × ℝ × ℝ) : ℝ :=
  let v1 := (b.1 - a.1, b.2 - a.2, b.3 - a.3)
  let v2 := (c.1 - a.1, c.2 - a.2, c.3 - a.3)
  let cross_product := (
    v1.2 * v2.3 - v1.3 * v2.2,
    v1.3 * v2.1 - v1.1 * v2.3,
    v1.1 * v2.2 - v1.2 * v2.1)
  let magnitude := Real.sqrt (cross_product.1^2 + cross_product.2^2 + cross_product.3^2)
  magnitude / 2

theorem triangle_area_correct :
  triangle_area (2, 1, -1) (1, 3, 0) (5, 2, 3) = 7 * Real.sqrt 3 / 2 := 
by
  sorry

end triangle_area_correct_l816_816625


namespace cos_315_eq_l816_816556

theorem cos_315_eq :
  let Q := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))
  in Q = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) →
     Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  intro hQ
  sorry

end cos_315_eq_l816_816556


namespace distinct_ordered_pairs_eq_49_l816_816174

open Nat

theorem distinct_ordered_pairs_eq_49 (a b : ℕ) (h1 : a + b = 50) (h2 : a > 0) (h3 : b > 0) :
  num_solutions (λ p : ℕ × ℕ, p.1 + p.2 = 50 ∧ p.1 > 0 ∧ p.2 > 0) = 49 :=
sorry

end distinct_ordered_pairs_eq_49_l816_816174


namespace sin_minus_cos_l816_816258

theorem sin_minus_cos (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < (Real.pi / 2)) (hθ3 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := by
  sorry

end sin_minus_cos_l816_816258


namespace gcd_n4_plus_27_n_plus_3_l816_816968

theorem gcd_n4_plus_27_n_plus_3 (n : ℕ) (h_pos : n > 9) : 
  gcd (n^4 + 27) (n + 3) = if n % 3 = 0 then 3 else 1 := 
by
  sorry

end gcd_n4_plus_27_n_plus_3_l816_816968


namespace cos_315_eq_l816_816562

theorem cos_315_eq :
  let Q := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))
  in Q = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) →
     Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  intro hQ
  sorry

end cos_315_eq_l816_816562


namespace total_tin_in_new_alloy_l816_816892

theorem total_tin_in_new_alloy (wA wB : ℝ) (rA_lead rA_tin rB_tin rB_copper : ℕ) :
  wA = 90 ∧ wB = 140 ∧ rA_lead = 3 ∧ rA_tin = 4 ∧ rB_tin = 2 ∧ rB_copper = 5 →
  let tin_A := (rA_tin : ℝ) / (rA_lead + rA_tin) * wA in
  let tin_B := (rB_tin : ℝ) / (rB_tin + rB_copper) * wB in
  tin_A + tin_B = 91.43 :=
by
  intros h
  sorry

end total_tin_in_new_alloy_l816_816892


namespace cos_315_eq_sqrt2_div_2_l816_816568

theorem cos_315_eq_sqrt2_div_2 : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
sorry

end cos_315_eq_sqrt2_div_2_l816_816568


namespace quadratic_equation_x_range_l816_816960

theorem quadratic_equation_x_range {x y : ℝ} 
  (h : 4 * y^2 + 6 * x * y + x + 10 = 0) : 
  x ≤ -17 / 9 ∨ x ≥ 7 / 3 :=
sory

end quadratic_equation_x_range_l816_816960


namespace angle_bisector_ratio_l816_816710

theorem angle_bisector_ratio
  (A B C D : Type)
  [linear_ordered_field A]
  [linear_ordered_field B]
  [linear_ordered_field C]
  [linear_ordered_field D]
  (AC CB AD DB : Type)
  (ratioAC_CB : ∀ x : Type,  AC = 2x ∧ CB = 3x)
  (angleBisectorTheorem : ∀ (x : Type), AD / DB = AC / CB) :
  AD / DB = 2 / 3 :=
by
  sorry

end angle_bisector_ratio_l816_816710


namespace integer_roots_of_polynomial_l816_816477

-- Define the polynomial with integer coefficients
def poly (b2 b1 : ℤ) := λ x : ℤ, x^3 + b2 * x^2 + b1 * x - 18

-- Define the possible integer roots of the polynomial
def possible_integer_roots := {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

-- Statement: Prove that the set of all possible integer roots of the polynomial
-- equals the given set of divisors of -18
theorem integer_roots_of_polynomial (b2 b1 : ℤ) :
  {x : ℤ | poly b2 b1 x = 0} ⊆ possible_integer_roots :=
by
  sorry

end integer_roots_of_polynomial_l816_816477


namespace least_integer_value_l816_816010

theorem least_integer_value 
  (x : ℤ) (h : |3 * x - 5| ≤ 22) : x = -5 ↔ ∃ (k : ℤ), k = -5 ∧ |3 * k - 5| ≤ 22 :=
by
  sorry

end least_integer_value_l816_816010


namespace range_of_m_l816_816692

theorem range_of_m (x y m : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 2*y = 4)
  (h_ineq : m^2 + (1/3)*m > 2/x + 1/(y+1)) : m ∈ Set.Ioo (-∞) (-4/3) ∪ Set.Ioo 1 (∞) :=
begin
  sorry
end

end range_of_m_l816_816692


namespace max_value_x_l816_816119

theorem max_value_x (x : ℝ) :
(6 + 5 * x + x^2) * sqrt (2 * x^2 - x^3 - x) ≤ 0 → x ≤ 1 := 
sorry

example : max_value_x 1 := 
by sorry

end max_value_x_l816_816119


namespace eccentricity_of_ellipse_l816_816159

axiom ellipse_def (a b: ℝ) (ha: a > 0) (hb: b > 0) (hab: a > b):
  ∃ x y: ℝ, (x/a) ^ 2 + (y/b)^ 2 = 1

axiom right_focus (a b: ℝ) (ha: a > 0) (hb: b > 0) (hab: a > b): 
  ∃ c: ℝ, c = (a ^ 2 - b ^ 2) ^ (1/2) ∧ c < a ∧ c > 0 

axiom line_intersect (a b c: ℝ) (ha: a > 0) (hb: b > 0) (hab: a > b) (hc: c = (a ^ 2 - b ^ 2) ^ (1/2)): 
  ∃ MN: ℝ, MN = 2 * (b ^ 2) / a ∧ MN = 3 * (a - c)

theorem eccentricity_of_ellipse (a b: ℝ) (ha: a > 0) (hb: b > 0) (hab: a > b) (MN: ℝ) (c: ℝ):
  2 * ((a ^ 2 - b ^ 2) / a ^ 2) - 3 * ((a ^ 2 - b ^ 2) ^ (1 / 2) / a) + 1 = 0 → 
  3 * (a - (a ^ 2 - b ^ 2) ^ (1 / 2)) = 2 * (b ^ 2) / a → 
  (a ^ 2 - b ^ 2) ^ (1 / 2)) / a = 1 / 2

end eccentricity_of_ellipse_l816_816159


namespace midpoint_intersection_l816_816375

structure Point where
  x : ℝ
  y : ℝ

def midpoint (A B : Point) : Point :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

def A : Point := ⟨2, -3⟩
def B : Point := ⟨14, 9⟩
def intersection_point : Point := ⟨8, 3⟩

theorem midpoint_intersection :
  midpoint A B = intersection_point :=
by
  sorry

end midpoint_intersection_l816_816375


namespace a_and_b_work_days_l816_816441

namespace WorkProblem

-- Definitions of the work rates as non-negative rational numbers
variables {A B C : ℚ}

-- The conditions given in the problem
def condition1 : Prop := B + C = 1/8
def condition2 : Prop := C + A = 1/12
def condition3 : Prop := C = 1/48

-- The statement that needs to be proved
theorem a_and_b_work_days
    (h1 : condition1)
    (h2 : condition2)
    (h3 : condition3) :
    A + B = 1/6 :=
by
  sorry -- proof is not required

end WorkProblem

end a_and_b_work_days_l816_816441


namespace linear_inequality_inequality_l816_816204

theorem linear_inequality_inequality (k : ℝ) (x : ℝ) : 
  (k - 1) * x ^ |k| + 3 ≥ 0 → is_linear x (k - 1) * x ^ |k| + 3 → k = -1 :=
by 
  sorry

end linear_inequality_inequality_l816_816204


namespace shells_per_friend_l816_816777

variables (J S C F : ℕ)

theorem shells_per_friend (hJ : J = 29) (hS : S = 17) (hC : C = 8) : 
    (J + S + C) / F = 54 / F :=
by
  rw [hJ, hS, hC]
  -- show intermediate steps if necessary or skip directly to sorry
  sorry

end shells_per_friend_l816_816777


namespace exists_two_distinct_sets_of_points_l816_816336

noncomputable def equilateral_triangle := 
  { vertices : Fin 3 → ℝ × ℝ //
    ∀ i j, i ≠ j → dist (vertices i) (vertices j) = 1 }

structure points_dividing_angles (T : equilateral_triangle) :=
  (P Q R : ℝ × ℝ)
  (P_cond : ∃ i, angle (T.vertices i, T.vertices ((i + 1) % 3), P) = π / 6)
  (Q_cond : ∃ i, angle (T.vertices i, T.vertices ((i + 2) % 3), Q) = π / 6)
  (R_cond : ∃ i, angle (T.vertices i, T.vertices ((i + 0) % 3), R) = π / 6)

theorem exists_two_distinct_sets_of_points
  (T : equilateral_triangle) :
  ∃ (S1 S2 : points_dividing_angles T),
    S1.P ≠ S2.P ∧ S1.Q ≠ S2.Q ∧ S1.R ≠ S2.R ∧
    dist S1.P S1.Q ≈ 0.268 ∧ dist S2.P S2.Q ≈ 0.366 :=
  sorry

end exists_two_distinct_sets_of_points_l816_816336


namespace sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816209

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816209


namespace find_angle_BAO_l816_816111

noncomputable def triangle_area (a b c : ℝ) (angle : ℝ) : ℝ :=
  0.5 * a * b * Real.sin angle

theorem find_angle_BAO (AB AO : ℝ) (angle_BAO : ℝ) (area_ratio : ℝ) (BO_OD_ratio : ℝ)
  (h_AB : AB = 15) (h_AO : AO = 8) (h_angle_BAO_gt_31 : angle_BAO > 31)
  (h_area_ratio : area_ratio = 0.5) (h_BO_OD_ratio : BO_OD_ratio = 0.5) :
  angle_BAO = 150 :=
by
  have area_AOB := triangle_area AO AB angle_BAO
  have area_AOD := triangle_area AO AO angle_BAO
  have h1 : area_AOB = area_ratio * area_AOD := by sorry
  have h2 : BO_OD_ratio = 0.5 := by sorry
  have h3 : area_AOB = 30 := by sorry
  sorry

end find_angle_BAO_l816_816111


namespace solve_inequality_l816_816148

theorem solve_inequality (a x : ℝ) (h1 : x ≥ a) (h2 : x ≥ 5 * a / 3) : 
  (a < -3 / 4 ∧ x ≥ a) ∨ 
  (-3 / 4 ≤ a ∧ a < -1 / 2 ∧ (a ≤ x ∧ x < 2 * a + 1 - real.sqrt (a + 3 / 4) ∨ x > 2 * a + 1 + real.sqrt (a + 3 / 4))) ∨ 
  (a ≥ -1 / 2 ∧ x > 2 * a + 1 + real.sqrt (a + 3 / 4)) :=
sorry

end solve_inequality_l816_816148


namespace sin_minus_cos_l816_816264

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end sin_minus_cos_l816_816264


namespace mr_greene_probability_l816_816792

noncomputable def probability_more_sons_or_daughters
  (children : ℕ) (twins : ℕ) (independent : ℕ) (p : ℚ) :=
  let total_combinations := 2 ^ independent
  let twins_combinations := 2
  let total_scenarios := total_combinations * twins_combinations
  let even_distribution := (choose independent (independent/2).to_nat) * 2
  let unequal_distribution := total_scenarios - even_distribution
  unequal_distribution / total_scenarios 

theorem mr_greene_probability :
  probability_more_sons_or_daughters 8 2 6 (49 / 64) = (49 : ℚ) / 64 :=
by
  unfold probability_more_sons_or_daughters
  sorry

end mr_greene_probability_l816_816792


namespace cone_height_90_deg_is_36_8_l816_816901

noncomputable def cone_height_volume (V : ℝ) (θ : ℝ) : ℝ :=
  if θ = π / 2 then
    let r := (3 * V / π)^(1/3) in r
  else
    0  -- Not valid if the angle isn't 90 degrees

theorem cone_height_90_deg_is_36_8 :
  cone_height_volume (16384 * π) (π / 2) = 36.8 :=
by
  sorry

end cone_height_90_deg_is_36_8_l816_816901


namespace each_tree_takes_one_square_foot_l816_816717

theorem each_tree_takes_one_square_foot (total_length : ℝ) (num_trees : ℕ) (gap_length : ℝ)
    (total_length_eq : total_length = 166) (num_trees_eq : num_trees = 16) (gap_length_eq : gap_length = 10) :
    (total_length - (((num_trees - 1) : ℝ) * gap_length)) / (num_trees : ℝ) = 1 :=
by
  rw [total_length_eq, num_trees_eq, gap_length_eq]
  sorry

end each_tree_takes_one_square_foot_l816_816717


namespace least_cost_l816_816387

noncomputable def total_length : ℝ := 10 * (6 / 12)
def cost_6_foot_rope : ℝ := 5
def cost_per_foot_6_foot_rope : ℝ := cost_6_foot_rope / 6
def cost_1_foot_rope : ℝ := 1.25
def total_cost_1_foot : ℝ := 5 * cost_1_foot_rope

theorem least_cost : min cost_6_foot_rope total_cost_1_foot = 5 := 
by sorry

end least_cost_l816_816387


namespace plane_intersection_perpendicular_l816_816074

theorem plane_intersection_perpendicular (α β γ : Plane) (T : Triangle) 
  (h1 : T.is_equilateral)
  (h2 : projection_orthogonal α β T.is_equilateral)
  (h3 : projection_orthogonal α γ T.is_equilateral) :
  is_perpendicular (line_of_intersection β α) (line_of_intersection β γ) := 
sorry

end plane_intersection_perpendicular_l816_816074


namespace finish_faster_three_times_l816_816413

-- Let a, b, and c be the respective hours each worker takes to finish their share of the work
variables (a b c : ℝ)

-- Assume all are positive values
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Define the total time taken when working in shifts
def total_time_shifts : ℝ := a + b + c

-- Define the completion condition for working in shifts
axiom work_shifts : total_time_shifts = a + b + c

-- Define the completion condition for working simultaneously
axiom work_simultaneous : total_time_shifts = total_time_shifts / 3

-- The theorem stating that they would finish the job 3 times faster if worked simultaneously
theorem finish_faster_three_times :
  (a + b + c) / ((a + b + c) / 3) = 3 :=
by sorry

end finish_faster_three_times_l816_816413


namespace problem_statement_l816_816180

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 2 else log 2 x

theorem problem_statement : f (f (-3/2)) = -1 :=
  sorry

end problem_statement_l816_816180


namespace trainB_destination_time_l816_816003

def trainA_speed : ℕ := 90
def trainB_speed : ℕ := 135
def trainA_time_after_meeting : ℕ := 9
def trainB_time_after_meeting (x : ℕ) : ℕ := 18 - 3 * x

theorem trainB_destination_time : (trainA_time_after_meeting, trainA_speed) = (9, 90) → 
  (trainB_speed, trainB_time_after_meeting 3) = (135, 3) := by
  sorry

end trainB_destination_time_l816_816003


namespace train_cross_time_is_nine_seconds_l816_816488

-- Define the speed of the train in km/hr and convert to m/s
def speed_km_hr := 30
def speed_m_s : ℝ := (speed_km_hr * 1000) / 3600

-- Define the length of the train in meters
def train_length := 75

-- Define the time it takes for the train to cross the pole
def train_cross_time : ℝ := train_length / speed_m_s

-- The statement that we want to prove
theorem train_cross_time_is_nine_seconds : train_cross_time = 9 := by
  sorry

end train_cross_time_is_nine_seconds_l816_816488


namespace sin_minus_cos_l816_816220

variable (θ : ℝ)
hypothesis (h1 : 0 < θ ∧ θ < π / 2)
hypothesis (h2 : Real.tan θ = 1 / 3)

theorem sin_minus_cos (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := 
by 
  sorry

end sin_minus_cos_l816_816220


namespace mean_score_is_82_l816_816795

noncomputable def mean_score 
  (M A m a : ℝ) 
  (hM : M = 90) 
  (hA : A = 75) 
  (hm : m / a = 4 / 5) : ℝ := 
  (M * m + A * a) / (m + a)

theorem mean_score_is_82 
  (M A m a : ℝ) 
  (hM : M = 90) 
  (hA : A = 75) 
  (hm : m / a = 4 / 5) : 
  mean_score M A m a hM hA hm = 82 := 
    sorry

end mean_score_is_82_l816_816795


namespace arithmetic_sequence_sum_l816_816891

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : ∃ d : ℝ, ∀ n, a (n + 1) = a n + d) 
  (h_condition : a 1 + a 3 + a 5 = 21) : 
  let S_5 := a 1 + a 2 + a 3 + a 4 + a 5 in
  S_5 = 35 := sorry

end arithmetic_sequence_sum_l816_816891


namespace smallest_number_is_C_l816_816368

def A : ℕ := 36
def B : ℕ := 27 + 5
def C : ℕ := 3 * 10
def D : ℕ := 40 - 3

theorem smallest_number_is_C :
  min (min A B) (min C D) = C :=
by
  -- Proof steps go here
  sorry

end smallest_number_is_C_l816_816368


namespace problem_f_f_1div9_eq_log3_2_l816_816668

def f : ℝ → ℝ
| x => if x > 0 then Real.log x / Real.log 3 else f (x + 2)

theorem problem_f_f_1div9_eq_log3_2 :
  f (f (1 / 9)) = Real.log 2 / Real.log 3 := by
  sorry

end problem_f_f_1div9_eq_log3_2_l816_816668


namespace sin_minus_cos_eq_l816_816228

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end sin_minus_cos_eq_l816_816228


namespace train_speed_correct_l816_816065

-- Define the problem conditions
def train_time : ℝ := 22.5
def train_length : ℝ := 250.02

-- Convert time to hours
def train_time_hours : ℝ := train_time / 3600

-- Convert length to kilometers
def train_length_km : ℝ := train_length / 1000

-- Define the speed to be proved
def train_speed_kmph : ℝ := train_length_km / train_time_hours

-- Proving the required speed
theorem train_speed_correct : train_speed_kmph = 40.0032 :=
by {
    -- The proof goes here
    sorry
}

end train_speed_correct_l816_816065


namespace trigonometric_identity_l816_816271

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end trigonometric_identity_l816_816271


namespace chi_square_test_not_significant_probability_winning_prize_probability_no_C_given_no_prize_l816_816039


-- Part 1: Chi-square test
theorem chi_square_test_not_significant :
  let a := 20
  let b := 75
  let c := 10
  let d := 45
  let n := 150
  let chi_square := (n * (a * d - b * c) ^ 2) / ( (a + b) * (c + d) * (a + c) * (b + d) )
  χ2 < 6.635 :=
by {
  sorry
}

-- Part 2: Probability Calculations

-- 1. Probability of winning a prize
theorem probability_winning_prize :
  let prob_a := (2 / 5 : ℝ)
  let prob_b := (2 / 5 : ℝ)
  let prob_c := (1 / 5 : ℝ)
  let prob_win := prob_a * prob_b * prob_c * 6
  prob_win = 24 / 125 :=
by {
  sorry
}

-- 2. Given not winning a prize, calculate the probability of not getting a C card
theorem probability_no_C_given_no_prize :
  let prob_no_c_given_no_prize := (64 / 125 : ℝ) / (1 - 24 / 125)
  prob_no_c_given_no_prize = 64 / 101 :=
by {
  sorry
}

end chi_square_test_not_significant_probability_winning_prize_probability_no_C_given_no_prize_l816_816039


namespace trigonometric_identity_l816_816270

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end trigonometric_identity_l816_816270


namespace coefficient_x5y2_in_expansion_l816_816663

theorem coefficient_x5y2_in_expansion :
  (∃ n : ℕ, (∑ i in (finset.range (n+1)), binomial n i * 2^(n-i) = 32) → 
    binomial 5 2 * binomial 3 1 * 2^2 = 120) :=
begin
  sorry,
end

end coefficient_x5y2_in_expansion_l816_816663


namespace sum_c_eq_n_square_minus_one_l816_816750

theorem sum_c_eq_n_square_minus_one (n k : ℕ)
  (a b : fin k.succ → ℕ)
  (h1 : a 0 = 1) (h2 : b 0 = 1)
  (h3 : a k = n) (h4 : b k = n)
  (h5 : ∀ i : fin k, (a i.succ, b i.succ) = (1 + a i, b i) ∨ (a i, 1 + b i))
  (c : fin k.succ → ℕ)
  (h6 : ∀ i : fin k, (c i.succ = a i.succ ∧ a i.succ ≠ a i) ∨ (c i.succ = b i.succ ∧ b i.succ ≠ b i)) :
  ∑ i in finset.range k.succ, c i = n^2 - 1 :=
by
  sorry

end sum_c_eq_n_square_minus_one_l816_816750


namespace no_polyhedron_with_fewer_than_six_edges_no_polyhedron_with_exactly_seven_edges_polyhedron_with_more_than_seven_edges_exist_l816_816444

-- Define what a polyhedron is and the necessary properties
def Polyhedron (P : Type) := nonempty (P → P → Prop)

-- Define properties of a tetrahedron and general polyhedron properties
def tetrahedron (P : Polyhedron) : Prop := 
  ∃ x y z w : P, 
  (x ≠ y) ∧ 
  (x ≠ z) ∧ 
  (x ≠ w) ∧ 
  (y ≠ z) ∧ 
  (y ≠ w) ∧ 
  (z ≠ w) ∧
  has_edges x y z w -- Assuming has_edges denotes the edges of the tetrahedron

-- The smallest polygonal face a polyhedron can have is a triangle
def triangular_face (P : Polyhedron) : Prop := 
  ∀ (F : P → Prop), 
  ∃ a b c : P, 
  (a ≠ b) ∧ 
  (b ≠ c) ∧ 
  (c ≠ a) ∧
  is_face a b c -- Assuming is_face denotes triangular faces

-- Additional properties related to the question
def non_degenerate (P : Polyhedron) : Prop := 
  ∀ (F : P → Prop), is_non_degenerate F -- is_non_degenerate is a placeholder for the specific non-degeneracy requirement

def additional_vertices (P : Polyhedron) : Prop := 
  ∃ A B : P, 
  ¬ collinear A B -- Placeholder definition to denote vertices not in the same plane as any face

-- Now, write the Lean 4 statements for the results to be proven
theorem no_polyhedron_with_fewer_than_six_edges (P : Polyhedron)
  (tetra : tetrahedron P) 
  (tri_face : triangular_face P)
  (non_deg : non_degenerate P)
  (add_vert : additional_vertices P) :
  ¬ ∃ Q : Polyhedron, edge_count Q < 6 :=
sorry

theorem no_polyhedron_with_exactly_seven_edges (P : Polyhedron)
  (tri_face : triangular_face P)
  (non_deg : non_degenerate P)
  (add_vert : additional_vertices P) :
  ¬ ∃ Q : Polyhedron, edge_count Q = 7 :=
sorry

theorem polyhedron_with_more_than_seven_edges_exist (P : Polyhedron)
  (tri_face : triangular_face P)
  (non_deg : non_degenerate P)
  (add_vert : additional_vertices P) :
  ∀ n > 7, ∃ Q : Polyhedron, edge_count Q = n :=
sorry

end no_polyhedron_with_fewer_than_six_edges_no_polyhedron_with_exactly_seven_edges_polyhedron_with_more_than_seven_edges_exist_l816_816444


namespace vegetables_harvest_problem_l816_816867

theorem vegetables_harvest_problem
  (same_area : ∀ (a b : ℕ), a = b)
  (first_field_harvest : ℕ := 900)
  (second_field_harvest : ℕ := 1500)
  (less_harvest_per_acre : ∀ (x : ℕ), x - 300 = y) :
  x = y ->
  900 / x = 1500 / (x + 300) :=
by
  sorry

end vegetables_harvest_problem_l816_816867


namespace sum_geom_series_l816_816945

variable (n : ℕ) (a₁ q : ℝ)
variable (h : q ≠ 1)

theorem sum_geom_series (n : ℕ) (a₁ q : ℝ) (h : q ≠ 1) : 
  let Sₙ := a₁ * (1 - q^n) / (1 - q) in
  Sₙ = a₁ * (1 - q^n) / (1 - q) :=
by
  sorry

end sum_geom_series_l816_816945


namespace subset_exists_l816_816769

theorem subset_exists (p : ℕ) (hp : Nat.Prime p) (A : Finset ℕ) (hA : A.card = p - 1) 
  (hA_div : ∀ a ∈ A, ¬ p ∣ a) :
  ∀ n ∈ Finset.range p, ∃ B ⊆ A, (B.sum id) % p = n :=
by
  -- Proof goes here
  sorry

end subset_exists_l816_816769


namespace find_f_zero_l816_816701

theorem find_f_zero (f : ℝ → ℝ) (h : ∀ x, f ((x + 1) / (x - 1)) = x^2 + 3) : f 0 = 4 :=
by
  -- The proof goes here.
  sorry

end find_f_zero_l816_816701


namespace bridge_length_l816_816064

/-- 
A train 145 metres long, travelling at 45 km/hr, can cross a bridge in 30 seconds. 
Prove that the length of the bridge is 230 meters.
-/
theorem bridge_length (train_length : ℝ) (speed_kmph : ℝ) (time_sec : ℝ) (bridge_length : ℝ) 
  (h1 : train_length = 145) 
  (h2 : speed_kmph = 45) 
  (h3 : time_sec = 30) 
  (h4 : bridge_length = 230) : 
  let speed_mps := speed_kmph * 1000 / 3600 in
  let total_distance := speed_mps * time_sec in
  let calculated_bridge_length := total_distance - train_length in
  calculated_bridge_length = bridge_length := 
by 
  sorry

end bridge_length_l816_816064


namespace sin_minus_cos_l816_816237

noncomputable def theta_condition (θ : ℝ) : Prop := (0 < θ) ∧ (θ < π / 2) ∧ (Real.tan θ = 1 / 3)

theorem sin_minus_cos (θ : ℝ) (hθ : theta_condition θ) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_minus_cos_l816_816237


namespace find_x_2018_l816_816322

def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def x : ℕ → ℕ
| 0 => 0  -- This is just to handle x_0, we know x_1 = 1.
| 1 => 1
| k+2 => x (k+1) + 1 - 4 * (floor ((k + 1 : ℝ) / 4) - floor ((k : ℝ) / 4))

theorem find_x_2018 : x 2018 = 2 :=
by
  sorry

end find_x_2018_l816_816322


namespace cost_of_fencing_theorem_l816_816857

noncomputable def cost_of_fencing (area : ℝ) (ratio_length_width : ℝ) (cost_per_meter_paise : ℝ) : ℝ :=
  let width := (area / (ratio_length_width * 2 * ratio_length_width * 3)).sqrt
  let length := ratio_length_width * 3 * width
  let perimeter := 2 * (length + width)
  let cost_per_meter_rupees := cost_per_meter_paise / 100
  perimeter * cost_per_meter_rupees

theorem cost_of_fencing_theorem :
  cost_of_fencing 3750 3 50 = 125 :=
by
  sorry

end cost_of_fencing_theorem_l816_816857


namespace zero_point_in_interval_l816_816585

noncomputable def f (x : ℝ) : ℝ := real.log x / real.log 2 + x

theorem zero_point_in_interval : 
  ∃ x : ℝ, x ∈ set.Ioo (1 / 2) 1 ∧ f x = 0 :=
sorry

end zero_point_in_interval_l816_816585


namespace range_of_a_l816_816699

def decreasing_range (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ 4 → y ≤ 4 → x < y → (x^2 + 2 * (a - 1) * x + 2) ≥ (y^2 + 2 * (a - 1) * y + 2)

theorem range_of_a (a : ℝ) : decreasing_range a ↔ a ≤ -3 := 
  sorry

end range_of_a_l816_816699


namespace probability_eta_geq_2_l816_816775

noncomputable def binomialPGeq (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  1 - (Finset.range k).sum (λ i, (nat.choose n i) * (p^i) * ((1 - p)^(n - i)))

theorem probability_eta_geq_2 (p : ℝ) (η ξ : ℕ → ℕ → ℝ) (hξ : ∀ a b, ξ a b = (nat.choose 2 a) * (p^a) * ((1 - p)^(2 - a)))
  (hη : ∀ a b, η a b = (nat.choose 3 a) * (p^a) * ((1 - p)^(3 - a))) 
  (h5_9 : binomialPGeq 2 p 1 = 5/9) :
  binomialPGeq 3 p 2 = 7/27 :=
by sorry

end probability_eta_geq_2_l816_816775


namespace pie_difference_l816_816871

theorem pie_difference (s1 s3 : ℚ) (h1 : s1 = 7/8) (h3 : s3 = 3/4) :
  s1 - s3 = 1/8 :=
by
  sorry

end pie_difference_l816_816871


namespace second_smallest_packs_of_hot_dogs_l816_816955

theorem second_smallest_packs_of_hot_dogs :
  ∃ n : ℤ, 12 * n ≡ 6 [MOD 8] ∧ ∀ m : ℤ, 12 * m ≡ 6 [MOD 8] → 0 < m ∧ m < 4 → n = 4 :=
sorry

end second_smallest_packs_of_hot_dogs_l816_816955


namespace sin_minus_cos_l816_816222

variable (θ : ℝ)
hypothesis (h1 : 0 < θ ∧ θ < π / 2)
hypothesis (h2 : Real.tan θ = 1 / 3)

theorem sin_minus_cos (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := 
by 
  sorry

end sin_minus_cos_l816_816222


namespace duration_of_resulting_video_l816_816930

theorem duration_of_resulting_video 
    (vasya_walk_time : ℕ) (petya_walk_time : ℕ) 
    (sync_meet_point : ℕ) :
    vasya_walk_time = 8 → petya_walk_time = 5 → sync_meet_point = sync_meet_point → 
    (vasya_walk_time - sync_meet_point + petya_walk_time) = 5 :=
by
  intros
  sorry

end duration_of_resulting_video_l816_816930


namespace cos_315_deg_l816_816547

noncomputable def cos_315 : ℝ :=
  Real.cos (315 * Real.pi / 180)

theorem cos_315_deg : cos_315 = Real.sqrt 2 / 2 :=
by
  sorry

end cos_315_deg_l816_816547


namespace total_wheels_in_garage_l816_816417

theorem total_wheels_in_garage :
  let cars := 2,
      wheels_per_car := 4,
      riding_lawnmower := 1,
      wheels_per_lawnmower := 4,
      bicycles := 3,
      wheels_per_bicycle := 2,
      tricycle := 1,
      wheels_per_tricycle := 3,
      unicycle := 1,
      wheels_per_unicycle := 1 in
  cars * wheels_per_car + riding_lawnmower * wheels_per_lawnmower + bicycles * wheels_per_bicycle + tricycle * wheels_per_tricycle + unicycle * wheels_per_unicycle = 22 := by
  sorry

end total_wheels_in_garage_l816_816417


namespace triangles_and_midpoints_l816_816422

noncomputable def triangle_area_equilateral (s : ℝ) : ℝ := (sqrt 3 / 4) * s^2

noncomputable def midpoint {P : Type*} [metric_space P] (a b : P) : P := (1/2:ℝ) • a + (1/2:ℝ) • b

theorem triangles_and_midpoints
  (A B C D E : Type*)
  [metric_space A]
  [metric_space B]
  [metric_space C]
  [metric_space D]
  [metric_space E]
  (side_length : ℝ)
  (s ≡ side_length)
  (hA : dist A B = side_length)
  (hB : dist B C = side_length)
  (hC : dist C D = side_length)
  (hD : dist D E = side_length)
  (hE : dist E A = side_length)
  (hAC : dist A C = side_length)
  (hCE : dist C E = side_length)
  (hBD : ∠ BCD < π / 3)
  (K := midpoint A C)
  (L := midpoint C E)
  (M := midpoint B D)
  (hKLM : triangle_area_equilateral (dist K L) = sqrt 3 / 5) :
  dist B D =  (2 - sqrt 3) / sqrt 5 :=
sorry

end triangles_and_midpoints_l816_816422


namespace cos_315_proof_l816_816550

noncomputable def cos_315_eq : Prop :=
  let theta := 45 in
  let angle := 360 - theta in
  ∃ c : real, (cos 45) = (sqrt 2 / 2) ∧ (cos (360 - theta) = cos theta) ∧ 315 = angle ∧ (forall a : real, cos a = cos (360 - a)) ∧ (315 ∈ (4th quadrant)) ∧ (cos 315 = cos 45)

theorem cos_315_proof : (cos 315 = sqrt 2 / 2) :=
begin
  -- We start by defining basic facts
  -- 315 = 360 - 45
  -- Cosine is positive in the fourth quadrant, known value \cos 45 = sqrt 2 / 2
  -- Cosine property \cos (360 - \theta) = \cos \theta
  let theta := 45,
  let angle := 360 - theta,
  have H1 : cos theta = sqrt 2 / 2,
  {
    -- known fact
    sorry,
  },
  have H2 : cos (360 - theta) = cos theta,
  {
    -- cosine property
    sorry,
  },
  have H3 : angle = 315,
  {
    -- subtraction identity
    sorry,
  },
  rw H3,
  rw H2,
  rw H1,
  refl,
end

end cos_315_proof_l816_816550


namespace cars_fell_in_lot_l816_816411

theorem cars_fell_in_lot (initial_cars went_out_cars came_in_cars final_cars: ℕ) (h1 : initial_cars = 25) 
    (h2 : went_out_cars = 18) (h3 : came_in_cars = 12) (h4 : final_cars = initial_cars - went_out_cars + came_in_cars) :
    initial_cars - final_cars = 6 :=
    sorry

end cars_fell_in_lot_l816_816411


namespace num_cars_to_take_l816_816395

-- Definitions of the conditions
def num_cars : ℕ := c
def num_taxis : ℕ := 6
def num_vans : ℕ := 2
def people_per_car : ℕ := 4
def people_per_taxi : ℕ := 6
def people_per_van : ℕ := 5
def total_people : ℕ := 58

-- Definition of the proof
theorem num_cars_to_take (c : ℕ) : 
  4 * c + 6 * num_taxis + 2 * num_vans = total_people → c = 3 := by
  sorry

end num_cars_to_take_l816_816395


namespace focus_of_parabola_l816_816090

theorem focus_of_parabola (x y p : ℝ) (h : y = 2 * x^2) (hp : 0 = x) : p = 1/8 :=
by 
  -- We could follow the same transformation steps here to prove the coordinates
  -- but as per the instruction, we only need to provide the theorem statement
  sorry

-- Translate the problem statement into the Lean theorem
example : focus_of_parabola 0 (1/8) := begin
  -- Proof will be performed here, but we won't provide it now
  sorry
end

end focus_of_parabola_l816_816090


namespace lateral_area_cylinder_l816_816903

-- Definitions of the given conditions
def base_radius : ℝ := 2
def height : ℝ := 4

-- The statement of the problem: proving the lateral area of the cylinder
theorem lateral_area_cylinder (r h : ℝ) (S : ℝ) : 
  r = base_radius → h = height → S = 2 * Real.pi * r * h → S = 16 * Real.pi :=
by
  intros hr hh hs
  rw [hr, hh] at hs
  exact hs

end lateral_area_cylinder_l816_816903


namespace q_is_false_given_conditions_l816_816302

theorem q_is_false_given_conditions
  (h₁: ¬(p ∧ q) = true) 
  (h₂: ¬¬p = true) 
  : q = false := 
sorry

end q_is_false_given_conditions_l816_816302


namespace each_friend_receives_9_puffs_l816_816079

theorem each_friend_receives_9_puffs :
  ∀ (initial_puffs : ℕ) (puffs_to_mom : ℕ) (puffs_to_sister : ℕ) (puffs_to_grandmother : ℕ) (puffs_to_dog : ℕ) (friends : ℕ),
  initial_puffs = 40 →
  puffs_to_mom = 3 →
  puffs_to_sister = 3 →
  puffs_to_grandmother = 5 →
  puffs_to_dog = 2 →
  friends = 3 →
  (initial_puffs - (puffs_to_mom + puffs_to_sister + puffs_to_grandmother + puffs_to_dog)) / friends = 9 :=
by
  intros initial_puffs puffs_to_mom puffs_to_sister puffs_to_grandmother puffs_to_dog friends
  assume h1 h2 h3 h4 h5 h6
  sorry

end each_friend_receives_9_puffs_l816_816079


namespace abs_inequality_solution_l816_816402

theorem abs_inequality_solution (x : ℝ) : (|3 - x| < 4) ↔ (-1 < x ∧ x < 7) :=
by
  sorry

end abs_inequality_solution_l816_816402


namespace triangle_area_ratio_l816_816664

variable (a b : ℕ → ℝ)
variable (S T : ℕ → ℝ)
variable (n : ℕ)

-- Given conditions
axiom h1 : ∀ n, S n = n * (a 1 + a n) / 2
axiom h2 : ∀ n, T n = n * (b 1 + b n) / 2
axiom h3 : ∀ n, S n / T n = (3 * n) / (2 * n + 9)

-- Prove the required ratio when n = 5
theorem triangle_area_ratio (h1 h2 h3) : 
  ∀ a b S T, (n = 5) → S (triangle_area_ratio.triangle_area_ratio_axiom.h1_tr_s.diff_in h5) / T (triangle_area_ratio.triangle_area_ratio_axiom.h2_tr_s.diff_in h5) = 15 / 19 :=
sorry

end triangle_area_ratio_l816_816664


namespace suitable_menu_fraction_l816_816096

noncomputable def total_dishes (vegetarian_dishes : ℕ) (fraction : ℚ) : ℕ := 
  vegetarian_dishes / fraction

def count_gluten_free (vegetarian_dishes : ℕ) (gluten_dishes : ℕ) : ℕ := 
  vegetarian_dishes - gluten_dishes

def count_nut_free (vegetarian_dishes : ℕ) (nut_dishes : ℕ) : ℕ := 
  vegetarian_dishes - nut_dishes

def count_safe_dishes (gluten_free : ℕ) (nut_free : ℕ) : ℕ := 
  gluten_free - nut_free

theorem suitable_menu_fraction
  (vegetarian_dishes : ℕ) 
  (fraction_of_menu : ℚ)
  (gluten_dishes : ℕ)
  (nut_dishes : ℕ)
  (h1 : vegetarian_dishes % fraction_of_menu.denom = 0)
  (h2 : gluten_dishes ≤ vegetarian_dishes)
  (h3 : nut_dishes ≤ vegetarian_dishes)
  (h4 : gluten_dishes + nut_dishes ≥ vegetarian_dishes) :
  count_safe_dishes (count_gluten_free vegetarian_dishes gluten_dishes) (count_nut_free vegetarian_dishes nut_dishes) / total_dishes vegetarian_dishes fraction_of_menu = 0 :=
by
  sorry

end suitable_menu_fraction_l816_816096


namespace solution_set_l816_816144

noncomputable def f : ℝ → ℝ := sorry

axiom f_increasing : ∀ a b : ℝ, a < b → f a < f b 
axiom f_condition : ∀ x : ℝ, f x + 2 > f' x
axiom f_at_zero : f 0 = 1

theorem solution_set :
  { x : ℝ | ∀ y : ℝ, f y = f x → ln (f x + 2) - ln 3 > x } = { x | x < 0 } :=
sorry

end solution_set_l816_816144


namespace perfect_square_divisors_sum_l816_816129

theorem perfect_square_divisors_sum:
  let is_perfect_square (x : Int) := ∃ (y : Int), y^2 = x
  let divides (a b : Int) := ∃ (k : Int), b = a * k
  let n_values := {n : Int | is_perfect_square (n^2 - 17 * n + 72) ∧ divides n 15}
  sum n_values = 15 :=
by
  sorry

end perfect_square_divisors_sum_l816_816129


namespace difference_five_three_numbers_specific_number_condition_l816_816131

def is_five_three_number (A : ℕ) : Prop :=
  let a := A / 1000
  let b := (A % 1000) / 100
  let c := (A % 100) / 10
  let d := A % 10
  a = 5 + c ∧ b = 3 + d

def M (A : ℕ) : ℕ :=
  let a := A / 1000
  let b := (A % 1000) / 100
  let c := (A % 100) / 10
  let d := A % 10
  a + c + 2 * (b + d)

def N (A : ℕ) : ℕ :=
  let b := (A % 1000) / 100
  b - 3

noncomputable def largest_five_three_number := 9946
noncomputable def smallest_five_three_number := 5300

theorem difference_five_three_numbers :
  largest_five_three_number - smallest_five_three_number = 4646 := by
  sorry

noncomputable def specific_five_three_number := 5401

theorem specific_number_condition {A : ℕ} (hA : is_five_three_number A) :
  (M A) % (N A) = 0 ∧ (M A) / (N A) % 5 = 0 → A = specific_five_three_number := by
  sorry

end difference_five_three_numbers_specific_number_condition_l816_816131


namespace sin_minus_cos_theta_l816_816246

theorem sin_minus_cos_theta (θ : ℝ) (h₀ : 0 < θ ∧ θ < (π / 2)) 
  (h₁ : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -√10 / 5 :=
by
  sorry

end sin_minus_cos_theta_l816_816246


namespace fewest_additional_seats_l816_816321

theorem fewest_additional_seats (total_seats : ℕ) (initially_occupied : ℕ):
  total_seats = 150 → initially_occupied = 2 → ∃ (additional_occupied : ℕ), additional_occupied = 49 :=
by
  intros total_seats_eq initially_occupied_eq
  use 49
  sorry

end fewest_additional_seats_l816_816321


namespace eccentricity_of_hyperbola_l816_816164

noncomputable def hyperbola_asymptote_eccentricity (a b : ℝ) (h1 : 4 * a = 3 * b) : ℝ :=
  let c := (a^2 + b^2).sqrt in
  c / a

theorem eccentricity_of_hyperbola (a b : ℝ) (h1 : 4 * a = 3 * b) :
  hyperbola_asymptote_eccentricity a b h1 = 5 / 3 :=
by
  sorry

end eccentricity_of_hyperbola_l816_816164


namespace cos_315_eq_sqrt2_div2_l816_816535

theorem cos_315_eq_sqrt2_div2 : Real.cos (315 * Real.pi / 180) = sqrt 2 / 2 := 
sorry

end cos_315_eq_sqrt2_div2_l816_816535


namespace trig_eq_solutions_l816_816128

theorem trig_eq_solutions (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  3 * Real.sin x = 1 + Real.cos (2 * x) ↔ x = Real.pi / 6 ∨ x = 5 * Real.pi / 6 :=
by
  sorry

end trig_eq_solutions_l816_816128


namespace opposite_event_of_hitting_target_at_least_once_is_both_shots_miss_l816_816912

/-- A person is shooting at a target, firing twice in succession. 
    The opposite event of "hitting the target at least once" is "both shots miss". -/
theorem opposite_event_of_hitting_target_at_least_once_is_both_shots_miss :
  ∀ (A B : Prop) (hits_target_at_least_once both_shots_miss : Prop), 
    (hits_target_at_least_once → (A ∨ B)) → (both_shots_miss ↔ ¬hits_target_at_least_once) ∧ 
    (¬(A ∧ B) → both_shots_miss) :=
by
  sorry

end opposite_event_of_hitting_target_at_least_once_is_both_shots_miss_l816_816912


namespace solve_x4_eq_16_l816_816613

theorem solve_x4_eq_16 (x : ℂ) : x^4 - 16 = 0 ↔ x = 2 ∨ x = -2 ∨ x = 2 * complex.I ∨ x = -2 * complex.I :=
by
  sorry

end solve_x4_eq_16_l816_816613


namespace cos_315_eq_sqrt2_div_2_l816_816528

theorem cos_315_eq_sqrt2_div_2 :
  let θ := 315.0
  in θ = 360.0 - 45.0 ∧ (cos 45 * (π / 180)) = (Real.sqrt 2 / 2) →
  (cos (θ * (π / 180))) = (Real.sqrt 2 / 2) :=
by
  intros θ
  -- Non-necessary proof omitted
  sorry

end cos_315_eq_sqrt2_div_2_l816_816528


namespace distance_to_the_place_l816_816471

-- Variables
variables (D : ℝ) (t1 t2 : ℝ)

-- Given conditions
def still_water_speed := 5 -- kmph
def current_speed_to := 1 -- kmph
def current_speed_back := 2 -- kmph
def total_time := 1 -- hour

-- Effective speeds
def speed_to := still_water_speed - current_speed_to -- 4 kmph
def speed_back := still_water_speed + current_speed_back -- 7 kmph

-- Equations based on conditions
def distance_to := speed_to * t1
def distance_back := speed_back * t2

def total_time_condition := t1 + t2 = total_time
def distance_condition := distance_to = distance_back

-- Proof statement
theorem distance_to_the_place (H1 : total_time_condition)
                              (H2 : distance_condition) :
    D = 28 / 11 := 
begin
    sorry
end

end distance_to_the_place_l816_816471


namespace sin_minus_cos_l816_816241

noncomputable def theta_condition (θ : ℝ) : Prop := (0 < θ) ∧ (θ < π / 2) ∧ (Real.tan θ = 1 / 3)

theorem sin_minus_cos (θ : ℝ) (hθ : theta_condition θ) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_minus_cos_l816_816241


namespace sin_minus_cos_eq_l816_816224

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end sin_minus_cos_eq_l816_816224


namespace sphere_surface_area_ratio_l816_816999

noncomputable def ratio_of_surface_areas (a : ℝ) (r : ℝ) (R : ℝ) :=
  (5 : ℝ)

noncomputable def given_conditions (a r R : ℝ) : Prop :=
  r = (sqrt 3 / 6) * a ∧
  R^2 = ( (sqrt 3 / 3 * a) ^ 2 + r^2 ) ∧
  R^2 = 5 * r^2

theorem sphere_surface_area_ratio (a r R : ℝ) (h : given_conditions a r R) : 
  ratio_of_surface_areas a r R = 5 :=
by
  sorry

end sphere_surface_area_ratio_l816_816999


namespace complex_number_solution_count_l816_816121

noncomputable def complexNumberCount : ℕ :=
  let z_satisfies (z : ℂ) : Prop := (|z| = 1) ∧ ((z ^ (5!)) - (z ^ (4!)) ∈ ℝ)
  have h : set.countable {z : ℂ | z_satisfies z} := sorry,
  set.finite.to_finset {z : ℂ | z_satisfies z}.finiteness

theorem complex_number_solution_count : complexNumberCount = 23 :=
  sorry

end complex_number_solution_count_l816_816121


namespace sum_of_altitudes_of_triangle_l816_816849

theorem sum_of_altitudes_of_triangle (hline : ∀ x y : ℝ, 12 * x + 5 * y = 60) :
  ∑ a in {5, 12, 60 / 13}.toFinset, a = 281 / 13 := 
sorry

end sum_of_altitudes_of_triangle_l816_816849


namespace max_y_times_z_minus_x_is_sqrt_7_l816_816881

noncomputable def max_y_times_z_minus_x (x y z : ℝ) (h1 : x^2 + z^2 = 1) (h2 : y^2 + 2 * y * (x + z) = 6) : ℝ :=
  y * (z - x)

theorem max_y_times_z_minus_x_is_sqrt_7 (x y z : ℝ) (h1 : x^2 + z^2 = 1) (h2 : y^2 + 2 * y * (x + z) = 6) :
  max_y_times_z_minus_x x y z h1 h2 ≤ sqrt 7 :=
sorry

end max_y_times_z_minus_x_is_sqrt_7_l816_816881


namespace calculate_difference_l816_816773

noncomputable def h := {2, 6, 10, 15, 17, 31, 35, 39, 50, 56, 70}
noncomputable def a := 7
noncomputable def b := 9

noncomputable def mean (s : Set ℝ) : ℝ :=
  (s.toFinset.sum id) / s.toFinset.card

noncomputable def median (s : Set ℝ) : ℝ :=
  let l := s.toList.qsort (≤)
  l.get (l.length / 2)

theorem calculate_difference :
  ((mean h * a) - b) - (median h - (a / b)) = 171.4141414141 :=
by
  sorry

end calculate_difference_l816_816773


namespace compare_abc_l816_816767

open Real

theorem compare_abc
  (a b c : ℝ)
  (ha : 0 < a ∧ a < π / 2)
  (hb : 0 < b ∧ b < π / 2)
  (hc : 0 < c ∧ c < π / 2)
  (h1 : cos a = a)
  (h2 : sin (cos b) = b)
  (h3 : cos (sin c) = c) :
  c > a ∧ a > b :=
sorry

end compare_abc_l816_816767


namespace sin_minus_cos_l816_816223

variable (θ : ℝ)
hypothesis (h1 : 0 < θ ∧ θ < π / 2)
hypothesis (h2 : Real.tan θ = 1 / 3)

theorem sin_minus_cos (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := 
by 
  sorry

end sin_minus_cos_l816_816223


namespace ellipse_equation_slope_range_x_axis_intersect_l816_816168

variable (a b x y : ℝ)

-- Given conditions
def ellipse (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def eccentricity (a b : ℝ) : Prop :=
  (a > b) ∧ (a > 0) ∧ (b > 0) ∧ (a^2 - b^2 = (a * sqrt (3/4))^2)

def tangent_to_line (r : ℝ) : Prop :=
  r = 1 / sqrt (1 + 1)

variable (P M N E : ℝ × ℝ)

def symmetric_points (M N : ℝ × ℝ) : Prop :=
  M.2 = -N.2

def line_through_points (P1 P2 : ℝ × ℝ) (k : ℝ) : Prop :=
  P2.2 - P1.2 = k * (P2.1 - P1.1)

noncomputable def line_tangency_point : ℝ × ℝ :=
  (1, 1)

variable [h_symm : symmetric_points M N]
variable [h_tangent : tangent_to_line 1]

theorem ellipse_equation : ∃ a b, a = 2 * b ∧ b = 1 ∧ ellipse x y a b :=
by
  sorry

theorem slope_range : 
  ∀ k, k ≠ 0 ∧ (- sqrt 3 / 6 < k ∧ k < 0 ∨ 0 < k ∧ k < sqrt 3 / 6) :=
by
  sorry

theorem x_axis_intersect : 
  ∃ x (P M N E : ℝ × ℝ), 
  symmetric_points M N ∧ 
  line_through_points P N k ∧ 
  line_through_points M E k ∧ 
  E.1 = 1 ∧ E.2 = 0 :=
by
  sorry

end ellipse_equation_slope_range_x_axis_intersect_l816_816168


namespace cos_315_proof_l816_816553

noncomputable def cos_315_eq : Prop :=
  let theta := 45 in
  let angle := 360 - theta in
  ∃ c : real, (cos 45) = (sqrt 2 / 2) ∧ (cos (360 - theta) = cos theta) ∧ 315 = angle ∧ (forall a : real, cos a = cos (360 - a)) ∧ (315 ∈ (4th quadrant)) ∧ (cos 315 = cos 45)

theorem cos_315_proof : (cos 315 = sqrt 2 / 2) :=
begin
  -- We start by defining basic facts
  -- 315 = 360 - 45
  -- Cosine is positive in the fourth quadrant, known value \cos 45 = sqrt 2 / 2
  -- Cosine property \cos (360 - \theta) = \cos \theta
  let theta := 45,
  let angle := 360 - theta,
  have H1 : cos theta = sqrt 2 / 2,
  {
    -- known fact
    sorry,
  },
  have H2 : cos (360 - theta) = cos theta,
  {
    -- cosine property
    sorry,
  },
  have H3 : angle = 315,
  {
    -- subtraction identity
    sorry,
  },
  rw H3,
  rw H2,
  rw H1,
  refl,
end

end cos_315_proof_l816_816553


namespace ratio_dislikes_to_likes_l816_816894

theorem ratio_dislikes_to_likes 
  (D : ℕ) 
  (h1 : D + 1000 = 2600) 
  (h2 : 3000 > 0) : 
  D / 3000 = 8 / 15 :=
by sorry

end ratio_dislikes_to_likes_l816_816894


namespace mike_scored_l816_816787

-- Defining the problem conditions
def max_marks : ℕ := 780
def pass_percentage : ℝ := 0.3
def shortfall : ℕ := 22
def passing_marks := (pass_percentage * max_marks)

-- Proving the number of marks Mike scored
theorem mike_scored : ℕ :=
  let M := passing_marks - shortfall in
  M = 212 := by
  sorry

end mike_scored_l816_816787


namespace solve_for_m_l816_816018

theorem solve_for_m (m : ℕ) :
  (\frac{1^m}{5^m}) * (\frac{1^{16}}{4^{16}}) = \frac{1}{2 * (10)^{31}} → m = 31 :=
by
  sorry

end solve_for_m_l816_816018


namespace power_of_sqrt2_minus_1_l816_816822

noncomputable def a (n : ℕ) : ℝ := (Real.sqrt 2 - 1) ^ n
noncomputable def b (n : ℕ) : ℝ := (Real.sqrt 2 + 1) ^ n
noncomputable def c (n : ℕ) : ℝ := (b n + a n) / 2
noncomputable def d (n : ℕ) : ℝ := (b n - a n) / 2

theorem power_of_sqrt2_minus_1 (n : ℕ) : a n = Real.sqrt (d n ^ 2 + 1) - Real.sqrt (d n ^ 2) :=
by
  sorry

end power_of_sqrt2_minus_1_l816_816822


namespace hyperbola_equation_shared_foci_asymptotes_line_through_hyperbola_focus_with_inclination_l816_816163

noncomputable def ellipse : Type := ∀ (x y : ℝ), (x^2 / 49 + y^2 / 24 = 1)
noncomputable def hyperbola : Type := ∀ (x y : ℝ), (x^2 / 9 - y^2 / 16 = 1)
noncomputable def line_through_focus : Type := ∀ (x y : ℝ), (sqrt(3) * x - y - 5 * sqrt(3) = 0)

-- Proof problem 1:
theorem hyperbola_equation_shared_foci_asymptotes :
  (∀ (x y : ℝ), (x^2 / 49 + y^2 / 24 = 1)) → (∀ (x y : ℝ), (y = 4/3 * x ∨ y = -4/3 * x)) → 
  (∀ (x y : ℝ), (x^2 / 9 - y^2 / 16 = 1)) :=
  by sorry

-- Proof problem 2:
theorem line_through_hyperbola_focus_with_inclination :
  (∀ (x y : ℝ), (x^2 / 9 - y^2 / 16 = 1)) → 
  (exists (f : ℝ × ℝ), f = (5, 0)) → 
  (∀ (x y : ℝ), (tan (π / 3) * (x - 5) = y)) → 
  (∀ (x y : ℝ), (sqrt(3) * x - y - 5 * sqrt(3) = 0)) :=
  by sorry

end hyperbola_equation_shared_foci_asymptotes_line_through_hyperbola_focus_with_inclination_l816_816163


namespace cos_315_is_sqrt2_div_2_l816_816520

noncomputable def cos_315_eq : Prop := 
  real.cos (315 * real.pi / 180) = real.sqrt 2 / 2

theorem cos_315_is_sqrt2_div_2 : cos_315_eq := by 
  sorry

end cos_315_is_sqrt2_div_2_l816_816520


namespace commute_time_difference_l816_816085

-- Define the conditions as constants
def distance_to_work : ℝ := 1.5
def walking_speed : ℝ := 3
def train_speed : ℝ := 20
def additional_train_time_minutes : ℝ := 10.5

-- The main proof problem
theorem commute_time_difference : 
  (distance_to_work / walking_speed * 60) - 
  ((distance_to_work / train_speed * 60) + additional_train_time_minutes) = 15 :=
by
  sorry

end commute_time_difference_l816_816085


namespace challenging_subsets_l816_816080
open Set

def is_challenging (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n = 2^a * (2^b + 1)

theorem challenging_subsets (X : Set ℕ) (n : ℕ) (hX : ∀ x ∈ X, is_challenging x) 
  (hX_bound : ∀ x ∈ X, x < 2^n) (hX_size : X.size ≥ 4 * (n - 1) / 3) :
  ∃ (A B : Set ℕ), A ⊆ X ∧ B ⊆ X ∧ A ∩ B = ∅ ∧ A.size = B.size ∧ A.sum = B.sum := 
sorry

end challenging_subsets_l816_816080


namespace proportion_correct_l816_816200

theorem proportion_correct (x y : ℝ) (h1 : 2 * y = 5 * x) (h2 : x ≠ 0 ∧ y ≠ 0) : x / y = 2 / 5 := 
sorry

end proportion_correct_l816_816200


namespace clyde_picked_bushels_l816_816460

theorem clyde_picked_bushels (weight_per_bushel : ℕ) (weight_per_cob : ℕ) (cobs_picked : ℕ) :
  weight_per_bushel = 56 →
  weight_per_cob = 1 / 2 →
  cobs_picked = 224 →
  cobs_picked * weight_per_cob / weight_per_bushel = 2 :=
by
  intros
  sorry

end clyde_picked_bushels_l816_816460


namespace num_ordered_pairs_l816_816170

theorem num_ordered_pairs :
  ∃ n : ℕ, n = 49 ∧ ∀ (a b : ℕ), a + b = 50 → 0 < a ∧ 0 < b → (1 ≤ a ∧ a < 50) :=
by
  sorry

end num_ordered_pairs_l816_816170


namespace count_three_digit_numbers_l816_816198

theorem count_three_digit_numbers : ∀ (smallest largest : ℕ), smallest = 100 → largest = 999 → (largest - smallest + 1 = 900) :=
by
  intros smallest largest h_smallest h_largest
  rw [h_smallest, h_largest]
  sorry

end count_three_digit_numbers_l816_816198


namespace solve_equation_l816_816384

theorem solve_equation (x : ℝ) (h : x ≠ 2) : 
  2 / (x - 2) = (1 + x) / (x - 2) + 1 → x = 3 / 2 := by
  sorry

end solve_equation_l816_816384


namespace remainder_p_div_x_plus_3_l816_816432

theorem remainder_p_div_x_plus_3
  (A B C : ℝ)
  (h : A * 243 + B * 27 + C * 3 = 7)
  : let p := λ x : ℝ, A * x^5 + B * x^3 + C * x + 4 in
  p (-3) = -3 := by
  intro p
  sorry

end remainder_p_div_x_plus_3_l816_816432


namespace geometric_intersection_l816_816476

variable (n : ℕ)
variable (A : Fin n → Point3D)
variable (B : Fin n → Point3D)
variable (plane : Plane)

-- Conditions as assumptions
-- Assume there is a function point_on that checks if a point is on a line
axiom Bi_on_AiAi1 : ∀ i : Fin n, point_on (B i) (line (A i) (A (i + 1) % n))
axiom even_number_Bi_lying_on_sides : even (count (λ i : Fin n, on_side (B i) polygon_sides))

-- Definition of the main property to prove
def main_property : Prop :=
  (∏ i, (length (segment (A i) (B i)) / length (segment (A (i + 1) % n) (B i)))) = 1

-- The theorem
theorem geometric_intersection : main_property n A B plane :=
by sorry

end geometric_intersection_l816_816476


namespace probability_of_floor_sqrt_l816_816354

noncomputable def probability_sqrt_floor (x : ℝ) (hx : 225 ≤ x ∧ x < 325) (hf1 : ⌊√x⌋ = 15) : ℝ :=
  let range1 := (225 : ℝ, 256 : ℝ)
  let range2 := (225 : ℝ, 228.01)
  let intersection := (225 : ℝ, 228.01)
  let length_intersection := intersection.2 - intersection.1
  let length_range1 := range1.2 - range1.1
  length_intersection / length_range1

theorem probability_of_floor_sqrt (x : ℝ) (hx : 225 ≤ x ∧ x < 325) (hf1 : ⌊√x⌋ = 15) :
  probability_sqrt_floor x hx hf1 = 301 / 3100 :=
by
  sorry

end probability_of_floor_sqrt_l816_816354


namespace rice_difference_l816_816052

theorem rice_difference : 
  let grains k := 3 ^ k in
  (grains 12) - (∑ i in Finset.range 10, grains (i + 1)) = 442869 := 
by
  let grains : ℕ → ℕ := λ k, 3 ^ k
  have sum_first_10 : ∑ i in Finset.range 10, grains (i + 1) = 88572 := sorry
  calc
    grains 12 - (∑ i in Finset.range 10, grains (i + 1))
        = 531441 - 88572 : by rw sum_first_10
    ... = 442869 : by norm_num

end rice_difference_l816_816052


namespace cos_315_eq_sqrt2_div2_l816_816533

theorem cos_315_eq_sqrt2_div2 : Real.cos (315 * Real.pi / 180) = sqrt 2 / 2 := 
sorry

end cos_315_eq_sqrt2_div2_l816_816533


namespace sin_minus_cos_l816_816216

variable (θ : ℝ)
hypothesis (h1 : 0 < θ ∧ θ < π / 2)
hypothesis (h2 : Real.tan θ = 1 / 3)

theorem sin_minus_cos (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := 
by 
  sorry

end sin_minus_cos_l816_816216


namespace range_of_H_l816_816011

noncomputable def H (x : ℝ) : ℝ := |x + 2| - |x - 3| + 3 * x

theorem range_of_H : set.Icc (-1 : ℝ) (⊤ : ℝ) = set.range H := sorry

end range_of_H_l816_816011


namespace parallel_lines_condition_l816_816660

theorem parallel_lines_condition (a b : ℝ) (h₁ : a * b = 4) :
  (2 * x + a * y - 1 = 0) ∧ (b * x + 2 * y + 1 = 0) → (ab = 4) ∧ (∃ (a b : ℝ), 2 * x + a * y - 1 = 0 → b * x + 2 * y + 1 = 0) :=
begin
  intros h_parallel,
  split,
  { exact h₁, },
  { sorry, }
end

end parallel_lines_condition_l816_816660


namespace f_increasing_interval_l816_816851

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 3 * x - 4)

def domain_f (x : ℝ) : Prop := (x < -1) ∨ (x > 4)

def increasing_g (a b : ℝ) : Prop := ∀ x y, a < x → x < y → y < b → (x^2 - 3 * x - 4 < y^2 - 3 * y - 4)

theorem f_increasing_interval :
  ∀ x, domain_f x → increasing_g 4 (a) → increasing_g 4 (b) → 
    (4 < x ∧ x < b) → (f x < f (b - 0.1)) := sorry

end f_increasing_interval_l816_816851


namespace sin_minus_cos_theta_l816_816242

theorem sin_minus_cos_theta (θ : ℝ) (h₀ : 0 < θ ∧ θ < (π / 2)) 
  (h₁ : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -√10 / 5 :=
by
  sorry

end sin_minus_cos_theta_l816_816242


namespace ball_problem_l816_816317

theorem ball_problem (a : ℕ) (h1 : 3 / a = 0.25) : a = 12 :=
by sorry

end ball_problem_l816_816317


namespace mo_money_l816_816790

open Nat

def number_of_students := 30

def percentage_valentine := 0.60

def cost_per_card := 2

def spending_percentage := 0.90

theorem mo_money :
  let number_of_valentines := percentage_valentine * number_of_students
  let total_cost := number_of_valentines * cost_per_card
  let mo_money_total := total_cost / spending_percentage
  mo_money_total = 40 := by
sorry

end mo_money_l816_816790


namespace prob_not_adjacent_l816_816719

theorem prob_not_adjacent :
  let total_ways := Nat.choose 10 2
  let adjacent_ways := 9
  let prob_adjacent := (adjacent_ways / total_ways : ℚ)
  let prob_not_adjacent := 1 - prob_adjacent
  prob_not_adjacent = 4 / 5 := by
  unfold total_ways adjacent_ways prob_adjacent prob_not_adjacent
  norm_num
  sorry

end prob_not_adjacent_l816_816719


namespace circle_equation_and_min_pq_value_l816_816991

theorem circle_equation_and_min_pq_value :
  (∃ (Cx Cy : ℝ), (Cx, Cy) = (2, 4) ∧ ∀ x y : ℝ, (x - 2)^2 + (y - 4)^2 = 25 ∧
  let l (m x y : ℝ) := (m + 2) * x + (2 * m + 1) * y - 7 * m - 8  in
  ∀ m : ℝ, let PQ := (fun (P Q : ℝ × ℝ) => sorry) in dist PQ.fst PQ.snd = 4 * Real.sqrt 5)
sorry

end circle_equation_and_min_pq_value_l816_816991


namespace sin_minus_cos_l816_816233

noncomputable def theta_condition (θ : ℝ) : Prop := (0 < θ) ∧ (θ < π / 2) ∧ (Real.tan θ = 1 / 3)

theorem sin_minus_cos (θ : ℝ) (hθ : theta_condition θ) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_minus_cos_l816_816233


namespace ant_socks_shoes_permutations_l816_816924

def num_permutations_socks_shoes : ℕ := 10!

theorem ant_socks_shoes_permutations :
  let num_legs := 6
  let total_items := 2 * num_legs
  let valid_orders := num_permutations_socks_shoes
in valid_orders = 10! := by
  sorry

end ant_socks_shoes_permutations_l816_816924


namespace determine_a_real_l816_816757

def is_positive_real (z : ℂ) : Prop := ∃ r : ℝ, r > 0 ∧ z = r

theorem determine_a_real (a : ℝ) (hi : is_positive_real ((a - complex.I)^2 * complex.I)) : a = 1 :=
sorry

end determine_a_real_l816_816757


namespace minimize_area_l816_816414

-- Definitions of geometric entities and their properties
open EuclideanGeometry

-- Let A, O, B be points forming an acute angle AOB.
variables (A O B M : Point)
-- Assume O is the vertex of the angle and M is a point inside the acute angle.
variable [angle A O B < π / 2]
variable (inside : InAngle A O B M)

-- Define the lines
variables (line_through : (M : Point) -> Line)

noncomputable def minimal_area_triangle :=
  let line_through_M := line_through M in
  let intersections := (line_through_M ∩ₗ Line A O, line_through_M ∩ₗ Line O B) in
  let (D, E) := intersections in
  (area_of_triangle M D E) = (minimize area_of_triangle)

theorem minimize_area (M_inside_AOB : inside M) :
  line_through M => minimal_area_triangle M_inside_AOB M line_through :=
sorry

end minimize_area_l816_816414


namespace cos_315_deg_l816_816545

noncomputable def cos_315 : ℝ :=
  Real.cos (315 * Real.pi / 180)

theorem cos_315_deg : cos_315 = Real.sqrt 2 / 2 :=
by
  sorry

end cos_315_deg_l816_816545


namespace cos_315_deg_l816_816546

noncomputable def cos_315 : ℝ :=
  Real.cos (315 * Real.pi / 180)

theorem cos_315_deg : cos_315 = Real.sqrt 2 / 2 :=
by
  sorry

end cos_315_deg_l816_816546


namespace find_MM0_length_l816_816909

noncomputable def sqrt3 : ℝ := Real.sqrt 3

def point_M0 : ℝ × ℝ := (1, 5)

def line_l_equation (x y : ℝ) : Prop := y - 5 = sqrt3 * (x - 1)

def line_intersection_equation (x y : ℝ) : Prop := x - y = 2 * sqrt3

def point_M : ℝ × ℝ := 
  let x := (5 - 3 * sqrt3) / (sqrt3 - 1) in
  let y := x - 2 * sqrt3 in
  (x, y)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem find_MM0_length :
  let x := sqrt3 - 2
  let y := -sqrt3 - 2
  distance (sqrt3 - 2, -sqrt3 - 2) (1, 5) = 5 - 3 * sqrt3 :=
by
  sorry

end find_MM0_length_l816_816909


namespace hannah_banana_flour_l816_816194

-- Define the conditions
def flour_needed (cups_of_mush : ℕ) : ℕ :=
  3 * cups_of_mush

def banana_mush (bananas : ℕ) : ℕ :=
  bananas / 4

-- Given problem statement and condition
theorem hannah_banana_flour :
  banana_mush 20 = 5 ∧ flour_needed 5 = 15 :=
by 
  split;
  sorry

end hannah_banana_flour_l816_816194


namespace lloyd_excess_rate_multiple_l816_816780

theorem lloyd_excess_rate_multiple :
  let h_regular := 7.5
  let r := 4.00
  let h_total := 10.5
  let e_total := 48
  let e_regular := h_regular * r
  let excess_hours := h_total - h_regular
  let e_excess := e_total - e_regular
  let m := e_excess / (excess_hours * r)
  m = 1.5 :=
by
  sorry

end lloyd_excess_rate_multiple_l816_816780


namespace stratified_sampling_l816_816480

theorem stratified_sampling 
  (total_teachers : ℕ)
  (senior_teachers : ℕ)
  (intermediate_teachers : ℕ)
  (junior_teachers : ℕ)
  (sample_size : ℕ)
  (x y z : ℕ) 
  (h1 : total_teachers = 150)
  (h2 : senior_teachers = 45)
  (h3 : intermediate_teachers = 90)
  (h4 : junior_teachers = 15)
  (h5 : sample_size = 30)
  (h6 : x + y + z = sample_size)
  (h7 : x * 10 = sample_size / 5)
  (h8 : y * 10 = (2 * sample_size) / 5)
  (h9 : z * 10 = sample_size / 15) :
  (x, y, z) = (9, 18, 3) := sorry

end stratified_sampling_l816_816480


namespace minimal_rope_cost_l816_816385

theorem minimal_rope_cost :
  let pieces_needed := 10
  let length_per_piece := 6 -- inches
  let total_length_needed := pieces_needed * length_per_piece -- inches
  let one_foot_length := 12 -- inches
  let cost_six_foot_rope := 5 -- dollars
  let cost_one_foot_rope := 1.25 -- dollars
  let six_foot_length := 6 * one_foot_length -- inches
  let one_foot_total_cost := (total_length_needed / one_foot_length) * cost_one_foot_rope
  let six_foot_total_cost := cost_six_foot_rope
  total_length_needed <= six_foot_length ∧ six_foot_total_cost < one_foot_total_cost →
  six_foot_total_cost = 5 := sorry

end minimal_rope_cost_l816_816385


namespace radius_of_inscribed_sphere_l816_816126

theorem radius_of_inscribed_sphere (a : ℝ) : 
  ∃ r : ℝ, r = (a * Real.sqrt 6) / 12 :=
by
  use (a * Real.sqrt 6) / 12
  sorry

end radius_of_inscribed_sphere_l816_816126


namespace train_cross_time_is_nine_seconds_l816_816487

-- Define the speed of the train in km/hr and convert to m/s
def speed_km_hr := 30
def speed_m_s : ℝ := (speed_km_hr * 1000) / 3600

-- Define the length of the train in meters
def train_length := 75

-- Define the time it takes for the train to cross the pole
def train_cross_time : ℝ := train_length / speed_m_s

-- The statement that we want to prove
theorem train_cross_time_is_nine_seconds : train_cross_time = 9 := by
  sorry

end train_cross_time_is_nine_seconds_l816_816487


namespace sin_minus_cos_eq_l816_816226

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end sin_minus_cos_eq_l816_816226


namespace simplest_quadratic_radical_l816_816501

-- Define the quadratic radicals
def option_A := Real.sqrt 2
def option_B := Real.sqrt (1 / 5)
def option_C := Real.sqrt 27
def option_D (a : ℝ) := Real.sqrt (a^2)

-- Define the predicate to check if one radical is simpler than another
def simpler (x y : ℝ) : Prop := x = Real.sqrt 2

-- Define the main theorem
theorem simplest_quadratic_radical (a : ℝ) : 
  simpler option_A (min (min option_B option_C) (option_D a)) := 
by 
  -- Skeleton proof
  sorry

end simplest_quadratic_radical_l816_816501


namespace is_real_is_complex_is_pure_imaginary_is_second_quadrant_l816_816973

def m (z : ℝ) : Prop :=
  z = m^2 * (1 / (m + 5) + complex.I) + (8 * m + 15) * complex.I +
      (m - 6) / (m + 5)

theorem is_real (m : ℝ) : (m = -3) ↔ (m^2 + 8 * m + 15 = 0 ∧ m + 5 ≠ 0) :=
sorry

theorem is_complex (m : ℝ) : (m ≠ -3 ∧ m ≠ -5) ↔ (m^2 + 8 * m + 15 ≠ 0 ∧ m + 5 ≠ 0) :=
sorry

theorem is_pure_imaginary (m : ℝ) : (m = 2) ↔ ((m^2 + m - 6) / (m + 5) = 0 ∧ m^2 + 8 * m + 15 ≠ 0) :=
sorry

theorem is_second_quadrant (m : ℝ) : (m < -5 ∨ -3 < m ∧ m < 2) ↔ ((m^2 + m - 6) / (m + 5) < 0 ∧ m^2 + 8 * m + 15 > 0) :=
sorry

end is_real_is_complex_is_pure_imaginary_is_second_quadrant_l816_816973


namespace P_at_n_plus_1_l816_816763

noncomputable def P (n : ℕ) (x : ℝ) : ℝ :=
  let Q (x : ℝ) : ℝ := ( (-1 : ℝ) ^ (n + 1) / Real.ofNat (Nat.factorial (n + 1))) * 
                        (x * (List.range (n + 1)).map x.sub).prod 
  in (Q x + x) / (x + 1)

theorem P_at_n_plus_1 (n : ℕ) : 
  P n (n + 1) = (n + 1 + (-1) ^ (n + 1)) / (n + 2) := 
by 
  sorry

end P_at_n_plus_1_l816_816763


namespace rate_of_interest_l816_816858

-- Definitions based on the conditions
def P := 2600
def SI (R : ℝ) := (P * R * 5) / 100

-- Statement of the problem
theorem rate_of_interest: ∀ R : ℝ, SI R = (P - 2080) → R = 4 :=
by
  sorry

end rate_of_interest_l816_816858


namespace six_digit_number_division_l816_816435

theorem six_digit_number_division :
  ∃ a b p : ℕ, 
    (111111 * a = 1111 * b * 233 + p) ∧ 
    (11111 * a = 111 * b * 233 + p - 1000) ∧
    (111111 * 7 = 777777) ∧
    (1111 * 3 = 3333) :=
by
  sorry

end six_digit_number_division_l816_816435


namespace f_is_odd_function_l816_816845

open Real

def f (x : ℝ) : ℝ := (1 / x) * log 2 (4 ^ x + 1) - 1

theorem f_is_odd_function : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

end f_is_odd_function_l816_816845


namespace sum_of_center_coordinates_l816_816855

theorem sum_of_center_coordinates (A B : ℝ × ℝ) (hx : A = (12, -8)) (hy : B = (-4, 2)) :
  let center := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  in center.1 + center.2 = 1 :=
by
  sorry

end sum_of_center_coordinates_l816_816855


namespace cos_315_eq_l816_816561

theorem cos_315_eq :
  let Q := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))
  in Q = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) →
     Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  intro hQ
  sorry

end cos_315_eq_l816_816561


namespace find_cost_price_l816_816925

theorem find_cost_price (SP : ℝ) (ProfitRate : ℝ) (SellingPrice : SP) : 
  (CP : ℝ) → SP = CP * (1 + ProfitRate) → CP = 180 :=
by
  assume CP : ℝ,
  assume h1 : SP = CP * (1 + ProfitRate),
  have h2 : SP = 207 := sorry, -- Given condition SellingPrice = 207
  have h3 : ProfitRate = 0.15 := sorry, -- Given condition ProfitRate = 15%
  sorry -- Skip the rest of the proof

end find_cost_price_l816_816925


namespace regular_price_per_can_l816_816398

variable (P : ℝ) -- Regular price per can

-- Condition: The regular price per can is discounted 15 percent when the soda is purchased in 24-can cases
def discountedPricePerCan (P : ℝ) : ℝ :=
  0.85 * P

-- Condition: The price of 72 cans purchased in 24-can cases is $18.36
def priceOf72CansInDollars : ℝ :=
  18.36

-- Predicate describing the condition that the price of 72 cans is 18.36
axiom h : (72 * discountedPricePerCan P) = priceOf72CansInDollars

theorem regular_price_per_can (P : ℝ) (h : (72 * discountedPricePerCan P) = priceOf72CansInDollars) : P = 0.30 :=
by
  sorry

end regular_price_per_can_l816_816398


namespace max_digit_sum_in_24_hour_format_l816_816904

theorem max_digit_sum_in_24_hour_format : ∃ (h m s : ℕ), (h < 24 ∧ m < 60 ∧ s < 60) ∧ (max_digit_sum h m s = 38) :=
sorry

def digit_sum (n : ℕ) : ℕ :=
  n / 10 + n % 10 -- Sum the digits of a two-digit number

def max_digit_sum (h m s : ℕ) : ℕ :=
  digit_sum h + digit_sum m + digit_sum s -- Sum of the digit sums of hours, minutes, and seconds

end max_digit_sum_in_24_hour_format_l816_816904


namespace simplified_form_l816_816828

def simplify_expression (x : ℝ) : ℝ :=
  (3 * x - 2) * (6 * x ^ 8 + 3 * x ^ 7 - 2 * x ^ 3 + x)

theorem simplified_form (x : ℝ) : 
  simplify_expression x = 18 * x ^ 9 - 3 * x ^ 8 - 6 * x ^ 7 - 6 * x ^ 4 - 4 * x ^ 3 + x :=
by
  sorry

end simplified_form_l816_816828


namespace set_contains_infinite_points_l816_816060

noncomputable
def set_with_symmetry (M : Set ℝ) (l0 l1 : ℝ) (α : ℝ) : Prop :=
  have intersect_at_angle : ∃ O : ℝ × ℝ, ∀ A ∈ M, ∃ B ∈ M, 
                            (B = rotate A O α ∨ B = rotate A O (-α)),
  M.nonempty → ∃ A B ∈ M, A ≠ B
  ∧ irrational (α / Real.pi)
  ∧ (∀ A ∈ M, ∃ n : ℕ, (rotate A (0, 0) (2 * n * α) ∈ M))

theorem set_contains_infinite_points
  {M : Set ℝ} {l0 l1 : ℝ} {α : ℝ}
  (h_symmetry : set_with_symmetry M l0 l1 α) :
  (∃ A B ∈ M, A ≠ B) → (∃ S : Set ℝ, infinite S ∧ S ⊆ M) :=
sorry

end set_contains_infinite_points_l816_816060


namespace matching_pair_probability_l816_816592

theorem matching_pair_probability :
  let total_socks := 22
  let blue_socks := 12
  let red_socks := 10
  let total_ways := (total_socks * (total_socks - 1)) / 2
  let blue_ways := (blue_socks * (blue_socks - 1)) / 2
  let red_ways := (red_socks * (red_socks - 1)) / 2
  let matching_ways := blue_ways + red_ways
  total_ways = 231 →
  blue_ways = 66 →
  red_ways = 45 →
  matching_ways = 111 →
  (matching_ways : ℝ) / total_ways = 111 / 231 := by sorry

end matching_pair_probability_l816_816592


namespace tangent_point_x_l816_816756

-- Define the function f based on the input parameter a
def f (a : ℝ) (x : ℝ) : ℝ := exp x + a / exp x

-- Define the even property condition
def f_even (a : ℝ) : Prop := ∀ x : ℝ, f(a, -x) = f(a, x)

-- Define the condition of the slope of the tangent line
def f_derivative (a : ℝ) (x : ℝ) : ℝ := exp(x) - 1 / exp(x)

-- State the main theorem, asserting the x-coordinate of the tangent point
theorem tangent_point_x (a : ℝ) (h_even : f_even a) (h_slope : ∃ x : ℝ, f_derivative a x = 3/2) : ∃ x : ℝ, x = Real.log 2 :=
by
  sorry

end tangent_point_x_l816_816756


namespace cone_height_90_deg_is_36_8_l816_816902

noncomputable def cone_height_volume (V : ℝ) (θ : ℝ) : ℝ :=
  if θ = π / 2 then
    let r := (3 * V / π)^(1/3) in r
  else
    0  -- Not valid if the angle isn't 90 degrees

theorem cone_height_90_deg_is_36_8 :
  cone_height_volume (16384 * π) (π / 2) = 36.8 :=
by
  sorry

end cone_height_90_deg_is_36_8_l816_816902


namespace max_min_diff_planes_l816_816588

theorem max_min_diff_planes (T : Simplex 3 4) 
  (planes : list (affine_hull ℝ (affine_plane ℝ 3))) 
  (interior_intersection : ∀ (p ∈ planes), ∃ (S : set (affine_hull ℝ (affine_plane ℝ 3))), 
      S ∩ (affine_hull ℝ (convex_hull ℝ (set.range T.points))) 
      = ⋃ (s ∈ (faces (convex_hull ℝ (set.range T.points)))), 
        (midpoint_segment ℝ s)) :
  (max_min_plane_difference planes) = 0 := 
sorry

end max_min_diff_planes_l816_816588


namespace sin_minus_cos_l816_816217

variable (θ : ℝ)
hypothesis (h1 : 0 < θ ∧ θ < π / 2)
hypothesis (h2 : Real.tan θ = 1 / 3)

theorem sin_minus_cos (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := 
by 
  sorry

end sin_minus_cos_l816_816217


namespace triangle_sides_l816_816927

-- Definitions from the conditions
def is_isosceles (a b c : ℕ) : Prop := a = b ∨ a = c ∨ b = c
def is_triangle (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

noncomputable def side_lengths (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
  if is_isosceles a b c ∧ is_triangle a b c ∧ a + b + c = 17 then
    (a, b, c)
  else
    (0, 0, 0)

theorem triangle_sides : 
  ∃ (a b c : ℕ), side_lengths a b c = (5, 5, 7) ∨ side_lengths a b c = (7, 7, 3) :=
sorry

end triangle_sides_l816_816927


namespace find_x_l816_816688

theorem find_x (x : ℝ) (h₁ : x > 0) (h₂ : x^4 = 390625) : x = 25 := 
by sorry

end find_x_l816_816688


namespace wheels_in_garage_l816_816415

-- Definitions of the entities within the problem
def cars : Nat := 2
def car_wheels : Nat := 4

def riding_lawnmower : Nat := 1
def lawnmower_wheels : Nat := 4

def bicycles : Nat := 3
def bicycle_wheels : Nat := 2

def tricycle : Nat := 1
def tricycle_wheels : Nat := 3

def unicycle : Nat := 1
def unicycle_wheels : Nat := 1

-- The total number of wheels in the garage
def total_wheels :=
  (cars * car_wheels) +
  (riding_lawnmower * lawnmower_wheels) +
  (bicycles * bicycle_wheels) +
  (tricycle * tricycle_wheels) +
  (unicycle * unicycle_wheels)

-- The theorem we wish to prove
theorem wheels_in_garage : total_wheels = 22 := by
  sorry

end wheels_in_garage_l816_816415


namespace combination_lock_code_l816_816380

-- Definitions: SEAS, EBB, BASS in an unknown base such that the subtraction holds in that base and digits are distinct.
def is_valid_combination (SEAS EBB BASS d : ℕ) : Prop :=
  let DECIMAL := 10 in
  let S := 9 in
  let E := 8 in
  let A := 7 in
  let B := 0 in
  SEAS = 1000 * S + 100 * E + 10 * A + S ∧
  EBB = 100 * E + 10 * B + B ∧
  BASS = 1000 * B + 100 * A + 10 * S + S ∧
  SEAS - EBB = BASS ∧
  B ≠ S ∧ B ≠ E ∧ B ≠ A ∧ S ≠ E ∧ S ≠ A ∧ E ≠ A

-- Problem: Prove that the valid combination lock code is 9790 in base 10.
theorem combination_lock_code :
  ∃ SEAS EBB BASS, ∃ d : ℕ, d = 10 ∧ is_valid_combination SEAS EBB BASS d ∧ SEAS = 9790 :=
by
  sorry

end combination_lock_code_l816_816380


namespace m_le_n_l816_816997

def polygon : Type := sorry  -- A placeholder definition for polygon.

variables (M : polygon) -- The polygon \( M \)
def max_non_overlapping_circles (M : polygon) : ℕ := sorry -- The maximum number of non-overlapping circles with diameter 1 inside \( M \).
def min_covering_circles (M : polygon) : ℕ := sorry -- The minimum number of circles with radius 1 required to cover \( M \).

theorem m_le_n (M : polygon) : min_covering_circles M ≤ max_non_overlapping_circles M :=
sorry

end m_le_n_l816_816997


namespace cos_315_proof_l816_816549

noncomputable def cos_315_eq : Prop :=
  let theta := 45 in
  let angle := 360 - theta in
  ∃ c : real, (cos 45) = (sqrt 2 / 2) ∧ (cos (360 - theta) = cos theta) ∧ 315 = angle ∧ (forall a : real, cos a = cos (360 - a)) ∧ (315 ∈ (4th quadrant)) ∧ (cos 315 = cos 45)

theorem cos_315_proof : (cos 315 = sqrt 2 / 2) :=
begin
  -- We start by defining basic facts
  -- 315 = 360 - 45
  -- Cosine is positive in the fourth quadrant, known value \cos 45 = sqrt 2 / 2
  -- Cosine property \cos (360 - \theta) = \cos \theta
  let theta := 45,
  let angle := 360 - theta,
  have H1 : cos theta = sqrt 2 / 2,
  {
    -- known fact
    sorry,
  },
  have H2 : cos (360 - theta) = cos theta,
  {
    -- cosine property
    sorry,
  },
  have H3 : angle = 315,
  {
    -- subtraction identity
    sorry,
  },
  rw H3,
  rw H2,
  rw H1,
  refl,
end

end cos_315_proof_l816_816549


namespace sufficient_but_not_necessary_for_parallel_l816_816167

noncomputable def condition_parallel (m : ℝ) : Prop :=
  (m - 1) / m = (m - 1) / 2

theorem sufficient_but_not_necessary_for_parallel (m : ℝ) :
  (condition_parallel m → m = 2) ∧ (¬(condition_parallel m ↔ m = 2)) :=
begin
  -- inserting the main conditions for proving the mathematically equivalent problem
  sorry
end

end sufficient_but_not_necessary_for_parallel_l816_816167


namespace simplify_expression_l816_816826

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end simplify_expression_l816_816826


namespace all_tell_truth_alice_bob_carol_truth_at_least_one_truth_l816_816572

noncomputable def alice_truth : ℝ := 0.70
noncomputable def bob_truth : ℝ := 0.60
noncomputable def carol_truth : ℝ := 0.80
noncomputable def david_truth : ℝ := 0.50
noncomputable def eric_truth : ℝ := 0.30

theorem all_tell_truth : alice_truth * bob_truth * carol_truth * david_truth * eric_truth = 0.042 := 
by sorry

theorem alice_bob_carol_truth : alice_truth * bob_truth * carol_truth = 0.336 :=
by sorry

theorem at_least_one_truth : 1 - (1 - alice_truth) * (1 - bob_truth) * (1 - carol_truth) * (1 - david_truth) * (1 - eric_truth) = 0.9916 :=
by sorry

end all_tell_truth_alice_bob_carol_truth_at_least_one_truth_l816_816572


namespace no_odd_total_rows_columns_l816_816648

open Function

def array_odd_column_row_count (n : ℕ) (array : ℕ → ℕ → ℤ) : Prop :=
  n % 2 = 1 ∧
  (∀ i j, 0 ≤ array i j ∧ array i j ≤ 1 ∧ array i j = -1 ∨ array i j = 1) →
  (∃ (rows cols : Finset ℕ),
    rows.card + cols.card = n ∧
    ∀ r ∈ rows, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array r) k = -1 ∧
    ∀ c ∈ cols, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array c) k = -1
    )

theorem no_odd_total_rows_columns (n : ℕ) (array : ℕ → ℕ → ℤ) :
  n % 2 = 1 →
  (∀ i j, 0 ≤ array i j ∧ array i j ≤ 1 ∧ (array i j = -1 ∨ array i j = 1)) →
  ¬ (∃ rows cols : Finset ℕ,
       rows.card + cols.card = n ∧
       ∀ r ∈ rows, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array r k = -1) ∧
       ∀ c ∈ cols, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array c k = -1)) :=
by
  intros h_array
  sorry

end no_odd_total_rows_columns_l816_816648


namespace sin_minus_cos_l816_816257

theorem sin_minus_cos (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < (Real.pi / 2)) (hθ3 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := by
  sorry

end sin_minus_cos_l816_816257


namespace proportion_of_overcrowded_and_passengers_l816_816848

-- Condition definitions
def cars_10_to_19 : ℕ → ℝ := λ N, 0.04 * N
def cars_20_to_29 : ℕ → ℝ := λ N, 0.06 * N
def cars_30_to_39 : ℕ → ℝ := λ N, 0.12 * N
def cars_40_to_49 : ℕ → ℝ := λ N, 0.18 * N
def cars_50_to_59 : ℕ → ℝ := λ N, 0.20 * N
def cars_60_to_69 : ℕ → ℝ := λ N, 0.14 * N
def cars_70_to_79 : ℕ → ℝ := λ N, 0.06 * N

def overcrowded_proportion : ℝ := 0.40
def min_passenger_proportion_in_overcrowded : ℝ := 0.31

-- Proof statement
theorem proportion_of_overcrowded_and_passengers (N : ℕ) :
  let P := overcrowded_proportion in
  let P' := min_passenger_proportion_in_overcrowded in
  let total_non_overcrowded_passengers := 19 * cars_10_to_19 N + 29 * cars_20_to_29 N + 39 * cars_30_to_39 N +
                                           49 * cars_40_to_49 N + 59 * cars_50_to_59 N in
  let total_overcrowded_passengers := 60 * cars_60_to_69 N + 70 * cars_70_to_79 N in
  let total_passengers := total_non_overcrowded_passengers + total_overcrowded_passengers in
  (total_overcrowded_passengers / total_passengers = P') ∧
  (P = 0.40) ∧
  (P' = 0.31) ∧
  (P' ≥ P) := 
sorry

end proportion_of_overcrowded_and_passengers_l816_816848


namespace min_people_with_all_items_in_Owlna_l816_816323

theorem min_people_with_all_items_in_Owlna (population : ℕ) 
(refrigerator_pct television_pct computer_pct air_conditioner_pct washing_machine_pct smartphone_pct : ℚ) 
(h_population : population = 5000) 
(h_refrigerator : refrigerator_pct = 0.72) 
(h_television : television_pct = 0.75)
(h_computer : computer_pct = 0.65)
(h_air_conditioner : air_conditioner_pct = 0.95)
(h_washing_machine : washing_machine_pct = 0.80)
(h_smartphone : smartphone_pct = 0.60) 
: ∃ (min_people : ℕ), min_people = 3000 :=
sorry

end min_people_with_all_items_in_Owlna_l816_816323


namespace infinite_sum_result_l816_816132

def S (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), 1 / (k + 1)^2

theorem infinite_sum_result :
  (∑' n : ℕ, 1 / ((n + 2) * S n * S (n + 1))) = 1 := 
sorry

end infinite_sum_result_l816_816132


namespace cos_315_eq_sqrt2_div_2_l816_816530

theorem cos_315_eq_sqrt2_div_2 :
  let θ := 315.0
  in θ = 360.0 - 45.0 ∧ (cos 45 * (π / 180)) = (Real.sqrt 2 / 2) →
  (cos (θ * (π / 180))) = (Real.sqrt 2 / 2) :=
by
  intros θ
  -- Non-necessary proof omitted
  sorry

end cos_315_eq_sqrt2_div_2_l816_816530


namespace shaded_areas_equal_l816_816730

theorem shaded_areas_equal (s : ℝ) (φ : ℝ) (h : 0 < φ ∧ φ < π / 4) :
  tan φ = 3 * φ ↔ (shaded_area_cond s φ) := sorry

def shaded_area_cond (s : ℝ) (φ : ℝ) : Prop := 
  let sector_area := (φ * s^2) / 2
  let triangle_area :=  (s^2 * tan φ) / 2
  sector_area = triangle_area

end shaded_areas_equal_l816_816730


namespace number_of_intersections_l816_816394

noncomputable def cos75 := real.cos (75 * real.pi / 180)
noncomputable def sin75 := real.sin (75 * real.pi / 180)
noncomputable def tan75 := sin75 / cos75

def parametric_line (t : ℝ) := (t * cos75, t * sin75)
def parametric_curve (θ : ℝ) := (3 * real.sin θ, 2 * real.cos θ)

theorem number_of_intersections : (∃ t θ : ℝ, parametric_line t = parametric_curve θ) → sorry = 2 :=
by
  sorry

end number_of_intersections_l816_816394


namespace curve_symmetry_origin_curve_not_symmetry_line_y_eq_x_area_enclosed_by_curve_gt_pi_area_enclosed_by_curve_not_lt_pi_l816_816576

theorem curve_symmetry_origin (x y : ℝ) : x^4 + y^2 = 1 → (-x)^4 + (-y)^2 = 1 := 
by 
  intros h 
  simp [h]

theorem curve_not_symmetry_line_y_eq_x (x y : ℝ) : ¬(x^4 + y^2 = 1 ↔ y^4 + x^2 = 1) := 
by 
  intro h
  have : ∀ a b : ℝ, a^4 + b^2 = b^4 + a^2 ↔ a = b 
    by 
      simp
  sorry -- Add further proofs for irrelevance

theorem area_enclosed_by_curve_gt_pi : 
  let f (x y : ℝ) := x^4 + y^2 = 1 in 
  ∃ A > pi, ∀ x y, f x y → A > pi := 
by 
  sorry -- Add detailed proof of integration over the curve area

theorem area_enclosed_by_curve_not_lt_pi : 
  let f (x y : ℝ) := x^4 + y^2 = 1 in 
  ¬ ∃ A, ∀ x y, f x y → A < pi := 
by 
  sorry -- Add detailed proof of integration over the curve area

end curve_symmetry_origin_curve_not_symmetry_line_y_eq_x_area_enclosed_by_curve_gt_pi_area_enclosed_by_curve_not_lt_pi_l816_816576


namespace nearest_adjacent_points_inequality_l816_816771

noncomputable def circle_length := 1

theorem nearest_adjacent_points_inequality {α : ℝ} (hα_irrational : irrational α) (hα_range : 0 < α ∧ α < 1)
  {n : ℕ} (hn : n ≥ 3) (P : ℕ → ℝ) (hP_def : ∀ k, 2 ≤ k ∧ k ≤ n → P k = (P (k - 1) + α) % circle_length)
  (a b : ℕ) (h_neighborhood : ∀ {m}, P m = P n → m = n) :
  (a + b ≤ n) :=
sorry

end nearest_adjacent_points_inequality_l816_816771


namespace mine_placement_count_l816_816097

theorem mine_placement_count : 
  ∀ grid : Matrix (Fin 3) (Fin 4) ℕ, 
  (grid ! (1, 1) = 2) →
  (grid ! (2, 2) = 3) →
  (∀ (i j : Fin 3) (ni nj : Fin 4), (1 <= ni) && (1 <= nj) && (| (i, j) - (ni, nj) | ≤ 1) → (grid ! (ni, nj) = 0 ∨ grid ! (ni, nj) = 1)) → 
  (count_ways grid = 96) :=
by
  sorry

end mine_placement_count_l816_816097


namespace sin_minus_cos_l816_816252

theorem sin_minus_cos (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < (Real.pi / 2)) (hθ3 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := by
  sorry

end sin_minus_cos_l816_816252


namespace complement_U_A_l816_816187

-- Define the sets U and A
def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1, 2}

-- State the theorem
theorem complement_U_A :
  U \ A = {0} :=
sorry

end complement_U_A_l816_816187


namespace pq_identity_l816_816353

theorem pq_identity (p q : ℝ) (h1 : p * q = 20) (h2 : p + q = 10) : p^2 + q^2 = 60 :=
sorry

end pq_identity_l816_816353


namespace sequence_natural_and_increasing_l816_816841

theorem sequence_natural_and_increasing
  (a q : ℕ) (h1 : q > 1) (h2 : ∀ n : ℕ, n > 0 → 
    (∃ k : ℕ, a_n = a * q ^ (n - 1) ∨ 
      (even n → a_{n+1} = 2 * a_n - a_{n-1}) ∧
      (odd n → a_{n+1} = (a_{n}^2) / (a_{n-1})
    ) :
  ∀ n : ℕ, a_n ∈ ℕ ∧ a_n < a_{n+1} :=
by
  sorry

end sequence_natural_and_increasing_l816_816841


namespace f_of_2012_l816_816640

noncomputable def f : ℝ → ℝ := sorry
axiom h1 : ∀ x : ℝ, f(x + 6) + f(x) = 2 * f(3)
axiom h2 : ∀ x : ℝ, f(x - 1) = f(2 - (x - 1))
axiom h3 : f(4) = 4

theorem f_of_2012 : f(2012) = -4 := by
  sorry

end f_of_2012_l816_816640


namespace book_arrangement_problem_l816_816404

-- Define the number of ways to arrange 3 mathematics books
def arrange_math_books := (3!).toNat

-- Define the number of ways to insert 3 Chinese books such that they are not adjacent to each other or any mathematics books
def insert_chinese_books := 2 * arrange_math_books

-- Define the total number of arrangements
def total_arrangements := insert_chinese_books * arrange_math_books

-- Theorem stating the number of ways to arrange the books such that no books of the same type are adjacent is 72
theorem book_arrangement_problem : total_arrangements = 72 := 
by
  -- Here, no proof steps are required as per the instructions. The "sorry" placeholder is used to skip the proof.
  sorry

end book_arrangement_problem_l816_816404


namespace initialMoneyDistribution_l816_816379

-- Definitions for the final provision of pence
def Pence := Int

structure PlayerMoney :=
  (Adams Baker Carter Dobson Edwards Francis Gudgeon : Pence)

-- Given final condition
def finalMoney : PlayerMoney :=
  { Adams := 32, Baker := 32, Carter := 32, Dobson := 32, Edwards := 32, Francis := 32, Gudgeon := 32 }

-- Conditions before each game
def preGame7 (p : PlayerMoney) : Prop :=
  p.Adams = 16 ∧ p.Baker = 16 ∧ p.Carter = 16 ∧ p.Dobson = 16 ∧ p.Edwards = 16 ∧ p.Francis = 16 ∧ p.Gudgeon = 32

-- Repeat Similarly for other games
def preGame6 (p : PlayerMoney) : Prop :=
  p.Adams = 8 ∧ p.Baker = 8 ∧ p.Carter = 8 ∧ p.Dobson = 8 ∧ p.Edwards = 8 ∧ p.Francis = 16 ∧ p.Gudgeon = 16

def preGame5 (p : PlayerMoney) : Prop :=
  p.Adams = 4 ∧ p.Baker = 4 ∧ p.Carter = 4 ∧ p.Dobson = 4 ∧ p.Edwards = 8 ∧ p.Francis = 8 ∧ p.Gudgeon = 8

def preGame4 (p : PlayerMoney) : Prop :=
  p.Adams = 2 ∧ p.Baker = 2 ∧ p.Carter = 2 ∧ p.Dobson = 4 ∧ p.Edwards = 4 ∧ p.Francis = 4 ∧ p.Gudgeon = 4

def preGame3 (p : PlayerMoney) : Prop :=
  p.Adams = 1 ∧ p.Baker = 1 ∧ p.Carter = 2 ∧ p.Dobson = 2 ∧ p.Edwards = 2 ∧ p.Francis = 2 ∧ p.Gudgeon = 2

def preGame2 (p : PlayerMoney) : Prop :=
  p.Adams = 0.5 ∧ p.Baker = 1 ∧ p.Carter = 1 ∧ p.Dobson = 1 ∧ p.Edwards = 1 ∧ p.Francis = 1 ∧ p.Gudgeon = 1

def preGame1 (p : PlayerMoney) : Prop :=
  p.Adams = 0.5 ∧ p.Baker = 0.25 ∧ p.Carter = 0.25 ∧ p.Dobson = 0.25 ∧ p.Edwards = 0.25 ∧ p.Francis = 0.25 ∧ p.Gudgeon = 0.25

-- Prove the initial money distribution
theorem initialMoneyDistribution : 
  ∃ p : PlayerMoney, preGame1 p ∧ preGame2 (PlayerMoney.mk p.Adams (2 * p.Baker) (2 * p.Carter) (2 * p.Dobson) (2 * p.Edwards) (2 * p.Francis) (2 * p.Gudgeon)) ∧
  preGame3 (PlayerMoney.mk p.Adams (2 * p.Baker) (2 * p.Carter) (2 * p.Dobson) (2 * p.Edwards) (2 * p.Francis) (2 * p.Gudgeon)) ∧
  preGame4 (PlayerMoney.mk p.Adams (2 * p.Baker) (2 * p.Carter) (2 * p.Dobson) (2 * p.Edwards) (2 * p.Francis) (2 * p.Gudgeon)) ∧
  preGame5 (PlayerMoney.mk p.Adams (2 * p.Baker) (2 * p.Carter) (2 * p.Dobson) (2 * p.Edwards) (2 * p.Francis) (2 * p.Gudgeon)) ∧
  preGame6 (PlayerMoney.mk p.Adams (2 * p.Baker) (2 * p.Carter) (2 * p.Dobson) (2 * p.Edwards) (2 * p.Francis) (2 * p.Gudgeon)) ∧
  preGame7 (PlayerMoney.mk (2 * p.Adams) (2 * p.Baker) (2 * p.Carter) (2 * p.Dobson) (2 * p.Edwards) (2 * p.Francis) (32)) ∧
  finalMoney (PlayerMoney.mk (2 * p.Adams) (2 * p.Baker) (2 * p.Carter) (2 * p.Dobson) (2 * p.Edwards) (2 * p.Francis) (32)) := sorry

end initialMoneyDistribution_l816_816379


namespace fruit_weight_sister_and_dad_l816_816802

-- Defining the problem statement and conditions
variable (strawberries_m blueberries_m raspberries_m : ℝ)
variable (strawberries_d blueberries_d raspberries_d : ℝ)
variable (strawberries_s blueberries_s raspberries_s : ℝ)
variable (total_weight : ℝ)

-- Given initial conditions
def conditions : Prop :=
  strawberries_m = 5 ∧
  blueberries_m = 3 ∧
  raspberries_m = 6 ∧
  strawberries_d = 2 * strawberries_m ∧
  blueberries_d = 2 * blueberries_m ∧
  raspberries_d = 2 * raspberries_m ∧
  strawberries_s = strawberries_m / 2 ∧
  blueberries_s = blueberries_m / 2 ∧
  raspberries_s = raspberries_m / 2 ∧
  total_weight = (strawberries_m + blueberries_m + raspberries_m) + 
                 (strawberries_d + blueberries_d + raspberries_d) + 
                 (strawberries_s + blueberries_s + raspberries_s)

-- Defining the property to prove
theorem fruit_weight_sister_and_dad :
  conditions strawberries_m blueberries_m raspberries_m strawberries_d blueberries_d raspberries_d strawberries_s blueberries_s raspberries_s total_weight →
  (strawberries_d + blueberries_d + raspberries_d) +
  (strawberries_s + blueberries_s + raspberries_s) = 35 := by
  sorry

end fruit_weight_sister_and_dad_l816_816802


namespace coeff_x3_in_expansion_l816_816732

theorem coeff_x3_in_expansion : 
  (∑ k in (finset.range 7), (binomial 6 k) * x^(k + 1)) + 0 = 15 :=
sorry

end coeff_x3_in_expansion_l816_816732


namespace find_x_range_l816_816661

-- Define g(x) for x < 0
def g_neg(x : ℝ) [h : x < 0] : ℝ := -Real.log(1 - x)

-- Define the odd function g(x)
def g(x : ℝ) : ℝ := if x < 0 then g_neg x else Real.log(1 + x)

-- Define f(x) as a piecewise function
def f(x : ℝ) : ℝ := if x ≤ 0 then x^3 else g(x)

-- Define the main statement to be proved
theorem find_x_range (x : ℝ) : f(2 - x^2) > f(x) ↔ x ∈ Ioo (-2 : ℝ) 1 :=
sorry

end find_x_range_l816_816661


namespace sum_represent_all_integers_l816_816034

theorem sum_represent_all_integers : ∃ (n : ℕ), n = 798 ∧ (∀ k : ℤ, (798 ≤ k ∧ k ≤ 898) → ∃ (a : Fin n → ℕ), 
  (∀ i : Fin n, 1 ≤ a i ∧ a i ≤ n ∧ ∃ j : Fin n, (a j = i + 1)) ∧ 
  k = ∑ i : Fin n, (a i : ℚ) / (i + 1)) :=
by 
  sorry

end sum_represent_all_integers_l816_816034


namespace sum_of_a_coefficients_l816_816139

theorem sum_of_a_coefficients:
  (∀ x : ℝ, (2 * x - 3) ^ 10 = (∑ i in finset.range 11, (a_coeff i) * (x - 1)^i)) →
  a_coeff 0 = 1 →
  (∑ i in finset.range 11 \ {0}, a_coeff i) = 0 :=
by
  sorry

end sum_of_a_coefficients_l816_816139


namespace determine_k_and_solution_l816_816947

theorem determine_k_and_solution :
  ∃ (k : ℚ), (5 * k * x^2 + 30 * x + 10 = 0 → k = 9/2) ∧
    (∃ (x : ℚ), (5 * (9/2) * x^2 + 30 * x + 10 = 0) ∧ x = -2/3) := by
  sorry

end determine_k_and_solution_l816_816947


namespace marker_and_notebook_cost_l816_816819

noncomputable def cost_of_marker_and_notebook : ℕ :=
let m := 10
let n := 4 in
m + n

theorem marker_and_notebook_cost {m n : ℕ} (hmn : 10 * m + 5 * n = 120) (hmarker_gt_notebook : m > n) (hn : n > 0) :
  m + n = 14 :=
by {
  have h1 : 2 * m + n = 24, by { linarith },
  have good_m_and_n : ∃ m n : ℕ, 2 * m + n = 24 ∧ m > n ∧ n > 0, from ⟨10, 4, by linarith, by linarith, by linarith⟩,
  cases good_m_and_n with m' hn'',
  cases hn'' with n' hn'''',
  exact hn'''').2.1
}

end marker_and_notebook_cost_l816_816819


namespace y1_gt_y2_for_line_through_points_l816_816653

theorem y1_gt_y2_for_line_through_points (x1 y1 x2 y2 k b : ℝ) 
  (h_line_A : y1 = k * x1 + b) 
  (h_line_B : y2 = k * x2 + b) 
  (h_k_neq_0 : k ≠ 0)
  (h_k_pos : k > 0)
  (h_b_nonneg : b ≥ 0)
  (h_x1_gt_x2 : x1 > x2) : 
  y1 > y2 := 
  sorry

end y1_gt_y2_for_line_through_points_l816_816653


namespace exists_reflection_point_l816_816335

-- Hypotheses: We are given two points A and B inside a concave spherical mirror
-- with center at the origin, and radius r.

variables {r : ℝ} {x1 y1 x2 y2 : ℝ} 

-- Coordinates of A and B respectively
def A := (x1, y1)
def B := (x2, y2)

-- Reflection Point X
def reflectionPoint (x y : ℝ) := 
  (x^2 + y^2 = r^2) ∧ 
  ((x1 * y2 + x2 * y1) * (x^2 - y^2) - 2 * (x1 * x2 - y1 * y2) * x * y + r^2 * ((x1 + x2) * y - (y1 + y2) * x) = 0)

-- Theorem statement in Lean
theorem exists_reflection_point : 
  ∃ (x y : ℝ), reflectionPoint (x y) :=
sorry

end exists_reflection_point_l816_816335


namespace number_of_integer_pairs_l816_816160

theorem number_of_integer_pairs (a b : ℕ) (h_pos: 0 < a)
(h_pos2: 0 < b) (h_eq: (1 / a : ℝ) - (1 / b : ℝ) = 1 / 2018) :
  ∃! (a, b : ℕ), (1 / a : ℝ) - (1 / b : ℝ) = 1 / 2018 :=
by
  sorry

end number_of_integer_pairs_l816_816160


namespace solve_x4_eq_16_l816_816612

theorem solve_x4_eq_16 (x : ℂ) : x^4 - 16 = 0 ↔ x = 2 ∨ x = -2 ∨ x = 2 * complex.I ∨ x = -2 * complex.I :=
by
  sorry

end solve_x4_eq_16_l816_816612


namespace point_on_coordinate_axes_l816_816146

-- Definitions and assumptions from the problem conditions
variables {a b : ℝ}

-- The theorem statement asserts that point M(a, b) must be located on the coordinate axes given ab = 0
theorem point_on_coordinate_axes (h : a * b = 0) : 
  (a = 0) ∨ (b = 0) :=
by
  sorry

end point_on_coordinate_axes_l816_816146


namespace find_solutions_in_positive_integers_l816_816959

theorem find_solutions_in_positive_integers :
  ∃ a b c x y z : ℕ,
  a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧
  a + b + c = x * y * z ∧ x + y + z = a * b * c ∧
  ((a = 3 ∧ b = 2 ∧ c = 1 ∧ x = 3 ∧ y = 2 ∧ z = 1) ∨
   (a = 5 ∧ b = 2 ∧ c = 1 ∧ x = 3 ∧ y = 3 ∧ z = 1) ∨
   (a = 3 ∧ b = 3 ∧ c = 1 ∧ x = 5 ∧ y = 2 ∧ z = 1)) :=
sorry

end find_solutions_in_positive_integers_l816_816959


namespace area_of_enclosed_region_l816_816962

theorem area_of_enclosed_region :
  let s := { p : ℝ × ℝ | |p.1 - 50| + |p.2| = |p.1 / 5| }
  (measure_theory.measure_space.volume (set.univ.inter s)).to_real / 2 = 347.5 :=
by
  sorry

end area_of_enclosed_region_l816_816962


namespace max_value_x_l816_816120

theorem max_value_x (x : ℝ) :
(6 + 5 * x + x^2) * sqrt (2 * x^2 - x^3 - x) ≤ 0 → x ≤ 1 := 
sorry

example : max_value_x 1 := 
by sorry

end max_value_x_l816_816120


namespace distinct_ordered_pairs_eq_49_l816_816173

open Nat

theorem distinct_ordered_pairs_eq_49 (a b : ℕ) (h1 : a + b = 50) (h2 : a > 0) (h3 : b > 0) :
  num_solutions (λ p : ℕ × ℕ, p.1 + p.2 = 50 ∧ p.1 > 0 ∧ p.2 > 0) = 49 :=
sorry

end distinct_ordered_pairs_eq_49_l816_816173


namespace find_x_range_l816_816577

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x) + abs x

-- Define the theorem to prove the desired range for x
theorem find_x_range : { x : ℝ | f (2 * x - 1) < f 3 } = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end find_x_range_l816_816577


namespace cos_315_eq_sqrt2_div_2_l816_816526

theorem cos_315_eq_sqrt2_div_2 :
  let θ := 315.0
  in θ = 360.0 - 45.0 ∧ (cos 45 * (π / 180)) = (Real.sqrt 2 / 2) →
  (cos (θ * (π / 180))) = (Real.sqrt 2 / 2) :=
by
  intros θ
  -- Non-necessary proof omitted
  sorry

end cos_315_eq_sqrt2_div_2_l816_816526


namespace maximum_value_of_function_l816_816138

theorem maximum_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1/2) :
  ∃ M, (∀ y, y = x * (1 - 2 * x) → y ≤ M) ∧ M = 1/8 :=
sorry

end maximum_value_of_function_l816_816138


namespace classroom_children_count_l816_816408

theorem classroom_children_count
  (C : ℕ)
  (h1 : 1 / 3 * C ∈ ℕ)
  (h2 : 2 / 3 * C = 30) :
  C = 45 := 
sorry

end classroom_children_count_l816_816408


namespace sin_minus_cos_l816_816235

noncomputable def theta_condition (θ : ℝ) : Prop := (0 < θ) ∧ (θ < π / 2) ∧ (Real.tan θ = 1 / 3)

theorem sin_minus_cos (θ : ℝ) (hθ : theta_condition θ) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_minus_cos_l816_816235


namespace part_a_part_b_l816_816590

-- Define the players
inductive Player
| Ann | Bob | Con | Dot | Eve | Fay | Guy | Hal

open Player

-- No group of five players has all possible games played among them
def no_group_of_five (g : Graph Player) : Prop :=
  ∀ (s : Finset Player), s.card = 5 → ∃ v w, v ∈ s ∧ w ∈ s ∧ ¬g.adj v w

-- Part (a): Construct an arrangement of 24 games satisfying the conditions
theorem part_a : ∃ (g : Graph Player), g.edge_finset.card = 24 ∧ no_group_of_five g := 
sorry

-- Part (b): Show that it's impossible to have more than 24 games satisfying the conditions
theorem part_b : ∀ (g : Graph Player), no_group_of_five g → g.edge_finset.card ≤ 24 := 
sorry

end part_a_part_b_l816_816590


namespace problem_equiv_l816_816110

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry
noncomputable def h (x : ℝ) : ℝ := sorry

theorem problem_equiv (f g h : ℝ → ℝ):
  (∀ x y : ℝ, (x - y) * f x + h x - x * y + y^2 ≤ h y ∧ h y ≤ (x - y) * g x + h x - x * y + y^2) →
  (∃ a b : ℝ, f = λ x, -x + a ∧ g = λ x, -x + a ∧ h = λ x, x^2 - a * x + b) :=
begin
  sorry
end

end problem_equiv_l816_816110


namespace curve_eq_proof_l816_816975

open Real

noncomputable def curve_eq (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + 4

theorem curve_eq_proof :
  (∇ f : ℝ → ℝ, (∀ y : ℝ, f y = (x^2 - 2*x)) → (f 3 = 4) :=
begin
  sorry
end

end curve_eq_proof_l816_816975


namespace trajectory_equation_line_equation_l816_816993

noncomputable def circle_trajectory (x y : ℝ) : Prop :=
  (x + real.sqrt 3) ^ 2 + y ^ 2 = 16

theorem trajectory_equation :
  ∀ (x y : ℝ), circle_trajectory x y →
  ∃ (a : ℝ), 
    (a > 0) ∧ 
    ellipse_trajectory : (real.sqrt x x^2 + 4*y^2) := 
  sorry

theorem line_equation :
  ∀ (A B C : (ℝ × ℝ)),
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 →
  symmetric (A B C) ∧ 
  line_symmetry (A B C) ∧
  minimum_area_ABC =
    (eqn: line_eq := 
    (y = x) ∨ (y = -x)) :=
  sorry

end trajectory_equation_line_equation_l816_816993


namespace sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816210

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816210


namespace binomial_8_5_l816_816078

theorem binomial_8_5 : nat.choose 8 5 = 56 := by
  sorry

end binomial_8_5_l816_816078


namespace ratio_perimeter_divided_by_a_l816_816400

open Real

noncomputable def square_vertices (a : ℝ) : set (ℝ × ℝ) :=
  {(-a, -a), (a, -a), (-a, a), (a, a)}

def cutting_line (x : ℝ) : ℝ := x

def perimeter_triangle (a : ℝ) : ℝ :=
  let side := 2 * a in side + side + side

theorem ratio_perimeter_divided_by_a (a : ℝ) (h : 0 < a) :
  let triangle_perimeter := perimeter_triangle a in
  triangle_perimeter / a = 6 :=
by
  sorry

end ratio_perimeter_divided_by_a_l816_816400


namespace least_cost_l816_816388

noncomputable def total_length : ℝ := 10 * (6 / 12)
def cost_6_foot_rope : ℝ := 5
def cost_per_foot_6_foot_rope : ℝ := cost_6_foot_rope / 6
def cost_1_foot_rope : ℝ := 1.25
def total_cost_1_foot : ℝ := 5 * cost_1_foot_rope

theorem least_cost : min cost_6_foot_rope total_cost_1_foot = 5 := 
by sorry

end least_cost_l816_816388


namespace find_q_l816_816677

variable (p q : ℝ)
variable (h1 : 1 < p)
variable (h2 : p < q)
variable (h3 : 1 / p + 1 / q = 1)
variable (h4 : p * q = 8)

theorem find_q : q = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end find_q_l816_816677


namespace jackson_markup_l816_816338

theorem jackson_markup :
  (∀ (S : ℝ), S = 8 -> 
   ∀ (profit_percent expense_percent : ℝ), profit_percent = 0.20 -> expense_percent = 0.10 ->
   let C := S - (S * expense_percent + S * profit_percent) in 
   (S - C) / C * 100 ≈ 43) := 
by
  intro S hS profit_percent hProfit expense_percent hExpense,
  simp [hS, hProfit, hExpense],
  let C := S - (S * expense_percent + S * profit_percent),
  calc
    (S - C) / C * 100
      = (S - (S * (1 - (1 - expense_percent - profit_percent)))) / (S * (1 - expense_percent - profit_percent)) * 100 : by
        norm_num [C]
      ... 
      sorry

end jackson_markup_l816_816338


namespace base5_division_l816_816596

-- Given conditions in decimal:
def n1_base10 : ℕ := 214
def n2_base10 : ℕ := 7

-- Convert the result back to base 5
def result_base5 : ℕ := 30  -- since 30 in decimal is 110 in base 5

theorem base5_division (h1 : 1324 = 214) (h2 : 12 = 7) : 1324 / 12 = 110 :=
by {
  -- these conditions help us bridge to the proof (intentionally left unproven here)
  sorry
}

end base5_division_l816_816596


namespace find_solutions_l816_816602

noncomputable def solutions_to_equation : set Complex :=
  {x | x^4 - 16 = 0}

theorem find_solutions : solutions_to_equation = {2, -2, Complex.I * 2, -Complex.I * 2} :=
sorry

end find_solutions_l816_816602


namespace probability_of_usable_gas_pipe_l816_816063

theorem probability_of_usable_gas_pipe (x y : ℝ)
  (hx : 75 ≤ x) 
  (hy : 75 ≤ y)
  (hxy : x + y ≤ 225) :
  (∃ x y, 0 < x ∧ 0 < y ∧ x < 300 ∧ y < 300 ∧ x + y > 75 ∧ (300 - x - y) ≥ 75) → 
  ((150 * 150) / (300 * 300 / 2) = (1 / 4)) :=
by {
  sorry
}

end probability_of_usable_gas_pipe_l816_816063


namespace greatest_expression_is_B_l816_816102

-- Definition of the expressions listed as options
noncomputable def exprA := Real.sqrt (Real.cbrt (6 * 7))
noncomputable def exprB := Real.sqrt (7 * Real.cbrt 6)
noncomputable def exprC := Real.sqrt (6 * Real.cbrt 7)
noncomputable def exprD := Real.cbrt (6 * Real.sqrt 7)
noncomputable def exprE := Real.cbrt (7 * Real.sqrt 6)

-- The proof statement for the greatest expression
theorem greatest_expression_is_B : 
  exprB > exprA ∧ exprB > exprC ∧ exprB > exprD ∧ exprB > exprE :=
sorry

end greatest_expression_is_B_l816_816102


namespace cannon_acceleration_correct_travel_time_correct_kinetic_energy_correct_efficiency_per_kg_correct_temperature_increase_correct_l816_816956

noncomputable def cannon_acceleration (s : ℝ) (v : ℝ) (u : ℝ) : ℝ :=
  (v^2 - u^2) / (2 * s)

noncomputable def travel_time (v : ℝ) (u : ℝ) (a : ℝ) : ℝ :=
  (v - u) / a

noncomputable def kinetic_energy (m : ℝ) (v : ℝ) : ℝ :=
  0.5 * m * v^2

noncomputable def efficiency_per_kg (E : ℝ) (m : ℝ) : ℝ :=
  E / m

noncomputable def temperature_increase (Q : ℝ) (m : ℝ) (c : ℝ) : ℝ :=
  Q / (m * c)

-- Define the conditions
def s := 12.2 -- length of the barrel in meters
def Mgunpowder := 90  -- mass of the gunpowder in kg
def Mcannon := 292  -- mass of the cannon in kg
def v := 800  -- final velocity in m/s
def u := 0  -- initial velocity in m/s
def c := 120  -- specific heat capacity in J/(kg°C)
def conversion_ratio := 0.7 -- conversion ratio for heat energy

-- Prove the acceleration
theorem cannon_acceleration_correct : cannon_acceleration s v u = 26229.5 := by
  sorry

-- Prove the time taken to travel the length of the barrel
theorem travel_time_correct : travel_time v u (cannon_acceleration s v u) = 0.0305 := by
  sorry

-- Prove the kinetic energy of the shell
theorem kinetic_energy_correct : kinetic_energy Mcannon v = 93440000 := by
  sorry

-- Prove the energy per kilogram of gunpowder
theorem efficiency_per_kg_correct : efficiency_per_kg (kinetic_energy Mcannon v) Mgunpowder = 1038222.22 := by
  sorry

-- Prove the temperature increase of the shell
theorem temperature_increase_correct : temperature_increase ((kinetic_energy Mcannon v) * conversion_ratio) Mcannon c = 1866 := by
  sorry

end cannon_acceleration_correct_travel_time_correct_kinetic_energy_correct_efficiency_per_kg_correct_temperature_increase_correct_l816_816956


namespace lines_intersect_at_single_point_l816_816450

-- Define the geometric entities
variables {Point Line : Type}
variables [linear_ordered_comm_ring ℝ]

-- Assume the existence of a function for the center of the circle
def circumcenter (A B C : Point) : Point := sorry

-- Assume the existence of a function for the altitude midpoint
def altitude_midpoint (A B C : Point) : Point := sorry

-- Define the specific parallel lines through midpoints parallel to given lines
def line_parallel_to (p1 p2 : Point) : Line := sorry

-- We need a definition of intersection of three lines
def lines_intersect_at_point (a b c: Line) : Point := sorry

-- Definitions based on given conditions
noncomputable def line_a (A B C : Point) : Line :=
  let O := circumcenter A B C in
  line_parallel_to (altitude_midpoint A B C) O

noncomputable def line_b (A B C : Point) : Line :=
  let O := circumcenter A B C in
  line_parallel_to (altitude_midpoint B A C) O

noncomputable def line_c (A B C : Point) : Line :=
  let O := circumcenter A B C in
  line_parallel_to (altitude_midpoint C A B) O

-- Prove the existence of a single intersection point
theorem lines_intersect_at_single_point :
  ∀ (A B C : Point),
    ∃ P : Point, P = lines_intersect_at_point (line_a A B C) (line_b A B C) (line_c A B C) :=
sorry

end lines_intersect_at_single_point_l816_816450


namespace combined_proposition_l816_816156

variable (a : ℝ)

def P : Prop := ∀ x ∈ set.Icc 1 2, x^2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem combined_proposition (h : P a ∧ q a) : a ≤ -2 ∨ a = 1 := 
sorry

end combined_proposition_l816_816156


namespace geometric_sequence_prediction_odd_or_even_l816_816952

/-- Condition: The first 50-meter return run is mandatory. -/
def P1 := 1

/-- Condition: Participants roll two fair dice to determine the number of return runs.-/
def probability_one_return_run := (1 : ℚ) / 3    -- Probability that sum of dice is divisible by 3

def probability_two_return_runs := (2 : ℚ) / 3   -- Probability that sum of dice is not divisible by 3

/-- The probability at each step:
P_2: The participant does one return run in the second round.
P_3: The participant does two return runs in the second round or one in the second and third rounds.
-/
noncomputable def P2 := probability_one_return_run

noncomputable def P3 := probability_two_return_runs + (probability_one_return_run * probability_one_return_run)

lemma P1_def : P1 = 1 := rfl
lemma P2_def : P2 = 1 / 3 := rfl
lemma P3_def : P3 = 2 / 3 + 1 / 9 := by norm_num  -- Simplified to 7 / 9

/-- The proof that the sequence {P_n - P_{n-1}} forms a geometric sequence with common ratio -2/3.-/
theorem geometric_sequence : ∀ (n : ℕ), n ≥ 2 → 
  (P_n - P_(n-1)) = -(2 : ℚ) / 3 * (P_(n-1) - P_(n-2)) :=
sorry

/-- P_n formula and prediction of even or odd return runs. -/
noncomputable def Pn (n : ℕ) : ℚ := (3 / 5) * (1 - (-2 / 3)^n)

theorem prediction_odd_or_even (n : ℕ) : (2 * (Pn n).denom ≤ 1 → True) :=
sorry

end geometric_sequence_prediction_odd_or_even_l816_816952


namespace primes_have_property_P_infinitely_many_composites_have_property_P_l816_816888

def property_P (n : ℕ) : Prop :=
  ∀ (a : ℕ), a > 0 → n ∣ a^n - 1 → n^2 ∣ a^n - 1

theorem primes_have_property_P : ∀ (p : ℕ), Prime p → property_P p :=
  sorry

theorem infinitely_many_composites_have_property_P : 
  ∃ (f : ℕ → ℕ), (∀ k, Composite (f k)) ∧ (∀ k, property_P (f k)) :=
  sorry

end primes_have_property_P_infinitely_many_composites_have_property_P_l816_816888


namespace parabola_focus_l816_816091

theorem parabola_focus : 
  ∀ x: ℝ, (∃ (f : ℝ → ℝ), (∀ x, f x = 4 * (x - 1)^2 - 3) → (1, -47/16) = (1, -3 + 1/(4 * 4))) :=
by
  intro x
  use (λ x, 4 * (x - 1)^2 - 3)
  intro h
  rw h
  have h₁ : 1 = 1 := rfl
  have h₂ : -47/16 = -3 + 1/(4 * 4) := by norm_num
  exact ⟨h₁, h₂⟩

end parabola_focus_l816_816091


namespace sum_absolute_b_eq_fraction_l816_816972

def P (x : ℚ) : ℚ :=
  1 - (2 / 5) * x + (1 / 8) * x^2 + (1 / 10) * x^3

noncomputable def Q (x : ℚ) : ℚ :=
  P(x) * P(x^4) * P(x^6) * P(x^8)

noncomputable def b : List ℚ :=
  (Polynomial.coeff (Q (Polynomial.C 1))).coeffs

noncomputable def abs_sum_b : ℚ :=
  b.sum (fun coeff => abs coeff)

theorem sum_absolute_b_eq_fraction :
  abs_sum_b = ((43 : ℚ) / 40)^4 :=
by
  sorry

end sum_absolute_b_eq_fraction_l816_816972


namespace sum_slope_y_intercept_l816_816093

def point := (ℕ × ℕ)
def slope (p1 p2 : point) : ℚ := (p2.2 - p1.2) / (p2.1 - p1.1)
def y_intercept (m : ℚ) (p : point) : ℚ := p.2 - m * p.1

theorem sum_slope_y_intercept {p1 p2 : point} (h₁ : p1 = (1, 3)) (h₂ : p2 = (3, 11)) :
  let m := slope p1 p2,
      b := y_intercept m p1
  in m + b = 3 :=
by
  sorry

end sum_slope_y_intercept_l816_816093


namespace trig_identity_l816_816283

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end trig_identity_l816_816283


namespace sum_of_nine_consecutive_even_integers_mod_10_l816_816013

theorem sum_of_nine_consecutive_even_integers_mod_10 : 
  (10112 + 10114 + 10116 + 10118 + 10120 + 10122 + 10124 + 10126 + 10128) % 10 = 0 := by
  sorry

end sum_of_nine_consecutive_even_integers_mod_10_l816_816013


namespace last_five_digits_correct_l816_816511

-- Defining the conditions
def isSequenceCorrect : Prop :=
  let digits := String.mk ["2", "2", "0", "0", "0", "2"]
  let total_digits := 1250
  digits.drop (total_digits - 1250 + 1245) = "20002"

-- Theorem to prove the last five digits being "20002"
theorem last_five_digits_correct :
  isSequenceCorrect = true :=
by
  sorry

end last_five_digits_correct_l816_816511


namespace tan_of_alpha_l816_816706

noncomputable def point_P : ℝ × ℝ := (1, -2)

theorem tan_of_alpha (α : ℝ) (h : ∃ (P : ℝ × ℝ), P = point_P ∧ P.2 / P.1 = -2) :
  Real.tan α = -2 :=
sorry

end tan_of_alpha_l816_816706


namespace find_solutions_l816_816605

noncomputable def solutions_to_equation : set Complex :=
  {x | x^4 - 16 = 0}

theorem find_solutions : solutions_to_equation = {2, -2, Complex.I * 2, -Complex.I * 2} :=
sorry

end find_solutions_l816_816605


namespace problem_statements_l816_816654

-- Define the given points and conditions
def A : (ℝ × ℝ) := (2, 3)
def B : (ℝ × ℝ) := (4, 1)
def l (x y : ℝ) := x - 2 * y + 2 = 0

-- Definition for coordinates of C
def C := (4, 3)

-- Proof of problem statements
theorem problem_statements :
  (l C.1 C.2) ∧
  (abs ((A.1 - B.1) * (A.2 - C.2) - (A.1 - C.1) * (A.2 - B.2)) = 4) ∧
  (∃ (a b : ℝ), (a > 0 ∧ b > 0 ∧ 4 / a + 3 / b = 1 ∧ a * b = 48 ∧ 
    abs ((1 / 2) * a * b) = 24 ∧ (a = 8 ∧ b = 6))) ∧
  (∃ (α : ℝ), (α = (π/4) ∧ abs ((1 / 2) * 6 * 8 / sin (2 * α)) = 24)) :=
  by
    -- Definitions and Propositions
    sorry

end problem_statements_l816_816654


namespace find_number_l816_816055

theorem find_number (x : ℝ) (h : x / 5 = 70 + x / 6) : x = 2100 := by
  sorry

end find_number_l816_816055


namespace group_partition_count_l816_816726

theorem group_partition_count :
  let men := 3,
      women := 4,
      group_of_four_men := 2,
      group_of_four_women := 2,
      group_of_four := (finset.card (finset.powerset (finset.range men)) = group_of_four_men) * 
                       (finset.card (finset.powerset (finset.range women)) = group_of_four_women)
  in group_of_four = 18 :=
by {
  let men := 3,
  let women := 4,
  let group_of_four_men := 2,
  let group_of_four_women := 2,
  have group_of_four_men_count : nat := nat.choose men group_of_four_men,
  have group_of_four_women_count : nat := nat.choose women group_of_four_women,
  have group_of_four := group_of_four_men_count * group_of_four_women_count,
  exact group_of_four
}

end group_partition_count_l816_816726


namespace no_constant_f_A_under_N_l816_816751

variable (p : ℕ)
variable (A : Set ℕ)

def f_A (n : ℕ) : ℕ := { x | x ∈ A }.to_finset.filter (λ x, x.val < n).card

theorem no_constant_f_A_under_N (hp : p.prime) (hA : A.infinite) : 
  ¬ ∃ N, ∀ n, n < N → f_A p A n = f_A p A (n + 1) :=
by
  sorry

end no_constant_f_A_under_N_l816_816751


namespace eval_g_at_3_l816_816686

def g (x : ℤ) : ℤ := 5 * x^3 + 7 * x^2 - 3 * x - 6

theorem eval_g_at_3 : g 3 = 183 := by
  sorry

end eval_g_at_3_l816_816686


namespace division_in_base_5_l816_816593

noncomputable def base5_quick_divide : nat := sorry

theorem division_in_base_5 (a b quotient : ℕ) (h1 : a = 1324) (h2 : b = 12) (h3 : quotient = 111) :
  ∃ c : ℕ, c = quotient ∧ a / b = quotient :=
by
  sorry

end division_in_base_5_l816_816593


namespace find_p_l816_816473

noncomputable def parabola (p : ℝ) : Set (ℝ × ℝ) :=
  { pxy : ℝ × ℝ | pxy.snd ^ 2 = 2 * p * pxy.fst }

def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

def is_centroid (O A B F : ℝ × ℝ) : Prop :=
  F = ((O.fst + A.fst + B.fst) / 3, (O.snd + A.snd + B.snd) / 3)

def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.fst - b.fst) ^ 2 + (a.snd - b.snd) ^ 2)

def sum_of_distances_eq (A B F : ℝ × ℝ) (d : ℝ) : Prop :=
  distance A F + distance B F = d

theorem find_p (p : ℝ) (O A B F : ℝ × ℝ)
  (h1 : parabola p A)
  (h2 : parabola p B)
  (h3 : sum_of_distances_eq A B F 10)
  (h4 : O = (0, 0))
  (h5 : focus p = F)
  (h6 : is_centroid O A B F)
: p = 4 := 
sorry

end find_p_l816_816473


namespace probability_AB_adjacent_l816_816137

theorem probability_AB_adjacent :
  let arrangement_count := 24 in -- Total number of arrangements
  let AB_adjacent_count := 12 in -- Number of arrangements where A and B are adjacent
  (AB_adjacent_count : ℚ) / arrangement_count = 1 / 2 :=
by
  sorry

end probability_AB_adjacent_l816_816137


namespace sample_mean_experimental_group_median_and_significance_l816_816926

namespace OzoneExperiment

def control_group : List ℝ := 
  [15.2, 18.8, 20.2, 21.3, 22.5, 23.2, 25.8, 26.5, 27.5, 30.1,
   32.6, 34.3, 34.8, 35.6, 35.6, 35.8, 36.2, 37.3, 40.5, 43.2]

def experimental_group : List ℝ := 
  [7.8, 9.2, 11.4, 12.4, 13.2, 15.5, 16.5, 18.0, 18.8, 19.2, 
   19.8, 20.2, 21.6, 22.8, 23.6, 23.9, 25.1, 28.2, 32.3, 36.5]

def combined : List ℝ :=
  control_group ++ experimental_group

-- Sample mean calculation
theorem sample_mean_experimental_group
  (ex_group_sum : ∑ x in experimental_group, x = 396 ) :
  (∑ x in experimental_group, x) / 20 = 19.8 :=
begin
  have divisor := 20,
  calc (∑ x in experimental_group, x) / divisor
      = 396 / divisor : by rw ex_group_sum
  ... = 19.8 : by norm_num
end

-- Median calculation and significance
theorem median_and_significance
  (sorted_combined := combined.sort (≤))
  (median_calculation : (sorted_combined[19] + sorted_combined[20]) / 2 = 23.4)
  (a b c d : ℕ) (h_table : a = 6 ∧ b = 14 ∧ c = 14 ∧ d = 6)
  (h_ksquare : (40 * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d)) = 6.4)
  (h_critical_value : 6.4 > 3.841) : 
  m = 23.4 ∧ (6.4 > 3.841) :=
begin
  sorry
end

end OzoneExperiment

end sample_mean_experimental_group_median_and_significance_l816_816926


namespace solve_x4_eq_16_l816_816616

theorem solve_x4_eq_16 (x : ℂ) : x^4 - 16 = 0 ↔ x = 2 ∨ x = -2 ∨ x = 2 * complex.I ∨ x = -2 * complex.I :=
by
  sorry

end solve_x4_eq_16_l816_816616


namespace reasoning_is_deductive_l816_816454

-- Definitions based on the conditions
structure Metal (α : Type) :=
  (conductsElectricity : α → Prop)
  (isMetal : α → Prop)

-- Specific instance related to iron
def iron : Type := unit

-- Hypotheses based on the problem's conditions
axiom H1 : ∀ x : iron, Metal iron
axiom H2 : Metal iron

-- Statement we need to prove
theorem reasoning_is_deductive : 
  (All_metals_conduct_electricity : ∀ x, (Metal iron).conductsElectricity x) →
  (iron_is_a_metal : (Metal iron).isMetal ()) →
  (reasoning : true) :=
by 
  sorry

end reasoning_is_deductive_l816_816454


namespace distinct_sequences_count_l816_816196

theorem distinct_sequences_count : 
  let letters := ['T', 'R', 'I', 'A', 'N', 'G', 'L', 'E']
  let startSeq := ['T', 'R']
  let endSeq := ['G', 'L']
  (∃ middle : List Char, (middle ⊆ (letters \ (startSeq ++ endSeq)) ∧ middle.length = 1)) →
  (startSeq ++ ['A'] ++ endSeq = ['T', 'R', 'A', 'G', 'L']) ∨
  (startSeq ++ ['I'] ++ endSeq = ['T', 'R', 'I', 'G', 'L']) ∨
  (startSeq ++ ['N'] ++ endSeq = ['T', 'R', 'N', 'G', 'L']) ∨
  (startSeq ++ ['E'] ++ endSeq = ['T', 'R', 'E', 'G', 'L'])

end distinct_sequences_count_l816_816196


namespace spending_at_school_store_l816_816793

theorem spending_at_school_store (notebook ruler pencil : ℕ) (n_pencils : ℕ) 
  (h_notebook : notebook = 35)
  (h_ruler : ruler = 18)
  (h_pencil : pencil = 7)
  (h_n_pencils : n_pencils = 3) :
  notebook + ruler + (pencil * n_pencils) = 74 :=
by
  rw [h_notebook, h_ruler, h_pencil, h_n_pencils]
  exact calc
    35 + 18 + (7 * 3) = 35 + 18 + 21 : by simp
                 ... = 74 : by simp

end spending_at_school_store_l816_816793


namespace calc_fraction_exp_l816_816512

theorem calc_fraction_exp : 
  (\left(\frac{3}{5}\right)^{10} * \(\left(\frac{2}{3}\right)^{-4}\)) = \(\frac{4782969}{156250000}\) :=
by
  sorry

end calc_fraction_exp_l816_816512


namespace polynomial_has_at_most_one_real_root_l816_816808

open Polynomial

noncomputable def P (n m : ℕ) : Polynomial ℝ :=
  ∑ i in Finset.range (m + 1), Polynomial.C (Nat.choose (n + i) n) * Polynomial.X ^ i

theorem polynomial_has_at_most_one_real_root (n m : ℕ) : 
  (P n m).real_roots.length ≤ 1 :=
sorry

end polynomial_has_at_most_one_real_root_l816_816808


namespace roses_cut_l816_816410

def r_before := 13
def r_after := 14

theorem roses_cut : r_after - r_before = 1 := by
  sorry

end roses_cut_l816_816410


namespace find_interest_rate_l816_816012

theorem find_interest_rate (P A : ℝ) (t : ℝ) (n : ℕ) (hP : P = 12000) (hA : A = 15200) (ht : t = 7) (hn : n = 1) : 
  let r := (A / P) ^ (1 / (t * n)) - 1 in
  r * 100 ≈ 3.32 :=
by 
  sorry

end find_interest_rate_l816_816012


namespace cos_315_eq_l816_816560

theorem cos_315_eq :
  let Q := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))
  in Q = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) →
     Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  intro hQ
  sorry

end cos_315_eq_l816_816560


namespace limit_x_to_infty_frac_x_minus_sin_x_over_x_plus_sin_x_l816_816116

theorem limit_x_to_infty_frac_x_minus_sin_x_over_x_plus_sin_x :
  tendsto (λ x : ℝ, (x - sin x) / (x + sin x)) at_top (𝓝 1) :=
sorry

end limit_x_to_infty_frac_x_minus_sin_x_over_x_plus_sin_x_l816_816116


namespace find_x_y_l816_816136

theorem find_x_y (x y : ℕ) (hx : x! = 24)
  (hy1 : (y+2) * (y+1) = 3 * (y-2) * (y-3))
  (hy2 : C y x / C (y+2) x = 1 / 3)
  (hy3 : A y x / C y x = 24) :
  x = 4 ∧ y = 8 :=
by sorry

end find_x_y_l816_816136


namespace hyperbola_eccentricity_l816_816672

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∃ y line_slope, ∀ x, y = (line_slope * x) ∧ (line_slope = Real.sqrt 3) ∧ 
        (x, y) ∈ { p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1 }) :

  b = Real.sqrt 3 * a → 
  let c := Real.sqrt (a^2 + b^2) in
  let e := c / a in
  e = 2 := 
sorry

end hyperbola_eccentricity_l816_816672


namespace math_problem_l816_816666

-- Definitions
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)
def point_P (x y : ℝ) : Prop := (x = 2 ∧ y = sqrt 2)
def eccentricity (c a : ℝ) : Prop := (c / a = sqrt 2 / 2)
def line_l (x : ℝ) : Prop := (x = 4)
def k1_k2_k3_constant (a b : ℝ) : Prop := (Σ (k1 k2 k3 : ℝ), k1 + k2 - 2 * k3 = 0)

-- Theorem
theorem math_problem 
  (a b c : ℝ)
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : point_P 2 (sqrt 2))
  (h4 : eccentricity c a = sqrt 2 / 2) 
  (h5 : line_l 4)
  : ∃ (a b : ℝ), ellipse 8 4 2 (sqrt 2) ∧ k1_k2_k3_constant 8 4 := 
sorry

end math_problem_l816_816666


namespace exists_vector_from_origin_to_line_parallel_l816_816578

def param_line (t : ℝ) : ℝ × ℝ :=
  (5 * t + 3, 2 * t + 3)

theorem exists_vector_from_origin_to_line_parallel (a b k : ℝ)
  (h : ∃ t : ℝ, (a, b) = param_line t ∧ k * (5, 2) = (a, b)) :
  (a, b) = (-1.5, -0.6) :=
by
  sorry

end exists_vector_from_origin_to_line_parallel_l816_816578


namespace contest_order_l816_816967

theorem contest_order (A B C D E : ℕ) :
  let ABCDE := [A, B, C, D, E]
  let DAECB := [D, A, E, C, B]
  let finished_order := [E, D, A, C, B]
  (∀ i, ABCDE[i] ≠ finished_order[i]) → -- No contestant finished in the position predicted by ABCDE
  ∀ i, ABCDE[i] i.succ ≠ finished_order[i] finished_order i.succ mod 5 →
  (∀ {i j}, i ≠ j → finished_order[i] ≠ DAECB[i] → finished_order[j] = DAECB[j]) ∧ -- Exactly two contestants finished in the places predicted in DAECB
  (∀ i j, (abs i j ≠ 1 ∨ abs (finished_order i) (finished_order j) ≠ 1) → -- Two disjoint pairs are not consecutive in actual order
  ∀ i j, (finished_order i j = DAECB i j ∧ finished_order.mod 5 frozenset [i, j])) :=
sorry

end contest_order_l816_816967


namespace correct_equations_and_area_l816_816996

noncomputable def parabola_C1_equation (x y : ℝ) : Prop := x^2 = 4 * y

def focus_C2 := (0 : ℝ, 1 : ℝ)

def ellipse_C2_equation (x y : ℝ) : Prop := (y^2 / 4) + (x^2 / 3) = 1

def circle_C3_equation (x y : ℝ) : Prop := x^2 + y^2 = 1

def area_constraint (S : ℝ) : Prop := (3 / 2) ≤ S ∧ S < (2 * Real.sqrt 6) / 3

theorem correct_equations_and_area :
    (∀ x y : ℝ, parabola_C1_equation x y) →
    (∀ x y : ℝ, ellipse_C2_equation x y) →
    (∀ x y : ℝ, circle_C3_equation x y) →
    ∀ S : ℝ, area_constraint S :=
by
    intros
    sorry

end correct_equations_and_area_l816_816996


namespace construct_triangle_l816_816942

variables {R : Type*} [Field R]

-- Define points A = (x1, y1) and B = (x3, y3)
structure Point :=
(x : R) 
(y : R)

-- Define the reflection of point A over a given line to obtain point A'
def reflect_over_line (A : Point) (a b c : R) : Point :=
let x' := A.x - (2 * a * (a * A.x + b * A.y + c)) / (a^2 + b^2) in
let y' := A.y - (2 * b * (a * A.x + b * A.y + c)) / (a^2 + b^2) in
⟨x', y'⟩

-- Define the line through two points
def line_through_points (P Q : Point) : R × R × R :=
let a := Q.y - P.y in
let b := P.x - Q.x in
let c := P.y * Q.x - P.x * Q.y in
(a, b, c)

-- Define the intersection of two lines
def line_intersection (l1 l2 : R × R × R) : Point :=
let ⟨a1, b1, c1⟩ := l1 in
let ⟨a2, b2, c2⟩ := l2 in
let determinant := a1 * b2 - a2 * b1 in
let x := (b1 * c2 - b2 * c1) / determinant in
let y := (a2 * c1 - a1 * c2) / determinant in
⟨x, y⟩

-- Define the conditions and main statement that proves question == answer given conditions
theorem construct_triangle
  (A B : Point)
  (a b c : R) :
  let A' := reflect_over_line A a b c in
  let line_AB' := line_through_points A' B in
  let bisector_line := (a, b, c) in
  let C := line_intersection line_AB' bisector_line in
  true :=
begin
  sorry
end

end construct_triangle_l816_816942


namespace factorize_m_l816_816863

theorem factorize_m (m : ℝ) : m^2 - 4 * m - 5 = (m + 1) * (m - 5) := 
sorry

end factorize_m_l816_816863


namespace measure_angle_A_l816_816739

-- Define the problem conditions
variable (A B C D : Type) [Triangle A B C]
variable (is_equilateral : AB = AC ∧ AC = BC)
variable (CD_bisects_ACB : Angle_bisector C D B)
variable (CD_eq_CB : CD = CB)

-- Define the proof statement
theorem measure_angle_A (h₁ : is_equilateral) (h₂ : CD_bisects_ACB) (h₃ : CD_eq_CB) : 
  ∠A = 60 := 
by
  sorry

end measure_angle_A_l816_816739


namespace trig_identity_l816_816281

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end trig_identity_l816_816281


namespace incorrect_ac_bc_impl_a_b_l816_816879

theorem incorrect_ac_bc_impl_a_b : ∀ (a b c : ℝ), (ac = bc → a = b) ↔ c ≠ 0 :=
by sorry

end incorrect_ac_bc_impl_a_b_l816_816879


namespace rate_of_current_l816_816470

variable (c : ℝ)

-- Define the given conditions
def speed_still_water : ℝ := 4.5
def time_ratio : ℝ := 2

-- Define the effective speeds
def speed_downstream : ℝ := speed_still_water + c
def speed_upstream : ℝ := speed_still_water - c

-- Define the condition that it takes twice as long to row upstream as downstream
def rowing_equation : Prop := 1 / speed_upstream = 2 * (1 / speed_downstream)

-- The Lean theorem stating the problem we need to prove
theorem rate_of_current (h : rowing_equation) : c = 1.5 := by
  sorry

end rate_of_current_l816_816470


namespace probability_first_four_hearts_and_fifth_king_l816_816922

theorem probability_first_four_hearts_and_fifth_king :
  let total_cards := 52
  let hearts := 13
  let kings := 4
  let prob_first_heart := (hearts : ℚ) / total_cards
  let prob_second_heart := (hearts - 1 : ℚ) / (total_cards - 1)
  let prob_third_heart := (hearts - 2 : ℚ) / (total_cards - 2)
  let prob_fourth_heart := (hearts - 3 : ℚ) / (total_cards - 3)
  let prob_fifth_king := (kings : ℚ) / (total_cards - 4)
  prob_first_heart * prob_second_heart * prob_third_heart * prob_fourth_heart * prob_fifth_king = 286 / 124900 :=
by
  -- Definitions
  let total_cards := 52
  let hearts := 13
  let kings := 4
  
  -- Probabilities
  let prob_first_heart := (hearts : ℚ) / total_cards
  let prob_second_heart := (hearts - 1 : ℚ) / (total_cards - 1)
  let prob_third_heart := (hearts - 2 : ℚ) / (total_cards - 2)
  let prob_fourth_heart := (hearts - 3 : ℚ) / (total_cards - 3)
  let prob_fifth_king := (kings : ℚ) / (total_cards - 4)
  
  -- Equality
  have h : prob_first_heart * prob_second_heart * prob_third_heart * prob_fourth_heart * prob_fifth_king = 
    (13 / 52) * (12 / 51) * (11 / 50) * (10 / 49) * (1 / 12),
  by sorry
  rw h,
  calc (13 / 52) * (12 / 51) * (11 / 50) * (10 / 49) * (1 / 12) = 286 / 124900 : sorry -- Skip actual multiplication steps

end probability_first_four_hearts_and_fifth_king_l816_816922


namespace find_value_l816_816989

theorem find_value (x y : ℝ) (h : x - 2 * y = 1) : 3 - 4 * y + 2 * x = 5 := sorry

end find_value_l816_816989


namespace matrix_multiplication_correct_l816_816515

def matrixA : Matrix (Fin 3) (Fin 3) ℤ := 
  ![[1, 2, 3], [4, 5, 6], [7, 8, 9]]

def matrixB : Matrix (Fin 3) (Fin 3) ℤ := 
  ![[1, 0, 1], [1, 1, 0], [0, 1, 1]]

def matrixResult : Matrix (Fin 3) (Fin 3) ℤ := 
  ![[3, 5, 4], [9, 11, 10], [15, 17, 16]]

theorem matrix_multiplication_correct : matrixA ⬝ matrixB = matrixResult :=
by
  -- The proof steps would go here
  sorry

end matrix_multiplication_correct_l816_816515


namespace zoe_bought_bottles_l816_816025

theorem zoe_bought_bottles
  (initial_bottles : ℕ)
  (drank_bottles : ℕ)
  (current_bottles : ℕ)
  (initial_bottles_eq : initial_bottles = 42)
  (drank_bottles_eq : drank_bottles = 25)
  (current_bottles_eq : current_bottles = 47) :
  ∃ bought_bottles : ℕ, bought_bottles = 30 :=
by
  sorry

end zoe_bought_bottles_l816_816025


namespace solve_x4_eq_16_l816_816614

theorem solve_x4_eq_16 (x : ℂ) : x^4 - 16 = 0 ↔ x = 2 ∨ x = -2 ∨ x = 2 * complex.I ∨ x = -2 * complex.I :=
by
  sorry

end solve_x4_eq_16_l816_816614


namespace find_triplets_l816_816622

theorem find_triplets (x y z : ℕ) (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) 
  (h₁ : x ≤ y ∧ y ≤ z) (h₂ : 1 / x + 1 / y + 1 / z = 1) : 
  (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 3 ∧ y = 3 ∧ z = 3) :=
sorry

end find_triplets_l816_816622


namespace trig_identity_l816_816280

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end trig_identity_l816_816280


namespace solve_x4_minus_16_eq_0_l816_816610

open Complex  -- Open the complex number notation

theorem solve_x4_minus_16_eq_0 :
  {x : ℂ | x^4 = 16} = {2, -2, 2 * Complex.I, -2 * Complex.I} :=
by sorry

end solve_x4_minus_16_eq_0_l816_816610


namespace units_digit_of_sum_of_sequence_l816_816428

theorem units_digit_of_sum_of_sequence : 
  (∑ n in (Finset.range 12).map (λ n, n + 1), (nat.factorial n + n) % 10) % 10 = 1 := 
by sorry

end units_digit_of_sum_of_sequence_l816_816428


namespace cos_315_eq_sqrt2_div_2_l816_816566

theorem cos_315_eq_sqrt2_div_2 : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
sorry

end cos_315_eq_sqrt2_div_2_l816_816566


namespace max_towns_condition_l816_816008

structure Town := 
  (name : String)

inductive LinkType
  | Air
  | Bus
  | Train

structure Link := 
  (t1 t2 : Town)
  (link_type : LinkType)

def has_link (links : List Link) (t1 t2 : Town) : Bool :=
  links.any (λ link => (link.t1 = t1 ∧ link.t2 = t2) ∨ (link.t1 = t2 ∧ link.t2 = t1))

def has_link_type (links : List Link) (t1 t2 : Town) (lt : LinkType) : Bool :=
  links.any (λ link => ((link.t1 = t1 ∧ link.t2 = t2) ∨ (link.t1 = t2 ∧ link.t2 = t1)) ∧ link.link_type = lt)

def valid_links (links : List Link) : Prop :=
  ∃ t1 t2, has_link_type links t1 t2 LinkType.Air ∧
  ∃ t3 t4, has_link_type links t3 t4 LinkType.Bus ∧
  ∃ t5 t6, has_link_type links t5 t6 LinkType.Train ∧
  ∀ t, ¬(∃ t1 t2, has_link_type links t t1 LinkType.Air ∧ 
  has_link_type links t t2 LinkType.Bus ∧ 
  has_link_type links t t2 LinkType.Train) ∧
  ∀ t1 t2 t3, has_link_type links t1 t2 LinkType.Air ∧
  has_link_type links t1 t3 LinkType.Air ∧
  has_link_type links t2 t3 LinkType.Air → false ∧
  ∀ t1 t2 t3, has_link_type links t1 t2 LinkType.Bus ∧
  has_link_type links t1 t3 LinkType.Bus ∧
  has_link_type links t2 t3 LinkType.Bus → false ∧
  ∀ t1 t2 t3, has_link_type links t1 t2 LinkType.Train ∧
  has_link_type links t1 t3 LinkType.Train ∧
  has_link_type links t2 t3 LinkType.Train → false

def max_towns : Nat := 4

theorem max_towns_condition (towns : List Town) (links : List Link) :
  valid_links links → 
  towns.length ≤ max_towns :=
sorry

end max_towns_condition_l816_816008


namespace arctan_sum_l816_816735

theorem arctan_sum (a b : ℝ) : 
  Real.arctan (a / (a + 2 * b)) + Real.arctan (b / (2 * a + b)) = Real.arctan (1 / 2) :=
by {
  sorry
}

end arctan_sum_l816_816735


namespace arithmetic_sufficient_arithmetic_not_necessary_l816_816646

variable (a : ℕ → ℝ)

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Proof for sufficient condition
theorem arithmetic_sufficient (h: is_arithmetic_sequence a) : a 1 + a 3 = 2 * a 2 :=
  by 
    obtain ⟨d, hd⟩ := h
    have ha1 : a 2 = a 1 + d := hd 1
    have ha2 : a 3 = a 2 + d := hd 2
    rw [ha1, ha2]
    ring

-- Proof for non-necessary condition
theorem arithmetic_not_necessary (h: a 1 + a 3 = 2 * a 2) : ¬is_arithmetic_sequence a :=
  sorry

end arithmetic_sufficient_arithmetic_not_necessary_l816_816646


namespace find_k_l816_816202

theorem find_k (k : ℤ) (h1 : |k| = 1) (h2 : k - 1 ≠ 0) : k = -1 :=
by
  sorry

end find_k_l816_816202


namespace cos_315_eq_sqrt2_div_2_l816_816565

theorem cos_315_eq_sqrt2_div_2 : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
sorry

end cos_315_eq_sqrt2_div_2_l816_816565


namespace numerator_divisible_by_prime_l816_816811

theorem numerator_divisible_by_prime (p : ℕ) (h_prime : Nat.Prime p) (h_gt_2 : p > 2) :
  p ∣ Nat.num (1 + (Finset.range (p - 1)).sum (λ k, 1 / (k + 1))) := sorry

end numerator_divisible_by_prime_l816_816811


namespace range_for_frequency_l816_816186

-- Define the sample data
def sample_data : List ℕ := [10, 8, 6, 10, 13, 8, 10, 12, 11, 7, 8, 9, 11, 9, 12, 9, 10, 11, 12, 11]

-- Define the total number of data points
def total_data_points : ℕ := 20

-- Define function to calculate frequency count
def frequency_count (freq : ℚ) : ℕ := (total_data_points : ℚ) * freq

-- Define the range to check
def range_11_5_to_13_5 : Set ℕ := { x | 11 < x ∧ x < 14 }

-- Define the range to check the frequency of
def range_frequency_check (data : List ℕ) (freq_count : ℕ) (range : Set ℕ) : Prop :=
  (data.filter (λ x, x ∈ range)).length = freq_count

-- Prove that the frequency of 0.2 corresponds to the range from 11.5 to 13.5
theorem range_for_frequency : range_frequency_check sample_data (frequency_count 0.2) range_11_5_to_13_5 :=
  by
    -- This theorem needs to be proven
    sorry

end range_for_frequency_l816_816186


namespace Triamoeba_Count_After_One_Week_l816_816073

def TriamoebaCount (n : ℕ) : ℕ :=
  3 ^ n

theorem Triamoeba_Count_After_One_Week : TriamoebaCount 7 = 2187 :=
by
  -- This is the statement to be proved
  sorry

end Triamoeba_Count_After_One_Week_l816_816073


namespace compare_a_b_c_l816_816984

noncomputable def a : ℝ := Real.log (11 / 10)
def b : ℝ := 1 / 10
def c : ℝ := 2 / 21

theorem compare_a_b_c : b > a ∧ a > c := by
  sorry

end compare_a_b_c_l816_816984


namespace total_wheels_in_garage_l816_816418

theorem total_wheels_in_garage :
  let cars := 2,
      wheels_per_car := 4,
      riding_lawnmower := 1,
      wheels_per_lawnmower := 4,
      bicycles := 3,
      wheels_per_bicycle := 2,
      tricycle := 1,
      wheels_per_tricycle := 3,
      unicycle := 1,
      wheels_per_unicycle := 1 in
  cars * wheels_per_car + riding_lawnmower * wheels_per_lawnmower + bicycles * wheels_per_bicycle + tricycle * wheels_per_tricycle + unicycle * wheels_per_unicycle = 22 := by
  sorry

end total_wheels_in_garage_l816_816418


namespace ducks_and_chickens_l816_816865

theorem ducks_and_chickens : 
  (∃ ducks chickens : ℕ, ducks = 7 ∧ chickens = 6 ∧ ducks + chickens = 13) :=
by
  sorry

end ducks_and_chickens_l816_816865


namespace simplify_expression_l816_816890

theorem simplify_expression :
  (sqrt 3 + 5) * (5 - sqrt 3) - (sqrt 8 + 2 * sqrt (1 / 2)) / sqrt 2 + sqrt ((sqrt 5 - 3) ^ 2) = 22 - sqrt 5 :=
by sorry

end simplify_expression_l816_816890


namespace area_of_square_ABCD_l816_816082

noncomputable def area_of_square (x : ℝ) : ℝ := x * x

theorem area_of_square_ABCD :
  ∀ (x : ℝ), (x = √5) → (∃ (E : ℝ), E = x - 1 ∧ (x - 1 ≥ 0) ∧ (triangle.orthogonal (sqrt 5) x x)) →
  area_of_square x = 5 :=
by
  intros x hx hE,
  sorry

end area_of_square_ABCD_l816_816082


namespace cos_angle_of_point_l816_816147

theorem cos_angle_of_point (P : ℝ × ℝ) (hP : P = (-4, 3))
    (a : ℝ) (ha : angle_of_point P = a) : 
    Real.cos a = -4 / 5 := 
by
  -- proof goes here
  sorry

end cos_angle_of_point_l816_816147


namespace trapezoid_shorter_diagonal_is_25_l816_816001

noncomputable def trapezoid_diagonal
  (AB CD AD BC : ℝ)
  (h : 0 < AB ∧ 0 < CD ∧ 0 < AD ∧ 0 < BC)
  (parallel : ∃ l, AB = CD + l) -- "parallel sides" condition placeholder.
  (acute_A : ∀ angle_A, angle_A < pi/2)
  (acute_B : ∀ angle_B, angle_B < pi/2) 
  : ℝ := sorry  -- length of the shorter diagonal to be defined

theorem trapezoid_shorter_diagonal_is_25 :
  trapezoid_diagonal 33 21 10 14 (by sorry){by sorry (by sorry) (by sorry)} = 25 := sorry

end trapezoid_shorter_diagonal_is_25_l816_816001


namespace true_proposition_is_C_l816_816504

open Set

variables {α : Type*} [TopologicalSpace α] (m n : Set α) (l : Set (Set α))

-- Definitions of conditions for each proposition
def prop_A : Prop :=
  (m ⊆ l) ∧ (¬(n ⊆ l)) ∧ (m ∩ n = ∅) ∧ (Disjoint m n) → (∀ l', n ∩ l = ∅ → n ⊆ l')

def prop_B : Prop :=
  (m ⊆ l) ∧ (∃ p, p ∈ n ∩ l) → m ∩ n = ∅

def prop_C : Prop :=
  (m ⊆ l) ∧ (∀ l', n ⊆ l') ∧ (∃ k, k ∈ m ∩ n) → (m ∩ n ≠ ∅)

def prop_D : Prop :=
  (∀ l', m ⊆ l') ∧ (∀ l', n ⊆ l') ∧ (∃ k, k ∈ m ∩ n) → (m ∩ n ≠ ∅)

-- The theorem stating the correct proposition
theorem true_proposition_is_C : prop_C m n l :=
sorry

end true_proposition_is_C_l816_816504


namespace cos_315_is_sqrt2_div_2_l816_816516

noncomputable def cos_315_eq : Prop := 
  real.cos (315 * real.pi / 180) = real.sqrt 2 / 2

theorem cos_315_is_sqrt2_div_2 : cos_315_eq := by 
  sorry

end cos_315_is_sqrt2_div_2_l816_816516


namespace sin_minus_cos_eq_l816_816230

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end sin_minus_cos_eq_l816_816230


namespace sin_4phi_l816_816289

theorem sin_4phi :
  (∃ φ : ℝ, exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 5) →
  ∃ φ : ℝ, Real.sin (4 * φ) = (12 * Real.sqrt 8) / 625 :=
by
  intros h
  cases h with φ hφ
  use φ
  sorry

end sin_4phi_l816_816289


namespace water_polo_team_selection_l816_816372

theorem water_polo_team_selection :
  let total_players := 20
  let team_size := 9
  let goalies := 2
  let remaining_players := total_players - goalies
  let combination (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  combination total_players goalies * combination remaining_players (team_size - goalies) = 6046560 :=
by
  -- Definitions and calculations to be filled here.
  sorry

end water_polo_team_selection_l816_816372


namespace garden_ratio_l816_816915

-- Define the given conditions
def garden_length : ℕ := 100
def garden_perimeter : ℕ := 300

-- Problem statement: Prove the ratio of the length to the width is 2:1
theorem garden_ratio : 
  ∃ (W L : ℕ), 
    L = garden_length ∧ 
    2 * L + 2 * W = garden_perimeter ∧ 
    L / W = 2 :=
by 
  sorry

end garden_ratio_l816_816915


namespace cos_315_proof_l816_816552

noncomputable def cos_315_eq : Prop :=
  let theta := 45 in
  let angle := 360 - theta in
  ∃ c : real, (cos 45) = (sqrt 2 / 2) ∧ (cos (360 - theta) = cos theta) ∧ 315 = angle ∧ (forall a : real, cos a = cos (360 - a)) ∧ (315 ∈ (4th quadrant)) ∧ (cos 315 = cos 45)

theorem cos_315_proof : (cos 315 = sqrt 2 / 2) :=
begin
  -- We start by defining basic facts
  -- 315 = 360 - 45
  -- Cosine is positive in the fourth quadrant, known value \cos 45 = sqrt 2 / 2
  -- Cosine property \cos (360 - \theta) = \cos \theta
  let theta := 45,
  let angle := 360 - theta,
  have H1 : cos theta = sqrt 2 / 2,
  {
    -- known fact
    sorry,
  },
  have H2 : cos (360 - theta) = cos theta,
  {
    -- cosine property
    sorry,
  },
  have H3 : angle = 315,
  {
    -- subtraction identity
    sorry,
  },
  rw H3,
  rw H2,
  rw H1,
  refl,
end

end cos_315_proof_l816_816552


namespace solution_set_of_inequality_l816_816358

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≥ 0 then x^2 - 4*x + 6 else x + 6

theorem solution_set_of_inequality :
  {x : ℝ | f x > f 1} = {x | -3 < x ∧ x < 1} ∪ {x | 3 < x} :=
by
  sorry

end solution_set_of_inequality_l816_816358


namespace solve_for_n_l816_816829

theorem solve_for_n (n : ℤ) : 9^n * 9^n * 9^n * 9^n = 81^4 ↔ n = 2 :=
by
  sorry

end solve_for_n_l816_816829


namespace det_example_1_simplified_form_det_at_4_l816_816006

-- Definition for second-order determinant
def second_order_determinant (a b c d : ℤ) : ℤ :=
  a * d - b * c

-- Part (1)
theorem det_example_1 :
  second_order_determinant 3 (-2) 4 (-3) = -1 :=
by
  sorry

-- Part (2) simplified determinant
def simplified_det (x : ℤ) : ℤ :=
  second_order_determinant (2 * x - 3) (x + 2) 2 4

-- Proving simplified determinant form
theorem simplified_form :
  ∀ x : ℤ, simplified_det x = 6 * x - 16 :=
by
  sorry

-- Proving specific case when x = 4
theorem det_at_4 :
  simplified_det 4 = 8 :=
by 
  sorry

end det_example_1_simplified_form_det_at_4_l816_816006


namespace box_length_is_10_l816_816908

theorem box_length_is_10
  (width height vol_cube num_cubes : ℕ)
  (h₀ : width = 13)
  (h₁ : height = 5)
  (h₂ : vol_cube = 5)
  (h₃ : num_cubes = 130) :
  (num_cubes * vol_cube) / (width * height) = 10 :=
by
  -- Proof steps will be filled here.
  sorry

end box_length_is_10_l816_816908


namespace slope_range_l816_816940

theorem slope_range (a b : ℝ) (h₁ : a ≠ -2) (h₂ : a ≠ 2) 
  (h₃ : a^2 / 4 + b^2 / 3 = 1) (h₄ : -2 ≤ b / (a - 2) ∧ b / (a - 2) ≤ -1) :
  (3 / 8 ≤ b / (a + 2) ∧ b / (a + 2) ≤ 3 / 4) :=
sorry

end slope_range_l816_816940


namespace positive_difference_of_squares_l816_816859

theorem positive_difference_of_squares 
  (a b : ℕ)
  (h1 : a + b = 60)
  (h2 : a - b = 16) : a^2 - b^2 = 960 :=
by
  sorry

end positive_difference_of_squares_l816_816859


namespace liu_xiang_hurdles_l816_816494

theorem liu_xiang_hurdles :
  let total_distance := 110
  let first_hurdle_distance := 13.72
  let last_hurdle_distance := 14.02
  let best_time_first_segment := 2.5
  let best_time_last_segment := 1.4
  let hurdle_cycle_time := 0.96
  let num_hurdles := 10
  (total_distance - first_hurdle_distance - last_hurdle_distance) / num_hurdles = 8.28 ∧
  best_time_first_segment + num_hurdles * hurdle_cycle_time + best_time_last_segment  = 12.1 :=
by
  sorry

end liu_xiang_hurdles_l816_816494


namespace smallest_lambda_l816_816998

theorem smallest_lambda (n : ℕ) (h₁ : ∀ i, 1 ≤ i → i ≤ n → Real) (h₂ : (∀ i, 1 ≤ i → i ≤ n → 0 < h₁ i) 
          → (∀ i, 1 ≤ i → i ≤ n → h₁ i < (Real.pi / 2)) → (∏ i in Finset.range n, Real.tan (h₁ (i + 1)) = 2^(n/2)) 
          → (∑ i in Finset.range n, Real.cos (h₁ (i + 1)) ≤ n - 1)) :=
sorry

end smallest_lambda_l816_816998


namespace vector_dot_product_sum_l816_816637

variables (A B C : ℝ)
noncomputable def is_right_triangle (C : ℝ) (right_angle : C = π / 2) :=
  (A = π / 6) ∧ (AB = 2)

theorem vector_dot_product_sum (h : is_right_triangle C) :
  let AB := 2,
      AC := sqrt 3,
      BC := 1,
      cos_pi_6 := real.cos (π / 6),
      cos_pi_3 := real.cos (π / 3)
  in (AB * BC * cos_pi_3 + BC * AC * cos_pi_6 + AC * AB * cos_pi_6) = -4 := 
by
  sorry

end vector_dot_product_sum_l816_816637


namespace Amith_current_age_l816_816037

variable (A D : ℕ)

theorem Amith_current_age
  (h1 : A - 5 = 3 * (D - 5))
  (h2 : A + 10 = 2 * (D + 10)) :
  A = 50 := by
  sorry

end Amith_current_age_l816_816037


namespace shark_feed_l816_816508

theorem shark_feed (S : ℝ) (h1 : S + S/2 + 5 * S = 26) : S = 4 := 
by sorry

end shark_feed_l816_816508


namespace cos_315_deg_l816_816542

noncomputable def cos_315 : ℝ :=
  Real.cos (315 * Real.pi / 180)

theorem cos_315_deg : cos_315 = Real.sqrt 2 / 2 :=
by
  sorry

end cos_315_deg_l816_816542


namespace hannah_banana_flour_l816_816195

-- Define the conditions
def flour_needed (cups_of_mush : ℕ) : ℕ :=
  3 * cups_of_mush

def banana_mush (bananas : ℕ) : ℕ :=
  bananas / 4

-- Given problem statement and condition
theorem hannah_banana_flour :
  banana_mush 20 = 5 ∧ flour_needed 5 = 15 :=
by 
  split;
  sorry

end hannah_banana_flour_l816_816195


namespace area_of_parallelogram_l816_816961

theorem area_of_parallelogram (base height : ℝ) (h_base : base = 12) (h_height : height = 8) :
  base * height = 96 := by
  sorry

end area_of_parallelogram_l816_816961


namespace new_pattern_area_ratio_l816_816041

theorem new_pattern_area_ratio (r : ℝ) (h₁ : r = 3) (h₂ : r * 2 * ℝ.pi = 12) (h₃ : 6 = 6) :
  let area_circle := ℝ.pi * r ^ 2
  in let area_rectangle := 12 * 6
  in area_circle / area_circle = 1 := 
by 
  sorry

end new_pattern_area_ratio_l816_816041


namespace find_q_l816_816678

variable (p q : ℝ)
variable (h1 : 1 < p)
variable (h2 : p < q)
variable (h3 : 1 / p + 1 / q = 1)
variable (h4 : p * q = 8)

theorem find_q : q = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end find_q_l816_816678


namespace exists_two_conditional_pairs_count_of_conditional_pairs_is_odd_l816_816798

-- Define the domain of the problem
def nonRepeatingNineDigitNumbers := {a | digits a = [1,2,3,4,5,6,7,8,9] ∧ nineDigitNumber a}

-- Define a conditional pair predicate
def conditional_pair (a b : ℕ) : Prop :=
  nineDigitNumber a ∧ nineDigitNumber b ∧ noRepeatingDigits a ∧ noRepeatingDigits b ∧ (a + b = 987654321)

-- Define the first proof problem to show the existence of at least two conditional pairs
theorem exists_two_conditional_pairs : ∃ a b : ℕ, conditional_pair a b ∧ conditional_pair b a :=
sorry

-- Define the second proof problem to show that the number of conditional pairs is odd
theorem count_of_conditional_pairs_is_odd : 
  (∃ n : ℕ, n > 0 ∧ @Finset.card (ℕ × ℕ) (finset.conditionalPairs) = (2 * n) + 1) := 
sorry

end exists_two_conditional_pairs_count_of_conditional_pairs_is_odd_l816_816798


namespace solve_quadratic_l816_816831

theorem solve_quadratic :
  ∀ x : ℝ, (x^2 - 3 * x + 2 = 0) → (x = 1 ∨ x = 2) :=
by sorry

end solve_quadratic_l816_816831


namespace find_radius_of_inscribed_circle_l816_816753

-- Define the right triangle ABC with specified angler
structure right_triangle (A B C : Type) :=
(angle_BAC_eq_90 : ∃ (A B C : Type), ∠BAC = 90°)
(A_eq_8 : AB = 8)
(BC_eq_10 : BC = 10)

-- Define circle tangent conditions
def circle_tangent_to_hypotenuse_and_side (O P : Type) (C : right_triangle) :=
∃ P, circle(O, r) tangent_to_hypotenuse_at P ∧ tangent_to_side AB

-- Define radius of circle
def radius_of_circle (C : right_triangle) (O P : Type) :=
∃ r, circle_tangent_to_hypotenuse_and_side O P C → r = √11

-- Putting it all together in a theorem statement
theorem find_radius_of_inscribed_circle (A B C O P : Type) (T : right_triangle A B C) :
  radius_of_circle T O P :=
begin
  sorry,
end

end find_radius_of_inscribed_circle_l816_816753


namespace length_of_AE_l816_816581

noncomputable def quadrilateral_ABCD (AB CD AC AE : ℝ) (EisMidPoint : (10:ℝ) = (15:ℝ) * (AE / (15 - AE))) : Prop :=
  AE = 6

theorem length_of_AE (AE AB CD AC : ℝ) (hConditions : AB = 10 ∧ CD = 15 ∧ AC = 15 ∧ (∃ E, (10 : ℝ) = (15:ℝ) * (AE / (15 - AE)))) :
  quadrilateral_ABCD AB CD AC AE hConditions.2.2.2.2 :=
by
  intros
  sorry

end length_of_AE_l816_816581


namespace total_number_of_balls_is_twelve_l816_816318

noncomputable def num_total_balls (a : ℕ) : Prop :=
(3 : ℚ) / a = (25 : ℚ) / 100

theorem total_number_of_balls_is_twelve : num_total_balls 12 :=
by sorry

end total_number_of_balls_is_twelve_l816_816318


namespace solve_equation_l816_816382

noncomputable def equation (x : ℝ) : Prop :=
  2 / (x - 2) = (1 + x) / (x - 2) + 1

theorem solve_equation : ∀ (x : ℝ), equation x ∧ x ≠ 2 ↔ x = 3 / 2 :=
by
  intro x
  split
  sorry
  sorry

end solve_equation_l816_816382


namespace sin_minus_cos_l816_816234

noncomputable def theta_condition (θ : ℝ) : Prop := (0 < θ) ∧ (θ < π / 2) ∧ (Real.tan θ = 1 / 3)

theorem sin_minus_cos (θ : ℝ) (hθ : theta_condition θ) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_minus_cos_l816_816234


namespace months_to_survive_l816_816311

theorem months_to_survive (P_survive : ℝ) (initial_population : ℕ) (expected_survivors : ℝ) (n : ℕ)
  (h1 : P_survive = 5 / 6)
  (h2 : initial_population = 200)
  (h3 : expected_survivors = 115.74)
  (h4 : initial_population * (P_survive ^ n) = expected_survivors) :
  n = 3 :=
sorry

end months_to_survive_l816_816311


namespace minimum_value_triangle_l816_816334

theorem minimum_value_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c / 3) :
  (∃ k : ℝ, k > 0 ∧ (∀ (x : ℝ), (x = (a * Real.cos A + b * Real.cos B) / (a * Real.cos B)) -> k ≤ x)) :=
begin
  use Real.sqrt 2,
  split,
  { exact Real.sqrt 2_pos },
  {
    intros x hx,
    sorry
  }

end minimum_value_triangle_l816_816334


namespace total_amount_shared_l816_816464

theorem total_amount_shared (a b c d : ℝ) (h1 : a = (1/3) * (b + c + d)) 
    (h2 : b = (2/7) * (a + c + d)) (h3 : c = (4/9) * (a + b + d)) 
    (h4 : d = (5/11) * (a + b + c)) (h5 : a = b + 20) (h6 : c = d - 15) 
    (h7 : (a + b + c + d) % 10 = 0) : a + b + c + d = 1330 :=
by
  sorry

end total_amount_shared_l816_816464


namespace min_transport_cost_l816_816405

theorem min_transport_cost :
  let large_truck_capacity := 7
  let large_truck_cost := 600
  let small_truck_capacity := 4
  let small_truck_cost := 400
  let total_goods := 20
  ∃ (n_large n_small : ℕ),
    n_large * large_truck_capacity + n_small * small_truck_capacity ≥ total_goods ∧ 
    (n_large * large_truck_cost + n_small * small_truck_cost) = 1800 :=
sorry

end min_transport_cost_l816_816405


namespace evaluate_diff_l816_816166

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x then log x / log 10 else -x

theorem evaluate_diff : f(10) - f(-100) = -8 := by
  -- Prove the statement based on the conditions provided
  sorry

end evaluate_diff_l816_816166


namespace product_of_fractions_l816_816101

theorem product_of_fractions :
  (3/4) * (4/5) * (5/6) * (6/7) = 3/7 :=
by
  sorry

end product_of_fractions_l816_816101


namespace tangent_line_at_x0_l816_816976

noncomputable def curve : ℝ → ℝ := λ x, 2 * x^2 + 3

def x0 : ℝ := -1

theorem tangent_line_at_x0 :
  ∃ m b : ℝ, (λ x, m * x + b) = λ x, -4 * x + 1 ∧ 
            (∀ x near x0, curve x = (m * x + b) + (x - x0) * (curve.deriv x0)) :=
begin
  sorry
end

end tangent_line_at_x0_l816_816976


namespace product_of_distances_slopes_sum_zero_l816_816733

noncomputable def ellipse (x y : ℝ) : Prop :=
  (x^2) / 6 + (y^2) / 3 = 1

structure Point :=
  (x : ℝ)
  (y : ℝ)

def P : Point := { x := 2, y := 2 }

def l1 (t : ℝ) (α : ℝ) : Point :=
  { x := 2 + t * Real.cos α, y := 2 + t * Real.sin α }

def l2 (t : ℝ) (α : ℝ) : Point :=
  { x := 2 - t * Real.cos α, y := 2 - t * Real.sin α }

axiom l1_slope (α : ℝ) : ∃ t, l1 t α ∧ ellipse (l1 t α).x (l1 t α).y
axiom l2_slope (α : ℝ) : ∃ t, l2 t α ∧ ellipse (l2 t α).x (l2 t α).y

theorem product_of_distances (α : ℝ) : 
  (∃ t1 t2, l1 t1 α ∧ l1 t2 α ∧ ellipse (l1 t1 α).x (l1 t1 α).y ∧ ellipse (l1 t2 α).x (l1 t2 α).y ∧ 
  (P.x - (l1 t1 α).x)^2 + (P.y - (l1 t1 α).y)^2 = PA^2 ∧ (P.x - (l1 t2 α).x)^2 + (P.y - (l1 t2 α).y)^2 = PB^2) → 
  (∃ t3 t4, l2 t3 α ∧ l2 t4 α ∧ ellipse (l2 t3 α).x (l2 t3 α).y ∧ ellipse (l2 t4 α).x (l2 t4 α).y ∧ 
  (P.x - (l2 t3 α).x)^2 + (P.y - (l2 t3 α).y)^2 = PC^2 ∧ (P.x - (l2 t4 α).x)^2 + (P.y - (l2 t4 α).y)^2 = PD^2) →
  PA * PB = PC * PD :=
sorry

theorem slopes_sum_zero (α : ℝ) : 
  let k1 := (l1 (idet α)).y - 4)/((l1 (idet α)).x - 4) in
  let k2 := (l2 (idet α)).y - 4)/((l2 (idet α)).x - 4) in
  k1 + k2 = 0 :=
sorry

end product_of_distances_slopes_sum_zero_l816_816733


namespace intersection_A_B_complement_l816_816755

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x ≤ 1}
def B_complement : Set ℝ := U \ B

theorem intersection_A_B_complement : A ∩ B_complement = {x | x > 1} := 
by 
  sorry

end intersection_A_B_complement_l816_816755


namespace sin_minus_cos_l816_816267

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end sin_minus_cos_l816_816267


namespace cos_eq_condition_l816_816698

theorem cos_eq_condition (x y : ℝ) : 
  (cos x * cos y = cos (x + y)) → 
  (∃ k : ℤ, x = k * Real.pi ∨ y = k * Real.pi) := 
by 
  sorry

end cos_eq_condition_l816_816698


namespace cos_315_eq_sqrt2_div_2_l816_816569

theorem cos_315_eq_sqrt2_div_2 : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
sorry

end cos_315_eq_sqrt2_div_2_l816_816569


namespace odell_kershaw_meetings_l816_816799

def distance (speed : ℝ) (time : ℝ) := speed * time
def circumference (radius : ℝ) := 2 * real.pi * radius
def angular_speed (speed circumference : ℝ) := (speed / circumference) * 2 * real.pi
def time_to_meet (angular_speed_rel : ℝ) := 2 * real.pi / angular_speed_rel
def meetings (total_time time_to_meet : ℝ) := ⌊ total_time / time_to_meet ⌋

theorem odell_kershaw_meetings :
  let radius_odell := 55
  let speed_odell := 260
  let radius_kershaw := 65
  let speed_kershaw := 280
  let time := 45
  let circumference_odell := circumference radius_odell
  let circumference_kershaw := circumference radius_kershaw
  let angular_speed_odell := angular_speed speed_odell circumference_odell
  let angular_speed_kershaw := angular_speed speed_kershaw circumference_kershaw
  let angular_speed_rel := angular_speed_odell + angular_speed_kershaw
  let meet_time := time_to_meet angular_speed_rel
  let total_meetings := meetings time meet_time
  total_meetings = 64 :=
begin
  sorry
end

end odell_kershaw_meetings_l816_816799


namespace solve_problem_l816_816680

open Real

noncomputable def problem_statement : Prop :=
  ∃ (p q : ℝ), 1 < p ∧ p < q ∧ (1 / p + 1 / q = 1) ∧ (p * q = 8) ∧ (q = 4 + 2 * sqrt 2)
  
theorem solve_problem : problem_statement :=
sorry

end solve_problem_l816_816680


namespace cost_of_fencing_l816_816884

/-- Define given conditions: -/
def sides_ratio (length width : ℕ) : Prop := length = 3 * width / 2

def park_area : ℕ := 3750

def paise_to_rupees (paise : ℕ) : ℕ := paise / 100

/-- Prove that the cost of fencing the park is 150 rupees: -/
theorem cost_of_fencing 
  (length width : ℕ) 
  (h : sides_ratio length width) 
  (h_area : length * width = park_area) 
  (cost_per_meter_paise : ℕ := 60) : 
  (length + width) * 2 * (paise_to_rupees cost_per_meter_paise) = 150 :=
by sorry

end cost_of_fencing_l816_816884


namespace probability_three_diagonals_intersect_l816_816870

theorem probability_three_diagonals_intersect :
  let n := 10
  let total_diagonals := ((nat.choose n 2) - n)
  let three_diagonals_ways := (nat.choose total_diagonals 3)
  let single_point_intersection_ways := n * (nat.choose (n-1) 3)
  (single_point_intersection_ways : ℚ) / (three_diagonals_ways : ℚ) = 840 / 6545 :=
by
  sorry

end probability_three_diagonals_intersect_l816_816870


namespace V4_of_polynomial_at_minus4_l816_816874

noncomputable def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
coeffs.foldr (λ a acc, a + acc * x) 0

theorem V4_of_polynomial_at_minus4 :
  let coeffs := [3, 5, 6, 79, -8, 35, 12] in
  horner coeffs (-4) = 220 := 
by
  let coeffs := [3, 5, 6, 79, -8, 35, 12]
  have h : horner coeffs (-4) = 220 := sorry
  exact h

end V4_of_polynomial_at_minus4_l816_816874


namespace find_the_number_l816_816448

theorem find_the_number (x : ℝ) (h : 100 - x = x + 40) : x = 30 :=
sorry

end find_the_number_l816_816448


namespace value_calculation_l816_816292

-- Define the given number
def given_number : ℝ := 93.75

-- Define the percentages as ratios
def forty_percent : ℝ := 0.4
def sixteen_percent : ℝ := 0.16

-- Calculate the intermediate value for 40% of the given number
def intermediate_value := forty_percent * given_number

-- Final value calculation for 16% of the intermediate value
def final_value := sixteen_percent * intermediate_value

-- The theorem to prove
theorem value_calculation : final_value = 6 := by
  -- Expanding definitions to substitute and simplify
  unfold final_value intermediate_value forty_percent sixteen_percent given_number
  -- Proving the correctness by calculating
  sorry

end value_calculation_l816_816292


namespace correct_quotient_l816_816883

theorem correct_quotient : 
  ∀ (D : ℕ) (Q : ℕ),
  (∃ Q_student : ℕ, 12 * Q_student = D ∧ Q_student = 42) →
  21 * Q = D → 
  Q = 24 :=
by
  intros D Q h1 h2
  cases h1 with Q_student h1_conditions
  cases h1_conditions with h1_left h1_right
  rw h1_right at h1_left
  rw h1_left at h2
  have h : Q = (12 * 42) / 21 := by
    rw [← mul_assoc, mul_div_cancel_left]
    exact h2
    exact by norm_num
  rw nat.div_eq_of_eq_mul at h
  exact h.symm
  exact sorry

end correct_quotient_l816_816883


namespace value_of_m_l816_816297

-- Defining the quadratic equation condition
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + 3 * x + m^2 - 4

-- Defining the condition where the constant term in the quadratic equation is 0
def constant_term_zero (m : ℝ) : Prop := m^2 - 4 = 0

-- Stating the proof problem: given the conditions, prove that m = -2
theorem value_of_m (m : ℝ) (h1 : constant_term_zero m) (h2 : m ≠ 2) : m = -2 :=
by {
  sorry -- Proof to be developed
}

end value_of_m_l816_816297


namespace problem_l816_816327

theorem problem : (1 * (2 + 3) * 4 * 5) = 100 := by
  sorry

end problem_l816_816327


namespace glutinous_rice_flour_leftover_l816_816337

theorem glutinous_rice_flour_leftover 
  (flour_per_cake : ℝ) (total_flour : ℝ) (conversion_factor : ℝ) : 
  flour_per_cake = 6.64 →
  total_flour = 202.66 →
  conversion_factor = 1000 →
  let cakes_made := ⌊total_flour / flour_per_cake⌋ in
  let flour_used := cakes_made * flour_per_cake in
  let flour_left := total_flour - flour_used in
  flour_left * conversion_factor = 3460 := 
by
  intros hp ht hc
  rw [hp, ht, hc]
  let cakes_made := ⌊202.66 / 6.64⌋
  have hn : cakes_made = 30 := by norm_num
  rw hn
  let flour_used := 30 * 6.64
  have hr : flour_used = 199.2 := by norm_num
  rw hr
  let flour_left := 202.66 - 199.2
  have hl : flour_left = 3.46 := by norm_num
  rw hl
  have hf : 3.46 * 1000 = 3460 := by norm_num
  exact hf

end glutinous_rice_flour_leftover_l816_816337


namespace monday_time_longer_by_52_percent_l816_816103

variable (x : ℝ) (hₓ : x > 0)

def time_sunday : ℝ := 100 / x

def time_monday_first32 : ℝ := 32 / (2 * x)

def time_monday_remaining68 : ℝ := 68 / (x / 2)

def time_monday : ℝ := time_monday_first32 x + time_monday_remaining68 x

def percent_increase : ℝ := ((time_monday x - time_sunday x) / time_sunday x) * 100

theorem monday_time_longer_by_52_percent : percent_increase x hₓ = 52 := by
  sorry

end monday_time_longer_by_52_percent_l816_816103


namespace area_of_triangle_zeros_of_f_l816_816352

noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.exp x - x - 1

-- Part (1)
theorem area_of_triangle (f'0 : deriv f 0 = -1) (f0 : f 0 = -2) : 
  let tangent_line (x : ℝ) := -x - 2 in
  let intercepts := {
    x_intercept := (-2 : ℝ),
    y_intercept := (-2 : ℝ)
  } in
  let area := (1 / 2) * |intercepts.x_intercept| * |intercepts.y_intercept| in
  area = 2 := sorry

-- Part (2)
theorem zeros_of_f : 
  ∃ x1 x2, f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 ∧ x1 + x2 = 0 := sorry

end area_of_triangle_zeros_of_f_l816_816352


namespace smallest_integer_n_l816_816020

theorem smallest_integer_n (n : ℕ) (h : Nat.lcm 60 n / Nat.gcd 60 n = 75) : n = 500 :=
sorry

end smallest_integer_n_l816_816020


namespace sin_minus_cos_theta_l816_816250

theorem sin_minus_cos_theta (θ : ℝ) (h₀ : 0 < θ ∧ θ < (π / 2)) 
  (h₁ : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -√10 / 5 :=
by
  sorry

end sin_minus_cos_theta_l816_816250


namespace sin_minus_cos_eq_l816_816227

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end sin_minus_cos_eq_l816_816227


namespace points_B_K1_K2_collinear_l816_816331

theorem points_B_K1_K2_collinear
  (A B C B1 B2 K1 K2 : Type)
  [IsScaleneTriangle A B C]
  (hB1 : ∃ (l : LineCircleIntersection), l ⟶ B1 ∧ BisectsAngle B A C)
  (hB2 : ∃ (m : LineCircleIntersection), m ⟶ B2 ∧ BisectsExteriorAngle B A C)
  (hK1 : ∃ (t : TangentToIncircle), t ⟶ K1 ∧ TangentAtPointOfIncircle B1)
  (hK2 : ∃ (t' : TangentToIncircle), t' ⟶ K2 ∧ TangentAtPointOfIncircle B2) :
  Collinear B K1 K2 :=
sorry

end points_B_K1_K2_collinear_l816_816331


namespace line_circle_relationship_l816_816092

theorem line_circle_relationship (m n : ℝ) :
  (∃ l : ℝ, ∃ t : ℝ, l = (t+1)*m + (t-1)*n ∧ l = 0) →
  (x^2 + y^2 = 2) →
  l.passes_through (-1, 1) →
  (intersection_or_tangent (x^2 + y^2 = 2) (∃ l : ℝ, ∃ t : ℝ, l = (t+1)*m + (t-1)*n)) :=
sorry

end line_circle_relationship_l816_816092


namespace general_term_of_sequence_l816_816645

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Sum of the first n terms
axiom sum_of_first_n_terms (n : ℕ) : S n = ∑ i in Finset.range (n + 1), a i

-- Condition given in the problem
axiom graph_lie_condition (n : ℕ) (hn : 0 < n) : (S n) / n = 3 * n - 2

-- We need to prove that this is always true
theorem general_term_of_sequence (n : ℕ) (hn : 0 < n) : a n = 6 * n - 5 :=
by
  sorry

end general_term_of_sequence_l816_816645


namespace arrangement_two_girls_next_to_each_other_l816_816977

theorem arrangement_two_girls_next_to_each_other :
  let boys := 4
  let girls := 3
  in (∃ arrangements, number_of_arrangements_exactly_two_girls_next_to_each_other boys girls arrangements ∧ arrangements = 2880) :=
by
  sorry

end arrangement_two_girls_next_to_each_other_l816_816977


namespace solve_quadratic_l816_816830

theorem solve_quadratic :
  ∀ x : ℝ, (x^2 - 3 * x + 2 = 0) → (x = 1 ∨ x = 2) :=
by sorry

end solve_quadratic_l816_816830


namespace particle_returns_to_initial_position_after_120_moves_l816_816474

noncomputable def omega := Complex.exp (Complex.I * Real.pi / 6)

noncomputable def particle_position (n : Nat) : ℂ :=
  6 * omega^n + 12 * (∑ i in Finset.range n, omega^i)

theorem particle_returns_to_initial_position_after_120_moves :
  particle_position 120 = 6 :=
by
  have h_omega_12 : omega^12 = 1 := by
    sorry  -- Contains the proof that omega^12 = 1

  have h_omega_120 : omega^120 = 1 := by
    rw [←Nat.mul_div_cancel 120 (by norm_num : 12 > 0), pow_mul, h_omega_12]
  
  rw [particle_position, h_omega_120]
  simp only [geom_sum]
  sorry  -- Contains the detailed steps to simplify the geometric series sum

end particle_returns_to_initial_position_after_120_moves_l816_816474


namespace log_relation_l816_816141

-- Definitions for the conditions
def a : ℝ := log 28 / log 4
def b : ℝ := log 35 / log 5
def c : ℝ := log 42 / log 6

-- The theorem statement correlating the conditions to the question
theorem log_relation : a > b ∧ b > c := 
by 
  sorry

end log_relation_l816_816141


namespace cos_315_is_sqrt2_div_2_l816_816519

noncomputable def cos_315_eq : Prop := 
  real.cos (315 * real.pi / 180) = real.sqrt 2 / 2

theorem cos_315_is_sqrt2_div_2 : cos_315_eq := by 
  sorry

end cos_315_is_sqrt2_div_2_l816_816519


namespace train_pass_tree_in_time_l816_816485

-- Definitions from the given conditions
def train_length : ℚ := 270  -- length in meters
def train_speed_km_per_hr : ℚ := 108  -- speed in km/hr

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (v : ℚ) : ℚ := v * (5 / 18)

-- Speed of the train in m/s
def train_speed_m_per_s : ℚ := km_per_hr_to_m_per_s train_speed_km_per_hr

-- Question translated into a proof problem
theorem train_pass_tree_in_time :
  train_length / train_speed_m_per_s = 9 :=
by
  sorry

end train_pass_tree_in_time_l816_816485


namespace friends_boat_crossing_impossible_l816_816966

theorem friends_boat_crossing_impossible : 
  ∀ (friends : Finset ℕ) (boat_capacity : ℕ), friends.card = 5 → boat_capacity ≥ 5 → 
  ¬ (∀ group : Finset ℕ, group ⊆ friends → group ≠ ∅ → group.card ≤ boat_capacity → 
  ∃ crossing : ℕ, (crossing = group.card ∧ group ⊆ friends)) :=
by
  intro friends boat_capacity friends_card boat_capacity_cond goal
  sorry

end friends_boat_crossing_impossible_l816_816966


namespace todd_initial_money_l816_816421

-- Definitions of the conditions
def cost_per_candy_bar : ℕ := 2
def number_of_candy_bars : ℕ := 4
def money_left : ℕ := 12
def total_money_spent := number_of_candy_bars * cost_per_candy_bar

-- The statement proving the initial amount of money Todd had
theorem todd_initial_money : 
  (total_money_spent + money_left) = 20 :=
by
  sorry

end todd_initial_money_l816_816421


namespace polynomial_exists_l816_816151

noncomputable def quintic_polynomial : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ :=
  { -- Notice: explicit set notation might need auxiliary definitions or theorem 
    -- specialized attributes/function within lean proofs that can be omitted 
    -- in this high-level description.
    a := 14, a1 := 15, a2 := 16, a3 := 17, a4 := 18, a5 := 19, a6 := 20
  }

theorem polynomial_exists :
  ∃ p : ℤ[X],
    (∀ k ∈ {14, 15, 17, 18, 20}, p.eval k = k) ∧
    (p.eval 16 = 0) :=
sorry

end polynomial_exists_l816_816151


namespace coefficient_x3_in_expansion_l816_816946

/-- The binomial coefficient function. -/
def binom (n k : ℕ) : ℕ := n.choose k

/-- Coefficient of x^3 in the expansion of (1 - 2x)^5 is -80. -/
theorem coefficient_x3_in_expansion : 
  let a := 1
  let b := -2 * x
  let n := 5
  let r := 3
  binom 5 3 * a^(n - r) * b^r = -80 :=
by
  let a := 1
  let b := -2 * x
  let n := 5
  let r := 3
  rw [←binom]
  sorry

end coefficient_x3_in_expansion_l816_816946


namespace sum_a_n_eq_neg2_l816_816665

noncomputable def x_n (n : ℕ) [fact (n > 0)] : ℝ := (n : ℝ) / (n + 1 : ℝ)
noncomputable def a_n (n : ℕ) [fact (n > 0)] : ℝ := Real.log10 (x_n n)

theorem sum_a_n_eq_neg2 : ∑ n in Finset.range 99, a_n (n + 1) = -2 :=
by
  sorry

end sum_a_n_eq_neg2_l816_816665


namespace sin_of_angle_opposite_leg_20_in_right_triangle_l816_816916

theorem sin_of_angle_opposite_leg_20_in_right_triangle :
  ∀ (h : ℝ), h = real.sqrt(20^2 + 21^2) → 
  sin (real.arcsin (20 / h)) = 20 / h :=
by
  intros h hyp
  have h_eq : h = 29 := by
    rw [hyp, real.sqrt_eq_rsqrt]
    norm_num
  rw [h_eq, real.arcsin_sin]
  -- conclude with the conditions satisfied for arcsin function
  norm_num
  sorry

end sin_of_angle_opposite_leg_20_in_right_triangle_l816_816916


namespace cos_315_eq_sqrt2_div_2_l816_816567

theorem cos_315_eq_sqrt2_div_2 : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
sorry

end cos_315_eq_sqrt2_div_2_l816_816567


namespace num_ordered_pairs_l816_816171

theorem num_ordered_pairs :
  ∃ n : ℕ, n = 49 ∧ ∀ (a b : ℕ), a + b = 50 → 0 < a ∧ 0 < b → (1 ≤ a ∧ a < 50) :=
by
  sorry

end num_ordered_pairs_l816_816171


namespace total_students_l816_816497

-- Defining the conditions
variable (H : ℕ) -- Number of students who ordered hot dogs
variable (students_ordered_burgers : ℕ) -- Number of students who ordered burgers

-- Given conditions
def burger_condition := students_ordered_burgers = 30
def hotdog_condition := students_ordered_burgers = 2 * H

-- Theorem to prove the total number of students
theorem total_students (H : ℕ) (students_ordered_burgers : ℕ) 
  (h1 : burger_condition students_ordered_burgers) 
  (h2 : hotdog_condition students_ordered_burgers H) : 
  students_ordered_burgers + H = 45 := 
by
  sorry

end total_students_l816_816497


namespace probability_negative_slope_l816_816158

open Classical

-- Definitions for A and B and their valid values
inductive AB : Type
| a1 : AB -- -3
| a2 : AB -- -1
| a3 : AB -- 1
| a4 : AB -- 2

open AB

-- Convert AB to their actual integer values
def toInt : AB → Int
| a1 => -3
| a2 => -1
| a3 => 1
| a4 => 2

def slope_is_negative (A B : AB) : Prop :=
  toInt A * toInt B > 0

def validPairs : List (AB × AB) :=
  [ (a1, a3), (a1, a4), (a2, a4), (a3, a1), (a3, a4), (a4, a1), (a4, a2) ]

def count_negative_slopes : Int :=
  (validPairs.filter (λ pair => slope_is_negative pair.1 pair.2)).length

def total_valid_pairs : Int := validPairs.length

theorem probability_negative_slope :
  (count_negative_slopes.toRat / total_valid_pairs.toRat = 1 / 3) :=
by
  -- Place holder proof
  sorry

end probability_negative_slope_l816_816158


namespace dot_product_a_b_l816_816681

namespace VectorDotProduct

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, -1)

theorem dot_product_a_b : a.1 * b.1 + a.2 * b.2 = -1 := by
  -- Referencing the components of tuples a and b
  have h1 : a.1 = 1 := rfl
  have h2 : a.2 = 2 := rfl
  have h3 : b.1 = 1 := rfl
  have h4 : b.2 = -1 := rfl
  have h5 : a.1 * b.1 = 1 * 1 := by rw [h1, h3]
  have h6 : a.2 * b.2 = 2 * (-1) := by rw [h2, h4]
  rw [h5, h6]
  norm_num
  -- Adding the two products
  sorry

end VectorDotProduct

end dot_product_a_b_l816_816681


namespace problem_proof_l816_816330

noncomputable def line_l := (t : ℝ) → (x, y) := (t, -sqrt 3 * t)

noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ :=
  (x, y) := (cos θ, 1 + sin θ)

noncomputable def polar_to_rect (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

noncomputable def polar_eq_C2 := ∀ θ, polar_to_rect (-2 * cos θ + 2 * sqrt 3 * sin θ) θ = (x, y) = (x, y) → x^2 + y^2 + 2x - 2 * sqrt 3 * y = 0

theorem problem_proof :
  (∀ θ, polar_to_rect (2 * sin θ) θ = curve_C1 θ) ∧
  (∀ θ, (let (ρ, θ) := curve_C1 θ in polar_to_rect ρ θ = polar_eq_C2)) ∧
  (let t := 2 / sqrt 3 in let OA := |line_l t|.1 in
   let OB := -2 * cos (2 * pi / 3) + 2 * sqrt 3 * sin (2 * pi / 3) in
   |OB - OA| = 4 - sqrt 3)
:= sorry

end problem_proof_l816_816330


namespace count_multiples_9_ending_in_7_l816_816684

theorem count_multiples_9_ending_in_7 :
  let n := 9 in
  let condition := λ x : ℕ, n * x < 800 ∧ n * x % 10 = 7 in
  (finset.card (finset.filter condition (finset.range 100))) = 9 :=
by
  sorry

end count_multiples_9_ending_in_7_l816_816684


namespace train_passes_jogger_in_36_seconds_l816_816443

/-- A jogger runs at 9 km/h, 240m ahead of a train moving at 45 km/h.
The train is 120m long. Prove the train passes the jogger in 36 seconds. -/
theorem train_passes_jogger_in_36_seconds
  (distance_ahead : ℝ)
  (jogger_speed_km_hr train_speed_km_hr train_length_m : ℝ)
  (jogger_speed_m_s train_speed_m_s relative_speed_m_s distance_to_cover time_to_pass : ℝ)
  (h1 : distance_ahead = 240)
  (h2 : jogger_speed_km_hr = 9)
  (h3 : train_speed_km_hr = 45)
  (h4 : train_length_m = 120)
  (h5 : jogger_speed_m_s = jogger_speed_km_hr * 1000 / 3600)
  (h6 : train_speed_m_s = train_speed_km_hr * 1000 / 3600)
  (h7 : relative_speed_m_s = train_speed_m_s - jogger_speed_m_s)
  (h8 : distance_to_cover = distance_ahead + train_length_m)
  (h9 : time_to_pass = distance_to_cover / relative_speed_m_s) :
  time_to_pass = 36 := 
sorry

end train_passes_jogger_in_36_seconds_l816_816443


namespace find_solutions_of_x4_minus_16_l816_816619

noncomputable def solution_set : Set Complex :=
  {2, -2, Complex.I * 2, -Complex.I * 2}

theorem find_solutions_of_x4_minus_16 :
  {x : Complex | x^4 - 16 = 0} = solution_set :=
by
  sorry

end find_solutions_of_x4_minus_16_l816_816619


namespace mean_marks_second_section_l816_816479

-- Definitions for the problem conditions
def num_students (section1 section2 section3 section4 : ℕ) : ℕ :=
  section1 + section2 + section3 + section4

def total_marks (section1 section2 section3 section4 : ℕ) (mean1 mean2 mean3 mean4 : ℝ) : ℝ :=
  section1 * mean1 + section2 * mean2 + section3 * mean3 + section4 * mean4

-- The final problem translated into a lean statement
theorem mean_marks_second_section :
  let section1 := 65
  let section2 := 35
  let section3 := 45
  let section4 := 42
  let mean1 := 50
  let mean3 := 55
  let mean4 := 45
  let overall_average := 51.95
  num_students section1 section2 section3 section4 = 187 →
  ((section1 : ℝ) * mean1 + (section2 : ℝ) * M + (section3 : ℝ) * mean3 + (section4 : ℝ) * mean4)
    = 187 * overall_average →
  M = 59.99 :=
by
  intros section1 section2 section3 section4 mean1 mean3 mean4 overall_average Hnum Htotal
  sorry

end mean_marks_second_section_l816_816479


namespace train_lost_time_l816_816486

-- Definitions of conditions
def speed_car : ℝ := 120
def speed_train (speed_car : ℝ) : ℝ := speed_car * 1.5
def distance : ℝ := 75
def time_without_stops (distance : ℝ) (speed : ℝ) : ℝ := distance / speed
def time_lost (time_car : ℝ) (time_train : ℝ) : ℝ := time_car - time_train

-- Theorem to prove the time lost is 12.5 minutes.
theorem train_lost_time :
  let car_time := time_without_stops distance speed_car,
      train_time := time_without_stops distance (speed_train speed_car),
      lost_time := time_lost car_time train_time
  in lost_time * 60 = 12.5 :=
by
  sorry

end train_lost_time_l816_816486


namespace total_students_l816_816498

-- Defining the conditions
variable (H : ℕ) -- Number of students who ordered hot dogs
variable (students_ordered_burgers : ℕ) -- Number of students who ordered burgers

-- Given conditions
def burger_condition := students_ordered_burgers = 30
def hotdog_condition := students_ordered_burgers = 2 * H

-- Theorem to prove the total number of students
theorem total_students (H : ℕ) (students_ordered_burgers : ℕ) 
  (h1 : burger_condition students_ordered_burgers) 
  (h2 : hotdog_condition students_ordered_burgers H) : 
  students_ordered_burgers + H = 45 := 
by
  sorry

end total_students_l816_816498


namespace min_value_of_c_l816_816293

noncomputable def isPerfectSquare (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m^2

noncomputable def isPerfectCube (x : ℕ) : Prop :=
  ∃ n : ℕ, x = n^3

theorem min_value_of_c (c : ℕ) :
  (∃ a b d e : ℕ, a = c-2 ∧ b = c-1 ∧ d = c+1 ∧ e = c+2 ∧ a < b ∧ b < c ∧ c < d ∧ d < e) ∧
  isPerfectSquare (3 * c) ∧
  isPerfectCube (5 * c) →
  c = 675 :=
sorry

end min_value_of_c_l816_816293


namespace sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816208

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_10_over_5_l816_816208


namespace combination_identity_example_l816_816938

-- Define the binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- State the problem using the combination identity and perform the required calculations to  show that C(8, 2) + C(8, 3) = 84.
theorem combination_identity_example : C 8 2 + C 8 3 = 84 :=
by
  -- We use the identity C(n + 1, k) = C(n, k) + C(n, k - 1)
  have h : C 9 3 = C 8 3 + C 8 2,
  -- Now, we calculate C(9, 3) = 9! / (3! * (9 - 3)!) = 84
  calc
    C 9 3 = 84 := by sorry
  -- Thus, conclude the proof by substituting and simplifying
  sorry

end combination_identity_example_l816_816938


namespace prime_power_factors_l816_816965

theorem prime_power_factors (x : ℕ) (hx : Nat.Prime x) : 
    (PrimeCount (4^11) + PrimeCount (x^5) + PrimeCount (11^2) = 29) :=
by
  have h₁ : PrimeCount (2^22) = 22 := 
    by sorry
  have h₂ : PrimeCount (11^2) = 2 := 
    by sorry
  have h₃ : PrimeCount (x^5) = 5 := 
    by sorry
  calc PrimeCount (2^22) + PrimeCount (x^5) + PrimeCount (11^2) 
       = 22 + 5 + 2 : by simp [h₁, h₂, h₃]
       ... = 29 : by norm_num

end prime_power_factors_l816_816965


namespace sin_minus_cos_theta_l816_816249

theorem sin_minus_cos_theta (θ : ℝ) (h₀ : 0 < θ ∧ θ < (π / 2)) 
  (h₁ : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -√10 / 5 :=
by
  sorry

end sin_minus_cos_theta_l816_816249


namespace captivating_quadruples_count_l816_816584

theorem captivating_quadruples_count :
  (∃ n : ℕ, n = 682) ↔ 
  ∃ (a b c d : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + d < b + c :=
sorry

end captivating_quadruples_count_l816_816584


namespace product_series_eq_l816_816933

theorem product_series_eq : 
  (∏ n in finset.range (10 - 2 + 1) + 2, ((1 : ℚ) + 1/n) * (1 - 1/n)) = 11 / 20 :=
by
  sorry

end product_series_eq_l816_816933


namespace area_of_polar_curve_l816_816033

open Real
open IntervalIntegrals

theorem area_of_polar_curve (φ : ℝ) : 
  (∫ φ in (0 : ℝ).. (π / 6), 3 * cos (3 * φ) ^ 2 / 2) * 6 = π / 4 :=
by
  -- The proof is skipped.
  sorry

end area_of_polar_curve_l816_816033


namespace find_natural_n_l816_816357

-- Define the sum of digits function s(n)
def sum_of_digits : ℕ → ℕ
| 0       := 0
| (n + 1) := (n + 1) % 10 + sum_of_digits ((n + 1) / 10)

theorem find_natural_n (n : ℕ) (h : n + sum_of_digits n = 2011) : n = 1991 := 
sorry

end find_natural_n_l816_816357


namespace log_problem_l816_816657

open Real

theorem log_problem (x y : ℝ) (h : log 10 x + log 10 y = 2 * log 10 (x - 2 * y)) :
  log (sqrt 2) x - log (sqrt 2) y = 4 :=
by
  sorry

end log_problem_l816_816657


namespace min_value_a2b3c_l816_816988

theorem min_value_a2b3c {m : ℝ} (hm : m > 0)
  (hineq : ∀ x : ℝ, |x + 1| + |2 * x - 1| ≥ m)
  {a b c : ℝ} (habc : a^2 + 2 * b^2 + 3 * c^2 = m) :
  a + 2 * b + 3 * c ≥ -3 :=
sorry

end min_value_a2b3c_l816_816988


namespace cos_315_is_sqrt2_div_2_l816_816518

noncomputable def cos_315_eq : Prop := 
  real.cos (315 * real.pi / 180) = real.sqrt 2 / 2

theorem cos_315_is_sqrt2_div_2 : cos_315_eq := by 
  sorry

end cos_315_is_sqrt2_div_2_l816_816518


namespace ball_returns_to_becca_after_16_throws_l816_816597

-- Define the problem statement and conditions
def num_girls : ℕ := 15

def throws (current_girl : ℕ) (skipped : ℕ) : ℕ :=
  (current_girl + skipped) % num_girls

def total_throws : ℕ :=
  let skip_increase := [5, 6, 7, 8, 9]
  let throw_length := 3 * skip_increase.length
  let throw_sequence := List.zip (List.range throw_length).map
    (fun n => throws (1 + n) (skip_increase[n / 3 % skip_increase.length]))
  -- Checks when the ball returns to girl 1
  throw_sequence.findIndex (fun girl => girl == 1) + 1

theorem ball_returns_to_becca_after_16_throws : total_throws = 16 :=
by
  sorry

end ball_returns_to_becca_after_16_throws_l816_816597


namespace max_voltage_on_capacitor_l816_816725

-- Define the given conditions
def C : ℝ := 2 * 10^(-6)  -- Capacitance in Farads
def L2 : ℝ := 1 * 10^(-3)  -- Inductance in Henrys
def I_max : ℝ := 5 * 10^(-3)  -- Maximum current in Amperes
def L1 : ℝ := 2 * 10^(-3)  -- Inductance in Henrys

-- State the theorem to prove the maximum voltage on the capacitor
theorem max_voltage_on_capacitor : 
  (√((L2 * I_max^2) / C - ((L1 + L2) * (L2 * (I_max / (L1 + L2)))^2) / C) = 90 * 10^(-3)) :=
sorry

end max_voltage_on_capacitor_l816_816725


namespace solve_x_l816_816896

section diamondsuit_problem

variables (a b c x : ℝ)

-- define binary operation
def diamondsuit (a b : ℝ) : ℝ :=
  if a ≠ 0 then if b ≠ 0 then a * b else 1 / a else 0

-- define the conditions
axiom diamondsuit_assoc : ∀ (a b c : ℝ), a ≠ 0 -> b ≠ 0 -> c ≠ 0 -> diamondsuit a (diamondsuit b c) = (diamondsuit a b) / c
axiom diamondsuit_idem : ∀ (a : ℝ), a ≠ 0 -> diamondsuit a a = 1

-- define the equation to solve
def equation : Prop := diamondsuit 504 (diamondsuit 7 x) = 50

-- state the theorem
theorem solve_x : equation x → x = 25 / 1764 := by
  sorry

end diamondsuit_problem

end solve_x_l816_816896


namespace squared_sum_inequality_l816_816638

variable {n : ℕ}
variable {x : fin n → ℝ}
def arithmetic_mean (x : fin n → ℝ) : ℝ := 
  (∑ k, x k) / n

theorem squared_sum_inequality (x : fin n → ℝ) :
  ∑ k, (x k - arithmetic_mean x) ^ 2 ≤ (1 / 2) * (∑ k, |x k - arithmetic_mean x|) ^ 2 :=
sorry

end squared_sum_inequality_l816_816638


namespace reading_time_sample_l816_816371

theorem reading_time_sample (total_residents sample_size : ℕ) (sample : List ℕ) :
  total_residents = 5000 ∧ sample_size = 200 ∧ sample.length = sample_size →
  sample = (List.range sample_size).map (fun n => -- some function giving the reading time of the nth sampled resident --) := sorry

end reading_time_sample_l816_816371


namespace bicycle_travel_rate_l816_816472

noncomputable def total_distance := 80 -- in km
noncomputable def total_time := 7 -- in hours
noncomputable def distance_on_foot := 32 -- in km
noncomputable def rate_on_foot := 8 -- in km/h

noncomputable def rate_on_bicycle := 16 -- in km/h

theorem bicycle_travel_rate :
  let time_on_foot := distance_on_foot / rate_on_foot,
      time_on_bicycle := total_time - time_on_foot,
      distance_on_bicycle := total_distance - distance_on_foot,
      rate_on_bicycle' := distance_on_bicycle / time_on_bicycle
  in rate_on_bicycle' = rate_on_bicycle := by
  sorry

end bicycle_travel_rate_l816_816472


namespace probability_one_shows_three_given_sum_seven_l816_816872

noncomputable def fair_die : List ℕ := [1, 2, 3, 4, 5, 6]

def all_possible_outcomes (dice: List ℕ) : List (ℕ × ℕ) := 
  List.product dice dice

def outcomes_with_sum_seven : List (ℕ × ℕ) :=
  List.filter (fun p => p.1 + p.2 = 7) (all_possible_outcomes fair_die)

def outcomes_with_three (outcomes: List (ℕ × ℕ)) : List (ℕ × ℕ) :=
  List.filter (fun p => p.1 = 3 ∨ p.2 = 3) outcomes

theorem probability_one_shows_three_given_sum_seven : 
  (outcomes_with_three outcomes_with_sum_seven).length / outcomes_with_sum_seven.length = 1 / 3 :=
by
  sorry

end probability_one_shows_three_given_sum_seven_l816_816872


namespace sequence_property_n_l816_816109

theorem sequence_property_n (n : ℕ) (h1 : n > 0): 
  (∃ (x : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ n → (x i) ∈ finset.range (n + 1) ∧ (∀ j k, 1 ≤ j ∧ j < k ∧ k ≤ n → x j ≠ x k)) ∧ 
  (∀ k, 1 ≤ k ∧ k ≤ n → k ∣ (finset.sum (finset.range n.succ) (λ i, x i)))
  ) ↔ n = 1 ∨ n = 3 :=
sorry

end sequence_property_n_l816_816109


namespace arithmetic_progression_implies_equality_l816_816108

theorem arithmetic_progression_implies_equality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ((a + b) / 2) = ((Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2)) / 2) → a = b :=
by
  sorry

end arithmetic_progression_implies_equality_l816_816108


namespace taxi_service_charge_l816_816744

theorem taxi_service_charge (initial_fee : ℝ) (additional_charge : ℝ) (increment : ℝ) (total_charge : ℝ) 
  (h_initial_fee : initial_fee = 2.25) 
  (h_additional_charge : additional_charge = 0.4) 
  (h_increment : increment = 2 / 5) 
  (h_total_charge : total_charge = 5.85) : 
  ∃ distance : ℝ, distance = 3.6 :=
by
  sorry

end taxi_service_charge_l816_816744


namespace decreasing_at_most_one_zero_l816_816299

variable {ℝ : Type} [LinearOrder ℝ] [TopologicalSpace ℝ]
variable {a b : ℝ}
variable {f : ℝ → ℝ}

theorem decreasing_at_most_one_zero (h_decreasing : ∀ x y ∈ set.Icc a b, x < y → f x > f y) 
    (h_continuous : continuous_on f (set.Icc a b)) : 
    ∀ x1 x2 ∈ set.Icc a b, f x1 = 0 → f x2 = 0 → x1 = x2 := 
by
  sorry

end decreasing_at_most_one_zero_l816_816299


namespace feeding_sequence_count_l816_816917

theorem feeding_sequence_count:
  ∃ (animals : Finset (String × String)), 
    (animals.card = 5) ∧ 
    (∀ animal ∈ animals, animal.fst ≠ animal.snd) ∧ 
    (start_animal : (String × String)) ∈ animals ∧ 
    (start_animal.fst = "male lion") ∧ 
    (feed_sequence_count : ℕ), 
    (feed_sequence_count = 5 * 4^2 * 3^2 * 2^2 * 1^2) :=
begin
  use (Finset.of_list [("male lion", "female lion"), ("male tiger", "female tiger"), ("male elephant", "female elephant"), 
                      ("male zebra", "female zebra"), ("male giraffe", "female giraffe")]),
  split, 
  { show (Finset.of_list [_,_,_,_,_]).card = 5, 
    unfold Finset.card,
    norm_num,
  },
  split, 
  { intros animal h, 
    let animals := (Finset.of_list [_,_,_,_,_]), 
    unfold Finset.mem at h,
    repeat { cases animal, 
             simp at h,
             assumption 
           },
  },
  split,
  { -- show starting animal exists in animals
    let animals := (Finset.of_list [_,_,_,_,_]), 
    show ("male lion", "female lion") ∈ animals,
    unfold Finset.mem,
    norm_num,
  },
  split,
  { -- start_animal.fst = "male lion"
    reflexivity,
  },
  use 2880,
  show  2880 = 5 * 4^2 * 3^2 * 2^2 * 1^2,
  norm_num,
  sorry
end

end feeding_sequence_count_l816_816917


namespace sqrt_sum_fractions_l816_816023

theorem sqrt_sum_fractions : sqrt ((16 / 25 : ℚ) + (9 / 4 : ℚ)) = (17 / 10 : ℚ) :=
by
  sorry

end sqrt_sum_fractions_l816_816023


namespace correct_interpretations_of_implication_l816_816000

theorem correct_interpretations_of_implication (p q : Prop) (h : p → q) : 
  (q can be deduced from p by reasoning) ∧ 
  (p → q) ∧ 
  (¬q → ¬p) ∧ 
  (p is a sufficient condition for q) ∧ 
  (q is a necessary condition for p) 
:= sorry

end correct_interpretations_of_implication_l816_816000


namespace total_vertical_distance_is_210_l816_816722

-- Define the characteristics of the rings
def ring_thickness := 2
def top_ring_outer_diameter := 30
def bottom_ring_outer_diameter := 4

-- Prove the total vertical distance given these characteristics
theorem total_vertical_distance_is_210 :
  let num_rings := (top_ring_outer_diameter - bottom_ring_outer_diameter) / ring_thickness + 1 in
  let top_ring_inner_diameter := top_ring_outer_diameter - ring_thickness in
  let bottom_ring_inner_diameter := bottom_ring_outer_diameter - ring_thickness in
  let total_distance := num_rings * ring_thickness in
  total_distance = 210 :=
by
  sorry

end total_vertical_distance_is_210_l816_816722


namespace sin_minus_cos_l816_816218

variable (θ : ℝ)
hypothesis (h1 : 0 < θ ∧ θ < π / 2)
hypothesis (h2 : Real.tan θ = 1 / 3)

theorem sin_minus_cos (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := 
by 
  sorry

end sin_minus_cos_l816_816218


namespace cos_315_proof_l816_816555

noncomputable def cos_315_eq : Prop :=
  let theta := 45 in
  let angle := 360 - theta in
  ∃ c : real, (cos 45) = (sqrt 2 / 2) ∧ (cos (360 - theta) = cos theta) ∧ 315 = angle ∧ (forall a : real, cos a = cos (360 - a)) ∧ (315 ∈ (4th quadrant)) ∧ (cos 315 = cos 45)

theorem cos_315_proof : (cos 315 = sqrt 2 / 2) :=
begin
  -- We start by defining basic facts
  -- 315 = 360 - 45
  -- Cosine is positive in the fourth quadrant, known value \cos 45 = sqrt 2 / 2
  -- Cosine property \cos (360 - \theta) = \cos \theta
  let theta := 45,
  let angle := 360 - theta,
  have H1 : cos theta = sqrt 2 / 2,
  {
    -- known fact
    sorry,
  },
  have H2 : cos (360 - theta) = cos theta,
  {
    -- cosine property
    sorry,
  },
  have H3 : angle = 315,
  {
    -- subtraction identity
    sorry,
  },
  rw H3,
  rw H2,
  rw H1,
  refl,
end

end cos_315_proof_l816_816555


namespace oxygen_atom_count_l816_816468

-- Definitions and conditions
def molecular_weight_C : ℝ := 12.01
def molecular_weight_H : ℝ := 1.008
def molecular_weight_O : ℝ := 16.00

def num_carbon_atoms : ℕ := 4
def num_hydrogen_atoms : ℕ := 1
def total_molecular_weight : ℝ := 65.0

-- Theorem statement
theorem oxygen_atom_count : 
  ∃ (num_oxygen_atoms : ℕ), 
  num_oxygen_atoms * molecular_weight_O = total_molecular_weight - (num_carbon_atoms * molecular_weight_C + num_hydrogen_atoms * molecular_weight_H) 
  ∧ num_oxygen_atoms = 1 :=
by
  sorry

end oxygen_atom_count_l816_816468


namespace negation_proof_l816_816655

theorem negation_proof :
  let P := λ x : ℝ, (1 / (x^2 - x - 2)) > 0
  ∀ x : ℝ, ¬ P x ↔ -1 <= x ∧ x <= 2 :=
by
  sorry

end negation_proof_l816_816655


namespace time_saved_by_taking_route_B_l816_816343

-- Defining the times for the routes A and B
def time_route_A_one_way : ℕ := 5
def time_route_B_one_way : ℕ := 2

-- The total round trip times
def time_route_A_round_trip : ℕ := 2 * time_route_A_one_way
def time_route_B_round_trip : ℕ := 2 * time_route_B_one_way

-- The statement to prove
theorem time_saved_by_taking_route_B :
  time_route_A_round_trip - time_route_B_round_trip = 6 :=
by
  -- Proof would go here
  sorry

end time_saved_by_taking_route_B_l816_816343


namespace total_students_are_45_l816_816495

theorem total_students_are_45 (burgers hot_dogs students : ℕ)
  (h1 : burgers = 30)
  (h2 : burgers = 2 * hot_dogs)
  (h3 : students = burgers + hot_dogs) : students = 45 :=
sorry

end total_students_are_45_l816_816495


namespace least_number_divisible_remainder_l816_816446

theorem least_number_divisible_remainder (n : ℕ) (h1 : n % 34 = 4) (h2 : n % 5 = 4) : n = 174 := 
sorry

end least_number_divisible_remainder_l816_816446


namespace initial_rhinoceroses_planned_l816_816491

namespace WildlifePreserve

def initial_rhinoceroses (watering_area grazing_area_per_rhinoceros : ℕ) (expected_increase_perc future_area : ℕ) : ℕ :=
  let R := future_area - watering_area in
  R / (grazing_area_per_rhinoceros * (1 + expected_increase_perc / 100))

theorem initial_rhinoceroses_planned (watering_area : ℕ) (grazing_area_per_rhinoceros : ℕ)
  (expected_increase_perc : ℕ) (future_area : ℕ) (R : ℕ) :
  watering_area = 10000 ∧ grazing_area_per_rhinoceros = 100 ∧ expected_increase_perc = 10 ∧ future_area = 890000 ∧ R = 8000 :=
by
  sorry

end WildlifePreserve

end initial_rhinoceroses_planned_l816_816491


namespace solve_equation_l816_816381

noncomputable def equation (x : ℝ) : Prop :=
  2 / (x - 2) = (1 + x) / (x - 2) + 1

theorem solve_equation : ∀ (x : ℝ), equation x ∧ x ≠ 2 ↔ x = 3 / 2 :=
by
  intro x
  split
  sorry
  sorry

end solve_equation_l816_816381


namespace distinct_prime_factors_inequality_l816_816155

theorem distinct_prime_factors_inequality
  (x y z : ℕ)
  (hx : x > 2)
  (hy : y > 1)
  (hxy : x^y + 1 = z^2) :
  let p := (nat.factors x).nodup.card in
  let q := (nat.factors y).nodup.card in
  p ≥ q + 2 :=
by 
  sorry

end distinct_prime_factors_inequality_l816_816155


namespace intersection_l816_816693

def setA : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) }
def setB : Set ℝ := { x | x^2 - 2 * x ≥ 0 }

theorem intersection: setA ∩ setB = { x : ℝ | x ≤ 0 } := by
  sorry

end intersection_l816_816693


namespace range_of_a_not_monotonic_l816_816671

theorem range_of_a_not_monotonic :
  ∀ (a : ℝ), (∃ x : ℝ, f(x) = 3 * x^2 - 2 * a * x - 8) →
  ¬ monotonic_on f (set.Ioo (1 : ℝ) (2 : ℝ)) ↔ (3 < a ∧ a < 6) :=
begin
  sorry
end

-- Additional auxiliary definitions if needed
def f (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - 2 * a * x - 8

-- Lean will handle the rest

end range_of_a_not_monotonic_l816_816671


namespace eval_expr_l816_816016

theorem eval_expr : 4 * (8 - 3 + 2) / 2 = 14 := 
by
  sorry

end eval_expr_l816_816016


namespace bcm_percentage_l816_816905

theorem bcm_percentage (total_chickens : ℕ) (bcm_hens : ℕ) (pct_bcm_hens : ℝ) (pct_bcm : ℝ) : 
  total_chickens = 100 ∧ bcm_hens = 16 ∧ pct_bcm_hens = 0.8 → pct_bcm = 0.20 :=
by
  intro h
  cases' h with ht h1
  cases' h1 with hb h2
  have h_eq : 80 * pct_bcm = 16 := by
    rw [ht, hb, h2]
    sorry  -- Here you would provide the derivation steps
  exact sorry -- Here is where the final conclusion would be stated

end bcm_percentage_l816_816905


namespace runners_time_to_finish_race_l816_816412

theorem runners_time_to_finish_race :
  let
    distance := 15   -- distance of the race in miles
    first_runner_pace := 6  -- minutes per mile
    second_runner_pace := 7  -- minutes per mile
    third_runner_pace := 8  -- minutes per mile
    first_runner_break_time := 3  -- minutes break
    second_runner_break_time := 5  -- minutes break
    third_runner_break_time := 7  -- minutes break
    first_runner_break_after := 42  -- minutes until break
    second_runner_break_after := 49  -- minutes until break
    third_runner_break_after := 56  -- minutes until break
  in
    -- Time calculated for each runner
    let
      first_runner_time :=
        let distance_covered := first_runner_break_after / first_runner_pace in
        let remaining_distance := distance - distance_covered in
        first_runner_break_after + first_runner_break_time + remaining_distance * first_runner_pace
      second_runner_time :=
        let distance_covered := second_runner_break_after / second_runner_pace in
        let remaining_distance := distance - distance_covered in
        second_runner_break_after + second_runner_break_time + remaining_distance * second_runner_pace
      third_runner_time :=
        let distance_covered := third_runner_break_after / third_runner_pace in
        let remaining_distance := distance - distance_covered in
        third_runner_break_after + third_runner_break_time + remaining_distance * third_runner_pace
    in
      first_runner_time = 93 ∧ second_runner_time = 110 ∧ third_runner_time = 127 :=
by
  sorry

end runners_time_to_finish_race_l816_816412


namespace interval_of_increase_g_l816_816844

-- Define the conditions
def is_monotonically_decreasing (f : ℝ → ℝ) := ∀ x1 x2, x1 < x2 → f x1 ≥ f x2

def linear_function (a b : ℝ) : ℝ → ℝ := λ x, a * x + b

-- Define the quadratic function and its interval of decrease
def quadratic_function : ℝ → ℝ := λ x, x^2 - 4 * x + 3
def interval_of_decrease : set ℝ := {x | x ≤ 2}

-- Main statement that needs to be proved
theorem interval_of_increase_g (a : ℝ) (h : is_monotonically_decreasing (linear_function a 1)) :
    a < 0 → (∀ x : ℝ, x ∈ interval_of_decrease → g(x) = a * (x^2 - 4 * x + 3)) :=
begin
    intro ha,
    sorry
end

end interval_of_increase_g_l816_816844


namespace cos_315_eq_sqrt2_div_2_l816_816524

theorem cos_315_eq_sqrt2_div_2 :
  let θ := 315.0
  in θ = 360.0 - 45.0 ∧ (cos 45 * (π / 180)) = (Real.sqrt 2 / 2) →
  (cos (θ * (π / 180))) = (Real.sqrt 2 / 2) :=
by
  intros θ
  -- Non-necessary proof omitted
  sorry

end cos_315_eq_sqrt2_div_2_l816_816524
