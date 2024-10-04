import Mathlib

namespace smallest_number_bounds_l665_665052

theorem smallest_number_bounds (a : ℕ → ℝ) (h_sum : (∑ i in finset.range 8, a i) = 4 / 3)
    (h_pos_sum : ∀ i : fin 8, 0 < ∑ j in (finset.univ \ {i}), a j) :
  -8 < a 0 ∧ a 0 ≤ 1 / 6 :=
by
  sorry

end smallest_number_bounds_l665_665052


namespace repeating_decimal_sum_eq_809_l665_665452

noncomputable def repeating_decimal_fraction_sum : ℚ :=
  let num: ℚ := 710 / 99 in
  num.num + num.denom

theorem repeating_decimal_sum_eq_809 : repeating_decimal_fraction_sum = 809 :=
by
  sorry

end repeating_decimal_sum_eq_809_l665_665452


namespace smallest_four_digit_divisible_by_43_l665_665089

theorem smallest_four_digit_divisible_by_43 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 43 = 0 ∧ n = 1032 :=
by
  use 1032
  split
  { linarith }
  split
  { linarith }
  split
  { norm_num }
  norm_num

end smallest_four_digit_divisible_by_43_l665_665089


namespace problem_sequence_inequality_l665_665537

def a (n : ℕ) : ℚ := 15 + (n - 1 : ℚ) * (-(2 / 3))

theorem problem_sequence_inequality :
  ∃ k : ℕ, (a k) * (a (k + 1)) < 0 ∧ k = 23 :=
by {
  use 23,
  sorry
}

end problem_sequence_inequality_l665_665537


namespace area_diff_of_right_triangle_l665_665420

theorem area_diff_of_right_triangle (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) (h4 : a^2 + b^2 = c^2) :
  let R := c / 2
      r := (a * b) / (a + b + c)
      area_circumcircle := π * R^2
      area_incircle := π * r^2
  in area_circumcircle - area_incircle = 21 * π :=
by {
  sorry
}

end area_diff_of_right_triangle_l665_665420


namespace number_of_boys_in_class_l665_665842

theorem number_of_boys_in_class (B : ℕ) (G : ℕ) (hG : G = 10) (h_combinations : (G * B * (B - 1)) / 2 = 1050) :
    B = 15 :=
by
  sorry

end number_of_boys_in_class_l665_665842


namespace angle_sum_less_90_l665_665157

-- Definitions and conditions
variables {α : Type*} [noncomputable_comm_ring α]

structure triangle :=
(A B C O X : α)
(acute_angled : \(\angle A < 90^\circ\) ∧ \(\angle B < 90^\circ\) ∧ \(\angle C < 90^\circ\))
(circumcenter : \(O\))
(foot_perpendicular : \(X\) is the foot of the perpendicular from \(A\) to \(BC\))
(angle_condition : \(\angle C\) ≥ \(\angle B\) + 30°)

-- The theorem to be proven
theorem angle_sum_less_90 (t : triangle) :
  \(\angle A + \angle COX < 90^\circ\) :=
sorry

end angle_sum_less_90_l665_665157


namespace polynomial_bound_l665_665392

theorem polynomial_bound {P : ℝ → ℝ} {n : ℤ} (hdeg : ∀ x, polynomial.degree P x ≤ 2 * n) 
  (hbound : ∀ k : ℤ, -n ≤ k ∧ k ≤ n → |P k| ≤ 1) :
  ∀ x, -n ≤ x ∧ x ≤ n → |P x| ≤ 2 ^ (2 * n) :=
sorry

end polynomial_bound_l665_665392


namespace number_of_completely_covered_squares_is_20_l665_665508

noncomputable def checkerboard_center : ℝ × ℝ := (3.5, 4.5)
noncomputable def disc_radius : ℝ := 5
noncomputable def square_side_length : ℝ := 1

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def is_square_completely_covered (center : ℝ × ℝ) (radius : ℝ) (square_bottom_left : ℝ × ℝ) : Prop :=
  let corners := [
    square_bottom_left,
    (square_bottom_left.1 + square_side_length, square_bottom_left.2),
    (square_bottom_left.1, square_bottom_left.2 + square_side_length),
    (square_bottom_left.1 + square_side_length, square_bottom_left.2 + square_side_length)
  ] in
  ∀ corner, corner ∈ corners → distance center corner ≤ radius

def count_completely_covered_squares (center : ℝ × ℝ) (radius : ℝ) : ℕ :=
  finset.card $ finset.filter (is_square_completely_covered center radius)
    (finset.product
      (finset.range 6).map (λ i, (i : ℝ))
      (finset.range 8).map (λ j, (j : ℝ)))

theorem number_of_completely_covered_squares_is_20 :
  count_completely_covered_squares checkerboard_center disc_radius = 20 :=
sorry

end number_of_completely_covered_squares_is_20_l665_665508


namespace initial_speed_l665_665521

variable (D T : ℝ) -- Total distance D and total time T
variable (S : ℝ)   -- Initial speed S

theorem initial_speed :
  (2 * D / 3) = (S * T / 3) →
  (35 = (D / (2 * T))) →
  S = 70 :=
by
  intro h1 h2
  -- Skipping the proof with 'sorry'
  sorry

end initial_speed_l665_665521


namespace quadrilateral_false_statement_l665_665992

-- Definitions for quadrilateral properties
def is_rhombus (q : ℝ × ℝ × ℝ × ℝ) : Prop := q.1 = q.2 ∧ q.3 = q.4
def equal_diagonals (d1 d2 : ℝ) : Prop := d1 = d2
def is_rectangle (q : ℝ × ℝ × ℝ × ℝ) : Prop := q.1 = q.2 ∧ q.3 = q.4 ∧ q.1 = 90 ∧ q.3 = 90
def perpendicular (a b : ℝ) : Prop := a * b = 0
def is_parallelogram (q : ℝ × ℝ × ℝ × ℝ) : Prop := q.1 = q.3 ∧ q.2 = q.4
def bisects (d1 d2 : ℝ) : Prop := d1 = d2 / 2

-- The problem statement
theorem quadrilateral_false_statement :
  ¬ (∀ (q : ℝ × ℝ × ℝ × ℝ) (d1 d2 : ℝ),
    (is_rhombus q ∧ equal_diagonals d1 d2 → q.1 = 90 ∧ q.2 = 90) ∧
    (is_rectangle q ∧ perpendicular d1 d2 → q.1 = q.2) ∧
    (is_parallelogram q ∧ perpendicular d1 d2 ∧ equal_diagonals d1 d2 → q.1 = 90 ∧ q.2 = 90) ∧
    (perpendicular d1 d2 ∧ bisects d1 d2 → q.1 = 90 ∧ q.2 = 90)) :=
sorry

end quadrilateral_false_statement_l665_665992


namespace fourth_number_value_l665_665799

variable (A B C D E F : ℝ)

theorem fourth_number_value 
  (h1 : A + B + C + D + E + F = 180)
  (h2 : A + B + C + D = 100)
  (h3 : D + E + F = 105) : 
  D = 25 := 
by 
  sorry

end fourth_number_value_l665_665799


namespace line_intersection_hyperbola_l665_665276

theorem line_intersection_hyperbola 
  (x y : ℝ) 
  (M : ℝ × ℝ) 
  (x1 y1 x2 y2 : ℝ) 
  (A B : ℝ × ℝ) 
  (hA : A = (x1, y1)) 
  (hB : B = (x2, y2)) 
  (hM : M = (1, 2)) 
  (hyperbola_eq : ∀ x y, x^2 - (y^2 / 2) = 1 → ∃ A B, A ≠ B ∧ midPoint ℝ A B = M) 
  (l_eq : ∀ k : ℝ, y - 2 = k * (x - 1)) :
  (∃ A B, (A ≠ B) ∧ (midPoint ℝ A B = M) ∧ (x1, y1) ∈ hyperbola_eq ∧ (x2, y2) ∈ hyperbola_eq ∧ (∃ k : ℝ, line_eq (1, 2) k ∧ equation l = "x - y + 1 = 0" ∧ length AB =  4 * sqrt 2)) := 
sorry


end line_intersection_hyperbola_l665_665276


namespace cube_volume_is_1728_l665_665949

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l665_665949


namespace number_of_tangent_lines_through_point_l665_665601

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - 2*x + 1
noncomputable def f_prime (x : ℝ) : ℝ := 3*x^2 - 2*x - 2

theorem number_of_tangent_lines_through_point :
  ∃! (x0 : ℝ), (tangent_point_through_x0 : (λ (x : ℝ), x = x0 ∨ x = x0)), 
  let tang_line := λ (x0 : ℝ), 1 - (x0^3 - x0^2 - 2*x0 + 1) = (3*x0^2 - 2*x0 - 2)*(-1 - x0) in
  tang_line (-1) ∧ tang_line 1 :=
begin
  sorry
end

end number_of_tangent_lines_through_point_l665_665601


namespace geom_seq_sum_3000_l665_665058

noncomputable
def sum_geom_seq (a r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then a * n
  else a * (1 - r ^ n) / (1 - r)

theorem geom_seq_sum_3000 (a r : ℝ) (h1: sum_geom_seq a r 1000 = 300) (h2: sum_geom_seq a r 2000 = 570) :
  sum_geom_seq a r 3000 = 813 :=
sorry

end geom_seq_sum_3000_l665_665058


namespace products_of_lengths_are_equal_l665_665112

noncomputable def convex_polygon (n : ℕ) := { p : ℕ → (ℝ × ℝ) // (polygon p n).convex } -- A type to represent a convex polygon with n sides

variables {n : ℕ} [fact (2 ≤ n)] (A : convex_polygon n) (B D : ℕ → (ℝ × ℝ))
           (O : ℝ × ℝ) 
           (on_side : ∀ i, 1 ≤ i → i ≤ n → B i ∈ Line (A.val i) (A.val (i+1 ≫ n)) ∧ D (i +1)] -- B and D are on sides respectively
           (parallelogram_construct : ∀ i, parallelogram (A.val i) (B i) (C i) (D i)) -- constructs parallelograms
           (lines_intersect : ∀ i j, i ≠ j → Line (A.val i) (Intersect (A.val] j) = some O)

open_locale big_operators

theorem products_of_lengths_are_equal : 
  ∏ i in finset.range n, dist (A.val i) (B i) = ∏ i in finset.range n, dist (A.val i) (D i) :=
sorry

end products_of_lengths_are_equal_l665_665112


namespace symmetric_point_origin_l665_665441

-- Define the original point
def original_point : ℝ × ℝ := (4, -1)

-- Define a function to find the symmetric point with respect to the origin
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- State the theorem
theorem symmetric_point_origin : symmetric_point original_point = (-4, 1) :=
sorry

end symmetric_point_origin_l665_665441


namespace norm_scalar_mul_l665_665618

variable {V : Type} [NormedAddCommGroup V] {v : V}

theorem norm_scalar_mul :
    ∥v∥ = 5 → ∥-4 • v∥ = 20 :=
by
  intro h
  have : ∥-4 • v∥ = | -4 | * ∥ v ∥ := norm_smul _ _
  rwa [h, abs_neg, abs_of_nonneg] at this
  exact le_of_lt (by norm_num : 4 > 0)
  sorry

end norm_scalar_mul_l665_665618


namespace min_lifespan_ensures_prob_l665_665820

noncomputable def lifespan_distribution : ℝ → ℝ :=
λ x, Pdf.normalPdf 1000 (30^2) x

theorem min_lifespan_ensures_prob :
  ∀ X : ℝ → ℝ, X = lifespan_distribution →
    (∫ x in 910..1090, X x) = 0.997 → (∃ min_lifespan : ℝ, min_lifespan = 910) :=
by sorry

end min_lifespan_ensures_prob_l665_665820


namespace factor_expression_l665_665218

theorem factor_expression (x : ℝ) : 5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l665_665218


namespace smallest_number_bounds_l665_665050

theorem smallest_number_bounds (a : ℕ → ℝ) (h_sum : (∑ i in finset.range 8, a i) = 4 / 3)
    (h_pos_sum : ∀ i : fin 8, 0 < ∑ j in (finset.univ \ {i}), a j) :
  -8 < a 0 ∧ a 0 ≤ 1 / 6 :=
by
  sorry

end smallest_number_bounds_l665_665050


namespace correct_average_l665_665996

theorem correct_average (avg_incorrect : ℕ) (old_num new_num : ℕ) (n : ℕ)
  (h_avg : avg_incorrect = 15)
  (h_old_num : old_num = 26)
  (h_new_num : new_num = 36)
  (h_n : n = 10) :
  (avg_incorrect * n + (new_num - old_num)) / n = 16 := by
  sorry

end correct_average_l665_665996


namespace angle_A_of_isosceles_triangle_l665_665485

/-- Given an isosceles triangle ABC with AB = BC = 8 and a height BD dropped to base AC.
    In triangle BCD, a median DE is drawn, and a circle inscribed in triangle BDE touches
    side BE at point K and side DE at point M, with segment KM equal to 2. Prove that the 
    measure of angle A in triangle ABC is 30 degrees. -/
theorem angle_A_of_isosceles_triangle
  (A B C D E K M : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  [MetricSpace D]
  [MetricSpace E]
  [MetricSpace K]
  [MetricSpace M]
  (hABC : ∀ (A B C : MetricSpace), isIsoscelesTriangle ABC A B C ∧ length AB = 8 ∧ length BC = 8)
  (hBD : ∀ (B D : MetricSpace), LineWithHeight B D)
  (hDE : ∀ (D E : MetricSpace), isMedian D E B C)
  (hCircle : ∃ (circle : Circle), inscribedInCircle circle triangle_BDE ∧ touches circle BE K ∧ touches circle DE M ∧ length KM = 2) :
  measureAngle T A B C = 30 := by
    sorry

end angle_A_of_isosceles_triangle_l665_665485


namespace solve_for_x_l665_665325

theorem solve_for_x : ∀ (x : ℝ), (2 * x + 3) / 5 = 11 → x = 26 :=
by {
  sorry
}

end solve_for_x_l665_665325


namespace problem_statement_l665_665662

noncomputable theory

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

def no_real_roots (f : ℝ → ℝ) := ∀ x : ℝ, (f x) ≠ 2 * x

lemma no_real_roots_of_composite (a b c : ℝ) (h : a ≠ 0) (h_no_roots : no_real_roots (quadratic a b c)) :
  no_real_roots (λ x, quadratic a b c (quadratic a b c x)) :=
sorry

theorem problem_statement (a b c : ℝ) (h : a ≠ 0) (h_no_real_roots : no_real_roots (quadratic a b c)) :
  ∀ x : ℝ, (quadratic a b c (quadratic a b c x) ≠ 4 * x) :=
begin
  apply no_real_roots_of_composite,
  assumption,
  assumption,
end

end problem_statement_l665_665662


namespace gloves_selection_l665_665062

theorem gloves_selection (total_pairs : ℕ) (total_gloves : ℕ) (num_to_select : ℕ) 
    (total_ways : ℕ) (no_pair_ways : ℕ) : 
    total_pairs = 4 → 
    total_gloves = 8 → 
    num_to_select = 4 → 
    total_ways = (Nat.choose total_gloves num_to_select) → 
    no_pair_ways = 2^total_pairs → 
    (total_ways - no_pair_ways) = 54 :=
by
  intros
  sorry

end gloves_selection_l665_665062


namespace common_divisors_35_64_l665_665604

theorem common_divisors_35_64 : 
  let divisors_35 := {d : ℕ | d ∣ 35}
  let divisors_64 := {d : ℕ | d ∣ 64}
  finset.card (finset.filter (λ x, x ∈ divisors_35) (finset.filter (λ y, y ∈ divisors_64) (finset.range (min 35 64 + 1)))) = 1 :=
by
  sorry

end common_divisors_35_64_l665_665604


namespace cube_volume_from_surface_area_example_cube_volume_l665_665973

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l665_665973


namespace inequality_for_positive_integers_l665_665751

theorem inequality_for_positive_integers 
  (a b : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h₂ : 1/a + 1/b = 1)
  (n : ℕ)
  (hn : n > 0) : 
  (a + b) ^ n - a ^ n - b ^ n ≥ 2^(2*n) - 2^(n + 1) :=
sorry

end inequality_for_positive_integers_l665_665751


namespace cube_volume_is_1728_l665_665945

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l665_665945


namespace length_MN_in_trapezoid_l665_665726

noncomputable def midpoint (x y : ℝ) : ℝ := (x + y) / 2

theorem length_MN_in_trapezoid
  (A B C D M N : Point)
  (BC AD : Line)
  (BC_parallel_AD : parallel BC AD)
  (BC_length : length BC = 1000)
  (AD_length : length AD = 2008)
  (angle_A_value : angle A = 37)
  (angle_D_value : angle D = 53)
  (M_midpoint_BC : midpoint M = midpoint B C)
  (N_midpoint_AD : midpoint N = midpoint A D):
  length (Segment M N) = 504 :=
begin
  sorry
end

end length_MN_in_trapezoid_l665_665726


namespace num_valid_integers_l665_665317

theorem num_valid_integers : 
  ({m : ℤ | m ≠ -4 ∧ m ≠ 4 ∧ 1 / |m| ≥ 1 / 5}.finite ∧ 
  ({m : ℤ | m ≠ -4 ∧ m ≠ 4 ∧ 1 / |m| ≥ 1 / 5}.toFinset.card = 8)) := 
by
  sorry

end num_valid_integers_l665_665317


namespace coat_price_proof_l665_665532

variable (W : ℝ) -- wholesale price
variable (currentPrice : ℝ) -- current price of the coat

-- Condition 1: The retailer marked up the coat by 90%.
def markup_90 : Prop := currentPrice = 1.9 * W

-- Condition 2: Further $4 increase achieves a 100% markup.
def increase_4 : Prop := 2 * W - currentPrice = 4

-- Theorem: The current price of the coat is $76.
theorem coat_price_proof (h1 : markup_90 W currentPrice) (h2 : increase_4 W currentPrice) : currentPrice = 76 :=
sorry

end coat_price_proof_l665_665532


namespace balls_in_boxes_l665_665319

theorem balls_in_boxes :
  (∃ x1 x2 x3 : ℕ, x1 + x2 + x3 = 7) →
  (multichoose 7 3) = 36 :=
by
  sorry

end balls_in_boxes_l665_665319


namespace value_of_x_l665_665686

theorem value_of_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end value_of_x_l665_665686


namespace simplify_expression_l665_665017

theorem simplify_expression (y : ℝ) : 5 * y + 7 * y + 8 * y = 20 * y :=
by
  sorry

end simplify_expression_l665_665017


namespace steps_to_MSG_l665_665559

theorem steps_to_MSG (steps_down total_steps : ℕ) (h₁ : steps_down = 676) (h₂ : total_steps = 991) :
  total_steps - steps_down = 315 := by
  rw [h₁, h₂]
  norm_num

end steps_to_MSG_l665_665559


namespace num_possible_lists_l665_665078

theorem num_possible_lists :
  let binA_balls := 8
  let binB_balls := 5
  let total_lists := binA_balls * binB_balls
  total_lists = 40 := by
{
  let binA_balls := 8
  let binB_balls := 5
  let total_lists := binA_balls * binB_balls
  show total_lists = 40
  exact rfl
}

end num_possible_lists_l665_665078


namespace amaya_total_time_l665_665173

-- Define the times as per the conditions
def first_segment : Nat := 35 + 5
def second_segment : Nat := 45 + 15
def third_segment : Nat := 20

-- Define the total time by summing up all segments
def total_time : Nat := first_segment + second_segment + third_segment

-- The theorem to prove
theorem amaya_total_time : total_time = 120 := by
  -- Let's explicitly state the expected result here
  have h1 : first_segment = 40 := rfl
  have h2 : second_segment = 60 := rfl
  have h3 : third_segment = 20 := rfl
  have h_sum : total_time = 40 + 60 + 20 := by
    rw [h1, h2, h3]
  simp [total_time, h_sum]
  -- Finally, the result is 120
  exact rfl

end amaya_total_time_l665_665173


namespace existence_of_f_and_g_l665_665001

noncomputable def Set_n (n : ℕ) : Set ℕ := { x | x ≥ 1 ∧ x ≤ n }

theorem existence_of_f_and_g (n : ℕ) (f g : ℕ → ℕ) :
  (∀ x ∈ Set_n n, (f (g x) = x ∨ g (f x) = x) ∧ ¬(f (g x) = x ∧ g (f x) = x)) ↔ Even n := sorry

end existence_of_f_and_g_l665_665001


namespace cube_volume_from_surface_area_l665_665879

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l665_665879


namespace find_line_equation_l665_665691

noncomputable def equation_of_perpendicular_line : Prop := 
  ∃ (l : ℝ → ℝ) (x y : ℝ), 
    (l x = 4*x/3 - 17/3) ∧ 
    (x = -2 ∧ y = -3) ∧ 
    (3*x + 4*y - 3 = 0)

theorem find_line_equation (A : ℝ × ℝ) (B : ℝ → Prop) :
    A = (-2, -3) → 
    (∀ x y : ℝ, B (3*x + 4*y - 3 = 0)) → 
     ∃ (a b c : ℝ), 4*a - 3*b - c = 0 :=
by 
    sorry

end find_line_equation_l665_665691


namespace cube_volume_from_surface_area_l665_665884

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l665_665884


namespace intersection_points_concyclic_l665_665080

-- Define the two parabolas in terms of their equations
def parabola1 (x y : ℝ) (p : ℝ) : Prop := x^2 + p * y = 0
def parabola2 (x y : ℝ) (y0 a b : ℝ) : Prop := (y - y0)^2 + a * x + b = 0

-- Statement: The four points of intersection of these parabolas lie on a single circle
theorem intersection_points_concyclic {p y0 a b : ℝ} :
  ∃ points : set (ℝ × ℝ),
    points.card = 4 ∧
    ∀ (x y : ℝ), 
      (parabola1 x y p ∧ parabola2 x y y0 a b) ↔ (x, y) ∈ points
    → ∃ center : ℝ × ℝ, ∃ radius : ℝ,
      ∀ (x y : ℝ), (x, y) ∈ points → (x - center.1)^2 + (y - center.2)^2 = radius^2 := by
  sorry

end intersection_points_concyclic_l665_665080


namespace sqrt_expression_meaningful_iff_l665_665334

theorem sqrt_expression_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = sqrt (2 * x - 4)) ↔ x ≥ 2 :=
by sorry

end sqrt_expression_meaningful_iff_l665_665334


namespace cube_volume_of_surface_area_l665_665977

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l665_665977


namespace hexagonal_grid_property_l665_665177

-- Definition of a triangular lattice and properties
def triangular_lattice : Type := Σ (m n : ℤ), { p : Prop // p = True }

-- Predicate to check if a midpoint of two vertices in a hexagonal grid is also a vertex in that grid
def midpoint_in_grid (v1 v2 : triangular_lattice) : Prop :=
  let ⟨m1, n1, _⟩ := v1
  let ⟨m2, n2, _⟩ := v2
  let midpoint := (((m1 + m2) / 2, (n1 + n2) / 2) : ℤ × ℤ)
  ∃ (v_mid : triangular_lattice), 
    (v_mid = ⟨(midpoint.1), (midpoint.2), trivial⟩)

-- Main proof statement
theorem hexagonal_grid_property : ∀ (vertices : Set triangular_lattice), 
  (∃ (n : ℕ), n = 9) → 
  (∃ (v1 v2 : vertices), midpoint_in_grid v1 v2) := 
begin
  sorry
end

end hexagonal_grid_property_l665_665177


namespace exists_infinite_set_no_square_factors_l665_665209

open Nat

theorem exists_infinite_set_no_square_factors :
  ∃ (M : Set ℕ), (Infinite M) ∧
  (∀ a b ∈ M, a < b → ∀ k : ℕ, k > 1 → k^2 ∤ (a + b)) := 
sorry

end exists_infinite_set_no_square_factors_l665_665209


namespace polynomial_roots_l665_665243

theorem polynomial_roots :
  (∀ x, x^3 - 3 * x^2 - x + 3 = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) := 
by
  sorry

end polynomial_roots_l665_665243


namespace tricycles_count_l665_665609

theorem tricycles_count (cars bicycles pickup_trucks tricycles : ℕ) (total_tires : ℕ) : 
  cars = 15 →
  bicycles = 3 →
  pickup_trucks = 8 →
  total_tires = 101 →
  4 * cars + 2 * bicycles + 4 * pickup_trucks + 3 * tricycles = total_tires →
  tricycles = 1 :=
by
  sorry

end tricycles_count_l665_665609


namespace total_cost_is_13_l665_665140

-- Definition of pencil cost
def pencil_cost : ℕ := 2

-- Definition of pen cost based on pencil cost
def pen_cost : ℕ := pencil_cost + 9

-- The total cost of both items
def total_cost := pencil_cost + pen_cost

theorem total_cost_is_13 : total_cost = 13 := by
  sorry

end total_cost_is_13_l665_665140


namespace cube_volume_is_1728_l665_665940

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l665_665940


namespace segment_PT_length_l665_665020

/-- Definition of a square with side length 4 -/
def square_side_length : ℝ := 4

/-- Segments PT and PU divide the square into four equal parts -/
def area_division : ℝ := (square_side_length ^ 2) / 4

/-- Theorem: Prove the length of segment PT -/
theorem segment_PT_length (a : ℝ) (h1 : square_side_length = 4) 
  (h2 : (area_division = 4)) : 
  a = Real.sqrt 20 :=
sorry

end segment_PT_length_l665_665020


namespace set_infinite_l665_665387

-- Define the conditions
variable (E : Set (ℝ × ℝ)) 
variable (H : ∀ x ∈ E, ∃ y z ∈ E, x = ((y.1 + z.1) / 2, (y.2 + z.2) / 2))

-- Define the proof statement
theorem set_infinite (E : Set (ℝ × ℝ)) (H : ∀ x ∈ E, ∃ y z ∈ E, x = ((y.1 + z.1) / 2, (y.2 + z.2) / 2)) : 
  Infinite E :=
sorry

end set_infinite_l665_665387


namespace smallest_four_digit_divisible_43_l665_665092

theorem smallest_four_digit_divisible_43 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 43 = 0 ∧ n = 1032 :=
by
  sorry

end smallest_four_digit_divisible_43_l665_665092


namespace gcd_correct_l665_665580

def gcd_765432_654321 : ℕ :=
  Nat.gcd 765432 654321

theorem gcd_correct : gcd_765432_654321 = 6 :=
by sorry

end gcd_correct_l665_665580


namespace spring_percentage_decrease_l665_665153

theorem spring_percentage_decrease 
  (X : ℝ)
  (fall_increase : X * 1.05)
  (total_change_decrease : X * 0.8505)
  (P : ℝ) :
  1 - P = 0.8505 / 1.05 → P = 0.19 :=
by
  intro h
  field_simp at h
  rw [sub_eq_iff_eq_add, ←eq_div_iff] at h
  linarith

end spring_percentage_decrease_l665_665153


namespace age_difference_l665_665839

variable (A B C X : ℕ)

theorem age_difference 
  (h1 : C = A - 13)
  (h2 : A + B = B + C + X) 
  : X = 13 :=
by
  sorry

end age_difference_l665_665839


namespace max_at_one_iff_l665_665272

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * (Real.log x) - a * x ^ 2 + (2 * a - 1) * x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := deriv (fun x => f x a) x

def g_monotonic_intervals (a : ℝ) : Set (ℝ × ℝ) :=
  if a ≤ 0 then
    {(0, +∞)}
  else
    {(0, 1 / (2 * a)), (1 / (2 * a), +∞)}

theorem max_at_one_iff (a : ℝ) :
  (deriv (fun x => f x a) 1 = 0) → (f (1 : ℝ) a = Real.sup (Set.range (fun x => f x a))) ↔ (a > 0.5) :=
sorry

end max_at_one_iff_l665_665272


namespace cube_volume_of_surface_area_l665_665978

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l665_665978


namespace diameter_other_endpoint_l665_665570

def center : ℝ × ℝ := (1, -2)
def endpoint1 : ℝ × ℝ := (4, 3)
def expected_endpoint2 : ℝ × ℝ := (7, -7)

theorem diameter_other_endpoint (c : ℝ × ℝ) (e1 e2 : ℝ × ℝ) (h₁ : c = center) (h₂ : e1 = endpoint1) : e2 = expected_endpoint2 :=
by
  sorry

end diameter_other_endpoint_l665_665570


namespace packs_per_box_is_eight_l665_665737

-- Defining the conditions
def boxes_sold : ℕ := 24
def packs_sold : ℕ := 192

-- Stating the theorem
theorem packs_per_box_is_eight : packs_sold / boxes_sold = 8 :=
by 
  -- Using the definition of the numbers
  have h_boxes : boxes_sold = 24 := rfl
  have h_packs : packs_sold = 192 := rfl
  
  -- Direct calculation showing the result
  rw [h_boxes, h_packs]
  norm_num
  sorry

end packs_per_box_is_eight_l665_665737


namespace find_value_a_g_is_odd_g_is_monotonic_decreasing_g_range_l665_665303

noncomputable def f (x : ℝ) := 3 ^ x
noncomputable def g (a : ℝ) (x : ℝ) := (1 - a ^ x) / (1 + a ^ x)

theorem find_value_a (a x : ℝ) (h : f (a + 2) = 81) : a = 2 := sorry

theorem g_is_odd (a : ℝ) (h : a > 1) : ∀ x : ℝ, g a (-x) = -g a x := sorry

theorem g_is_monotonic_decreasing (a : ℝ) (h : a > 1) : ∀ x1 x2 : ℝ, x1 < x2 → g a x1 > g a x2 := sorry

theorem g_range (a : ℝ) (h : a > 1) : set.range (g a) = set.Ioo (-1) 1 := sorry

end find_value_a_g_is_odd_g_is_monotonic_decreasing_g_range_l665_665303


namespace cube_volume_from_surface_area_l665_665881

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l665_665881


namespace age_difference_is_13_l665_665836

variables (A B C X : ℕ)
variables (total_age_A_B total_age_B_C : ℕ)

-- Conditions
def condition1 : Prop := total_age_A_B = total_age_B_C + X
def condition2 : Prop := C = A - 13

-- Theorem statement
theorem age_difference_is_13 (h1: condition1 total_age_A_B total_age_B_C X)
                             (h2: condition2 A C) :
  X = 13 :=
sorry

end age_difference_is_13_l665_665836


namespace sum_3000_l665_665059

-- Definitions for the conditions
def geom_sum (a r : ℝ) (n : ℕ) := a * (1 - r^n) / (1 - r)

variables (a r : ℝ)
axiom sum_1000 : geom_sum a r 1000 = 300
axiom sum_2000 : geom_sum a r 2000 = 570

-- The property to prove
theorem sum_3000 : geom_sum a r 3000 = 813 :=
sorry

end sum_3000_l665_665059


namespace min_Q_value_l665_665267

def Q (k : ℤ) : ℚ := sorry
  
theorem min_Q_value : ∀ (k : ℤ), odd k → (1 ≤ k ∧ k ≤ 199) → Q k = 75 / 101 :=
by
  sorry

end min_Q_value_l665_665267


namespace ratio_a_b_l665_665021

theorem ratio_a_b (a b : ℝ) (h : ∫ x in 0..2, -a * x^2 + b = 0) : a / b = 3 / 4 :=
by
  sorry

end ratio_a_b_l665_665021


namespace cube_volume_of_surface_area_l665_665980

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l665_665980


namespace coefficient_x3_eq_binom_51_4_l665_665746

theorem coefficient_x3_eq_binom_51_4 :
  let f (x : ℕ) := (1 + x)^3 + (1 + x)^4 + ⋯ + (1 + x)^50,
      p (x : ℕ) := ∑ k in finset.range (50 - 3 + 1), (k + 3).choose 3
  in p (3) = (51).choose 4 :=
by sorry

end coefficient_x3_eq_binom_51_4_l665_665746


namespace jameson_badminton_medals_l665_665367

theorem jameson_badminton_medals:
  ∀ (total track: ℕ) (swimming_multiple: ℕ),
  total = 20 → track = 5 → swimming_multiple = 2 →
  ∃ (badminton: ℕ), badminton = 20 - (track + swimming_multiple * track) ∧ badminton = 5 :=
by
  intros total track swimming_multiple ht ht5 hsm
  use 5
  simp [ht5, hsm, ht]
  sorry

end jameson_badminton_medals_l665_665367


namespace x_value_for_perfect_square_l665_665262

theorem x_value_for_perfect_square (a b c x : ℝ) :
  (a + b - x)^2 + (b + c - x)^2 + (c + a - x)^2 = (a + b + c - 2*x)^2 ↔
  x = (a + b + c) / 2 :=
begin
  sorry
end

end x_value_for_perfect_square_l665_665262


namespace problem_statement_l665_665328

/-- Define n as the largest positive integer such that n^2 < 2018. -/
def n : ℕ := 44

/-- Define m as the smallest positive integer such that 2018 < m^2. -/
def m : ℕ := 45

/-- Proof that m^2 - n^2 = 89 given the conditions on m and n. -/
theorem problem_statement : m^2 - n^2 = 89 := 
by
  rw [m, n]
  exact Nat.sub_self sorry

end problem_statement_l665_665328


namespace sin_inequality_solution_set_l665_665355

theorem sin_inequality_solution_set : 
  {x : ℝ | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin x < - Real.sqrt 3 / 2} =
  {x : ℝ | (4 * Real.pi / 3) < x ∧ x < (5 * Real.pi / 3)} := by
  sorry

end sin_inequality_solution_set_l665_665355


namespace cube_volume_of_surface_area_l665_665981

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l665_665981


namespace opposite_of_abs_neg_pi_l665_665826

theorem opposite_of_abs_neg_pi : (|(-real.pi)| = real.pi) → -(real.pi) = -real.pi :=
by {
  intro h,
  exact congr_arg has_neg.neg h,
}

end opposite_of_abs_neg_pi_l665_665826


namespace log_eq_neg_third_l665_665207

theorem log_eq_neg_third {x : ℝ} (h : log 64 (3 * x + 2) = -1 / 3) : x = -7 / 12 :=
by
  sorry

end log_eq_neg_third_l665_665207


namespace total_cats_reception_l665_665740

/-- Definitions of the conditions in the problem. -/
def total_adults := 120
def percentage_female := 0.60
def kittens_per_litter := 3
def percentage_with_kittens := 0.40

/-- Proof that the total number of cats, including kittens, received by the shelter is 207. -/
theorem total_cats_reception :
  (total_adults + Nat.floor ((total_adults * percentage_female) * percentage_with_kittens * kittens_per_litter) ) = 207 := by
  sorry

end total_cats_reception_l665_665740


namespace optimal_import_quantity_l665_665120

-- Define the conditions
variables (annual_volume : ℕ) (shipping_cost : ℕ) (rent_cost_per_unit : ℕ) 
          (num_imports : ℕ) (import_quantity : ℕ)

def total_cost (import_quantity : ℕ) : ℕ :=
  let num_imports := annual_volume / import_quantity in
  let shipping_cost_total := num_imports * shipping_cost in
  let rent_cost_total := (import_quantity / 2) * rent_cost_per_unit * num_imports in
  shipping_cost_total + rent_cost_total

theorem optimal_import_quantity (annual_volume : ℕ) (shipping_cost : ℕ) (rent_cost_per_unit : ℕ) :
  annual_volume = 10000 ∧ shipping_cost = 100 ∧ rent_cost_per_unit = 2 → import_quantity = 1000 :=
by
  sorry

end optimal_import_quantity_l665_665120


namespace cube_volume_from_surface_area_l665_665875

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l665_665875


namespace quadrilateral_false_statement_l665_665993

-- Definitions for quadrilateral properties
def is_rhombus (q : ℝ × ℝ × ℝ × ℝ) : Prop := q.1 = q.2 ∧ q.3 = q.4
def equal_diagonals (d1 d2 : ℝ) : Prop := d1 = d2
def is_rectangle (q : ℝ × ℝ × ℝ × ℝ) : Prop := q.1 = q.2 ∧ q.3 = q.4 ∧ q.1 = 90 ∧ q.3 = 90
def perpendicular (a b : ℝ) : Prop := a * b = 0
def is_parallelogram (q : ℝ × ℝ × ℝ × ℝ) : Prop := q.1 = q.3 ∧ q.2 = q.4
def bisects (d1 d2 : ℝ) : Prop := d1 = d2 / 2

-- The problem statement
theorem quadrilateral_false_statement :
  ¬ (∀ (q : ℝ × ℝ × ℝ × ℝ) (d1 d2 : ℝ),
    (is_rhombus q ∧ equal_diagonals d1 d2 → q.1 = 90 ∧ q.2 = 90) ∧
    (is_rectangle q ∧ perpendicular d1 d2 → q.1 = q.2) ∧
    (is_parallelogram q ∧ perpendicular d1 d2 ∧ equal_diagonals d1 d2 → q.1 = 90 ∧ q.2 = 90) ∧
    (perpendicular d1 d2 ∧ bisects d1 d2 → q.1 = 90 ∧ q.2 = 90)) :=
sorry

end quadrilateral_false_statement_l665_665993


namespace number_of_assignments_l665_665850

-- Define the conditions: number of doctors and nurses.
def doctors := {1, 2}  -- Two doctors, represented as a set containing 1 and 2.
def nurses := {1, 2, 3, 4}  -- Four nurses, represented as a set containing 1, 2, 3, and 4.

-- Define the statement to be proved.
theorem number_of_assignments :
  ∃ (num_ways : ℕ), num_ways = 12 :=
by
  -- Skipping the proof here
  sorry

end number_of_assignments_l665_665850


namespace trigonometric_proof_l665_665192

noncomputable def cos30 : ℝ := Real.sqrt 3 / 2
noncomputable def tan60 : ℝ := Real.sqrt 3
noncomputable def sin45 : ℝ := Real.sqrt 2 / 2
noncomputable def cos45 : ℝ := Real.sqrt 2 / 2

theorem trigonometric_proof :
  2 * cos30 - tan60 + sin45 * cos45 = 1 / 2 :=
by
  sorry

end trigonometric_proof_l665_665192


namespace exists_member_with_special_property_l665_665405

theorem exists_member_with_special_property (M : Finset ℕ) (M_1 M_2 M_3 M_4 M_5 M_6: Finset ℕ) 
  (h_union: M_1 ∪ M_2 ∪ M_3 ∪ M_4 ∪ M_5 ∪ M_6 = M)
  (h_card: M.card = 1978)
  (h_disjoint: ∀ i j, 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6 ∧ i ≠ j → Disjoint (Finset.nth i [M_1, M_2, M_3, M_4, M_5, M_6]) (Finset.nth j [M_1, M_2, M_3, M_4, M_5, M_6])) :
  ∃ m₀ ∈ M, (∃ i, 1 ≤ i ∧ i ≤ 6 ∧ ∃ m₁ m₂ ∈ (Finset.nth i [M_1, M_2, M_3, M_4, M_5, M_6]), m₀ = m₁ + m₂ ∧ m₁ ≠ m₂) 
  ∨ (∃ i, 1 ≤ i ∧ i ≤ 6 ∧ ∃ m ∈ (Finset.nth i [M_1, M_2, M_3, M_4, M_5, M_6]), m₀ = 2 * m)
  :=
sorry

end exists_member_with_special_property_l665_665405


namespace trigonometric_identity_l665_665659

noncomputable def f (α : ℝ) : ℝ :=
  (sin (7 * Real.pi - α) * cos (α + 3 * Real.pi / 2) * cos (3 * Real.pi + α)) /
  (sin (α - 3 * Real.pi / 2) * cos (α + 5 * Real.pi / 2) * tan (α - 5 * Real.pi))

theorem trigonometric_identity (α : ℝ)
  (hα_quad : π / 2 < α ∧ α < π) -- α is in the second quadrant
  (hα_cond : cos (3 * Real.pi / 2 + α) = 1 / 7) :
  f α = - (4 * Real.sqrt 3) / 7 :=
sorry

end trigonometric_identity_l665_665659


namespace determinant_of_triangle_angles_zero_l665_665761

noncomputable def prove_det (
  A B C : ℝ
) (h : A + B + C = π) : 
  matrix ℝ (fin 3) (fin 3) := 
  ![
    ![cos A ^ 2, tan A, 1],
    ![cos B ^ 2, tan B, 1],
    ![cos C ^ 2, tan C, 1]
  ]

theorem determinant_of_triangle_angles_zero (A B C : ℝ) (h : A + B + C = π) :
  matrix.det (prove_det A B C h) = 0 := 
sorry

end determinant_of_triangle_angles_zero_l665_665761


namespace cube_volume_of_surface_area_l665_665982

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l665_665982


namespace cube_volume_l665_665927

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l665_665927


namespace cube_volume_l665_665901

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l665_665901


namespace cube_volume_from_surface_area_l665_665916

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l665_665916


namespace find_polynomial_l665_665397

noncomputable def polynomial_p (n : ℕ) (P : ℝ → ℝ → ℝ) :=
  ∀ t x y a b c : ℝ,
    (P (t * x) (t * y) = t ^ n * P x y) ∧
    (P (a + b) c + P (b + c) a + P (c + a) b = 0) ∧
    (P 1 0 = 1)

theorem find_polynomial (n : ℕ) (P : ℝ → ℝ → ℝ) (h : polynomial_p n P) :
  ∀ x y : ℝ, P x y = x^n - y^n :=
sorry

end find_polynomial_l665_665397


namespace variance_of_scores_l665_665546

def scores : List ℕ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]

def mean (xs : List ℕ) : ℚ := xs.sum / xs.length

def variance (xs : List ℕ) : ℚ :=
  let m := mean xs
  (xs.map (λ x => (x - m)^2)).sum / xs.length

theorem variance_of_scores : variance scores = 4 := by
  sorry

end variance_of_scores_l665_665546


namespace central_angle_part1_central_angle_chord_l665_665455

-- Define the problem statements
noncomputable def part1 (r l α : ℝ) : Prop :=
  (2 * r + l = 8) ∧ ((1 / 2) * l * r = 3) ∧ (α = l / r)

-- Define the proof goal for part 1
theorem central_angle_part1 : ∃ (α : ℝ), part1 1 6 α ∨ part1 3 2 α :=
sorry

-- Define the problem statements for part 2
noncomputable def part2 (r l α : ℝ) : Prop :=
  (2 * r + l = 8) ∧ (l / r = 2) ∧ (r = 2) ∧ (l = 4) ∧ (α = l / r) ∧ (l > 0) ∧ (α ≥ 0)

-- Define the proof goal for part 2
theorem central_angle_chord : ∃ (α : ℝ) (AB : ℝ), part2 2 4 α ∧ AB = 4 * real.sin 1 :=
sorry

end central_angle_part1_central_angle_chord_l665_665455


namespace amelia_wins_l665_665176

noncomputable def ameliaWinsProbability (a b : ℚ) : ℚ :=
  let heads_heads := a * b
  let amelia_wins_first := a * (1 - b)
  let blaine_wins_first := (1 - a) * b
  let both_tails := (1 - a) * (1 - b)
  let geometric_sum := (both_tails : ℚ) / (1 - both_tails)
  let amelia_wins_subsequent := (1 - a) * geometric_sum * a
  amelia_wins_first + amelia_wins_subsequent

theorem amelia_wins (Ha : 3 / 7) (Hb : 1 / 4) :
  ameliaWinsProbability (3/7) (1/4) = 9 / 14 :=
by
  sorry

end amelia_wins_l665_665176


namespace determinant_cos_tan_l665_665760

theorem determinant_cos_tan {A B C : ℝ} (h : A + B + C = π) :
  Matrix.det !![!![cos A ^ 2, tan A, 1],
                !![cos B ^ 2, tan B, 1],
                !![cos C ^ 2, tan C, 1]] = 0 :=
by sorry

end determinant_cos_tan_l665_665760


namespace relationship_f_neg2_f_expr_l665_665558

noncomputable def f : ℝ → ℝ := sorry  -- f is some function ℝ → ℝ, the exact definition is not provided

axiom even_function : ∀ x : ℝ, f (-x) = f x -- f is an even function
axiom increasing_on_negatives : ∀ x y : ℝ, x < y ∧ y < 0 → f x < f y -- f is increasing on (-∞, 0)

theorem relationship_f_neg2_f_expr (a : ℝ) : f (-2) ≥ f (a^2 - 4 * a + 6) := by
  -- proof omitted
  sorry

end relationship_f_neg2_f_expr_l665_665558


namespace Celine_change_l665_665147

def laptop_base_price := 600
def smartphone_base_price := 400
def tablet_base_price := 250
def headphone_base_price := 100

def laptop_discount_rate := 0.15
def smartphone_increase_rate := 0.10
def tablet_discount_rate := 0.20
def sales_tax_rate := 0.06

def laptops_bought := 2
def smartphones_bought := 3
def tablets_bought := 4
def headphones_bought := 6

def total_money_available := 6000

noncomputable def laptop_final_price := laptop_base_price * (1 - laptop_discount_rate)
noncomputable def smartphone_final_price := smartphone_base_price * (1 + smartphone_increase_rate)
noncomputable def tablet_final_price := tablet_base_price * (1 - tablet_discount_rate)
noncomputable def headphone_final_price := headphone_base_price

noncomputable def total_cost_before_tax := 
  laptops_bought * laptop_final_price + 
  smartphones_bought * smartphone_final_price + 
  tablets_bought * tablet_final_price + 
  headphones_bought * headphone_final_price

noncomputable def sales_tax := total_cost_before_tax * sales_tax_rate
noncomputable def total_cost_after_tax := total_cost_before_tax + sales_tax
noncomputable def change := total_money_available - total_cost_after_tax

theorem Celine_change : change = 2035.60 := by
  sorry

end Celine_change_l665_665147


namespace transformation_proof_l665_665574

theorem transformation_proof
  (S : ℝ^3 → ℝ^3)
  (h_lin : ∀ (a b : ℝ) (u v : ℝ^3), S (a • u + b • v) = a • S u + b • S v)
  (h_cross : ∀ (u v : ℝ^3), S (u ×ᵥ v) = (S u) ×ᵥ (S v))
  (h1 : S ⟨4, 8, 2⟩ = ⟨3, 0, 9⟩)
  (h2 : S ⟨-4, 2, 8⟩ = ⟨3, 9, 0⟩) :
  S ⟨2, 10, 16⟩ = ⟨12.9, 9.9, 9.9⟩ :=
sorry

end transformation_proof_l665_665574


namespace matilda_jellybeans_l665_665401

/-- Suppose Matilda has half as many jellybeans as Matt.
    Suppose Matt has ten times as many jellybeans as Steve.
    Suppose Steve has 84 jellybeans.
    Then Matilda has 420 jellybeans. -/
theorem matilda_jellybeans
    (matilda_jellybeans : ℕ)
    (matt_jellybeans : ℕ)
    (steve_jellybeans : ℕ)
    (h1 : matilda_jellybeans = matt_jellybeans / 2)
    (h2 : matt_jellybeans = 10 * steve_jellybeans)
    (h3 : steve_jellybeans = 84) : matilda_jellybeans = 420 := 
sorry

end matilda_jellybeans_l665_665401


namespace angle_bisector_altitude_l665_665077

theorem angle_bisector_altitude (A B C D E : Type)
  (angle_A : A = 37)
  (angle_B : B = 75)
  (angle_C : C = 180 - 37 - 75)
  (angle_bisector : D = C / 2)
  (altitude : E = 90)
  : D = 34 :=
by
  -- Definitions of angles and properties of the triangle
  let angle_A := 37
  let angle_B := 75
  let angle_C := 68 -- Third angle calculated as 180 - 37 - 75
  let angle_bisector := 34 -- Angle bisector of angle C which is 68/2
  let altitude := 90
  sorry

end angle_bisector_altitude_l665_665077


namespace octagon_area_inscribed_in_circle_l665_665531

theorem octagon_area_inscribed_in_circle (R : ℝ) (hR : R = 4) :
  let θ := pi / 4
  let s := 2 * R * sin (θ / 2)
  let triangle_area := (1 / 2) * R^2 * sin θ
  let octagon_area := 8 * triangle_area
  octagon_area = 64 * real.sqrt 2 :=
by {
  sorry
}

end octagon_area_inscribed_in_circle_l665_665531


namespace factorial_divisibility_l665_665687

theorem factorial_divisibility :
  ∃ x : ℕ, x = 25 ∧ (10! - 2 * (nat.factorial 25)^2) % 10^5 = 0 :=
by
  exists 25
  split
  { rfl }
  { sorry }

end factorial_divisibility_l665_665687


namespace home_library_capacity_l665_665739

variables (books_owned books_to_buy : ℕ) (full_percent total_capacity : ℝ)

def total_books (books_owned books_to_buy : ℕ) : ℕ :=
  books_owned + books_to_buy

def full_capacity (total_books full_percent : ℝ) : ℝ :=
  total_books / full_percent

theorem home_library_capacity
  (h_books_owned : books_owned = 120)
  (h_books_to_buy : books_to_buy = 240)
  (h_full_percent : full_percent = 0.90)
  (h_total_capacity : total_capacity = full_capacity (total_books books_owned books_to_buy) full_percent) :
  total_capacity = 400 :=
by
  sorry

end home_library_capacity_l665_665739


namespace cube_volume_l665_665896

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l665_665896


namespace first_player_always_wins_when_m_gt_2n_first_player_always_wins_when_m_gt_alpha_n_l665_665065

-- Definition of winning strategy
def win_position (m n : ℕ) : Prop :=
  ∃ p q : ℕ, m = p * n + q ∧ q < n

-- The first proof problem
theorem first_player_always_wins_when_m_gt_2n (m n : ℕ) (h1 : m > n) (h2 : m > 2 * n) :
  win_position m n := sorry

-- Definition of Golden ratio
noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

-- The second proof problem
theorem first_player_always_wins_when_m_gt_alpha_n (m n : ℕ) (h1 : m > n) (α : ℝ) (h2 : α >= golden_ratio) (h3 : m > α * n) :
  win_position m n := sorry

end first_player_always_wins_when_m_gt_2n_first_player_always_wins_when_m_gt_alpha_n_l665_665065


namespace palindrome_digital_clock_l665_665855

theorem palindrome_digital_clock (no_leading_zero : ∀ h : ℕ, h < 10 → ¬ ∃ h₂ : ℕ, h₂ = h * 1000)
                                 (max_hour : ∀ h : ℕ, h ≥ 24 → false) :
  ∃ n : ℕ, n = 61 := by
  sorry

end palindrome_digital_clock_l665_665855


namespace cost_of_bag_l665_665416

variable (cost_per_bag : ℝ)
variable (chips_per_bag : ℕ := 24)
variable (calories_per_chip : ℕ := 10)
variable (total_calories : ℕ := 480)
variable (total_cost : ℝ := 4)

theorem cost_of_bag :
  (chips_per_bag * (total_calories / calories_per_chip / chips_per_bag) = (total_calories / calories_per_chip)) →
  (total_cost / (total_calories / (calories_per_chip * chips_per_bag))) = 2 :=
by
  sorry

end cost_of_bag_l665_665416


namespace train_stops_per_hour_l665_665217

-- Define the speeds of the train
def speed_without_stoppages : ℝ := 54
def speed_with_stoppages : ℝ := 36

-- Calculate the reduction in speed due to stoppages
def speed_reduction : ℝ := speed_without_stoppages - speed_with_stoppages

-- The original speed must be converted to km per minute
def speed_without_stoppages_per_minute : ℝ := speed_without_stoppages / 60

-- Calculation for the number of minutes the train stops per hour
theorem train_stops_per_hour : (speed_reduction / speed_without_stoppages_per_minute) = 20 := by
  sorry

end train_stops_per_hour_l665_665217


namespace not_necessary_property_of_similar_triangles_l665_665987

theorem not_necessary_property_of_similar_triangles :
  ¬(∀ (△ ABC △ DEF: Triangle), 
  (Corresponding_angles_are_equal △ ABC △ DEF) ∧ 
  (Corresponding_sides_are_proportional △ ABC △ DEF) ∧ 
  (Ratio_of_corresponding_heights_equal_ratio_of_similarity △ ABC △ DEF) → 
  (Corresponding_sides_are_equal △ ABC △ DEF)) :=
sorry

end not_necessary_property_of_similar_triangles_l665_665987


namespace length_of_common_external_tangent_l665_665469

-- Define the radii of the two circles
def r1 : ℕ := 16
def r2 : ℕ := 25

-- The sum of the radii of the two circles
def dist_centers : ℕ := r1 + r2

-- The difference of the radii of the two circles
def diff_radii : ℕ := abs (r2 - r1)

-- The square of the distance between the centers
def dist_centers_sq : ℕ := dist_centers ^ 2

-- The square of the difference of the radii
def diff_radii_sq : ℕ := diff_radii ^ 2

-- Applying Pythagorean theorem to find the square of the length of the common external tangent
def tangent_sq : ℕ := dist_centers_sq - diff_radii_sq

-- Prove that the length of the common external tangent
theorem length_of_common_external_tangent : sqrt tangent_sq = 40 := by
  sorry

end length_of_common_external_tangent_l665_665469


namespace men_in_first_group_l665_665331

/-
  Let x be the number of men in the first group.
  Given:
  1. x men can reap 120 acres in 36 days.
  2. 24 men can reap 413.33333333333337 acres in 62 days.

  Prove that x = 12.
-/

theorem men_in_first_group (x : ℕ) :
  (x * 36 / 120) = (24 * 62 / 413.33333333333337) → 
  x = 12 :=
by
  sorry

end men_in_first_group_l665_665331


namespace nine_odot_three_l665_665825

-- Definition of operation ⊙
def op (a b : ℝ) : ℝ := a + (3 * a^2) / (2 * b)

-- Statement of the problem
theorem nine_odot_three : op 9 3 = 49.5 := 
  sorry

end nine_odot_three_l665_665825


namespace find_u_l665_665061

open Real

variables (a b r : ℝ)

def vec_a : ℝ × ℝ × ℝ := (4, 2, -3)
def vec_b : ℝ × ℝ × ℝ := (1, 3, -2)
def vec_r : ℝ × ℝ × ℝ := (5, 2, -7)
def cross_product (u : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

def dot_product (u : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem find_u :
  ∃ (s t u : ℝ), vec_r = s * (4, 2, -3) + t * (1, 3, -2) + u * cross_product (4, 2, -3) (1, 3, -2) ∧
  u = -11 / 10 :=
sorry

end find_u_l665_665061


namespace quadratic_function_passes_through_origin_l665_665447

theorem quadratic_function_passes_through_origin (a : ℝ) :
  ((a - 1) * 0^2 - 0 + a^2 - 1 = 0) → a = -1 :=
by
  intros h
  sorry

end quadratic_function_passes_through_origin_l665_665447


namespace sqrt_meaningful_iff_l665_665336

theorem sqrt_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = sqrt (2 * x - 4)) ↔ x ≥ 2 :=
by 
  sorry

end sqrt_meaningful_iff_l665_665336


namespace negation_equiv_l665_665821

-- Define the original proposition
def P := ∀ x ∈ ℝ, abs x + x^2 ≥ 0

-- State the equivalence of the negation of the proposition
theorem negation_equiv : ¬ P ↔ ∃ x ∈ ℝ, abs x + x^2 < 0 := sorry

end negation_equiv_l665_665821


namespace decorative_plate_painted_fraction_l665_665512

noncomputable def fraction_painted_area (total_area painted_area : ℕ) : ℚ :=
  painted_area / total_area

theorem decorative_plate_painted_fraction :
  let side_length := 4
  let total_area := side_length * side_length
  let painted_smaller_squares := 6
  fraction_painted_area total_area painted_smaller_squares = 3 / 8 :=
by
  sorry

end decorative_plate_painted_fraction_l665_665512


namespace sum_of_coefficients_1_to_7_l665_665275

noncomputable def polynomial_expression (x : ℝ) : ℝ :=
  (1 + x) * (1 - 2 * x) ^ 7

theorem sum_of_coefficients_1_to_7 :
  let a₀ := (polynomial_expression 0)
  let a₈ := ((Nat.choose 7 7 : ℝ) * (-2) ^ 7)
  let sum := a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇
  (polynomial_expression 1) = -2 →
  sum = 125 := by
  sorry

end sum_of_coefficients_1_to_7_l665_665275


namespace min_trips_is_157_l665_665811

theorem min_trips_is_157 :
  ∃ x y : ℕ, 31 * x + 32 * y = 5000 ∧ x + y = 157 :=
sorry

end min_trips_is_157_l665_665811


namespace number_of_hens_l665_665995

theorem number_of_hens (H C : ℕ) (h1 : H + C = 44) (h2 : 2 * H + 4 * C = 128) : H = 24 :=
by
  sorry

end number_of_hens_l665_665995


namespace average_sales_l665_665840

-- Define the cost calculation for each special weekend
noncomputable def valentines_day_sales_per_ticket : Real :=
  ((4 * 2.20) + (6 * 1.50) + (7 * 1.20)) / 10

noncomputable def st_patricks_day_sales_per_ticket : Real :=
  ((3 * 2.00) + 6.25 + (8 * 1.00)) / 8

noncomputable def christmas_sales_per_ticket : Real :=
  ((6 * 2.15) + (4.25 + (4.25 / 3.0)) + (9 * 1.10)) / 9

-- Define the combined average snack sales
noncomputable def combined_average_sales_per_ticket : Real :=
  ((4 * 2.20) + (6 * 1.50) + (7 * 1.20) + (3 * 2.00) + 6.25 + (8 * 1.00) + (6 * 2.15) + (4.25 + (4.25 / 3.0)) + (9 * 1.10)) / 27

-- Proof problem as a Lean theorem
theorem average_sales : 
  valentines_day_sales_per_ticket = 2.62 ∧ 
  st_patricks_day_sales_per_ticket = 2.53 ∧ 
  christmas_sales_per_ticket = 3.16 ∧ 
  combined_average_sales_per_ticket = 2.78 :=
by 
  sorry

end average_sales_l665_665840


namespace cube_volume_l665_665907

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l665_665907


namespace cube_volume_l665_665906

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l665_665906


namespace problem_statement_l665_665109

theorem problem_statement (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  ∃ x ∈ { (a^2 + 1) / b^2, (b^2 + 1) / c^2, (c^2 + 1) / d^2, (d^2 + 1) / a^2 }, x ≥ 2 :=
by
  sorry

end problem_statement_l665_665109


namespace sum_first_100_odd_l665_665093

-- Define the sequence of odd numbers.
def odd (n : ℕ) : ℕ := 2 * n + 1

-- Define the sum of the first n odd natural numbers.
def sumOdd (n : ℕ) : ℕ := (n * (n + 1))

-- State the theorem.
theorem sum_first_100_odd : sumOdd 100 = 10000 :=
by
  -- Skipping the proof as per the instructions
  sorry

end sum_first_100_odd_l665_665093


namespace smallest_number_bounds_l665_665049

theorem smallest_number_bounds (a : ℕ → ℝ) (h_sum : (∑ i in finset.range 8, a i) = 4 / 3)
    (h_pos_sum : ∀ i : fin 8, 0 < ∑ j in (finset.univ \ {i}), a j) :
  -8 < a 0 ∧ a 0 ≤ 1 / 6 :=
by
  sorry

end smallest_number_bounds_l665_665049


namespace b_n_formula_T_n_bound_l665_665634

def seq_a (n : ℕ) : ℕ := if n = 0 then 0 else 3 * n - 1

def seq_b (n : ℕ) : ℕ :=
  seq_a (3^n)

def c_n (n : ℕ) : ℝ := 
  (3 * ↑n) / (↑(seq_b n) + 1)

def T_n (n : ℕ) : ℝ :=
  (finset.range n).sum (λ i, c_n (i+1))

theorem b_n_formula (n : ℕ) : seq_b n = 3^(n+1) - 1 := sorry

theorem T_n_bound (n : ℕ) : T_n n < 3 / 4 := sorry

end b_n_formula_T_n_bound_l665_665634


namespace gcd_correct_l665_665579

def gcd_765432_654321 : ℕ :=
  Nat.gcd 765432 654321

theorem gcd_correct : gcd_765432_654321 = 6 :=
by sorry

end gcd_correct_l665_665579


namespace jimsSiblingsAreAustinAndSue_l665_665044

structure Child :=
  (name : String)
  (eyeColor : String)
  (hairColor : String)

def children : List Child :=
  [ {name := "Benjamin", eyeColor := "Blue", hairColor := "Black"},
    {name := "Jim", eyeColor := "Brown", hairColor := "Blonde"},
    {name := "Nadeen", eyeColor := "Brown", hairColor := "Black"},
    {name := "Austin", eyeColor := "Blue", hairColor := "Blonde"},
    {name := "Tevyn", eyeColor := "Blue", hairColor := "Black"},
    {name := "Sue", eyeColor := "Blue", hairColor := "Blonde"} ]

def shareCharacteristic (c1 c2 : Child) : Prop :=
  c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor

def areSiblings (c1 c2 c3 : Child) : Prop :=
  shareCharacteristic c1 c2 ∧ shareCharacteristic c2 c3 ∧ shareCharacteristic c1 c3

theorem jimsSiblingsAreAustinAndSue :
  ∃ (austin : Child) (sue : Child),
    (austin.name = "Austin") ∧ (sue.name = "Sue") ∧ areSiblings (children.get! 1) austin sue :=
by
  sorry

end jimsSiblingsAreAustinAndSue_l665_665044


namespace number_of_committees_l665_665616

theorem number_of_committees 
  (maths econ : ℕ)
  (maths_count econ_count total_count : ℕ)
  (h_maths : maths = 3)
  (h_econ : econ = 10)
  (h_total : total_count = 7) 
  (h_maths_count : maths_count + econ_count = 7)
  : ∑ i in finset.Icc 1 3, binom (maths + econ) i * binom (maths + econ - i) (total_count - i) = 1596 :=
by
  have : binom 13 7 - binom 10 7 = 1596, from by {
    calc
    binom 13 7 - binom 10 7
        = 1716 - 120 : by norm_num
    ... = 1596 : rfl
  }
  rw this
  sorry

end number_of_committees_l665_665616


namespace mrs_hilt_found_nickels_l665_665775

theorem mrs_hilt_found_nickels : 
  ∀ (total cents quarter cents dime cents nickel cents : ℕ), 
    total = 45 → 
    quarter = 25 → 
    dime = 10 → 
    nickel = 5 → 
    ((total - (quarter + dime)) / nickel) = 2 := 
by
  intros total quarter dime nickel h_total h_quarter h_dime h_nickel
  sorry

end mrs_hilt_found_nickels_l665_665775


namespace purely_imaginary_z_range_of_m_l665_665298

-- Part 1: Given a purely imaginary complex number z such that (z + 2) / (1 - i) + z is real, prove z = -2/3 * i.
theorem purely_imaginary_z (z : ℂ) (h1 : z.im ≠ 0) (h2 : (z + 2) / (1 - complex.I) + z ∈ ℝ) : z = -(2/3) * complex.I := 
sorry

-- Part 2: If the point represented by (m - z)^2 is in the first quadrant, prove the range of m is (2/3, +∞)
theorem range_of_m (z : ℂ) (m : ℝ) (h1 : z = -(2/3) * complex.I) 
  (h2 : ((m : ℂ) - z)^2.re > 0) (h3 : ((m : ℂ) - z)^2.im > 0) : m > (2/3) := 
sorry

end purely_imaginary_z_range_of_m_l665_665298


namespace actual_miles_traveled_l665_665131

-- Conditions
def skips_4_7 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ digits 10 n → d ≠ 4 ∧ d ≠ 7

def faulty_odometer_reading : ℕ := 5006

-- Question to prove the actual miles
theorem actual_miles_traveled : faulty_odometer_reading = 5006 → skips_4_7 faulty_odometer_reading → actual_miles_by_faulty_odometer faulty_odometer_reading = 1721 :=
by sorry

-- Auxiliary defintion to calculate the actual miles from faulty odometer.
def actual_miles_by_faulty_odometer (n : ℕ) : ℕ :=
sorry

end actual_miles_traveled_l665_665131


namespace trigonometric_identity_l665_665457

theorem trigonometric_identity :
  (sin (92 * real.pi / 180) - sin (32 * real.pi / 180) * cos (60 * real.pi / 180)) / cos (32 * real.pi / 180) = sqrt 3 / 2 :=
by 
  sorry

end trigonometric_identity_l665_665457


namespace peter_runs_more_than_andrew_each_day_l665_665012

-- Define the constants based on the conditions
def miles_andrew : ℕ := 2
def total_days : ℕ := 5
def total_miles : ℕ := 35

-- Define a theorem to prove the number of miles Peter runs more than Andrew each day
theorem peter_runs_more_than_andrew_each_day : 
  ∃ x : ℕ, total_days * (miles_andrew + x) + total_days * miles_andrew = total_miles ∧ x = 3 :=
by
  sorry

end peter_runs_more_than_andrew_each_day_l665_665012


namespace find_PQ_length_l665_665725

-- Definitions of the points and line segments
structure Point where
  x : ℝ
  y : ℝ

structure LineSegment where
  start : Point
  end : Point

def midpoint (A B : Point) : Point :=
  { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

def length (A B : Point) : ℝ :=
  real.sqrt ((A.x - B.x) ^ 2 + (A.y - B.y) ^ 2)

-- Given conditions
axiom B : Point
axiom C : Point
axiom A : Point
axiom D : Point

axiom BC : LineSegment
axiom AD : LineSegment

instance : has_coe_to_fun LineSegment (λ _, Point × Point) := ⟨λ S, (S.start, S.end)⟩

-- Horizontal segments with specific lengths
axiom h1 : B.y = C.y
axiom h2 : A.y = D.y
axiom h3 : length B C = 1500
axiom h4 : length A D = 2500
axiom angleA : ∃ (θ : ℝ), θ = 40 * real.pi / 180
axiom angleD : ∃ (φ : ℝ), φ = 50 * real.pi / 180

-- Midpoints of segments BC and AD respectively
noncomputable def P : Point := midpoint B C
noncomputable def Q : Point := midpoint A D

-- Required proof
theorem find_PQ_length : length P Q = 500 := sorry

end find_PQ_length_l665_665725


namespace isosceles_triangle_perimeter_l665_665181

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 4) (h2 : b = 9) (h3 : isosceles a b) : 
  (perimeter a b) = 22 := 
  sorry

def isosceles (a b : ℝ) : Prop := (a = b ∨ a ≠ b)
def perimeter (a b : ℝ) : ℝ := if (a = b) then 2 * a + b else if (a ≠ b) then a + a + b else 0

end isosceles_triangle_perimeter_l665_665181


namespace arccos_one_eq_zero_l665_665197

-- Define the inverse cosine function and its properties
noncomputable def arccos (x : ℝ) : ℝ := 
  if x = 1 then 0 
  else sorry -- Simulating full definition relevant to the proof context

-- State the proposition
theorem arccos_one_eq_zero :
  arccos 1 = 0 :=
begin
  -- Proof will be provided here
  sorry
end

end arccos_one_eq_zero_l665_665197


namespace principal_amount_l665_665863

variable (SI R T P : ℝ)

-- Given conditions
axiom SI_def : SI = 2500
axiom R_def : R = 10
axiom T_def : T = 5

-- Main theorem statement
theorem principal_amount : SI = (P * R * T) / 100 → P = 5000 :=
by
  sorry

end principal_amount_l665_665863


namespace value_of_x_l665_665685

theorem value_of_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end value_of_x_l665_665685


namespace value_of_expression_l665_665476

def expression (x y z : ℤ) : ℤ :=
  x^2 + y^2 - z^2 + 2 * x * y + x * y * z

theorem value_of_expression (x y z : ℤ) (h1 : x = 2) (h2 : y = -3) (h3 : z = 1) : 
  expression x y z = -7 := by
  sorry

end value_of_expression_l665_665476


namespace total_hair_cut_l665_665586

-- Definitions from conditions
def first_cut : ℝ := 0.375
def second_cut : ℝ := 0.5

-- The theorem stating the math problem
theorem total_hair_cut : first_cut + second_cut = 0.875 := by
  sorry

end total_hair_cut_l665_665586


namespace sum_of_reversed_all_odd_digits_l665_665442

noncomputable def A (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def B (a b c : ℕ) : ℕ := 100 * c + 10 * b + a
def all_odd_digits (n : ℕ) : Prop := n.digits 10 ⟨· % 2 = 1⟩

theorem sum_of_reversed_all_odd_digits :
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ all_odd_digits (A a b c + B a b c) :=
sorry

end sum_of_reversed_all_odd_digits_l665_665442


namespace triangle_concurrency_l665_665790

-- Define Triangle Structure
structure Triangle (α : Type*) :=
(A B C : α)

-- Define Medians, Angle Bisectors, and Altitudes Concurrency Conditions
noncomputable def medians_concurrent {α : Type*} [MetricSpace α] (T : Triangle α) : Prop := sorry
noncomputable def angle_bisectors_concurrent {α : Type*} [MetricSpace α] (T : Triangle α) : Prop := sorry
noncomputable def altitudes_concurrent {α : Type*} [MetricSpace α] (T : Triangle α) : Prop := sorry

-- Main Theorem Statement
theorem triangle_concurrency {α : Type*} [MetricSpace α] (T : Triangle α) :
  medians_concurrent T ∧ angle_bisectors_concurrent T ∧ altitudes_concurrent T :=
by 
  -- Proof outline: Prove each concurrency condition
  sorry

end triangle_concurrency_l665_665790


namespace no_always_1x3_rectangle_l665_665269

/-- From a sheet of graph paper measuring 8 x 8 cells, 12 rectangles of size 1 x 2 were cut out along the grid lines. 
Prove that it is not necessarily possible to always find a 1 x 3 checkered rectangle in the remaining part. -/
theorem no_always_1x3_rectangle (grid_size : ℕ) (rectangles_removed : ℕ) (rect_size : ℕ) :
  grid_size = 64 → rectangles_removed * rect_size = 24 → ¬ (∀ remaining_cells, remaining_cells ≥ 0 → remaining_cells ≤ 64 → ∃ (x y : ℕ), remaining_cells = x * y ∧ x = 1 ∧ y = 3) :=
  by
  intro h1 h2 h3
  /- Exact proof omitted for brevity -/
  sorry

end no_always_1x3_rectangle_l665_665269


namespace triangle_similar_l665_665461

-- Define the structures for points and triangles
structure Point := (x : ℝ) (y : ℝ)

structure Triangle :=
(A : Point)
(B : Point)
(C : Point)

-- Given conditions
def similar (T1 T2 : Triangle) : Prop := sorry -- This captures the similarity property

def sameOrientation (T1 T2 : Triangle) : Prop := sorry -- To capture same orientation

def collinear (p1 p2 p3 : Point) : Prop := sorry -- To represent collinearity

def ratioEqual (p1 p2 p3 p4 p5 p6 : Point) : Prop :=
  let r1 := dist p1 p2 / dist p2 p3
  let r2 := dist p4 p5 / dist p5 p6
  let r3 := dist p4 p6 / dist p6 p6
  r1 = r2 ∧ r2 = r3

-- Main theorem to prove
theorem triangle_similar
  (T1 T2 : Triangle)
  (A B C : Point)
  (h1 : similar T1 T2)
  (h2 : sameOrientation T1 T2)
  (h3 : collinear T1.A T2.A A)
  (h4 : collinear T1.B T2.B B)
  (h5 : collinear T1.C T2.C C)
  (h6 : ratioEqual T1.A A T2.A T1.B B T2.B T1.C C T2.C) :
  similar ⟨A, B, C⟩ T1 := 
  sorry

end triangle_similar_l665_665461


namespace evaluate_expression_l665_665744

theorem evaluate_expression (a b c d : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 2018) 
  (h2 : 3 * a + 8 * b + 24 * c + 37 * d = 2018) : 
  3 * b + 8 * c + 24 * d + 37 * a = 1215 :=
by 
  sorry

end evaluate_expression_l665_665744


namespace cost_of_pencils_and_pens_l665_665806

theorem cost_of_pencils_and_pens (a b : ℝ) (h1 : 4 * a + b = 2.60) (h2 : a + 3 * b = 2.15) : 3 * a + 2 * b = 2.63 :=
sorry

end cost_of_pencils_and_pens_l665_665806


namespace rectangle_ratio_l665_665526

open Real

theorem rectangle_ratio (A B C D E : Point) (rat : ℚ) : 
  let area_rect := 1
  let area_pentagon := (7 / 10 : ℚ)
  let area_triangle_AEC := 3 / 10
  let area_triangle_ECD := 1 / 5
  let x := 3 * EA
  let y := 2 * EA
  let diag_longer_side := sqrt (5 * EA ^ 2)
  let diag_shorter_side := EA * sqrt 5
  let ratio := sqrt 5 
  ( area_pentagon == area_rect * (7 / 10) ) →
  ( area_triangle_AEC + area_pentagon = area_rect ) →
  ( area_triangle_AEC == area_rect - area_pentagon ) →
  ( ratio == diag_longer_side / diag_shorter_side ) :=
  sorry

end rectangle_ratio_l665_665526


namespace smallest_value_condition_l665_665048

theorem smallest_value_condition 
  (a : Fin 8 → ℝ)
  (h_sum : ∑ i, a i = 4 / 3)
  (h_pos_sum : ∀ i, 0 < ∑ j, if j == i then 0 else a j) :
  -8 < (Finset.min' Finset.univ (λ i, a i)) ∧ (Finset.min' Finset.univ (λ i, a i)) ≤ 1 / 6 :=
by
  sorry

end smallest_value_condition_l665_665048


namespace watermelon_prices_l665_665466

theorem watermelon_prices :
  ∃ (m n : ℚ), 
    (9 * m + n = 42) ∧ 
    (6 * m + 10 * n = 42) ∧ 
    (m > n) ∧ 
    (n > 0) ∧ 
    (m = 4.5) ∧ 
    (n = 1.5) :=
by {
  let m := (9 * 4.5 + 1.5) / 9,
  let n := (42 - 9 * 4.5),
  use [m, n],
  have h1: 9 * m + n = 42 := by {
    rw [m, n],
    norm_num,
  },
  have h2: 6 * m + 10 * n = 42 := by {
    rw [m, n],
    norm_num,
  },
  have h3: m > n := by {
    rw [m, n],
    norm_num,
  },
  have h4: n > 0 := by {
    rw n,
    norm_num,
  },
  exact ⟨m, n, h1, h2, h3, h4⟩,
  sorry
}

end watermelon_prices_l665_665466


namespace roots_of_polynomial_l665_665234

def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

theorem roots_of_polynomial :
  {x : ℝ | P x = 0} = {1, -1, 3} := 
sorry

end roots_of_polynomial_l665_665234


namespace no_solutions_for_specific_a_l665_665592

theorem no_solutions_for_specific_a (a : ℝ) :
  (a < -9) ∨ (a > 0) →
  ¬ ∃ x : ℝ, 5 * |x - 4 * a| + |x - a^2| + 4 * x - 3 * a = 0 :=
by sorry

end no_solutions_for_specific_a_l665_665592


namespace false_statement_l665_665990

-- Define the geometrical conditions based on the problem statements
variable {A B C D: Type}

-- A rhombus with equal diagonals is a square
def rhombus_with_equal_diagonals_is_square (R : A) : Prop := 
  ∀ (a b : A), a = b → true

-- A rectangle with perpendicular diagonals is a square
def rectangle_with_perpendicular_diagonals_is_square (Rec : B) : Prop :=
  ∀ (a b : B), a = b → true

-- A parallelogram with perpendicular and equal diagonals is a square
def parallelogram_with_perpendicular_and_equal_diagonals_is_square (P : C) : Prop :=
  ∀ (a b : C), a = b → true

-- A quadrilateral with perpendicular and bisecting diagonals is a square
def quadrilateral_with_perpendicular_and_bisecting_diagonals_is_square (Q : D) : Prop :=
  ∀ (a b : D), (a = b) → true 

-- The main theorem: Statement D is false
theorem false_statement (Q : D) : ¬ (quadrilateral_with_perpendicular_and_bisecting_diagonals_is_square Q) := 
  sorry

end false_statement_l665_665990


namespace negation_proof_l665_665556

-- Definitions based on conditions
def atMostTwoSolutions (solutions : ℕ) : Prop := solutions ≤ 2
def atLeastThreeSolutions (solutions : ℕ) : Prop := solutions ≥ 3

-- Statement of the theorem
theorem negation_proof (solutions : ℕ) : atMostTwoSolutions solutions ↔ ¬ atLeastThreeSolutions solutions :=
by
  sorry

end negation_proof_l665_665556


namespace large_triangle_distinguishable_l665_665844

def EquilateralTriangleColors (colors : Finset ℕ) : Prop := 
  colors.card = 8

def NoAdjacentSameColor (colors : Finset ℕ) (adjacency : List (ℕ × ℕ)) : Prop := 
  ∀ (i j : ℕ), (i, j) ∈ adjacency → (colors i ≠ colors j)

noncomputable def NumberOfDistinguishableTriangles (colors : Fin n) (adjacency : List (ℕ × ℕ)) : ℕ :=
  colors.card * (colors.card - 1) * (colors.card - 2) * (colors.card - 3)

theorem large_triangle_distinguishable (colors : Finset ℕ) (adjacency : List (ℕ × ℕ)) :
  EquilateralTriangleColors colors →
  NoAdjacentSameColor colors adjacency →
  NumberOfDistinguishableTriangles colors adjacency = 1680 :=
by
  intros
  sorry

end large_triangle_distinguishable_l665_665844


namespace polynomial_roots_l665_665242

theorem polynomial_roots :
  (∀ x, x^3 - 3 * x^2 - x + 3 = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) := 
by
  sorry

end polynomial_roots_l665_665242


namespace tan_subtraction_l665_665270

theorem tan_subtraction :
  ∀ (α : ℝ), tan α = 2 → tan (α - π / 4) = 1 / 3 :=
by sorry

end tan_subtraction_l665_665270


namespace symmetry_center_l665_665665

-- Define the function
def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

-- Define the conditions
theorem symmetry_center (ω : ℝ) (φ : ℝ) (hω : ω > 0) (hφ : |φ| < Real.pi / 2)
  (hx : f ω φ (Real.pi / 3) = 1) (hperiod : 2 * Real.pi / ω = 4 * Real.pi) :
  ∃ k : ℤ, k * 2 * Real.pi - 2 * Real.pi / 3 = -2 * Real.pi / 3 := 
by
  sorry

end symmetry_center_l665_665665


namespace price_of_20_percent_stock_l665_665251

theorem price_of_20_percent_stock (annual_income : ℝ) (investment : ℝ) (dividend_rate : ℝ) (price_of_stock : ℝ) :
  annual_income = 1000 →
  investment = 6800 →
  dividend_rate = 20 →
  price_of_stock = 136 :=
by
  intros h_income h_investment h_dividend_rate
  sorry

end price_of_20_percent_stock_l665_665251


namespace first_day_of_month_is_thursday_l665_665025

theorem first_day_of_month_is_thursday :
  (27 - 7 - 7 - 7 + 1) % 7 = 4 :=
by
  sorry

end first_day_of_month_is_thursday_l665_665025


namespace find_vd_l665_665350

theorem find_vd
  (EFGH : Type)
  [Rectangle EFGH]
  (J : Point EFGH)
  (on_fg : J ∈ FG)
  (right_angle_1 : ∠E J H = 90)
  (UV : Line EFGH)
  (perpendicular_fg : Perpendicular UV FG)
  (equal_segments : SegmentLength F J = SegmentLength J U)
  (intersection : IntersectsAt JH UV K)
  (L : Point EFGH)
  (on_gh : L ∈ GH)
  (le_passes : PassesThrough L E K)
  (je_length : SegmentLength J E = 24)
  (ek_length : SegmentLength E K = 20)
  (jk_length : SegmentLength J K = 16) :
  SegmentLength V D = 36 / 5 :=
by sorry

end find_vd_l665_665350


namespace cube_volume_from_surface_area_l665_665953

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l665_665953


namespace PlayStation_cost_l665_665011

def birthday_money : ℝ := 200
def christmas_money : ℝ := 150
def price_per_game : ℝ := 7.5
def games_to_sell : ℕ := 20
def total_gift_money : ℝ := birthday_money + christmas_money
def total_games_money : ℝ := games_to_sell * price_per_game
def total_money : ℝ := total_gift_money + total_games_money

theorem PlayStation_cost : total_money = 500 := by
  sorry

end PlayStation_cost_l665_665011


namespace max_value_inequality_l665_665610

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (abc * (a + b + c) / ((a + b)^2 * (b + c)^2) ≤ 1 / 4) :=
sorry

end max_value_inequality_l665_665610


namespace length_of_train_is_174_96_l665_665545

noncomputable def speed_km_per_hr : ℝ := 70
noncomputable def time_seconds : ℝ := 9

def speed_m_per_s : ℝ := speed_km_per_hr * (1000 / 3600)
def length_of_train : ℝ := speed_m_per_s * time_seconds

theorem length_of_train_is_174_96 :
  length_of_train = 174.96 :=
by
  unfold speed_km_per_hr time_seconds speed_m_per_s length_of_train
  sorry

end length_of_train_is_174_96_l665_665545


namespace projection_is_correct_l665_665626

variables (a b : ℝ × ℝ)
def dot_product (u v : ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2

def magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2^2)

def projection (a b : ℝ × ℝ) : ℝ :=
(dot_product a b) / (magnitude b)

theorem projection_is_correct : 
a = (2, 3) → b = (-4, 7) → projection a b = (Real.sqrt 65) / 5 :=
by
  intros ha hb
  rw [ha, hb]
  unfold projection dot_product magnitude
  sorry

end projection_is_correct_l665_665626


namespace sum_of_lengths_of_square_pyramid_l665_665063

theorem sum_of_lengths_of_square_pyramid
  (edge_length : ℝ)
  (h1 : ∑ n in finset.range 18, edge_length = 81)
  (h_same_edges : ∀ (e1 e2 : ℝ), e1 = e2) :
  ∑ n in finset.range 8, edge_length = 36 := 
by
  sorry

end sum_of_lengths_of_square_pyramid_l665_665063


namespace exists_uniform_subgrid_l665_665571

theorem exists_uniform_subgrid (color : Fin 3 × Fin 7 → Bool) :
  ∃ (m n : ℕ) (r1 r2 c1 c2 : ℕ), 
    2 ≤ m ∧ m ≤ 3 ∧ 2 ≤ n ∧ n ≤ 7 ∧ 
    r1 < r2 ∧ r2 - r1 + 1 = m ∧ 
    c1 < c2 ∧ c2 - c1 + 1 = n ∧ 
    (color (⟨r1, Nat.lt_of_lt_of_le (Nat.succ_lt_succ (Nat.succ_pos _)) (by linarith)⟩, ⟨c1, Nat.succ_lt_succ (Nat.succ_pos 6)⟩) = 
     color (⟨r1, Nat.lt_of_lt_of_le (Nat.succ_lt_succ (Nat.succ_pos _)) (by linarith)⟩, ⟨c2, Nat.succ_lt_succ (Nat.succ_pos 6)⟩)) ∧
    (color (⟨r2, Nat.lt_of_lt_of_le (Nat.succ_lt_succ (Nat.succ_pos _)) (by linarith)⟩, ⟨c1, Nat.succ_lt_succ (Nat.succ_pos 6)⟩) = 
     color (⟨r2, Nat.lt_of_lt_of_le (Nat.succ_lt_succ (Nat.succ_pos _)) (by linarith)⟩, ⟨c2, Nat.succ_lt_succ (Nat.succ_pos 6)⟩)) :=
sorry

end exists_uniform_subgrid_l665_665571


namespace problem_DE_length_l665_665419

theorem problem_DE_length
  (AB AD : ℝ)
  (AB_eq : AB = 7)
  (AD_eq : AD = 10)
  (area_eq : 7 * CE = 140)
  (DC CE DE : ℝ)
  (DC_eq : DC = 7)
  (CE_eq : CE = 20)
  : DE = Real.sqrt 449 :=
by
  sorry

end problem_DE_length_l665_665419


namespace cube_volume_from_surface_area_l665_665889

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l665_665889


namespace factor_expression_l665_665227

theorem factor_expression (x : ℝ) : 
  5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by sorry

end factor_expression_l665_665227


namespace total_cost_l665_665139

theorem total_cost (cost_pencil cost_pen : ℕ) 
(h1 : cost_pen = cost_pencil + 9) 
(h2 : cost_pencil = 2) : 
cost_pencil + cost_pen = 13 := 
by 
  -- Proof would go here 
  sorry

end total_cost_l665_665139


namespace partial_fraction_decomposition_l665_665031

theorem partial_fraction_decomposition :
  ∃ (a b c : ℤ), (0 ≤ a ∧ a < 5) ∧ (0 ≤ b ∧ b < 13) ∧ (1 / 2015 = a / 5 + b / 13 + c / 31) ∧ (a + b = 14) :=
sorry

end partial_fraction_decomposition_l665_665031


namespace false_propositions_count_l665_665824

/-- Complementary events must be mutually exclusive events. -/
def complementary_events_mutually_exclusive : Prop :=
  ∀ (Ω : Type) (A : set Ω), P(Aᶜ) + P(A) = 1

/-- If A and B are two events, then P(A ∪ B) = P(A) + P(B). -/
def event_union_probability (A B : set Ω) [measurable_space Ω] (P : measure Ω) : Prop :=
  P(A ∪ B) = P(A) + P(B)

/-- If events A, B, and C are pairwise mutually exclusive, then P(A) + P(B) + P(C) = 1. -/
def pairwise_exclusive_sum_one (A B C : set Ω) [measurable_space Ω] (P : measure Ω) : Prop :=
  (∀ x, A x -> ¬ B x ∧ ¬ C x) ∧
  (∀ x, B x -> ¬ A x ∧ ¬ C x) ∧
  (∀ x, C x -> ¬ A x ∧ ¬ B x) ∧
  (P(A) + P(B) + P(C) = 1)

theorem false_propositions_count :
  num_false_propositions = 2 := sorry

end false_propositions_count_l665_665824


namespace ratio_of_pieces_l665_665024

def total_length (len: ℕ) := len = 35
def longer_piece (len: ℕ) := len = 20

theorem ratio_of_pieces (shorter len_shorter : ℕ) : 
  total_length 35 →
  longer_piece 20 →
  shorter = 35 - 20 →
  len_shorter = 15 →
  (20:ℚ) / (len_shorter:ℚ) = (4:ℚ) / (3:ℚ) :=
by
  sorry

end ratio_of_pieces_l665_665024


namespace polynomial_roots_l665_665241

theorem polynomial_roots :
  (∀ x, x^3 - 3*x^2 - x + 3 = 0 ↔ (x = 1 ∨ x = -1 ∨ x = 3)) :=
by
  intro x
  split
  {
    intro h
    have h1 : x = 1 ∨ x = -1 ∨ x = 3
    {
      sorry
    }
    exact h1
  }
  {
    intro h
    cases h
    {
      rw h
      simp
    }
    {
      cases h
      {
        rw h
        simp
      }
      {
        rw h
        simp
      }
    }
  }

end polynomial_roots_l665_665241


namespace parabola_coeff_sum_l665_665801

theorem parabola_coeff_sum (a b c : ℝ) 
    (h1 : ∃ y0 x0, x0 = a * (y + 3)^2 + 6) 
    (h2 : x = a*y^2 + b*y + c) 
    (h3 : x y = (4, -1)) 
    (h4 : x y = (6, -3)) : (a + b + c = -2) :=
sorry

end parabola_coeff_sum_l665_665801


namespace gcd_765432_654321_l665_665582

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 :=
by 
  sorry

end gcd_765432_654321_l665_665582


namespace expected_value_of_X_is_5_over_3_l665_665515

-- Define the probabilities of getting an interview with company A, B, and C
def P_A : ℚ := 2 / 3
def P_BC (p : ℚ) : ℚ := p

-- Define the random variable X representing the number of interview invitations
def X (P_A P_BC : ℚ) : ℚ := sorry

-- Define the probability of receiving no interview invitations
def P_X_0 (P_A P_BC : ℚ) : ℚ := (1 - P_A) * (1 - P_BC)^2

-- Given condition that P(X=0) is 1/12
def condition_P_X_0 (P_A P_BC : ℚ) : Prop := P_X_0 P_A P_BC = 1 / 12

-- Given p = 1/2 as per the problem solution
def p : ℚ := 1 / 2

-- Expected value of X
def E_X (P_A P_BC : ℚ) : ℚ := (1 * (2 * P_BC * (1 - P_BC) + 2 * P_BC^2 * (1 - P_BC) + (1 - P_A) * P_BC^2)) +
                               (2 * (P_A * P_BC * (1 - P_BC) + P_A * (1 - P_BC)^2 + P_BC * P_BC * (1 - P_A))) +
                               (3 * (P_A * P_BC^2))

-- Theorem proving the expected value of X given the above conditions
theorem expected_value_of_X_is_5_over_3 : E_X P_A (P_BC p) = 5 / 3 :=
by
  -- here you will write the proof later
  sorry

end expected_value_of_X_is_5_over_3_l665_665515


namespace acute_triangle_perpendicular_l665_665611

theorem acute_triangle_perpendicular
  (A B C P Q D E F J : Type)
  [ht : ∀ (a b c : Type), angular a b c < 90]
  (h1 : ∃ BC : segment, BC.contains P ∧ AB = BP)
  (h2 : ∃ CB : segment, CB.contains Q ∧ AC = CQ)
  (excircle : excircle A B C)
  (h3 : excircle.tangent AB D )
  (h4 : excircle.tangent AC E )
  (h5 : extension PD (≠ J, intersects QE F))
  : perpendicular AF FJ :=
by
  sorry

end acute_triangle_perpendicular_l665_665611


namespace limit_n_b_n_l665_665576

open Real

def M (x : ℝ) := x - (x^3) / 3

def iterate_M (x : ℝ) (n : ℕ) : ℝ := nat.iterate M n x

noncomputable def b_n (n : ℕ) : ℝ := iterate_M (25 / n) n

theorem limit_n_b_n : filter.tendsto (λ n : ℕ, n * b_n n) filter.at_top (nhds (9 / 25)) :=
sorry

end limit_n_b_n_l665_665576


namespace magnitude_complex_number_l665_665587

-- Define the complex number
def a : ℝ := 1 / 3
def b : ℝ := - (5 / 7)
def z : ℂ := complex.mk a b

-- Proof statement
theorem magnitude_complex_number :
  complex.abs z = real.sqrt 274 / 21 :=
by
  -- Applying the magnitude formula for complex numbers
  have h : complex.abs z = real.sqrt (a * a + b * b) := sorry,
  -- Simplify the result as per provided calculations
  calc 
    real.sqrt (a * a + b * b) = real.sqrt (1 / 9 + 25 / 49) : by sorry
                       ...     = real.sqrt (274 / 441) : by sorry
                       ...     = real.sqrt 274 / real.sqrt 441 : by sorry
                       ...     = real.sqrt 274 / 21 : by sorry

end magnitude_complex_number_l665_665587


namespace log_equation_solution_l665_665428

theorem log_equation_solution :
  (∃ x : ℝ, log 8 x + log 2 (x ^ 3) = 9) ↔ x = 2 ^ (27 / 10) :=
by
  sorry

end log_equation_solution_l665_665428


namespace angle_R_values_l665_665728

theorem angle_R_values (P Q : ℝ) (h1: 5 * Real.sin P + 2 * Real.cos Q = 5) (h2: 2 * Real.sin Q + 5 * Real.cos P = 3) : 
  ∃ R : ℝ, R = Real.arcsin (1/20) ∨ R = 180 - Real.arcsin (1/20) :=
by
  sorry

end angle_R_values_l665_665728


namespace base_7_to_10_equivalence_l665_665084

theorem base_7_to_10_equivalence : 
  let n := 6 * 7^4 + 5 * 7^3 + 4 * 7^2 + 3 * 7^1 + 2 * 7^0
  in n = 16340 :=
by
  let n := 6 * 7^4 + 5 * 7^3 + 4 * 7^2 + 3 * 7^1 + 2 * 7^0
  show n = 16340
  -- The proof would go here, but we use sorry for the placeholder
  sorry

end base_7_to_10_equivalence_l665_665084


namespace smallest_positive_integer_divisible_by_20_l665_665602

/--
Given n ≥ 4, the smallest n for which any n distinct integers always allow 
the selection of four a, b, c, d such that 20 divides (a + b - c - d) is 9.
-/
theorem smallest_positive_integer_divisible_by_20 (n : ℕ) (h : n ≥ 4) :
  (∃ a b c d : ℤ, ∀ S : Finset ℤ, S.card = n → ∃ (a b c d ∈ S),
    20 ∣ (a + b - c - d)) ↔ n = 9 := 
sorry

end smallest_positive_integer_divisible_by_20_l665_665602


namespace greatest_value_k_l665_665650

theorem greatest_value_k (k : ℝ) (h : ∀ x : ℝ, (x - 1) ∣ (x^2 + 2*k*x - 3*k^2)) : k ≤ 1 :=
  by
  sorry

end greatest_value_k_l665_665650


namespace j_def_l665_665446

def h (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ 1 then 1 - x
  else if 1 ≤ x ∧ x ≤ 3 then real.sqrt (4 - (x - 3)^2) + 1
  else if 3 ≤ x ∧ x ≤ 4 then 2 * (x - 3) + 1
  else 0

def j (x : ℝ) : ℝ := h (6 - x)

theorem j_def (x : ℝ) : j(x) = h(6 - x) := by 
  simp [j]
  sorry

end j_def_l665_665446


namespace factor_expression_l665_665228

theorem factor_expression (x : ℝ) : 
  5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by sorry

end factor_expression_l665_665228


namespace cube_volume_l665_665899

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l665_665899


namespace log_equation_solution_l665_665426

theorem log_equation_solution (x : ℝ) : log 8 x + log 2 (x ^ 3) = 9 → x = 2 ^ 2.7 :=
by
  sorry

end log_equation_solution_l665_665426


namespace probability_defective_probability_defective_given_second_l665_665119
noncomputable theory

-- Define the conditions: output proportions and defect rates of the workshops
def output_proportions : List ℚ := [0.15, 0.2, 0.3, 0.35]
def defect_rates : List ℚ := [0.04, 0.03, 0.02, 0.01]

-- Part (1) question: proving the probability of selecting a defective product
theorem probability_defective :
  let defective_prob := (output_proportions.zip defect_rates).map (λ (p, r), p * r) in
  defective_prob.sum = 0.0215 :=
sorry

-- Part (2) question: proving the conditional probability using Bayes' theorem
theorem probability_defective_given_second :
  let defective_prob := (output_proportions.zip defect_rates).map (λ (p, r), p * r) in
  let total_defective := defective_prob.sum in
  let second_workshop_defective := 0.2 * 0.03 in
  (second_workshop_defective / total_defective) = (12 / 43) :=
sorry

end probability_defective_probability_defective_given_second_l665_665119


namespace tan_equiv_l665_665293

variable (α : ℝ)

-- Given condition
def tan_condition : Prop := tan (π / 7 + α) = 5

-- The theorem we need to prove
theorem tan_equiv (h : tan_condition α) : tan (6 * π / 7 - α) = -5 := 
by 
  sorry

end tan_equiv_l665_665293


namespace close_to_one_below_l665_665554

theorem close_to_one_below (k l m n : ℕ) (h1 : k > l) (h2 : l > m) (h3 : m > n) (hk : k = 43) (hl : l = 7) (hm : m = 3) (hn : n = 2) :
  (1 : ℚ) / k + 1 / l + 1 / m + 1 / n < 1 := by
  sorry

end close_to_one_below_l665_665554


namespace function_solution_l665_665732

def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x) + (0.5 + x) * f(1 - x) = 1

def correct_function (x : ℝ) : ℝ :=
  if x = 0.5 then 0.5 else 1 / (0.5 - x)

theorem function_solution : satisfies_equation correct_function :=
  sorry

end function_solution_l665_665732


namespace number_of_fixed_points_f_2011_l665_665486

-- Define the function f based on the given conditions
def f : ℕ → ℕ
| 1       := 1
| (2 * n) := f n
| (4 * n + 1) := 2 * f (2 * n + 1) - f n
| (4 * n + 3) := 3 * f (2 * n + 1) - 2 * f n
| _ := sorry  -- this is just a placeholder for non-covered cases

-- Define the number 2011
def num := 2011

-- Define the concept of fixed points
def is_fixed_point (f : ℕ → ℕ) (x : ℕ) : Prop :=
  f x = x

-- Define the concept of counting fixed points up to a certain number
def count_fixed_points (f : ℕ → ℕ) (up_to : ℕ) : ℕ :=
  finset.card (finset.filter (is_fixed_point f) (finset.range (up_to + 1)))

-- The main theorem to prove there are 90 fixed points for f up to 2011 based on the given conditions
theorem number_of_fixed_points_f_2011 : count_fixed_points f num = 90 :=
sorry

end number_of_fixed_points_f_2011_l665_665486


namespace inverse_of_97_mod_98_l665_665588

theorem inverse_of_97_mod_98 : 97 * 97 ≡ 1 [MOD 98] :=
by
  sorry

end inverse_of_97_mod_98_l665_665588


namespace factor_expression_l665_665222

noncomputable def factored_expression (x : ℝ) : ℝ :=
  5 * x * (x + 2) + 9 * (x + 2)

theorem factor_expression (x : ℝ) : 
  factored_expression x = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l665_665222


namespace polynomial_div_remainder_l665_665205

theorem polynomial_div_remainder :
  ∀ x : ℂ, remainder ((x^5 - x^3 + x - 1) * (x^3 - 1)) (x^2 - x + 1) = x^2 - 2x + 1 :=
by
  intro x
  sorry

end polynomial_div_remainder_l665_665205


namespace cube_volume_of_surface_area_l665_665975

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l665_665975


namespace salary_percentage_proof_l665_665007

variable {E S : ℝ}

-- Given conditions
def num_employees_before_july1 : ℝ := E
def avg_salary_before_july1 : ℝ := S
def num_employees_after_july1 : ℝ := 0.9 * E
def avg_salary_after_july1 : ℝ := 1.1 * S

-- Total salary before and after July 1
def total_salary_before_july1 : ℝ := num_employees_before_july1 * avg_salary_before_july1
def total_salary_after_july1 : ℝ := num_employees_after_july1 * avg_salary_after_july1

theorem salary_percentage_proof :
  total_salary_after_july1 = 0.99 * total_salary_before_july1 :=
by
  sorry

end salary_percentage_proof_l665_665007


namespace johns_money_l665_665165

theorem johns_money (total_money ali_less nada_more: ℤ) (h1: total_money = 67) 
  (h2: ali_less = -5) (h3: nada_more = 4): 
  ∃ (j: ℤ), (n: ℤ), ali_less = n - 5 ∧ nada_more = 4 * n ∧ total_money = n + (n - 5) + (4 * n) → j = 48 :=
by
  sorry

end johns_money_l665_665165


namespace eccentricity_of_ellipse_l665_665809

theorem eccentricity_of_ellipse :
  (∃ θ : Real, (x = 3 * Real.cos θ) ∧ (y = 4 * Real.sin θ))
  → (∃ e : Real, e = Real.sqrt 7 / 4) := 
sorry

end eccentricity_of_ellipse_l665_665809


namespace solve_product_real_parts_roots_eq_proof_l665_665765

noncomputable def solve_product_real_parts_roots_eq : Prop :=
  let i := Complex.I
  ∃ (z1 z2 : ℂ), (z1 * z1 + 3 * z1 = 8 - 2 * i) ∧
                  (z2 * z2 + 3 * z2 = 8 - 2 * i) ∧
                  (z1 + z2 = -3 + 0) ∧ -- Sum of roots (property of quadratic equations ax^2 + bx + c = 0)
                  (z1 * z2 = 8 - 2i) ∧ -- Product of roots (property of quadratic equations ax^2 + bx + c = 0)
                  ((z1.re * z2.re) = -4)

theorem solve_product_real_parts_roots_eq_proof :
  solve_product_real_parts_roots_eq :=
by
  sorry

end solve_product_real_parts_roots_eq_proof_l665_665765


namespace candidate_lost_by_1650_votes_l665_665116

theorem candidate_lost_by_1650_votes (total_votes : ℕ) (pct_candidate : ℝ) (pct_rival : ℝ) : 
  total_votes = 5500 → 
  pct_candidate = 0.35 → 
  pct_rival = 0.65 → 
  ((pct_rival * total_votes) - (pct_candidate * total_votes)) = 1650 := 
by
  intros h1 h2 h3
  sorry

end candidate_lost_by_1650_votes_l665_665116


namespace product_of_S_n_l665_665277

noncomputable def S_n (n : ℕ) (h : n > 0) : ℚ :=
  let a : ℕ → ℚ := λ n, 1 / (2 : ℚ)^n
  let term := λ n, 1 / ((Real.log2 (a n)) * (Real.log2 (a (n + 1))))
  ∑ i in Finset.range n, term (i + 1)

theorem product_of_S_n :
  (Finset.range 10).prod (λ n, S_n (n + 1) (by linarith)) = 1 / 11 := by
  sorry

end product_of_S_n_l665_665277


namespace correct_divisor_l665_665471

noncomputable def incorrect_divisor : ℝ := 30.6 / 204

def move_decimal_one_place_left (x : ℝ) : ℝ :=
  x / 10

theorem correct_divisor :
  ∃ d : ℝ, (30.6 / 204) / 10 = d :=
sorry

end correct_divisor_l665_665471


namespace number_of_second_set_matches_is_15_l665_665502

-- Definitions of the conditions given in the problem
def avg_runs_first_set : ℕ := 50
def num_first_matches : ℕ := 30
def avg_runs_second_set : ℕ := 26
def total_num_matches : ℕ := 45
def total_avg_runs_all_matches : ℕ := 42

-- Stating the main proof problem
theorem number_of_second_set_matches_is_15 :
  let total_runs_first_set := avg_runs_first_set * num_first_matches
      avg_runs_all := total_avg_runs_all_matches
      total_runs_all := avg_runs_all * total_num_matches
  in ∃ x : ℕ, total_runs_first_set + avg_runs_second_set * x = total_runs_all ∧ x = 15 :=
by 
  let total_runs_first_set := avg_runs_first_set * num_first_matches
  let total_runs_all := total_avg_runs_all_matches * total_num_matches
  existsi 15
  simp [total_runs_first_set, total_runs_all]
  sorry

end number_of_second_set_matches_is_15_l665_665502


namespace evaluate_expression_l665_665608

def acbd (a b c d : ℝ) : ℝ := a * d - b * c

theorem evaluate_expression (x : ℝ) (h : x^2 - 3 * x + 1 = 0) :
  acbd (x + 1) (x - 2) (3 * x) (x - 1) = 1 := 
by
  sorry

end evaluate_expression_l665_665608


namespace work_done_l665_665652

-- Define points A and B as vectors
def A : EuclideanSpace ℝ ℕ := EuclideanSpace.of 2 [2, 0]
def B : EuclideanSpace ℝ ℕ := EuclideanSpace.of 2 [4, 0]

-- Define the force vector F
def F : EuclideanSpace ℝ ℕ := EuclideanSpace.of 2 [2, 3]

-- Define the displacement vector AB
def AB := B - A

-- Define the dot product for Euclidean space
def dot_product (v w : EuclideanSpace ℝ ℕ) := v.inner w

-- Prove that the work done by F on the object moving from A to B is 4 joules
theorem work_done : dot_product F AB = 4 :=
by
  sorry

end work_done_l665_665652


namespace geometric_sequence_fourth_term_l665_665445

theorem geometric_sequence_fourth_term (a₁ a₂ a₃ : ℝ) (r : ℝ)
    (h₁ : a₁ = 5^(3/4))
    (h₂ : a₂ = 5^(1/2))
    (h₃ : a₃ = 5^(1/4))
    (geometric_seq : a₂ = a₁ * r ∧ a₃ = a₂ * r) :
    a₃ * r = 1 := 
by
  sorry

end geometric_sequence_fourth_term_l665_665445


namespace factor_expression_l665_665229

theorem factor_expression (x : ℝ) : 
  5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by sorry

end factor_expression_l665_665229


namespace rectangle_side_l665_665454

theorem rectangle_side (x : ℝ) (w : ℝ) (P : ℝ) (hP : P = 30) (h : 2 * (x + w) = P) : w = 15 - x :=
by
  -- Proof goes here
  sorry

end rectangle_side_l665_665454


namespace circle_outside_triangle_area_l665_665396

theorem circle_outside_triangle_area (P Q R S T : Point) (r : ℝ) :
  angle PRQ = 90 ∧ PQ = 12 ∧
  (∃ O : Point, circle O r ∧ tangent S PQ ∧ tangent T PR) ∧
  (∃ S' T' : Point, S' ≠ S ∧ S' ≠ T ∧ T' ≠ S ∧ T' ≠ T ∧
                   (diametrically_opposite S O S') ∧ 
                   (diametrically_opposite T O T') ∧ 
                   lies_on Q'R S' ∧ 
                   lies_on Q'R T') →

  let quarter_circle_area := (1 / 4) * π * r^2,
      triangle_area := (1 / 2) * r^2,
      outside_area := quarter_circle_area - triangle_area in
  outside_area = 4 * π - 8 :=
begin
  sorry
end

end circle_outside_triangle_area_l665_665396


namespace folded_rectangle_ratio_l665_665524

-- Define the conditions
def original_area (a b : ℝ) := a * b
def pentagon_area (a b : ℝ) := (7 / 10) * original_area a b

-- Define the ratio to prove
def ratio_of_sides (a b : ℝ) := a / b

-- Define the theorem to prove the ratio equals sqrt(5)
theorem folded_rectangle_ratio (a b : ℝ) (h: a > b) 
  (A1 : pentagon_area a b = (7 / 10) * original_area a b) :
  ratio_of_sides a b = real.sqrt 5 :=
  sorry

end folded_rectangle_ratio_l665_665524


namespace sunil_interest_l665_665432

noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem sunil_interest :
  let A := 19828.80
  let r := 0.08
  let n := 1
  let t := 2
  let P := 19828.80 / (1 + 0.08) ^ 2
  P * (1 + r / n) ^ (n * t) = 19828.80 →
  A - P = 2828.80 :=
by
  sorry

end sunil_interest_l665_665432


namespace triangle_AC_l665_665364

theorem triangle_AC {A B C : Type} [LinearOrderedField A] [DivInvMonoid A] [DecidableEq A] 
  (BC : A) (AB AC : A) (B : A) (area : A) 
  (h1 : B = (π / 3 : A)) 
  (h2 : AB = 4) 
  (h3 : (1 / 2 : A) * 4 * BC * Real.sin (π / 3) = 3 * Real.sqrt 3) :
  AC = Real.sqrt 13 :=
by
  sorry

end triangle_AC_l665_665364


namespace largest_root_polynomial_intersection_l665_665032

/-
Given a polynomial P(x) = x^6 - 15x^5 + 74x^4 - 130x^3 + a * x^2 + b * x
and a line L(x) = c * x - 24,
such that P(x) stays above L(x) except at three distinct values of x where they intersect,
and one of the intersections is a root of triple multiplicity.
Prove that the largest value of x for which P(x) = L(x) is 6.
-/
theorem largest_root_polynomial_intersection (a b c : ℝ) (P L : ℝ → ℝ) (x : ℝ) :
  P x = x^6 - 15*x^5 + 74*x^4 - 130*x^3 + a*x^2 + b*x →
  L x = c*x - 24 →
  (∀ x, P x ≥ L x) ∨ (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ P x1 = L x1 ∧ P x2 = L x2 ∧ P x3 = L x3 ∧
  (∃ x0 : ℝ, x1 = x0 ∧ x2 = x0 ∧ x3 = x0 ∧ ∃ k : ℕ, k = 3)) →
  x = 6 :=
sorry

end largest_root_polynomial_intersection_l665_665032


namespace shape_is_cone_l665_665606

-- Definition for spherical coordinates
structure SphericalCoord where
  ρ      : ℝ    -- radius
  θ      : ℝ    -- azimuthal angle
  φ      : ℝ    -- polar angle

-- Constant for the given angle
constant c : ℝ

-- Define the condition for the problem
def fixed_phi (P : SphericalCoord) : Prop := P.φ = 2 * c

-- The proposition to be proved (shape description based on the fixed φ angle)
theorem shape_is_cone (P : SphericalCoord) (hφ : fixed_phi P) : 
  ∃ (Vx Vy : ℝ), P.ρ = Vx ^ 2 + Vy ^ 2 + (tan (2 * c) * sqrt (Vx ^ 2 + Vy ^ 2)) ^ 2 := sorry

end shape_is_cone_l665_665606


namespace susan_wins_at_least_once_in_three_games_l665_665023

noncomputable def probability_winning_at_least_once_in_three_games 
  (dice_probability : ℚ) (games : ℚ) : ℚ :=
  1 - (1 - dice_probability) ^ games

theorem susan_wins_at_least_once_in_three_games :
  probability_winning_at_least_once_in_three_games (1 / 9) 3 = 217 / 729 :=
by
  sorry

end susan_wins_at_least_once_in_three_games_l665_665023


namespace number_of_correct_statements_is_2_l665_665178

-- The statements to be evaluated
def statement_1 (C : Type) [metric_space C] [normed_group C] [normed_space ℝ C] (l : C → Prop) (chord : C) :=
  ∃ center : C, ∀ line : C, l line → chord ⊥ line → center ∈ line

def statement_2 (C : Type) [metric_space C] [normed_group C] [normed_space ℝ C] (diameter : C) :=
  ∃ tangent : C, ∀ line : C, line = tangent → tangent ∈ diameter.end_points → perpendicular line diameter ∧ meet_the_circle_in_one_point line

def statement_3 (C : Type) [metric_space C] [normed_group C] [normed_space ℝ C] (quad : C) :=
  ∃ circle : C, ∀ quadrilateral : C, quad = quadrilateral → complementary_diagonals quadrilateral → quadrilateral.vertices ⊆ circle

def statement_4 (C : Type) [metric_space C] [normed_group C] [normed_space ℝ C] (P : C) :=
  ∃ circle : C, ∀ tangents : C, tangents_drawn_outside P circle → line_passing_contact_points ⊥ line_joining_point_to_center P

-- The proof problem: prove that the number of correct statements is 2
theorem number_of_correct_statements_is_2 : 
  (∃ (n : ℕ), n = 2 ∧ 
    (statement_1 = false ∧ 
     statement_2 = false ∧ 
     statement_3 = true ∧ 
     statement_4 = true)) :=
sorry

end number_of_correct_statements_is_2_l665_665178


namespace matches_start_with_l665_665408

-- Let M be the number of matches Nate started with
variables (M : ℕ)

-- Given conditions
def dropped_creek (dropped : ℕ) := dropped = 10
def eaten_by_dog (eaten : ℕ) := eaten = 2 * 10
def matches_left (final_matches : ℕ) := final_matches = 40

-- Prove that the number of matches Nate started with is 70
theorem matches_start_with 
  (h1 : dropped_creek 10)
  (h2 : eaten_by_dog 20)
  (h3 : matches_left 40) 
  : M = 70 :=
sorry

end matches_start_with_l665_665408


namespace cube_volume_l665_665933

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l665_665933


namespace no_domovoi_exists_l665_665553

variables {Domovoi Creature : Type}

def likes_pranks (c : Creature) : Prop := sorry
def likes_cleanliness_order (c : Creature) : Prop := sorry
def is_domovoi (c : Creature) : Prop := sorry

axiom all_domovoi_like_pranks : ∀ (c : Creature), is_domovoi c → likes_pranks c
axiom all_domovoi_like_cleanliness : ∀ (c : Creature), is_domovoi c → likes_cleanliness_order c
axiom cleanliness_implies_no_pranks : ∀ (c : Creature), likes_cleanliness_order c → ¬ likes_pranks c

theorem no_domovoi_exists : ¬ ∃ (c : Creature), is_domovoi c := 
sorry

end no_domovoi_exists_l665_665553


namespace smallest_positive_x_for_g_maximum_l665_665200

def g (x : ℝ) : ℝ := sin (x / 4) + sin (x / 13)

theorem smallest_positive_x_for_g_maximum : 
  ∃ (x : ℝ), (0 < x) ∧ (g x = 2) ∧ (∀ y, y > 0 → g y = 2 → x ≤ y) ∧ x = 8190 :=
by
  sorry

end smallest_positive_x_for_g_maximum_l665_665200


namespace discount_percentage_l665_665422

theorem discount_percentage (marked_price sale_price cost_price : ℝ) (gain1 gain2 : ℝ)
  (h1 : gain1 = 0.35)
  (h2 : gain2 = 0.215)
  (h3 : sale_price = 30)
  (h4 : cost_price = marked_price / (1 + gain1))
  (h5 : marked_price = cost_price * (1 + gain2)) :
  ((sale_price - marked_price) / sale_price) * 100 = 10.009 :=
sorry

end discount_percentage_l665_665422


namespace area_swept_out_by_PQ_l665_665265

noncomputable def area_of_swept_figure (t : ℝ) : ℝ :=
  if t ∈ Icc (-1 : ℝ) 0 then
    if t < -0.5 then
      -(t : ℝ) * -(t + 1 : ℝ)
    else if t < 0.5 then
      (t ^ 2 + 1/4)
    else
      (t + 1 : ℝ)
  else
    0

theorem area_swept_out_by_PQ : ∫ x in (Icc (-1:ℝ) (0:ℝ)), area_of_swept_figure x = 13 / 12 :=
by
  sorry

end area_swept_out_by_PQ_l665_665265


namespace exists_integer_point_in_convex_set_l665_665742

open Set

theorem exists_integer_point_in_convex_set (K : Set (ℝ × ℝ))
  (hK1 : Convex ℝ K)
  (hK2 : ∀ (x y : ℝ), (x, y) ∈ K → (-x, -y) ∈ K)
  (hK3 : (measure_theory.measure_of (K)).to_real > 4) :
  ∃ (m n : ℤ), (m, n) ≠ (0, 0) ∧ ((m:ℝ), (n:ℝ)) ∈ K :=
sorry

end exists_integer_point_in_convex_set_l665_665742


namespace angle_ADC_proof_l665_665362

theorem angle_ADC_proof
  (A B C D E : Type)
  [triangle : Triangle A B C]
  (h_ABC_eq_60 : ∠ABC = 60)
  (h_AD_bisects_BAC : AngleBisector AD ∠BAC)
  (h_DC_bisects_BCA : AngleBisector DC ∠BCA)
  (h_BE_bisects_ABC : AngleBisector BE ∠ABC) :
  ∠ADC = 120 :=
by
  sorry

end angle_ADC_proof_l665_665362


namespace find_ellipse_equation_l665_665291

theorem find_ellipse_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0)
  (h_ellipse_eqn : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (h_perp : ∃ (P : ℝ × ℝ), P ∈ C ∧ ∃ (m n : ℝ), |m| + |n| = 2a ∧ ∃ (PF1 PF2 : ℝ × ℝ),
    ∥PF1∥ = m ∧ ∥PF2∥ = n ∧ PF1 ⋅ PF2 = 0)
  (h_area : ∃ (m n : ℝ), m * n = 36)
  (h_perimeter : ∃ (m n : ℝ), m + n = 18) :
  ∃ x y : ℝ, (x^2 / 25 + y^2 / 9 = 1) := 
by
  sorry

end find_ellipse_equation_l665_665291


namespace cube_volume_from_surface_area_l665_665925

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l665_665925


namespace minimum_boxes_l665_665569

theorem minimum_boxes (x y z : ℕ) (h1 : 50 * x = 40 * y) (h2 : 50 * x = 25 * z) :
  x + y + z = 17 :=
by
  -- Prove that given these equations, the minimum total number of boxes (x + y + z) is 17
  sorry

end minimum_boxes_l665_665569


namespace value_of_b_l665_665711

theorem value_of_b (b : ℝ) (x : ℝ) (h : x = 1) (h_eq : 3 * x^2 - b * x + 3 = 0) : b = 6 :=
by
  sorry

end value_of_b_l665_665711


namespace range_of_2x_plus_y_l665_665656

-- Define the circle equation's condition.
def circle_equation (x y θ : ℝ) : Prop :=
  y^2 - 6 * y * (Real.sin θ) + x^2 - 8 * x * (Real.cos θ) + 7 * (Real.cos θ)^2 + 8 = 0

-- The parametric equations of the trajectory of the circle's center C.
def parametric_trajectory (θ : ℝ) : ℝ × ℝ :=
  (4 * Real.cos θ, 3 * Real.sin θ)

-- The range of values for 2x + y given x and y are defined by the parametric equations above.
theorem range_of_2x_plus_y (θ : ℝ) (h : 0 ≤ θ ∧ θ < 2 * Real.pi) :
  let (x, y) := parametric_trajectory θ in
  -Real.sqrt 73 ≤ 2 * x + y ∧ 2 * x + y ≤ Real.sqrt 73 :=
sorry

end range_of_2x_plus_y_l665_665656


namespace base7_65432_to_dec_is_16340_l665_665087

def base7_to_dec (n : ℕ) : ℕ :=
  6 * 7^4 + 5 * 7^3 + 4 * 7^2 + 3 * 7^1 + 2 * 7^0

theorem base7_65432_to_dec_is_16340 : base7_to_dec 65432 = 16340 :=
by
  sorry

end base7_65432_to_dec_is_16340_l665_665087


namespace factor_expression_l665_665223

noncomputable def factored_expression (x : ℝ) : ℝ :=
  5 * x * (x + 2) + 9 * (x + 2)

theorem factor_expression (x : ℝ) : 
  factored_expression x = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l665_665223


namespace smallest_value_condition_l665_665047

theorem smallest_value_condition 
  (a : Fin 8 → ℝ)
  (h_sum : ∑ i, a i = 4 / 3)
  (h_pos_sum : ∀ i, 0 < ∑ j, if j == i then 0 else a j) :
  -8 < (Finset.min' Finset.univ (λ i, a i)) ∧ (Finset.min' Finset.univ (λ i, a i)) ≤ 1 / 6 :=
by
  sorry

end smallest_value_condition_l665_665047


namespace problem_statement_l665_665377

noncomputable def x_n (b n : ℕ) : ℕ := 
  let a := (b^n - 1) / (b - 1)
  let b_term := 2 * b^n * ((b^n - 1) / (b - 1))
  let c := 5 * b^(2*n)
  in a + b_term + c

theorem problem_statement (b : ℕ) (hb : b > 5) :
  (∃ M : ℕ, ∀ n : ℕ, n > M → ∃ k : ℕ, k^2 = x_n b n) ↔ b = 10 :=
sorry

end problem_statement_l665_665377


namespace exists_k_bound_poly_l665_665487

theorem exists_k_bound_poly (n : ℕ) (a : Fin n.succ → ℝ) :
  ∃ k : Fin n.succ, ∀ x : ℝ, 0 ≤ x → x ≤ 1 →
    (Finset.sum (Finset.range n.succ) (λ i, a i * x^i) ≤ Finset.sum (Finset.range k.succ) a) :=
by sorry

end exists_k_bound_poly_l665_665487


namespace cylinder_cut_area_l665_665115

-- Definitions for the cylinder and conditions
def radius : ℕ := 8
def height : ℕ := 10
def arc_measure : ℕ := 90

-- Theorem statement
theorem cylinder_cut_area :
  ∃ (d e f : ℕ), f % 4 ≠ 0 ∧ f ≠ 1 ∧ (d * π + e * (sqrt f)) = 320 * π ∧ (d + e + f) = 321 :=
by {
  let d := 320,
  let e := 0,
  let f := 1,
  use [d, e, f],
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { sorry, }
}

end cylinder_cut_area_l665_665115


namespace common_prime_divisors_count_48_80_l665_665318

theorem common_prime_divisors_count_48_80 :
  (set_of (λ p, prime p ∧ p ∣ 48 ∧ p ∣ 80)).to_finset.card = 1 := 
sorry

end common_prime_divisors_count_48_80_l665_665318


namespace hexagon_angles_sum_l665_665354

theorem hexagon_angles_sum (α β γ δ ε ζ : ℝ)
  (h1 : α + γ + ε = 180)
  (h2 : β + δ + ζ = 180) : 
  α + β + γ + δ + ε + ζ = 360 :=
by 
  sorry

end hexagon_angles_sum_l665_665354


namespace trajectory_equation_of_other_focus_l665_665283

theorem trajectory_equation_of_other_focus 
  (A B C : (ℝ × ℝ)) 
  (hA : A = (-7, 0))
  (hB : B = (7, 0))
  (hC : C = (2, -12))
: ∃ (D : ℝ × ℝ), (∀ (x y : ℝ), ((x < 0) → ((x, y) ∈ set_of (λ (p : ℝ × ℝ), (p.1^2 / 1) - (p.2^2 / 48) = 1)))) := 
sorry

end trajectory_equation_of_other_focus_l665_665283


namespace circle_center_and_radius_l665_665308

theorem circle_center_and_radius :
  ∀ (C : Set (ℝ × ℝ)), (∀ x y : ℝ, (x, y) ∈ C ↔ x^2 + y^2 + y = 0) →
  (∃ c : ℝ × ℝ, c = (0, -1/2)) ∧ (∃ r : ℝ, r = 1/2) :=
by
  intros C h
  have eq1 : ∀ x y : ℝ, x^2 + y^2 + y = 0 ↔ x^2 + (y + 1/2)^2 = 1/4
   sorry

end circle_center_and_radius_l665_665308


namespace steps_to_split_stones_l665_665843

theorem steps_to_split_stones (n : ℕ) (hn : n ≥ 2) : 
  ∃ m : ℕ, m-1 < Real.log2 n ∧ Real.log2 n ≤ m :=
by
  sorry

end steps_to_split_stones_l665_665843


namespace roots_of_polynomial_l665_665248

-- Define the polynomial P(x) = x^3 - 3x^2 - x + 3
def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

-- Define the statement to prove the roots of the polynomial
theorem roots_of_polynomial :
  ∀ x : ℝ, (P x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) :=
by
  sorry

end roots_of_polynomial_l665_665248


namespace find_total_number_of_pages_l665_665789

variable (x : ℕ)

-- condition1: On the first day, Serena read (1/6)x + 10 pages.
def remaining_pages_day1 (x : ℕ) := (5 / 6 : ℚ) * x - 10

-- condition2: On the second day, she read (1/5) of the remaining pages plus 13 more.
def remaining_pages_day2 (x : ℕ) := (4 / 5 : ℚ) * (remaining_pages_day1 x) - 13

-- condition3: On the third day, she read (1/4) of the remaining pages plus 15 more.
def remaining_pages_day3 (x : ℕ) := (3 / 4 : ℚ) * (remaining_pages_day2 x) - 15

-- condition4: At the end of the third day, there were 48 pages left.
theorem find_total_number_of_pages (x : ℕ) :
  remaining_pages_day3 x = 48 → x = 153 :=
by
  sorry

end find_total_number_of_pages_l665_665789


namespace john_savings_remaining_l665_665373

theorem john_savings_remaining 
  (saved_base8 : ℕ := 5555) 
  (ticket_cost : ℕ := 1200) : 
  nat.of_digits 8 saved_base8 - ticket_cost = 1725 := 
by
  -- Convert saved_base8 to base 10
  let saved_base10 := 5 * 8^3 + 5 * 8^2 + 5 * 8 + 5
  have : nat.of_digits 8 saved_base8 = saved_base10, by sorry
  -- Subtract the ticket cost
  have : saved_base10 - ticket_cost = 1725, by sorry
  sorry

end john_savings_remaining_l665_665373


namespace geom_seq_sum_3000_l665_665057

noncomputable
def sum_geom_seq (a r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then a * n
  else a * (1 - r ^ n) / (1 - r)

theorem geom_seq_sum_3000 (a r : ℝ) (h1: sum_geom_seq a r 1000 = 300) (h2: sum_geom_seq a r 2000 = 570) :
  sum_geom_seq a r 3000 = 813 :=
sorry

end geom_seq_sum_3000_l665_665057


namespace sin_cos_sum_l665_665495

theorem sin_cos_sum :
  sin (18 * π / 180) * cos (12 * π / 180) + cos (18 * π / 180) * sin (12 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_sum_l665_665495


namespace find_vector_b_l665_665003

-- Definitions and conditions from the problem
def vector_a : ℝ × ℝ := (2, -1)
def is_collinear_and_same_direction (b : ℝ × ℝ) (a : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ b = (k * a.1, k * a.2)
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Theorem statement: prove the result
theorem find_vector_b (b : ℝ × ℝ) :
  is_collinear_and_same_direction b vector_a ∧ magnitude b = 2 * real.sqrt 5 → b = (4, -2) :=
by
  sorry

end find_vector_b_l665_665003


namespace symmetry_axis_of_function_l665_665255

theorem symmetry_axis_of_function : 
  let f (x : ℝ) := sin (2 * x) - sqrt 3 * cos (2 * x) in
  exists (axis : ℝ), axis = π / 12 ∧ (∀ x : ℝ, f (2 * axis - x) = f (2 * axis + x)) := 
by
  intro f
  use π / 12
  split
  . rfl
  . sorry

end symmetry_axis_of_function_l665_665255


namespace cube_volume_l665_665894

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l665_665894


namespace john_savings_remaining_l665_665374

theorem john_savings_remaining 
  (saved_base8 : ℕ := 5555) 
  (ticket_cost : ℕ := 1200) : 
  nat.of_digits 8 saved_base8 - ticket_cost = 1725 := 
by
  -- Convert saved_base8 to base 10
  let saved_base10 := 5 * 8^3 + 5 * 8^2 + 5 * 8 + 5
  have : nat.of_digits 8 saved_base8 = saved_base10, by sorry
  -- Subtract the ticket cost
  have : saved_base10 - ticket_cost = 1725, by sorry
  sorry

end john_savings_remaining_l665_665374


namespace roots_of_polynomial_l665_665249

-- Define the polynomial P(x) = x^3 - 3x^2 - x + 3
def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

-- Define the statement to prove the roots of the polynomial
theorem roots_of_polynomial :
  ∀ x : ℝ, (P x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) :=
by
  sorry

end roots_of_polynomial_l665_665249


namespace oscar_leap_more_than_piper_hop_l665_665008

noncomputable def difference_leap_hop : ℝ :=
let number_of_poles := 51
let total_distance := 7920 -- in feet
let Elmer_strides_per_gap := 44
let Oscar_leaps_per_gap := 15
let Piper_hops_per_gap := 22
let number_of_gaps := number_of_poles - 1
let Elmer_total_strides := Elmer_strides_per_gap * number_of_gaps
let Oscar_total_leaps := Oscar_leaps_per_gap * number_of_gaps
let Piper_total_hops := Piper_hops_per_gap * number_of_gaps
let Elmer_stride_length := total_distance / Elmer_total_strides
let Oscar_leap_length := total_distance / Oscar_total_leaps
let Piper_hop_length := total_distance / Piper_total_hops
Oscar_leap_length - Piper_hop_length

theorem oscar_leap_more_than_piper_hop :
  difference_leap_hop = 3.36 := by
  sorry

end oscar_leap_more_than_piper_hop_l665_665008


namespace cube_volume_from_surface_area_l665_665883

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l665_665883


namespace emily_typing_speed_l665_665212

theorem emily_typing_speed :
  (∃ words : ℕ, ∃ hours : ℕ, words = 10800 ∧ hours = 3) →
  (∃ words_per_minute : ℕ, words_per_minute = 60) :=
by
  intro h
  cases h with words hw
  cases hw with hours hw_eqs
  cases hw_eqs with words_eq hours_eq
  use 60
  sorry

end emily_typing_speed_l665_665212


namespace cube_volume_of_surface_area_l665_665985

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l665_665985


namespace meals_combinations_l665_665482

theorem meals_combinations (total_dishes special_dish: ℕ) 
  (h_total_dishes: total_dishes = 12) 
  (h_special_dish: special_dish = 1):
  let remaining_dishes := total_dishes - special_dish in 
  (remaining_dishes * special_dish) + (special_dish * remaining_dishes) + (remaining_dishes * remaining_dishes) = 143 := 
by 
  sorry

end meals_combinations_l665_665482


namespace cube_volume_l665_665935

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l665_665935


namespace fraction_is_one_over_three_l665_665808

variable (x : ℚ) -- Let the fraction x be a rational number
variable (num : ℚ) -- Let the number be a rational number

theorem fraction_is_one_over_three (h1 : num = 45) (h2 : x * num - 5 = 10) : x = 1 / 3 := by
  sorry

end fraction_is_one_over_three_l665_665808


namespace cube_volume_of_surface_area_l665_665974

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l665_665974


namespace sum_of_sequence_l665_665654

theorem sum_of_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, ∑ i in Finset.range (n+1), a i = S (n+1)) ∧
  (∀ n, (1 : ℚ) / (a n + 1) = 2 / (a (n+1) + 1)) ∧
  a 2 = 1 →
  S 7 = 120 :=
sorry

end sum_of_sequence_l665_665654


namespace minimum_value_of_f_range_of_a_l665_665282

noncomputable def f (x : ℝ) := x * Real.log x
noncomputable def g (x a : ℝ) := -x^2 + a * x - 3

theorem minimum_value_of_f :
  ∃ x_min : ℝ, ∀ x : ℝ, 0 < x → f x ≥ -1/Real.exp 1 := sorry -- This statement asserts that the minimum value of f(x) is -1/e.

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → 2 * f x ≥ g x a) → a ≤ 4 := sorry -- This statement asserts that if 2f(x) ≥ g(x) for all x > 0, then a is at most 4.

end minimum_value_of_f_range_of_a_l665_665282


namespace alley_width_l665_665133

noncomputable def width_of_alley (L : ℝ) : ℝ :=
  L * (1 + real.sqrt 3) / 2

theorem alley_width (L : ℝ) (H : ℝ) (h_angle1 : real.sin (real.pi / 3) = H / L)
  (h_angle2 : H = L * real.sqrt 3 / 2) : 
    ∃ d, d = width_of_alley L := by
  use width_of_alley L
  sorry

end alley_width_l665_665133


namespace total_silk_dyed_correct_l665_665847

-- Define the conditions
def green_silk_yards : ℕ := 61921
def pink_silk_yards : ℕ := 49500
def total_silk_yards : ℕ := green_silk_yards + pink_silk_yards

-- State the theorem to be proved
theorem total_silk_dyed_correct : total_silk_yards = 111421 := by
  sorry

end total_silk_dyed_correct_l665_665847


namespace find_treasure_l665_665522

-- Definitions of points and conditions
variables (A B C1 C2 C3 A1 B1 D1 D2 D3 K : Type) 

-- Conditions:
-- Defining the perpendicular construction and rotation
def perpendicular_equal_dist (C A A1 : Type) : Prop :=
  sorry  -- Placeholder for a precise definition involving distance and perpendicularity

def intersect (A B1 B A1 D1 : Type) : Prop :=
  sorry  -- Placeholder for intersection of lines

-- Assumptions:
axiom hC1 (C1 A A1 B B1 D1 : Type) : perpendicular_equal_dist C1 A A1 ∧ perpendicular_equal_dist C1 B B1 ∧ intersect A B1 B A1 D1
axiom hC2 (C2 A A1 B B1 D2 : Type) : perpendicular_equal_dist C2 A A1 ∧ perpendicular_equal_dist C2 B B1 ∧ intersect A B1 B A1 D2
axiom hC3 (C3 A A1 B B1 D3 : Type) : perpendicular_equal_dist C3 A A1 ∧ perpendicular_equal_dist C3 B B1 ∧ intersect A B1 B A1 D3

-- Point K is equidistant from D1, D2, D3
def is_equidistant (K D1 D2 D3 : Type) : Prop :=
  sorry  -- Placeholder for equidistant property definition

axiom hK : is_equidistant K D1 D2 D3

-- The theorem statement
theorem find_treasure (A B K : Type) [h : is_equidistant K D1 D2 D3] : 
  midpoint A B = K :=
sorry  -- Proof to be provided

end find_treasure_l665_665522


namespace solve_angle_A_solve_perimeter_l665_665671

variables {A B C : ℝ} {a b c : ℝ}

def triangle (A B C : ℝ) (a b c : ℝ) : Prop := 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π

theorem solve_angle_A (h : sqrt 3 * sin A * cos A - sin A ^ 2 = 0) : 
  A = π / 3 :=
sorry

theorem solve_perimeter (hA : A = π / 3) (hmn : (1, sin C) = (2, sin B)) (ha: a = 3) 
  (ht: triangle A B C a b c) : 
  a + b + c = 3 * (1 + sqrt 3) :=
sorry

end solve_angle_A_solve_perimeter_l665_665671


namespace count_two_digit_even_numbers_l665_665658

theorem count_two_digit_even_numbers :
  let digits := {2, 3, 5, 7, 8}
  in (∃ n : ℕ, n = 8) ↔
  (∃ digit1 digit2 : ℕ,
    digit1 ∈ digits ∧
    digit2 ∈ digits ∧
    digit2 % 2 = 0 ∧
    digit1 ≠ digit2 ∧
    n = 4 * 2) := sorry

end count_two_digit_even_numbers_l665_665658


namespace factor_expression_l665_665225

noncomputable def factored_expression (x : ℝ) : ℝ :=
  5 * x * (x + 2) + 9 * (x + 2)

theorem factor_expression (x : ℝ) : 
  factored_expression x = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l665_665225


namespace melanie_dimes_l665_665404

theorem melanie_dimes : 
  let start_dimes := 8
  let given_away_dimes := 7
  let received_dimes := 4
  start_dimes - given_away_dimes + received_dimes = 5 :=
by
  -- Definitions from the conditions
  let start_dimes := 8
  let given_away_dimes := 7
  let received_dimes := 4
  
  -- main proof statement
  calc
    start_dimes - given_away_dimes + received_dimes
      = 8 - 7 + 4 : by rfl
  ... = 1 + 4   : by rfl
  ... = 5       : by rfl

end melanie_dimes_l665_665404


namespace find_x_value_l665_665715

variables (A B C O G M N : Type)
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup O] [AddCommGroup G] [AddCommGroup M] [AddCommGroup N]

-- Conditions
variables (OA OB OC OG OM ON : A)
variables (x : ℝ)

-- OM = 2MA
axiom om_eq_two_ma (OA : A) (OM : A) : OM = (2 / 3 : ℝ) • OA

-- N is the midpoint of BC
axiom on_eq_midpoint_bc (ON OB OC : A) : ON = (1 / 2 : ℝ) • (OB + OC)

-- Given OG
axiom og_eq_given (OG OA OB OC : A) (x : ℝ) : OG = (1 / 3 : ℝ) • OA + (x / 4 : ℝ) • OB + (x / 4 : ℝ) • OC

-- Collinearity condition
axiom collinear_condition (λ : ℝ) (ON OM OG : A) :
  OG = λ • ON + (1 - λ) • OM → (G M N : G) -- Represent collinearity here

-- Proof of x = 1 given conditions
theorem find_x_value (OA OB OC OG OM ON : A) (x : ℝ) (λ : ℝ)
  (h1 : OM = (2 / 3 : ℝ) • OA)
  (h2 : ON = (1 / 2 : ℝ) • (OB + OC))
  (h3 : OG = (1 / 3 : ℝ) • OA + (x / 4 : ℝ) • OB + (x / 4 : ℝ) • OC)
  (h4 : OG = λ • ON + (1 - λ) • OM) :
  x = 1 :=
sorry

end find_x_value_l665_665715


namespace minji_total_water_intake_l665_665773

variable (morning_water : ℝ)
variable (afternoon_water : ℝ)

theorem minji_total_water_intake (h_morning : morning_water = 0.26) (h_afternoon : afternoon_water = 0.37):
  morning_water + afternoon_water = 0.63 :=
sorry

end minji_total_water_intake_l665_665773


namespace matilda_jellybeans_l665_665400

/-- Suppose Matilda has half as many jellybeans as Matt.
    Suppose Matt has ten times as many jellybeans as Steve.
    Suppose Steve has 84 jellybeans.
    Then Matilda has 420 jellybeans. -/
theorem matilda_jellybeans
    (matilda_jellybeans : ℕ)
    (matt_jellybeans : ℕ)
    (steve_jellybeans : ℕ)
    (h1 : matilda_jellybeans = matt_jellybeans / 2)
    (h2 : matt_jellybeans = 10 * steve_jellybeans)
    (h3 : steve_jellybeans = 84) : matilda_jellybeans = 420 := 
sorry

end matilda_jellybeans_l665_665400


namespace arithmetic_mean_solution_l665_665437

/-- Given the arithmetic mean of six expressions is 30, prove the values of x and y are as follows. -/
theorem arithmetic_mean_solution (x y : ℝ) (h : ((2 * x - y) + 20 + (3 * x + y) + 16 + (x + 5) + (y + 8)) / 6 = 30) (hy : y = 10) : 
  x = 18.5 :=
by {
  sorry
}

end arithmetic_mean_solution_l665_665437


namespace midpoint_coordinates_line_slope_l665_665351

-- Conditions: Definitions of the line l and curve C
def line_l (α t : ℝ) := (x=2+t*cos α, y=√3+t*sin α)
def curve_C (θ : ℝ) := (x=2*cos θ, y=sin θ)

-- Points: Definitions of points P, A and B
def P := (2, √3)

-- Problem 1: Prove the coordinates of the midpoint M for α=π/3
theorem midpoint_coordinates (α : ℝ) (hα : α = π/3) :
  ∀ t1 t2 : ℝ, 
  let A := (2 + t1 * cos α, √3 + t1 * sin α) in
  let B := (2 + t2 * cos α, √3 + t2 * sin α) in
  midpoint (line_l α t1) (line_l α t2) = (12/13, -√3/13) :=
sorry

-- Problem 2: Prove the slope of the line l given |PA| * |PB| = |OP|^2, P(2, √3)
theorem line_slope (α : ℝ)
  (a_on_line : (2 + t * cos α, √3 + t * sin α))
  (on_curve_A : ∃ θ1 : ℝ, (2 + t * cos α, √3 + t * sin α) = (2*cos θ1, sin θ1))
  (on_curve_B : ∃ θ2 : ℝ, (2 + t * cos α, √3 + t * sin α) = (2*cos θ2, sin θ2))
  (cond : |PA| * |PB| = |OP|^2)
  : slope (2 + t * cos α, √3 + t * sin α) = √5/4 :=
sorry

end midpoint_coordinates_line_slope_l665_665351


namespace min_theta_symmetry_l665_665033

theorem min_theta_symmetry 
  (x : ℝ)
  (θ : ℝ)
  (h₀ : θ > 0)
  (hx : ∀ x, y = sqrt 3 * cos x + sin x)
  (trans_fn : ∀ x, y = 2 * cos (x - π / 6 - θ)) 
  : θ = π / 6 - 1 (↔ θ = 5 * π / 6) :=
sorry

end min_theta_symmetry_l665_665033


namespace angles_equal_l665_665849

theorem angles_equal {α β γ α1 β1 γ1 : ℝ} (h1 : α + β + γ = 180) (h2 : α1 + β1 + γ1 = 180) 
  (h_eq_or_sum_to_180 : (α = α1 ∨ α + α1 = 180) ∧ (β = β1 ∨ β + β1 = 180) ∧ (γ = γ1 ∨ γ + γ1 = 180)) :
  α = α1 ∧ β = β1 ∧ γ = γ1 := 
by 
  sorry

end angles_equal_l665_665849


namespace problem1_problem2_l665_665564

-- Problem 1
theorem problem1 : (1/2) * real.sqrt 24 - real.sqrt 3 * real.sqrt 2 = 0 :=
by
  sorry

-- Problem 2
theorem problem2 : (2 * real.sqrt 3 + 3 * real.sqrt 2)^2 = 30 + 12 * real.sqrt 6 :=
by
  sorry

end problem1_problem2_l665_665564


namespace academy_league_total_games_l665_665798

theorem academy_league_total_games (teams : ℕ) (plays_each_other_twice games_non_conference : ℕ) 
  (h_teams : teams = 8)
  (h_plays_each_other_twice : plays_each_other_twice = 2 * teams * (teams - 1) / 2)
  (h_games_non_conference : games_non_conference = 6 * teams) :
  (plays_each_other_twice + games_non_conference) = 104 :=
by
  sorry

end academy_league_total_games_l665_665798


namespace trig_identity_l665_665190

theorem trig_identity :
  (2 * (Real.cos (Real.pi / 6)) - Real.tan (Real.pi / 3) + (Real.sin (Real.pi / 4) * Real.cos (Real.pi / 4)) = 1/2) :=
by
  -- Here we use known trigonometric values at specific angles
  have h1 : Real.cos (Real.pi / 6) = sqrt 3 / 2 := Real.cos_pi_div_six,
  have h2 : Real.tan (Real.pi / 3) = sqrt 3 := Real.tan_pi_div_three,
  have h3 : Real.sin (Real.pi / 4) = sqrt 2 / 2 := Real.sin_pi_div_four,
  have h4 : Real.cos (Real.pi / 4) = sqrt 2 / 2 := Real.cos_pi_div_four,

  -- Use these known values to simplify the expression
  calc
    (2 * (Real.cos (Real.pi / 6)) - Real.tan (Real.pi / 3) + (Real.sin (Real.pi / 4) * Real.cos (Real.pi / 4)))
        = 2 * (sqrt 3 / 2) - sqrt 3 + (sqrt 2 / 2 * sqrt 2 / 2) : by rw [h1, h2, h3, h4]
    ... = sqrt 3 - sqrt 3 + 1/2 : by norm_num
    ... = 1/2 : by norm_num

end trig_identity_l665_665190


namespace regular_polygon_sides_l665_665146

theorem regular_polygon_sides (n : ℕ) (h₁ : 2 < n) (h₂ : ∀ (k : ℕ), k = 180 * (n - 2) / n ⇒ k = 150) : n = 12 :=
by 
  sorry

end regular_polygon_sides_l665_665146


namespace sequence_is_integer_l665_665022

theorem sequence_is_integer (k : ℕ) : ∀ n : ℕ, ∃ (a : ℕ → ℤ), a 0 = 0 ∧ 
  (∀ n, a (n + 1) = (k + 1) * a n + k * (a n + 1) + 2 * int.sqrt (k * (k + 1) * a n * (a n + 1))) := 
by
  sorry

end sequence_is_integer_l665_665022


namespace greatest_multiple_less_l665_665088

theorem greatest_multiple_less (a b target: ℕ) (h_lcm: Nat.lcm a b = 60) (h_target: target = 125) : 
  ∃ k: ℕ, k * 60 < target ∧ ∀ m: ℕ, m * 60 < target → m * 60 ≤ k * 60 :=
by
  have h : k = 2 := sorry
  exact ⟨k, sorry⟩

end greatest_multiple_less_l665_665088


namespace domain_of_sqrt_log_l665_665596

noncomputable def domain_of_function : Set ℝ := 
  {x : ℝ | (-Real.sqrt 2) ≤ x ∧ x < -1 ∨ 1 < x ∧ x ≤ Real.sqrt 2}

theorem domain_of_sqrt_log : ∀ x : ℝ, 
  (∃ y : ℝ, y = Real.sqrt (Real.log (x^2 - 1) / Real.log (1/2)) ∧ 
  y ≥ 0) ↔ x ∈ domain_of_function := 
by
  sorry

end domain_of_sqrt_log_l665_665596


namespace number_of_dimes_l665_665346

theorem number_of_dimes (k : ℕ) (dimes quarters : ℕ) (value : ℕ)
  (h1 : 3 * k = dimes)
  (h2 : 2 * k = quarters)
  (h3 : value = (10 * dimes) + (25 * quarters))
  (h4 : value = 400) :
  dimes = 15 :=
by {
  sorry
}

end number_of_dimes_l665_665346


namespace cube_volume_l665_665910

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l665_665910


namespace find_f_5_l665_665620

def f : ℕ → ℕ
| x := if x ≥ 9 then x - 3 else f (x + 6)

theorem find_f_5 : f 5 = 8 := by
  sorry

end find_f_5_l665_665620


namespace number_of_sheep_l665_665107

theorem number_of_sheep (S H : ℕ)
  (h1 : S / H = 4 / 7)
  (h2 : H * 230 = 12880) :
  S = 32 :=
by
  sorry

end number_of_sheep_l665_665107


namespace Bellas_position_at_102_l665_665562

-- Define the movement conditions and rules
def initial_position : (ℤ × ℤ) := (0, 0)

def move_north (p : ℤ × ℤ) (n_units : ℕ) : ℤ × ℤ :=
  (p.1, p.2 + n_units)

def move_south (p : ℤ × ℤ) (n_units : ℕ) : ℤ × ℤ :=
  (p.1, p.2 - n_units)

def move_east (p : ℤ × ℤ) (n_units : ℕ) : ℤ × ℤ :=
  (p.1 + n_units, p.2)

def move_west (p : ℤ × ℤ) (n_units : ℕ) : ℤ × ℤ :=
  (p.1 - n_units, p.2)

-- General function to determine new position
def move (p : ℤ × ℤ) (direction : string) (n_units : ℕ) : ℤ × ℤ :=
  match direction with
  | "north" => move_north p n_units
  | "south" => move_south p n_units
  | "east" => move_east p n_units
  | "west" => move_west p n_units
  | _ => p

-- Define Bella's movement function. Simplified here for brevity.
def bellas_position (steps : ℕ) : ℤ × ℤ := sorry

-- The final proof statement
theorem Bellas_position_at_102 :
  bellas_position 102 = (-23, 29) :=
sorry

end Bellas_position_at_102_l665_665562


namespace problem_statement_l665_665329

theorem problem_statement (a b c : ℝ)
  (h : a * b * c = ( Real.sqrt ( (a + 2) * (b + 3) ) ) / (c + 1)) :
  6 * 15 * 7 = 1.5 :=
sorry

end problem_statement_l665_665329


namespace cube_volume_l665_665937

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l665_665937


namespace john_savings_remaining_l665_665371

theorem john_savings_remaining :
  let saved_base8 : ℕ := 5 * 8^3 + 5 * 8^2 + 5 * 8^1 + 5 * 8^0
  let ticket_cost : ℕ := 1200
  (saved_base8 - ticket_cost) = 1725 :=
by
  let saved_base8 := 5 * 8^3 + 5 * 8^2 + 5 * 8^1 + 5 * 8^0
  let ticket_cost := 1200
  show (saved_base8 - ticket_cost) = 1725 from sorry

end john_savings_remaining_l665_665371


namespace trig_identity_l665_665191

theorem trig_identity :
  (2 * (Real.cos (Real.pi / 6)) - Real.tan (Real.pi / 3) + (Real.sin (Real.pi / 4) * Real.cos (Real.pi / 4)) = 1/2) :=
by
  -- Here we use known trigonometric values at specific angles
  have h1 : Real.cos (Real.pi / 6) = sqrt 3 / 2 := Real.cos_pi_div_six,
  have h2 : Real.tan (Real.pi / 3) = sqrt 3 := Real.tan_pi_div_three,
  have h3 : Real.sin (Real.pi / 4) = sqrt 2 / 2 := Real.sin_pi_div_four,
  have h4 : Real.cos (Real.pi / 4) = sqrt 2 / 2 := Real.cos_pi_div_four,

  -- Use these known values to simplify the expression
  calc
    (2 * (Real.cos (Real.pi / 6)) - Real.tan (Real.pi / 3) + (Real.sin (Real.pi / 4) * Real.cos (Real.pi / 4)))
        = 2 * (sqrt 3 / 2) - sqrt 3 + (sqrt 2 / 2 * sqrt 2 / 2) : by rw [h1, h2, h3, h4]
    ... = sqrt 3 - sqrt 3 + 1/2 : by norm_num
    ... = 1/2 : by norm_num

end trig_identity_l665_665191


namespace sum_b_lt_two_div_three_l665_665279

noncomputable def a (n: ℕ) : ℕ := 3^n

-- Assuming S_n represents the sum of the first n terms of a sequence
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := (Finset.range n).sum a

def b (a : ℕ → ℕ) (n : ℕ) : ℝ := 1 / (Real.logb 3 (a n) * (Real.logb 3 (a n))^2 + 1)

theorem sum_b_lt_two_div_three (n : ℕ) (h : n > 0) : 
  (Finset.range n).sum (b a) < 2 / 3 := by
  sorry

end sum_b_lt_two_div_three_l665_665279


namespace coloring_no_monochromatic_clique_l665_665388

theorem coloring_no_monochromatic_clique (k n : ℕ) (hk : k ≥ 3) (hn : n > 2^(k / 2)) :
  ∃ (coloring : (fin n) × (fin n) → bool),
  ∀ (subset : finset (fin n)), subset.card = k → 
  ∃ (e₁ e₂ : (fin n) × (fin n)), e₁ ∈ subset.image (λ i, (i, i)) ∧ e₂ ∈ subset.image (λ i, (i, i)) ∧ 
  coloring e₁ ≠ coloring e₂ := 
sorry

end coloring_no_monochromatic_clique_l665_665388


namespace ella_needed_score_l665_665211

def current_scores : List ℕ := [92, 82, 75, 65, 88]

def desired_increase : ℝ := 4

def target_average (current_avg : ℝ) := current_avg + desired_increase

def required_score (scores : List ℕ) (target_avg : ℝ) :=
  let num_tests := scores.length + 1
  let total_current_score := scores.foldl (· + ·) 0
  let needed_score := target_avg * num_tests - total_current_score
  needed_score

theorem ella_needed_score :
  required_score current_scores (target_average (402 / 5)) = 105 :=
by
  sorry

end ella_needed_score_l665_665211


namespace john_has_48_l665_665162

variable (Ali Nada John : ℕ)

theorem john_has_48 
  (h1 : Ali + Nada + John = 67)
  (h2 : Ali = Nada - 5)
  (h3 : John = 4 * Nada) : 
  John = 48 := 
by 
  sorry

end john_has_48_l665_665162


namespace length_of_CD_l665_665782

variable {x y u v : ℝ}

def divides_C_R (x y : ℝ) : Prop := x / y = 3 / 5
def divides_C_S (u v : ℝ) : Prop := u / v = 4 / 7
def length_RS (u x y v : ℝ) : Prop := u = x + 5 ∧ v = y - 5
def length_CD (x y : ℝ) : ℝ := x + y

theorem length_of_CD (h1 : divides_C_R x y) (h2 : divides_C_S u v) (h3 : length_RS u x y v) :
  length_CD x y = 40 :=
sorry

end length_of_CD_l665_665782


namespace linear_term_coefficient_l665_665718

-- Define the given equation
def equation (x : ℝ) : ℝ := x^2 - 2022*x - 2023

-- The goal is to prove that the coefficient of the linear term in equation is -2022
theorem linear_term_coefficient : ∀ x : ℝ, equation x = x^2 - 2022*x - 2023 → -2022 = -2022 :=
by
  intros x h
  sorry

end linear_term_coefficient_l665_665718


namespace find_other_number_l665_665421

theorem find_other_number (x y : ℤ) (h1 : 3 * x + 2 * y = 160) (h2 : x = 36 ∨ y = 36) :
    y = 26 ∨ x = 26 :=
by 
  cases h2 with
  | inl h3 =>
    rw [h3] at h1
    have : 2 * y = 160 - 3 * 36 := by linarith
    have : y = (160 - 108) / 2 := by linarith
    exact Or.inl (by linarith)
  | inr h3 =>
    rw [h3] at h1
    have : 3 * x = 160 - 2 * 36 := by linarith
    have : x = (160 - 72) / 3 := by linarith
    exact Or.inr (by linarith)

end find_other_number_l665_665421


namespace total_cost_l665_665137

theorem total_cost (cost_pencil cost_pen : ℕ) 
(h1 : cost_pen = cost_pencil + 9) 
(h2 : cost_pencil = 2) : 
cost_pencil + cost_pen = 13 := 
by 
  -- Proof would go here 
  sorry

end total_cost_l665_665137


namespace polynomial_degree_and_divisibility_l665_665745

def polynomial_divisible_by_13 (n : ℕ) (f : (Fin n → ℤ) → ℤ) : Prop :=
  let N : ℕ := (Finset.filter (λ x, f x % 13 = 0) (Finset.pi (Finset.range n) (λ _, Finset.range 13))).card in
  N % 13 = 0

theorem polynomial_degree_and_divisibility (n : ℕ) (f : (Fin n → ℤ) → ℤ) (h_poly : ∀ x, degree f < n ) :
  polynomial_divisible_by_13 n f := 
  by 
    sorry

end polynomial_degree_and_divisibility_l665_665745


namespace pattern_cost_l665_665771

theorem pattern_cost (fabric_cost_per_yard : ℕ) (yards_of_fabric : ℕ) (thread_cost : ℕ) (spools_of_thread : ℕ) (total_cost : ℕ) (pattern_cost : ℕ) :
  fabric_cost_per_yard = 24 →
  yards_of_fabric = 5 →
  thread_cost = 3 →
  spools_of_thread = 2 →
  total_cost = 141 →
  pattern_cost = total_cost - (yards_of_fabric * fabric_cost_per_yard + spools_of_thread * thread_cost) →
  pattern_cost = 15 :=
by 
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  simp
  sorry

end pattern_cost_l665_665771


namespace cube_volume_l665_665891

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l665_665891


namespace balls_in_boxes_l665_665320

theorem balls_in_boxes :
  (∃ x1 x2 x3 : ℕ, x1 + x2 + x3 = 7) →
  (multichoose 7 3) = 36 :=
by
  sorry

end balls_in_boxes_l665_665320


namespace five_nat_numbers_product_1000_l665_665572

theorem five_nat_numbers_product_1000 :
  ∃ (a b c d e : ℕ), 
    a * b * c * d * e = 1000 ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
    c ≠ d ∧ c ≠ e ∧ 
    d ≠ e := 
by
  sorry

end five_nat_numbers_product_1000_l665_665572


namespace animal_sighting_ratio_l665_665700

theorem animal_sighting_ratio
  (jan_sightings : ℕ)
  (feb_sightings : ℕ)
  (march_sightings : ℕ)
  (total_sightings : ℕ)
  (h1 : jan_sightings = 26)
  (h2 : feb_sightings = 3 * jan_sightings)
  (h3 : total_sightings = jan_sightings + feb_sightings + march_sightings)
  (h4 : total_sightings = 143) :
  (march_sightings : ℚ) / (feb_sightings : ℚ) = 1 / 2 :=
by
  sorry

end animal_sighting_ratio_l665_665700


namespace initial_red_beads_count_l665_665067

noncomputable def green_beads : ℕ := 1
noncomputable def brown_beads : ℕ := 2
noncomputable def beads_left_in_container : ℕ := 4
noncomputable def beads_taken_out : ℕ := 2

theorem initial_red_beads_count :
  let total_beads_before := beads_left_in_container + beads_taken_out in
  let red_beads := total_beads_before - green_beads - brown_beads in
  red_beads = 3 :=
by
  sorry

end initial_red_beads_count_l665_665067


namespace nathan_tomato_plants_l665_665409

theorem nathan_tomato_plants (T: ℕ) : 
  5 * 14 + T * 16 = 186 * 7 / 6 + 9 * 10 :=
  sorry

end nathan_tomato_plants_l665_665409


namespace find_real_part_l665_665296

theorem find_real_part (a : ℝ) : 
    let z1 := a + 3 * complex.I
    let z2 := 3 + 4 * complex.I
    (∃ b : ℝ, z1 / z2 = b * complex.I) → a = -4 :=
by
    sorry

end find_real_part_l665_665296


namespace power_function_sum_l665_665294

-- Define the function f(x) and the conditions
def f (x : ℝ) (a b : ℝ) : ℝ := a * x ^ (2 * a + 1) - b + 1

-- Define the proposition that a + b = 2 given that f(x) is a power function
theorem power_function_sum (a b : ℝ) (h_power_function : f(2) a b = f(2) a 0) : a + b = 2 :=
by
  sorry

end power_function_sum_l665_665294


namespace length_WU_is_24sqrt13_over_13_l665_665196

-- Definitions of the given conditions
def circle (center : Point) (radius : ℝ) : Set Point := 
  {p | dist center p = radius}

variable (K : Point) (O : Point)
variable (W U : Point)
variable (ω1 : Set Point := circle K 4)
variable (ω2 : Set Point := circle O 6)
variable (HK_WU_intersects : W ∈ ω1 ∧ W ∈ ω2 ∧ U ∈ ω1 ∧ U ∈ ω2)
variable (incenter_KWU_lies_ω2 : (incenter ⟨K, W, U⟩) ∈ ω2)

-- The main theorem to be proven
theorem length_WU_is_24sqrt13_over_13
  (HK_WU_intersects : W ∈ ω1 ∧ W ∈ ω2 ∧ U ∈ ω1 ∧ U ∈ ω2)
  (incenter_KWU_lies_ω2 : (incenter ⟨K, W, U⟩) ∈ ω2) :
  dist W U = (24 * Real.sqrt 13) / 13 :=
sorry

end length_WU_is_24sqrt13_over_13_l665_665196


namespace cube_volume_l665_665902

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l665_665902


namespace distribution_scheme_count_l665_665830

-- Define the people and communities
inductive Person
| A | B | C
deriving DecidableEq, Repr

inductive Community
| C1 | C2 | C3 | C4 | C5 | C6 | C7
deriving DecidableEq, Repr

-- Define a function to count the number of valid distribution schemes
def countDistributionSchemes : Nat :=
  -- This counting is based on recognizing the problem involves permutations and combinations,
  -- the specific detail logic is omitted since we are only writing the statement, no proof.
  336

-- The main theorem statement
theorem distribution_scheme_count :
  countDistributionSchemes = 336 :=
sorry

end distribution_scheme_count_l665_665830


namespace log_equation_solution_l665_665429

theorem log_equation_solution :
  (∃ x : ℝ, log 8 x + log 2 (x ^ 3) = 9) ↔ x = 2 ^ (27 / 10) :=
by
  sorry

end log_equation_solution_l665_665429


namespace probability_of_rolling_5_is_1_over_9_l665_665108

def num_sides_dice : ℕ := 6

def favorable_combinations : List (ℕ × ℕ) :=
[(1, 4), (2, 3), (3, 2), (4, 1)]

def total_combinations : ℕ :=
num_sides_dice * num_sides_dice

def favorable_count : ℕ := favorable_combinations.length

def probability_rolling_5 : ℚ :=
favorable_count / total_combinations

theorem probability_of_rolling_5_is_1_over_9 :
  probability_rolling_5 = 1 / 9 :=
sorry

end probability_of_rolling_5_is_1_over_9_l665_665108


namespace rhombus_area_l665_665784

noncomputable def correct_diagonal_EG : ℝ := 20 - 5
noncomputable def side_length : ℝ := 40 / 4
noncomputable def half_diagonal_EG : ℝ := correct_diagonal_EG / 2

theorem rhombus_area
  (rhombus : Type)
  (a_side : ℝ)
  (d1_length : ℝ)
  (d2_length : ℝ)
  (side_eq : a_side = side_length)
  (d1_eq : d1_length = correct_diagonal_EG)
  (double_diag_d2 : 2 * (sqrt (side_length^2 - (correct_diagonal_EG / 2)^2)) = d2_length) :
  (a_side = side_length) →
  (d1_length = correct_diagonal_EG) →
  area = (d1_length * d2_length) / 2 :=
by
  sorry

end rhombus_area_l665_665784


namespace total_seating_arrangements_l665_665005

/-- Mr. and Mrs. Lopez, along with their two children and a friend, are taking a trip in their new family car. 
The car has two seats in the front row, three seats in the middle row, and two seats in the back row. 
Either Mr. Lopez or Mrs. Lopez must sit in the driver's seat. The goal is to prove that the total number 
of possible seating arrangements is 96. -/
def seating_arrangements : Nat :=
  let num_drivers := 2
  let num_passengers_front := 4
  let arrangements_middle := 3!
  let arrangements_back := 2!
  num_drivers * num_passengers_front * arrangements_middle * arrangements_back

theorem total_seating_arrangements : seating_arrangements = 96 :=
by
  unfold seating_arrangements
  simp
  sorry

end total_seating_arrangements_l665_665005


namespace saree_sale_price_l665_665042

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price - (discount * price)

theorem saree_sale_price :
  let price_175 := 175
  let discount_30 := 0.30
  let discount_25 := 0.25
  let discount_15 := 0.15
  let discount_10 := 0.10
  let first_price := apply_discount price_175 discount_30
  let second_price := apply_discount first_price discount_25
  let third_price := apply_discount second_price discount_15
  let fourth_price := apply_discount third_price discount_10
  fourth_price ≈ 70.28 :=
by
  sorry

end saree_sale_price_l665_665042


namespace beneficial_to_buy_card_breakeven_visits_l665_665998

section PartA
variables (visits_per_week : ℕ) (weeks_per_year : ℕ) (average_check : ℝ) (card_cost : ℝ) (discount_rate : ℝ)

def total_savings (visits_per_week weeks_per_year : ℕ) (average_check discount_rate : ℝ) : ℝ :=
  (visits_per_week * weeks_per_year) * (average_check * discount_rate)

theorem beneficial_to_buy_card (h1 : visits_per_week = 3) (h2 : weeks_per_year = 52) (h3 : average_check = 900) (h4 : card_cost = 30000) (h5 : discount_rate = 0.30) :
  total_savings visits_per_week weeks_per_year average_check discount_rate > card_cost :=
sorry
end PartA

section PartB
variables (average_check : ℝ) (card_cost : ℝ) (discount_rate : ℝ)

def breakeven_visits_per_year (average_check card_cost discount_rate : ℝ) : ℝ :=
  card_cost / (average_check * discount_rate)

theorem breakeven_visits (h1 : average_check = 600) (h2 : card_cost = 30000) (h3 : discount_rate = 0.30) :
  breakeven_visits_per_year average_check card_cost discount_rate = 167 :=
sorry
end PartB

end beneficial_to_buy_card_breakeven_visits_l665_665998


namespace standard_deviation_constant_addition_l665_665688

-- Assuming we have a function stddev that computes the standard deviation of a set of real numbers
def stddev (s : Finset ℝ) : ℝ := sorry

variables (a b c k : ℝ)

theorem standard_deviation_constant_addition (d : ℝ) (h_d : stddev {a, b, c} = d) (h_new : stddev {a + k, b + k, c + k} = 2) :
  d = 2 :=
by sorry

end standard_deviation_constant_addition_l665_665688


namespace balls_in_boxes_l665_665321

theorem balls_in_boxes :
  (Nat.choose (7 + 3 - 1) (3 - 1)) = 36 := by
  sorry

end balls_in_boxes_l665_665321


namespace PQ_CM_gt_AB_l665_665349

-- Define the isosceles triangle with the given properties
structure Triangle :=
(A B C M Q P : Point)  -- Points A, B, C, Midpoint M of AB, Midpoint Q of AM, Point P on AC
(base_eq : B ≠ C)      -- Base condition BC ≠ 0
(isosceles : A ≠ B ∧ B = C) -- Isosceles condition
(midpoint_M : M = (A + B) / 2) -- Midpoint of AB condition
(midpoint_Q : Q = (A + M) / 2) -- Midpoint of AM condition
(point_P : P = (3/4 * A + 1/4 * C)) -- Condition AP = 3PC

theorem PQ_CM_gt_AB (T : Triangle) : dist(T.Q, T.P) + dist(T.C, T.M) > dist(T.A, T.B) :=
sorry

end PQ_CM_gt_AB_l665_665349


namespace region_area_l665_665527

theorem region_area (r θ : ℝ) (hθ : θ = π / 2) :
  let A := (1 / 2) * r^2 * (θ - Real.sin θ)
  in (3 * A = (75 * π - 150) / 4) :=
by
  -- setting the radius and using the given conditions 
  have hr : r = 5 := by simp,
  -- subst θ which is given by the condition
  subst hθ,
  -- this calculates area using substitution of given values.
  let area := (1 / 2) * (5^2) * (π / 2 - Real.sin (π / 2)),
  have h_area: area = 25 / 2 * (π / 2 - 1) := by simp [Real.sin_pi_div_two],
  calc
    3 * (25 / 2 * (π / 2 - 1)) = 3 * (25 / 2) * ((π / 2) - 1) : by ring
                           ... = 75 / 2 * (π / 2 - 1) : by ring
                           ... = 75 / 2 * π / 2 - 75 / 2 : by ring
                           ... = (75 / 4) * π - 75 / 2 : by ring
                           ... = 75 * π / 4 - 150 / 4 : by ring
                           ... = (75 * π - 150) / 4 : by ring

end region_area_l665_665527


namespace sum_of_coefficients_binomial_expansion_l665_665459

theorem sum_of_coefficients_binomial_expansion : 
  (Finset.sum (Finset.range 8) (λ k, Nat.choose 7 k)) = 128 := 
by
  sorry

end sum_of_coefficients_binomial_expansion_l665_665459


namespace no_plane_parallel_to_both_through_point_l665_665312

structure Line : Type where
  -- Defining a line in 3D space
  p1 : EuclideanSpace ℝ 3
  p2 : EuclideanSpace ℝ 3
  h : p1 ≠ p2

structure Plane (p : EuclideanSpace ℝ 3) (n : EuclideanSpace ℝ 3) : Prop where
  -- A plane through point p with normal vector n
  exists_xy : ∃ x y : EuclideanSpace ℝ 3, n = x - y ∧ x ≠ y -- at least two points defining the plane

variables (a b : Line) (A : EuclideanSpace ℝ 3)

def is_parallel (l1 l2 : Line) : Prop :=
  -- Definition of parallel lines
  l1.p1 - l1.p2 = l2.p1 - l2.p2
  
def plane_through_point (π : Plane A (a.p1 - a.p2)) : Prop := 
  -- Definition of a plane passing through point and parallel to lines a and b
  ¬a ∈ π ∧ ¬b ∈ π ∧ is_parallel a {p1 := A, p2 := A + (a.p1 - a.p2)} ∧ is_parallel b {p1 := A, p2 := A + (b.p1 - b.p2)}

theorem no_plane_parallel_to_both_through_point (h1 : ¬a ∈ Plane A)
                                               (h2 : ¬b ∈ Plane A) 
                                               (Hb : A ≠ a.p1 ∧ A ≠ b.p1)
                                               : 
  ∃ (π : Plane A), plane_through_point π :=
begin
  -- Hypotheses that define the conditions that make a proper existence 
  sorry
end

end no_plane_parallel_to_both_through_point_l665_665312


namespace log_equation_solution_l665_665425

theorem log_equation_solution (x : ℝ) : log 8 x + log 2 (x ^ 3) = 9 → x = 2 ^ 2.7 :=
by
  sorry

end log_equation_solution_l665_665425


namespace people_own_pets_at_least_l665_665064

-- Definitions based on given conditions
def people_owning_only_dogs : ℕ := 15
def people_owning_only_cats : ℕ := 10
def people_owning_only_cats_and_dogs : ℕ := 5
def people_owning_cats_dogs_snakes : ℕ := 3
def total_snakes : ℕ := 59

-- Theorem statement to prove the total number of people owning pets
theorem people_own_pets_at_least : 
  people_owning_only_dogs + people_owning_only_cats + people_owning_only_cats_and_dogs + people_owning_cats_dogs_snakes ≥ 33 :=
by {
  -- Proof steps will go here
  sorry
}

end people_own_pets_at_least_l665_665064


namespace common_ratio_of_geometric_sequence_l665_665637

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n+1) = a n * q) → 
  (a 1 + a 5 = 17) → 
  (a 2 * a 4 = 16) → 
  (∀ i j, i < j → a i < a j) → 
  q = 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l665_665637


namespace set_equiv_l665_665693

-- Definition of the set A according to the conditions
def A : Set ℚ := { z : ℚ | ∃ p q : ℕ, z = p / (q : ℚ) ∧ p + q = 5 ∧ p > 0 ∧ q > 0 }

-- The target set we want to prove A is equal to
def target_set : Set ℚ := { 1/4, 2/3, 3/2, 4 }

-- The theorem to prove that both sets are equal
theorem set_equiv : A = target_set :=
by
  sorry -- Proof goes here

end set_equiv_l665_665693


namespace find_polynomial_l665_665232

theorem find_polynomial (p : ℤ[X]) (h : ∀ n : ℕ, 0 < n → ∣ p.eval n ∣ 2^n - 1) : p = 1 := 
sorry

end find_polynomial_l665_665232


namespace octagon_area_inscribed_in_circle_l665_665530

theorem octagon_area_inscribed_in_circle (R : ℝ) (hR : R = 4) :
  let θ := pi / 4
  let s := 2 * R * sin (θ / 2)
  let triangle_area := (1 / 2) * R^2 * sin θ
  let octagon_area := 8 * triangle_area
  octagon_area = 64 * real.sqrt 2 :=
by {
  sorry
}

end octagon_area_inscribed_in_circle_l665_665530


namespace vector_angle_condition_l665_665674

noncomputable def a (alpha : ℝ) : ℝ × ℝ := (Real.sin alpha, Real.cos alpha)
noncomputable def b (beta : ℝ) : ℝ × ℝ := (Real.cos beta, Real.sin beta)

def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def vector_sub (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ :=
  (v₁.1 - v₂.1, v₁.2 - v₂.2)

noncomputable def theta (alpha beta : ℝ) : ℝ := Real.arccos ((Real.sin alpha * Real.cos beta + Real.cos alpha * Real.sin beta) /
  (Real.sqrt ((Real.sin alpha) ^ 2 + (Real.cos alpha) ^ 2) *
   Real.sqrt ((Real.cos beta) ^ 2 + (Real.sin beta) ^ 2)))

theorem vector_angle_condition
  (alpha beta : ℝ) :
  magnitude (vector_sub (a alpha) (b beta)) = 1 ↔ theta alpha beta = Real.pi / 3 :=
by
  sorry

end vector_angle_condition_l665_665674


namespace number_of_kids_at_circus_l665_665430

theorem number_of_kids_at_circus (K A : ℕ) 
(h1 : ∀ x, 5 * x = 1 / 2 * 10 * x)
(h2 : 5 * K + 10 * A = 50) : K = 2 :=
sorry

end number_of_kids_at_circus_l665_665430


namespace cube_volume_from_surface_area_l665_665888

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l665_665888


namespace expression_simplifies_to_one_l665_665478

theorem expression_simplifies_to_one :
  ( (105^2 - 8^2) / (80^2 - 13^2) ) * ( (80 - 13) * (80 + 13) / ( (105 - 8) * (105 + 8) ) ) = 1 :=
by
  sorry

end expression_simplifies_to_one_l665_665478


namespace calc_value_of_ab_bc_ca_l665_665385

theorem calc_value_of_ab_bc_ca (a b c : ℝ) (h1 : a + b + c = 35) (h2 : ab + bc + ca = 320) (h3 : abc = 600) : 
  (a + b) * (b + c) * (c + a) = 10600 := 
by sorry

end calc_value_of_ab_bc_ca_l665_665385


namespace right_triangle_ratio_l665_665533

theorem right_triangle_ratio (x : ℝ) :
  let AB := 3 * x
  let BC := 4 * x
  let AC := (AB ^ 2 + BC ^ 2).sqrt
  let h := AC
  let AD := 16 / 21 * h / (16 / 21 + 1)
  let CD := h / (16 / 21 + 1)
  (CD / AD) = 21 / 16 :=
by 
  sorry

end right_triangle_ratio_l665_665533


namespace cube_volume_from_surface_area_l665_665919

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l665_665919


namespace tan_8pi_over_3_eq_neg_sqrt3_l665_665568

open Real

theorem tan_8pi_over_3_eq_neg_sqrt3 : tan (8 * π / 3) = -√3 :=
by
  sorry

end tan_8pi_over_3_eq_neg_sqrt3_l665_665568


namespace primes_in_arithmetic_sequence_l665_665858

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_in_arithmetic_sequence (p : ℕ) :
  is_prime p ∧ is_prime (p + 2) ∧ is_prime (p + 4) → p = 3 :=
by
  intro h
  sorry

end primes_in_arithmetic_sequence_l665_665858


namespace octagon_area_equals_l665_665529

noncomputable def area_of_inscribed_octagon (r : ℝ) : ℝ :=
  let s := 4 * real.sqrt (2 - real.sqrt 2)
  let A_triangle := 4 * real.sqrt 2
  8 * A_triangle

theorem octagon_area_equals : area_of_inscribed_octagon 4 = 32 * real.sqrt 2 :=
by
  unfold area_of_inscribed_octagon
  rw [nat.cast_bit0, nat.cast_bit0]
  rw [mul_assoc, ← mul_assoc 8]
  rw [mul_comm 8 (4 * _), ← mul_assoc]
  have : 8 * 4 * real.sqrt 2 = 32 * real.sqrt 2 := by sorry
  exact this

end octagon_area_equals_l665_665529


namespace max_distance_curve_line_l665_665039

noncomputable def curve_param_x (θ : ℝ) : ℝ := 1 + Real.cos θ
noncomputable def curve_param_y (θ : ℝ) : ℝ := Real.sin θ
noncomputable def line (x y : ℝ) : Prop := x + y + 2 = 0

theorem max_distance_curve_line 
  (θ : ℝ) 
  (x := curve_param_x θ) 
  (y := curve_param_y θ) :
  ∃ (d : ℝ), 
    (∀ t : ℝ, curve_param_x t = x ∧ curve_param_y t = y → d ≤ (abs (x + y + 2)) / Real.sqrt (1^2 + 1^2)) 
    ∧ d = (3 * Real.sqrt 2) / 2 + 1 :=
sorry

end max_distance_curve_line_l665_665039


namespace smallest_four_digit_divisible_by_43_l665_665090

theorem smallest_four_digit_divisible_by_43 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 43 = 0 ∧ n = 1032 :=
by
  use 1032
  split
  { linarith }
  split
  { linarith }
  split
  { norm_num }
  norm_num

end smallest_four_digit_divisible_by_43_l665_665090


namespace constant_term_expansion_l665_665253

theorem constant_term_expansion : 
  let C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  let expr := (x^2 + 1) * ((1 / x - 1)^5)
  (constant_term expr) = -11 :=
by
  sorry

end constant_term_expansion_l665_665253


namespace exists_irrat_gt_one_neq_floor_forall_l665_665208

def irrational_gt_one (x : ℝ) : Prop := irrational x ∧ x > 1

theorem exists_irrat_gt_one_neq_floor_forall (m n : ℕ) (m_pos : 0 < m) (n_pos : 0 < n) :
  ∃ (a b : ℝ), irrational_gt_one a ∧ irrational_gt_one b ∧ ∀ m n, m > 0 → n > 0 → (⌊a^m⌋ : ℤ) ≠ (⌊b^n⌋ : ℤ) :=
begin
  let a := 2 + real.sqrt 3,
  let b := (5 + real.sqrt 17) / 2,
  use [a, b],
  split,
  { split,
    { exact irrational_add irrational_two irrational_sqrt_3 },
    { linarith } },
  split,
  { split,
    { exact irrational_div (irrational_add irrational_two irrational_sqrt_17) (by norm_num) },
    { linarith } },
  sorry
end

end exists_irrat_gt_one_neq_floor_forall_l665_665208


namespace real_roots_of_equation_l665_665496

noncomputable def f (x : ℝ) : ℝ := Real.sin x

theorem real_roots_of_equation (a : ℝ) :
  ∃ n : ℕ, n ∈ {0, 1, 2} ∧
  ∃ (x1 x2 x3 : ℝ),
  (∀ x ∈ (0 : ℝ, Real.pi), f^2(x) + 2 * f(x) + a = 0) :=
sorry

end real_roots_of_equation_l665_665496


namespace quadrilateral_CD_length_l665_665714

theorem quadrilateral_CD_length (A B C D E : Point) (AB BD BC CD : Length)
  (h_b_a_d: Angle A B D = Angle D A C)
  (h_b_a_bd: Angle B A D = Angle B C D)
  (h1: AB = 10)
  (h2: BD = 12)
  (h3: BC = 7):
  CD = 20 :=
  sorry

end quadrilateral_CD_length_l665_665714


namespace floor_length_l665_665036

variable (b l : ℝ)

theorem floor_length :
  (l = 3 * b) →
  (3 * b ^ 2 = 128) →
  l = 19.59 :=
by
  intros h1 h2
  sorry

end floor_length_l665_665036


namespace semicircle_radius_l665_665536

noncomputable def radius_of_inscribed_semicircle (BD height : ℝ) (h_base : BD = 20) (h_height : height = 21) : ℝ :=
  let AB := Real.sqrt (21^2 + 10^2)
  let s := 2 * Real.sqrt 541
  let area := 20 * 21
  (area) / (s * 2)

theorem semicircle_radius (BD height : ℝ) (h_base : BD = 20) (h_height : height = 21)
  : radius_of_inscribed_semicircle BD height h_base h_height = 210 / Real.sqrt 541 :=
sorry

end semicircle_radius_l665_665536


namespace domain_of_function_l665_665029

theorem domain_of_function :
  ∀ x : ℝ, (0 ≤ x ∧ x * (x - 1) ≥ 0) ↔ (x = 0 ∨ x ≥ 1) :=
by sorry

end domain_of_function_l665_665029


namespace folded_rectangle_ratio_l665_665523

-- Define the conditions
def original_area (a b : ℝ) := a * b
def pentagon_area (a b : ℝ) := (7 / 10) * original_area a b

-- Define the ratio to prove
def ratio_of_sides (a b : ℝ) := a / b

-- Define the theorem to prove the ratio equals sqrt(5)
theorem folded_rectangle_ratio (a b : ℝ) (h: a > b) 
  (A1 : pentagon_area a b = (7 / 10) * original_area a b) :
  ratio_of_sides a b = real.sqrt 5 :=
  sorry

end folded_rectangle_ratio_l665_665523


namespace find_W_from_conditions_l665_665651

theorem find_W_from_conditions :
  ∀ (x y : ℝ), (y = 1 / x ∧ y = |x| + 1) → (x + y = Real.sqrt 5) :=
by
  sorry

end find_W_from_conditions_l665_665651


namespace gcd_765432_654321_l665_665581

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 :=
by 
  sorry

end gcd_765432_654321_l665_665581


namespace most_reasonable_sampling_method_l665_665073

-- Define the types of educational stages and sampling methods.
inductive EducationalStage
| primary 
| junior_high
| high

inductive SamplingMethod
| simple_random
| stratified_by_gender
| stratified_by_stage
| systematic

-- Define the conditions given in the problem.
def different_vision_conditions_by_stage : Prop :=
  ∃ (x y : EducationalStage), x ≠ y ∧ (vision_condition x ≠ vision_condition y)

def no_different_vision_conditions_by_gender : Prop :=
  ∀ (m f : Gender), vision_condition m = vision_condition f

-- Using the conditions to state the most reasonable sampling method.
theorem most_reasonable_sampling_method
  (h1 : different_vision_conditions_by_stage)
  (h2 : no_different_vision_conditions_by_gender) :
  ∃ (method : SamplingMethod), method = SamplingMethod.stratified_by_stage :=
  sorry

end most_reasonable_sampling_method_l665_665073


namespace prob_at_least_one_black_without_replacement_prob_exactly_one_black_with_replacement_l665_665702

-- Definitions for the conditions
def white_balls : ℕ := 4
def black_balls : ℕ := 2

-- Total number of balls
def total_balls : ℕ := white_balls + black_balls

-- Part (I): Without Replacement
theorem prob_at_least_one_black_without_replacement : 
  (20 - 4) / 20 = 4 / 5 :=
by sorry

-- Part (II): With Replacement
theorem prob_exactly_one_black_with_replacement : 
  (3 * 2 * 4 * 4) / (6 * 6 * 6) = 4 / 9 :=
by sorry

end prob_at_least_one_black_without_replacement_prob_exactly_one_black_with_replacement_l665_665702


namespace certain_event_l665_665097

variable (A B C D : Prop)

def event_A := A
def event_B := B
def event_C := C
def event_D := D

theorem certain_event :
  (event_A = false) ∧ (event_B = false) ∧ (event_C = false) ∧ (event_D = true) :=
by 
  sorry

end certain_event_l665_665097


namespace inclination_angle_range_l665_665458

theorem inclination_angle_range (α : ℝ) (h : -1 ≤ sin α ∧ sin α ≤ 1) :
  ∃ θ ∈ set.range (λ α, real.atan (-sin α)), θ ∈ {θ | 0 ≤ θ ∧ θ < π ∧ (θ ≤ π/4 ∨ 3*π/4 ≤ θ)} :=
sorry

end inclination_angle_range_l665_665458


namespace coach_spending_difference_l665_665787

theorem coach_spending_difference :
  let 
    -- Prices and quantities for Coach A
    basketball_price : ℝ := 29
    soccer_ball_price : ℝ := 15
    basketballs_quantity : ℕ := 10
    soccer_balls_quantity : ℕ := 5
    discount_A : ℝ := 0.05
    -- Prices and quantities for Coach B
    baseball_price : ℝ := 2.50
    baseball_bat_price : ℝ := 18
    hockey_stick_price : ℝ := 25
    mask_price : ℝ := 72
    baseballs_quantity : ℕ := 14
    hockey_sticks_quantity : ℕ := 4
    discount_voucher_B : ℝ := 10

    -- Total spent by Coach A
    total_before_discount_A := (basketball_price * basketballs_quantity + soccer_ball_price * soccer_balls_quantity : ℝ)
    discount_amount_A := total_before_discount_A * discount_A
    total_after_discount_A := total_before_discount_A - discount_amount_A

    -- Total spent by Coach B
    total_before_discount_B := (baseball_price * baseballs_quantity + baseball_bat_price + hockey_stick_price * hockey_sticks_quantity + mask_price : ℝ)
    total_after_discount_B := total_before_discount_B - discount_voucher_B
  
    -- Difference
    difference := total_after_discount_A - total_after_discount_B
  in
    difference = 131.75 :=
by 
  -- the actual mathematical proof would go here
  sorry

end coach_spending_difference_l665_665787


namespace flip_coins_all_same_side_l665_665462

theorem flip_coins_all_same_side :
  ∀ (n : ℕ), n = 2019 →
  ∃ seq : (ℕ → list (fin n → bool)),
  ∀ initial_config : fin n → bool,
  ∀ final_config : fin n → bool,
    (∀ i, 0 ≤ list.length (seq i) ∧ list.length (seq i) = i + 1) →
    (final_config = seq.foldl (λ (acc : fin n → bool) (seq_i : list (fin n → bool)), flip_coins acc seq_i) initial_config →
    (∀ k : fin n, final_config k = tt ∨ final_config k = ff)) :=
begin
  sorry
end

-- Helper function to flip the coins based on the operations defined
def flip_coins (config : fin 2019 → bool) (ops : list (fin 2019 → bool)) : fin 2019 → bool :=
  λ i, xor (config i) (foldl (λ acc op, xor acc (op i)) ff ops)


end flip_coins_all_same_side_l665_665462


namespace volumes_relation_l665_665352

-- Definitions and conditions based on the problem
variables {a b c : ℝ} (h_triangle : a > b) (h_triangle2 : b > c) (h_acute : 0 < θ ∧ θ < π)

-- The heights from vertices
variables (AD BE CF : ℝ)

-- Volumes of the tetrahedrons formed after folding
variables (V1 V2 V3 : ℝ)

-- The heights are given:
noncomputable def height_AD (BC : ℝ) (theta : ℝ) := AD
noncomputable def height_BE (CA : ℝ) (theta : ℝ) := BE
noncomputable def height_CF (AB : ℝ) (theta : ℝ) := CF

-- Using these heights and the acute nature of the triangle
noncomputable def volume_V1 (BC : ℝ) (CA : ℝ) (AB : ℝ) := V1
noncomputable def volume_V2 (BC : ℝ) (CA : ℝ) (AB : ℝ) := V2
noncomputable def volume_V3 (BC : ℝ) (CA : ℝ) (AB : ℝ) := V3

-- The theorem stating the relationship between volumes
theorem volumes_relation
  (h_triangle: a > b)
  (h_triangle2: b > c)
  (h_acute: 0 < θ ∧ θ < π)
  (h_volumes: V1 > V2 ∧ V2 > V3):
  V1 > V2 ∧ V2 > V3 :=
sorry

end volumes_relation_l665_665352


namespace angle_between_diagonal_and_base_side_of_prism_l665_665035

-- We define a regular triangular prism with edge length 1 and square lateral faces.
structure RegularTriangularPrism :=
  (edge_length : ℝ := 1)
  (is_regular : Prop := true)
  (lateral_faces_are_squares : Prop := true)

-- Definitions to capture the conditions
def regularTriangularPrism : RegularTriangularPrism := { edge_length := 1, is_regular := true, lateral_faces_are_squares := true }

-- Main statement to prove the angle between the diagonal of a lateral face and a non-intersecting side of the base
theorem angle_between_diagonal_and_base_side_of_prism (P : RegularTriangularPrism) :
  P.is_regular → P.lateral_faces_are_squares → ∀ (α : ℝ), α = Real.arccos (Real.sqrt 2 / 4) :=
by
  intro h1 h2
  existsi Real.arccos (Real.sqrt 2 / 4)
  sorry

end angle_between_diagonal_and_base_side_of_prism_l665_665035


namespace point_farthest_from_origin_l665_665480

theorem point_farthest_from_origin :
  let dist (p : ℝ × ℝ) := Real.sqrt (p.1^2 + p.2^2) in
  ∀ p ∈ [{(0, 5)}, {(1, 2)}, {(3, -4)}, {(6, 0)}, {(-1, -2)}],
    dist (6, 0) ≥ dist p :=
by
  intros p hp
  let dist := λ (p : ℝ × ℝ), Real.sqrt (p.1^2 + p.2^2)
  have h : p = (0, 5) ∨ p = (1, 2) ∨ p = (3, -4) ∨ p = (6, 0) ∨ p = (-1, -2) := sorry
  cases h
  · simp [h, dist]
  · cases h
    · simp [h, dist]
    · cases h
      · simp [h, dist]
      · cases h
        · simp [h, dist]
        · simp [h, dist]
  sorry

end point_farthest_from_origin_l665_665480


namespace biker_bob_additional_north_distance_is_approx_l665_665561

-- Defining the distances given in the problem
def west_distance : ℝ := 30
def east_distance : ℝ := 15
def initial_north_distance : ℝ := 6
def total_distance : ℝ := 28.30194339616981

-- Using the Pythagorean theorem to compute the additional northward distance
def additional_north_distance : ℝ :=
  real.sqrt(total_distance^2 - (west_distance - east_distance)^2) - initial_north_distance

-- Proving the additional northward distance is approximately 17.98 miles
theorem biker_bob_additional_north_distance_is_approx : 
  abs (additional_north_distance - 17.98) < 0.01 := 
sorry

end biker_bob_additional_north_distance_is_approx_l665_665561


namespace cube_volume_from_surface_area_example_cube_volume_l665_665968

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l665_665968


namespace equation_of_parabola_passing_through_points_l665_665631

noncomputable def parabola (x : ℝ) (b c : ℝ) : ℝ :=
  x^2 + b * x + c

theorem equation_of_parabola_passing_through_points :
  ∃ (b c : ℝ), 
    (parabola 0 b c = 5) ∧ (parabola 3 b c = 2) ∧
    (∀ x, parabola x b c = x^2 - 4 * x + 5) := 
by
  sorry

end equation_of_parabola_passing_through_points_l665_665631


namespace percentage_time_in_park_l665_665741

/-- Define the number of trips Laura takes to the park. -/
def number_of_trips : ℕ := 6

/-- Define time spent at the park per trip in hours. -/
def time_at_park_per_trip : ℝ := 2

/-- Define time spent walking per trip in hours. -/
def time_walking_per_trip : ℝ := 0.5

/-- Define the total time for all trips. -/
def total_time_for_all_trips : ℝ := (time_at_park_per_trip + time_walking_per_trip) * number_of_trips

/-- Define the total time spent in the park for all trips. -/
def total_time_in_park : ℝ := time_at_park_per_trip * number_of_trips

/-- Prove that the percentage of the total time spent in the park is 80%. -/
theorem percentage_time_in_park : total_time_in_park / total_time_for_all_trips * 100 = 80 :=
by
  sorry

end percentage_time_in_park_l665_665741


namespace triangle_ACB_probability_l665_665465

noncomputable def probability_acute_triangle : ℝ := 
let circle : Type := ℝ // (λ x, x = 2 * π) in
let random_points (n : ℕ) : list ℝ := 
  range (n) |>.map (λ _, real.uniform 0 (2 * π)) in
let A, B, C := random_points 3 in
let triangle_is_acute (A B C : ℝ) : Prop :=
  abs (B - C) < π ∧ abs (B - A) < π ∧ abs (C - A) < π in
if triangle_is_acute A B C then 1/4 else 0/4

theorem triangle_ACB_probability :
  (∀ (A B C : ℝ), probability_acute_triangle = 1 / 4) := sorry

end triangle_ACB_probability_l665_665465


namespace alpha_lt_beta_of_acute_l665_665323

open Real

theorem alpha_lt_beta_of_acute (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : 2 * sin α = sin α * cos β + cos α * sin β) : α < β :=
by
  sorry

end alpha_lt_beta_of_acute_l665_665323


namespace value_of_expression_l665_665484

variables {a b c d e f : ℝ}

theorem value_of_expression
  (h1 : a * b * c = 65)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 1 / 4 :=
sorry

end value_of_expression_l665_665484


namespace point_inside_all_triangles_l665_665002

-- Define vertices of the triangle
variables {A1 A2 A3 : Point}

-- (Condition) Define point A4
def A4 := point_dividing_unequal_thirds A1 A2 1 2  -- Closer to A1

-- (Condition) Define point An+1 for n >= 4
def A (n : Nat) : Point
| 4 => A4
| (n + 1) => point_dividing_unequal_thirds (A (n - 1)) (A (n - 2)) 1 2  -- Closer to A_{n-2}

-- We need to prove that there exists a point (like T) inside each triangle AnAn+1An+2 for n >= 1
theorem point_inside_all_triangles : ∃ (T : Point), ∀ n ≥ 1, is_inside_triangle T (A n) (A (n+1)) (A (n+2)) :=
sorry

end point_inside_all_triangles_l665_665002


namespace Maxwell_age_l665_665344

theorem Maxwell_age :
  ∀ (sister_age maxwell_age : ℕ),
    (sister_age = 2) → 
    (maxwell_age + 2 = 2 * (sister_age + 2)) →
    (maxwell_age = 6) :=
by
  intros sister_age maxwell_age h1 h2
  -- Definitions and hypotheses come directly from conditions
  sorry

end Maxwell_age_l665_665344


namespace bicycle_profit_theorem_l665_665535

def bicycle_profit_problem : Prop :=
  let CP_A : ℝ := 120
  let SP_C : ℝ := 225
  let profit_percentage_B : ℝ := 0.25
  -- intermediate calculations
  let CP_B : ℝ := SP_C / (1 + profit_percentage_B)
  let SP_A : ℝ := CP_B
  let Profit_A : ℝ := SP_A - CP_A
  let Profit_Percentage_A : ℝ := (Profit_A / CP_A) * 100
  -- final statement to prove
  Profit_Percentage_A = 50

theorem bicycle_profit_theorem : bicycle_profit_problem := 
by
  sorry

end bicycle_profit_theorem_l665_665535


namespace cost_price_of_article_l665_665187

theorem cost_price_of_article (SP : ℝ) (profit_percent : ℝ) (CP : ℝ) : 
  SP = 800 → profit_percent = 25 → CP = SP * 4 / 5 := 
by
  intros hSP hProfit
  rw [hSP, hProfit]
  sorry

end cost_price_of_article_l665_665187


namespace probability_common_letters_l665_665585

open Set

def letters_GEOMETRY : Finset Char := {'G', 'E', 'O', 'M', 'T', 'R', 'Y'}
def letters_RHYME : Finset Char := {'R', 'H', 'Y', 'M', 'E'}

def common_letters : Finset Char := letters_GEOMETRY ∩ letters_RHYME

theorem probability_common_letters :
  (common_letters.card : ℚ) / (letters_GEOMETRY.card : ℚ) = 1 / 2 := by
  sorry

end probability_common_letters_l665_665585


namespace distance_sum_range_l665_665013

-- Define points as tuples
def B := (0, 0)
def D := (6, 8)
def A := (15, 0)

-- Calculate Euclidean distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Distance BD
def BD := distance B D

-- Distance AD
def AD := distance A D

-- Theorem: AD + BD falls between 22 and 23
theorem distance_sum_range (B D A : ℝ × ℝ) (BD AD : ℝ) (hBD : BD = distance B D) (hAD : AD = distance A D) : 
  (22 < AD + BD) ∧ (AD + BD < 23) := 
by
  -- Proof is skipped
  sorry

end distance_sum_range_l665_665013


namespace cube_volume_l665_665900

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l665_665900


namespace square_of_number_ending_in_5_l665_665489

theorem square_of_number_ending_in_5 (d : ℤ) : (10 * d + 5)^2 = 100 * d * (d + 1) + 25 :=
by
  sorry

end square_of_number_ending_in_5_l665_665489


namespace length_major_axis_eq_six_l665_665448

-- Define the given equation of the ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 9) = 1

-- The theorem stating the length of the major axis
theorem length_major_axis_eq_six (x y : ℝ) (h : ellipse_equation x y) : 
  2 * (Real.sqrt 9) = 6 :=
by
  sorry

end length_major_axis_eq_six_l665_665448


namespace find_largest_t_l665_665257

theorem find_largest_t (t : ℝ) : 
  (15 * t^2 - 38 * t + 14) / (4 * t - 3) + 6 * t = 7 * t - 2 → t ≤ 1 := 
by 
  intro h
  sorry

end find_largest_t_l665_665257


namespace polynomial_roots_l665_665240

theorem polynomial_roots :
  (∀ x, x^3 - 3*x^2 - x + 3 = 0 ↔ (x = 1 ∨ x = -1 ∨ x = 3)) :=
by
  intro x
  split
  {
    intro h
    have h1 : x = 1 ∨ x = -1 ∨ x = 3
    {
      sorry
    }
    exact h1
  }
  {
    intro h
    cases h
    {
      rw h
      simp
    }
    {
      cases h
      {
        rw h
        simp
      }
      {
        rw h
        simp
      }
    }
  }

end polynomial_roots_l665_665240


namespace cube_volume_from_surface_area_l665_665866

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l665_665866


namespace inscribed_shapes_area_ratio_l665_665475

theorem inscribed_shapes_area_ratio (r : ℝ) (ht : t = (2 * r) / √3) (hs : s = √2 * r) :
  let A_t := (√3 / 4) * t^2
  let A_s := s^2
  A_t / A_s = √3 / 3 :=
by
  sorry

end inscribed_shapes_area_ratio_l665_665475


namespace area_quadrilateral_midpoints_l665_665436

theorem area_quadrilateral_midpoints (S d1 d2 : ℝ) (α : ℝ) 
  (hS : S = (1/2) * d1 * d2 * real.sin α) : 
  let S_midpoints := (1/2) * S in
  S_midpoints = (1/2) * S :=
by sorry

end area_quadrilateral_midpoints_l665_665436


namespace construct_pq_parallel_ac_l665_665633

variables {A B C P Q M : Point}

theorem construct_pq_parallel_ac (hABC : Triangle A B C) 
    (hM_on_AC : Between A M C ∧ (M ≠ A ∧ M ≠ C)) : 
    ∃ P Q : Point, (OnLine P (Line A B)) ∧ (OnLine Q (Line B C)) 
                ∧ Parallel (Line P Q) (Line A C) 
                ∧ RightAngle (Angle P M Q) :=
by
  -- The detailed proof will go here.
  sorry

end construct_pq_parallel_ac_l665_665633


namespace weight_of_B_l665_665105

theorem weight_of_B (A B C : ℕ) (h1 : A + B + C = 90) (h2 : A + B = 50) (h3 : B + C = 56) : B = 16 := 
sorry

end weight_of_B_l665_665105


namespace cube_volume_from_surface_area_l665_665952

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l665_665952


namespace investor_bought_shares_at_30_l665_665510

/-- The face value of the share is Rs. 60, 
    the company pays a 12.5% dividend, 
    and the investor gets a 25% return on investment.
    
    Prove that the investor bought the shares at Rs. 30 per share. -/
theorem investor_bought_shares_at_30 
  (face_value : ℝ)
  (company_dividend_percentage : ℝ)
  (investor_roi_percentage : ℝ)
  (dividend_per_share : ℝ) (P : ℝ) : 
  face_value = 60 → 
  company_dividend_percentage = 0.125 →
  investor_roi_percentage = 0.25 →
  dividend_per_share = company_dividend_percentage * face_value → 
  dividend_per_share = investor_roi_percentage * P → 
  P = 30 := 
by
  intros hp hv hd ho hd_eq hp_eq
  sorry

end investor_bought_shares_at_30_l665_665510


namespace num_valid_four_digit_numbers_l665_665316

-- Define the relevant conditions and sets
def valid_first_digits : set ℕ := {1, 3, 5}
def valid_second_digits : set ℕ := {2, 4, 6}

def is_valid_four_digit_number (d1 d2 d3 d4 : ℕ) : Prop :=
  d1 ∈ valid_first_digits ∧
  d2 ∈ valid_second_digits ∧
  d2 ≠ d1 ∧
  d3 > d1 ∧
  d3 ≠ d1 ∧ d3 ≠ d2 ∧
  d4 ≠ d1 ∧ d4 ≠ d2 ∧ d4 ≠ d3

-- The main theorem stating the number of valid four-digit numbers
theorem num_valid_four_digit_numbers : ∃ n, n = 357 ∧
  ∀ (d1 d2 d3 d4 : ℕ), is_valid_four_digit_number d1 d2 d3 d4 → 
  (d1, d2, d3, d4) ∈ finset.univ.filter (λ x, is_valid_four_digit_number x.1 x.2 x.3 x.4) :=
sorry

end num_valid_four_digit_numbers_l665_665316


namespace alice_probability_at_least_one_multiple_of_4_l665_665552

def probability_multiple_of_4 : ℚ :=
  1 - (45 / 60)^3

theorem alice_probability_at_least_one_multiple_of_4 :
  probability_multiple_of_4 = 37 / 64 :=
by
  sorry

end alice_probability_at_least_one_multiple_of_4_l665_665552


namespace maximum_prism_volume_l665_665440

theorem maximum_prism_volume 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h2 : b + c = 8) 
  (h3 : b = 2) : 
  ∃ V_max : ℝ, V_max = 32 := 
begin
  sorry
end

end maximum_prism_volume_l665_665440


namespace speed_of_stream_l665_665519

-- Define the problem conditions
variables (b s : ℝ)
axiom cond1 : 21 = b + s
axiom cond2 : 15 = b - s

-- State the theorem
theorem speed_of_stream : s = 3 :=
sorry

end speed_of_stream_l665_665519


namespace problem1_problem2_problem3_l665_665306

noncomputable def f (x : ℝ) : ℝ := Real.log x - x
noncomputable def a_n (n : ℕ) : ℝ := 1 + 1 / (2 : ℝ) ^ n

theorem problem1 : ∀ x ∈ (Set.Ici 0), f x ≤ f 1 := sorry

theorem problem2 {k : ℝ} : (∀ x ∈ (Set.Ioi 2), x * f x + x ^ 2 - k * x + k > 0) → k ≤ 2 * Real.log 2 := sorry

theorem problem3 : ∀ n : ℕ, (∏ i in finset.range (n + 1), a_n i) < Real.exp 1 := sorry

end problem1_problem2_problem3_l665_665306


namespace heart_biscuits_for_each_dog_l665_665407

-- Define the total number of items Mrs. Heine buys
def total_items : ℕ := 12

-- Define the number of sets of puppy boots
def boots_per_dog : ℕ := 1
def total_boot_sets : ℕ := 2 * boots_per_dog

-- Define the number of heart biscuits each dog gets
def biscuits_per_dog (B : ℕ) : Prop :=
  2 * B + total_boot_sets = total_items

-- State the theorem
theorem heart_biscuits_for_each_dog : ∃ B : ℕ, biscuits_per_dog B ∧ B = 4 :=
begin
  existsi 4,
  split,
  {
    unfold biscuits_per_dog, 
    norm_num,
  },
  {
    refl,
  },
end

end heart_biscuits_for_each_dog_l665_665407


namespace Nadal_wins_championship_probability_l665_665709

open Probability

noncomputable def probability_of_Nadal_winning_championship : ℝ :=
  let p := 2 / 3
  let q := 1 / 3
  let outcomes (k : ℕ) := Nat.choose (3 + k) k
  let individual_prob (k : ℕ) := (outcomes k) * (p ^ 4) * (q ^ k)
  (individual_prob 0) + (individual_prob 1) + (individual_prob 2) + (individual_prob 3)

theorem Nadal_wins_championship_probability :
  round (100 * probability_of_Nadal_winning_championship) = 89 :=
sorry

end Nadal_wins_championship_probability_l665_665709


namespace intersection_eq_singleton_zero_l665_665310

def setA : Set ℝ := { x | x^2 + 2x = 0 }
def setB : Set ℝ := { x | x^2 - 2x ≤ 0 }

theorem intersection_eq_singleton_zero : setA ∩ setB = {0} :=
by
  sorry

end intersection_eq_singleton_zero_l665_665310


namespace find_h_l665_665701

open Real

-- Given conditions
variables (A B C D : Point) -- points denoting the vertices of the triangle and the foot of the altitude from B
variables (h : ℝ) -- the length of the altitude from B to AC
variables (θ_BCA θ_BAC : ℝ) -- angles at C and A respectively

-- Define tangent values for the given angles
def tan_angle_BCA : Prop := tan θ_BCA = 1
def tan_angle_BAC : Prop := tan θ_BAC = 1 / 7

-- Define the perimeter condition
def perimeter_condition (P : ℝ) : Prop := 
  P = 24 + 18 * sqrt 2

-- Define the altitude intersection condition
def altitude_condition (m : ℝ) : Prop :=
  m = h

-- The final proof obligation
theorem find_h (P : ℝ) (h_val : ℝ) :
  tan_angle_BCA ∧ tan_angle_BAC ∧ perimeter_condition P ∧ altitude_condition h_val →
  h_val = 3 :=
sorry

end find_h_l665_665701


namespace price_reduction_l665_665149

theorem price_reduction (x y : ℕ) (h1 : (13 - x) * y = 781) (h2 : y ≤ 100) : x = 2 :=
sorry

end price_reduction_l665_665149


namespace hexagons_in_50th_ring_l665_665201

theorem hexagons_in_50th_ring : ∀ (n : ℕ), (n = 50) → (6 * n = 300) :=
by
  intros n hn
  rw hn
  simp
  exact Nat.mul_eq_one₀ 6 50 (by norm_num)


end hexagons_in_50th_ring_l665_665201


namespace find_angle_y_l665_665717

-- Definitions related to the problem
def is_parallel (m n : ℝ → ℝ) : Prop :=
∀ x y, m x = m y ∧ n x = n y → x = y

def corresponding_angles (θ₁ θ₂ : ℝ) : Prop :=
θ₁ = θ₂

def supplementary (θ₁ θ₂ : ℝ) : Prop :=
θ₁ + θ₂ = 180

variable (m n : ℝ → ℝ) -- Representing lines m and n
variable (θ : ℝ) -- Representing the known angle 40 degrees
variable (y : ℝ) -- Representing the angle y to be found

-- Given conditions
axiom parallel_lines : is_parallel m n
axiom transversal_intersects : corresponding_angles θ 40
axiom known_angle_is_40 : θ = 40

-- Proof statement that angle y is 140 degrees
theorem find_angle_y : supplementary θ y → y = 140 :=
begin
  intros,
  sorry
end

end find_angle_y_l665_665717


namespace cube_volume_from_surface_area_l665_665959

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l665_665959


namespace amaya_movie_watching_time_l665_665168

theorem amaya_movie_watching_time :
  let uninterrupted_time_1 := 35
  let uninterrupted_time_2 := 45
  let uninterrupted_time_3 := 20
  let rewind_time_1 := 5
  let rewind_time_2 := 15
  let total_uninterrupted := uninterrupted_time_1 + uninterrupted_time_2 + uninterrupted_time_3
  let total_rewind := rewind_time_1 + rewind_time_2
  let total_time := total_uninterrupted + total_rewind
  total_time = 120 := by
  sorry

end amaya_movie_watching_time_l665_665168


namespace map_distance_eq_311_9024_inch_l665_665009

theorem map_distance_eq_311_9024_inch 
  (actual_distance_between_mountains : ℝ)
  (ram_distance_on_map : ℝ)
  (ram_actual_distance : ℝ)
  :
  actual_distance_between_mountains = 136 →
  ram_distance_on_map = 34 →
  ram_actual_distance = 14.82 →
  let scale := ram_distance_on_map / ram_actual_distance in
  let distance_on_map := scale * actual_distance_between_mountains in
  distance_on_map ≈ 311.9024 :=
begin
  intros h1 h2 h3,
  let scale := ram_distance_on_map / ram_actual_distance,
  let distance_on_map := scale * actual_distance_between_mountains,
  show distance_on_map ≈ 311.9024,
  sorry
end

end map_distance_eq_311_9024_inch_l665_665009


namespace minimum_total_trips_l665_665813

theorem minimum_total_trips :
  ∃ (x y : ℕ), (31 * x + 32 * y = 5000) ∧ (x + y = 157) :=
by
  sorry

end minimum_total_trips_l665_665813


namespace correct_statement_of_geometric_sequence_l665_665832

variable {k r : ℝ}
variable {m : ℤ}

theorem correct_statement_of_geometric_sequence (a_n : ℕ → ℝ) (S_n : ℕ → ℝ)
  (h1 : ∀ n, S_n n = k ^ n + r ^ m)
  (h2 : ∀ n, a_n n = S_n (n + 1) - S_n n)
  (h3 : ∀ n, a_n (n + 1) / a_n n = a_n (n + 2) / a_n (n + 1))
  (h4 : k ≠ 0 ∧ k ≠ 1) :
  r = -1 ∧ odd m := 
sorry

end correct_statement_of_geometric_sequence_l665_665832


namespace cube_volume_from_surface_area_l665_665887

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l665_665887


namespace prime_sum_correct_l665_665603

open Nat

def is_valid_prime (p : ℕ) : Prop :=
  p.Prime ∧ p ≡ 3 [MOD 5] ∧ p ≡ 2 [MOD 7]

noncomputable def prime_sum_1_to_200 : ℕ :=
  ∑ p in (filter (λ n, is_valid_prime n) (range 201)), id p

theorem prime_sum_correct : prime_sum_1_to_200 = 383 :=
  sorry

end prime_sum_correct_l665_665603


namespace interest_paid_percentage_l665_665010

noncomputable def percent_interest_paid (purchase_price down_payment monthly_rate months : ℝ) : ℝ :=
  let P := purchase_price - down_payment
  let n := 1
  let t := months / 12
  let A := P * (1 + monthly_rate / n) ^ (n * t)
  let interest := A - P
  (interest / purchase_price) * 100

theorem interest_paid_percentage :
  percent_interest_paid 130 30 0.015 18 ≈ 1.8 :=
by
  sorry

end interest_paid_percentage_l665_665010


namespace function_expression_of_y_maximum_yield_l665_665071

-- Conditions
def average_yield (x : ℕ) : ℝ :=
  if 2 ≤ x ∧ x ≤ 8 then (-0.5) * x + 5 else 0

def total_yield (x : ℕ) : ℝ :=
  x * average_yield x

-- Part 1: Function expression of y in terms of x
theorem function_expression_of_y :
  ∀ x : ℕ, 2 ≤ x ∧ x ≤ 8 → average_yield x = -0.5 * x + 5 :=
by
  intros x hx
  sorry

-- Part 2: Maximum yield calculation
theorem maximum_yield :
  ∃ x : ℕ, 2 ≤ x ∧ x ≤ 8 ∧
    (∀ y : ℕ, 2 ≤ y ∧ y ≤ 8 → total_yield y ≤ total_yield x) ∧
    total_yield x = 12.5 :=
by
  use 5
  split
  { exact ⟨by norm_num, by norm_num⟩ }
  split
  { intros y hy
    sorry }
  { sorry }

end function_expression_of_y_maximum_yield_l665_665071


namespace product_of_possible_values_l665_665677

theorem product_of_possible_values (x : ℝ) (h : (x + 3) * (x - 4) = 18) : ∃ a b, x = a ∨ x = b ∧ a * b = -30 :=
by 
  sorry

end product_of_possible_values_l665_665677


namespace travel_two_roads_l665_665494

theorem travel_two_roads (cities : Finset ℕ) (incoming outgoing : ℕ → Finset ℕ)
  (h_cities_size : cities.card = 101)
  (h_incoming : ∀ c ∈ cities, (incoming c).card = 50)
  (h_outgoing : ∀ c ∈ cities, (outgoing c).card = 50)
  (h_incoming_in_cities : ∀ c ∈ cities, ∀ c' ∈ incoming c, c' ∈ cities)
  (h_outgoing_in_cities : ∀ c ∈ cities, ∀ c' ∈ outgoing c, c' ∈ cities) :
  ∀ A B ∈ cities, ∃ C ∈ cities, C ∈ outgoing A ∧ B ∈ outgoing C :=
by {
    sorry
}

end travel_two_roads_l665_665494


namespace rahul_matches_l665_665418

theorem rahul_matches (avg_1 avg_2 runs : ℕ) (m : ℕ) 
  (h1 : avg_1 = 52) (h2 : runs = 78) (h3 : avg_2 = 54) 
  (h4 : (avg_1 * m + runs) / (m + 1) = avg_2) : m = 12 := 
by
  subst h1
  subst h2
  subst h3
  rw [(52 * m + 78) / (m + 1), 54] at h4
  sorry

end rahul_matches_l665_665418


namespace problem1_problem2_problem3_problem4_l665_665194

theorem problem1 : (-7) - | -9 | - (-11) - 3 = -8 := 
by {
    have h1 : | -9 | = 9, by sorry,
    have h2 : (-7) - 9 + 11 - 3 = -16 + 8, by sorry,
    have h3 : -16 + 8 = -8, by sorry,
    exact h3,
}

theorem problem2 : 5.6 + (-0.9) + 4.4 + (-8.1) = 1 := 
by {
    have h1 : 5.6 + 4.4 = 10.0, by sorry,
    have h2 : -0.9 - 8.1 = -9.0, by sorry,
    have h3 : 10.0 + (-9.0) = 1.0, by sorry,
    exact h3,
}

theorem problem3 : (-1/6) + (1/3) + (-1/12) = 1/12 := 
by {
    have h1 : -1/6 = -2/12, by sorry,
    have h2 : 1/3 = 4/12, by sorry,
    have h3 : -2/12 + 4/12 - 1/12 = (4 - 2 - 1) / 12, by sorry,
    have h4 : (4 - 2 - 1) / 12 = 1/12, by sorry,
    exact h4,
}

theorem problem4 : (2/5) - | -1.5 | - 2.25 - (-2.75) = -0.6 := 
by {
    have h1 : | -1.5 | = 1.5, by sorry,
    have h2 : 2/5 = 0.4, by sorry,
    have h3 : 0.4 - 1.5 - 2.25 + 2.75 = -1.1 - 2.25 + 2.75, by sorry,
    have h4 : -1.1 - 2.25 + 2.75 = -0.6, by sorry,
    exact h4,
}

end problem1_problem2_problem3_problem4_l665_665194


namespace max_value_of_expression_l665_665384

noncomputable def max_value_expression (a c x : ℝ) : ℝ :=
  3 * (a - x) * (x + real.sqrt (x^2 + c^2))

theorem max_value_of_expression (a c : ℝ) (ha : 0 < a) (hc : 0 < c) :
  ∃ x, max_value_expression a c x = (3 / 2) * (a^2 + c^2) :=
by
  sorry

end max_value_of_expression_l665_665384


namespace total_shaded_area_correct_l665_665716

-- Definitions of given conditions and entities
structure Square := (side : ℝ)
structure TruncatedPyramid := 
  (base : Square) 
  (top : Square) 
  (height : ℝ) 
  (side_length_eq : base.side ≠ top.side)

def ABCD : Square := Square.mk 7
def EFGH : Square := Square.mk 1

-- Define the truncated pyramid data
def pyramid : TruncatedPyramid := 
  { base := ABCD
  , top := EFGH
  , height := 4
  , side_length_eq := by sorry
  }

-- Condition about trapezoidal faces being partially shaded
def trapezoids_partially_shaded : Prop := true -- Placeholder, refine as needed

-- Question to be proved: total shaded area
def total_shaded_area (p : TruncatedPyramid) : ℝ := 15/4 * 5    -- 15/4 * h; h = 5

-- The Lean theorem statement
theorem total_shaded_area_correct : total_shaded_area pyramid = 75 / 4 := by sorry

end total_shaded_area_correct_l665_665716


namespace center_of_circle_l665_665721

-- Define the circle in polar coordinates
def circle_polar (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

-- Define the center of the circle in polar coordinates
def center_polar (ρ θ : ℝ) : Prop := (ρ = 1 ∧ θ = Real.pi / 2) ∨ (ρ = 1 ∧ θ = 3 * Real.pi / 2)

-- The theorem states that the center of the given circle in polar coordinates is (1, π/2) or (1, 3π/2)
theorem center_of_circle : ∃ (ρ θ : ℝ), circle_polar ρ θ → center_polar ρ θ :=
by
  -- The center of the circle given the condition in polar coordinate system is (1, π/2) or (1, 3π/2)
  sorry

end center_of_circle_l665_665721


namespace min_divisors_f_20_l665_665391

theorem min_divisors_f_20 (f : Fin 22 → ℕ)
  (h : ∀ (m n : Fin 22), m * n ∣ f m + f n) :
  ∃ d, d = 2016 ∧ d = Nat.divisors_count (f ⟨19, by norm_num⟩) :=
by
  sorry

end min_divisors_f_20_l665_665391


namespace pentagon_largest_angle_l665_665027

theorem pentagon_largest_angle (x : ℝ) (h : 2 * x + 3 * x + 4 * x + 5 * x + 6 * x = 540) : 6 * x = 162 :=
sorry

end pentagon_largest_angle_l665_665027


namespace polynomial_root_sum_eq_48_l665_665040

theorem polynomial_root_sum_eq_48 {r s t : ℕ} (h1 : r * s * t = 2310) 
  (h2 : r > 0) (h3 : s > 0) (h4 : t > 0) : r + s + t = 48 :=
sorry

end polynomial_root_sum_eq_48_l665_665040


namespace cube_volume_l665_665932

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l665_665932


namespace max_rooks_l665_665862

/-
Given:
1. A 300x300 chessboard.
2. Each rook attacks all squares in its row and column unless blocked by another piece.
3. Each rook must attack at most one other rook.

Prove:
The maximum number of rooks that can be placed on this chessboard is 400.
-/

theorem max_rooks (n : ℕ) (h_n : n = 300) :
  ∃ r : ℕ, r = 400 ∧ ∀ placement : matrix (fin n) (fin n) bool, 
    (∀ i j, placement i j → ∀ i', i' ≠ i → ¬placement i' j ∨ ∀ j', j' ≠ j → ¬placement i j') →
    (∃ count, count ≤ r) :=
by
  sorry

end max_rooks_l665_665862


namespace cube_volume_of_surface_area_l665_665984

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l665_665984


namespace circles_max_ab_l665_665672

theorem circles_max_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ (x y : ℝ), (x + a)^2 + (y - 2)^2 = 1 ∧ (x - b)^2 + (y - 2)^2 = 4) →
  a + b = 3 →
  ab ≤ 9 / 4 := 
  by
  sorry

end circles_max_ab_l665_665672


namespace new_water_height_l665_665375

-- Definitions for the given conditions
def tank_width : ℝ := 50
def tank_length : ℝ := 16
def tank_height : ℝ := 25
def initial_water_height : ℝ := 15
def cube_side : ℝ := 10

-- The theorem statement that proves the new height of the water
theorem new_water_height : 
  let cube_volume := cube_side ^ 3 in
  let initial_water_volume := tank_width * tank_length * initial_water_height in
  let new_water_volume := initial_water_volume + cube_volume in
  let base_area := tank_width * tank_length in
  new_water_volume / base_area = 16.25 := 
by
  sorry

end new_water_height_l665_665375


namespace greatest_four_digit_multiple_of_23_l665_665860

theorem greatest_four_digit_multiple_of_23 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 23 = 0 ∧ n = 9978 :=
by
  use 9978
  sorry

end greatest_four_digit_multiple_of_23_l665_665860


namespace scientific_notation_of_16907_l665_665699

theorem scientific_notation_of_16907 :
  16907 = 1.6907 * 10^4 :=
sorry

end scientific_notation_of_16907_l665_665699


namespace function_satisfies_equation_l665_665735

noncomputable def f (x : ℝ) : ℝ :=
  if x = 0.5 then 0.5 else 1 / (0.5 - x)

theorem function_satisfies_equation :
  ∀ x : ℝ, f x + (0.5 + x) * f (1 - x) = 1 :=
by
  intro x
  unfold f
  split_ifs
  case h =>
    sorry -- Proof when x = 0.5
  case h_1 =>
    sorry -- Proof when x ≠ 0.5

end function_satisfies_equation_l665_665735


namespace cube_volume_from_surface_area_l665_665873

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l665_665873


namespace jordan_rectangle_width_l665_665483

theorem jordan_rectangle_width (a_c : ℕ) (len_c : ℕ) (wid_c : ℕ) (len_j : ℕ) :
  (len_c = 5) → (wid_c = 24) → (len_j = 4) →  a_c = len_c * wid_c →  (len_j * (a_c / len_j) = a_c) →
  (a_c = 120) → (a_c / len_j = 30) :=
by
  intros len_c_eq wid_c_eq len_j_eq area_c_eq equal_areas area_value 
  rw [len_c_eq, wid_c_eq] at area_c_eq
  have area_c_120 : 5 * 24 = 120 := by norm_num
  rw area_c_120 at area_c_eq
  have len_j_4 : 4 = len_j := by rw len_j_eq
  have vanished : (120 / 4 = 30) := by norm_num
  rw vanished
  apply rfl
  sorry

-- end statement without the "sorry"

end jordan_rectangle_width_l665_665483


namespace equations_of_sides_of_triangle_ABC_l665_665284

noncomputable def point : Type := (ℝ × ℝ)

def A : point := (-2, 2)
def B : point := (-2, -2)
def C : point := (6, 6)

def line (p1 p2 : point) : Type :=
{ l : ℝ × ℝ × ℝ // l.1 * (p1.1) + l.2 * (p1.2) + l.3 = 0 ∧ l.1 * (p2.1) + l.2 * (p2.2) + l.3 = 0 }

def line_AB : line A B := ⟨(1, 0, 2), by sorry⟩
def line_AC : line A C := ⟨(1, -2, 6), by sorry⟩
def line_BC : line B C := ⟨(1, -1, 0), by sorry⟩

theorem equations_of_sides_of_triangle_ABC :
  ∃ (L1 L2 L3 : line A B ∧ line A C ∧ line B C),
    L1.1 = (1, 0, 2) ∧
    L2.1 = (1, -2, 6) ∧
    L3.1 = (1, -1, 0) :=
begin
  use line_AB,
  use line_AC,
  use line_BC,
  split,
  { refl },
  split,
  { refl },
  { refl }
end

end equations_of_sides_of_triangle_ABC_l665_665284


namespace percentage_increase_breadth_l665_665818

theorem percentage_increase_breadth {L B : Real} (hL : L > 0) (hB : B > 0) 
  (increase_length : 1.30 * L)
  (increase_area : 1.885 * (L * B)) :
  ∃ p : ℝ, 1.30 * (1 + p / 100) = 1.885 ∧ p = 45 :=
by {
  sorry
}

end percentage_increase_breadth_l665_665818


namespace find_unit_vector_l665_665383

theorem find_unit_vector (a b c d : ℝ × ℝ) (vec_a vec_b vec_c vec_d vec_d0 : ℝ × ℝ) :
  vec_a = (2, 3) →
  vec_b = (-4, 6) →
  vec_c = (2, -1) →
  vec_d = (vec_a.1 - vec_b.1 + 2 * vec_c.1, vec_a.2 - vec_b.2 + 2 * vec_c.2) →
  vec_d0 = (2 * Real.sqrt 5 / 5, -(Real.sqrt 5) / 5) →
  let mag_d := Real.sqrt (vec_d.1 ^ 2 + vec_d.2 ^ 2) in
  vec_d0 = (vec_d.1 / mag_d, vec_d.2 / mag_d) :=
by
  sorry

end find_unit_vector_l665_665383


namespace equality_of_segments_l665_665423

-- Declare necessary points and segments
variables {A B C D E F H : Type} [segment : has_mul Type]

-- Declare conditions: equality of angles
variables (h_angle_bad_hcd : Prop) (h_angle_deh_bah : Prop) (h_angle_hac_hfd : Prop)

theorem equality_of_segments (h_angle_bad_hcd : Prop) (h_angle_deh_bah : Prop) (h_angle_hac_hfd : Prop)
  : (BD * DC = AD * HD) ∧ (AH * HD = BH * HE) ∧ (AH * HD = CH * HF) :=
by {
  sorry, 
}

end equality_of_segments_l665_665423


namespace deshaun_read_books_over_summer_l665_665575

theorem deshaun_read_books_over_summer 
  (summer_days : ℕ)
  (average_pages_per_book : ℕ)
  (ratio_closest_person : ℝ)
  (pages_read_per_day_second_person : ℕ)
  (books_read : ℕ)
  (total_pages_second_person_read : ℕ)
  (h1 : summer_days = 80)
  (h2 : average_pages_per_book = 320)
  (h3 : ratio_closest_person = 0.75)
  (h4 : pages_read_per_day_second_person = 180)
  (h5 : total_pages_second_person_read = pages_read_per_day_second_person * summer_days)
  (h6 : books_read * average_pages_per_book = total_pages_second_person_read / ratio_closest_person) :
  books_read = 60 :=
by {
  sorry
}

end deshaun_read_books_over_summer_l665_665575


namespace problem1_part1_problem1_part2_l665_665295

variable (x : ℝ)

def f (x : ℝ) : ℝ :=
  if x > 0 then
    x^2 - 4 * x + 3
  else if x < 0 then
    -(x^2 + 4 * x + 3)
  else
    0

theorem problem1_part1 : f (f (-1)) = 0 := sorry

theorem problem1_part2 : 
  (∀ x : ℝ, 
  (if x > 0 then f x = x^2 - 4 * x + 3 
  else if x < 0 then f x = -(x^2 + 4 * x + 3) 
  else f x = 0)) := sorry

end problem1_part1_problem1_part2_l665_665295


namespace possible_divisor_of_p_l665_665756

theorem possible_divisor_of_p (p q r s : ℕ)
  (hpq : ∃ x y, p = 40 * x ∧ q = 40 * y ∧ Nat.gcd p q = 40)
  (hqr : ∃ u v, q = 45 * u ∧ r = 45 * v ∧ Nat.gcd q r = 45)
  (hrs : ∃ w z, r = 60 * w ∧ s = 60 * z ∧ Nat.gcd r s = 60)
  (hsp : ∃ t, Nat.gcd s p = 100 * t ∧ 100 ≤ Nat.gcd s p ∧ Nat.gcd s p < 1000) :
  7 ∣ p :=
sorry

end possible_divisor_of_p_l665_665756


namespace cube_volume_from_surface_area_l665_665950

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l665_665950


namespace cube_volume_from_surface_area_l665_665955

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l665_665955


namespace find_number_of_sides_l665_665600

theorem find_number_of_sides (n : ℕ) (h : n - (n * (n - 3)) / 2 = 3) : n = 3 := 
sorry

end find_number_of_sides_l665_665600


namespace maximum_unmarried_women_30_or_older_l665_665412

theorem maximum_unmarried_women_30_or_older (n_people : ℕ) (n_women : ℕ) (n_under30 : ℕ) (n_30orover : ℕ) 
  (p_married : ℕ) (p_married_women_30orover : ℕ) : 
  (n_people = 80) →
  (n_women = 1/4 * 80) →
  (n_under30 = 1/3 * n_women) →
  (n_30orover = 2/3 * n_women) →
  (p_married = 3/4 * n_people) →
  (∀ w < 25, ¬ (w ∈ married)) →
  (p_married_women_30orover = 2/5 * p_married) →
  (max_unmarried_women_30orover = 0) :=
by
  sorry

end maximum_unmarried_women_30_or_older_l665_665412


namespace arithmetic_sequence_sufficient_but_not_necessary_l665_665710

theorem arithmetic_sequence_sufficient_but_not_necessary
  (a : ℕ → ℤ)  -- Assuming the sequence maps to integers for generality.
  (arithmetic_seq : ∃ d : ℤ, ∀ n m : ℕ, a n = a m + d * (n - m))
  (m n p q : ℕ)
  (hm : 0 < m)
  (hn : 0 < n)
  (hp : 0 < p)
  (hq : 0 < q) :
  (m + n = p + q) → (a m + a n = a p + a q) →
  (a m + a n = a p + a q) ∧ ¬((m + n = p + q) → for_all (a m + a n = a p + a q) ) :=
by
  sorry

end arithmetic_sequence_sufficient_but_not_necessary_l665_665710


namespace proof_number_of_correct_propositions_l665_665453

def proposition_1 (A B : ℝ) (h : ∀ (A B : ℝ), A > B ↔ ¬ ∃ A B, (0 < A ∧ A ≤ π) ∧ (0 < B ∧ B ≤ π) → ∀ (A B : ℝ), sin A > sin B → A > B) :=
  ∀ (A B : ℝ), sin A > sin B → ¬ (∀ (A B : ℝ), A > B)

def proposition_2 (x y : ℝ) : Prop :=
  (x ≠ 2 ∨ y ≠ 3) → (x + y ≠ 5)

def proposition_3 (h : ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0 → ∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
  ∀ x : ℝ, ¬ (x^3 - x^2 + 1 ≤ 0)

def correct_propositions_count (A B x y : ℝ) (h1 : proposition_1 A B (by sorry)) (h2 : proposition_2 x y) (h3 : proposition_3 (by sorry)) : Nat :=
  3

theorem proof_number_of_correct_propositions (A B x y : ℝ) (h1 : proposition_1 A B (by sorry)) (h2 : proposition_2 x y) (h3 : proposition_3 (by sorry)) :
  correct_propositions_count A B x y h1 h2 h3 = 3 :=
by sorry

end proof_number_of_correct_propositions_l665_665453


namespace cube_volume_from_surface_area_l665_665877

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l665_665877


namespace wall_length_proof_l665_665467

noncomputable def volume_of_brick (length width height : ℝ) : ℝ := length * width * height

noncomputable def total_volume (brick_volume num_of_bricks : ℝ) : ℝ := brick_volume * num_of_bricks

theorem wall_length_proof
  (height_of_wall : ℝ) (width_of_walls : ℝ) (num_of_bricks : ℝ)
  (length_of_brick width_of_brick height_of_brick : ℝ)
  (total_volume_of_bricks : ℝ) :
  total_volume (volume_of_brick length_of_brick width_of_brick height_of_brick) num_of_bricks = total_volume_of_bricks →
  volume_of_brick length_of_wall height_of_wall width_of_walls = total_volume_of_bricks →
  height_of_wall = 600 →
  width_of_walls = 2 →
  num_of_bricks = 2909.090909090909 →
  length_of_brick = 5 →
  width_of_brick = 11 →
  height_of_brick = 6 →
  total_volume_of_bricks = 960000 →
  length_of_wall = 800 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end wall_length_proof_l665_665467


namespace masha_more_cakes_l665_665708

theorem masha_more_cakes (S : ℝ) (m n : ℝ) (H1 : S > 0) (H2 : m > 0) (H3 : n > 0) 
  (H4 : 2 * S * (m + n) ≤ S * m + (1/3) * S * n) :
  m > n := 
by 
  sorry

end masha_more_cakes_l665_665708


namespace cube_volume_from_surface_area_example_cube_volume_l665_665965

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l665_665965


namespace algebraic_expression_evaluation_l665_665479

theorem algebraic_expression_evaluation (a b : ℝ) (h : -2 * a + 3 * b + 8 = 18) : 9 * b - 6 * a + 2 = 32 := by
  sorry

end algebraic_expression_evaluation_l665_665479


namespace solution_set_x2_minus_5x_plus_4_range_of_a_if_x2_plus_ax_plus_4_gt_0_l665_665500

-- Problem 1: Solution Set of the Inequality
theorem solution_set_x2_minus_5x_plus_4 : 
  {x : ℝ | x^2 - 5 * x + 4 > 0} = {x : ℝ | x < 1 ∨ x > 4} :=
sorry

-- Problem 2: Range of Values for a
theorem range_of_a_if_x2_plus_ax_plus_4_gt_0 (a : ℝ) (h : ∀ x : ℝ, x^2 + a * x + 4 > 0) :
  -4 < a ∧ a < 4 :=
sorry

end solution_set_x2_minus_5x_plus_4_range_of_a_if_x2_plus_ax_plus_4_gt_0_l665_665500


namespace polynomial_value_at_x_l665_665301

theorem polynomial_value_at_x :
  ∀ (p q : ℝ), 
    (2 * p - q + 3 = p + 4 * q) ∧ 
    (p - 5 * q + 3 ≠ 0) → 
    (let x := 5 * (p + q + 1) in (x ^ 2 + 6 * x + 6) = 46) :=
begin
  intros p q h,
  sorry
end

end polynomial_value_at_x_l665_665301


namespace simplify_fraction_l665_665016

variables {a b c x y z : ℝ}

theorem simplify_fraction :
  (cx * (a^2 * x^2 + 3 * a^2 * y^2 + c^2 * z^2) + bz * (a^2 * x^2 + 3 * c^2 * x^2 + c^2 * y^2)) / (cx + bz) =
  a^2 * x^2 + c^2 * y^2 + c^2 * z^2 :=
sorry

end simplify_fraction_l665_665016


namespace problem_solution_l665_665307

def f (x : ℝ) : ℝ := |x - 1|
def g (x a : ℝ) : ℝ := Real.log 2 (f (x + 3) + f x - 2 * a)

theorem problem_solution :
  (∀ x, f x < x + |x + 1| ↔ x > 0) ∧ (∀ g : ℝ → ℝ, ∀ a : ℝ, (∀ x, g x ∈ ℝ) → a < 3 / 2) := by
  sorry

end problem_solution_l665_665307


namespace cube_volume_from_surface_area_example_cube_volume_l665_665967

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l665_665967


namespace cube_volume_of_surface_area_l665_665983

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l665_665983


namespace calculate_y_coordinate_l665_665829

theorem calculate_y_coordinate (a b c: ℝ) (ha: (2 : ℝ) = 2) (hb: (6 : ℝ) = 6) (hc: (10 : ℝ) = 10)
  (h1: b - a = 12) (h2: c - b = 12) (h_y2: a = 8) (h_y6: b = 20) (h_y10: c = 32):
  let m := (20 - 8)/(6 -2) in
  let x := 50 in
  let y := 8 + 48 * m in
  y = 152 :=
by
  sorry

end calculate_y_coordinate_l665_665829


namespace cube_volume_l665_665912

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l665_665912


namespace ratio_AD_EC_l665_665363

variables (A B C D E F : Type) [triangle : Triangle A B C]
variables (trisect : Trisects (Angle B) B D E)
variables (equal_side : (Segment B D) = (Segment B E))
variables (bisect : Bisects (Angle A) A F all (Point F B C))
include triangle trisect equal_side bisect

theorem ratio_AD_EC : (SegmentRatio (Segment A D) (Segment E C) = 1) :=
by
  sorry

end ratio_AD_EC_l665_665363


namespace distribute_points_l665_665492

theorem distribute_points (a b c : ℕ) (h1 : c = a + b) (h2 : a + b + c = 10) : 
  ∃ (p : Finset (ℕ × ℕ × ℕ)), p.card = 4 ∧ (∀ x ∈ p, let a := x.1, b := x.2.1, c := x.2.2 in c = a + b ∧ a + b + c = 10) :=
sorry

end distribute_points_l665_665492


namespace position_2007_is_ADCB_l665_665443

-- Define initial position ABCD and the transformations
def initial_position : string := "ABCD"
def reflect_vertical (s : string) : string :=
  match s.toList with
  | ['A', 'B', 'C', 'D'] => "DCBA"
  | ['D', 'C', 'B', 'A'] => "ABCD"
  | ['A', 'D', 'C', 'B'] => "BADC"
  | ['B', 'A', 'D', 'C'] => "CBAD"
  | _ => s

def rotate_clockwise (s : string) : string :=
  match s.toList with
  | ['D', 'C', 'B', 'A'] => "ADCB"
  | ['A', 'B', 'C', 'D'] => "ADCB"
  | ['B', 'A', 'D', 'C'] => "CBAD"
  | ['A', 'D', 'C', 'B'] => "CBAD"
  | _ => s

-- Function to generate the nth position in the sequence
def move_n_positions (n : ℕ) : string :=
  let rec loop (i : ℕ) (pos : string) : string :=
    if i = 0 then pos
    else if i % 2 = 1 then loop (i - 1) (rotate_clockwise pos)
    else loop (i - 1) (reflect_vertical pos)
  loop n initial_position

-- The theorem we aim to prove
theorem position_2007_is_ADCB : move_n_positions 2007 = "ADCB" := 
  sorry

end position_2007_is_ADCB_l665_665443


namespace inequality_proof_l665_665378

theorem inequality_proof (a b c : ℝ) (h1 : 2 * a^2 + b^2 = 9 * c^2) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) :
  (2 * c / a) + (c / b) ≥ real.sqrt 3 :=
sorry

end inequality_proof_l665_665378


namespace part1_part2_part3_l665_665660

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := (1 - x) / (a * x) + log x

-- Condition 1: f(x) is increasing on [1, +∞) implies a ≥ 1
theorem part1 (a : ℝ) (h1 : a > 0) : (∀ x : ℝ, x ∈ set.Ici (1 : ℝ) → (f x a)' ≥ 0) → a ≥ 1 :=
sorry

-- Condition 2: Discuss the monotonicity of f(x)
theorem part2 (a : ℝ) (hA : a ≠ 0) : 
  ( (a < 0 → ∀ x : ℝ, x > 0 → (f x a)' > 0) ∧ 
    (a > 0 → (∀ x : ℝ, x > 1 / a → (f x a)' > 0 ∧ ∀ x : ℝ, 0 < x ∧ x < 1 / a → (f x a)' < 0) ) ) :=
sorry

-- Condition 3: For a = 1 and n > 1, prove the inequality
theorem part3 {n : ℕ} (h : n > 1) : 
  log n > (finset.sum (finset.range (n+1) \ set.Icc 0 1) (λ k, 1 / (k + 1))) :=
sorry

end part1_part2_part3_l665_665660


namespace probability_third_is_three_l665_665763

open Finset

variable {α : Type}

-- Define S as the set of all permutations of the sequence 1, 2, 3, 4
def S : Finset (Array α) := univ.filter (fun a => perm a (Array.mk [1, 2, 3, 4]))

-- Define S' as the subset of S where the first term is not 2
def S' : Finset (Array α) := S.filter (fun a => a[0] ≠ 2)

-- Define function to check if the third term is 3
def third_is_three (a : Array α) : Prop := a[2] = 3

-- Statement of the problem to prove
theorem probability_third_is_three {a b : ℕ}
    (h : (probability_third_is_three : fraction) = 1 / 3) :
    a + b = 4 :=
sorry

end probability_third_is_three_l665_665763


namespace factor_expression_l665_665220

theorem factor_expression (x : ℝ) : 5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l665_665220


namespace exponent_division_l665_665198

theorem exponent_division : (23 ^ 11) / (23 ^ 8) = 12167 := 
by {
  sorry
}

end exponent_division_l665_665198


namespace problem_l665_665327

theorem problem (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 + 2007 = 2008 :=
by
  sorry

end problem_l665_665327


namespace multiplicative_inverse_sum_is_zero_l665_665649

theorem multiplicative_inverse_sum_is_zero (a b : ℝ) (h : a * b = 1) :
  a^(2015) * b^(2016) + a^(2016) * b^(2017) + a^(2017) * b^(2016) + a^(2016) * b^(2015) = 0 :=
sorry

end multiplicative_inverse_sum_is_zero_l665_665649


namespace planes_divide_space_l665_665456

-- Definition of a triangular prism
def triangular_prism (V : Type) (P : Set (Set V)) : Prop :=
  ∃ (A B C D E F : V),
    P = {{A, B, C}, {D, E, F}, {A, B, D, E}, {B, C, E, F}, {C, A, F, D}}

-- The condition: planes containing the faces of a triangular prism
def planes_containing_faces (V : Type) (P : Set (Set V)) : Prop :=
  triangular_prism V P

-- Proof statement: The planes containing the faces of a triangular prism divide the space into 21 parts
theorem planes_divide_space (V : Type) (P : Set (Set V))
  (h : planes_containing_faces V P) :
  ∃ parts : ℕ, parts = 21 := by
  sorry

end planes_divide_space_l665_665456


namespace speed_of_train_b_l665_665081

-- Definitions for the conditions
def length_train_a : ℝ := 175 -- in meters
def length_train_b : ℝ := 150 -- in meters
def speed_train_a : ℝ := 54 -- in km/hr
def time_crossing : ℝ := 13 -- in seconds

-- Converting speeds and distance to consistent units
def total_length : ℝ := length_train_a + length_train_b -- in meters
def relative_speed := total_length / time_crossing -- in m/s
def relative_speed_kmh := relative_speed * 3.6 -- in km/hr

-- The statement that needs to be proven
theorem speed_of_train_b : 
  relative_speed_kmh - speed_train_a = 36 :=
by 
  -- Placeholder for the proof
  sorry

end speed_of_train_b_l665_665081


namespace cube_volume_from_surface_area_l665_665886

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l665_665886


namespace circle_center_l665_665594

theorem circle_center {x y : ℝ} :
  4 * x^2 - 8 * x + 4 * y^2 - 16 * y + 20 = 0 → (x, y) = (1, 2) :=
by
  sorry

end circle_center_l665_665594


namespace part_1_a_part_1_b_part_2_l665_665365

variable (n : ℕ)
variable (a : ℕ → ℕ)
variable (b : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (T : ℕ → ℕ)
variable (c : ℕ → ℕ)

-- Arithmetic sequence a_n
def a_n := 2 * n - 1

-- Geometric sequence b_n with first term 1 and common ratio 2
def b_n := 2^(n-1)

-- Sum of first n terms S_n of sequence b_n
axiom Hb : ∀ n, S n = 2 * b n - 1

-- Sum of first n terms T_n of sequence c_n = a_n * b_n
def c_n := a_n n * b_n n
axiom Hc : ∀ n, T n = (∑ i in finset.range (n + 1), c i)

-- Proof statements
theorem part_1_a : a 3 = 5 → a 7 = 13 → a n = a_n := sorry

theorem part_1_b : (S 1 = 1) → (∀ n ≥ 2, S n = 2 * b n - 1 ∧ b n = 2 * b (n-1)) → b n = b_n := sorry

theorem part_2 : (∀ n, c n = a_n n * b_n n) → (∀ n, T n = (∑ i in finset.range (n + 1), c i)) → T n = (2 * n - 3) * 2^n + 3 := sorry

end part_1_a_part_1_b_part_2_l665_665365


namespace effective_distance_calculation_l665_665503

noncomputable def boat_speed_still_water := 65 -- km/hr
noncomputable def current_speed := 15 -- km/hr
noncomputable def wind_effect := 1.10 -- 10% increase
noncomputable def travel_time_minutes := 25 -- minutes
noncomputable def time_lost_minutes := 2 -- minutes

noncomputable def effective_speed_still_water := boat_speed_still_water * wind_effect
noncomputable def effective_speed_downstream := effective_speed_still_water + current_speed
noncomputable def effective_travel_time_hours := (travel_time_minutes - time_lost_minutes) / 60

noncomputable def effective_distance := effective_speed_downstream * effective_travel_time_hours

theorem effective_distance_calculation : effective_distance ≈ 33.17 :=
by
  sorry

end effective_distance_calculation_l665_665503


namespace roots_of_polynomial_l665_665246

-- Define the polynomial P(x) = x^3 - 3x^2 - x + 3
def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

-- Define the statement to prove the roots of the polynomial
theorem roots_of_polynomial :
  ∀ x : ℝ, (P x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) :=
by
  sorry

end roots_of_polynomial_l665_665246


namespace radii_of_equal_circles_l665_665712

theorem radii_of_equal_circles (R : ℝ) 
  (h1 : ∀ (ABC : Triangle), (∡ BAC = 120))
  (h2 : ∀ (O : Point), inscribed_circle_radius(ABC, O, R)) 
  (h3 : ∃ (O1 O2 : Circle), radius(O1) = radius(O2) ∧ (tangent O1 ABC) ∧ (tangent O2 ABC) ∧ (tangent O1 O2)) :
    let x1 := R / 3,
        x2 := (3 - 2 * Real.sqrt 2) * R / 3 in
    (radius (O1) = x1 ∨ radius (O1) = x2) ∧ (radius (O2) = x1 ∨ radius (O2) = x2) :=
sorry

end radii_of_equal_circles_l665_665712


namespace find_k_l665_665675

def a : ℝ × ℝ × ℝ := (2, 1, 4)
def b : ℝ × ℝ × ℝ := (1, 0, 2)
def k := 15 / 31

theorem find_k (ha : a = (2, 1, 4)) (hb : b = (1, 0, 2)) : 
    (a.fst + b.fst, a.snd + b.snd, a.snd + b.snd) • (k * a.fst - b.fst, k * a.snd - b.snd, k * a.snd - b.snd) = 0 :=
sorry

end find_k_l665_665675


namespace student_count_estimate_l665_665160

theorem student_count_estimate 
  (n : Nat) 
  (h1 : 80 ≤ n) 
  (h2 : 100 ≤ n) 
  (h3 : 20 * n = 8000) : 
  n = 400 := 
by 
  sorry

end student_count_estimate_l665_665160


namespace total_pages_read_correct_l665_665772

-- Definition of the problem conditions
def first_week_books := 5
def first_week_book_pages := 300
def first_week_magazines := 3
def first_week_magazine_pages := 120
def first_week_newspapers := 2
def first_week_newspaper_pages := 50

def second_week_books := 2 * first_week_books
def second_week_book_pages := 350
def second_week_magazines := 4
def second_week_magazine_pages := 150
def second_week_newspapers := 1
def second_week_newspaper_pages := 60

def third_week_books := 3 * first_week_books
def third_week_book_pages := 400
def third_week_magazines := 5
def third_week_magazine_pages := 125
def third_week_newspapers := 1
def third_week_newspaper_pages := 70

-- Total pages read in each week
def first_week_total_pages : Nat :=
  (first_week_books * first_week_book_pages) +
  (first_week_magazines * first_week_magazine_pages) +
  (first_week_newspapers * first_week_newspaper_pages)

def second_week_total_pages : Nat :=
  (second_week_books * second_week_book_pages) +
  (second_week_magazines * second_week_magazine_pages) +
  (second_week_newspapers * second_week_newspaper_pages)

def third_week_total_pages : Nat :=
  (third_week_books * third_week_book_pages) +
  (third_week_magazines * third_week_magazine_pages) +
  (third_week_newspapers * third_week_newspaper_pages)

-- Grand total pages read over three weeks
def total_pages_read : Nat :=
  first_week_total_pages + second_week_total_pages + third_week_total_pages

-- Theorem statement to be proven
theorem total_pages_read_correct :
  total_pages_read = 12815 :=
by
  -- Proof will be provided here
  sorry

end total_pages_read_correct_l665_665772


namespace xiao_ming_games_needed_l665_665481

theorem xiao_ming_games_needed {total_played_winning_percentage: ℕ} 
  (h₁ : total_played_winning_percentage = 20)
  (h₂ : 95% of 20 games are won) 
  (h₃ : Xiao Ming wins(subsequently)) : 
  additional_games_to_play == 5 :=
sorry

end xiao_ming_games_needed_l665_665481


namespace area_quadrilateral_pmqn_l665_665511

theorem area_quadrilateral_pmqn : 
  let side_length : ℝ := 2
  let P : EuclideanSpace ℝ (Fin 3) := ⟨0, 0, 0⟩
  let Q : EuclideanSpace ℝ (Fin 3) := ⟨2, 0, 0⟩
  let M : EuclideanSpace ℝ (Fin 3) := ⟨1, 1, 0⟩
  let N : EuclideanSpace ℝ (Fin 3) := ⟨2, 1, 1⟩
  let PM : ℝ := dist P M
  let PQ : ℝ := dist P Q
  let QN : ℝ := dist Q N
  let MN : ℝ := dist M N
  ∃ (a b h : ℝ), a = 2 ∧ b = 2 * Real.sqrt 2 ∧ h = 2 ∧ 
    (PM = Real.sqrt 5) ∧ (QN = Real.sqrt 5) ∧ (MN = 2 * Real.sqrt 2) →
    abs ((1 / 2) * (a + b) * h - (2 + 2 * Real.sqrt 2)) < 1e-6 :=
by
  sorry

end area_quadrilateral_pmqn_l665_665511


namespace total_buttons_l665_665542

-- Defining the given conditions
def green_buttons : ℕ := 90
def yellow_buttons : ℕ := green_buttons + 10
def blue_buttons : ℕ := green_buttons - 5

-- Stating the theorem to prove the total number of buttons
theorem total_buttons : green_buttons + yellow_buttons + blue_buttons = 275 :=
by 
  sorry

end total_buttons_l665_665542


namespace multiplication_correct_l665_665356

theorem multiplication_correct :
  23 * 195 = 4485 :=
by
  sorry

end multiplication_correct_l665_665356


namespace line_plane_relationship_l665_665645

-- Define vectors a and n
def vec_a : ℝ × ℝ × ℝ := (3, -2, -1)
def vec_n : ℝ × ℝ × ℝ := (1, 2, -1)

-- Dot product calculation
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
    v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Theorem stating the required proof
theorem line_plane_relationship
    (h : dot_product vec_a vec_n = 0) :
    (vec_a.1 * vec_n.1 + vec_a.2 * vec_n.2 + vec_a.3 * vec_n.3 = 0) →
    (line_parallel_plane vec_a vec_n ∨ line_contained_in_plane vec_a vec_n) := 
begin
    sorry -- Proof goes here
end

-- Definitions for line being parallel to and contained within a plane
def line_parallel_plane (d : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) : Prop :=
    dot_product d n = 0

def line_contained_in_plane (d : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) : Prop :=
    dot_product d n = 0 -- Need additional definitions for a line to be fully contained, omitted for brevity


end line_plane_relationship_l665_665645


namespace max_intersection_points_of_three_triangles_l665_665835

theorem max_intersection_points_of_three_triangles : 
  ∀ (triangles : List Triangle), triangles.length = 3 → 
  (∃ p : List (Point × Point × Point), p.length = 12) → 
  max_intersection_points triangles = 18 :=
by
  sorry

end max_intersection_points_of_three_triangles_l665_665835


namespace turtle_total_l665_665076

theorem turtle_total (K : ℕ) (h1 : ∃ n : ℕ, Kris_has n ∧ Trey_has (5 * n))
  (h2 : ∃ m : ℕ, Kris_has (K / 4) ∧ Kristen_has K) :
  K + K / 4 + 5 * (K / 4) = 30 := by
existsi K
have h3 : Kris_has (K / 4) from sorry
have h4 : Trey_has (5 * (K / 4)) from sorry
have h5 : Kristen_has K from sorry
rw [h3, h4, h5]
sorry

end turtle_total_l665_665076


namespace pyramid_volume_l665_665439

theorem pyramid_volume 
  (α β R : ℝ) 
  (h_α : 0 < α ∧ α < π)
  (h_β : 0 < β ∧ β < π / 2)
  (h_R : 0 < R) :
  volume_of_pyramid α β R = (4 / 3) * R^3 * (Real.sin (2 * β))^2 * (Real.sin β)^2 * (Real.sin α) := 
sorry

noncomputable def volume_of_pyramid (α β R : ℝ) : ℝ :=
  (1 / 3) * base_area α β R * height β R

noncomputable def base_area (α β R : ℝ) : ℝ :=
  2 * R^2 * (Real.sin (2 * β))^2 * (Real.sin α)

noncomputable def height (β R : ℝ) : ℝ :=
  2 * R * (Real.sin β)^2


end pyramid_volume_l665_665439


namespace cube_volume_from_surface_area_example_cube_volume_l665_665972

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l665_665972


namespace number_of_pencils_l665_665041

variable (P L : ℕ)

-- Conditions
def condition1 : Prop := P / L = 5 / 6
def condition2 : Prop := L = P + 5

-- Statement to prove
theorem number_of_pencils (h1 : condition1 P L) (h2 : condition2 P L) : L = 30 :=
  sorry

end number_of_pencils_l665_665041


namespace alice_probability_at_least_one_multiple_of_4_l665_665551

def probability_multiple_of_4 : ℚ :=
  1 - (45 / 60)^3

theorem alice_probability_at_least_one_multiple_of_4 :
  probability_multiple_of_4 = 37 / 64 :=
by
  sorry

end alice_probability_at_least_one_multiple_of_4_l665_665551


namespace root_shifted_is_root_of_quadratic_with_integer_coeffs_l665_665823

theorem root_shifted_is_root_of_quadratic_with_integer_coeffs
  (a b c t : ℤ)
  (h : a ≠ 0)
  (h_root : a * t^2 + b * t + c = 0) :
  ∃ (x : ℤ), a * x^2 + (4 * a + b) * x + (4 * a + 2 * b + c) = 0 :=
by {
  sorry
}

end root_shifted_is_root_of_quadratic_with_integer_coeffs_l665_665823


namespace max_intersection_points_l665_665720

noncomputable def α := Type

def A : set (α) := {P | ∃ x y r : ℝ, r > 0 ∧ (P.1 - x)^2 + (P.2 - y)^2 = r^2}
def B : set (α) := {Q | ∃ m c : ℝ, Q.2 = m * Q.1 + c}

theorem max_intersection_points (α : Type) (A B : set α) :
  set.countable (A ∩ B) → 2 :=
sorry

end max_intersection_points_l665_665720


namespace factor_expression_l665_665221

theorem factor_expression (x : ℝ) : 5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l665_665221


namespace smallest_value_bounds_l665_665056

variable {a : Fin 8 → ℝ}

theorem smallest_value_bounds
  (h1 : (∑ i, a i) = 4 / 3)
  (h2 : ∀ j, (∑ i, if i = j then 0 else a i) > 0) :
  ∃ a1, -8 < a1 ∧ a1 ≤ 1 / 6 :=
begin
  let a1 := a 0,
  use a1,
  split,
  { sorry },
  { sorry }
end

end smallest_value_bounds_l665_665056


namespace frequency_of_hits_l665_665184

theorem frequency_of_hits (n m : ℕ) (h_n : n = 20) (h_m : m = 15) : (m / n : ℚ) = 0.75 := by
  sorry

end frequency_of_hits_l665_665184


namespace solve_inequality_a_eq_1_range_of_a_l665_665305

open Real

/-
Problem statement 1: Given \( f(x) = |x + 2| + |x - 1| \) when \( a = 1 \)
Prove that the solution set for \( f(x) \leq 5 \) is \( [-3, 2] \).
-/
theorem solve_inequality_a_eq_1 (x : ℝ) : (|x + 2| + |x - 1| ≤ 5) ↔ (-3 ≤ x ∧ x ≤ 2) := sorry

/-
Problem statement 2: Given \( f(x) = |x + 2a| + |x - 1| \) and the condition \( f(x) \geq 2 \) for all \( x \in \mathbb{R} \)
Prove that \( a \geq \frac{1}{2} \text{ or } a \leq -\frac{3}{2} \).
-/
theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 2a| + |x - 1| ≥ 2) ↔ (a ≥ 1/2 ∨ a ≤ -3/2) := sorry

end solve_inequality_a_eq_1_range_of_a_l665_665305


namespace arithmetic_sequence_problem_l665_665280

variable (a : ℕ → ℤ) -- defining the sequence {a_n}
variable (S : ℕ → ℤ) -- defining the sum of the first n terms S_n

theorem arithmetic_sequence_problem (m : ℕ) (h1 : m > 1) 
  (h2 : a (m - 1) + a (m + 1) - a m ^ 2 = 0) 
  (h3 : S (2 * m - 1) = 38) 
  : m = 10 :=
sorry

end arithmetic_sequence_problem_l665_665280


namespace cube_volume_l665_665930

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l665_665930


namespace fx_properties_l665_665304

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem fx_properties :
  (∃ x : ℝ, f(x) = 0) ∧
  (∀ x : ℝ, f(-x) = -f(x)) ∧
  (∀ x : ℝ, 1 - Real.cos x ≥ 0) :=
by
  sorry

end fx_properties_l665_665304


namespace mike_total_spent_on_toys_l665_665406

theorem mike_total_spent_on_toys :
  let marbles := 9.05
  let football := 4.95
  let baseball := 6.52
  marbles + football + baseball = 20.52 :=
by
  sorry

end mike_total_spent_on_toys_l665_665406


namespace min_squared_distance_l665_665803

theorem min_squared_distance 
  (x y z : ℝ)
  (h1 : y = x * r) 
  (h2 : z = x * r^2) 
  (h3 : z ≥ 1) 
  (h4 : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h5 : y * z = (x * y + x * z) / 2) :
  (x - 1)^2 + (y - 1)^2 + (z - 1)^2 = 18 := 
begin
  sorry
end

end min_squared_distance_l665_665803


namespace faith_overtime_hours_per_day_l665_665230

noncomputable def normal_pay_per_hour : ℝ := 13.50
noncomputable def normal_daily_hours : ℕ := 8
noncomputable def normal_weekly_days : ℕ := 5
noncomputable def total_weekly_earnings : ℝ := 675
noncomputable def overtime_rate_multiplier : ℝ := 1.5

noncomputable def normal_weekly_hours := normal_daily_hours * normal_weekly_days
noncomputable def normal_weekly_earnings := normal_pay_per_hour * normal_weekly_hours
noncomputable def overtime_earnings := total_weekly_earnings - normal_weekly_earnings
noncomputable def overtime_pay_per_hour := normal_pay_per_hour * overtime_rate_multiplier
noncomputable def total_overtime_hours := overtime_earnings / overtime_pay_per_hour
noncomputable def overtime_hours_per_day := total_overtime_hours / normal_weekly_days

theorem faith_overtime_hours_per_day :
  overtime_hours_per_day = 1.33 := 
by 
  sorry

end faith_overtime_hours_per_day_l665_665230


namespace smallest_possible_sum_l665_665285

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_ne : x ≠ y) (h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 50 :=
sorry

end smallest_possible_sum_l665_665285


namespace line_perpendicular_passing_through_point_l665_665597

theorem line_perpendicular_passing_through_point :
  ∃ (a b c : ℝ), (∀ (x y : ℝ), 2 * x + y - 2 = 0 ↔ a * x + b * y + c = 0) ∧ 
                (a, b) ≠ (0, 0) ∧ 
                (a * -1 + b * 4 + c = 0) ∧ 
                (a * 1/2 + b * (-2) ≠ -4) :=
by { sorry }

end line_perpendicular_passing_through_point_l665_665597


namespace households_surveyed_l665_665136

theorem households_surveyed (h_total : 160 = 
  let h_neither := 80 in
  let h_only_A := 60 in
  let h_both := 5 in
  let h_only_B := 3 * h_both in
  h_neither + h_only_A + h_both + h_only_B) : 
  True :=
by
  sorry

end households_surveyed_l665_665136


namespace arrangement_probability_l665_665605

theorem arrangement_probability :
  let total_ways := Nat.factorial 8 / (Nat.factorial 5 * Nat.factorial 3),
      probability := 1 / total_ways
  in probability = (1 : ℚ) / 56 :=
by
  sorry

end arrangement_probability_l665_665605


namespace relationship_among_g_a_f_b_and_zero_l665_665000

noncomputable def f (x : ℝ) := Real.exp x + x - 2
noncomputable def g (x : ℝ) := Real.log x + x ^ 2 - 3

-- Define the hypotheses
variables {a b : ℝ}
hypothesis f_a_zero : f a = 0
hypothesis g_b_zero : g b = 0

-- State the theorem
theorem relationship_among_g_a_f_b_and_zero (ha : f a = 0) (hb : g b = 0) : g a < 0 ∧ 0 < f b :=
  sorry

end relationship_among_g_a_f_b_and_zero_l665_665000


namespace average_M_possibilities_l665_665750

theorem average_M_possibilities (M : ℝ) (h1 : 12 < M) (h2 : M < 25) :
    (12 = (8 + 15 + M) / 3) ∨ (15 = (8 + 15 + M) / 3) :=
  sorry

end average_M_possibilities_l665_665750


namespace jameson_badminton_medals_l665_665368

theorem jameson_badminton_medals (total_medals track_medals : ℕ) (swimming_medals : ℕ) :
  total_medals = 20 →
  track_medals = 5 →
  swimming_medals = 2 * track_medals →
  total_medals - (track_medals + swimming_medals) = 5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end jameson_badminton_medals_l665_665368


namespace arcs_eq_l665_665780

--Define the setup for the circles
variables (O A : Point)
variables (r : ℝ) -- radius of the larger circle and half of the smaller circle's diameter.
variables (circle1 circle2 : Set Point)

-- Conditions for the problem
def larger_circle := {p : Point | dist O p = r}
def smaller_circle := {p : Point | dist M p = r / 2}
def is_midtpoint (M : Point) := dist O M = dist M A ∧ dist O A = 2 * dist O M

-- Points B and C are intersections of smaller circles with the larger circle
variables (B C : Point)
axiom B_on_circles : B ∈ larger_circle ∧ B ∈ smaller_circle
axiom C_on_circles : C ∈ larger_circle ∧ C ∈ smaller_circle

-- Define angles at the center of larger circle subtended by arcs AB and AC
def angle_at_center (p1 p2 : Point) : ℝ := 
  acos ((dist p1 O)^2 + (dist p2 O)^2 - (dist p1 p2)^2 / (2 * (dist p1 O) * (dist p2 O)))

-- The statement of proof problem
theorem arcs_eq : 
  angle_at_center O B = angle_at_center O C →
  arc_length larger_circle O B = arc_length larger_circle O C :=
sorry

end arcs_eq_l665_665780


namespace div_eq_implies_eq_l665_665994

theorem div_eq_implies_eq (a b : ℕ) (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b :=
sorry

end div_eq_implies_eq_l665_665994


namespace verify_distinct_outcomes_l665_665856

def i : ℂ := Complex.I

theorem verify_distinct_outcomes :
  ∃! S, ∀ n : ℤ, n % 8 = n → S = i^n + i^(-n)
  := sorry

end verify_distinct_outcomes_l665_665856


namespace cube_volume_from_surface_area_l665_665956

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l665_665956


namespace decreasing_function_slope_l665_665696

theorem decreasing_function_slope {a : ℝ} : (∀ x : ℝ, (1 - a) * x + 2 < (1 - a) * (x + 1) + 2) ↔ a > 1 :=
by
  intro a
  split
  {
    intro h
    let x := 0
    have h0: (1 - a) * x + 2 < (1 - a) * (x + 1) + 2 := h x
    simp at h0
    linarith
  }
  {
    intro ha
    intro x
    have h: 1 - a < 0 := by linarith
    linarith [h]
  }

end decreasing_function_slope_l665_665696


namespace amaya_total_time_l665_665175

-- Define the times as per the conditions
def first_segment : Nat := 35 + 5
def second_segment : Nat := 45 + 15
def third_segment : Nat := 20

-- Define the total time by summing up all segments
def total_time : Nat := first_segment + second_segment + third_segment

-- The theorem to prove
theorem amaya_total_time : total_time = 120 := by
  -- Let's explicitly state the expected result here
  have h1 : first_segment = 40 := rfl
  have h2 : second_segment = 60 := rfl
  have h3 : third_segment = 20 := rfl
  have h_sum : total_time = 40 + 60 + 20 := by
    rw [h1, h2, h3]
  simp [total_time, h_sum]
  -- Finally, the result is 120
  exact rfl

end amaya_total_time_l665_665175


namespace slope_of_midpoints_l665_665566

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

theorem slope_of_midpoints :
  let p1 := (0 : ℝ, 0 : ℝ),
      p2 := (3 : ℝ, 5 : ℝ),
      p3 := (4 : ℝ, 1 : ℝ),
      p4 := (7 : ℝ, -2 : ℝ),
      m1 := midpoint p1 p2,
      m2 := midpoint p3 p4 in
  slope m1 m2 = -3 / 4 :=
by
  sorry

end slope_of_midpoints_l665_665566


namespace intersection_A_B_l665_665640

-- Definition of set A
def A (x : ℝ) : Prop := -1 < x ∧ x < 2

-- Definition of set B
def B (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 0

-- Theorem stating the intersection of sets A and B
theorem intersection_A_B (x : ℝ) : (A x ∧ B x) ↔ (-1 < x ∧ x ≤ 0) :=
by sorry

end intersection_A_B_l665_665640


namespace prove_inequality_l665_665768

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

theorem prove_inequality (a b : ℝ) (h1 : |a| < 2) (h2 : |b| < 2) :
  |a + b| + |a - b| < f(0) := 
sorry

end prove_inequality_l665_665768


namespace evaluate_49sq_minus_25sq_l665_665216

theorem evaluate_49sq_minus_25sq : (49^2 - 25^2 = 1776) :=
by
  let a := 49
  let b := 25
  have h1 : (a + b) = 74 := by norm_num
  have h2 : (a - b) = 24 := by norm_num
  have h3 : (a + b) * (a - b) = 1776 := by norm_num
  rw [←add_sub_cancel (a²) (b²)]
  exact h3

end evaluate_49sq_minus_25sq_l665_665216


namespace sale_in_fifth_month_l665_665516

theorem sale_in_fifth_month 
    (a1 a2 a3 a4 a6 : ℕ) 
    (avg_sale : ℕ)
    (H_avg : avg_sale = 8500)
    (H_a1 : a1 = 8435) 
    (H_a2 : a2 = 8927) 
    (H_a3 : a3 = 8855) 
    (H_a4 : a4 = 9230) 
    (H_a6 : a6 = 6991) : 
    ∃ a5 : ℕ, (a1 + a2 + a3 + a4 + a5 + a6) / 6 = avg_sale ∧ a5 = 8562 := 
by
    sorry

end sale_in_fifth_month_l665_665516


namespace r_plus_s_value_l665_665450

theorem r_plus_s_value :
  (∃ (r s : ℝ) (line_intercepts : ∀ x y, y = -1/2 * x + 8 ∧ ((x = 16 ∧ y = 0) ∨ (x = 0 ∧ y = 8))), 
    s = -1/2 * r + 8 ∧ (16 * 8 / 2) = 2 * (16 * s / 2) ∧ r + s = 12) :=
sorry

end r_plus_s_value_l665_665450


namespace total_hours_is_900_l665_665114

-- Definitions for the video length, speeds, and number of videos watched
def video_length : ℕ := 100
def lila_speed : ℕ := 2
def roger_speed : ℕ := 1
def num_videos : ℕ := 6

-- Definition of total hours watched
def total_hours_watched : ℕ :=
  let lila_time_per_video := video_length / lila_speed
  let roger_time_per_video := video_length / roger_speed
  (lila_time_per_video * num_videos) + (roger_time_per_video * num_videos)

-- Prove that the total hours watched is 900
theorem total_hours_is_900 : total_hours_watched = 900 :=
by
  -- Proving the equation step-by-step
  sorry

end total_hours_is_900_l665_665114


namespace points_with_exactly_two_area_bisecting_lines_l665_665069

/-- In an equilateral triangle, determine the points through which exactly two area-bisecting lines pass. -/
theorem points_with_exactly_two_area_bisecting_lines
  {ABC : Triangle}
  (h_eq : ABC.is_equilateral)
  (h_g : ∀ (P : Point), P ∈ centroid(ABC) ↔ (∃ (L : line), L.bisects_area(ABC P)))
  (h_t : ∀ (P : Point), ∃ (hyp : Hyperbola), P ∈ hyp ∧ L.tangent_to(hyp) ∧ L.bisects_area(ABC))
  (interior_points : Set Point := { P | ∃ (arc : Arc), P ∈ interior(arc) ∧ arc.is_part_of_hyperbolas(h_t) }) :
  (∃  (interior_points) (area_bisecting_lines P : Line), (P ∈ interior_points) ↔ (∃ L1 L2, L1.bisects_area(ABC) ∧ L2.bisects_area(ABC) ∧ P ∈ L1 ∧ P ∈ L2)) sorry

end points_with_exactly_two_area_bisecting_lines_l665_665069


namespace least_value_x_l665_665339

theorem least_value_x (x : ℕ) (p q : ℕ) (h_prime_p : Nat.Prime p) (h_prime_q : Nat.Prime q)
  (h_distinct : p ≠ q) (h_diff : q - p = 3) (h_even_prim : x / (11 * p * q) = 2) : x = 770 := by
  sorry

end least_value_x_l665_665339


namespace average_income_N_O_l665_665438

variable (M N O : ℝ)

-- Condition declaration
def condition1 : Prop := M + N = 10100
def condition2 : Prop := M + O = 10400
def condition3 : Prop := M = 4000

-- Theorem statement
theorem average_income_N_O (h1 : condition1 M N) (h2 : condition2 M O) (h3 : condition3 M) :
  (N + O) / 2 = 6250 :=
sorry

end average_income_N_O_l665_665438


namespace count_distinct_products_l665_665748

noncomputable def number_of_distinct_products : ℕ :=
  let T := { n : ℕ | n > 0 ∧ ∃ a b c : ℕ, 0 ≤ a ∧ a ≤ 6 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ n = 2^a * 3^b * 5^c } in
  (325 - 7 : ℕ)

theorem count_distinct_products (T : set ℕ) (hT : ∀ n, n ∈ T ↔ n > 0 ∧ ∃ a b c : ℕ, 0 ≤ a ∧ a ≤ 6 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ n = 2^a * 3^b * 5^c) :
  number_of_distinct_products = 318 := by
  sorry

end count_distinct_products_l665_665748


namespace avg_minutes_listened_l665_665134
-- Import all necessary libraries

-- Define the main problem as a theorem in Lean 4
theorem avg_minutes_listened (total_audience : ℕ) (lecture_duration : ℝ) (perc_entire_lecture : ℝ) (perc_no_lecture : ℝ) (perc_half_lecture_of_remaining : ℝ) :
  lecture_duration = 90 → perc_entire_lecture = 0.3 → perc_no_lecture = 0.15 → perc_half_lecture_of_remaining = 0.4 →
  total_audience = 100 →
  let 
    entire_lecture_count := total_audience * perc_entire_lecture,
    no_lecture_count := total_audience * perc_no_lecture,
    remaining_audience := total_audience - entire_lecture_count - no_lecture_count,
    half_lecture_count := remaining_audience * perc_half_lecture_of_remaining,
    quarter_lecture_count := remaining_audience - half_lecture_count,
    total_minutes_heard_entire := entire_lecture_count * lecture_duration,
    total_minutes_heard_none := no_lecture_count * 0,
    total_minutes_heard_half := half_lecture_count * (lecture_duration / 2),
    total_minutes_heard_quarter := quarter_lecture_count * (lecture_duration / 4),
    total_minutes_heard := total_minutes_heard_entire + total_minutes_heard_none + total_minutes_heard_half + total_minutes_heard_quarter,
    avg_minutes_heard := total_minutes_heard / total_audience 
  in
    avg_minutes_heard = 44 :=
by
  sorry

end avg_minutes_listened_l665_665134


namespace moles_NaOH_combined_l665_665599

-- Define the conditions
def CH3COOH := ℝ
def NaOH := ℝ
def NaCH3COO := ℝ

variable (moles_CH3COOH : CH3COOH) (moles_NaCH3COO : NaCH3COO)
axiom H1 : moles_CH3COOH = 1 -- 1 mole of CH3COOH
axiom H2 : moles_NaCH3COO = 1 -- 1 mole of NaCH3COO is formed

-- Prove that the number of moles of NaOH combined is 1
theorem moles_NaOH_combined (moles_NaOH : NaOH) : moles_NaOH = 1 :=
by
  -- Placeholder to indicate we need to prove this theorem
  sorry

end moles_NaOH_combined_l665_665599


namespace trigonometric_proof_l665_665193

noncomputable def cos30 : ℝ := Real.sqrt 3 / 2
noncomputable def tan60 : ℝ := Real.sqrt 3
noncomputable def sin45 : ℝ := Real.sqrt 2 / 2
noncomputable def cos45 : ℝ := Real.sqrt 2 / 2

theorem trigonometric_proof :
  2 * cos30 - tan60 + sin45 * cos45 = 1 / 2 :=
by
  sorry

end trigonometric_proof_l665_665193


namespace transformation_sine_graph_l665_665848

theorem transformation_sine_graph : 
  ∀ x, sin (2 * x - π / 6) = sin (2 * (x - π / 6)) :=
by 
  sorry

end transformation_sine_graph_l665_665848


namespace octagon_area_equals_l665_665528

noncomputable def area_of_inscribed_octagon (r : ℝ) : ℝ :=
  let s := 4 * real.sqrt (2 - real.sqrt 2)
  let A_triangle := 4 * real.sqrt 2
  8 * A_triangle

theorem octagon_area_equals : area_of_inscribed_octagon 4 = 32 * real.sqrt 2 :=
by
  unfold area_of_inscribed_octagon
  rw [nat.cast_bit0, nat.cast_bit0]
  rw [mul_assoc, ← mul_assoc 8]
  rw [mul_comm 8 (4 * _), ← mul_assoc]
  have : 8 * 4 * real.sqrt 2 = 32 * real.sqrt 2 := by sorry
  exact this

end octagon_area_equals_l665_665528


namespace max_students_gcd_l665_665203

def numPens : Nat := 1802
def numPencils : Nat := 1203
def numErasers : Nat := 1508
def numNotebooks : Nat := 2400

theorem max_students_gcd : Nat.gcd (Nat.gcd (Nat.gcd numPens numPencils) numErasers) numNotebooks = 1 := by
  sorry

end max_students_gcd_l665_665203


namespace probability_team_c_more_points_l665_665210

noncomputable section
open Nat ProbabilityTheory

def round_robin_tournament (n : ℕ) : Prop := ∀ T : Finset ℕ, T.card = n

def win_probability (p : ℚ) := ∀ n : ℕ, ∀ i j : ℕ, i ≠ j → 0 < p ∧ p < 1

def points_awarded (games_played : ℕ) (wins : ℕ) : ℕ := wins

theorem probability_team_c_more_points 
  (n : ℕ) (games_played : ℕ) (p : ℚ) (team_c_wins_first_game : ℕ) :
  round_robin_tournament n → 
  ∀ i j : ℕ, i ≠ j → win_probability p → 
  (∑ k in Finset.range 7, if k = 0 then 1 else 0) + 
  ∑ k in Finset.range 6, (if (binomial 6 k : ℚ) * (p ^ k) * ((1 - p) ^ (6 - k)) > 
                          (binomial 6 k : ℚ) * ((1 - p) ^ k) * (p ^ (6 - k)) then 1 else 0)
  = (11 / 32 : ℚ) := 
sorry

end probability_team_c_more_points_l665_665210


namespace cylinder_height_max_lateral_surface_area_l665_665730

noncomputable def cylinder_max_lateral_surface_area_height (R : ℝ) (h : ℝ) : Prop :=
  ∀ (r : ℝ), r^2 + (h/2)^2 = R^2 → 2*π*r*h = 4*π*r*√(R^2 - r^2) → h = R * √2

theorem cylinder_height_max_lateral_surface_area (R : ℝ) :
  cylinder_max_lateral_surface_area_height R (R * √2) :=
by
  intros r hr₁ hr₂
  sorry

end cylinder_height_max_lateral_surface_area_l665_665730


namespace average_salary_of_managers_l665_665126

theorem average_salary_of_managers 
    (num_managers num_associates : ℕ) 
    (avg_salary_associates avg_salary_company : ℝ) 
    (H_managers : num_managers = 15) 
    (H_associates : num_associates = 75) 
    (H_avg_associates : avg_salary_associates = 30000) 
    (H_avg_company : avg_salary_company = 40000) : 
    ∃ M : ℝ, 15 * M + 75 * 30000 = 90 * 40000 ∧ M = 90000 := 
by
    use 90000
    rw [H_managers, H_associates, H_avg_associates, H_avg_company]
    split
    · linarith
    · rfl

end average_salary_of_managers_l665_665126


namespace orchestra_members_count_l665_665827

theorem orchestra_members_count :
  ∃ n ∈ set.Icc 150 300, 
    (n % 6 = 1) ∧ 
    (n % 8 = 3) ∧ 
    (n % 9 = 5) ∧ 
    n = 211 :=
by
  sorry

end orchestra_members_count_l665_665827


namespace ratio_am_to_eu_swallow_l665_665435

theorem ratio_am_to_eu_swallow
  (max_weight_am : ℕ)
  (max_weight_eu : ℕ)
  (total_swallow : ℕ)
  (max_combined_weight : ℕ)
  (h1 : max_weight_am = 5)
  (h2 : max_weight_eu = 10)
  (h3 : total_swallow = 90)
  (h4 : max_combined_weight = 600)
  : (2:1) :=
by
  sorry

end ratio_am_to_eu_swallow_l665_665435


namespace find_a_l665_665664

noncomputable def f (a : ℝ) : ℝ → ℝ := λ x, (a * x) / (x^2 + 3)

theorem find_a (a : ℝ) (h : (derivative (f a) 1) = 1/2) : a = 4 :=
by {
  sorry
}

end find_a_l665_665664


namespace cube_volume_l665_665905

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l665_665905


namespace lana_trip_longer_by_25_percent_l665_665468

-- Define the dimensions of the rectangular field
def length_field : ℕ := 3
def width_field : ℕ := 1

-- Define Tom's path distance
def tom_path_distance : ℕ := length_field + width_field

-- Define Lana's path distance
def lana_path_distance : ℕ := 2 + 1 + 1 + 1

-- Define the percentage increase calculation
def percentage_increase (initial final : ℕ) : ℕ :=
  (final - initial) * 100 / initial

-- Define the theorem to be proven
theorem lana_trip_longer_by_25_percent :
  percentage_increase tom_path_distance lana_path_distance = 25 :=
by
  sorry

end lana_trip_longer_by_25_percent_l665_665468


namespace binomial_coefficient_computation_l665_665083

open Nat

theorem binomial_coefficient_computation (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : k < n) :
  Nat.choose n k = (list.prod (list.range' (n - k + 1) k)) / k.factorial := by
sorry

end binomial_coefficient_computation_l665_665083


namespace problem_statement_l665_665300

noncomputable def f (x : ℝ) : ℝ := 
if x ∈ Set.Ioo (-π / 2) (π / 2) then
  exp x + sin x
else
  exp (π - x) + sin (π - x)

theorem problem_statement :
  f (5 * π / 6) < f (π / 4) ∧ f (π / 4) < f (π / 3) :=
by
  sorry

end problem_statement_l665_665300


namespace cube_volume_from_surface_area_l665_665920

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l665_665920


namespace net_rate_of_pay_l665_665129

theorem net_rate_of_pay
  (hours_travelled : ℕ)
  (speed : ℕ)
  (fuel_efficiency : ℕ)
  (pay_per_mile : ℝ)
  (price_per_gallon : ℝ)
  (net_rate_of_pay : ℝ) :
  hours_travelled = 3 →
  speed = 50 →
  fuel_efficiency = 25 →
  pay_per_mile = 0.60 →
  price_per_gallon = 2.50 →
  net_rate_of_pay = 25 := by
  sorry

end net_rate_of_pay_l665_665129


namespace incorrect_statement_of_rhombus_l665_665099

-- Conditions
def is_parallelogram (q : Type) : Prop := 
  ∀ (a b c d : q), (a = c ∧ b = d) → (// parallelogram conditions on a, b, c, d)

def is_rhombus (q : Type) : Prop := 
  ∀ (a b c d : q), (a ⊥ b ∧ c ⊥ d) → (// rhombus conditions on a, b, c, d)

def median_of_triangle (t : Type) : Prop := 
  ∀ (a b c : t), median (a, b, c) → (// median conditions on a, b, c)

def median_of_right_triangle (t : Type) : Prop := 
  ∀ (a b c : t), right_triangle (a, b, c) → (// median on hypotenuse conditions on a, b, c)

-- Statement
theorem incorrect_statement_of_rhombus (q : Type) (t : Type) :
  ∀ (a b c d : q), 
    ¬is_rhombus q ∧ is_parallelogram q ∧ median_of_triangle t ∧ median_of_right_triangle t → 
    incorrect_statement B :=
sorry

end incorrect_statement_of_rhombus_l665_665099


namespace lyapunov_stability_l665_665560

theorem lyapunov_stability {x y : ℝ → ℝ} 
  (h_sys1 : ∀ t, deriv x t = -y t)
  (h_sys2 : ∀ t, deriv y t = x t)
  (h_init : x 0 = 0 ∧ y 0 = 0) :
  ∀ ε > 0, ∃ δ > 0, ∀ (t ≥ 0), (|x t| < ε ∧ |y t| < ε) :=
sorry

end lyapunov_stability_l665_665560


namespace max_magnitude_l665_665292

variables {α : Type*} [inner_product_space ℝ α]

-- Define the unit vectors and their conditions
variables (a b : α)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1)
variables (h_ab : ⟪a, b⟫ = 0)

-- Define the vectors
def vA := (1 / 2) • a + (1 / 2) • b
def vB := (1 / 3) • a + (2 / 3) • b
def vC := (3 / 4) • a + (1 / 4) • b
def vD := (-1 / 5) • a + (6 / 5) • b

-- The statement we want to prove
theorem max_magnitude :
  ∥vD∥ ≥ ∥vA∥ ∧ ∥vD∥ ≥ ∥vB∥ ∧ ∥vD∥ ≥ ∥vC∥ :=
sorry

end max_magnitude_l665_665292


namespace train_crosses_signal_pole_in_18_seconds_l665_665501

/--
A 300-meter train crosses a 400-meter platform in 42 seconds.
Prove that the time taken for the train to cross a signal pole is 18 seconds.
-/
theorem train_crosses_signal_pole_in_18_seconds (length_train length_platform time_cross_platform : ℝ)
  (h_train : length_train = 300)
  (h_platform : length_platform = 400)
  (h_time : time_cross_platform = 42) : 
  let distance := length_train + length_platform in 
  let speed := distance / time_cross_platform in
  let time_to_cross_pole := length_train / speed in 
  time_to_cross_pole = 18 :=
by 
  sorry

end train_crosses_signal_pole_in_18_seconds_l665_665501


namespace range_of_a_if_p_and_not_q_l665_665638

open Real

def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a^2 = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

theorem range_of_a_if_p_and_not_q : 
  (∃ a : ℝ, (p a ∧ ¬q a)) → 
  (∀ a : ℝ, (p a ∧ ¬q a) → (-1 ≤ a ∧ a < 0)) :=
sorry

end range_of_a_if_p_and_not_q_l665_665638


namespace series_evaluation_l665_665215

noncomputable def series_sum : ℝ :=
  ∑' m : ℕ, (∑' n : ℕ, (m^2 * n) / (3^m * (n * 3^m + m * 3^n)))

theorem series_evaluation : series_sum = 9 / 32 :=
by
  sorry

end series_evaluation_l665_665215


namespace function_satisfies_equation_l665_665734

noncomputable def f (x : ℝ) : ℝ :=
  if x = 0.5 then 0.5 else 1 / (0.5 - x)

theorem function_satisfies_equation :
  ∀ x : ℝ, f x + (0.5 + x) * f (1 - x) = 1 :=
by
  intro x
  unfold f
  split_ifs
  case h =>
    sorry -- Proof when x = 0.5
  case h_1 =>
    sorry -- Proof when x ≠ 0.5

end function_satisfies_equation_l665_665734


namespace John_finishes_at_610PM_l665_665370

def TaskTime : Nat := 55
def StartTime : Nat := 14 * 60 + 30 -- 2:30 PM in minutes
def EndSecondTask : Nat := 16 * 60 + 20 -- 4:20 PM in minutes

theorem John_finishes_at_610PM (h1 : TaskTime * 2 = EndSecondTask - StartTime) : 
  (EndSecondTask + TaskTime * 2) = (18 * 60 + 10) :=
by
  sorry

end John_finishes_at_610PM_l665_665370


namespace part_a_root_sets_part_b_integer_pairs_count_l665_665764

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 2

theorem part_a_root_sets :
  (λ f, {x | f x = x} = {1, 2}) (λ x, x^2 - 3*x + 2) ∧
  (λ f, {x | (f ∘ f) x = x} = {1, 2}) (λ x, x^4 - 4*x^3 + 6*x^2 - 5*x + 2) :=
begin
  -- proof skipped
  sorry
end

noncomputable def f_gen (x : ℝ) (a b : ℝ) : ℝ := x^2 - a * x + b

theorem part_b_integer_pairs_count :
  (∃! (n : ℕ), n = 43) :=
begin
  -- proof skipped
  sorry
end

end part_a_root_sets_part_b_integer_pairs_count_l665_665764


namespace surface_area_of_circumsphere_around_tetrahedron_l665_665724

noncomputable def circumsphere_surface_area_tetrahedron 
  (a b c p : EuclideanSpace ℝ (Fin 3)) 
  (side_length : ℝ)
  (dihedral_angle : Real.Angle) :
  ℝ :=
  if (side_length = 6) ∧ (∠(p - a) = 120) then 
    84 * Real.pi 
  else 
    0

theorem surface_area_of_circumsphere_around_tetrahedron 
  (a b c p : EuclideanSpace ℝ (Fin 3)) 
  (side_length : ℝ := 6)
  (dihedral_angle : Real.Angle := 120) :
  circumsphere_surface_area_tetrahedron a b c p side_length dihedral_angle = 84 * Real.pi := by 
  sorry

end surface_area_of_circumsphere_around_tetrahedron_l665_665724


namespace value_of_x_l665_665684

theorem value_of_x (x : ℝ) : (2 : ℝ) = 1 / (4 * x + 2) → x = -3 / 8 := 
by
  intro h
  sorry

end value_of_x_l665_665684


namespace finite_lattice_points_on_line_l_l665_665360

def point (α : Type*) := (α × α)

def line_through (p1 p2 : point ℝ) : set (point ℝ) :=
{ p : point ℝ | ∃ t : ℝ, p.1 = p1.1 + t * (p2.1 - p1.1) ∧ p.2 = p1.2 + t * (p2.2 - p1.2) }

def is_lattice_point (p : point ℤ) : point ℝ := (p.1, p.2)

def is_finite_lattice_points (line : set (point ℝ)) : Prop :=
∃ S : set (point ℤ), finite S ∧ ∀ p : point ℤ, is_lattice_point p ∈ line → p ∈ S

theorem finite_lattice_points_on_line_l :
  let A := (-2 * Real.sqrt 2, -1 + Real.sqrt 2) in
  let B := (0, Real.sqrt 2) in
  is_finite_lattice_points (line_through A B) :=
sorry

end finite_lattice_points_on_line_l_l665_665360


namespace alice_probability_multiple_of_4_l665_665550

noncomputable def probability_one_multiple_of_4 (choices : ℕ) : ℚ :=
  let p_not_multiple_of_4 : ℚ := 45 / 60
  let p_all_not_multiple_of_4 : ℚ := p_not_multiple_of_4 ^ choices
  1 - p_all_not_multiple_of_4

theorem alice_probability_multiple_of_4 :
  probability_one_multiple_of_4 3 = 37 / 64 :=
by
  sorry

end alice_probability_multiple_of_4_l665_665550


namespace solution_set_l665_665577

noncomputable def f : ℝ → ℝ := sorry
axiom f'_lt_one_third (x : ℝ) : deriv f x < 1 / 3
axiom f_at_two : f 2 = 1

theorem solution_set : {x : ℝ | 0 < x ∧ x < 4} = {x : ℝ | f (Real.logb 2 x) > (Real.logb 2 x + 1) / 3} :=
by
  sorry

end solution_set_l665_665577


namespace sum_of_three_squares_l665_665166

theorem sum_of_three_squares (s t : ℤ) (h1 : 3 * s + 2 * t = 27)
                             (h2 : 2 * s + 3 * t = 23) (h3 : s + 2 * t = 13) :
  3 * s = 21 :=
sorry

end sum_of_three_squares_l665_665166


namespace cube_volume_l665_665895

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l665_665895


namespace solution_exists_x_pos_l665_665680

theorem solution_exists_x_pos (x : ℝ) (hx_pos : 0 < x) :
    (√(12 * x) * √(5 * x) * √(7 * x) * √(21 * x) = 42) ↔ x = √(21 / 47) :=
by
  sorry

end solution_exists_x_pos_l665_665680


namespace cube_volume_is_1728_l665_665944

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l665_665944


namespace expected_number_of_liking_pairs_l665_665263

theorem expected_number_of_liking_pairs
  (students : Finset ℕ) 
  (h_count : students.card = 15)
  (p : ℝ)
  (h_prob : p = 0.6) :
  (0.6 * (students.card.choose 2) = 63) :=
by 
  rw [h_count, h_prob];
  norm_num;
  sorry

end expected_number_of_liking_pairs_l665_665263


namespace magnitude_a_eq_2_times_magnitude_b_a_perp_b_find_t_l665_665723

-- Step 1: Define vectors
def a := (Real.sqrt 3, -1 : ℝ × ℝ)
def b := (1/2, Real.sqrt 3 / 2 : ℝ × ℝ)

-- Step 2: Prove magnitudes
theorem magnitude_a_eq_2_times_magnitude_b :
  Real.sqrt ((√3)^2 + (-1)^2) = 2 * Real.sqrt ((1/2)^2 + (Real.sqrt 3 / 2)^2) :=
by sorry

-- Step 3: Prove orthogonality
theorem a_perp_b : 
  (√3 * (1/2) - 1 * (Real.sqrt 3 / 2)) = 0 :=
by sorry

-- Step 4: Define x and y vectors with parameter t
def x (t : ℝ) := (Real.sqrt 3, -1) + ((t - 3) * 1/2, (t - 3) * Real.sqrt 3 / 2)
def y (t : ℝ) := (-(Real.sqrt 3), -1) + (t * 1 / 2, t * Real.sqrt 3 / 2)

-- Step 5: Prove condition and find t
theorem find_t (t : ℝ) :
  ((Real.sqrt 3, -1) + ((t - 3)*1/2, (t - 3) * Real.sqrt 3 / 2))•(-(Real.sqrt 3, -1) + (t * 1/2, t * Real.sqrt 3 / 2)) = 0 →
  (t = -1 ∨ t = 4) :=
by sorry

end magnitude_a_eq_2_times_magnitude_b_a_perp_b_find_t_l665_665723


namespace period_and_max_value_l665_665474

-- Define the function y
def y (x : Real) : Real := cos (x / 3) + 2

-- Prove that y has a period of 6*pi and maximum value of 3
theorem period_and_max_value :
  (∀ x : Real, y (x + 6 * π) = y x) ∧ (∀ x : Real, y x ≤ 3) ∧ (∃ x : Real, y x = 3) :=
by
  sorry

end period_and_max_value_l665_665474


namespace find_interest_rate_l665_665593

noncomputable def principal := 8000
noncomputable def compound_interest := 484.76847061839544
noncomputable def time_period := 1.5
noncomputable def times_compounded := 2
noncomputable def amount := 8484.76847061839544

theorem find_interest_rate :
  ∃ r : ℝ, amount = principal * (1 + r / times_compounded) ^ (times_compounded * time_period) ∧ r = 0.0397350993377484 :=
sorry

end find_interest_rate_l665_665593


namespace snail_meeting_times_correct_l665_665538

noncomputable def snail_meeting_times : list string :=
  let start_time := 0 -- 12:00 PM in minutes
  let full_circle_time := 120 -- 2 hours in minutes
  let snail_speed := 180 / full_circle_time -- degrees per minute (3 degrees/min)
  let minute_hand_speed := 360 / 60 -- degrees per minute (6 degrees/min)
  let first_encounter := 40 -- 12:40 PM in minutes
  let second_encounter := 40 + 40 -- 1:20 PM in minutes
  ["12:40 PM", "1:20 PM"]

theorem snail_meeting_times_correct : snail_meeting_times = ["12:40 PM", "1:20 PM"] :=
  by
    sorry

end snail_meeting_times_correct_l665_665538


namespace units_digit_31_2020_units_digit_37_2020_l665_665111

theorem units_digit_31_2020 : ((31 ^ 2020) % 10) = 1 := by
  sorry

theorem units_digit_37_2020 : ((37 ^ 2020) % 10) = 1 := by
  sorry

end units_digit_31_2020_units_digit_37_2020_l665_665111


namespace rectangle_ratio_l665_665525

open Real

theorem rectangle_ratio (A B C D E : Point) (rat : ℚ) : 
  let area_rect := 1
  let area_pentagon := (7 / 10 : ℚ)
  let area_triangle_AEC := 3 / 10
  let area_triangle_ECD := 1 / 5
  let x := 3 * EA
  let y := 2 * EA
  let diag_longer_side := sqrt (5 * EA ^ 2)
  let diag_shorter_side := EA * sqrt 5
  let ratio := sqrt 5 
  ( area_pentagon == area_rect * (7 / 10) ) →
  ( area_triangle_AEC + area_pentagon = area_rect ) →
  ( area_triangle_AEC == area_rect - area_pentagon ) →
  ( ratio == diag_longer_side / diag_shorter_side ) :=
  sorry

end rectangle_ratio_l665_665525


namespace cube_volume_l665_665931

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l665_665931


namespace range_of_function_x_geq_0_l665_665679

theorem range_of_function_x_geq_0 :
  ∀ (x : ℝ), x ≥ 0 → ∃ (y : ℝ), y ≥ 3 ∧ (y = x^2 + 2 * x + 3) :=
by
  sorry

end range_of_function_x_geq_0_l665_665679


namespace failed_english_is_45_l665_665348

-- Definitions of the given conditions
def total_students : ℝ := 1 -- representing 100%
def failed_hindi : ℝ := 0.35
def failed_both : ℝ := 0.2
def passed_both : ℝ := 0.4

-- The goal is to prove that the percentage of students who failed in English is 45%

theorem failed_english_is_45 :
  let failed_at_least_one := total_students - passed_both
  let failed_english := failed_at_least_one - failed_hindi + failed_both
  failed_english = 0.45 :=
by
  -- The steps and manipulation will go here, but for now we skip with sorry
  sorry

end failed_english_is_45_l665_665348


namespace problems_per_page_l665_665006

-- Define the initial conditions
def total_problems : ℕ := 101
def finished_problems : ℕ := 47
def remaining_pages : ℕ := 6

-- State the theorem
theorem problems_per_page : 54 / remaining_pages = 9 :=
by
  -- Sorry is used to ignore the proof step
  sorry

end problems_per_page_l665_665006


namespace total_buttons_l665_665540

theorem total_buttons (green buttons: ℕ) (yellow buttons: ℕ) (blue buttons: ℕ) (total buttons: ℕ) 
(h1: green = 90) (h2: yellow = green + 10) (h3: blue = green - 5) : total = green + yellow + blue → total = 275 :=
by
  sorry

end total_buttons_l665_665540


namespace max_score_43_l665_665195

def scores : List ℕ := [10, 7, 6, 8, 5, 9, 8, 8, 5, 6]
def times : List ℚ := [2/3, 1/2, 1/3, 2/3, 1/4, 2/3, 1/2, 2/5, 1/5, 1/4]
def costs : List ℕ := [1000, 700, 300, 800, 200, 900, 900, 600, 400, 600]

def isCorrect (vector : List Bool) : Prop :=
  vector.length = 10 ∧
  (vector.zip times).filter (λ p, p.fst).map (λ p, p.snd).sum < (3 : ℚ) ∧
  (vector.zip costs).filter (λ p, p.fst).map (λ p, p.snd).sum ≤ 3500

def totalScore (vector : List Bool) : ℕ :=
  (vector.zip scores).filter (λ p, p.fst).map (λ p, p.snd).sum

theorem max_score_43 :
  ∃ (vector : List Bool), isCorrect vector ∧ totalScore vector = 43 :=
by
  sorry

end max_score_43_l665_665195


namespace matilda_jellybeans_l665_665403

theorem matilda_jellybeans (steve_jellybeans : ℕ) (h_steve : steve_jellybeans = 84)
  (h_matt : ℕ) (h_matt_calc : h_matt = 10 * steve_jellybeans)
  (h_matilda : ℕ) (h_matilda_calc : h_matilda = h_matt / 2) :
  h_matilda = 420 := by
  sorry

end matilda_jellybeans_l665_665403


namespace polynomial_coefficients_sum_l665_665274

theorem polynomial_coefficients_sum :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ), 
  (∀ x : ℚ, (3 * x - 2)^9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 +
                            a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + 
                            a_7 * x^7 + a_8 * x^8 + a_9 * x^9) →
  (a_0 = -512) →
  ((a_0 + a_1 * (1/3) + a_2 * (1/3)^2 + a_3 * (1/3)^3 + 
    a_4 * (1/3)^4 + a_5 * (1/3)^5 + a_6 * (1/3)^6 + 
    a_7 * (1/3)^7 + a_8 * (1/3)^8 + a_9 * (1/3)^9) = -1) →
  (a_1 / 3 + a_2 / 3^2 + a_3 / 3^3 + a_4 / 3^4 + a_5 / 3^5 + 
   a_6 / 3^6 + a_7 / 3^7 + a_8 / 3^8 + a_9 / 3^9 = 511) :=
by 
  -- The proof would go here
  sorry

end polynomial_coefficients_sum_l665_665274


namespace parallelogram_area_l665_665749

open Real

variables (p q : EuclideanSpace ℝ (Fin 3))
hypothesis (norm_p : ‖p‖ = 1)
hypothesis (norm_q : ‖q‖ = 1)
hypothesis (angle_pq : real.arccos (inner p q / (‖p‖ * ‖q‖)) = π / 4)

theorem parallelogram_area : 
  ∥(p - q) × (p + q)∥ / 2 = √2 / 4 :=
by
  sorry

end parallelogram_area_l665_665749


namespace minimum_value_of_x_l665_665326

theorem minimum_value_of_x (x : ℝ) (h_pos : x > 0) (h_log : log 3 x ≥ log 3 9 - (1 / 3) * log 3 x) : x ≥ Real.sqrt 27 :=
sorry

end minimum_value_of_x_l665_665326


namespace exists_m_such_that_m_poly_is_zero_mod_p_l665_665268

theorem exists_m_such_that_m_poly_is_zero_mod_p (p : ℕ) (hp_prime : Nat.Prime p) (hp_mod : p % 7 = 1) :
  ∃ m : ℕ, m > 0 ∧ (m^3 + m^2 - 2*m - 1) % p = 0 := 
sorry

end exists_m_such_that_m_poly_is_zero_mod_p_l665_665268


namespace sum_of_digits_in_7_pow_1500_l665_665094

-- Define the problem and conditions
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10
def sum_of_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

theorem sum_of_digits_in_7_pow_1500 :
  sum_of_digits (7^1500) = 2 :=
by
  sorry

end sum_of_digits_in_7_pow_1500_l665_665094


namespace smallest_shift_a_l665_665797

theorem smallest_shift_a {f : ℝ → ℝ} (hf : ∀ x, f (x - 12) = f x) : ∃ a > 0, (∀ x, f ((x - a) / 3) = f (x / 3)) ∧ a = 36 :=
by {
  use 36,
  split,
  { exact by norm_num },
  { intro x,
    have : (x - 36) / 3 = x / 3 - 12, by ring,
    rw [this, ←hf (x/3)],
  }
} sorry

end smallest_shift_a_l665_665797


namespace cube_volume_is_1728_l665_665947

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l665_665947


namespace find_x_value_l665_665070

def acid_solution (m : ℕ) (x : ℕ) (h : m > 25) : Prop :=
  let initial_acid := m^2 / 100
  let total_volume := m + x
  let new_acid_concentration := (m - 5) / 100 * (m + x)
  initial_acid = new_acid_concentration

theorem find_x_value (m : ℕ) (h : m > 25) (x : ℕ) :
  (acid_solution m x h) → x = 5 * m / (m - 5) :=
sorry

end find_x_value_l665_665070


namespace odd_square_mod_eight_l665_665014

theorem odd_square_mod_eight (k : ℤ) : ((2 * k + 1) ^ 2) % 8 = 1 := 
sorry

end odd_square_mod_eight_l665_665014


namespace employee_pay_per_week_l665_665851

theorem employee_pay_per_week (total_pay : ℝ) (ratio : ℝ) (pay_b : ℝ)
  (h1 : total_pay = 570)
  (h2 : ratio = 1.5)
  (h3 : total_pay = pay_b * (ratio + 1)) :
  pay_b = 228 :=
sorry

end employee_pay_per_week_l665_665851


namespace smallest_k_l665_665864

theorem smallest_k (k : ℕ) : 128 ^ k > 8 ^ 25 + 1000 ↔ k ≥ 11 :=
by
  have h1 : 128 = 2 ^ 7 := by norm_num
  have h2 : 8 = 2 ^ 3 := by norm_num
  have h3 : 128 ^ k = (2 ^ 7) ^ k := by rw [h1]
  have h4 : 8 ^ 25 = (2 ^ 3) ^ 25 := by rw [h2]
  have h5 : (2 ^ 7) ^ k = 2 ^ (7 * k) := by rw [pow_mul]
  have h6 : (2 ^ 3) ^ 25 = 2 ^ (3 * 25) := by rw [pow_mul]
  rw [h3, h5, h4, h6]
  split
  · intro h
    have : 2 ^ (7 * k) > 2 ^ 75 := lt_of_le_of_lt (nat.pow_le_pow_of_le_right zero_lt_two (nat.lt_add_of_pos_right zero_lt_one)) h
    have : 7 * k > 75 := lt_of_le_of_ne this (pow_eq_pow _)
    linarith
  · intro h
    calc
      2 ^ (7 * k) ≥ 2 ^ 77 : pow_le_pow_of_le_right (zero_lt_two) (mul_le_mul_right' (by norm_num) _)
      ... > 2 ^ 75 + 1000 : by linarith (pow_lt_pow _ 75 77)

end smallest_k_l665_665864


namespace airplane_altitude_l665_665548

theorem airplane_altitude (d_Alice_Bob : ℝ) (angle_Alice : ℝ) (angle_Bob : ℝ) (altitude : ℝ) : 
  d_Alice_Bob = 8 ∧ angle_Alice = 45 ∧ angle_Bob = 30 → altitude = 16 / 3 :=
by
  intros h
  rcases h with ⟨h1, ⟨h2, h3⟩⟩
  -- you may insert the proof here if needed
  sorry

end airplane_altitude_l665_665548


namespace amaya_movie_watching_time_l665_665167

theorem amaya_movie_watching_time :
  let uninterrupted_time_1 := 35
  let uninterrupted_time_2 := 45
  let uninterrupted_time_3 := 20
  let rewind_time_1 := 5
  let rewind_time_2 := 15
  let total_uninterrupted := uninterrupted_time_1 + uninterrupted_time_2 + uninterrupted_time_3
  let total_rewind := rewind_time_1 + rewind_time_2
  let total_time := total_uninterrupted + total_rewind
  total_time = 120 := by
  sorry

end amaya_movie_watching_time_l665_665167


namespace find_D_coordinates_l665_665037

theorem find_D_coordinates:
  ∀ (A B C : (ℝ × ℝ)), 
  A = (-2, 5) ∧ C = (3, 7) ∧ B = (-3, 0) →
  ∃ D : (ℝ × ℝ), D = (2, 2) :=
by
  sorry

end find_D_coordinates_l665_665037


namespace question1_question2_l665_665509

def energy_cost (units: ℕ) : ℝ :=
  if units <= 100 then
    units * 0.5
  else
    100 * 0.5 + (units - 100) * 0.8

theorem question1 :
  energy_cost 130 = 74 := by
  sorry

theorem question2 (units: ℕ) (H: energy_cost units = 90) :
  units = 150 := by
  sorry

end question1_question2_l665_665509


namespace arithmetic_mean_geom_mean_ratio_l665_665332

theorem arithmetic_mean_geom_mean_ratio {a b : ℝ} (h1 : (a + b) / 2 = 3 * Real.sqrt (a * b)) (h2 : a > b) (h3 : b > 0) : 
  (∃ k : ℤ, k = 34 ∧ abs ((a / b) - 34) ≤ 0.5) :=
sorry

end arithmetic_mean_geom_mean_ratio_l665_665332


namespace cube_volume_from_surface_area_l665_665870

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l665_665870


namespace smallest_four_digit_divisible_43_l665_665091

theorem smallest_four_digit_divisible_43 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 43 = 0 ∧ n = 1032 :=
by
  sorry

end smallest_four_digit_divisible_43_l665_665091


namespace range_of_m_l665_665623

theorem range_of_m (p q : Prop) (m : ℝ) (h₀ : ∀ x : ℝ, p ↔ (x^2 - 8 * x - 20 ≤ 0)) 
  (h₁ : ∀ x : ℝ, q ↔ (x^2 - 2 * x + 1 - m^2 ≤ 0)) (hm : m > 0) 
  (hsuff : (∃ x : ℝ, x > 10 ∨ x < -2) → (∃ x : ℝ, x < 1 - m ∨ x > 1 + m)) :
  0 < m ∧ m ≤ 3 :=
sorry

end range_of_m_l665_665623


namespace equilateral_triangle_area_l665_665828

theorem equilateral_triangle_area (p : ℝ) (h : p > 0) :
    let x := 2 * p / 3 in
    let area := (sqrt 3 / 4) * x^2 in
    area = (sqrt 3 * p^2) / 9 :=
by
  sorry

end equilateral_triangle_area_l665_665828


namespace log_equation_solution_l665_665424

theorem log_equation_solution (x : ℝ) : log 8 x + log 2 (x ^ 3) = 9 → x = 2 ^ 2.7 :=
by
  sorry

end log_equation_solution_l665_665424


namespace equation_of_line_l665_665449

theorem equation_of_line (A B : ℝ × ℝ) (M : ℝ × ℝ) (hM : M = (-1, 2)) (hA : A.2 = 0) (hB : B.1 = 0) (hMid : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  ∃ (a b c : ℝ), (a = 2 ∧ b = -1 ∧ c = 4) ∧ ∀ (x y : ℝ), y = a * x + b * y + c → 2 * x - y + 4 = 0 := 
  sorry

end equation_of_line_l665_665449


namespace line_through_intersections_of_tangents_l665_665845

noncomputable def point_inside_ellipse (x y : ℝ) :=
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 < 1)

theorem line_through_intersections_of_tangents (x y : ℝ) :
  point_inside_ellipse (sqrt 5) (sqrt 2) →
  (∃ k1 k2 : ℝ, k1 * x / 9 + k2 * y / 5 = 1) →
  (\sqrt{5}/9) * x + (\sqrt{2}/5) * y = 1 := sorry

end line_through_intersections_of_tangents_l665_665845


namespace find_a_l665_665359

-- Define the structure of a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ
deriving DecidableEq

-- Define the points A, B, and C
def A : Point3D := {x := 1, y := 0, z := 2}
def B : Point3D := {x := 2, y := 1, z := 0}
def C (a : ℝ) : Point3D := {x := 0, y := a, z := 1}

-- Define the vector between two points
def vector (P Q : Point3D) : ℝ × ℝ × ℝ := 
  (Q.x - P.x, Q.y - P.y, Q.z - P.z)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Condition and proof statement for perpendicular vectors pointing to a being -1.
theorem find_a (a : ℝ) (h : dot_product (vector A B) (vector A (C a)) = 0) : a = -1 :=
  by sorry

end find_a_l665_665359


namespace factor_expression_l665_665224

noncomputable def factored_expression (x : ℝ) : ℝ :=
  5 * x * (x + 2) + 9 * (x + 2)

theorem factor_expression (x : ℝ) : 
  factored_expression x = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l665_665224


namespace car_speed_is_104_mph_l665_665118

noncomputable def speed_of_car_in_mph
  (fuel_efficiency_km_per_liter : ℝ) -- car travels 64 km per liter
  (fuel_consumption_gallons : ℝ) -- fuel tank decreases by 3.9 gallons
  (time_hours : ℝ) -- period of 5.7 hours
  (gallon_to_liter : ℝ) -- 1 gallon is 3.8 liters
  (km_to_mile : ℝ) -- 1 mile is 1.6 km
  : ℝ :=
  let fuel_consumption_liters := fuel_consumption_gallons * gallon_to_liter
  let distance_km := fuel_efficiency_km_per_liter * fuel_consumption_liters
  let distance_miles := distance_km / km_to_mile
  let speed_mph := distance_miles / time_hours
  speed_mph

theorem car_speed_is_104_mph 
  (fuel_efficiency_km_per_liter : ℝ := 64)
  (fuel_consumption_gallons : ℝ := 3.9)
  (time_hours : ℝ := 5.7)
  (gallon_to_liter : ℝ := 3.8)
  (km_to_mile : ℝ := 1.6)
  : speed_of_car_in_mph fuel_efficiency_km_per_liter fuel_consumption_gallons time_hours gallon_to_liter km_to_mile = 104 :=
  by
    sorry

end car_speed_is_104_mph_l665_665118


namespace complex_pow_cos_sin_l665_665410

theorem complex_pow_cos_sin (n : ℕ) (h : n > 0) :
  ((1 / 2 : ℂ) + (complex.I * (real.sqrt 3 / 2))) ^ n =
  complex.cos (n * real.pi / 3) + complex.I * complex.sin (n * real.pi / 3) :=
sorry

end complex_pow_cos_sin_l665_665410


namespace cube_volume_from_surface_area_l665_665921

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l665_665921


namespace original_price_l665_665831

theorem original_price (P: ℝ) (h: 0.80 * 1.15 * P = 46) : P = 50 :=
by sorry

end original_price_l665_665831


namespace correct_statements_l665_665098

def Polyhedron (P : Type) := 
  ∀ (s : P), exists (x y : P), Parallel x y

def Prism (P : Type) := 
  Polyhedron P ∧ 
  ∀ (l : P), LateralEdgesEqual l ∧ LateralFacesParallelograms l

def Cylinder (C : Type) := 
  ∀ (plane : C), CrossSectionIsCircleOrRectangle plane

theorem correct_statements (P : Type) (C : Type) :
  (Prism P → Polyhedron P ∧ (∀ l, LateralEdgesEqual l ∧ LateralFacesParallelograms l)) ∧
  (Polyhedron P → exists x y : P, Parallel x y) ∧
  ¬(Polyhedron P ∧ (∃ x y : P, Parallel x y ∧ Trapezoidal x y)) ∧
  ¬(Cylinder C ∧ (forall plane, CrossSectionIsCircleOrRectangle plane ∨ CrossSectionIsEllipse plane)) :=
by 
  sorry

end correct_statements_l665_665098


namespace total_cost_l665_665138

theorem total_cost (cost_pencil cost_pen : ℕ) 
(h1 : cost_pen = cost_pencil + 9) 
(h2 : cost_pencil = 2) : 
cost_pencil + cost_pen = 13 := 
by 
  -- Proof would go here 
  sorry

end total_cost_l665_665138


namespace projection_vector_correct_l665_665289

noncomputable def vector_proj (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let b_magnitude_sq := b.1 * b.1 + b.2 * b.2 + b.3 * b.3
  let scale := dot_product / b_magnitude_sq
  (scale * b.1, scale * b.2, scale * b.3)

theorem projection_vector_correct :
  vector_proj (1, 0, 1) (1, 1, 1) = (2/3, 2/3, 2/3) :=
  sorry

end projection_vector_correct_l665_665289


namespace min_value_of_x_l665_665681

theorem min_value_of_x (x : ℝ) (h_pos : 0 < x) (h_ineq : log x ≥ log 2 + (1/2) * log x) : x ≥ 4 := by
  sorry

end min_value_of_x_l665_665681


namespace cube_volume_from_surface_area_l665_665960

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l665_665960


namespace pq_eq_oa_l665_665393

-- Definitions based on conditions
def ellipse_eq (a b : ℝ) : Prop := ∀ x y : ℝ, (x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1)
def focus (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 - b ^ 2)
def point_on_ellipse (a b : ℝ) (u v : ℝ) : Prop := u ^ 2 + v ^ 2 = 1
def parallel_tangent (a b x y : ℝ) : Prop := - (b ^ 2 * x) / (a ^ 2 * y)
def line_eq_through_center (a b u v : ℝ) : Prop := ∀ x : ℝ, (y = - (b * u / (a * v)) * x)  -- Chord CD
def line_eq_PF (a b c u v ε : ℝ) : Prop := ∀ x : ℝ, (y - b * v = (b * v / (a * u - ε * c)) * (x - a * u))

-- Theorem Statement
theorem pq_eq_oa (a b u v c ε : ℝ) (P : point_on_ellipse a b u v) (F : focus a b = c)
  (H_parallel : parallel_tangent a b u v) (H_cd : line_eq_through_center a b u v)
  (H_pf : line_eq_PF a b c u v ε) :
  let PQ := Real.sqrt ((a ^ 2 * u ^ 2 + b ^ 2 * v ^ 2 + c ^ 2 - 2 * ε * a * c * u)) in
  PQ = a := sorry
 
end pq_eq_oa_l665_665393


namespace magnitude_of_complex_l665_665302

def complex_magnitude (z : ℂ) : ℝ := complex.abs z

theorem magnitude_of_complex (i : ℂ) (h_i: i * i = -1) :
  complex_magnitude ((3 + i) ^ 2 * i) = 10 :=
by
  sorry

end magnitude_of_complex_l665_665302


namespace trigonometric_expression_l665_665643

variables {α β : ℝ}

theorem trigonometric_expression
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : tan (α + β) = -1) :
  (cos (β - α) - sin (α + β)) / (cos α * cos β) = 2 :=
by {
  -- Sorry placeholder indicates that the proof is omitted
  sorry
}

end trigonometric_expression_l665_665643


namespace non_congruent_squares_on_6_by_6_grid_l665_665676

theorem non_congruent_squares_on_6_by_6_grid : 
  let grid_size := 6 in
  let count_regular_squares := grid_size * (grid_size - 1) * (grid_size - 1 + 1) / 2 in
  let count_rotated_squares := grid_size * (grid_size - 1) * (grid_size - 1 + 1) / 2 in
  count_regular_squares + count_rotated_squares = 110 :=
by
  let grid_size := 6
  let count_regular_squares := (25 + 16 + 9 + 4 + 1)
  let count_rotated_squares := (25 + 16 + 9 + 4 + 1)
  trivial

end non_congruent_squares_on_6_by_6_grid_l665_665676


namespace factor_expression_l665_665219

theorem factor_expression (x : ℝ) : 5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l665_665219


namespace AD_perpendicular_EF_l665_665158

theorem AD_perpendicular_EF
  (ABC : Triangle)
  (A B C D E F : Point)
  (h1 : angle A B C < 45)
  (h2 : BD = CD)
  (h3 : ∠ B D C = 4 * ∠ B A C)
  (h4 : E = Reflection C A B)
  (h5 : F = Reflection B A C)
  : Perpendicular (Line_through A D) (Line_through E F) := 
sorry

end AD_perpendicular_EF_l665_665158


namespace inequality_solution_l665_665271

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x

theorem inequality_solution (k : ℝ) (hk : 0 < k) : 
  (∀ x : ℝ, f' x + k * (1 - x) * f x > 0 ↔ 
      if 0 < k ∧ k < 1 then 1 < x ∧ x < 1 / k else 
      if k = 1 then false else 
      if k > 1 then 1 / k < x ∧ x < 1 else false) :=
sorry

end inequality_solution_l665_665271


namespace quadratic_is_square_of_binomial_l665_665590

noncomputable def find_a : ℚ :=
let r := (18 / 8 : ℚ) in
let s := (4 : ℚ) in
r^2

theorem quadratic_is_square_of_binomial (a : ℚ) :
  (∃ r s : ℚ, r^2 = a ∧ 2 * r * s = 18 ∧ s^2 = 16) ↔ a = find_a :=
by
  sorry

end quadratic_is_square_of_binomial_l665_665590


namespace razorback_shop_jersey_sales_l665_665026

theorem razorback_shop_jersey_sales :
  let price_per_jersey := 165
  let jerseys_sold := 156
  price_per_jersey * jerseys_sold = 25740 :=
by
  let price_per_jersey := 165
  let jerseys_sold := 156
  show price_per_jersey * jerseys_sold = 25740, from
  sorry

end razorback_shop_jersey_sales_l665_665026


namespace cube_volume_of_surface_area_l665_665979

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l665_665979


namespace find_x_l665_665278

variable {x : ℝ}

def point_A := (3, 4) : ℝ × ℝ
def point_B := (x, 7) : ℝ × ℝ

def slope (p1 p2 : ℝ × ℝ) := (p2.2 - p1.2) / (p2.1 - p1.1)

theorem find_x (h_slope : slope point_A point_B = 3) : x = 4 :=
by
  -- Proof will go here
  sorry

end find_x_l665_665278


namespace nth_term_of_series_l665_665161

def algebraic_series (n : ℕ) : ℝ := 
  if h : n > 0 then (-1) ^ (n - 1) * (2 * n - 1) 
  else 0

theorem nth_term_of_series (n : ℕ) (h : n > 0) : 
  (x - 3 * x^2 + 5 * x^3 - 7 * x^4 + 9 * x^5 + ...) !! n = algebraic_series n * x^n :=
sorry

end nth_term_of_series_l665_665161


namespace number_of_valid_three_digit_numbers_l665_665015

-- Definitions based on conditions
def hundred_place_digits : Finset ℕ := {2, 4}
def odd_digits : Finset ℕ := {1, 3, 5}

-- Main statement, without proof
theorem number_of_valid_three_digit_numbers : 
  (∑ x in hundred_place_digits, (∑ y in odd_digits, (∑ z in odd_digits.erase y, 2))) = 12 := 
sorry

end number_of_valid_three_digit_numbers_l665_665015


namespace trigonometric_identity_l665_665206

theorem trigonometric_identity :
  (3 / (Real.sin (20 * Real.pi / 180))^2) - 
  (1 / (Real.cos (20 * Real.pi / 180))^2) + 
  64 * (Real.sin (20 * Real.pi / 180))^2 = 32 :=
by sorry

end trigonometric_identity_l665_665206


namespace determine_house_height_l665_665861

-- Definitions for the conditions
def house_shadow : ℚ := 75
def tree_height : ℚ := 15
def tree_shadow : ℚ := 20

-- Desired Height of Lily's house
def house_height : ℚ := 56

-- Theorem stating the height of the house
theorem determine_house_height :
  (house_shadow / tree_shadow = house_height / tree_height) -> house_height = 56 :=
  by
  unfold house_shadow tree_height tree_shadow house_height
  sorry

end determine_house_height_l665_665861


namespace cube_volume_from_surface_area_example_cube_volume_l665_665964

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l665_665964


namespace cube_volume_from_surface_area_l665_665915

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l665_665915


namespace cube_volume_from_surface_area_l665_665885

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l665_665885


namespace ricky_magic_money_box_l665_665785

theorem ricky_magic_money_box : 
  let penny_value (pennies : ℕ) (factor : ℕ) (add : ℕ) := pennies * factor + add in
  let coins_on_monday := 2 * 2 in -- 2 pennies doubled
  let coins_on_tuesday := penny_value coins_on_monday 3 5 in -- tripled + 5 pennies (nickel)
  let coins_on_wednesday := penny_value coins_on_tuesday 4 10 in -- quadrupled + 10 pennies (dime)
  let coins_on_thursday := penny_value coins_on_wednesday 5 25 in -- quintupled + 25 pennies (quarter)
  let coins_on_friday := penny_value coins_on_thursday 6 50 in -- sextupled + 50 pennies (half dollar)
  let coins_on_saturday := coins_on_friday * 7 in -- septupled
  let coins_on_sunday := coins_on_saturday * 8 in -- octupled
  coins_on_sunday = 142240 :=
by
  let penny_value (pennies : ℕ) (factor : ℕ) (add : ℕ) := pennies * factor + add in
  let coins_on_monday := 2 * 2 in 
  let coins_on_tuesday := penny_value coins_on_monday 3 5 in
  let coins_on_wednesday := penny_value coins_on_tuesday 4 10 in
  let coins_on_thursday := penny_value coins_on_wednesday 5 25 in
  let coins_on_friday := penny_value coins_on_thursday 6 50 in
  let coins_on_saturday := coins_on_friday * 7 in
  let coins_on_sunday := coins_on_saturday * 8 in
  sorry

end ricky_magic_money_box_l665_665785


namespace volleyball_team_arrangements_l665_665389

theorem volleyball_team_arrangements (n : ℕ) (n_pos : 0 < n) :
  ∃ arrangements : ℕ, arrangements = 2^n * (Nat.factorial n)^2 :=
sorry

end volleyball_team_arrangements_l665_665389


namespace smallest_value_condition_l665_665046

theorem smallest_value_condition 
  (a : Fin 8 → ℝ)
  (h_sum : ∑ i, a i = 4 / 3)
  (h_pos_sum : ∀ i, 0 < ∑ j, if j == i then 0 else a j) :
  -8 < (Finset.min' Finset.univ (λ i, a i)) ∧ (Finset.min' Finset.univ (λ i, a i)) ≤ 1 / 6 :=
by
  sorry

end smallest_value_condition_l665_665046


namespace cube_volume_from_surface_area_l665_665874

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l665_665874


namespace area_of_triangle_with_perimeter_11_l665_665151

noncomputable def semi_perimeter (a b c : ℕ) := (a + b + c) / 2

noncomputable def area_heron (a b c : ℕ) : ℝ := 
  real.sqrt (semi_perimeter a b c 
    * (semi_perimeter a b c - a) 
    * (semi_perimeter a b c - b) 
    * (semi_perimeter a b c - c))

theorem area_of_triangle_with_perimeter_11 : 
  ∀ (a b c : ℕ), a + b + c = 11 ∧ a + b > c ∧ a + c > b ∧ b + c > a 
  -> area_heron a b c = (5 * real.sqrt 11) / 4 :=
by
  intros a b c h
  sorry

end area_of_triangle_with_perimeter_11_l665_665151


namespace cube_volume_from_surface_area_example_cube_volume_l665_665969

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l665_665969


namespace find_line_equation_l665_665690

theorem find_line_equation 
  (A : ℝ × ℝ) (hA : A = (-2, -3)) 
  (h_perpendicular : ∃ k b : ℝ, ∀ x y, 3 * x + 4 * y - 3 = 0 → k * x + y = b) :
  ∃ k' b' : ℝ, (∀ x y, k' * x + y = b' → y = (4 / 3) * x + 1 / 3) ∧ (k' = 4 ∧ b' = -1) :=
by
  sorry

end find_line_equation_l665_665690


namespace outer_boundary_diameter_l665_665557

-- Definitions from the conditions
def fountain_diameter : ℝ := 10
def garden_ring_width : ℝ := 8
def walking_path_width : ℝ := 6

-- Prove the diameter of the outer boundary of the walking path
theorem outer_boundary_diameter (fountain_diameter garden_ring_width walking_path_width : ℝ) :
  2 * (fountain_diameter / 2 + garden_ring_width + walking_path_width) = 38 := 
by
  -- calculate the radius
  let radius := fountain_diameter / 2 + garden_ring_width + walking_path_width
  -- complete the proof
  have diameter := 2 * radius
  show diameter = 38 from sorry

end outer_boundary_diameter_l665_665557


namespace sum_digits_parity_polynomial_l665_665264

def sum_of_digits (n : ℕ) : ℕ :=
  (n.toString.data.map (λ c, c.toNat - '0'.toNat)).sum

theorem sum_digits_parity_polynomial (n : ℕ) (a : Fin n → ℕ) (h1 : 2 ≤ n) 
  (h2 : ∀ i : Fin n, 0 < a i) : ¬ ∀ k : ℕ, (sum_of_digits k % 2) = (sum_of_digits (∑ i, a i * k^i) % 2) :=
sorry

end sum_digits_parity_polynomial_l665_665264


namespace apples_per_pie_l665_665731

theorem apples_per_pie (total_apples : ℕ) (unripe_apples : ℕ) (pies : ℕ) (ripe_apples : ℕ)
  (H1 : total_apples = 34)
  (H2 : unripe_apples = 6)
  (H3 : pies = 7)
  (H4 : ripe_apples = total_apples - unripe_apples) :
  ripe_apples / pies = 4 := by
  sorry

end apples_per_pie_l665_665731


namespace student_2005_calls_out_1_l665_665338

def counting_pattern (n : ℕ) : ℕ :=
  let pattern := [1, 2, 3, 4, 3, 2, 1]
  pattern[(n - 1) % 7]

theorem student_2005_calls_out_1 :
  counting_pattern 2005 = 1 :=
by
  sorry

end student_2005_calls_out_1_l665_665338


namespace min_value_x_plus_2y_l665_665433

theorem min_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2y + 2 * x * y = 8) :
  x + 2y ≥ 4 :=
sorry

end min_value_x_plus_2y_l665_665433


namespace cube_volume_from_surface_area_l665_665878

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l665_665878


namespace cube_volume_l665_665913

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l665_665913


namespace cube_volume_from_surface_area_l665_665958

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l665_665958


namespace sequence_properties_l665_665632

variable {a : ℕ → ℤ}

-- Conditions
axiom seq_add : ∀ (p q : ℕ), 1 ≤ p → 1 ≤ q → a (p + q) = a p + a q
axiom a2_neg4 : a 2 = -4

-- Theorem statement: We need to prove a6 = -12 and a_n = -2n for all n
theorem sequence_properties :
  (a 6 = -12) ∧ ∀ n : ℕ, 1 ≤ n → a n = -2 * n :=
by
  sorry

end sequence_properties_l665_665632


namespace expression_for_f_l665_665621

variable {R : Type*} [CommRing R]

def f (x : R) : R := sorry

theorem expression_for_f (x : R) :
  (f (x-1) = x^2 + 4*x - 5) → (f x = x^2 + 6*x) := by
  sorry

end expression_for_f_l665_665621


namespace sum_MN_MK_eq_14_sqrt4_3_l665_665121

theorem sum_MN_MK_eq_14_sqrt4_3
  (MN MK : ℝ)
  (area: ℝ)
  (angle_LMN : ℝ)
  (h_area : area = 49)
  (h_angle_LMN : angle_LMN = 30) :
  MN + MK = 14 * (Real.sqrt (Real.sqrt 3)) :=
by
  sorry

end sum_MN_MK_eq_14_sqrt4_3_l665_665121


namespace min_a3_b3_eq_t3_div4_l665_665386

theorem min_a3_b3_eq_t3_div4 (a b t : ℝ) (h : a + b = t) : ∃ ab : ℝ, (a * b = ab ∧ ab ≤ t^2 / 4 ∧ min (a^3 + b^3) = t^3 / 4) :=
by sorry

end min_a3_b3_eq_t3_div4_l665_665386


namespace log_expression_in_terms_of_a_l665_665619

noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

variable (a : ℝ) (h : a = log3 2)

theorem log_expression_in_terms_of_a : log3 8 - 2 * log3 6 = a - 2 :=
by
  sorry

end log_expression_in_terms_of_a_l665_665619


namespace place_two_in_front_l665_665694

-- Define the conditions: the original number has hundreds digit h, tens digit t, and units digit u.
variables (h t u : ℕ)

-- Define the function representing the placement of the digit 2 before the three-digit number.
def new_number (h t u : ℕ) : ℕ :=
  2000 + 100 * h + 10 * t + u

-- State the theorem that proves the new number formed is as stated.
theorem place_two_in_front : new_number h t u = 2000 + 100 * h + 10 * t + u :=
by sorry

end place_two_in_front_l665_665694


namespace any_polyhedron_intersects_with_plane_to_form_triangle_l665_665470

theorem any_polyhedron_intersects_with_plane_to_form_triangle (P : Polyhedron) :
  ∃ plane : Plane, is_triangle (P ∩ plane) := sorry

end any_polyhedron_intersects_with_plane_to_form_triangle_l665_665470


namespace billboards_in_second_hour_l665_665414

theorem billboards_in_second_hour 
  (first_hour : ℕ) (third_hour : ℕ) (average : ℕ) (B : ℕ) 
  (h1 : first_hour = 17) 
  (h2 : third_hour = 23) 
  (h3 : average = 20) 
  (h4 : B = (average * 3) - (first_hour + third_hour)) : 
  B = 20 := 
by
  rw [h1, h2] at h4
  simp at h4
  exact h4

end billboards_in_second_hour_l665_665414


namespace sum_of_common_ratios_l665_665497

variables {k p r : ℝ}
variables (a_1 a_2 b_1 b_2 : ℝ)

-- Conditions
def sequence1 := ∃ k p, k ≠ 0 ∧ p ≠ 1 ∧ p ≠ 0 ∧ a_1 = k * p ∧ a_2 = k * p^2
def sequence2 := ∃ k r, k ≠ 0 ∧ r ≠ 1 ∧ r ≠ 0 ∧ b_1 = k * r ∧ b_2 = k * r^2

-- Given condition
def given_condition := a_2 - b_2 = 5 * (a_1 - b_1)

-- The theorem
theorem sum_of_common_ratios (hs1 : sequence1) (hs2 : sequence2) (h : given_condition) :
    p + r = 5 :=
by
  sorry

end sum_of_common_ratios_l665_665497


namespace niraek_donuts_covered_l665_665186

noncomputable def surface_area (r : ℕ) : ℝ := 4 * real.pi * (r ^ 2)

noncomputable def time_per_donut (r : ℕ) (rate : ℕ) : ℝ := surface_area r / rate

noncomputable def lcm_times (t1 t2 t3 t4 : ℝ) : ℝ :=
  let lcm_nat (a b : ℕ) := a * b / (nat.gcd a b)
  in real.lcm (lcm_nat t1.to_nat t2.to_nat) (lcm_nat t3.to_nat t4.to_nat)

noncomputable def donuts_covered (time_per_donut_Niraek total_time : ℝ) : ℕ :=
  (total_time / time_per_donut_Niraek).to_nat

theorem niraek_donuts_covered :
  let t_Niraek := time_per_donut 5 1
  let t_Theo := time_per_donut 7 2
  let t_Akshaj := time_per_donut 9 3
  let t_Luna := time_per_donut 11 4
  let total_time := lcm_times t_Niraek t_Theo t_Akshaj t_Luna
  donuts_covered t_Niraek total_time = 7203 :=
by
  sorry

end niraek_donuts_covered_l665_665186


namespace total_time_to_watch_movie_l665_665170

-- Define the conditions and the question
def uninterrupted_viewing_time : ℕ := 35 + 45 + 20
def rewinding_time : ℕ := 5 + 15
def total_time : ℕ := uninterrupted_viewing_time + rewinding_time

-- Lean statement of the proof problem
theorem total_time_to_watch_movie : total_time = 120 := by
  -- This is where the proof would go
  sorry

end total_time_to_watch_movie_l665_665170


namespace partition_exists_iff_divisible_by_3_l665_665297

variable (p q : ℕ)

/-- A division of positive integers exists if and only if p + q is divisible by 3 --/
theorem partition_exists_iff_divisible_by_3 
  (h_coprime : Nat.gcd p q = 1) 
  (h_pos_p : p > 0) 
  (h_pos_q : q > 0) 
  (h_ne : p ≠ q) :
  (∃ A B C : set ℕ, ∀ z : ℕ, z > 0 → 
      (z ∈ A ∧ z + p ∈ B ∧ z + q ∈ C) ∨ (z ∈ B ∧ z + p ∈ C ∧ z + q ∈ A) ∨
      (z ∈ C ∧ z + p ∈ A ∧ z + q ∈ B)) ↔ (p + q) % 3 = 0 :=
sorry

end partition_exists_iff_divisible_by_3_l665_665297


namespace cube_volume_from_surface_area_l665_665914

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l665_665914


namespace roots_of_p_l665_665250

def p (x : ℂ) : ℂ := x^3 + x^2 - 4 * x - 2

theorem roots_of_p : {z : ℂ | p z = 0 } = {1, -1 + complex.i, -1 - complex.i} :=
by 
  sorry

end roots_of_p_l665_665250


namespace cube_volume_from_surface_area_example_cube_volume_l665_665962

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l665_665962


namespace max_value_of_m_l665_665273

theorem max_value_of_m (x m : ℝ) (h1 : x^2 - 4*x - 5 > 0) (h2 : x^2 - 2*x + 1 - m^2 > 0) (hm : m > 0) 
(hsuff : ∀ (x : ℝ), (x < -1 ∨ x > 5) → (x > m + 1 ∨ x < 1 - m)) : m ≤ 2 :=
sorry

end max_value_of_m_l665_665273


namespace infinite_solutions_exist_l665_665417

variable {p q : ℕ} -- Natural numbers
variable {x y : ℕ} -- Positive integers

-- Define the condition that pq is not a perfect square
def not_perfect_square (p q : ℕ) : Prop :=
  ∀ n : ℕ, n * n ≠ p * q

theorem infinite_solutions_exist (h1 : p > 0) (h2 : q > 0) 
  (h3 : ∃ x y : ℕ, p * x * x + q * y * y = 1) 
  (h4 : not_perfect_square p q) :
  ∃ f : ℕ → ℕ × ℕ,
    ∀ n : ℕ, (let (x, y) := f n in p * x * x + q * y * y = 1) := 
sorry

end infinite_solutions_exist_l665_665417


namespace determinant_cos_tan_l665_665759

theorem determinant_cos_tan {A B C : ℝ} (h : A + B + C = π) :
  Matrix.det !![!![cos A ^ 2, tan A, 1],
                !![cos B ^ 2, tan B, 1],
                !![cos C ^ 2, tan C, 1]] = 0 :=
by sorry

end determinant_cos_tan_l665_665759


namespace round_wins_probability_l665_665345

theorem round_wins_probability : 
  let p_A := (1:ℚ)/2
  let p_C := (1:ℚ)/6
  let p_M := 2 * p_C
  let total_rounds := 7
  let prob_A := p_A^4
  let prob_M := p_M^2
  let prob_C := p_C
  let specific_seq_prob := prob_A * prob_M * prob_C
  let arrangements := (nat.factorial total_rounds) / ((nat.factorial 4) * (nat.factorial 2) * (nat.factorial 1))
  (specific_seq_prob * arrangements = 35 / 288) :=
sorry

end round_wins_probability_l665_665345


namespace product_of_x_values_l665_665324

noncomputable def find_product_of_x : ℚ :=
  let x1 := -20
  let x2 := -20 / 7
  (x1 * x2)

theorem product_of_x_values :
  (∃ x : ℚ, abs (20 / x + 4) = 3) ->
  find_product_of_x = 400 / 7 :=
by
  sorry

end product_of_x_values_l665_665324


namespace leak_empties_tank_in_30_hours_l665_665781

def A : ℝ := 1 / 10
def L_effective : ℝ := 1 / 15
def leak_rate : ℝ := A - L_effective
def leak_time : ℝ := 1 / leak_rate

theorem leak_empties_tank_in_30_hours :
  leak_time = 30 := by
  sorry

end leak_empties_tank_in_30_hours_l665_665781


namespace polynomial_roots_l665_665238

theorem polynomial_roots :
  (∀ x, x^3 - 3*x^2 - x + 3 = 0 ↔ (x = 1 ∨ x = -1 ∨ x = 3)) :=
by
  intro x
  split
  {
    intro h
    have h1 : x = 1 ∨ x = -1 ∨ x = 3
    {
      sorry
    }
    exact h1
  }
  {
    intro h
    cases h
    {
      rw h
      simp
    }
    {
      cases h
      {
        rw h
        simp
      }
      {
        rw h
        simp
      }
    }
  }

end polynomial_roots_l665_665238


namespace cube_volume_from_surface_area_l665_665872

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l665_665872


namespace no_such_complex_number_l665_665260

noncomputable def complex_numbers_with_condition :=
  {z : ℂ // ‖z‖ = 1 ∧ ‖(z^3 / (conj z)^3) + ((conj z)^3 / z^3)‖ = 3}

theorem no_such_complex_number : complex_numbers_with_condition = ∅ :=
by {
  sorry
}

end no_such_complex_number_l665_665260


namespace Maxwell_age_l665_665343

theorem Maxwell_age :
  ∀ (sister_age maxwell_age : ℕ),
    (sister_age = 2) → 
    (maxwell_age + 2 = 2 * (sister_age + 2)) →
    (maxwell_age = 6) :=
by
  intros sister_age maxwell_age h1 h2
  -- Definitions and hypotheses come directly from conditions
  sorry

end Maxwell_age_l665_665343


namespace colored_segments_sum_bound_l665_665413

theorem colored_segments_sum_bound (segments : set (set ℝ)) :
  (∃ (seg : set (set ℝ)), segments = seg ∧ ∀ s ∈ seg, s ⊆ [0, 1] ∧ disjoint s seg) ∧
  (∀ (x y : ℝ), x ∈ ⋃₀ segments → y ∈ ⋃₀ segments → x ≠ y → |x - y| ≠ 0.1) →
  (∑' s in segments, measure s) ≤ 0.5 :=
sorry

end colored_segments_sum_bound_l665_665413


namespace john_has_48_l665_665163

variable (Ali Nada John : ℕ)

theorem john_has_48 
  (h1 : Ali + Nada + John = 67)
  (h2 : Ali = Nada - 5)
  (h3 : John = 4 * Nada) : 
  John = 48 := 
by 
  sorry

end john_has_48_l665_665163


namespace total_buttons_l665_665541

theorem total_buttons (green buttons: ℕ) (yellow buttons: ℕ) (blue buttons: ℕ) (total buttons: ℕ) 
(h1: green = 90) (h2: yellow = green + 10) (h3: blue = green - 5) : total = green + yellow + blue → total = 275 :=
by
  sorry

end total_buttons_l665_665541


namespace intersection_of_sets_l665_665641

variable (M : Set ℤ) (N : Set ℤ)

theorem intersection_of_sets :
  M = {-2, -1, 0, 1, 2} →
  N = {x | x ≥ 3 ∨ x ≤ -2} →
  M ∩ N = {-2} :=
by
  intros hM hN
  sorry

end intersection_of_sets_l665_665641


namespace cube_volume_l665_665911

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l665_665911


namespace perfect_square_trinomial_l665_665624

theorem perfect_square_trinomial (m : ℝ) :
  (∃ a b : ℝ, (x : ℝ) → (x^2 + 2 * (m - 1) * x + 16) = (a * x + b)^2) → (m = 5 ∨ m = -3) :=
by
  sorry

end perfect_square_trinomial_l665_665624


namespace integral_one_interval_l665_665986

noncomputable def integral_of_one_eq_one : Prop :=
  ∫ x in set.Icc 0 1, (1:ℝ) = 1

theorem integral_one_interval : integral_of_one_eq_one :=
  by sorry

end integral_one_interval_l665_665986


namespace cube_volume_is_1728_l665_665938

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l665_665938


namespace probability_of_one_of_A_and_B_is_2_over_3_l665_665615

-- Defining the set of all people
def people : List String := ["A", "B", "C", "D"]

-- Defining the set of all possible pairs of selections
def pairs : List (String × String) := [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]

-- Defining the set of pairs where exactly one of A and B is selected
def pairs_with_one_of_A_and_B : List (String × String) := [("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")]

-- The probability can be calculated as the ratio of the favorable outcomes to the total outcomes
def probability_one_of_A_and_B : ℚ := (pairs_with_one_of_A_and_B.length.toRat / pairs.length.toRat)

-- Proving that the probability of selecting exactly one of A and B is 2/3
theorem probability_of_one_of_A_and_B_is_2_over_3 : 
  probability_one_of_A_and_B = 2 / 3 :=
by
  sorry  -- Proof is not provided, but the statement is correct

end probability_of_one_of_A_and_B_is_2_over_3_l665_665615


namespace handshakes_among_ten_men_l665_665434

/--
 Define a function that returns the number of handshakes that occur among ten men, 
 each deciding to shake hands only with men lighter than himself.
-/
def number_of_handshakes : ℕ :=
0 

theorem handshakes_among_ten_men :
  -- Declaring conditions for the proof
  (∀ i j : ℕ, i > j → i ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ∧ j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → i shakes_hands_with j = false) 
  ∧
  -- Defining the set of all men
  ∀ m : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 
  number_of_handshakes = 0 :=
by {
  sorry
}

end handshakes_among_ten_men_l665_665434


namespace sqrt_meaningful_iff_l665_665337

theorem sqrt_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = sqrt (2 * x - 4)) ↔ x ≥ 2 :=
by 
  sorry

end sqrt_meaningful_iff_l665_665337


namespace polygon_interior_angles_eq_360_l665_665266

theorem polygon_interior_angles_eq_360 (n : ℕ) (h : (n - 2) * 180 = 360) : n = 4 :=
sorry

end polygon_interior_angles_eq_360_l665_665266


namespace gcd_153_68_eq_17_l665_665202

theorem gcd_153_68_eq_17 : Int.gcd 153 68 = 17 :=
by
  sorry

end gcd_153_68_eq_17_l665_665202


namespace symmetric_point_A_is_B_l665_665804

/-
  Define the symmetric point function for reflecting a point across the origin.
  Define the coordinate of point A.
  Assert that the symmetric point of A has coordinates (-2, 6).
-/

structure Point where
  x : ℤ
  y : ℤ

def symmetric_point (p : Point) : Point :=
  Point.mk (-p.x) (-p.y)

def A : Point := ⟨2, -6⟩

def B : Point := ⟨-2, 6⟩

theorem symmetric_point_A_is_B : symmetric_point A = B := by
  sorry

end symmetric_point_A_is_B_l665_665804


namespace polynomial_roots_l665_665245

theorem polynomial_roots :
  (∀ x, x^3 - 3 * x^2 - x + 3 = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) := 
by
  sorry

end polynomial_roots_l665_665245


namespace no_real_values_of_y_satisfy_series_condition_l665_665261

noncomputable def series (y : ℝ) := ∑' n : ℕ, (if even n then 2 else -1) * y^n

theorem no_real_values_of_y_satisfy_series_condition (y : ℝ) (h : |y| < 1) :
  y = series y → false :=
by
  sorry

end no_real_values_of_y_satisfy_series_condition_l665_665261


namespace remainder_T_mod_2027_l665_665379

noncomputable def T : ℕ := ∑ j in Finset.range 73, Nat.choose 2024 j

theorem remainder_T_mod_2027 :
  let p := 2027 in
  Nat.Prime p →
  T % p = 1369 :=
by
  intro p hp
  have hp_prime : Nat.Prime p := hp
  -- The necessary calculations and argument steps would go here.
  sorry

end remainder_T_mod_2027_l665_665379


namespace cube_volume_from_surface_area_l665_665954

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l665_665954


namespace octagon_area_l665_665841

-- Define the condition where x = π / 6
def x : ℝ := Real.pi / 6

-- Define the condition where the total arc length sums to 2π
def total_arc_length : ℝ := 4 * x + 4 * (2 * x)

-- Define the area of the octagon
def area_of_octagon (R : ℝ) : ℝ := R^2 * (Real.sqrt 3 + 1)

-- The Lean 4 statement to prove is:
theorem octagon_area (R : ℝ) (hx : x = Real.pi / 6) 
    (h_total : total_arc_length = 2 * Real.pi) : 
    area_of_octagon R = R^2 * (Real.sqrt 3 + 1) := by sorry

end octagon_area_l665_665841


namespace option_d_functions_equal_l665_665988

theorem option_d_functions_equal (x : ℝ) (k : ℤ) (hk : x ≠ (k * π + π / 2)) :
  (tan x) = (sin x / cos x) :=
sorry

end option_d_functions_equal_l665_665988


namespace smallest_number_bounds_l665_665051

theorem smallest_number_bounds (a : ℕ → ℝ) (h_sum : (∑ i in finset.range 8, a i) = 4 / 3)
    (h_pos_sum : ∀ i : fin 8, 0 < ∑ j in (finset.univ \ {i}), a j) :
  -8 < a 0 ∧ a 0 ≤ 1 / 6 :=
by
  sorry

end smallest_number_bounds_l665_665051


namespace sqrt_floor_squared_eq_25_l665_665214

theorem sqrt_floor_squared_eq_25 :
  (⌊Real.sqrt 30⌋ : ℕ) = 5 →
  ⌊Real.sqrt 30⌋ ^ 2 = 25 :=
by
  sorry

end sqrt_floor_squared_eq_25_l665_665214


namespace depth_of_well_l665_665148

-- Define the variables and conditions
def position (t : ℝ) : ℝ := 20 * t^2

theorem depth_of_well (d : ℝ) (t1 t2 : ℝ) (h1 : position t1 = d) (h2 : t1 + t2 = 10.2) (h3 : t2 = d / 1050) : d = 1360 :=
by
  have h_t1 : t1 = Real.sqrt(d / 20) := by sorry
  have h_t2 : t2 = d / 1050 := h3
  have h_combined : Real.sqrt(d / 20) + d / 1050 = 10.2 := by sorry
  have h_solution : d = 1360 := by sorry
  exact h_solution

end depth_of_well_l665_665148


namespace smallest_value_bounds_l665_665053

variable {a : Fin 8 → ℝ}

theorem smallest_value_bounds
  (h1 : (∑ i, a i) = 4 / 3)
  (h2 : ∀ j, (∑ i, if i = j then 0 else a i) > 0) :
  ∃ a1, -8 < a1 ∧ a1 ≤ 1 / 6 :=
begin
  let a1 := a 0,
  use a1,
  split,
  { sorry },
  { sorry }
end

end smallest_value_bounds_l665_665053


namespace geom_seq_general_formula_sum_first_terms_b_seq_l665_665395

def geom_seq (n : ℕ) := {a : ℕ // ∀ i, a (i+1) = 2 * a i ∧ a 0 = 1 }

variables (S_n S_2n : ℕ → ℕ)
variables (n : ℕ)
  
theorem geom_seq_general_formula
    (a : ℕ → ℕ) 
    (h1 : a 1 * a 2 * a 3 = 8)
    (h2 : ∀ n, S_2n n = 3 * (a 1 + a 3 + a 5 + ... + a (2*n-1))) :
    ∀ n, a n = 2^(n-1) :=
sorry

theorem sum_first_terms_b_seq
    (a S_n : ℕ → ℕ)
    (b : ℕ → ℕ := λ n, n * S_n n)
    (h1 : a 1 * a 2 * a 3 = 8)
    (h2 : ∀ n, S_2n n = 3 * (a 1 + a 3 + a 5 + ... + a (2*n-1)))
    (h3 : S_n n = 2^n - 1) :
  ∀ n, ∑ i in range n, b i = (n-1) * 2^(n+1) + 2 - (n * (n+1) / 2) :=
sorry

end geom_seq_general_formula_sum_first_terms_b_seq_l665_665395


namespace find_f_7_l665_665752

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 4) = f x
axiom piecewise_function (x : ℝ) (h1 : 0 < x) (h2 : x < 2) : f x = 2 * x^3

theorem find_f_7 : f 7 = -2 := by
  sorry

end find_f_7_l665_665752


namespace sqrt_expression_meaningful_iff_l665_665335

theorem sqrt_expression_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = sqrt (2 * x - 4)) ↔ x ≥ 2 :=
by sorry

end sqrt_expression_meaningful_iff_l665_665335


namespace tea_mixture_ratio_l665_665729

theorem tea_mixture_ratio (x y : ℝ) (hx64 : x * 64) (hy74 : y * 74) (hmix : 69 * (x + y)) :
  64 * x + 74 * y = 69 * (x + y) → x = y :=
by
  intro h
  have h1 : 64 * x + 74 * y = 69 * x + 69 * y, from calc
    64 * x + 74 * y = 69 * (x + y) : h
        ... = 69 * x + 69 * y   : by ring
  have h2 : 64 * x + 74 * y = 69 * x + 69 * y, from h1
  have h3 : 74 * y - 69 * y = 69 * x - 64 * x, from eq_sub_of_add_eq h2
  have h4 : 5 * y = 5 * x, from calc
    74 * y - 69 * y = 69 * x - 64 * x : h3
    5 * y = 5 * x                     : by ring
  exact eq_of_mul_eq_mul_left (by norm_num) h4

end tea_mixture_ratio_l665_665729


namespace geometric_sequence_general_term_l665_665653

variable {n : ℕ}
variable {a : ℝ}
variable {a_n : ℕ → ℝ}

/-- Given that the sum of the first n terms of a geometric sequence {a_n} is Sn = 2^n + a,
    prove that the general term a_n = 2^(n-1). -/
theorem geometric_sequence_general_term (h : ∀ n, (2:ℝ)^n + a = ∑ i in finset.range n.succ, a_n i) : 
  a_n n = (2:ℝ)^(n - 1) := 
sorry

end geometric_sequence_general_term_l665_665653


namespace find_term_and_difference_l665_665100

theorem find_term_and_difference :
  ∃ n : ℕ, 15 * n ≤ 2016 ∧ 2016 < 15 * (n + 1) ∧ (15 * (n + 1) - 2016) = 9 := 
by {
  use 134,
  split,
  { linarith, },
  split,
  { norm_num1, },
  { norm_num1, },
  sorry
}

end find_term_and_difference_l665_665100


namespace tan_sum_identity_l665_665110

theorem tan_sum_identity : (1 + Real.tan (Real.pi / 180)) * (1 + Real.tan (44 * Real.pi / 180)) = 2 := 
by sorry

end tan_sum_identity_l665_665110


namespace longest_intersecting_segment_exists_l665_665627

open Convex Polygon

-- Define the convex polygon K
variable (K : Polygon)
variable [hK : Convex K]

-- Define the point P inside K
variable (P : Point)
variable (hP : P ∈ K)

-- Theorem: There exists a direction such that the line passing through P intersects the longest segment in K.
theorem longest_intersecting_segment_exists 
  (K : Polygon) [hK : Convex K] (P : Point) (hP : P ∈ K) :
  ∃ d : Direction, ∃ l : Line, l ∈ parallel_lines_through(d, P) ∧ intersect_longest_segment(l, K) :=
sorry

end longest_intersecting_segment_exists_l665_665627


namespace y_gets_0_l665_665150

noncomputable def amount_y_gets_per_rupee_x_gets (x y z a : ℝ) : Prop :=
   (x + y + z = 117) ∧ (y = a * x) ∧ (z = 0.5 * x) ∧ (y = 27) → (a = 0.45)

theorem y_gets_0.45_per_rupee_x (x y z a : ℝ) :
  amount_y_gets_per_rupee_x_gets x y z a :=
by sorry

end y_gets_0_l665_665150


namespace probability_ball_in_cube_l665_665145

theorem probability_ball_in_cube :
  let bounds := (set.Icc (-2 : ℝ) 2) ×ˢ (set.Icc (-2 : ℝ) 2) ×ˢ (set.Icc (-2 : ℝ) 2),
      sphere := λ (x y z : ℝ), x^2 + y^2 + z^2 ≤ 4,
      cube_volume := 64,
      sphere_volume := (32 * real.pi) / 3
  in (volume {p ∈ bounds | sphere p.1 p.2.1 p.2.2}) / cube_volume = real.pi / 6 := sorry

end probability_ball_in_cube_l665_665145


namespace cube_volume_l665_665936

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l665_665936


namespace cube_volume_from_surface_area_example_cube_volume_l665_665971

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l665_665971


namespace triangle_right_angle_locations_l665_665783

open Real

/-- There exist exactly 8 locations for the point C on the coordinate plane such that 
    triangle ABC is a right triangle with area 15 square units and AB = 10. -/
theorem triangle_right_angle_locations :
  ∃ (A B C : ℝ × ℝ), 
    dist A B = 10 ∧
    (∃ (r : ℝ) (s : ℝ), (C = (r, s) ∧ (let area := (1 / 2) * abs ((fst B - fst A) * (snd C - snd A) - (fst C - fst A) * (snd B - snd A)) in 
      area = 15 ∧ 
      (dist A C = sqrt ((fst C - fst A)^2 + (snd C -snd A)^2) ∧ dist B C = sqrt ((fst C - fst B)^2 + (snd C - snd B)^2)))) ∧ 
      (r = -5 ∨ r = 5 ∨ r = (√34) ∨ r = -(√34)) ∧ (s = 3 ∨ s = -3))
    in 8 := 
sorry

end triangle_right_angle_locations_l665_665783


namespace soccer_students_l665_665719

variable (T : ℕ) (B : ℕ) (girls_not_playing_soccer : ℕ)
variable (percent_boys_playing_soccer : ℝ)

def total_students := T
def number_boys := B
def percent_boys := 0.86
def number_girls_not_playing := girls_not_playing_soccer
def total_students_playing_soccer (S : ℕ) :=
  S = 35 / 0.14

theorem soccer_students (T B : ℕ) (girls_not_playing_soccer : ℕ) (percent_boys_playing_soccer : ℝ) :
  T = 420 → B = 320 → girls_not_playing_soccer = 65 → percent_boys_playing_soccer = 0.86 →
  ∃ S : ℕ, number_girl_students_playing_soccer = 100 - 65 
    ∧ 0.14 * S = 35 ∧ S = 250 := 
by {
  sorry
}

end soccer_students_l665_665719


namespace Crups_Arogs_Brafs_l665_665704

variables (Arogs Crups Brafs Dramps : Type)

def Are_Subset (X Y: Type) : Prop := ∀ x : X, ∃ y : Y, x = y

-- Given conditions
axiom Arogs_subset_Brafs: Are_Subset Arogs Brafs
axiom Crups_subset_Brafs: Are_Subset Crups Brafs
axiom Arogs_subset_Dramps: Are_Subset Arogs Dramps
axiom Crups_subset_Dramps: Are_Subset Crups Dramps

-- Goal to prove
theorem Crups_Arogs_Brafs
  (Arogs_subset_Brafs: Are_Subset Arogs Brafs)
  (Crups_subset_Brafs: Are_Subset Crups Brafs)
  (Arogs_subset_Dramps: Are_Subset Arogs Dramps)
  (Crups_subset_Dramps: Are_Subset Crups Dramps):
   Are_Subset Crups Arogs ∧ Are_Subset Crups Brafs :=
sorry

end Crups_Arogs_Brafs_l665_665704


namespace work_ratio_l665_665104

theorem work_ratio 
  (m b : ℝ) 
  (h : 7 * m + 2 * b = 6 * (m + b)) : 
  m / b = 4 := 
sorry

end work_ratio_l665_665104


namespace cube_volume_l665_665929

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l665_665929


namespace inclination_angle_of_line_l665_665817

noncomputable def inclination_angle : Real :=
  let line_eq : Real → Real → Prop := λ x y, sqrt 3 * x + y - 1 = 0
  let slope : Real := -sqrt 3
  let theta : Real := arctan slope + if slope < 0 then π else 0
  abs θ

theorem inclination_angle_of_line : inclination_angle = 2 * π / 3 := 
  sorry

end inclination_angle_of_line_l665_665817


namespace cube_volume_l665_665904

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l665_665904


namespace cube_volume_l665_665898

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l665_665898


namespace john_savings_remaining_l665_665372

theorem john_savings_remaining :
  let saved_base8 : ℕ := 5 * 8^3 + 5 * 8^2 + 5 * 8^1 + 5 * 8^0
  let ticket_cost : ℕ := 1200
  (saved_base8 - ticket_cost) = 1725 :=
by
  let saved_base8 := 5 * 8^3 + 5 * 8^2 + 5 * 8^1 + 5 * 8^0
  let ticket_cost := 1200
  show (saved_base8 - ticket_cost) = 1725 from sorry

end john_savings_remaining_l665_665372


namespace sum_of_reversed_integers_l665_665101

-- Definitions of properties and conditions
def reverse_digits (m n : ℕ) : Prop :=
  let to_digits (x : ℕ) : List ℕ := x.digits 10
  to_digits m = (to_digits n).reverse

-- The main theorem statement
theorem sum_of_reversed_integers
  (m n : ℕ)
  (h_rev: reverse_digits m n)
  (h_prod: m * n = 1446921630) :
  m + n = 79497 :=
sorry

end sum_of_reversed_integers_l665_665101


namespace cube_volume_from_surface_area_l665_665882

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l665_665882


namespace third_consecutive_even_integer_l665_665460

theorem third_consecutive_even_integer (n : ℤ) (h : (n + 2) + (n + 6) = 156) : (n + 4) = 78 :=
sorry

end third_consecutive_even_integer_l665_665460


namespace problem_statement_l665_665754

variable (n : ℕ) (X : Set ℕ)
variable (x y z w : ℕ)
variable (S : Set (ℕ × ℕ × ℕ))

-- Define the set X
def X (n : ℕ) : Set ℕ := {i | i ∈ finset.range (n+1)}

-- Define the set S
def S (X : Set ℕ) : Set (ℕ × ℕ × ℕ) := 
  {t | t.1 < t.2 ∧ t.2 < t.3 ∨ t.2 < t.3 ∧ t.3 < t.1 ∨ t.3 < t.1 ∧ t.1 < t.2}

-- Our main theorem statement
theorem problem_statement
  (n_ge_4 : n ≥ 4)
  (x_in_X : x ∈ X n) (y_in_X : y ∈ X n) (z_in_X : z ∈ X n) (w_in_X : w ∈ X n)
  (xyz_in_S : (x, y, z) ∈ S (X n))
  (zwx_in_S : (z, w, x) ∈ S (X n)) :
  ((y, z, w) ∈ S (X n)) ∧ ((x, y, w) ∈ S (X n)) := 
sorry

end problem_statement_l665_665754


namespace monotonic_decreasing_interval_of_f_l665_665038

noncomputable def f (x : ℝ) : ℝ := Real.logb 3 (3 * x^2 - 2 * x - 1)

theorem monotonic_decreasing_interval_of_f : 
  ∀ x, f x = Real.logb 3 (3 * x^2 - 2 * x - 1) → 
         (-∞ < x ∧ x < -1/3) ∨ (1 < x ∧ x < ∞) → 
         (∀ a b, a < b → f a ≥ f b) :=
by
  sorry

end monotonic_decreasing_interval_of_f_l665_665038


namespace perpendicular_BE_CF_l665_665361

noncomputable def right_triangle {α : Type*} [linear_ordered_field α] 
  (A B C D E F : α) :=
  ∃ (right_triangle : α × α × α × α × α × α), 
    let ⟨A, B, C, D, E, F⟩ := right_triangle in
    ∃ (hypotenuse : α), 
      A ≠ C ∧ B ≠ C ∧ A + B = hypotenuse
    ∧ ∃ (CD : α), 
      D = C * CD / hypotenuse
    ∧ point E ∈ segment [CD]
    ∧ point F ∈ segment [DA]
    ∧ CE / CD = AF / AD.

theorem perpendicular_BE_CF 
  {A B C D E F : Type*} [linear_ordered_field α] 
  (tr : right_triangle A B C D E F) :
  BE ⟂ CF := 
sorry

end perpendicular_BE_CF_l665_665361


namespace intersection_medians_circle_l665_665778

variables (A B C D K M O: Type)
variables [rectangle A B C D] [circumscribed K A B C D]
variables (h1 : ∃ (M : Segment AD), AM / MD = 2) (O : center A B C D)

theorem intersection_medians_circle 
: (∃ S, is_intersection_point_of_medians S (OKD) ∧ S ∈ CircumscribedCircle (COD)) :=
by sorry

end intersection_medians_circle_l665_665778


namespace jameson_badminton_medals_l665_665366

theorem jameson_badminton_medals:
  ∀ (total track: ℕ) (swimming_multiple: ℕ),
  total = 20 → track = 5 → swimming_multiple = 2 →
  ∃ (badminton: ℕ), badminton = 20 - (track + swimming_multiple * track) ∧ badminton = 5 :=
by
  intros total track swimming_multiple ht ht5 hsm
  use 5
  simp [ht5, hsm, ht]
  sorry

end jameson_badminton_medals_l665_665366


namespace points_A_B_D_are_collinear_l665_665313

noncomputable def vectors_are_collinear (a b : Vector ℝ 3) 
    (AB BC CD : Vector ℝ 3) 
    (h1 : AB = a + 2 • b) 
    (h2 : BC = -5 • a + 6 • b) 
    (h3 : CD = 7 • a - 2 • b) : Prop := 
  let BD := BC + CD in
  ∃ k : ℝ, BD = k • AB

theorem points_A_B_D_are_collinear (a b : Vector ℝ 3)
    (AB BC CD : Vector ℝ 3) 
    (h1 : AB = a + 2 • b) 
    (h2 : BC = -5 • a + 6 • b)
    (h3 : CD = 7 • a - 2 • b) : vectors_are_collinear a b AB BC CD h1 h2 h3 :=
  sorry

end points_A_B_D_are_collinear_l665_665313


namespace cube_volume_from_surface_area_l665_665961

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l665_665961


namespace vector_parallel_x_l665_665673

theorem vector_parallel_x : 
  ∀ (x : ℝ), let a := (4 : ℝ, -2 : ℝ), b := (x, 5 : ℝ) in 
  (a.1 * b.2 + a.2 * b.1 = 0) → x = -10 :=
begin
  intros x a b h,
  sorry
end

end vector_parallel_x_l665_665673


namespace sum_of_digits_10_pow_50_minus_75_l665_665584

theorem sum_of_digits_10_pow_50_minus_75 : 
  ∑ d in (10^50 - 75).digits, d = 439 :=
sorry

end sum_of_digits_10_pow_50_minus_75_l665_665584


namespace tan_sum_l665_665647

theorem tan_sum (alpha beta : ℝ) (h1 : sin alpha + sin beta = 1/4) (h2 : cos alpha + cos beta = 1/3) : 
  tan (alpha + beta) = 24 / 7 :=
by
  sorry

end tan_sum_l665_665647


namespace burpees_percentage_contribution_l665_665213

theorem burpees_percentage_contribution :
  let total_time : ℝ := 20
  let jumping_jacks : ℝ := 30
  let pushups : ℝ := 22
  let situps : ℝ := 45
  let burpees : ℝ := 15
  let lunges : ℝ := 25

  let jumping_jacks_rate := jumping_jacks / total_time
  let pushups_rate := pushups / total_time
  let situps_rate := situps / total_time
  let burpees_rate := burpees / total_time
  let lunges_rate := lunges / total_time

  let total_rate := jumping_jacks_rate + pushups_rate + situps_rate + burpees_rate + lunges_rate

  (burpees_rate / total_rate) * 100 = 10.95 :=
by
  sorry

end burpees_percentage_contribution_l665_665213


namespace regression_line_and_estimate_l665_665185

theorem regression_line_and_estimate :
  let x_bar := 4
  let y_bar := 5.4
  let eq1 := 5.4 = 4 * b + a
  let eq2 := 8 * b + a - (7 * b + a) = 1.1
  (∃ b a, eq1 ∧ eq2) →
  linear_regression_eq : (b = 0.55) ∧ (a = 3.2) →
  regression_line_eq : ∀ x,  x * 0.55 + 3.2 =
  (service_life_10_eq : (10 * 0.55 + 3.2) = 8.7 :=
sorry

end regression_line_and_estimate_l665_665185


namespace find_b_l665_665390

def vec3 := ℝ × ℝ × ℝ

def a : vec3 := (3, 1, 4)

def dot_product (v1 v2 : vec3) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def cross_product (v1 v2 : vec3) : vec3 :=
  (v1.2 * v2.3 - v1.3 * v2.2, v1.3 * v2.1 - v1.1 * v2.3, v1.1 * v2.2 - v1.2 * v2.1)

def b : vec3 := (-1.346, 3.144, 2.577)  -- These coordinates are rounded to three decimal places

theorem find_b :
  (dot_product a b = 20) ∧
  (cross_product a b = (-10, 1, 2)) :=
by
  -- Assuming the correct values of b were found through prior computation
  sorry

end find_b_l665_665390


namespace point_P_lower_left_l665_665539

variable (a b : ℕ)
variable (P1 P2 : ℚ)

-- Conditions
def isParallel (a b : ℕ) : Prop := (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 4) ∨ (a = 3 ∧ b = 6)
def P1 (a b : ℕ) : ℚ := if isParallel a b then 1 / 18 else 0
def P2 (a b : ℕ) : ℚ := if isParallel a b then 0 else 11 / 12

-- Question and correct answer
theorem point_P_lower_left (a b : ℕ) : (P1 a b, P2 a b) ∈ {p : ℚ × ℚ | p.1 + 2 * p.2 < 2} :=
by
  sorry

end point_P_lower_left_l665_665539


namespace simplify_expression_l665_665791

variable (x y : ℤ) -- Assume x and y are integers for simplicity

theorem simplify_expression : (5 - 2 * x) - (8 - 6 * x + 3 * y) = -3 + 4 * x - 3 * y := by
  sorry

end simplify_expression_l665_665791


namespace range_of_m_and_n_l665_665639

theorem range_of_m_and_n (m n : ℝ) :
  (∃ (x y : ℝ), x = 2 ∧ y = 3 ∧ 2 * x - y + m > 0 ∧ ¬(x + y - n ≤ 0)) →
  m > -1 ∧ n < 5 :=
by
  intro h
  rcases h with ⟨x, y, hx, hy, hA, hB⟩
  rw [hx, hy, mul_comm] at hA hB
  -- Simplify the conditions into the final inequalities
  have h1 : m > -1,
  { linarith }
  have h2 : n < 5,
  { linarith }
  exact ⟨h1, h2⟩

end range_of_m_and_n_l665_665639


namespace negation_of_forall_pos_l665_665822

open Real

theorem negation_of_forall_pos (h : ∀ x : ℝ, x^2 - x + 1 > 0) : 
  ¬(∀ x : ℝ, x^2 - x + 1 > 0) ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 :=
by
  sorry

end negation_of_forall_pos_l665_665822


namespace hyperbola_dot_product_l665_665629

theorem hyperbola_dot_product (a : ℝ) (c : ℝ) (P F1 F2 : ℝ×ℝ) (e : ℝ) (x y : ℝ)
    (h_hyperbola : x^2 - (y^2) / 3 = 1)
    (h_f1 : F1 = (-2, 0))
    (h_f2 : F2 = (2, 0))
    (h_eccentricity : e = c / a)
    (h_sin_ratio : ∀ (P : ℝ × ℝ), (sin (angle P F2 F1) / sin (angle P F1 F2)) = e)
    (h_P_on_hyperbola : x^2 - (y^2) / 3 = 1)
    (h_a : a = 1)
    (h_c : c = 2)
    (h_distance_PF1 : dist P F1 = 4)
    (h_distance_PF2 : dist P F2 = 2)
    (h_angle_cos : cos (angle P F2 F1) = 1/4) :
  (F2P : ℝ := (dist P F2)) -> 
  (F2F1 : ℝ := dist F2 F1) -> 
  abs ((F2P * F2F1) * cos (angle P F2 F1)) = 2 :=
sorry

end hyperbola_dot_product_l665_665629


namespace max_y_coordinate_l665_665259

theorem max_y_coordinate (θ : ℝ) : 
  let r := cos (2 * θ), y := r * sin θ in
  y ≤ 2 / 3 := sorry

end max_y_coordinate_l665_665259


namespace volume_of_tangent_pyramid_l665_665358

theorem volume_of_tangent_pyramid
  (A B C D P Q K L : Point)
  (length_AB : dist A B = 4)
  (length_CD : dist C D = 4)
  (length_AC : dist A C = 3)
  (length_AD : dist A D = 3)
  (length_BC : dist B C = 3)
  (length_BD : dist B D = 3)
  (inscribed_sphere : ∃ S, touches S (mk_face A B C) ∧ touches S (mk_face A B D) ∧ touches S (mk_face B C D) ∧ touches S (mk_face A C D)) :
  volume (mk_pyramid P Q K L) = 8 / 375 := 
sorry

end volume_of_tangent_pyramid_l665_665358


namespace function_solution_l665_665733

def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x) + (0.5 + x) * f(1 - x) = 1

def correct_function (x : ℝ) : ℝ :=
  if x = 0.5 then 0.5 else 1 / (0.5 - x)

theorem function_solution : satisfies_equation correct_function :=
  sorry

end function_solution_l665_665733


namespace average_speed_correct_l665_665159

-- Define conditions
def first_leg_distance := 300
def second_leg_distance := 200
def third_leg_distance := 120
def first_leg_speed := 120
def second_leg_speed := 100
def third_leg_speed := 80
def layover_time := 2

-- Time calculations for each leg
def time_first_leg := first_leg_distance / first_leg_speed
def time_second_leg := second_leg_distance / second_leg_speed
def time_third_leg := third_leg_distance / third_leg_speed

-- Total distance and total flying time
def total_distance := first_leg_distance + second_leg_distance + third_leg_distance
def total_time_flying := time_first_leg + time_second_leg + time_third_leg

-- Average speed calculation
def average_speed := total_distance / total_time_flying

-- Theorem proving average speed for the total trip excluding layover
theorem average_speed_correct : average_speed = 103.33 := by
  sorry

end average_speed_correct_l665_665159


namespace real_solutions_of_quadratic_eq_l665_665591

def integer_part (x : ℝ) : ℤ :=
  int.floor x

def quadratic_equation (x : ℝ) : ℝ :=
  4 * x * x - 40 * (integer_part x) + 51

theorem real_solutions_of_quadratic_eq :
  {x : ℝ | quadratic_equation x = 0} =
  {x | x = real.sqrt 29 / 2 ∨ x = real.sqrt 189 / 2 ∨ x = real.sqrt 229 / 2 ∨ x = real.sqrt 269 / 2} :=
by
  -- sorry for the solution
  simp [quadratic_equation, integer_part]
  sorry -- proving the equation solutions

end real_solutions_of_quadratic_eq_l665_665591


namespace cube_volume_l665_665928

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l665_665928


namespace largest_k_l665_665612

-- Define the system of equations and conditions
def system_valid (x y k : ℝ) : Prop := 
  2 * x + y = k ∧ 
  3 * x + y = 3 ∧ 
  x - 2 * y ≥ 1

-- Define the proof problem as a theorem in Lean
theorem largest_k (x y : ℝ) :
  ∀ k : ℝ, system_valid x y k → k ≤ 2 := 
sorry

end largest_k_l665_665612


namespace mr_bird_speed_to_work_l665_665004

theorem mr_bird_speed_to_work (
  d t : ℝ
) (h1 : d = 45 * (t + 4 / 60)) 
  (h2 : d = 55 * (t - 2 / 60))
  (h3 : t = 29 / 60)
  (d_eq : d = 24.75) :
  (24.75 / (29 / 60)) = 51.207 := 
sorry

end mr_bird_speed_to_work_l665_665004


namespace cube_volume_from_surface_area_l665_665924

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l665_665924


namespace volume_of_pyramid_l665_665068

noncomputable section

structure Cone :=
(apex : Point)
(sl_height : ℝ)
(angle_axis : ℝ)

def pyramid_volume (cones : List Cone) (volume : ℝ) : Prop :=
  let α := cones[0].angle_axis
  let β := cones[2].angle_axis
  let l := cones[0].sl_height
  let AO1 := l * Real.cos α
  let AO2 := AO1
  let AO3 := l * Real.cos β
  let α_plus_β := α + β
  let O1O3 := l * Real.sin α_plus_β
  let O2O3 := O1O3
  let BC := 6 * Real.cos α * Math.sqrt (Real.cos (3 * π / 8)^2 - Real.cos (π / 8)^2) 
  volume = 9 * Math.sqrt (Real.sqrt 2 + 1)

theorem volume_of_pyramid :
  ∀ (cone1 cone2 cone3 : Cone) (V : ℝ), 
  cone1.angle_axis = π / 8 →
  cone2.angle_axis = π / 8 →
  cone3.angle_axis = π / 4 →
  cone1.sl_height = 6 →
  cone2.sl_height = 6 →
  cone3.sl_height = 6 →
  pyramid_volume [cone1, cone2, cone3] V :=
by
  intros
  simp [pyramid_volume]
  sorry

end volume_of_pyramid_l665_665068


namespace exists_odd_white_2x2_square_l665_665796

def is_black_or_white (color : bool) : Prop :=
  color = tt ∨ color = ff

def is_valid_color_diff (num_black num_white : ℕ) : Prop :=
  |(num_black : ℤ) - (num_white : ℤ)| = 404

theorem exists_odd_white_2x2_square :
  ∀ (grid : Fin 200 → Fin 200 → bool),
    (∀ r s, is_black_or_white (grid r s)) →
    (is_valid_color_diff
      (Finset.card (Finset.filter (λ (c : Fin 200 × Fin 200), grid c.fst c.snd = tt) Finset.univ))
      (Finset.card (Finset.filter (λ (c : Fin 200 × Fin 200), grid c.fst c.snd = ff) Finset.univ))) →
    ∃ (r : Fin 199) (s : Fin 199),
      ((grid r s) = ff ∧ (grid r (s + 1)) = ff ∧ (grid (r + 1) s) = ff ∧ (grid (r + 1) (s + 1)) = tt) ∨
      ((grid r s) = ff ∧ (grid r (s + 1)) = tt ∧ (grid (r + 1) s) = tt ∧ (grid (r + 1) (s + 1)) = ff) ∨
      ((grid r s) = tt ∧ (grid r (s + 1)) = ff ∧ (grid (r + 1) s) = ff ∧ (grid (r + 1) (s + 1)) = tt) ∨
      ((grid r s) = tt ∧ (grid r (s + 1)) = tt ∧ (grid (r + 1) s) = tt ∧ (grid (r + 1) (s + 1)) = ff)
by
  sorry

end exists_odd_white_2x2_square_l665_665796


namespace picture_edge_distance_l665_665143

theorem picture_edge_distance 
    (wall_width : ℕ) 
    (picture_width : ℕ) 
    (centered : Bool) 
    (h_w : wall_width = 22) 
    (h_p : picture_width = 4) 
    (h_c : centered = true) : 
    ∃ (distance : ℕ), distance = 9 := 
by
  sorry

end picture_edge_distance_l665_665143


namespace cube_volume_from_surface_area_example_cube_volume_l665_665966

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l665_665966


namespace directrix_of_parabola_l665_665256

theorem directrix_of_parabola (a : ℝ) (h : a = -4) : ∃ k : ℝ, k = 1/16 ∧ ∀ x : ℕ, y = ax ^ 2 → y = k := 
by 
  sorry

end directrix_of_parabola_l665_665256


namespace bernardo_prob_greater_silvia_l665_665563

open_locale classical
noncomputable theory

/--
Bernardo randomly picks 3 distinct numbers from the set {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, while Silvia randomly picks 3 distinct numbers from the set {1, 2, 3, 4, 5, 6, 7, 8, 9}. Both then arrange their chosen numbers in descending order to form a 3-digit number. Prove that the probability that Bernardo's number is greater than Silvia's number is 9/14.
-/
theorem bernardo_prob_greater_silvia :
  let S : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let T : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let bernardo_choices : finset (finset ℕ) := S.powerset.filter (λ s, s.card = 3)
  let silvia_choices : finset (finset ℕ) := T.powerset.filter (λ s, s.card = 3)
  let bernardo_3_digit : finset ℕ := bernardo_choices.image (λ s, (s.sort (≥)).foldl (λ acc n, acc * 10 + n) 0)
  let silvia_3_digit : finset ℕ := silvia_choices.image (λ s, (s.sort (≥)).foldl (λ acc n, acc * 10 + n) 0)
  let num_bigger := bernardo_3_digit.filter (λ b, ∀ s ∈ silvia_3_digit, b > s)
  (num_bigger.card : ℚ) / (bernardo_3_digit.card * silvia_3_digit.card) = 9 / 14 :=
by sorry

end bernardo_prob_greater_silvia_l665_665563


namespace cube_volume_l665_665903

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l665_665903


namespace woman_traveled_by_bus_l665_665156

noncomputable def travel_by_bus : ℕ :=
  let total_distance := 1800
  let distance_by_plane := total_distance / 4
  let distance_by_train := total_distance / 6
  let distance_by_taxi := total_distance / 8
  let remaining_distance := total_distance - (distance_by_plane + distance_by_train + distance_by_taxi)
  let distance_by_rental := remaining_distance * 2 / 3
  distance_by_rental / 2

theorem woman_traveled_by_bus :
  travel_by_bus = 275 :=
by 
  sorry

end woman_traveled_by_bus_l665_665156


namespace number_of_solutions_l665_665583

theorem number_of_solutions :
  ∃! x ∈ set.Icc (0 : ℝ) π, tan (π / 2 * sin x) = cot (π / 2 * cos x) :=
by
  sorry

end number_of_solutions_l665_665583


namespace r_minus_p_value_l665_665333

theorem r_minus_p_value (p q r : ℝ)
  (h₁ : (p + q) / 2 = 10)
  (h₂ : (q + r) / 2 = 22) :
  r - p = 24 :=
by
  sorry

end r_minus_p_value_l665_665333


namespace cube_volume_is_1728_l665_665939

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l665_665939


namespace purely_imaginary_z_range_of_m_l665_665299

-- Part 1: Given a purely imaginary complex number z such that (z + 2) / (1 - i) + z is real, prove z = -2/3 * i.
theorem purely_imaginary_z (z : ℂ) (h1 : z.im ≠ 0) (h2 : (z + 2) / (1 - complex.I) + z ∈ ℝ) : z = -(2/3) * complex.I := 
sorry

-- Part 2: If the point represented by (m - z)^2 is in the first quadrant, prove the range of m is (2/3, +∞)
theorem range_of_m (z : ℂ) (m : ℝ) (h1 : z = -(2/3) * complex.I) 
  (h2 : ((m : ℂ) - z)^2.re > 0) (h3 : ((m : ℂ) - z)^2.im > 0) : m > (2/3) := 
sorry

end purely_imaginary_z_range_of_m_l665_665299


namespace smallest_value_condition_l665_665045

theorem smallest_value_condition 
  (a : Fin 8 → ℝ)
  (h_sum : ∑ i, a i = 4 / 3)
  (h_pos_sum : ∀ i, 0 < ∑ j, if j == i then 0 else a j) :
  -8 < (Finset.min' Finset.univ (λ i, a i)) ∧ (Finset.min' Finset.univ (λ i, a i)) ≤ 1 / 6 :=
by
  sorry

end smallest_value_condition_l665_665045


namespace area_under_sin_curve_l665_665252

noncomputable def area_of_plane_figure : ℝ :=
  ∫ x in (- (π/2))..(5 * π / 4), real.abs (real.sin x)

theorem area_under_sin_curve :
  area_of_plane_figure = 4 - real.sqrt 2 / 2 :=
sorry

end area_under_sin_curve_l665_665252


namespace z_imaginary_iff_z_pure_imaginary_iff_z_not_second_quadrant_l665_665095

noncomputable def z (m : ℝ) : ℂ := ⟨m^2 - 5*m + 6, -3*m⟩

theorem z_imaginary_iff (m : ℝ) : Im (z m) ≠ 0 ↔ m ≠ 0 :=
sorry

theorem z_pure_imaginary_iff (m : ℝ) : Re (z m) = 0 ↔ (m = 2 ∨ m = 3) :=
sorry

theorem z_not_second_quadrant (m : ℝ) : (Re (z m) > 0 ∧ Im (z m) > 0) ↔ false :=
sorry

end z_imaginary_iff_z_pure_imaginary_iff_z_not_second_quadrant_l665_665095


namespace gambler_final_amount_l665_665132

-- Define initial amount of money
def initial_amount := 100

-- Define the multipliers
def win_multiplier := 4 / 3
def loss_multiplier := 2 / 3
def double_win_multiplier := 5 / 3

-- Define the gambler scenario (WWLWLWLW)
def scenario := [double_win_multiplier, win_multiplier, loss_multiplier, win_multiplier, loss_multiplier, win_multiplier, loss_multiplier, win_multiplier]

-- Function to compute final amount given initial amount, number of wins and losses, and the scenario
def final_amount (initial: ℚ) (multipliers: List ℚ) : ℚ :=
  multipliers.foldl (· * ·) initial

-- Prove that the final amount after all multipliers are applied is approximately equal to 312.12
theorem gambler_final_amount : abs (final_amount initial_amount scenario - 312.12) < 0.01 :=
by
  sorry

end gambler_final_amount_l665_665132


namespace units_digit_fraction_l665_665567

theorem units_digit_fraction (h1 : 30 = 2 * 3 * 5) (h2 : 31 = 31) (h3 : 32 = 2^5) 
    (h4 : 33 = 3 * 11) (h5 : 34 = 2 * 17) (h6 : 35 = 5 * 7) (h7 : 7200 = 2^4 * 3^2 * 5^2) :
    ((30 * 31 * 32 * 33 * 34 * 35) / 7200) % 10 = 2 :=
by
  sorry

end units_digit_fraction_l665_665567


namespace cube_volume_from_surface_area_l665_665868

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l665_665868


namespace tangent_line_circle_l665_665357

theorem tangent_line_circle {a : ℝ} (h₁ : a > 0) :
  let line_cartesian := ∀ x y : ℝ, x + (√3) * y + 1 = 0,
      circle_cartesian := ∀ x y : ℝ, (x - a)^2 + y^2 = a^2 in
  (∀ x y : ℝ, line_cartesian x y → circle_cartesian x y → real.abs (a + 1) / 2 = a) → 
  a = 1 :=
by
  sorry

end tangent_line_circle_l665_665357


namespace drinks_carton_volume_l665_665513

/-- Given a drinks carton formed by four congruent triangles with 
    sides QP = RS = 4 cm and PR = PS = QR = QS = 10 cm,
    the volume of the carton is 16/3 * (sqrt 23) cm^3. -/
theorem drinks_carton_volume :
  ∀ (QP RS PR PS QR QS : ℝ),
  QP = 4 ∧ RS = 4 ∧ PR = 10 ∧ PS = 10 ∧ QR = 10 ∧ QS = 10 →
  ∃ (V : ℝ), V = (16 / 3) * real.sqrt 23 :=
by
  intros QP RS PR PS QR QS h,
  sorry

end drinks_carton_volume_l665_665513


namespace angle_tangent_circle_l665_665766

-- Definitions and given conditions
variable {Ω₁ Ω₂ : Type} [circle Ω₁] [circle Ω₂]
variable {O₁ O₂ : point} 
variable {A P X Y Q R : point}
variable (h₀ : tangent_internally Ω₁ Ω₂ A)
variable (h₁ : on_circle P Ω₂)
variable (h₂ : tangent_point P X Ω₁ Ω₂ Q)
variable (h₃ : tangent_point P Y Ω₁ Ω₂ R)

-- The proof statement
theorem angle_tangent_circle (h₀ : tangent_internally Ω₁ Ω₂ A)
  (h₁ : on_circle P Ω₂) (h₂ : tangent_point P X Ω₁ Ω₂ Q)
  (h₃ : tangent_point P Y Ω₁ Ω₂ R) :
  angle Q A R = 2 * angle X A Y :=
sorry

end angle_tangent_circle_l665_665766


namespace diagonals_parallel_sides_l665_665079

-- Define the geometric context
variables {P Q R S A B C D A₁ B₁ C₁ D₁ : Type}

-- Define the conditions (conditions directly from a))
def sides (A B C D : Type) := -- Parallelism definitions can be added here 
  (is_parallel A B C D)  -- Placeholder definition for parallel sides of parallelograms

def inscribed_parallelograms (P Q R S A B C D A₁ B₁ C₁ D₁ : Type) :=
  (on_segment A PQ ∧ on_segment B QR ∧ on_segment C RS ∧ on_segment D SP) ∧
  (on_segment A₁ PQ ∧ on_segment B₁ QR ∧ on_segment C₁ RS ∧ on_segment D₁ SP) ∧
  (sides A B C D) ∧ (sides A₁ B₁ C₁ D₁)

-- Define the theorem to be proven (question and answer from c))
theorem diagonals_parallel_sides (h : inscribed_parallelograms P Q R S A B C D A₁ B₁ C₁ D₁) :
  (is_parallel_diagonal PQ RS A B C D) ∧ (is_parallel_diagonal PR QS A B C D):=
begin
  sorry -- Placeholder for the actual proof steps
end

end diagonals_parallel_sides_l665_665079


namespace cube_volume_l665_665908

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l665_665908


namespace diff_between_roots_is_correct_l665_665019

-- Given the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Coefficients for our specific problem
def a : ℝ := 5 + 3 * Real.sqrt 2
def b : ℝ := -(1 - Real.sqrt 2)
def c : ℝ := -1

-- Define the function for finding roots of quadratic equations
noncomputable def roots (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let root1 := (-b + Real.sqrt discriminant) / (2 * a)
  let root2 := (-b - Real.sqrt discriminant) / (2 * a)
  (root1, root2)

-- Define the difference between root1 and root2
noncomputable def diff_roots (a b c : ℝ) : ℝ :=
  let (root1, root2) := roots a b c
  root1 - root2

-- Define the goal
theorem diff_between_roots_is_correct :
  diff_roots a b c = 2 * Real.sqrt 2 + 1 :=
  sorry

end diff_between_roots_is_correct_l665_665019


namespace min_value_reciprocal_sum_l665_665287

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  (a = 1 ∧ b = 1) → (1 / a + 1 / b = 2) := by
  intros h
  sorry

end min_value_reciprocal_sum_l665_665287


namespace minimum_value_proof_minimum_value_achieved_l665_665288

noncomputable def minimum_value_M (a b : ℝ) := 
  sqrt (1 + 2 * a^2) + 2 * sqrt ( (5 / 12)^2 + b^2 )

theorem minimum_value_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 1) : 
  minimum_value_M a b ≥ (5 * sqrt 34) / 12 :=
begin
  sorry
end

-- Additional theorem to demonstrate achieving the minimum value
theorem minimum_value_achieved (a b : ℝ) (ha : a = 3 / 4) (hb : b = 1 / 4) : 
  minimum_value_M a b = (5 * sqrt 34) / 12 :=
begin
  sorry
end

end minimum_value_proof_minimum_value_achieved_l665_665288


namespace price_restoration_l665_665182

theorem price_restoration {P : ℝ} (hP : P > 0) :
  (P - 0.85 * P) / (0.85 * P) * 100 = 17.65 :=
by
  sorry

end price_restoration_l665_665182


namespace dot_product_AB_BC_l665_665342

theorem dot_product_AB_BC (AB BC : ℝ) (B : ℝ) 
  (h1 : AB = 3) (h2 : BC = 4) (h3 : B = π/6) :
  (AB * BC * Real.cos (π - B) = -6 * Real.sqrt 3) :=
by
  rw [h1, h2, h3]
  sorry

end dot_product_AB_BC_l665_665342


namespace odd_function_neg_x_value_l665_665815

theorem odd_function_neg_x_value (f : ℝ → ℝ) (hf_odd : ∀ x, f (-x) = -f x) 
    (hf_pos : ∀ x, 0 < x → f x = -x + 1) :
    ∀ x, x < 0 → f x = -x - 1 :=
by
  intros x hx
  have hx_pos : -x > 0 := neg_pos.mpr hx
  rw [← hf_pos (-x) hx_pos]
  rw [hf_odd x]
  sorry

end odd_function_neg_x_value_l665_665815


namespace collinearity_condition_l665_665747

/-- Define the quadrilateral and the intersection points -/
structure Quadrilateral :=
(A B C D E F : Point)
(half_lines_intersect : [AB] ∩ [DC] = E ∧ [BC] ∩ [AD] = F) -- Representation of intersections

/-- Define the tangency points of incircle and excircles -/
structure TangencyPoints :=
(X1 X2 X3 X4 : Point)
(incircle_EBC_tangency : tangent_point (incircle E B C) = X1)
(incircle_FCD_tangency : tangent_point (incircle F C D) = X2)
(excircle_EAD_tangency : tangent_point (excircle E A D) = X3)
(excircle_FAB_tangency : tangent_point (excircle F A B) = X4)

/-- Main theorem statement -/
theorem collinearity_condition 
  (Q : Quadrilateral)
  (T : TangencyPoints) :
  collinear Q.X1 Q.X3 Q.E ↔ collinear Q.X2 Q.X4 Q.F :=
sorry

end collinearity_condition_l665_665747


namespace prime_of_form_a2_minus_1_l665_665233

theorem prime_of_form_a2_minus_1 (a : ℕ) (p : ℕ) (ha : a ≥ 2) (hp : p = a^2 - 1) (prime_p : Nat.Prime p) : p = 3 := 
by 
  sorry

end prime_of_form_a2_minus_1_l665_665233


namespace cube_volume_is_1728_l665_665942

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l665_665942


namespace total_cost_is_13_l665_665141

-- Definition of pencil cost
def pencil_cost : ℕ := 2

-- Definition of pen cost based on pencil cost
def pen_cost : ℕ := pencil_cost + 9

-- The total cost of both items
def total_cost := pencil_cost + pen_cost

theorem total_cost_is_13 : total_cost = 13 := by
  sorry

end total_cost_is_13_l665_665141


namespace sequence_odd_for_odd_n_l665_665670

theorem sequence_odd_for_odd_n (a : ℕ → ℤ) (h₁ : a 1 = 2) (h₂ : a 2 = 7)
  (h₃ : ∀ n > 1, -1/2 < (a (n+1) - (-a n ^ 2 / a (n-1) ^ 2)) ∧ (a (n+1) - (-a n ^ 2 / a (n-1) ^ 2)) < 1/2) :
  ∀ n > 1, odd n → odd (a n) :=
by
  sorry

end sequence_odd_for_odd_n_l665_665670


namespace cube_volume_l665_665890

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l665_665890


namespace average_salary_of_managers_l665_665124

theorem average_salary_of_managers (m_avg : ℝ) (assoc_avg : ℝ) (company_avg : ℝ)
  (managers : ℕ) (associates : ℕ) (total_employees : ℕ)
  (h_assoc_avg : assoc_avg = 30000) (h_company_avg : company_avg = 40000)
  (h_managers : managers = 15) (h_associates : associates = 75) (h_total_employees : total_employees = 90)
  (h_total_employees_def : total_employees = managers + associates)
  (h_total_salary_managers : ∀ m_avg, total_employees * company_avg = managers * m_avg + associates * assoc_avg) :
  m_avg = 90000 :=
by
  sorry

end average_salary_of_managers_l665_665124


namespace problem1_problem2_l665_665043

constant A B C a b c : ℝ
constant area : ℝ

-- Conditions
axiom angle_condition : sin (A + C) = 8 * sin (B / 2) ^ 2
axiom sides_sum : a + c = 6
axiom triangle_area : area = 2

-- Proof problem statements
theorem problem1 : cos B = 15 / 17 := by
  sorry

theorem problem2 : b = 2 := by
  have h_cosB : cos B = 15 / 17 := by sorry
  have h_sinB : sin B = 8 / 17 := by sorry
  have h_ac : a * c = 17 / 2 := by sorry
  sorry

end problem1_problem2_l665_665043


namespace length_of_wire_l665_665802

theorem length_of_wire
  (distance_between_poles : ℝ)
  (height_of_pole1 : ℝ)
  (height_of_pole2 : ℝ)
  (h_distance : distance_between_poles = 16)
  (h_pole1 : height_of_pole1 = 7)
  (h_pole2 : height_of_pole2 = 18) :
  let vertical_gap := height_of_pole2 - height_of_pole1 in
  let wire_length := real.sqrt (distance_between_poles ^ 2 + vertical_gap ^ 2) in
  wire_length = real.sqrt 377 := by
s

end length_of_wire_l665_665802


namespace quadratic_geometric_sequence_intersection_l665_665697

noncomputable def discriminant {a b c : ℝ} (h : b^2 = a * c) : ℝ :=
  b^2 - 4 * a * c

theorem quadratic_geometric_sequence_intersection
  (a b c : ℝ)
  (h : b^2 = a * c) :
  let Δ := discriminant h in Δ ≤ 0 → ∃ x₁ x₂ : ℝ, x₁ = x₂ ∨ x₁ ≠ x₂ :=
by
  sorry

end quadratic_geometric_sequence_intersection_l665_665697


namespace cube_volume_from_surface_area_example_cube_volume_l665_665970

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l665_665970


namespace solve_quadratic_eq_l665_665018

theorem solve_quadratic_eq (x : ℝ) : (x^2 - 2*x + 1 = 9) → (x = 4 ∨ x = -2) :=
by
  intro h
  sorry

end solve_quadratic_eq_l665_665018


namespace cube_volume_is_1728_l665_665943

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l665_665943


namespace line_in_plane_or_parallel_or_intersects_perpendicular_within_plane_l665_665630

variables {a : Line} {α : Plane}

-- Definitions of the conditions
def line_in_plane (a : Line) (α : Plane) : Prop := a ∈ α
def line_parallel_plane (a : Line) (α : Plane) : Prop := ∀ b, b ∈ α → ¬ ∃ c, c ∈ α ∧ c.parallel b
def line_intersects_plane (a : Line) (α : Plane) : Prop := ∃ p, p ∈ a ∧ p ∈ α

-- The theorem to prove the conclusion
theorem line_in_plane_or_parallel_or_intersects_perpendicular_within_plane
  (h1 : line_in_plane a α ∨ line_parallel_plane a α ∨ line_intersects_plane a α) :
  ∃ l, l ∈ α ∧ l.perpendicular a :=
sorry

end line_in_plane_or_parallel_or_intersects_perpendicular_within_plane_l665_665630


namespace cube_volume_is_1728_l665_665941

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l665_665941


namespace area_of_rectangle_l665_665463

theorem area_of_rectangle (length width : ℝ) (h1 : length = 15) (h2 : width = length * 0.9) : length * width = 202.5 := by
  sorry

end area_of_rectangle_l665_665463


namespace log_equation_solution_l665_665427

theorem log_equation_solution :
  (∃ x : ℝ, log 8 x + log 2 (x ^ 3) = 9) ↔ x = 2 ^ (27 / 10) :=
by
  sorry

end log_equation_solution_l665_665427


namespace cube_volume_l665_665897

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l665_665897


namespace max_tan_C_l665_665341

noncomputable def max_tan_C_of_triangle 
  (A B C : Type) (AB AC BC : ℝ)
  (h₁ : AB = 2) 
  (h₂ : AC^2 - BC^2 = 6) : ℝ :=
  if h : ∃ a b : ℝ, a^2 + b^2 - 2 * a * b * sqrt (5/3) = 4
  then (2 * sqrt 5 / 5)
  else 0

theorem max_tan_C 
  (A B C : Type) (AB AC BC : ℝ)
  (h₁ : AB = 2)
  (h₂ : AC^2 - BC^2 = 6) : 
  max_tan_C_of_triangle A B C AB AC BC h₁ h₂ = 2 * sqrt 5 / 5 := sorry

end max_tan_C_l665_665341


namespace mass_percentage_Na_in_NaClO_l665_665258

theorem mass_percentage_Na_in_NaClO :
  let mass_Na : ℝ := 22.99
  let mass_Cl : ℝ := 35.45
  let mass_O : ℝ := 16.00
  let mass_NaClO : ℝ := mass_Na + mass_Cl + mass_O
  (mass_Na / mass_NaClO) * 100 = 30.89 := by
sorry

end mass_percentage_Na_in_NaClO_l665_665258


namespace find_alpha_beta_condition_l665_665770

variable (α β : ℕ)

theorem find_alpha_beta_condition : 
  (21 + α + β) % 9 = 0 ∧ (α - β + 13) % 11 = 0 → 
  α = 2 ∧ β = 4 :=
begin
  sorry
end

end find_alpha_beta_condition_l665_665770


namespace cost_of_big_bottle_proof_l665_665028

noncomputable def cost_of_big_bottle (C : ℕ) : Prop :=
  (C = 2700)

theorem cost_of_big_bottle_proof :
  (∀ (big_bottle_volume small_bottle_volume : ℕ) 
     (small_bottle_cost savings small_bottle_count : ℕ),
     big_bottle_volume = 30 →
     small_bottle_volume = 6 →
     small_bottle_cost = 600 →
     savings = 300 →
     small_bottle_count = big_bottle_volume / small_bottle_volume →
     small_bottle_cost * small_bottle_count - savings = C →
     cost_of_big_bottle C
  ) :=
by
  intros big_bottle_volume small_bottle_volume small_bottle_cost savings small_bottle_count
         big_bottle_volume_eq small_bottle_volume_eq small_bottle_cost_eq savings_eq small_bottle_count_eq cost_eq
  rw [big_bottle_volume_eq, small_bottle_volume_eq, small_bottle_cost_eq, savings_eq, small_bottle_count_eq] at cost_eq
  sorry

end cost_of_big_bottle_proof_l665_665028


namespace find_smallest_M_l665_665777

def flag := list bool -- True represents yellow, False represents blue

def diverse_set (flags: list (fin N → bool)) : Prop :=
  ∃ (perm : fin N → fin N), ∀ (i : fin N), flags[perm i] i = flags[perm 0] 0

theorem find_smallest_M (N : ℕ) (hN : N ≥ 4) :
  (∃ M, ∀ (flags : list (fin N → bool)), flags.length = M → (∃ (ds : list (fin N → bool)), ds ⊆ flags ∧ ds.length = N ∧ diverse_set ds)) :=
begin
  use 2^(N-2) + 1,
  intros flags hflags,
  sorry
end

end find_smallest_M_l665_665777


namespace sum_3000_l665_665060

-- Definitions for the conditions
def geom_sum (a r : ℝ) (n : ℕ) := a * (1 - r^n) / (1 - r)

variables (a r : ℝ)
axiom sum_1000 : geom_sum a r 1000 = 300
axiom sum_2000 : geom_sum a r 2000 = 570

-- The property to prove
theorem sum_3000 : geom_sum a r 3000 = 813 :=
sorry

end sum_3000_l665_665060


namespace minimum_value_of_exponential_expr_l665_665682

theorem minimum_value_of_exponential_expr (x y : ℝ) (h1 : x + 2 * y = 2) (h2 : 3^x > 0) (h3 : 9^y > 0) :
  3^x + 9^y ≥ 6 :=
sorry

end minimum_value_of_exponential_expr_l665_665682


namespace amaya_movie_watching_time_l665_665169

theorem amaya_movie_watching_time :
  let uninterrupted_time_1 := 35
  let uninterrupted_time_2 := 45
  let uninterrupted_time_3 := 20
  let rewind_time_1 := 5
  let rewind_time_2 := 15
  let total_uninterrupted := uninterrupted_time_1 + uninterrupted_time_2 + uninterrupted_time_3
  let total_rewind := rewind_time_1 + rewind_time_2
  let total_time := total_uninterrupted + total_rewind
  total_time = 120 := by
  sorry

end amaya_movie_watching_time_l665_665169


namespace erica_has_correct_amount_l665_665786

def total_amount : ℝ := 450.32
def sams_amount : ℝ := 325.67
def ericas_amount : ℝ := 124.65

theorem erica_has_correct_amount :
  ericas_amount = total_amount - sams_amount :=
by
  unfold total_amount sams_amount ericas_amount
  simp
  exact rfl

end erica_has_correct_amount_l665_665786


namespace largest_k_for_square_l665_665490

theorem largest_k_for_square (points_per_side : ℕ) (k : ℕ) (h_points : points_per_side = 100) (h_k : k < 200) : 
  (∃ pts : finset (ℝ × ℝ), pts.card = points_per_side * 4 ∧ 
  (∀ p ∈ pts, p.1 ∈ (finset.range points_per_side : set ℝ) ∨ p.2 ∈ (finset.range points_per_side : set ℝ)) ∧
  (∀ s1 s2 ∈ pts, s1 ≠ s2 → s1 ≠ s2)) → k = 150 := 
by
  sorry

end largest_k_for_square_l665_665490


namespace min_empty_squares_after_move_l665_665857

-- Definitions taken from conditions
def checkerboard_size := 99
def flies_start_position : Matrix checkerboard_size checkerboard_size (fun _ _ => Bool) := sorry 
def fly_move : (fin checkerboard_size) → (fin checkerboard_size) → (fin checkerboard_size × fin checkerboard_size) := sorry

-- Theorem statement derived from the problem and solution
theorem min_empty_squares_after_move :
  ∃ empty_squares : Fin checkerboard_size → Fin checkerboard_size → Bool,
  (∀ i j, (¬ empty_squares i j) = flies_start_position i j) ∧ 
  (∑ i j, if empty_squares i j then 1 else 0) = 99 :=
sorry

end min_empty_squares_after_move_l665_665857


namespace min_toys_to_add_l665_665096

theorem min_toys_to_add (T x : ℕ) (h1 : T % 12 = 3) (h2 : T % 18 = 3) :
  ((T + x) % 7 = 0) → x = 4 :=
by
  sorry

end min_toys_to_add_l665_665096


namespace total_time_to_watch_movie_l665_665172

-- Define the conditions and the question
def uninterrupted_viewing_time : ℕ := 35 + 45 + 20
def rewinding_time : ℕ := 5 + 15
def total_time : ℕ := uninterrupted_viewing_time + rewinding_time

-- Lean statement of the proof problem
theorem total_time_to_watch_movie : total_time = 120 := by
  -- This is where the proof would go
  sorry

end total_time_to_watch_movie_l665_665172


namespace sum_binomial_coeff_eq_l665_665030

theorem sum_binomial_coeff_eq 
  (n m : ℕ) :
  (∑ k in (set.iota n).filter (λ k, true), (-1) ^ k * nat.choose n k * nat.choose m (n - k)) = 
  if m = n then 1 else if m < n then 0 else 0 := 
  sorry

end sum_binomial_coeff_eq_l665_665030


namespace cube_volume_from_surface_area_example_cube_volume_l665_665963

theorem cube_volume_from_surface_area (s : ℝ) (surface_area : ℝ) (volume : ℝ)
  (h_surface_area : surface_area = 6 * s^2) 
  (h_given_surface_area : surface_area = 864) :
  volume = s^3 :=
sorry

theorem example_cube_volume :
  ∃ (s volume : ℝ), (6 * s^2 = 864) ∧ (volume = s^3) ∧ (volume = 1728) :=
begin
  use 12,
  use 1728,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end cube_volume_from_surface_area_example_cube_volume_l665_665963


namespace age_difference_l665_665838

variable (A B C X : ℕ)

theorem age_difference 
  (h1 : C = A - 13)
  (h2 : A + B = B + C + X) 
  : X = 13 :=
by
  sorry

end age_difference_l665_665838


namespace roots_of_polynomial_l665_665247

-- Define the polynomial P(x) = x^3 - 3x^2 - x + 3
def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

-- Define the statement to prove the roots of the polynomial
theorem roots_of_polynomial :
  ∀ x : ℝ, (P x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) :=
by
  sorry

end roots_of_polynomial_l665_665247


namespace find_line_equation_l665_665692

noncomputable def equation_of_perpendicular_line : Prop := 
  ∃ (l : ℝ → ℝ) (x y : ℝ), 
    (l x = 4*x/3 - 17/3) ∧ 
    (x = -2 ∧ y = -3) ∧ 
    (3*x + 4*y - 3 = 0)

theorem find_line_equation (A : ℝ × ℝ) (B : ℝ → Prop) :
    A = (-2, -3) → 
    (∀ x y : ℝ, B (3*x + 4*y - 3 = 0)) → 
     ∃ (a b c : ℝ), 4*a - 3*b - c = 0 :=
by 
    sorry

end find_line_equation_l665_665692


namespace tom_rope_stories_l665_665074

/-- Define the conditions given in the problem. --/
def story_length : ℝ := 10
def rope_length : ℝ := 20
def loss_percentage : ℝ := 0.25
def pieces_of_rope : ℕ := 4

/-- Theorem to prove the number of stories Tom can lower the rope down. --/
theorem tom_rope_stories (story_length rope_length loss_percentage : ℝ) (pieces_of_rope : ℕ) : 
    story_length = 10 → 
    rope_length = 20 →
    loss_percentage = 0.25 →
    pieces_of_rope = 4 →
    pieces_of_rope * rope_length * (1 - loss_percentage) / story_length = 6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end tom_rope_stories_l665_665074


namespace distance_between_parallel_sides_l665_665636

-- Define an equilateral triangle
def equilateral_triangle (ABC : Type) :=
  ∃ (S : ℝ), (S > 0)

-- Define the smaller triangle inside it
def smaller_triangle (A₁B₁C₁ : Type) (ABC : Type) :=
  ∃ (Q : ℝ), (Q > 0 ∧ Q ≤ S)

-- Define the distance calculation based on the given conditions
theorem distance_between_parallel_sides {ABC A₁B₁C₁ : Type} (S Q : ℝ) 
  (h_eq_triangle : equilateral_triangle ABC) 
  (h_sm_triangle : smaller_triangle A₁B₁C₁ ABC) : 
  (Q ≥ (1 / 4) * S → d = (sqrt(3) / 3) * (sqrt(S) - sqrt(Q))) ∧ 
  (Q < (1 / 4) * S → d = (sqrt(3) / 3) * (sqrt(S) - sqrt(Q)) ∨ d = (sqrt(3) / 3) * (sqrt(S) + sqrt(Q))) :=
by
  sorry

end distance_between_parallel_sides_l665_665636


namespace investor_should_choose_first_plan_l665_665179

noncomputable def plan1Dist : probability_distribution ℝ := 
  gaussian 8 3

noncomputable def plan2Dist : probability_distribution ℝ := 
  gaussian 6 2

theorem investor_should_choose_first_plan :
  (plan1Dist.prob (set.Ioi 5)) > (plan2Dist.prob (set.Ioi 5)) :=
by sorry

end investor_should_choose_first_plan_l665_665179


namespace train_speed_l665_665997

theorem train_speed 
  (length : ℝ)
  (time : ℝ)
  (relative_speed : ℝ)
  (conversion_factor : ℝ)
  (h_length : length = 120)
  (h_time : time = 4)
  (h_relative_speed : relative_speed = 60)
  (h_conversion_factor : conversion_factor = 3.6) :
  (relative_speed / 2) * conversion_factor = 108 :=
by
  sorry

end train_speed_l665_665997


namespace angle_bisector_segment_length_l665_665727

theorem angle_bisector_segment_length
  (A B C E : Type)
  [triangle A B C]
  [angle_bisector A E (B, C)]
  (AE : ℝ) (hAe : AE = 5)
  (l : Type) (hL : line_through_point_at_angle E l (30 : ℝ))
  (BAC_segment : ℝ) (hBac_segment : BAC_segment = 2 * real.sqrt 3) :
  ∃ BE EC : ℝ, BE = 5 / 2 ∧ EC = 5 / 3 :=
by sorry

end angle_bisector_segment_length_l665_665727


namespace radius_of_wheel_l665_665154

-- Definitions corresponding to conditions
def total_distance : ℝ := 316.8
def num_revolutions : ℕ := 300
def circumference (total_distance : ℝ) (num_revolutions : ℕ) : ℝ := total_distance / num_revolutions
def π : ℝ := Real.pi

-- The statement to prove
theorem radius_of_wheel :
  let C := circumference total_distance num_revolutions in
  let radius_m := C / (2 * π) in
  radius_m * 100 = 16.8 :=
by
  sorry

end radius_of_wheel_l665_665154


namespace average_cost_for_9_hours_l665_665807

-- This defines the hourly parking cost calculation given the conditions.
def total_cost (hours : ℕ) : ℝ :=
  if hours <= 2 then 9.0
  else 9.0 + (hours - 2) * 1.75

-- Define the average cost per hour.
def average_cost_per_hour (hours : ℕ) : ℝ :=
  total_cost hours / hours

-- State the theorem that we need to prove.
theorem average_cost_for_9_hours : average_cost_per_hour 9 = 2.36 := sorry

end average_cost_for_9_hours_l665_665807


namespace circular_field_diameter_l665_665595

theorem circular_field_diameter :
  ∀ (rate cost : ℝ) (π : ℝ), rate = 2.50 ∧ cost = 109.96 ∧ π = 3.14159 →
  let C := cost / rate in
  let d := C / π in
  d ≈ 14 := 
by
  intros rate cost π h
  rcases h with ⟨hrate, hcost, hpi⟩
  let C := cost / rate
  let d := C / π
  calc
    d = 43.984 / 3.14159 : by rw [hcost, hrate, hpi]; norm_num
    ... ≈ 14 : by norm_num

end circular_field_diameter_l665_665595


namespace distance_points_l665_665188

theorem distance_points : 
  let P1 := (2, -1)
  let P2 := (7, 6)
  dist P1 P2 = Real.sqrt 74 :=
by
  sorry

end distance_points_l665_665188


namespace matilda_jellybeans_l665_665402

theorem matilda_jellybeans (steve_jellybeans : ℕ) (h_steve : steve_jellybeans = 84)
  (h_matt : ℕ) (h_matt_calc : h_matt = 10 * steve_jellybeans)
  (h_matilda : ℕ) (h_matilda_calc : h_matilda = h_matt / 2) :
  h_matilda = 420 := by
  sorry

end matilda_jellybeans_l665_665402


namespace width_of_each_glass_pane_l665_665155

noncomputable def width_of_pane (num_panes : ℕ) (total_area : ℝ) (length_of_pane : ℝ) : ℝ :=
  total_area / num_panes / length_of_pane

theorem width_of_each_glass_pane :
  width_of_pane 8 768 12 = 8 := by
  sorry

end width_of_each_glass_pane_l665_665155


namespace part1_part2_l665_665499

def sin60 := Real.sin (Real.pi / 3)
def cos60 := Real.cos (Real.pi / 3)
def tan45 := Real.tan (Real.pi / 4)
def cos30 := Real.cos (Real.pi / 6)

theorem part1 : sin60^2 + cos60^2 - tan45 = 0 := by
  sorry

theorem part2 : tan45^2 - Real.sqrt (cos30^2 - 2 * cos30 + 1) = sqrt(3) / 2 := by
  sorry

end part1_part2_l665_665499


namespace chocolate_game_winning_strategy_chocolate_game_starting_player_wins_chocolate_game_starting_player_loses_l665_665493

-- Define the initial problem for part (a)
theorem chocolate_game_winning_strategy (rows columns : ℕ) (marked_row marked_col : ℕ) :
  rows = 6 → columns = 8 → (2 ⊕ 5 ⊕ 1 ⊕ 4 ≠ 0) :=
by sorry

-- Define the problem for part (b)
theorem chocolate_game_starting_player_wins (m n : ℕ) :
  (m % 2 = 0) ∧ (n % 2 = 0) → true :=
by sorry

-- Define the problem for part (c)
theorem chocolate_game_starting_player_loses (m n : ℕ) :
  (m % 2 = 1) ∧ (n % 2 = 1) → false :=
by sorry

end chocolate_game_winning_strategy_chocolate_game_starting_player_wins_chocolate_game_starting_player_loses_l665_665493


namespace largest_sampled_item_l665_665072

theorem largest_sampled_item (n : ℕ) (m : ℕ) (a : ℕ) (k : ℕ)
  (hn : n = 360)
  (hm : m = 30)
  (hk : k = n / m)
  (ha : a = 105) :
  ∃ b, b = 433 ∧ (∃ i, i < m ∧ a = 1 + i * k) → (∃ j, j < m ∧ b = 1 + j * k) :=
by
  sorry

end largest_sampled_item_l665_665072


namespace cube_volume_from_surface_area_l665_665876

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l665_665876


namespace balls_in_boxes_l665_665322

theorem balls_in_boxes :
  (Nat.choose (7 + 3 - 1) (3 - 1)) = 36 := by
  sorry

end balls_in_boxes_l665_665322


namespace number_of_factors_m_l665_665753

-- Definition of m
def m : ℕ := 2^5 * 3^6 * 5^7 * 7^8

-- The number of natural-number factors of m
theorem number_of_factors_m : nat.factors_count m = 3024 := sorry

end number_of_factors_m_l665_665753


namespace primes_p_p2_p4_l665_665472

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem primes_p_p2_p4 (p : ℕ) (hp : is_prime p) (hp2 : is_prime (p + 2)) (hp4 : is_prime (p + 4)) :
  p = 3 :=
sorry

end primes_p_p2_p4_l665_665472


namespace incorrect_assumption_division_by_zero_l665_665488

theorem incorrect_assumption_division_by_zero (a b c : ℝ) (h1 : a > b) (h2 : a = b + c) (h3 : c > 0) :
  a(a - b - c) = b(a - b - c) → 0 = 0 ∧ a ≠ b :=
by
  intro h
  have h4 : a - b = c := by linarith
  have h5 : a - b - c = 0 := by linarith
  rw [h5, mul_zero] at h
  rw [h5, mul_zero] at h
  exact ⟨eq.refl 0, by { rw [h2, add_assoc, add_zero] at h1, linarith }⟩

end incorrect_assumption_division_by_zero_l665_665488


namespace cube_volume_from_surface_area_l665_665951

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l665_665951


namespace tank_capacity_ratio_l665_665444

-- Definitions from the problem conditions
def tank1_filled : ℝ := 300
def tank2_filled : ℝ := 450
def tank2_percentage_filled : ℝ := 0.45
def additional_needed : ℝ := 1250

-- Theorem statement
theorem tank_capacity_ratio (C1 C2 : ℝ) 
  (h1 : tank1_filled + tank2_filled + additional_needed = C1 + C2)
  (h2 : tank2_filled = tank2_percentage_filled * C2) : 
  C1 / C2 = 2 :=
by
  sorry

end tank_capacity_ratio_l665_665444


namespace ratio_of_areas_l665_665431

-- Define the side length of the squares
def s : ℝ := 1

-- Establish the conditions
def IC : ℝ := 3 / 4
def HI : ℝ := 1 / 4
def HD : ℝ := 3 / 4
def DE : ℝ := 1 / 4

-- Define the areas
def area_triangle_JIC : ℝ := (1 / 2) * IC * s
def area_triangle_ABD : ℝ := (1 / 2) * HI * s
def area_pentagon_AJICB : ℝ := s^2 - area_triangle_ABD + area_triangle_JIC
def total_area_squares : ℝ := 3 * s^2

-- The final proof statement
theorem ratio_of_areas : area_pentagon_AJICB / total_area_squares = 3 / 8 :=
by
  sorry

end ratio_of_areas_l665_665431


namespace larger_box_cost_l665_665776

-- Definitions based on the conditions

def ounces_large : ℕ := 30
def ounces_small : ℕ := 20
def cost_small : ℝ := 3.40
def price_per_ounce_better_value : ℝ := 0.16

-- The statement to prove
theorem larger_box_cost :
  30 * price_per_ounce_better_value = 4.80 :=
by sorry

end larger_box_cost_l665_665776


namespace exists_quadrilateral_pyramid_two_perpendicular_faces_exists_hexagonal_pyramid_three_perpendicular_faces_l665_665103

/-- There exists a quadrilateral pyramid such that two non-adjacent faces are perpendicular to the base plane. -/
theorem exists_quadrilateral_pyramid_two_perpendicular_faces :
  ∃ (A B C D O : Point3D), let ABCD := Quadrilateral A B C D in 
  ∃ ABOD BCDO : Face,
  is_perpendicular ABOD ABCD ∧
  is_perpendicular BCDO ABCD :=
sorry

/-- There exists a hexagonal pyramid such that three lateral faces are perpendicular to the base plane. -/
theorem exists_hexagonal_pyramid_three_perpendicular_faces :
  ∃ (A B C D E F O : Point3D), let ABCDEF := Hexagon A B C D E F in
  ∃ α β γ : Face,
  (α ≠ β ∧ β ≠ γ ∧ α ≠ γ) ∧
  is_perpendicular α ABCDEF ∧
  is_perpendicular β ABCDEF ∧
  is_perpendicular γ ABCDEF :=
sorry

end exists_quadrilateral_pyramid_two_perpendicular_faces_exists_hexagonal_pyramid_three_perpendicular_faces_l665_665103


namespace base_7_to_10_equivalence_l665_665085

theorem base_7_to_10_equivalence : 
  let n := 6 * 7^4 + 5 * 7^3 + 4 * 7^2 + 3 * 7^1 + 2 * 7^0
  in n = 16340 :=
by
  let n := 6 * 7^4 + 5 * 7^3 + 4 * 7^2 + 3 * 7^1 + 2 * 7^0
  show n = 16340
  -- The proof would go here, but we use sorry for the placeholder
  sorry

end base_7_to_10_equivalence_l665_665085


namespace pinocchio_cannot_always_determine_treasure_location_l665_665491

def pinocchio_determine_treasure_location (radio : ℕ → ℕ → bool) (is_lying: bool) : Prop :=
  ¬∀ (c1 c2 : ℕ), radio c1 c2 = (radio c2 c1 ≠ is_lying)

theorem pinocchio_cannot_always_determine_treasure_location (radio : ℕ → ℕ → bool) (is_lying: bool) :
  pinocchio_determine_treasure_location radio is_lying :=
by
  sorry

end pinocchio_cannot_always_determine_treasure_location_l665_665491


namespace base7_65432_to_dec_is_16340_l665_665086

def base7_to_dec (n : ℕ) : ℕ :=
  6 * 7^4 + 5 * 7^3 + 4 * 7^2 + 3 * 7^1 + 2 * 7^0

theorem base7_65432_to_dec_is_16340 : base7_to_dec 65432 = 16340 :=
by
  sorry

end base7_65432_to_dec_is_16340_l665_665086


namespace smallest_prime_factor_2145_l665_665865

theorem smallest_prime_factor_2145 : nat.min_fac 2145 = 3 :=
by sorry

end smallest_prime_factor_2145_l665_665865


namespace part1_proof_part2_proof_l665_665281

-- Condition definitions
def E : (ℝ × ℝ) := (1, (2 * Real.sqrt 3) / 3)
def focal_length : ℝ := 2
def k1 : ℝ := sorry
def k2 : ℝ := sorry
def k1_plus_k2 : Prop := k1 + k2 = 1

-- Part 1: Prove the standard equation of the ellipse
theorem part1_proof :
  ∃ (a b : ℝ), a = Real.sqrt 3 ∧ b = Real.sqrt 2 ∧ (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 3 + y^2 / 2 = 1)) :=
sorry

-- Part 2: Prove the line MN passes through a fixed point if k1 + k2 = 1
theorem part2_proof :
  k1_plus_k2 →
  ∀ M N: (ℝ × ℝ), (M = ((1 + 1) / 2, (1 + 1) / 2)) ∧ (N = ((1 + 1) / 2, (1 + 1) / 2)) → 
  ∃ (fixed : ℝ × ℝ), fixed = (0, -(2 / 3)) ∧ (let line := λ (x : ℝ), -(2 / 3) in line 0 = -(2 / 3)) :=
sorry

end part1_proof_part2_proof_l665_665281


namespace breakfast_cost_total_l665_665613

def muffin_price : ℝ := 2
def fruit_cup_price : ℝ := 3
def coffee_price : ℝ := 1.5
def discount_percentage : ℝ := 0.10
def voucher : ℝ := 2

def cost_without_discount (muffins : ℕ) (fruit_cups : ℕ) (coffees : ℕ) : ℝ :=
  muffins * muffin_price + fruit_cups * fruit_cup_price + coffees * coffee_price

def apply_discount (total : ℝ) (percentage : ℝ) : ℝ :=
  total - total * percentage

def francis_cost :=
  let subtotal := muffin_price * 2 + fruit_cup_price in
  let discounted := apply_discount subtotal discount_percentage in
  discounted + fruit_cup_price + coffee_price

def kiera_cost :=
  let subtotal := muffin_price * 2 + fruit_cup_price in
  let discounted := apply_discount subtotal discount_percentage in
  discounted + coffee_price

def david_cost :=
  let subtotal := muffin_price * 3 + fruit_cup_price + coffee_price in
  subtotal - voucher

def total_cost :=
  francis_cost + kiera_cost + david_cost

theorem breakfast_cost_total : total_cost = 27.10 :=
  by
    -- Skipping the proof
    sorry

end breakfast_cost_total_l665_665613


namespace boys_count_l665_665411

variables (B : ℕ) -- Number of boys
variables (total_children happy_children sad_children neither_happy_nor_sad_children girls happy_boys sad_girls boys_neither_happy_nor_sad: ℕ)

-- Given conditions
axiom total_children_eq : total_children = 60
axiom girls_eq : girls = 44
axiom boys_eq : B = total_children - girls

-- Proving the number of boys
theorem boys_count : B = 16 :=
by
  rw [total_children_eq, girls_eq] at boys_eq
  rw boys_eq
  exact rfl

end boys_count_l665_665411


namespace determinant_of_triangle_angles_zero_l665_665762

noncomputable def prove_det (
  A B C : ℝ
) (h : A + B + C = π) : 
  matrix ℝ (fin 3) (fin 3) := 
  ![
    ![cos A ^ 2, tan A, 1],
    ![cos B ^ 2, tan B, 1],
    ![cos C ^ 2, tan C, 1]
  ]

theorem determinant_of_triangle_angles_zero (A B C : ℝ) (h : A + B + C = π) :
  matrix.det (prove_det A B C h) = 0 := 
sorry

end determinant_of_triangle_angles_zero_l665_665762


namespace pillbox_days_count_l665_665183

noncomputable def calculate_days (total_pills : Nat) (pills_left : Nat) (pills_per_day : Nat) : Nat :=
  (total_pills - pills_left) / pills_per_day

theorem pillbox_days_count :
  let total_pills := 3 * 120 + 2 * 30
  let pills_left := 350
  let pills_per_day := 5
  calculate_days total_pills pills_left pills_per_day = 14 :=
by
  let total_pills := 3 * 120 + 2 * 30
  let pills_left := 350
  let pills_per_day := 5
  calc
    calculate_days total_pills pills_left pills_per_day
        = (total_pills - pills_left) / pills_per_day : rfl
    ... = (420 - 350) / 5 : by rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul, Nat.linear_rec]
    ... = 70 / 5 : rfl
    ... = 14 : rfl

end pillbox_days_count_l665_665183


namespace distance_midpoint_parabola_y_axis_l665_665642

theorem distance_midpoint_parabola_y_axis (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (hA : y1 ^ 2 = x1) (hB : y2 ^ 2 = x2) 
  (h_focus : ∀ {p : ℝ × ℝ}, p = (x1, y1) ∨ p = (x2, y2) → |p.1 - 1/4| = |p.1 + 1/4|)
  (h_dist : |x1 - 1/4| + |x2 - 1/4| = 3) :
  abs ((x1 + x2) / 2) = 5 / 4 :=
by sorry

end distance_midpoint_parabola_y_axis_l665_665642


namespace cube_volume_l665_665909

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l665_665909


namespace cos_2phi_eq_3_over_5_l665_665667

theorem cos_2phi_eq_3_over_5 (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π)
  (h3 : ∀ x, f(x, φ) = sin(x + φ) - 2 * cos(x + φ))
  (h4 : f(π / 2, φ) = f(3 * π / 2, φ)) :
  cos(2 * φ) = 3 / 5 :=
by
  sorry

end cos_2phi_eq_3_over_5_l665_665667


namespace cube_volume_l665_665926

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l665_665926


namespace hyperbola_eccentricity_l665_665668

theorem hyperbola_eccentricity (a b c : ℝ) (A B F : ℝ × ℝ)
  (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_A_on_asymptote : A = (c, b * c / a))
  (h_B_on_asymptote : B = (c / 2, -b * c / (2 * a)))
  (h_AF_perpendicular_x : A.2 = 0)
  (h_AB_perpendicular_OB : A.1 * B.1 + A.2 * B.2 = 0)
  (h_BF_parallel_OA : B.2 * F.1 = B.1 * F.2) :
  ∀ e : ℝ, e = c / a → (c^2 = a^2 + b^2) → b^2 = a^2 / 3 → e = 2 * real.sqrt 3 / 3 :=
by 
  sorry

end hyperbola_eccentricity_l665_665668


namespace number_of_solutions_l665_665204

theorem number_of_solutions (n : ℕ) :
  (∃ s : set ℝ, (∀ x ∈ s, 3 * (Real.cos x)^3 - 7 * (Real.cos x)^2 + 3 * Real.cos x + 1 = 0 ∧ 0 ≤ x ∧ x ≤ Real.pi) ∧ s.card = n) ↔ n = 5 :=
by
  sorry

end number_of_solutions_l665_665204


namespace relationship_of_a_b_c_l665_665628

-- Define the function and its properties
variable {f : ℝ → ℝ}

-- Conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_inequality : ∀ x, x < 0 → f x + x * f'' x > 0

-- Definition of variables
def a := (2 ^ 0.6) * f (2 ^ 0.6)
def b := (Real.log 2) * f (Real.log 2)
def c := (Real.log 2⁻³) * f (Real.log 2⁻³)

-- Goal
theorem relationship_of_a_b_c : b > a ∧ a > c :=
by 
  sorry

end relationship_of_a_b_c_l665_665628


namespace extreme_value_at_zero_l665_665598

noncomputable def extreme_value (x : ℝ) : ℝ :=
  (x^2 - 1)^3 + 1

theorem extreme_value_at_zero : ∃ x : ℝ, extreme_value x = 0 :=
by {
  use 0,
  rw [extreme_value],
  norm_num,
  sorry -- proof left out as per instructions
}

end extreme_value_at_zero_l665_665598


namespace Xiaoxi_has_largest_final_answer_l665_665376

def Laura_final : ℕ := 8 - 2 * 3 + 3
def Navin_final : ℕ := (8 * 3) - 2 + 3
def Xiaoxi_final : ℕ := (8 - 2 + 3) * 3

theorem Xiaoxi_has_largest_final_answer : 
  Xiaoxi_final > Laura_final ∧ Xiaoxi_final > Navin_final :=
by
  unfold Laura_final Navin_final Xiaoxi_final
  -- Proof steps would go here, but we skip them as per instructions
  sorry

end Xiaoxi_has_largest_final_answer_l665_665376


namespace smallest_value_bounds_l665_665054

variable {a : Fin 8 → ℝ}

theorem smallest_value_bounds
  (h1 : (∑ i, a i) = 4 / 3)
  (h2 : ∀ j, (∑ i, if i = j then 0 else a i) > 0) :
  ∃ a1, -8 < a1 ∧ a1 ≤ 1 / 6 :=
begin
  let a1 := a 0,
  use a1,
  split,
  { sorry },
  { sorry }
end

end smallest_value_bounds_l665_665054


namespace difference_of_percentages_l665_665340

variable (x y : ℝ)

theorem difference_of_percentages :
  (0.60 * (50 + x)) - (0.45 * (30 + y)) = 16.5 + 0.60 * x - 0.45 * y := 
sorry

end difference_of_percentages_l665_665340


namespace ellipse_area_correct_l665_665703

noncomputable def ellipse_area : ℝ :=
  let a := 11 in
  let b := Real.sqrt (1089 / 40) in
  π * a * b

theorem ellipse_area_correct :
  let major_axis_endpoints := ((-12 : ℝ), 3), (10, 3)
  let point_on_ellipse := (8, 6)
  let center := (-1, 3)
  let a := 11
  let b := Real.sqrt (1089 / 40)
  a = 11 ∧ center = (-1, 3) ∧ b = Real.sqrt (1089 / 40) ∧
  ellipse_area = (181.5 / Real.sqrt 10) * π := by
  sorry

end ellipse_area_correct_l665_665703


namespace polynomial_roots_l665_665239

theorem polynomial_roots :
  (∀ x, x^3 - 3*x^2 - x + 3 = 0 ↔ (x = 1 ∨ x = -1 ∨ x = 3)) :=
by
  intro x
  split
  {
    intro h
    have h1 : x = 1 ∨ x = -1 ∨ x = 3
    {
      sorry
    }
    exact h1
  }
  {
    intro h
    cases h
    {
      rw h
      simp
    }
    {
      cases h
      {
        rw h
        simp
      }
      {
        rw h
        simp
      }
    }
  }

end polynomial_roots_l665_665239


namespace parabola_vertex_l665_665805

-- Define the parabola equation
def parabola_equation (x : ℝ) : ℝ := (x - 2)^2 + 5

-- State the theorem to find the vertex
theorem parabola_vertex : ∃ h k : ℝ, ∀ x : ℝ, parabola_equation x = (x - h)^2 + k ∧ h = 2 ∧ k = 5 :=
by
  sorry

end parabola_vertex_l665_665805


namespace number_of_distinct_paths_l665_665578

theorem number_of_distinct_paths (r u : ℕ) (h : r = 7 ∧ u = 3) :
  (nat.choose (r + u) u) = 120 := 
begin
  sorry
end

end number_of_distinct_paths_l665_665578


namespace roots_of_polynomial_l665_665237

def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

theorem roots_of_polynomial :
  {x : ℝ | P x = 0} = {1, -1, 3} := 
sorry

end roots_of_polynomial_l665_665237


namespace find_m_l665_665663

def f (x : ℝ) : ℝ :=
  if x < 1 then x + 3 else x^2 - 2 * x

theorem find_m (m : ℝ) (h : f m = 3) : m = 0 ∨ m = 3 := by
  sorry

end find_m_l665_665663


namespace depth_of_channel_l665_665106

theorem depth_of_channel (h : ℝ) 
  (top_width : ℝ := 12) (bottom_width : ℝ := 6) (area : ℝ := 630) :
  1 / 2 * (top_width + bottom_width) * h = area → h = 70 :=
sorry

end depth_of_channel_l665_665106


namespace johns_money_l665_665164

theorem johns_money (total_money ali_less nada_more: ℤ) (h1: total_money = 67) 
  (h2: ali_less = -5) (h3: nada_more = 4): 
  ∃ (j: ℤ), (n: ℤ), ali_less = n - 5 ∧ nada_more = 4 * n ∧ total_money = n + (n - 5) + (4 * n) → j = 48 :=
by
  sorry

end johns_money_l665_665164


namespace factor_expression_l665_665226

theorem factor_expression (x : ℝ) : 
  5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by sorry

end factor_expression_l665_665226


namespace exists_valid_long_sequence_l665_665707

def valid_sequence (s : List ℕ) : Prop :=
  (s.sum = 20) ∧
  (∀ n ∈ s, n ≠ 3) ∧
  (∀ i j, i < j → (s.drop i).take (j - i + 1).sum ≠ 3)

theorem exists_valid_long_sequence :
  ∃ s : List ℕ, valid_sequence s ∧ s.length > 10 :=
by
  sorry

end exists_valid_long_sequence_l665_665707


namespace reward_model_satisfies_conditions_l665_665122

def f1 (x : ℝ) := (1/4) * x
def f2 (x : ℝ) := Real.log x / Real.log 10 + 1 -- logrithm in base 10
def f3 (x : ℝ) := (3/2)^x
def f4 (x : ℝ) := Real.sqrt x

theorem reward_model_satisfies_conditions :
  ∀ f : ℝ → ℝ,
    (f = f1 ∨ f = f2 ∨ f = f3 ∨ f = f4) →
    (∀ x : ℝ, 10 ≤ x ∧ x ≤ 1000 → (f x) ≤ (1/4) * x) →
    (∀ x : ℝ, 10 ≤ x ∧ x ≤ 1000 → (f x) ≤ 5) →
    (∀ x : ℝ, 10 ≤ x ∧ x ≤ 1000 → f x ≤ f (x + 1)) →
    (f = f2) := sorry

end reward_model_satisfies_conditions_l665_665122


namespace cube_volume_from_surface_area_l665_665867

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l665_665867


namespace total_cost_is_13_l665_665142

-- Definition of pencil cost
def pencil_cost : ℕ := 2

-- Definition of pen cost based on pencil cost
def pen_cost : ℕ := pencil_cost + 9

-- The total cost of both items
def total_cost := pencil_cost + pen_cost

theorem total_cost_is_13 : total_cost = 13 := by
  sorry

end total_cost_is_13_l665_665142


namespace value_of_n_times_s_l665_665382

-- Define T as the set of nonzero real numbers
def T : Set Real := { x : Real | x ≠ 0 }

-- Define the function g with its properties as conditions
axiom g : T → T
axiom g_property1 : ∀ x : Real, x ∈ T → g (1 / x) = 2 * x * g x
axiom g_property2 : ∀ (x y : Real), x ∈ T → y ∈ T → x + y ∈ T → g (1 / x) + g (1 / y) = 2 + g (1 / (x + y))

-- The theorem we want to prove
theorem value_of_n_times_s : 1 * 2 = 2 :=
by
  sorry

end value_of_n_times_s_l665_665382


namespace cube_volume_from_surface_area_l665_665871

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l665_665871


namespace quadratic_completing_square_l665_665795

theorem quadratic_completing_square :
  ∀ x : ℝ, 4 * x^2 - 8 * x - 288 = 0 → ∃ r s : ℝ, (x + r)^2 = s ∧ s = 73 :=
by
  assume x h
  sorry

end quadratic_completing_square_l665_665795


namespace find_c_find_k_l665_665314

def a : ℝ × ℝ × ℝ := (1, 1, 0)

def b (c : ℝ) : ℝ × ℝ × ℝ := (-1, 0, c)

def vec_add (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2, u.3 + v.3)

def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (u.1^2 + u.2^2 + u.3^2)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem find_c (c : ℝ) : magnitude (vec_add a (b c)) = real.sqrt 5 ↔ c = 2 ∨ c = -2 :=
sorry

theorem find_k (k c : ℝ) (Hc : c = 2 ∨ c = -2) :
  dot_product (vec_add (k • a) (b c)) (vec_add (2 • a) (-b c)) = 0 ↔ k = 7 / 5 :=
sorry

end find_c_find_k_l665_665314


namespace false_statement_l665_665991

-- Define the geometrical conditions based on the problem statements
variable {A B C D: Type}

-- A rhombus with equal diagonals is a square
def rhombus_with_equal_diagonals_is_square (R : A) : Prop := 
  ∀ (a b : A), a = b → true

-- A rectangle with perpendicular diagonals is a square
def rectangle_with_perpendicular_diagonals_is_square (Rec : B) : Prop :=
  ∀ (a b : B), a = b → true

-- A parallelogram with perpendicular and equal diagonals is a square
def parallelogram_with_perpendicular_and_equal_diagonals_is_square (P : C) : Prop :=
  ∀ (a b : C), a = b → true

-- A quadrilateral with perpendicular and bisecting diagonals is a square
def quadrilateral_with_perpendicular_and_bisecting_diagonals_is_square (Q : D) : Prop :=
  ∀ (a b : D), (a = b) → true 

-- The main theorem: Statement D is false
theorem false_statement (Q : D) : ¬ (quadrilateral_with_perpendicular_and_bisecting_diagonals_is_square Q) := 
  sorry

end false_statement_l665_665991


namespace paris_weekday_study_hours_l665_665810

def weekday_study_hours (weeks : ℕ) (total_weekend_hours_per_week : ℕ) (total_study_hours : ℕ) (weekday_study_hours : ℕ) : ℕ :=
  let weekday_hours_per_week := 5 * weekday_study_hours
  let total_weekday_hours := weeks * weekday_hours_per_week
  let total_weekend_hours := weeks * total_weekend_hours_per_week
  total_study_hours - total_weekend_hours

theorem paris_weekday_study_hours:
  ∀ (weeks : ℕ) (total_weekend_hours_per_week : ℕ) (total_study_hours : ℕ) (weekday_study_hours : ℕ),
  weeks = 15 → total_weekend_hours_per_week = 9 → total_study_hours = 360 → weekday_study_hours = 3 →
  weekday_study_hours weeks total_weekend_hours_per_week total_study_hours = 3 :=
by {
  intros,
  simp,
  sorry
}

end paris_weekday_study_hours_l665_665810


namespace cube_volume_l665_665834

theorem cube_volume (s : ℝ) (h1 : 6 * s^2 = 1734) : s^3 = 4913 := by
  sorry

end cube_volume_l665_665834


namespace sum_of_basic_terms_divisible_by_4_l665_665464

open Matrix

def basic_term {n : ℕ} (A : Matrix (Fin n) (Fin n) ℤ) (σ : Equiv.Perm (Fin n)) : ℤ :=
  ∏ i, A i (σ i)

def sum_of_basic_terms {n : ℕ} [Fact (4 ≤ n)] (A : Matrix (Fin n) (Fin n) ℤ) : ℤ :=
  ∑ σ in Equiv.Perm.univ (Fin n), basic_term A σ

theorem sum_of_basic_terms_divisible_by_4 {n : ℕ} [Fact (4 ≤ n)] (A : Matrix (Fin n) (Fin n) ℤ)
  (h : ∀ i j, A i j = 1 ∨ A i j = -1) :
  sum_of_basic_terms A % 4 = 0 :=
by
  sorry

end sum_of_basic_terms_divisible_by_4_l665_665464


namespace probability_other_side_red_given_seen_red_l665_665505

-- Definition of conditions
def total_cards := 9
def black_black_cards := 5
def black_red_cards := 2
def red_red_cards := 2
def red_sides := (2 * red_red_cards) + black_red_cards -- Total number of red sides
def favorable_red_red_sides := 2 * red_red_cards      -- Number of red sides on fully red cards

-- The required probability
def probability_other_side_red_given_red : ℚ := sorry

-- The main statement to prove
theorem probability_other_side_red_given_seen_red :
  probability_other_side_red_given_red = 2/3 :=
sorry

end probability_other_side_red_given_seen_red_l665_665505


namespace problem_part1_problem_part2_l665_665290

open Real

def vec_a : ℝ × ℝ × ℝ := (2, -1, -2)
def vec_b : ℝ × ℝ × ℝ := (1, 1, -4)

def vec_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def vec_scalar_mul (s : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (s * v.1, s * v.2, s * v.3)

def vec_dot (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def vec_norm (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

theorem problem_part1 :
  vec_norm (vec_sub (vec_scalar_mul 2 vec_a) (vec_scalar_mul 3 vec_b)) = 3 * sqrt 10 :=
  sorry

theorem problem_part2 (k : ℝ) (h : vec_dot (vec_sub (vec_scalar_mul 1 vec_a) 
                      (vec_scalar_mul (-2) vec_b)) (vec_sub (vec_scalar_mul k vec_a) 
                      (vec_scalar_mul 3 vec_b)) = 0) : k = 5 :=
  sorry

end problem_part1_problem_part2_l665_665290


namespace smallest_value_of_y_l665_665757

theorem smallest_value_of_y (x y z d : ℝ) (h1 : x = y - d) (h2 : z = y + d) (h3 : x * y * z = 125) (h4 : 0 < x ∧ 0 < y ∧ 0 < z) : y ≥ 5 :=
by
  -- Officially, the user should navigate through the proof, but we conclude with 'sorry' as placeholder
  sorry

end smallest_value_of_y_l665_665757


namespace triangle_angles_and_type_l665_665152

theorem triangle_angles_and_type
  (largest_angle : ℝ)
  (smallest_angle : ℝ)
  (middle_angle : ℝ)
  (h1 : largest_angle = 90)
  (h2 : largest_angle = 3 * smallest_angle)
  (h3 : largest_angle + smallest_angle + middle_angle = 180) :
  (largest_angle = 90 ∧ middle_angle = 60 ∧ smallest_angle = 30 ∧ largest_angle = 90) := by
  sorry

end triangle_angles_and_type_l665_665152


namespace intersection_S_T_eq_T_l665_665767

noncomputable def S : Set ℝ := { y | ∃ x : ℝ, y = 3^x }
noncomputable def T : Set ℝ := { y | ∃ x : ℝ, y = x^2 + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l665_665767


namespace beneficial_to_buy_card_breakeven_visits_l665_665999

section PartA
variables (visits_per_week : ℕ) (weeks_per_year : ℕ) (average_check : ℝ) (card_cost : ℝ) (discount_rate : ℝ)

def total_savings (visits_per_week weeks_per_year : ℕ) (average_check discount_rate : ℝ) : ℝ :=
  (visits_per_week * weeks_per_year) * (average_check * discount_rate)

theorem beneficial_to_buy_card (h1 : visits_per_week = 3) (h2 : weeks_per_year = 52) (h3 : average_check = 900) (h4 : card_cost = 30000) (h5 : discount_rate = 0.30) :
  total_savings visits_per_week weeks_per_year average_check discount_rate > card_cost :=
sorry
end PartA

section PartB
variables (average_check : ℝ) (card_cost : ℝ) (discount_rate : ℝ)

def breakeven_visits_per_year (average_check card_cost discount_rate : ℝ) : ℝ :=
  card_cost / (average_check * discount_rate)

theorem breakeven_visits (h1 : average_check = 600) (h2 : card_cost = 30000) (h3 : discount_rate = 0.30) :
  breakeven_visits_per_year average_check card_cost discount_rate = 167 :=
sorry
end PartB

end beneficial_to_buy_card_breakeven_visits_l665_665999


namespace smallest_value_bounds_l665_665055

variable {a : Fin 8 → ℝ}

theorem smallest_value_bounds
  (h1 : (∑ i, a i) = 4 / 3)
  (h2 : ∀ j, (∑ i, if i = j then 0 else a i) > 0) :
  ∃ a1, -8 < a1 ∧ a1 ≤ 1 / 6 :=
begin
  let a1 := a 0,
  use a1,
  split,
  { sorry },
  { sorry }
end

end smallest_value_bounds_l665_665055


namespace students_neither_music_nor_art_l665_665506

theorem students_neither_music_nor_art
  (total_students : ℕ) (students_music : ℕ) (students_art : ℕ) (students_both : ℕ)
  (h_total : total_students = 500)
  (h_music : students_music = 30)
  (h_art : students_art = 10)
  (h_both : students_both = 10)
  : total_students - (students_music + students_art - students_both) = 460 :=
by
  rw [h_total, h_music, h_art, h_both]
  norm_num
  sorry

end students_neither_music_nor_art_l665_665506


namespace roots_of_polynomial_l665_665236

def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

theorem roots_of_polynomial :
  {x : ℝ | P x = 0} = {1, -1, 3} := 
sorry

end roots_of_polynomial_l665_665236


namespace product_of_roots_is_integer_l665_665859

theorem product_of_roots_is_integer :
  (81^(1/4 : ℝ)) * (27^(1/3 : ℝ)) * (16^(1/2 : ℝ)) = 36 := by
  sorry

end product_of_roots_is_integer_l665_665859


namespace sum_of_common_ratios_l665_665498

theorem sum_of_common_ratios (k p r : ℝ) (h1 : k ≠ 0) (h2 : k * (p^2) - k * (r^2) = 5 * (k * p - k * r)) (h3 : p ≠ r) : p + r = 5 :=
sorry

end sum_of_common_ratios_l665_665498


namespace ellipse_equation_and_eccentricity_existence_of_P_l665_665635

-- Define the conditions and the problem
structure Point where
  x : ℝ
  y : ℝ

def ellipse (a b : ℝ) (p : Point) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (p.x^2 / a^2 + p.y^2 / b^2 = 1)

def line (A B : Point) : Prop :=
  A.x + A.y = 4 ∧ B.x + B.y = 4

-- Question Ⅰ: Equation of the ellipse and its eccentricity
theorem ellipse_equation_and_eccentricity (a b c e : ℝ) (C : Point → Prop) (A B : Point) :
  (C A ∧ C B ∧ ellipse a b A ∧ ellipse a b B) →
  a = 2 →
  b = 1 →
  C = λ p, (p.x^2 / 4 + p.y^2 = 1) ∧
  c = sqrt (a^2 - b^2) →
  e = c / a →
  (C = λ p, (p.x^2 / 4 + p.y^2 = 1)) ∧ e = (sqrt 3 / 2) :=
sorry

-- Question Ⅱ: Existence of point P on the line such that PAQB is a parallelogram
theorem existence_of_P (C : Point → Prop) (A B Q P : Point) :
  (C A ∧ C B ∧ C Q ∧ line P) →
  (A = ⟨2, 0⟩ ∧ B = ⟨0, 1⟩ ∧ P.x + P.y = 4 ∧ (C = λ p, p.x^2 / 4 + p.y^2 = 1)) →
  ∃ P, ((P.x, P.y) = (18 / 5, 2 / 5) ∨ (P.x, P.y) = (2, 2)) ∧ sorry := sorry

end ellipse_equation_and_eccentricity_existence_of_P_l665_665635


namespace value_of_x_l665_665683

theorem value_of_x (x : ℝ) : (2 : ℝ) = 1 / (4 * x + 2) → x = -3 / 8 := 
by
  intro h
  sorry

end value_of_x_l665_665683


namespace vegetarian_family_l665_665706

theorem vegetarian_family (eat_veg eat_non_veg eat_both : ℕ) (total_veg : ℕ) 
  (h1 : eat_non_veg = 8) (h2 : eat_both = 11) (h3 : total_veg = 26)
  : eat_veg = total_veg - eat_both := by
  sorry

end vegetarian_family_l665_665706


namespace find_b_l665_665657

theorem find_b (a b : ℤ) (h1 : 3 * a + 1 = 1) (h2 : b - a = 2) : b = 2 := by
  sorry

end find_b_l665_665657


namespace rearrangement_inequality_l665_665066

variable (n : ℕ)
variable (a b : Fin n.succ → ℝ)  -- Indexing starts from 1 to match with n.
variable (H1 : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n.succ → a i > a j)
variable (H2 : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n.succ → b i > b j)

theorem rearrangement_inequality : 
  (∑ i in Finset.range n.succ, a i.succ * b i.succ) > (∑ i in Finset.range n.succ, a i.succ * b (n.succ - i)) :=
sorry

end rearrangement_inequality_l665_665066


namespace part_1_part_2_l665_665144

variables (P : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ) (α : ℝ)
def circle (O : ℝ × ℝ) (r : ℝ) := { P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2 }
def point := P = (-1, 2)
def inside_circle := circle (0, 0) 8

-- Condition for part (1)
def line_AB_slope_1 := α = 3 * Real.pi / 4
def line_AB_1 := ∃ k : ℝ, k = -1 ∧ P.2 - 2 = k * (P.1 + 1)
def distance_AB_1 := ∃ d : ℝ, d = abs (-1) / Real.sqrt 2
def length_chord_AB_1 := ∃ AB : ℝ, AB = 2 * Real.sqrt (8 - (√2 / 2)^2)

theorem part_1 : inside_circle P → line_AB_slope_1 → length_chord_AB_1 = Real.sqrt 30 :=
sorry

-- Condition for part (2)
def midpoint_AB := ∃ A B : ℝ × ℝ, P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2 ∧
  circle (0, 0) 8 A ∧ circle (0, 0) 8 B
def perpendicular_OP_AB := ∃ k_OP k_AB : ℝ, k_OP = -2 ∧ k_AB = 1 / 2 ∧
  k_OP * k_AB = -1
def line_AB_2 := ∃ k : ℝ, k = 1 / 2 ∧ P.2 - 2 = 1 / 2 * (P.1 + 1) ∧
  ∃ c : ℝ, x - 2 * y + 5 = 0

theorem part_2 : midpoint_AB → perpendicular_OP_AB → line_AB_2 = line_AB_2 :=
sorry

end part_1_part_2_l665_665144


namespace average_salary_of_managers_l665_665125

theorem average_salary_of_managers 
    (num_managers num_associates : ℕ) 
    (avg_salary_associates avg_salary_company : ℝ) 
    (H_managers : num_managers = 15) 
    (H_associates : num_associates = 75) 
    (H_avg_associates : avg_salary_associates = 30000) 
    (H_avg_company : avg_salary_company = 40000) : 
    ∃ M : ℝ, 15 * M + 75 * 30000 = 90 * 40000 ∧ M = 90000 := 
by
    use 90000
    rw [H_managers, H_associates, H_avg_associates, H_avg_company]
    split
    · linarith
    · rfl

end average_salary_of_managers_l665_665125


namespace hyperbola_center_l665_665254

theorem hyperbola_center :
  ∃ (h k : ℝ), h = 1 ∧ k = -2 ∧
  (∀ x y : ℝ, (4 * y + 8)^2 / 7^2 - (5 * x - 5)^2 / 3^2 = 1 →
  (x = h ∨ y = k)) :=
by
  use 1
  use -2
  split
  sorry
  split
  sorry
  intros x y h_eq
  split
  intro hyp_x
  sorry

end hyperbola_center_l665_665254


namespace interval_of_decrease_l665_665816

def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) - cos (2 * x)
def g (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6)

theorem interval_of_decrease (x : ℝ) :
  g(x) ≤ |g(π / 6)| ∀ x ∈ ℝ → ∃ k : ℤ, x ∈ set.Icc (k * π + π / 12) (k * π + 7 * π / 12) :=
sorry

end interval_of_decrease_l665_665816


namespace min_value_reciprocal_sum_l665_665286

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  (a = 1 ∧ b = 1) → (1 / a + 1 / b = 2) := by
  intros h
  sorry

end min_value_reciprocal_sum_l665_665286


namespace cube_volume_from_surface_area_l665_665880

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l665_665880


namespace propositions_verification_l665_665661

noncomputable def f (x : ℝ) : ℝ := 2 / x + log x

theorem propositions_verification :
  (∃ xx : ℝ, xx = 2 ∧ ∀ y : ℝ, f' f xx = 0) ∧
  ¬ (∃ xx : ℝ, f xx = 0 ∧ xx > 0) ∧
  ¬ (∃ k : ℝ, k > 0 ∧ ∀ x : ℝ, x > 0 → f x > k * x) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ < x₂ → f x₁ = f x₂ → x₁ + x₂ > 4) :=
sorry

end propositions_verification_l665_665661


namespace probability_of_exactly_three_positives_l665_665736

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_of_exactly_three_positives :
  let p := 2/5
  let n := 7
  let k := 3
  let positive_prob := p^k
  let negative_prob := (1 - p)^(n - k)
  let binomial_coefficient := choose n k
  binomial_coefficient * positive_prob * negative_prob = 22680/78125 := 
by
  sorry

end probability_of_exactly_three_positives_l665_665736


namespace complementary_angle_l665_665330

theorem complementary_angle (angle_deg : ℕ) (angle_min : ℕ) 
  (h1 : angle_deg = 37) (h2 : angle_min = 38) : 
  exists (comp_deg : ℕ) (comp_min : ℕ), comp_deg = 52 ∧ comp_min = 22 :=
by
  sorry

end complementary_angle_l665_665330


namespace solve_sine_equation_l665_665794

theorem solve_sine_equation :
  ∀ (x : ℝ), 
    (∃ (k : ℤ), 
       (x = k * 360) ∨ (x = 90 + k * 360)) ↔ 
    (sin (x + 15) + sin (x + 45) + sin (x + 75) = sin 15 + sin 45 + sin 75) := 
by
  sorry

end solve_sine_equation_l665_665794


namespace lucy_flour_used_l665_665399

theorem lucy_flour_used
  (initial_flour : ℕ := 500)
  (final_flour : ℕ := 130)
  (flour_needed_to_buy : ℤ := 370)
  (used_flour : ℕ) :
  initial_flour - used_flour = 2 * final_flour → used_flour = 240 :=
by
  sorry

end lucy_flour_used_l665_665399


namespace calculate_tan_cofunction_identity_l665_665646

variable {θ : ℝ}

theorem calculate_tan_cofunction_identity 
  (h1 : sin θ = 1 / 3)
  (h2 : θ ∈ set.Ioo (Real.pi / 2) Real.pi) :
  Real.tan (3 * Real.pi / 2 + θ) = 2 * Real.sqrt 2 := sorry

end calculate_tan_cofunction_identity_l665_665646


namespace correct_propositions_count_l665_665555

theorem correct_propositions_count :
  let p1 := false,  -- ① is incorrect
      p2 := false,  -- ② is incorrect
      p3 := false,  -- ③ is incorrect
      p4 := true,   -- ④ is correct
      p5 := false   -- ⑤ is incorrect
  in (if p1 then 1 else 0) + (if p2 then 1 else 0) + (if p3 then 1 else 0) + (if p4 then 1 else 0) + (if p5 then 1 else 0) = 1 := 
by {
  sorry
}

end correct_propositions_count_l665_665555


namespace profit_when_sold_at_double_price_l665_665518

-- Define the problem parameters

-- Assume cost price (CP)
def CP : ℕ := 100

-- Define initial selling price (SP) with 50% profit
def SP : ℕ := CP + (CP / 2)

-- Define new selling price when sold at double the initial selling price
def SP2 : ℕ := 2 * SP

-- Define profit when sold at SP2
def profit : ℕ := SP2 - CP

-- Define the percentage profit
def profit_percentage : ℕ := (profit * 100) / CP

-- The proof goal: if selling at double the price, percentage profit is 200%
theorem profit_when_sold_at_double_price : profit_percentage = 200 :=
by {sorry}

end profit_when_sold_at_double_price_l665_665518


namespace exists_x_for_ax2_plus_2x_plus_a_lt_0_l665_665698

theorem exists_x_for_ax2_plus_2x_plus_a_lt_0 (a : ℝ) : (∃ x : ℝ, a * x^2 + 2 * x + a < 0) ↔ a < 1 :=
by
  sorry

end exists_x_for_ax2_plus_2x_plus_a_lt_0_l665_665698


namespace length_of_BE_in_triangle_A_BE_and_parallelogram_ABCD_l665_665347

theorem length_of_BE_in_triangle_A_BE_and_parallelogram_ABCD :
  ∀ (AB BC BE : ℝ) (angle_ABC : ℝ), 
    4.5 = AB ∧ 6 = BC ∧ 60 = angle_ABC ∧ 
    (2.25 * Math.sqrt 3 * BE) = 13.5 * Math.sqrt 3 → 
    BE = 6 :=
by
  intros AB BC BE angle_ABC,
  assume h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  have : 2.25 * Math.sqrt 3 * BE = 13.5 * Math.sqrt 3 := h6,
  sorry

end length_of_BE_in_triangle_A_BE_and_parallelogram_ABCD_l665_665347


namespace find_product_l665_665800

theorem find_product (a b c d : ℝ) 
  (h_avg : (a + b + c + d) / 4 = 7.1)
  (h_rel : 2.5 * a = b - 1.2 ∧ b - 1.2 = c + 4.8 ∧ c + 4.8 = 0.25 * d) :
  a * b * c * d = 49.6 := 
sorry

end find_product_l665_665800


namespace cube_volume_of_surface_area_l665_665976

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l665_665976


namespace cube_volume_l665_665892

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l665_665892


namespace cube_volume_from_surface_area_l665_665917

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l665_665917


namespace max_value_on_interval_l665_665451

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_value_on_interval : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → f x ≤ 5 :=
by
  sorry

end max_value_on_interval_l665_665451


namespace chocolate_syrup_amount_l665_665135

theorem chocolate_syrup_amount (x : ℝ) (H1 : 2 * x + 6 = 14) : x = 4 :=
by
  sorry

end chocolate_syrup_amount_l665_665135


namespace largest_real_part_l665_665758

noncomputable def largest_real_part_of_sum (z w : ℂ) (hz : abs z = 1) (hw : abs w = 1) 
  (hzw : z * conj w + conj z * w = 2) : ℝ := 2

theorem largest_real_part (z w : ℂ) (hz : abs z = 1) (hw : abs w = 1) 
  (hzw : z * conj w + conj z * w = 2) : real_part (z + w) ≤ largest_real_part_of_sum z w hz hw hzw :=
by sorry

end largest_real_part_l665_665758


namespace roots_of_polynomial_l665_665235

def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

theorem roots_of_polynomial :
  {x : ℝ | P x = 0} = {1, -1, 3} := 
sorry

end roots_of_polynomial_l665_665235


namespace original_cuboid_volume_l665_665127

-- Define the conditions
def original_cuboid_height (h : ℝ) : Prop :=
  let l := h + 1 in
  let surface_area_increase := 24 in
  6 * h == 6 * (h + 1) + surface_area_increase

-- Define the volume calculation
def cuboid_volume (l w h : ℝ) : ℝ := l * w * h

-- Main theorem to prove
theorem original_cuboid_volume : ∃ h : ℝ, h = 5 → ∃ l : ℝ, l = 6 → cuboid_volume l l h = 180 :=
by 
  use 5
  use 6
  sorry

end original_cuboid_volume_l665_665127


namespace sam_walking_speed_l665_665614

variable (s : ℝ)
variable (t : ℝ)
variable (fred_speed : ℝ := 2)
variable (sam_distance : ℝ := 25)
variable (total_distance : ℝ := 35)

theorem sam_walking_speed :
  (total_distance - sam_distance) = fred_speed * t ∧
  sam_distance = s * t →
  s = 5 := 
by
  intros
  sorry

end sam_walking_speed_l665_665614


namespace count_even_three_digit_numbers_l665_665082

theorem count_even_three_digit_numbers : 
  let digits := {0, 1, 2, 3, 4}
  let hundreds := {1, 2, 3}
  let units := {0, 2, 4}
  ∃ n : ℕ, 
  (n = 57) ∧ 
  ∀ d1 d2 d3 : ℕ, 
  d1 ∈ hundreds → d2 ∈ digits → d3 ∈ units → 
  (d1 ≠ d2 ∨ d1 ≠ d3 ∨ d2 ≠ d3) →
  (100 * d1 + 10 * d2 + d3) < 400 → 
  even (100 * d1 + 10 * d2 + d3) → 
  (100 * d1 + 10 * d2 + d3) < 400 *→ 
  ∃ k, k < 1000 ∧ k = 100 * d1 + 10 * d2 + d3 := 
by 
{
  sorry
}

end count_even_three_digit_numbers_l665_665082


namespace jameson_badminton_medals_l665_665369

theorem jameson_badminton_medals (total_medals track_medals : ℕ) (swimming_medals : ℕ) :
  total_medals = 20 →
  track_medals = 5 →
  swimming_medals = 2 * track_medals →
  total_medals - (track_medals + swimming_medals) = 5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end jameson_badminton_medals_l665_665369


namespace alice_probability_multiple_of_4_l665_665549

noncomputable def probability_one_multiple_of_4 (choices : ℕ) : ℚ :=
  let p_not_multiple_of_4 : ℚ := 45 / 60
  let p_all_not_multiple_of_4 : ℚ := p_not_multiple_of_4 ^ choices
  1 - p_all_not_multiple_of_4

theorem alice_probability_multiple_of_4 :
  probability_one_multiple_of_4 3 = 37 / 64 :=
by
  sorry

end alice_probability_multiple_of_4_l665_665549


namespace floor_1000_cos_P_l665_665713

variables {P Q R S : Type} [InnerProductSpace ℝ P] [InnerProductSpace ℝ Q] [InnerProductSpace ℝ R] [InnerProductSpace ℝ S]
variables (cos : ℝ) (PR QS : ℝ)

-- Conditions given in the problem
def convex_quadrilateral (PQRS : Prop) : Prop := sorry
def angles_congruent (angle_P angle_R : ℝ) : Prop := angle_P = angle_R
def sides_equal := (PQ RS : ℝ) (PQ = 200 ∧ RS = 200)
def sides_unequal := PR ≠ QS
def perimeter := PQ + RS + PR + QS = 680

-- Lean statement to prove the final answer 
theorem floor_1000_cos_P : 
  convex_quadrilateral PQRS →
  angles_congruent P R →
  sides_equal PQ RS →
  sides_unequal PR QS →
  perimeter PQ RS PR QS →
  ∃ (cos : ℝ), ℝ.floor (1000 * cos P) = 700 := 
by 
  sorry

end floor_1000_cos_P_l665_665713


namespace evaluate_expression_l665_665477

theorem evaluate_expression :
  (∏ i in Finset.range 10, (i + 1)) / (∑ i in Finset.range 10, (i + 1) ^ 2) = 9428 := by
  sorry

end evaluate_expression_l665_665477


namespace graph_translation_right_eq_abs_l665_665075

def translate_right (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x, f (x - a)

theorem graph_translation_right_eq_abs :
  ∃ f : ℝ → ℝ, translate_right f 2 = abs := sorry

end graph_translation_right_eq_abs_l665_665075


namespace decrypt_plaintext_after_ciphertext_of_four_l665_665846

def encrypt (x : ℕ) (a : ℝ) : ℝ := log a (x + 2)

theorem decrypt_plaintext_after_ciphertext_of_four :
  ∃ x : ℕ, let a := 2 in
  encrypt 6 a = 3 ∧ log a (x + 2) = 4 → x = 14 :=
  by
    sorry

end decrypt_plaintext_after_ciphertext_of_four_l665_665846


namespace charge_per_block_l665_665738

noncomputable def family_vacation_cost : ℝ := 1000
noncomputable def family_members : ℝ := 5
noncomputable def walk_start_fee : ℝ := 2
noncomputable def dogs_walked : ℝ := 20
noncomputable def total_blocks : ℝ := 128

theorem charge_per_block : 
  (family_vacation_cost / family_members) = 200 →
  (dogs_walked * walk_start_fee) = 40 →
  ((family_vacation_cost / family_members) - (dogs_walked * walk_start_fee)) = 160 →
  (((family_vacation_cost / family_members) - (dogs_walked * walk_start_fee)) / total_blocks) = 1.25 :=
by intros h1 h2 h3; sorry

end charge_per_block_l665_665738


namespace area_of_regionM_l665_665353

/-
Define the conditions as separate predicates in Lean.
-/

def cond1 (x y : ℝ) : Prop := y - x ≥ abs (x + y)

def cond2 (x y : ℝ) : Prop := (x^2 + 8*x + y^2 + 6*y) / (2*y - x - 8) ≤ 0

/-
Define region \( M \) by combining the conditions.
-/

def regionM (x y : ℝ) : Prop := cond1 x y ∧ cond2 x y

/-
Define the main theorem to compute the area of the region \( M \).
-/

theorem area_of_regionM : 
  ∀ x y : ℝ, (regionM x y) → (calculateAreaOfM) := sorry

/-
A placeholder definition to calculate the area of M. 
-/

noncomputable def calculateAreaOfM : ℝ := 8

end area_of_regionM_l665_665353


namespace initial_amount_l665_665415

-- Define the conditions
def cost_small_glass : ℕ := 3
def cost_large_glass : ℕ := 5
def num_small_glasses : ℕ := 8
def num_large_glasses : ℕ := 5
def change_left : ℕ := 1

-- Define the pieces based on conditions
def total_cost_small_glasses : ℕ := num_small_glasses * cost_small_glass
def total_cost_large_glasses : ℕ := num_large_glasses * cost_large_glass
def total_cost_glasses : ℕ := total_cost_small_glasses + total_cost_large_glasses

-- The theorem we need to prove
theorem initial_amount (h1 : total_cost_small_glasses = 24)
                       (h2 : total_cost_large_glasses = 25)
                       (h3 : total_cost_glasses = 49) : total_cost_glasses + change_left = 50 :=
by sorry

end initial_amount_l665_665415


namespace parallel_vectors_tan_eq_one_dot_product_inequality_l665_665315

-- Define the vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
def vector_b : ℝ × ℝ := (1, 1)

-- Problem 1: Prove that if a is parallel to b, then tan x = 1
theorem parallel_vectors_tan_eq_one (x : ℝ) :
  vector_a x = (k * 1, k * 1) → Real.tan x = 1 := sorry

-- Define the dot product f(x) and its inequality condition
def dot_product (x : ℝ) : ℝ := (vector_a x).1 * vector_b.1 + (vector_a x).2 * vector_b.2
def f (x : ℝ) : ℝ := dot_product x

-- Problem 2: Prove the range of m
theorem dot_product_inequality (m : ℝ) :
  (∀ x : ℝ, f x > m) → m < -Real.sqrt 2 := sorry

end parallel_vectors_tan_eq_one_dot_product_inequality_l665_665315


namespace cyclic_quad_product_equality_l665_665852

theorem cyclic_quad_product_equality
  (A B C D P : Point)
  (H1 : Circle A B C D)
  (H2 : TangentAt P A)
  (H3 : TangentAt P C)
  (H4 : P = meet (TangentLineAt A) (TangentLineAt C))
  (H5 : meet (Line P B) Circle = D) :
  dist A B * dist C D = dist B C * dist D A :=
by 
  sorry

end cyclic_quad_product_equality_l665_665852


namespace shaded_area_is_24_or_26_l665_665180

theorem shaded_area_is_24_or_26 :
    ∀ (legs_length : ℕ) (partition_count : ℕ) (shaded_fraction : ℚ),
    isosceles_right_triangle legs_length →
    legs_length = 10 →
    partition_count = 25 →
    shaded_fraction = 1/2 →
    (let total_area := (1/2 : ℚ) * legs_length * legs_length,
         small_triangle_area := total_area / partition_count,
         shaded_area := (small_triangle_area * (⟦𝑛 in {12, 13} : ℕ | 𝑛 * 2 * shaded_fraction = partition_count * shaded_fraction⟧) ))
    in shaded_area = 24 ∨ shaded_area = 26 :=
by sorry

end shaded_area_is_24_or_26_l665_665180


namespace length_of_24_l665_665607

def length_of_integer (k : ℕ) := (1 : ℕ) → ℕ 

theorem length_of_24 : length_of_integer 24 = 4 := 
by
  sorry

end length_of_24_l665_665607


namespace find_line_equation_l665_665689

theorem find_line_equation 
  (A : ℝ × ℝ) (hA : A = (-2, -3)) 
  (h_perpendicular : ∃ k b : ℝ, ∀ x y, 3 * x + 4 * y - 3 = 0 → k * x + y = b) :
  ∃ k' b' : ℝ, (∀ x y, k' * x + y = b' → y = (4 / 3) * x + 1 / 3) ∧ (k' = 4 ∧ b' = -1) :=
by
  sorry

end find_line_equation_l665_665689


namespace mass_of_tetrahedron_is_2_l665_665565

noncomputable def mass_of_tetrahedron : ℝ :=
  ∫ x in 0..10, ∫ y in 0..(8 * (1 - x / 10)), ∫ z in 0..(3 * (1 - x / 10 - y / 8)), 
    (1 + x / 10 + y / 8 + z / 3) ^ (-6) 

theorem mass_of_tetrahedron_is_2 : mass_of_tetrahedron = 2 := 
  sorry

end mass_of_tetrahedron_is_2_l665_665565


namespace ratio_of_average_speed_l665_665504

noncomputable def boat_speed_still_water : ℝ := 24
noncomputable def current_speed : ℝ := 6
noncomputable def distance_downstream : ℝ := 3
noncomputable def distance_upstream : ℝ := 3

def average_speed_round_trip (d1 d2 v1 v2 : ℝ) : ℝ :=
  let downstream_speed := v1 + v2
  let upstream_speed := v1 - v2
  let time_downstream := d1 / downstream_speed
  let time_upstream := d2 / upstream_speed
  let total_time := time_downstream + time_upstream
  let total_distance := d1 + d2
  total_distance / total_time

theorem ratio_of_average_speed :
  average_speed_round_trip distance_downstream distance_upstream boat_speed_still_water current_speed / boat_speed_still_water = 15 / 16 := by
  sorry

end ratio_of_average_speed_l665_665504


namespace distinct_terms_in_expansion_l665_665189

theorem distinct_terms_in_expansion : 
  let f := (λ (a b : ℤ), (6 * a ^ 2 + 5 * a * b - 6 * b ^ 2) ^ 6) in
  ∀ a b : ℤ, ∃ n : ℕ, n = 7 ∧ (number_of_distinct_terms (f a b) = n) :=
by
  sorry

end distinct_terms_in_expansion_l665_665189


namespace concurrency_of_midline_altitude_intersections_l665_665769

theorem concurrency_of_midline_altitude_intersections
  (a b c : ℝ)
  (triangle : Triangle ℝ)
  (F1 F2 F3 : Point ℝ)
  (m_a m_b m_c : Line ℝ)
  (M_a M_b M_c : Point ℝ)
  (H : Point ℝ) -- Assuming H as orthocenter

  (h1 : midpoint (triangle.side1) = F1)
  (h2 : midpoint (triangle.side2) = F2)
  (h3 : midpoint (triangle.side3) = F3)
  (h4 : is_parallel (midline_parallel_to_side1 triangle) triangle.altitude1)
  (h5 : is_parallel (midline_parallel_to_side2 triangle) triangle.altitude2)
  (h6 : is_parallel (midline_parallel_to_side3 triangle) triangle.altitude3)
  (h7 : intersection (midline_parallel_to_side1 triangle) triangle.altitude1 = M_a)
  (h8 : intersection (midline_parallel_to_side2 triangle) triangle.altitude2 = M_b)
  (h9 : intersection (midline_parallel_to_side3 triangle) triangle.altitude3 = M_c) :
  
  concurrent_lines (Line.mk F1 M_a) (Line.mk F2 M_b) (Line.mk F3 M_c) :=
  sorry

end concurrency_of_midline_altitude_intersections_l665_665769


namespace travel_sequences_count_l665_665788

-- Definitions based on the given conditions
def natural_scenic_spots : Finset String := {"A", "B", "C"}
def cultural_and_historical_spots : Finset String := {"a", "b", "c"}
def all_spots : Finset String := natural_scenic_spots ∪ cultural_and_historical_spots

def is_cultural (spot : String) : Prop := spot = "a" ∨ spot = "b" ∨ spot = "c"

-- Given conditions as definitions
def valid_sequence (s : List String) : Prop :=
  s.length = 4 ∧
  s.any (λ x => x = "A") ∧
  (¬ s.head? = some "A") ∧
  is_cultural (s.getLast (by simp [s.length_pos_of_ne_nil]))

-- The proof statement
theorem travel_sequences_count : 
  (Finset.filter (λ s => valid_sequence s) (all_spots.powerset.filter (λ s => s.card = 4))).card = 144 := 
  sorry

end travel_sequences_count_l665_665788


namespace angle_BOP_eq_angle_COQ_l665_665507

variables {A B C P Q O : Point}
variables {triangle : Triangle A B C}
variables {inscribed_circle : Circle O}
variables (P_on_AB : OnSegment P (segment A B))
variables (Q_on_extension_AC : OnExtension Q (segment A C) C)
variables (PQ_tangent : TangentTo PQ inscribed_circle at a_point)

-- The goal is to prove that ∠BOP = ∠COQ
theorem angle_BOP_eq_angle_COQ
  (h1 : IsInscribedCircle triangle inscribed_circle)
  (h2 : OnSegment P (segment A B))
  (h3 : OnExtension Q (segment A C) C)
  (h4 : TangentTo PQ inscribed_circle at a_point) :
  angle B O P = angle C O Q :=
sorry

end angle_BOP_eq_angle_COQ_l665_665507


namespace gcd_step_equation_l665_665854

theorem gcd_step_equation (a b : ℕ) (h_a : a = 98) (h_b : b = 63) :
  ∃ d e f, a - b = d ∧ b - d = e ∧ d - e = f ∧ e - f = 21 :=
by
  -- Insert the full conditionals for successive subtraction steps
  have h₁ : a - b = 35, from
    by rw [h_a, h_b]; norm_num,
  have h₂ : b - (a - b) = 28, from
    by rw [h₁, h_b]; norm_num,
  have h₃ : (a - b) - (b - (a - b)) = 7, from
    by rw [h₁, h₂]; norm_num,
  have h₄ : (b - (a - b)) - ((a - b) - (b - (a - b))) = 21, from
    by rw [h₂, h₃]; norm_num,
  use [(a - b), (b - (a - b)), ((a - b) - (b - (a - b))), 21],
  refine ⟨h₁, h₂, h₃, h₄⟩

end gcd_step_equation_l665_665854


namespace perpendicular_line_to_plane_l665_665695

section PerpendicularLineToPlane

variables (u : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ)

def is_scalar_multiple (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2, k * b.3)

theorem perpendicular_line_to_plane : 
    is_scalar_multiple (1, -2, 0) (-2, -4, 0) :=
sorry

end PerpendicularLineToPlane

end perpendicular_line_to_plane_l665_665695


namespace cube_volume_from_surface_area_l665_665869

theorem cube_volume_from_surface_area (SA : ℝ) (h : SA = 864) : exists (V : ℝ), V = 1728 :=
by
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  have h1 : s ^ 2 = 144 := by sorry
  have h2 : s = 12 := by sorry
  use V
  rw h2
  exact calc
    V = 12 ^ 3 : by rw h2
    ... = 1728 : by norm_num


end cube_volume_from_surface_area_l665_665869


namespace max_cylinder_radius_fits_l665_665102

noncomputable def crate_max_radius : ℝ := 2.5

def crate_dimensions : ℝ × ℝ × ℝ := (5, 8, 12)

def cylinder_fits_in_orientation (r h : ℝ) (dim1 dim2 dim3 : ℝ) : Prop :=
  (r * 2 ≤ dim1 ∧ r * 2 ≤ dim2 ∧ h ≤ dim3) ∧
  (r * 2 ≤ dim1 ∧ h ≤ dim2 ∧ r * 2 ≤ dim3) ∧
  (h ≤ dim1 ∧ r * 2 ≤ dim2 ∧ r * 2 ≤ dim3)

theorem max_cylinder_radius_fits :
  ∃ r : ℝ, r = crate_max_radius ∧ 
  ∀ (r' h : ℝ), r' ≤ r ∧ (cylinder_fits_in_orientation r' (12) 5 8 ∨ 
                           cylinder_fits_in_orientation r' (8) 5 12 ∨ 
                           cylinder_fits_in_orientation r' (5) 8 12) :=
sorry

end max_cylinder_radius_fits_l665_665102


namespace find_a_l665_665309

open Set
open Real

def A : Set ℝ := {-1, 1}
def B (a : ℝ) : Set ℝ := {x | a * x ^ 2 = 1}

theorem find_a (a : ℝ) (h : (A ∩ (B a)) = (B a)) : a = 1 :=
sorry

end find_a_l665_665309


namespace cube_volume_from_surface_area_l665_665918

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l665_665918


namespace third_range_is_correct_l665_665113

theorem third_range_is_correct (R1 R2_prime R3_prime : ℕ) (hR1 : R1 = 30) (hR2_prime : R2_prime = 18) (hR3_prime : R3_prime = 26) : 
  let R2 := R1 + R2_prime in
  let R3 := R1 + R3_prime in
  R3 = 56 := 
by {
  sorry
}

end third_range_is_correct_l665_665113


namespace min_trips_is_157_l665_665812

theorem min_trips_is_157 :
  ∃ x y : ℕ, 31 * x + 32 * y = 5000 ∧ x + y = 157 :=
sorry

end min_trips_is_157_l665_665812


namespace minimum_value_x_plus_y_l665_665648

theorem minimum_value_x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 2 * y = x * y) :
  x + y = 3 + 2 * Real.sqrt 2 :=
sorry

end minimum_value_x_plus_y_l665_665648


namespace range_of_t_l665_665381

variable (M : Set ℝ) (t : ℝ) (N : Set ℝ)
variable h1 : M = {x : ℝ | -4 < x ∧ x < 3}
variable h2 : N = {x : ℝ | t + 2 < x ∧ x < 2 * t - 1}
variable h3 : M ∩ N = N

theorem range_of_t : ∃ t : ℝ, (-∞ < t) ∧ (t ≤ 3) := sorry

end range_of_t_l665_665381


namespace quadrilateral_perimeter_l665_665473

noncomputable def EG (FH : ℝ) : ℝ := Real.sqrt ((FH + 5) ^ 2 + FH ^ 2)

theorem quadrilateral_perimeter 
  (EF FH GH : ℝ) 
  (h1 : EF = 12)
  (h2 : FH = 7)
  (h3 : GH = FH) :
  EF + FH + GH + EG FH = 26 + Real.sqrt 193 :=
by
  rw [h1, h2, h3]
  sorry

end quadrilateral_perimeter_l665_665473


namespace total_time_to_watch_movie_l665_665171

-- Define the conditions and the question
def uninterrupted_viewing_time : ℕ := 35 + 45 + 20
def rewinding_time : ℕ := 5 + 15
def total_time : ℕ := uninterrupted_viewing_time + rewinding_time

-- Lean statement of the proof problem
theorem total_time_to_watch_movie : total_time = 120 := by
  -- This is where the proof would go
  sorry

end total_time_to_watch_movie_l665_665171


namespace ellipse_equation_line_equation_rectangle_l665_665655

/-- Proving the equation of the given ellipse -/
theorem ellipse_equation
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c = 2)
  (h4 : c / a = Real.sqrt 6 / 3)
  (h5 : a^2 = b^2 + c^2) :
  (a = Real.sqrt 6 ∧ b = Real.sqrt 2 ∧ (∀ x y : ℝ, ((x^2 / a^2) + (y^2 / b^2) = 1) → (x^2 / 6 + y^2 / 2 = 1))) :=
sorry

/-- Proving the equation of line l when the quadrilateral MF1NF2 is a rectangle -/
theorem line_equation_rectangle 
  (k : ℝ)
  (h1 : k ≠ 0)
  (h2 : ∀ x y : ℝ, ((x + 3 * k * y = 0 ∧ x^2 + 3 * y^2 = 6) → 
     ((4 - (3 * k * (Real.sqrt (2 / (1 + 3 * k^2))))^2 - (Real.sqrt (2 / (1 + 3 * k^2)))^2 = 0) → 
      (k = Real.sqrt 3 / 3 ∨ k = -Real.sqrt 3 / 3)))) :
  ∀ x, y, (x, y) satisfies the equation of line l :
  (y = Real.sqrt 3 / 3 * (x - 2) ∨ y = -Real.sqrt 3 / 3 * (x - 2)) :=
sorry

end ellipse_equation_line_equation_rectangle_l665_665655


namespace tradesman_gain_l665_665544

-- Conditions: defrauding percentages for items A, B, and C, and equal spending
def defraud_percent_A_buy : ℝ := 0.30
def defraud_percent_A_sell : ℝ := 0.30
def defraud_percent_B_buy : ℝ := 0.20
def defraud_percent_B_sell : ℝ := 0.10
def defraud_percent_C_buy : ℝ := 0.10
def defraud_percent_C_sell : ℝ := 0.20
def spending_on_each_item : ℝ := 100.0

-- Function to calculate effective worth after defrauding during the purchase
def effective_worth (spending defraud_buy : ℝ) : ℝ :=
  spending / (1 - defraud_buy)

-- Function to calculate selling price after defrauding during the sell
def selling_price (worth defraud_sell : ℝ) : ℝ :=
  worth * (1 + defraud_sell)

-- Function to calculate gain for a single item
def gain (spending worth : ℝ) : ℝ :=
  worth - spending

-- Calculations for each item
def gain_A : ℝ :=
  let worth_A := effective_worth spending_on_each_item defraud_percent_A_buy
  let selling_price_A := selling_price worth_A defraud_percent_A_sell
  gain spending_on_each_item selling_price_A

def gain_B : ℝ :=
  let worth_B := effective_worth spending_on_each_item defraud_percent_B_buy
  let selling_price_B := selling_price worth_B defraud_percent_B_sell
  gain spending_on_each_item selling_price_B

def gain_C : ℝ :=
  let worth_C := effective_worth spending_on_each_item defraud_percent_C_buy
  let selling_price_C := selling_price worth_C defraud_percent_C_sell
  gain spending_on_each_item selling_price_C

-- Total gain
def total_gain : ℝ :=
  gain_A + gain_B + gain_C

-- Total outlay
def total_outlay : ℝ :=
  3 * spending_on_each_item

-- Overall percentage gain
def overall_percentage_gain : ℝ :=
  (total_gain / total_outlay) * 100

-- The target theorem to be proved
theorem tradesman_gain :
  overall_percentage_gain = 52.18 :=
by
  sorry

end tradesman_gain_l665_665544


namespace adding_books_multiplying_books_l665_665779

-- Define the conditions
def num_books_first_shelf : ℕ := 4
def num_books_second_shelf : ℕ := 5
def num_books_third_shelf : ℕ := 6

-- Define the first question and prove its correctness
theorem adding_books :
  num_books_first_shelf + num_books_second_shelf + num_books_third_shelf = 15 :=
by
  -- The proof steps would go here, but they are replaced with sorry for now
  sorry

-- Define the second question and prove its correctness
theorem multiplying_books :
  num_books_first_shelf * num_books_second_shelf * num_books_third_shelf = 120 :=
by
  -- The proof steps would go here, but they are replaced with sorry for now
  sorry

end adding_books_multiplying_books_l665_665779


namespace parity_f_increasing_f_l665_665666

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := x + (1 / x)

-- Confirming f(1) = 2 implies a = -1
def a : ℝ := -1

-- Parity proof: f(-x) = -f(x)
theorem parity_f : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  -- Proof needed
  sorry

-- Monotonicity proof: f is increasing on (1, ∞)
theorem increasing_f : ∀ x1 x2 : ℝ, 1 < x1 → 1 < x2 → x1 < x2 → f x1 < f x2 := by
  intro x1 x2 hx1 hx2 hlt
  -- Proof needed
  sorry

end parity_f_increasing_f_l665_665666


namespace country_partition_l665_665705

variable (V : Type) [Fintype V] -- finite type for the vertices
variable (E : V → V → Prop) -- predicate representing edges
variable (n : ℕ) (k : ℕ) (hv : Fintype.card V = 2001) 
  -- condition that there are exactly 2001 vertices
variable (hE : ∀ v : V, ∃ u : V, E v u) 
  -- every vertex has at least one road leading out of it
variable (hE_max : ∀ v : V, ¬ ∀ u : V, E v u) 
  -- no vertex is connected to all other vertices
variable [DecidableRel E] -- edges relation is decidable

def dominating_set (D : finset V) : Prop :=
  ∀ v, v ∉ D → ∃ u ∈ D, E v u
  -- definition of a dominating set

variable (hD : ∀ D : finset V, dominating_set V E D → D.card ≥ k) 
  -- every dominating set contains at least k elements

theorem country_partition : ∃ (P : finpartition V), 
  P.parts.card = 2001 - k ∧ ∀ p : finset V, p ∈ P.parts → ∀ v w : V, v ≠ w ∧ v ∈ p ∧ w ∈ p → ¬ E v w := by
  sorry -- proof goes here

end country_partition_l665_665705


namespace least_pieces_l665_665774

theorem least_pieces (miran junga minsu : ℕ) 
  (h1 : miran = 6) (h2 : junga = 13) (h3 : minsu = 10) : 
  miran ≤ junga ∧ miran ≤ minsu :=
by 
  rw [h1, h2, h3]
  exact ⟨Nat.le_of_eq (by rfl), Nat.le_of_eq (by rfl)⟩

end least_pieces_l665_665774


namespace price_drops_to_min_at_550_price_function_expression_profit_at_specific_orders_l665_665130

-- The conditions given in the problem
def cost_per_component := 40
def initial_factory_price := 60
def price_decrease_rate := 0.02
def min_factory_price := 51

-- Function for the actual factory price based on the order quantity
def factory_price (x : ℕ) : ℝ :=
  if x ≤ 100 then initial_factory_price
  else if x < 550 then 62 - (x / 50)
  else min_factory_price

-- Function for the factory profit based on the order quantity
def factory_profit (x : ℕ) : ℝ :=
  match x with
  | x if 0 < x && x ≤ 100 => 20 * x
  | x if 100 < x && x < 500 => 22 * x - (x^2 / 50)
  | 500 => 6000
  | 1000 => 11000
  | _ => sorry -- this handles other general cases, leaving it unspecified

-- Proof statements
theorem price_drops_to_min_at_550 :
  factory_price 550 = 51 := by sorry

theorem price_function_expression :
  ∀ (x : ℕ), 
  (x < 100 → factory_price x = 60) ∧
  (100 < x ∧ x < 550 → factory_price x = 62 - (x / 50)) ∧
  (x ≥ 550 → factory_price x = 51) := by sorry

theorem profit_at_specific_orders :
  factory_profit 500 = 6000 ∧
  factory_profit 1000 = 11000 := by sorry

end price_drops_to_min_at_550_price_function_expression_profit_at_specific_orders_l665_665130


namespace cube_volume_from_surface_area_l665_665923

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l665_665923


namespace quadratic_inequality_solution_l665_665394

theorem quadratic_inequality_solution (a : ℝ) (h₀ : 1 < a) (h₁ : a < 2) :
  {x : ℝ | x^2 - (a^2 + 3a + 2) * x + 3 * a * (a^2 + 2) < 0} = 
  set.Ioo (a^2 + 2) (3 * a) := 
by 
  sorry

end quadratic_inequality_solution_l665_665394


namespace linear_function_passes_through_1_0_l665_665517

theorem linear_function_passes_through_1_0 (k : ℝ) (h : k ≠ 0) (h_pass : 4 = k * (-1) - k) :
  ∃ x y, (x = 1 ∧ y = 0 ∧ y = k * x - k) :=
by
  use 1
  use 0
  have h_k : k = -2 := by
    solve1 {
      rw [mul_neg, add_right_neg] at h_pass,
      linarith,
    }
  rw [h_k] at *
  exact ⟨rfl, rfl, by linarith⟩

end linear_function_passes_through_1_0_l665_665517


namespace cube_volume_from_surface_area_l665_665957

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 864) : ∃ V : ℝ, V = 1728 :=
by
  -- Assume surface area formula S = 6s^2, solve steps skipped and go directly to conclusion
  let s := real.sqrt (864 / 6)
  let V := s ^ 3
  use V
  sorry

end cube_volume_from_surface_area_l665_665957


namespace zero_of_transformed_function_l665_665678

variables (f : ℝ → ℝ)
variable (x0 : ℝ)

-- Condition: f is an odd function
def is_odd_function : Prop := ∀ x, f (-x) = -f x

-- Condition: x0 is a zero of y = f(x) + e^x
def is_zero_of_original_function : Prop := f x0 + Real.exp x0 = 0

-- The theorem we want to prove:
theorem zero_of_transformed_function (h1 : is_odd_function f) (h2 : is_zero_of_original_function f x0) :
  Real.exp (-x0) * f (-x0) - 1 = 0 :=
sorry

end zero_of_transformed_function_l665_665678


namespace friedas_reaches_boundary_in_3_hops_l665_665573

noncomputable def friedasProbability (gridSize : ℕ) : ℝ :=
  if gridSize = 4 then 1 else 0

theorem friedas_reaches_boundary_in_3_hops :
  ∀ (start : ℕ × ℕ),
    start = (2, 2) →
    friedasProbability 4 = 1 :=
by
  intros start h_start
  rw h_start
  simp [friedasProbability]
  sorry

end friedas_reaches_boundary_in_3_hops_l665_665573


namespace probability_neither_red_nor_purple_l665_665117

theorem probability_neither_red_nor_purple 
  (total_balls : ℕ) 
  (white_balls : ℕ) 
  (green_balls : ℕ) 
  (yellow_balls : ℕ) 
  (red_balls : ℕ) 
  (purple_balls : ℕ) 
  (h_total_balls : total_balls = 60)
  (h_white_balls : white_balls = 22)
  (h_green_balls : green_balls = 18)
  (h_yellow_balls : yellow_balls = 17)
  (h_red_balls : red_balls = 3)
  (h_purple_balls : purple_balls = 1):
  let non_red_purple_balls := total_balls - (red_balls + purple_balls) in
  let probability := (non_red_purple_balls : ℚ) / total_balls in
  probability = 14 / 15 := 
by {
  -- Insert the proof here
  sorry
}

end probability_neither_red_nor_purple_l665_665117


namespace A_inter_B_empty_iff_A_union_B_eq_B_iff_l665_665311

open Set

variable (a x : ℝ)

def A (a : ℝ) : Set ℝ := {x | 1 - a ≤ x ∧ x ≤ 1 + a}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem A_inter_B_empty_iff {a : ℝ} :
  (A a ∩ B = ∅) ↔ 0 ≤ a ∧ a ≤ 4 :=
by 
  sorry

theorem A_union_B_eq_B_iff {a : ℝ} :
  (A a ∪ B = B) ↔ a < -4 :=
by
  sorry

end A_inter_B_empty_iff_A_union_B_eq_B_iff_l665_665311


namespace simplify_fraction_subtraction_l665_665589

theorem simplify_fraction_subtraction :
  (5 / 15 : ℚ) - (2 / 45) = 13 / 45 :=
by
  -- (The proof will go here)
  sorry

end simplify_fraction_subtraction_l665_665589


namespace girl_scout_cookie_sales_l665_665514

theorem girl_scout_cookie_sales :
  ∃ C P : ℝ, C + P = 1585 ∧ 1.25 * C + 0.75 * P = 1586.25 ∧ P = 790 :=
by
  sorry

end girl_scout_cookie_sales_l665_665514


namespace amaya_total_time_l665_665174

-- Define the times as per the conditions
def first_segment : Nat := 35 + 5
def second_segment : Nat := 45 + 15
def third_segment : Nat := 20

-- Define the total time by summing up all segments
def total_time : Nat := first_segment + second_segment + third_segment

-- The theorem to prove
theorem amaya_total_time : total_time = 120 := by
  -- Let's explicitly state the expected result here
  have h1 : first_segment = 40 := rfl
  have h2 : second_segment = 60 := rfl
  have h3 : third_segment = 20 := rfl
  have h_sum : total_time = 40 + 60 + 20 := by
    rw [h1, h2, h3]
  simp [total_time, h_sum]
  -- Finally, the result is 120
  exact rfl

end amaya_total_time_l665_665174


namespace inverse_function_of_f_is_g_l665_665034

   -- Define the original function
   def f (x : ℝ) : ℝ := 2^x + 3

   -- Define the proposed inverse function
   def g (y : ℝ) : ℝ := log y / log 2 -- Lean uses natural logarithm by default, hence the transformation

   -- The main statement to be proved is given below:
   theorem inverse_function_of_f_is_g (x : ℝ) (hx : x > 3) : f (g x) = x :=
   by
     sorry
   
end inverse_function_of_f_is_g_l665_665034


namespace compute_value_l665_665755

theorem compute_value {p q : ℝ} (h1 : 3 * p^2 - 5 * p - 8 = 0) (h2 : 3 * q^2 - 5 * q - 8 = 0) :
  (5 * p^3 - 5 * q^3) / (p - q) = 245 / 9 :=
by
  sorry

end compute_value_l665_665755


namespace gcd_leq_floor_cuberoot_sum_l665_665398

theorem gcd_leq_floor_cuberoot_sum
  (a b c : ℕ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_A : ∃ (A : ℕ), A = (a^2 + 1) / (b * c) + (b^2 + 1) / (c * a) + (c^2 + 1) / (a * b) ∧ A % 1 = 0 ) :
  Nat.gcd (Nat.gcd a b) c ≤ (Nat.floor (Real.cbrt (a + b + c))) :=
sorry

end gcd_leq_floor_cuberoot_sum_l665_665398


namespace medicine_dosage_per_kg_l665_665128

theorem medicine_dosage_per_kg :
  ∀ (child_weight parts dose_per_part total_dose dose_per_kg : ℕ),
    (child_weight = 30) →
    (parts = 3) →
    (dose_per_part = 50) →
    (total_dose = parts * dose_per_part) →
    (dose_per_kg = total_dose / child_weight) →
    dose_per_kg = 5 :=
by
  intros child_weight parts dose_per_part total_dose dose_per_kg
  intros h1 h2 h3 h4 h5
  sorry

end medicine_dosage_per_kg_l665_665128


namespace three_of_a_kind_probability_l665_665793

noncomputable def probability_three_of_a_kind_after_reroll (D : Finset (Fin 6)) 
  (no_three_of_a_kind : ∀ (x ∈ D), D.count x ≤ 2) 
  (two_pairs : ∃ (x y : Fin 6), x ≠ y ∧ D.count x = 2 ∧ D.count y = 2) : ℚ :=
if two_rerolls : (Finset.univ.image (λ (x : Fin 6), (D ∪ (Finset.singleton x)).count x)) = {1, 1} then 
  2/3 else 0

-- The main theorem assertion, no need for the proof here.
theorem three_of_a_kind_probability :
  ∀ (D : Finset (Fin 6)), 
  (∀ (x ∈ D), D.count x ≤ 2) → 
  (∃ (x y : Fin 6), x ≠ y ∧ D.count x = 2 ∧ D.count y = 2) →
  probability_three_of_a_kind_after_reroll D sorry = 2/3 :=
sorry

end three_of_a_kind_probability_l665_665793


namespace part1_part2a_part2b_l665_665617

-- Definitions and conditions
def vector_a : ℝ × ℝ := (1, -2)
def vector_b : ℝ × ℝ := (-3, 2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def scalar_mul (k : ℝ) (u : ℝ × ℝ) : ℝ × ℝ := (k * u.1, k * u.2)
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def collinear (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- Proof statements

-- Part 1: Verify the dot product computation
theorem part1 : dot_product (vector_add vector_a vector_b) (vector_sub vector_a vector_b) = -8 := by
  sorry

-- Part 2a: Verify the value of k for parallel vectors
theorem part2a : collinear (vector_add (scalar_mul (1/3) vector_a) vector_b) (vector_sub vector_a (scalar_mul 3 vector_b)) := by
  sorry

-- Part 2b: Verify antiparallel direction
theorem part2b : collinear (vector_add (scalar_mul (1/3) vector_a) vector_b) (scalar_mul (-1) (vector_sub vector_a (scalar_mul 3 vector_b))) := by
  sorry

end part1_part2a_part2b_l665_665617


namespace f_2016_eq_cos_l665_665622

-- Conditions as definitions
def f_series : ℕ → (ℝ → ℝ)
| 0 := cos
| (n + 1) := (f_series n)'

-- The theorem to be proven, with the correct answer
theorem f_2016_eq_cos : f_series 2016 = cos :=
sorry

end f_2016_eq_cos_l665_665622


namespace induction_divisibility_l665_665853

theorem induction_divisibility (k x y : ℕ) (h : k > 0) :
  (x^(2*k-1) + y^(2*k-1)) ∣ (x + y) → 
  (x^(2*k+1) + y^(2*k+1)) ∣ (x + y) :=
sorry

end induction_divisibility_l665_665853


namespace polynomial_roots_l665_665244

theorem polynomial_roots :
  (∀ x, x^3 - 3 * x^2 - x + 3 = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) := 
by
  sorry

end polynomial_roots_l665_665244


namespace correct_units_l665_665231

-- Definitions based on problem conditions
def goose_egg_weight := 100
def ten_goose_eggs_weight := 1000
def xiaoming_weight := 28000
def xiaoming_height := 135
def small_rice_bag_weight := 5000
def rubber_eraser_weight := 25

-- Target units in grams and centimeters where applicable
axiom goose_egg_weight_unit : goose_egg_weight = 100 -- grams
axiom ten_goose_eggs_weight_unit : ten_goose_eggs_weight = 1000 -- grams 
axiom xiaoming_weight_unit : xiaoming_weight = 28000 -- grams
axiom xiaoming_height_unit : xiaoming_height = 135 -- centimeters
axiom small_rice_bag_weight_unit : small_rice_bag_weight = 5000 -- grams
axiom rubber_eraser_weight_unit : rubber_eraser_weight = 25 -- grams

-- Proving equivalence to expected answers
theorem correct_units :
  goose_egg_weight_unit ∧ 
  ten_goose_eggs_weight_unit ∧ 
  xiaoming_weight_unit ∧ 
  xiaoming_height_unit ∧ 
  small_rice_bag_weight_unit ∧ 
  rubber_eraser_weight_unit :=
by
  split; try { sorry }

end correct_units_l665_665231


namespace price_of_fruit_juice_l665_665534

theorem price_of_fruit_juice (F : ℝ)
  (Sandwich_price : ℝ := 2)
  (Hamburger_price : ℝ := 2)
  (Hotdog_price : ℝ := 1)
  (Selene_purchases : ℝ := 3 * Sandwich_price + F)
  (Tanya_purchases : ℝ := 2 * Hamburger_price + 2 * F)
  (Total_spent : Selene_purchases + Tanya_purchases = 16) :
  F = 2 :=
by
  sorry

end price_of_fruit_juice_l665_665534


namespace minimum_modulus_z_l665_665625

noncomputable def min_modulus_of_complex_root (z : ℂ) : ℝ :=
|z|

theorem minimum_modulus_z (z : ℂ)
  (h : ∃ x : ℝ, (x^2 - 2 * z * x + (3 / 4 : ℂ) + complex.I = 0 ∧ x.real)) : 
  min_modulus_of_complex_root z = 1 :=
sorry

end minimum_modulus_z_l665_665625


namespace total_buttons_l665_665543

-- Defining the given conditions
def green_buttons : ℕ := 90
def yellow_buttons : ℕ := green_buttons + 10
def blue_buttons : ℕ := green_buttons - 5

-- Stating the theorem to prove the total number of buttons
theorem total_buttons : green_buttons + yellow_buttons + blue_buttons = 275 :=
by 
  sorry

end total_buttons_l665_665543


namespace median_product_sum_l665_665819

-- Let's define the lengths of medians and distances from a point P to these medians
variables {s1 s2 s3 d1 d2 d3 : ℝ}

-- Define the conditions
def is_median_lengths (s1 s2 s3 : ℝ) : Prop := 
  ∃ (A B C : ℝ × ℝ), -- vertices of the triangle
    (s1 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 2) ∧
    (s2 = ((C.1 - B.1)^2 + (C.2 - B.2)^2) / 2) ∧
    (s3 = ((A.1 - C.1)^2 + (A.2 - C.2)^2) / 2)

def distances_to_medians (d1 d2 d3 : ℝ) : Prop :=
  ∃ (P A B C : ℝ × ℝ), -- point P and vertices of the triangle
    (d1 = dist P ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) ∧
    (d2 = dist P ((A.1 + C.1) / 2, (A.2 + C.2) / 2)) ∧
    (d3 = dist P ((A.1 + B.1) / 2, (A.2 + B.2) / 2))

-- The theorem which we need to prove
theorem median_product_sum (h_medians : is_median_lengths s1 s2 s3) 
  (h_distances : distances_to_medians d1 d2 d3) :
  s1 * d1 + s2 * d2 + s3 * d3 = 0 := sorry

end median_product_sum_l665_665819


namespace average_salary_of_managers_l665_665123

theorem average_salary_of_managers (m_avg : ℝ) (assoc_avg : ℝ) (company_avg : ℝ)
  (managers : ℕ) (associates : ℕ) (total_employees : ℕ)
  (h_assoc_avg : assoc_avg = 30000) (h_company_avg : company_avg = 40000)
  (h_managers : managers = 15) (h_associates : associates = 75) (h_total_employees : total_employees = 90)
  (h_total_employees_def : total_employees = managers + associates)
  (h_total_salary_managers : ∀ m_avg, total_employees * company_avg = managers * m_avg + associates * assoc_avg) :
  m_avg = 90000 :=
by
  sorry

end average_salary_of_managers_l665_665123


namespace cube_volume_l665_665934

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l665_665934


namespace cyclic_polygon_condition_l665_665380

variable {n : ℕ} (A : fin n → ℝ) (b c : fin n → ℝ)

theorem cyclic_polygon_condition (h_n : 4 ≤ n) (convex : convex_polygon A) :
  cyclic_polygon A ↔ 
  ∃ (b c : fin n → ℝ), ∀ i j, 1 ≤ i → i < j → j ≤ n → (dist (A i) (A j) = (b j) * (c i) - (b i) * (c j)) :=
sorry

end cyclic_polygon_condition_l665_665380


namespace num_palindromes_l665_665520

def palindrome (n : ℕ) : Prop :=
  ∀ (k : ℕ), k < n → take k (to_list n) = reverse (drop (n - k) (to_list n))

theorem num_palindromes (n : ℕ) (h : ∃ k : ℕ, k = 2 * n + 1 ∧ n > 0) : 
  9 * 10^n = count { m | palindrome m ∧ length (to_list m) = 2 * n + 1 } :=
by
  sorry

end num_palindromes_l665_665520


namespace age_difference_is_13_l665_665837

variables (A B C X : ℕ)
variables (total_age_A_B total_age_B_C : ℕ)

-- Conditions
def condition1 : Prop := total_age_A_B = total_age_B_C + X
def condition2 : Prop := C = A - 13

-- Theorem statement
theorem age_difference_is_13 (h1: condition1 total_age_A_B total_age_B_C X)
                             (h2: condition2 A C) :
  X = 13 :=
sorry

end age_difference_is_13_l665_665837


namespace unit_conversion_factor_l665_665833

theorem unit_conversion_factor (u : ℝ) (h₁ : u = 5) (h₂ : (u * 0.9)^2 = 20.25) : u = 5 → (1 : ℝ) = 0.9  :=
sorry

end unit_conversion_factor_l665_665833


namespace cube_volume_is_1728_l665_665946

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l665_665946


namespace cube_volume_is_1728_l665_665948

noncomputable def cube_volume_from_surface_area (A : ℝ) (h : A = 864) : ℝ := 
  let s := real.sqrt (A / 6) in
  s^3

theorem cube_volume_is_1728 : cube_volume_from_surface_area 864 (by rfl) = 1728 :=
sorry

end cube_volume_is_1728_l665_665948


namespace sum_powers_i_l665_665199

variable (i : ℂ)

theorem sum_powers_i (h : i^2 = -1) : i^(703) + i^(702) + i^(701) + i^(700) + ... + i^2 + i + 1 = 1 := 
sorry

end sum_powers_i_l665_665199


namespace simplify_expression_l665_665792

variable (a : ℝ)

theorem simplify_expression (h₁ : a ≠ -3) (h₂ : a ≠ 1) :
  (1 - 4/(a + 3)) / ((a^2 - 2*a + 1) / (2*a + 6)) = 2 / (a - 1) :=
sorry

end simplify_expression_l665_665792


namespace nsq_in_S_l665_665743

def is_sum_of_two_squares (x : ℕ) : Prop := 
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = x

def S : set ℕ := { n | is_sum_of_two_squares (n - 1) ∧ is_sum_of_two_squares n ∧ is_sum_of_two_squares (n + 1) }

theorem nsq_in_S (n : ℕ) (hn : n ∈ S) : n^2 ∈ S :=
sorry

end nsq_in_S_l665_665743


namespace minimum_total_trips_l665_665814

theorem minimum_total_trips :
  ∃ (x y : ℕ), (31 * x + 32 * y = 5000) ∧ (x + y = 157) :=
by
  sorry

end minimum_total_trips_l665_665814


namespace correlation_coefficient_correct_l665_665989

variable (r : ℝ)

theorem correlation_coefficient_correct :
  (|r| ≤ 1) →
  (∀ x y : ℝ, |x| < |y| ∧ |y| ≤ 1 → x < y) →
  (|r| = 1 → "strong correlation") ∧ (|r| = 0 → "no correlation") →
  "The correct statement about the correlation coefficient is B" :=
by
  sorry

end correlation_coefficient_correct_l665_665989


namespace range_x1_x2_l665_665669

theorem range_x1_x2
  (x1 x2 x3 : ℝ)
  (hx3_le_x2 : x3 ≤ x2)
  (hx2_le_x1 : x2 ≤ x1)
  (hx_sum : x1 + x2 + x3 = 1)
  (hfx_sum : (x1^2) + (x2^2) + (x3^2) = 1) :
  (2 / 3 : ℝ) ≤ x1 + x2 ∧ x1 + x2 ≤ (4 / 3 : ℝ) :=
sorry

end range_x1_x2_l665_665669


namespace cube_volume_from_surface_area_l665_665922

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end cube_volume_from_surface_area_l665_665922


namespace cube_volume_l665_665893

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l665_665893


namespace scheduling_plans_count_l665_665547

open Function

/--
There are 7 employees to be scheduled from October 1 to October 7, 
one per day, such that:
- A and B are scheduled on consecutive days,
- C is not scheduled on October 1st,
- D is not scheduled on October 7th.

We need to prove that the total number of different scheduling plans is 1008.
-/
theorem scheduling_plans_count :
  let employees : Fin 7 := ⟨6, by decide⟩ -- Fix employees as Fin 7 since we have 7 employees.
  let count_schedule_plans (p : List (Fin 7)) : Bool :=
    p.length = 7 ∧ -- 7 employees scheduled
    ∃ i, p[i] = (0, 1) ∨ p[i] = (6, 7) ∨ -- A and B scheduled consecutively
           (∃ j, p[j] = (C, 7) ∧ (1 ≤ j ∧ j ≤ 5)) ∨ -- C not on October 1st
           (∃ k, p[k] = (D, 1) ∧ (0 ≤ k ∧ k ≤ 5)) -- D not on October 7th
    in nat.factorial 7 = 1008 := sorry

end scheduling_plans_count_l665_665547


namespace binom_16_10_is_8008_l665_665644

theorem binom_16_10_is_8008 (h1 : binom 15 8 = 6435) (h2 : binom 15 9 = 5005) (h3 : binom 17 10 = 19448) : binom 16 10 = 8008 := 
by 
  sorry

end binom_16_10_is_8008_l665_665644


namespace correct_statistics_statement_l665_665722

noncomputable def deyang_city_statistics : Prop :=
  let test_scores : ℕ → Prop := λ n, n = 1200
  let sample_size : ℕ := 100
  let total_students : ℕ := 1200
  let selected_students : ℕ := 100
  (∀ n, n < 1200 → test_scores n) →
  (selected_students = sample_size) →
  %( "The test score of each eighth-grade student in the school is an individual." )

theorem correct_statistics_statement :
  deyang_city_statistics := by sorry

end correct_statistics_statement_l665_665722
