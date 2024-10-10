import Mathlib

namespace right_triangle_area_l3995_399562

theorem right_triangle_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  b = (2/3) * a →
  b = (2/3) * c →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 32/9 := by
sorry

end right_triangle_area_l3995_399562


namespace max_value_cos_sin_sum_l3995_399539

theorem max_value_cos_sin_sum :
  ∃ (M : ℝ), M = 5 ∧ ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ M :=
by sorry

end max_value_cos_sin_sum_l3995_399539


namespace line_plane_intersection_l3995_399527

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary operations and relations
variable (intersect : Plane → Plane → Line)
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (intersects : Line → Line → Prop)

-- State the theorem
theorem line_plane_intersection 
  (m n : Line) (α β : Plane) :
  intersect α β = m → subset n α → 
  (parallel m n) ∨ (intersects m n) := by
  sorry

end line_plane_intersection_l3995_399527


namespace union_of_A_and_B_l3995_399534

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 3} := by sorry

end union_of_A_and_B_l3995_399534


namespace not_always_increasing_sum_of_increasing_and_decreasing_l3995_399568

-- Define the concept of an increasing function
def Increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the concept of a decreasing function
def Decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem not_always_increasing_sum_of_increasing_and_decreasing :
  ¬(∀ f g : ℝ → ℝ, Increasing f → Decreasing g → Increasing (λ x ↦ f x + g x)) :=
sorry

end not_always_increasing_sum_of_increasing_and_decreasing_l3995_399568


namespace problem_part1_problem_part2_l3995_399502

-- Part 1
theorem problem_part1 (m n : ℕ) (h1 : m > 0) (h2 : n > 0) 
  (h3 : 3 * m + 2 * n = 225) (h4 : Nat.gcd m n = 15) : 
  m + n = 105 := by
  sorry

-- Part 2
theorem problem_part2 (m n : ℕ) (h1 : m > 0) (h2 : n > 0) 
  (h3 : 3 * m + 2 * n = 225) (h4 : Nat.lcm m n = 45) : 
  m + n = 90 := by
  sorry

end problem_part1_problem_part2_l3995_399502


namespace rubiks_cube_probabilities_l3995_399533

/-- The probability of person A solving the cube within 30 seconds -/
def prob_A : ℝ := 0.8

/-- The probability of person B solving the cube within 30 seconds -/
def prob_B : ℝ := 0.6

/-- The probability of person A succeeding on their third attempt -/
def prob_A_third_attempt : ℝ := (1 - prob_A) * (1 - prob_A) * prob_A

/-- The probability that at least one of them succeeds on their first attempt -/
def prob_at_least_one_first_attempt : ℝ := 1 - (1 - prob_A) * (1 - prob_B)

theorem rubiks_cube_probabilities :
  prob_A_third_attempt = 0.032 ∧ prob_at_least_one_first_attempt = 0.92 := by
  sorry

end rubiks_cube_probabilities_l3995_399533


namespace two_pairs_four_shoes_l3995_399537

/-- Given that a person buys a certain number of pairs of shoes, and each pair consists of a certain number of shoes, calculate the total number of new shoes. -/
def total_new_shoes (pairs_bought : ℕ) (shoes_per_pair : ℕ) : ℕ :=
  pairs_bought * shoes_per_pair

/-- Theorem stating that buying 2 pairs of shoes, with 2 shoes per pair, results in 4 new shoes. -/
theorem two_pairs_four_shoes :
  total_new_shoes 2 2 = 4 := by
  sorry

#eval total_new_shoes 2 2

end two_pairs_four_shoes_l3995_399537


namespace divisibility_by_five_l3995_399597

theorem divisibility_by_five (d : Nat) : 
  d ≤ 9 → (41830 + d) % 5 = 0 ↔ d = 0 ∨ d = 5 := by
  sorry

end divisibility_by_five_l3995_399597


namespace max_value_of_x_plus_inv_x_l3995_399595

theorem max_value_of_x_plus_inv_x (x : ℝ) (h : 15 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 17 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 17 :=
by sorry

end max_value_of_x_plus_inv_x_l3995_399595


namespace monomial_sum_condition_l3995_399581

theorem monomial_sum_condition (a b : ℝ) (m n : ℕ) :
  (∃ k : ℝ, ∃ p q : ℕ, 2 * a^(m+2) * b^(2*n+2) + a^3 * b^8 = k * a^p * b^q) →
  m = 1 ∧ n = 3 := by
  sorry

end monomial_sum_condition_l3995_399581


namespace no_proper_divisor_sum_set_equality_l3995_399591

theorem no_proper_divisor_sum_set_equality (n : ℕ) : 
  (∃ (d₁ d₂ d₃ : ℕ), 1 < d₁ ∧ d₁ < n ∧ d₁ ∣ n ∧
                     1 < d₂ ∧ d₂ < n ∧ d₂ ∣ n ∧
                     1 < d₃ ∧ d₃ < n ∧ d₃ ∣ n ∧
                     d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃) →
  ¬∃ (m : ℕ), {x : ℕ | ∃ (a b : ℕ), 1 < a ∧ a < n ∧ a ∣ n ∧
                                   1 < b ∧ b < n ∧ b ∣ n ∧
                                   x = a + b} =
              {y : ℕ | 1 < y ∧ y < m ∧ y ∣ m} :=
by sorry

end no_proper_divisor_sum_set_equality_l3995_399591


namespace no_matching_pyramids_l3995_399566

/-- Represents a convex n-sided pyramid -/
structure NSidedPyramid (n : ℕ) :=
  (convex : Bool)
  (dihedralAngles : Fin n → ℝ)

/-- Represents a triangular pyramid -/
structure TriangularPyramid :=
  (dihedralAngles : Fin 4 → ℝ)

/-- The theorem stating that no such pair of pyramids exists -/
theorem no_matching_pyramids :
  ∀ (n : ℕ) (nPyramid : NSidedPyramid n) (tPyramid : TriangularPyramid),
    n ≥ 4 →
    nPyramid.convex = true →
    (∃ (i j k l : Fin n),
      i ≠ j ∧ i ≠ k ∧ i ≠ l ∧
      j ≠ k ∧ j ≠ l ∧
      k ≠ l ∧
      nPyramid.dihedralAngles i = tPyramid.dihedralAngles 0 ∧
      nPyramid.dihedralAngles j = tPyramid.dihedralAngles 1 ∧
      nPyramid.dihedralAngles k = tPyramid.dihedralAngles 2 ∧
      nPyramid.dihedralAngles l = tPyramid.dihedralAngles 3) →
    False :=
by sorry

end no_matching_pyramids_l3995_399566


namespace students_per_group_l3995_399528

theorem students_per_group 
  (total_students : ℕ) 
  (students_not_picked : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 36) 
  (h2 : students_not_picked = 9) 
  (h3 : num_groups = 3) : 
  (total_students - students_not_picked) / num_groups = 9 := by
sorry

end students_per_group_l3995_399528


namespace lcm_from_product_and_hcf_l3995_399540

theorem lcm_from_product_and_hcf (a b : ℕ+) :
  a * b = 82500 → Nat.gcd a b = 55 → Nat.lcm a b = 1500 := by
  sorry

end lcm_from_product_and_hcf_l3995_399540


namespace chloe_shoes_altered_l3995_399519

/-- Given the cost per shoe and total cost, calculate the number of pairs of shoes to be altered. -/
def shoesAltered (costPerShoe : ℕ) (totalCost : ℕ) : ℕ :=
  (totalCost / costPerShoe) / 2

/-- Theorem: Given the specific costs, prove that Chloe wants to get 14 pairs of shoes altered. -/
theorem chloe_shoes_altered :
  shoesAltered 37 1036 = 14 := by
  sorry

end chloe_shoes_altered_l3995_399519


namespace fifth_power_sum_l3995_399544

theorem fifth_power_sum (x : ℝ) (h : x + 1/x = -5) : x^5 + 1/x^5 = -2525 := by
  sorry

end fifth_power_sum_l3995_399544


namespace apple_pies_count_l3995_399557

def total_apple_weight : ℕ := 120
def applesauce_fraction : ℚ := 1/2
def pounds_per_pie : ℕ := 4

theorem apple_pies_count :
  (total_apple_weight * (1 - applesauce_fraction) / pounds_per_pie : ℚ) = 15 := by
  sorry

end apple_pies_count_l3995_399557


namespace equation_solution_l3995_399507

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ = 2 ∧ x₂ = (-1 - Real.sqrt 17) / 2) ∧
  (∀ x : ℝ, x^2 - |x - 1| - 3 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end equation_solution_l3995_399507


namespace matthew_egg_rolls_l3995_399514

/-- Given the egg roll consumption of Alvin, Patrick, and Matthew, prove that Matthew ate 6 egg rolls. -/
theorem matthew_egg_rolls (alvin patrick matthew : ℕ) : 
  alvin = 4 →
  patrick = alvin / 2 →
  matthew = 3 * patrick →
  matthew = 6 := by
sorry

end matthew_egg_rolls_l3995_399514


namespace triangle_properties_l3995_399547

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def acute_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def triangle_conditions (t : Triangle) : Prop :=
  acute_triangle t ∧
  t.a = 2 * t.b * Real.sin t.A ∧
  t.a = 3 * Real.sqrt 3 ∧
  t.c = 5

-- State the theorem
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = Real.pi/6 ∧ 
  t.b = Real.sqrt 7 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B) = (15 * Real.sqrt 3)/4 :=
sorry

end triangle_properties_l3995_399547


namespace watch_cost_price_l3995_399504

/-- Proves that the cost price of a watch is 280 Rs. given specific selling conditions -/
theorem watch_cost_price (selling_price : ℝ) : 
  (selling_price = 0.54 * 280) →  -- Sold at 46% loss
  (selling_price + 140 = 1.04 * 280) →  -- If sold for 140 more, 4% gain
  280 = 280 := by sorry

end watch_cost_price_l3995_399504


namespace conic_is_ellipse_iff_l3995_399592

/-- A conic section represented by the equation x^2 + 9y^2 - 6x + 27y = k --/
def conic (k : ℝ) (x y : ℝ) : Prop :=
  x^2 + 9*y^2 - 6*x + 27*y = k

/-- Predicate for a non-degenerate ellipse --/
def is_nondegenerate_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b h k : ℝ), a > 0 ∧ b > 0 ∧ 
    ∀ x y, f x y ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

theorem conic_is_ellipse_iff (k : ℝ) :
  is_nondegenerate_ellipse (conic k) ↔ k > -117/4 :=
sorry

end conic_is_ellipse_iff_l3995_399592


namespace geometric_sequence_ninth_term_l3995_399588

/-- A geometric sequence with positive terms and common ratio 2 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem geometric_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_prod : a 3 * a 13 = 16) :
  a 9 = 8 := by
sorry

end geometric_sequence_ninth_term_l3995_399588


namespace not_both_follow_control_principle_option_d_is_incorrect_l3995_399510

/-- Represents an experimental approach -/
inductive ExperimentalApproach
| BlankControl
| RepeatWithSameSoil

/-- Represents a scientific principle -/
inductive ScientificPrinciple
| Control
| Repeatability

/-- Function to determine which principle an approach follows -/
def principleFollowed (approach : ExperimentalApproach) : ScientificPrinciple :=
  match approach with
  | ExperimentalApproach.BlankControl => ScientificPrinciple.Control
  | ExperimentalApproach.RepeatWithSameSoil => ScientificPrinciple.Repeatability

/-- Theorem stating that not both approaches follow the control principle -/
theorem not_both_follow_control_principle :
  ¬(principleFollowed ExperimentalApproach.BlankControl = ScientificPrinciple.Control ∧
     principleFollowed ExperimentalApproach.RepeatWithSameSoil = ScientificPrinciple.Control) :=
by sorry

/-- Main theorem proving that the statement in option D is incorrect -/
theorem option_d_is_incorrect :
  ¬(∀ (approach : ExperimentalApproach), principleFollowed approach = ScientificPrinciple.Control) :=
by sorry

end not_both_follow_control_principle_option_d_is_incorrect_l3995_399510


namespace square_plot_area_l3995_399536

/-- The area of a square plot with side length 50.5 m is 2550.25 square meters. -/
theorem square_plot_area : 
  let side_length : ℝ := 50.5
  let area : ℝ := side_length * side_length
  area = 2550.25 := by
  sorry

end square_plot_area_l3995_399536


namespace cycle_iff_minimal_cut_l3995_399570

-- Define a planar multigraph
structure PlanarMultigraph where
  V : Type*  -- Vertex set
  E : Type*  -- Edge set
  is_planar : Bool
  is_connected : Bool

-- Define a dual graph
def DualGraph (G : PlanarMultigraph) : PlanarMultigraph := sorry

-- Define a cycle in a graph
def is_cycle (G : PlanarMultigraph) (E : Set G.E) : Prop := sorry

-- Define a cut in a graph
def is_cut (G : PlanarMultigraph) (E : Set G.E) : Prop := sorry

-- Define a minimal cut
def is_minimal_cut (G : PlanarMultigraph) (E : Set G.E) : Prop := sorry

-- Define the dual edge set
def dual_edge_set (G : PlanarMultigraph) (E : Set G.E) : Set (DualGraph G).E := sorry

-- Main theorem
theorem cycle_iff_minimal_cut (G : PlanarMultigraph) (E : Set G.E) :
  is_cycle G E ↔ is_minimal_cut (DualGraph G) (dual_edge_set G E) := by sorry

end cycle_iff_minimal_cut_l3995_399570


namespace intersection_equality_implies_m_value_l3995_399555

theorem intersection_equality_implies_m_value (m : ℝ) : 
  ({3, 4, m^2 - 3*m - 1} ∩ {2*m, -3} : Set ℝ) = {-3} → m = 1 := by
  sorry

end intersection_equality_implies_m_value_l3995_399555


namespace complex_square_eq_abs_square_iff_real_l3995_399522

open Complex

theorem complex_square_eq_abs_square_iff_real (z : ℂ) :
  (z - 1)^2 = abs (z - 1)^2 ↔ z.im = 0 :=
sorry

end complex_square_eq_abs_square_iff_real_l3995_399522


namespace money_distribution_l3995_399560

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (bc_sum : B + C = 350)
  (c_amount : C = 50) :
  A + C = 200 :=
by sorry

end money_distribution_l3995_399560


namespace school_population_after_additions_l3995_399567

theorem school_population_after_additions 
  (initial_girls : ℕ) 
  (initial_boys : ℕ) 
  (initial_teachers : ℕ) 
  (additional_girls : ℕ) 
  (additional_boys : ℕ) 
  (additional_teachers : ℕ) 
  (h1 : initial_girls = 732) 
  (h2 : initial_boys = 761) 
  (h3 : initial_teachers = 54) 
  (h4 : additional_girls = 682) 
  (h5 : additional_boys = 8) 
  (h6 : additional_teachers = 3) : 
  initial_girls + initial_boys + initial_teachers + 
  additional_girls + additional_boys + additional_teachers = 2240 :=
by
  sorry


end school_population_after_additions_l3995_399567


namespace onions_sum_to_285_l3995_399556

/-- The total number of onions grown by Sara, Sally, Fred, Amy, and Matthew -/
def total_onions (sara sally fred amy matthew : ℕ) : ℕ :=
  sara + sally + fred + amy + matthew

/-- Theorem stating that the total number of onions grown is 285 -/
theorem onions_sum_to_285 :
  total_onions 40 55 90 25 75 = 285 := by
  sorry

end onions_sum_to_285_l3995_399556


namespace factorization_proof_l3995_399516

theorem factorization_proof (a b c : ℝ) : 
  (a^2 + 2*b^2 - 2*c^2 + 3*a*b + a*c = (a + b - c)*(a + 2*b + 2*c)) ∧
  (a^2 - 2*b^2 - 2*c^2 - a*b + 5*b*c - a*c = (a - 2*b + c)*(a + b - 2*c)) := by
  sorry

end factorization_proof_l3995_399516


namespace number_multiplied_by_9999_l3995_399526

theorem number_multiplied_by_9999 :
  ∃ x : ℕ, x * 9999 = 724817410 ∧ x = 72492 := by
  sorry

end number_multiplied_by_9999_l3995_399526


namespace replaced_person_weight_l3995_399551

/-- Proves that the weight of a replaced person is 65 kg given the conditions of the problem -/
theorem replaced_person_weight (n : ℕ) (old_avg new_avg new_weight : ℝ) : 
  n = 8 ∧ 
  new_avg - old_avg = 3.5 ∧
  new_weight = 93 →
  (n * new_avg - new_weight) / (n - 1) = 65 :=
by sorry

end replaced_person_weight_l3995_399551


namespace good_carrots_count_l3995_399509

theorem good_carrots_count (vanessa_carrots : ℕ) (mom_carrots : ℕ) (bad_carrots : ℕ) 
  (h1 : vanessa_carrots = 17)
  (h2 : mom_carrots = 14)
  (h3 : bad_carrots = 7) :
  vanessa_carrots + mom_carrots - bad_carrots = 24 :=
by
  sorry

end good_carrots_count_l3995_399509


namespace exactly_one_real_solution_iff_l3995_399511

/-- The cubic equation in x with parameter a -/
def cubic_equation (a x : ℝ) : ℝ :=
  x^3 - a*x^2 - (a+1)*x + a^2 - 2

/-- The condition for exactly one real solution -/
def has_exactly_one_real_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, cubic_equation a x = 0

/-- Theorem stating the condition for exactly one real solution -/
theorem exactly_one_real_solution_iff (a : ℝ) :
  has_exactly_one_real_solution a ↔ a < 7/4 :=
sorry

end exactly_one_real_solution_iff_l3995_399511


namespace age_ratio_proof_l3995_399512

/-- Given the ages of three people A, B, and C, prove that the ratio of B's age to C's age is 2:1 --/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →
  a + b + c = 27 →
  b = 10 →
  b / c = 2 / 1 := by sorry

end age_ratio_proof_l3995_399512


namespace or_false_sufficient_not_necessary_for_and_false_l3995_399523

theorem or_false_sufficient_not_necessary_for_and_false (p q : Prop) :
  (¬(p ∨ q) → ¬p ∧ ¬q) ∧ ∃ (p q : Prop), (¬p ∧ ¬q) ∧ ¬¬(p ∨ q) := by
  sorry

end or_false_sufficient_not_necessary_for_and_false_l3995_399523


namespace rectangle_cylinder_volume_ratio_l3995_399518

theorem rectangle_cylinder_volume_ratio :
  let rectangle_length : ℝ := 10
  let rectangle_width : ℝ := 6
  let cylinder_A_height : ℝ := rectangle_width
  let cylinder_A_circumference : ℝ := rectangle_length
  let cylinder_B_height : ℝ := rectangle_length
  let cylinder_B_circumference : ℝ := rectangle_width
  let cylinder_A_volume : ℝ := (cylinder_A_circumference^2 * cylinder_A_height) / (4 * π)
  let cylinder_B_volume : ℝ := (cylinder_B_circumference^2 * cylinder_B_height) / (4 * π)
  max cylinder_A_volume cylinder_B_volume / min cylinder_A_volume cylinder_B_volume = 5 / 3 := by
sorry

end rectangle_cylinder_volume_ratio_l3995_399518


namespace A_intersect_B_eq_open_interval_l3995_399546

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 > 0}
def B : Set ℝ := {x : ℝ | x / (x - 1) < 0}

-- State the theorem
theorem A_intersect_B_eq_open_interval : A ∩ B = Set.Ioo 0 1 := by
  sorry

end A_intersect_B_eq_open_interval_l3995_399546


namespace power_sum_equality_l3995_399584

theorem power_sum_equality : 2^300 + (-2^301) = -2^300 := by
  sorry

end power_sum_equality_l3995_399584


namespace expression_change_l3995_399543

theorem expression_change (x a : ℝ) (b : ℝ) (h : a > 0) : 
  let f := fun x => x^3 - b
  let δ := fun (ε : ℝ) => f (x + ε) - (b + a^2) - (f x - b)
  (δ a = 3*x^2*a + 3*x*a^2 + a^3 - a^2) ∧ 
  (δ (-a) = -3*x^2*a + 3*x*a^2 - a^3 - a^2) :=
by sorry

end expression_change_l3995_399543


namespace harry_apples_l3995_399530

theorem harry_apples (x : ℕ) : x + 5 = 84 → x = 79 := by
  sorry

end harry_apples_l3995_399530


namespace cubic_inequality_reciprocal_l3995_399576

theorem cubic_inequality_reciprocal (a b : ℝ) (h1 : a^3 > b^3) (h2 : a * b > 0) :
  1 / a < 1 / b := by
sorry

end cubic_inequality_reciprocal_l3995_399576


namespace thousand_power_division_l3995_399552

theorem thousand_power_division :
  1000 * (1000^1000) / (500^1000) = 2^1001 * 500 := by
  sorry

end thousand_power_division_l3995_399552


namespace total_factories_to_check_l3995_399563

theorem total_factories_to_check (first_group : ℕ) (second_group : ℕ) (remaining : ℕ) :
  first_group = 69 → second_group = 52 → remaining = 48 →
  first_group + second_group + remaining = 169 := by
  sorry

end total_factories_to_check_l3995_399563


namespace intersection_point_inside_circle_l3995_399578

/-- The intersection point of two lines is inside a circle iff a is within a specific range -/
theorem intersection_point_inside_circle (a : ℝ) :
  let P : ℝ × ℝ := (a, 3 * a)  -- Intersection point of y = x + 2a and y = 2x + a
  (P.1 - 1)^2 + (P.2 - 1)^2 < 4 ↔ -1/5 < a ∧ a < 1 := by
  sorry

end intersection_point_inside_circle_l3995_399578


namespace quadratic_equation_roots_l3995_399521

/-- Given a quadratic equation x^2 + mx - 2 = 0 where -1 is a root,
    prove that m = -1 and the other root is 2 -/
theorem quadratic_equation_roots (m : ℝ) : 
  ((-1 : ℝ)^2 + m*(-1) - 2 = 0) → 
  (m = -1 ∧ ∃ r : ℝ, r ≠ -1 ∧ r^2 + m*r - 2 = 0 ∧ r = 2) :=
by sorry

end quadratic_equation_roots_l3995_399521


namespace binomial_identities_l3995_399596

theorem binomial_identities (n k : ℕ) (h : k ≤ n) :
  (n.factorial = n.choose k * k.factorial * (n - k).factorial) ∧
  (n.choose k = (n - 1).choose k + (n - 1).choose (k - 1)) := by
  sorry

end binomial_identities_l3995_399596


namespace trig_identity_l3995_399569

theorem trig_identity : 
  Real.sin (155 * π / 180) * Real.sin (55 * π / 180) + 
  Real.cos (25 * π / 180) * Real.cos (55 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end trig_identity_l3995_399569


namespace matrix_N_property_l3995_399572

theorem matrix_N_property :
  ∃ (N : Matrix (Fin 3) (Fin 3) ℝ),
    (∀ (u : Fin 3 → ℝ), N.mulVec u = (3 : ℝ) • u) ∧
    N = ![![3, 0, 0], ![0, 3, 0], ![0, 0, 3]] := by
  sorry

end matrix_N_property_l3995_399572


namespace blood_donation_theorem_l3995_399577

/-- Represents the number of people for each blood type -/
structure BloodDonors where
  typeO : Nat
  typeA : Nat
  typeB : Nat
  typeAB : Nat

/-- Calculates the number of ways to select one person to donate blood -/
def selectOneDonor (donors : BloodDonors) : Nat :=
  donors.typeO + donors.typeA + donors.typeB + donors.typeAB

/-- Calculates the number of ways to select one person from each blood type -/
def selectFourDonors (donors : BloodDonors) : Nat :=
  donors.typeO * donors.typeA * donors.typeB * donors.typeAB

/-- Theorem statement for the blood donation problem -/
theorem blood_donation_theorem (donors : BloodDonors) :
  selectOneDonor donors = donors.typeO + donors.typeA + donors.typeB + donors.typeAB ∧
  selectFourDonors donors = donors.typeO * donors.typeA * donors.typeB * donors.typeAB := by
  sorry

/-- Example with the given numbers -/
def example_donors : BloodDonors :=
  { typeO := 28, typeA := 7, typeB := 9, typeAB := 3 }

#eval selectOneDonor example_donors  -- Expected: 47
#eval selectFourDonors example_donors  -- Expected: 5292

end blood_donation_theorem_l3995_399577


namespace min_bullseyes_for_victory_l3995_399553

/-- Represents the possible scores in the archery tournament -/
inductive Score
  | bullseye : Score
  | ten : Score
  | five : Score
  | three : Score
  | zero : Score

/-- Convert a Score to its numerical value -/
def score_value : Score → Nat
  | Score.bullseye => 12
  | Score.ten => 10
  | Score.five => 5
  | Score.three => 3
  | Score.zero => 0

/-- The total number of shots in the tournament -/
def total_shots : Nat := 120

/-- The number of shots already taken -/
def shots_taken : Nat := 60

/-- Alex's lead after half the tournament -/
def alex_lead : Nat := 70

/-- Alex's minimum score per shot -/
def alex_min_score : Nat := 5

/-- The maximum possible score per shot -/
def max_score_per_shot : Nat := 12

/-- Theorem: The minimum number of consecutive bullseyes Alex needs to guarantee victory is 51 -/
theorem min_bullseyes_for_victory :
  ∀ n : Nat,
  (∀ m : Nat, m < n → 
    ∃ opponent_score : Nat,
    opponent_score ≤ (total_shots - shots_taken) * max_score_per_shot ∧
    alex_lead + n * score_value Score.bullseye + (total_shots - shots_taken - n) * alex_min_score ≤ opponent_score) ∧
  (∀ opponent_score : Nat,
   opponent_score ≤ (total_shots - shots_taken) * max_score_per_shot →
   alex_lead + n * score_value Score.bullseye + (total_shots - shots_taken - n) * alex_min_score > opponent_score) →
  n = 51 := by
  sorry

end min_bullseyes_for_victory_l3995_399553


namespace sine_negative_half_solutions_l3995_399538

theorem sine_negative_half_solutions : 
  ∃! (s : Finset ℝ), 
    (∀ x ∈ s, 0 ≤ x ∧ x < 2*π ∧ Real.sin x = -0.5) ∧ 
    s.card = 2 := by
  sorry

end sine_negative_half_solutions_l3995_399538


namespace diamonds_in_figure_l3995_399525

-- Define the sequence of figures
def F (n : ℕ) : ℕ :=
  2 * n^2 - 2 * n + 1

-- State the theorem
theorem diamonds_in_figure (n : ℕ) (h : n ≥ 1) : 
  F n = 2 * n^2 - 2 * n + 1 :=
by sorry

-- Verify the result for F_20
example : F 20 = 761 :=
by sorry

end diamonds_in_figure_l3995_399525


namespace shop_length_calculation_l3995_399549

/-- Given a shop with specified rent and dimensions, calculate its length -/
theorem shop_length_calculation (monthly_rent : ℕ) (width : ℕ) (annual_rent_per_sqft : ℕ) :
  monthly_rent = 2400 →
  width = 8 →
  annual_rent_per_sqft = 360 →
  (monthly_rent * 12) / (width * annual_rent_per_sqft) = 10 := by
  sorry

end shop_length_calculation_l3995_399549


namespace intersection_B_complement_A_l3995_399594

def I : Finset Nat := {1, 2, 3, 4, 5}
def A : Finset Nat := {2, 3, 5}
def B : Finset Nat := {1, 3}

theorem intersection_B_complement_A : B ∩ (I \ A) = {1} := by
  sorry

end intersection_B_complement_A_l3995_399594


namespace max_side_length_of_triangle_l3995_399565

theorem max_side_length_of_triangle (a b c : ℕ) : 
  a < b → b < c → a + b + c = 24 → c ≤ 11 := by
  sorry

end max_side_length_of_triangle_l3995_399565


namespace sufficient_not_necessary_l3995_399564

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x > 1 → 1 / x < 1) ∧
  (∃ x, 1 / x < 1 ∧ ¬(x > 1)) :=
sorry

end sufficient_not_necessary_l3995_399564


namespace smallest_n_multiple_of_seven_l3995_399500

theorem smallest_n_multiple_of_seven (x y : ℤ) 
  (h1 : (2*x + 3) % 7 = 0) 
  (h2 : (3*y - 4) % 7 = 0) : 
  (∃ n : ℕ+, (3*x^2 + 2*x*y + y^2 + n) % 7 = 0 ∧ 
   ∀ m : ℕ+, m < n → (3*x^2 + 2*x*y + y^2 + m) % 7 ≠ 0) → 
  (∃ n : ℕ+, n = 4 ∧ (3*x^2 + 2*x*y + y^2 + n) % 7 = 0 ∧ 
   ∀ m : ℕ+, m < n → (3*x^2 + 2*x*y + y^2 + m) % 7 ≠ 0) :=
by sorry

end smallest_n_multiple_of_seven_l3995_399500


namespace bounded_sequence_with_distance_condition_l3995_399559

theorem bounded_sequence_with_distance_condition :
  ∃ (a : ℕ → ℝ), 
    (∃ (C D : ℝ), ∀ n, C ≤ a n ∧ a n ≤ D) ∧ 
    (∀ (n m : ℕ), n > m → |a m - a n| ≥ 1 / (n - m : ℝ)) := by
  sorry

end bounded_sequence_with_distance_condition_l3995_399559


namespace max_value_expression_l3995_399587

theorem max_value_expression (a b c : ℝ) (h1 : b > a) (h2 : a > c) (h3 : b ≠ 0) :
  ((2*a + 3*b)^2 + (b - c)^2 + (2*c - a)^2) / b^2 ≤ 27 := by
sorry

end max_value_expression_l3995_399587


namespace log_division_simplification_l3995_399582

theorem log_division_simplification : 
  Real.log 8 / Real.log (1/8) = -1 := by sorry

end log_division_simplification_l3995_399582


namespace fraction_meaningful_condition_l3995_399571

theorem fraction_meaningful_condition (x : ℝ) :
  (∃ y : ℝ, y = 3 / (x - 1)) ↔ x ≠ 1 := by sorry

end fraction_meaningful_condition_l3995_399571


namespace sphere_volume_ratio_l3995_399585

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : 4 * Real.pi * r₁^2 / (4 * Real.pi * r₂^2) = 1 / 9) :
  (4 / 3) * Real.pi * r₁^3 / ((4 / 3) * Real.pi * r₂^3) = 1 / 27 := by
sorry

end sphere_volume_ratio_l3995_399585


namespace extreme_value_implies_m_plus_n_11_l3995_399542

/-- A function f with an extreme value of 0 at x = -1 -/
def f (m n : ℝ) (x : ℝ) : ℝ := x^3 + 3*m*x^2 + n*x + m^2

/-- The derivative of f -/
def f' (m n : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*m*x + n

theorem extreme_value_implies_m_plus_n_11 (m n : ℝ) :
  (f m n (-1) = 0) →
  (f' m n (-1) = 0) →
  (m + n = 11) :=
by sorry

end extreme_value_implies_m_plus_n_11_l3995_399542


namespace complex_modulus_problem_l3995_399503

theorem complex_modulus_problem (z : ℂ) : z = (2 * I) / (1 - I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l3995_399503


namespace original_average_age_is_40_l3995_399513

/-- Proves that the original average age of a class is 40 years given specific conditions. -/
theorem original_average_age_is_40 
  (N : ℕ) -- Original number of students
  (A : ℝ) -- Original average age
  (new_students : ℕ) -- Number of new students
  (new_age : ℝ) -- Average age of new students
  (age_decrease : ℝ) -- Decrease in average age after new students join
  (h1 : N = 2) -- Original number of students is 2
  (h2 : new_students = 2) -- 2 new students join
  (h3 : new_age = 32) -- Average age of new students is 32
  (h4 : age_decrease = 4) -- Average age decreases by 4
  (h5 : (A * N + new_age * new_students) / (N + new_students) = A - age_decrease) -- New average age equation
  : A = 40 := by
  sorry

end original_average_age_is_40_l3995_399513


namespace car_speeds_problem_l3995_399586

/-- Proves that given the problem conditions, the speeds of the two cars are 60 km/h and 90 km/h -/
theorem car_speeds_problem (total_distance : ℝ) (meeting_distance : ℝ) (speed_difference : ℝ)
  (h1 : total_distance = 200)
  (h2 : meeting_distance = 80)
  (h3 : speed_difference = 30)
  (h4 : meeting_distance / speed_a = (total_distance - meeting_distance) / (speed_a + speed_difference))
  (speed_a : ℝ)
  (speed_b : ℝ)
  (h5 : speed_b = speed_a + speed_difference) :
  speed_a = 60 ∧ speed_b = 90 :=
by
  sorry

end car_speeds_problem_l3995_399586


namespace factorization_of_quadratic_l3995_399580

theorem factorization_of_quadratic (m : ℝ) : m^2 - 4*m = m*(m - 4) := by
  sorry

end factorization_of_quadratic_l3995_399580


namespace sequence_property_l3995_399505

/-- Given a sequence where the nth term is of the form 32000+n + m/n = (2000+n) 3(m/n),
    prove that when n = 2016, (n³)/(n²) = 2016 -/
theorem sequence_property : 
  ∀ n : ℕ, n = 2016 → (n^3 : ℚ) / (n^2 : ℚ) = 2016 := by
  sorry

end sequence_property_l3995_399505


namespace distance_between_tangent_circles_l3995_399545

/-- The distance between centers of two internally tangent circles -/
def distance_between_centers (r₁ r₂ : ℝ) : ℝ := |r₂ - r₁|

/-- Theorem: The distance between centers of two internally tangent circles
    with radii 3 and 4 is 1 -/
theorem distance_between_tangent_circles :
  let r₁ : ℝ := 3
  let r₂ : ℝ := 4
  distance_between_centers r₁ r₂ = 1 := by sorry

end distance_between_tangent_circles_l3995_399545


namespace smallest_solution_of_equation_l3995_399506

theorem smallest_solution_of_equation (x : ℝ) : 
  (1 / (x - 1) + 1 / (x - 5) = 4 / (x - 4)) → 
  x ≥ (5 - Real.sqrt 33) / 2 := by
  sorry

end smallest_solution_of_equation_l3995_399506


namespace inscribed_sphere_radius_tetrahedron_l3995_399501

/-- Given a tetrahedron with volume V, face areas S₁, S₂, S₃, S₄, and an inscribed sphere of radius R,
    prove that R = 3V / (S₁ + S₂ + S₃ + S₄) -/
theorem inscribed_sphere_radius_tetrahedron 
  (V : ℝ) 
  (S₁ S₂ S₃ S₄ : ℝ) 
  (R : ℝ) 
  (h_volume : V > 0)
  (h_areas : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0)
  (h_inscribed : R > 0) :
  R = 3 * V / (S₁ + S₂ + S₃ + S₄) := by
  sorry

end inscribed_sphere_radius_tetrahedron_l3995_399501


namespace p_sufficient_t_l3995_399598

-- Define the propositions
variable (p q r s t : Prop)

-- Define the conditions
axiom p_r_sufficient_q : (p → q) ∧ (r → q)
axiom s_necessary_sufficient_q : (q ↔ s)
axiom t_necessary_s : (s → t)
axiom t_sufficient_r : (t → r)

-- Theorem to prove
theorem p_sufficient_t : p → t := by sorry

end p_sufficient_t_l3995_399598


namespace stripe_width_for_equal_areas_l3995_399599

/-- Given a rectangle with dimensions 40 cm × 20 cm and two perpendicular stripes of equal width,
    prove that the width of the stripes for equal white and gray areas is 30 - 5√5 cm. -/
theorem stripe_width_for_equal_areas : ∃ (x : ℝ),
  x > 0 ∧ x < 20 ∧
  (40 * x + 20 * x - x^2 = (40 * 20) / 2) ∧
  x = 30 - 5 * Real.sqrt 5 := by
  sorry

end stripe_width_for_equal_areas_l3995_399599


namespace not_p_neither_sufficient_nor_necessary_for_q_l3995_399531

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := 5 * x - 6 ≤ x^2

-- Define what it means for ¬p to be neither sufficient nor necessary for q
def neither_sufficient_nor_necessary : Prop :=
  (∃ x : ℝ, ¬(p x) ∧ ¬(q x)) ∧ 
  (∃ y : ℝ, ¬(p y) ∧ q y) ∧ 
  (∃ z : ℝ, p z ∧ q z)

-- Theorem statement
theorem not_p_neither_sufficient_nor_necessary_for_q : 
  neither_sufficient_nor_necessary :=
sorry

end not_p_neither_sufficient_nor_necessary_for_q_l3995_399531


namespace petya_strategy_exists_l3995_399573

theorem petya_strategy_exists (opponent_choice : ℚ → ℚ) : 
  ∃ (a b c : ℚ), 
    ∃ (x y : ℂ), 
      (x^3 + a*x^2 + b*x + c = 0) ∧ 
      (y^3 + a*y^2 + b*y + c = 0) ∧ 
      (y - x = 2014) ∧
      ((a = opponent_choice b ∧ c = opponent_choice 0) ∨ 
       (b = opponent_choice a ∧ c = opponent_choice 0) ∨ 
       (a = opponent_choice c ∧ b = opponent_choice 0)) :=
by sorry

end petya_strategy_exists_l3995_399573


namespace expansion_equality_l3995_399508

theorem expansion_equality (x : ℝ) : 24 * (x + 3) * (2 * x - 4) = 48 * x^2 + 48 * x - 288 := by
  sorry

end expansion_equality_l3995_399508


namespace cat_grooming_time_is_640_l3995_399593

/-- Represents the time taken to groom a cat -/
def catGroomingTime (
  clipTime : ℕ)  -- Time to clip one nail in seconds
  (cleanEarTime : ℕ)  -- Time to clean one ear in seconds
  (shampooTime : ℕ)  -- Time to shampoo in minutes
  (clawsPerFoot : ℕ)  -- Number of claws per foot
  (feetCount : ℕ)  -- Number of feet
  (earCount : ℕ)  -- Number of ears
  (secondsPerMinute : ℕ)  -- Number of seconds in a minute
  : ℕ :=
  (clipTime * clawsPerFoot * feetCount) +  -- Time for clipping nails
  (cleanEarTime * earCount) +  -- Time for cleaning ears
  (shampooTime * secondsPerMinute)  -- Time for shampooing

theorem cat_grooming_time_is_640 :
  catGroomingTime 10 90 5 4 4 2 60 = 640 := by
  sorry

#eval catGroomingTime 10 90 5 4 4 2 60

end cat_grooming_time_is_640_l3995_399593


namespace quadratic_always_nonnegative_l3995_399548

theorem quadratic_always_nonnegative : ∀ x : ℝ, x^2 - x + 1 ≥ 0 := by
  sorry

end quadratic_always_nonnegative_l3995_399548


namespace faster_train_length_l3995_399535

/-- Calculates the length of a faster train given the speeds of two trains and the time taken for the faster train to cross a man in the slower train. -/
theorem faster_train_length
  (faster_speed slower_speed : ℝ)
  (crossing_time : ℝ)
  (h1 : faster_speed = 72)
  (h2 : slower_speed = 36)
  (h3 : crossing_time = 37)
  (h4 : faster_speed > slower_speed) :
  let relative_speed := faster_speed - slower_speed
  let relative_speed_ms := relative_speed * (1000 / 3600)
  relative_speed_ms * crossing_time = 370 := by
  sorry

#check faster_train_length

end faster_train_length_l3995_399535


namespace students_dislike_both_l3995_399574

/-- Given a class of students and their food preferences, calculate the number of students who don't like either food. -/
theorem students_dislike_both (total : ℕ) (like_fries : ℕ) (like_burgers : ℕ) (like_both : ℕ) 
  (h1 : total = 25)
  (h2 : like_fries = 15)
  (h3 : like_burgers = 10)
  (h4 : like_both = 6)
  (h5 : like_both ≤ like_fries ∧ like_both ≤ like_burgers) :
  total - (like_fries + like_burgers - like_both) = 6 := by
  sorry

#check students_dislike_both

end students_dislike_both_l3995_399574


namespace seniority_ordering_l3995_399590

-- Define the colleagues
inductive Colleague
| Tom
| Jerry
| Sam

-- Define the seniority relation
def more_senior (a b : Colleague) : Prop := sorry

-- Define the statements
def statement_I : Prop := more_senior Colleague.Jerry Colleague.Tom ∧ more_senior Colleague.Jerry Colleague.Sam
def statement_II : Prop := more_senior Colleague.Sam Colleague.Tom ∨ more_senior Colleague.Sam Colleague.Jerry
def statement_III : Prop := more_senior Colleague.Jerry Colleague.Tom ∨ more_senior Colleague.Sam Colleague.Tom

-- Theorem statement
theorem seniority_ordering :
  -- Exactly one statement is true
  (statement_I ∧ ¬statement_II ∧ ¬statement_III) ∨
  (¬statement_I ∧ statement_II ∧ ¬statement_III) ∨
  (¬statement_I ∧ ¬statement_II ∧ statement_III) →
  -- Seniority relation is transitive
  (∀ a b c : Colleague, more_senior a b → more_senior b c → more_senior a c) →
  -- Seniority relation is asymmetric
  (∀ a b : Colleague, more_senior a b → ¬more_senior b a) →
  -- All colleagues have different seniorities
  (∀ a b : Colleague, a ≠ b → more_senior a b ∨ more_senior b a) →
  -- The correct seniority ordering
  more_senior Colleague.Jerry Colleague.Tom ∧ more_senior Colleague.Tom Colleague.Sam :=
sorry

end seniority_ordering_l3995_399590


namespace disk_intersection_theorem_l3995_399517

-- Define a type for colors
inductive Color
  | Red
  | White
  | Green

-- Define a type for disks
structure Disk where
  color : Color
  center : ℝ × ℝ
  radius : ℝ

-- Define a function to check if two disks intersect
def intersect (d1 d2 : Disk) : Prop :=
  let (x1, y1) := d1.center
  let (x2, y2) := d2.center
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 ≤ (d1.radius + d2.radius) ^ 2

-- Define a function to check if three disks have a common point
def commonPoint (d1 d2 d3 : Disk) : Prop :=
  ∃ (x y : ℝ), 
    (x - d1.center.1) ^ 2 + (y - d1.center.2) ^ 2 ≤ d1.radius ^ 2 ∧
    (x - d2.center.1) ^ 2 + (y - d2.center.2) ^ 2 ≤ d2.radius ^ 2 ∧
    (x - d3.center.1) ^ 2 + (y - d3.center.2) ^ 2 ≤ d3.radius ^ 2

-- State the theorem
theorem disk_intersection_theorem (disks : Finset Disk) :
  (disks.card = 6) →
  (∃ (r1 r2 w1 w2 g1 g2 : Disk), 
    r1 ∈ disks ∧ r2 ∈ disks ∧ w1 ∈ disks ∧ w2 ∈ disks ∧ g1 ∈ disks ∧ g2 ∈ disks ∧
    r1.color = Color.Red ∧ r2.color = Color.Red ∧
    w1.color = Color.White ∧ w2.color = Color.White ∧
    g1.color = Color.Green ∧ g2.color = Color.Green) →
  (∀ (r w g : Disk), r ∈ disks → w ∈ disks → g ∈ disks →
    r.color = Color.Red → w.color = Color.White → g.color = Color.Green →
    commonPoint r w g) →
  (∃ (c : Color), ∃ (d1 d2 : Disk), d1 ∈ disks ∧ d2 ∈ disks ∧
    d1.color = c ∧ d2.color = c ∧ intersect d1 d2) :=
by sorry

end disk_intersection_theorem_l3995_399517


namespace triangle_sine_cosine_inequality_l3995_399589

theorem triangle_sine_cosine_inequality (A B C : ℝ) (h : A + B + C = π) :
  (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) < 2 := by
  sorry

end triangle_sine_cosine_inequality_l3995_399589


namespace select_products_l3995_399561

theorem select_products (total : ℕ) (qualified : ℕ) (unqualified : ℕ) (select : ℕ) 
    (h1 : total = qualified + unqualified) 
    (h2 : total = 50) 
    (h3 : qualified = 47) 
    (h4 : unqualified = 3) 
    (h5 : select = 4) : 
    (Nat.choose unqualified 1 * Nat.choose qualified 3 + 
     Nat.choose unqualified 2 * Nat.choose qualified 2 + 
     Nat.choose unqualified 3 * Nat.choose qualified 1) = 
    (Nat.choose total 4 - Nat.choose qualified 4) := by
  sorry

end select_products_l3995_399561


namespace current_velocity_l3995_399515

-- Define the rowing speeds
def downstream_speed (v c : ℝ) : ℝ := v + c
def upstream_speed (v c : ℝ) : ℝ := v - c

-- Define the conditions of the problem
def downstream_distance : ℝ := 32
def upstream_distance : ℝ := 14
def trip_time : ℝ := 6

-- Theorem statement
theorem current_velocity :
  ∃ (v c : ℝ),
    downstream_speed v c * trip_time = downstream_distance ∧
    upstream_speed v c * trip_time = upstream_distance ∧
    c = 1.5 := by
  sorry

end current_velocity_l3995_399515


namespace min_value_theorem_l3995_399583

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 3*b = 7) :
  (1 / (1 + a)) + (4 / (2 + b)) ≥ (13 + 4 * Real.sqrt 3) / 14 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3*b₀ = 7 ∧
    (1 / (1 + a₀)) + (4 / (2 + b₀)) = (13 + 4 * Real.sqrt 3) / 14 :=
sorry

end min_value_theorem_l3995_399583


namespace difference_A_B_l3995_399532

def A : ℕ → ℕ
  | 0 => 41
  | n + 1 => (2*n + 1) * (2*n + 2) + A n

def B : ℕ → ℕ
  | 0 => 1
  | n + 1 => (2*n) * (2*n + 1) + B n

theorem difference_A_B : A 20 - B 20 = 380 := by
  sorry

end difference_A_B_l3995_399532


namespace more_girls_than_boys_l3995_399554

theorem more_girls_than_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) : 
  total_students = 42 →
  boy_ratio = 3 →
  girl_ratio = 4 →
  ∃ (k : ℕ), 
    (boy_ratio * k + girl_ratio * k = total_students) ∧
    (girl_ratio * k - boy_ratio * k = 6) :=
by sorry

end more_girls_than_boys_l3995_399554


namespace remaining_perimeter_l3995_399579

/-- The perimeter of the remaining shape after cutting out two squares from a rectangle. -/
theorem remaining_perimeter (rectangle_length rectangle_width square1_side square2_side : ℕ) :
  rectangle_length = 50 ∧ 
  rectangle_width = 20 ∧ 
  square1_side = 12 ∧ 
  square2_side = 4 →
  2 * (rectangle_length + rectangle_width) + 4 * square1_side + 4 * square2_side = 204 := by
  sorry

end remaining_perimeter_l3995_399579


namespace count_pairs_satisfying_condition_l3995_399558

theorem count_pairs_satisfying_condition : 
  (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2 < 50 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 50) (Finset.range 50))).card = 204 := by
  sorry

end count_pairs_satisfying_condition_l3995_399558


namespace archie_antibiotics_duration_l3995_399575

/-- Calculates the number of days Archie can take antibiotics given the cost, 
    daily frequency, and available money. -/
def daysOfAntibiotics (costPerDose : ℚ) (dosesPerDay : ℕ) (availableMoney : ℚ) : ℚ :=
  availableMoney / (costPerDose * dosesPerDay)

/-- Proves that Archie can take antibiotics for 7 days given the specified conditions. -/
theorem archie_antibiotics_duration :
  daysOfAntibiotics 3 3 63 = 7 := by
sorry

end archie_antibiotics_duration_l3995_399575


namespace work_efficiency_l3995_399529

/-- Given two workers A and B, where A can finish a work in 18 days and B can do the same work in half the time taken by A, this theorem proves that working together, they can finish 1/6 of the work in one day. -/
theorem work_efficiency (days_A : ℕ) (days_B : ℕ) : 
  days_A = 18 → 
  days_B = days_A / 2 → 
  (1 : ℚ) / days_A + (1 : ℚ) / days_B = (1 : ℚ) / 6 := by
  sorry

end work_efficiency_l3995_399529


namespace work_completion_time_l3995_399524

theorem work_completion_time 
  (n : ℕ) -- number of persons
  (t : ℝ) -- time to complete the work
  (h : t = 12) -- given condition that work is completed in 12 days
  : (2 * n) * (3 : ℝ) = n * t / 2 := by
  sorry

end work_completion_time_l3995_399524


namespace sufficient_not_necessary_l3995_399520

theorem sufficient_not_necessary (a b : ℝ) :
  (((a - b) * a^2 < 0) → (a < b)) ∧
  (∃ a b : ℝ, (a < b) ∧ ((a - b) * a^2 ≥ 0)) :=
by sorry

end sufficient_not_necessary_l3995_399520


namespace number_always_divisible_by_396_l3995_399550

/-- Represents a permutation of digits 0 to 9 -/
def DigitPermutation := Fin 10 → Fin 10

/-- Constructs the number based on the given permutation -/
def constructNumber (p : DigitPermutation) : ℕ :=
  -- Implementation details omitted
  sorry

/-- The theorem to be proved -/
theorem number_always_divisible_by_396 (p : DigitPermutation) :
  396 ∣ constructNumber p := by
  sorry

end number_always_divisible_by_396_l3995_399550


namespace probability_at_least_one_woman_l3995_399541

def total_people : ℕ := 10
def num_men : ℕ := 7
def num_women : ℕ := 3
def selection_size : ℕ := 3

theorem probability_at_least_one_woman :
  let prob_no_women := (num_men.choose selection_size : ℚ) / (total_people.choose selection_size : ℚ)
  (1 : ℚ) - prob_no_women = 17 / 24 := by
  sorry

end probability_at_least_one_woman_l3995_399541
