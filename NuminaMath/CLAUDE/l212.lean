import Mathlib

namespace NUMINAMATH_CALUDE_existence_of_even_functions_l212_21206

open Real

theorem existence_of_even_functions :
  ∃ a : ℝ, ∀ x : ℝ,
    (fun x => x^2 + (π - a) * x) (-x) = (fun x => x^2 + (π - a) * x) x ∧
    cos (-2*x + a) = cos (2*x + a) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_even_functions_l212_21206


namespace NUMINAMATH_CALUDE_sequence_q_value_max_q_value_l212_21237

-- Define the arithmetic sequence a_n
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

-- Define the geometric sequence b_n
def geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ := b₁ * q^(n - 1)

-- Define the set E
structure E where
  m : ℕ+
  p : ℕ+
  r : ℕ+
  h_order : m < p ∧ p < r

theorem sequence_q_value
  (a₁ d b₁ q : ℝ)
  (hq : q ≠ 1 ∧ q ≠ -1)
  (h_equality : arithmetic_sequence a₁ d 1 + geometric_sequence b₁ q 2 =
                arithmetic_sequence a₁ d 2 + geometric_sequence b₁ q 3 ∧
                arithmetic_sequence a₁ d 2 + geometric_sequence b₁ q 3 =
                arithmetic_sequence a₁ d 3 + geometric_sequence b₁ q 1) :
  q = -1/2 :=
sorry

theorem max_q_value
  (a₁ d b₁ q : ℝ)
  (e : E)
  (hq : q ≠ 1 ∧ q ≠ -1)
  (h_arithmetic : ∃ (k : ℝ), k > 1 ∧ e.p = e.m + k ∧ e.r = e.p + k)
  (h_equality : arithmetic_sequence a₁ d e.m + geometric_sequence b₁ q e.p =
                arithmetic_sequence a₁ d e.p + geometric_sequence b₁ q e.r ∧
                arithmetic_sequence a₁ d e.p + geometric_sequence b₁ q e.r =
                arithmetic_sequence a₁ d e.r + geometric_sequence b₁ q e.m) :
  q ≤ -(1/2)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_sequence_q_value_max_q_value_l212_21237


namespace NUMINAMATH_CALUDE_parents_disagree_tuition_increase_l212_21218

theorem parents_disagree_tuition_increase 
  (total_parents : ℕ) 
  (agree_percentage : ℚ) 
  (h1 : total_parents = 800) 
  (h2 : agree_percentage = 20 / 100) : 
  total_parents - (total_parents * agree_percentage).floor = 640 := by
sorry

end NUMINAMATH_CALUDE_parents_disagree_tuition_increase_l212_21218


namespace NUMINAMATH_CALUDE_inequality_proof_l212_21226

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l212_21226


namespace NUMINAMATH_CALUDE_arrangement_count_l212_21294

theorem arrangement_count : ℕ := by
  -- Define the total number of people
  let total_people : ℕ := 7

  -- Define the number of boys and girls
  let num_boys : ℕ := 5
  let num_girls : ℕ := 2

  -- Define that boy A must be in the middle
  let boy_A_position : ℕ := (total_people + 1) / 2

  -- Define that the girls must be adjacent
  let girls_adjacent : Prop := true

  -- The number of ways to arrange them
  let arrangement_ways : ℕ := 192

  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l212_21294


namespace NUMINAMATH_CALUDE_derivatives_correct_l212_21257

-- Function 1
def f₁ (x : ℝ) : ℝ := 3 * x^3 - 4 * x

-- Function 2
def f₂ (x : ℝ) : ℝ := (2 * x - 1) * (3 * x + 2)

-- Function 3
def f₃ (x : ℝ) : ℝ := x^2 * (x^3 - 4)

theorem derivatives_correct :
  (∀ x, deriv f₁ x = 9 * x^2 - 4) ∧
  (∀ x, deriv f₂ x = 12 * x + 1) ∧
  (∀ x, deriv f₃ x = 5 * x^4 - 8 * x) := by
  sorry

end NUMINAMATH_CALUDE_derivatives_correct_l212_21257


namespace NUMINAMATH_CALUDE_movie_concessions_cost_l212_21210

/-- Calculates the amount spent on concessions given the total cost of a movie trip and ticket prices. -/
theorem movie_concessions_cost 
  (total_cost : ℝ) 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (adult_ticket_price : ℝ) 
  (child_ticket_price : ℝ) 
  (h1 : total_cost = 76) 
  (h2 : num_adults = 5) 
  (h3 : num_children = 2) 
  (h4 : adult_ticket_price = 10) 
  (h5 : child_ticket_price = 7) : 
  total_cost - (num_adults * adult_ticket_price + num_children * child_ticket_price) = 12 := by
sorry


end NUMINAMATH_CALUDE_movie_concessions_cost_l212_21210


namespace NUMINAMATH_CALUDE_equality_preservation_l212_21297

theorem equality_preservation (x y : ℝ) : x = y → x - 2 = y - 2 := by
  sorry

end NUMINAMATH_CALUDE_equality_preservation_l212_21297


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l212_21205

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 → -- The angles are complementary
  a = 5 * b → -- The ratio of the angles' measures is 5:1
  |a - b| = 60 := by -- The positive difference between the angles is 60°
sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l212_21205


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l212_21209

/-- Given an arithmetic progression where the sum of n terms is 2n^2 + 3n for every n,
    this function represents the r-th term of the progression. -/
def arithmeticProgressionTerm (r : ℕ) : ℕ := 4 * r + 1

/-- The sum of the first n terms of the arithmetic progression. -/
def arithmeticProgressionSum (n : ℕ) : ℕ := 2 * n^2 + 3 * n

/-- Theorem stating that the r-th term of the arithmetic progression is 4r + 1,
    given that the sum of n terms is 2n^2 + 3n for every n. -/
theorem arithmetic_progression_rth_term (r : ℕ) :
  arithmeticProgressionTerm r = arithmeticProgressionSum r - arithmeticProgressionSum (r - 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l212_21209


namespace NUMINAMATH_CALUDE_acai_berry_juice_cost_per_litre_l212_21272

/-- The cost per litre of açaí berry juice given the following conditions:
  * The superfruit juice cocktail costs $1399.45 per litre to make.
  * The mixed fruit juice costs $262.85 per litre.
  * 37 litres of mixed fruit juice are used.
  * 24.666666666666668 litres of açaí berry juice are used.
-/
theorem acai_berry_juice_cost_per_litre :
  let cocktail_cost_per_litre : ℝ := 1399.45
  let mixed_fruit_juice_cost_per_litre : ℝ := 262.85
  let mixed_fruit_juice_volume : ℝ := 37
  let acai_berry_juice_volume : ℝ := 24.666666666666668
  let total_volume : ℝ := mixed_fruit_juice_volume + acai_berry_juice_volume
  let mixed_fruit_juice_total_cost : ℝ := mixed_fruit_juice_cost_per_litre * mixed_fruit_juice_volume
  let cocktail_total_cost : ℝ := cocktail_cost_per_litre * total_volume
  let acai_berry_juice_total_cost : ℝ := cocktail_total_cost - mixed_fruit_juice_total_cost
  let acai_berry_juice_cost_per_litre : ℝ := acai_berry_juice_total_cost / acai_berry_juice_volume
  acai_berry_juice_cost_per_litre = 3105.99 :=
by
  sorry


end NUMINAMATH_CALUDE_acai_berry_juice_cost_per_litre_l212_21272


namespace NUMINAMATH_CALUDE_money_sum_l212_21238

theorem money_sum (a b : ℕ) (h1 : (1 : ℚ) / 3 * a = (1 : ℚ) / 4 * b) (h2 : b = 484) : a + b = 847 := by
  sorry

end NUMINAMATH_CALUDE_money_sum_l212_21238


namespace NUMINAMATH_CALUDE_inequalities_solution_sets_l212_21267

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := x^2 > 1
def inequality2 (x : ℝ) : Prop := -x^2 + 2*x + 3 > 0

-- Define the solution sets
def solution_set1 : Set ℝ := {x | x < -1 ∨ x > 1}
def solution_set2 : Set ℝ := {x | -1 < x ∧ x < 3}

-- Theorem statement
theorem inequalities_solution_sets :
  (∀ x : ℝ, inequality1 x ↔ x ∈ solution_set1) ∧
  (∀ x : ℝ, inequality2 x ↔ x ∈ solution_set2) :=
sorry

end NUMINAMATH_CALUDE_inequalities_solution_sets_l212_21267


namespace NUMINAMATH_CALUDE_sequence_problem_l212_21207

theorem sequence_problem (n : ℕ+) (b : ℕ → ℝ)
  (h0 : b 0 = 25)
  (h1 : b 1 = 56)
  (hn : b n = 0)
  (hk : ∀ k : ℕ, 1 ≤ k → k < n → b (k + 1) = b (k - 1) - 7 / b k) :
  n = 201 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l212_21207


namespace NUMINAMATH_CALUDE_orthocenter_on_vertex_implies_right_angled_l212_21291

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The orthocenter of a triangle --/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Predicate to check if a point is a vertex of a triangle --/
def is_vertex (p : ℝ × ℝ) (t : Triangle) : Prop :=
  p = t.A ∨ p = t.B ∨ p = t.C

/-- Predicate to check if a triangle is right-angled --/
def is_right_angled (t : Triangle) : Prop := sorry

/-- Theorem: If the orthocenter of a triangle coincides with one of its vertices,
    then the triangle is right-angled --/
theorem orthocenter_on_vertex_implies_right_angled (t : Triangle) :
  is_vertex (orthocenter t) t → is_right_angled t := by sorry

end NUMINAMATH_CALUDE_orthocenter_on_vertex_implies_right_angled_l212_21291


namespace NUMINAMATH_CALUDE_circle_symmetry_l212_21296

/-- The equation of a circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Symmetry of a circle with respect to the origin -/
def symmetric_to_origin (c1 c2 : Circle) : Prop :=
  c1.center.1 = -c2.center.1 ∧ c1.center.2 = -c2.center.2 ∧ c1.radius = c2.radius

theorem circle_symmetry (c : Circle) :
  symmetric_to_origin c ⟨(-2, 1), 1⟩ →
  c = ⟨(2, -1), 1⟩ := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetry_l212_21296


namespace NUMINAMATH_CALUDE_dylan_ice_cube_trays_l212_21242

/-- The number of ice cube trays Dylan needs to fill -/
def num_trays_to_fill (glass_cubes : ℕ) (pitcher_multiplier : ℕ) (tray_capacity : ℕ) : ℕ :=
  ((glass_cubes + glass_cubes * pitcher_multiplier) + tray_capacity - 1) / tray_capacity

/-- Theorem stating that Dylan needs to fill 2 ice cube trays -/
theorem dylan_ice_cube_trays : 
  num_trays_to_fill 8 2 12 = 2 := by
  sorry

#eval num_trays_to_fill 8 2 12

end NUMINAMATH_CALUDE_dylan_ice_cube_trays_l212_21242


namespace NUMINAMATH_CALUDE_min_value_of_f_l212_21235

def f (x : ℝ) := x^3 - 3*x + 1

theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ f x = -1 ∧ ∀ y ∈ Set.Icc 0 3, f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l212_21235


namespace NUMINAMATH_CALUDE_line_passes_through_K_min_distance_AC_dot_product_range_l212_21228

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 12

-- Define the line l
def line_l (m x y : ℝ) : Prop := m*x - y - 2*m + 3 = 0

-- Define the point K
def point_K : ℝ × ℝ := (2, 3)

-- Define the intersection points A and C
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ circle_M x y ∧ line_l m x y}

-- Theorem 1: Line l passes through point K for all m
theorem line_passes_through_K (m : ℝ) : line_l m (point_K.1) (point_K.2) :=
sorry

-- Theorem 2: Minimum distance between intersection points is 4
theorem min_distance_AC :
  ∃ (m : ℝ), ∀ (A C : ℝ × ℝ), A ∈ intersection_points m → C ∈ intersection_points m →
  A ≠ C → ‖A - C‖ ≥ 4 :=
sorry

-- Theorem 3: Range of dot product MA · MC
theorem dot_product_range (M : ℝ × ℝ) (m : ℝ) :
  M = (4, 5) →
  ∀ (A C : ℝ × ℝ), A ∈ intersection_points m → C ∈ intersection_points m →
  -12 ≤ (A - M) • (C - M) ∧ (A - M) • (C - M) ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_K_min_distance_AC_dot_product_range_l212_21228


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l212_21270

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 4*x + 4 ≥ 0) ↔ (∃ x : ℝ, x^2 - 4*x + 4 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l212_21270


namespace NUMINAMATH_CALUDE_pipe_filling_time_l212_21258

/-- Given two pipes A and B that can fill a tank, this theorem proves the time
    it takes for pipe B to fill the tank alone, given the times for pipe A
    and both pipes together. -/
theorem pipe_filling_time (time_A time_both : ℝ) (h1 : time_A = 10)
    (h2 : time_both = 20 / 3) : 
    (1 / time_A + 1 / 20 = 1 / time_both) := by
  sorry

#check pipe_filling_time

end NUMINAMATH_CALUDE_pipe_filling_time_l212_21258


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l212_21212

theorem quadratic_roots_sum_bound (p : ℝ) (r₁ r₂ : ℝ) : 
  (r₁ ≠ r₂) →
  (r₁^2 + p*r₁ + 7 = 0) →
  (r₂^2 + p*r₂ + 7 = 0) →
  |r₁ + r₂| > 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l212_21212


namespace NUMINAMATH_CALUDE_henry_final_book_count_l212_21234

/-- Calculates the final number of books Henry has after decluttering and acquiring new ones. -/
def final_book_count (initial_books : ℕ) (boxed_books : ℕ) (room_books : ℕ) 
  (coffee_table_books : ℕ) (cookbooks : ℕ) (new_books : ℕ) : ℕ :=
  initial_books - (3 * boxed_books + room_books + coffee_table_books + cookbooks) + new_books

/-- Theorem stating that Henry ends up with 23 books after the process. -/
theorem henry_final_book_count : 
  final_book_count 99 15 21 4 18 12 = 23 := by
  sorry

end NUMINAMATH_CALUDE_henry_final_book_count_l212_21234


namespace NUMINAMATH_CALUDE_invitations_per_package_l212_21299

theorem invitations_per_package (friends : ℕ) (packs : ℕ) (h1 : friends = 10) (h2 : packs = 5) :
  friends / packs = 2 := by
sorry

end NUMINAMATH_CALUDE_invitations_per_package_l212_21299


namespace NUMINAMATH_CALUDE_larger_cube_surface_area_l212_21283

theorem larger_cube_surface_area (small_cube_surface_area : ℝ) (num_small_cubes : ℕ) :
  small_cube_surface_area = 24 →
  num_small_cubes = 125 →
  ∃ (larger_cube_surface_area : ℝ), larger_cube_surface_area = 600 := by
  sorry

end NUMINAMATH_CALUDE_larger_cube_surface_area_l212_21283


namespace NUMINAMATH_CALUDE_final_sum_theorem_l212_21254

theorem final_sum_theorem (a b S : ℝ) (h : a + b = S) :
  2 * (a + 3) + 2 * (b + 3) = 2 * S + 12 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_theorem_l212_21254


namespace NUMINAMATH_CALUDE_sin_cos_identity_l212_21230

theorem sin_cos_identity (α : Real) (h : Real.sin α ^ 2 + Real.sin α = 1) :
  Real.cos α ^ 4 + Real.cos α ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l212_21230


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l212_21293

def total_people : ℕ := 15
def num_men : ℕ := 10
def num_women : ℕ := 5
def selected : ℕ := 5

theorem probability_at_least_one_woman :
  let p := 1 - (num_men.choose selected / total_people.choose selected)
  p = 917 / 1001 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l212_21293


namespace NUMINAMATH_CALUDE_triangle_properties_l212_21251

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sin t.A + Real.sqrt 3 * Real.cos t.A = 0)
  (h2 : t.a = 2 * Real.sqrt 7)
  (h3 : t.b = 2) :
  t.A = 2 * Real.pi / 3 ∧ 
  t.c = 4 ∧ 
  1/2 * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l212_21251


namespace NUMINAMATH_CALUDE_dave_ice_cubes_l212_21240

/-- Given that Dave started with 2 ice cubes and ended with 9 ice cubes in total,
    prove that he made 7 additional ice cubes. -/
theorem dave_ice_cubes (initial : Nat) (final : Nat) (h1 : initial = 2) (h2 : final = 9) :
  final - initial = 7 := by
  sorry

end NUMINAMATH_CALUDE_dave_ice_cubes_l212_21240


namespace NUMINAMATH_CALUDE_tram_route_difference_l212_21278

/-- Represents a point on the circular tram line -/
inductive TramStop
| Circus
| Park
| Zoo

/-- Represents the distance between two points on the tram line -/
def distance (a b : TramStop) : ℝ := sorry

/-- The total circumference of the tram line -/
def circumference : ℝ := sorry

theorem tram_route_difference :
  let park_to_zoo := distance TramStop.Park TramStop.Zoo
  let park_to_circus_via_zoo := distance TramStop.Park TramStop.Zoo + distance TramStop.Zoo TramStop.Circus
  let park_to_circus_direct := distance TramStop.Park TramStop.Circus
  
  -- The distance from Park to Zoo via Circus is three times longer than the direct route
  distance TramStop.Park TramStop.Zoo + distance TramStop.Zoo TramStop.Circus + distance TramStop.Circus TramStop.Park = 3 * park_to_zoo →
  
  -- The distance from Circus to Zoo via Park is half as long as the direct route
  distance TramStop.Circus TramStop.Park + park_to_zoo = (1/2) * distance TramStop.Circus TramStop.Zoo →
  
  -- The difference between the longer and shorter routes from Park to Circus is 1/12 of the total circumference
  park_to_circus_via_zoo - park_to_circus_direct = (1/12) * circumference :=
by sorry

end NUMINAMATH_CALUDE_tram_route_difference_l212_21278


namespace NUMINAMATH_CALUDE_cupcakes_sold_correct_l212_21215

/-- Represents the number of cupcakes Katie sold at the bake sale -/
def cupcakes_sold (initial : ℕ) (additional : ℕ) (remaining : ℕ) : ℕ :=
  initial + additional - remaining

/-- Proves that the number of cupcakes sold is correct given the initial,
    additional, and remaining cupcakes -/
theorem cupcakes_sold_correct (initial : ℕ) (additional : ℕ) (remaining : ℕ) :
  cupcakes_sold initial additional remaining = initial + additional - remaining :=
by sorry

end NUMINAMATH_CALUDE_cupcakes_sold_correct_l212_21215


namespace NUMINAMATH_CALUDE_polynomial_sum_equality_l212_21223

theorem polynomial_sum_equality : 
  let p (x : ℝ) := 4 * x^2 - 2 * x + 1
  let q (x : ℝ) := -3 * x^2 + x - 5
  let r (x : ℝ) := 2 * x^2 - 4 * x + 3
  ∀ x, p x + q x + r x = 3 * x^2 - 5 * x - 1 := by
    sorry

end NUMINAMATH_CALUDE_polynomial_sum_equality_l212_21223


namespace NUMINAMATH_CALUDE_equation_solution_expression_simplification_l212_21269

-- Part 1: Equation solution
theorem equation_solution (x : ℝ) :
  (x + 3) / (x - 3) - 4 / (x + 3) = 1 ↔ x = -15 :=
sorry

-- Part 2: Expression simplification
theorem expression_simplification (x : ℝ) (h : x ≠ 2) (h' : x ≠ -3) :
  (x - 3) / (x - 2) / (x + 2 - 5 / (x - 2)) = 1 / (x + 3) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_expression_simplification_l212_21269


namespace NUMINAMATH_CALUDE_new_person_weight_l212_21246

/-- Given a group of 8 people where one person weighing 65 kg is replaced by a new person,
    and the average weight of the group increases by 4.2 kg,
    prove that the weight of the new person is 98.6 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_increase : Real) (replaced_weight : Real) :
  initial_count = 8 →
  weight_increase = 4.2 →
  replaced_weight = 65 →
  (initial_count : Real) * weight_increase + replaced_weight = 98.6 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l212_21246


namespace NUMINAMATH_CALUDE_polynomial_equality_l212_21241

-- Define the theorem
theorem polynomial_equality (n : ℕ) (f g : ℝ → ℝ) (x : Fin (n + 1) → ℝ) :
  (∀ (k : Fin (n + 1)), (deriv^[k] f) (x k) = (deriv^[k] g) (x k)) →
  (∀ (y : ℝ), f y = g y) :=
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l212_21241


namespace NUMINAMATH_CALUDE_table_tennis_match_probability_l212_21265

/-- The probability of Player A winning a single game -/
def p_A : ℝ := 0.6

/-- The probability of Player B winning a single game -/
def p_B : ℝ := 0.4

/-- The probability of Player A winning the match in a best-of-three format -/
def p_A_wins_match : ℝ := p_A * p_A + p_A * p_B * p_A + p_B * p_A * p_A

theorem table_tennis_match_probability :
  p_A + p_B = 1 →
  p_A_wins_match = 0.648 := by
  sorry

end NUMINAMATH_CALUDE_table_tennis_match_probability_l212_21265


namespace NUMINAMATH_CALUDE_wheat_bread_served_l212_21273

/-- The number of loaves of wheat bread served at a restaurant -/
def wheat_bread : ℝ := 0.9 - 0.4

/-- The total number of loaves served at the restaurant -/
def total_loaves : ℝ := 0.9

/-- The number of loaves of white bread served at the restaurant -/
def white_bread : ℝ := 0.4

/-- Theorem stating that the number of loaves of wheat bread served is 0.5 -/
theorem wheat_bread_served : wheat_bread = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_wheat_bread_served_l212_21273


namespace NUMINAMATH_CALUDE_sarahs_trip_distance_l212_21284

theorem sarahs_trip_distance :
  ∀ y : ℚ, (y / 4 + 25 + y / 6 = y) → y = 300 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_trip_distance_l212_21284


namespace NUMINAMATH_CALUDE_four_zeros_when_a_positive_l212_21256

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then a * x + 1 else Real.log x / Real.log 3

def F (a : ℝ) (x : ℝ) : ℝ :=
  f a (f a x) + 1

theorem four_zeros_when_a_positive (a : ℝ) (h : a > 0) :
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    F a x₁ = 0 ∧ F a x₂ = 0 ∧ F a x₃ = 0 ∧ F a x₄ = 0 ∧
    ∀ x, F a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄ :=
sorry

end

end NUMINAMATH_CALUDE_four_zeros_when_a_positive_l212_21256


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_line_equation_l212_21290

/-- A line that forms an isosceles right-angled triangle with the coordinate axes -/
structure IsoscelesRightTriangleLine where
  a : ℝ
  eq : (x y : ℝ) → Prop
  passes_through : eq 2 3
  isosceles_right : ∀ (x y : ℝ), eq x y → (x / a + y / a = 1) ∨ (x / a + y / (-a) = 1)

/-- The equation of the line is either x + y - 5 = 0 or x - y + 1 = 0 -/
theorem isosceles_right_triangle_line_equation (l : IsoscelesRightTriangleLine) :
  (∀ x y, l.eq x y ↔ x + y - 5 = 0) ∨ (∀ x y, l.eq x y ↔ x - y + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_line_equation_l212_21290


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l212_21263

theorem fraction_to_decimal : (63 : ℚ) / (2^3 * 5^4) = 0.0126 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l212_21263


namespace NUMINAMATH_CALUDE_hospital_staff_count_l212_21281

theorem hospital_staff_count (total : ℕ) (doctor_ratio nurse_ratio : ℕ) 
  (h1 : total = 500)
  (h2 : doctor_ratio = 7)
  (h3 : nurse_ratio = 8) : 
  ∃ (nurses : ℕ), nurses = 264 ∧ 
    ∃ (doctors : ℕ), doctors + nurses = total ∧ 
      doctor_ratio * nurses = nurse_ratio * doctors :=
sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l212_21281


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l212_21201

theorem least_positive_integer_multiple_of_53 :
  ∀ x : ℕ+, x < 21 → ¬(∃ k : ℤ, (3*x)^2 + 2*43*3*x + 43^2 = 53*k) ∧
  ∃ k : ℤ, (3*21)^2 + 2*43*3*21 + 43^2 = 53*k :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l212_21201


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l212_21250

theorem rectangle_area_proof (l w : ℝ) : 
  (l + 3.5) * (w - 1.5) = l * w ∧ 
  (l - 3.5) * (w + 2) = l * w → 
  l * w = 630 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l212_21250


namespace NUMINAMATH_CALUDE_perpendicular_planes_line_l212_21282

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_line 
  (a : Line) (α β : Plane) (l : Line)
  (h1 : perp_planes α β)
  (h2 : l = intersect α β)
  (h3 : perp_line_plane a β) :
  subset a α ∧ perp_lines a l :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_line_l212_21282


namespace NUMINAMATH_CALUDE_base_is_twelve_l212_21222

/-- Represents a number system with a given base -/
structure NumberSystem where
  base : ℕ
  base_gt_5 : base > 5

/-- Converts a number from base b to decimal -/
def to_decimal (n : ℕ) (b : ℕ) : ℕ :=
  (n / 10) * b + (n % 10)

/-- Theorem: In a number system where the square of 24 is 554, the base of the system is 12 -/
theorem base_is_twelve (ns : NumberSystem) 
  (h : (to_decimal 24 ns.base)^2 = to_decimal 554 ns.base) : 
  ns.base = 12 := by
  sorry


end NUMINAMATH_CALUDE_base_is_twelve_l212_21222


namespace NUMINAMATH_CALUDE_petya_more_likely_to_win_l212_21287

/-- Represents the game setup with two boxes of candies --/
structure CandyGame where
  total_candies : ℕ
  prob_two_caramels : ℝ

/-- Defines the game rules --/
def game : CandyGame :=
  { total_candies := 25
  , prob_two_caramels := 0.54 }

/-- Calculates the probability of Vasya winning --/
def vasya_win_prob (g : CandyGame) : ℝ :=
  1 - g.prob_two_caramels

/-- Calculates the probability of Petya winning --/
def petya_win_prob (g : CandyGame) : ℝ :=
  1 - vasya_win_prob g

/-- Theorem stating that Petya has a higher chance of winning --/
theorem petya_more_likely_to_win :
  petya_win_prob game > vasya_win_prob game :=
sorry

end NUMINAMATH_CALUDE_petya_more_likely_to_win_l212_21287


namespace NUMINAMATH_CALUDE_simplify_expression_find_k_l212_21227

-- Problem 1
theorem simplify_expression (x : ℝ) :
  (2*x + 1)^2 - (2*x + 1)*(2*x - 1) + (x + 1)*(x - 3) = x^2 + 2*x - 1 := by
  sorry

-- Problem 2
theorem find_k (x y k : ℝ) 
  (eq1 : x + y = 1)
  (eq2 : k*x + (k - 1)*y = 7)
  (eq3 : 3*x - 2*y = 5) :
  k = 33/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_find_k_l212_21227


namespace NUMINAMATH_CALUDE_derivative_of_sin_cubed_inverse_l212_21224

noncomputable def f (x : ℝ) : ℝ := Real.sin (1 / x) ^ 3

theorem derivative_of_sin_cubed_inverse (x : ℝ) (hx : x ≠ 0) :
  deriv f x = -3 / x^2 * Real.sin (1 / x)^2 * Real.cos (1 / x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_sin_cubed_inverse_l212_21224


namespace NUMINAMATH_CALUDE_sqrt_7_simplest_l212_21261

-- Define the concept of simplest square root
def is_simplest_sqrt (x : ℝ) : Prop :=
  x > 0 ∧ ∀ y : ℝ, y > 0 → y^2 = x → y = Real.sqrt x

-- Define the set of square roots to compare
def sqrt_set : Set ℝ := {Real.sqrt 24, Real.sqrt (1/3), Real.sqrt 7, Real.sqrt 0.2}

-- Theorem statement
theorem sqrt_7_simplest :
  ∀ x ∈ sqrt_set, x ≠ Real.sqrt 7 → ¬(is_simplest_sqrt x) ∧ is_simplest_sqrt (Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_7_simplest_l212_21261


namespace NUMINAMATH_CALUDE_odd_function_sum_zero_l212_21233

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_sum_zero (f : ℝ → ℝ) (h : OddFunction f) :
  f (-2) + f (-1) + f 0 + f 1 + f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_zero_l212_21233


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_edge_sum_l212_21288

theorem rectangular_parallelepiped_edge_sum (a b c : ℕ) (V : ℕ) : 
  V = a * b * c → 
  V.Prime → 
  V > 2 → 
  Odd (a + b + c) := by
sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_edge_sum_l212_21288


namespace NUMINAMATH_CALUDE_opposite_roots_iff_ab_eq_c_l212_21211

-- Define the cubic polynomial f(x) = x^3 + a x^2 + b x + c
def f (a b c x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

-- Define a predicate for when two roots are opposite numbers
def has_opposite_roots (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, f a b c x = 0 ∧ f a b c y = 0 ∧ y = -x

-- State the theorem
theorem opposite_roots_iff_ab_eq_c (a b c : ℝ) (h : b ≤ 0) :
  has_opposite_roots a b c ↔ a * b = c :=
sorry

end NUMINAMATH_CALUDE_opposite_roots_iff_ab_eq_c_l212_21211


namespace NUMINAMATH_CALUDE_hari_join_time_l212_21213

/-- Represents the number of months in a year -/
def monthsInYear : ℕ := 12

/-- Praveen's initial investment in rupees -/
def praveenInvestment : ℚ := 3500

/-- Hari's investment in rupees -/
def hariInvestment : ℚ := 9000.000000000002

/-- Profit sharing ratio for Praveen -/
def praveenShare : ℚ := 2

/-- Profit sharing ratio for Hari -/
def hariShare : ℚ := 3

/-- Theorem stating when Hari joined the business -/
theorem hari_join_time : 
  ∃ (x : ℕ), x < monthsInYear ∧ 
  (praveenInvestment * monthsInYear) / (hariInvestment * (monthsInYear - x)) = praveenShare / hariShare ∧
  x = 5 := by sorry

end NUMINAMATH_CALUDE_hari_join_time_l212_21213


namespace NUMINAMATH_CALUDE_line_point_sum_l212_21208

/-- The line equation y = -2/5 * x + 10 -/
def line_equation (x y : ℝ) : Prop := y = -2/5 * x + 10

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (25, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 10)

/-- Point T is on line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
  r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- The area of triangle POQ is 4 times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop := 
  abs ((P.1 * Q.2 - Q.1 * P.2) / 2) = 4 * abs ((P.1 * s - r * P.2) / 2)

/-- Main theorem -/
theorem line_point_sum (r s : ℝ) 
  (h1 : line_equation r s) 
  (h2 : T_on_PQ r s) 
  (h3 : area_condition r s) : 
  r + s = 21.25 := by sorry

end NUMINAMATH_CALUDE_line_point_sum_l212_21208


namespace NUMINAMATH_CALUDE_total_amount_proof_l212_21232

/-- Given an amount divided into two parts, where one part is invested at 3% and the other at 5%,
    this theorem proves that the total amount is 4000 when the first part is 2800 and
    the total annual interest is 144. -/
theorem total_amount_proof (T A : ℝ) : 
  A = 2800 → 
  0.03 * A + 0.05 * (T - A) = 144 → 
  T = 4000 := by
sorry

end NUMINAMATH_CALUDE_total_amount_proof_l212_21232


namespace NUMINAMATH_CALUDE_triple_sharp_58_l212_21243

def sharp (N : ℝ) : ℝ := 0.5 * N + 1

theorem triple_sharp_58 : sharp (sharp (sharp 58)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_triple_sharp_58_l212_21243


namespace NUMINAMATH_CALUDE_part1_part2_l212_21264

-- Define propositions p, q, and r as functions of a
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + (a - 1) * x + a^2 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (2 * a^2 - a)^x < (2 * a^2 - a)^y

def r (a : ℝ) : Prop := (2 * a - 1) / (a - 2) ≤ 1

-- Define the range of a for part 1
def range_a (a : ℝ) : Prop := (a ≥ -1 ∧ a < -1/2) ∨ (a > 1/3 ∧ a ≤ 1)

-- Theorem for part 1
theorem part1 (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_a a := by sorry

-- Theorem for part 2
theorem part2 : (∀ a : ℝ, ¬(p a) → r a) ∧ ¬(∀ a : ℝ, r a → ¬(p a)) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l212_21264


namespace NUMINAMATH_CALUDE_probability_yellow_ball_l212_21249

/-- The probability of drawing a yellow ball from a bag containing white and yellow balls -/
theorem probability_yellow_ball (total_balls : ℕ) (white_balls yellow_balls : ℕ) 
  (h1 : total_balls = white_balls + yellow_balls)
  (h2 : total_balls > 0)
  (h3 : white_balls = 2)
  (h4 : yellow_balls = 3) :
  (yellow_balls : ℚ) / total_balls = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_yellow_ball_l212_21249


namespace NUMINAMATH_CALUDE_triangle_inequality_l212_21266

theorem triangle_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  Real.sqrt (3 * (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a))) ≥ 
  Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l212_21266


namespace NUMINAMATH_CALUDE_trapezoid_division_l212_21225

/-- Represents a trapezoid with the given side lengths -/
structure Trapezoid where
  short_base : ℝ
  long_base : ℝ
  side1 : ℝ
  side2 : ℝ

/-- Represents a point that divides a line segment -/
structure DivisionPoint where
  ratio : ℝ

/-- 
Given a trapezoid with parallel sides of length 3 and 9, and non-parallel sides of length 4 and 6,
if a line parallel to the bases divides the trapezoid into two trapezoids of equal perimeters,
then this line divides each of the non-parallel sides in the ratio 3:2.
-/
theorem trapezoid_division (t : Trapezoid) (d : DivisionPoint) : 
  t.short_base = 3 ∧ t.long_base = 9 ∧ t.side1 = 4 ∧ t.side2 = 6 →
  (t.long_base - t.short_base) * d.ratio + t.short_base = 
    (t.side1 * d.ratio + t.side2 * d.ratio) / 2 →
  d.ratio = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_division_l212_21225


namespace NUMINAMATH_CALUDE_smallest_multiple_l212_21292

theorem smallest_multiple (n : ℕ) : n = 204 ↔ 
  n > 0 ∧ 
  17 ∣ n ∧ 
  n % 43 = 11 ∧ 
  ∀ m : ℕ, m > 0 → 17 ∣ m → m % 43 = 11 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l212_21292


namespace NUMINAMATH_CALUDE_g_sum_zero_l212_21204

/-- The function g(x) = x^2 - 2013x -/
def g (x : ℝ) : ℝ := x^2 - 2013*x

/-- 
If g(a) = g(b) and a ≠ b, then g(a + b) = 0
-/
theorem g_sum_zero (a b : ℝ) (h1 : g a = g b) (h2 : a ≠ b) : g (a + b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_zero_l212_21204


namespace NUMINAMATH_CALUDE_cubic_sum_product_l212_21217

theorem cubic_sum_product (a b c : ℝ) 
  (h1 : a + b + c = 5)
  (h2 : a^2 + b^2 + c^2 = 15)
  (h3 : a^3 + b^3 + c^3 = 47) :
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) = 625 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_product_l212_21217


namespace NUMINAMATH_CALUDE_longestAltitudesSum_eq_17_l212_21277

/-- A triangle with sides 5, 12, and 13 -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 5
  hb : b = 12
  hc : c = 13

/-- The sum of the lengths of the two longest altitudes in the special triangle -/
def longestAltitudesSum (t : SpecialTriangle) : ℝ := sorry

/-- Theorem stating that the sum of the lengths of the two longest altitudes is 17 -/
theorem longestAltitudesSum_eq_17 (t : SpecialTriangle) : longestAltitudesSum t = 17 := by sorry

end NUMINAMATH_CALUDE_longestAltitudesSum_eq_17_l212_21277


namespace NUMINAMATH_CALUDE_composite_evaluation_l212_21202

/-- A polynomial with coefficients either 0 or 1 -/
def BinaryPolynomial (P : Polynomial ℤ) : Prop :=
  ∀ i, P.coeff i = 0 ∨ P.coeff i = 1

/-- A polynomial is nonconstant -/
def IsNonconstant (P : Polynomial ℤ) : Prop :=
  ∃ i > 0, P.coeff i ≠ 0

theorem composite_evaluation
  (P : Polynomial ℤ)
  (h_binary : BinaryPolynomial P)
  (h_factorizable : ∃ (f g : Polynomial ℤ), P = f * g ∧ IsNonconstant f ∧ IsNonconstant g) :
  ∃ (a b : ℤ), a > 1 ∧ b > 1 ∧ P.eval 2 = a * b :=
sorry

end NUMINAMATH_CALUDE_composite_evaluation_l212_21202


namespace NUMINAMATH_CALUDE_happy_snakes_not_purple_l212_21262

structure Snake where
  happy : Bool
  purple : Bool
  canAdd : Bool
  canSubtract : Bool

def TomSnakes : Set Snake := sorry

theorem happy_snakes_not_purple :
  ∀ s ∈ TomSnakes,
  (s.happy → s.canAdd) ∧
  (s.purple → ¬s.canSubtract) ∧
  (¬s.canSubtract → ¬s.canAdd) →
  (s.happy → ¬s.purple) := by
  sorry

#check happy_snakes_not_purple

end NUMINAMATH_CALUDE_happy_snakes_not_purple_l212_21262


namespace NUMINAMATH_CALUDE_triangular_prism_sum_l212_21285

/-- A triangular prism is a three-dimensional shape with two triangular bases and three rectangular faces. -/
structure TriangularPrism where
  -- We don't need to define the internal structure, just the concept

/-- The number of faces in a triangular prism -/
def num_faces (prism : TriangularPrism) : ℕ := 5

/-- The number of edges in a triangular prism -/
def num_edges (prism : TriangularPrism) : ℕ := 9

/-- The number of vertices in a triangular prism -/
def num_vertices (prism : TriangularPrism) : ℕ := 6

/-- Theorem: The sum of the number of faces, edges, and vertices of a triangular prism is 20 -/
theorem triangular_prism_sum (prism : TriangularPrism) :
  num_faces prism + num_edges prism + num_vertices prism = 20 := by
  sorry

end NUMINAMATH_CALUDE_triangular_prism_sum_l212_21285


namespace NUMINAMATH_CALUDE_max_profit_is_2180_l212_21236

/-- Represents the production plan for items A and B -/
structure ProductionPlan where
  itemA : ℕ
  itemB : ℕ

/-- Calculates the profit for a given production plan -/
def profit (plan : ProductionPlan) : ℕ :=
  80 * plan.itemA + 100 * plan.itemB

/-- Checks if a production plan is feasible given the resource constraints -/
def isFeasible (plan : ProductionPlan) : Prop :=
  10 * plan.itemA + 70 * plan.itemB ≤ 700 ∧
  23 * plan.itemA + 40 * plan.itemB ≤ 642

/-- Theorem stating that the maximum profit is 2180 thousand rubles -/
theorem max_profit_is_2180 :
  ∃ (optimalPlan : ProductionPlan),
    isFeasible optimalPlan ∧
    profit optimalPlan = 2180 ∧
    ∀ (plan : ProductionPlan), isFeasible plan → profit plan ≤ 2180 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_is_2180_l212_21236


namespace NUMINAMATH_CALUDE_rhombus_diagonals_perpendicular_converse_is_false_inverse_is_false_contrapositive_l212_21275

-- Define a type for quadrilaterals
structure Quadrilateral where
  -- Add necessary fields

-- Define what it means for a quadrilateral to be a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  sorry

-- Define what it means for diagonals to be perpendicular
def diagonals_perpendicular (q : Quadrilateral) : Prop :=
  sorry

-- The original statement
theorem rhombus_diagonals_perpendicular :
  ∀ q : Quadrilateral, is_rhombus q → diagonals_perpendicular q :=
sorry

-- The converse (which is false)
theorem converse_is_false :
  ¬(∀ q : Quadrilateral, diagonals_perpendicular q → is_rhombus q) :=
sorry

-- The inverse (which is false)
theorem inverse_is_false :
  ¬(∀ q : Quadrilateral, ¬is_rhombus q → ¬diagonals_perpendicular q) :=
sorry

-- The contrapositive (which is true)
theorem contrapositive :
  ∀ q : Quadrilateral, ¬diagonals_perpendicular q → ¬is_rhombus q :=
sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_perpendicular_converse_is_false_inverse_is_false_contrapositive_l212_21275


namespace NUMINAMATH_CALUDE_courtney_marble_count_l212_21214

/-- The number of marbles in Courtney's collection -/
def total_marbles (jar1 jar2 jar3 : ℕ) : ℕ := jar1 + jar2 + jar3

/-- Theorem: Courtney's total marble count -/
theorem courtney_marble_count :
  ∀ (jar1 jar2 jar3 : ℕ),
    jar1 = 80 →
    jar2 = 2 * jar1 →
    jar3 = jar1 / 4 →
    total_marbles jar1 jar2 jar3 = 260 := by
  sorry

#check courtney_marble_count

end NUMINAMATH_CALUDE_courtney_marble_count_l212_21214


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l212_21252

theorem quadratic_roots_sum (m n : ℝ) : 
  m^2 + 2*m - 2022 = 0 → 
  n^2 + 2*n - 2022 = 0 → 
  m^2 + 3*m + n = 2020 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l212_21252


namespace NUMINAMATH_CALUDE_intersection_M_N_l212_21279

def M : Set ℝ := {x | ∃ t : ℝ, x = Real.exp (-t * Real.log 2)}
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l212_21279


namespace NUMINAMATH_CALUDE_probability_two_boys_or_two_girls_l212_21298

/-- The probability of selecting either two boys or two girls from a group of 5 students -/
theorem probability_two_boys_or_two_girls (total_students : ℕ) (num_boys : ℕ) (num_girls : ℕ) :
  total_students = 5 →
  num_boys = 2 →
  num_girls = 3 →
  (Nat.choose num_girls 2 + Nat.choose num_boys 2) / Nat.choose total_students 2 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_boys_or_two_girls_l212_21298


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l212_21203

-- Define the conditions
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x - a ≤ 0
def q (a : ℝ) : Prop := a > 0 ∨ a < -1

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ a : ℝ, q a → p a) ∧ 
  (∃ a : ℝ, p a ∧ ¬(q a)) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l212_21203


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l212_21216

def f (x : ℝ) : ℝ := x^2 - 1

theorem f_even_and_increasing :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l212_21216


namespace NUMINAMATH_CALUDE_strawberries_eaten_l212_21200

-- Define the initial number of strawberries
def initial_strawberries : ℕ := 35

-- Define the remaining number of strawberries
def remaining_strawberries : ℕ := 33

-- Theorem to prove
theorem strawberries_eaten : initial_strawberries - remaining_strawberries = 2 := by
  sorry

end NUMINAMATH_CALUDE_strawberries_eaten_l212_21200


namespace NUMINAMATH_CALUDE_price_calculation_l212_21289

/-- Calculates the total price for jewelry and paintings after a price increase -/
def total_price (
  original_jewelry_price : ℕ
  ) (original_painting_price : ℕ
  ) (jewelry_price_increase : ℕ
  ) (painting_price_increase_percent : ℕ
  ) (jewelry_quantity : ℕ
  ) (painting_quantity : ℕ
  ) : ℕ :=
  let new_jewelry_price := original_jewelry_price + jewelry_price_increase
  let new_painting_price := original_painting_price + (original_painting_price * painting_price_increase_percent) / 100
  (new_jewelry_price * jewelry_quantity) + (new_painting_price * painting_quantity)

theorem price_calculation :
  total_price 30 100 10 20 2 5 = 680 := by
  sorry

end NUMINAMATH_CALUDE_price_calculation_l212_21289


namespace NUMINAMATH_CALUDE_meaningful_fraction_l212_21248

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = (x + 3) / (x - 4)) ↔ x ≠ 4 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l212_21248


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l212_21245

theorem circle_area_from_circumference (C : ℝ) (h : C = 36) :
  let r := C / (2 * Real.pi)
  (Real.pi * r^2) = 324 / Real.pi := by sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l212_21245


namespace NUMINAMATH_CALUDE_rectangle_area_l212_21286

theorem rectangle_area (square_area : ℝ) (rectangle_length_factor : ℝ) : 
  square_area = 64 →
  rectangle_length_factor = 3 →
  let square_side := Real.sqrt square_area
  let rectangle_width := square_side
  let rectangle_length := rectangle_length_factor * rectangle_width
  rectangle_width * rectangle_length = 192 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l212_21286


namespace NUMINAMATH_CALUDE_max_distance_between_sine_cosine_graphs_l212_21220

theorem max_distance_between_sine_cosine_graphs : 
  ∃ (C : ℝ), C = 4 ∧ ∀ m : ℝ, |2 * Real.sin m - 2 * Real.sqrt 3 * Real.cos m| ≤ C ∧ 
  ∃ m : ℝ, |2 * Real.sin m - 2 * Real.sqrt 3 * Real.cos m| = C :=
sorry

end NUMINAMATH_CALUDE_max_distance_between_sine_cosine_graphs_l212_21220


namespace NUMINAMATH_CALUDE_sphere_polyhedra_radii_ratio_l212_21259

/-- The ratio of radii for a sequence of spheres inscribed in and circumscribed around
    regular polyhedra (octahedron, icosahedron, dodecahedron, tetrahedron, hexahedron) -/
theorem sphere_polyhedra_radii_ratio :
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ),
    r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧ r₄ > 0 ∧ r₅ > 0 ∧ r₆ > 0 ∧
    (r₂ / r₁ = Real.sqrt (9 + 4 * Real.sqrt 5)) ∧
    (r₃ / r₂ = Real.sqrt (27 + 12 * Real.sqrt 5)) ∧
    (r₄ / r₃ = 3 * Real.sqrt (5 + 2 * Real.sqrt 5)) ∧
    (r₅ / r₄ = 3 * Real.sqrt 15) ∧
    (r₆ / r₅ = 3 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_sphere_polyhedra_radii_ratio_l212_21259


namespace NUMINAMATH_CALUDE_only_eq1_has_zero_constant_term_l212_21255

-- Define the equations
def eq1 (x : ℝ) := x^2 + x = 0
def eq2 (x : ℝ) := 2*x^2 - x - 12 = 0
def eq3 (x : ℝ) := 2*(x^2 - 1) = 3*(x - 1)
def eq4 (x : ℝ) := 2*(x^2 + 1) = x + 4

-- Define a function to check if an equation has zero constant term
def has_zero_constant_term (eq : ℝ → Prop) : Prop :=
  ∃ a b : ℝ, ∀ x, eq x ↔ a*x^2 + b*x = 0

-- Theorem statement
theorem only_eq1_has_zero_constant_term :
  has_zero_constant_term eq1 ∧
  ¬has_zero_constant_term eq2 ∧
  ¬has_zero_constant_term eq3 ∧
  ¬has_zero_constant_term eq4 :=
sorry

end NUMINAMATH_CALUDE_only_eq1_has_zero_constant_term_l212_21255


namespace NUMINAMATH_CALUDE_three_digit_sum_reduction_l212_21260

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  let sum := d1 + d2 + d3
  let n_plus_3 := n + 3
  let d1_new := n_plus_3 / 100
  let d2_new := (n_plus_3 / 10) % 10
  let d3_new := n_plus_3 % 10
  let sum_new := d1_new + d2_new + d3_new
  sum_new = sum / 3

theorem three_digit_sum_reduction :
  ∀ n : ℕ, is_valid_number n ↔ n = 117 ∨ n = 207 ∨ n = 108 :=
sorry

end NUMINAMATH_CALUDE_three_digit_sum_reduction_l212_21260


namespace NUMINAMATH_CALUDE_min_value_expression_l212_21247

theorem min_value_expression (a b c : ℝ) (h1 : c > b) (h2 : b > a) (h3 : c ≠ 0) :
  ((a + b)^2 + (b + c)^2 + (c - a)^2) / c^2 ≥ 2 ∧
  ∃ a' b' c', c' > b' ∧ b' > a' ∧ c' ≠ 0 ∧
    ((a' + b')^2 + (b' + c')^2 + (c' - a')^2) / c'^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l212_21247


namespace NUMINAMATH_CALUDE_largest_pot_cost_is_correct_l212_21239

/-- Represents the cost of flower pots and their properties -/
structure FlowerPots where
  num_pots : ℕ
  total_cost_after_discount : ℚ
  discount_per_pot : ℚ
  price_difference : ℚ

/-- Calculates the cost of the largest pot before discount -/
def largest_pot_cost (fp : FlowerPots) : ℚ :=
  let total_discount := fp.num_pots * fp.discount_per_pot
  let total_cost_before_discount := fp.total_cost_after_discount + total_discount
  let smallest_pot_cost := (total_cost_before_discount - (fp.num_pots - 1) * fp.num_pots / 2 * fp.price_difference) / fp.num_pots
  smallest_pot_cost + (fp.num_pots - 1) * fp.price_difference

/-- Theorem stating that the cost of the largest pot before discount is $1.85 -/
theorem largest_pot_cost_is_correct (fp : FlowerPots) 
  (h1 : fp.num_pots = 6)
  (h2 : fp.total_cost_after_discount = 33/4)  -- $8.25 as a fraction
  (h3 : fp.discount_per_pot = 1/10)           -- $0.10 as a fraction
  (h4 : fp.price_difference = 3/20)           -- $0.15 as a fraction
  : largest_pot_cost fp = 37/20 := by         -- $1.85 as a fraction
  sorry

#eval largest_pot_cost {num_pots := 6, total_cost_after_discount := 33/4, discount_per_pot := 1/10, price_difference := 3/20}

end NUMINAMATH_CALUDE_largest_pot_cost_is_correct_l212_21239


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_two_l212_21221

-- Define the curve
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x

-- Theorem statement
theorem tangent_line_at_point_one_two :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 3*x - 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_two_l212_21221


namespace NUMINAMATH_CALUDE_remainder_theorem_l212_21253

theorem remainder_theorem (n m : ℤ) 
  (hn : n % 37 = 15) 
  (hm : m % 47 = 21) : 
  (3 * n + 2 * m) % 59 = 28 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l212_21253


namespace NUMINAMATH_CALUDE_defeated_candidate_percentage_approx_l212_21276

/-- Represents an election result -/
structure ElectionResult where
  total_votes : ℕ
  invalid_votes : ℕ
  margin_of_defeat : ℕ

/-- Calculates the percentage of votes for the defeated candidate -/
def defeated_candidate_percentage (result : ElectionResult) : ℚ :=
  let valid_votes := result.total_votes - result.invalid_votes
  let defeated_votes := (valid_votes - result.margin_of_defeat) / 2
  (defeated_votes : ℚ) / (valid_votes : ℚ) * 100

/-- Theorem stating that the percentage of votes for the defeated candidate is approximately 45.03% -/
theorem defeated_candidate_percentage_approx (result : ElectionResult)
  (h1 : result.total_votes = 90830)
  (h2 : result.invalid_votes = 83)
  (h3 : result.margin_of_defeat = 9000) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ 
  |defeated_candidate_percentage result - 45.03| < ε :=
sorry

end NUMINAMATH_CALUDE_defeated_candidate_percentage_approx_l212_21276


namespace NUMINAMATH_CALUDE_minimum_trips_moscow_l212_21274

theorem minimum_trips_moscow (x y : ℕ) : 
  (31 * x + 32 * y = 5000) → 
  (∀ a b : ℕ, 31 * a + 32 * b = 5000 → x + y ≤ a + b) →
  x + y = 157 := by
sorry

end NUMINAMATH_CALUDE_minimum_trips_moscow_l212_21274


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l212_21219

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_geometric_sequence_problem
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (h_non_zero : ∀ n, a n ≠ 0)
  (h_condition : a 3 - 2 * (a 6)^2 + 3 * a 7 = 0)
  (h_equal : b 6 = a 6) :
  b 1 * b 7 * b 10 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l212_21219


namespace NUMINAMATH_CALUDE_count_3033_arrangements_l212_21295

/-- The set of digits in the number 3033 -/
def digits : Finset Nat := {0, 3}

/-- A function that counts the number of four-digit numbers that can be formed from the given digits -/
def countFourDigitNumbers (d : Finset Nat) : Nat :=
  (d.filter (· ≠ 0)).card * d.card * d.card * d.card

/-- Theorem stating that the number of different four-digit numbers formed from 3033 is 1 -/
theorem count_3033_arrangements : countFourDigitNumbers digits = 1 := by
  sorry

end NUMINAMATH_CALUDE_count_3033_arrangements_l212_21295


namespace NUMINAMATH_CALUDE_fraction_1800_1809_equals_4_13_l212_21229

/-- The number of states that joined the union during 1800-1809. -/
def states_1800_1809 : ℕ := 8

/-- The total number of states in Jennifer's collection. -/
def total_states : ℕ := 26

/-- The fraction of states that joined during 1800-1809 out of the first 26 states. -/
def fraction_1800_1809 : ℚ := states_1800_1809 / total_states

theorem fraction_1800_1809_equals_4_13 : fraction_1800_1809 = 4 / 13 := by sorry

end NUMINAMATH_CALUDE_fraction_1800_1809_equals_4_13_l212_21229


namespace NUMINAMATH_CALUDE_billy_reading_speed_l212_21280

/-- Represents Billy's reading speed in pages per hour -/
def reading_speed (
  free_time_per_day : ℕ)  -- Free time per day in hours
  (weekend_days : ℕ)      -- Number of weekend days
  (gaming_percentage : ℚ) -- Percentage of time spent gaming
  (pages_per_book : ℕ)    -- Number of pages in each book
  (books_read : ℕ)        -- Number of books read
  : ℚ :=
  let total_free_time := free_time_per_day * weekend_days
  let reading_time := total_free_time * (1 - gaming_percentage)
  let total_pages := pages_per_book * books_read
  total_pages / reading_time

theorem billy_reading_speed :
  reading_speed 8 2 (3/4) 80 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_billy_reading_speed_l212_21280


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l212_21244

/-- The y-intercept of the line x + 2y + 6 = 0 is -3 -/
theorem y_intercept_of_line (x y : ℝ) : x + 2*y + 6 = 0 → y = -3 := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l212_21244


namespace NUMINAMATH_CALUDE_lcm_problem_l212_21271

theorem lcm_problem (a b c : ℕ+) (h1 : a = 15) (h2 : b = 25) (h3 : Nat.lcm (Nat.lcm a b) c = 525) : c = 7 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l212_21271


namespace NUMINAMATH_CALUDE_birthday_crayons_l212_21231

/-- The number of crayons Paul got for his birthday -/
def initial_crayons : ℕ := 1453

/-- The number of crayons Paul gave away -/
def crayons_given : ℕ := 563

/-- The number of crayons Paul lost -/
def crayons_lost : ℕ := 558

/-- The number of crayons Paul had left -/
def crayons_left : ℕ := 332

/-- Theorem stating that the initial number of crayons equals the sum of crayons given away, lost, and left -/
theorem birthday_crayons : initial_crayons = crayons_given + crayons_lost + crayons_left := by
  sorry

end NUMINAMATH_CALUDE_birthday_crayons_l212_21231


namespace NUMINAMATH_CALUDE_balloon_arrangement_count_l212_21268

/-- The number of letters in the word BALLOON -/
def n : ℕ := 7

/-- The number of times 'L' appears in BALLOON -/
def k₁ : ℕ := 2

/-- The number of times 'O' appears in BALLOON -/
def k₂ : ℕ := 2

/-- The number of unique arrangements of letters in BALLOON -/
def balloon_arrangements : ℕ := n.factorial / (k₁.factorial * k₂.factorial)

theorem balloon_arrangement_count : balloon_arrangements = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangement_count_l212_21268
