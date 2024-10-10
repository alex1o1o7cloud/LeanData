import Mathlib

namespace symmetric_trapezoid_feasibility_l2437_243756

/-- Represents a symmetric trapezoid with one parallel side equal to the legs -/
structure SymmetricTrapezoid where
  /-- Length of the legs -/
  a : ℝ
  /-- Distance from the intersection point of the diagonals to one endpoint of the other parallel side -/
  b : ℝ
  /-- Assumption that a and b are positive -/
  a_pos : a > 0
  b_pos : b > 0

/-- Theorem stating the feasibility condition for constructing the symmetric trapezoid -/
theorem symmetric_trapezoid_feasibility (t : SymmetricTrapezoid) :
  (∃ (trapezoid : SymmetricTrapezoid), trapezoid.a = t.a ∧ trapezoid.b = t.b) ↔ 3 * t.b > 2 * t.a := by
  sorry

end symmetric_trapezoid_feasibility_l2437_243756


namespace f_composed_with_g_l2437_243751

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 1
def g (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem f_composed_with_g : f (1 + g 4) = 11 := by
  sorry

end f_composed_with_g_l2437_243751


namespace probability_calculation_l2437_243709

/-- The probability of selecting one qualified and one unqualified product -/
def probability_one_qualified_one_unqualified : ℚ :=
  3 / 5

/-- The total number of products -/
def total_products : ℕ := 5

/-- The number of qualified products -/
def qualified_products : ℕ := 3

/-- The number of unqualified products -/
def unqualified_products : ℕ := 2

/-- The number of products selected for inspection -/
def selected_products : ℕ := 2

theorem probability_calculation :
  probability_one_qualified_one_unqualified = 
    (qualified_products.choose 1 * unqualified_products.choose 1 : ℚ) / 
    (total_products.choose selected_products) :=
by sorry

end probability_calculation_l2437_243709


namespace triangle_tangent_identity_l2437_243732

theorem triangle_tangent_identity (α β γ : Real) (h : α + β + γ = PI) :
  Real.tan (α/2) * Real.tan (β/2) + Real.tan (β/2) * Real.tan (γ/2) + Real.tan (γ/2) * Real.tan (α/2) = 1 := by
  sorry

end triangle_tangent_identity_l2437_243732


namespace proportion_solution_l2437_243748

theorem proportion_solution (x : ℝ) : (x / 5 = 1.05 / 7) → x = 0.75 := by
  sorry

end proportion_solution_l2437_243748


namespace bathroom_cleaning_time_is_15_l2437_243729

/-- Represents the time spent on various tasks in minutes -/
structure TaskTimes where
  total : ℕ
  laundry : ℕ
  room : ℕ
  homework : ℕ

/-- Calculates the time spent cleaning the bathroom given the times for other tasks -/
def bathroomCleaningTime (t : TaskTimes) : ℕ :=
  t.total - (t.laundry + t.room + t.homework)

theorem bathroom_cleaning_time_is_15 (t : TaskTimes) 
  (h1 : t.total = 120)
  (h2 : t.laundry = 30)
  (h3 : t.room = 35)
  (h4 : t.homework = 40) :
  bathroomCleaningTime t = 15 := by
  sorry

#eval bathroomCleaningTime { total := 120, laundry := 30, room := 35, homework := 40 }

end bathroom_cleaning_time_is_15_l2437_243729


namespace grandmother_inheritance_l2437_243718

/-- Proves that if 5 people equally split an amount of money and each receives $105,500, then the total amount is $527,500. -/
theorem grandmother_inheritance (num_people : ℕ) (amount_per_person : ℕ) (total_amount : ℕ) :
  num_people = 5 →
  amount_per_person = 105500 →
  total_amount = num_people * amount_per_person →
  total_amount = 527500 :=
by
  sorry

end grandmother_inheritance_l2437_243718


namespace infinitely_many_pairs_divisibility_l2437_243711

theorem infinitely_many_pairs_divisibility :
  ∀ n : ℕ, ∃ a b : ℤ, a > n ∧ (a * (a + 1)) ∣ (b^2 + 1) := by
  sorry

end infinitely_many_pairs_divisibility_l2437_243711


namespace intersection_equals_open_interval_l2437_243785

-- Define the sets M and N
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define the open interval (0, 1)
def open_interval_zero_one : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem statement
theorem intersection_equals_open_interval : M ∩ N = open_interval_zero_one := by
  sorry

end intersection_equals_open_interval_l2437_243785


namespace product_divisible_by_all_product_prime_factorization_divisibility_condition_l2437_243733

theorem product_divisible_by_all : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 10 → (45 * 56) % n = 0 := by sorry

theorem product_prime_factorization : 
  ∃ (k : ℕ), 45 * 56 = 2^3 * 3^2 * 5 * 7 * k ∧ k ≥ 1 := by sorry

theorem divisibility_condition (a b c d : ℕ) : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 10 → (2^a * 3^b * 5^c * 7^d) % n = 0 → a ≥ 3 ∧ b ≥ 2 ∧ c ≥ 1 ∧ d ≥ 1 := by sorry

end product_divisible_by_all_product_prime_factorization_divisibility_condition_l2437_243733


namespace existence_of_sum_of_cubes_l2437_243772

theorem existence_of_sum_of_cubes :
  ∃ (a b c d : ℕ), a^3 + b^3 + c^3 + d^3 = 100^100 := by
  sorry

end existence_of_sum_of_cubes_l2437_243772


namespace red_blue_difference_after_border_l2437_243764

/-- Represents a hexagonal figure with blue and red tiles -/
structure HexFigure where
  blue_tiles : ℕ
  red_tiles : ℕ

/-- Adds a border to a hexagonal figure, alternating between blue and red tiles -/
def add_border (fig : HexFigure) : HexFigure :=
  { blue_tiles := fig.blue_tiles + 12,
    red_tiles := fig.red_tiles + 12 }

/-- The initial hexagonal figure -/
def initial_figure : HexFigure :=
  { blue_tiles := 10,
    red_tiles := 20 }

theorem red_blue_difference_after_border :
  (add_border initial_figure).red_tiles - (add_border initial_figure).blue_tiles = 10 := by
  sorry

end red_blue_difference_after_border_l2437_243764


namespace shells_remaining_calculation_l2437_243713

/-- The number of shells Lino picked up in the morning -/
def shells_picked_up : ℝ := 324.0

/-- The number of shells Lino put back in the afternoon -/
def shells_put_back : ℝ := 292.00

/-- The number of shells Lino has in all -/
def shells_remaining : ℝ := shells_picked_up - shells_put_back

/-- Theorem stating that the number of shells Lino has in all
    is equal to the difference between shells picked up and shells put back -/
theorem shells_remaining_calculation :
  shells_remaining = 32.0 := by sorry

end shells_remaining_calculation_l2437_243713


namespace brendas_age_l2437_243710

/-- Given that Addison's age is four times Brenda's age, Janet is seven years older than Brenda,
    and Addison and Janet are twins, prove that Brenda is 7/3 years old. -/
theorem brendas_age (addison janet brenda : ℚ)
  (h1 : addison = 4 * brenda)
  (h2 : janet = brenda + 7)
  (h3 : addison = janet) :
  brenda = 7 / 3 := by
  sorry

end brendas_age_l2437_243710


namespace order_relationship_l2437_243783

theorem order_relationship (a b c : ℝ) : 
  a = Real.exp (1/2) - 1 → 
  b = Real.log (3/2) → 
  c = 5/12 → 
  a > c ∧ c > b := by
sorry

end order_relationship_l2437_243783


namespace overlapping_area_area_covered_by_both_strips_l2437_243776

/-- The area covered by both strips -/
def S : ℝ := 13.5

/-- The length of the original rectangular strip -/
def total_length : ℝ := 16

/-- The length of the left strip -/
def left_length : ℝ := 9

/-- The length of the right strip -/
def right_length : ℝ := 7

/-- The area covered only by the left strip -/
def left_area : ℝ := 27

/-- The area covered only by the right strip -/
def right_area : ℝ := 18

theorem overlapping_area :
  (left_area + S) / (right_area + S) = left_length / right_length :=
by sorry

theorem area_covered_by_both_strips : S = 13.5 :=
by sorry

end overlapping_area_area_covered_by_both_strips_l2437_243776


namespace cylinder_diameter_l2437_243799

/-- The diameter of a cylinder given its height and volume -/
theorem cylinder_diameter (h : ℝ) (v : ℝ) (h_pos : h > 0) (v_pos : v > 0) :
  let d := 2 * Real.sqrt (9 / Real.pi)
  h = 5 ∧ v = 45 → d * d * Real.pi * h / 4 = v := by
  sorry

end cylinder_diameter_l2437_243799


namespace melody_civics_pages_l2437_243793

/-- The number of pages Melody needs to read for her English class -/
def english_pages : ℕ := 20

/-- The number of pages Melody needs to read for her Science class -/
def science_pages : ℕ := 16

/-- The number of pages Melody needs to read for her Chinese class -/
def chinese_pages : ℕ := 12

/-- The fraction of pages Melody will read tomorrow for each class -/
def read_fraction : ℚ := 1/4

/-- The total number of pages Melody will read tomorrow -/
def total_pages_tomorrow : ℕ := 14

/-- The number of pages Melody needs to read for her Civics class -/
def civics_pages : ℕ := 8

theorem melody_civics_pages :
  (english_pages : ℚ) * read_fraction +
  (science_pages : ℚ) * read_fraction +
  (chinese_pages : ℚ) * read_fraction +
  (civics_pages : ℚ) * read_fraction = total_pages_tomorrow :=
sorry

end melody_civics_pages_l2437_243793


namespace inequality_comparison_l2437_243730

theorem inequality_comparison (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / a > 1 / b) ∧ 
  (abs a > abs b) ∧ 
  (a^2 > b^2) ∧
  ¬(∀ a b, a < b ∧ b < 0 → 1 / (a - b) > 1 / a) := by
sorry

end inequality_comparison_l2437_243730


namespace area_of_specific_circumscribed_rectangle_l2437_243738

/-- A rectangle circumscribed around a right triangle -/
structure CircumscribedRectangle where
  /-- Length of one leg of the right triangle -/
  leg1 : ℝ
  /-- Length of the other leg of the right triangle -/
  leg2 : ℝ
  /-- The legs are positive -/
  leg1_pos : 0 < leg1
  leg2_pos : 0 < leg2

/-- The area of a rectangle circumscribed around a right triangle -/
def area (r : CircumscribedRectangle) : ℝ := r.leg1 * r.leg2

/-- Theorem: The area of a rectangle circumscribed around a right triangle
    with legs of length 5 and 6 is equal to 30 square units -/
theorem area_of_specific_circumscribed_rectangle :
  ∃ (r : CircumscribedRectangle), r.leg1 = 5 ∧ r.leg2 = 6 ∧ area r = 30 := by
  sorry


end area_of_specific_circumscribed_rectangle_l2437_243738


namespace negation_equivalence_l2437_243770

theorem negation_equivalence :
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ (∀ n : ℕ, 2^n ≤ 1000) := by
  sorry

end negation_equivalence_l2437_243770


namespace intersection_of_P_and_Q_l2437_243792

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x > 1}
def Q : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {x : ℝ | 1 < x ∧ x ≤ 2} := by
  sorry

end intersection_of_P_and_Q_l2437_243792


namespace intersection_line_slope_l2437_243749

/-- Given two circles in the plane, this theorem states that the slope of the line
passing through their intersection points is -1/3. -/
theorem intersection_line_slope (x y : ℝ) :
  (x^2 + y^2 - 6*x + 4*y - 8 = 0) ∧ 
  (x^2 + y^2 - 8*x - 2*y + 10 = 0) →
  (∃ m b : ℝ, y = m*x + b ∧ m = -1/3) :=
by sorry

end intersection_line_slope_l2437_243749


namespace zero_is_self_opposite_l2437_243763

/-- Two real numbers are opposite if they have the same magnitude but opposite signs, or both are zero. -/
def are_opposite (a b : ℝ) : Prop := (a = -b) ∨ (a = 0 ∧ b = 0)

/-- Zero is its own opposite number. -/
theorem zero_is_self_opposite : are_opposite 0 0 := by
  sorry

end zero_is_self_opposite_l2437_243763


namespace quadratic_sum_l2437_243702

/-- A quadratic function with vertex (h, k) and passing through point (x₀, y₀) -/
def quadratic_function (a b c h k x₀ y₀ : ℝ) : Prop :=
  ∀ x, a * (x - h)^2 + k = a * x^2 + b * x + c ∧
  a * (x₀ - h)^2 + k = y₀

theorem quadratic_sum (a b c : ℝ) :
  quadratic_function a b c 2 5 1 8 →
  a - b + c = 32 := by
  sorry

end quadratic_sum_l2437_243702


namespace inequality_solution_set_l2437_243714

theorem inequality_solution_set :
  let S := {x : ℝ | (1 / 3 : ℝ) < x ∧ x < (5 / 3 : ℝ) ∧ x ≠ 1}
  ∀ x : ℝ, x ∈ S ↔ (1 / |x - 1| : ℝ) > (3 / 2 : ℝ) := by
sorry

end inequality_solution_set_l2437_243714


namespace fourth_side_length_is_correct_l2437_243782

/-- A quadrilateral inscribed in a circle with radius 150√3, where three sides have length 300 --/
structure InscribedQuadrilateral where
  radius : ℝ
  three_side_length : ℝ
  h_radius : radius = 150 * Real.sqrt 3
  h_three_sides : three_side_length = 300

/-- The length of the fourth side of the inscribed quadrilateral --/
def fourth_side_length (q : InscribedQuadrilateral) : ℝ := 562.5

/-- Theorem stating that the fourth side length is correct --/
theorem fourth_side_length_is_correct (q : InscribedQuadrilateral) :
  fourth_side_length q = 562.5 := by
  sorry

end fourth_side_length_is_correct_l2437_243782


namespace set_equality_l2437_243719

def M : Set ℝ := {x | x^2 - 2012*x - 2013 > 0}
def N (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

theorem set_equality (a b : ℝ) : 
  M ∪ N a b = Set.univ ∧ 
  M ∩ N a b = Set.Ioo 2013 2014 →
  a = -2013 ∧ b = -2014 := by
sorry

end set_equality_l2437_243719


namespace min_distance_between_curves_l2437_243788

/-- The minimum distance between two points on different curves with the same y-coordinate -/
theorem min_distance_between_curves : ∃ (min_dist : ℝ),
  (∀ (a x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 →
    a = 2 * (x₁ + 1) →
    a = x₂ + Real.log x₂ →
    |x₂ - x₁| ≥ min_dist) ∧
  (∃ (a x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧
    a = 2 * (x₁ + 1) ∧
    a = x₂ + Real.log x₂ ∧
    |x₂ - x₁| = min_dist) ∧
  min_dist = 3 / 2 := by
  sorry

end min_distance_between_curves_l2437_243788


namespace complex_number_problem_l2437_243704

theorem complex_number_problem (i : ℂ) (h : i^2 = -1) :
  let z_i := ((i + 1) / (i - 1))^2016
  let z := 1 / i
  z = -i :=
by sorry

end complex_number_problem_l2437_243704


namespace middle_number_proof_l2437_243705

theorem middle_number_proof (x y z : ℕ) (hxy : x < y) (hyz : y < z)
  (sum_xy : x + y = 22) (sum_xz : x + z = 29) (sum_yz : y + z = 37) : y = 15 := by
  sorry

end middle_number_proof_l2437_243705


namespace prob_more_ones_than_sixes_proof_l2437_243740

/-- The number of possible outcomes when rolling five fair six-sided dice -/
def total_outcomes : ℕ := 6^5

/-- The number of ways to roll an equal number of 1's and 6's when rolling five fair six-sided dice -/
def equal_ones_and_sixes : ℕ := 2334

/-- The probability of rolling more 1's than 6's when rolling five fair six-sided dice -/
def prob_more_ones_than_sixes : ℚ := 2711 / 7776

theorem prob_more_ones_than_sixes_proof :
  prob_more_ones_than_sixes = 1/2 * (1 - equal_ones_and_sixes / total_outcomes) :=
by sorry

end prob_more_ones_than_sixes_proof_l2437_243740


namespace parabola_translation_l2437_243750

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk 1 0 0  -- y = x^2
  let translated := translate original 3 2
  y = translated.a * (x - 3)^2 + translated.b * (x - 3) + translated.c ↔
  y = (x - 3)^2 + 2 := by
  sorry

end parabola_translation_l2437_243750


namespace largest_n_for_factorization_l2437_243777

theorem largest_n_for_factorization : 
  ∀ n : ℤ, 
  (∃ A B : ℤ, ∀ x : ℤ, 3 * x^2 + n * x + 72 = (3 * x + A) * (x + B)) →
  n ≤ 217 :=
by sorry

end largest_n_for_factorization_l2437_243777


namespace factors_of_72_l2437_243717

theorem factors_of_72 : Nat.card (Nat.divisors 72) = 12 := by sorry

end factors_of_72_l2437_243717


namespace fresh_grapes_weight_l2437_243735

/-- The weight of fresh grapes required to produce a given weight of dried grapes -/
theorem fresh_grapes_weight (fresh_water_content : ℝ) (dried_water_content : ℝ) (dried_weight : ℝ) :
  fresh_water_content = 0.8 →
  dried_water_content = 0.2 →
  dried_weight = 10 →
  (1 - fresh_water_content) * (dried_weight / (1 - dried_water_content)) = 40 :=
by sorry

end fresh_grapes_weight_l2437_243735


namespace age_difference_l2437_243746

/-- Proves that z is 1.2 decades younger than x given the condition on ages -/
theorem age_difference (x y z : ℝ) (h : x + y = y + z + 12) : (x - z) / 10 = 1.2 := by
  sorry

end age_difference_l2437_243746


namespace irregular_shape_area_l2437_243779

/-- The area of an irregular shape formed by removing a smaller rectangle and a right triangle from a larger rectangle --/
theorem irregular_shape_area (large_length large_width small_length small_width triangle_base triangle_height : ℝ)
  (h1 : large_length = 10)
  (h2 : large_width = 6)
  (h3 : small_length = 4)
  (h4 : small_width = 3)
  (h5 : triangle_base = small_length)
  (h6 : triangle_height = 3) :
  large_length * large_width - (small_length * small_width + 1/2 * triangle_base * triangle_height) = 42 := by
  sorry

end irregular_shape_area_l2437_243779


namespace quadratic_roots_relation_l2437_243726

theorem quadratic_roots_relation (m p q : ℝ) (hm : m ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0) :
  (∃ r₁ r₂ : ℝ, (r₁ + r₂ = -q ∧ r₁ * r₂ = m) ∧
               (3*r₁ + 3*r₂ = -m ∧ 9*r₁*r₂ = p)) →
  p / q = 27 := by
sorry

end quadratic_roots_relation_l2437_243726


namespace all_cards_same_number_l2437_243715

theorem all_cards_same_number (m : ℕ) (cards : Fin m → ℕ) : 
  (∀ i : Fin m, 1 ≤ cards i ∧ cards i ≤ m) →
  (∀ s : Finset (Fin m), (s.sum cards) % (m + 1) ≠ 0) →
  ∀ i j : Fin m, cards i = cards j :=
sorry

end all_cards_same_number_l2437_243715


namespace garden_length_ratio_l2437_243780

/-- Given a rectangular property and a rectangular garden, this theorem proves
    the ratio of the garden's length to the property's length. -/
theorem garden_length_ratio
  (property_length : ℝ)
  (property_width : ℝ)
  (garden_area : ℝ)
  (h_property_length : property_length = 2250)
  (h_property_width : property_width = 1000)
  (h_garden_area : garden_area = 28125)
  (garden_width : ℝ)
  (h_garden_width_pos : garden_width > 0) :
  garden_area / garden_width / property_length = 12.5 / garden_width :=
by sorry

end garden_length_ratio_l2437_243780


namespace min_value_reciprocal_product_l2437_243724

theorem min_value_reciprocal_product (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_mean : 2 = (2 * a + b) / 2) : 
  ∀ x y, x > 0 → y > 0 → (2 = (2 * x + y) / 2) → 1 / (a * b) ≤ 1 / (x * y) :=
sorry

end min_value_reciprocal_product_l2437_243724


namespace dans_initial_money_l2437_243728

/-- The amount of money Dan has left after buying the candy bar -/
def money_left : ℝ := 3

/-- The cost of the candy bar -/
def candy_cost : ℝ := 2

/-- Dan's initial amount of money -/
def initial_money : ℝ := money_left + candy_cost

theorem dans_initial_money : initial_money = 5 := by sorry

end dans_initial_money_l2437_243728


namespace find_true_product_l2437_243767

theorem find_true_product (a b : ℕ) : 
  b = 2 * a →
  136 * (10 * b + a) = 136 * (10 * a + b) + 1224 →
  136 * (10 * a + b) = 1632 := by
sorry

end find_true_product_l2437_243767


namespace inequality_proof_l2437_243742

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * c) * (Real.sqrt a + Real.sqrt b + Real.sqrt c) + (a + b + c)^2 ≥ 
  4 * Real.sqrt (3 * a * b * c * (a + b + c)) :=
by sorry

end inequality_proof_l2437_243742


namespace remainder_theorem_l2437_243762

theorem remainder_theorem (P D D' Q Q' R R' : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = D' * Q' + R')
  (h3 : R < D)
  (h4 : R' < D') :
  P % (2 * D * D') = D * R' + R :=
sorry

end remainder_theorem_l2437_243762


namespace point_on_inverse_proportion_graph_l2437_243784

/-- Proves that the point (-2, 2) lies on the graph of the inverse proportion function y = -4/x -/
theorem point_on_inverse_proportion_graph :
  let f : ℝ → ℝ := λ x => -4 / x
  f (-2) = 2 := by sorry

end point_on_inverse_proportion_graph_l2437_243784


namespace smallest_number_with_remainders_l2437_243739

theorem smallest_number_with_remainders (n : ℕ) : 
  (n > 1) →
  (n % 3 = 1) → 
  (n % 4 = 1) → 
  (n % 7 = 1) → 
  (∀ m : ℕ, m > 1 → m % 3 = 1 → m % 4 = 1 → m % 7 = 1 → n ≤ m) →
  (n = 85 ∧ 84 < n ∧ n ≤ 107) := by
sorry

end smallest_number_with_remainders_l2437_243739


namespace equation_roots_properties_l2437_243727

open Real

theorem equation_roots_properties (m : ℝ) (θ : ℝ) :
  θ ∈ Set.Ioo 0 π →
  (∀ x, 2 * x^2 - (Real.sqrt 3 + 1) * x + m = 0 ↔ x = sin θ ∨ x = cos θ) →
  (m = Real.sqrt 3 / 2) ∧
  ((tan θ * sin θ) / (tan θ - 1) + cos θ / (1 - tan θ) = (Real.sqrt 3 + 1) / 2) ∧
  ((sin θ = Real.sqrt 3 / 2 ∧ cos θ = 1 / 2) ∨ (sin θ = 1 / 2 ∧ cos θ = Real.sqrt 3 / 2)) ∧
  (θ = π / 3 ∨ θ = π / 6) := by
  sorry

#check equation_roots_properties

end equation_roots_properties_l2437_243727


namespace polygon_angles_l2437_243716

theorem polygon_angles (n : ℕ) : 
  (n - 2) * 180 + (180 - 180 / n) = 2007 → n = 13 := by
  sorry

end polygon_angles_l2437_243716


namespace evaluate_expression_l2437_243707

theorem evaluate_expression (a b : ℕ) (h1 : a = 2009) (h2 : b = 2010) :
  2 * (b^3 - a*b^2 - a^2*b + a^3) = 24240542 := by
  sorry

end evaluate_expression_l2437_243707


namespace train_distance_problem_l2437_243798

theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 20) (h2 : v2 = 25) (h3 : d = 70) :
  let t := d / (v2 - v1)
  let x := v1 * t
  (x + (x + d)) = 630 := by sorry

end train_distance_problem_l2437_243798


namespace lcm_of_180_and_504_l2437_243743

theorem lcm_of_180_and_504 : Nat.lcm 180 504 = 2520 := by
  sorry

end lcm_of_180_and_504_l2437_243743


namespace three_digit_base_problem_l2437_243771

theorem three_digit_base_problem :
  ∃! (x y z b : ℕ),
    x * b^2 + y * b + z = 1993 ∧
    x + y + z = 22 ∧
    x < b ∧ y < b ∧ z < b ∧
    b > 10 ∧
    x = 2 ∧ y = 15 ∧ z = 5 ∧ b = 28 := by
  sorry

end three_digit_base_problem_l2437_243771


namespace probability_of_white_ball_l2437_243755

def num_black_balls : ℕ := 6
def num_white_balls : ℕ := 5

def total_balls : ℕ := num_black_balls + num_white_balls

theorem probability_of_white_ball :
  (num_white_balls : ℚ) / (total_balls : ℚ) = 5 / 11 := by
  sorry

end probability_of_white_ball_l2437_243755


namespace exists_valid_sequence_l2437_243722

def is_valid_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ k, k ≥ 1 → a (2*k + 1) = (a (2*k) + a (2*k + 2)) / 2) ∧
  (∀ k, k ≥ 1 → a (2*k) = Real.sqrt (a (2*k - 1) * a (2*k + 1)))

theorem exists_valid_sequence : ∃ a : ℕ → ℝ, is_valid_sequence a :=
sorry

end exists_valid_sequence_l2437_243722


namespace perfect_square_quadratic_l2437_243774

theorem perfect_square_quadratic (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, x^2 + (m + 2) * x + 36 = y^2) →
  m = 10 ∨ m = -14 :=
by sorry

end perfect_square_quadratic_l2437_243774


namespace cos_sum_24_144_264_l2437_243775

theorem cos_sum_24_144_264 :
  Real.cos (24 * π / 180) + Real.cos (144 * π / 180) + Real.cos (264 * π / 180) =
    (3 - Real.sqrt 5) / 4 - Real.sin (3 * π / 180) * Real.sqrt (10 + 2 * Real.sqrt 5) := by
  sorry

end cos_sum_24_144_264_l2437_243775


namespace reciprocal_of_negative_one_l2437_243766

theorem reciprocal_of_negative_one :
  (∃ x : ℝ, x * (-1) = 1) ∧ (∀ y : ℝ, y * (-1) = 1 → y = -1) :=
by sorry

end reciprocal_of_negative_one_l2437_243766


namespace opposite_of_negative_one_third_l2437_243794

theorem opposite_of_negative_one_third :
  let x : ℚ := -1/3
  let opposite (y : ℚ) : ℚ := -y
  opposite x = 1/3 := by
  sorry

end opposite_of_negative_one_third_l2437_243794


namespace sticker_problem_l2437_243773

theorem sticker_problem (initial_stickers : ℚ) : 
  let lost_stickers := (1 : ℚ) / 3 * initial_stickers
  let found_stickers := (3 : ℚ) / 4 * lost_stickers
  let remaining_stickers := initial_stickers - lost_stickers + found_stickers
  initial_stickers - remaining_stickers = (1 : ℚ) / 12 * initial_stickers :=
by sorry

end sticker_problem_l2437_243773


namespace expression_simplification_l2437_243781

theorem expression_simplification (n : ℝ) (h : n = Real.sqrt 2 + 1) :
  ((n + 3) / (n^2 - 1) - 1 / (n + 1)) / (2 / (n + 1)) = Real.sqrt 2 := by
  sorry

end expression_simplification_l2437_243781


namespace inequality_proof_l2437_243745

theorem inequality_proof (x y z : ℝ) : x^2 + y^2 + z^2 ≥ Real.sqrt 2 * (x*y + y*z) := by
  sorry

end inequality_proof_l2437_243745


namespace curve_intersection_property_m_range_l2437_243769

/-- The curve C defined by y² = 4x for x > 0 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4*p.1 ∧ p.1 > 0}

/-- The line passing through (m, 0) with slope 1/t -/
def line (m t : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = t*p.2 + m}

/-- The dot product of vectors FA and FB where F is (1, 0) -/
def dot_product (A B : ℝ × ℝ) : ℝ := (A.1 - 1)*(B.1 - 1) + A.2*B.2

theorem curve_intersection_property :
  ∃ (m : ℝ), m > 0 ∧
  ∀ (t : ℝ), ∀ (A B : ℝ × ℝ),
    A ∈ C → B ∈ C → A ∈ line m t → B ∈ line m t → A ≠ B →
    dot_product A B < 0 :=
sorry

theorem m_range (m : ℝ) :
  (∀ (t : ℝ), ∀ (A B : ℝ × ℝ),
    A ∈ C → B ∈ C → A ∈ line m t → B ∈ line m t → A ≠ B →
    dot_product A B < 0) ↔
  3 - 2*Real.sqrt 2 < m ∧ m < 3 + 2*Real.sqrt 2 :=
sorry

end curve_intersection_property_m_range_l2437_243769


namespace count_valid_arrangements_l2437_243790

/-- The number of valid 18-letter arrangements of 6 D's, 6 E's, and 6 F's -/
def valid_arrangements : ℕ :=
  Finset.sum (Finset.range 7) (fun m => (Nat.choose 6 m) ^ 3)

/-- Theorem stating the number of valid arrangements -/
theorem count_valid_arrangements :
  valid_arrangements =
    (Finset.sum (Finset.range 7) (fun m => (Nat.choose 6 m) ^ 3)) := by
  sorry

end count_valid_arrangements_l2437_243790


namespace aiyannas_cookie_count_l2437_243731

/-- The number of cookies Alyssa has -/
def alyssas_cookies : ℕ := 129

/-- The number of additional cookies Aiyanna has compared to Alyssa -/
def additional_cookies : ℕ := 11

/-- The number of cookies Aiyanna has -/
def aiyannas_cookies : ℕ := alyssas_cookies + additional_cookies

theorem aiyannas_cookie_count : aiyannas_cookies = 140 := by
  sorry

end aiyannas_cookie_count_l2437_243731


namespace min_value_theorem_l2437_243725

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2 * y = 5) :
  ∃ (min_val : ℝ), min_val = 4 * Real.sqrt 3 ∧
  ∀ (z : ℝ), z = (x + 1) * (2 * y + 1) / Real.sqrt (x * y) → z ≥ min_val :=
by sorry

end min_value_theorem_l2437_243725


namespace gcd_lcm_sum_l2437_243778

theorem gcd_lcm_sum : Nat.gcd 54 24 + Nat.lcm 48 18 = 150 := by
  sorry

end gcd_lcm_sum_l2437_243778


namespace min_value_theorem_l2437_243760

theorem min_value_theorem (x : ℝ) (h : x > 2) :
  x + 4 / (x - 2) ≥ 6 ∧ (x + 4 / (x - 2) = 6 ↔ x = 4) := by sorry

end min_value_theorem_l2437_243760


namespace rapid_advance_min_cost_l2437_243720

/-- Represents a ride model with its capacity and price -/
structure RideModel where
  capacity : ℕ
  price : ℕ

/-- The minimum amount needed to spend on tickets for a group -/
def minTicketCost (model1 model2 : RideModel) (groupSize : ℕ) : ℕ :=
  sorry

theorem rapid_advance_min_cost :
  let model1 : RideModel := { capacity := 7, price := 65 }
  let model2 : RideModel := { capacity := 5, price := 50 }
  let groupSize : ℕ := 73
  minTicketCost model1 model2 groupSize = 685 := by sorry

end rapid_advance_min_cost_l2437_243720


namespace binop_commutative_l2437_243737

-- Define a binary operation on a type
def BinOp (α : Type) := α → α → α

-- Define the properties of the binary operation
class MyBinOp (α : Type) (op : BinOp α) where
  left_cancel : ∀ a b : α, op a (op a b) = b
  right_cancel : ∀ a b : α, op (op a b) b = a

-- State the theorem
theorem binop_commutative {α : Type} (op : BinOp α) [MyBinOp α op] :
  ∀ a b : α, op a b = op b a := by
  sorry

end binop_commutative_l2437_243737


namespace opposite_numbers_l2437_243754

theorem opposite_numbers : -(-(3 : ℤ)) = -(-3) :=
by sorry

end opposite_numbers_l2437_243754


namespace board_cut_ratio_l2437_243736

/-- Given a board of length 69 inches cut into two pieces,
    where one piece is a multiple of the other and the longer piece is 46 inches,
    prove that the ratio of the longer piece to the shorter piece is 2:1 -/
theorem board_cut_ratio (total_length shorter_length longer_length : ℝ)
  (h1 : total_length = 69)
  (h2 : total_length = shorter_length + longer_length)
  (h3 : ∃ (m : ℝ), longer_length = m * shorter_length)
  (h4 : longer_length = 46) :
  longer_length / shorter_length = 2 := by
  sorry

end board_cut_ratio_l2437_243736


namespace strawberry_distribution_l2437_243789

/-- Represents the distribution of strawberries in buckets -/
structure StrawberryDistribution where
  buckets : Fin 5 → ℕ

/-- The initial distribution of strawberries -/
def initial_distribution : StrawberryDistribution :=
  { buckets := λ _ => 60 }

/-- Removes a specified number of strawberries from each bucket -/
def remove_from_each (d : StrawberryDistribution) (amount : ℕ) : StrawberryDistribution :=
  { buckets := λ i => d.buckets i - amount }

/-- Adds strawberries to specific buckets -/
def add_to_buckets (d : StrawberryDistribution) (additions : Fin 5 → ℕ) : StrawberryDistribution :=
  { buckets := λ i => d.buckets i + additions i }

/-- The final distribution of strawberries after all adjustments -/
def final_distribution : StrawberryDistribution :=
  add_to_buckets
    (remove_from_each initial_distribution 20)
    (λ i => match i with
      | 0 => 15
      | 1 => 15
      | 2 => 25
      | _ => 0)

/-- Theorem stating the final distribution of strawberries -/
theorem strawberry_distribution :
  final_distribution.buckets = λ i => match i with
    | 0 => 55
    | 1 => 55
    | 2 => 65
    | _ => 40 := by sorry

end strawberry_distribution_l2437_243789


namespace quadratic_equation_standard_form_quadratic_coefficients_l2437_243706

theorem quadratic_equation_standard_form :
  ∀ x : ℝ, (x + 5) * (3 + x) = 2 * x^2 ↔ x^2 - 8 * x - 15 = 0 :=
by sorry

theorem quadratic_coefficients (a b c : ℝ) :
  (∀ x : ℝ, (x + 5) * (3 + x) = 2 * x^2 ↔ a * x^2 + b * x + c = 0) →
  a = 1 ∧ b = -8 ∧ c = -15 :=
by sorry

end quadratic_equation_standard_form_quadratic_coefficients_l2437_243706


namespace focus_of_parabola_l2437_243721

/-- The parabola defined by the equation y^2 = 4x -/
def parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of a parabola with equation y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- Theorem: The focus of the parabola y^2 = 4x has coordinates (1, 0) -/
theorem focus_of_parabola :
  focus ∈ {p : ℝ × ℝ | p.1 > 0 ∧ ∀ q ∈ parabola, (p.1 - q.1)^2 + (p.2 - q.2)^2 = (p.1 + q.1)^2} :=
sorry

end focus_of_parabola_l2437_243721


namespace intersection_line_slope_l2437_243701

-- Define the circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y + 12 = 0

-- Define the intersection points
def intersection_points (C D : ℝ × ℝ) : Prop :=
  circle1 C.1 C.2 ∧ circle1 D.1 D.2 ∧ circle2 C.1 C.2 ∧ circle2 D.1 D.2

-- Theorem statement
theorem intersection_line_slope (C D : ℝ × ℝ) 
  (h : intersection_points C D) : 
  (D.2 - C.2) / (D.1 - C.1) = -1 := by sorry

end intersection_line_slope_l2437_243701


namespace division_problem_l2437_243796

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 171 →
  quotient = 8 →
  remainder = 3 →
  dividend = divisor * quotient + remainder →
  divisor = 21 := by
sorry

end division_problem_l2437_243796


namespace dave_initial_apps_l2437_243758

/-- The number of apps Dave had initially -/
def initial_apps : ℕ := sorry

/-- The number of apps Dave had after adding one -/
def apps_after_adding : ℕ := 18

theorem dave_initial_apps : 
  initial_apps = 17 :=
by
  sorry

end dave_initial_apps_l2437_243758


namespace probability_same_group_l2437_243791

def total_items : ℕ := 6
def group_size : ℕ := 2
def num_groups : ℕ := 3
def items_to_choose : ℕ := 2

theorem probability_same_group :
  (num_groups * (group_size.choose items_to_choose)) / (total_items.choose items_to_choose) = 1 / 5 := by
  sorry

end probability_same_group_l2437_243791


namespace no_two_digit_primes_with_digit_sum_12_l2437_243795

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem no_two_digit_primes_with_digit_sum_12 :
  ∀ n : ℕ, is_two_digit n → Nat.Prime n → digit_sum n = 12 → False :=
by
  sorry

end no_two_digit_primes_with_digit_sum_12_l2437_243795


namespace tour_group_size_l2437_243741

theorem tour_group_size (initial_groups : ℕ) (initial_avg : ℕ) (remaining_groups : ℕ) (remaining_avg : ℕ) :
  initial_groups = 10 →
  initial_avg = 9 →
  remaining_groups = 9 →
  remaining_avg = 8 →
  (initial_groups * initial_avg) - (remaining_groups * remaining_avg) = 18 :=
by
  sorry

end tour_group_size_l2437_243741


namespace evelyns_marbles_l2437_243786

/-- The number of marbles Evelyn has in total -/
def total_marbles (initial : ℕ) (from_henry : ℕ) (from_grace : ℕ) (cards : ℕ) (marbles_per_card : ℕ) : ℕ :=
  initial + from_henry + from_grace + cards * marbles_per_card

/-- Theorem stating that Evelyn's total number of marbles is 140 -/
theorem evelyns_marbles :
  total_marbles 95 9 12 6 4 = 140 := by
  sorry

#eval total_marbles 95 9 12 6 4

end evelyns_marbles_l2437_243786


namespace scheduleArrangements_eq_180_l2437_243768

/-- The number of ways to schedule 4 out of 6 people over 3 days -/
def scheduleArrangements : ℕ :=
  Nat.choose 6 1 * Nat.choose 5 1 * Nat.choose 4 2

/-- Theorem stating that the number of scheduling arrangements is 180 -/
theorem scheduleArrangements_eq_180 : scheduleArrangements = 180 := by
  sorry

end scheduleArrangements_eq_180_l2437_243768


namespace shopkeeper_gain_percentage_l2437_243747

/-- Calculates the gain percentage of a shopkeeper using false weights -/
theorem shopkeeper_gain_percentage (true_weight false_weight : ℕ) : 
  true_weight = 1000 → 
  false_weight = 960 → 
  (true_weight - false_weight) * 100 / true_weight = 4 := by
  sorry

#check shopkeeper_gain_percentage

end shopkeeper_gain_percentage_l2437_243747


namespace smallest_x_value_l2437_243752

theorem smallest_x_value (y : ℕ+) (x : ℕ) 
  (h : (857 : ℚ) / 1000 = (y : ℚ) / ((210 : ℚ) + x)) : 
  ∀ x' : ℕ, x' ≥ x → x = 0 :=
sorry

end smallest_x_value_l2437_243752


namespace undefined_expression_l2437_243797

theorem undefined_expression (y : ℝ) : 
  y^2 - 16*y + 64 = 0 → y = 8 :=
by sorry

end undefined_expression_l2437_243797


namespace triangle_side_length_l2437_243787

theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  c = 2 → b = 2 * a → Real.cos C = (1 : ℝ) / 4 → 
  (a^2 + b^2 - c^2) / (2 * a * b) = Real.cos C → a = 1 := by
sorry

end triangle_side_length_l2437_243787


namespace add_self_eq_two_mul_l2437_243744

theorem add_self_eq_two_mul (a : ℝ) : a + a = 2 * a := by sorry

end add_self_eq_two_mul_l2437_243744


namespace simplify_complex_square_root_l2437_243765

theorem simplify_complex_square_root : 
  Real.sqrt ((9^8 + 3^14) / (9^6 + 3^15)) = Real.sqrt (15/14) := by
  sorry

end simplify_complex_square_root_l2437_243765


namespace length_PS_is_sqrt_32_5_l2437_243723

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S T : ℝ × ℝ)

-- Define the conditions
def is_valid_quadrilateral (quad : Quadrilateral) : Prop :=
  let d_PT := Real.sqrt ((quad.P.1 - quad.T.1)^2 + (quad.P.2 - quad.T.2)^2)
  let d_TR := Real.sqrt ((quad.T.1 - quad.R.1)^2 + (quad.T.2 - quad.R.2)^2)
  let d_QT := Real.sqrt ((quad.Q.1 - quad.T.1)^2 + (quad.Q.2 - quad.T.2)^2)
  let d_TS := Real.sqrt ((quad.T.1 - quad.S.1)^2 + (quad.T.2 - quad.S.2)^2)
  let d_PQ := Real.sqrt ((quad.P.1 - quad.Q.1)^2 + (quad.P.2 - quad.Q.2)^2)
  d_PT = 5 ∧ d_TR = 4 ∧ d_QT = 7 ∧ d_TS = 2 ∧ d_PQ = 7

-- Theorem statement
theorem length_PS_is_sqrt_32_5 (quad : Quadrilateral) 
  (h : is_valid_quadrilateral quad) : 
  Real.sqrt ((quad.P.1 - quad.S.1)^2 + (quad.P.2 - quad.S.2)^2) = Real.sqrt 32.5 := by
  sorry

end length_PS_is_sqrt_32_5_l2437_243723


namespace triangle_area_l2437_243708

theorem triangle_area (a b c : ℝ) (A : ℝ) (S : ℝ) : 
  b^2 - b*c - 2*c^2 = 0 →
  a = Real.sqrt 6 →
  Real.cos A = 7/8 →
  S = (Real.sqrt 15)/2 := by
sorry

end triangle_area_l2437_243708


namespace isosceles_triangle_side_lengths_l2437_243703

/-- An isosceles triangle with integer side lengths and perimeter 10 --/
structure IsoscelesTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  isIsosceles : (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)
  perimeter : a + b + c = 10

/-- The possible side lengths of the isosceles triangle --/
def validSideLengths (t : IsoscelesTriangle) : Prop :=
  (t.a = 3 ∧ t.b = 3 ∧ t.c = 4) ∨ (t.a = 4 ∧ t.b = 4 ∧ t.c = 2)

/-- Theorem stating that the only possible side lengths are (3, 3, 4) or (4, 4, 2) --/
theorem isosceles_triangle_side_lengths (t : IsoscelesTriangle) : validSideLengths t := by
  sorry

end isosceles_triangle_side_lengths_l2437_243703


namespace segment_length_product_l2437_243761

theorem segment_length_product (a : ℝ) : 
  (∃ a₁ a₂ : ℝ, 
    (∀ a, ((3*a - 5)^2 + (2*a - 4)^2 = 34) ↔ (a = a₁ ∨ a = a₂)) ∧
    (a₁ * a₂ = -722/169)) :=
by sorry

end segment_length_product_l2437_243761


namespace mean_median_mode_equality_l2437_243734

/-- Represents the days of the week -/
inductive Weekday
  | Saturday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- A month with its properties -/
structure Month where
  totalDays : Nat
  startDay : Weekday
  frequencies : Weekday → Nat

/-- Calculates the mean of the frequencies -/
def calculateMean (m : Month) : ℚ :=
  (m.frequencies Weekday.Saturday +
   m.frequencies Weekday.Sunday +
   m.frequencies Weekday.Monday +
   m.frequencies Weekday.Tuesday +
   m.frequencies Weekday.Wednesday +
   m.frequencies Weekday.Thursday +
   m.frequencies Weekday.Friday) / 7

/-- Calculates the median day -/
def calculateMedian (m : Month) : Weekday :=
  Weekday.Tuesday  -- Since the 15th day (median) is a Tuesday

/-- Calculates the median of the modes -/
def calculateMedianOfModes (m : Month) : ℚ := 4

/-- The theorem to be proved -/
theorem mean_median_mode_equality (m : Month)
  (h1 : m.totalDays = 29)
  (h2 : m.startDay = Weekday.Saturday)
  (h3 : m.frequencies Weekday.Saturday = 5)
  (h4 : m.frequencies Weekday.Sunday = 4)
  (h5 : m.frequencies Weekday.Monday = 4)
  (h6 : m.frequencies Weekday.Tuesday = 4)
  (h7 : m.frequencies Weekday.Wednesday = 4)
  (h8 : m.frequencies Weekday.Thursday = 4)
  (h9 : m.frequencies Weekday.Friday = 4) :
  calculateMean m = calculateMedianOfModes m ∧
  calculateMedianOfModes m = (calculateMedian m).rec 4 4 4 4 4 4 4 := by
  sorry

end mean_median_mode_equality_l2437_243734


namespace extended_quadrilateral_area_l2437_243759

/-- Represents a quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  -- The area of the original quadrilateral
  area : ℝ
  -- The lengths of the sides and their extensions
  ef : ℝ
  fg : ℝ
  gh : ℝ
  he : ℝ

/-- Theorem stating the area of the extended quadrilateral -/
theorem extended_quadrilateral_area (q : ExtendedQuadrilateral)
  (h_area : q.area = 25)
  (h_ef : q.ef = 5)
  (h_fg : q.fg = 7)
  (h_gh : q.gh = 9)
  (h_he : q.he = 8) :
  q.area + 2 * q.area = 75 := by
  sorry

end extended_quadrilateral_area_l2437_243759


namespace largest_divisor_of_product_l2437_243712

theorem largest_divisor_of_product (n : ℕ) (h : Even n) (h' : n > 0) :
  (∃ (k : ℕ), (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) = 105 * k) ∧
  (∀ (m : ℕ), m > 105 → ¬(∀ (n : ℕ), Even n → n > 0 →
    ∃ (k : ℕ), (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) = m * k)) :=
by sorry

end largest_divisor_of_product_l2437_243712


namespace fraction_sum_equality_l2437_243757

theorem fraction_sum_equality : (3 : ℚ) / 5 - 1 / 10 + 2 / 15 = 19 / 30 := by
  sorry

end fraction_sum_equality_l2437_243757


namespace correct_average_marks_l2437_243753

/-- Proves that the correct average marks for a class of 50 students is 82.8,
    given an initial average of 85 and three incorrectly recorded marks. -/
theorem correct_average_marks
  (num_students : ℕ)
  (initial_average : ℚ)
  (incorrect_mark1 incorrect_mark2 incorrect_mark3 : ℕ)
  (correct_mark1 correct_mark2 correct_mark3 : ℕ)
  (h_num_students : num_students = 50)
  (h_initial_average : initial_average = 85)
  (h_incorrect1 : incorrect_mark1 = 95)
  (h_incorrect2 : incorrect_mark2 = 78)
  (h_incorrect3 : incorrect_mark3 = 120)
  (h_correct1 : correct_mark1 = 45)
  (h_correct2 : correct_mark2 = 58)
  (h_correct3 : correct_mark3 = 80) :
  (num_students : ℚ) * initial_average - (incorrect_mark1 - correct_mark1 + incorrect_mark2 - correct_mark2 + incorrect_mark3 - correct_mark3 : ℚ) / num_students = 82.8 :=
by
  sorry

end correct_average_marks_l2437_243753


namespace expected_BBR_sequences_l2437_243700

/-- Represents a standard deck of 52 cards -/
structure Deck :=
  (cards : Fin 52)

/-- Represents a sequence of three cards -/
structure ThreeCardSequence :=
  (first : Deck)
  (second : Deck)
  (third : Deck)

/-- Checks if a card is black -/
def is_black (card : Deck) : Prop :=
  sorry

/-- Checks if a card is red -/
def is_red (card : Deck) : Prop :=
  sorry

/-- Checks if a sequence is BBR (two black cards followed by a red card) -/
def is_BBR (seq : ThreeCardSequence) : Prop :=
  is_black seq.first ∧ is_black seq.second ∧ is_red seq.third

/-- The probability of a specific BBR sequence -/
def BBR_probability : ℚ :=
  13 / 51

/-- The number of possible starting positions for a BBR sequence -/
def num_starting_positions : ℕ :=
  26

/-- The expected number of BBR sequences in a standard 52-card deck dealt in a circle -/
theorem expected_BBR_sequences :
  (num_starting_positions : ℚ) * BBR_probability = 338 / 51 :=
sorry

end expected_BBR_sequences_l2437_243700
