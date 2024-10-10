import Mathlib

namespace largest_inscribed_circle_radius_l2617_261770

/-- A quadrilateral with given side lengths -/
structure Quadrilateral :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DA : ℝ)

/-- The largest possible inscribed circle in a quadrilateral -/
def largest_inscribed_circle (q : Quadrilateral) : ℝ := sorry

/-- Theorem: The radius of the largest inscribed circle in the given quadrilateral is 2√6 -/
theorem largest_inscribed_circle_radius :
  ∀ q : Quadrilateral,
    q.AB = 15 ∧ q.BC = 10 ∧ q.CD = 8 ∧ q.DA = 13 →
    largest_inscribed_circle q = 2 * Real.sqrt 6 :=
by sorry

end largest_inscribed_circle_radius_l2617_261770


namespace min_value_theorem_l2617_261793

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 / x + 1 / y = 5) :
  3 * x + 4 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 / x₀ + 1 / y₀ = 5 ∧ 3 * x₀ + 4 * y₀ = 5 :=
sorry

end min_value_theorem_l2617_261793


namespace red_light_is_random_event_l2617_261743

/-- Definition of a random event -/
def is_random_event (event : Type) : Prop :=
  ∃ (outcome : event → Prop) (probability : event → ℝ),
    (∀ e : event, 0 ≤ probability e ∧ probability e ≤ 1) ∧
    (∀ e : event, outcome e ↔ probability e > 0)

/-- Representation of passing through an intersection with a traffic signal -/
inductive TrafficSignalEvent
| RedLight
| GreenLight
| YellowLight

/-- Theorem stating that encountering a red light at an intersection is a random event -/
theorem red_light_is_random_event :
  is_random_event TrafficSignalEvent :=
sorry

end red_light_is_random_event_l2617_261743


namespace counterexample_21_l2617_261703

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem counterexample_21 :
  ¬(is_prime 21) ∧ ¬(is_prime (21 + 3)) :=
by sorry

end counterexample_21_l2617_261703


namespace sqrt_D_rationality_l2617_261732

/-- Given integers a and b where b = a + 2, and c = ab, 
    D is defined as a² + b² + c². This theorem states that 
    √D can be either rational or irrational. -/
theorem sqrt_D_rationality (a : ℤ) : 
  ∃ (D : ℚ), D = (a^2 : ℚ) + ((a+2)^2 : ℚ) + ((a*(a+2))^2 : ℚ) ∧ 
  (∃ (x : ℚ), x^2 = D) ∨ (∀ (x : ℚ), x^2 ≠ D) :=
sorry

end sqrt_D_rationality_l2617_261732


namespace robins_pieces_l2617_261730

theorem robins_pieces (gum_packages : ℕ) (candy_packages : ℕ) (pieces_per_package : ℕ) : 
  gum_packages = 28 → candy_packages = 14 → pieces_per_package = 6 →
  gum_packages * pieces_per_package + candy_packages * pieces_per_package = 252 := by
sorry

end robins_pieces_l2617_261730


namespace age_interchange_problem_l2617_261783

theorem age_interchange_problem :
  let valid_pair := λ (t n : ℕ) =>
    t > 30 ∧
    n > 0 ∧
    30 + n < 100 ∧
    t + n < 100 ∧
    (t + n) / 10 = (30 + n) % 10 ∧
    (t + n) % 10 = (30 + n) / 10
  (∃! l : List (ℕ × ℕ), l.length = 21 ∧ ∀ p ∈ l, valid_pair p.1 p.2) :=
by sorry

end age_interchange_problem_l2617_261783


namespace statue_cost_proof_l2617_261751

theorem statue_cost_proof (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) :
  selling_price = 620 →
  profit_percentage = 0.25 →
  selling_price = original_cost * (1 + profit_percentage) →
  original_cost = 496 :=
by sorry

end statue_cost_proof_l2617_261751


namespace carl_needs_sixty_more_bags_l2617_261792

/-- The number of additional gift bags Carl needs to make for his open house -/
def additional_bags_needed (guaranteed_visitors : ℕ) (possible_visitors : ℕ) (extravagant_bags : ℕ) (average_bags : ℕ) : ℕ :=
  (guaranteed_visitors + possible_visitors) - (extravagant_bags + average_bags)

/-- Theorem stating that Carl needs to make 60 more gift bags -/
theorem carl_needs_sixty_more_bags :
  additional_bags_needed 50 40 10 20 = 60 := by
  sorry

end carl_needs_sixty_more_bags_l2617_261792


namespace widest_strip_width_l2617_261787

theorem widest_strip_width (w1 w2 w3 : ℕ) (hw1 : w1 = 45) (hw2 : w2 = 60) (hw3 : w3 = 70) :
  Nat.gcd w1 (Nat.gcd w2 w3) = 5 := by
  sorry

end widest_strip_width_l2617_261787


namespace johnson_family_reunion_l2617_261715

theorem johnson_family_reunion (num_children : ℕ) (num_adults : ℕ) (num_blue_adults : ℕ) : 
  num_children = 45 →
  num_adults = num_children / 3 →
  num_blue_adults = num_adults / 3 →
  num_adults - num_blue_adults = 10 :=
by
  sorry

end johnson_family_reunion_l2617_261715


namespace poster_spacing_proof_l2617_261722

/-- Calculates the equal distance between posters and from the ends of the wall -/
def equal_distance (wall_width : ℕ) (num_posters : ℕ) (poster_width : ℕ) : ℕ :=
  (wall_width - num_posters * poster_width) / (num_posters + 1)

/-- Theorem stating that the equal distance is 20 cm given the problem conditions -/
theorem poster_spacing_proof :
  equal_distance 320 6 30 = 20 := by
  sorry

end poster_spacing_proof_l2617_261722


namespace deleted_pictures_count_l2617_261735

def zoo_pictures : ℕ := 15
def museum_pictures : ℕ := 18
def remaining_pictures : ℕ := 2

theorem deleted_pictures_count :
  zoo_pictures + museum_pictures - remaining_pictures = 31 := by
  sorry

end deleted_pictures_count_l2617_261735


namespace points_per_game_l2617_261774

theorem points_per_game (total_points : ℕ) (num_games : ℕ) (points_per_game : ℕ) : 
  total_points = 91 → 
  num_games = 13 → 
  total_points = num_games * points_per_game → 
  points_per_game = 7 := by
sorry

end points_per_game_l2617_261774


namespace range_of_a_l2617_261779

-- Define the complex number z
def z (a : ℝ) : ℂ := 1 + a * Complex.I

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (Complex.abs (z a) ≤ 2) ↔ (a ≥ -Real.sqrt 3 ∧ a ≤ Real.sqrt 3) :=
by sorry

end range_of_a_l2617_261779


namespace total_amount_paid_l2617_261782

-- Define the purchased amounts, rates, and discounts
def grape_amount : ℝ := 3
def mango_amount : ℝ := 9
def orange_amount : ℝ := 5
def banana_amount : ℝ := 7

def grape_rate : ℝ := 70
def mango_rate : ℝ := 55
def orange_rate : ℝ := 40
def banana_rate : ℝ := 20

def grape_discount : ℝ := 0.05
def mango_discount : ℝ := 0.10
def orange_discount : ℝ := 0.08
def banana_discount : ℝ := 0

def sales_tax : ℝ := 0.05

-- Define the theorem
theorem total_amount_paid : 
  let grape_cost := grape_amount * grape_rate
  let mango_cost := mango_amount * mango_rate
  let orange_cost := orange_amount * orange_rate
  let banana_cost := banana_amount * banana_rate

  let grape_discounted := grape_cost * (1 - grape_discount)
  let mango_discounted := mango_cost * (1 - mango_discount)
  let orange_discounted := orange_cost * (1 - orange_discount)
  let banana_discounted := banana_cost * (1 - banana_discount)

  let total_discounted := grape_discounted + mango_discounted + orange_discounted + banana_discounted
  let total_with_tax := total_discounted * (1 + sales_tax)

  total_with_tax = 1017.45 := by
    sorry

end total_amount_paid_l2617_261782


namespace inequality_implication_l2617_261727

theorem inequality_implication (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 * (3^(2*x)) - 3^x + a^2 - a - 3 > 0) → 
  a < -1 ∨ a > 2 := by
sorry

end inequality_implication_l2617_261727


namespace prism_has_315_edges_l2617_261723

/-- A prism is a polyhedron with two congruent and parallel faces (bases) connected by rectangular faces. -/
structure Prism where
  num_edges : ℕ

/-- The number of edges in a prism is always a multiple of 3. -/
axiom prism_edges_multiple_of_three (p : Prism) : ∃ k : ℕ, p.num_edges = 3 * k

/-- The prism has more than 310 edges. -/
axiom edges_greater_than_310 (p : Prism) : p.num_edges > 310

/-- The prism has fewer than 320 edges. -/
axiom edges_less_than_320 (p : Prism) : p.num_edges < 320

/-- The number of edges in the prism is odd. -/
axiom edges_odd (p : Prism) : Odd p.num_edges

theorem prism_has_315_edges (p : Prism) : p.num_edges = 315 := by
  sorry

end prism_has_315_edges_l2617_261723


namespace positive_rational_function_uniqueness_l2617_261716

/-- A function from positive rationals to positive rationals -/
def PositiveRationalFunction := {f : ℚ → ℚ // ∀ x, 0 < x → 0 < f x}

/-- The property that f(x+1) = f(x) + 1 for all positive rationals x -/
def HasUnitPeriod (f : PositiveRationalFunction) : Prop :=
  ∀ x : ℚ, 0 < x → f.val (x + 1) = f.val x + 1

/-- The property that f(1/x) = 1/f(x) for all positive rationals x -/
def HasInverseProperty (f : PositiveRationalFunction) : Prop :=
  ∀ x : ℚ, 0 < x → f.val (1 / x) = 1 / f.val x

/-- The main theorem: if a function satisfies both properties, it must be the identity function -/
theorem positive_rational_function_uniqueness (f : PositiveRationalFunction) 
    (h1 : HasUnitPeriod f) (h2 : HasInverseProperty f) : 
    ∀ x : ℚ, 0 < x → f.val x = x := by
  sorry

end positive_rational_function_uniqueness_l2617_261716


namespace strange_number_theorem_l2617_261717

theorem strange_number_theorem : ∃! x : ℝ, (x - 7) * 7 = (x - 11) * 11 := by
  sorry

end strange_number_theorem_l2617_261717


namespace cube_surface_area_ratio_l2617_261719

theorem cube_surface_area_ratio :
  let original_volume : ℝ := 1000
  let removed_volume : ℝ := 64
  let original_side : ℝ := original_volume ^ (1/3)
  let removed_side : ℝ := removed_volume ^ (1/3)
  let shaded_area : ℝ := removed_side ^ 2
  let total_surface_area : ℝ := 
    3 * original_side ^ 2 + 
    3 * removed_side ^ 2 + 
    3 * (original_side ^ 2 - removed_side ^ 2)
  shaded_area / total_surface_area = 2 / 75
  := by sorry

end cube_surface_area_ratio_l2617_261719


namespace right_triangle_from_equations_l2617_261758

theorem right_triangle_from_equations (a b c x : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a + b > c → b + c > a → c + a > b →
  x^2 + 2*a*x + b^2 = 0 →
  x^2 + 2*c*x - b^2 = 0 →
  a^2 = b^2 + c^2 :=
by sorry

end right_triangle_from_equations_l2617_261758


namespace expression_equals_one_l2617_261705

theorem expression_equals_one (x z : ℝ) (h1 : x ≠ z) (h2 : x ≠ -z) :
  (x / (x - z) - z / (x + z)) / (z / (x - z) + x / (x + z)) = 1 := by
  sorry

end expression_equals_one_l2617_261705


namespace candidate_x_wins_by_16_percent_l2617_261777

/-- Represents the election scenario with given conditions -/
structure ElectionScenario where
  repubRatio : ℚ
  demRatio : ℚ
  repubVoteX : ℚ
  demVoteX : ℚ
  (ratio_positive : repubRatio > 0 ∧ demRatio > 0)
  (vote_percentages : repubVoteX ≥ 0 ∧ repubVoteX ≤ 1 ∧ demVoteX ≥ 0 ∧ demVoteX ≤ 1)

/-- Calculates the percentage by which candidate X is expected to win -/
def winPercentage (e : ElectionScenario) : ℚ :=
  let totalVoters := e.repubRatio + e.demRatio
  let votesForX := e.repubRatio * e.repubVoteX + e.demRatio * e.demVoteX
  let votesForY := totalVoters - votesForX
  (votesForX - votesForY) / totalVoters * 100

/-- Theorem stating that under the given conditions, candidate X wins by 16% -/
theorem candidate_x_wins_by_16_percent :
  ∀ e : ElectionScenario,
    e.repubRatio = 3 ∧
    e.demRatio = 2 ∧
    e.repubVoteX = 4/5 ∧
    e.demVoteX = 1/4 →
    winPercentage e = 16 := by
  sorry


end candidate_x_wins_by_16_percent_l2617_261777


namespace balloon_solution_l2617_261784

/-- The number of balloons Allan and Jake have in the park -/
def balloon_problem (allan_balloons jake_initial_balloons jake_bought_balloons : ℕ) : Prop :=
  allan_balloons - (jake_initial_balloons + jake_bought_balloons) = 1

/-- Theorem stating the solution to the balloon problem -/
theorem balloon_solution :
  balloon_problem 6 2 3 := by
  sorry

end balloon_solution_l2617_261784


namespace smallest_number_of_eggs_l2617_261749

theorem smallest_number_of_eggs : ∀ n : ℕ,
  (∃ c : ℕ, n = 12 * c - 3) →  -- Eggs are in containers of 12, with 3 containers having 11 eggs
  n > 200 →                   -- More than 200 eggs
  n ≥ 201                     -- The smallest possible number is at least 201
:= by
  sorry

end smallest_number_of_eggs_l2617_261749


namespace birthday_stickers_l2617_261738

theorem birthday_stickers (initial_stickers final_stickers : ℕ) 
  (h1 : initial_stickers = 39)
  (h2 : final_stickers = 61) :
  final_stickers - initial_stickers = 22 := by
sorry

end birthday_stickers_l2617_261738


namespace continuous_function_solution_l2617_261712

open Set
open Function
open Real

theorem continuous_function_solution {f : ℝ → ℝ} (hf : Continuous f) 
  (hdom : ∀ x, x ∈ Ioo (-1) 1 → f x ≠ 0) 
  (heq : ∀ x ∈ Ioo (-1) 1, (1 - x^2) * f ((2*x) / (1 + x^2)) = (1 + x^2)^2 * f x) :
  ∃ c : ℝ, ∀ x ∈ Ioo (-1) 1, f x = c / (1 - x^2) :=
sorry

end continuous_function_solution_l2617_261712


namespace gcd_105_90_l2617_261773

theorem gcd_105_90 : Nat.gcd 105 90 = 15 := by
  sorry

end gcd_105_90_l2617_261773


namespace chord_length_line_circle_intersection_l2617_261745

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length_line_circle_intersection (x y : ℝ) :
  (3 * x + 4 * y - 5 = 0) →  -- Line equation
  (x^2 + y^2 = 4) →          -- Circle equation
  ∃ (A B : ℝ × ℝ),           -- Intersection points A and B
    (3 * A.1 + 4 * A.2 - 5 = 0) ∧
    (A.1^2 + A.2^2 = 4) ∧
    (3 * B.1 + 4 * B.2 - 5 = 0) ∧
    (B.1^2 + B.2^2 = 4) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 :=
by
  sorry

end chord_length_line_circle_intersection_l2617_261745


namespace grey_pairs_coincide_l2617_261744

/-- Represents the number of triangles of each color in one half of the shape -/
structure TriangleCounts where
  orange : Nat
  green : Nat
  grey : Nat

/-- Represents the number of pairs of triangles that coincide when folded -/
structure CoincidingPairs where
  orange : Nat
  green : Nat
  orangeGrey : Nat

theorem grey_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) :
  counts.orange = 4 →
  counts.green = 6 →
  counts.grey = 9 →
  pairs.orange = 3 →
  pairs.green = 4 →
  pairs.orangeGrey = 1 →
  ∃ (grey_pairs : Nat), grey_pairs = 6 ∧ 
    grey_pairs = counts.grey - (pairs.orangeGrey + (counts.green - 2 * pairs.green)) :=
by sorry

end grey_pairs_coincide_l2617_261744


namespace geometric_progression_proof_l2617_261731

/-- Given a geometric progression with b₃ = -1 and b₆ = 27/8,
    prove that the first term b₁ = -4/9 and the common ratio q = -3/2 -/
theorem geometric_progression_proof (b : ℕ → ℚ) :
  b 3 = -1 ∧ b 6 = 27/8 →
  (∃ q : ℚ, ∀ n : ℕ, b (n + 1) = b n * q) →
  b 1 = -4/9 ∧ (∀ n : ℕ, b (n + 1) = b n * (-3/2)) := by
sorry

end geometric_progression_proof_l2617_261731


namespace discount_percentage_proof_l2617_261772

theorem discount_percentage_proof (coat_price pants_price : ℝ)
  (coat_discount pants_discount : ℝ) :
  coat_price = 100 →
  pants_price = 50 →
  coat_discount = 0.3 →
  pants_discount = 0.4 →
  let total_original := coat_price + pants_price
  let total_savings := coat_price * coat_discount + pants_price * pants_discount
  let savings_percentage := total_savings / total_original * 100
  savings_percentage = 100 / 3 := by
sorry

end discount_percentage_proof_l2617_261772


namespace multiples_of_5_ending_in_0_less_than_200_l2617_261775

def count_multiples_of_5_ending_in_0 (upper_bound : ℕ) : ℕ :=
  (upper_bound - 1) / 10

theorem multiples_of_5_ending_in_0_less_than_200 :
  count_multiples_of_5_ending_in_0 200 = 19 := by
  sorry

end multiples_of_5_ending_in_0_less_than_200_l2617_261775


namespace total_annual_income_percentage_l2617_261714

def initial_investment : ℝ := 2800
def initial_rate : ℝ := 0.05
def additional_investment : ℝ := 1400
def additional_rate : ℝ := 0.08

theorem total_annual_income_percentage :
  let total_investment := initial_investment + additional_investment
  let total_income := initial_investment * initial_rate + additional_investment * additional_rate
  (total_income / total_investment) * 100 = 6 := by
sorry

end total_annual_income_percentage_l2617_261714


namespace polynomial_derivative_value_l2617_261721

theorem polynomial_derivative_value (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (3*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 240 := by
sorry

end polynomial_derivative_value_l2617_261721


namespace special_function_range_l2617_261797

/-- A monotonically increasing function satisfying the given properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 < x → 0 < y → f (x * y) = f x + f y) ∧
  (∀ x y, 0 < x → 0 < y → x < y → f x < f y) ∧
  (f 3 = 1)

/-- The theorem statement -/
theorem special_function_range (f : ℝ → ℝ) (hf : SpecialFunction f) :
  {x : ℝ | 0 < x ∧ f x + f (x - 8) ≤ 2} = Set.Ioo 8 9 := by
  sorry

end special_function_range_l2617_261797


namespace red_non_honda_percentage_l2617_261746

/-- Calculates the percentage of red non-Honda cars in Chennai --/
theorem red_non_honda_percentage
  (total_cars : ℕ) 
  (honda_cars : ℕ) 
  (red_honda_ratio : ℚ) 
  (total_red_ratio : ℚ) 
  (h1 : total_cars = 900)
  (h2 : honda_cars = 500)
  (h3 : red_honda_ratio = 90 / 100)
  (h4 : total_red_ratio = 60 / 100) :
  (total_red_ratio * total_cars - red_honda_ratio * honda_cars) / (total_cars - honda_cars) = 9 / 40 :=
by
  sorry

#eval (9 : ℚ) / 40 -- This should evaluate to 0.225 or 22.5%

end red_non_honda_percentage_l2617_261746


namespace triangle_property_l2617_261790

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- S is the area of triangle ABC -/
def area (t : Triangle) : ℝ := sorry

/-- Main theorem -/
theorem triangle_property (t : Triangle) (h : 4 * Real.sqrt 3 * area t = t.a^2 - (t.b - t.c)^2) :
  t.A = 2 * Real.pi / 3 ∧ 2 / 3 ≤ (t.b^2 + t.c^2) / t.a^2 ∧ (t.b^2 + t.c^2) / t.a^2 < 1 :=
by sorry

end

end triangle_property_l2617_261790


namespace group_photo_arrangements_eq_12_l2617_261763

/-- The number of ways to arrange 1 teacher, 2 female students, and 2 male students in a row,
    where the two female students are separated only by the teacher. -/
def group_photo_arrangements : ℕ :=
  let teacher : ℕ := 1
  let female_students : ℕ := 2
  let male_students : ℕ := 2
  let teacher_and_females : ℕ := 1  -- Treat teacher and females as one unit
  let remaining_elements : ℕ := teacher_and_females + male_students
  (female_students.factorial) * (remaining_elements.factorial)

theorem group_photo_arrangements_eq_12 : group_photo_arrangements = 12 := by
  sorry

end group_photo_arrangements_eq_12_l2617_261763


namespace lcm_of_numbers_in_ratio_l2617_261762

theorem lcm_of_numbers_in_ratio (a b : ℕ) (h_ratio : a * 5 = b * 4) (h_smaller : a = 36) : 
  Nat.lcm a b = 1620 := by
  sorry

end lcm_of_numbers_in_ratio_l2617_261762


namespace squares_after_six_operations_l2617_261781

/-- Calculates the number of squares after n operations -/
def num_squares (n : ℕ) : ℕ := 5 + 3 * n

/-- The number of squares after 6 operations is 29 -/
theorem squares_after_six_operations :
  num_squares 6 = 29 := by
  sorry

end squares_after_six_operations_l2617_261781


namespace triangle_altitude_l2617_261759

/-- Given a triangle with area 720 square feet and base 36 feet, prove its altitude is 40 feet -/
theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) :
  area = 720 →
  base = 36 →
  area = (1/2) * base * altitude →
  altitude = 40 := by
sorry

end triangle_altitude_l2617_261759


namespace sum_of_cubes_l2617_261788

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : a^3 + b^3 = 1008 := by
  sorry

end sum_of_cubes_l2617_261788


namespace cone_slant_height_l2617_261768

/-- 
Given a cone whose lateral surface unfolds to a semicircle and whose base radius is 1,
prove that its slant height is 2.
-/
theorem cone_slant_height (r : ℝ) (l : ℝ) : 
  r = 1 → -- radius of the base is 1
  2 * π * r = π * l → -- lateral surface unfolds to a semicircle
  l = 2 := by sorry

end cone_slant_height_l2617_261768


namespace tangent_line_and_positivity_l2617_261798

open Real

noncomputable def f (a x : ℝ) : ℝ := (x - a) * log x + (1/2) * x

theorem tangent_line_and_positivity (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (∀ x : ℝ, (f a x₀ - f a x) = (1/2) * (x₀ - x)) → a = 1) ∧
  ((1/(2*exp 1)) < a ∧ a < 2 * sqrt (exp 1) → 
    ∀ x : ℝ, x > 0 → f a x > 0) :=
sorry

end tangent_line_and_positivity_l2617_261798


namespace problem_solution_l2617_261756

theorem problem_solution (a b : ℝ) (h1 : a - b = 7) (h2 : a * b = 18) :
  a^2 + b^2 = 85 ∧ (a + b)^2 = 121 := by
  sorry

end problem_solution_l2617_261756


namespace alcohol_solution_percentage_l2617_261706

theorem alcohol_solution_percentage 
  (initial_volume : ℝ) 
  (initial_percentage : ℝ) 
  (added_alcohol : ℝ) 
  (added_water : ℝ) 
  (h1 : initial_volume = 40)
  (h2 : initial_percentage = 5)
  (h3 : added_alcohol = 5.5)
  (h4 : added_water = 4.5) :
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let new_alcohol := initial_alcohol + added_alcohol
  let new_volume := initial_volume + added_alcohol + added_water
  let new_percentage := (new_alcohol / new_volume) * 100
  new_percentage = 15 := by
sorry

end alcohol_solution_percentage_l2617_261706


namespace x_range_l2617_261742

theorem x_range (x : ℝ) (h1 : 1 / x ≤ 3) (h2 : 1 / x ≥ -2) : x ≥ 1 / 3 := by
  sorry

end x_range_l2617_261742


namespace other_items_percentage_correct_l2617_261726

/-- The percentage of money spent on other items in Jill's shopping trip -/
def other_items_percentage : ℝ := 
  let total := 100
  let clothing_percentage := 45
  let food_percentage := 45
  let clothing_tax_rate := 5
  let other_items_tax_rate := 10
  let total_tax_percentage := 3.25
  10

/-- Theorem stating that the percentage spent on other items is correct -/
theorem other_items_percentage_correct : 
  let total := 100
  let clothing_percentage := 45
  let food_percentage := 45
  let clothing_tax_rate := 5
  let other_items_tax_rate := 10
  let total_tax_percentage := 3.25
  (clothing_percentage + food_percentage + other_items_percentage = total) ∧
  (clothing_tax_rate * clothing_percentage / 100 + 
   other_items_tax_rate * other_items_percentage / 100 = total_tax_percentage) := by
  sorry

#check other_items_percentage_correct

end other_items_percentage_correct_l2617_261726


namespace amber_guppies_problem_l2617_261713

/-- The number of guppies Amber initially bought -/
def initial_guppies : ℕ := 7

/-- The number of baby guppies Amber saw in the first sighting (3 dozen) -/
def first_sighting : ℕ := 36

/-- The total number of guppies Amber has after the second sighting -/
def total_guppies : ℕ := 52

/-- The number of additional baby guppies Amber saw two days after the first sighting -/
def additional_guppies : ℕ := total_guppies - (initial_guppies + first_sighting)

theorem amber_guppies_problem :
  additional_guppies = 9 := by sorry

end amber_guppies_problem_l2617_261713


namespace max_x_value_l2617_261718

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 10) (prod_eq : x*y + x*z + y*z = 20) :
  x ≤ 10/3 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ + y₀ + z₀ = 10 ∧ x₀*y₀ + x₀*z₀ + y₀*z₀ = 20 ∧ x₀ = 10/3 :=
by sorry

end max_x_value_l2617_261718


namespace prop_a_prop_b_prop_c_prop_d_l2617_261737

-- Proposition A
theorem prop_a (a b : ℝ) (h : b > a ∧ a > 0) : 1 / a > 1 / b := by sorry

-- Proposition B
theorem prop_b : ∃ a b c : ℝ, a > b ∧ a * c ≤ b * c := by sorry

-- Proposition C
theorem prop_c (a b c : ℝ) (h1 : a * c^2 > b * c^2) (h2 : c ≠ 0) : a > b := by sorry

-- Proposition D
theorem prop_d : 
  (∃ x : ℝ, x > -3 ∧ x^2 ≤ 9) ↔ ¬(∀ x : ℝ, x > -3 → x^2 > 9) := by sorry

end prop_a_prop_b_prop_c_prop_d_l2617_261737


namespace prime_divisors_condition_l2617_261728

theorem prime_divisors_condition (a n : ℕ) (ha : a > 2) :
  (∀ p : ℕ, Nat.Prime p → p ∣ (a^n - 1) → p ∣ (a^(3^2016) - 1)) →
  ∃ l : ℕ, l > 0 ∧ a = 2^l - 1 ∧ n = 2 := by
  sorry

end prime_divisors_condition_l2617_261728


namespace complex_equation_solution_l2617_261786

theorem complex_equation_solution : ∃ (a : ℝ), 
  (1 - Complex.I : ℂ) = (2 + a * Complex.I) / (1 + Complex.I) ∧ a = 0 := by
  sorry

end complex_equation_solution_l2617_261786


namespace same_lunch_group_probability_l2617_261711

/-- The number of students in the school -/
def total_students : ℕ := 900

/-- The number of lunch groups -/
def num_groups : ℕ := 6

/-- The number of students in each lunch group -/
def students_per_group : ℕ := total_students / num_groups

/-- The probability of a single student being assigned to a specific group -/
def prob_single_student : ℚ := 1 / num_groups

/-- The number of specific students we're interested in -/
def num_specific_students : ℕ := 4

theorem same_lunch_group_probability :
  (prob_single_student ^ (num_specific_students - 1) : ℚ) = 1 / 216 :=
sorry

end same_lunch_group_probability_l2617_261711


namespace range_of_m_l2617_261750

-- Define the set A
def A : Set ℝ := {y | ∃ x ∈ Set.Icc (1/4 : ℝ) 2, y = x^2 - (3/2)*x + 1}

-- Define the set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x + m^2 ≥ 1}

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (m : ℝ) (x : ℝ) : Prop := x ∈ B m

-- State the theorem
theorem range_of_m :
  (∀ x, p x → q m x) ↔ (m ≥ 3/4 ∨ m ≤ -3/4) :=
sorry

end range_of_m_l2617_261750


namespace function_range_l2617_261725

theorem function_range (θ : ℝ) : 
  ∀ x : ℝ, 2 - Real.sqrt 3 ≤ (x^2 + 2*x*Real.sin θ + 2) / (x^2 + 2*x*Real.cos θ + 2) 
         ∧ (x^2 + 2*x*Real.sin θ + 2) / (x^2 + 2*x*Real.cos θ + 2) ≤ 2 + Real.sqrt 3 := by
  sorry

end function_range_l2617_261725


namespace a_value_when_A_equals_B_l2617_261791

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + 1 = 0}

-- Define the set B
def B : Set ℝ := {1, 2}

-- Theorem statement
theorem a_value_when_A_equals_B (a : ℝ) : A a = B → a = -3 := by
  sorry

end a_value_when_A_equals_B_l2617_261791


namespace triangle_ABC_properties_l2617_261752

theorem triangle_ABC_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧
  (A + B + C = π) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) ∧ (b / Real.sin B = c / Real.sin C) →
  -- Given equation
  (Real.sqrt 3 * Real.sin B + b * Real.cos A = c) →
  -- Prove angle B
  (B = π / 6) ∧
  -- Prove area when a = √3 * c and b = 2
  (a = Real.sqrt 3 * c ∧ b = 2 → 
   (1 / 2) * a * b * Real.sin C = Real.sqrt 3) := by
sorry

end triangle_ABC_properties_l2617_261752


namespace largest_sum_and_simplification_l2617_261771

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/5 + 1/2, 1/5 + 1/6, 1/5 + 1/4, 1/5 + 1/8, 1/5 + 1/9]
  (∀ x ∈ sums, 1/5 + 1/2 ≥ x) ∧ (1/5 + 1/2 = 7/10) :=
by sorry

end largest_sum_and_simplification_l2617_261771


namespace A_knitting_time_l2617_261764

/-- The number of days it takes person A to knit a pair of socks -/
def days_A : ℝ := by sorry

/-- The number of days it takes person B to knit a pair of socks -/
def days_B : ℝ := 6

/-- The number of days it takes A and B together to knit two pairs of socks -/
def days_together : ℝ := 4

/-- The number of pairs of socks A and B knit together in 4 days -/
def pairs_together : ℝ := 2

theorem A_knitting_time :
  (1 / days_A + 1 / days_B) * days_together = pairs_together ∧ days_A = 3 := by sorry

end A_knitting_time_l2617_261764


namespace greatest_n_roots_on_unit_circle_l2617_261700

theorem greatest_n_roots_on_unit_circle : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (z : ℂ), z ≠ 0 → (z + 1)^n = z^n + 1 → Complex.abs z = 1) ∧
  (∀ (m : ℕ), m > n → ∃ (w : ℂ), w ≠ 0 ∧ (w + 1)^m = w^m + 1 ∧ Complex.abs w ≠ 1) ∧
  n = 7 := by
sorry

end greatest_n_roots_on_unit_circle_l2617_261700


namespace cost_price_calculation_l2617_261789

-- Define the markup percentage
def markup : ℝ := 0.15

-- Define the selling price
def selling_price : ℝ := 6400

-- Theorem statement
theorem cost_price_calculation :
  ∃ (cost_price : ℝ), cost_price * (1 + markup) = selling_price :=
by
  sorry


end cost_price_calculation_l2617_261789


namespace divisibility_1001_l2617_261766

theorem divisibility_1001 (n : ℕ) : 1001 ∣ n → 7 ∣ n ∧ 11 ∣ n ∧ 13 ∣ n := by
  sorry

end divisibility_1001_l2617_261766


namespace organizationalStructureIsCorrect_l2617_261720

-- Define the types of diagrams
inductive Diagram
  | Flowchart
  | ProcessFlow
  | KnowledgeStructure
  | OrganizationalStructure

-- Define the properties a diagram should have
structure DiagramProperties where
  reflectsRelationships : Bool
  showsVerticalHorizontal : Bool
  reflectsOrganizationalStructure : Bool
  interpretsOrganizationalFunctions : Bool

-- Define a function to check if a diagram has the required properties
def hasRequiredProperties (d : Diagram) : DiagramProperties :=
  match d with
  | Diagram.OrganizationalStructure => {
      reflectsRelationships := true,
      showsVerticalHorizontal := true,
      reflectsOrganizationalStructure := true,
      interpretsOrganizationalFunctions := true
    }
  | _ => {
      reflectsRelationships := false,
      showsVerticalHorizontal := false,
      reflectsOrganizationalStructure := false,
      interpretsOrganizationalFunctions := false
    }

-- Theorem: The Organizational Structure Diagram is the correct choice for describing factory composition
theorem organizationalStructureIsCorrect :
  ∀ (d : Diagram),
    (hasRequiredProperties d).reflectsRelationships ∧
    (hasRequiredProperties d).showsVerticalHorizontal ∧
    (hasRequiredProperties d).reflectsOrganizationalStructure ∧
    (hasRequiredProperties d).interpretsOrganizationalFunctions
    →
    d = Diagram.OrganizationalStructure :=
  sorry

end organizationalStructureIsCorrect_l2617_261720


namespace ellipse_condition_l2617_261708

def ellipse_equation (x y k : ℝ) : Prop :=
  3 * x^2 + 6 * y^2 - 12 * x + 18 * y = k

def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  ∃ (a b h c d : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), ellipse_equation x y k ↔ (x - h)^2 / a^2 + (y - d)^2 / b^2 = 1

theorem ellipse_condition (k : ℝ) :
  is_non_degenerate_ellipse k ↔ k > -51/2 :=
sorry

end ellipse_condition_l2617_261708


namespace distribute_teachers_count_l2617_261796

/-- The number of ways to distribute 6 teachers across 4 neighborhoods --/
def distribute_teachers : ℕ :=
  let n_teachers : ℕ := 6
  let n_neighborhoods : ℕ := 4
  let distribution_3111 : ℕ := (Nat.choose n_teachers 3) * (Nat.factorial n_neighborhoods)
  let distribution_2211 : ℕ := 
    (Nat.choose n_teachers 2) * (Nat.choose (n_teachers - 2) 2) * 
    (Nat.factorial n_neighborhoods) / (Nat.factorial 2)
  distribution_3111 + distribution_2211

/-- Theorem stating that the number of distribution schemes is 1560 --/
theorem distribute_teachers_count : distribute_teachers = 1560 := by
  sorry

end distribute_teachers_count_l2617_261796


namespace product_evaluation_l2617_261707

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end product_evaluation_l2617_261707


namespace radical_combination_l2617_261748

theorem radical_combination (x : ℝ) : (2 + x = 5 - 2*x) → x = 1 := by
  sorry

end radical_combination_l2617_261748


namespace polar_to_rectangular_conversion_l2617_261795

theorem polar_to_rectangular_conversion :
  let r : ℝ := 4
  let θ : ℝ := 3 * Real.pi / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = -2 * Real.sqrt 2 ∧ y = 2 * Real.sqrt 2 := by
  sorry

end polar_to_rectangular_conversion_l2617_261795


namespace tan_product_thirty_degrees_l2617_261769

theorem tan_product_thirty_degrees :
  let A : Real := 30 * π / 180
  let B : Real := 30 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) = (4 + 2 * Real.sqrt 3) / 3 := by
sorry

end tan_product_thirty_degrees_l2617_261769


namespace bridesmaids_count_l2617_261741

/-- Represents the makeup requirements for bridesmaids --/
structure MakeupRequirements where
  lipGlossPerTube : ℕ
  mascaraPerTube : ℕ
  lipGlossTubs : ℕ
  tubesPerLipGlossTub : ℕ
  mascaraTubs : ℕ
  tubesPerMascaraTub : ℕ

/-- Represents the makeup styles chosen by bridesmaids --/
inductive MakeupStyle
  | Glam
  | Natural

/-- Calculates the total number of bridesmaids given the makeup requirements --/
def totalBridesmaids (req : MakeupRequirements) : ℕ :=
  let totalLipGloss := req.lipGlossTubs * req.tubesPerLipGlossTub * req.lipGlossPerTube
  let totalMascara := req.mascaraTubs * req.tubesPerMascaraTub * req.mascaraPerTube
  let glamBridesmaids := totalLipGloss / 3  -- Each glam bridesmaid needs 2 lip gloss + 1 natural
  min glamBridesmaids (totalMascara / 2)  -- Each bridesmaid needs at least 1 mascara

/-- Proves that given the specific makeup requirements, there are 24 bridesmaids --/
theorem bridesmaids_count (req : MakeupRequirements) 
    (h1 : req.lipGlossPerTube = 3)
    (h2 : req.mascaraPerTube = 5)
    (h3 : req.lipGlossTubs = 6)
    (h4 : req.tubesPerLipGlossTub = 2)
    (h5 : req.mascaraTubs = 4)
    (h6 : req.tubesPerMascaraTub = 3) :
    totalBridesmaids req = 24 := by
  sorry

#eval totalBridesmaids { 
  lipGlossPerTube := 3, 
  mascaraPerTube := 5, 
  lipGlossTubs := 6, 
  tubesPerLipGlossTub := 2, 
  mascaraTubs := 4, 
  tubesPerMascaraTub := 3 
}

end bridesmaids_count_l2617_261741


namespace special_hexagon_perimeter_l2617_261702

/-- An equilateral hexagon with specific properties -/
structure SpecialHexagon where
  -- Side length of the hexagon
  side : ℝ
  -- Assertion that the hexagon is equilateral
  is_equilateral : True
  -- Three nonadjacent acute interior angles measure 45°
  has_three_45deg_angles : True
  -- The enclosed area of the hexagon is 12√3
  area_eq_12root3 : side^2 * (3 * Real.sqrt 2 / 4 + Real.sqrt 3 / 2 - Real.sqrt 6 / 4) = 12 * Real.sqrt 3

/-- The perimeter of a SpecialHexagon is 24 -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : h.side * 6 = 24 := by
  sorry

end special_hexagon_perimeter_l2617_261702


namespace same_root_implies_a_equals_three_l2617_261767

theorem same_root_implies_a_equals_three (a : ℝ) :
  (∃ x : ℝ, 3 * x - 2 * a = 0 ∧ 2 * x + 3 * a - 13 = 0) →
  a = 3 := by
sorry

end same_root_implies_a_equals_three_l2617_261767


namespace triangle_angle_determinant_zero_l2617_261739

theorem triangle_angle_determinant_zero (A B C : ℝ) (h : A + B + C = π) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![![Real.cos A ^ 2, Real.tan A, 1],
                                        ![Real.cos B ^ 2, Real.tan B, 1],
                                        ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det M = 0 := by sorry

end triangle_angle_determinant_zero_l2617_261739


namespace c_profit_is_21000_l2617_261729

/-- Calculates the profit for a partner given the total profit, total parts, and the partner's parts. -/
def calculateProfit (totalProfit : ℕ) (totalParts : ℕ) (partnerParts : ℕ) : ℕ :=
  (totalProfit / totalParts) * partnerParts

/-- Proves that given the specified conditions, C's profit is $21000. -/
theorem c_profit_is_21000 (totalProfit : ℕ) (a_parts b_parts c_parts : ℕ) :
  totalProfit = 56700 →
  a_parts = 8 →
  b_parts = 9 →
  c_parts = 10 →
  calculateProfit totalProfit (a_parts + b_parts + c_parts) c_parts = 21000 := by
  sorry

#eval calculateProfit 56700 27 10

end c_profit_is_21000_l2617_261729


namespace chord_intersection_lengths_l2617_261710

/-- Given a circle with radius 7, perpendicular diameters EF and GH, and a chord EJ of length 12
    intersecting GH at M, prove that GM = 7 + √13 and MH = 7 - √13 -/
theorem chord_intersection_lengths (O : ℝ × ℝ) (E F G H J M : ℝ × ℝ) :
  let r : ℝ := 7
  let circle := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}
  (E ∈ circle) ∧ (F ∈ circle) ∧ (G ∈ circle) ∧ (H ∈ circle) ∧ (J ∈ circle) →
  (E.1 - F.1) * (G.2 - H.2) = 0 ∧ (E.2 - F.2) * (G.1 - H.1) = 0 →
  (E.1 - J.1)^2 + (E.2 - J.2)^2 = 12^2 →
  M.1 = (G.1 + H.1) / 2 ∧ M.2 = (G.2 + H.2) / 2 →
  (M.1 - G.1)^2 + (M.2 - G.2)^2 = (7 + Real.sqrt 13)^2 ∧
  (M.1 - H.1)^2 + (M.2 - H.2)^2 = (7 - Real.sqrt 13)^2 :=
by sorry

end chord_intersection_lengths_l2617_261710


namespace max_value_problem_min_value_problem_l2617_261776

theorem max_value_problem (x : ℝ) (h : x < 1) :
  ∃ y : ℝ, y = (4 * x^2 - 3 * x) / (x - 1) ∧ 
  ∀ z : ℝ, z = (4 * x^2 - 3 * x) / (x - 1) → z ≤ y ∧ y = 1 :=
sorry

theorem min_value_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  ∃ y : ℝ, y = 4 / (a + 1) + 1 / b ∧
  ∀ z : ℝ, z = 4 / (a + 1) + 1 / b → y ≤ z ∧ y = 3 + 2 * Real.sqrt 2 :=
sorry

end max_value_problem_min_value_problem_l2617_261776


namespace simplify_expression_l2617_261799

theorem simplify_expression (y : ℝ) : 5*y - 3*y + 7*y - 2*y + 6*y = 13*y := by
  sorry

end simplify_expression_l2617_261799


namespace train_speed_l2617_261778

/-- Given a train crossing a bridge, calculate its speed in km/hr -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 150)
  (h2 : bridge_length = 225)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed

end train_speed_l2617_261778


namespace correct_operation_result_l2617_261736

theorem correct_operation_result (x : ℝ) : 
  (x / 8 - 12 = 32) → (x * 8 + 12 = 2828) := by sorry

end correct_operation_result_l2617_261736


namespace jenny_max_sales_l2617_261785

/-- Represents a neighborhood where Jenny can sell cookies. -/
structure Neighborhood where
  homes : ℕ
  boxesPerHome : ℕ

/-- Calculates the total sales for a given neighborhood. -/
def totalSales (n : Neighborhood) (pricePerBox : ℕ) : ℕ :=
  n.homes * n.boxesPerHome * pricePerBox

/-- Theorem stating that the maximum amount Jenny can make is $50. -/
theorem jenny_max_sales : 
  let neighborhoodA : Neighborhood := { homes := 10, boxesPerHome := 2 }
  let neighborhoodB : Neighborhood := { homes := 5, boxesPerHome := 5 }
  let pricePerBox : ℕ := 2
  max (totalSales neighborhoodA pricePerBox) (totalSales neighborhoodB pricePerBox) = 50 := by
  sorry

end jenny_max_sales_l2617_261785


namespace sum_of_exponents_15_factorial_l2617_261740

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largestPerfectSquareDivisor (n : ℕ) : ℕ :=
  sorry

def sumOfPrimeFactorExponents (n : ℕ) : ℕ :=
  sorry

theorem sum_of_exponents_15_factorial :
  sumOfPrimeFactorExponents (largestPerfectSquareDivisor (factorial 15).sqrt) = 9 :=
sorry

end sum_of_exponents_15_factorial_l2617_261740


namespace product_xyz_l2617_261733

theorem product_xyz (x y z : ℚ) 
  (eq1 : 3 * x + 4 * y = 60)
  (eq2 : 6 * x - 4 * y = 12)
  (eq3 : 2 * x - 3 * z = 38) :
  x * y * z = -1584 / 3 := by
  sorry

end product_xyz_l2617_261733


namespace boys_joining_group_l2617_261754

theorem boys_joining_group (total : ℕ) (initial_boys : ℕ) (initial_girls : ℕ) (boys_joining : ℕ) :
  total = 48 →
  initial_boys + initial_girls = total →
  initial_boys * 5 = initial_girls * 3 →
  (initial_boys + boys_joining) * 3 = initial_girls * 5 →
  boys_joining = 32 := by
sorry

end boys_joining_group_l2617_261754


namespace combined_population_l2617_261753

/-- The combined population of New England and New York given their relative populations -/
theorem combined_population (new_england_pop : ℕ) (new_york_pop : ℕ) :
  new_england_pop = 2100000 →
  new_york_pop = (2 : ℕ) * new_england_pop / 3 →
  new_england_pop + new_york_pop = 3500000 :=
by sorry

end combined_population_l2617_261753


namespace max_pies_without_ingredients_l2617_261701

theorem max_pies_without_ingredients (total_pies : ℕ) 
  (nuts_fraction : ℚ) (berries_fraction : ℚ) (cream_fraction : ℚ) (choc_chips_fraction : ℚ)
  (h_total : total_pies = 48)
  (h_nuts : nuts_fraction = 1/3)
  (h_berries : berries_fraction = 1/2)
  (h_cream : cream_fraction = 3/5)
  (h_choc_chips : choc_chips_fraction = 1/4) :
  ∃ (max_without : ℕ), max_without ≤ total_pies ∧ 
  max_without = total_pies - ⌈cream_fraction * total_pies⌉ ∧
  max_without = 19 :=
sorry

end max_pies_without_ingredients_l2617_261701


namespace quadratic_inequality_solution_l2617_261704

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x : ℝ, x^2 + (a+1)*x + a*b > 0 ↔ x < -1 ∨ x > 4) → a + b = -3 :=
by sorry

end quadratic_inequality_solution_l2617_261704


namespace negative_sum_l2617_261755

theorem negative_sum (x w : ℝ) 
  (hx : -1 < x ∧ x < 0) 
  (hw : -2 < w ∧ w < -1) : 
  x + w < 0 := by
  sorry

end negative_sum_l2617_261755


namespace crispy_red_plum_pricing_l2617_261760

theorem crispy_red_plum_pricing (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x > 5) 
  (first_batch_cost : ℝ := 12000)
  (second_batch_cost : ℝ := 11000)
  (price_difference : ℝ := 5)
  (quantity_difference : ℝ := 40) :
  first_batch_cost / x = second_batch_cost / (x - price_difference) - quantity_difference := by
sorry

end crispy_red_plum_pricing_l2617_261760


namespace arcsin_cos_arcsin_plus_arccos_sin_arccos_l2617_261765

theorem arcsin_cos_arcsin_plus_arccos_sin_arccos (x : ℝ) : 
  Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = π / 2 := by
  sorry

end arcsin_cos_arcsin_plus_arccos_sin_arccos_l2617_261765


namespace certain_and_uncertain_digits_l2617_261709

def value : ℝ := 945.673
def absolute_error : ℝ := 0.03

def is_certain (digit : ℕ) (place_value : ℝ) : Prop :=
  place_value > absolute_error

def is_uncertain (digit : ℕ) (place_value : ℝ) : Prop :=
  place_value < absolute_error

theorem certain_and_uncertain_digits :
  (is_certain 9 100) ∧
  (is_certain 4 10) ∧
  (is_certain 5 1) ∧
  (is_certain 6 0.1) ∧
  (is_uncertain 7 0.01) ∧
  (is_uncertain 3 0.001) :=
by sorry

end certain_and_uncertain_digits_l2617_261709


namespace inequality_holds_l2617_261757

theorem inequality_holds (f : ℝ → ℝ) (a b x : ℝ) 
  (h_f : ∀ x, f x = 4 * x - 1)
  (h_a : a > 0)
  (h_b : b > 0)
  (h_x : |x - 2*b| < b)
  (h_ab : a ≤ 4*b) : 
  (x + a)^2 + |f x - 3*b| < a^2 := by
sorry

end inequality_holds_l2617_261757


namespace ralphs_tv_time_l2617_261794

/-- The number of hours Ralph watches TV in one week -/
def total_tv_hours (weekday_hours weekday_days weekend_hours weekend_days : ℕ) : ℕ :=
  weekday_hours * weekday_days + weekend_hours * weekend_days

theorem ralphs_tv_time : total_tv_hours 4 5 6 2 = 32 := by
  sorry

end ralphs_tv_time_l2617_261794


namespace disjunction_false_implies_both_false_l2617_261780

theorem disjunction_false_implies_both_false (p q : Prop) :
  (¬(p ∨ q)) → (¬p ∧ ¬q) := by
  sorry

end disjunction_false_implies_both_false_l2617_261780


namespace intersection_of_lines_l2617_261761

theorem intersection_of_lines :
  ∃! (x y : ℚ), 5 * x - 3 * y = 7 ∧ 4 * x + 2 * y = 18 ∧ x = 34 / 11 ∧ y = 31 / 11 := by
  sorry

end intersection_of_lines_l2617_261761


namespace shirt_ironing_time_l2617_261724

/-- The number of days per week Hayden irons his clothes -/
def days_per_week : ℕ := 5

/-- The number of minutes Hayden spends ironing his pants each day -/
def pants_ironing_time : ℕ := 3

/-- The total number of minutes Hayden spends ironing over 4 weeks -/
def total_ironing_time : ℕ := 160

/-- The number of weeks in the period -/
def num_weeks : ℕ := 4

theorem shirt_ironing_time :
  ∃ (shirt_time : ℕ),
    shirt_time * (days_per_week * num_weeks) = 
      total_ironing_time - (pants_ironing_time * days_per_week * num_weeks) ∧
    shirt_time = 5 := by
  sorry

end shirt_ironing_time_l2617_261724


namespace election_winner_percentage_l2617_261747

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  winner_votes = 490 →
  margin = 280 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 7/10 :=
by
  sorry

end election_winner_percentage_l2617_261747


namespace grid_product_theorem_l2617_261734

def grid := Fin 3 → Fin 3 → ℕ

def is_valid_grid (g : grid) : Prop :=
  (∀ i j, g i j ∈ Finset.range 10) ∧
  (∀ i j k, i ≠ j → g i k ≠ g j k) ∧
  (∀ i j k, i ≠ j → g k i ≠ g k j)

def row_product (g : grid) (i : Fin 3) : ℕ :=
  (g i 0) * (g i 1) * (g i 2)

def col_product (g : grid) (j : Fin 3) : ℕ :=
  (g 0 j) * (g 1 j) * (g 2 j)

def all_products_equal (g : grid) (P : ℕ) : Prop :=
  (∀ i : Fin 3, row_product g i = P) ∧
  (∀ j : Fin 3, col_product g j = P)

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem grid_product_theorem :
  ∀ (g : grid) (P : ℕ),
    is_valid_grid g →
    all_products_equal g P →
    P = Nat.sqrt (factorial 9) ∧
    (P = 1998 ∨ P = 2000) :=
by sorry

end grid_product_theorem_l2617_261734
