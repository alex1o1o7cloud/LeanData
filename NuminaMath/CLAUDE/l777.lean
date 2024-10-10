import Mathlib

namespace distribute_five_into_three_l777_77744

/-- The number of ways to distribute n distinct objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to distribute 5 distinct objects into 3 distinct containers is 3^5 -/
theorem distribute_five_into_three : distribute 5 3 = 3^5 := by
  sorry

end distribute_five_into_three_l777_77744


namespace arithmetic_mean_problem_l777_77707

theorem arithmetic_mean_problem (x : ℕ) : 
  let numbers := [3, 117, 915, 138, 1917, 2114, x]
  (numbers.sum % 7 = 7) →
  (numbers.sum / numbers.length : ℚ) = 745 := by
sorry

end arithmetic_mean_problem_l777_77707


namespace joanna_money_problem_l777_77757

/-- Joanna's money problem -/
theorem joanna_money_problem (J : ℚ) 
  (h1 : J + 3 * J + J / 2 = 36) : J = 8 := by
  sorry

end joanna_money_problem_l777_77757


namespace factor_polynomial_l777_77715

theorem factor_polynomial (x : ℝ) : 54 * x^5 - 135 * x^9 = 27 * x^5 * (2 - 5 * x^4) := by sorry

end factor_polynomial_l777_77715


namespace translated_circle_center_l777_77766

/-- Given a point A(1,1) and a point P(m,n) on the circle centered at A, 
    if P is symmetric with P' with respect to the origin after translation,
    then the coordinates of A' are (1-2m, 1-2n) -/
theorem translated_circle_center (m n : ℝ) : 
  let A : ℝ × ℝ := (1, 1)
  let P : ℝ × ℝ := (m, n)
  let O : ℝ × ℝ := (0, 0)
  ∃ (A' : ℝ × ℝ), 
    (∃ (P' : ℝ × ℝ), P'.1 = -P.1 ∧ P'.2 = -P.2) →  -- P and P' are symmetric about origin
    (∃ (r : ℝ), (P.1 - A.1)^2 + (P.2 - A.2)^2 = r^2) →  -- P is on circle centered at A
    A' = (1 - 2*m, 1 - 2*n) :=
sorry

end translated_circle_center_l777_77766


namespace line_properties_l777_77726

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_properties :
  (∃ line_vector : ℝ → ℝ × ℝ × ℝ,
    (line_vector (-2) = (2, 4, 7)) ∧
    (line_vector 1 = (-1, 0, -3))) →
  (∃ line_vector : ℝ → ℝ × ℝ × ℝ,
    (line_vector (-2) = (2, 4, 7)) ∧
    (line_vector 1 = (-1, 0, -3)) ∧
    (line_vector (-1) = (1, 8, 5)) ∧
    (¬ ∃ t : ℝ, line_vector t = (3, 604, -6))) :=
by sorry

end line_properties_l777_77726


namespace vector_parallel_perpendicular_l777_77769

/-- Two vectors are parallel if their corresponding components are proportional -/
def IsParallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

/-- Two vectors are perpendicular if their dot product is zero -/
def IsPerpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a and b, prove the values of m for parallel and perpendicular cases -/
theorem vector_parallel_perpendicular (m : ℝ) :
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (1, 2)
  (IsParallel a b → m = 1/2) ∧
  (IsPerpendicular a b → m = -2) := by
  sorry

end vector_parallel_perpendicular_l777_77769


namespace travel_time_difference_l777_77768

/-- Proves that the difference in travel time between two cars is 2 hours -/
theorem travel_time_difference (distance : ℝ) (speed_r speed_p : ℝ) : 
  distance = 600 ∧ 
  speed_r = 50 ∧ 
  speed_p = speed_r + 10 →
  distance / speed_r - distance / speed_p = 2 := by
sorry


end travel_time_difference_l777_77768


namespace sum_243_62_base5_l777_77728

/-- Converts a natural number to its base 5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Adds two numbers in base 5 representation --/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry  -- Implementation details omitted

/-- Theorem: The sum of 243 and 62 in base 5 is 2170₅ --/
theorem sum_243_62_base5 :
  addBase5 (toBase5 243) (toBase5 62) = [0, 7, 1, 2] :=
sorry

end sum_243_62_base5_l777_77728


namespace remaining_cooking_times_l777_77738

/-- Calculates the remaining cooking time in seconds for a food item -/
def remainingCookingTime (recommendedTime actualTime : ℕ) : ℕ :=
  (recommendedTime - actualTime) * 60

/-- Represents the cooking times for different food items -/
structure CookingTimes where
  frenchFries : ℕ
  chickenNuggets : ℕ
  mozzarellaSticks : ℕ

/-- Theorem stating the remaining cooking times for each food item -/
theorem remaining_cooking_times 
  (recommended : CookingTimes) 
  (actual : CookingTimes) : 
  remainingCookingTime recommended.frenchFries actual.frenchFries = 600 ∧
  remainingCookingTime recommended.chickenNuggets actual.chickenNuggets = 780 ∧
  remainingCookingTime recommended.mozzarellaSticks actual.mozzarellaSticks = 300 :=
by
  sorry

#check remaining_cooking_times (CookingTimes.mk 12 18 8) (CookingTimes.mk 2 5 3)

end remaining_cooking_times_l777_77738


namespace files_deleted_l777_77725

/-- Given the initial number of files and the number of files left after deletion,
    prove that the number of files deleted is 14. -/
theorem files_deleted (initial_files : ℕ) (files_left : ℕ) 
  (h1 : initial_files = 21) 
  (h2 : files_left = 7) : 
  initial_files - files_left = 14 := by
  sorry


end files_deleted_l777_77725


namespace function_difference_implies_m_value_l777_77792

/-- The function f(x) = 4x^2 - 3x + 5 -/
def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5

/-- The function g(x) = x^2 - mx - 8, parameterized by m -/
def g (m : ℝ) (x : ℝ) : ℝ := x^2 - m * x - 8

/-- Theorem stating that if f(5) - g(5) = 15, then m = -11.6 -/
theorem function_difference_implies_m_value :
  ∃ m : ℝ, f 5 - g m 5 = 15 → m = -11.6 := by
  sorry

end function_difference_implies_m_value_l777_77792


namespace weighted_average_is_correct_l777_77787

def english_score : Rat := 76 / 120
def english_weight : Nat := 2

def math_score : Rat := 65 / 150
def math_weight : Nat := 3

def physics_score : Rat := 82 / 100
def physics_weight : Nat := 2

def chemistry_score : Rat := 67 / 80
def chemistry_weight : Nat := 1

def biology_score : Rat := 85 / 100
def biology_weight : Nat := 2

def history_score : Rat := 92 / 150
def history_weight : Nat := 1

def geography_score : Rat := 58 / 75
def geography_weight : Nat := 1

def total_weight : Nat := english_weight + math_weight + physics_weight + chemistry_weight + biology_weight + history_weight + geography_weight

def weighted_average_score : Rat :=
  (english_score * english_weight +
   math_score * math_weight +
   physics_score * physics_weight +
   chemistry_score * chemistry_weight +
   biology_score * biology_weight +
   history_score * history_weight +
   geography_score * geography_weight) / total_weight

theorem weighted_average_is_correct : weighted_average_score = 67755 / 1000 := by
  sorry

end weighted_average_is_correct_l777_77787


namespace child_money_distribution_l777_77710

/-- Prove that for three children with shares in the ratio 2:3:4, 
    where the second child's share is $300, the total amount is $900. -/
theorem child_money_distribution (a b c : ℕ) : 
  a + b + c = 9 ∧ 
  2 * b = 3 * a ∧ 
  4 * b = 3 * c ∧ 
  b = 300 → 
  a + b + c = 900 := by
sorry

end child_money_distribution_l777_77710


namespace small_forward_duration_l777_77743

/-- Represents the duration of footage for each player in seconds. -/
structure PlayerFootage where
  pointGuard : ℕ
  shootingGuard : ℕ
  smallForward : ℕ
  powerForward : ℕ
  center : ℕ

/-- Calculates the total duration of all players' footage in seconds. -/
def totalDuration (pf : PlayerFootage) : ℕ :=
  pf.pointGuard + pf.shootingGuard + pf.smallForward + pf.powerForward + pf.center

/-- The number of players in the team. -/
def numPlayers : ℕ := 5

/-- The average duration per player in seconds. -/
def avgDurationPerPlayer : ℕ := 120 -- 2 minutes = 120 seconds

theorem small_forward_duration (pf : PlayerFootage) 
    (h1 : pf.pointGuard = 130)
    (h2 : pf.shootingGuard = 145)
    (h3 : pf.powerForward = 60)
    (h4 : pf.center = 180)
    (h5 : totalDuration pf = numPlayers * avgDurationPerPlayer) :
    pf.smallForward = 85 := by
  sorry

end small_forward_duration_l777_77743


namespace factor_expression_l777_77700

theorem factor_expression (x : ℝ) : 3*x*(x-5) - 2*(x-5) + 4*x*(x-5) = (x-5)*(7*x-2) := by
  sorry

end factor_expression_l777_77700


namespace isosceles_triangle_perimeter_l777_77784

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 5 → b = 5 → c = 2 →
  (a = b) →  -- isosceles condition
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →  -- triangle inequality
  a + b + c = 12 := by
  sorry

end isosceles_triangle_perimeter_l777_77784


namespace sufficient_not_necessary_l777_77788

theorem sufficient_not_necessary (a : ℝ) : 
  (a > 1 → 1/a < 1) ∧ (∃ b : ℝ, b ≤ 1 ∧ 1/b < 1) := by
  sorry

end sufficient_not_necessary_l777_77788


namespace heat_required_temperature_dependent_specific_heat_l777_77794

/-- The amount of heat required to heat a body with temperature-dependent specific heat capacity. -/
theorem heat_required_temperature_dependent_specific_heat
  (m : ℝ) (c₀ : ℝ) (α : ℝ) (t₁ t₂ : ℝ)
  (hm : m = 2)
  (hc₀ : c₀ = 150)
  (hα : α = 0.05)
  (ht₁ : t₁ = 20)
  (ht₂ : t₂ = 100)
  : ∃ Q : ℝ, Q = 96000 ∧ Q = m * (c₀ * (1 + α * t₂) + c₀ * (1 + α * t₁)) / 2 * (t₂ - t₁) :=
by sorry

end heat_required_temperature_dependent_specific_heat_l777_77794


namespace chef_cherry_pies_l777_77701

/-- Given a chef with an initial number of cherries, some used cherries, and a fixed number of cherries required per pie, 
    this function calculates the maximum number of additional pies that can be made with the remaining cherries. -/
def max_additional_pies (initial_cherries used_cherries cherries_per_pie : ℕ) : ℕ :=
  (initial_cherries - used_cherries) / cherries_per_pie

/-- Theorem stating that for the given values, the maximum number of additional pies is 4. -/
theorem chef_cherry_pies : max_additional_pies 500 350 35 = 4 := by
  sorry

end chef_cherry_pies_l777_77701


namespace max_triangle_area_l777_77713

theorem max_triangle_area (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 1 ≤ b ∧ b ≤ 2) (hc : 2 ≤ c ∧ c ≤ 3)
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  ∃ (area : ℝ), area ≤ 1 ∧ 
  ∀ (area' : ℝ), (∃ (a' b' c' : ℝ), 
    0 ≤ a' ∧ a' ≤ 1 ∧ 
    1 ≤ b' ∧ b' ≤ 2 ∧ 
    2 ≤ c' ∧ c' ≤ 3 ∧ 
    a' + b' > c' ∧ a' + c' > b' ∧ b' + c' > a' ∧
    area' = (a' + b' + c') * (a' + b' - c') * (a' - b' + c') * (-a' + b' + c') / 16) → 
  area' ≤ area :=
sorry

end max_triangle_area_l777_77713


namespace texas_california_plate_difference_l777_77718

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible California license plates -/
def california_plates : ℕ := num_letters^3 * num_digits^3

/-- The number of possible Texas license plates -/
def texas_plates : ℕ := num_letters^3 * num_digits^4

/-- The difference in the number of possible license plates between Texas and California -/
def plate_difference : ℕ := texas_plates - california_plates

theorem texas_california_plate_difference :
  plate_difference = 158184000 := by
  sorry

end texas_california_plate_difference_l777_77718


namespace largest_prime_factor_of_1739_l777_77791

theorem largest_prime_factor_of_1739 :
  (Nat.factors 1739).maximum? = some 47 := by
  sorry

end largest_prime_factor_of_1739_l777_77791


namespace range_of_a_l777_77753

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 2

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc a 3, f x ∈ Set.Icc (-1) 3) ∧
  (∀ y ∈ Set.Icc (-1) 3, ∃ x ∈ Set.Icc a 3, f x = y) →
  a ∈ Set.Icc (-1) 1 :=
by sorry

end range_of_a_l777_77753


namespace mike_initial_marbles_l777_77712

/-- The number of marbles Mike gave to Sam -/
def marbles_given : ℕ := 4

/-- The number of marbles Mike has left -/
def marbles_left : ℕ := 4

/-- The initial number of marbles Mike had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem mike_initial_marbles : initial_marbles = 8 := by
  sorry

end mike_initial_marbles_l777_77712


namespace curve_composition_l777_77745

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop :=
  (x - Real.sqrt (-y^2 + 2*y + 8)) * Real.sqrt (x - y) = 0

-- Define the line segment
def line_segment (x y : ℝ) : Prop :=
  x = y ∧ -2 ≤ y ∧ y ≤ 4

-- Define the minor arc
def minor_arc (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 9 ∧ x ≥ 0

-- Theorem stating that the curve consists of a line segment and a minor arc
theorem curve_composition :
  ∀ x y : ℝ, curve_equation x y ↔ (line_segment x y ∨ minor_arc x y) :=
sorry

end curve_composition_l777_77745


namespace perpendicular_line_through_circle_l777_77763

/-- Given a circle C and a line l in polar coordinates, 
    this theorem proves the equation of a line passing through C 
    and perpendicular to l. -/
theorem perpendicular_line_through_circle 
  (C : ℝ → ℝ) 
  (l : ℝ → ℝ → ℝ) 
  (h_C : ∀ θ, C θ = 2 * Real.cos θ) 
  (h_l : ∀ ρ θ, l ρ θ = ρ * Real.cos θ - ρ * Real.sin θ - 4) :
  ∃ f : ℝ → ℝ → ℝ, 
    (∀ ρ θ, f ρ θ = ρ * (Real.cos θ + Real.sin θ) - 1) ∧
    (∃ θ₀, C θ₀ = f (C θ₀) θ₀) ∧
    (∀ ρ₁ θ₁ ρ₂ θ₂, 
      l ρ₁ θ₁ = 0 → l ρ₂ θ₂ = 0 → f ρ₁ θ₁ = 0 → f ρ₂ θ₂ = 0 →
      (ρ₁ * Real.cos θ₁ - ρ₂ * Real.cos θ₂) * (ρ₁ * Real.sin θ₁ - ρ₂ * Real.sin θ₂) = 
      -(ρ₁ * Real.cos θ₁ - ρ₂ * Real.cos θ₂) * (ρ₁ * Real.sin θ₁ - ρ₂ * Real.sin θ₂)) :=
by
  sorry

end perpendicular_line_through_circle_l777_77763


namespace only_345_is_pythagorean_triple_l777_77762

/-- Checks if three numbers form a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- Proof that (3, 4, 5) is the only Pythagorean triple among the given sets -/
theorem only_345_is_pythagorean_triple :
  (¬ isPythagoreanTriple 1 2 3) ∧
  (isPythagoreanTriple 3 4 5) ∧
  (¬ isPythagoreanTriple 4 5 6) ∧
  (¬ isPythagoreanTriple 7 8 9) :=
sorry

end only_345_is_pythagorean_triple_l777_77762


namespace kristy_ate_two_cookies_l777_77755

def cookies_problem (total_baked : ℕ) (brother_took : ℕ) (friend1_took : ℕ) (friend2_took : ℕ) (friend3_took : ℕ) (cookies_left : ℕ) : Prop :=
  total_baked = 22 ∧
  brother_took = 1 ∧
  friend1_took = 3 ∧
  friend2_took = 5 ∧
  friend3_took = 5 ∧
  cookies_left = 6 ∧
  total_baked - (brother_took + friend1_took + friend2_took + friend3_took + cookies_left) = 2

theorem kristy_ate_two_cookies :
  ∀ (total_baked brother_took friend1_took friend2_took friend3_took cookies_left : ℕ),
  cookies_problem total_baked brother_took friend1_took friend2_took friend3_took cookies_left →
  total_baked - (brother_took + friend1_took + friend2_took + friend3_took + cookies_left) = 2 :=
by
  sorry

end kristy_ate_two_cookies_l777_77755


namespace decagon_diagonals_l777_77775

/-- The number of sides in a decagon -/
def decagon_sides : ℕ := 10

/-- Formula for the number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end decagon_diagonals_l777_77775


namespace gondor_thursday_laptops_l777_77702

/-- Represents the earnings and repair counts for Gondor --/
structure GondorEarnings where
  phone_repair_price : ℕ
  laptop_repair_price : ℕ
  monday_phones : ℕ
  tuesday_phones : ℕ
  wednesday_laptops : ℕ
  total_earnings : ℕ

/-- Calculates the number of laptops repaired on Thursday --/
def thursday_laptops (g : GondorEarnings) : ℕ :=
  let mon_tue_wed_earnings := g.phone_repair_price * (g.monday_phones + g.tuesday_phones) + 
                              g.laptop_repair_price * g.wednesday_laptops
  let thursday_earnings := g.total_earnings - mon_tue_wed_earnings
  thursday_earnings / g.laptop_repair_price

/-- Theorem stating that Gondor repaired 4 laptops on Thursday --/
theorem gondor_thursday_laptops :
  let g : GondorEarnings := {
    phone_repair_price := 10,
    laptop_repair_price := 20,
    monday_phones := 3,
    tuesday_phones := 5,
    wednesday_laptops := 2,
    total_earnings := 200
  }
  thursday_laptops g = 4 := by sorry

end gondor_thursday_laptops_l777_77702


namespace arithmetic_sequence_a5_zero_l777_77749

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_a5_zero
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_d_nonzero : d ≠ 0)
  (h_condition : a 3 + a 9 = a 10 - a 8) :
  a 5 = 0 := by
sorry

end arithmetic_sequence_a5_zero_l777_77749


namespace complex_fraction_equality_l777_77734

theorem complex_fraction_equality : 
  (((4.625 - 13/18 * 9/26) / (9/4) + 2.5 / 1.25 / 6.75) / (1 + 53/68)) /
  ((1/2 - 0.375) / 0.125 + (5/6 - 7/12) / (0.358 - 1.4796 / 13.7)) = 17/27 := by
  sorry

end complex_fraction_equality_l777_77734


namespace cistern_emptied_in_8_minutes_l777_77748

/-- Given a pipe that can empty 2/3 of a cistern in 10 minutes,
    this function calculates the part of the cistern that will be empty in t minutes. -/
def cisternEmptied (t : ℚ) : ℚ :=
  (2/3) * (t / 10)

/-- Theorem stating that the part of the cistern emptied in 8 minutes is 8/15. -/
theorem cistern_emptied_in_8_minutes :
  cisternEmptied 8 = 8/15 := by
  sorry

end cistern_emptied_in_8_minutes_l777_77748


namespace inscribed_rectangle_area_bound_l777_77722

/-- A triangle in a 2D plane --/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- A rectangle in a 2D plane --/
structure Rectangle where
  vertices : Fin 4 → ℝ × ℝ

/-- The area of a triangle --/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The area of a rectangle --/
def rectangleArea (r : Rectangle) : ℝ := sorry

/-- Predicate to check if a rectangle is inscribed in a triangle --/
def isInscribed (r : Rectangle) (t : Triangle) : Prop := sorry

/-- Theorem: The area of a rectangle inscribed in a triangle does not exceed half the area of the triangle --/
theorem inscribed_rectangle_area_bound (t : Triangle) (r : Rectangle) :
  isInscribed r t → rectangleArea r ≤ (1/2) * triangleArea t := by
  sorry

end inscribed_rectangle_area_bound_l777_77722


namespace ceiling_floor_sum_l777_77773

theorem ceiling_floor_sum : ⌈(7 : ℚ) / 3⌉ + ⌊-(7 : ℚ) / 3⌋ = 0 := by
  sorry

end ceiling_floor_sum_l777_77773


namespace view_characteristics_l777_77776

/-- Represents the view described in the problem -/
structure View where
  endless_progress : Bool
  unlimited_capacity : Bool
  unlimited_resources : Bool

/-- Represents the characteristics of the view -/
structure ViewCharacteristics where
  emphasizes_subjective_initiative : Bool
  ignores_objective_conditions : Bool

/-- Theorem stating that the given view unilaterally emphasizes subjective initiative
    while ignoring objective conditions and laws -/
theorem view_characteristics (v : View) 
  (h1 : v.endless_progress = true)
  (h2 : v.unlimited_capacity = true)
  (h3 : v.unlimited_resources = true) :
  ∃ (c : ViewCharacteristics), 
    c.emphasizes_subjective_initiative ∧ c.ignores_objective_conditions := by
  sorry


end view_characteristics_l777_77776


namespace part_1_part_2_l777_77727

-- Define the sets M, N, and H
def M : Set ℝ := {x | 1 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}
def H (a : ℝ) : Set ℝ := {x | |x - a| ≤ 2}

-- Define the custom set operation ∆
def triangleOp (A B : Set ℝ) : Set ℝ := A ∩ (Set.univ \ B)

-- Theorem for part (1)
theorem part_1 :
  triangleOp M N = {x | 1 < x ∧ x < 2} ∧
  triangleOp N M = {x | 3 ≤ x ∧ x ≤ 4} := by sorry

-- Theorem for part (2)
theorem part_2 (a : ℝ) :
  triangleOp (triangleOp N M) (H a) =
    if a ≥ 4 ∨ a ≤ -1 then
      {x | 1 < x ∧ x < 2}
    else if 3 < a ∧ a < 4 then
      {x | 1 < x ∧ x < a - 2}
    else if -1 < a ∧ a < 0 then
      {x | a + 2 < x ∧ x < 2}
    else
      ∅ := by sorry

end part_1_part_2_l777_77727


namespace current_speed_l777_77795

/-- The speed of the current given a woman's swimming times -/
theorem current_speed (v c : ℝ) 
  (h1 : v + c = 64 / 8)  -- Downstream speed
  (h2 : v - c = 24 / 8)  -- Upstream speed
  : c = 2.5 := by
  sorry

end current_speed_l777_77795


namespace farm_field_area_l777_77772

/-- The area of a farm field given specific ploughing conditions --/
theorem farm_field_area (planned_rate : ℝ) (actual_rate : ℝ) (extra_days : ℕ) (area_left : ℝ) : 
  planned_rate = 120 →
  actual_rate = 85 →
  extra_days = 2 →
  area_left = 40 →
  ∃ (planned_days : ℝ), 
    planned_rate * planned_days = actual_rate * (planned_days + extra_days) + area_left ∧
    planned_rate * planned_days = 720 := by
  sorry

end farm_field_area_l777_77772


namespace collinear_points_k_value_l777_77789

/-- Three points are collinear if and only if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- The value of k for which the points (2, -3), (k, k + 2), and (-3k + 4, 1) are collinear. -/
theorem collinear_points_k_value :
  ∃ k : ℝ, collinear 2 (-3) k (k + 2) (-3 * k + 4) 1 ∧
    (k = (17 + Real.sqrt 505) / (-6) ∨ k = (17 - Real.sqrt 505) / (-6)) := by
  sorry

end collinear_points_k_value_l777_77789


namespace number_of_men_l777_77797

theorem number_of_men (M : ℕ) (W : ℝ) : 
  (W / (M * 20 : ℝ) = W / ((M - 4) * 25 : ℝ)) → M = 20 := by
  sorry

end number_of_men_l777_77797


namespace curve_transformation_l777_77720

theorem curve_transformation (x : ℝ) : 
  Real.sin (2 * x) = Real.sin (2 * (x + π / 8) + π / 4) := by sorry

end curve_transformation_l777_77720


namespace arithmetic_sequence_sum_l777_77709

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 4 + a 7 = 39) →
  (a 2 + a 5 + a 8 = 33) →
  (a 3 + a 6 + a 9 = 27) :=
by
  sorry

end arithmetic_sequence_sum_l777_77709


namespace function_range_l777_77731

theorem function_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f x = |x - 3| + |x - a|) →
  (∀ x : ℝ, f x ≥ 4) →
  (a ≤ -1 ∨ a ≥ 7) :=
by sorry

end function_range_l777_77731


namespace score_difference_is_3_4_l777_77779

-- Define the score distribution
def score_distribution : List (ℝ × ℝ) := [
  (60, 0.15),
  (75, 0.20),
  (88, 0.25),
  (92, 0.10),
  (98, 0.30)
]

-- Define the mean score
def mean_score : ℝ := (score_distribution.map (λ (score, freq) => score * freq)).sum

-- Define the median score
def median_score : ℝ := 88

-- Theorem statement
theorem score_difference_is_3_4 :
  |median_score - mean_score| = 3.4 := by sorry

end score_difference_is_3_4_l777_77779


namespace wind_velocity_problem_l777_77733

/-- Represents the relationship between pressure, area, and velocity -/
def pressure_relation (k : ℝ) (A : ℝ) (V : ℝ) : ℝ := k * A * V^3

theorem wind_velocity_problem (k : ℝ) :
  (pressure_relation k 1 8 = 1) →
  (pressure_relation k 9 12 = 27) :=
by sorry

end wind_velocity_problem_l777_77733


namespace sandy_clothes_spending_l777_77785

/-- The amount Sandy spent on clothes -/
def total_spent (shorts_cost shirt_cost jacket_cost : ℚ) : ℚ :=
  shorts_cost + shirt_cost + jacket_cost

/-- Theorem: Sandy's total spending on clothes -/
theorem sandy_clothes_spending :
  total_spent 13.99 12.14 7.43 = 33.56 := by
  sorry

end sandy_clothes_spending_l777_77785


namespace subset_condition_iff_m_geq_three_l777_77765

theorem subset_condition_iff_m_geq_three (m : ℝ) : 
  (∀ x : ℝ, x^2 - x ≤ 0 → x^2 - 4*x + m ≥ 0) ↔ m ≥ 3 := by sorry

end subset_condition_iff_m_geq_three_l777_77765


namespace subtract_decimals_l777_77761

theorem subtract_decimals : 3.79 - 2.15 = 1.64 := by
  sorry

end subtract_decimals_l777_77761


namespace ava_lily_trees_l777_77704

/-- The number of apple trees planted by Ava and Lily -/
def total_trees (ava_trees lily_trees : ℕ) : ℕ :=
  ava_trees + lily_trees

/-- Theorem stating the total number of apple trees planted by Ava and Lily -/
theorem ava_lily_trees :
  ∀ (ava_trees lily_trees : ℕ),
    ava_trees = 9 →
    ava_trees = lily_trees + 3 →
    total_trees ava_trees lily_trees = 15 :=
by
  sorry


end ava_lily_trees_l777_77704


namespace nonnegative_solutions_count_l777_77798

theorem nonnegative_solutions_count : ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -6*x := by sorry

end nonnegative_solutions_count_l777_77798


namespace unique_three_digit_sum_27_l777_77764

/-- A three-digit number is a natural number between 100 and 999 inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

/-- The main theorem: there is exactly one three-digit number whose digits sum to 27 -/
theorem unique_three_digit_sum_27 : ∃! n : ℕ, ThreeDigitNumber n ∧ sumOfDigits n = 27 := by
  sorry


end unique_three_digit_sum_27_l777_77764


namespace min_value_theorem_min_value_achieved_l777_77732

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 25 := by sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 1 ∧ (2 / a₀ + 3 / b₀) = 25 := by sorry

end min_value_theorem_min_value_achieved_l777_77732


namespace square_garden_perimeter_l777_77717

theorem square_garden_perimeter (area : Real) (perimeter : Real) :
  area = 90.25 →
  area = 2 * perimeter + 14.25 →
  perimeter = 38 := by
  sorry

end square_garden_perimeter_l777_77717


namespace range_of_a_for_positive_x_l777_77767

theorem range_of_a_for_positive_x (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 2*x - a = 3*x - 4) ↔ a < 4 :=
sorry

end range_of_a_for_positive_x_l777_77767


namespace bandit_gem_distribution_theorem_l777_77706

/-- Represents the distribution of precious stones for a bandit -/
structure GemDistribution where
  rubies : ℕ
  sapphires : ℕ
  emeralds : ℕ
  sum_is_100 : rubies + sapphires + emeralds = 100

/-- The proposition to be proven -/
theorem bandit_gem_distribution_theorem (bandits : Finset GemDistribution) 
    (h : bandits.card = 102) :
  (∃ b1 b2 : GemDistribution, b1 ∈ bandits ∧ b2 ∈ bandits ∧ b1 ≠ b2 ∧
    b1.rubies = b2.rubies ∧ b1.sapphires = b2.sapphires ∧ b1.emeralds = b2.emeralds) ∨
  (∃ b1 b2 : GemDistribution, b1 ∈ bandits ∧ b2 ∈ bandits ∧ b1 ≠ b2 ∧
    b1.rubies ≠ b2.rubies ∧ b1.sapphires ≠ b2.sapphires ∧ b1.emeralds ≠ b2.emeralds) :=
by
  sorry

end bandit_gem_distribution_theorem_l777_77706


namespace zoo_open_hours_proof_l777_77754

/-- The number of hours the zoo is open in one day -/
def zoo_open_hours : ℕ := 8

/-- The number of new visitors entering the zoo every hour -/
def visitors_per_hour : ℕ := 50

/-- The percentage of total visitors who go to the gorilla exhibit -/
def gorilla_exhibit_percentage : ℚ := 80 / 100

/-- The number of visitors who go to the gorilla exhibit in one day -/
def gorilla_exhibit_visitors : ℕ := 320

/-- Theorem stating that the zoo is open for 8 hours given the conditions -/
theorem zoo_open_hours_proof :
  zoo_open_hours * visitors_per_hour * gorilla_exhibit_percentage = gorilla_exhibit_visitors :=
by sorry

end zoo_open_hours_proof_l777_77754


namespace symmetric_point_coordinates_l777_77771

/-- Given a line l: 2x + y - 5 = 0 and a point M(-1, 2), 
    this function returns the coordinates of the symmetric point Q with respect to l. -/
def symmetricPoint (l : ℝ → ℝ → Prop) (M : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := M
  -- Define the symmetric point Q
  let Q := (3, 4)
  Q

/-- The theorem states that the symmetric point of M(-1, 2) with respect to
    the line 2x + y - 5 = 0 is (3, 4). -/
theorem symmetric_point_coordinates :
  let l : ℝ → ℝ → Prop := fun x y ↦ 2 * x + y - 5 = 0
  let M : ℝ × ℝ := (-1, 2)
  symmetricPoint l M = (3, 4) := by
  sorry


end symmetric_point_coordinates_l777_77771


namespace equation_solution_l777_77782

theorem equation_solution : 
  ∃ x : ℝ, (x / (x - 1) - 1 = 1) ∧ (x = 2) := by
  sorry

end equation_solution_l777_77782


namespace arithmetic_sum_10_to_100_l777_77778

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: The sum of the arithmetic series from 10 to 100 with common difference 1 is 5005 -/
theorem arithmetic_sum_10_to_100 :
  arithmetic_sum 10 100 1 = 5005 := by
  sorry

#eval arithmetic_sum 10 100 1

end arithmetic_sum_10_to_100_l777_77778


namespace quadrilateral_area_is_13_l777_77780

/-- The area of a quadrilateral with vertices (0, 0), (4, 0), (4, 3), and (2, 5) -/
def quadrilateral_area : ℝ :=
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (4, 0)
  let v3 : ℝ × ℝ := (4, 3)
  let v4 : ℝ × ℝ := (2, 5)
  -- Define the area calculation here
  0 -- placeholder, replace with actual calculation

/-- Theorem: The area of the quadrilateral is 13 -/
theorem quadrilateral_area_is_13 : quadrilateral_area = 13 := by
  sorry

end quadrilateral_area_is_13_l777_77780


namespace equation_system_solution_l777_77742

/-- Given a system of equations with constants m and n, prove that the solution for a and b is (4, 1) -/
theorem equation_system_solution (m n : ℝ) : 
  (∃ x y : ℝ, -2*m*x + 5*y = 15 ∧ x + 7*n*y = 14 ∧ x = 5 ∧ y = 2) →
  (∃ a b : ℝ, -2*m*(a+b) + 5*(a-2*b) = 15 ∧ (a+b) + 7*n*(a-2*b) = 14 ∧ a = 4 ∧ b = 1) :=
by sorry

end equation_system_solution_l777_77742


namespace sqrt_product_simplification_l777_77747

theorem sqrt_product_simplification (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) = 120 * y * Real.sqrt (3 * y) := by
  sorry

end sqrt_product_simplification_l777_77747


namespace gcd_of_polynomial_and_linear_l777_77736

theorem gcd_of_polynomial_and_linear (a : ℤ) (h : ∃ k : ℤ, a = 360 * k) :
  Int.gcd (a^2 + 6*a + 8) (a + 4) = 4 := by sorry

end gcd_of_polynomial_and_linear_l777_77736


namespace common_roots_cubic_polynomials_l777_77703

theorem common_roots_cubic_polynomials :
  ∀ (a b : ℝ),
  (∃ (r s : ℝ), r ≠ s ∧
    (r^3 + a*r^2 + 15*r + 10 = 0) ∧
    (r^3 + b*r^2 + 18*r + 12 = 0) ∧
    (s^3 + a*s^2 + 15*s + 10 = 0) ∧
    (s^3 + b*s^2 + 18*s + 12 = 0)) →
  a = 6 ∧ b = 7 :=
by sorry

end common_roots_cubic_polynomials_l777_77703


namespace equation_solution_l777_77752

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 3/5 ∧ 
  (∀ x : ℝ, (x - 3)^2 + 4*x*(x - 3) = 0 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end equation_solution_l777_77752


namespace elevator_weight_problem_l777_77783

/-- Given 6 people in an elevator with an average weight of 152 lbs, 
    prove that if a 7th person enters and the new average becomes 151 lbs, 
    then the weight of the 7th person is 145 lbs. -/
theorem elevator_weight_problem (people : ℕ) (avg_weight_before : ℝ) (avg_weight_after : ℝ) :
  people = 6 →
  avg_weight_before = 152 →
  avg_weight_after = 151 →
  (people * avg_weight_before + (avg_weight_after * (people + 1) - people * avg_weight_before)) = 145 :=
by sorry

end elevator_weight_problem_l777_77783


namespace scientific_notation_of_1373100000000_l777_77781

theorem scientific_notation_of_1373100000000 :
  (1373100000000 : ℝ) = 1.3731 * (10 ^ 12) := by sorry

end scientific_notation_of_1373100000000_l777_77781


namespace ordering_abc_l777_77724

theorem ordering_abc :
  let a : ℝ := 31/32
  let b : ℝ := Real.cos (1/4)
  let c : ℝ := 4 * Real.sin (1/4)
  c > b ∧ b > a := by sorry

end ordering_abc_l777_77724


namespace sqrt_four_squared_l777_77739

theorem sqrt_four_squared : (Real.sqrt 4)^2 = 4 := by
  sorry

end sqrt_four_squared_l777_77739


namespace horners_first_step_l777_77721

-- Define the polynomial coefficients
def a₅ : ℝ := 0.5
def a₄ : ℝ := 4
def a₃ : ℝ := 0
def a₂ : ℝ := -3
def a₁ : ℝ := 1
def a₀ : ℝ := -1

-- Define the polynomial
def f (x : ℝ) : ℝ := a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

-- Define the point at which to evaluate the polynomial
def x : ℝ := 3

-- State the theorem
theorem horners_first_step :
  a₅ * x + a₄ = 5.5 :=
sorry

end horners_first_step_l777_77721


namespace unfinished_courses_l777_77774

/-- Given the conditions of a construction project, calculate the number of unfinished courses in the last wall. -/
theorem unfinished_courses
  (courses_per_wall : ℕ)
  (bricks_per_course : ℕ)
  (total_walls : ℕ)
  (bricks_used : ℕ)
  (h1 : courses_per_wall = 6)
  (h2 : bricks_per_course = 10)
  (h3 : total_walls = 4)
  (h4 : bricks_used = 220) :
  (courses_per_wall * bricks_per_course * total_walls - bricks_used) / bricks_per_course = 2 :=
by sorry

end unfinished_courses_l777_77774


namespace cone_lateral_area_l777_77799

/-- The lateral area of a cone with specific properties -/
theorem cone_lateral_area (base_diameter : ℝ) (slant_height : ℝ) :
  base_diameter = 6 →
  slant_height = 6 →
  (1 / 2) * (2 * Real.pi) * (base_diameter / 2) * slant_height = 18 * Real.pi := by
  sorry

end cone_lateral_area_l777_77799


namespace triangle_cosine_theorem_l777_77737

theorem triangle_cosine_theorem (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_cos_A : Real.cos A = 4/5) (h_cos_B : Real.cos B = 7/25) : 
  Real.cos C = 44/125 := by
  sorry

end triangle_cosine_theorem_l777_77737


namespace aartis_work_completion_time_l777_77770

/-- If Aarti can complete three times a piece of work in 15 days, 
    then she can complete one piece of work in 5 days. -/
theorem aartis_work_completion_time :
  ∀ (work_completion_time : ℝ),
  (3 * work_completion_time = 15) →
  work_completion_time = 5 :=
by sorry

end aartis_work_completion_time_l777_77770


namespace triangle_hypotenuse_l777_77760

theorem triangle_hypotenuse (x y : ℝ) (h : ℝ) : 
  (1/3 : ℝ) * π * y^2 * x = 1200 * π →
  (1/3 : ℝ) * π * x^2 * (2*x) = 3840 * π →
  x^2 + y^2 = h^2 →
  h = 2 * Real.sqrt 131 := by
sorry

end triangle_hypotenuse_l777_77760


namespace park_perimeter_l777_77714

/-- Given a square park with a road inside, proves that the perimeter is 600 meters -/
theorem park_perimeter (s : ℝ) : 
  s > 0 →  -- The side length is positive
  s^2 - (s - 6)^2 = 1764 →  -- The area of the road is 1764 sq meters
  4 * s = 600 :=  -- The perimeter is 600 meters
by
  sorry

end park_perimeter_l777_77714


namespace japanese_students_fraction_l777_77719

theorem japanese_students_fraction (j : ℕ) (s : ℕ) : 
  s = 2 * j →
  (3 * s / 8 + j / 4) / (j + s) = 1 / 3 := by
sorry

end japanese_students_fraction_l777_77719


namespace possible_values_of_a_l777_77730

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x + a / x
  else if x < 0 then -(Real.log (-x) + a / (-x))
  else 0

-- Define the theorem
theorem possible_values_of_a (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) →  -- f is odd
  (∃ x₁ x₂ x₃ x₄ : ℝ,
    x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧  -- x₁ < x₂ < x₃ < x₄
    x₁ + x₄ = 0 ∧  -- x₁ + x₄ = 0
    ∃ r : ℝ,  -- Existence of common ratio r for geometric sequence
      f a x₂ = r * f a x₁ ∧
      f a x₃ = r * f a x₂ ∧
      f a x₄ = r * f a x₃ ∧
    ∃ d : ℝ,  -- Existence of common difference d for arithmetic sequence
      x₂ = x₁ + d ∧
      x₃ = x₂ + d ∧
      x₄ = x₃ + d) →
  a ≤ Real.sqrt 3 / (2 * Real.exp 1) :=
by sorry

end possible_values_of_a_l777_77730


namespace new_triangle_area_ratio_l777_77786

/-- Represents a triangle -/
structure Triangle where
  area : ℝ

/-- Represents a point on a side of a triangle -/
structure PointOnSide where
  distance_ratio : ℝ

/-- Creates a new triangle from points on the sides of an original triangle -/
def new_triangle_from_points (original : Triangle) (p1 p2 p3 : PointOnSide) : Triangle :=
  sorry

theorem new_triangle_area_ratio (T : Triangle) :
  let p1 : PointOnSide := { distance_ratio := 1/3 }
  let p2 : PointOnSide := { distance_ratio := 1/3 }
  let p3 : PointOnSide := { distance_ratio := 1/3 }
  let new_T := new_triangle_from_points T p1 p2 p3
  new_T.area = (1/9) * T.area := by
  sorry

end new_triangle_area_ratio_l777_77786


namespace expression_simplification_l777_77705

theorem expression_simplification (a b : ℝ) 
  (h1 : a ≠ b/2) (h2 : a ≠ -b/2) (h3 : a ≠ -b) (h4 : a ≠ 0) (h5 : b ≠ 0) :
  (((a - b) / (2*a - b) - (a^2 + b^2 + a) / (2*a^2 + a*b - b^2)) / 
   ((4*b^4 + 4*a*b^2 + a^2) / (2*b^2 + a))) * (b^2 + b + a*b + a) = 
  (b + 1) / (b - 2*a) := by
sorry


end expression_simplification_l777_77705


namespace karen_cindy_crayon_difference_l777_77790

theorem karen_cindy_crayon_difference :
  let karen_crayons : ℕ := 639
  let cindy_crayons : ℕ := 504
  karen_crayons - cindy_crayons = 135 :=
by sorry

end karen_cindy_crayon_difference_l777_77790


namespace min_value_4x_minus_y_l777_77716

theorem min_value_4x_minus_y (x y : ℝ) 
  (h1 : x - y ≥ 0) 
  (h2 : x + y - 4 ≥ 0) 
  (h3 : x ≤ 4) : 
  ∃ (m : ℝ), m = 6 ∧ ∀ (x' y' : ℝ), 
    x' - y' ≥ 0 → x' + y' - 4 ≥ 0 → x' ≤ 4 → 4 * x' - y' ≥ m :=
by sorry

end min_value_4x_minus_y_l777_77716


namespace square_root_possible_value_l777_77741

theorem square_root_possible_value (a : ℝ) : 
  (a = -1 ∨ a = -6 ∨ a = 3 ∨ a = -7) → 
  (∃ x : ℝ, x^2 = a) ↔ a = 3 :=
by sorry

end square_root_possible_value_l777_77741


namespace custom_mult_four_three_l777_77711

-- Define the custom multiplication operation
def custom_mult (a b : ℤ) : ℤ := a^2 - a*b + b^2

-- Theorem statement
theorem custom_mult_four_three : custom_mult 4 3 = 13 := by
  sorry

end custom_mult_four_three_l777_77711


namespace power_zero_of_sum_one_l777_77758

theorem power_zero_of_sum_one (a : ℝ) (h : a ≠ -1) : (a + 1)^0 = 1 := by
  sorry

end power_zero_of_sum_one_l777_77758


namespace sqrt_2n_equals_64_l777_77796

theorem sqrt_2n_equals_64 (n : ℝ) : Real.sqrt (2 * n) = 64 → n = 2048 := by
  sorry

end sqrt_2n_equals_64_l777_77796


namespace grocery_shop_sales_l777_77729

theorem grocery_shop_sales (sales1 sales2 sales3 sales4 sales6 average_sale : ℕ)
  (h1 : sales1 = 6735)
  (h2 : sales2 = 6927)
  (h3 : sales3 = 6855)
  (h4 : sales4 = 7230)
  (h5 : sales6 = 4691)
  (h6 : average_sale = 6500) :
  ∃ sales5 : ℕ, sales5 = 6562 ∧
  (sales1 + sales2 + sales3 + sales4 + sales5 + sales6) / 6 = average_sale := by
  sorry

end grocery_shop_sales_l777_77729


namespace no_integer_cube_equals_3n2_plus_3n_plus_7_l777_77777

theorem no_integer_cube_equals_3n2_plus_3n_plus_7 :
  ¬ ∃ (x n : ℤ), x^3 = 3*n^2 + 3*n + 7 := by
  sorry

end no_integer_cube_equals_3n2_plus_3n_plus_7_l777_77777


namespace probability_three_black_balls_l777_77746

theorem probability_three_black_balls (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + black_balls + white_balls →
  red_balls = 10 →
  black_balls = 8 →
  white_balls = 3 →
  (Nat.choose black_balls 3 : ℚ) / (Nat.choose total_balls 3 : ℚ) = 4 / 95 := by
  sorry

end probability_three_black_balls_l777_77746


namespace triangle_side_length_l777_77751

/-- Checks if three lengths can form a valid triangle --/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: In a triangle with sides 5, 8, and a, the only valid value for a is 9 --/
theorem triangle_side_length : ∃! a : ℝ, is_valid_triangle 5 8 a ∧ a > 0 := by
  sorry

end triangle_side_length_l777_77751


namespace savings_ratio_l777_77708

theorem savings_ratio (initial_savings : ℝ) (may_savings : ℝ) :
  initial_savings = 10 →
  may_savings = 160 →
  ∃ (r : ℝ), r > 0 ∧ may_savings = initial_savings * r^4 →
  r = 2 := by
sorry

end savings_ratio_l777_77708


namespace cosine_sum_less_than_sum_of_cosines_l777_77793

theorem cosine_sum_less_than_sum_of_cosines (α β : Real) 
  (h_α : 0 < α ∧ α < π / 2) (h_β : 0 < β ∧ β < π / 2) : 
  Real.cos (α + β) < Real.cos α + Real.cos β := by
  sorry

end cosine_sum_less_than_sum_of_cosines_l777_77793


namespace cats_problem_l777_77759

/-- Given an initial number of cats and the number of female and male kittens,
    calculate the total number of cats. -/
def total_cats (initial : ℕ) (female_kittens : ℕ) (male_kittens : ℕ) : ℕ :=
  initial + female_kittens + male_kittens

/-- Theorem stating that given 2 initial cats, 3 female kittens, and 2 male kittens,
    the total number of cats is 7. -/
theorem cats_problem : total_cats 2 3 2 = 7 := by
  sorry

end cats_problem_l777_77759


namespace arithmetic_calculation_l777_77740

theorem arithmetic_calculation : 8 - 7 + 6 * 5 + 4 - 3 * 2 + 1 - 0 = 30 := by
  sorry

end arithmetic_calculation_l777_77740


namespace solve_linear_equation_l777_77735

theorem solve_linear_equation :
  ∀ x : ℚ, 3 * x + 4 = -6 * x - 11 → x = -5/3 := by
  sorry

end solve_linear_equation_l777_77735


namespace sum_of_decimals_l777_77756

def repeating_decimal_6 : ℚ := 2/3
def repeating_decimal_2 : ℚ := 2/9

theorem sum_of_decimals : 
  repeating_decimal_6 - repeating_decimal_2 + (1/4 : ℚ) = 25/36 := by
  sorry

end sum_of_decimals_l777_77756


namespace smallest_solution_quadratic_l777_77723

theorem smallest_solution_quadratic (x : ℝ) : 
  (2 * x^2 + 30 * x - 84 = x * (x + 15)) → x ≥ -28 :=
by sorry

end smallest_solution_quadratic_l777_77723


namespace fencing_rate_proof_l777_77750

/-- Given a rectangular plot with the following properties:
  - The length is 10 meters more than the width
  - The perimeter is 300 meters
  - The total fencing cost is 1950 Rs
  Prove that the rate per meter for fencing is 6.5 Rs -/
theorem fencing_rate_proof (width : ℝ) (length : ℝ) (perimeter : ℝ) (total_cost : ℝ) :
  length = width + 10 →
  perimeter = 300 →
  perimeter = 2 * (length + width) →
  total_cost = 1950 →
  total_cost / perimeter = 6.5 := by
sorry

end fencing_rate_proof_l777_77750
