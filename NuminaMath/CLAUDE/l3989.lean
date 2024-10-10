import Mathlib

namespace round_trip_ticket_percentage_l3989_398967

/-- Given that 25% of all passengers held round-trip tickets and took their cars aboard,
    and 60% of passengers with round-trip tickets did not take their cars aboard,
    prove that 62.5% of all passengers held round-trip tickets. -/
theorem round_trip_ticket_percentage
  (total_passengers : ℝ)
  (h1 : total_passengers > 0)
  (h2 : (25 : ℝ) / 100 * total_passengers = (40 : ℝ) / 100 * ((100 : ℝ) / 100 * total_passengers)) :
  (62.5 : ℝ) / 100 * total_passengers = (100 : ℝ) / 100 * total_passengers :=
sorry

end round_trip_ticket_percentage_l3989_398967


namespace inequality_proof_l3989_398905

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end inequality_proof_l3989_398905


namespace water_purification_minimum_processes_l3989_398903

theorem water_purification_minimum_processes : ∃ n : ℕ,
  (∀ m : ℕ, m < n → (0.8 ^ m : ℝ) ≥ 0.05) ∧
  (0.8 ^ n : ℝ) < 0.05 := by
  sorry

end water_purification_minimum_processes_l3989_398903


namespace tan_product_from_cos_sum_diff_l3989_398937

theorem tan_product_from_cos_sum_diff (α β : ℝ) 
  (h1 : Real.cos (α + β) = 3/5) 
  (h2 : Real.cos (α - β) = 4/5) : 
  Real.tan α * Real.tan β = 1/7 := by
  sorry

end tan_product_from_cos_sum_diff_l3989_398937


namespace propositions_truth_l3989_398913

-- Define the propositions
def proposition1 (a b : ℝ) : Prop := a > b → a^2 > b^2
def proposition2 (x y : ℝ) : Prop := x + y = 0 → (x = -y ∧ y = -x)
def proposition3 (x : ℝ) : Prop := x^2 < 4 → -2 < x ∧ x < 2

-- State the theorem
theorem propositions_truth : 
  (∀ x y : ℝ, x = -y ∧ y = -x → x + y = 0) ∧ 
  (∀ x : ℝ, (x ≥ 2 ∨ x ≤ -2) → x^2 ≥ 4) :=
sorry

end propositions_truth_l3989_398913


namespace exists_F_for_P_l3989_398968

/-- A ternary polynomial with real coefficients -/
def TernaryPolynomial := ℝ → ℝ → ℝ → ℝ

/-- The conditions that P must satisfy -/
def SatisfiesConditions (P : TernaryPolynomial) : Prop :=
  ∀ x y z : ℝ, 
    P x y z = P x y (x*y - z) ∧
    P x y z = P x (z*x - y) z ∧
    P x y z = P (y*z - x) y z

/-- The theorem statement -/
theorem exists_F_for_P (P : TernaryPolynomial) (h : SatisfiesConditions P) :
  ∃ F : ℝ → ℝ, ∀ x y z : ℝ, P x y z = F (x^2 + y^2 + z^2 - x*y*z) :=
sorry

end exists_F_for_P_l3989_398968


namespace decagon_equilateral_triangles_l3989_398911

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- The number of distinct equilateral triangles with at least two vertices 
    from a given set of points -/
def countDistinctEquilateralTriangles (points : Set (ℝ × ℝ)) : ℕ := sorry

/-- The theorem stating the number of distinct equilateral triangles 
    in a regular decagon -/
theorem decagon_equilateral_triangles (d : RegularPolygon 10) :
  countDistinctEquilateralTriangles (Set.range d.vertices) = 90 := by sorry

end decagon_equilateral_triangles_l3989_398911


namespace polynomial_simplification_l3989_398999

theorem polynomial_simplification (r : ℝ) :
  (2 * r^3 + 4 * r^2 + 5 * r - 3) - (r^3 + 6 * r^2 + 8 * r - 7) = r^3 - 2 * r^2 - 3 * r + 4 := by
  sorry

end polynomial_simplification_l3989_398999


namespace scissors_cost_l3989_398956

theorem scissors_cost (initial_amount : ℕ) (num_scissors : ℕ) (num_erasers : ℕ) 
  (eraser_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 100 → 
  num_scissors = 8 → 
  num_erasers = 10 → 
  eraser_cost = 4 → 
  remaining_amount = 20 → 
  ∃ (scissor_cost : ℕ), 
    scissor_cost = 5 ∧ 
    initial_amount = num_scissors * scissor_cost + num_erasers * eraser_cost + remaining_amount :=
by sorry

end scissors_cost_l3989_398956


namespace sum_coefficients_when_binomial_sum_is_8_l3989_398923

/-- Given a natural number n, this function represents the sum of the binomial coefficients
    of the expansion of (x^2 - 2/x)^n when x = 1 -/
def sumBinomialCoefficients (n : ℕ) : ℤ := (-1 : ℤ) ^ n

/-- Given a natural number n, this function represents the sum of the coefficients
    of the expansion of (x^2 - 2/x)^n when x = 1 -/
def sumCoefficients (n : ℕ) : ℤ := ((-1 : ℤ) - 2) ^ n

theorem sum_coefficients_when_binomial_sum_is_8 :
  ∃ n : ℕ, sumBinomialCoefficients n = 8 ∧ sumCoefficients n = -1 := by
  sorry

end sum_coefficients_when_binomial_sum_is_8_l3989_398923


namespace five_integers_exist_l3989_398935

theorem five_integers_exist : ∃ (a b c d e : ℤ),
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  a * b * c = 8 ∧
  c * d * e = 27 :=
by
  sorry

end five_integers_exist_l3989_398935


namespace inscribed_square_area_l3989_398936

/-- A square with an inscribed circle -/
structure InscribedSquare :=
  (side : ℝ)
  (radius : ℝ)
  (h_radius : radius = side / 2)

/-- A point on the inscribed circle -/
structure CirclePoint (s : InscribedSquare) :=
  (x : ℝ)
  (y : ℝ)
  (h_on_circle : x^2 + y^2 = s.radius^2)

/-- Theorem: If a point on the inscribed circle is 1 unit from one side
    and 2 units from another side, then the area of the square is 100 -/
theorem inscribed_square_area
  (s : InscribedSquare)
  (p : CirclePoint s)
  (h_dist1 : p.x = 1 ∨ p.y = 1)
  (h_dist2 : p.x = 2 ∨ p.y = 2) :
  s.side^2 = 100 :=
sorry

end inscribed_square_area_l3989_398936


namespace fourth_degree_reduction_l3989_398996

theorem fourth_degree_reduction (a b c d : ℝ) :
  ∃ (A B C k : ℝ), ∀ (t x : ℝ),
    (t^4 + a*t^3 + b*t^2 + c*t + d = 0) ↔
    (t = x + k ∧ x^4 = A*x^2 + B*x + C) :=
sorry

end fourth_degree_reduction_l3989_398996


namespace cricket_bat_price_l3989_398987

/-- Calculates the final price of an item after two consecutive sales with given profit percentages -/
def finalPrice (initialCost : ℚ) (profit1 : ℚ) (profit2 : ℚ) : ℚ :=
  initialCost * (1 + profit1) * (1 + profit2)

/-- Theorem stating that a cricket bat initially costing $154, sold twice with 20% and 25% profit, results in a final price of $231 -/
theorem cricket_bat_price : 
  finalPrice 154 (20/100) (25/100) = 231 := by sorry

end cricket_bat_price_l3989_398987


namespace number_transformation_l3989_398938

theorem number_transformation (x : ℝ) : (x * (5/6) / 10 + 2/3) = 3/4 * x + 3/4 := by
  sorry

end number_transformation_l3989_398938


namespace simplify_expression_l3989_398965

theorem simplify_expression (x y : ℝ) : 7*x + 8*y - 3*x + 4*y + 10 = 4*x + 12*y + 10 := by
  sorry

end simplify_expression_l3989_398965


namespace small_glass_cost_is_three_l3989_398914

/-- The cost of a small glass given Peter's purchase information -/
def small_glass_cost (total_money : ℕ) (num_small : ℕ) (num_large : ℕ) (large_cost : ℕ) (change : ℕ) : ℕ :=
  ((total_money - change) - (num_large * large_cost)) / num_small

/-- Theorem stating that the cost of a small glass is $3 given the problem conditions -/
theorem small_glass_cost_is_three :
  small_glass_cost 50 8 5 5 1 = 3 := by
  sorry

end small_glass_cost_is_three_l3989_398914


namespace committee_selection_l3989_398933

theorem committee_selection (n : ℕ) : 
  (n.choose 3 = 20) → (n.choose 4 = 15) := by
  sorry

end committee_selection_l3989_398933


namespace quadratic_vertex_ordinate_l3989_398902

theorem quadratic_vertex_ordinate (b c : ℤ) :
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ (x₁^2 + b*x₁ + c = 2017) ∧ (x₂^2 + b*x₂ + c = 2017)) →
  (-(b^2 - 4*c) / 4 = -1016064) :=
by sorry

end quadratic_vertex_ordinate_l3989_398902


namespace roots_sum_of_powers_l3989_398986

theorem roots_sum_of_powers (γ δ : ℝ) : 
  γ^2 - 3*γ - 2 = 0 → δ^2 - 3*δ - 2 = 0 → 3*γ^4 + 7*δ^3 = -135 := by
  sorry

end roots_sum_of_powers_l3989_398986


namespace rent_increase_theorem_l3989_398919

theorem rent_increase_theorem (num_friends : ℕ) (initial_avg_rent : ℚ) 
  (increased_rent : ℚ) (increase_percentage : ℚ) :
  num_friends = 4 →
  initial_avg_rent = 800 →
  increased_rent = 1400 →
  increase_percentage = 20 / 100 →
  let total_rent : ℚ := initial_avg_rent * num_friends
  let new_increased_rent : ℚ := increased_rent * (1 + increase_percentage)
  let new_total_rent : ℚ := total_rent - increased_rent + new_increased_rent
  let new_avg_rent : ℚ := new_total_rent / num_friends
  new_avg_rent = 870 := by sorry

end rent_increase_theorem_l3989_398919


namespace pages_copied_for_fifteen_dollars_l3989_398981

/-- The number of pages that can be copied given the cost per page and available money. -/
def pages_copied (cost_per_page : ℚ) (available_money : ℚ) : ℚ :=
  (available_money * 100) / cost_per_page

/-- Theorem: Given a cost of 5 cents per page and $15 available, 300 pages can be copied. -/
theorem pages_copied_for_fifteen_dollars :
  pages_copied (5 : ℚ) (15 : ℚ) = 300 := by
  sorry

end pages_copied_for_fifteen_dollars_l3989_398981


namespace expression_simplification_l3989_398921

theorem expression_simplification (x : ℝ) :
  x - 3 * (2 + x) + 4 * (2 - x) - 5 * (1 + 3 * x) + 2 * x^2 = 2 * x^2 - 21 * x - 3 :=
by sorry

end expression_simplification_l3989_398921


namespace range_of_a_l3989_398977

theorem range_of_a (a : ℝ) : 
  (∅ : Set ℝ) ⊂ {x : ℝ | x^2 ≤ a} → a ∈ Set.Ici (0 : ℝ) := by sorry

end range_of_a_l3989_398977


namespace circle_segment_perimeter_l3989_398959

/-- Given a circle with radius 7 and a central angle of 270°, 
    the perimeter of the segment formed by this angle is equal to 14 + 10.5π. -/
theorem circle_segment_perimeter (r : ℝ) (angle : ℝ) : 
  r = 7 → angle = 270 * π / 180 → 
  2 * r + (angle / (2 * π)) * (2 * π * r) = 14 + 10.5 * π :=
by sorry

end circle_segment_perimeter_l3989_398959


namespace coffee_cost_l3989_398941

theorem coffee_cost (sandwich_cost coffee_cost : ℕ) : 
  (3 * sandwich_cost + 2 * coffee_cost = 630) →
  (2 * sandwich_cost + 3 * coffee_cost = 690) →
  coffee_cost = 162 := by
sorry

end coffee_cost_l3989_398941


namespace original_selling_price_l3989_398975

/-- Proves that the original selling price is $24000 given the conditions --/
theorem original_selling_price (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) 
  (h1 : cost_price = 20000)
  (h2 : discount_rate = 0.1)
  (h3 : profit_rate = 0.08) : 
  ∃ (selling_price : ℝ), 
    selling_price = 24000 ∧ 
    (1 - discount_rate) * selling_price = cost_price * (1 + profit_rate) :=
by
  sorry

end original_selling_price_l3989_398975


namespace max_notebooks_charlie_can_buy_l3989_398922

theorem max_notebooks_charlie_can_buy (available : ℝ) (cost_per_notebook : ℝ) 
  (h1 : available = 12) (h2 : cost_per_notebook = 1.45) : 
  ⌊available / cost_per_notebook⌋ = 8 := by
  sorry

end max_notebooks_charlie_can_buy_l3989_398922


namespace cone_base_circumference_l3989_398918

/-- Given a circular piece of paper with radius 5 inches, when a 300° sector is removed
    and the remaining sector is used to form a right circular cone,
    the circumference of the base of the cone is 25π/3 inches. -/
theorem cone_base_circumference :
  let original_radius : ℝ := 5
  let removed_angle : ℝ := 300
  let full_circle_angle : ℝ := 360
  let remaining_fraction : ℝ := (full_circle_angle - removed_angle) / full_circle_angle
  let cone_base_circumference : ℝ := 2 * π * original_radius * remaining_fraction
  cone_base_circumference = 25 * π / 3 := by
sorry

end cone_base_circumference_l3989_398918


namespace distance_AB_l3989_398978

def A : ℝ × ℝ := (-3, 2)
def B : ℝ × ℝ := (1, -1)

theorem distance_AB : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 5 := by
  sorry

end distance_AB_l3989_398978


namespace solution_opposite_implies_a_l3989_398900

theorem solution_opposite_implies_a (a : ℝ) : 
  (∃ x : ℝ, 5 * x - 1 = 2 * x + a) ∧ 
  (∃ y : ℝ, 4 * y + 3 = 7) ∧
  (∀ x y : ℝ, (5 * x - 1 = 2 * x + a ∧ 4 * y + 3 = 7) → x = -y) →
  a = -4 := by
sorry

end solution_opposite_implies_a_l3989_398900


namespace work_completion_time_l3989_398951

theorem work_completion_time (q p : ℝ) (h1 : q = 20) 
  (h2 : 4 * (1/p + 1/q) = 1 - 0.5333333333333333) : p = 15 := by
  sorry

end work_completion_time_l3989_398951


namespace pascal_contest_participants_l3989_398983

theorem pascal_contest_participants (male_count : ℕ) (ratio_male : ℕ) (ratio_female : ℕ) : 
  male_count = 21 → ratio_male = 3 → ratio_female = 7 → 
  male_count + (male_count * ratio_female / ratio_male) = 70 := by
sorry

end pascal_contest_participants_l3989_398983


namespace parabola_point_coordinates_l3989_398917

theorem parabola_point_coordinates :
  ∀ x y : ℝ,
  y = x^2 →
  |y| = |x| + 3 →
  ((x = 1 ∧ y = 4) ∨ (x = -1 ∧ y = 4)) := by
  sorry

end parabola_point_coordinates_l3989_398917


namespace calculation_proof_l3989_398964

theorem calculation_proof : |Real.sqrt 3 - 2| - 2 * Real.tan (π / 3) + (π - 2023) ^ 0 + Real.sqrt 27 = 3 := by
  sorry

end calculation_proof_l3989_398964


namespace simplify_absolute_difference_l3989_398912

theorem simplify_absolute_difference (a b : ℝ) (h : a + b < 0) :
  |a + b - 1| - |3 - a - b| = -2 := by sorry

end simplify_absolute_difference_l3989_398912


namespace art_show_ratio_l3989_398943

/-- Given an artist who painted 153 pictures and sold 72, prove that the ratio of
    remaining pictures to sold pictures, when simplified to lowest terms, is 9:8. -/
theorem art_show_ratio :
  let total_pictures : ℕ := 153
  let sold_pictures : ℕ := 72
  let remaining_pictures : ℕ := total_pictures - sold_pictures
  let ratio := (remaining_pictures, sold_pictures)
  (ratio.1.gcd ratio.2 = 9) ∧
  (ratio.1 / ratio.1.gcd ratio.2 = 9) ∧
  (ratio.2 / ratio.1.gcd ratio.2 = 8) := by
sorry


end art_show_ratio_l3989_398943


namespace vegetable_sale_mass_l3989_398979

theorem vegetable_sale_mass (carrots zucchini broccoli : ℝ) 
  (h_carrots : carrots = 15)
  (h_zucchini : zucchini = 13)
  (h_broccoli : broccoli = 8) :
  (carrots + zucchini + broccoli) / 2 = 18 := by
  sorry

end vegetable_sale_mass_l3989_398979


namespace triangle_right_angled_from_arithmetic_progression_l3989_398953

/-- Given a triangle with side lengths a, b, c, and incircle diameter 2r
    forming an arithmetic progression, prove that the triangle is right-angled. -/
theorem triangle_right_angled_from_arithmetic_progression 
  (a b c r : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_arithmetic : ∃ (d : ℝ), b = a + d ∧ c = b + d ∧ 2*r = c + d) :
  ∃ (A B C : ℝ), A + B + C = π ∧ max A B = π/2 ∧ max B C = π/2 ∧ max C A = π/2 := by
  sorry

end triangle_right_angled_from_arithmetic_progression_l3989_398953


namespace binomial_expansion_103_minus_2_pow_5_l3989_398984

theorem binomial_expansion_103_minus_2_pow_5 :
  (103 - 2)^5 = 10510100501 := by
  sorry

end binomial_expansion_103_minus_2_pow_5_l3989_398984


namespace subtraction_result_l3989_398947

theorem subtraction_result : (3.75 : ℝ) - 1.4 = 2.35 := by
  sorry

end subtraction_result_l3989_398947


namespace specific_hill_ground_depth_l3989_398994

/-- Represents a cone-shaped hill -/
structure ConeHill where
  height : ℝ
  aboveGroundVolumeFraction : ℝ

/-- Calculates the depth of the ground at the base of a cone-shaped hill -/
def groundDepth (hill : ConeHill) : ℝ :=
  hill.height * (1 - (hill.aboveGroundVolumeFraction ^ (1/3)))

/-- Theorem stating that for a specific cone-shaped hill, the ground depth is 355 feet -/
theorem specific_hill_ground_depth :
  let hill : ConeHill := { height := 5000, aboveGroundVolumeFraction := 1/5 }
  groundDepth hill = 355 := by
  sorry

end specific_hill_ground_depth_l3989_398994


namespace quadratic_equation_solution_l3989_398934

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = 4 + 3 * Real.sqrt 2 ∧ 
  x₂ = 4 - 3 * Real.sqrt 2 ∧ 
  x₁^2 - 8*x₁ - 2 = 0 ∧ 
  x₂^2 - 8*x₂ - 2 = 0 := by
  sorry

end quadratic_equation_solution_l3989_398934


namespace sixty_percent_of_40_minus_four_fifths_of_25_l3989_398980

theorem sixty_percent_of_40_minus_four_fifths_of_25 : (60 / 100 * 40) - (4 / 5 * 25) = 4 := by
  sorry

end sixty_percent_of_40_minus_four_fifths_of_25_l3989_398980


namespace donation_analysis_l3989_398969

/-- Represents the donation amounts and their frequencies --/
def donation_data : List (ℕ × ℕ) := [(5, 1), (10, 5), (15, 3), (20, 1)]

/-- Total number of students in the sample --/
def sample_size : ℕ := 10

/-- Total number of students in the school --/
def school_size : ℕ := 2200

/-- Calculates the mode of the donation data --/
def mode (data : List (ℕ × ℕ)) : ℕ := sorry

/-- Calculates the median of the donation data --/
def median (data : List (ℕ × ℕ)) : ℕ := sorry

/-- Calculates the average donation amount --/
def average (data : List (ℕ × ℕ)) : ℚ := sorry

/-- Estimates the total donation for the school --/
def estimate_total (avg : ℚ) (school_size : ℕ) : ℕ := sorry

theorem donation_analysis :
  mode donation_data = 10 ∧
  median donation_data = 10 ∧
  average donation_data = 12 ∧
  estimate_total (average donation_data) school_size = 26400 := by sorry

end donation_analysis_l3989_398969


namespace count_four_digit_numbers_l3989_398950

theorem count_four_digit_numbers :
  let first_four_digit : Nat := 1000
  let last_four_digit : Nat := 9999
  (last_four_digit - first_four_digit + 1 : Nat) = 9000 := by
sorry

end count_four_digit_numbers_l3989_398950


namespace max_value_is_35_l3989_398930

def Digits : Finset ℕ := {1, 2, 5, 6}

def Expression (a b c d : ℕ) : ℕ := (a - b)^2 + c * d

theorem max_value_is_35 :
  ∃ (a b c d : ℕ),
    a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    Expression a b c d = 35 ∧
    ∀ (w x y z : ℕ),
      w ∈ Digits → x ∈ Digits → y ∈ Digits → z ∈ Digits →
      w ≠ x → w ≠ y → w ≠ z → x ≠ y → x ≠ z → y ≠ z →
      Expression w x y z ≤ 35 :=
by sorry

end max_value_is_35_l3989_398930


namespace regular_polygon_perimeter_l3989_398946

/-- A regular polygon with side length 8 units and exterior angle 72 degrees has a perimeter of 40 units. -/
theorem regular_polygon_perimeter (s : ℝ) (θ : ℝ) (n : ℕ) : 
  s = 8 → θ = 72 → θ = 360 / n → n * s = 40 := by
  sorry

end regular_polygon_perimeter_l3989_398946


namespace initial_friends_correct_l3989_398945

/-- The number of friends initially playing the game -/
def initial_friends : ℕ := 2

/-- The number of new players that joined -/
def new_players : ℕ := 2

/-- The number of lives each player has -/
def lives_per_player : ℕ := 6

/-- The total number of lives after new players joined -/
def total_lives : ℕ := 24

/-- Theorem stating that the number of initial friends is correct -/
theorem initial_friends_correct : 
  lives_per_player * (initial_friends + new_players) = total_lives := by
  sorry

end initial_friends_correct_l3989_398945


namespace min_value_of_expression_l3989_398990

theorem min_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / y + y / z + z / x + x / z ≥ 4 ∧
  (x / y + y / z + z / x + x / z = 4 ↔ x = y ∧ y = z) :=
sorry

end min_value_of_expression_l3989_398990


namespace distance_after_walk_l3989_398901

theorem distance_after_walk (west_distance : ℝ) (north_distance : ℝ) :
  west_distance = 10 →
  north_distance = 10 →
  ∃ (total_distance : ℝ), total_distance^2 = west_distance^2 + north_distance^2 :=
by
  sorry

end distance_after_walk_l3989_398901


namespace intersection_of_A_and_B_l3989_398907

def A : Set ℝ := {x | -3 ≤ x ∧ x < 4}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x | -2 ≤ x ∧ x < 4} := by
  sorry

end intersection_of_A_and_B_l3989_398907


namespace old_phone_plan_cost_l3989_398958

theorem old_phone_plan_cost 
  (new_plan_cost : ℝ) 
  (price_increase_percentage : ℝ) 
  (h1 : new_plan_cost = 195) 
  (h2 : price_increase_percentage = 0.30) : 
  new_plan_cost / (1 + price_increase_percentage) = 150 := by
sorry

end old_phone_plan_cost_l3989_398958


namespace uncool_parents_count_l3989_398904

theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ) :
  total = 35 →
  cool_dads = 18 →
  cool_moms = 22 →
  both_cool = 11 →
  total - (cool_dads + cool_moms - both_cool) = 6 :=
by sorry

end uncool_parents_count_l3989_398904


namespace even_power_minus_one_factorization_l3989_398985

theorem even_power_minus_one_factorization (n : ℕ) (h1 : Even n) (h2 : n > 4) :
  ∃ (a b c : ℕ), (a > 1 ∧ b > 1 ∧ c > 1) ∧ (2^n - 1 = a * b * c) :=
sorry

end even_power_minus_one_factorization_l3989_398985


namespace real_part_of_complex_product_l3989_398963

theorem real_part_of_complex_product : Complex.re ((1 + 3 * Complex.I) * Complex.I) = -3 := by
  sorry

end real_part_of_complex_product_l3989_398963


namespace henri_drove_farther_l3989_398970

/-- Proves that Henri drove 305 miles farther than Gervais -/
theorem henri_drove_farther (gervais_avg_daily : ℕ) (gervais_days : ℕ) (henri_total : ℕ) 
  (h1 : gervais_avg_daily = 315)
  (h2 : gervais_days = 3)
  (h3 : henri_total = 1250) :
  henri_total - (gervais_avg_daily * gervais_days) = 305 := by
  sorry

#check henri_drove_farther

end henri_drove_farther_l3989_398970


namespace quadratic_decreasing_condition_l3989_398927

/-- Given a quadratic function y = (x - m)^2 - 1, if it decreases as x increases
    when x ≤ 3, then m ≥ 3. -/
theorem quadratic_decreasing_condition (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ 3 → 
    ((x₁ - m)^2 - 1) > ((x₂ - m)^2 - 1)) → 
  m ≥ 3 := by
  sorry

end quadratic_decreasing_condition_l3989_398927


namespace both_selected_probability_l3989_398954

def prob_X : ℚ := 1/5
def prob_Y : ℚ := 2/7

theorem both_selected_probability : prob_X * prob_Y = 2/35 := by
  sorry

end both_selected_probability_l3989_398954


namespace green_light_probability_theorem_l3989_398915

/-- Represents the duration of traffic light colors in seconds -/
structure TrafficLightDuration where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of encountering a green light -/
def greenLightProbability (d : TrafficLightDuration) : ℚ :=
  d.green / (d.red + d.yellow + d.green)

/-- Theorem: The probability of encountering a green light at the given intersection is 8/15 -/
theorem green_light_probability_theorem (d : TrafficLightDuration) 
    (h1 : d.red = 30)
    (h2 : d.yellow = 5)
    (h3 : d.green = 40) : 
  greenLightProbability d = 8 / 15 := by
  sorry

end green_light_probability_theorem_l3989_398915


namespace domain_of_f_l3989_398906

noncomputable def f (x : ℝ) : ℝ := (x^3 - 3*x^2 + 5*x - 2) / (x^3 - 5*x^2 + 8*x - 4)

theorem domain_of_f :
  Set.range f = {x : ℝ | x ∈ (Set.Iio 1) ∪ (Set.Ioo 1 2) ∪ (Set.Ioo 2 4) ∪ (Set.Ioi 4)} :=
by sorry

end domain_of_f_l3989_398906


namespace associated_equation_l3989_398976

def equation1 (x : ℝ) : Prop := 5 * x - 2 = 0

def equation2 (x : ℝ) : Prop := 3/4 * x + 1 = 0

def equation3 (x : ℝ) : Prop := x - (3 * x + 1) = -5

def inequality_system (x : ℝ) : Prop := 2 * x - 5 > 3 * x - 8 ∧ -4 * x + 3 < x - 4

theorem associated_equation : 
  ∃ (x : ℝ), equation3 x ∧ inequality_system x ∧
  (∀ (y : ℝ), equation1 y → ¬inequality_system y) ∧
  (∀ (y : ℝ), equation2 y → ¬inequality_system y) :=
sorry

end associated_equation_l3989_398976


namespace min_product_sum_l3989_398957

/-- Triangle ABC with side lengths a, b, c and height h from A to BC -/
structure Triangle :=
  (a b c h : ℝ)
  (positive_a : 0 < a)
  (positive_b : 0 < b)
  (positive_c : 0 < c)
  (positive_h : 0 < h)

/-- The problem statement -/
theorem min_product_sum (t : Triangle) (h1 : t.c = 10) (h2 : t.h = 3) :
  let min_product := Real.sqrt ((t.c^2 * t.h^2) / 4)
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b = min_product ∧ a + b = 4 * Real.sqrt 10 :=
sorry

end min_product_sum_l3989_398957


namespace initial_mean_calculation_l3989_398972

theorem initial_mean_calculation (n : ℕ) (correct_value wrong_value : ℝ) (correct_mean : ℝ) (M : ℝ) :
  n = 30 →
  correct_value = 145 →
  wrong_value = 135 →
  correct_mean = 140.33333333333334 →
  n * M + (correct_value - wrong_value) = n * correct_mean →
  M = 140 :=
by sorry

end initial_mean_calculation_l3989_398972


namespace waiter_customers_l3989_398998

/-- Calculates the total number of customers given the number of tables and people per table -/
def total_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) : ℕ :=
  num_tables * (women_per_table + men_per_table)

/-- Proves that given 7 tables with 7 women and 2 men each, the total number of customers is 63 -/
theorem waiter_customers : total_customers 7 7 2 = 63 := by
  sorry

end waiter_customers_l3989_398998


namespace hua_luogeng_birthday_factorization_l3989_398952

theorem hua_luogeng_birthday_factorization (h : 19101112 = 1163 * 16424) :
  Nat.Prime 1163 ∧ ¬Nat.Prime 16424 := by
  sorry

end hua_luogeng_birthday_factorization_l3989_398952


namespace maggie_subscriptions_to_parents_l3989_398974

-- Define the price per subscription
def price_per_subscription : ℕ := 5

-- Define the number of subscriptions sold to different people
def subscriptions_to_grandfather : ℕ := 1
def subscriptions_to_next_door : ℕ := 2
def subscriptions_to_another_neighbor : ℕ := 2 * subscriptions_to_next_door

-- Define the total earnings
def total_earnings : ℕ := 55

-- Define the number of subscriptions sold to parents
def subscriptions_to_parents : ℕ := 4

-- Theorem to prove
theorem maggie_subscriptions_to_parents :
  subscriptions_to_parents * price_per_subscription +
  (subscriptions_to_grandfather + subscriptions_to_next_door + subscriptions_to_another_neighbor) * price_per_subscription =
  total_earnings :=
by sorry

end maggie_subscriptions_to_parents_l3989_398974


namespace wheel_probability_l3989_398932

theorem wheel_probability (p_D p_E p_F p_G : ℚ) : 
  p_D = 3/8 → p_E = 1/4 → p_G = 1/8 → 
  p_D + p_E + p_F + p_G = 1 →
  p_F = 1/4 := by
sorry

end wheel_probability_l3989_398932


namespace fraction_sum_l3989_398993

theorem fraction_sum : (3 : ℚ) / 9 + (6 : ℚ) / 12 = (5 : ℚ) / 6 := by
  sorry

end fraction_sum_l3989_398993


namespace complex_magnitude_equality_l3989_398929

theorem complex_magnitude_equality (n : ℝ) (hn : n > 0) :
  Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 10 → n = 15 := by
sorry

end complex_magnitude_equality_l3989_398929


namespace halloween_goodie_bags_l3989_398948

theorem halloween_goodie_bags (vampire_students pumpkin_students : ℕ)
  (pack_size pack_cost individual_cost total_cost : ℕ) :
  vampire_students = 11 →
  pumpkin_students = 14 →
  pack_size = 5 →
  pack_cost = 3 →
  individual_cost = 1 →
  total_cost = 17 →
  vampire_students + pumpkin_students = 25 :=
by sorry

end halloween_goodie_bags_l3989_398948


namespace andy_problem_solving_l3989_398995

theorem andy_problem_solving (last_problem : ℕ) (total_solved : ℕ) (h1 : last_problem = 125) (h2 : total_solved = 51) : 
  last_problem - total_solved + 1 = 75 := by
sorry

end andy_problem_solving_l3989_398995


namespace students_playing_both_sports_l3989_398989

def total_students : ℕ := 460
def football_players : ℕ := 325
def cricket_players : ℕ := 175
def neither_players : ℕ := 50

theorem students_playing_both_sports : 
  total_students - neither_players = football_players + cricket_players - 90 := by
sorry

end students_playing_both_sports_l3989_398989


namespace equation_solution_l3989_398939

theorem equation_solution : ∃ x : ℚ, (54 - x / 6 * 3 = 36) ∧ (x = 36) := by
  sorry

end equation_solution_l3989_398939


namespace complex_modulus_problem_l3989_398925

theorem complex_modulus_problem (z : ℂ) (h : (z - Complex.I) * Complex.I = 2 + 3 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_modulus_problem_l3989_398925


namespace distance_home_to_school_l3989_398920

/-- Represents Johnny's journey to and from school -/
structure JourneySegment where
  speed : ℝ
  time : ℝ
  distance : ℝ
  (distance_eq : distance = speed * time)

/-- Represents Johnny's complete journey -/
structure Journey where
  jog : JourneySegment
  bike : JourneySegment
  bus : JourneySegment

/-- The journey satisfies the given conditions -/
def journey_conditions (j : Journey) : Prop :=
  j.jog.speed = 5 ∧
  j.bike.speed = 10 ∧
  j.bus.speed = 30 ∧
  j.jog.time = 1 ∧
  j.bike.time = 1 ∧
  j.bus.time = 1

/-- The theorem stating the distance from home to school -/
theorem distance_home_to_school (j : Journey) 
  (h : journey_conditions j) : 
  j.bus.distance - j.bike.distance = 20 := by
  sorry


end distance_home_to_school_l3989_398920


namespace simplify_and_evaluate_l3989_398992

theorem simplify_and_evaluate (a b : ℝ) : 
  a = Real.tan (π / 3) → 
  b = Real.sin (π / 3) → 
  ((b^2 + a^2) / a - 2 * b) / (1 - b / a) = Real.sqrt 3 / 2 := by
  sorry

end simplify_and_evaluate_l3989_398992


namespace max_sum_of_factors_l3989_398973

theorem max_sum_of_factors (a b : ℕ) (h : a * b = 48) : 
  ∃ (x y : ℕ), x * y = 48 ∧ x + y ≤ a + b ∧ x + y = 49 :=
by sorry

end max_sum_of_factors_l3989_398973


namespace wallpaper_overlap_area_l3989_398960

/-- Given the total area of wallpaper and areas covered by exactly two and three layers,
    calculate the actual area of the wall covered by overlapping wallpapers. -/
theorem wallpaper_overlap_area (total_area double_layer triple_layer : ℝ) 
    (h1 : total_area = 300)
    (h2 : double_layer = 30)
    (h3 : triple_layer = 45) :
    total_area - (2 * double_layer - double_layer) - (3 * triple_layer - triple_layer) = 180 := by
  sorry


end wallpaper_overlap_area_l3989_398960


namespace largest_of_eight_consecutive_integers_l3989_398991

theorem largest_of_eight_consecutive_integers (n : ℕ) 
  (h1 : n > 0) 
  (h2 : n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) = 2024) : 
  n + 7 = 256 := by
  sorry

#check largest_of_eight_consecutive_integers

end largest_of_eight_consecutive_integers_l3989_398991


namespace intersection_equality_implies_range_l3989_398966

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

-- Define the range of a
def range_a : Set ℝ := {a | a = 1 ∨ a ≤ -1}

-- Theorem statement
theorem intersection_equality_implies_range (a : ℝ) : 
  A ∩ B a = B a → a ∈ range_a :=
sorry

end intersection_equality_implies_range_l3989_398966


namespace inequality_proof_l3989_398962

theorem inequality_proof (a b c d : ℤ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0) 
  (h5 : a * d = b * c) : 
  (a - d)^2 ≥ 4*d + 8 := by
sorry

end inequality_proof_l3989_398962


namespace smallest_three_digit_prime_product_l3989_398928

theorem smallest_three_digit_prime_product : ∃ (n a b : ℕ),
  (100 ≤ n ∧ n < 1000) ∧  -- n is a three-digit positive integer
  (n = a * b * (10 * a + b)) ∧  -- n is the product of a, b, and 10a+b
  (a < 10 ∧ b < 10) ∧  -- a and b are each less than 10
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime (10 * a + b) ∧  -- a, b, and 10a+b are prime
  a ≠ b ∧ a ≠ (10 * a + b) ∧ b ≠ (10 * a + b) ∧  -- a, b, and 10a+b are distinct
  (∀ (m c d : ℕ), 
    (100 ≤ m ∧ m < 1000) →
    (m = c * d * (10 * c + d)) →
    (c < 10 ∧ d < 10) →
    Nat.Prime c → Nat.Prime d → Nat.Prime (10 * c + d) →
    c ≠ d → c ≠ (10 * c + d) → d ≠ (10 * c + d) →
    n ≤ m) ∧
  n = 138 :=
by sorry

end smallest_three_digit_prime_product_l3989_398928


namespace solution_proof_l3989_398982

/-- Custom operation for 2x2 matrices -/
def matrix_op (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem stating the solution to the given equation -/
theorem solution_proof :
  ∃ x : ℝ, matrix_op (x + 1) (x + 2) (x - 3) (x - 1) = 27 ∧ x = 22 := by
  sorry

end solution_proof_l3989_398982


namespace building_height_from_shadows_l3989_398949

/-- Given a flagstaff and a building casting shadows under similar conditions,
    calculate the height of the building. -/
theorem building_height_from_shadows
  (flagstaff_height : ℝ)
  (flagstaff_shadow : ℝ)
  (building_shadow : ℝ)
  (flagstaff_height_pos : 0 < flagstaff_height)
  (flagstaff_shadow_pos : 0 < flagstaff_shadow)
  (building_shadow_pos : 0 < building_shadow)
  (h_flagstaff : flagstaff_height = 17.5)
  (h_flagstaff_shadow : flagstaff_shadow = 40.25)
  (h_building_shadow : building_shadow = 28.75) :
  flagstaff_height / flagstaff_shadow * building_shadow = 12.4375 := by
sorry

end building_height_from_shadows_l3989_398949


namespace smallest_n_congruence_l3989_398940

theorem smallest_n_congruence (n : ℕ+) : 
  (5 * n.val ≡ 2345 [MOD 26]) ↔ n = 1 := by sorry

end smallest_n_congruence_l3989_398940


namespace min_value_z_l3989_398926

theorem min_value_z (a b : ℝ) : 
  ∃ (m : ℝ), ∀ (x y : ℝ), x^2 + y^2 ≤ 25 ∧ 2*x + y ≤ 5 → 
  x^2 + y^2 - 2*a*x - 2*b*y ≥ m ∧ m ≥ -a^2 - b^2 := by
  sorry

end min_value_z_l3989_398926


namespace two_intersection_points_l3989_398944

/-- Define the first curve -/
def curve1 (x y : ℝ) : Prop :=
  (x + 2*y - 6) * (2*x - y + 4) = 0

/-- Define the second curve -/
def curve2 (x y : ℝ) : Prop :=
  (x - 3*y + 2) * (4*x + y - 14) = 0

/-- Define an intersection point -/
def is_intersection (x y : ℝ) : Prop :=
  curve1 x y ∧ curve2 x y

/-- The theorem stating that there are exactly two distinct intersection points -/
theorem two_intersection_points :
  ∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    ∀ (x y : ℝ), is_intersection x y → (x, y) = p1 ∨ (x, y) = p2 :=
by
  sorry

end two_intersection_points_l3989_398944


namespace special_list_median_l3989_398908

/-- The sum of integers from 1 to n -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The list where each integer n from 1 to 100 appears n times -/
def special_list : List ℕ := sorry

/-- The median of a list is the average of the middle two elements when the list has even length -/
def median (l : List ℕ) : ℚ := sorry

theorem special_list_median :
  median special_list = 71 := by sorry

end special_list_median_l3989_398908


namespace equation_solution_difference_l3989_398961

theorem equation_solution_difference : ∃ (s₁ s₂ : ℝ),
  (s₁^2 - 5*s₁ - 24) / (s₁ + 3) = 3*s₁ + 10 ∧
  (s₂^2 - 5*s₂ - 24) / (s₂ + 3) = 3*s₂ + 10 ∧
  s₁ ≠ s₂ ∧
  |s₁ - s₂| = 16 :=
by
  sorry

end equation_solution_difference_l3989_398961


namespace temperature_85_at_latest_time_l3989_398916

/-- The temperature function in Denver, CO, where t is time in hours past noon -/
def temperature (t : ℝ) : ℝ := -t^2 + 14*t + 40

/-- The latest time when the temperature is 85 degrees -/
def latest_85_degrees : ℝ := 9

theorem temperature_85_at_latest_time :
  temperature latest_85_degrees = 85 ∧
  ∀ t > latest_85_degrees, temperature t ≠ 85 := by
sorry

end temperature_85_at_latest_time_l3989_398916


namespace odd_periodic_function_property_l3989_398942

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_periodic : is_periodic f 5) 
  (h_value : f 7 = 9) : 
  f 2020 - f 2018 = 9 := by
  sorry

end odd_periodic_function_property_l3989_398942


namespace rectangle_area_error_percent_l3989_398955

/-- Theorem: Error percent in rectangle area calculation --/
theorem rectangle_area_error_percent
  (L W : ℝ)  -- L and W represent the actual length and width of the rectangle
  (h_positive_L : L > 0)
  (h_positive_W : W > 0)
  (measured_length : ℝ := 1.05 * L)  -- Length measured 5% in excess
  (measured_width : ℝ := 0.96 * W)   -- Width measured 4% in deficit
  (actual_area : ℝ := L * W)
  (calculated_area : ℝ := measured_length * measured_width)
  : (calculated_area - actual_area) / actual_area * 100 = 0.8 := by
  sorry

end rectangle_area_error_percent_l3989_398955


namespace count_even_three_digit_numbers_less_than_700_l3989_398931

def valid_digits : List Nat := [1, 2, 3, 4, 5, 6]

def is_even (n : Nat) : Bool :=
  n % 2 = 0

def is_three_digit (n : Nat) : Bool :=
  100 ≤ n ∧ n < 1000

def count_valid_numbers : Nat :=
  (valid_digits.filter (· < 7)).length *
  valid_digits.length *
  (valid_digits.filter is_even).length

theorem count_even_three_digit_numbers_less_than_700 :
  count_valid_numbers = 108 := by
  sorry

end count_even_three_digit_numbers_less_than_700_l3989_398931


namespace geometric_series_first_term_l3989_398971

theorem geometric_series_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 120)
  : a = 240 / 7 :=
by sorry

end geometric_series_first_term_l3989_398971


namespace A_cubed_is_zero_l3989_398909

open Matrix

theorem A_cubed_is_zero {α : Type*} [Field α] (A : Matrix (Fin 2) (Fin 2) α) 
  (h1 : A ^ 4 = 0)
  (h2 : Matrix.trace A = 0) :
  A ^ 3 = 0 := by
  sorry

end A_cubed_is_zero_l3989_398909


namespace correct_equation_l3989_398910

theorem correct_equation : (-3)^2 = 9 ∧ 
  (-2)^3 ≠ -6 ∧ 
  ¬(∀ x, x^2 = 4 → x = 2 ∨ x = -2) ∧ 
  (Real.sqrt 2)^2 ≠ 4 := by
  sorry

end correct_equation_l3989_398910


namespace rate_of_change_kinetic_energy_l3989_398988

/-- The rate of change of kinetic energy for a system with increasing mass -/
theorem rate_of_change_kinetic_energy
  (M : ℝ)  -- Initial mass of the system
  (v : ℝ)  -- Constant velocity of the system
  (ρ : ℝ)  -- Rate of mass increase
  (h1 : M > 0)  -- Mass is positive
  (h2 : v ≠ 0)  -- Velocity is non-zero
  (h3 : ρ > 0)  -- Rate of mass increase is positive
  : 
  ∃ (K : ℝ → ℝ), -- Kinetic energy as a function of time
    (∀ t, K t = (1/2) * (M + ρ * t) * v^2) ∧ 
    (∀ t, deriv K t = (1/2) * ρ * v^2) :=
sorry

end rate_of_change_kinetic_energy_l3989_398988


namespace product_sum_bounds_l3989_398924

theorem product_sum_bounds (x y z t : ℝ) 
  (sum_zero : x + y + z + t = 0) 
  (sum_squares_one : x^2 + y^2 + z^2 + t^2 = 1) : 
  -1 ≤ x*y + y*z + z*t + t*x ∧ x*y + y*z + z*t + t*x ≤ 0 := by
  sorry

end product_sum_bounds_l3989_398924


namespace morning_earnings_l3989_398997

/-- Represents the types of vehicles William washes --/
inductive VehicleType
  | NormalCar
  | BigSUV
  | Minivan

/-- Represents a customer's order --/
structure Order where
  vehicles : List VehicleType
  multipleVehicles : Bool

def basePrice (v : VehicleType) : ℚ :=
  match v with
  | VehicleType.NormalCar => 15
  | VehicleType.BigSUV => 25
  | VehicleType.Minivan => 20

def washTime (v : VehicleType) : ℚ :=
  match v with
  | VehicleType.NormalCar => 1
  | VehicleType.BigSUV => 2
  | VehicleType.Minivan => 1.5

def applyDiscount (price : ℚ) : ℚ :=
  price * (1 - 0.1)

def calculateOrderPrice (o : Order) : ℚ :=
  let baseTotal := (o.vehicles.map basePrice).sum
  if o.multipleVehicles then applyDiscount baseTotal else baseTotal

def morningOrders : List Order :=
  [
    { vehicles := [VehicleType.NormalCar, VehicleType.NormalCar, VehicleType.NormalCar,
                   VehicleType.BigSUV, VehicleType.BigSUV, VehicleType.Minivan],
      multipleVehicles := false },
    { vehicles := [VehicleType.NormalCar, VehicleType.NormalCar, VehicleType.BigSUV],
      multipleVehicles := true }
  ]

theorem morning_earnings :
  (morningOrders.map calculateOrderPrice).sum = 164.5 := by sorry

end morning_earnings_l3989_398997
