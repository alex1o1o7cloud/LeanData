import Mathlib

namespace projection_theorem_l438_43877

/-- Given vectors a, b, and c in ℝ², prove that the projection of a onto c is 4 -/
theorem projection_theorem (a b c : ℝ × ℝ) : 
  a = (4, 2) → b = (2, 1) → c = (3, 4) → a.1 / b.1 = a.2 / b.2 →
  (a.1 * c.1 + a.2 * c.2) / Real.sqrt (c.1^2 + c.2^2) = 4 := by
  sorry

end projection_theorem_l438_43877


namespace geometric_sequence_third_term_l438_43831

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the theorem
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_prod : a 2 * a 4 = 16)
  (h_sum : a 1 + a 5 = 17) :
  a 3 = 4 :=
sorry

end geometric_sequence_third_term_l438_43831


namespace distance_product_sum_bound_l438_43885

/-- Given an equilateral triangle with side length 1 and a point P inside it,
    let a, b, c be the distances from P to the three sides of the triangle. -/
def DistancesFromPoint (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = Real.sqrt 3 / 2

/-- The sum of products of distances from a point inside an equilateral triangle
    to its sides is bounded. -/
theorem distance_product_sum_bound {a b c : ℝ} (h : DistancesFromPoint a b c) :
  0 < a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 1 / 4 := by
  sorry

end distance_product_sum_bound_l438_43885


namespace lcm_48_180_l438_43864

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  sorry

end lcm_48_180_l438_43864


namespace water_conservation_l438_43861

/-- Represents the amount of water in tons, where negative values indicate waste and positive values indicate savings. -/
def WaterAmount : Type := ℤ

/-- Records the water amount given the number of tons wasted or saved. -/
def recordWaterAmount (tons : ℤ) : WaterAmount := tons

theorem water_conservation (waste : WaterAmount) (save : ℤ) :
  waste = recordWaterAmount (-10) →
  recordWaterAmount save = recordWaterAmount 30 :=
by sorry

end water_conservation_l438_43861


namespace polynomial_factorization_l438_43846

theorem polynomial_factorization (x : ℝ) :
  (∃ (a b c d : ℝ), x^2 - 1 = (a*x + b) * (c*x + d)) ∧
  (∀ (a b c d : ℝ), x + 1 ≠ (a*x + b) * (c*x + d)) ∧
  (∀ (a b c d : ℝ), x^2 + x + 1 ≠ (a*x + b) * (c*x + d)) ∧
  (∀ (a b c d : ℝ), x^2 + 4 ≠ (a*x + b) * (c*x + d)) :=
by sorry

end polynomial_factorization_l438_43846


namespace vehicle_wheels_count_l438_43854

/-- Proves that each vehicle has 4 wheels given the problem conditions -/
theorem vehicle_wheels_count (total_vehicles : ℕ) (total_wheels : ℕ) 
  (h1 : total_vehicles = 25)
  (h2 : total_wheels = 100) :
  total_wheels / total_vehicles = 4 := by
  sorry

#check vehicle_wheels_count

end vehicle_wheels_count_l438_43854


namespace polynomial_remainder_l438_43876

theorem polynomial_remainder (x : ℝ) : 
  (x^4 - 2*x^3 + x + 5) % (x - 2) = 7 := by
  sorry

end polynomial_remainder_l438_43876


namespace square_plus_integer_l438_43887

theorem square_plus_integer (y : ℝ) : y^2 + 14*y + 48 = (y+7)^2 - 1 := by
  sorry

end square_plus_integer_l438_43887


namespace clock_equivalent_hours_l438_43802

theorem clock_equivalent_hours : ∃ h : ℕ, h > 6 ∧ h ≡ h^2 [ZMOD 24] ∧ ∀ k : ℕ, k > 6 ∧ k < h → ¬(k ≡ k^2 [ZMOD 24]) :=
by
  -- The proof goes here
  sorry

end clock_equivalent_hours_l438_43802


namespace problem_statement_l438_43847

theorem problem_statement (n b : ℝ) : n = 2^(1/10) ∧ n^b = 16 → b = 40 := by
  sorry

end problem_statement_l438_43847


namespace smallest_angle_for_tan_equation_l438_43810

theorem smallest_angle_for_tan_equation :
  ∃ x : ℝ, x > 0 ∧ x < 2 * Real.pi ∧
  Real.tan (6 * x) = (Real.sin x - Real.cos x) / (Real.sin x + Real.cos x) ∧
  x = 45 * Real.pi / (7 * 180) ∧
  ∀ y : ℝ, y > 0 → y < 2 * Real.pi →
    Real.tan (6 * y) = (Real.sin y - Real.cos y) / (Real.sin y + Real.cos y) →
    x ≤ y :=
by sorry

end smallest_angle_for_tan_equation_l438_43810


namespace magnitude_of_b_l438_43843

def a : ℝ × ℝ := (2, 3)

theorem magnitude_of_b (b : ℝ × ℝ) 
  (h : (a.1 + b.1, a.2 + b.2) • (a.1 - b.1, a.2 - b.2) = 0) : 
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 13 := by
  sorry

end magnitude_of_b_l438_43843


namespace expression_equality_l438_43856

/-- Given two real numbers a and b, prove that the expression
    "the difference between three times the number for A and the number for B
    divided by the sum of the number for A and twice the number for B"
    is equal to (3a - b) / (a + 2b) -/
theorem expression_equality (a b : ℝ) : 
  (3 * a - b) / (a + 2 * b) = 
  (3 * a - b) / (a + 2 * b) := by sorry

end expression_equality_l438_43856


namespace max_value_of_x_plus_inverse_l438_43834

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  (∀ y : ℝ, y > 0 → y^2 + 1/y^2 = 13 → x + 1/x ≥ y + 1/y) ∧ x + 1/x = Real.sqrt 15 :=
sorry

end max_value_of_x_plus_inverse_l438_43834


namespace equal_color_diagonals_l438_43819

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → Point

/-- A coloring of vertices of a polygon -/
def VertexColoring (n : ℕ) := Fin n → Bool

/-- The number of diagonals with both endpoints of a given color -/
def numSameColorDiagonals (n : ℕ) (coloring : VertexColoring n) (color : Bool) : ℕ := sorry

theorem equal_color_diagonals 
  (polygon : RegularPolygon 20) 
  (coloring : VertexColoring 20)
  (h_black_count : (Finset.filter (fun i => coloring i = true) (Finset.univ : Finset (Fin 20))).card = 10)
  (h_white_count : (Finset.filter (fun i => coloring i = false) (Finset.univ : Finset (Fin 20))).card = 10) :
  numSameColorDiagonals 20 coloring true = numSameColorDiagonals 20 coloring false := by
  sorry


end equal_color_diagonals_l438_43819


namespace fifth_sample_number_l438_43879

def isValidNumber (n : ℕ) : Bool :=
  1 ≤ n ∧ n ≤ 700

def findNthValidNumber (sequence : List ℕ) (n : ℕ) : Option ℕ :=
  let validNumbers := sequence.filter isValidNumber
  let uniqueValidNumbers := validNumbers.eraseDups
  uniqueValidNumbers.get? (n - 1)

theorem fifth_sample_number (sequence : List ℕ) : 
  findNthValidNumber sequence 5 = some 328 := by
  sorry

end fifth_sample_number_l438_43879


namespace function_properties_l438_43842

/-- Given a function f(x) = 2x - a/x where f(1) = 3, this theorem proves
    properties about the value of a, the parity of f, and its monotonicity. -/
theorem function_properties (a : ℝ) (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = 2*x - a/x)
    (h_f1 : f 1 = 3) :
  (a = -1) ∧ 
  (∀ x, f (-x) = -f x) ∧
  (∀ x₁ x₂, 1 < x₂ ∧ x₂ < x₁ → f x₂ < f x₁) :=
by sorry


end function_properties_l438_43842


namespace irrational_approximation_l438_43825

theorem irrational_approximation (x : ℝ) (h_irr : Irrational x) (h_pos : x > 0) :
  ∀ n : ℕ, ∃ p q : ℤ, q > n ∧ q > 0 ∧ |x - (p : ℝ) / q| < 1 / q^2 := by
  sorry

end irrational_approximation_l438_43825


namespace two_numbers_difference_l438_43899

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 := by
  sorry

end two_numbers_difference_l438_43899


namespace rectangle_area_l438_43868

theorem rectangle_area (width : ℝ) (length : ℝ) : 
  length = 4 * width → 
  2 * length + 2 * width = 200 → 
  length * width = 1600 := by
  sorry

end rectangle_area_l438_43868


namespace magic_trick_minimum_cards_l438_43821

/-- The number of possible colors for the cards -/
def num_colors : ℕ := 2017

/-- The strategy function type for the assistant -/
def Strategy : Type := Fin num_colors → Fin num_colors

/-- The minimum number of cards needed for the trick -/
def min_cards : ℕ := 2018

theorem magic_trick_minimum_cards :
  ∀ (n : ℕ), n < min_cards →
    ¬∃ (s : Strategy),
      ∀ (colors : Fin n → Fin num_colors),
        ∃ (i : Fin n),
          ∀ (j : Fin n),
            j ≠ i →
              s (colors j) = colors i := by sorry

end magic_trick_minimum_cards_l438_43821


namespace ball_count_problem_l438_43888

/-- Proves that given the initial ratio of green to yellow balls is 3:7, 
    and after removing 9 balls of each color the new ratio becomes 1:3, 
    the original number of balls in the bag was 90. -/
theorem ball_count_problem (g y : ℕ) : 
  g * 7 = y * 3 →  -- initial ratio is 3:7
  (g - 9) * 3 = (y - 9) * 1 →  -- new ratio is 1:3 after removing 9 of each
  g + y = 90 := by  -- total number of balls is 90
sorry

end ball_count_problem_l438_43888


namespace magic_triangle_max_sum_l438_43800

theorem magic_triangle_max_sum (a b c d e f : ℕ) : 
  a ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  b ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  c ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  d ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  e ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  f ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f →
  a + b + c = c + d + e ∧ c + d + e = e + f + a →
  a + b + c ≤ 48 := by
sorry

end magic_triangle_max_sum_l438_43800


namespace bush_current_age_l438_43871

def matt_future_age : ℕ := 25
def years_to_future : ℕ := 10
def age_difference : ℕ := 3

theorem bush_current_age : 
  matt_future_age - years_to_future - age_difference = 12 := by
  sorry

end bush_current_age_l438_43871


namespace sweets_problem_l438_43829

/-- The number of sweets initially on the table -/
def initial_sweets : ℕ := 50

/-- The number of sweets Jack took -/
def jack_sweets (total : ℕ) : ℕ := total / 2 + 4

/-- The number of sweets remaining after Jack -/
def after_jack (total : ℕ) : ℕ := total - jack_sweets total

/-- The number of sweets Paul took -/
def paul_sweets (remaining : ℕ) : ℕ := remaining / 3 + 5

/-- The number of sweets remaining after Paul -/
def after_paul (remaining : ℕ) : ℕ := remaining - paul_sweets remaining

/-- Olivia took the last 9 sweets -/
def olivia_sweets : ℕ := 9

theorem sweets_problem :
  after_paul (after_jack initial_sweets) = olivia_sweets :=
sorry

end sweets_problem_l438_43829


namespace mass_of_compound_l438_43866

/-- The mass of a compound given its molecular weight and number of moles. -/
def mass (molecular_weight : ℝ) (moles : ℝ) : ℝ :=
  molecular_weight * moles

/-- Theorem: The mass of 7 moles of a compound with a molecular weight of 588 g/mol is 4116 g. -/
theorem mass_of_compound : mass 588 7 = 4116 := by
  sorry

end mass_of_compound_l438_43866


namespace kimberly_skittles_l438_43805

def skittles_problem (initial_skittles : ℚ) 
                     (eaten_skittles : ℚ) 
                     (given_skittles : ℚ) 
                     (promotion_skittles : ℚ) 
                     (exchange_skittles : ℚ) : Prop :=
  let remaining_after_eating := initial_skittles - eaten_skittles
  let remaining_after_giving := remaining_after_eating - given_skittles
  let after_promotion := remaining_after_giving + promotion_skittles
  let final_skittles := after_promotion + exchange_skittles
  final_skittles = 18

theorem kimberly_skittles : 
  skittles_problem 7.5 2.25 1.5 3.75 10.5 := by
  sorry

end kimberly_skittles_l438_43805


namespace minimum_framing_for_specific_photo_l438_43808

/-- Calculates the minimum framing needed for a scaled photograph with a border -/
def minimum_framing (original_width original_height scale_factor border_width : ℕ) : ℕ :=
  let scaled_width := original_width * scale_factor
  let scaled_height := original_height * scale_factor
  let total_width := scaled_width + 2 * border_width
  let total_height := scaled_height + 2 * border_width
  let perimeter := 2 * (total_width + total_height)
  (perimeter + 11) / 12  -- Round up to the nearest foot

theorem minimum_framing_for_specific_photo :
  minimum_framing 5 7 5 3 = 12 := by
  sorry

end minimum_framing_for_specific_photo_l438_43808


namespace power_of_five_sum_equality_l438_43844

theorem power_of_five_sum_equality (x : ℕ) : 5^6 + 5^6 + 5^6 = 5^x ↔ x = 6 := by
  sorry

end power_of_five_sum_equality_l438_43844


namespace books_left_l438_43886

/-- Given that Paul had 242 books initially and sold 137 books, prove that he has 105 books left. -/
theorem books_left (initial_books : ℕ) (sold_books : ℕ) (h1 : initial_books = 242) (h2 : sold_books = 137) :
  initial_books - sold_books = 105 := by
  sorry

end books_left_l438_43886


namespace not_proper_subset_of_itself_l438_43892

def main_set : Set ℕ := {1, 2, 3}

theorem not_proper_subset_of_itself : ¬(main_set ⊂ main_set) := by
  sorry

end not_proper_subset_of_itself_l438_43892


namespace line_perpendicular_to_plane_l438_43818

/-- Two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def is_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two lines are different -/
def are_different (l1 l2 : Line) : Prop := sorry

theorem line_perpendicular_to_plane 
  (m n : Line) (β : Plane) 
  (h1 : are_different m n)
  (h2 : are_parallel m n) 
  (h3 : is_perpendicular_to_plane n β) : 
  is_perpendicular_to_plane m β := by sorry

end line_perpendicular_to_plane_l438_43818


namespace farmers_additional_cost_l438_43893

/-- The additional cost for Farmer Brown's new hay requirements -/
def additional_cost (original_bales : ℕ) (original_price : ℕ) (new_bales : ℕ) (new_price : ℕ) : ℕ :=
  new_bales * new_price - original_bales * original_price

/-- Theorem: The additional cost for Farmer Brown's new requirements is $210 -/
theorem farmers_additional_cost :
  additional_cost 10 15 20 18 = 210 := by
  sorry

end farmers_additional_cost_l438_43893


namespace a_n_properties_l438_43881

def a_n (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2^n + 1 else 2^n - 1

theorem a_n_properties : ∀ n : ℕ,
  (∃ m : ℕ, if n % 2 = 0 then a_n n = 5 * m^2 else a_n n = m^2) :=
by sorry

end a_n_properties_l438_43881


namespace mn_equation_solutions_l438_43811

theorem mn_equation_solutions (m n : ℤ) : 
  m^2 * n^2 + m^2 + n^2 + 10*m*n + 16 = 0 ↔ (m = 2 ∧ n = -2) ∨ (m = -2 ∧ n = 2) :=
by sorry

end mn_equation_solutions_l438_43811


namespace number_division_remainder_l438_43826

theorem number_division_remainder (N : ℤ) (D : ℕ) 
  (h1 : N % 125 = 40) 
  (h2 : N % D = 11) : 
  D = 29 := by
sorry

end number_division_remainder_l438_43826


namespace count_perfect_square_factors_l438_43883

/-- The number of perfect square factors of 360 -/
def perfectSquareFactors : ℕ := 4

/-- The prime factorization of 360 -/
def primeFactorization : List (ℕ × ℕ) := [(2, 3), (3, 2), (5, 1)]

/-- Theorem stating that the number of perfect square factors of 360 is 4 -/
theorem count_perfect_square_factors :
  (List.sum (List.map (fun (p : ℕ × ℕ) => (p.2 / 2 + 1)) primeFactorization)) = perfectSquareFactors := by
  sorry

end count_perfect_square_factors_l438_43883


namespace gnome_ratio_l438_43894

/-- Represents the properties of garden gnomes -/
structure GnomeProperties where
  total : Nat
  bigNoses : Nat
  blueHatsBigNoses : Nat
  redHatsSmallNoses : Nat

/-- Theorem: The ratio of gnomes with red hats to total gnomes is 3:4 -/
theorem gnome_ratio (g : GnomeProperties) 
  (h1 : g.total = 28)
  (h2 : g.bigNoses = g.total / 2)
  (h3 : g.blueHatsBigNoses = 6)
  (h4 : g.redHatsSmallNoses = 13) :
  (g.redHatsSmallNoses + (g.bigNoses - g.blueHatsBigNoses)) * 4 = g.total * 3 := by
  sorry

#check gnome_ratio

end gnome_ratio_l438_43894


namespace prob_six_heads_and_return_l438_43863

/-- The number of nodes in the circular arrangement -/
def num_nodes : ℕ := 5

/-- The total number of coin flips -/
def num_flips : ℕ := 12

/-- The number of heads we're interested in -/
def target_heads : ℕ := 6

/-- Represents the movement on the circular arrangement -/
def net_movement (heads : ℕ) : ℤ :=
  (heads : ℤ) - (num_flips - heads : ℤ)

/-- The condition for returning to the starting node -/
def returns_to_start (heads : ℕ) : Prop :=
  net_movement heads % (num_nodes : ℤ) = 0

/-- The probability of flipping exactly 'heads' number of heads in 'num_flips' flips -/
def prob_heads (heads : ℕ) : ℚ :=
  (Nat.choose num_flips heads : ℚ) / 2^num_flips

/-- The main theorem to prove -/
theorem prob_six_heads_and_return :
  returns_to_start target_heads ∧ prob_heads target_heads = 231 / 1024 := by
  sorry


end prob_six_heads_and_return_l438_43863


namespace average_people_moving_per_hour_l438_43839

/-- The number of people moving to Texas in 5 days -/
def people_moving : ℕ := 3500

/-- The number of days -/
def days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the average number of people moving per hour -/
def average_per_hour : ℚ :=
  people_moving / (days * hours_per_day)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem average_people_moving_per_hour :
  round_to_nearest average_per_hour = 29 := by
  sorry

end average_people_moving_per_hour_l438_43839


namespace alcohol_dilution_l438_43851

/-- Given an initial solution and added water, calculate the new alcohol percentage -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_percentage : ℝ) (added_water : ℝ) :
  initial_volume = 15 →
  initial_percentage = 26 →
  added_water = 5 →
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let total_volume := initial_volume + added_water
  let new_percentage := (initial_alcohol / total_volume) * 100
  new_percentage = 19.5 := by
  sorry

#check alcohol_dilution

end alcohol_dilution_l438_43851


namespace orthocenter_of_triangle_l438_43827

/-- The orthocenter of a triangle in 3D space -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC is (4/5, 38/5, 59/5) -/
theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (2, 4, 6)
  let B : ℝ × ℝ × ℝ := (6, 5, 3)
  let C : ℝ × ℝ × ℝ := (4, 6, 7)
  orthocenter A B C = (4/5, 38/5, 59/5) := by sorry

end orthocenter_of_triangle_l438_43827


namespace scientific_notation_of_56_99_million_l438_43824

def million : ℝ := 1000000

theorem scientific_notation_of_56_99_million :
  56.99 * million = 5.699 * (10 : ℝ) ^ 7 :=
sorry

end scientific_notation_of_56_99_million_l438_43824


namespace unique_k_satisfying_equation_l438_43833

theorem unique_k_satisfying_equation : ∃! k : ℕ, 10^k - 1 = 9*k^2 := by sorry

end unique_k_satisfying_equation_l438_43833


namespace min_floor_sum_l438_43884

theorem min_floor_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  34 ≤ ⌊(a^2 + b^2) / c⌋ + ⌊(b^2 + c^2) / a⌋ + ⌊(c^2 + a^2) / b⌋ :=
by sorry

end min_floor_sum_l438_43884


namespace optimal_prevention_plan_l438_43806

/-- Represents a preventive measure -/
structure PreventiveMeasure where
  cost : ℝ
  effectiveness : ℝ

/-- Calculates the total cost given a set of preventive measures -/
def totalCost (baseProbability : ℝ) (baseLoss : ℝ) (measures : List PreventiveMeasure) : ℝ :=
  let preventionCost := measures.foldl (fun acc m => acc + m.cost) 0
  let incidentProbability := measures.foldl (fun acc m => acc * (1 - m.effectiveness)) baseProbability
  preventionCost + incidentProbability * baseLoss

theorem optimal_prevention_plan 
  (baseProbability : ℝ)
  (baseLoss : ℝ)
  (measureA : PreventiveMeasure)
  (measureB : PreventiveMeasure)
  (h1 : baseProbability = 0.3)
  (h2 : baseLoss = 400)
  (h3 : measureA.cost = 45)
  (h4 : measureA.effectiveness = 0.9)
  (h5 : measureB.cost = 30)
  (h6 : measureB.effectiveness = 0.85) :
  totalCost baseProbability baseLoss [measureA, measureB] < 
  min 
    (totalCost baseProbability baseLoss [])
    (min 
      (totalCost baseProbability baseLoss [measureA])
      (totalCost baseProbability baseLoss [measureB])) := by
  sorry

#check optimal_prevention_plan

end optimal_prevention_plan_l438_43806


namespace vector_magnitude_l438_43813

theorem vector_magnitude (a b : ℝ × ℝ) : 
  let angle := π / 6
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  dot_product = 3 ∧ 
  magnitude_a = 3 →
  Real.sqrt (b.1^2 + b.2^2) = (2 * Real.sqrt 3) / 3 :=
by sorry

end vector_magnitude_l438_43813


namespace F_zeros_and_reciprocal_sum_l438_43809

noncomputable def F (x : ℝ) : ℝ := 1 / (2 * x) + Real.log (x / 2)

theorem F_zeros_and_reciprocal_sum :
  (∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ F x₁ = 0 ∧ F x₂ = 0 ∧
    (∀ (x : ℝ), x > 0 ∧ F x = 0 → x = x₁ ∨ x = x₂)) ∧
  (∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ F x₁ = 0 ∧ F x₂ = 0 →
    1 / x₁ + 1 / x₂ > 4) :=
by sorry

end F_zeros_and_reciprocal_sum_l438_43809


namespace smallest_n_for_factorial_sum_l438_43848

def lastFourDigits (n : ℕ) : ℕ := n % 10000

def isValidSequence (seq : List ℕ) : Prop :=
  ∀ x ∈ seq, x ≤ 15 ∧ x > 0

theorem smallest_n_for_factorial_sum : 
  (∃ (seq : List ℕ), 
    seq.length = 3 ∧ 
    isValidSequence seq ∧ 
    lastFourDigits (seq.map Nat.factorial).sum = 2001) ∧ 
  (∀ (n : ℕ) (seq : List ℕ), 
    n < 3 → 
    seq.length = n → 
    isValidSequence seq → 
    lastFourDigits (seq.map Nat.factorial).sum ≠ 2001) :=
sorry

end smallest_n_for_factorial_sum_l438_43848


namespace inequality_solution_set_l438_43850

theorem inequality_solution_set (x : ℝ) : 
  (3 * x - 5 > 11 - 2 * x) ↔ (x > 16 / 5) := by
  sorry

end inequality_solution_set_l438_43850


namespace concrete_pillars_amount_l438_43836

/-- Calculates the concrete needed for supporting pillars with environmental factors --/
def concrete_for_pillars (total_concrete : ℝ) (roadway_concrete : ℝ) (anchor_concrete : ℝ) (env_factor : ℝ) : ℝ :=
  let total_anchor_concrete := 2 * anchor_concrete
  let initial_pillar_concrete := total_concrete - roadway_concrete - total_anchor_concrete
  let pillar_increase := initial_pillar_concrete * env_factor
  initial_pillar_concrete + pillar_increase

/-- Theorem stating the amount of concrete needed for supporting pillars --/
theorem concrete_pillars_amount : 
  concrete_for_pillars 4800 1600 700 0.05 = 1890 := by
  sorry

end concrete_pillars_amount_l438_43836


namespace north_pond_duck_count_l438_43859

/-- The number of ducks in Lake Michigan -/
def lake_michigan_ducks : ℕ := 100

/-- The number of ducks in North Pond -/
def north_pond_ducks : ℕ := 2 * lake_michigan_ducks + 6

/-- Theorem stating that North Pond has 206 ducks -/
theorem north_pond_duck_count : north_pond_ducks = 206 := by
  sorry

end north_pond_duck_count_l438_43859


namespace log_5_125_l438_43828

-- Define the logarithm function
noncomputable def log (a : ℝ) (N : ℝ) : ℝ :=
  Real.log N / Real.log a

-- Theorem statement
theorem log_5_125 : log 5 125 = 3 := by
  sorry


end log_5_125_l438_43828


namespace max_digit_sum_2016_l438_43812

/-- A function that sums the digits of a natural number -/
def sumDigits (n : ℕ) : ℕ := sorry

/-- A function that repeatedly sums the digits until a single digit is obtained -/
def repeatSumDigits (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number has exactly 2016 digits -/
def has2016Digits (n : ℕ) : Prop := sorry

theorem max_digit_sum_2016 :
  ∀ n : ℕ, has2016Digits n → repeatSumDigits n ≤ 9 ∧ ∃ m : ℕ, has2016Digits m ∧ repeatSumDigits m = 9 := by
  sorry

end max_digit_sum_2016_l438_43812


namespace golf_balls_needed_l438_43820

def weekend_goal : ℕ := 48
def saturday_balls : ℕ := 16
def sunday_balls : ℕ := 18

theorem golf_balls_needed : weekend_goal - (saturday_balls + sunday_balls) = 14 := by
  sorry

end golf_balls_needed_l438_43820


namespace tomato_price_proof_l438_43869

/-- The original price per pound of tomatoes -/
def original_price : ℝ := 0.80

/-- The proportion of tomatoes remaining after discarding ruined ones -/
def remaining_proportion : ℝ := 0.90

/-- The profit percentage as a decimal -/
def profit_percentage : ℝ := 0.12

/-- The selling price per pound of remaining tomatoes -/
def selling_price : ℝ := 0.9956

theorem tomato_price_proof :
  selling_price * remaining_proportion = original_price * (1 + profit_percentage) := by
  sorry


end tomato_price_proof_l438_43869


namespace system_solution_ratio_l438_43840

theorem system_solution_ratio (x y c d : ℝ) : 
  (4 * x - 2 * y = c) →
  (5 * y - 10 * x = d) →
  d ≠ 0 →
  c / d = 0 := by
sorry

end system_solution_ratio_l438_43840


namespace choir_average_age_l438_43816

theorem choir_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℚ) 
  (avg_age_males : ℚ) 
  (total_people : ℕ) 
  (h1 : num_females = 10) 
  (h2 : num_males = 15) 
  (h3 : avg_age_females = 30) 
  (h4 : avg_age_males = 35) 
  (h5 : total_people = num_females + num_males) :
  (num_females * avg_age_females + num_males * avg_age_males) / total_people = 33 := by
  sorry

end choir_average_age_l438_43816


namespace bridget_apples_l438_43837

theorem bridget_apples (x : ℕ) : 
  (x : ℚ) / 3 + 5 + 4 + 4 = x → x = 22 :=
by sorry

end bridget_apples_l438_43837


namespace smallest_prime_with_digit_sum_22_l438_43867

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

theorem smallest_prime_with_digit_sum_22 :
  ∃ (p : ℕ), is_prime p ∧ digit_sum p = 22 ∧
  ∀ (q : ℕ), is_prime q → digit_sum q = 22 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_22_l438_43867


namespace quadratic_roots_sum_of_squares_l438_43807

theorem quadratic_roots_sum_of_squares (p q : ℝ) : 
  p^2 - 5*p + 6 = 0 → q^2 - 5*q + 6 = 0 → p^2 + q^2 = 13 := by
sorry

end quadratic_roots_sum_of_squares_l438_43807


namespace quadratic_factorization_l438_43874

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end quadratic_factorization_l438_43874


namespace negative_subtraction_l438_43845

theorem negative_subtraction (a b : ℤ) : -5 - (-2) = -3 := by
  sorry

end negative_subtraction_l438_43845


namespace gcd_228_1995_l438_43897

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l438_43897


namespace min_value_of_squares_l438_43857

theorem min_value_of_squares (a b c d : ℝ) (h1 : a * b = 3) (h2 : c + 3 * d = 0) :
  (a - c)^2 + (b - d)^2 ≥ 18/5 := by
  sorry

end min_value_of_squares_l438_43857


namespace expression_evaluation_l438_43895

theorem expression_evaluation (b x : ℝ) (h : x = b + 9) :
  2*x - b + 5 = b + 23 := by sorry

end expression_evaluation_l438_43895


namespace cafeteria_pies_l438_43858

/-- Given a cafeteria with initial apples, apples handed out, and apples needed per pie,
    calculate the number of pies that can be made. -/
def calculate_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

/-- Theorem stating that with 62 initial apples, 8 apples handed out, and 9 apples per pie,
    the cafeteria can make 6 pies. -/
theorem cafeteria_pies :
  calculate_pies 62 8 9 = 6 := by
  sorry

end cafeteria_pies_l438_43858


namespace smallest_solution_l438_43862

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The equation in the problem -/
def equation (x : ℝ) : Prop :=
  floor (x^2) - (floor x)^2 = 17

/-- Theorem stating that 7√2 is the smallest solution -/
theorem smallest_solution :
  ∀ x : ℝ, equation x → x ≥ 7 * Real.sqrt 2 :=
sorry

end smallest_solution_l438_43862


namespace sin_equality_theorem_l438_43835

theorem sin_equality_theorem (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * π / 180) = Real.sin (720 * π / 180)) ↔ (n = 0 ∨ n = 180) := by
sorry

end sin_equality_theorem_l438_43835


namespace snail_path_count_l438_43898

theorem snail_path_count (n : ℕ) : 
  (number_of_paths : ℕ) = (Nat.choose (2 * n) n) ^ 2 :=
by
  sorry

where
  number_of_paths : ℕ := 
    count_closed_paths_on_graph_paper (2 * n)

  count_closed_paths_on_graph_paper (steps : ℕ) : ℕ := 
    -- Returns the number of distinct paths on graph paper
    -- that start and end at the same vertex
    -- and have a total length of 'steps'
    sorry

end snail_path_count_l438_43898


namespace inequalities_theorem_l438_43853

theorem inequalities_theorem (a b c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : c < 0) :
  (c / a > c / b) ∧ ((a - c)^c < (b - c)^c) ∧ (b * Real.exp a > a * Real.exp b) := by
  sorry

end inequalities_theorem_l438_43853


namespace unique_phone_number_l438_43838

def is_valid_phone_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

def first_upgrade (n : ℕ) : ℕ :=
  let d := n.div 100000
  let r := n.mod 100000
  d * 1000000 + 8 * 100000 + r

def second_upgrade (n : ℕ) : ℕ :=
  2000000000 + n

theorem unique_phone_number :
  ∃! n : ℕ, is_valid_phone_number n ∧ 
    second_upgrade (first_upgrade n) = 81 * n :=
by
  sorry

end unique_phone_number_l438_43838


namespace largest_prime_factor_of_factorial_sum_l438_43865

theorem largest_prime_factor_of_factorial_sum : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (Nat.factorial 7 + Nat.factorial 8) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 7 + Nat.factorial 8) → q ≤ p :=
by
  -- The proof goes here
  sorry

end largest_prime_factor_of_factorial_sum_l438_43865


namespace q_is_false_l438_43870

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q :=
sorry

end q_is_false_l438_43870


namespace father_daughter_ages_l438_43823

theorem father_daughter_ages (father_age daughter_age : ℕ) : 
  father_age = 4 * daughter_age ∧ father_age = daughter_age + 30 →
  father_age = 40 ∧ daughter_age = 10 := by
  sorry

end father_daughter_ages_l438_43823


namespace square_product_theorem_l438_43849

class FiniteSquareRing (R : Type) extends Ring R where
  finite : Finite R
  square_sum_is_square : ∀ a b : R, ∃ c : R, a ^ 2 + b ^ 2 = c ^ 2

theorem square_product_theorem {R : Type} [FiniteSquareRing R] :
  ∀ a b c : R, ∃ d : R, 2 * a * b * c = d ^ 2 := by
  sorry

end square_product_theorem_l438_43849


namespace share_sale_value_l438_43832

/-- The value of the business in rs -/
def business_value : ℚ := 10000

/-- The fraction of the business owned by the man -/
def man_ownership : ℚ := 1/3

/-- The fraction of the man's shares that he sells -/
def sold_fraction : ℚ := 3/5

/-- The amount the man receives for selling his shares -/
def sold_amount : ℚ := 2000

theorem share_sale_value :
  sold_fraction * man_ownership * business_value = sold_amount := by
  sorry

end share_sale_value_l438_43832


namespace seating_arrangements_l438_43896

def n : ℕ := 8

def numArrangements : ℕ := n.factorial - (n-1).factorial * 2

theorem seating_arrangements (n : ℕ) (h : n = 8) : 
  numArrangements = 30240 := by
  sorry

end seating_arrangements_l438_43896


namespace puppies_brought_in_puppies_brought_in_solution_l438_43873

theorem puppies_brought_in (initial_puppies : ℕ) (adoption_rate : ℕ) (adoption_days : ℕ) : ℕ :=
  let total_adopted := adoption_rate * adoption_days
  total_adopted - initial_puppies

theorem puppies_brought_in_solution :
  puppies_brought_in 2 4 9 = 34 := by
  sorry

end puppies_brought_in_puppies_brought_in_solution_l438_43873


namespace unpainted_cubes_count_l438_43875

/-- Represents a cube with painted strips on its faces -/
structure PaintedCube where
  size : Nat
  totalUnitCubes : Nat
  stripsPerFace : Nat
  stripWidth : Nat
  stripLength : Nat

/-- Calculates the number of unpainted unit cubes in a painted cube -/
def unpaintedCubes (cube : PaintedCube) : Nat :=
  cube.totalUnitCubes - paintedCubes cube
where
  /-- Helper function to calculate the number of painted unit cubes -/
  paintedCubes (cube : PaintedCube) : Nat :=
    let totalPainted := 6 * cube.stripsPerFace * cube.stripLength
    let edgeOverlaps := 12 * cube.stripWidth / 2
    let cornerOverlaps := 8
    totalPainted - edgeOverlaps - cornerOverlaps

/-- Theorem stating that a 6x6x6 cube with specific painted strips has 170 unpainted unit cubes -/
theorem unpainted_cubes_count :
  let cube : PaintedCube := {
    size := 6,
    totalUnitCubes := 216,
    stripsPerFace := 2,
    stripWidth := 1,
    stripLength := 6
  }
  unpaintedCubes cube = 170 := by
  sorry


end unpainted_cubes_count_l438_43875


namespace sqrt_equation_solution_l438_43822

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (8 + 3 * z) = 12 :=
by
  -- The proof goes here
  sorry

end sqrt_equation_solution_l438_43822


namespace possible_values_of_p_l438_43889

theorem possible_values_of_p (a b c p : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_eq : a + 1/b = p ∧ b + 1/c = p ∧ c + 1/a = p) :
  p = 1 ∨ p = -1 := by
  sorry

end possible_values_of_p_l438_43889


namespace water_distribution_l438_43817

/-- A water distribution problem for four neighborhoods. -/
theorem water_distribution (total : ℕ) (left_for_fourth : ℕ) : 
  total = 1200 → 
  left_for_fourth = 350 → 
  ∃ (first second third fourth : ℕ),
    first + second + third + fourth = total ∧
    second = 2 * first ∧
    third = second + 100 ∧
    fourth = left_for_fourth ∧
    first = 150 := by
  sorry

end water_distribution_l438_43817


namespace photo_arrangement_probability_l438_43815

/-- The number of boys -/
def num_boys : ℕ := 2

/-- The number of girls -/
def num_girls : ℕ := 5

/-- The total number of people -/
def total_people : ℕ := num_boys + num_girls

/-- The number of girls between the boys -/
def girls_between : ℕ := 3

/-- The probability of the specific arrangement -/
def probability : ℚ := 1 / 7

theorem photo_arrangement_probability :
  (num_boys = 2) →
  (num_girls = 5) →
  (girls_between = 3) →
  (probability = 1 / 7) :=
by sorry

end photo_arrangement_probability_l438_43815


namespace sum_of_fractions_inequality_l438_43830

theorem sum_of_fractions_inequality (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (a^4 + b^4) / (a^6 + b^6) + (b^4 + c^4) / (b^6 + c^6) + (c^4 + a^4) / (c^6 + a^6) ≤ 1 / (a * b * c) := by
  sorry

end sum_of_fractions_inequality_l438_43830


namespace circumscribed_square_area_l438_43852

/-- Given a circle with an inscribed square of perimeter p, 
    the area of the square that circumscribes the circle is p²/8 -/
theorem circumscribed_square_area (p : ℝ) (p_pos : p > 0) : 
  let inscribed_square_perimeter := p
  let circumscribed_square_area := p^2 / 8
  inscribed_square_perimeter = p → circumscribed_square_area = p^2 / 8 := by
sorry

end circumscribed_square_area_l438_43852


namespace smallest_four_digit_in_pascals_triangle_l438_43803

def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (k m : ℕ), n = Nat.choose m k

theorem smallest_four_digit_in_pascals_triangle :
  (∀ n : ℕ, n < 1000 → ¬(is_in_pascals_triangle n ∧ n ≥ 1000)) ∧
  is_in_pascals_triangle 1000 :=
sorry

end smallest_four_digit_in_pascals_triangle_l438_43803


namespace root_implies_k_value_l438_43882

theorem root_implies_k_value (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 9 * x + 8 = 0 ∧ x = 1) → k = 1 := by
  sorry

end root_implies_k_value_l438_43882


namespace history_homework_time_l438_43872

/-- Represents the time in minutes for each homework subject and the total available time. -/
structure HomeworkTime where
  total : Nat
  math : Nat
  english : Nat
  science : Nat
  special_project : Nat

/-- Calculates the time remaining for history homework given the times for other subjects. -/
def history_time (hw : HomeworkTime) : Nat :=
  hw.total - (hw.math + hw.english + hw.science + hw.special_project)

/-- Proves that given the specified homework times, the remaining time for history is 25 minutes. -/
theorem history_homework_time :
  let hw : HomeworkTime := {
    total := 180,  -- 3 hours in minutes
    math := 45,
    english := 30,
    science := 50,
    special_project := 30
  }
  history_time hw = 25 := by sorry

end history_homework_time_l438_43872


namespace max_gumdrops_l438_43804

/-- Represents the candy purchasing problem with given constraints --/
def CandyProblem (total_budget : ℕ) (bulk_cost gummy_cost gumdrop_cost : ℕ) 
                 (min_bulk min_gummy : ℕ) : Prop :=
  let remaining_budget := total_budget - (min_bulk * bulk_cost + min_gummy * gummy_cost)
  remaining_budget / gumdrop_cost = 28

/-- Theorem stating the maximum number of gumdrops that can be purchased --/
theorem max_gumdrops : 
  CandyProblem 224 8 6 4 10 5 := by
  sorry

#check max_gumdrops

end max_gumdrops_l438_43804


namespace smallest_odd_probability_l438_43841

/-- The probability that the smallest number in a lottery draw is odd -/
theorem smallest_odd_probability (n : ℕ) (k : ℕ) (h1 : n = 90) (h2 : k = 5) :
  let prob := (1 : ℚ) / 2 + (44 : ℚ) * (Nat.choose 45 3 : ℚ) / (2 * (Nat.choose n k : ℚ))
  ∃ (ε : ℚ), abs (prob - 0.5142) < ε ∧ ε > 0 := by
  sorry

end smallest_odd_probability_l438_43841


namespace total_supplies_is_1260_l438_43855

/-- The total number of supplies given the number of rows and items per row -/
def total_supplies (rows : ℕ) (crayons_per_row : ℕ) (colored_pencils_per_row : ℕ) (graphite_pencils_per_row : ℕ) : ℕ :=
  rows * (crayons_per_row + colored_pencils_per_row + graphite_pencils_per_row)

/-- Theorem stating that the total number of supplies is 1260 -/
theorem total_supplies_is_1260 : total_supplies 28 12 15 18 = 1260 := by
  sorry

end total_supplies_is_1260_l438_43855


namespace system_solution_l438_43801

theorem system_solution : ∃! (x y : ℝ), x - y = -5 ∧ 3 * x + 2 * y = 10 := by
  sorry

end system_solution_l438_43801


namespace married_men_fraction_l438_43891

theorem married_men_fraction (total_women : ℕ) (single_women : ℕ) :
  single_women = (3 : ℕ) * total_women / 7 →
  (total_women - single_women) / (total_women + (total_women - single_women)) = 4 / 11 := by
  sorry

end married_men_fraction_l438_43891


namespace fraction_simplification_l438_43880

theorem fraction_simplification (a x : ℝ) (ha : a > 0) (hx : x > 0) :
  (a * Real.sqrt x - x * Real.sqrt a) / (Real.sqrt a - Real.sqrt x) = Real.sqrt (a * x) := by
  sorry

end fraction_simplification_l438_43880


namespace negative_four_star_two_simplify_a_minus_b_cubed_specific_values_l438_43860

-- Define the * operation
def star (x y : ℚ) : ℚ := x^2 - 3*y + 3

-- Theorem 1
theorem negative_four_star_two : star (-4) 2 = 13 := by sorry

-- Theorem 2
theorem simplify_a_minus_b_cubed (a b : ℚ) : 
  star (a - b) ((a - b)^2) = -2*a^2 - 2*b^2 + 4*a*b + 3 := by sorry

-- Theorem 3
theorem specific_values : 
  star (-2 - (1/2)) ((-2 - (1/2))^2) = -13/2 := by sorry

end negative_four_star_two_simplify_a_minus_b_cubed_specific_values_l438_43860


namespace hardcover_count_l438_43890

/-- Represents the purchase of a book series -/
structure BookPurchase where
  total_volumes : ℕ
  paperback_price : ℕ
  hardcover_price : ℕ
  total_cost : ℕ

/-- Theorem stating that under given conditions, the number of hardcover books is 6 -/
theorem hardcover_count (purchase : BookPurchase)
  (h_total : purchase.total_volumes = 8)
  (h_paperback : purchase.paperback_price = 10)
  (h_hardcover : purchase.hardcover_price = 20)
  (h_cost : purchase.total_cost = 140) :
  ∃ (h : ℕ), h = 6 ∧ 
    h * purchase.hardcover_price + (purchase.total_volumes - h) * purchase.paperback_price = purchase.total_cost :=
by sorry

end hardcover_count_l438_43890


namespace painted_faces_count_l438_43814

/-- Represents a cube with a given side length -/
structure Cube :=
  (side_length : ℕ)

/-- Represents a painted cube with three adjacent painted faces -/
structure PaintedCube extends Cube :=
  (painted_faces : Fin 3)

/-- Counts the number of unit cubes with at least two painted faces when a painted cube is cut into unit cubes -/
def count_multi_painted_faces (c : PaintedCube) : ℕ :=
  sorry

/-- Theorem stating that a 4x4x4 cube painted on three adjacent faces, when cut into unit cubes, has 14 cubes with at least two painted faces -/
theorem painted_faces_count (c : PaintedCube) (h : c.side_length = 4) : 
  count_multi_painted_faces c = 14 :=
sorry

end painted_faces_count_l438_43814


namespace james_semesters_paid_l438_43878

/-- Calculates the number of semesters paid for given the units per semester, cost per unit, and total cost. -/
def semesters_paid (units_per_semester : ℕ) (cost_per_unit : ℕ) (total_cost : ℕ) : ℕ :=
  total_cost / (units_per_semester * cost_per_unit)

/-- Proves that given 20 units per semester, $50 per unit, and $2000 total cost, the number of semesters paid for is 2. -/
theorem james_semesters_paid :
  semesters_paid 20 50 2000 = 2 := by
  sorry

end james_semesters_paid_l438_43878
