import Mathlib

namespace NUMINAMATH_CALUDE_rulers_produced_l3169_316938

theorem rulers_produced (rulers_per_minute : ℕ) (minutes : ℕ) : 
  rulers_per_minute = 8 → minutes = 15 → rulers_per_minute * minutes = 120 := by
  sorry

end NUMINAMATH_CALUDE_rulers_produced_l3169_316938


namespace NUMINAMATH_CALUDE_contrapositive_prop2_true_l3169_316975

-- Proposition 1
axiom prop1 : ∀ a b : ℝ, a > b → (1 / a) < (1 / b)

-- Proposition 2
axiom prop2 : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → (x + 2) * (x - 3) ≤ 0

-- Theorem: The contrapositive of Proposition 2 is true
theorem contrapositive_prop2_true :
  ∀ x : ℝ, (x + 2) * (x - 3) > 0 → (x < -2 ∨ x > 0) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_prop2_true_l3169_316975


namespace NUMINAMATH_CALUDE_sum_positive_when_difference_exceeds_absolute_value_l3169_316940

theorem sum_positive_when_difference_exceeds_absolute_value
  (a b : ℝ) (h : a - |b| > 0) : b + a > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_when_difference_exceeds_absolute_value_l3169_316940


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l3169_316988

theorem geometric_arithmetic_sequence :
  ∀ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 →
  (y^2 = x*z) →                   -- geometric progression
  (2*y = x + z - 16) →            -- arithmetic progression after subtracting 16 from z
  ((y-2)^2 = x*(z-16)) →          -- geometric progression after subtracting 2 from y
  ((x = 1 ∧ y = 5 ∧ z = 25) ∨ (x = 1/9 ∧ y = 13/9 ∧ z = 169/9)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l3169_316988


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l3169_316939

theorem binomial_coefficient_two (n : ℕ) (h : n > 0) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l3169_316939


namespace NUMINAMATH_CALUDE_cookie_difference_l3169_316970

theorem cookie_difference (initial_sweet initial_salty sweet_eaten salty_eaten : ℕ) :
  initial_sweet = 39 →
  initial_salty = 6 →
  sweet_eaten = 32 →
  salty_eaten = 23 →
  sweet_eaten - salty_eaten = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l3169_316970


namespace NUMINAMATH_CALUDE_odd_function_symmetry_l3169_316914

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the property of being increasing on an interval
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem odd_function_symmetry :
  is_odd f →
  is_increasing_on f 3 7 →
  f 4 = 5 →
  is_increasing_on f (-7) (-3) ∧ f (-4) = -5 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_symmetry_l3169_316914


namespace NUMINAMATH_CALUDE_profit_increase_march_to_june_l3169_316954

/-- Calculates the total percent increase in profits from March to June given monthly changes -/
theorem profit_increase_march_to_june 
  (march_profit : ℝ) 
  (april_increase : ℝ) 
  (may_decrease : ℝ) 
  (june_increase : ℝ) 
  (h1 : april_increase = 0.4) 
  (h2 : may_decrease = 0.2) 
  (h3 : june_increase = 0.5) : 
  (((1 + june_increase) * (1 - may_decrease) * (1 + april_increase) - 1) * 100 = 68) := by
sorry

end NUMINAMATH_CALUDE_profit_increase_march_to_june_l3169_316954


namespace NUMINAMATH_CALUDE_terminal_side_first_quadrant_l3169_316981

-- Define the angle in degrees
def angle : ℝ := -330

-- Define the quadrants
inductive Quadrant
  | first
  | second
  | third
  | fourth

-- Define a function to determine the quadrant of an angle
def angle_quadrant (θ : ℝ) : Quadrant :=
  sorry

-- Theorem statement
theorem terminal_side_first_quadrant :
  angle_quadrant angle = Quadrant.first :=
sorry

end NUMINAMATH_CALUDE_terminal_side_first_quadrant_l3169_316981


namespace NUMINAMATH_CALUDE_transformed_area_l3169_316994

-- Define the transformation matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 8, -2]

-- Define the original region T and its area
def area_T : ℝ := 12

-- Define the transformed region T' and its area
def area_T' : ℝ := |A.det| * area_T

-- Theorem statement
theorem transformed_area :
  area_T' = 456 :=
sorry

end NUMINAMATH_CALUDE_transformed_area_l3169_316994


namespace NUMINAMATH_CALUDE_files_deleted_l3169_316909

theorem files_deleted (initial_apps : ℕ) (initial_files : ℕ) (final_apps : ℕ) (final_files : ℕ) 
  (h1 : initial_apps = 17)
  (h2 : initial_files = 21)
  (h3 : final_apps = 3)
  (h4 : final_files = 7) :
  initial_files - final_files = 14 := by
  sorry

end NUMINAMATH_CALUDE_files_deleted_l3169_316909


namespace NUMINAMATH_CALUDE_y_sum_equals_4360_l3169_316980

/-- Given real numbers y₁ to y₈ satisfying four equations, 
    prove that a specific linear combination of these numbers equals 4360 -/
theorem y_sum_equals_4360 
  (y₁ y₂ y₃ y₄ y₅ y₆ y₇ y₈ : ℝ) 
  (eq1 : y₁ + 4*y₂ + 9*y₃ + 16*y₄ + 25*y₅ + 36*y₆ + 49*y₇ + 64*y₈ = 2)
  (eq2 : 4*y₁ + 9*y₂ + 16*y₃ + 25*y₄ + 36*y₅ + 49*y₆ + 64*y₇ + 81*y₈ = 15)
  (eq3 : 9*y₁ + 16*y₂ + 25*y₃ + 36*y₄ + 49*y₅ + 64*y₆ + 81*y₇ + 100*y₈ = 156)
  (eq4 : 16*y₁ + 25*y₂ + 36*y₃ + 49*y₄ + 64*y₅ + 81*y₆ + 100*y₇ + 121*y₈ = 1305) :
  25*y₁ + 36*y₂ + 49*y₃ + 64*y₄ + 81*y₅ + 100*y₆ + 121*y₇ + 144*y₈ = 4360 := by
  sorry


end NUMINAMATH_CALUDE_y_sum_equals_4360_l3169_316980


namespace NUMINAMATH_CALUDE_ball_probabilities_l3169_316935

/-- Represents a box of balls -/
structure Box where
  red : ℕ
  blue : ℕ

/-- The initial state of Box A -/
def box_a : Box := ⟨2, 4⟩

/-- The initial state of Box B -/
def box_b : Box := ⟨3, 3⟩

/-- The number of balls drawn from Box A -/
def balls_drawn : ℕ := 2

/-- Probability of drawing at least one blue ball from Box A -/
def prob_blue_from_a : ℚ := 14/15

/-- Probability of drawing a blue ball from Box B after transfer -/
def prob_blue_from_b : ℚ := 13/24

theorem ball_probabilities :
  (prob_blue_from_a = 14/15) ∧ (prob_blue_from_b = 13/24) := by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l3169_316935


namespace NUMINAMATH_CALUDE_typhoon_tree_problem_l3169_316933

theorem typhoon_tree_problem (initial_trees : ℕ) 
  (h1 : initial_trees = 13) 
  (dead_trees : ℕ) 
  (surviving_trees : ℕ) 
  (h2 : surviving_trees = dead_trees + 1) 
  (h3 : dead_trees + surviving_trees = initial_trees) : 
  dead_trees = 6 := by
sorry

end NUMINAMATH_CALUDE_typhoon_tree_problem_l3169_316933


namespace NUMINAMATH_CALUDE_suzy_book_count_l3169_316944

/-- Calculates the final number of books Suzy has after three days of transactions. -/
def final_book_count (initial : ℕ) (wed_out : ℕ) (thur_in : ℕ) (thur_out : ℕ) (fri_in : ℕ) : ℕ :=
  initial - wed_out + thur_in - thur_out + fri_in

/-- Theorem stating that given the specific transactions over three days, 
    Suzy ends up with 80 books. -/
theorem suzy_book_count : 
  final_book_count 98 43 23 5 7 = 80 := by
  sorry

end NUMINAMATH_CALUDE_suzy_book_count_l3169_316944


namespace NUMINAMATH_CALUDE_complex_on_line_l3169_316998

theorem complex_on_line (z : ℂ) (a : ℝ) :
  z = (2 + a * Complex.I) / (1 + Complex.I) →
  (z.re = -z.im) →
  a = 0 := by sorry

end NUMINAMATH_CALUDE_complex_on_line_l3169_316998


namespace NUMINAMATH_CALUDE_b_22_mod_35_l3169_316962

/-- Concatenates integers from 1 to n --/
def b (n : ℕ) : ℕ :=
  sorry

/-- The main theorem --/
theorem b_22_mod_35 : b 22 % 35 = 17 := by
  sorry

end NUMINAMATH_CALUDE_b_22_mod_35_l3169_316962


namespace NUMINAMATH_CALUDE_article_profit_percentage_l3169_316967

theorem article_profit_percentage (cost : ℝ) (reduced_sell : ℝ) (new_profit_percent : ℝ) :
  cost = 40 →
  reduced_sell = 8.40 →
  new_profit_percent = 30 →
  let new_cost := cost * 0.80
  let new_sell := new_cost * (1 + new_profit_percent / 100)
  let orig_sell := new_sell + reduced_sell
  let profit := orig_sell - cost
  let profit_percent := (profit / cost) * 100
  profit_percent = 25 := by
  sorry

end NUMINAMATH_CALUDE_article_profit_percentage_l3169_316967


namespace NUMINAMATH_CALUDE_mudits_age_l3169_316926

theorem mudits_age : ∃ (x : ℕ), x + 16 = 3 * (x - 4) ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_mudits_age_l3169_316926


namespace NUMINAMATH_CALUDE_profit_calculation_l3169_316948

/-- Given that the cost price of 55 articles equals the selling price of n articles,
    and the percent profit is 10.000000000000004%, prove that n equals 50. -/
theorem profit_calculation (C S : ℝ) (n : ℕ) 
    (h1 : 55 * C = n * S)
    (h2 : (S - C) / C * 100 = 10.000000000000004) :
    n = 50 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l3169_316948


namespace NUMINAMATH_CALUDE_sixth_angle_measure_l3169_316930

/-- The sum of interior angles of a hexagon is 720 degrees -/
def hexagon_angle_sum : ℝ := 720

/-- The given measures of five angles in the hexagon -/
def given_angles : List ℝ := [134, 108, 122, 99, 87]

/-- Theorem: In a hexagon where five of the interior angles measure 134°, 108°, 122°, 99°, and 87°, 
    the measure of the sixth angle is 170°. -/
theorem sixth_angle_measure :
  let sum_given_angles := given_angles.sum
  hexagon_angle_sum - sum_given_angles = 170 := by sorry

end NUMINAMATH_CALUDE_sixth_angle_measure_l3169_316930


namespace NUMINAMATH_CALUDE_pig_weight_problem_l3169_316951

theorem pig_weight_problem (x y : ℝ) (h1 : x - y = 72) (h2 : x + y = 348) : x = 210 := by
  sorry

end NUMINAMATH_CALUDE_pig_weight_problem_l3169_316951


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l3169_316995

theorem largest_prime_divisor (crayons paper : ℕ) 
  (h1 : crayons = 385) (h2 : paper = 95) : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ crayons ∧ p ∣ paper ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ crayons → q ∣ paper → q ≤ p := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l3169_316995


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l3169_316920

theorem real_part_of_complex_product : ∃ z : ℂ, z = (1 - Complex.I) * (2 + Complex.I) ∧ z.re = 3 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l3169_316920


namespace NUMINAMATH_CALUDE_watch_loss_percentage_l3169_316950

def watch_problem (selling_price_loss : ℝ) (selling_price_profit : ℝ) (profit_percentage : ℝ) : Prop :=
  let cost_price := selling_price_profit / (1 + profit_percentage / 100)
  let loss := cost_price - selling_price_loss
  let loss_percentage := (loss / cost_price) * 100
  selling_price_loss < cost_price ∧ 
  selling_price_profit > cost_price ∧
  loss_percentage = 5

theorem watch_loss_percentage : 
  watch_problem 1140 1260 5 := by sorry

end NUMINAMATH_CALUDE_watch_loss_percentage_l3169_316950


namespace NUMINAMATH_CALUDE_least_number_of_trees_least_number_of_trees_is_168_l3169_316937

theorem least_number_of_trees : ℕ → Prop :=
  fun n => (n % 4 = 0) ∧ 
           (n % 7 = 0) ∧ 
           (n % 6 = 0) ∧ 
           (n % 4 = 0) ∧ 
           (n ≥ 100) ∧ 
           (∀ m : ℕ, m < n → ¬(least_number_of_trees m))

theorem least_number_of_trees_is_168 : 
  least_number_of_trees 168 := by sorry

end NUMINAMATH_CALUDE_least_number_of_trees_least_number_of_trees_is_168_l3169_316937


namespace NUMINAMATH_CALUDE_irrational_plus_five_less_than_five_necessary_for_less_than_three_l3169_316993

-- Define the property of being irrational
def IsIrrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ p / q

-- Proposition ②
theorem irrational_plus_five (a : ℝ) : IsIrrational (a + 5) ↔ IsIrrational a := by sorry

-- Proposition ④
theorem less_than_five_necessary_for_less_than_three (a : ℝ) : a < 3 → a < 5 := by sorry

end NUMINAMATH_CALUDE_irrational_plus_five_less_than_five_necessary_for_less_than_three_l3169_316993


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l3169_316918

theorem cricket_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (h1 : total_players = 55)
  (h2 : throwers = 37)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0)
  : total_players - (total_players - throwers) / 3 = 49 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l3169_316918


namespace NUMINAMATH_CALUDE_shaded_triangle_probability_l3169_316974

theorem shaded_triangle_probability 
  (total_triangles : ℕ) 
  (shaded_triangles : ℕ) 
  (h1 : total_triangles > 4) 
  (h2 : total_triangles = 10) 
  (h3 : shaded_triangles = 4) : 
  (shaded_triangles : ℚ) / total_triangles = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_shaded_triangle_probability_l3169_316974


namespace NUMINAMATH_CALUDE_red_and_green_peaches_count_l3169_316965

/-- Given a basket of peaches, prove that the total number of red and green peaches is 22. -/
theorem red_and_green_peaches_count (red_peaches green_peaches : ℕ) 
  (h1 : red_peaches = 6)
  (h2 : green_peaches = 16) : 
  red_peaches + green_peaches = 22 := by
sorry

end NUMINAMATH_CALUDE_red_and_green_peaches_count_l3169_316965


namespace NUMINAMATH_CALUDE_potions_needed_for_owl_l3169_316968

/-- The number of Knuts in a Sickle -/
def knuts_per_sickle : ℕ := 23

/-- The number of Sickles in a Galleon -/
def sickles_per_galleon : ℕ := 17

/-- The cost of the owl in Galleons, Sickles, and Knuts -/
def owl_cost : ℕ × ℕ × ℕ := (2, 1, 5)

/-- The worth of each potion in Knuts -/
def potion_worth : ℕ := 9

/-- The function to calculate the total cost in Knuts -/
def total_cost_in_knuts (cost : ℕ × ℕ × ℕ) : ℕ :=
  cost.1 * sickles_per_galleon * knuts_per_sickle + 
  cost.2.1 * knuts_per_sickle + 
  cost.2.2

/-- The theorem stating the number of potions needed -/
theorem potions_needed_for_owl : 
  (total_cost_in_knuts owl_cost) / potion_worth = 90 := by
  sorry

end NUMINAMATH_CALUDE_potions_needed_for_owl_l3169_316968


namespace NUMINAMATH_CALUDE_radii_ratio_in_regular_hexagonal_pyramid_l3169_316934

/-- A regular hexagonal pyramid with circumscribed and inscribed spheres -/
structure RegularHexagonalPyramid where
  /-- Radius of the circumscribed sphere -/
  R : ℝ
  /-- Radius of the inscribed sphere -/
  r : ℝ
  /-- The center of the circumscribed sphere lies on the surface of the inscribed sphere -/
  center_on_surface : R = r * (1 + Real.sqrt 21 / 3)

/-- The ratio of the radii of the circumscribed sphere to the inscribed sphere
    in a regular hexagonal pyramid where the center of the circumscribed sphere
    lies on the surface of the inscribed sphere is (3 + √21) / 3 -/
theorem radii_ratio_in_regular_hexagonal_pyramid (p : RegularHexagonalPyramid) :
  p.R / p.r = (3 + Real.sqrt 21) / 3 := by
  sorry

end NUMINAMATH_CALUDE_radii_ratio_in_regular_hexagonal_pyramid_l3169_316934


namespace NUMINAMATH_CALUDE_ad_purchase_cost_is_108000_l3169_316952

/-- Represents the dimensions of an ad space -/
structure AdSpace where
  length : ℝ
  width : ℝ

/-- Represents the cost and quantity information for ad purchases -/
structure AdPurchase where
  numCompanies : ℕ
  adSpacesPerCompany : ℕ
  adSpace : AdSpace
  costPerSquareFoot : ℝ

/-- Calculates the total cost of ad purchases for multiple companies -/
def totalAdCost (purchase : AdPurchase) : ℝ :=
  purchase.numCompanies * purchase.adSpacesPerCompany * 
  purchase.adSpace.length * purchase.adSpace.width * 
  purchase.costPerSquareFoot

/-- Theorem stating that the total cost for the given ad purchase scenario is $108,000 -/
theorem ad_purchase_cost_is_108000 : 
  totalAdCost {
    numCompanies := 3,
    adSpacesPerCompany := 10,
    adSpace := { length := 12, width := 5 },
    costPerSquareFoot := 60
  } = 108000 := by
  sorry

end NUMINAMATH_CALUDE_ad_purchase_cost_is_108000_l3169_316952


namespace NUMINAMATH_CALUDE_fifteenth_prime_l3169_316949

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

theorem fifteenth_prime : nthPrime 15 = 47 := by sorry

end NUMINAMATH_CALUDE_fifteenth_prime_l3169_316949


namespace NUMINAMATH_CALUDE_smallest_double_square_triple_cube_l3169_316989

theorem smallest_double_square_triple_cube : ∃! k : ℕ, 
  (∃ m : ℕ, k = 2 * m^2) ∧ 
  (∃ n : ℕ, k = 3 * n^3) ∧ 
  (∀ j : ℕ, j < k → ¬(∃ x : ℕ, j = 2 * x^2) ∨ ¬(∃ y : ℕ, j = 3 * y^3)) ∧
  k = 648 := by
sorry

end NUMINAMATH_CALUDE_smallest_double_square_triple_cube_l3169_316989


namespace NUMINAMATH_CALUDE_food_expenditure_increase_l3169_316961

/-- Represents the annual income in thousand yuan -/
def annual_income : ℝ → ℝ := id

/-- Represents the annual food expenditure in thousand yuan -/
def annual_food_expenditure (x : ℝ) : ℝ := 2.5 * x + 3.2

/-- Theorem stating that when annual income increases by 1, 
    annual food expenditure increases by 2.5 -/
theorem food_expenditure_increase (x : ℝ) : 
  annual_food_expenditure (annual_income x + 1) - annual_food_expenditure (annual_income x) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_food_expenditure_increase_l3169_316961


namespace NUMINAMATH_CALUDE_complex_subtraction_l3169_316923

theorem complex_subtraction (z₁ z₂ : ℂ) (h₁ : z₁ = 3 + 4*I) (h₂ : z₂ = 1 + I) : 
  z₁ - z₂ = 2 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l3169_316923


namespace NUMINAMATH_CALUDE_product_expansion_sum_l3169_316910

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x, (2*x^2 - 3*x + 5)*(9 - x) = a*x^3 + b*x^2 + c*x + d) → 
  9*a + 3*b + c + d = 58 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l3169_316910


namespace NUMINAMATH_CALUDE_distance_to_stream_is_six_l3169_316912

/-- Represents a trapezoidal forest with a stream -/
structure TrapezidalForest where
  side1 : ℝ  -- Length of the side closest to Wendy's house
  side2 : ℝ  -- Length of the opposite parallel side
  area : ℝ   -- Total area of the forest
  stream_divides_in_half : Bool  -- Whether the stream divides the area in half

/-- The distance from either parallel side to the stream in the trapezoidal forest -/
def distance_to_stream (forest : TrapezidalForest) : ℝ :=
  sorry

/-- Theorem stating that the distance to the stream is 6 miles for the given forest -/
theorem distance_to_stream_is_six (forest : TrapezidalForest) 
  (h1 : forest.side1 = 8)
  (h2 : forest.side2 = 14)
  (h3 : forest.area = 132)
  (h4 : forest.stream_divides_in_half = true) :
  distance_to_stream forest = 6 :=
  sorry

end NUMINAMATH_CALUDE_distance_to_stream_is_six_l3169_316912


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3169_316991

theorem arithmetic_calculation : 4 * 6 * 8 - 24 / 6 = 188 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3169_316991


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3169_316903

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron -/
def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 3 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 3,
    PR := 4,
    PS := 5,
    QR := 5,
    QS := Real.sqrt 34,
    RS := Real.sqrt 41
  }
  volume t = 3 := by sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3169_316903


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l3169_316928

/-- The ratio of volumes of two cubes, one with sides of 2 meters and another with sides of 100 centimeters. -/
theorem cube_volume_ratio : 
  let cube1_side : ℝ := 2  -- Side length of Cube 1 in meters
  let cube2_side : ℝ := 100 / 100  -- Side length of Cube 2 in meters (100 cm converted to m)
  let cube1_volume := cube1_side ^ 3
  let cube2_volume := cube2_side ^ 3
  cube1_volume / cube2_volume = 8 := by sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l3169_316928


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3169_316953

theorem complex_number_quadrant : ∃ (z : ℂ), z = (2 + 4*I) / (1 + I) ∧ (z.re > 0 ∧ z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3169_316953


namespace NUMINAMATH_CALUDE_optimal_purchase_plan_l3169_316905

/-- Represents the daily transportation capacity of machine A in tons -/
def machine_A_capacity : ℝ := 90

/-- Represents the daily transportation capacity of machine B in tons -/
def machine_B_capacity : ℝ := 100

/-- Represents the cost of machine A in yuan -/
def machine_A_cost : ℝ := 15000

/-- Represents the cost of machine B in yuan -/
def machine_B_cost : ℝ := 20000

/-- Represents the total number of machines to be purchased -/
def total_machines : ℕ := 30

/-- Represents the minimum daily transportation requirement in tons -/
def min_daily_transportation : ℝ := 2880

/-- Represents the maximum purchase amount in yuan -/
def max_purchase_amount : ℝ := 550000

/-- Represents the optimal number of A machines to purchase -/
def optimal_A_machines : ℕ := 12

/-- Represents the optimal number of B machines to purchase -/
def optimal_B_machines : ℕ := 18

/-- Represents the total purchase amount for the optimal plan in yuan -/
def optimal_purchase_amount : ℝ := 54000

theorem optimal_purchase_plan :
  (machine_B_capacity = machine_A_capacity + 10) ∧
  (450 / machine_A_capacity = 500 / machine_B_capacity) ∧
  (optimal_A_machines + optimal_B_machines = total_machines) ∧
  (optimal_A_machines * machine_A_capacity + optimal_B_machines * machine_B_capacity ≥ min_daily_transportation) ∧
  (optimal_A_machines * machine_A_cost + optimal_B_machines * machine_B_cost = optimal_purchase_amount) ∧
  (optimal_purchase_amount ≤ max_purchase_amount) ∧
  (∀ a b : ℕ, a + b = total_machines →
    a * machine_A_capacity + b * machine_B_capacity ≥ min_daily_transportation →
    a * machine_A_cost + b * machine_B_cost ≤ max_purchase_amount →
    a * machine_A_cost + b * machine_B_cost ≥ optimal_purchase_amount) := by
  sorry


end NUMINAMATH_CALUDE_optimal_purchase_plan_l3169_316905


namespace NUMINAMATH_CALUDE_roots_sum_magnitude_l3169_316969

theorem roots_sum_magnitude (p : ℝ) (r₁ r₂ : ℝ) : 
  (∃ x : ℝ, x^2 + p*x + 12 = 0) →
  r₁^2 + p*r₁ + 12 = 0 →
  r₂^2 + p*r₂ + 12 = 0 →
  r₁ ≠ r₂ →
  |r₁ + r₂| > 6 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_magnitude_l3169_316969


namespace NUMINAMATH_CALUDE_no_intersection_points_l3169_316947

theorem no_intersection_points : 
  ¬∃ (x y : ℝ), (9 * x^2 + y^2 = 9) ∧ (x^2 + 16 * y^2 = 16) := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_points_l3169_316947


namespace NUMINAMATH_CALUDE_expression_simplification_l3169_316979

theorem expression_simplification (y : ℝ) : 
  (3 - 4*y) * (3 + 4*y) + (3 + 4*y)^2 = 18 + 24*y := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3169_316979


namespace NUMINAMATH_CALUDE_sum_of_even_integers_between_0_and_18_l3169_316996

theorem sum_of_even_integers_between_0_and_18 : 
  (Finset.filter (fun n => n % 2 = 0) (Finset.range 18)).sum id = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_between_0_and_18_l3169_316996


namespace NUMINAMATH_CALUDE_claire_balloons_l3169_316900

/-- The number of balloons Claire has at the end of the fair -/
def final_balloons (initial : ℕ) (floated_away : ℕ) (given_away : ℕ) (grabbed : ℕ) : ℕ :=
  initial - floated_away - given_away + grabbed

/-- Theorem stating that Claire ends up with 40 balloons -/
theorem claire_balloons : final_balloons 50 12 9 11 = 40 := by
  sorry

end NUMINAMATH_CALUDE_claire_balloons_l3169_316900


namespace NUMINAMATH_CALUDE_cos_to_sin_shift_l3169_316916

theorem cos_to_sin_shift (x : ℝ) : 
  Real.cos (2 * x + π / 3) = Real.sin (2 * (x + 5 * π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_cos_to_sin_shift_l3169_316916


namespace NUMINAMATH_CALUDE_benny_turnips_l3169_316906

theorem benny_turnips (melanie_turnips benny_turnips total_turnips : ℕ) 
  (h1 : melanie_turnips = 139)
  (h2 : total_turnips = 252)
  (h3 : melanie_turnips + benny_turnips = total_turnips) : 
  benny_turnips = 113 := by
  sorry

end NUMINAMATH_CALUDE_benny_turnips_l3169_316906


namespace NUMINAMATH_CALUDE_p_on_x_axis_p_on_line_through_a_m_in_third_quadrant_l3169_316966

-- Define point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (2*m + 5, 3*m + 3)

-- Define point A
def A : ℝ × ℝ := (-5, 1)

-- Define point M as a function of m
def M (m : ℝ) : ℝ × ℝ := (2*m + 7, 3*m + 6)

-- Theorem 1
theorem p_on_x_axis (m : ℝ) : 
  (P m).2 = 0 → m = -1 := by sorry

-- Theorem 2
theorem p_on_line_through_a (m : ℝ) :
  (P m).1 = A.1 → P m = (-5, -12) := by sorry

-- Theorem 3
theorem m_in_third_quadrant (m : ℝ) :
  (M m).1 < 0 ∧ (M m).2 < 0 ∧ |(M m).1| = 7 → M m = (-7, -15) := by sorry

end NUMINAMATH_CALUDE_p_on_x_axis_p_on_line_through_a_m_in_third_quadrant_l3169_316966


namespace NUMINAMATH_CALUDE_eight_power_twelve_sum_equals_two_power_y_l3169_316976

theorem eight_power_twelve_sum_equals_two_power_y (y : ℕ) : 
  (8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 + 8^12 = 2^y) → y = 39 := by
sorry

end NUMINAMATH_CALUDE_eight_power_twelve_sum_equals_two_power_y_l3169_316976


namespace NUMINAMATH_CALUDE_fox_max_berries_l3169_316999

/-- The number of bear cubs --/
def num_cubs : ℕ := 100

/-- The initial number of berries for the n-th bear cub --/
def initial_berries (n : ℕ) : ℕ := 2^(n-1)

/-- The total number of berries initially --/
def total_berries : ℕ := 2^num_cubs - 1

/-- The maximum number of berries the fox can eat --/
def max_fox_berries : ℕ := 2^num_cubs - (num_cubs + 1)

theorem fox_max_berries :
  ∀ (redistribution : ℕ → ℕ → ℕ),
  (∀ (a b : ℕ), redistribution a b ≤ a + b) →
  (∀ (a b : ℕ), redistribution a b = redistribution b a) →
  (∃ (final_berries : ℕ), ∀ (i : ℕ), i ≤ num_cubs → redistribution (initial_berries i) final_berries = final_berries) →
  (total_berries - num_cubs * final_berries) ≤ max_fox_berries :=
sorry

end NUMINAMATH_CALUDE_fox_max_berries_l3169_316999


namespace NUMINAMATH_CALUDE_auction_starting_price_l3169_316915

/-- Auction price calculation -/
theorem auction_starting_price
  (final_price : ℕ)
  (price_increase : ℕ)
  (bids_per_person : ℕ)
  (num_bidders : ℕ)
  (h1 : final_price = 65)
  (h2 : price_increase = 5)
  (h3 : bids_per_person = 5)
  (h4 : num_bidders = 2) :
  final_price - (price_increase * bids_per_person * num_bidders) = 15 :=
by sorry

end NUMINAMATH_CALUDE_auction_starting_price_l3169_316915


namespace NUMINAMATH_CALUDE_equal_sum_sequence_sixth_term_l3169_316977

/-- An Equal Sum Sequence is a sequence where the sum of each term and its next term is always the same constant. -/
def EqualSumSequence (a : ℕ → ℝ) (c : ℝ) :=
  ∀ n, a n + a (n + 1) = c

theorem equal_sum_sequence_sixth_term
  (a : ℕ → ℝ)
  (h1 : EqualSumSequence a 5)
  (h2 : a 1 = 2) :
  a 6 = 3 := by
sorry

end NUMINAMATH_CALUDE_equal_sum_sequence_sixth_term_l3169_316977


namespace NUMINAMATH_CALUDE_jen_final_distance_l3169_316973

/-- Calculates the final distance from the starting point for a person walking
    at a constant rate, forward for a certain time, then back for another time. -/
def final_distance (rate : ℝ) (forward_time : ℝ) (back_time : ℝ) : ℝ :=
  rate * forward_time - rate * back_time

/-- Theorem stating that given the specific conditions of Jen's walk,
    her final distance from the starting point is 4 miles. -/
theorem jen_final_distance :
  let rate : ℝ := 4
  let forward_time : ℝ := 2
  let back_time : ℝ := 1
  final_distance rate forward_time back_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_jen_final_distance_l3169_316973


namespace NUMINAMATH_CALUDE_custom_ops_simplification_and_evaluation_l3169_316902

/-- Custom addition operation for rational numbers -/
def star (a b : ℚ) : ℚ := a + b

/-- Custom subtraction operation for rational numbers -/
def otimes (a b : ℚ) : ℚ := a - b

/-- Theorem stating the simplification and evaluation of the given expression -/
theorem custom_ops_simplification_and_evaluation :
  ∀ a b : ℚ, 
  (star (a^2 * b) (3 * a * b) + otimes (5 * a^2 * b) (4 * a * b) = 6 * a^2 * b - a * b) ∧
  (star (5^2 * 3) (3 * 5 * 3) + otimes (5 * 5^2 * 3) (4 * 5 * 3) = 435) := by sorry

end NUMINAMATH_CALUDE_custom_ops_simplification_and_evaluation_l3169_316902


namespace NUMINAMATH_CALUDE_a_range_l3169_316925

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 < 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≤ 0

-- Define the theorem
theorem a_range (a : ℝ) : 
  (a > 0) →
  (∀ x, ¬(p x) → ¬(q x a)) →
  (∃ x, ¬(p x) ∧ (q x a)) →
  a ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_a_range_l3169_316925


namespace NUMINAMATH_CALUDE_other_number_is_nine_l3169_316963

theorem other_number_is_nine (x : ℝ) (h1 : (x + 5) / 2 = 7) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_other_number_is_nine_l3169_316963


namespace NUMINAMATH_CALUDE_exists_divisible_by_15_with_sqrt_between_30_and_30_5_l3169_316907

theorem exists_divisible_by_15_with_sqrt_between_30_and_30_5 :
  ∃ n : ℕ+, 15 ∣ n ∧ 30 < (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 30.5 :=
by
  use 900
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_15_with_sqrt_between_30_and_30_5_l3169_316907


namespace NUMINAMATH_CALUDE_quartic_to_quadratic_reduction_l3169_316922

/-- Given a quartic equation and a substitution, prove it can be reduced to two quadratic equations -/
theorem quartic_to_quadratic_reduction (a b c : ℝ) (x y : ℝ) :
  (a * x^4 + b * x^3 + c * x^2 + b * x + a = 0) →
  (y = x + 1/x) →
  ∃ (y₁ y₂ : ℝ),
    (a * y^2 + b * y + (c - 2*a) = 0) ∧
    (x^2 - y₁ * x + 1 = 0 ∨ x^2 - y₂ * x + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quartic_to_quadratic_reduction_l3169_316922


namespace NUMINAMATH_CALUDE_solution_to_system_l3169_316945

theorem solution_to_system :
  ∃ (x y : ℚ), 3 * x - 24 * y = 3 ∧ x - 3 * y = 4 ∧ x = 29/5 ∧ y = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_l3169_316945


namespace NUMINAMATH_CALUDE_trivia_game_total_score_l3169_316956

theorem trivia_game_total_score :
  let team_a : Int := 2
  let team_b : Int := 9
  let team_c : Int := 4
  let team_d : Int := -3
  let team_e : Int := 7
  let team_f : Int := 0
  let team_g : Int := 5
  let team_h : Int := -2
  team_a + team_b + team_c + team_d + team_e + team_f + team_g + team_h = 22 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_total_score_l3169_316956


namespace NUMINAMATH_CALUDE_max_k_for_intersecting_circles_l3169_316932

/-- The maximum value of k for which a circle with radius 1 centered on the line y = kx - 2 
    intersects the circle x² + y² - 8x + 15 = 0 -/
theorem max_k_for_intersecting_circles : 
  ∃ (max_k : ℝ), max_k = 4/3 ∧ 
  (∀ k : ℝ, (∃ x y : ℝ, 
    y = k * x - 2 ∧ 
    (∃ cx cy : ℝ, (cx - x)^2 + (cy - y)^2 = 1 ∧ 
      cx^2 + cy^2 - 8*cx + 15 = 0)) → 
    k ≤ max_k) ∧
  (∃ x y : ℝ, 
    y = max_k * x - 2 ∧ 
    (∃ cx cy : ℝ, (cx - x)^2 + (cy - y)^2 = 1 ∧ 
      cx^2 + cy^2 - 8*cx + 15 = 0)) :=
sorry

end NUMINAMATH_CALUDE_max_k_for_intersecting_circles_l3169_316932


namespace NUMINAMATH_CALUDE_longer_worm_length_l3169_316941

/-- Given two worms, where one is 0.1 inch long and the other is 0.7 inches longer,
    prove that the longer worm is 0.8 inches long. -/
theorem longer_worm_length (short_worm long_worm : ℝ) 
  (h1 : short_worm = 0.1)
  (h2 : long_worm = short_worm + 0.7) :
  long_worm = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_longer_worm_length_l3169_316941


namespace NUMINAMATH_CALUDE_unique_seven_l3169_316943

/-- A function that returns true if the given positive integer n results in
    exactly one term with a rational coefficient in the binomial expansion
    of (√3x + ∛2)^n -/
def has_one_rational_term (n : ℕ+) : Prop :=
  ∃! r : ℕ, r ≤ n ∧ 3 ∣ r ∧ 2 ∣ (n - r)

/-- Theorem stating that 7 is the only positive integer satisfying the condition -/
theorem unique_seven : ∀ n : ℕ+, has_one_rational_term n ↔ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_seven_l3169_316943


namespace NUMINAMATH_CALUDE_system_solution_l3169_316904

theorem system_solution (x y : ℝ) :
  (2 / (x^2 + y^2) + x^2 * y^2 = 2) ∧
  (x^4 + y^4 + 3 * x^2 * y^2 = 5) ↔
  ((x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3169_316904


namespace NUMINAMATH_CALUDE_toy_gift_box_discount_l3169_316911

theorem toy_gift_box_discount (cost_price marked_price discount profit_margin : ℝ) : 
  cost_price = 160 →
  marked_price = 240 →
  discount = 20 →
  profit_margin = 20 →
  marked_price * (1 - discount / 100) = cost_price * (1 + profit_margin / 100) :=
by sorry

end NUMINAMATH_CALUDE_toy_gift_box_discount_l3169_316911


namespace NUMINAMATH_CALUDE_sum_first_150_remainder_l3169_316972

theorem sum_first_150_remainder (n : Nat) (divisor : Nat) : n = 150 → divisor = 11200 → 
  (n * (n + 1) / 2) % divisor = 125 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_150_remainder_l3169_316972


namespace NUMINAMATH_CALUDE_min_k_value_l3169_316942

theorem min_k_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ k : ℝ, (1 / a + 1 / b + k / (a + b) ≥ 0)) →
  (∃ k_min : ℝ, k_min = -4 ∧ ∀ k : ℝ, (1 / a + 1 / b + k / (a + b) ≥ 0) → k ≥ k_min) :=
by sorry

end NUMINAMATH_CALUDE_min_k_value_l3169_316942


namespace NUMINAMATH_CALUDE_colin_running_time_l3169_316901

def total_miles : ℕ := 4
def first_mile_time : ℕ := 6
def fourth_mile_time : ℕ := 4
def average_time : ℕ := 5

theorem colin_running_time (second_mile_time third_mile_time : ℕ) 
  (h1 : second_mile_time = third_mile_time) 
  (h2 : first_mile_time + second_mile_time + third_mile_time + fourth_mile_time = total_miles * average_time) : 
  second_mile_time = 5 ∧ third_mile_time = 5 := by
  sorry

#check colin_running_time

end NUMINAMATH_CALUDE_colin_running_time_l3169_316901


namespace NUMINAMATH_CALUDE_total_height_difference_l3169_316986

def heightProblem (anne cathy bella daisy ellie : ℝ) : Prop :=
  anne = 80 ∧
  cathy = anne / 2 ∧
  bella = 3 * anne ∧
  daisy = (cathy + anne) / 2 ∧
  ellie = Real.sqrt (bella * cathy) ∧
  |bella - cathy| + |bella - daisy| + |bella - ellie| + 
  |cathy - daisy| + |cathy - ellie| + |daisy - ellie| = 638

theorem total_height_difference :
  ∃ (anne cathy bella daisy ellie : ℝ),
    heightProblem anne cathy bella daisy ellie :=
by
  sorry

end NUMINAMATH_CALUDE_total_height_difference_l3169_316986


namespace NUMINAMATH_CALUDE_bear_weight_gain_ratio_l3169_316992

theorem bear_weight_gain_ratio :
  let total_weight : ℝ := 1000
  let berry_weight : ℝ := total_weight / 5
  let small_animal_weight : ℝ := 200
  let salmon_weight : ℝ := (total_weight - berry_weight - small_animal_weight) / 2
  let acorn_weight : ℝ := total_weight - berry_weight - small_animal_weight - salmon_weight
  acorn_weight / berry_weight = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bear_weight_gain_ratio_l3169_316992


namespace NUMINAMATH_CALUDE_unique_common_point_modulo25_l3169_316983

/-- Given two congruences on modulo 25 graph paper, prove there's exactly one common point with x-coordinate 1 --/
theorem unique_common_point_modulo25 : ∃! p : ℕ × ℕ, 
  p.1 < 25 ∧ 
  p.2 < 25 ∧
  p.2 ≡ 10 * p.1 + 3 [ZMOD 25] ∧ 
  p.2 ≡ p.1^2 + 15 * p.1 + 20 [ZMOD 25] ∧
  p.1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_common_point_modulo25_l3169_316983


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3169_316957

theorem linear_equation_solution (a : ℝ) :
  (1 : ℝ) * a + (-2 : ℝ) = (3 : ℝ) → a = (5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3169_316957


namespace NUMINAMATH_CALUDE_least_non_special_fraction_l3169_316919

/-- Represents a fraction in the form (2^a - 2^b) / (2^c - 2^d) where a, b, c, d are positive integers -/
def SpecialFraction (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ+), n = (2^a.val - 2^b.val) / (2^c.val - 2^d.val)

/-- The least positive integer that cannot be represented as a SpecialFraction is 11 -/
theorem least_non_special_fraction : (∀ k < 11, SpecialFraction k) ∧ ¬SpecialFraction 11 := by
  sorry

end NUMINAMATH_CALUDE_least_non_special_fraction_l3169_316919


namespace NUMINAMATH_CALUDE_dog_spots_l3169_316958

/-- The number of spots on dogs problem -/
theorem dog_spots (rover_spots : ℕ) (cisco_spots : ℕ) (granger_spots : ℕ)
  (h1 : rover_spots = 46)
  (h2 : cisco_spots = rover_spots / 2 - 5)
  (h3 : granger_spots = 5 * cisco_spots) :
  granger_spots + cisco_spots = 108 := by
  sorry

end NUMINAMATH_CALUDE_dog_spots_l3169_316958


namespace NUMINAMATH_CALUDE_line_slope_angle_l3169_316927

/-- The slope angle of a line given by parametric equations -/
def slope_angle (x y : ℝ → ℝ) : ℝ := sorry

theorem line_slope_angle :
  let x : ℝ → ℝ := λ t => Real.sin θ + t * Real.sin (15 * π / 180)
  let y : ℝ → ℝ := λ t => Real.cos θ - t * Real.sin (75 * π / 180)
  slope_angle x y = 105 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_line_slope_angle_l3169_316927


namespace NUMINAMATH_CALUDE_marks_used_days_ratio_l3169_316959

def total_allotted_days : ℕ := 20
def hours_per_day : ℕ := 8
def unused_hours : ℕ := 80

theorem marks_used_days_ratio :
  let unused_days : ℕ := unused_hours / hours_per_day
  let used_days : ℕ := total_allotted_days - unused_days
  (used_days : ℚ) / total_allotted_days = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_marks_used_days_ratio_l3169_316959


namespace NUMINAMATH_CALUDE_min_value_theorem_l3169_316921

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (1 / a + 2 / b) ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = 1 ∧ 1 / a₀ + 2 / b₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3169_316921


namespace NUMINAMATH_CALUDE_campers_rowing_morning_l3169_316913

theorem campers_rowing_morning (hiking_morning : ℕ) (rowing_afternoon : ℕ) (total_campers : ℕ) :
  hiking_morning = 4 →
  rowing_afternoon = 26 →
  total_campers = 71 →
  total_campers - (hiking_morning + rowing_afternoon) = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_campers_rowing_morning_l3169_316913


namespace NUMINAMATH_CALUDE_correct_hours_calculation_l3169_316971

/-- Calculates the number of hours worked given hourly rates and total payment -/
def hours_worked (bricklayer_rate electrician_rate total_payment : ℚ) : ℚ :=
  total_payment / (bricklayer_rate + electrician_rate)

/-- Theorem stating that the calculated hours worked is correct -/
theorem correct_hours_calculation 
  (bricklayer_rate electrician_rate total_payment : ℚ) 
  (h1 : bricklayer_rate = 12)
  (h2 : electrician_rate = 16)
  (h3 : total_payment = 1350) :
  hours_worked bricklayer_rate electrician_rate total_payment = 1350 / 28 :=
by sorry

end NUMINAMATH_CALUDE_correct_hours_calculation_l3169_316971


namespace NUMINAMATH_CALUDE_cathy_final_state_l3169_316946

/-- Represents a direction on the coordinate plane -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a position on the coordinate plane -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents Cathy's state after each move -/
structure CathyState :=
  (position : Position)
  (direction : Direction)
  (moveNumber : Nat)
  (distanceTraveled : Nat)

/-- Calculates the next direction after turning right -/
def turnRight (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.East
  | Direction.East => Direction.South
  | Direction.South => Direction.West
  | Direction.West => Direction.North

/-- Calculates the distance for a given move number -/
def moveDistance (n : Nat) : Nat :=
  2 * n

/-- Updates the position based on the current direction and distance -/
def updatePosition (p : Position) (d : Direction) (dist : Nat) : Position :=
  match d with
  | Direction.North => ⟨p.x, p.y + dist⟩
  | Direction.East => ⟨p.x + dist, p.y⟩
  | Direction.South => ⟨p.x, p.y - dist⟩
  | Direction.West => ⟨p.x - dist, p.y⟩

/-- Performs a single move and updates Cathy's state -/
def move (state : CathyState) : CathyState :=
  let newMoveNumber := state.moveNumber + 1
  let distance := moveDistance newMoveNumber
  let newPosition := updatePosition state.position state.direction distance
  let newDirection := turnRight state.direction
  let newDistanceTraveled := state.distanceTraveled + distance
  ⟨newPosition, newDirection, newMoveNumber, newDistanceTraveled⟩

/-- Performs n moves starting from the given initial state -/
def performMoves (initialState : CathyState) (n : Nat) : CathyState :=
  match n with
  | 0 => initialState
  | m + 1 => move (performMoves initialState m)

/-- The main theorem to prove -/
theorem cathy_final_state :
  let initialState : CathyState := ⟨⟨2, -3⟩, Direction.North, 0, 0⟩
  let finalState := performMoves initialState 12
  finalState.position = ⟨-10, -15⟩ ∧ finalState.distanceTraveled = 146 := by
  sorry


end NUMINAMATH_CALUDE_cathy_final_state_l3169_316946


namespace NUMINAMATH_CALUDE_compound_propositions_truth_count_l3169_316936

theorem compound_propositions_truth_count
  (p q : Prop)
  (hp : p)
  (hq : ¬q) :
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) ∧ (¬q) :=
by sorry

end NUMINAMATH_CALUDE_compound_propositions_truth_count_l3169_316936


namespace NUMINAMATH_CALUDE_roots_pure_imaginary_l3169_316985

theorem roots_pure_imaginary (k : ℝ) (hk : k > 0) :
  ∃ (b c : ℝ), ∀ (z : ℂ), 8 * z^2 - 5 * I * z - k = 0 → z = b * I ∨ z = c * I :=
by sorry

end NUMINAMATH_CALUDE_roots_pure_imaginary_l3169_316985


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3169_316917

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The focal distance of a hyperbola -/
def focal_distance (h : Hyperbola) : ℝ := sorry

/-- The distance from a focus to an asymptote of a hyperbola -/
def focus_to_asymptote_distance (h : Hyperbola) : ℝ := sorry

/-- Theorem: If the distance from a focus to an asymptote is 1/4 of the focal distance,
    then the eccentricity is 2√3/3 -/
theorem hyperbola_eccentricity (h : Hyperbola) 
    (h_dist : focus_to_asymptote_distance h = (1/4) * focal_distance h) : 
    eccentricity h = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3169_316917


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l3169_316924

theorem x_squared_plus_reciprocal (x : ℝ) (h : 59 = x^4 + 1/x^4) : x^2 + 1/x^2 = Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l3169_316924


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_line_l3169_316929

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- m is contained in α -/
def contained_in (m : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- m is perpendicular to β -/
def perpendicular_line_plane (m : Line3D) (β : Plane3D) : Prop :=
  sorry

/-- α is perpendicular to β -/
def perpendicular_planes (α β : Plane3D) : Prop :=
  sorry

/-- If a line m is contained in a plane α and is perpendicular to another plane β, 
    then α is perpendicular to β -/
theorem perpendicular_planes_from_line 
  (m : Line3D) (α β : Plane3D) : 
  contained_in m α → perpendicular_line_plane m β → perpendicular_planes α β :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_line_l3169_316929


namespace NUMINAMATH_CALUDE_rabbit_logs_l3169_316955

theorem rabbit_logs (cuts pieces : ℕ) (h1 : cuts = 10) (h2 : pieces = 16) :
  pieces - cuts = 6 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_logs_l3169_316955


namespace NUMINAMATH_CALUDE_lily_initial_money_l3169_316978

def celery_cost : ℝ := 5
def cereal_original_cost : ℝ := 12
def cereal_discount : ℝ := 0.5
def bread_cost : ℝ := 8
def milk_original_cost : ℝ := 10
def milk_discount : ℝ := 0.1
def potato_cost : ℝ := 1
def potato_quantity : ℕ := 6
def coffee_budget : ℝ := 26

def total_cost : ℝ := 
  celery_cost + 
  cereal_original_cost * (1 - cereal_discount) + 
  bread_cost + 
  milk_original_cost * (1 - milk_discount) + 
  potato_cost * potato_quantity +
  coffee_budget

theorem lily_initial_money : total_cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_lily_initial_money_l3169_316978


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l3169_316987

/-- Given vectors a and b in ℝ², prove that if k*a + b is perpendicular to a - 3*b, then k = 19 -/
theorem perpendicular_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-3, 2))
    (h3 : (k * a.1 + b.1, k * a.2 + b.2) • (a.1 - 3 * b.1, a.2 - 3 * b.2) = 0) :
  k = 19 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l3169_316987


namespace NUMINAMATH_CALUDE_city_of_pythagoras_schools_l3169_316997

/-- Represents a student in the math contest -/
structure Student where
  school : Nat
  rank : Nat

/-- The math contest setup -/
structure MathContest where
  numSchools : Nat
  students : Finset Student

theorem city_of_pythagoras_schools (contest : MathContest) : contest.numSchools = 40 :=
  by
  have h1 : ∀ s : Student, s ∈ contest.students → s.rank ≤ 4 * contest.numSchools :=
    sorry
  have h2 : ∀ s1 s2 : Student, s1 ∈ contest.students → s2 ∈ contest.students → s1 ≠ s2 → s1.rank ≠ s2.rank :=
    sorry
  have h3 : ∃ andrea : Student, andrea ∈ contest.students ∧
    andrea.rank = (2 * contest.numSchools) ∨ andrea.rank = (2 * contest.numSchools + 1) :=
    sorry
  have h4 : ∃ beth : Student, beth ∈ contest.students ∧ beth.rank = 41 :=
    sorry
  have h5 : ∃ carla : Student, carla ∈ contest.students ∧ carla.rank = 82 :=
    sorry
  have h6 : ∃ andrea beth carla : Student, 
    andrea ∈ contest.students ∧ beth ∈ contest.students ∧ carla ∈ contest.students ∧
    andrea.school = beth.school ∧ andrea.school = carla.school ∧
    andrea.rank < beth.rank ∧ andrea.rank < carla.rank :=
    sorry
  sorry


end NUMINAMATH_CALUDE_city_of_pythagoras_schools_l3169_316997


namespace NUMINAMATH_CALUDE_sector_area_of_circle_l3169_316960

/-- Given a circle with circumference 16π, prove that the area of a sector
    subtending a central angle of 90° is 16π. -/
theorem sector_area_of_circle (C : ℝ) (θ : ℝ) (h1 : C = 16 * Real.pi) (h2 : θ = 90) :
  let r := C / (2 * Real.pi)
  let A := Real.pi * r^2
  let sector_area := (θ / 360) * A
  sector_area = 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sector_area_of_circle_l3169_316960


namespace NUMINAMATH_CALUDE_spectators_count_l3169_316931

/-- The number of wristbands given to each spectator -/
def wristbands_per_person : ℕ := 2

/-- The total number of wristbands distributed -/
def total_wristbands : ℕ := 250

/-- The number of people who watched the game -/
def spectators : ℕ := total_wristbands / wristbands_per_person

theorem spectators_count : spectators = 125 := by
  sorry

end NUMINAMATH_CALUDE_spectators_count_l3169_316931


namespace NUMINAMATH_CALUDE_odd_function_property_l3169_316982

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h1 : is_odd_function f)
  (h2 : is_even_function (fun x ↦ f (x + 2)))
  (h3 : f (-1) = -1) :
  f 2017 + f 2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3169_316982


namespace NUMINAMATH_CALUDE_range_of_c_l3169_316964

/-- The range of c for which y = c^x is a decreasing function and x^2 - √2x + c > 0 does not hold for all x ∈ ℝ -/
theorem range_of_c (c : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → c^x₁ > c^x₂) → -- y = c^x is a decreasing function
  (¬∀ x : ℝ, x^2 - Real.sqrt 2 * x + c > 0) → -- negation of q
  ((∀ x₁ x₂ : ℝ, x₁ < x₂ → c^x₁ > c^x₂) ∨ (∀ x : ℝ, x^2 - Real.sqrt 2 * x + c > 0)) → -- p or q
  0 < c ∧ c ≤ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_range_of_c_l3169_316964


namespace NUMINAMATH_CALUDE_sum_35_25_base6_l3169_316908

/-- Represents a number in base 6 --/
def Base6 := Nat

/-- Converts a base 6 number to a natural number --/
def to_nat (b : Base6) : Nat := sorry

/-- Converts a natural number to a base 6 number --/
def from_nat (n : Nat) : Base6 := sorry

/-- Adds two base 6 numbers --/
def add_base6 (a b : Base6) : Base6 := from_nat (to_nat a + to_nat b)

theorem sum_35_25_base6 :
  add_base6 (from_nat 35) (from_nat 25) = from_nat 104 := by sorry

end NUMINAMATH_CALUDE_sum_35_25_base6_l3169_316908


namespace NUMINAMATH_CALUDE_cattle_truck_capacity_l3169_316990

/-- Calculates the capacity of a cattle transport truck given the total number of cattle,
    distance to safety, truck speed, and total transport time. -/
theorem cattle_truck_capacity
  (total_cattle : ℕ)
  (distance : ℝ)
  (speed : ℝ)
  (total_time : ℝ)
  (h_total_cattle : total_cattle = 400)
  (h_distance : distance = 60)
  (h_speed : speed = 60)
  (h_total_time : total_time = 40)
  : ℕ :=
by
  sorry

#check cattle_truck_capacity

end NUMINAMATH_CALUDE_cattle_truck_capacity_l3169_316990


namespace NUMINAMATH_CALUDE_diamond_value_l3169_316984

/-- Represents a digit (0-9) -/
def Digit := Fin 10

theorem diamond_value (diamond : Digit) :
  (9 * diamond.val + 6 = 10 * diamond.val + 3) → diamond.val = 3 := by
  sorry

end NUMINAMATH_CALUDE_diamond_value_l3169_316984
