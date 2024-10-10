import Mathlib

namespace dime_count_proof_l3578_357848

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Calculates the number of dimes given the total amount, number of quarters, and number of nickels -/
def calculate_dimes (total_amount : ℕ) (num_quarters : ℕ) (num_nickels : ℕ) : ℕ :=
  (total_amount * cents_per_dollar - (num_quarters * quarter_value + num_nickels * nickel_value)) / dime_value

theorem dime_count_proof (total_amount : ℕ) (num_quarters : ℕ) (num_nickels : ℕ) 
  (h1 : total_amount = 4)
  (h2 : num_quarters = 10)
  (h3 : num_nickels = 6) :
  calculate_dimes total_amount num_quarters num_nickels = 12 := by
  sorry

end dime_count_proof_l3578_357848


namespace total_tickets_bought_l3578_357838

/-- Represents the cost of an adult ticket in dollars -/
def adult_ticket_cost : ℚ := 5.5

/-- Represents the cost of a child ticket in dollars -/
def child_ticket_cost : ℚ := 3.5

/-- Represents the total cost of all tickets bought in dollars -/
def total_cost : ℚ := 83.5

/-- Represents the number of children's tickets bought -/
def num_child_tickets : ℕ := 16

/-- Theorem stating that the total number of tickets bought is 21 -/
theorem total_tickets_bought : ℕ := by
  sorry

end total_tickets_bought_l3578_357838


namespace algebraic_expression_value_l3578_357861

theorem algebraic_expression_value (a b : ℝ) (h : a - 2*b = 2) : 2*a - 4*b + 1 = 5 := by
  sorry

end algebraic_expression_value_l3578_357861


namespace fraction_division_problem_solution_l3578_357831

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem problem_solution :
  (5 : ℚ) / 6 / ((11 : ℚ) / 12) = 10 / 11 := by sorry

end fraction_division_problem_solution_l3578_357831


namespace almas_test_score_l3578_357864

/-- Proves that Alma's test score is 45 given the specified conditions. -/
theorem almas_test_score (alma_age melina_age carlos_age alma_score carlos_score : ℕ) : 
  alma_age + melina_age + carlos_age = 3 * alma_score →
  melina_age = 3 * alma_age →
  carlos_age = 4 * alma_age →
  melina_age = 60 →
  carlos_score = 2 * alma_score + 15 →
  carlos_score - alma_score = melina_age →
  alma_score = 45 := by
  sorry

#check almas_test_score

end almas_test_score_l3578_357864


namespace min_value_theorem_l3578_357851

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y = 2) :
  (2 / x) + (1 / y) ≥ 9 / 2 := by
sorry

end min_value_theorem_l3578_357851


namespace trihedral_angle_obtuse_angles_l3578_357858

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  AOB : ℝ
  BOC : ℝ
  COA : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Theorem: If all plane angles of a trihedral angle are obtuse, then all dihedral angles are obtuse -/
theorem trihedral_angle_obtuse_angles (t : TrihedralAngle)
  (h_AOB : t.AOB > π / 2)
  (h_BOC : t.BOC > π / 2)
  (h_COA : t.COA > π / 2) :
  t.α > π / 2 ∧ t.β > π / 2 ∧ t.γ > π / 2 := by
  sorry

end trihedral_angle_obtuse_angles_l3578_357858


namespace partial_multiplication_reconstruction_l3578_357859

/-- Represents a partially visible digit (0-9 or unknown) -/
inductive PartialDigit
  | Known (n : Fin 10)
  | Unknown

/-- Represents a partially visible number -/
def PartialNumber := List PartialDigit

/-- Represents a multiplication step in the written method -/
structure MultiplicationStep where
  multiplicand : PartialNumber
  multiplier : PartialNumber
  partialProducts : List PartialNumber
  result : PartialNumber

/-- Check if a number matches a partial number -/
def matchesPartial (n : ℕ) (pn : PartialNumber) : Prop := sorry

/-- The main theorem to prove -/
theorem partial_multiplication_reconstruction 
  (step : MultiplicationStep)
  (h1 : step.multiplicand.length = 3)
  (h2 : step.multiplier.length = 3)
  (h3 : matchesPartial 56576 step.result)
  : ∃ (a b : ℕ), 
    a * b = 56500 ∧ 
    matchesPartial a step.multiplicand ∧ 
    matchesPartial b step.multiplier :=
sorry

end partial_multiplication_reconstruction_l3578_357859


namespace prob_at_least_one_of_three_l3578_357878

/-- The probability that at least one of three independent events occurs, 
    given that each event has a probability of 1/3. -/
theorem prob_at_least_one_of_three (p : ℝ) (h_p : p = 1 / 3) :
  1 - (1 - p)^3 = 19 / 27 := by
  sorry

end prob_at_least_one_of_three_l3578_357878


namespace distribute_four_to_three_l3578_357898

/-- The number of ways to distribute n distinct objects into k distinct containers,
    with each container having at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r objects from n distinct objects. -/
def choose (n r : ℕ) : ℕ := sorry

/-- The number of ways to arrange n distinct objects in k positions. -/
def arrange (n k : ℕ) : ℕ := sorry

theorem distribute_four_to_three :
  distribute 4 3 = 36 :=
by
  have h1 : distribute 4 3 = choose 4 2 * arrange 3 3 := sorry
  sorry


end distribute_four_to_three_l3578_357898


namespace f_even_when_a_zero_f_minimum_when_a_between_neg_one_and_one_l3578_357845

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * abs (x - a)

-- Statement 1: When a = 0, f is an even function
theorem f_even_when_a_zero :
  ∀ x : ℝ, f 0 x = f 0 (-x) := by sorry

-- Statement 2: When -1 < a < 1, f achieves a minimum value of a^2
theorem f_minimum_when_a_between_neg_one_and_one :
  ∀ a : ℝ, -1 < a → a < 1 → ∀ x : ℝ, f a x ≥ a^2 := by sorry

end f_even_when_a_zero_f_minimum_when_a_between_neg_one_and_one_l3578_357845


namespace pr_length_l3578_357804

-- Define the triangles and their side lengths
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

def PQR : Triangle := { side1 := 30, side2 := 18, side3 := 22.5 }
def STU : Triangle := { side1 := 24, side2 := 18, side3 := 18 }

-- Define similarity of triangles
def similar (t1 t2 : Triangle) : Prop :=
  t1.side1 / t2.side1 = t1.side2 / t2.side2 ∧
  t1.side1 / t2.side1 = t1.side3 / t2.side3

-- Theorem statement
theorem pr_length :
  similar PQR STU → PQR.side3 = 22.5 :=
by
  sorry

end pr_length_l3578_357804


namespace product_of_y_coordinates_l3578_357887

/-- Theorem: Product of y-coordinates for point Q -/
theorem product_of_y_coordinates (y₁ y₂ : ℝ) : 
  (((4 - (-2))^2 + (y₁ - (-3))^2 = 7^2) ∧
   ((4 - (-2))^2 + (y₂ - (-3))^2 = 7^2)) →
  y₁ * y₂ = -4 := by
sorry

end product_of_y_coordinates_l3578_357887


namespace actual_average_height_after_correction_actual_average_height_is_184_cm_l3578_357841

/-- The actual average height of boys in a class after correcting measurement errors -/
theorem actual_average_height_after_correction (num_boys : ℕ) 
  (initial_avg : ℝ) (wrong_heights : Fin 4 → ℝ) (correct_heights : Fin 4 → ℝ) : ℝ :=
  let inch_to_cm : ℝ := 2.54
  let total_initial_height : ℝ := num_boys * initial_avg
  let height_difference : ℝ := (wrong_heights 0 - correct_heights 0) + 
                                (wrong_heights 1 - correct_heights 1) + 
                                (wrong_heights 2 - correct_heights 2) + 
                                (wrong_heights 3 * inch_to_cm - correct_heights 3 * inch_to_cm)
  let corrected_total_height : ℝ := total_initial_height - height_difference
  let actual_avg : ℝ := corrected_total_height / num_boys
  actual_avg

/-- The actual average height of boys in the class is 184.00 cm (rounded to two decimal places) -/
theorem actual_average_height_is_184_cm : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ 
  |actual_average_height_after_correction 75 185 
    (λ i => [170, 195, 160, 70][i]) 
    (λ i => [140, 165, 190, 64][i]) - 184| < ε :=
by
  sorry

end actual_average_height_after_correction_actual_average_height_is_184_cm_l3578_357841


namespace not_perfect_square_l3578_357824

theorem not_perfect_square (n : ℕ) : ∀ m : ℕ, 4 * n^2 + 4 * n + 4 ≠ m^2 := by
  sorry

end not_perfect_square_l3578_357824


namespace percentage_problem_l3578_357871

theorem percentage_problem (x : ℝ) (p : ℝ) 
  (h1 : (p / 100) * x = 400)
  (h2 : (120 / 100) * x = 2400) : 
  p = 20 := by sorry

end percentage_problem_l3578_357871


namespace apps_deleted_l3578_357822

theorem apps_deleted (initial_apps new_apps final_apps : ℕ) :
  initial_apps = 10 →
  new_apps = 11 →
  final_apps = 4 →
  initial_apps + new_apps - final_apps = 17 :=
by
  sorry

end apps_deleted_l3578_357822


namespace inscribed_square_side_length_l3578_357879

/-- Given a right triangle PQR with legs of length 9 and 12, prove that a square inscribed
    with one side on the hypotenuse and vertices on the other two sides has side length 45/8 -/
theorem inscribed_square_side_length (P Q R : ℝ × ℝ) 
  (right_angle_P : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0)
  (leg_PQ : (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 9^2)
  (leg_PR : (R.1 - P.1)^2 + (R.2 - P.2)^2 = 12^2)
  (square : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop)
  (inscribed : ∃ (A B C D : ℝ × ℝ), square A B C D ∧ 
    (A.1 - Q.1) * (R.1 - Q.1) + (A.2 - Q.2) * (R.2 - Q.2) = 0 ∧
    (∃ t : ℝ, 0 < t ∧ t < 1 ∧ B = (t * Q.1 + (1 - t) * R.1, t * Q.2 + (1 - t) * R.2)) ∧
    (∃ u : ℝ, 0 < u ∧ u < 1 ∧ D = (u * P.1 + (1 - u) * Q.1, u * P.2 + (1 - u) * Q.2)) ∧
    (∃ v : ℝ, 0 < v ∧ v < 1 ∧ C = (v * P.1 + (1 - v) * R.1, v * P.2 + (1 - v) * R.2)))
  : ∃ (A B C D : ℝ × ℝ), square A B C D ∧ 
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = (45/8)^2 := by
  sorry

end inscribed_square_side_length_l3578_357879


namespace waiter_tables_l3578_357856

theorem waiter_tables (people_per_table : ℕ) (total_customers : ℕ) (h1 : people_per_table = 9) (h2 : total_customers = 63) :
  total_customers / people_per_table = 7 := by
  sorry

end waiter_tables_l3578_357856


namespace students_per_computer_l3578_357823

theorem students_per_computer :
  let initial_students : ℕ := 82
  let additional_students : ℕ := 16
  let total_students : ℕ := initial_students + additional_students
  let computers_after_increase : ℕ := 49
  let students_per_computer : ℚ := initial_students / (total_students / computers_after_increase : ℚ)
  students_per_computer = 2 := by
sorry

end students_per_computer_l3578_357823


namespace monthly_income_of_P_l3578_357891

/-- Given the average monthly incomes of three individuals P, Q, and R,
    prove that the monthly income of P is 4000. -/
theorem monthly_income_of_P (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 →
  (Q + R) / 2 = 6250 →
  (P + R) / 2 = 5200 →
  P = 4000 := by
  sorry

end monthly_income_of_P_l3578_357891


namespace gcd_problems_l3578_357830

theorem gcd_problems :
  (Nat.gcd 840 1764 = 84) ∧ (Nat.gcd 153 119 = 17) := by
  sorry

end gcd_problems_l3578_357830


namespace determinant_of_specific_matrix_l3578_357805

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 4, -2; 0, 3, 1; 5, -1, 3]
  Matrix.det A = 70 := by
sorry

end determinant_of_specific_matrix_l3578_357805


namespace inequality_proof_l3578_357885

theorem inequality_proof (a b : ℝ) (h : 1 / a < 1 / b ∧ 1 / b < 0) :
  (a + b < a * b) ∧ (b / a + a / b > 2) := by sorry

end inequality_proof_l3578_357885


namespace equal_shaded_areas_condition_l3578_357835

/-- Given a circle with radius s and an angle φ, where 0 < φ < π/4,
    this theorem states the necessary and sufficient condition for
    the equality of two specific areas related to the circle. --/
theorem equal_shaded_areas_condition (s : ℝ) (φ : ℝ) 
    (h1 : 0 < φ) (h2 : φ < π/4) (h3 : s > 0) :
  let sector_area := φ * s^2 / 2
  let triangle_area := s^2 * Real.tan φ / 2
  sector_area = triangle_area ↔ Real.tan φ = 3 * φ :=
sorry

end equal_shaded_areas_condition_l3578_357835


namespace hyperbola_eccentricity_l3578_357836

/-- Given a hyperbola with equation x²/a² - y²/2 = 1 where a > √2, 
    if the angle between its asymptotes is π/3, 
    then its eccentricity is 2√3/3 -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > Real.sqrt 2) :
  let angle_between_asymptotes := π / 3
  let slope_of_asymptote := Real.sqrt 2 / a
  let eccentricity := Real.sqrt (a^2 + 2) / a
  (angle_between_asymptotes = π / 3 ∧ 
   slope_of_asymptote = Real.tan (π / 6)) →
  eccentricity = 2 * Real.sqrt 3 / 3 := by
  sorry

end hyperbola_eccentricity_l3578_357836


namespace equation_equivalence_l3578_357837

theorem equation_equivalence : ∀ x : ℝ, (x = 3) ↔ (x - 3 = 0) := by
  sorry

end equation_equivalence_l3578_357837


namespace baba_yaga_powder_division_l3578_357807

/-- Represents the weight measurement system with a possible consistent error --/
structure ScaleSystem where
  total_shown : ℤ
  part1_shown : ℤ
  part2_shown : ℤ
  error : ℤ

/-- The actual weights of the two parts of the powder --/
def actual_weights (s : ScaleSystem) : ℤ × ℤ :=
  (s.part1_shown - s.error, s.part2_shown - s.error)

/-- Theorem stating the correct weights given the scale measurements --/
theorem baba_yaga_powder_division (s : ScaleSystem) 
  (h1 : s.total_shown = 6)
  (h2 : s.part1_shown = 3)
  (h3 : s.part2_shown = 2)
  (h4 : s.total_shown = s.part1_shown + s.part2_shown - s.error) :
  actual_weights s = (4, 3) := by
  sorry


end baba_yaga_powder_division_l3578_357807


namespace winning_percentage_correct_l3578_357865

/-- Represents the percentage of votes secured by the winning candidate -/
def winning_percentage : ℝ := 70

/-- Represents the total number of valid votes -/
def total_votes : ℕ := 450

/-- Represents the majority of votes by which the winning candidate won -/
def vote_majority : ℕ := 180

/-- Theorem stating that the winning percentage is correct given the conditions -/
theorem winning_percentage_correct :
  (winning_percentage / 100 * total_votes : ℝ) -
  ((100 - winning_percentage) / 100 * total_votes : ℝ) = vote_majority :=
sorry

end winning_percentage_correct_l3578_357865


namespace bag_probabilities_l3578_357810

/-- Definition of the bag of balls -/
structure Bag where
  total : ℕ
  red : ℕ
  yellow : ℕ

/-- Initial bag configuration -/
def initialBag : Bag := ⟨20, 5, 15⟩

/-- Probability of picking a ball of a certain color -/
def probability (bag : Bag) (color : ℕ) : ℚ :=
  color / bag.total

/-- Add balls to the bag -/
def addBalls (bag : Bag) (redAdd : ℕ) (yellowAdd : ℕ) : Bag :=
  ⟨bag.total + redAdd + yellowAdd, bag.red + redAdd, bag.yellow + yellowAdd⟩

theorem bag_probabilities (bag : Bag := initialBag) :
  (probability bag bag.yellow > probability bag bag.red) ∧
  (probability bag bag.red = 1/4) ∧
  (probability (addBalls bag 40 0) (bag.red + 40) = 3/4) ∧
  (probability (addBalls bag 14 4) (bag.red + 14) = 
   probability (addBalls bag 14 4) (bag.yellow + 4)) :=
by sorry

end bag_probabilities_l3578_357810


namespace zinc_copper_mixture_weight_l3578_357832

theorem zinc_copper_mixture_weight 
  (zinc_weight : Real) 
  (zinc_copper_ratio : Real) 
  (h1 : zinc_weight = 28.8) 
  (h2 : zinc_copper_ratio = 9 / 11) : 
  zinc_weight + (zinc_weight * (1 / zinc_copper_ratio)) = 64 := by
  sorry

end zinc_copper_mixture_weight_l3578_357832


namespace sets_intersection_empty_l3578_357863

-- Define the sets A, B, and C
def A : Set (ℝ × ℝ) := {p | p.2^2 - p.1 - 1 = 0}
def B : Set (ℝ × ℝ) := {p | 4*p.1^2 + 2*p.1 - 2*p.2 + 5 = 0}
def C (k b : ℕ) : Set (ℝ × ℝ) := {p | p.2 = k * p.1 + b}

-- State the theorem
theorem sets_intersection_empty :
  ∃! k b : ℕ, (A ∪ B) ∩ C k b = ∅ ∧ k = 1 ∧ b = 2 := by sorry

end sets_intersection_empty_l3578_357863


namespace online_store_prices_l3578_357862

/-- Represents the pricing structure for an online store --/
structure StorePricing where
  flatFee : ℝ
  commissionRate : ℝ

/-- Calculates the final price for a given store --/
def calculateFinalPrice (costPrice profit : ℝ) (store : StorePricing) : ℝ :=
  let sellingPrice := costPrice + profit
  sellingPrice + store.flatFee + store.commissionRate * sellingPrice

theorem online_store_prices (costPrice : ℝ) (profitRate : ℝ) 
    (storeA storeB storeC : StorePricing) : 
    costPrice = 18 ∧ 
    profitRate = 0.2 ∧
    storeA = { flatFee := 0, commissionRate := 0.2 } ∧
    storeB = { flatFee := 5, commissionRate := 0.1 } ∧
    storeC = { flatFee := 0, commissionRate := 0.15 } →
    let profit := profitRate * costPrice
    calculateFinalPrice costPrice profit storeA = 25.92 ∧
    calculateFinalPrice costPrice profit storeB = 28.76 ∧
    calculateFinalPrice costPrice profit storeC = 24.84 := by
  sorry

end online_store_prices_l3578_357862


namespace monotonic_increasing_not_implies_positive_derivative_l3578_357817

theorem monotonic_increasing_not_implies_positive_derivative :
  ∃ (f : ℝ → ℝ) (a b : ℝ), a < b ∧
    (∀ x y, a < x ∧ x < y ∧ y < b → f x ≤ f y) ∧
    ¬(∀ x, a < x ∧ x < b → (deriv f x) > 0) :=
by sorry

end monotonic_increasing_not_implies_positive_derivative_l3578_357817


namespace dove_flag_dimensions_l3578_357883

/-- Represents the shape of a dove on a square grid -/
structure DoveShape where
  area : ℝ
  perimeter_type : List String
  grid_type : String

/-- Represents the dimensions of a rectangular flag -/
structure FlagDimensions where
  length : ℝ
  height : ℝ

/-- Theorem: Given a dove shape with area 192 cm² on a square grid, 
    the flag dimensions are 24 cm × 16 cm -/
theorem dove_flag_dimensions 
  (dove : DoveShape) 
  (h1 : dove.area = 192) 
  (h2 : dove.perimeter_type = ["quarter-circle", "straight line"])
  (h3 : dove.grid_type = "square") :
  ∃ (flag : FlagDimensions), flag.length = 24 ∧ flag.height = 16 :=
by sorry

end dove_flag_dimensions_l3578_357883


namespace percentage_goldfish_special_food_l3578_357875

-- Define the parameters
def total_goldfish : ℕ := 50
def food_per_goldfish : ℚ := 3/2
def special_food_cost : ℚ := 3
def total_special_food_cost : ℚ := 45

-- Define the theorem
theorem percentage_goldfish_special_food :
  (((total_special_food_cost / special_food_cost) / food_per_goldfish) / total_goldfish) * 100 = 20 := by
  sorry

end percentage_goldfish_special_food_l3578_357875


namespace square_area_12m_l3578_357814

/-- The area of a square with side length 12 meters is 144 square meters. -/
theorem square_area_12m : 
  let side_length : ℝ := 12
  let area : ℝ := side_length ^ 2
  area = 144 := by sorry

end square_area_12m_l3578_357814


namespace floor_of_e_l3578_357808

theorem floor_of_e : ⌊Real.exp 1⌋ = 2 := by
  sorry

end floor_of_e_l3578_357808


namespace factorization_left_to_right_l3578_357852

theorem factorization_left_to_right (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end factorization_left_to_right_l3578_357852


namespace inequality_relation_to_line_l3578_357828

-- Define the line equation
def line_equation (x y a : ℝ) : Prop := x + (a - 1) * y + 3 = 0

-- Define the inequality
def inequality (x y a : ℝ) : Prop := x + (a - 1) * y + 3 > 0

-- Theorem statement
theorem inequality_relation_to_line :
  ∀ (a : ℝ), 
    (a > 1 → ∀ (x y : ℝ), inequality x y a → ¬(line_equation x y a)) ∧
    (a < 1 → ∀ (x y : ℝ), ¬(inequality x y a) → line_equation x y a) :=
by sorry

end inequality_relation_to_line_l3578_357828


namespace canoe_rental_cost_l3578_357860

/-- Represents the daily rental cost and quantities for canoes and kayaks --/
structure RentalInfo where
  canoe_cost : ℝ
  kayak_cost : ℝ
  canoe_count : ℕ
  kayak_count : ℕ

/-- Calculates the total revenue from canoe and kayak rentals --/
def total_revenue (r : RentalInfo) : ℝ :=
  r.canoe_cost * r.canoe_count + r.kayak_cost * r.kayak_count

/-- Theorem stating the canoe rental cost given the problem conditions --/
theorem canoe_rental_cost :
  ∀ (r : RentalInfo),
    r.kayak_cost = 15 →
    r.canoe_count = (3 * r.kayak_count) / 2 →
    total_revenue r = 288 →
    r.canoe_count = r.kayak_count + 4 →
    r.canoe_cost = 14 := by
  sorry

end canoe_rental_cost_l3578_357860


namespace cupcake_flour_requirement_l3578_357816

-- Define the given quantities
def total_flour : ℝ := 6
def flour_for_cakes : ℝ := 4
def flour_per_cake : ℝ := 0.5
def flour_for_cupcakes : ℝ := 2
def price_per_cake : ℝ := 2.5
def price_per_cupcake : ℝ := 1
def total_earnings : ℝ := 30

-- Define the theorem
theorem cupcake_flour_requirement :
  ∃ (flour_per_cupcake : ℝ),
    flour_per_cupcake * (flour_for_cupcakes / flour_per_cupcake) = 
      total_earnings - (flour_for_cakes / flour_per_cake) * price_per_cake ∧
    flour_per_cupcake = 0.2 := by
  sorry

end cupcake_flour_requirement_l3578_357816


namespace reciprocal_square_roots_l3578_357812

theorem reciprocal_square_roots (a b c d : ℂ) : 
  (a^4 - a^2 - 5 = 0 ∧ b^4 - b^2 - 5 = 0 ∧ c^4 - c^2 - 5 = 0 ∧ d^4 - d^2 - 5 = 0) →
  (5 * (1/a)^4 + (1/a)^2 - 1 = 0 ∧ 5 * (1/b)^4 + (1/b)^2 - 1 = 0 ∧
   5 * (1/c)^4 + (1/c)^2 - 1 = 0 ∧ 5 * (1/d)^4 + (1/d)^2 - 1 = 0) :=
by sorry

end reciprocal_square_roots_l3578_357812


namespace cosine_roots_of_equation_l3578_357809

theorem cosine_roots_of_equation : 
  let f (t : ℝ) := 32 * t^5 - 40 * t^3 + 10 * t - Real.sqrt 3
  (f (Real.cos (6 * π / 180)) = 0) →
  (f (Real.cos (78 * π / 180)) = 0) ∧
  (f (Real.cos (150 * π / 180)) = 0) ∧
  (f (Real.cos (222 * π / 180)) = 0) ∧
  (f (Real.cos (294 * π / 180)) = 0) :=
by sorry

end cosine_roots_of_equation_l3578_357809


namespace sum_of_digits_of_B_is_7_l3578_357857

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The property that the sum of digits of a number is congruent to the number itself modulo 9 -/
axiom sum_of_digits_mod_9 (n : ℕ) : sumOfDigits n ≡ n [ZMOD 9]

/-- A is the sum of digits of 4444^444 -/
def A : ℕ := sumOfDigits (4444^444)

/-- B is the sum of digits of A -/
def B : ℕ := sumOfDigits A

/-- The main theorem: the sum of digits of B is 7 -/
theorem sum_of_digits_of_B_is_7 : sumOfDigits B = 7 := by sorry

end sum_of_digits_of_B_is_7_l3578_357857


namespace vanessa_savings_time_l3578_357854

def dress_cost : ℕ := 120
def initial_savings : ℕ := 25
def weekly_allowance : ℕ := 30
def arcade_expense : ℕ := 15
def snack_expense : ℕ := 5

def weekly_savings : ℕ := weekly_allowance - arcade_expense - snack_expense

theorem vanessa_savings_time : 
  ∃ (weeks : ℕ), 
    weeks * weekly_savings + initial_savings ≥ dress_cost ∧ 
    (weeks - 1) * weekly_savings + initial_savings < dress_cost ∧
    weeks = 10 := by
  sorry

end vanessa_savings_time_l3578_357854


namespace f_neg_one_eq_three_l3578_357843

/-- Given a function f(x) = x^2 - 2x, prove that f(-1) = 3 -/
theorem f_neg_one_eq_three (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - 2*x) : f (-1) = 3 := by
  sorry

end f_neg_one_eq_three_l3578_357843


namespace shaded_area_in_grid_l3578_357870

/-- The area of a shape in a 3x3 grid formed by a 3x1 rectangle with one 1x1 square removed -/
theorem shaded_area_in_grid (grid_size : Nat) (square_side_length : ℝ) 
  (h1 : grid_size = 3) 
  (h2 : square_side_length = 1) : ℝ := by
  sorry

#check shaded_area_in_grid

end shaded_area_in_grid_l3578_357870


namespace triangle_vector_equality_l3578_357849

-- Define the triangle ABC and point D
structure Triangle :=
  (A B C D : ℝ × ℝ)

-- Define vectors m and n
def m (t : Triangle) : ℝ × ℝ := t.B - t.C
def n (t : Triangle) : ℝ × ℝ := t.D - t.C

-- State the theorem
theorem triangle_vector_equality (t : Triangle) 
  (h1 : t.D.1 = t.A.1 + (2/3) * (t.B.1 - t.A.1) ∧ t.D.2 = t.A.2 + (2/3) * (t.B.2 - t.A.2)) :
  t.A - t.C = -1/2 * (m t) + 3/2 * (n t) := by
  sorry

end triangle_vector_equality_l3578_357849


namespace factor_tree_value_l3578_357866

theorem factor_tree_value : ∀ (A B C D E : ℕ),
  A = B * C →
  B = 3 * D →
  D = 3 * 2 →
  C = 5 * E →
  E = 5 * 2 →
  A = 900 := by
  sorry

end factor_tree_value_l3578_357866


namespace learning_time_difference_l3578_357877

def hours_english : ℕ := 6
def hours_chinese : ℕ := 2
def hours_spanish : ℕ := 3
def hours_french : ℕ := 1

theorem learning_time_difference : 
  (hours_english + hours_chinese) - (hours_spanish + hours_french) = 4 := by
  sorry

end learning_time_difference_l3578_357877


namespace dinner_savings_l3578_357874

theorem dinner_savings (total_savings : ℝ) (individual_savings : ℝ) : 
  total_savings > 0 →
  individual_savings > 0 →
  total_savings = 2 * individual_savings →
  (3/4) * total_savings + 2 * (6 * 1.5 + 1) = total_savings →
  individual_savings = 40 := by
sorry

end dinner_savings_l3578_357874


namespace trigonometric_identities_l3578_357895

theorem trigonometric_identities :
  (Real.cos (780 * π / 180) = 1 / 2) ∧ 
  (Real.sin (-45 * π / 180) = -Real.sqrt 2 / 2) := by
  sorry

end trigonometric_identities_l3578_357895


namespace k_value_l3578_357826

/-- The function f(x) = 4x^2 + 3x + 5 -/
def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 5

/-- The function g(x) = x^2 + kx - 7 with parameter k -/
def g (k : ℝ) (x : ℝ) : ℝ := x^2 + k * x - 7

/-- Theorem stating that if f(5) - g(5) = 20, then k = 82/5 -/
theorem k_value (k : ℝ) : f 5 - g k 5 = 20 → k = 82 / 5 := by
  sorry

end k_value_l3578_357826


namespace area_FGCD_l3578_357897

/-- Represents a trapezoid ABCD with the given properties -/
structure Trapezoid where
  ab : ℝ
  cd : ℝ
  altitude : ℝ
  ab_positive : 0 < ab
  cd_positive : 0 < cd
  altitude_positive : 0 < altitude

/-- Theorem stating the area of quadrilateral FGCD in the given trapezoid -/
theorem area_FGCD (t : Trapezoid) (h1 : t.ab = 10) (h2 : t.cd = 26) (h3 : t.altitude = 15) :
  let fg := (t.ab + t.cd) / 2 - 5 / 2
  (fg + t.cd) / 2 * t.altitude = 311.25 := by sorry

end area_FGCD_l3578_357897


namespace thirteen_fifth_power_mod_seven_l3578_357827

theorem thirteen_fifth_power_mod_seven : (13^5 : ℤ) ≡ 6 [ZMOD 7] := by
  sorry

end thirteen_fifth_power_mod_seven_l3578_357827


namespace height_estimate_l3578_357834

/-- Given a survey of 1500 first-year high school students' heights:
    - The height range [160cm, 170cm] is divided into two groups of 5cm each
    - 'a' is the height of the histogram rectangle for [160cm, 165cm]
    - 'b' is the height of the histogram rectangle for [165cm, 170cm]
    - 1 unit of height in the histogram corresponds to 1500 students
    Then, the estimated number of students with heights in [160cm, 170cm] is 7500(a+b) -/
theorem height_estimate (a b : ℝ) : ℝ :=
  let total_students : ℕ := 1500
  let group_width : ℝ := 5
  let scale : ℝ := 1500
  7500 * (a + b)

#check height_estimate

end height_estimate_l3578_357834


namespace infinitely_many_perfect_squares_l3578_357896

theorem infinitely_many_perfect_squares (n k : ℕ+) : 
  ∃ (S : Set (ℕ+ × ℕ+)), Set.Infinite S ∧ 
  ∀ (pair : ℕ+ × ℕ+), pair ∈ S → 
  ∃ (m : ℕ), (pair.1 * 2^(pair.2.val) - 7 : ℤ) = m^2 := by
sorry

end infinitely_many_perfect_squares_l3578_357896


namespace chameleon_color_change_l3578_357846

theorem chameleon_color_change (total : ℕ) (blue_initial red_initial : ℕ) 
  (blue_final red_final : ℕ) (changed : ℕ) : 
  total = 140 →
  total = blue_initial + red_initial →
  total = blue_final + red_final →
  blue_initial = 5 * blue_final →
  red_final = 3 * red_initial →
  changed = blue_initial - blue_final →
  changed = 80 := by
sorry

end chameleon_color_change_l3578_357846


namespace toys_per_day_l3578_357825

def total_weekly_production : ℕ := 5505
def working_days_per_week : ℕ := 5

theorem toys_per_day :
  total_weekly_production / working_days_per_week = 1101 :=
by
  sorry

end toys_per_day_l3578_357825


namespace tic_tac_toe_tie_games_l3578_357880

theorem tic_tac_toe_tie_games 
  (amy_wins : ℚ) 
  (lily_wins : ℚ) 
  (h1 : amy_wins = 5 / 12) 
  (h2 : lily_wins = 1 / 4) : 
  1 - (amy_wins + lily_wins) = 1 / 3 := by
sorry

end tic_tac_toe_tie_games_l3578_357880


namespace ten_object_rotation_l3578_357844

/-- Represents a circular arrangement of n objects -/
def CircularArrangement (n : ℕ) := Fin n

/-- The operation of switching two objects in the arrangement -/
def switch (arr : CircularArrangement n) (i j : Fin n) : CircularArrangement n :=
  sorry

/-- Checks if the arrangement is rotated one position clockwise -/
def isRotatedOneStep (original rotated : CircularArrangement n) : Prop :=
  sorry

/-- The minimum number of switches required to rotate the arrangement one step -/
def minSwitches (n : ℕ) : ℕ :=
  sorry

theorem ten_object_rotation (arr : CircularArrangement 10) :
  ∃ (switches : List (Fin 10 × Fin 10)),
    switches.length = 9 ∧
    isRotatedOneStep arr (switches.foldl (λ a (i, j) => switch a i j) arr) :=
  sorry

end ten_object_rotation_l3578_357844


namespace cubic_minus_three_divisibility_l3578_357894

theorem cubic_minus_three_divisibility (n : ℕ) (h : n > 1) :
  (n - 1) ∣ (n^3 - 3) ↔ n = 2 ∨ n = 3 := by
  sorry

end cubic_minus_three_divisibility_l3578_357894


namespace smallest_divisor_partition_l3578_357886

/-- A function that returns the sum of divisors of a positive integer -/
def sumOfDivisors (n : ℕ+) : ℕ := sorry

/-- A function that checks if the divisors of a number can be partitioned into three sets with equal sums -/
def canPartitionDivisors (n : ℕ+) : Prop := sorry

/-- The theorem stating that 120 is the smallest positive integer with the required property -/
theorem smallest_divisor_partition :
  (∀ m : ℕ+, m < 120 → ¬(canPartitionDivisors m)) ∧ 
  (canPartitionDivisors 120) := by sorry

end smallest_divisor_partition_l3578_357886


namespace evaluate_expression_l3578_357892

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 2) :
  4 * x^(y + 1) + 5 * y^(x + 1) = 188 := by
  sorry

end evaluate_expression_l3578_357892


namespace gcd_204_85_l3578_357872

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l3578_357872


namespace imaginary_part_of_z_l3578_357847

theorem imaginary_part_of_z (z : ℂ) (h : z + (3 - 4*I) = 1) : z.im = 4 := by
  sorry

end imaginary_part_of_z_l3578_357847


namespace journey_speed_problem_l3578_357819

/-- Proves that given a journey of 3 km, if traveling at speed v km/hr results in arriving 7 minutes late, 
    and traveling at 12 km/hr results in arriving 8 minutes early, then v = 6 km/hr. -/
theorem journey_speed_problem (v : ℝ) : 
  (3 / v - 3 / 12 = 15 / 60) → v = 6 := by
  sorry

end journey_speed_problem_l3578_357819


namespace inequality_proof_l3578_357806

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1/a + 1/b + 1/c = a + b + c) :
  1/(2*a + b + c)^2 + 1/(2*b + c + a)^2 + 1/(2*c + a + b)^2 ≤ 3/16 := by
  sorry

end inequality_proof_l3578_357806


namespace expression_simplification_l3578_357882

theorem expression_simplification (x y z : ℝ) 
  (h_pos : 0 < z ∧ z < y ∧ y < x) : 
  (x^z * y^x * z^y) / (z^z * y^y * x^x) = (x/z)^(z-y) := by
  sorry

end expression_simplification_l3578_357882


namespace intersection_of_A_and_B_l3578_357802

-- Define the sets A and B
def A : Set ℝ := {x | x > 2 ∨ x < -1}
def B : Set ℝ := {x | (x + 1) * (4 - x) < 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | x > 3 ∨ x < -1} := by sorry

end intersection_of_A_and_B_l3578_357802


namespace composite_sum_l3578_357873

theorem composite_sum (a b c d : ℕ) (h1 : c > b) (h2 : a + b + c + d = a * b - c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a + c = x * y :=
sorry

end composite_sum_l3578_357873


namespace min_value_expression_l3578_357888

theorem min_value_expression (r s t : ℝ) 
  (h1 : 1 ≤ r) (h2 : r ≤ s) (h3 : s ≤ t) (h4 : t ≤ 4) :
  (r - 1)^2 + (s/r - 1)^2 + (t/s - 1)^2 + (4/t - 1)^2 ≥ 12 - 8 * Real.sqrt 2 :=
sorry

end min_value_expression_l3578_357888


namespace two_common_tangents_l3578_357881

-- Define the circles
def C1 (x y : ℝ) : Prop := (x - 5)^2 + (y - 3)^2 = 9
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 9 = 0

-- Define the number of common tangent lines
def num_common_tangents : ℕ := 2

-- Theorem statement
theorem two_common_tangents :
  num_common_tangents = 2 :=
sorry

end two_common_tangents_l3578_357881


namespace women_half_of_total_l3578_357821

/-- Represents the number of bones in different types of skeletons -/
structure BoneCount where
  woman : ℕ
  man : ℕ
  child : ℕ

/-- Represents the count of different types of skeletons -/
structure SkeletonCount where
  women : ℕ
  men : ℕ
  children : ℕ

theorem women_half_of_total (bc : BoneCount) (sc : SkeletonCount) : 
  bc.woman = 20 →
  bc.man = bc.woman + 5 →
  bc.child = bc.woman / 2 →
  sc.men = sc.children →
  sc.women + sc.men + sc.children = 20 →
  bc.woman * sc.women + bc.man * sc.men + bc.child * sc.children = 375 →
  2 * sc.women = sc.women + sc.men + sc.children := by
  sorry

#check women_half_of_total

end women_half_of_total_l3578_357821


namespace rectangle_width_length_ratio_l3578_357829

/-- Given a rectangle with width w, length 10, and perimeter 30, 
    prove that the ratio of width to length is 1:2 -/
theorem rectangle_width_length_ratio 
  (w : ℝ) 
  (h1 : w > 0)
  (h2 : 2 * w + 2 * 10 = 30) : 
  w / 10 = 1 / 2 := by
  sorry

end rectangle_width_length_ratio_l3578_357829


namespace olya_always_wins_l3578_357890

/-- Represents an archipelago with a given number of islands -/
structure Archipelago where
  num_islands : Nat
  connections : List (Nat × Nat)

/-- Represents a game played on an archipelago -/
inductive GameResult
  | OlyaWins
  | MaximWins

/-- The game played by Olya and Maxim on the archipelago -/
def play_game (a : Archipelago) : GameResult :=
  sorry

/-- Theorem stating that Olya always wins the game on an archipelago with 2009 islands -/
theorem olya_always_wins :
  ∀ (a : Archipelago), a.num_islands = 2009 → play_game a = GameResult.OlyaWins :=
sorry

end olya_always_wins_l3578_357890


namespace triangle_problem_l3578_357869

theorem triangle_problem (A B C : Real) (a b c : Real) :
  B = 2 * C →
  c = 2 →
  a = 1 →
  b = Real.sqrt 6 ∧
  Real.sin (2 * B - π / 3) = (7 * Real.sqrt 3 - Real.sqrt 15) / 16 := by
  sorry

end triangle_problem_l3578_357869


namespace complex_fraction_equality_l3578_357815

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (hab : a + b ≠ 0) (h : a^3 + a^2*b + a*b^2 + b^3 = 0) : 
  (a^12 + b^12) / (a + b)^12 = 2/81 := by
sorry

end complex_fraction_equality_l3578_357815


namespace middle_angle_range_l3578_357867

theorem middle_angle_range (α β γ : Real) : 
  (0 ≤ α) → (0 ≤ β) → (0 ≤ γ) →  -- angles are non-negative
  (α + β + γ = 180) →             -- sum of angles in a triangle
  (α ≤ β) → (β ≤ γ) →             -- β is the middle angle
  (0 < β) ∧ (β < 90) :=           -- conclusion
by sorry

end middle_angle_range_l3578_357867


namespace max_min_f_on_interval_l3578_357811

def f (x : ℝ) := x^3 - 3*x + 1

theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-3) 0, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-3) 0, f x = max) ∧
    (∀ x ∈ Set.Icc (-3) 0, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-3) 0, f x = min) ∧
    max = 3 ∧ min = -17 := by
  sorry

end max_min_f_on_interval_l3578_357811


namespace marcus_pebbles_l3578_357868

theorem marcus_pebbles (P : ℕ) : 
  P / 2 + 30 = 39 → P = 18 := by
  sorry

end marcus_pebbles_l3578_357868


namespace gcd_problem_l3578_357820

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 887 * (2 * k + 1)) :
  Int.gcd (3 * b^2 + 47 * b + 91) (b + 17) = 3 := by
  sorry

end gcd_problem_l3578_357820


namespace myfavorite_sum_l3578_357800

def letters : Finset Char := {'m', 'y', 'f', 'a', 'v', 'o', 'r', 'i', 't', 'e'}
def digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem myfavorite_sum (f : Char → Nat) 
  (h1 : Function.Bijective f)
  (h2 : ∀ c ∈ letters, f c ∈ digits) :
  (letters.sum fun c => f c) = 45 := by
  sorry

end myfavorite_sum_l3578_357800


namespace dan_bought_18_stickers_l3578_357839

/-- The number of stickers Dan bought -/
def stickers_bought (initial_stickers : ℕ) : ℕ := 18

theorem dan_bought_18_stickers (initial_stickers : ℕ) :
  let cindy_remaining := initial_stickers - 15
  let dan_total := initial_stickers + stickers_bought initial_stickers
  dan_total = cindy_remaining + 33 :=
by
  sorry

end dan_bought_18_stickers_l3578_357839


namespace joggers_meet_time_l3578_357803

def lap_times : List Nat := [3, 5, 9, 10]

def start_time : Nat := 7 * 60  -- 7:00 AM in minutes since midnight

theorem joggers_meet_time (lcm_result : Nat) 
  (h1 : lcm_result = Nat.lcm (Nat.lcm (Nat.lcm 3 5) 9) 10)
  (h2 : ∀ t ∈ lap_times, lcm_result % t = 0)
  (h3 : ∀ m : Nat, (∀ t ∈ lap_times, m % t = 0) → m ≥ lcm_result) :
  (start_time + lcm_result) % (24 * 60) = 8 * 60 + 30 := by sorry

end joggers_meet_time_l3578_357803


namespace second_boy_speed_l3578_357899

/-- Given two boys walking in the same direction for 7 hours, with the first boy
    walking at 4 kmph and ending up 10.5 km apart, prove that the speed of the
    second boy is 5.5 kmph. -/
theorem second_boy_speed (v : ℝ) 
  (h1 : (v - 4) * 7 = 10.5) : v = 5.5 := by
  sorry

end second_boy_speed_l3578_357899


namespace intercept_sum_l3578_357833

theorem intercept_sum (m : ℕ) (x_0 y_0 : ℕ) : m = 17 →
  (2 * x_0) % m = 3 →
  (5 * y_0) % m = m - 3 →
  x_0 < m →
  y_0 < m →
  x_0 + y_0 = 22 := by
  sorry

end intercept_sum_l3578_357833


namespace units_digit_sum_of_powers_l3578_357813

theorem units_digit_sum_of_powers : ∃ n : ℕ, n < 10 ∧ (35^87 + 93^53) % 10 = n ∧ n = 8 := by
  sorry

end units_digit_sum_of_powers_l3578_357813


namespace price_reduction_achieves_profit_l3578_357850

/-- Represents the store's sales and pricing data -/
structure StoreSales where
  initial_cost : ℝ
  initial_price : ℝ
  january_sales : ℝ
  march_sales : ℝ
  sales_increase_per_yuan : ℝ
  desired_profit : ℝ

/-- Calculates the required price reduction to achieve the desired profit -/
def calculate_price_reduction (s : StoreSales) : ℝ :=
  sorry

/-- Theorem stating that the calculated price reduction achieves the desired profit -/
theorem price_reduction_achieves_profit (s : StoreSales) 
  (h1 : s.initial_cost = 25)
  (h2 : s.initial_price = 40)
  (h3 : s.january_sales = 256)
  (h4 : s.march_sales = 400)
  (h5 : s.sales_increase_per_yuan = 5)
  (h6 : s.desired_profit = 4250) :
  let y := calculate_price_reduction s
  (s.initial_price - y - s.initial_cost) * (s.march_sales + s.sales_increase_per_yuan * y) = s.desired_profit :=
by sorry

end price_reduction_achieves_profit_l3578_357850


namespace arithmetic_progression_sum_l3578_357840

theorem arithmetic_progression_sum (a b : ℝ) : 
  0 < a ∧ 0 < b ∧ 
  4 < a ∧ a < b ∧ b < 16 ∧
  (b - a = a - 4) ∧ 
  (16 - b = b - a) ∧
  (b - a ≠ a - 4) →
  a + b = 20 := by sorry

end arithmetic_progression_sum_l3578_357840


namespace triangular_array_coins_l3578_357818

-- Define the sum of the first N natural numbers
def triangular_sum (N : ℕ) : ℕ := N * (N + 1) / 2

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem triangular_array_coins :
  ∃ N : ℕ, triangular_sum N = 5050 ∧ sum_of_digits N = 1 :=
sorry

end triangular_array_coins_l3578_357818


namespace stratified_sample_distribution_l3578_357893

/-- Represents the number of students in each grade -/
structure GradeDistribution where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Calculates the total number of students -/
def totalStudents (d : GradeDistribution) : ℕ :=
  d.grade10 + d.grade11 + d.grade12

/-- Represents the sample size for each grade -/
structure SampleDistribution where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Calculates the total sample size -/
def totalSample (s : SampleDistribution) : ℕ :=
  s.grade10 + s.grade11 + s.grade12

theorem stratified_sample_distribution 
  (population : GradeDistribution)
  (sample : SampleDistribution) :
  totalStudents population = 4000 →
  population.grade10 = 32 * k →
  population.grade11 = 33 * k →
  population.grade12 = 35 * k →
  totalSample sample = 200 →
  sample.grade10 = 64 ∧ sample.grade11 = 66 ∧ sample.grade12 = 70 :=
by sorry


end stratified_sample_distribution_l3578_357893


namespace fraction_to_decimal_l3578_357855

theorem fraction_to_decimal : (13 : ℚ) / 200 = (52 : ℚ) / 100 := by sorry

end fraction_to_decimal_l3578_357855


namespace equilateral_triangle_from_inscribed_circles_l3578_357889

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- The angles of the triangle -/
  angles : Fin 3 → ℝ
  /-- Sum of angles is 180° -/
  sum_angles : (angles 0) + (angles 1) + (angles 2) = π
  /-- All angles are positive -/
  all_positive : ∀ i, 0 < angles i

/-- Represents the process of inscribing circles and forming new triangles -/
def inscribe_circle (t : TriangleWithInscribedCircle) : TriangleWithInscribedCircle :=
  sorry

/-- The theorem to be proved -/
theorem equilateral_triangle_from_inscribed_circles 
  (t : TriangleWithInscribedCircle) : 
  (∀ i, (inscribe_circle (inscribe_circle t)).angles i = t.angles i) → 
  (∀ i, t.angles i = π / 3) :=
sorry

end equilateral_triangle_from_inscribed_circles_l3578_357889


namespace quadrilateral_area_theorem_l3578_357876

noncomputable def quadrilateral_area (P Q A B : ℝ × ℝ) : ℝ :=
  sorry

theorem quadrilateral_area_theorem (P Q A B : ℝ × ℝ) :
  let d := 3 -- distance between P and Q
  let r1 := Real.sqrt 3 -- radius of circle centered at P
  let r2 := 3 -- radius of circle centered at Q
  dist P Q = d ∧
  dist P A = r1 ∧
  dist Q A = r2 ∧
  dist P B = r1 ∧
  dist Q B = r2
  →
  quadrilateral_area P Q A B = (3 * Real.sqrt 5) / 2 := by
  sorry

end quadrilateral_area_theorem_l3578_357876


namespace polynomial_rational_difference_l3578_357801

theorem polynomial_rational_difference (f : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →
  (∀ x y : ℝ, ∃ q : ℚ, x - y = q → ∃ r : ℚ, f x - f y = r) →
  ∃ b : ℚ, ∃ c : ℝ, ∀ x, f x = b * x + c :=
sorry

end polynomial_rational_difference_l3578_357801


namespace tan_product_special_angles_l3578_357884

theorem tan_product_special_angles : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) = 2 * Real.sqrt 7 := by
  sorry

end tan_product_special_angles_l3578_357884


namespace trig_simplification_l3578_357842

theorem trig_simplification :
  (1 + Real.cos (20 * π / 180)) / (2 * Real.sin (20 * π / 180)) -
  Real.sin (10 * π / 180) * ((1 / Real.tan (5 * π / 180)) - Real.tan (5 * π / 180)) =
  Real.sqrt 3 / 2 := by sorry

end trig_simplification_l3578_357842


namespace lidia_apps_to_buy_l3578_357853

-- Define the given conditions
def average_app_cost : ℕ := 4
def total_budget : ℕ := 66
def remaining_money : ℕ := 6

-- Define the number of apps to buy
def apps_to_buy : ℕ := (total_budget - remaining_money) / average_app_cost

-- Theorem statement
theorem lidia_apps_to_buy : apps_to_buy = 15 := by
  sorry

end lidia_apps_to_buy_l3578_357853
