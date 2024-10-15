import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l1186_118662

theorem inequality_proof (α x y z : ℝ) (hα : α > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x * y + y * z + z * x = α) :
  (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ (1/3) * (x / z + z / x + 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1186_118662


namespace NUMINAMATH_CALUDE_two_digit_number_digit_difference_l1186_118649

/-- Given a two-digit number where the difference between the original number
    and the number with interchanged digits is 45, prove that the difference
    between its two digits is 5. -/
theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 45 → x - y = 5 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_digit_difference_l1186_118649


namespace NUMINAMATH_CALUDE_certain_number_minus_32_l1186_118635

theorem certain_number_minus_32 (x : ℤ) (h : x - 48 = 22) : x - 32 = 38 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_minus_32_l1186_118635


namespace NUMINAMATH_CALUDE_well_capacity_1200_gallons_l1186_118661

/-- The capacity of a well filled by two pipes -/
def well_capacity (rate1 rate2 : ℝ) (time : ℝ) : ℝ :=
  (rate1 + rate2) * time

/-- Theorem stating the capacity of the well -/
theorem well_capacity_1200_gallons (rate1 rate2 time : ℝ) 
  (h1 : rate1 = 48)
  (h2 : rate2 = 192)
  (h3 : time = 5) :
  well_capacity rate1 rate2 time = 1200 := by
  sorry

end NUMINAMATH_CALUDE_well_capacity_1200_gallons_l1186_118661


namespace NUMINAMATH_CALUDE_jose_painting_time_l1186_118616

/-- The time it takes for Alex to paint a car alone -/
def alex_time : ℝ := 5

/-- The time it takes for Jose and Alex to paint a car together -/
def combined_time : ℝ := 2.91666666667

/-- The time it takes for Jose to paint a car alone -/
def jose_time : ℝ := 7

/-- Theorem stating that given Alex's time and the combined time, Jose's time is 7 days -/
theorem jose_painting_time : 
  1 / alex_time + 1 / jose_time = 1 / combined_time :=
sorry

end NUMINAMATH_CALUDE_jose_painting_time_l1186_118616


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l1186_118675

theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y : ℝ) :
  S.card = 60 →
  x ∈ S →
  y ∈ S →
  x = 50 →
  y = 65 →
  (S.sum id) / S.card = 42 →
  ((S.sum id) - x - y) / (S.card - 2) = 41.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l1186_118675


namespace NUMINAMATH_CALUDE_solutions_of_equation_1_sum_of_reciprocals_squared_difference_of_solutions_l1186_118600

-- Question 1
theorem solutions_of_equation_1 (x : ℝ) :
  (x + 5 / x = -6) ↔ (x = -1 ∨ x = -5) :=
sorry

-- Question 2
theorem sum_of_reciprocals (m n : ℝ) :
  (m - 3 / m = 4) ∧ (n - 3 / n = 4) → 1 / m + 1 / n = -4 / 3 :=
sorry

-- Question 3
theorem squared_difference_of_solutions (a : ℝ) (x₁ x₂ : ℝ) :
  a ≠ 0 →
  (x₁ + (a^2 + 2*a) / (x₁ + 1) = 2*a + 1) →
  (x₂ + (a^2 + 2*a) / (x₂ + 1) = 2*a + 1) →
  (x₁ - x₂)^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_solutions_of_equation_1_sum_of_reciprocals_squared_difference_of_solutions_l1186_118600


namespace NUMINAMATH_CALUDE_recommendation_plans_count_l1186_118639

/-- The number of universities --/
def num_universities : ℕ := 3

/-- The number of students to be recommended --/
def num_students : ℕ := 4

/-- The maximum number of students a university can accept --/
def max_students_per_university : ℕ := 2

/-- The function that calculates the number of recommendation plans --/
noncomputable def num_recommendation_plans : ℕ := sorry

/-- Theorem stating that the number of recommendation plans is 54 --/
theorem recommendation_plans_count : num_recommendation_plans = 54 := by sorry

end NUMINAMATH_CALUDE_recommendation_plans_count_l1186_118639


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l1186_118602

/-- The ratio of volumes of cylinders formed from a rectangle --/
theorem cylinder_volume_ratio (w h : ℝ) (hw : w = 9) (hh : h = 12) :
  let v1 := π * (w / (2 * π))^2 * h
  let v2 := π * (h / (2 * π))^2 * w
  max v1 v2 / min v1 v2 = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l1186_118602


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l1186_118653

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The curve xy = 2 -/
def on_curve (p : Point) : Prop :=
  p.x * p.y = 2

/-- The circle that intersects the curve at four points -/
structure Circle where
  center : Point
  radius : ℝ

/-- A point lies on the circle -/
def on_circle (c : Circle) (p : Point) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The theorem stating the fourth intersection point -/
theorem fourth_intersection_point (c : Circle) 
    (h1 : on_curve ⟨4, 1/2⟩ ∧ on_circle c ⟨4, 1/2⟩)
    (h2 : on_curve ⟨-2, -1⟩ ∧ on_circle c ⟨-2, -1⟩)
    (h3 : on_curve ⟨1/4, 8⟩ ∧ on_circle c ⟨1/4, 8⟩)
    (h4 : ∃ p, on_curve p ∧ on_circle c p ∧ p ≠ ⟨4, 1/2⟩ ∧ p ≠ ⟨-2, -1⟩ ∧ p ≠ ⟨1/4, 8⟩) :
    ∃ p, p = ⟨-1/8, -16⟩ ∧ on_curve p ∧ on_circle c p :=
sorry

end NUMINAMATH_CALUDE_fourth_intersection_point_l1186_118653


namespace NUMINAMATH_CALUDE_uniform_cost_calculation_l1186_118621

/-- Calculates the total cost of uniforms for a student --/
def uniformCost (
  numUniforms : ℕ
) (
  pantsCost : ℚ
) (
  shirtCostMultiplier : ℚ
) (
  tieCostFraction : ℚ
) (
  socksCost : ℚ
) (
  jacketCostMultiplier : ℚ
) (
  shoesCost : ℚ
) (
  discountRate : ℚ
) (
  discountThreshold : ℕ
) : ℚ :=
  sorry

theorem uniform_cost_calculation :
  uniformCost 5 20 2 (1/5) 3 3 40 (1/10) 3 = 1039.5 := by
  sorry

end NUMINAMATH_CALUDE_uniform_cost_calculation_l1186_118621


namespace NUMINAMATH_CALUDE_sequence_periodicity_l1186_118678

/-- A cubic polynomial with rational coefficients -/
def CubicPolynomial (α : Type) [Field α] := α → α

/-- A sequence of rational numbers -/
def RationalSequence := ℕ → ℚ

/-- The statement that a sequence satisfies q_n = p(q_{n+1}) for all positive n -/
def SatisfiesRelation (p : CubicPolynomial ℚ) (q : RationalSequence) :=
  ∀ n : ℕ, q n = p (q (n + 1))

/-- The theorem stating the existence of a period for the sequence -/
theorem sequence_periodicity
  (p : CubicPolynomial ℚ)
  (q : RationalSequence)
  (h : SatisfiesRelation p q) :
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, q (n + k) = q n :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l1186_118678


namespace NUMINAMATH_CALUDE_range_of_t_l1186_118641

-- Define the solution set
def solution_set : Set ℤ := {1, 2, 3}

-- Define the inequality condition
def inequality_condition (t : ℝ) (x : ℤ) : Prop :=
  |3 * (x : ℝ) + t| < 4

-- Define the main theorem
theorem range_of_t :
  ∀ t : ℝ,
  (∀ x : ℤ, x ∈ solution_set ↔ inequality_condition t x) →
  -7 < t ∧ t < -5 :=
sorry

end NUMINAMATH_CALUDE_range_of_t_l1186_118641


namespace NUMINAMATH_CALUDE_job_completion_time_l1186_118692

theorem job_completion_time (a_time b_time : ℕ) (remaining_fraction : ℚ) : 
  a_time = 15 → b_time = 20 → remaining_fraction = 8/15 →
  ∃ (days_worked : ℕ), days_worked = 4 ∧
    (1 - remaining_fraction) = days_worked * (1/a_time + 1/b_time) :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_l1186_118692


namespace NUMINAMATH_CALUDE_farm_animals_difference_l1186_118631

theorem farm_animals_difference : 
  ∀ (pigs dogs sheep : ℕ), 
    pigs = 42 → 
    sheep = 48 → 
    pigs = dogs → 
    pigs + dogs - sheep = 36 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_difference_l1186_118631


namespace NUMINAMATH_CALUDE_tan_alpha_sqrt_three_l1186_118605

theorem tan_alpha_sqrt_three (α : Real) (h : ∃ (x y : Real), x = 1 ∧ y = Real.sqrt 3 ∧ Real.cos α = x / Real.sqrt (x^2 + y^2) ∧ Real.sin α = y / Real.sqrt (x^2 + y^2)) : 
  Real.tan α = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_sqrt_three_l1186_118605


namespace NUMINAMATH_CALUDE_duplicated_page_number_l1186_118664

/-- The sum of natural numbers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem statement -/
theorem duplicated_page_number :
  ∀ k : ℕ,
  (k ≤ 70) →
  (sum_to_n 70 + k = 2550) →
  (k = 65) :=
by sorry

end NUMINAMATH_CALUDE_duplicated_page_number_l1186_118664


namespace NUMINAMATH_CALUDE_unique_prime_product_l1186_118677

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem unique_prime_product :
  ∀ n : ℕ,
  n ≠ 2103 →
  (∃ p q r : ℕ, is_prime p ∧ is_prime q ∧ is_prime r ∧ distinct p q r ∧ n = p * q * r) →
  ¬(∃ p1 p2 p3 : ℕ, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ distinct p1 p2 p3 ∧ p1 + p2 + p3 = 59) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_product_l1186_118677


namespace NUMINAMATH_CALUDE_tangent_to_exponential_l1186_118650

theorem tangent_to_exponential (k : ℝ) :
  (∃ x : ℝ, k * x = Real.exp x ∧ k = Real.exp x) → k = Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_to_exponential_l1186_118650


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l1186_118607

theorem circle_equation_k_value (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 14*x + y^2 + 8*y - k = 0 ↔ (x + 7)^2 + (y + 4)^2 = 25) → 
  k = 40 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l1186_118607


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1186_118680

theorem quadratic_inequality_solution (c : ℝ) : 
  (∀ x : ℝ, x^2 + x - c < 0 ↔ -2 < x ∧ x < 1) → c = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1186_118680


namespace NUMINAMATH_CALUDE_expression_value_l1186_118644

theorem expression_value (a b : ℝ) (h : a + b = 1) :
  a^3 + b^3 + 3*(a^3*b + a*b^3) + 6*(a^3*b^2 + a^2*b^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1186_118644


namespace NUMINAMATH_CALUDE_young_inequality_l1186_118655

theorem young_inequality (p q a b : ℝ) : 
  0 < p → 0 < q → 1 / p + 1 / q = 1 → 0 < a → 0 < b →
  a * b ≤ a^p / p + b^q / q := by
  sorry

end NUMINAMATH_CALUDE_young_inequality_l1186_118655


namespace NUMINAMATH_CALUDE_age_difference_l1186_118660

/-- Given the ages of Mehki, Jordyn, and Zrinka, prove that Mehki is 10 years older than Jordyn. -/
theorem age_difference (mehki_age jordyn_age zrinka_age : ℕ) 
  (h1 : jordyn_age = 2 * zrinka_age)
  (h2 : zrinka_age = 6)
  (h3 : mehki_age = 22) :
  mehki_age - jordyn_age = 10 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l1186_118660


namespace NUMINAMATH_CALUDE_room_area_l1186_118667

theorem room_area (breadth length : ℝ) : 
  length = 3 * breadth →
  2 * (length + breadth) = 16 →
  length * breadth = 12 := by
sorry

end NUMINAMATH_CALUDE_room_area_l1186_118667


namespace NUMINAMATH_CALUDE_completing_square_result_l1186_118689

theorem completing_square_result (x : ℝ) : 
  x^2 - 4*x - 1 = 0 → (x - 2)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l1186_118689


namespace NUMINAMATH_CALUDE_isosceles_triangle_leg_range_l1186_118622

theorem isosceles_triangle_leg_range (x : ℝ) : 
  (∃ (base : ℝ), base > 0 ∧ x + x + base = 10 ∧ x + x > base ∧ x + base > x) ↔ 
  (5/2 < x ∧ x < 5) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_leg_range_l1186_118622


namespace NUMINAMATH_CALUDE_percentage_men_correct_l1186_118687

/-- The percentage of men in a college class. -/
def percentage_men : ℝ := 40

theorem percentage_men_correct :
  let women_science_percentage : ℝ := 30
  let non_science_percentage : ℝ := 60
  let men_science_percentage : ℝ := 55.00000000000001
  let women_percentage : ℝ := 100 - percentage_men
  let science_percentage : ℝ := 100 - non_science_percentage
  (women_science_percentage / 100 * women_percentage + 
   men_science_percentage / 100 * percentage_men = science_percentage) ∧
  (percentage_men ≥ 0 ∧ percentage_men ≤ 100) :=
by sorry

end NUMINAMATH_CALUDE_percentage_men_correct_l1186_118687


namespace NUMINAMATH_CALUDE_employed_females_percentage_l1186_118699

theorem employed_females_percentage (total_population : ℝ) 
  (employed_percentage : ℝ) (employed_males_percentage : ℝ) :
  employed_percentage = 96 →
  employed_males_percentage = 24 →
  (employed_percentage - employed_males_percentage) / employed_percentage * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l1186_118699


namespace NUMINAMATH_CALUDE_solve_for_y_l1186_118603

theorem solve_for_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : y = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1186_118603


namespace NUMINAMATH_CALUDE_valid_placements_count_l1186_118612

/-- Represents a grid with rows and columns -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents the placement of crosses in a grid -/
structure CrossPlacement :=
  (grid : Grid)
  (num_crosses : ℕ)

/-- Counts the number of valid cross placements in a grid -/
def count_valid_placements (cp : CrossPlacement) : ℕ :=
  sorry

/-- The specific grid and cross placement for our problem -/
def our_problem : CrossPlacement :=
  { grid := { rows := 3, cols := 4 },
    num_crosses := 4 }

/-- Theorem stating that the number of valid placements for our problem is 36 -/
theorem valid_placements_count :
  count_valid_placements our_problem = 36 :=
sorry

end NUMINAMATH_CALUDE_valid_placements_count_l1186_118612


namespace NUMINAMATH_CALUDE_man_downstream_speed_l1186_118609

/-- Calculates the downstream speed of a person given their upstream speed and the stream speed. -/
def downstream_speed (upstream_speed stream_speed : ℝ) : ℝ :=
  upstream_speed + 2 * stream_speed

/-- Theorem: Given a man's upstream speed of 8 km/h and a stream speed of 1 km/h, his downstream speed is 10 km/h. -/
theorem man_downstream_speed :
  let upstream_speed : ℝ := 8
  let stream_speed : ℝ := 1
  downstream_speed upstream_speed stream_speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_man_downstream_speed_l1186_118609


namespace NUMINAMATH_CALUDE_quadratic_polynomial_special_roots_l1186_118643

theorem quadratic_polynomial_special_roots (p q : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + p*x + q
  (∃ α β : ℝ, (f α = 0 ∧ f β = 0) ∧ 
   ((α = f 0 ∧ β = f 1) ∨ (α = f 1 ∧ β = f 0))) →
  f 6 = 71/2 - p := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_special_roots_l1186_118643


namespace NUMINAMATH_CALUDE_transformed_function_point_l1186_118613

def f : ℝ → ℝ := fun _ ↦ 8

theorem transformed_function_point (h : f 3 = 8) :
  let g : ℝ → ℝ := fun x ↦ 2 * (4 * f (3 * x - 1) + 6)
  g 2 = 38 ∧ 2 + 19 = 21 := by
  sorry

end NUMINAMATH_CALUDE_transformed_function_point_l1186_118613


namespace NUMINAMATH_CALUDE_pet_shop_total_l1186_118658

-- Define the number of each type of animal
def num_kittens : ℕ := 32
def num_hamsters : ℕ := 15
def num_birds : ℕ := 30

-- Define the total number of animals
def total_animals : ℕ := num_kittens + num_hamsters + num_birds

-- Theorem to prove
theorem pet_shop_total : total_animals = 77 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_total_l1186_118658


namespace NUMINAMATH_CALUDE_two_trains_problem_l1186_118618

/-- The problem of two trains approaching each other -/
theorem two_trains_problem (length1 length2 speed1 clear_time : ℝ) 
  (h1 : length1 = 120)
  (h2 : length2 = 300)
  (h3 : speed1 = 42)
  (h4 : clear_time = 20.99832013438925) : 
  ∃ speed2 : ℝ, speed2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_two_trains_problem_l1186_118618


namespace NUMINAMATH_CALUDE_magazine_boxes_l1186_118623

theorem magazine_boxes (total_magazines : ℕ) (magazines_per_box : ℚ) : 
  total_magazines = 150 → magazines_per_box = 11.5 → 
  ⌈(total_magazines : ℚ) / magazines_per_box⌉ = 14 := by
  sorry

end NUMINAMATH_CALUDE_magazine_boxes_l1186_118623


namespace NUMINAMATH_CALUDE_f_min_at_neg_three_l1186_118698

/-- The function f(x) = x^2 + 6x + 1 -/
def f (x : ℝ) : ℝ := x^2 + 6*x + 1

/-- Theorem stating that f(x) is minimized when x = -3 -/
theorem f_min_at_neg_three :
  ∀ x : ℝ, f (-3) ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_f_min_at_neg_three_l1186_118698


namespace NUMINAMATH_CALUDE_min_value_implies_a_equals_two_l1186_118694

theorem min_value_implies_a_equals_two (x y a : ℝ) :
  x + 3*y + 5 ≥ 0 →
  x + y - 1 ≤ 0 →
  x + a ≥ 0 →
  (∀ x' y', x' + 3*y' + 5 ≥ 0 → x' + y' - 1 ≤ 0 → x' + 2*y' ≥ x + 2*y) →
  x + 2*y = -4 →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_equals_two_l1186_118694


namespace NUMINAMATH_CALUDE_sqrt_40_div_sqrt_5_l1186_118683

theorem sqrt_40_div_sqrt_5 : Real.sqrt 40 / Real.sqrt 5 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_40_div_sqrt_5_l1186_118683


namespace NUMINAMATH_CALUDE_exists_winning_strategy_l1186_118666

/-- Represents a card in the game -/
structure Card where
  id : Nat
  deriving Repr

/-- Represents the state of the game -/
structure GameState where
  player1_cards : List Card
  player2_cards : List Card
  deriving Repr

/-- Represents the strength relationship between cards -/
def beats (card1 card2 : Card) : Bool := sorry

/-- Represents a single turn in the game -/
def play_turn (state : GameState) : GameState := sorry

/-- Represents the strategy chosen by the players -/
def strategy (state : GameState) : GameState := sorry

/-- Theorem stating that there exists a strategy to end the game -/
theorem exists_winning_strategy 
  (n : Nat) 
  (initial_state : GameState) 
  (h1 : initial_state.player1_cards.length + initial_state.player2_cards.length = n) 
  (h2 : ∀ c1 c2 : Card, c1 ≠ c2 → (beats c1 c2 ∨ beats c2 c1)) :
  ∃ (final_state : GameState), 
    (final_state.player1_cards.length = 0 ∨ final_state.player2_cards.length = 0) ∧
    (∃ k : Nat, (strategy^[k]) initial_state = final_state) :=
sorry

end NUMINAMATH_CALUDE_exists_winning_strategy_l1186_118666


namespace NUMINAMATH_CALUDE_melody_cutouts_l1186_118672

/-- Given that Melody planned to paste 4 cut-outs on each card and made 6 cards in total,
    prove that the total number of cut-outs she made is 24. -/
theorem melody_cutouts (cutouts_per_card : ℕ) (total_cards : ℕ) 
  (h1 : cutouts_per_card = 4) 
  (h2 : total_cards = 6) : 
  cutouts_per_card * total_cards = 24 := by
  sorry

end NUMINAMATH_CALUDE_melody_cutouts_l1186_118672


namespace NUMINAMATH_CALUDE_find_other_number_l1186_118663

theorem find_other_number (A B : ℕ+) (h1 : Nat.lcm A B = 2310) (h2 : Nat.gcd A B = 83) (h3 : A = 210) : B = 913 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l1186_118663


namespace NUMINAMATH_CALUDE_min_sum_squares_with_real_root_l1186_118629

theorem min_sum_squares_with_real_root (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → a^2 + b^2 ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_with_real_root_l1186_118629


namespace NUMINAMATH_CALUDE_count_valid_numbers_l1186_118690

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ n = 34 * (n / 100 + (n / 10 % 10) + (n % 10))

theorem count_valid_numbers :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_valid_number n) ∧ S.card = 4 ∧
  (∀ m : ℕ, is_valid_number m → m ∈ S) :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l1186_118690


namespace NUMINAMATH_CALUDE_distribute_negative_three_l1186_118679

theorem distribute_negative_three (x y : ℝ) : -3 * (x - x * y) = -3 * x + 3 * x * y := by
  sorry

end NUMINAMATH_CALUDE_distribute_negative_three_l1186_118679


namespace NUMINAMATH_CALUDE_total_pure_acid_in_mixture_l1186_118628

def solution1_concentration : ℝ := 0.20
def solution1_volume : ℝ := 8
def solution2_concentration : ℝ := 0.35
def solution2_volume : ℝ := 5

theorem total_pure_acid_in_mixture :
  let pure_acid1 := solution1_concentration * solution1_volume
  let pure_acid2 := solution2_concentration * solution2_volume
  pure_acid1 + pure_acid2 = 3.35 := by sorry

end NUMINAMATH_CALUDE_total_pure_acid_in_mixture_l1186_118628


namespace NUMINAMATH_CALUDE_max_glow_count_max_glow_count_for_given_conditions_l1186_118656

/-- The maximum number of times a light can glow in a given time range -/
theorem max_glow_count (total_duration : ℕ) (glow_interval : ℕ) : ℕ :=
  (total_duration / glow_interval : ℕ)

/-- Proof that the maximum number of glows is 236 for the given conditions -/
theorem max_glow_count_for_given_conditions :
  max_glow_count 4969 21 = 236 := by
  sorry

end NUMINAMATH_CALUDE_max_glow_count_max_glow_count_for_given_conditions_l1186_118656


namespace NUMINAMATH_CALUDE_circle_origin_inside_l1186_118640

theorem circle_origin_inside (m : ℝ) : 
  (∀ x y : ℝ, (x - m)^2 + (y + m)^2 < 4 → x^2 + y^2 = 0) → 
  -Real.sqrt 2 < m ∧ m < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_origin_inside_l1186_118640


namespace NUMINAMATH_CALUDE_geometric_increasing_condition_l1186_118684

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_increasing_condition (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (is_increasing_sequence a ↔ a 1 < a 2 ∧ a 2 < a 3) :=
sorry

end NUMINAMATH_CALUDE_geometric_increasing_condition_l1186_118684


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l1186_118617

-- Problem 1
theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((1) * (2 * a^(1/2) * b^(1/3)) * (a^(2/3) * b^(1/2))) / ((1/3) * a^(1/6) * b^(5/6)) = 6 * a :=
sorry

-- Problem 2
theorem evaluate_expression :
  (2 * (9/16)^(1/2) + 10^(Real.log 9 / Real.log 10 - 2 * Real.log 2 / Real.log 10) + 
   Real.log (4 * Real.exp 3) - Real.log 8 / Real.log 9 * Real.log 33 / Real.log 4) = 7/2 :=
sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l1186_118617


namespace NUMINAMATH_CALUDE_island_population_theorem_l1186_118665

theorem island_population_theorem (a b c d : ℝ) 
  (h1 : a / (a + b) = 0.65)  -- 65% of blue-eyed are brunettes
  (h2 : b / (b + c) = 0.7)   -- 70% of blondes have blue eyes
  (h3 : c / (c + d) = 0.1)   -- 10% of green-eyed are blondes
  (h4 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) -- All populations are positive
  : d / (a + b + c + d) = 0.54 := by
  sorry

#check island_population_theorem

end NUMINAMATH_CALUDE_island_population_theorem_l1186_118665


namespace NUMINAMATH_CALUDE_square_between_500_600_l1186_118691

theorem square_between_500_600 : ∃ n : ℕ, 
  500 < n^2 ∧ n^2 ≤ 600 ∧ (n-1)^2 < 500 := by
  sorry

end NUMINAMATH_CALUDE_square_between_500_600_l1186_118691


namespace NUMINAMATH_CALUDE_fraction_power_product_l1186_118615

theorem fraction_power_product : (9/8 : ℚ)^4 * (8/9 : ℚ)^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l1186_118615


namespace NUMINAMATH_CALUDE_smallest_number_is_five_l1186_118610

theorem smallest_number_is_five (x y z : ℕ) 
  (sum_xy : x + y = 20) 
  (sum_xz : x + z = 27) 
  (sum_yz : y + z = 37) : 
  min x (min y z) = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_is_five_l1186_118610


namespace NUMINAMATH_CALUDE_complex_magnitude_l1186_118647

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1186_118647


namespace NUMINAMATH_CALUDE_tolu_pencils_tolu_wants_three_pencils_l1186_118674

/-- The problem of determining the number of pencils Tolu wants -/
theorem tolu_pencils (pencil_price : ℚ) (robert_pencils melissa_pencils : ℕ) 
  (total_spent : ℚ) : ℕ :=
  let tolu_pencils := (total_spent - pencil_price * (robert_pencils + melissa_pencils)) / pencil_price
  3

/-- The main theorem stating that Tolu wants 3 pencils -/
theorem tolu_wants_three_pencils : 
  tolu_pencils (20 / 100) 5 2 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tolu_pencils_tolu_wants_three_pencils_l1186_118674


namespace NUMINAMATH_CALUDE_pears_for_20_apples_l1186_118670

/-- The number of apples that cost the same as 5 oranges -/
def apples_per_5_oranges : ℕ := 10

/-- The number of oranges that cost the same as 4 pears -/
def oranges_per_4_pears : ℕ := 3

/-- The number of apples we want to find the equivalent pears for -/
def target_apples : ℕ := 20

/-- The function to calculate the number of pears equivalent to a given number of apples -/
def pears_for_apples (n : ℕ) : ℚ :=
  (n : ℚ) * 5 / apples_per_5_oranges * 4 / oranges_per_4_pears

theorem pears_for_20_apples :
  pears_for_apples target_apples = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pears_for_20_apples_l1186_118670


namespace NUMINAMATH_CALUDE_race_elimination_proof_l1186_118646

/-- The number of racers at the start of the race -/
def initial_racers : ℕ := 100

/-- The number of racers in the final section -/
def final_racers : ℕ := 30

/-- The fraction of racers remaining after the second segment -/
def second_segment_fraction : ℚ := 2/3

/-- The fraction of racers remaining after the third segment -/
def third_segment_fraction : ℚ := 1/2

/-- The number of racers eliminated after the first segment -/
def eliminated_first_segment : ℕ := 10

theorem race_elimination_proof :
  (↑final_racers : ℚ) = third_segment_fraction * second_segment_fraction * (initial_racers - eliminated_first_segment) :=
sorry

end NUMINAMATH_CALUDE_race_elimination_proof_l1186_118646


namespace NUMINAMATH_CALUDE_sum_of_powers_l1186_118696

theorem sum_of_powers (x : ℝ) (h1 : x^2020 - 3*x + 2 = 0) (h2 : x ≠ 1) :
  x^2019 + x^2018 + x^2017 + x^2016 + x^2015 + x^2014 + x^2013 + x^2012 + x^2011 + x^2010 +
  x^2009 + x^2008 + x^2007 + x^2006 + x^2005 + x^2004 + x^2003 + x^2002 + x^2001 + x^2000 +
  x^1999 + x^1998 + x^1997 + x^1996 + x^1995 + x^1994 + x^1993 + x^1992 + x^1991 + x^1990 +
  x^1989 + x^1988 + x^1987 + x^1986 + x^1985 + x^1984 + x^1983 + x^1982 + x^1981 + x^1980 +
  x^1979 + x^1978 + x^1977 + x^1976 + x^1975 + x^1974 + x^1973 + x^1972 + x^1971 + x^1970 +
  -- ... (continue for all powers from 1969 to 2)
  x^2 + x + 1 - 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1186_118696


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1186_118695

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum sequence
  arithmetic_seq : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- Main theorem about properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
    (h1 : seq.S 6 < seq.S 7) (h2 : seq.S 7 > seq.S 8) :
  seq.d < 0 ∧ seq.S 9 < seq.S 6 ∧ ∀ n, seq.S n ≤ seq.S 7 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1186_118695


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l1186_118659

theorem largest_prime_factor_of_expression : 
  let n : ℤ := 16^4 + 3*16^2 + 2 - 17^4
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ n.natAbs ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ n.natAbs → q ≤ p ∧ p = 547 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l1186_118659


namespace NUMINAMATH_CALUDE_point_coordinates_l1186_118634

-- Define the point P
def P : ℝ × ℝ := sorry

-- Define the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Define the distance to x-axis
def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

-- Define the distance to y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  |p.1|

-- State the theorem
theorem point_coordinates :
  in_fourth_quadrant P ∧
  distance_to_x_axis P = 1 ∧
  distance_to_y_axis P = 2 →
  P = (2, -1) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l1186_118634


namespace NUMINAMATH_CALUDE_tight_sequence_x_range_arithmetic_sequence_is_tight_geometric_sequence_tight_condition_l1186_118630

def is_tight_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (1/2 : ℝ) ≤ a (n+1) / a n ∧ a (n+1) / a n ≤ 2

theorem tight_sequence_x_range (a : ℕ → ℝ) (h : is_tight_sequence a)
  (h1 : a 1 = 1) (h2 : a 2 = 3/2) (h3 : a 4 = 4) :
  2 ≤ a 3 ∧ a 3 ≤ 3 := by sorry

theorem arithmetic_sequence_is_tight (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 1 > 0) (h2 : 0 < d) (h3 : d ≤ a 1)
  (h4 : ∀ n : ℕ, n > 0 → a (n+1) = a n + d) :
  is_tight_sequence a := by sorry

def partial_sum (a : ℕ → ℝ) : ℕ → ℝ
| 0 => 0
| n+1 => partial_sum a n + a (n+1)

theorem geometric_sequence_tight_condition (a : ℕ → ℝ) (q : ℝ)
  (h : ∀ n : ℕ, n > 0 → a (n+1) = q * a n) :
  (is_tight_sequence a ∧ is_tight_sequence (partial_sum a)) ↔ 1/2 ≤ q ∧ q ≤ 1 := by sorry

end NUMINAMATH_CALUDE_tight_sequence_x_range_arithmetic_sequence_is_tight_geometric_sequence_tight_condition_l1186_118630


namespace NUMINAMATH_CALUDE_factorial_units_digit_zero_sum_factorials_units_digit_l1186_118604

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def sumFactorialsUnitsDigit (n : ℕ) : ℕ :=
  unitsDigit ((List.range n).map factorial).sum

theorem factorial_units_digit_zero (n : ℕ) (h : n ≥ 5) :
  unitsDigit (factorial n) = 0 := by sorry

theorem sum_factorials_units_digit :
  sumFactorialsUnitsDigit 2010 = 3 := by sorry

end NUMINAMATH_CALUDE_factorial_units_digit_zero_sum_factorials_units_digit_l1186_118604


namespace NUMINAMATH_CALUDE_eightieth_digit_is_one_l1186_118633

def sequence_digit (n : ℕ) : ℕ :=
  if n ≤ 102 then
    let num := 60 - ((n - 1) / 2)
    if n % 2 = 0 then num % 10 else (num / 10) % 10
  else
    sorry -- Handle single-digit numbers if needed

theorem eightieth_digit_is_one :
  sequence_digit 80 = 1 := by
  sorry

end NUMINAMATH_CALUDE_eightieth_digit_is_one_l1186_118633


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1186_118697

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 1 - Real.sqrt 5 * Complex.I → z = Real.sqrt 5 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1186_118697


namespace NUMINAMATH_CALUDE_quadratic_difference_theorem_l1186_118645

theorem quadratic_difference_theorem (a b : ℝ) :
  (∀ x y : ℝ, (a*x^2 + 2*x*y - x) - (3*x^2 - 2*b*x*y + 3*y) = (-x + 3*y)) →
  a^2 - 4*b = 13 := by
sorry

end NUMINAMATH_CALUDE_quadratic_difference_theorem_l1186_118645


namespace NUMINAMATH_CALUDE_smallest_gcd_qr_l1186_118601

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 300) (h2 : Nat.gcd p r = 450) :
  ∃ (q' r' : ℕ+), Nat.gcd p q' = 300 ∧ Nat.gcd p r' = 450 ∧ Nat.gcd q' r' = 150 ∧
  ∀ (q'' r'' : ℕ+), Nat.gcd p q'' = 300 → Nat.gcd p r'' = 450 → Nat.gcd q'' r'' ≥ 150 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_qr_l1186_118601


namespace NUMINAMATH_CALUDE_inequality_proof_l1186_118632

theorem inequality_proof (a b θ : Real) 
  (h1 : a > b) (h2 : b > 1) (h3 : 0 < θ) (h4 : θ < π / 2) :
  a * Real.log (Real.sin θ) / Real.log b < b * Real.log (Real.sin θ) / Real.log a :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1186_118632


namespace NUMINAMATH_CALUDE_equal_area_trapezoid_result_l1186_118686

/-- 
A trapezoid with bases differing by 150 units, where x is the length of the segment 
parallel to the bases that divides the trapezoid into two equal-area regions.
-/
structure EqualAreaTrapezoid where
  base_diff : ℝ := 150
  x : ℝ
  divides_equally : x > 0

/-- 
The greatest integer not exceeding x^2/120 for an EqualAreaTrapezoid is 3000.
-/
theorem equal_area_trapezoid_result (t : EqualAreaTrapezoid) : 
  ⌊(t.x^2 / 120)⌋ = 3000 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_trapezoid_result_l1186_118686


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l1186_118651

theorem smallest_constant_inequality (D : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + 4 ≥ D * (x + y + z)) ↔ D ≤ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l1186_118651


namespace NUMINAMATH_CALUDE_workshop_salary_problem_l1186_118642

theorem workshop_salary_problem (total_workers : ℕ) (avg_salary : ℝ) 
  (technicians : ℕ) (technician_avg_salary : ℝ) :
  total_workers = 28 →
  avg_salary = 8000 →
  technicians = 7 →
  technician_avg_salary = 14000 →
  (total_workers * avg_salary - technicians * technician_avg_salary) / (total_workers - technicians) = 6000 :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_salary_problem_l1186_118642


namespace NUMINAMATH_CALUDE_phillip_initial_vinegar_l1186_118638

/-- The number of jars Phillip has -/
def num_jars : ℕ := 4

/-- The number of cucumbers Phillip has -/
def num_cucumbers : ℕ := 10

/-- The number of pickles each cucumber makes -/
def pickles_per_cucumber : ℕ := 6

/-- The number of pickles each jar can hold -/
def pickles_per_jar : ℕ := 12

/-- The amount of vinegar (in ounces) needed per jar of pickles -/
def vinegar_per_jar : ℕ := 10

/-- The amount of vinegar (in ounces) left after making pickles -/
def vinegar_left : ℕ := 60

/-- Theorem stating that Phillip started with 100 ounces of vinegar -/
theorem phillip_initial_vinegar : 
  (min num_jars ((num_cucumbers * pickles_per_cucumber) / pickles_per_jar)) * vinegar_per_jar + vinegar_left = 100 := by
  sorry

end NUMINAMATH_CALUDE_phillip_initial_vinegar_l1186_118638


namespace NUMINAMATH_CALUDE_max_value_theorem_l1186_118619

theorem max_value_theorem (u v : ℝ) 
  (h1 : 2 * u + 3 * v ≤ 10) 
  (h2 : 4 * u + v ≤ 9) : 
  u + 2 * v ≤ 6.1 ∧ ∃ (u₀ v₀ : ℝ), 2 * u₀ + 3 * v₀ ≤ 10 ∧ 4 * u₀ + v₀ ≤ 9 ∧ u₀ + 2 * v₀ = 6.1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1186_118619


namespace NUMINAMATH_CALUDE_abcd_equation_solutions_l1186_118693

theorem abcd_equation_solutions :
  ∀ (A B C D : ℕ),
    0 ≤ A ∧ A ≤ 9 ∧
    0 ≤ B ∧ B ≤ 9 ∧
    0 ≤ C ∧ C ≤ 9 ∧
    0 ≤ D ∧ D ≤ 9 ∧
    1000 ≤ 1000 * A + 100 * B + 10 * C + D ∧
    1000 * A + 100 * B + 10 * C + D ≤ 9999 ∧
    1000 * A + 100 * B + 10 * C + D = (10 * A + D) * (101 * A + 10 * D) →
    (A = 1 ∧ B = 0 ∧ C = 1 ∧ D = 0) ∨
    (A = 1 ∧ B = 2 ∧ C = 2 ∧ D = 1) ∨
    (A = 1 ∧ B = 4 ∧ C = 5 ∧ D = 2) ∨
    (A = 1 ∧ B = 7 ∧ C = 0 ∧ D = 3) ∨
    (A = 1 ∧ B = 9 ∧ C = 7 ∧ D = 4) :=
by sorry

end NUMINAMATH_CALUDE_abcd_equation_solutions_l1186_118693


namespace NUMINAMATH_CALUDE_quadratic_with_property_has_negative_root_l1186_118626

/-- A quadratic polynomial with the given property has at least one negative root -/
theorem quadratic_with_property_has_negative_root (f : ℝ → ℝ) 
  (h1 : ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0) 
  (h2 : ∀ (a b : ℝ), f (a^2 + b^2) ≥ f (2*a*b)) :
  ∃ (x : ℝ), x < 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_with_property_has_negative_root_l1186_118626


namespace NUMINAMATH_CALUDE_triangle_side_length_l1186_118614

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if the area is √3, B = 60°, and a² + c² = 3ac, then b = 2√2 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →  -- Area condition
  (B = π/3) →  -- 60° in radians
  (a^2 + c^2 = 3*a*c) →  -- Given condition
  (b = 2 * Real.sqrt 2) := by
sorry


end NUMINAMATH_CALUDE_triangle_side_length_l1186_118614


namespace NUMINAMATH_CALUDE_jar_water_problem_l1186_118611

theorem jar_water_problem (s l w : ℚ) : 
  s > 0 ∧ l > 0 ∧ s < l ∧ w > 0 →  -- s: smaller jar capacity, l: larger jar capacity, w: water amount
  w = (1/6) * s ∧ w = (1/5) * l → 
  (2 * w) / l = 2/5 := by sorry

end NUMINAMATH_CALUDE_jar_water_problem_l1186_118611


namespace NUMINAMATH_CALUDE_solve_equation_l1186_118685

theorem solve_equation (x : ℝ) (h : 7 * (x - 1) = 21) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1186_118685


namespace NUMINAMATH_CALUDE_coin_value_proof_l1186_118648

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The number of pennies -/
def num_pennies : ℕ := 9

/-- The number of nickels -/
def num_nickels : ℕ := 4

/-- The number of dimes -/
def num_dimes : ℕ := 3

/-- The total value of the coins in dollars -/
def total_value : ℚ := num_pennies * penny_value + num_nickels * nickel_value + num_dimes * dime_value

theorem coin_value_proof : total_value = 59 / 100 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_proof_l1186_118648


namespace NUMINAMATH_CALUDE_partition_6_3_l1186_118606

/-- Represents a partition of n into at most k parts -/
def Partition (n : ℕ) (k : ℕ) := { p : List ℕ // p.length ≤ k ∧ p.sum = n }

/-- Counts the number of partitions of n into at most k indistinguishable parts -/
def countPartitions (n : ℕ) (k : ℕ) : ℕ := sorry

theorem partition_6_3 : countPartitions 6 3 = 6 := by sorry

end NUMINAMATH_CALUDE_partition_6_3_l1186_118606


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l1186_118688

theorem unique_quadratic_solution (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b > 0 ∧ 
   ∃! x : ℝ, x^2 + (b + 2/b) * x + c = 0) ↔ 
  c = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l1186_118688


namespace NUMINAMATH_CALUDE_inscribed_triangle_ratio_l1186_118636

-- Define the ellipse
def ellipse (p q : ℝ) (x y : ℝ) : Prop :=
  x^2 / p^2 + y^2 / q^2 = 1

-- Define an equilateral triangle
def equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  (dist A B = dist B C) ∧ (dist B C = dist C A)

-- Define that a point is on a line segment
def on_segment (P A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = ((1 - t) • A.1 + t • B.1, (1 - t) • A.2 + t • B.2)

theorem inscribed_triangle_ratio (p q : ℝ) (A B C F₁ F₂ : ℝ × ℝ) :
  ellipse p q A.1 A.2 →
  ellipse p q B.1 B.2 →
  ellipse p q C.1 C.2 →
  B = (0, q) →
  A.2 = C.2 →
  equilateral_triangle A B C →
  on_segment F₁ B C →
  on_segment F₂ A B →
  dist F₁ F₂ = 2 →
  dist A B / dist F₁ F₂ = 8/5 :=
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_ratio_l1186_118636


namespace NUMINAMATH_CALUDE_solution_verification_l1186_118657

-- Define the system of equations
def equation1 (x : ℝ) : Prop := 0.05 * x + 0.07 * (30 + x) = 14.9
def equation2 (x y : ℝ) : Prop := 0.03 * y - 5.6 = 0.07 * x

-- Theorem statement
theorem solution_verification :
  ∃ (x y : ℝ), equation1 x ∧ equation2 x y ∧ x = 106.67 ∧ y = 435.567 := by
  sorry

end NUMINAMATH_CALUDE_solution_verification_l1186_118657


namespace NUMINAMATH_CALUDE_largest_prime_square_root_l1186_118624

theorem largest_prime_square_root (p : ℕ) (a b : ℕ+) (h_prime : Nat.Prime p) 
  (h_eq : (p : ℝ) = (b.val : ℝ) / 2 * Real.sqrt ((a.val - b.val : ℝ) / (a.val + b.val))) :
  p ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_square_root_l1186_118624


namespace NUMINAMATH_CALUDE_smallest_in_set_l1186_118668

theorem smallest_in_set : 
  let S : Set ℤ := {0, -1, 1, 2}
  ∀ x ∈ S, -1 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_in_set_l1186_118668


namespace NUMINAMATH_CALUDE_factorization_equality_l1186_118671

theorem factorization_equality (x : ℝ) : 
  32 * x^4 - 48 * x^7 + 16 * x^2 = 16 * x^2 * (2 * x^2 - 3 * x^5 + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1186_118671


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1186_118652

theorem product_of_three_numbers (x y z : ℝ) 
  (sum_eq : x + y + z = 30)
  (first_eq : x = 3 * (y + z))
  (second_eq : y = 5 * z) :
  x * y * z = 175.78125 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1186_118652


namespace NUMINAMATH_CALUDE_equation_solution_l1186_118654

theorem equation_solution : 
  ∃! x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) ∧ x = -48/23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1186_118654


namespace NUMINAMATH_CALUDE_sum_of_solutions_l1186_118669

-- Define the equation
def equation (M : ℝ) : Prop := M * (M - 8) = -8

-- Theorem statement
theorem sum_of_solutions : 
  ∃ (M₁ M₂ : ℝ), equation M₁ ∧ equation M₂ ∧ M₁ + M₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l1186_118669


namespace NUMINAMATH_CALUDE_employed_males_percentage_l1186_118627

theorem employed_males_percentage (population : ℝ) 
  (h1 : population > 0) 
  (employed_percentage : ℝ) 
  (h2 : employed_percentage = 0.64) 
  (employed_females_percentage : ℝ) 
  (h3 : employed_females_percentage = 0.140625) : 
  (employed_percentage * (1 - employed_females_percentage)) * population / population = 0.5496 := by
sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l1186_118627


namespace NUMINAMATH_CALUDE_sqrt_factorial_over_88_l1186_118608

theorem sqrt_factorial_over_88 : 
  let factorial_10 : ℕ := 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
  let n : ℚ := factorial_10 / 88
  Real.sqrt n = (180 * Real.sqrt 7) / Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_factorial_over_88_l1186_118608


namespace NUMINAMATH_CALUDE_tank_water_volume_l1186_118682

/-- Calculates the final volume of water in a tank after evaporation, draining, and rainfall. -/
theorem tank_water_volume 
  (initial_volume : ℕ) 
  (evaporated_volume : ℕ) 
  (drained_volume : ℕ) 
  (rain_duration : ℕ) 
  (rain_rate : ℕ) 
  (rain_interval : ℕ) 
  (h1 : initial_volume = 6000)
  (h2 : evaporated_volume = 2000)
  (h3 : drained_volume = 3500)
  (h4 : rain_duration = 30)
  (h5 : rain_rate = 350)
  (h6 : rain_interval = 10) :
  initial_volume - evaporated_volume - drained_volume + 
  (rain_duration / rain_interval) * rain_rate = 1550 :=
by
  sorry

#check tank_water_volume

end NUMINAMATH_CALUDE_tank_water_volume_l1186_118682


namespace NUMINAMATH_CALUDE_complex_sum_nonzero_components_l1186_118625

theorem complex_sum_nonzero_components (a b : ℝ) :
  (a : ℂ) + b * Complex.I = (1 - Complex.I)^10 + (1 + Complex.I)^10 →
  a ≠ 0 ∧ b ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_nonzero_components_l1186_118625


namespace NUMINAMATH_CALUDE_scott_runs_84_miles_per_month_l1186_118673

/-- Scott's weekly running schedule -/
structure RunningSchedule where
  mon_to_wed : ℕ  -- Miles run Monday through Wednesday (daily)
  thu_fri : ℕ     -- Miles run Thursday and Friday (daily)

/-- Calculate total miles run in a week -/
def weekly_miles (schedule : RunningSchedule) : ℕ :=
  schedule.mon_to_wed * 3 + schedule.thu_fri * 2

/-- Calculate total miles run in a month -/
def monthly_miles (schedule : RunningSchedule) (weeks : ℕ) : ℕ :=
  weekly_miles schedule * weeks

/-- Scott's actual running schedule -/
def scotts_schedule : RunningSchedule :=
  { mon_to_wed := 3, thu_fri := 6 }

/-- Theorem: Scott runs 84 miles in a month with 4 weeks -/
theorem scott_runs_84_miles_per_month : 
  monthly_miles scotts_schedule 4 = 84 := by sorry

end NUMINAMATH_CALUDE_scott_runs_84_miles_per_month_l1186_118673


namespace NUMINAMATH_CALUDE_original_average_age_proof_l1186_118637

theorem original_average_age_proof (initial_avg : ℝ) (new_students : ℕ) (new_students_avg : ℝ) (avg_decrease : ℝ) :
  initial_avg = 40 →
  new_students = 12 →
  new_students_avg = 34 →
  avg_decrease = 4 →
  initial_avg = 40 := by
sorry

end NUMINAMATH_CALUDE_original_average_age_proof_l1186_118637


namespace NUMINAMATH_CALUDE_symmetric_curve_correct_l1186_118676

/-- The equation of a curve symmetric to y^2 = 4x with respect to the line x = 2 -/
def symmetric_curve_equation (x y : ℝ) : Prop :=
  y^2 = 16 - 4*x

/-- The original curve equation -/
def original_curve_equation (x y : ℝ) : Prop :=
  y^2 = 4*x

/-- The line of symmetry -/
def symmetry_line : ℝ := 2

/-- Theorem stating that the symmetric curve equation is correct -/
theorem symmetric_curve_correct :
  ∀ x y : ℝ, symmetric_curve_equation x y ↔ 
  original_curve_equation (2*symmetry_line - x) y :=
by sorry

end NUMINAMATH_CALUDE_symmetric_curve_correct_l1186_118676


namespace NUMINAMATH_CALUDE_parabola_equation_l1186_118681

-- Define the line on which the focus lies
def focus_line (x y : ℝ) : Prop := x + 2 * y + 3 = 0

-- Define the two possible standard equations for the parabola
def vertical_parabola (x y : ℝ) : Prop := x^2 = -6 * y
def horizontal_parabola (x y : ℝ) : Prop := y^2 = -12 * x

-- Theorem statement
theorem parabola_equation (C : Set (ℝ × ℝ)) :
  (∃ (x y : ℝ), (x, y) ∈ C ∧ focus_line x y) →
  (∀ (x y : ℝ), (x, y) ∈ C → vertical_parabola x y ∨ horizontal_parabola x y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1186_118681


namespace NUMINAMATH_CALUDE_minimize_resistance_l1186_118620

/-- Represents the resistance of a component assembled using six resistors. -/
noncomputable def totalResistance (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) (R₁ R₂ R₃ R₄ R₅ R₆ : ℝ) : ℝ :=
  sorry -- Definition of total resistance based on the given configuration

/-- Theorem stating the condition for minimizing the total resistance of the component. -/
theorem minimize_resistance
  (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ)
  (h₁ : a₁ > a₂) (h₂ : a₂ > a₃) (h₃ : a₃ > a₄) (h₄ : a₄ > a₅) (h₅ : a₅ > a₆)
  (h₆ : a₁ > 0) (h₇ : a₂ > 0) (h₈ : a₃ > 0) (h₉ : a₄ > 0) (h₁₀ : a₅ > 0) (h₁₁ : a₆ > 0) :
  ∃ (R₁ R₂ : ℝ), 
    (R₁ = a₁ ∧ R₂ = a₂) ∨ (R₁ = a₂ ∧ R₂ = a₁) ∧
    ∀ (S₁ S₂ S₃ S₄ S₅ S₆ : ℝ),
      totalResistance a₁ a₂ a₃ a₄ a₅ a₆ R₁ R₂ a₃ a₄ a₅ a₆ ≤ 
      totalResistance a₁ a₂ a₃ a₄ a₅ a₆ S₁ S₂ S₃ S₄ S₅ S₆ :=
by
  sorry

end NUMINAMATH_CALUDE_minimize_resistance_l1186_118620
