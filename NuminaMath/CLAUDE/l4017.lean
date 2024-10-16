import Mathlib

namespace NUMINAMATH_CALUDE_cereal_servings_l4017_401785

theorem cereal_servings (cups_per_serving : ℝ) (total_cups_needed : ℝ) 
  (h1 : cups_per_serving = 2.0)
  (h2 : total_cups_needed = 36) :
  total_cups_needed / cups_per_serving = 18 := by
  sorry

end NUMINAMATH_CALUDE_cereal_servings_l4017_401785


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l4017_401734

theorem sum_of_a_and_b (a b : ℝ) (h1 : a + 3*b = 27) (h2 : 5*a + 4*b = 47) : 
  a + b = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l4017_401734


namespace NUMINAMATH_CALUDE_probability_after_removing_pairs_l4017_401762

/-- Represents a deck of cards -/
structure Deck :=
  (total : ℕ)
  (numbers : ℕ)
  (cards_per_number : ℕ)

/-- Represents the state after removing pairs -/
structure RemovedPairs :=
  (pairs_removed : ℕ)

/-- Calculates the probability of selecting a pair from the remaining deck -/
def probability_of_pair (d : Deck) (r : RemovedPairs) : ℚ :=
  sorry

/-- The main theorem -/
theorem probability_after_removing_pairs :
  let d : Deck := ⟨80, 20, 4⟩
  let r : RemovedPairs := ⟨3⟩
  probability_of_pair d r = 105 / 2701 :=
sorry

end NUMINAMATH_CALUDE_probability_after_removing_pairs_l4017_401762


namespace NUMINAMATH_CALUDE_inverse_89_mod_90_l4017_401740

theorem inverse_89_mod_90 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 89 ∧ (89 * x) % 90 = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_89_mod_90_l4017_401740


namespace NUMINAMATH_CALUDE_alphabet_letter_count_l4017_401743

theorem alphabet_letter_count (total : ℕ) (both : ℕ) (straight_only : ℕ) :
  total = 40 →
  both = 8 →
  straight_only = 24 →
  total = both + straight_only + (total - (both + straight_only)) →
  (total - (both + straight_only)) = 8 :=
by sorry

end NUMINAMATH_CALUDE_alphabet_letter_count_l4017_401743


namespace NUMINAMATH_CALUDE_reservoir_capacity_l4017_401731

theorem reservoir_capacity (x : ℝ) 
  (h1 : x > 0) -- Ensure the capacity is positive
  (h2 : (1/4) * x + 100 = (3/8) * x) -- Condition from initial state to final state
  : x = 800 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_capacity_l4017_401731


namespace NUMINAMATH_CALUDE_horse_speed_around_square_field_l4017_401753

/-- Given a square field with area 900 km² and a horse that takes 10 hours to run around it,
    prove that the horse's speed is 12 km/h. -/
theorem horse_speed_around_square_field : 
  let field_area : ℝ := 900
  let time_to_run_around : ℝ := 10
  let horse_speed : ℝ := 4 * Real.sqrt field_area / time_to_run_around
  horse_speed = 12 := by
  sorry

end NUMINAMATH_CALUDE_horse_speed_around_square_field_l4017_401753


namespace NUMINAMATH_CALUDE_total_average_marks_specific_classes_l4017_401749

/-- The total average marks of students in three classes -/
def total_average_marks (class1_students : ℕ) (class1_avg : ℚ)
                        (class2_students : ℕ) (class2_avg : ℚ)
                        (class3_students : ℕ) (class3_avg : ℚ) : ℚ :=
  (class1_avg * class1_students + class2_avg * class2_students + class3_avg * class3_students) /
  (class1_students + class2_students + class3_students)

/-- Theorem stating the total average marks of students in three specific classes -/
theorem total_average_marks_specific_classes :
  total_average_marks 47 52 33 68 40 75 = 7688 / 120 :=
by sorry

end NUMINAMATH_CALUDE_total_average_marks_specific_classes_l4017_401749


namespace NUMINAMATH_CALUDE_paint_needed_to_buy_l4017_401736

theorem paint_needed_to_buy (total_paint : ℕ) (available_paint : ℕ) : 
  total_paint = 333 → available_paint = 157 → total_paint - available_paint = 176 := by
  sorry

end NUMINAMATH_CALUDE_paint_needed_to_buy_l4017_401736


namespace NUMINAMATH_CALUDE_cubic_factorization_l4017_401759

theorem cubic_factorization (m : ℝ) : m^3 - 6*m^2 + 9*m = m*(m-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l4017_401759


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l4017_401777

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_quadratic_inequality : 
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ + (1/4 : ℝ) ≤ 0) ↔ 
  (∀ x : ℝ, x^2 - x + (1/4 : ℝ) > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l4017_401777


namespace NUMINAMATH_CALUDE_nathan_blankets_l4017_401790

-- Define the warmth provided by each blanket
def warmth_per_blanket : ℕ := 3

-- Define the total warmth provided by the blankets Nathan used
def total_warmth : ℕ := 21

-- Define the number of blankets Nathan used (half of the total)
def blankets_used : ℕ := total_warmth / warmth_per_blanket

-- Define the total number of blankets in Nathan's closet
def total_blankets : ℕ := 2 * blankets_used

-- Theorem stating that the total number of blankets is 14
theorem nathan_blankets : total_blankets = 14 := by
  sorry

end NUMINAMATH_CALUDE_nathan_blankets_l4017_401790


namespace NUMINAMATH_CALUDE_incorrect_transformation_l4017_401788

theorem incorrect_transformation (a b : ℝ) (h : a > b) : ¬(3 - a > 3 - b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_transformation_l4017_401788


namespace NUMINAMATH_CALUDE_ellipse_equation_l4017_401720

/-- Given an ellipse with equation x²/a² + 25y²/(9a²) = 1, prove that the equation
    of the ellipse is x² + 25/9 * y² = 1 under the following conditions:
    - Points A and B are on the ellipse
    - F₂ is the right focus of the ellipse
    - |AF₂| + |BF₂| = 8/5 * a
    - Distance from midpoint of AB to left directrix is 3/2 -/
theorem ellipse_equation (a : ℝ) (A B F₂ : ℝ × ℝ) :
  (∀ x y, x^2/a^2 + 25*y^2/(9*a^2) = 1 → (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) →
  (F₂.1 > 0) →
  (Real.sqrt ((A.1 - F₂.1)^2 + (A.2 - F₂.2)^2) + Real.sqrt ((B.1 - F₂.1)^2 + (B.2 - F₂.2)^2) = 8/5 * a) →
  (((A.1 + B.1)/2 + 5/4*a) = 3/2) →
  (∀ x y, x^2 + 25/9 * y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l4017_401720


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4017_401747

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 6*x + 2
  ∃ x₁ x₂ : ℝ, x₁ = 3 + Real.sqrt 7 ∧ x₂ = 3 - Real.sqrt 7 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4017_401747


namespace NUMINAMATH_CALUDE_book_pages_l4017_401764

theorem book_pages : ∀ (total : ℕ), 
  (total : ℚ) * (1 - 2/5) * (1 - 5/8) = 36 → 
  total = 120 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_l4017_401764


namespace NUMINAMATH_CALUDE_equation_root_implies_m_value_l4017_401797

theorem equation_root_implies_m_value (x m : ℝ) :
  (∃ x, (x - 1) / (x - 4) = m / (x - 4)) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_root_implies_m_value_l4017_401797


namespace NUMINAMATH_CALUDE_unit_vector_parallel_to_3_4_l4017_401725

def is_unit_vector (v : ℝ × ℝ) : Prop :=
  v.1^2 + v.2^2 = 1

def is_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem unit_vector_parallel_to_3_4 :
  ∃ (v : ℝ × ℝ), is_unit_vector v ∧ is_parallel v (3, 4) ∧
  (v = (3/5, 4/5) ∨ v = (-3/5, -4/5)) :=
sorry

end NUMINAMATH_CALUDE_unit_vector_parallel_to_3_4_l4017_401725


namespace NUMINAMATH_CALUDE_running_program_weekly_increase_l4017_401782

theorem running_program_weekly_increase 
  (initial_distance : ℝ) 
  (final_distance : ℝ) 
  (program_duration : ℕ) 
  (increase_duration : ℕ) 
  (h1 : initial_distance = 3)
  (h2 : final_distance = 7)
  (h3 : program_duration = 5)
  (h4 : increase_duration = 4)
  : (final_distance - initial_distance) / increase_duration = 1 := by
  sorry

end NUMINAMATH_CALUDE_running_program_weekly_increase_l4017_401782


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_to_prism_volume_l4017_401712

theorem cube_surface_area_equal_to_prism_volume (a b c : ℝ) (h1 : a = 10) (h2 : b = 3) (h3 : c = 30) :
  let prism_volume := a * b * c
  let cube_edge := (prism_volume) ^ (1/3 : ℝ)
  let cube_surface_area := 6 * cube_edge ^ 2
  cube_surface_area = 6 * 900 ^ (2/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_to_prism_volume_l4017_401712


namespace NUMINAMATH_CALUDE_total_amount_is_234_l4017_401729

/-- Represents the division of money among three parties -/
structure MoneyDivision where
  x : ℚ
  y : ℚ
  z : ℚ

/-- Theorem stating the total amount given the conditions -/
theorem total_amount_is_234 
  (div : MoneyDivision) 
  (h1 : div.y = div.x * (45/100))  -- y gets 45 paisa for each rupee x gets
  (h2 : div.z = div.x * (50/100))  -- z gets 50 paisa for each rupee x gets
  (h3 : div.y = 54)                -- The share of y is Rs. 54
  : div.x + div.y + div.z = 234 := by
  sorry

#check total_amount_is_234

end NUMINAMATH_CALUDE_total_amount_is_234_l4017_401729


namespace NUMINAMATH_CALUDE_y_derivative_y_derivative_at_zero_l4017_401745

-- Define y as a function of x
variable (y : ℝ → ℝ)

-- Define the condition e^y + xy = e
variable (h : ∀ x, Real.exp (y x) + x * (y x) = Real.exp 1)

-- Theorem for y'
theorem y_derivative (x : ℝ) : 
  deriv y x = -(y x) / (Real.exp (y x) + x) := by sorry

-- Theorem for y'(0)
theorem y_derivative_at_zero : 
  deriv y 0 = -(1 / Real.exp 1) := by sorry

end NUMINAMATH_CALUDE_y_derivative_y_derivative_at_zero_l4017_401745


namespace NUMINAMATH_CALUDE_max_xy_value_l4017_401794

theorem max_xy_value (x y : ℝ) (hx : x < 0) (hy : y < 0) (heq : 3*x + y = -2) :
  (∀ z : ℝ, z = x*y → z ≤ 1/3) ∧ ∃ z : ℝ, z = x*y ∧ z = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l4017_401794


namespace NUMINAMATH_CALUDE_toms_promotion_expenses_l4017_401741

/-- Represents the problem of calculating Tom's promotion expenses --/
def TomsDoughBallPromotion (flour_needed : ℕ) (flour_bag_size : ℕ) (flour_bag_cost : ℕ)
  (salt_needed : ℕ) (salt_cost_per_pound : ℚ) (tickets_sold : ℕ) (ticket_price : ℕ) (profit : ℕ) : Prop :=
  let flour_bags := flour_needed / flour_bag_size
  let flour_cost := flour_bags * flour_bag_cost
  let salt_cost := (salt_needed : ℚ) * salt_cost_per_pound
  let revenue := tickets_sold * ticket_price
  let promotion_cost := revenue - profit - flour_cost - (salt_cost.num / salt_cost.den)
  promotion_cost = 1000

/-- The theorem stating that Tom's promotion expenses are $1000 --/
theorem toms_promotion_expenses :
  TomsDoughBallPromotion 500 50 20 10 (1/5) 500 20 8798 :=
sorry

end NUMINAMATH_CALUDE_toms_promotion_expenses_l4017_401741


namespace NUMINAMATH_CALUDE_total_handshakes_l4017_401770

-- Define the number of twin sets and triplet sets
def twin_sets : ℕ := 12
def triplet_sets : ℕ := 8

-- Define the total number of twins and triplets
def total_twins : ℕ := twin_sets * 2
def total_triplets : ℕ := triplet_sets * 3

-- Define the number of handshakes for each twin and triplet
def twin_handshakes : ℕ := (total_twins - 2) + (total_triplets * 3 / 4)
def triplet_handshakes : ℕ := (total_triplets - 3) + (total_twins * 1 / 4)

-- Theorem to prove
theorem total_handshakes : 
  (total_twins * twin_handshakes + total_triplets * triplet_handshakes) / 2 = 804 := by
  sorry

end NUMINAMATH_CALUDE_total_handshakes_l4017_401770


namespace NUMINAMATH_CALUDE_rational_sum_l4017_401786

theorem rational_sum (a₁ a₂ a₃ a₄ : ℚ) : 
  ({a₁ * a₂, a₁ * a₃, a₁ * a₄, a₂ * a₃, a₂ * a₄, a₃ * a₄} : Finset ℚ) = 
    {-24, -2, -3/2, -1/8, 1, 3} →
  a₁ + a₂ + a₃ + a₄ = 9/4 ∨ a₁ + a₂ + a₃ + a₄ = -9/4 := by
sorry

end NUMINAMATH_CALUDE_rational_sum_l4017_401786


namespace NUMINAMATH_CALUDE_painted_cube_probability_l4017_401710

/-- Represents a cube with side length 5 and two adjacent faces painted --/
structure PaintedCube :=
  (side_length : ℕ)
  (painted_faces : ℕ)

/-- Calculates the total number of unit cubes in the large cube --/
def total_cubes (c : PaintedCube) : ℕ :=
  c.side_length ^ 3

/-- Calculates the number of unit cubes with exactly two painted faces --/
def two_painted_faces (c : PaintedCube) : ℕ :=
  (c.side_length - 2) ^ 2

/-- Calculates the number of unit cubes with no painted faces --/
def no_painted_faces (c : PaintedCube) : ℕ :=
  total_cubes c - (2 * c.side_length ^ 2 - c.side_length)

/-- Calculates the probability of selecting one cube with two painted faces
    and one cube with no painted faces --/
def probability (c : PaintedCube) : ℚ :=
  (two_painted_faces c * no_painted_faces c : ℚ) /
  (total_cubes c * (total_cubes c - 1) / 2 : ℚ)

/-- The main theorem stating the probability for a 5x5x5 cube with two painted faces --/
theorem painted_cube_probability :
  let c := PaintedCube.mk 5 2
  probability c = 24 / 258 := by
  sorry


end NUMINAMATH_CALUDE_painted_cube_probability_l4017_401710


namespace NUMINAMATH_CALUDE_tim_additional_water_consumption_l4017_401776

/-- Represents the amount of water Tim drinks -/
structure WaterConsumption where
  bottles_per_day : ℕ
  quarts_per_bottle : ℚ
  total_ounces_per_week : ℕ
  ounces_per_quart : ℕ
  days_per_week : ℕ

/-- Calculates the additional ounces of water Tim drinks daily -/
def additional_daily_ounces (w : WaterConsumption) : ℚ :=
  ((w.total_ounces_per_week : ℚ) - 
   (w.bottles_per_day : ℚ) * w.quarts_per_bottle * (w.ounces_per_quart : ℚ) * (w.days_per_week : ℚ)) / 
  (w.days_per_week : ℚ)

/-- Theorem stating that Tim drinks an additional 20 ounces of water daily -/
theorem tim_additional_water_consumption :
  let w : WaterConsumption := {
    bottles_per_day := 2,
    quarts_per_bottle := 3/2,
    total_ounces_per_week := 812,
    ounces_per_quart := 32,
    days_per_week := 7
  }
  additional_daily_ounces w = 20 := by
  sorry

end NUMINAMATH_CALUDE_tim_additional_water_consumption_l4017_401776


namespace NUMINAMATH_CALUDE_blue_paint_calculation_l4017_401761

/-- Represents the ratio of paints (red:blue:yellow:white) -/
structure PaintRatio :=
  (red : ℕ)
  (blue : ℕ)
  (yellow : ℕ)
  (white : ℕ)

/-- Calculates the amount of blue paint needed given a paint ratio and the amount of white paint used -/
def blue_paint_needed (ratio : PaintRatio) (white_paint : ℕ) : ℕ :=
  (ratio.blue * white_paint) / ratio.white

/-- Theorem stating that given the specific paint ratio and 16 quarts of white paint, 12 quarts of blue paint are needed -/
theorem blue_paint_calculation (ratio : PaintRatio) (h1 : ratio.red = 2) (h2 : ratio.blue = 3) 
    (h3 : ratio.yellow = 1) (h4 : ratio.white = 4) (white_paint : ℕ) (h5 : white_paint = 16) : 
  blue_paint_needed ratio white_paint = 12 := by
  sorry

#eval blue_paint_needed {red := 2, blue := 3, yellow := 1, white := 4} 16

end NUMINAMATH_CALUDE_blue_paint_calculation_l4017_401761


namespace NUMINAMATH_CALUDE_ratio_problem_l4017_401760

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 7) :
  a / c = 105 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l4017_401760


namespace NUMINAMATH_CALUDE_unique_element_quadratic_set_l4017_401772

theorem unique_element_quadratic_set (a : ℝ) :
  (∃! x : ℝ, a * x^2 - 2 * x + 1 = 0) ↔ (a = 0 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_element_quadratic_set_l4017_401772


namespace NUMINAMATH_CALUDE_prop_range_m_l4017_401755

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- State the theorem
theorem prop_range_m : 
  ∀ m : ℝ, ¬(p m ∨ q m) → m ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_prop_range_m_l4017_401755


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4017_401748

theorem imaginary_part_of_complex_fraction : Complex.im (Complex.I / (1 - Complex.I)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4017_401748


namespace NUMINAMATH_CALUDE_max_y_value_l4017_401727

theorem max_y_value (x y : ℤ) (h : x * y + 7 * x + 6 * y = 8) : 
  y ≤ 43 ∧ ∃ (x₀ y₀ : ℤ), x₀ * y₀ + 7 * x₀ + 6 * y₀ = 8 ∧ y₀ = 43 :=
by sorry

end NUMINAMATH_CALUDE_max_y_value_l4017_401727


namespace NUMINAMATH_CALUDE_trig_simplification_l4017_401733

theorem trig_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) =
  1/2 * (1 - Real.cos x ^ 2 - Real.cos y ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l4017_401733


namespace NUMINAMATH_CALUDE_positive_integer_solutions_m_value_when_sum_zero_fixed_solution_integer_m_for_integer_x_l4017_401775

-- Define the system of equations
def system (x y m : ℝ) : Prop :=
  x + 2*y - 6 = 0 ∧ x - 2*y + m*x + 5 = 0

-- Theorem 1: Positive integer solutions
theorem positive_integer_solutions :
  ∀ x y : ℤ, x > 0 ∧ y > 0 ∧ x + 2*y - 6 = 0 → (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 1) :=
sorry

-- Theorem 2: Value of m when x + y = 0
theorem m_value_when_sum_zero :
  ∀ x y m : ℝ, system x y m ∧ x + y = 0 → m = -13/6 :=
sorry

-- Theorem 3: Fixed solution regardless of m
theorem fixed_solution :
  ∀ m : ℝ, 0 - 2*2.5 + m*0 + 5 = 0 :=
sorry

-- Theorem 4: Integer values of m for integer x
theorem integer_m_for_integer_x :
  ∀ x : ℤ, ∀ m : ℤ, (∃ y : ℝ, system x y m) → m = -1 ∨ m = -3 :=
sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_m_value_when_sum_zero_fixed_solution_integer_m_for_integer_x_l4017_401775


namespace NUMINAMATH_CALUDE_orange_bags_count_l4017_401719

theorem orange_bags_count (weight_per_bag : ℕ) (total_weight : ℕ) (h1 : weight_per_bag = 23) (h2 : total_weight = 1035) :
  total_weight / weight_per_bag = 45 := by
  sorry

end NUMINAMATH_CALUDE_orange_bags_count_l4017_401719


namespace NUMINAMATH_CALUDE_initial_water_percentage_in_milk_initial_water_percentage_is_five_percent_l4017_401769

/-- Proves that the initial water percentage in milk is 5% given the conditions -/
theorem initial_water_percentage_in_milk : ℝ → Prop :=
  λ initial_water_percentage =>
    let initial_volume : ℝ := 10
    let pure_milk_added : ℝ := 15
    let final_water_percentage : ℝ := 2
    let final_volume : ℝ := 25
    let initial_water_volume : ℝ := initial_water_percentage / 100 * initial_volume
    let final_water_volume : ℝ := final_water_percentage / 100 * final_volume
    initial_water_volume = final_water_volume ∧ initial_water_percentage = 5

/-- The theorem holds true -/
theorem initial_water_percentage_is_five_percent : 
  ∃ x : ℝ, initial_water_percentage_in_milk x :=
sorry

end NUMINAMATH_CALUDE_initial_water_percentage_in_milk_initial_water_percentage_is_five_percent_l4017_401769


namespace NUMINAMATH_CALUDE_other_juice_cost_is_five_l4017_401763

/-- Represents the cost and quantity information for a juice bar order --/
structure JuiceOrder where
  totalSpent : ℕ
  pineappleCost : ℕ
  pineappleSpent : ℕ
  totalPeople : ℕ

/-- Calculates the cost per glass of the other type of juice --/
def otherJuiceCost (order : JuiceOrder) : ℕ :=
  let pineappleGlasses := order.pineappleSpent / order.pineappleCost
  let otherGlasses := order.totalPeople - pineappleGlasses
  let otherSpent := order.totalSpent - order.pineappleSpent
  otherSpent / otherGlasses

/-- Theorem stating that the cost of the other type of juice is $5 per glass --/
theorem other_juice_cost_is_five (order : JuiceOrder) 
  (h1 : order.totalSpent = 94)
  (h2 : order.pineappleCost = 6)
  (h3 : order.pineappleSpent = 54)
  (h4 : order.totalPeople = 17) :
  otherJuiceCost order = 5 := by
  sorry

end NUMINAMATH_CALUDE_other_juice_cost_is_five_l4017_401763


namespace NUMINAMATH_CALUDE_bobs_corn_field_efficiency_l4017_401757

/-- Given a corn field with a certain number of rows and stalks per row,
    and a total harvest in bushels, calculate the number of stalks needed per bushel. -/
def stalks_per_bushel (rows : ℕ) (stalks_per_row : ℕ) (total_bushels : ℕ) : ℕ :=
  (rows * stalks_per_row) / total_bushels

/-- Theorem stating that for Bob's corn field, 8 stalks are needed per bushel. -/
theorem bobs_corn_field_efficiency :
  stalks_per_bushel 5 80 50 = 8 := by
  sorry

end NUMINAMATH_CALUDE_bobs_corn_field_efficiency_l4017_401757


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l4017_401701

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 200) (h2 : y = 250) : x + y = 700 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l4017_401701


namespace NUMINAMATH_CALUDE_consecutive_integers_square_difference_consecutive_odd_integers_square_difference_l4017_401799

theorem consecutive_integers_square_difference (n : ℤ) : 
  ∃ k : ℤ, (n + 2)^2 - n^2 = 4 * k :=
sorry

theorem consecutive_odd_integers_square_difference (n : ℤ) : 
  ∃ k : ℤ, (2*n + 3)^2 - (2*n - 1)^2 = 8 * k :=
sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_difference_consecutive_odd_integers_square_difference_l4017_401799


namespace NUMINAMATH_CALUDE_reduced_price_is_three_l4017_401705

/-- Represents the price reduction and quantity increase for apples -/
structure ApplePriceReduction where
  reduction_percent : ℝ
  additional_apples : ℕ
  fixed_price : ℝ

/-- Calculates the reduced price per dozen apples given the price reduction information -/
def reduced_price_per_dozen (info : ApplePriceReduction) : ℝ :=
  sorry

/-- Theorem stating that for the given conditions, the reduced price per dozen is 3 Rs -/
theorem reduced_price_is_three (info : ApplePriceReduction) 
  (h1 : info.reduction_percent = 40)
  (h2 : info.additional_apples = 64)
  (h3 : info.fixed_price = 40) : 
  reduced_price_per_dozen info = 3 :=
sorry

end NUMINAMATH_CALUDE_reduced_price_is_three_l4017_401705


namespace NUMINAMATH_CALUDE_wedge_volume_l4017_401756

/-- The volume of a wedge formed by two planar cuts in a cylindrical log. -/
theorem wedge_volume (d : ℝ) (angle : ℝ) (h : ℝ) (m : ℕ) : 
  d = 16 →
  angle = 60 →
  h = d →
  (1 / 6) * π * (d / 2)^2 * h = m * π →
  m = 171 := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l4017_401756


namespace NUMINAMATH_CALUDE_cosine_product_sqrt_eight_l4017_401737

theorem cosine_product_sqrt_eight : 
  Real.sqrt ((3 - Real.cos (π / 8) ^ 2) * (3 - Real.cos (π / 4) ^ 2) * (3 - Real.cos (3 * π / 8) ^ 2)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_sqrt_eight_l4017_401737


namespace NUMINAMATH_CALUDE_complex_sum_equals_one_l4017_401703

theorem complex_sum_equals_one (w : ℂ) (h : w = Complex.exp (Complex.I * (6 * Real.pi / 11))) :
  w / (1 + w^2) + w^3 / (1 + w^6) + w^4 / (1 + w^8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_one_l4017_401703


namespace NUMINAMATH_CALUDE_avery_build_time_l4017_401742

theorem avery_build_time (tom_time : ℝ) (total_time : ℝ) : 
  tom_time = 4 →
  (1 / 2 + 1 / tom_time) + 1 / tom_time = 1 →
  2 = total_time :=
by sorry

end NUMINAMATH_CALUDE_avery_build_time_l4017_401742


namespace NUMINAMATH_CALUDE_birth_year_problem_l4017_401781

theorem birth_year_problem :
  ∃! x : ℕ, 1750 < x ∧ x < 1954 ∧
  (7 * x) % 13 = 11 ∧
  (13 * x) % 11 = 7 ∧
  1954 - x = 86 := by
  sorry

end NUMINAMATH_CALUDE_birth_year_problem_l4017_401781


namespace NUMINAMATH_CALUDE_dice_sum_repetition_l4017_401778

theorem dice_sum_repetition (n : ℕ) (m : ℕ) (h1 : n = 21) (h2 : m = 22) :
  m > n → ∀ f : ℕ → ℕ, ∃ i j, i < j ∧ j < m ∧ f i = f j :=
by sorry

end NUMINAMATH_CALUDE_dice_sum_repetition_l4017_401778


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l4017_401787

/-- Given a geometric sequence {a_n}, prove that a_1^2 + a_3^2 ≥ 2a_2^2 -/
theorem geometric_sequence_inequality (a : ℕ → ℝ) (q : ℝ) 
  (h : ∀ n, a (n + 1) = a n * q) : 
  a 1^2 + a 3^2 ≥ 2 * a 2^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l4017_401787


namespace NUMINAMATH_CALUDE_range_of_f_range_of_f_complete_l4017_401738

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem range_of_f :
  ∀ y ∈ Set.range f,
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ f x = y) →
  2 ≤ y ∧ y ≤ 6 :=
by sorry

theorem range_of_f_complete :
  ∀ y : ℝ, 2 ≤ y ∧ y ≤ 6 →
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ f x = y :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_range_of_f_complete_l4017_401738


namespace NUMINAMATH_CALUDE_simplify_fraction_l4017_401784

theorem simplify_fraction : (126 : ℚ) / 11088 = 1 / 88 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4017_401784


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l4017_401767

def N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]

theorem matrix_equation_proof :
  N^3 - 3 • N^2 + 3 • N = !![6, 12; 3, 6] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l4017_401767


namespace NUMINAMATH_CALUDE_sum_704_159_base12_l4017_401780

/-- Represents a number in base 12 --/
def Base12 : Type := List (Fin 12)

/-- Converts a base 10 number to base 12 --/
def toBase12 (n : ℕ) : Base12 :=
  sorry

/-- Converts a base 12 number to base 10 --/
def toBase10 (b : Base12) : ℕ :=
  sorry

/-- Adds two base 12 numbers --/
def addBase12 (a b : Base12) : Base12 :=
  sorry

/-- Theorem: The sum of 704₁₂ and 159₁₂ in base 12 is 861₁₂ --/
theorem sum_704_159_base12 :
  addBase12 (toBase12 704) (toBase12 159) = toBase12 861 :=
sorry

end NUMINAMATH_CALUDE_sum_704_159_base12_l4017_401780


namespace NUMINAMATH_CALUDE_rectangular_table_capacity_l4017_401707

/-- The number of square tables arranged in a row -/
def num_tables : ℕ := 8

/-- The number of people that can sit evenly spaced around one square table -/
def people_per_square_table : ℕ := 12

/-- The number of sides in a square table -/
def sides_per_square : ℕ := 4

/-- Calculate the number of people that can sit on one side of a square table -/
def people_per_side : ℕ := people_per_square_table / sides_per_square

/-- The number of people that can sit on the long side of the rectangular table -/
def long_side_capacity : ℕ := num_tables * people_per_side

/-- The number of people that can sit on the short side of the rectangular table -/
def short_side_capacity : ℕ := 2 * people_per_side

/-- The total number of people that can sit around the rectangular table -/
def total_capacity : ℕ := 2 * long_side_capacity + 2 * short_side_capacity

theorem rectangular_table_capacity :
  total_capacity = 60 := by sorry

end NUMINAMATH_CALUDE_rectangular_table_capacity_l4017_401707


namespace NUMINAMATH_CALUDE_largest_angle_convex_hexagon_l4017_401732

/-- The measure of the largest angle in a convex hexagon with specific interior angles -/
theorem largest_angle_convex_hexagon :
  ∀ x : ℝ,
  (x + 2) + (2 * x + 4) + (3 * x - 6) + (4 * x + 8) + (5 * x - 10) + (6 * x + 12) = 720 →
  max (x + 2) (max (2 * x + 4) (max (3 * x - 6) (max (4 * x + 8) (max (5 * x - 10) (6 * x + 12))))) = 215 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_convex_hexagon_l4017_401732


namespace NUMINAMATH_CALUDE_blood_expires_february_5_l4017_401722

def seconds_per_day : ℕ := 24 * 60 * 60

def february_days : ℕ := 28

def blood_expiration_seconds : ℕ := Nat.factorial 9

def days_until_expiration : ℕ := blood_expiration_seconds / seconds_per_day

theorem blood_expires_february_5 :
  days_until_expiration = 4 →
  (1 : ℕ) + days_until_expiration = 5 :=
by sorry

end NUMINAMATH_CALUDE_blood_expires_february_5_l4017_401722


namespace NUMINAMATH_CALUDE_triangle_covering_theorem_l4017_401726

/-- A triangle represented by its vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A convex polygon represented by its vertices -/
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)

/-- Predicate to check if a triangle covers a convex polygon -/
def covers (t : Triangle) (p : ConvexPolygon) : Prop := sorry

/-- Predicate to check if two triangles are congruent -/
def congruent (t1 t2 : Triangle) : Prop := sorry

/-- Predicate to check if a line is parallel to or coincident with a side of a polygon -/
def parallel_or_coincident_with_side (line : ℝ × ℝ → ℝ × ℝ → Prop) (p : ConvexPolygon) : Prop := sorry

theorem triangle_covering_theorem (ABC : Triangle) (M : ConvexPolygon) :
  covers ABC M →
  ∃ T : Triangle, congruent T ABC ∧ covers T M ∧
    ∃ side : ℝ × ℝ → ℝ × ℝ → Prop, parallel_or_coincident_with_side side M :=
by sorry

end NUMINAMATH_CALUDE_triangle_covering_theorem_l4017_401726


namespace NUMINAMATH_CALUDE_hyperbola_tangent_orthogonal_l4017_401704

/-- Hyperbola C: 2x^2 - y^2 = 1 -/
def Hyperbola (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1

/-- Circle: x^2 + y^2 = 1 -/
def UnitCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Line with slope k passing through (x, y) -/
def Line (k b x y : ℝ) : Prop := y = k * x + b

/-- Tangent condition for a line to the unit circle -/
def IsTangent (k b : ℝ) : Prop := b^2 = k^2 + 1

/-- Perpendicularity condition for two vectors -/
def IsOrthogonal (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

/-- Main theorem -/
theorem hyperbola_tangent_orthogonal (k b x1 y1 x2 y2 : ℝ) :
  |k| < Real.sqrt 2 →
  Hyperbola x1 y1 →
  Hyperbola x2 y2 →
  Line k b x1 y1 →
  Line k b x2 y2 →
  IsTangent k b →
  IsOrthogonal x1 y1 x2 y2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_tangent_orthogonal_l4017_401704


namespace NUMINAMATH_CALUDE_nonagon_arithmetic_mean_property_l4017_401768

/-- A regular nonagon with numbers placed on its vertices -/
structure NumberedNonagon where
  vertices : Fin 9 → ℕ
  is_sequential : ∀ i : Fin 9, vertices i = 2016 + i.val

/-- Three vertices of a regular nonagon form an equilateral triangle if they are equally spaced -/
def forms_equilateral_triangle (i j k : Fin 9) : Prop :=
  (j - i) % 9 = (k - j) % 9 ∧ (k - j) % 9 = (i - k) % 9

/-- The arithmetic mean property for three numbers -/
def satisfies_arithmetic_mean (a b c : ℕ) : Prop :=
  2 * b = a + c

theorem nonagon_arithmetic_mean_property (n : NumberedNonagon) :
  ∀ i j k : Fin 9, forms_equilateral_triangle i j k →
    satisfies_arithmetic_mean (n.vertices i) (n.vertices j) (n.vertices k) :=
sorry

end NUMINAMATH_CALUDE_nonagon_arithmetic_mean_property_l4017_401768


namespace NUMINAMATH_CALUDE_meaningful_iff_condition_l4017_401774

def is_meaningful (x : ℝ) : Prop :=
  x ≥ -1 ∧ x ≠ 0

theorem meaningful_iff_condition (x : ℝ) :
  is_meaningful x ↔ (∃ y : ℝ, y^2 = x + 1) ∧ x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_iff_condition_l4017_401774


namespace NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l4017_401783

-- Define the set of points satisfying the inequalities
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 > -1/2 * p.1 + 6 ∧ p.2 > 3 * p.1 - 4}

-- Define the first quadrant
def Q1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}

-- Define the second quadrant
def Q2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0}

-- Theorem statement
theorem points_in_quadrants_I_and_II : S ⊆ Q1 ∪ Q2 := by
  sorry


end NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l4017_401783


namespace NUMINAMATH_CALUDE_opposite_of_negative_four_thirds_l4017_401750

theorem opposite_of_negative_four_thirds :
  -(-(4/3 : ℚ)) = 4/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_four_thirds_l4017_401750


namespace NUMINAMATH_CALUDE_max_value_sum_of_inverses_l4017_401708

theorem max_value_sum_of_inverses (a b : ℝ) (h : a + b = 4) :
  (∀ x y : ℝ, x + y = 4 → 1 / (x^2 + 1) + 1 / (y^2 + 1) ≤ 1 / (a^2 + 1) + 1 / (b^2 + 1)) →
  1 / (a^2 + 1) + 1 / (b^2 + 1) = (Real.sqrt 5 + 2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_inverses_l4017_401708


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l4017_401711

theorem quadratic_always_positive (n : ℤ) : 6 * n^2 - 7 * n + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l4017_401711


namespace NUMINAMATH_CALUDE_cos_angle_minus_pi_half_l4017_401766

/-- 
Given an angle α in a plane rectangular coordinate system whose terminal side 
passes through the point (4, -3), prove that cos(α - π/2) = -3/5.
-/
theorem cos_angle_minus_pi_half (α : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 4 ∧ r * Real.sin α = -3) →
  Real.cos (α - π/2) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_minus_pi_half_l4017_401766


namespace NUMINAMATH_CALUDE_division_of_fraction_by_integer_l4017_401718

theorem division_of_fraction_by_integer : 
  (3 : ℚ) / 7 / 4 = (3 : ℚ) / 28 := by
sorry

end NUMINAMATH_CALUDE_division_of_fraction_by_integer_l4017_401718


namespace NUMINAMATH_CALUDE_max_points_for_successful_teams_l4017_401715

/-- Represents the number of teams in the tournament -/
def num_teams : ℕ := 15

/-- Represents the number of teams that scored at least N points -/
def num_successful_teams : ℕ := 6

/-- Represents the points awarded for a win -/
def win_points : ℕ := 3

/-- Represents the points awarded for a draw -/
def draw_points : ℕ := 1

/-- Represents the points awarded for a loss -/
def loss_points : ℕ := 0

/-- Theorem stating the maximum value of N -/
theorem max_points_for_successful_teams :
  ∃ (N : ℕ), 
    (∀ (n : ℕ), n > N → 
      ¬∃ (team_scores : Fin num_teams → ℕ),
        (∀ i j, i ≠ j → team_scores i + team_scores j ≤ win_points) ∧
        (∃ (successful : Fin num_teams → Prop),
          (∃ (k : Fin num_successful_teams), ∀ i, successful i ↔ team_scores i ≥ n))) ∧
    (∃ (team_scores : Fin num_teams → ℕ),
      (∀ i j, i ≠ j → team_scores i + team_scores j ≤ win_points) ∧
      (∃ (successful : Fin num_teams → Prop),
        (∃ (k : Fin num_successful_teams), ∀ i, successful i ↔ team_scores i ≥ N))) ∧
    N = 34 := by
  sorry

end NUMINAMATH_CALUDE_max_points_for_successful_teams_l4017_401715


namespace NUMINAMATH_CALUDE_cubic_roots_eighth_power_sum_l4017_401793

theorem cubic_roots_eighth_power_sum (r s : ℂ) : 
  (r^3 - r^2 * Real.sqrt 5 - r + 1 = 0) → 
  (s^3 - s^2 * Real.sqrt 5 - s + 1 = 0) → 
  r^8 + s^8 = 47 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_eighth_power_sum_l4017_401793


namespace NUMINAMATH_CALUDE_infinite_geometric_sum_l4017_401754

/-- The sum of an infinite geometric sequence with first term 1 and common ratio -1/2 is 2/3 -/
theorem infinite_geometric_sum : 
  ∀ (a : ℕ → ℚ), 
  (a 0 = 1) → 
  (∀ n : ℕ, a (n + 1) = a n * (-1/2)) → 
  (∑' n, a n) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_infinite_geometric_sum_l4017_401754


namespace NUMINAMATH_CALUDE_tangent_line_sum_l4017_401728

/-- Given a function f where the tangent line at x=2 is 2x+y-3=0, prove f(2) + f'(2) = -3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : DifferentiableAt ℝ f 2) 
  (h_tangent : ∀ x y, y = f x → (x = 2 → 2*x + y - 3 = 0)) : 
  f 2 + deriv f 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l4017_401728


namespace NUMINAMATH_CALUDE_solve_x_equation_l4017_401798

theorem solve_x_equation : ∃ x : ℝ, (0.6 * x = x / 3 + 110) ∧ x = 412.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_x_equation_l4017_401798


namespace NUMINAMATH_CALUDE_existence_of_subset_with_property_P_l4017_401723

-- Define the property P for a subset A and a natural number m
def property_P (A : Set ℕ) (m : ℕ) : Prop :=
  ∀ k : ℕ, ∃ a : ℕ → ℕ, 
    (∀ i, i < k → a i ∈ A) ∧
    (∀ i, i < k - 1 → 1 ≤ a (i + 1) - a i ∧ a (i + 1) - a i ≤ m)

-- Main theorem
theorem existence_of_subset_with_property_P 
  (r : ℕ) (partition : Fin r → Set ℕ) 
  (partition_properties : 
    (∀ i j, i ≠ j → partition i ∩ partition j = ∅) ∧ 
    (⋃ i, partition i) = Set.univ) :
  ∃ (i : Fin r) (m : ℕ), property_P (partition i) m :=
sorry

end NUMINAMATH_CALUDE_existence_of_subset_with_property_P_l4017_401723


namespace NUMINAMATH_CALUDE_intersection_when_a_is_3_subset_iff_a_geq_4_l4017_401792

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x - a < 0}

-- Theorem 1: A ∩ B = [1,3) when a = 3
theorem intersection_when_a_is_3 : A ∩ B 3 = Set.Icc 1 3 := by sorry

-- Theorem 2: A ⊆ B if and only if a ≥ 4
theorem subset_iff_a_geq_4 : ∀ a : ℝ, A ⊆ B a ↔ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_3_subset_iff_a_geq_4_l4017_401792


namespace NUMINAMATH_CALUDE_farm_animals_count_l4017_401758

/-- Represents a farm with hens, cows, and ducks -/
structure Farm where
  hens : ℕ
  cows : ℕ
  ducks : ℕ

/-- The total number of heads in the farm -/
def total_heads (f : Farm) : ℕ := f.hens + f.cows + f.ducks

/-- The total number of feet in the farm -/
def total_feet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows + 2 * f.ducks

/-- Theorem stating the number of cows and the sum of hens and ducks in the farm -/
theorem farm_animals_count (f : Farm) 
  (h1 : total_heads f = 72) 
  (h2 : total_feet f = 212) : 
  f.cows = 34 ∧ f.hens + f.ducks = 38 := by
  sorry


end NUMINAMATH_CALUDE_farm_animals_count_l4017_401758


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l4017_401752

/-- The sum of the squares of the coefficients of the fully simplified expression 
    3(x^3 - 4x + 5) - 5(2x^3 - x^2 + 3x - 2) is equal to 1428 -/
theorem sum_of_squares_of_coefficients : ∃ (a b c d : ℤ),
  (∀ x : ℝ, 3 * (x^3 - 4*x + 5) - 5 * (2*x^3 - x^2 + 3*x - 2) = a*x^3 + b*x^2 + c*x + d) ∧
  a^2 + b^2 + c^2 + d^2 = 1428 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l4017_401752


namespace NUMINAMATH_CALUDE_hiking_trip_days_l4017_401779

/-- Represents the hiking trip scenario -/
structure HikingTrip where
  rateUp : ℝ
  rateDown : ℝ
  distanceDown : ℝ
  days : ℝ

/-- The hiking trip satisfies the given conditions -/
def validHikingTrip (trip : HikingTrip) : Prop :=
  trip.rateUp = 6 ∧
  trip.rateDown = 1.5 * trip.rateUp ∧
  trip.distanceDown = 18 ∧
  trip.rateUp * trip.days = trip.rateDown * trip.days

/-- The number of days for the hiking trip is 2 -/
theorem hiking_trip_days (trip : HikingTrip) (h : validHikingTrip trip) : trip.days = 2 := by
  sorry


end NUMINAMATH_CALUDE_hiking_trip_days_l4017_401779


namespace NUMINAMATH_CALUDE_money_ratio_l4017_401700

/-- Prove that the ratio of Alison's money to Brittany's money is 1:2 -/
theorem money_ratio (kent_money : ℝ) (brooke_money : ℝ) (brittany_money : ℝ) (alison_money : ℝ)
  (h1 : kent_money = 1000)
  (h2 : brooke_money = 2 * kent_money)
  (h3 : brittany_money = 4 * brooke_money)
  (h4 : alison_money = 4000) :
  alison_money / brittany_money = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_money_ratio_l4017_401700


namespace NUMINAMATH_CALUDE_min_value_fraction_l4017_401735

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hab : a * b = 1) :
  (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l4017_401735


namespace NUMINAMATH_CALUDE_sum_negative_implies_one_negative_l4017_401744

theorem sum_negative_implies_one_negative (a b : ℚ) : a + b < 0 → a < 0 ∨ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_negative_implies_one_negative_l4017_401744


namespace NUMINAMATH_CALUDE_smallest_integer_m_l4017_401773

theorem smallest_integer_m (x y m : ℝ) : 
  (2 * x + y = 4) →
  (x + 2 * y = -3 * m + 2) →
  (x - y > -3/2) →
  (∀ k : ℤ, k < m → ¬(∃ x y : ℝ, 2 * x + y = 4 ∧ x + 2 * y = -3 * (k : ℝ) + 2 ∧ x - y > -3/2)) →
  m = -1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_m_l4017_401773


namespace NUMINAMATH_CALUDE_binomial_odd_iff_binary_condition_l4017_401717

def has_one_at_position (m : ℕ) (pos : ℕ) : Prop :=
  (m / 2^pos) % 2 = 1

theorem binomial_odd_iff_binary_condition (n k : ℕ) :
  Nat.choose n k % 2 = 1 ↔ ∀ pos, has_one_at_position k pos → has_one_at_position n pos :=
sorry

end NUMINAMATH_CALUDE_binomial_odd_iff_binary_condition_l4017_401717


namespace NUMINAMATH_CALUDE_derivative_ln_2x_squared_minus_4_l4017_401713

open Real

theorem derivative_ln_2x_squared_minus_4 (x : ℝ) (h : x^2 ≠ 2) :
  deriv (λ x => log (2 * x^2 - 4)) x = 2 * x / (x^2 - 2) :=
by sorry

end NUMINAMATH_CALUDE_derivative_ln_2x_squared_minus_4_l4017_401713


namespace NUMINAMATH_CALUDE_min_value_theorem_l4017_401706

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ (min_val : ℝ), min_val = 3 + 2 * Real.sqrt 2 ∧
  ∀ (z : ℝ), z = 2 / x + 1 / y → z ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4017_401706


namespace NUMINAMATH_CALUDE_rotation180_maps_points_and_is_isometry_l4017_401724

-- Define the points
def A : ℝ × ℝ := (-2, 1)
def A' : ℝ × ℝ := (2, -1)
def B : ℝ × ℝ := (-1, 4)
def B' : ℝ × ℝ := (1, -4)

-- Define the rotation function
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Theorem statement
theorem rotation180_maps_points_and_is_isometry :
  (rotate180 A = A') ∧ 
  (rotate180 B = B') ∧ 
  (∀ p q : ℝ × ℝ, dist p q = dist (rotate180 p) (rotate180 q)) := by
  sorry


end NUMINAMATH_CALUDE_rotation180_maps_points_and_is_isometry_l4017_401724


namespace NUMINAMATH_CALUDE_infinite_sum_equals_nine_eighties_l4017_401730

/-- The infinite sum of 2n / (n^4 + 16) from n=1 to infinity equals 9/80 -/
theorem infinite_sum_equals_nine_eighties :
  (∑' n : ℕ+, (2 * n : ℝ) / (n^4 + 16)) = 9 / 80 := by sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_nine_eighties_l4017_401730


namespace NUMINAMATH_CALUDE_intersection_M_N_l4017_401721

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4017_401721


namespace NUMINAMATH_CALUDE_quadratic_equations_sum_l4017_401739

theorem quadratic_equations_sum (x y : ℝ) : 
  9 * x^2 - 36 * x - 81 = 0 → 
  y^2 + 6 * y + 9 = 0 → 
  (x + y = -1 + Real.sqrt 13) ∨ (x + y = -1 - Real.sqrt 13) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_sum_l4017_401739


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l4017_401789

theorem greatest_integer_fraction_inequality :
  ∀ x : ℤ, (7 : ℚ) / 9 > (x : ℚ) / 13 ↔ x ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l4017_401789


namespace NUMINAMATH_CALUDE_morning_ride_l4017_401716

theorem morning_ride (x : ℝ) (h : x + 5*x = 12) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_morning_ride_l4017_401716


namespace NUMINAMATH_CALUDE_cat_in_bag_change_l4017_401709

theorem cat_in_bag_change (p : ℕ) (h : 0 < p ∧ p ≤ 1000) : 
  ∃ (change : ℕ), change = 1000 - p := by
  sorry

end NUMINAMATH_CALUDE_cat_in_bag_change_l4017_401709


namespace NUMINAMATH_CALUDE_work_completion_equivalence_l4017_401795

/-- The number of days needed for the first group to complete the work -/
def days_first_group : ℕ := 96

/-- The number of men in the second group -/
def men_second_group : ℕ := 40

/-- The number of days needed for the second group to complete the work -/
def days_second_group : ℕ := 60

/-- The number of men in the first group -/
def men_first_group : ℕ := 25

theorem work_completion_equivalence :
  men_first_group * days_first_group = men_second_group * days_second_group :=
by sorry

#check work_completion_equivalence

end NUMINAMATH_CALUDE_work_completion_equivalence_l4017_401795


namespace NUMINAMATH_CALUDE_congruence_solution_l4017_401751

theorem congruence_solution : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 13258 [MOD 16] := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l4017_401751


namespace NUMINAMATH_CALUDE_oplus_neg_two_three_oplus_inequality_l4017_401765

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := 2 * a - 3 * b

-- Theorem 1: (-2) ⊕ 3 = -13
theorem oplus_neg_two_three : oplus (-2) 3 = -13 := by sorry

-- Theorem 2: For all real x, ((-3/2x+1) ⊕ (-1-2x)) > ((3x-2) ⊕ (x+1))
theorem oplus_inequality (x : ℝ) : oplus (-3/2*x+1) (-1-2*x) > oplus (3*x-2) (x+1) := by sorry

end NUMINAMATH_CALUDE_oplus_neg_two_three_oplus_inequality_l4017_401765


namespace NUMINAMATH_CALUDE_probability_three_girls_out_of_six_l4017_401791

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem probability_three_girls_out_of_six :
  binomial_probability 6 3 (1/2) = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_girls_out_of_six_l4017_401791


namespace NUMINAMATH_CALUDE_zero_in_interval_l4017_401702

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

theorem zero_in_interval :
  ∃ c : ℝ, c ∈ Set.Ioo 1 2 ∧ f c = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l4017_401702


namespace NUMINAMATH_CALUDE_student_survey_l4017_401714

theorem student_survey (french_english : ℕ) (french_not_english : ℕ) 
  (percent_not_french : ℚ) :
  french_english = 20 →
  french_not_english = 60 →
  percent_not_french = 60 / 100 →
  ∃ (total : ℕ), total = 200 ∧ 
    (french_english + french_not_english : ℚ) = (1 - percent_not_french) * total :=
by sorry

end NUMINAMATH_CALUDE_student_survey_l4017_401714


namespace NUMINAMATH_CALUDE_field_length_proof_l4017_401771

theorem field_length_proof (l w : ℝ) (h1 : l = 2 * w) (h2 : (8 * 8) = (1 / 98) * (l * w)) : l = 112 := by
  sorry

end NUMINAMATH_CALUDE_field_length_proof_l4017_401771


namespace NUMINAMATH_CALUDE_max_F_value_l4017_401746

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  units : Nat
  is_four_digit : thousands ≥ 1 ∧ thousands ≤ 9

/-- Checks if a number is an "eternal number" -/
def is_eternal (n : FourDigitNumber) : Prop :=
  n.hundreds + n.tens + n.units = 12

/-- Swaps digits as described in the problem -/
def swap_digits (n : FourDigitNumber) : FourDigitNumber :=
  { thousands := n.hundreds
  , hundreds := n.thousands
  , tens := n.units
  , units := n.tens
  , is_four_digit := by sorry }

/-- Calculates F(M) as defined in the problem -/
def F (m : FourDigitNumber) : Int :=
  let n := swap_digits m
  let m_val := 1000 * m.thousands + 100 * m.hundreds + 10 * m.tens + m.units
  let n_val := 1000 * n.thousands + 100 * n.hundreds + 10 * n.tens + n.units
  (m_val - n_val) / 9

/-- Main theorem -/
theorem max_F_value (m : FourDigitNumber) 
  (h1 : is_eternal m)
  (h2 : m.thousands = m.hundreds - m.units)
  (h3 : (F m) % 9 = 0) :
  F m ≤ 9 ∧ ∃ (m' : FourDigitNumber), F m' = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_F_value_l4017_401746


namespace NUMINAMATH_CALUDE_wide_flag_height_l4017_401796

/-- Given the following conditions:
  - Total fabric: 1000 square feet
  - Square flags: 4 feet by 4 feet
  - Wide rectangular flags: 5 feet by unknown height
  - Tall rectangular flags: 3 feet by 5 feet
  - 16 square flags made
  - 20 wide flags made
  - 10 tall flags made
  - 294 square feet of fabric left
Prove that the height of the wide rectangular flags is 3 feet. -/
theorem wide_flag_height (total_fabric : ℝ) (square_side : ℝ) (wide_width : ℝ) 
  (tall_width tall_height : ℝ) (num_square num_wide num_tall : ℕ) (fabric_left : ℝ)
  (h_total : total_fabric = 1000)
  (h_square : square_side = 4)
  (h_wide_width : wide_width = 5)
  (h_tall : tall_width = 3 ∧ tall_height = 5)
  (h_num_square : num_square = 16)
  (h_num_wide : num_wide = 20)
  (h_num_tall : num_tall = 10)
  (h_fabric_left : fabric_left = 294) :
  ∃ (wide_height : ℝ), wide_height = 3 ∧ 
  total_fabric = num_square * square_side^2 + num_wide * wide_width * wide_height + 
                 num_tall * tall_width * tall_height + fabric_left :=
by sorry

end NUMINAMATH_CALUDE_wide_flag_height_l4017_401796
