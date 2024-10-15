import Mathlib

namespace NUMINAMATH_CALUDE_fermat_sum_divisibility_l3655_365561

theorem fermat_sum_divisibility (x y z : ℤ) 
  (hx : ¬ 7 ∣ x) (hy : ¬ 7 ∣ y) (hz : ¬ 7 ∣ z)
  (h_sum : (7:ℤ)^3 ∣ (x^7 + y^7 + z^7)) :
  (7:ℤ)^2 ∣ (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_fermat_sum_divisibility_l3655_365561


namespace NUMINAMATH_CALUDE_james_out_of_pocket_cost_l3655_365560

theorem james_out_of_pocket_cost (doctor_charge : ℝ) (insurance_coverage_percent : ℝ) 
  (h1 : doctor_charge = 300)
  (h2 : insurance_coverage_percent = 80) :
  doctor_charge * (1 - insurance_coverage_percent / 100) = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_out_of_pocket_cost_l3655_365560


namespace NUMINAMATH_CALUDE_website_earnings_per_visit_l3655_365585

/-- Calculates the earnings per visit for a website -/
def earnings_per_visit (monthly_visits : ℕ) (daily_earnings : ℚ) : ℚ :=
  (30 * daily_earnings) / monthly_visits

/-- Theorem: Given 30,000 monthly visits and $10 daily earnings, the earnings per visit is $0.01 -/
theorem website_earnings_per_visit : 
  earnings_per_visit 30000 10 = 1/100 := by
  sorry

end NUMINAMATH_CALUDE_website_earnings_per_visit_l3655_365585


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_plus_reciprocal_l3655_365587

def z : ℂ := 1 + Complex.I

theorem imaginary_part_of_z_plus_reciprocal (z : ℂ) (h : z = 1 + Complex.I) :
  Complex.im (z + z⁻¹) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_plus_reciprocal_l3655_365587


namespace NUMINAMATH_CALUDE_lucas_initial_money_l3655_365524

/-- Proves that Lucas' initial amount of money is $20 given the problem conditions --/
theorem lucas_initial_money :
  ∀ (initial_money : ℕ) 
    (avocado_count : ℕ) 
    (avocado_price : ℕ) 
    (change : ℕ),
  avocado_count = 3 →
  avocado_price = 2 →
  change = 14 →
  initial_money = avocado_count * avocado_price + change →
  initial_money = 20 := by
sorry

end NUMINAMATH_CALUDE_lucas_initial_money_l3655_365524


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3655_365539

theorem system_of_equations_solution :
  ∃ (x y : ℝ), (2 * x + y = 4 ∧ x + 2 * y = 5) ∧ (x = 1 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3655_365539


namespace NUMINAMATH_CALUDE_complex_number_modulus_l3655_365525

theorem complex_number_modulus : 
  let z : ℂ := (1 + 3*I) / (1 - I)
  Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l3655_365525


namespace NUMINAMATH_CALUDE_pencil_count_l3655_365541

/-- The total number of colored pencils Cheryl, Cyrus, and Madeline have -/
def total_pencils (cheryl : ℕ) (cyrus : ℕ) (madeline : ℕ) : ℕ :=
  cheryl + cyrus + madeline

/-- Theorem stating the total number of colored pencils given the conditions -/
theorem pencil_count :
  ∀ (cheryl cyrus madeline : ℕ),
    cheryl = 3 * cyrus →
    madeline = 63 →
    cheryl = 2 * madeline →
    total_pencils cheryl cyrus madeline = 231 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l3655_365541


namespace NUMINAMATH_CALUDE_concert_tickets_theorem_l3655_365599

/-- Represents the ticket sales for a concert --/
structure ConcertTickets where
  regularPrice : ℕ
  discountGroup1Size : ℕ
  discountGroup1Percentage : ℕ
  discountGroup2Size : ℕ
  discountGroup2Percentage : ℕ
  totalRevenue : ℕ

/-- Calculates the total number of people who bought tickets --/
def totalPeople (ct : ConcertTickets) : ℕ :=
  ct.discountGroup1Size + ct.discountGroup2Size +
  ((ct.totalRevenue -
    (ct.discountGroup1Size * (ct.regularPrice * (100 - ct.discountGroup1Percentage) / 100)) -
    (ct.discountGroup2Size * (ct.regularPrice * (100 - ct.discountGroup2Percentage) / 100)))
   / ct.regularPrice)

/-- Theorem stating that given the concert conditions, 48 people bought tickets --/
theorem concert_tickets_theorem (ct : ConcertTickets)
  (h1 : ct.regularPrice = 20)
  (h2 : ct.discountGroup1Size = 10)
  (h3 : ct.discountGroup1Percentage = 40)
  (h4 : ct.discountGroup2Size = 20)
  (h5 : ct.discountGroup2Percentage = 15)
  (h6 : ct.totalRevenue = 820) :
  totalPeople ct = 48 := by
  sorry

#eval totalPeople {
  regularPrice := 20,
  discountGroup1Size := 10,
  discountGroup1Percentage := 40,
  discountGroup2Size := 20,
  discountGroup2Percentage := 15,
  totalRevenue := 820
}

end NUMINAMATH_CALUDE_concert_tickets_theorem_l3655_365599


namespace NUMINAMATH_CALUDE_power_of_product_with_exponent_l3655_365547

theorem power_of_product_with_exponent (x y : ℝ) : (-x * y^3)^2 = x^2 * y^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_with_exponent_l3655_365547


namespace NUMINAMATH_CALUDE_angela_beth_age_ratio_l3655_365517

theorem angela_beth_age_ratio :
  ∀ (angela_age beth_age : ℕ),
    (angela_age - 5 + beth_age - 5 = 45) →  -- Five years ago, sum of ages was 45
    (angela_age + 5 = 44) →                 -- In five years, Angela will be 44
    (angela_age : ℚ) / beth_age = 39 / 16   -- Ratio of current ages is 39:16
    :=
by
  sorry

end NUMINAMATH_CALUDE_angela_beth_age_ratio_l3655_365517


namespace NUMINAMATH_CALUDE_friday_pushups_equal_total_l3655_365520

def monday_pushups : ℕ := 5
def tuesday_pushups : ℕ := 7
def wednesday_pushups : ℕ := 2 * tuesday_pushups
def thursday_pushups : ℕ := (monday_pushups + tuesday_pushups + wednesday_pushups) / 2
def total_monday_to_thursday : ℕ := monday_pushups + tuesday_pushups + wednesday_pushups + thursday_pushups

theorem friday_pushups_equal_total : total_monday_to_thursday = 39 := by
  sorry

end NUMINAMATH_CALUDE_friday_pushups_equal_total_l3655_365520


namespace NUMINAMATH_CALUDE_least_rectangle_area_for_two_squares_l3655_365512

theorem least_rectangle_area_for_two_squares :
  ∃ (A : ℝ), A = Real.sqrt 2 ∧
  (∀ (a b : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ a^2 + b^2 = 1 →
    ∃ (w h : ℝ), w ≥ 0 ∧ h ≥ 0 ∧ w * h = A ∧ a ≤ w ∧ b ≤ h) ∧
  (∀ (A' : ℝ), A' < A →
    ∃ (a b : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ a^2 + b^2 = 1 ∧
      ∀ (w h : ℝ), w ≥ 0 ∧ h ≥ 0 ∧ w * h = A' → (a > w ∨ b > h)) :=
by sorry

end NUMINAMATH_CALUDE_least_rectangle_area_for_two_squares_l3655_365512


namespace NUMINAMATH_CALUDE_scaled_prism_marbles_l3655_365564

/-- Represents a triangular prism-shaped container -/
structure TriangularPrism where
  baseArea : ℝ
  height : ℝ
  marbles : ℕ

/-- Scales the dimensions of a triangular prism by a given factor -/
def scalePrism (p : TriangularPrism) (factor : ℝ) : TriangularPrism :=
  { baseArea := p.baseArea * factor^2
  , height := p.height * factor
  , marbles := p.marbles }

/-- Theorem: Scaling a triangular prism by a factor of 2 results in 8 times the marbles -/
theorem scaled_prism_marbles (p : TriangularPrism) :
  (scalePrism p 2).marbles = 8 * p.marbles :=
by sorry

end NUMINAMATH_CALUDE_scaled_prism_marbles_l3655_365564


namespace NUMINAMATH_CALUDE_triangle_third_side_valid_third_side_l3655_365531

/-- Checks if three lengths can form a valid triangle -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_third_side (x : ℝ) : 
  (is_valid_triangle 7 10 x ∧ x > 0) ↔ (3 < x ∧ x < 17) :=
sorry

theorem valid_third_side : 
  is_valid_triangle 7 10 11 ∧ 
  ¬(is_valid_triangle 7 10 20) ∧ 
  ¬(is_valid_triangle 7 10 3) ∧ 
  ¬(is_valid_triangle 7 10 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_valid_third_side_l3655_365531


namespace NUMINAMATH_CALUDE_circle_condition_l3655_365579

theorem circle_condition (f : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ ∀ (x y : ℝ), x^2 + y^2 - 4*x + 6*y + f = 0 ↔ (x - 2)^2 + (y + 3)^2 = r^2) ↔
  f < 13 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l3655_365579


namespace NUMINAMATH_CALUDE_distance_to_reflection_distance_z_to_z_reflected_l3655_365578

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection (x y : ℝ) : 
  let z : ℝ × ℝ := (x, y)
  let z_reflected : ℝ × ℝ := (x, -y)
  Real.sqrt ((z.1 - z_reflected.1)^2 + (z.2 - z_reflected.2)^2) = 2 * |y| :=
by sorry

/-- The specific case for the point Z(5, 2) --/
theorem distance_z_to_z_reflected : 
  let z : ℝ × ℝ := (5, 2)
  let z_reflected : ℝ × ℝ := (5, -2)
  Real.sqrt ((z.1 - z_reflected.1)^2 + (z.2 - z_reflected.2)^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_reflection_distance_z_to_z_reflected_l3655_365578


namespace NUMINAMATH_CALUDE_range_of_a_l3655_365575

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 2*x + 2
  else x + a/x + 3*a

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ a ∈ Set.Iio 0 ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3655_365575


namespace NUMINAMATH_CALUDE_equation_solution_l3655_365569

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (27 + 9*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 :=
by
  use 9
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3655_365569


namespace NUMINAMATH_CALUDE_negation_of_implication_negation_of_positive_square_l3655_365532

theorem negation_of_implication (P Q : Prop) :
  ¬(P → Q) ↔ (P ∧ ¬Q) :=
by sorry

theorem negation_of_positive_square :
  ¬(∀ x : ℝ, x > 0 → x^2 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_negation_of_positive_square_l3655_365532


namespace NUMINAMATH_CALUDE_choir_arrangements_choir_arrangement_count_l3655_365577

theorem choir_arrangements (total_boys : Nat) (total_girls : Nat) 
  (selected_boys : Nat) (selected_girls : Nat) : Nat :=
  let boy_selections := Nat.choose total_boys selected_boys
  let girl_selections := Nat.choose total_girls selected_girls
  let boy_arrangements := Nat.factorial selected_boys
  let girl_positions := Nat.factorial (selected_boys + 1) / Nat.factorial (selected_boys + 1 - selected_girls)
  boy_selections * girl_selections * boy_arrangements * girl_positions

theorem choir_arrangement_count : 
  choir_arrangements 4 3 2 2 = 216 := by sorry

end NUMINAMATH_CALUDE_choir_arrangements_choir_arrangement_count_l3655_365577


namespace NUMINAMATH_CALUDE_only_negative_number_l3655_365554

theorem only_negative_number (a b c d : ℝ) : 
  a = 2023 → b = -2023 → c = 1 / 2023 → d = 0 →
  (b < 0 ∧ a > 0 ∧ c > 0 ∧ d = 0) := by
  sorry

end NUMINAMATH_CALUDE_only_negative_number_l3655_365554


namespace NUMINAMATH_CALUDE_david_meets_paul_probability_l3655_365538

/-- The probability of David arriving while Paul is still present -/
theorem david_meets_paul_probability : 
  let arrival_window : ℝ := 60  -- 60 minutes between 1:00 PM and 2:00 PM
  let paul_wait_time : ℝ := 30  -- Paul waits for 30 minutes
  let favorable_area : ℝ := (arrival_window - paul_wait_time) * paul_wait_time / 2 + paul_wait_time * paul_wait_time
  let total_area : ℝ := arrival_window * arrival_window
  (favorable_area / total_area) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_david_meets_paul_probability_l3655_365538


namespace NUMINAMATH_CALUDE_max_distance_C₁_intersections_l3655_365516

noncomputable section

-- Define the curves
def C₁ (t α : ℝ) : ℝ × ℝ := (t * Real.cos α, t * Real.sin α)

def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * Real.sqrt 3 * Real.sin θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

def C₃ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * Real.cos θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the set of valid parameters
def ValidParams : Set ℝ := {α | 0 ≤ α ∧ α ≤ Real.pi}

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem max_distance_C₁_intersections :
  ∃ (max_dist : ℝ), max_dist = 4 ∧
  ∀ (t₁ t₂ θ₁ θ₂ α : ℝ), α ∈ ValidParams →
    distance (C₁ t₁ α) (C₂ θ₁) = 0 →
    distance (C₁ t₂ α) (C₃ θ₂) = 0 →
    distance (C₁ t₁ α) (C₁ t₂ α) ≤ max_dist :=
sorry

end

end NUMINAMATH_CALUDE_max_distance_C₁_intersections_l3655_365516


namespace NUMINAMATH_CALUDE_triathlon_speed_l3655_365583

/-- Triathlon problem -/
theorem triathlon_speed (total_time : ℝ) (swim_dist swim_speed : ℝ) (run_dist run_speed : ℝ) (bike_dist : ℝ) :
  total_time = 2 →
  swim_dist = 0.5 →
  swim_speed = 3 →
  run_dist = 4 →
  run_speed = 8 →
  bike_dist = 20 →
  (swim_dist / swim_speed + run_dist / run_speed + bike_dist / (bike_dist / (total_time - (swim_dist / swim_speed + run_dist / run_speed))) = total_time) →
  bike_dist / (total_time - (swim_dist / swim_speed + run_dist / run_speed)) = 15 := by
sorry


end NUMINAMATH_CALUDE_triathlon_speed_l3655_365583


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3655_365504

theorem geometric_sequence_common_ratio 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 27) 
  (h₂ : a₂ = 54) 
  (h₃ : a₃ = 108) 
  (h_geom : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) : 
  ∃ r : ℝ, r = 2 ∧ a₂ = a₁ * r ∧ a₃ = a₂ * r :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3655_365504


namespace NUMINAMATH_CALUDE_calculation_problems_l3655_365523

theorem calculation_problems :
  (∀ a : ℝ, a^3 * a + (-a^2)^3 / a^2 = 0) ∧
  (Real.sqrt 5 - Real.sqrt 2) * (Real.sqrt 5 + Real.sqrt 2) + (Real.sqrt 3 - 1)^2 = 7 - 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_calculation_problems_l3655_365523


namespace NUMINAMATH_CALUDE_fourth_grade_students_l3655_365542

/-- The number of students in fourth grade at the start of the year. -/
def initial_students : ℕ := 33

/-- The number of students who left during the year. -/
def students_left : ℕ := 18

/-- The number of new students who came during the year. -/
def new_students : ℕ := 14

/-- The number of students at the end of the year. -/
def final_students : ℕ := 29

theorem fourth_grade_students : 
  initial_students - students_left + new_students = final_students := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l3655_365542


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l3655_365528

/-- A rectangular garden with length three times its width and area 507 square meters has a width of 13 meters. -/
theorem rectangular_garden_width (width length area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 507 →
  width = 13 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l3655_365528


namespace NUMINAMATH_CALUDE_smallest_x_y_sum_l3655_365543

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

def is_fourth_power (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 4

theorem smallest_x_y_sum :
  ∃! (x y : ℕ), 
    x > 0 ∧ y > 0 ∧
    is_square (720 * x) ∧
    is_fourth_power (720 * y) ∧
    (∀ (x' y' : ℕ), x' > 0 ∧ y' > 0 ∧ 
      is_square (720 * x') ∧ is_fourth_power (720 * y') → 
      x ≤ x' ∧ y ≤ y') ∧
    x + y = 1130 :=
  sorry

end NUMINAMATH_CALUDE_smallest_x_y_sum_l3655_365543


namespace NUMINAMATH_CALUDE_max_triplets_coordinate_plane_l3655_365536

/-- Given 100 points on a coordinate plane, prove that the maximum number of triplets (A, B, C) 
    where A and B have the same y-coordinate and B and C have the same x-coordinate is 8100. -/
theorem max_triplets_coordinate_plane (points : Finset (ℝ × ℝ)) 
    (h : points.card = 100) : 
  (Finset.sum points (fun B => 
    (points.filter (fun A => A.2 = B.2)).card * 
    (points.filter (fun C => C.1 = B.1)).card
  )) ≤ 8100 := by
  sorry

end NUMINAMATH_CALUDE_max_triplets_coordinate_plane_l3655_365536


namespace NUMINAMATH_CALUDE_yazhong_point_problem1_yazhong_point_problem2_yazhong_point_problem3_1_yazhong_point_problem3_2_l3655_365567

/-- Definition of Yazhong point -/
def is_yazhong_point (a b m : ℝ) : Prop :=
  |m - a| = |m - b|

/-- Problem 1 -/
theorem yazhong_point_problem1 :
  is_yazhong_point (-5) 1 (-2) :=
sorry

/-- Problem 2 -/
theorem yazhong_point_problem2 :
  is_yazhong_point (-5/2) (13/2) 2 ∧ |(-5/2) - (13/2)| = 9 :=
sorry

/-- Problem 3 part 1 -/
theorem yazhong_point_problem3_1 :
  (∃ b : ℝ, -4 ≤ b ∧ b ≤ -2 ∧ is_yazhong_point (-6) b (-5)) ∧
  (∃ b : ℝ, -4 ≤ b ∧ b ≤ -2 ∧ is_yazhong_point (-6) b (-4)) ∧
  (∀ m : ℤ, (∃ b : ℝ, -4 ≤ b ∧ b ≤ -2 ∧ is_yazhong_point (-6) b (m : ℝ)) → m = -5 ∨ m = -4) :=
sorry

/-- Problem 3 part 2 -/
theorem yazhong_point_problem3_2 :
  (∀ n : ℤ, is_yazhong_point (-6) (6 : ℝ) 0 ∧ -4 + n ≤ 6 ∧ 6 ≤ -2 + n → n = 8 ∨ n = 9 ∨ n = 10) ∧
  (∀ n : ℤ, n = 8 ∨ n = 9 ∨ n = 10 → is_yazhong_point (-6) (6 : ℝ) 0 ∧ -4 + n ≤ 6 ∧ 6 ≤ -2 + n) :=
sorry

end NUMINAMATH_CALUDE_yazhong_point_problem1_yazhong_point_problem2_yazhong_point_problem3_1_yazhong_point_problem3_2_l3655_365567


namespace NUMINAMATH_CALUDE_village_population_problem_l3655_365500

theorem village_population_problem (final_population : ℕ) 
  (h1 : final_population = 3168) : ∃ initial_population : ℕ,
  (initial_population : ℝ) * 0.9 * 0.8 = final_population ∧ 
  initial_population = 4400 := by
  sorry

end NUMINAMATH_CALUDE_village_population_problem_l3655_365500


namespace NUMINAMATH_CALUDE_coffee_ratio_problem_l3655_365551

/-- Given two types of coffee, p and v, mixed into two blends x and y, 
    prove that the ratio of p to v in y is 1 to 5. -/
theorem coffee_ratio_problem (total_p total_v x_p x_v y_p y_v : ℚ) : 
  total_p = 24 →
  total_v = 25 →
  x_p / x_v = 4 / 1 →
  x_p = 20 →
  total_p = x_p + y_p →
  total_v = x_v + y_v →
  y_p / y_v = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_coffee_ratio_problem_l3655_365551


namespace NUMINAMATH_CALUDE_inverse_sum_mod_thirteen_l3655_365509

theorem inverse_sum_mod_thirteen : 
  (((3⁻¹ : ZMod 13) + (4⁻¹ : ZMod 13) + (5⁻¹ : ZMod 13))⁻¹ : ZMod 13) = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_mod_thirteen_l3655_365509


namespace NUMINAMATH_CALUDE_expression_value_l3655_365544

theorem expression_value (a b : ℝ) (h1 : a ≠ b) 
  (h2 : 1 / (a^2 + 1) + 1 / (b^2 + 1) = 2 / (a * b + 1)) :
  1 / (a^2 + 1) + 1 / (b^2 + 1) + 2 / (a * b + 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l3655_365544


namespace NUMINAMATH_CALUDE_problem_solution_l3655_365537

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 1021 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3655_365537


namespace NUMINAMATH_CALUDE_smallest_positive_angle_l3655_365511

open Real

-- Define the equation
def equation (x : ℝ) : Prop :=
  tan (5 * x * π / 180) = (1 - sin (x * π / 180)) / (1 + sin (x * π / 180))

-- State the theorem
theorem smallest_positive_angle :
  ∃ (x : ℝ), x > 0 ∧ x < 10 ∧ equation x ∧ ∀ (y : ℝ), 0 < y ∧ y < x → ¬(equation y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_l3655_365511


namespace NUMINAMATH_CALUDE_oscillating_bounded_example_unbounded_oscillations_example_l3655_365508

-- Part a
def oscillating_bounded (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧
  ∀ ε > 0, ∀ X : ℝ, ∃ x₁ x₂ : ℝ, 
    x₁ > X ∧ x₂ > X ∧ 
    f x₁ < a + ε ∧ f x₂ > b - ε

theorem oscillating_bounded_example (a b : ℝ) (h : a < b) :
  oscillating_bounded (fun x ↦ a + (b - a) * Real.sin x ^ 2) a b :=
sorry

-- Part b
def unbounded_oscillations (f : ℝ → ℝ) : Prop :=
  ∀ M : ℝ, ∃ X : ℝ, ∀ x > X, ∃ y > x, 
    (f y > M ∧ f x < -M) ∨ (f y < -M ∧ f x > M)

theorem unbounded_oscillations_example :
  unbounded_oscillations (fun x ↦ x * Real.sin x) :=
sorry

end NUMINAMATH_CALUDE_oscillating_bounded_example_unbounded_oscillations_example_l3655_365508


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3655_365568

/-- Given two lines L1 and L2 in a 2D plane, where:
    - L1 has equation mx - m²y = 1
    - L2 is perpendicular to L1
    - L1 and L2 intersect at point P(2,1)
    Prove that the equation of L2 is x + y - 3 = 0 -/
theorem perpendicular_line_equation (m : ℝ) :
  (∀ x y, m * x - m^2 * y = 1 → x = 2 ∧ y = 1) →
  (∃ k : ℝ, k * m = -1) →
  ∀ x y, x + y - 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3655_365568


namespace NUMINAMATH_CALUDE_vector_properties_l3655_365534

def e₁ : ℝ × ℝ := (1, 0)
def e₂ : ℝ × ℝ := (0, 1)
def a : ℝ × ℝ := (3 * e₁.1 - 2 * e₂.1, 3 * e₁.2 - 2 * e₂.2)
def b : ℝ × ℝ := (4 * e₁.1 + e₂.1, 4 * e₁.2 + e₂.2)

theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 = 10) ∧
  (Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5 * Real.sqrt 2) ∧
  (((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 10 * Real.sqrt 221 / 221) := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l3655_365534


namespace NUMINAMATH_CALUDE_fourth_term_is_one_tenth_l3655_365586

def sequence_term (n : ℕ) : ℚ := 2 / (n^2 + n : ℚ)

theorem fourth_term_is_one_tenth : sequence_term 4 = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_one_tenth_l3655_365586


namespace NUMINAMATH_CALUDE_milk_water_ratio_l3655_365502

theorem milk_water_ratio (initial_volume : ℚ) (initial_milk_ratio : ℚ) (initial_water_ratio : ℚ) (added_water : ℚ) : 
  initial_volume = 45 →
  initial_milk_ratio = 4 →
  initial_water_ratio = 1 →
  added_water = 18 →
  let initial_milk := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let initial_water := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let new_water := initial_water + added_water
  let new_milk_ratio := initial_milk / new_water
  let new_water_ratio := new_water / new_water
  (new_milk_ratio : ℚ) / (new_water_ratio : ℚ) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_milk_water_ratio_l3655_365502


namespace NUMINAMATH_CALUDE_triangle_sum_theorem_l3655_365559

def triangle_numbers : Finset ℕ := {1, 3, 5, 7, 9, 11}

theorem triangle_sum_theorem (vertex_sum midpoint_sum : ℕ → ℕ → ℕ → ℕ) 
  (h1 : vertex_sum 19 19 19 = (vertex_sum 1 3 5 + vertex_sum 7 9 11))
  (h2 : ∀ a b c, a ∈ triangle_numbers → b ∈ triangle_numbers → c ∈ triangle_numbers → 
    a ≠ b ∧ b ≠ c ∧ a ≠ c → vertex_sum a b c + midpoint_sum a b c = 19)
  (h3 : ∀ a b c d e f, a ∈ triangle_numbers → b ∈ triangle_numbers → c ∈ triangle_numbers →
    d ∈ triangle_numbers → e ∈ triangle_numbers → f ∈ triangle_numbers →
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ a → 
    vertex_sum a c e + midpoint_sum b d f = 19) :
  ∃ a b c, a ∈ triangle_numbers ∧ b ∈ triangle_numbers ∧ c ∈ triangle_numbers ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ vertex_sum a b c = 21 :=
sorry

end NUMINAMATH_CALUDE_triangle_sum_theorem_l3655_365559


namespace NUMINAMATH_CALUDE_geoffrey_initial_wallet_l3655_365582

/-- The amount of money Geoffrey had initially in his wallet --/
def initial_wallet_amount : ℕ := 50

/-- The amount Geoffrey received from his grandmother --/
def grandmother_gift : ℕ := 20

/-- The amount Geoffrey received from his aunt --/
def aunt_gift : ℕ := 25

/-- The amount Geoffrey received from his uncle --/
def uncle_gift : ℕ := 30

/-- The cost of each game --/
def game_cost : ℕ := 35

/-- The number of games Geoffrey bought --/
def num_games : ℕ := 3

/-- The amount left after the purchase --/
def amount_left : ℕ := 20

theorem geoffrey_initial_wallet :
  initial_wallet_amount = 
    (amount_left + num_games * game_cost) - (grandmother_gift + aunt_gift + uncle_gift) :=
by sorry

end NUMINAMATH_CALUDE_geoffrey_initial_wallet_l3655_365582


namespace NUMINAMATH_CALUDE_min_rows_for_hockey_arena_l3655_365540

/-- Represents a hockey arena with rows of seats -/
structure Arena where
  seats_per_row : ℕ
  total_students : ℕ
  max_students_per_school : ℕ

/-- Calculates the minimum number of rows required in the arena -/
def min_rows_required (arena : Arena) : ℕ :=
  sorry

/-- The theorem stating the minimum number of rows required for the given conditions -/
theorem min_rows_for_hockey_arena :
  let arena : Arena := {
    seats_per_row := 168,
    total_students := 2016,
    max_students_per_school := 45
  }
  min_rows_required arena = 16 := by sorry

end NUMINAMATH_CALUDE_min_rows_for_hockey_arena_l3655_365540


namespace NUMINAMATH_CALUDE_point_coordinates_l3655_365549

theorem point_coordinates (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : 1 < b) (hb' : b < 2) :
  (0 < a/2 ∧ a/2 < 1/2 ∧ 2 < b+1 ∧ b+1 < 3) ∧
  (-1 < a-1 ∧ a-1 < 0 ∧ 0 < b/2 ∧ b/2 < 1) ∧
  (-1 < -a ∧ -a < 0 ∧ -2 < -b ∧ -b < -1) ∧
  (0 < 1-a ∧ 1-a < 1 ∧ 0 < b-1 ∧ b-1 < 1) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l3655_365549


namespace NUMINAMATH_CALUDE_distance_difference_l3655_365535

def time : ℝ := 6
def carlos_distance : ℝ := 108
def daniel_distance : ℝ := 90

theorem distance_difference : carlos_distance - daniel_distance = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l3655_365535


namespace NUMINAMATH_CALUDE_product_units_digit_base_7_l3655_365545

theorem product_units_digit_base_7 : 
  (359 * 72) % 7 = 4 := by sorry

end NUMINAMATH_CALUDE_product_units_digit_base_7_l3655_365545


namespace NUMINAMATH_CALUDE_coin_packing_inequality_l3655_365574

/-- Given a circular table of radius R and n non-overlapping circular coins of radius r
    placed on it such that no more coins can be added, prove that R / r ≤ 2√n + 1 --/
theorem coin_packing_inequality (R r : ℝ) (n : ℕ) 
    (h_positive_R : R > 0) 
    (h_positive_r : r > 0) 
    (h_positive_n : n > 0) 
    (h_non_overlapping : ∀ (i j : ℕ), i < n → j < n → i ≠ j → 
      ∃ (x_i y_i x_j y_j : ℝ), (x_i - x_j)^2 + (y_i - y_j)^2 ≥ 4*r^2)
    (h_within_table : ∀ (i : ℕ), i < n → 
      ∃ (x_i y_i : ℝ), x_i^2 + y_i^2 ≤ (R - r)^2)
    (h_no_more_coins : ∀ (x y : ℝ), x^2 + y^2 ≤ (R - r)^2 → 
      ∃ (i : ℕ), i < n ∧ ∃ (x_i y_i : ℝ), (x - x_i)^2 + (y - y_i)^2 < 4*r^2) :
  R / r ≤ 2 * Real.sqrt n + 1 := by
sorry

end NUMINAMATH_CALUDE_coin_packing_inequality_l3655_365574


namespace NUMINAMATH_CALUDE_inequality_part_1_inequality_part_2_l3655_365584

-- Part I
theorem inequality_part_1 : 
  ∀ x : ℝ, (|x - 3| + |x + 5| ≥ 2 * |x + 5|) ↔ (x ≤ -1) := by sorry

-- Part II
theorem inequality_part_2 : 
  ∀ a : ℝ, (∀ x : ℝ, |x - a| + |x + 5| ≥ 6) ↔ (a ≥ 1 ∨ a ≤ -11) := by sorry

end NUMINAMATH_CALUDE_inequality_part_1_inequality_part_2_l3655_365584


namespace NUMINAMATH_CALUDE_min_sum_fraction_l3655_365595

def Digits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def IsValidSelection (a b c d : Nat) : Prop :=
  a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def SumFraction (a b c d : Nat) : Rat :=
  a / b + c / d

theorem min_sum_fraction :
  ∃ (a b c d : Nat), IsValidSelection a b c d ∧
    (∀ (w x y z : Nat), IsValidSelection w x y z →
      SumFraction a b c d ≤ SumFraction w x y z) ∧
    SumFraction a b c d = 17 / 15 :=
  sorry

end NUMINAMATH_CALUDE_min_sum_fraction_l3655_365595


namespace NUMINAMATH_CALUDE_circle_tangent_range_l3655_365546

/-- Given a circle with equation x^2 + y^2 + ax + 2y + a^2 = 0 and a fixed point A(1, 2),
    this theorem states the range of values for a that allows two tangents from point A to the circle. -/
theorem circle_tangent_range (a : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + a*x + 2*y + a^2 = 0) →
  (∃ (t : ℝ), (1 + t*(-a/2 - 1))^2 + (2 + t*(-1 - (-1)))^2 = ((4 - 3*a^2)/4)) →
  a ∈ Set.Ioo (-2*Real.sqrt 3/3) (2*Real.sqrt 3/3) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_range_l3655_365546


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_negation_of_specific_proposition_l3655_365533

theorem negation_of_universal_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_specific_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 3 ≥ 0) ↔ (∃ x : ℝ, x^2 + 2*x + 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_negation_of_specific_proposition_l3655_365533


namespace NUMINAMATH_CALUDE_special_line_equation_l3655_365553

/-- A line passing through point (3, -1) with equal absolute values of intercepts on both axes -/
structure SpecialLine where
  -- The equation of the line in the form ax + by + c = 0
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through (3, -1)
  passes_through : a * 3 + b * (-1) + c = 0
  -- The line has equal absolute values of intercepts on both axes
  equal_intercepts : |a / b| = |b / a| ∨ (a = 0 ∧ b ≠ 0) ∨ (b = 0 ∧ a ≠ 0)

/-- The possible equations for the special line -/
def possible_equations (l : SpecialLine) : Prop :=
  (l.a = 1 ∧ l.b = 1 ∧ l.c = -2) ∨
  (l.a = 1 ∧ l.b = -1 ∧ l.c = -4) ∨
  (l.a = 3 ∧ l.b = 1 ∧ l.c = 0)

/-- Theorem stating that any SpecialLine must have one of the possible equations -/
theorem special_line_equation (l : SpecialLine) : possible_equations l := by
  sorry

end NUMINAMATH_CALUDE_special_line_equation_l3655_365553


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3655_365514

/-- A rhombus with a diagonal of length 6 and side length satisfying x^2 - 7x + 12 = 0 has a perimeter of 16 -/
theorem rhombus_perimeter (a b c d : ℝ) (h1 : a = b ∧ b = c ∧ c = d) 
  (h2 : ∃ (diag : ℝ), diag = 6) 
  (h3 : a^2 - 7*a + 12 = 0) : 
  a + b + c + d = 16 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3655_365514


namespace NUMINAMATH_CALUDE_cranberries_count_l3655_365581

/-- The number of cranberries picked by Iris's sister -/
def cranberries : ℕ := 20

/-- The number of blueberries picked by Iris -/
def blueberries : ℕ := 30

/-- The number of raspberries picked by Iris's brother -/
def raspberries : ℕ := 10

/-- The total number of berries picked -/
def total_berries : ℕ := blueberries + cranberries + raspberries

/-- The number of fresh berries -/
def fresh_berries : ℕ := (2 * total_berries) / 3

/-- The number of berries that can be sold -/
def sellable_berries : ℕ := fresh_berries / 2

theorem cranberries_count : sellable_berries = 20 := by sorry

end NUMINAMATH_CALUDE_cranberries_count_l3655_365581


namespace NUMINAMATH_CALUDE_hotel_to_ticket_ratio_l3655_365576

/-- Represents the trip expenses and calculates the ratio of hotel cost to ticket cost. -/
def tripExpenses (initialAmount ticketCost amountLeft : ℚ) : ℚ × ℚ := by
  -- Define total spent
  let totalSpent := initialAmount - amountLeft
  -- Define hotel cost
  let hotelCost := totalSpent - ticketCost
  -- Calculate the ratio
  let ratio := hotelCost / ticketCost
  -- Return the simplified ratio
  exact (1, 2)

/-- Theorem stating that the ratio of hotel cost to ticket cost is 1:2 for the given values. -/
theorem hotel_to_ticket_ratio :
  tripExpenses 760 300 310 = (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_hotel_to_ticket_ratio_l3655_365576


namespace NUMINAMATH_CALUDE_distinct_values_count_l3655_365573

def expression := 3^3^3^3

def parenthesization1 := 3^(3^(3^3))
def parenthesization2 := 3^((3^3)^3)
def parenthesization3 := ((3^3)^3)^3
def parenthesization4 := (3^(3^3))^3
def parenthesization5 := (3^3)^(3^3)

def distinct_values : Finset ℕ := {parenthesization1, parenthesization2, parenthesization3, parenthesization4, parenthesization5}

theorem distinct_values_count :
  Finset.card distinct_values = 3 :=
sorry

end NUMINAMATH_CALUDE_distinct_values_count_l3655_365573


namespace NUMINAMATH_CALUDE_right_triangle_in_square_l3655_365588

theorem right_triangle_in_square (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ (s : ℝ), s > 0 ∧ s^2 = 16 ∧ a^2 + b^2 = s^2) →
  a * b = 16 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_in_square_l3655_365588


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3655_365591

theorem pure_imaginary_complex_number (x : ℝ) :
  (x^2 - 1 : ℂ) + (x + 1 : ℂ) * Complex.I = (0 : ℂ) + (y : ℂ) * Complex.I →
  x = 1 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3655_365591


namespace NUMINAMATH_CALUDE_regression_line_y_change_l3655_365548

/-- Represents a linear regression equation of the form ŷ = a + bx -/
structure RegressionLine where
  a : ℝ
  b : ℝ

/-- The change in y when x increases by 1 unit -/
def yChange (line : RegressionLine) : ℝ := line.b

theorem regression_line_y_change 
  (line : RegressionLine) 
  (h : line = { a := 2, b := -1.5 }) : 
  yChange line = -1.5 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_y_change_l3655_365548


namespace NUMINAMATH_CALUDE_second_train_speed_l3655_365556

/-- Proves that the speed of the second train is 80 kmph given the conditions of the problem -/
theorem second_train_speed (first_train_speed : ℝ) (time_difference : ℝ) (meeting_distance : ℝ) :
  first_train_speed = 40 →
  time_difference = 1 →
  meeting_distance = 80 →
  (meeting_distance - first_train_speed * time_difference) / time_difference = 80 :=
by
  sorry

#check second_train_speed

end NUMINAMATH_CALUDE_second_train_speed_l3655_365556


namespace NUMINAMATH_CALUDE_intersection_length_circle_line_l3655_365527

/-- The intersection length of a circle and a line --/
theorem intersection_length_circle_line : 
  ∃ (A B : ℝ × ℝ),
    (A.1^2 + (A.2 - 1)^2 = 1) ∧ 
    (B.1^2 + (B.2 - 1)^2 = 1) ∧
    (A.1 - A.2 + 2 = 0) ∧ 
    (B.1 - B.2 + 2 = 0) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_length_circle_line_l3655_365527


namespace NUMINAMATH_CALUDE_flight_duration_sum_l3655_365505

def flight_duration (departure_hour : Nat) (departure_minute : Nat)
                    (arrival_hour : Nat) (arrival_minute : Nat) : Nat :=
  (arrival_hour * 60 + arrival_minute) - (departure_hour * 60 + departure_minute)

theorem flight_duration_sum (h m : Nat) :
  flight_duration 15 42 18 57 = h * 60 + m →
  0 < m →
  m < 60 →
  h + m = 18 := by
  sorry

end NUMINAMATH_CALUDE_flight_duration_sum_l3655_365505


namespace NUMINAMATH_CALUDE_average_salary_l3655_365570

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def num_people : ℕ := 5

theorem average_salary :
  (salary_A + salary_B + salary_C + salary_D + salary_E) / num_people = 8000 :=
by
  sorry

end NUMINAMATH_CALUDE_average_salary_l3655_365570


namespace NUMINAMATH_CALUDE_complex_fraction_equation_solution_l3655_365593

theorem complex_fraction_equation_solution :
  ∃ x : ℚ, 3 + 1 / (2 + 1 / (3 + 3 / (4 + x))) = 225/68 ∧ x = -50/19 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equation_solution_l3655_365593


namespace NUMINAMATH_CALUDE_circle_properties_l3655_365580

def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*y - 16 = -y^2 + 26*x + 36

theorem circle_properties :
  ∃ (p q s : ℝ),
    (∀ x y, circle_equation x y ↔ (x - p)^2 + (y - q)^2 = s^2) ∧
    p = 13 ∧
    q = 2 ∧
    s = 15 ∧
    p + q + s = 30 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3655_365580


namespace NUMINAMATH_CALUDE_expression_equals_forty_times_ten_to_power_l3655_365507

theorem expression_equals_forty_times_ten_to_power : 
  (3^1506 + 7^1507)^2 - (3^1506 - 7^1507)^2 = 40 * 10^1506 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_forty_times_ten_to_power_l3655_365507


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3655_365590

theorem polynomial_factorization (a b c : ℝ) : 
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = 
  (a - b) * (b - c) * (c - a) * ((a + b + c)^3 - 3*a*b*c) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3655_365590


namespace NUMINAMATH_CALUDE_ellipse_parallelogram_area_l3655_365526

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop := x^2 + 2*y^2 = 1

-- Define the slope product condition
def slope_product (x1 y1 x2 y2 : ℝ) : Prop := (y1/x1) * (y2/x2) = -1/2

-- Define the area of the parallelogram
def parallelogram_area (x1 y1 x2 y2 : ℝ) : ℝ := 2 * |x1*y2 - x2*y1|

-- Theorem statement
theorem ellipse_parallelogram_area 
  (x1 y1 x2 y2 : ℝ) 
  (h1 : on_ellipse x1 y1) 
  (h2 : on_ellipse x2 y2) 
  (h3 : slope_product x1 y1 x2 y2) : 
  parallelogram_area x1 y1 x2 y2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parallelogram_area_l3655_365526


namespace NUMINAMATH_CALUDE_andrena_has_three_more_than_debelyn_l3655_365510

/-- Represents the number of dolls each person has -/
structure DollCounts where
  debelyn : ℕ
  christel : ℕ
  andrena : ℕ

/-- The initial state of doll ownership -/
def initial : DollCounts :=
  { debelyn := 20
  , christel := 24
  , andrena := 0 }

/-- The state after doll transfers -/
def final : DollCounts :=
  { debelyn := initial.debelyn - 2
  , christel := initial.christel - 5
  , andrena := initial.andrena + 2 + 5 }

theorem andrena_has_three_more_than_debelyn :
  final.andrena = final.christel + 2 →
  final.andrena - final.debelyn = 3 := by
  sorry

end NUMINAMATH_CALUDE_andrena_has_three_more_than_debelyn_l3655_365510


namespace NUMINAMATH_CALUDE_circle_equation_l3655_365506

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) -/
theorem circle_equation (x y : ℝ) : 
  (∀ D E F : ℝ, (x^2 + y^2 + D*x + E*y + F = 0) → 
    (0^2 + 0^2 + D*0 + E*0 + F = 0 ∧ 
     4^2 + 0^2 + D*4 + E*0 + F = 0 ∧ 
     (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0)) →
  (x^2 + y^2 - 4*x - 6*y = 0 ↔ 
    (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3655_365506


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l3655_365501

/-- Given a line expressed in vector form, prove its slope-intercept form --/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y - (-4)) = 0 →
  y = 2 * x - 10 := by
sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l3655_365501


namespace NUMINAMATH_CALUDE_son_age_problem_l3655_365598

theorem son_age_problem (father_age son_age : ℕ) : 
  father_age = son_age + 26 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 24 := by
sorry

end NUMINAMATH_CALUDE_son_age_problem_l3655_365598


namespace NUMINAMATH_CALUDE_coeff_bound_squared_poly_l3655_365571

/-- A polynomial with non-negative coefficients where no coefficient exceeds p(0) -/
structure NonNegPolynomial (n : ℕ) where
  p : Polynomial ℝ
  degree_eq : p.degree = n
  non_neg_coeff : ∀ i, 0 ≤ p.coeff i
  coeff_bound : ∀ i, p.coeff i ≤ p.coeff 0

/-- The coefficient of x^(n+1) in p(x)^2 is at most p(1)^2 / 2 -/
theorem coeff_bound_squared_poly {n : ℕ} (p : NonNegPolynomial n) :
  (p.p ^ 2).coeff (n + 1) ≤ (p.p.eval 1) ^ 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_coeff_bound_squared_poly_l3655_365571


namespace NUMINAMATH_CALUDE_steve_berry_picking_earnings_l3655_365565

/-- The amount of money earned per pound of lingonberries -/
def price_per_pound : ℕ := 2

/-- The amount of lingonberries picked on Monday -/
def monday_picking : ℕ := 8

/-- The amount of lingonberries picked on Tuesday -/
def tuesday_picking : ℕ := 3 * monday_picking

/-- The amount of lingonberries picked on Wednesday -/
def wednesday_picking : ℕ := 0

/-- The amount of lingonberries picked on Thursday -/
def thursday_picking : ℕ := 18

/-- The total money Steve wanted to make -/
def total_money : ℕ := 100

/-- Theorem stating that the total money Steve wanted to make is correct -/
theorem steve_berry_picking_earnings :
  (monday_picking + tuesday_picking + wednesday_picking + thursday_picking) * price_per_pound = total_money := by
  sorry

end NUMINAMATH_CALUDE_steve_berry_picking_earnings_l3655_365565


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3655_365521

theorem cube_volume_problem (a : ℝ) : 
  (a - 2) * a * (a + 2) = a^3 - 8 → a^3 = 8 := by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3655_365521


namespace NUMINAMATH_CALUDE_sum_remainder_mod_9_l3655_365566

theorem sum_remainder_mod_9 : (98134 + 98135 + 98136 + 98137 + 98138 + 98139) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_9_l3655_365566


namespace NUMINAMATH_CALUDE_complement_union_complement_equals_intersection_l3655_365530

theorem complement_union_complement_equals_intersection (P Q : Set α) :
  (Pᶜᶜ ∪ Qᶜ)ᶜ = P ∩ Q := by
  sorry

end NUMINAMATH_CALUDE_complement_union_complement_equals_intersection_l3655_365530


namespace NUMINAMATH_CALUDE_c_share_is_36_l3655_365562

/-- Represents the rental information for a person --/
structure RentalInfo where
  oxen : ℕ
  months : ℕ

/-- Calculates the share of rent for a given person --/
def calculateShare (totalRent : ℚ) (totalOxMonths : ℕ) (info : RentalInfo) : ℚ :=
  totalRent * (info.oxen * info.months : ℚ) / totalOxMonths

theorem c_share_is_36 
  (totalRent : ℚ)
  (a b c : RentalInfo)
  (h_total_rent : totalRent = 140)
  (h_a : a = ⟨10, 7⟩)
  (h_b : b = ⟨12, 5⟩)
  (h_c : c = ⟨15, 3⟩) :
  calculateShare totalRent (a.oxen * a.months + b.oxen * b.months + c.oxen * c.months) c = 36 := by
  sorry

#check c_share_is_36

end NUMINAMATH_CALUDE_c_share_is_36_l3655_365562


namespace NUMINAMATH_CALUDE_solve_cake_problem_l3655_365503

def cake_problem (cost_per_cake : ℕ) (john_payment : ℕ) : Prop :=
  ∃ (num_cakes : ℕ),
    cost_per_cake = 12 ∧
    john_payment = 18 ∧
    num_cakes * cost_per_cake = 2 * john_payment ∧
    num_cakes = 3

theorem solve_cake_problem :
  ∀ (cost_per_cake : ℕ) (john_payment : ℕ),
    cake_problem cost_per_cake john_payment :=
by
  sorry

end NUMINAMATH_CALUDE_solve_cake_problem_l3655_365503


namespace NUMINAMATH_CALUDE_divisibility_by_48_l3655_365596

theorem divisibility_by_48 (a b c : ℕ) 
  (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c)
  (ga : a > 3) (gb : b > 3) (gc : c > 3) : 
  48 ∣ ((a - b) * (b - c) * (c - a)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_48_l3655_365596


namespace NUMINAMATH_CALUDE_disjunction_is_true_l3655_365563

def p : Prop := 1 ∈ {x : ℝ | (x + 2) * (x - 3) < 0}

def q : Prop := (∅ : Set ℕ) = {0}

theorem disjunction_is_true : p ∨ q := by sorry

end NUMINAMATH_CALUDE_disjunction_is_true_l3655_365563


namespace NUMINAMATH_CALUDE_inscribed_cube_properties_l3655_365558

/-- Given a cube with surface area 54 square meters, containing an inscribed sphere 
    which in turn contains an inscribed smaller cube, prove the surface area and volume 
    of the inner cube. -/
theorem inscribed_cube_properties (outer_cube : Real) (sphere : Real) (inner_cube : Real) :
  (6 * outer_cube ^ 2 = 54) →
  (sphere = outer_cube / 2) →
  (inner_cube * Real.sqrt 3 = outer_cube) →
  (6 * inner_cube ^ 2 = 18 ∧ inner_cube ^ 3 = 3 * Real.sqrt 3) := by
  sorry

#check inscribed_cube_properties

end NUMINAMATH_CALUDE_inscribed_cube_properties_l3655_365558


namespace NUMINAMATH_CALUDE_shaded_area_octagon_with_sectors_l3655_365529

/-- The area of the shaded region in a regular octagon with circular sectors --/
theorem shaded_area_octagon_with_sectors (side_length : Real) (sector_radius : Real) : 
  side_length = 5 → sector_radius = 3 → 
  ∃ (shaded_area : Real), shaded_area = 100 - 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_octagon_with_sectors_l3655_365529


namespace NUMINAMATH_CALUDE_cayley_hamilton_for_B_l3655_365515

def B : Matrix (Fin 3) (Fin 3) ℝ := !![1, 2, 3; 2, 1, 2; 3, 2, 1]

theorem cayley_hamilton_for_B :
  ∃ (s t u : ℝ), 
    B^3 + s • B^2 + t • B + u • (1 : Matrix (Fin 3) (Fin 3) ℝ) = 0 ∧ 
    s = -7 ∧ t = 2 ∧ u = -9 := by
  sorry

end NUMINAMATH_CALUDE_cayley_hamilton_for_B_l3655_365515


namespace NUMINAMATH_CALUDE_lucas_payment_l3655_365519

/-- Calculates the payment for window cleaning based on given conditions --/
def calculate_payment (stories : ℕ) (windows_per_floor : ℕ) (payment_per_window : ℕ) 
  (deduction_per_2_days : ℕ) (days_taken : ℕ) : ℕ :=
  let total_windows := stories * windows_per_floor
  let base_payment := total_windows * payment_per_window
  let time_deductions := (days_taken / 2) * deduction_per_2_days
  base_payment - time_deductions

/-- Theorem stating that Lucas' father will pay him $33 --/
theorem lucas_payment :
  calculate_payment 4 5 2 1 14 = 33 := by
  sorry

end NUMINAMATH_CALUDE_lucas_payment_l3655_365519


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l3655_365589

variable (n : ℕ) (hn : n ≥ 3) (hodd : Odd n)
variable (A B C : Polynomial ℝ)

theorem polynomial_equation_solution :
  A^n + B^n + C^n = 0 →
  ∃ (a b c : ℝ) (D : Polynomial ℝ),
    a^n + b^n + c^n = 0 ∧
    A = a • D ∧
    B = b • D ∧
    C = c • D :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l3655_365589


namespace NUMINAMATH_CALUDE_ihsan_children_l3655_365557

/-- The number of children each person has (except great-great-grandchildren) -/
def n : ℕ := 7

/-- The total number of people in the family, including Ihsan -/
def total_people : ℕ := 2801

/-- Theorem stating that n satisfies the conditions of the problem -/
theorem ihsan_children :
  n + n^2 + n^3 + n^4 + 1 = total_people :=
by sorry

end NUMINAMATH_CALUDE_ihsan_children_l3655_365557


namespace NUMINAMATH_CALUDE_one_painted_face_probability_l3655_365550

/-- Represents a cube with painted faces -/
structure PaintedCube where
  side_length : ℕ
  painted_faces : ℕ
  painted_faces_adjacent : Bool

/-- Calculates the number of unit cubes with exactly one painted face -/
def one_painted_face_count (c : PaintedCube) : ℕ :=
  if c.painted_faces_adjacent then
    2 * (c.side_length^2 - c.side_length) - (c.side_length - 1)
  else
    c.painted_faces * (c.side_length^2 - c.side_length)

/-- Theorem stating the probability of selecting a unit cube with one painted face -/
theorem one_painted_face_probability (c : PaintedCube) 
  (h1 : c.side_length = 5)
  (h2 : c.painted_faces = 2)
  (h3 : c.painted_faces_adjacent = true) :
  (one_painted_face_count c : ℚ) / (c.side_length^3 : ℚ) = 41 / 125 := by
  sorry

end NUMINAMATH_CALUDE_one_painted_face_probability_l3655_365550


namespace NUMINAMATH_CALUDE_vectors_orthogonal_l3655_365518

-- Define the vectors
def v1 : Fin 2 → ℝ := ![3, 7]
def v2 (x : ℝ) : Fin 2 → ℝ := ![x, -4]

-- Define orthogonality condition
def isOrthogonal (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

-- State the theorem
theorem vectors_orthogonal :
  isOrthogonal v1 (v2 (28/3)) := by sorry

end NUMINAMATH_CALUDE_vectors_orthogonal_l3655_365518


namespace NUMINAMATH_CALUDE_second_negative_integer_l3655_365592

theorem second_negative_integer (n : ℤ) : 
  n < 0 → -11 * n + 5 = 93 → n = -8 :=
by sorry

end NUMINAMATH_CALUDE_second_negative_integer_l3655_365592


namespace NUMINAMATH_CALUDE_coefficient_of_y_l3655_365555

theorem coefficient_of_y (x y : ℝ) (a : ℝ) : 
  x / (2 * y) = 3 / 2 → 
  (7 * x + a * y) / (x - 2 * y) = 26 → 
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_y_l3655_365555


namespace NUMINAMATH_CALUDE_lottery_probability_l3655_365552

theorem lottery_probability (total_tickets : Nat) (winning_tickets : Nat) (people : Nat) :
  total_tickets = 10 →
  winning_tickets = 3 →
  people = 5 →
  (1 : ℚ) - (Nat.choose (total_tickets - winning_tickets) people : ℚ) / (Nat.choose total_tickets people : ℚ) = 11 / 12 :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l3655_365552


namespace NUMINAMATH_CALUDE_fifteen_valid_pairs_l3655_365594

/-- A function that constructs the number 7ABABA from single digits A and B -/
def constructNumber (A B : Nat) : Nat :=
  700000 + 10000 * A + 1000 * B + 100 * A + 10 * B + A

/-- Predicate to check if a number is a single digit -/
def isSingleDigit (n : Nat) : Prop := n < 10

/-- The main theorem stating that there are exactly 15 valid pairs (A, B) -/
theorem fifteen_valid_pairs :
  ∃! (validPairs : Finset (Nat × Nat)),
    validPairs.card = 15 ∧
    ∀ (A B : Nat),
      (A, B) ∈ validPairs ↔
        isSingleDigit A ∧
        isSingleDigit B ∧
        (constructNumber A B % 6 = 0) :=
  sorry

end NUMINAMATH_CALUDE_fifteen_valid_pairs_l3655_365594


namespace NUMINAMATH_CALUDE_find_y_l3655_365513

theorem find_y (x : ℝ) (y : ℝ) : 
  ((100 + 200 + 300 + x) / 4 = 250) →
  ((300 + 150 + 100 + x + y) / 5 = 200) →
  y = 50 := by
sorry

end NUMINAMATH_CALUDE_find_y_l3655_365513


namespace NUMINAMATH_CALUDE_simplify_expression_l3655_365572

theorem simplify_expression :
  (-2)^2006 + (-1)^3007 + 1^3010 - (-2)^2007 = -2^2006 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3655_365572


namespace NUMINAMATH_CALUDE_johns_total_earnings_l3655_365522

/-- Calculates the total earnings given the initial bonus, growth rate, and new salary -/
def total_earnings (initial_bonus : ℝ) (growth_rate : ℝ) (new_salary : ℝ) : ℝ :=
  let new_bonus := initial_bonus * (1 + growth_rate)
  new_salary + new_bonus

/-- Theorem: John's total earnings this year are $210,500 -/
theorem johns_total_earnings :
  let initial_bonus : ℝ := 10000
  let growth_rate : ℝ := 0.05
  let new_salary : ℝ := 200000
  total_earnings initial_bonus growth_rate new_salary = 210500 := by
  sorry

#eval total_earnings 10000 0.05 200000

end NUMINAMATH_CALUDE_johns_total_earnings_l3655_365522


namespace NUMINAMATH_CALUDE_max_set_size_l3655_365597

def is_valid_set (s : Finset Nat) : Prop :=
  s.card > 0 ∧ 10 ∉ s ∧ s.sum (λ x => x^2) = 2500

theorem max_set_size :
  (∃ (s : Finset Nat), is_valid_set s ∧ s.card = 17) ∧
  (∀ (s : Finset Nat), is_valid_set s → s.card ≤ 17) :=
sorry

end NUMINAMATH_CALUDE_max_set_size_l3655_365597
