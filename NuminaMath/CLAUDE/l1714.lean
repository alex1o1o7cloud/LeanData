import Mathlib

namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l1714_171441

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  group_size : ℕ
  first_group_number : ℕ

/-- The theorem for systematic sampling -/
theorem systematic_sampling_theorem (s : SystematicSampling)
  (h1 : s.total_students = 300)
  (h2 : s.sample_size = 20)
  (h3 : s.group_size = s.total_students / s.sample_size)
  (h4 : s.first_group_number < s.group_size)
  (h5 : 231 = s.first_group_number + 15 * s.group_size) :
  s.first_group_number = 6 := by
  sorry

#check systematic_sampling_theorem

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l1714_171441


namespace NUMINAMATH_CALUDE_triangle_midpoint_x_sum_l1714_171433

theorem triangle_midpoint_x_sum (a b c : ℝ) (S : ℝ) : 
  a + b + c = S → 
  (a + b) / 2 + (b + c) / 2 + (c + a) / 2 = S :=
by sorry

end NUMINAMATH_CALUDE_triangle_midpoint_x_sum_l1714_171433


namespace NUMINAMATH_CALUDE_pipe_fill_time_l1714_171488

/-- Time for pipe to fill tank without leak -/
def T : ℝ := 5

/-- Time to fill tank with leak -/
def fill_time_with_leak : ℝ := 10

/-- Time for leak to empty full tank -/
def leak_empty_time : ℝ := 10

/-- Theorem: The pipe fills the tank in 5 hours without the leak -/
theorem pipe_fill_time :
  (1 / T - 1 / leak_empty_time = 1 / fill_time_with_leak) →
  T = 5 := by
  sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l1714_171488


namespace NUMINAMATH_CALUDE_log_sum_equals_four_l1714_171458

theorem log_sum_equals_four : Real.log 64 / Real.log 8 + Real.log 81 / Real.log 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_four_l1714_171458


namespace NUMINAMATH_CALUDE_intersection_line_slope_l1714_171466

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 11 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 14*x + 12*y + 60 = 0

-- Define the line passing through the intersection points
def intersectionLine (x y : ℝ) : Prop := 10*x - 10*y - 71 = 0

-- Theorem statement
theorem intersection_line_slope :
  ∀ (x1 y1 x2 y2 : ℝ),
  circle1 x1 y1 ∧ circle1 x2 y2 ∧
  circle2 x1 y1 ∧ circle2 x2 y2 ∧
  intersectionLine x1 y1 ∧ intersectionLine x2 y2 ∧
  x1 ≠ x2 →
  (y2 - y1) / (x2 - x1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l1714_171466


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1714_171493

/-- Given a and b are real numbers satisfying the equation a - bi = (1 + i)i^3,
    prove that a = 1 and b = -1. -/
theorem complex_equation_solution (a b : ℝ) :
  (a : ℂ) - b * Complex.I = (1 + Complex.I) * Complex.I^3 →
  a = 1 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1714_171493


namespace NUMINAMATH_CALUDE_andreas_living_room_area_l1714_171432

/-- The area of Andrea's living room floor, given a carpet covering 20% of it --/
theorem andreas_living_room_area 
  (carpet_length : ℝ) 
  (carpet_width : ℝ) 
  (carpet_coverage_percent : ℝ) 
  (h1 : carpet_length = 4)
  (h2 : carpet_width = 9)
  (h3 : carpet_coverage_percent = 20) : 
  carpet_length * carpet_width / (carpet_coverage_percent / 100) = 180 := by
sorry

end NUMINAMATH_CALUDE_andreas_living_room_area_l1714_171432


namespace NUMINAMATH_CALUDE_angle_B_range_l1714_171410

theorem angle_B_range (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = 180) (h5 : A ≤ B) (h6 : B ≤ C) (h7 : 2 * B = 5 * A) :
  0 < B ∧ B ≤ 75 := by
sorry

end NUMINAMATH_CALUDE_angle_B_range_l1714_171410


namespace NUMINAMATH_CALUDE_jerry_can_carry_l1714_171453

/-- Given the following conditions:
  * There are 28 cans to be recycled
  * The total time taken is 350 seconds
  * It takes 30 seconds to drain the cans
  * It takes 10 seconds to walk each way (to and from the sink/recycling bin)
  Prove that Jerry can carry 4 cans at once. -/
theorem jerry_can_carry (total_cans : ℕ) (total_time : ℕ) (drain_time : ℕ) (walk_time : ℕ) :
  total_cans = 28 →
  total_time = 350 →
  drain_time = 30 →
  walk_time = 10 →
  (total_time / (drain_time + 2 * walk_time) : ℚ) * (total_cans / (total_time / (drain_time + 2 * walk_time)) : ℚ) = 4 :=
by sorry

end NUMINAMATH_CALUDE_jerry_can_carry_l1714_171453


namespace NUMINAMATH_CALUDE_square_of_sum_l1714_171445

theorem square_of_sum (x y k m : ℝ) (h1 : x * y = k) (h2 : x^2 + y^2 = m) :
  (x + y)^2 = m + 2*k := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l1714_171445


namespace NUMINAMATH_CALUDE_orange_pyramid_count_l1714_171449

/-- Calculates the number of oranges in a pyramid layer given its width and length -/
def layer_oranges (width : ℕ) (length : ℕ) : ℕ := width * length

/-- Calculates the total number of oranges in a pyramid stack -/
def total_oranges (base_width : ℕ) (base_length : ℕ) : ℕ :=
  let base := layer_oranges base_width base_length
  let layer2 := layer_oranges (base_width - 1) (base_length - 1)
  let layer3 := layer_oranges (base_width - 2) (base_length - 2)
  let layer4 := layer_oranges (base_width - 3) (base_length - 3)
  let layer5 := layer_oranges (base_width - 4) (base_length - 4)
  let layer6 := layer_oranges (base_width - 5) (base_length - 5)
  let layer7 := layer_oranges (base_width - 6) (base_length - 6)
  base + layer2 + layer3 + layer4 + layer5 + layer6 + layer7 + 1

theorem orange_pyramid_count :
  total_oranges 7 10 = 225 := by
  sorry

end NUMINAMATH_CALUDE_orange_pyramid_count_l1714_171449


namespace NUMINAMATH_CALUDE_inequality_problem_l1714_171412

theorem inequality_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 + b^2 + 2 ≥ 2*a + 2*b) ∧ 
  (Real.log (a*b + 1) ≥ 0) ∧ 
  (Real.sqrt (a + b) ≥ 2) ∧ 
  ¬(∀ (x y : ℝ), x > 0 → y > 0 → x^3 + y^3 ≥ 2*x*y^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l1714_171412


namespace NUMINAMATH_CALUDE_prob_odd_add_only_prob_odd_with_multiply_l1714_171428

-- Define the calculator operations
inductive Operation
| Add
| Multiply

-- Define the calculator state
structure CalculatorState where
  display : ℕ
  lastOp : Option Operation

-- Define the probability of getting an odd number
def probOdd (ops : List Operation) : ℚ :=
  sorry

-- Theorem for part (a)
theorem prob_odd_add_only :
  ∀ (n : ℕ), probOdd (List.replicate n Operation.Add) = 1/2 :=
sorry

-- Theorem for part (b)
theorem prob_odd_with_multiply (n : ℕ) :
  probOdd (List.cons Operation.Multiply (List.replicate n Operation.Add)) < 1/2 :=
sorry

end NUMINAMATH_CALUDE_prob_odd_add_only_prob_odd_with_multiply_l1714_171428


namespace NUMINAMATH_CALUDE_cost_per_chicken_problem_l1714_171408

/-- Given a total number of birds, a fraction of ducks, and the total cost to feed chickens,
    calculate the cost per chicken. -/
def cost_per_chicken (total_birds : ℕ) (duck_fraction : ℚ) (total_cost : ℚ) : ℚ :=
  let chicken_fraction : ℚ := 1 - duck_fraction
  let num_chickens : ℚ := chicken_fraction * total_birds
  total_cost / num_chickens

theorem cost_per_chicken_problem :
  cost_per_chicken 15 (1/3) 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_chicken_problem_l1714_171408


namespace NUMINAMATH_CALUDE_mei_wendin_equation_theory_l1714_171411

theorem mei_wendin_equation_theory :
  ∀ x y : ℚ,
  (3 * x + 6 * y = 47/10) →
  (5 * x + 3 * y = 11/2) →
  (x = 9/10 ∧ y = 1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_mei_wendin_equation_theory_l1714_171411


namespace NUMINAMATH_CALUDE_kids_in_movie_l1714_171419

theorem kids_in_movie (riverside_total : ℕ) (westside_total : ℕ) (mountaintop_total : ℕ)
  (riverside_denied_percent : ℚ) (westside_denied_percent : ℚ) (mountaintop_denied_percent : ℚ)
  (h1 : riverside_total = 120)
  (h2 : westside_total = 90)
  (h3 : mountaintop_total = 50)
  (h4 : riverside_denied_percent = 20/100)
  (h5 : westside_denied_percent = 70/100)
  (h6 : mountaintop_denied_percent = 1/2) :
  ↑riverside_total - ↑riverside_total * riverside_denied_percent +
  ↑westside_total - ↑westside_total * westside_denied_percent +
  ↑mountaintop_total - ↑mountaintop_total * mountaintop_denied_percent = 148 := by
  sorry

end NUMINAMATH_CALUDE_kids_in_movie_l1714_171419


namespace NUMINAMATH_CALUDE_existence_of_equal_differences_l1714_171486

theorem existence_of_equal_differences (n : ℕ) (a : Fin (2 * n) → ℕ)
  (h_n : n ≥ 3)
  (h_a : ∀ i j : Fin (2 * n), i < j → a i < a j)
  (h_bounds : ∀ i : Fin (2 * n), 1 ≤ a i ∧ a i ≤ n^2) :
  ∃ i₁ i₂ i₃ i₄ i₅ i₆ : Fin (2 * n),
    i₁ < i₂ ∧ i₂ ≤ i₃ ∧ i₃ < i₄ ∧ i₄ ≤ i₅ ∧ i₅ < i₆ ∧
    a i₂ - a i₁ = a i₄ - a i₃ ∧ a i₄ - a i₃ = a i₆ - a i₅ :=
by sorry

end NUMINAMATH_CALUDE_existence_of_equal_differences_l1714_171486


namespace NUMINAMATH_CALUDE_dani_initial_pants_l1714_171480

/-- Represents the number of pants Dani receives each year as a reward -/
def yearly_reward : ℕ := 4 * 2

/-- Represents the number of years -/
def years : ℕ := 5

/-- Represents the total number of pants Dani will have after 5 years -/
def total_pants : ℕ := 90

/-- Calculates the number of pants Dani initially had -/
def initial_pants : ℕ := total_pants - (yearly_reward * years)

theorem dani_initial_pants :
  initial_pants = 50 := by sorry

end NUMINAMATH_CALUDE_dani_initial_pants_l1714_171480


namespace NUMINAMATH_CALUDE_isabella_babysitting_weeks_l1714_171476

/-- Calculates the number of weeks Isabella has been babysitting -/
def weeks_babysitting (hourly_rate : ℚ) (hours_per_day : ℚ) (days_per_week : ℚ) (total_earnings : ℚ) : ℚ :=
  total_earnings / (hourly_rate * hours_per_day * days_per_week)

/-- Proves that Isabella has been babysitting for 7 weeks -/
theorem isabella_babysitting_weeks :
  weeks_babysitting 5 5 6 1050 = 7 := by
  sorry

end NUMINAMATH_CALUDE_isabella_babysitting_weeks_l1714_171476


namespace NUMINAMATH_CALUDE_gcd_324_243_l1714_171475

theorem gcd_324_243 : Nat.gcd 324 243 = 81 := by
  sorry

end NUMINAMATH_CALUDE_gcd_324_243_l1714_171475


namespace NUMINAMATH_CALUDE_die_probability_l1714_171474

theorem die_probability (total_faces : ℕ) (red_faces : ℕ) (yellow_faces : ℕ) (blue_faces : ℕ)
  (h1 : total_faces = 11)
  (h2 : red_faces = 5)
  (h3 : yellow_faces = 4)
  (h4 : blue_faces = 2)
  (h5 : total_faces = red_faces + yellow_faces + blue_faces) :
  (yellow_faces : ℚ) / total_faces * (blue_faces : ℚ) / total_faces = 8 / 121 := by
  sorry

end NUMINAMATH_CALUDE_die_probability_l1714_171474


namespace NUMINAMATH_CALUDE_x_less_than_y_l1714_171431

theorem x_less_than_y : 123456789 * 123456786 < 123456788 * 123456787 := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_y_l1714_171431


namespace NUMINAMATH_CALUDE_a_neg_two_sufficient_not_necessary_l1714_171479

/-- The line l₁ with equation x + ay - 2 = 0 -/
def l₁ (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + a * p.2 - 2 = 0}

/-- The line l₂ with equation (a+1)x - ay + 1 = 0 -/
def l₂ (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (a + 1) * p.1 - a * p.2 + 1 = 0}

/-- Two lines are parallel if they have the same slope -/
def parallel (l₁ l₂ : Set (ℝ × ℝ)) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ l₁ ↔ (x, k * y) ∈ l₂

/-- Theorem stating that a = -2 is a sufficient but not necessary condition for l₁ ∥ l₂ -/
theorem a_neg_two_sufficient_not_necessary :
  (∃ (a : ℝ), a ≠ -2 ∧ parallel (l₁ a) (l₂ a)) ∧
  (parallel (l₁ (-2)) (l₂ (-2))) := by
  sorry

end NUMINAMATH_CALUDE_a_neg_two_sufficient_not_necessary_l1714_171479


namespace NUMINAMATH_CALUDE_expression_simplification_l1714_171429

theorem expression_simplification (a : ℝ) (h : a = 4) :
  (1 - (a + 1) / a) / ((a^2 - 1) / (a^2 - a)) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1714_171429


namespace NUMINAMATH_CALUDE_three_boys_ages_exist_l1714_171423

theorem three_boys_ages_exist : ∃ (A B C : ℝ), 
  A + B + C = 29.5 ∧ 
  C = 11.3 ∧ 
  (A = 2 * B ∨ B = 2 * C ∨ A = 2 * C) ∧
  A > 0 ∧ B > 0 ∧ C > 0 := by
  sorry

end NUMINAMATH_CALUDE_three_boys_ages_exist_l1714_171423


namespace NUMINAMATH_CALUDE_vector_projection_l1714_171456

/-- Given vectors a and b in ℝ², prove that the projection of a onto 2√3b is √65/5 -/
theorem vector_projection (a b : ℝ × ℝ) (h1 : a = (2, 3)) (h2 : b = (-4, 7)) :
  let proj := (a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2) * Real.sqrt 3 * 2
  proj = Real.sqrt 65 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_l1714_171456


namespace NUMINAMATH_CALUDE_gerbil_weight_l1714_171498

/-- The combined weight of two gerbils given the weights and relationships of three gerbils -/
theorem gerbil_weight (scruffy muffy puffy : ℕ) 
  (h1 : scruffy = 12)
  (h2 : muffy = scruffy - 3)
  (h3 : puffy = muffy + 5) :
  puffy + muffy = 23 := by
  sorry

end NUMINAMATH_CALUDE_gerbil_weight_l1714_171498


namespace NUMINAMATH_CALUDE_a_value_l1714_171461

def round_down_tens (n : ℕ) : ℕ :=
  (n / 10) * 10

theorem a_value (A : ℕ) : 
  A < 10 → 
  round_down_tens (A * 1000 + 567) = 2560 → 
  A = 2 := by
sorry

end NUMINAMATH_CALUDE_a_value_l1714_171461


namespace NUMINAMATH_CALUDE_solve_for_t_l1714_171464

theorem solve_for_t (s t : ℤ) (eq1 : 9 * s + 5 * t = 108) (eq2 : s = t - 2) : t = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_t_l1714_171464


namespace NUMINAMATH_CALUDE_product_x_z_l1714_171422

-- Define the parallelogram EFGH
structure Parallelogram :=
  (E F G H : ℝ × ℝ)
  (is_parallelogram : True)  -- This is a placeholder for the parallelogram property

-- Define the lengths of the sides
def side_length (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem product_x_z (EFGH : Parallelogram) 
  (EF_length : side_length EFGH.E EFGH.F = 52)
  (FG_length : ∃ z, side_length EFGH.F EFGH.G = 4 * z^2 + 4)
  (GH_length : ∃ x, side_length EFGH.G EFGH.H = 5 * x + 6)
  (HE_length : side_length EFGH.H EFGH.E = 16) :
  ∃ x z, x * z = 46 * Real.sqrt 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_product_x_z_l1714_171422


namespace NUMINAMATH_CALUDE_city_tax_solution_l1714_171403

/-- Represents the tax system of a city --/
structure CityTax where
  residents : ℕ
  taxPerResident : ℕ

/-- The conditions of the tax system --/
def taxConditions (ct : CityTax) : Prop :=
  (ct.residents + 3000) * (ct.taxPerResident - 10) = ct.residents * ct.taxPerResident ∧
  (ct.residents - 1000) * (ct.taxPerResident + 10) = ct.residents * ct.taxPerResident

/-- The theorem stating the solution to the problem --/
theorem city_tax_solution (ct : CityTax) (h : taxConditions ct) :
  ct.residents = 3000 ∧ ct.taxPerResident = 20 ∧ ct.residents * ct.taxPerResident = 60000 := by
  sorry


end NUMINAMATH_CALUDE_city_tax_solution_l1714_171403


namespace NUMINAMATH_CALUDE_bus_arrival_probability_l1714_171417

/-- The probability of a bus arriving on time for a single ride -/
def p : ℝ := 0.9

/-- The number of total rides -/
def n : ℕ := 5

/-- The number of on-time arrivals we're interested in -/
def k : ℕ := 4

/-- The binomial probability of k successes in n trials with probability p -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k)

theorem bus_arrival_probability :
  binomial_probability n k p = 0.328 := by
  sorry

end NUMINAMATH_CALUDE_bus_arrival_probability_l1714_171417


namespace NUMINAMATH_CALUDE_cyclic_sum_square_inequality_l1714_171459

theorem cyclic_sum_square_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) : 
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 ≥ 4 * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅ + x₅*x₁) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_square_inequality_l1714_171459


namespace NUMINAMATH_CALUDE_original_ghee_quantity_l1714_171420

/-- Represents the composition of ghee -/
structure GheeComposition where
  pure : Rat
  vanaspati : Rat
  impurities : Rat

/-- The original ghee composition -/
def originalComposition : GheeComposition :=
  { pure := 40/100, vanaspati := 30/100, impurities := 30/100 }

/-- The desired final ghee composition -/
def desiredComposition : GheeComposition :=
  { pure := 45/100, vanaspati := 25/100, impurities := 30/100 }

/-- The amount of pure ghee added (in kg) -/
def addedPureGhee : Rat := 20

/-- Theorem stating the original quantity of blended ghee -/
theorem original_ghee_quantity : 
  ∃ (x : Rat), 
    (originalComposition.pure * x + addedPureGhee = desiredComposition.pure * (x + addedPureGhee)) ∧
    (originalComposition.vanaspati * x = desiredComposition.vanaspati * (x + addedPureGhee)) ∧
    x = 220 := by
  sorry

end NUMINAMATH_CALUDE_original_ghee_quantity_l1714_171420


namespace NUMINAMATH_CALUDE_shirt_pricing_l1714_171426

theorem shirt_pricing (total_shirts : ℕ) (first_shirt_price second_shirt_price : ℚ) 
  (remaining_shirts : ℕ) (min_avg_remaining : ℚ) :
  total_shirts = 6 →
  first_shirt_price = 40 →
  second_shirt_price = 50 →
  remaining_shirts = 4 →
  min_avg_remaining = 52.5 →
  (first_shirt_price + second_shirt_price + remaining_shirts * min_avg_remaining) / total_shirts = 50 := by
  sorry

end NUMINAMATH_CALUDE_shirt_pricing_l1714_171426


namespace NUMINAMATH_CALUDE_planes_perpendicular_l1714_171478

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : parallel m n) 
  (h4 : parallel_plane m α) 
  (h5 : perpendicular n β) : 
  perpendicular_plane α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l1714_171478


namespace NUMINAMATH_CALUDE_forty_six_in_sequence_l1714_171473

def laila_sequence (n : ℕ) : ℕ :=
  4 + 7 * (n - 1)

theorem forty_six_in_sequence : ∃ n : ℕ, laila_sequence n = 46 := by
  sorry

end NUMINAMATH_CALUDE_forty_six_in_sequence_l1714_171473


namespace NUMINAMATH_CALUDE_ned_bomb_diffusion_l1714_171491

/-- Ned's bomb diffusion problem -/
theorem ned_bomb_diffusion (total_flights : ℕ) (time_per_flight : ℕ) (bomb_timer : ℕ) (time_spent : ℕ)
  (h1 : total_flights = 20)
  (h2 : time_per_flight = 11)
  (h3 : bomb_timer = 72)
  (h4 : time_spent = 165) :
  bomb_timer - (total_flights - time_spent / time_per_flight) * time_per_flight = 17 := by
  sorry

#check ned_bomb_diffusion

end NUMINAMATH_CALUDE_ned_bomb_diffusion_l1714_171491


namespace NUMINAMATH_CALUDE_age_puzzle_l1714_171416

theorem age_puzzle (A : ℕ) (h : A = 32) : ∃ N : ℚ, N * (A + 4) - 4 * (A - 4) = A ∧ N = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l1714_171416


namespace NUMINAMATH_CALUDE_parking_lot_wheels_l1714_171439

/-- Calculates the total number of wheels in a parking lot with specific vehicles and conditions -/
theorem parking_lot_wheels : 
  let cars := 14
  let bikes := 5
  let unicycles := 3
  let twelve_wheeler_trucks := 2
  let eighteen_wheeler_truck := 1
  let cars_with_missing_wheel := 2
  let truck_with_damaged_wheels := 1
  let damaged_wheels := 3

  let car_wheels := cars * 4 - cars_with_missing_wheel * 1
  let bike_wheels := bikes * 2
  let unicycle_wheels := unicycles * 1
  let twelve_wheeler_truck_wheels := twelve_wheeler_trucks * 12 - damaged_wheels
  let eighteen_wheeler_truck_wheels := eighteen_wheeler_truck * 18

  car_wheels + bike_wheels + unicycle_wheels + twelve_wheeler_truck_wheels + eighteen_wheeler_truck_wheels = 106 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_wheels_l1714_171439


namespace NUMINAMATH_CALUDE_at_least_one_angle_le_30_deg_l1714_171455

-- Define a triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define a point P
variable (P : Point)

-- Define that P is inside the triangle
def isInside (P : Point) (t : Triangle) : Prop := sorry

-- Define the angle between three points
def angle (A B C : Point) : ℝ := sorry

-- Theorem statement
theorem at_least_one_angle_le_30_deg (t : Triangle) (P : Point) 
  (h : isInside P t) : 
  (angle P t.A t.B ≤ 30) ∨ (angle P t.B t.C ≤ 30) ∨ (angle P t.C t.A ≤ 30) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_angle_le_30_deg_l1714_171455


namespace NUMINAMATH_CALUDE_ab_squared_nonpositive_l1714_171457

theorem ab_squared_nonpositive (a b : ℝ) (h : 7 * a + 9 * |b| = 0) : a * b^2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_squared_nonpositive_l1714_171457


namespace NUMINAMATH_CALUDE_complex_roots_on_circle_l1714_171434

theorem complex_roots_on_circle : 
  ∀ z : ℂ, (z + 2)^5 = 64 * z^5 → Complex.abs (z + 2/15) = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_on_circle_l1714_171434


namespace NUMINAMATH_CALUDE_sum_of_solutions_l1714_171424

theorem sum_of_solutions (x y : ℝ) 
  (hx : (x - 1)^3 + 2015*(x - 1) = -1) 
  (hy : (y - 1)^3 + 2015*(y - 1) = 1) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l1714_171424


namespace NUMINAMATH_CALUDE_distance_between_points_l1714_171430

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 6)
  let p2 : ℝ × ℝ := (5, 2)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 5 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l1714_171430


namespace NUMINAMATH_CALUDE_exam_duration_l1714_171402

/-- Proves that the examination time is 30 hours given the specified conditions -/
theorem exam_duration (total_questions : ℕ) (type_a_questions : ℕ) (type_a_time : ℝ) :
  total_questions = 200 →
  type_a_questions = 10 →
  type_a_time = 17.142857142857142 →
  (total_questions - type_a_questions) * (type_a_time / 2) + type_a_questions * type_a_time = 30 * 60 := by
  sorry

end NUMINAMATH_CALUDE_exam_duration_l1714_171402


namespace NUMINAMATH_CALUDE_digit_2457_is_5_l1714_171438

/-- The decimal number constructed by concatenating integers from 1 to 999 -/
def x : ℝ := sorry

/-- The nth digit after the decimal point in the number x -/
def digit_at (n : ℕ) : ℕ := sorry

/-- The theorem stating that the 2457th digit of x is 5 -/
theorem digit_2457_is_5 : digit_at 2457 = 5 := by sorry

end NUMINAMATH_CALUDE_digit_2457_is_5_l1714_171438


namespace NUMINAMATH_CALUDE_garden_area_increase_l1714_171454

/-- Proves that changing a 60-foot by 20-foot rectangular garden to a square garden 
    with the same perimeter results in an increase of 400 square feet in area. -/
theorem garden_area_increase : 
  let rectangle_length : ℝ := 60
  let rectangle_width : ℝ := 20
  let rectangle_area := rectangle_length * rectangle_width
  let perimeter := 2 * (rectangle_length + rectangle_width)
  let square_side := perimeter / 4
  let square_area := square_side * square_side
  square_area - rectangle_area = 400 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_increase_l1714_171454


namespace NUMINAMATH_CALUDE_garden_perimeter_l1714_171460

/-- A rectangular garden with given diagonal and area has a specific perimeter. -/
theorem garden_perimeter (a b : ℝ) : 
  a > 0 → b > 0 → -- Positive side lengths
  a^2 + b^2 = 15^2 → -- Diagonal condition
  a * b = 54 → -- Area condition
  2 * (a + b) = 2 * Real.sqrt 333 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l1714_171460


namespace NUMINAMATH_CALUDE_equilateral_triangle_complex_l1714_171415

/-- Given complex numbers a, b, c forming an equilateral triangle with side length 24
    and |a + b + c| = 48, prove that |ab + ac + bc| = 768. -/
theorem equilateral_triangle_complex (a b c : ℂ) :
  (∃ (w : ℂ), w^3 = 1 ∧ w ≠ 1 ∧ b - a = 24 * w ∧ c - a = 24 * w^2) →
  Complex.abs (a + b + c) = 48 →
  Complex.abs (a * b + a * c + b * c) = 768 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_complex_l1714_171415


namespace NUMINAMATH_CALUDE_family_egg_count_l1714_171446

/-- Calculates the final number of eggs a family has after various events --/
theorem family_egg_count (initial_eggs : ℚ) 
                          (mother_used : ℚ) 
                          (father_used : ℚ) 
                          (chicken1_laid : ℚ) 
                          (chicken2_laid : ℚ) 
                          (chicken3_laid : ℚ) 
                          (chicken4_laid : ℚ) 
                          (oldest_child_took : ℚ) 
                          (youngest_child_broke : ℚ) : 
  initial_eggs = 25 ∧ 
  mother_used = 7.5 ∧ 
  father_used = 2.5 ∧ 
  chicken1_laid = 2.5 ∧ 
  chicken2_laid = 3 ∧ 
  chicken3_laid = 4.5 ∧ 
  chicken4_laid = 1 ∧ 
  oldest_child_took = 1.5 ∧ 
  youngest_child_broke = 0.5 → 
  initial_eggs - (mother_used + father_used) + 
  (chicken1_laid + chicken2_laid + chicken3_laid + chicken4_laid) - 
  (oldest_child_took + youngest_child_broke) = 24 := by
  sorry


end NUMINAMATH_CALUDE_family_egg_count_l1714_171446


namespace NUMINAMATH_CALUDE_prime_squared_minus_one_divisible_by_24_l1714_171477

theorem prime_squared_minus_one_divisible_by_24 (n : ℕ) 
  (h_prime : Nat.Prime n) (h_not_two : n ≠ 2) (h_not_three : n ≠ 3) :
  24 ∣ (n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_squared_minus_one_divisible_by_24_l1714_171477


namespace NUMINAMATH_CALUDE_algebraic_simplification_l1714_171490

theorem algebraic_simplification (a b : ℝ) : -3*a*(2*a - 4*b + 2) + 6*a = -6*a^2 + 12*a*b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l1714_171490


namespace NUMINAMATH_CALUDE_decimal_25_equals_base5_100_l1714_171463

/-- Converts a natural number to its base 5 representation --/
def toBaseFive (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBaseFive (n / 5)

/-- Theorem: The decimal number 25 is equivalent to 100₅ in base 5 --/
theorem decimal_25_equals_base5_100 : toBaseFive 25 = [0, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_decimal_25_equals_base5_100_l1714_171463


namespace NUMINAMATH_CALUDE_marias_green_towels_l1714_171436

theorem marias_green_towels :
  ∀ (green_towels : ℕ),
  (green_towels + 21 : ℕ) - 34 = 22 →
  green_towels = 35 :=
by sorry

end NUMINAMATH_CALUDE_marias_green_towels_l1714_171436


namespace NUMINAMATH_CALUDE_max_area_rectangle_l1714_171401

def is_valid_rectangle (x y : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ x ≥ y

def cost (x y : ℕ) : ℕ :=
  2 * (3 * x + 5 * y)

def area (x y : ℕ) : ℕ :=
  x * y

theorem max_area_rectangle :
  ∃ (x y : ℕ), is_valid_rectangle x y ∧ cost x y ≤ 100 ∧
  area x y = 40 ∧
  ∀ (a b : ℕ), is_valid_rectangle a b → cost a b ≤ 100 → area a b ≤ 40 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l1714_171401


namespace NUMINAMATH_CALUDE_sun_division_l1714_171409

/-- Represents the share distribution among x, y, and z -/
structure ShareDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The problem statement -/
theorem sun_division (s : ShareDistribution) :
  s.x = 1 →                -- For each rupee x gets
  s.y = 0.45 →             -- y gets 45 paisa (0.45 rupees)
  s.z = 0.5 →              -- z gets 50 paisa (0.5 rupees)
  s.y * (1 / 0.45) = 45 →  -- The share of y is Rs. 45
  s.x * (1 / 0.45) + s.y * (1 / 0.45) + s.z * (1 / 0.45) = 195 := by
  sorry

#check sun_division

end NUMINAMATH_CALUDE_sun_division_l1714_171409


namespace NUMINAMATH_CALUDE_paint_remaining_is_three_eighths_l1714_171484

/-- The fraction of paint remaining after two days of use -/
def paint_remaining (initial_amount : ℚ) (first_day_use : ℚ) (second_day_use : ℚ) : ℚ :=
  initial_amount - (first_day_use * initial_amount) - (second_day_use * (initial_amount - first_day_use * initial_amount))

/-- Theorem stating that the fraction of paint remaining after two days is 3/8 -/
theorem paint_remaining_is_three_eighths :
  paint_remaining 1 (1/4) (1/2) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_paint_remaining_is_three_eighths_l1714_171484


namespace NUMINAMATH_CALUDE_extreme_value_condition_l1714_171468

/-- The function f(x) = ax + ln(x) has an extreme value -/
def has_extreme_value (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y > 0 → a * x + Real.log x ≥ a * y + Real.log y) ∨
                   (∀ y : ℝ, y > 0 → a * x + Real.log x ≤ a * y + Real.log y)

/-- a ≤ 0 is a necessary but not sufficient condition for f(x) = ax + ln(x) to have an extreme value -/
theorem extreme_value_condition (a : ℝ) :
  has_extreme_value a → a ≤ 0 ∧ ∃ b : ℝ, b ≤ 0 ∧ ¬has_extreme_value b :=
sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l1714_171468


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l1714_171404

theorem largest_constant_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) ≤ (Real.sqrt 6 / 2) * Real.sqrt (x + y + z) ∧
  ∀ k : ℝ, (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    a / Real.sqrt (b + c) + b / Real.sqrt (c + a) + c / Real.sqrt (a + b) ≤ k * Real.sqrt (a + b + c)) →
  k ≤ Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l1714_171404


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1714_171485

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, x * (x - 1) ≤ 0 ↔ 0 ≤ x ∧ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1714_171485


namespace NUMINAMATH_CALUDE_jans_cable_sections_l1714_171447

theorem jans_cable_sections (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hxy : y < x) :
  z = (51 * x) / (160 * y) → z = ((51 : ℕ) / 160) * (x / y) := by
sorry

end NUMINAMATH_CALUDE_jans_cable_sections_l1714_171447


namespace NUMINAMATH_CALUDE_expression_simplification_l1714_171470

theorem expression_simplification :
  let a := Real.sqrt 2
  let b := Real.sqrt 3
  b > a ∧ (8 : ℝ) ^ (1/3) = 2 →
  |a - b| + (8 : ℝ) ^ (1/3) - a * (a - 1) = b := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l1714_171470


namespace NUMINAMATH_CALUDE_fair_hair_percentage_l1714_171442

/-- Given that 10% of employees are women with fair hair and 40% of fair-haired employees
    are women, prove that 25% of employees have fair hair. -/
theorem fair_hair_percentage
  (total_employees : ℕ)
  (women_fair_hair_percentage : ℚ)
  (women_percentage_of_fair_hair : ℚ)
  (h1 : women_fair_hair_percentage = 1 / 10)
  (h2 : women_percentage_of_fair_hair = 2 / 5)
  : (total_employees : ℚ) * 1 / 4 = (total_employees : ℚ) * women_fair_hair_percentage / women_percentage_of_fair_hair :=
by sorry

end NUMINAMATH_CALUDE_fair_hair_percentage_l1714_171442


namespace NUMINAMATH_CALUDE_ant_count_approximation_l1714_171467

/-- Represents the dimensions of a rectangular field in feet -/
structure FieldDimensions where
  width : ℝ
  length : ℝ

/-- Calculates the number of ants in a rectangular field given the specified conditions -/
def calculateAnts (field : FieldDimensions) (antDensity : ℝ) (rockCoverage : ℝ) : ℝ :=
  let inchesPerFoot : ℝ := 12
  let fieldAreaInches : ℝ := field.width * field.length * inchesPerFoot * inchesPerFoot
  let antHabitatArea : ℝ := fieldAreaInches * (1 - rockCoverage)
  antHabitatArea * antDensity

/-- Theorem stating that the number of ants in the field is approximately 26 million -/
theorem ant_count_approximation :
  let field : FieldDimensions := { width := 200, length := 500 }
  let antDensity : ℝ := 2  -- ants per square inch
  let rockCoverage : ℝ := 0.1  -- 10% of the field covered by rocks
  abs (calculateAnts field antDensity rockCoverage - 26000000) ≤ 500000 := by
  sorry


end NUMINAMATH_CALUDE_ant_count_approximation_l1714_171467


namespace NUMINAMATH_CALUDE_emma_yield_calculation_l1714_171489

/-- The annual yield percentage of Emma's investment -/
def emma_yield : ℝ := 18.33

/-- Emma's investment amount -/
def emma_investment : ℝ := 300

/-- Briana's investment amount -/
def briana_investment : ℝ := 500

/-- Briana's annual yield percentage -/
def briana_yield : ℝ := 10

/-- The difference in return-on-investment after 2 years -/
def roi_difference : ℝ := 10

/-- The number of years for the investment -/
def years : ℝ := 2

theorem emma_yield_calculation :
  emma_investment * (emma_yield / 100) * years - 
  briana_investment * (briana_yield / 100) * years = roi_difference :=
sorry

end NUMINAMATH_CALUDE_emma_yield_calculation_l1714_171489


namespace NUMINAMATH_CALUDE_lesser_solution_quadratic_l1714_171448

theorem lesser_solution_quadratic (x : ℝ) : 
  x^2 + 10*x - 24 = 0 ∧ (∀ y : ℝ, y^2 + 10*y - 24 = 0 → x ≤ y) → x = -12 := by
  sorry

end NUMINAMATH_CALUDE_lesser_solution_quadratic_l1714_171448


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l1714_171400

/-- Given the initial conditions and new averages after each person enters,
    prove that the weights of X, Y, and Z are 195 lbs, 141 lbs, and 126 lbs respectively. -/
theorem elevator_weight_problem (initial_people : Nat) (initial_avg : ℝ)
    (avg_after_X : ℝ) (avg_after_Y : ℝ) (avg_after_Z : ℝ)
    (h1 : initial_people = 6)
    (h2 : initial_avg = 160)
    (h3 : avg_after_X = 165)
    (h4 : avg_after_Y = 162)
    (h5 : avg_after_Z = 158) :
    ∃ (X Y Z : ℝ),
      X = 195 ∧
      Y = 141 ∧
      Z = 126 ∧
      (initial_people : ℝ) * initial_avg + X = (initial_people + 1 : ℝ) * avg_after_X ∧
      ((initial_people + 1 : ℝ) * avg_after_X + Y = (initial_people + 2 : ℝ) * avg_after_Y) ∧
      ((initial_people + 2 : ℝ) * avg_after_Y + Z = (initial_people + 3 : ℝ) * avg_after_Z) :=
by sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l1714_171400


namespace NUMINAMATH_CALUDE_bus_problem_l1714_171425

/-- The number of children initially on the bus -/
def initial_children : ℕ := 5

/-- The number of children who got off the bus -/
def children_off : ℕ := 63

/-- The number of children who got on the bus -/
def children_on : ℕ := children_off + 9

/-- The number of children on the bus after the changes -/
def final_children : ℕ := 14

theorem bus_problem :
  initial_children - children_off + children_on = final_children :=
by sorry

end NUMINAMATH_CALUDE_bus_problem_l1714_171425


namespace NUMINAMATH_CALUDE_different_color_chips_probability_l1714_171418

/-- The probability of drawing two chips of different colors from a bag containing
    7 blue chips, 5 red chips, 4 yellow chips, and 3 green chips, when drawing
    with replacement. -/
theorem different_color_chips_probability
  (blue : ℕ) (red : ℕ) (yellow : ℕ) (green : ℕ)
  (h_blue : blue = 7)
  (h_red : red = 5)
  (h_yellow : yellow = 4)
  (h_green : green = 3) :
  let total := blue + red + yellow + green
  (blue * (total - blue) + red * (total - red) + yellow * (total - yellow) + green * (total - green)) / (total * total) = 262 / 361 :=
by sorry

end NUMINAMATH_CALUDE_different_color_chips_probability_l1714_171418


namespace NUMINAMATH_CALUDE_length_BC_is_44_div_3_l1714_171492

/-- Two externally tangent circles with a common external tangent line -/
structure TangentCircles where
  /-- Center of the first circle -/
  A : ℝ × ℝ
  /-- Center of the second circle -/
  B : ℝ × ℝ
  /-- Radius of the first circle -/
  r₁ : ℝ
  /-- Radius of the second circle -/
  r₂ : ℝ
  /-- Point where the external tangent line intersects ray AB -/
  C : ℝ × ℝ
  /-- The circles are externally tangent -/
  externally_tangent : dist A B = r₁ + r₂
  /-- The line through C is externally tangent to both circles -/
  is_external_tangent : ∃ (D E : ℝ × ℝ), 
    dist A D = r₁ ∧ dist B E = r₂ ∧ 
    (C.1 - D.1) * (A.1 - D.1) + (C.2 - D.2) * (A.2 - D.2) = 0 ∧
    (C.1 - E.1) * (B.1 - E.1) + (C.2 - E.2) * (B.2 - E.2) = 0
  /-- C lies on ray AB -/
  C_on_ray_AB : ∃ (t : ℝ), t ≥ 0 ∧ C = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

/-- The length of BC in the TangentCircles configuration -/
def length_BC (tc : TangentCircles) : ℝ :=
  dist tc.B tc.C

/-- The main theorem: length of BC is 44/3 -/
theorem length_BC_is_44_div_3 (tc : TangentCircles) (h₁ : tc.r₁ = 7) (h₂ : tc.r₂ = 4) : 
  length_BC tc = 44 / 3 := by
  sorry


end NUMINAMATH_CALUDE_length_BC_is_44_div_3_l1714_171492


namespace NUMINAMATH_CALUDE_john_climbs_70_feet_l1714_171482

/-- Calculates the total height climbed by John given the number of flights, height per flight, and additional ladder length. -/
def totalHeightClimbed (numFlights : ℕ) (flightHeight : ℕ) (additionalLadderLength : ℕ) : ℕ :=
  let stairsHeight := numFlights * flightHeight
  let ropeHeight := stairsHeight / 2
  let ladderHeight := ropeHeight + additionalLadderLength
  stairsHeight + ropeHeight + ladderHeight

/-- Theorem stating that under the given conditions, John climbs a total of 70 feet. -/
theorem john_climbs_70_feet :
  totalHeightClimbed 3 10 10 = 70 := by
  sorry

end NUMINAMATH_CALUDE_john_climbs_70_feet_l1714_171482


namespace NUMINAMATH_CALUDE_max_elephants_is_1036_l1714_171499

/-- The number of union members --/
def union_members : ℕ := 28

/-- The number of non-union members --/
def non_union_members : ℕ := 37

/-- The function that calculates the total number of elephants given the number
    of elephants per union member and per non-union member --/
def total_elephants (elephants_per_union : ℕ) (elephants_per_non_union : ℕ) : ℕ :=
  union_members * elephants_per_union + non_union_members * elephants_per_non_union

/-- The theorem stating that 1036 is the maximum number of elephants that can be distributed --/
theorem max_elephants_is_1036 :
  ∃ (eu en : ℕ), 
    eu ≠ en ∧ 
    eu > 0 ∧ 
    en > 0 ∧
    total_elephants eu en = 1036 ∧
    (∀ (x y : ℕ), x ≠ y → x > 0 → y > 0 → total_elephants x y ≤ 1036) :=
sorry

end NUMINAMATH_CALUDE_max_elephants_is_1036_l1714_171499


namespace NUMINAMATH_CALUDE_ines_initial_amount_l1714_171413

/-- The amount of money Ines had in her purse initially -/
def initial_amount : ℕ := 20

/-- The number of pounds of peaches Ines bought -/
def peaches_bought : ℕ := 3

/-- The cost per pound of peaches -/
def cost_per_pound : ℕ := 2

/-- The amount of money Ines had left after buying peaches -/
def amount_left : ℕ := 14

/-- Theorem stating that Ines had $20 in her purse initially -/
theorem ines_initial_amount :
  initial_amount = peaches_bought * cost_per_pound + amount_left :=
by sorry

end NUMINAMATH_CALUDE_ines_initial_amount_l1714_171413


namespace NUMINAMATH_CALUDE_range_of_t_l1714_171452

theorem range_of_t (a b c t : ℝ) 
  (eq1 : 6 * a = 2 * b - 6)
  (eq2 : 6 * a = 3 * c)
  (cond1 : b ≥ 0)
  (cond2 : c ≤ 2)
  (def_t : t = 2 * a + b - c) :
  0 ≤ t ∧ t ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_t_l1714_171452


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1714_171435

/-- The number of terms between 400 and 600 in an arithmetic sequence -/
theorem arithmetic_sequence_terms (a₁ : ℝ) (d : ℝ) (n : ℕ) :
  a₁ = 110 →
  d = 6 →
  (∃ k₁ k₂ : ℕ, 
    a₁ + (k₁ - 1) * d ≥ 400 ∧
    a₁ + (k₁ - 1) * d < a₁ + k₁ * d ∧
    a₁ + (k₂ - 1) * d ≤ 600 ∧
    a₁ + k₂ * d > 600 ∧
    k₂ - k₁ + 1 = 33) :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1714_171435


namespace NUMINAMATH_CALUDE_initial_number_proof_l1714_171496

theorem initial_number_proof (x : ℝ) : (x - 1/4) / (1/2) = 4.5 → x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l1714_171496


namespace NUMINAMATH_CALUDE_product_xyz_equals_two_l1714_171483

theorem product_xyz_equals_two
  (x y z : ℝ)
  (h1 : x + 1 / y = 2)
  (h2 : y + 1 / z = 2)
  (h3 : x + 1 / z = 3) :
  x * y * z = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_equals_two_l1714_171483


namespace NUMINAMATH_CALUDE_parallelogram_area_l1714_171462

/-- A parallelogram bounded by lines y = a, y = -b, x = -c + 2y, and x = d - 2y -/
structure Parallelogram where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d

/-- The area of the parallelogram -/
def area (p : Parallelogram) : ℝ :=
  p.a * p.d + p.a * p.c + p.b * p.d + p.b * p.c

theorem parallelogram_area (p : Parallelogram) :
  area p = p.a * p.d + p.a * p.c + p.b * p.d + p.b * p.c := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1714_171462


namespace NUMINAMATH_CALUDE_fraction_sum_equals_point_three_l1714_171440

theorem fraction_sum_equals_point_three :
  5 / 50 + 4 / 40 + 6 / 60 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_point_three_l1714_171440


namespace NUMINAMATH_CALUDE_complex_expression_equals_negative_i_l1714_171427

theorem complex_expression_equals_negative_i :
  let i : ℂ := Complex.I
  (1 + 2*i) * i^3 + 2*i^2 = -i :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equals_negative_i_l1714_171427


namespace NUMINAMATH_CALUDE_expression_result_l1714_171444

theorem expression_result : (3.242 * 14) / 100 = 0.45388 := by
  sorry

end NUMINAMATH_CALUDE_expression_result_l1714_171444


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1714_171405

/-- The probability of getting exactly k successes in n trials --/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 4 heads in 10 flips of a coin with 3/7 probability of heads --/
theorem coin_flip_probability : 
  binomial_probability 10 4 (3/7) = 69874560 / 282576201 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1714_171405


namespace NUMINAMATH_CALUDE_roses_given_l1714_171421

-- Define the total number of students
def total_students : ℕ := 28

-- Define the relationship between flowers
def flower_relationship (daffodils roses tulips : ℕ) : Prop :=
  roses = 4 * daffodils ∧ tulips = 10 * roses

-- Define the total number of flowers given
def total_flowers (boys girls : ℕ) : ℕ := boys * girls

-- Define the constraint that the total number of students is the sum of boys and girls
def student_constraint (boys girls : ℕ) : Prop :=
  boys + girls = total_students

-- Theorem statement
theorem roses_given (boys girls daffodils roses tulips : ℕ) :
  student_constraint boys girls →
  flower_relationship daffodils roses tulips →
  total_flowers boys girls = daffodils + roses + tulips →
  roses = 16 := by
  sorry

end NUMINAMATH_CALUDE_roses_given_l1714_171421


namespace NUMINAMATH_CALUDE_highway_traffic_l1714_171451

/-- The number of vehicles involved in accidents per 100 million vehicles -/
def accident_rate : ℚ := 75

/-- The total number of vehicles involved in accidents -/
def total_accidents : ℕ := 4500

/-- The number of vehicles (in millions) that traveled on the highway -/
def total_vehicles : ℕ := 6000

theorem highway_traffic :
  (accident_rate / 100000000) * (total_vehicles * 1000000) = total_accidents :=
sorry

end NUMINAMATH_CALUDE_highway_traffic_l1714_171451


namespace NUMINAMATH_CALUDE_solubility_product_scientific_notation_l1714_171443

theorem solubility_product_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 0.0000000028 = a * (10 : ℝ) ^ n :=
by sorry

end NUMINAMATH_CALUDE_solubility_product_scientific_notation_l1714_171443


namespace NUMINAMATH_CALUDE_zachary_sold_40_games_l1714_171472

/-- Represents the sale of video games by three friends -/
structure VideoGameSale where
  /-- Amount of money Zachary received -/
  zachary_amount : ℝ
  /-- Price per game Zachary sold -/
  price_per_game : ℝ
  /-- Total amount received by all three friends -/
  total_amount : ℝ

/-- Theorem stating that Zachary sold 40 games given the conditions -/
theorem zachary_sold_40_games (sale : VideoGameSale)
  (h1 : sale.price_per_game = 5)
  (h2 : sale.zachary_amount + (sale.zachary_amount * 1.3) + (sale.zachary_amount * 1.3 + 50) = sale.total_amount)
  (h3 : sale.total_amount = 770) :
  sale.zachary_amount / sale.price_per_game = 40 := by
sorry


end NUMINAMATH_CALUDE_zachary_sold_40_games_l1714_171472


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_sum_of_roots_specific_cubic_l1714_171437

theorem sum_of_roots_cubic : ∀ (a b c d : ℝ),
  (∃ x y z : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ∧
                a * y^3 + b * y^2 + c * y + d = 0 ∧
                a * z^3 + b * z^2 + c * z + d = 0 ∧
                (∀ w : ℝ, a * w^3 + b * w^2 + c * w + d = 0 → w = x ∨ w = y ∨ w = z)) →
  x + y + z = -b / a :=
by sorry

theorem sum_of_roots_specific_cubic :
  ∃ x y z : ℝ, x^3 - 3*x^2 - 12*x - 7 = 0 ∧
              y^3 - 3*y^2 - 12*y - 7 = 0 ∧
              z^3 - 3*z^2 - 12*z - 7 = 0 ∧
              (∀ w : ℝ, w^3 - 3*w^2 - 12*w - 7 = 0 → w = x ∨ w = y ∨ w = z) ∧
              x + y + z = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_sum_of_roots_specific_cubic_l1714_171437


namespace NUMINAMATH_CALUDE_max_red_dragons_l1714_171406

-- Define the dragon colors
inductive DragonColor
| Red
| Green
| Blue

-- Define the structure of a dragon
structure Dragon where
  color : DragonColor
  heads : Fin 3 → Bool  -- Each head is either truthful (true) or lying (false)

-- Define the statements made by each head
def headStatements (d : Dragon) (left right : DragonColor) : Prop :=
  (d.heads 0 = (left = DragonColor.Green)) ∧
  (d.heads 1 = (right = DragonColor.Blue)) ∧
  (d.heads 2 = (left ≠ DragonColor.Red ∧ right ≠ DragonColor.Red))

-- Define the condition that at least one head tells the truth
def atLeastOneTruthful (d : Dragon) : Prop :=
  ∃ i : Fin 3, d.heads i = true

-- Define the arrangement of dragons around the table
def validArrangement (arrangement : Fin 530 → Dragon) : Prop :=
  ∀ i : Fin 530,
    let left := arrangement ((i.val - 1 + 530) % 530)
    let right := arrangement ((i.val + 1) % 530)
    headStatements (arrangement i) left.color right.color ∧
    atLeastOneTruthful (arrangement i)

-- The main theorem
theorem max_red_dragons :
  ∀ arrangement : Fin 530 → Dragon,
    validArrangement arrangement →
    (∃ n : Nat, n ≤ 176 ∧ (∀ i : Fin 530, (arrangement i).color = DragonColor.Red → i.val < n)) :=
sorry

end NUMINAMATH_CALUDE_max_red_dragons_l1714_171406


namespace NUMINAMATH_CALUDE_juggling_balls_count_l1714_171469

theorem juggling_balls_count (balls_per_juggler : ℕ) (number_of_jugglers : ℕ) (total_balls : ℕ) : 
  balls_per_juggler = 6 → 
  number_of_jugglers = 378 → 
  total_balls = balls_per_juggler * number_of_jugglers → 
  total_balls = 2268 := by
sorry

end NUMINAMATH_CALUDE_juggling_balls_count_l1714_171469


namespace NUMINAMATH_CALUDE_catholic_tower_height_l1714_171465

/-- Given two towers and a grain between them, prove the height of the second tower --/
theorem catholic_tower_height 
  (church_height : ℝ) 
  (total_distance : ℝ) 
  (grain_distance : ℝ) 
  (h : ℝ → church_height = 150 ∧ total_distance = 350 ∧ grain_distance = 150) :
  ∃ (catholic_height : ℝ), 
    catholic_height = 50 * Real.sqrt 5 ∧ 
    (church_height^2 + grain_distance^2 = 
     catholic_height^2 + (total_distance - grain_distance)^2) :=
by sorry

end NUMINAMATH_CALUDE_catholic_tower_height_l1714_171465


namespace NUMINAMATH_CALUDE_su_buqing_star_distance_l1714_171487

theorem su_buqing_star_distance (d : ℝ) : d = 218000000 → d = 2.18 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_su_buqing_star_distance_l1714_171487


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l1714_171471

theorem triangle_angle_calculation (a c : ℝ) (C : ℝ) (hA : a = 1) (hC : c = Real.sqrt 3) (hAngle : C = 2 * Real.pi / 3) :
  ∃ (A : ℝ), A = Real.pi / 6 ∧ 0 < A ∧ A < Real.pi ∧ 
  Real.sin A = a * Real.sin C / c ∧
  A + C < Real.pi :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l1714_171471


namespace NUMINAMATH_CALUDE_rectangular_to_spherical_conversion_l1714_171414

def rectangular_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ := sorry

theorem rectangular_to_spherical_conversion :
  let (ρ, θ, φ) := rectangular_to_spherical (4 * Real.sqrt 2) (-4) 4
  ρ = 8 ∧ θ = 7 * Real.pi / 4 ∧ φ = Real.pi / 3 ∧
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_spherical_conversion_l1714_171414


namespace NUMINAMATH_CALUDE_flower_bed_fraction_l1714_171497

/-- Represents a rectangular yard with two congruent isosceles right triangular flower beds -/
structure FlowerYard where
  length : ℝ
  width : ℝ
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ
  (short_side_positive : trapezoid_short_side > 0)
  (long_side_positive : trapezoid_long_side > 0)
  (short_less_than_long : trapezoid_short_side < trapezoid_long_side)
  (width_eq : width = (trapezoid_long_side - trapezoid_short_side) / 2)
  (length_eq : length = trapezoid_long_side)

/-- The fraction of the yard occupied by the flower beds is 1/5 -/
theorem flower_bed_fraction (yard : FlowerYard) (h1 : yard.trapezoid_short_side = 15) 
    (h2 : yard.trapezoid_long_side = 25) : 
  (2 * yard.width ^ 2) / (yard.length * yard.width) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_fraction_l1714_171497


namespace NUMINAMATH_CALUDE_ceiling_sqrt_sum_l1714_171481

theorem ceiling_sqrt_sum : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 27⌉ * 2 + ⌈Real.sqrt 243⌉ = 30 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_sum_l1714_171481


namespace NUMINAMATH_CALUDE_turtleneck_sweater_profit_l1714_171407

/-- Represents the pricing strategy and profit calculation for a store selling turtleneck sweaters -/
theorem turtleneck_sweater_profit (C : ℝ) : 
  let initial_markup := 0.20
  let new_year_markup := 0.25
  let february_discount := 0.08
  let SP1 := C * (1 + initial_markup)
  let SP2 := SP1 * (1 + new_year_markup)
  let SPF := SP2 * (1 - february_discount)
  let profit := SPF - C
  profit = 0.38 * C :=
by sorry

end NUMINAMATH_CALUDE_turtleneck_sweater_profit_l1714_171407


namespace NUMINAMATH_CALUDE_sphere_volume_surface_area_ratio_l1714_171495

theorem sphere_volume_surface_area_ratio 
  (r₁ r₂ : ℝ) 
  (h_positive₁ : r₁ > 0) 
  (h_positive₂ : r₂ > 0) 
  (h_volume_ratio : (4 / 3 * Real.pi * r₁^3) / (4 / 3 * Real.pi * r₂^3) = 8 / 27) : 
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4 / 9 := by
  sorry

#check sphere_volume_surface_area_ratio

end NUMINAMATH_CALUDE_sphere_volume_surface_area_ratio_l1714_171495


namespace NUMINAMATH_CALUDE_no_professors_are_student_council_members_l1714_171450

-- Define the universe
variable (U : Type)

-- Define the predicates
variable (Professor : U → Prop)
variable (StudentCouncilMember : U → Prop)
variable (Wise : U → Prop)

-- State the theorem
theorem no_professors_are_student_council_members
  (h1 : ∀ x, Professor x → Wise x)
  (h2 : ∀ x, StudentCouncilMember x → ¬Wise x) :
  ∀ x, Professor x → ¬StudentCouncilMember x :=
by sorry

end NUMINAMATH_CALUDE_no_professors_are_student_council_members_l1714_171450


namespace NUMINAMATH_CALUDE_competition_winner_and_probability_l1714_171494

def prob_A_win_round1 : ℚ := 3/5
def prob_B_win_round1 : ℚ := 3/4
def prob_A_win_round2 : ℚ := 3/5
def prob_B_win_round2 : ℚ := 1/2

def prob_A_win_competition : ℚ := prob_A_win_round1 * prob_A_win_round2
def prob_B_win_competition : ℚ := prob_B_win_round1 * prob_B_win_round2

theorem competition_winner_and_probability :
  (prob_B_win_competition > prob_A_win_competition) ∧
  (1 - (1 - prob_A_win_competition) * (1 - prob_B_win_competition) = 3/5) := by
  sorry

end NUMINAMATH_CALUDE_competition_winner_and_probability_l1714_171494
