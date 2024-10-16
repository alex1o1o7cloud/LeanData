import Mathlib

namespace NUMINAMATH_CALUDE_peters_height_l1133_113357

theorem peters_height (tree_height : ℝ) (tree_shadow : ℝ) (peter_shadow : ℝ) :
  tree_height = 100 → 
  tree_shadow = 25 → 
  peter_shadow = 1.5 → 
  (tree_height / tree_shadow) * peter_shadow * 12 = 72 := by
  sorry

end NUMINAMATH_CALUDE_peters_height_l1133_113357


namespace NUMINAMATH_CALUDE_total_nails_calculation_l1133_113397

/-- The number of nails left at each station -/
def nails_per_station : ℕ := 7

/-- The number of stations visited -/
def stations_visited : ℕ := 20

/-- The total number of nails brought -/
def total_nails : ℕ := nails_per_station * stations_visited

theorem total_nails_calculation : total_nails = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_nails_calculation_l1133_113397


namespace NUMINAMATH_CALUDE_plane_perp_theorem_l1133_113375

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a plane and a line
variable (perp_plane_line : Plane → Line → Prop)

-- Define the intersection operation between planes
variable (intersect : Plane → Plane → Line)

-- State the theorem
theorem plane_perp_theorem 
  (α β : Plane) (l : Line) 
  (h1 : perp_planes α β) 
  (h2 : intersect α β = l) :
  ∀ γ : Plane, perp_plane_line γ l → 
    perp_planes γ α ∧ perp_planes γ β :=
sorry

end NUMINAMATH_CALUDE_plane_perp_theorem_l1133_113375


namespace NUMINAMATH_CALUDE_jerrys_shelf_l1133_113380

/-- The number of books on Jerry's shelf -/
def num_books : ℕ := 3

/-- The initial number of action figures -/
def initial_action_figures : ℕ := 4

/-- The number of action figures added -/
def added_action_figures : ℕ := 2

/-- The difference between action figures and books -/
def difference : ℕ := 3

theorem jerrys_shelf :
  num_books = 3 ∧
  initial_action_figures + added_action_figures = num_books + difference :=
sorry

end NUMINAMATH_CALUDE_jerrys_shelf_l1133_113380


namespace NUMINAMATH_CALUDE_discount_percentage_l1133_113317

theorem discount_percentage (srp mp paid : ℝ) : 
  srp = 1.2 * mp →  -- SRP is 20% higher than MP
  paid = 0.6 * mp →  -- John paid 60% of MP (40% off)
  paid / srp = 0.5 :=  -- John paid 50% of SRP
by sorry

end NUMINAMATH_CALUDE_discount_percentage_l1133_113317


namespace NUMINAMATH_CALUDE_original_population_multiple_of_three_l1133_113308

theorem original_population_multiple_of_three (x y z : ℕ) 
  (h1 : y * y = x * x + 121)
  (h2 : z * z = y * y + 121) : 
  ∃ k : ℕ, x * x = 3 * k :=
sorry

end NUMINAMATH_CALUDE_original_population_multiple_of_three_l1133_113308


namespace NUMINAMATH_CALUDE_negation_equivalence_l1133_113379

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 > Real.exp x) ↔ (∀ x : ℝ, x^2 ≤ Real.exp x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1133_113379


namespace NUMINAMATH_CALUDE_water_current_speed_l1133_113343

/-- The speed of a water current given swimmer's speed and time against current -/
theorem water_current_speed (swimmer_speed : ℝ) (distance : ℝ) (time : ℝ) :
  swimmer_speed = 4 →
  distance = 5 →
  time = 2.5 →
  ∃ (current_speed : ℝ), 
    time = distance / (swimmer_speed - current_speed) ∧
    current_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_current_speed_l1133_113343


namespace NUMINAMATH_CALUDE_christopher_age_l1133_113326

theorem christopher_age (c g : ℕ) : 
  c = 2 * g →                  -- Christopher is 2 times as old as Gabriela now
  c - 9 = 5 * (g - 9) →        -- Nine years ago, Christopher was 5 times as old as Gabriela
  c = 24                       -- Christopher's current age is 24
  := by sorry

end NUMINAMATH_CALUDE_christopher_age_l1133_113326


namespace NUMINAMATH_CALUDE_book_sale_gain_percentage_l1133_113327

/-- Proves that given a book with a cost price CP, if selling it for 0.9 * CP
    results in Rs. 720, and selling it for Rs. 880 results in a gain,
    then the percentage of gain is 10%. -/
theorem book_sale_gain_percentage
  (CP : ℝ)  -- Cost price of the book
  (h1 : 0.9 * CP = 720)  -- Selling at 10% loss gives Rs. 720
  (h2 : 880 > CP)  -- Selling at Rs. 880 results in a gain
  : (880 - CP) / CP * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_gain_percentage_l1133_113327


namespace NUMINAMATH_CALUDE_tangent_identities_l1133_113386

theorem tangent_identities :
  (∃ (f : ℝ → ℝ), (∀ x, f x = Real.tan x) ∧
    (f (π / 7) * f (2 * π / 7) * f (3 * π / 7) = Real.sqrt 7) ∧
    (f (π / 7)^2 + f (2 * π / 7)^2 + f (3 * π / 7)^2 = 21)) :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_identities_l1133_113386


namespace NUMINAMATH_CALUDE_prob_same_color_is_89_169_l1133_113306

def num_blue_balls : ℕ := 8
def num_yellow_balls : ℕ := 5
def total_balls : ℕ := num_blue_balls + num_yellow_balls

def prob_same_color : ℚ :=
  (num_blue_balls / total_balls) ^ 2 + (num_yellow_balls / total_balls) ^ 2

theorem prob_same_color_is_89_169 :
  prob_same_color = 89 / 169 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_89_169_l1133_113306


namespace NUMINAMATH_CALUDE_complex_modulus_sum_l1133_113346

theorem complex_modulus_sum : Complex.abs (3 - 3*I) + Complex.abs (3 + 3*I) = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sum_l1133_113346


namespace NUMINAMATH_CALUDE_large_cups_sold_is_five_l1133_113359

/-- Represents the number of cups sold for each size --/
structure CupsSold where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total revenue based on the number of cups sold --/
def totalRevenue (cups : CupsSold) : ℕ :=
  cups.small + 2 * cups.medium + 3 * cups.large

theorem large_cups_sold_is_five :
  ∃ (cups : CupsSold),
    totalRevenue cups = 50 ∧
    cups.small = 11 ∧
    2 * cups.medium = 24 ∧
    cups.large = 5 := by
  sorry

end NUMINAMATH_CALUDE_large_cups_sold_is_five_l1133_113359


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1133_113367

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) is equal to 9/4. -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (3*n - 2) / (n * (n + 1) * (n + 3))) = 9/4 := by
  sorry


end NUMINAMATH_CALUDE_infinite_series_sum_l1133_113367


namespace NUMINAMATH_CALUDE_equation_solution_l1133_113365

theorem equation_solution : ∃ y : ℝ, y > 0 ∧ 6 * y^(1/4) - 3 * (y / y^(3/4)) = 12 + y^(1/4) ∧ y = 1296 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1133_113365


namespace NUMINAMATH_CALUDE_trajectory_and_constant_slope_l1133_113378

noncomputable section

-- Define the points A and P
def A : ℝ × ℝ := (3, -6)
def P : ℝ × ℝ := (1, -2)

-- Define the curve
def on_curve (Q : ℝ × ℝ) : Prop :=
  let (x, y) := Q
  (x^2 + y^2) / ((x - 3)^2 + (y + 6)^2) = 1/4

-- Define complementary angles
def complementary_angles (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

-- Define the theorem
theorem trajectory_and_constant_slope :
  -- Part 1: Equation of the curve
  (∀ Q : ℝ × ℝ, on_curve Q ↔ (Q.1 + 1)^2 + (Q.2 - 2)^2 = 20) ∧
  -- Part 2: Constant slope of BC
  (∀ B C : ℝ × ℝ, 
    on_curve B ∧ on_curve C ∧ 
    (∃ m1 m2 : ℝ, 
      complementary_angles m1 m2 ∧
      (B.2 - P.2) = m1 * (B.1 - P.1) ∧
      (C.2 - P.2) = m2 * (C.1 - P.1)) →
    (C.2 - B.2) / (C.1 - B.1) = -1/2) :=
sorry

end

end NUMINAMATH_CALUDE_trajectory_and_constant_slope_l1133_113378


namespace NUMINAMATH_CALUDE_max_salary_in_soccer_league_l1133_113377

/-- Represents a soccer team with salary constraints -/
structure SoccerTeam where
  numPlayers : ℕ
  minSalary : ℕ
  totalSalaryCap : ℕ

/-- Calculates the maximum possible salary for a single player in the team -/
def maxSinglePlayerSalary (team : SoccerTeam) : ℕ :=
  team.totalSalaryCap - (team.numPlayers - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player
    in a semi-professional soccer league with given constraints -/
theorem max_salary_in_soccer_league :
  let team : SoccerTeam := {
    numPlayers := 25,
    minSalary := 15000,
    totalSalaryCap := 850000
  }
  maxSinglePlayerSalary team = 490000 := by
  sorry

#eval maxSinglePlayerSalary {
  numPlayers := 25,
  minSalary := 15000,
  totalSalaryCap := 850000
}

end NUMINAMATH_CALUDE_max_salary_in_soccer_league_l1133_113377


namespace NUMINAMATH_CALUDE_unique_minimum_cost_plan_l1133_113300

/-- Represents a bus rental plan -/
structure BusRentalPlan where
  busA : ℕ  -- Number of Bus A
  busB : ℕ  -- Number of Bus B

/-- Checks if a bus rental plan is valid -/
def isValidPlan (p : BusRentalPlan) : Prop :=
  let totalPeople := 16 + 284
  let totalCapacity := 30 * p.busA + 42 * p.busB
  let totalCost := 300 * p.busA + 400 * p.busB
  let totalBuses := p.busA + p.busB
  totalCapacity ≥ totalPeople ∧
  totalCost ≤ 3100 ∧
  2 * totalBuses ≤ 16

/-- The set of all valid bus rental plans -/
def validPlans : Set BusRentalPlan :=
  {p : BusRentalPlan | isValidPlan p}

/-- The rental cost of a plan -/
def rentalCost (p : BusRentalPlan) : ℕ :=
  300 * p.busA + 400 * p.busB

theorem unique_minimum_cost_plan :
  ∃! p : BusRentalPlan, p ∈ validPlans ∧
    ∀ q ∈ validPlans, rentalCost p ≤ rentalCost q ∧
    rentalCost p = 2900 := by
  sorry

#check unique_minimum_cost_plan

end NUMINAMATH_CALUDE_unique_minimum_cost_plan_l1133_113300


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l1133_113391

theorem complex_magnitude_product : Complex.abs ((7 - 4*Complex.I) * (5 + 3*Complex.I)) = Real.sqrt 2210 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l1133_113391


namespace NUMINAMATH_CALUDE_num_event_committees_l1133_113390

/-- The number of teams in the tournament -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members selected from the host team -/
def host_selection : ℕ := 4

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 3

/-- The total number of members in the event committee -/
def committee_size : ℕ := 16

/-- Theorem stating the number of possible event committees -/
theorem num_event_committees : 
  (num_teams : ℕ) * (Nat.choose team_size host_selection) * 
  (Nat.choose team_size non_host_selection)^(num_teams - 1) = 3443073600 := by
  sorry

end NUMINAMATH_CALUDE_num_event_committees_l1133_113390


namespace NUMINAMATH_CALUDE_train_speed_crossing_bridge_l1133_113369

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 250) 
  (h2 : bridge_length = 150) 
  (h3 : crossing_time = 20) : 
  (train_length + bridge_length) / crossing_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_crossing_bridge_l1133_113369


namespace NUMINAMATH_CALUDE_right_triangle_tangent_l1133_113337

theorem right_triangle_tangent (n : ℕ) (a h : ℝ) (α : ℝ) :
  Odd n →
  0 < n →
  0 < a →
  0 < h →
  0 < α →
  α < π / 2 →
  Real.tan α = (4 * n * h) / ((n^2 - 1) * a) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_tangent_l1133_113337


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l1133_113342

theorem imaginary_part_of_one_minus_i_squared (i : ℂ) : 
  Complex.im ((1 - i)^2) = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l1133_113342


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1133_113315

theorem smallest_integer_satisfying_inequality : 
  ∃ x : ℤ, (2 * x : ℚ) / 5 + 3 / 4 > 7 / 5 ∧ 
  ∀ y : ℤ, y < x → (2 * y : ℚ) / 5 + 3 / 4 ≤ 7 / 5 :=
by
  use 2
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1133_113315


namespace NUMINAMATH_CALUDE_power_product_simplification_l1133_113364

theorem power_product_simplification :
  3000 * (3000 ^ 2999) = 3000 ^ 3000 := by sorry

end NUMINAMATH_CALUDE_power_product_simplification_l1133_113364


namespace NUMINAMATH_CALUDE_abs_seven_minus_sqrt_two_l1133_113368

theorem abs_seven_minus_sqrt_two (h : Real.sqrt 2 < 7) : 
  |7 - Real.sqrt 2| = 7 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_seven_minus_sqrt_two_l1133_113368


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l1133_113388

theorem chocolate_milk_probability : 
  let n : ℕ := 7  -- total number of days
  let k : ℕ := 5  -- number of days with chocolate milk
  let p : ℚ := 1/2  -- probability of bottling chocolate milk each day
  (n.choose k) * p^k * (1-p)^(n-k) = 21/128 := by
sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l1133_113388


namespace NUMINAMATH_CALUDE_correct_height_proof_l1133_113395

/-- Proves the correct height of a boy in a class given certain conditions -/
theorem correct_height_proof (n : ℕ) (initial_avg : ℝ) (wrong_height : ℝ) (actual_avg : ℝ) :
  n = 35 →
  initial_avg = 184 →
  wrong_height = 166 →
  actual_avg = 182 →
  ∃ (correct_height : ℝ), correct_height = 236 ∧
    n * actual_avg = n * initial_avg - wrong_height + correct_height :=
by sorry

end NUMINAMATH_CALUDE_correct_height_proof_l1133_113395


namespace NUMINAMATH_CALUDE_smallest_marble_collection_l1133_113307

theorem smallest_marble_collection (N : ℕ) : 
  N > 1 ∧ 
  N % 9 = 2 ∧ 
  N % 10 = 2 ∧ 
  N % 11 = 2 ∧ 
  (∀ m : ℕ, m > 1 ∧ m % 9 = 2 ∧ m % 10 = 2 ∧ m % 11 = 2 → m ≥ N) →
  N = 992 := by
sorry

end NUMINAMATH_CALUDE_smallest_marble_collection_l1133_113307


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1133_113399

/-- The distance between the foci of the ellipse 9x^2 + 36y^2 = 1296 is 12√3 -/
theorem ellipse_foci_distance (x y : ℝ) :
  (9 * x^2 + 36 * y^2 = 1296) →
  (∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = (12 * Real.sqrt 3)^2 ∧
    ∀ (p : ℝ × ℝ), 9 * p.1^2 + 36 * p.2^2 = 1296 →
      Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) +
      Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) =
      2 * Real.sqrt (144)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1133_113399


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1133_113387

theorem algebraic_expression_value (a b : ℝ) (h : a = b + 1) :
  a^2 - 2*a*b + b^2 + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1133_113387


namespace NUMINAMATH_CALUDE_melanie_plums_theorem_l1133_113322

/-- Represents the number of plums Melanie has -/
def plums_remaining (initial_plums : ℕ) (plums_given_away : ℕ) : ℕ :=
  initial_plums - plums_given_away

/-- Theorem stating that Melanie's remaining plums are correctly calculated -/
theorem melanie_plums_theorem (initial_plums : ℕ) (plums_given_away : ℕ) 
  (h : initial_plums ≥ plums_given_away) :
  plums_remaining initial_plums plums_given_away = initial_plums - plums_given_away :=
by sorry

end NUMINAMATH_CALUDE_melanie_plums_theorem_l1133_113322


namespace NUMINAMATH_CALUDE_power_sum_equality_l1133_113332

theorem power_sum_equality : 2^567 + 9^5 / 3^2 = 2^567 + 6561 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l1133_113332


namespace NUMINAMATH_CALUDE_sin_240_degrees_l1133_113311

theorem sin_240_degrees : Real.sin (240 * π / 180) = -(1/2) := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l1133_113311


namespace NUMINAMATH_CALUDE_jason_grass_cutting_time_l1133_113338

/-- The time Jason spends cutting grass over a weekend -/
def time_cutting_grass (time_per_lawn : ℕ) (lawns_per_day : ℕ) (days : ℕ) : ℕ :=
  time_per_lawn * lawns_per_day * days

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℕ) : ℕ :=
  minutes / 60

theorem jason_grass_cutting_time :
  let time_per_lawn := 30
  let lawns_per_day := 8
  let days := 2
  minutes_to_hours (time_cutting_grass time_per_lawn lawns_per_day days) = 8 := by
  sorry

end NUMINAMATH_CALUDE_jason_grass_cutting_time_l1133_113338


namespace NUMINAMATH_CALUDE_needle_cylinder_height_gt_six_l1133_113319

/-- Represents the properties of a cylinder formed by needles piercing a skein of yarn -/
structure NeedleCylinder where
  num_needles : ℕ
  needle_radius : ℝ
  cylinder_radius : ℝ

/-- The theorem stating that the height of the cylinder must be greater than 6 -/
theorem needle_cylinder_height_gt_six (nc : NeedleCylinder)
  (h_num_needles : nc.num_needles = 72)
  (h_needle_radius : nc.needle_radius = 1)
  (h_cylinder_radius : nc.cylinder_radius = 6) :
  ∀ h : ℝ, h > 6 → 
    2 * π * nc.cylinder_radius^2 + 2 * π * nc.cylinder_radius * h > 
    2 * π * nc.num_needles * nc.needle_radius^2 + 2 * π * nc.cylinder_radius^2 :=
by sorry

end NUMINAMATH_CALUDE_needle_cylinder_height_gt_six_l1133_113319


namespace NUMINAMATH_CALUDE_problem_solution_l1133_113355

theorem problem_solution : 
  (∃ n : ℕ, 140 * 5 = n * 100 ∧ n % 10 ≠ 0) ∧ 
  (4 * 150 - 7 = 593) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1133_113355


namespace NUMINAMATH_CALUDE_det_transformation_indeterminate_l1133_113348

variable (a b c d : ℝ)

def det2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem det_transformation_indeterminate :
  det2 a b c d = 5 →
  ∃ (x y : ℝ), x ≠ y ∧
    det2 (3*a + 1) (3*b + 1) (3*c + 1) (3*d + 1) = x ∧
    det2 (3*a + 1) (3*b + 1) (3*c + 1) (3*d + 1) = y :=
by sorry

end NUMINAMATH_CALUDE_det_transformation_indeterminate_l1133_113348


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1133_113339

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (3*x - 2)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = -63 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1133_113339


namespace NUMINAMATH_CALUDE_rectangle_division_l1133_113361

/-- If a rectangle with an area of 59.6 square centimeters is divided into 4 equal parts, 
    then the area of one part is 14.9 square centimeters. -/
theorem rectangle_division (total_area : ℝ) (num_parts : ℕ) (area_of_part : ℝ) : 
  total_area = 59.6 → 
  num_parts = 4 → 
  area_of_part = total_area / num_parts → 
  area_of_part = 14.9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_division_l1133_113361


namespace NUMINAMATH_CALUDE_existence_of_equal_elements_l1133_113389

theorem existence_of_equal_elements
  (p q n : ℕ+)
  (h_sum : p + q < n)
  (x : Fin (n + 1) → ℤ)
  (h_boundary : x 0 = 0 ∧ x (Fin.last n) = 0)
  (h_diff : ∀ i : Fin n, x i.succ - x i = p ∨ x i.succ - x i = -q) :
  ∃ i j : Fin (n + 1), i < j ∧ (i, j) ≠ (0, Fin.last n) ∧ x i = x j :=
by sorry

end NUMINAMATH_CALUDE_existence_of_equal_elements_l1133_113389


namespace NUMINAMATH_CALUDE_fixed_point_on_chord_min_distance_perpendicular_chords_l1133_113394

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line l
def line_l (x y : ℝ) : Prop := y = -1

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define a point on line l
structure PointOnLineL where
  x : ℝ
  y : ℝ
  on_line : line_l x y

-- Define the chord of tangent points
def chord_of_tangent_points (P : PointOnLineL) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ (A B : PointOnParabola), 
    (A.x * P.x = 2 * (A.y - 1)) ∧ 
    (B.x * P.x = 2 * (B.y - 1)) ∧ 
    (P.x * x = 2 * (y - 1))}

-- Theorem 1: The chord of tangent points passes through (0, 1)
theorem fixed_point_on_chord (P : PointOnLineL) :
  (0, 1) ∈ chord_of_tangent_points P :=
sorry

-- Theorem 2: Minimum distance between P and Q when chords are perpendicular
theorem min_distance_perpendicular_chords :
  ∃ (P Q : PointOnLineL),
    (P.x * Q.x = -4) →
    (∀ (P' Q' : PointOnLineL), (P'.x * Q'.x = -4) → 
      (P'.x - Q'.x)^2 ≥ (P.x - Q.x)^2) ∧
    P.x = -2 ∧ Q.x = 2 ∧ (P.x - Q.x)^2 = 16 :=
sorry

end NUMINAMATH_CALUDE_fixed_point_on_chord_min_distance_perpendicular_chords_l1133_113394


namespace NUMINAMATH_CALUDE_trapezoid_solutions_l1133_113328

def is_trapezoid_solution (b₁ b₂ : ℕ) : Prop :=
  b₁ % 8 = 0 ∧ b₂ % 8 = 0 ∧ (b₁ + b₂) * 50 / 2 = 1400 ∧ b₁ > 0 ∧ b₂ > 0

theorem trapezoid_solutions :
  ∃! (solutions : List (ℕ × ℕ)), solutions.length = 3 ∧
    ∀ (b₁ b₂ : ℕ), (b₁, b₂) ∈ solutions ↔ is_trapezoid_solution b₁ b₂ :=
sorry

end NUMINAMATH_CALUDE_trapezoid_solutions_l1133_113328


namespace NUMINAMATH_CALUDE_cylinder_height_relation_l1133_113376

theorem cylinder_height_relation :
  ∀ (r₁ h₁ r₂ h₂ : ℝ),
  r₁ > 0 → h₁ > 0 → r₂ > 0 → h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_relation_l1133_113376


namespace NUMINAMATH_CALUDE_laptop_sale_price_l1133_113330

theorem laptop_sale_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price = 600 ∧ discount1 = 0.25 ∧ discount2 = 0.10 →
  (original_price * (1 - discount1) * (1 - discount2)) / original_price = 0.675 := by
sorry

end NUMINAMATH_CALUDE_laptop_sale_price_l1133_113330


namespace NUMINAMATH_CALUDE_max_single_player_salary_l1133_113335

/-- Represents the number of players in a team -/
def num_players : ℕ := 18

/-- Represents the minimum salary for each player in dollars -/
def min_salary : ℕ := 20000

/-- Represents the maximum total salary for the team in dollars -/
def max_total_salary : ℕ := 800000

/-- Theorem stating the maximum possible salary for a single player -/
theorem max_single_player_salary :
  ∃ (max_salary : ℕ),
    max_salary = 460000 ∧
    max_salary + (num_players - 1) * min_salary = max_total_salary ∧
    ∀ (salary : ℕ),
      salary + (num_players - 1) * min_salary ≤ max_total_salary →
      salary ≤ max_salary :=
by sorry

end NUMINAMATH_CALUDE_max_single_player_salary_l1133_113335


namespace NUMINAMATH_CALUDE_square_area_problem_l1133_113303

theorem square_area_problem (s : ℝ) : 
  (0.8 * s) * (5 * s) = s^2 + 15.18 → s^2 = 5.06 := by sorry

end NUMINAMATH_CALUDE_square_area_problem_l1133_113303


namespace NUMINAMATH_CALUDE_simplify_sqrt_quadratic_l1133_113333

theorem simplify_sqrt_quadratic (x : ℝ) (h : x < 2) : 
  Real.sqrt (x^2 - 4*x + 4) = 2 - x := by
sorry

end NUMINAMATH_CALUDE_simplify_sqrt_quadratic_l1133_113333


namespace NUMINAMATH_CALUDE_specific_arc_rectangle_boundary_l1133_113318

/-- Represents a rectangle with quarter-circle arcs on its corners -/
structure ArcRectangle where
  area : ℝ
  length_width_ratio : ℝ
  divisions : ℕ

/-- Calculates the boundary length of the ArcRectangle -/
def boundary_length (r : ArcRectangle) : ℝ :=
  sorry

/-- Theorem stating the boundary length of a specific ArcRectangle -/
theorem specific_arc_rectangle_boundary :
  let r : ArcRectangle := { area := 72, length_width_ratio := 2, divisions := 3 }
  boundary_length r = 4 * Real.pi + 24 := by
  sorry

end NUMINAMATH_CALUDE_specific_arc_rectangle_boundary_l1133_113318


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1133_113356

-- Problem 1
theorem problem_1 (x y : ℝ) : (x - y)^3 * (y - x)^2 = (x - y)^5 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (-3 * a^2)^3 = -27 * a^6 := by sorry

-- Problem 3
theorem problem_3 (x : ℝ) (h : x ≠ 0) : x^10 / (2*x)^2 = x^8 / 4 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1133_113356


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_theorem_l1133_113312

/-- Represents an isosceles trapezoid with inscribed and circumscribed circles. -/
structure IsoscelesTrapezoid where
  r : ℝ  -- radius of inscribed circle
  R : ℝ  -- radius of circumscribed circle
  k : ℝ  -- ratio of R to r
  h_k_def : k = R / r
  h_k_pos : k > 0

/-- The angles and permissible k values for an isosceles trapezoid. -/
def trapezoid_properties (t : IsoscelesTrapezoid) : Prop :=
  let angle := Real.arcsin (1 / t.k * Real.sqrt ((1 + Real.sqrt (1 + 4 * t.k ^ 2)) / 2))
  (∀ θ, θ = angle ∨ θ = Real.pi - angle → 
    θ.cos * t.r = t.r ∧ θ.sin * t.R = t.R / 2) ∧ 
  t.k > Real.sqrt 2

/-- Main theorem about isosceles trapezoid properties. -/
theorem isosceles_trapezoid_theorem (t : IsoscelesTrapezoid) : 
  trapezoid_properties t := by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_theorem_l1133_113312


namespace NUMINAMATH_CALUDE_corrected_mean_l1133_113374

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) :
  n = 20 ∧ original_mean = 36 ∧ incorrect_value = 40 ∧ correct_value = 25 →
  (n * original_mean - (incorrect_value - correct_value)) / n = 35.25 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l1133_113374


namespace NUMINAMATH_CALUDE_coin_game_winning_strategy_l1133_113324

/-- Represents the state of the coin game -/
structure GameState where
  piles : List Nat
  deriving Repr

/-- Checks if a player has a winning strategy given the current game state -/
def hasWinningStrategy (state : GameState) : Prop :=
  let n1 := (state.piles.filter (· = 1)).length
  let evenPiles := state.piles.filter (· % 2 = 0)
  let sumEvenPiles := (evenPiles.map (λ x => x / 2)).sum
  Odd n1 ∨ Odd sumEvenPiles

/-- The main theorem stating the winning condition for the coin game -/
theorem coin_game_winning_strategy (state : GameState) :
  hasWinningStrategy state ↔ 
  Odd (state.piles.filter (· = 1)).length ∨ 
  Odd ((state.piles.filter (· % 2 = 0)).map (λ x => x / 2)).sum :=
by sorry


end NUMINAMATH_CALUDE_coin_game_winning_strategy_l1133_113324


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l1133_113349

-- Define propositions P and Q
def P (a b : ℝ) : Prop := a > b ∧ b > 0
def Q (a b : ℝ) : Prop := a^2 > b^2

-- Theorem stating that P is sufficient but not necessary for Q
theorem P_sufficient_not_necessary_for_Q :
  (∀ a b : ℝ, P a b → Q a b) ∧
  ¬(∀ a b : ℝ, Q a b → P a b) :=
sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l1133_113349


namespace NUMINAMATH_CALUDE_smallest_non_special_number_twenty_two_is_non_special_l1133_113352

def triangle_number (k : ℕ) : ℕ := k * (k + 1) / 2

def is_prime_power (n : ℕ) : Prop :=
  ∃ (p k : ℕ), p.Prime ∧ k > 0 ∧ n = p ^ k

def is_prime_plus_one (n : ℕ) : Prop :=
  ∃ p : ℕ, p.Prime ∧ n = p + 1

theorem smallest_non_special_number :
  ∀ n : ℕ, n < 22 →
    (∃ k : ℕ, n = triangle_number k) ∨
    is_prime_power n ∨
    is_prime_plus_one n :=
  sorry

theorem twenty_two_is_non_special :
  ¬(∃ k : ℕ, 22 = triangle_number k) ∧
  ¬is_prime_power 22 ∧
  ¬is_prime_plus_one 22 :=
  sorry

end NUMINAMATH_CALUDE_smallest_non_special_number_twenty_two_is_non_special_l1133_113352


namespace NUMINAMATH_CALUDE_log_identity_l1133_113302

-- Define the logarithm base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem log_identity : (log5 (3 * log5 25))^2 = (1 + log5 1.2)^2 := by sorry

end NUMINAMATH_CALUDE_log_identity_l1133_113302


namespace NUMINAMATH_CALUDE_fruits_remaining_l1133_113383

theorem fruits_remaining (initial_apples : ℕ) (plum_ratio : ℚ) (picked_ratio : ℚ) : 
  initial_apples = 180 → 
  plum_ratio = 1 / 3 → 
  picked_ratio = 3 / 5 → 
  (initial_apples + (↑initial_apples * plum_ratio)) * (1 - picked_ratio) = 96 := by
sorry

end NUMINAMATH_CALUDE_fruits_remaining_l1133_113383


namespace NUMINAMATH_CALUDE_custom_mult_eleven_twelve_l1133_113321

/-- Custom multiplication operation for integers -/
def custom_mult (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem stating that for y = 11, y * 12 = 110 under the custom multiplication -/
theorem custom_mult_eleven_twelve :
  let y : ℤ := 11
  custom_mult y 12 = 110 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_eleven_twelve_l1133_113321


namespace NUMINAMATH_CALUDE_simplify_fraction_l1133_113310

theorem simplify_fraction (k : ℝ) : 
  let expression := (6 * k + 12) / 6
  ∃ (c d : ℤ), expression = c * k + d ∧ (c : ℚ) / d = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1133_113310


namespace NUMINAMATH_CALUDE_min_value_a_plus_9b_l1133_113316

theorem min_value_a_plus_9b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_arith_seq : (1/a + 1/b) / 2 = 1/2) : 
  ∀ x y : ℝ, x > 0 → y > 0 → (1/x + 1/y) / 2 = 1/2 → x + 9*y ≥ 16 ∧ 
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ (1/x₀ + 1/y₀) / 2 = 1/2 ∧ x₀ + 9*y₀ = 16 :=
by sorry

#check min_value_a_plus_9b

end NUMINAMATH_CALUDE_min_value_a_plus_9b_l1133_113316


namespace NUMINAMATH_CALUDE_cos_150_degrees_l1133_113334

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l1133_113334


namespace NUMINAMATH_CALUDE_additional_daily_intake_l1133_113344

/-- Proves that given a total milk consumption goal and a time frame, 
    the additional daily intake required can be calculated. -/
theorem additional_daily_intake 
  (total_milk : ℝ) 
  (weeks : ℝ) 
  (current_daily : ℝ) 
  (h1 : total_milk = 105) 
  (h2 : weeks = 3) 
  (h3 : current_daily = 3) : 
  ∃ (additional : ℝ), 
    additional = (total_milk / (weeks * 7)) - current_daily ∧ 
    additional = 2 := by
  sorry

end NUMINAMATH_CALUDE_additional_daily_intake_l1133_113344


namespace NUMINAMATH_CALUDE_sergio_fruit_sales_l1133_113305

/-- Calculates the total amount of money earned from fruit sales given the production of mangoes -/
def totalFruitSales (mangoProduction : ℕ) : ℕ :=
  let appleProduction := 2 * mangoProduction
  let orangeProduction := mangoProduction + 200
  let totalProduction := appleProduction + mangoProduction + orangeProduction
  totalProduction * 50

/-- Theorem stating that given the conditions, Mr. Sergio's total sales amount to $90,000 -/
theorem sergio_fruit_sales : totalFruitSales 400 = 90000 := by
  sorry

end NUMINAMATH_CALUDE_sergio_fruit_sales_l1133_113305


namespace NUMINAMATH_CALUDE_unique_k_divisibility_l1133_113385

theorem unique_k_divisibility (a b l : ℕ) (ha : a > 1) (hb : b > 1) 
  (hodd_a : Odd a) (hodd_b : Odd b) (hsum : a + b = 2^l) :
  ∀ k : ℕ, k > 0 → (k^2 ∣ a^k + b^k) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_divisibility_l1133_113385


namespace NUMINAMATH_CALUDE_prob_exactly_one_hit_prob_distribution_X_expected_value_X_l1133_113382

-- Define the probabilities and scores
def prob_A_hit : ℝ := 0.8
def prob_B_hit : ℝ := 0.5
def score_A_hit : ℕ := 5
def score_B_hit : ℕ := 10

-- Define the random variable X for the total score
def X : ℕ → ℝ
| 0 => (1 - prob_A_hit)^2 * (1 - prob_B_hit)
| 5 => 2 * prob_A_hit * (1 - prob_A_hit) * (1 - prob_B_hit)
| 10 => prob_A_hit^2 * (1 - prob_B_hit) + (1 - prob_A_hit)^2 * prob_B_hit
| 15 => 2 * prob_A_hit * (1 - prob_A_hit) * prob_B_hit
| 20 => prob_A_hit^2 * prob_B_hit
| _ => 0

-- Theorem for the probability of exactly one hit
theorem prob_exactly_one_hit : 
  2 * prob_A_hit * (1 - prob_A_hit) * (1 - prob_B_hit) + (1 - prob_A_hit)^2 * prob_B_hit = 0.18 := 
by sorry

-- Theorem for the probability distribution of X
theorem prob_distribution_X : 
  X 0 = 0.02 ∧ X 5 = 0.16 ∧ X 10 = 0.34 ∧ X 15 = 0.16 ∧ X 20 = 0.32 := 
by sorry

-- Theorem for the expected value of X
theorem expected_value_X : 
  0 * X 0 + 5 * X 5 + 10 * X 10 + 15 * X 15 + 20 * X 20 = 13.0 := 
by sorry

end NUMINAMATH_CALUDE_prob_exactly_one_hit_prob_distribution_X_expected_value_X_l1133_113382


namespace NUMINAMATH_CALUDE_painter_problem_l1133_113336

theorem painter_problem (total_rooms : ℕ) (time_per_room : ℕ) (time_left : ℕ) 
  (h1 : total_rooms = 11)
  (h2 : time_per_room = 7)
  (h3 : time_left = 63) :
  total_rooms - (time_left / time_per_room) = 2 := by
  sorry

end NUMINAMATH_CALUDE_painter_problem_l1133_113336


namespace NUMINAMATH_CALUDE_elder_age_l1133_113314

/-- Given two persons with an age difference of 20 years, where 4 years ago
    the elder was 5 times as old as the younger, the present age of the elder person is 29 years. -/
theorem elder_age (y e : ℕ) : 
  e = y + 20 →                   -- The ages differ by 20 years
  e - 4 = 5 * (y - 4) →          -- 4 years ago, elder was 5 times younger's age
  e = 29                         -- Elder's present age is 29
  := by sorry

end NUMINAMATH_CALUDE_elder_age_l1133_113314


namespace NUMINAMATH_CALUDE_equal_fractions_sum_l1133_113362

theorem equal_fractions_sum (n : ℕ) (sum : ℚ) (fraction : ℚ) :
  n = 450 →
  sum = 1 / 12 →
  n * fraction = sum →
  fraction = 1 / 5400 := by
  sorry

end NUMINAMATH_CALUDE_equal_fractions_sum_l1133_113362


namespace NUMINAMATH_CALUDE_area_triangle_with_four_circles_l1133_113313

/-- The area of an equilateral triangle containing four unit circles --/
theorem area_triangle_with_four_circles : 
  ∀ (side_length : ℝ),
  (∃ (r : ℝ), r = 1 ∧ 
   side_length = 4 * (r * Real.sqrt 3)) →
  (3 / 4 : ℝ) * Real.sqrt 3 * side_length^2 = 12 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_area_triangle_with_four_circles_l1133_113313


namespace NUMINAMATH_CALUDE_function_properties_l1133_113354

-- Define the function f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem function_properties :
  ∀ (a b c : ℝ),
  (f' a b 1 = 3) →  -- Tangent line condition
  (f a b c 1 = 2) →  -- Point condition
  (f' a b (-2) = 0) →  -- Extreme value condition
  (a = 2 ∧ b = -4 ∧ c = 5) ∧  -- Correct values of a, b, c
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, f' 2 (-4) x ≥ 0)  -- Monotonically increasing condition
  :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1133_113354


namespace NUMINAMATH_CALUDE_zeros_and_remainder_l1133_113396

def factorial_product : ℕ := (List.range 50).foldl (λ acc n => acc * (n + 1).factorial) 1

theorem zeros_and_remainder : 
  (∃ m : ℕ, factorial_product = m * (10^12)) ∧ 
  ¬(∃ m : ℕ, factorial_product = m * (10^13)) ∧
  12 % 500 = 12 := by
  sorry

end NUMINAMATH_CALUDE_zeros_and_remainder_l1133_113396


namespace NUMINAMATH_CALUDE_divisible_by_38_count_l1133_113372

def numbers : List Nat := [3624, 36024, 360924, 3609924, 36099924, 360999924, 3609999924]

theorem divisible_by_38_count :
  (numbers.filter (·.mod 38 = 0)).length = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_38_count_l1133_113372


namespace NUMINAMATH_CALUDE_impossible_to_use_all_parts_l1133_113398

theorem impossible_to_use_all_parts (p q r : ℕ) : 
  ¬∃ (x y z : ℕ), 
    (2 * x + 2 * z = 2 * p + 2 * r + 2) ∧ 
    (2 * x + y = 2 * p + q + 1) ∧ 
    (y + z = q + r) :=
by sorry

end NUMINAMATH_CALUDE_impossible_to_use_all_parts_l1133_113398


namespace NUMINAMATH_CALUDE_no_six_if_mean_and_median_two_l1133_113345

/-- Represents the result of 5 dice rolls -/
def DiceRolls := Fin 5 → Nat

/-- The mean of the dice rolls is 2 -/
def mean_is_2 (rolls : DiceRolls) : Prop :=
  (rolls 0 + rolls 1 + rolls 2 + rolls 3 + rolls 4) / 5 = 2

/-- The median of the dice rolls is 2 -/
def median_is_2 (rolls : DiceRolls) : Prop :=
  ∃ (p : Equiv (Fin 5) (Fin 5)), 
    rolls (p 2) = 2 ∧ 
    (∀ i < 2, rolls (p i) ≤ 2) ∧ 
    (∀ i > 2, rolls (p i) ≥ 2)

/-- The theorem stating that if the mean and median are 2, then 6 cannot appear in the rolls -/
theorem no_six_if_mean_and_median_two (rolls : DiceRolls) 
  (h_mean : mean_is_2 rolls) (h_median : median_is_2 rolls) : 
  ∀ i, rolls i ≠ 6 := by
  sorry

end NUMINAMATH_CALUDE_no_six_if_mean_and_median_two_l1133_113345


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1133_113301

open Real

theorem solution_set_of_inequality (f : ℝ → ℝ) (f_diff : Differentiable ℝ f) :
  (f 0 = 2) →
  (∀ x, f x + deriv f x > 1) →
  (∀ x, (exp x * f x > exp x + 1) ↔ x > 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1133_113301


namespace NUMINAMATH_CALUDE_solution_to_equation_l1133_113353

theorem solution_to_equation (x y : ℝ) :
  4 * x^2 * y^2 = 4 * x * y + 3 ↔ y = 3 / (2 * x) ∨ y = -1 / (2 * x) :=
sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1133_113353


namespace NUMINAMATH_CALUDE_batch_composition_l1133_113325

/-- Represents the characteristics of a product type -/
structure ProductType where
  volume : ℝ  -- Volume per unit in m³
  mass : ℝ    -- Mass per unit in tons

/-- Represents a batch of products -/
structure Batch where
  typeA : ProductType
  typeB : ProductType
  totalVolume : ℝ
  totalMass : ℝ

/-- Theorem: Given the specific product characteristics and total volume and mass,
    prove that the batch consists of 5 units of type A and 8 units of type B -/
theorem batch_composition (b : Batch)
    (h1 : b.typeA.volume = 0.8)
    (h2 : b.typeA.mass = 0.5)
    (h3 : b.typeB.volume = 2)
    (h4 : b.typeB.mass = 1)
    (h5 : b.totalVolume = 20)
    (h6 : b.totalMass = 10.5) :
    ∃ (x y : ℝ), x = 5 ∧ y = 8 ∧
    x * b.typeA.volume + y * b.typeB.volume = b.totalVolume ∧
    x * b.typeA.mass + y * b.typeB.mass = b.totalMass :=
  sorry


end NUMINAMATH_CALUDE_batch_composition_l1133_113325


namespace NUMINAMATH_CALUDE_lcm_gcd_equation_solutions_l1133_113329

def solution_pairs : List (Nat × Nat) := [(3, 6), (4, 6), (4, 4), (6, 4), (6, 3)]

theorem lcm_gcd_equation_solutions :
  ∀ a b : Nat,
    a > 0 ∧ b > 0 →
    (Nat.lcm a b + Nat.gcd a b + a + b = a * b) ↔ (a, b) ∈ solution_pairs := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_equation_solutions_l1133_113329


namespace NUMINAMATH_CALUDE_equal_numbers_product_l1133_113340

theorem equal_numbers_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 15 →
  a = 10 →
  b = 18 →
  c = d →
  c * d = 256 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l1133_113340


namespace NUMINAMATH_CALUDE_conic_common_chords_l1133_113331

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Conic where
  equation : ℝ → ℝ → ℝ

-- Define the problem setup
def are_tangent (c1 c2 : Conic) (p1 p2 : Point) : Prop := sorry

def have_common_points (c1 c2 : Conic) (n : ℕ) : Prop := sorry

def line_through_points (p1 p2 : Point) : Line := sorry

def intersection_point (l1 l2 : Line) : Point := sorry

def common_chord (c1 c2 : Conic) : Line := sorry

def passes_through (l : Line) (p : Point) : Prop := sorry

-- State the theorem
theorem conic_common_chords 
  (Γ Γ₁ Γ₂ : Conic) 
  (A B C D : Point) :
  are_tangent Γ Γ₁ A B →
  are_tangent Γ Γ₂ C D →
  have_common_points Γ₁ Γ₂ 4 →
  ∃ (chord1 chord2 : Line),
    chord1 = common_chord Γ₁ Γ₂ ∧
    chord2 = common_chord Γ₁ Γ₂ ∧
    chord1 ≠ chord2 ∧
    passes_through chord1 (intersection_point (line_through_points A B) (line_through_points C D)) ∧
    passes_through chord2 (intersection_point (line_through_points A B) (line_through_points C D)) :=
by sorry

end NUMINAMATH_CALUDE_conic_common_chords_l1133_113331


namespace NUMINAMATH_CALUDE_fish_population_estimate_l1133_113384

/-- The number of fish tagged on April 1 -/
def tagged_april : ℕ := 120

/-- The number of fish captured on August 1 -/
def captured_august : ℕ := 150

/-- The number of tagged fish found in the August 1 sample -/
def tagged_in_august : ℕ := 5

/-- The proportion of fish that left the pond between April 1 and August 1 -/
def left_pond : ℚ := 3/10

/-- The proportion of fish in the August sample that were not in the pond in April -/
def new_fish_proportion : ℚ := 1/2

/-- The estimated number of fish in the pond on April 1 -/
def fish_population : ℕ := 1800

/-- Theorem stating that given the conditions, the fish population on April 1 was 1800 -/
theorem fish_population_estimate :
  tagged_april = 120 →
  captured_august = 150 →
  tagged_in_august = 5 →
  left_pond = 3/10 →
  new_fish_proportion = 1/2 →
  fish_population = 1800 := by
  sorry

end NUMINAMATH_CALUDE_fish_population_estimate_l1133_113384


namespace NUMINAMATH_CALUDE_xf_is_even_l1133_113366

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

-- Theorem statement
theorem xf_is_even (f : ℝ → ℝ) (h : OddFunction f) :
  EvenFunction (fun x ↦ x * f x) := by
  sorry

end NUMINAMATH_CALUDE_xf_is_even_l1133_113366


namespace NUMINAMATH_CALUDE_downstream_speed_l1133_113304

/-- Represents the speed of a rower in different conditions -/
structure RowerSpeed where
  upstream : ℝ
  still_water : ℝ
  downstream : ℝ

/-- 
Given a rower's speed upstream and in still water, 
calculates and proves the rower's speed downstream
-/
theorem downstream_speed (r : RowerSpeed) 
  (h_upstream : r.upstream = 35)
  (h_still : r.still_water = 40) :
  r.downstream = 45 := by
  sorry

#check downstream_speed

end NUMINAMATH_CALUDE_downstream_speed_l1133_113304


namespace NUMINAMATH_CALUDE_min_value_expression_l1133_113347

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x + 1/y + 1) * (x + 1/y - 2022) + (y + 1/x + 1) * (y + 1/x - 2022) ≥ -2048042 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1133_113347


namespace NUMINAMATH_CALUDE_cat_monitored_area_percentage_l1133_113341

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangular room -/
structure Room where
  width : ℝ
  height : ℝ

/-- Calculates the area of a room -/
def roomArea (r : Room) : ℝ := r.width * r.height

/-- Calculates the area that a cat can monitor in a room -/
noncomputable def monitoredArea (r : Room) (catPosition : Point) : ℝ := sorry

/-- Theorem stating that a cat at (3, 8) in a 10x8 room monitors 66.875% of the area -/
theorem cat_monitored_area_percentage (r : Room) (catPos : Point) :
  r.width = 10 ∧ r.height = 8 ∧ catPos.x = 3 ∧ catPos.y = 8 →
  monitoredArea r catPos / roomArea r = 66.875 / 100 := by sorry

end NUMINAMATH_CALUDE_cat_monitored_area_percentage_l1133_113341


namespace NUMINAMATH_CALUDE_sum_positive_from_inequality_l1133_113350

theorem sum_positive_from_inequality (x y : ℝ) 
  (h : (3:ℝ)^x + (5:ℝ)^y > (3:ℝ)^(-y) + (5:ℝ)^(-x)) : 
  x + y > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_from_inequality_l1133_113350


namespace NUMINAMATH_CALUDE_sequence_general_term_l1133_113392

theorem sequence_general_term (n : ℕ) (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ k, S k = 2 * k^2 - 3 * k) →
  (∀ k, k ≥ 1 → a k = S k - S (k - 1)) →
  (∀ k, a k = 4 * k - 5) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1133_113392


namespace NUMINAMATH_CALUDE_tom_initial_investment_l1133_113320

/-- Represents the initial investment of Tom in rupees -/
def tom_investment : ℝ := sorry

/-- Represents Jose's investment in rupees -/
def jose_investment : ℝ := 45000

/-- Represents the total profit earned in rupees -/
def total_profit : ℝ := 45000

/-- Represents Jose's share of the profit in rupees -/
def jose_profit_share : ℝ := 25000

/-- Represents the number of months Tom's money was invested -/
def tom_investment_months : ℝ := 12

/-- Represents the number of months Jose's money was invested -/
def jose_investment_months : ℝ := 10

/-- Theorem stating that Tom's initial investment was 30000 rupees -/
theorem tom_initial_investment : 
  tom_investment = 30000 := by sorry

end NUMINAMATH_CALUDE_tom_initial_investment_l1133_113320


namespace NUMINAMATH_CALUDE_dress_discount_problem_l1133_113363

theorem dress_discount_problem (P : ℝ) (D : ℝ) : 
  P - 61.2 = 4.5 → P * (1 - D) * 1.25 = 61.2 → D = 0.255 := by
sorry

end NUMINAMATH_CALUDE_dress_discount_problem_l1133_113363


namespace NUMINAMATH_CALUDE_expression_value_l1133_113381

theorem expression_value (x y : ℝ) (h : x = 2*y + 1) : x^2 - 4*x*y + 4*y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1133_113381


namespace NUMINAMATH_CALUDE_vacation_duration_l1133_113370

theorem vacation_duration (plane_cost hotel_cost_per_day total_cost : ℕ) 
  (h1 : plane_cost = 48)
  (h2 : hotel_cost_per_day = 24)
  (h3 : total_cost = 120) :
  ∃ d : ℕ, d = 3 ∧ plane_cost + hotel_cost_per_day * d = total_cost := by
  sorry

end NUMINAMATH_CALUDE_vacation_duration_l1133_113370


namespace NUMINAMATH_CALUDE_race_distance_l1133_113373

theorem race_distance (p q : ℝ) (d : ℝ) : 
  p = 1.2 * q →  -- p is 20% faster than q
  d = q * (d + 50) / p →  -- race ends in a tie
  d + 50 = 300 :=  -- p runs 300 meters
by sorry

end NUMINAMATH_CALUDE_race_distance_l1133_113373


namespace NUMINAMATH_CALUDE_greatest_int_prime_abs_quadratic_l1133_113358

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def f (x : ℤ) : ℤ := |4*x^2 - 39*x + 21|

theorem greatest_int_prime_abs_quadratic : 
  ∀ x : ℤ, x > 8 → ¬(is_prime (f x).toNat) ∧ is_prime (f 8).toNat :=
sorry

end NUMINAMATH_CALUDE_greatest_int_prime_abs_quadratic_l1133_113358


namespace NUMINAMATH_CALUDE_square_pentagon_intersections_l1133_113360

/-- A square inscribed in a circle -/
structure InscribedSquare :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- A regular pentagon inscribed in a circle -/
structure InscribedPentagon :=
  (vertices : Fin 5 → ℝ × ℝ)

/-- Predicate to check if two polygons share a vertex -/
def ShareVertex (s : InscribedSquare) (p : InscribedPentagon) : Prop :=
  ∃ (i : Fin 4) (j : Fin 5), s.vertices i = p.vertices j

/-- The number of intersections between two polygons -/
def NumIntersections (s : InscribedSquare) (p : InscribedPentagon) : ℕ := sorry

/-- Theorem stating that a square and a regular pentagon inscribed in the same circle,
    not sharing any vertices, intersect at exactly 8 points -/
theorem square_pentagon_intersections
  (s : InscribedSquare) (p : InscribedPentagon)
  (h : ¬ ShareVertex s p) :
  NumIntersections s p = 8 :=
sorry

end NUMINAMATH_CALUDE_square_pentagon_intersections_l1133_113360


namespace NUMINAMATH_CALUDE_smallest_positive_integer_d_l1133_113371

theorem smallest_positive_integer_d : ∃ d : ℕ+, d = 4 ∧
  (∀ d' : ℕ+, d' < d →
    ¬∃ x y : ℝ, x^2 + y^2 = 100 ∧ y = 2*x + d' ∧ x^2 + y^2 = 100 * d') ∧
  ∃ x y : ℝ, x^2 + y^2 = 100 ∧ y = 2*x + d ∧ x^2 + y^2 = 100 * d :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_d_l1133_113371


namespace NUMINAMATH_CALUDE_cos_120_degrees_l1133_113309

theorem cos_120_degrees : Real.cos (2 * Real.pi / 3) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l1133_113309


namespace NUMINAMATH_CALUDE_percentage_problem_l1133_113351

theorem percentage_problem (p : ℝ) (h1 : 0.5 * 10 = p / 100 * 500 - 20) : p = 5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1133_113351


namespace NUMINAMATH_CALUDE_uncool_parents_count_l1133_113323

/-- Proves the number of students with uncool parents in a music class -/
theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ) 
  (h1 : total = 40)
  (h2 : cool_dads = 25)
  (h3 : cool_moms = 19)
  (h4 : both_cool = 8) :
  total - (cool_dads + cool_moms - both_cool) = 4 := by
  sorry

end NUMINAMATH_CALUDE_uncool_parents_count_l1133_113323


namespace NUMINAMATH_CALUDE_toothpicks_in_specific_grid_l1133_113393

/-- Calculates the number of toothpicks in a modified grid -/
def toothpicks_in_modified_grid (length width corner_size : ℕ) : ℕ :=
  let vertical_lines := length + 1
  let horizontal_lines := width + 1
  let corner_lines := corner_size + 1
  let total_without_corner := vertical_lines * width + horizontal_lines * length
  let corner_toothpicks := corner_lines * corner_size * 2
  total_without_corner - corner_toothpicks

/-- Theorem stating the number of toothpicks in the specific grid described in the problem -/
theorem toothpicks_in_specific_grid :
  toothpicks_in_modified_grid 70 45 5 = 6295 :=
by sorry

end NUMINAMATH_CALUDE_toothpicks_in_specific_grid_l1133_113393
