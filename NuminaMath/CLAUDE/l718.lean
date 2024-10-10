import Mathlib

namespace investment_problem_l718_71808

theorem investment_problem (total_interest : ℝ) (amount_at_11_percent : ℝ) :
  total_interest = 0.0975 →
  amount_at_11_percent = 3750 →
  ∃ (total_amount : ℝ) (amount_at_9_percent : ℝ),
    total_amount = amount_at_9_percent + amount_at_11_percent ∧
    0.09 * amount_at_9_percent + 0.11 * amount_at_11_percent = total_interest * total_amount ∧
    total_amount = 10000 :=
by sorry

end investment_problem_l718_71808


namespace f_of_f_zero_l718_71812

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 + 2 * x - 1

-- State the theorem
theorem f_of_f_zero : f (f 0) = 1 := by sorry

end f_of_f_zero_l718_71812


namespace apps_files_difference_l718_71887

/-- Represents the contents of Dave's phone -/
structure PhoneContents where
  apps : ℕ
  files : ℕ

/-- The initial state of Dave's phone -/
def initial : PhoneContents := { apps := 24, files := 9 }

/-- The final state of Dave's phone -/
def final : PhoneContents := { apps := 12, files := 5 }

/-- The theorem stating the difference between apps and files in the final state -/
theorem apps_files_difference : final.apps - final.files = 7 := by
  sorry

end apps_files_difference_l718_71887


namespace pipe_laying_efficiency_l718_71873

theorem pipe_laying_efficiency 
  (n : ℕ) 
  (sequential_length : ℝ) 
  (h1 : n = 7) 
  (h2 : sequential_length = 60) :
  let individual_work_time := sequential_length / (6 * n)
  let total_time := n * individual_work_time
  let simultaneous_rate := n * (sequential_length / total_time)
  simultaneous_rate * total_time = 130 := by
sorry

end pipe_laying_efficiency_l718_71873


namespace unit_vectors_equal_squared_magnitude_l718_71818

/-- Two unit vectors in a plane have equal squared magnitudes. -/
theorem unit_vectors_equal_squared_magnitude
  (e₁ e₂ : ℝ × ℝ)
  (h₁ : ‖e₁‖ = 1)
  (h₂ : ‖e₂‖ = 1) :
  ‖e₁‖^2 = ‖e₂‖^2 := by
  sorry

end unit_vectors_equal_squared_magnitude_l718_71818


namespace cos_difference_from_sum_of_sin_and_cos_l718_71810

theorem cos_difference_from_sum_of_sin_and_cos 
  (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1/2) 
  (h2 : Real.cos A + Real.cos B = 5/4) : 
  Real.cos (A - B) = 13/32 := by
  sorry

end cos_difference_from_sum_of_sin_and_cos_l718_71810


namespace magnification_factor_l718_71876

theorem magnification_factor (magnified_diameter actual_diameter : ℝ) 
  (h1 : magnified_diameter = 0.2)
  (h2 : actual_diameter = 0.0002) :
  magnified_diameter / actual_diameter = 1000 := by
sorry

end magnification_factor_l718_71876


namespace sin_equality_l718_71879

theorem sin_equality (x : ℝ) (h : Real.sin (x + π/4) = 1/3) :
  Real.sin (4*x) - 2 * Real.cos (3*x) * Real.sin x = -7/9 := by
  sorry

end sin_equality_l718_71879


namespace x_value_after_z_doubled_l718_71867

theorem x_value_after_z_doubled (x y z_original z_doubled : ℚ) : 
  x = (1 / 3) * y →
  y = (1 / 4) * z_doubled →
  z_original = 48 →
  z_doubled = 2 * z_original →
  x = 8 := by sorry

end x_value_after_z_doubled_l718_71867


namespace complex_square_simplification_l718_71850

theorem complex_square_simplification :
  (5 - 3 * Real.sqrt 2 * Complex.I) ^ 2 = 43 - 30 * Real.sqrt 2 * Complex.I :=
by sorry

end complex_square_simplification_l718_71850


namespace employee_pay_solution_exists_and_unique_l718_71864

/-- Represents the weekly pay of employees X, Y, and Z -/
structure EmployeePay where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Conditions for the employee pay problem -/
def satisfiesConditions (pay : EmployeePay) : Prop :=
  pay.x = 1.2 * pay.y ∧
  pay.z = 0.75 * pay.x ∧
  pay.x + pay.y + pay.z = 1540

/-- Theorem stating the existence and uniqueness of the solution -/
theorem employee_pay_solution_exists_and_unique :
  ∃! pay : EmployeePay, satisfiesConditions pay :=
sorry

end employee_pay_solution_exists_and_unique_l718_71864


namespace a_10_equals_21_l718_71835

def S (n : ℕ+) : ℕ := n^2 + 2*n

def a (n : ℕ+) : ℕ := S n - S (n-1)

theorem a_10_equals_21 : a 10 = 21 := by sorry

end a_10_equals_21_l718_71835


namespace stewart_farm_sheep_count_l718_71816

theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ),
    sheep * 7 = horses * 5 →
    horses * 230 = 12880 →
    sheep = 40 := by
sorry

end stewart_farm_sheep_count_l718_71816


namespace simplest_quadratic_radical_l718_71824

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℝ) : Prop :=
  ∃ m : ℝ, n = m^2

-- Define what it means for a quadratic radical to be in its simplest form
def isSimplestQuadraticRadical (x : ℝ) : Prop :=
  x > 0 ∧ ¬(isPerfectSquare x) ∧ ∀ y z : ℝ, (y > 0 ∧ z > 0 ∧ x = y * z) → ¬(isPerfectSquare y)

-- State the theorem
theorem simplest_quadratic_radical :
  ¬(isSimplestQuadraticRadical 0.5) ∧
  ¬(isSimplestQuadraticRadical 8) ∧
  ¬(isSimplestQuadraticRadical 27) ∧
  ∀ a : ℝ, isSimplestQuadraticRadical (a^2 + 1) :=
by sorry

end simplest_quadratic_radical_l718_71824


namespace decagon_partition_impossible_l718_71898

/-- A partition of a polygon into triangles -/
structure TrianglePartition (n : ℕ) where
  black_sides : ℕ
  white_sides : ℕ
  is_valid : black_sides - white_sides = n

/-- Property that the number of sides in a valid triangle partition is divisible by 3 -/
def sides_divisible_by_three (partition : TrianglePartition n) : Prop :=
  partition.black_sides % 3 = 0 ∧ partition.white_sides % 3 = 0

theorem decagon_partition_impossible :
  ¬ ∃ (partition : TrianglePartition 10), sides_divisible_by_three partition :=
sorry

end decagon_partition_impossible_l718_71898


namespace binomial_12_choose_3_l718_71886

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_choose_3_l718_71886


namespace sufficient_condition_l718_71883

theorem sufficient_condition (x y : ℝ) : x > 3 ∧ y > 3 → x + y > 6 ∧ x * y > 9 := by
  sorry

end sufficient_condition_l718_71883


namespace safeties_count_l718_71857

/-- Represents the scoring of a football team -/
structure FootballScore where
  fieldGoals : ℕ      -- number of four-point field goals
  threePointGoals : ℕ -- number of three-point goals
  safeties : ℕ        -- number of two-point safeties

/-- Calculates the total score for a given FootballScore -/
def totalScore (score : FootballScore) : ℕ :=
  4 * score.fieldGoals + 3 * score.threePointGoals + 2 * score.safeties

/-- Theorem: Given the conditions, the number of safeties is 6 -/
theorem safeties_count (score : FootballScore) :
  (4 * score.fieldGoals = 2 * 3 * score.threePointGoals) →
  (score.safeties = score.threePointGoals + 2) →
  (totalScore score = 50) →
  score.safeties = 6 :=
by sorry

end safeties_count_l718_71857


namespace inscribed_rectangle_area_l718_71829

/-- The area of a rectangle inscribed in the ellipse x^2/4 + y^2/8 = 1,
    with sides parallel to the coordinate axes and length twice its width -/
theorem inscribed_rectangle_area :
  ∀ (a b : ℝ),
  (a > 0) →
  (b > 0) →
  (a = 2 * b) →
  (a^2 / 4 + b^2 / 8 = 1) →
  4 * a * b = 32 / 3 := by
sorry

end inscribed_rectangle_area_l718_71829


namespace tax_calculation_l718_71838

theorem tax_calculation (total_earnings deductions tax_paid : ℝ) 
  (h1 : total_earnings = 100000)
  (h2 : deductions = 30000)
  (h3 : tax_paid = 12000) : 
  ∃ (taxed_at_10_percent : ℝ),
    taxed_at_10_percent = 20000 ∧
    tax_paid = 0.1 * taxed_at_10_percent + 
               0.2 * (total_earnings - deductions - taxed_at_10_percent) :=
by sorry

end tax_calculation_l718_71838


namespace total_cost_of_pipes_l718_71880

def copper_length : ℝ := 10
def plastic_length : ℝ := copper_length + 5
def cost_per_meter : ℝ := 4

theorem total_cost_of_pipes : copper_length * cost_per_meter + plastic_length * cost_per_meter = 100 := by
  sorry

end total_cost_of_pipes_l718_71880


namespace tetrahedron_volume_bound_l718_71842

-- Define a tetrahedron type
structure Tetrahedron :=
  (edges : Fin 6 → ℝ)

-- Define the volume of a tetrahedron
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

-- Define the condition that at least 5 edges are not greater than 2
def at_least_five_short_edges (t : Tetrahedron) : Prop :=
  ∃ (long_edge : Fin 6), ∀ (e : Fin 6), e ≠ long_edge → t.edges e ≤ 2

-- Theorem statement
theorem tetrahedron_volume_bound (t : Tetrahedron) :
  at_least_five_short_edges t → volume t ≤ 1 := by
  sorry

end tetrahedron_volume_bound_l718_71842


namespace biology_exam_failure_count_l718_71825

theorem biology_exam_failure_count : 
  ∀ (total_students : ℕ) 
    (perfect_score_fraction : ℚ)
    (passing_score_fraction : ℚ),
  total_students = 80 →
  perfect_score_fraction = 2/5 →
  passing_score_fraction = 1/2 →
  (total_students : ℚ) * perfect_score_fraction +
  (total_students : ℚ) * (1 - perfect_score_fraction) * passing_score_fraction +
  (total_students : ℚ) * (1 - perfect_score_fraction) * (1 - passing_score_fraction) = 
  (total_students : ℚ) →
  (total_students : ℚ) * (1 - perfect_score_fraction) * (1 - passing_score_fraction) = 24 :=
by sorry

end biology_exam_failure_count_l718_71825


namespace tan_alpha_negative_three_l718_71806

theorem tan_alpha_negative_three (α : Real) (h : Real.tan α = -3) :
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = 3 ∧
  Real.sin α ^ 2 + Real.sin α * Real.cos α + 2 = 13/5 := by
  sorry

end tan_alpha_negative_three_l718_71806


namespace trajectory_and_constant_product_l718_71885

-- Define the points and circles
def G : ℝ × ℝ := (5, 4)
def A : ℝ × ℝ := (1, 0)

def C1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 4)^2 = 25

-- Define the lines
def l1 (k x y : ℝ) : Prop := k * x - y - k = 0
def l2 (x y : ℝ) : Prop := x + 2 * y + 2 = 0

-- Define the trajectory C2
def C2 (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define the theorem
theorem trajectory_and_constant_product :
  ∃ (M N : ℝ × ℝ) (k : ℝ),
    (∀ x y, C2 x y ↔ (∃ E F : ℝ × ℝ, C1 E.1 E.2 ∧ C1 F.1 F.2 ∧ 
      (x, y) = ((E.1 + F.1) / 2, (E.2 + F.2) / 2))) ∧
    l1 k M.1 M.2 ∧ 
    l1 k N.1 N.2 ∧ 
    l2 N.1 N.2 ∧
    C2 M.1 M.2 ∧
    (M.1 - A.1)^2 + (M.2 - A.2)^2 * ((N.1 - A.1)^2 + (N.2 - A.2)^2) = 36 :=
by sorry


end trajectory_and_constant_product_l718_71885


namespace lewis_harvest_earnings_l718_71817

/-- Calculates the total earnings during harvest season after paying rent -/
def harvest_earnings (weekly_earnings : ℕ) (weekly_rent : ℕ) (harvest_weeks : ℕ) : ℕ :=
  (weekly_earnings - weekly_rent) * harvest_weeks

/-- Theorem: Lewis's earnings during harvest season -/
theorem lewis_harvest_earnings :
  harvest_earnings 403 49 233 = 82782 := by
  sorry

end lewis_harvest_earnings_l718_71817


namespace arithmetic_simplification_l718_71875

theorem arithmetic_simplification : 2 - (-3) - 4 - (-5) * 2 - 6 - (-7) = 12 := by
  sorry

end arithmetic_simplification_l718_71875


namespace geometric_sequence_sum_l718_71856

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a (n + 1) = a n * r) →  -- geometric sequence definition
  r > 1 →  -- increasing sequence
  a 1 + a 3 + a 5 = 21 →  -- given condition
  a 3 = 6 →  -- given condition
  a 5 + a 7 + a 9 = 84 :=
by sorry

end geometric_sequence_sum_l718_71856


namespace magnitude_of_AB_l718_71830

-- Define points A and B
def A : ℝ × ℝ := (-3, 4)
def B : ℝ × ℝ := (5, -2)

-- Define vector AB
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Theorem: The magnitude of vector AB is 10
theorem magnitude_of_AB : Real.sqrt (vectorAB.1^2 + vectorAB.2^2) = 10 := by
  sorry


end magnitude_of_AB_l718_71830


namespace arithmetic_sequence_sum_l718_71870

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: The sum of the arithmetic sequence with first term 2, last term 29, and common difference 3 is 155 -/
theorem arithmetic_sequence_sum : arithmetic_sum 2 29 3 = 155 := by
  sorry

end arithmetic_sequence_sum_l718_71870


namespace andrea_rhinestones_l718_71832

theorem andrea_rhinestones (total : ℕ) (bought : ℚ) (found : ℚ) : 
  total = 120 → 
  bought = 2 / 5 → 
  found = 1 / 6 → 
  total - (total * bought + total * found) = 52 := by
sorry

end andrea_rhinestones_l718_71832


namespace parabola_points_distance_l718_71804

/-- A parabola defined by y = 9x^2 - 3x + 2 -/
def parabola (x y : ℝ) : Prop := y = 9 * x^2 - 3 * x + 2

/-- The origin (0,0) is the midpoint of two points -/
def origin_is_midpoint (p q : ℝ × ℝ) : Prop :=
  (p.1 + q.1) / 2 = 0 ∧ (p.2 + q.2) / 2 = 0

/-- The square of the distance between two points -/
def square_distance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem parabola_points_distance (p q : ℝ × ℝ) :
  parabola p.1 p.2 ∧ parabola q.1 q.2 ∧ origin_is_midpoint p q →
  square_distance p q = 580 / 9 := by
  sorry

end parabola_points_distance_l718_71804


namespace angelina_speed_l718_71833

/-- Angelina's walk from home to grocery to gym -/
def angelina_walk (v : ℝ) : Prop :=
  let home_to_grocery_distance : ℝ := 180
  let grocery_to_gym_distance : ℝ := 240
  let home_to_grocery_time : ℝ := home_to_grocery_distance / v
  let grocery_to_gym_time : ℝ := grocery_to_gym_distance / (2 * v)
  home_to_grocery_time = grocery_to_gym_time + 40

theorem angelina_speed : ∃ v : ℝ, angelina_walk v ∧ 2 * v = 3 := by sorry

end angelina_speed_l718_71833


namespace reciprocal_of_negative_2023_l718_71801

theorem reciprocal_of_negative_2023 :
  ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end reciprocal_of_negative_2023_l718_71801


namespace line_general_form_l718_71890

/-- A line passing through a point with a given slope -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The general form of a line equation: ax + by + c = 0 -/
structure GeneralForm where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a line with slope -3 passing through the point (1, 2),
    its general form equation is 3x + y - 5 = 0 -/
theorem line_general_form (l : Line) 
    (h1 : l.slope = -3)
    (h2 : l.point = (1, 2)) :
    ∃ (g : GeneralForm), g.a = 3 ∧ g.b = 1 ∧ g.c = -5 :=
by sorry

end line_general_form_l718_71890


namespace fish_population_estimate_l718_71896

/-- Represents the number of fish of a particular species in the pond -/
structure FishPopulation where
  speciesA : ℕ
  speciesB : ℕ
  speciesC : ℕ

/-- Represents the number of tagged fish caught in the second round -/
structure TaggedCatch where
  speciesA : ℕ
  speciesB : ℕ
  speciesC : ℕ

/-- Calculates the estimated population of a species based on the initial tagging and second catch -/
def estimatePopulation (initialTagged : ℕ) (secondCatchTotal : ℕ) (taggedInSecondCatch : ℕ) : ℕ :=
  (initialTagged * secondCatchTotal) / taggedInSecondCatch

/-- Theorem stating the estimated fish population given the initial tagging and second catch data -/
theorem fish_population_estimate 
  (initialTagged : ℕ) 
  (secondCatchTotal : ℕ) 
  (taggedCatch : TaggedCatch) : 
  initialTagged = 40 →
  secondCatchTotal = 180 →
  taggedCatch.speciesA = 3 →
  taggedCatch.speciesB = 5 →
  taggedCatch.speciesC = 2 →
  let estimatedPopulation := FishPopulation.mk
    (estimatePopulation initialTagged secondCatchTotal taggedCatch.speciesA)
    (estimatePopulation initialTagged secondCatchTotal taggedCatch.speciesB)
    (estimatePopulation initialTagged secondCatchTotal taggedCatch.speciesC)
  estimatedPopulation.speciesA = 2400 ∧ 
  estimatedPopulation.speciesB = 1440 ∧ 
  estimatedPopulation.speciesC = 3600 := by
  sorry


end fish_population_estimate_l718_71896


namespace mary_flour_amount_l718_71826

/-- Given a recipe that requires a total amount of flour and the amount still needed to be added,
    calculate the amount of flour already put in. -/
def flour_already_added (total : ℕ) (to_add : ℕ) : ℕ :=
  total - to_add

/-- Theorem: Mary has already put in 2 cups of flour -/
theorem mary_flour_amount : flour_already_added 8 6 = 2 := by
  sorry

end mary_flour_amount_l718_71826


namespace tangent_circle_radius_l718_71872

theorem tangent_circle_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 1) (h₂ : r₂ = 2) (h₃ : r₃ = 3) :
  ∃ r : ℝ, r > 0 ∧
  (r₁ + r)^2 + (r₂ + r)^2 = (r₃ - r)^2 + (r₁ + r₂)^2 ∧
  r = 6/7 := by
sorry

end tangent_circle_radius_l718_71872


namespace scaled_model_height_l718_71877

/-- Represents a cylindrical monument --/
structure CylindricalMonument where
  height : ℝ
  baseRadius : ℝ
  volume : ℝ

/-- Represents a scaled model of the monument --/
structure ScaledModel where
  volume : ℝ
  height : ℝ

/-- Theorem stating the relationship between the original monument and its scaled model --/
theorem scaled_model_height 
  (monument : CylindricalMonument) 
  (model : ScaledModel) : 
  monument.height = 100 ∧ 
  monument.baseRadius = 20 ∧ 
  monument.volume = 125600 ∧ 
  model.volume = 1.256 → 
  model.height = 1 := by
  sorry


end scaled_model_height_l718_71877


namespace kho_kho_problem_l718_71843

/-- Represents the number of students who left to play kho-kho -/
def students_who_left (initial_boys initial_girls remaining_girls : ℕ) : ℕ :=
  initial_girls - remaining_girls

/-- Proves that 8 girls left to play kho-kho given the problem conditions -/
theorem kho_kho_problem (initial_boys initial_girls remaining_girls : ℕ) :
  initial_boys = initial_girls →
  initial_boys + initial_girls = 32 →
  initial_boys = 2 * remaining_girls →
  students_who_left initial_boys initial_girls remaining_girls = 8 :=
by
  sorry

#check kho_kho_problem

end kho_kho_problem_l718_71843


namespace sixty_four_to_five_sixths_l718_71828

theorem sixty_four_to_five_sixths (h : 64 = 2^6) : 64^(5/6) = 32 := by
  sorry

end sixty_four_to_five_sixths_l718_71828


namespace polygon_sides_l718_71820

/-- A polygon with equal internal angles and external angles equal to 2/3 of the adjacent internal angles has 5 sides. -/
theorem polygon_sides (n : ℕ) (internal_angle : ℝ) (external_angle : ℝ) : 
  n > 2 →
  internal_angle > 0 →
  external_angle > 0 →
  (n : ℝ) * internal_angle = (n - 2 : ℝ) * 180 →
  external_angle = (2 / 3) * internal_angle →
  internal_angle + external_angle = 180 →
  n = 5 := by
sorry

end polygon_sides_l718_71820


namespace madeline_and_brother_total_l718_71839

/-- Given Madeline has $48 and her brother has half as much, prove that they have $72 together. -/
theorem madeline_and_brother_total (madeline_amount : ℕ) (brother_amount : ℕ) : 
  madeline_amount = 48 → 
  brother_amount = madeline_amount / 2 → 
  madeline_amount + brother_amount = 72 := by
sorry

end madeline_and_brother_total_l718_71839


namespace fraction_division_equality_l718_71807

theorem fraction_division_equality : (3 : ℚ) / 7 / (5 / 2) = 6 / 35 := by
  sorry

end fraction_division_equality_l718_71807


namespace parabola_focus_l718_71819

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -4*y

-- Define the focus of a parabola
def focus (p : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  let (x, y) := p
  parabola x y ∧ x = 0 ∧ y = -1

-- Theorem statement
theorem parabola_focus :
  ∃ p : ℝ × ℝ, focus p parabola :=
sorry

end parabola_focus_l718_71819


namespace problem_solution_l718_71899

theorem problem_solution : (((Real.sqrt 25 - 1) / 2) ^ 2 + 3) ⁻¹ * 10 = 10 / 7 := by
  sorry

end problem_solution_l718_71899


namespace no_solutions_for_sqrt_equation_l718_71803

theorem no_solutions_for_sqrt_equation :
  ¬∃ x : ℝ, x ≥ 4 ∧ Real.sqrt (x + 9 - 6 * Real.sqrt (x - 4)) + Real.sqrt (x + 16 - 8 * Real.sqrt (x - 4)) = 2 :=
by sorry

end no_solutions_for_sqrt_equation_l718_71803


namespace sqrt_x_minus_one_real_l718_71881

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
  sorry

end sqrt_x_minus_one_real_l718_71881


namespace time_to_finish_problems_l718_71813

/-- The time required to finish all problems given the number of math and spelling problems and the rate of problem-solving. -/
theorem time_to_finish_problems
  (math_problems : ℕ)
  (spelling_problems : ℕ)
  (problems_per_hour : ℕ)
  (h1 : math_problems = 18)
  (h2 : spelling_problems = 6)
  (h3 : problems_per_hour = 4) :
  (math_problems + spelling_problems) / problems_per_hour = 6 :=
by sorry

end time_to_finish_problems_l718_71813


namespace value_of_c_l718_71836

theorem value_of_c : ∃ c : ℝ, 
  (∀ x : ℝ, x * (4 * x + 2) < c ↔ -5/2 < x ∧ x < 3) ∧ c = 45 := by
  sorry

end value_of_c_l718_71836


namespace anns_age_l718_71822

theorem anns_age (a b : ℕ) : 
  a + b = 50 → 
  b = (2 * a / 3 : ℚ) + 2 * (a - b) → 
  a = 26 := by
sorry

end anns_age_l718_71822


namespace average_page_count_l718_71814

theorem average_page_count (n : ℕ) (g1 g2 g3 : ℕ) (p1 p2 p3 : ℕ) :
  n = g1 + g2 + g3 →
  g1 = g2 ∧ g2 = g3 →
  (g1 * p1 + g2 * p2 + g3 * p3) / n = 2 →
  n = 15 ∧ g1 = 5 ∧ p1 = 2 ∧ p2 = 3 ∧ p3 = 1 →
  (g1 * p1 + g2 * p2 + g3 * p3) / n = 2 :=
by sorry

end average_page_count_l718_71814


namespace pentagonal_prism_sum_l718_71859

/-- A pentagonal prism is a three-dimensional geometric shape with pentagonal bases and rectangular lateral faces. -/
structure PentagonalPrism where
  /-- The number of faces in a pentagonal prism -/
  faces : Nat
  /-- The number of edges in a pentagonal prism -/
  edges : Nat
  /-- The number of vertices in a pentagonal prism -/
  vertices : Nat
  /-- The faces of a pentagonal prism consist of 2 pentagonal bases and 5 rectangular lateral faces -/
  faces_def : faces = 7
  /-- The edges of a pentagonal prism consist of 10 edges from the two pentagons and 5 edges connecting them -/
  edges_def : edges = 15
  /-- The vertices of a pentagonal prism are the 5 vertices from each of the two pentagonal bases -/
  vertices_def : vertices = 10

/-- The sum of faces, edges, and vertices of a pentagonal prism is 32 -/
theorem pentagonal_prism_sum (p : PentagonalPrism) : p.faces + p.edges + p.vertices = 32 := by
  sorry

end pentagonal_prism_sum_l718_71859


namespace stadium_sections_theorem_l718_71865

theorem stadium_sections_theorem : 
  ∃ (N : ℕ), N > 0 ∧ 
  (∃ (A C : ℕ), 7 * A = 11 * C ∧ N = A + C) ∧ 
  (∀ (M : ℕ), M > 0 → 
    (∃ (A C : ℕ), 7 * A = 11 * C ∧ M = A + C) → M ≥ N) ∧
  N = 18 :=
sorry

end stadium_sections_theorem_l718_71865


namespace article_price_calculation_l718_71884

/-- The original price of an article before discounts and tax -/
def original_price : ℝ := 259.20

/-- The final price of the article after discounts and tax -/
def final_price : ℝ := 144

/-- The first discount rate -/
def discount1 : ℝ := 0.12

/-- The second discount rate -/
def discount2 : ℝ := 0.22

/-- The third discount rate -/
def discount3 : ℝ := 0.15

/-- The sales tax rate -/
def tax_rate : ℝ := 0.06

theorem article_price_calculation (ε : ℝ) (hε : ε > 0) :
  ∃ (price : ℝ), 
    abs (price - original_price) < ε ∧ 
    price * (1 - discount1) * (1 - discount2) * (1 - discount3) * (1 + tax_rate) = final_price :=
sorry

end article_price_calculation_l718_71884


namespace increasing_function_inequality_l718_71858

theorem increasing_function_inequality (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_inequality : ∀ x₁ x₂, f x₁ + f x₂ ≥ f (-x₁) + f (-x₂)) :
  ∀ x₁ x₂, x₁ + x₂ ≥ 0 :=
by sorry

end increasing_function_inequality_l718_71858


namespace range_of_m_l718_71892

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 - m*x - m ≥ 0) → m ∈ Set.Icc (-4) 0 :=
sorry

end range_of_m_l718_71892


namespace sum_of_coefficients_l718_71837

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5 + (1 + x)^6 + (1 + x)^7 + (1 + x)^8
           = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 502 := by
sorry

end sum_of_coefficients_l718_71837


namespace mothers_age_twice_lucys_l718_71866

/-- Given Lucy's age and her mother's age in 2012, find the year when the mother's age will be twice Lucy's age -/
theorem mothers_age_twice_lucys (lucy_age_2012 : ℕ) (mother_age_multiplier : ℕ) : 
  lucy_age_2012 = 10 →
  mother_age_multiplier = 5 →
  ∃ (years_after_2012 : ℕ),
    (lucy_age_2012 + years_after_2012) * 2 = (lucy_age_2012 * mother_age_multiplier + years_after_2012) ∧
    2012 + years_after_2012 = 2042 :=
by sorry

end mothers_age_twice_lucys_l718_71866


namespace largest_prime_factor_of_9871_l718_71834

theorem largest_prime_factor_of_9871 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 9871 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 9871 → q ≤ p :=
by sorry

end largest_prime_factor_of_9871_l718_71834


namespace inequality_theorem_l718_71863

theorem inequality_theorem (x y : ℝ) 
  (h1 : y ≥ 0) 
  (h2 : y * (y + 1) ≤ (x + 1)^2) 
  (h3 : y * (y - 1) ≤ x^2) : 
  y * (y - 1) ≤ x^2 ∧ y * (y + 1) ≤ (x + 1)^2 := by
  sorry

end inequality_theorem_l718_71863


namespace infinite_primes_4n_plus_3_l718_71888

theorem infinite_primes_4n_plus_3 :
  ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p ∧ p % 4 = 3) →
  ∃ q, Nat.Prime q ∧ q % 4 = 3 ∧ q ∉ S :=
sorry

end infinite_primes_4n_plus_3_l718_71888


namespace trigonometric_identity_l718_71882

theorem trigonometric_identity : 
  Real.cos (π / 12) * Real.cos (5 * π / 12) + Real.cos (π / 8)^2 - 1/2 = (Real.sqrt 2 + 1) / 4 := by
  sorry

end trigonometric_identity_l718_71882


namespace div_remainder_theorem_l718_71809

theorem div_remainder_theorem : 
  ∃ k : ℕ, 3^19 = k * 1162261460 + 7 :=
sorry

end div_remainder_theorem_l718_71809


namespace number_of_male_students_l718_71874

theorem number_of_male_students 
  (total_average : ℝ) 
  (male_average : ℝ) 
  (female_average : ℝ) 
  (num_female : ℕ) 
  (h1 : total_average = 90) 
  (h2 : male_average = 84) 
  (h3 : female_average = 92) 
  (h4 : num_female = 24) :
  ∃ (num_male : ℕ), 
    num_male = 8 ∧ 
    (num_male : ℝ) * male_average + (num_female : ℝ) * female_average = 
      ((num_male : ℝ) + (num_female : ℝ)) * total_average :=
by sorry

end number_of_male_students_l718_71874


namespace original_number_proof_l718_71895

theorem original_number_proof (h1 : 213 * 16 = 3408) 
  (h2 : ∃ x, x * 21.3 = 34.080000000000005) : 
  ∃ x, x * 21.3 = 34.080000000000005 ∧ x = 1.6 :=
by sorry

end original_number_proof_l718_71895


namespace lcm_ten_times_gcd_characterization_l718_71811

theorem lcm_ten_times_gcd_characterization (a b : ℕ+) :
  Nat.lcm a b = 10 * Nat.gcd a b ↔
  (∃ d : ℕ+, (a = d ∧ b = 10 * d) ∨
             (a = 2 * d ∧ b = 5 * d) ∨
             (a = 5 * d ∧ b = 2 * d) ∨
             (a = 10 * d ∧ b = d)) :=
by sorry

end lcm_ten_times_gcd_characterization_l718_71811


namespace ticket_probability_problem_l718_71840

theorem ticket_probability_problem : ∃! n : ℕ, 
  1 ≤ n ∧ n ≤ 20 ∧ 
  (↑(Finset.filter (λ x => x % n = 0) (Finset.range 20)).card / 20 : ℚ) = 3/10 ∧
  n = 3 := by
  sorry

end ticket_probability_problem_l718_71840


namespace equation_solution_l718_71821

theorem equation_solution (a : ℚ) : -3 / (a - 3) = 3 / (a + 2) → a = 1/2 := by
  sorry

end equation_solution_l718_71821


namespace hunter_frog_count_l718_71844

/-- The total number of frogs Hunter saw in the pond -/
def total_frogs (initial : ℕ) (on_logs : ℕ) (babies : ℕ) : ℕ :=
  initial + on_logs + babies

/-- Theorem stating the total number of frogs Hunter saw -/
theorem hunter_frog_count :
  total_frogs 5 3 24 = 32 := by
  sorry

end hunter_frog_count_l718_71844


namespace tan_315_degrees_l718_71848

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end tan_315_degrees_l718_71848


namespace f_properties_l718_71851

def f (x : ℝ) : ℝ := x^2 - 2*x + 1

theorem f_properties :
  (∃ (x : ℝ), f x = 0 ∧ x = 1) ∧
  (f 0 * f 2 > 0) ∧
  (∃ (x : ℝ), x > 0 ∧ x < 2 ∧ f x = 0) ∧
  (¬ ∀ (x y : ℝ), x < y ∧ y < 0 → f x > f y) :=
sorry

end f_properties_l718_71851


namespace cos_sin_2theta_l718_71827

theorem cos_sin_2theta (θ : ℝ) (h : 3 * Real.sin θ = Real.cos θ) :
  Real.cos (2 * θ) + Real.sin (2 * θ) = 7 / 5 := by
  sorry

end cos_sin_2theta_l718_71827


namespace circle_circumference_l718_71897

/-- The circumference of a circle with radius 36 is 72π -/
theorem circle_circumference (π : ℝ) (h : π > 0) : ∃ (k : ℝ), k * π = 2 * π * 36 ∧ k = 72 := by
  sorry

end circle_circumference_l718_71897


namespace bacteria_growth_l718_71862

theorem bacteria_growth (n : ℕ) : (∀ k < n, 4 * 3^k ≤ 500) ∧ 4 * 3^n > 500 → n = 5 := by
  sorry

end bacteria_growth_l718_71862


namespace problem_solution_l718_71878

theorem problem_solution : 48 / (7 - 3/4 + 1/8) = 128/17 := by
  sorry

end problem_solution_l718_71878


namespace equipment_theorem_l718_71891

/-- Represents the sales data for equipment A and B -/
structure SalesData where
  a : ℕ  -- quantity of A
  b : ℕ  -- quantity of B
  total : ℕ  -- total amount in yuan

/-- Represents the problem setup -/
structure EquipmentProblem where
  sale1 : SalesData
  sale2 : SalesData
  totalPieces : ℕ
  maxRatio : ℕ  -- max ratio of A to B
  maxCost : ℕ

/-- The main theorem to prove -/
theorem equipment_theorem (p : EquipmentProblem) 
  (h1 : p.sale1 = ⟨20, 10, 1100⟩)
  (h2 : p.sale2 = ⟨25, 20, 1750⟩)
  (h3 : p.totalPieces = 50)
  (h4 : p.maxRatio = 2)
  (h5 : p.maxCost = 2000) :
  ∃ (priceA priceB : ℕ),
    priceA = 30 ∧ 
    priceB = 50 ∧ 
    (∃ (validPlans : Finset (ℕ × ℕ)),
      validPlans.card = 9 ∧
      ∀ (plan : ℕ × ℕ), plan ∈ validPlans ↔ 
        (plan.1 + plan.2 = p.totalPieces ∧
         plan.1 ≤ p.maxRatio * plan.2 ∧
         plan.1 * priceA + plan.2 * priceB ≤ p.maxCost)) :=
by sorry

end equipment_theorem_l718_71891


namespace base_ten_arithmetic_l718_71805

theorem base_ten_arithmetic : (456 + 123) - 579 = 0 := by
  sorry

end base_ten_arithmetic_l718_71805


namespace game_theorem_l718_71889

/-- Represents the outcome of a single round -/
inductive RoundOutcome
| OddDifference
| EvenDifference

/-- Represents the game state -/
structure GameState :=
  (playerAPoints : ℤ)
  (playerBPoints : ℤ)

/-- The game rules -/
def gameRules (n : ℕ+) (outcome : RoundOutcome) (state : GameState) : GameState :=
  match outcome with
  | RoundOutcome.OddDifference  => ⟨state.playerAPoints - 2, state.playerBPoints + 2⟩
  | RoundOutcome.EvenDifference => ⟨state.playerAPoints + n, state.playerBPoints - n⟩

/-- The probability of an odd difference in a single round -/
def probOddDifference : ℚ := 3/5

/-- The probability of an even difference in a single round -/
def probEvenDifference : ℚ := 2/5

/-- The expected value of player A's points after the game -/
def expectedValue (n : ℕ+) : ℚ := (6 * n - 18) / 5

/-- The theorem to be proved -/
theorem game_theorem (n : ℕ+) :
  (∀ m : ℕ+, m < n → expectedValue m ≤ 0) ∧
  expectedValue n > 0 ∧
  n = 4 →
  (probOddDifference^3 + 3 * probOddDifference^2 * probEvenDifference) *
  (3 * probOddDifference * probEvenDifference^2) / 
  (1 - probEvenDifference^3) = 4/13 := by
  sorry


end game_theorem_l718_71889


namespace three_heads_in_a_row_probability_l718_71861

def coin_flips : ℕ := 6

def favorable_outcomes : ℕ := 12

def total_outcomes : ℕ := 2^coin_flips

def probability : ℚ := favorable_outcomes / total_outcomes

theorem three_heads_in_a_row_probability :
  probability = 3/16 := by sorry

end three_heads_in_a_row_probability_l718_71861


namespace max_point_condition_l718_71871

/-- The function f(x) defined as (x-a)^2 * (x-1) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x - a)^2 * (x - 1)

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := (x - a) * (3*x - a - 2)

theorem max_point_condition (a : ℝ) :
  (∀ x : ℝ, f a x ≤ f a a) ↔ a < 1 := by sorry

end max_point_condition_l718_71871


namespace solve_problem_l718_71815

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base^(digits.length - 1 - i)) 0

def problem : Prop :=
  let base_6_num := [5, 4, 3, 2, 1, 0]
  let base_7_num := [4, 3, 2, 1, 0]
  (base_to_decimal base_6_num 6) - (base_to_decimal base_7_num 7) = 34052

theorem solve_problem : problem := by
  sorry

end solve_problem_l718_71815


namespace peter_reading_time_l718_71852

/-- Given that Peter reads three times as fast as Kristin and Kristin reads half of her 20 books in 540 hours, prove that Peter takes 18 hours to read one book. -/
theorem peter_reading_time (peter_speed : ℝ) (kristin_speed : ℝ) (kristin_half_books : ℕ) (kristin_half_time : ℝ) : 
  peter_speed = 3 * kristin_speed →
  kristin_half_books = 10 →
  kristin_half_time = 540 →
  peter_speed = 1 / 18 := by
sorry

end peter_reading_time_l718_71852


namespace orthocenter_position_in_isosceles_triangle_l718_71802

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : Vector2D :=
  sorry

/-- Checks if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

theorem orthocenter_position_in_isosceles_triangle 
  (t : Triangle) 
  (h_isosceles : isIsosceles t) 
  (h_sides : t.a = 5 ∧ t.b = 5 ∧ t.c = 6) :
  ∃ (m n : ℝ), 
    let H := orthocenter t
    let A := Vector2D.mk 0 0
    let B := Vector2D.mk t.c 0
    let C := Vector2D.mk (t.c / 2) (Real.sqrt (t.a^2 - (t.c / 2)^2))
    H.x = m * B.x + n * C.x ∧
    H.y = m * B.y + n * C.y ∧
    m + n = 21 / 32 :=
  sorry

end orthocenter_position_in_isosceles_triangle_l718_71802


namespace inequality_proof_l718_71894

theorem inequality_proof (a b c d e p q : ℝ) 
  (hp : 0 < p) 
  (hpa : p ≤ a) (hpb : p ≤ b) (hpc : p ≤ c) (hpd : p ≤ d) (hpe : p ≤ e)
  (haq : a ≤ q) (hbq : b ≤ q) (hcq : c ≤ q) (hdq : d ≤ q) (heq : e ≤ q) :
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) ≤ 
    25 + 6 * (Real.sqrt (q/p) - Real.sqrt (p/q))^2 := by
  sorry

end inequality_proof_l718_71894


namespace abs_2x_minus_5_l718_71869

theorem abs_2x_minus_5 (x : ℝ) (h : |2*x - 3| - 3 + 2*x = 0) : |2*x - 5| = 5 - 2*x := by
  sorry

end abs_2x_minus_5_l718_71869


namespace park_area_l718_71868

theorem park_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) : 
  width > 0 → 
  length > 0 → 
  length = 3 * width → 
  perimeter = 2 * (width + length) → 
  perimeter = 72 → 
  area = width * length → 
  area = 243 := by sorry

end park_area_l718_71868


namespace laura_workout_speed_l718_71841

theorem laura_workout_speed :
  ∃! x : ℝ, x > 0 ∧ (30 / (3 * x + 2) + 3 / x = (230 - 10) / 60) := by
  sorry

end laura_workout_speed_l718_71841


namespace det_A_equals_l718_71860

-- Define the matrix as a function of y
def A (y : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![y^2 + 1, 2*y, 2*y;
     2*y, y^2 + 3, 2*y;
     2*y, 2*y, y^2 + 5]

-- State the theorem
theorem det_A_equals (y : ℝ) : 
  Matrix.det (A y) = y^6 + y^4 + 35*y^2 + 15 - 32*y := by
  sorry

end det_A_equals_l718_71860


namespace min_cost_theorem_l718_71893

/-- Represents the voting system in country Y -/
structure VotingSystem where
  total_voters : Nat
  sellable_voters : Nat
  preference_voters : Nat
  initial_votes : Nat
  votes_to_win : Nat

/-- Calculates the number of votes a candidate can secure based on the price offered -/
def supply_function (system : VotingSystem) (price : Nat) : Nat :=
  if price = 0 then system.initial_votes
  else if price ≤ system.sellable_voters then min (system.initial_votes + price) system.total_voters
  else min (system.initial_votes + system.sellable_voters) system.total_voters

/-- Calculates the minimum cost to win the election -/
def min_cost_to_win (system : VotingSystem) : Nat :=
  let required_additional_votes := system.votes_to_win - system.initial_votes
  required_additional_votes * (required_additional_votes + 1)

/-- The main theorem stating the minimum cost to win the election -/
theorem min_cost_theorem (system : VotingSystem) 
    (h1 : system.total_voters = 35)
    (h2 : system.sellable_voters = 14)
    (h3 : system.preference_voters = 21)
    (h4 : system.initial_votes = 10)
    (h5 : system.votes_to_win = 18) :
    min_cost_to_win system = 162 := by
  sorry

#eval min_cost_to_win { total_voters := 35, sellable_voters := 14, preference_voters := 21, initial_votes := 10, votes_to_win := 18 }

end min_cost_theorem_l718_71893


namespace wage_cut_and_raise_l718_71847

theorem wage_cut_and_raise (original_wage : ℝ) (h : original_wage > 0) :
  let cut_wage := 0.7 * original_wage
  let required_raise := (original_wage / cut_wage) - 1
  ∃ ε > 0, abs (required_raise - 0.4286) < ε :=
by sorry

end wage_cut_and_raise_l718_71847


namespace problem_solution_l718_71800

theorem problem_solution : (3 - Real.pi) ^ 0 - 3 ^ (-1 : ℤ) = 2/3 := by
  sorry

end problem_solution_l718_71800


namespace circle_common_chord_l718_71855

variables (a b x y : ℝ)

-- Define the first circle
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*a*x = 0

-- Define the second circle
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*b*y = 0

-- Define the resulting circle
def resultCircle (x y : ℝ) : Prop := (a^2 + b^2)*(x^2 + y^2) - 2*a*b*(b*x + a*y) = 0

-- Theorem statement
theorem circle_common_chord (hb : b ≠ 0) :
  ∃ (x y : ℝ), circle1 a x y ∧ circle2 b x y →
  resultCircle a b x y ∧
  ∀ (x' y' : ℝ), resultCircle a b x' y' →
    ∃ (t : ℝ), x' = x + t*(y - x) ∧ y' = y + t*(x - y) :=
sorry

end circle_common_chord_l718_71855


namespace complex_expression_equals_negative_two_l718_71831

theorem complex_expression_equals_negative_two :
  (Real.sqrt 6 + Real.sqrt 2) * (Real.sqrt 3 - 2) * Real.sqrt (Real.sqrt 3 + 2) = -2 := by
  sorry

end complex_expression_equals_negative_two_l718_71831


namespace max_principals_in_period_l718_71823

/-- Represents the duration of the period in years -/
def period_duration : ℕ := 8

/-- Represents the duration of a principal's term in years -/
def term_duration : ℕ := 4

/-- Represents the maximum number of non-overlapping terms that can fit within the period -/
def max_principals : ℕ := period_duration / term_duration

theorem max_principals_in_period :
  max_principals = 2 :=
sorry

end max_principals_in_period_l718_71823


namespace workshop_workers_count_l718_71845

/-- Proves that the total number of workers in a workshop is 28 given the salary conditions --/
theorem workshop_workers_count : ∀ (W : ℕ) (N : ℕ),
  W = N + 7 →  -- Total workers = Non-technicians + Technicians
  W * 8000 = 7 * 14000 + N * 6000 →  -- Total salary equation
  W = 28 := by
  sorry

end workshop_workers_count_l718_71845


namespace difference_has_7_in_thousands_l718_71849

/-- Given a number with 3 in the ten-thousands place (28943712) and its local value (30000) -/
def local_value_of_3 : ℕ := 30000

/-- The difference between an unknown number and the local value of 3 -/
def difference (x : ℕ) : ℕ := x - local_value_of_3

/-- Check if a number has 7 in the thousands place -/
def has_7_in_thousands (n : ℕ) : Prop :=
  (n / 1000) % 10 = 7

/-- The local value of 7 in the thousands place -/
def local_value_of_7_in_thousands : ℕ := 7000

/-- Theorem: If the difference has 7 in the thousands place, 
    then the local value of 7 in the difference is 7000 -/
theorem difference_has_7_in_thousands (x : ℕ) :
  has_7_in_thousands (difference x) →
  (difference x / 1000) % 10 * 1000 = local_value_of_7_in_thousands :=
by
  sorry

end difference_has_7_in_thousands_l718_71849


namespace certain_number_proof_l718_71854

theorem certain_number_proof (x : ℝ) : 3 * (x + 8) = 36 → x = 4 := by
  sorry

end certain_number_proof_l718_71854


namespace cans_for_reduced_people_l718_71853

/-- Given 600 cans feed 40 people, proves the number of cans needed for 30% fewer people is 420 -/
theorem cans_for_reduced_people (total_cans : ℕ) (original_people : ℕ) (reduction_percent : ℚ) : 
  total_cans = 600 → 
  original_people = 40 → 
  reduction_percent = 30 / 100 →
  (total_cans / original_people : ℚ) * (original_people * (1 - reduction_percent) : ℚ) = 420 := by
  sorry

end cans_for_reduced_people_l718_71853


namespace truck_toll_calculation_l718_71846

/-- Calculate the toll for a truck based on its number of axles -/
def toll (x : ℕ) : ℚ := 2.5 + 0.5 * (x - 2)

/-- Calculate the number of axles for a truck given its wheel configuration -/
def axle_count (total_wheels front_wheels other_axle_wheels : ℕ) : ℕ :=
  1 + (total_wheels - front_wheels) / other_axle_wheels

theorem truck_toll_calculation (total_wheels front_wheels other_axle_wheels : ℕ) 
  (h1 : total_wheels = 18)
  (h2 : front_wheels = 2)
  (h3 : other_axle_wheels = 4) :
  toll (axle_count total_wheels front_wheels other_axle_wheels) = 4 := by
  sorry

end truck_toll_calculation_l718_71846
