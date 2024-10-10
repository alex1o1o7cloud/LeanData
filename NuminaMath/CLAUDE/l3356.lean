import Mathlib

namespace model1_is_best_fitting_l3356_335653

-- Define the structure for a regression model
structure RegressionModel where
  name : String
  r_squared : Real

-- Define the four models
def model1 : RegressionModel := ⟨"Model 1", 0.98⟩
def model2 : RegressionModel := ⟨"Model 2", 0.80⟩
def model3 : RegressionModel := ⟨"Model 3", 0.50⟩
def model4 : RegressionModel := ⟨"Model 4", 0.25⟩

-- Define a list of all models
def allModels : List RegressionModel := [model1, model2, model3, model4]

-- Define a function to determine if a model is the best fitting
def isBestFitting (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, model.r_squared ≥ m.r_squared

-- Theorem stating that Model 1 is the best fitting model
theorem model1_is_best_fitting :
  isBestFitting model1 allModels := by
  sorry

end model1_is_best_fitting_l3356_335653


namespace detergent_calculation_l3356_335695

/-- The amount of detergent used per pound of clothes -/
def detergent_per_pound : ℝ := 2

/-- The number of pounds of clothes washed -/
def pounds_of_clothes : ℝ := 9

/-- The total amount of detergent used -/
def total_detergent : ℝ := detergent_per_pound * pounds_of_clothes

theorem detergent_calculation : total_detergent = 18 := by
  sorry

end detergent_calculation_l3356_335695


namespace two_cones_intersection_angle_l3356_335665

/-- Represents a cone with height and base radius -/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents the configuration of two cones -/
structure TwoCones where
  cone1 : Cone
  cone2 : Cone
  commonVertex : Bool
  touchingEachOther : Bool
  touchingPlane : Bool

/-- The angle between the line of intersection of the base planes and the touching plane -/
def intersectionAngle (tc : TwoCones) : ℝ := sorry

theorem two_cones_intersection_angle 
  (tc : TwoCones) 
  (h1 : tc.cone1 = tc.cone2) 
  (h2 : tc.cone1.height = 2) 
  (h3 : tc.cone1.baseRadius = 1) 
  (h4 : tc.commonVertex = true) 
  (h5 : tc.touchingEachOther = true) 
  (h6 : tc.touchingPlane = true) : 
  intersectionAngle tc = Real.pi / 3 := by sorry

end two_cones_intersection_angle_l3356_335665


namespace box_volume_l3356_335620

-- Define the set of possible volumes
def possibleVolumes : Set ℕ := {180, 240, 300, 360, 450}

-- Theorem statement
theorem box_volume (a b c : ℕ) (h1 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h2 : ∃ (k : ℕ), k > 0 ∧ a = 2 * k ∧ b = 3 * k ∧ c = 5 * k) :
  (a * b * c) ∈ possibleVolumes ↔ a * b * c = 240 := by
  sorry

end box_volume_l3356_335620


namespace total_cards_l3356_335692

theorem total_cards (initial_cards : ℕ) (added_cards : ℕ) : 
  initial_cards = 4 → added_cards = 3 → initial_cards + added_cards = 7 :=
by sorry

end total_cards_l3356_335692


namespace adjusted_smallest_part_is_correct_l3356_335645

-- Define the total amount
def total : ℚ := 100

-- Define the proportions
def proportions : List ℚ := [1, 3, 4, 6]

-- Define the extra amount added to the smallest part
def extra : ℚ := 12

-- Define the function to calculate the adjusted smallest part
def adjusted_smallest_part (total : ℚ) (proportions : List ℚ) (extra : ℚ) : ℚ :=
  let sum_proportions := proportions.sum
  let smallest_part := total * (proportions.head! / sum_proportions)
  smallest_part + extra

-- Theorem statement
theorem adjusted_smallest_part_is_correct :
  adjusted_smallest_part total proportions extra = 19 + 1/7 := by
  sorry

end adjusted_smallest_part_is_correct_l3356_335645


namespace boat_speed_proof_l3356_335643

/-- The speed of the boat in still water -/
def boat_speed : ℝ := 10

/-- The speed of the stream -/
def stream_speed : ℝ := 2

/-- The distance traveled -/
def distance : ℝ := 36

/-- The time difference between upstream and downstream travel -/
def time_difference : ℝ := 1.5

theorem boat_speed_proof :
  (distance / (boat_speed - stream_speed) - distance / (boat_speed + stream_speed) = time_difference) ∧
  (boat_speed > stream_speed) :=
by sorry

end boat_speed_proof_l3356_335643


namespace angle_on_straight_line_l3356_335691

/-- Given a straight line ABC with two angles, one measuring 40° and the other measuring x°, 
    prove that the value of x is 140°. -/
theorem angle_on_straight_line (x : ℝ) : 
  x + 40 = 180 → x = 140 := by
  sorry

end angle_on_straight_line_l3356_335691


namespace fraction_numerator_l3356_335650

theorem fraction_numerator (y : ℝ) (h : y > 0) :
  ∃ x : ℝ, (x / y) * y + (3 * y) / 10 = 0.7 * y ∧ x = 2 := by
  sorry

end fraction_numerator_l3356_335650


namespace square_field_area_l3356_335655

/-- Prove that a square field with the given conditions has an area of 27889 square meters -/
theorem square_field_area (wire_cost_per_meter : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  wire_cost_per_meter = 1.4 →
  total_cost = 932.4 →
  gate_width = 1 →
  num_gates = 2 →
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    total_cost = wire_cost_per_meter * (4 * side_length - num_gates * gate_width) ∧
    side_length^2 = 27889 :=
by sorry

end square_field_area_l3356_335655


namespace chocolate_bars_distribution_l3356_335609

theorem chocolate_bars_distribution (total_bars : ℕ) (num_people : ℕ) (bars_for_two : ℕ) : 
  total_bars = 12 → num_people = 3 → bars_for_two = (total_bars / num_people) * 2 → bars_for_two = 8 := by
  sorry

end chocolate_bars_distribution_l3356_335609


namespace second_field_rows_l3356_335639

/-- Represents a corn field with a certain number of full rows -/
structure CornField where
  rows : ℕ

/-- Represents a farm with two corn fields -/
structure Farm where
  field1 : CornField
  field2 : CornField

def cobsPerRow : ℕ := 4

theorem second_field_rows (farm : Farm) 
  (h1 : farm.field1.rows = 13) 
  (h2 : farm.field1.rows * cobsPerRow + farm.field2.rows * cobsPerRow = 116) : 
  farm.field2.rows = 16 := by
  sorry

end second_field_rows_l3356_335639


namespace stating_unique_dissection_l3356_335673

/-- A type representing a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Additional structure details would go here, but are omitted for simplicity

/-- A type representing a right triangle with integer-ratio sides -/
structure IntegerRatioRightTriangle where
  -- Additional structure details would go here, but are omitted for simplicity

/-- 
Predicate indicating whether a regular n-sided polygon can be 
completely dissected into integer-ratio right triangles 
-/
def canBeDissected (n : ℕ) : Prop :=
  ∃ (p : RegularPolygon n) (triangles : Set IntegerRatioRightTriangle), 
    -- The formal definition of complete dissection would go here
    True  -- Placeholder

/-- 
Theorem stating that 4 is the only integer n ≥ 3 for which 
a regular n-sided polygon can be completely dissected into 
integer-ratio right triangles 
-/
theorem unique_dissection : 
  ∀ n : ℕ, n ≥ 3 → (canBeDissected n ↔ n = 4) := by
  sorry


end stating_unique_dissection_l3356_335673


namespace expression_evaluation_l3356_335627

theorem expression_evaluation :
  let x : ℚ := -3
  let expr := ((-2 * x^3 - 6*x) / (-2*x)) - 2*(3*x + 1)*(3*x - 1) + 7*x*(x - 1)
  expr = -64 := by
sorry

end expression_evaluation_l3356_335627


namespace cereal_eating_time_l3356_335624

def fat_rate : ℚ := 1 / 20
def thin_rate : ℚ := 1 / 30
def average_rate : ℚ := 1 / 24
def total_cereal : ℚ := 5

theorem cereal_eating_time :
  let combined_rate := fat_rate + thin_rate + average_rate
  (total_cereal / combined_rate) = 40 := by sorry

end cereal_eating_time_l3356_335624


namespace algebraic_expression_value_l3356_335663

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 2*x - 2 = 0) :
  x*(x+2) + (x+1)^2 = 5 := by
  sorry

end algebraic_expression_value_l3356_335663


namespace A_divisible_by_1980_l3356_335651

def A : ℕ := sorry

theorem A_divisible_by_1980 : 1980 ∣ A := by sorry

end A_divisible_by_1980_l3356_335651


namespace wheel_speed_proof_l3356_335640

/-- Proves that the original speed of a wheel is 7.5 mph given specific conditions -/
theorem wheel_speed_proof (wheel_circumference : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) :
  wheel_circumference = 15 →  -- circumference in feet
  speed_increase = 8 →        -- speed increase in mph
  time_decrease = 1/3 →       -- time decrease in seconds
  ∃ (original_speed : ℝ),
    original_speed = 7.5 ∧    -- original speed in mph
    (original_speed + speed_increase) * (3600 * (15 / (5280 * original_speed)) - time_decrease / 3600) =
    15 / 5280 * 3600 :=
by
  sorry


end wheel_speed_proof_l3356_335640


namespace gift_fund_equations_correct_l3356_335670

/-- Represents the crowdfunding scenario for teachers' New Year gift package. -/
structure GiftFundScenario where
  x : ℕ  -- number of teachers
  y : ℕ  -- price of the gift package

/-- The correct system of equations for the gift fund scenario. -/
def correct_equations (s : GiftFundScenario) : Prop :=
  18 * s.x = s.y + 3 ∧ 17 * s.x = s.y - 4

/-- Theorem stating that the given system of equations correctly describes the gift fund scenario. -/
theorem gift_fund_equations_correct (s : GiftFundScenario) : correct_equations s :=
sorry

end gift_fund_equations_correct_l3356_335670


namespace range_of_a_l3356_335689

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ 2 * x > a - x^2) → a < 8 := by
  sorry

end range_of_a_l3356_335689


namespace quadratic_function_properties_l3356_335698

-- Define the quadratic function f
def f : ℝ → ℝ := fun x ↦ 2 * x^2 - 4 * x + 3

-- State the theorem
theorem quadratic_function_properties :
  (∀ x : ℝ, f x ≥ 1) ∧  -- minimum value is 1
  (f 0 = 3) ∧ (f 2 = 3) ∧  -- f(0) = f(2) = 3
  (∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-3) (-1) → f x > 2 * x + 2 * m + 1) → m < 5) :=
by sorry

end quadratic_function_properties_l3356_335698


namespace largest_table_sum_l3356_335618

def numbers : List ℕ := [2, 3, 5, 7, 11, 17, 19]

def is_valid_arrangement (top : List ℕ) (left : List ℕ) : Prop :=
  top.length = 3 ∧ left.length = 3 ∧ (top ++ left).toFinset ⊆ numbers.toFinset

def table_sum (top : List ℕ) (left : List ℕ) : ℕ :=
  (top.sum * left.sum)

theorem largest_table_sum :
  ∀ (top left : List ℕ), is_valid_arrangement top left →
  table_sum top left ≤ 1024 :=
sorry

end largest_table_sum_l3356_335618


namespace park_group_problem_l3356_335631

theorem park_group_problem (girls boys : ℕ) (groups : ℕ) (group_size : ℕ) : 
  girls = 14 → 
  boys = 11 → 
  groups = 3 → 
  group_size = 25 → 
  groups * group_size = girls + boys + (groups * group_size - (girls + boys)) →
  groups * group_size - (girls + boys) = 50 := by
  sorry

#check park_group_problem

end park_group_problem_l3356_335631


namespace cubic_function_nonnegative_l3356_335688

theorem cubic_function_nonnegative (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, a * x^3 - 3 * x + 1 ≥ 0) ↔ a = 4 :=
by sorry

end cubic_function_nonnegative_l3356_335688


namespace bisector_line_l3356_335633

/-- Given two lines l₁ and l₂, and a point P, this theorem states that
    the line passing through P and (4, 0) bisects the line segment formed by
    its intersections with l₁ and l₂. -/
theorem bisector_line (P : ℝ × ℝ) (l₁ l₂ : Set (ℝ × ℝ)) :
  P = (0, 1) →
  l₁ = {(x, y) | 2*x + y - 8 = 0} →
  l₂ = {(x, y) | x - 3*y + 10 = 0} →
  ∃ (A B : ℝ × ℝ),
    A ∈ l₁ ∧
    B ∈ l₂ ∧
    (∃ (t : ℝ), A = (1-t) • P + t • (4, 0) ∧ B = (1-t) • (4, 0) + t • P) :=
by sorry

end bisector_line_l3356_335633


namespace larger_number_proof_l3356_335693

theorem larger_number_proof (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : max x y = 23 := by
  sorry

end larger_number_proof_l3356_335693


namespace annulus_area_l3356_335607

/-- The area of an annulus formed by two concentric circles -/
theorem annulus_area (B C RW : ℝ) (h1 : B > C) (h2 : B^2 - (C+5)^2 = RW^2) :
  (π * B^2) - (π * (C+5)^2) = π * RW^2 := by
  sorry

end annulus_area_l3356_335607


namespace sum_smallest_largest_prime_1_to_50_l3356_335603

theorem sum_smallest_largest_prime_1_to_50 : 
  ∃ (p q : ℕ), 
    p.Prime ∧ q.Prime ∧ 
    1 < p ∧ p ≤ 50 ∧ 
    1 < q ∧ q ≤ 50 ∧ 
    (∀ r : ℕ, r.Prime → 1 < r → r ≤ 50 → p ≤ r ∧ r ≤ q) ∧ 
    p + q = 49 :=
by sorry

end sum_smallest_largest_prime_1_to_50_l3356_335603


namespace max_xy_geometric_mean_l3356_335644

theorem max_xy_geometric_mean (x y : ℝ) : 
  x^2 = (1 + 2*y) * (1 - 2*y) → 
  ∃ (k : ℝ), k = x*y ∧ k ≤ (1/4 : ℝ) ∧ 
  ∃ (x₀ y₀ : ℝ), x₀^2 = (1 + 2*y₀) * (1 - 2*y₀) ∧ x₀ * y₀ = (1/4 : ℝ) :=
by sorry

end max_xy_geometric_mean_l3356_335644


namespace renovation_calculation_l3356_335679

/-- Represents the dimensions and characteristics of a bedroom --/
structure Bedroom where
  length : ℝ
  width : ℝ
  height : ℝ
  unpaintable_area : ℝ
  fixed_furniture_area : ℝ

/-- Calculates the total area to be painted in all bedrooms --/
def total_paintable_area (b : Bedroom) (num_bedrooms : ℕ) : ℝ :=
  num_bedrooms * (2 * (b.length * b.height + b.width * b.height) - b.unpaintable_area)

/-- Calculates the total carpet area for all bedrooms --/
def total_carpet_area (b : Bedroom) (num_bedrooms : ℕ) : ℝ :=
  num_bedrooms * (b.length * b.width - b.fixed_furniture_area)

/-- Theorem stating the correct paintable area and carpet area --/
theorem renovation_calculation (b : Bedroom) (h1 : b.length = 14)
    (h2 : b.width = 11) (h3 : b.height = 9) (h4 : b.unpaintable_area = 70)
    (h5 : b.fixed_furniture_area = 24) :
    total_paintable_area b 4 = 1520 ∧ total_carpet_area b 4 = 520 := by
  sorry


end renovation_calculation_l3356_335679


namespace cube_surface_area_l3356_335634

theorem cube_surface_area (volume : ℝ) (h : volume = 1331) :
  let side := (volume ^ (1/3 : ℝ))
  6 * side^2 = 726 := by
sorry

end cube_surface_area_l3356_335634


namespace circle_equation_circle_properties_l3356_335637

theorem circle_equation (x y : ℝ) : 
  (x^2 + y^2 + 2*x - 4*y - 6 = 0) ↔ ((x + 1)^2 + (y - 2)^2 = 11) :=
by sorry

theorem circle_properties :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-1, 2) ∧ 
    radius = Real.sqrt 11 ∧
    ∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y - 6 = 0 ↔ 
      (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_equation_circle_properties_l3356_335637


namespace cookies_left_l3356_335629

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of cookies Meena starts with -/
def initial_dozens : ℕ := 5

/-- The number of dozens of cookies sold to Mr. Stone -/
def sold_to_stone : ℕ := 2

/-- The number of cookies bought by Brock -/
def bought_by_brock : ℕ := 7

/-- Katy buys twice as many cookies as Brock -/
def bought_by_katy : ℕ := 2 * bought_by_brock

/-- The theorem stating that Meena has 15 cookies left -/
theorem cookies_left : 
  initial_dozens * dozen - sold_to_stone * dozen - bought_by_brock - bought_by_katy = 15 := by
  sorry


end cookies_left_l3356_335629


namespace quadratic_inequality_solution_l3356_335675

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + 2 < 0 ↔ 1 < x ∧ x < 2) →
  a = 3 := by
sorry

end quadratic_inequality_solution_l3356_335675


namespace purchase_price_problem_l3356_335612

/-- A linear function relating purchase quantity (y) to unit price (x) -/
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem purchase_price_problem (k b : ℝ) 
  (h1 : 1000 = linear_function k b 800)
  (h2 : 2000 = linear_function k b 700) :
  linear_function k b 5000 = 400 := by sorry

end purchase_price_problem_l3356_335612


namespace ceiling_negative_sqrt_theorem_l3356_335601

theorem ceiling_negative_sqrt_theorem :
  ⌈-Real.sqrt ((64 : ℝ) / 9 - 1)⌉ = -2 := by
  sorry

end ceiling_negative_sqrt_theorem_l3356_335601


namespace other_number_proof_l3356_335626

theorem other_number_proof (A B : ℕ) (hA : A = 24) (hHCF : Nat.gcd A B = 17) (hLCM : Nat.lcm A B = 312) : B = 221 := by
  sorry

end other_number_proof_l3356_335626


namespace residue_calculation_l3356_335677

theorem residue_calculation : (204 * 15 - 16 * 8 + 5) % 17 = 12 := by
  sorry

end residue_calculation_l3356_335677


namespace sample_size_is_hundred_l3356_335605

/-- Represents a statistical study on student scores -/
structure ScoreStudy where
  population_size : ℕ
  extracted_size : ℕ

/-- Defines the sample size of a score study -/
def sample_size (study : ScoreStudy) : ℕ := study.extracted_size

/-- Theorem stating that for the given study, the sample size is 100 -/
theorem sample_size_is_hundred (study : ScoreStudy) 
  (h1 : study.population_size = 1000)
  (h2 : study.extracted_size = 100) : 
  sample_size study = 100 := by
  sorry

#check sample_size_is_hundred

end sample_size_is_hundred_l3356_335605


namespace simplify_and_evaluate_l3356_335657

theorem simplify_and_evaluate : 
  let x : ℤ := -2
  (2 * x + 1) * (x - 2) - (2 - x)^2 = -4 := by
  sorry

end simplify_and_evaluate_l3356_335657


namespace erica_pie_percentage_l3356_335647

theorem erica_pie_percentage :
  ∀ (apple_fraction cherry_fraction : ℚ),
    apple_fraction = 1/5 →
    cherry_fraction = 3/4 →
    (apple_fraction + cherry_fraction) * 100 = 95 := by
  sorry

end erica_pie_percentage_l3356_335647


namespace intersection_sum_reciprocal_constant_l3356_335619

/-- The curve C representing the locus of centers of the moving circle M -/
def curve_C (x y : ℝ) : Prop :=
  x > 0 ∧ x^2 / 4 - y^2 / 12 = 1

/-- A point on the curve C -/
structure PointOnC where
  x : ℝ
  y : ℝ
  on_curve : curve_C x y

/-- The origin point O -/
def O : ℝ × ℝ := (0, 0)

/-- Distance squared between two points -/
def dist_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem intersection_sum_reciprocal_constant
  (P Q : PointOnC)
  (h_perp : (P.x * Q.x + P.y * Q.y = 0)) : -- OP ⊥ OQ condition
  1 / dist_squared O (P.x, P.y) + 1 / dist_squared O (Q.x, Q.y) = 1/6 :=
sorry

end intersection_sum_reciprocal_constant_l3356_335619


namespace loan_amount_to_B_l3356_335662

/-- Calculates the simple interest given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem loan_amount_to_B (amountToC : ℚ) (timeB timeC : ℚ) (rate : ℚ) (totalInterest : ℚ) :
  amountToC = 3000 →
  timeB = 2 →
  timeC = 4 →
  rate = 8 →
  totalInterest = 1760 →
  ∃ amountToB : ℚ, 
    simpleInterest amountToB rate timeB + simpleInterest amountToC rate timeC = totalInterest ∧
    amountToB = 5000 := by
  sorry

end loan_amount_to_B_l3356_335662


namespace volleyball_team_starters_l3356_335638

/-- The number of ways to choose k items from n distinguishable items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players on the volleyball team -/
def total_players : ℕ := 16

/-- The number of starters to be chosen -/
def num_starters : ℕ := 6

/-- Theorem: The number of ways to choose 6 starters from 16 players is 8008 -/
theorem volleyball_team_starters : choose total_players num_starters = 8008 := by
  sorry

end volleyball_team_starters_l3356_335638


namespace product_increase_by_2016_l3356_335659

theorem product_increase_by_2016 : ∃ (a b c : ℕ), 
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2016 :=
sorry

end product_increase_by_2016_l3356_335659


namespace sara_quarters_l3356_335674

theorem sara_quarters (initial : ℕ) : 
  initial + 49 = 70 → initial = 21 := by
  sorry

end sara_quarters_l3356_335674


namespace ratio_of_averages_l3356_335632

theorem ratio_of_averages (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (4 + 20 + x) / 3 = (y + 16) / 2) :
  x / y = 3 / 2 := by
sorry

end ratio_of_averages_l3356_335632


namespace max_shipping_cost_l3356_335606

/-- The maximum shipping cost per unit for an electronic component manufacturer --/
theorem max_shipping_cost (production_cost : ℝ) (fixed_costs : ℝ) (units : ℕ) (selling_price : ℝ)
  (h1 : production_cost = 80)
  (h2 : fixed_costs = 16500)
  (h3 : units = 150)
  (h4 : selling_price = 193.33) :
  ∃ (shipping_cost : ℝ), shipping_cost ≤ 3.33 ∧
    units * (production_cost + shipping_cost) + fixed_costs ≤ units * selling_price :=
by sorry

end max_shipping_cost_l3356_335606


namespace button_probability_l3356_335678

/-- Represents a jar containing buttons of two colors -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- Represents the state of two jars after button transfer -/
structure JarState where
  jarA : Jar
  jarB : Jar

def initialJarA : Jar := { red := 7, blue := 9 }

def buttonTransfer (initial : Jar) : JarState :=
  { jarA := { red := initial.red - 3, blue := initial.blue - 2 },
    jarB := { red := 3, blue := 2 } }

def probability_red (jar : Jar) : ℚ :=
  jar.red / (jar.red + jar.blue)

theorem button_probability (initial : Jar := initialJarA) :
  let final := buttonTransfer initial
  let probA := probability_red final.jarA
  let probB := probability_red final.jarB
  probA * probB = 12 / 55 := by
  sorry

end button_probability_l3356_335678


namespace polynomial_root_sum_l3356_335694

theorem polynomial_root_sum (a b c d e : ℝ) : 
  a ≠ 0 → 
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = 5 ∨ x = -3 ∨ x = 2 ∨ x = (-(b + c + d) / a - 4)) →
  (b + c + d) / a = -6 := by
  sorry

end polynomial_root_sum_l3356_335694


namespace janets_dress_pockets_janets_dress_pockets_correct_l3356_335628

theorem janets_dress_pockets : ℕ → ℕ
  | total_dresses =>
    let dresses_with_pockets := total_dresses / 2
    let dresses_with_two_pockets := dresses_with_pockets / 3
    let dresses_with_three_pockets := dresses_with_pockets - dresses_with_two_pockets
    let total_pockets := dresses_with_two_pockets * 2 + dresses_with_three_pockets * 3
    total_pockets

theorem janets_dress_pockets_correct : janets_dress_pockets 24 = 32 := by
  sorry

end janets_dress_pockets_janets_dress_pockets_correct_l3356_335628


namespace average_of_three_numbers_l3356_335600

theorem average_of_three_numbers (y : ℝ) : 
  (15 + 28 + y) / 3 = 25 → y = 32 := by
  sorry

end average_of_three_numbers_l3356_335600


namespace triangle_sin_A_l3356_335648

theorem triangle_sin_A (a b : ℝ) (sinB : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : sinB = 2/3) :
  let sinA := a * sinB / b
  sinA = 1/2 := by sorry

end triangle_sin_A_l3356_335648


namespace least_number_with_remainders_l3356_335610

theorem least_number_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 41 = 5) ∧ 
  (n % 23 = 5) ∧ 
  (∀ m : ℕ, m > 0 → m % 41 = 5 → m % 23 = 5 → m ≥ n) ∧
  n = 948 := by
sorry

end least_number_with_remainders_l3356_335610


namespace quadratic_inequality_solution_set_l3356_335671

/-- Given a quadratic inequality x² + bx + c < 0 with solution set (-1, 2),
    prove that bx² + x + c < 0 has solution set ℝ -/
theorem quadratic_inequality_solution_set
  (b c : ℝ)
  (h : Set.Ioo (-1 : ℝ) 2 = {x | x^2 + b*x + c < 0}) :
  Set.univ = {x | b*x^2 + x + c < 0} :=
sorry

end quadratic_inequality_solution_set_l3356_335671


namespace inverse_inequality_l3356_335676

theorem inverse_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end inverse_inequality_l3356_335676


namespace perimeter_ratio_of_squares_l3356_335615

theorem perimeter_ratio_of_squares (s1 s2 : Real) (h : s1^2 / s2^2 = 16 / 25) :
  (4 * s1) / (4 * s2) = 4 / 5 := by sorry

end perimeter_ratio_of_squares_l3356_335615


namespace clara_score_reversal_l3356_335652

theorem clara_score_reversal (a b : ℕ) :
  (∃ (second_game third_game : ℕ),
    second_game = 45 ∧
    third_game = 54 ∧
    (10 * b + a) + second_game + third_game = (10 * a + b) + second_game + third_game + 132) →
  (10 * b + a) - (10 * a + b) = 126 :=
by sorry

end clara_score_reversal_l3356_335652


namespace tan_value_for_given_sin_cos_sum_l3356_335658

theorem tan_value_for_given_sin_cos_sum (θ : Real) 
  (h1 : Real.sin θ + Real.cos θ = Real.sqrt 5 / 5)
  (h2 : θ ∈ Set.Icc 0 Real.pi) : 
  Real.tan θ = -2 := by
sorry

end tan_value_for_given_sin_cos_sum_l3356_335658


namespace auto_credit_percentage_l3356_335623

/-- Given that automobile finance companies extended $57 billion of credit, which is 1/3 of the
    total automobile installment credit, and the total consumer installment credit outstanding
    is $855 billion, prove that automobile installment credit accounts for 20% of all outstanding
    consumer installment credit. -/
theorem auto_credit_percentage (finance_company_credit : ℝ) (total_consumer_credit : ℝ)
    (h1 : finance_company_credit = 57)
    (h2 : total_consumer_credit = 855)
    (h3 : finance_company_credit = (1/3) * (3 * finance_company_credit)) :
    (3 * finance_company_credit) / total_consumer_credit = 0.2 := by
  sorry

end auto_credit_percentage_l3356_335623


namespace circle_area_when_equal_to_circumference_l3356_335654

/-- Given a circle where the circumference and area are numerically equal,
    and the diameter is 4, prove that the area is 4π. -/
theorem circle_area_when_equal_to_circumference (r : ℝ) : 
  2 * Real.pi * r = Real.pi * r^2 →   -- Circumference equals area
  4 = 2 * r →                         -- Diameter is 4
  Real.pi * r^2 = 4 * Real.pi :=      -- Area is 4π
by
  sorry

#check circle_area_when_equal_to_circumference

end circle_area_when_equal_to_circumference_l3356_335654


namespace pqr_sum_fraction_prime_l3356_335635

theorem pqr_sum_fraction_prime (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  (∃ k : ℕ, p * q * r = k * (p + q + r)) → 
  Nat.Prime (p * q * r / (p + q + r)) :=
by sorry

end pqr_sum_fraction_prime_l3356_335635


namespace number_difference_l3356_335685

theorem number_difference (L S : ℤ) (h1 : ∃ X, L = 2*S + X) (h2 : L + S = 27) (h3 : L = 19) :
  L - 2*S = 3 := by
  sorry

end number_difference_l3356_335685


namespace min_touches_equal_total_buttons_l3356_335686

/-- Represents a button in the array -/
inductive ButtonState
| OFF
| ON

/-- Represents the array of buttons -/
def ButtonArray := Fin 40 → Fin 50 → ButtonState

/-- The initial state of the array where all buttons are OFF -/
def initialState : ButtonArray := λ _ _ => ButtonState.OFF

/-- The final state of the array where all buttons are ON -/
def finalState : ButtonArray := λ _ _ => ButtonState.ON

/-- Represents a touch operation on a button -/
def touch (array : ButtonArray) (row : Fin 40) (col : Fin 50) : ButtonArray :=
  λ r c => if r = row ∨ c = col then
    match array r c with
    | ButtonState.OFF => ButtonState.ON
    | ButtonState.ON => ButtonState.OFF
  else
    array r c

/-- The theorem stating that the minimum number of touches to switch all buttons from OFF to ON
    is equal to the total number of buttons in the array -/
theorem min_touches_equal_total_buttons :
  ∃ (touches : List (Fin 40 × Fin 50)),
    touches.length = 40 * 50 ∧
    touches.foldl (λ acc (r, c) => touch acc r c) initialState = finalState ∧
    ∀ (touches' : List (Fin 40 × Fin 50)),
      touches'.foldl (λ acc (r, c) => touch acc r c) initialState = finalState →
      touches'.length ≥ 40 * 50 := by
  sorry


end min_touches_equal_total_buttons_l3356_335686


namespace tv_price_change_l3356_335608

theorem tv_price_change (P : ℝ) (x : ℝ) : 
  P * (1 - x / 100) * 1.3 = P * 1.17 → x = 10 := by
  sorry

end tv_price_change_l3356_335608


namespace complex_magnitude_problem_l3356_335604

theorem complex_magnitude_problem (m : ℂ) :
  (((4 : ℂ) + m * Complex.I) / ((1 : ℂ) + 2 * Complex.I)).im = 0 →
  Complex.abs (m + 6 * Complex.I) = 10 := by
  sorry

end complex_magnitude_problem_l3356_335604


namespace expression_always_defined_l3356_335664

theorem expression_always_defined (x : ℝ) : 
  ∃ y : ℝ, y = x^2 / (2*x^2 + 1) :=
by
  sorry

end expression_always_defined_l3356_335664


namespace min_value_theorem_l3356_335699

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 2/b ≥ (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end min_value_theorem_l3356_335699


namespace problem_solution_l3356_335666

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

noncomputable def f_derivative (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := f_derivative a b x * Real.exp x

theorem problem_solution (a b : ℝ) :
  f_derivative a b 1 = 2*a ∧ f_derivative a b 2 = -b →
  (a = -3/2 ∧ b = -3) ∧
  (∀ x, g a b x ≥ g a b 1) ∧
  (∀ x, g a b x ≤ g a b (-2)) ∧
  g a b 1 = -3 * Real.exp 1 ∧
  g a b (-2) = 15 * Real.exp (-2) :=
by sorry

end problem_solution_l3356_335666


namespace intersection_segment_length_l3356_335697

/-- The length of the line segment AB, where A and B are the intersection points
    of the line y = √3 x and the circle (x + √3)² + (y + 2)² = 1, is equal to √3. -/
theorem intersection_segment_length :
  let line := {p : ℝ × ℝ | p.2 = Real.sqrt 3 * p.1}
  let circle := {p : ℝ × ℝ | (p.1 + Real.sqrt 3)^2 + (p.2 + 2)^2 = 1}
  let intersection := {p : ℝ × ℝ | p ∈ line ∩ circle}
  ∃ A B : ℝ × ℝ, A ∈ intersection ∧ B ∈ intersection ∧ A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 3 :=
by sorry

end intersection_segment_length_l3356_335697


namespace eccentricity_of_special_ellipse_l3356_335684

/-- Theorem: Eccentricity of a special ellipse -/
theorem eccentricity_of_special_ellipse 
  (a b c : ℝ) 
  (h_pos : a > b ∧ b > 0) 
  (h_c : c = Real.sqrt (a^2 - b^2)) 
  (P : ℝ × ℝ) 
  (h_P_on_ellipse : (P.1^2 / a^2) + (P.2^2 / b^2) = 1) 
  (l : ℝ → ℝ) 
  (h_l : ∀ x, l x = -a^2 / c) 
  (Q : ℝ × ℝ) 
  (h_PQ_perp_l : (P.1 - Q.1) * 1 + (P.2 - Q.2) * 0 = 0) 
  (F : ℝ × ℝ) 
  (h_F : F = (-c, 0)) 
  (h_isosceles : (P.1 - F.1)^2 + (P.2 - F.2)^2 = (Q.1 - F.1)^2 + (Q.2 - F.2)^2) :
  c / a = Real.sqrt 2 / 2 := by
sorry

end eccentricity_of_special_ellipse_l3356_335684


namespace sqrt_3_irrational_l3356_335622

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end sqrt_3_irrational_l3356_335622


namespace clarinet_rate_is_40_l3356_335696

/-- The hourly rate for clarinet lessons --/
def clarinet_rate : ℝ := 40

/-- The number of hours of clarinet lessons per week --/
def clarinet_hours_per_week : ℝ := 3

/-- The number of hours of piano lessons per week --/
def piano_hours_per_week : ℝ := 5

/-- The hourly rate for piano lessons --/
def piano_rate : ℝ := 28

/-- The difference in annual cost between piano and clarinet lessons --/
def annual_cost_difference : ℝ := 1040

/-- The number of weeks in a year --/
def weeks_per_year : ℝ := 52

theorem clarinet_rate_is_40 : 
  piano_hours_per_week * piano_rate * weeks_per_year = 
  clarinet_hours_per_week * clarinet_rate * weeks_per_year + annual_cost_difference :=
by sorry

end clarinet_rate_is_40_l3356_335696


namespace octagon_area_l3356_335611

/-- The area of an octagon formed by removing equilateral triangles from the corners of a square -/
theorem octagon_area (square_side : ℝ) (triangle_side : ℝ) : 
  square_side = 1 + Real.sqrt 3 →
  triangle_side = 1 →
  let square_area := square_side ^ 2
  let triangle_area := (Real.sqrt 3 / 4) * triangle_side ^ 2
  let octagon_area := square_area - 4 * triangle_area
  octagon_area = 4 + Real.sqrt 3 := by
sorry

end octagon_area_l3356_335611


namespace common_power_theorem_l3356_335621

theorem common_power_theorem (a b x y : ℕ) : 
  a > 1 → b > 1 → x > 1 → y > 1 → 
  Nat.gcd a b = 1 → 
  x^a = y^b → 
  ∃ n : ℕ, n > 1 ∧ x = n^b ∧ y = n^a := by
sorry

end common_power_theorem_l3356_335621


namespace smaller_solution_of_quadratic_l3356_335672

theorem smaller_solution_of_quadratic (x : ℝ) :
  x^2 - 13*x + 36 = 0 →
  (∃ y : ℝ, y ≠ x ∧ y^2 - 13*y + 36 = 0) →
  (∀ z : ℝ, z^2 - 13*z + 36 = 0 → z ≥ 4) ∧
  (∃ w : ℝ, w^2 - 13*w + 36 = 0 ∧ w = 4) :=
by sorry

end smaller_solution_of_quadratic_l3356_335672


namespace vanessa_music_files_l3356_335616

/-- The number of music files Vanessa initially had -/
def initial_music_files : ℕ := 13

/-- The number of video files Vanessa initially had -/
def video_files : ℕ := 30

/-- The number of files deleted -/
def deleted_files : ℕ := 10

/-- The number of files remaining after deletion -/
def remaining_files : ℕ := 33

theorem vanessa_music_files :
  initial_music_files + video_files = remaining_files + deleted_files :=
by sorry

end vanessa_music_files_l3356_335616


namespace sqrt_sum_theorem_l3356_335681

theorem sqrt_sum_theorem (a : ℝ) (h : a + 1/a = 3) : 
  Real.sqrt a + 1 / Real.sqrt a = Real.sqrt 5 := by
sorry

end sqrt_sum_theorem_l3356_335681


namespace system_solution_l3356_335690

theorem system_solution : ∃ (x y : ℚ), 
  (x - 30) / 3 = (2 * y + 7) / 4 ∧ 
  x - y = 10 ∧ 
  x = -81/2 ∧ 
  y = -101/2 := by
  sorry

end system_solution_l3356_335690


namespace seating_theorem_l3356_335687

/-- The number of seating arrangements for 3 people in 6 seats with exactly two adjacent empty seats -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) (adjacent_empty : ℕ) : ℕ :=
  2 * Nat.factorial (people + 1)

theorem seating_theorem :
  seating_arrangements 6 3 2 = 48 :=
by sorry

end seating_theorem_l3356_335687


namespace sharon_coffee_cost_l3356_335661

/-- Calculates the total cost of coffee pods for Sharon's vacation -/
def coffee_cost (vacation_days : ℕ) (light_daily : ℕ) (medium_daily : ℕ) (decaf_daily : ℕ)
  (light_box_qty : ℕ) (medium_box_qty : ℕ) (decaf_box_qty : ℕ)
  (light_box_price : ℕ) (medium_box_price : ℕ) (decaf_box_price : ℕ) : ℕ :=
  let light_pods := vacation_days * light_daily
  let medium_pods := vacation_days * medium_daily
  let decaf_pods := vacation_days * decaf_daily
  let light_boxes := (light_pods + light_box_qty - 1) / light_box_qty
  let medium_boxes := (medium_pods + medium_box_qty - 1) / medium_box_qty
  let decaf_boxes := (decaf_pods + decaf_box_qty - 1) / decaf_box_qty
  light_boxes * light_box_price + medium_boxes * medium_box_price + decaf_boxes * decaf_box_price

/-- Theorem stating that the total cost for Sharon's vacation coffee is $80 -/
theorem sharon_coffee_cost :
  coffee_cost 40 2 1 1 20 25 30 10 12 8 = 80 :=
by sorry


end sharon_coffee_cost_l3356_335661


namespace class_average_height_l3356_335683

/-- The average height of a class of girls -/
theorem class_average_height 
  (total_girls : ℕ) 
  (group1_girls : ℕ) 
  (group1_avg_height : ℝ) 
  (group2_avg_height : ℝ) 
  (h1 : total_girls = 40)
  (h2 : group1_girls = 30)
  (h3 : group1_avg_height = 160)
  (h4 : group2_avg_height = 156) :
  (group1_girls * group1_avg_height + (total_girls - group1_girls) * group2_avg_height) / total_girls = 159 := by
  sorry


end class_average_height_l3356_335683


namespace white_car_rental_cost_l3356_335646

/-- Represents the cost of renting a white car per minute -/
def white_car_cost : ℝ := 2

/-- Represents the number of red cars -/
def red_cars : ℕ := 3

/-- Represents the number of white cars -/
def white_cars : ℕ := 2

/-- Represents the cost of renting a red car per minute -/
def red_car_cost : ℝ := 3

/-- Represents the total rental time in minutes -/
def rental_time : ℕ := 3 * 60

/-- Represents the total earnings -/
def total_earnings : ℝ := 2340

theorem white_car_rental_cost :
  red_cars * red_car_cost * rental_time + white_cars * white_car_cost * rental_time = total_earnings :=
by sorry

end white_car_rental_cost_l3356_335646


namespace inequality_problem_l3356_335642

/-- Given positive real numbers a and b such that 1/a + 1/b = 2√2, 
    prove the minimum value of a² + b² and the value of ab under certain conditions. -/
theorem inequality_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h : 1/a + 1/b = 2 * Real.sqrt 2) : 
  (∃ (min : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 2 * Real.sqrt 2 → x^2 + y^2 ≥ min ∧ 
    a^2 + b^2 = min) ∧ 
  ((a - b)^2 ≥ 4 * (a*b)^3 → a * b = 1) := by
  sorry

end inequality_problem_l3356_335642


namespace range_of_y_l3356_335617

theorem range_of_y (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by
  sorry

end range_of_y_l3356_335617


namespace balance_weights_theorem_l3356_335613

/-- The double factorial of an odd number -/
def oddDoubleFactorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | k + 2 => (k + 1) * oddDoubleFactorial k

/-- The number of ways to place weights on a balance -/
def balanceWeights (n : ℕ) : ℕ :=
  oddDoubleFactorial (2 * n - 1)

/-- Theorem: The number of ways to place n weights (2^0, 2^1, ..., 2^(n-1)) on a balance,
    such that the right pan is never heavier than the left pan, is equal to (2n-1)!! -/
theorem balance_weights_theorem (n : ℕ) (h : n > 0) :
  balanceWeights n = oddDoubleFactorial (2 * n - 1) :=
by
  sorry

#eval balanceWeights 3  -- Expected output: 15
#eval balanceWeights 4  -- Expected output: 105

end balance_weights_theorem_l3356_335613


namespace cleaning_earnings_proof_l3356_335641

/-- Calculates the total earnings for cleaning all rooms in a building -/
def total_earnings (floors : ℕ) (rooms_per_floor : ℕ) (hours_per_room : ℕ) (dollars_per_hour : ℕ) : ℕ :=
  floors * rooms_per_floor * hours_per_room * dollars_per_hour

/-- Proves that the total earnings for cleaning the given building is $32,000 -/
theorem cleaning_earnings_proof :
  total_earnings 10 20 8 20 = 32000 := by
  sorry

end cleaning_earnings_proof_l3356_335641


namespace symmetric_points_sum_l3356_335630

/-- Given two points A and B that are symmetric about the y-axis, 
    prove that the sum of the offsets from A's coordinates equals 1. -/
theorem symmetric_points_sum (m n : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    A = (1 + m, 1 - n) ∧ 
    B = (-3, 2) ∧ 
    (A.1 = -B.1) ∧ 
    (A.2 = B.2)) →
  m + n = 1 := by
sorry

end symmetric_points_sum_l3356_335630


namespace min_cross_section_area_and_volume_ratio_l3356_335656

/-- A regular triangular pyramid inscribed in a sphere -/
structure RegularTriangularPyramid (R : ℝ) where
  /-- The radius of the circumscribing sphere -/
  radius : ℝ
  /-- The height of the pyramid -/
  height : ℝ
  /-- The height is 4R/3 -/
  height_eq : height = 4 * R / 3

/-- A cross-section of the pyramid passing through a median of its base -/
structure CrossSection (R : ℝ) (pyramid : RegularTriangularPyramid R) where
  /-- The area of the cross-section -/
  area : ℝ
  /-- The ratio of the volumes of the two parts divided by the cross-section -/
  volume_ratio : ℚ × ℚ

/-- The theorem stating the minimum area of the cross-section and the volume ratio -/
theorem min_cross_section_area_and_volume_ratio (R : ℝ) (pyramid : RegularTriangularPyramid R) :
  ∃ (cs : CrossSection R pyramid),
    cs.area = 2 * Real.sqrt 2 / Real.sqrt 33 * R^2 ∧
    cs.volume_ratio = (3, 19) ∧
    ∀ (other_cs : CrossSection R pyramid), cs.area ≤ other_cs.area :=
sorry

end min_cross_section_area_and_volume_ratio_l3356_335656


namespace max_elevation_is_650_l3356_335625

/-- The elevation function of a ball thrown vertically upward -/
def s (t : ℝ) : ℝ := 100 * t - 4 * t^2 + 25

/-- Theorem: The maximum elevation reached by the ball is 650 feet -/
theorem max_elevation_is_650 : 
  ∃ t_max : ℝ, ∀ t : ℝ, s t ≤ s t_max ∧ s t_max = 650 := by
  sorry

end max_elevation_is_650_l3356_335625


namespace hawks_lost_percentage_l3356_335668

/-- Represents a team's game statistics -/
structure TeamStats where
  total_games : ℕ
  win_ratio : ℚ
  loss_ratio : ℚ

/-- Calculates the percentage of games lost -/
def percent_lost (stats : TeamStats) : ℚ :=
  (stats.loss_ratio / (stats.win_ratio + stats.loss_ratio)) * 100

theorem hawks_lost_percentage :
  let hawks : TeamStats := {
    total_games := 64,
    win_ratio := 5,
    loss_ratio := 3
  }
  percent_lost hawks = 37.5 := by sorry

end hawks_lost_percentage_l3356_335668


namespace land_value_calculation_l3356_335649

/-- Proves that if Blake gives Connie $20,000, and the value of the land Connie buys triples in one year,
    then half of the land's value after one year is $30,000. -/
theorem land_value_calculation (initial_amount : ℕ) (value_multiplier : ℕ) : 
  initial_amount = 20000 → value_multiplier = 3 → 
  (initial_amount * value_multiplier) / 2 = 30000 := by
  sorry

end land_value_calculation_l3356_335649


namespace alice_savings_l3356_335636

/-- Alice's savings calculation --/
theorem alice_savings (sales : ℝ) (basic_salary : ℝ) (commission_rate : ℝ) (savings_rate : ℝ) :
  sales = 2500 →
  basic_salary = 240 →
  commission_rate = 0.02 →
  savings_rate = 0.1 →
  (basic_salary + sales * commission_rate) * savings_rate = 29 := by
  sorry

end alice_savings_l3356_335636


namespace sara_pumpkins_left_l3356_335680

def pumpkins_left (initial : ℕ) (eaten_by_rabbits : ℕ) (eaten_by_raccoons : ℕ) (given_away : ℕ) : ℕ :=
  initial - eaten_by_rabbits - eaten_by_raccoons - given_away

theorem sara_pumpkins_left : 
  pumpkins_left 43 23 5 7 = 8 := by sorry

end sara_pumpkins_left_l3356_335680


namespace min_grades_for_average_l3356_335614

theorem min_grades_for_average (n : ℕ) (s : ℕ) : n ≥ 51 ↔ 
  (∃ s : ℕ, (4.5 : ℝ) < (s : ℝ) / (n : ℝ) ∧ (s : ℝ) / (n : ℝ) < 4.51) :=
sorry

end min_grades_for_average_l3356_335614


namespace jenna_bob_difference_l3356_335667

/-- Prove that Jenna has $20 less than Bob in her account given the conditions. -/
theorem jenna_bob_difference (bob phil jenna : ℕ) : 
  bob = 60 → 
  phil = bob / 3 → 
  jenna = 2 * phil → 
  bob - jenna = 20 := by
sorry

end jenna_bob_difference_l3356_335667


namespace double_root_values_l3356_335660

theorem double_root_values (b₃ b₂ b₁ : ℤ) (r : ℤ) :
  (∀ x : ℝ, x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 72 = (x - r : ℝ)^2 * ((x - r)^2 + c * (x - r) + d))
  → (r = -6 ∨ r = -3 ∨ r = -2 ∨ r = -1 ∨ r = 1 ∨ r = 2 ∨ r = 3 ∨ r = 6) :=
by sorry

end double_root_values_l3356_335660


namespace polynomial_division_remainder_l3356_335682

theorem polynomial_division_remainder : 
  let dividend := fun z : ℝ => 4 * z^3 + 5 * z^2 - 20 * z + 7
  let divisor := fun z : ℝ => 4 * z - 3
  let quotient := fun z : ℝ => z^2 + 2 * z + 1/4
  let remainder := fun z : ℝ => -15 * z + 31/4
  ∀ z : ℝ, dividend z = divisor z * quotient z + remainder z :=
by
  sorry

end polynomial_division_remainder_l3356_335682


namespace quadratic_function_properties_l3356_335602

/-- A quadratic function with real coefficients -/
def f (a b x : ℝ) := x^2 + a*x + b

/-- The theorem statement -/
theorem quadratic_function_properties
  (a b : ℝ)
  (h_range : Set.range (f a b) = Set.Ici 0)
  (c m : ℝ)
  (h_solution_set : { x | f a b x < m } = Set.Ioo c (c + 2*Real.sqrt 2)) :
  m = 2 ∧
  ∃ (min_value : ℝ),
    min_value = 3 + 2*Real.sqrt 2 ∧
    ∀ (x y : ℝ), x > 1 → y > 0 → x + y = m →
      1 / (x - 1) + 2 / y ≥ min_value :=
by sorry

end quadratic_function_properties_l3356_335602


namespace annual_interest_rate_is_33_point_33_percent_l3356_335669

/-- Represents the banker's gain in rupees -/
def bankers_gain : ℝ := 360

/-- Represents the banker's discount in rupees -/
def bankers_discount : ℝ := 1360

/-- Represents the time period in years -/
def time : ℝ := 3

/-- Calculates the true discount based on banker's discount and banker's gain -/
def true_discount : ℝ := bankers_discount - bankers_gain

/-- Calculates the present value based on banker's discount and banker's gain -/
def present_value : ℝ := bankers_discount - bankers_gain

/-- Calculates the face value as the sum of present value and true discount -/
def face_value : ℝ := present_value + true_discount

/-- Theorem stating that the annual interest rate is 100/3 percent -/
theorem annual_interest_rate_is_33_point_33_percent :
  ∃ (r : ℝ), r = 100 / 3 ∧ true_discount = (present_value * r * time) / 100 :=
sorry

end annual_interest_rate_is_33_point_33_percent_l3356_335669
