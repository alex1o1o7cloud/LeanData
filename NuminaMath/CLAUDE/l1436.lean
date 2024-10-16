import Mathlib

namespace NUMINAMATH_CALUDE_bridget_block_collection_l1436_143669

/-- The number of groups of blocks in Bridget's collection -/
def num_groups : ℕ := 82

/-- The number of blocks in each group -/
def blocks_per_group : ℕ := 10

/-- The total number of blocks in Bridget's collection -/
def total_blocks : ℕ := num_groups * blocks_per_group

theorem bridget_block_collection :
  total_blocks = 820 :=
by sorry

end NUMINAMATH_CALUDE_bridget_block_collection_l1436_143669


namespace NUMINAMATH_CALUDE_parabola_intersection_midpoint_l1436_143680

/-- Given two parabolas that intersect at points A and B, prove that if the sum of the x-coordinate
    and y-coordinate of the midpoint of AB is 2017, then c = 4031. -/
theorem parabola_intersection_midpoint (c : ℝ) : 
  let f (x : ℝ) := x^2 - 2*x - 3
  let g (x : ℝ) := -x^2 + 4*x + c
  ∃ A B : ℝ × ℝ, 
    (f A.1 = A.2 ∧ g A.1 = A.2) ∧ 
    (f B.1 = B.2 ∧ g B.1 = B.2) ∧
    ((A.1 + B.1) / 2 + (A.2 + B.2) / 2 = 2017) →
  c = 4031 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_midpoint_l1436_143680


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l1436_143663

/-- Given two vectors a and b in R², where a = (1, m) and b = (3, -2),
    if a + b is perpendicular to b, then m = 8. -/
theorem vector_perpendicular_condition (m : ℝ) :
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (3, -2)
  (a.1 + b.1, a.2 + b.2) • b = 0 → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l1436_143663


namespace NUMINAMATH_CALUDE_sum_of_powers_equals_power_of_sum_l1436_143654

theorem sum_of_powers_equals_power_of_sum : 5^5 + 5^5 + 5^5 + 5^5 = 5^6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_equals_power_of_sum_l1436_143654


namespace NUMINAMATH_CALUDE_squirrel_calories_proof_l1436_143691

/-- The number of squirrels Brandon can catch in 1 hour -/
def squirrels_per_hour : ℕ := 6

/-- The number of rabbits Brandon can catch in 1 hour -/
def rabbits_per_hour : ℕ := 2

/-- The number of calories in each rabbit -/
def calories_per_rabbit : ℕ := 800

/-- The additional calories Brandon gets from catching squirrels instead of rabbits in 1 hour -/
def additional_calories : ℕ := 200

/-- The number of calories in each squirrel -/
def calories_per_squirrel : ℕ := 300

theorem squirrel_calories_proof :
  squirrels_per_hour * calories_per_squirrel = 
  rabbits_per_hour * calories_per_rabbit + additional_calories :=
by sorry

end NUMINAMATH_CALUDE_squirrel_calories_proof_l1436_143691


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1436_143665

def normal_pretzel_price : ℝ := 4
def discounted_pretzel_price : ℝ := 3.5
def normal_chip_price : ℝ := 7
def discounted_chip_price : ℝ := 6
def pretzel_discount_threshold : ℕ := 3
def chip_discount_threshold : ℕ := 2

def pretzel_packs_bought : ℕ := 3
def chip_packs_bought : ℕ := 4

def calculate_pretzel_cost (packs : ℕ) : ℝ :=
  if packs ≥ pretzel_discount_threshold then
    packs * discounted_pretzel_price
  else
    packs * normal_pretzel_price

def calculate_chip_cost (packs : ℕ) : ℝ :=
  if packs ≥ chip_discount_threshold then
    packs * discounted_chip_price
  else
    packs * normal_chip_price

theorem total_cost_calculation :
  calculate_pretzel_cost pretzel_packs_bought + calculate_chip_cost chip_packs_bought = 34.5 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1436_143665


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l1436_143666

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x^2 - 3 * x + 1) / Real.log (1/3)

theorem f_decreasing_on_interval : 
  ∀ x y, 1 < x → x < y → f y < f x := by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l1436_143666


namespace NUMINAMATH_CALUDE_bacteria_urea_phenol_red_l1436_143647

/-- Represents the color of the phenol red indicator -/
inductive IndicatorColor
| Blue
| Red
| Black
| Brown

/-- Represents the pH level of the medium -/
inductive pHLevel
| Acidic
| Neutral
| Alkaline

/-- Represents a culture medium -/
structure CultureMedium where
  nitrogenSource : String
  indicator : String
  pH : pHLevel

/-- Represents the bacterial culture -/
structure BacterialCulture where
  medium : CultureMedium
  bacteriaPresent : Bool

/-- Function to determine the color of phenol red based on pH -/
def phenolRedColor (pH : pHLevel) : IndicatorColor :=
  match pH with
  | pHLevel.Alkaline => IndicatorColor.Red
  | _ => IndicatorColor.Blue  -- Simplified for this problem

/-- Main theorem to prove -/
theorem bacteria_urea_phenol_red 
  (culture : BacterialCulture)
  (h1 : culture.medium.nitrogenSource = "urea")
  (h2 : culture.medium.indicator = "phenol red")
  (h3 : culture.bacteriaPresent = true) :
  phenolRedColor culture.medium.pH = IndicatorColor.Red :=
sorry

end NUMINAMATH_CALUDE_bacteria_urea_phenol_red_l1436_143647


namespace NUMINAMATH_CALUDE_expand_polynomial_l1436_143623

theorem expand_polynomial (x : ℝ) : 
  (x - 3) * (x + 3) * (x^2 + 9) * (x - 1) = x^5 - x^4 - 81*x + 81 := by
sorry

end NUMINAMATH_CALUDE_expand_polynomial_l1436_143623


namespace NUMINAMATH_CALUDE_quadrilateral_area_theorem_l1436_143605

-- Define the triangle and its division
structure DividedTriangle where
  total_area : ℝ
  triangle1_area : ℝ
  triangle2_area : ℝ
  triangle3_area : ℝ
  quadrilateral_area : ℝ
  division_valid : 
    total_area = triangle1_area + triangle2_area + triangle3_area + quadrilateral_area

-- State the theorem
theorem quadrilateral_area_theorem (t : DividedTriangle) 
  (h1 : t.triangle1_area = 4)
  (h2 : t.triangle2_area = 9)
  (h3 : t.triangle3_area = 9) :
  t.quadrilateral_area = 36 := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_area_theorem_l1436_143605


namespace NUMINAMATH_CALUDE_jovana_shells_total_l1436_143629

/-- The total amount of shells in Jovana's bucket -/
def total_shells (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that given the initial and additional amounts of shells,
    the total amount in Jovana's bucket is 17 pounds -/
theorem jovana_shells_total :
  total_shells 5 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_jovana_shells_total_l1436_143629


namespace NUMINAMATH_CALUDE_crayon_theorem_l1436_143613

/-- The number of crayons the other friend has -/
def other_friend_crayons (lizzie_crayons : ℕ) : ℕ :=
  lizzie_crayons * 4 / 3

theorem crayon_theorem (lizzie_crayons : ℕ) 
  (h1 : lizzie_crayons = 27) : 
  other_friend_crayons lizzie_crayons = 18 :=
by
  sorry

#eval other_friend_crayons 27

end NUMINAMATH_CALUDE_crayon_theorem_l1436_143613


namespace NUMINAMATH_CALUDE_contest_paths_count_l1436_143632

/-- Represents the grid structure for the word "CONTEST" --/
inductive ContestGrid
| C : ContestGrid
| O : ContestGrid → ContestGrid
| N : ContestGrid → ContestGrid
| T : ContestGrid → ContestGrid
| E : ContestGrid → ContestGrid
| S : ContestGrid → ContestGrid

/-- Counts the number of paths to form "CONTEST" in the given grid --/
def countContestPaths (grid : ContestGrid) : ℕ :=
  match grid with
  | ContestGrid.C => 1
  | ContestGrid.O g => 2 * countContestPaths g
  | ContestGrid.N g => 2 * countContestPaths g
  | ContestGrid.T g => 2 * countContestPaths g
  | ContestGrid.E g => 2 * countContestPaths g
  | ContestGrid.S g => 2 * countContestPaths g

/-- The contest grid structure --/
def contestGrid : ContestGrid :=
  ContestGrid.S (ContestGrid.E (ContestGrid.T (ContestGrid.N (ContestGrid.O (ContestGrid.C)))))

theorem contest_paths_count :
  countContestPaths contestGrid = 127 :=
sorry

end NUMINAMATH_CALUDE_contest_paths_count_l1436_143632


namespace NUMINAMATH_CALUDE_triangle_line_equations_l1436_143672

/-- Triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given triangle ABC, returns the equation of line AB -/
def line_AB (t : Triangle) : LineEquation :=
  { a := 3, b := 8, c := 15 }

/-- Given triangle ABC, returns the equation of the altitude from C to AB -/
def altitude_C (t : Triangle) : LineEquation :=
  { a := 8, b := -3, c := 6 }

theorem triangle_line_equations (t : Triangle) 
  (h1 : t.A = (-5, 0)) 
  (h2 : t.B = (3, -3)) 
  (h3 : t.C = (0, 2)) : 
  (line_AB t = { a := 3, b := 8, c := 15 }) ∧ 
  (altitude_C t = { a := 8, b := -3, c := 6 }) := by
  sorry

end NUMINAMATH_CALUDE_triangle_line_equations_l1436_143672


namespace NUMINAMATH_CALUDE_bogatyr_age_l1436_143600

/-- Represents the ages of five wine brands -/
structure WineAges where
  carlo_rosi : ℕ
  franzia : ℕ
  twin_valley : ℕ
  beaulieu_vineyard : ℕ
  bogatyr : ℕ

/-- Defines the relationships between wine ages -/
def valid_wine_ages (ages : WineAges) : Prop :=
  ages.carlo_rosi = 40 ∧
  ages.franzia = 3 * ages.carlo_rosi ∧
  ages.carlo_rosi = 4 * ages.twin_valley ∧
  ages.beaulieu_vineyard = ages.twin_valley / 2 ∧
  ages.bogatyr = 2 * ages.franzia

/-- Theorem: Given the relationships between wine ages, Bogatyr's age is 240 years -/
theorem bogatyr_age (ages : WineAges) (h : valid_wine_ages ages) : ages.bogatyr = 240 := by
  sorry

end NUMINAMATH_CALUDE_bogatyr_age_l1436_143600


namespace NUMINAMATH_CALUDE_planes_parallel_or_intersect_l1436_143638

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Parallel relation between lines -/
def Line.parallel (l1 l2 : Line) : Prop := sorry

/-- A line is contained in a plane -/
def Line.contained_in (l : Line) (p : Plane) : Prop := sorry

/-- Two planes are parallel -/
def Plane.parallel (p1 p2 : Plane) : Prop := sorry

/-- Two planes intersect -/
def Plane.intersect (p1 p2 : Plane) : Prop := sorry

/-- Main theorem: Given the conditions, planes α and β are either parallel or intersecting -/
theorem planes_parallel_or_intersect (α β : Plane) (a b c : Line) 
  (h1 : a.parallel b) (h2 : b.parallel c)
  (h3 : a.contained_in α) (h4 : b.contained_in β) (h5 : c.contained_in β) :
  Plane.parallel α β ∨ Plane.intersect α β := by sorry

end NUMINAMATH_CALUDE_planes_parallel_or_intersect_l1436_143638


namespace NUMINAMATH_CALUDE_f_properties_l1436_143624

def f (a x : ℝ) : ℝ := x^2 + |x - a| - 1

theorem f_properties (a : ℝ) :
  (∀ x, f a x = f a (-x)) ↔ a = 0 ∧
  (∀ x, f a x ≥ -a - 5/4) ∧ (a ≤ -1/2 → ∃ x, f a x = -a - 5/4) ∧
  (∀ x, f a x ≥ a^2 - 1) ∧ (-1/2 < a → a ≤ 1/2 → ∃ x, f a x = a^2 - 1) ∧
  (∀ x, f a x ≥ a - 5/4) ∧ (1/2 < a → ∃ x, f a x = a - 5/4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1436_143624


namespace NUMINAMATH_CALUDE_heat_of_formation_C6H6_value_l1436_143694

-- Define the heat changes for the given reactions
def heat_change_C2H2 : ℝ := 226.7
def heat_change_3C2H2_to_C6H6 : ℝ := 631.1
def heat_change_C6H6_gas_to_liquid : ℝ := -33.9

-- Define the function to calculate the heat of formation
def heat_of_formation_C6H6 : ℝ :=
  -3 * heat_change_C2H2 + heat_change_3C2H2_to_C6H6 - heat_change_C6H6_gas_to_liquid

-- Theorem statement
theorem heat_of_formation_C6H6_value :
  heat_of_formation_C6H6 = -82.9 := by sorry

end NUMINAMATH_CALUDE_heat_of_formation_C6H6_value_l1436_143694


namespace NUMINAMATH_CALUDE_birthday_money_calculation_l1436_143676

def playstation_cost : ℝ := 500
def christmas_money : ℝ := 150
def game_price : ℝ := 7.5
def games_to_sell : ℕ := 20

theorem birthday_money_calculation :
  let total_from_games : ℝ := game_price * (games_to_sell : ℝ)
  let remaining_money_needed : ℝ := playstation_cost - christmas_money - total_from_games
  remaining_money_needed = 200 := by sorry

end NUMINAMATH_CALUDE_birthday_money_calculation_l1436_143676


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l1436_143667

/-- The time it takes for a ball to hit the ground when thrown upward -/
theorem ball_hitting_ground_time :
  let initial_speed : ℝ := 5
  let initial_height : ℝ := 10
  let gravity : ℝ := 9.8
  let motion_equation (t : ℝ) : ℝ := -4.9 * t^2 + initial_speed * t + initial_height
  ∃ (t : ℝ), t > 0 ∧ motion_equation t = 0 ∧ t = 10/7 :=
by sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l1436_143667


namespace NUMINAMATH_CALUDE_f_value_at_107_5_l1436_143616

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_value_at_107_5 (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_period : ∀ x, f (x + 3) = -1 / f x)
  (h_neg : ∀ x, x < 0 → f x = 4 * x) :
  f 107.5 = 1/10 := by
sorry

end NUMINAMATH_CALUDE_f_value_at_107_5_l1436_143616


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_pi_third_l1436_143656

theorem cos_2alpha_plus_pi_third (α : ℝ) (h : Real.sin (α - π/3) = 2/3) :
  Real.cos (2*α + π/3) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_pi_third_l1436_143656


namespace NUMINAMATH_CALUDE_key_west_turtle_race_time_l1436_143693

/-- Represents the race times of turtles in the Key West Turtle Race -/
structure TurtleRaceTimes where
  greta : Float
  george : Float
  gloria : Float
  gary : Float
  gwen : Float

/-- Calculates the total race time for all turtles -/
def total_race_time (times : TurtleRaceTimes) : Float :=
  times.greta + times.george + times.gloria + times.gary + times.gwen

/-- Theorem stating the total race time for the given conditions -/
theorem key_west_turtle_race_time : ∃ (times : TurtleRaceTimes),
  times.greta = 6.5 ∧
  times.george = times.greta - 1.5 ∧
  times.gloria = 2 * times.george ∧
  times.gary = times.george + times.gloria + 1.75 ∧
  times.gwen = (times.greta + times.george) * 0.6 ∧
  total_race_time times = 45.15 := by
  sorry

end NUMINAMATH_CALUDE_key_west_turtle_race_time_l1436_143693


namespace NUMINAMATH_CALUDE_janes_apples_l1436_143607

theorem janes_apples (num_baskets : ℕ) (apples_taken : ℕ) (apples_remaining : ℕ) :
  num_baskets = 4 →
  apples_taken = 3 →
  apples_remaining = 13 →
  (num_baskets * (apples_remaining + apples_taken)) = 64 :=
by sorry

end NUMINAMATH_CALUDE_janes_apples_l1436_143607


namespace NUMINAMATH_CALUDE_min_workers_for_profit_l1436_143628

/-- Represents the company's financial model -/
structure CompanyModel where
  maintenance_fee : ℕ  -- Daily maintenance fee in dollars
  hourly_wage : ℕ      -- Hourly wage per worker in dollars
  widgets_per_hour : ℕ -- Widgets produced per worker per hour
  widget_price : ℚ     -- Selling price per widget in dollars
  work_hours : ℕ       -- Work hours per day

/-- Calculates the daily cost for a given number of workers -/
def daily_cost (model : CompanyModel) (workers : ℕ) : ℕ :=
  model.maintenance_fee + model.hourly_wage * workers * model.work_hours

/-- Calculates the daily revenue for a given number of workers -/
def daily_revenue (model : CompanyModel) (workers : ℕ) : ℚ :=
  (model.widgets_per_hour : ℚ) * model.widget_price * (workers : ℚ) * (model.work_hours : ℚ)

/-- Theorem stating the minimum number of workers needed for profit -/
theorem min_workers_for_profit (model : CompanyModel) 
  (h_maintenance : model.maintenance_fee = 600)
  (h_wage : model.hourly_wage = 20)
  (h_widgets : model.widgets_per_hour = 6)
  (h_price : model.widget_price = 7/2)
  (h_hours : model.work_hours = 7) :
  ∃ n : ℕ, (∀ m : ℕ, m ≥ n → daily_revenue model m > daily_cost model m) ∧
           (∀ m : ℕ, m < n → daily_revenue model m ≤ daily_cost model m) ∧
           n = 86 :=
sorry

end NUMINAMATH_CALUDE_min_workers_for_profit_l1436_143628


namespace NUMINAMATH_CALUDE_next_perfect_square_l1436_143651

theorem next_perfect_square (n : ℤ) (x : ℤ) (h1 : Even n) (h2 : x = n^2) :
  (n + 1)^2 = x + 2*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_next_perfect_square_l1436_143651


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1436_143698

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4}

-- Define set M
def M : Set ℕ := {0, 1}

-- Define set N
def N : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_intersection_theorem :
  (Set.compl M ∩ N) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1436_143698


namespace NUMINAMATH_CALUDE_number_of_subsets_l1436_143648

universe u

def card {α : Type u} (s : Set α) : ℕ := sorry

theorem number_of_subsets (M A B : Set ℕ) : 
  card M = 10 →
  A ⊆ M →
  B ⊆ M →
  A ∩ B = ∅ →
  card A = 2 →
  card B = 3 →
  card {X : Set ℕ | A ⊆ X ∧ X ⊆ M} = 256 := by sorry

end NUMINAMATH_CALUDE_number_of_subsets_l1436_143648


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1436_143620

/-- The eccentricity of a hyperbola with equation x^2 - y^2/4 = 1 is √5 -/
theorem hyperbola_eccentricity :
  let a : ℝ := 1  -- semi-major axis
  let b : ℝ := 2  -- semi-minor axis
  let c : ℝ := Real.sqrt (a^2 + b^2)  -- distance from center to focus
  let e : ℝ := c / a  -- eccentricity
  e = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1436_143620


namespace NUMINAMATH_CALUDE_storage_tubs_price_l1436_143661

/-- Calculates the total price Alison paid for storage tubs after discount -/
def total_price_after_discount (
  large_count : ℕ
  ) (medium_count : ℕ
  ) (small_count : ℕ
  ) (large_price : ℚ
  ) (medium_price : ℚ
  ) (small_price : ℚ
  ) (small_discount : ℚ
  ) : ℚ :=
  let large_medium_total := large_count * large_price + medium_count * medium_price
  let small_total := small_count * small_price * (1 - small_discount)
  large_medium_total + small_total

/-- Theorem stating the total price Alison paid for storage tubs after discount -/
theorem storage_tubs_price :
  total_price_after_discount 4 6 8 8 6 4 (1/10) = 968/10 :=
by
  sorry


end NUMINAMATH_CALUDE_storage_tubs_price_l1436_143661


namespace NUMINAMATH_CALUDE_comparison_of_powers_and_log_l1436_143658

theorem comparison_of_powers_and_log : 7^(3/10) > (3/10)^7 ∧ (3/10)^7 > Real.log (3/10) := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_powers_and_log_l1436_143658


namespace NUMINAMATH_CALUDE_line_equation_length_BC_l1436_143681

-- Problem 1
def projection_point : ℝ × ℝ := (2, -1)

theorem line_equation (l : Set (ℝ × ℝ)) (h : projection_point ∈ l) :
  l = {(x, y) | 2*x - y - 5 = 0} := by sorry

-- Problem 2
def point_A : ℝ × ℝ := (4, -1)
def midpoint_AB : ℝ × ℝ := (3, 2)
def centroid : ℝ × ℝ := (4, 2)

theorem length_BC :
  let B : ℝ × ℝ := (2*midpoint_AB.1 - point_A.1, 2*midpoint_AB.2 - point_A.2)
  let C : ℝ × ℝ := (3*centroid.1 - point_A.1 - B.1, 3*centroid.2 - point_A.2 - B.2)
  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_line_equation_length_BC_l1436_143681


namespace NUMINAMATH_CALUDE_bicycle_cost_calculation_l1436_143640

/-- Given two bicycles sold at a certain price, with specified profit and loss percentages,
    calculate the total cost of both bicycles. -/
theorem bicycle_cost_calculation 
  (selling_price : ℚ) 
  (profit_percent : ℚ) 
  (loss_percent : ℚ) : 
  selling_price = 990 →
  profit_percent = 10 / 100 →
  loss_percent = 10 / 100 →
  ∃ (cost1 cost2 : ℚ),
    cost1 * (1 + profit_percent) = selling_price ∧
    cost2 * (1 - loss_percent) = selling_price ∧
    cost1 + cost2 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_calculation_l1436_143640


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1436_143653

theorem rationalize_denominator : 
  (Real.sqrt 18 - Real.sqrt 2 + Real.sqrt 27) / (Real.sqrt 3 + Real.sqrt 2) = 5 - Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1436_143653


namespace NUMINAMATH_CALUDE_passengers_on_time_l1436_143622

theorem passengers_on_time (total : ℕ) (late : ℕ) (h1 : total = 14720) (h2 : late = 213) :
  total - late = 14507 := by
  sorry

end NUMINAMATH_CALUDE_passengers_on_time_l1436_143622


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l1436_143675

-- Define the solution set
def solution_set (a b c : ℝ) : Set ℝ :=
  {x | x ≤ -2 ∨ x ≥ 6}

-- Define the quadratic inequality
def quadratic_inequality (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c ≤ 0

-- Theorem statement
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : ∀ x, x ∈ solution_set a b c ↔ quadratic_inequality a b c x) :
  a < 0 ∧
  (∀ x, -1/6 < x ∧ x < 1/2 ↔ c * x^2 - b * x + a < 0) ∧
  a + b + c > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l1436_143675


namespace NUMINAMATH_CALUDE_smallest_matching_end_digits_correct_l1436_143603

/-- The smallest positive integer M such that M and M^2 end in the same sequence of three non-zero digits in base 10 -/
def smallest_matching_end_digits : ℕ := 376

/-- Check if a number ends with the given three digits -/
def ends_with (n : ℕ) (xyz : ℕ) : Prop :=
  n % 1000 = xyz

/-- The property that M and M^2 end with the same three non-zero digits -/
def has_matching_end_digits (M : ℕ) : Prop :=
  ∃ (xyz : ℕ), xyz ≥ 100 ∧ xyz < 1000 ∧ ends_with M xyz ∧ ends_with (M^2) xyz

theorem smallest_matching_end_digits_correct :
  has_matching_end_digits smallest_matching_end_digits ∧
  ∀ M : ℕ, M < smallest_matching_end_digits → ¬(has_matching_end_digits M) :=
sorry

end NUMINAMATH_CALUDE_smallest_matching_end_digits_correct_l1436_143603


namespace NUMINAMATH_CALUDE_equation_solution_l1436_143642

def solution_set : Set (ℕ × ℕ) :=
  {(0, 1), (1, 1), (3, 25), (4, 31), (5, 41), (8, 85)}

theorem equation_solution :
  {(a, b) : ℕ × ℕ | a * b + 2 = a ^ 3 + 2 * b} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1436_143642


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1436_143652

/-- The area of a quadrilateral with one diagonal of length 50 cm and offsets of 10 cm and 8 cm is 450 cm². -/
theorem quadrilateral_area (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) 
  (h1 : diagonal = 50) 
  (h2 : offset1 = 10) 
  (h3 : offset2 = 8) : 
  (1/2 * diagonal * offset1) + (1/2 * diagonal * offset2) = 450 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1436_143652


namespace NUMINAMATH_CALUDE_polynomial_identity_l1436_143630

theorem polynomial_identity (a a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x, (1 - 2*x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) : 
  (a + a₂ + a₄)^2 - (a₁ + a₃)^2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1436_143630


namespace NUMINAMATH_CALUDE_lillian_candy_count_l1436_143684

def initial_candies : ℕ := 88
def additional_candies : ℕ := 5

theorem lillian_candy_count :
  initial_candies + additional_candies = 93 := by sorry

end NUMINAMATH_CALUDE_lillian_candy_count_l1436_143684


namespace NUMINAMATH_CALUDE_volume_removed_percentage_l1436_143626

/-- Proves that removing six 4 cm cubes from a 20 cm × 15 cm × 10 cm box removes 12.8% of its volume -/
theorem volume_removed_percentage (box_length box_width box_height cube_side : ℝ) 
  (num_cubes_removed : ℕ) : 
  box_length = 20 → 
  box_width = 15 → 
  box_height = 10 → 
  cube_side = 4 → 
  num_cubes_removed = 6 → 
  (num_cubes_removed * cube_side^3) / (box_length * box_width * box_height) * 100 = 12.8 := by
  sorry

end NUMINAMATH_CALUDE_volume_removed_percentage_l1436_143626


namespace NUMINAMATH_CALUDE_last_digit_of_difference_l1436_143699

/-- A function that returns the last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- A function that checks if a number is a power of 10 -/
def isPowerOfTen (n : ℕ) : Prop := ∃ k : ℕ, n = 10^k

theorem last_digit_of_difference (p q : ℕ) 
  (hp : p > 0) (hq : q > 0) 
  (hpq : p > q)
  (hpLast : lastDigit p ≠ 0) 
  (hqLast : lastDigit q ≠ 0)
  (hProduct : isPowerOfTen (p * q)) : 
  lastDigit (p - q) ≠ 5 := by
sorry

end NUMINAMATH_CALUDE_last_digit_of_difference_l1436_143699


namespace NUMINAMATH_CALUDE_sin_cos_105_15_identity_l1436_143634

theorem sin_cos_105_15_identity : 
  Real.sin (105 * π / 180) * Real.sin (15 * π / 180) - 
  Real.cos (105 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_105_15_identity_l1436_143634


namespace NUMINAMATH_CALUDE_smallest_absolute_value_is_zero_l1436_143606

theorem smallest_absolute_value_is_zero : 
  ∀ q : ℚ, |0| ≤ |q| :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_is_zero_l1436_143606


namespace NUMINAMATH_CALUDE_simplify_expression_l1436_143689

theorem simplify_expression (b : ℝ) (h : b ≠ 2) :
  2 - 2 / (2 + b / (2 - b)) = 4 / (4 - b) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1436_143689


namespace NUMINAMATH_CALUDE_geometric_sequence_roots_l1436_143612

theorem geometric_sequence_roots (m n : ℝ) : 
  (∃ a b c d : ℝ, 
    (a^2 - m*a + 2) * (a^2 - n*a + 2) = 0 ∧
    (b^2 - m*b + 2) * (b^2 - n*b + 2) = 0 ∧
    (c^2 - m*c + 2) * (c^2 - n*c + 2) = 0 ∧
    (d^2 - m*d + 2) * (d^2 - n*d + 2) = 0 ∧
    a = (1/2 : ℝ) ∧
    ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) →
  |m - n| = (3/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_roots_l1436_143612


namespace NUMINAMATH_CALUDE_room_area_ratio_problem_l1436_143644

/-- Proof of room area ratio problem -/
theorem room_area_ratio_problem (original_length original_width increase : ℕ) 
  (total_area : ℕ) (num_equal_rooms : ℕ) :
  let new_length : ℕ := original_length + increase
  let new_width : ℕ := original_width + increase
  let equal_room_area : ℕ := new_length * new_width
  let total_equal_rooms_area : ℕ := num_equal_rooms * equal_room_area
  let largest_room_area : ℕ := total_area - total_equal_rooms_area
  original_length = 13 ∧ 
  original_width = 18 ∧ 
  increase = 2 ∧
  total_area = 1800 ∧
  num_equal_rooms = 4 →
  largest_room_area / equal_room_area = 2 := by
sorry

end NUMINAMATH_CALUDE_room_area_ratio_problem_l1436_143644


namespace NUMINAMATH_CALUDE_car_distribution_l1436_143696

theorem car_distribution (total_cars_per_column : ℕ) 
                         (total_zhiguli : ℕ) 
                         (zhiguli_first : ℕ) 
                         (zhiguli_second : ℕ) :
  total_cars_per_column = 28 →
  total_zhiguli = 11 →
  zhiguli_first + zhiguli_second = total_zhiguli →
  (total_cars_per_column - zhiguli_first) = 2 * (total_cars_per_column - zhiguli_second) →
  (total_cars_per_column - zhiguli_first = 21 ∧ total_cars_per_column - zhiguli_second = 24) :=
by
  sorry

#check car_distribution

end NUMINAMATH_CALUDE_car_distribution_l1436_143696


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1436_143678

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and an asymptote x/3 + y = 0 is √10/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b/a = 1/3) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 10 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1436_143678


namespace NUMINAMATH_CALUDE_m_plus_n_is_zero_l1436_143608

-- Define the complex function f
def f (m n z : ℂ) : ℂ := z^2 + m*z + n

-- State the theorem
theorem m_plus_n_is_zero (m n : ℂ) 
  (h : ∀ z : ℂ, Complex.abs z = 1 → Complex.abs (f m n z) = 1) : 
  m + n = 0 := by
  sorry

end NUMINAMATH_CALUDE_m_plus_n_is_zero_l1436_143608


namespace NUMINAMATH_CALUDE_cubic_equation_transformation_l1436_143621

theorem cubic_equation_transformation (p q r : ℝ) : 
  (p^3 - 5*p^2 + 6*p - 7 = 0) → 
  (q^3 - 5*q^2 + 6*q - 7 = 0) → 
  (r^3 - 5*r^2 + 6*r - 7 = 0) → 
  (∀ x : ℝ, x^3 - 10*x^2 + 25*x + 105 = 0 ↔ 
    (x = (p + q + r)/(p - 1) ∨ x = (p + q + r)/(q - 1) ∨ x = (p + q + r)/(r - 1))) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_transformation_l1436_143621


namespace NUMINAMATH_CALUDE_john_ate_12_ounces_l1436_143688

/-- The amount of steak John ate given the original weight, burned portion, and eating percentage -/
def steak_eaten (original_weight : ℝ) (burned_portion : ℝ) (eating_percentage : ℝ) : ℝ :=
  (1 - burned_portion) * original_weight * eating_percentage

/-- Theorem stating that John ate 12 ounces of steak -/
theorem john_ate_12_ounces : 
  let original_weight : ℝ := 30
  let burned_portion : ℝ := 1/2
  let eating_percentage : ℝ := 0.8
  steak_eaten original_weight burned_portion eating_percentage = 12 := by
  sorry

end NUMINAMATH_CALUDE_john_ate_12_ounces_l1436_143688


namespace NUMINAMATH_CALUDE_total_face_masks_produced_l1436_143617

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Represents the duration of Manolo's shift in hours -/
def shift_duration : ℕ := 4

/-- Represents the time to make one face-mask in the first hour (in minutes) -/
def first_hour_rate : ℕ := 4

/-- Represents the time to make one face-mask after the first hour (in minutes) -/
def subsequent_rate : ℕ := 6

/-- Calculates the number of face-masks made in the first hour -/
def first_hour_production : ℕ := minutes_per_hour / first_hour_rate

/-- Calculates the number of face-masks made in the subsequent hours -/
def subsequent_hours_production : ℕ := (shift_duration - 1) * minutes_per_hour / subsequent_rate

/-- Theorem: The total number of face-masks produced in a four-hour shift is 45 -/
theorem total_face_masks_produced :
  first_hour_production + subsequent_hours_production = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_face_masks_produced_l1436_143617


namespace NUMINAMATH_CALUDE_f_properties_l1436_143677

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + 1 / x

-- Theorem statement
theorem f_properties (a : ℝ) :
  (∀ x > 0, (deriv (f a)) x = 0 → x = 1) →
  (a = 0) ∧
  (∀ x > 0, f 0 x ≤ x * Real.exp x - x + 1 / x - 1) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1436_143677


namespace NUMINAMATH_CALUDE_b_income_percentage_over_c_l1436_143639

/-- Prove that B's monthly income is 12% more than C's monthly income given the specified conditions --/
theorem b_income_percentage_over_c (a_annual_income b_monthly_income c_monthly_income : ℚ) : 
  c_monthly_income = 16000 →
  a_annual_income = 537600 →
  a_annual_income / 12 / b_monthly_income = 5 / 2 →
  (b_monthly_income - c_monthly_income) / c_monthly_income = 12 / 100 := by
  sorry

end NUMINAMATH_CALUDE_b_income_percentage_over_c_l1436_143639


namespace NUMINAMATH_CALUDE_long_track_five_times_short_track_l1436_143646

/-- Represents the lengths of the short and long tracks -/
structure TrackLengths where
  short : ℝ
  long : ℝ

/-- Represents the training schedule for a week -/
structure WeekSchedule where
  days : ℕ
  longTracksPerDay : ℕ
  shortTracksPerDay : ℕ

/-- Calculates the total distance run in a week -/
def totalDistance (t : TrackLengths) (w : WeekSchedule) : ℝ :=
  w.days * (w.longTracksPerDay * t.long + w.shortTracksPerDay * t.short)

theorem long_track_five_times_short_track 
  (t : TrackLengths) 
  (w1 w2 : WeekSchedule) 
  (h1 : w1.days = 6 ∧ w1.longTracksPerDay = 1 ∧ w1.shortTracksPerDay = 2)
  (h2 : w2.days = 7 ∧ w2.longTracksPerDay = 1 ∧ w2.shortTracksPerDay = 1)
  (h3 : totalDistance t w1 = 5000)
  (h4 : totalDistance t w1 = totalDistance t w2) :
  t.long = 5 * t.short := by
  sorry

end NUMINAMATH_CALUDE_long_track_five_times_short_track_l1436_143646


namespace NUMINAMATH_CALUDE_loan_amount_proof_l1436_143683

/-- The interest rate at which A lends to B (as a decimal) -/
def rate_A_to_B : ℚ := 15 / 100

/-- The interest rate at which B lends to C (as a decimal) -/
def rate_B_to_C : ℚ := 185 / 1000

/-- The number of years for which the loan is given -/
def years : ℕ := 3

/-- The gain of B in the given period -/
def gain_B : ℕ := 294

/-- The amount lent by A to B -/
def amount_lent : ℕ := 2800

theorem loan_amount_proof :
  ∃ (P : ℕ), 
    (P : ℚ) * rate_B_to_C * years - (P : ℚ) * rate_A_to_B * years = gain_B ∧
    P = amount_lent :=
by sorry

end NUMINAMATH_CALUDE_loan_amount_proof_l1436_143683


namespace NUMINAMATH_CALUDE_inequality_condition_max_value_l1436_143682

-- Define the functions
def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|
def h (a x : ℝ) : ℝ := |f x| + g a x

-- Statement 1
theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, f x ≥ g a x) ↔ a ≤ -2 :=
sorry

-- Statement 2
theorem max_value (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ 
    ∀ y ∈ Set.Icc 0 2, h a x ≥ h a y) ∧
  (∃ m : ℝ, (∀ x ∈ Set.Icc 0 2, h a x ≤ m) ∧
    m = if a ≥ -3 then a + 3 else 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_max_value_l1436_143682


namespace NUMINAMATH_CALUDE_same_heads_probability_l1436_143649

-- Define the probability of getting a specific number of heads when tossing two coins
def prob_heads (n : Nat) : ℚ :=
  if n = 0 then 1/4
  else if n = 1 then 1/2
  else if n = 2 then 1/4
  else 0

-- Define the probability of both people getting the same number of heads
def prob_same_heads : ℚ :=
  (prob_heads 0)^2 + (prob_heads 1)^2 + (prob_heads 2)^2

-- Theorem statement
theorem same_heads_probability : prob_same_heads = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_same_heads_probability_l1436_143649


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l1436_143690

-- Define the types for lines and planes
def Line : Type := ℝ × ℝ × ℝ → Prop
def Plane : Type := ℝ × ℝ × ℝ → Prop

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular_planes (p1 : Plane) (p2 : Plane) : Prop := sorry

-- State the theorem
theorem line_plane_perpendicularity (l : Line) (α β : Plane) :
  perpendicular l α → parallel l β → perpendicular_planes α β := by sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l1436_143690


namespace NUMINAMATH_CALUDE_cricket_average_problem_l1436_143687

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  innings : Nat
  totalRuns : Nat
  deriving Repr

/-- Calculates the average runs per innings -/
def averageRuns (player : CricketPlayer) : Rat :=
  player.totalRuns / player.innings

theorem cricket_average_problem (player : CricketPlayer) 
  (h1 : player.innings = 20)
  (h2 : averageRuns { innings := player.innings + 1, totalRuns := player.totalRuns + 158 } = 
        averageRuns player + 6) :
  averageRuns player = 32 := by
  sorry


end NUMINAMATH_CALUDE_cricket_average_problem_l1436_143687


namespace NUMINAMATH_CALUDE_log_problem_l1436_143604

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_problem (x : ℝ) 
  (h1 : x < 1) 
  (h2 : (log10 x)^3 - 3 * log10 x = 522) : 
  (log10 x)^4 - log10 (x^4) = 6597 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l1436_143604


namespace NUMINAMATH_CALUDE_parallelogram_rotational_symmetry_l1436_143615

/-- A polygon in a 2D plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  is_closed : vertices.length ≥ 3

/-- A parallelogram is a quadrilateral with opposite sides parallel -/
def is_parallelogram (p : Polygon) : Prop :=
  p.vertices.length = 4 ∧
  ∃ (a b c d : ℝ × ℝ), p.vertices = [a, b, c, d] ∧
    (b.1 - a.1, b.2 - a.2) = (d.1 - c.1, d.2 - c.2) ∧
    (c.1 - b.1, c.2 - b.2) = (a.1 - d.1, a.2 - d.2)

/-- Rotation by 180 degrees around a point -/
def rotate_180 (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - p.1, 2 * center.2 - p.2)

/-- A polygon coincides with itself after 180-degree rotation -/
def coincides_after_rotation (p : Polygon) : Prop :=
  ∃ (center : ℝ × ℝ), 
    ∀ v ∈ p.vertices, (rotate_180 v center) ∈ p.vertices

theorem parallelogram_rotational_symmetry :
  ∀ (p : Polygon), is_parallelogram p → coincides_after_rotation p :=
sorry

end NUMINAMATH_CALUDE_parallelogram_rotational_symmetry_l1436_143615


namespace NUMINAMATH_CALUDE_company_employees_l1436_143633

theorem company_employees (december_employees : ℕ) (percentage_increase : ℚ) 
  (h1 : december_employees = 987)
  (h2 : percentage_increase = 127 / 1000) : 
  ∃ january_employees : ℕ, 
    (january_employees : ℚ) * (1 + percentage_increase) = december_employees ∧ 
    january_employees = 875 := by
  sorry

end NUMINAMATH_CALUDE_company_employees_l1436_143633


namespace NUMINAMATH_CALUDE_petunias_per_flat_is_8_l1436_143673

/-- Represents the number of petunias in each flat -/
def petunias_per_flat : ℕ := sorry

/-- The total number of flats of petunias -/
def petunia_flats : ℕ := 4

/-- The total number of flats of roses -/
def rose_flats : ℕ := 3

/-- The number of roses in each flat -/
def roses_per_flat : ℕ := 6

/-- The number of Venus flytraps -/
def venus_flytraps : ℕ := 2

/-- The amount of fertilizer needed for each petunia (in ounces) -/
def fertilizer_per_petunia : ℕ := 8

/-- The amount of fertilizer needed for each rose (in ounces) -/
def fertilizer_per_rose : ℕ := 3

/-- The amount of fertilizer needed for each Venus flytrap (in ounces) -/
def fertilizer_per_flytrap : ℕ := 2

/-- The total amount of fertilizer needed (in ounces) -/
def total_fertilizer : ℕ := 314

theorem petunias_per_flat_is_8 :
  petunias_per_flat = 8 :=
by sorry

end NUMINAMATH_CALUDE_petunias_per_flat_is_8_l1436_143673


namespace NUMINAMATH_CALUDE_function_domain_implies_m_range_l1436_143650

/-- Given a function f(x) = 1 / √(mx² + mx + 1) with domain R, 
    prove that m must be in the range [0, 4) -/
theorem function_domain_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, mx^2 + mx + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
sorry

end NUMINAMATH_CALUDE_function_domain_implies_m_range_l1436_143650


namespace NUMINAMATH_CALUDE_max_power_sum_l1436_143643

theorem max_power_sum (c d : ℕ) : 
  d > 1 → 
  c^d < 630 → 
  (∀ (x y : ℕ), y > 1 → x^y < 630 → c^d ≥ x^y) → 
  c + d = 27 := by
sorry

end NUMINAMATH_CALUDE_max_power_sum_l1436_143643


namespace NUMINAMATH_CALUDE_exists_m_for_all_x_m_range_when_exists_x_l1436_143662

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Theorem 1: Existence of m such that m + f(x) > 0 for all x
theorem exists_m_for_all_x (m : ℝ) : 
  (∀ x, m + f x > 0) ↔ m > -2 := by sorry

-- Theorem 2: Range of m when there exists x such that m - f(x) > 0
theorem m_range_when_exists_x (m : ℝ) :
  (∃ x, m - f x > 0) → m > 2 := by sorry

end NUMINAMATH_CALUDE_exists_m_for_all_x_m_range_when_exists_x_l1436_143662


namespace NUMINAMATH_CALUDE_sum_of_first_100_digits_l1436_143660

/-- The decimal expansion of 1/10101 -/
def decimal_expansion : ℕ → ℕ
| n => sorry

/-- The sum of the first n digits in the decimal expansion of 1/10101 -/
def digit_sum (n : ℕ) : ℕ :=
  (List.range n).map decimal_expansion |>.sum

/-- Theorem: The sum of the first 100 digits after the decimal point in 1/10101 is 450 -/
theorem sum_of_first_100_digits : digit_sum 100 = 450 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_100_digits_l1436_143660


namespace NUMINAMATH_CALUDE_positive_real_inequality_l1436_143697

theorem positive_real_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y^2016 ≥ 1) :
  x^2016 + y > 1 - 1/100 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l1436_143697


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l1436_143659

def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem twentieth_term_of_sequence (a₁ a₂ : ℝ) (h : a₂ = a₁ + 5) :
  arithmeticSequence a₁ (a₂ - a₁) 20 = 98 :=
by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l1436_143659


namespace NUMINAMATH_CALUDE_sum_of_squares_remainder_l1436_143671

theorem sum_of_squares_remainder (a b c d e : ℕ) (ha : a = 445876) (hb : b = 985420) (hc : c = 215546) (hd : d = 656452) (he : e = 387295) :
  (a^2 + b^2 + c^2 + d^2 + e^2) % 8 = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_remainder_l1436_143671


namespace NUMINAMATH_CALUDE_folded_paper_triangle_perimeter_l1436_143641

/-- A square piece of paper with side length 2 is folded such that vertex C meets edge AB at point C',
    making C'B = 2/3. Edge BC intersects edge AD at point E. -/
theorem folded_paper_triangle_perimeter :
  ∀ (A B C D C' E : ℝ × ℝ),
    -- Square conditions
    A = (0, 2) ∧ B = (0, 0) ∧ C = (2, 0) ∧ D = (2, 2) →
    -- Folding conditions
    C' = (0, 4/3) →
    -- Intersection condition
    E = (2, 0) →
    -- Perimeter calculation
    dist A E + dist E C' + dist C' A = 4 :=
by sorry

end NUMINAMATH_CALUDE_folded_paper_triangle_perimeter_l1436_143641


namespace NUMINAMATH_CALUDE_basket_count_l1436_143674

theorem basket_count (apples_per_basket : ℕ) (total_apples : ℕ) (h1 : apples_per_basket = 17) (h2 : total_apples = 629) :
  total_apples / apples_per_basket = 37 := by
sorry

end NUMINAMATH_CALUDE_basket_count_l1436_143674


namespace NUMINAMATH_CALUDE_arithmetic_progression_cube_sum_l1436_143601

theorem arithmetic_progression_cube_sum (k x y z : ℤ) :
  (x < y ∧ y < z) →  -- x, y, z form an increasing sequence
  (z - y = y - x) →  -- x, y, z form an arithmetic progression
  (k * y^3 = x^3 + z^3) →  -- given equation
  ∃ t : ℤ, k = 2 * (3 * t^2 + 1) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_cube_sum_l1436_143601


namespace NUMINAMATH_CALUDE_juice_sales_theorem_l1436_143695

/-- Represents the capacity of a can in liters -/
structure CanCapacity where
  large : ℝ
  medium : ℝ
  liter : ℝ

/-- Represents the daily sales data -/
structure DailySales where
  large : ℕ
  medium : ℕ
  liter : ℕ

/-- Calculates the total volume of juice sold in a day -/
def dailyVolume (c : CanCapacity) (s : DailySales) : ℝ :=
  c.large * s.large + c.medium * s.medium + c.liter * s.liter

theorem juice_sales_theorem (c : CanCapacity) 
  (s1 s2 s3 : DailySales) : 
  c.liter = 1 →
  s1 = ⟨1, 4, 0⟩ →
  s2 = ⟨2, 0, 6⟩ →
  s3 = ⟨1, 3, 3⟩ →
  dailyVolume c s1 = dailyVolume c s2 →
  dailyVolume c s2 = dailyVolume c s3 →
  (dailyVolume c s1 + dailyVolume c s2 + dailyVolume c s3) = 54 := by
  sorry

#check juice_sales_theorem

end NUMINAMATH_CALUDE_juice_sales_theorem_l1436_143695


namespace NUMINAMATH_CALUDE_fifteenth_term_is_negative_one_l1436_143610

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  first_term : ℤ
  common_diff : ℤ

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first_term + (n - 1 : ℤ) * seq.common_diff

theorem fifteenth_term_is_negative_one
  (seq : ArithmeticSequence)
  (h21 : nth_term seq 21 = 17)
  (h22 : nth_term seq 22 = 20) :
  nth_term seq 15 = -1 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_is_negative_one_l1436_143610


namespace NUMINAMATH_CALUDE_absolute_value_equation_sum_l1436_143637

theorem absolute_value_equation_sum (n : ℝ) : 
  (∃ n₁ n₂ : ℝ, |3 * n₁ - 8| = 5 ∧ |3 * n₂ - 8| = 5 ∧ n₁ ≠ n₂ ∧ n₁ + n₂ = 16/3) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_sum_l1436_143637


namespace NUMINAMATH_CALUDE_coin_count_theorem_l1436_143631

theorem coin_count_theorem (quarters_piles : Nat) (quarters_per_pile : Nat)
                           (dimes_piles : Nat) (dimes_per_pile : Nat)
                           (nickels_piles : Nat) (nickels_per_pile : Nat)
                           (pennies_piles : Nat) (pennies_per_pile : Nat) :
  quarters_piles = 7 →
  quarters_per_pile = 4 →
  dimes_piles = 4 →
  dimes_per_pile = 2 →
  nickels_piles = 6 →
  nickels_per_pile = 5 →
  pennies_piles = 3 →
  pennies_per_pile = 8 →
  quarters_piles * quarters_per_pile +
  dimes_piles * dimes_per_pile +
  nickels_piles * nickels_per_pile +
  pennies_piles * pennies_per_pile = 90 := by
  sorry

end NUMINAMATH_CALUDE_coin_count_theorem_l1436_143631


namespace NUMINAMATH_CALUDE_smallest_factor_sum_l1436_143609

theorem smallest_factor_sum (b : ℕ) (p q : ℤ) : 
  (∀ x, x^2 + b*x + 2040 = (x + p) * (x + q)) →
  (∀ b' : ℕ, b' < b → 
    ¬∃ p' q' : ℤ, ∀ x, x^2 + b'*x + 2040 = (x + p') * (x + q')) →
  b = 94 :=
sorry

end NUMINAMATH_CALUDE_smallest_factor_sum_l1436_143609


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1436_143645

theorem imaginary_part_of_complex_fraction (i : ℂ) : 
  i^2 = -1 → Complex.im (i^5 / (1 - i)) = 1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1436_143645


namespace NUMINAMATH_CALUDE_sam_distance_l1436_143664

/-- Given Marguerite's travel details and Sam's driving time, prove Sam's distance traveled. -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) 
  (h1 : marguerite_distance = 150)
  (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) :
  (marguerite_distance / marguerite_time) * sam_time = 200 := by
  sorry

end NUMINAMATH_CALUDE_sam_distance_l1436_143664


namespace NUMINAMATH_CALUDE_tiger_escape_distance_l1436_143655

/-- Represents the speed and duration of each phase of the tiger's escape --/
structure EscapePhase where
  speed : ℝ
  duration : ℝ

/-- Calculates the total distance traveled by the tiger --/
def totalDistance (phases : List EscapePhase) : ℝ :=
  phases.foldl (fun acc phase => acc + phase.speed * phase.duration) 0

/-- The escape phases of the tiger --/
def tigerEscapePhases : List EscapePhase := [
  { speed := 25, duration := 1 },
  { speed := 35, duration := 2 },
  { speed := 20, duration := 1.5 },
  { speed := 10, duration := 1 },
  { speed := 50, duration := 0.5 }
]

theorem tiger_escape_distance :
  totalDistance tigerEscapePhases = 160 := by
  sorry

end NUMINAMATH_CALUDE_tiger_escape_distance_l1436_143655


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l1436_143685

-- Define the polynomials
def p (x : ℝ) : ℝ := 7 * x^2 + 5 * x + 3
def q (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + 1

-- State the theorem
theorem polynomial_product_expansion :
  ∀ x : ℝ, p x * q x = 21 * x^5 + 29 * x^4 + 19 * x^3 + 13 * x^2 + 5 * x + 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l1436_143685


namespace NUMINAMATH_CALUDE_first_year_x_exceeds_y_l1436_143614

def commodity_x_price (year : ℕ) : ℚ :=
  420/100 + (year - 2001) * 30/100

def commodity_y_price (year : ℕ) : ℚ :=
  440/100 + (year - 2001) * 20/100

theorem first_year_x_exceeds_y :
  (∀ y : ℕ, 2001 < y ∧ y < 2004 → commodity_x_price y ≤ commodity_y_price y) ∧
  commodity_x_price 2004 > commodity_y_price 2004 :=
by sorry

end NUMINAMATH_CALUDE_first_year_x_exceeds_y_l1436_143614


namespace NUMINAMATH_CALUDE_shortest_side_length_l1436_143657

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first segment of the divided side -/
  a : ℝ
  /-- The length of the second segment of the divided side -/
  b : ℝ
  /-- The length of the shortest side of the triangle -/
  shortest_side : ℝ
  /-- Assumption that all lengths are positive -/
  r_pos : r > 0
  a_pos : a > 0
  b_pos : b > 0
  shortest_side_pos : shortest_side > 0

/-- Theorem stating the length of the shortest side in the specific triangle -/
theorem shortest_side_length (t : InscribedCircleTriangle) 
    (h1 : t.r = 5) 
    (h2 : t.a = 9) 
    (h3 : t.b = 15) : 
  t.shortest_side = 17 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_length_l1436_143657


namespace NUMINAMATH_CALUDE_expansion_sum_l1436_143635

theorem expansion_sum (d : ℝ) (h : d ≠ 0) :
  let expansion := (15*d + 21 + 17*d^2) * (3*d + 4)
  ∃ (a b c e : ℝ), expansion = a*d^3 + b*d^2 + c*d + e ∧ a + b + c + e = 371 := by
  sorry

end NUMINAMATH_CALUDE_expansion_sum_l1436_143635


namespace NUMINAMATH_CALUDE_regular_washes_count_l1436_143692

/-- Represents the number of gallons of water used for different types of washes --/
structure WaterUsage where
  heavy : ℕ
  regular : ℕ
  light : ℕ

/-- Represents the number of different types of washes --/
structure Washes where
  heavy : ℕ
  regular : ℕ
  light : ℕ
  bleached : ℕ

/-- Calculates the total water usage for a given set of washes --/
def calculateWaterUsage (usage : WaterUsage) (washes : Washes) : ℕ :=
  usage.heavy * washes.heavy +
  usage.regular * washes.regular +
  usage.light * washes.light +
  usage.light * washes.bleached

/-- Theorem stating that there are 3 regular washes given the problem conditions --/
theorem regular_washes_count (usage : WaterUsage) (washes : Washes) :
  usage.heavy = 20 →
  usage.regular = 10 →
  usage.light = 2 →
  washes.heavy = 2 →
  washes.light = 1 →
  washes.bleached = 2 →
  calculateWaterUsage usage washes = 76 →
  washes.regular = 3 := by
  sorry

end NUMINAMATH_CALUDE_regular_washes_count_l1436_143692


namespace NUMINAMATH_CALUDE_circle_rectangle_area_difference_l1436_143670

/-- Given a rectangle with diagonal 10 and length-to-width ratio 2:1, and a circle with radius 5,
    prove that the difference between the circle's area and the rectangle's area is 25π - 40. -/
theorem circle_rectangle_area_difference :
  let rectangle_diagonal : ℝ := 10
  let length_width_ratio : ℚ := 2 / 1
  let circle_radius : ℝ := 5
  let rectangle_width : ℝ := (rectangle_diagonal ^ 2 / (1 + length_width_ratio ^ 2)) ^ (1 / 2 : ℝ)
  let rectangle_length : ℝ := length_width_ratio * rectangle_width
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  let circle_area : ℝ := π * circle_radius ^ 2
  circle_area - rectangle_area = 25 * π - 40 := by
  sorry

end NUMINAMATH_CALUDE_circle_rectangle_area_difference_l1436_143670


namespace NUMINAMATH_CALUDE_count_solutions_eq_338350_l1436_143602

/-- The number of distinct integer solutions to |x| + |y| < 100 -/
def count_solutions : ℕ :=
  (Finset.sum (Finset.range 100) (fun k => (k + 1)^2) : ℕ)

/-- Theorem stating the correct number of solutions -/
theorem count_solutions_eq_338350 : count_solutions = 338350 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_eq_338350_l1436_143602


namespace NUMINAMATH_CALUDE_prime_form_and_infinitude_l1436_143618

theorem prime_form_and_infinitude (p : ℕ) :
  (Prime p ∧ p ≥ 3) →
  (∃! k : ℕ, k ≥ 1 ∧ (p = 4*k - 1 ∨ p = 4*k + 1)) ∧
  Set.Infinite {p : ℕ | Prime p ∧ ∃ k : ℕ, p = 4*k - 1} :=
by sorry

end NUMINAMATH_CALUDE_prime_form_and_infinitude_l1436_143618


namespace NUMINAMATH_CALUDE_smallest_n_for_irreducible_fractions_l1436_143636

theorem smallest_n_for_irreducible_fractions : 
  ∃ (n : ℕ), n = 35 ∧ 
  (∀ k : ℕ, 7 ≤ k ∧ k ≤ 31 → Nat.gcd k (n + k + 2) = 1) ∧
  (∀ m : ℕ, m < n → ∃ k : ℕ, 7 ≤ k ∧ k ≤ 31 ∧ Nat.gcd k (m + k + 2) ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_irreducible_fractions_l1436_143636


namespace NUMINAMATH_CALUDE_mashed_potatoes_suggestion_l1436_143627

theorem mashed_potatoes_suggestion (bacon_count : ℕ) (difference : ℕ) : 
  bacon_count = 394 → 
  difference = 63 → 
  bacon_count + difference = 457 :=
by
  sorry

end NUMINAMATH_CALUDE_mashed_potatoes_suggestion_l1436_143627


namespace NUMINAMATH_CALUDE_field_area_change_l1436_143625

theorem field_area_change (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let new_length := 1.35 * a
  let new_width := 0.86 * b
  let initial_area := a * b
  let new_area := new_length * new_width
  (new_area - initial_area) / initial_area = 0.161 := by
sorry

end NUMINAMATH_CALUDE_field_area_change_l1436_143625


namespace NUMINAMATH_CALUDE_september_to_august_ratio_l1436_143611

def july_earnings : ℕ := 150
def august_earnings : ℕ := 3 * july_earnings
def total_earnings : ℕ := 1500

def september_earnings_ratio (x : ℚ) : Prop :=
  july_earnings + august_earnings + x * august_earnings = total_earnings

theorem september_to_august_ratio :
  ∃ x : ℚ, september_earnings_ratio x ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_september_to_august_ratio_l1436_143611


namespace NUMINAMATH_CALUDE_square_cut_perimeter_l1436_143679

/-- The perimeter of a figure formed by cutting a square and rearranging it -/
theorem square_cut_perimeter (s : ℝ) (h : s = 100) : 
  let rect_length : ℝ := s
  let rect_width : ℝ := s / 2
  let perimeter : ℝ := 3 * rect_length + 4 * rect_width
  perimeter = 500 := by sorry

end NUMINAMATH_CALUDE_square_cut_perimeter_l1436_143679


namespace NUMINAMATH_CALUDE_vector_sum_zero_l1436_143668

variable {V : Type*} [AddCommGroup V]

theorem vector_sum_zero (A B C : V) : (B - A) + (A - C) - (B - C) = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_zero_l1436_143668


namespace NUMINAMATH_CALUDE_simplify_expression_l1436_143619

theorem simplify_expression (a : ℝ) (h : a ≤ (1/2 : ℝ)) :
  Real.sqrt (1 - 4*a + 4*a^2) + |2*a - 1| = 2 - 4*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1436_143619


namespace NUMINAMATH_CALUDE_perfect_squares_of_cube_sums_l1436_143686

theorem perfect_squares_of_cube_sums : 
  ∃ (a b c d : ℕ),
    (1^3 + 2^3 = a^2) ∧ 
    (1^3 + 2^3 + 3^3 = b^2) ∧ 
    (1^3 + 2^3 + 3^3 + 4^3 = c^2) ∧ 
    (1^3 + 2^3 + 3^3 + 4^3 + 5^3 = d^2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_of_cube_sums_l1436_143686
