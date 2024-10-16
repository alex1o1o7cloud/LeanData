import Mathlib

namespace NUMINAMATH_CALUDE_corresponding_sides_proportional_in_similar_triangles_l3838_383801

-- Define what it means for two triangles to be similar
def similar_triangles (t1 t2 : Triangle) : Prop := sorry

-- Define what it means for sides to be corresponding
def corresponding_sides (s1 : Segment) (t1 : Triangle) (s2 : Segment) (t2 : Triangle) : Prop := sorry

-- Define what it means for two segments to be proportional
def proportional (s1 s2 : Segment) : Prop := sorry

-- Theorem statement
theorem corresponding_sides_proportional_in_similar_triangles 
  (t1 t2 : Triangle) (s1 s3 : Segment) (s2 s4 : Segment) :
  similar_triangles t1 t2 →
  corresponding_sides s1 t1 s2 t2 →
  corresponding_sides s3 t1 s4 t2 →
  proportional s1 s2 ∧ proportional s3 s4 := by sorry

end NUMINAMATH_CALUDE_corresponding_sides_proportional_in_similar_triangles_l3838_383801


namespace NUMINAMATH_CALUDE_container_volume_ratio_l3838_383881

/-- Represents a container with a certain volume and amount of juice -/
structure Container where
  volume : ℝ
  juice : ℝ

/-- The problem setup -/
def problem_setup (container1 container2 : Container) : Prop :=
  container1.juice = 3/4 * container1.volume ∧
  container2.juice = 0 ∧
  container2.juice + container1.juice/2 = 5/8 * container2.volume

/-- The theorem to prove -/
theorem container_volume_ratio 
  (container1 container2 : Container) 
  (h : problem_setup container1 container2) : 
  container1.volume / container2.volume = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l3838_383881


namespace NUMINAMATH_CALUDE_no_distinct_natural_power_sum_equality_l3838_383870

theorem no_distinct_natural_power_sum_equality :
  ∀ (x y z t : ℕ),
    x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t →
    x^x + y^y ≠ z^z + t^t :=
by
  sorry

end NUMINAMATH_CALUDE_no_distinct_natural_power_sum_equality_l3838_383870


namespace NUMINAMATH_CALUDE_necessary_sufficient_condition_l3838_383844

theorem necessary_sufficient_condition (a b : ℝ) (h : |a| > |b|) :
  (a - b > 0) ↔ (a + b > 0) :=
sorry

end NUMINAMATH_CALUDE_necessary_sufficient_condition_l3838_383844


namespace NUMINAMATH_CALUDE_four_students_three_communities_l3838_383805

/-- The number of ways to assign students to communities -/
def assignStudents (num_students : ℕ) (num_communities : ℕ) : ℕ :=
  num_communities ^ num_students

/-- Theorem stating that assigning 4 students to 3 communities results in 3^4 arrangements -/
theorem four_students_three_communities :
  assignStudents 4 3 = 3^4 := by
  sorry

end NUMINAMATH_CALUDE_four_students_three_communities_l3838_383805


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3838_383899

noncomputable section

-- Define the hyperbola and its properties
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

-- Define the foci
def LeftFocus (a c : ℝ) : ℝ × ℝ := (-c, 0)
def RightFocus (a c : ℝ) : ℝ × ℝ := (c, 0)

-- Define the point P on the right branch of the hyperbola
def P (a b : ℝ) : ℝ × ℝ := sorry

-- Define the perpendicular bisector of PF₁
def PerpendicularBisectorPF₁ (a b c : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the distance from origin to line PF₁
def DistanceOriginToPF₁ (a b c : ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_eccentricity (a b c : ℝ) :
  (P a b ∈ Hyperbola a b) →
  (RightFocus a c ∈ PerpendicularBisectorPF₁ a b c) →
  (DistanceOriginToPF₁ a b c = a) →
  (c / a = 5 / 3) := by
  sorry

end

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3838_383899


namespace NUMINAMATH_CALUDE_sam_total_money_l3838_383828

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The number of pennies Sam earned -/
def num_pennies : ℕ := 15

/-- The number of nickels Sam earned -/
def num_nickels : ℕ := 11

/-- The number of dimes Sam earned -/
def num_dimes : ℕ := 21

/-- The number of quarters Sam earned -/
def num_quarters : ℕ := 29

/-- The total value of Sam's coins in dollars -/
def total_value : ℚ := 
  num_pennies * penny_value + 
  num_nickels * nickel_value + 
  num_dimes * dime_value + 
  num_quarters * quarter_value

theorem sam_total_money : total_value = 10.05 := by
  sorry

end NUMINAMATH_CALUDE_sam_total_money_l3838_383828


namespace NUMINAMATH_CALUDE_total_distance_meters_l3838_383855

def distance_feet : ℝ := 30
def feet_to_meters : ℝ := 0.3048
def num_trips : ℕ := 4

theorem total_distance_meters : 
  distance_feet * feet_to_meters * (num_trips : ℝ) = 36.576 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_meters_l3838_383855


namespace NUMINAMATH_CALUDE_divisors_of_36_count_divisors_of_36_l3838_383889

theorem divisors_of_36 : Finset Int → Prop :=
  fun s => ∀ n : Int, n ∈ s ↔ 36 % n = 0

theorem count_divisors_of_36 : 
  ∃ s : Finset Int, divisors_of_36 s ∧ s.card = 18 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_36_count_divisors_of_36_l3838_383889


namespace NUMINAMATH_CALUDE_novel_sales_theorem_l3838_383846

/-- Represents the sale of a novel in hardback and paperback versions -/
structure NovelSales where
  hardback_before_paperback : ℕ
  paperback_total : ℕ
  paperback_to_hardback_ratio : ℕ

/-- Calculates the total number of copies sold given the sales data -/
def total_copies_sold (sales : NovelSales) : ℕ :=
  sales.hardback_before_paperback + 
  sales.paperback_total + 
  (sales.paperback_total / sales.paperback_to_hardback_ratio)

/-- Theorem stating that given the conditions, the total number of copies sold is 440400 -/
theorem novel_sales_theorem (sales : NovelSales) 
  (h1 : sales.hardback_before_paperback = 36000)
  (h2 : sales.paperback_to_hardback_ratio = 9)
  (h3 : sales.paperback_total = 363600) :
  total_copies_sold sales = 440400 := by
  sorry

#eval total_copies_sold ⟨36000, 363600, 9⟩

end NUMINAMATH_CALUDE_novel_sales_theorem_l3838_383846


namespace NUMINAMATH_CALUDE_relay_race_arrangements_l3838_383863

/-- The number of people in the class -/
def total_people : ℕ := 5

/-- The number of people to be selected for the race -/
def selected_people : ℕ := 4

/-- The set of possible first runners -/
inductive FirstRunner
| A
| B
| C

/-- The set of possible last runners -/
inductive LastRunner
| A
| B

/-- The total number of different arrangements for the order of runners -/
def total_arrangements : ℕ := 24

theorem relay_race_arrangements :
  total_arrangements = 24 :=
sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_l3838_383863


namespace NUMINAMATH_CALUDE_sue_chewing_gums_l3838_383894

theorem sue_chewing_gums (mary_gums sam_gums total_gums : ℕ) (sue_gums : ℕ) :
  mary_gums = 5 →
  sam_gums = 10 →
  total_gums = 30 →
  total_gums = mary_gums + sam_gums + sue_gums →
  sue_gums = 15 := by
  sorry

end NUMINAMATH_CALUDE_sue_chewing_gums_l3838_383894


namespace NUMINAMATH_CALUDE_sphere_part_volume_l3838_383816

theorem sphere_part_volume (circumference : ℝ) (h : circumference = 18 * Real.pi) :
  let radius := circumference / (2 * Real.pi)
  let sphere_volume := (4 / 3) * Real.pi * radius ^ 3
  let part_volume := sphere_volume / 6
  part_volume = 162 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_part_volume_l3838_383816


namespace NUMINAMATH_CALUDE_no_prime_interior_angles_l3838_383877

def interior_angle (n : ℕ) : ℚ := (180 * (n - 2)) / n

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem no_prime_interior_angles :
  ∀ n : ℕ, 10 ≤ n → n < 20 →
    ¬(∃ k : ℕ, interior_angle n = k ∧ is_prime k) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_interior_angles_l3838_383877


namespace NUMINAMATH_CALUDE_percentage_of_returns_l3838_383858

/-- Calculate the percentage of customers returning books -/
theorem percentage_of_returns (total_customers : ℕ) (price_per_book : ℚ) (sales_after_returns : ℚ) :
  total_customers = 1000 →
  price_per_book = 15 →
  sales_after_returns = 9450 →
  (((total_customers : ℚ) * price_per_book - sales_after_returns) / price_per_book) / (total_customers : ℚ) * 100 = 37 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_returns_l3838_383858


namespace NUMINAMATH_CALUDE_car_savings_calculation_l3838_383847

theorem car_savings_calculation 
  (monthly_earnings : ℕ) 
  (car_cost : ℕ) 
  (total_earnings : ℕ) 
  (h1 : monthly_earnings = 4000)
  (h2 : car_cost = 45000)
  (h3 : total_earnings = 360000) :
  car_cost / (total_earnings / monthly_earnings) = 500 := by
sorry

end NUMINAMATH_CALUDE_car_savings_calculation_l3838_383847


namespace NUMINAMATH_CALUDE_inscribed_box_sphere_radius_l3838_383860

theorem inscribed_box_sphere_radius (a b c s : ℝ) : 
  a > 0 → b > 0 → c > 0 → s > 0 →
  (a + b + c = 18) →
  (2 * a * b + 2 * b * c + 2 * a * c = 216) →
  (4 * s^2 = a^2 + b^2 + c^2) →
  s = Real.sqrt 27 := by
sorry

end NUMINAMATH_CALUDE_inscribed_box_sphere_radius_l3838_383860


namespace NUMINAMATH_CALUDE_monday_sunday_speed_ratio_l3838_383829

/-- Proves that the ratio of speeds on Monday (first 32 miles) to Sunday is 2:1 -/
theorem monday_sunday_speed_ratio 
  (total_distance : ℝ) 
  (sunday_speed : ℝ) 
  (monday_first_distance : ℝ) 
  (monday_first_speed : ℝ) :
  total_distance = 120 →
  monday_first_distance = 32 →
  (total_distance / sunday_speed) * 1.6 = 
    (monday_first_distance / monday_first_speed) + 
    ((total_distance - monday_first_distance) / (sunday_speed / 2)) →
  monday_first_speed / sunday_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_monday_sunday_speed_ratio_l3838_383829


namespace NUMINAMATH_CALUDE_yarns_are_zorps_and_xings_l3838_383837

variable (U : Type) -- Universe set

-- Define the subsets
variable (Zorp Xing Yarn Wit Vamp : Set U)

-- State the given conditions
variable (h1 : Zorp ⊆ Xing)
variable (h2 : Yarn ⊆ Xing)
variable (h3 : Wit ⊆ Zorp)
variable (h4 : Yarn ⊆ Wit)
variable (h5 : Yarn ⊆ Vamp)

-- Theorem to prove
theorem yarns_are_zorps_and_xings : Yarn ⊆ Zorp ∧ Yarn ⊆ Xing := by sorry

end NUMINAMATH_CALUDE_yarns_are_zorps_and_xings_l3838_383837


namespace NUMINAMATH_CALUDE_jungkook_red_balls_l3838_383835

/-- The number of boxes Jungkook bought -/
def num_boxes : ℕ := 2

/-- The number of red balls in each box -/
def balls_per_box : ℕ := 3

/-- The total number of red balls Jungkook has -/
def total_balls : ℕ := num_boxes * balls_per_box

theorem jungkook_red_balls : total_balls = 6 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_red_balls_l3838_383835


namespace NUMINAMATH_CALUDE_max_sum_of_digits_l3838_383836

/-- Represents a nonzero digit (1-9) -/
def NonzeroDigit := {d : ℕ // 1 ≤ d ∧ d ≤ 9}

/-- An is an n-digit integer with all digits equal to a -/
def An (a : NonzeroDigit) (n : ℕ+) : ℕ := a.val * (10^n.val - 1) / 9

/-- Bn is an n-digit integer with all digits equal to b -/
def Bn (b : NonzeroDigit) (n : ℕ+) : ℕ := b.val * (10^n.val - 1) / 9

/-- Cn is a 3n-digit integer with all digits equal to c -/
def Cn (c : NonzeroDigit) (n : ℕ+) : ℕ := c.val * (10^(3*n.val) - 1) / 9

/-- The equation Cn - Bn = An^3 is satisfied for at least two values of n -/
def SatisfiesEquation (a b c : NonzeroDigit) : Prop :=
  ∃ n₁ n₂ : ℕ+, n₁ ≠ n₂ ∧ Cn c n₁ - Bn b n₁ = (An a n₁)^3 ∧ Cn c n₂ - Bn b n₂ = (An a n₂)^3

theorem max_sum_of_digits (a b c : NonzeroDigit) (h : SatisfiesEquation a b c) :
  a.val + b.val + c.val ≤ 19 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_l3838_383836


namespace NUMINAMATH_CALUDE_distance_after_two_hours_l3838_383871

/-- Anna's jogging speed in miles per minute -/
def anna_speed : ℚ := 1 / 20

/-- Mark's running speed in miles per minute -/
def mark_speed : ℚ := 3 / 40

/-- The time period in minutes -/
def time_period : ℚ := 2 * 60

/-- The theorem stating the distance between Anna and Mark after 2 hours -/
theorem distance_after_two_hours :
  anna_speed * time_period + mark_speed * time_period = 9 := by sorry

end NUMINAMATH_CALUDE_distance_after_two_hours_l3838_383871


namespace NUMINAMATH_CALUDE_circle_equation_correct_l3838_383812

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Define the center of the circle
def center : ℝ × ℝ := (1, 1)

-- Define the point that the circle passes through
def point_on_circle : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem circle_equation_correct :
  -- The circle passes through the point (1, 0)
  circle_equation point_on_circle.1 point_on_circle.2 ∧
  -- The center is at the intersection of x=1 and x+y=2
  center.1 = 1 ∧ center.1 + center.2 = 2 ∧
  -- The equation represents a circle with the given center
  ∀ x y : ℝ, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l3838_383812


namespace NUMINAMATH_CALUDE_saturday_zoo_visitors_l3838_383842

/-- The number of people who visited the zoo on Friday -/
def friday_visitors : ℕ := 1250

/-- The number of people who visited the zoo on Saturday -/
def saturday_visitors : ℕ := 3 * friday_visitors

/-- Theorem stating the number of visitors on Saturday -/
theorem saturday_zoo_visitors : saturday_visitors = 3750 := by sorry

end NUMINAMATH_CALUDE_saturday_zoo_visitors_l3838_383842


namespace NUMINAMATH_CALUDE_jerry_has_36_stickers_l3838_383859

/-- Given the number of stickers for Fred, calculate the number of stickers for Jerry. -/
def jerrys_stickers (freds_stickers : ℕ) : ℕ :=
  let georges_stickers := freds_stickers - 6
  3 * georges_stickers

/-- Prove that Jerry has 36 stickers given the conditions in the problem. -/
theorem jerry_has_36_stickers :
  jerrys_stickers 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_jerry_has_36_stickers_l3838_383859


namespace NUMINAMATH_CALUDE_local_min_implies_a_eq_2_l3838_383884

/-- The function f(x) = x(x-a)^2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x * (x - a)^2

/-- f has a local minimum at x = 2 -/
def has_local_min_at_2 (a : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - 2| < δ → f a x ≥ f a 2

theorem local_min_implies_a_eq_2 :
  ∀ a : ℝ, has_local_min_at_2 a → a = 2 := by sorry

end NUMINAMATH_CALUDE_local_min_implies_a_eq_2_l3838_383884


namespace NUMINAMATH_CALUDE_inequality_proof_l3838_383853

theorem inequality_proof (a b : ℝ) (ha : a > 1) (hb : b > 1) : 
  (a^2 / (b - 1)) + (b^2 / (a - 1)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3838_383853


namespace NUMINAMATH_CALUDE_negation_existence_proposition_l3838_383845

theorem negation_existence_proposition :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_existence_proposition_l3838_383845


namespace NUMINAMATH_CALUDE_flu_outbreak_l3838_383834

theorem flu_outbreak (initial_infected : ℕ) (infected_after_two_rounds : ℕ) :
  initial_infected = 1 →
  infected_after_two_rounds = 81 →
  ∃ (avg_infected_per_round : ℕ),
    avg_infected_per_round = 8 ∧
    initial_infected + avg_infected_per_round + avg_infected_per_round * (avg_infected_per_round + 1) = infected_after_two_rounds ∧
    infected_after_two_rounds * avg_infected_per_round + infected_after_two_rounds = 729 :=
by sorry

end NUMINAMATH_CALUDE_flu_outbreak_l3838_383834


namespace NUMINAMATH_CALUDE_gumball_theorem_l3838_383878

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine :=
  (red : Nat)
  (white : Nat)
  (blue : Nat)
  (green : Nat)

/-- The least number of gumballs needed to guarantee four of the same color -/
def leastGumballsNeeded (machine : GumballMachine) : Nat :=
  13

/-- Theorem stating that for the given gumball machine, 
    the least number of gumballs needed is 13 -/
theorem gumball_theorem (machine : GumballMachine) 
  (h1 : machine.red = 10)
  (h2 : machine.white = 9)
  (h3 : machine.blue = 8)
  (h4 : machine.green = 7) :
  leastGumballsNeeded machine = 13 := by
  sorry

#check gumball_theorem

end NUMINAMATH_CALUDE_gumball_theorem_l3838_383878


namespace NUMINAMATH_CALUDE_residue_negative_437_mod_13_l3838_383826

theorem residue_negative_437_mod_13 :
  ∃ (k : ℤ), -437 = 13 * k + 5 ∧ (0 : ℤ) ≤ 5 ∧ 5 < 13 := by
  sorry

end NUMINAMATH_CALUDE_residue_negative_437_mod_13_l3838_383826


namespace NUMINAMATH_CALUDE_girls_percentage_in_class_l3838_383831

theorem girls_percentage_in_class (total_students : ℕ) 
  (boys_basketball_percentage : ℚ)
  (girls_basketball_ratio : ℚ)
  (girls_basketball_percentage : ℚ) :
  total_students = 25 →
  boys_basketball_percentage = 2/5 →
  girls_basketball_ratio = 2 →
  girls_basketball_percentage = 4/5 →
  (∃ (girls_percentage : ℚ), girls_percentage = 3/5) :=
by sorry

end NUMINAMATH_CALUDE_girls_percentage_in_class_l3838_383831


namespace NUMINAMATH_CALUDE_partnership_profit_l3838_383890

/-- Calculates the total profit of a partnership given the investments and one partner's share of the profit. -/
def calculate_total_profit (a_investment b_investment c_investment c_profit : ℕ) : ℕ :=
  let ratio_sum := (a_investment / (a_investment.gcd b_investment).gcd c_investment) +
                   (b_investment / (a_investment.gcd b_investment).gcd c_investment) +
                   (c_investment / (a_investment.gcd b_investment).gcd c_investment)
  let c_ratio := c_investment / (a_investment.gcd b_investment).gcd c_investment
  (ratio_sum * c_profit) / c_ratio

/-- The total profit of the partnership is 80000 given the investments and C's profit share. -/
theorem partnership_profit :
  calculate_total_profit 27000 72000 81000 36000 = 80000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_l3838_383890


namespace NUMINAMATH_CALUDE_gum_pack_size_l3838_383848

/-- The number of pieces of banana gum Luke has initially -/
def banana_gum : ℕ := 28

/-- The number of pieces of apple gum Luke has initially -/
def apple_gum : ℕ := 36

/-- The number of pieces of gum in each complete pack -/
def y : ℕ := 14

theorem gum_pack_size :
  (banana_gum - 2 * y) * (apple_gum + 3 * y) = banana_gum * apple_gum := by
  sorry

#check gum_pack_size

end NUMINAMATH_CALUDE_gum_pack_size_l3838_383848


namespace NUMINAMATH_CALUDE_sector_area_l3838_383807

/-- The area of a circular sector given its arc length and radius -/
theorem sector_area (l r : ℝ) (hl : l > 0) (hr : r > 0) : 
  (l * r) / 2 = (l * r) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3838_383807


namespace NUMINAMATH_CALUDE_range_of_a_l3838_383874

open Set Real

theorem range_of_a (a : ℝ) : 
  let A : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 3}
  let B : Set ℝ := Ioi 5
  (A ∩ B = ∅) → a ∈ Iic 2 ∪ Ici 3 := by
sorry


end NUMINAMATH_CALUDE_range_of_a_l3838_383874


namespace NUMINAMATH_CALUDE_equiangular_polygons_unique_angle_l3838_383804

theorem equiangular_polygons_unique_angle : ∃! x : ℝ,
  0 < x ∧ x < 180 ∧
  ∃ n₁ : ℕ, n₁ ≥ 3 ∧ x = 180 - 360 / n₁ ∧
  ∃ n₃ : ℕ, n₃ ≥ 3 ∧ 3/2 * x = 180 - 360 / n₃ ∧
  n₁ ≠ n₃ := by sorry

end NUMINAMATH_CALUDE_equiangular_polygons_unique_angle_l3838_383804


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3838_383854

/-- A quadratic expression of the form 15x^2 + ax + 15 can be factored into two linear binomial 
    factors with integer coefficients if and only if a = 34 -/
theorem quadratic_factorization (a : ℤ) : 
  (∃ (m n p q : ℤ), 15 * X^2 + a * X + 15 = (m * X + n) * (p * X + q)) ↔ a = 34 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3838_383854


namespace NUMINAMATH_CALUDE_spelling_contest_questions_spelling_contest_total_questions_l3838_383865

/-- Given a spelling contest with two competitors, Drew and Carla, prove the total number of questions asked. -/
theorem spelling_contest_questions (drew_correct : ℕ) (drew_wrong : ℕ) (carla_correct : ℕ) : ℕ :=
  let drew_total := drew_correct + drew_wrong
  let carla_wrong := 2 * drew_wrong
  let carla_total := carla_correct + carla_wrong
  drew_total + carla_total

/-- Prove that the total number of questions in the spelling contest is 52. -/
theorem spelling_contest_total_questions : spelling_contest_questions 20 6 14 = 52 := by
  sorry

end NUMINAMATH_CALUDE_spelling_contest_questions_spelling_contest_total_questions_l3838_383865


namespace NUMINAMATH_CALUDE_minimum_workers_needed_l3838_383827

/-- The number of units completed per worker per day in the first process -/
def process1_rate : ℕ := 48

/-- The number of units completed per worker per day in the second process -/
def process2_rate : ℕ := 32

/-- The number of units completed per worker per day in the third process -/
def process3_rate : ℕ := 28

/-- The minimum number of workers needed for the first process -/
def workers1 : ℕ := 14

/-- The minimum number of workers needed for the second process -/
def workers2 : ℕ := 21

/-- The minimum number of workers needed for the third process -/
def workers3 : ℕ := 24

/-- The theorem stating the minimum number of workers needed for each process -/
theorem minimum_workers_needed :
  (∃ n : ℕ, n > 0 ∧ 
    n = process1_rate * workers1 ∧ 
    n = process2_rate * workers2 ∧ 
    n = process3_rate * workers3) ∧
  (∀ w1 w2 w3 : ℕ, 
    (∃ m : ℕ, m > 0 ∧ 
      m = process1_rate * w1 ∧ 
      m = process2_rate * w2 ∧ 
      m = process3_rate * w3) →
    w1 ≥ workers1 ∧ w2 ≥ workers2 ∧ w3 ≥ workers3) :=
by sorry

end NUMINAMATH_CALUDE_minimum_workers_needed_l3838_383827


namespace NUMINAMATH_CALUDE_parabola_properties_l3838_383818

/-- Parabola C: y^2 = x with focus F -/
structure Parabola where
  focus : ℝ × ℝ
  equation : (ℝ × ℝ) → Prop

/-- Point on the parabola -/
structure PointOnParabola (C : Parabola) where
  point : ℝ × ℝ
  on_parabola : C.equation point

/-- Theorem about the slope of line AB and the length of AB when collinear with focus -/
theorem parabola_properties (C : Parabola) 
    (hC : C.focus = (1/4, 0) ∧ C.equation = fun p => p.2^2 = p.1) 
    (A B : PointOnParabola C) 
    (hAB : A.point ≠ B.point ∧ A.point ≠ (0, 0) ∧ B.point ≠ (0, 0)) :
  (∃ k : ℝ, k = (A.point.2 - B.point.2) / (A.point.1 - B.point.1) → 
    k = 1 / (A.point.2 + B.point.2)) ∧
  (∃ AB : ℝ, (∃ t : ℝ, (1 - t) • A.point + t • B.point = C.focus) → 
    AB = A.point.1 + B.point.1 + 1/2) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l3838_383818


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3838_383893

theorem quadratic_factorization (x : ℝ) : 15 * x^2 + 10 * x - 20 = 5 * (x - 1) * (3 * x + 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3838_383893


namespace NUMINAMATH_CALUDE_cylinder_radius_l3838_383800

/-- Given a cylinder with height 8 cm and surface area 130π cm², prove its base circle radius is 5 cm. -/
theorem cylinder_radius (h : ℝ) (S : ℝ) (r : ℝ) 
  (height_eq : h = 8)
  (surface_area_eq : S = 130 * Real.pi)
  (surface_area_formula : S = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) :
  r = 5 := by sorry

end NUMINAMATH_CALUDE_cylinder_radius_l3838_383800


namespace NUMINAMATH_CALUDE_topsoil_cost_l3838_383895

/-- The cost of premium topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil to be purchased -/
def cubic_yards_to_purchase : ℝ := 7

/-- Theorem: The total cost of purchasing 7 cubic yards of premium topsoil is 1512 dollars -/
theorem topsoil_cost : 
  cost_per_cubic_foot * cubic_feet_per_cubic_yard * cubic_yards_to_purchase = 1512 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_l3838_383895


namespace NUMINAMATH_CALUDE_position_of_2008_l3838_383810

/-- Define the position of a number in the pattern -/
structure Position where
  row : Nat
  column : Nat

/-- Function to calculate the position of a number in the pattern -/
noncomputable def calculatePosition (n : Nat) : Position :=
  sorry  -- The actual implementation would go here

/-- Theorem stating that 2008 is in row 18, column 45 -/
theorem position_of_2008 : calculatePosition 2008 = ⟨18, 45⟩ := by
  sorry

#check position_of_2008

end NUMINAMATH_CALUDE_position_of_2008_l3838_383810


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l3838_383820

/-- Represents the capacity of a tank with a leak and two inlet pipes. -/
def tank_capacity : Real :=
  let leak_rate : Real := tank_capacity / 6
  let pipe_a_rate : Real := 3.5 * 60
  let pipe_b_rate : Real := 4.5 * 60
  let net_rate_both_pipes : Real := pipe_a_rate + pipe_b_rate - leak_rate
  let net_rate_pipe_a : Real := pipe_a_rate - leak_rate
  tank_capacity

/-- Theorem stating the capacity of the tank under given conditions. -/
theorem tank_capacity_proof :
  let leak_rate : Real := tank_capacity / 6
  let pipe_a_rate : Real := 3.5 * 60
  let pipe_b_rate : Real := 4.5 * 60
  let net_rate_both_pipes : Real := pipe_a_rate + pipe_b_rate - leak_rate
  let net_rate_pipe_a : Real := pipe_a_rate - leak_rate
  (net_rate_pipe_a * 1 + net_rate_both_pipes * 7 - leak_rate * 8 = 0) →
  tank_capacity = 1338.75 := by
  sorry

#eval tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_proof_l3838_383820


namespace NUMINAMATH_CALUDE_fishing_trips_l3838_383864

theorem fishing_trips (shelly_catch : ℕ) (sam_catch : ℕ) (total_catch : ℕ) :
  shelly_catch = 3 →
  sam_catch = 2 →
  total_catch = 25 →
  (total_catch / (shelly_catch + sam_catch) : ℕ) = 5 := by
sorry

end NUMINAMATH_CALUDE_fishing_trips_l3838_383864


namespace NUMINAMATH_CALUDE_parallel_line_equation_l3838_383851

/-- The equation of a line passing through (-1, 2) and parallel to 2x + y - 5 = 0 is 2x + y = 0 -/
theorem parallel_line_equation :
  let P : ℝ × ℝ := (-1, 2)
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ 2 * x + y - 5 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ 2 * x + y = 0
  (∀ x y, l₂ x y ↔ (2 * P.1 + P.2 = 2 * x + y ∧ ∀ x₁ y₁ x₂ y₂, l₁ x₁ y₁ → l₁ x₂ y₂ → 
    2 * (x₂ - x₁) = y₁ - y₂)) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l3838_383851


namespace NUMINAMATH_CALUDE_inequality_proof_l3838_383868

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3838_383868


namespace NUMINAMATH_CALUDE_logarithm_exponent_equality_special_case_2019_l3838_383892

theorem logarithm_exponent_equality (x : ℝ) (hx : x > 1) : 
  x^(Real.log (Real.log x)) = (Real.log x)^(Real.log x) :=
by
  sorry

-- The main theorem
theorem special_case_2019 : 
  2019^(Real.log (Real.log 2019)) - (Real.log 2019)^(Real.log 2019) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_logarithm_exponent_equality_special_case_2019_l3838_383892


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l3838_383832

theorem sum_of_fractions_equals_one 
  (p q r u v w : ℝ) 
  (eq1 : 17 * u + q * v + r * w = 0)
  (eq2 : p * u + 29 * v + r * w = 0)
  (eq3 : p * u + q * v + 56 * w = 0)
  (h_p : p ≠ 17)
  (h_u : u ≠ 0) :
  p / (p - 17) + q / (q - 29) + r / (r - 56) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l3838_383832


namespace NUMINAMATH_CALUDE_units_digit_of_sum_factorials_l3838_383823

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

def sum_factorials (n : ℕ) : ℕ := (Finset.range n).sum (λ i => factorial (i + 1))

theorem units_digit_of_sum_factorials :
  (sum_factorials 100) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_factorials_l3838_383823


namespace NUMINAMATH_CALUDE_planting_methods_eq_120_l3838_383879

/-- The number of ways to select and plant vegetables -/
def plantingMethods (totalVarieties : ℕ) (selectedVarieties : ℕ) (plots : ℕ) : ℕ :=
  Nat.choose totalVarieties selectedVarieties * Nat.factorial plots

/-- Theorem stating the number of planting methods for the given scenario -/
theorem planting_methods_eq_120 :
  plantingMethods 5 4 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_planting_methods_eq_120_l3838_383879


namespace NUMINAMATH_CALUDE_mica_pasta_purchase_l3838_383862

theorem mica_pasta_purchase (pasta_price : ℝ) (beef_price : ℝ) (sauce_price : ℝ) 
  (quesadilla_price : ℝ) (total_budget : ℝ) :
  pasta_price = 1.5 →
  beef_price = 8 →
  sauce_price = 2 →
  quesadilla_price = 6 →
  total_budget = 15 →
  (total_budget - (beef_price * 0.25 + sauce_price * 2 + quesadilla_price)) / pasta_price = 2 :=
by
  sorry

#check mica_pasta_purchase

end NUMINAMATH_CALUDE_mica_pasta_purchase_l3838_383862


namespace NUMINAMATH_CALUDE_train_journey_length_l3838_383866

theorem train_journey_length 
  (speed_on_time : ℝ) 
  (speed_late : ℝ) 
  (late_time : ℝ) 
  (h1 : speed_on_time = 100)
  (h2 : speed_late = 80)
  (h3 : late_time = 1/3)
  : ∃ (distance : ℝ), distance = 400/3 ∧ 
    distance / speed_on_time = distance / speed_late - late_time :=
by sorry

end NUMINAMATH_CALUDE_train_journey_length_l3838_383866


namespace NUMINAMATH_CALUDE_arithmetic_sequence_b_formula_l3838_383838

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def b (a : ℕ → ℝ) (n : ℕ) : ℝ := a (3^n)

theorem arithmetic_sequence_b_formula (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 = 8 →
  a 8 = 26 →
  ∀ n : ℕ, b a n = 3^(n+1) + 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_b_formula_l3838_383838


namespace NUMINAMATH_CALUDE_square_minus_three_times_l3838_383867

/-- The expression "square of a minus 3 times b" is equivalent to a^2 - 3*b -/
theorem square_minus_three_times (a b : ℝ) : (a^2 - 3*b) = (a^2 - 3*b) := by sorry

end NUMINAMATH_CALUDE_square_minus_three_times_l3838_383867


namespace NUMINAMATH_CALUDE_closest_to_200_l3838_383813

def problem_value : ℝ := 2.54 * 7.89 * (4.21 + 5.79)

def options : List ℝ := [150, 200, 250, 300, 350]

theorem closest_to_200 :
  ∀ x ∈ options, x ≠ 200 → |problem_value - 200| < |problem_value - x| :=
by sorry

end NUMINAMATH_CALUDE_closest_to_200_l3838_383813


namespace NUMINAMATH_CALUDE_multiplications_in_thirty_minutes_l3838_383856

/-- Represents the number of multiplications a computer can perform per second -/
def multiplications_per_second : ℕ := 20000

/-- Represents the number of minutes we want to calculate for -/
def minutes : ℕ := 30

/-- Represents the number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- Theorem stating that the computer can perform 36,000,000 multiplications in 30 minutes -/
theorem multiplications_in_thirty_minutes :
  multiplications_per_second * minutes * seconds_per_minute = 36000000 := by
  sorry

end NUMINAMATH_CALUDE_multiplications_in_thirty_minutes_l3838_383856


namespace NUMINAMATH_CALUDE_partner_calculation_l3838_383824

theorem partner_calculation (x : ℝ) : 3 * (3 * (x + 2) - 2) = 3 * (3 * x + 4) := by
  sorry

#check partner_calculation

end NUMINAMATH_CALUDE_partner_calculation_l3838_383824


namespace NUMINAMATH_CALUDE_soda_difference_l3838_383896

theorem soda_difference (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 79) (h2 : diet_soda = 53) : 
  regular_soda - diet_soda = 26 := by
  sorry

end NUMINAMATH_CALUDE_soda_difference_l3838_383896


namespace NUMINAMATH_CALUDE_square_1011_position_l3838_383822

-- Define the possible positions of the square
inductive SquarePosition
| ABCD
| BCDA
| DCBA

-- Define the transformation function
def transform (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.BCDA
  | SquarePosition.BCDA => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.ABCD

-- Define the function to get the nth position
def nthPosition (n : Nat) : SquarePosition :=
  match n % 4 with
  | 0 => SquarePosition.ABCD
  | 1 => SquarePosition.BCDA
  | 2 => SquarePosition.DCBA
  | _ => SquarePosition.ABCD

-- Theorem stating that the 1011th square is in position DCBA
theorem square_1011_position :
  nthPosition 1011 = SquarePosition.DCBA := by
  sorry

end NUMINAMATH_CALUDE_square_1011_position_l3838_383822


namespace NUMINAMATH_CALUDE_page_lines_increase_l3838_383886

theorem page_lines_increase (original : ℕ) (new : ℕ) (increase_percent : ℚ) : 
  new = 240 ∧ 
  increase_percent = 50 ∧ 
  new = original + (increase_percent / 100 : ℚ) * original →
  new - original = 80 := by
  sorry

end NUMINAMATH_CALUDE_page_lines_increase_l3838_383886


namespace NUMINAMATH_CALUDE_inequality_solution_set_F_zero_points_range_l3838_383815

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) : ℝ := |2*x - 1|

-- Define the inequality
def inequality (x : ℝ) : Prop := f (x + 5) ≤ x * g x

-- Define the function F
def F (x a : ℝ) : ℝ := f (x + 2) + f x + a

-- Theorem for the solution set of the inequality
theorem inequality_solution_set :
  {x : ℝ | inequality x} = {x : ℝ | x ≥ 2} :=
sorry

-- Theorem for the range of a when F has zero points
theorem F_zero_points_range (a : ℝ) :
  (∃ x, F x a = 0) ↔ a ∈ Set.Iic (-2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_F_zero_points_range_l3838_383815


namespace NUMINAMATH_CALUDE_area_covered_by_strips_l3838_383876

/-- Represents a rectangular strip -/
structure Strip where
  length : ℝ
  width : ℝ

/-- Calculates the area of a strip -/
def stripArea (s : Strip) : ℝ := s.length * s.width

/-- Calculates the total area of strips without considering overlaps -/
def totalAreaNoOverlap (strips : List Strip) : ℝ :=
  (strips.map stripArea).sum

/-- Represents an overlap between two strips -/
structure Overlap where
  length : ℝ
  width : ℝ

/-- Calculates the area of an overlap -/
def overlapArea (o : Overlap) : ℝ := o.length * o.width

/-- Calculates the total area of overlaps -/
def totalOverlapArea (overlaps : List Overlap) : ℝ :=
  (overlaps.map overlapArea).sum

/-- Theorem: The area covered by five strips with given dimensions and overlaps is 58 -/
theorem area_covered_by_strips :
  let strips : List Strip := List.replicate 5 ⟨12, 1⟩
  let overlaps : List Overlap := List.replicate 4 ⟨0.5, 1⟩
  totalAreaNoOverlap strips - totalOverlapArea overlaps = 58 := by
  sorry

end NUMINAMATH_CALUDE_area_covered_by_strips_l3838_383876


namespace NUMINAMATH_CALUDE_tileC_in_rectangleY_l3838_383817

-- Define a tile with four sides
structure Tile :=
  (top : ℕ) (right : ℕ) (bottom : ℕ) (left : ℕ)

-- Define the four tiles
def tileA : Tile := ⟨5, 3, 1, 6⟩
def tileB : Tile := ⟨3, 6, 2, 5⟩
def tileC : Tile := ⟨2, 7, 0, 3⟩
def tileD : Tile := ⟨6, 2, 4, 7⟩

-- Define a function to check if a tile has unique sides
def hasUniqueSides (t : Tile) (others : List Tile) : Prop :=
  (t.right ∉ others.map (λ tile => tile.left)) ∧
  (t.bottom ∉ others.map (λ tile => tile.top))

-- Define the theorem
theorem tileC_in_rectangleY :
  hasUniqueSides tileC [tileA, tileB, tileD] ∧
  ¬hasUniqueSides tileA [tileB, tileC, tileD] ∧
  ¬hasUniqueSides tileB [tileA, tileC, tileD] ∧
  ¬hasUniqueSides tileD [tileA, tileB, tileC] :=
sorry

end NUMINAMATH_CALUDE_tileC_in_rectangleY_l3838_383817


namespace NUMINAMATH_CALUDE_manufacturing_sector_degrees_l3838_383880

/-- Represents the number of degrees in a full circle -/
def full_circle_degrees : ℝ := 360

/-- Represents the percentage of the circle occupied by the manufacturing department -/
def manufacturing_percentage : ℝ := 45

/-- Theorem: The manufacturing department sector in the circle graph occupies 162 degrees -/
theorem manufacturing_sector_degrees : 
  (manufacturing_percentage / 100) * full_circle_degrees = 162 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_sector_degrees_l3838_383880


namespace NUMINAMATH_CALUDE_max_sin_product_right_triangle_l3838_383897

/-- For any right triangle ABC with angle C = 90°, the maximum value of sin A sin B is 1/2. -/
theorem max_sin_product_right_triangle (A B C : Real) : 
  0 ≤ A ∧ 0 ≤ B ∧ -- Angles are non-negative
  A + B + C = π ∧ -- Sum of angles in a triangle is π
  C = π / 2 → -- Right angle at C
  ∀ (x y : Real), 0 ≤ x ∧ 0 ≤ y ∧ x + y + π/2 = π → 
    Real.sin A * Real.sin B ≤ Real.sin x * Real.sin y ∧
    Real.sin A * Real.sin B ≤ (1 : Real) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_sin_product_right_triangle_l3838_383897


namespace NUMINAMATH_CALUDE_four_digit_multiples_of_five_l3838_383821

theorem four_digit_multiples_of_five : 
  (Finset.filter (fun n => n % 5 = 0) (Finset.range 9000)).card = 1800 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_multiples_of_five_l3838_383821


namespace NUMINAMATH_CALUDE_seven_fourth_mod_hundred_l3838_383806

theorem seven_fourth_mod_hundred : 7^4 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_seven_fourth_mod_hundred_l3838_383806


namespace NUMINAMATH_CALUDE_polyhedron_edges_l3838_383872

theorem polyhedron_edges (F V E : ℕ) : F + V - E = 2 → F = 6 → V = 8 → E = 12 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_edges_l3838_383872


namespace NUMINAMATH_CALUDE_max_quarters_sasha_l3838_383875

/-- Represents the value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- Represents the value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- Represents the value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- Represents the total amount Sasha has in dollars -/
def total_amount : ℚ := 480 / 100

theorem max_quarters_sasha : 
  ∀ q : ℕ, 
    (q : ℚ) * quarter_value + 
    (2 * q : ℚ) * nickel_value + 
    (q : ℚ) * dime_value ≤ total_amount → 
    q ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_max_quarters_sasha_l3838_383875


namespace NUMINAMATH_CALUDE_age_difference_l3838_383888

def age_problem (a b c : ℕ) : Prop :=
  b = 2 * c ∧ a + b + c = 27 ∧ b = 10

theorem age_difference (a b c : ℕ) (h : age_problem a b c) : a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3838_383888


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3838_383841

def M : Set Nat := {1, 2, 3}
def N : Set Nat := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3838_383841


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3838_383873

/-- The equation of a circle in polar coordinates -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

/-- The equation of a line in polar coordinates -/
def line_equation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

/-- Theorem stating that the line is tangent to the circle -/
theorem line_tangent_to_circle :
  ∃ (ρ₀ θ₀ : ℝ), circle_equation ρ₀ θ₀ ∧ line_equation ρ₀ θ₀ ∧
  ∀ (ρ θ : ℝ), circle_equation ρ θ ∧ line_equation ρ θ → (ρ, θ) = (ρ₀, θ₀) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3838_383873


namespace NUMINAMATH_CALUDE_sum_of_twos_and_threes_l3838_383843

/-- The number of ways to write 1800 as a sum of twos and threes with exactly 600 terms -/
def num_ways : ℕ :=
  (Finset.range 301).card

theorem sum_of_twos_and_threes :
  num_ways = 301 ∧
  ∀ n : ℕ, n ∈ Finset.range 301 →
    ∃ x y : ℕ,
      2 * x + 3 * y = 1800 ∧
      x + y = 600 ∧
      x = 3 * n ∧
      y = 2 * (300 - n) :=
by sorry

#eval num_ways

end NUMINAMATH_CALUDE_sum_of_twos_and_threes_l3838_383843


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3838_383819

def U : Set Int := {-1, -2, -3, -4, 0}
def A : Set Int := {-1, -2, 0}
def B : Set Int := {-3, -4, 0}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {-3, -4} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3838_383819


namespace NUMINAMATH_CALUDE_ab_length_not_unique_l3838_383808

/-- Given two line segments AC and BC with lengths 1 and 3 respectively,
    the length of AB cannot be uniquely determined. -/
theorem ab_length_not_unique (AC BC : ℝ) (hAC : AC = 1) (hBC : BC = 3) :
  ¬ ∃! AB : ℝ, (0 < AB ∧ AB < AC + BC) ∨ (AB = AC + BC ∨ AB = |BC - AC|) :=
sorry

end NUMINAMATH_CALUDE_ab_length_not_unique_l3838_383808


namespace NUMINAMATH_CALUDE_min_groups_for_class_l3838_383811

/-- Given a class of 30 students and a maximum group size of 12,
    proves that the minimum number of equal-sized groups is 3. -/
theorem min_groups_for_class (total_students : ℕ) (max_group_size : ℕ) :
  total_students = 30 →
  max_group_size = 12 →
  ∃ (group_size : ℕ), 
    group_size ≤ max_group_size ∧
    total_students % group_size = 0 ∧
    (total_students / group_size = 3) ∧
    ∀ (other_size : ℕ), 
      other_size ≤ max_group_size →
      total_students % other_size = 0 →
      total_students / other_size ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_groups_for_class_l3838_383811


namespace NUMINAMATH_CALUDE_complex_number_simplification_l3838_383830

theorem complex_number_simplification :
  3 * (2 - 5 * Complex.I) - 4 * (1 + 3 * Complex.I) = 2 - 27 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l3838_383830


namespace NUMINAMATH_CALUDE_vector_relation_in_triangle_l3838_383839

/-- Given a triangle ABC and a point D, if AB = 4DB, then CD = (1/4)CA + (3/4)CB -/
theorem vector_relation_in_triangle (A B C D : EuclideanSpace ℝ (Fin 3)) :
  (B - A) = 4 • (B - D) →
  (D - C) = (1/4) • (A - C) + (3/4) • (B - C) := by
  sorry

end NUMINAMATH_CALUDE_vector_relation_in_triangle_l3838_383839


namespace NUMINAMATH_CALUDE_share_difference_after_tax_l3838_383861

/-- Represents the share ratios for p, q, r, and s respectively -/
def shareRatios : Fin 4 → ℕ
  | 0 => 3
  | 1 => 7
  | 2 => 12
  | 3 => 5

/-- Represents the tax rates for p, q, r, and s respectively -/
def taxRates : Fin 4 → ℚ
  | 0 => 1/10
  | 1 => 15/100
  | 2 => 1/5
  | 3 => 1/4

/-- The difference between p and q's shares after tax deduction -/
def differenceAfterTax : ℚ := 2400

theorem share_difference_after_tax :
  let x : ℚ := differenceAfterTax / (shareRatios 1 * (1 - taxRates 1) - shareRatios 0 * (1 - taxRates 0))
  let qShare : ℚ := shareRatios 1 * x * (1 - taxRates 1)
  let rShare : ℚ := shareRatios 2 * x * (1 - taxRates 2)
  abs (rShare - qShare - 2695.38) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_share_difference_after_tax_l3838_383861


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3838_383814

/-- Given a line L1 with equation 3x - 6y = 9, prove that the line L2 with equation y = (1/2)x - 1
    is parallel to L1 and passes through the point (2,0). -/
theorem parallel_line_through_point (x y : ℝ) : 
  (3 * x - 6 * y = 9) →  -- Equation of line L1
  (y = (1/2) * x - 1) →  -- Equation of line L2
  (∃ m b : ℝ, y = m * x + b ∧ m = 1/2) →  -- L2 is in slope-intercept form with slope 1/2
  (0 = (1/2) * 2 - 1) →  -- L2 passes through (2,0)
  (∀ x₁ y₁ x₂ y₂ : ℝ, (3 * x₁ - 6 * y₁ = 9 ∧ 3 * x₂ - 6 * y₂ = 9) → 
    ((y₂ - y₁) / (x₂ - x₁) = 1/2)) →  -- Slope of L1 is 1/2
  (y = (1/2) * x - 1)  -- Conclusion: equation of L2
  := by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3838_383814


namespace NUMINAMATH_CALUDE_decimal_operation_order_l3838_383887

theorem decimal_operation_order : ¬ ∀ (a b c : ℚ), a + b - c = a + (b - c) := by
  sorry

end NUMINAMATH_CALUDE_decimal_operation_order_l3838_383887


namespace NUMINAMATH_CALUDE_player_A_can_destroy_six_cups_six_cups_is_maximum_l3838_383852

/-- Represents the state of the game with cups and pebbles -/
structure GameState where
  cups : ℕ
  pebbles : List ℕ

/-- Represents a move in the game -/
inductive Move
  | redistribute : List ℕ → Move
  | destroy_empty : Move
  | switch : ℕ → ℕ → Move

/-- Player A's strategy function -/
def player_A_strategy (state : GameState) : List ℕ :=
  sorry

/-- Player B's action function -/
def player_B_action (state : GameState) (move : Move) : GameState :=
  sorry

/-- Simulates the game for a given number of moves -/
def play_game (initial_state : GameState) (num_moves : ℕ) : GameState :=
  sorry

/-- Theorem stating that player A can guarantee at least 6 cups are destroyed -/
theorem player_A_can_destroy_six_cups :
  ∃ (strategy : GameState → List ℕ),
    ∀ (num_moves : ℕ),
      let final_state := play_game {cups := 10, pebbles := List.replicate 10 10} num_moves
      final_state.cups ≤ 4 :=
sorry

/-- Theorem stating that 6 is the maximum number of cups that can be guaranteed to be destroyed -/
theorem six_cups_is_maximum :
  ∀ (strategy : GameState → List ℕ),
    ∃ (num_moves : ℕ),
      let final_state := play_game {cups := 10, pebbles := List.replicate 10 10} num_moves
      final_state.cups > 4 :=
sorry

end NUMINAMATH_CALUDE_player_A_can_destroy_six_cups_six_cups_is_maximum_l3838_383852


namespace NUMINAMATH_CALUDE_stating_roper_lawn_cutting_l3838_383825

/-- Represents the number of times Mr. Roper cuts his lawn in different periods --/
structure LawnCutting where
  summer_months : ℕ  -- Number of months from April to September
  winter_months : ℕ  -- Number of months from October to March
  summer_cuts : ℕ    -- Number of cuts per month in summer
  average_cuts : ℕ   -- Average number of cuts per month over a year
  total_months : ℕ   -- Total number of months in a year

/-- 
Theorem stating that given the conditions, 
Mr. Roper cuts his lawn 3 times a month from October to March 
-/
theorem roper_lawn_cutting (l : LawnCutting) 
  (h1 : l.summer_months = 6)
  (h2 : l.winter_months = 6)
  (h3 : l.summer_cuts = 15)
  (h4 : l.average_cuts = 9)
  (h5 : l.total_months = 12) :
  (l.total_months * l.average_cuts - l.summer_months * l.summer_cuts) / l.winter_months = 3 := by
  sorry

#check roper_lawn_cutting

end NUMINAMATH_CALUDE_stating_roper_lawn_cutting_l3838_383825


namespace NUMINAMATH_CALUDE_at_least_three_same_purchase_l3838_383849

/-- Represents a purchase combination of items -/
structure Purchase where
  threeYuanItems : Nat
  fiveYuanItems : Nat
  deriving Repr

/-- The set of all valid purchase combinations -/
def validPurchases : Finset Purchase :=
  sorry

/-- The number of valid purchase combinations -/
def numCombinations : Nat :=
  Finset.card validPurchases

theorem at_least_three_same_purchase (n : Nat) (h : n = 25) :
  ∀ (purchases : Fin n → Purchase),
    (∀ i, purchases i ∈ validPurchases) →
    ∃ (p : Purchase) (i j k : Fin n),
      i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
      purchases i = p ∧ purchases j = p ∧ purchases k = p :=
  sorry

end NUMINAMATH_CALUDE_at_least_three_same_purchase_l3838_383849


namespace NUMINAMATH_CALUDE_triangle_side_length_l3838_383891

/-- 
Given a triangle XYZ where:
- y = 7
- z = 3
- cos(Y - Z) = 40/41
Prove that x² = 56.1951
-/
theorem triangle_side_length (X Y Z : ℝ) (x y z : ℝ) :
  y = 7 →
  z = 3 →
  Real.cos (Y - Z) = 40 / 41 →
  x ^ 2 = 56.1951 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3838_383891


namespace NUMINAMATH_CALUDE_rays_grocery_bill_l3838_383869

def hamburger_price : ℝ := 5.00
def crackers_price : ℝ := 3.50
def vegetable_price : ℝ := 2.00
def vegetable_bags : ℕ := 4
def cheese_price : ℝ := 3.50
def discount_rate : ℝ := 0.10

def total_before_discount : ℝ :=
  hamburger_price + crackers_price + (vegetable_price * vegetable_bags) + cheese_price

def discount_amount : ℝ := total_before_discount * discount_rate

def total_after_discount : ℝ := total_before_discount - discount_amount

theorem rays_grocery_bill :
  total_after_discount = 18.00 := by
  sorry

end NUMINAMATH_CALUDE_rays_grocery_bill_l3838_383869


namespace NUMINAMATH_CALUDE_smallest_n_proof_l3838_383809

/-- The capacity of adults on a single bench section -/
def adult_capacity : ℕ := 8

/-- The capacity of children on a single bench section -/
def child_capacity : ℕ := 12

/-- Predicate to check if a number of bench sections can seat an equal number of adults and children -/
def can_seat_equally (n : ℕ) : Prop :=
  ∃ (x : ℕ), x > 0 ∧ adult_capacity * n = x ∧ child_capacity * n = x

/-- The smallest positive integer number of bench sections that can seat an equal number of adults and children -/
def smallest_n : ℕ := 3

theorem smallest_n_proof :
  (can_seat_equally smallest_n) ∧
  (∀ m : ℕ, m > 0 ∧ m < smallest_n → ¬(can_seat_equally m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_proof_l3838_383809


namespace NUMINAMATH_CALUDE_variance_2xi_plus_3_l3838_383802

variable (ξ : ℝ → ℝ)

-- D represents the variance operator
def D (X : ℝ → ℝ) : ℝ := sorry

-- Given condition
axiom variance_xi : D ξ = 2

-- Theorem to prove
theorem variance_2xi_plus_3 : D (fun ω => 2 * ξ ω + 3) = 8 := by sorry

end NUMINAMATH_CALUDE_variance_2xi_plus_3_l3838_383802


namespace NUMINAMATH_CALUDE_last_two_digits_of_sum_factorials_l3838_383883

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_of_sum_factorials :
  sum_factorials 9 % 100 = 13 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_of_sum_factorials_l3838_383883


namespace NUMINAMATH_CALUDE_distance_to_larger_cross_section_l3838_383898

/-- A right octagonal pyramid with two parallel cross sections -/
structure OctagonalPyramid where
  /-- Area of the smaller cross section in square feet -/
  area_small : ℝ
  /-- Area of the larger cross section in square feet -/
  area_large : ℝ
  /-- Distance between the two cross sections in feet -/
  distance_between : ℝ

/-- The distance from the apex to the larger cross section -/
def distance_to_larger (p : OctagonalPyramid) : ℝ := sorry

theorem distance_to_larger_cross_section
  (p : OctagonalPyramid)
  (h1 : p.area_small = 400 * Real.sqrt 2)
  (h2 : p.area_large = 900 * Real.sqrt 2)
  (h3 : p.distance_between = 10) :
  distance_to_larger p = 30 := by sorry

end NUMINAMATH_CALUDE_distance_to_larger_cross_section_l3838_383898


namespace NUMINAMATH_CALUDE_circle_points_count_l3838_383857

/-- A circle with n equally spaced points, labeled from 1 to n. -/
structure LabeledCircle where
  n : ℕ
  points : Fin n → ℕ
  labeled_from_1_to_n : ∀ i, points i = i.val + 1

/-- Two points are diametrically opposite if their distance is half the total number of points. -/
def diametrically_opposite (c : LabeledCircle) (i j : Fin c.n) : Prop :=
  (j.val - i.val) % c.n = c.n / 2

/-- The main theorem: if points 7 and 35 are diametrically opposite in a labeled circle, then n = 56. -/
theorem circle_points_count (c : LabeledCircle) 
  (h : ∃ (i j : Fin c.n), c.points i = 7 ∧ c.points j = 35 ∧ diametrically_opposite c i j) : 
  c.n = 56 := by
  sorry

end NUMINAMATH_CALUDE_circle_points_count_l3838_383857


namespace NUMINAMATH_CALUDE_geometric_series_r_value_l3838_383840

theorem geometric_series_r_value (b r : ℝ) (h1 : r ≠ 1) (h2 : r ≠ -1) : 
  (b / (1 - r) = 18) → 
  (b * r^2 / (1 - r^2) = 6) → 
  r = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_r_value_l3838_383840


namespace NUMINAMATH_CALUDE_sin_pi_over_two_minus_pi_over_six_l3838_383850

theorem sin_pi_over_two_minus_pi_over_six :
  Real.sin (π / 2 - π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_over_two_minus_pi_over_six_l3838_383850


namespace NUMINAMATH_CALUDE_grape_juice_percentage_in_mixture_l3838_383885

/-- Represents a mixture with a certain volume and grape juice percentage -/
structure Mixture where
  volume : ℝ
  percentage : ℝ

/-- Calculates the total volume of grape juice in a mixture -/
def grapeJuiceVolume (m : Mixture) : ℝ := m.volume * m.percentage

/-- The problem statement -/
theorem grape_juice_percentage_in_mixture : 
  let mixtureA : Mixture := { volume := 15, percentage := 0.3 }
  let mixtureB : Mixture := { volume := 40, percentage := 0.2 }
  let mixtureC : Mixture := { volume := 25, percentage := 0.1 }
  let pureGrapeJuice : ℝ := 10

  let totalGrapeJuice := grapeJuiceVolume mixtureA + grapeJuiceVolume mixtureB + 
                         grapeJuiceVolume mixtureC + pureGrapeJuice
  let totalVolume := mixtureA.volume + mixtureB.volume + mixtureC.volume + pureGrapeJuice

  let resultPercentage := totalGrapeJuice / totalVolume

  abs (resultPercentage - 0.2778) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_grape_juice_percentage_in_mixture_l3838_383885


namespace NUMINAMATH_CALUDE_equation_pattern_l3838_383882

theorem equation_pattern (n : ℕ) : 2*n * (2*n + 2) + 1 = (2*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_pattern_l3838_383882


namespace NUMINAMATH_CALUDE_total_frisbees_sold_l3838_383833

/-- Represents the number of frisbees sold at $3 each -/
def x : ℕ := sorry

/-- Represents the number of frisbees sold at $4 each -/
def y : ℕ := sorry

/-- The total receipts from frisbee sales -/
def total_receipts : ℕ := 200

/-- The minimum number of $4 frisbees sold -/
def min_four_dollar_frisbees : ℕ := 20

/-- Theorem stating the total number of frisbees sold -/
theorem total_frisbees_sold :
  (3 * x + 4 * y = total_receipts) →
  (y ≥ min_four_dollar_frisbees) →
  (x + y = 60) := by
  sorry

end NUMINAMATH_CALUDE_total_frisbees_sold_l3838_383833


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3838_383803

open Real

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- Triangle inequality
  cos B = 2/5 →
  sin A * cos B - (2*c - cos A) * sin B = 0 →
  b = 1/2 ∧
  ∀ a' c', 0 < a' ∧ 0 < c' →
    a' + b + c' ≤ Real.sqrt 30 / 6 + 1/2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3838_383803
