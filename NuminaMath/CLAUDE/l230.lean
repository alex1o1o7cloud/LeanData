import Mathlib

namespace gold_coin_percentage_l230_23021

theorem gold_coin_percentage 
  (total_objects : ℝ) 
  (beads_and_rings_percent : ℝ) 
  (beads_ratio : ℝ) 
  (silver_coins_percent : ℝ) 
  (h1 : beads_and_rings_percent = 30) 
  (h2 : beads_ratio = 1/2) 
  (h3 : silver_coins_percent = 35) : 
  let coins_percent := 100 - beads_and_rings_percent
  let gold_coins_percent := coins_percent * (100 - silver_coins_percent) / 100
  gold_coins_percent = 45.5 := by
sorry

end gold_coin_percentage_l230_23021


namespace carlos_has_largest_answer_l230_23066

def alice_calculation (x : ℕ) : ℕ := ((x - 3) * 3) + 5

def bob_calculation (x : ℕ) : ℕ := (x^2 - 4) + 5

def carlos_calculation (x : ℕ) : ℕ := (x - 2 + 3)^2

theorem carlos_has_largest_answer :
  let initial_number := 12
  carlos_calculation initial_number > alice_calculation initial_number ∧
  carlos_calculation initial_number > bob_calculation initial_number :=
by sorry

end carlos_has_largest_answer_l230_23066


namespace apple_slice_packing_l230_23011

/-- The number of apple slices per group that satisfies the packing conditions -/
def apple_slices_per_group : ℕ := sorry

/-- The number of grapes per group -/
def grapes_per_group : ℕ := 9

/-- The smallest total number of grapes -/
def smallest_total_grapes : ℕ := 18

theorem apple_slice_packing :
  (apple_slices_per_group > 0) ∧
  (apple_slices_per_group * (smallest_total_grapes / grapes_per_group) = smallest_total_grapes) ∧
  (apple_slices_per_group ∣ smallest_total_grapes) ∧
  (grapes_per_group ∣ apple_slices_per_group * grapes_per_group) →
  apple_slices_per_group = 9 := by sorry

end apple_slice_packing_l230_23011


namespace largest_non_representable_l230_23067

/-- Coin denominations in Limonia -/
def coin_denominations (n : ℕ) : List ℕ :=
  List.range (n + 1) |> List.map (λ i => 2^(n - i) * 3^i)

/-- A number is representable if it can be expressed as a sum of coin denominations -/
def is_representable (s : ℕ) (n : ℕ) : Prop :=
  ∃ (coeffs : List ℕ), s = List.sum (List.zipWith (·*·) coeffs (coin_denominations n))

/-- The largest non-representable amount in Limonia's currency system -/
theorem largest_non_representable (n : ℕ) :
  ¬ is_representable (3^(n+1) - 2^(n+2)) n ∧
  ∀ s, s > 3^(n+1) - 2^(n+2) → is_representable s n :=
sorry

end largest_non_representable_l230_23067


namespace mod_equivalence_problem_l230_23000

theorem mod_equivalence_problem : ∃! m : ℤ, 0 ≤ m ∧ m ≤ 8 ∧ m ≡ 500000 [ZMOD 9] ∧ m = 5 := by
  sorry

end mod_equivalence_problem_l230_23000


namespace max_intersection_points_quadrilateral_circle_l230_23057

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A quadrilateral in a plane -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- The number of intersection points between a line segment and a circle -/
def intersectionPointsLineSegmentCircle (segment : (ℝ × ℝ) × (ℝ × ℝ)) (circle : Circle) : ℕ :=
  sorry

/-- The number of intersection points between a quadrilateral and a circle -/
def intersectionPointsQuadrilateralCircle (quad : Quadrilateral) (circle : Circle) : ℕ :=
  sorry

/-- Theorem: The maximum number of intersection points between a quadrilateral and a circle is 8 -/
theorem max_intersection_points_quadrilateral_circle :
  ∀ (quad : Quadrilateral) (circle : Circle),
    intersectionPointsQuadrilateralCircle quad circle ≤ 8 ∧
    ∃ (quad' : Quadrilateral) (circle' : Circle),
      intersectionPointsQuadrilateralCircle quad' circle' = 8 :=
by
  sorry

end max_intersection_points_quadrilateral_circle_l230_23057


namespace probability_all_white_balls_l230_23096

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def drawn_balls : ℕ := 7

theorem probability_all_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 8 / 6435 :=
sorry

end probability_all_white_balls_l230_23096


namespace sum_integers_between_two_and_eleven_l230_23065

theorem sum_integers_between_two_and_eleven : 
  (Finset.range 8).sum (fun i => i + 3) = 52 := by
  sorry

end sum_integers_between_two_and_eleven_l230_23065


namespace complement_A_intersect_B_l230_23042

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x > 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

-- Statement to prove
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {x | 1 < x ∧ x ≤ 2} := by sorry

end complement_A_intersect_B_l230_23042


namespace society_member_property_l230_23075

theorem society_member_property (n : ℕ) (h : n = 1978) :
  ∀ (f : Fin n → Fin 6),
  ∃ (i j k : Fin n),
    f i = f j ∧ f i = f k ∧
    (i.val = j.val + k.val ∨ i.val = 2 * j.val) :=
by sorry

end society_member_property_l230_23075


namespace dan_seashells_given_l230_23053

/-- The number of seashells Dan gave to Jessica -/
def seashells_given (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

theorem dan_seashells_given :
  seashells_given 56 22 = 34 := by
  sorry

end dan_seashells_given_l230_23053


namespace parabola_point_distance_l230_23001

/-- Given a parabola y^2 = x with focus F(1/4, 0), prove that for any point A(x₀, y₀) on the parabola,
    if AF = |5/4 * x₀|, then x₀ = 1. -/
theorem parabola_point_distance (x₀ y₀ : ℝ) : 
  y₀^2 = x₀ →  -- Point A is on the parabola
  ((x₀ - 1/4)^2 + y₀^2)^(1/2) = |5/4 * x₀| →  -- AF = |5/4 * x₀|
  x₀ = 1 := by
sorry

end parabola_point_distance_l230_23001


namespace purple_nails_count_l230_23092

/-- The number of nails painted purple -/
def purple_nails : ℕ := sorry

/-- The number of nails painted blue -/
def blue_nails : ℕ := 8

/-- The number of nails painted striped -/
def striped_nails : ℕ := sorry

/-- The total number of nails -/
def total_nails : ℕ := 20

/-- The difference in percentage points between blue and striped nails -/
def percentage_difference : ℝ := 10

theorem purple_nails_count : purple_nails = 6 := by
  have h1 : purple_nails + blue_nails + striped_nails = total_nails := sorry
  have h2 : (blue_nails : ℝ) / total_nails * 100 - (striped_nails : ℝ) / total_nails * 100 = percentage_difference := sorry
  sorry

end purple_nails_count_l230_23092


namespace chosen_number_l230_23077

theorem chosen_number (x : ℝ) : (x / 8) - 160 = 12 → x = 1376 := by
  sorry

end chosen_number_l230_23077


namespace fraction_of_students_with_As_l230_23055

theorem fraction_of_students_with_As (fraction_B : ℝ) (fraction_A_or_B : ℝ) 
  (h1 : fraction_B = 0.2) 
  (h2 : fraction_A_or_B = 0.9) : 
  ∃ fraction_A : ℝ, fraction_A + fraction_B = fraction_A_or_B ∧ fraction_A = 0.7 := by
  sorry

end fraction_of_students_with_As_l230_23055


namespace tom_finishes_30_min_after_anna_l230_23099

/-- Represents the race scenario with given parameters -/
structure RaceScenario where
  distance : ℝ
  anna_speed : ℝ
  tom_speed : ℝ

/-- Calculates the finish time difference between Tom and Anna -/
def finishTimeDifference (race : RaceScenario) : ℝ :=
  race.distance * (race.tom_speed - race.anna_speed)

/-- Theorem stating that in the given race scenario, Tom finishes 30 minutes after Anna -/
theorem tom_finishes_30_min_after_anna :
  let race : RaceScenario := {
    distance := 15,
    anna_speed := 7,
    tom_speed := 9
  }
  finishTimeDifference race = 30 := by sorry

end tom_finishes_30_min_after_anna_l230_23099


namespace billy_crayons_l230_23004

/-- The number of crayons left after a monkey and hippopotamus eat some crayons -/
def crayons_left (total : ℕ) (monkey_ate : ℕ) : ℕ :=
  total - (monkey_ate + 2 * monkey_ate)

/-- Theorem stating that given 200 total crayons, if a monkey eats 64 crayons,
    then 8 crayons are left -/
theorem billy_crayons : crayons_left 200 64 = 8 := by
  sorry

end billy_crayons_l230_23004


namespace bus_interval_is_30_minutes_l230_23044

/-- Represents a bus station schedule -/
structure BusSchedule where
  operatingHoursPerDay : ℕ
  operatingDays : ℕ
  totalBuses : ℕ

/-- Calculates the time interval between bus departures in minutes -/
def calculateInterval (schedule : BusSchedule) : ℕ :=
  let minutesPerDay := schedule.operatingHoursPerDay * 60
  let busesPerDay := schedule.totalBuses / schedule.operatingDays
  minutesPerDay / busesPerDay

/-- Theorem: The time interval between bus departures is 30 minutes -/
theorem bus_interval_is_30_minutes (schedule : BusSchedule) 
    (h1 : schedule.operatingHoursPerDay = 12)
    (h2 : schedule.operatingDays = 5)
    (h3 : schedule.totalBuses = 120) :
  calculateInterval schedule = 30 := by
  sorry

#eval calculateInterval { operatingHoursPerDay := 12, operatingDays := 5, totalBuses := 120 }

end bus_interval_is_30_minutes_l230_23044


namespace triangle_formation_theorem_l230_23020

/-- Triangle inequality theorem checker -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Set of three line segments -/
structure TriangleSet where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: Only the set (5, 13, 12) can form a triangle -/
theorem triangle_formation_theorem :
  let set_a := TriangleSet.mk 3 10 5
  let set_b := TriangleSet.mk 4 8 4
  let set_c := TriangleSet.mk 5 13 12
  let set_d := TriangleSet.mk 2 7 4
  satisfies_triangle_inequality set_c.a set_c.b set_c.c ∧
  ¬satisfies_triangle_inequality set_a.a set_a.b set_a.c ∧
  ¬satisfies_triangle_inequality set_b.a set_b.b set_b.c ∧
  ¬satisfies_triangle_inequality set_d.a set_d.b set_d.c :=
by
  sorry

end triangle_formation_theorem_l230_23020


namespace polygon_sides_from_angle_sum_l230_23035

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) : 
  angle_sum = 720 → (n - 2) * 180 = angle_sum → n = 6 := by
  sorry

end polygon_sides_from_angle_sum_l230_23035


namespace diophantine_logarithm_equation_l230_23045

theorem diophantine_logarithm_equation : ∃ (X Y Z : ℕ+), 
  (Nat.gcd X.val (Nat.gcd Y.val Z.val) = 1) ∧ 
  (X.val : ℝ) * (Real.log 5 / Real.log 100) + (Y.val : ℝ) * (Real.log 4 / Real.log 100) = Z.val ∧
  X.val + Y.val + Z.val = 4 := by
  sorry

end diophantine_logarithm_equation_l230_23045


namespace tangent_lines_range_l230_23063

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- The function g(x) = 2x^3 - 6x^2 --/
def g (x : ℝ) : ℝ := 2*x^3 - 6*x^2

/-- The derivative of g(x) --/
def g' (x : ℝ) : ℝ := 6*x^2 - 12*x

/-- Theorem: If three distinct tangent lines to f(x) pass through A(2, m), then -6 < m < 2 --/
theorem tangent_lines_range (m : ℝ) : 
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (m - (f a)) = (f' a) * (2 - a) ∧
    (m - (f b)) = (f' b) * (2 - b) ∧
    (m - (f c)) = (f' c) * (2 - c)) →
  -6 < m ∧ m < 2 :=
by sorry

end tangent_lines_range_l230_23063


namespace chips_and_juice_weight_l230_23025

/-- Given the weight of chips and juice bottles, calculate the total weight of a specific quantity -/
theorem chips_and_juice_weight
  (chip_weight : ℝ) -- Weight of a bag of chips
  (juice_weight : ℝ) -- Weight of a bottle of juice
  (h1 : 2 * chip_weight = 800) -- Weight of 2 bags of chips is 800 g
  (h2 : chip_weight = juice_weight + 350) -- A bag of chips is 350 g heavier than a bottle of juice
  : 5 * chip_weight + 4 * juice_weight = 2200 := by
  sorry

end chips_and_juice_weight_l230_23025


namespace square_overlap_area_difference_l230_23052

theorem square_overlap_area_difference :
  ∀ (side_large side_small overlap_area : ℝ),
    side_large = 10 →
    side_small = 7 →
    overlap_area = 9 →
    side_large > 0 →
    side_small > 0 →
    overlap_area > 0 →
    (side_large^2 - overlap_area) - (side_small^2 - overlap_area) = 51 :=
by
  sorry

end square_overlap_area_difference_l230_23052


namespace largest_divisor_of_n_l230_23091

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 36 ∣ n^3) : 
  ∀ d : ℕ, d ∣ n → d ≤ 6 :=
by sorry

end largest_divisor_of_n_l230_23091


namespace zoo_trip_admission_cost_l230_23061

theorem zoo_trip_admission_cost 
  (total_budget : ℕ) 
  (bus_rental_cost : ℕ) 
  (num_students : ℕ) 
  (h1 : total_budget = 350) 
  (h2 : bus_rental_cost = 100) 
  (h3 : num_students = 25) :
  (total_budget - bus_rental_cost) / num_students = 10 :=
by sorry

end zoo_trip_admission_cost_l230_23061


namespace lcm_gcf_product_24_150_l230_23088

theorem lcm_gcf_product_24_150 : Nat.lcm 24 150 * Nat.gcd 24 150 = 3600 := by
  sorry

end lcm_gcf_product_24_150_l230_23088


namespace ball_rolling_cycloid_l230_23032

/-- Represents the path of a ball rolling down a smooth cycloidal trough -/
noncomputable def path (a g t : ℝ) : ℝ :=
  4 * a * (1 - Real.cos (t * Real.sqrt (g / (4 * a))))

/-- Time for the ball to roll from the start to the lowest point along the cycloid -/
noncomputable def time_cycloid (a g : ℝ) : ℝ :=
  Real.pi * Real.sqrt (a / g)

/-- Time for the ball to roll from the start to the lowest point along a straight line -/
noncomputable def time_straight (a g : ℝ) : ℝ :=
  Real.sqrt (a * (4 + Real.pi^2) / g)

theorem ball_rolling_cycloid (a g : ℝ) (ha : a > 0) (hg : g > 0) :
  (∀ t, path a g t = 4 * a * (1 - Real.cos (t * Real.sqrt (g / (4 * a))))) ∧
  time_cycloid a g = Real.pi * Real.sqrt (a / g) ∧
  time_straight a g = Real.sqrt (a * (4 + Real.pi^2) / g) ∧
  time_cycloid a g < time_straight a g :=
sorry

end ball_rolling_cycloid_l230_23032


namespace history_paper_pages_l230_23079

/-- The number of days Stacy has to complete her history paper -/
def days_to_complete : ℕ := 3

/-- The number of pages Stacy must write per day to finish on time -/
def pages_per_day : ℕ := 11

/-- The total number of pages in Stacy's history paper -/
def total_pages : ℕ := days_to_complete * pages_per_day

theorem history_paper_pages : total_pages = 33 := by
  sorry

end history_paper_pages_l230_23079


namespace gcf_of_120_180_300_l230_23016

theorem gcf_of_120_180_300 : Nat.gcd 120 (Nat.gcd 180 300) = 60 := by
  sorry

end gcf_of_120_180_300_l230_23016


namespace one_rupee_coins_count_l230_23010

/-- Represents the types of coins in the bag -/
inductive CoinType
  | OneRupee
  | FiftyPaise
  | TwentyFivePaise

/-- The value of a coin in paise -/
def coinValue (c : CoinType) : ℕ :=
  match c with
  | CoinType.OneRupee => 100
  | CoinType.FiftyPaise => 50
  | CoinType.TwentyFivePaise => 25

/-- The total value of all coins in the bag in paise -/
def totalValue : ℕ := 105 * 100

/-- The number of each type of coin in the bag -/
def numEachCoin : ℕ := 60

/-- The total number of coins in the bag -/
def totalCoins : ℕ := 3 * numEachCoin

theorem one_rupee_coins_count :
  ∃ (n : ℕ), n = numEachCoin ∧
    n * coinValue CoinType.OneRupee +
    n * coinValue CoinType.FiftyPaise +
    n * coinValue CoinType.TwentyFivePaise = totalValue ∧
    3 * n = totalCoins := by
  sorry

end one_rupee_coins_count_l230_23010


namespace field_trip_participation_l230_23082

/-- Given a class of students where:
    - 4/5 of students left on the first vehicle
    - Of those who stayed, 1/3 didn't want to go
    - When another vehicle was found, 1/2 of the remaining students who wanted to go were able to join
    Prove that the fraction of students who went on the field trip is 13/15 -/
theorem field_trip_participation (total_students : ℕ) (total_students_pos : total_students > 0) :
  let first_vehicle := (4 : ℚ) / 5 * total_students
  let stayed_behind := total_students - first_vehicle
  let not_wanting_to_go := (1 : ℚ) / 3 * stayed_behind
  let wanting_to_go := stayed_behind - not_wanting_to_go
  let additional_joiners := (1 : ℚ) / 2 * wanting_to_go
  first_vehicle + additional_joiners = (13 : ℚ) / 15 * total_students :=
by sorry

end field_trip_participation_l230_23082


namespace cone_height_ratio_l230_23038

/-- Proves the ratio of heights for a cone with reduced height --/
theorem cone_height_ratio (base_circumference : ℝ) (original_height : ℝ) (shorter_volume : ℝ) :
  base_circumference = 18 * Real.pi →
  original_height = 36 →
  shorter_volume = 270 * Real.pi →
  ∃ (shorter_height : ℝ),
    shorter_height / original_height = 5 / 18 ∧
    shorter_volume = (1 / 3) * Real.pi * (base_circumference / (2 * Real.pi))^2 * shorter_height :=
by sorry

end cone_height_ratio_l230_23038


namespace total_trees_equals_sum_our_park_total_is_55_l230_23034

/-- Represents the number of walnut trees in a park -/
structure WalnutPark where
  initial : Nat  -- Initial number of walnut trees
  planted : Nat  -- Number of walnut trees planted

/-- Calculates the total number of walnut trees after planting -/
def total_trees (park : WalnutPark) : Nat :=
  park.initial + park.planted

/-- Theorem: The total number of walnut trees after planting is the sum of initial and planted trees -/
theorem total_trees_equals_sum (park : WalnutPark) : 
  total_trees park = park.initial + park.planted := by
  sorry

/-- The specific park instance from the problem -/
def our_park : WalnutPark := { initial := 22, planted := 33 }

/-- Theorem: The total number of walnut trees in our park after planting is 55 -/
theorem our_park_total_is_55 : total_trees our_park = 55 := by
  sorry

end total_trees_equals_sum_our_park_total_is_55_l230_23034


namespace inequality_theorem_equality_condition_l230_23039

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c) ≥ 1 / (b + c) + 1 / (c + a) + 1 / (a + b) :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c) = 1 / (b + c) + 1 / (c + a) + 1 / (a + b)) ↔ 
  (a = b ∧ b = c) :=
by sorry

end inequality_theorem_equality_condition_l230_23039


namespace round_trip_average_speed_l230_23086

/-- Calculates the average speed for a round trip boat journey on a river -/
theorem round_trip_average_speed
  (upstream_speed : ℝ)
  (downstream_speed : ℝ)
  (river_current : ℝ)
  (h1 : upstream_speed = 4)
  (h2 : downstream_speed = 7)
  (h3 : river_current = 2)
  : (2 / ((1 / (upstream_speed - river_current)) + (1 / (downstream_speed + river_current)))) = 36 / 11 := by
  sorry

end round_trip_average_speed_l230_23086


namespace egg_price_decrease_impact_l230_23094

-- Define the types
structure CakeShop where
  eggDemand : ℝ
  productionScale : ℝ
  cakeOutput : ℝ
  marketSupply : ℝ

-- Define the egg price change
def eggPriceDecrease : ℝ := 0.05

-- Define the impact of egg price decrease on cake shops
def impactOnCakeShop (shop : CakeShop) (priceDecrease : ℝ) : CakeShop :=
  { eggDemand := shop.eggDemand * (1 + priceDecrease),
    productionScale := shop.productionScale * (1 + priceDecrease),
    cakeOutput := shop.cakeOutput * (1 + priceDecrease),
    marketSupply := shop.marketSupply * (1 + priceDecrease) }

-- Theorem statement
theorem egg_price_decrease_impact (shop : CakeShop) :
  let newShop := impactOnCakeShop shop eggPriceDecrease
  newShop.eggDemand > shop.eggDemand ∧
  newShop.productionScale > shop.productionScale ∧
  newShop.cakeOutput > shop.cakeOutput ∧
  newShop.marketSupply > shop.marketSupply :=
by sorry

end egg_price_decrease_impact_l230_23094


namespace sequence_difference_l230_23095

/-- The sequence a_n with sum S_n = n^2 + 2n for n ∈ ℕ* -/
def S (n : ℕ+) : ℕ := n.val^2 + 2*n.val

/-- The n-th term of the sequence -/
def a (n : ℕ+) : ℕ := 2*n.val + 1

theorem sequence_difference (n m : ℕ+) (h : m.val - n.val = 5) :
  a m - a n = 10 := by
  sorry

end sequence_difference_l230_23095


namespace siena_bookmarks_l230_23085

/-- The number of bookmarked pages Siena will have at the end of March -/
def bookmarks_end_of_march (daily_rate : ℕ) (current_bookmarks : ℕ) : ℕ :=
  current_bookmarks + daily_rate * 31

/-- Theorem stating that Siena will have 1330 bookmarked pages at the end of March -/
theorem siena_bookmarks :
  bookmarks_end_of_march 30 400 = 1330 := by
  sorry

end siena_bookmarks_l230_23085


namespace rectangle_perimeter_l230_23037

theorem rectangle_perimeter (L W : ℝ) (h : L * W - (L - 4) * (W - 4) = 168) :
  2 * (L + W) = 92 := by
  sorry

end rectangle_perimeter_l230_23037


namespace cone_height_ratio_l230_23060

theorem cone_height_ratio (base_circumference : Real) (original_height : Real) (new_volume : Real) :
  base_circumference = 24 * Real.pi →
  original_height = 36 →
  new_volume = 432 * Real.pi →
  ∃ (new_height : Real),
    (1 / 3) * Real.pi * ((base_circumference / (2 * Real.pi)) ^ 2) * new_height = new_volume ∧
    new_height / original_height = 1 / 4 :=
by sorry

end cone_height_ratio_l230_23060


namespace union_of_sets_l230_23013

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 4, 5}
  A ∪ B = {1, 2, 3, 4, 5} := by
sorry

end union_of_sets_l230_23013


namespace infinitely_many_non_representable_l230_23030

def sum_of_powers (x₁ x₂ x₃ x₄ x₅ : ℕ) : ℕ :=
  x₁^3 + x₂^5 + x₃^7 + x₄^9 + x₅^11

def representable (n : ℕ) : Prop :=
  ∃ x₁ x₂ x₃ x₄ x₅ : ℕ, sum_of_powers x₁ x₂ x₃ x₄ x₅ = n

theorem infinitely_many_non_representable :
  ∃ S : Set ℕ, (Set.Infinite S) ∧ (∀ n ∈ S, ¬representable n) := by
  sorry

end infinitely_many_non_representable_l230_23030


namespace equation_solution_l230_23019

theorem equation_solution : ∃ x : ℝ, (10 - x)^2 = (x - 2)^2 + 8 ∧ x = 5.5 := by sorry

end equation_solution_l230_23019


namespace first_tribe_term_longer_l230_23083

/-- Represents the calendar system of the first tribe -/
structure Tribe1Calendar where
  months_per_year : Nat := 12
  days_per_month : Nat := 30

/-- Represents the calendar system of the second tribe -/
structure Tribe2Calendar where
  moons_per_year : Nat := 13
  weeks_per_moon : Nat := 4
  days_per_week : Nat := 7

/-- Calculates the number of days for the first tribe's term -/
def tribe1_term_days (cal : Tribe1Calendar) : Nat :=
  7 * cal.months_per_year * cal.days_per_month +
  1 * cal.days_per_month +
  18

/-- Calculates the number of days for the second tribe's term -/
def tribe2_term_days (cal : Tribe2Calendar) : Nat :=
  6 * cal.moons_per_year * cal.weeks_per_moon * cal.days_per_week +
  12 * cal.weeks_per_moon * cal.days_per_week +
  1 * cal.days_per_week +
  3

/-- Theorem stating that the first tribe's term is longer -/
theorem first_tribe_term_longer (cal1 : Tribe1Calendar) (cal2 : Tribe2Calendar) :
  tribe1_term_days cal1 > tribe2_term_days cal2 := by
  sorry

end first_tribe_term_longer_l230_23083


namespace quadratic_equation_properties_l230_23033

/-- Given a quadratic equation x^2 - 3x + k - 2 = 0 with two real roots x1 and x2 -/
theorem quadratic_equation_properties (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1^2 - 3*x1 + k - 2 = 0)
  (h2 : x2^2 - 3*x2 + k - 2 = 0)
  (h3 : x1 ≠ x2) :
  (k ≤ 17/4) ∧ 
  (x1 + x2 - x1*x2 = 1 → k = 4) := by
  sorry


end quadratic_equation_properties_l230_23033


namespace line_intercepts_and_point_l230_23015

/-- Given a line 3x + 5y + c = 0 where the sum of x- and y-intercepts is 16,
    prove that c = -30 and the point (2, 24/5) lies on the line. -/
theorem line_intercepts_and_point (c : ℝ) : 
  (∃ (x y : ℝ), 3*x + 5*y + c = 0 ∧ x + y = 16) → 
  (c = -30 ∧ 3*2 + 5*(24/5) + c = 0) :=
by sorry

end line_intercepts_and_point_l230_23015


namespace odd_function_product_negative_l230_23051

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_product_negative
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_nonzero : ∀ x, f x ≠ 0) :
  ∀ x, f x * f (-x) < 0 :=
by sorry

end odd_function_product_negative_l230_23051


namespace fraction_calculation_l230_23043

theorem fraction_calculation : (500^2 : ℝ) / (152^2 - 148^2) = 208.333 := by sorry

end fraction_calculation_l230_23043


namespace dorchester_puppies_washed_l230_23002

/-- Calculates the number of puppies washed given the daily base pay, per-puppy pay, and total earnings. -/
def puppies_washed (base_pay per_puppy_pay total_earnings : ℚ) : ℚ :=
  (total_earnings - base_pay) / per_puppy_pay

/-- Proves that Dorchester washed 16 puppies given the specified conditions. -/
theorem dorchester_puppies_washed :
  puppies_washed 40 2.25 76 = 16 := by
  sorry

#eval puppies_washed 40 2.25 76

end dorchester_puppies_washed_l230_23002


namespace exists_valid_five_by_five_division_l230_23036

/-- Represents a square grid -/
structure SquareGrid :=
  (side : ℕ)

/-- Represents a division of a square grid -/
structure GridDivision :=
  (grid : SquareGrid)
  (num_parts : ℕ)
  (segment_length : ℕ)

/-- Checks if a division of a square grid is valid -/
def is_valid_division (d : GridDivision) : Prop :=
  d.grid.side * d.grid.side % d.num_parts = 0 ∧
  d.segment_length ≤ 16

/-- Theorem: There exists a valid division of a 5x5 square grid into 5 equal parts
    with total segment length not exceeding 16 units -/
theorem exists_valid_five_by_five_division :
  ∃ (d : GridDivision), d.grid.side = 5 ∧ d.num_parts = 5 ∧ is_valid_division d :=
sorry

end exists_valid_five_by_five_division_l230_23036


namespace block_has_twelve_floors_l230_23070

/-- Represents a block of flats with the given conditions -/
structure BlockOfFlats where
  half_floors : ℕ
  apartments_per_floor_1 : ℕ
  apartments_per_floor_2 : ℕ
  max_residents_per_apartment : ℕ
  max_total_residents : ℕ

/-- The number of floors in the block of flats -/
def total_floors (b : BlockOfFlats) : ℕ := 2 * b.half_floors

/-- The theorem stating that the block of flats has 12 floors -/
theorem block_has_twelve_floors (b : BlockOfFlats)
  (h1 : b.apartments_per_floor_1 = 6)
  (h2 : b.apartments_per_floor_2 = 5)
  (h3 : b.max_residents_per_apartment = 4)
  (h4 : b.max_total_residents = 264) :
  total_floors b = 12 := by
  sorry

#check block_has_twelve_floors

end block_has_twelve_floors_l230_23070


namespace continuous_bounded_function_theorem_l230_23097

theorem continuous_bounded_function_theorem (f : ℝ → ℝ) 
  (hcont : Continuous f) 
  (hbound : ∃ M, ∀ x, |f x| ≤ M) 
  (heq : ∀ x y, (f x)^2 - (f y)^2 = f (x + y) * f (x - y)) :
  ∃ a b : ℝ, ∀ x, f x = b * Real.sin (π * x / (2 * a)) := by
sorry

end continuous_bounded_function_theorem_l230_23097


namespace work_completion_original_men_l230_23054

theorem work_completion_original_men (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) : 
  initial_days = 55 → absent_men = 15 → final_days = 60 → 
  ∃ (original_men : ℕ), 
    original_men * initial_days = (original_men - absent_men) * final_days ∧
    original_men = 180 := by
  sorry

end work_completion_original_men_l230_23054


namespace min_moves_to_equalize_l230_23023

/-- Represents the state of coin stacks -/
structure CoinStacks :=
  (stack1 : ℕ)
  (stack2 : ℕ)
  (stack3 : ℕ)
  (stack4 : ℕ)

/-- Represents a move in the coin stacking game -/
def move (s : CoinStacks) : CoinStacks := sorry

/-- Checks if all stacks have equal coins -/
def is_equal (s : CoinStacks) : Prop := 
  s.stack1 = s.stack2 ∧ s.stack2 = s.stack3 ∧ s.stack3 = s.stack4

/-- The initial state of coin stacks -/
def initial_state : CoinStacks := ⟨9, 7, 5, 10⟩

/-- Applies n moves to a given state -/
def apply_moves (s : CoinStacks) (n : ℕ) : CoinStacks := sorry

/-- The main theorem stating the minimum number of moves required -/
theorem min_moves_to_equalize : 
  ∃ (n : ℕ), n = 11 ∧ is_equal (apply_moves initial_state n) ∧ 
  ∀ (m : ℕ), m < n → ¬is_equal (apply_moves initial_state m) :=
sorry

end min_moves_to_equalize_l230_23023


namespace books_added_to_bin_l230_23049

theorem books_added_to_bin (initial_books : ℕ) (books_sold : ℕ) (final_books : ℕ)
  (h1 : initial_books = 4)
  (h2 : books_sold = 3)
  (h3 : final_books = 11)
  (h4 : initial_books ≥ books_sold) :
  final_books - (initial_books - books_sold) = 10 := by
  sorry

end books_added_to_bin_l230_23049


namespace four_digit_greater_than_product_l230_23008

theorem four_digit_greater_than_product (a b c d : ℕ) : 
  a ≤ 9 → b ≤ 9 → c ≤ 9 → d ≤ 9 → 
  (1000 * a + 100 * b + 10 * c + d > (10 * a + b) * (10 * c + d)) ∨ 
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) := by
  sorry

end four_digit_greater_than_product_l230_23008


namespace sweater_shirt_price_difference_l230_23093

theorem sweater_shirt_price_difference : 
  let shirt_total : ℚ := 400
  let shirt_count : ℕ := 25
  let sweater_total : ℚ := 1500
  let sweater_count : ℕ := 75
  let shirt_avg : ℚ := shirt_total / shirt_count
  let sweater_avg : ℚ := sweater_total / sweater_count
  sweater_avg - shirt_avg = 4 := by
sorry

end sweater_shirt_price_difference_l230_23093


namespace big_bottle_volume_proof_l230_23022

/-- The volume of a big bottle of mango juice in ounces -/
def big_bottle_volume : ℝ := 30

/-- The cost of a big bottle in pesetas -/
def big_bottle_cost : ℝ := 2700

/-- The volume of a small bottle in ounces -/
def small_bottle_volume : ℝ := 6

/-- The cost of a small bottle in pesetas -/
def small_bottle_cost : ℝ := 600

/-- The amount saved by buying a big bottle instead of equivalent small bottles in pesetas -/
def savings : ℝ := 300

theorem big_bottle_volume_proof :
  big_bottle_volume = 30 :=
by sorry

end big_bottle_volume_proof_l230_23022


namespace least_three_digit_multiple_of_11_l230_23024

theorem least_three_digit_multiple_of_11 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 11 ∣ n → 110 ≤ n :=
by sorry

end least_three_digit_multiple_of_11_l230_23024


namespace first_group_size_l230_23006

/-- The number of days it takes the first group to complete the work -/
def first_group_days : ℕ := 30

/-- The number of days it takes 20 men to complete the work -/
def second_group_days : ℕ := 24

/-- The number of men in the second group -/
def second_group_men : ℕ := 20

/-- The number of men in the first group -/
def first_group_men : ℕ := (second_group_men * second_group_days) / first_group_days

theorem first_group_size :
  first_group_men = 16 :=
by sorry

end first_group_size_l230_23006


namespace x_times_one_minus_f_eq_one_l230_23018

noncomputable def x : ℝ := (3 + Real.sqrt 8) ^ 1000
noncomputable def n : ℤ := ⌊x⌋
noncomputable def f : ℝ := x - n

theorem x_times_one_minus_f_eq_one : x * (1 - f) = 1 := by sorry

end x_times_one_minus_f_eq_one_l230_23018


namespace halfway_fraction_l230_23031

theorem halfway_fraction : (3 / 4 + 5 / 6) / 2 = 19 / 24 := by
  sorry

end halfway_fraction_l230_23031


namespace units_digit_of_72_cubed_minus_24_cubed_l230_23014

theorem units_digit_of_72_cubed_minus_24_cubed : ∃ n : ℕ, 72^3 - 24^3 ≡ 4 [MOD 10] ∧ n * 10 + 4 = 72^3 - 24^3 := by
  sorry

end units_digit_of_72_cubed_minus_24_cubed_l230_23014


namespace adams_collection_worth_80_dollars_l230_23062

/-- The value of Adam's coin collection -/
def adams_collection_value (total_coins : ℕ) (sample_coins : ℕ) (sample_value : ℕ) : ℕ :=
  total_coins * (sample_value / sample_coins)

/-- Theorem: Adam's coin collection is worth 80 dollars -/
theorem adams_collection_worth_80_dollars :
  adams_collection_value 20 5 20 = 80 :=
by sorry

end adams_collection_worth_80_dollars_l230_23062


namespace unique_number_equality_l230_23087

theorem unique_number_equality : ∃! x : ℝ, (x / 2) + 6 = 2 * x - 6 := by
  sorry

end unique_number_equality_l230_23087


namespace divisibility_problem_l230_23089

theorem divisibility_problem (n : ℕ) : 
  n > 2 → 
  (((1 + n + n * (n - 1) / 2 + n * (n - 1) * (n - 2) / 6) ∣ 2^2000) ↔ (n = 3 ∨ n = 7 ∨ n = 23)) :=
by sorry

end divisibility_problem_l230_23089


namespace figures_per_shelf_is_eleven_l230_23078

/-- The number of shelves in Adam's room -/
def num_shelves : ℕ := 4

/-- The total number of action figures that can fit on all shelves -/
def total_figures : ℕ := 44

/-- The number of action figures that can fit on each shelf -/
def figures_per_shelf : ℕ := total_figures / num_shelves

/-- Theorem: The number of action figures that can fit on each shelf is 11 -/
theorem figures_per_shelf_is_eleven : figures_per_shelf = 11 := by
  sorry

end figures_per_shelf_is_eleven_l230_23078


namespace green_shirt_pairs_l230_23027

theorem green_shirt_pairs (total_students : ℕ) (red_students : ℕ) (green_students : ℕ) 
  (total_pairs : ℕ) (red_red_pairs : ℕ) :
  total_students = 150 →
  red_students = 60 →
  green_students = 90 →
  total_pairs = 75 →
  red_red_pairs = 28 →
  total_students = red_students + green_students →
  ∃ (green_green_pairs : ℕ), green_green_pairs = 43 ∧ 
    green_green_pairs + red_red_pairs + (total_students - 2 * red_red_pairs - 2 * green_green_pairs) / 2 = total_pairs :=
by sorry

end green_shirt_pairs_l230_23027


namespace cars_with_no_features_l230_23041

theorem cars_with_no_features (total : ℕ) (airbags : ℕ) (power_windows : ℕ) (sunroofs : ℕ)
  (airbags_power : ℕ) (airbags_sunroofs : ℕ) (power_sunroofs : ℕ) (all_features : ℕ) :
  total = 80 →
  airbags = 45 →
  power_windows = 40 →
  sunroofs = 25 →
  airbags_power = 20 →
  airbags_sunroofs = 15 →
  power_sunroofs = 10 →
  all_features = 8 →
  total - (airbags + power_windows + sunroofs - airbags_power - airbags_sunroofs - power_sunroofs + all_features) = 7 :=
by sorry

end cars_with_no_features_l230_23041


namespace two_burritos_five_quesadillas_cost_l230_23081

/-- The price of a burrito in dollars -/
def burrito_price : ℝ := sorry

/-- The price of a quesadilla in dollars -/
def quesadilla_price : ℝ := sorry

/-- The condition that one burrito and four quesadillas cost $3.50 -/
axiom condition1 : burrito_price + 4 * quesadilla_price = 3.50

/-- The condition that four burritos and one quesadilla cost $4.10 -/
axiom condition2 : 4 * burrito_price + quesadilla_price = 4.10

/-- The theorem stating that two burritos and five quesadillas cost $5.02 -/
theorem two_burritos_five_quesadillas_cost :
  2 * burrito_price + 5 * quesadilla_price = 5.02 := by sorry

end two_burritos_five_quesadillas_cost_l230_23081


namespace iron_ball_surface_area_l230_23012

/-- The surface area of a spherical iron ball that displaces a specific volume of water -/
theorem iron_ball_surface_area (r : ℝ) (h : ℝ) (R : ℝ) : 
  r = 10 → h = 5/3 → (4/3) * Real.pi * R^3 = Real.pi * r^2 * h → 4 * Real.pi * R^2 = 100 * Real.pi :=
by sorry

end iron_ball_surface_area_l230_23012


namespace complex_power_magnitude_l230_23074

theorem complex_power_magnitude : 
  Complex.abs ((1/2 : ℂ) + (Complex.I * (Real.sqrt 3)/2))^12 = 1 := by sorry

end complex_power_magnitude_l230_23074


namespace ferry_journey_difference_l230_23028

/-- Represents a ferry with its speed and travel time -/
structure Ferry where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a ferry -/
def distance (f : Ferry) : ℝ := f.speed * f.time

theorem ferry_journey_difference :
  let ferry_p : Ferry := { speed := 8, time := 3 }
  let ferry_q : Ferry := { speed := ferry_p.speed + 1, time := (3 * distance ferry_p) / (ferry_p.speed + 1) }
  ferry_q.time - ferry_p.time = 5 := by
  sorry

end ferry_journey_difference_l230_23028


namespace ticket_price_difference_l230_23068

def prebought_count : ℕ := 20
def prebought_price : ℕ := 155
def gate_count : ℕ := 30
def gate_price : ℕ := 200

theorem ticket_price_difference : 
  gate_count * gate_price - prebought_count * prebought_price = 2900 := by
  sorry

end ticket_price_difference_l230_23068


namespace magic_king_episodes_l230_23029

/-- The total number of episodes in the Magic King show -/
def total_episodes : ℕ :=
  let first_three_seasons := 3 * 20
  let seasons_four_to_eight := 5 * 25
  let seasons_nine_to_eleven := 3 * 30
  let last_three_seasons := 3 * 15
  let holiday_specials := 5
  first_three_seasons + seasons_four_to_eight + seasons_nine_to_eleven + last_three_seasons + holiday_specials

/-- Theorem stating that the total number of episodes is 325 -/
theorem magic_king_episodes : total_episodes = 325 := by
  sorry

end magic_king_episodes_l230_23029


namespace equation_solutions_l230_23040

theorem equation_solutions :
  (∃ x : ℚ, 3 * x - 1 = x + 7 ∧ x = 4) ∧
  (∃ x : ℚ, (x + 1) / 2 - 1 = (1 - 2 * x) / 3 ∧ x = 5 / 7) :=
by sorry

end equation_solutions_l230_23040


namespace smallest_bounded_area_l230_23046

/-- The area of the smallest region bounded by y = x^2 and x^2 + y^2 = 9 -/
theorem smallest_bounded_area : 
  let f (x : ℝ) := x^2
  let g (x y : ℝ) := x^2 + y^2 = 9
  let intersection_x := Real.sqrt ((Real.sqrt 37 - 1) / 2)
  let bounded_area := (2/3) * (((-1 + Real.sqrt 37) / 2)^(3/2))
  ∃ (area : ℝ), area = bounded_area ∧ 
    (∀ x y, -intersection_x ≤ x ∧ x ≤ intersection_x ∧ 
            y = f x ∧ g x y → 
            area = ∫ x in -intersection_x..intersection_x, f x) :=
by sorry


end smallest_bounded_area_l230_23046


namespace definite_integral_2x_minus_1_l230_23047

theorem definite_integral_2x_minus_1 :
  ∫ x in (0:ℝ)..3, (2*x - 1) = 6 := by sorry

end definite_integral_2x_minus_1_l230_23047


namespace last_locker_opened_l230_23076

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
| Open
| Closed

/-- Represents the locker-opening process -/
def lockerProcess (n : Nat) : Nat → Nat :=
  sorry

/-- The number of lockers in the hall -/
def totalLockers : Nat := 512

/-- Theorem stating that the last locker opened is number 509 -/
theorem last_locker_opened :
  ∃ (process : Nat → Nat → LockerState),
    (∀ k, k ≤ totalLockers → process totalLockers k = LockerState.Open) ∧
    (∀ k, k < 509 → ∃ m, m < 509 ∧ process totalLockers m = LockerState.Open ∧ m > k) ∧
    process totalLockers 509 = LockerState.Open :=
  sorry

end last_locker_opened_l230_23076


namespace floor_painted_by_all_colors_l230_23084

/-- Represents the percentage of floor painted by each painter -/
structure PainterCoverage where
  red : Real
  green : Real
  blue : Real

/-- Theorem: Given the paint coverage, at least 10% of the floor is painted by all three colors -/
theorem floor_painted_by_all_colors (coverage : PainterCoverage) 
  (h_red : coverage.red = 75)
  (h_green : coverage.green = -70)
  (h_blue : coverage.blue = -65) :
  ∃ (all_colors_coverage : Real),
    all_colors_coverage ≥ 10 ∧ 
    all_colors_coverage ≤ 100 ∧
    all_colors_coverage ≤ coverage.red ∧
    all_colors_coverage ≤ -coverage.green ∧
    all_colors_coverage ≤ -coverage.blue :=
sorry

end floor_painted_by_all_colors_l230_23084


namespace product_of_divisors_equal_implies_equal_l230_23064

/-- Product of divisors function -/
def p (x : ℤ) : ℤ := sorry

/-- Theorem: If the product of divisors of two integers are equal, then the integers are equal -/
theorem product_of_divisors_equal_implies_equal (m n : ℤ) : p m = p n → m = n := by
  sorry

end product_of_divisors_equal_implies_equal_l230_23064


namespace geometric_sequence_minimum_value_l230_23073

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geo : GeometricSequence a)
  (h_relation : a 7 = a 6 + 2 * a 5)
  (h_product : ∃ (m n : ℕ), m ≠ n ∧ a m * a n = 16 * (a 1)^2) :
  (∃ (m n : ℕ), m ≠ n ∧ 1 / m + 4 / n = 3 / 2) ∧
  (∀ (m n : ℕ), m ≠ n → 1 / m + 4 / n ≥ 3 / 2) :=
sorry

end geometric_sequence_minimum_value_l230_23073


namespace credits_needed_is_84_l230_23072

/-- The number of credits needed to buy cards for a game -/
def credits_needed : ℕ :=
  let red_card_cost : ℕ := 3
  let blue_card_cost : ℕ := 5
  let total_cards_required : ℕ := 20
  let red_cards_owned : ℕ := 8
  let blue_cards_needed : ℕ := total_cards_required - red_cards_owned
  red_card_cost * red_cards_owned + blue_card_cost * blue_cards_needed

theorem credits_needed_is_84 : credits_needed = 84 := by
  sorry

end credits_needed_is_84_l230_23072


namespace dodecagon_diagonals_l230_23048

/-- The number of diagonals in a convex n-gon --/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides --/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by sorry

end dodecagon_diagonals_l230_23048


namespace fraction_irreducibility_l230_23090

/-- The fraction (3n^2 + 2n + 4) / (n + 1) is irreducible if and only if n is not congruent to 4 modulo 5 -/
theorem fraction_irreducibility (n : ℤ) : 
  (Int.gcd (3*n^2 + 2*n + 4) (n + 1) = 1) ↔ (n % 5 ≠ 4) := by
  sorry

end fraction_irreducibility_l230_23090


namespace mod_product_253_649_l230_23059

theorem mod_product_253_649 (n : ℕ) : 
  253 * 649 ≡ n [ZMOD 100] → 0 ≤ n → n < 100 → n = 97 := by
  sorry

end mod_product_253_649_l230_23059


namespace non_degenerate_ellipse_condition_l230_23005

/-- The equation of an ellipse in general form -/
def ellipse_equation (x y k : ℝ) : Prop :=
  2 * x^2 + 9 * y^2 - 12 * x - 27 * y = k

/-- Condition for the equation to represent a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  k > -135/4

/-- Theorem stating the condition for a non-degenerate ellipse -/
theorem non_degenerate_ellipse_condition :
  ∀ k, (∃ x y, ellipse_equation x y k) ∧ is_non_degenerate_ellipse k ↔
    (∀ x y, ellipse_equation x y k → is_non_degenerate_ellipse k) :=
by sorry

end non_degenerate_ellipse_condition_l230_23005


namespace negation_of_exp_gt_ln_proposition_l230_23050

open Real

theorem negation_of_exp_gt_ln_proposition :
  (¬ ∀ x : ℝ, x > 0 → exp x > log x) ↔ (∃ x : ℝ, x > 0 ∧ exp x ≤ log x) :=
by sorry

end negation_of_exp_gt_ln_proposition_l230_23050


namespace tournament_games_theorem_l230_23026

/-- Represents a single-elimination tournament. -/
structure Tournament :=
  (num_teams : ℕ)
  (no_ties : Bool)

/-- Calculates the number of games needed to determine a winner in a single-elimination tournament. -/
def games_to_win (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem stating that a single-elimination tournament with 23 teams and no ties requires 22 games to determine a winner. -/
theorem tournament_games_theorem (t : Tournament) (h1 : t.num_teams = 23) (h2 : t.no_ties = true) : 
  games_to_win t = 22 := by
  sorry

end tournament_games_theorem_l230_23026


namespace range_of_sine_function_l230_23017

open Real

theorem range_of_sine_function :
  let f : ℝ → ℝ := λ x ↦ sin (2 * x + π / 4)
  let domain : Set ℝ := { x | π / 4 ≤ x ∧ x ≤ π / 2 }
  ∀ y ∈ Set.range (f ∘ (λ x ↦ x : domain → ℝ)),
    -Real.sqrt 2 / 2 ≤ y ∧ y ≤ Real.sqrt 2 / 2 :=
by sorry

end range_of_sine_function_l230_23017


namespace jake_peaches_l230_23080

/-- 
Given that Steven has 13 peaches and Jake has six fewer peaches than Steven,
prove that Jake has 7 peaches.
-/
theorem jake_peaches (steven_peaches : ℕ) (jake_peaches : ℕ) 
  (h1 : steven_peaches = 13)
  (h2 : jake_peaches = steven_peaches - 6) :
  jake_peaches = 7 := by
  sorry

end jake_peaches_l230_23080


namespace dog_food_per_meal_l230_23069

/-- Calculates the amount of dog food each dog eats per meal given the total amount bought,
    amount left after a week, number of dogs, and number of meals per day. -/
theorem dog_food_per_meal
  (total_food : ℝ)
  (food_left : ℝ)
  (num_dogs : ℕ)
  (meals_per_day : ℕ)
  (days_per_week : ℕ)
  (h1 : total_food = 30)
  (h2 : food_left = 9)
  (h3 : num_dogs = 3)
  (h4 : meals_per_day = 2)
  (h5 : days_per_week = 7)
  : (total_food - food_left) / (num_dogs * meals_per_day * days_per_week) = 0.5 := by
  sorry

#check dog_food_per_meal

end dog_food_per_meal_l230_23069


namespace product_divisible_by_4_probability_l230_23009

def is_divisible_by_4 (n : ℕ) : Prop := ∃ k : ℕ, n = 4 * k

def count_pairs_divisible_by_4 (n : ℕ) : ℕ :=
  let even_count := n / 2
  let multiple_of_4_count := n / 4
  (even_count.choose 2) + multiple_of_4_count * (even_count - multiple_of_4_count)

theorem product_divisible_by_4_probability (n : ℕ) (hn : n = 20) :
  (count_pairs_divisible_by_4 n : ℚ) / (n.choose 2) = 7 / 19 :=
sorry

end product_divisible_by_4_probability_l230_23009


namespace strip_coloring_problem_l230_23058

/-- The number of valid colorings for a strip of length n -/
def validColorings : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validColorings (n + 1) + validColorings n

/-- The problem statement -/
theorem strip_coloring_problem :
  validColorings 9 = 89 := by sorry

end strip_coloring_problem_l230_23058


namespace min_throws_correct_l230_23003

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice thrown -/
def numDice : ℕ := 4

/-- The minimum possible sum when rolling the dice -/
def minSum : ℕ := numDice

/-- The maximum possible sum when rolling the dice -/
def maxSum : ℕ := numDice * numFaces

/-- The number of possible unique sums -/
def numUniqueSums : ℕ := maxSum - minSum + 1

/-- The minimum number of throws required to ensure the same sum is rolled twice -/
def minThrows : ℕ := numUniqueSums + 1

/-- Theorem stating that minThrows is the minimum number of throws required -/
theorem min_throws_correct :
  minThrows = 22 ∧
  ∀ n : ℕ, n < minThrows → ∃ outcome : Fin n → Fin (maxSum - minSum + 1),
    Function.Injective outcome :=
by sorry

end min_throws_correct_l230_23003


namespace quadruple_solution_l230_23098

theorem quadruple_solution (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (eq1 : a * b + c * d = 8)
  (eq2 : a * b * c * d = 8 + a + b + c + d) :
  a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 := by
sorry

end quadruple_solution_l230_23098


namespace distance_to_focus_l230_23007

/-- The distance from a point on a parabola to its focus -/
theorem distance_to_focus (x : ℝ) : 
  x^2 = 16 → -- Point A(x, 4) is on the parabola x^2 = 4y
  ∃ (f : ℝ × ℝ), -- There exists a focus f
    (∀ (p : ℝ × ℝ), p.2 = p.1^2 / 4 → dist p f = p.2 + 1) ∧ -- Definition of parabola
    dist (x, 4) f = 5 -- The distance from A to the focus is 5
:= by sorry

end distance_to_focus_l230_23007


namespace dictionary_cost_l230_23071

def total_cost : ℕ := 8 + 29
def dinosaur_book_cost : ℕ := 19
def cookbook_cost : ℕ := 7
def savings : ℕ := 8
def additional_needed : ℕ := 29

theorem dictionary_cost : 
  total_cost - (dinosaur_book_cost + cookbook_cost) = 11 := by
sorry

end dictionary_cost_l230_23071


namespace T_minus_n_is_even_l230_23056

/-- The number of non-empty subsets with integer average -/
def T (n : ℕ) : ℕ := sorry

/-- Theorem: T_n - n is even for all n > 1 -/
theorem T_minus_n_is_even (n : ℕ) (h : n > 1) : Even (T n - n) := by
  sorry

end T_minus_n_is_even_l230_23056
