import Mathlib

namespace NUMINAMATH_CALUDE_probability_king_of_diamonds_l3789_378941

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)

/-- The game setup with two standard decks -/
def game_setup (d : Deck) : Prop :=
  d.cards = 52 ∧ d.ranks = 13 ∧ d.suits = 4

/-- The total number of cards in the combined deck -/
def total_cards (d : Deck) : Nat :=
  2 * d.cards

/-- The number of Kings of Diamonds in the combined deck -/
def kings_of_diamonds : Nat := 2

/-- The probability of drawing a King of Diamonds from the top of the combined deck -/
theorem probability_king_of_diamonds (d : Deck) :
  game_setup d →
  (kings_of_diamonds : ℚ) / (total_cards d) = 1 / 52 :=
by sorry

end NUMINAMATH_CALUDE_probability_king_of_diamonds_l3789_378941


namespace NUMINAMATH_CALUDE_double_earnings_cars_needed_l3789_378993

/-- Represents the earnings and sales of a car salesman -/
structure CarSalesman where
  baseSalary : ℕ
  commissionPerCar : ℕ
  marchEarnings : ℕ

/-- Calculates the number of cars needed to be sold to reach a target earning -/
def carsNeededForTarget (s : CarSalesman) (targetEarnings : ℕ) : ℕ :=
  ((targetEarnings - s.baseSalary) / s.commissionPerCar : ℕ)

/-- Theorem: A car salesman needs to sell 15 cars in April to double his March earnings -/
theorem double_earnings_cars_needed (s : CarSalesman) 
    (h1 : s.baseSalary = 1000)
    (h2 : s.commissionPerCar = 200)
    (h3 : s.marchEarnings = 2000) : 
  carsNeededForTarget s (2 * s.marchEarnings) = 15 := by
  sorry

#eval carsNeededForTarget ⟨1000, 200, 2000⟩ 4000

end NUMINAMATH_CALUDE_double_earnings_cars_needed_l3789_378993


namespace NUMINAMATH_CALUDE_arc_length_sector_l3789_378972

/-- The arc length of a circular sector with central angle 90° and radius 6 is 3π. -/
theorem arc_length_sector (θ : ℝ) (r : ℝ) (h1 : θ = 90) (h2 : r = 6) :
  (θ / 360) * (2 * Real.pi * r) = 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_arc_length_sector_l3789_378972


namespace NUMINAMATH_CALUDE_shape_to_square_transformation_exists_l3789_378933

/-- A shape on a graph paper --/
structure GraphShape where
  -- Add necessary fields to represent the shape

/-- A triangle on a graph paper --/
structure Triangle where
  -- Add necessary fields to represent a triangle

/-- A square on a graph paper --/
structure Square where
  -- Add necessary fields to represent a square

/-- Function to divide a shape into triangles --/
def divideIntoTriangles (shape : GraphShape) : List Triangle :=
  sorry

/-- Function to check if a list of triangles can form a square --/
def canFormSquare (triangles : List Triangle) : Bool :=
  sorry

/-- Theorem stating that there exists a shape that can be divided into 5 triangles
    which can be reassembled to form a square --/
theorem shape_to_square_transformation_exists :
  ∃ (shape : GraphShape),
    let triangles := divideIntoTriangles shape
    triangles.length = 5 ∧ canFormSquare triangles :=
by
  sorry

end NUMINAMATH_CALUDE_shape_to_square_transformation_exists_l3789_378933


namespace NUMINAMATH_CALUDE_joan_seashells_left_l3789_378918

/-- The number of seashells Joan has left after giving some to Sam -/
def seashells_left (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

/-- Theorem stating that Joan has 27 seashells left -/
theorem joan_seashells_left : seashells_left 70 43 = 27 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_left_l3789_378918


namespace NUMINAMATH_CALUDE_pioneer_camp_group_l3789_378960

theorem pioneer_camp_group (x y z w : ℕ) : 
  x + y + z + w = 23 →
  10 * x + 11 * y + 12 * z + 13 * w = 253 →
  z = (3 : ℕ) / 2 * w →
  z = 6 := by
  sorry

end NUMINAMATH_CALUDE_pioneer_camp_group_l3789_378960


namespace NUMINAMATH_CALUDE_nell_initial_cards_l3789_378923

/-- The number of baseball cards Nell initially had -/
def initial_cards : ℕ := sorry

/-- The number of cards Nell has at the end -/
def final_cards : ℕ := 154

/-- The number of cards Nell gave to Jeff -/
def cards_given_to_jeff : ℕ := 301

/-- The number of new cards Nell bought -/
def new_cards_bought : ℕ := 60

/-- The number of cards Nell traded away to Sam -/
def cards_traded_away : ℕ := 45

/-- The number of cards Nell received from Sam -/
def cards_received : ℕ := 30

/-- Theorem stating that Nell's initial number of baseball cards was 410 -/
theorem nell_initial_cards :
  initial_cards = 410 :=
by sorry

end NUMINAMATH_CALUDE_nell_initial_cards_l3789_378923


namespace NUMINAMATH_CALUDE_increase_decrease_theorem_l3789_378945

theorem increase_decrease_theorem (k r s N : ℝ) 
  (hk : k > 0) (hr : r > 0) (hs : s > 0) (hN : N > 0) (hr_bound : r < 80) :
  N * (1 + k / 100) * (1 - r / 100) + 10 * s > N ↔ k > 100 * r / (100 - r) := by
sorry

end NUMINAMATH_CALUDE_increase_decrease_theorem_l3789_378945


namespace NUMINAMATH_CALUDE_common_tangent_sum_l3789_378927

/-- Parabola P₁ -/
def P₁ (x y : ℝ) : Prop := y = 2 * x^2 + 125 / 100

/-- Parabola P₂ -/
def P₂ (x y : ℝ) : Prop := x = 2 * y^2 + 65 / 4

/-- Common tangent line L -/
def L (x y a b c : ℝ) : Prop := a * x + b * y = c

/-- The slope of L is rational -/
def rational_slope (a b : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ a / b = p / q

theorem common_tangent_sum (a b c : ℕ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    P₁ x₁ y₁ ∧ P₂ x₂ y₂ ∧
    L x₁ y₁ a b c ∧ L x₂ y₂ a b c ∧
    rational_slope a b ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    Nat.gcd a (Nat.gcd b c) = 1) →
  a + b + c = 289 := by
  sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l3789_378927


namespace NUMINAMATH_CALUDE_triangle_problem_l3789_378959

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.a > t.c)
  (h2 : t.b * t.a * Real.cos t.B = 2)
  (h3 : Real.cos t.B = 1/3)
  (h4 : t.b = 3) :
  (t.a = 3 ∧ t.c = 2) ∧ 
  Real.cos (t.B - t.C) = 23/27 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3789_378959


namespace NUMINAMATH_CALUDE_calcium_phosphate_yield_l3789_378979

/-- Represents the coefficients of a chemical reaction --/
structure ReactionCoefficients where
  fe2o3 : ℚ
  caco3 : ℚ
  ca3po42 : ℚ

/-- Represents the available moles of reactants --/
structure AvailableMoles where
  fe2o3 : ℚ
  caco3 : ℚ

/-- Calculates the theoretical yield of Ca3(PO4)2 based on the balanced reaction and available moles --/
def theoreticalYield (coeff : ReactionCoefficients) (available : AvailableMoles) : ℚ :=
  min 
    (available.fe2o3 * coeff.ca3po42 / coeff.fe2o3)
    (available.caco3 * coeff.ca3po42 / coeff.caco3)

/-- Theorem stating the theoretical yield of Ca3(PO4)2 for the given reaction and available moles --/
theorem calcium_phosphate_yield : 
  let coeff : ReactionCoefficients := ⟨2, 6, 3⟩
  let available : AvailableMoles := ⟨4, 10⟩
  theoreticalYield coeff available = 5 := by
  sorry

end NUMINAMATH_CALUDE_calcium_phosphate_yield_l3789_378979


namespace NUMINAMATH_CALUDE_sin_90_degrees_l3789_378999

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l3789_378999


namespace NUMINAMATH_CALUDE_height_range_is_75cm_l3789_378980

/-- The range of a set of values is the difference between the maximum and minimum values. -/
def range (max min : ℝ) : ℝ := max - min

/-- The heights of five students at Gleeson Middle School. -/
structure StudentHeights where
  num_students : ℕ
  max_height : ℝ
  min_height : ℝ

/-- The range of heights of the students is 75 cm. -/
theorem height_range_is_75cm (heights : StudentHeights) 
  (h1 : heights.num_students = 5)
  (h2 : heights.max_height = 175)
  (h3 : heights.min_height = 100) : 
  range heights.max_height heights.min_height = 75 := by
sorry

end NUMINAMATH_CALUDE_height_range_is_75cm_l3789_378980


namespace NUMINAMATH_CALUDE_sqrt_3_simplest_l3789_378908

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → x = Real.sqrt y → ¬∃ (a b : ℝ), b > 1 ∧ y = a^2 * b

theorem sqrt_3_simplest :
  is_simplest_sqrt (Real.sqrt 3) ∧
  ¬is_simplest_sqrt (Real.sqrt 0.1) ∧
  ¬is_simplest_sqrt (Real.sqrt 8) ∧
  ¬is_simplest_sqrt (Real.sqrt (1/2)) :=
sorry

end NUMINAMATH_CALUDE_sqrt_3_simplest_l3789_378908


namespace NUMINAMATH_CALUDE_sin_monotone_increasing_l3789_378947

/-- The function f(x) = 3sin(π/6 - 2x) is monotonically increasing in the intervals [kπ + π/3, kπ + 5π/6] for all k ∈ ℤ -/
theorem sin_monotone_increasing (k : ℤ) :
  StrictMonoOn (fun x => 3 * Real.sin (π / 6 - 2 * x)) (Set.Icc (k * π + π / 3) (k * π + 5 * π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_sin_monotone_increasing_l3789_378947


namespace NUMINAMATH_CALUDE_ryan_has_twenty_more_l3789_378965

/-- The number of stickers each person has -/
structure StickerCount where
  karl : ℕ
  ryan : ℕ
  ben : ℕ

/-- The conditions of the sticker problem -/
def StickerProblem (s : StickerCount) : Prop :=
  s.karl = 25 ∧
  s.ryan > s.karl ∧
  s.ben = s.ryan - 10 ∧
  s.karl + s.ryan + s.ben = 105

/-- The theorem stating Ryan has 20 more stickers than Karl -/
theorem ryan_has_twenty_more (s : StickerCount) 
  (h : StickerProblem s) : s.ryan - s.karl = 20 := by
  sorry

#check ryan_has_twenty_more

end NUMINAMATH_CALUDE_ryan_has_twenty_more_l3789_378965


namespace NUMINAMATH_CALUDE_price_after_discounts_l3789_378936

def initial_price : Float := 9649.12
def discount1 : Float := 0.20
def discount2 : Float := 0.10
def discount3 : Float := 0.05

def apply_discount (price : Float) (discount : Float) : Float :=
  price * (1 - discount)

def final_price : Float :=
  apply_discount (apply_discount (apply_discount initial_price discount1) discount2) discount3

theorem price_after_discounts :
  final_price = 6600.09808 := by sorry

end NUMINAMATH_CALUDE_price_after_discounts_l3789_378936


namespace NUMINAMATH_CALUDE_unique_solution_for_odd_prime_l3789_378996

theorem unique_solution_for_odd_prime (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (m n : ℕ), 
    (2 : ℚ) / p = 1 / m + 1 / n ∧
    m > n ∧
    m = p * (p + 1) / 2 ∧
    n = 2 / (p + 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_odd_prime_l3789_378996


namespace NUMINAMATH_CALUDE_dye_mixture_amount_l3789_378983

/-- The total amount of mixture obtained by combining a fraction of water and a fraction of vinegar -/
def mixture_amount (water_total : ℚ) (vinegar_total : ℚ) (water_fraction : ℚ) (vinegar_fraction : ℚ) : ℚ :=
  water_fraction * water_total + vinegar_fraction * vinegar_total

/-- Theorem stating that the mixture amount for the given problem is 27 liters -/
theorem dye_mixture_amount :
  mixture_amount 20 18 (3/5) (5/6) = 27 := by
  sorry

end NUMINAMATH_CALUDE_dye_mixture_amount_l3789_378983


namespace NUMINAMATH_CALUDE_right_triangles_on_circle_l3789_378926

theorem right_triangles_on_circle (n : ℕ) (h : n = 100) :
  ¬ (∃ (t : ℕ), t = 1000 ∧ t = (n / 2) * (n - 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangles_on_circle_l3789_378926


namespace NUMINAMATH_CALUDE_apple_cost_theorem_l3789_378989

theorem apple_cost_theorem (cost_two_dozen : ℝ) (h : cost_two_dozen = 15.60) :
  let cost_per_dozen : ℝ := cost_two_dozen / 2
  let cost_four_dozen : ℝ := 4 * cost_per_dozen
  cost_four_dozen = 31.20 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_theorem_l3789_378989


namespace NUMINAMATH_CALUDE_difference_of_squares_l3789_378954

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3789_378954


namespace NUMINAMATH_CALUDE_boys_age_l3789_378906

theorem boys_age (x : ℕ) : x + 4 = 2 * (x - 6) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_boys_age_l3789_378906


namespace NUMINAMATH_CALUDE_f_explicit_formula_l3789_378938

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_explicit_formula 
  (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_period : has_period f 2)
  (h_known : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| :=
sorry

end NUMINAMATH_CALUDE_f_explicit_formula_l3789_378938


namespace NUMINAMATH_CALUDE_fraction_comparison_l3789_378948

theorem fraction_comparison : 
  (10^1966 + 1) / (10^1967 + 1) > (10^1967 + 1) / (10^1968 + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3789_378948


namespace NUMINAMATH_CALUDE_michelle_racks_l3789_378930

/-- The number of drying racks Michelle owns -/
def current_racks : ℕ := 3

/-- The number of pounds of pasta per drying rack -/
def pasta_per_rack : ℕ := 3

/-- The number of cups of flour needed to make one pound of pasta -/
def flour_per_pound : ℕ := 2

/-- The number of cups in each bag of flour -/
def cups_per_bag : ℕ := 8

/-- The number of bags of flour Michelle has -/
def num_bags : ℕ := 3

/-- The total number of cups of flour Michelle has -/
def total_flour : ℕ := num_bags * cups_per_bag

/-- The total pounds of pasta Michelle can make -/
def total_pasta : ℕ := total_flour / flour_per_pound

/-- The number of racks needed for all the pasta Michelle can make -/
def racks_needed : ℕ := total_pasta / pasta_per_rack

theorem michelle_racks :
  current_racks = racks_needed - 1 :=
sorry

end NUMINAMATH_CALUDE_michelle_racks_l3789_378930


namespace NUMINAMATH_CALUDE_min_value_of_f_l3789_378962

/-- The function f(x) with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := (x^2 - 1) * (x^2 + a*x + b)

/-- The theorem stating the minimum value of f(x) -/
theorem min_value_of_f (a b : ℝ) :
  (∀ x : ℝ, f a b x = f a b (4 - x)) →
  (∃ x₀ : ℝ, ∀ x : ℝ, f a b x₀ ≤ f a b x) ∧
  (∃ x₁ : ℝ, f a b x₁ = -16) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3789_378962


namespace NUMINAMATH_CALUDE_point_line_plane_relation_l3789_378957

-- Define the types for point, line, and plane
variable (Point Line Plane : Type)

-- Define the relations
variable (lies_on : Point → Line → Prop)
variable (is_in : Line → Plane → Prop)

-- Define the set membership and subset relations
variable (mem : Point → Line → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem point_line_plane_relation 
  (P : Point) (m : Line) (α : Plane) 
  (h1 : lies_on P m) 
  (h2 : is_in m α) : 
  mem P m ∧ subset m α :=
sorry

end NUMINAMATH_CALUDE_point_line_plane_relation_l3789_378957


namespace NUMINAMATH_CALUDE_jean_domino_friends_l3789_378901

theorem jean_domino_friends :
  ∀ (total_dominoes : ℕ) (dominoes_per_player : ℕ) (total_players : ℕ),
    total_dominoes = 28 →
    dominoes_per_player = 7 →
    total_players * dominoes_per_player = total_dominoes →
    total_players - 1 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_jean_domino_friends_l3789_378901


namespace NUMINAMATH_CALUDE_flight_departure_time_l3789_378970

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Adds a duration in minutes to a Time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  ⟨totalMinutes / 60, totalMinutes % 60, sorry⟩

theorem flight_departure_time :
  let checkInTime : ℕ := 120 -- 2 hours in minutes
  let drivingTime : ℕ := 45
  let parkingTime : ℕ := 15
  let latestDepartureTime : Time := ⟨17, 0, sorry⟩ -- 5:00 pm
  let flightDepartureTime : Time := addMinutes latestDepartureTime (checkInTime + drivingTime + parkingTime)
  flightDepartureTime = ⟨20, 0, sorry⟩ -- 8:00 pm
:= by sorry

end NUMINAMATH_CALUDE_flight_departure_time_l3789_378970


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_two_l3789_378928

theorem gcd_of_powers_of_two : Nat.gcd (2^1005 - 1) (2^1016 - 1) = 2^11 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_two_l3789_378928


namespace NUMINAMATH_CALUDE_same_number_on_four_dice_l3789_378995

theorem same_number_on_four_dice : 
  let n : ℕ := 6  -- number of sides on each die
  let k : ℕ := 4  -- number of dice
  (1 : ℚ) / n^(k-1) = (1 : ℚ) / 216 :=
by sorry

end NUMINAMATH_CALUDE_same_number_on_four_dice_l3789_378995


namespace NUMINAMATH_CALUDE_trivia_team_members_l3789_378904

theorem trivia_team_members (absent_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) : 
  absent_members = 6 → points_per_member = 3 → total_points = 27 → 
  ∃ (total_members : ℕ), total_members = 15 ∧ 
  points_per_member * (total_members - absent_members) = total_points :=
by
  sorry

end NUMINAMATH_CALUDE_trivia_team_members_l3789_378904


namespace NUMINAMATH_CALUDE_part_one_part_two_l3789_378971

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

-- Part 1
theorem part_one :
  (Set.compl (B (1/2))) ∩ (A (1/2)) = {x : ℝ | 9/4 ≤ x ∧ x < 5/2} := by sorry

-- Part 2
theorem part_two :
  ∀ a : ℝ, (A a ⊆ B a) ↔ (a ≥ -1/2 ∧ a ≤ (3 - Real.sqrt 5) / 2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3789_378971


namespace NUMINAMATH_CALUDE_equation_solution_l3789_378973

theorem equation_solution : ∃ x : ℝ, 3 * x + 6 = |(-23 + 9)| ∧ x = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3789_378973


namespace NUMINAMATH_CALUDE_line_parameterization_l3789_378939

/-- Given a line y = (3/4)x - 15 parameterized by (x,y) = (f(t), 20t - 10),
    prove that f(t) = (80/3)t + (20/3) -/
theorem line_parameterization (f : ℝ → ℝ) :
  (∀ x y, y = (3/4) * x - 15 ↔ ∃ t, x = f t ∧ y = 20 * t - 10) →
  f = λ t => (80/3) * t + 20/3 := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l3789_378939


namespace NUMINAMATH_CALUDE_revenue_maximized_at_13_l3789_378922

/-- Revenue function for book sales -/
def R (p : ℝ) : ℝ := p * (130 - 5 * p)

/-- Theorem stating that the revenue is maximized at p = 13 -/
theorem revenue_maximized_at_13 :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 26 ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 26 → R p ≥ R q ∧
  p = 13 :=
sorry

end NUMINAMATH_CALUDE_revenue_maximized_at_13_l3789_378922


namespace NUMINAMATH_CALUDE_line_up_arrangement_count_l3789_378929

/-- The number of different arrangements of 5 students (2 boys and 3 girls) where only two girls are adjacent. -/
def arrangement_count : ℕ := 24

/-- The total number of students in the line-up. -/
def total_students : ℕ := 5

/-- The number of boys in the line-up. -/
def num_boys : ℕ := 2

/-- The number of girls in the line-up. -/
def num_girls : ℕ := 3

/-- The number of adjacent girls in each arrangement. -/
def adjacent_girls : ℕ := 2

theorem line_up_arrangement_count :
  arrangement_count = 24 ∧
  total_students = num_boys + num_girls ∧
  num_boys = 2 ∧
  num_girls = 3 ∧
  adjacent_girls = 2 :=
sorry

end NUMINAMATH_CALUDE_line_up_arrangement_count_l3789_378929


namespace NUMINAMATH_CALUDE_salary_solution_l3789_378964

def salary_problem (J F M A May : ℕ) : Prop :=
  (J + F + M + A) / 4 = 8000 ∧
  (F + M + A + May) / 4 = 8700 ∧
  J = 3700 ∧
  May = 6500

theorem salary_solution :
  ∀ J F M A May : ℕ,
    salary_problem J F M A May →
    May = 6500 := by
  sorry

end NUMINAMATH_CALUDE_salary_solution_l3789_378964


namespace NUMINAMATH_CALUDE_john_boxes_l3789_378969

/-- The number of boxes each person has -/
structure Boxes where
  stan : ℕ
  joseph : ℕ
  jules : ℕ
  john : ℕ

/-- The conditions of the problem -/
def problem_conditions (b : Boxes) : Prop :=
  b.stan = 100 ∧
  b.joseph = b.stan - (80 * b.stan / 100) ∧
  b.jules = b.joseph + 5 ∧
  b.john > b.jules

/-- The theorem to prove -/
theorem john_boxes (b : Boxes) (h : problem_conditions b) : b.john = 30 := by
  sorry

end NUMINAMATH_CALUDE_john_boxes_l3789_378969


namespace NUMINAMATH_CALUDE_alan_told_seven_jokes_l3789_378951

/-- The number of jokes Jessy told on Saturday -/
def jessy_jokes : ℕ := 11

/-- The number of jokes Alan told on Saturday -/
def alan_jokes : ℕ := sorry

/-- The total number of jokes both told over two Saturdays -/
def total_jokes : ℕ := 54

/-- Theorem stating that Alan told 7 jokes on Saturday -/
theorem alan_told_seven_jokes :
  alan_jokes = 7 ∧
  jessy_jokes + alan_jokes + 2 * jessy_jokes + 2 * alan_jokes = total_jokes :=
sorry

end NUMINAMATH_CALUDE_alan_told_seven_jokes_l3789_378951


namespace NUMINAMATH_CALUDE_unique_prime_pair_l3789_378943

theorem unique_prime_pair : ∃! (p q : ℕ), 
  Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 3 ∧ q ≠ 3 ∧
  (∀ α : ℤ, (α ^ (3 * p * q) - α) % (3 * p * q) = 0) ∧
  p = 11 ∧ q = 17 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_pair_l3789_378943


namespace NUMINAMATH_CALUDE_second_term_of_specific_sequence_l3789_378914

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem second_term_of_specific_sequence :
  ∀ (d : ℝ),
  arithmetic_sequence 2020 d 1 = 2020 ∧
  arithmetic_sequence 2020 d 5 = 4040 →
  arithmetic_sequence 2020 d 2 = 2525 :=
by
  sorry

end NUMINAMATH_CALUDE_second_term_of_specific_sequence_l3789_378914


namespace NUMINAMATH_CALUDE_book_reading_increase_l3789_378982

theorem book_reading_increase (matt_last_year matt_this_year pete_last_year pete_this_year : ℕ) 
  (h1 : pete_last_year = 2 * matt_last_year)
  (h2 : pete_this_year = 2 * pete_last_year)
  (h3 : pete_last_year + pete_this_year = 300)
  (h4 : matt_this_year = 75) :
  (matt_this_year - matt_last_year) * 100 / matt_last_year = 50 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_increase_l3789_378982


namespace NUMINAMATH_CALUDE_sun_city_has_12000_people_l3789_378991

/-- The population of Willowdale City -/
def willowdale_population : ℕ := 2000

/-- The population of Roseville City -/
def roseville_population : ℕ := 3 * willowdale_population - 500

/-- The population of Sun City -/
def sun_city_population : ℕ := 2 * roseville_population + 1000

/-- Theorem stating that Sun City has 12000 people -/
theorem sun_city_has_12000_people : sun_city_population = 12000 := by
  sorry

end NUMINAMATH_CALUDE_sun_city_has_12000_people_l3789_378991


namespace NUMINAMATH_CALUDE_sixteen_bananas_equal_nineteen_grapes_l3789_378917

/-- The cost relationship between bananas, oranges, and grapes -/
structure FruitCosts where
  banana_orange_ratio : ℚ  -- 4 bananas = 3 oranges
  orange_grape_ratio : ℚ   -- 5 oranges = 8 grapes

/-- Calculate the number of grapes equivalent in cost to a given number of bananas -/
def grapes_for_bananas (costs : FruitCosts) (num_bananas : ℕ) : ℕ :=
  let oranges : ℚ := (num_bananas : ℚ) * costs.banana_orange_ratio
  let grapes : ℚ := oranges * costs.orange_grape_ratio
  grapes.ceil.toNat

/-- Theorem stating that 16 bananas cost as much as 19 grapes -/
theorem sixteen_bananas_equal_nineteen_grapes (costs : FruitCosts) 
    (h1 : costs.banana_orange_ratio = 3/4)
    (h2 : costs.orange_grape_ratio = 8/5) : 
  grapes_for_bananas costs 16 = 19 := by
  sorry

#eval grapes_for_bananas ⟨3/4, 8/5⟩ 16

end NUMINAMATH_CALUDE_sixteen_bananas_equal_nineteen_grapes_l3789_378917


namespace NUMINAMATH_CALUDE_k_value_l3789_378992

def length (k : ℕ) : ℕ :=
  (Nat.factors k).length

theorem k_value (k : ℕ) (h1 : k > 1) (h2 : length k = 4) (h3 : k = 2 * 2 * 2 * 3) :
  k = 24 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l3789_378992


namespace NUMINAMATH_CALUDE_flu_spread_l3789_378968

theorem flu_spread (initial_infected : ℕ) (total_infected : ℕ) (x : ℝ) : 
  initial_infected = 1 →
  total_infected = 81 →
  (1 + x)^2 = total_infected →
  x ≥ 0 →
  ∃ (y : ℝ), y = x ∧ (initial_infected : ℝ) + y + y^2 = total_infected :=
sorry

end NUMINAMATH_CALUDE_flu_spread_l3789_378968


namespace NUMINAMATH_CALUDE_probability_approx_0647_l3789_378942

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of the white rectangle in the center -/
structure CenterRectangle where
  length : ℝ
  width : ℝ

/-- Represents the circular sheet used to cover the field -/
structure CircularSheet where
  diameter : ℝ

/-- Calculate the probability of the circular sheet covering part of the white area -/
def probability_of_covering_white_area (field : FieldDimensions) (center_rect : CenterRectangle) (sheet : CircularSheet) (num_circles : ℕ) (circle_radius : ℝ) : ℝ :=
  sorry

/-- The main theorem stating the probability is approximately 0.647 -/
theorem probability_approx_0647 :
  let field := FieldDimensions.mk 12 10
  let center_rect := CenterRectangle.mk 4 2
  let sheet := CircularSheet.mk 1.5
  let num_circles := 5
  let circle_radius := 1
  abs (probability_of_covering_white_area field center_rect sheet num_circles circle_radius - 0.647) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_probability_approx_0647_l3789_378942


namespace NUMINAMATH_CALUDE_total_match_sticks_l3789_378955

/-- Given the number of boxes ordered by Farrah -/
def num_boxes : ℕ := 4

/-- The number of matchboxes in each box -/
def matchboxes_per_box : ℕ := 20

/-- The number of sticks in each matchbox -/
def sticks_per_matchbox : ℕ := 300

/-- Theorem stating the total number of match sticks ordered by Farrah -/
theorem total_match_sticks : 
  num_boxes * matchboxes_per_box * sticks_per_matchbox = 24000 := by
  sorry

end NUMINAMATH_CALUDE_total_match_sticks_l3789_378955


namespace NUMINAMATH_CALUDE_solve_candy_store_problem_l3789_378976

def candy_store_problem (initial_money : ℚ) (gum_packs : ℕ) (gum_price : ℚ) 
  (chocolate_bars : ℕ) (candy_canes : ℕ) (candy_cane_price : ℚ) (money_left : ℚ) : Prop :=
  ∃ (chocolate_bar_price : ℚ),
    initial_money = 
      gum_packs * gum_price + 
      chocolate_bars * chocolate_bar_price + 
      candy_canes * candy_cane_price + 
      money_left ∧
    chocolate_bar_price = 1

theorem solve_candy_store_problem :
  candy_store_problem 10 3 1 5 2 (1/2) 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_candy_store_problem_l3789_378976


namespace NUMINAMATH_CALUDE_p_true_and_q_false_l3789_378920

-- Define proposition p
def p : Prop := ∀ x : ℝ, x > 0 → Real.log (x + 1) > 0

-- Define proposition q
def q : Prop := ∀ a b : ℝ, a > b → a^2 > b^2

-- Theorem to prove
theorem p_true_and_q_false : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_p_true_and_q_false_l3789_378920


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3789_378946

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem quadratic_function_properties (a b : ℝ) :
  (f a b (-a/2 + 1) ≤ f a b (a^2 + 5/4)) ∧
  (f a b 1 + f a b 3 - 2 * f a b 2 = 2) ∧
  (max (|f a b 1|) (max (|f a b 2|) (|f a b 3|)) ≥ 1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3789_378946


namespace NUMINAMATH_CALUDE_book_page_words_l3789_378984

theorem book_page_words (total_pages : ℕ) (max_words_per_page : ℕ) (total_words_mod : ℕ) :
  total_pages = 150 →
  max_words_per_page = 120 →
  total_words_mod = 221 →
  ∃ (words_per_page : ℕ),
    words_per_page ≤ max_words_per_page ∧
    Nat.Prime words_per_page ∧
    (total_pages * words_per_page) % total_words_mod = 220 ∧
    words_per_page = 67 :=
by sorry

end NUMINAMATH_CALUDE_book_page_words_l3789_378984


namespace NUMINAMATH_CALUDE_domino_division_exists_l3789_378974

/-- Represents a domino placement on a square grid -/
structure DominoTiling (n : ℕ) :=
  (dominoes : ℕ)
  (valid : dominoes * 2 = n * n)

/-- Represents a line dividing a square grid -/
inductive DividingLine
  | Horizontal (row : ℕ)
  | Vertical (col : ℕ)

/-- Checks if a dividing line results in two valid domino tilings -/
def validDivision (n : ℕ) (tiling : DominoTiling n) (line : DividingLine) : Prop :=
  match line with
  | DividingLine.Horizontal row =>
    ∃ (upper lower : ℕ), 
      upper + lower = tiling.dominoes ∧
      upper * 2 = row * n ∧
      lower * 2 = (n - row) * n
  | DividingLine.Vertical col =>
    ∃ (left right : ℕ),
      left + right = tiling.dominoes ∧
      left * 2 = col * n ∧
      right * 2 = (n - col) * n

/-- Main theorem: There exists a dividing line for a 6x6 square with 18 dominoes -/
theorem domino_division_exists :
  ∃ (line : DividingLine), validDivision 6 ⟨18, rfl⟩ line := by
  sorry

end NUMINAMATH_CALUDE_domino_division_exists_l3789_378974


namespace NUMINAMATH_CALUDE_S_minimum_at_n_min_l3789_378916

/-- The sequence a_n with general term 2n - 49 -/
def a (n : ℕ) : ℤ := 2 * n - 49

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) : ℤ := n * (a 1 + a n) / 2

/-- The value of n for which S_n reaches its minimum -/
def n_min : ℕ := 24

theorem S_minimum_at_n_min :
  ∀ k : ℕ, k ≠ 0 → S n_min ≤ S k :=
sorry

end NUMINAMATH_CALUDE_S_minimum_at_n_min_l3789_378916


namespace NUMINAMATH_CALUDE_digit_product_sum_l3789_378940

/-- A function that checks if a number is a three-digit number with all digits the same -/
def isTripleDigit (n : Nat) : Prop :=
  ∃ d, d ∈ Finset.range 10 ∧ n = d * 100 + d * 10 + d

/-- A function that converts a two-digit number to its decimal representation -/
def twoDigitToDecimal (a b : Nat) : Nat := 10 * a + b

theorem digit_product_sum : 
  ∃ (A B C D E : Nat), 
    A ∈ Finset.range 10 ∧ 
    B ∈ Finset.range 10 ∧ 
    C ∈ Finset.range 10 ∧ 
    D ∈ Finset.range 10 ∧ 
    E ∈ Finset.range 10 ∧ 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (twoDigitToDecimal A B) * (twoDigitToDecimal C D) = E * 100 + E * 10 + E ∧
    A + B + C + D + E = 21 :=
sorry

end NUMINAMATH_CALUDE_digit_product_sum_l3789_378940


namespace NUMINAMATH_CALUDE_min_c_value_l3789_378967

theorem min_c_value (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧  -- consecutive integers
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧  -- consecutive integers
  ∃ m : ℕ, b + c + d = m^2 ∧  -- b + c + d is a perfect square
  ∃ n : ℕ, a + b + c + d + e = n^3 ∧  -- a + b + c + d + e is a perfect cube
  ∀ c' : ℕ, (∃ a' b' d' e' : ℕ, 
    a' < b' ∧ b' < c' ∧ c' < d' ∧ d' < e' ∧
    b' = a' + 1 ∧ c' = b' + 1 ∧ d' = c' + 1 ∧ e' = d' + 1 ∧
    ∃ m' : ℕ, b' + c' + d' = m'^2 ∧
    ∃ n' : ℕ, a' + b' + c' + d' + e' = n'^3) →
  c' ≥ c →
  c = 675 :=
sorry

end NUMINAMATH_CALUDE_min_c_value_l3789_378967


namespace NUMINAMATH_CALUDE_second_train_length_second_train_length_solution_l3789_378963

/-- Calculates the length of the second train given the speeds of both trains,
    the time they take to clear each other, and the length of the first train. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (clear_time : ℝ) 
  (length1 : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_ms := relative_speed * (1000 / 3600)
  let total_distance := relative_speed_ms * clear_time
  total_distance - length1

/-- The length of the second train is approximately 1984 meters. -/
theorem second_train_length_solution :
  ∃ ε > 0, abs (second_train_length 75 65 7.353697418492236 121 - 1984) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_second_train_length_second_train_length_solution_l3789_378963


namespace NUMINAMATH_CALUDE_prob_one_or_two_sunny_days_l3789_378998

-- Define the probability of rain
def rain_prob : ℚ := 3/5

-- Define the number of days
def num_days : ℕ := 5

-- Function to calculate the probability of exactly k sunny days
def prob_k_sunny_days (k : ℕ) : ℚ :=
  (num_days.choose k) * (1 - rain_prob)^k * rain_prob^(num_days - k)

-- Theorem statement
theorem prob_one_or_two_sunny_days :
  prob_k_sunny_days 1 + prob_k_sunny_days 2 = 378/625 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_or_two_sunny_days_l3789_378998


namespace NUMINAMATH_CALUDE_a_neg_two_sufficient_not_necessary_l3789_378977

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

def z (a : ℝ) : ℂ := Complex.mk (a^2 - 4) (a + 1)

theorem a_neg_two_sufficient_not_necessary :
  (∃ (a : ℝ), a ≠ -2 ∧ is_pure_imaginary (z a)) ∧
  (∀ (a : ℝ), a = -2 → is_pure_imaginary (z a)) :=
sorry

end NUMINAMATH_CALUDE_a_neg_two_sufficient_not_necessary_l3789_378977


namespace NUMINAMATH_CALUDE_translate_down_two_units_l3789_378910

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically by a given amount -/
def translateVertically (l : Line) (amount : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - amount }

theorem translate_down_two_units :
  let original := Line.mk (-2) 0
  let translated := translateVertically original 2
  translated = Line.mk (-2) (-2) := by sorry

end NUMINAMATH_CALUDE_translate_down_two_units_l3789_378910


namespace NUMINAMATH_CALUDE_expression_simplification_l3789_378956

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (x - 1 - (2 * x - 2) / (x + 1)) / ((x^2 - x) / (2 * x + 2)) = 2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3789_378956


namespace NUMINAMATH_CALUDE_non_intersecting_chords_eq_catalan_number_l3789_378953

/-- The number of ways to draw n non-intersecting chords joining 2n points on a circle's circumference -/
def numberOfNonIntersectingChords (n : ℕ) : ℕ :=
  Nat.choose (2 * n) n / (n + 1)

/-- The nth Catalan number -/
def catalanNumber (n : ℕ) : ℕ :=
  Nat.choose (2 * n) n / (n + 1)

theorem non_intersecting_chords_eq_catalan_number :
  numberOfNonIntersectingChords 6 = catalanNumber 6 := by
  sorry

end NUMINAMATH_CALUDE_non_intersecting_chords_eq_catalan_number_l3789_378953


namespace NUMINAMATH_CALUDE_workshop_production_balance_l3789_378921

/-- Represents the production balance in a workshop --/
theorem workshop_production_balance 
  (total_workers : ℕ) 
  (bolts_per_worker : ℕ) 
  (nuts_per_worker : ℕ) 
  (nuts_per_bolt : ℕ) 
  (x : ℕ) : 
  total_workers = 16 → 
  bolts_per_worker = 1200 → 
  nuts_per_worker = 2000 → 
  nuts_per_bolt = 2 → 
  x ≤ total_workers →
  2 * bolts_per_worker * x = nuts_per_worker * (total_workers - x) := by
  sorry

#check workshop_production_balance

end NUMINAMATH_CALUDE_workshop_production_balance_l3789_378921


namespace NUMINAMATH_CALUDE_abs_neg_three_squared_plus_four_l3789_378978

theorem abs_neg_three_squared_plus_four : |-3^2 + 4| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_squared_plus_four_l3789_378978


namespace NUMINAMATH_CALUDE_sector_arc_length_l3789_378966

/-- Given a sector with central angle 120° and area 300π cm², its arc length is 20π cm. -/
theorem sector_arc_length (θ : ℝ) (S : ℝ) (l : ℝ) : 
  θ = 120 * π / 180 → 
  S = 300 * π → 
  l = 20 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l3789_378966


namespace NUMINAMATH_CALUDE_math_festival_divisibility_l3789_378988

/-- The year of the first math festival -/
def first_festival_year : ℕ := 1990

/-- The base year for calculating festival years -/
def base_year : ℕ := 1989

/-- Predicate to check if a given ordinal number satisfies the divisibility condition -/
def satisfies_condition (N : ℕ) : Prop :=
  (base_year + N) % N = 0

theorem math_festival_divisibility :
  (∃ (first : ℕ), first > 0 ∧ satisfies_condition first ∧
    ∀ (k : ℕ), 0 < k ∧ k < first → ¬satisfies_condition k) ∧
  (∃ (last : ℕ), satisfies_condition last ∧
    ∀ (k : ℕ), k > last → ¬satisfies_condition k) :=
sorry

end NUMINAMATH_CALUDE_math_festival_divisibility_l3789_378988


namespace NUMINAMATH_CALUDE_complex_real_condition_l3789_378903

theorem complex_real_condition (a : ℝ) : 
  let Z : ℂ := (a - 5) / (a^2 + 4*a - 5) + (a^2 + 2*a - 15) * I
  Z.im = 0 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l3789_378903


namespace NUMINAMATH_CALUDE_sign_sum_zero_l3789_378985

theorem sign_sum_zero (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_sum : a + b + c = 0) :
  a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| = 0 := by
  sorry

end NUMINAMATH_CALUDE_sign_sum_zero_l3789_378985


namespace NUMINAMATH_CALUDE_order_of_roots_l3789_378911

theorem order_of_roots (a b c : ℝ) 
  (ha : a = 4^(2/3)) 
  (hb : b = 3^(2/3)) 
  (hc : c = 25^(1/3)) : 
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_order_of_roots_l3789_378911


namespace NUMINAMATH_CALUDE_max_area_right_triangle_l3789_378937

theorem max_area_right_triangle (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a^2 + b^2 = 8^2 → (1/2) * a * b ≤ 16 := by
sorry

end NUMINAMATH_CALUDE_max_area_right_triangle_l3789_378937


namespace NUMINAMATH_CALUDE_point_coordinates_l3789_378900

/-- Given a point P that is 2 units right and 4 units up from the origin (0,0),
    prove that the coordinates of P are (2,4). -/
theorem point_coordinates (P : ℝ × ℝ) 
  (h1 : P.1 = 2)  -- P is 2 units right from the origin
  (h2 : P.2 = 4)  -- P is 4 units up from the origin
  : P = (2, 4) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l3789_378900


namespace NUMINAMATH_CALUDE_line_slope_value_l3789_378961

/-- Given a line l passing through points A(3, m+1) and B(4, 2m+1) with slope π/4, prove that m = 1 -/
theorem line_slope_value (m : ℝ) : 
  (∃ l : Set (ℝ × ℝ), 
    (3, m + 1) ∈ l ∧ 
    (4, 2*m + 1) ∈ l ∧ 
    (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l → (x₂, y₂) ∈ l → x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = Real.pi / 4)) →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_line_slope_value_l3789_378961


namespace NUMINAMATH_CALUDE_book_writing_time_difference_l3789_378907

theorem book_writing_time_difference
  (woody_time : ℕ)
  (total_time : ℕ)
  (h1 : woody_time = 18)
  (h2 : total_time = 39)
  (h3 : woody_time < total_time - woody_time) :
  total_time - 2 * woody_time = 3 :=
sorry

end NUMINAMATH_CALUDE_book_writing_time_difference_l3789_378907


namespace NUMINAMATH_CALUDE_bills_piggy_bank_l3789_378925

theorem bills_piggy_bank (x : ℕ) : 
  (∀ week : ℕ, week ≥ 1 ∧ week ≤ 8 → x + 2 * week = 3 * x) →
  x + 2 * 8 = 24 :=
by sorry

end NUMINAMATH_CALUDE_bills_piggy_bank_l3789_378925


namespace NUMINAMATH_CALUDE_gwens_spent_money_l3789_378905

/-- Gwen's birthday money problem -/
theorem gwens_spent_money (initial_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 5)
  (h2 : remaining_amount = 2) :
  initial_amount - remaining_amount = 3 := by
sorry

end NUMINAMATH_CALUDE_gwens_spent_money_l3789_378905


namespace NUMINAMATH_CALUDE_mandy_book_ratio_l3789_378934

/-- Represents Mandy's book reading progression --/
structure BookReading where
  initial_length : ℕ
  initial_age : ℕ
  current_length : ℕ

/-- Calculates the ratio of book length at twice the starting age to initial book length --/
def length_ratio (r : BookReading) : ℚ :=
  let twice_age_length := r.initial_length * (r.current_length / (4 * 3 * r.initial_length))
  twice_age_length / r.initial_length

/-- Theorem stating the ratio of book length at twice Mandy's starting age to her initial book length --/
theorem mandy_book_ratio : 
  ∀ (r : BookReading), 
  r.initial_length = 8 ∧ 
  r.initial_age = 6 ∧ 
  r.current_length = 480 → 
  length_ratio r = 5 := by
  sorry

#eval length_ratio { initial_length := 8, initial_age := 6, current_length := 480 }

end NUMINAMATH_CALUDE_mandy_book_ratio_l3789_378934


namespace NUMINAMATH_CALUDE_max_product_constraint_l3789_378975

theorem max_product_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2*b = 1) :
  (∀ a' b' : ℝ, 0 < a' → 0 < b' → a' + 2*b' = 1 → a'*b' ≤ a*b) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l3789_378975


namespace NUMINAMATH_CALUDE_equation_solutions_l3789_378932

def solution_set : Set (ℤ × ℤ) :=
  {(2, 1), (1, 0), (2, 2), (0, 0), (1, 2), (0, 1)}

theorem equation_solutions :
  ∀ (x y : ℤ), (x + y = x^2 - x*y + y^2) ↔ (x, y) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3789_378932


namespace NUMINAMATH_CALUDE_total_fish_is_996_l3789_378935

/-- The number of fish each friend has -/
structure FishCount where
  max : ℕ
  sam : ℕ
  joe : ℕ
  harry : ℕ

/-- The conditions of the fish distribution among friends -/
def fish_distribution (fc : FishCount) : Prop :=
  fc.max = 6 ∧
  fc.sam = 3 * fc.max ∧
  fc.joe = 9 * fc.sam ∧
  fc.harry = 5 * fc.joe

/-- The total number of fish for all friends -/
def total_fish (fc : FishCount) : ℕ :=
  fc.max + fc.sam + fc.joe + fc.harry

/-- Theorem stating that the total number of fish is 996 -/
theorem total_fish_is_996 (fc : FishCount) (h : fish_distribution fc) : total_fish fc = 996 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_is_996_l3789_378935


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l3789_378990

theorem scientific_notation_equality : 21500000 = 2.15 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l3789_378990


namespace NUMINAMATH_CALUDE_point_coordinates_l3789_378981

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The third quadrant of the Cartesian coordinate system -/
def third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (A : Point) 
  (h1 : third_quadrant A)
  (h2 : distance_to_x_axis A = 2)
  (h3 : distance_to_y_axis A = 3) :
  A.x = -3 ∧ A.y = -2 :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l3789_378981


namespace NUMINAMATH_CALUDE_maria_coin_stacks_maria_coin_stacks_proof_l3789_378958

/-- Given that Maria has a total of 15 coins and each stack contains 3 coins,
    prove that the number of stacks she has is 5. -/
theorem maria_coin_stacks : ℕ → ℕ → ℕ → Prop :=
  fun (total_coins : ℕ) (coins_per_stack : ℕ) (num_stacks : ℕ) =>
    total_coins = 15 ∧ coins_per_stack = 3 →
    num_stacks * coins_per_stack = total_coins →
    num_stacks = 5

#check maria_coin_stacks

/-- Proof of the theorem -/
theorem maria_coin_stacks_proof : maria_coin_stacks 15 3 5 := by
  sorry

end NUMINAMATH_CALUDE_maria_coin_stacks_maria_coin_stacks_proof_l3789_378958


namespace NUMINAMATH_CALUDE_clerical_employee_fraction_l3789_378949

/-- Proves that the fraction of clerical employees is 4/15 given the conditions -/
theorem clerical_employee_fraction :
  let total_employees : ℕ := 3600
  let clerical_fraction : ℚ := 4/15
  let reduction_factor : ℚ := 3/4
  let remaining_fraction : ℚ := 1/5
  (clerical_fraction * total_employees : ℚ) * reduction_factor =
    remaining_fraction * total_employees :=
by sorry

end NUMINAMATH_CALUDE_clerical_employee_fraction_l3789_378949


namespace NUMINAMATH_CALUDE_environmental_law_support_l3789_378919

theorem environmental_law_support (men : ℕ) (women : ℕ) 
  (men_support_percent : ℚ) (women_support_percent : ℚ) 
  (h1 : men = 200) 
  (h2 : women = 800) 
  (h3 : men_support_percent = 75 / 100) 
  (h4 : women_support_percent = 65 / 100) : 
  (men_support_percent * men + women_support_percent * women) / (men + women) = 67 / 100 := by
  sorry

end NUMINAMATH_CALUDE_environmental_law_support_l3789_378919


namespace NUMINAMATH_CALUDE_a4_range_l3789_378924

theorem a4_range (a₁ a₂ a₃ a₄ : ℝ) 
  (sum_zero : a₁ + a₂ + a₃ = 0)
  (quad_eq : a₁ * a₄^2 + a₂ * a₄ - a₂ = 0)
  (order : a₁ > a₂ ∧ a₂ > a₃) :
  -1/2 - Real.sqrt 5/2 < a₄ ∧ a₄ < -1/2 + Real.sqrt 5/2 := by
sorry

end NUMINAMATH_CALUDE_a4_range_l3789_378924


namespace NUMINAMATH_CALUDE_correct_proportions_l3789_378987

/-- Represents the count of shirts for each color --/
structure ShirtCounts where
  yellow : ℕ
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Represents the proportion of shirts for each color --/
structure ShirtProportions where
  yellow : ℚ
  red : ℚ
  blue : ℚ
  green : ℚ

/-- Given shirt counts, calculates the correct proportions --/
def calculateProportions (counts : ShirtCounts) : ShirtProportions :=
  let total := counts.yellow + counts.red + counts.blue + counts.green
  { yellow := counts.yellow / total
  , red := counts.red / total
  , blue := counts.blue / total
  , green := counts.green / total }

/-- Theorem stating that given the specific shirt counts, the calculated proportions are correct --/
theorem correct_proportions (counts : ShirtCounts)
  (h1 : counts.yellow = 8)
  (h2 : counts.red = 4)
  (h3 : counts.blue = 2)
  (h4 : counts.green = 2) :
  let props := calculateProportions counts
  props.yellow = 1/2 ∧ props.red = 1/4 ∧ props.blue = 1/8 ∧ props.green = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_correct_proportions_l3789_378987


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l3789_378913

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l3789_378913


namespace NUMINAMATH_CALUDE_solution_system_equations_l3789_378997

theorem solution_system_equations (a b c d : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_order : a > b ∧ b > c ∧ c > d) :
  ∃ (x y z t : ℝ),
    (|a - b| * y + |a - c| * z + |a - d| * t = 1) ∧
    (|b - a| * x + |b - c| * z + |b - d| * t = 1) ∧
    (|c - a| * x + |c - b| * y + |c - d| * t = 1) ∧
    (|d - a| * x + |d - b| * y + |d - c| * z = 1) ∧
    (x = 1 / (a - d)) ∧
    (y = 0) ∧
    (z = 0) ∧
    (t = 1 / (a - d)) := by
  sorry

end NUMINAMATH_CALUDE_solution_system_equations_l3789_378997


namespace NUMINAMATH_CALUDE_third_term_is_four_l3789_378902

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ) : ℕ := 2^n - 1

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℕ := S n - S (n-1)

/-- Theorem: The third term of the sequence is 4 -/
theorem third_term_is_four : a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_four_l3789_378902


namespace NUMINAMATH_CALUDE_total_baseball_cards_l3789_378909

/-- The number of baseball cards each person has -/
structure BaseballCards where
  carlos : ℕ
  matias : ℕ
  jorge : ℕ
  ella : ℕ

/-- The conditions of the baseball card problem -/
def baseball_card_problem (cards : BaseballCards) : Prop :=
  cards.carlos = 20 ∧
  cards.matias = cards.carlos - 6 ∧
  cards.jorge = cards.matias ∧
  cards.ella = 2 * (cards.jorge + cards.matias)

/-- The theorem stating the total number of baseball cards -/
theorem total_baseball_cards (cards : BaseballCards) 
  (h : baseball_card_problem cards) : 
  cards.carlos + cards.matias + cards.jorge + cards.ella = 104 := by
  sorry


end NUMINAMATH_CALUDE_total_baseball_cards_l3789_378909


namespace NUMINAMATH_CALUDE_horner_method_v3_l3789_378986

def f (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v3 (a₆ a₅ a₄ a₃ : ℝ) (x : ℝ) : ℝ :=
  ((a₆ * x + a₅) * x + a₄) * x + a₃

theorem horner_method_v3 :
  horner_v3 3 5 6 79 (-4) = -57 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l3789_378986


namespace NUMINAMATH_CALUDE_switches_in_position_A_after_process_l3789_378931

/-- Represents a switch position -/
inductive Position
| A | B | C | D | E

/-- Advances a position cyclically -/
def advance_position (p : Position) : Position :=
  match p with
  | Position.A => Position.B
  | Position.B => Position.C
  | Position.C => Position.D
  | Position.D => Position.E
  | Position.E => Position.A

/-- Represents a switch with its label and position -/
structure Switch :=
  (x y z w : Nat)
  (pos : Position)

/-- The total number of switches -/
def total_switches : Nat := 6860

/-- Creates the initial set of switches -/
def initial_switches : Finset Switch :=
  sorry

/-- Advances a switch and its divisors -/
def advance_switches (switches : Finset Switch) (i : Nat) : Finset Switch :=
  sorry

/-- Performs the entire 6860-step process -/
def process (switches : Finset Switch) : Finset Switch :=
  sorry

/-- Counts switches in position A -/
def count_position_A (switches : Finset Switch) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem switches_in_position_A_after_process :
  count_position_A (process initial_switches) = 6455 :=
sorry

end NUMINAMATH_CALUDE_switches_in_position_A_after_process_l3789_378931


namespace NUMINAMATH_CALUDE_seventh_triangular_is_28_l3789_378994

/-- Triangular number function -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The seventh triangular number is 28 -/
theorem seventh_triangular_is_28 : triangular 7 = 28 := by sorry

end NUMINAMATH_CALUDE_seventh_triangular_is_28_l3789_378994


namespace NUMINAMATH_CALUDE_toby_photo_shoot_l3789_378950

/-- The number of photos Toby took in the photo shoot -/
def photos_in_shoot (initial : ℕ) (deleted_bad : ℕ) (cat_pics : ℕ) (deleted_after : ℕ) (final : ℕ) : ℕ :=
  final - (initial - deleted_bad + cat_pics - deleted_after)

theorem toby_photo_shoot :
  photos_in_shoot 63 7 15 3 84 = 16 := by
  sorry

end NUMINAMATH_CALUDE_toby_photo_shoot_l3789_378950


namespace NUMINAMATH_CALUDE_steven_peach_apple_difference_l3789_378915

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 17

/-- The number of apples Steven has -/
def steven_apples : ℕ := 16

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - 6

/-- The number of apples Jake has -/
def jake_apples : ℕ := steven_apples + 8

/-- Theorem stating that Steven has 1 more peach than apples -/
theorem steven_peach_apple_difference :
  steven_peaches - steven_apples = 1 := by sorry

end NUMINAMATH_CALUDE_steven_peach_apple_difference_l3789_378915


namespace NUMINAMATH_CALUDE_circle_radius_proof_l3789_378952

theorem circle_radius_proof (r p q : ℕ) (m n : ℕ+) :
  -- r is an odd integer
  Odd r →
  -- p and q are prime numbers
  Nat.Prime p →
  Nat.Prime q →
  -- (p^m, q^n) is on the circle with radius r
  p^(m:ℕ) * p^(m:ℕ) + q^(n:ℕ) * q^(n:ℕ) = r * r →
  -- The radius r is equal to 5
  r = 5 := by sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l3789_378952


namespace NUMINAMATH_CALUDE_unique_solution_iff_b_eq_two_or_six_l3789_378944

/-- The function g(x) = x^2 + bx + 2b -/
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 2*b

/-- The statement that |g(x)| ≤ 3 has exactly one solution -/
def has_unique_solution (b : ℝ) : Prop :=
  ∃! x, |g b x| ≤ 3

/-- Theorem: The inequality |x^2 + bx + 2b| ≤ 3 has exactly one solution
    if and only if b = 2 or b = 6 -/
theorem unique_solution_iff_b_eq_two_or_six :
  ∀ b : ℝ, has_unique_solution b ↔ (b = 2 ∨ b = 6) := by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_b_eq_two_or_six_l3789_378944


namespace NUMINAMATH_CALUDE_equation_roots_l3789_378912

theorem equation_roots : 
  let f : ℝ → ℝ := λ x => (21 / (x^2 - 9)) - (3 / (x - 3)) - 2
  ∀ x : ℝ, f x = 0 ↔ x = -3 ∨ x = 5 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_l3789_378912
