import Mathlib

namespace alok_order_cost_l247_24776

def chapati_quantity : ℕ := 16
def chapati_price : ℕ := 6
def rice_quantity : ℕ := 5
def rice_price : ℕ := 45
def vegetable_quantity : ℕ := 7
def vegetable_price : ℕ := 70

def total_cost : ℕ := chapati_quantity * chapati_price + 
                      rice_quantity * rice_price + 
                      vegetable_quantity * vegetable_price

theorem alok_order_cost : total_cost = 811 := by
  sorry

end alok_order_cost_l247_24776


namespace bob_candies_l247_24701

/-- Given that Jennifer bought twice as many candies as Emily, Jennifer bought three times as many
    candies as Bob, and Emily bought 6 candies, prove that Bob bought 4 candies. -/
theorem bob_candies (emily_candies : ℕ) (jennifer_candies : ℕ) (bob_candies : ℕ)
  (h1 : jennifer_candies = 2 * emily_candies)
  (h2 : jennifer_candies = 3 * bob_candies)
  (h3 : emily_candies = 6) :
  bob_candies = 4 := by
  sorry

end bob_candies_l247_24701


namespace arc_length_300_degrees_l247_24792

/-- The length of an arc with radius 2 and central angle 300° is 10π/3 -/
theorem arc_length_300_degrees (r : Real) (θ : Real) : 
  r = 2 → θ = 300 * Real.pi / 180 → r * θ = 10 * Real.pi / 3 := by
  sorry

end arc_length_300_degrees_l247_24792


namespace peter_banana_purchase_l247_24722

def problem (initial_amount : ℕ) 
            (potato_price potato_quantity : ℕ)
            (tomato_price tomato_quantity : ℕ)
            (cucumber_price cucumber_quantity : ℕ)
            (banana_price : ℕ)
            (remaining_amount : ℕ) : Prop :=
  let potato_cost := potato_price * potato_quantity
  let tomato_cost := tomato_price * tomato_quantity
  let cucumber_cost := cucumber_price * cucumber_quantity
  let total_cost := potato_cost + tomato_cost + cucumber_cost
  let banana_cost := initial_amount - remaining_amount - total_cost
  banana_cost / banana_price = 14

theorem peter_banana_purchase :
  problem 500 2 6 3 9 4 5 5 426 := by
  sorry

end peter_banana_purchase_l247_24722


namespace hiker_first_pack_weight_hiker_first_pack_weight_proof_l247_24795

/-- Calculates the weight of the first pack for a hiker given specific conditions --/
theorem hiker_first_pack_weight
  (supplies_per_mile : Real)
  (hiking_rate : Real)
  (hours_per_day : Real)
  (days : Real)
  (resupply_ratio : Real)
  (h1 : supplies_per_mile = 0.5)
  (h2 : hiking_rate = 2.5)
  (h3 : hours_per_day = 8)
  (h4 : days = 5)
  (h5 : resupply_ratio = 0.25)
  : Real :=
  let total_distance := hiking_rate * hours_per_day * days
  let total_supplies := supplies_per_mile * total_distance
  let resupply_weight := resupply_ratio * total_supplies
  let first_pack_weight := total_supplies - resupply_weight
  37.5

theorem hiker_first_pack_weight_proof : hiker_first_pack_weight 0.5 2.5 8 5 0.25 rfl rfl rfl rfl rfl = 37.5 := by
  sorry

end hiker_first_pack_weight_hiker_first_pack_weight_proof_l247_24795


namespace inequality_proof_l247_24794

theorem inequality_proof (x y z : ℝ) : 
  x^2 / (x^2 + 2*y*z) + y^2 / (y^2 + 2*z*x) + z^2 / (z^2 + 2*x*y) ≥ 1 := by
  sorry

end inequality_proof_l247_24794


namespace regular_star_polygon_points_l247_24780

-- Define the structure of a regular star polygon
structure RegularStarPolygon where
  n : ℕ  -- number of points
  A : ℝ  -- measure of each Aᵢ angle in degrees
  B : ℝ  -- measure of each Bᵢ angle in degrees

-- Define the properties of the regular star polygon
def is_valid_regular_star_polygon (p : RegularStarPolygon) : Prop :=
  p.A > 0 ∧ p.B > 0 ∧  -- angles are positive
  p.A = p.B + 15 ∧     -- Aᵢ is 15° more than Bᵢ
  p.n * (p.A + p.B) = 360  -- sum of external angles is 360°

-- Theorem: A regular star polygon with the given conditions has 24 points
theorem regular_star_polygon_points (p : RegularStarPolygon) :
  is_valid_regular_star_polygon p → p.n = 24 :=
by sorry

end regular_star_polygon_points_l247_24780


namespace complex_modulus_problem_l247_24768

theorem complex_modulus_problem (z : ℂ) (h : (1 - 2*I)*z = 5*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l247_24768


namespace part1_part2_l247_24757

-- Define the quadratic expression
def quadratic (a x : ℝ) : ℝ := (a - 2) * x^2 + 2 * (a - 2) * x - 4

-- Part 1
theorem part1 : 
  ∀ x : ℝ, quadratic (-2) x < 0 ↔ x ≠ -1 :=
sorry

-- Part 2
theorem part2 : 
  (∀ x : ℝ, quadratic a x < 0) ↔ a ∈ Set.Ioc (-2) 2 :=
sorry

end part1_part2_l247_24757


namespace books_per_shelf_l247_24793

theorem books_per_shelf (total_books : ℕ) (num_shelves : ℕ) (h1 : total_books = 315) (h2 : num_shelves = 7) :
  total_books / num_shelves = 45 := by
sorry

end books_per_shelf_l247_24793


namespace percentage_problem_l247_24709

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 40 = 4 / 5 * 25 + 6 → P = 65 := by
  sorry

end percentage_problem_l247_24709


namespace polynomial_roots_sum_l247_24798

theorem polynomial_roots_sum (a b c d m n : ℝ) : 
  (∃ (z : ℂ), z^3 + a*z + b = 0 ∧ z^3 + c*z^2 + d = 0) →
  (-20 : ℂ)^3 + a*(-20 : ℂ) + b = 0 →
  (-21 : ℂ)^3 + c*(-21 : ℂ)^2 + d = 0 →
  m > 0 →
  n > 0 →
  (m + Complex.I * Real.sqrt n : ℂ)^3 + a*(m + Complex.I * Real.sqrt n : ℂ) + b = 0 →
  (m + Complex.I * Real.sqrt n : ℂ)^3 + c*(m + Complex.I * Real.sqrt n : ℂ)^2 + d = 0 →
  m + n = 330 := by
sorry

end polynomial_roots_sum_l247_24798


namespace longFurredBrownCount_l247_24785

/-- Represents the number of dogs in a kennel with specific characteristics. -/
structure DogKennel where
  total : ℕ
  longFurred : ℕ
  brown : ℕ
  neither : ℕ

/-- Calculates the number of long-furred brown dogs in the kennel. -/
def longFurredBrown (k : DogKennel) : ℕ :=
  k.longFurred + k.brown - (k.total - k.neither)

/-- Theorem stating the number of long-furred brown dogs in a specific kennel configuration. -/
theorem longFurredBrownCount :
  let k : DogKennel := {
    total := 45,
    longFurred := 26,
    brown := 30,
    neither := 8
  }
  longFurredBrown k = 27 := by sorry

end longFurredBrownCount_l247_24785


namespace odd_integer_dividing_power_plus_one_l247_24767

theorem odd_integer_dividing_power_plus_one (n : ℕ) : 
  n ≥ 1 → 
  Odd n → 
  (n ∣ 3^n + 1) → 
  n = 1 := by
  sorry

end odd_integer_dividing_power_plus_one_l247_24767


namespace min_distance_point_to_curve_l247_24721

theorem min_distance_point_to_curve (α : Real) (h : α ∈ Set.Icc 0 Real.pi) :
  let P : Prod Real Real := (1 + Real.cos α, Real.sin α)
  let C : Set (Prod Real Real) := {Q : Prod Real Real | Q.1 + Q.2 = 9}
  (∃ (d : Real), d = 4 * Real.sqrt 2 - 1 ∧
    ∀ Q ∈ C, Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≥ d) :=
by sorry

end min_distance_point_to_curve_l247_24721


namespace rose_purchase_problem_l247_24735

theorem rose_purchase_problem :
  ∃! (x y : ℤ), 
    (y = 1) ∧ 
    (x > 0) ∧
    (100 / x : ℚ) - (200 / (x + 10) : ℚ) = 80 / 12 ∧
    x = 5 ∧
    y = 1 := by
  sorry

end rose_purchase_problem_l247_24735


namespace bird_triangle_theorem_l247_24787

/-- A bird's position on a regular n-gon --/
structure BirdPosition (n : ℕ) where
  vertex : Fin n

/-- The type of a triangle --/
inductive TriangleType
  | Acute
  | Obtuse
  | RightAngled

/-- Determine the type of a triangle formed by three birds on a regular n-gon --/
def triangleType (n : ℕ) (a b c : BirdPosition n) : TriangleType := sorry

/-- A permutation of birds --/
def BirdPermutation (n : ℕ) := Fin n → Fin n

/-- The main theorem --/
theorem bird_triangle_theorem (n : ℕ) (h : n ≥ 3 ∧ n ≠ 5) :
  ∀ (perm : BirdPermutation n),
  ∃ (a b c : Fin n),
    triangleType n ⟨a⟩ ⟨b⟩ ⟨c⟩ = triangleType n ⟨perm a⟩ ⟨perm b⟩ ⟨perm c⟩ :=
sorry

end bird_triangle_theorem_l247_24787


namespace geometric_sequence_sixth_term_l247_24737

/-- Given a geometric sequence {a_n} with a_1 = 1, a_2 = 2, and a_3 = 4, prove that a_6 = 32 -/
theorem geometric_sequence_sixth_term (a : ℕ → ℝ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 2) 
  (h3 : a 3 = 4) 
  (h_geom : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n) : 
  a 6 = 32 := by
  sorry

end geometric_sequence_sixth_term_l247_24737


namespace intersection_sufficient_not_necessary_for_union_l247_24786

-- Define the sets M and P
def M : Set ℝ := {x | x > 1}
def P : Set ℝ := {x | x < 4}

-- State the theorem
theorem intersection_sufficient_not_necessary_for_union :
  (∀ x, x ∈ M ∩ P → x ∈ M ∪ P) ∧
  (∃ x, x ∈ M ∪ P ∧ x ∉ M ∩ P) := by
  sorry

end intersection_sufficient_not_necessary_for_union_l247_24786


namespace monotone_increasing_interval_l247_24742

/-- A function f is even if f(x) = f(-x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The distance between intersections of a function with a horizontal line -/
def IntersectionDistance (f : ℝ → ℝ) (y : ℝ) : ℝ → ℝ → ℝ :=
  λ x₁ x₂ => |x₂ - x₁|

theorem monotone_increasing_interval
  (ω φ : ℝ)
  (f : ℝ → ℝ)
  (hω : ω > 0)
  (hφ : 0 < φ ∧ φ < π)
  (hf : f = λ x => 2 * Real.sin (ω * x + φ))
  (heven : EvenFunction f)
  (hmin : ∃ x₁ x₂, f x₁ = 2 ∧ f x₂ = 2 ∧ 
    ∀ y₁ y₂, f y₁ = 2 → f y₂ = 2 → 
    IntersectionDistance f 2 y₁ y₂ ≥ IntersectionDistance f 2 x₁ x₂ ∧
    IntersectionDistance f 2 x₁ x₂ = π) :
  StrictMonoOn f (Set.Ioo (-π/2) (-π/4)) :=
sorry

end monotone_increasing_interval_l247_24742


namespace quadratic_root_implies_a_leq_4_l247_24763

/-- The quadratic function f(x) = x^2 + 4x + a has a real root implies a ≤ 4 -/
theorem quadratic_root_implies_a_leq_4 (a : ℝ) :
  (∃ x : ℝ, x^2 + 4*x + a = 0) → a ≤ 4 := by
  sorry

end quadratic_root_implies_a_leq_4_l247_24763


namespace milk_cartons_per_stack_l247_24772

theorem milk_cartons_per_stack (total_cartons : ℕ) (num_stacks : ℕ) 
  (h1 : total_cartons = 799)
  (h2 : num_stacks = 133)
  (h3 : total_cartons % num_stacks = 0) :
  total_cartons / num_stacks = 6 := by
  sorry

end milk_cartons_per_stack_l247_24772


namespace g_range_l247_24762

noncomputable def g (x : ℝ) : ℝ := 
  (Real.cos x ^ 3 + 5 * Real.cos x ^ 2 + 2 * Real.cos x + 3 * Real.sin x ^ 2 - 9) / (Real.cos x - 1)

theorem g_range (x : ℝ) (h : Real.cos x ≠ 1) : 
  6 ≤ g x ∧ g x < 12 := by sorry

end g_range_l247_24762


namespace norm_scalar_multiple_l247_24718

theorem norm_scalar_multiple (v : ℝ × ℝ) :
  ‖v‖ = 7 → ‖(5 : ℝ) • v‖ = 35 := by
  sorry

end norm_scalar_multiple_l247_24718


namespace sangita_flying_hours_l247_24706

/-- Calculates the required flying hours per month to meet a pilot certification goal -/
def required_hours_per_month (total_required : ℕ) (day_hours : ℕ) (night_hours : ℕ) (cross_country_hours : ℕ) (months : ℕ) : ℕ :=
  (total_required - (day_hours + night_hours + cross_country_hours)) / months

/-- Proves that Sangita needs to fly 220 hours per month to meet her goal -/
theorem sangita_flying_hours : 
  required_hours_per_month 1500 50 9 121 6 = 220 := by
  sorry

end sangita_flying_hours_l247_24706


namespace mixture_composition_l247_24764

/-- Represents the composition of a solution --/
structure Solution :=
  (a : ℝ)  -- Percentage of chemical A
  (b : ℝ)  -- Percentage of chemical B
  (sum_to_100 : a + b = 100)

/-- The problem statement --/
theorem mixture_composition 
  (X : Solution)
  (Y : Solution)
  (Z : Solution)
  (h_X : X.a = 40)
  (h_Y : Y.a = 50)
  (h_Z : Z.a = 30)
  : ∃ (x y z : ℝ),
    x + y + z = 100 ∧
    x * X.a / 100 + y * Y.a / 100 + z * Z.a / 100 = 46 ∧
    x = 40 ∧ y = 60 ∧ z = 0 := by
  sorry

end mixture_composition_l247_24764


namespace scientific_notation_34_million_l247_24712

theorem scientific_notation_34_million : 
  ∃ (a : ℝ) (n : ℤ), 34000000 = a * (10 : ℝ)^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.4 ∧ n = 7 := by
  sorry

end scientific_notation_34_million_l247_24712


namespace min_value_theorem_l247_24726

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3) :
  ∀ x, 2 * a + b + c ≥ x → x ≤ 2 * Real.sqrt 3 - 2 :=
by sorry

end min_value_theorem_l247_24726


namespace fencing_requirement_l247_24748

/-- A rectangular field with specific properties. -/
structure RectangularField where
  length : ℝ
  width : ℝ
  area : ℝ
  uncovered_side : ℝ

/-- The theorem stating the fencing requirement for the given field. -/
theorem fencing_requirement (field : RectangularField) 
  (h1 : field.area = 680)
  (h2 : field.uncovered_side = 80)
  (h3 : field.area = field.length * field.width)
  (h4 : field.length = field.uncovered_side) :
  2 * field.width + field.uncovered_side = 97 := by
  sorry

end fencing_requirement_l247_24748


namespace peach_ripeness_difference_l247_24751

def bowl_of_peaches (total_peaches initial_ripe ripening_rate days_passed peaches_eaten : ℕ) : ℕ :=
  let ripe_peaches := initial_ripe + ripening_rate * days_passed - peaches_eaten
  let unripe_peaches := total_peaches - ripe_peaches
  ripe_peaches - unripe_peaches

theorem peach_ripeness_difference :
  bowl_of_peaches 18 4 2 5 3 = 4 := by
  sorry

#eval bowl_of_peaches 18 4 2 5 3

end peach_ripeness_difference_l247_24751


namespace greatest_power_under_500_l247_24714

theorem greatest_power_under_500 (a b : ℕ) : 
  a > 0 → b > 1 → a^b < 500 → (∀ x y : ℕ, x > 0 → y > 1 → x^y < 500 → x^y ≤ a^b) → a + b = 24 := by
  sorry

end greatest_power_under_500_l247_24714


namespace divisible_by_27_l247_24775

theorem divisible_by_27 (n : ℕ) : ∃ k : ℤ, (10 ^ n : ℤ) + 18 * n - 1 = 27 * k := by
  sorry

end divisible_by_27_l247_24775


namespace geometric_series_common_ratio_l247_24781

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -14/27
  let a₃ : ℚ := 56/216
  let r : ℚ := a₂ / a₁
  r = -16/27 := by sorry

end geometric_series_common_ratio_l247_24781


namespace money_ratio_is_two_to_one_l247_24766

/-- The ratio of Peter's money to John's money -/
def money_ratio : ℚ :=
  let peter_money : ℕ := 320
  let quincy_money : ℕ := peter_money + 20
  let andrew_money : ℕ := quincy_money + (quincy_money * 15 / 100)
  let total_money : ℕ := 1200 + 11
  let john_money : ℕ := total_money - peter_money - quincy_money - andrew_money
  peter_money / john_money

theorem money_ratio_is_two_to_one : money_ratio = 2 := by
  sorry

end money_ratio_is_two_to_one_l247_24766


namespace badge_exchange_l247_24746

theorem badge_exchange (t : ℕ) (v : ℕ) : 
  v = t + 5 →
  (v - (24 * v) / 100 + (20 * t) / 100) + 1 = (t - (20 * t) / 100 + (24 * v) / 100) →
  t = 45 ∧ v = 50 :=
by sorry

end badge_exchange_l247_24746


namespace rectangle_area_l247_24770

theorem rectangle_area (x : ℝ) (h : x > 0) : 
  ∃ (width length : ℝ), 
    width > 0 ∧ 
    length = 2 * width ∧ 
    x^2 = width^2 + length^2 ∧ 
    width * length = (2/5) * x^2 := by
  sorry

end rectangle_area_l247_24770


namespace prime_sum_squares_divisibility_l247_24703

theorem prime_sum_squares_divisibility (p : ℕ) (h1 : Nat.Prime p) 
  (h2 : ∃ k : ℕ, 3 * p + 10 = (k^2 + (k+1)^2 + (k+2)^2 + (k+3)^2 + (k+4)^2 + (k+5)^2)) :
  36 ∣ (p - 7) := by
  sorry

end prime_sum_squares_divisibility_l247_24703


namespace mobile_purchase_price_l247_24769

def grinder_price : ℝ := 15000
def grinder_loss_percent : ℝ := 0.04
def mobile_profit_percent : ℝ := 0.10
def total_profit : ℝ := 200

theorem mobile_purchase_price (mobile_price : ℝ) : 
  (grinder_price * (1 - grinder_loss_percent) + mobile_price * (1 + mobile_profit_percent)) - 
  (grinder_price + mobile_price) = total_profit → 
  mobile_price = 8000 := by
sorry

end mobile_purchase_price_l247_24769


namespace arithmetic_sequence_a6_l247_24778

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence with a₂ = 2 and S₄ = 9, a₆ = 4 -/
theorem arithmetic_sequence_a6 (seq : ArithmeticSequence) 
    (h1 : seq.a 2 = 2) 
    (h2 : seq.S 4 = 9) : 
  seq.a 6 = 4 := by
  sorry

end arithmetic_sequence_a6_l247_24778


namespace pipe_filling_time_l247_24734

/-- Proves that Pipe A takes 20 minutes to fill the tank alone given the conditions -/
theorem pipe_filling_time (t : ℝ) : 
  t > 0 →  -- Pipe A fills the tank in t minutes (t must be positive)
  (t / 4 > 0) →  -- Pipe B fills the tank in t/4 minutes (t/4 must be positive)
  (1 / t + 1 / (t / 4) = 1 / 4) →  -- When both pipes are open, it takes 4 minutes to fill the tank
  t = 20 := by
sorry


end pipe_filling_time_l247_24734


namespace sampling_method_is_systematic_l247_24797

/-- Represents a sampling method -/
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic
  | Other

/-- Represents a factory's production line -/
structure ProductionLine where
  product : Type
  conveyorBelt : Bool
  inspectionInterval : ℕ
  fixedPosition : Bool

/-- Determines the sampling method based on the production line characteristics -/
def determineSamplingMethod (line : ProductionLine) : SamplingMethod :=
  if line.conveyorBelt && line.inspectionInterval > 0 && line.fixedPosition then
    SamplingMethod.Systematic
  else
    SamplingMethod.Other

/-- Theorem: The sampling method for the given production line is systematic sampling -/
theorem sampling_method_is_systematic (line : ProductionLine) 
  (h1 : line.conveyorBelt = true)
  (h2 : line.inspectionInterval = 10)
  (h3 : line.fixedPosition = true) :
  determineSamplingMethod line = SamplingMethod.Systematic :=
sorry

end sampling_method_is_systematic_l247_24797


namespace total_first_grade_muffins_l247_24758

def mrs_brier_muffins : ℕ := 218
def mrs_macadams_muffins : ℕ := 320
def mrs_flannery_muffins : ℕ := 417
def mrs_smith_muffins : ℕ := 292
def mr_jackson_muffins : ℕ := 389

theorem total_first_grade_muffins :
  mrs_brier_muffins + mrs_macadams_muffins + mrs_flannery_muffins +
  mrs_smith_muffins + mr_jackson_muffins = 1636 := by
  sorry

end total_first_grade_muffins_l247_24758


namespace initial_cards_count_l247_24782

/-- The number of cards Jennifer had initially -/
def initial_cards : ℕ := sorry

/-- The number of cards eaten by the hippopotamus -/
def eaten_cards : ℕ := 61

/-- The number of cards remaining after some were eaten -/
def remaining_cards : ℕ := 11

/-- Theorem stating that the initial number of cards is 72 -/
theorem initial_cards_count : initial_cards = 72 := by sorry

end initial_cards_count_l247_24782


namespace fourth_roll_three_prob_l247_24713

-- Define the probabilities for each die
def fair_die_prob : ℚ := 1 / 6
def biased_die_three_prob : ℚ := 1 / 2
def biased_die_other_prob : ℚ := 1 / 10

-- Define the probability of selecting each die
def die_selection_prob : ℚ := 1 / 2

-- Define the number of rolls
def num_rolls : ℕ := 4

-- Define the event of rolling three threes in a row
def three_threes_event : Prop := True

-- Theorem statement
theorem fourth_roll_three_prob :
  three_threes_event →
  (die_selection_prob * fair_die_prob^3 * fair_die_prob +
   die_selection_prob * biased_die_three_prob^3 * biased_die_three_prob) /
  (die_selection_prob * fair_die_prob^3 +
   die_selection_prob * biased_die_three_prob^3) = 41 / 84 :=
by sorry

end fourth_roll_three_prob_l247_24713


namespace decimal_has_three_digits_l247_24711

-- Define the decimal number
def decimal : ℚ := 0.049

-- Theorem stating that the decimal has 3 digits after the decimal point
theorem decimal_has_three_digits : 
  (decimal * 1000).num % 1000 ≠ 0 ∧ (decimal * 100).num % 100 = 0 :=
sorry

end decimal_has_three_digits_l247_24711


namespace product_divisibility_l247_24715

theorem product_divisibility (a b c : ℤ) 
  (h1 : (a + b + c)^2 = -(a*b + a*c + b*c))
  (h2 : a + b ≠ 0)
  (h3 : b + c ≠ 0)
  (h4 : a + c ≠ 0) :
  (∃ k : ℤ, (a + b) * (a + c) = k * (b + c)) ∧
  (∃ k : ℤ, (b + c) * (b + a) = k * (a + c)) ∧
  (∃ k : ℤ, (c + a) * (c + b) = k * (a + b)) :=
sorry

end product_divisibility_l247_24715


namespace range_of_k_prove_k_range_l247_24720

-- Define sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}
def B (k : ℝ) : Set ℝ := {x | x < k}

-- Theorem statement
theorem range_of_k (k : ℝ) :
  (A ∪ B k = B k) → k > 2 := by
  sorry

-- The range of k
def k_range : Set ℝ := {k | k > 2}

-- Theorem to prove the range of k
theorem prove_k_range :
  ∀ k, (A ∪ B k = B k) ↔ k ∈ k_range := by
  sorry

end range_of_k_prove_k_range_l247_24720


namespace smallest_sum_four_consecutive_primes_div_by_five_l247_24730

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A function that returns the sum of four consecutive primes starting from the nth prime -/
def sumFourConsecutivePrimes (n : ℕ) : ℕ :=
  nthPrime n + nthPrime (n + 1) + nthPrime (n + 2) + nthPrime (n + 3)

/-- The main theorem -/
theorem smallest_sum_four_consecutive_primes_div_by_five :
  ∀ n : ℕ, sumFourConsecutivePrimes n % 5 = 0 → sumFourConsecutivePrimes n ≥ 60 :=
sorry

end smallest_sum_four_consecutive_primes_div_by_five_l247_24730


namespace intersection_distance_squared_l247_24710

def Circle (center : ℝ × ℝ) (radius : ℝ) := { p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 }

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ := (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2

theorem intersection_distance_squared :
  let circle1 := Circle (5, 0) 5
  let circle2 := Circle (0, 5) 5
  ∀ C D : ℝ × ℝ, C ∈ circle1 ∧ C ∈ circle2 ∧ D ∈ circle1 ∧ D ∈ circle2 ∧ C ≠ D →
  distance_squared C D = 50 := by
sorry

end intersection_distance_squared_l247_24710


namespace keanu_refills_l247_24728

/-- Calculates the number of refills needed for a round trip given the tank capacity, fuel consumption rate, and one-way distance. -/
def refills_needed (tank_capacity : ℚ) (consumption_per_40_miles : ℚ) (one_way_distance : ℚ) : ℚ :=
  let consumption_per_mile := consumption_per_40_miles / 40
  let round_trip_distance := one_way_distance * 2
  let total_consumption := round_trip_distance * consumption_per_mile
  (total_consumption / tank_capacity).ceil

/-- Theorem stating that for the given conditions, 14 refills are needed. -/
theorem keanu_refills :
  refills_needed 8 8 280 = 14 := by
  sorry

end keanu_refills_l247_24728


namespace cricket_team_average_age_l247_24749

theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (remaining_players_age_diff : ℕ),
    team_size = 11 →
    captain_age = 24 →
    wicket_keeper_age_diff = 3 →
    remaining_players_age_diff = 1 →
    ∃ (team_average_age : ℚ),
      team_average_age = 21 ∧
      (team_size : ℚ) * team_average_age = 
        captain_age + (captain_age + wicket_keeper_age_diff) + 
        ((team_size - 2) : ℚ) * (team_average_age - remaining_players_age_diff) :=
by
  sorry

end cricket_team_average_age_l247_24749


namespace geometric_sequence_fourth_term_l247_24736

/-- A geometric sequence of positive integers -/
def GeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℕ)
  (h_geom : GeometricSequence a)
  (h_first : a 1 = 5)
  (h_fifth : a 5 = 1280) :
  a 4 = 320 :=
sorry

end geometric_sequence_fourth_term_l247_24736


namespace congruence_problem_l247_24760

theorem congruence_problem (x : ℤ) :
  (5 * x + 9) % 19 = 3 → (3 * x + 15) % 19 = 0 := by
  sorry

end congruence_problem_l247_24760


namespace correct_quotient_l247_24779

theorem correct_quotient (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 12 = 35) : D / 21 = 20 := by
  sorry

end correct_quotient_l247_24779


namespace sum_fraction_inequality_l247_24774

theorem sum_fraction_inequality (x y z : ℝ) (h : x + y + z = x*y + y*z + z*x) :
  x / (x^2 + 1) + y / (y^2 + 1) + z / (z^2 + 1) ≥ -1/2 := by
  sorry

end sum_fraction_inequality_l247_24774


namespace children_neither_happy_nor_sad_l247_24759

/-- Given a group of children with known happiness status and gender distribution,
    calculate the number of children who are neither happy nor sad. -/
theorem children_neither_happy_nor_sad
  (total_children : ℕ)
  (happy_children : ℕ)
  (sad_children : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (happy_boys : ℕ)
  (sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : boys = 18)
  (h5 : girls = 42)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : total_children = boys + girls)
  : total_children - (happy_children + sad_children) = 20 := by
  sorry

end children_neither_happy_nor_sad_l247_24759


namespace perpendicular_vectors_coefficient_l247_24747

/-- Given two vectors in the plane, if one is perpendicular to a linear combination of both,
    then the coefficient in the linear combination is -5. -/
theorem perpendicular_vectors_coefficient (a b : ℝ × ℝ) (t : ℝ) :
  a = (1, -1) →
  b = (6, -4) →
  (a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) →
  t = -5 := by
  sorry

end perpendicular_vectors_coefficient_l247_24747


namespace or_propositions_true_l247_24761

-- Define the properties of square and rectangle diagonals
def square_diagonals_perpendicular : Prop := True
def rectangle_diagonals_bisect : Prop := True

-- Theorem statement
theorem or_propositions_true : 
  ((2 = 2) ∨ (2 > 2)) ∧ 
  (square_diagonals_perpendicular ∨ rectangle_diagonals_bisect) := by
  sorry

end or_propositions_true_l247_24761


namespace increasing_square_neg_func_l247_24789

/-- Given an increasing function f: ℝ → ℝ with f(x) < 0 for all x,
    the function g(x) = x^2 * f(x) is increasing on (-∞, 0) -/
theorem increasing_square_neg_func
  (f : ℝ → ℝ)
  (h_incr : ∀ x y, x < y → f x < f y)
  (h_neg : ∀ x, f x < 0) :
  ∀ x y, x < y → x < 0 → y < 0 → x^2 * f x < y^2 * f y :=
by sorry

end increasing_square_neg_func_l247_24789


namespace limit_f_at_origin_l247_24717

/-- The function f(x, y) = (x^2 + y^2)^2 x^2 y^2 -/
def f (x y : ℝ) : ℝ := (x^2 + y^2)^2 * x^2 * y^2

/-- The limit of f(x, y) as x and y approach 0 is 1 -/
theorem limit_f_at_origin :
  ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ, x^2 + y^2 < δ^2 → |f x y - 1| < ε :=
by sorry

end limit_f_at_origin_l247_24717


namespace cubic_equation_root_difference_l247_24731

theorem cubic_equation_root_difference (a b c : ℚ) : 
  ∃ (p q r : ℚ), p^3 + a*p^2 + b*p + c = 0 ∧ 
                  q^3 + a*q^2 + b*q + c = 0 ∧ 
                  r^3 + a*r^2 + b*r + c = 0 ∧ 
                  (q - p = 2014 ∨ r - q = 2014 ∨ r - p = 2014) :=
by sorry

end cubic_equation_root_difference_l247_24731


namespace jerrys_average_increase_l247_24745

theorem jerrys_average_increase (initial_average : ℝ) (fourth_test_score : ℝ) : 
  initial_average = 78 →
  fourth_test_score = 86 →
  (3 * initial_average + fourth_test_score) / 4 - initial_average = 2 := by
  sorry

end jerrys_average_increase_l247_24745


namespace f_at_seven_l247_24777

/-- The polynomial f(x) = 7x^5 + 12x^4 - 5x^3 - 6x^2 + 3x - 5 -/
def f (x : ℝ) : ℝ := 7*x^5 + 12*x^4 - 5*x^3 - 6*x^2 + 3*x - 5

/-- Theorem stating that f(7) = 144468 -/
theorem f_at_seven : f 7 = 144468 := by
  sorry

end f_at_seven_l247_24777


namespace cement_mixture_weight_l247_24796

/-- A cement mixture with sand, water, and gravel -/
structure CementMixture where
  total_weight : ℝ
  sand_ratio : ℝ
  water_ratio : ℝ
  gravel_weight : ℝ

/-- Properties of the cement mixture -/
def is_valid_mixture (m : CementMixture) : Prop :=
  m.sand_ratio = 1/2 ∧
  m.water_ratio = 1/5 ∧
  m.gravel_weight = 15 ∧
  m.sand_ratio + m.water_ratio + m.gravel_weight / m.total_weight = 1

/-- Theorem stating that the total weight of the mixture is 50 pounds -/
theorem cement_mixture_weight (m : CementMixture) (h : is_valid_mixture m) : 
  m.total_weight = 50 := by
  sorry

end cement_mixture_weight_l247_24796


namespace rectangle_in_circle_l247_24707

/-- A rectangle with sides 7 cm and 24 cm is inscribed in a circle. -/
theorem rectangle_in_circle (a b r : ℝ) (h1 : a = 7) (h2 : b = 24) 
  (h3 : a^2 + b^2 = (2*r)^2) : 
  (2 * π * r = 25 * π) ∧ (a * b = 168) := by
  sorry

end rectangle_in_circle_l247_24707


namespace sin_n_equals_cos_630_l247_24790

theorem sin_n_equals_cos_630 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * π / 180) = Real.cos (630 * π / 180) ↔ n = 0 ∨ n = 180 ∨ n = -180) :=
by sorry

end sin_n_equals_cos_630_l247_24790


namespace complex_product_theorem_l247_24724

theorem complex_product_theorem (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2)
  (h2 : Complex.abs z₂ = 3)
  (h3 : 3 * z₁ - 2 * z₂ = 2 - Complex.I) :
  z₁ * z₂ = -18/5 + 24/5 * Complex.I := by
  sorry

end complex_product_theorem_l247_24724


namespace solution_is_ten_l247_24753

-- Define the * operation
def star (a b : ℝ) : ℝ := 3 * a - b

-- State the theorem
theorem solution_is_ten :
  ∃ x : ℝ, star 2 (star 5 x) = 1 ∧ x = 10 := by
  sorry

end solution_is_ten_l247_24753


namespace orthogonal_circles_on_radical_axis_l247_24754

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the orthogonality condition
def is_orthogonal (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  c1.radius^2 = (x1 - x2)^2 + (y1 - y2)^2 - c2.radius^2

-- Define the radical axis
def on_radical_axis (p : ℝ × ℝ) (c1 c2 : Circle) : Prop :=
  let (x, y) := p
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x - x1)^2 + (y - y1)^2 - c1.radius^2 = (x - x2)^2 + (y - y2)^2 - c2.radius^2

-- Main theorem
theorem orthogonal_circles_on_radical_axis (S1 S2 : Circle) (O : ℝ × ℝ) :
  (∃ r : ℝ, r > 0 ∧ is_orthogonal ⟨O, r⟩ S1 ∧ is_orthogonal ⟨O, r⟩ S2) ↔
  (on_radical_axis O S1 S2 ∧ O ≠ S1.center ∧ O ≠ S2.center) :=
sorry

end orthogonal_circles_on_radical_axis_l247_24754


namespace jenny_ate_65_squares_l247_24750

-- Define the number of chocolate squares Mike ate
def mike_squares : ℕ := 20

-- Define the number of chocolate squares Jenny ate
def jenny_squares : ℕ := 3 * mike_squares + 5

-- Theorem to prove
theorem jenny_ate_65_squares : jenny_squares = 65 := by
  sorry

end jenny_ate_65_squares_l247_24750


namespace solar_panel_installation_time_l247_24788

/-- Calculates the number of hours needed to install solar panels given the costs of various items --/
def solar_panel_installation_hours (land_acres : ℕ) (land_cost_per_acre : ℕ) 
  (house_cost : ℕ) (cow_count : ℕ) (cow_cost : ℕ) (chicken_count : ℕ) 
  (chicken_cost : ℕ) (solar_panel_hourly_rate : ℕ) (solar_panel_equipment_fee : ℕ) 
  (total_cost : ℕ) : ℕ :=
  let land_cost := land_acres * land_cost_per_acre
  let cows_cost := cow_count * cow_cost
  let chickens_cost := chicken_count * chicken_cost
  let costs_before_solar := land_cost + house_cost + cows_cost + chickens_cost
  let solar_panel_total_cost := total_cost - costs_before_solar
  let installation_cost := solar_panel_total_cost - solar_panel_equipment_fee
  installation_cost / solar_panel_hourly_rate

theorem solar_panel_installation_time : 
  solar_panel_installation_hours 30 20 120000 20 1000 100 5 100 6000 147700 = 6 := by
  sorry

end solar_panel_installation_time_l247_24788


namespace orange_count_after_changes_l247_24725

/-- The number of oranges in a bin after removing some and adding new ones. -/
def oranges_in_bin (initial : ℕ) (removed : ℕ) (added : ℕ) : ℕ :=
  initial - removed + added

/-- Theorem stating that starting with 50 oranges, removing 40, and adding 24 results in 34 oranges. -/
theorem orange_count_after_changes : oranges_in_bin 50 40 24 = 34 := by
  sorry

end orange_count_after_changes_l247_24725


namespace fencing_tournament_l247_24752

theorem fencing_tournament (n : ℕ) : n > 0 → (
  let total_participants := 4*n
  let total_bouts := (total_participants * (total_participants - 1)) / 2
  let womens_wins := 2*n*(3*n)
  let mens_wins := 3*n*(n + 3*n - 1)
  womens_wins * 3 = mens_wins * 2 ∧ womens_wins + mens_wins = total_bouts
) → n = 4 := by sorry

end fencing_tournament_l247_24752


namespace pipe_fill_time_pipe_B_fill_time_l247_24705

/-- Given two pipes A and B that can fill a tank, this theorem proves the time it takes for pipe B to fill the tank. -/
theorem pipe_fill_time (fill_time_A : ℝ) (fill_time_both : ℝ) (fill_amount : ℝ) : ℝ :=
  let fill_rate_A := 1 / fill_time_A
  let fill_rate_both := fill_amount / fill_time_both
  let fill_rate_B := fill_rate_both - fill_rate_A
  1 / fill_rate_B

/-- The main theorem that proves the time it takes for pipe B to fill the tank under the given conditions. -/
theorem pipe_B_fill_time : pipe_fill_time 16 12.000000000000002 (5/4) = 24 := by
  sorry

end pipe_fill_time_pipe_B_fill_time_l247_24705


namespace rectangular_plot_breadth_l247_24783

/-- Given a rectangular plot where the length is thrice the breadth 
    and the area is 972 sq m, prove that the breadth is 18 meters. -/
theorem rectangular_plot_breadth : 
  ∀ (breadth length area : ℝ),
  length = 3 * breadth →
  area = length * breadth →
  area = 972 →
  breadth = 18 := by
sorry

end rectangular_plot_breadth_l247_24783


namespace red_tetrahedron_volume_l247_24799

/-- The volume of a tetrahedron formed by red vertices in a cube with alternately colored vertices -/
theorem red_tetrahedron_volume (cube_side_length : ℝ) (h : cube_side_length = 8) :
  let cube_volume := cube_side_length ^ 3
  let green_tetrahedron_volume := (1 / 3) * (1 / 2 * cube_side_length ^ 2) * cube_side_length
  let red_tetrahedron_volume := cube_volume - 4 * green_tetrahedron_volume
  red_tetrahedron_volume = 512 / 3 := by
  sorry

end red_tetrahedron_volume_l247_24799


namespace new_student_weight_l247_24704

theorem new_student_weight (n : ℕ) (w_avg_initial w_avg_final w_new : ℝ) :
  n = 29 →
  w_avg_initial = 28 →
  w_avg_final = 27.2 →
  (n : ℝ) * w_avg_initial = ((n : ℝ) + 1) * w_avg_final - w_new →
  w_new = 4 := by sorry

end new_student_weight_l247_24704


namespace milk_price_problem_l247_24739

theorem milk_price_problem (initial_cost initial_bottles subsequent_cost : ℝ) : 
  initial_cost = 108 →
  subsequent_cost = 90 →
  ∃ (price : ℝ), 
    initial_bottles * price = initial_cost ∧
    (initial_bottles + 1) * (price * 0.25) = subsequent_cost →
    price = 12 := by
  sorry

end milk_price_problem_l247_24739


namespace cubic_value_in_set_l247_24700

theorem cubic_value_in_set (A : Set ℝ) (a : ℝ) 
  (h1 : 5 ∈ A) 
  (h2 : a^2 + 2*a + 4 ∈ A) 
  (h3 : 7 ∈ A) : 
  a^3 = 1 ∨ a^3 = -27 := by
sorry

end cubic_value_in_set_l247_24700


namespace average_words_per_page_l247_24765

/-- Proves that for a book with given specifications, the average number of words per page is 1250 --/
theorem average_words_per_page
  (sheets : ℕ)
  (total_words : ℕ)
  (pages_per_sheet : ℕ)
  (h1 : sheets = 12)
  (h2 : total_words = 240000)
  (h3 : pages_per_sheet = 16) :
  total_words / (sheets * pages_per_sheet) = 1250 :=
by sorry

end average_words_per_page_l247_24765


namespace wall_width_calculation_l247_24784

/-- Calculates the width of a wall given its other dimensions and the number and size of bricks used. -/
theorem wall_width_calculation 
  (wall_length wall_height : ℝ) 
  (brick_length brick_width brick_height : ℝ)
  (num_bricks : ℕ) : 
  wall_length = 800 ∧ 
  wall_height = 600 ∧
  brick_length = 125 ∧ 
  brick_width = 11.25 ∧ 
  brick_height = 6 ∧
  num_bricks = 1280 →
  ∃ (wall_width : ℝ), 
    wall_width = 22.5 ∧
    wall_length * wall_height * wall_width = 
      num_bricks * (brick_length * brick_width * brick_height) := by
  sorry


end wall_width_calculation_l247_24784


namespace max_average_raise_l247_24738

theorem max_average_raise (R S C A : ℝ) : 
  0.05 < R ∧ R < 0.10 →
  0.07 < S ∧ S < 0.12 →
  0.04 < C ∧ C < 0.09 →
  0.06 < A ∧ A < 0.15 →
  (R + S + C + A) / 4 ≤ 0.085 →
  ∃ (R' S' C' A' : ℝ),
    0.05 < R' ∧ R' < 0.10 ∧
    0.07 < S' ∧ S' < 0.12 ∧
    0.04 < C' ∧ C' < 0.09 ∧
    0.06 < A' ∧ A' < 0.15 ∧
    (R' + S' + C' + A') / 4 = 0.085 :=
by sorry

end max_average_raise_l247_24738


namespace right_triangle_check_l247_24755

/-- Checks if three numbers can form a right-angled triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem right_triangle_check :
  ¬ is_right_triangle 1 3 4 ∧
  ¬ is_right_triangle 2 3 4 ∧
  ¬ is_right_triangle 1 1 (Real.sqrt 3) ∧
  is_right_triangle 5 12 13 :=
by sorry

end right_triangle_check_l247_24755


namespace maria_coin_count_l247_24727

/-- Represents the number of stacks for each coin type -/
structure CoinStacks where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Represents the number of coins in each stack for each coin type -/
structure CoinsPerStack where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total number of coins given the number of stacks and coins per stack -/
def totalCoins (stacks : CoinStacks) (perStack : CoinsPerStack) : ℕ :=
  stacks.pennies * perStack.pennies +
  stacks.nickels * perStack.nickels +
  stacks.dimes * perStack.dimes

theorem maria_coin_count :
  let stacks : CoinStacks := { pennies := 3, nickels := 5, dimes := 7 }
  let perStack : CoinsPerStack := { pennies := 10, nickels := 8, dimes := 4 }
  totalCoins stacks perStack = 98 := by
  sorry

end maria_coin_count_l247_24727


namespace model_b_sample_size_l247_24744

/-- Calculates the number of items to be sampled from a stratum in stratified sampling -/
def stratifiedSampleSize (totalPopulation : ℕ) (stratumSize : ℕ) (totalSampleSize : ℕ) : ℕ :=
  (stratumSize * totalSampleSize) / totalPopulation

theorem model_b_sample_size :
  let totalProduction : ℕ := 9200
  let modelBProduction : ℕ := 6000
  let totalSampleSize : ℕ := 46
  stratifiedSampleSize totalProduction modelBProduction totalSampleSize = 30 := by
  sorry

end model_b_sample_size_l247_24744


namespace product_of_sums_and_differences_l247_24771

theorem product_of_sums_and_differences (P Q R S : ℝ) : 
  P = Real.sqrt 2011 + Real.sqrt 2010 →
  Q = -Real.sqrt 2011 - Real.sqrt 2010 →
  R = Real.sqrt 2011 - Real.sqrt 2010 →
  S = Real.sqrt 2010 - Real.sqrt 2011 →
  P * Q * R * S = -1 := by
  sorry

end product_of_sums_and_differences_l247_24771


namespace hyperbola_standard_equation_l247_24729

/-- Theorem: For a hyperbola passing through the point (4, √3) with asymptotes y = ± (1/2)x, 
    its standard equation is x²/4 - y² = 1 -/
theorem hyperbola_standard_equation 
  (passes_through : (4 : ℝ)^2 / 4 - 3 = 1) 
  (asymptotes : ∀ (x y : ℝ), y = (1/2) * x ∨ y = -(1/2) * x) :
  ∀ (x y : ℝ), x^2 / 4 - y^2 = 1 := by
  sorry

end hyperbola_standard_equation_l247_24729


namespace arithmetic_sequence_sum_property_l247_24773

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_property
  (seq : ArithmeticSequence)
  (h1 : seq.S 3 = 9)
  (h2 : seq.S 6 = 36) :
  seq.a 7 + seq.a 8 + seq.a 9 = 45 := by
  sorry

end arithmetic_sequence_sum_property_l247_24773


namespace paper_strip_sequence_l247_24791

theorem paper_strip_sequence : ∃ (a : Fin 10 → ℝ), 
  a 0 = 9 ∧ 
  a 8 = 5 ∧ 
  ∀ i : Fin 8, a i + a (i + 1) + a (i + 2) = 14 := by
  sorry

end paper_strip_sequence_l247_24791


namespace marbles_fraction_l247_24719

theorem marbles_fraction (total_marbles : ℕ) (marbles_taken : ℕ) :
  total_marbles = 100 →
  marbles_taken = 11 →
  (marbles_taken : ℚ) / (total_marbles : ℚ) = 0.11 := by
  sorry

end marbles_fraction_l247_24719


namespace x_squared_geq_one_necessary_not_sufficient_l247_24723

theorem x_squared_geq_one_necessary_not_sufficient :
  (∀ x : ℝ, x > 1 → x^2 ≥ 1) ∧
  (∃ x : ℝ, x^2 ≥ 1 ∧ ¬(x > 1)) :=
by sorry

end x_squared_geq_one_necessary_not_sufficient_l247_24723


namespace square_area_thirteen_l247_24708

/-- The area of a square with vertices at (1, 1), (-2, 3), (-1, 8), and (2, 4) is 13 square units. -/
theorem square_area_thirteen : 
  let P : ℝ × ℝ := (1, 1)
  let Q : ℝ × ℝ := (-2, 3)
  let R : ℝ × ℝ := (-1, 8)
  let S : ℝ × ℝ := (2, 4)
  let square_area := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
  square_area = 13 := by sorry

end square_area_thirteen_l247_24708


namespace equation_solution_l247_24743

theorem equation_solution : ∀ x : ℝ, 4 * x^2 - (x - 1)^2 = 0 ↔ x = -1 ∨ x = 1/3 := by
  sorry

end equation_solution_l247_24743


namespace rectangular_field_diagonal_l247_24741

/-- Proves that a rectangular field with one side of 15 m and an area of 120 m² has a diagonal of 17 m -/
theorem rectangular_field_diagonal (side : ℝ) (area : ℝ) (diagonal : ℝ) : 
  side = 15 → area = 120 → diagonal = 17 → 
  area = side * (area / side) ∧ diagonal^2 = side^2 + (area / side)^2 := by
  sorry

end rectangular_field_diagonal_l247_24741


namespace probability_closer_to_center_l247_24740

/-- The probability of a randomly chosen point within a circle of radius 5 being closer to the center than to the boundary, given an inner concentric circle of radius 2 -/
theorem probability_closer_to_center (outer_radius inner_radius : ℝ) : 
  outer_radius = 5 → 
  inner_radius = 2 → 
  (π * inner_radius^2) / (π * outer_radius^2) = 4 / 25 := by
  sorry

end probability_closer_to_center_l247_24740


namespace parallelogram45_diag_product_l247_24733

/-- A parallelogram with one angle of 45° -/
structure Parallelogram45 where
  a : ℝ
  b : ℝ
  d₁ : ℝ
  d₂ : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_diag₁ : d₁^2 = a^2 + b^2 + Real.sqrt 2 * a * b
  h_diag₂ : d₂^2 = a^2 + b^2 - Real.sqrt 2 * a * b

/-- The product of squared diagonals equals the sum of fourth powers of sides -/
theorem parallelogram45_diag_product (p : Parallelogram45) :
    p.d₁^2 * p.d₂^2 = p.a^4 + p.b^4 := by
  sorry

end parallelogram45_diag_product_l247_24733


namespace cleaning_fluid_purchase_l247_24716

theorem cleaning_fluid_purchase :
  ∃ (x y : ℕ), 
    30 * x + 20 * y = 160 ∧ 
    x + y = 7 ∧
    ∀ (a b : ℕ), 30 * a + 20 * b = 160 → x + y ≤ a + b :=
by sorry

end cleaning_fluid_purchase_l247_24716


namespace expression_evaluation_l247_24702

theorem expression_evaluation : 
  -14 - (-2)^3 * (1/4) - 16 * ((1/2) - (1/4) + (3/8)) = -22 := by
  sorry

end expression_evaluation_l247_24702


namespace linear_function_theorem_l247_24756

theorem linear_function_theorem (k b : ℝ) :
  (∃ (x y : ℝ), y = k * x + b ∧ x = 0 ∧ y = -2) →
  (1/2 * |2/k| * 2 = 3) →
  ((k = 2/3 ∧ b = -2) ∨ (k = -2/3 ∧ b = -2)) :=
by sorry

end linear_function_theorem_l247_24756


namespace student_grade_problem_l247_24732

theorem student_grade_problem (courses_last_year : ℕ) (courses_year_before : ℕ) 
  (avg_grade_year_before : ℚ) (avg_grade_two_years : ℚ) :
  courses_last_year = 6 →
  courses_year_before = 5 →
  avg_grade_year_before = 40 →
  avg_grade_two_years = 72 →
  (courses_year_before * avg_grade_year_before + 
   courses_last_year * (592 : ℚ) / 6) / (courses_year_before + courses_last_year) = 
  avg_grade_two_years :=
by sorry

end student_grade_problem_l247_24732
