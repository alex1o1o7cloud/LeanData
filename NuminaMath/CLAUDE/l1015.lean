import Mathlib

namespace metaPopulation2050_l1015_101539

-- Define the initial population and year
def initialPopulation : ℕ := 150
def initialYear : ℕ := 2005

-- Define the doubling period and target year
def doublingPeriod : ℕ := 20
def targetYear : ℕ := 2050

-- Define the population growth function
def populationGrowth (years : ℕ) : ℕ :=
  initialPopulation * (2 ^ (years / doublingPeriod))

-- Theorem statement
theorem metaPopulation2050 :
  populationGrowth (targetYear - initialYear) = 600 := by
  sorry

end metaPopulation2050_l1015_101539


namespace original_acid_percentage_l1015_101541

theorem original_acid_percentage (x y : ℝ) :
  (y / (x + y + 1) = 1 / 5) →
  ((y + 1) / (x + y + 2) = 1 / 3) →
  (y / (x + y) = 1 / 4) :=
by sorry

end original_acid_percentage_l1015_101541


namespace triangle_problem_l1015_101517

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧
  Real.sqrt 3 * a = 2 * c * Real.sin A ∧  -- Given condition
  c = Real.sqrt 7 ∧  -- Given condition
  1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 →  -- Area condition
  C = π/3 ∧ a + b = 5 := by
sorry

end triangle_problem_l1015_101517


namespace quadratic_roots_condition_l1015_101561

/-- 
For a quadratic equation x^2 - mx - 1 = 0 to have two roots, 
one greater than 2 and the other less than 2, m must be in the range (3/2, +∞).
-/
theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x < 2 ∧ y > 2 ∧ x^2 - m*x - 1 = 0 ∧ y^2 - m*y - 1 = 0) ↔ 
  m > 3/2 :=
sorry

end quadratic_roots_condition_l1015_101561


namespace quadratic_coefficient_for_specific_parabola_l1015_101598

/-- A quadratic function with vertex (h, k) passing through point (x₀, y₀) has coefficient a -/
def quadratic_coefficient (h k x₀ y₀ : ℚ) : ℚ :=
  (y₀ - k) / ((x₀ - h)^2)

theorem quadratic_coefficient_for_specific_parabola :
  quadratic_coefficient 2 (-3) 6 (-63) = -15/4 := by sorry

end quadratic_coefficient_for_specific_parabola_l1015_101598


namespace playground_area_l1015_101524

/-- Given a rectangular playground with perimeter 90 feet and length three times the width,
    prove that its area is 380.625 square feet. -/
theorem playground_area (w : ℝ) (l : ℝ) :
  (2 * l + 2 * w = 90) →  -- Perimeter is 90 feet
  (l = 3 * w) →           -- Length is three times the width
  (l * w = 380.625) :=    -- Area is 380.625 square feet
by sorry

end playground_area_l1015_101524


namespace trader_profit_percentage_l1015_101530

theorem trader_profit_percentage (original_price : ℝ) (original_price_positive : original_price > 0) : 
  let discount_rate : ℝ := 0.20
  let purchase_price : ℝ := original_price * (1 - discount_rate)
  let markup_rate : ℝ := 0.60
  let selling_price : ℝ := purchase_price * (1 + markup_rate)
  let profit : ℝ := selling_price - original_price
  let profit_percentage : ℝ := (profit / original_price) * 100
  profit_percentage = 28 := by
sorry

end trader_profit_percentage_l1015_101530


namespace henrys_classical_cds_l1015_101525

/-- Given Henry's CD collection, prove the number of classical CDs --/
theorem henrys_classical_cds :
  ∀ (country rock classical : ℕ),
    country = 23 →
    country = rock + 3 →
    rock = 2 * classical →
    classical = 10 := by
  sorry

end henrys_classical_cds_l1015_101525


namespace days_between_appointments_l1015_101549

/-- Represents the waiting periods for Mark's vaccine appointments -/
structure VaccineWaitingPeriod where
  totalWait : ℕ
  initialWait : ℕ
  finalWait : ℕ

/-- Theorem stating the number of days between first and second appointments -/
theorem days_between_appointments (mark : VaccineWaitingPeriod)
  (h1 : mark.totalWait = 38)
  (h2 : mark.initialWait = 4)
  (h3 : mark.finalWait = 14) :
  mark.totalWait - mark.initialWait - mark.finalWait = 20 := by
  sorry

#check days_between_appointments

end days_between_appointments_l1015_101549


namespace triangle_area_is_twelve_l1015_101596

/-- The area of a triangular region bounded by the coordinate axes and the line 3x + 2y = 12 -/
def triangleArea : ℝ := 12

/-- The equation of the line bounding the triangular region -/
def lineEquation (x y : ℝ) : Prop := 3 * x + 2 * y = 12

theorem triangle_area_is_twelve :
  ∃ (x_intercept y_intercept : ℝ),
    lineEquation x_intercept 0 ∧
    lineEquation 0 y_intercept ∧
    triangleArea = (1 / 2) * x_intercept * y_intercept :=
by sorry

end triangle_area_is_twelve_l1015_101596


namespace thirty_percent_of_eighty_l1015_101514

theorem thirty_percent_of_eighty : ∃ x : ℝ, (30 / 100) * x = 24 ∧ x = 80 := by sorry

end thirty_percent_of_eighty_l1015_101514


namespace unique_positive_integer_solution_l1015_101557

theorem unique_positive_integer_solution : 
  ∃! (x : ℕ), (4 * x)^2 - 2 * x = 8062 := by
sorry

end unique_positive_integer_solution_l1015_101557


namespace white_tulips_multiple_of_seven_l1015_101594

/-- The number of red tulips -/
def red_tulips : ℕ := 91

/-- The number of identical bouquets that can be made -/
def num_bouquets : ℕ := 7

/-- The number of white tulips -/
def white_tulips : ℕ := sorry

/-- Proposition stating that the number of white tulips is a multiple of 7 -/
theorem white_tulips_multiple_of_seven :
  ∃ k : ℕ, white_tulips = 7 * k ∧ red_tulips % num_bouquets = 0 :=
sorry

end white_tulips_multiple_of_seven_l1015_101594


namespace starting_lineups_count_l1015_101591

def total_players : ℕ := 15
def lineup_size : ℕ := 6
def injured_players : ℕ := 1
def incompatible_players : ℕ := 2

theorem starting_lineups_count :
  (Nat.choose (total_players - incompatible_players - injured_players + 1) (lineup_size - 1)) * 2 +
  (Nat.choose (total_players - incompatible_players - injured_players) lineup_size) = 3498 := by
  sorry

end starting_lineups_count_l1015_101591


namespace valentino_farm_birds_l1015_101537

/-- The number of birds on Mr. Valentino's farm -/
def total_birds (chickens ducks turkeys : ℕ) : ℕ := chickens + ducks + turkeys

/-- Theorem stating the total number of birds on Mr. Valentino's farm -/
theorem valentino_farm_birds :
  ∃ (chickens ducks turkeys : ℕ),
    chickens = 200 ∧
    ducks = 2 * chickens ∧
    turkeys = 3 * ducks ∧
    total_birds chickens ducks turkeys = 1800 :=
by sorry

end valentino_farm_birds_l1015_101537


namespace bridge_length_calculation_l1015_101526

/-- The length of a bridge given train characteristics and crossing time -/
theorem bridge_length_calculation 
  (train_length : ℝ) 
  (crossing_time : ℝ) 
  (train_speed_kmph : ℝ) 
  (h1 : train_length = 120)
  (h2 : crossing_time = 26.997840172786177)
  (h3 : train_speed_kmph = 36) :
  ∃ bridge_length : ℝ, 
    bridge_length = 149.97840172786177 ∧ 
    bridge_length = (train_speed_kmph * 1000 / 3600) * crossing_time - train_length :=
by sorry

end bridge_length_calculation_l1015_101526


namespace darias_remaining_debt_l1015_101593

def savings : ℕ := 500
def couch_price : ℕ := 750
def table_price : ℕ := 100
def lamp_price : ℕ := 50

theorem darias_remaining_debt : 
  (couch_price + table_price + lamp_price) - savings = 400 := by
  sorry

end darias_remaining_debt_l1015_101593


namespace pushkin_pension_is_survivor_l1015_101538

-- Define the types of pensions
inductive PensionType
| Retirement
| Disability
| Survivor

-- Define a structure for a pension
structure Pension where
  recipient : String
  year_assigned : Nat
  is_lifelong : Bool
  type : PensionType

-- Define Pushkin's family pension
def pushkin_family_pension : Pension :=
  { recipient := "Pushkin's wife and daughters"
  , year_assigned := 1837
  , is_lifelong := true
  , type := PensionType.Survivor }

-- Theorem statement
theorem pushkin_pension_is_survivor :
  pushkin_family_pension.type = PensionType.Survivor :=
by sorry

end pushkin_pension_is_survivor_l1015_101538


namespace ball_probability_l1015_101550

/-- Given a bag of 100 balls with specific color distributions, 
    prove that the probability of choosing a ball that is neither red nor purple is 0.8 -/
theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 100)
  (h_white : white = 50)
  (h_green : green = 20)
  (h_yellow : yellow = 10)
  (h_red : red = 17)
  (h_purple : purple = 3)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 0.8 := by
sorry

end ball_probability_l1015_101550


namespace power_of_product_l1015_101583

theorem power_of_product (a b : ℝ) : (2 * a * b^2)^3 = 8 * a^3 * b^6 := by
  sorry

end power_of_product_l1015_101583


namespace order_silk_total_l1015_101579

/-- The total yards of silk dyed for an order, given the yards of green and pink silk. -/
def total_silk_dyed (green_silk : ℕ) (pink_silk : ℕ) : ℕ :=
  green_silk + pink_silk

/-- Theorem stating that the total yards of silk dyed for the order is 111421 yards. -/
theorem order_silk_total : 
  total_silk_dyed 61921 49500 = 111421 := by
  sorry

end order_silk_total_l1015_101579


namespace count_pairs_eq_five_l1015_101592

/-- The number of pairs of natural numbers (a, b) satisfying the given conditions -/
def count_pairs : ℕ := 5

/-- Predicate to check if a pair of natural numbers satisfies the equation -/
def satisfies_equation (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 6

/-- The main theorem stating that there are exactly 5 pairs satisfying the conditions -/
theorem count_pairs_eq_five :
  (∃! (s : Finset (ℕ × ℕ)), s.card = count_pairs ∧ 
    (∀ (p : ℕ × ℕ), p ∈ s ↔ 
      (p.1 ≥ p.2 ∧ satisfies_equation p.1 p.2))) :=
by sorry

end count_pairs_eq_five_l1015_101592


namespace tom_distance_before_karen_wins_l1015_101572

/-- Represents the car race scenario between Karen and Tom -/
def CarRace (karen_speed tom_speed : ℝ) (karen_delay : ℝ) (winning_margin : ℝ) : Prop :=
  let race_time := (karen_delay * karen_speed + winning_margin) / (karen_speed - tom_speed)
  tom_speed * race_time = 24

/-- Theorem stating the distance Tom drives before Karen wins -/
theorem tom_distance_before_karen_wins :
  CarRace 60 45 (4/60) 4 :=
by sorry

end tom_distance_before_karen_wins_l1015_101572


namespace x_squared_is_quadratic_x_squared_eq_zero_is_quadratic_l1015_101564

/-- A quadratic equation in one variable is of the form ax² + bx + c = 0, where a, b, and c are constants, and a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x² is a quadratic equation in one variable -/
theorem x_squared_is_quadratic : is_quadratic_equation (λ x => x^2) := by
  sorry

/-- The equation x² = 0 is equivalent to the function f(x) = x² -/
theorem x_squared_eq_zero_is_quadratic : is_quadratic_equation (λ x => x^2 - 0) := by
  sorry

end x_squared_is_quadratic_x_squared_eq_zero_is_quadratic_l1015_101564


namespace min_value_theorem_l1015_101543

theorem min_value_theorem (a k b m n : ℝ) : 
  a > 0 → a ≠ 1 → 
  (∀ x, a^(x-1) + 1 = b → x = k) → 
  m + n = b - k → m > 0 → n > 0 → 
  (9/m + 1/n ≥ 16 ∧ ∃ m n, 9/m + 1/n = 16) :=
by sorry

end min_value_theorem_l1015_101543


namespace xaxaxa_divisible_by_seven_l1015_101527

-- Define a function to create the six-digit number XAXAXA
def makeNumber (X A : Nat) : Nat :=
  100000 * X + 10000 * A + 1000 * X + 100 * A + 10 * X + A

-- Theorem statement
theorem xaxaxa_divisible_by_seven (X A : Nat) (h1 : X < 10) (h2 : A < 10) :
  (makeNumber X A) % 7 = 0 := by
  sorry

end xaxaxa_divisible_by_seven_l1015_101527


namespace weight_of_scaled_object_l1015_101554

/-- Given two similar three-dimensional objects where one has all dimensions 3 times
    larger than the other, if the smaller object weighs 10 grams, 
    then the larger object weighs 270 grams. -/
theorem weight_of_scaled_object (weight_small : ℝ) (scale_factor : ℝ) :
  weight_small = 10 →
  scale_factor = 3 →
  weight_small * scale_factor^3 = 270 := by
sorry


end weight_of_scaled_object_l1015_101554


namespace hyperbola_asymptote_a_l1015_101551

/-- Given a hyperbola with equation x²/a² - y²/9 = 1 where a > 0,
    if the asymptote equations are 3x ± 2y = 0, then a = 2 -/
theorem hyperbola_asymptote_a (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1 →
    (3 * x + 2 * y = 0 ∨ 3 * x - 2 * y = 0)) →
  a = 2 := by
  sorry

end hyperbola_asymptote_a_l1015_101551


namespace find_number_l1015_101568

theorem find_number (x : ℝ) : (2 * x - 8 = -12) → x = -2 := by
  sorry

end find_number_l1015_101568


namespace vacuum_cleaner_cost_l1015_101503

theorem vacuum_cleaner_cost (dishwasher_cost coupon_value total_spent : ℕ) 
  (h1 : dishwasher_cost = 450)
  (h2 : coupon_value = 75)
  (h3 : total_spent = 625) :
  ∃ (vacuum_cost : ℕ), vacuum_cost = 250 ∧ vacuum_cost + dishwasher_cost - coupon_value = total_spent :=
by sorry

end vacuum_cleaner_cost_l1015_101503


namespace centroid_division_weight_theorem_l1015_101587

/-- Represents a triangle with a given total weight -/
structure WeightedTriangle where
  totalWeight : ℝ
  weightProportionalToArea : Bool

/-- Represents a line passing through the centroid of a triangle -/
structure CentroidLine where
  triangle : WeightedTriangle

/-- Represents the two parts of a triangle divided by a centroid line -/
structure DividedTriangle where
  centroidLine : CentroidLine
  part1Weight : ℝ
  part2Weight : ℝ

/-- The theorem to be proved -/
theorem centroid_division_weight_theorem (t : WeightedTriangle) (l : CentroidLine) (d : DividedTriangle) :
  t.totalWeight = 900 ∧ t.weightProportionalToArea = true ∧ l.triangle = t ∧ d.centroidLine = l →
  d.part1Weight ≥ 400 ∧ d.part2Weight ≥ 400 :=
by sorry

end centroid_division_weight_theorem_l1015_101587


namespace coconut_ratio_l1015_101519

theorem coconut_ratio (paolo_coconuts : ℕ) (dante_sold : ℕ) (dante_remaining : ℕ) :
  paolo_coconuts = 14 →
  dante_sold = 10 →
  dante_remaining = 32 →
  (dante_remaining : ℚ) / paolo_coconuts = 16 / 7 := by
  sorry

end coconut_ratio_l1015_101519


namespace fraction_equivalence_l1015_101589

theorem fraction_equivalence : 
  ∀ (n : ℚ), (3 + n) / (5 + n) = 5 / 6 → n = 7 := by sorry

end fraction_equivalence_l1015_101589


namespace polynomial_factorization_l1015_101566

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a) := by
  sorry

end polynomial_factorization_l1015_101566


namespace number_puzzle_l1015_101548

theorem number_puzzle : ∃! x : ℝ, 0.8 * x + 20 = x := by sorry

end number_puzzle_l1015_101548


namespace nails_per_plank_l1015_101540

theorem nails_per_plank (total_nails : ℕ) (total_planks : ℕ) (h1 : total_nails = 4) (h2 : total_planks = 2) :
  total_nails / total_planks = 2 := by
sorry

end nails_per_plank_l1015_101540


namespace sufficient_condition_for_vector_norm_equality_l1015_101544

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- For non-zero vectors a and b, if a + 2b = 0, then |a - b| = |a| + |b| -/
theorem sufficient_condition_for_vector_norm_equality 
  (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) (h : a + 2 • b = 0) : 
  ‖a - b‖ = ‖a‖ + ‖b‖ := by
  sorry

end sufficient_condition_for_vector_norm_equality_l1015_101544


namespace range_of_f_l1015_101569

noncomputable def f (x : ℝ) : ℝ := 3 * (x + 5) * (x - 4) / (x + 5)

theorem range_of_f :
  Set.range f = {y : ℝ | y < -27 ∨ y > -27} := by sorry

end range_of_f_l1015_101569


namespace residue_of_5_1234_mod_19_l1015_101508

theorem residue_of_5_1234_mod_19 : 
  (5 : ℤ)^1234 ≡ 7 [ZMOD 19] := by
  sorry

end residue_of_5_1234_mod_19_l1015_101508


namespace badge_exchange_problem_l1015_101565

theorem badge_exchange_problem (V T : ℕ) :
  V = T + 5 →
  (V - V * 24 / 100 + T * 20 / 100) = (T - T * 20 / 100 + V * 24 / 100 - 1) →
  V = 50 ∧ T = 45 := by
sorry

end badge_exchange_problem_l1015_101565


namespace max_value_of_f_l1015_101577

noncomputable def f (x : ℝ) : ℝ := x * Real.sqrt (18 - x) + Real.sqrt (18 * x - x ^ 3)

theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 18 ∧
  f x = 2 * Real.sqrt 17 ∧
  ∀ y ∈ Set.Icc 0 18, f y ≤ f x :=
sorry

end max_value_of_f_l1015_101577


namespace root_implies_b_value_l1015_101512

-- Define the polynomial
def f (a b : ℚ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 15

-- State the theorem
theorem root_implies_b_value (a : ℚ) :
  (∃ b : ℚ, f a b (3 + Real.sqrt 5) = 0) →
  (∃ b : ℚ, b = -37/2) :=
by sorry

end root_implies_b_value_l1015_101512


namespace skew_quadrilateral_angle_sum_less_than_360_l1015_101509

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The angle between three points in 3D space -/
noncomputable def angle (A B C : Point3D) : ℝ := sorry

/-- Four points are non-coplanar if they do not lie in the same plane -/
def nonCoplanar (A B C D : Point3D) : Prop := sorry

/-- A skew quadrilateral is formed by four non-coplanar points -/
structure SkewQuadrilateral where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  nonCoplanar : nonCoplanar A B C D

theorem skew_quadrilateral_angle_sum_less_than_360 (quad : SkewQuadrilateral) :
  angle quad.A quad.B quad.C + angle quad.B quad.C quad.D +
  angle quad.C quad.D quad.A + angle quad.D quad.A quad.B < 2 * π :=
sorry

end skew_quadrilateral_angle_sum_less_than_360_l1015_101509


namespace walnut_trees_before_planting_l1015_101511

theorem walnut_trees_before_planting 
  (initial : ℕ) 
  (planted : ℕ) 
  (final : ℕ) 
  (h1 : planted = 6) 
  (h2 : final = 10) 
  (h3 : final = initial + planted) : 
  initial = 4 :=
by sorry

end walnut_trees_before_planting_l1015_101511


namespace fraction_equality_l1015_101505

theorem fraction_equality (x : ℝ) : (4 + x) / (7 + x) = (2 + x) / (3 + x) ↔ x = -1 := by
  sorry

end fraction_equality_l1015_101505


namespace debate_club_green_teams_l1015_101570

theorem debate_club_green_teams 
  (total_members : ℕ) 
  (red_members : ℕ) 
  (green_members : ℕ) 
  (total_teams : ℕ) 
  (red_red_teams : ℕ) : 
  total_members = 132 → 
  red_members = 48 → 
  green_members = 84 → 
  total_teams = 66 → 
  red_red_teams = 15 → 
  ∃ (green_green_teams : ℕ), green_green_teams = 33 ∧ 
    green_green_teams = (green_members - (total_members - 2 * red_red_teams - red_members)) / 2 :=
by sorry

end debate_club_green_teams_l1015_101570


namespace perimeter_is_72_l1015_101585

/-- A geometric figure formed by six identical squares arranged into a larger rectangle,
    with two smaller identical squares placed inside. -/
structure GeometricFigure where
  /-- The side length of each of the six identical squares forming the larger rectangle -/
  side_length : ℝ
  /-- The total area of the figure -/
  total_area : ℝ
  /-- The area of the figure is equal to the area of six squares -/
  area_eq : total_area = 6 * side_length^2

/-- The perimeter of the geometric figure -/
def perimeter (fig : GeometricFigure) : ℝ :=
  2 * (3 * fig.side_length + 2 * fig.side_length) + 2 * fig.side_length

theorem perimeter_is_72 (fig : GeometricFigure) (h : fig.total_area = 216) :
  perimeter fig = 72 := by
  sorry

end perimeter_is_72_l1015_101585


namespace remainder_123456789012_mod_252_l1015_101500

theorem remainder_123456789012_mod_252 : 123456789012 % 252 = 108 := by
  sorry

end remainder_123456789012_mod_252_l1015_101500


namespace sally_has_five_balloons_l1015_101553

/-- The number of blue balloons Sally has -/
def sallys_balloons (total joan jessica : ℕ) : ℕ :=
  total - joan - jessica

/-- Theorem stating that Sally has 5 blue balloons given the conditions -/
theorem sally_has_five_balloons :
  sallys_balloons 16 9 2 = 5 := by
  sorry

end sally_has_five_balloons_l1015_101553


namespace school_boys_count_l1015_101535

theorem school_boys_count (total_girls : ℕ) (girl_boy_difference : ℕ) (boys : ℕ) : 
  total_girls = 697 →
  girl_boy_difference = 228 →
  total_girls = boys + girl_boy_difference →
  boys = 469 := by
sorry

end school_boys_count_l1015_101535


namespace inequality_holds_iff_l1015_101581

theorem inequality_holds_iff (a : ℝ) : 
  (∀ x : ℝ, -3 < (x^2 + a*x - 2) / (x^2 - x + 1) ∧ (x^2 + a*x - 2) / (x^2 - x + 1) < 2) ↔ 
  (-1 < a ∧ a < 2) :=
by sorry

end inequality_holds_iff_l1015_101581


namespace brian_always_wins_l1015_101580

/-- Represents the game board -/
structure GameBoard :=
  (n : ℕ)

/-- Represents a player in the game -/
inductive Player := | Albus | Brian

/-- Represents a position on the game board -/
structure Position :=
  (x : ℤ)
  (y : ℤ)

/-- Represents the state of the game -/
structure GameState :=
  (board : GameBoard)
  (position : Position)
  (current_player : Player)
  (move_distance : ℕ)

/-- Checks if a position is within the game board -/
def is_valid_position (board : GameBoard) (pos : Position) : Prop :=
  abs pos.x ≤ board.n ∧ abs pos.y ≤ board.n

/-- Defines the initial game state -/
def initial_state (n : ℕ) : GameState :=
  { board := { n := n },
    position := { x := 0, y := 0 },
    current_player := Player.Albus,
    move_distance := 1 }

/-- Theorem: Brian always has a winning strategy -/
theorem brian_always_wins (n : ℕ) :
  ∃ (strategy : GameState → Position),
    ∀ (game : GameState),
      game.current_player = Player.Brian →
      is_valid_position game.board (strategy game) →
      ¬is_valid_position game.board
        {x := 2 * game.position.x - (strategy game).x,
         y := 2 * game.position.y - (strategy game).y} :=
sorry

end brian_always_wins_l1015_101580


namespace largest_difference_in_grid_l1015_101567

/-- A type representing a 20x20 grid of integers -/
def Grid := Fin 20 → Fin 20 → Fin 400

/-- The property that a grid contains all integers from 1 to 400 -/
def contains_all_integers (g : Grid) : Prop :=
  ∀ n : Fin 400, ∃ i j : Fin 20, g i j = n

/-- The property that there exist two numbers in the same row or column with a difference of at least N -/
def has_difference_at_least (g : Grid) (N : ℕ) : Prop :=
  ∃ i j k : Fin 20, (g i j).val + N ≤ (g i k).val ∨ (g j i).val + N ≤ (g k i).val

/-- The main theorem: 209 is the largest N satisfying the condition -/
theorem largest_difference_in_grid :
  (∀ g : Grid, contains_all_integers g → has_difference_at_least g 209) ∧
  ¬(∀ g : Grid, contains_all_integers g → has_difference_at_least g 210) :=
sorry

end largest_difference_in_grid_l1015_101567


namespace max_reflections_theorem_l1015_101518

/-- The angle between the two reflecting lines in degrees -/
def angle : ℝ := 12

/-- The maximum angle of incidence in degrees -/
def max_incidence : ℝ := 90

/-- The maximum number of reflections possible -/
def max_reflections : ℕ := 7

/-- Theorem stating the maximum number of reflections possible given the angle between lines -/
theorem max_reflections_theorem : 
  ∀ n : ℕ, (n : ℝ) * angle ≤ max_incidence ↔ n ≤ max_reflections :=
by sorry

end max_reflections_theorem_l1015_101518


namespace store_profit_ratio_l1015_101563

/-- Represents the cost and sales information for a product. -/
structure Product where
  cost : ℝ
  markup : ℝ
  salesRatio : ℝ

/-- Represents the store's product lineup. -/
structure Store where
  peachSlices : Product
  riceCrispyTreats : Product
  sesameSnacks : Product

theorem store_profit_ratio (s : Store) : 
  s.peachSlices.cost = 2 * s.sesameSnacks.cost ∧
  s.peachSlices.markup = 0.2 ∧
  s.riceCrispyTreats.markup = 0.3 ∧
  s.sesameSnacks.markup = 0.2 ∧
  s.peachSlices.salesRatio = 1 ∧
  s.riceCrispyTreats.salesRatio = 3 ∧
  s.sesameSnacks.salesRatio = 2 ∧
  (s.peachSlices.markup * s.peachSlices.cost * s.peachSlices.salesRatio +
   s.riceCrispyTreats.markup * s.riceCrispyTreats.cost * s.riceCrispyTreats.salesRatio +
   s.sesameSnacks.markup * s.sesameSnacks.cost * s.sesameSnacks.salesRatio) = 
  0.25 * (s.peachSlices.cost * s.peachSlices.salesRatio +
          s.riceCrispyTreats.cost * s.riceCrispyTreats.salesRatio +
          s.sesameSnacks.cost * s.sesameSnacks.salesRatio) →
  s.riceCrispyTreats.cost / s.sesameSnacks.cost = 4 / 3 := by
sorry


end store_profit_ratio_l1015_101563


namespace divisibility_power_increase_l1015_101502

theorem divisibility_power_increase (k m n : ℕ) (a : ℕ → ℕ) :
  (m^n ∣ a k) → (m^(n+1) ∣ a (k*m)) :=
sorry

end divisibility_power_increase_l1015_101502


namespace range_of_a_l1015_101545

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1/2 then (1/2)^(x - 1/2) else Real.log x / Real.log a

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → a ∈ Set.Icc (Real.sqrt 2 / 2) 1 ∧ a < 1 :=
sorry

end range_of_a_l1015_101545


namespace negative_fraction_comparison_l1015_101558

theorem negative_fraction_comparison : -4/5 < -2/3 := by
  sorry

end negative_fraction_comparison_l1015_101558


namespace percentage_in_70to79_is_25_percent_l1015_101521

/-- Represents the score ranges in Ms. Hernandez's biology class -/
inductive ScoreRange
  | Above90
  | Range80to89
  | Range70to79
  | Range60to69
  | Below60

/-- The frequency of students in each score range -/
def frequency (range : ScoreRange) : ℕ :=
  match range with
  | ScoreRange.Above90 => 5
  | ScoreRange.Range80to89 => 9
  | ScoreRange.Range70to79 => 7
  | ScoreRange.Range60to69 => 4
  | ScoreRange.Below60 => 3

/-- The total number of students in the class -/
def totalStudents : ℕ := 
  frequency ScoreRange.Above90 +
  frequency ScoreRange.Range80to89 +
  frequency ScoreRange.Range70to79 +
  frequency ScoreRange.Range60to69 +
  frequency ScoreRange.Below60

/-- The percentage of students who scored in the 70%-79% range -/
def percentageIn70to79Range : ℚ :=
  (frequency ScoreRange.Range70to79 : ℚ) / (totalStudents : ℚ) * 100

theorem percentage_in_70to79_is_25_percent :
  percentageIn70to79Range = 25 := by
  sorry

end percentage_in_70to79_is_25_percent_l1015_101521


namespace qin_jiushao_correct_f_3_value_qin_jiushao_f_3_l1015_101559

-- Define the polynomial coefficients
def a₀ : ℝ := -0.8
def a₁ : ℝ := 1.7
def a₂ : ℝ := -2.6
def a₃ : ℝ := 3.5
def a₄ : ℝ := 2
def a₅ : ℝ := 4

-- Define Qin Jiushao's algorithm
def qin_jiushao (x : ℝ) : ℝ :=
  let v₀ := a₅
  let v₁ := v₀ * x + a₄
  let v₂ := v₁ * x + a₃
  let v₃ := v₂ * x + a₂
  let v₄ := v₃ * x + a₁
  v₄ * x + a₀

-- Define the polynomial function
def f (x : ℝ) : ℝ := a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

-- Theorem stating that Qin Jiushao's algorithm gives the correct result for f(3)
theorem qin_jiushao_correct : qin_jiushao 3 = f 3 := by sorry

-- Theorem stating that f(3) equals 1209.4
theorem f_3_value : f 3 = 1209.4 := by sorry

-- Main theorem combining the above results
theorem qin_jiushao_f_3 : qin_jiushao 3 = 1209.4 := by sorry

end qin_jiushao_correct_f_3_value_qin_jiushao_f_3_l1015_101559


namespace f_decreasing_interval_l1015_101523

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- State the theorem
theorem f_decreasing_interval :
  ∀ x ∈ Set.Ioo 0 2, ∀ y ∈ Set.Ioo 0 2, x < y → f x > f y :=
by sorry

end f_decreasing_interval_l1015_101523


namespace no_prime_between_30_40_congruent_7_mod_9_l1015_101510

theorem no_prime_between_30_40_congruent_7_mod_9 : ¬ ∃ (n : ℕ), Nat.Prime n ∧ 30 < n ∧ n < 40 ∧ n % 9 = 7 := by
  sorry

end no_prime_between_30_40_congruent_7_mod_9_l1015_101510


namespace prop_2_prop_4_l1015_101578

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define evenness for a function
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define symmetry about a vertical line
def SymmetricAbout (g : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, g (a + x) = g (a - x)

-- Proposition ②
theorem prop_2 (h : IsEven (fun x ↦ f (x + 2))) : SymmetricAbout f 2 := by sorry

-- Proposition ④
theorem prop_4 : SymmetricAbout (fun x ↦ f (x - 2)) 2 ∧ SymmetricAbout (fun x ↦ f (2 - x)) 2 := by sorry

end prop_2_prop_4_l1015_101578


namespace cubic_function_property_l1015_101571

/-- Given a cubic function f(x) = ax³ + bx + 1, prove that if f(m) = 6, then f(-m) = -4 -/
theorem cubic_function_property (a b m : ℝ) : 
  (fun x => a * x^3 + b * x + 1) m = 6 →
  (fun x => a * x^3 + b * x + 1) (-m) = -4 := by
sorry

end cubic_function_property_l1015_101571


namespace green_ball_count_l1015_101542

/-- Given a box of balls where the ratio of blue to green balls is 5:3 and there are 15 blue balls,
    prove that the number of green balls is 9. -/
theorem green_ball_count (blue : ℕ) (green : ℕ) (h1 : blue = 15) (h2 : blue * 3 = green * 5) : green = 9 := by
  sorry

end green_ball_count_l1015_101542


namespace spatial_quadrilateral_angle_sum_l1015_101533

-- Define a spatial quadrilateral
structure SpatialQuadrilateral :=
  (A B C D : Real)

-- State the theorem
theorem spatial_quadrilateral_angle_sum 
  (q : SpatialQuadrilateral) : q.A + q.B + q.C + q.D ≤ 360 := by
  sorry

end spatial_quadrilateral_angle_sum_l1015_101533


namespace melanie_plum_count_l1015_101575

/-- The number of plums Melanie picked -/
def melanie_picked : ℝ := 7.0

/-- The number of plums Sam gave to Melanie -/
def sam_gave : ℝ := 3.0

/-- The total number of plums Melanie has now -/
def total_plums : ℝ := melanie_picked + sam_gave

theorem melanie_plum_count : total_plums = 10.0 := by
  sorry

end melanie_plum_count_l1015_101575


namespace remaining_note_denomination_l1015_101501

theorem remaining_note_denomination 
  (total_amount : ℕ) 
  (total_notes : ℕ) 
  (fifty_notes : ℕ) 
  (h1 : total_amount = 10350)
  (h2 : total_notes = 36)
  (h3 : fifty_notes = 17) :
  (total_amount - 50 * fifty_notes) / (total_notes - fifty_notes) = 500 := by
  sorry

end remaining_note_denomination_l1015_101501


namespace ball_max_height_l1015_101552

/-- The height function of the ball -/
def h (t : ℝ) : ℝ := -20 * t^2 + 70 * t + 20

/-- The maximum height reached by the ball -/
def max_height : ℝ := 81.25

theorem ball_max_height :
  ∃ t : ℝ, h t = max_height ∧ ∀ s : ℝ, h s ≤ h t :=
by sorry

end ball_max_height_l1015_101552


namespace complex_equation_solution_l1015_101582

/-- Given a complex number z satisfying (z - 3i)(2 + i) = 5i, prove that z = 2 + 5i -/
theorem complex_equation_solution (z : ℂ) (h : (z - 3*Complex.I)*(2 + Complex.I) = 5*Complex.I) : 
  z = 2 + 5*Complex.I := by
  sorry

end complex_equation_solution_l1015_101582


namespace bet_winnings_l1015_101597

theorem bet_winnings (initial_amount : ℚ) : 
  initial_amount > 0 →
  initial_amount + 2 * initial_amount = 1200 →
  initial_amount = 400 := by
sorry

end bet_winnings_l1015_101597


namespace square_field_area_l1015_101534

/-- The area of a square field with side length 17 meters is 289 square meters. -/
theorem square_field_area :
  ∀ (side_length area : ℝ),
  side_length = 17 →
  area = side_length * side_length →
  area = 289 :=
by sorry

end square_field_area_l1015_101534


namespace correct_calculation_result_l1015_101576

theorem correct_calculation_result : ∃ x : ℕ, (40 + x = 52) ∧ (20 * x = 240) := by
  sorry

end correct_calculation_result_l1015_101576


namespace x_twelfth_power_is_one_l1015_101595

theorem x_twelfth_power_is_one (x : ℂ) (h : x + 1/x = -1) : x^12 = 1 := by
  sorry

end x_twelfth_power_is_one_l1015_101595


namespace absolute_value_equation_unique_solution_l1015_101574

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 5| = |x + 3| :=
by sorry

end absolute_value_equation_unique_solution_l1015_101574


namespace same_function_l1015_101555

theorem same_function (x : ℝ) : (x^3 + x) / (x^2 + 1) = x := by
  sorry

end same_function_l1015_101555


namespace max_value_of_a_max_value_is_tight_l1015_101513

theorem max_value_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2*x - a ≥ 0) → a ≤ -1 := by
  sorry

theorem max_value_is_tight : ∃ a : ℝ, a = -1 ∧ (∀ x : ℝ, x^2 - 2*x - a ≥ 0) := by
  sorry

end max_value_of_a_max_value_is_tight_l1015_101513


namespace square_sum_equals_five_l1015_101516

theorem square_sum_equals_five (a b c : ℝ) 
  (h : a + b + c + 3 = 2 * (Real.sqrt a + Real.sqrt (b + 1) + Real.sqrt (c - 1))) :
  a^2 + b^2 + c^2 = 5 := by
sorry

end square_sum_equals_five_l1015_101516


namespace assignment_schemes_with_girl_l1015_101529

theorem assignment_schemes_with_girl (num_boys num_girls : ℕ) 
  (h1 : num_boys = 4) 
  (h2 : num_girls = 3) 
  (total_people : ℕ := num_boys + num_girls) 
  (tasks : ℕ := 3) : 
  (total_people * (total_people - 1) * (total_people - 2)) - 
  (num_boys * (num_boys - 1) * (num_boys - 2)) = 186 := by
  sorry

#check assignment_schemes_with_girl

end assignment_schemes_with_girl_l1015_101529


namespace find_a_value_l1015_101515

theorem find_a_value (a : ℝ) : 
  (∀ x : ℝ, (x^2 - 4*x + a) + |x - 3| ≤ 5) ∧ 
  (∃ x : ℝ, x > 3 → (x^2 - 4*x + a) + |x - 3| > 5) →
  a = 8 := by
sorry

end find_a_value_l1015_101515


namespace triangle_area_l1015_101547

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    if b = 1, c = √3, and angle C = 2π/3, then the area of the triangle is √3/4 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = 1 → c = Real.sqrt 3 → C = 2 * Real.pi / 3 →
  (1/2) * b * c * Real.sin C = Real.sqrt 3 / 4 := by
sorry

end triangle_area_l1015_101547


namespace smallest_value_satisfying_equation_l1015_101556

theorem smallest_value_satisfying_equation :
  ∃ (x : ℝ), x = 3 ∧ ∀ (y : ℝ), (⌊y⌋ = 3 + 50 * (y - ⌊y⌋)) → y ≥ x :=
by sorry

end smallest_value_satisfying_equation_l1015_101556


namespace geometric_sequence_sum_l1015_101586

/-- A geometric sequence with sum of first n terms S_n -/
def GeometricSequence (S : ℕ → ℝ) : Prop :=
  ∃ (a r : ℝ), ∀ n : ℕ, S n = a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (S : ℕ → ℝ) :
  GeometricSequence S →
  S 5 = 10 →
  S 10 = 50 →
  S 15 = 210 := by
sorry

end geometric_sequence_sum_l1015_101586


namespace school_boys_count_l1015_101507

theorem school_boys_count (girls : ℕ) (difference : ℕ) (boys : ℕ) : 
  girls = 739 → difference = 402 → girls = boys + difference → boys = 337 := by
  sorry

end school_boys_count_l1015_101507


namespace solution_set_circle_plus_l1015_101520

/-- Custom operation ⊕ -/
def circle_plus (a b : ℝ) : ℝ := -2 * a + b

/-- Theorem stating the solution set of x ⊕ 4 > 0 -/
theorem solution_set_circle_plus (x : ℝ) :
  circle_plus x 4 > 0 ↔ x < 2 := by sorry

end solution_set_circle_plus_l1015_101520


namespace simplify_expression_l1015_101536

theorem simplify_expression (x : ℝ) : 105*x - 57*x + 8 = 48*x + 8 := by
  sorry

end simplify_expression_l1015_101536


namespace money_remaining_l1015_101531

/-- Given an initial amount of money and an amount spent, 
    the remaining amount is the difference between the two. -/
theorem money_remaining (initial spent : ℕ) : 
  initial = 16 → spent = 8 → initial - spent = 8 := by
  sorry

end money_remaining_l1015_101531


namespace collinear_vectors_dot_product_l1015_101599

/-- Given two collinear vectors m and n, prove their dot product is -17/2 -/
theorem collinear_vectors_dot_product :
  ∀ (k : ℝ),
  let m : ℝ × ℝ := (2*k - 1, k)
  let n : ℝ × ℝ := (4, 1)
  (∃ (t : ℝ), m = t • n) →  -- collinearity condition
  m.1 * n.1 + m.2 * n.2 = -17/2 :=
by
  sorry

end collinear_vectors_dot_product_l1015_101599


namespace sum_of_a_and_b_l1015_101546

-- Define a and b as real numbers
variable (a b : ℝ)

-- Theorem stating that the sum of a and b is equal to a + b
theorem sum_of_a_and_b : (a + b) = (a + b) := by sorry

end sum_of_a_and_b_l1015_101546


namespace min_value_of_expression_l1015_101528

theorem min_value_of_expression :
  (∀ x : ℝ, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2024 ≥ 2008) ∧
  (∃ x : ℝ, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2024 = 2008) := by
  sorry

end min_value_of_expression_l1015_101528


namespace chessboard_selection_divisibility_l1015_101588

theorem chessboard_selection_divisibility (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ k : ℕ, p! - p = p^5 * k := by
  sorry

end chessboard_selection_divisibility_l1015_101588


namespace givenVectorIsDirectionVector_l1015_101573

/-- A line in 2D space --/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A vector in 2D space --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The line x-3y+1=0 --/
def givenLine : Line2D :=
  { a := 1, b := -3, c := 1 }

/-- The vector (3,1) --/
def givenVector : Vector2D :=
  { x := 3, y := 1 }

/-- Definition: A vector is a direction vector of a line if it's parallel to the line --/
def isDirectionVector (v : Vector2D) (l : Line2D) : Prop :=
  v.x * l.b = -v.y * l.a

/-- Theorem: The vector (3,1) is a direction vector of the line x-3y+1=0 --/
theorem givenVectorIsDirectionVector : isDirectionVector givenVector givenLine := by
  sorry

end givenVectorIsDirectionVector_l1015_101573


namespace fibonacci_arithmetic_sequence_l1015_101590

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_arithmetic_sequence (n : ℕ) :
  n > 0 ∧ 
  n + (n + 3) + (n + 4) = 3000 ∧ 
  (fib n < fib (n + 3) ∧ fib (n + 3) < fib (n + 4)) ∧
  (fib (n + 4) - fib (n + 3) = fib (n + 3) - fib n) →
  n = 997 := by
sorry

end fibonacci_arithmetic_sequence_l1015_101590


namespace square_pyramid_sum_l1015_101584

/-- A square pyramid is a polyhedron with a square base and triangular faces meeting at an apex. -/
structure SquarePyramid where
  -- We don't need to define the internal structure, just the concept

/-- The number of faces in a square pyramid -/
def num_faces (sp : SquarePyramid) : ℕ := 5

/-- The number of edges in a square pyramid -/
def num_edges (sp : SquarePyramid) : ℕ := 8

/-- The number of vertices in a square pyramid -/
def num_vertices (sp : SquarePyramid) : ℕ := 5

/-- The sum of faces, edges, and vertices in a square pyramid is 18 -/
theorem square_pyramid_sum (sp : SquarePyramid) : 
  num_faces sp + num_edges sp + num_vertices sp = 18 := by
  sorry

end square_pyramid_sum_l1015_101584


namespace rectangle_to_square_perimeter_l1015_101506

/-- Given a rectangle that forms a square when its width is doubled and length is halved,
    this theorem relates the perimeter of the resulting square to the original rectangle's perimeter. -/
theorem rectangle_to_square_perimeter (w l P : ℝ) 
  (h1 : w > 0) 
  (h2 : l > 0)
  (h3 : 2 * w = l / 2)  -- Condition for forming a square
  (h4 : P = 4 * (2 * w)) -- Perimeter of the square
  : 2 * (w + l) = 5/4 * P := by
  sorry

end rectangle_to_square_perimeter_l1015_101506


namespace program_size_calculation_l1015_101532

/-- Calculates the size of a downloaded program given the download speed and time -/
theorem program_size_calculation (download_speed : ℝ) (download_time : ℝ) : 
  download_speed = 50 → download_time = 2 → 
  download_speed * download_time * 60 * 60 / 1024 = 351.5625 := by
  sorry

#check program_size_calculation

end program_size_calculation_l1015_101532


namespace quadrilateral_front_view_solids_l1015_101522

-- Define the possible solids
inductive Solid
| Cone
| Cylinder
| TriangularPyramid
| RectangularPrism

-- Define a property for having a quadrilateral front view
def has_quadrilateral_front_view (s : Solid) : Prop :=
  match s with
  | Solid.Cylinder => True
  | Solid.RectangularPrism => True
  | _ => False

-- Theorem statement
theorem quadrilateral_front_view_solids :
  ∀ s : Solid, has_quadrilateral_front_view s ↔ (s = Solid.Cylinder ∨ s = Solid.RectangularPrism) :=
by sorry

end quadrilateral_front_view_solids_l1015_101522


namespace lcm_14_21_45_l1015_101562

theorem lcm_14_21_45 : Nat.lcm 14 (Nat.lcm 21 45) = 630 := by sorry

end lcm_14_21_45_l1015_101562


namespace farmer_randy_planting_rate_l1015_101504

/-- Represents the cotton planting problem for Farmer Randy -/
structure CottonPlanting where
  total_acres : ℕ
  total_days : ℕ
  first_crew_tractors : ℕ
  first_crew_days : ℕ
  second_crew_tractors : ℕ
  second_crew_days : ℕ

/-- Calculates the acres per tractor per day needed to meet the planting deadline -/
def acres_per_tractor_per_day (cp : CottonPlanting) : ℚ :=
  cp.total_acres / (cp.first_crew_tractors * cp.first_crew_days + cp.second_crew_tractors * cp.second_crew_days)

/-- Theorem stating that for Farmer Randy's specific situation, each tractor needs to plant 68 acres per day -/
theorem farmer_randy_planting_rate :
  let cp : CottonPlanting := {
    total_acres := 1700,
    total_days := 5,
    first_crew_tractors := 2,
    first_crew_days := 2,
    second_crew_tractors := 7,
    second_crew_days := 3
  }
  acres_per_tractor_per_day cp = 68 := by
  sorry

end farmer_randy_planting_rate_l1015_101504


namespace sufficient_not_necessary_condition_necessary_not_sufficient_condition_l1015_101560

-- Statement B
theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x > 1 ∧ y > 1 → x + y > 2) ∧
  ¬(x + y > 2 → x > 1 ∧ y > 1) :=
sorry

-- Statement C
theorem necessary_not_sufficient_condition (a b : ℝ) :
  (a > b ∧ b > 0 → 1 / a < 1 / b) ∧
  ¬(1 / a < 1 / b → a > b ∧ b > 0) :=
sorry

end sufficient_not_necessary_condition_necessary_not_sufficient_condition_l1015_101560
