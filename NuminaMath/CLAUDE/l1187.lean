import Mathlib

namespace larger_number_proof_l1187_118707

theorem larger_number_proof (L S : ℕ) (hL : L > S) (h1 : L - S = 2500) (h2 : L = 6 * S + 15) : L = 2997 := by
  sorry

end larger_number_proof_l1187_118707


namespace chris_age_l1187_118744

def problem (a b c : ℚ) : Prop :=
  -- The average of Amy's, Ben's, and Chris's ages is 10
  (a + b + c) / 3 = 10 ∧
  -- Five years ago, Chris was twice the age that Amy is now
  c - 5 = 2 * a ∧
  -- In 4 years, Ben's age will be 3/4 of Amy's age at that time
  b + 4 = 3 / 4 * (a + 4)

theorem chris_age (a b c : ℚ) (h : problem a b c) : c = 263 / 11 := by
  sorry

end chris_age_l1187_118744


namespace candy_box_problem_l1187_118765

theorem candy_box_problem (n : ℕ) : n ≤ 200 →
  (n % 2 = 1 ∧ n % 3 = 1 ∧ n % 4 = 1 ∧ n % 6 = 1) →
  n % 11 = 0 →
  n = 121 := by
sorry

end candy_box_problem_l1187_118765


namespace starting_lineup_count_l1187_118723

/-- Represents a football team with its composition and eligibility rules. -/
structure FootballTeam where
  totalMembers : ℕ
  offensiveLinemenEligible : ℕ
  tightEndEligible : ℕ
  
/-- Calculates the number of ways to choose a starting lineup for a given football team. -/
def chooseStartingLineup (team : FootballTeam) : ℕ :=
  team.offensiveLinemenEligible * 
  team.tightEndEligible * 
  (team.totalMembers - 2) * 
  (team.totalMembers - 3) * 
  (team.totalMembers - 4)

/-- Theorem stating that for the given team composition, there are 5760 ways to choose a starting lineup. -/
theorem starting_lineup_count : 
  chooseStartingLineup ⟨12, 4, 2⟩ = 5760 := by
  sorry

end starting_lineup_count_l1187_118723


namespace two_flies_problem_l1187_118726

/-- Two flies crawling on a wall problem -/
theorem two_flies_problem (d v : ℝ) (h1 : d > 0) (h2 : v > 0) :
  let t1 := 2 * d / v
  let t2 := 5 * d / (2 * v)
  let avg_speed1 := 2 * d / t1
  let avg_speed2 := 2 * d / t2
  t1 < t2 ∧ avg_speed1 > avg_speed2 := by
  sorry

#check two_flies_problem

end two_flies_problem_l1187_118726


namespace ellipse_properties_l1187_118763

open Real

theorem ellipse_properties (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  let e := (Real.sqrt 6) / 3
  let d := (Real.sqrt 3) / 2
  let ellipse := fun (x y : ℝ) => x^2 / a^2 + y^2 / b^2 = 1
  let line := fun (k m x : ℝ) => k * x + m
  let A := (0, -b)
  let B := (a, 0)
  let distance_to_AB := d

  (e^2 * a^2 = a^2 - b^2) →
  (distance_to_AB^2 * (a^2 + b^2) = a^2 * b^2) →
  (∃ (C D : ℝ × ℝ) (k m : ℝ), k ≠ 0 ∧ m ≠ 0 ∧
    ellipse C.1 C.2 ∧ ellipse D.1 D.2 ∧
    C.2 = line k m C.1 ∧ D.2 = line k m D.1 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2) →
  (a^2 = 3 ∧ b^2 = 1 ∧
   (let k := (Real.sqrt 6) / 3
    let m := 3 / 2
    let area_ACD := 5 / 4
    ∃ (C D : ℝ × ℝ),
      ellipse C.1 C.2 ∧ ellipse D.1 D.2 ∧
      C.2 = line k m C.1 ∧ D.2 = line k m D.1 ∧
      (C.1 - A.1)^2 + (C.2 - A.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2 ∧
      area_ACD = 1/2 * abs ((C.1 - A.1) * (D.2 - A.2) - (C.2 - A.2) * (D.1 - A.1))))
  := by sorry

end ellipse_properties_l1187_118763


namespace circle_center_coordinates_l1187_118729

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  equation : (ℝ × ℝ) → Prop

-- Define our specific circle
def myCircle : Circle :=
  { center := (2, -1),
    equation := fun (x, y) => (x - 2)^2 + (y + 1)^2 = 3 }

-- Theorem statement
theorem circle_center_coordinates :
  myCircle.center = (2, -1) :=
by sorry

end circle_center_coordinates_l1187_118729


namespace ellipse_equation_constants_l1187_118735

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  passingPoint : Point
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ

/-- Check if a point satisfies the ellipse equation -/
def satisfiesEllipseEquation (e : Ellipse) (p : Point) : Prop :=
  (p.x - e.h)^2 / e.a^2 + (p.y - e.k)^2 / e.b^2 = 1

/-- The main theorem to prove -/
theorem ellipse_equation_constants : ∃ (e : Ellipse),
  e.focus1 = ⟨2, 2⟩ ∧
  e.focus2 = ⟨2, 6⟩ ∧
  e.passingPoint = ⟨14, -3⟩ ∧
  e.a > 0 ∧
  e.b > 0 ∧
  satisfiesEllipseEquation e e.passingPoint ∧
  e.a = 8 * Real.sqrt 3 ∧
  e.b = 14 ∧
  e.h = 2 ∧
  e.k = 4 := by
  sorry

end ellipse_equation_constants_l1187_118735


namespace jims_taxi_charge_l1187_118766

/-- Proves that the additional charge per 2/5 of a mile is $0.30 for Jim's taxi service -/
theorem jims_taxi_charge (initial_fee : ℚ) (total_charge : ℚ) (trip_distance : ℚ) :
  initial_fee = 2.25 →
  total_charge = 4.95 →
  trip_distance = 3.6 →
  (total_charge - initial_fee) / (trip_distance / (2/5)) = 0.30 := by
sorry

end jims_taxi_charge_l1187_118766


namespace parallel_vector_scalar_l1187_118736

/-- Given two 2D vectors a and b, find the scalar m such that m*a + b is parallel to a - 2*b -/
theorem parallel_vector_scalar (a b : ℝ × ℝ) (h1 : a = (2, 3)) (h2 : b = (-1, 2)) :
  ∃ m : ℝ, m * a.1 + b.1 = k * (a.1 - 2 * b.1) ∧ 
           m * a.2 + b.2 = k * (a.2 - 2 * b.2) ∧ 
           m = -1/2 :=
by sorry

end parallel_vector_scalar_l1187_118736


namespace strip_arrangement_area_l1187_118792

/-- Represents a rectangular paper strip -/
structure Strip where
  length : ℝ
  width : ℝ

/-- Calculates the area of a strip -/
def stripArea (s : Strip) : ℝ := s.length * s.width

/-- Calculates the overlap area between two perpendicular strips -/
def overlapArea (s1 s2 : Strip) : ℝ := s1.width * s2.width

/-- Represents the arrangement of strips on the table -/
structure StripArrangement where
  horizontalStrips : Fin 2 → Strip
  verticalStrips : Fin 2 → Strip

/-- Calculates the total area covered by the strips -/
def totalCoveredArea (arrangement : StripArrangement) : ℝ :=
  let totalStripArea := (Finset.sum (Finset.range 2) (λ i => stripArea (arrangement.horizontalStrips i))) +
                        (Finset.sum (Finset.range 2) (λ i => stripArea (arrangement.verticalStrips i)))
  let totalOverlapArea := Finset.sum (Finset.range 2) (λ i =>
                            Finset.sum (Finset.range 2) (λ j =>
                              overlapArea (arrangement.horizontalStrips i) (arrangement.verticalStrips j)))
  totalStripArea - totalOverlapArea

theorem strip_arrangement_area :
  ∀ (arrangement : StripArrangement),
    (∀ i : Fin 2, arrangement.horizontalStrips i = ⟨8, 1⟩) →
    (∀ i : Fin 2, arrangement.verticalStrips i = ⟨8, 1⟩) →
    totalCoveredArea arrangement = 28 := by
  sorry

end strip_arrangement_area_l1187_118792


namespace john_total_running_distance_l1187_118772

/-- The number of days from Monday to Saturday, inclusive -/
def days_ran : ℕ := 6

/-- The distance John ran each day in meters -/
def daily_distance : ℕ := 1700

/-- The total distance John ran before getting injured -/
def total_distance : ℕ := days_ran * daily_distance

/-- Theorem stating that the total distance John ran is 10200 meters -/
theorem john_total_running_distance :
  total_distance = 10200 := by sorry

end john_total_running_distance_l1187_118772


namespace odd_function_increasing_function_symmetry_more_than_two_roots_l1187_118748

-- Define the function f
def f (b c x : ℝ) : ℝ := x * abs x + b * x + c

-- Theorem statements
theorem odd_function (b : ℝ) :
  ∀ x, f b 0 x = -f b 0 (-x) := by sorry

theorem increasing_function (c : ℝ) :
  ∀ x y, x < y → f 0 c x < f 0 c y := by sorry

theorem symmetry (b c : ℝ) :
  ∀ x, f b c x - c = -(f b c (-x) - c) := by sorry

theorem more_than_two_roots :
  ∃ b c : ℝ, ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  f b c x₁ = 0 ∧ f b c x₂ = 0 ∧ f b c x₃ = 0 := by sorry

end odd_function_increasing_function_symmetry_more_than_two_roots_l1187_118748


namespace solution_set_inequality_l1187_118730

theorem solution_set_inequality (x : ℝ) : 
  (abs (x - 1) + abs (x - 2) ≥ 5) ↔ (x ≤ -1 ∨ x ≥ 4) := by
  sorry

end solution_set_inequality_l1187_118730


namespace michaels_brother_initial_money_l1187_118757

/-- Proof that Michael's brother initially had $17 -/
theorem michaels_brother_initial_money :
  ∀ (michael_money : ℕ) (brother_money_after : ℕ) (candy_cost : ℕ),
    michael_money = 42 →
    brother_money_after = 35 →
    candy_cost = 3 →
    ∃ (brother_initial_money : ℕ),
      brother_initial_money = 17 ∧
      brother_money_after = brother_initial_money + michael_money / 2 - candy_cost :=
by
  sorry

#check michaels_brother_initial_money

end michaels_brother_initial_money_l1187_118757


namespace shanghai_population_equality_l1187_118791

/-- The population of Shanghai in millions -/
def shanghai_population : ℝ := 16.3

/-- Scientific notation representation of Shanghai's population -/
def shanghai_population_scientific : ℝ := 1.63 * 10^7

/-- Theorem stating that the population of Shanghai expressed in millions 
    is equal to its representation in scientific notation -/
theorem shanghai_population_equality : 
  shanghai_population * 10^6 = shanghai_population_scientific := by
  sorry

end shanghai_population_equality_l1187_118791


namespace profit_percentage_l1187_118777

theorem profit_percentage (C P : ℝ) (h : (2/3) * P = 0.9 * C) :
  (P - C) / C = 0.35 := by
  sorry

end profit_percentage_l1187_118777


namespace common_divisors_84_90_l1187_118784

theorem common_divisors_84_90 : 
  (Finset.filter (λ x => x ∣ 84 ∧ x ∣ 90) (Finset.range (min 84 90 + 1))).card = 8 := by
  sorry

end common_divisors_84_90_l1187_118784


namespace ceiling_times_x_equals_156_l1187_118728

theorem ceiling_times_x_equals_156 :
  ∃ x : ℝ, x > 0 ∧ ⌈x⌉ = 13 ∧ ⌈x⌉ * x = 156 ∧ x = 12 := by
  sorry

end ceiling_times_x_equals_156_l1187_118728


namespace encode_decode_natural_numbers_l1187_118715

/-- Given a list of 100 natural numbers, we can encode them into a single number. -/
theorem encode_decode_natural_numbers :
  ∃ (encode : (Fin 100 → ℕ) → ℕ) (decode : ℕ → (Fin 100 → ℕ)),
    ∀ (nums : Fin 100 → ℕ), decode (encode nums) = nums :=
by sorry

end encode_decode_natural_numbers_l1187_118715


namespace sin_negative_390_degrees_l1187_118774

theorem sin_negative_390_degrees : 
  Real.sin ((-390 : ℝ) * π / 180) = -1/2 := by sorry

end sin_negative_390_degrees_l1187_118774


namespace pear_trees_count_total_trees_sum_apple_tree_exists_pear_tree_exists_l1187_118747

/-- The number of trees in the garden -/
def total_trees : ℕ := 46

/-- The number of pear trees in the garden -/
def pear_trees : ℕ := 27

/-- The number of apple trees in the garden -/
def apple_trees : ℕ := total_trees - pear_trees

/-- Theorem stating that the number of pear trees is 27 -/
theorem pear_trees_count : pear_trees = 27 := by sorry

/-- Theorem stating that the sum of apple and pear trees equals the total number of trees -/
theorem total_trees_sum : apple_trees + pear_trees = total_trees := by sorry

/-- Theorem stating that among any 28 trees, there is at least one apple tree -/
theorem apple_tree_exists (subset : Finset ℕ) (h : subset.card = 28) (h2 : subset ⊆ Finset.range total_trees) : 
  ∃ (tree : ℕ), tree ∈ subset ∧ tree < apple_trees := by sorry

/-- Theorem stating that among any 20 trees, there is at least one pear tree -/
theorem pear_tree_exists (subset : Finset ℕ) (h : subset.card = 20) (h2 : subset ⊆ Finset.range total_trees) : 
  ∃ (tree : ℕ), tree ∈ subset ∧ tree ≥ apple_trees := by sorry

end pear_trees_count_total_trees_sum_apple_tree_exists_pear_tree_exists_l1187_118747


namespace min_trips_required_l1187_118750

def trays_per_trip : ℕ := 9
def trays_table1 : ℕ := 17
def trays_table2 : ℕ := 55

def total_trays : ℕ := trays_table1 + trays_table2

theorem min_trips_required : (total_trays + trays_per_trip - 1) / trays_per_trip = 8 := by
  sorry

end min_trips_required_l1187_118750


namespace flour_recipe_total_l1187_118739

/-- The amount of flour required for Mary's cake recipe -/
def flour_recipe (flour_added : ℕ) (flour_to_add : ℕ) : ℕ :=
  flour_added + flour_to_add

/-- Theorem: The total amount of flour required by the recipe is equal to 
    the sum of the flour already added and the flour still to be added -/
theorem flour_recipe_total (flour_added flour_to_add : ℕ) :
  flour_recipe flour_added flour_to_add = flour_added + flour_to_add :=
by
  sorry

#eval flour_recipe 3 6  -- Should evaluate to 9

end flour_recipe_total_l1187_118739


namespace no_fishes_brought_home_l1187_118727

/-- Represents the number of fishes caught from a lake -/
def FishesCaught : Type := ℕ

/-- Represents whether all youngling fishes are returned -/
def ReturnedYounglings : Type := Bool

/-- Calculates the number of fishes brought home -/
def fishesBroughtHome (caught : List FishesCaught) (returned : ReturnedYounglings) : ℕ :=
  sorry

/-- Theorem: If all youngling fishes are returned, no fishes are brought home -/
theorem no_fishes_brought_home (caught : List FishesCaught) :
  fishesBroughtHome caught true = 0 := by
  sorry

end no_fishes_brought_home_l1187_118727


namespace circle_tangent_to_x_axis_at_origin_l1187_118711

-- Define a circle using its general equation
def Circle (D E F : ℝ) := {(x, y) : ℝ × ℝ | x^2 + y^2 + D*x + E*y + F = 0}

-- Define what it means for a circle to be tangent to the x-axis at the origin
def TangentToXAxisAtOrigin (c : Set (ℝ × ℝ)) : Prop :=
  (0, 0) ∈ c ∧ ∀ y ≠ 0, (0, y) ∉ c

-- Theorem statement
theorem circle_tangent_to_x_axis_at_origin (D E F : ℝ) :
  TangentToXAxisAtOrigin (Circle D E F) → D = 0 ∧ E ≠ 0 ∧ F ≠ 0 :=
by sorry

end circle_tangent_to_x_axis_at_origin_l1187_118711


namespace collinear_points_p_value_l1187_118732

/-- Three points are collinear if they lie on the same straight line -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

theorem collinear_points_p_value :
  ∀ p : ℝ, collinear 1 (-2) 3 4 6 (p/3) → p = 39 := by
  sorry

end collinear_points_p_value_l1187_118732


namespace crackers_distribution_l1187_118773

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) : 
  total_crackers = 36 →
  num_friends = 18 →
  total_crackers = num_friends * crackers_per_friend →
  crackers_per_friend = 2 := by
  sorry

end crackers_distribution_l1187_118773


namespace reflection_sum_l1187_118740

/-- Given a point A with coordinates (x, y), when reflected over the y-axis to point B,
    the sum of all coordinate values of A and B equals 2y. -/
theorem reflection_sum (x y : ℝ) : 
  let A := (x, y)
  let B := (-x, y)
  x + y + (-x) + y = 2 * y := by sorry

end reflection_sum_l1187_118740


namespace pens_purchased_l1187_118764

theorem pens_purchased (total_cost : ℝ) (num_pencils : ℕ) (pencil_price : ℝ) (pen_price : ℝ)
  (h1 : total_cost = 570)
  (h2 : num_pencils = 75)
  (h3 : pencil_price = 2)
  (h4 : pen_price = 14) :
  (total_cost - num_pencils * pencil_price) / pen_price = 30 :=
by
  sorry

end pens_purchased_l1187_118764


namespace perpendicular_slope_l1187_118770

/-- The slope of a line perpendicular to 4x - 5y = 10 is -5/4 -/
theorem perpendicular_slope (x y : ℝ) :
  (4 * x - 5 * y = 10) → 
  (slope_of_perpendicular_line : ℝ) = -5/4 := by
  sorry

end perpendicular_slope_l1187_118770


namespace hoseok_addition_l1187_118795

theorem hoseok_addition (x : ℤ) : x + 56 = 110 → x = 54 := by
  sorry

end hoseok_addition_l1187_118795


namespace sun_rise_position_l1187_118798

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the visibility of a circle above a line -/
inductive Visibility
  | Small
  | Half
  | Full

/-- Determines the positional relationship between a line and a circle -/
inductive PositionalRelationship
  | Tangent
  | Separate
  | ExternallyTangent
  | Intersecting

/-- 
  Given a circle and a line where only a small portion of the circle is visible above the line,
  prove that the positional relationship between the line and circle is intersecting.
-/
theorem sun_rise_position (c : Circle) (l : Line) (v : Visibility) :
  v = Visibility.Small → PositionalRelationship.Intersecting = 
    (let relationship := sorry -- Define the actual relationship based on c and l
     relationship) := by
  sorry


end sun_rise_position_l1187_118798


namespace simple_interest_principal_l1187_118756

/-- Simple interest calculation -/
theorem simple_interest_principal (interest : ℚ) (time : ℚ) (rate : ℚ) (principal : ℚ) :
  interest = principal * rate * time ∧
  interest = 10.92 ∧
  time = 6 ∧
  rate = 7 / 100 / 12 →
  principal = 26 := by sorry

end simple_interest_principal_l1187_118756


namespace probability_theorem_l1187_118780

def is_valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 60 ∧ 1 ≤ b ∧ b ≤ 60 ∧ a ≠ b

def satisfies_condition (a b : ℕ) : Prop :=
  ∃ (m : ℕ), a * b + a + b = 6 * m - 1

def total_pairs : ℕ := Nat.choose 60 2

def favorable_pairs : ℕ := total_pairs - Nat.choose 50 2

theorem probability_theorem :
  (favorable_pairs : ℚ) / total_pairs = 91 / 295 := by sorry

end probability_theorem_l1187_118780


namespace christopher_stroll_distance_l1187_118733

/-- Given Christopher's strolling speed and time, calculate the distance he strolled. -/
theorem christopher_stroll_distance 
  (speed : ℝ) 
  (time : ℝ) 
  (h1 : speed = 4) 
  (h2 : time = 1.25) : 
  speed * time = 5 := by
  sorry

end christopher_stroll_distance_l1187_118733


namespace sum_of_decimals_l1187_118753

theorem sum_of_decimals : 5.47 + 2.58 + 1.95 = 10.00 := by
  sorry

end sum_of_decimals_l1187_118753


namespace divisor_problem_l1187_118746

theorem divisor_problem (d : ℕ) (h : d > 0) :
  (∃ n : ℤ, n % d = 3 ∧ (2 * n) % d = 2) → d = 4 := by
  sorry

end divisor_problem_l1187_118746


namespace remainder_problem_l1187_118712

theorem remainder_problem (x : ℤ) : x % 62 = 7 → (x + 11) % 31 = 18 := by
  sorry

end remainder_problem_l1187_118712


namespace subset_implies_a_values_l1187_118767

theorem subset_implies_a_values (a : ℝ) : 
  let A : Set ℝ := {-1, 1}
  let B : Set ℝ := {x | a * x + 2 = 0}
  B ⊆ A → a ∈ ({-2, 0, 2} : Set ℝ) := by
sorry

end subset_implies_a_values_l1187_118767


namespace extreme_value_of_f_l1187_118751

-- Define the function f
def f (x a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the solution set condition
def solution_set (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, f x < 0 ↔ (x < m + 1 ∧ x ≠ m)

-- Theorem statement
theorem extreme_value_of_f (a b c m : ℝ) :
  solution_set (f · a b c) m →
  ∃ x, f x a b c = -4/27 ∧ ∀ y, f y a b c ≥ -4/27 :=
sorry

end extreme_value_of_f_l1187_118751


namespace probability_from_odds_l1187_118769

/-- Given odds in favor of an event as a ratio of two natural numbers -/
def OddsInFavor : Type := ℕ × ℕ

/-- Calculate the probability of an event given its odds in favor -/
def probability (odds : OddsInFavor) : ℚ :=
  let (favorable, unfavorable) := odds
  favorable / (favorable + unfavorable)

theorem probability_from_odds :
  let odds : OddsInFavor := (3, 5)
  probability odds = 3 / 8 := by
  sorry

end probability_from_odds_l1187_118769


namespace sqrt_calculations_l1187_118779

theorem sqrt_calculations : 
  (2 * Real.sqrt 12 + Real.sqrt 75 - 12 * Real.sqrt (1/3) = 5 * Real.sqrt 3) ∧
  (6 * Real.sqrt (8/5) / (2 * Real.sqrt 2) * (-1/2 * Real.sqrt 60) = -6 * Real.sqrt 3) :=
by sorry

end sqrt_calculations_l1187_118779


namespace expression_simplification_l1187_118781

theorem expression_simplification :
  1 / ((1 / (Real.sqrt 2 + 1)) + (2 / (Real.sqrt 3 - 1))) = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end expression_simplification_l1187_118781


namespace cubic_polynomial_uniqueness_l1187_118755

theorem cubic_polynomial_uniqueness (p q r : ℝ) (Q : ℝ → ℝ) :
  (p^3 + 4*p^2 + 6*p + 8 = 0) →
  (q^3 + 4*q^2 + 6*q + 8 = 0) →
  (r^3 + 4*r^2 + 6*r + 8 = 0) →
  (∃ a b c d : ℝ, ∀ x, Q x = a*x^3 + b*x^2 + c*x + d) →
  (Q p = q + r) →
  (Q q = p + r) →
  (Q r = p + q) →
  (Q (p + q + r) = -20) →
  (∀ x, Q x = 5/4*x^3 + 4*x^2 + 23/4*x + 6) :=
by sorry

end cubic_polynomial_uniqueness_l1187_118755


namespace set_equality_l1187_118782

-- Define the universal set U as ℝ
def U := ℝ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end set_equality_l1187_118782


namespace animal_biscuit_problem_l1187_118731

theorem animal_biscuit_problem :
  ∀ (dogs cats : ℕ),
  dogs + cats = 10 →
  6 * dogs + 5 * cats = 56 →
  dogs = 6 ∧ cats = 4 :=
by
  sorry

end animal_biscuit_problem_l1187_118731


namespace decreasing_quadratic_function_parameter_range_l1187_118722

/-- If f(x) = x^2 - 2(1-a)x + 2 is a decreasing function on (-∞, 4], then a ∈ (-∞, -3] -/
theorem decreasing_quadratic_function_parameter_range (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → (x^2 - 2*(1-a)*x + 2) > (y^2 - 2*(1-a)*y + 2)) →
  a ∈ Set.Iic (-3 : ℝ) :=
by sorry

end decreasing_quadratic_function_parameter_range_l1187_118722


namespace max_volume_triangular_prism_l1187_118761

/-- Represents a triangular prism with rectangular bases -/
structure TriangularPrism where
  l : ℝ  -- length of the base
  w : ℝ  -- width of the base
  h : ℝ  -- height of the prism

/-- The sum of the areas of two lateral faces and one base is 30 -/
def area_constraint (p : TriangularPrism) : Prop :=
  2 * p.h * p.l + p.l * p.w = 30

/-- The volume of the prism -/
def volume (p : TriangularPrism) : ℝ :=
  p.l * p.w * p.h

/-- Theorem: The maximum volume of the triangular prism is 112.5 -/
theorem max_volume_triangular_prism :
  ∃ (p : TriangularPrism), area_constraint p ∧
    (∀ (q : TriangularPrism), area_constraint q → volume q ≤ volume p) ∧
    volume p = 112.5 :=
sorry

end max_volume_triangular_prism_l1187_118761


namespace sharp_nested_30_l1187_118706

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.6 * N + 2

-- Theorem statement
theorem sharp_nested_30 : sharp (sharp (sharp (sharp 30))) = 8.24 := by sorry

end sharp_nested_30_l1187_118706


namespace number_divisibility_l1187_118790

theorem number_divisibility (x : ℝ) : x / 14.5 = 171 → x = 2479.5 := by
  sorry

end number_divisibility_l1187_118790


namespace binomial_product_simplification_l1187_118701

theorem binomial_product_simplification (x : ℝ) :
  (4 * x + 3) * (x - 6) = 4 * x^2 - 21 * x - 18 := by
  sorry

end binomial_product_simplification_l1187_118701


namespace lemon_pie_degree_measure_l1187_118719

theorem lemon_pie_degree_measure (total_students : ℕ) (chocolate_pref : ℕ) (apple_pref : ℕ) (blueberry_pref : ℕ) 
  (h_total : total_students = 45)
  (h_chocolate : chocolate_pref = 15)
  (h_apple : apple_pref = 10)
  (h_blueberry : blueberry_pref = 9)
  (h_remaining : (total_students - (chocolate_pref + apple_pref + blueberry_pref)) % 2 = 0) :
  let remaining := total_students - (chocolate_pref + apple_pref + blueberry_pref)
  let lemon_pref := remaining / 2
  ↑lemon_pref / ↑total_students * 360 = 48 := by
sorry

end lemon_pie_degree_measure_l1187_118719


namespace rooster_weight_unit_l1187_118702

/-- Represents units of mass measurement -/
inductive MassUnit
  | Kilogram
  | Ton
  | Gram

/-- The weight of a rooster in some unit -/
def roosterWeight : ℝ := 3

/-- Predicate to determine if a unit is appropriate for measuring rooster weight -/
def isAppropriateUnit (unit : MassUnit) : Prop :=
  match unit with
  | MassUnit.Kilogram => True
  | _ => False

theorem rooster_weight_unit :
  isAppropriateUnit MassUnit.Kilogram :=
sorry

end rooster_weight_unit_l1187_118702


namespace subcommittee_formation_count_l1187_118745

-- Define the number of Republicans and Democrats in the Senate committee
def totalRepublicans : ℕ := 10
def totalDemocrats : ℕ := 7

-- Define the number of Republicans and Democrats needed for the subcommittee
def subcommitteeRepublicans : ℕ := 4
def subcommitteeDemocrats : ℕ := 3

-- Theorem statement
theorem subcommittee_formation_count :
  (Nat.choose totalRepublicans subcommitteeRepublicans) *
  (Nat.choose totalDemocrats subcommitteeDemocrats) = 7350 := by
  sorry

end subcommittee_formation_count_l1187_118745


namespace composition_injective_implies_first_injective_l1187_118725

theorem composition_injective_implies_first_injective
  (f g : ℝ → ℝ) (h : Function.Injective (g ∘ f)) :
  Function.Injective f := by
  sorry

end composition_injective_implies_first_injective_l1187_118725


namespace range_of_m_l1187_118760

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) : 
  (f = λ x => 1 + Real.sin (2 * x)) →
  (g = λ x => 2 * (Real.cos x)^2 + m) →
  (∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), f x₀ ≥ g x₀) →
  m ≤ Real.sqrt 2 := by
sorry

end range_of_m_l1187_118760


namespace last_two_digits_product_l1187_118799

theorem last_two_digits_product (n : ℤ) : 
  (∃ k : ℤ, n = 6 * k) →  -- n is divisible by 6
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n % 100 = 10 * a + b ∧ a + b = 12) →  -- sum of last two digits is 12
  (∃ x y : ℕ, x < 10 ∧ y < 10 ∧ n % 100 = 10 * x + y ∧ x * y = 32 ∨ x * y = 36) :=
by sorry

end last_two_digits_product_l1187_118799


namespace garden_perimeter_l1187_118762

theorem garden_perimeter : 
  ∀ (length breadth perimeter : ℝ),
  length = 260 →
  breadth = 190 →
  perimeter = 2 * (length + breadth) →
  perimeter = 900 :=
by
  sorry

end garden_perimeter_l1187_118762


namespace problem_solution_l1187_118742

theorem problem_solution (x y : ℝ) (h : x^2 * (y^2 + 1) = 1) :
  (x * y < 1) ∧ (x^2 * y ≥ -1/2) ∧ (x^2 + x * y ≤ 5/4) := by
  sorry

end problem_solution_l1187_118742


namespace cos_squared_alpha_minus_pi_fourth_l1187_118758

theorem cos_squared_alpha_minus_pi_fourth (α : ℝ) 
  (h : Real.sin (2 * α) = 1 / 3) : 
  Real.cos (α - π / 4) ^ 2 = 2 / 3 := by
  sorry

end cos_squared_alpha_minus_pi_fourth_l1187_118758


namespace product_divisible_by_sum_implies_inequality_l1187_118717

theorem product_divisible_by_sum_implies_inequality (m n : ℕ) 
  (h : (m + n) ∣ (m * n)) : 
  m + n ≤ (Nat.gcd m n)^2 := by
sorry

end product_divisible_by_sum_implies_inequality_l1187_118717


namespace no_solution_exists_l1187_118718

theorem no_solution_exists (k : ℕ) (hk : k > 1) : ¬ ∃ n : ℕ+, ∀ i : ℕ, 1 ≤ i ∧ i ≤ k → n / i = 2 := by
  sorry

end no_solution_exists_l1187_118718


namespace factorial_ratio_l1187_118794

theorem factorial_ratio : (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end factorial_ratio_l1187_118794


namespace complex_exponential_form_l1187_118700

theorem complex_exponential_form (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ) (θ : ℝ), z = r * Complex.exp (Complex.I * θ) ∧ θ = π / 3 := by
  sorry

end complex_exponential_form_l1187_118700


namespace line_parameterization_l1187_118713

/-- Given a line y = 2x - 30 parameterized by (x, y) = (f(t), 20t - 10), 
    prove that f(t) = 10t + 10 -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t : ℝ, 2 * (f t) - 30 = 20 * t - 10) → 
  (∀ t : ℝ, f t = 10 * t + 10) := by
  sorry

end line_parameterization_l1187_118713


namespace shaded_area_grid_l1187_118703

/-- The area of the shaded region in a grid with specific properties -/
theorem shaded_area_grid (total_width total_height large_triangle_base large_triangle_height small_triangle_base small_triangle_height : ℝ) 
  (hw : total_width = 15)
  (hh : total_height = 5)
  (hlb : large_triangle_base = 15)
  (hlh : large_triangle_height = 3)
  (hsb : small_triangle_base = 3)
  (hsh : small_triangle_height = 4) :
  total_width * total_height - (large_triangle_base * large_triangle_height / 2) + (small_triangle_base * small_triangle_height / 2) = 58.5 := by
sorry

end shaded_area_grid_l1187_118703


namespace interest_years_calculation_l1187_118783

/-- Given simple interest, compound interest, and interest rate, calculate the number of years -/
theorem interest_years_calculation (simple_interest compound_interest : ℝ) (rate : ℝ) 
  (h1 : simple_interest = 600)
  (h2 : compound_interest = 609)
  (h3 : rate = 0.03)
  (h4 : simple_interest = rate * (compound_interest / (rate * ((1 + rate)^2 - 1))))
  (h5 : compound_interest = (simple_interest / (rate * 2)) * ((1 + rate)^2 - 1)) :
  ∃ (n : ℕ), n = 2 ∧ 
    simple_interest = (compound_interest / ((1 + rate)^n - 1)) * rate * n ∧
    compound_interest = (simple_interest / (rate * n)) * ((1 + rate)^n - 1) :=
sorry

end interest_years_calculation_l1187_118783


namespace modulus_of_complex_power_l1187_118708

theorem modulus_of_complex_power : Complex.abs ((2 : ℂ) + Complex.I) ^ 8 = 625 := by
  sorry

end modulus_of_complex_power_l1187_118708


namespace homework_duration_decrease_l1187_118754

/-- Represents the decrease in homework duration over two adjustments --/
theorem homework_duration_decrease (initial_duration final_duration : ℝ) (x : ℝ) :
  initial_duration = 120 →
  final_duration = 60 →
  initial_duration * (1 - x)^2 = final_duration :=
by sorry

end homework_duration_decrease_l1187_118754


namespace allan_bought_two_balloons_l1187_118704

/-- The number of balloons Allan bought at the park -/
def balloons_bought (allan_initial jake_brought total : ℕ) : ℕ :=
  total - (allan_initial + jake_brought)

/-- Theorem: Allan bought 2 balloons at the park -/
theorem allan_bought_two_balloons : balloons_bought 3 5 10 = 2 := by
  sorry

end allan_bought_two_balloons_l1187_118704


namespace union_of_sets_l1187_118776

theorem union_of_sets : 
  let A : Set ℕ := {0, 1, 3}
  let B : Set ℕ := {1, 2, 4}
  A ∪ B = {0, 1, 2, 3, 4} := by
sorry

end union_of_sets_l1187_118776


namespace lcm_of_five_numbers_l1187_118797

theorem lcm_of_five_numbers : Nat.lcm 53 (Nat.lcm 71 (Nat.lcm 89 (Nat.lcm 103 200))) = 788045800 := by
  sorry

end lcm_of_five_numbers_l1187_118797


namespace equation_two_complex_roots_l1187_118752

/-- The equation under consideration -/
def equation (x k : ℂ) : Prop :=
  x / (x + 2) + x / (x + 3) = k * x

/-- The equation has exactly two complex roots -/
def has_two_complex_roots (k : ℂ) : Prop :=
  ∃! (r₁ r₂ : ℂ), r₁ ≠ r₂ ∧ ∀ x, equation x k ↔ x = 0 ∨ x = r₁ ∨ x = r₂

/-- The main theorem stating the condition for the equation to have exactly two complex roots -/
theorem equation_two_complex_roots :
  ∀ k : ℂ, has_two_complex_roots k ↔ k = 2*I ∨ k = -2*I :=
sorry

end equation_two_complex_roots_l1187_118752


namespace lesser_number_problem_l1187_118775

theorem lesser_number_problem (x y : ℝ) (h_sum : x + y = 70) (h_product : x * y = 1050) : 
  min x y = 30 := by
sorry

end lesser_number_problem_l1187_118775


namespace df_length_l1187_118734

/-- Right triangle ABC with square ABDE and angle bisector intersection -/
structure RightTriangleWithSquare where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  -- Conditions
  right_angle : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0
  ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 21
  bc_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 28
  square_abde : (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0 ∧
                (E.1 - B.1) * (D.1 - B.1) + (E.2 - B.2) * (D.2 - B.2) = 0 ∧
                Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  f_on_de : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = (t * D.1 + (1 - t) * E.1, t * D.2 + (1 - t) * E.2)
  f_on_bisector : ∃ s : ℝ, s > 0 ∧ F = (C.1 + s * (A.1 + B.1 - 2 * C.1), C.2 + s * (A.2 + B.2 - 2 * C.2))

/-- The length of DF is 15 -/
theorem df_length (t : RightTriangleWithSquare) : 
  Real.sqrt ((t.D.1 - t.F.1)^2 + (t.D.2 - t.F.2)^2) = 15 := by
  sorry


end df_length_l1187_118734


namespace min_value_inverse_sum_l1187_118724

theorem min_value_inverse_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) :
  1 / a + 3 / b ≥ 16 ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ 1 / a + 3 / b = 16 := by
  sorry

end min_value_inverse_sum_l1187_118724


namespace polynomial_division_remainder_l1187_118716

theorem polynomial_division_remainder : ∃ q r : Polynomial ℚ, 
  (X : Polynomial ℚ)^4 = (X^2 + 4*X + 1) * q + r ∧ 
  r.degree < (X^2 + 4*X + 1).degree ∧ 
  r = -56*X - 15 := by sorry

end polynomial_division_remainder_l1187_118716


namespace abs_a_minus_3_l1187_118721

theorem abs_a_minus_3 (a : ℝ) (h : ∀ x : ℝ, (a - 2) * x > a - 2 ↔ x < 1) : 
  |a - 3| = 3 - a := by
  sorry

end abs_a_minus_3_l1187_118721


namespace old_clock_slow_l1187_118714

/-- Represents the number of minutes between hand overlaps on the old clock -/
def overlap_interval : ℕ := 66

/-- Represents the number of minutes in a standard day -/
def standard_day_minutes : ℕ := 24 * 60

/-- Represents the number of hand overlaps in a standard day -/
def overlaps_per_day : ℕ := 22

theorem old_clock_slow (old_clock_day : ℕ) 
  (h1 : old_clock_day = overlap_interval * overlaps_per_day) : 
  old_clock_day - standard_day_minutes = 12 := by
  sorry

end old_clock_slow_l1187_118714


namespace simplify_expression_l1187_118738

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b - 4) - 2*b^2 = 9*b^3 + 4*b^2 - 12*b := by
  sorry

end simplify_expression_l1187_118738


namespace set_relationships_l1187_118789

-- Define set A
def A : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- Define set B
def B : Set (ℝ × ℝ) := {p | p.2 = p.1^2 + 1}

-- Theorem statement
theorem set_relationships :
  (1 ∉ B) ∧ (2 ∈ A) := by
  sorry

end set_relationships_l1187_118789


namespace min_value_x_plus_2y_l1187_118709

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2 / x) + (1 / y) = 1) : x + 2*y ≥ 8 := by
  sorry

end min_value_x_plus_2y_l1187_118709


namespace star_op_and_comparison_l1187_118737

-- Define the * operation for non-zero integers
def star_op (a b : ℤ) : ℚ := (a⁻¹ : ℚ) + (b⁻¹ : ℚ)

-- Theorem statement
theorem star_op_and_comparison 
  (a b : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : a + b = 10) 
  (h4 : a * b = 24) : 
  star_op a b = 5 / 12 ∧ a * b > a + b := by
  sorry

end star_op_and_comparison_l1187_118737


namespace r_share_l1187_118720

/-- Given a total amount divided among three people P, Q, and R, with specified ratios,
    calculate R's share. -/
theorem r_share (total : ℕ) (p q r : ℕ) : 
  total = 1210 →
  5 * q = 4 * p →
  9 * r = 10 * q →
  r = 400 := by
  sorry


end r_share_l1187_118720


namespace eighth_group_selection_l1187_118741

/-- Represents a systematic sampling of students -/
structure StudentSampling where
  totalStudents : Nat
  sampledStudents : Nat
  groupSize : Nat
  numberGroups : Nat
  selectedFromThirdGroup : Nat

/-- Calculates the number of the student selected from a given group -/
def selectedFromGroup (s : StudentSampling) (group : Nat) : Nat :=
  s.selectedFromThirdGroup + (group - 3) * s.groupSize

/-- Theorem stating the number of the student selected from the eighth group -/
theorem eighth_group_selection (s : StudentSampling) 
  (h1 : s.totalStudents = 50)
  (h2 : s.sampledStudents = 10)
  (h3 : s.groupSize = 5)
  (h4 : s.numberGroups = 10)
  (h5 : s.selectedFromThirdGroup = 12) :
  selectedFromGroup s 8 = 37 := by
  sorry

end eighth_group_selection_l1187_118741


namespace project_completion_time_l1187_118787

theorem project_completion_time
  (a b c d e : ℝ)
  (h1 : 1/a + 1/b + 1/c + 1/d = 1/6)
  (h2 : 1/b + 1/c + 1/d + 1/e = 1/8)
  (h3 : 1/a + 1/e = 1/12)
  : e = 48 := by
  sorry

end project_completion_time_l1187_118787


namespace goldbach_138_max_diff_l1187_118743

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem goldbach_138_max_diff :
  ∃ (p q : ℕ), 
    is_prime p ∧ 
    is_prime q ∧ 
    p + q = 138 ∧ 
    p ≠ q ∧
    ∀ (r s : ℕ), is_prime r → is_prime s → r + s = 138 → r ≠ s → s - r ≤ 124 :=
sorry

end goldbach_138_max_diff_l1187_118743


namespace min_soldiers_in_formation_l1187_118788

/-- Represents a rectangular formation of soldiers -/
structure SoldierFormation where
  columns : ℕ
  rows : ℕ
  new_uniforms : ℕ

/-- Checks if the formation satisfies the given conditions -/
def is_valid_formation (f : SoldierFormation) : Prop :=
  f.new_uniforms = (f.columns * f.rows) / 100 ∧
  f.new_uniforms ≥ (3 * f.columns) / 10 ∧
  f.new_uniforms ≥ (2 * f.rows) / 5

/-- The theorem stating the minimum number of soldiers -/
theorem min_soldiers_in_formation :
  ∀ f : SoldierFormation, is_valid_formation f → f.columns * f.rows ≥ 1200 :=
by sorry

end min_soldiers_in_formation_l1187_118788


namespace tax_reduction_theorem_l1187_118710

/-- Proves that if a tax rate is reduced by X%, consumption increases by 12%,
    and the resulting revenue decreases by 14.88%, then X = 24. -/
theorem tax_reduction_theorem (X : ℝ) (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let new_tax_rate := T - (X / 100) * T
  let new_consumption := C + (12 / 100) * C
  let original_revenue := T * C
  let new_revenue := new_tax_rate * new_consumption
  new_revenue = (1 - 14.88 / 100) * original_revenue →
  X = 24 := by
  sorry

end tax_reduction_theorem_l1187_118710


namespace incorrect_inequality_l1187_118785

theorem incorrect_inequality (a b : ℝ) (h : a < b) : ¬(-4*a < -4*b) := by
  sorry

end incorrect_inequality_l1187_118785


namespace solution_set_ax_gt_b_l1187_118749

theorem solution_set_ax_gt_b (a b : ℝ) :
  let S := {x : ℝ | a * x > b}
  (a > 0 → S = {x : ℝ | x > b / a}) ∧
  (a < 0 → S = {x : ℝ | x < b / a}) ∧
  (a = 0 ∧ b ≥ 0 → S = ∅) ∧
  (a = 0 ∧ b < 0 → S = Set.univ) :=
by sorry

end solution_set_ax_gt_b_l1187_118749


namespace equation_solution_l1187_118786

theorem equation_solution : ∃! x : ℝ, 3 * x + 1 = x - 3 :=
  by
    use -2
    constructor
    · -- Prove that -2 satisfies the equation
      sorry
    · -- Prove uniqueness
      sorry

#check equation_solution

end equation_solution_l1187_118786


namespace pirate_treasure_probability_l1187_118759

theorem pirate_treasure_probability : 
  let n : ℕ := 8  -- Total number of islands
  let k : ℕ := 4  -- Number of islands with treasure
  let p_treasure : ℚ := 1/3  -- Probability of treasure and no traps
  let p_neither : ℚ := 1/2  -- Probability of neither treasure nor traps
  Nat.choose n k * p_treasure^k * p_neither^(n-k) = 35/648 := by
sorry

end pirate_treasure_probability_l1187_118759


namespace rock_skipping_theorem_l1187_118705

/-- The number of times Bob can skip a rock -/
def bob_skips : ℕ := 12

/-- The number of times Jim can skip a rock -/
def jim_skips : ℕ := 15

/-- The number of rocks each person skipped -/
def rocks_skipped : ℕ := 10

/-- The total number of skips for both Bob and Jim -/
def total_skips : ℕ := bob_skips * rocks_skipped + jim_skips * rocks_skipped

theorem rock_skipping_theorem : total_skips = 270 := by
  sorry

end rock_skipping_theorem_l1187_118705


namespace decimal_multiplication_l1187_118796

theorem decimal_multiplication : (0.2 : ℝ) * 0.8 = 0.16 := by
  sorry

end decimal_multiplication_l1187_118796


namespace remaining_water_volume_l1187_118793

/-- Given a cup with 2 liters of water, after pouring out x milliliters 4 times, 
    the remaining volume in milliliters is equal to 2000 - 4x. -/
theorem remaining_water_volume (x : ℝ) : 
  2000 - 4 * x = (2 : ℝ) * 1000 - 4 * x := by sorry

end remaining_water_volume_l1187_118793


namespace quadratic_roots_distinct_l1187_118771

theorem quadratic_roots_distinct (m : ℝ) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - m*x₁ - 1 = 0 ∧ x₂^2 - m*x₂ - 1 = 0 := by
  sorry

end quadratic_roots_distinct_l1187_118771


namespace sum_at_two_and_neg_two_l1187_118778

/-- A cubic polynomial with specific properties -/
structure CubicPolynomial (k : ℝ) where
  Q : ℝ → ℝ
  is_cubic : ∃ a b c : ℝ, ∀ x, Q x = a * x^3 + b * x^2 + c * x + k
  at_zero : Q 0 = k
  at_one : Q 1 = 3 * k
  at_neg_one : Q (-1) = 4 * k

/-- The sum of the polynomial evaluated at 2 and -2 equals 22k -/
theorem sum_at_two_and_neg_two (k : ℝ) (p : CubicPolynomial k) :
  p.Q 2 + p.Q (-2) = 22 * k := by sorry

end sum_at_two_and_neg_two_l1187_118778


namespace emma_numbers_l1187_118768

theorem emma_numbers : ∃ (a b : ℤ), 
  ((a = 17 ∧ b = 31) ∨ (a = 31 ∧ b = 17)) ∧ 3 * a + 4 * b = 161 := by
  sorry

end emma_numbers_l1187_118768
