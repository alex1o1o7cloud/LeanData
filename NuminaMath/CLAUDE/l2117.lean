import Mathlib

namespace quadratic_root_range_l2117_211769

theorem quadratic_root_range (m : ℝ) : 
  (∃ x y : ℝ, x < -1 ∧ y > 1 ∧ 
   x^2 + (m-1)*x + m^2 - 2 = 0 ∧
   y^2 + (m-1)*y + m^2 - 2 = 0) →
  0 < m ∧ m < 1 :=
sorry

end quadratic_root_range_l2117_211769


namespace normal_distribution_probability_l2117_211703

-- Define a random variable X following N(0,1) distribution
def X : Real → Real := sorry

-- Define the probability measure for X
def P (s : Set Real) : Real := sorry

-- State the theorem
theorem normal_distribution_probability 
  (h1 : ∀ s, P s = ∫ x in s, X x)
  (h2 : P {x | x ≤ 1} = 0.8413) :
  P {x | -1 < x ∧ x < 0} = 0.3413 := by
  sorry

end normal_distribution_probability_l2117_211703


namespace num_employees_correct_l2117_211771

/-- The number of employees in an organization, excluding the manager -/
def num_employees : ℕ := 15

/-- The average monthly salary of employees, excluding the manager -/
def avg_salary : ℕ := 1800

/-- The increase in average salary when the manager's salary is added -/
def avg_increase : ℕ := 150

/-- The manager's monthly salary -/
def manager_salary : ℕ := 4200

/-- Theorem stating that the number of employees is correct given the conditions -/
theorem num_employees_correct :
  (avg_salary * num_employees + manager_salary) / (num_employees + 1) = avg_salary + avg_increase :=
by sorry

end num_employees_correct_l2117_211771


namespace common_tangents_count_l2117_211701

/-- Two circles in a plane -/
structure CirclePair :=
  (c1 c2 : Set ℝ × ℝ)
  (r1 r2 : ℝ)
  (h_unequal : r1 ≠ r2)

/-- The number of common tangents for a pair of circles -/
def num_common_tangents (cp : CirclePair) : ℕ := sorry

/-- Theorem stating that the number of common tangents for unequal circles is always 0, 1, 2, 3, or 4 -/
theorem common_tangents_count (cp : CirclePair) :
  num_common_tangents cp = 0 ∨
  num_common_tangents cp = 1 ∨
  num_common_tangents cp = 2 ∨
  num_common_tangents cp = 3 ∨
  num_common_tangents cp = 4 :=
sorry

end common_tangents_count_l2117_211701


namespace arithmetic_geometric_progression_l2117_211716

theorem arithmetic_geometric_progression (a b : ℝ) : 
  (1 - a = b - 1) ∧ (1 = a^2 * b^2) → 
  ((a = 1 ∧ b = 1) ∨ 
   (a = 1 + Real.sqrt 2 ∧ b = 1 - Real.sqrt 2) ∨ 
   (a = 1 - Real.sqrt 2 ∧ b = 1 + Real.sqrt 2)) := by
sorry

end arithmetic_geometric_progression_l2117_211716


namespace march_greatest_percent_difference_l2117_211704

/-- Represents the sales data for a group in a given month -/
structure SalesData where
  drummers : ℕ
  buglePlayers : ℕ

/-- Represents the fixed costs for each group -/
structure FixedCosts where
  drummers : ℕ
  buglePlayers : ℕ

/-- Calculates the net earnings for a group given sales and fixed cost -/
def netEarnings (sales : ℕ) (cost : ℕ) : ℤ :=
  (sales : ℤ) - (sales * cost : ℤ)

/-- Calculates the percent difference between two integer values -/
def percentDifference (a b : ℤ) : ℚ :=
  if b ≠ 0 then (a - b : ℚ) / (b.natAbs : ℚ) * 100
  else if a > 0 then 100
  else if a < 0 then -100
  else 0

/-- Theorem stating that March has the greatest percent difference in net earnings -/
theorem march_greatest_percent_difference 
  (sales : Fin 5 → SalesData) 
  (costs : FixedCosts) 
  (h_jan : sales 0 = ⟨150, 100⟩)
  (h_feb : sales 1 = ⟨200, 150⟩)
  (h_mar : sales 2 = ⟨180, 180⟩)
  (h_apr : sales 3 = ⟨120, 160⟩)
  (h_may : sales 4 = ⟨80, 120⟩)
  (h_costs : costs = ⟨1, 2⟩) :
  ∀ (i : Fin 5), i ≠ 2 → 
    (abs (percentDifference 
      (netEarnings (sales 2).drummers costs.drummers)
      (netEarnings (sales 2).buglePlayers costs.buglePlayers)) ≥
     abs (percentDifference
      (netEarnings (sales i).drummers costs.drummers)
      (netEarnings (sales i).buglePlayers costs.buglePlayers))) :=
by sorry

end march_greatest_percent_difference_l2117_211704


namespace apple_distribution_l2117_211775

theorem apple_distribution (total_apples : Nat) (total_bags : Nat) (x : Nat) : 
  total_apples = 109 →
  total_bags = 20 →
  (∃ (a b : Nat), a + b = total_bags ∧ a * x + b * 3 = total_apples) →
  (x = 10 ∨ x = 52) := by
  sorry

end apple_distribution_l2117_211775


namespace subset_condition_disjoint_condition_l2117_211779

-- Define the sets A and S
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 7}
def S (k : ℝ) : Set ℝ := {x : ℝ | k + 1 ≤ x ∧ x ≤ 2*k - 1}

-- Theorem for condition (1)
theorem subset_condition (k : ℝ) : A ⊇ S k ↔ k ≤ 4 := by sorry

-- Theorem for condition (2)
theorem disjoint_condition (k : ℝ) : A ∩ S k = ∅ ↔ k < 2 ∨ k > 6 := by sorry

end subset_condition_disjoint_condition_l2117_211779


namespace bryan_has_more_candies_l2117_211746

/-- Given that Bryan has 50 candies and Ben has 20 candies, 
    prove that Bryan has 30 more candies than Ben. -/
theorem bryan_has_more_candies :
  let bryan_candies : ℕ := 50
  let ben_candies : ℕ := 20
  bryan_candies - ben_candies = 30 := by
  sorry

end bryan_has_more_candies_l2117_211746


namespace peruvian_coffee_cost_l2117_211768

/-- Proves that the cost of Peruvian coffee beans per pound is approximately $2.29 given the specified conditions --/
theorem peruvian_coffee_cost (colombian_cost : ℝ) (total_weight : ℝ) (mix_price : ℝ) (colombian_weight : ℝ) :
  colombian_cost = 5.50 →
  total_weight = 40 →
  mix_price = 4.60 →
  colombian_weight = 28.8 →
  ∃ (peruvian_cost : ℝ), abs (peruvian_cost - 2.29) < 0.01 :=
by
  sorry

end peruvian_coffee_cost_l2117_211768


namespace sum_of_squares_l2117_211705

open BigOperators

/-- Given a sequence {aₙ} where the sum of the first n terms is 3ⁿ - 1,
    prove that the sum of squares of the first n terms is (1/2)(9ⁿ - 1) -/
theorem sum_of_squares (a : ℕ → ℝ) (n : ℕ) :
  (∀ k : ℕ, k ≤ n → ∑ i in Finset.range k, a i = 3^k - 1) →
  ∑ i in Finset.range n, (a i)^2 = (1/2) * (9^n - 1) :=
by sorry

end sum_of_squares_l2117_211705


namespace square_difference_401_399_l2117_211763

theorem square_difference_401_399 : 401^2 - 399^2 = 1600 := by
  sorry

end square_difference_401_399_l2117_211763


namespace tower_height_range_l2117_211776

theorem tower_height_range (h : ℝ) : 
  (¬(h ≥ 200)) ∧ (¬(h ≤ 150)) ∧ (¬(h ≤ 180)) → h ∈ Set.Ioo 180 200 := by
  sorry

end tower_height_range_l2117_211776


namespace jogger_speed_l2117_211781

theorem jogger_speed (usual_distance : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) :
  usual_distance = 30 →
  faster_speed = 16 →
  extra_distance = 10 →
  ∃ (usual_speed : ℝ),
    usual_speed * (usual_distance + extra_distance) / faster_speed = usual_distance ∧
    usual_speed = 12 := by
  sorry

end jogger_speed_l2117_211781


namespace find_m_l2117_211765

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (2 * x + 2) + 3

-- State the theorem
theorem find_m : ∃ (m : ℝ), f (1/2 * (2 * m + 2) - 1) = 2 * (2 * m + 2) + 3 ∧ f m = 6 ∧ m = -1/4 := by
  sorry

end find_m_l2117_211765


namespace special_ellipse_equation_l2117_211790

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The foci of the ellipse are on the x-axis -/
  foci_on_x_axis : Bool
  /-- The center of the ellipse is at the origin -/
  center_at_origin : Bool
  /-- The four vertices of a square with side length 2 are on the minor axis and coincide with the foci -/
  square_vertices_on_minor_axis : Bool
  /-- The side length of the square is 2 -/
  square_side_length : ℝ

/-- The standard equation of an ellipse -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem stating the standard equation of the special ellipse -/
theorem special_ellipse_equation (E : SpecialEllipse) (x y : ℝ) :
  E.foci_on_x_axis ∧
  E.center_at_origin ∧
  E.square_vertices_on_minor_axis ∧
  E.square_side_length = 2 →
  standard_equation 4 2 x y :=
sorry

end special_ellipse_equation_l2117_211790


namespace cosine_sum_equality_l2117_211741

theorem cosine_sum_equality : 
  Real.cos (47 * π / 180) * Real.cos (13 * π / 180) - 
  Real.cos (43 * π / 180) * Real.sin (167 * π / 180) = 1/2 := by
  sorry

end cosine_sum_equality_l2117_211741


namespace constant_term_expansion_l2117_211791

theorem constant_term_expansion (a : ℝ) : 
  (∃ (f : ℝ → ℝ), f x = (a * x^2 + 2 / Real.sqrt x)^5 ∧ 
   (∃ (c : ℝ), c = 80 ∧ (∀ x, f x = c + x * (f x - c) / x))) → a = 1 := by
  sorry

end constant_term_expansion_l2117_211791


namespace sam_reading_speed_l2117_211700

/-- Proves that given Dustin can read 75 pages in an hour and reads 34 more pages than Sam in 40 minutes, Sam can read 72 pages in an hour. -/
theorem sam_reading_speed (dustin_pages_per_hour : ℕ) (extra_pages : ℕ) : 
  dustin_pages_per_hour = 75 → 
  dustin_pages_per_hour * (40 : ℚ) / 60 - extra_pages = 
    (72 : ℚ) * (40 : ℚ) / 60 → 
  extra_pages = 34 →
  72 = 72 := by sorry

end sam_reading_speed_l2117_211700


namespace min_button_presses_l2117_211742

/-- Represents the state of the room --/
structure RoomState where
  armedMines : ℕ
  closedDoors : ℕ

/-- Represents the actions of pressing buttons --/
inductive ButtonPress
  | Red
  | Yellow
  | Green

/-- Defines the effect of pressing a button on the room state --/
def pressButton (state : RoomState) (button : ButtonPress) : RoomState :=
  match button with
  | ButtonPress.Red => ⟨state.armedMines + 1, state.closedDoors⟩
  | ButtonPress.Yellow => 
      if state.armedMines ≥ 2 
      then ⟨state.armedMines - 2, state.closedDoors + 1⟩ 
      else ⟨3, 3⟩  -- Reset condition
  | ButtonPress.Green => 
      if state.closedDoors ≥ 2 
      then ⟨state.armedMines, state.closedDoors - 2⟩ 
      else ⟨3, 3⟩  -- Reset condition

/-- Defines the initial state of the room --/
def initialState : RoomState := ⟨3, 3⟩

/-- Defines the goal state (all mines disarmed and all doors opened) --/
def goalState : RoomState := ⟨0, 0⟩

/-- Theorem stating that the minimum number of button presses to reach the goal state is 9 --/
theorem min_button_presses : 
  ∃ (sequence : List ButtonPress), 
    sequence.length = 9 ∧ 
    (sequence.foldl pressButton initialState = goalState) ∧
    (∀ (otherSequence : List ButtonPress), 
      otherSequence.foldl pressButton initialState = goalState → 
      otherSequence.length ≥ 9) := by
  sorry


end min_button_presses_l2117_211742


namespace complex_equality_l2117_211706

theorem complex_equality (a : ℝ) : 
  (Complex.re ((a - Complex.I) * (1 - Complex.I) * Complex.I) = 
   Complex.im ((a - Complex.I) * (1 - Complex.I) * Complex.I)) → 
  a = 0 := by
  sorry

end complex_equality_l2117_211706


namespace window_offer_savings_l2117_211783

/-- Represents the store's window offer -/
structure WindowOffer where
  regularPrice : ℕ
  buyQuantity : ℕ
  freeQuantity : ℕ

/-- Calculates the cost for a given number of windows under the offer -/
def costUnderOffer (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  let fullSets := windowsNeeded / (offer.buyQuantity + offer.freeQuantity)
  let remainder := windowsNeeded % (offer.buyQuantity + offer.freeQuantity)
  (fullSets * offer.buyQuantity + min remainder offer.buyQuantity) * offer.regularPrice

/-- Calculates the savings for a given number of windows under the offer -/
def savingsUnderOffer (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  windowsNeeded * offer.regularPrice - costUnderOffer offer windowsNeeded

/-- The main theorem to prove -/
theorem window_offer_savings (offer : WindowOffer) 
  (h1 : offer.regularPrice = 100)
  (h2 : offer.buyQuantity = 8)
  (h3 : offer.freeQuantity = 2)
  (dave_windows : ℕ) (doug_windows : ℕ)
  (h4 : dave_windows = 9)
  (h5 : doug_windows = 10) :
  savingsUnderOffer offer (dave_windows + doug_windows) - 
  (savingsUnderOffer offer dave_windows + savingsUnderOffer offer doug_windows) = 100 := by
  sorry

end window_offer_savings_l2117_211783


namespace pear_sales_l2117_211726

/-- Given a salesman who sold pears, prove that if he sold twice as much in the afternoon
    than in the morning, and 480 kilograms in total, then he sold 320 kilograms in the afternoon. -/
theorem pear_sales (morning_sales afternoon_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  morning_sales + afternoon_sales = 480 →
  afternoon_sales = 320 := by
  sorry

#check pear_sales

end pear_sales_l2117_211726


namespace sector_radius_gt_two_l2117_211785

theorem sector_radius_gt_two (R : ℝ) (l : ℝ) (h : R > 0) (h_l : l > 0) :
  (1/2 * l * R = 2 * R + l) → R > 2 := by
  sorry

end sector_radius_gt_two_l2117_211785


namespace seven_pencils_of_one_color_l2117_211743

/-- A box of colored pencils -/
structure ColoredPencilBox where
  pencils : Finset ℕ
  colors : ℕ → Finset ℕ
  total_pencils : pencils.card = 25
  color_property : ∀ s : Finset ℕ, s ⊆ pencils → s.card = 5 → ∃ c, (s ∩ colors c).card ≥ 2

/-- There are at least seven pencils of one color in the box -/
theorem seven_pencils_of_one_color (box : ColoredPencilBox) : 
  ∃ c, (box.pencils ∩ box.colors c).card ≥ 7 := by
  sorry


end seven_pencils_of_one_color_l2117_211743


namespace aaron_erasers_l2117_211723

/-- The number of erasers Aaron gives away -/
def erasers_given : ℕ := 34

/-- The number of erasers Aaron ends with -/
def erasers_left : ℕ := 47

/-- The initial number of erasers Aaron had -/
def initial_erasers : ℕ := erasers_given + erasers_left

theorem aaron_erasers : initial_erasers = 81 := by
  sorry

end aaron_erasers_l2117_211723


namespace smallest_n_for_trig_inequality_l2117_211736

theorem smallest_n_for_trig_inequality : 
  (∃ (n : ℕ), n > 0 ∧ ∀ (x : ℝ), Real.sin x ^ n + Real.cos x ^ n ≤ 2 / n) ∧ 
  (∀ (n : ℕ), n > 0 ∧ (∀ (x : ℝ), Real.sin x ^ n + Real.cos x ^ n ≤ 2 / n) → n ≥ 4) :=
by sorry

end smallest_n_for_trig_inequality_l2117_211736


namespace least_months_to_triple_l2117_211749

def interest_factor : ℝ := 1.06

def exceeds_triple (t : ℕ) : Prop :=
  interest_factor ^ t > 3

theorem least_months_to_triple : ∃ (t : ℕ), t = 20 ∧ exceeds_triple t ∧ ∀ (k : ℕ), k < t → ¬exceeds_triple k :=
sorry

end least_months_to_triple_l2117_211749


namespace fraction_always_defined_l2117_211725

theorem fraction_always_defined (x : ℝ) : (x^2 + 1) ≠ 0 := by
  sorry

#check fraction_always_defined

end fraction_always_defined_l2117_211725


namespace domain_of_g_l2117_211715

-- Define the function f with domain (-1, 0)
def f : Set ℝ := {x : ℝ | -1 < x ∧ x < 0}

-- Define the function g(x) = f(2x+1)
def g (x : ℝ) : Prop := (2 * x + 1) ∈ f

-- Theorem statement
theorem domain_of_g :
  {x : ℝ | g x} = {x : ℝ | -1 < x ∧ x < -1/2} :=
by sorry

end domain_of_g_l2117_211715


namespace equal_area_rectangles_l2117_211764

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem equal_area_rectangles (carol_rect jordan_rect : Rectangle) 
  (h1 : carol_rect.length = 5)
  (h2 : carol_rect.width = 24)
  (h3 : jordan_rect.length = 4)
  (h4 : area carol_rect = area jordan_rect) :
  jordan_rect.width = 30 := by
  sorry

end equal_area_rectangles_l2117_211764


namespace power_of_one_third_l2117_211799

theorem power_of_one_third (a b : ℕ) : 
  (2^a : ℕ) * (5^b : ℕ) = 200 → 
  (∀ k : ℕ, 2^k ∣ 200 → k ≤ a) →
  (∀ k : ℕ, 5^k ∣ 200 → k ≤ b) →
  (1/3 : ℚ)^(b - a) = 3 := by sorry

end power_of_one_third_l2117_211799


namespace hcl_formed_l2117_211778

-- Define the chemical reaction
structure Reaction where
  ch4 : ℕ
  cl2 : ℕ
  ccl4 : ℕ
  hcl : ℕ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.ch4 = 1 ∧ r.cl2 = 4 ∧ r.ccl4 = 1 ∧ r.hcl = 4

-- Define the given amounts of reactants
def given_reactants (r : Reaction) : Prop :=
  r.ch4 = 1 ∧ r.cl2 = 4

-- Theorem: Given the reactants and balanced equation, prove that 4 moles of HCl are formed
theorem hcl_formed (r : Reaction) 
  (h1 : balanced_equation r) 
  (h2 : given_reactants r) : 
  r.hcl = 4 := by
  sorry


end hcl_formed_l2117_211778


namespace chicken_count_l2117_211712

theorem chicken_count (coop run free_range : ℕ) : 
  coop = 14 →
  run = 2 * coop →
  free_range = 2 * run - 4 →
  free_range = 52 := by
sorry

end chicken_count_l2117_211712


namespace rectangle_area_problem_l2117_211757

theorem rectangle_area_problem (l w : ℚ) : 
  (l + 3) * (w - 1) = l * w ∧ (l - 3) * (w + 2) = l * w → l * w = -90 / 121 := by
  sorry

end rectangle_area_problem_l2117_211757


namespace geometric_sequence_sum_implies_a_eq_neg_one_l2117_211738

/-- A geometric sequence with sum of first n terms given by Sn = 4^n + a -/
def GeometricSequence (a : ℝ) := ℕ → ℝ

/-- Sum of first n terms of the geometric sequence -/
def SumFirstNTerms (seq : GeometricSequence a) (n : ℕ) : ℝ := 4^n + a

/-- The ratio between consecutive terms in a geometric sequence is constant -/
def IsGeometric (seq : GeometricSequence a) : Prop :=
  ∀ n : ℕ, seq (n + 2) / seq (n + 1) = seq (n + 1) / seq n

theorem geometric_sequence_sum_implies_a_eq_neg_one :
  ∀ a : ℝ, ∃ seq : GeometricSequence a,
    (∀ n : ℕ, SumFirstNTerms seq n = 4^n + a) →
    IsGeometric seq →
    a = -1 :=
sorry

end geometric_sequence_sum_implies_a_eq_neg_one_l2117_211738


namespace expected_percentage_proof_l2117_211709

/-- The probability of rain for a county on Monday -/
def prob_rain_monday : ℝ := 0.70

/-- The probability of rain for a county on Tuesday -/
def prob_rain_tuesday : ℝ := 0.80

/-- The probability of rain for a county on Wednesday -/
def prob_rain_wednesday : ℝ := 0.60

/-- The proportion of counties with chance of rain on Monday -/
def prop_counties_monday : ℝ := 0.60

/-- The proportion of counties with chance of rain on Tuesday -/
def prop_counties_tuesday : ℝ := 0.55

/-- The proportion of counties with chance of rain on Wednesday -/
def prop_counties_wednesday : ℝ := 0.40

/-- The proportion of counties that received rain on at least one day -/
def prop_counties_with_rain : ℝ := 0.80

/-- The expected percentage of counties that will receive rain on all three days -/
def expected_percentage : ℝ :=
  prop_counties_monday * prob_rain_monday *
  prop_counties_tuesday * prob_rain_tuesday *
  prop_counties_wednesday * prob_rain_wednesday *
  prop_counties_with_rain

theorem expected_percentage_proof :
  expected_percentage = 0.60 * 0.70 * 0.55 * 0.80 * 0.40 * 0.60 * 0.80 :=
by sorry

end expected_percentage_proof_l2117_211709


namespace three_lines_not_necessarily_coplanar_l2117_211754

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a line in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Define a plane in 3D space
structure Plane3D where
  point : Point3D
  normal : Point3D

-- Function to check if a line lies on a plane
def lineOnPlane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

-- Function to check if a point is on a line
def pointOnLine (pt : Point3D) (l : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem three_lines_not_necessarily_coplanar :
  ∃ (p : Point3D) (l1 l2 l3 : Line3D),
    pointOnLine p l1 ∧ pointOnLine p l2 ∧ pointOnLine p l3 ∧
    ¬∃ (plane : Plane3D), lineOnPlane l1 plane ∧ lineOnPlane l2 plane ∧ lineOnPlane l3 plane :=
sorry

end three_lines_not_necessarily_coplanar_l2117_211754


namespace imaginary_part_of_z_plus_reciprocal_l2117_211732

theorem imaginary_part_of_z_plus_reciprocal (z : ℂ) (h : z = 1 + I) :
  (z + z⁻¹).im = 1/2 := by sorry

end imaginary_part_of_z_plus_reciprocal_l2117_211732


namespace pascal_triangle_30th_row_28th_number_l2117_211719

/-- The number of elements in the row of Pascal's triangle we're considering -/
def row_length : ℕ := 30

/-- The position of the number we're looking for (1-indexed) -/
def target_position : ℕ := 28

/-- The value we're proving the target position contains -/
def target_value : ℕ := 406

theorem pascal_triangle_30th_row_28th_number :
  Nat.choose (row_length - 1) (target_position - 1) = target_value := by
  sorry

end pascal_triangle_30th_row_28th_number_l2117_211719


namespace park_expansion_area_ratio_l2117_211702

theorem park_expansion_area_ratio :
  ∀ s : ℝ, s > 0 →
  (s^2) / ((3*s)^2) = 1/9 := by
sorry

end park_expansion_area_ratio_l2117_211702


namespace exists_lt_function_sum_product_l2117_211793

/-- A function f: ℝ → ℝ is non-constant if there exist two distinct real numbers that map to different values. -/
def NonConstant (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f x ≠ f y

/-- For any non-constant function f: ℝ → ℝ, there exist real numbers x and y such that f(x+y) < f(xy). -/
theorem exists_lt_function_sum_product (f : ℝ → ℝ) (h : NonConstant f) :
  ∃ x y : ℝ, f (x + y) < f (x * y) := by
  sorry

end exists_lt_function_sum_product_l2117_211793


namespace quadratic_roots_expression_l2117_211747

theorem quadratic_roots_expression (r s : ℝ) : 
  (3 * r^2 - 5 * r - 8 = 0) → 
  (3 * s^2 - 5 * s - 8 = 0) → 
  r ≠ s →
  (9 * r^2 - 9 * s^2) / (r - s) = 15 := by
sorry

end quadratic_roots_expression_l2117_211747


namespace bus_travel_fraction_l2117_211750

/-- Given a journey with a total distance of 90 kilometers, where 1/5 of the distance is traveled by foot,
    12 kilometers are traveled by car, and the remaining distance is traveled by bus,
    prove that the fraction of the total distance traveled by bus is 2/3. -/
theorem bus_travel_fraction (total_distance : ℝ) (foot_fraction : ℝ) (car_distance : ℝ) :
  total_distance = 90 ∧ foot_fraction = 1/5 ∧ car_distance = 12 →
  (total_distance - foot_fraction * total_distance - car_distance) / total_distance = 2/3 := by
sorry

end bus_travel_fraction_l2117_211750


namespace polynomial_value_l2117_211710

/-- A polynomial with integer coefficients where each coefficient is between 0 and 3 (inclusive) -/
def IntPolynomial (n : ℕ) := { p : Polynomial ℤ // ∀ i, 0 ≤ p.coeff i ∧ p.coeff i < 4 }

/-- The theorem stating that if P(2) = 66, then P(3) = 111 for the given polynomial -/
theorem polynomial_value (n : ℕ) (P : IntPolynomial n) 
  (h : P.val.eval 2 = 66) : P.val.eval 3 = 111 := by
  sorry

end polynomial_value_l2117_211710


namespace geese_percentage_among_non_swans_l2117_211767

theorem geese_percentage_among_non_swans 
  (total : ℝ) 
  (geese_percent : ℝ) 
  (swan_percent : ℝ) 
  (heron_percent : ℝ) 
  (duck_percent : ℝ) 
  (h1 : geese_percent = 30)
  (h2 : swan_percent = 25)
  (h3 : heron_percent = 10)
  (h4 : duck_percent = 35)
  (h5 : geese_percent + swan_percent + heron_percent + duck_percent = 100) :
  (geese_percent / (100 - swan_percent)) * 100 = 40 := by
  sorry

end geese_percentage_among_non_swans_l2117_211767


namespace inequality_and_quadratic_solution_l2117_211786

theorem inequality_and_quadratic_solution :
  ∃ (m : ℤ), 1 < m ∧ m < 4 ∧
  ∀ (x : ℝ), 1 < x ∧ x < 4 →
  (x^2 - 2*x - m = 0 ↔ (x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3)) :=
by sorry

end inequality_and_quadratic_solution_l2117_211786


namespace perpendicular_lines_a_value_l2117_211787

/-- Given two lines a²x + y + 7 = 0 and x - 2ay + 1 = 0 that are perpendicular,
    prove that a = 0 or a = 2 -/
theorem perpendicular_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, a^2*x + y + 7 = 0 ∧ x - 2*a*y + 1 = 0 → 
    (a^2 : ℝ) * (1 / (-2*a)) = -1) →
  a = 0 ∨ a = 2 :=
by sorry

end perpendicular_lines_a_value_l2117_211787


namespace hyperbola_asymptotes_tangent_to_circle_l2117_211789

theorem hyperbola_asymptotes_tangent_to_circle (m : ℝ) : 
  m > 0 →
  (∀ x y : ℝ, y^2 - x^2 / m^2 = 1) →
  (∀ x y : ℝ, x^2 + y^2 - 4*y + 3 = 0) →
  (∃ x y : ℝ, y = m*x ∧ x^2 + y^2 - 4*y + 3 = 0) →
  m = Real.sqrt 3 / 3 :=
sorry

end hyperbola_asymptotes_tangent_to_circle_l2117_211789


namespace arithmetic_sequence_formula_l2117_211713

/-- An arithmetic sequence {aₙ} with a₅ = 9 and a₁ + a₇ = 14 has the general formula aₙ = 2n - 1 -/
theorem arithmetic_sequence_formula (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence
  a 5 = 9 →
  a 1 + a 7 = 14 →
  ∀ n : ℕ, a n = 2 * n - 1 := by
sorry

end arithmetic_sequence_formula_l2117_211713


namespace equation_solution_l2117_211788

theorem equation_solution (a b c : ℤ) :
  (∀ x : ℝ, (x - a)*(x - 5) + 2 = (x + b)*(x + c)) →
  a = 2 := by
sorry

end equation_solution_l2117_211788


namespace rectangle_toothpicks_l2117_211796

/-- The number of toothpicks needed to form a rectangle --/
def toothpicks_in_rectangle (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Theorem: A rectangle with length 20 and width 10 requires 430 toothpicks --/
theorem rectangle_toothpicks : toothpicks_in_rectangle 20 10 = 430 := by
  sorry

#eval toothpicks_in_rectangle 20 10

end rectangle_toothpicks_l2117_211796


namespace swimming_class_problem_l2117_211794

theorem swimming_class_problem (N : ℕ) (non_swimmers : ℕ) (signed_up : ℕ) (not_signed_up : ℕ) :
  N / 4 = non_swimmers →
  non_swimmers / 2 = signed_up →
  not_signed_up = 4 →
  signed_up + not_signed_up = non_swimmers →
  N = 32 ∧ N - non_swimmers = 24 := by
  sorry

end swimming_class_problem_l2117_211794


namespace probability_all_same_suit_l2117_211711

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of cards dealt to a player -/
def handSize : ℕ := 13

/-- The number of suits in a standard deck -/
def numSuits : ℕ := 4

/-- The number of cards in each suit -/
def cardsPerSuit : ℕ := deckSize / numSuits

theorem probability_all_same_suit :
  (numSuits : ℚ) / (deckSize.choose handSize : ℚ) =
  (numSuits : ℚ) / (Nat.choose deckSize handSize : ℚ) := by
  sorry

/-- The probability of all cards in a hand being from the same suit -/
def probabilitySameSuit : ℚ :=
  (numSuits : ℚ) / (Nat.choose deckSize handSize : ℚ)

end probability_all_same_suit_l2117_211711


namespace prob_at_least_four_same_l2117_211729

-- Define a die as having 6 sides
def die_sides : ℕ := 6

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the probability of all five dice showing the same number
def prob_all_same : ℚ := 1 / die_sides^(num_dice - 1)

-- Define the probability of exactly four dice showing the same number
def prob_four_same : ℚ := 
  (num_dice : ℚ) * (1 / die_sides^(num_dice - 2)) * ((die_sides - 1 : ℚ) / die_sides)

-- Theorem statement
theorem prob_at_least_four_same : 
  prob_all_same + prob_four_same = 13 / 648 :=
sorry

end prob_at_least_four_same_l2117_211729


namespace certain_number_proof_l2117_211753

theorem certain_number_proof (x : ℝ) : 
  (x + 40 + 60) / 3 = (10 + 80 + 15) / 3 + 5 → x = 20 := by
sorry

end certain_number_proof_l2117_211753


namespace nonreal_roots_product_l2117_211724

theorem nonreal_roots_product (x : ℂ) : 
  (x^4 - 6*x^3 + 15*x^2 - 20*x = 984) →
  (∃ a b : ℂ, (a ≠ b) ∧ (a.im ≠ 0) ∧ (b.im ≠ 0) ∧
   (x^4 - 6*x^3 + 15*x^2 - 20*x = 984 → (x = a ∨ x = b ∨ x.im = 0)) ∧
   (a * b = 4 - Real.sqrt 1000)) :=
sorry

end nonreal_roots_product_l2117_211724


namespace remaining_payment_l2117_211734

/-- Given a product with a 5% deposit of $50, prove that the remaining amount to be paid is $950 -/
theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (total_price : ℝ) : 
  deposit = 50 ∧ 
  deposit_percentage = 0.05 ∧ 
  deposit = deposit_percentage * total_price → 
  total_price - deposit = 950 := by
  sorry

end remaining_payment_l2117_211734


namespace x_value_l2117_211777

theorem x_value (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : x = 9 := by
  sorry

end x_value_l2117_211777


namespace joan_total_cents_l2117_211766

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The number of quarters Joan has -/
def num_quarters : ℕ := 12

/-- The number of dimes Joan has -/
def num_dimes : ℕ := 8

/-- The number of nickels Joan has -/
def num_nickels : ℕ := 15

/-- The number of pennies Joan has -/
def num_pennies : ℕ := 25

/-- The total value of Joan's coins in cents -/
theorem joan_total_cents : 
  num_quarters * quarter_value + 
  num_dimes * dime_value + 
  num_nickels * nickel_value + 
  num_pennies * penny_value = 480 := by
  sorry

end joan_total_cents_l2117_211766


namespace inequality_theorem_l2117_211733

theorem inequality_theorem (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  a + b + (1/2 : ℝ) ≥ Real.sqrt a + Real.sqrt b := by
sorry

end inequality_theorem_l2117_211733


namespace probability_bound_l2117_211722

def is_divisible_by_four (n : ℕ) : Prop := n % 4 = 0

def count_even (n : ℕ) : ℕ := n / 2

def count_divisible_by_four (n : ℕ) : ℕ := n / 4

def probability_three_integers (n : ℕ) : ℚ :=
  let total := n.choose 3
  let favorable := (count_even n).choose 3 + (count_divisible_by_four n) * ((n - count_divisible_by_four n).choose 2)
  favorable / total

theorem probability_bound (n : ℕ) (h : n = 2017) :
  1/8 < probability_three_integers n ∧ probability_three_integers n < 1/3 := by
  sorry

end probability_bound_l2117_211722


namespace product_equals_32_l2117_211752

theorem product_equals_32 : 
  (1 / 4 : ℚ) * 8 * (1 / 16 : ℚ) * 32 * (1 / 64 : ℚ) * 128 * (1 / 256 : ℚ) * 512 * (1 / 1024 : ℚ) * 2048 = 32 := by
  sorry

end product_equals_32_l2117_211752


namespace combined_discount_rate_l2117_211760

/-- Calculate the combined rate of discount for three items -/
theorem combined_discount_rate
  (bag_marked : ℝ) (shoes_marked : ℝ) (hat_marked : ℝ)
  (bag_discounted : ℝ) (shoes_discounted : ℝ) (hat_discounted : ℝ)
  (h_bag : bag_marked = 150 ∧ bag_discounted = 120)
  (h_shoes : shoes_marked = 100 ∧ shoes_discounted = 80)
  (h_hat : hat_marked = 50 ∧ hat_discounted = 40) :
  let total_marked := bag_marked + shoes_marked + hat_marked
  let total_discounted := bag_discounted + shoes_discounted + hat_discounted
  let discount_rate := (total_marked - total_discounted) / total_marked
  discount_rate = 0.2 := by
sorry

end combined_discount_rate_l2117_211760


namespace permutations_of_repeated_letters_l2117_211748

def phrase : String := "mathstest"

def repeated_letters (s : String) : List Char :=
  s.toList.filter (fun c => s.toList.count c > 1)

def unique_permutations (letters : List Char) : ℕ :=
  Nat.factorial letters.length / (Nat.factorial (letters.count 's') * Nat.factorial (letters.count 't'))

theorem permutations_of_repeated_letters :
  unique_permutations (repeated_letters phrase) = 10 := by
  sorry

end permutations_of_repeated_letters_l2117_211748


namespace rearrange_incongruent_sums_l2117_211751

/-- Given two lists of 2014 integers that are pairwise incongruent modulo 2014,
    there exists a permutation of the second list such that the pairwise sums
    of corresponding elements from both lists are incongruent modulo 4028. -/
theorem rearrange_incongruent_sums
  (x y : Fin 2014 → ℤ)
  (hx : ∀ i j, i ≠ j → x i % 2014 ≠ x j % 2014)
  (hy : ∀ i j, i ≠ j → y i % 2014 ≠ y j % 2014) :
  ∃ σ : Equiv.Perm (Fin 2014),
    ∀ i j, i ≠ j → (x i + y (σ i)) % 4028 ≠ (x j + y (σ j)) % 4028 :=
by sorry

end rearrange_incongruent_sums_l2117_211751


namespace isosceles_triangle_rectangle_equal_area_l2117_211755

/-- Given an isosceles triangle and a rectangle with equal areas,
    where the rectangle's length is twice its width and
    the triangle's base equals the rectangle's width,
    prove that the triangle's height is four times the rectangle's width. -/
theorem isosceles_triangle_rectangle_equal_area
  (w h : ℝ) -- w: width of rectangle, h: height of triangle
  (hw : w > 0) -- assume width is positive
  (triangle_area : ℝ → ℝ → ℝ) -- area function for triangle
  (rectangle_area : ℝ → ℝ → ℝ) -- area function for rectangle
  (h_triangle_area : triangle_area w h = 1/2 * w * h) -- definition of triangle area
  (h_rectangle_area : rectangle_area w (2*w) = 2 * w^2) -- definition of rectangle area
  (h_equal_area : triangle_area w h = rectangle_area w (2*w)) -- areas are equal
  : h = 4 * w :=
by sorry

end isosceles_triangle_rectangle_equal_area_l2117_211755


namespace min_xy_value_l2117_211728

theorem min_xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2*x + y + 6 = x*y) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2*a + b + 6 = a*b → x*y ≤ a*b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y + 6 = x*y ∧ x*y = 18 :=
sorry

end min_xy_value_l2117_211728


namespace coefficient_x2y2_in_expansion_l2117_211798

-- Define the binomial expansion function
def binomialExpand (n : ℕ) (x : ℝ) : ℝ := (1 + x) ^ n

-- Define the coefficient extraction function
def coefficientOf (term : ℕ × ℕ) (expansion : ℝ → ℝ → ℝ) : ℝ :=
  sorry -- Placeholder for the actual implementation

theorem coefficient_x2y2_in_expansion :
  coefficientOf (2, 2) (fun x y => binomialExpand 3 x * binomialExpand 4 y) = 18 := by
  sorry

#check coefficient_x2y2_in_expansion

end coefficient_x2y2_in_expansion_l2117_211798


namespace square_sum_bound_l2117_211721

theorem square_sum_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a^2 + b^2 ≥ 1/2 := by
  sorry

end square_sum_bound_l2117_211721


namespace sqrt_plus_reciprocal_inequality_l2117_211731

theorem sqrt_plus_reciprocal_inequality (x : ℝ) (hx : x > 0) : 
  Real.sqrt x + 1 / Real.sqrt x ≥ 2 ∧ 
  (Real.sqrt x + 1 / Real.sqrt x = 2 ↔ x = 1) := by
sorry

end sqrt_plus_reciprocal_inequality_l2117_211731


namespace product_inequality_l2117_211761

theorem product_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end product_inequality_l2117_211761


namespace class_average_l2117_211740

theorem class_average (total_students : ℕ) (high_scorers : ℕ) (zero_scorers : ℕ) (high_score : ℕ) (rest_average : ℚ) : 
  total_students = 25 →
  high_scorers = 5 →
  zero_scorers = 3 →
  high_score = 95 →
  rest_average = 45 →
  let rest_students := total_students - high_scorers - zero_scorers
  let total_score := (high_scorers * high_score + rest_students * rest_average)
  (total_score / total_students : ℚ) = 49.6 := by
sorry

end class_average_l2117_211740


namespace sea_glass_collection_l2117_211773

/-- Sea glass collection problem -/
theorem sea_glass_collection (blanche_green blanche_red rose_red rose_blue : ℕ) 
  (h1 : blanche_green = 12)
  (h2 : blanche_red = 3)
  (h3 : rose_red = 9)
  (h4 : rose_blue = 11)
  : 2 * (blanche_red + rose_red) + 3 * rose_blue = 57 := by
  sorry

end sea_glass_collection_l2117_211773


namespace undefined_values_l2117_211730

theorem undefined_values (a : ℝ) : 
  (a + 2) / (a^2 - 9) = 0/0 ↔ a = -3 ∨ a = 3 :=
by sorry

end undefined_values_l2117_211730


namespace largest_five_digit_divisible_by_97_l2117_211782

theorem largest_five_digit_divisible_by_97 : 
  ∀ n : ℕ, n ≤ 99999 ∧ n ≥ 10000 ∧ n % 97 = 0 → n ≤ 99930 :=
by
  sorry

end largest_five_digit_divisible_by_97_l2117_211782


namespace polar_to_rect_transformation_l2117_211744

/-- Given a point with rectangular coordinates (10, 3) and polar coordinates (r, θ),
    prove that the point with polar coordinates (r², 2θ) has rectangular coordinates (91, 60). -/
theorem polar_to_rect_transformation (r θ : ℝ) (h1 : r * Real.cos θ = 10) (h2 : r * Real.sin θ = 3) :
  (r^2 * Real.cos (2*θ), r^2 * Real.sin (2*θ)) = (91, 60) := by
  sorry


end polar_to_rect_transformation_l2117_211744


namespace mean_practice_hours_l2117_211714

def practice_hours : List ℕ := [1, 2, 3, 4, 5, 8, 10]
def student_counts : List ℕ := [4, 5, 3, 7, 2, 3, 1]

def total_hours : ℕ := (List.zip practice_hours student_counts).map (fun (h, c) => h * c) |>.sum
def total_students : ℕ := student_counts.sum

theorem mean_practice_hours :
  (total_hours : ℚ) / (total_students : ℚ) = 95 / 25 := by sorry

#eval (95 : ℚ) / 25  -- This should evaluate to 3.8

end mean_practice_hours_l2117_211714


namespace uncle_height_l2117_211780

/-- Represents the heights of James and his uncle before and after James' growth spurt -/
structure HeightScenario where
  james_initial : ℝ
  uncle : ℝ
  james_growth : ℝ
  height_diff_after : ℝ

/-- The conditions of the problem -/
def problem_conditions (h : HeightScenario) : Prop :=
  h.james_initial = (2/3) * h.uncle ∧
  h.james_growth = 10 ∧
  h.height_diff_after = 14 ∧
  h.uncle = (h.james_initial + h.james_growth + h.height_diff_after)

/-- The theorem stating that given the problem conditions, the uncle's height is 72 inches -/
theorem uncle_height (h : HeightScenario) : 
  problem_conditions h → h.uncle = 72 := by sorry

end uncle_height_l2117_211780


namespace same_last_four_digits_theorem_l2117_211735

theorem same_last_four_digits_theorem (N : ℕ) (a b c d : Fin 10) :
  (a ≠ 0) →
  (N % 10000 = a * 1000 + b * 100 + c * 10 + d) →
  ((N + 2) % 10000 = a * 1000 + b * 100 + c * 10 + d) →
  (a * 100 + b * 10 + c = 999) :=
by sorry

end same_last_four_digits_theorem_l2117_211735


namespace hoseok_wire_length_l2117_211708

/-- The length of wire Hoseok bought, given the conditions of the problem -/
def wire_length (triangle_side_length : ℝ) (remaining_wire : ℝ) : ℝ :=
  3 * triangle_side_length + remaining_wire

/-- Theorem stating that the length of wire Hoseok bought is 72 cm -/
theorem hoseok_wire_length :
  wire_length 19 15 = 72 := by
  sorry

end hoseok_wire_length_l2117_211708


namespace employee_survey_40_50_l2117_211758

/-- Represents the number of employees to be selected in a stratified sampling -/
def stratified_sample (total : ℕ) (group : ℕ) (sample_size : ℕ) : ℚ :=
  (group : ℚ) / (total : ℚ) * (sample_size : ℚ)

/-- Proves that the number of employees aged 40-50 to be selected is 12 -/
theorem employee_survey_40_50 :
  let total_employees : ℕ := 350
  let over_50 : ℕ := 70
  let under_40 : ℕ := 175
  let survey_size : ℕ := 40
  let employees_40_50 : ℕ := total_employees - over_50 - under_40
  stratified_sample total_employees employees_40_50 survey_size = 12 := by
  sorry

end employee_survey_40_50_l2117_211758


namespace fraction_addition_theorem_l2117_211727

theorem fraction_addition_theorem (a b c d x : ℚ) 
  (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : c ≠ d) 
  (h4 : (a + x) / (b + x) = c / d) : 
  x = (a * d - b * c) / (c - d) := by
  sorry

end fraction_addition_theorem_l2117_211727


namespace concatenatedDecimal_irrational_l2117_211784

/-- The infinite decimal formed by concatenating all natural numbers after the decimal point -/
noncomputable def concatenatedDecimal : ℝ :=
  sorry

/-- Function that generates the n-th digit of the concatenatedDecimal -/
def nthDigit (n : ℕ) : ℕ :=
  sorry

theorem concatenatedDecimal_irrational : Irrational concatenatedDecimal :=
  sorry

end concatenatedDecimal_irrational_l2117_211784


namespace class_3_1_fairy_tales_l2117_211772

theorem class_3_1_fairy_tales (andersen : ℕ) (grimm : ℕ) (both : ℕ) (total : ℕ) :
  andersen = 20 →
  grimm = 27 →
  both = 8 →
  total = 55 →
  andersen + grimm - both ≠ total :=
by
  sorry

end class_3_1_fairy_tales_l2117_211772


namespace quadratic_inequality_solution_l2117_211737

theorem quadratic_inequality_solution (a c : ℝ) : 
  (∀ x, (1/3 < x ∧ x < 1/2) ↔ a*x^2 + 5*x + c > 0) →
  a = -6 ∧ c = -1 := by
sorry

end quadratic_inequality_solution_l2117_211737


namespace x_ln_x_squared_necessary_not_sufficient_l2117_211718

theorem x_ln_x_squared_necessary_not_sufficient (x : ℝ) (h1 : 1 < x) (h2 : x < Real.exp 1) :
  (∀ x, x * Real.log x < 1 → x * (Real.log x)^2 < 1) ∧
  (∃ x, x * (Real.log x)^2 < 1 ∧ x * Real.log x ≥ 1) :=
by sorry

end x_ln_x_squared_necessary_not_sufficient_l2117_211718


namespace multiply_b_equals_five_l2117_211797

theorem multiply_b_equals_five (a b x : ℝ) 
  (h1 : 4 * a = x * b) 
  (h2 : a * b ≠ 0) 
  (h3 : (a / 5) / (b / 4) = 1) : 
  x = 5 := by sorry

end multiply_b_equals_five_l2117_211797


namespace base_b_not_divisible_by_five_l2117_211745

theorem base_b_not_divisible_by_five (b : ℤ) : b ∈ ({5, 6, 7, 9, 10} : Set ℤ) →
  (3 * b^3 - 2 * b^2 - b - 2) % 5 ≠ 0 ↔ b = 5 ∨ b = 10 := by
  sorry

end base_b_not_divisible_by_five_l2117_211745


namespace min_value_a_l2117_211756

theorem min_value_a (a : ℝ) : 
  (∀ x y : ℝ, |y + 4| - |y| ≤ 2^x + a / (2^x)) → 
  a ≥ 4 :=
by sorry

end min_value_a_l2117_211756


namespace unique_geometric_sequence_l2117_211774

/-- A geometric sequence with the given properties has a unique first term of 1/3 -/
theorem unique_geometric_sequence (a : ℝ) (a_n : ℕ → ℝ) : 
  a > 0 ∧ 
  (∀ n, a_n (n + 1) = a_n n * (a_n 2 / a_n 1)) ∧ 
  a_n 1 = a ∧
  (∃ q : ℝ, (a_n 1 + 1) * q = a_n 2 + 2 ∧ (a_n 2 + 2) * q = a_n 3 + 3) ∧
  (∃! q : ℝ, q ≠ 0 ∧ a * q^2 - 4 * a * q + 3 * a - 1 = 0) →
  a = 1/3 := by
sorry

end unique_geometric_sequence_l2117_211774


namespace roses_count_l2117_211717

def total_roses : ℕ := 500

def red_roses : ℕ := (total_roses * 5) / 8

def remaining_after_red : ℕ := total_roses - red_roses

def yellow_roses : ℕ := remaining_after_red / 8

def pink_roses : ℕ := (remaining_after_red * 2) / 8

def remaining_after_yellow_pink : ℕ := remaining_after_red - yellow_roses - pink_roses

def white_roses : ℕ := remaining_after_yellow_pink / 2

def purple_roses : ℕ := remaining_after_yellow_pink / 2

theorem roses_count : red_roses + white_roses + purple_roses = 430 := by
  sorry

end roses_count_l2117_211717


namespace ben_bought_three_cards_l2117_211707

/-- The number of cards Ben bought -/
def cards_bought : ℕ := 3

/-- The number of cards Tim had -/
def tim_cards : ℕ := 20

/-- The number of cards Ben initially had -/
def ben_initial_cards : ℕ := 37

theorem ben_bought_three_cards :
  (ben_initial_cards + cards_bought = 2 * tim_cards) ∧
  (cards_bought = 3) := by
  sorry

end ben_bought_three_cards_l2117_211707


namespace new_average_weight_l2117_211759

theorem new_average_weight 
  (A B C D E : ℝ)
  (h1 : (A + B + C) / 3 = 70)
  (h2 : (A + B + C + D) / 4 = 70)
  (h3 : E = D + 3)
  (h4 : A = 81) :
  (B + C + D + E) / 4 = 68 := by
sorry

end new_average_weight_l2117_211759


namespace count_six_digit_numbers_with_seven_l2117_211770

def digits : Finset ℕ := Finset.range 10

def multiset_count (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) k

def six_digit_numbers_with_seven : ℕ :=
  (multiset_count 9 5) +
  (multiset_count 9 4) +
  (multiset_count 9 3) +
  (multiset_count 9 2) +
  (multiset_count 9 1) +
  (multiset_count 9 0)

theorem count_six_digit_numbers_with_seven :
  six_digit_numbers_with_seven = 2002 := by sorry

end count_six_digit_numbers_with_seven_l2117_211770


namespace expression_evaluation_l2117_211720

/-- Convert a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The result of the given expression in base 10 -/
def result : ℚ :=
  (toBase10 [4, 5, 2] 8 : ℚ) / (toBase10 [3, 1] 3) +
  (toBase10 [3, 0, 2] 5 : ℚ) / (toBase10 [2, 2] 4)

theorem expression_evaluation :
  result = 33.966666666666665 := by sorry

end expression_evaluation_l2117_211720


namespace like_terms_exponent_value_l2117_211792

theorem like_terms_exponent_value (a b : ℝ) (n m : ℕ) :
  (∃ k : ℝ, k * a^(n+1) * b^n = -3 * a^(2*m) * b^3) →
  n^m = 9 := by
sorry

end like_terms_exponent_value_l2117_211792


namespace expression_factorization_l2117_211762

theorem expression_factorization (a b c : ℝ) : 
  2*(a+b)*(b+c)*(a+3*b+2*c) + 2*(b+c)*(c+a)*(b+3*c+2*a) + 
  2*(c+a)*(a+b)*(c+3*a+2*b) + 9*(a+b)*(b+c)*(c+a) = 
  (a + 3*b + 2*c)*(b + 3*c + 2*a)*(c + 3*a + 2*b) := by
sorry

end expression_factorization_l2117_211762


namespace largest_multiple_of_15_with_8_0_digits_l2117_211795

/-- A function that checks if all digits of a natural number are either 8 or 0 -/
def all_digits_eight_or_zero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 8 ∨ d = 0

/-- The theorem statement -/
theorem largest_multiple_of_15_with_8_0_digits :
  ∃ n : ℕ, n > 0 ∧ 15 ∣ n ∧ all_digits_eight_or_zero n ∧
  (∀ m : ℕ, m > n → ¬(15 ∣ m ∧ all_digits_eight_or_zero m)) ∧
  n / 15 = 592 := by
sorry

end largest_multiple_of_15_with_8_0_digits_l2117_211795


namespace circle_equation_proof_l2117_211739

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents the equation of a circle -/
structure CircleEquation where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Checks if a point satisfies the circle equation -/
def satisfiesCircleEquation (p : Point2D) (c : CircleEquation) : Prop :=
  p.x^2 + p.y^2 + c.D * p.x + c.E * p.y + c.F = 0

/-- Theorem: The equation x^2 + y^2 - 4x - 6y = 0 represents a circle passing through (0,0), (4,0), and (-1,1) -/
theorem circle_equation_proof :
  let c : CircleEquation := ⟨-4, -6, 0⟩
  satisfiesCircleEquation ⟨0, 0⟩ c ∧
  satisfiesCircleEquation ⟨4, 0⟩ c ∧
  satisfiesCircleEquation ⟨-1, 1⟩ c :=
by sorry

end circle_equation_proof_l2117_211739
