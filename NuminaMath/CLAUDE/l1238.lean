import Mathlib

namespace sams_total_nickels_l1238_123843

/-- Sam's initial number of nickels -/
def initial_nickels : ℕ := 24

/-- Number of nickels Sam's dad gave him -/
def additional_nickels : ℕ := 39

/-- Theorem: Sam's total number of nickels after receiving more from his dad -/
theorem sams_total_nickels : initial_nickels + additional_nickels = 63 := by
  sorry

end sams_total_nickels_l1238_123843


namespace expression_equality_l1238_123881

theorem expression_equality : 
  Real.sqrt 4 * 4^(1/2 : ℝ) + 16 / 4 * 2 - Real.sqrt 8 = 12 - 2 * Real.sqrt 2 := by
  sorry

end expression_equality_l1238_123881


namespace grid_arrangements_eq_six_l1238_123834

/-- The number of ways to arrange 3 distinct elements in 3 positions -/
def arrangements_of_three : ℕ := 3 * 2 * 1

/-- The number of ways to arrange digits 1, 2, and 3 in three boxes of a 2x2 grid,
    with the fourth box fixed -/
def grid_arrangements : ℕ := arrangements_of_three

theorem grid_arrangements_eq_six :
  grid_arrangements = 6 := by sorry

end grid_arrangements_eq_six_l1238_123834


namespace graph_above_condition_l1238_123805

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2|
def g (x m : ℝ) : ℝ := -|x + 3| + m

-- State the theorem
theorem graph_above_condition (m : ℝ) :
  (∀ x : ℝ, f x > g x m) → m < 5 := by
  sorry

end graph_above_condition_l1238_123805


namespace segment_ratio_l1238_123806

/-- Given two line segments with equally spaced points, prove the ratio of their lengths -/
theorem segment_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ x : ℝ, x > 0 ∧ a = 9*x ∧ b = 99*x) → b / a = 11 := by
  sorry

end segment_ratio_l1238_123806


namespace sqrt_comparison_l1238_123809

theorem sqrt_comparison : Real.sqrt 7 - Real.sqrt 6 < Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end sqrt_comparison_l1238_123809


namespace tom_profit_is_8798_l1238_123810

/-- Calculates the profit for Tom's dough ball project -/
def dough_ball_profit (
  flour_needed : ℕ)  -- Amount of flour needed in pounds
  (flour_bag_size : ℕ)  -- Size of each flour bag in pounds
  (flour_bag_cost : ℕ)  -- Cost of each flour bag in dollars
  (salt_needed : ℕ)  -- Amount of salt needed in pounds
  (salt_cost_per_pound : ℚ)  -- Cost of salt per pound in dollars
  (promotion_cost : ℕ)  -- Cost of promotion in dollars
  (tickets_sold : ℕ)  -- Number of tickets sold
  (ticket_price : ℕ)  -- Price of each ticket in dollars
  : ℤ :=
  let flour_bags := (flour_needed + flour_bag_size - 1) / flour_bag_size
  let flour_cost := flour_bags * flour_bag_cost
  let salt_cost := (salt_needed : ℚ) * salt_cost_per_pound
  let total_cost := flour_cost + salt_cost.ceil + promotion_cost
  let revenue := tickets_sold * ticket_price
  revenue - total_cost

/-- Theorem stating that Tom's profit is $8798 -/
theorem tom_profit_is_8798 :
  dough_ball_profit 500 50 20 10 (2/10) 1000 500 20 = 8798 := by
  sorry

end tom_profit_is_8798_l1238_123810


namespace jar_water_problem_l1238_123883

theorem jar_water_problem (s l w : ℚ) : 
  s > 0 ∧ l > 0 ∧ s < l ∧ w > 0 →  -- s: smaller jar capacity, l: larger jar capacity, w: water amount
  w = (1/6) * s ∧ w = (1/5) * l → 
  (2 * w) / l = 2/5 := by sorry

end jar_water_problem_l1238_123883


namespace partition_6_3_l1238_123850

/-- Represents a partition of n into at most k parts -/
def Partition (n : ℕ) (k : ℕ) := { p : List ℕ // p.length ≤ k ∧ p.sum = n }

/-- Counts the number of partitions of n into at most k indistinguishable parts -/
def countPartitions (n : ℕ) (k : ℕ) : ℕ := sorry

theorem partition_6_3 : countPartitions 6 3 = 6 := by sorry

end partition_6_3_l1238_123850


namespace sin_alpha_value_l1238_123876

theorem sin_alpha_value (α : Real) :
  (∃ (t : Real), t * (Real.sin (30 * π / 180)) = Real.cos α ∧
                 t * (-Real.cos (30 * π / 180)) = Real.sin α) →
  Real.sin α = -Real.sqrt 3 / 2 := by
  sorry

end sin_alpha_value_l1238_123876


namespace multiplicative_inverse_152_mod_367_l1238_123882

theorem multiplicative_inverse_152_mod_367 :
  ∃ a : ℕ, a < 367 ∧ (152 * a) % 367 = 1 ∧ a = 248 := by
  sorry

end multiplicative_inverse_152_mod_367_l1238_123882


namespace furniture_purchase_price_l1238_123892

theorem furniture_purchase_price :
  let marked_price : ℝ := 132
  let discount_rate : ℝ := 0.1
  let profit_rate : ℝ := 0.1
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  ∃ (purchase_price : ℝ),
    selling_price - purchase_price = profit_rate * purchase_price ∧
    purchase_price = 108 :=
by sorry

end furniture_purchase_price_l1238_123892


namespace union_A_B_intersection_A_complement_B_l1238_123812

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 - 4*x ≤ 0}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | -1 ≤ x ∧ x ≤ 4} := by sorry

-- Theorem for A ∩ (ℝ \ B)
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x | -1 ≤ x ∧ x < 0} := by sorry

end union_A_B_intersection_A_complement_B_l1238_123812


namespace rhombus_diagonal_l1238_123803

theorem rhombus_diagonal (d1 : ℝ) (area : ℝ) (d2 : ℝ) : 
  d1 = 25 → area = 375 → area = (d1 * d2) / 2 → d2 = 30 := by
  sorry

end rhombus_diagonal_l1238_123803


namespace isabella_house_paintable_area_l1238_123816

-- Define the problem parameters
def num_bedrooms : ℕ := 4
def room_length : ℝ := 15
def room_width : ℝ := 12
def room_height : ℝ := 9
def unpaintable_area : ℝ := 80

-- Define the function to calculate the paintable area
def paintable_area : ℝ :=
  let total_wall_area := num_bedrooms * (2 * (room_length * room_height + room_width * room_height))
  total_wall_area - (num_bedrooms * unpaintable_area)

-- State the theorem
theorem isabella_house_paintable_area :
  paintable_area = 1624 := by sorry

end isabella_house_paintable_area_l1238_123816


namespace eighth_term_of_sequence_l1238_123811

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem eighth_term_of_sequence (a₁ d : ℝ) :
  arithmetic_sequence a₁ d 4 = 25 →
  arithmetic_sequence a₁ d 6 = 49 →
  arithmetic_sequence a₁ d 8 = 73 := by
sorry

end eighth_term_of_sequence_l1238_123811


namespace smallest_gcd_qr_l1238_123867

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 300) (h2 : Nat.gcd p r = 450) :
  ∃ (q' r' : ℕ+), Nat.gcd p q' = 300 ∧ Nat.gcd p r' = 450 ∧ Nat.gcd q' r' = 150 ∧
  ∀ (q'' r'' : ℕ+), Nat.gcd p q'' = 300 → Nat.gcd p r'' = 450 → Nat.gcd q'' r'' ≥ 150 :=
by sorry

end smallest_gcd_qr_l1238_123867


namespace johns_patients_l1238_123879

/-- The number of patients John sees each day at the first hospital -/
def patients_first_hospital : ℕ := sorry

/-- The number of patients John sees each day at the second hospital -/
def patients_second_hospital : ℕ := sorry

/-- The number of days John works per year -/
def work_days_per_year : ℕ := 5 * 50

/-- The total number of patients John treats in a year -/
def total_patients_per_year : ℕ := 11000

theorem johns_patients :
  patients_first_hospital = 20 ∧
  patients_second_hospital = (6 * patients_first_hospital) / 5 ∧
  work_days_per_year * (patients_first_hospital + patients_second_hospital) = total_patients_per_year :=
sorry

end johns_patients_l1238_123879


namespace green_peaches_count_l1238_123817

/-- The number of green peaches in a basket -/
def num_green_peaches : ℕ := sorry

/-- The number of red peaches in the basket -/
def num_red_peaches : ℕ := 6

/-- The total number of red and green peaches in the basket -/
def total_red_green_peaches : ℕ := 22

/-- Theorem stating that the number of green peaches is 16 -/
theorem green_peaches_count : num_green_peaches = 16 := by sorry

end green_peaches_count_l1238_123817


namespace square_area_ratio_l1238_123875

theorem square_area_ratio (s : ℝ) (h : s > 0) : 
  let d := s * Real.sqrt 2
  let side_larger := 2 * d
  (side_larger ^ 2) / (s ^ 2) = 8 := by sorry

end square_area_ratio_l1238_123875


namespace cristinas_pace_cristina_pace_is_3_l1238_123870

/-- Cristina's pace in a race with Nicky -/
theorem cristinas_pace (race_length : ℝ) (head_start : ℝ) (nickys_pace : ℝ) (catch_up_time : ℝ) : ℝ :=
  let cristinas_distance := nickys_pace * catch_up_time
  cristinas_distance / catch_up_time

/-- The main theorem stating Cristina's pace -/
theorem cristina_pace_is_3 : 
  cristinas_pace 300 12 3 30 = 3 := by
  sorry

end cristinas_pace_cristina_pace_is_3_l1238_123870


namespace triangle_perimeter_theorem_l1238_123841

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the perimeter function
def perimeter (t : Triangle) : ℝ := sorry

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Define the ray function
def ray (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define the intersection function
def intersect (s₁ s₂ : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

theorem triangle_perimeter_theorem (ABC : Triangle) (X Y M : ℝ × ℝ) :
  perimeter ABC = 4 →
  X ∈ ray ABC.A ABC.B →
  Y ∈ ray ABC.A ABC.C →
  distance ABC.A X = 1 →
  distance ABC.A Y = 1 →
  M ∈ intersect (Set.Icc ABC.B ABC.C) (Set.Icc X Y) →
  (perimeter ⟨ABC.A, ABC.B, M⟩ = 2 ∨ perimeter ⟨ABC.A, ABC.C, M⟩ = 2) := by
  sorry

end triangle_perimeter_theorem_l1238_123841


namespace a_correct_S_correct_l1238_123838

/-- The number of different selection methods for two non-empty subsets A and B of {1,2,3,...,n}
    where the smallest number in B is greater than the largest number in A. -/
def a (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 0
  else if n = 2 then 1
  else n * 2^(n-1) - 2^n + 1

/-- The sum of the first n terms of the sequence a_n. -/
def S (n : ℕ) : ℕ := (n - 3) * 2^n + n + 3

theorem a_correct (n : ℕ) : a n = n * 2^(n-1) - 2^n + 1 := by sorry

theorem S_correct (n : ℕ) : S n = (n - 3) * 2^n + n + 3 := by sorry

end a_correct_S_correct_l1238_123838


namespace triangle_inradius_l1238_123802

/-- The inradius of a triangle with side lengths 13, 84, and 85 is 6 -/
theorem triangle_inradius : ∀ (a b c r : ℝ),
  a = 13 ∧ b = 84 ∧ c = 85 →
  a^2 + b^2 = c^2 →
  (a + b + c) / 2 * r = (a * b) / 2 →
  r = 6 :=
by sorry

end triangle_inradius_l1238_123802


namespace remainder_theorem_l1238_123898

theorem remainder_theorem (z : ℕ) (hz : z > 0) (hz_div : 4 ∣ z) :
  (z * (2 + 4 + z) + 3) % 2 = 1 := by
sorry

end remainder_theorem_l1238_123898


namespace lions_volleyball_games_l1238_123800

theorem lions_volleyball_games 
  (initial_win_rate : Real) 
  (initial_win_rate_value : initial_win_rate = 0.60)
  (final_win_rate : Real) 
  (final_win_rate_value : final_win_rate = 0.55)
  (tournament_wins : Nat) 
  (tournament_wins_value : tournament_wins = 8)
  (tournament_losses : Nat) 
  (tournament_losses_value : tournament_losses = 4) :
  ∃ (total_games : Nat), 
    total_games = 40 ∧ 
    (initial_win_rate * (total_games - tournament_wins - tournament_losses) + tournament_wins) / total_games = final_win_rate :=
by sorry

end lions_volleyball_games_l1238_123800


namespace perpendicular_bisector_eq_l1238_123824

/-- The perpendicular bisector of a line segment MN is the set of all points
    equidistant from M and N. This theorem proves that for M(2, 4) and N(6, 2),
    the equation of the perpendicular bisector is 2x - y - 5 = 0. -/
theorem perpendicular_bisector_eq (x y : ℝ) :
  let M : ℝ × ℝ := (2, 4)
  let N : ℝ × ℝ := (6, 2)
  (x - M.1)^2 + (y - M.2)^2 = (x - N.1)^2 + (y - N.2)^2 ↔ 2*x - y - 5 = 0 := by
sorry

end perpendicular_bisector_eq_l1238_123824


namespace cos_alpha_value_l1238_123877

theorem cos_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.sin (π/6 - α) = -1/3) : Real.cos α = (2 * Real.sqrt 6 - 1) / 6 := by
  sorry

end cos_alpha_value_l1238_123877


namespace quadratic_sum_l1238_123854

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (8 * x^2 - 48 * x - 128 = a * (x + b)^2 + c) ∧ (a + b + c = -195) := by
  sorry

end quadratic_sum_l1238_123854


namespace coefficient_x90_is_minus_one_l1238_123889

/-- The sequence of factors in the polynomial expansion -/
def factors : List (ℕ → ℤ) := [
  (λ n => if n = 1 then -1 else 1),
  (λ n => if n = 2 then -2 else 1),
  (λ n => if n = 3 then -3 else 1),
  (λ n => if n = 4 then -4 else 1),
  (λ n => if n = 5 then -5 else 1),
  (λ n => if n = 6 then -6 else 1),
  (λ n => if n = 7 then -7 else 1),
  (λ n => if n = 8 then -8 else 1),
  (λ n => if n = 9 then -9 else 1),
  (λ n => if n = 10 then -10 else 1),
  (λ n => if n = 11 then -11 else 1),
  (λ n => if n = 13 then -13 else 1)
]

/-- The coefficient of x^90 in the expansion -/
def coefficient_x90 : ℤ := -1

/-- Theorem stating that the coefficient of x^90 in the expansion is -1 -/
theorem coefficient_x90_is_minus_one :
  coefficient_x90 = -1 := by sorry

end coefficient_x90_is_minus_one_l1238_123889


namespace decimal_representation_contradiction_l1238_123831

theorem decimal_representation_contradiction (m n : ℕ) (h_n : n ≤ 100) :
  ∃ (k : ℕ) (B : ℕ), (1000 * B : ℚ) / n = 167 + (k : ℚ) / 1000 → False :=
by sorry

end decimal_representation_contradiction_l1238_123831


namespace mike_baseball_cards_l1238_123840

theorem mike_baseball_cards (initial_cards new_cards : ℕ) 
  (h1 : initial_cards = 64) 
  (h2 : new_cards = 18) : 
  initial_cards + new_cards = 82 := by
  sorry

end mike_baseball_cards_l1238_123840


namespace cricket_collection_l1238_123880

theorem cricket_collection (initial_crickets : ℕ) (additional_crickets : ℕ) : 
  initial_crickets = 7 → additional_crickets = 4 → initial_crickets + additional_crickets = 11 :=
by
  sorry

end cricket_collection_l1238_123880


namespace smallest_n_for_integer_T_l1238_123893

-- Define T_n as a function of n
def T (n : ℕ) : ℚ := sorry

-- Define the property of being the smallest positive integer n for which T_n is an integer
def is_smallest_integer_T (n : ℕ) : Prop :=
  (T n).isInt ∧ ∀ m : ℕ, m < n → ¬(T m).isInt

-- Theorem statement
theorem smallest_n_for_integer_T :
  is_smallest_integer_T 504 := by sorry

end smallest_n_for_integer_T_l1238_123893


namespace square_difference_l1238_123855

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 5) : (x - y)^2 = 16 := by
  sorry

end square_difference_l1238_123855


namespace unique_point_for_equal_angles_l1238_123828

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The focus point -/
def F : ℝ × ℝ := (2, 0)

/-- Check if a line passes through a point -/
def line_passes_through (m b : ℝ) (point : ℝ × ℝ) : Prop :=
  point.2 = m * point.1 + b

/-- Check if a point is on a line passing through F -/
def point_on_line_through_F (x y : ℝ) : Prop :=
  ∃ m b : ℝ, line_passes_through m b (x, y) ∧ line_passes_through m b F

/-- The angle equality condition -/
def angle_equality (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ * (x₂ - p) + y₂ * (x₁ - p) = 0

/-- The main theorem -/
theorem unique_point_for_equal_angles :
  ∃! p : ℝ, p > 0 ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ,
    is_on_ellipse x₁ y₁ ∧ is_on_ellipse x₂ y₂ ∧
    point_on_line_through_F x₁ y₁ ∧ point_on_line_through_F x₂ y₂ →
    angle_equality p x₁ y₁ x₂ y₂) ∧
  p = 1.2 := by
  sorry

end unique_point_for_equal_angles_l1238_123828


namespace cloud9_total_amount_l1238_123829

/-- Represents the pricing structure for Cloud 9 Diving Company --/
structure PricingStructure where
  individualDiscount : Float
  groupDiscount5to10 : Float
  groupDiscount11to20 : Float
  groupDiscount21Plus : Float
  earlyBirdDiscount : Float

/-- Represents a booking group --/
structure BookingGroup where
  participants : Nat
  totalCost : Float
  earlyBird : Bool

/-- Represents the refund structure --/
structure RefundStructure where
  individualRefund1 : Float
  individualRefund2 : Float
  groupRefund : Float

/-- Calculate the total amount taken by Cloud 9 Diving Company --/
def calculateTotalAmount (
  pricing : PricingStructure
) (
  individualBookings : Float
) (
  individualEarlyBird : Float
) (
  groupA : BookingGroup
) (
  groupB : BookingGroup
) (
  groupC : BookingGroup
) (
  refunds : RefundStructure
) : Float :=
  sorry

/-- Theorem stating the total amount taken by Cloud 9 Diving Company --/
theorem cloud9_total_amount :
  let pricing : PricingStructure := {
    individualDiscount := 0
    groupDiscount5to10 := 0.05
    groupDiscount11to20 := 0.10
    groupDiscount21Plus := 0.15
    earlyBirdDiscount := 0.03
  }
  let groupA : BookingGroup := {
    participants := 8
    totalCost := 6000
    earlyBird := true
  }
  let groupB : BookingGroup := {
    participants := 15
    totalCost := 9000
    earlyBird := false
  }
  let groupC : BookingGroup := {
    participants := 22
    totalCost := 15000
    earlyBird := true
  }
  let refunds : RefundStructure := {
    individualRefund1 := 500 * 3
    individualRefund2 := 300 * 2
    groupRefund := 800
  }
  calculateTotalAmount pricing 12000 3000 groupA groupB groupC refunds = 35006.50 :=
sorry

end cloud9_total_amount_l1238_123829


namespace intersection_angle_cosine_l1238_123849

/-- The ellipse C₁ -/
def C₁ (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1

/-- The hyperbola C₂ -/
def C₂ (x y : ℝ) : Prop := x^2/3 - y^2 = 1

/-- The foci of both curves -/
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

/-- The cosine of the angle F₁PF₂ -/
noncomputable def cos_angle (P : ℝ × ℝ) : ℝ :=
  let d₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let d₂ := Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2)
  let d := Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2)
  (d₁^2 + d₂^2 - d^2) / (2 * d₁ * d₂)

theorem intersection_angle_cosine :
  ∀ (x y : ℝ), C₁ x y → C₂ x y → cos_angle (x, y) = 1/3 := by sorry

end intersection_angle_cosine_l1238_123849


namespace valid_placements_count_l1238_123884

/-- Represents a grid with rows and columns -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents the placement of crosses in a grid -/
structure CrossPlacement :=
  (grid : Grid)
  (num_crosses : ℕ)

/-- Counts the number of valid cross placements in a grid -/
def count_valid_placements (cp : CrossPlacement) : ℕ :=
  sorry

/-- The specific grid and cross placement for our problem -/
def our_problem : CrossPlacement :=
  { grid := { rows := 3, cols := 4 },
    num_crosses := 4 }

/-- Theorem stating that the number of valid placements for our problem is 36 -/
theorem valid_placements_count :
  count_valid_placements our_problem = 36 :=
sorry

end valid_placements_count_l1238_123884


namespace constant_q_value_l1238_123866

theorem constant_q_value (p q : ℝ) (h : ∀ x : ℝ, (x + 3) * (x + p) = x^2 + q*x + 12) : q = 7 := by
  sorry

end constant_q_value_l1238_123866


namespace tribe_leadership_proof_l1238_123872

def tribe_leadership_arrangements (n : ℕ) : ℕ :=
  n * (n - 1).choose 2 * (n - 3).choose 2 * (n - 5).choose 2

theorem tribe_leadership_proof (n : ℕ) (h : n = 11) :
  tribe_leadership_arrangements n = 207900 := by
  sorry

end tribe_leadership_proof_l1238_123872


namespace sum_bc_value_l1238_123863

theorem sum_bc_value (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 40)
  (h2 : a + d = 6)
  (h3 : a ≠ d) :
  b + c = 20 / 3 := by
  sorry

end sum_bc_value_l1238_123863


namespace sampled_classes_proportional_prob_at_least_one_grade12_prob_both_classes_selected_l1238_123859

/-- Represents the number of classes in each grade -/
structure GradeClasses where
  grade10 : Nat
  grade11 : Nat
  grade12 : Nat

/-- Represents the number of classes sampled from each grade -/
structure SampledClasses where
  grade10 : Nat
  grade11 : Nat
  grade12 : Nat

/-- The total number of classes across all grades -/
def totalClasses (gc : GradeClasses) : Nat :=
  gc.grade10 + gc.grade11 + gc.grade12

/-- The number of classes to be sampled -/
def totalSampled : Nat := 9

/-- The school's grade distribution -/
def schoolClasses : GradeClasses :=
  { grade10 := 16, grade11 := 12, grade12 := 8 }

/-- Theorem stating that the sampled classes are proportional to the total classes in each grade -/
theorem sampled_classes_proportional (sc : SampledClasses) :
    sc.grade10 * totalClasses schoolClasses = schoolClasses.grade10 * totalSampled ∧
    sc.grade11 * totalClasses schoolClasses = schoolClasses.grade11 * totalSampled ∧
    sc.grade12 * totalClasses schoolClasses = schoolClasses.grade12 * totalSampled :=
  sorry

/-- The probability of selecting at least one class from grade 12 -/
def probAtLeastOneGrade12 : Rat := 7 / 10

/-- Theorem stating the probability of selecting at least one class from grade 12 -/
theorem prob_at_least_one_grade12 (sc : SampledClasses) :
    probAtLeastOneGrade12 = 7 / 10 :=
  sorry

/-- The probability of selecting both class A from grade 11 and class B from grade 12 -/
def probBothClassesSelected : Rat := 1 / 6

/-- Theorem stating the probability of selecting both class A from grade 11 and class B from grade 12 -/
theorem prob_both_classes_selected (sc : SampledClasses) :
    probBothClassesSelected = 1 / 6 :=
  sorry

end sampled_classes_proportional_prob_at_least_one_grade12_prob_both_classes_selected_l1238_123859


namespace no_y_intercepts_l1238_123836

theorem no_y_intercepts (y : ℝ) : ¬ ∃ y, 3 * y^2 - 5 * y + 6 = 0 := by
  sorry

end no_y_intercepts_l1238_123836


namespace circle_equation_k_value_l1238_123851

theorem circle_equation_k_value (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 14*x + y^2 + 8*y - k = 0 ↔ (x + 7)^2 + (y + 4)^2 = 25) → 
  k = 40 := by
  sorry

end circle_equation_k_value_l1238_123851


namespace transformed_function_point_l1238_123885

def f : ℝ → ℝ := fun _ ↦ 8

theorem transformed_function_point (h : f 3 = 8) :
  let g : ℝ → ℝ := fun x ↦ 2 * (4 * f (3 * x - 1) + 6)
  g 2 = 38 ∧ 2 + 19 = 21 := by
  sorry

end transformed_function_point_l1238_123885


namespace prime_power_equation_l1238_123815

theorem prime_power_equation (p : ℕ) (x y : ℕ) (h_prime : Nat.Prime p) (h_eq : x^4 - y^4 = p * (x^3 - y^3)) :
  (x = 0 ∧ y = p) ∨ (x = p ∧ y = 0) := by
  sorry

end prime_power_equation_l1238_123815


namespace sarah_new_shirts_l1238_123887

/-- Given that Sarah initially had 9 shirts and now has a total of 17 shirts,
    prove that she bought 8 new shirts. -/
theorem sarah_new_shirts (initial_shirts : ℕ) (total_shirts : ℕ) (new_shirts : ℕ) :
  initial_shirts = 9 →
  total_shirts = 17 →
  new_shirts = total_shirts - initial_shirts →
  new_shirts = 8 := by
  sorry

end sarah_new_shirts_l1238_123887


namespace barChartMostEffective_l1238_123826

-- Define an enumeration for chart types
inductive ChartType
  | BarChart
  | LineChart
  | PieChart

-- Define a function to evaluate the effectiveness of a chart type for comparing quantities
def effectivenessForQuantityComparison (chart : ChartType) : Nat :=
  match chart with
  | ChartType.BarChart => 3
  | ChartType.LineChart => 2
  | ChartType.PieChart => 1

-- Theorem stating that BarChart is the most effective for quantity comparison
theorem barChartMostEffective :
  ∀ (chart : ChartType),
    chart ≠ ChartType.BarChart →
    effectivenessForQuantityComparison ChartType.BarChart > effectivenessForQuantityComparison chart :=
by
  sorry


end barChartMostEffective_l1238_123826


namespace expand_product_l1238_123899

theorem expand_product (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end expand_product_l1238_123899


namespace percentage_difference_l1238_123853

theorem percentage_difference (x y : ℝ) (h : x = y * (1 - 0.35)) :
  y = x * (1 + 0.35) := by
  sorry

end percentage_difference_l1238_123853


namespace gcd_binomial_coefficients_l1238_123837

theorem gcd_binomial_coefficients (n : ℕ+) :
  (∃ (p : ℕ) (k : ℕ+), Nat.Prime p ∧ n = p^(k:ℕ)) ↔
  (∃ m : ℕ, m > 1 ∧ (∀ i : Fin (n-1), m ∣ Nat.choose n i.val.succ)) := by
  sorry

end gcd_binomial_coefficients_l1238_123837


namespace ten_team_league_max_points_l1238_123894

/-- Represents a football league with n teams -/
structure FootballLeague where
  n : ℕ
  points_per_win : ℕ
  points_per_draw : ℕ
  points_per_loss : ℕ

/-- The maximum possible points for each team in the league -/
def max_points_per_team (league : FootballLeague) : ℕ :=
  sorry

/-- Theorem stating that in a 10-team league with 3 points for a win, 
    1 for a draw, and 0 for a loss, the maximum points per team is 13 -/
theorem ten_team_league_max_points :
  let league := FootballLeague.mk 10 3 1 0
  max_points_per_team league = 13 :=
sorry

end ten_team_league_max_points_l1238_123894


namespace unique_prime_sum_l1238_123825

/-- Given seven distinct positive integers not exceeding 7, prove that 179 is the only prime expressible as abcd + efg -/
theorem unique_prime_sum (a b c d e f g : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
  e ≠ f ∧ e ≠ g ∧
  f ≠ g ∧
  0 < a ∧ a ≤ 7 ∧
  0 < b ∧ b ≤ 7 ∧
  0 < c ∧ c ≤ 7 ∧
  0 < d ∧ d ≤ 7 ∧
  0 < e ∧ e ≤ 7 ∧
  0 < f ∧ f ≤ 7 ∧
  0 < g ∧ g ≤ 7 →
  (∃ p : ℕ, Nat.Prime p ∧ p = a * b * c * d + e * f * g) ↔ (a * b * c * d + e * f * g = 179) :=
by sorry

end unique_prime_sum_l1238_123825


namespace largest_c_value_l1238_123896

theorem largest_c_value (c : ℝ) : (3 * c + 7) * (c - 2) = 9 * c → c ≤ 2 := by
  sorry

end largest_c_value_l1238_123896


namespace exam_grade_logic_l1238_123842

theorem exam_grade_logic 
  (student : Type) 
  (received_A : student → Prop)
  (all_mc_correct : student → Prop)
  (problem_solving_90_percent : student → Prop)
  (h : ∀ s : student, (all_mc_correct s ∨ problem_solving_90_percent s) → received_A s) :
  ∀ s : student, ¬(received_A s) → (¬(all_mc_correct s) ∧ ¬(problem_solving_90_percent s)) :=
by sorry

end exam_grade_logic_l1238_123842


namespace apple_count_theorem_l1238_123813

def is_valid_apple_count (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ (n % 6 = 0)

theorem apple_count_theorem :
  ∀ n : ℕ, is_valid_apple_count n ↔ (n = 72 ∨ n = 78) :=
by sorry

end apple_count_theorem_l1238_123813


namespace cubic_poly_max_value_l1238_123857

/-- A cubic monic polynomial with roots a, b, and c -/
def cubic_monic_poly (a b c : ℝ) : ℝ → ℝ :=
  fun x => x^3 + (-(a + b + c)) * x^2 + (a*b + b*c + c*a) * x - a*b*c

/-- The theorem statement -/
theorem cubic_poly_max_value (a b c : ℝ) :
  let P := cubic_monic_poly a b c
  P 1 = 91 ∧ P (-1) = -121 →
  (∀ x y z : ℝ, (x*y + y*z + z*x) / (x*y*z + x + y + z) ≤ 7) ∧
  (∃ x y z : ℝ, (x*y + y*z + z*x) / (x*y*z + x + y + z) = 7) :=
by sorry

end cubic_poly_max_value_l1238_123857


namespace arithmetic_sequence_problem_l1238_123844

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- If a₂ + a₄ = 2 and S₂ + S₄ = 1 for an arithmetic sequence, then a₁₀ = 8 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 2 + seq.a 4 = 2) 
  (h2 : seq.S 2 + seq.S 4 = 1) : 
  seq.a 10 = 8 := by
  sorry

end arithmetic_sequence_problem_l1238_123844


namespace ac_unit_final_price_l1238_123830

/-- Calculates the final price of an air-conditioning unit after multiple price changes -/
def finalPrice (originalPrice : ℝ) (christmasDiscount : ℝ) (energyEfficientDiscount : ℝ)
                (priceIncrease : ℝ) (productionCostIncrease : ℝ) (seasonalDiscount : ℝ) : ℝ :=
  let price1 := originalPrice * (1 - christmasDiscount)
  let price2 := price1 * (1 - energyEfficientDiscount)
  let price3 := price2 * (1 + priceIncrease)
  let price4 := price3 * (1 + productionCostIncrease)
  price4 * (1 - seasonalDiscount)

/-- Theorem stating the final price of the air-conditioning unit -/
theorem ac_unit_final_price :
  finalPrice 470 0.16 0.07 0.12 0.08 0.10 = 399.71 := by
  sorry

end ac_unit_final_price_l1238_123830


namespace original_number_l1238_123845

theorem original_number (x : ℝ) : (x * 1.2 = 480) → x = 400 := by
  sorry

end original_number_l1238_123845


namespace parabola_intersection_slope_l1238_123827

/-- Parabola defined by y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Point M -/
def point_M : ℝ × ℝ := (-1, 2)

/-- Line passing through focus with slope k -/
def line (k x : ℝ) : ℝ := k * (x - focus.1)

/-- Intersection points of the line and parabola -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola p.1 p.2 ∧ p.2 = line k p.1}

/-- Angle AMB is 90 degrees -/
def right_angle (A B : ℝ × ℝ) : Prop :=
  (A.2 - point_M.2) * (B.2 - point_M.2) = -(A.1 - point_M.1) * (B.1 - point_M.1)

theorem parabola_intersection_slope :
  ∀ k : ℝ, ∃ A B : ℝ × ℝ,
    A ∈ intersection_points k ∧
    B ∈ intersection_points k ∧
    A ≠ B ∧
    right_angle A B →
    k = 1 := by sorry

end parabola_intersection_slope_l1238_123827


namespace p_sufficient_not_necessary_for_q_l1238_123871

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, x^2 - x - 20 > 0 → 1 - x^2 < 0) ∧
  (∃ x : ℝ, 1 - x^2 < 0 ∧ ¬(x^2 - x - 20 > 0)) := by
  sorry

end p_sufficient_not_necessary_for_q_l1238_123871


namespace quadratic_inequality_no_solution_l1238_123869

theorem quadratic_inequality_no_solution 
  (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c < 0) ↔ a < 0 ∧ b^2 - 4*a*c < 0 :=
sorry

end quadratic_inequality_no_solution_l1238_123869


namespace stratified_sampling_expectation_l1238_123890

theorem stratified_sampling_expectation
  (total_population : ℕ)
  (sample_size : ℕ)
  (category_size : ℕ)
  (h1 : total_population = 100)
  (h2 : sample_size = 20)
  (h3 : category_size = 30) :
  (sample_size : ℚ) / total_population * category_size = 6 := by
sorry

end stratified_sampling_expectation_l1238_123890


namespace negation_of_universal_proposition_l1238_123864

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by sorry

end negation_of_universal_proposition_l1238_123864


namespace absolute_value_inequality_l1238_123823

theorem absolute_value_inequality (x a : ℝ) 
  (h1 : |x - 4| + |x - 3| < a) 
  (h2 : a > 0) : 
  a > 1 := by
  sorry

end absolute_value_inequality_l1238_123823


namespace product_inequality_l1238_123839

theorem product_inequality (a b x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1)
  (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hx₃ : 0 < x₃) (hx₄ : 0 < x₄) (hx₅ : 0 < x₅)
  (hx : x₁ * x₂ * x₃ * x₄ * x₅ = 1) :
  (a*x₁ + b) * (a*x₂ + b) * (a*x₃ + b) * (a*x₄ + b) * (a*x₅ + b) ≥ 1 := by
sorry

end product_inequality_l1238_123839


namespace smallest_possible_N_l1238_123818

def is_valid_arrangement (a b c d e f : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
  a + b + c + d + e + f = 2520 ∧
  a ≥ 5 ∧ b ≥ 5 ∧ c ≥ 5 ∧ d ≥ 5 ∧ e ≥ 5 ∧ f ≥ 5

def N (a b c d e f : ℕ) : ℕ :=
  max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f))))

theorem smallest_possible_N :
  ∀ a b c d e f : ℕ, is_valid_arrangement a b c d e f →
  N a b c d e f ≥ 506 ∧
  (∃ a' b' c' d' e' f' : ℕ, is_valid_arrangement a' b' c' d' e' f' ∧ N a' b' c' d' e' f' = 506) :=
sorry

end smallest_possible_N_l1238_123818


namespace value_of_a_l1238_123862

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2

-- State the theorem
theorem value_of_a (a : ℝ) :
  (∀ x, (deriv (f a)) x = a) →  -- The derivative of f is constant and equal to a
  (deriv (f a)) 1 = 2 →         -- The derivative of f at x = 1 is 2
  a = 2 :=                      -- Then a must be equal to 2
by
  sorry

end value_of_a_l1238_123862


namespace polynomial_simplification_l1238_123819

theorem polynomial_simplification (x : ℝ) :
  (12 * x^10 + 5 * x^9 + 3 * x^8) + (2 * x^12 + 9 * x^10 + 4 * x^8 + 6 * x^4 + 7 * x^2 + 10) =
  2 * x^12 + 21 * x^10 + 5 * x^9 + 7 * x^8 + 6 * x^4 + 7 * x^2 + 10 :=
by sorry

end polynomial_simplification_l1238_123819


namespace count_integers_with_repeated_digits_eq_168_l1238_123820

/-- The number of positive three-digit integers less than 700 with at least two identical digits -/
def count_integers_with_repeated_digits : ℕ :=
  let total_three_digit_integers := 700 - 100
  let integers_without_repeated_digits := 6 * 9 * 8
  total_three_digit_integers - integers_without_repeated_digits

theorem count_integers_with_repeated_digits_eq_168 :
  count_integers_with_repeated_digits = 168 := by
  sorry

end count_integers_with_repeated_digits_eq_168_l1238_123820


namespace smallest_number_is_five_l1238_123874

theorem smallest_number_is_five (x y z : ℕ) 
  (sum_xy : x + y = 20) 
  (sum_xz : x + z = 27) 
  (sum_yz : y + z = 37) : 
  min x (min y z) = 5 := by
  sorry

end smallest_number_is_five_l1238_123874


namespace flag_designs_count_l1238_123821

/-- The number of available colors for the flag stripes -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The total number of possible flag designs -/
def total_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem stating that the total number of possible flag designs is 27 -/
theorem flag_designs_count : total_flag_designs = 27 := by
  sorry

end flag_designs_count_l1238_123821


namespace height_for_specific_configuration_l1238_123895

/-- Represents a configuration of three perpendicular rods fixed at one end -/
structure RodConfiguration where
  rod1 : ℝ
  rod2 : ℝ
  rod3 : ℝ

/-- Calculates the height of the fixed point above the plane for a given rod configuration -/
def height_above_plane (config : RodConfiguration) : ℝ :=
  sorry

/-- Theorem stating that for rods of lengths 1, 2, and 3, the height is 6/7 -/
theorem height_for_specific_configuration :
  let config : RodConfiguration := { rod1 := 1, rod2 := 2, rod3 := 3 }
  height_above_plane config = 6/7 :=
by sorry

end height_for_specific_configuration_l1238_123895


namespace sum_x_y_is_85_l1238_123858

/-- An arithmetic sequence with known terms 10, x, 30, y, 65 -/
structure ArithmeticSequence where
  x : ℝ
  y : ℝ
  isArithmetic : ∃ d : ℝ, x = 10 + d ∧ 30 = x + d ∧ y = 30 + 2*d ∧ 65 = y + d

/-- The sum of x and y in the arithmetic sequence is 85 -/
theorem sum_x_y_is_85 (seq : ArithmeticSequence) : seq.x + seq.y = 85 := by
  sorry

end sum_x_y_is_85_l1238_123858


namespace mark_sold_one_less_l1238_123897

/-- Given:
  n: total number of boxes allocated
  M: number of boxes Mark sold
  A: number of boxes Ann sold
-/
theorem mark_sold_one_less (n M A : ℕ) : 
  n = 8 → 
  M < n → 
  M ≥ 1 → 
  A = n - 2 → 
  A ≥ 1 → 
  M + A < n → 
  M = 7 :=
by sorry

end mark_sold_one_less_l1238_123897


namespace tommy_pencil_case_items_l1238_123814

/-- The number of items in Tommy's pencil case -/
theorem tommy_pencil_case_items (pencils : ℕ) (pens : ℕ) (eraser : ℕ) 
    (h1 : pens = 2 * pencils) 
    (h2 : eraser = 1)
    (h3 : pencils = 4) : 
  pencils + pens + eraser = 13 := by
  sorry

end tommy_pencil_case_items_l1238_123814


namespace simplify_sqrt_expression_l1238_123878

theorem simplify_sqrt_expression :
  (Real.sqrt 600 / Real.sqrt 75) - (Real.sqrt 243 / Real.sqrt 108) = (4 * Real.sqrt 2 - 3 * Real.sqrt 3) / 2 := by
  sorry

end simplify_sqrt_expression_l1238_123878


namespace lottery_first_prize_probability_l1238_123861

/-- The probability of winning a first prize in a lottery -/
theorem lottery_first_prize_probability
  (total_tickets : ℕ)
  (first_prizes : ℕ)
  (h_total : total_tickets = 150)
  (h_first : first_prizes = 5) :
  (first_prizes : ℚ) / total_tickets = 1 / 30 := by
  sorry

end lottery_first_prize_probability_l1238_123861


namespace man_downstream_speed_l1238_123847

/-- Calculates the downstream speed of a person given their upstream speed and the stream speed. -/
def downstream_speed (upstream_speed stream_speed : ℝ) : ℝ :=
  upstream_speed + 2 * stream_speed

/-- Theorem: Given a man's upstream speed of 8 km/h and a stream speed of 1 km/h, his downstream speed is 10 km/h. -/
theorem man_downstream_speed :
  let upstream_speed : ℝ := 8
  let stream_speed : ℝ := 1
  downstream_speed upstream_speed stream_speed = 10 := by
  sorry

end man_downstream_speed_l1238_123847


namespace triangle_inequality_triangle_equality_l1238_123888

/-- Triangle ABC with sides a, b, c, where a ≥ b and a ≥ c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_ge_b : a ≥ b
  a_ge_c : a ≥ c
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

/-- Circumradius of a triangle -/
def circumradius (t : Triangle) : ℝ := sorry

/-- Inradius of a triangle -/
def inradius (t : Triangle) : ℝ := sorry

/-- Length of centroidal axis from vertex A -/
def centroidal_axis_length (t : Triangle) : ℝ := sorry

/-- Altitude from vertex A to side BC -/
def altitude_a (t : Triangle) : ℝ := sorry

/-- A triangle is equilateral if all sides are equal -/
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

theorem triangle_inequality (t : Triangle) :
  circumradius t / (2 * inradius t) ≥ centroidal_axis_length t / altitude_a t :=
sorry

theorem triangle_equality (t : Triangle) :
  circumradius t / (2 * inradius t) = centroidal_axis_length t / altitude_a t ↔ is_equilateral t :=
sorry

end triangle_inequality_triangle_equality_l1238_123888


namespace total_revenue_calculation_l1238_123891

def fair_tickets : ℕ := 60
def fair_ticket_price : ℕ := 15
def baseball_ticket_price : ℕ := 10

theorem total_revenue_calculation :
  let baseball_tickets := fair_tickets / 3
  let fair_revenue := fair_tickets * fair_ticket_price
  let baseball_revenue := baseball_tickets * baseball_ticket_price
  fair_revenue + baseball_revenue = 1100 := by
sorry

end total_revenue_calculation_l1238_123891


namespace percent_of_y_l1238_123808

theorem percent_of_y (y : ℝ) (h : y > 0) : (1 * y / 20 + 3 * y / 10) / y * 100 = 35 := by
  sorry

end percent_of_y_l1238_123808


namespace inverse_zero_product_l1238_123833

theorem inverse_zero_product (a b : ℝ) : a = 0 → a * b = 0 := by
  sorry

end inverse_zero_product_l1238_123833


namespace find_number_l1238_123848

theorem find_number : ∃! x : ℝ, (((48 - x) * 4 - 26) / 2) = 37 := by
  sorry

end find_number_l1238_123848


namespace x0_in_N_l1238_123852

def M : Set ℝ := {x | ∃ k : ℤ, x = k + 1/2}
def N : Set ℝ := {x | ∃ k : ℤ, x = k/2 + 1}

theorem x0_in_N (x0 : ℝ) (h : x0 ∈ M) : x0 ∈ N := by
  sorry

end x0_in_N_l1238_123852


namespace function_composition_distribution_l1238_123822

-- Define real-valued functions on ℝ
variable (f g h : ℝ → ℝ)

-- Define function composition
def comp (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (g x)

-- Define pointwise multiplication of functions
def mult (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f x * g x

-- Statement of the theorem
theorem function_composition_distribution :
  ∀ x : ℝ, (comp (mult f g) h) x = (mult (comp f h) (comp g h)) x :=
by sorry

end function_composition_distribution_l1238_123822


namespace not_perfect_square_l1238_123807

theorem not_perfect_square (n : ℕ) : ¬∃ (m : ℕ), m^2 = 4*n + 2 := by
  sorry

end not_perfect_square_l1238_123807


namespace siblings_combined_age_l1238_123886

/-- The combined age of five siblings -/
def combined_age (aaron_age henry_sister_age henry_age alice_age eric_age : ℕ) : ℕ :=
  aaron_age + henry_sister_age + henry_age + alice_age + eric_age

theorem siblings_combined_age :
  ∀ (aaron_age henry_sister_age henry_age alice_age eric_age : ℕ),
    aaron_age = 15 →
    henry_sister_age = 3 * aaron_age →
    henry_age = 4 * henry_sister_age →
    alice_age = aaron_age - 2 →
    eric_age = henry_sister_age + alice_age →
    combined_age aaron_age henry_sister_age henry_age alice_age eric_age = 311 :=
by
  sorry

end siblings_combined_age_l1238_123886


namespace dans_remaining_marbles_l1238_123865

/-- The number of green marbles Dan has after Mike took some -/
def remaining_green_marbles (initial : ℕ) (taken : ℕ) : ℕ :=
  initial - taken

/-- Proof that Dan has 9 green marbles after Mike took 23 -/
theorem dans_remaining_marbles :
  remaining_green_marbles 32 23 = 9 := by
  sorry

end dans_remaining_marbles_l1238_123865


namespace cube_surface_division_l1238_123801

-- Define a cube
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 12)
  faces : Finset (Fin 6)
  bodyDiagonals : Finset (Fin 4)

-- Define a plane
structure Plane where
  normal : Vector ℝ 3

-- Define the function to erect planes perpendicular to body diagonals
def erectPerpendicularPlanes (c : Cube) : Finset Plane := sorry

-- Define the function to count surface parts
def countSurfaceParts (c : Cube) (planes : Finset Plane) : ℕ := sorry

-- Theorem statement
theorem cube_surface_division (c : Cube) :
  let perpendicularPlanes := erectPerpendicularPlanes c
  countSurfaceParts c perpendicularPlanes = 14 := by sorry

end cube_surface_division_l1238_123801


namespace triangle_properties_l1238_123873

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b * Real.cos t.C + t.c * Real.cos t.B = 2 * (Real.sqrt 3 / 3) * t.a * Real.cos t.B ∧
  t.b = Real.sqrt 7 ∧
  t.c = 2 * Real.sqrt 3 ∧
  t.a > t.b

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = π/6 ∧ (1/2 * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 7 / 2) := by
  sorry

end triangle_properties_l1238_123873


namespace inequality_not_always_true_l1238_123832

theorem inequality_not_always_true
  (a b c d : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : d ≠ 0) :
  ¬(∀ d, (a + d)^2 > (b + d)^2) ∧
  (a + c * d > b + c * d) ∧
  (a^2 - c * d > b^2 - c * d) ∧
  (a / c > b / c) ∧
  (Real.sqrt a * d^2 > Real.sqrt b * d^2) :=
sorry

end inequality_not_always_true_l1238_123832


namespace cylinder_volume_ratio_l1238_123868

/-- The ratio of volumes of cylinders formed from a rectangle --/
theorem cylinder_volume_ratio (w h : ℝ) (hw : w = 9) (hh : h = 12) :
  let v1 := π * (w / (2 * π))^2 * h
  let v2 := π * (h / (2 * π))^2 * w
  max v1 v2 / min v1 v2 = 16 / 3 := by
sorry

end cylinder_volume_ratio_l1238_123868


namespace parabola_focus_distance_l1238_123860

/-- The value of p for a parabola y² = 2px (p > 0) where the distance between (-2, 3) and the focus is 5 -/
theorem parabola_focus_distance (p : ℝ) : 
  p > 0 → 
  (∃ (x y : ℝ), y^2 = 2*p*x) → 
  let focus := (p, 0)
  Real.sqrt ((p - (-2))^2 + (0 - 3)^2) = 5 → 
  p = 2 := by sorry

end parabola_focus_distance_l1238_123860


namespace sqrt_factorial_over_88_l1238_123846

theorem sqrt_factorial_over_88 : 
  let factorial_10 : ℕ := 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
  let n : ℚ := factorial_10 / 88
  Real.sqrt n = (180 * Real.sqrt 7) / Real.sqrt 11 := by
  sorry

end sqrt_factorial_over_88_l1238_123846


namespace cara_don_meeting_l1238_123804

/-- Cara and Don walk towards each other's houses. -/
theorem cara_don_meeting 
  (distance_between_homes : ℝ) 
  (cara_speed : ℝ) 
  (don_speed : ℝ) 
  (don_start_delay : ℝ) 
  (h1 : distance_between_homes = 45) 
  (h2 : cara_speed = 6) 
  (h3 : don_speed = 5) 
  (h4 : don_start_delay = 2) : 
  ∃ x : ℝ, x = 30 ∧ 
  x + don_speed * (x / cara_speed - don_start_delay) = distance_between_homes :=
by
  sorry


end cara_don_meeting_l1238_123804


namespace probability_two_blue_buttons_l1238_123856

/-- Represents a jar with buttons -/
structure Jar :=
  (red : ℕ)
  (blue : ℕ)

/-- The probability of an event -/
def Probability := ℚ

/-- Initial state of Jar C -/
def initial_jar_c : Jar := ⟨5, 10⟩

/-- Number of buttons removed from each color -/
def removed_buttons : ℕ := 2

/-- Final state of Jar C after removal -/
def final_jar_c : Jar := ⟨initial_jar_c.red - removed_buttons, initial_jar_c.blue - 2 * removed_buttons⟩

/-- State of Jar D after receiving removed buttons -/
def jar_d : Jar := ⟨removed_buttons, 2 * removed_buttons⟩

/-- Theorem stating the probability of choosing two blue buttons -/
theorem probability_two_blue_buttons : 
  (final_jar_c.red + final_jar_c.blue : ℚ) = 3/5 * (initial_jar_c.red + initial_jar_c.blue) →
  (final_jar_c.blue : ℚ) / (final_jar_c.red + final_jar_c.blue) * 
  (jar_d.blue : ℚ) / (jar_d.red + jar_d.blue) = 4/9 :=
by sorry

end probability_two_blue_buttons_l1238_123856


namespace max_volume_cube_l1238_123835

/-- Given a constant sum of edges, the volume of a rectangular prism is maximized when it is a cube -/
theorem max_volume_cube (s : ℝ) (hs : s > 0) :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 3 * s →
  a * b * c ≤ s^3 ∧ (a * b * c = s^3 ↔ a = s ∧ b = s ∧ c = s) :=
by sorry

end max_volume_cube_l1238_123835
