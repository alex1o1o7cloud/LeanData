import Mathlib

namespace interest_rate_calculation_l1988_198882

/-- Calculates the simple interest rate given principal, amount, and time -/
def calculate_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  ((amount - principal) * 100) / (principal * time)

/-- Theorem stating that the interest rate is approximately 1.11% -/
theorem interest_rate_calculation (principal amount : ℚ) (time : ℕ) 
  (h_principal : principal = 900)
  (h_amount : amount = 950)
  (h_time : time = 5) :
  abs (calculate_interest_rate principal amount time - 1.11) < 0.01 := by
  sorry

#eval calculate_interest_rate 900 950 5

end interest_rate_calculation_l1988_198882


namespace damien_jogging_distance_l1988_198831

/-- The number of miles Damien jogs per day on weekdays -/
def miles_per_day : ℕ := 5

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weeks -/
def num_weeks : ℕ := 3

/-- The total distance Damien runs over three weeks -/
def total_distance : ℕ := miles_per_day * weekdays_per_week * num_weeks

theorem damien_jogging_distance :
  total_distance = 75 := by
  sorry

end damien_jogging_distance_l1988_198831


namespace count_integers_satisfying_inequality_l1988_198827

theorem count_integers_satisfying_inequality :
  (Finset.filter (fun n : ℤ => (n - 3) * (n + 5) * (n + 9) < 0)
    (Finset.Icc (-13 : ℤ) 13)).card = 11 := by
  sorry

end count_integers_satisfying_inequality_l1988_198827


namespace lemonade_stand_solution_l1988_198892

/-- Represents the lemonade stand problem --/
def lemonade_stand_problem (G : ℚ) : Prop :=
  let glasses_per_gallon : ℚ := 16
  let cost_per_gallon : ℚ := 3.5
  let price_per_glass : ℚ := 1
  let glasses_drunk : ℚ := 5
  let glasses_unsold : ℚ := 6
  let net_profit : ℚ := 14
  let total_glasses := G * glasses_per_gallon
  let glasses_sold := total_glasses - glasses_drunk - glasses_unsold
  let revenue := glasses_sold * price_per_glass
  let cost := G * cost_per_gallon
  revenue - cost = net_profit

/-- The solution to the lemonade stand problem --/
theorem lemonade_stand_solution :
  ∃ G : ℚ, lemonade_stand_problem G ∧ G = 2 := by
  sorry

end lemonade_stand_solution_l1988_198892


namespace rescue_mission_analysis_l1988_198869

def daily_distances : List Int := [14, -9, 8, -7, 13, -6, 10, -5]
def fuel_consumption : Rat := 1/2
def fuel_capacity : Nat := 29

theorem rescue_mission_analysis :
  let net_distance := daily_distances.sum
  let max_distance := daily_distances.scanl (· + ·) 0 |>.map abs |>.maximum
  let total_distance := daily_distances.map abs |>.sum
  let fuel_needed := fuel_consumption * total_distance - fuel_capacity
  (net_distance = 18 ∧ 
   max_distance = some 23 ∧ 
   fuel_needed = 7) := by sorry

end rescue_mission_analysis_l1988_198869


namespace tangent_slope_at_one_l1988_198877

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The slope of the tangent line to f(x) at x = 1 is 2 -/
theorem tangent_slope_at_one : 
  (deriv f) 1 = 2 := by sorry

end tangent_slope_at_one_l1988_198877


namespace range_of_a_l1988_198872

/-- Given a real number a, we define the following propositions: -/
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0

def q (x a : ℝ) : Prop := x > a

/-- The main theorem stating the range of values for a -/
theorem range_of_a (a : ℝ) :
  (∀ x, ¬(p x) → ¬(q x a)) →  -- Sufficient condition
  ¬(∀ x, ¬(q x a) → ¬(p x)) →  -- Not necessary condition
  a ≥ 1 :=
sorry

end range_of_a_l1988_198872


namespace jack_plates_left_l1988_198804

def plates_left (flower_plates checked_plates striped_plates : ℕ) : ℕ :=
  let polka_plates := checked_plates ^ 2
  let wave_plates := (4 * checked_plates) / 9
  let smashed_flower := (flower_plates * 10) / 100
  let smashed_checked := (checked_plates * 15) / 100
  let smashed_striped := (striped_plates * 20) / 100
  flower_plates - smashed_flower + checked_plates - smashed_checked + 
  striped_plates - smashed_striped + polka_plates + wave_plates

theorem jack_plates_left : plates_left 6 9 3 = 102 := by
  sorry

end jack_plates_left_l1988_198804


namespace triangle_side_length_l1988_198838

theorem triangle_side_length (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 120) (h3 : b = 45) (h4 : c = 15) 
  (side_b : ℝ) (h5 : side_b = 4 * Real.sqrt 6) : 
  side_b * Real.sin a / Real.sin b = 12 := by
sorry

end triangle_side_length_l1988_198838


namespace expansion_unique_solution_l1988_198850

/-- The number of terms in the expansion of (a+b+c+d+e+1)^n that include all five variables
    a, b, c, d, e, each to some positive power. -/
def numTerms (n : ℕ) : ℕ := Nat.choose n 5

/-- The proposition that 16 is the unique positive integer n such that the expansion of
    (a+b+c+d+e+1)^n contains exactly 2002 terms with all five variables a, b, c, d, e
    each to some positive power. -/
theorem expansion_unique_solution : 
  ∃! (n : ℕ), n > 0 ∧ numTerms n = 2002 ∧ n = 16 := by sorry

end expansion_unique_solution_l1988_198850


namespace min_value_of_f_l1988_198893

def f (x : ℝ) := x^2 - 2*x + 3

theorem min_value_of_f :
  ∃ (min : ℝ), min = 2 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → f x ≥ min :=
by sorry

end min_value_of_f_l1988_198893


namespace sum_of_seventh_powers_l1988_198814

/-- Given a sequence of sums of powers of a and b, prove that a^7 + b^7 = 29 -/
theorem sum_of_seventh_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^7 + b^7 = 29 := by sorry

end sum_of_seventh_powers_l1988_198814


namespace sector_angle_l1988_198817

/-- Given a circular sector with arc length and area both equal to 6,
    prove that the central angle in radians is 3. -/
theorem sector_angle (r : ℝ) (α : ℝ) : 
  r * α = 6 →  -- arc length formula
  (1 / 2) * r * α = 6 →  -- area formula
  α = 3 := by sorry

end sector_angle_l1988_198817


namespace potato_bag_weight_l1988_198802

theorem potato_bag_weight (bag_weight : ℝ) (h : bag_weight = 36) :
  bag_weight / (bag_weight / 2) = 2 ∧ bag_weight = 36 := by
  sorry

end potato_bag_weight_l1988_198802


namespace xyz_sum_product_l1988_198825

theorem xyz_sum_product (x y z : ℝ) 
  (h1 : 3 * (x + y + z) = x^2 + y^2 + z^2) 
  (h2 : x + y + z = 3) : 
  x * y + x * z + y * z = 0 := by
sorry

end xyz_sum_product_l1988_198825


namespace function_identity_l1988_198858

theorem function_identity (f : ℕ → ℕ) : 
  (∀ m n : ℕ, f (m + f n) = f (f m) + f n) → 
  (∀ n : ℕ, f n = n) := by
sorry

end function_identity_l1988_198858


namespace simplify_cube_roots_product_l1988_198870

theorem simplify_cube_roots_product : 
  (1 + 27) ^ (1/3 : ℝ) * (1 + 27 ^ (1/3 : ℝ)) ^ (1/3 : ℝ) * (4 : ℝ) ^ (1/2 : ℝ) = 2 * 112 ^ (1/3 : ℝ) :=
by sorry

end simplify_cube_roots_product_l1988_198870


namespace conflict_graph_contains_k4_l1988_198868

/-- A graph representing conflicts between mafia clans -/
structure ConflictGraph where
  /-- The set of vertices (clans) in the graph -/
  vertices : Finset Nat
  /-- The set of edges (conflicts) in the graph -/
  edges : Finset (Nat × Nat)
  /-- The number of vertices is 20 -/
  vertex_count : vertices.card = 20
  /-- Each vertex has a degree of at least 14 -/
  min_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≥ 14

/-- A complete subgraph of size 4 -/
def CompleteSubgraph4 (g : ConflictGraph) : Prop :=
  ∃ (a b c d : Nat), a ∈ g.vertices ∧ b ∈ g.vertices ∧ c ∈ g.vertices ∧ d ∈ g.vertices ∧
    (a, b) ∈ g.edges ∧ (a, c) ∈ g.edges ∧ (a, d) ∈ g.edges ∧
    (b, c) ∈ g.edges ∧ (b, d) ∈ g.edges ∧
    (c, d) ∈ g.edges

/-- Theorem: Every ConflictGraph contains a complete subgraph of size 4 -/
theorem conflict_graph_contains_k4 (g : ConflictGraph) : CompleteSubgraph4 g := by
  sorry

end conflict_graph_contains_k4_l1988_198868


namespace min_value_of_f_l1988_198895

open Real

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 + 2

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), m = 50/27 ∧ ∀ (x : ℝ), x > 0 → f x ≥ m :=
sorry

end min_value_of_f_l1988_198895


namespace complex_pure_imaginary_ratio_l1988_198853

theorem complex_pure_imaginary_ratio (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) 
  (h : ∃ (y : ℝ), (5 - 3 * Complex.I) * (m + n * Complex.I) = y * Complex.I) : 
  m / n = -3 / 5 := by
  sorry

end complex_pure_imaginary_ratio_l1988_198853


namespace total_distributions_l1988_198889

def number_of_balls : ℕ := 8
def number_of_boxes : ℕ := 3

def valid_distribution (d : List ℕ) : Prop :=
  d.length = number_of_boxes ∧
  d.sum = number_of_balls ∧
  d.all (· > 0) ∧
  d.Pairwise (· ≠ ·)

def count_distributions : ℕ := sorry

theorem total_distributions :
  count_distributions = 2688 := by sorry

end total_distributions_l1988_198889


namespace merchant_discount_l1988_198847

/-- Prove that given a 75% markup and a 57.5% profit after discount, the discount offered is 10%. -/
theorem merchant_discount (C : ℝ) (C_pos : C > 0) : 
  let M := 1.75 * C  -- Marked up price (75% markup)
  let S := 1.575 * C -- Selling price (57.5% profit)
  let D := (M - S) / M * 100 -- Discount percentage
  D = 10 := by sorry

end merchant_discount_l1988_198847


namespace greatest_integer_b_no_real_roots_l1988_198800

theorem greatest_integer_b_no_real_roots : 
  ∀ b : ℤ, (∀ x : ℝ, x^2 + b*x + 10 ≠ 0) → b ≤ 6 :=
by sorry

end greatest_integer_b_no_real_roots_l1988_198800


namespace not_integer_fraction_l1988_198871

theorem not_integer_fraction (a b : ℕ) (ha : a > b) (hb : b > 2) :
  ¬ (∃ k : ℤ, (2^a + 1 : ℤ) = k * (2^b - 1)) := by
  sorry

end not_integer_fraction_l1988_198871


namespace right_triangle_from_number_and_reciprocal_l1988_198878

theorem right_triangle_from_number_and_reciprocal (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let s := (a + 1/a) / 2
  let d := (a - 1/a) / 2
  let p := a * (1/a)
  s^2 = d^2 + p^2 := by sorry

end right_triangle_from_number_and_reciprocal_l1988_198878


namespace distance_to_y_axis_l1988_198805

theorem distance_to_y_axis (x : ℝ) :
  let P : ℝ × ℝ := (x, -8)
  let distance_to_x_axis := |P.2|
  let distance_to_y_axis := |P.1|
  distance_to_x_axis = (1/2 : ℝ) * distance_to_y_axis →
  distance_to_y_axis = 16 := by
  sorry

end distance_to_y_axis_l1988_198805


namespace lukes_stickers_l1988_198883

theorem lukes_stickers (initial_stickers birthday_stickers given_to_sister used_on_card final_stickers : ℕ) :
  initial_stickers = 20 →
  birthday_stickers = 20 →
  given_to_sister = 5 →
  used_on_card = 8 →
  final_stickers = 39 →
  ∃ (bought_stickers : ℕ),
    bought_stickers = 12 ∧
    initial_stickers + birthday_stickers + bought_stickers = final_stickers + given_to_sister + used_on_card :=
by sorry

end lukes_stickers_l1988_198883


namespace batsman_average_theorem_l1988_198880

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat
  lastInningScore : Nat

/-- Calculates the new average of a batsman after an additional inning -/
def newAverage (performance : BatsmanPerformance) : Nat :=
  (performance.totalRuns + performance.lastInningScore) / (performance.innings + 1)

/-- Theorem: If a batsman's average increases by 5 after scoring 85 in the 11th inning,
    then his new average is 35 -/
theorem batsman_average_theorem (performance : BatsmanPerformance) 
  (h1 : performance.innings = 10)
  (h2 : performance.lastInningScore = 85)
  (h3 : performance.averageIncrease = 5) :
  newAverage performance = 35 := by
  sorry

#check batsman_average_theorem

end batsman_average_theorem_l1988_198880


namespace basketball_tournament_l1988_198890

theorem basketball_tournament (x : ℕ) : 
  (3 * x / 4 : ℚ) = x - x / 4 ∧ 
  (2 * (x + 4) / 3 : ℚ) = (x + 4) - (x + 4) / 3 ∧ 
  (2 * (x + 4) / 3 : ℚ) = 3 * x / 4 + 9 ∧ 
  ((x + 4) / 3 : ℚ) = x / 4 + 5 → 
  x = 76 := by
sorry

end basketball_tournament_l1988_198890


namespace hyperbola_parabola_coincidence_l1988_198833

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 3 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Define the right vertex of the hyperbola
def hyperbola_right_vertex (a : ℝ) : ℝ × ℝ := (a, 0)

theorem hyperbola_parabola_coincidence (a : ℝ) (h : a > 0) :
  hyperbola_right_vertex a = parabola_focus → a = 2 := by
  sorry

end hyperbola_parabola_coincidence_l1988_198833


namespace conference_handshakes_l1988_198876

theorem conference_handshakes (n : ℕ) (h : n = 30) : (n * (n - 1)) / 2 = 435 := by
  sorry

end conference_handshakes_l1988_198876


namespace wednesdays_in_jan_feb_2012_l1988_198803

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in the year 2012 -/
structure Date2012 where
  month : Nat
  day : Nat

/-- Returns the day of the week for a given date in 2012 -/
def dayOfWeek (d : Date2012) : DayOfWeek :=
  sorry

/-- Returns the number of days in a given month of 2012 -/
def daysInMonth (m : Nat) : Nat :=
  sorry

/-- Counts the number of Wednesdays in a given month of 2012 -/
def countWednesdays (month : Nat) : Nat :=
  sorry

theorem wednesdays_in_jan_feb_2012 :
  (dayOfWeek ⟨1, 1⟩ = DayOfWeek.Sunday) →
  (countWednesdays 1 = 4 ∧ countWednesdays 2 = 5) :=
by sorry

end wednesdays_in_jan_feb_2012_l1988_198803


namespace house_purchase_l1988_198875

/-- Represents a number in base s -/
def BaseS (n : ℕ) (s : ℕ) : ℕ → ℕ
| 0 => 0
| (k+1) => (n % s) + s * BaseS (n / s) s k

theorem house_purchase (s : ℕ) 
  (h1 : BaseS 530 s 2 + BaseS 450 s 2 = BaseS 1100 s 3) : s = 8 :=
sorry

end house_purchase_l1988_198875


namespace g_of_3_equals_101_l1988_198854

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := 5 * x^3 - 4 * x^2 + 3 * x - 7

-- Theorem stating that g(3) = 101
theorem g_of_3_equals_101 : g 3 = 101 := by
  sorry

end g_of_3_equals_101_l1988_198854


namespace possible_values_of_d_l1988_198859

theorem possible_values_of_d (a b c d : ℕ) 
  (h : (a * d - 1) / (a + 1) + (b * d - 1) / (b + 1) + (c * d - 1) / (c + 1) = d) :
  d = 1 ∨ d = 2 ∨ d = 3 := by
  sorry

end possible_values_of_d_l1988_198859


namespace contractor_problem_l1988_198823

/-- Represents the initial number of people hired by the contractor -/
def initial_people : ℕ := 10

/-- Represents the total number of days allocated for the job -/
def total_days : ℕ := 100

/-- Represents the number of days worked before firing people -/
def days_before_firing : ℕ := 20

/-- Represents the fraction of work completed before firing people -/
def work_fraction_before_firing : ℚ := 1/4

/-- Represents the number of people fired -/
def people_fired : ℕ := 2

/-- Represents the number of days needed to complete the job after firing people -/
def days_after_firing : ℕ := 75

theorem contractor_problem :
  ∃ (p : ℕ), 
    p = initial_people ∧
    p * days_before_firing = work_fraction_before_firing * (p * total_days) ∧
    (p - people_fired) * days_after_firing = (1 - work_fraction_before_firing) * (p * total_days) :=
by sorry

end contractor_problem_l1988_198823


namespace linda_spent_correct_l1988_198896

/-- The total amount Linda spent on school supplies -/
def linda_total_spent : ℝ := 6.80

/-- The cost of a single notebook -/
def notebook_cost : ℝ := 1.20

/-- The number of notebooks Linda bought -/
def notebook_quantity : ℕ := 3

/-- The cost of a box of pencils -/
def pencil_box_cost : ℝ := 1.50

/-- The cost of a box of pens -/
def pen_box_cost : ℝ := 1.70

/-- Theorem stating that the total amount Linda spent is correct -/
theorem linda_spent_correct :
  linda_total_spent = notebook_cost * (notebook_quantity : ℝ) + pencil_box_cost + pen_box_cost := by
  sorry

end linda_spent_correct_l1988_198896


namespace gum_cost_l1988_198846

/-- Given that P packs of gum can be purchased for C coins,
    and 1 pack of gum costs 3 coins, prove that X packs of gum cost 3X coins. -/
theorem gum_cost (P C X : ℕ) (h1 : C = 3 * P) (h2 : X > 0) : 3 * X = C * X / P :=
sorry

end gum_cost_l1988_198846


namespace apartment_number_change_l1988_198830

/-- Represents a building with apartments and entrances. -/
structure Building where
  num_entrances : ℕ
  apartments_per_entrance : ℕ

/-- Calculates the apartment number given the entrance number and apartment number within the entrance. -/
def apartment_number (b : Building) (entrance : ℕ) (apartment_in_entrance : ℕ) : ℕ :=
  (entrance - 1) * b.apartments_per_entrance + apartment_in_entrance

/-- Theorem stating that if an apartment's number changes from 636 to 242 when entrance numbering is reversed in a 5-entrance building, then the total number of apartments is 985. -/
theorem apartment_number_change (b : Building) 
  (h1 : b.num_entrances = 5)
  (h2 : ∃ (e1 e2 a : ℕ), 
    apartment_number b e1 a = 636 ∧ 
    apartment_number b (b.num_entrances - e1 + 1) a = 242) :
  b.num_entrances * b.apartments_per_entrance = 985 := by
  sorry

#check apartment_number_change

end apartment_number_change_l1988_198830


namespace rosy_fish_count_l1988_198809

def lilly_fish : ℕ := 10
def total_fish : ℕ := 24

theorem rosy_fish_count : ∃ (rosy_fish : ℕ), rosy_fish = total_fish - lilly_fish ∧ rosy_fish = 14 := by
  sorry

end rosy_fish_count_l1988_198809


namespace peters_age_l1988_198820

theorem peters_age (peter_age jacob_age : ℕ) : 
  (peter_age - 10 = (jacob_age - 10) / 3) →
  (jacob_age = peter_age + 12) →
  peter_age = 16 := by
sorry

end peters_age_l1988_198820


namespace athlete_heartbeats_l1988_198884

/-- Calculates the total number of heartbeats during a race. -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (race_distance : ℕ) : ℕ :=
  heart_rate * pace * race_distance

/-- Proves that the athlete's heart beats 28800 times during the 30-mile race. -/
theorem athlete_heartbeats :
  let heart_rate : ℕ := 160  -- heartbeats per minute
  let pace : ℕ := 6          -- minutes per mile
  let race_distance : ℕ := 30 -- miles
  total_heartbeats heart_rate pace race_distance = 28800 := by
  sorry


end athlete_heartbeats_l1988_198884


namespace box_surface_area_is_744_l1988_198807

/-- The surface area of an open box formed by removing square corners from a rectangular sheet --/
def boxSurfaceArea (length width cornerSize : ℕ) : ℕ :=
  length * width - 4 * (cornerSize * cornerSize)

/-- Theorem stating that the surface area of the specified box is 744 square units --/
theorem box_surface_area_is_744 :
  boxSurfaceArea 40 25 8 = 744 := by
  sorry

end box_surface_area_is_744_l1988_198807


namespace matrix_power_negative_identity_l1988_198841

open Matrix

theorem matrix_power_negative_identity
  (A : Matrix (Fin 2) (Fin 2) ℚ)
  (h : ∃ (n : ℕ), n ≠ 0 ∧ A ^ n = -1 • 1) :
  A ^ 2 = -1 • 1 ∨ A ^ 3 = -1 • 1 := by
  sorry

end matrix_power_negative_identity_l1988_198841


namespace experimental_fertilizer_height_is_135_l1988_198813

-- Define the heights of plants with different fertilizers
def control_height : ℝ := 36

def bone_meal_height : ℝ := 1.25 * control_height

def cow_manure_height : ℝ := 2 * bone_meal_height

def experimental_fertilizer_height : ℝ := 1.5 * cow_manure_height

-- Theorem to prove
theorem experimental_fertilizer_height_is_135 :
  experimental_fertilizer_height = 135 := by
  sorry

end experimental_fertilizer_height_is_135_l1988_198813


namespace circle_is_point_l1988_198843

/-- The equation of the supposed circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 - 2*y + 5 = 0

/-- The center of the supposed circle -/
def center : ℝ × ℝ := (-2, 1)

theorem circle_is_point :
  ∀ (x y : ℝ), circle_equation x y ↔ (x, y) = center :=
sorry

end circle_is_point_l1988_198843


namespace brand_w_households_l1988_198865

theorem brand_w_households (total : ℕ) (neither : ℕ) (both : ℕ) : 
  total = 200 →
  neither = 80 →
  both = 40 →
  ∃ (w b : ℕ), w + b + both + neither = total ∧ b = 3 * both ∧ w = 40 :=
by sorry

end brand_w_households_l1988_198865


namespace axis_of_symmetry_l1988_198849

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x, f x = f (6 - x)) :
  is_symmetric_about f 3 := by sorry

end axis_of_symmetry_l1988_198849


namespace arithmetic_seq_sum_l1988_198837

-- Define an arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_seq_sum (a : ℕ → ℝ) :
  is_arithmetic_seq a → a 2 = 5 → a 6 = 33 → a 3 + a 5 = 38 := by
  sorry

end arithmetic_seq_sum_l1988_198837


namespace modified_goldbach_for_2024_l1988_198855

theorem modified_goldbach_for_2024 :
  ∃ (p q : ℕ), p ≠ q ∧ Prime p ∧ Prime q ∧ p + q = 2024 :=
by
  sorry

#check modified_goldbach_for_2024

end modified_goldbach_for_2024_l1988_198855


namespace man_crossing_street_speed_l1988_198821

/-- Proves that a man crossing a 600 m street in 5 minutes has a speed of 7.2 km/h -/
theorem man_crossing_street_speed :
  let distance_m : ℝ := 600
  let time_min : ℝ := 5
  let distance_km : ℝ := distance_m / 1000
  let time_h : ℝ := time_min / 60
  let speed_km_h : ℝ := distance_km / time_h
  speed_km_h = 7.2 := by sorry

end man_crossing_street_speed_l1988_198821


namespace min_coins_for_eternal_collection_l1988_198835

/-- Represents the JMO kingdom with its citizens and coin distribution. -/
structure Kingdom (n : ℕ) where
  /-- The number of citizens in the kingdom is 2^n. -/
  citizens : ℕ := 2^n
  /-- The value of paper bills used in the kingdom. -/
  bill_value : ℕ := 2^n
  /-- The possible values of coins in the kingdom. -/
  coin_values : List ℕ := List.range n |>.map (fun a => 2^a)

/-- The sum of digits function in base 2. -/
def sum_of_digits (a : ℕ) : ℕ := sorry

/-- Theorem stating the minimum number of coins required for the king to collect money every night eternally. -/
theorem min_coins_for_eternal_collection (n : ℕ) (h : n > 0) : 
  ∃ (S : ℕ), S = n * 2^(n-1) ∧ 
  ∀ (S' : ℕ), S' < S → ¬(∃ (distribution : ℕ → ℕ), 
    (∀ i, i < 2^n → distribution i ≤ sum_of_digits i) ∧
    (∀ t : ℕ, ∃ (new_distribution : ℕ → ℕ), 
      (∀ i, i < 2^n → new_distribution i = distribution ((i + 1) % 2^n) + 1) ∧
      (∀ i, i < 2^n → new_distribution i ≤ sum_of_digits i))) :=
sorry

end min_coins_for_eternal_collection_l1988_198835


namespace alyssa_kittens_l1988_198888

/-- The number of kittens Alyssa initially had -/
def initial_kittens : ℕ := 8

/-- The number of kittens Alyssa gave to her friends -/
def given_away_kittens : ℕ := 4

/-- The number of kittens Alyssa now has -/
def remaining_kittens : ℕ := initial_kittens - given_away_kittens

theorem alyssa_kittens : remaining_kittens = 4 := by
  sorry

end alyssa_kittens_l1988_198888


namespace rectangular_box_width_l1988_198828

/-- The width of a rectangular box that fits in a wooden box -/
theorem rectangular_box_width :
  let wooden_box_length : ℝ := 8 -- in meters
  let wooden_box_width : ℝ := 7 -- in meters
  let wooden_box_height : ℝ := 6 -- in meters
  let rect_box_length : ℝ := 4 / 100 -- in meters
  let rect_box_height : ℝ := 6 / 100 -- in meters
  let max_boxes : ℕ := 2000000
  ∃ (w : ℝ),
    w > 0 ∧
    (wooden_box_length * wooden_box_width * wooden_box_height) / 
    (rect_box_length * w * rect_box_height) = max_boxes ∧
    w = 7 / 100 -- width in meters
  := by sorry


end rectangular_box_width_l1988_198828


namespace apple_tree_yield_l1988_198836

theorem apple_tree_yield (apple_trees peach_trees : ℕ) 
  (peach_yield total_yield : ℝ) (h1 : apple_trees = 30) 
  (h2 : peach_trees = 45) (h3 : peach_yield = 65) 
  (h4 : total_yield = 7425) : 
  (total_yield - peach_trees * peach_yield) / apple_trees = 150 := by
sorry

end apple_tree_yield_l1988_198836


namespace stating_min_toothpicks_theorem_l1988_198848

/-- Represents a figure made of toothpicks and triangles -/
structure TriangleFigure where
  total_toothpicks : ℕ
  upward_1triangles : ℕ
  downward_1triangles : ℕ
  upward_2triangles : ℕ

/-- 
  Given a TriangleFigure, calculates the minimum number of toothpicks 
  that must be removed to eliminate all triangles
-/
def min_toothpicks_to_remove (figure : TriangleFigure) : ℕ :=
  sorry

/-- 
  Theorem stating that for the given figure, 
  the minimum number of toothpicks to remove is 15
-/
theorem min_toothpicks_theorem (figure : TriangleFigure) 
  (h1 : figure.total_toothpicks = 60)
  (h2 : figure.upward_1triangles = 22)
  (h3 : figure.downward_1triangles = 14)
  (h4 : figure.upward_2triangles = 4) :
  min_toothpicks_to_remove figure = 15 :=
by sorry

end stating_min_toothpicks_theorem_l1988_198848


namespace inequality_statements_l1988_198811

theorem inequality_statements :
  (∃ a b : ℝ, a > b ∧ a^2 ≤ b^2) ∧
  (∀ a b : ℝ, a > b ↔ a^3 > b^3) ∧
  (∃ a b : ℝ, a > b ∧ |a| ≤ |b|) ∧
  (∃ a b c : ℝ, a * c^2 ≤ b * c^2 ∧ a ≤ b) :=
by sorry

end inequality_statements_l1988_198811


namespace sum_of_squares_of_roots_l1988_198829

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 1 = 0 → 
  x₂^2 - 3*x₂ - 1 = 0 → 
  x₁^2 + x₂^2 = 11 := by
sorry

end sum_of_squares_of_roots_l1988_198829


namespace towel_area_decrease_l1988_198834

theorem towel_area_decrease (L B : ℝ) (h_positive : L > 0 ∧ B > 0) : 
  let original_area := L * B
  let new_length := 0.8 * L
  let new_breadth := 0.8 * B
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area = 0.36 := by
sorry

end towel_area_decrease_l1988_198834


namespace horse_feeding_amount_l1988_198857

/-- Calculates the amount of food each horse receives at each feeding --/
def food_per_horse_per_feeding (num_horses : ℕ) (feedings_per_day : ℕ) (days : ℕ) (bags_bought : ℕ) (pounds_per_bag : ℕ) : ℕ :=
  (bags_bought * pounds_per_bag) / (num_horses * feedings_per_day * days)

/-- Theorem stating the amount of food each horse receives at each feeding --/
theorem horse_feeding_amount :
  food_per_horse_per_feeding 25 2 60 60 1000 = 20 := by
  sorry

#eval food_per_horse_per_feeding 25 2 60 60 1000

end horse_feeding_amount_l1988_198857


namespace arith_progression_poly_j_value_l1988_198861

/-- A polynomial of degree 4 with four distinct real roots in arithmetic progression -/
structure ArithProgressionPoly where
  j : ℝ
  k : ℝ
  roots : Fin 4 → ℝ
  distinct : ∀ i j, i ≠ j → roots i ≠ roots j
  arithmetic_progression : ∃ (a d : ℝ), ∀ i, roots i = a + i * d
  is_root : ∀ i, (roots i)^4 + j * (roots i)^2 + k * (roots i) + 400 = 0

/-- The value of j in an ArithProgressionPoly is -40 -/
theorem arith_progression_poly_j_value (p : ArithProgressionPoly) : p.j = -40 := by
  sorry

end arith_progression_poly_j_value_l1988_198861


namespace min_value_of_permutation_sum_l1988_198867

theorem min_value_of_permutation_sum :
  ∀ (x₁ x₂ x₃ x₄ x₅ : ℕ),
  (x₁ :: x₂ :: x₃ :: x₄ :: x₅ :: []).Perm [1, 2, 3, 4, 5] →
  (∀ (y₁ y₂ y₃ y₄ y₅ : ℕ),
    (y₁ :: y₂ :: y₃ :: y₄ :: y₅ :: []).Perm [1, 2, 3, 4, 5] →
    x₁ + 2*x₂ + 3*x₃ + 4*x₄ + 5*x₅ ≤ y₁ + 2*y₂ + 3*y₃ + 4*y₄ + 5*y₅) →
  x₁ + 2*x₂ + 3*x₃ + 4*x₄ + 5*x₅ = 35 :=
by sorry

end min_value_of_permutation_sum_l1988_198867


namespace function_evaluation_l1988_198899

/-- Given a function f(x) = x^2 + 1, prove that f(a+1) = a^2 + 2a + 2 -/
theorem function_evaluation (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 1
  f (a + 1) = a^2 + 2*a + 2 := by
sorry

end function_evaluation_l1988_198899


namespace piecewise_representation_of_f_l1988_198844

def f (x : ℝ) := |x - 1| + 1

theorem piecewise_representation_of_f :
  ∀ x : ℝ, f x = if x ≥ 1 then x else 2 - x := by
  sorry

end piecewise_representation_of_f_l1988_198844


namespace num_planes_determined_by_skew_lines_l1988_198824

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line in 3D space
  -- (We don't need to fully implement this for the statement)

/-- A point in 3D space -/
structure Point3D where
  -- Define properties of a point in 3D space
  -- (We don't need to fully implement this for the statement)

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane in 3D space
  -- (We don't need to fully implement this for the statement)

/-- Two lines are skew if they are not parallel and do not intersect -/
def areSkewLines (l1 l2 : Line3D) : Prop :=
  sorry

/-- A point lies on a line -/
def pointOnLine (p : Point3D) (l : Line3D) : Prop :=
  sorry

/-- A plane is determined by a line and a point not on that line -/
def planeDeterminedByLineAndPoint (l : Line3D) (p : Point3D) : Plane3D :=
  sorry

/-- The number of unique planes determined by two skew lines and points on them -/
def numUniquePlanes (a b : Line3D) (pointsOnA pointsOnB : Finset Point3D) : ℕ :=
  sorry

theorem num_planes_determined_by_skew_lines 
  (a b : Line3D) 
  (pointsOnA pointsOnB : Finset Point3D) 
  (h_skew : areSkewLines a b)
  (h_pointsA : ∀ p ∈ pointsOnA, pointOnLine p a)
  (h_pointsB : ∀ p ∈ pointsOnB, pointOnLine p b)
  (h_countA : pointsOnA.card = 5)
  (h_countB : pointsOnB.card = 4) :
  numUniquePlanes a b pointsOnA pointsOnB = 5 := by
  sorry

end num_planes_determined_by_skew_lines_l1988_198824


namespace jade_cal_difference_l1988_198818

-- Define the number of transactions for each person
def mabel_transactions : ℕ := 90
def anthony_transactions : ℕ := mabel_transactions + (mabel_transactions / 10)
def cal_transactions : ℕ := (2 * anthony_transactions) / 3
def jade_transactions : ℕ := 83

-- Theorem to prove
theorem jade_cal_difference : jade_transactions - cal_transactions = 17 := by
  sorry

end jade_cal_difference_l1988_198818


namespace min_inverse_sum_min_inverse_sum_achieved_l1988_198874

theorem min_inverse_sum (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_sum : x + y = 12) (h_prod : x * y = 20) : 
  (1 / x + 1 / y) ≥ 3 / 5 := by
  sorry

theorem min_inverse_sum_achieved (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_sum : x + y = 12) (h_prod : x * y = 20) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 12 ∧ x * y = 20 ∧ 1 / x + 1 / y = 3 / 5 := by
  sorry

end min_inverse_sum_min_inverse_sum_achieved_l1988_198874


namespace missing_number_proof_l1988_198894

theorem missing_number_proof (x : ℝ) : x + Real.sqrt (-4 + 6 * 4 / 3) = 13 ↔ x = 11 := by
  sorry

end missing_number_proof_l1988_198894


namespace impossibility_theorem_l1988_198897

theorem impossibility_theorem (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) : 
  ¬(((1 - a) * b > 1/4) ∧ ((1 - b) * c > 1/4) ∧ ((1 - c) * a > 1/4)) := by
  sorry

end impossibility_theorem_l1988_198897


namespace variable_value_l1988_198887

theorem variable_value : 
  ∀ (a n some_variable : ℤ) (x : ℝ),
  (3 * x + 2) * (2 * x - 7) = a * x^2 + some_variable * x + n →
  a - n + some_variable = 3 →
  some_variable = -17 := by
sorry

end variable_value_l1988_198887


namespace eight_divided_by_repeating_decimal_l1988_198842

-- Define the repeating decimal 0.888...
def repeating_decimal : ℚ := 8 / 9

-- State the theorem
theorem eight_divided_by_repeating_decimal : 8 / repeating_decimal = 9 := by
  sorry

end eight_divided_by_repeating_decimal_l1988_198842


namespace add_squares_l1988_198826

theorem add_squares (a : ℝ) : 2 * a^2 + a^2 = 3 * a^2 := by
  sorry

end add_squares_l1988_198826


namespace arithmetic_geometric_mean_inequality_l1988_198862

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hc_def : c = (a + b) / 2) :
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) := by
  sorry

end arithmetic_geometric_mean_inequality_l1988_198862


namespace rabbit_catches_cat_l1988_198801

/-- Proves that a rabbit catches up to a cat in 1 hour given their speeds and the cat's head start -/
theorem rabbit_catches_cat (rabbit_speed cat_speed : ℝ) (head_start : ℝ) : 
  rabbit_speed = 25 →
  cat_speed = 20 →
  head_start = 0.25 →
  (rabbit_speed - cat_speed) * 1 = cat_speed * head_start := by
  sorry

#check rabbit_catches_cat

end rabbit_catches_cat_l1988_198801


namespace inequality_solution_set_l1988_198812

def solution_set : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem inequality_solution_set :
  ∀ x : ℝ, (x - 2) / (x + 1) ≤ 0 ↔ x ∈ solution_set :=
by sorry

end inequality_solution_set_l1988_198812


namespace michelle_score_l1988_198885

/-- Michelle's basketball game record --/
theorem michelle_score (total_score : ℕ) (num_players : ℕ) (other_players : ℕ) (avg_other_score : ℕ) : 
  total_score = 72 →
  num_players = 8 →
  other_players = 7 →
  avg_other_score = 6 →
  total_score - (other_players * avg_other_score) = 30 := by
sorry

end michelle_score_l1988_198885


namespace evaluate_expression_l1988_198866

theorem evaluate_expression : 
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) + 5 = 15 := by
  sorry

end evaluate_expression_l1988_198866


namespace family_buffet_employees_l1988_198806

theorem family_buffet_employees (total : ℕ) (dining : ℕ) (snack : ℕ) (two_restaurants : ℕ) (all_restaurants : ℕ) : 
  total = 39 →
  dining = 18 →
  snack = 12 →
  two_restaurants = 4 →
  all_restaurants = 3 →
  ∃ family : ℕ, family = 20 ∧ 
    family + dining + snack - two_restaurants - 2 * all_restaurants + all_restaurants = total :=
by sorry

end family_buffet_employees_l1988_198806


namespace courtyard_stones_l1988_198840

theorem courtyard_stones (stones : ℕ) (trees : ℕ) (birds : ℕ) : 
  trees = 3 * stones →
  birds = 2 * (trees + stones) →
  birds = 400 →
  stones = 40 := by
sorry

end courtyard_stones_l1988_198840


namespace sum_of_numbers_with_lcm_and_ratio_l1988_198819

/-- Given two positive integers with LCM 420 and in the ratio 4:7, prove their sum is 165 -/
theorem sum_of_numbers_with_lcm_and_ratio (a b : ℕ+) : 
  Nat.lcm a b = 420 → a * 7 = b * 4 → a + b = 165 := by sorry

end sum_of_numbers_with_lcm_and_ratio_l1988_198819


namespace investment_proof_l1988_198839

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proof of the investment problem -/
theorem investment_proof :
  let principal : ℝ := 315.84
  let rate : ℝ := 0.12
  let time : ℕ := 6
  let final_value : ℝ := 635.48
  abs (compound_interest principal rate time - final_value) < 0.01 := by
sorry


end investment_proof_l1988_198839


namespace quadratic_equation_solution_l1988_198808

theorem quadratic_equation_solution :
  let equation := fun x : ℂ => 3 * x^2 + 7 - (6 * x - 4)
  let solution1 := 1 + (2 * Real.sqrt 6 / 3) * I
  let solution2 := 1 - (2 * Real.sqrt 6 / 3) * I
  let a : ℝ := 1
  let b : ℝ := 2 * Real.sqrt 6 / 3
  (equation solution1 = 0) ∧
  (equation solution2 = 0) ∧
  (a + b^2 = 11/3) := by
  sorry

end quadratic_equation_solution_l1988_198808


namespace g_range_l1988_198864

noncomputable def g (x : ℝ) : ℝ := 
  (Real.cos x ^ 3 + 3 * Real.cos x ^ 2 - 4 * Real.cos x + 5 * Real.sin x ^ 2 - 7) / (Real.cos x - 2)

theorem g_range : 
  ∀ y : ℝ, (∃ x : ℝ, Real.cos x ≠ 2 ∧ g x = y) ↔ 1 ≤ y ∧ y ≤ 2 := by
  sorry

end g_range_l1988_198864


namespace limit_of_f_difference_quotient_l1988_198891

def f (x : ℝ) := x^2

theorem limit_of_f_difference_quotient :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f Δx) - (f 0)) / Δx - 0| < ε := by sorry

end limit_of_f_difference_quotient_l1988_198891


namespace stone_count_is_odd_l1988_198852

/-- Represents the assembly of stones along a road -/
structure StoneAssembly where
  stone_interval : ℝ  -- Distance between stones in meters
  total_distance : ℝ  -- Total distance covered in meters

/-- Theorem: The total number of stones is odd given the conditions -/
theorem stone_count_is_odd (assembly : StoneAssembly) 
  (h_interval : assembly.stone_interval = 10)
  (h_distance : assembly.total_distance = 4800) : 
  ∃ (n : ℕ), (2 * n + 1) * assembly.stone_interval = assembly.total_distance / 2 := by
  sorry

#check stone_count_is_odd

end stone_count_is_odd_l1988_198852


namespace cubic_polynomial_root_monic_cubic_integer_coeffs_l1988_198873

theorem cubic_polynomial_root (x : ℝ) : x = Real.rpow 5 (1/3) + 2 →
  x^3 - 6*x^2 + 12*x - 13 = 0 := by sorry

theorem monic_cubic_integer_coeffs :
  ∃ (a b c : ℤ), ∀ (x : ℝ), x^3 - 6*x^2 + 12*x - 13 = x^3 + a*x^2 + b*x + c := by sorry

end cubic_polynomial_root_monic_cubic_integer_coeffs_l1988_198873


namespace real_number_groups_ratio_l1988_198881

theorem real_number_groups_ratio (k : ℝ) (hk : k > 0) : 
  ∃ (group : Set ℝ) (a b c : ℝ), 
    (group ∪ (Set.univ \ group) = Set.univ) ∧ 
    (group ∩ (Set.univ \ group) = ∅) ∧
    (a ∈ group ∧ b ∈ group ∧ c ∈ group) ∧
    (a < b ∧ b < c) ∧
    ((c - b) / (b - a) = k) :=
sorry

end real_number_groups_ratio_l1988_198881


namespace smallest_angle_tan_equation_l1988_198845

theorem smallest_angle_tan_equation (x : Real) : 
  (x > 0) →
  (x < 9 * Real.pi / 180) →
  (Real.tan (4 * x) ≠ (Real.cos x - Real.sin x) / (Real.cos x + Real.sin x)) :=
by sorry

end smallest_angle_tan_equation_l1988_198845


namespace mutually_exclusive_pairs_l1988_198816

-- Define a type for events
inductive Event
| SevenRing
| EightRing
| AtLeastOneHit
| AHitsBMisses
| AtLeastOneBlack
| BothRed
| NoBlack
| ExactlyOneRed

-- Define a function to check if two events are mutually exclusive
def mutuallyExclusive (e1 e2 : Event) : Prop :=
  ¬(∃ (outcome : Set Event), outcome.Nonempty ∧ e1 ∈ outcome ∧ e2 ∈ outcome)

-- Define the pairs of events
def pair1 : (Event × Event) := (Event.SevenRing, Event.EightRing)
def pair2 : (Event × Event) := (Event.AtLeastOneHit, Event.AHitsBMisses)
def pair3 : (Event × Event) := (Event.AtLeastOneBlack, Event.BothRed)
def pair4 : (Event × Event) := (Event.NoBlack, Event.ExactlyOneRed)

-- State the theorem
theorem mutually_exclusive_pairs :
  mutuallyExclusive pair1.1 pair1.2 ∧
  ¬(mutuallyExclusive pair2.1 pair2.2) ∧
  mutuallyExclusive pair3.1 pair3.2 ∧
  mutuallyExclusive pair4.1 pair4.2 := by
  sorry

end mutually_exclusive_pairs_l1988_198816


namespace polynomial_equality_l1988_198832

theorem polynomial_equality (x : ℝ) (h : ℝ → ℝ) : 
  4 * x^5 + 5 * x^3 - 3 * x + h x = 2 * x^3 - 4 * x^2 + 9 * x + 2 → 
  h x = -4 * x^5 - 3 * x^3 - 4 * x^2 + 12 * x + 2 := by
sorry

end polynomial_equality_l1988_198832


namespace find_B_l1988_198860

theorem find_B (A B : ℕ) (h1 : A < 10) (h2 : B < 10) (h3 : 6 * 100 * A + 5 + 100 * B + 3 = 748) : B = 1 := by
  sorry

end find_B_l1988_198860


namespace painting_job_completion_time_l1988_198856

/-- Represents the painting job with given conditions -/
structure PaintingJob where
  original_men : ℕ
  original_days : ℕ
  additional_men : ℕ
  efficiency_increase : ℚ

/-- Calculates the number of days required to complete the job with additional skilled workers -/
def days_with_skilled_workers (job : PaintingJob) : ℚ :=
  let total_man_days := job.original_men * job.original_days
  let original_daily_output := job.original_men
  let skilled_daily_output := job.additional_men * (1 + job.efficiency_increase)
  let total_daily_output := original_daily_output + skilled_daily_output
  total_man_days / total_daily_output

/-- The main theorem stating that the job will be completed in 4 days -/
theorem painting_job_completion_time :
  let job := PaintingJob.mk 10 6 4 (1/4)
  days_with_skilled_workers job = 4 := by
  sorry

#eval days_with_skilled_workers (PaintingJob.mk 10 6 4 (1/4))

end painting_job_completion_time_l1988_198856


namespace japanese_turtle_crane_problem_l1988_198863

/-- Represents the number of cranes in the cage. -/
def num_cranes : ℕ := sorry

/-- Represents the number of turtles in the cage. -/
def num_turtles : ℕ := sorry

/-- The total number of heads in the cage. -/
def total_heads : ℕ := 35

/-- The total number of feet in the cage. -/
def total_feet : ℕ := 94

/-- The number of feet a crane has. -/
def crane_feet : ℕ := 2

/-- The number of feet a turtle has. -/
def turtle_feet : ℕ := 4

/-- Theorem stating that the system of equations correctly represents the Japanese turtle and crane problem. -/
theorem japanese_turtle_crane_problem :
  (num_cranes + num_turtles = total_heads) ∧
  (crane_feet * num_cranes + turtle_feet * num_turtles = total_feet) :=
sorry

end japanese_turtle_crane_problem_l1988_198863


namespace diophantine_equation_solution_l1988_198851

theorem diophantine_equation_solution :
  ∀ (p q r k : ℕ),
    p.Prime ∧ q.Prime ∧ r.Prime ∧ k > 0 →
    p^2 + q^2 + 49*r^2 = 9*k^2 - 101 →
    ((p = 3 ∧ q = 5 ∧ r = 3 ∧ k = 8) ∨ (p = 5 ∧ q = 3 ∧ r = 3 ∧ k = 8)) :=
by sorry

end diophantine_equation_solution_l1988_198851


namespace infinitely_many_satisfying_points_l1988_198898

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2}

-- Define the diameter endpoints
def DiameterEndpoints (center : ℝ × ℝ) (radius : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((center.1 - radius, center.2), (center.1 + radius, center.2))

-- Define the distance squared between two points
def DistanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Define the set of points P satisfying the condition
def SatisfyingPoints (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | p ∈ Circle center radius ∧
               let (a, b) := DiameterEndpoints center radius
               DistanceSquared p a + DistanceSquared p b = 10}

-- Theorem statement
theorem infinitely_many_satisfying_points (center : ℝ × ℝ) :
  Set.Infinite (SatisfyingPoints center 2) :=
sorry

end infinitely_many_satisfying_points_l1988_198898


namespace gamma_value_l1988_198822

/-- Given that γ is directly proportional to the square of δ, 
    and γ = 25 when δ = 5, prove that γ = 64 when δ = 8 -/
theorem gamma_value (γ δ : ℝ) (h1 : ∃ (k : ℝ), ∀ x, γ = k * x^2) 
  (h2 : γ = 25 ∧ δ = 5) : 
  (δ = 8 → γ = 64) := by
  sorry


end gamma_value_l1988_198822


namespace tangent_condition_l1988_198886

/-- The equation of the curve -/
def curve_eq (x y : ℝ) : Prop := y^2 - 4*x - 2*y + 1 = 0

/-- The equation of the line -/
def line_eq (k x y : ℝ) : Prop := y = k*x + 2

/-- The line is tangent to the curve -/
def is_tangent (k : ℝ) : Prop :=
  ∃! x y, curve_eq x y ∧ line_eq k x y

/-- The main theorem -/
theorem tangent_condition :
  ∀ k, is_tangent k ↔ (k = -2 + 2*Real.sqrt 2 ∨ k = -2 - 2*Real.sqrt 2) :=
sorry

end tangent_condition_l1988_198886


namespace optimal_investment_plan_l1988_198815

/-- Represents an investment project --/
structure Project where
  maxProfitRate : ℝ
  maxLossRate : ℝ

/-- Represents an investment plan --/
structure InvestmentPlan where
  projectA : ℝ
  projectB : ℝ

def totalInvestment (plan : InvestmentPlan) : ℝ :=
  plan.projectA + plan.projectB

def potentialProfit (plan : InvestmentPlan) (projectA projectB : Project) : ℝ :=
  plan.projectA * projectA.maxProfitRate + plan.projectB * projectB.maxProfitRate

def potentialLoss (plan : InvestmentPlan) (projectA projectB : Project) : ℝ :=
  plan.projectA * projectA.maxLossRate + plan.projectB * projectB.maxLossRate

theorem optimal_investment_plan 
  (projectA : Project)
  (projectB : Project)
  (h_profitA : projectA.maxProfitRate = 1)
  (h_profitB : projectB.maxProfitRate = 0.5)
  (h_lossA : projectA.maxLossRate = 0.3)
  (h_lossB : projectB.maxLossRate = 0.1)
  (optimalPlan : InvestmentPlan)
  (h_optimalA : optimalPlan.projectA = 40000)
  (h_optimalB : optimalPlan.projectB = 60000) :
  (∀ plan : InvestmentPlan, 
    totalInvestment plan ≤ 100000 ∧ 
    potentialLoss plan projectA projectB ≤ 18000 →
    potentialProfit plan projectA projectB ≤ potentialProfit optimalPlan projectA projectB) ∧
  totalInvestment optimalPlan ≤ 100000 ∧
  potentialLoss optimalPlan projectA projectB ≤ 18000 :=
sorry

end optimal_investment_plan_l1988_198815


namespace ruby_height_l1988_198879

/-- Given the heights of various people, prove Ruby's height -/
theorem ruby_height
  (janet_height : ℕ)
  (charlene_height : ℕ)
  (pablo_height : ℕ)
  (ruby_height : ℕ)
  (h1 : janet_height = 62)
  (h2 : charlene_height = 2 * janet_height)
  (h3 : pablo_height = charlene_height + 70)
  (h4 : ruby_height = pablo_height - 2)
  : ruby_height = 192 := by
  sorry

end ruby_height_l1988_198879


namespace symmetric_sine_cosine_l1988_198810

theorem symmetric_sine_cosine (φ : ℝ) (h1 : |φ| < π/2) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (x + φ) - Real.sqrt 3 * Real.cos (x + φ)
  (∀ x, f (2*π - x) = f x) →
  Real.cos (2*φ) = 1/2 := by
  sorry

end symmetric_sine_cosine_l1988_198810
