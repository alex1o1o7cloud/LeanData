import Mathlib

namespace path_1310_to_1315_l597_59741

/-- Represents a point in the cyclic path --/
def CyclicPoint := ℕ

/-- The length of one cycle in the path --/
def cycleLength : ℕ := 6

/-- Converts a given point to its equivalent position within a cycle --/
def toCyclicPosition (n : ℕ) : CyclicPoint :=
  n % cycleLength

/-- Checks if two points are equivalent in the cyclic representation --/
def areEquivalentPoints (a b : ℕ) : Prop :=
  toCyclicPosition a = toCyclicPosition b

theorem path_1310_to_1315 :
  areEquivalentPoints 1310 2 ∧ 
  areEquivalentPoints 1315 3 ∧
  (1315 - 1310 = cycleLength + 3) := by
  sorry

#check path_1310_to_1315

end path_1310_to_1315_l597_59741


namespace expression_value_l597_59790

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end expression_value_l597_59790


namespace steve_growth_l597_59723

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

theorem steve_growth :
  let original_height := feet_inches_to_inches 5 6
  let new_height := 72
  new_height - original_height = 6 := by
  sorry

end steve_growth_l597_59723


namespace multiplication_result_l597_59725

theorem multiplication_result : 72515 * 10005 = 724787425 := by
  sorry

end multiplication_result_l597_59725


namespace power_mod_seventeen_l597_59748

theorem power_mod_seventeen : 5^2023 ≡ 11 [ZMOD 17] := by sorry

end power_mod_seventeen_l597_59748


namespace distance_between_AB_l597_59764

/-- The distance between two points A and B, where two motorcyclists meet twice -/
def distance_AB : ℝ := 125

/-- The distance of the first meeting point from B -/
def distance_first_meeting : ℝ := 50

/-- The distance of the second meeting point from A -/
def distance_second_meeting : ℝ := 25

/-- Theorem stating that the distance between A and B is 125 km -/
theorem distance_between_AB : 
  distance_AB = distance_first_meeting + distance_second_meeting :=
by sorry

end distance_between_AB_l597_59764


namespace vector_sum_not_necessarily_zero_l597_59727

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given arbitrary points O, A, B, C in a real vector space V, 
    the vector expression OA + OC + BO + CO is not necessarily zero. -/
theorem vector_sum_not_necessarily_zero (O A B C : V) :
  ¬ (∀ (O A B C : V), O + A + C + B + O + C + O = 0) :=
by sorry

end vector_sum_not_necessarily_zero_l597_59727


namespace certain_number_value_l597_59788

theorem certain_number_value : 
  ∀ x : ℝ,
  (28 + x + 42 + 78 + 104) / 5 = 62 →
  ∃ y : ℝ,
  (y + 62 + 98 + 124 + x) / 5 = 78 ∧
  y = 106 :=
by
  sorry

end certain_number_value_l597_59788


namespace trigonometric_expression_simplification_l597_59743

theorem trigonometric_expression_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.cos (10 * π / 180) = 8 * Real.sqrt 3 / 3 := by
  sorry

end trigonometric_expression_simplification_l597_59743


namespace power_function_increasing_l597_59742

theorem power_function_increasing (m : ℝ) : 
  (∀ x > 0, Monotone (fun x => (m^2 - 4*m + 1) * x^(m^2 - 2*m - 3))) → m = 4 := by
  sorry

end power_function_increasing_l597_59742


namespace projection_difference_l597_59787

/-- Represents a projection type -/
inductive ProjectionType
| Parallel
| Central

/-- Represents the behavior of projection lines -/
inductive ProjectionLineBehavior
| Parallel
| Converging

/-- Defines the projection line behavior for a given projection type -/
def projectionLineBehavior (p : ProjectionType) : ProjectionLineBehavior :=
  match p with
  | ProjectionType.Parallel => ProjectionLineBehavior.Parallel
  | ProjectionType.Central => ProjectionLineBehavior.Converging

/-- Theorem stating the difference between parallel and central projections -/
theorem projection_difference :
  ∀ (p : ProjectionType),
    (p = ProjectionType.Parallel ∧ projectionLineBehavior p = ProjectionLineBehavior.Parallel) ∨
    (p = ProjectionType.Central ∧ projectionLineBehavior p = ProjectionLineBehavior.Converging) :=
by sorry

end projection_difference_l597_59787


namespace bottles_not_in_crate_l597_59728

/-- Given the number of bottles per crate, total bottles, and number of crates,
    calculate the number of bottles that will not be placed in a crate. -/
theorem bottles_not_in_crate
  (bottles_per_crate : ℕ)
  (total_bottles : ℕ)
  (num_crates : ℕ)
  (h1 : bottles_per_crate = 12)
  (h2 : total_bottles = 130)
  (h3 : num_crates = 10) :
  total_bottles - (bottles_per_crate * num_crates) = 10 :=
by sorry

end bottles_not_in_crate_l597_59728


namespace arithmetic_sequence_a20_l597_59766

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a20 (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 3 + a 5 = 105 →
  a 2 + a 4 + a 6 = 99 →
  a 20 = 1 := by
sorry

end arithmetic_sequence_a20_l597_59766


namespace max_projection_area_specific_tetrahedron_l597_59786

/-- Represents a tetrahedron with two adjacent equilateral triangular faces --/
structure Tetrahedron where
  side_length : ℝ
  dihedral_angle : ℝ

/-- Calculates the maximum projection area of a rotating tetrahedron --/
def max_projection_area (t : Tetrahedron) : ℝ :=
  sorry

/-- The theorem stating the maximum projection area of the specific tetrahedron --/
theorem max_projection_area_specific_tetrahedron :
  let t : Tetrahedron := { side_length := 1, dihedral_angle := π / 3 }
  max_projection_area t = Real.sqrt 3 / 4 := by
  sorry

end max_projection_area_specific_tetrahedron_l597_59786


namespace least_common_multiple_first_ten_l597_59717

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ j : ℕ, j ≤ 10 ∧ j > 0 ∧ m % j ≠ 0) ∧
  n = 2520 :=
by sorry

end least_common_multiple_first_ten_l597_59717


namespace radio_price_rank_l597_59796

theorem radio_price_rank (n : ℕ) (prices : Finset ℕ) (radio_price : ℕ) :
  n = 58 →
  prices.card = n + 1 →
  radio_price ∈ prices →
  (∀ p ∈ prices, p ≤ radio_price) →
  (prices.filter (λ p => p < radio_price)).card = 41 →
  (prices.filter (λ p => p ≤ radio_price)).card = n + 1 :=
by
  sorry

end radio_price_rank_l597_59796


namespace min_perimeter_rectangle_l597_59724

/-- The minimum perimeter of a rectangle with area 100 is 40 -/
theorem min_perimeter_rectangle (x y : ℝ) (h : x * y = 100) :
  2 * (x + y) ≥ 40 := by
  sorry

end min_perimeter_rectangle_l597_59724


namespace price_decrease_l597_59738

theorem price_decrease (original_price : ℝ) (decreased_price : ℝ) : 
  decreased_price = original_price * (1 - 0.24) ∧ decreased_price = 608 → 
  original_price = 800 := by
sorry

end price_decrease_l597_59738


namespace sixth_grade_total_l597_59798

theorem sixth_grade_total (girls boys : ℕ) : 
  girls = boys + 2 →
  girls / 11 + 22 = (girls - girls / 11) / 2 + 22 →
  girls + boys = 86 :=
by sorry

end sixth_grade_total_l597_59798


namespace inequality_conversions_l597_59752

theorem inequality_conversions (x : ℝ) : 
  ((5 * x > 4 * x - 1) ↔ (x > -1)) ∧ 
  ((-x - 2 < 7) ↔ (x > -9)) := by
  sorry

end inequality_conversions_l597_59752


namespace value_of_k_l597_59757

theorem value_of_k (a b k : ℝ) (h1 : 2 * a = k) (h2 : 3 * b = k) (h3 : k ≠ 1) (h4 : 2 * a + b = a * b) : k = 18 := by
  sorry

end value_of_k_l597_59757


namespace equality_proof_l597_59700

theorem equality_proof (n : ℕ) (h : n ≥ 2) :
  n * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) :=
by sorry

end equality_proof_l597_59700


namespace sector_central_angle_l597_59776

/-- Given a sector with arc length 4 cm and area 4 cm², prove that its central angle is 2 radians. -/
theorem sector_central_angle (r : ℝ) (θ : ℝ) : 
  r * θ = 4 → (1/2) * r^2 * θ = 4 → θ = 2 := by
  sorry

end sector_central_angle_l597_59776


namespace unknown_bill_value_is_five_l597_59739

/-- Represents the value of a US dollar bill -/
inductive USBill
| One
| Two
| Five
| Ten
| Twenty
| Fifty
| Hundred

/-- The wallet contents before purchase -/
structure Wallet where
  twenties : Nat
  unknown_bills : Nat
  unknown_bill_value : USBill
  loose_coins : Rat

def Wallet.total_value (w : Wallet) : Rat :=
  20 * w.twenties + 
  (match w.unknown_bill_value with
   | USBill.One => 1
   | USBill.Two => 2
   | USBill.Five => 5
   | USBill.Ten => 10
   | USBill.Twenty => 20
   | USBill.Fifty => 50
   | USBill.Hundred => 100) * w.unknown_bills +
  w.loose_coins

theorem unknown_bill_value_is_five (w : Wallet) (h1 : w.twenties = 2) 
  (h2 : w.loose_coins = 9/2) (h3 : Wallet.total_value w - 35/2 = 42) :
  w.unknown_bill_value = USBill.Five := by
  sorry

end unknown_bill_value_is_five_l597_59739


namespace removed_term_is_16th_l597_59735

/-- Sum of the first n terms of the sequence -/
def S (n : ℕ) : ℕ := 2 * n^2 - n

/-- The k-th term of the sequence -/
def a (k : ℕ) : ℕ := 4 * k - 3

theorem removed_term_is_16th :
  ∀ k : ℕ,
  (S 21 - a k = 40 * 20) →
  k = 16 := by
sorry

end removed_term_is_16th_l597_59735


namespace direct_variation_problem_l597_59779

theorem direct_variation_problem (k : ℝ) :
  (∀ x y : ℝ, 5 * y = k * x^2) →
  (5 * 8 = k * 2^2) →
  (5 * 32 = k * 4^2) :=
by
  sorry

end direct_variation_problem_l597_59779


namespace smallest_integer_solution_eight_satisfies_smallest_integer_is_eight_l597_59793

theorem smallest_integer_solution : ∀ x : ℤ, x < 3 * x - 15 → x ≥ 8 :=
by
  sorry

theorem eight_satisfies : (8 : ℤ) < 3 * 8 - 15 :=
by
  sorry

theorem smallest_integer_is_eight : 
  (∀ x : ℤ, x < 3 * x - 15 → x ≥ 8) ∧ (8 < 3 * 8 - 15) :=
by
  sorry

end smallest_integer_solution_eight_satisfies_smallest_integer_is_eight_l597_59793


namespace jills_lavender_candles_l597_59761

/-- Represents the number of candles made with each scent -/
structure CandleCounts where
  lavender : ℕ
  coconut : ℕ
  almond : ℕ
  jasmine : ℕ

/-- Represents the amount of scent required for each candle -/
structure ScentRequirements where
  lavender : ℕ
  coconut : ℕ
  almond : ℕ
  jasmine : ℕ

/-- Theorem stating the number of lavender candles Jill made -/
theorem jills_lavender_candles 
  (req : ScentRequirements)
  (counts : CandleCounts)
  (h1 : req.lavender = 10)
  (h2 : req.coconut = 8)
  (h3 : req.almond = 12)
  (h4 : req.jasmine = 14)
  (h5 : counts.lavender = 3 * counts.coconut)
  (h6 : counts.almond = 2 * counts.jasmine)
  (h7 : counts.almond = 10)
  (h8 : req.coconut * counts.coconut = (5/2) * req.almond * counts.almond)
  : counts.lavender = 111 := by
  sorry

#check jills_lavender_candles

end jills_lavender_candles_l597_59761


namespace goose_eggs_theorem_l597_59789

theorem goose_eggs_theorem (total_eggs : ℕ) : 
  (1 : ℚ) / 4 * (4 : ℚ) / 5 * (3 : ℚ) / 5 * total_eggs = 120 →
  total_eggs = 1000 := by
  sorry

end goose_eggs_theorem_l597_59789


namespace quadratic_roots_imply_c_value_l597_59770

theorem quadratic_roots_imply_c_value (c : ℝ) :
  (∀ x : ℝ, x^2 + 5*x + c = 0 ↔ x = (-5 + Real.sqrt c) / 2 ∨ x = (-5 - Real.sqrt c) / 2) →
  c = 5 := by
sorry

end quadratic_roots_imply_c_value_l597_59770


namespace rings_cost_theorem_l597_59710

/-- The cost of one ring in dollars -/
def cost_per_ring : ℕ := 24

/-- The number of index fingers a person has -/
def index_fingers_per_person : ℕ := 2

/-- The total cost for buying rings for all index fingers of a person -/
def total_cost : ℕ := cost_per_ring * index_fingers_per_person

theorem rings_cost_theorem : total_cost = 48 := by
  sorry

end rings_cost_theorem_l597_59710


namespace factorization_valid_l597_59726

theorem factorization_valid (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorization_valid_l597_59726


namespace cube_root_equation_solution_l597_59762

theorem cube_root_equation_solution :
  ∃ x : ℝ, (x ≠ 0 ∧ (5 - 2/x)^(1/3) = -3) → x = 1/16 := by
  sorry

end cube_root_equation_solution_l597_59762


namespace wonderful_class_size_l597_59736

/-- Represents the number of students in Mrs. Wonderful's class -/
def class_size : ℕ := 18

/-- Represents the number of girls in the class -/
def girls : ℕ := class_size / 2 - 2

/-- Represents the number of boys in the class -/
def boys : ℕ := girls + 4

/-- The total number of jelly beans Mrs. Wonderful brought -/
def total_jelly_beans : ℕ := 420

/-- The number of jelly beans left after distribution -/
def remaining_jelly_beans : ℕ := 6

/-- Theorem stating that the given conditions result in 18 students -/
theorem wonderful_class_size : 
  (3 * girls * girls + 2 * boys * boys = total_jelly_beans - remaining_jelly_beans) ∧
  (boys = girls + 4) ∧
  (class_size = girls + boys) := by sorry

end wonderful_class_size_l597_59736


namespace dexter_card_count_l597_59774

/-- The number of boxes filled with basketball cards -/
def basketball_boxes : ℕ := 9

/-- The number of cards in each basketball box -/
def cards_per_basketball_box : ℕ := 15

/-- The number of cards in each football box -/
def cards_per_football_box : ℕ := 20

/-- The difference in number of boxes between basketball and football cards -/
def box_difference : ℕ := 3

/-- The total number of cards Dexter has -/
def total_cards : ℕ := 
  (basketball_boxes * cards_per_basketball_box) + 
  ((basketball_boxes - box_difference) * cards_per_football_box)

theorem dexter_card_count : total_cards = 255 := by
  sorry

end dexter_card_count_l597_59774


namespace balcony_price_is_eight_l597_59718

/-- Represents the theater ticket sales scenario --/
structure TheaterSales where
  totalTickets : ℕ
  totalCost : ℕ
  orchestraPrice : ℕ
  balconyExcess : ℕ

/-- Calculates the price of a balcony seat given the theater sales data --/
def balconyPrice (sales : TheaterSales) : ℕ :=
  let orchestraTickets := (sales.totalTickets - sales.balconyExcess) / 2
  let balconyTickets := sales.totalTickets - orchestraTickets
  (sales.totalCost - orchestraTickets * sales.orchestraPrice) / balconyTickets

/-- Theorem stating that the balcony price is $8 given the specific sales data --/
theorem balcony_price_is_eight :
  balconyPrice ⟨370, 3320, 12, 190⟩ = 8 := by
  sorry

#eval balconyPrice ⟨370, 3320, 12, 190⟩

end balcony_price_is_eight_l597_59718


namespace carrot_juice_distribution_l597_59763

-- Define the set of glass volumes
def glassVolumes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Define a type for a distribution of glasses
def Distribution := List (List ℕ)

-- Define the property of a valid distribution
def isValidDistribution (d : Distribution) : Prop :=
  d.length = 3 ∧
  d.all (fun l => l.length = 3) ∧
  d.all (fun l => l.sum = 15) ∧
  d.join.toFinset = glassVolumes.toFinset

-- State the theorem
theorem carrot_juice_distribution :
  ∃ (d1 d2 : Distribution),
    isValidDistribution d1 ∧
    isValidDistribution d2 ∧
    d1 ≠ d2 ∧
    ∀ (d : Distribution), isValidDistribution d → (d = d1 ∨ d = d2) :=
  sorry

end carrot_juice_distribution_l597_59763


namespace money_left_after_candy_purchase_l597_59714

def lollipop_cost : ℚ := 1.5
def gummy_pack_cost : ℚ := 2
def lollipop_count : ℕ := 4
def gummy_pack_count : ℕ := 2
def initial_money : ℚ := 15

def total_spent : ℚ := lollipop_cost * lollipop_count + gummy_pack_cost * gummy_pack_count

theorem money_left_after_candy_purchase : 
  initial_money - total_spent = 5 := by
  sorry

end money_left_after_candy_purchase_l597_59714


namespace cupcake_milk_calculation_l597_59702

/-- The number of cupcakes in a full recipe -/
def full_recipe_cupcakes : ℕ := 24

/-- The number of quarts of milk needed for a full recipe -/
def full_recipe_quarts : ℕ := 3

/-- The number of pints in a quart -/
def pints_per_quart : ℕ := 2

/-- The number of cupcakes we want to make -/
def target_cupcakes : ℕ := 6

/-- The amount of milk in pints needed for the target number of cupcakes -/
def milk_needed : ℚ := 1.5

theorem cupcake_milk_calculation :
  (target_cupcakes : ℚ) * (full_recipe_quarts * pints_per_quart : ℚ) / full_recipe_cupcakes = milk_needed :=
sorry

end cupcake_milk_calculation_l597_59702


namespace chair_price_l597_59754

theorem chair_price (total_cost : ℕ) (num_desks : ℕ) (num_chairs : ℕ) (desk_price : ℕ) :
  total_cost = 1236 →
  num_desks = 5 →
  num_chairs = 8 →
  desk_price = 180 →
  (total_cost - num_desks * desk_price) / num_chairs = 42 := by
  sorry

end chair_price_l597_59754


namespace trivia_team_points_per_member_l597_59732

theorem trivia_team_points_per_member 
  (total_members : ℝ) 
  (absent_members : ℝ) 
  (total_points : ℝ) 
  (h1 : total_members = 5.0) 
  (h2 : absent_members = 2.0) 
  (h3 : total_points = 6.0) : 
  total_points / (total_members - absent_members) = 2.0 := by
sorry

end trivia_team_points_per_member_l597_59732


namespace cloth_coloring_problem_l597_59749

/-- Represents the work done by a group of men coloring cloth -/
structure ClothColoring where
  men : ℕ
  days : ℝ
  length : ℝ

/-- The problem statement -/
theorem cloth_coloring_problem (group1 group2 : ClothColoring) :
  group1.men = 4 ∧
  group1.days = 2 ∧
  group2.men = 5 ∧
  group2.days = 1.2 ∧
  group2.length = 36 ∧
  group1.men * group1.days * group1.length = group2.men * group2.days * group2.length →
  group1.length = 27 := by
  sorry

end cloth_coloring_problem_l597_59749


namespace points_form_hyperbola_l597_59737

/-- The set of points (x,y) defined by x = 2cosh(t) and y = 4sinh(t) for real t forms a hyperbola -/
theorem points_form_hyperbola :
  ∀ (t x y : ℝ), x = 2 * Real.cosh t ∧ y = 4 * Real.sinh t →
  x^2 / 4 - y^2 / 16 = 1 := by
sorry

end points_form_hyperbola_l597_59737


namespace coprime_sequence_solution_l597_59778

/-- Represents the sequence of ones and twos constructed from multiples of a, b, and c -/
def constructSequence (a b c : ℕ) : List ℕ := sorry

/-- Checks if two natural numbers are coprime -/
def areCoprime (x y : ℕ) : Prop := Nat.gcd x y = 1

theorem coprime_sequence_solution :
  ∀ a b c : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 →
    areCoprime a b ∧ areCoprime b c ∧ areCoprime a c →
    (let seq := constructSequence a b c
     seq.count 1 = 356 ∧ 
     seq.count 2 = 36 ∧
     seq.take 16 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2]) →
    a = 7 ∧ b = 9 ∧ c = 23 := by
  sorry

end coprime_sequence_solution_l597_59778


namespace math_books_probability_math_books_probability_is_one_sixth_l597_59722

/-- The probability of selecting 2 math books from a shelf with 2 math books and 2 physics books -/
theorem math_books_probability : ℚ :=
  let total_books : ℕ := 4
  let math_books : ℕ := 2
  let books_to_pick : ℕ := 2
  let total_combinations := Nat.choose total_books books_to_pick
  let favorable_combinations := Nat.choose math_books books_to_pick
  (favorable_combinations : ℚ) / total_combinations

/-- The probability of selecting 2 math books from a shelf with 2 math books and 2 physics books is 1/6 -/
theorem math_books_probability_is_one_sixth : math_books_probability = 1 / 6 := by
  sorry

end math_books_probability_math_books_probability_is_one_sixth_l597_59722


namespace work_completion_time_l597_59765

/-- Proves that given two people who can complete a task in 4 days together, 
    and one of them can complete it in 12 days alone, 
    the other person can complete the task in 24 days alone. -/
theorem work_completion_time 
  (joint_time : ℝ) 
  (person1_time : ℝ) 
  (h1 : joint_time = 4) 
  (h2 : person1_time = 12) : 
  ∃ person2_time : ℝ, 
    person2_time = 24 ∧ 
    1 / joint_time = 1 / person1_time + 1 / person2_time :=
by
  sorry

end work_completion_time_l597_59765


namespace tangent_line_passes_through_fixed_point_l597_59783

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line l
def line_l (x : ℝ) : Prop := x = 2

-- Define a point P on line l
def point_P (t : ℝ) : ℝ × ℝ := (2, t)

-- Define the equation of the common chord AB
def common_chord (t x y : ℝ) : Prop := 2*x + t*y = 1

-- Theorem statement
theorem tangent_line_passes_through_fixed_point :
  ∀ t : ℝ, ∃ A B : ℝ × ℝ,
    circle_C A.1 A.2 ∧
    circle_C B.1 B.2 ∧
    common_chord t A.1 A.2 ∧
    common_chord t B.1 B.2 ∧
    common_chord t (1/2) 0 :=
sorry

end tangent_line_passes_through_fixed_point_l597_59783


namespace salary_increase_proof_l597_59712

def original_salary : ℝ := 60
def percentage_increase : ℝ := 13.333333333333334
def new_salary : ℝ := 68

theorem salary_increase_proof :
  original_salary * (1 + percentage_increase / 100) = new_salary := by
  sorry

end salary_increase_proof_l597_59712


namespace solve_comic_problem_l597_59734

def comic_problem (pages_per_comic : ℕ) (found_pages : ℕ) (total_comics : ℕ) : Prop :=
  let repaired_comics := found_pages / pages_per_comic
  let untorn_comics := total_comics - repaired_comics
  untorn_comics = 5

theorem solve_comic_problem :
  comic_problem 25 150 11 := by
  sorry

end solve_comic_problem_l597_59734


namespace parallel_lines_angle_l597_59745

/-- Two lines are parallel -/
def parallel (l m : Set (ℝ × ℝ)) : Prop := sorry

/-- A point lies on a line -/
def on_line (P : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop := sorry

/-- Angle measure in degrees -/
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

theorem parallel_lines_angle (l m t : Set (ℝ × ℝ)) (P Q C : ℝ × ℝ) :
  parallel l m →
  on_line P l →
  on_line Q m →
  on_line P t →
  on_line Q t →
  on_line C m →
  angle_measure P Q C = 50 →
  angle_measure A P Q = 130 →
  true := by sorry

end parallel_lines_angle_l597_59745


namespace least_positive_difference_l597_59794

def geometric_sequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => r * geometric_sequence a₁ r n

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

def sequence_A : ℕ → ℝ := geometric_sequence 3 2

def sequence_B : ℕ → ℝ := arithmetic_sequence 15 30

def valid_term_A (n : ℕ) : Prop := sequence_A n ≤ 300

def valid_term_B (n : ℕ) : Prop := sequence_B n ≤ 300

theorem least_positive_difference :
  ∃ (m n : ℕ), valid_term_A m ∧ valid_term_B n ∧
    ∀ (i j : ℕ), valid_term_A i → valid_term_B j →
      |sequence_A m - sequence_B n| ≤ |sequence_A i - sequence_B j| ∧
      |sequence_A m - sequence_B n| = 3 :=
sorry

end least_positive_difference_l597_59794


namespace all_divisors_of_30240_l597_59792

theorem all_divisors_of_30240 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 9 → 30240 % n = 0 := by
  sorry

end all_divisors_of_30240_l597_59792


namespace complex_number_value_l597_59713

theorem complex_number_value (z : ℂ) (h : z * Complex.I = -1 + Complex.I) : z = 1 + Complex.I := by
  sorry

end complex_number_value_l597_59713


namespace stream_speed_l597_59791

/-- Proves that given a man's swimming speed in still water and the relationship
    between upstream and downstream swimming times, the speed of the stream is 0.5 km/h. -/
theorem stream_speed (swimming_speed : ℝ) (upstream_time_ratio : ℝ) :
  swimming_speed = 1.5 →
  upstream_time_ratio = 2 →
  ∃ (stream_speed : ℝ),
    (swimming_speed + stream_speed) * 1 = (swimming_speed - stream_speed) * upstream_time_ratio ∧
    stream_speed = 0.5 := by
  sorry

end stream_speed_l597_59791


namespace m_range_l597_59756

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the condition that m is positive
def m_positive (m : ℝ) : Prop := m > 0

-- Define the condition about the relationship between p and q
def condition (m : ℝ) : Prop := 
  ∀ x, (¬(p x) → ¬(q x m)) ∧ ∃ x, (¬(p x) ∧ q x m)

-- State the theorem
theorem m_range (m : ℝ) : 
  m_positive m → condition m → m ≥ 9 :=
sorry

end m_range_l597_59756


namespace max_yellow_apples_max_total_apples_l597_59773

/-- Represents the number of apples of each color in the basket -/
structure Basket :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Represents the number of apples Alyona has taken -/
structure TakenApples :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Checks if Alyona should stop taking apples -/
def shouldStop (taken : TakenApples) : Prop :=
  taken.green < taken.yellow ∧ taken.yellow < taken.red

/-- The initial state of the basket -/
def initialBasket : Basket :=
  { green := 10, yellow := 13, red := 18 }

/-- Theorem stating the maximum number of yellow apples Alyona can take -/
theorem max_yellow_apples :
  ∃ (taken : TakenApples),
    taken.yellow = initialBasket.yellow ∧
    taken.yellow ≤ initialBasket.yellow ∧
    ¬(shouldStop taken) ∧
    ∀ (other : TakenApples),
      other.yellow > taken.yellow →
      shouldStop other ∨ other.yellow > initialBasket.yellow :=
sorry

/-- Theorem stating the maximum total number of apples Alyona can take -/
theorem max_total_apples :
  ∃ (taken : TakenApples),
    taken.green + taken.yellow + taken.red = 39 ∧
    taken.green ≤ initialBasket.green ∧
    taken.yellow ≤ initialBasket.yellow ∧
    taken.red ≤ initialBasket.red ∧
    ¬(shouldStop taken) ∧
    ∀ (other : TakenApples),
      other.green + other.yellow + other.red > 39 →
      shouldStop other ∨
      other.green > initialBasket.green ∨
      other.yellow > initialBasket.yellow ∨
      other.red > initialBasket.red :=
sorry

end max_yellow_apples_max_total_apples_l597_59773


namespace complementary_angle_adjustment_l597_59758

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary (a b : ℝ) : Prop := a + b = 90

/-- The ratio of two real numbers is 4:5 -/
def ratio_4_to_5 (a b : ℝ) : Prop := 5 * a = 4 * b

theorem complementary_angle_adjustment (a b : ℝ) 
  (h1 : complementary a b) 
  (h2 : ratio_4_to_5 a b) :
  complementary (1.1 * a) (0.92 * b) := by
  sorry

end complementary_angle_adjustment_l597_59758


namespace optimal_feeding_program_l597_59740

/-- Represents a feeding program for animals -/
structure FeedingProgram where
  x : ℝ  -- Amount of first feed in kg
  y : ℝ  -- Amount of second feed in kg

/-- Nutrient requirements for each animal per day -/
def nutrientRequirements : ℝ × ℝ × ℝ := (45, 60, 5)

/-- Nutrient content of first feed per kg -/
def firstFeedContent : ℝ × ℝ := (10, 10)

/-- Nutrient content of second feed per kg -/
def secondFeedContent : ℝ × ℝ × ℝ := (10, 20, 5)

/-- Cost of feeds in Ft/q -/
def feedCosts : ℝ × ℝ := (30, 120)

/-- Feeding loss percentages -/
def feedingLoss : ℝ × ℝ := (0.1, 0.2)

/-- Check if a feeding program satisfies nutrient requirements -/
def satisfiesRequirements (fp : FeedingProgram) : Prop :=
  let (reqA, reqB, reqC) := nutrientRequirements
  let (firstA, firstB) := firstFeedContent
  let (secondA, secondB, secondC) := secondFeedContent
  firstA * fp.x + secondA * fp.y ≥ reqA ∧
  firstB * fp.x + secondB * fp.y ≥ reqB ∧
  secondC * fp.y ≥ reqC

/-- Calculate the cost of a feeding program -/
def calculateCost (fp : FeedingProgram) : ℝ :=
  let (costFirst, costSecond) := feedCosts
  costFirst * fp.x + costSecond * fp.y

/-- Calculate the feeding loss of a feeding program -/
def calculateLoss (fp : FeedingProgram) : ℝ :=
  let (lossFirst, lossSecond) := feedingLoss
  lossFirst * fp.x + lossSecond * fp.y

/-- Theorem stating that (4, 1) is the optimal feeding program -/
theorem optimal_feeding_program :
  let optimalProgram := FeedingProgram.mk 4 1
  satisfiesRequirements optimalProgram ∧
  ∀ fp : FeedingProgram, satisfiesRequirements fp →
    calculateCost optimalProgram ≤ calculateCost fp ∧
    calculateLoss optimalProgram ≤ calculateLoss fp :=
by sorry

end optimal_feeding_program_l597_59740


namespace solution_set_when_a_is_2_range_of_a_when_f_geq_4_l597_59768

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} := by sorry

-- Part 2
theorem range_of_a_when_f_geq_4 :
  ∀ x a : ℝ, f x a ≥ 4 → a ≤ -1 ∨ a ≥ 3 := by sorry

end solution_set_when_a_is_2_range_of_a_when_f_geq_4_l597_59768


namespace smallest_b_for_factorization_l597_59771

theorem smallest_b_for_factorization : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b*x + 1764 = (x + p) * (x + q)) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ¬(∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b'*x + 1764 = (x + p) * (x + q))) ∧
  b = 84 :=
by sorry

end smallest_b_for_factorization_l597_59771


namespace cosine_sum_theorem_l597_59719

theorem cosine_sum_theorem (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos y + Real.cos z = 1)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 0) : 
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 1 := by
  sorry

end cosine_sum_theorem_l597_59719


namespace girls_multiple_of_five_l597_59775

/-- Represents the number of students in a group -/
structure GroupComposition :=
  (boys : ℕ)
  (girls : ℕ)

/-- Checks if a given number of boys and girls can be divided into the specified number of groups -/
def canDivideIntoGroups (totalBoys totalGirls groups : ℕ) : Prop :=
  ∃ (composition : GroupComposition),
    composition.boys * groups = totalBoys ∧
    composition.girls * groups = totalGirls

theorem girls_multiple_of_five (totalBoys totalGirls : ℕ) :
  totalBoys = 10 →
  canDivideIntoGroups totalBoys totalGirls 5 →
  ∃ (k : ℕ), totalGirls = 5 * k ∧ k ≥ 1 :=
by sorry

end girls_multiple_of_five_l597_59775


namespace boys_can_be_truthful_l597_59781

/-- Represents the possible grades a student can receive -/
inductive Grade
  | Three
  | Four
  | Five

/-- Compares two grades -/
def Grade.gt (a b : Grade) : Prop :=
  match a, b with
  | Five, Three => True
  | Five, Four => True
  | Four, Three => True
  | _, _ => False

/-- Represents the grades of a student for three tests -/
structure StudentGrades :=
  (test1 : Grade)
  (test2 : Grade)
  (test3 : Grade)

/-- Checks if one student has higher grades than another for at least two tests -/
def higherGradesInTwoTests (a b : StudentGrades) : Prop :=
  (a.test1.gt b.test1 ∧ a.test2.gt b.test2) ∨
  (a.test1.gt b.test1 ∧ a.test3.gt b.test3) ∨
  (a.test2.gt b.test2 ∧ a.test3.gt b.test3)

/-- The main theorem stating that there exists a set of grades satisfying all conditions -/
theorem boys_can_be_truthful :
  ∃ (valera seryozha dima : StudentGrades),
    higherGradesInTwoTests valera seryozha ∧
    higherGradesInTwoTests seryozha dima ∧
    higherGradesInTwoTests dima valera :=
  sorry

end boys_can_be_truthful_l597_59781


namespace nine_candies_four_bags_l597_59747

/-- The number of ways to distribute distinct candies among bags --/
def distribute_candies (num_candies : ℕ) (num_bags : ℕ) : ℕ :=
  num_bags ^ (num_candies - num_bags)

/-- Theorem stating the number of ways to distribute 9 distinct candies among 4 bags --/
theorem nine_candies_four_bags : 
  distribute_candies 9 4 = 1024 :=
sorry

end nine_candies_four_bags_l597_59747


namespace product_of_roots_equation_l597_59744

theorem product_of_roots_equation (y : ℝ) (h1 : y > 0) 
  (h2 : Real.sqrt (5 * y) * Real.sqrt (15 * y) * Real.sqrt (2 * y) * Real.sqrt (6 * y) = 6) : 
  y = 1 / Real.sqrt 5 := by
sorry

end product_of_roots_equation_l597_59744


namespace sarah_reading_time_l597_59746

/-- Calculates the reading time in hours for a given number of books -/
def reading_time (words_per_minute : ℕ) (words_per_page : ℕ) (pages_per_book : ℕ) (num_books : ℕ) : ℕ :=
  let total_words := words_per_page * pages_per_book * num_books
  let total_minutes := total_words / words_per_minute
  total_minutes / 60

/-- Theorem stating that Sarah's reading time for 6 books is 20 hours -/
theorem sarah_reading_time :
  reading_time 40 100 80 6 = 20 := by
  sorry

end sarah_reading_time_l597_59746


namespace circle_omega_area_l597_59767

/-- Circle ω with points A and B, and tangent lines intersecting on x-axis -/
structure Circle_omega where
  /-- Point A on the circle -/
  A : ℝ × ℝ
  /-- Point B on the circle -/
  B : ℝ × ℝ
  /-- The tangent lines at A and B intersect on the x-axis -/
  tangent_intersection_on_x_axis : Prop

/-- Theorem: Area of circle ω is 120375π/9600 -/
theorem circle_omega_area (ω : Circle_omega) 
  (h1 : ω.A = (5, 15)) 
  (h2 : ω.B = (13, 9)) : 
  ∃ (r : ℝ), r > 0 ∧ π * r^2 = 120375 * π / 9600 := by
  sorry

end circle_omega_area_l597_59767


namespace inequality_proof_l597_59705

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (1/2) * ((a - b)^2 + (b - c)^2 + (c - a)^2) ≤ a^2 + b^2 + c^2 - 3 * (a^2 * b^2 * c^2)^(1/3) ∧
  a^2 + b^2 + c^2 - 3 * (a^2 * b^2 * c^2)^(1/3) ≤ (a - b)^2 + (b - c)^2 + (c - a)^2 := by
sorry

end inequality_proof_l597_59705


namespace allison_wins_probability_l597_59759

def allison_cube : Fin 6 → ℕ := λ _ => 6

def brian_cube : Fin 6 → ℕ := λ i => i.val + 1

def noah_cube : Fin 6 → ℕ := λ i => if i.val < 3 then 3 else 5

def prob_brian_less_than_6 : ℚ := 5 / 6

def prob_noah_less_than_6 : ℚ := 1

theorem allison_wins_probability :
  (prob_brian_less_than_6 * prob_noah_less_than_6 : ℚ) = 5 / 6 := by sorry

end allison_wins_probability_l597_59759


namespace train_length_calculation_l597_59716

/-- The length of a train given its speed, a man's walking speed, and the time it takes to cross the man. -/
theorem train_length_calculation (train_speed man_speed : ℝ) (crossing_time : ℝ) :
  train_speed = 63 →
  man_speed = 3 →
  crossing_time = 41.9966402687785 →
  ∃ (train_length : ℝ), abs (train_length - 700) < 0.1 :=
by
  sorry

end train_length_calculation_l597_59716


namespace cos_angle_between_vectors_l597_59720

/-- Given vectors a and b in ℝ², where a = (2, 1) and a + 2b = (4, 5),
    the cosine of the angle between a and b is equal to 4/5. -/
theorem cos_angle_between_vectors (a b : ℝ × ℝ) :
  a = (2, 1) →
  a + 2 • b = (4, 5) →
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = 4/5 := by
  sorry

end cos_angle_between_vectors_l597_59720


namespace age_difference_equals_first_ratio_l597_59784

/-- Represents the age ratio of four siblings -/
structure AgeRatio :=
  (a b c d : ℕ)

/-- Calculates the age difference between the first two siblings given their age ratio and total future age -/
def ageDifference (ratio : AgeRatio) (totalFutureAge : ℕ) : ℚ :=
  let x : ℚ := (totalFutureAge - 20 : ℚ) / (ratio.a + ratio.b + ratio.c + ratio.d : ℚ)
  ratio.a * x - ratio.b * x

/-- Theorem: The age difference between the first two siblings is equal to the first number in the ratio -/
theorem age_difference_equals_first_ratio 
  (ratio : AgeRatio) 
  (totalFutureAge : ℕ) 
  (h1 : ratio.a = 4) 
  (h2 : ratio.b = 3) 
  (h3 : ratio.c = 7) 
  (h4 : ratio.d = 5) 
  (h5 : totalFutureAge = 230) : 
  ageDifference ratio totalFutureAge = ratio.a := by
  sorry

#eval ageDifference ⟨4, 3, 7, 5⟩ 230

end age_difference_equals_first_ratio_l597_59784


namespace tire_rotation_mileage_l597_59797

theorem tire_rotation_mileage (total_tires : ℕ) (simultaneous_tires : ℕ) (total_miles : ℕ) :
  total_tires = 7 →
  simultaneous_tires = 6 →
  total_miles = 42000 →
  (total_miles * simultaneous_tires) / total_tires = 36000 :=
by sorry

end tire_rotation_mileage_l597_59797


namespace unique_number_satisfying_conditions_l597_59753

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + d + 10 * m

theorem unique_number_satisfying_conditions :
  ∃! n : ℕ,
    Odd n ∧
    contains_digit n 5 ∧
    3 ∣ n ∧
    12^2 < n ∧
    n < 13^2 :=
by sorry

end unique_number_satisfying_conditions_l597_59753


namespace batsman_inning_number_l597_59731

/-- Represents the batting statistics of a cricket player -/
structure BattingStats where
  totalRuns : ℕ
  innings : ℕ
  average : ℚ

/-- Calculates the new average after adding runs to the existing stats -/
def newAverage (stats : BattingStats) (newRuns : ℕ) : ℚ :=
  (stats.totalRuns + newRuns) / (stats.innings + 1)

theorem batsman_inning_number (stats : BattingStats) (h1 : newAverage stats 88 = 40)
    (h2 : stats.average = 37) : stats.innings + 1 = 17 := by
  sorry

#check batsman_inning_number

end batsman_inning_number_l597_59731


namespace population_increase_in_one_day_l597_59750

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate (people per 2 seconds) -/
def birth_rate : ℕ := 10

/-- Represents the death rate (people per 2 seconds) -/
def death_rate : ℕ := 2

/-- Calculates the net population increase over one day -/
def net_population_increase : ℕ :=
  (seconds_per_day / 2) * birth_rate - (seconds_per_day / 2) * death_rate

theorem population_increase_in_one_day :
  net_population_increase = 345600 := by sorry

end population_increase_in_one_day_l597_59750


namespace rectangle_perimeter_reduction_l597_59755

theorem rectangle_perimeter_reduction (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  2 * (0.9 * a + 0.8 * b) = 0.88 * 2 * (a + b) → 
  2 * (0.8 * a + 0.9 * b) = 0.82 * 2 * (a + b) :=
by sorry

end rectangle_perimeter_reduction_l597_59755


namespace sum_of_non_visible_numbers_l597_59782

/-- Represents a standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The sum of numbers on a standard six-sided die -/
def sumOfDie : ℕ := 21

/-- The total number of faces on four dice -/
def totalFaces : ℕ := 24

/-- The number of visible faces -/
def visibleFaces : ℕ := 9

/-- The list of visible numbers -/
def visibleNumbers : List ℕ := [1, 2, 3, 3, 4, 5, 5, 6, 6]

/-- The theorem stating the sum of non-visible numbers -/
theorem sum_of_non_visible_numbers :
  (4 * sumOfDie) - (visibleNumbers.sum) = 49 := by sorry

end sum_of_non_visible_numbers_l597_59782


namespace distinct_prime_factors_of_divisor_sum_800_l597_59772

-- Define the sum of positive divisors function
def sum_of_divisors (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem distinct_prime_factors_of_divisor_sum_800 :
  (Nat.factors (sum_of_divisors 800)).length = 4 := by sorry

end distinct_prime_factors_of_divisor_sum_800_l597_59772


namespace absolute_value_not_always_greater_than_zero_l597_59721

theorem absolute_value_not_always_greater_than_zero : 
  ¬ (∀ x : ℝ, |x| > 0) :=
by sorry

end absolute_value_not_always_greater_than_zero_l597_59721


namespace parabola_shift_l597_59760

-- Define the initial parabola
def initial_parabola (x y : ℝ) : Prop :=
  y = -1/3 * (x - 2)^2

-- Define the shift
def shift_right : ℝ := 1
def shift_down : ℝ := 2

-- Define the resulting parabola
def resulting_parabola (x y : ℝ) : Prop :=
  y = -1/3 * (x - 3)^2 - 2

-- Theorem statement
theorem parabola_shift :
  ∀ x y : ℝ, initial_parabola x y →
  resulting_parabola (x + shift_right) (y - shift_down) :=
by sorry

end parabola_shift_l597_59760


namespace steve_earnings_l597_59708

def total_copies : ℕ := 1000000
def advance_copies : ℕ := 100000
def price_per_copy : ℚ := 2
def agent_percentage : ℚ := 1/10

theorem steve_earnings : 
  (total_copies - advance_copies) * price_per_copy * (1 - agent_percentage) = 1620000 := by
  sorry

end steve_earnings_l597_59708


namespace quadratic_equation_solution_l597_59730

theorem quadratic_equation_solution (b : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x - b = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 - 2*y - b = 0 ∧ y = 3) :=
by
  sorry

end quadratic_equation_solution_l597_59730


namespace eighty_one_power_ten_equals_three_power_q_l597_59729

theorem eighty_one_power_ten_equals_three_power_q (q : ℕ) : 81^10 = 3^q → q = 40 := by
  sorry

end eighty_one_power_ten_equals_three_power_q_l597_59729


namespace share_difference_l597_59703

theorem share_difference (total : ℚ) (p m s : ℚ) : 
  total = 730 →
  p + m + s = total →
  4 * p = 3 * m →
  3 * m = 3.5 * s →
  m - s = 36.5 := by
  sorry

end share_difference_l597_59703


namespace texas_integrated_school_students_l597_59777

theorem texas_integrated_school_students (original_classes : ℕ) (students_per_class : ℕ) (new_classes : ℕ) : 
  original_classes = 15 → 
  students_per_class = 20 → 
  new_classes = 5 → 
  (original_classes + new_classes) * students_per_class = 400 := by
sorry

end texas_integrated_school_students_l597_59777


namespace opposite_numbers_fifth_power_sum_l597_59707

theorem opposite_numbers_fifth_power_sum (a b : ℝ) : 
  a + b = 0 → a^5 + b^5 = 0 := by sorry

end opposite_numbers_fifth_power_sum_l597_59707


namespace total_points_in_game_l597_59785

def rounds : ℕ := 177
def points_per_round : ℕ := 46

theorem total_points_in_game : rounds * points_per_round = 8142 := by
  sorry

end total_points_in_game_l597_59785


namespace cricket_runs_total_l597_59799

theorem cricket_runs_total (a b c : ℕ) : 
  (a : ℚ) / b = 1 / 3 →
  (b : ℚ) / c = 1 / 5 →
  c = 75 →
  a + b + c = 95 := by
  sorry

end cricket_runs_total_l597_59799


namespace range_of_m_l597_59715

/-- Given the conditions of P and Q, prove that the range of m is [9, +∞) -/
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (|(4 - x) / 3| ≤ 2 → (x + m - 1) * (x - m - 1) ≤ 0)) ∧
  (∃ x : ℝ, |(4 - x) / 3| > 2 ∧ (x + m - 1) * (x - m - 1) > 0) ∧
  (m > 0) →
  m ≥ 9 := by
  sorry

end range_of_m_l597_59715


namespace cafeteria_earnings_proof_l597_59709

/-- Calculates the earnings from selling fruits in a cafeteria -/
def cafeteria_earnings (initial_apples initial_oranges : ℕ) 
                       (apple_price orange_price : ℚ) 
                       (remaining_apples remaining_oranges : ℕ) : ℚ :=
  let sold_apples := initial_apples - remaining_apples
  let sold_oranges := initial_oranges - remaining_oranges
  sold_apples * apple_price + sold_oranges * orange_price

/-- Proves that the cafeteria earns $49.00 for the given conditions -/
theorem cafeteria_earnings_proof :
  cafeteria_earnings 50 40 0.80 0.50 10 6 = 49.00 := by
  sorry

end cafeteria_earnings_proof_l597_59709


namespace dilute_herbal_essence_l597_59751

/-- Proves that adding 7.5 ounces of water to a 15-ounce solution containing 60% essence
    results in a new solution with 40% essence -/
theorem dilute_herbal_essence :
  let initial_weight : ℝ := 15
  let initial_concentration : ℝ := 0.6
  let final_concentration : ℝ := 0.4
  let water_added : ℝ := 7.5
  let essence_amount : ℝ := initial_weight * initial_concentration
  let final_weight : ℝ := initial_weight + water_added
  essence_amount / final_weight = final_concentration := by sorry

end dilute_herbal_essence_l597_59751


namespace decimal_difference_l597_59795

/- Define the repeating decimal 0.̅72 -/
def repeating_decimal : ℚ := 72 / 99

/- Define the terminating decimal 0.72 -/
def terminating_decimal : ℚ := 72 / 100

/- Theorem statement -/
theorem decimal_difference :
  repeating_decimal - terminating_decimal = 2 / 275 := by
  sorry

end decimal_difference_l597_59795


namespace probability_factor_less_than_seven_l597_59780

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (· ∣ n) (Finset.range (n + 1))

theorem probability_factor_less_than_seven :
  let all_factors := factors 60
  let factors_less_than_seven := all_factors.filter (· < 7)
  (factors_less_than_seven.card : ℚ) / all_factors.card = 1 / 2 := by
sorry

end probability_factor_less_than_seven_l597_59780


namespace gcd_g_y_equals_one_l597_59704

theorem gcd_g_y_equals_one (y : ℤ) (h : ∃ k : ℤ, y = 34567 * k) :
  let g : ℤ → ℤ := λ y => (3*y+4)*(8*y+3)*(14*y+5)*(y+14)
  Int.gcd (g y) y = 1 := by
  sorry

end gcd_g_y_equals_one_l597_59704


namespace list_median_is_106_l597_59711

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def list_length : ℕ := sequence_sum 150

def median_position : ℕ := (list_length + 1) / 2

theorem list_median_is_106 : ∃ (n : ℕ), 
  n = 106 ∧ 
  sequence_sum (n - 1) < median_position ∧ 
  median_position ≤ sequence_sum n :=
sorry

end list_median_is_106_l597_59711


namespace bus_trip_time_calculation_l597_59733

/-- Calculates the new trip time given the original time, original speed, distance increase factor, and new speed -/
def new_trip_time (original_time : ℚ) (original_speed : ℚ) (distance_increase : ℚ) (new_speed : ℚ) : ℚ :=
  (original_time * original_speed * (1 + distance_increase)) / new_speed

/-- Proves that the new trip time is 256/35 hours given the specified conditions -/
theorem bus_trip_time_calculation :
  let original_time : ℚ := 16 / 3  -- 5 1/3 hours
  let original_speed : ℚ := 80
  let distance_increase : ℚ := 1 / 5  -- 20% increase
  let new_speed : ℚ := 70
  new_trip_time original_time original_speed distance_increase new_speed = 256 / 35 := by
  sorry

#eval new_trip_time (16/3) 80 (1/5) 70

end bus_trip_time_calculation_l597_59733


namespace carls_open_house_l597_59769

/-- Carl's open house problem -/
theorem carls_open_house 
  (definite_attendees : ℕ) 
  (potential_attendees : ℕ)
  (extravagant_bags : ℕ)
  (average_bags : ℕ)
  (h1 : definite_attendees = 50)
  (h2 : potential_attendees = 40)
  (h3 : extravagant_bags = 10)
  (h4 : average_bags = 20) :
  definite_attendees + potential_attendees - (extravagant_bags + average_bags) = 60 :=
by sorry

end carls_open_house_l597_59769


namespace triangle_max_area_l597_59701

/-- Given a triangle ABC where a^2 + b^2 + 2c^2 = 8, 
    the maximum area of the triangle is 2√5/5 -/
theorem triangle_max_area (a b c : ℝ) (h : a^2 + b^2 + 2*c^2 = 8) :
  ∃ (S : ℝ), S = (2 * Real.sqrt 5) / 5 ∧ 
  (∀ (S' : ℝ), S' = (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) → S' ≤ S) := by
  sorry


end triangle_max_area_l597_59701


namespace division_result_l597_59706

theorem division_result : (45 : ℝ) / 0.05 = 900 := by sorry

end division_result_l597_59706
