import Mathlib

namespace park_cycling_time_l3481_348145

/-- Proves that for a rectangular park with given specifications, 
    a cyclist completes one round in 8 minutes -/
theorem park_cycling_time (length width : ℝ) (area perimeter : ℝ) (speed : ℝ) :
  width = 4 * length →
  area = length * width →
  area = 102400 →
  perimeter = 2 * (length + width) →
  speed = 12 * 1000 / 60 →
  (perimeter / speed) = 8 :=
by sorry

end park_cycling_time_l3481_348145


namespace fraction_comparison_l3481_348181

theorem fraction_comparison : (1 : ℚ) / 4 = 24999999 / (10^8 : ℚ) + 1 / (4 * 10^8 : ℚ) := by
  sorry

end fraction_comparison_l3481_348181


namespace bob_has_62_pennies_l3481_348174

/-- The number of pennies Alex currently has -/
def alex_pennies : ℕ := sorry

/-- The number of pennies Bob currently has -/
def bob_pennies : ℕ := sorry

/-- If Alex gives Bob two pennies, Bob will have four times as many pennies as Alex has left -/
axiom condition1 : bob_pennies + 2 = 4 * (alex_pennies - 2)

/-- If Bob gives Alex two pennies, Bob will have three times as many pennies as Alex has -/
axiom condition2 : bob_pennies - 2 = 3 * (alex_pennies + 2)

/-- Bob currently has 62 pennies -/
theorem bob_has_62_pennies : bob_pennies = 62 := by sorry

end bob_has_62_pennies_l3481_348174


namespace keaton_yearly_earnings_l3481_348177

/-- Represents Keaton's farm earnings --/
def farm_earnings (orange_harvest_interval : ℕ) (orange_harvest_value : ℕ) 
                  (apple_harvest_interval : ℕ) (apple_harvest_value : ℕ) : ℕ :=
  let orange_harvests_per_year := 12 / orange_harvest_interval
  let apple_harvests_per_year := 12 / apple_harvest_interval
  orange_harvests_per_year * orange_harvest_value + apple_harvests_per_year * apple_harvest_value

/-- Theorem stating Keaton's yearly earnings --/
theorem keaton_yearly_earnings : farm_earnings 2 50 3 30 = 420 := by
  sorry

end keaton_yearly_earnings_l3481_348177


namespace two_hundred_twenty_fifth_number_with_digit_sum_2018_l3481_348146

def digit_sum (n : ℕ) : ℕ := sorry

def nth_number_with_digit_sum (n : ℕ) (sum : ℕ) : ℕ := sorry

theorem two_hundred_twenty_fifth_number_with_digit_sum_2018 :
  nth_number_with_digit_sum 225 2018 = 39 * 10^224 + (10^224 - 10) * 9 + 8 :=
sorry

end two_hundred_twenty_fifth_number_with_digit_sum_2018_l3481_348146


namespace arithmetic_sequence_l3481_348126

def a (n : ℕ) : ℤ := 3 * n + 1

theorem arithmetic_sequence :
  ∀ n : ℕ, a (n + 1) - a n = (3 : ℤ) := by sorry

end arithmetic_sequence_l3481_348126


namespace number_transformation_l3481_348141

theorem number_transformation (initial_number : ℕ) : 
  initial_number = 6 → 3 * ((2 * initial_number) + 9) = 63 := by
  sorry

end number_transformation_l3481_348141


namespace sum_of_cubes_l3481_348165

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -3) : a^3 + b^3 = 26 := by
  sorry

end sum_of_cubes_l3481_348165


namespace snooker_ticket_difference_l3481_348190

/-- Represents the ticket sales for a snooker tournament --/
structure TicketSales where
  vipPrice : ℕ
  regularPrice : ℕ
  totalTickets : ℕ
  totalRevenue : ℕ

/-- Calculates the difference between regular and VIP tickets sold --/
def ticketDifference (sales : TicketSales) : ℕ :=
  let vipTickets := (sales.totalRevenue - sales.regularPrice * sales.totalTickets) / 
                    (sales.vipPrice - sales.regularPrice)
  let regularTickets := sales.totalTickets - vipTickets
  regularTickets - vipTickets

/-- Theorem stating the difference in ticket sales --/
theorem snooker_ticket_difference :
  let sales : TicketSales := {
    vipPrice := 45,
    regularPrice := 20,
    totalTickets := 320,
    totalRevenue := 7500
  }
  ticketDifference sales = 232 := by
  sorry


end snooker_ticket_difference_l3481_348190


namespace domain_of_f_l3481_348180

noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.arccos (Real.sin x))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | ∀ k : ℤ, x ≠ k * Real.pi} :=
sorry

end domain_of_f_l3481_348180


namespace correct_calculation_l3481_348156

theorem correct_calculation (a b : ℝ) : 9 * a^2 * b - 9 * a^2 * b = 0 := by
  sorry

end correct_calculation_l3481_348156


namespace orange_count_l3481_348147

/-- Given the ratio of mangoes : oranges : apples and the number of mangoes and apples,
    calculate the number of oranges -/
theorem orange_count (mango_ratio orange_ratio apple_ratio mango_count apple_count : ℕ) :
  mango_ratio ≠ 0 →
  orange_ratio ≠ 0 →
  apple_ratio ≠ 0 →
  mango_ratio = 10 →
  orange_ratio = 2 →
  apple_ratio = 3 →
  mango_count = 120 →
  apple_count = 36 →
  mango_count / mango_ratio = apple_count / apple_ratio →
  (mango_count / mango_ratio) * orange_ratio = 24 := by
sorry

end orange_count_l3481_348147


namespace acid_concentration_increase_l3481_348175

theorem acid_concentration_increase (initial_volume initial_concentration water_removed : ℝ) :
  initial_volume = 18 →
  initial_concentration = 0.4 →
  water_removed = 6 →
  let acid_amount := initial_volume * initial_concentration
  let final_volume := initial_volume - water_removed
  let final_concentration := acid_amount / final_volume
  final_concentration = 0.6 :=
by sorry

end acid_concentration_increase_l3481_348175


namespace find_d_l3481_348178

theorem find_d (a b c d : ℕ+) 
  (h1 : a.val ^ 2 = c.val * (d.val + 20)) 
  (h2 : b.val ^ 2 = c.val * (d.val - 18)) : 
  d.val = 180 := by
  sorry

end find_d_l3481_348178


namespace trigonometric_equation_l3481_348104

theorem trigonometric_equation (α : Real) 
  (h : (5 * Real.sin α - Real.cos α) / (Real.cos α + Real.sin α) = 1) : 
  Real.tan α = 1/2 ∧ 
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) + Real.sin α * Real.cos α = 17/5 := by
  sorry

end trigonometric_equation_l3481_348104


namespace weight_of_replaced_person_l3481_348129

/-- Given a group of 6 persons where one person is replaced by a new person weighing 79.8 kg,
    and the average weight increases by 1.8 kg, prove that the replaced person weighed 69 kg. -/
theorem weight_of_replaced_person
  (initial_count : ℕ)
  (new_person_weight : ℝ)
  (average_increase : ℝ)
  (h1 : initial_count = 6)
  (h2 : new_person_weight = 79.8)
  (h3 : average_increase = 1.8) :
  ∃ (replaced_weight : ℝ),
    replaced_weight = 69 ∧
    new_person_weight = replaced_weight + (initial_count : ℝ) * average_increase :=
by sorry

end weight_of_replaced_person_l3481_348129


namespace test_probability_l3481_348176

/-- The probability of answering exactly k questions correctly out of n questions,
    where the probability of answering each question correctly is p. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of answering exactly 2 questions correctly out of 6 questions,
    where the probability of answering each question correctly is 1/3, is 240/729. -/
theorem test_probability : binomial_probability 6 2 (1/3) = 240/729 := by
  sorry

end test_probability_l3481_348176


namespace eggs_left_l3481_348131

/-- Given a box with 47 eggs, if Harry takes 5 eggs and Susan takes x eggs,
    then the number of eggs left in the box is equal to 42 - x. -/
theorem eggs_left (x : ℕ) : 47 - 5 - x = 42 - x := by
  sorry

end eggs_left_l3481_348131


namespace quadratic_roots_difference_squared_l3481_348179

theorem quadratic_roots_difference_squared :
  ∀ a b : ℝ,
  (6 * a^2 + 13 * a - 28 = 0) →
  (6 * b^2 + 13 * b - 28 = 0) →
  (a - b)^2 = 841 / 36 :=
by sorry

end quadratic_roots_difference_squared_l3481_348179


namespace triangle_side_calculation_l3481_348101

/-- Given a triangle ABC with angles B = 60°, C = 75°, and side a = 4,
    prove that side b = 2√6 -/
theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) : 
  B = π / 3 →  -- 60° in radians
  C = 5 * π / 12 →  -- 75° in radians
  a = 4 →
  A + B + C = π →  -- Sum of angles in a triangle
  a / Real.sin A = b / Real.sin B →  -- Law of Sines
  b = 2 * Real.sqrt 6 := by
sorry

end triangle_side_calculation_l3481_348101


namespace employee_payment_percentage_l3481_348109

theorem employee_payment_percentage (total payment_B : ℝ) 
  (h1 : total = 570)
  (h2 : payment_B = 228) : 
  (total - payment_B) / payment_B * 100 = 150 := by
  sorry

end employee_payment_percentage_l3481_348109


namespace complement_of_A_union_B_l3481_348153

def U : Set ℕ := {1, 2, 3, 4, 5}

def A : Set ℕ := {x ∈ U | x^2 - 3*x + 2 = 0}

def B : Set ℕ := {x ∈ U | ∃ α ∈ A, x = 2*α}

theorem complement_of_A_union_B (h : Set ℕ) : 
  h = U \ (A ∪ B) → h = {3, 5} := by sorry

end complement_of_A_union_B_l3481_348153


namespace carbon_atoms_in_compound_l3481_348166

/-- Represents the number of atoms of each element in a compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (carbonWeight oxygenWeight hydrogenWeight : ℕ) : ℕ :=
  c.carbon * carbonWeight + c.hydrogen * hydrogenWeight + c.oxygen * oxygenWeight

/-- Theorem: A compound with 4 Hydrogen and 2 Oxygen atoms, and molecular weight 60,
    must have 2 Carbon atoms -/
theorem carbon_atoms_in_compound (c : Compound) 
    (h1 : c.hydrogen = 4)
    (h2 : c.oxygen = 2)
    (h3 : molecularWeight c 12 16 1 = 60) :
    c.carbon = 2 := by
  sorry

end carbon_atoms_in_compound_l3481_348166


namespace log_inequality_implies_a_range_l3481_348183

theorem log_inequality_implies_a_range (a : ℝ) : 
  (∃ (loga : ℝ → ℝ → ℝ), loga a 3 < 1) → (a > 3 ∨ (0 < a ∧ a < 1)) :=
by sorry

end log_inequality_implies_a_range_l3481_348183


namespace cake_eaten_after_four_trips_l3481_348187

/-- The fraction of cake eaten after n trips, where on each trip 1/3 of the remaining cake is eaten -/
def cakeEaten (n : ℕ) : ℚ :=
  1 - (2/3)^n

/-- The theorem stating that after 4 trips, 40/81 of the cake is eaten -/
theorem cake_eaten_after_four_trips :
  cakeEaten 4 = 40/81 := by sorry

end cake_eaten_after_four_trips_l3481_348187


namespace inscribed_sphere_volume_l3481_348163

/-- The volume of a sphere inscribed in a cube with edge length 4 is 32π/3 -/
theorem inscribed_sphere_volume (cube_edge : ℝ) (sphere_volume : ℝ) :
  cube_edge = 4 →
  sphere_volume = (4 / 3) * π * (cube_edge / 2)^3 →
  sphere_volume = (32 * π) / 3 := by
  sorry

end inscribed_sphere_volume_l3481_348163


namespace largest_multiple_six_negation_greater_than_neg_150_l3481_348107

theorem largest_multiple_six_negation_greater_than_neg_150 :
  ∀ n : ℤ, (∃ k : ℤ, n = 6 * k) → -n > -150 → n ≤ 144 :=
by
  sorry

end largest_multiple_six_negation_greater_than_neg_150_l3481_348107


namespace marshmallow_challenge_l3481_348161

/-- The marshmallow challenge problem -/
theorem marshmallow_challenge (haley michael brandon sofia : ℕ) : 
  haley = 8 →
  michael = 3 * haley →
  brandon = michael / 2 →
  sofia = 2 * (haley + brandon) →
  haley + michael + brandon + sofia = 84 := by
  sorry

end marshmallow_challenge_l3481_348161


namespace tax_rate_on_remaining_income_l3481_348121

def total_earnings : ℝ := 100000
def deductions : ℝ := 30000
def first_bracket_limit : ℝ := 20000
def first_bracket_rate : ℝ := 0.1
def total_tax : ℝ := 12000

def taxable_income : ℝ := total_earnings - deductions

def tax_on_first_bracket : ℝ := first_bracket_limit * first_bracket_rate

def remaining_taxable_income : ℝ := taxable_income - first_bracket_limit

theorem tax_rate_on_remaining_income : 
  (total_tax - tax_on_first_bracket) / remaining_taxable_income = 0.2 := by sorry

end tax_rate_on_remaining_income_l3481_348121


namespace sum_coordinates_of_D_l3481_348184

/-- Given a point M that is the midpoint of line segment CD, 
    prove that the sum of coordinates of D is 12 -/
theorem sum_coordinates_of_D (M C D : ℝ × ℝ) : 
  M = (2, 5) → 
  C = (1/2, 3/2) → 
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 12 := by
sorry

end sum_coordinates_of_D_l3481_348184


namespace matthew_egg_rolls_l3481_348106

/-- The number of egg rolls eaten by each person -/
structure EggRolls where
  kimberly : ℕ
  alvin : ℕ
  patrick : ℕ
  matthew : ℕ

/-- The conditions of the egg roll problem -/
def EggRollConditions (e : EggRolls) : Prop :=
  e.kimberly = 5 ∧
  e.alvin = 2 * e.kimberly - 1 ∧
  e.patrick = e.alvin / 2 ∧
  e.matthew = 2 * e.patrick

theorem matthew_egg_rolls (e : EggRolls) (h : EggRollConditions e) : e.matthew = 8 := by
  sorry

#check matthew_egg_rolls

end matthew_egg_rolls_l3481_348106


namespace third_bottle_volume_is_250ml_l3481_348197

/-- Represents the volume of milk in a bottle -/
structure MilkBottle where
  volume : ℝ
  unit : String

/-- Converts liters to milliliters -/
def litersToMilliliters (liters : ℝ) : ℝ := liters * 1000

/-- Calculates the volume of the third milk bottle -/
def thirdBottleVolume (bottle1 : MilkBottle) (bottle2 : MilkBottle) (totalVolume : ℝ) : ℝ :=
  litersToMilliliters totalVolume - (litersToMilliliters bottle1.volume + bottle2.volume)

/-- Theorem: The third milk bottle contains 250 milliliters -/
theorem third_bottle_volume_is_250ml 
  (bottle1 : MilkBottle) 
  (bottle2 : MilkBottle) 
  (totalVolume : ℝ) :
  bottle1.volume = 2 ∧ 
  bottle1.unit = "liters" ∧
  bottle2.volume = 750 ∧ 
  bottle2.unit = "milliliters" ∧
  totalVolume = 3 →
  thirdBottleVolume bottle1 bottle2 totalVolume = 250 := by
  sorry

end third_bottle_volume_is_250ml_l3481_348197


namespace jenga_players_l3481_348158

/-- The number of players in a Jenga game -/
def num_players : ℕ := 5

/-- The initial number of blocks in the Jenga tower -/
def initial_blocks : ℕ := 54

/-- The number of full rounds played -/
def full_rounds : ℕ := 5

/-- The number of blocks remaining after 5 full rounds and one additional move -/
def remaining_blocks : ℕ := 28

/-- The number of blocks removed in the 6th round before the tower falls -/
def extra_blocks_removed : ℕ := 1

theorem jenga_players :
  initial_blocks - remaining_blocks = full_rounds * num_players + extra_blocks_removed :=
sorry

end jenga_players_l3481_348158


namespace gcd_linear_combination_l3481_348185

theorem gcd_linear_combination (a b : ℤ) (h : Nat.gcd a.natAbs b.natAbs = 1) :
  Nat.gcd (11 * a + 2 * b).natAbs (18 * a + 5 * b).natAbs = 1 := by
  sorry

end gcd_linear_combination_l3481_348185


namespace odd_and_div_by_5_probability_l3481_348100

/-- A set of digits to form a four-digit number -/
def digits : Finset Nat := {8, 5, 9, 7}

/-- Predicate for a number being odd and divisible by 5 -/
def is_odd_and_div_by_5 (n : Nat) : Prop :=
  n % 2 = 1 ∧ n % 5 = 0

/-- The total number of possible four-digit numbers -/
def total_permutations : Nat := Nat.factorial 4

/-- The number of valid permutations (odd and divisible by 5) -/
def valid_permutations : Nat := Nat.factorial 3

/-- The probability of forming a number that is odd and divisible by 5 -/
def probability : Rat := valid_permutations / total_permutations

theorem odd_and_div_by_5_probability :
  probability = 1 / 4 := by sorry

end odd_and_div_by_5_probability_l3481_348100


namespace expected_socks_taken_l3481_348119

/-- Represents a collection of socks -/
structure SockCollection where
  pairs : ℕ  -- number of pairs of socks
  nonIdentical : Bool  -- whether all pairs are non-identical

/-- Represents the process of selecting socks -/
def selectSocks (sc : SockCollection) : ℕ → ℕ
  | 0 => 0
  | n + 1 => n + 1  -- simplified representation of sock selection

/-- Expected number of socks taken until a pair is found -/
def expectedSocksTaken (sc : SockCollection) : ℝ :=
  2 * sc.pairs

/-- Theorem stating the expected number of socks taken is 2p -/
theorem expected_socks_taken (sc : SockCollection) (h1 : sc.nonIdentical = true) :
  expectedSocksTaken sc = 2 * sc.pairs := by
  sorry

#check expected_socks_taken

end expected_socks_taken_l3481_348119


namespace seths_ice_cream_purchase_l3481_348192

/-- Seth's ice cream purchase problem -/
theorem seths_ice_cream_purchase
  (ice_cream_cost : ℕ → ℕ)
  (yogurt_cost : ℕ)
  (yogurt_quantity : ℕ)
  (cost_difference : ℕ)
  (h1 : yogurt_quantity = 2)
  (h2 : ∀ n, ice_cream_cost n = 6 * n)
  (h3 : yogurt_cost = 1)
  (h4 : ∃ x : ℕ, ice_cream_cost x = yogurt_quantity * yogurt_cost + cost_difference)
  (h5 : cost_difference = 118) :
  ∃ x : ℕ, ice_cream_cost x = yogurt_quantity * yogurt_cost + cost_difference ∧ x = 20 := by
  sorry

end seths_ice_cream_purchase_l3481_348192


namespace square_division_reversible_l3481_348160

/-- A square of cells can be divided into equal figures -/
structure CellSquare where
  side : ℕ
  total_cells : ℕ
  total_cells_eq : total_cells = side * side

/-- A division of a cell square into equal figures -/
structure SquareDivision (square : CellSquare) where
  num_figures : ℕ
  cells_per_figure : ℕ
  division_valid : square.total_cells = num_figures * cells_per_figure

theorem square_division_reversible (square : CellSquare) 
  (div1 : SquareDivision square) :
  ∃ (div2 : SquareDivision square), 
    div2.num_figures = div1.cells_per_figure ∧ 
    div2.cells_per_figure = div1.num_figures :=
sorry

end square_division_reversible_l3481_348160


namespace two_digit_sum_l3481_348193

/-- Given two single-digit natural numbers A and B, if 6A + B2 = 77, then B = 1 -/
theorem two_digit_sum (A B : ℕ) : 
  A < 10 → B < 10 → (60 + A) + (10 * B + 2) = 77 → B = 1 := by
  sorry

end two_digit_sum_l3481_348193


namespace stone_arrangement_exists_l3481_348110

theorem stone_arrangement_exists (P : ℕ) (h : P = 23) : ∃ (F : ℕ → ℤ), 
  F 0 = 0 ∧ 
  F 1 = 1 ∧ 
  (∀ i : ℕ, i ≥ 2 → F i = 3 * F (i - 1) - F (i - 2)) ∧
  F 12 % P = 0 :=
by sorry

end stone_arrangement_exists_l3481_348110


namespace correct_pairing_l3481_348159

structure Couple where
  wife : String
  husband : String
  wife_bottles : Nat
  husband_bottles : Nat

def total_bottles : Nat := 44

def couples : List Couple := [
  ⟨"Anna", "Smith", 2, 8⟩,
  ⟨"Betty", "White", 3, 9⟩,
  ⟨"Carol", "Green", 4, 8⟩,
  ⟨"Dorothy", "Brown", 5, 5⟩
]

theorem correct_pairing : 
  (couples.map (λ c => c.wife_bottles + c.husband_bottles)).sum = total_bottles ∧
  (∃ c ∈ couples, c.husband = "Brown" ∧ c.wife_bottles = c.husband_bottles) ∧
  (∃ c ∈ couples, c.husband = "Green" ∧ c.husband_bottles = 2 * c.wife_bottles) ∧
  (∃ c ∈ couples, c.husband = "White" ∧ c.husband_bottles = 3 * c.wife_bottles) ∧
  (∃ c ∈ couples, c.husband = "Smith" ∧ c.husband_bottles = 4 * c.wife_bottles) :=
by sorry

end correct_pairing_l3481_348159


namespace undefined_values_sum_l3481_348188

theorem undefined_values_sum (f : ℝ → ℝ) (h : f = λ x => 5*x / (3*x^2 - 9*x + 6)) : 
  ∃ C D : ℝ, (3*C^2 - 9*C + 6 = 0) ∧ (3*D^2 - 9*D + 6 = 0) ∧ (C + D = 3) := by
  sorry

end undefined_values_sum_l3481_348188


namespace sin_240_degrees_l3481_348117

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_degrees_l3481_348117


namespace solution_set_of_inequality_l3481_348143

theorem solution_set_of_inequality (x : ℝ) :
  (x + 3) / (2 * x - 1) < 0 ↔ -3 < x ∧ x < 1/2 :=
by sorry

end solution_set_of_inequality_l3481_348143


namespace lucy_snowballs_l3481_348196

theorem lucy_snowballs (charlie_snowballs : ℕ) (difference : ℕ) (lucy_snowballs : ℕ) : 
  charlie_snowballs = 50 → 
  difference = 31 → 
  charlie_snowballs = lucy_snowballs + difference → 
  lucy_snowballs = 19 := by
sorry

end lucy_snowballs_l3481_348196


namespace intersection_M_and_naturals_l3481_348154

def M : Set ℝ := {x | (x + 2) / (x - 1) ≤ 0}

theorem intersection_M_and_naturals :
  M ∩ Set.range (Nat.cast : ℕ → ℝ) = {0} := by sorry

end intersection_M_and_naturals_l3481_348154


namespace susans_books_l3481_348149

/-- Proves that Susan has 600 books given the conditions of the problem -/
theorem susans_books (susan_books : ℕ) (lidia_books : ℕ) : 
  lidia_books = 4 * susan_books → -- Lidia's collection is four times bigger than Susan's
  susan_books + lidia_books = 3000 → -- Total books is 3000
  susan_books = 600 := by
sorry

end susans_books_l3481_348149


namespace min_xy_equals_nine_l3481_348171

theorem min_xy_equals_nine (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_eq : 1 / (x + 1) + 1 / (y + 1) = 1 / 2) :
  ∀ z, z = x * y → z ≥ 9 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ 1 / (a + 1) + 1 / (b + 1) = 1 / 2 ∧ a * b = 9 :=
by sorry

end min_xy_equals_nine_l3481_348171


namespace min_value_theorem_l3481_348169

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 2/b = 4) :
  9/4 ≤ a + 2*b ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 1/a₀ + 2/b₀ = 4 ∧ a₀ + 2*b₀ = 9/4 :=
by sorry

#check min_value_theorem

end min_value_theorem_l3481_348169


namespace evaluate_expression_l3481_348152

theorem evaluate_expression : 5 - (-3)^(2 - (1 - 3)) = -76 := by
  sorry

end evaluate_expression_l3481_348152


namespace move_up_coordinates_l3481_348135

/-- Moving a point up in a 2D coordinate system -/
def move_up (p : ℝ × ℝ) (n : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + n)

/-- The theorem states that moving a point up by n units results in the expected coordinates -/
theorem move_up_coordinates (x y n : ℝ) :
  move_up (x, y) n = (x, y + n) := by
  sorry

end move_up_coordinates_l3481_348135


namespace smallest_angle_of_triangle_l3481_348195

theorem smallest_angle_of_triangle (y : ℝ) (h : y + 40 + 70 = 180) :
  min (min 40 70) y = 40 := by sorry

end smallest_angle_of_triangle_l3481_348195


namespace equal_squares_on_8x7_board_l3481_348199

/-- Represents a rectangular board with alternating light and dark squares. -/
structure AlternatingBoard :=
  (rows : Nat)
  (columns : Nat)

/-- Counts the number of dark squares on the board. -/
def count_dark_squares (board : AlternatingBoard) : Nat :=
  (board.rows / 2) * ((board.columns + 1) / 2) + 
  ((board.rows + 1) / 2) * (board.columns / 2)

/-- Counts the number of light squares on the board. -/
def count_light_squares (board : AlternatingBoard) : Nat :=
  ((board.rows + 1) / 2) * ((board.columns + 1) / 2) + 
  (board.rows / 2) * (board.columns / 2)

/-- Theorem stating that for an 8x7 alternating board, the number of dark squares equals the number of light squares. -/
theorem equal_squares_on_8x7_board :
  let board : AlternatingBoard := ⟨8, 7⟩
  count_dark_squares board = count_light_squares board := by
  sorry

#eval count_dark_squares ⟨8, 7⟩
#eval count_light_squares ⟨8, 7⟩

end equal_squares_on_8x7_board_l3481_348199


namespace golden_ratio_bounds_l3481_348122

theorem golden_ratio_bounds : 
  let φ := (Real.sqrt 5 - 1) / 2
  0.6 < φ ∧ φ < 0.7 := by sorry

end golden_ratio_bounds_l3481_348122


namespace complement_intersection_theorem_l3481_348164

open Set

-- Define the universal set U
def U : Set Int := {-1, 0, 1, 2, 3}

-- Define set A
def A : Set Int := {-1, 0}

-- Define set B
def B : Set Int := {0, 1, 2}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ A) ∩ B = {1, 2} := by sorry

end complement_intersection_theorem_l3481_348164


namespace g_composition_of_three_l3481_348136

def g (x : ℝ) : ℝ := 3 * x - 5

theorem g_composition_of_three : g (g (g 3)) = 16 := by
  sorry

end g_composition_of_three_l3481_348136


namespace expand_and_simplify_fraction_l3481_348157

theorem expand_and_simplify_fraction (y : ℝ) (h : y ≠ 0) :
  (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 := by
  sorry

end expand_and_simplify_fraction_l3481_348157


namespace opposite_faces_l3481_348170

/-- Represents the six faces of a cube -/
inductive Face : Type
  | xiao : Face  -- 小
  | xue  : Face  -- 学
  | xi   : Face  -- 希
  | wang : Face  -- 望
  | bei  : Face  -- 杯
  | sai  : Face  -- 赛

/-- Defines the adjacency relationship between faces -/
def adjacent : Face → Face → Prop :=
  sorry

/-- Defines the opposite relationship between faces -/
def opposite : Face → Face → Prop :=
  sorry

/-- The cube configuration satisfies the given conditions -/
axiom cube_config :
  adjacent Face.xue Face.xiao ∧
  adjacent Face.xue Face.xi ∧
  adjacent Face.xue Face.wang ∧
  adjacent Face.xue Face.sai

/-- Theorem stating the opposite face relationships -/
theorem opposite_faces :
  opposite Face.xi Face.sai ∧
  opposite Face.wang Face.xiao ∧
  opposite Face.bei Face.xue :=
by sorry

end opposite_faces_l3481_348170


namespace greatest_four_digit_sum_15_l3481_348189

/-- A function that returns true if a number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that returns the product of digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- The theorem stating that the sum of digits of the greatest four-digit number
    with digit product 36 is 15 -/
theorem greatest_four_digit_sum_15 :
  ∃ M : ℕ, is_four_digit M ∧ 
           digit_product M = 36 ∧ 
           (∀ n : ℕ, is_four_digit n → digit_product n = 36 → n ≤ M) ∧
           digit_sum M = 15 := by
  sorry

end greatest_four_digit_sum_15_l3481_348189


namespace positive_numbers_inequality_l3481_348133

theorem positive_numbers_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2 * Real.sqrt (a*b*c)) := by
  sorry

end positive_numbers_inequality_l3481_348133


namespace coefficient_x4_expansion_l3481_348134

def binomial_coefficient (n k : ℕ) : ℕ := sorry

theorem coefficient_x4_expansion :
  let n : ℕ := 8
  let k : ℕ := 4
  let a : ℝ := 1
  let b : ℝ := 3 * Real.sqrt 3
  binomial_coefficient n k * a^(n-k) * b^k = 51030 := by sorry

end coefficient_x4_expansion_l3481_348134


namespace stratified_sampling_theorem_university_sample_sizes_correct_l3481_348124

/-- Represents a stratum in a population -/
structure Stratum where
  size : ℕ

/-- Represents a population with stratified sampling -/
structure StratifiedPopulation where
  total : ℕ
  strata : List Stratum
  sample_size : ℕ

/-- Calculates the number of samples for a given stratum -/
def sample_size_for_stratum (pop : StratifiedPopulation) (stratum : Stratum) : ℕ :=
  (pop.sample_size * stratum.size) / pop.total

/-- Theorem: The sum of samples from all strata equals the total sample size -/
theorem stratified_sampling_theorem (pop : StratifiedPopulation) 
  (h : pop.total = (pop.strata.map Stratum.size).sum) :
  (pop.strata.map (sample_size_for_stratum pop)).sum = pop.sample_size := by
  sorry

/-- The university population -/
def university_pop : StratifiedPopulation :=
  { total := 5600
  , strata := [⟨1300⟩, ⟨3000⟩, ⟨1300⟩]
  , sample_size := 280 }

/-- Theorem: The calculated sample sizes for the university population are correct -/
theorem university_sample_sizes_correct :
  (university_pop.strata.map (sample_size_for_stratum university_pop)) = [65, 150, 65] := by
  sorry

end stratified_sampling_theorem_university_sample_sizes_correct_l3481_348124


namespace sum_remainder_l3481_348130

theorem sum_remainder (S : ℤ) : S = (2 * 3^500) / 3 → S % 1000 = 2 := by
  sorry

end sum_remainder_l3481_348130


namespace printing_presses_l3481_348137

theorem printing_presses (time1 time2 : ℝ) (newspapers1 newspapers2 : ℕ) (presses2 : ℕ) :
  time1 = 6 →
  time2 = 9 →
  newspapers1 = 8000 →
  newspapers2 = 6000 →
  presses2 = 2 →
  ∃ (presses1 : ℕ), 
    (presses1 : ℝ) * (newspapers2 : ℝ) / (time2 * presses2) = newspapers1 / time1 ∧
    presses1 = 4 :=
by sorry


end printing_presses_l3481_348137


namespace triangle_area_l3481_348151

/-- Given a triangle ABC with side length a = 6, angle B = 30°, and angle C = 120°,
    prove that its area is 9√3. -/
theorem triangle_area (a b c : ℝ) (A B C : Real) : 
  a = 6 → B = 30 * Real.pi / 180 → C = 120 * Real.pi / 180 →
  (1/2) * a * b * Real.sin C = 9 * Real.sqrt 3 := by
  sorry

end triangle_area_l3481_348151


namespace concert_attendance_theorem_l3481_348102

/-- Represents the relationship between number of attendees and ticket price -/
structure ConcertAttendance where
  n : ℕ  -- number of attendees
  t : ℕ  -- ticket price in dollars
  k : ℕ  -- constant of proportionality
  h : n * t = k  -- inverse proportionality relationship

/-- Given initial conditions and final ticket price, calculates the final number of attendees -/
def calculate_attendance (initial : ConcertAttendance) (final_price : ℕ) : ℕ :=
  initial.k / final_price

theorem concert_attendance_theorem (initial : ConcertAttendance) 
    (h1 : initial.n = 300) 
    (h2 : initial.t = 50) 
    (h3 : calculate_attendance initial 75 = 200) : 
  calculate_attendance initial 75 = 200 := by
  sorry

#check concert_attendance_theorem

end concert_attendance_theorem_l3481_348102


namespace valid_outfits_count_l3481_348113

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of pants available -/
def num_pants : ℕ := 5

/-- The number of hats available -/
def num_hats : ℕ := 8

/-- The number of colors shared by shirts, pants, and hats -/
def num_shared_colors : ℕ := 5

/-- The number of additional colors for shirts and hats -/
def num_additional_colors : ℕ := 2

/-- The total number of outfit combinations -/
def total_combinations : ℕ := num_shirts * num_pants * num_hats

/-- The number of combinations where shirt and hat have the same color -/
def same_color_combinations : ℕ := num_shared_colors * num_pants

/-- The number of valid outfit combinations -/
def valid_combinations : ℕ := total_combinations - same_color_combinations

theorem valid_outfits_count :
  valid_combinations = 295 := by sorry

end valid_outfits_count_l3481_348113


namespace arc_length_of_sector_l3481_348123

theorem arc_length_of_sector (θ : Real) (r : Real) (L : Real) : 
  θ = 120 → r = 3/2 → L = θ / 360 * (2 * Real.pi * r) → L = Real.pi := by
  sorry

end arc_length_of_sector_l3481_348123


namespace remainder_of_expression_l3481_348148

theorem remainder_of_expression (p t : ℕ) (hp : p > t) (ht : t > 1) :
  (92^p * 5^(p + t) + 11^t * 6^(p*t)) % 10 = 1 := by
  sorry

end remainder_of_expression_l3481_348148


namespace correct_diagnosis_l3481_348162

structure Doctor where
  name : String
  statements : List String

structure Patient where
  diagnosis : List String

def homeopath : Doctor :=
  { name := "Homeopath"
  , statements := 
    [ "The patient has a strong astigmatism"
    , "The patient smokes too much"
    , "The patient has a tropical fever"
    ]
  }

def therapist : Doctor :=
  { name := "Therapist"
  , statements := 
    [ "The patient has a strong astigmatism"
    , "The patient doesn't eat well"
    , "The patient suffers from high blood pressure"
    ]
  }

def ophthalmologist : Doctor :=
  { name := "Ophthalmologist"
  , statements := 
    [ "The patient has a strong astigmatism"
    , "The patient is near-sighted"
    , "The patient has no signs of retinal detachment"
    ]
  }

def correct_statements : List (Doctor × Nat) :=
  [ (homeopath, 1)
  , (therapist, 0)
  , (ophthalmologist, 0)
  ]

theorem correct_diagnosis (doctors : List Doctor) 
  (correct : List (Doctor × Nat)) : 
  ∃ (p : Patient), 
    p.diagnosis = 
      [ "I have a strong astigmatism"
      , "I smoke too much"
      , "I am not eating well enough!"
      , "I do not have tropical fever"
      ] :=
  sorry

end correct_diagnosis_l3481_348162


namespace sixPeopleArrangements_l3481_348173

/-- The number of ways to arrange n distinct objects. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange six people in a row with Person A and Person B adjacent. -/
def adjacentArrangements : ℕ :=
  arrangements 2 * arrangements 5

/-- The number of ways to arrange six people in a row with Person A and Person B not adjacent. -/
def nonAdjacentArrangements : ℕ :=
  arrangements 4 * arrangements 2

/-- The number of ways to arrange six people in a row with exactly two people between Person A and Person B. -/
def twoPersonsBetweenArrangements : ℕ :=
  arrangements 2 * arrangements 2 * arrangements 3

/-- The number of ways to arrange six people in a row with Person A not at the left end and Person B not at the right end. -/
def notAtEndsArrangements : ℕ :=
  arrangements 6 - 2 * arrangements 5 + arrangements 4

theorem sixPeopleArrangements :
  adjacentArrangements = 240 ∧
  nonAdjacentArrangements = 480 ∧
  twoPersonsBetweenArrangements = 144 ∧
  notAtEndsArrangements = 504 := by
  sorry

end sixPeopleArrangements_l3481_348173


namespace solve_for_a_l3481_348132

theorem solve_for_a (x y a : ℝ) : 
  x + y = 1 → 
  2 * x + y = 0 → 
  a * x - 3 * y = 0 → 
  a = -6 := by
sorry

end solve_for_a_l3481_348132


namespace fair_coin_probability_difference_l3481_348128

def probability_n_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1/2)^n

theorem fair_coin_probability_difference : 
  (probability_n_heads 4 3) - (probability_n_heads 4 4) = 7/16 := by
  sorry

end fair_coin_probability_difference_l3481_348128


namespace problem_hexagon_area_l3481_348155

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- A hexagon defined by six points -/
structure Hexagon where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point
  p5 : Point
  p6 : Point

/-- The area of a hexagon -/
def hexagonArea (h : Hexagon) : ℝ := sorry

/-- The specific hexagon in the problem -/
def problemHexagon : Hexagon := {
  p1 := { x := 0, y := 0 },
  p2 := { x := 2, y := 4 },
  p3 := { x := 6, y := 4 },
  p4 := { x := 8, y := 0 },
  p5 := { x := 6, y := -4 },
  p6 := { x := 2, y := -4 }
}

/-- Theorem stating that the area of the problem hexagon is 16 square units -/
theorem problem_hexagon_area : hexagonArea problemHexagon = 16 := by sorry

end problem_hexagon_area_l3481_348155


namespace shoe_selection_theorem_l3481_348144

/-- The number of ways to select 4 shoes from 10 pairs such that 2 form a pair and 2 do not -/
def shoeSelectionWays (totalPairs : Nat) : Nat :=
  if totalPairs = 10 then
    Nat.choose totalPairs 1 * Nat.choose (totalPairs - 1) 2 * 4
  else
    0

theorem shoe_selection_theorem :
  shoeSelectionWays 10 = 1440 := by
  sorry

end shoe_selection_theorem_l3481_348144


namespace train_length_l3481_348168

/-- The length of a train given its speed, the speed of a person it passes, and the time it takes to pass them. -/
theorem train_length (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) :
  train_speed = 63 →
  person_speed = 3 →
  passing_time = 53.99568034557235 →
  ∃ (length : ℝ), abs (length - 899.93) < 0.01 ∧
  length = (train_speed - person_speed) * (5 / 18) * passing_time :=
sorry

end train_length_l3481_348168


namespace circle_properties_l3481_348138

/-- Given a circle with circumference 31.4 decimeters, prove its diameter, radius, and area -/
theorem circle_properties (C : Real) (h : C = 31.4) :
  ∃ (d r A : Real),
    d = 10 ∧ 
    r = 5 ∧ 
    A = 78.5 ∧
    C = 2 * Real.pi * r ∧
    d = 2 * r ∧
    A = Real.pi * r^2 := by
  sorry

end circle_properties_l3481_348138


namespace sequences_sum_and_diff_total_l3481_348150

def arithmetic_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def sequence1_sum : ℤ := arithmetic_sum 4 10 6
def sequence2_sum : ℤ := arithmetic_sum 12 10 6

theorem sequences_sum_and_diff_total : 
  (sequence1_sum + sequence2_sum) + (sequence2_sum - sequence1_sum) = 444 := by
  sorry

end sequences_sum_and_diff_total_l3481_348150


namespace number_problem_l3481_348142

theorem number_problem (N : ℝ) :
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 10 →
  (40/100 : ℝ) * N = 120 := by
sorry

end number_problem_l3481_348142


namespace sum_of_coordinates_B_l3481_348114

/-- Given point A at (0, 0), point B on the line y = 6, and the slope of segment AB is 3/4,
    the sum of the x- and y-coordinates of point B is 14. -/
theorem sum_of_coordinates_B (B : ℝ × ℝ) : 
  B.2 = 6 ∧ (B.2 - 0) / (B.1 - 0) = 3/4 → B.1 + B.2 = 14 := by sorry

end sum_of_coordinates_B_l3481_348114


namespace frank_lamp_purchase_l3481_348139

theorem frank_lamp_purchase (cheapest_lamp : ℕ) (frank_money : ℕ) :
  cheapest_lamp = 20 →
  frank_money = 90 →
  frank_money - (3 * cheapest_lamp) = 30 :=
by
  sorry

end frank_lamp_purchase_l3481_348139


namespace first_year_after_2010_with_digit_sum_4_l3481_348116

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isFirstYearAfter2010WithDigitSum4 (year : Nat) : Prop :=
  year > 2010 ∧ 
  sumOfDigits year = 4 ∧
  ∀ y, 2010 < y ∧ y < year → sumOfDigits y ≠ 4

theorem first_year_after_2010_with_digit_sum_4 :
  isFirstYearAfter2010WithDigitSum4 2011 := by
  sorry

end first_year_after_2010_with_digit_sum_4_l3481_348116


namespace room_width_calculation_l3481_348111

/-- Given a rectangular room with specified length, total paving cost, and paving rate per square meter, 
    prove that the width of the room is as calculated. -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) (width : ℝ) : 
  length = 7 →
  total_cost = 29925 →
  rate_per_sqm = 900 →
  width = total_cost / rate_per_sqm / length →
  width = 4.75 := by
  sorry

end room_width_calculation_l3481_348111


namespace angle_sum_tangent_l3481_348127

theorem angle_sum_tangent (a β : Real) (ha : 0 < a ∧ a < π/2) (hβ : 0 < β ∧ β < π/2)
  (tan_a : Real.tan a = 2) (tan_β : Real.tan β = 3) :
  a + β = 3 * π / 4 := by
sorry

end angle_sum_tangent_l3481_348127


namespace group_size_proof_l3481_348105

theorem group_size_proof (n : ℕ) (W : ℝ) : 
  (W + 25) / n - W / n = 2.5 → n = 10 := by
  sorry

end group_size_proof_l3481_348105


namespace sophias_book_length_l3481_348118

theorem sophias_book_length (total_pages : ℕ) : 
  (2 : ℚ) / 3 * total_pages = (1 : ℚ) / 3 * total_pages + 90 → 
  total_pages = 270 := by
sorry

end sophias_book_length_l3481_348118


namespace T_equiv_horizontal_lines_l3481_348140

/-- The set of points R forming a right triangle PQR with area 4, where P(2,0) and Q(-2,0) -/
def T : Set (ℝ × ℝ) :=
  {R | ∃ (x y : ℝ), R = (x, y) ∧ 
       ((x - 2)^2 + y^2) * ((x + 2)^2 + y^2) = 16 * (x^2 + y^2) ∧
       (abs ((x - 2) * y - (x + 2) * y)) = 8}

/-- The set of points with y-coordinate equal to 2 or -2 -/
def horizontal_lines : Set (ℝ × ℝ) :=
  {R | ∃ (x y : ℝ), R = (x, y) ∧ (y = 2 ∨ y = -2)}

theorem T_equiv_horizontal_lines : T = horizontal_lines := by
  sorry

end T_equiv_horizontal_lines_l3481_348140


namespace expected_games_at_negative_one_l3481_348112

/-- The expected number of games in a best-of-five series -/
def f (x : ℝ) : ℝ :=
  3 * (x^3 + (1-x)^3) + 
  4 * (3*x^3*(1-x) + 3*(1-x)^3*x) + 
  5 * (6*x^2*(1-x)^2)

/-- Theorem: The expected number of games when x = -1 is 21 -/
theorem expected_games_at_negative_one : f (-1) = 21 := by
  sorry

end expected_games_at_negative_one_l3481_348112


namespace exists_m_in_range_l3481_348191

def sequence_x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (sequence_x n ^ 2 + 7 * sequence_x n + 12) / (sequence_x n + 8)

theorem exists_m_in_range :
  ∃ m : ℕ, 81 ≤ m ∧ m ≤ 242 ∧
  sequence_x m ≤ 5 + 1 / 2^15 ∧
  ∀ k : ℕ, 0 < k ∧ k < m → sequence_x k > 5 + 1 / 2^15 := by
  sorry

end exists_m_in_range_l3481_348191


namespace intersection_M_N_l3481_348125

open Set

def M : Set ℝ := {x : ℝ | 3 * x - x^2 > 0}
def N : Set ℝ := {x : ℝ | x^2 - 4 * x + 3 > 0}

theorem intersection_M_N : M ∩ N = Ioo 0 1 := by sorry

end intersection_M_N_l3481_348125


namespace polynomial_identity_l3481_348108

theorem polynomial_identity (x : ℝ) :
  (5 * x^3 - 32 * x^2 + 75 * x - 71 = 
   5 * (x - 2)^3 + (-2) * (x - 2)^2 + 7 * (x - 2) + (-9)) ∧
  (∀ (a b c d : ℝ), 
    (∀ x : ℝ, 5 * x^3 - 32 * x^2 + 75 * x - 71 = 
      a * (x - 2)^3 + b * (x - 2)^2 + c * (x - 2) + d) →
    a = 5 ∧ b = -2 ∧ c = 7 ∧ d = -9) := by
sorry

end polynomial_identity_l3481_348108


namespace rectangular_field_area_l3481_348186

theorem rectangular_field_area (width length : ℝ) : 
  width > 0 →
  length > 0 →
  width = (1/3) * length →
  2 * (width + length) = 80 →
  width * length = 300 := by
sorry

end rectangular_field_area_l3481_348186


namespace figure_100_squares_l3481_348194

/-- The number of nonoverlapping unit squares in figure n -/
def f (n : ℕ) : ℕ := 2 * n^3 + n^2 + 2 * n + 1

theorem figure_100_squares :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 25 ∧ f 3 = 63 → f 100 = 2010201 := by
  sorry

end figure_100_squares_l3481_348194


namespace quadratic_roots_theorem_l3481_348198

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop := x^2 - 2*x + m = 0

-- Theorem statement
theorem quadratic_roots_theorem (m : ℝ) (h : m < 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m ∧ quadratic_equation x₂ m) ∧
  (quadratic_equation (-1) m → m = -3 ∧ quadratic_equation 3 m) :=
sorry

end quadratic_roots_theorem_l3481_348198


namespace longest_side_of_triangle_l3481_348120

-- Define the triangle
def triangle (x : ℝ) : Fin 3 → ℝ
| 0 => 8
| 1 => 2*x + 5
| 2 => 3*x - 1
| _ => 0  -- This case is never reached due to Fin 3

-- State the theorem
theorem longest_side_of_triangle :
  ∃ x : ℝ, 
    (triangle x 0 + triangle x 1 + triangle x 2 = 45) ∧ 
    (∀ i : Fin 3, triangle x i ≤ 18.8) ∧
    (∃ i : Fin 3, triangle x i = 18.8) :=
by
  sorry

end longest_side_of_triangle_l3481_348120


namespace specific_card_draw_probability_l3481_348167

theorem specific_card_draw_probability : 
  let deck_size : ℕ := 52
  let prob_specific_card : ℚ := 1 / deck_size
  let prob_both_specific_cards : ℚ := prob_specific_card * prob_specific_card
  prob_both_specific_cards = 1 / 2704 := by
  sorry

end specific_card_draw_probability_l3481_348167


namespace negation_of_absolute_sine_bound_l3481_348115

theorem negation_of_absolute_sine_bound :
  (¬ ∀ x : ℝ, |Real.sin x| ≤ 1) ↔ (∃ x : ℝ, |Real.sin x| > 1) := by
  sorry

end negation_of_absolute_sine_bound_l3481_348115


namespace inequality_proof_l3481_348103

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) : 
  (a * b ≤ 1/8) ∧ (Real.sqrt a + Real.sqrt b ≤ Real.sqrt 6 / 2) := by
  sorry

end inequality_proof_l3481_348103


namespace reciprocal_problem_l3481_348182

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 4) : 150 * (1 / x) = 300 := by
  sorry

end reciprocal_problem_l3481_348182


namespace tetrahedron_volume_prove_tetrahedron_volume_l3481_348172

/-- Tetrahedron ABCD with given properties -/
structure Tetrahedron where
  /-- Length of edge AB in cm -/
  ab_length : ℝ
  /-- Area of face ABC in cm² -/
  abc_area : ℝ
  /-- Area of face ABD in cm² -/
  abd_area : ℝ
  /-- Angle between faces ABC and ABD in radians -/
  face_angle : ℝ
  /-- Conditions on the tetrahedron -/
  ab_length_eq : ab_length = 3
  abc_area_eq : abc_area = 15
  abd_area_eq : abd_area = 12
  face_angle_eq : face_angle = Real.pi / 6

/-- The volume of the tetrahedron is 20 cm³ -/
theorem tetrahedron_volume (t : Tetrahedron) : ℝ :=
  20

#check tetrahedron_volume

/-- Proof of the tetrahedron volume -/
theorem prove_tetrahedron_volume (t : Tetrahedron) :
  tetrahedron_volume t = 20 := by
  sorry

end tetrahedron_volume_prove_tetrahedron_volume_l3481_348172
